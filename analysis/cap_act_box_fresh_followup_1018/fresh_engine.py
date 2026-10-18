"""Explicit one-task derivative of the unchanged registered F51 engine.

Only task selection, prefix RNG advancement, fixed cap initialization, and
metadata differ. fresh_followup.py checks step/train_task/capture AST identity.
"""
from __future__ import annotations
import hashlib, json, math, os, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F
from src.cap_act_box_2x2_1018 import (
    B, C, H, I, CAP_ARMS, ARM_ORDER, COND, LR, EXPERIMENT, RADIUS_TASK,
    row_norms, project_rows_, monitor_, cap_stats, unit_diagnostics,
    put, save_npz, sha256_file,
)

def run_fresh(cap_arm: str, seeds: list[int], n_tasks: int, device, out: Path, lr: float = LR,
        cifar: C.Cifar100 | None = None, fresh: bool = False, graph: bool = True,
        progress=None, debug: dict | None = None, observer=None,
        steps_hard: int = C.STEPS_PER_TASK, steps_easy: int = C.STEPS_PER_TASK,
        activation: str = "ReLU", schedule: dict | None = None,
        target_task: int = 29, fixed_radii=None) -> dict:
    """Original F51 kernel; one saved late task from P0 with a fixed task-1 radius."""
    assert n_tasks == 1 and target_task in (21, 23, 25, 27, 29)
    assert cap_arm in ("ref", "cap12") and not fresh and schedule is None
    assert fixed_radii is not None and len(fixed_radii) == 2
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    if cap_arm not in CAP_ARMS:
        raise SystemExit(f"unknown cap arm {cap_arm!r}; known: {','.join(ARM_ORDER)}")
    layers = CAP_ARMS[cap_arm]
    arm, cond = ("ELU" if activation == "ELU" else "R"), COND
    act = C.make_act(arm)
    R = len(seeds)
    cifar = cifar or C.Cifar100()
    X_all = cifar.inputs("train", cond, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", cond, device)
    Y_test = cifar.test_y.to(device)
    input_tensor_hash=I.tree_hash(X_all)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    # Advance the original CPU batch generators without training the prefix.
    for previous in range(1, target_task):
        for r, seed in enumerate(seeds):
            rr = torch.cat([tr_rows[q] for q in plans[r][previous - 1][1]])
            C.batch_indices(g_batch[seed], rr, steps_hard if previous % 2 else steps_easy)

    init = [q.detach() for s in seeds for q in C.init_params(arm, s, device)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    out.mkdir(parents=True,exist_ok=True)
    I.save_pt(out / "initial.pt",dict(P=P0,seeds=seeds))
    act.init_state(R, device, key=f"{arm}|{seeds}|{cond}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    hook = None if debug is None else debug.get("hook")
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]

    # ---- cap state: static tensors the captured step reads by address
    rad = [torch.full((R, P[2 * l].shape[1], 1), math.inf, device=device) for l in range(2)]
    r_saved = [q.to(device=device, dtype=torch.float32).clone() for q in fixed_radii]
    if cap_arm == "cap12":
        for dst, src in zip(rad, r_saved):
            dst.copy_(src)
    rows_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    rem_acc = torch.zeros(R, 2, dtype=torch.float64, device=device)
    viol_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    excess_acc = torch.full((R, 2), -math.inf, dtype=torch.float64, device=device)

    # ---- one training step on static tensors (the host's, plus the projection)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, C.BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)

    # Persistent graph-readable per-layer lrs, modified only at task boundaries.
    layer_lr = torch.full((3,), lr, device=device)
    u_sum = torch.zeros(R, 2, dtype=torch.float64, device=device)
    u_steps = torch.zeros((), dtype=torch.long, device=device)
    radius_trace = []
    batch_manifest = []
    cap_before_max = torch.zeros(R,2,100,dtype=torch.float64,device=device)
    cap_after_max = torch.zeros_like(cap_before_max)
    unit_hits = torch.zeros(R,2,100,dtype=torch.long,device=device)
    unit_removed = torch.zeros_like(cap_before_max)
    def persist_units(t, Xt):
        u = unit_diagnostics(P, Xt, act, activation)
        for l in range(2):
            u[f"radius_l{l+1}"] = r_saved[l].detach().cpu().numpy()
        u.update(projection_before_norm_max=cap_before_max.cpu().numpy(),
                 projection_after_norm_max=cap_after_max.cpu().numpy(),
                 projection_unit_hits=unit_hits.cpu().numpy(),
                 projection_unit_removed=unit_removed.cpu().numpy())
        save_npz(out / f"units_t{t:02d}.npz", u)
    post = getattr(act, "post_update", None)

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]            # gathers, no arithmetic
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, C.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)     # pre-update: the paper's a_j
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            before_w = [P[i].detach().clone() for i in (0,2)]
            for pi, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(layer_lr[pi // 2] * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)
            if hook is not None:
                hook("pre", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)
            for l in layers:                                     # spec 2.4: W1 first, then W2
                W = P[2 * l]
                n64b = torch.linalg.vector_norm(W.double(), dim=2, keepdim=True)
                n32 = project_rows_(W, rad[l], rows_acc[:, l], rem_acc[:, l],
                                     P[2*l+1] if cap_arm == "cap12b" else None)
                monitor_(W, rad[l], n32, n64b, viol_acc[:, l], excess_acc[:, l])
                unit_hits[:,l].add_((n32>rad[l])[...,0])
                unit_removed[:,l].add_((n32.double()-rad[l].double()).clamp(min=0)[...,0])
                cap_before_max[:,l].copy_(torch.maximum(cap_before_max[:,l],n64b[...,0]))
                cap_after_max[:,l].copy_(torch.maximum(cap_after_max[:,l],W.double().norm(dim=2)))
            for l in range(2):
                before_norm = before_w[l].double().flatten(1).norm(dim=1)
                displacement = (P[2*l].double()-before_w[l].double()).flatten(1).norm(dim=1)
                rel = torch.where(before_norm>0, displacement/before_norm, torch.nan)
                u_sum[:,l].add_(rel.square())
            u_steps.add_(1)
            if hook is not None:
                hook("post", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    mutable = (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit,
               rows_acc, rem_acc, viol_acc, excess_acc, u_sum, u_steps,
               cap_before_max, cap_after_max, unit_hits, unit_removed)
    if use_graph:
        keep = [q.detach().clone() for q in mutable]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, C.BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, C.BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip(mutable, keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    def train_task(batches, tc0: int) -> int:
        """`batches.shape[1]` steps on (R, steps, 32) global row indices.  New Adam counter."""
        tc = tc0
        acc_sum.zero_()
        u_sum.zero_(); u_steps.zero_()
        cap_before_max.zero_();cap_after_max.zero_();unit_hits.zero_();unit_removed.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        rows_acc.zero_(); rem_acc.zero_(); viol_acc.zero_(); excess_acc.fill_(-math.inf)
        for j in range(batches.shape[1]):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, C.BATCH, device)
            if cg is not None:
                cg.replay()
            else:
                step()
            if debug is not None:
                debug.setdefault("online", []).append(last_hit.cpu().clone())
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, cap_rows, diverged, tc = [], [], [], 0
    last_hard = target_task
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")
    totals = {"rows_w1": 0, "rows_w2": 0, "viol_w1": 0, "viol_w2": 0}

    for t in (target_task,):
        hard = t % 2 == 1
        steps = steps_hard if hard else steps_easy
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        batch_manifest.append(dict(task=t,batch_hash=I.tree_hash(batches),
            image_ids_hash=I.tree_hash(rows_t),label_hash=I.tree_hash([Y_all[q] for q in rows_t])))
        put(out / "stream_manifest.json",dict(seeds=seeds,initial_hash=I.tree_hash(P0),data_sha256=cifar.sha256,input_tensor_sha256=input_tensor_hash,tasks=batch_manifest))
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        if debug is not None:
            debug.setdefault("batches", []).append(batches.cpu().clone())
        with torch.no_grad():
            values = [1.,1.] if schedule is None else schedule[str(t)]
            layer_lr.copy_(torch.tensor([lr*values[0],lr*values[1],lr],device=device))
        if hook is not None:
            hook("task_start", t=t, rad=rad)
        t0 = time.time()
        tc = train_task(batches, tc)
        step_ms = 1e3 * (time.time() - t0) / steps
        if hook is not None:
            hook("task_end", t=t, P=P, m=adam_m, v=adam_v, rad=rad)

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": arm,
                                 "cap_arm": cap_arm, "cond": cond,
                                 "step": (t - 1) * C.STEPS_PER_TASK + max(int(bad_step[r]), 0)})
                rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            extra = cap_stats(P, Xt, act, r_saved)
            if observer is not None:
                observer(t, dict(P=P, m=adam_m, v=adam_v, Xt=Xt, act=act, live=live, rad=rad))
            persist_units(t, Xt)
            del Xt
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
        for r in live:
            rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
            cr = {"cap_arm": cap_arm, "seed": seeds[r], "slot": r, "task": t, "hard": int(hard),
                  "U_l1":float((u_sum[r,0]/u_steps).sqrt()), "U_l2":float((u_sum[r,1]/u_steps).sqrt()),
                  "actual_lr_l1":float(layer_lr[0]), "actual_lr_l2":float(layer_lr[1]), "actual_lr_l3":float(layer_lr[2])}
            for l in range(2):
                cr[f"rows_w{l + 1}"] = int(rows_acc[r, l])
                cr[f"rem_w{l + 1}"] = float(rem_acc[r, l])
            for l in range(2):
                cr[f"viol_w{l + 1}"] = int(viol_acc[r, l])
            for l in range(2):
                cr[f"excess_w{l + 1}"] = float(excess_acc[r, l])
            for k in ("ratio_max_l1", "ratio_med_l1", "ratio_max_l2", "ratio_med_l2",
                      "mu2_norm", "b1_mean", "b2_mean"):
                cr[k] = float(extra[k][r]) if k in extra else float("nan")
            cap_rows.append(cr)
        for l in range(2):
            totals[f"rows_w{l + 1}"] += int(rows_acc[:, l].sum())
            totals[f"viol_w{l + 1}"] += int(viol_acc[:, l].sum())
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        cap_rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        H.write_csv(out / "cap_task.csv", cap_rows)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {cap_arm} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"rows w1 {int(rows_acc[:, 0].sum())} w2 {int(rows_acc[:, 1].sum())} "
                 f"viol {int(viol_acc.sum())}  {step_ms:.2f} ms/step  {el / 60:.1f} min")

    radii = {} if r_saved[0] is None else {
        f"r_l{l + 1}": r_saved[l][..., 0].cpu().numpy().astype(np.float32) for l in range(2)}
    if radii:
        np.savez(out / "radii.npz", seeds=np.asarray(seeds), **radii)
    prov = {"run_id": EXPERIMENT, **B.git_state(), "cap_arm": cap_arm,
            "cap_layers": [f"W{l + 1}" for l in layers], "radius_task": RADIUS_TASK,
            "cap_from_task": target_task,
            "followup": "post hoc matched fresh control; no training prefix",
            "cap_rule": "after every fresh-task Adam update, including first: n = ||w_i|| (float32); "
                        "rows with n > r_i become w_i * r_i / n; others untouched; W1 then W2; " +
                        ("b1/b2 multiplied by corresponding W-only row coefficient; " if cap_arm == "cap12b" else "biases untouched; ") +
                        "W3/b3, Adam moments, step counter, RNG untouched",
            "radii_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in radii.items()},
            "rows_written_total": {k: totals[k] for k in ("rows_w1", "rows_w2")},
            "violations_total": {k: totals[k] for k in ("viol_w1", "viol_w2")},
            "fresh_cap": cap_arm,
            "fresh_radius": "fixed original task-1-end radii; no initial projection",
            "arm": arm, "cond": cond, "seeds": seeds, "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": C.STEPS_PER_TASK, "steps_hard": steps_hard,
            "steps_easy": steps_easy, "batch": C.BATCH, "hidden": C.HIDDEN,
            "layer_shapes": C.layer_shapes(arm), "n_params": C.n_params(arm),
            "dims": list(C.DIMS), "n_classes": C.N_CLASSES, "hard_classes": C.HARD_CLASSES,
            "per_class_train": C.PER_CLASS_TRAIN, "classes_used": C.CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "std": {"mean": C.STD_MEAN, "std": C.STD_STD, "planes": "R,G,B x 1024"},
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"],
            "fresh_control": {"task": target_task, "n": R},
            "engine": "native run() of cifar5p1_mlp_0920 via inherited cap host; activation=" + activation + "; std; "
                      "stacked baddbmm, autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "activation": activation, "native_dphi": "autograd of host phi, including zero convention",
            "bias_scaling": cap_arm == "cap12b", "s_correction": 0,
            "host_module_sha256": sha256_file(Path(C.__file__)),
            "engine_sha256": sha256_file(Path(__file__)),
            "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "spec": "specs/spec_cap_act_box_fresh_followup_1018.md"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")
    return prov
