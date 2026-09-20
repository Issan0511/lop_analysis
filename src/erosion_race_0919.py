#!/usr/bin/env python3
"""Dense early dynamics and confinement-only mean-displacement intervention.

The parent is rlcifar_mlp_battle_0918. Its streams, stacked forward, CE gradient,
Adam expression and adaptive-Snake update are preserved. Diagnostic state never
feeds the natural update. See specs/spec_erosion_race_0919.md for interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rlcifar_mlp_battle_0918 as E

H, RC = E.H, E.RC
EXPERIMENT = "erosion_race_0919"
ARMS = ("R", "GELU", "ELU", "LR", "LK001", "SNA")
BOUNDARIES = (1, 10, 30, 75, 150, 300, 750, 1500, 3000, 5000,
              7500, 10000, 15000, 22500, 30000)
FLUX_NAMES = ("down", "up", "net", "conf", "hist", "label", "correction",
              "roundoff", "conf_common", "correction_common")
LR, BETA1, BETA2, EPS = 1e-3, .9, .999, 1e-8


def split_adam_components(m_conf, m_hist, m_total, v_total, inv_c1, inv_c2,
                          lr=LR):
    """Additive first-moment attribution under the actual shared Adam denominator.

    These are attributions of the actual preconditioned update, not independent
    optimizers. In particular the label remainder can have either sign.
    """
    denominator = (v_total * inv_c2).sqrt() + EPS
    m_label = m_total - m_conf - m_hist
    return tuple(-(lr * (m * inv_c1) / denominator)
                 for m in (m_conf, m_hist, m_label))


def confinement_correction(uW, ub, mean_x):
    """Remove only negative confinement mean displacement, minimally in (W,b).

    Shapes: uW=(runs, units, inputs), ub=(runs, units), mean_x=(runs, inputs).
    Returns correction W, correction b, and the original mean displacement.
    """
    displacement = (uW * mean_x[:, None, :]).sum(-1) + ub
    amount = -displacement.clamp(max=0) / (mean_x.square().sum(-1)[:, None] + 1)
    return amount[:, :, None] * mean_x[:, None, :], amount, displacement


def intervention_active(mode: str, task: int, step: int) -> bool:
    if mode == "base" or task != 1:
        return False
    return 1 <= step <= 5000 if mode == "early" else 5001 <= step <= 10000


def state_digest(state: dict) -> str:
    """Content hash independent of torch serialization container metadata."""
    digest = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            a = obj.detach().cpu().contiguous().numpy()
            digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str):
                digest.update(str(k).encode()); visit(obj[k])
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                visit(v)
        else:
            digest.update(repr(obj).encode())
    visit(state)
    return digest.hexdigest()


def run(arm: str, mode: str, seeds: list[int], n_tasks: int, epochs: int,
        device, out: Path, cifar=None, graph: bool = True, observe: bool = True,
        max_steps: int | None = None, progress=None) -> dict:
    if arm not in ARMS or mode not in ("base", "early", "late"):
        raise ValueError((arm, mode))
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    if n_tasks < 1 or epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("Positive task, epoch and step counts required")
    device, out = torch.device(device), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists() or (out / "provenance.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {out}")
    progress = progress or (lambda message: print(message, flush=True))
    started = time.time()
    start_git_state = E.git_state()
    code_paths = [Path(__file__), Path(E.__file__), Path(H.__file__), Path(RC.__file__),
                  H.REPO / "specs" / f"spec_{EXPERIMENT}.md"]
    source_sha256 = {str(p.relative_to(H.REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in code_paths if p.exists()}
    cifar = cifar or RC.Cifar10()
    R = len(seeds)
    X = torch.stack([E.slot_inputs(cifar, s, "raw", device) for s in seeds])
    init = [H.init_params(s, device, E.DIMS) for s in seeds]
    P = [torch.stack([p[i].detach() for p in init]).contiguous().requires_grad_(True)
         for i in range(6)]
    initial_params_sha256 = state_digest({"P": P})
    act = E.make_act(arm)
    act.init_state(R, device, key=f"{arm}|{seeds}|{['raw']}")
    adam_m, adam_v = [torch.zeros_like(p) for p in P], [torch.zeros_like(p) for p in P]
    m_conf, m_hist = [torch.zeros_like(p) for p in P[:2]], [torch.zeros_like(p) for p in P[:2]]
    g_lab = {s: H.stream("rlc_labels", s) for s in seeds}
    g_batch = {s: H.stream("rlc_batch", s) for s in seeds}
    mean_x = X.mean(1)
    probe_x = X[:, :64]
    with torch.no_grad():
        initial_forward = E.forward(P, X, act)
        initial_z = initial_forward[0]
        initial_output_sd = [initial_forward[k].std(1, unbiased=False) for k in (1, 3)]
        sigma0 = initial_z.std(1, unbiased=False).clamp_min(1e-6)
        mask = initial_z[:, :64] > 0
        count = mask.sum((1, 2)).clamp_min(1)
        weighted_mask = mask.to(X.dtype) / sigma0[:, None, :] / count[:, None, None]
        projected_x = torch.bmm(weighted_mask.transpose(1, 2), probe_x)
        projected_b = weighted_mask.sum(1)
    ar = torch.arange(R, device=device)[:, None]
    static_idx = torch.zeros(R, E.BATCH, dtype=torch.long, device=device)
    Ydev = torch.zeros(R, E.N_IMAGES, dtype=torch.long, device=device)
    inv_c1, inv_c2, active = [torch.zeros((), device=device) for _ in range(3)]
    acc_sum = torch.zeros(R, device=device)
    flux = torch.zeros(R, len(FLUX_NAMES), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    tc = 0

    def signed_component(uW, ub):
        return (uW * projected_x).sum((1, 2)) + (ub * projected_b).sum(1)

    def step():
        if observe:
            with torch.no_grad():
                before_W, before_b = P[0].clone(), P[1].clone()
        xb, yb = X[ar, static_idx], Ydev[ar, static_idx]
        z1, a1, z2, a2, logits = E.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(logits.reshape(-1, E.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, E.BATCH).mean(1)
        # The actual loss and its gradient are exactly the parent's expressions.
        grads = torch.autograd.grad(lossv.sum(), P, retain_graph=True)
        conf_loss = (torch.logsumexp(logits, -1) - logits.mean(-1)).mean(1).sum()
        conf_grads = torch.autograd.grad(conf_loss, P[:2])
        with torch.no_grad():
            hit = (logits.detach().argmax(-1) == yb).float().mean(1)
            acc_sum.add_(hit)
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(BETA1).add_(gr, alpha=1 - BETA1)
                vi.mul_(BETA2).addcmul_(gr, gr, value=1 - BETA2)
                p.sub_(LR * (mi * inv_c1) / ((vi * inv_c2).sqrt() + EPS))
            for mc, mh, gc in zip(m_conf, m_hist, conf_grads):
                mc.mul_(BETA1).add_(gc, alpha=1 - BETA1)
                mh.mul_(BETA1)
            parts = [split_adam_components(mc, mh, mt, vt, inv_c1, inv_c2)
                     for mc, mh, mt, vt in zip(m_conf, m_hist, adam_m[:2], adam_v[:2])]
            cW, cb, common = confinement_correction(parts[0][0], parts[1][0], mean_x)
            cW, cb = cW * active, cb * active
            if mode != "base":
                P[0].add_(cW); P[1].add_(cb)
            if observe:
                actual_dW, actual_db = P[0] - before_W, P[1] - before_b
                dz = torch.baddbmm(actual_db[:, None, :], probe_x, actual_dW.transpose(1, 2))
                down = (-dz).clamp_min(0).mul(weighted_mask).sum((1, 2))
                up = dz.clamp_min(0).mul(weighted_mask).sum((1, 2))
                net = dz.mul(weighted_mask).sum((1, 2))
                conf, hist, label = [signed_component(parts[0][j], parts[1][j]) for j in range(3)]
                correction = signed_component(cW, cb)
                residual = net - conf - hist - label - correction
                corrected_common = (cW * mean_x[:, None, :]).sum(-1) + cb
                flux.add_(torch.stack((down, up, net, conf, hist, label, correction,
                                       residual, common.mean(1), corrected_common.mean(1)), 1))
            act.update(z1.detach(), z2.detach())

    use_graph = graph and device.type == "cuda"
    cg = None
    if use_graph:
        touched = (*P, *adam_m, *adam_v, *m_conf, *m_hist, acc_sum, flux, step_t, bad_step)
        keep = [q.detach().clone() for q in touched]
        keep_act = act.state()
        inv_c1.fill_(1); inv_c2.fill_(1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip(touched, keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    rows, task_rows, unit_states, stream_hashes = [], [], [], []
    @torch.no_grad()
    def capture(task, local_step, start_step, last_acc, rank=False):
        z1, a1, z2, a2, logits = E.forward(P, X, act)
        ce = F.cross_entropy(logits.reshape(-1, E.N_CLASSES), Ydev.reshape(-1),
                             reduction="none").view(R, E.N_IMAGES).mean(1)
        accuracy = (logits.argmax(-1) == Ydev).float().mean(1)
        advantage = (logits.gather(-1, Ydev[:, :, None]).squeeze(-1) - logits.mean(-1)).mean(1)
        confidence = (torch.logsumexp(logits, -1) - logits.mean(-1)).mean(1)
        interval = local_step - start_step
        unit = {"task": np.asarray(task), "step": np.asarray(local_step)}
        metrics = []
        for li, (z, a) in enumerate(((z1, a1), (z2, a2))):
            d = act.dphi(z, li)
            values = {"zmean": z.mean(1), "zsd": z.std(1, unbiased=False),
                      "outputsd": a.std(1, unbiased=False), "gain": d.square().mean(1),
                      "ppos": (z > 0).float().mean(1),
                      "dead": (d.abs().amax(1) < E.DEAD_TOL).float()}
            metrics.append(values)
            for key, value in values.items():
                unit[f"{key}_l{li + 1}"] = value.cpu().numpy()
        unit_states.append(unit)
        f_cpu, acc_cpu = flux.cpu(), acc_sum.cpu()
        for r, seed in enumerate(seeds):
            row = {"arm": arm, "mode": mode, "seed": seed, "task": task,
                   "step": local_step, "global_step": tc, "window_start": start_step,
                   "window_steps": interval, "loss": float(ce[r]), "acc": float(accuracy[r]),
                   "advantage": float(advantage[r]), "confidence": float(confidence[r]),
                   "online_acc_window": float((acc_cpu[r] - last_acc[r]) / interval) if interval else float("nan"),
                   "initial_positive_count": int(count[r]), "observed": observe}
            for j, key in enumerate(FLUX_NAMES):
                row[key] = float(f_cpu[r, j]) if observe else float("nan")
            for li, vals in enumerate(metrics):
                for key, value in vals.items():
                    row[f"{key}_l{li + 1}"] = float(value[r].mean() if key in ("dead", "ppos") else value[r].median())
                eligible = initial_output_sd[li][r] > 1e-6
                ratio = vals["outputsd"][r] / initial_output_sd[li][r].clamp_min(1e-6)
                row[f"output_sd_ratio_l{li + 1}"] = float(ratio[eligible].median()) if bool(eligible.any()) else float("nan")
                row[f"output_sd_ratio_n_l{li + 1}"] = int(eligible.sum())
                if rank:
                    row[f"eff_rank_l{li + 1}"] = H.eff_rank((a1, a2)[li][r])
            rows.append(row)
        flux.zero_()
        H.write_csv(out / "diagnostics.csv", rows)
        return acc_cpu.clone()

    steps_per_task = E.STEPS_PER_EPOCH * epochs
    for task in range(1, n_tasks + 1):
        labels = [RC.task_labels(g_lab[s]) for s in seeds]
        Ydev.copy_(torch.stack(labels).to(device))
        label_hash = {str(s): hashlib.sha256(y.numpy().tobytes()).hexdigest()
                      for s, y in zip(seeds, labels)}
        order_hash = {s: hashlib.sha256() for s in seeds}
        for mc, mh, mt in zip(m_conf, m_hist, adam_m):
            mc.zero_(); mh.copy_(mt)
        acc_sum.zero_(); flux.zero_(); step_t.zero_(); bad_step.fill_(-1)
        last_acc = torch.zeros(R)
        last_acc = capture(task, 0, 0, last_acc, rank=True)
        last_boundary = local_step = 0
        task_start = time.time()
        for epoch in range(epochs):
            orders = [torch.randperm(E.N_IMAGES, generator=g_batch[s]) for s in seeds]
            for s, order in zip(seeds, orders):
                order_hash[s].update(order.numpy().tobytes())
            ORD = torch.stack(orders).to(device)
            for j in range(E.STEPS_PER_EPOCH):
                local_step += 1; tc += 1
                static_idx.copy_(ORD[:, j * E.BATCH:(j + 1) * E.BATCH])
                inv_c1.fill_(1 / (1 - BETA1 ** tc)); inv_c2.fill_(1 / (1 - BETA2 ** tc))
                active.fill_(float(intervention_active(mode, task, local_step)))
                if cg is None:
                    step()
                else:
                    cg.replay()
                end_run = max_steps is not None and tc >= max_steps
                if local_step in BOUNDARIES or local_step == steps_per_task or end_run:
                    last_acc = capture(task, local_step, last_boundary, last_acc,
                                       rank=(local_step == steps_per_task or end_run))
                    last_boundary = local_step
                if end_run:
                    break
            if max_steps is not None and tc >= max_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        finite = all(bool(torch.isfinite(p).all()) for p in P)
        if not finite or bool((bad_step >= 0).any()):
            raise FloatingPointError(f"Nonfinite run {arm}/{mode} task {task}")
        metrics, _, _ = E.evaluate(P, X, Ydev, act)
        for r, seed in enumerate(seeds):
            task_rows.append({"arm": arm, "mode": mode, "seed": seed, "cond": "raw",
                              "slot": r, "lr": LR, "task": task, "iv": mode,
                              "steps": local_step, "complete_task": local_step == steps_per_task,
                              "online_acc": float(acc_sum[r]) / local_step,
                              "memo_acc": metrics[r]["acc"], **metrics[r]})
        H.write_csv(out / "per_task.csv", task_rows)
        stream_hashes.append({"task": task, "labels": label_hash,
                              "orders": {str(s): h.hexdigest() for s, h in order_hash.items()},
                              "order_epochs_drawn": epoch + 1})
        progress(f"{arm}/{mode} task={task} steps={local_step} "
                 f"wall={time.time() - task_start:.1f}s total={time.time() - started:.1f}s")
        if max_steps is not None and tc >= max_steps:
            break
    state = {"P": [p.detach().cpu() for p in P], "m": [p.cpu() for p in adam_m],
             "v": [p.cpu() for p in adam_v], "tc": tc,
             "V": [v.cpu() for v in act.V] if act.adaptive else None,
             "m_conf": [p.cpu() for p in m_conf], "m_hist": [p.cpu() for p in m_hist],
             "g_lab": {s: g_lab[s].get_state() for s in seeds},
             "g_batch": {s: g_batch[s].get_state() for s in seeds}}
    torch.save(state, out / "final.pt")
    np.savez_compressed(out / "unit_metrics.npz", seeds=np.asarray(seeds),
                        **{k: np.stack([u[k] for u in unit_states]) for k in unit_states[0]})
    digest = state_digest(state)
    provenance = {"run_id": EXPERIMENT, **start_git_state, "arm": arm, "mode": mode,
                  "seeds": seeds, "R": R, "conds": ["raw"], "tasks": n_tasks,
                  "epochs": epochs, "steps_per_task": steps_per_task, "completed_steps": tc,
                  "max_steps": max_steps, "observed": observe, "graph": use_graph,
                  "lr": LR, "optimizer": "parent elementwise Adam", "batch": E.BATCH,
                  "dims": E.DIMS, "data_sha256": cifar.sha256,
                  "source_sha256": source_sha256, "initial_params_sha256": initial_params_sha256,
                  "task_stream_sha256": stream_hashes,
                  "subset_sha256": {str(s): hashlib.sha256(RC.subset_idx(s).numpy().tobytes()).hexdigest() for s in seeds},
                  "rng_roles": ["rlc_subset", "rlc_labels", "rlc_batch", "init"],
                  "probe": "first 64 subset images; fixed initial positive L1 mask; initial population sd >= 1e-6",
                  "flux": "window sums of mean perpoint ((Wafter-Wbefore)x+(bafter-bbefore))/sigma0 on fixed initial positive mask; up/down rectify before averaging",
                  "components": "shared actual Adam denominator and global correction; task-reset conf and inherited history",
                  "intervention": "task1 early1:5000 or late5001:10000; remove negative conf mean displacement minimally in W1,b1; moments untouched",
                  "engine": "parent stacked baddbmm + extra conf autograd and independent diagnostics",
                  "state_sha256": digest, "torch": torch.__version__, "device": str(device),
                  "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "wall_clock_s": time.time() - started}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--mode", default="base", choices=("base", "early", "late"))
    parser.add_argument("--seeds", default="1001-1005")
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--observe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.data_dir:
        RC.DATA_DIR = args.data_dir
    torch.set_num_threads(args.threads)
    device = H.setup(args.device)
    out = args.out or H.REPO / "results" / EXPERIMENT / f"{args.arm}_{args.mode}"
    run(args.arm, args.mode, E.parse_ints(args.seeds), args.tasks, args.epochs,
        device, out, graph=not args.no_graph, observe=args.observe, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
