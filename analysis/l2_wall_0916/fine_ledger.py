#!/usr/bin/env python3
"""POSTHOC addendum to l2_wall_0916 (registered=0, analysis_grade=posthoc_not_preregistered).
Written after the spec's chimera re-run showed that, at the registered 6 points per task, the layer-2 cross term
dW2 . dmu2 is 40% of the t0-t10 transport.  This re-runs the chimera box again (same engine, streams and loop as
src.l2_wall_0916.run('chimera'), bit-checked against the committed layer_chimera_rl_0914) and takes the ledger and
the 3-factor Shapley split at every CUDA-graph block (25 updates) for tasks 1..TASKS, summed into windows.
Run from the repo root:  python3 analysis/l2_wall_0916/fine_ledger.py [--tasks 10]
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import l2_wall_0916 as R
from src import layer_chimera_rl_0914 as C

OUT = ROOT / "results/l2_wall_0916/fine_ledger"
COARSE = (0, 75, 375, 1500, 3000, 6000)


@torch.no_grad()
def probe(e, K1):
    W1, b1, W2, b2 = (e.p[i].double() for i in range(4))
    x = e.cx.double()
    N = x.shape[1]
    z1 = torch.bmm(x, W1.transpose(1, 2)) + b1[:, None, :]
    a1 = R.act64(z1, K1)
    mu1 = x.mean(1)
    mu2 = a1.mean(1)
    r2 = mu2.norm(dim=-1)
    e2 = mu2 / r2[:, None]
    ac = a1 - mu2[:, None, :]
    S2 = torch.bmm(ac.transpose(1, 2), ac) / N
    z2 = torch.bmm(a1, W2.transpose(1, 2)) + b2[:, None, :]
    zbar2 = z2.mean(1)
    sig2 = z2.std(1, unbiased=False)
    safe = torch.where(sig2 > 0, sig2, torch.full_like(sig2, float("nan")))
    q2 = torch.bmm(W2, e2[:, :, None])[..., 0]
    p2 = torch.bmm(a1, e2[:, :, None])[..., 0]
    rho2 = q2 * q2 * p2.var(1, unbiased=False)[:, None] / safe.square()
    wabs2 = W2.abs().sum(-1)
    return dict(W1=W1, b1=b1, mu1=mu1, zbar1=z1.mean(1),
                Wu1=W1.abs().sum(-1) * x.abs().amax((1, 2))[:, None] + b1.abs(),
                W2=W2, b2=b2, mu2=mu2, e2=e2, r2=r2, S2=S2, zbar2=zbar2, d2=zbar2 / safe, rho2=rho2, sig2=sig2,
                Wu2=wabs2 * a1.abs().amax((1, 2))[:, None] + b2.abs(), wabs2=wabs2, beta=torch.zeros_like(b2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mod = C
    e, models, K1, K2, kind = R.build("chimera", "cuda")
    ref = R.committed("chimera")
    gperm = {s: mod.stream("env_perm_0913", s) for s in range(3)}
    glabel = {s: mod.stream("env_labels_0913", s) for s in range(3)}
    gbatch = {s: mod.stream("env_batch_0913", s) for s in range(3)}
    ax = mod.read_idx(mod.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = mod.read_idx(mod.DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    subset = {s: torch.randperm(len(ax), generator=mod.stream("rl_subset", s))[:mod.N].numpy() for s in range(3)}
    xs = {s: torch.tensor(ax[subset[s]], device=e.device) for s in range(3)}
    ys_cpu = {s: torch.tensor(ay[subset[s]]) for s in range(3)}
    ys = {s: ys_cpu[s].to(e.device) for s in range(3)}
    FIELDS = ("meas", "self", "up", "cross", "bias", "stretch", "rot", "sh_d_W", "sh_d_mu", "sh_d_S", "d_d",
              "sh_rho_W", "sh_rho_mu", "sh_rho_S", "d_rho", "sh_sig_W", "sh_sig_S", "d_sig")
    acc_task = {}                                   # (task, field) -> [M,100] float64 sums (switch interval counted in the new task)
    acc_coarse = {}                                 # (task, coarse_end, field) -> sums
    profile = {f: [] for f in ("self", "up", "cross", "bias", "meas")}   # per block, unit mean [M]
    prof_pts, resid, bits, coarse_pts = [], {}, [], {}
    prev = None

    def add(key, L):
        for f in FIELDS:
            v = torch.nan_to_num(L[f], nan=0.0)
            acc = acc_task if len(key) == 1 else acc_coarse
            k = key + (f,)
            acc[k] = acc[k] + v if k in acc else v.clone()

    def step_probe(task, step):
        nonlocal prev
        cur = probe(e, K1)
        if step in COARSE:
            coarse_pts[task, step] = {k: cur[k].detach().cpu().numpy() for k in ("mu2", "zbar2", "d2", "rho2")}
        if prev is not None:
            L, Rr = R.ledger(prev, cur, mod.N)
            for k, v in Rr.items():
                resid[k] = max(resid.get(k, 0.0), float(torch.nan_to_num(v, nan=0.0).max()))
            add((task,), L)
            cend = next(c for c in COARSE if c >= step) if step > 0 else 0
            add((task, cend), L)
            for f in profile:
                profile[f].append(torch.nan_to_num(L[f], nan=0.0).mean(-1).cpu().numpy())
            prof_pts.append((task, step))
        prev = cur

    e.capture()
    t0 = time.monotonic()
    for task in range(1, a.tasks + 1):
        steps = mod.STEPS
        epochs_needed = -(-(steps * mod.BATCH) // mod.N)
        orders = {}
        for s in range(3):
            flat = torch.stack([torch.randperm(mod.N, generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:steps * mod.BATCH]
            orders[s] = flat.reshape(steps, mod.BATCH)
        perm_cpu = {s: torch.randperm(784, generator=gperm[s]) for s in range(3)}
        perm = {s: perm_cpu[s].to(e.device) for s in range(3)}
        labels_cpu = {s: torch.randint(10, (mod.N,), generator=glabel[s]) for s in range(3)}
        labels = {s: labels_cpu[s].to(e.device) for s in range(3)}
        orderstack = torch.stack([orders[m["seed"]] for m in models], 1).to(e.device)
        floors = {}
        for s in range(3):
            floors[s, "PM"] = float(torch.bincount(ys_cpu[s], minlength=10).max()) / mod.N
            floors[s, "RL"] = float(torch.bincount(labels_cpu[s], minlength=10).max()) / mod.N
        for j, m in enumerate(models):
            s = m["seed"]
            e.cx[j].copy_(xs[s][:, perm[s]] if m["env"] == "PM" else xs[s])
            yy = ys[s] if m["env"] == "PM" else labels[s]
            e.cy[j].copy_(torch.nn.functional.one_hot(yy, 10))
        e.acc.zero_(); e.ce.zero_()
        step_probe(task, 0)
        for step in range(0, steps, mod.BLOCK):
            e.indices.copy_(orderstack[step:step + mod.BLOCK]); e.replay_block()
            step_probe(task, step + mod.BLOCK)
        met, units = e.evaluate()
        oa = (e.acc / steps).cpu().numpy(); oc = (e.ce / steps).cpu().numpy()
        bits.append(R.bitcheck(ref, "chimera", models, task, met, units, oa, oc, floors))
        print(f"[{time.monotonic() - t0:7.1f}s] fine task {task}/{a.tasks} bit units={bits[-1]['units_exact']} rows={bits[-1]['rows_exact']}", flush=True)
    # consistency with the main run's coarse diagnostics (same float64 recomputation at the same states)
    D = np.load(ROOT / "results/l2_wall_0916/logs/diag_chimera.npz")
    pts = list(zip(D["task"].tolist(), D["step"].tolist()))
    cons = 0.0
    for (t, s), v in coarse_pts.items():
        i = pts.index((t, s))
        cons = max(cons, float(np.nanmax(np.abs(D["u_zbar_l2"][i] - v["zbar2"]) / (1 + np.abs(v["zbar2"])))))
    Lc = np.load(ROOT / "results/l2_wall_0916/logs/ledger_chimera.npz")
    rows = []
    for j, m in enumerate(models):
        for (wa, wb) in ((0, 5), (5, 10), (0, 10), (0, 2), (2, 3), (3, 5)):
            if wb > a.tasks:
                continue
            row = dict(model=R.model_key("chimera", m), window=f"t{wa}-t{wb}")
            for f in FIELDS:
                tot = sum(acc_task[(t, f)][j] for t in range(wa + 1, wb + 1))
                row[f"fine_mean_{f}"] = float(tot.mean().cpu())
                row[f"fine_med_{f}"] = float(tot.median().cpu())
            # the registered coarse ledger over the same window (points: end of task wa .. end of task wb)
            i0 = 0 if wa == 0 else max(i for i, p in enumerate(pts) if p[0] == wa)
            i1 = max(i for i, p in enumerate(pts) if p[0] == wb)
            for f in ("meas", "self", "up", "cross", "bias", "sh_d_W", "sh_d_mu", "sh_d_S", "sh_rho_W", "sh_rho_mu", "sh_rho_S"):
                row[f"coarse_mean_{f}"] = float(Lc[f"l_{f}"][i0:i1, j].sum(0).mean())
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "fine_ledger_windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    np.savez_compressed(ROOT / "results/l2_wall_0916/logs/fine_ledger_profile.npz",
                        task=np.array([p[0] for p in prof_pts]), step=np.array([p[1] for p in prof_pts]),
                        **{f: np.stack(v) for f, v in profile.items()},
                        **{f"task_{f}": np.stack([acc_task[(t, f)].cpu().numpy() for t in range(1, a.tasks + 1)]) for f in FIELDS})
    checks = dict(tasks=a.tasks, blocks=len(prof_pts), identity_max_ratio_to_tol=resid,
                  identity_pass=all(v <= 1 for v in resid.values()),
                  bit_all_exact=all(b["units_exact"] and b["rows_exact"] for b in bits),
                  coarse_point_consistency_max_rel=cons, wall_seconds=time.monotonic() - t0,
                  code_sha256=R.sha(Path(__file__)), runner_sha256=R.sha(R.__file__),
                  registered=0, analysis_grade="posthoc_not_preregistered")
    (out / "checks_fine.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
