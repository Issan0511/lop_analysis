#!/usr/bin/env python3
"""The side run of respdyn_ee_0917 (spec section 6): the natural RL ELU->ELU trajectory with the second
layer's training derivative never exactly 0.

    OMP_NUM_THREADS=1 python3 src/respdyn_nat_0917.py --variant f64_felu --seed 0 --out results/respdyn_ee_0917/nat/f64_felu_s0

resp_ee_0917 found that the box's ELU (expm1, float32) trains with a derivative that is exactly 0 for
z < ln 2^-24 = -16.64, so the collapsed second layer is dead in the ReLU sense and "a small but nonzero
derivative cannot learn" was never tested.  This runner repeats the natural prefix (same init, images,
labels, orders, Adam, 6000 updates per task) with the dtype and the activation switched:

    f64_felu  float64 everything, F.elu (its backward is exp(z) from the input: > 0 down to z = -708)
    f32_felu  float32, F.elu (> 0 down to z = -87 with flush_denormal on)
    f64_host  float64, the host ELU (expm1; its derivative is exactly 0 below z = -36.7)
    f32_host  float32, the host ELU: resp_ee's prefix itself (check S8: bit-equal to RE.run_prefix)

The update loop is RE.train_task's arithmetic with the forward's activation and the dtype as parameters.
No arms, no bit identity with the float32 run (a different trajectory), reported per task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256, file_sha256; not touched
from src import resp_ee_0917 as RE               # the box's constants and helpers; not touched

EXPERIMENT = "respdyn_ee_0917"
PREREG_COMMIT = "7f15cad5bc7456250998e4db009ace2055c8adc6"                               # set with the main runner's
LR, BETA1, BETA2, EPS = RE.LR, RE.BETA1, RE.BETA2, RE.EPS
N_IMG, BATCH, SPE = RE.N_IMG, RE.BATCH, RE.SPE
EPOCHS = RE.EPOCHS
N_TASKS = RE.PREFIX_T                            # 22
WEIGHTS = RE.WEIGHTS


class FELU:
    """torch.nn.functional.elu (alpha 1): the backward is exp(x) from the input, not expm1 + 1."""
    name = "F.elu"

    @staticmethod
    def phi(z):
        return F.elu(z)


VARIANTS = {"f64_felu": (torch.float64, FELU), "f32_felu": (torch.float32, FELU),
            "f64_host": (torch.float64, RE.ACT), "f32_host": (torch.float32, RE.ACT)}


def forward(params, x, act):
    W1, b1, W2, b2, W3, b3 = params
    z1 = x @ W1.T + b1
    a1 = act.phi(z1)
    z2 = a1 @ W2.T + b2
    a2 = act.phi(z2)
    return z1, a1, z2, a2, a2 @ W3.T + b3


def train_task(params, adam, x, y, g_batch, epochs, act):
    """RE.train_task (fwd None) with forward(.., act)."""
    spt = SPE * epochs
    pts = set(RE.eps_points(spt))
    ce = torch.empty(spt)
    acc = torch.empty(spt)
    stp = torch.empty(spt, len(WEIGHTS))
    epsf = {}
    m, v, tc = adam
    for ep in range(epochs):
        order = torch.randperm(N_IMG, generator=g_batch).to(x.device)
        xs, ys = x[order], y[order]
        for j in range(SPE):
            s = ep * SPE + j
            xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
            out = forward(params, xb, act)
            loss = F.cross_entropy(out[4], yb)
            ce[s] = loss.detach()
            acc[s] = (out[4].detach().argmax(1) == yb).float().mean()
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                tc[0] += 1
                c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]
                for k, (p, gr, mi, vi) in enumerate(zip(params, grads, m, v)):
                    mi.mul_(BETA1).add_(gr, alpha=1 - BETA1)
                    vi.mul_(BETA2).addcmul_(gr, gr, value=1 - BETA2)
                    upd = LR * (mi / c1) / ((vi / c2).sqrt() + EPS)
                    p -= upd
                    if k % 2 == 0:
                        stp[s, k // 2] = upd.square().mean().sqrt()
                if s + 1 in pts:
                    epsf[s + 1] = [float(((v[k] / c2).sqrt() < EPS).double().mean()) for k in WEIGHTS]
    return ce, acc, stp, epsf


def layer2_point(params, x, act) -> dict:
    """The second layer on the 1200 images: the preactivation and the derivative training uses,
    taken from autograd through the variant's own activation in the run's dtype."""
    with torch.no_grad():
        z1, a1, z2, a2, logits = forward(params, x, act)
    zr = z2.detach().clone().requires_grad_(True)
    g, = torch.autograd.grad(act.phi(zr).sum(), zr)
    g = g.detach().double()
    z64 = z2.double()
    n = g.shape[0]
    s1 = g.sum(0)
    neff = torch.where(s1 > 0, s1 * s1 / (n * g.square().sum(0)).clamp(min=1e-300), torch.zeros_like(s1))
    pos = g[g > 0]
    return {"zbar2": float(z64.mean()), "zbar2_unit_min": float(z64.mean(0).min()),
            "z2_max": float(z64.max()), "z2_min": float(z64.min()),
            "gtr2": float(g.mean()), "zero2": float((g == 0).double().mean()),
            "below_2p24_2": float((g < 2.0 ** -24).double().mean()),
            "log10_g2_median": float(pos.log10().median()) if pos.numel() else float("-inf"),
            "deadunits2": float((g == 0).all(0).double().mean()), "neffT2": float(neff.mean()),
            "zero_share_below_m1664": float((z64 < math.log(2.0 ** -24)).double().mean()),
            "logits": logits}


def run_nat(variant: str, seed: int, mnist: H.Mnist, epochs: int = EPOCHS, n_tasks: int = N_TASKS,
            progress: bool = False) -> tuple[list[dict], dict]:
    dtype, act = VARIANTS[variant]
    params = [p.detach().to(dtype).clone().requires_grad_(True)
              for p in H.init_params(seed, H.setup("cpu"))]
    adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    idx = RL.subset_idx(seed)
    x = mnist.train_x[idx].to(dtype)
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    spt = SPE * epochs
    rows = []
    t0 = time.time()
    for t in range(1, n_tasks + 1):
        t_task = time.time()
        y = RL.task_labels(g_lab)
        ce, acc, stp, epsf = train_task(params, adam, x, y, g_batch, epochs, act)
        lp = layer2_point(params, x, act)
        logits = lp.pop("logits")
        memo = float((logits.argmax(1) == y).float().mean())
        with torch.no_grad():
            c2 = 1 - BETA2 ** adam[2][0]
            eps_w2 = float(((adam[1][2] / c2).sqrt() < EPS).double().mean())
        row = {"variant": variant, "seed": seed, "task": t, "dtype": str(dtype), "act": act.name,
               **RE.task_row(ce, acc, stp, epsf, y, spt), "memo_acc": memo, "eps_w2_end": eps_w2,
               **lp, "state_sha256": SH.state_sha256(params, adam, act), "sec": time.time() - t_task}
        rows.append(row)
        if progress:
            print(f"[{time.time() - t0:7.1f}s] {variant} s{seed} task {t}: online {row['online_acc']:.4f} "
                  f"zbar2 {row['zbar2']:+.1f} zero2 {row['zero2']:.3f}", flush=True)
    info = {"param_dtypes": sorted({str(p.dtype) for p in params}), "x_dtype": str(x.dtype),
            "moment_dtypes": sorted({str(q.dtype) for q in adam[0] + adam[1]})}
    return rows, info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    git0 = {"hash": H.git_hash(), "dirty": SH.git_dirty(["src", "analysis/respdyn_ee_0917"])}
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, info = run_nat(args.variant, args.seed, mnist, args.epochs, args.tasks, progress=True)
    RE.write_csv(out / "nat.csv", rows)
    root = Path(__file__).resolve().parents[1]
    prov = {"experiment": EXPERIMENT, "prereg_commit": PREREG_COMMIT, "variant": args.variant,
            "seed": args.seed, "epochs_per_task": args.epochs, "tasks": args.tasks, **info,
            "git_hash": git0["hash"], "git_dirty_code": git0["dirty"],
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "flush_denormal": RE._flush_is_on(),
            "code_sha256": {f"src/{n}": SH.file_sha256(root / "src" / n) for n in
                            ("respdyn_nat_0917.py", "resp_ee_0917.py", "pmnist_0905.py", "pmnist_rlmnist_0906.py")},
            "data_sha256": mnist.sha256, "seconds_total": time.time() - t0,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"wrote {out} ({len(rows)} tasks, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
