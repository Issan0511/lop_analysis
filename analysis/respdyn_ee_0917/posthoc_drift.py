#!/usr/bin/env python3
"""respdyn_ee_0917 -- POST HOC (not registered): what the moving field is made of.

    OMP_NUM_THREADS=1 python3 analysis/respdyn_ee_0917/posthoc_drift.py

For each seed, replay the shadows (the natural continuations N2, N5, N10, N20) from the saved branch
states with RE.train_task (bit-equal to the run's shadows, check S-t0) and read the full-batch second-layer
field z2 at s = 0, 75, 150, 300, 750, 1500, 3000, 6000 of the first continuation task.  The field change
D(s) = z2(s) - z2(0) (1200 images x 100 units, float64) is split into
    uniform  the mean over units and images
    unit     each unit's mean over images, minus the uniform part
    image    the rest (how each unit's response pattern over images changed)
and reported as RMS over (unit, image), with the dead share of the moved field
phi'_train(z2(s)) == 0 (what a dyn arm that has not moved itself would train with).
Also the label structure of the image part: for each unit, the fraction of the image part's variance
explained by the shadow's own task labels (between-class share, 10 classes), against the same share for
the arm's labels (a label draw the shadow never saw; the control).
Writes results/respdyn_ee_0917/posthoc_drift.csv and prints seed means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)
torch.set_flush_denormal(True)
from src import pmnist_0905 as H            # noqa: E402
from src import pmnist_rlmnist_0906 as RL   # noqa: E402
from src import resp_ee_0917 as RE          # noqa: E402
from src import respdyn_ee_0917 as R        # noqa: E402

OUT = REPO / "results" / "respdyn_ee_0917"
POINTS = (0, 75, 150, 300, 750, 1500, 3000, 6000)
SHADOWS = {"N10 (S2dyn_10r)": (10, 2), "N2 (R2dyn_10)": (2, 10), "N5 (S2dyn_5r)": (5, 2), "N20 (S2dyn_20r)": (20, 2)}


def between_share(D: torch.Tensor, y: torch.Tensor) -> float:
    """Mean over units of (between-class variance / total variance) of D[:, unit] under labels y."""
    tot = D.var(0, unbiased=False)
    mu = D.mean(0)
    btw = torch.zeros_like(tot)
    for c in range(10):
        m = y == c
        if m.any():
            btw += m.double().mean() * (D[m].mean(0) - mu) ** 2
    ok = tot > 0
    return float((btw[ok] / tot[ok]).mean()) if ok.any() else float("nan")


def main() -> None:
    H.setup("cpu")
    mnist = H.Mnist(torch.device("cpu"))
    rows = []
    for seed in range(10):
        cks = torch.load(OUT / "runs" / f"s{seed}" / "branch_states.pt", weights_only=False)
        x = mnist.train_x[RL.subset_idx(seed)]
        for name, (src, arm_branch) in SHADOWS.items():
            ck = cks[src]
            params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
            adam = ([q.clone() for q in ck["m"]], [q.clone() for q in ck["v"]], [ck["tc"]])
            y_sh = RL.task_labels(RE.gen_from(ck["g_lab"]))
            y_arm = RL.task_labels(RE.gen_from(cks[arm_branch]["g_lab"]))
            gb = RE.gen_from(ck["g_batch"])
            z0 = None
            done = 0
            for s in POINTS:
                if s > done:
                    RE.train_task(params, adam, x, y_sh, gb, (s - done) // RE.SPE)
                    done = s
                with torch.no_grad():
                    z2 = R.z2_full(params, x)
                z = z2.double()
                if z0 is None:
                    z0 = z
                D = z - z0
                u = D.mean()
                unit = D.mean(0) - u
                img = D - D.mean(0)
                gt = RE.dphi_train(z2)
                rows.append({"seed": seed, "shadow": name, "s": s, "mean_shift": float(u),
                             "rms_total": float(D.square().mean().sqrt()),
                             "rms_unit": float(unit.square().mean().sqrt()),
                             "rms_image": float(img.square().mean().sqrt()),
                             "dead_share": float((gt == 0).double().mean()),
                             "between_own_labels": between_share(img, y_sh) if s > 0 else float("nan"),
                             "between_arm_labels": between_share(img, y_arm) if s > 0 else float("nan")})
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "posthoc_drift.csv", index=False)
    g = df.groupby(["shadow", "s"]).mean(numeric_only=True).drop(columns="seed")
    pd.set_option("display.width", 200)
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
