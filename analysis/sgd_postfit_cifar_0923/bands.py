#!/usr/bin/env python3
"""Band stock of W1 per arm (spec §3, descriptive): N_b(t) = ||W1_t Q_b||_F^2 at the task ends.

    python3 analysis/sgd_postfit_cifar_0923/bands.py [--root results/sgd_postfit_cifar_0923] \
        [--basis DIR] [--tasks 10,25,50]

The basis is altlabels' band_ledger basis (per seed: eigenvectors of the centred image
covariance of that seed's 1,200 std images, then mu_perp), cached as basis_std_s<seed>.npz
(Q: 3072 x 1200).  Bands as band_ledger.EIGBANDS: top 0-10, mid1 10-100, mid2 100-439,
low 439-1199, mu 1199; comp = the rest of ||W1||^2.  Writes bands.csv (arm, seed, t, band, N).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASIS = Path("/tmp/claude-1000/-home-issan-Projects-claude/cc0afbce-b1e9-456c-95f4-43f5a640c220/"
              "scratchpad/opus_altlabels_analysis/basis_cache")
BANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199),
         ("mu", 1199, 1200))
ARMS = ("A", "AR", "F", "S_lo", "S_mid", "S_hi", "A_ce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923"))
    ap.add_argument("--basis", default=str(BASIS))
    ap.add_argument("--tasks", default="1,10,25,50")
    a = ap.parse_args()
    root, basis = Path(a.root), Path(a.basis)
    tasks = [int(x) for x in a.tasks.split(",")]
    rows = []
    for s in range(10):
        Q = np.load(basis / f"basis_std_s{s}.npz")["Q"].astype(np.float64)
        for x in ARMS:
            d = root / x / "snap" / f"LR_std_seed{s}"
            if not d.exists():
                continue
            for t in tasks:
                f = d / f"t{t:02d}.npz"
                if not f.exists():
                    continue
                W = np.load(f)["W1"].astype(np.float64)
                C = W @ Q
                tot = float((W ** 2).sum())
                acc = 0.0
                for name, lo, hi in BANDS:
                    n = float((C[:, lo:hi] ** 2).sum())
                    acc += n
                    rows.append({"arm": x, "seed": s, "t": t, "band": name, "N": n})
                rows.append({"arm": x, "seed": s, "t": t, "band": "comp", "N": tot - acc})
                rows.append({"arm": x, "seed": s, "t": t, "band": "all", "N": tot})
    with open(root / "bands.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "t", "band", "N"])
        w.writeheader()
        w.writerows(rows)
    names = [b[0] for b in BANDS] + ["comp", "all"]
    print("median N_b over seeds")
    for t in tasks:
        for x in ARMS:
            v = {b: [q["N"] for q in rows if q["arm"] == x and q["t"] == t and q["band"] == b]
                 for b in names}
            if not v["all"]:
                continue
            print(f"t{t:02d} {x:6s} " + " ".join(f"{b} {np.median(v[b]):9.0f}" for b in names))


if __name__ == "__main__":
    main()
