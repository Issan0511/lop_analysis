#!/usr/bin/env python3
"""How fragile a fitted network is to a change of its conv features (exploratory).

    python3 analysis/sna_cnn_cause_1009/fragility.py --bundle A --arms SNA,CV3FC06,CV06FC3 --tasks 1,10

At a checkpoint (the end of task t, fitted to that task's labels), the conv weights and biases
are moved by a random direction of relative size eps (per tensor: delta = eps * |W| * u/|u|),
everything else held, and the fit is re-read on the 1200 images: accuracy and CE on the task's
own labels.  Also with the conv moved along its real drift over the NEXT task (checkpoint t+1
minus t, scaled by s), when that checkpoint exists.  Five random directions per eps, averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_lib as L  # noqa: E402

E, CN = L.E, L.CN
ROOT = L.ROOT / "results" / "sna_cnn_cause_1009"


@torch.no_grad()
def fit_read(B) -> tuple[float, float]:
    acc, ce = 0.0, 0.0
    for i0 in range(0, CN.N_IMAGES, 200):
        lg = E.forward(B.P, B.X[:, i0:i0 + 200], B.act)[8][0]
        y = B.Y[0, i0:i0 + 200]
        acc += float((lg.argmax(1) == y).sum())
        ce += float(F.cross_entropy(lg, y, reduction="sum"))
    return acc / CN.N_IMAGES, ce / CN.N_IMAGES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--arms", default="SNA,CV3FC06,CV06FC3")
    ap.add_argument("--tasks", default="1,10")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--eps", default="0.01,0.03,0.1")
    args = ap.parse_args()
    L.H.setup("cuda")
    eps_l = [float(x) for x in args.eps.split(",")]
    g = torch.Generator(device="cpu").manual_seed(7)
    print("| arm | t | acc0 | " + " | ".join(f"acc@{e}" for e in eps_l) + " | " +
          " | ".join(f"ce@{e}" for e in eps_l) + " | drift-acc | rel drift c1 | rel drift c2 |")
    for arm in args.arms.split(","):
        for t in [int(x) for x in args.tasks.split(",")]:
            res = []
            for s in L.E.parse_seeds(args.seeds):
                f = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t:02d}.pt"
                if not f.exists():
                    continue
                B, _ = L.load_run(f)
                base = [p.detach().clone() for p in B.P[:4]]
                acc0, ce0 = fit_read(B)
                r = {"acc0": acc0, "ce0": ce0}
                for e in eps_l:
                    a_l, c_l = [], []
                    for k in range(5):
                        with torch.no_grad():
                            for i in range(4):
                                u = torch.randn(base[i].shape, generator=g).to(base[i].device)
                                B.P[i].copy_(base[i] + e * base[i].norm() * u / u.norm())
                        a, c = fit_read(B)
                        a_l.append(a); c_l.append(c)
                    r[f"acc@{e}"], r[f"ce@{e}"] = np.mean(a_l), np.mean(c_l)
                f2 = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t + 1:02d}.pt"
                if f2.exists():
                    nxt = torch.load(f2, map_location="cpu", weights_only=False)["P"]
                    with torch.no_grad():
                        for i in range(4):
                            B.P[i][0].copy_(nxt[i])
                    r["drift_acc"], _ = fit_read(B)
                    r["rel_c1"] = float((nxt[0] - base[0][0].cpu()).norm() / base[0][0].cpu().norm())
                    r["rel_c2"] = float((nxt[2] - base[2][0].cpu()).norm() / base[2][0].cpu().norm())
                res.append(r)
                del B
            if not res:
                continue
            m = {k: np.mean([q[k] for q in res if k in q]) if any(k in q for q in res) else np.nan
                 for k in res[0]}
            print(f"| {arm} | {t} | {m['acc0']:.3f} | " + " | ".join(f"{m[f'acc@{e}']:.3f}" for e in eps_l)
                  + " | " + " | ".join(f"{m[f'ce@{e}']:.2f}" for e in eps_l)
                  + f" | {m.get('drift_acc', np.nan):.3f} | {m.get('rel_c1', np.nan):.3f} | {m.get('rel_c2', np.nan):.3f} |  n={len(res)}")


if __name__ == "__main__":
    main()
