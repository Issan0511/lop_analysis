#!/usr/bin/env python3
"""adamw_dose2_1011 A1: the registered depth rule (spec section 2.3), as arithmetic.

    python -m analysis.adamw_dose2_1011.rule            # prints the registered table and the line check
    python -m analysis.adamw_dose2_1011.rule --bands    # recomputes the spread bands from the cap_cifar_ee_0920 backup

Rule (the parent's, applied to the RL-CIFAR x ELU/std box):
    ||w_l||*  = eta * sqrt(d_l / (2 (1 - c)))                 rotational equilibrium, c = fl32(1 - eta*lambda)
    ||w_l||   = kappa_l * ||w_l||*,  kappa_l in [1.6, 2.3]    P180's measured correction
    s1        = rho1 * ||w1||                                 layer-1 projection spread rho1 = s1 / ||w1||
    s2        = rho2 * ||w2||,  rho2 = f * s1                 layer 2's projection spread grows with layer 1's width
    m         = r * s2,  r in [-1.95, -1.85]                  relative position of the box's ref (V12 table 12)
    Z         = Phi(z0 / s2 - r)                              Gaussian fraction of layer-2 (image, unit) pairs whose
                                                              float32 training derivative is exactly 0 (z < z0)
with z0 = ln 2^-150 = -103.972 (native F.elu autograd: exp(z) underflows) and the forward-saturation line
zf = -16.982 (CUDA F.elu returns exactly -1 below it).  Corners: shallow (K_min, r = -1.85), centre (geometric
centre of K, r = -1.90), deep (K_max, r = -1.95), K = kappa1 * kappa2 * rho1 * f.
Class: SAVES if Z(deep) < 0.1; FAILS if Z(shallow) >= 0.1; BORDERLINE otherwise.  Binary call: SAVES iff Z(centre) < 0.1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ETA = 1e-3
D1, D2 = 3072, 100
Z0 = math.log(2.0 ** -150)          # -103.972: exp(z) rounds to 0 in float32 (no flush; cap_cifar_ee environment)
ZF_CUDA = -16.982107162475586       # largest float32 z with F.elu(z) == -1 on this GPU (bisection, spec 1.1)
ZF_CPU = math.log(2.0 ** -24)       # -16.636: the MNIST host's line (output+1 form, P180)
KAPPA = (1.6, 2.3)                  # P180: row norm / rotational equilibrium, per layer
R_BAND = (-1.85, -1.95)             # V12 table 12: ELU std layer 2 of the ref after the first decline
R_CENTRE = -1.90
# rho1 = s1/||w1|| and f = rho2/s1 (unit medians, 10-seed means) over the known states in which both layers move
# together: ref t1-t50, cap12 t2-t50, cap12_bfix t2-t50 of cap_cifar_ee_0920 (recomputed by --bands)
RHO1 = (4.4792, 9.6086)
F_BAND = (0.6667, 1.1602)
LAMBDAS = {"wd3e-2": 0.03, "wd1e-1": 0.1, "wd1e0": 1.0}
WD_TASKS = 49                        # tasks 2-50 carry the decay
STEPS = 30000


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def c32(lam: float) -> float:
    return float(np.float32(1 - ETA * lam))


def w_star(lam: float, d: int) -> float:
    return ETA * math.sqrt(d / (2.0 * (1.0 - c32(lam))))


def k_corners() -> dict:
    kmin = KAPPA[0] ** 2 * RHO1[0] * F_BAND[0]
    kmax = KAPPA[1] ** 2 * RHO1[1] * F_BAND[1]
    return {"shallow": (kmin, R_BAND[0]), "centre": (math.sqrt(kmin * kmax), R_CENTRE), "deep": (kmax, R_BAND[1])}


def predict(lam: float) -> dict:
    base = w_star(lam, D1) * w_star(lam, D2)
    out = {"lam": lam, "c_hex": c32(lam).hex(), "one_minus_c": 1 - c32(lam), "tau_updates": 1 / (1 - c32(lam)),
           "tau_over_run": (1 / (1 - c32(lam))) / (WD_TASKS * STEPS), "w1_star": w_star(lam, D1),
           "w2_star": w_star(lam, D2)}
    for name, (K, r) in k_corners().items():
        s2 = base * K
        out[name] = {"K": K, "r": r, "s2": s2, "m": r * s2, "Z": Phi(Z0 / s2 - r), "F": Phi(ZF_CUDA / s2 - r)}
    zd, zs, zc = out["deep"]["Z"], out["shallow"]["Z"], out["centre"]["Z"]
    out["class"] = "SAVES" if zd < 0.1 else ("FAILS" if zs >= 0.1 else "BORDERLINE")
    out["call"] = "SAVES" if zc < 0.1 else "FAILS"
    return out


def table() -> list[dict]:
    return [dict(arm=a, **predict(l)) for a, l in LAMBDAS.items()]


def backup_root() -> Path:
    return Path("/home/issan/Projects/obsidian-research-data/cap_cifar_ee_0920/results/cap_cifar_ee_0920")


def load_units(arm: str, t: int) -> dict:
    root = backup_root()
    p = root / "prefix/units.npz" if t == 1 else root / f"arms/{arm}/units_t{t:02d}.npz"
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def bands() -> dict:
    """rho1 and f over the registered state set (unit medians, 10-seed means)."""
    rho1, f = [], []
    states = [("ref", t) for t in range(1, 51)] + [(a, t) for a in ("cap12", "cap12_bfix") for t in range(2, 51)]
    for arm, t in states:
        u = load_units(arm, t)
        r1 = np.median(u["zsd_l1"] / u["Wnorm_l1"], axis=1)
        s1 = np.median(u["zsd_l1"], axis=1)
        r2 = np.median(u["zsd_l2"] / u["Wnorm_l2"], axis=1)
        rho1.append(float(r1.mean()))
        f.append(float((r2 / s1).mean()))
    return {"rho1": (min(rho1), max(rho1)), "f": (min(f), max(f)), "n_states": len(states)}


def line_check() -> list[dict]:
    """Gaussian exact-zero fraction Phi(z/s2 - r) against the measured one, for z0 and for the MNIST line."""
    rows = []
    for arm, ts in (("ref", (2, 3, 4, 5, 10, 50)), ("cap1", (3, 5, 10, 50)), ("cap2", (3, 5, 10, 50)),
                    ("cap12", (2, 10, 50))):
        for t in ts:
            u = load_units(arm, t)
            s2 = np.median(u["zsd_l2"], axis=1)
            r = np.median(u["zmean_over_sd_l2"], axis=1)
            rows.append({"arm": arm, "task": t, "s2": float(s2.mean()), "r": float(r.mean()),
                         "Z_meas": float(u["zero_l2"].mean()),
                         "Z_pred_z0": float(np.mean([Phi(Z0 / a - b) for a, b in zip(s2, r)])),
                         "Z_pred_mnist_line": float(np.mean([Phi(ZF_CPU / a - b) for a, b in zip(s2, r)]))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.bands:
        b = bands()
        print(json.dumps(b))
        assert abs(b["rho1"][0] - RHO1[0]) < 5e-5 and abs(b["rho1"][1] - RHO1[1]) < 5e-5, b
        assert abs(b["f"][0] - F_BAND[0]) < 5e-5 and abs(b["f"][1] - F_BAND[1]) < 5e-5, b
        for r in line_check():
            print(json.dumps(r))
    t = table()
    for row in t:
        print(f"{row['arm']:7s} lam {row['lam']:<5g} tau {row['tau_updates']:8.0f} ({row['tau_over_run']:.4f} run) "
              f"w1* {row['w1_star']:.3f} w2* {row['w2_star']:.4f} | "
              + " | ".join(f"{k}: s2 {row[k]['s2']:.1f} m {row[k]['m']:.1f} Z {row[k]['Z']:.3f} F {row[k]['F']:.2f}"
                           for k in ("shallow", "centre", "deep"))
              + f" | {row['class']} call {row['call']}")
    if a.json:
        Path(a.json).write_text(json.dumps(t, indent=2))


if __name__ == "__main__":
    main()
