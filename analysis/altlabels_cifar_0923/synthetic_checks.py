#!/usr/bin/env python3
"""synthetic_checks.py -- altlabels_cifar_0923 §6 S7, the analysis side.

Six checks on made-up weight sequences whose answers are known in closed form.  They drive
ledger.py's OWN functions (slot_arrays / rows_cycle / rows_lags, with the snapshot loader
replaced by a callable), so a bug in the band algebra or in the window bookkeeping fails
here and not only on real data.

  C1  fixed A/B endpoints      -> same-label displacement exactly 0, r_{6:50} = 1/45,
                                  lag-1 cosine -1, lag-2 cosine +1, centre never moves
  C2  growing two-cycle        -> W_t = R_t u_{A/B}, R_t proportional to t^0.5: r stays
                                  small while the amplitude grows (a small r does NOT mean
                                  a bounded walk)
  C3  zero increments          -> every cosine is undefined (NaN), never 0 or 1, and r is
                                  reported with a non-positive-denominator status
  C4  identity on random data  -> dN = d + e to 1e-6 relative (S7's registered tolerance)
  C5  band sum                 -> sum over top/mid1/mid2/low/mu/comp = all, every t
  C6  variance split           -> V over the 4 eigen bands = the total variance, and
                                  V_mu = V_comp = 0 exactly

Writes results/altlabels_cifar_0923/synthetic_checks.json.  No training, no real data:
the basis is a random orthonormal 3072x1200 frame, which is all the algebra sees.

Usage: synthetic_checks.py [--out DIR] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NT = str(min(8, int(os.environ.get("ALTLAB_THREADS", "8"))))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, NT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np                                                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger as LG                                                      # noqa: E402

NDIM, NSPAN, NPC = LG.NDIM, LG.NSPAN, LG.NPC
NUNIT = 100
TMAX = 50
BANDS = LG.BANDNAMES


def fake_arm(tasks=TMAX, cycle=2) -> LG.Arm:
    return LG.Arm(name="SYN", act="LR", cond="std", tasks=tasks, cycle=cycle,
                  layout="run", root=Path("/nonexistent"), seeds=(0,))


def random_basis(rng) -> tuple:
    """Q (3072, 1200) orthonormal and 1199 positive eigenvalues, decreasing."""
    A = rng.standard_normal((NDIM, NSPAN))
    Q, _ = np.linalg.qr(A)
    lam = np.sort(rng.random(NPC) * 10.0 + 0.01)[::-1].copy()
    return Q, lam


def run_seq(Ws: dict, Q, lam, tasks=TMAX, cycle=2):
    arm = fake_arm(tasks, cycle)
    S = LG.slot_arrays(arm, 0, Q, lam, W_of=lambda t: Ws.get(t))
    cy = LG.rows_cycle(arm, 0, S)
    lg = LG.rows_lags(arm, 0, S)
    return arm, S, cy, lg


def band_of(rows, band, **kw):
    for r in rows:
        if r["band"] == band and all(r[k] == v for k, v in kw.items()):
            return r
    return None


def nearly(a, b, tol):
    return bool(np.isfinite(a) and abs(a - b) <= tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LG.OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    Q, lam = random_basis(rng)
    res = {}

    # ---------------------------------------------------------------- C1
    WA = rng.standard_normal((NUNIT, NDIM))
    WB = WA + rng.standard_normal((NUNIT, NDIM)) * 0.5
    Ws = {t: (WA if t % 2 == 1 else WB) for t in range(TMAX + 1)}
    arm, S, cy, lg = run_seq(Ws, Q, lam)
    disp = {}
    for b in BANDS:
        rows = [r for r in cy if r["band"] == b]
        disp[b] = {
            "max_same_label_return": float(np.nanmax([max(r["ret_odd"], r["ret_even"])
                                                      for r in rows if r["j"] >= 2])),
            "max_center_disp": float(np.nanmax([r["center_disp"] for r in rows if r["j"] >= 2])),
            "r_6_50": band_of(lg, b, window="all")["r_b"],
            "C1_cos": band_of(lg, b, window="all")["C1_cos"],
            "C2_cos": band_of(lg, b, window="all")["C2_cos"],
            "max_E_identity_residual": float(np.nanmax(np.abs([r["E_identity_residual"]
                                                               for r in rows]))),
        }
    scale = float((WA ** 2).sum())
    res["C1_fixed_AB_endpoints"] = {
        "what": "W_t alternates between two fixed endpoints; 45 increments over t = 6..50",
        "expect": {"same_label_return": 0.0, "center_disp": 0.0, "r_6_50": 1 / 45,
                   "C1_cos": -1.0, "C2_cos": +1.0},
        "per_band": disp,
        "pass": bool(
            all(d["max_same_label_return"] <= 1e-12 * scale for d in disp.values())
            and all(d["max_center_disp"] <= 1e-12 * scale for d in disp.values())
            and all(nearly(d["r_6_50"], 1 / 45, 1e-10) for d in disp.values())
            and all(nearly(d["C1_cos"], -1.0, 1e-10) for d in disp.values())
            and all(nearly(d["C2_cos"], +1.0, 1e-10) for d in disp.values())
            and all(d["max_E_identity_residual"] <= 1e-9 * scale for d in disp.values())),
        "r_6_50_exact": 1 / 45,
    }

    # ---------------------------------------------------------------- C2
    uA = rng.standard_normal((NUNIT, NDIM))
    uA /= np.linalg.norm(uA)
    uB = rng.standard_normal((NUNIT, NDIM))
    uB /= np.linalg.norm(uB)
    R = lambda t: 100.0 * math_sqrt(max(t, 1))                            # noqa: E731
    Ws = {t: R(t) * (uA if t % 2 == 1 else uB) for t in range(TMAX + 1)}
    arm, S2, cy2, lg2 = run_seq(Ws, Q, lam)
    ka = BANDS.index("all")
    g2 = {b: {"r_6_50": band_of(lg2, b, window="all")["r_b"],
              "C1_cos": band_of(lg2, b, window="all")["C1_cos"],
              "C2_cos": band_of(lg2, b, window="all")["C2_cos"]} for b in BANDS}
    amp = float(S2["N"][50, ka] / S2["N"][25, ka])
    res["C2_growing_two_cycle"] = {
        "what": "W_t = R_t u_{A/B} with R_t ~ t^0.5: a growing alternation",
        "expect": "r small (the walk keeps cancelling) while N(50)/N(25) = 2 exactly",
        "N50_over_N25": amp, "N50_over_N25_exact": 2.0,
        "sigma_free_note": "amplitude grows by sqrt(2) in norm; r stays near the 1/45 floor",
        "per_band": g2,
        "pass": bool(nearly(amp, 2.0, 1e-9)
                     and all(0 < g2[b]["r_6_50"] < 0.15 for b in BANDS)
                     and all(g2[b]["C1_cos"] < -0.9 for b in BANDS)),
    }

    # ---------------------------------------------------------------- C3
    W0 = rng.standard_normal((NUNIT, NDIM))
    Ws = {t: W0.copy() for t in range(TMAX + 1)}
    arm, S3, cy3, lg3 = run_seq(Ws, Q, lam)
    z = {b: {"C1_cos": band_of(lg3, b, window="all")["C1_cos"],
             "C1_energy": band_of(lg3, b, window="all")["C1_energy"],
             "r_b": band_of(lg3, b, window="all")["r_b"],
             "r_status": band_of(lg3, b, window="all")["r_status"],
             "n_cos_1": band_of(lg3, b, window="all")["n_cos_1"]} for b in BANDS}
    res["C3_zero_increments"] = {
        "what": "W never moves: every increment is exactly zero",
        "expect": "cosine undefined (NaN, not 0 and not 1); r has a non-positive denominator",
        "per_band": z,
        "pass": bool(all(not np.isfinite(z[b]["C1_cos"]) for b in BANDS)
                     and all(z[b]["n_cos_1"] == 0 for b in BANDS)
                     and all(not np.isfinite(z[b]["r_b"]) for b in BANDS)
                     and all(z[b]["r_status"] == "denom_nonpositive" for b in BANDS)),
    }

    # ---------------------------------------------------------------- C4/C5/C6
    Ws = {t: rng.standard_normal((NUNIT, NDIM)) * (1.0 + 0.1 * t) for t in range(TMAX + 1)}
    arm, S4, cy4, lg4 = run_seq(Ws, Q, lam)
    N, dd, ee, rr, V = S4["N"], S4["d"], S4["e"], S4["res"], S4["V"]
    rel = np.abs(rr[1:]) / np.maximum(np.abs(N[1:]), 1e-300)
    res["C4_identity"] = {
        "what": "dN_b = d_b + e_b on 51 independent random W (d and e come from an "
                "independent projection of dW, so this tests the projection too)",
        "max_abs_residual": float(np.nanmax(np.abs(rr))),
        "max_rel_residual": float(np.nanmax(rel)),
        "tolerance": LG.IDENTITY_RTOL,
        "pass": bool(np.nanmax(rel) <= LG.IDENTITY_RTOL),
    }
    part = [BANDS.index(b) for b in ("top", "mid1", "mid2", "low", "mu", "comp")]
    ssum = N[:, part].sum(1) - N[:, ka]
    res["C5_band_sum"] = {
        "what": "top + mid1 + mid2 + low + mu + comp = all, at every t",
        "max_abs_err": float(np.nanmax(np.abs(ssum))),
        "max_rel_err": float(np.nanmax(np.abs(ssum) / np.maximum(N[:, ka], 1e-300))),
        "pass": bool(np.nanmax(np.abs(ssum) / np.maximum(N[:, ka], 1e-300)) <= 1e-12),
    }
    eig = [BANDS.index(b) for b in LG.EIGNAMES]
    vsum = V[:, eig].sum(1) - V[:, ka]
    res["C6_variance_split"] = {
        "what": "V over the 4 eigen bands = the total pre-activation variance; mu_perp and "
                "the complement carry none",
        "max_abs_err": float(np.nanmax(np.abs(vsum))),
        "max_rel_err": float(np.nanmax(np.abs(vsum) / np.maximum(V[:, ka], 1e-300))),
        "V_mu_max": float(np.nanmax(np.abs(V[:, BANDS.index("mu")]))),
        "V_comp_max": float(np.nanmax(np.abs(V[:, BANDS.index("comp")]))),
        "pass": bool(np.nanmax(np.abs(vsum) / np.maximum(V[:, ka], 1e-300)) <= 1e-12
                     and np.nanmax(np.abs(V[:, BANDS.index("mu")])) == 0.0
                     and np.nanmax(np.abs(V[:, BANDS.index("comp")])) == 0.0),
    }

    # ---------------------------------------------------------------- C7 (windows)
    # the lag pairing must use only increments with BOTH indices inside the window
    row = band_of(lg4, "all", window="late")
    n_inc = row["n_increments"]
    ok_pairs = all(row[f"n_pairs_{k}"] == max(n_inc - k, 0) for k in range(1, 7))
    res["C7_window_pairing"] = {
        "what": "late = t26-50: 25 increments, so lag k has 25 - k pairs, both inside",
        "n_increments": n_inc,
        "n_pairs": {f"k{k}": row[f"n_pairs_{k}"] for k in range(1, 7)},
        "pass": bool(n_inc == 25 and ok_pairs),
    }

    res["_meta"] = {
        "run_id": LG.EXPERIMENT, "stage": "synthetic_checks",
        "written": time.strftime("%FT%T%z"), **LG.git_state(),
        "basis": "random orthonormal 3072x1200 (numpy default_rng seed "
                 f"{a.seed}), 1199 random decreasing eigenvalues",
        "n_units": NUNIT, "tmax": TMAX, "numpy": np.__version__,
        "wall_clock_s": time.time() - t0,
    }
    res["_meta"]["all_pass"] = bool(all(v.get("pass") for k, v in res.items()
                                        if k != "_meta"))
    p = out / "synthetic_checks.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        if k != "_meta":
            print(f"{'PASS' if v.get('pass') else 'FAIL'}  {k}")
    print(f"all_pass = {res['_meta']['all_pass']}")
    print(f"wrote {p}  ({time.time()-t0:.1f}s)")
    if not res["_meta"]["all_pass"]:
        sys.exit(1)


def math_sqrt(x):
    return float(np.sqrt(x))


if __name__ == "__main__":
    main()
