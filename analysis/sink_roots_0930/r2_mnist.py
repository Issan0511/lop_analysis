#!/usr/bin/env python3
"""Round 2 engine arms (spec_sink_roots_0930_round2.md): §1 R3prime_bwrelu and §9 R3_readout_scale_clamp.
§1: the registered rho_med and label come from analyze_r3.py (rerun with bwrelu in ARMS).  Here: survivors (open at the
    switch and at the task end, tasks 5-30) ratio of means -sum R / sum P and f_top = -sum dtop_R / sum dtop_P
    (push 0 -> 200 updates, return 200 -> task end), per seed, for base / bwfloor / bwabs / bwrelu.
§9: end-of-task top V = max(top, 0); OLS slope of V' on V over units alive at t (windows 51-100, 101-150, and 51-150);
    the tightest upper line (top_affine.py (b)); top median and closing probability by the age of the alive episode
    (age_top.py, t >= 51, age 12+ pooled); mu+ / mu-.  The clamp itself was checked on a 2-task run with checkpoints
    (GELU seed 0, clamp from task 2: every unit's |v^c| = 1.8000 after task 2; the natural median after task 1 is 0.96).
    Operationalisation written before the runs finished: "flat with age" = median top at age 12+ / age 1 within [0.8, 1.25]."""
import json
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def survivors(name):
    a = np.load(RAW / name / "arrays.npz"); g = list(a["grid"]); i0, i200, iT = g.index(0), g.index(200), len(g) - 1
    m, top, k = a["u_m1"], a["u_top1"], a["u_k1"]
    P = R = TP = TR = 0.0
    for t in range(5, 31):
        ti = t - 1; sv = (k[ti, i0] > 0) & (k[ti, iT] > 0)
        P += (m[ti, i200] - m[ti, i0])[sv].sum(); R += (m[ti, iT] - m[ti, i200])[sv].sum()
        TP += (top[ti, i200] - top[ti, i0])[sv].sum(); TR += (top[ti, iT] - top[ti, i200])[sv].sum()
    return dict(ratio_surv=float(-R / P), f_top=float(-TR / TP))


def bwrelu():
    L = ["## §1 R3prime_bwrelu: survivors, tasks 5-30 (ratio of sums; f_top = -sum dtop_R / sum dtop_P)"]
    J = {}
    for act in ("GELU", "SILU"):
        for arm in ("base", "bwfloor", "bwabs", "bwrelu"):
            rows = []
            for s in (0, 1, 2):
                n = f"R3_{arm}_{act}_s{s}"
                if (RAW / n / "provenance.json").exists():
                    J[n] = survivors(n); rows.append(J[n])
            if rows:
                L.append(f"   {act:4s} {arm:8s}: ratio(surv) {' '.join(f'{r['ratio_surv']:.3f}' for r in rows)} | f_top {' '.join(f'{r['f_top']:.3f}' for r in rows)}")
    return L, J


def affine(V0, V1):
    xs, ys = V0.ravel(), V1.ravel()
    al = xs > 0
    slope = float(np.polyfit(xs[al], ys[al], 1)[0]) if al.sum() > 10 else float("nan")
    bins = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 1e9)]
    pts = []
    for b in bins:
        msk = (xs == 0) if b == (-1, 0) else ((xs > b[0]) & (xs <= b[1]))
        if msk.sum() >= 30:
            pts.append((xs[msk].mean(), ys[msk].mean()))
    P = np.array(pts); best = None
    for rho in np.linspace(0, 0.99, 991):
        a = max(0.0, float(np.max(P[:, 1] - rho * P[:, 0]))); bnd = a / (1 - rho)
        if best is None or bnd < best[0]:
            best = (bnd, a, rho)
    return slope, best


def roclamp():
    L = ["\n## §9 R3_readout_scale_clamp (GELU, clamp of |v^c| from task 51)"]
    J = {}
    for arm in ("none", "r1.8", "r14"):
        tops = []
        for s in (0, 1):
            n = f"R3ro_{arm}_GELU_s{s}"
            if (RAW / n / "provenance.json").exists():
                a = np.load(RAW / n / "arrays.npz"); tops.append((a["u_top1"][:, -1, :].astype(float), a))
        if not tops:
            continue
        row = {}
        for lo, hi in ((51, 100), (101, 150), (51, 150)):
            V0 = np.concatenate([np.maximum(t[lo - 1:hi - 1], 0) for t, _ in tops]); V1 = np.concatenate([np.maximum(t[lo:hi], 0) for t, _ in tops])
            sl, best = affine(V0, V1)
            row[f"slope_{lo}_{hi}"] = sl; row[f"rho_{lo}_{hi}"] = best[2]; row[f"a_{lo}_{hi}"] = best[1]
        by_age, close_by_age = {}, {}
        for top, _ in tops:
            al = top > 0
            for u in range(top.shape[1]):
                age = 0
                for t in range(top.shape[0]):
                    if al[t, u]:
                        age = age + 1 if (t > 0 and al[t - 1, u]) else 1
                        if t >= 50:
                            by_age.setdefault(min(age, 12), []).append(top[t, u])
                            if t + 1 < top.shape[0]:
                                close_by_age.setdefault(min(age, 12), []).append(0 if al[t + 1, u] else 1)
                    else:
                        age = 0
        med_age = {a_: float(np.median(v)) for a_, v in sorted(by_age.items())}
        row["top_by_age"] = med_age; row["n_by_age"] = {a_: len(v) for a_, v in sorted(by_age.items())}
        row["close_by_age"] = {a_: float(np.mean(v)) for a_, v in sorted(close_by_age.items())}
        row["age12_over_age1"] = med_age.get(12, float("nan")) / med_age.get(1, float("nan")) if 1 in med_age else float("nan")
        mp = mm = 0.0; oc = oo = co = cc = 0
        for top, _ in tops:
            for t in range(50, top.shape[0] - 1):
                x, y = top[t] > 0, top[t + 1] > 0
                oc += (x & ~y).sum(); oo += x.sum(); co += (~x & y).sum(); cc += (~x).sum()
        row["mu_plus"] = oc / max(oo, 1); row["mu_minus"] = co / max(cc, 1)
        vc = []   # the arrays hold no readout weights (u_ro_abs is a per-input readout statistic, not |v^c|); the clamp is checked separately
        J[arm] = row
        L.append(f"   {arm:5s} ({len(tops)} seeds): OLS slope 51-100 {row['slope_51_100']:.2f} 101-150 {row['slope_101_150']:.2f} 51-150 {row['slope_51_150']:.2f}"
                 f" | upper-line rho {row['rho_51_150']:.2f} a {row['a_51_150']:.2f} | top at age 12+/age 1 {row['age12_over_age1']:.2f}"
                 f" | mu+ {row['mu_plus']:.3f} mu- {row['mu_minus']:.3f}")
        L.append("        top by age: " + " ".join(f"{a_}:{v:.2f}(n{row['n_by_age'][a_]})" for a_, v in med_age.items()))
        L.append("        P(close next) by age: " + " ".join(f"{a_}:{v:.3f}" for a_, v in row["close_by_age"].items()))
    for arm, lab in (("r1.8", "slope >= 0.5 and age12+/age1 >= 2"), ("r14", "slope <= 0.3 and age12+/age1 in [0.8,1.25]")):
        if arm in J:
            r = J[arm]
            ok = (r["slope_51_150"] >= 0.5 and r["age12_over_age1"] >= 2) if arm == "r1.8" else (r["slope_51_150"] <= 0.3 and 0.8 <= r["age12_over_age1"] <= 1.25)
            L.append(f"   prediction {arm}: {lab} -> {'HIT' if ok else 'MISS'}")
    return L, J


if __name__ == "__main__":
    out = {}; lines = []
    for fn in (bwrelu, roclamp):
        L, J = fn(); lines += L; out[fn.__name__] = J
    txt = "\n".join(lines)
    print(txt)
    (RES / "round2_mnist.txt").write_text(txt + "\n")
    (RES / "round2_mnist.json").write_text(json.dumps(out, indent=1, default=str))
