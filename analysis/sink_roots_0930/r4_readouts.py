#!/usr/bin/env python3
"""Round 4 readouts (spec_sink_roots_0930_round4.md §3).  All unit statistics are medians over the 100 units at a task's end
unless said otherwise; 'per task' changes are task end minus task head (the head of task t is the end of task t-1 except
for RR1, whose bias moves at the switch; for RR1 the end-to-end change end(t) - end(t-1) is also given).
A7: dm = sbar * d(m/s) + (m/s)bar * ds per unit (midpoint of head and end).  All-unit ratio = R3's rho_med over all units.
Components (growth f4 definitions, from the saved task heads): a = w . mu_hat, w_perp = w - a mu_hat, x_top = argmax_n z_n,
px = mu_hat . x_top, v_top = w_perp . x_top, c_perp = mean v_top / mean |w_perp| (means over units), m - b = |mu| a."""
import glob, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = ROOT / "results" / "sink_roots_0930"
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    d = RAW / name
    if not (d / "provenance.json").exists():
        return None
    a = np.load(d / "arrays.npz"); rows = json.load(open(d / "rows.json"))
    return a, rows


def task_stats(a, rows):
    g = list(a["grid"]); i0, i200 = g.index(0), g.index(200)
    m, k, top, s = a["u_m1"], a["u_k1"], a["u_top1"], a["u_s1"]
    out = []
    for ti in range(m.shape[0]):
        mh, me, sh, se = m[ti, i0], m[ti, -1], s[ti, i0], s[ti, -1]
        msh, mse = mh / sh, me / se
        a7_rel = 0.5 * (sh + se) * (mse - msh); a7_w = 0.5 * (msh + mse) * (se - sh)
        r = dict(task=ti + 1, k=float(np.median(k[ti, -1])), top_s=float(np.median(top[ti, -1] / se)), m_s=float(np.median(mse)),
                 m=float(np.median(me)), s=float(np.median(se)), kap=float(np.median((top[ti, -1] - me) / se)),
                 acc=float(a["s_acc"][ti, -1]), cov0=float(a["s_cov1_zero"][ti, -1]), allclosed=float(np.mean(k[ti, -1] == 0)),
                 online=float(rows[ti]["online_acc"]), dm=float(np.median(me - mh)), ds=float(np.median(se - sh)),
                 dms=float(np.median(mse - msh)), a7_rel=float(np.median(a7_rel)), a7_w=float(np.median(a7_w)),
                 top_alive=float(np.median(top[ti, -1][top[ti, -1] > 0])) if (top[ti, -1] > 0).any() else float("nan"))
        if ti > 0:
            r["dm_e2e"] = float(np.median(me - m[ti - 1, -1])); r["dms_e2e"] = float(np.median(mse - m[ti - 1, -1] / s[ti - 1, -1]))
        if "s_wcap_at" in a.files:
            r["wcap_at"] = float(a["s_wcap_at"][ti, -1])
        # all-unit ratio pieces (push 0->200, return 200->end)
        r["P_all"] = float(np.median(m[ti, i200] - mh)); r["R_all"] = float(np.median(me - m[ti, i200]))
        out.append(r)
    return out


def window(ts, lo, hi, key):
    v = [r[key] for r in ts if lo <= r["task"] <= hi and key in r and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def ratio_all(ts, lo, hi):
    P = [r["P_all"] for r in ts if lo <= r["task"] <= hi]; R = [r["R_all"] for r in ts if lo <= r["task"] <= hi]
    return float(-np.median(R) / np.median(P)) if P and np.median(P) != 0 else float("nan")


def rates(a, lo, hi):
    top = a["u_top1"][:, -1, :]; oc = oo = co = cc = 0
    for t in range(lo - 1, min(hi, top.shape[0] - 1)):
        x, y = top[t] > 0, top[t + 1] > 0
        oc += (x & ~y).sum(); oo += x.sum(); co += (~x & y).sum(); cc += (~x).sum()
    return oc / max(oo, 1), co / max(cc, 1)


def components(name, seed, tasks, act_ink=False):
    import sink_roots_mnist_0930 as E
    X, _, _ = E.load_mnist(seed); X = X.double().numpy()
    mu = X.mean(0); mh = mu / np.linalg.norm(mu); proj = X @ mh
    out = {}
    for t in tasks:
        f = RAW / name / f"state_before_t{t:03d}.npz"
        if not f.exists():
            continue
        st = np.load(f); W = st["P0"].astype(np.float64); b = st["P1"].astype(np.float64)
        a = W @ mh; Wp = W - a[:, None] * mh[None, :]; Z = X @ W.T + b; it = Z.argmax(0)
        vtop = np.einsum("hd,hd->h", Wp, X[it]); rp = np.linalg.norm(Wp, axis=1)
        out[t] = dict(a=float(a.mean()), m_minus_b=float((Z.mean(0) - b).mean()), vtop=float(vtop.mean()), rp=float(rp.mean()),
                      c_perp=float(vtop.mean() / rp.mean()), px=float(proj[it].mean()), r2=float((W ** 2).sum(1).mean()),
                      top_minus_b=float((Z.max(0) - b).mean()))
    return out


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:+.3f}" if abs(x) < 10 else f"{x:+.1f}"


def main():
    L, J = [], {}
    groups = sorted({re.sub(r"_s\d$", "", Path(d).name) for d in glob.glob(str(RAW / "*_s[0-9]"))
                     if re.match(r"(G1a|G1b|G1c|RR1|capL|G3tail|RR3bfloor|G2same|RR2noise|G4cap|BSD|CAPABS)_", Path(d).name)}
                    | {"R3_base_ELU", "R5_main_ELU", "R3_base_GELU", "R3_base_SILU", "R3_base_LR"})
    for gname in groups:
        runs = [(s, load(f"{gname}_s{s}")) for s in (0, 1, 2)]
        runs = [(s, x) for s, x in runs if x is not None]
        if not runs:
            continue
        L.append(f"\n## {gname} ({len(runs)} seeds)")
        per = {}
        for s, (a, rows) in runs:
            ts = task_stats(a, rows); per[s] = ts; nt = len(ts)
            wins = [(5, 30)] + [(w0, min(w1, nt)) for w0, w1 in ((25, 30), (51, 80), (55, 100), (60, 80), (20, 30), (50, 60), (151, 200), (196, 200), (26, 30)) if nt >= w0]
            row = {}
            for lo, hi in wins:
                for key in ("acc", "online", "k", "top_s", "m_s", "dm", "ds", "dms", "a7_rel", "a7_w", "allclosed", "cov0", "dm_e2e", "dms_e2e", "top_alive"):
                    row[f"{key}_{lo}_{hi}"] = window(ts, lo, hi, key)
                row[f"ratio_all_{lo}_{hi}"] = ratio_all(ts, lo, hi)
            for t in (5, 10, 12, 20, 30, 50, 55, 60, 80, 100, 200):
                if t <= nt:
                    for key in ("k", "top_s", "m_s", "acc", "cov0", "allclosed", "wcap_at"):
                        if key in ts[t - 1]:
                            row[f"{key}_t{t}"] = ts[t - 1][key]
            if "s_wcap_at" in a.files:
                w = a["s_wcap_at"][:, -1]; hit = np.where(np.nan_to_num(w) > 0.5)[0]
                row["cap_first_task_over_half"] = int(hit[0] + 1) if len(hit) else -1
            if nt >= 200:
                mp, mm = rates(a, 151, 200); row["mu_plus_151_200"], row["mu_minus_151_200"] = float(mp), float(mm)
            if a["cover"].size:                                   # inputs that no unit opens at the last task's end, and their accuracy
                cov, cor = a["cover"], a["correct_end"]
                unc = cov[-1, 1] == 0
                row[f"acc_uncovered_t{nt}"] = float(cor[-1][unc].mean()) if unc.any() else float("nan")
                row[f"n_uncovered_t{nt}"] = int(unc.sum())
            if re.match(r"(G3tail|RR3bfloor|G2same|RR1|G1|R3_base)", gname):
                row["components"] = components(f"{gname}_s{s}", s, (5, 10, 20, 30, 50, 55, 60, 80, 100))
            J[f"{gname}_s{s}"] = row
        keys = [k for k in J[f"{gname}_s{runs[0][0]}"] if k != "components"]
        for k in keys:
            vals = [J[f"{gname}_s{s}"].get(k, float("nan")) for s, _ in runs]
            L.append(f"   {k:28s} " + " ".join(fmt(float(v)) if not isinstance(v, int) else f"{v:6d}" for v in vals))
        comps = [J[f"{gname}_s{s}"].get("components") for s, _ in runs]
        if comps and comps[0]:
            for t in sorted(comps[0]):
                L.append(f"   comp t{t:3d} " + " | ".join(f"a {c[t]['a']:+.3f} m-b {c[t]['m_minus_b']:+.2f} vtop {c[t]['vtop']:+.2f} |w_perp| {c[t]['rp']:.2f} c_perp {c[t]['c_perp']:.3f} px {c[t]['px']:.2f} r2 {c[t]['r2']:.1f}"
                                                  for c in comps if c and t in c))
    txt = "\n".join(L)
    print(txt)
    (RES / "round4_readouts.txt").write_text(txt + "\n")
    (RES / "round4_readouts.json").write_text(json.dumps(J, indent=1, default=float))


if __name__ == "__main__":
    main()
