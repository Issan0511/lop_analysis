#!/usr/bin/env python3
"""Round 2 §2 tau_eff_vs_fit_speed (spec_sink_roots_0930_round2.md): per group (act, K, T) the error-minimising tau of the
memory filter (plain kernel; error = state-mean |predicted mean P - observed S^oh>0|, tau*lambda1 grid 1e-4..1e6, 41 points),
its ratio to eta*t_fit/N and to eta*T/N, and the three registered judgements (a) (b) (c)."""
import glob, json, re
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round2/tau/filter")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
FACS = np.logspace(-4, 6, 41)
ETA, N = 1e-3, 1200
TAG = "plain"


def load():
    G = {}
    for f in sorted(glob.glob(str(RAW / "ftau_*.npy"))):
        m = re.match(r"ftau_(\w+?)_K(\d+)_T(\d+)", Path(f).stem)
        G[(m.group(1), int(m.group(2)), int(m.group(3)))] = list(np.load(f, allow_pickle=True))
    return G


def meanP_at(r, tau):
    """predicted mean P at an arbitrary tau, interpolated on the grid in log(tau*lambda1)"""
    l1 = r[TAG]["l1N"] * N
    x = np.log10(np.clip(tau * l1, FACS[0], FACS[-1]))
    return float(np.interp(x, np.log10(FACS), [q["meanP"] for q in r[TAG]["rows"][:41]]))


def group(rs):
    err = np.array([np.mean([abs(r[TAG]["rows"][j]["meanP"] - r["Soh_pos"]) for r in rs]) for j in range(41)])
    j = int(np.argmin(err)); l1 = np.median([r[TAG]["l1N"] * N for r in rs])
    tf = np.array([r["tfit"] for r in rs]); tf90 = np.array([r["tfit90"] for r in rs])
    T = rs[0]["T"]
    out = dict(n=len(rs), obs_oh=float(np.mean([r["Soh_pos"] for r in rs])), obs_S=float(np.mean([r["S_pos"] for r in rs])),
               taul1_best=float(FACS[j]), err_best=float(err[j]), tau_best=float(FACS[j] / l1), lambda1=float(l1), edge=bool(j in (0, 40)),
               err_nat=float(np.mean([abs(r[TAG]["rows"][41]["meanP"] - r["Soh_pos"]) for r in rs])),
               err_natK=float(np.mean([abs(r[TAG]["rows"][42]["meanP"] - r["Soh_pos"]) for r in rs])),
               tfit_defined=float(np.mean(tf > 0)), tfit_med=float(np.median(tf[tf > 0])) if (tf > 0).any() else float("nan"),
               tfit90_med=float(np.median(tf90)), acc_old=float(np.mean([r["acc_end_old"] for r in rs])), T=T)
    out["r"] = out["tau_best"] / (ETA * out["tfit_med"] / N) if out["tfit_defined"] >= 0.5 else float("nan")
    out["r90"] = out["tau_best"] / (ETA * out["tfit90_med"] / N)
    out["rT"] = out["tau_best"] / (ETA * T / N)
    return out


def main():
    G = load()
    S = {k: group(v) for k, v in G.items()}
    lines = ["act K T | n | obs S^oh>0 (S>0) | best tau*l1 (err) | err at eta T/N, /K | tau_best | t_fit med (defined) | t_fit90 med | "
             "r = tau_best/(eta t_fit/N) | r90 | rT = tau_best/(eta T/N) | old acc"]
    for k in sorted(S):
        s = S[k]
        lines.append(f"{k[0]:4s} K{k[1]:2d} T{k[2]:5d} | {s['n']:2d} | {s['obs_oh']:.2f} ({s['obs_S']:.2f}) | {s['taul1_best']:.1e} ({s['err_best']:.3f}) | "
                     f"{s['err_nat']:.3f}, {s['err_natK']:.3f} | {s['tau_best']:.2e} | {s['tfit_med']:.0f} ({s['tfit_defined']:.2f}) | {s['tfit90_med']:.0f} | "
                     f"{s['r']:.3g} | {s['r90']:.3g} | {s['rT']:.3g} | {s['acc_old']:.2f}" + (" | EDGE (argmin at the end of the grid)" if s['edge'] else ""))
    # (a) spread of r across groups with t_fit defined, against the spread of rT over the same groups
    ok = [k for k in S if np.isfinite(S[k]["r"])]
    r = np.array([S[k]["r"] for k in ok]); rT = np.array([S[k]["rT"] for k in ok])
    spr, sprT = (float(r.max() / r.min()), float(rT.max() / rT.min())) if len(ok) >= 2 else (float("nan"),) * 2
    a_lab = "SUPPORTED" if (spr <= 3 and spr < sprT) else "NOT_SUPPORTED"
    lines.append(f"\n(a) groups with t_fit defined: {len(ok)} of {len(S)}; spread max/min of r {spr:.2f} vs of rT {sprT:.2f} -> {a_lab}"
                 f"  (grid step x1.78; r range {r.min():.3g}..{r.max():.3g})" if len(ok) >= 2 else "\n(a) fewer than 2 groups with t_fit defined")
    okn = [k for k in ok if not S[k]["edge"]]
    if len(okn) >= 2 and len(okn) < len(ok):
        rn = np.array([S[k]["r"] for k in okn]); rTn = np.array([S[k]["rT"] for k in okn])
        lines.append(f"    without groups whose argmin is at the grid's end: {len(okn)} groups, spread of r {rn.max() / rn.min():.2f} (rT {rTn.max() / rTn.min():.2f})")
    ok4 = [k for k in ok if k[2] == 4000]
    if len(ok4) >= 2:
        r4 = np.array([S[k]["r"] for k in ok4]); rT4 = np.array([S[k]["rT"] for k in ok4])
        lines.append(f"    T 4000 only: {len(ok4)} groups, spread of r {r4.max() / r4.min():.2f} (rT {rT4.max() / rT4.min():.2f})")
    # (b) K = 2 with tau = c * eta t_fit / N, c = median r over the K = 10 groups
    c = float(np.median([S[k]["r"] for k in ok if k[1] == 10])) if any(k[1] == 10 for k in ok) else float("nan")
    b_rows = []; err_all = []; err_natK = []
    for k, rs in sorted(G.items()):
        if k[1] != 2:
            continue
        e = [meanP_at(q, c * ETA * q["tfit"] / N) - q["Soh_pos"] for q in rs if q["tfit"] > 0 and np.isfinite(c)]
        en = [q[TAG]["rows"][42]["meanP"] - q["Soh_pos"] for q in rs]
        err_all += e; err_natK += en
        b_rows.append(f"    {k[0]:4s} T{k[2]:5d}: n {len(e)}/{len(rs)} mean signed error {np.mean(e) if e else float('nan'):+.3f} (eta T/(NK): {np.mean(en):+.3f})")
    b_lab = ("SUPPORTED" if abs(np.mean(err_all)) <= 0.08 else "NOT_SUPPORTED") if err_all else "UNDEFINED"
    lines.append(f"\n(b) c = median r over K10 groups = {c:.3g}; K = 2 states with t_fit defined: mean signed error {np.mean(err_all) if err_all else float('nan'):+.3f}"
                 f" (the natural eta T/(NK): {np.mean(err_natK):+.3f}) -> {b_lab}")
    lines += b_rows
    # (c) tau_best(T 16000) / tau_best(T 4000)
    c_rows = []; npass = 0; npair = 0
    for act in ("ELU", "LR", "GELU"):
        for K in (2, 10):
            a4, a16 = S.get((act, K, 4000)), S.get((act, K, 16000))
            if a4 and a16:
                q = a16["tau_best"] / a4["tau_best"]; npair += 1; npass += q <= 1.5
                c_rows.append(f"    {act:4s} K{K:2d}: tau_best 16000/4000 = {q:.2f}")
    lines.append(f"\n(c) pairs with ratio <= 1.5: {npass} of {npair} -> {'SUPPORTED' if npair and npass == npair else 'NOT_SUPPORTED'}")
    lines += c_rows
    txt = "\n".join(lines)
    print(txt)
    (RES / "round2_tau.txt").write_text(txt + "\n")
    (RES / "round2_tau.json").write_text(json.dumps({f"{k[0]}_K{k[1]}_T{k[2]}": v for k, v in S.items()}, indent=1))


if __name__ == "__main__":
    main()
