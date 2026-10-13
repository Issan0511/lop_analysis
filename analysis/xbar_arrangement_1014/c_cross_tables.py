"""C (xbar_arrangement_1014): cross-state tables from c_summary.json (+ c_strat_sensitivity_*.json if present) -> results/.../c_cross_tables.md"""
import os, json, glob
import numpy as np
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
S = json.load(open(os.path.join(RES, "c_summary.json"))); st = [k for k in ("a01_s0", "a03_s0", "a05_s0", "a01_s1", "a03_s1", "a05_s1") if k in S]; L = []
L.append("| state | n | MATCH (log lines) | max rel diff vs P149 json | P149 ratio medians full / probe |"); L.append("|---|---|---|---|---|")
for k in st:
    x = S[k]; L.append(f"| {k} | {x['n']} | {x['match_lines']} {'MATCH' if x['match'] else 'MISMATCH'} | {max(x['rel_vs_p149'].values()):.1e} | {x['p149_ratio_medians']['full'][0]:.2f} / {x['p149_ratio_medians']['probe'][0]:.2f} |")
L.append("")
for nm in ("full", "probe", "readout"):
    L.append(f"**{nm}**\n"); L.append("| state | actual/E_full med [min,max] | actual/E_strat med [min,max] | z_full med [10,90] | z_strat med [10,90] | gap closed med [10,90] | E_strat>0 | E_strat<E_full | actual<E_strat | actual>E_strat |"); L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in st:
        c = S[k]["C"][nm]; n = S[k]["n"]; g = c["gap_closed"]
        L.append(f"| {k} | {c['ratio_full'][0]:.2f} [{c['ratio_full'][1]:.2f}, {c['ratio_full'][2]:.2f}] | {c['ratio_strat'][0]:.2f} [{c['ratio_strat'][1]:.2f}, {c['ratio_strat'][2]:.2f}] | {c['z_full'][0]:+.1f} [{c['z_full'][1]:+.1f}, {c['z_full'][2]:+.1f}] | {c['z_strat'][0]:+.1f} [{c['z_strat'][1]:+.1f}, {c['z_strat'][2]:+.1f}] | "
                 f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] (n_den<=0: {c['gap_den_le0']}) | {c['Estrat_pos']}/{n} | {c['Estrat_lt_Efull']}/{n} | {c['actual_lt_Estrat']}/{n} | {c['actual_gt_Estrat']}/{n} |")
    L.append("")
sens = sorted(glob.glob(os.path.join(RES, "c_strat_sensitivity_*_s*.json")))
if sens:
    L.append("**Sensitivity: gap closed (median over units with E_full > actual) vs number of K-strata; last column = deciles of |x~|^2** (exact closed form; S = 1200 would give 1 by construction)\n")
    for p, A, E, lab in (("f", "Ag", "Ef", "full"), ("b", "Aw", "Eb", "probe"), ("r", "Ar", "Er", "readout")):
        J0 = json.load(open(sens[0])); sch = J0["meta"]["schemes"]
        L.append(f"{lab}\n"); L.append("| state | corr(K, abs(x~)^2) | " + " | ".join(sch) + " |"); L.append("|---|---|" + "---|" * len(sch))
        for fpath in sens:
            J = json.load(open(fpath)); rows = J["rows"]; f_ = lambda k: np.array([r[k] for r in rows]); den = f_(E) - f_(A); ok = den > 0
            v = [np.median((f_(E) - f_(f"{p}_{s}"))[ok] / den[ok]) for s in sch]
            L.append(f"| {J['meta']['tag']}_s{J['meta']['seed']} | {J['meta']['corr_K_xnorm2']:.3f} | " + " | ".join(f"{x:+.2f}" for x in v) + " |")
        L.append("")
open(os.path.join(RES, "c_cross_tables.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
