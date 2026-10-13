"""C (xbar_arrangement_1014): summarize c_strat_expect_<tag>_s<seed>.json -> tables C1-C4 + MATCH check against P149 (json + log lines).
usage: c_summarize.py [state ...]   (state = a01_s0 etc.; default: all present)"""
import os, sys, json, glob, re
import numpy as np
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
REF = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003"
def p149_ref(tag, seed):
    if seed == 0: return os.path.join(REF, "stage10_1005", f"p149_expect_{tag}.json"), os.path.join(REF, "stage10_1005", f"p149_expect_{tag}.log")
    return os.path.join(REF, "stage16_1008_seed1", "out", f"p149_expect_{tag}_s{seed}.json"), os.path.join(REF, "stage16_1008_seed1", "logs", f"p149_expect_{tag}_s{seed}.log")
def p149_lines(rows, tag):
    f_ = lambda k: np.array([r[k] for r in rows]); n = len(rows); L = []
    L.append(f"{tag}: units {n} | Cov(e', d) > 0: {int((f_('cov_ed') > 0).sum())} | lemma vs MC mean (full): max |diff|/sd*sqrt(150) {np.max(np.abs(f_('Ef') - f_('mc_f')) / (f_('sd_f') / np.sqrt(150))):.1f}")
    for nm, S, E, sd, T in (("full M", "Ag", "Ef", "sd_f", "Tbar_f"), ("probe M (gate path removed)", "Aw", "Eb", "sd_b", "Tbar_b")):
        ra = f_(S) / f_(E); L.append(f"   gate, {nm}: expectation > 0: {int((f_(E) > 0).sum())}/{n} | actual > 0: {int((f_(S) > 0).sum())}/{n} | actual/expectation median {np.median(ra):.2f} min {ra.min():.2f} max {ra.max():.2f} | margin E/sd median {np.median(f_(E) / f_(sd)):.1f} min {(f_(E) / f_(sd)).min():.1f} | z median {np.median((f_(S) - f_(E)) / f_(sd)):+.1f} | Tbar median {np.median(f_(T)):.3f} max {f_(T).max():.3f}")
    L.append(f"   screening (actual / probe-form actual): median {np.median(f_('Ag') / f_('Aw')):.2f} min {(f_('Ag') / f_('Aw')).min():.2f} max {(f_('Ag') / f_('Aw')).max():.2f}")
    rr = f_("Ar") / f_("Er"); L.append(f"   readout: expectation > 0: {int((f_('Er') > 0).sum())}/{n} | actual > 0: {int((f_('Ar') > 0).sum())}/{n} | actual/expectation median {np.median(rr):.3f} | Cov(u, phi) > 0: {int((f_('covr') > 0).sum())}/{n}")
    return L
q = lambda x, p: float(np.percentile(x, p))
def fmt_mmm(x): return f"{np.median(x):.2f} [{x.min():.2f}, {x.max():.2f}]"
def fmt_m19(x, s="+.1f"): return f"{np.median(x):{s}} [{q(x, 10):{s}}, {q(x, 90):{s}}]"
states = sys.argv[1:] or sorted(os.path.basename(p)[len("c_strat_expect_"):-5] for p in glob.glob(os.path.join(RES, "c_strat_expect_*_s*.json")))
summary = {}; md = []
for st in states:
    tag, s = st.split("_s"); seed = int(s); J = json.load(open(os.path.join(RES, f"c_strat_expect_{tag}_s{seed}.json"))); rows = J["rows"]; meta = J["meta"]; n = len(rows)
    f_ = lambda k: np.array([r[k] for r in rows], dtype=float)
    # ---- MATCH vs P149 ----
    jp, lp = p149_ref(tag, seed); ref = json.load(open(jp)); assert [r["unit"] for r in ref] == [r["unit"] for r in rows]
    rel = {k: float(max(abs(r[k] - o[k]) / max(abs(o[k]), 1e-300) for r, o in zip(rows, ref))) for k in ("Ag", "Ef", "sd_f", "mc_f", "Aw", "Eb", "sd_b", "Ar", "Er")}
    logl = [l.rstrip("\n") for l in open(lp) if l.startswith(f"{tag}: units") or l.startswith("   gate,") or l.startswith("   screening") or l.startswith("   readout")]
    mine = p149_lines(rows, tag); line_match = [a_ == b_ for a_, b_ in zip(mine, logl)] if len(logl) == len(mine) else [False] * len(mine)
    rr = {nm: (np.median(f_(S) / f_(E)), (f_(S) / f_(E)).min(), (f_(S) / f_(E)).max()) for nm, S, E in (("full", "Ag", "Ef"), ("probe", "Aw", "Eb"))}
    md.append(f"\n### {tag} seed {seed} (alive units {n}, lambda {meta['lam']:.4g}, threads {meta['threads']})\n")
    md.append(f"MATCH vs P149: log lines identical {sum(line_match)}/{len(mine)} -> {'MATCH' if all(line_match) else 'MISMATCH'}; per-unit max relative diff vs P149 json: " + ", ".join(f"{k} {v:.1e}" for k, v in rel.items()))
    if not all(line_match):
        for a_, b_, ok in zip(mine, logl, line_match):
            if not ok: md.append(f"  - mine: `{a_.strip()}`\n  - P149: `{b_.strip()}`")
    # ---- C1 ----
    md.append("\nC1 (median [min, max] for ratios; median [10%, 90%] for z)\n")
    md.append("| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |")
    md.append("|---|---|---|---|---|---|---|---|")
    S_ = {}
    for nm, S, E, sd, Es, sds, mcs in (("full", "Ag", "Ef", "sd_f", "Es_f", "sd_s_f", "mc_s_f"), ("probe", "Aw", "Eb", "sd_b", "Es_b", "sd_s_b", "mc_s_b"), ("readout", "Ar", "Er", "sd_u_r", "Es_r", "sd_s_r", "mc_s_r")):
        A = f_(S); E0 = f_(E); s0 = f_(sd); E1 = f_(Es); s1 = f_(sds); r0 = A / E0; r1 = A / E1; z0 = (A - E0) / s0; z1 = (A - E1) / s1
        den = E0 - A; gap = (E0 - E1) / den; okden = den > 0
        chk = np.abs(E1 - f_(mcs)) / (s1 / np.sqrt(150))
        S_[nm] = dict(ratio_full=[float(np.median(r0)), float(r0.min()), float(r0.max())], ratio_strat=[float(np.median(r1)), float(r1.min()), float(r1.max())],
                      z_full=[float(np.median(z0)), q(z0, 10), q(z0, 90)], z_strat=[float(np.median(z1)), q(z1, 10), q(z1, 90)],
                      gap_closed=[float(np.median(gap[okden])), q(gap[okden], 10), q(gap[okden], 90)] if okden.any() else None, gap_den_le0=int((~okden).sum()),
                      gap_closed_min=float(gap[okden].min()) if okden.any() else None, gap_closed_max=float(gap[okden].max()) if okden.any() else None,
                      Efull_pos=int((E0 > 0).sum()), Estrat_pos=int((E1 > 0).sum()), actual_pos=int((A > 0).sum()), actual_lt_Estrat=int((A < E1).sum()), actual_gt_Estrat=int((A > E1).sum()),
                      actual_lt_Efull=int((A < E0).sum()), Estrat_lt_Efull=int((E1 < E0).sum()), strat_exact_vs_mc_max=float(chk.max()),
                      margin_full=float(np.median(E0 / s0)), margin_strat=float(np.median(E1 / s1)), sd_ratio=float(np.median(s1 / s0)),
                      z_strat_lt_m2=int((z1 < -2).sum()), z_strat_gt_p2=int((z1 > 2).sum()))
        if nm == "readout":
            S_[nm]["z_full_note"] = "readout sd_full = MC sd of 150 uniform permutations (rng 15, new; P149 had no readout sd)"
        md.append(f"| {nm} | {fmt_mmm(r0)} | {fmt_mmm(r1)} | {fmt_m19(z0)} | {fmt_m19(z1)} | {np.median(E0 / s0):.1f} | {np.median(E1 / s1):.1f} | {np.median(s1 / s0):.2f} |")
    md.append("\n(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)\n")
    md.append("C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual\n")
    md.append("| M | median [10%, 90%] | min | max | units with E_full <= actual |")
    md.append("|---|---|---|---|---|")
    for nm in ("full", "probe", "readout"):
        x = S_[nm]
        if x["gap_closed"] is None: md.append(f"| {nm} | - | - | - | {x['gap_den_le0']} |"); continue
        md.append(f"| {nm} | {x['gap_closed'][0]:.2f} [{x['gap_closed'][1]:.2f}, {x['gap_closed'][2]:.2f}] | {x['gap_closed_min']:.2f} | {x['gap_closed_max']:.2f} | {x['gap_den_le0']} |")
    md.append("\nC3 counts\n")
    md.append("| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for nm in ("full", "probe", "readout"):
        x = S_[nm]; md.append(f"| {nm} | {x['Efull_pos']}/{n} | {x['Estrat_pos']}/{n} | {x['actual_pos']}/{n} | {x['Estrat_lt_Efull']}/{n} | {x['actual_lt_Estrat']}/{n} | {x['actual_gt_Estrat']}/{n} | {x['z_strat_lt_m2']}/{n} | {x['z_strat_gt_p2']}/{n} | {x['strat_exact_vs_mc_max']:.1f} |")
    # ---- C4 ----
    P = lambda k: np.array([r["prof"][k] for r in rows])
    ef = P("e_frac"); df = P("dma_frac"); prof = {}
    for mv in ("Mf", "Mb"):
        dg = P(f"{mv}_diag"); rw = P(f"{mv}_row"); dga = np.array([r["prof"][f"{mv}_diag_all"] for r in rows]); rwa = np.array([r["prof"][f"{mv}_row_all"] for r in rows])
        prof[mv] = dict(diag_rel=np.median(dg / dga[:, None], 0), row_rel=np.median(rw / rwa[:, None], 0), row_neg=(rw < 0).sum(0), rowall_neg=int((rwa < 0).sum()), diag_neg=(dg < 0).sum(0),
                        row_over_diag=np.median(rw / dg, 0))
    pdma = f_("p_open")
    md.append(f"\nC4 per K-decile profiles (decile 1 = dimmest; pooled = median over {n} units; M profiles are decile mean / all-input mean per unit, then median)\n")
    hdr = "| quantity | " + " | ".join(f"D{k + 1}" for k in range(10)) + " |"; md.append(hdr); md.append("|---|" + "---|" * 10)
    row = lambda name, v, s=".2f": md.append(f"| {name} | " + " | ".join(f"{x:{s}}" for x in v) + " |")
    row("K mean", meta["K_dec_mean"], ".1f"); row("abs(x~)^2 mean", meta["xnorm2_dec_mean"], ".1f")
    row("e' mass frac (median)", np.median(ef, 0), ".3f"); row("e' mass frac (mean)", ef.mean(0), ".3f")
    row("(d-a) mass frac (median)", np.median(df, 0), ".3f"); row("(d-a) mass frac (mean)", df.mean(0), ".3f")
    for mv, lab in (("Mf", "full"), ("Mb", "probe")):
        row(f"M_nn rel ({lab})", prof[mv]["diag_rel"], ".2f"); row(f"row sum rel ({lab})", prof[mv]["row_rel"], ".2f"); row(f"row sum / M_nn ({lab})", prof[mv]["row_over_diag"], ".2f")
        row(f"units with decile row-sum mean < 0 ({lab})", prof[mv]["row_neg"], "d")
    md.append(f"\n(units with all-input mean row sum < 0: full {prof['Mf']['rowall_neg']}, probe {prof['Mb']['rowall_neg']}; units with decile M_nn mean < 0: full max {int(prof['Mf']['diag_neg'].max())}, probe max {int(prof['Mb']['diag_neg'].max())}; open fraction p median {np.median(pdma):.3f} [{pdma.min():.3f}, {pdma.max():.3f}])")
    summary[st] = dict(n=n, lam=meta["lam"], match_lines=f"{sum(line_match)}/{len(mine)}", match=bool(all(line_match)), rel_vs_p149=rel, p149_ratio_medians={k: [float(x) for x in v] for k, v in rr.items()}, C=S_,
                       C4=dict(K_mean=meta["K_dec_mean"], xnorm2=meta["xnorm2_dec_mean"], e_frac_med=np.median(ef, 0).tolist(), dma_frac_med=np.median(df, 0).tolist(),
                               **{f"{mv}_{k}": (v.tolist() if hasattr(v, "tolist") else v) for mv in prof for k, v in prof[mv].items()}))
open(os.path.join(RES, "c_tables.md"), "w").write("\n".join(md) + "\n")
json.dump(summary, open(os.path.join(RES, "c_summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else int(o))
print("\n".join(md))
