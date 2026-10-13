"""xbar_arrangement_1014 agent B: append section 8 (iso and iso+common surrogates) to REPORT_B_stratified_null.md and write b_summary_iso.json.
Run after b_report.py (which rewrites sections 1-7). Inputs: results/xbar_arrangement_1014/b_cond_<tag>.json (actual, strat, flat),
b_iso_<tag>.json, b_isoc_<tag>.json (b_iso_null.py modes iso / isoc)."""
import os, json
import numpy as np
WT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014"
RES = os.path.join(WT, "results/xbar_arrangement_1014"); REP = os.path.join(WT, "analysis/xbar_arrangement_1014/REPORT_B_stratified_null.md")
TAGS = ["a003s0", "a003s1", "a01s0", "a01s1", "a02s0", "a02s1", "a03s0", "a03s1", "a05s0", "a05s1", "ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2"]
LEAKY = TAGS[:10]; ELU = TAGS[10:]; NULLS = ("strat", "flat", "iso", "isoc")
D = {}
for t in TAGS:
    c = json.load(open(os.path.join(RES, f"b_cond_{t}.json"))); i = json.load(open(os.path.join(RES, f"b_iso_{t}.json"))); ic = json.load(open(os.path.join(RES, f"b_isoc_{t}.json")))
    D[t] = dict(match=(c["match"]["verdict"], i["match_cobirth_counts"], ic["match_cobirth_counts"]), widths=dict(iso=i["widths"], isoc=ic["widths"]),
                common=ic["common_component"], B5=c["B5"])
    for q in ("N", "C"):
        D[t][q] = dict(actual=c[q]["actual"], strat=c[q]["strat"], flat=c[q]["flat"], iso=i[q]["iso"], isoc=ic[q]["isoc"])


def fm(x, n=3): return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"
def rg(s, n=3, integer=False):
    if integer: return f"{s['median']:.0f} [{s['q05']:.0f}, {s['q95']:.0f}]"
    return f"{fm(s['median'], n)} [{fm(s['q05'], n)}, {fm(s['q95'], n)}]"
def a_(t, q, key): return D[t][q]["actual"][key]
def s_(t, q, nl, key): return D[t][q][nl]["summary"][key]
def inband(t, q, nl, key): s = s_(t, q, nl, key); return s["q05"] <= a_(t, q, key) <= s["q95"]
def cnt(q, nl, key, tags=TAGS): return sum(inband(t, q, nl, key) for t in tags)
def ratio(t, q, nl): return s_(t, q, nl, "mu_pooled")["median"] / a_(t, q, "mu_pooled")
def rr(v, n=2): return f"{min(v):.{n}f}–{max(v):.{n}f}"


L = []; P = L.append
P("## 8. iso and iso+common surrogates (coordinator follow-ups)\n")
P("Same box, definitions, 14 conditions, targets and statistics as §1–6; script `analysis/xbar_arrangement_1014/b_iso_null.py` (mode `iso` / `isoc`), tables by `b_report_iso.py`. "
  "Per phase-2 state and alive unit k: x̄ = mean_n Xt_n (785), K̄ = mean K = x̄·x̄, m_k = mean_n z_kn, u⊥_kn = z_kn − m_k K_n/K̄, s⊥_k = sd_n(u⊥_kn) (ddof 0).\n")
P("- **iso**: g ~ N(0, I_785), g ← g − x̄(x̄·g)/K̄, u = Xt·g, u rescaled to sd_n(u) = s⊥_k, z^iso_kn = m_k K_n/K̄ + u_n with the actual m_k; gates 1[z^iso > 0]. Seeds `default_rng(2000 + r)`.")
P("- **iso+common (isoc)**: ū⊥_n = mean over alive k of u⊥_kn; the isotropic u (fresh g, same projection) is rescaled to sd_n(u) = sqrt(max(s⊥_k² − sd_n(ū⊥)², 0.25 s⊥_k²)); "
  "z^isoc_kn = m_k K_n/K̄ + ū⊥_n + u_n. Seeds `default_rng(3000 + r)`.")
P("- Both: one g per (unit, state, draw), R = 20, drawn state by state T = 11..100 for all 100 columns; candidate eligibility = the unit's actual alive status; the target's own T_i, O_i, k_i stay actual. "
  "Counts reproduce cobirth_seeds in all 28 runs (14 iso + 14 isoc: MATCH). The iso rerun of ELU_s1_E2 after adding the isoc mode gave bit-identical draws.\n")
P("Criterion: the null reproduces a pooled statistic when the actual value lies in the [5%, 95%] range of its 20 draws. 'ratio' = (median over draws of the pooled mean) ÷ (actual pooled mean).\n")

# verdicts first
P("### 8.0 Verdicts\n")
for q, nm in (("N", "N"), ("C", "C")):
    P(f"- **iso reproduces μ of {nm}**: ratio leaky {rr([ratio(t,q,'iso') for t in LEAKY])}, ELU {rr([ratio(t,q,'iso') for t in ELU])}; pooled mean in band {cnt(q,'iso','mu_pooled')}/14 "
      f"(leaky {cnt(q,'iso','mu_pooled',LEAKY)}/10, ELU {cnt(q,'iso','mu_pooled',ELU)}/4).")
for q, nm in (("N", "N"), ("C", "C")):
    P(f"- **iso reproduces pooled Fano of {nm}**: {cnt(q,'iso','pool_fano')}/14 in band (leaky {cnt(q,'iso','pool_fano',LEAKY)}/10, ELU {cnt(q,'iso','pool_fano',ELU)}/4); "
      f"actual leaky {rr([a_(t,q,'pool_fano') for t in LEAKY])}, ELU {rr([a_(t,q,'pool_fano') for t in ELU])}; iso leaky {rr([s_(t,q,'iso','pool_fano')['median'] for t in LEAKY])}, ELU {rr([s_(t,q,'iso','pool_fano')['median'] for t in ELU])}. "
      f"Median F_i in band {cnt(q,'iso','F_median')}/14.")
for q, nm in (("N", "N"), ("C", "C")):
    P(f"- **iso+common reproduces μ of {nm}**: ratio leaky {rr([ratio(t,q,'isoc') for t in LEAKY])}, ELU {rr([ratio(t,q,'isoc') for t in ELU])}; in band {cnt(q,'isoc','mu_pooled')}/14 "
      f"(leaky {cnt(q,'isoc','mu_pooled',LEAKY)}/10, ELU {cnt(q,'isoc','mu_pooled',ELU)}/4).")
for q, nm in (("N", "N"), ("C", "C")):
    P(f"- **iso+common reproduces pooled Fano of {nm}**: {cnt(q,'isoc','pool_fano')}/14 in band (leaky {cnt(q,'isoc','pool_fano',LEAKY)}/10, ELU {cnt(q,'isoc','pool_fano',ELU)}/4); "
      f"isoc leaky {rr([s_(t,q,'isoc','pool_fano')['median'] for t in LEAKY])}, ELU {rr([s_(t,q,'isoc','pool_fano')['median'] for t in ELU])}. Median F_i in band {cnt(q,'isoc','F_median')}/14.")
P("- **iso vs strat vs flat vs iso+common**, in-band counts out of 14 (pooled Fano N / pooled Fano C / pooled mean N / pooled mean C / median F_i N / median F_i C): "
  + "; ".join(f"{nl} {cnt('N',nl,'pool_fano')}/{cnt('C',nl,'pool_fano')}/{cnt('N',nl,'mu_pooled')}/{cnt('C',nl,'mu_pooled')}/{cnt('N',nl,'F_median')}/{cnt('C',nl,'F_median')}" for nl in NULLS) + ".")
P("- **ratio of pooled means null ÷ actual** (N; C), leaky | ELU: " + "; ".join(
    f"{nl} N {rr([ratio(t,'N',nl) for t in LEAKY])} | {rr([ratio(t,'N',nl) for t in ELU])}, C {rr([ratio(t,'C',nl) for t in LEAKY])} | {rr([ratio(t,'C',nl) for t in ELU])}" for nl in NULLS) + ".")
gapF = lambda q, nl, tags: float(np.median([abs(s_(t, q, nl, 'pool_fano')['median'] - a_(t, q, 'pool_fano')) / a_(t, q, 'pool_fano') for t in tags]))
gapM = lambda q, nl, tags: float(np.median([abs(ratio(t, q, nl) - 1) for t in tags]))
P("- **Does adding the common component close the gap?** median over conditions of |null − actual|/actual, pooled Fano (N; C) and pooled mean (N; C): "
  + "; ".join(f"{nl}: Fano leaky {gapF('N',nl,LEAKY):.3f}; {gapF('C',nl,LEAKY):.3f}, ELU {gapF('N',nl,ELU):.3f}; {gapF('C',nl,ELU):.3f} — mean leaky {gapM('N',nl,LEAKY):.3f}; {gapM('C',nl,LEAKY):.3f}, ELU {gapM('N',nl,ELU):.3f}; {gapM('C',nl,ELU):.3f}" for nl in ("iso", "isoc")) + ".")
mx = lambda nl: ', '.join(f"{s_(t, 'N', nl, 'max')['median']:.0f}" for t in ELU)
P(f"- **ELU (B4)**: max N actual {', '.join(str(a_(t,'N','max')) for t in ELU)}; iso {mx('iso')}; "
  f"isoc {mx('isoc')}. "
  f"#N≥10 actual {', '.join(str(a_(t,'N','n_ge10')) for t in ELU)}; iso {', '.join(rg(s_(t,'N','iso','n_ge10'), integer=True) for t in ELU)}; isoc {', '.join(rg(s_(t,'N','isoc','n_ge10'), integer=True) for t in ELU)}.")
wr = lambda nl, tags: rr([D[t]["widths"][nl]["ratio_iso_over_actual"] for t in tags])
P(f"- **Widths** (realized mean p ÷ actual mean p): iso leaky {wr('iso', LEAKY)}, ELU {wr('iso', ELU)}; isoc leaky {wr('isoc', LEAKY)}, ELU {wr('isoc', ELU)}. "
  f"Common component sd(ū⊥)/mean s⊥ (median over states): leaky {rr([D[t]['common']['sd_ratio']['median'] for t in LEAKY], 3)}, ELU {rr([D[t]['common']['sd_ratio']['median'] for t in ELU], 3)}; "
  f"variance share leaky {rr([D[t]['common']['var_share']['median'] for t in LEAKY], 3)}, ELU {rr([D[t]['common']['var_share']['median'] for t in ELU], 3)}; residual floor never active (0 in all states).")
for q in ("N", "C"):
    bi = lambda nl, key, tags: rr([D[t][q][nl]["B2_F"][key] for t in tags])
    sp = lambda nl, tags: rr([D[t][q][nl]["B3"]["spearman_iT"] for t in tags])
    vr = lambda nl, tags: rr([D[t][q][nl]["B3"]["varratio_median"] for t in tags], 3)
    P(f"- **{q}_i per target (B2) and common cause (B3)**: actual F_i inside the 5–95% band, iso leaky {bi('iso','frac_inside_90',LEAKY)}, ELU {bi('iso','frac_inside_90',ELU)}; "
      f"isoc leaky {bi('isoc','frac_inside_90',LEAKY)}, ELU {bi('isoc','frac_inside_90',ELU)} (strat leaky {bi('strat','frac_inside_90',LEAKY)}, ELU {bi('strat','frac_inside_90',ELU)}). "
      f"Spearman (i,T) iso leaky {sp('iso',LEAKY)}, ELU {sp('iso',ELU)}; isoc leaky {sp('isoc',LEAKY)}, ELU {sp('isoc',ELU)}. "
      f"Variance share (median var ratio) iso leaky {vr('iso',LEAKY)}, ELU {vr('iso',ELU)}; isoc leaky {vr('isoc',LEAKY)}, ELU {vr('isoc',ELU)}.")
P("")

# widths and common component
P("### 8.1 Realized widths and size of the common component\n")
P("p = open fraction k/1200, mean over alive units, pooled over phase-2 states (and draws for the surrogates). Unit-level columns: draw r = 0, all alive (unit, state) pairs. "
  "Common component: sd ratio = sd_n(ū⊥)/mean_k s⊥_k; var share = sd_n(ū⊥)²/mean_k s⊥_k²; unit R² = median over alive units of corr_n(u⊥_k, ū⊥)²; floor active = share of alive units with s⊥_k² − sd(ū⊥)² < 0.25 s⊥_k² (isoc residual floored). Median [5%, 95%] over the 90 phase-2 states.\n")
P("| condition | p actual | p iso | iso/act | p isoc | isoc/act | iso unit-level Pearson(p) | iso units with k=0 | sd(ū⊥)/mean s⊥ | var share | unit R² | floor active |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    wi = D[t]["widths"]["iso"]; wc = D[t]["widths"]["isoc"]; cc = D[t]["common"]
    P(f"| {t} | {wi['p_actual_mean_over_alive_units']:.4f} | {wi['p_iso_mean_over_alive_units']:.4f} | {wi['ratio_iso_over_actual']:.3f} | {wc['p_iso_mean_over_alive_units']:.4f} | {wc['ratio_iso_over_actual']:.3f} | "
      f"{wi['r0_unit_level']['pearson_p']:.3f} | {wi['r0_unit_level']['frac_iso_k0']:.3f} | {rg(cc['sd_ratio'])} | {rg(cc['var_share'])} | {rg(cc['unit_r2_median'])} | {rg(cc['floor_frac'])} |")
P("")
P("Brightness profile realized by the surrogates (occupancy of alive-unit gates per K-decile, phase 2 pooled; decile 0 = smallest K):\n")
P("| condition | source | " + " | ".join(f"d{d}" for d in range(10)) + " | d9/d0 |")
P("|---|---|" + "---|" * 11)
for t in TAGS:
    rows = [("actual", D[t]["widths"]["iso"]["occupancy_by_decile_actual"]), ("iso", D[t]["widths"]["iso"]["occupancy_by_decile_iso"]), ("isoc", D[t]["widths"]["isoc"]["occupancy_by_decile_iso"])]
    for nm, o in rows:
        P(f"| {t} | {nm} | " + " | ".join(f"{v:.4f}" for v in o) + f" | {o[9] / o[0]:.3f} |")
P("")

# B1
for q in ("N", "C"):
    P(f"### 8.{2 if q == 'N' else 3} B1 — {q}_i: actual vs strat / flat / iso / iso+common (median [5%, 95%] over 20 draws)\n")
    P("| condition | pooled Fano: actual | strat | flat | iso | isoc | median F_i: actual | iso | isoc |")
    P("|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        P(f"| {t} | {fm(a_(t,q,'pool_fano'))} | {fm(s_(t,q,'strat','pool_fano')['median'])} | {fm(s_(t,q,'flat','pool_fano')['median'])} | {rg(s_(t,q,'iso','pool_fano'))} | {rg(s_(t,q,'isoc','pool_fano'))} | "
          f"{fm(a_(t,q,'F_median'))} | {rg(s_(t,q,'iso','F_median'))} | {rg(s_(t,q,'isoc','F_median'))} |")
    P("")
    P("| condition | pooled mean: actual | iso | isoc | ratio strat | flat | iso | isoc | median μ_i: actual | iso | isoc |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        P(f"| {t} | {fm(a_(t,q,'mu_pooled'))} | {rg(s_(t,q,'iso','mu_pooled'))} | {rg(s_(t,q,'isoc','mu_pooled'))} | {ratio(t,q,'strat'):.3f} | {ratio(t,q,'flat'):.3f} | {ratio(t,q,'iso'):.3f} | {ratio(t,q,'isoc'):.3f} | "
          f"{fm(a_(t,q,'mu_median'))} | {rg(s_(t,q,'iso','mu_median'))} | {rg(s_(t,q,'isoc','mu_median'))} |")
    P("")
    P(f"| condition | max {q}: actual | strat | flat | iso | isoc | #({q} ≥ 10): actual | strat | flat | iso | isoc | targets μ=0 per draw: iso | isoc |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        P(f"| {t} | {a_(t,q,'max')} | {s_(t,q,'strat','max')['median']:.0f} | {s_(t,q,'flat','max')['median']:.0f} | {rg(s_(t,q,'iso','max'), integer=True)} | {rg(s_(t,q,'isoc','max'), integer=True)} | "
          f"{a_(t,q,'n_ge10')} | {s_(t,q,'strat','n_ge10')['median']:.0f} | {s_(t,q,'flat','n_ge10')['median']:.0f} | {rg(s_(t,q,'iso','n_ge10'), integer=True)} | {rg(s_(t,q,'isoc','n_ge10'), integer=True)} | "
          f"{D[t][q]['iso']['n_F_nan_total'] / 20:.1f} | {D[t][q]['isoc']['n_F_nan_total'] / 20:.1f} |")
    P("")

# B2
P("### 8.4 B2 — per-target position of the actual F_i and μ_i among the target's 20 draws\n")
for q in ("N", "C"):
    P(f"**{q}_i**\n")
    P("| condition | iso F: n | inside 90% | above q95 | below q05 | isoc F: n | inside 90% | above q95 | below q05 | iso μ_i: inside | above | below | isoc μ_i: inside | above | below |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        bi = D[t][q]["iso"]["B2_F"]; bc = D[t][q]["isoc"]["B2_F"]; mi = D[t][q]["iso"]["B2_mu"]; mc = D[t][q]["isoc"]["B2_mu"]
        P(f"| {t} | {bi['n_used']} | {bi['frac_inside_90']:.2f} | {bi['frac_above_q95']:.2f} | {bi['frac_below_q05']:.2f} | {bc['n_used']} | {bc['frac_inside_90']:.2f} | {bc['frac_above_q95']:.2f} | {bc['frac_below_q05']:.2f} | "
          f"{mi['frac_inside_90']:.2f} | {mi['frac_above_q95']:.2f} | {mi['frac_below_q05']:.2f} | {mc['frac_inside_90']:.2f} | {mc['frac_above_q95']:.2f} | {mc['frac_below_q05']:.2f} |")
    P("")

# B3
P("### 8.5 B3 — common-cause check against the iso / iso+common mean series (definitions as §5)\n")
for q in ("N", "C"):
    P(f"**{q}_i**\n")
    P("| condition | iso: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | isoc: Spearman (i,T) | within-target Spearman | var ratio [q25, q75] | noise-corr. | strat Spearman (ref.) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for t in TAGS:
        b = D[t][q]["iso"]["B3"]; c = D[t][q]["isoc"]["B3"]
        P(f"| {t} | {b['spearman_iT']:.3f} | {b['within_target_spearman_median']:.3f} | {b['varratio_median']:.4f} [{b['varratio_q25']:.4f}, {b['varratio_q75']:.4f}] | {b['varratio_noisecorr_median']:.4f} | "
          f"{c['spearman_iT']:.3f} | {c['within_target_spearman_median']:.3f} | {c['varratio_median']:.4f} [{c['varratio_q25']:.4f}, {c['varratio_q75']:.4f}] | {c['varratio_noisecorr_median']:.4f} | {D[t][q]['strat']['B3']['spearman_iT']:.3f} |")
    P("")
P("### 8.6 Files\n")
P("- `/home/issan/Projects/obsidian-research-data/xbar_arrangement_1014/agent_b/b_arrays_iso_<tag>.npz`: keys N, C (int16, −1 = not alive), Niso_mean, Ciso_mean (float, −1 = not alive), Niso_draws, Ciso_draws (int16 [r, T, i]), T_list (11..100).")
P("- `.../agent_b/b_arrays_isoc_<tag>.npz`: same with prefix Nisoc_/Cisoc_.")
P("- `results/xbar_arrangement_1014/b_iso_<tag>.json`, `b_isoc_<tag>.json` (per condition, incl. per-target arrays, widths, common-component sizes), `b_summary_iso.json` (this section's numbers). Logs `.../agent_b/logs/b_iso_<tag>.log`, `b_isoc_<tag>.log`.\n")

summ = {t: dict(match=D[t]["match"], common=D[t]["common"], widths=D[t]["widths"],
                **{q: {nl: dict(ratio_mu_pooled=ratio(t, q, nl), inband={k: inband(t, q, nl, k) for k in ("pool_fano", "F_median", "mu_pooled", "mu_median", "max", "n_ge10")},
                                summary=s_(t, q, nl, "pool_fano") | {}, ) for nl in NULLS} | {"actual": D[t][q]["actual"]} for q in ("N", "C")}) for t in TAGS}
json.dump(dict(conditions=summ, criterion="actual inside [q05, q95] of 20 draws"), open(os.path.join(RES, "b_summary_iso.json"), "w"), indent=1)
txt = open(REP).read()
k = txt.find("## 8. iso and iso+common surrogates")
if k >= 0: txt = txt[:k]
open(REP, "w").write(txt.rstrip("\n") + "\n\n" + "\n".join(L) + "\n")
print("\n".join(L[:20]))
