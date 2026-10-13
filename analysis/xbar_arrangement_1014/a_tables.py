"""xbar_arrangement_1014 / agent A: build the tables of REPORT_A_occupancy.md from results/xbar_arrangement_1014/a_occ_<tag>.json.
usage: a_tables.py  -> writes analysis/xbar_arrangement_1014/a_tables.md (generated tables only)"""
import json, os
import numpy as np
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/analysis/xbar_arrangement_1014/a_tables.md"
TAGS = ["LR0.03_s0", "LR0.03_s1", "LR0.1_s0", "LR0.1_s1", "LR0.2_s0", "LR0.2_s1", "LR0.3_s0", "LR0.3_s1", "LR0.5_s0", "LR0.5_s1", "ELU_s0", "ELU_s1"]
R = {t: json.load(open(os.path.join(RES, f"a_occ_{t}.json"))) for t in TAGS}

def f3(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "nan"
    return f"{x:.{d}f}"
def mq(s, d=3): return f"{f3(s['med'], d)} [{f3(s['q10'], d)}, {f3(s['q90'], d)}]"
L = []
def P(*a): L.append(" ".join(str(x) for x in a))

P("## 0. Inputs and identity checks [恒等式]\n")
P("| file | K min | K max | K mean | K sd | all K ≥ 1 | K̄ − mean K | max\\|m − mean_n z\\| | max\\|mean_n u⊥\\| | max\\|z̄ − (m̄K/K̄ + ū⊥)\\| | octile sizes |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    r = R[t]; K = r["K"]; c = r["checks"]
    P(f"| {t} | {K['min']:.2f} | {K['max']:.2f} | {K['mean']:.2f} | {K['sd']:.2f} | {K['all_ge_1']} | {K['Kbar_minus_meanK']:.1e} | {c['m_max_abs']:.1e} | {c['umean_max_abs']:.1e} | {c['zbar_identity_max_abs']:.1e} | {'/'.join(str(x) for x in r['oct_count'])} |")
P("")
P("K-octile mean K (octile 1 = darkest):\n")
P("| file | " + " | ".join(f"o{o+1}" for o in range(8)) + " |"); P("|---|" + "---|" * 8)
for t in TAGS: P(f"| {t} | " + " | ".join(f"{x:.1f}" for x in R[t]["oct_Kmean"]) + " |")

P("\n## A1. Occupancy profile over phase 2 (T 11..100, 90 states) [実測]\n")
P("Entries: median [10%, 90%] over the 90 states. `sim` = the same statistic for f drawn from the model itself (independent permutation of each unit's u⊥ over inputs, 10 draws per state, mean over draws).\n")
P("| file | H_alive | F_flat = mean p | Pearson(f,K) | sim | Spearman(f,K) | sim | Pearson(F̂,K) | Pearson(f,F̂) | sim | Spearman(f,F̂) | Pearson(f,F̂_G) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    s = R[t]["phase2_summary"]
    P(f"| {t} | {mq(s['H_alive'],0)} | {mq(s['F_flat'],4)} | {mq(s['pearson_fK'],2)} | {f3(s['sim_pearson_fK']['med'],2)} | {mq(s['spearman_fK'],2)} | {f3(s['sim_spearman_fK']['med'],2)} | {mq(s['pearson_FhatK'],2)} | {mq(s['pearson_f_Fhat'],2)} | {f3(s['sim_pearson_f_Fhat']['med'],2)} | {mq(s['spearman_f_Fhat'],2)} | {mq(s['pearson_f_FG'],2)} |")
P("")
P("| file | max_n f_n | max over states | K at argmax f | K mean | max_n F̂ | #inputs f ≥ 1/2 (max over states) | #states with any f ≥ 1/2 | sim: max f | sim: #f ≥ 1/2 (max) | #alive units with m>0 (max) |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    s = R[t]["phase2_summary"]
    P(f"| {t} | {mq(s['max_f'])} | {f3(s['max_f']['max'])} | {mq(s['K_argmax_f'],1)} | {R[t]['K']['mean']:.1f} | {mq(s['max_Fhat'])} | {s['n_f_ge_half']['max']:.0f} | {s['n_states_any_f_ge_half']} | {mq(s['sim_max_f'])} | {s['sim_n_f_ge_half']['max']:.1f} | {s['n_m_pos']['max']:.0f} |")
P("\n### A1 octile tables (medians over the 90 phase-2 states; octile 1 = darkest)\n")
for t in TAGS:
    s = R[t]["phase2_summary"]
    P(f"**{t}**\n")
    P("| octile | mean K | f | F̂ | F̂_G | f/F̂ | f/F̂ (10–90%) | f/F̂_G | f drawn from model (sim) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for o in range(8):
        P(f"| {o+1} | {R[t]['oct_Kmean'][o]:.1f} | {s['oct_f']['med'][o]:.4f} | {s['oct_Fhat']['med'][o]:.4f} | {s['oct_FG']['med'][o]:.4f} | {s['ratio_f_Fhat']['med'][o]:.2f} | {s['ratio_f_Fhat']['q10'][o]:.2f}–{s['ratio_f_Fhat']['q90'][o]:.2f} | {s['ratio_f_FG']['med'][o]:.2f} | {s['sim_oct_f']['med'][o]:.4f} |")
    of = s['oct_f']['med']; oF = s['oct_Fhat']['med']
    P(f"\noctile-1/octile-8 contrast: f {of[0]/of[7]:.1f}, F̂ {oF[0]/oF[7]:.1f}\n")

P("\n### A1 diagnostic (extra, not in the task list): relaxations of the model and the K-dependence of u⊥ [実測]\n")
P("F̂_oct: each unit's empirical tail of u⊥ taken over the inputs of n's own K-octile only. F̂_scale: u⊥ divided by its within-octile sd, the standardised values taken as exchangeable over all inputs. "
  "D_o = var_{n∈o}(f_n) / mean_{n∈o}[Σ_k P̂_oct,kn(1−P̂_oct,kn)/H²] (≈ 1 for independent units with u⊥ exchangeable within the octile). Medians over the 90 phase-2 states.\n")
P("| file | f/F̂ o1…o8 | f/F̂_oct o1…o8 | f/F̂_scale o1…o8 | D_o o1…o8 | Pearson(f,F̂_oct) |")
P("|---|---|---|---|---|---|")
for t in TAGS:
    st = json.load(open(os.path.join(RES, f"a_strat_{t}.json"))); j = lambda v: " ".join(f"{x:.2f}" for x in v)
    P(f"| {t} | {j(R[t]['phase2_summary']['ratio_f_Fhat']['med'])} | {j(st['r_oct']['med'])} | {j(st['r_scale']['med'])} | {j(st['D_oct']['med'])} | {st['p_f_Foct']['med']:.2f} |")
HE = json.load(open(os.path.join(RES, "a_hetero.json")))
P("\nK-dependence of the spread of u⊥ (exchangeability check beyond the linear correlation; units pooled over T = 11, 21, …, 91, 100; `init` = initial weights):\n")
P("| file | corr(\\|u⊥\\|, K) q10/50/90 | sd of u⊥ in octile / s⊥ (median over units) o1…o8 | frac u⊥>0 in octile o1…o8 | init: corr(\\|u⊥\\|,K) q50 | init: sd ratio o1…o8 |")
P("|---|---|---|---|---|---|")
for t in TAGS:
    h = HE[t]["phase2_sampled"]; i = HE[t]["init"]; j = lambda v: " ".join(f"{x:.2f}" for x in v)
    P(f"| {t} | {j(h['corr_absu_K_q'])} | {j(h['sd_oct_over_sp_med'])} | {j(h['frac_pos_oct_med'])} | {i['corr_absu_K_q'][1]:.2f} | {j(i['sd_oct_over_sp_med'])} |")

P("\n## A2. Mean unit over phase 2 [実測; z̄ = m̄K/K̄ + ū⊥ is 恒等式]\n")
P("Entries: median [10%, 90%] over 90 states unless marked. ρ_n = ū⊥_n/(|m̄|K_n/K̄); z̄_n > 0 ⇔ ρ_n > 1 when m̄ < 0. `sd ratio indep` = sd_n(ū⊥)/(√(Σ_k s⊥_k²)/H) (= 1 for uncorrelated patterns [恒等式]).\n")
P("| file | #inputs z̄>0 (max) | #inputs median_k z>0 (max) | m̄ | max_n z̄ | max_n ρ (median) | max_n ρ (max over states) | sd_n(ū⊥)/mean s⊥ | 1/√H | (sd ratio)·√H | sd ratio indep |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    s = R[t]["phase2_summary"]
    P(f"| {t} | {s['n_zbar_pos']['max']:.0f} | {s['n_median_pos']['max']:.0f} | {mq(s['mbar'],2)} | {mq(s['max_zbar'],2)} | {f3(s['max_rho']['med'])} | {f3(s['max_rho']['max'])} | {mq(s['sd_ubar_over_mean_sp'])} | {f3(s['inv_sqrtH']['med'])} | {mq(s['sd_ubar_x_sqrtH'],2)} | {mq(s['sd_ubar_over_indep'],2)} |")

P("\n## A3. Pattern independence across units (T = 11, 21, …, 91, 100 pooled) [実測]\n")
P("`init` = the same statistics at the initial weights (init0, init1) of the same run, as a reference.\n")
P("| file | pairs | mean corr | sd | q01 | q99 | frac \\|corr\\|>0.5 | mean \\|corr\\| | init: mean | init: sd | init: frac>0.5 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    a = R[t]["A3"]["pooled_sampled"]; i = R[t]["A3"]["0"]
    P(f"| {t} | {a['n_pairs']} | {a['mean']:.3f} | {a['sd']:.3f} | {a['q01']:.3f} | {a['q99']:.3f} | {a['frac_abs_gt_half']:.4f} | {a['mean_abs']:.3f} | {i['mean']:.3f} | {i['sd']:.3f} | {i['frac_abs_gt_half']:.4f} |")
P("")
P("| file | units | corr(u⊥_k, K) q10/50/90 | skew q10/50/90 | excess kurt q10/50/90 | init: corr(u⊥,K) q10/50/90 | init: skew q50 | init: kurt q50 |")
P("|---|---|---|---|---|---|---|---|")
for t in TAGS:
    a = R[t]["A3"]["pooled_sampled"]; i = R[t]["A3"]["0"]
    j = lambda v, d=2: "/".join(f"{x:.{d}f}" for x in v)
    P(f"| {t} | {a['n_units']} | {j(a['uK_q'])} | {j(a['skew_q'])} | {j(a['exkurt_q'])} | {j(i['uK_q'])} | {i['skew_q'][1]:.2f} | {i['exkurt_q'][1]:.2f} |")
P("\nmean pairwise corr per sampled state (T = 11, 21, …, 91, 100):\n")
P("| file | " + " | ".join(str(T) for T in R[TAGS[0]]["sampled"]) + " |"); P("|---|" + "---|" * len(R[TAGS[0]]["sampled"]))
for t in TAGS: P(f"| {t} | " + " | ".join(f"{R[t]['A3'][str(T)]['mean']:.3f}" for T in R[t]["sampled"]) + " |")

P("\n## A4. Transition T = 0 (init, extra row) … 12 [実測]\n")
for t in TAGS:
    P(f"**{t}**\n")
    P("| T | H_alive | #units p>1/2 | #inputs f≥1/2 | #inputs z̄>0 | #inputs median_k z>0 | mean r | mean p | m̄ | #units m>0 | Pearson(f,K) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for T in range(0, 13):
        a = R[t]["A4"][str(T)]
        P(f"| {T}{' (init)' if T == 0 else ''} | {a['H_alive']} | {a['n_p_gt_half']} | {a['n_f_ge_half']} | {a['n_zbar_pos']} | {a['n_median_pos']} | {a['mean_r']:.3f} | {a['mean_p']:.4f} | {a['mbar']:.2f} | {a['n_m_pos']} | {a['pearson_fK']:.2f} |")
    P("")

P("\n## A5. Brightness of threshold inputs (T = 11, 21, …, 91, 100 pooled) [実測 vs model]\n")
P("Actual = octile fractions of the K of the 3 threshold inputs of every alive unit, pooled. Model columns (each a fraction over octiles):\n"
  "- `spec`: ∝ (1/H) Σ_k ĝ_k(c_kn), ĝ_k = Gaussian KDE of u⊥_k with bandwidth 0.3 s⊥_k, c_kn = −m_k K_n/K̄ (as written in the task).\n"
  "- `unit`: the same with each unit's ĝ_k(c_k·) normalised to sum 1 over n (each unit contributes exactly 3 inputs).\n"
  "- `unit·K`: `unit` with weight K_n ĝ_k(c_kn) (first-order effect of the log K term of the score).\n"
  "- `MC`: the model run through the actual selection rule: per unit, an independent random permutation of u⊥_k over inputs, z_sim = m_k K/K̄ + u⊥_perm, τ and the score recomputed, top 3; 50 draws per state.\n"
  "- `near` (extra gate, not in the task): the 3 inputs with the smallest |z| (no K term).\n")
for t in TAGS:
    p = R[t]["A5"]["pooled"]
    for gate in [g for g in ("leaky", "E1", "E2", "near") if g in p]:
        g = p[gate]
        P(f"**{t} — {gate}** (pooled n = {g['n_inputs']} threshold inputs)\n")
        P("| | " + " | ".join(f"o{o+1}" for o in range(8)) + " | mean K | frac K < median K |")
        P("|---|" + "---|" * 10)
        P("| actual | " + " | ".join(f"{x:.3f}" for x in g["frac"]) + f" | {g['meanK']:.2f} | {g['frac_below_medK']:.3f} |")
        P("| spec | " + " | ".join(f"{x:.3f}" for x in g["pred_spec"]) + f" | {g['meanK_spec']:.2f} | {g['below_spec']:.3f} |")
        P("| unit | " + " | ".join(f"{x:.3f}" for x in g["pred_unit"]) + f" | {g['meanK_unit']:.2f} | {g['below_unit']:.3f} |")
        P("| unit·K | " + " | ".join(f"{x:.3f}" for x in g["pred_unitK"]) + f" | {g['meanK_unitK']:.2f} | {g['below_unitK']:.3f} |")
        mcb = sum(g["pred_mc"][:4])
        P("| MC | " + " | ".join(f"{x:.3f}" for x in g["pred_mc"]) + f" | {g['meanK_mc']:.2f} | (o1–o4) {mcb:.3f} |")
        P("")
    P(f"K of open gates ({t}): unit-weighted mean {p['Kopen_unitw']:.2f}, model {p['Kopen_unitw_model']:.2f}, mean K {p['meanK']:.2f}\n")

P("\n### A5 compact (pooled; o1 fraction and frac below median K; uniform = 0.125 / 0.5)\n")
P("| file | gate | actual o1 | spec o1 | MC o1 | actual <med | spec <med | MC <med (o1–o4) | actual mean K | MC mean K | mean K | K open (unit-w) | model |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    p = R[t]["A5"]["pooled"]
    for gate in [g for g in ("leaky", "E1", "E2", "near") if g in p]:
        g = p[gate]
        P(f"| {t} | {gate} | {g['frac'][0]:.3f} | {g['pred_spec'][0]:.3f} | {g['pred_mc'][0]:.3f} | {g['frac_below_medK']:.3f} | {g['below_spec']:.3f} | {sum(g['pred_mc'][:4]):.3f} | {g['meanK']:.2f} | {g['meanK_mc']:.2f} | {p['meanK']:.2f} | {p['Kopen_unitw']:.2f} | {p['Kopen_unitw_model']:.2f} |")
open(OUT, "w").write("# xbar_arrangement_1014 agent A: generated tables (a_tables.py)\n\n" + "\n".join(L) + "\n")
print("wrote", OUT, len(L), "table lines")
