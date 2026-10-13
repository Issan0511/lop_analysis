"""xbar_arrangement_1014 / agent A follow-up: append the isotropic-directions tables to a_tables.md
(reads key "iso_followup" of results/xbar_arrangement_1014/a_occ_<tag>.json; replaces an earlier copy of this section)."""
import json, os
import numpy as np
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/analysis/xbar_arrangement_1014/a_tables.md"
MARK = "## Follow-up (a2): isotropic-directions model"
TAGS = ["LR0.03_s0", "LR0.03_s1", "LR0.1_s0", "LR0.1_s1", "LR0.2_s0", "LR0.2_s1", "LR0.3_s0", "LR0.3_s1", "LR0.5_s0", "LR0.5_s1", "ELU_s0", "ELU_s1"]
J = {t: json.load(open(os.path.join(RES, f"a_occ_{t}.json"))) for t in TAGS}
I = {t: J[t]["iso_followup"] for t in TAGS}
def mq(s, d=3): return f"{s['med']:.{d}f} [{s['q10']:.{d}f}, {s['q90']:.{d}f}]"
def j(v, d=2): return " ".join(f"{x:.{d}f}" for x in v)
L = []; P = L.append
P(MARK + "\n")
P("Model: w⊥ ~ N(0, I) projected orthogonal to x̄; u⊥_kn = w⊥·x̃_n; x̃⊥_n = x̃_n − (K_n/K̄)x̄; cot θ_n = K_n/(|x̄|·|x̃⊥_n|). "
  "f_iso: per state and alive unit, g ~ N(0, I_785), g ← g − x̄(x̄·g)/K̄, u = Xt·g rescaled to sd_n(u) = s⊥_k, z_iso = m_k K_n/K̄ + u (actual m_k); occupancy averaged over 20 draws. "
  "Script a2_iso.py; numbers in the key `iso_followup` of a_occ_<file>.json. Medians [10%, 90%] over the 90 phase-2 states unless marked.\n")
s0 = I["LR0.1_s0"]; s1 = I["LR0.1_s1"]
P(f"Data: |x̃⊥| seed 0 min/mean/max {s0['nperp']['min']:.2f}/{s0['nperp']['mean']:.2f}/{s0['nperp']['max']:.2f}, seed 1 {s1['nperp']['min']:.2f}/{s1['nperp']['mean']:.2f}/{s1['nperp']['max']:.2f}; "
  f"cot θ seed 0 {s0['cot']['min']:.3f}/{s0['cot']['mean']:.3f}/{s0['cot']['max']:.3f}, seed 1 {s1['cot']['min']:.3f}/{s1['cot']['mean']:.3f}/{s1['cot']['max']:.3f}; "
  f"Spearman(K, cot θ) {s0['spearman_K_cot']:.3f} / {s1['spearman_K_cot']:.3f}; Pearson(K, |x̃⊥|) {s0['pearson_K_nperp']:.3f} / {s1['pearson_K_nperp']:.3f}.\n")
P("### (1) cot θ vs K, and the iso occupancy profile [実測]\n")
P("| file | Spearman(f, cot θ) | Spearman(f, K) | #states cot θ more negative /90 | Pearson(f, f_iso) | Spearman(f, f_iso) | Pearson(f_iso, K) | max f | max f_iso (20-draw mean profile) | max f_iso (single draw, mean) | mean p | mean p iso |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    q = I[t]["q1"]
    P(f"| {t} | {mq(q['sp_f_cot'],2)} | {mq(q['sp_f_K'],2)} | {q['n_states_cot_more_negative']} | {mq(q['p_f_fiso'],2)} | {mq(q['sp_f_fiso'],2)} | {mq(q['p_fiso_K'],2)} | {mq(q['max_f'])} | {mq(q['max_fiso_avg'])} | {mq(q['max_fiso_draw'])} | {mq(q['mean_p'],4)} | {mq(q['mean_p_iso'],4)} |")
P("\nf/f_iso per K-octile (median over states; octile 1 = darkest) and the 10–90% range of octiles 1 and 8:\n")
P("| file | o1 | o2 | o3 | o4 | o5 | o6 | o7 | o8 | o1 10–90% | o8 10–90% | min–max of the 8 medians |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    r = I[t]["q1"]["ratio"]
    P(f"| {t} | " + " | ".join(f"{x:.2f}" for x in r["med"]) + f" | {r['q10'][0]:.2f}–{r['q90'][0]:.2f} | {r['q10'][7]:.2f}–{r['q90'][7]:.2f} | {min(r['med']):.2f}–{max(r['med']):.2f} |")
P("\n### (2) Spread law: sd over alive units of u⊥_kn regressed on |x̃⊥_n| (over the 1,200 inputs) [実測]\n")
P("`rel. intercept` = intercept / mean_n sd. `norm.` = the same with u⊥_kn/s⊥_k (extra). Under the iso model the intercept is 0.\n")
P("| file | slope | intercept | R² | rel. intercept | norm.: R² | norm.: rel. intercept | init: slope | init: intercept | init: R² | init: rel. intercept | init norm.: R² | init norm.: rel. int. |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    q = I[t]["q2"]; i = q["init"]
    P(f"| {t} | {mq(q['slope'])} | {mq(q['intercept'],2)} | {mq(q['r2'],2)} | {mq(q['rel_intercept'],2)} | {mq(q['n_r2'],2)} | {mq(q['n_rel_intercept'],2)} | {i['slope']:.4f} | {i['intercept']:.4f} | {i['r2']:.2f} | {i['rel_intercept']:.3f} | {i['n_r2']:.2f} | {i['n_rel_intercept']:.3f} |")
P("\n### (3) Trained common component: eigenvalues of the alive×alive pattern-correlation matrix vs the iso null (H units, 20 draws) [実測]\n")
P("λ_i/null = actual i-th eigenvalue / mean over draws of the null i-th eigenvalue. `#>null max` = number of actual eigenvalues above the largest null λ1 over 20 draws. |cos(v1, 1)| = alignment of the top eigenvector with the all-units direction (1/√H for a random direction). T = 0 is the initial weights.\n")
P("| file | T | H | mean corr | null mean corr | sd(ū⊥)·√H/mean s⊥ | λ1 | null λ1 (mean / max) | λ1/null | λ2/null | λ3/null | #>null max | \\|cos(v1, 1)\\| |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    for T, q in I[t]["q3"].items():
        P(f"| {t} | {T} | {q['H']} | {q['mean_corr']:.3f} | {q['null_mean_corr']:.3f} | {q['sd_ubar_sqrtH_over_mean_sp']:.2f} | {q['lam'][0]:.2f} | {q['null_lam_mean'][0]:.2f} / {q['null_lam1_max']:.2f} | {q['ratio_lam'][0]:.2f} | {q['ratio_lam'][1]:.2f} | {q['ratio_lam'][2]:.2f} | {q['n_above_null_max']} | {q['cos_v1_ones']:.2f} |")
P("\n### (4) Where the common component lives (T = 100) [実測]\n")
P("cos = cosine of w̄⊥ = mean_k w⊥_k (alive) with the top-5 eigenvectors of Σ⊥ in weight space. var share = λ_i (v_i·w̄⊥)²/(w̄⊥ᵀΣ⊥w̄⊥) = share of var_n(ū⊥) carried by PC i (extra); reference: λ_i/tr Σ⊥. "
  "Persistent input = the input that is the argmax of f in most phase-2 states (count /90). Rank 1 = largest ū⊥ (or cot θ).\n")
P("| file | cos PC1…PC5 | var share PC1…PC5 | λ_i/tr Σ⊥ PC1…PC5 | Spearman(ū⊥, cot θ) | Spearman(ū⊥, \\|x̃⊥\\|) | Spearman(ū⊥, K) | argmax ū⊥: n (K) | persistent argmax f: n (K, count) | coincide | rank of persistent in ū⊥ | rank of persistent in cot θ | \\|x̃⊥\\| of persistent |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in TAGS:
    q = I[t]["q4"]
    P(f"| {t} | {j(q['cos_wbar_pc'],3)} | {j(q['var_share_pc'],3)} | {j(q['pc_eig_frac'],3)} | {q['sp_ubar_cot']:.2f} | {q['sp_ubar_nperp']:.2f} | {q['sp_ubar_K']:.2f} | {q['argmax_ubar']} ({q['K_argmax_ubar']:.2f}) | {q['persistent_argmax_f']} ({q['K_persistent']:.2f}, {q['persistent_count']}) | {q['coincide_ubar_persistent']} | {q['rank_of_persistent_in_ubar']} | {q['cot_rank_of_persistent']} | {q['nperp_persistent']:.2f} |")
P("\n### (5) Participation ratio of Σ⊥ and the sd of pairwise pattern correlations (one line per seed) [実測; data only for PR]\n")
P("| seed | PR(Σ⊥) | 1/√PR | iso null sd (100 units, 20 draws) | init actual sd | phase-2 actual sd (T = 11, 21, …, 100 pooled) per file |")
P("|---|---|---|---|---|---|")
for s in (0, 1):
    ts = [t for t in TAGS if t.endswith(f"s{s}")]; q = I[ts[0]]["q5"]
    acts = ", ".join(f"{t.rsplit('_', 1)[0]} {J[t]['A3']['pooled_sampled']['sd']:.3f}" for t in ts)
    P(f"| {s} | {q['PR']:.2f} | {q['inv_sqrt_PR']:.3f} | {q['null_sd']:.3f} | {J[ts[0]]['A3']['0']['sd']:.3f} | {acts} |")
txt = open(OUT).read().split(MARK)[0].rstrip("\n") + "\n\n"
open(OUT, "w").write(txt + "\n".join(L) + "\n")
print("appended", len(L), "lines to", OUT)
