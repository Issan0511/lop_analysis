"""seed1_report_tables.py: prints the markdown tables of SEED1_REPORT.md from results/cobirth_seeds_1013/seed1.json (no computation beyond formatting / ratios)."""
import json
D = json.load(open("/home/issan/Projects/claude/wt/cobirth_seeds_1013/results/cobirth_seeds_1013/seed1.json")); C = D["conditions"]
S1 = ["a01s1", "a03s1", "a05s1"]; S0 = ["a01s0", "a03s0", "a05s0"]; LAB = {"a01s1": "a0.1 s1", "a03s1": "a0.3 s1", "a05s1": "a0.5 s1", "a01s0": "a0.1 s0", "a03s0": "a0.3 s0", "a05s0": "a0.5 s0"}
KN = {"presence": "presence", "births": "births"}
def f(x, n=3): return "—" if x is None else f"{x:.{n}f}"
def e(x): return "—" if x is None else (f"{x:.1e}".replace("e-0", "e−").replace("e-", "e−").replace("e+00", ""))
print("### per-target / pooled")
for t in S1:
    print(f"\n#### {LAB[t]}\n")
    print("| 量 | μ_i 中央値 [q10, q90] | F_i q10 / **q50** / q90 / 最大 | F_i > 1 | Σ_i Δ_i | Σ_i (s_i² − μ_i) | Δ_i > 0 | プール平均 λ | プール Fano（ddof=1） | 標的内 / 標的間 | 最大 N | P(N≥2) 観測 / Pois / 混合 | P(N≥3) 観測 / Pois / 混合 | P(N≥5) 観測 (件) / Pois / 混合 | N≥10 件 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("presence", "births"):
        r = C[t][k]; mq = r["mu_quantiles"]; fq = r["fano_quantiles"]; po = r["pooled"]
        print(f"| {k} | {mq['0.5']:.3f} [{mq['0.1']:.3f}, {mq['0.9']:.3f}] | {fq['0.1']:.3f} / **{fq['0.5']:.3f}** / {fq['0.9']:.3f} / {fq['max']:.3f} | {r['share_fano_gt1']:.2f} | {r['sum_delta']:.2f} | {r['sum_var1_minus_mu']:.2f} | {r['share_delta_gt0']:.2f} | {po['mean']:.3f} | {po['fano']:.3f} | {po['fano_within']:.3f} / {po['fano_between']:.3f} | {po['max']} | {po['ge2']:.4f} / {po['poisson_ge2']:.4f} / {po['mixture_ge2']:.4f} | {po['ge3']:.4f} / {po['poisson_ge3']:.5f} / {po['mixture_ge3']:.5f} | {po['ge5']:.5f} ({po['n_ge5']}) / {e(po['poisson_ge5'])} / {e(po['mixture_ge5'])} | {po['n_ge10']} |")
    po = C[t]["presence"]["pooled"]; print(f"\n度数表（presence、値 0,1,2,…）: 観測 {', '.join(str(int(x)) for x in po['hist_obs'])} / Poisson {', '.join(f'{x:.1f}' for x in po['hist_poisson_same_mean'])} / 混合 {', '.join(f'{x:.1f}' for x in po['hist_mixture_per_target_poisson'])}")
    print(f"P(N≥3) の過小見積もり（観測 / 混合）: presence {po['ge3'] / po['mixture_ge3']:.1f} 倍、births {C[t]['births']['pooled']['ge3'] / C[t]['births']['pooled']['mixture_ge3']:.1f} 倍")
print("\n### nulls")
print("| 腕・量 | F_i 中央値 観測 | 帰無 1 shift 95% / p | 帰無 1b（k_i 4 分位層別）95% / p | 帰無 2（標的シャッフル）95% / p | 最大 N 観測 / 帰無 1 / p / 帰無 1b / p | P(N≥3) 観測 / 帰無 1 / 帰無 1b / p(1b) | Σ_i Δ_i 観測 / 帰無 1 / 帰無 1b 中央値 [95%] | 標的ごと p_F<0.05: 帰無 1 / 1b / 2 | sd_i(μ_i) 観測 / 帰無 2 / p |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in S1 + S0:
    for k in ("presence", "births"):
        r = C[t][k]; a = r["null_candidate_series"]["shift"]; b = r["null_candidate_series"]["strat"]; c = r["null_target_shuffle"]
        print(f"| {LAB[t]} {k} | {a['observed']['median_fano']:.3f} | [{a['null']['median_fano']['0.025']:.3f}, {a['null']['median_fano']['0.975']:.3f}] / {a['p_two_sided']['median_fano']:.3f} | [{b['null']['median_fano']['0.025']:.3f}, {b['null']['median_fano']['0.975']:.3f}] / {b['p_two_sided']['median_fano']:.3f} | [{c['null']['median_fano']['0.025']:.3f}, {c['null']['median_fano']['0.975']:.3f}] / {c['p_two_sided']['median_fano']:.3f} | {a['observed']['max_N']} / [{a['null']['max_N']['0.025']:.0f}, {a['null']['max_N']['0.975']:.0f}] / {a['p_two_sided']['max_N']:.3f} / [{b['null']['max_N']['0.025']:.0f}, {b['null']['max_N']['0.975']:.0f}] / {b['p_two_sided']['max_N']:.3f} | {a['observed']['pooled_ge3']:.4f} / [{a['null']['pooled_ge3']['0.025']:.4f}, {a['null']['pooled_ge3']['0.975']:.4f}] / [{b['null']['pooled_ge3']['0.025']:.4f}, {b['null']['pooled_ge3']['0.975']:.4f}] / {b['p_two_sided']['pooled_ge3']:.3f} | {a['observed']['sum_delta']:.2f} / [{a['null']['sum_delta']['0.025']:.2f}, {a['null']['sum_delta']['0.975']:.2f}] / {b['null']['sum_delta']['0.5']:.2f} [{b['null']['sum_delta']['0.025']:.2f}, {b['null']['sum_delta']['0.975']:.2f}] | {a['per_target']['n_p_fano_lt05']} / {b['per_target']['n_p_fano_lt05']} / {c['per_target']['n_p_fano_lt05']} | {c['observed']['sd_mu_across_targets']:.4f} / [{c['null']['sd_mu_across_targets']['0.025']:.4f}, {c['null']['sd_mu_across_targets']['0.975']:.4f}] / {c['p_two_sided']['sd_mu_across_targets']:.3f} |")
print("\n残り（帰無 1b で説明されない分）:")
for t in S1 + S0:
    r = C[t]["presence"]; b = r["null_candidate_series"]["strat"]; obs = r["sum_delta"]; med = b["null"]["sum_delta"]["0.5"]
    print(f"- {LAB[t]}: Σ Δ 観測 {obs:.2f}、帰無 1b 中央値 {med:.2f}（{100 * med / obs:.0f}%）、残り {obs - med:.2f}（{100 * (obs - med) / obs:.0f}%）；F 中央値の超過 = 観測 − 帰無 1b の中央値 {r['fano_quantiles']['0.5'] - b['null']['median_fano']['0.5']:.3f}")
print("\n### M3 decomposition (T 11..100)")
print("| 腕 | n（標的-課題） | N 平均 / Fano | unit 間の割合 / unit 内 Fano | 幅区分 間 / 内 Fano | unit×幅 内 Fano | 課題間の割合 | C 間 / C 内 Fano | **(C, 幅) 間 / 内 Fano** | C 平均 / Fano / 最大 | C の標的ごと Fano q10/50/90 | 帰無 U: C 平均 / Fano | C/C_U | 通過率 N/C 観測 / 帰無 | 観測/λ_U | births: unit 間 / (C,幅) 内 Fano |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in S1 + S0:
    v = C[t]["m3_decomp"]["M1win"]; pl = v["pool"]; cf = pl["C_fano_per_target"]
    print(f"| {LAB[t]} | {v['n']} | {v['N_by_unit']['mean']:.3f} / {v['N_by_unit']['fano']:.3f} | {v['N_by_unit']['share_across']:.4f} / {v['N_by_unit']['fano_within']:.3f} | {v['N_by_kbin_fine']['share_across']:.3f} / {v['N_by_kbin_fine']['fano_within']:.3f} | {v['N_by_unit_x_kbin_fine']['fano_within']:.3f} | {v['N_by_task']['share_across']:.3f} | {v['N_by_pool']['share_across']:.3f} / {v['N_by_pool']['fano_within']:.3f} | {v['N_by_pool_x_kbin_fine']['share_across']:.3f} / **{v['N_by_pool_x_kbin_fine']['fano_within']:.3f}** | {pl['C_mean']:.3f} / {pl['C_fano']:.3f} / {pl['C_max']} | {cf['0.1']:.2f} / {cf['0.5']:.2f} / {cf['0.9']:.2f} | {pl['CU_mean']:.3f} / {pl['CU_total_fano']:.3f} | {pl['ratio_C_CU']:.3f} | {pl['pass_rate_N_over_C']:.4f} / {pl['pass_rate_null']:.4f} | {pl['ratio_obs_over_lamU']:.2f} | {v['B_by_unit']['share_across']:.4f} / {v['B_by_pool_x_kbin_fine']['fano_within']:.3f} |")
print("\n### Cantelli (presence)")
print("| 腕 | n* の取り方 | n* 中央値 / 最小 | Cantelli 最大（標的）/ 中央値 / q90 | Σ_i（1 課題の期待標的数の上界） | (i,T) 単位の最大 | Poisson 独立 最大 | 経験 N_i ≥ n*_i（件）/ 最大頻度 |")
print("|---|---|---|---|---|---|---|---|")
for t in S1 + S0:
    for lab in ("nbeta_nearest_state", "nbeta_min_over_states", "nbeta_single_state", "arm_min_all_states", "arm_median_all_states", "ten"):
        c = C[t]["presence"]["cantelli"].get(lab)
        if c is None: continue
        if t in S0 and lab in ("nbeta_nearest_state", "nbeta_min_over_states", "arm_median_all_states"): continue
        cc = c["cantelli"]
        print(f"| {LAB[t]} | {lab} | {f(c['nstar']['median'], 1)} / {f(c['nstar']['min'], 2)} | {e(cc['max'])} (u{cc['argmax_target']}) / {e(cc['median'])} / {e(cc['q90'])} | {cc['sum_over_targets']:.3f} | {e(cc['max_single_target_task'])} | {e(c['poisson_indep']['max'])} | {c['empirical']['n_target_tasks_ge_nstar']} / {c['empirical']['max']:.4f} |")
print("\n### break-even n* (presence): n*_p(i) = μ_i + σ_i √((1−p)/p)")
print("| 腕 | p | 最大（標的、μ, σ²） | 中央値 | q90 | n_β 最小（全状態） | 最大 break-even に対する比 | 標的ごとの余裕 min_i (n_β,min(i) − n*_p(i)) | n_β,min(i) ≤ n*_p(i) の標的 |")
print("|---|---|---|---|---|---|---|---|---|")
for t in S1 + S0:
    for p in ("0.32", "0.26", "0.19", "0.14", "0.11", "0.08"):
        b = C[t]["presence"]["breakeven_nstar"][p]
        print(f"| {LAB[t]} | {p} | {b['max']:.2f}（u{b['argmax_target']}、{b['mu_at_max']:.3f}, {b['var_at_max']:.3f}） | {b['median']:.2f} | {b['q90']:.2f} | {b['n_beta_min_over_targets']:.2f} | ×{b['ratio_min_nbeta_over_max_breakeven']:.2f} | {b['margin_min_over_targets_nbeta_minus_breakeven']:.2f} | {b['n_targets_nbeta_le_breakeven']}/{b['n_targets_with_nbeta']} |")
print("\n### windows (presence)")
for t in S1:
    print(f"- {LAB[t]}: " + "; ".join(f"{w}: μ̄ {v['mean_mu']:.3f}, F 中央値 {v['median_fano']:.2f}, プール Fano {v['pooled_fano']:.2f}, P(N≥2) {v['pooled_ge2']:.3f}, 最大 {v['max']}" for w, v in C[t]["presence"]["windows"].items()))
print("\n### stoch / P177 anchors")
for t in S1 + S0:
    s = C[t]["stoch_anchor"]; p = C[t]["p177_anchor"]
    print(f"- {LAB[t]}: rows t10..100 {s['rows_t10_100']}, births t11..100 {s['births_t11_100']}, n_t≥1 {s['target_tasks_nt_ge1']}, n_t 平均/最大 {s['nt_mean']:.3f}/{s['nt_max']}, S {s['S_mean']:.3f}/{s['S_max']}, B {s['B_mean']:.3f}/{s['B_max']}, n_t+1 {s['n1_mean']:.3f}/{s['n1_max']}" + (f" MATCH {s['match']}" if 'match' in s else "") + f" | ref t{p.get('ref_t_stoch')} 最大 {p.get('ref_max')}、along t2/10/20/40/100 {p['max_bridges_per_target_along']}、寿命 1 の割合 {p['lifetime1_share_t10_60']:.4f}、相 2 最大 {p['max_bridges_per_target_phase2_T11_100']}" + (f" MATCH ref {p['match_ref_max']} along {p['match_along']} life1 {p['match_life1']}" if 'match_ref_max' in p else ""))
