#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v1_redraw_analyze.py — v1_redraw.py の出力（ラベル引き直しの測度での N_i(T+1)）を D1 の縛りと突き合わせる。

(1) 標的ごとの μ^rd_i・Var^rd_i・F^rd_i（引き直し R 本）と、出発状態 θ(T) の実測の材料（cos・ω̄(T_i)）＋ μ^rd_i を (4.5) に入れた F の上界。
    F^rd > 上界 の標的の数と、NB ブートストラップの p 値（真の F が上界ならば F^rd がそれ以上になる確率）。
(2) P^rd(N_i ≥ n*)（n* = n_β・10・腕の最小）対 Cantelli（F の上界・F^rd）対 Poisson。
(3) (G3) の直接検査: 累積 Δz_jn = Z_end − Z0（4,000 歩）の unit 間相関（引き直しをまたいだ経験相関）を、
    1 歩の式 cos(v_j^c, v_k^c)·ω_jk(n) と比べる。|ρ_emp| ≤ |cos| が継承されるか、符号が合うか、Pearson(ρ_emp, cos·ω)。
    加えて累積 Δz の正規性（歪度・尖度）(G1)。
(4) 実際の次の状態（ref end0[T]）の N_i と、引き直し分布の位置（例: a0.3 T=17 → 実際の u55 は 14）。
出力: results/cobirth_1013/v1_redraw_<arm>_T<T>.json、ログ obsidian-research-data/cobirth_1013/v1/v1_redraw_analyze_<arm>_T<T>.log
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1_lib as L  # noqa: E402

WT = "/home/issan/Projects/claude/wt/cobirth_1013"
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/v1"
LOG = []


def log(s):
    print(s, flush=True); LOG.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a0.3"); ap.add_argument("--T", type=int, default=17); ap.add_argument("--R-boot", type=int, default=4000)
    args = ap.parse_args(); arm, T = args.arm, args.T
    z = np.load(f"{RAW}/redraw_{arm}_T{T}.npz", allow_pickle=True)
    isb0, isb, Nn, Bn, al = z["isb0"], z["isb"], z["N"].astype(int), z["B"].astype(int), z["alive"]
    Zend, Z0 = z["Z_end"].astype(np.float64), z["Z0"].astype(np.float64)
    R = int(z["R"]); H = Nn.shape[1]
    st = L.load_state(arm, T)                              # 出発状態
    d, a = L.load_ref(arm)
    nb, nb_task = L.nbeta_table(arm)
    nb_fill = nb.copy(); nb_fill[np.isnan(nb_fill)] = np.nanmedian(nb)
    rng = np.random.default_rng(5)
    log(f"== {arm} T={T} (start end0[{T-1}], redraw task {T+1}), R {R}, self-test {z['selftest']}")
    alive0 = np.where(st.alive)[0]
    # 死んだ標的（引き直しの終端で k=0）は N=0 のまま入っているので、生存で条件づけた版も出す
    alive_frac = al.mean(0)
    mu_rd = Nn.mean(0); var_rd = Nn.var(0, ddof=1); F_rd = np.where(mu_rd > 0, var_rd / np.maximum(mu_rd, 1e-12), np.nan)
    muB = Bn.mean(0); varB = Bn.var(0, ddof=1); F_B = np.where(muB > 0, varB / np.maximum(muB, 1e-12), np.nan)
    maxN = Nn.max(0)
    log(f"[rd] 標的 μ^rd 中央値 {np.nanmedian(mu_rd[alive0]):.3f} q90 {np.nanquantile(mu_rd[alive0], .9):.3f} 最大 {mu_rd.max():.3f}; F^rd 中央値 {np.nanmedian(F_rd[alive0]):.3f} q90 {np.nanquantile(F_rd[alive0], .9):.3f} 最大 {np.nanmax(F_rd):.3f}; F^rd>1 の割合 {np.nanmean(F_rd[alive0] > 1):.2f}; 新生 F 中央値 {np.nanmedian(F_B[alive0]):.3f}; 課題末に死ぬ標的の割合 中央値 {1-np.median(alive_frac[alive0]):.3f}")
    pooled = Nn[:, alive0].reshape(-1)
    log(f"[rd] プール: 平均 {pooled.mean():.3f} Fano {pooled.var(ddof=1)/pooled.mean():.3f} 最大 {pooled.max()} P(N≥3) {(pooled>=3).mean():.4f} (Poisson 同平均 {L.poisson_tail(pooled.mean(), 3):.4f}) P(N≥5) {(pooled>=5).mean():.5f} ({L.poisson_tail(pooled.mean(),5):.2e}) P(N≥10) {(pooled>=10).mean():.2e}")
    # ---- (1) F の上界（出発状態の材料 + μ^rd） ----
    inputs = np.unique(st.Tin[:, alive0].reshape(-1)); om, omf = L.omega_inputs(st, inputs); pos = {int(n): q for q, n in enumerate(inputs)}
    Fb = np.full(H, np.nan); Fb_cos = np.full(H, np.nan); pv = np.full(H, np.nan); Fb_rdcond = np.full(H, np.nan)
    for i in alive0:
        cand = alive0[alive0 != i]; qs = [pos[int(n)] for n in st.Tin[:, i]]
        ob = np.mean([om[q] for q in qs], axis=0)[np.ix_(cand, cand)]; cc = st.cos[np.ix_(cand, cand)]
        ju = np.triu_indices(len(cand), 1); rho = (cc * ob)[ju]
        mu = mu_rd[i]
        if mu <= 0:
            continue
        Fb[i], _, _ = L.fano_bound_from_rho(np.maximum(rho, 0), mu, len(cand))
        Fb_cos[i], _, _ = L.fano_bound_from_rho(np.maximum(cc[ju], 0), mu, len(cand))
        if np.isfinite(Fb[i]):
            if Fb[i] <= 1 + 1e-9:
                sim = rng.poisson(mu, size=(args.R_boot, R))
            else:
                sim = rng.negative_binomial(mu / (Fb[i] - 1), 1 / Fb[i], size=(args.R_boot, R))
            Fs = sim.var(1, ddof=1) / np.maximum(sim.mean(1), 1e-12); pv[i] = float((Fs >= F_rd[i]).mean())
    ok = np.isfinite(Fb) & np.isfinite(F_rd)
    res = dict(arm=arm, T=T, R=R, n_alive_start=int(len(alive0)),
               mu_rd=dict(median=float(np.nanmedian(mu_rd[alive0])), q90=float(np.nanquantile(mu_rd[alive0], .9)), max=float(mu_rd.max())),
               F_rd=dict(median=float(np.nanmedian(F_rd[alive0])), q90=float(np.nanquantile(F_rd[alive0], .9)), max=float(np.nanmax(F_rd)), share_gt1=float(np.nanmean(F_rd[alive0] > 1))),
               F_births_rd=dict(median=float(np.nanmedian(F_B[alive0])), q90=float(np.nanquantile(F_B[alive0], .9)), max=float(np.nanmax(F_B))),
               pooled=dict(mean=float(pooled.mean()), fano=float(pooled.var(ddof=1) / pooled.mean()), max=int(pooled.max()), P_ge3=float((pooled >= 3).mean()), P_ge5=float((pooled >= 5).mean()), P_ge10=float((pooled >= 10).mean()), poisson_ge3=L.poisson_tail(pooled.mean(), 3), poisson_ge5=L.poisson_tail(pooled.mean(), 5)),
               F_bound=dict(median=float(np.nanmedian(Fb[ok])), q90=float(np.nanquantile(Fb[ok], .9)), max=float(np.nanmax(Fb[ok])), median_cos_only=float(np.nanmedian(Fb_cos[ok]))),
               share_Frd_exceeds_bound=float((F_rd[ok] > Fb[ok]).mean()), nb_boot_share_p_lt_0p05=float(np.nanmean(pv[ok] < 0.05)), nb_boot_share_p_lt_0p01=float(np.nanmean(pv[ok] < 0.01)), n_targets=int(ok.sum()))
    log(f"[1] F 上界（出発状態の cos·ω̄ ＋ μ^rd）中央値 {res['F_bound']['median']:.2f} q90 {res['F_bound']['q90']:.2f} 最大 {res['F_bound']['max']:.2f} (cos だけ {res['F_bound']['median_cos_only']:.2f}); F^rd > 上界 の標的 {res['share_Frd_exceeds_bound']:.2f}; NB p<0.05 {res['nb_boot_share_p_lt_0p05']:.2f} p<0.01 {res['nb_boot_share_p_lt_0p01']:.2f} (n {ok.sum()})")
    worst = np.where(ok)[0][np.argsort(-(F_rd[ok] - Fb[ok]))[:5]]
    for i in worst:
        log(f"    超過の大きい標的 u{i}: μ^rd {mu_rd[i]:.3f} F^rd {F_rd[i]:.2f} 上界 {Fb[i]:.2f} p {pv[i]:.3f} max N {maxN[i]} k0 {int(st.k[i])}")
    # ---- (2) 裾 ----
    tab = {}
    for label, nstar in (("n_beta", nb_fill), ("ten", np.full(H, 10.0)), ("arm_min", np.full(H, np.nanmin(nb))), ("five", np.full(H, 5.0)), ("three", np.full(H, 3.0))):
        emp = np.array([(Nn[:, i] >= nstar[i]).mean() for i in range(H)])
        cb = np.array([L.cantelli(mu_rd[i], Fb[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        cr = np.array([L.cantelli(mu_rd[i], F_rd[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        po = np.array([L.poisson_tail(mu_rd[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        viol = int(np.nansum(emp[ok] > cb[ok]))
        tab[label] = dict(nstar_median=float(np.nanmedian(nstar[ok])), emp_max=float(np.nanmax(emp[ok])), emp_sum=float(np.nansum(emp[ok])), n_targets_emp_pos=int((emp[ok] > 0).sum()),
                          cantelli_bound_max=float(np.nanmax(cb[ok])), cantelli_bound_median=float(np.nanmedian(cb[ok])), cantelli_bound_sum=float(np.nansum(cb[ok])),
                          cantelli_Frd_max=float(np.nanmax(cr[ok])), poisson_max=float(np.nanmax(po[ok])), poisson_sum=float(np.nansum(po[ok])),
                          n_targets_emp_exceeds_cantelli_bound=viol, n_targets_emp_exceeds_poisson=int(np.nansum(emp[ok] > po[ok])))
        log(f"[2] n*={label} (中央値 {tab[label]['nstar_median']:.1f}): 経験 P^rd(N≥n*) 最大 {tab[label]['emp_max']:.4f} Σ_i {tab[label]['emp_sum']:.3f} (正の標的 {tab[label]['n_targets_emp_pos']}); Cantelli(F上界) 最大 {tab[label]['cantelli_bound_max']:.3g} 中央値 {tab[label]['cantelli_bound_median']:.3g} Σ_i {tab[label]['cantelli_bound_sum']:.3g}; Poisson 最大 {tab[label]['poisson_max']:.2g} Σ_i {tab[label]['poisson_sum']:.3g}; 経験 > Cantelli の標的 {viol}; 経験 > Poisson の標的 {tab[label]['n_targets_emp_exceeds_poisson']}")
    res["tails"] = tab
    # ---- (3) (G3): 累積 Δz の unit 間相関 対 1 歩の cos·ω ----
    dZ = Zend - Z0[None]                                     # [R,N,H]
    # 入力: 狭い標的（k ≤ 8、無ければ最も狭い 6 個）の閾値の入力 ＋ 無作為 12
    narrow = alive0[np.argsort(st.k[alive0])[:6]]
    tin = np.unique(st.Tin[:, narrow].reshape(-1)); rnd = rng.choice(np.setdiff1d(np.arange(st.N), tin), 12, replace=False)
    S = np.concatenate([tin, rnd]); omS, _ = L.omega_inputs(st, S)
    iu = np.triu_indices(len(alive0), 1)
    rows = []; all_emp = []; all_pred = []; all_cos = []; viol_cos = 0; npairs = 0; skew = []; kurt = []
    for q, n in enumerate(S):
        Xr = dZ[:, n, :][:, alive0]                          # [R, n_alive]
        Cm = np.corrcoef(Xr.T); emp = Cm[iu]
        pred = (st.cos[np.ix_(alive0, alive0)] * omS[q][np.ix_(alive0, alive0)])[iu]; cs = st.cos[np.ix_(alive0, alive0)][iu]
        all_emp.append(emp); all_pred.append(pred); all_cos.append(cs)
        se = 1 / math.sqrt(R)
        viol_cos += int((np.abs(emp) > np.abs(cs) + 2 * se).sum()); npairs += emp.size
        m = Xr.mean(0); s = Xr.std(0); zz = (Xr - m) / np.maximum(s, 1e-30)
        skew.append(np.median(np.abs((zz ** 3).mean(0)))); kurt.append(np.median((zz ** 4).mean(0) - 3))
        rows.append(dict(input=int(n), in_threshold_set=bool(n in tin), pearson_emp_vs_cos_omega=float(np.corrcoef(emp, pred)[0, 1]), pearson_emp_vs_cos=float(np.corrcoef(emp, cs)[0, 1]),
                         sign_agree_with_cos=float((np.sign(emp) == np.sign(cs)).mean()), abs_emp_median=float(np.median(np.abs(emp))), abs_pred_median=float(np.median(np.abs(pred))),
                         frac_abs_emp_gt_abs_cos_plus2se=float((np.abs(emp) > np.abs(cs) + 2 * se).mean()), slope_emp_on_pred=float(np.polyfit(pred, emp, 1)[0])))
    all_emp = np.concatenate(all_emp); all_pred = np.concatenate(all_pred); all_cos = np.concatenate(all_cos)
    g3 = dict(n_inputs=int(len(S)), n_threshold_inputs=int(len(tin)), R=R, pearson_all=float(np.corrcoef(all_emp, all_pred)[0, 1]), pearson_all_vs_cos=float(np.corrcoef(all_emp, all_cos)[0, 1]),
              slope_all=float(np.polyfit(all_pred, all_emp, 1)[0]), sign_agree_all=float((np.sign(all_emp) == np.sign(all_cos)).mean()),
              abs_emp_quantiles=np.quantile(np.abs(all_emp), [.5, .9, .99, 1.0]).tolist(), abs_pred_quantiles=np.quantile(np.abs(all_pred), [.5, .9, .99, 1.0]).tolist(),
              frac_abs_emp_gt_abs_cos_plus2se=float(viol_cos / npairs), mc_se=float(1 / math.sqrt(R)),
              cum_dz_abs_skew_median=float(np.median(skew)), cum_dz_excess_kurt_median=float(np.median(kurt)), per_input=rows)
    res["G3_cumulative_corr"] = g3
    log(f"[3] (G3) 累積 Δz（4,000 歩）の unit 間相関 対 1 歩の cos·ω: Pearson {g3['pearson_all']:.3f} (対 cos だけ {g3['pearson_all_vs_cos']:.3f}), 回帰の傾き {g3['slope_all']:.3f}, 符号一致 {g3['sign_agree_all']:.3f}; |ρ_emp| 分位 50/90/99/100 {np.round(g3['abs_emp_quantiles'],3).tolist()} 対 |cos·ω| {np.round(g3['abs_pred_quantiles'],3).tolist()}; |ρ_emp| > |cos|+2se の対の割合 {g3['frac_abs_emp_gt_abs_cos_plus2se']:.3f} (se {g3['mc_se']:.3f}); 累積 Δz の |歪度| 中央値 {g3['cum_dz_abs_skew_median']:.2f} 過剰尖度 {g3['cum_dz_excess_kurt_median']:.2f}")
    # 分散の比（累積 対 1 歩の式 × 4000）: ラベル項が歩数に比例して積もるか
    coef = L.ETA ** 2 / (st.N ** 2 * L.C_CLASSES)
    var1 = np.array([coef * np.diag(st.G) * ((st.Theta[n][:, None] * st.D) ** 2).sum(0) for n in S])    # [len S, H]
    varcum = dZ[:, S, :].var(0, ddof=1)                                                                  # [len S, H]
    ratio = (varcum / np.maximum(var1, 1e-300))[:, alive0]
    res["cum_var_over_onestep_var"] = dict(median=float(np.median(ratio)), q10=float(np.quantile(ratio, .1)), q90=float(np.quantile(ratio, .9)), note="SGD 全データ 1 歩の式の分散に対する累積 4,000 歩（Adam, batch16）の分散の比。独立和なら 4000 × (N/B) 程度の桁")
    log(f"[3] 累積分散 / 1 歩の式の分散: 中央値 {res['cum_var_over_onestep_var']['median']:.3g} (q10 {res['cum_var_over_onestep_var']['q10']:.3g} q90 {res['cum_var_over_onestep_var']['q90']:.3g})")
    # ---- (5) 等しい p の近似 (4.5) 対 候補ごとの p̂_ij を入れた (4.4) そのもの（閾値が違う 2 変量正規の Cov） ----
    phat = isb.mean(0)                                        # [H(i), H(j)] 引き直しの測度での周辺確率
    Fhet = np.full(H, np.nan); n_cand_pos = np.zeros(H, int)
    for i in alive0:
        pj = phat[i].copy(); pj[i] = 0.0
        js = np.where(pj > 0)[0]
        if len(js) < 2 or mu_rd[i] <= 0:
            continue
        n_cand_pos[i] = len(js)
        qs = [pos[int(n)] for n in st.Tin[:, i]]
        ob = np.mean([om[q] for q in qs], axis=0)
        aj = {int(j): L.Phibar_inv(min(pj[j], 0.499)) for j in js}
        cov_sum = 0.0
        for x in range(len(js)):
            for y_ in range(x + 1, len(js)):
                j, k = int(js[x]), int(js[y_])
                rho = st.cos[j, k] * ob[j, k]
                if rho <= 0:
                    continue
                cov_sum += 2 * (L.joint_upper_1d(aj[j], aj[k], min(rho, 0.999)) - pj[j] * pj[k])
        Fhet[i] = 1.0 + (cov_sum - float((pj ** 2).sum())) / mu_rd[i]
    okh = np.isfinite(Fhet) & ok
    res["F_bound_unequal_p"] = dict(median=float(np.nanmedian(Fhet[okh])), q90=float(np.nanquantile(Fhet[okh], .9)), max=float(np.nanmax(Fhet[okh])),
                                    share_Frd_exceeds=float((F_rd[okh] > Fhet[okh]).mean()), n_targets=int(okh.sum()), n_cand_pos_median=float(np.median(n_cand_pos[okh])),
                                    note="(4.4) を候補ごとの p̂_ij（引き直し R 本の周辺）と a_j = Φ̄⁻¹(p̂_ij) で評価。p̂=0 の候補は 0 とした（標本の和）")
    log(f"[5] (4.4) を候補ごとの p̂_ij で評価した F: 中央値 {res['F_bound_unequal_p']['median']:.2f} q90 {res['F_bound_unequal_p']['q90']:.2f} 最大 {res['F_bound_unequal_p']['max']:.2f}（等しい p の (4.5): 中央値 {res['F_bound']['median']:.2f}）; F^rd > この値 の標的 {res['F_bound_unequal_p']['share_Frd_exceeds']:.2f}; p̂>0 の候補数 中央値 {res['F_bound_unequal_p']['n_cand_pos_median']:.0f}")
    # ---- (6) 共通原因の検査: 標的の実現した閾値集合 T_i(t+1)（と C_i）で条件づけた残りの分散 ----
    # Z_end から t+1 の T_i と C_i を再計算（なめらかなゲート、stoch と同じ）
    Kv = st.Kv
    Tkeys = np.empty((R, H), dtype=object); Ckey = np.zeros((R, H), int); kend = np.zeros((R, H), int)
    for r in range(R):
        Zr = Zend[r]; tau = 0.1 * Zr.std(0) + 1e-12; s = 0.5 * (1 + np.tanh(0.5 * Zr / tau)); Dp = (1 - a) * s * (1 - s) / tau
        E = Kv[:, None] * Dp; Tin = np.argsort(-E, axis=0)[:3, :]; Or = Zr > 0; alr = Or.any(0)
        for i in alive0:
            Tkeys[r, i] = tuple(sorted(Tin[:, i].tolist()))
            thr = Or[Tin[:, i], :].sum(0); thr[i] = 0; thr[~alr] = 0
            Ckey[r, i] = int((thr >= 2).sum()); kend[r, i] = int(Or[:, i].sum())
    def within_decomp(keys, vals):
        groups = {}
        for kk_, v in zip(keys, vals):
            groups.setdefault(kk_, []).append(v)
        tot_var = np.var(vals, ddof=1); mean = np.mean(vals)
        num = 0.0; den = 0; between = 0.0; sizes = []
        for g in groups.values():
            g = np.array(g, float); sizes.append(len(g))
            if len(g) >= 2:
                num += (len(g) - 1) * g.var(ddof=1); den += len(g) - 1
            between += len(g) * (g.mean() - mean) ** 2
        within_var = num / den if den > 0 else np.nan
        return dict(n_groups=len(groups), max_group=max(sizes), within_var=within_var, within_fano=within_var / mean if mean > 0 else np.nan,
                    share_between=between / (len(vals) * tot_var) if tot_var > 0 else np.nan, n_in_groups_ge2=den)
    rowsT, rowsC, rowsTk = [], [], []
    for i in alive0:
        if mu_rd[i] <= 0:
            continue
        vals = Nn[:, i].astype(float)
        rowsT.append(within_decomp([Tkeys[r, i] for r in range(R)], vals))
        rowsC.append(within_decomp([int(Ckey[r, i]) for r in range(R)], vals))
        rowsTk.append(within_decomp([(Tkeys[r, i], int(kend[r, i])) for r in range(R)], vals))
    def summ(rows):
        f = lambda key: [x[key] for x in rows if np.isfinite(x[key])]
        return dict(n_targets=len(rows), n_groups_median=float(np.median(f("n_groups"))), max_group_median=float(np.median(f("max_group"))),
                    within_fano_median=float(np.median(f("within_fano"))), within_fano_q90=float(np.quantile(f("within_fano"), .9)),
                    share_between_median=float(np.median(f("share_between"))), share_within_fano_gt1=float(np.mean(np.array(f("within_fano")) > 1)))
    res["common_cause"] = dict(by_T_i=summ(rowsT), by_C_i=summ(rowsC), by_T_i_and_k=summ(rowsTk),
                               note="同じ出発状態からの引き直し R 本を、標的 i の t+1 の実現した閾値集合 T_i（3 入力の組）／閾値のプール C_i／(T_i, k_i) で群に分け、群内の分散（プール）と群間の分の割合。群内 Fano ≤ 1 なら、共誕は標的の実現した状態（共通原因）を与えれば独立以下")
    for nm in ("by_T_i", "by_C_i", "by_T_i_and_k"):
        c = res["common_cause"][nm]
        log(f"[6] {nm}: 群数 中央値 {c['n_groups_median']:.0f} 最大群 {c['max_group_median']:.0f}; 群内 Fano 中央値 {c['within_fano_median']:.3f} q90 {c['within_fano_q90']:.3f} (>1 の標的 {c['share_within_fano_gt1']:.2f}); 群間の分の割合 中央値 {c['share_between_median']:.3f}")
    # ---- (4) 実際の次の状態 ----
    st1 = L.load_state(arm, T + 1); isb1, _, _, _ = L.bridges(st1); N1 = isb1.sum(1)
    top = np.argsort(-N1)[:5]
    act = []
    for i in top:
        pr = float((Nn[:, i] >= N1[i]).mean()) if N1[i] > 0 else float("nan")
        act.append(dict(unit=int(i), actual_N=int(N1[i]), mu_rd=float(mu_rd[i]), F_rd=float(F_rd[i]) if np.isfinite(F_rd[i]) else None, max_rd=int(maxN[i]), P_rd_ge_actual=pr, k_start=int(st.k[i]), k_actual_end=int(st1.k[i])))
        log(f"[4] 実際の T+1={T+1} で N={N1[i]} の u{i}: 引き直し μ {mu_rd[i]:.3f} F {F_rd[i]:.2f} 最大 {maxN[i]} P^rd(N ≥ {N1[i]}) {pr:.4f}; k 出発 {int(st.k[i])} → 実際 {int(st1.k[i])}")
    res["actual_next_state_top5"] = act
    res["actual_vs_redraw_total_pairs"] = dict(actual=int(isb1.sum()), redraw_mean=float(isb.sum((1, 2)).mean()), redraw_q=np.quantile(isb.sum((1, 2)), [.05, .5, .95]).tolist())
    log(f"[4] 橋対の総数: 実際 {int(isb1.sum())}, 引き直し 平均 {res['actual_vs_redraw_total_pairs']['redraw_mean']:.1f} (5/50/95% {np.round(res['actual_vs_redraw_total_pairs']['redraw_q'],1).tolist()})")
    out = f"{WT}/results/cobirth_1013/v1_redraw_{arm}_T{T}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=float)
    open(f"{RAW}/v1_redraw_analyze_{arm}_T{T}.log", "w").write("\n".join(LOG) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
