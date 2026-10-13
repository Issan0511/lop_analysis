#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v1_verify.py — cobirth_1013 V1（検証役）。D1 の DERIVATION.md を独立に検算する。

Part A: 補題 1 (1.3) の独立な再導出の数値検算。D1 と別の状態（a0.3 T70、a0.1 T25、a0.5 T90。M1 の約束 T = end0[T-1]）、
        別の入力の選び方、別の seed、MC は閉じた形でなく torch autograd（訓練と同じ平均 CE・where(z>0,1,a)）で勾配を取る。
        加えて、バッチ 16 を無作為に引いたときの全共分散を E_B[Cov_Y] と Cov_B(E_Y) に分け、第 2 項の大きさを測る。
Part B: 補題 3（Plackett）と上界 (4.2)(4.3) の独立な数値検算（1 次元積分 vs 累積積分 vs MC）。
Part C: 空虚さの検査。相 2 の全状態で、実測の材料（cos(v^c)・ω_jk(T_i)・μ_i（M1）・n_β（p172c））を (4.5)(4.6) に入れ、
        標的ごとの F の上界と ρ_R の上界を出し、M1 の実測 F_i・Poisson・閾値 14〜32% と並べる。
        ω の床（a² の項）だけで ω がどこまで決まるかも測る。
出力: results/cobirth_1013/v1_verify.json、obsidian-research-data/cobirth_1013/v1/v1_verify.log、v1_partC_<arm>.npz
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1_lib as L  # noqa: E402

WT = "/home/issan/Projects/claude/wt/cobirth_1013"
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/v1"
os.makedirs(RAW, exist_ok=True)
LOG = []


def log(s):
    print(s, flush=True); LOG.append(s)


# ======================= Part A =======================
def pick_inputs_v1(st, n_inputs, rng):
    cands = np.where((st.k >= 2) & (st.k <= 6))[0]
    if len(cands) == 0:
        cands = np.where((st.k >= 1) & (st.k <= 12))[0]
    if len(cands) == 0:                                  # a0.5 の相 2 は狭い unit が無い: 最も狭い生きた unit 5 個から
        al = np.where(st.alive)[0]; cands = al[np.argsort(st.k[al])[:5]]
    tgt = int(cands[rng.integers(len(cands))])
    Tn = st.Tin[:, tgt].tolist()
    On = np.where(st.O[:, tgt])[0].tolist()[:4]
    chosen = list(dict.fromkeys(Tn + On))
    rest = [int(m) for m in rng.permutation(st.N) if int(m) not in chosen][: n_inputs - len(chosen)]
    return np.array(chosen + rest, int), tgt, chosen


def formula_cov(st, S, coef=None):
    """C[(j,n),(k,n')] = coef ⟨v_j^c, v_k^c⟩ Σ_m Θ_{S_n m} Θ_{S_n' m} d_jm d_km （自分で einsum）"""
    if coef is None:
        coef = L.ETA ** 2 / (st.N ** 2 * L.C_CLASSES)
    ThS = st.Theta[S]                                   # [s,N]
    M = np.einsum("nm,mj,pm,mk->jnkp", ThS, st.D, ThS, st.D, optimize=True)   # [H,s,H,s]
    Cv = st.G[:, None, :, None] * M
    s = len(S)
    return coef * Cv.reshape(st.H * s, st.H * s)


def mc_autograd(st, S, R, rng):
    import torch
    torch.set_num_threads(1)
    X = torch.tensor(st.X); W1 = torch.tensor(st.W1, requires_grad=True); b1 = torch.tensor(st.b1, requires_grad=True)
    W2 = torch.tensor(st.W2); b3 = torch.tensor(st.b3); XtS = torch.tensor(st.Xt[S])
    out = np.empty((R, st.H, len(S)))
    for r in range(R):
        y = torch.tensor(rng.integers(0, L.C_CLASSES, size=st.N))
        z = X @ W1.T + b1
        h = torch.where(z > 0, z, st.a * z)
        logits = h @ W2.T + b3
        loss = (logits.logsumexp(-1) - logits.gather(1, y[:, None]).squeeze(1)).mean()
        gW1, gb1 = torch.autograd.grad(loss, (W1, b1))
        dWt = -L.ETA * torch.cat([gW1, gb1[:, None]], 1)         # [H,785]
        out[r] = (dWt @ XtS.T).numpy()                            # [H,s]
    return out.reshape(R, -1)


def compare(C_mc, C_ex, R, name):
    num = np.linalg.norm(C_mc - C_ex); den = np.linalg.norm(C_ex)
    diag = np.diag(C_ex)
    pred = math.sqrt((np.outer(diag, diag) + C_ex ** 2).sum() / (R - 1)) / den
    ok = diag > 0
    rel_diag = np.abs(np.diag(C_mc)[ok] - diag[ok]) / diag[ok]
    r = dict(rel_frobenius_error=float(num / den), predicted_mc_rel_frobenius=float(pred), ratio=float(num / den / pred),
             diag_rel_err_median=float(np.median(rel_diag)), diag_rel_err_max=float(rel_diag.max()),
             diag_mc_sd_pred=float(math.sqrt(2.0 / (R - 1))), n_coords=int(C_ex.shape[0]), R=int(R))
    log(f"  [{name}] rel Frobenius {r['rel_frobenius_error']:.4f} (MC 予測 {pred:.4f}, 比 {r['ratio']:.2f}); 対角 相対誤差 中央値 {r['diag_rel_err_median']:.4f} 最大 {r['diag_rel_err_max']:.4f} (MC sd 予測 {r['diag_mc_sd_pred']:.4f})")
    return r


def corr_check(st, S, C_mc, C_ex, R):
    s = len(S); alive = np.where(st.alive)[0]
    om, _ = L.omega_inputs(st, S)
    diffs, dex = [], []
    for n in range(s):
        idx = alive * s + n
        sm = C_mc[np.ix_(idx, idx)]; se = C_ex[np.ix_(idx, idx)]
        rm = sm / np.sqrt(np.outer(np.diag(sm), np.diag(sm))); re = se / np.sqrt(np.outer(np.diag(se), np.diag(se)))
        fac = st.cos[np.ix_(alive, alive)] * om[n][np.ix_(alive, alive)]
        iu = np.triu_indices(len(alive), 1)
        diffs.append((rm - fac)[iu]); dex.append(np.abs(re - fac)[iu].max())
    diffs = np.concatenate(diffs)
    r = dict(max_abs_diff_mc_vs_cos_omega=float(np.abs(diffs).max()), rms_diff=float(np.sqrt((diffs ** 2).mean())),
             mc_se=float(1 / math.sqrt(R)), max_abs_diff_exact_vs_cos_omega=float(max(dex)), n_pairs=int(diffs.size))
    log(f"  [corr] ρ_MC − cos·ω: rms {r['rms_diff']:.4f} 最大 {r['max_abs_diff_mc_vs_cos_omega']:.4f} (MC se {r['mc_se']:.4f}); 厳密式 ρ 対 cos·ω 最大 {r['max_abs_diff_exact_vs_cos_omega']:.1e}")
    return r


def minibatch_decomp(st, S, R, rng, B=16):
    """全共分散（B と Y を同時に引く）= E_B[Cov_Y|B] + Cov_B(E_Y|B)。第 1 項の式: (η²/(B N C)) ⟨v^c⟩ Σ_m ΘΘ d d。"""
    s = len(S); ThS = st.Theta[S]; PW = st.P @ st.W2          # ⟨v_j, p_m⟩
    tot = np.empty((R, st.H, s)); mu_only = np.empty((R, st.H, s))
    muvec = (PW - st.W2.mean(0)[None, :]) * st.D               # d_jm ⟨v_j^c, p_m − u⟩  [N,H]
    for r in range(R):
        b = rng.choice(st.N, B, replace=False); y = rng.integers(0, L.C_CLASSES, size=B)
        delta = (PW[b] - st.W2[y, :]) * st.D[b]
        tot[r] = (-(L.ETA / B) * ThS[:, b] @ delta).T
        mu_only[r] = (-(L.ETA / B) * ThS[:, b] @ muvec[b]).T
    C_tot = np.cov(tot.reshape(R, -1).T); C_mu = np.cov(mu_only.reshape(R, -1).T)
    C_lab = formula_cov(st, S, coef=L.ETA ** 2 / (B * st.N * L.C_CLASSES))
    r = dict(B=B, R=R, frob_label_term=float(np.linalg.norm(C_lab)), frob_mu_sampling_term_mc=float(np.linalg.norm(C_mu)),
             frob_total_mc=float(np.linalg.norm(C_tot)),
             ratio_mu_term_over_label_term=float(np.linalg.norm(C_mu) / np.linalg.norm(C_lab)),
             rel_err_total_vs_label_plus_mu=float(np.linalg.norm(C_tot - C_lab - C_mu) / np.linalg.norm(C_tot)),
             rel_err_total_vs_label_only=float(np.linalg.norm(C_tot - C_lab) / np.linalg.norm(C_tot)))
    # 相関の形: 全共分散の相関 対 cos·ω
    alive = np.where(st.alive)[0]; om, _ = L.omega_inputs(st, S); d = []
    for n in range(s):
        idx = alive * s + n; sm = C_tot[np.ix_(idx, idx)]
        rm = sm / np.sqrt(np.outer(np.diag(sm), np.diag(sm)))
        fac = st.cos[np.ix_(alive, alive)] * om[n][np.ix_(alive, alive)]
        d.append((rm - fac)[np.triu_indices(len(alive), 1)])
    d = np.concatenate(d)
    r["rms_corr_total_vs_cos_omega"] = float(np.sqrt((d ** 2).mean())); r["max_corr_total_vs_cos_omega"] = float(np.abs(d).max())
    log(f"  [minibatch B={B}] ‖Cov_B(E_Y)‖/‖E_B Cov_Y‖ = {r['ratio_mu_term_over_label_term']:.3f}; 全共分散 対 (ラベル項+μ項) 相対誤差 {r['rel_err_total_vs_label_plus_mu']:.3f}, 対 ラベル項だけ {r['rel_err_total_vs_label_only']:.3f}; 全共分散の相関 − cos·ω: rms {r['rms_corr_total_vs_cos_omega']:.3f} 最大 {r['max_corr_total_vs_cos_omega']:.3f} (MC se {1/math.sqrt(R):.3f})")
    return r


def part_A(states, R, n_inputs, seed):
    out = {}
    for arm, T in states:
        rng = np.random.default_rng(seed + T)
        st = L.load_state(arm, T)
        Phi_ = np.where(st.Z > 0, st.Z, st.a * st.Z); lg = Phi_ @ st.W2.T + st.b3; lg -= lg.max(1, keepdims=True); P = np.exp(lg); st.P = P / P.sum(1, keepdims=True)
        S, tgt, chosen = pick_inputs_v1(st, n_inputs, rng)
        log(f"[A] {arm} T={T} (end0[{T-1}]) a={st.a} alive {int(st.alive.sum())} 標的 {tgt} k={int(st.k[tgt])} 入力 {len(S)} (T_i∪O_i {chosen})")
        C_ex = formula_cov(st, S)
        t0 = time.time(); mc = mc_autograd(st, S, R, rng); C_mc = np.cov(mc.T)
        res = dict(arm=arm, T=T, end0_index=T - 1, a=st.a, target=tgt, inputs=S.tolist(), target_inputs=chosen, R=R, mc_seconds=time.time() - t0)
        res["fullbatch_autograd"] = compare(C_mc, C_ex, R, f"autograd fullbatch R={R}")
        res["corr"] = corr_check(st, S, C_mc, C_ex, R)
        mu_ex = (-(L.ETA / st.N) * st.Theta[S] @ ((st.P @ st.W2 - st.W2.mean(0)[None, :]) * st.D)).T.reshape(-1)
        se = mc.std(0, ddof=1) / math.sqrt(R); ok = se > 0; z = np.abs(mc.mean(0)[ok] - mu_ex[ok]) / se[ok]
        res["mean_check"] = dict(max_abs_z=float(z.max()), frac_gt3=float((z > 3).mean()))
        log(f"  [mean] |MC 平均 − μ|/se 最大 {z.max():.2f}, >3 の割合 {(z>3).mean():.4f}")
        res["minibatch"] = minibatch_decomp(st, S, R, rng)
        out[f"{arm}_T{T}"] = res
    return out


# ======================= Part B =======================
def part_B():
    res = {}
    rng = np.random.default_rng(31)
    for p in (0.003, 0.01):
        a = L.Phibar_inv(p); rows = []
        r_grid, cum = L.plackett_cum_table(a)
        for rho in (0.05, 0.15, 0.3, 0.5, 0.8):
            joint = L.joint_upper_1d(a, a, rho); cov1d = joint - p * p
            cov_pl = float(np.interp(rho, r_grid, cum))
            n = 4_000_000; X = rng.standard_normal(n); Y = rho * X + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
            jm = float(((X > a) & (Y > a)).mean()); se = math.sqrt(jm * (1 - jm) / n)
            b42 = L.cov_bound_42(a, a, rho); b43 = L.R_bound_43(a, a, rho)
            gordon_ok = bool(L.Phibar(a) >= L.phi(a) * a / (1 + a * a))
            rows.append(dict(rho=rho, a=a, p=p, cov_1d_integral=cov1d, cov_plackett_cum=cov_pl, cov_mc=jm - p * p, cov_mc_se=se,
                             z_mc=(jm - p * p - cov1d) / se, rel_diff_1d_vs_plackett=abs(cov1d - cov_pl) / cov1d,
                             bound42=b42, bound42_over_exact=b42 / cov1d, R_exact=cov1d / (p * p), R_bound43=b43,
                             bound43_over_Rexact=b43 / (cov1d / p / p), bound42_holds=bool(b42 >= cov1d), bound43_holds=bool(b43 >= cov1d / p / p), gordon_holds=gordon_ok))
            log(f"[B] p={p} ρ={rho}: Cov 1D積分 {cov1d:.4e} / Plackett累積 {cov_pl:.4e} (相対差 {abs(cov1d-cov_pl)/cov1d:.1e}) / MC {jm-p*p:.4e} (z {(jm-p*p-cov1d)/se:+.2f}); (4.2)/厳密 {b42/cov1d:.2f}; R 厳密 {cov1d/p/p:.2f} 対 (4.3) {b43:.2f}; 成立 {b42>=cov1d} {b43>=cov1d/p/p}")
        res[f"p{p}"] = rows
    # 負の ρ で Cov ≤ 0（Slepian）
    a = L.Phibar_inv(0.003); neg = [L.joint_upper_1d(a, a, rho) - 0.003 ** 2 for rho in (-0.1, -0.3, -0.6)]
    res["negative_rho_cov"] = neg
    log(f"[B] ρ<0 の Cov: {['%.2e' % v for v in neg]} (すべて ≤ 0 か: {all(v <= 1e-12 for v in neg)})")
    return res


# ======================= Part C =======================
def part_C(arm, m1, Ts, nb, nb_T, R_boot, rng):
    d, a = L.load_ref(arm)
    pt = m1["arms"][{"a0.1": "a01", "a0.3": "a03", "a0.5": "a05"}[arm]]["presence"]["per_target"]
    mu_m1 = np.array(pt["mu"]); F_m1 = np.array(pt["fano"]); nal = np.array(pt["n_alive"]); mx_m1 = np.array(pt["max"])
    H = 100; nT = len(Ts)
    Fb_mean = np.full((nT, H), np.nan); Fb_max = np.full((nT, H), np.nan); Fb_cos = np.full((nT, H), np.nan); Fb_43 = np.full((nT, H), np.nan)
    meanR = np.full((nT, H), np.nan); frac_rho_gt03 = np.full((nT, H), np.nan); n_c_arr = np.full((nT, H), np.nan)
    om_med = np.full(nT, np.nan); om_floor_med = np.full(nT, np.nan); om_rel_overlap_med = np.full(nT, np.nan); om_corr_floor = np.full(nT, np.nan)
    rho_q = np.full((nT, 5), np.nan); cos_abs_q = np.full((nT, 5), np.nan)
    t0 = time.time()
    for ti, T in enumerate(Ts):
        st = L.load_state(arm, T, d, a)
        alive = np.where(st.alive)[0]
        inputs = np.unique(st.Tin[:, alive].reshape(-1))
        om, omf = L.omega_inputs(st, inputs)                   # [n_in,H,H]
        pos = {int(n): q for q, n in enumerate(inputs)}
        iu = np.triu_indices(len(alive), 1)
        # ω の床の検査（生きた unit の対 × 閾値の入力）
        o_all = np.concatenate([om[q][np.ix_(alive, alive)][iu] for q in range(len(inputs))])
        of_all = np.concatenate([omf[q][np.ix_(alive, alive)][iu] for q in range(len(inputs))])
        om_med[ti] = np.median(o_all); om_floor_med[ti] = np.median(of_all)
        om_rel_overlap_med[ti] = np.median((o_all - of_all) / o_all); om_corr_floor[ti] = np.corrcoef(o_all, of_all)[0, 1]
        cs = st.cos[np.ix_(alive, alive)][iu]
        cos_abs_q[ti] = np.quantile(np.abs(cs), [.1, .5, .9, .99, 1.0])
        rho_all = []
        for i in alive:
            cand = alive[alive != i]
            if len(cand) < 2:
                continue
            qs = [pos[int(n)] for n in st.Tin[:, i]]
            ob = np.mean([om[q] for q in qs], axis=0)[np.ix_(cand, cand)]
            obx = np.max([om[q] for q in qs], axis=0)[np.ix_(cand, cand)]
            cc = st.cos[np.ix_(cand, cand)]
            ju = np.triu_indices(len(cand), 1)
            rho_mean = (cc * ob)[ju]; rho_max = (cc * obx)[ju]; rho_cos = cc[ju]
            rho_all.append(rho_mean)
            mu = mu_m1[i]; n_c = len(cand); n_c_arr[ti, i] = n_c
            Fm, F43, aa = L.fano_bound_from_rho(np.maximum(rho_mean, 0), mu, n_c)
            Fx, _, _ = L.fano_bound_from_rho(np.maximum(rho_max, 0), mu, n_c)
            Fc, _, _ = L.fano_bound_from_rho(np.maximum(rho_cos, 0), mu, n_c)
            Fb_mean[ti, i] = Fm; Fb_max[ti, i] = Fx; Fb_cos[ti, i] = Fc; Fb_43[ti, i] = F43
            meanR[ti, i] = (Fm - 1) / (mu * (n_c - 1) / n_c) if mu > 0 else np.nan
            frac_rho_gt03[ti, i] = float((rho_mean > 0.3).mean())
        rho_all = np.concatenate(rho_all); rho_q[ti] = np.quantile(rho_all, [.1, .5, .9, .99, 1.0])
        if ti % 15 == 0 or ti == nT - 1:
            log(f"  [C {arm}] T={T}: alive {len(alive)}, inputs {len(inputs)}, ω 中央値 {om_med[ti]:.3f} (床 {om_floor_med[ti]:.3f}, 重なり分の割合 中央値 {om_rel_overlap_med[ti]:.3f}, corr(ω,床) {om_corr_floor[ti]:.3f}), ρ=cos·ω̄ 分位 {np.round(rho_q[ti],3).tolist()}, F上界 中央値 {np.nanmedian(Fb_mean[ti]):.2f} ({time.time()-t0:.0f}s)")
    # ---- 標的ごとの集計 ----
    with np.errstate(invalid="ignore"):
        Fb_i = np.nanmean(Fb_mean, 0); Fb_i_max = np.nanmax(Fb_mean, 0); Fbx_i = np.nanmean(Fb_max, 0); Fbc_i = np.nanmean(Fb_cos, 0); Fb43_i = np.nanmean(Fb_43, 0)
        meanR_i = np.nanmean(meanR, 0)
    nb_fill = nb.copy(); med_nb = np.nanmedian(nb); nb_fill[np.isnan(nb_fill)] = med_nb
    res = dict(arm=arm, a=a, n_states=nT, Ts=[int(t) for t in Ts], nbeta_state_T=nb_T, nbeta_filled=int(np.isnan(nb).sum()),
               omega_median_over_states=float(np.nanmedian(om_med)), omega_floor_median=float(np.nanmedian(om_floor_med)),
               omega_overlap_share_median=float(np.nanmedian(om_rel_overlap_med)), corr_omega_floor_median=float(np.nanmedian(om_corr_floor)),
               rho_quantiles_10_50_90_99_100=np.nanmedian(rho_q, 0).tolist(), abs_cos_quantiles=np.nanmedian(cos_abs_q, 0).tolist())
    log(f"[C {arm}] ω 中央値 {res['omega_median_over_states']:.3f}, 床だけの ω 中央値 {res['omega_floor_median']:.3f}, 重なり分の割合 {res['omega_overlap_share_median']:.3f}, corr(ω, 床) {res['corr_omega_floor_median']:.3f}")
    log(f"[C {arm}] ρ = cos·ω̄(T_i) 分位 10/50/90/99/100: {np.round(res['rho_quantiles_10_50_90_99_100'],3).tolist()}; |cos| 分位 {np.round(res['abs_cos_quantiles'],3).tolist()}")
    # 表: F の上界 対 実測 F
    ok = np.isfinite(Fb_i) & (mu_m1 > 0)
    res["F_bound_per_target"] = dict(median=float(np.nanmedian(Fb_i[ok])), q90=float(np.nanquantile(Fb_i[ok], .9)), max=float(np.nanmax(Fb_i[ok])),
                                     median_using_max_omega=float(np.nanmedian(Fbx_i[ok])), median_cos_only=float(np.nanmedian(Fbc_i[ok])),
                                     median_bound43=float(np.nanmedian(Fb43_i[ok])), mean_R_median=float(np.nanmedian(meanR_i[ok])))
    res["F_measured_M1"] = dict(median=float(np.median(F_m1)), q90=float(np.quantile(F_m1, .9)), max=float(F_m1.max()))
    share_exceed = float((F_m1[ok] > Fb_i[ok]).mean())
    # NB bootstrap: 真の F が上界ならば実測 F がそれ以上になる確率
    pvals = np.full(H, np.nan)
    for i in np.where(ok)[0]:
        mu, Fb, n = mu_m1[i], Fb_i[i], int(nal[i])
        if Fb <= 1.0 + 1e-9:
            sim = rng.poisson(mu, size=(R_boot, n))
        else:
            rr = mu / (Fb - 1.0); pp = 1.0 / Fb
            sim = rng.negative_binomial(rr, pp, size=(R_boot, n))
        Fs = sim.var(1, ddof=1) / np.maximum(sim.mean(1), 1e-12)
        pvals[i] = float((Fs >= F_m1[i]).mean())
    res["measured_F_exceeds_bound_share"] = share_exceed
    res["nb_bootstrap"] = dict(R=R_boot, share_p_lt_0p05=float(np.nanmean(pvals[ok] < 0.05)), share_p_lt_0p01=float(np.nanmean(pvals[ok] < 0.01)),
                               n_targets=int(ok.sum()), note="H0: N_i(T) iid 負の二項（平均 μ_i, Fano = F の上界）。時系列の分散 ≥ 条件つき分散なので超過は模型 G の反証ではなく測度の差を含む")
    log(f"[C {arm}] F 上界 標的ごと 中央値 {res['F_bound_per_target']['median']:.2f} q90 {res['F_bound_per_target']['q90']:.2f} 最大 {res['F_bound_per_target']['max']:.2f} (cos だけ {res['F_bound_per_target']['median_cos_only']:.2f}, (4.3) 版 {res['F_bound_per_target']['median_bound43']:.2f}); 実測 F (M1) 中央値 {res['F_measured_M1']['median']:.2f} q90 {res['F_measured_M1']['q90']:.2f} 最大 {res['F_measured_M1']['max']:.2f}; 実測 > 上界 の標的 {share_exceed:.2f}; NB ブートストラップ p<0.05 の割合 {res['nb_bootstrap']['share_p_lt_0p05']:.2f}")
    # Cantelli
    thr = {"a0.1": (0.32, 0.14), "a0.3": (0.26, 0.11), "a0.5": (0.19, 0.08)}[arm]
    tab = {}
    for label, nstar in (("n_beta", nb_fill), ("ten", np.full(H, 10.0)), ("arm_min", np.full(H, np.nanmin(nb))), ("arm_median", np.full(H, med_nb)), ("nbeta_x0.5", 0.5 * nb_fill)):
        rb = np.array([L.cantelli(mu_m1[i], Fb_i[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        rm = np.array([L.cantelli(mu_m1[i], F_m1[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        rind = np.array([L.cantelli(mu_m1[i], 1.0, nstar[i]) if ok[i] else np.nan for i in range(H)])
        pois = np.array([L.poisson_tail(mu_m1[i], nstar[i]) if ok[i] else np.nan for i in range(H)])
        emp = np.array([np.nan] * H)
        tab[label] = dict(nstar_median=float(np.nanmedian(nstar[ok])), nstar_min=float(np.nanmin(nstar[ok])),
                          rhoR_bound_max=float(np.nanmax(rb)), rhoR_bound_median=float(np.nanmedian(rb)), rhoR_bound_sum=float(np.nansum(rb)),
                          rhoR_measuredF_max=float(np.nanmax(rm)), rhoR_measuredF_median=float(np.nanmedian(rm)),
                          rhoR_indep_F1_max=float(np.nanmax(rind)), rhoR_indep_F1_median=float(np.nanmedian(rind)),
                          poisson_max=float(np.nanmax(pois)), poisson_median=float(np.nanmedian(pois)),
                          n_targets_bound_gt_thr_c1=int(np.nansum(rb > thr[0])), n_targets_bound_gt_thr_c2p8=int(np.nansum(rb > thr[1])),
                          worst_target=int(np.nanargmax(rb)), worst_target_mu=float(mu_m1[int(np.nanargmax(rb))]), worst_target_Fb=float(Fb_i[int(np.nanargmax(rb))]), worst_target_nstar=float(nstar[int(np.nanargmax(rb))]),
                          worst_target_measured_F=float(F_m1[int(np.nanargmax(rb))]), worst_target_measured_max_N=int(mx_m1[int(np.nanargmax(rb))]))
        log(f"[C {arm}] Cantelli n*={label} (中央値 {tab[label]['nstar_median']:.1f}, 最小 {tab[label]['nstar_min']:.1f}): ρ_R 上界（F 上界）最大 {tab[label]['rhoR_bound_max']:.3g} 中央値 {tab[label]['rhoR_bound_median']:.3g} Σ_i {tab[label]['rhoR_bound_sum']:.3g}; （実測 F）最大 {tab[label]['rhoR_measuredF_max']:.3g} 中央値 {tab[label]['rhoR_measuredF_median']:.3g}; 独立 F=1 最大 {tab[label]['rhoR_indep_F1_max']:.3g}; Poisson 最大 {tab[label]['poisson_max']:.2g} 中央値 {tab[label]['poisson_median']:.2g}; 閾値超え {tab[label]['n_targets_bound_gt_thr_c1']}/{tab[label]['n_targets_bound_gt_thr_c2p8']} 標的 (c=1 {thr[0]}, c=2.8 {thr[1]}); 最悪標的 u{tab[label]['worst_target']} μ {tab[label]['worst_target_mu']:.2f} F上界 {tab[label]['worst_target_Fb']:.2f} n* {tab[label]['worst_target_nstar']:.1f} 実測F {tab[label]['worst_target_measured_F']:.2f} 実測 max N {tab[label]['worst_target_measured_max_N']}")
    res["cantelli"] = tab; res["threshold_c1_c2p8"] = thr
    np.savez_compressed(f"{RAW}/v1_partC_{arm}.npz", Ts=np.array(Ts), Fb_mean=Fb_mean, Fb_max=Fb_max, Fb_cos=Fb_cos, Fb_43=Fb_43, meanR=meanR,
                        frac_rho_gt03=frac_rho_gt03, n_c=n_c_arr, mu_m1=mu_m1, F_m1=F_m1, n_alive_m1=nal, nbeta=nb, nbeta_filled=nb_fill, pvals_nb=pvals,
                        om_med=om_med, om_floor_med=om_floor_med, om_rel_overlap_med=om_rel_overlap_med, om_corr_floor=om_corr_floor, rho_q=rho_q, cos_abs_q=cos_abs_q)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=3000); ap.add_argument("--n-inputs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261013); ap.add_argument("--skip", default="")
    ap.add_argument("--Tstep", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time(); res = dict(seed=args.seed)
    if "A" not in args.skip:
        res["partA"] = part_A([("a0.3", 70), ("a0.1", 25), ("a0.5", 90)], args.R, args.n_inputs, args.seed)
    if "B" not in args.skip:
        res["partB"] = part_B()
    if "C" not in args.skip:
        m1 = json.load(open(f"{WT}/results/cobirth_1013/m1_fano.json"))
        rng = np.random.default_rng(args.seed)
        res["partC"] = {}
        for arm in ("a0.1", "a0.3", "a0.5"):
            nb, nb_task = L.nbeta_table(arm)
            Ts = list(range(11, 101, args.Tstep))
            res["partC"][arm] = part_C(arm, m1, Ts, nb, nb_task - 1, 2000, rng)
    res["seconds"] = time.time() - t0
    out = f"{WT}/results/cobirth_1013/v1_verify.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    with open(f"{RAW}/v1_verify.log", "w") as f:
        f.write("\n".join(LOG) + "\n")
    log(f"wrote {out} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
