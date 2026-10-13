#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_cov_formula.py — cobirth_1013 D1 (a) の数値検算（走なし。保存重みの 1 状態で新ラベルのモンテカルロ）。

検算する式（DERIVATION.md 補題 1〜2）:
  (1) 全データ SGD 1 歩: Cov_Y(Δz_jn, Δz_kn') = (η²/(N²K)) ⟨v_j^c, v_k^c⟩ Σ_m Θ_nm Θ_n'm d_jm d_km
  (1B) バッチ版（バッチ B 固定）: Σ_m → Σ_{m∈B}、η/N → η/|B|
  (1A) Adam 線形領域版 [模型]: Θ_nm → Θ̃^{(j)}_nm = Σ_c x̃_nc x̃_mc / √v⁰_jc、係数 η(1−β₁)。v⁰ は合成（E_Y[g²]）で代数だけを検算
  (2) 同じ入力での相関 = cos(v_j^c, v_k^c) × ω_jk(n)、ω = Σ Θ² d_j d_k / √(Σ Θ² d_j² Σ Θ² d_k²)
  勾配の閉じた形そのものは torch autograd（訓練と同じ平均 CE・where(z>0,1,a)）と照合する。
  付録: 1 歩の Berry–Esseen 比、Plackett の恒等式と上界 (3)(4) の MC 照合、Fano・Cantelli の表。

判定の幅は固定値でなく MC の標本分布から出す（Frobenius 相対誤差の予測 = √(Σ_ab (C_aa C_bb + C_ab²)/(R−1)) / ‖C‖_F）。
課題番号: --task-end t は end0[t−1]（1 始まりの課題 t の末。stoch/p161 の「課題 t」= end0[t−2] とは 1 ずれる）。
出力: results/cobirth_1013/check_cov_formula.json、ログ obsidian-research-data/cobirth_1013/d1/check_cov_formula.log
使い方: OMP_NUM_THREADS=1 python check_cov_formula.py [--arm a0.1] [--task-end 40] [--R 2000] [--n-inputs 24]
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json, math, time, argparse
import numpy as np

ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT)
from common import load_X  # noqa: E402

WT = "/home/issan/Projects/claude/wt/cobirth_1013"
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_1013/d1"
ARMS = {"a0.1": (ROOT + "/review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1),
        "a0.3": (ROOT + "/review/opus_lrslope/out/LRa0.3_s0.npz", 0.3),
        "a0.5": (ROOT + "/review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
ETA, K, BATCH, BETA1 = 1e-3, 10, 16, 0.9   # run_bwd.py / run_slope.py: lr 1e-3, Adam (0.9, 0.999), batch 16

LOG = []
def log(s):
    print(s, flush=True); LOG.append(s)

def softmax(l):
    l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)

class St:
    pass

def load_state(arm, task_end):
    path, a = ARMS[arm]
    d = np.load(path, allow_pickle=True)
    if "slope" in d.files: a = float(d["slope"])
    i = task_end - 1
    st = St(); st.arm, st.a, st.task_end, st.idx = arm, a, task_end, i
    st.W1 = d["end0"][i].astype(np.float64); st.b1 = d["end1"][i].astype(np.float64)
    st.W2 = d["end2"][i].astype(np.float64); st.b3 = d["end3"][i].astype(np.float64)
    st.X = load_X(d["subset"]); st.N = st.X.shape[0]
    st.Xt = np.hstack([st.X, np.ones((st.N, 1))])
    st.Wt = np.hstack([st.W1, st.b1[:, None]])
    st.Z = st.Xt @ st.Wt.T                          # [N,H]
    st.D = np.where(st.Z > 0, 1.0, a)               # hard leaky の微分（訓練と同じ where(z>0,1,a)）
    st.Phi = np.where(st.Z > 0, st.Z, a * st.Z)
    st.logits = st.Phi @ st.W2.T + st.b3
    st.P = softmax(st.logits)
    st.V = st.W2.T                                   # [H,K], v_j = V[j]
    st.Vc = st.V - st.V.mean(1, keepdims=True)
    st.G = st.Vc @ st.Vc.T                           # ⟨v_j^c, v_k^c⟩
    st.Theta = st.Xt @ st.Xt.T                       # Θ_nm = x_n·x_m + 1 ≥ 1
    st.PW = st.P @ st.W2                             # [N,H]: ⟨v_j, p_m⟩
    st.k = (st.Z > 0).sum(0); st.alive = st.k > 0
    st.H = st.Z.shape[1]
    return st

def pick_inputs(st, n_inputs, rng):
    """標的 1 つ（狭い: 3 ≤ k ≤ 8）の閾値の入力 T_i（e′ 上位 3）と開いた入力（≤ 5）＋ 無作為な入力で n_inputs 個。"""
    tau = 0.1 * st.Z.std(0) + 1e-12
    s = 1.0 / (1.0 + np.exp(-st.Z / tau))
    Dp = (1 - st.a) * s * (1 - s) / tau
    Kv = st.Xt @ st.Xt.mean(0)
    E = Kv[:, None] * Dp
    cands = np.where((st.k >= 3) & (st.k <= 8))[0]
    chosen, tgt = [], None
    if len(cands):
        tgt = int(cands[rng.integers(len(cands))])
        T = np.argsort(-E[:, tgt])[:3].tolist()
        O = np.where(st.Z[:, tgt] > 0)[0].tolist()[:5]
        chosen = list(dict.fromkeys(T + O))
    rest = [int(m) for m in rng.permutation(st.N) if int(m) not in chosen][: n_inputs - len(chosen)]
    S = np.array(chosen + rest, dtype=int)
    return S, tgt, chosen

def exact_cov(st, S, Th=None, coef=None):
    """C[(j,n),(k,p)] = coef ⟨v_j^c,v_k^c⟩ Σ_m Th[j,n,m] Th[k,p,m]。Th: [H,s,N]（省略時 Θ_nm d_jm）。"""
    s = len(S)
    if Th is None:
        Th = st.Theta[S][None, :, :] * st.D.T[:, None, :]
    if coef is None:
        coef = ETA ** 2 / (st.N ** 2 * K)
    A = Th.reshape(st.H * s, st.N)
    M = A @ A.T
    return coef * np.kron(st.G, np.ones((s, s))) * M, Th

def mc_fullbatch(st, S, R, rng):
    """Δz_S [R, H*s]（順序 j*s+n）。全データ SGD 1 歩、新ラベル iid 一様。"""
    s = len(S); ThS = st.Theta[S]
    out = np.empty((R, st.H, s))
    for r in range(R):
        y = rng.integers(0, K, size=st.N)
        delta = (st.PW - st.W2[y, :]) * st.D          # [N,H]: d_jm ⟨v_j, p_m − e_{Y_m}⟩
        out[r] = (-(ETA / st.N) * ThS @ delta).T      # [H,s]
    return out.reshape(R, st.H * s)

def compare(C_mc, C_ex, R, name):
    num = np.linalg.norm(C_mc - C_ex); den = np.linalg.norm(C_ex)
    diag = np.diag(C_ex)
    pred = math.sqrt((np.outer(diag, diag) + C_ex ** 2).sum() / (R - 1)) / den
    dmc, dex = np.diag(C_mc), diag
    ok = dex > 0
    rel_diag = np.abs(dmc[ok] - dex[ok]) / dex[ok]
    r = dict(rel_frobenius_error=float(num / den), predicted_mc_rel_frobenius=float(pred),
             observed_over_predicted=float(num / den / pred),
             diag_rel_error_median=float(np.median(rel_diag)), diag_rel_error_q90=float(np.quantile(rel_diag, .9)),
             diag_rel_error_max=float(rel_diag.max()), diag_mc_sd_pred=float(math.sqrt(2.0 / (R - 1))),
             n_coords=int(C_ex.shape[0]), R=int(R))
    log(f"[{name}] rel Frobenius err {r['rel_frobenius_error']:.4f} (MC 予測 {pred:.4f}, 比 {r['observed_over_predicted']:.2f}); "
        f"対角の相対誤差 中央値 {r['diag_rel_error_median']:.4f} q90 {r['diag_rel_error_q90']:.4f} 最大 {r['diag_rel_error_max']:.4f} (MC 予測 sd {r['diag_mc_sd_pred']:.4f})")
    return r

def corr_check(st, S, C_mc, C_ex, R):
    """同じ入力での相関 ρ_jk(n) の MC 値と cos × ω の差。生きた unit の対のみ。"""
    s = len(S); H = st.H
    cos = st.G / np.sqrt(np.outer(np.diag(st.G), np.diag(st.G)))
    Th = st.Theta[S][None, :, :] * st.D.T[:, None, :]     # [H,s,N]
    nrm = np.sqrt((Th ** 2).sum(2))                        # [H,s]
    alive = np.where(st.alive)[0]
    diffs, rho_ex_all, om_all = [], [], []
    for n in range(s):
        idx = alive * s + n
        sub_mc = C_mc[np.ix_(idx, idx)]; sub_ex = C_ex[np.ix_(idx, idx)]
        d_mc = np.sqrt(np.diag(sub_mc)); d_ex = np.sqrt(np.diag(sub_ex))
        rho_mc = sub_mc / np.outer(d_mc, d_mc); rho_ex = sub_ex / np.outer(d_ex, d_ex)
        om = (Th[alive, n, :] @ Th[alive, n, :].T) / np.outer(nrm[alive, n], nrm[alive, n])
        fac = cos[np.ix_(alive, alive)] * om
        iu = np.triu_indices(len(alive), 1)
        diffs.append(np.abs(rho_mc - fac)[iu]); rho_ex_all.append(np.abs(rho_ex - fac)[iu].max()); om_all.append(om[iu])
    diffs = np.concatenate(diffs); om_all = np.concatenate(om_all)
    r = dict(max_abs_diff_rho_mc_vs_cos_omega=float(diffs.max()), median_abs_diff=float(np.median(diffs)),
             q99_abs_diff=float(np.quantile(diffs, .99)), mc_se_rho_approx=float(1.0 / math.sqrt(R)),
             max_abs_diff_rho_exact_vs_cos_omega=float(max(rho_ex_all)),
             omega_quantiles_alive_pairs=[float(np.quantile(om_all, q)) for q in (.1, .5, .9, .99)],
             omega_min=float(om_all.min()), omega_max=float(om_all.max()),
             abs_cos_quantiles_alive_pairs=[float(np.quantile(np.abs(cos[np.ix_(alive, alive)][np.triu_indices(len(alive), 1)]), q)) for q in (.1, .5, .9, .99, 1.0)],
             n_alive=int(len(alive)))
    log(f"[corr] |ρ_MC − cos·ω| 最大 {r['max_abs_diff_rho_mc_vs_cos_omega']:.4f} 中央値 {r['median_abs_diff']:.4f} q99 {r['q99_abs_diff']:.4f} (MC se ≈ {r['mc_se_rho_approx']:.4f}); "
        f"厳密式の ρ と cos·ω の差 最大 {r['max_abs_diff_rho_exact_vs_cos_omega']:.1e}; ω 分位 (10/50/90/99%) {np.round(r['omega_quantiles_alive_pairs'],3).tolist()} 最小 {r['omega_min']:.3f}; "
        f"|cos| 分位 (10/50/90/99/100%) {np.round(r['abs_cos_quantiles_alive_pairs'],3).tolist()}")
    return r

def mean_check(st, S, out, R):
    """E_Y[Δz_jn] = μ_jn = −(η/N) Σ_m Θ_nm d_jm ⟨v_j, p_m − u⟩ の MC 照合（se 単位）。"""
    s = len(S)
    mu = (-(ETA / st.N) * st.Theta[S] @ ((st.PW - st.W2.mean(0)[None, :]) * st.D)).T.reshape(-1)   # [H*s]
    m_mc = out.mean(0); se = out.std(0, ddof=1) / math.sqrt(R)
    ok = se > 0
    zsc = np.abs(m_mc[ok] - mu[ok]) / se[ok]
    r = dict(max_abs_z=float(zsc.max()), frac_abs_z_gt_3=float((zsc > 3).mean()), n=int(ok.sum()))
    log(f"[mean] |MC 平均 − μ|/se 最大 {r['max_abs_z']:.2f}, >3 の割合 {r['frac_abs_z_gt_3']:.4f} (n {r['n']})")
    return r

def autograd_check(st, rng, ndraw=3):
    import torch
    torch.set_num_threads(1)
    X = torch.tensor(st.X); Xt = torch.tensor(st.Xt)
    W1 = torch.tensor(st.W1, requires_grad=True); b1 = torch.tensor(st.b1, requires_grad=True)
    W2 = torch.tensor(st.W2); b3 = torch.tensor(st.b3)
    errs = []
    for _ in range(ndraw):
        y = rng.integers(0, K, size=st.N); yt = torch.tensor(y)
        z = X @ W1.T + b1
        h = torch.where(z > 0, z, st.a * z)
        logits = h @ W2.T + b3
        loss = (logits.logsumexp(-1) - logits.gather(1, yt[:, None]).squeeze(1)).mean()   # run_bwd.py と同じ平均 CE
        gW1, gb1 = torch.autograd.grad(loss, (W1, b1))
        dWt = -ETA * torch.cat([gW1, gb1[:, None]], 1)
        dz_auto = (Xt @ dWt.T).numpy()
        delta = (st.PW - st.W2[y, :]) * st.D
        dz_closed = -(ETA / st.N) * st.Theta @ delta
        errs.append(float(np.abs(dz_auto - dz_closed).max() / np.abs(dz_closed).max()))
    r = dict(max_rel_error_vs_autograd=float(max(errs)), ndraw=ndraw, torch=torch.__version__)
    log(f"[autograd] 閉じた形 Δz と torch autograd の SGD 1 歩の差 / max|Δz| 最大 {r['max_rel_error_vs_autograd']:.2e} (float64)")
    return r

def batch_check(st, S, R, rng):
    B = np.sort(rng.choice(st.N, BATCH, replace=False))
    s = len(S)
    Th = st.Theta[S][:, B][None, :, :] * st.D[B].T[:, None, :]          # [H,s,|B|]
    A = Th.reshape(st.H * s, BATCH)
    C_ex = (ETA ** 2 / (BATCH ** 2 * K)) * np.kron(st.G, np.ones((s, s))) * (A @ A.T)
    out = np.empty((R, st.H, s)); ThSB = st.Theta[S][:, B]
    for r in range(R):
        y = rng.integers(0, K, size=BATCH)
        delta = (st.PW[B] - st.W2[y, :]) * st.D[B]
        out[r] = (-(ETA / BATCH) * ThSB @ delta).T
    C_mc = np.cov(out.reshape(R, -1).T)
    r = compare(C_mc, C_ex, R, "batch16 fixed B")
    r["batch"] = B.tolist()
    return r

def adam_linear_check(st, S, R, rng, R_v0=200):
    """[模型] 線形領域の Adam: Δw̃_jc = −η(1−β₁) g_jc/√v⁰_jc（v⁰ は合成: E_Y[g_jc²] の MC 推定）。
    Δz_S の MC と、Θ̃^{(j)} を入れた式 (1A) の照合。恒等式 Σ_c x̃_nc g_jc/√v⁰_jc = (1/N) Σ_m Θ̃^{(j)}_nm δ_jm も 2 本で確認。"""
    s = len(S)
    # 合成 v⁰
    v0 = np.zeros((st.H, st.Xt.shape[1]))
    for _ in range(R_v0):
        y = rng.integers(0, K, size=st.N)
        g = ((st.PW - st.W2[y, :]) * st.D).T @ st.Xt / st.N          # [H,785]
        v0 += g ** 2
    v0 = v0 / R_v0 + 1e-30
    inv = 1.0 / np.sqrt(v0)                                           # [H,785]
    ThS = np.empty((st.H, s, st.N))
    for j in range(st.H):
        ThS[j] = (st.Xt[S] * inv[j][None, :]) @ st.Xt.T              # Θ̃^{(j)}_nm, n ∈ S
    Th = ThS * st.D.T[:, None, :]
    coef = (ETA * (1 - BETA1)) ** 2 / (st.N ** 2 * K)
    C_ex, _ = exact_cov(st, S, Th=Th, coef=coef)
    out = np.empty((R, st.H, s)); id_err = []
    for r in range(R):
        y = rng.integers(0, K, size=st.N)
        delta = (st.PW - st.W2[y, :]) * st.D                           # [N,H]
        out[r] = -(ETA * (1 - BETA1) / st.N) * np.einsum("jnm,mj->jn", ThS, delta)
        if r < 2:
            g = delta.T @ st.Xt / st.N
            dz_direct = -(ETA * (1 - BETA1)) * (st.Xt[S] @ (g * inv).T).T   # [H,s]
            id_err.append(float(np.abs(dz_direct - out[r]).max() / np.abs(out[r]).max()))
    C_mc = np.cov(out.reshape(R, -1).T)
    r = compare(C_mc, C_ex, R, "Adam linear (model, synthetic v0)")
    r["identity_err_direct_vs_Theta_tilde"] = float(max(id_err))
    r["note"] = "v0 は合成（E_Y[g^2] の MC 推定）。実際の Adam 状態ではない。代数の検算のみ。"
    return r

def berry_esseen(st, S):
    """1 歩の ξ_jn = Σ_m X_m（独立）の Berry–Esseen 比 0.56 Σ E|X_m|³ / (Σ Var X_m)^{3/2}（Shevtsova 2010 の定数 0.56）。"""
    alive = np.where(st.alive)[0]
    e3 = (np.abs(st.Vc) ** 3).mean(1); var = (st.Vc ** 2).mean(1)            # E|ε_j|³, Var ε_j
    kurt = e3 / var ** 1.5                                                     # unit ごと
    Th = st.Theta[S][None, :, :] * st.D.T[:, None, :]                          # [H,s,N]
    geo = (Th ** 3).sum(2) / (Th ** 2).sum(2) ** 1.5                            # [H,s]
    be = 0.56 * kurt[:, None] * geo
    be = be[alive]
    r = dict(be_ratio_quantiles=[float(np.quantile(be, q)) for q in (.1, .5, .9, 1.0)],
             effective_n_quantiles=[float(np.quantile(1.0 / geo[alive] ** 2, q)) for q in (.1, .5, .9)])
    log(f"[BE] 1 歩の Berry–Esseen 上界 分位 (10/50/90/100%) {np.round(r['be_ratio_quantiles'],4).tolist()}; 実効 N (1/geo²) 分位 {np.round(r['effective_n_quantiles'],1).tolist()}")
    return r

# ---- Plackett / Fano / Cantelli（2 変量正規の閾値越えの共分散） ----
def Phi(x): return 0.5 * math.erfc(-x / math.sqrt(2.0))
def Phibar(x): return 0.5 * math.erfc(x / math.sqrt(2.0))
def Phibar_inv(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phibar(mid) > p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
def phi2(a, b, r):
    q = 1.0 - r * r
    return np.exp(-(a * a - 2 * r * a * b + b * b) / (2 * q)) / (2 * math.pi * np.sqrt(q))
def plackett_cov(a, b, rho, n=2000):
    """Cov(1[X>a],1[Y>b]) = ∫_0^ρ φ₂(a,b;r) dr（Simpson）。ρ ≤ 0 では ≤ 0 なので 0 を返す（正の部分）。"""
    if rho <= 0: return 0.0
    r = np.linspace(0, rho, n + 1); f = phi2(a, b, r)
    h = rho / n
    return float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))
def cov_bound3(a, b, rho):
    return rho / (2 * math.pi * math.sqrt(1 - rho * rho)) * math.exp(-(a * a + b * b) / (2 * (1 + rho)))
def R_bound4(a, b, rho):
    return rho * (1 + a * a) * (1 + b * b) / (a * b * math.sqrt(1 - rho * rho)) * math.exp((a * a + b * b) * rho / (2 * (1 + rho)))

def plackett_mc(rng, p=0.003, rhos=(0.05, 0.15, 0.3), n=2_000_000):
    a = Phibar_inv(p); rows = []
    for rho in rhos:
        X = rng.standard_normal(n); Y = rho * X + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
        joint = float(((X > a) & (Y > a)).mean())
        cov_mc = joint - Phibar(a) ** 2
        se = math.sqrt(joint * (1 - joint) / n)
        cp = plackett_cov(a, a, rho); cb = cov_bound3(a, a, rho)
        rows.append(dict(rho=rho, a=a, p=p, cov_mc=cov_mc, cov_mc_se=se, cov_plackett=cp, cov_bound3=cb,
                         z_mc_vs_plackett=(cov_mc - cp) / se, bound3_over_exact=cb / cp,
                         rel_excess_exact=cp / Phibar(a) ** 2, R_bound4=R_bound4(a, a, rho)))
        log(f"[Plackett] ρ {rho}: Cov MC {cov_mc:.3e} (se {se:.1e}) 対 Plackett {cp:.3e} (z {(cov_mc-cp)/se:+.2f}); 上界 (3)/厳密 {cb/cp:.2f}; R 厳密 {cp/Phibar(a)**2:.2f} 対 上界 (4) {R_bound4(a,a,rho):.2f}")
    return rows

def fano_cantelli_table(n_cand=99, mus=(0.3, 1.5), nstars=(3.6, 12.8, 20.0, 33.0), rhobars=(0.05, 0.15, 0.3)):
    rows = []
    for mu in mus:
        p = mu / n_cand; a = Phibar_inv(p)
        for rb in rhobars:
            cov = plackett_cov(a, a, rb); Rex = cov / p ** 2; Rb = R_bound4(a, a, rb)
            F_ex = 1 + mu * Rex * (n_cand - 1) / n_cand; F_b = 1 + mu * Rb * (n_cand - 1) / n_cand
            for ns in nstars:
                if ns <= mu: continue
                rr_ex = mu * F_ex / (mu * F_ex + (ns - mu) ** 2); rr_b = mu * F_b / (mu * F_b + (ns - mu) ** 2)
                rr_ind = mu / (mu + (ns - mu) ** 2)
                kk = math.ceil(ns); pois = 1 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(kk))
                rows.append(dict(mu=mu, p_per_cand=p, a=a, rhobar=rb, R_exact_gauss=Rex, R_bound4=Rb, F_exact_gauss=F_ex, F_bound=F_b,
                                 nstar=ns, rhoR_cantelli_exact_gauss=rr_ex, rhoR_cantelli_bound=rr_b, rhoR_cantelli_independent=rr_ind,
                                 poisson_tail_ceil=pois))
    return rows

def state_pair_R(st, S, p=0.003):
    """実状態: 生きた unit の全対 × S の入力で ρ_jk(n) = cos × ω を計算し、R(ρ⁺) の対平均（Plackett 厳密と上界 (4)）。"""
    alive = np.where(st.alive)[0]; s = len(S)
    cos = st.G / np.sqrt(np.outer(np.diag(st.G), np.diag(st.G)))
    Th = st.Theta[S][None, :, :] * st.D.T[:, None, :]
    nrm = np.sqrt((Th ** 2).sum(2))
    a = Phibar_inv(p)
    grid = np.linspace(0, 0.995, 400)
    g_ex = np.array([plackett_cov(a, a, r, n=400) for r in grid]) / p ** 2
    g_b = np.array([R_bound4(a, a, r) if r > 0 else 0.0 for r in grid])
    rhos = []
    for n in range(s):
        om = (Th[alive, n, :] @ Th[alive, n, :].T) / np.outer(nrm[alive, n], nrm[alive, n])
        rho = (cos[np.ix_(alive, alive)] * om)[np.triu_indices(len(alive), 1)]
        rhos.append(rho)
    rho = np.concatenate(rhos); rp = np.clip(rho, 0, 0.995)
    Rex = np.interp(rp, grid, g_ex); Rb = np.interp(rp, grid, g_b)
    r = dict(p=p, a=a, n_pairs_inputs=int(rho.size),
             rho_quantiles=[float(np.quantile(rho, q)) for q in (.1, .5, .9, .99, 1.0)],
             abs_rho_quantiles=[float(np.quantile(np.abs(rho), q)) for q in (.5, .9, .99, 1.0)],
             frac_rho_pos=float((rho > 0).mean()), frac_rho_gt_0p3=float((rho > 0.3).mean()),
             mean_R_exact_gauss=float(Rex.mean()), mean_R_bound4=float(Rb.mean()),
             share_R_exact_from_rho_gt_0p3=float(Rex[rho > 0.3].sum() / Rex.sum()),
             F_minus1_per_mu_exact=float(Rex.mean()), note="F ≤ 1 + μ·mean_R（対平均。等しい p の近似）")
    log(f"[state] ρ=cos·ω の分位 (10/50/90/99/100%) {np.round(r['rho_quantiles'],3).tolist()}; ρ>0 の割合 {r['frac_rho_pos']:.3f}; ρ>0.3 {r['frac_rho_gt_0p3']:.3f}; "
        f"mean R 厳密正規 {r['mean_R_exact_gauss']:.2f} / 上界 (4) {r['mean_R_bound4']:.2f}; R の質量のうち ρ>0.3 から {r['share_R_exact_from_rho_gt_0p3']:.2f}")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a0.1"); ap.add_argument("--task-end", type=int, default=40)
    ap.add_argument("--R", type=int, default=2000); ap.add_argument("--n-inputs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1013); ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(a_.seed)
    st = load_state(a_.arm, a_.task_end)
    log(f"state {a_.arm} task_end {a_.task_end} (= end0[{st.idx}]), a {st.a}, N {st.N}, H {st.H}, alive {int(st.alive.sum())}, k 中央値 {float(np.median(st.k[st.alive])):.0f}, acc(旧ラベルなし) —")
    S, tgt, chosen = pick_inputs(st, a_.n_inputs, rng)
    log(f"inputs S ({len(S)}): 標的 {tgt} (k={int(st.k[tgt]) if tgt is not None else None}) の T_i∪O_i {chosen} ＋ 無作為 {len(S)-len(chosen)}")
    res = dict(arm=a_.arm, task_end=a_.task_end, end0_index=st.idx, a=st.a, eta=ETA, K=K, N=st.N, H=st.H, R=a_.R, seed=a_.seed,
               inputs=S.tolist(), target=tgt, target_inputs=chosen,
               convention="task_end t = end0[t-1] (1 始まりの課題 t の末)。stoch/p161 の『課題 t』= end0[t-2] とは 1 ずれる")
    # (1) 全データ SGD
    C_ex, _ = exact_cov(st, S)
    out = mc_fullbatch(st, S, a_.R, rng)
    C_mc = np.cov(out.T)
    res["fullbatch_sgd"] = compare(C_mc, C_ex, a_.R, "fullbatch SGD R=%d" % a_.R)
    # R の半分でも（1/√R の確認）
    Rh = a_.R // 4
    C_mc_h = np.cov(out[:Rh].T)
    res["fullbatch_sgd_quarter_R"] = compare(C_mc_h, C_ex, Rh, "fullbatch SGD R=%d" % Rh)
    res["mean"] = mean_check(st, S, out, a_.R)
    res["corr_cos_omega"] = corr_check(st, S, C_mc, C_ex, a_.R)
    res["autograd"] = autograd_check(st, rng)
    res["batch16"] = batch_check(st, S, a_.R, rng)
    res["adam_linear_model"] = adam_linear_check(st, S, min(a_.R, 1000), rng)
    res["berry_esseen"] = berry_esseen(st, S)
    res["plackett"] = plackett_mc(rng)
    res["fano_cantelli_table"] = fano_cantelli_table()
    res["state_pairs"] = state_pair_R(st, S)
    res["seconds"] = time.time() - t0
    out_json = a_.out or f"{WT}/results/cobirth_1013/check_cov_formula_{a_.arm}_t{a_.task_end}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True); os.makedirs(RAW, exist_ok=True)
    with open(out_json, "w") as f: json.dump(res, f, indent=1, ensure_ascii=False)
    with open(f"{RAW}/check_cov_formula_{a_.arm}_t{a_.task_end}.log", "w") as f: f.write("\n".join(LOG) + "\n")
    log(f"wrote {out_json} ({res['seconds']:.0f}s)")

if __name__ == "__main__":
    main()
