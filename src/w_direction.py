"""W の主方向と入力二次構造の整列指標 (spec_center_selfcov_0814 §2.2)。

定義を1か所に集約し、Phase 0 (numpy, npz から) と lop_metrics (torch, 学習中) で
同一の量を計算する。

- e1^W: dead 行をゼロ化した W の第1**右**特異ベクトル (= E[wwᵀ] の最大固有ベクトル)。
  raw = 中心化なし (主判定)、pca = 行方向 (ユニット方向) 中心化。
- e1^Σ: Σ = I + (κ−1)uuᵀ の最大固有ベクトル。κ>1 で u、κ=1 は縮退で NaN。
- e1^{E[xxᵀ]}: Σ + µµᵀ の最大固有ベクトル (解析的に 2 次元部分空間 span{u, µ̂} 内で解く)。
- 統計は |cos| (軸吸引なので符号は自発的、§1.6)。signed も参考に返す。
"""
import math

import numpy as np
import torch


def spike_dir_vec(spike_dir, d, np_out=True):
    """GaussEnv と同一の u (envs.GaussEnv.__init__ に対応)。"""
    if spike_dir == "ones":
        u = np.full(d, 1.0 / math.sqrt(d))
    elif spike_dir == "alt":
        n = d - (d % 2)
        u = np.zeros(d)
        u[:n] = np.tile([1.0, -1.0], n // 2)
        u = u / np.linalg.norm(u)
    else:
        raise ValueError(f"unknown spike_dir: {spike_dir}")
    return u if np_out else torch.as_tensor(u)


def e1_second_moment(u, kappa, mu):
    """E[xxᵀ] = I + (κ−1)uuᵀ + µµᵀ の最大固有ベクトル (解析)。

    非等方成分は span{u, µ̂} に載るので、その 2 次元で 2x2 対称行列を対角化すれば足りる
    (残りの方向の固有値は 1 で、非等方成分の最大固有値 ≥ 1 に負けない)。
    κ=1 かつ µ=0 なら等方 → None (縮退)。"""
    nmu = float(np.linalg.norm(mu))
    a = float(kappa) - 1.0
    if a <= 0 and nmu <= 1e-12:
        return None
    if nmu <= 1e-12:
        return u
    m = mu / nmu
    if a <= 0:
        return m
    # 基底 {u, m⊥} を Gram-Schmidt で作る
    p = float(u @ m)
    r = m - p * u
    nr = float(np.linalg.norm(r))
    if nr < 1e-12:                       # µ ∥ u
        return u
    e2 = r / nr
    # M = a·uuᵀ + nmu²·mmᵀ を {u, e2} 基底で表現 (m = p·u + nr·e2)
    b = nmu ** 2
    M = np.array([[a + b * p * p, b * p * nr],
                  [b * p * nr, b * nr * nr]])
    w, V = np.linalg.eigh(M)
    top = V[:, int(np.argmax(w))]
    v = top[0] * u + top[1] * e2
    return v / np.linalg.norm(v)


def _e1_of_W(W_alive, center_rows):
    """[h,d] の第1右特異ベクトル。center_rows なら行方向を中心化 (PCA 版)。"""
    A = W_alive - W_alive.mean(axis=0, keepdims=True) if center_rows else W_alive
    if A.shape[0] < 1 or not np.isfinite(A).all():
        return None, np.nan
    try:
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, np.nan
    if s[0] <= 1e-30:
        return None, np.nan
    top1 = float(s[0] ** 2 / max((s ** 2).sum(), 1e-30))
    return Vt[0], top1


def _abscos(a, b):
    if a is None or b is None:
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return np.nan
    return float(abs(a @ b) / (na * nb))


def w_dir_metrics_np(W, dead, u, mu, kappa=None, prev_e1=None):
    """1 run 分。W [h,d], dead [h] bool, u [d] or None, mu [d]。

    u=None (条件A のような等方 Σ) では Σ 系は全て NaN (§2.2 の縮退規約)。
    kappa=None のときは u が与えられていれば e1^Σ = u とみなす (Phase 0 用)。
    返り値には e1 自体 (e1_vec) を含み、呼び出し側が e1_stability に使う。"""
    alive = ~np.asarray(dead, dtype=bool)
    out = dict(cos_e1W_e1Sig=np.nan, cos_e1W_e1M2=np.nan, cos_e1W_e1Sig_pca=np.nan,
               cos_e1W_mu=np.nan, cos_e1W_e1Sig_signed=np.nan,
               top1_frac_alive=np.nan, w_norm_mean=np.nan, srank_alive=np.nan,
               e1_stability=np.nan, e1_vec=None, n_alive=int(alive.sum()))
    if alive.sum() < 2:
        return out
    Wa = np.asarray(W, dtype=np.float64)[alive]
    if not np.isfinite(Wa).all():
        return out
    e1, top1 = _e1_of_W(Wa, center_rows=False)
    e1p, _ = _e1_of_W(Wa, center_rows=True)
    out["top1_frac_alive"] = top1
    out["w_norm_mean"] = float(np.linalg.norm(Wa, axis=1).mean())
    s = np.linalg.svd(Wa, compute_uv=False)
    out["srank_alive"] = float((s ** 2).sum() / max(s[0] ** 2, 1e-30))
    out["e1_vec"] = e1

    kap = 1.0 if kappa is None else float(kappa)
    e1sig = u if (u is not None and (kappa is None or kap > 1)) else None
    e1m2 = e1_second_moment(u, kap, np.asarray(mu, dtype=np.float64)) \
        if u is not None else None
    out["cos_e1W_e1Sig"] = _abscos(e1, e1sig)
    out["cos_e1W_e1Sig_pca"] = _abscos(e1p, e1sig)
    out["cos_e1W_e1M2"] = _abscos(e1, e1m2)
    mu = np.asarray(mu, dtype=np.float64)
    out["cos_e1W_mu"] = _abscos(e1, mu) if np.linalg.norm(mu) > 1e-12 else np.nan
    if e1 is not None and e1sig is not None:
        out["cos_e1W_e1Sig_signed"] = float(e1 @ e1sig
                                            / max(np.linalg.norm(e1), 1e-30))
    out["e1_stability"] = _abscos(e1, prev_e1)
    return out


def random_floor(d):
    """d 次元独立ベクトル対の E|cos| ≈ sqrt(2/(pi d))。"""
    return math.sqrt(2.0 / (math.pi * d))
