#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v1_lib.py — cobirth_1013 V1（検証役）の共通部品。D1 の check_cov_formula.py とは独立に書いた。

- load_state(arm, T): M1 の約束（T = 1..100、状態 = end0[T-1]）で保存重みを読む。
- bridges(st): stoch_bridges.py bridges_at の基準を字句どおりに写した緩い橋の判定（標的 i 行・候補 j 列）。
- omega_inputs(st, inputs): 入力 n ごとの ω_jk(n) = Σ_m Θ_nm² d_jm d_km / √(Σ Θ² d_j² Σ Θ² d_k²)（硬いゲート）。
  床だけの版 ω_floor（(1-a)² の重なり項を 0 にしたもの）も返す。
- plackett_R_table(a): r ∈ [0, 0.999] の格子で Cov(1[X>a],1[Y>a]) = ∫_0^ρ φ₂(a,a;r) dr（累積台形）と R = Cov/p²。
  独立な検算として、1 次元積分 P(X>a,Y>b) = ∫_a^∞ φ(x) Φ̄((b − ρx)/√(1−ρ²)) dx も持つ。
- fano_bound(...): D1 定理 4 (4.5) を標的ごとに評価（R の対平均、a = Φ̄⁻¹(μ_i/n_c)）。
- cantelli(mu, F, nstar)。
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, math
import numpy as np

ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT)
from common import load_X  # noqa: E402

ARMS = {"a0.1": (ROOT + "/review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1),
        "a0.3": (ROOT + "/review/opus_lrslope/out/LRa0.3_s0.npz", 0.3),
        "a0.5": (ROOT + "/review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
C_CLASSES = 10
ETA = 1e-3

_erfv = np.vectorize(math.erf)
_erfcv = np.vectorize(math.erfc)


def Phi(x):
    return 0.5 * (1.0 + _erfv(np.asarray(x, float) / math.sqrt(2.0)))


def Phibar(x):
    return 0.5 * _erfcv(np.asarray(x, float) / math.sqrt(2.0))


def Phibar_inv(p):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(Phibar(mid)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / math.sqrt(2 * math.pi)


class St:
    pass


def load_ref(arm):
    path, a = ARMS[arm]
    d = np.load(path, allow_pickle=True)
    if "slope" in d.files:
        a = float(d["slope"])
    return d, a


def load_state(arm, T, d=None, a=None):
    """T = 1..100（M1 の約束）: 状態 = end0[T-1]。stoch の t = T+1。"""
    if d is None:
        d, a = load_ref(arm)
    st = St()
    st.arm, st.a, st.T, st.idx = arm, a, T, T - 1
    st.W1 = d["end0"][T - 1].astype(np.float64)
    st.b1 = d["end1"][T - 1].astype(np.float64)
    st.W2 = d["end2"][T - 1].astype(np.float64)
    st.b3 = d["end3"][T - 1].astype(np.float64)
    st.X = load_X(d["subset"])
    st.N = st.X.shape[0]
    st.Xt = np.hstack([st.X, np.ones((st.N, 1))])
    st.Wt = np.hstack([st.W1, st.b1[:, None]])
    fill_derived(st)
    return st


def state_from_params(arm, a, X, W1, b1, W2, b3):
    st = St()
    st.arm, st.a, st.T, st.idx = arm, a, None, None
    st.W1 = np.asarray(W1, np.float64); st.b1 = np.asarray(b1, np.float64)
    st.W2 = np.asarray(W2, np.float64); st.b3 = np.asarray(b3, np.float64)
    st.X = X; st.N = X.shape[0]
    st.Xt = np.hstack([st.X, np.ones((st.N, 1))])
    st.Wt = np.hstack([st.W1, st.b1[:, None]])
    fill_derived(st)
    return st


def fill_derived(st):
    st.Z = st.Xt @ st.Wt.T                      # [N,H]
    st.H = st.Z.shape[1]
    st.O = st.Z > 0
    st.D = np.where(st.O, 1.0, st.a)            # 硬いゲート（訓練の backward と同じ）
    st.k = st.O.sum(0)
    st.alive = st.k > 0
    st.V = st.W2.T                              # [H,C]
    st.Vc = st.V - st.V.mean(1, keepdims=True)
    st.G = st.Vc @ st.Vc.T
    nrm = np.sqrt(np.diag(st.G))
    with np.errstate(invalid="ignore", divide="ignore"):
        st.cos = st.G / np.outer(nrm, nrm)
    st.cos[~np.isfinite(st.cos)] = 0.0
    st.Theta = st.Xt @ st.Xt.T                  # Θ_nm = x_n·x_m + 1 ≥ 1
    st.xbar_t = st.Xt.mean(0)
    st.Kv = st.Xt @ st.xbar_t                   # K_n = x_n·x̄ + 1
    # なめらかなゲート（橋の判定の閾値の入力の選び方。stoch_bridges.py と同じ）
    st.tau = 0.1 * st.Z.std(0) + 1e-12
    s = 0.5 * (1 + np.tanh(0.5 * st.Z / st.tau))
    st.Dp = (1 - st.a) * s * (1 - s) / st.tau
    st.E = st.Kv[:, None] * st.Dp               # e′_in
    st.Tin = np.argsort(-st.E, axis=0)[:3, :]   # [3,H] 閾値の入力


def bridges(st):
    """stoch_bridges.bridges_at の写し。isb[i, j] = j が標的 i の緩い橋。"""
    O = st.O; H = st.H
    alive = O.any(0); k = O.sum(0); need = (k + 1) // 2
    Oi = O.astype(np.int32); ov = Oi.T @ Oi
    T = st.Tin
    thr = np.stack([O[T[:, i], :].sum(0) for i in range(H)], 0)
    isb = (thr >= 2) & (ov >= need[:, None])
    isb[np.diag_indices(H)] = False
    isb[~alive, :] = False; isb[:, ~alive] = False
    return isb, thr, ov, need


def omega_inputs(st, inputs):
    """入力 n ∈ inputs ごとの ω_jk(n)（[len, H, H]）と床だけの版。"""
    inputs = np.asarray(inputs, int)
    out = np.empty((len(inputs), st.H, st.H)); out_floor = np.empty_like(out)
    a = st.a
    for q, n in enumerate(inputs):
        t = st.Theta[n] ** 2                                  # [N]
        A = st.D * t[:, None]                                 # [N,H]: Θ_nm² d_jm  (we need Σ_m Θ² d_j d_k)
        Gf = st.D.T @ A                                       # [H,H] = Σ_m Θ_nm² d_jm d_km
        dg = np.sqrt(np.diag(Gf))
        out[q] = Gf / np.outer(dg, dg)
        # 床だけ: (1-a)² の重なり項を落とす。 d_j d_k = a² + a(1-a)(o_j + o_k) + (1-a)² o_j o_k
        qj = (t[:, None] * st.O).sum(0)                       # Σ_{m∈O_j} Θ²
        Gfl = a * a * t.sum() + a * (1 - a) * (qj[:, None] + qj[None, :])
        out_floor[q] = Gfl / np.outer(dg, dg)
    return out, out_floor


# ---------------- Plackett ----------------
def plackett_cum_table(a, b=None, rmax=0.999, n=4000):
    """r 格子と Cov(ρ) = ∫_0^ρ φ₂(a,b;r) dr（累積台形）。"""
    if b is None:
        b = a
    r = np.linspace(0.0, rmax, n + 1)
    q = 1.0 - r * r
    f = np.exp(-(a * a - 2 * r * a * b + b * b) / (2 * q)) / (2 * math.pi * np.sqrt(q))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    return r, cum


def joint_upper_1d(a, b, rho, n=20000, xmax=12.0):
    """独立な検算: P(X>a, Y>b) = ∫_a^∞ φ(x) Φ̄((b − ρx)/√(1−ρ²)) dx（台形）。"""
    x = np.linspace(a, xmax, n + 1)
    g = phi(x) * Phibar((b - rho * x) / math.sqrt(1 - rho * rho))
    return float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(x)))


def cov_bound_42(a, b, rho):
    return rho / (2 * math.pi * math.sqrt(1 - rho * rho)) * math.exp(-(a * a + b * b) / (2 * (1 + rho)))


def R_bound_43(a, b, rho):
    return rho * (1 + a * a) * (1 + b * b) / (a * b * math.sqrt(1 - rho * rho)) * math.exp((a * a + b * b) * rho / (2 * (1 + rho)))


def R_of_rho(rho_pos, a):
    """R(ρ⁺; a, a) = Cov/p²（厳密な 2 変量正規）を格子補間で。"""
    p = float(Phibar(a))
    r, cum = plackett_cum_table(a)
    return np.interp(np.clip(rho_pos, 0, 0.999), r, cum) / (p * p)


def R43_of_rho(rho_pos, a):
    rp = np.clip(rho_pos, 0, 0.999)
    out = np.zeros_like(rp)
    m = rp > 0
    out[m] = rp[m] * (1 + a * a) ** 2 / (a * a * np.sqrt(1 - rp[m] ** 2)) * np.exp((2 * a * a) * rp[m] / (2 * (1 + rp[m])))
    return out


def fano_bound_from_rho(rho_plus_pairs, mu, n_c):
    """(4.5): F ≤ 1 + μ ((n_c−1)/n_c) mean_{j≠k} R(ρ⁺; a, a)、a = Φ̄⁻¹(μ/n_c)。"""
    if mu <= 0 or n_c <= 1:
        return 1.0, 1.0, float("nan")
    p = mu / n_c
    if p >= 0.5:
        return float("nan"), float("nan"), float("nan")
    a = Phibar_inv(p)
    Rex = R_of_rho(rho_plus_pairs, a)
    R43 = R43_of_rho(rho_plus_pairs, a)
    fac = mu * (n_c - 1) / n_c
    return 1.0 + fac * float(Rex.mean()), 1.0 + fac * float(R43.mean()), a


def cantelli(mu, F, nstar):
    if nstar <= mu:
        return 1.0
    s2 = mu * F
    return s2 / (s2 + (nstar - mu) ** 2)


def poisson_tail(mu, nstar):
    kk = math.ceil(nstar)
    return 1.0 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(kk))


def nbeta_table(arm):
    """H4_bound の per-unit n_β（p172c、1 状態）。a0.1 は T=39（M1）、a0.3/a0.5 は T=99。無い unit は nan。"""
    B = ROOT + "/review/stage4_1002/block_1003/stage14_1007_h4/agent_bound/"
    f = {"a0.1": "p172c_a01_t40_real.json", "a0.3": "p172c_a03_t100_real.json", "a0.5": "p172c_a05_t100_real.json"}[arm]
    import json
    d = json.load(open(B + f))
    nb = np.full(100, np.nan)
    for r in d["real"]:
        nb[int(r["i"])] = float(r["n_b"])
    return nb, int(d["task"])
