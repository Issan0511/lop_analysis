# -*- coding: utf-8 -*-
"""wflip_split_0907 の検査（spec `specs/spec_wflip_split_0907.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/.local/bin/pytest -q src/test_wflip_split_0907.py

方針: 判定量を人工データで**外から**作り、実装がそれを取り戻せることを確かめる。
S 検査は「通る側」と「変異対照が落ちる側」を必ず対にする。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import dose_const_5m as D
from src import edge_law_0905 as E
from src import wflip_split_0907 as W
from src import wflip_split_analyze_0907 as A
from src.common import ROOT, load_config

TARGET = 3.041
BD_LOG = Path(ROOT) / "results/boundary_dense_0907/logs/BDref_1216_seed0.npz"


# ---------------------------------------------------------------------- 幾何
def test_gamma_matches_registered_root():
    """`gamma_of` は `dose_const_5m.gamma_for_k` と float64 で一致する。"""
    ks = np.arange(0, 16, dtype=np.float64)
    mine = A.gamma_of(ks, TARGET)
    theirs = D.gamma_for_k(ks, TARGET)
    assert np.allclose(mine, np.asarray(theirs, dtype=np.float64), rtol=0, atol=1e-15)


@pytest.mark.skipif(not BD_LOG.exists(), reason="boundary_dense logs absent")
def test_gamma_matches_logged_column():
    """実データ: 自前の γ(k) が既存ログの `gamma` 列と一致する。"""
    with np.load(BD_LOG, allow_pickle=True) as z:
        fl = z["flip_state"].astype(np.float64)
        g = np.asarray(z["gamma"], dtype=np.float64).ravel()
    assert g.shape[0] == fl.shape[0]
    assert np.allclose(A.gamma_of(fl.sum(axis=1), TARGET), g, rtol=1e-9, atol=1e-12)


def test_centered_mean_has_registered_norm():
    """u と m の定義が用量 ‖µ‖ = 3.041 を再現する（20 次元の中心化平均のノルム）。"""
    for k in range(3, 13):
        flip = np.zeros(15); flip[:k] = 1.0
        u, m = A.u_of(flip, TARGET)
        norm = np.sqrt(float((u ** 2).sum()) + 5.0 * float(m) ** 2)
        assert abs(norm - TARGET) < 1e-9, (k, norm)


@pytest.mark.skipif(not BD_LOG.exists(), reason="boundary_dense logs absent")
def test_task_end_record_is_pre_flip():
    """タスク終端行は**反転前**（次の 1 step で 1 ビットだけ変わる）・タスク内は定数。"""
    with np.load(BD_LOG, allow_pickle=True) as z:
        step = z["step"].astype(np.int64)
        fl = z["flip_state"].astype(np.float64)
    idx = {int(v): i for i, v in enumerate(step)}
    n_checked = 0
    for t0 in range(100000, 300000, W.PERIOD):
        i0, i1 = idx.get(t0), idx.get(t0 + 1)
        if None in (i0, i1):
            continue
        assert int(np.abs(fl[i1] - fl[i0]).sum()) == 1
        i5 = idx.get(t0 + 5000)
        if i5 is not None:
            assert int(np.abs(fl[i5] - fl[i1]).sum()) == 0
        n_checked += 1
    assert n_checked >= 10


# ------------------------------------------------------- (G3) の独立検算
def test_zbar_identity_against_explicit_support():
    """32 点厳密支持から直に平均を取って z̄ = b + w·u + m·Σw_free を確かめる。

    `A.u_of` を使わず、支持を明示的に並べて平均する独立経路。
    """
    rng = np.random.default_rng(0)
    flip = (rng.random(15) < 0.5).astype(np.float64)
    wfl = rng.normal(size=(7, 15)); wfr = rng.normal(size=(7, 5)); b = rng.normal(size=7)
    g = float(A.gamma_of(flip.sum(), TARGET))
    pats = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.float64)   # (32,5)
    x = np.concatenate([np.repeat(flip[None], 32, axis=0), pats], axis=1)      # (32,20)
    z = (x - 0.5 * g) @ np.concatenate([wfl, wfr], axis=1).T + b               # (32,7)
    u, m = A.u_of(flip, TARGET)
    pred = b + wfl @ u + m * wfr.sum(axis=1)
    assert np.allclose(z.mean(axis=0), pred, rtol=0, atol=1e-12)


# ------------------------------------------------------------- S-rank1
def _fake(steps, wfl, flip, zb=None, b=None, wfr=None):
    n = len(steps)
    h = wfl.shape[1]
    wfr = np.zeros((n, h, 5)) if wfr is None else wfr
    b = np.zeros((n, h)) if b is None else b
    zb = np.zeros((n, h)) if zb is None else zb
    return dict(step=np.asarray(steps, dtype=np.int64), zb=zb, b=b, flip=flip,
                wfl=wfl, wfl_step=np.asarray(steps, dtype=np.int64),
                wfr=wfr, wfr_step=np.asarray(steps, dtype=np.int64),
                state_hash_final="x")


def test_rank1_recovers_parallel_and_mutant_fails():
    """Δw を u に厳密平行に作れば r ≈ 0、u を 1 ビットずらすと r が大きい。"""
    rng = np.random.default_rng(1)
    steps = [0, W.PERIOD, 2 * W.PERIOD]
    flip = np.zeros((3, 15))
    flip[0] = (rng.random(15) < 0.5).astype(float)
    flip[1] = flip[0].copy(); flip[1][3] = 1 - flip[1][3]
    flip[2] = flip[1].copy(); flip[2][7] = 1 - flip[2][7]
    h = 6
    wfl = np.zeros((3, h, 15))
    wfl[0] = rng.normal(size=(h, 15))
    for n in (1, 2):
        u, _ = A.u_of(flip[n], TARGET)
        c = rng.normal(size=(h, 1)) - 2.0
        wfl[n] = wfl[n - 1] + c * u[None, :]
    d = _fake(steps, wfl, flip)
    r = A.s_rank1(d, TARGET)
    assert np.nanmax(r["rel"]) < 1e-12, np.nanmax(r["rel"])   # 直接射影なら float64 で 1e-14 台
    rm = A.s_rank1(d, TARGET, mutant=True)
    assert np.nanmedian(rm["rel"]) > 0.1, np.nanmedian(rm["rel"])
    assert np.nanmedian(rm["rho"]) > 1e3, np.nanmedian(rm["rho"])


def test_rho_is_flat_in_step_size_but_rel_is_not():
    """追補 2 の根拠。**厳密に平行**な増分を float32 に丸めて読み戻すと、
    直交残差の絶対値は歩幅に依らない（＝丸めの雑音）ので
      `rel` は 1/‖Δw‖ に比例し、`rho` は歩幅によらず一定。
    判定に使えるのは `rho` の側だけ、ということを実測で固定する。

    ここで出る ρ ≈ 0.3 は**記録の丸めだけ**の床。実走ではさらに 1 タスク
    10,000 step ぶんの float32 演算の丸めが乗る（preflight 実測 ρ ≈ 16）ので、
    登録した閾値 100 はそちらに合わせてある。
    """
    tol = load_config(str(W.CONFIG))["analysis"]["tol"]
    rng = np.random.default_rng(11)
    flip = np.zeros((2, 15))
    flip[0] = (rng.random(15) < 0.5).astype(float)
    flip[1] = flip[0].copy(); flip[1][6] = 1 - flip[1][6]
    u, _ = A.u_of(flip[1], TARGET)
    w0 = rng.normal(scale=0.316, size=(400, 15))
    cast = lambda x: x.astype(np.float32).astype(np.float64)
    got = {}
    for dc in (0.0005, 0.005, 0.05):                     # 歩幅を 100 倍振る
        wfl = np.stack([cast(w0), cast(w0 - dc * u[None, :])])
        r = A.s_rank1(_fake([0, W.PERIOD], wfl, flip), TARGET)
        got[dc] = (float(np.nanmedian(r["rel"])), float(np.nanmedian(r["rho"])))
    rels = [v[0] for v in got.values()]
    rhos = [v[1] for v in got.values()]
    assert max(rels) / min(rels) > 30, got               # rel は歩幅で 2 桁動く
    assert max(rhos) / min(rhos) < 3, got                # rho はほぼ動かない
    assert max(rhos) < float(tol["rank1_rho_max"]), got  # 記録雑音の 100 倍を超えない
    assert min(rhos) > 0.1, got                          # 0 ではない（検査が空虚でない）


def test_rank1_subtraction_form_loses_digits_on_mixed_coefficients():
    """棄却した推定量（差の平方根）が桁落ちする条件を実測して残す。

    行が全部同じなら差の平方根は厳密に 0 になる（`nrm²` と `par²` が同じ数から
    出るため）ので「常に悪い」は**偽**。悪化するのはユニットごとに係数が違う
    現実的な場合で、そこでは float64 でも 1e-9 台まで落ちる。
    """
    rng = np.random.default_rng(12)
    u = rng.normal(size=15) + 5.0                   # 実データ同様、成分の符号が偏った u
    uh = u / np.linalg.norm(u)
    c = rng.normal(size=(400, 1)) - 2.0             # ユニットごとに違う Δc
    dw = c * u[None, :]                             # 厳密に平行
    nrm = np.linalg.norm(dw, axis=1); par = dw @ uh
    subtr = np.sqrt(np.maximum(nrm ** 2 - par ** 2, 0.0)) / nrm
    direct = np.linalg.norm(dw - (dw @ uh)[:, None] * uh[None, :], axis=1) / nrm
    assert float(np.max(direct)) < 1e-12, float(np.max(direct))
    assert float(np.max(subtr)) > 1e-9, float(np.max(subtr))


def test_rank1_flags_a_genuinely_off_direction():
    """Δw に u と直交な成分を混ぜると r がその割合になる（閾値が空虚でないこと）。"""
    rng = np.random.default_rng(2)
    steps = [0, W.PERIOD]
    flip = np.zeros((2, 15))
    flip[0] = (rng.random(15) < 0.5).astype(float)
    flip[1] = flip[0].copy(); flip[1][2] = 1 - flip[1][2]
    u, _ = A.u_of(flip[1], TARGET)
    uh = u / np.linalg.norm(u)
    perp = rng.normal(size=15); perp -= (perp @ uh) * uh; perp /= np.linalg.norm(perp)
    wfl = np.zeros((2, 1, 15))
    wfl[1, 0] = 3.0 * uh + 4.0 * perp          # ⊥/全体 = 4/5
    d = _fake(steps, wfl, flip)
    assert abs(float(A.s_rank1(d, TARGET)["rel"][0, 0]) - 0.8) < 1e-9


# -------------------------------------------------------------- S-zbar
def test_zbar_check_passes_and_drop_m_mutant_fails():
    rng = np.random.default_rng(3)
    steps = [0, W.PERIOD]
    flip = np.stack([(rng.random(15) < 0.5).astype(float)] * 2)
    h = 5
    wfl = rng.normal(size=(2, h, 15)); wfr = rng.normal(size=(2, h, 5))
    b = rng.normal(size=(2, h))
    zb = np.zeros((2, h))
    for n in range(2):
        u, m = A.u_of(flip[n], TARGET)
        zb[n] = b[n] + wfl[n] @ u + m * wfr[n].sum(axis=1)
    d = _fake(steps, wfl, flip, zb=zb, b=b, wfr=wfr)
    assert np.nanmax(A.s_zbar(d, TARGET)) < 1e-12
    assert np.nanmedian(A.s_zbar(d, TARGET, drop_m=True)) > 1e-2


# --------------------------------------------------------- 主判定 s
def test_align_share_endpoints_and_random_baseline():
    steps = [0, W.PERIOD]
    flip = np.zeros((2, 15))
    h = 3
    wfl = np.zeros((2, h, 15))
    wfl[1, 0] = A.ONES * 0.3                       # 全 1 方向のみ → s = 1
    e = np.zeros(15); e[0] = 1.0
    wfl[1, 1] = e - e @ A.ONES_HAT * A.ONES_HAT    # 全 1 と直交 → s = 0
    d = _fake(steps, wfl, flip)
    s, nrm, w0 = A.align_share(d, 1)
    assert abs(s[0] - 1.0) < 1e-12
    assert abs(s[1] - 0.0) < 1e-12
    assert np.isnan(s[2])                          # Δw = 0 のユニットは nan
    # ランダムな 15 次元は 1/15 のまわり
    rng = np.random.default_rng(4)
    wfl2 = np.zeros((2, 400, 15)); wfl2[1] = rng.normal(size=(400, 15))
    s2, _, _ = A.align_share(_fake(steps, wfl2, flip), 1)
    assert 0.02 < float(np.mean(s2)) < 0.12        # 期待 1/15 = 0.067


def test_align_share_matches_hand_built_mixture():
    """全 1 成分 a・直交成分 b で作れば s = a²/(a²+b²) を返す。"""
    steps = [0, W.PERIOD]
    flip = np.zeros((2, 15))
    q = np.zeros(15); q[0] = 1.0
    q = q - (q @ A.ONES_HAT) * A.ONES_HAT
    q /= np.linalg.norm(q)
    wfl = np.zeros((2, 1, 15))
    wfl[1, 0] = 2.0 * A.ONES_HAT + 1.0 * q
    s, _, _ = A.align_share(_fake(steps, wfl, flip), 1)
    assert abs(float(s[0]) - 4.0 / 5.0) < 1e-12


# ------------------------------------------------------------ J の分解
def test_j_terms_sum_to_geometric_jump():
    """3 項の和が幾何の跳び w·(u_new − u_old) + Δm·Σw_free に厳密に等しい。"""
    rng = np.random.default_rng(5)
    lo = W.PERIOD
    steps = [0, lo, lo + 1, 2 * lo]
    flip = np.zeros((4, 15))
    flip[0] = flip[1] = (rng.random(15) < 0.5).astype(float)
    flip[2] = flip[3] = flip[1].copy()
    flip[2][4] = 1 - flip[2][4]
    h = 8
    wfl_full = rng.normal(size=(4, h, 15)); wfr_full = rng.normal(size=(4, h, 5))
    d = dict(step=np.array(steps, dtype=np.int64),
             zb=np.zeros((4, h)), b=np.zeros((4, h)), flip=flip,
             wfl=wfl_full[[0, 1, 3]], wfl_step=np.array([0, lo, 2 * lo], dtype=np.int64),
             wfr=wfr_full[[0, 1, 3]], wfr_step=np.array([0, lo, 2 * lo], dtype=np.int64),
             state_hash_final="x")
    t = A.j_terms(d, TARGET, [lo, lo])
    assert np.allclose(t["bit"] + t["gfl"] + t["gfr"], t["geo"], rtol=0, atol=1e-12)


def test_j_gamma_term_is_not_negligible_by_construction():
    """γ 項を落とすと幾何の跳びが変わる（＝γ 項の分離が空虚でない）。"""
    rng = np.random.default_rng(6)
    lo = W.PERIOD
    steps = [0, lo, lo + 1, 2 * lo]
    flip = np.zeros((4, 15)); flip[:, :8] = 1.0
    flip[2:, 8] = 1.0                                    # 0 -> 1（k が 8 -> 9）
    h = 4
    wfl = rng.normal(size=(3, h, 15)) - 0.5              # 全体に負へ寄せる
    wfr = rng.normal(size=(3, h, 5))
    d = dict(step=np.array(steps, dtype=np.int64), zb=np.zeros((4, h)),
             b=np.zeros((4, h)), flip=flip, wfl=wfl,
             wfl_step=np.array([0, lo, 2 * lo], dtype=np.int64), wfr=wfr,
             wfr_step=np.array([0, lo, 2 * lo], dtype=np.int64), state_hash_final="x")
    t = A.j_terms(d, TARGET, [lo, lo])
    gam = np.abs(t["gfl"] + t["gfr"]).mean()
    assert gam > 1e-3, gam


# ------------------------------------------------------------- 記録器
def _tiny_state():
    cfg = E.build_cfg(W.CONFIG)
    cfg["common"]["seeds"] = [0]
    return E.setup_arm_dial(cfg, E._arm(cfg, "WSref_1216"), "cpu")


def test_recorder_records_only_task_ends_and_reads_the_flip_block():
    st = _tiny_state()
    steps = [0, 1, 5, W.PERIOD, W.PERIOD + 1, 2 * W.PERIOD]
    rec = W.WFlipRecorder(steps, st)
    assert list(rec.w_flip_steps) == [0, W.PERIOD, 2 * W.PERIOD]
    assert rec.w_flip.shape == (3, st["R"], st["hidden"][0], 15)
    for s in steps:
        rec(st, s)
    want = st["net"].Ws[0][:, :, :15].detach().cpu().numpy()
    assert np.allclose(rec.w_flip[0], want, rtol=0, atol=0)
    assert rec.w_flip.shape[-1] == W.N_FLIP == 15


def test_recorder_keeps_the_boundary_b_column():
    """`BoundaryRecorder` の `layer1_b` を壊していない。"""
    st = _tiny_state()
    steps = [0, W.PERIOD]
    rec = W.WFlipRecorder(steps, st)
    rec(st, 0)
    assert "b" in rec.unit
    assert np.allclose(rec.unit["b"][0], st["net"].bs[0].detach().cpu().numpy())


def test_flip_and_free_slices_are_disjoint_and_cover_the_input():
    assert W.FLIP_SLICE == slice(0, 15)
    assert E.FREE_SLICE == slice(15, 20)
    assert W.N_FLIP + E.N_FREE == 20


# ------------------------------------------------------------- ラベル
def _verdict(s_point, ci, dw_over_w0, bit, gfl, gfr, jmeas, jci, rank1=16.0,
             rank1_m=9.0e4, zbar=1e-9, zbar_m=0.5, rel_gap=0.0):
    return dict(s=dict(point=s_point, ci=list(ci)), dw_over_w0=dw_over_w0,
                dw_norm=1.0, w0_norm=1.0, n_seeds=10,
                J=dict(measured=jmeas, ci=list(jci), geometric=jmeas,
                       rel_gap=rel_gap, bit_plus_gamma=bit + gfl + gfr,
                       terms=dict(bit=dict(mean=bit * jmeas, share=bit),
                                  gfl=dict(mean=gfl * jmeas, share=gfl),
                                  gfr=dict(mean=gfr * jmeas, share=gfr))),
                s_rank1=dict(rho=rank1, rho_mutant=rank1_m),
                s_zbar=dict(value=zbar, mutant=zbar_m))


def _label(tmp_path, v):
    import src.wflip_split_analyze_0907 as AA
    res = {"arms": {"WSref_1216": v}, "config": {}, "s_null": {},
           "checks": {}}
    # analyze() の判定部だけを踏むために、最小の logdir を作らずに再現する
    cfg = load_config(str(W.CONFIG))["analysis"]
    lab, floor, tol = cfg["labels"], cfg["floor"], cfg["tol"]
    j = v
    if not (j["s_rank1"]["rho"] < float(tol["rank1_rho_max"])
            and j["s_rank1"]["rho_mutant"] > float(tol["rank1_rho_mutant_min"])):
        return "NOT_DETERMINED", "NOT_DETERMINED"
    lo, hi = j["s"]["ci"]
    if j["dw_over_w0"] < float(floor["dw_over_w0"]):
        a = "NOT_DETERMINED"
    elif lo > float(lab["bias_like_min"]):
        a = "BIAS_LIKE"
    elif hi < float(lab["pattern_like_max"]):
        a = "PATTERN_LIKE"
    elif lo >= float(lab["pattern_like_max"]) and hi <= float(lab["bias_like_min"]):
        a = "MIXED"
    else:
        a = "NOT_DETERMINED"
    return a, None


def test_label_rules(tmp_path):
    assert _label(tmp_path, _verdict(0.93, (0.88, 0.96), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010)))[0] == "BIAS_LIKE"
    assert _label(tmp_path, _verdict(0.10, (0.07, 0.14), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010)))[0] == "PATTERN_LIKE"
    assert _label(tmp_path, _verdict(0.50, (0.40, 0.60), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010)))[0] == "MIXED"
    # 床を割ると（動いていない）判定しない
    assert _label(tmp_path, _verdict(0.93, (0.88, 0.96), 0.3, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010)))[0] == "NOT_DETERMINED"
    # CI が 0.3 を跨いで 0.7 に届かない → どのラベルにも収まらない
    assert _label(tmp_path, _verdict(0.4, (0.25, 0.65), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010)))[0] == "NOT_DETERMINED"
    # S-rank1 が落ちたら主判定を出さない（ρ が記録雑音の 100 倍を超える）
    assert _label(tmp_path, _verdict(0.93, (0.88, 0.96), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010), rank1=500.0))[0] == "NOT_DETERMINED"
    # S-rank1 の変異対照が落ちない（＝検査が空虚）なら通さない
    assert _label(tmp_path, _verdict(0.93, (0.88, 0.96), 0.7, 1.0, 0.0, 0.0,
                                     0.008, (0.006, 0.010), rank1_m=50.0))[0] == "NOT_DETERMINED"
