# -*- coding: utf-8 -*-
"""zeta_field_0908 の検査（読み規則 `specs/spec_zeta_field_0908.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m pytest -q src/test_zeta_field_0908.py

肝は 2 本: **ζ で閉じる世界**と**ζ/W で閉じる世界**を人工データで作り、判定が
それぞれを正しく別のラベルに振り分けること（＝検査が空虚でないこと）。
"""
from __future__ import annotations

import numpy as np
import pytest

from src import drift_law_0907 as D
from src import zeta_field_0908 as Z

NB = 200          # 検査ではブートストラップを軽くする


# ------------------------------------------------------------- 前提の固定
def test_zeta_definition_is_inherited_unchanged():
    """ζ は N の**両端**を除く 8 タスク平均（drift_law 追補 1）。ここを変えない。"""
    assert D.EXCLUDE == (0, 1)
    assert D.HALF == 4
    assert D.WINDOW == (10, 199)


# --------------------------------------------------------- 零点の内挿
def test_zero_crossing_is_exact_on_a_line():
    x = np.array([-4.0, -3.0, -2.0, -1.0, 0.0])
    y = 0.5 - 0.2 * (x + 4.0)              # 零点は x = -1.5
    assert abs(Z.zero_crossing(dict(x=x, y=y)) - (-1.5)) < 1e-12


def test_zero_crossing_requires_exactly_one_plus_to_minus():
    x = np.arange(6.0)
    assert np.isnan(Z.zero_crossing(dict(x=x, y=np.ones(6))))          # 交差なし
    assert np.isnan(Z.zero_crossing(dict(x=x, y=-np.ones(6))))         # 交差なし
    two = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])                  # + → − が 3 本
    assert np.isnan(Z.zero_crossing(dict(x=x, y=two)))
    one = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    assert abs(Z.zero_crossing(dict(x=x, y=one)) - 2.5) < 1e-12
    # − → + だけ（不安定な不動点）は採らない
    up = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert np.isnan(Z.zero_crossing(dict(x=up, y=up)))


def test_profile_drops_thin_bins_and_reports_n():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = -0.02 * x
    p = Z.profile(x, y)
    assert p["x"].size >= 8 and p["n"] == 5000
    assert np.all(np.diff(p["x"]) > 0)                 # ビン中央値は単調
    assert Z.profile(x[:50], y[:50])["x"].size == 0     # 足りなければ空


def test_drift_floor_needs_opposite_signs_and_size():
    ok = dict(x=np.arange(5.0), y=np.array([0.02, 0.01, 0.0, -0.01, -0.02]))
    assert Z.drift_floor_ok(ok)
    small = dict(x=np.arange(5.0), y=np.array([0.002, 0.001, 0.0, -0.001, -0.002]))
    assert not Z.drift_floor_ok(small)                 # 床 0.005 に届かない
    same = dict(x=np.arange(5.0), y=np.array([0.02, 0.015, 0.012, 0.011, 0.01]))
    assert not Z.drift_floor_ok(same)                  # 符号が変わらない


# ---------------------------------------------- 2 つの世界を作り分ける
def _world(kind: str, n_seed: int = 10, n: int = 6000, slope: float = 0.02,
           zeta0: float = -3.0, seed: int = 1) -> list[dict]:
    """kind='zeta': 零点が固定 ζ0。 kind='ratio': 零点が −W（上端則の世界）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_seed):
        W = np.exp(rng.uniform(np.log(1.5), np.log(4.5), n))
        center = zeta0 if kind == "zeta" else -W
        zeta = center + rng.uniform(-3.0, 3.0, n)
        N = -slope * (zeta - center) + rng.normal(0.0, 0.01, n)
        # p_up は「ζ と W から決まる」量の代理（32 点の一様版）
        p_up = np.clip(0.5 + zeta / (2.0 * W), 0.0, 1.0)
        rows.append(dict(N=N, zeta=zeta, p_up=p_up, W=W))
    return rows


def _judge(rows):
    edges = Z.tercile_edges(rows)
    r = Z.ratios(Z.terciles(rows, edges))
    ci = Z.boot_ratios(rows, edges, nboot=NB, seed=Z.RNG_SEED)
    return r, ci, Z.label_closure(r, ci)


def test_world_that_closes_in_zeta_is_labelled_so():
    r, ci, lab = _judge(_world("zeta"))
    assert r["W_spread"] >= Z.FLOOR_W_SPREAD, r["W_spread"]
    assert r["drift_ok"]
    assert lab == "CLOSES_IN_ZETA", (lab, r, ci)
    assert r["A"] < 1.1 and r["B"] > 1.5, (r["A"], r["B"])


def test_world_that_closes_in_the_ratio_is_labelled_so():
    r, ci, lab = _judge(_world("ratio"))
    assert r["W_spread"] >= Z.FLOOR_W_SPREAD, r["W_spread"]
    assert r["drift_ok"]
    assert lab == "CLOSES_IN_RATIO", (lab, r, ci)
    assert r["B"] < 1.1 and r["A"] > 1.5, (r["A"], r["B"])
    # 上端則の世界なら ζ*/W̃ は −1 のまわり
    ter = Z.terciles(_world("ratio"), Z.tercile_edges(_world("ratio")))
    for t in ter:
        assert abs(t["zeta_star"] / t["W_med"] + 1.0) < 0.15, t


def test_the_two_worlds_are_not_both_accepted():
    """同じ判定器が両方を同じラベルにしないこと（＝識別力がある）。"""
    _, _, a = _judge(_world("zeta"))
    _, _, b = _judge(_world("ratio"))
    assert a != b and {a, b} == {"CLOSES_IN_ZETA", "CLOSES_IN_RATIO"}


# ------------------------------------------------------------------ 床
def test_narrow_W_is_not_determined():
    """W が広がらないと A と B は同じ量になるので判定しない（床 1）。"""
    rng = np.random.default_rng(2)
    rows = []
    for _ in range(10):
        W = np.exp(rng.uniform(np.log(2.0), np.log(2.2), 6000))   # 広がり 1.1 倍
        zeta = -3.0 + rng.uniform(-3.0, 3.0, 6000)
        rows.append(dict(N=-0.02 * (zeta + 3.0) + rng.normal(0, 0.01, 6000),
                         zeta=zeta, p_up=np.clip(0.5 + zeta / (2 * W), 0, 1), W=W))
    r, ci, lab = _judge(rows)
    assert r["W_spread"] < Z.FLOOR_W_SPREAD
    assert lab == "NOT_DETERMINED", (lab, r)


def test_weak_drift_is_not_determined():
    """両端ビンの |E[N]| が床に届かなければ判定しない（床 2）。"""
    rows = _world("zeta", slope=0.0005)
    r, ci, lab = _judge(rows)
    assert not r["drift_ok"]
    assert lab == "NOT_DETERMINED", (lab, r)


# --------------------------------------------------------------- ラベル
def _r(A, B, C=1.0, W=2.0, ok=True):
    return dict(A=A, B=B, C=C, W_spread=W, drift_ok=ok,
                n_valid_zeta=3, n_valid_p=3)


def _c(A, B, C=(1.0, 1.1)):
    return dict(A=list(A), B=list(B), C=list(C))


def test_label_rules_use_the_ci():
    assert Z.label_closure(_r(1.02, 2.0), _c((0.98, 1.10), (1.7, 2.4))) == "CLOSES_IN_ZETA"
    assert Z.label_closure(_r(2.0, 1.02), _c((1.7, 2.4), (0.98, 1.10))) == "CLOSES_IN_RATIO"
    assert Z.label_closure(_r(2.0, 1.8), _c((1.7, 2.4), (1.5, 2.1))) == "NEITHER"
    # CI が 1.25 を跨いだら決めない
    assert Z.label_closure(_r(1.2, 2.0), _c((1.05, 1.40), (1.7, 2.4))) == "NOT_DETERMINED"
    # 床を割ったら決めない
    assert Z.label_closure(_r(1.02, 2.0, W=1.2), _c((0.98, 1.10), (1.7, 2.4))) == "NOT_DETERMINED"
    assert Z.label_closure(_r(1.02, 2.0, ok=False), _c((0.98, 1.10), (1.7, 2.4))) == "NOT_DETERMINED"
    # 両方 1.25 以下は識別できていない
    assert Z.label_closure(_r(1.02, 1.03), _c((0.98, 1.10), (0.99, 1.12))) == "NOT_DETERMINED"


def test_pstar_label_rules():
    assert Z.label_pstar(_r(1.0, 1.0, C=1.05), _c((1, 1), (1, 1), (0.95, 1.15))) == "INVARIANT"
    assert Z.label_pstar(_r(1.0, 1.0, C=2.0), _c((1, 1), (1, 1), (1.6, 2.5))) == "VARIES"
    assert Z.label_pstar(_r(1.0, 1.0, C=1.3), _c((1, 1), (1, 1), (1.1, 1.6))) == "NOT_DETERMINED"
    assert Z.label_pstar(_r(1.0, 1.0, C=1.05, ok=False),
                         _c((1, 1), (1, 1), (0.95, 1.15))) == "NOT_DETERMINED"


def test_terciles_are_cut_once_on_the_pooled_data():
    """ブートストラップで三分位の境界が動かない（spec §6-7）。"""
    rows = _world("ratio", n_seed=4, n=2000)
    e = Z.tercile_edges(rows)
    sub = Z.tercile_edges([rows[0]])
    assert not np.allclose(e, sub)          # seed ごとに切ると違う値になる（＝固定が効く）
    t1 = Z.terciles(rows, e)
    t2 = Z.terciles(rows, e)
    assert [x["W_med"] for x in t1] == [x["W_med"] for x in t2]
