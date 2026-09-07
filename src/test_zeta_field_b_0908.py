# -*- coding: utf-8 -*-
"""zeta_field_b_0908 の検査（読み規則 `specs/spec_zeta_field_b_0908.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m pytest -q src/test_zeta_field_b_0908.py

肝は 3 本:
  (1) PAVA が非増加を厳密に返し、重み付き最小二乗として正しいこと
  (2) 3 つの世界（ζ で閉じる / 上端則 / affine）を作ると別々のラベルになること
  (3) 訂正 A・B が `zeta_field_0908` の落ち方を実際に救うこと
"""
from __future__ import annotations

import numpy as np

from src import zeta_field_b_0908 as B


# ------------------------------------------------------------- PAVA
def test_pava_is_nonincreasing_and_preserves_weighted_mean():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40); w = rng.uniform(0.5, 2.0, 40)
    f = B.pava_nonincreasing(y, w)
    assert f.size == y.size
    assert np.all(np.diff(f) <= 1e-12)                       # 非増加
    assert abs(float((f * w).sum() - (y * w).sum())) < 1e-9  # 重み付き総和が保存
    # 既に非増加なら恒等
    z = np.sort(rng.normal(size=20))[::-1]
    assert np.allclose(B.pava_nonincreasing(z, np.ones(20)), z)


def test_pava_pools_only_the_violating_block():
    y = np.array([3.0, 1.0, 2.0, 0.0])       # 1 と 2 が違反 → 平均 1.5 に潰れる
    f = B.pava_nonincreasing(y, np.ones(4))
    assert np.allclose(f, [3.0, 1.5, 1.5, 0.0])


REAL_X = np.array([-2.761, -2.433, -2.228, -2.047, -1.876,
                   -1.706, -1.525, -1.327, -1.092, -0.716])
REAL_Y = np.array([0.0154, 0.0282, 0.0236, 0.0162, 0.0148,
                   0.0026, -0.0020, -0.0018, 0.0064, 0.0024])


def test_pava_would_have_destroyed_the_crossing():
    """**追補 1 の根拠**: 単調回帰は棄却する（採用していたら零点が消えていた）。

    `zeta_field_0908` の LRoff0 第 1 三分位の実プロファイル。折れ目の上下に
    線形域が 2 つあるので E[N|ζ] は単調でなく（上がる → 沈む → 0 へ戻る）、
    単調当てはめは沈下の谷を上側の平坦域に吸収して**全域を正**にしてしまう。
    """
    f = B.pava_nonincreasing(REAL_Y, np.full(10, 6000.0))
    assert np.all(np.diff(f) <= 1e-12)
    assert f.min() > 0.0, f              # 零点が消える
    assert np.isnan(B.stable_zero(REAL_X, f))


def test_real_profile_passes_the_corrected_floor():
    """同じ列が、追補 1 の 3 条件（交差 1 本・振れ幅・右下がり）では通る。"""
    star, rng_, ok = B.star_of(dict(x=REAL_X, y=REAL_Y, c=np.full(10, 6000.0)))
    assert ok, (star, rng_, B.spearman(REAL_X, REAL_Y))
    assert rng_ > B.FLOOR_RANGE
    assert B.spearman(REAL_X, REAL_Y) <= B.FLOOR_RHO
    assert -2.0 < star < -1.2, star      # 生の内挿は -1.60
    # 旧 `zeta_field_0908` の床（両端ビンが逆符号）はこの列では落ちていた
    assert not (REAL_Y[0] * REAL_Y[-1] < 0)


def test_stable_zero_counts_only_plus_to_minus():
    x = np.arange(6.0)
    assert np.isnan(B.stable_zero(x, np.ones(6)))
    assert np.isnan(B.stable_zero(x, -np.ones(6)))
    assert np.isnan(B.stable_zero(x, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])))
    assert abs(B.stable_zero(x, np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])) - 2.5) < 1e-12
    # − → + だけ（不安定な不動点）は数えない
    assert np.isnan(B.stable_zero(x, np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])))


def test_spearman_floor_rejects_a_flat_but_wide_profile():
    """振れ幅だけ大きくて右下がりでない列は落とす（雑音で床を通さない）。"""
    x = np.arange(10.0)
    y = np.array([0.02, -0.03, 0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.03, -0.02])
    _, rng_, ok = B.star_of(dict(x=x, y=y, c=np.full(10, 6000.0)))
    assert rng_ > B.FLOOR_RANGE          # 振れ幅は通るのに
    assert not ok                        # 右下がりでないので落ちる


# ------------------------------------------- 3 つの世界を作り分ける
def _world(kind: str, n_seed: int = 6, n: int = 9000, slope: float = 0.03,
           seed: int = 3) -> list[dict]:
    """kind: 'zeta'（零点が固定）/ 'edge'（零点が −W）/ 'affine'（零点が α + βW）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_seed):
        W = np.exp(rng.uniform(np.log(1.2), np.log(4.5), n))
        if kind == "zeta":
            c = np.full(n, -2.5)
        elif kind == "edge":
            c = -W
        else:
            c = 1.5 - 1.5 * W
        zeta = c + rng.uniform(-2.5, 2.5, n)
        N = -slope * (zeta - c) + rng.normal(0.0, 0.012, n)
        p_up = np.clip(0.5 + zeta / (2.0 * W), 0.0, 1.0)
        rows.append(dict(N=N, zeta=zeta, p_up=p_up, W=W))
    return rows


def _fit(rows):
    return B.arm_fit(rows, B.sextile_edges(rows))


def test_three_worlds_give_the_expected_slope_and_intercept():
    fz, fe, fa = _fit(_world("zeta")), _fit(_world("edge")), _fit(_world("affine"))
    for f in (fz, fe, fa):
        assert f["qualified"], f
        assert f["n_good"] >= B.MIN_SEX
    assert abs(fz["beta"]) < 0.2, fz["beta"]                     # ζ で閉じる → β ≈ 0
    assert abs(fe["beta"] + 1.0) < 0.2, fe["beta"]               # 上端則 → β ≈ −1
    assert abs(fe["alpha_hat"]) < 0.2, fe["alpha_hat"]           #        かつ α ≈ 0
    assert abs(fa["beta"] + 1.5) < 0.3, fa["beta"]               # affine → β ≈ −1.5
    assert fa["alpha_hat"] > 0.2, fa["alpha_hat"]                #        かつ α > 0


def _labels_from(kind: str):
    """同じ世界の腕を 6 本並べて判定部だけ通す。"""
    fits = {f"A{i}": _fit(_world(kind, seed=10 + i)) for i in range(6)}
    m = B.medians(fits, tuple(fits))
    b = [f["beta"] for f in fits.values()]
    a = [abs(f["alpha_hat"]) for f in fits.values()]
    ci = dict(beta=[float(np.min(b)), float(np.max(b))],
              abs_alpha_hat=[float(np.min(a)), float(np.max(a))],
              C=[np.nan, np.nan])
    return B.label_form(m, ci), m, ci


def test_labels_separate_the_three_worlds():
    lz, _, _ = _labels_from("zeta")
    le, _, _ = _labels_from("edge")
    la, _, _ = _labels_from("affine")
    assert lz == "CLOSES_IN_ZETA", lz
    assert le == "EDGE_LAW", le
    assert la == "AFFINE", la
    assert len({lz, le, la}) == 3


# ------------------------------------------------------------------ 床
def test_weak_drift_disqualifies_by_range_not_by_end_bins():
    """訂正 B: 床は振れ幅。傾きを 1/30 にすると資格を失う。"""
    f = _fit(_world("edge", slope=0.001))
    assert not f["qualified"], f
    assert f["n_good"] < B.MIN_SEX


def test_narrow_W_disqualifies():
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(6):
        W = np.exp(rng.uniform(np.log(2.0), np.log(2.2), 9000))
        zeta = -2.5 + rng.uniform(-2.5, 2.5, 9000)
        rows.append(dict(N=-0.03 * (zeta + 2.5) + rng.normal(0, 0.012, 9000),
                         zeta=zeta, p_up=np.clip(0.5 + zeta / (2 * W), 0, 1), W=W))
    f = _fit(rows)
    assert f["W_spread"] < B.FLOOR_W_SPREAD
    assert not f["qualified"]


def test_arm_floor_is_not_a_full_and():
    """訂正 C: 六分位 6 本中 4 本・腕 11 本中 5 本で足りる（全部の AND にしない）。"""
    assert B.MIN_SEX == 4 and B.NSEX == 6
    assert B.MIN_ARMS == 5 and len(B.HELD_OUT) == 11
    fits = {f"A{i}": _fit(_world("edge", seed=20 + i)) for i in range(5)}
    fits["bad"] = _fit(_world("edge", slope=0.001, seed=99))     # 資格なしを 1 本混ぜる
    m = B.medians(fits, tuple(fits))
    assert m["n_arms"] == 5 and not fits["bad"]["qualified"]
    ci = dict(beta=[-1.1, -0.9], abs_alpha_hat=[0.0, 0.15], C=[np.nan, np.nan])
    assert B.label_form(m, ci) == "EDGE_LAW"


def test_not_determined_when_too_few_arms():
    m = dict(beta=-1.0, abs_alpha_hat=0.1, n_arms=4)
    ci = dict(beta=[-1.1, -0.9], abs_alpha_hat=[0.0, 0.15], C=[np.nan, np.nan])
    assert B.label_form(m, ci) == "NOT_DETERMINED"


def test_pstar_label_and_quantisation_guard():
    assert B.label_pstar(dict(C=1.05, n_arms=4), dict(C=[0.95, 1.15])) == "INVARIANT"
    assert B.label_pstar(dict(C=2.0, n_arms=4), dict(C=[1.6, 2.5])) == "VARIES"
    assert B.label_pstar(dict(C=1.3, n_arms=4), dict(C=[1.1, 1.6])) == "NOT_DETERMINED"
    assert B.label_pstar(dict(C=1.05, n_arms=2), dict(C=[0.95, 1.15])) == "NOT_DETERMINED"
    # 量子化で分解できない腕（p* < 2/32）は median_C から落ちる
    fits = {"a": dict(C=1.1, p_median=0.01), "b": dict(C=1.2, p_median=0.10),
            "c": dict(C=1.3, p_median=0.20)}
    assert B.median_C(fits, ("a", "b", "c"))["n_arms"] == 2


def test_held_out_arms_do_not_overlap_the_burned_ones():
    assert not (set(B.HELD_OUT) & set(B.BURNED))
    assert set(B.PSTAR_ARMS) <= set(B.HELD_OUT)
    assert len(B.PSTAR_ARMS) == 6
