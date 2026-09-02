"""valley_off_0903 の単体検査（事前登録 specs/spec_valley_off_0903.md）。

判定ラベル・予測の照合・凍結率の定義・代用（対照の p_hat）・config の防御を見る。
本走を回さずに通るものだけを置く。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from .common import ROOT, load_config
from .valley_off_0903 import (ARM_ORDER, BASELINE, CONTROL_ORDER, LABEL_ORDER,
                              PRIMARY, _geometry, _labels, _sign_test,
                              _unit_summary, validate_config)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(str(Path(ROOT) / "configs" / "valley_off_0903.yaml"))


# ---------------------------------------------------------------------------
# config の防御
# ---------------------------------------------------------------------------
def test_registered_config_validates(cfg):
    for stage in ("preflight", "smoke", "run", "analyze"):
        validate_config(cfg, stage=stage)


def test_arms_are_off_arms(cfg):
    assert [a["name"] for a in cfg["arms"]] == list(ARM_ORDER)
    for arm in cfg["arms"]:
        assert arm["target_dose"] is None and arm["target_mu_norm"] is None
        assert arm["centered_layers"] == []
        assert float(arm["dial"]) == 1.0


@pytest.mark.parametrize("mutate", [
    lambda c: c["arms"][0].update(target_dose=12.16),       # オラクルを掛けたら別の走
    lambda c: c["arms"][0].update(target_mu_norm=3.041),
    lambda c: c["arms"][0].update(centered_layers=[1]),
    lambda c: c["arms"][0].update(dial=3.0),
    lambda c: c["arms"][0].update(hidden=[100, 100]),       # 2 層は段 2（対象外）
    lambda c: c["common"].update(generator_offset=20260903),
    lambda c: c["common"].update(total_steps=1_000_000),
    lambda c: c["common"].update(seeds=[0, 1, 2]),
    lambda c: c["phase1"].update(unfit_floor=1e-12),
    lambda c: c["phase1"].update(recalibrate_floor=True),
    lambda c: c["phase1"].update(onset_threshold=0.1),
    lambda c: c["phase1"].update(bootstrap_seed=20260829),  # 使い回し禁止
    lambda c: c["phase1"].update(window_points_are_task_ends_only=False),
    lambda c: c["intervention"].update(oracle=True),
    lambda c: c["valley_off"]["design"].update(freeze_source_column="layer1_zbar"),
    lambda c: c["valley_off"]["design"].update(freeze_phi_prime_threshold=1e-3),
    lambda c: c["valley_off"]["labels"].update(order=list(reversed(LABEL_ORDER))),
    lambda c: c["controls"].update(baseline="E_off"),
    lambda c: c["sanity"].update(omp_num_threads=4),
])
def test_deviations_are_rejected(cfg, mutate):
    c = copy.deepcopy(cfg)
    mutate(c)
    with pytest.raises(ValueError):
        validate_config(c, stage="run")


# ---------------------------------------------------------------------------
# 幾何（谷底・凍結深さ）
# ---------------------------------------------------------------------------
def test_registered_valley_depths_match_numeric_roots(cfg):
    tol = float(cfg["valley_off"]["design"]["u_fr_spec_rel_tol"])
    for arm, want in (("G_off", 0.7519), ("S_off", 1.2785)):
        geo = _geometry(cfg, arm)
        assert abs(geo["u_star_numeric"] - want) <= tol * want
        assert geo["has_valley"]
    for arm in CONTROL_ORDER:
        geo = _geometry(cfg, arm)
        assert not geo["has_valley"]          # ReLU / ELU / leaky に谷は無い


def test_freeze_depth_uses_the_registered_threshold(cfg):
    geo = _geometry(cfg, PRIMARY)
    assert geo["threshold"] == 1e-6
    assert geo["u_fr_numeric"] > geo["u_star_numeric"] > 0


# ---------------------------------------------------------------------------
# 判定ラベル（spec §6）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a1,a2,a3,onset,want", [
    (True, True, True, 10, "FLIGHT_WITHOUT_ORACLE"),
    (True, False, True, 10, "WORSE_UNTESTABLE_AT_CEILING"),
    (False, False, False, 0, "FLIGHT_NEEDS_OFFSET"),
    (False, False, True, 2, "FLIGHT_NEEDS_OFFSET"),      # ≤2 が先に当たる
    (False, False, True, 5, "FLIGHT_SLOW"),
    (True, True, False, 10, "PARTIAL"),
    (False, True, False, 9, "PARTIAL"),
])
def test_label_table(cfg, a1, a2, a3, onset, want):
    label, hits = _labels(cfg, a1, a2, a3, onset)
    assert label == want
    assert want in hits


def test_co_satisfied_keeps_every_row_that_held(cfg):
    label, hits = _labels(cfg, True, False, True, 10)
    assert label == "WORSE_UNTESTABLE_AT_CEILING"
    assert "FLIGHT_WITHOUT_ORACLE" in hits      # 当たっていた行も残す


def test_label_order_is_the_registered_one(cfg):
    assert list(cfg["valley_off"]["labels"]["order"]) == list(LABEL_ORDER)


# ---------------------------------------------------------------------------
# 符号検定
# ---------------------------------------------------------------------------
def test_sign_test_counts_and_p():
    got = _sign_test(np.array([1.0, 1.0, 1.0, -1.0, 0.0]))
    assert (got["n_positive"], got["n_negative"], got["n_ties"]) == (3, 1, 1)
    assert 0.0 < got["p_two_sided"] <= 1.0
    assert _sign_test(np.zeros(4))["n_ties"] == 4


# ---------------------------------------------------------------------------
# ユニット別量（新規腕の列あり / 対照の代用）
# ---------------------------------------------------------------------------
def _fake_log(tmp_path: Path, *, with_units: bool, period: int = 10_000) -> Path:
    steps = np.arange(0, 5_000_001, 1_000_000 // 2, dtype=np.int64)
    steps = np.array([490 * period, 495 * period, 500 * period], dtype=np.int64)
    n, units = len(steps), 4
    p_hat = np.array([[0.0, 0.0, 0.5, 0.0]] * n, dtype=np.float32)
    zbar = np.array([[-9.0, -12.0, 1.0, -3.0]] * n, dtype=np.float32)
    payload = dict(step=steps, layer1_p_hat=p_hat, layer1_zbar=zbar,
                   flip_state=np.zeros((n, 5), dtype=np.float32),
                   task_period=np.int64(period))
    if with_units:
        payload["layer1_mob"] = np.array(
            [[1e-9, 1e-12, 0.6, 1e-3]] * n, dtype=np.float32)
        payload["layer1_zmax"] = np.array(
            [[-8.0, -11.0, 2.0, -0.5]] * n, dtype=np.float32)
        payload["layer1_v_unit"] = np.ones((n, units), dtype=np.float32)
    path = tmp_path / f"arm_{'units' if with_units else 'proxy'}.npz"
    np.savez_compressed(path, **payload)
    return path


def test_unit_summary_uses_this_runs_logger(cfg, tmp_path):
    geo = _geometry(cfg, PRIMARY)
    got = _unit_summary(cfg, _fake_log(tmp_path, with_units=True), geo, proxy=False)
    assert got["submerged_source"] == "layer1_zmax"
    assert got["submerged_frac"] == 0.75                 # zmax <= 0 の 3/4
    assert got["frozen_source"] == "layer1_mob"
    assert got["frozen_frac"] == 0.5                     # |mob| < 1e-6 の 2/4
    # 谷の向こう = zmax <= -u*（GELU β=1 で 0.7519）
    assert got["beyond_valley_frac"] == 0.5     # -0.5 は谷底 -0.7519 の手前
    assert got["depth_median"] == pytest.approx(9.0)     # 沈下ユニットの -zbar の中央値


def test_unit_summary_proxy_is_exact_on_relu_and_blank_elsewhere(cfg, tmp_path):
    path = _fake_log(tmp_path, with_units=False)
    relu = _unit_summary(cfg, path, _geometry(cfg, "R_off"), proxy=True)
    assert relu["submerged_source"].startswith("layer1_p_hat")
    assert relu["submerged_frac"] == 0.75
    assert relu["frozen_frac"] == 0.75                   # ReLU では E_x phi' = p_hat
    assert "exact on ReLU" in relu["frozen_source"]
    assert np.isnan(relu["beyond_valley_frac"])          # ReLU に谷は無い
    elu = _unit_summary(cfg, path, _geometry(cfg, "E_off"), proxy=True)
    assert np.isnan(elu["frozen_frac"])                  # 代用が成立しない = 空欄
    assert "proxy invalid" in elu["frozen_source"]


def test_baseline_is_the_relu_off_arm(cfg):
    assert BASELINE == "R_off" and cfg["controls"]["baseline"] == BASELINE
    assert cfg["controls"]["reference_run"] == "results/gate_dose_0830"
