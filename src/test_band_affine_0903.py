"""band_affine_0903 の単体検査（走なし・合成データと committed 値の照合）。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from .common import ROOT
from . import band_affine_0903 as B


def _synth(p_rows: list[list[float]], zmax_rows: list[list[float]] | None = None) -> dict:
    n = len(p_rows)
    step = np.arange(1, n + 1, dtype=np.int64) * B.PERIOD
    d = dict(step=step, layer1_p_hat=np.asarray(p_rows, dtype=np.float32),
             unfit=np.full(n, 0.1))
    if zmax_rows is not None:
        d["layer1_zmax"] = np.asarray(zmax_rows, dtype=np.float32)
    return d


def test_three_classes_partition_and_thresholds():
    p = [[0.0, 1.0, 1 / 32, 31 / 32]]
    d = _synth(p, zmax_rows=[[-0.5, 2.0, -3.0, 1.0]])
    st = B.seed_stats(d, [1, 1], 1e-16, 0.05, shallow_thr=1.0)
    assert st["sub_frac"] == 0.25 and st["surf_frac"] == 0.25 and st["band_frac"] == 0.5
    assert st["sub_frac"] + st["band_frac"] + st["surf_frac"] == 1.0
    assert st["surf_among_alive"] == pytest.approx(1 / 3)
    assert st["near_off_frac"] == 0.25 and st["near_on_frac"] == 0.25
    # 浅い沈下は -1.0 < zmax <= 0 の 1 ユニットだけ（-3.0 は深い、正の 2 つは非沈下）
    assert st["shallow_sub_frac"] == 0.25
    assert st["non_affine_frac"] == pytest.approx(0.75)


def test_window_is_task_end_records_only():
    step = np.array([0, 5_000, 10_000, 15_000, 20_000], dtype=np.int64)
    idx = B._window_idx(step, [1, 2])
    assert list(step[idx]) == [10_000, 20_000]


def test_band_hist_counts_only_interior_bins():
    d = _synth([[0.0, 1.0, 1 / 32, 1 / 32, 16 / 32]])
    h = B.band_hist(d, [1, 1])
    assert h.sum() == 3 and h[0] == 2 and h[15] == 1


def test_spearman_matches_known_values():
    assert B._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert B._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # 同順位の平均順位
    assert B._spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-4)


def test_series_splits_task_boundary():
    step = np.array([9_000, 10_000, 11_000, 20_000], dtype=np.int64)
    d = dict(step=step, layer1_p_hat=np.array([[0.5], [0.0], [0.5], [0.0]], dtype=np.float32))
    rows = B.band_series(d)
    pre = {r["step"]: r["band_frac"] for r in rows if r["position"] == "pre_boundary"}
    post = {r["step"]: r["band_frac"] for r in rows if r["position"] == "post_boundary"}
    assert pre == {10_000: 0.0, 20_000: 0.0} and post == {11_000: 1.0}


@pytest.mark.parametrize("arm", ["R_1216", "G_b1_1216", "S_b0p3_1216"])
def test_matches_committed_dial_table(arm):
    """委託先 gate_dial_0902 の committed 値と窓・沈下率が bit 一致する。"""
    table = {}
    with open(Path(ROOT) / "results/gate_dial_0902/dial_table.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[r["arm"]] = r
    a = next(x for x in B._arm_table() if x["arm"] == arm)
    vals = [B.seed_stats(B._load(a["logdir"] / f"{arm}_seed{s}.npz"), B.WINDOWS["5M"],
                         1e-16, 0.05, a["shallow_thr"]) for s in range(B.N_SEED)]
    assert float(np.median([v["log10_u"] for v in vals])) == pytest.approx(
        float(table[arm]["median_log10_U_5m"]), abs=1e-9)
    assert float(np.median([v["sub_frac"] for v in vals])) == pytest.approx(
        float(table[arm]["submerged_frac"]), abs=1e-9)
