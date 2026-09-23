"""Synthetic fixtures only; these tests never read production outcomes."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import analyze as analyzer

from analyze import (Series, _transition_rows, bootstrap_median_ci,
                     group_verdicts, low_response_onset, paired_conda,
                     pre_onset_window, preflight, run_analysis,
                     summarize_seed_tables)
from plots import make_plots


def _task_table(seed=100, arm="k1", *, tasks=30, low_at=None):
    task = np.arange(1, tasks + 1)
    values = np.full(tasks, 10.)
    derivative = np.ones(tasks)
    if low_at is not None:
        derivative[task >= low_at] = 0.
    return pd.DataFrame({"environment": "conda", "arm": arm, "seed": seed,
                         "source_class": "prospective", "metric": "effective",
                         "task": task, "V": values, "V_next": values,
                         "Q": np.full(tasks, 2.), "X": np.full(tasks, -1.),
                         "covariance_term": np.zeros(tasks),
                         "derivative_absmean": derivative,
                         "activeunit_frac": np.ones(tasks)})


def test_low_response_first_three_and_conditional_window():
    table = _task_table(tasks=50, low_at=32)
    assert low_response_onset(table) == 32
    report = pre_onset_window(table, 32)
    assert report["active_window_status"] == "AVAILABLE"
    assert (report["active_window_first_task"], report["active_window_last_task"]) == (12, 31)
    assert report["active_window_balance"] == pytest.approx(1.)
    assert pre_onset_window(table, 15)["active_window_status"] == "INSUFFICIENT_ACTIVE_HISTORY"
    assert low_response_onset(_task_table(tasks=50, low_at=None)) is None


def test_seed_summary_p4_and_closure_uninformative():
    effective = _task_table()
    raw = effective.copy()
    raw["metric"] = "raw"
    # Raw V doubles smoothly while effective V stays flat. P4 uses the
    # preregistered slope gap and effective P2, never a raw P2 verdict.
    raw["V"] = 10. * raw.task
    raw["V_next"] = 10. * (raw.task + 1)
    raw["Q"] = 10.
    raw["X"] = 0.
    units = pd.DataFrame({"environment": ["conda"], "arm": ["k1"],
                          "seed": [100], "metric": ["effective"],
                          "mean_sigma": [1.], "mean_D": [1.]})
    summary, closure = summarize_seed_tables(pd.concat([effective, raw]), units)
    eff = summary[summary.metric == "effective"].iloc[0]
    assert eff.P1 == "PASS" and eff.P2 == "PASS"
    assert eff.P3 == "UNINFORMATIVE"
    assert eff.P4 == "SEPARATED"
    assert summary[summary.metric == "raw"].iloc[0].P1 == "NOT_APPLICABLE"
    assert closure.task.min() == 16 and closure.task.max() == 30
    assert np.allclose(closure.model_V, closure.observed_V)


def test_seed_bootstrap_and_group_requires_four_of_five():
    lo, hi = bootstrap_median_ci([1., 2., 3., 4., 5.])
    assert (lo, hi) == bootstrap_median_ci([1., 2., 3., 4., 5.])
    assert lo <= 3 <= hi
    rows = []
    for seed in range(100, 105):
        rows.append({"environment": "conda", "arm": "k1", "metric": "effective",
                     "seed": seed, "source_class": "prospective",
                     "P1": "PASS" if seed < 104 else "FAIL", "P2": "FAIL",
                     "P3": "UNINFORMATIVE", "P4": "NOT_ESTABLISHED",
                     "late_nearbalance": 1., "late_loglog_slope": .5,
                     "heldout_mape": .02, "raw_minus_effective_slope": 0.})
    group = group_verdicts(pd.DataFrame(rows), pd.DataFrame())
    assert group.set_index("question").loc["P1", "verdict"] == "PASS"
    assert group.set_index("question").loc["P2", "verdict"] == "FAIL"
    assert group.set_index("question").loc["P3", "verdict"] == "UNINFORMATIVE"
    one_missing = group_verdicts(pd.DataFrame(rows[:-1]), pd.DataFrame())
    assert one_missing.set_index("question").loc["P1", "verdict"] == "INCOMPLETE"


def test_paired_seed_ratios_and_direction():
    rows = []
    for seed in range(100, 105):
        for arm, d in (("k1", 2.), ("k7", 3.), ("k1_sgd", 1.), ("k1_eps1e3", 1.)):
            rows.append({"environment": "conda", "arm": arm, "metric": "effective",
                         "seed": seed, "late_mean_sqrtQ": d, "late_mean_V": 10.,
                         "late_mean_sigma": math.sqrt(10.)})
    pairs = paired_conda(pd.DataFrame(rows))
    assert pairs.P5_pass.all() and pairs.P6_pass.all()
    assert pairs.P5_D_k7_over_k1.median() == pytest.approx(1.5)
    assert pairs.P6_D_A_over_S.median() == pytest.approx(2.)


def _write_conda_fixture(raw, tasks=100):
    root = raw / "conda"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"tasks": tasks, "seeds": [100],
        "primary_period": 10000, "frequency_period": 1000}))
    seed_dir = root / "k1" / "seed100"
    seed_dir.mkdir(parents=True)
    (seed_dir / "status.json").write_text(json.dumps({"status": "COMPLETED", "last_saved_task": tasks}))
    for t in range(tasks + 1):
        w = np.zeros((2, 20), dtype=np.float32)
        w[:, -5:] = (1 + .001 * t)
        np.savez(seed_dir / f"t{t:03d}.npz", W=w, b=np.zeros(2), mu=np.ones(20),
                 flip_state=np.zeros(15), mse=1., task_start_mse=2., w_sigma2=2.,
                 derivative_exactzero_frac=0., derivative_absmean=1., activeunit_frac=1.)
    return Series("conda", "k1", 100, tasks, "conda", seed_dir)


def test_conda_loader_and_incomplete_gate(tmp_path):
    raw = tmp_path / "raw"
    series = _write_conda_fixture(raw)
    assert preflight(series)["status"] == "COMPLETE"
    rows, units, _ = _transition_rows(series, cifar_data=tmp_path)
    assert len(rows) == 2 * 100
    assert len(units) == 2 * 2
    assert {row["metric"] for row in rows} == {"raw", "effective"}
    assert all(math.isfinite(row["mean_shift_sq"]) for row in rows if row["metric"] == "effective")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="incomplete"):
        run_analysis(raw, out, make_plots=False)
    assert (out / "completion.csv").is_file()
    result = run_analysis(raw, out, allow_incomplete=True, make_plots=False)
    assert result["completion"].status.eq("COMPLETE").sum() == 1
    assert "INCOMPLETE" in (out / "summary.md").read_text()
    assert (out / "verdict.csv").is_file()
    assert (out / "source_windows.csv").is_file()
    paths = make_plots(result["transitions"], result["seed_summary"],
                       result["closure"], result["paired"], out)
    assert paths and all(path.is_file() for path in paths)


def test_allow_incomplete_reports_missing_without_claims(tmp_path):
    result = run_analysis(tmp_path / "absent_raw", tmp_path / "out",
                          allow_incomplete=True, make_plots=False)
    assert result["completion"].status.eq("MISSING").all()
    assert result["verdict"].empty
    assert "INCOMPLETE" in (tmp_path / "out" / "summary.md").read_text()


def test_pm_loader_keeps_reference_and_actual_covariance_separate(tmp_path, monkeypatch):
    # Two-input synthetic bank keeps this orchestration test independent of
    # 784-dimensional MNIST files and production data.
    monkeypatch.setattr(analyzer, "_checked_covariance", lambda cov, inputs: cov)
    sd = tmp_path / "seed_100"
    sd.mkdir()
    np.savez(sd / "reference_bank.npz", covariance=np.eye(2), mean=np.zeros(2))
    diagnostics = []
    for t in range(21):
        np.savez(sd / f"state_{t:03d}.npz", W1=np.array([[1. + .01*t, 0.], [0., 2.]]),
                 b1=np.zeros(2))
        if t:
            np.savez(sd / f"task_{t:03d}_input.npz", covariance=np.diag([1. + .1*t, 1.]),
                     mean=np.array([.01*t, 0.]))
            diagnostics.append({"task": t, "online_acc": .2, "task_start_acc": .2,
                                "train_acc": .3, "active_unit_frac_l1": 1.,
                                "active_unit_frac_l2": 1., "derivative_abs_mean_l1": 1.,
                                "derivative_abs_mean_l2": 1.})
    (sd / "per_task.json").write_text(json.dumps(diagnostics))
    series = Series("mnist_pm", "LR", 100, 20, "pm", sd, optimizer="adam")
    rows, units, _ = _transition_rows(series, cifar_data=tmp_path)
    frame = pd.DataFrame(rows)
    assert set(frame.metric) == {"reference", "actual", "raw"}
    actual = frame[frame.metric == "actual"].sort_values("task")
    reference = frame[frame.metric == "reference"].sort_values("task")
    assert reference.covariance_term.eq(0).all()
    assert actual.iloc[0].covariance_term == pytest.approx(0.)
    assert (actual.iloc[1:].covariance_term > 0).all()
    np.testing.assert_allclose(actual.identity_residual, 0., atol=1e-12)
    summaries, _ = summarize_seed_tables(frame, pd.DataFrame(units))
    assert summaries[summaries.metric == "actual"].iloc[0].P1 == "NOT_APPLICABLE"
