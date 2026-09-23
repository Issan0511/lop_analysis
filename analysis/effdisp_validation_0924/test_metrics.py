"""Numerical invariants and fit-window checks for metrics.py."""

import math

import numpy as np
import pandas as pd
import pytest

from metrics import summarize_task_table, transition_metrics, validate_covariance_matrix


def test_raw_and_factor_identity_per_unit_and_sum():
    w = np.array([[1., 2.], [-3., 4.]])
    wn = np.array([[2., 0.], [-1., 3.]])
    for kwargs in ({}, {"factor": np.array([[2., 0.], [1., 3.]])},
                   {"variance": np.array([4., 9.])}):
        unit, total = transition_metrics(w, wn, **kwargs)
        np.testing.assert_allclose(unit.delta_V, unit.Q + 2 * unit.X, atol=1e-12)
        np.testing.assert_allclose(unit.identity_residual, 0., atol=1e-12)
        assert abs(total["identity_residual"]) < 1e-12
        assert total["V"] == pytest.approx(unit.V.sum())
        assert total["Q"] == pytest.approx(unit.Q.sum())
        assert total["X"] == pytest.approx(unit.X.sum())
    _, raw = transition_metrics(w, wn)
    assert raw["V"] == pytest.approx(np.sum(w * w))
    assert raw["Q"] == pytest.approx(np.sum((wn-w) ** 2))
    assert raw["X"] == pytest.approx(np.sum(w * (wn-w)))


def test_moving_covariance_exact_term_and_mean_channel():
    w = np.array([[1., 2.], [2., -1.]])
    wn = np.array([[2., 4.], [1., 3.]])
    f0 = np.diag([2., 1.])
    f1 = np.array([[1., 1.], [0., 2.]])
    unit, total = transition_metrics(
        w, wn, factor=f0, factor_next=f1,
        mean_prev=np.array([1., 0.]), mean_next=np.array([2., 1.]),
        bias_prev=np.array([0., -1.]), bias_next=np.array([1., 2.]))
    sigma_diff = f1 @ f1.T - f0 @ f0.T
    expected = np.einsum("ui,ij,uj->u", wn, sigma_diff, wn)
    np.testing.assert_allclose(unit.covariance_term, expected, atol=1e-12)
    np.testing.assert_allclose(unit.residual, expected, atol=1e-12)
    np.testing.assert_allclose(unit.identity_residual, 0., atol=1e-12)
    shift = wn @ np.array([2., 1.]) + np.array([1., 2.]) - (w @ np.array([1., 0.]) + np.array([0., -1.]))
    np.testing.assert_allclose(unit.mean_shift_sq, shift ** 2)
    assert total["mean_shift_sq"] == pytest.approx(np.sum(shift ** 2))


def test_dense_covariance_matches_factor_fixed_and_moving():
    w = np.array([[1., 2., -1.], [3., 0., 2.]])
    wn = np.array([[2., 1., -2.], [1., 4., 2.]])
    f0 = np.array([[2., 0.], [1., 3.], [0., 1.]])
    f1 = np.array([[1., 2.], [-1., 0.], [2., 1.]])
    c0 = f0 @ f0.T
    c1 = f1 @ f1.T
    fact_rows, fact = transition_metrics(w, wn, factor=f0, factor_next=f1)
    dense_rows, dense = transition_metrics(w, wn, covariance=c0, covariance_next=c1)
    for name in ("V", "Q", "X", "V_next", "delta_V", "covariance_term", "residual", "identity_residual"):
        np.testing.assert_allclose(dense_rows[name], fact_rows[name], atol=1e-12)
        assert dense[name] == pytest.approx(fact[name], abs=1e-12)
    fixed_rows, _ = transition_metrics(w, wn, covariance=c0)
    np.testing.assert_allclose(fixed_rows.identity_residual, 0., atol=1e-12)
    # Prevalidated matrices can skip repeated cubic PSD checks.
    checked = validate_covariance_matrix(c0, 3)
    fast_rows, _ = transition_metrics(w, wn, covariance=checked, validate_covariance=False)
    np.testing.assert_allclose(fast_rows.V, fixed_rows.V)


def test_dense_covariance_rejects_invalid_but_accepts_tiny_roundoff():
    w = np.ones((1, 2))
    with pytest.raises(ValueError, match="at most one"):
        transition_metrics(w, w, factor=np.eye(2), covariance=np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        transition_metrics(w, w, covariance=np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        transition_metrics(w, w, covariance=np.array([[1., .1], [0., 1.]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        transition_metrics(w, w, covariance=np.diag([1., -.01]))
    validate_covariance_matrix(np.diag([1., -1e-12]), 2)


def test_zero_update_and_unequal_q_aggregate():
    w = np.array([[1.], [1.]])
    same, total = transition_metrics(w, w)
    np.testing.assert_array_equal(same.D_norm, 0.)
    assert same.balance.isna().all() and math.isnan(total["balance"])
    wn = np.array([[0.], [3.]])  # Q = 1, 4; X = -1, 2.
    rows, total = transition_metrics(w, wn)
    assert total["balance"] == pytest.approx(-2. / 5.)
    assert total["balance"] != pytest.approx(rows.balance.mean())
    assert total["c"] == pytest.approx(total["X"] / math.sqrt(total["V"] * total["Q"]))


def _table(n=30):
    # Exactly stationary observations; Q/X can be independently altered to
    # test fitting because the summary deliberately does not impose identity.
    return pd.DataFrame({"task": np.arange(1, n + 1),
                         "V": np.full(n, 10.), "V_next": np.full(n, 10.),
                         "Q": np.full(n, 2.), "X": np.full(n, -1.)})


def test_holdout_fit_uses_only_tasks_6_through_split():
    a = _table()
    fitted = summarize_task_table(a)
    assert fitted["calibration_first_task"] == 6
    assert fitted["calibration_last_task"] == 15
    assert fitted["heldout_first_task"] == 16
    assert fitted["qbar"] == pytest.approx(2.)
    assert fitted["gamma"] == pytest.approx(0.1)
    assert fitted["heldout_mape"] == pytest.approx(0.)
    assert fitted["P1"] == "PASS" and fitted["P2"] == "PASS"
    assert fitted["P3"] == "UNINFORMATIVE"
    assert fitted["late_halves_ratio"] == pytest.approx(1.)
    assert fitted["late_activity"] == pytest.approx(2.)
    changed = a.copy()
    changed.loc[changed.task > 15, ["Q", "X"]] = [1000., 500.]
    b = summarize_task_table(changed)
    assert b["qbar"] == fitted["qbar"]
    assert b["gamma"] == fitted["gamma"]
    assert b["heldout_mape"] == fitted["heldout_mape"]
    assert b["late_nearbalance"] != fitted["late_nearbalance"]


def test_minimum_length_and_negative_prediction_exposed():
    with pytest.raises(ValueError, match="at least 20"):
        summarize_task_table(_table(19))
    a = _table(20)
    a.loc[(a.task >= 6) & (a.task <= 10), "X"] = -20.
    result = summarize_task_table(a)
    assert result["unstable_gamma"]
    assert result["negative_prediction"]
    assert result["late_n_transitions"] == 10


def test_degenerate_late_activity_is_explicit():
    a = _table(20)
    a["Q"] = 0.
    a["X"] = 0.
    result = summarize_task_table(a)
    assert result["late_zero_Q_count"] == 10
    assert result["stopped"] and result["P1"] == "FAIL"
    assert math.isnan(result["late_nearbalance"])
    b = a.copy()
    b["V"] = 0.
    b["V_next"] = 0.
    invalid = summarize_task_table(b)
    assert invalid["late_invalid_V"]
    assert invalid["P2"] == "FAIL"


def test_moving_covariance_summary_diagnostics_have_no_fixed_metric_verdict():
    a = _table(20)
    a["covariance_term"] = 1.
    with pytest.raises(ValueError, match="fixed_covariance=False"):
        summarize_task_table(a)
    result = summarize_task_table(a, fixed_covariance=False)
    assert result["P1"] == result["P2"] == result["P3"] == "NOT_APPLICABLE"
