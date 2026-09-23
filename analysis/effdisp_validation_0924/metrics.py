"""Float64 bookkeeping for effective displacement across task transitions.

Rows of ``W`` are units. A covariance may be represented by ``factor``
(``Sigma = factor @ factor.T``), diagonal ``variance``, dense ``covariance``,
or neither (identity).
The previous covariance defines V, Q, and X. A next covariance, if supplied,
defines V_next and the explicit covariance-change term.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return result


def _vector(value: np.ndarray | None, size: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _project(weights: np.ndarray, factor: np.ndarray | None,
             variance: np.ndarray | None) -> np.ndarray:
    inputs = weights.shape[1]
    if factor is not None and variance is not None:
        raise ValueError("provide factor or variance, not both")
    if factor is not None:
        f = _matrix(factor, "factor")
        if f.shape[0] != inputs:
            raise ValueError("factor input dimension does not match W")
        return weights @ f
    if variance is not None:
        v = np.asarray(variance, dtype=np.float64)
        if v.shape != (inputs,) or not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("variance must be a finite nonnegative input vector")
        return weights * np.sqrt(v)
    return weights


def validate_covariance_matrix(covariance: np.ndarray, inputs: int) -> np.ndarray:
    """Return symmetric float64 covariance after shape, finite, and PSD checks.

    A Cholesky check with tiny diagonal jitter accepts numerical negative
    eigenvalues at roughly 1e-10 of the matrix scale. Call once per saved
    covariance, then pass ``validate_covariance=False`` on repeated metrics.
    """
    return _covariance(covariance, inputs, check_psd=True)


def _covariance(covariance: np.ndarray, inputs: int, *, check_psd: bool) -> np.ndarray:
    c = _matrix(covariance, "covariance")
    if c.shape != (inputs, inputs):
        raise ValueError("covariance must have shape (inputs, inputs)")
    scale = max(1., float(np.linalg.norm(c, ord=np.inf)))
    tolerance = 1e-10 * scale
    if np.max(np.abs(c - c.T)) > tolerance:
        raise ValueError("covariance must be symmetric")
    c = (c + c.T) * 0.5
    if check_psd:
        try:
            np.linalg.cholesky(c + tolerance * np.eye(inputs))
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive semidefinite") from exc
    return c


def _terms(w: np.ndarray, d: np.ndarray, *, factor: np.ndarray | None,
           variance: np.ndarray | None, covariance: np.ndarray | None,
           validate_covariance: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sum(z is not None for z in (factor, variance, covariance)) > 1:
        raise ValueError("provide at most one of factor, variance, covariance")
    if covariance is not None:
        c = _covariance(covariance, w.shape[1], check_psd=validate_covariance)
        wc = w @ c
        dc = d @ c
        v = np.einsum("ij,ij->i", wc, w)
        q = np.einsum("ij,ij->i", dc, d)
        x = np.einsum("ij,ij->i", wc, d)
        wn = w + d
        vn = np.einsum("ij,ij->i", wn @ c, wn)
        return v, q, x, vn
    pw = _project(w, factor, variance)
    pd_ = _project(d, factor, variance)
    v = np.einsum("ij,ij->i", pw, pw)
    q = np.einsum("ij,ij->i", pd_, pd_)
    x = np.einsum("ij,ij->i", pw, pd_)
    vn = np.einsum("ij,ij->i", pw + pd_, pw + pd_)
    return v, q, x, vn


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def transition_metrics(
    w_prev: np.ndarray,
    w_next: np.ndarray,
    *,
    factor: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    factor_next: np.ndarray | None = None,
    variance_next: np.ndarray | None = None,
    covariance_next: np.ndarray | None = None,
    validate_covariance: bool = True,
    mean_prev: np.ndarray | None = None,
    mean_next: np.ndarray | None = None,
    bias_prev: np.ndarray | None = None,
    bias_next: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return per-unit rows and a summed transition row.

    ``c = X/sqrt(V Q)``, ``rho = sqrt(Q/V)``, and ``balance = -2X/Q``.
    Undefined ratios are NaN, including zero-update balance. ``residual`` is
    the literal ``V_next - V - Q - 2X``; with moving covariance it equals
    ``covariance_term`` up to roundoff. ``identity_residual`` subtracts that
    term as well. No ratio is clipped.

    The optional ``mean_shift_sq`` is the squared change in mean preactivation
    of each unit, including bias. If either mean is supplied, the other
    defaults to the same input mean; similarly, a missing bias side defaults
    to zero. This channel is separate from centered-covariance V/Q/X.
    """
    w = _matrix(w_prev, "w_prev")
    wn = _matrix(w_next, "w_next")
    if w.shape != wn.shape:
        raise ValueError("w_prev and w_next must have the same shape")
    d = wn - w
    v, q, x, vn_prev = _terms(w, d, factor=factor, variance=variance,
                              covariance=covariance, validate_covariance=validate_covariance)
    # A next covariance is optional; absent means the previous covariance.
    if factor_next is None and variance_next is None and covariance_next is None:
        vn = vn_prev
    else:
        if sum(z is not None for z in (factor_next, variance_next, covariance_next)) > 1:
            raise ValueError("provide at most one next covariance representation")
        vn = _terms(wn, np.zeros_like(wn), factor=factor_next, variance=variance_next,
                    covariance=covariance_next, validate_covariance=validate_covariance)[0]
    cov_term = vn - vn_prev
    delta = vn - v
    residual = delta - q - 2 * x
    rows = pd.DataFrame({
        "unit": np.arange(w.shape[0]), "V": v, "Q": q, "X": x,
        "V_next": vn, "delta_V": delta, "D_norm": np.sqrt(q),
        "c": _safe_ratio(x, np.sqrt(v * q)),
        "rho": _safe_ratio(np.sqrt(q), np.sqrt(v)),
        "balance": _safe_ratio(-2 * x, q),
        "covariance_term": cov_term, "residual": residual,
        "identity_residual": residual - cov_term,
    })
    if any(z is not None for z in (mean_prev, mean_next, bias_prev, bias_next)):
        mu0 = mean_prev if mean_prev is not None else mean_next
        mu1 = mean_next if mean_next is not None else mean_prev
        m0 = _vector(mu0, w.shape[1], "mean_prev")
        m1 = _vector(mu1, w.shape[1], "mean_next")
        b0 = _vector(bias_prev, w.shape[0], "bias_prev")
        b1 = _vector(bias_next, w.shape[0], "bias_next")
        shift = wn @ m1 + b1 - (w @ m0 + b0)
        rows["mean_shift_sq"] = shift * shift
    total = {k: float(np.sum(rows[k].to_numpy(dtype=np.float64), dtype=np.float64)) for k in
             ("V", "Q", "X", "V_next", "delta_V", "covariance_term",
              "residual", "identity_residual")}
    total["D_norm"] = math.sqrt(total["Q"])
    total["c"] = (total["X"] / math.sqrt(total["V"] * total["Q"])
                  if total["V"] * total["Q"] > 0 else math.nan)
    total["rho"] = (math.sqrt(total["Q"] / total["V"])
                    if total["V"] > 0 else math.nan)
    total["balance"] = (-2 * total["X"] / total["Q"]
                        if total["Q"] > 0 else math.nan)
    if "mean_shift_sq" in rows:
        total["mean_shift_sq"] = float(np.sum(rows["mean_shift_sq"].to_numpy(dtype=np.float64), dtype=np.float64))
    return rows, total


def summarize_task_table(table: pd.DataFrame, *, fixed_covariance: bool = True) -> dict[str, float | int | bool | str]:
    """Calibrate a constant-coefficient closure and report held-out trajectory.

    Requires one row per consecutive endpoint task ``t`` (transition
    ``W_(t-1) -> W_t``) with
    ``V``, ``Q``, ``X``, and ``V_next``. At least 20 transitions are required.
    Calibration uses tasks 6 through floor(T/2), where T is the last task;
    held-out prediction starts at observed V at task floor(T/2) and uses
    no later Q/X. MAPE is evaluated at each held-out endpoint. Diagnostics
    on the final max(10, ceil(20% of transitions)) transitions are independent
    of closure fitting. P1–P3 verdicts apply only to a fixed covariance.
    For a moving covariance, pass ``fixed_covariance=False`` to retain
    diagnostics while marking the verdicts inapplicable. A nonzero supplied
    ``covariance_term`` cannot be marked fixed.
    """
    needed = {"task", "V", "Q", "X", "V_next"}
    if not needed.issubset(table.columns):
        raise ValueError(f"missing columns: {sorted(needed - set(table.columns))}")
    if fixed_covariance and "covariance_term" in table:
        terms = pd.to_numeric(table["covariance_term"], errors="raise").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(terms)) or np.any(np.abs(terms) > 1e-10):
            raise ValueError("nonzero covariance_term requires fixed_covariance=False")
    a = table.sort_values("task").reset_index(drop=True).copy()
    if len(a) < 20:
        raise ValueError("at least 20 transitions are required")
    task = a["task"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(task)) or not np.all(task == np.floor(task)) or not np.all(np.diff(task) == 1):
        raise ValueError("task must contain consecutive integer transitions")
    for name in ("V", "Q", "X", "V_next"):
        a[name] = pd.to_numeric(a[name], errors="raise").astype(np.float64)
        if not np.all(np.isfinite(a[name])):
            raise ValueError(f"{name} must be finite")
    v = a["V"].to_numpy()
    vn = a["V_next"].to_numpy()
    q = a["Q"].to_numpy()
    x = a["X"].to_numpy()
    if np.any(v < 0) or np.any(vn < 0) or np.any(q < 0):
        raise ValueError("V, V_next, and Q must be nonnegative")
    if not np.allclose(v[1:], vn[:-1], rtol=1e-10, atol=1e-12):
        raise ValueError("adjacent V and V_next values must agree")
    split = int(task[-1] // 2)
    cal = (task >= 6) & (task <= split)
    held = task > split
    if not np.any(cal) or not np.any(held):
        raise ValueError("calibration and held-out windows must both be nonempty")
    qbar = float(np.mean(q[cal], dtype=np.float64))
    sum_cal_v = float(np.sum(v[cal], dtype=np.float64))
    gamma = -float(np.sum(x[cal], dtype=np.float64)) / sum_cal_v if sum_cal_v > 0 else math.nan
    initial_v = float(v[held][0])
    actual = vn[held]
    predictions = np.empty(len(actual), dtype=np.float64)
    current = initial_v
    for i in range(len(predictions)):
        current = (1 - 2 * gamma) * current + qbar
        predictions[i] = current
    baseline = np.full(len(actual), initial_v, dtype=np.float64)
    def mape(pred: np.ndarray) -> float:
        if np.any(actual <= 0):
            return math.nan
        return float(np.mean(np.abs(pred - actual) / actual, dtype=np.float64))
    late_n = min(len(a), max(10, math.ceil(0.2 * len(a))))
    late = slice(-late_n, None)
    endpoint_v = np.r_[v[late][0], vn[late]]
    endpoint_t = np.r_[task[late][0] - 1, task[late]]
    log_slope = (float(np.polyfit(np.log(endpoint_t), np.log(endpoint_v), 1)[0])
                 if np.all(endpoint_t > 0) and np.all(endpoint_v > 0) else math.nan)
    late_q = float(np.sum(q[late], dtype=np.float64))
    late_invalid_v = bool(np.any(v[late] <= 0) or np.any(vn[late] <= 0))
    late_zero_q_count = int(np.count_nonzero(q[late] == 0))
    late_mean_v = float(np.mean(v[late], dtype=np.float64))
    late_activity = late_q / late_mean_v if late_mean_v > 0 else math.nan
    late_c = _safe_ratio(x[late], np.sqrt(v[late] * q[late]))
    late_negative_c_fraction = float(np.mean(late_c < 0))
    late_v = v[late]
    halves = late_n // 2
    late_halves_ratio = (float(np.mean(late_v[halves:]) / np.mean(late_v[:halves]))
                         if np.mean(late_v[:halves]) > 0 else math.nan)
    late_balance = -2 * float(np.sum(x[late], dtype=np.float64)) / late_q if late_q > 0 else math.nan
    p1 = (not late_invalid_v
          and np.isfinite(late_balance) and 0.9 <= late_balance <= 1.1
          and np.isfinite(late_activity) and late_activity >= 0.1
          and late_negative_c_fraction >= 0.8)
    p2 = p1 and np.isfinite(log_slope) and abs(log_slope) <= 0.2 and 0.9 <= late_halves_ratio <= 1.1
    model_mape = mape(predictions)
    baseline_mape = mape(baseline)
    stable_gamma = bool(np.isfinite(gamma) and 0 < gamma < 1)
    positive_predictions = bool(np.all(np.isfinite(predictions) & (predictions > 0)))
    if np.isfinite(model_mape) and np.isfinite(baseline_mape) and model_mape < 0.05 and baseline_mape < 0.05:
        p3 = "UNINFORMATIVE"
    elif (stable_gamma and positive_predictions and np.isfinite(model_mape)
          and model_mape <= 0.2 and np.isfinite(baseline_mape)
          and model_mape <= 0.8 * baseline_mape):
        p3 = "PASS"
    else:
        p3 = "FAIL"
    return {
        "n_transitions": len(a), "calibration_first_task": int(task[cal][0]),
        "calibration_last_task": int(task[cal][-1]),
        "heldout_first_task": int(task[held][0]),
        "heldout_last_task": int(task[held][-1]),
        "qbar": qbar, "gamma": gamma, "split_V": initial_v,
        "heldout_mape": model_mape, "constant_split_mape": baseline_mape,
        "negative_prediction": bool(np.any(predictions < 0)),
        "unstable_gamma": not stable_gamma,
        "positive_predictions": positive_predictions,
        "P3": p3 if fixed_covariance else "NOT_APPLICABLE",
        "late_n_transitions": late_n,
        "late_endpoint_relative_change": (float((endpoint_v[-1] - endpoint_v[0]) / endpoint_v[0])
                                          if endpoint_v[0] > 0 else math.nan),
        "late_median_relative_increment": (float(np.median((vn[late] - v[late]) / v[late]))
                                           if np.all(v[late] > 0) else math.nan),
        "late_loglog_slope": log_slope,
        "late_halves_ratio": late_halves_ratio,
        "late_nearbalance": late_balance,
        "late_activity": late_activity,
        "late_invalid_V": late_invalid_v,
        "late_zero_Q_count": late_zero_q_count,
        "late_negative_c_fraction": late_negative_c_fraction,
        "P1": ("PASS" if p1 else "FAIL") if fixed_covariance else "NOT_APPLICABLE",
        "P2": ("PASS" if p2 else "FAIL") if fixed_covariance else "NOT_APPLICABLE",
        "stopped": bool(late_q == 0 or (np.isfinite(late_activity) and late_activity < 0.1)),
    }


def summarize_task_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply ``summarize_task_table`` independently to named task tables."""
    return pd.DataFrame([{"series": name, **summarize_task_table(table)}
                         for name, table in tables.items()])
