"""Small numerical and paired-decision checks; no CIFAR data or GPU needed."""

from __future__ import annotations

import math
import hashlib
import json

import numpy as np
import pandas as pd

from analysis.effdisp_imagegeom_0924 import analyze as A


def test_projection_scales_with_input_and_records_mean_channel():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(21, 4)) + np.array([.2, -.3, .4, 1.0])
    centered, mean = x - x.mean(0), x.mean(0)
    w0, w1 = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    b0, b1 = rng.normal(size=3), rng.normal(size=3)
    original, p1 = A.projection_terms(centered, mean, w0, b0, w1, b1)
    scaled, _ = A.projection_terms(2*centered, 2*mean, w0, b0, w1, b1)
    equivalent, _ = A.projection_terms(2*centered, 2*mean, w0/2, b0, w1/2, b1)
    for key in ("V", "Q", "X", "V_next"):
        np.testing.assert_allclose(scaled[key], 4*original[key], rtol=1e-13)
        np.testing.assert_allclose(equivalent[key], original[key], rtol=1e-13)
    np.testing.assert_allclose(equivalent["mean_shift_sq"], original["mean_shift_sq"])
    np.testing.assert_allclose(original["mean_shift_sq"],
                               np.sum(((w1-w0)@mean+b1-b0)**2))
    np.testing.assert_allclose(p1, centered@w1.T/math.sqrt(len(x)), rtol=1e-13, atol=1e-14)
    assert abs(original["identity_residual"]) < 1e-12


def test_covariance_rank_participation_and_top10_projection():
    rng = np.random.default_rng(38)
    raw = rng.normal(size=(A.N_IMAGES, 12))
    orthogonal, _ = np.linalg.qr(raw - raw.mean(0))
    singular = np.arange(12, 0, -1, dtype=float)
    x = orthogonal * singular * math.sqrt(A.N_IMAGES)
    stats, values, top = A.covariance_spectrum(x)
    assert stats["numeric_rank"] == 12
    expected_eigen = singular**2
    np.testing.assert_allclose(values, singular, rtol=1e-12)
    np.testing.assert_allclose(stats["cov_participation_ratio"],
                               expected_eigen.sum()**2 / np.sum(expected_eigen**2))
    np.testing.assert_allclose(stats["top10_cov_trace_fraction"],
                               expected_eigen[:10].sum()/expected_eigen.sum())
    duplicated = np.tile(x, (1, 4))
    dup_stats, dup_values, dup_top = A.covariance_spectrum(
        duplicated, reference={"singular_values": values, "top_left": top}, scale=2)
    np.testing.assert_allclose(dup_values, 2*values)
    np.testing.assert_allclose(dup_stats["cov_participation_ratio"],
                               stats["cov_participation_ratio"])
    assert dup_stats["numeric_rank"] == stats["numeric_rank"]
    np.testing.assert_allclose(np.abs(dup_top.T@top), np.eye(10), atol=1e-12)
    mean = np.zeros(12)
    w0 = np.zeros((1, 12))
    w11 = w0.copy()
    w11[0, 10] = 1
    outside, _ = A.projection_terms(x, mean, w0, np.zeros(1), w11, np.zeros(1), top)
    assert outside["Q"] > 0
    assert outside["top10_Q_fraction"] < 1e-24
    w1 = w0.copy()
    w1[0, 0] = 1
    inside, _ = A.projection_terms(x, mean, w0, np.zeros(1), w1, np.zeros(1), top)
    np.testing.assert_allclose(inside["top10_Q_fraction"], 1, atol=1e-12)


def test_saved_geometry_transform_audit():
    rgb = np.zeros((A.N_IMAGES, 3072), dtype=np.float32)
    rgb[:, :1024] = .2
    rgb[:, 1024:2048] = .5
    rgb[:, 2048:] = .8
    z = rgb.reshape(A.N_IMAGES, 3, 32, 32)
    dup = np.repeat(np.repeat(z, 2, axis=2), 2, axis=3).reshape(A.N_IMAGES, -1)
    gray = (z[:, 0] * np.float32(.299) + z[:, 1] * np.float32(.587)
            + z[:, 2] * np.float32(.114)).reshape(A.N_IMAGES, -1)
    avg = z.reshape(A.N_IMAGES, 3, 16, 2, 16, 2).mean(axis=(3, 5)).reshape(A.N_IMAGES, -1)
    A._verify_duplication(rgb, dup, dup/2)
    A._verify_geometry(rgb, gray, avg)
    gray[0, 0] += .01
    with np.testing.assert_raises(ValueError):
        A._verify_geometry(rgb, gray, avg)


def _synthetic_windows() -> pd.DataFrame:
    rows = []
    for seed in A.SEEDS:
        for geometry in A.GEOMETRIES:
            for optimizer in A.OPTIMIZERS:
                for window in ("early", "late"):
                    d_eff, r, c = 5.0, 10.0, -.20
                    if geometry == "dup64_half" and optimizer == "adam" and window == "early":
                        d_eff = 6.5
                    if geometry == "dup64_half" and optimizer == "sgd" and window == "late":
                        d_eff, r = 5.25, 10.5
                    if geometry == "dup64" and window == "early":
                        # Adam and SGD each pass in four seeds, but only three
                        # seeds pass in both. E3 checks each optimizer separately.
                        d_eff = 6.5 if (optimizer == "adam" and seed != 304) or (optimizer == "sgd" and seed != 300) else 5
                    if geometry != "rgb32" and window == "late":
                        c = -.19
                    rows.append({"geometry": geometry, "optimizer": optimizer,
                                 "seed": seed, "window": window, "R": r,
                                 "d_eff": d_eff, "c": c, "rho": .5,
                                 "train_acc": .5, "top10_Q_fraction": .4,
                                 "mean_shift_sq": 1., "raw_W_sq": 1., "raw_D_sq": 1.})
    return pd.DataFrame(rows)


def test_paired_windows_and_registered_e_verdicts():
    windows = _synthetic_windows()
    contrasts = A.paired_contrasts(windows)
    row = contrasts[(contrasts.contrast == "geometry_over_rgb32")
                    & (contrasts.geometry == "dup64_half")
                    & (contrasts.optimizer == "adam")
                    & (contrasts.window == "early")
                    & (contrasts.metric == "d_eff")]
    assert len(row) == 5
    np.testing.assert_allclose(row.ratio, 1.3)
    grouped = A.paired_groups(contrasts)
    selected = grouped[(grouped.contrast == "geometry_over_rgb32")
                       & (grouped.geometry == "dup64_half")
                       & (grouped.optimizer == "adam")
                       & (grouped.window == "early")
                       & (grouped.metric == "d_eff")
                       & (grouped.statistic == "ratio")].iloc[0]
    assert selected.n_seed == 5 and selected.n_valid == 5
    np.testing.assert_allclose([selected.seed_median, selected.ci95_low, selected.ci95_high],
                               [1.3, 1.3, 1.3])
    verdicts = A.preregistered_e_verdicts(contrasts).set_index("question")
    for question in ("E1", "E2", "E3", "E5"):
        assert verdicts.loc[question, "verdict"] == "PASS"
    assert verdicts.loc["E3", "adam_n_pass"] == 4
    assert verdicts.loc["E3", "sgd_n_pass"] == 4


def test_window_median_is_within_seed_before_pairing():
    records = []
    for geometry in ("rgb32", "dup64"):
        for task in (*range(6, 16), *range(41, 51)):
            value = 2.0 if geometry == "dup64" else 1.0
            if task in (6, 41):
                value *= 100
            records.append({"geometry": geometry, "optimizer": "adam", "seed": 300,
                            "task": task, **{k: value for k in
                            ("R", "d_eff", "c", "rho", "train_acc", "top10_Q_fraction",
                             "mean_shift_sq", "raw_W_sq", "raw_D_sq")}})
    windows = A.seed_window_medians(pd.DataFrame(records))
    assert len(windows) == 4
    assert set(windows[windows.geometry == "rgb32"].d_eff) == {1.0}
    assert set(windows[windows.geometry == "dup64"].d_eff) == {2.0}


def test_low_response_requires_three_consecutive_tasks_and_is_censored():
    table = pd.DataFrame({"task": np.arange(1, 51), "V": np.ones(50),
                          "Q": np.zeros(50), "X": np.zeros(50),
                          "V_next": np.ones(50), "R": np.ones(50),
                          "d_eff": np.zeros(50), "c": np.full(50, np.nan),
                          "rho": np.zeros(50), "train_acc": np.full(50, .1),
                          "top10_Q_fraction": np.full(50, np.nan),
                          "mean_shift_sq": np.zeros(50), "raw_W_sq": np.ones(50),
                          "raw_W_next_sq": np.ones(50), "raw_D_sq": np.zeros(50),
                          "raw_X": np.zeros(50),
                          "mean_abs_dphi_l1": np.ones(50), "mean_abs_dphi_l2": np.ones(50),
                          "activeunit_frac_l1": np.ones(50), "activeunit_frac_l2": np.ones(50)})
    table.loc[table.task.isin((10, 11)), "activeunit_frac_l1"] = .01
    assert A.low_response_onset(table) is None
    table.loc[table.task == 12, "mean_abs_dphi_l2"] = 1e-9
    assert A.low_response_onset(table) == 10
    table.insert(0, "geometry", "rgb32")
    table.insert(1, "optimizer", "adam")
    table.insert(2, "seed", 300)
    summary, _ = A.summarize_seeds(table)
    assert summary.loc[0, "dynamics_label"] == "LOW_RESPONSE"
    assert summary.loc[0, "low_response_onset_task"] == 10
    assert summary.loc[0, "P2"] == "FAIL"


def test_terminal_divergence_preflight_requires_exact_done_and_snapshot_boundary(tmp_path):
    failed_key = ("rgb32", "adam", 300)
    failed_status = None
    for geometry, optimizer, seed, directory in A.expected_series(tmp_path):
        directory.mkdir(parents=True)
        is_failed = (geometry, optimizer, seed) == failed_key
        status = {"geometry": geometry, "optimizer": optimizer, "seed": seed,
                  "status": "DIVERGED" if is_failed else "COMPLETED",
                  "last_saved_task": 2 if is_failed else A.TASKS,
                  "first_nonfinite_task": 3 if is_failed else None}
        (directory / "status.json").write_text(json.dumps(status))
        if is_failed:
            failed_status = status
        for task in range(status["last_saved_task"] + 1):
            (directory / f"t{task:03d}.npz").touch()
        bank = tmp_path / "banks" / geometry / f"seed{seed}.npz"
        bank.parent.mkdir(parents=True, exist_ok=True)
        bank.touch()
        labels = tmp_path / "labels" / f"seed{seed}.npz"
        labels.parent.mkdir(parents=True, exist_ok=True)
        labels.touch()
    assert failed_status is not None
    source = A.REPO / "src/effdisp_imagegeom_0924.py"
    spec = A.REPO / "specs/spec_effdisp_imagegeom_0924.md"
    metadata = {"git_hash": "a"*40, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                "finished_at_utc": "2026-09-24T00:00:00Z", "tasks": A.TASKS,
                "seeds": list(A.SEEDS), "epochs": 400, "batch": 16,
                "images_per_seed": A.N_IMAGES, "steps_per_task": 30000,
                "args": {"tasks": A.TASKS, "seeds": list(A.SEEDS)},
                "geometries": list(A.GEOMETRIES), "geometry_dims": A.GEOMETRIES,
                "optimizers": list(A.OPTIMIZERS),
                "optimizer_settings": {"adam": {"lr": .001, "betas": [.9, .999], "eps": 1e-8},
                                       "sgd": {"lr": .01, "momentum": 0}},
                "cases": [{"geometry": g, "optimizer": o, "seed": s,
                           "dims": [A.GEOMETRIES[g], 100, 100, 10]}
                          for g in A.GEOMETRIES for o in A.OPTIMIZERS for s in A.SEEDS]}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    done = {"status": "FAILED", "expected_cases": 50, "completed_cases": 49,
            "failed_cases": [failed_status], "git_hash": metadata["git_hash"],
            "source_sha256": metadata["source_sha256"]}
    done_path = tmp_path / "DONE.json"
    done_path.write_text(json.dumps(done))
    completion, _ = A.preflight(tmp_path)
    assert completion.attrs.get("preflight_errors") is None
    failed = completion[(completion.geometry == "rgb32") & (completion.optimizer == "adam")
                        & (completion.seed == 300)].iloc[0]
    assert failed.status == "DIVERGED" and failed.last_saved_task == 2
    # A failure at task 3 leaves the registered early window unavailable.
    short = pd.DataFrame([{"geometry": "rgb32", "optimizer": "adam", "seed": 300,
                           "task": t, **{k: 1.0 for k in
                           ("R", "d_eff", "c", "rho", "train_acc", "top10_Q_fraction",
                            "mean_shift_sq", "raw_W_sq", "raw_D_sq")}}
                          for t in range(1, 3)])
    windows = A.seed_window_medians(short, completion[(completion.geometry == "rgb32")
                                                 & (completion.optimizer == "adam")
                                                 & (completion.seed == 300)])
    failed_windows = windows[(windows.geometry == "rgb32") & (windows.optimizer == "adam")
                     & (windows.seed == 300)]
    assert not failed_windows.available.any()
    assert set(failed_windows.series_status) == {"DIVERGED"}
    invalid = A.paired_contrasts(windows)
    assert (invalid[(invalid.seed == 300) & (invalid.geometry == "rgb32")
                    & (invalid.optimizer == "adam")].availability == "MISSING_OR_NONFINITE_WINDOW").all()
    assert invalid.ratio.isna().all()
    # A failure at task 30 still leaves the entire registered early window.
    later_completion = pd.DataFrame([
        {"geometry": "rgb32", "optimizer": "adam", "seed": 300, "status": "DIVERGED",
         "last_saved_task": 29, "first_nonfinite_task": 30},
        {"geometry": "dup64", "optimizer": "adam", "seed": 300, "status": "COMPLETE",
         "last_saved_task": 50, "first_nonfinite_task": None}])
    later = pd.DataFrame([{"geometry": geometry, "optimizer": "adam", "seed": 300,
                           "task": t, **{k: 2.0 if geometry == "dup64" else 1.0 for k in
                           ("R", "d_eff", "c", "rho", "train_acc", "top10_Q_fraction",
                            "mean_shift_sq", "raw_W_sq", "raw_D_sq")}}
                          for geometry, last in (("rgb32", 29), ("dup64", 50))
                          for t in range(1, last+1)])
    later_windows = A.seed_window_medians(later, later_completion)
    early = later_windows[(later_windows.geometry == "rgb32") & (later_windows.window == "early")].iloc[0]
    late = later_windows[(later_windows.geometry == "rgb32") & (later_windows.window == "late")].iloc[0]
    assert early.available and early.series_status == "DIVERGED"
    assert not late.available and late.unavailable_reason == "MISSING_TASKS"
    later_pairs = A.paired_contrasts(later_windows)
    early_pair = later_pairs[(later_pairs.geometry == "dup64") & (later_pairs.optimizer == "adam")
                             & (later_pairs.seed == 300) & (later_pairs.window == "early")
                             & (later_pairs.metric == "d_eff")].iloc[0]
    late_pair = later_pairs[(later_pairs.geometry == "dup64") & (later_pairs.optimizer == "adam")
                            & (later_pairs.seed == 300) & (later_pairs.window == "late")
                            & (later_pairs.metric == "d_eff")].iloc[0]
    assert early_pair.availability == "AVAILABLE" and early_pair.terminal_divergence
    assert early_pair.ratio == 2
    assert late_pair.availability == "MISSING_OR_NONFINITE_WINDOW" and math.isnan(late_pair.ratio)
    done["failed_cases"] = []
    done_path.write_text(json.dumps(done))
    completion, _ = A.preflight(tmp_path)
    assert any("DONE failed list" in item for item in completion.attrs["preflight_errors"])
    done["failed_cases"] = [failed_status]
    done_path.write_text(json.dumps(done))
    (tmp_path / "rgb32" / "adam" / "seed300" / "t002.npz").unlink()
    completion, _ = A.preflight(tmp_path)
    assert any("nonterminal" in item for item in completion.attrs["preflight_errors"])
    (tmp_path / "rgb32" / "adam" / "seed300" / "t002.npz").touch()
    failed_status["status"] = "RUNNING"
    (tmp_path / "rgb32" / "adam" / "seed300" / "status.json").write_text(json.dumps(failed_status))
    completion, _ = A.preflight(tmp_path)
    assert any("nonterminal" in item for item in completion.attrs["preflight_errors"])
