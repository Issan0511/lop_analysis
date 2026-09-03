"""Tests for ``src.bshare_posthoc_0903`` on synthetic logs (numpy only).

The synthetic trajectories are built by literally applying ``Δw = c·x'`` and
``Δb = c`` with random negative ``c`` and condA inputs, then logging
``M``/``B``/``denom``/``p_hat`` the way ``exact_layer_record`` does (32-pattern
support, ``M = w·µ'/denom``).  The analysis must then recover the geometric
share to within the finite-sample spread of ``x'·µ'``.

``test_rotation_free_*`` uses a hand-built log instead, so that the µ'
reorientation term has an exactly known size.
"""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import bshare_posthoc_0903 as P

F, FREE = 15, 5
PATTERNS = np.array(list(itertools.product([0, 1], repeat=FREE)), np.float64)


def gamma_for_k(k, target):
    disc = (k + 2.5) ** 2 - 20.0 * (k + 1.25 - target ** 2)
    return ((k + 2.5) - np.sqrt(disc)) / 10.0


def _support(flip, offset):
    X = np.concatenate([np.tile(flip, (32, 1)), PATTERNS], axis=1) - offset
    return X                                  # (32, 20)


def _record(W, b, X):
    mu = X.mean(axis=0)
    z = X @ W.T + b                           # (32, H)
    centered = (X - mu) @ W.T
    denom = np.sqrt((centered ** 2).mean(axis=0))
    M, B = (W @ mu) / denom, b / denom
    return dict(M=M, B=B, denom=denom, p_hat=(z > 0).mean(axis=0),
                zbar=z.mean(axis=0), zmax=z.max(axis=0))


def synth_log(seed, target, *, H=40, n_rec=60, steps_per_rec=50, seed_id=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0.3, 0.1, (H, F + FREE))    # start well above the wall
    b = rng.normal(0.5, 0.1, H)
    flip = rng.integers(0, 2, F).astype(np.float64)
    logs = {k: [] for k in ("M", "B", "denom", "p_hat", "zbar", "zmax")}
    flips, steps = [], []
    for r in range(n_rec):
        k = flip.sum()
        offset = 0.5 * gamma_for_k(k, target) if target is not None else 0.0
        rec = _record(W, b, _support(flip, offset))
        for key, val in rec.items():
            logs[key].append(val)
        flips.append(flip.copy())
        steps.append(r * steps_per_rec)
        for _ in range(steps_per_rec):        # Δw = c x', Δb = c with c < 0
            x = np.concatenate([flip, rng.integers(0, 2, FREE)]) - offset
            c = -rng.uniform(0.0, 0.02, H)
            W += c[:, None] * x[None, :]
            b += c
        if r % 7 == 6:                        # occasional task boundary
            j = rng.integers(0, F)
            flip[j] = 1 - flip[j]
    out = {f"layer1_{k}": np.array(v, np.float32) for k, v in logs.items()}
    out["step"] = np.array(steps, np.int64)
    out["flip_state"] = np.array(flips, np.float64)
    out["seed"] = np.int64(seed_id)
    out["target_mu_norm"] = np.float64(np.nan if target is None else target)
    return out


def handmade_log():
    """Two units, six records, one task flip between records 2 and 3.

    Unit 0's ``w·µ'`` walks down by 1 per interval but jumps +4 at the flip, so
    the endpoint difference is exactly 0 while the flip-free accumulation is
    exactly −4.  ``b`` walks down by 0.1 per interval with no jump.
    """
    wmu = np.array([[10.0, 5.0], [9.0, 5.0], [8.0, 5.0],
                    [12.0, 5.0], [11.0, 5.0], [10.0, 5.0]])
    b = np.array([[1.0, 2.0], [0.9, 2.0], [0.8, 2.0],
                  [0.7, 2.0], [0.6, 2.0], [0.5, 2.0]])
    p_hat = np.full((6, 2), 0.5)
    p_hat[5, 0] = 0.0                          # unit 0 submerges at record 5
    flip = np.zeros((6, F))
    flip[3:, 0] = 1.0                          # the single task boundary
    return {"layer1_M": wmu.astype(np.float32),
            "layer1_B": b.astype(np.float32),
            "layer1_denom": np.ones((6, 2), np.float32),
            "layer1_p_hat": p_hat.astype(np.float32),
            "layer1_zbar": (wmu + b).astype(np.float32),
            "flip_state": flip, "step": np.arange(6, dtype=np.int64) * 1000,
            "seed": np.int64(0), "target_mu_norm": np.float64(3.041)}


def handmade_log_raw():
    """``handmade_log`` with no oracle dose, so ‖µ'‖² = k + 1.25 varies (k: 0→1)."""
    z = handmade_log()
    z["target_mu_norm"] = np.float64(np.nan)
    return z


class ShareRecovery(unittest.TestCase):
    def test_oracle_arm_recovers_geometric_share(self):
        for target, expect in ((3.041, 1 / (1 + 3.041 ** 2)), (2.333, 1 / (1 + 2.333 ** 2))):
            res = P.analyse_seed(synth_log(1, target))
            s = P.summarise_seed(res)
            self.assertGreater(s["n_units"], 10)
            self.assertAlmostEqual(res["pred_unit"][np.isfinite(res["pred_unit"])][0], expect)
            # the share is a |c|-weighted mean of x'·µ'; allow the sampling spread
            self.assertLess(abs(s["share_of_medians"] - expect), 0.01, (target, s))
            self.assertLess(abs(s["med_share_unit"] - expect), 0.01, (target, s))
            self.assertLess(abs(s["share_of_medians_ff"] - expect), 0.01, (target, s))

    def test_raw_arm_prediction_tracks_k(self):
        res = P.analyse_seed(synth_log(2, None))
        s = P.summarise_seed(res)
        self.assertEqual(res["pred_kind"], "raw")
        self.assertGreater(s["n_units"], 10)
        self.assertLess(abs(s["share_of_medians"] - s["med_pred"]), 0.012, s)

    def test_raw_prediction_is_reciprocal_of_mean_not_mean_of_reciprocals(self):
        k = np.array([2.0, 8.0, 14.0])
        recip_of_mean = P.predicted_share_raw_window(k)
        mean_of_recip = float(np.mean(P.predicted_share_raw(k)))
        self.assertAlmostEqual(recip_of_mean, 1 / (1 + 8.0 + 1.25))
        self.assertLess(recip_of_mean, mean_of_recip)          # Jensen, 1/x convex
        res = P.analyse_seed(synth_log(2, None))
        g = np.isfinite(res["pred_unit"])
        self.assertTrue(np.all(res["pred_unit"][g] <= res["pred_meanrecip"][g] + 1e-12))
        # a window that actually spans a change in k separates the two
        raw = P.analyse_seed(handmade_log_raw())
        self.assertEqual(raw["pred_kind"], "raw")
        k_int = np.array([0.0, 0.0, 0.0, 1.0, 1.0])          # records 0..4
        self.assertAlmostEqual(raw["pred_unit"][0], 1 / (1 + k_int.mean() + 1.25))
        self.assertAlmostEqual(raw["pred_meanrecip"][0],
                               float(np.mean(1 / (1 + k_int + 1.25))))
        self.assertLess(raw["pred_unit"][0], raw["pred_meanrecip"][0])
        # the flip-free prediction only sees the intervals it accumulated
        self.assertAlmostEqual(raw["pred_ff"][0],
                               1 / (1 + np.array([0.0, 0.0, 1.0, 1.0]).mean() + 1.25))

    def test_rotation_free_estimator_removes_the_mu_jump(self):
        res = P.analyse_seed(handmade_log())
        self.assertEqual(res["counts"]["ok"], 1)
        self.assertEqual(res["counts"]["never_submerged"], 1)
        # endpoint difference: the +4 reorientation exactly cancels the descent
        self.assertAlmostEqual(res["d_wmu"][0], 0.0, places=5)
        self.assertAlmostEqual(res["d_b"][0], -0.5, places=5)
        self.assertAlmostEqual(res["share_unit"][0], 1.0, places=5)
        # flip-free accumulation drops the one contaminated interval
        self.assertEqual(int(res["n_ff"][0]), 4)
        self.assertAlmostEqual(res["d_wmu_ff"][0], -4.0, places=5)
        self.assertAlmostEqual(res["d_b_ff"][0], -0.4, places=5)
        self.assertAlmostEqual(res["share_ff"][0], -0.4 / -4.4, places=5)
        # and the endpoint routes do NOT both descend here - that is the point
        self.assertFalse(res["same_sign"][0])
        self.assertEqual(int(res["t_sub"][0]), 5000)
        self.assertEqual(int(res["win_len"][0]), 5)

    def test_all_nan_gamma_does_not_kill_the_flip_free_mask(self):
        """The raw ("off") arms log ``gamma`` as all-NaN; NaN == NaN is False."""
        z = handmade_log_raw()
        z["gamma"] = np.full(6, np.nan)
        res = P.analyse_seed(z)
        self.assertEqual(int(res["n_ff"][0]), 4)
        self.assertAlmostEqual(res["d_wmu_ff"][0], -4.0, places=5)
        # a real oracle gamma that moves with the flip still masks that interval
        z2 = handmade_log()
        z2["gamma"] = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.assertEqual(int(P.analyse_seed(z2)["n_ff"][0]), 4)
        z3 = handmade_log()
        z3["gamma"] = np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.2])   # extra move
        self.assertEqual(int(P.analyse_seed(z3)["n_ff"][0]), 3)

    def test_start_record_moves_the_window(self):
        z = handmade_log()
        res = P.analyse_seed(z, start_record=2)
        # records 2..5: the weight route now ends ABOVE where it started, so the
        # unit is dropped as no_descent - but the endpoint differences are kept
        self.assertEqual(res["counts"]["no_descent"], 1)
        self.assertAlmostEqual(res["d_wmu_all"][0], 10.0 - 8.0, places=5)
        self.assertAlmostEqual(res["d_b_all"][0], 0.5 - 0.8, places=5)
        self.assertTrue(np.isnan(res["d_wmu"][0]))
        self.assertEqual(int(res["win_len"][0]), 3)
        self.assertEqual(int(res["n_ff"][0]), 2)               # intervals 3→4, 4→5
        with self.assertRaises(P.SanityError):
            P.analyse_seed(z, start_record=5)

    def test_same_sign_flag_marks_a_climbing_bias_route(self):
        z = handmade_log()
        b = np.asarray(z["layer1_B"], np.float64)
        b[:, 0] = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])     # bias route climbs
        z["layer1_B"] = b.astype(np.float32)
        z["layer1_zbar"] = (np.asarray(z["layer1_M"], np.float64) + b).astype(np.float32)
        res = P.analyse_seed(z)
        # d_wmu = 0, d_b = +0.5 -> no net descent at all
        self.assertEqual(res["counts"]["no_descent"], 1)
        self.assertTrue(np.isfinite(res["d_b_all"][0]))
        self.assertFalse(res["same_sign"][0])

    def test_exclusions_are_counted(self):
        z = synth_log(3, 3.041, n_rec=3, steps_per_rec=1)   # too short to submerge
        res = P.analyse_seed(z)
        self.assertEqual(sum(res["counts"].values()), res["width"])
        self.assertGreater(res["counts"]["never_submerged"], 0)

    def test_zbar_sanity_trips(self):
        z = synth_log(4, 3.041)
        z["layer1_zbar"] = z["layer1_zbar"] + 1.0
        with self.assertRaises(P.SanityError):
            P.analyse_seed(z)

    def test_end_to_end_writes_registered_zero(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "run"
            (src / "logs").mkdir(parents=True)
            for arm, target in (("R_933", 2.333), ("R_off", None)):
                for sd in range(2):
                    np.savez_compressed(src / "logs" / f"{arm}_seed{sd}.npz",
                                        **synth_log(10 + sd, target, seed_id=sd))
            out = src / "posthoc_bshare_0903"
            res = P.run(src, out, min_descent=0.0, arms_filter=None)
            self.assertTrue((out / "bshare_by_arm.csv").exists())
            self.assertTrue(all(r["registered"] == 0 for r in res["arm_rows"]))
            self.assertEqual(res["provenance"]["analysis_grade"], P.ANALYSIS_GRADE)
            self.assertIsNone(res["provenance"]["arms_filter"])
            self.assertEqual(res["provenance"]["start_record"], 0)
            self.assertIn("git_dirty", res["provenance"])
            self.assertEqual(len(res["provenance"]["module_sha256"]), 64)
            # synth_log carries layer1_zmax, so the cross-check is live here
            self.assertIn("present and checked", res["provenance"]["zmax_cross_check"])
            self.assertFalse(P.analyse_seed(handmade_log())["zmax_checked"])
            arms = {r["arm"]: r for r in res["arm_rows"]}
            self.assertAlmostEqual(arms["R_933"]["seedmed_med_pred"], 1 / (1 + 2.333 ** 2))
            self.assertLess(abs(arms["R_933"]["diff_share_minus_pred"]), 0.01)
            body = (out / "summary.md").read_text()
            self.assertIn("diagnostics", body)
            self.assertIn("mu_titration_0823", body)
            head = (out / "bshare_by_unit.csv").read_text().splitlines()[0]
            for col in ("routes_same_sign", "share_unit_ff", "pred_meanrecip",
                        "n_ff_intervals", "win_len"):
                self.assertIn(col, head)

    def test_arms_filter_is_recorded_and_can_miss(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "run"
            (src / "logs").mkdir(parents=True)
            np.savez_compressed(src / "logs" / "R_933_seed0.npz",
                                **synth_log(10, 2.333, seed_id=0))
            out = src / "o"
            res = P.run(src, out, min_descent=0.0, arms_filter=["R_933"])
            self.assertEqual(res["provenance"]["arms_filter"], ["R_933"])
            with self.assertRaises(P.SanityError):
                P.run(src, out, min_descent=0.0, arms_filter=["nope"])


if __name__ == "__main__":
    unittest.main()
