"""Tests for ``src.bshare_posthoc_0903`` on synthetic logs (numpy only).

The synthetic trajectories are built by literally applying ``Δw = c·x'`` and
``Δb = c`` with random negative ``c`` and condA inputs, then logging
``M``/``B``/``denom``/``p_hat`` the way ``exact_layer_record`` does (32-pattern
support, ``M = w·µ'/denom``).  The analysis must then recover the geometric
share to within the finite-sample spread of ``x'·µ'``.
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

    def test_raw_arm_prediction_tracks_k(self):
        res = P.analyse_seed(synth_log(2, None))
        s = P.summarise_seed(res)
        self.assertEqual(res["pred_kind"], "raw")
        self.assertGreater(s["n_units"], 10)
        self.assertLess(abs(s["share_of_medians"] - s["med_pred"]), 0.012, s)

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
            arms = {r["arm"]: r for r in res["arm_rows"]}
            self.assertAlmostEqual(arms["R_933"]["seedmed_pred"], 1 / (1 + 2.333 ** 2))
            self.assertLess(abs(arms["R_933"]["diff_share_minus_pred"]), 0.01)


if __name__ == "__main__":
    unittest.main()
