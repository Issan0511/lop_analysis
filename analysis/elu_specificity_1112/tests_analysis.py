"""Independent synthetic checks of analysis, without loading data/models/CUDA.

Run with the project environment:
  python -m unittest analysis.elu_specificity_1112.tests_analysis -v
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

_path = Path(__file__).with_name("analyze.py")
if not _path.exists():
    _path = Path(__file__).with_name("elu_analyze.py")
_spec = importlib.util.spec_from_file_location("elu_analysis_under_test", _path)
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)


def fixture(study="A", activation="ELU", ln="off", seed=0, age=10, arm="S1"):
    # Deliberately construct the protocol independently from implementation constants.
    horizon, every, n = (4000, 100, 1200) if study == "A" else (1500, 25, 2500)
    grid = sorted(set([0, 1, 5, 25, 50, 100, 199, 200, 201, horizon]
                      + list(range(0, horizon+1, every))))
    j = seed if study == "A" else seed-10252010
    coeff = {"ELU": .03, "R": .01, "LR": .02, "SILU": .015}[activation]
    age_factor = (2 if age == 10 else 1) if study == "A" else 1
    benefit = .02 + coeff*age_factor*(1 if ln == "off" else .5) + j*.0001
    p1 = .55+j*.001
    s1 = .60+j*.001
    losses = {"S1": s1, "S01": s1-benefit-.01, "P1": p1, "P01": p1-benefit,
              "Z_keep": .35+j*.001, "Z_clear": .30+j*.001, "F0": .40+j*.001}
    source_act = activation if study == "A" else "R"
    return dict(study=study, model="mlp" if study == "A" else "vit", activation=activation,
        source_activation=source_act, ln=ln, seed=seed, t_end=age, arm=arm,
        horizon=horizon, n_examples=n, complete=True,
        source_id=f"{study}_{source_act}_{ln}_{seed}_{age}", source_sha256="a"*64,
        provenance=dict(execution_git_hash="b"*40, spec_sha256="c"*64, code_sha256="d"*64),
        checks=dict(passed=True), tracks=[], curve=[dict(step=s, acc=1-losses[arm], ce=2.0) for s in grid])


def full_fixture():
    rows = []
    for st, acts, seeds, ages in (("A", ("ELU", "R", "LR", "SILU"), range(10), (1, 10)),
                                  ("B", ("R", "ELU"), range(10252010, 10252015), (30,))):
        for act, ln, seed, age, arm in itertools.product(acts, ("off", "on"), seeds, ages,
                ("S1", "S01", "P1", "P01", "Z_keep", "Z_clear", "F0")):
            rows.append(fixture(st, act, ln, seed, age, arm))
    return rows


def independent_trapezoid(x, y):
    return sum((y[i]+y[i+1])*(x[i+1]-x[i])/2 for i in range(len(x)-1))


class MetricsTests(unittest.TestCase):
    def test_integrals_match_analytic_linear_curve(self):
        r = fixture()
        for q in r["curve"]:
            q["acc"] = .1 + .9*q["step"]/4000
            q["ce"] = 4*(1-q["step"]/4000)
        m = a.metric_row(r)
        self.assertAlmostEqual(m["U"], .45)
        self.assertAlmostEqual(m["U_acute"], .8775)
        self.assertAlmostEqual(m["U_late"], .4275)
        self.assertAlmostEqual(m["U"], (200*m["U_acute"]+3800*m["U_late"])/4000)
        self.assertAlmostEqual(m["ce_area"], 8000)
        self.assertAlmostEqual(m["ce_auc_mean"], 2)
        self.assertEqual(m["accuracy_end"], 1)
        self.assertEqual(m["ce_end"], 0)
        self.assertEqual(m["T90_first"], 3600)
        self.assertEqual(m["T90_sustained"], 3600)

    def test_piecewise_integral_independent_loop(self):
        r = fixture(study="B", activation="R", seed=10252010, age=30)
        for q in r["curve"]:
            q["acc"] = .2 if q["step"] < 200 else .8
            q["ce"] = 1.3 + q["step"]*.0001
        m = a.metric_row(r)
        x = [q["step"] for q in r["curve"]]
        self.assertAlmostEqual(m["U"], independent_trapezoid(x, [1-q["acc"] for q in r["curve"]])/1500)
        self.assertAlmostEqual(m["ce_area"], independent_trapezoid(x, [q["ce"] for q in r["curve"]]))

    def test_first_sustained_and_zero_followup(self):
        r = fixture()
        for q in r["curve"]:
            q["acc"] = .95 if q["step"] in (25, 4000) else .8
        m = a.metric_row(r)
        self.assertEqual(m["T90_first"], 25)
        self.assertEqual(m["T90_sustained"], 4000)
        self.assertEqual(m["T90_sustained_observation_duration"], 0)
        self.assertFalse(m["T90_sustained_censored"])
        r["curve"][-1]["acc"] = .8
        m = a.metric_row(r)
        self.assertEqual(m["T90_first"], 25)
        self.assertIsNone(m["T90_sustained"])
        self.assertTrue(m["T90_sustained_censored"])
        self.assertIsNone(m["T90_sustained_observation_duration"])

    def test_never_hit_stays_null_and_float32_boundary(self):
        r = fixture()
        m = a.metric_row(r)
        self.assertIsNone(m["T90_first"])
        self.assertTrue(m["T90_first_censored"])
        self.assertEqual(m["censor_horizon"], 4000)
        r["curve"][-1]["acc"] = float(np.float32(.9))
        self.assertEqual(a.metric_row(r)["T90_first"], 4000)
        r["curve"][-1]["acc"] = 1079/1200
        self.assertIsNone(a.metric_row(r)["T90_first"])


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = full_fixture()

    def test_full_expected_design(self):
        result = a.validate_design(self.records)
        self.assertEqual(result["actual"], 1260)
        self.assertEqual(result["counts"], {"A": 1120, "B": 140})
        self.assertEqual(result["sources"], 170)
        self.assertFalse(result["source_file_bytes_verified"])

    def test_missing_duplicate_wrong_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete/wrong design"):
            a.validate_design(self.records[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate curve"):
            a.validate_design(self.records+[self.records[0]])
        r = fixture(seed=12)
        with self.assertRaisesRegex(ValueError, "Wrong seed"):
            a.validate_record(r)

    def test_nonfinite_metadata_or_diagnostic_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            r = fixture()
            r["curve"][0]["diagnostics"] = {"nested": [bad]}
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                a.validate_record(r)
        r = fixture()
        r["tracks"] = [{"g_norm": float("nan")}]
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            a.validate_record(r)

    def test_grid_and_core_metadata_rejections(self):
        mutations = [lambda r: r["curve"].pop(),
            lambda r: r["curve"].pop(3),
            lambda r: r["curve"].insert(3, copy.deepcopy(r["curve"][3])),
            lambda r: r["curve"].reverse(),
            lambda r: r.update(complete=False), lambda r: r.update(horizon=26),
            lambda r: r.update(n_examples=1199), lambda r: r.update(seed=True),
            lambda r: r.update(model="vit"), lambda r: r.update(source_activation="R"),
            lambda r: r["curve"][0].update(step=0.0),
            lambda r: r["curve"][0].update(acc=1.1),
            lambda r: r["curve"][0].update(acc=True),
            lambda r: r["curve"][0].update(ce=-1),
            lambda r: r.update(steps_run=3999), lambda r: r.update(source_sha256="bad"),
            lambda r: r.update(checks={"passed": False})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                r = fixture()
                mutation(r)
                with self.assertRaises(ValueError):
                    a.validate_record(r)

    def test_pair_source_or_execution_mismatch_rejected(self):
        for field, value, regex in (("source_sha256", "f"*64, "Different source"),
            ("execution_git_hash", "f"*40, "Mixed execution commits"),
            ("spec_sha256", "f"*64, "Mixed specification"),
            ("code_sha256", "f"*64, "Mixed execution code")):
            changed = copy.deepcopy(self.records[0])
            if field == "source_sha256":
                changed[field] = value
            else:
                changed["provenance"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, regex):
                a.validate_design([changed]+self.records[1:])

    def test_B_activation_pair_source_mismatch_rejected(self):
        records = list(self.records)
        i = next(i for i, r in enumerate(records) if r["study"] == "B" and r["activation"] == "ELU")
        records[i] = copy.deepcopy(records[i])
        records[i]["source_sha256"] = "9"*64
        with self.assertRaisesRegex(ValueError, "Different source"):
            a.validate_design(records)

    def test_JSON_loader_ignores_source_but_rejects_curve_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"source_metadata.json").write_text("not curve JSON")
            (root/"curve_ok.json").write_text(json.dumps(fixture()))
            records = a.load_records(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0]["_sha256"]), 64)
            (root/"curve_bad.json").write_text('{"study":"A", "study":"B"}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                a.load_records(root)
            (root/"curve_bad.json").write_text('{"bad":NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                a.load_records(root)


class PairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(a.metric_row(r) for r in full_fixture())

    def find_plan(self, **criteria):
        matches = [p for p in a.contrast_plan() if all(p.get(k) == v for k, v in criteria.items())]
        self.assertEqual(len(matches), 1, criteria)
        return matches[0]

    def test_scale_zero_and_body_utility_signs(self):
        for name, expected in (("B_P", .08045), ("B_S", .09045), ("scale_by_P", .01),
                               ("Z_P", .20), ("M_Z", .05), ("body_utility", -.10)):
            values, complete = a.paired_values(self.df, a.base_terms("A", "ELU", "off", 10, name), "U", "A")
            self.assertTrue(complete.all())
            self.assertAlmostEqual(values.mean(), expected)

    def test_direct_activation_LN_age_interactions(self):
        checks = [
            (dict(study="A", kind="activation", estimand="B_P", activation="ELU-R", ln="off", t_end=10), .04),
            (dict(study="A", kind="activation_by_LN", estimand="B_P", activation="ELU-R", t_end=10), .02),
            (dict(study="A", kind="LN", estimand="B_P", activation="ELU", t_end=10), .03),
            (dict(study="A", kind="age_by_activation", estimand="B_P", activation="ELU-R", ln="off"), .02),
            (dict(study="B", kind="current_activation", estimand="B_P", ln="off"), .02),
            (dict(study="B", kind="current_activation_by_LN_history", estimand="B_P"), .01)]
        for criteria, expected in checks:
            p = self.find_plan(**criteria)
            values, complete = a.paired_values(self.df, p["terms"], "U", p["study"])
            np.testing.assert_allclose(values, expected, atol=2e-15)
            self.assertTrue(complete.all())
            result = a.estimate(values, a.bootstrap_indices(p["study"]))
            self.assertAlmostEqual(result["ci_low"], expected)
            self.assertAlmostEqual(result["ci_high"], expected)

    def test_row_order_does_not_break_seed_pairing(self):
        terms = a.base_terms("A", "ELU", "off", 10, "B_P")
        original, _ = a.paired_values(self.df, terms, "U", "A")
        shuffled, _ = a.paired_values(self.df.sample(frac=1, random_state=97), terms, "U", "A")
        np.testing.assert_array_equal(original, shuffled)
        with self.assertRaisesRegex(ValueError, "Missing/duplicate paired seeds"):
            a.paired_values(self.df.drop(self.df[(self.df.study == "A") & (self.df.activation == "ELU") &
                (self.df.ln == "off") & (self.df.t_end == 10) & (self.df.arm == "P1")].index[0]), terms, "U", "A")

    def test_same_bootstrap_draws_and_no_global_rng_mutation(self):
        np.random.seed(32)
        before = np.random.get_state()
        ix = a.bootstrap_indices("A")
        np.testing.assert_array_equal(ix, a.bootstrap_indices("A"))
        self.assertEqual(ix.shape, (10000, 10))
        after = np.random.get_state()
        for x, y in zip(before, after):
            np.testing.assert_equal(x, y)
        x = np.arange(10, dtype=float)
        result = a.estimate(x, ix)
        means = [sum(x[i] for i in indices)/10 for indices in ix]
        self.assertAlmostEqual(result["estimate"], 4.5)
        np.testing.assert_allclose([result["ci_low"], result["ci_high"]], np.percentile(means, [2.5, 97.5]))

    def test_censor_excludes_entire_contrast_not_selected_seeds(self):
        df = self.df.copy()
        df["T90_first"] = 100.0
        sel = (df.study == "A") & (df.activation == "ELU") & (df.ln == "off") & (df.t_end == 10) & (df.arm == "P1") & (df.seed == 3)
        df.loc[sel, "T90_first"] = np.nan
        values, complete = a.paired_values(df, a.base_terms("A", "ELU", "off", 10, "B_P"), "T90_first", "A")
        self.assertEqual(int(complete.sum()), 9)
        self.assertTrue(np.isnan(values[3]))
        with self.assertRaisesRegex(ValueError, "Nonfinite/censored"):
            a.estimate(values, a.bootstrap_indices("A"))
        means = a.summarize_arms(df)
        arm = means[(means.study == "A") & (means.activation == "ELU") & (means.ln == "off") &
            (means.t_end == 10) & (means.arm == "P1") & (means.metric == "T90_first")].iloc[0]
        self.assertEqual(arm.n_censored, 1)
        self.assertFalse(arm.estimable)
        self.assertTrue(pd.isna(arm.estimate))

    def test_all_planned_contrasts_and_primary_count(self):
        plans = a.contrast_plan()
        self.assertEqual(len(plans), 270)
        self.assertEqual(sum(p["primary_U"] for p in plans), 16)
        for p in plans:
            self.assertAlmostEqual(sum(p["terms"].values()), 0)
            self.assertEqual(set(k[0] for k in p["terms"]), {p["study"]})
        contrasts, seed = a.summarize_contrasts(self.df)
        self.assertEqual(len(contrasts), 2430)
        self.assertEqual(int(contrasts.primary.sum()), 16)
        censored = contrasts[contrasts.metric.str.startswith("T90")]
        self.assertFalse(censored.estimable.any())
        self.assertTrue(censored.estimate.isna().all())
        self.assertEqual(set(censored.verdict), {"censored"})
        self.assertTrue(seed[seed.metric.str.startswith("T90")].value.isna().all())
        self.assertEqual(set(contrasts[contrasts.primary].metric), {"U"})
        text = a.summary_text(self.df, contrasts, dict(actual=1260))
        self.assertIn("not equivalence", text)
        self.assertIn("never replaced", text)
        self.assertIn("ReLU-trained", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
