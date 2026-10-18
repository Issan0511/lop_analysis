#!/usr/bin/env python3
"""Synthetic tests of verdict.py (no real run is read).

    cd analysis/cifar5p1_cnn_budget_1019 && python3 -m unittest test_verdict -v
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V  # noqa: E402

CFG = {"R": "R_lr0.0003", "ELU": "ELU_a3.6_lr0.0003", "KKT1": "KKT1_lr0.0003", "SNAfrz1": "SNAfrz1_lr0.0003"}


def wcsv(p: Path, rows):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fake(root: Path, eff=None, status="COMPLETED", drop=None, diverge=None):
    """eff[(arm, budget)] = {"first780", "all", "fresh29", "logW", "r", "phi", "dead", "grow"} offsets; a value may be
    a function of the seed position."""
    eff = eff or {}
    rng = np.random.default_rng(5)
    base = {s: 0.70 + 0.02 * rng.standard_normal() for s in V.SEEDS}
    get = lambda e, k, i: (e.get(k, 0.0)(i) if callable(e.get(k, 0.0)) else e.get(k, 0.0))
    for b in V.BUDGETS:
        d = root / f"b{b}"
        d.mkdir(parents=True)
        rows, probe, fresh, frozen = [], [], [], {}
        for a in V.ARMS:
            e = eff.get((a, b), {})
            for i, s in enumerate(V.SEEDS):
                for t in range(1, 31):
                    if drop == (a, b) and s == 2009 and t == 30:
                        continue
                    v = base[s] + get(e, "all", i)
                    rows.append({"arm": a, "cfg": CFG[a], "lr": 0.0003, "budget": b, "seed": s, "slot": i, "task": t,
                                 "online_acc": v, "first100_online_acc": v - 0.2,
                                 "first780_online_acc": base[s] + get(e, "first780", i), "train_acc": v + 0.1,
                                 "test_acc": v, "diverged": "True" if diverge == (a, b) and t > 10 else "False",
                                 "bad_step": -1})
                for site in V.SITES:
                    for bd in range(0, 31):
                        grow = 1.0 + 0.05 * bd * (1 + get(e, "grow", i))
                        s_med = grow * (np.exp(get(e, "logW", i)) if bd == 30 else np.exp(get(e, "w2", i)) if bd == 2
                                        else 1.0)
                        probe.append({"arm": a, "cfg": CFG[a], "seed": s, "boundary": bd, "site": site,
                                      "s_med": s_med, "m_med": -0.1, "r_med": -0.5 + (get(e, "r", i) if bd == 30 else 0),
                                      "dmean_med": 0.5 + (get(e, "phi", i) if bd == 30 else 0), "ppos_med": 0.3,
                                      "dead_frac": get(e, "dead", i) if bd == 30 else 0.0, "lowresp_frac": 0.1,
                                      "zeroout_frac": 0.0, "eff_rank": 10.0})
                for t in V.FRESH:
                    g = get(e, "fresh29", i) if t == 29 else -0.01
                    fresh.append({"arm": a, "cfg": CFG[a], "budget": b, "seed": s, "task": t, "fresh_bad_step": -1,
                                  "gap_online_acc": g, "gap_first780_online_acc": g, "gap_first100_online_acc": g,
                                  "gap_test_acc": g})
                if a == "SNAfrz1":
                    frozen[f"{CFG[a]}|{s}"] = {site: [1.0 - b / 1e5] * 16 for site in V.SITES}
        wcsv(d / "per_task.csv", rows)
        wcsv(d / "probe_stats.csv", probe)
        wcsv(d / "fresh_control.csv", fresh)
        (d / "frozen_alpha.json").write_text(json.dumps({"runs": frozen}))
        (d / "provenance.json").write_text(json.dumps({"status": status}))


class Tests(unittest.TestCase):
    def go(self, eff=None, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "cifar5p1_cnn_budget_1019"
            fake(root, eff, **kw)
            res = V.compute(root, prev_check=False)
            self.assertIn("読みの規則", V.summary_md(res))
            json.dumps(res, default=float)
            return res

    def test_labels(self):
        alt = lambda i: 0.01 * (-1) ** i
        r = self.go({("ELU", 7800): {"first780": -0.02, "logW": 0.5}, ("KKT1", 7800): {"first780": +0.02},
                     ("R", 2340): {"first780": alt}, ("SNAfrz1", 2340): {"fresh29": +0.03, "logW": -0.5}})
        F = r["families"]
        self.assertEqual(F["F1_first780"]["ELU_7800_minus_780"]["label"], "LOWER")
        self.assertEqual(F["F1_first780"]["KKT1_7800_minus_780"]["label"], "HIGHER")
        self.assertEqual(F["F1_first780"]["R_2340_minus_780"]["label"], "INCONCLUSIVE")
        self.assertEqual(F["F4_logW30"]["ELU_7800_minus_780"]["label"], "HIGHER")
        self.assertEqual(F["F4_logW30"]["SNAfrz1_2340_minus_780"]["label"], "LOWER")
        self.assertEqual(F["F3_fresh29"]["SNAfrz1_2340_minus_780"]["label"], "HIGHER")
        self.assertEqual(F["F2_all"]["ELU_7800_minus_780"]["label"], "INCONCLUSIVE")
        self.assertEqual(len(F["F1_first780"]), 8)
        fa = r["descriptive"]["frozen_alpha_diff_B_minus_780"]["7800_f2"]
        self.assertAlmostEqual(fa["mean"], -(7800 - 780) / 1e5)

    def test_rules(self):
        # ELU: width grows more with budget, fresh gap below 0, F1 not LOWER, no dead -> healthy
        r = self.go({("ELU", 7800): {"logW": 0.5, "fresh29": -0.02}})
        self.assertTrue(r["rules"]["ELU_7800"]["1_width_keeps_growing"])
        self.assertTrue(r["ELU_healthy_after_growth"])
        self.assertFalse(r["rules"]["ELU_7800"]["next_intervention_candidate"])
        # one seed with a dead FC2 unit -> not healthy
        r = self.go({("ELU", 7800): {"logW": 0.5, "fresh29": -0.02, "dead": lambda i: 0.01 if i == 3 else 0.0}})
        self.assertFalse(r["ELU_healthy_after_growth"])
        # fresh gap interval not below 0 -> not healthy
        r = self.go({("ELU", 7800): {"logW": 0.5, "fresh29": lambda i: 0.02 * (-1) ** i}})
        self.assertFalse(r["ELU_healthy_after_growth"])
        # KKT1: growth + F1 LOWER -> candidate; growth + (r, phi) drop -> candidate via rule 2; no growth -> none
        r = self.go({("KKT1", 7800): {"logW": 0.5, "first780": -0.02},
                     ("SNAfrz1", 7800): {"logW": 0.5, "r": -0.3, "phi": -0.1},
                     ("R", 7800): {"first780": -0.02}})
        self.assertTrue(r["rules"]["KKT1_7800"]["next_intervention_candidate"])
        self.assertTrue(r["rules"]["SNAfrz1_7800"]["2_negative_shift_and_response_drop"])
        self.assertTrue(r["rules"]["SNAfrz1_7800"]["next_intervention_candidate"])
        self.assertFalse(r["rules"]["R_7800"]["1_width_keeps_growing"])
        self.assertFalse(r["rules"]["R_7800"]["next_intervention_candidate"])
        self.assertEqual(sorted(r["next_intervention_candidates"]), ["KKT1_7800", "SNAfrz1_7800"])
        # within-run growth must be positive too: F4 HIGHER but the width at boundary 2 above that at 30 -> rule 1 fails
        r = self.go({("KKT1", 7800): {"logW": 0.5, "w2": 2.0, "first780": -0.02}})
        g = r["descriptive"]["growth_logW30_minus_logW2"]["KKT1_7800_f2"]
        self.assertEqual(r["families"]["F4_logW30"]["KKT1_7800_minus_780"]["label"], "HIGHER")
        self.assertLess(g["interval"][1], 0)
        self.assertFalse(r["rules"]["KKT1_7800"]["1_width_keeps_growing"])
        self.assertFalse(r["rules"]["KKT1_7800"]["next_intervention_candidate"])

    def test_incomplete(self):
        r = self.go({}, drop=("KKT1", 2340))
        self.assertEqual(r["families"]["F1_first780"]["KKT1_2340_minus_780"]["label"], "INCOMPLETE")
        self.assertNotEqual(r["families"]["F1_first780"]["KKT1_7800_minus_780"]["label"], "INCOMPLETE")
        r = self.go({}, drop=("ELU", 780))
        self.assertEqual(r["families"]["F2_all"]["ELU_7800_minus_780"]["label"], "INCOMPLETE")
        self.assertEqual(r["families"]["F2_all"]["ELU_2340_minus_780"]["label"], "INCOMPLETE")
        r = self.go({}, diverge=("R", 7800))
        self.assertEqual(r["families"]["F4_logW30"]["R_7800_minus_780"]["label"], "INCOMPLETE")
        r = self.go({}, status="FAILED")
        self.assertTrue(all(v["label"] == "INCOMPLETE" for fam in r["families"].values() for v in fam.values()))
        self.assertFalse(r["ELU_healthy_after_growth"])

    def test_running_refused(self):
        with self.assertRaises(SystemExit):
            self.go({}, status="RUNNING")

    def test_bootstrap_and_thresholds(self):
        idx = np.random.default_rng(1019).integers(0, 10, size=(20000, 10))
        d = np.arange(10) / 1000
        x = V.iv(d, idx, V.Q8)
        means = d[idx].mean(1)
        self.assertEqual(x["interval"], [float(np.quantile(means, 0.003125)), float(np.quantile(means, 0.996875))])
        self.assertEqual(V.Q8, (0.003125, 0.996875))
        self.assertEqual(V.label({"interval": [0.0, 1.0]}), "INCONCLUSIVE")
        self.assertEqual(V.label({"interval": [-1.0, 0.0]}), "INCONCLUSIVE")
        self.assertEqual(V.label({"interval": [1e-12, 1.0]}), "HIGHER")
        self.assertEqual(V.iv(np.r_[d[:9], np.nan], idx, V.Q8)["label"], "INCOMPLETE")

    def test_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "cifar5p1_cnn_budget_1019"
            fake(root, {})
            prev = Path(tmp) / Path(*V.PREV)
            rows = V.read_csv(root / "b780" / "per_task.csv")
            for b, arms in (("A", ("R", "SNAfrz1")), ("B", ("ELU", "KKT1"))):
                (prev / b).mkdir(parents=True)
                q = [{**{k: v for k, v in r.items() if k not in ("budget", "first780_online_acc")},
                      "slot": "99", "gate_mean": "0.5"} for r in rows if r["arm"] in arms]
                q += [{**q[0], "cfg": "SNA_lr0.0003", "arm": "SNA"}]
                wcsv(prev / b / "per_task.csv", q)
            res = V.compute(root)
            self.assertEqual(res["S_reuse_780"]["label"], "PASS")
            self.assertNotIn("slot", res["S_reuse_780"]["columns"])
            self.assertNotIn("gate_mean", res["S_reuse_780"]["columns"])
            self.assertIn("online_acc", res["S_reuse_780"]["columns"])
            q = V.read_csv(prev / "B" / "per_task.csv")
            q[5]["test_acc"] = "0.123"
            wcsv(prev / "B" / "per_task.csv", q)
            res = V.compute(root)
            self.assertEqual(res["S_reuse_780"]["label"], "FAIL")
            self.assertEqual(res["S_reuse_780"]["mismatches_per_column"], {"test_acc": 1})


if __name__ == "__main__":
    unittest.main()
