#!/usr/bin/env python3
"""Synthetic tests of verdict.py (no real run is read).

    python3 -m unittest analysis/neokunekune_cifar5p1_cnn_1018/test_verdict.py -v
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict as V  # noqa: E402

SITES = ("c1", "c2", "f1", "f2")


def fake_run(d: Path, neo_minus_sna, neo_minus_frz, status="COMPLETED", drop=0, task1_differs=False,
             noise=0.002, alternating=False):
    rng = np.random.default_rng(7)
    rows = []
    base = {s: 0.55 + 0.02 * rng.standard_normal() for s in V.SEEDS}
    for i, s in enumerate(V.SEEDS):
        e, e2 = noise * rng.standard_normal(), noise * rng.standard_normal()
        if alternating:                      # per-seed differences centred exactly on the threshold
            e, e2 = noise * (-1) ** i, noise * (-1) ** (i + 1)
        late = {"R": base[s] - 0.10, "SNA": base[s], "SNAfrz1": base[s] + neo_minus_sna - neo_minus_frz + e2,
                "NEO": base[s] + neo_minus_sna + e}
        for a in V.ARMS:
            for t in range(1, 31):
                val = late[a] if t > 1 else base[s] + (0.0 if a != "R" else -0.1)
                if task1_differs and a == "NEO" and t == 1:
                    val += 1e-3
                r = {"arm": a, "seed": s, "slot": 0, "task": t, "hard": int(t % 2 == 1),
                     "online_acc": val, "first100_online_acc": val - 0.1, "train_acc": val + 0.1,
                     "test_acc": val - 0.05, "switch_ce": 9.0, "ce_step1": 9.0, "gate_mean": 0.9,
                     "gate_first10": 0.1, "gate_last10": 1.0, "mult_mean": 0.9, "sum_beta_eff": 7.0,
                     "diverged": False, "bad_step": -1}
                for site in SITES:
                    for f in ("alpha_med", "two_alpha_sqrtV_med", "zsd_med", "dmean_med"):
                        r[f"{f}_{site}"] = 1.0
                rows.append(r)
    rows = rows[:len(rows) - drop]
    with (d / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / "provenance.json").write_text(json.dumps({"status": status}))
    cal = {f"{a}_seed{s}": {"lref": 0.5, "floor_hit": False} for a in ("SNA", "SNAfrz1", "NEO") for s in V.SEEDS}
    (d / "calibration.json").write_text(json.dumps({"runs": cal}))


class VerdictTests(unittest.TestCase):
    def run_case(self, *a, **k):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake_run(d, *a, **k)
            res = V.compute(d)
            md = V.summary_md(res)
            self.assertIn("判定", md)
            return res

    def test_better_and_noninferior(self):
        r = self.run_case(0.02, -0.001)
        self.assertEqual(r["registered_verdict"]["D1"], "BETTER_THAN_SNA")
        self.assertEqual(r["registered_verdict"]["D2"], "NONINFERIOR_TO_FREEZE")
        self.assertTrue(r["registered_verdict"]["both_met"])

    def test_worse_and_inferior(self):
        r = self.run_case(-0.02, -0.03)
        self.assertEqual(r["registered_verdict"]["D1"], "WORSE_THAN_SNA")
        self.assertEqual(r["registered_verdict"]["D2"], "INFERIOR_TO_FREEZE")

    def test_inconclusive(self):
        r = self.run_case(0.0, -0.005, noise=0.01, alternating=True)
        self.assertEqual(r["registered_verdict"]["D1"], "INCONCLUSIVE")
        self.assertEqual(r["registered_verdict"]["D2"], "INCONCLUSIVE")

    def test_incomplete(self):
        r = self.run_case(0.02, 0.0, drop=1)
        self.assertEqual(r["registered_verdict"].get("label"), "INCOMPLETE_OR_FAILED")
        r = self.run_case(0.02, 0.0, status="FAILED_NONFINITE")
        self.assertEqual(r["registered_verdict"].get("label"), "INCOMPLETE_OR_FAILED")
        r = self.run_case(0.02, 0.0, task1_differs=True)
        self.assertEqual(r["registered_verdict"].get("label"), "INCOMPLETE_OR_FAILED")

    def test_running_refused(self):
        with self.assertRaises(SystemExit):
            self.run_case(0.02, 0.0, status="RUNNING")

    def test_bootstrap_rule(self):
        idx = V.boot_index(10)
        self.assertEqual(idx.shape, (20000, 10))
        self.assertTrue(np.array_equal(idx, np.random.default_rng(1018).integers(0, 10, size=(20000, 10))))
        d = np.arange(10) / 1000
        iv = V.interval(d, idx, V.Q_MAIN)
        self.assertAlmostEqual(iv["mean"], d.mean())
        means = d[idx].mean(1)
        self.assertEqual(iv["interval"], [float(np.quantile(means, .0125)), float(np.quantile(means, .9875))])
        # an end equal to the threshold does not meet it
        self.assertEqual(V.label_d1({"interval": [0.0, 1.0]}), "INCONCLUSIVE")
        self.assertEqual(V.label_d2({"interval": [-0.005, 1.0]}), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
