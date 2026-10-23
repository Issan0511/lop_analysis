#!/usr/bin/env python3
"""Synthetic tests of verdict.py (no real run is read).

    cd analysis/neokunekune_methods_cifar5p1_cnn_1018 && python3 -m unittest test_verdict -v
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

BUNDLE_OF = {"R": "A", "SNA": "A", "SNAfrz1": "A", "NEO": "A", "SN06": "B", "SN3": "B", "KKT1": "B", "ELU": "B",
             "LK": "C", "RSL": "C", "ReDo": "C", "L2I": "C"}


def fake(d: Path, offsets: dict, drop_arm=None, status="COMPLETED", diverge_arm=None, lr="0.0001"):
    rng = np.random.default_rng(5)
    base = {s: 0.70 + 0.02 * rng.standard_normal() for s in V.SEEDS}
    rows = {b: [] for b in V.BUNDLES}
    for i, s in enumerate(V.SEEDS):
        for a in V.ARMS12:
            off = offsets.get(a, 0.0)
            off = off(i) if callable(off) else off
            for t in range(1, 31):
                if a == drop_arm and s == 2009 and t == 30:
                    continue
                val = base[s] + off
                rows[BUNDLE_OF[a]].append({
                    "arm": a, "cfg": f"{a}_lr{lr}", "seed": s, "task": t, "online_acc": val,
                    "first100_online_acc": val - 0.2, "train_acc": val + 0.1, "test_acc": val,
                    "diverged": "True" if (a == diverge_arm and t > 10) else "False", "bad_step": -1,
                    "n_redo_c1": 0, "n_redo_c2": 0, "n_redo_f1": 1 if a == "ReDo" else 0, "n_redo_f2": 0})
    for b, rs in rows.items():
        (d / b).mkdir(parents=True)
        with open(d / b / "per_task.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        (d / b / "provenance.json").write_text(json.dumps({"status": status}))


class Tests(unittest.TestCase):
    def go(self, offsets, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake(d, offsets, **kw)
            res = V.compute(d, None, None)
            self.assertIn("判定", V.summary_md(res))
            return res

    def test_labels(self):
        alt = lambda i: 0.01 * (-1) ** i                      # mean exactly 0 against NEO
        r = self.go({"NEO": 0.0, "SN06": -0.02, "L2I": +0.02, "KKT1": alt, "SNAfrz1": 0.03})
        m = r["families"]["main"]
        self.assertEqual(m["NEO_minus_SN06"]["label"], "BETTER_THAN_SN06")
        self.assertEqual(m["NEO_minus_L2I"]["label"], "WORSE_THAN_L2I")
        self.assertEqual(m["NEO_minus_KKT1"]["label"], "INCONCLUSIVE")
        self.assertTrue(m["NEO_minus_SN06"]["noninferior_0.5pt"])
        self.assertFalse(m["NEO_minus_L2I"]["noninferior_0.5pt"])
        s = r["families"]["secondary"]
        self.assertEqual(s["SNAfrz1_minus_L2I"]["label"], "BETTER_THAN_L2I")

    def test_incomplete(self):
        r = self.go({"SN06": -0.02}, drop_arm="SN06")
        self.assertEqual(r["families"]["main"]["NEO_minus_SN06"]["label"], "INCOMPLETE")
        self.assertNotEqual(r["families"]["main"]["NEO_minus_SN3"]["label"], "INCOMPLETE")
        r = self.go({"RSL": -0.02}, diverge_arm="RSL")
        self.assertEqual(r["families"]["main"]["NEO_minus_RSL"]["label"], "INCOMPLETE")
        r = self.go({}, status="FAILED")
        self.assertTrue(all(v["label"] == "INCOMPLETE" for v in r["families"]["main"].values()))

    def test_running_refused(self):
        with self.assertRaises(SystemExit):
            self.go({}, status="RUNNING")

    def test_bootstrap_and_thresholds(self):
        idx = np.random.default_rng(1018).integers(0, 10, size=(20000, 10))
        d = np.arange(10) / 1000
        iv = V.interval(d, idx)
        means = d[idx].mean(1)
        self.assertEqual(iv["interval"], [float(np.quantile(means, 0.003125)), float(np.quantile(means, 0.996875))])
        self.assertEqual(V.label({"interval": [0.0, 1.0]}, "X"), "INCONCLUSIVE")
        self.assertEqual(V.label({"interval": [-1.0, 0.0]}, "X"), "INCONCLUSIVE")

    def test_nochange(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake(d, {})
            prev = d / "prev.csv"
            rows = [r for r in csv.DictReader(open(d / "A" / "per_task.csv"))]
            with open(prev, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task", "online_acc"])
                w.writeheader()
                for r in rows:
                    w.writerow({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "online_acc": r["online_acc"]})
            self.assertEqual(V.nochange_main(d, prev)["label"], "PASS")
            rows[0]["online_acc"] = "0.123"
            with open(prev, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task", "online_acc"])
                w.writeheader()
                for r in rows:
                    w.writerow({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "online_acc": r["online_acc"]})
            self.assertEqual(V.nochange_main(d, prev)["label"], "FAIL")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake(d, {}, lr="0.001")
            self.assertFalse(V.nochange_main(d, d / "A" / "per_task.csv")["applicable"])


if __name__ == "__main__":
    unittest.main()
