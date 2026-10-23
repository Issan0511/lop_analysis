#!/usr/bin/env python3
"""Synthetic tests of verdict.py (no real run is read).

    cd analysis/cifar5p1_mlp_snfix_1019 && python3 -m unittest test_verdict -v
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


def wcsv(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fake(root: Path, off: dict, drop_arm=None):
    rng = np.random.default_rng(3)
    base = {s: 0.60 + 0.02 * rng.standard_normal() for s in range(10)}
    for arm in V.ALL:
        o = off.get(arm, 0.0)
        rows, fresh = [], []
        for s in range(10):
            oo = o(s) if callable(o) else o
            for t in range(1, 31):
                if arm == drop_arm:
                    continue
                late = t >= 21
                acc = base[s] + oo - (0.03 if late else 0.0)
                r = {"arm": arm, "cond": "std", "seed": s, "slot": s, "lr": 0.0001, "task": t, "hard": t % 2,
                     "online_acc": acc, "train_acc": acc + 0.1, "test_acc": acc - 0.05, "dead_frac_l2": 0.0,
                     "zeroout_l2": 0.0, "eff_rank_l2": 40.0, "w_norm_l1": 0.7, "mob_l2": 0.8, "zsd_l1": 3.0, "zsd_l2": 3.0}
                if arm in V.SNAKE:
                    r.update({"alpha_med_l1": 0.2, "alpha_med_l2": 0.2, "two_alpha_W_med_l1": 1.2 + 0.01 * t,
                              "two_alpha_W_med_l2": 1.2})
                rows.append(r)
            fresh.append({"arm": arm, "cond": "std", "seed": s, "lr": 0.0001, "task": 29, "fresh_online_acc": 0.7,
                          "continual_online_acc": 0.65, "fresh_gap": 0.05 - (0.1 if arm == "SN3" else 0.0)})
        if not rows:
            continue
        d = root / V.cell(arm)
        wcsv(d / "per_task.csv", rows)
        wcsv(d / "fresh_control.csv", fresh)
        (d / "provenance.json").write_text(json.dumps({"arm": arm, "cond": "std", "lr": 1e-4, "hidden": 100, "n_params": 1,
                                                       "n_tasks": 30, "steps_per_task": 780, "divergences": []}))
    (root / V.cell("SNAfrz1") / "frozen_alpha.json").write_text(json.dumps(
        {"l1": [[0.2] * 100] * 10, "l2": [[0.3] * 100] * 10, "unchanged_at_later_task_starts": True}))


def host_copy(root: Path, host: Path):
    for arm in V.OLD:
        for f in ("per_task.csv", "fresh_control.csv"):
            (host / V.cell(arm)).mkdir(parents=True, exist_ok=True)
            (host / V.cell(arm) / f).write_bytes((root / V.cell(arm) / f).read_bytes())


class Tests(unittest.TestCase):
    def go(self, off, tamper=False, drop_arm=None, tamper_eff=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "cifar5p1_mlp_snfix_1019"
            host = Path(tmp) / "results" / "cifar5p1_mlp_0920"
            fake(root, off, drop_arm)
            if drop_arm is None:
                host_copy(root, host)
            if tamper:
                p = host / V.cell("ELU") / "per_task.csv"
                p.write_bytes(p.read_bytes().replace(b"0.0001", b"0.00010", 1))
            if tamper_eff:
                p = host / V.cell("R") / "per_task.csv"
                p.write_bytes(p.read_bytes().replace(b",40.0,", b",40.00000001,", 1))
            res = V.compute(root)
            self.assertIn("族 N", V.summary_md(res))
            json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            return res

    def test_labels_and_readings(self):
        alt = lambda s: 0.01 * (-1) ** s
        res = self.go({"SNA": 0.05, "SN3": 0.03, "SN06": lambda s: 0.05 + 0.01 * (-1) ** s, "SNAfrz1": 0.06,
                       "KKT1": 0.05, "ELU": 0.02})
        v = {f"{t['a'].split('_')[0]}-{t['b'].split('_')[0]}": t["verdict"] for t in res["family_N"]}
        self.assertEqual(v["SN3-SNA"], "B_WINS")
        self.assertEqual(v["SN06-SNA"], "TIE")
        self.assertEqual(v["SNAfrz1-SNA"], "A_WINS")
        self.assertEqual(v["SN06-ELU"], "A_WINS")
        self.assertEqual(v["SN06-SN3"], "A_WINS")              # +0.02 +- 0.01: positive in every seed
        self.assertEqual(len(res["family_N"]), 10)
        self.assertEqual(res["vs_best"]["leader"], "SNAfrz1")
        self.assertEqual(len(res["vs_best"]["tests"]), 18)
        self.assertTrue(any("SNAfrz1 は SNA に勝った" in s for s in res["readings"]))
        self.assertFalse(any("変わらない" in s for s in res["readings"]))
        self.assertEqual(res["S_reuse_0920"]["label"], "PASS")
        q = res["Q_new_arms"]
        self.assertTrue(q["SN06"]["Q1_drop_pos"] and q["SN06"]["Q2_gap_pos"] and not q["SN06"]["Q3_collapse"])
        self.assertFalse(q["SN3"]["Q2_gap_pos"])
        p = res["predictions_claude"]
        self.assertTrue(p["P1_SN3_vs_SNA_B_WINS"] and p["P2_SN06_vs_SNA_TIE"] and p["P5_leader_is_snake"])
        self.assertFalse(p["P3_SNAfrz1_vs_SNA_TIE"] or p["P4_new_arms_all_Q2"])
        self.assertAlmostEqual(res["two_alpha_W_trace"]["SN06"]["t30"]["two_alpha_W_med_l1"], 1.5)
        self.assertAlmostEqual(res["SNAfrz1_frozen_alpha"]["median_l2"], 0.3)

    def test_tie_reading(self):
        res = self.go({"SNA": 0.05, "SN06": lambda s: 0.05 + 0.01 * (-1) ** s,
                       "SNAfrz1": lambda s: 0.05 - 0.01 * (-1) ** s})
        self.assertTrue(any("変わらない" in s for s in res["readings"]))

    def test_reuse_fail(self):
        res = self.go({}, tamper=True)
        self.assertEqual(res["S_reuse_0920"]["label"], "FAIL")
        self.assertFalse(res["S_reuse_0920"]["per_arm"]["ELU"]["per_task.csv"])
        self.assertTrue(res["S_reuse_0920"]["per_arm"]["ELU"]["fresh_control.csv"])
        self.assertFalse(res["predictions_claude"]["P6_reuse_pass"])

    def test_reuse_eff_rank_only(self):
        res = self.go({}, tamper_eff=True)
        ru = res["S_reuse_0920"]
        self.assertEqual(ru["label"], "FAIL")
        self.assertEqual(ru["trajectory_label"], "PASS")
        self.assertAlmostEqual(ru["eff_rank_max_rel_diff"], 1e-8 / 40.00000001, places=12)
        res = self.go({}, tamper=True)
        self.assertEqual(res["S_reuse_0920"]["trajectory_label"], "FAIL")

    def test_missing_arm(self):
        res = self.go({"SNA": 0.05}, drop_arm="SN3")
        self.assertIn(V.cell("SN3"), res["missing_cells"])
        v = {f"{t['a'].split('_')[0]}-{t['b'].split('_')[0]}": t["verdict"] for t in res["family_N"]}
        self.assertEqual(v["SN3-SNA"], "INCOMPLETE")
        self.assertEqual(len(res["vs_best"]["tests"]), 17)

    def test_unfinished_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "cifar5p1_mlp_snfix_1019"
            fake(root, {})
            (root / V.cell("SN06") / "provenance.json").unlink()
            with self.assertRaises(SystemExit):
                V.compute(root)


if __name__ == "__main__":
    unittest.main()
