"""CPU checks: paired arithmetic, bad data rejection, original engine streams/resume."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.ch_chb_performance_1011 import common as C
from analysis.ch_chb_performance_1011 import launch as L
from analysis.ch_chb_performance_1011 import report as R


def fixture():
    out = {a: [] for a in C.ARMS}
    for a in C.ARMS:
        for index, s in enumerate(C.SEEDS):
            for t in range(1, 201):
                late = .3 + index / 1000 + ((index + 1) / 100 if a == "CHB" else 0)
                row = dict(arm=a, cond="raw", seed=s, task=t, slot=index, iv="none", lr=C.LR,
                           online_acc=.9 if t <= 50 else late, memo_acc=.98)
                row.update({k: .2 for k in C.DIAGNOSTICS})
                out[a].append(row)
    return out


class ArithmeticTests(unittest.TestCase):
    def test_late_difference_and_drop_difference_can_have_opposite_directions(self):
        data = fixture()
        for arm, early, late in (("CH", .95, .85), ("CHB", .60, .59)):
            for row in data[arm]:
                row["online_acc"] = early if row["task"] <= 50 else late
        result, _ = R.aggregate(data)
        self.assertAlmostEqual(result["primary"]["mean"], -.26)
        self.assertEqual(result["primary"]["direction"], "CHB_lower")
        secondary = result["secondary_drop_difference"]
        self.assertAlmostEqual(secondary["mean"], .09)
        self.assertEqual(secondary["contrast"], "drop_CH_minus_CHB")
        self.assertEqual(secondary["direction"], "CH_drops_more")
        reverse = R.paired_ci([-.09] * 10, contrast="drop_CH_minus_CHB")
        self.assertEqual(reverse["direction"], "CHB_drops_more")

    def test_known_paired_result_and_seed_order(self):
        data = fixture()
        result, _ = R.aggregate(data)
        self.assertAlmostEqual(result["primary"]["mean"], .055, places=14)
        expected_sd = math.sqrt(9.166666666666667) / 100
        self.assertAlmostEqual(result["primary"]["sample_sd"], expected_sd, places=14)
        half = 2.2621571628540993 * expected_sd / math.sqrt(10)
        np.testing.assert_allclose(result["primary"]["ci95"], [.055 - half, .055 + half], atol=1e-14)
        self.assertEqual(result["primary"]["positive"], 10)
        # Shuffle each arm independently: pairing follows seed identity, never row order.
        np.random.default_rng(123).shuffle(data["CH"])
        np.random.default_rng(456).shuffle(data["CHB"])
        self.assertEqual(R.aggregate(data)[0], result)

    def test_zero_variance_and_signs(self):
        self.assertEqual(R.paired_ci([.125] * 10)["ci95"], [.125, .125])
        self.assertEqual(R.paired_ci([-.125] * 10)["direction"], "CHB_lower")
        self.assertEqual(R.paired_ci([0.] * 10)["direction"], "uncertain")
        self.assertEqual(R.paired_ci([-.1, .1] * 5)["negative"], 5)
        for bad in ([1.] * 9, [float("nan")] * 10, [float("inf")] * 10):
            with self.assertRaises(ValueError):
                R.paired_ci(bad)

    def test_missing_duplicate_and_invalid_values_rejected(self):
        rows = fixture()["CH"]
        mutants = []
        mutants.append(rows[:-1])
        mutants.append(rows[:-1] + [rows[0]])
        for key, value in (("seed", 0), ("seed", 10.5), ("task", 201), ("task", 1.5),
                           ("slot", 9), ("cond", "std"), ("arm", "CHB"), ("iv", "other"),
                           ("online_acc", "nan"), ("online_acc", "inf"), ("online_acc", -1),
                           ("online_acc", 1.001), ("memo_acc", 1.1), ("zsd_l1", -1),
                           ("gate_zero_frac_l2", 1.5), ("w_norm_l2", float("inf"))):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            mutants.append(changed)
        changed = copy.deepcopy(rows)
        del changed[0]["online_acc"]
        mutants.append(changed)
        for index, mutant in enumerate(mutants):
            with self.subTest(mutant=index), self.assertRaises(ValueError):
                C.validate_rows(mutant, "CH")

    def test_plots_and_json_finite(self):
        result, valid = R.aggregate(fixture())
        json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as temp:
            R.draw(Path(temp), valid, result)
            for name in ("performance.png", "paired_performance.png", "diagnostics.png"):
                self.assertGreater((Path(temp) / name).stat().st_size, 1000)


class LauncherTests(unittest.TestCase):
    def test_manifest_fresh_output_and_explicit_same_run_resume(self):
        wanted = dict(config=C.config(), git_hash="a" * 40, source_sha256={"file": "hash"})
        with tempfile.TemporaryDirectory() as temp, patch.object(C, "OUT", Path(temp)), \
                patch.object(L, "manifest_for", return_value=wanted.copy()):
            args = SimpleNamespace(resume=False)
            first = L.prepare_manifest(args, "a" * 40)
            self.assertEqual(first["config"], C.config())
            with self.assertRaises(ValueError):
                L.prepare_manifest(args, "a" * 40)
            args.resume = True
            self.assertEqual(L.prepare_manifest(args, "a" * 40), first)
            with patch.object(L, "manifest_for", return_value=dict(wanted, git_hash="b" * 40)):
                with self.assertRaises(ValueError):
                    L.prepare_manifest(args, "b" * 40)
        with tempfile.TemporaryDirectory() as temp, patch.object(C, "OUT", Path(temp)), \
                patch.object(L, "manifest_for", return_value=wanted.copy()):
            (Path(temp) / "CH").mkdir()
            with self.assertRaises(ValueError):
                L.prepare_manifest(SimpleNamespace(resume=False), "a" * 40)

    def test_gpu_process_classification(self):
        self.assertTrue(L.is_python_job("/path/.venv/bin/python\0-m\0src.train", "/path/.venv/bin/python"))
        self.assertTrue(L.is_python_job("python3.11 train.py", "/usr/bin/python3.11"))
        self.assertFalse(L.is_python_job("/opt/chrome/chrome --type=gpu-process", "chrome"))
        self.assertFalse(L.is_python_job("/usr/libexec/gnome-remote-desktop-daemon", "gnome-remote-desktop-daemon"))
        self.assertFalse(L.is_python_job("/usr/bin/Xorg", "/usr/bin/Xorg"))

    def test_polling_and_stop_guard_without_gpu(self):
        with self.assertRaises(ValueError):
            L.await_gpu(None, 299)
        with tempfile.TemporaryDirectory() as temp, patch.object(C, "OUT", Path(temp)):
            (Path(temp) / "STOP").touch()
            with patch.object(L, "compute_jobs", side_effect=AssertionError("must not query GPU")):
                with self.assertRaises(InterruptedError):
                    L.await_gpu(None, 300)

    def test_gpu_wait_releases_lock_and_obeys_interval(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(C, "OUT", Path(temp)), \
                patch.object(L, "compute_jobs", side_effect=[[dict(pid=123, command="python train.py")], []]), \
                patch.object(L.time, "sleep") as sleep:
            with (Path(temp) / "lock").open("a") as lock:
                L.await_gpu(lock, 300)
            sleep.assert_called_once_with(300)

    def test_checkpoint_metadata_rejected_before_loading_tensors(self):
        bad = dict(meta=C.expected_meta("CH", "a" * 40))
        bad["meta"]["seeds"] = list(range(10))
        with self.assertRaises(ValueError):
            C.validate_checkpoint(bad, "CH", "a" * 40)


class EngineTests(unittest.TestCase):
    def test_production_checkpoint_shapes_and_state_validation(self):
        import torch
        from src import relu_doors_0919 as E
        torch.set_num_threads(2)
        init = [E.H.init_params(s, torch.device("cpu"), E.DIMS) for s in C.SEEDS]
        params = [torch.stack([p[index].detach() for p in init]) for index in range(6)]
        st = dict(meta=C.expected_meta("CH", "a" * 40), t=1, tc=30000,
                  P=params, m=[torch.zeros_like(p) for p in params], v=[torch.zeros_like(p) for p in params],
                  alive=torch.ones(10, dtype=torch.bool), diverged=[],
                  rows=[r for r in fixture()["CH"] if r["task"] == 1],
                  g_lab={s: E.H.stream("rlc_labels", s).get_state() for s in C.SEEDS},
                  g_batch={s: E.H.stream("rlc_batch", s).get_state() for s in C.SEEDS},
                  act_state=dict(m=[torch.zeros((10, 100)), torch.zeros((10, 100))]))
        self.assertEqual(C.validate_checkpoint(st, "CH", "a" * 40), 1)
        for key, value in (("tc", 29999), ("t", 201), ("diverged", [dict(seed=10)]),
                           ("act_state", dict(m=[])), ("g_lab", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_checkpoint(dict(st, **{key: value}), "CH", "a" * 40)
        st["P"][0] = st["P"][0].transpose(1, 2)
        with self.assertRaises(ValueError):
            C.validate_checkpoint(st, "CH", "a" * 40)

    def test_original_engine_stream_pairing_bias_and_cpu_resume(self):
        import torch
        from src import relu_doors_0919 as E
        torch.set_num_threads(2)
        device = E.H.setup("cpu")
        # Synthetic images for implementation validation only, distinct from the real run.
        class FakeCifar:
            sha256 = {"synthetic_cpu_check": "not_cifar_results"}
            def __init__(self):
                self.x = torch.rand((1200, 3072), generator=torch.Generator().manual_seed(872431))
            def images(self, idx, target):
                return self.x.to(target).clone()
        cifar = FakeCifar()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            debug, final = {}, {}
            for arm in C.ARMS:
                debug[arm] = {}
                E.run(arm, [100001], ["raw"], 2, 1, device, root / arm, lam=C.LAM[arm],
                      beta=C.BETA, debug=debug[arm], cifar=cifar, snapshots=False,
                      checkpoint=True, graph=False, progress=lambda _: None)
                final[arm] = torch.load(root / arm / "ckpt.pt", map_location="cpu", weights_only=False)
            for left, right in zip(debug["CH"]["init"], debug["CHB"]["init"]):
                self.assertTrue(torch.equal(left, right))
            self.assertTrue(bool(debug["CHB"]["init"][1].abs().sum() > 0))
            self.assertTrue(torch.equal(final["CHB"]["P"][1], torch.zeros_like(final["CHB"]["P"][1])))
            for field in ("labels", "orders"):
                for left, right in zip(debug["CH"][field], debug["CHB"][field]):
                    self.assertTrue(torch.equal(left, right), field)
            for tasks, resume in ((1, False), (2, True)):
                E.run("CHB", [100001], ["raw"], tasks, 1, device, root / "resume", lam=C.LAM["CHB"],
                      beta=C.BETA, cifar=cifar, snapshots=False, checkpoint=True,
                      resume=resume, graph=False, progress=lambda _: None)
            resumed = torch.load(root / "resume" / "ckpt.pt", map_location="cpu", weights_only=False)
            for key in ("P", "m", "v"):
                for left, right in zip(final["CHB"][key], resumed[key]):
                    self.assertTrue(torch.equal(left, right), key)
            for left, right in zip(final["CHB"]["act_state"]["m"], resumed["act_state"]["m"]):
                self.assertTrue(torch.equal(left, right), "EMA")
            for key in ("g_lab", "g_batch"):
                self.assertTrue(torch.equal(final["CHB"][key][100001], resumed[key][100001]), key)
            self.assertEqual(final["CHB"]["tc"], resumed["tc"])
            for left, right in zip(final["CHB"]["rows"], resumed["rows"]):
                self.assertEqual(left, right)
        params = [torch.ones((1, 2, 2)), torch.ones((1, 2)), torch.ones((1, 2, 2)),
                  torch.ones((1, 2)), torch.ones((1, 2, 2)), torch.ones((1, 2))]
        E.make_act("CHB", C.LAM["CHB"], C.BETA).post_update(params, C.LR)
        self.assertTrue(torch.equal(params[1], torch.zeros_like(params[1])))
        self.assertTrue(torch.equal(params[3], torch.full_like(params[3], 1 - C.LR * C.LAM["CHB"])))
        self.assertTrue(torch.equal(params[5], torch.ones_like(params[5])))


def run_checks():
    start = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=[""]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(all_pass=result.wasSuccessful(), tests_run=result.testsRun,
                  errors=[(str(test), msg) for test, msg in result.errors],
                  failures=[(str(test), msg) for test, msg in result.failures],
                  source_sha256=C.sources(), git_hash=C.git("rev-parse", "HEAD"),
                  device="cpu", synthetic_check_seed=100001, created_at=C.utc(),
                  elapsed_seconds=time.time() - start)
    C.put(C.OUT / "checks" / "cpu.json", record)
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(run_checks())
