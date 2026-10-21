"""CPU-only independent regressions for the preregistered CH′ response experiment.

Run: python -m unittest discover -s tests -p test_cnn_response_chb_1021.py -v
No CIFAR files, archived checkpoints, CUDA contexts, or intervention data are read.
"""
from __future__ import annotations

import copy
import io
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnn_response_chb_1021 as M
from src import cnn_center_feat_1011 as CF
from src import sna_cnn_cause_1009 as E


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        torch.isnan(a) if a.is_floating_point() else torch.zeros_like(a, dtype=torch.bool),
        torch.isnan(b) if b.is_floating_point() else torch.zeros_like(b, dtype=torch.bool)
    ) and bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))))


def make_network(seed=10, runs=2):
    P = E.stack_init(list(range(seed, seed + runs)), torch.device("cpu"))
    act = M.ResponseDoors(runs, "feat", torch.device("cpu"))
    g = torch.Generator().manual_seed(303 + seed)
    with torch.no_grad():
        for m in act.m:
            m.copy_(torch.rand(m.shape, generator=g) * .05)
    return P, act


def fake_data(n=1200):
    """Full production architecture with synthetic images and frozen source states."""
    sources = {}
    for arm, seed in (("CHf", 10), ("CHf_B", 20)):
        P, act = make_network(seed, runs=1)
        sources[arm] = dict(P=[p.detach().clone() for p in P],
                            m=[torch.zeros_like(p) for p in P],
                            v=[torch.zeros_like(p) for p in P], tc=900000.,
                            act=act.state(), bad=torch.zeros(1, dtype=torch.bool),
                            g_lab={10: torch.Generator().manual_seed(1021).get_state()},
                            g_batch={10: torch.Generator().manual_seed(1022).get_state()},
                            kappa=.2, enabled=arm == "CHf_B")
    g = torch.Generator().manual_seed(1023)
    X = torch.randn(1, n, 3, 32, 32, generator=g) * .2
    return SimpleNamespace(seeds=[10], device=torch.device("cpu"), sources=sources,
                           cifar=None, inputs=lambda _: X, Xcenter=X,
                           identity={"synthetic": True},
                           ids=torch.arange(n)[None, :])


@torch.no_grad()
def independent_projection(P, xb, act, refs, artificial=True):
    """Separate sequential implementation; direct ReLU difference is safe here."""
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    widths, activities = [], []
    for l in range(4):
        w, b = P[2*l:2*l+2]
        if l < 2:
            q = F.conv2d(x, w.reshape(R * 16, -1, 5, 5), padding=2, groups=R)
            layout = q.reshape(B, R, 16, -1).permute(1, 0, 2, 3).double()
            width = layout.var(1, correction=0).mean(-1).sqrt()
        else:
            q = torch.bmm(x, w.transpose(1, 2))
            width = q.double().var(1, correction=0).sqrt()
        cap = .2 * width
        b.copy_(torch.where(b.double().abs() > cap, b.double().clamp(-cap, cap).to(b.dtype), b))
        z = q + (b.reshape(1, R*16, 1, 1) if l < 2 else b[:, None, :])
        z0, zd = refs[0][l], refs[1][l]
        a = z.clamp(min=0)
        if artificial:
            a = z0.clamp(min=0) + ((zd + (z-z0)).clamp(min=0) - zd.clamp(min=0))
        widths.append(width)
        activities.append(a)
        if l < 2:
            x = F.max_pool2d(a, 2) - act.sub_map(l, act.m)
            if l == 1:
                x = x.reshape(B, R, 1024).transpose(0, 1)
        else:
            x = a - act.m[l][:, None, :]
    return widths, activities


class SyntheticResponseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.P, self.act = make_network()
        self.PD, self.actD = make_network(30)
        self.x = torch.randn(2, 4, 3, 32, 32,
                             generator=torch.Generator().manual_seed(912)) * .2
        self.field = M.Field(self.P, self.act, self.PD, self.actD)

    def test_stable_difference_large_positive_and_exact_zero_derivative(self):
        donor = torch.tensor([1e20], dtype=torch.float32)
        delta = torch.tensor([1.], requires_grad=True)
        naive = (donor + delta).clamp(min=0) - donor.clamp(min=0)
        self.assertEqual(naive.item(), 0.)
        stable = M.stable_relu_diff(donor, delta)
        self.assertEqual(stable.item(), 1.)
        self.assertEqual(torch.autograd.grad(stable.sum(), delta)[0].item(), 1.)
        donor = torch.tensor([-3., -1., 0., 1., 3.], dtype=torch.float64)
        delta = torch.tensor([1., 2., 0., -1., -4.], dtype=torch.float64, requires_grad=True)
        result = M.stable_relu_diff(donor, delta)
        torch.testing.assert_close(result, torch.tensor([0., 1., 0., -1., -3.], dtype=torch.float64), rtol=0, atol=0)
        native_input = (donor + delta).detach().requires_grad_()
        expected = torch.autograd.grad(native_input.clamp(min=0).sum(), native_input)[0]
        torch.testing.assert_close(torch.autograd.grad(result.sum(), delta)[0], expected, rtol=0, atol=0)
        self.assertEqual(expected[2].item(), 1.)
        self.assertEqual(expected[3].item(), 1.)

    def test_finite_difference_and_frozen_reference_gradients(self):
        z0 = torch.tensor([-.7, .6, 1.2, -.9], dtype=torch.float64, requires_grad=True)
        zd = torch.tensor([-.8, .9, -.2, .4], dtype=torch.float64, requires_grad=True)
        z = (z0.detach() + torch.tensor([.1, -.2, .5, -.7], dtype=torch.float64)).requires_grad_()
        a, ze, derivative = self.field.apply(z, z0, zd)
        g = torch.autograd.grad(a.sum(), (z, z0, zd), allow_unused=True)
        self.assertIsNone(g[1])
        self.assertIsNone(g[2])
        eps = 1e-6
        finite = (self.field.apply(z.detach()+eps, z0, zd)[0] - self.field.apply(z.detach()-eps, z0, zd)[0]) / (2*eps)
        torch.testing.assert_close(g[0], finite, atol=2e-10, rtol=0)
        torch.testing.assert_close(g[0], (ze >= 0).to(z.dtype), atol=0, rtol=0)
        torch.testing.assert_close(derivative, g[0], atol=0, rtol=0)

    def test_self_field_reconstructs_large_scale_changes_and_zero_gate(self):
        reference = torch.tensor([1e8, -1e8, 1e6, -1e6, 1e8, -1e8], dtype=torch.float32)
        current = torch.tensor([1.5, 2.5, 1000000.125, -999999.875, 0., 0.], requires_grad=True)
        activity, effective, derivative = self.field.apply(current, reference, reference)
        native = current.clamp(min=0)
        self.assertTrue(torch.equal(activity, native))
        self.assertTrue(torch.equal(effective, current.to(effective.dtype)))
        native_gradient = torch.autograd.grad(native.sum(), current)[0]
        self.assertTrue(torch.equal(torch.autograd.grad(activity.sum(), current)[0], native_gradient))
        self.assertTrue(torch.equal(derivative, native_gradient.to(derivative.dtype)))

    def test_all_four_initial_activities_inputs_pool_winners_logits_and_loss(self):
        natural, natural_inputs = CF.forward(self.P, self.x, self.act, with_inputs=True)
        out, ze, de, inputs = M.forward(self.P, self.x, self.act, self.field, with_inputs=True)
        for got, expected in zip((*out, *inputs), (*natural, *natural_inputs)):
            self.assertTrue(equal(got, expected), f"initial mismatch {tuple(got.shape)}")
        for l in (0, 1):
            actual_winners = F.max_pool2d(out[2*l+1], 2, return_indices=True)[1]
            expected_winners = F.max_pool2d(natural[2*l+1], 2, return_indices=True)[1]
            self.assertTrue(torch.equal(actual_winners, expected_winners))
        y = torch.arange(8).remainder(10)
        self.assertTrue(torch.equal(F.cross_entropy(out[8].reshape(-1, 10), y),
                                    F.cross_entropy(natural[8].reshape(-1, 10), y)))
        donor = CF.forward(self.PD, self.x, self.actD)
        for l in range(4):
            self.assertTrue(torch.equal(ze[l], donor[2*l]))
            self.assertTrue(torch.equal(de[l], (donor[2*l] >= 0).to(de[l].dtype)))

    def test_self_field_initial_parameter_gradient_matches_natural(self):
        field = M.Field(self.P, self.act, self.P, self.act)
        out = M.forward(self.P, self.x, self.act, field)[0]
        natural = CF.forward(self.P, self.x, self.act)
        y = torch.arange(8).remainder(10)
        g = torch.autograd.grad(F.cross_entropy(out[8].reshape(-1, 10), y), self.P)
        gn = torch.autograd.grad(F.cross_entropy(natural[8].reshape(-1, 10), y), self.P)
        for a, b in zip(g, gn):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_frozen_field_does_not_alias_live_parameters_or_ema(self):
        refs = self.field.anchors(self.x)
        before = M.tree_hash(refs)
        self.assertFalse(any(t.requires_grad for row in refs for t in row))
        with torch.no_grad():
            self.P[0].add_(.1)
            self.PD[4].mul_(2.)
            self.act.m[0].add_(.5)
            self.actD.m[2].sub_(.5)
        self.assertEqual(M.tree_hash(self.field.anchors(self.x)), before)

    def test_image_channel_and_layer_reference_mutations_are_detected(self):
        refs = self.field.anchors(self.x)
        z0, zd = refs
        mutations = (
            (0, zd[0].flip(0), "image ID"),
            (0, zd[0].reshape(4, 2, 16, 32, 32).roll(1, 2).reshape_as(zd[0]), "channel ID"),
            (2, zd[3], "layer ID"),
        )
        for l, wrong, label in mutations:
            z = z0[l].detach().requires_grad_()
            correct = self.field.apply(z, z0[l], zd[l])[0]
            mutated = self.field.apply(z, z0[l], wrong)[0]
            # At the anchor, values intentionally cannot detect a donor-index bug.
            self.assertTrue(torch.equal(correct, mutated), label)
            good_grad = torch.autograd.grad(correct.sum(), z, retain_graph=True)[0]
            bad_grad = torch.autograd.grad(mutated.sum(), z)[0]
            self.assertFalse(torch.equal(good_grad, bad_grad), label)
        wrong_receiver = ([q.flip(0) if l < 2 else q.flip(1) for l, q in enumerate(z0)], zd)
        altered = M.forward(self.P, self.x, self.act, self.field, refs=wrong_receiver)[0]
        correct = M.forward(self.P, self.x, self.act, self.field, refs=refs)[0]
        self.assertFalse(torch.equal(altered[8], correct[8]))

    def test_cached_references_preserve_seed_image_channel_and_layer_ids(self):
        X = torch.randn(2, 48, 3, 32, 32,
                        generator=torch.Generator().manual_seed(633)) * .2
        self.field.build_cache(X, torch.arange(48)[None, :].expand(2, -1), chunk=16)
        ids = torch.tensor([[31, 0, 47, 15, 16, 1, 31, 35, 5, 20, 45, 22, 4, 17, 44, 18],
                            [46, 17, 2, 33, 19, 45, 4, 17, 26, 13, 8, 39, 40, 6, 47, 0]])
        xb = X[torch.arange(2)[:, None], ids]
        gathered = self.field.refs_ids(ids)
        independent = self.field.anchors(xb)
        for kind in range(2):
            for l in range(4):
                self.assertTrue(torch.equal(gathered[kind][l], independent[kind][l]), (kind, l))
        # Values alone would miss donor misindexing, so inspect effective z/gates.
        _, ze, de = M.forward(self.P, xb, self.act, self.field, refs=gathered)
        for l in range(4):
            self.assertTrue(torch.equal(ze[l], independent[1][l]))
            self.assertTrue(torch.equal(de[l], (independent[1][l] >= 0).to(de[l].dtype)))
        correct = gathered[1][0].clone()
        self.field.cache[1][0][1, ids[1, 0], 3, 2] += 9.
        corrupted = self.field.refs_ids(ids)[1][0]
        self.assertFalse(torch.equal(correct, corrupted))

    def test_biasless_width_uses_image_axis_then_position_average(self):
        image = np.arange(3, dtype=np.float64)[:, None, None, None, None]
        scale = np.arange(1, 3, dtype=np.float64)[None, :, None, None, None] * np.arange(1, 17, dtype=np.float64)[None, None, :, None, None]
        offset = 100*np.arange(2)[None, None, None, :, None] + 1000*np.arange(3)[None, None, None, None, :]
        q = image*scale + offset
        expected = np.empty((2, 16))
        for r in range(2):
            for c in range(16):
                variances = [np.var(q[:, r, c, h, w], ddof=0) for h in range(2) for w in range(3)]
                expected[r, c] = math.sqrt(sum(variances)/len(variances))
        width = M.image_width(torch.from_numpy(q.reshape(3, 32, 2, 3)), True, 2)
        np.testing.assert_allclose(width.numpy(), expected, rtol=0, atol=1e-12)
        wrong = torch.from_numpy(q).var((0, 3, 4), correction=0).sqrt()
        self.assertFalse(torch.allclose(width, wrong))

    def test_ema_uses_actual_artificial_activity_while_v_records_raw_z(self):
        out = CF.forward(self.P, self.x, self.act)
        zs = [out[2*l].detach() for l in range(4)]
        activities = [out[2*l+1].detach() - 1.7 for l in range(4)]
        old_m = [q.clone() for q in self.act.m]
        old_v = [q.clone() for q in self.act.V]
        expected = []
        for l, a in enumerate(activities):
            bm = F.max_pool2d(a, 2).mean(0).reshape(2, -1) if l < 2 else a.mean(1)
            expected.append(old_m[l].mul(.99).add(bm, alpha=.01))
        self.act.update_actual(zs, activities)
        for l in range(4):
            self.assertTrue(torch.equal(self.act.m[l], expected[l]))
            v = old_v[l].mul(.99).add(.01*self.act.batch_var(zs[l], l))
            self.assertTrue(torch.equal(self.act.V[l], v))
            self.assertTrue(bool((self.act.m[l] < 0).any()))
        self.assertTrue(torch.equal(self.act.hupd, torch.ones_like(self.act.hupd)))

    def test_bias_projection_independent_sequential_artificial_oracle(self):
        refs = self.field.anchors(self.x)
        with torch.no_grad():
            for l in range(4):
                self.P[2*l+1].add_(.4)
        oracle = [q.detach().clone() for q in self.P]
        wrong = [q.detach().clone() for q in self.P]
        frozen = M.tree_hash(self.act.state())
        weights = [q.detach().clone() for q in self.P]
        stale_inputs = M.forward(self.P, self.x, self.act, self.field, refs=refs, with_inputs=True)[3]
        widths, _ = independent_projection(oracle, self.x, self.act, refs)
        wrong_widths, _ = independent_projection(wrong, self.x, self.act, refs, artificial=False)
        trace = M.project_bias(self.P, self.x, self.act, self.field, refs=refs)
        for l in range(4):
            torch.testing.assert_close(self.P[2*l+1], oracle[2*l+1], atol=3e-7, rtol=2e-6)
            torch.testing.assert_close(trace[l]["s"], widths[l], atol=3e-7, rtol=2e-6)
            self.assertTrue(torch.equal(self.P[2*l], weights[2*l]))
        self.assertTrue(any(not torch.allclose(widths[l], wrong_widths[l]) for l in (1, 2, 3)))
        stale_widths = []
        for l in (1, 2, 3):
            q = (F.conv2d(stale_inputs[l], weights[2*l].reshape(32, -1, 5, 5), padding=2, groups=2)
                 if l < 2 else torch.bmm(stale_inputs[l], weights[2*l].transpose(1, 2)))
            stale_widths.append(M.image_width(q, l < 2, 2))
        self.assertTrue(any(not torch.allclose(widths[l], q) for l, q in zip((1, 2, 3), stale_widths)))
        old_ema = copy.deepcopy(self.act)
        for m in old_ema.m:
            m.add_(.3)
        old_ema_params = [q.detach().clone() for q in weights]
        old_ema_widths, _ = independent_projection(old_ema_params, self.x, old_ema, refs)
        self.assertTrue(any(not torch.allclose(widths[l], old_ema_widths[l]) for l in (2, 3)))
        self.assertTrue(torch.equal(self.P[8], weights[8]))
        self.assertTrue(torch.equal(self.P[9], weights[9]))
        self.assertEqual(M.tree_hash(self.act.state()), frozen)

    def test_complete_save_load_replays_state_and_random_streams(self):
        data = fake_data()
        engine = M.Engine(data, "CHB_swap", graph=False)
        engine.new_labels()
        engine.new_order()
        engine.step(0)
        saved = engine.snapshot()
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        engine.step(1)
        engine.new_labels()
        engine.new_order()
        expected = engine.snapshot()
        replay = M.Engine(data, "CHB_swap", graph=False)
        replay.restore(loaded)
        self.assertEqual(M.tree_hash(replay.snapshot()), M.tree_hash(saved))
        replay.step(1)
        replay.new_labels()
        replay.new_order()
        self.assertEqual(M.tree_hash(replay.snapshot()), M.tree_hash(expected))

    def test_natural_self_complete_state_after_one_and_75_cpu_steps(self):
        from analysis.cnn_response_chb_1021 import preflight as checks
        data = fake_data()
        for prefix in ("CH", "CHB"):
            actual = M.Engine(data, prefix+"_self", graph=False)
            reference = M.Engine(data, prefix+"_nat", graph=False)
            checks.prepare(actual)
            checks.prepare(reference)
            for j in range(75):
                actual.step(j)
                checks.original_natural_step(reference, j)
                if j in (0, 74):
                    result = checks.comparison(checks.natural_state(actual), checks.natural_state(reference))
                    self.assertTrue(result["passed"], (prefix, j+1, [q for q in result["leaves"] if not q["passed"]]))

    def test_verifier_detects_complete_state_mutations_and_strict_threshold(self):
        from analysis.cnn_response_chb_1021 import preflight as checks
        reference = dict(P=[torch.tensor([0., 1.], dtype=torch.float64)],
                         state={"act": {"fixA": [torch.tensor([float("nan")])] }},
                         g_batch={10: torch.tensor([1, 2, 3], dtype=torch.uint8)})
        self.assertTrue(checks.comparison(reference, reference, exact=True)["passed"])
        changed = copy.deepcopy(reference)
        changed["P"][0][0] = 1e-4
        self.assertFalse(checks.comparison(changed, reference)["passed"])
        changed["P"][0][0] = .9999e-4
        self.assertTrue(checks.comparison(changed, reference)["passed"])
        changed["g_batch"][10][1] = 9
        self.assertFalse(checks.comparison(changed, reference)["passed"])
        self.assertFalse(checks.comparison(dict(P=reference["P"]), reference)["passed"])

    def test_verifier_rejects_identical_divergence_and_nonfinite_outputs(self):
        from analysis.cnn_response_chb_1021 import preflight as checks
        for key, value in (("P", torch.tensor([float("nan")])),
                           ("logits", torch.tensor([float("nan")])),
                           ("ce_steps", torch.tensor([float("inf")])),
                           ("bad", torch.tensor([True]))):
            invalid = {key: value}
            self.assertFalse(checks.comparison(invalid, invalid)["passed"], key)
            self.assertFalse(checks.comparison(invalid, invalid, exact=True)["passed"], key)

    def test_save_load_refuses_changed_frozen_field_identity(self):
        engine = M.Engine(fake_data(n=16), "CH_swap", graph=False)
        saved = engine.snapshot()
        saved["field"]["PD"][0].add_(.01)
        with self.assertRaisesRegex(RuntimeError, "[Ff]rozen field"):
            engine.restore(saved)

    def test_diagnostics_leave_parameters_adam_ema_fields_and_rng_immutable(self):
        engine = M.Engine(fake_data(), "CH_swap", graph=False)
        engine.new_labels()
        engine.new_order()
        engine.step(0)
        before = M.tree_hash(engine.snapshot())
        before_cache = M.tree_hash(engine.field.cache)
        cpu_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())
        rows, units, arrays = M.diagnose(engine, update=1, chunk=20)
        self.assertEqual(len(rows), 1)
        self.assertTrue(units)
        self.assertTrue(arrays)
        self.assertEqual(M.tree_hash(engine.snapshot()), before)
        self.assertEqual(M.tree_hash(engine.field.cache), before_cache)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(np.random.get_state()[0], numpy_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
