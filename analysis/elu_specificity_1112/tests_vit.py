"""Data-free, CPU-only validation of the Study B execution paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from . import vit as V
from analysis.joudaki_vit_battle_0919.model import make_model

torch.set_num_threads(1)


def small_model(seed, ln):
    config = dict(
        img_size=8, patch_size=4, in_channels=3, num_classes=6,
        embed_dim=12, depth=2, n_heads=3, mlp_ratio=2., qkv_bias=True,
        dropout_p=0., attn_drop_rate=0., activation='gelu',
        normalization='layer' if ln else 'none', normalization_affine=True,
    )
    return make_model('R', seed, 'cpu', config)


class ToyBox(V.ViT):
    def __init__(self, activation='R', ln=False):
        self.seed, self.t = 19, 30
        self.source_arm = 'SL' if ln else 'S'
        self.activation, self.engine = 'R', 'eager'
        self.model = small_model(self.seed, ln)
        self.initial = copy.deepcopy(self.model.state_dict())
        self.C = SimpleNamespace(build_optimizer=lambda model: torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.))
        self.opt = self.C.build_optimizer(self.model)
        gen = torch.Generator().manual_seed(1112)
        self.X = torch.randn(196, 3, 8, 8, generator=gen)
        self.old = torch.randint(3, (196,), generator=gen)
        self.new = torch.randint(3, (196,), generator=gen)
        self.rows = torch.tensor([1, 3, 5])
        self.order = [torch.arange(128), torch.arange(128, 196)] * 14
        self.compiled = self.model
        self.n, self.eval_every = 196, 25
        for idx in self.order[:3]:
            self.opt.zero_grad(set_to_none=True)
            F.cross_entropy(self.forward(self.X[idx]), self.old[idx]).backward()
            self.opt.step()
        self.archived = copy.deepcopy(self.model.state_dict())
        self.activation = activation
        V.replace_activations(self.model, activation, self.seed)
        self.source_snapshot = V.R.snapshot(self)
        self.base = copy.deepcopy(self.model.state_dict())


def bank_for(box):
    W, b, rows = box.head()
    return V.ops.geometry_bank(W[rows], b[rows], box.features().mean(0),
                               box.seed, 'vit', 30)


def test_rebuild_and_derivatives():
    for ln in (False, True):
        box = ToyBox('R', ln)
        checks = V.native_rebuild_check(
            box.model, box.X[:17], box.new[:17], box.rows, box.seed)
        assert checks['output_exact'] and checks['all_parameter_gradients_exact']
        original = box.model(box.X[:17]).detach()
        ids = [id(p) for p in box.parameters()]
        V.replace_activations(box.model, 'ELU', box.seed)
        assert [id(p) for p in box.parameters()] == ids
        changed = box.model(box.X[:17])
        assert not torch.equal(original, changed)
        assert V.derivative(torch.tensor([-1., 0., 1.]), 'R').tolist() == [0., 1., 1.]
        for module in V.activations(box.model):
            z = torch.tensor([-20., -2., 0., 2.], requires_grad=True)
            result = module(z)
            grad, = torch.autograd.grad(result.sum(), z)
            assert torch.equal(result, F.elu(z))
            assert torch.allclose(grad, V.derivative(z, 'ELU'), rtol=2e-7, atol=0.)


def test_feature_forward_and_groups():
    for ln in (False, True):
        for activation in ('R', 'ELU'):
            box = ToyBox(activation, ln)
            before = V.R.digest(box.model.state_dict())
            features = V.feature_batch(box.model, box.X[:17])
            out = box.model.layers['out'](features)
            assert torch.equal(out, box.model(box.X[:17]))
            groups = V.parameter_groups(box)
            disjoint = ('trunk', 'final_gamma', 'final_beta', 'head')
            parameters = [id(p) for name in disjoint for p in groups[name]]
            assert len(parameters) == len(set(parameters))
            assert set(parameters) == {id(p) for p in box.parameters()}
            assert {id(p) for p in groups['body']} == {id(p) for p in box.body()}
            assert {id(p) for p in groups['body']} == {
                id(p) for name in ('trunk', 'final_gamma', 'final_beta') for p in groups[name]}
            assert bool(groups['final_gamma']) == ln
            assert bool(groups['final_beta']) == ln
            box.features()
            checks = V.engine_check(box)
            assert all(row['maxabs'] == 0 for row in checks.values())
            assert before == V.R.digest(box.model.state_dict())


def test_policy_invariants():
    for activation in ('R', 'ELU'):
        for ln in (False, True):
            box = ToyBox(activation, ln)
            bank = bank_for(box)
            source = V.R.fingerprints(box)
            W, b, rows = box.head()
            original = W.detach().clone()
            original_b = b.detach().clone()
            original_mean_logits = original[rows] @ bank['mu0'] + original_b[rows]
            for arm in V.ARMS:
                fork = V.prepare_arm(box, arm, bank)
                W, b, rows = box.head()
                actual = V.R.fingerprints(box)
                if arm != 'F0':
                    assert actual['body'] == source['body']
                if arm in ('S1', 'S01', 'P1', 'P01', 'Z_keep'):
                    assert actual['body_adam'] == source['body_adam']
                if arm in ('Z_keep', 'Z_clear', 'F0'):
                    assert torch.count_nonzero(W) == torch.count_nonzero(b) == 0
                else:
                    assert torch.allclose(W[rows] @ bank['mu0'] + b[rows],
                                          original_mean_logits, rtol=2e-5, atol=2e-6)
                    inactive = torch.tensor([0, 2, 4])
                    assert torch.equal(W[inactive], original[inactive])
                    assert torch.equal(b[inactive], original_b[inactive])
                if arm == 'S1':
                    assert torch.equal(W, original) and torch.equal(b, original_b)
                if arm == 'P1':
                    assert torch.equal(W[rows].sort(0).values, original[rows].sort(0).values)
                if arm in ('Z_clear', 'F0'):
                    assert not box.opt.state
                if arm == 'F0':
                    for name, p in box.model.named_parameters():
                        if not name.startswith('layers.out.'):
                            assert torch.equal(p, box.initial[name])
                    assert actual['body'] != source['body']
                assert fork['head_adam_empty']


def test_warm_shape_order_restoration():
    box = ToyBox('ELU', True)
    bank = bank_for(box)
    V.prepare_arm(box, 'P01', bank)
    box.engine = 'compile'  # Exercise warmup itself without invoking a compiler.
    before = V.R.snapshot(box)
    fingerprints = V.R.fingerprints(box)
    checks = V.warm_engine(box)
    assert checks == dict(batch_sizes=[128, 68], optimizer_steps=0, state_and_rng_restored=True)
    assert V.R.digest(V.R.snapshot(box)) == V.R.digest(before)
    assert V.R.fingerprints(box) == fingerprints
    assert all(p.grad is None for p in box.parameters())
    box.order = [torch.arange(68), torch.arange(68, 196)]
    try:
        V.warm_engine(box)
    except AssertionError:
        pass
    else:
        raise AssertionError('Reversed warmup shapes were accepted')


def test_diagnostics_and_deterministic_trajectory():
    for activation in ('R', 'ELU'):
        box = ToyBox(activation, True)
        bank = bank_for(box)
        V.prepare_arm(box, 'S01', bank)
        first, tracks = V.trajectory(box, 26)
        V.prepare_arm(box, 'S01', bank)
        second, repeated_tracks = V.trajectory(box, 26)
        assert V.R.digest(first) == V.R.digest(second)
        assert V.R.digest(tracks) == V.R.digest(repeated_tracks)
        assert [r['step'] for r in first] == [0, 1, 5, 25, 26]
        assert [r['step'] for r in tracks] == [1, 25, 26]
        for result in (first[0], first[-2], first[-1]):
            assert len(result['diagnostics']['layers']) == 2
            for layer in result['diagnostics']['layers']:
                assert 0 <= layer['positive_occupancy'] <= 1
                assert 0 <= layer['small_derivative_fraction'] <= 1
                assert 0 <= layer['mean_abs_derivative'] <= 1
        assert tracks[0]['groups']['final_gamma']['n_parameters'] == 12
        for track in tracks:
            groups = track['groups']
            assert groups['body']['n_parameters'] == (groups['trunk']['n_parameters']
                + groups['final_gamma']['n_parameters'] + groups['final_beta']['n_parameters'])
            for metric in ('parameter_norm', 'gradient_norm', 'update_norm'):
                total_squared = sum(groups[name][metric] ** 2
                    for name in ('trunk', 'final_gamma', 'final_beta'))
                assert abs(groups['body'][metric] ** 2 - total_squared) <= 1e-10 * max(1., total_squared)
        assert tracks[0]['groups']['head']['update_norm'] > 0


def test_atomic_finite_and_grid():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.json'
        V.save_json(path, dict(value=1.))
        try:
            V.save_json(path, dict(value=float('nan')))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Nonfinite JSON accepted')
        assert json.loads(path.read_text()) == dict(value=1.)
        assert list(Path(directory).iterdir()) == [path]
    assert {199, 200, 201, 1500} <= set(V.points(1500))
    assert V.points(1) == [0, 1]


def main():
    tests = [
        test_rebuild_and_derivatives,
        test_feature_forward_and_groups,
        test_policy_invariants,
        test_warm_shape_order_restoration,
        test_diagnostics_and_deterministic_trajectory,
        test_atomic_finite_and_grid,
    ]
    for test in tests:
        test()
        print('PASS', test.__name__, flush=True)
    print('PASS Study B synthetic CPU suite:', len(tests), 'groups', flush=True)


if __name__ == '__main__':
    main()

