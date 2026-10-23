"""Scientific preflight: upstream equivalence, paired randomness and exact resume."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import PairedBatches, TaskImages, task_classes, transform
from .model import ARM_ORDER, MODEL_CONFIG, BattleActivation, activations, make_model, reset_head
from .report import sign_test, summarize
from .run import DATA, RAW, REPO, atomic_json, get_rng, set_rng, sha256
from .upstream.vit import VisionTransformer


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['eager', 'fused', 'compile'], default='eager')
    engine = parser.parse_args().engine
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = 'cuda'
    results = {}
    tiny = dict(MODEL_CONFIG, embed_dim=32, depth=2, n_heads=4, mlp_ratio=2.)
    # The model is checked at full registered size, in training mode including dropout.
    torch.manual_seed(100)
    upstream = VisionTransformer(**MODEL_CONFIG).to(device)
    wrapped = make_model('GELU', 100, device)
    assert_tree_equal(dict(upstream.named_parameters()), dict(wrapped.named_parameters()))
    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.tensor([3, 9], device=device)
    rng = get_rng()
    a = upstream(x)
    F.cross_entropy(a, y).backward()
    set_rng(rng)
    b = wrapped(x)
    F.cross_entropy(b, y).backward()
    assert torch.equal(a, b)
    for p, q in zip(upstream.parameters(), wrapped.parameters()):
        assert torch.equal(p.grad, q.grad)
    results['upstream_full_size_gelu_logits_and_gradients_bit_equal'] = True
    del upstream, wrapped

    initial = None
    for arm in ARM_ORDER:
        m = make_model(arm, 100, device, tiny)
        current = {k: v.detach().cpu().clone() for k, v in m.named_parameters()}
        if initial is None:
            initial = current
        assert_tree_equal(initial, current)
        logits = m(x)
        F.cross_entropy(logits[:, :5], torch.tensor([0, 2], device=device)).backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        assert bool(torch.isfinite(logits).all())
    results['13_arms_same_initial_parameters_and_finite_backward'] = True

    for arm in ('SNA', 'KKA', 'KKA23', 'KKT1'):
        m = BattleActivation(arm, 8, 100, 0).to(device)
        z = torch.randn(3, 5, 8, device=device)
        expected = .99 * torch.ones(8, device=device) + .01 * z.flatten(0, 1).var(0, unbiased=False)
        m(z)
        assert torch.equal(m.V, torch.ones_like(m.V))
        m.update_ema()
        assert torch.equal(m.V, expected)
        m.eval()
        before = m.V.clone()
        a, b = m(z), m(z)
        assert torch.equal(a, b) and torch.equal(m.V, before)
        assert not m.V.requires_grad
    results['ema_uses_batch_and_tokens_eval_frozen'] = True

    m = BattleActivation('RSL', 8, 100, 0).to(device)
    z = torch.randn(3, 5, 8, device=device)
    before = get_rng()
    a, b = m(z), m(z)
    assert not torch.equal(a, b)
    assert_tree_equal(before, get_rng())
    m.eval()
    assert torch.equal(m(z), m(z))
    results['rsl_has_private_rng_and_deterministic_eval'] = True

    # Masked loss creates no gradient in inactive head rows; head reset does not reset optimizer.
    m = make_model('GELU', 100, device, tiny)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    classes = torch.tensor([4, 7, 9, 20, 30], device=device)
    F.cross_entropy(m(x)[:, classes], torch.tensor([0, 2], device=device)).backward()
    head = m.layers['out']
    excluded = torch.ones(200, device=device, dtype=torch.bool)
    excluded[classes] = False
    assert bool((head.weight.grad[excluded] == 0).all())
    opt.step()
    moment = opt.state[head.weight]['exp_avg'].clone()
    reset_head(m)
    assert not bool(head.weight.any()) and not bool(head.bias.any())
    assert torch.equal(moment, opt.state[head.weight]['exp_avg'])
    results['task_mask_and_head_reset_preserve_adam_moments'] = True

    batches = list(PairedBatches(2500, 128, 500, 100, 0))
    assert sum(map(len, batches)) == 62500
    assert batches == list(PairedBatches(2500, 128, 500, 100, 0))
    assert sorted(idx for batch in batches[:20] for idx, _ in batch) == list(range(2500))
    assert batches != list(PairedBatches(2500, 128, 500, 101, 0))
    classes_by_task = task_classes(100)
    assert sorted(sum(classes_by_task, [])) == list(range(200))
    dataset = TaskImages(DATA, classes_by_task[0], 'train')
    before = get_rng()
    a = transform(dataset.images[0], 123)
    assert torch.equal(a, transform(dataset.images[0], 123))
    assert not torch.equal(a, transform(dataset.images[0], 124))
    assert_tree_equal(before, get_rng())
    results['data_pairing_and_500_steps_62500_presentations'] = True

    assert sign_test([1]*5)['p'] == .0625
    assert sign_test([1]*10)['winner'] == 1
    assert sign_test([-1]*10)['winner'] == -1
    assert sign_test([0]*10)['p'] == 1
    with tempfile.TemporaryDirectory() as root:
        try:
            summarize(root, Path(root) / 'report')
            raise AssertionError('partial report accepted')
        except ValueError as error:
            assert 'ranking refused' in str(error)
    results['exact_sign_tests_and_partial_report_guard'] = True

    # Full production architecture + real dataset: uninterrupted versus task-boundary resume.
    # Both an adaptive arm and the stochastic activation must restore exactly.
    smoke_root = RAW / 'preflight' / f'checks_{int(time.time())}'
    for arm in ('KKA', 'RSL'):
        uninterrupted, resumed = smoke_root / arm / 'continuous', smoke_root / arm / 'resumed'
        base = [sys.executable, '-m', 'analysis.joudaki_vit_battle_0919.run', '--arm', arm,
                '--seed', '100', '--tasks', '2', '--steps', '2', '--smoke', '--engine', engine]
        subprocess.run(base + ['--out', str(uninterrupted)], cwd=REPO, check=True)
        subprocess.run(base + ['--out', str(resumed), '--stop-after-task', '1'], cwd=REPO, check=True)
        assert not (resumed / 'done.json').exists()
        subprocess.run(base + ['--out', str(resumed)], cwd=REPO, check=True)
        aa = torch.load(uninterrupted / 'checkpoint.pt', weights_only=False, map_location='cpu')
        bb = torch.load(resumed / 'checkpoint.pt', weights_only=False, map_location='cpu')
        for key in ('config', 'model', 'optimizer', 'rng', 'diagnostics'):
            assert_tree_equal(aa[key], bb[key])
        for arow, brow in zip(aa['rows'], bb['rows']):
            for key in arow:
                if key not in ('seconds', 'train_seconds'):
                    assert arow[key] == brow[key], (key, arow[key], brow[key])
        for task in (0, 1, 2):
            a, b = np.load(uninterrupted / f'preact_t{task:02d}.npz'), np.load(resumed / f'preact_t{task:02d}.npz')
            assert a.files == b.files
            assert all(np.array_equal(a[key], b[key]) for key in a.files)
        results[f'{arm}_real_data_resume_parameters_optimizer_rng_preacts_bit_equal'] = True
    results['all_pass'] = all(results.values())
    results['smoke_root'] = str(smoke_root)
    results['dataset_zip_sha256'] = sha256(DATA.parent / 'tiny-imagenet-200.zip')
    results['engine'] = engine
    atomic_json(REPO / f'results/joudaki_vit_battle_0919/checks_{engine}.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
