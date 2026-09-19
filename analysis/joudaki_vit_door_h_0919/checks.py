"""Scientific preflight: upstream equivalence, paired randomness and exact resume."""
import json
import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import PairedBatches, TaskImages, task_classes, transform
from .model import ARM_ORDER, DOOR_BETA, MODEL_CONFIG, BattleActivation, activations, make_model, reset_head
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


def smoke_command(module, arm, out, seed, tasks, steps, engine, extra=()):
    return [sys.executable, '-m', f'analysis.{module}.run', '--arm', arm, '--seed', str(seed),
            '--tasks', str(tasks), '--steps', str(steps), '--smoke', '--engine', engine,
            '--out', str(out), *extra]


def compare_runs(first, second, keys, drop_suffix=None):
    """Bit equality of two finished smoke runs, over everything the run persists."""
    aa = torch.load(first / 'checkpoint.pt', weights_only=False, map_location='cpu')
    bb = torch.load(second / 'checkpoint.pt', weights_only=False, map_location='cpu')
    for key in keys:
        x, y = aa[key], bb[key]
        if key == 'model' and drop_suffix is not None:
            x = {k: v for k, v in x.items() if not k.endswith(drop_suffix)}
            y = {k: v for k, v in y.items() if not k.endswith(drop_suffix)}
        assert_tree_equal(x, y)
    for arow, brow in zip(aa['rows'], bb['rows']):
        for key in arow:
            # 'arm' names the run, and wall-clock is not part of the trajectory.
            if key not in ('seconds', 'train_seconds', 'arm'):
                assert arow[key] == brow[key], (key, arow[key], brow[key])
    # Strict pattern: diagnose() writes through a 'preact_tNN.npz.tmp.npz' sidecar.
    tasks = sorted(int(p.name[len('preact_t'):-len('.npz')])
                   for p in first.glob('preact_t[0-9][0-9].npz'))
    assert tasks, f'no preactivation snapshots in {first}'
    for task in tasks:
        x = np.load(first / f'preact_t{task:02d}.npz')
        y = np.load(second / f'preact_t{task:02d}.npz')
        assert x.files == y.files
        assert all(np.array_equal(x[key], y[key]) for key in x.files)


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
    results['4_arms_same_initial_parameters_and_finite_backward'] = True

    # S-H, part 1: one exact EMA step on the door, on the same sample axis as V.
    z = torch.randn(40, 7, 8, device=device)
    for arm, raw in (('RH', torch.clamp(z, min=0)), ('GH', F.gelu(z))):
        m = BattleActivation(arm, 8, 100, 0).to(device)
        expected = .01 * raw.flatten(0, 1).mean(0)
        m(z)
        assert torch.equal(m.M, torch.zeros_like(m.M))      # forward must not move M
        m.update_ema()
        assert torch.equal(m.M, expected)
        m.eval()
        before = m.M.clone()
        a, b = m(z), m(z)
        assert torch.equal(a, b) and torch.equal(m.M, before)
        assert not m.M.requires_grad
    results['door_ema_uses_batch_and_tokens_eval_frozen'] = True

    # S-H, part 2: with the door open the per-channel DC of the output goes to zero.
    # The tolerance is arithmetic, not a guessed constant. Three terms, largest first:
    #   1/DOOR_BETA  the EMA is a first-order filter, so each step's rounding is
    #                amplified by its gain at the fixed point (this term is ~all of it);
    #   log2(n)      the mean is a pairwise reduction over n samples;
    #   2            the subtraction and the final mean each round once.
    # Measured on CPU: residual 2.54e-6, of which 2.53e-6 is the EMA fixed point.
    steps = 2000
    raw = torch.clamp(z, min=0)
    m = BattleActivation('RH', 8, 100, 0).to(device)
    for _ in range(steps):
        m(z)
        m.update_ema()
    m.eval()
    target = raw.flatten(0, 1).mean(0)
    samples = raw.flatten(0, 1).shape[0]
    bound = ((2. / DOOR_BETA) + math.log2(samples) + 2.) \
        * torch.finfo(torch.float32).eps * float(raw.abs().max())
    lag = .99 ** steps * float(target.abs().max())
    residual = float(m(z).flatten(0, 1).mean(0).abs().max())
    # Mutation: subtract 2M rather than M. A working device must separate from it.
    mutant = float((raw - 2 * m.M).flatten(0, 1).mean(0).abs().max())
    assert residual <= bound + lag, (residual, bound, lag)
    assert mutant - residual >= 100 * (bound + lag), (mutant, residual, bound, lag)
    results['door_removes_dc_and_mutant_separates'] = True
    results['door_dc_residual'] = residual
    results['door_dc_bound'] = bound + lag
    results['door_dc_mutant'] = mutant

    # S-grad: the door subtracts a constant, so the local Jacobian must be untouched
    # and no gradient may reach M. M is given a nonzero value so the test has teeth.
    zg = torch.randn(3, 5, 1536, device=device)
    cotangent = torch.randn_like(zg)
    gradients = {}
    for arm in ARM_ORDER:
        m = BattleActivation(arm, 1536, 100, 0).to(device)
        if m.M is not None:
            m.M.copy_(torch.randn(1536, device=device))
        leaf = zg.detach().clone().requires_grad_(True)
        m(leaf).backward(cotangent)
        gradients[arm] = leaf.grad.detach().clone()
        assert m.M is None or m.M.grad is None
    assert torch.equal(gradients['R'], gradients['RH'])
    assert torch.equal(gradients['GELU'], gradients['GH'])
    results['door_jacobian_bit_equal_and_M_has_no_grad'] = True

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
    # S-resume: full production architecture on real images, uninterrupted versus a
    # task-boundary resume. M rides in the model state_dict, so the model comparison
    # is what proves the door's EMA is restored rather than restarted at zero.
    for arm in ('RH', 'GH'):
        uninterrupted, resumed = smoke_root / arm / 'continuous', smoke_root / arm / 'resumed'
        base = smoke_command('joudaki_vit_door_h_0919', arm, uninterrupted, 100, 2, 2, engine)
        subprocess.run(base, cwd=REPO, check=True)
        paused = smoke_command('joudaki_vit_door_h_0919', arm, resumed, 100, 2, 2, engine)
        subprocess.run(paused + ['--stop-after-task', '1'], cwd=REPO, check=True)
        assert not (resumed / 'done.json').exists()
        subprocess.run(paused, cwd=REPO, check=True)
        compare_runs(uninterrupted, resumed, ('config', 'model', 'optimizer', 'rng', 'diagnostics'))
        state = torch.load(resumed / 'checkpoint.pt', weights_only=False, map_location='cpu')['model']
        door = [v for k, v in state.items() if k.endswith('.M')]
        assert len(door) == MODEL_CONFIG['depth'] and any(bool(v.any()) for v in door)
        results[f'{arm}_real_data_resume_parameters_optimizer_rng_preacts_M_bit_equal'] = True

    # S-off: hold M at zero and the door arm must reproduce its base arm exactly.
    # a - 0.0 is exact in IEEE754, so this is bit equality, not a tolerance.
    for door, base_arm in (('RH', 'R'), ('GH', 'GELU')):
        off, reference = smoke_root / f'soff_{door}', smoke_root / f'soff_{base_arm}'
        subprocess.run(smoke_command('joudaki_vit_door_h_0919', door, off, 100, 1, 2, engine,
                                     ('--door-frozen',)), cwd=REPO, check=True)
        subprocess.run(smoke_command('joudaki_vit_door_h_0919', base_arm, reference, 100, 1, 2, engine),
                       cwd=REPO, check=True)
        state = torch.load(off / 'checkpoint.pt', weights_only=False, map_location='cpu')['model']
        frozen = [v for k, v in state.items() if k.endswith('.M')]
        assert len(frozen) == MODEL_CONFIG['depth'] and not any(bool(v.any()) for v in frozen)
        compare_runs(off, reference, ('model', 'optimizer', 'rng', 'diagnostics'), drop_suffix='.M')
        results[f'{door}_with_zero_door_bit_equals_{base_arm}'] = True

    # S-nochange: the two extra arms must not move the arms that already exist.
    # The unmodified battle package sits beside this one in the tree, so the same
    # seed and task can be run through both builds and compared byte for byte.
    for arm in ('R', 'GELU'):
        added, before = smoke_root / f'snochange_{arm}_door', smoke_root / f'snochange_{arm}_battle'
        subprocess.run(smoke_command('joudaki_vit_door_h_0919', arm, added, 0, 1, 500, engine),
                       cwd=REPO, check=True)
        subprocess.run(smoke_command('joudaki_vit_battle_0919', arm, before, 0, 1, 500, engine),
                       cwd=REPO, check=True)
        compare_runs(added, before, ('model', 'optimizer', 'rng', 'diagnostics'))
        results[f'{arm}_unchanged_by_adding_door_arms'] = True

    results['all_pass'] = all(v is True for v in results.values() if isinstance(v, bool))
    results['smoke_root'] = str(smoke_root)
    results['dataset_zip_sha256'] = sha256(DATA.parent / 'tiny-imagenet-200.zip')
    results['engine'] = engine
    atomic_json(REPO / f'results/joudaki_vit_door_h_0919/checks_{engine}.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
