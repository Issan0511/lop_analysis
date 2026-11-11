"""Study A: genuine activation x final-affine-LN histories and seven forks.

No data are loaded and no training runs at import. The no-LN arithmetic is
intentionally the canonical functional stage1 engine, including its one-hot
loss, full-set float32 evaluation and Adam parameter order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from src import stage1_ctrl_1111 as M
from analysis.readout_separation_1111 import ops
from . import common as shared

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs/spec_elu_specificity_1112.md'
DATA = Path('/home/issan/Projects/obsidian-research-data')
OLD_STATES = DATA / 'stage1_ctrl_1111/results/stage1_ctrl_1111/raw'
OLD_SEPARATION = DATA / 'readout_separation_1111/results/readout_separation_1111/raw'
OLD_RESIDUAL = DATA / 'p_residual_loading_1111/results/p_residual_loading_1111/raw'
ARMS = ('S1', 'S01', 'P1', 'P01', 'Z_keep', 'Z_clear', 'F0')
ACTIVATIONS = ('ELU', 'R', 'LR', 'SILU')
LN_EPS = 1e-5


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def digest(tree):
    h = hashlib.sha256()
    def feed(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); feed(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x: feed(v)
        else:
            h.update(repr(x).encode())
    feed(tree)
    return h.hexdigest()


def finite(tree, where='root'):
    if torch.is_tensor(tree):
        if not bool(torch.isfinite(tree).all()): raise FloatingPointError(where)
    elif isinstance(tree, np.ndarray):
        if not np.isfinite(tree).all(): raise FloatingPointError(where)
    elif isinstance(tree, dict):
        for k, v in tree.items(): finite(v, where + '.' + str(k))
    elif isinstance(tree, (tuple, list)):
        for k, v in enumerate(tree): finite(v, where + '.' + str(k))
    elif isinstance(tree, (float, np.floating)) and not np.isfinite(tree):
        raise FloatingPointError(where)


def save_json(path, obj):
    finite(obj)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        name = f.name
        json.dump(obj, f, indent=1, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(name, path)


def save_state(path, state):
    finite(state)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        name = f.name
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(name, path)


def code_hashes():
    paths = [Path(__file__), Path(M.__file__), Path(ops.__file__),
             ROOT/'analysis/loading_bridge_1111/static_probe.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def provenance(sources=()):
    # Keep the shared execution inventory/digest identical in Studies A and B.
    # The local inventory also identifies the narrower source replay dependency.
    result = shared.provenance(sources)
    result.update(git_hash=result['execution_git_hash'], local_code_hashes=code_hashes())
    return result


def points(limit):
    return sorted({0, 1, 5, 25, 50, 100, 199, 200, 201, limit, *range(0, limit+1, 100)} & set(range(limit+1)))


def initial_params(seed, ln, dims=(784, 100, 10)):
    """The four common tensors consume exactly the original private stream."""
    p = M.initial(seed, dims)
    if ln == 'on': p += [torch.ones(dims[1]), torch.zeros(dims[1])]
    elif ln != 'off': raise ValueError(ln)
    return p


class MLP:
    """Functional model; parameter indices0..3 never change with LN."""
    def __init__(self, params, activation, ln):
        if activation not in ACTIVATIONS or ln not in ('off', 'on'):
            raise ValueError((activation, ln))
        if len(params) != (6 if ln == 'on' else 4): raise ValueError('Wrong parameter count')
        self.activation, self.ln = activation, ln
        self.P = M.new_params(params)
        self.opt = M.make_opt('adam', self.P, .001)

    def features(self, x, auxiliary=False):
        z = x @ self.P[0].T + self.P[1]
        a = M.activ(z, self.activation)
        h = F.layer_norm(a, (a.shape[-1],), self.P[4], self.P[5], LN_EPS) if self.ln == 'on' else a
        return (h, z, a) if auxiliary else h

    def forward(self, x):
        return self.features(x) @ self.P[2].T + self.P[3]

    def body(self):
        return self.P[:2] + self.P[4:]

    def head(self):
        return self.P[2:4]

    def snapshot(self):
        return dict(params_and_adam=M.snapshot(self.P, self.opt), torch_rng=torch.get_rng_state().clone())

    def restore(self, state):
        self.P, self.opt = M.restore(state['params_and_adam'], 'adam', .001)
        torch.set_rng_state(state['torch_rng'])
        for p in self.P: p.grad = None

    def fingerprints(self):
        return dict(body=digest(self.body()), head=digest(self.head()),
                    body_adam=digest([self.opt.state.get(p, {}) for p in self.body()]),
                    head_adam=digest([self.opt.state.get(p, {}) for p in self.head()]),
                    rng=digest(torch.get_rng_state()))

    def groups(self):
        out = dict(body=self.body(), trunk=self.P[:2], head=self.head())
        if self.ln == 'on': out.update(final_gamma=[self.P[4]], final_beta=[self.P[5]])
        return out


def loss_for(logits, one_hot):
    return (logits.logsumexp(1) - (logits * one_hot).sum(1)).mean()


def one_step(model, x, one_hot):
    loss = loss_for(model.forward(x), one_hot)
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Training loss')
    model.opt.zero_grad(set_to_none=True); loss.backward(); model.opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, x, y):
    # Full1200 float32 matmul, exactly M.evaluate when LN is off.
    z = model.forward(x)
    finite(z, 'evaluation logits')
    acc = float((z.argmax(-1) == y).float().mean())
    ce = float((z.logsumexp(-1) - z.gather(1, y[:, None]).squeeze(1)).mean())
    return acc, ce


def norm(values):
    return float(sum((v.detach().double().square().sum() for v in values), torch.tensor(0., dtype=torch.float64)).sqrt())


def tensor_stats(x):
    d = x.double(); mu = d.mean(0); centered = d - mu
    return dict(mean=float(d.mean()), mean_norm=float(mu.norm()), rms=float(d.square().mean().sqrt()),
                centered_rms=float(centered.square().mean().sqrt()), std=float(d.std(unbiased=False)))


def actual_activation_derivative(z, activation):
    """Actual canonical-autograd slope at the float32 training preactivation.

    ELU here is the archived expm1(clamp_max) implementation. Its float32
    backward can round the negative-tail slope to zero where an analytic
    exp(z) or the ViT's F.elu still gives a positive value. Keep that numerical
    distinction visible; never replace the training activation to remove it.
    This temporary independent leaf does not touch model parameter gradients.
    """
    if z.dtype != torch.float32:
        raise ValueError('Derivative diagnostic must use float32 training values')
    with torch.enable_grad():
        u = z.detach().requires_grad_(True)
        value = M.activ(u, activation)
        derivative, = torch.autograd.grad(value, u, grad_outputs=torch.ones_like(value))
    finite(derivative, 'actual activation derivative')
    return derivative.detach()


@torch.no_grad()
def measure(model, x, old, new, step, diagnostics=False):
    h, z, a = model.features(x, auxiliary=True)
    logits = h @ model.P[2].T + model.P[3]
    finite(logits, 'measurement logits')
    acc = float((logits.argmax(-1) == new).float().mean())
    ce = float((logits.logsumexp(-1)-logits.gather(1, new[:, None]).squeeze(1)).mean())
    out = dict(step=int(step), acc=acc, ce=ce)
    if diagnostics:
        lp = logits.double().log_softmax(1); prob = lp.exp(); w = model.P[2].double()
        derivative = actual_activation_derivative(z, model.activation).double().abs()
        d = dict(head_centered_norm=float((w-w.mean(0)).norm()), head=tensor_stats(w),
                 head_bias_norm=float(model.P[3].double().norm()), feature=tensor_stats(h),
                 activation_before_ln=tensor_stats(a), preactivation=tensor_stats(z),
                 logit=tensor_stats(logits),
                 logit_class_centered_rms=float((logits.double()-logits.double().mean(1, keepdim=True)).square().mean().sqrt()),
                 old_acc=float((logits.argmax(1)==old).float().mean()),
                 confidence=float(prob.max(1).values.mean()), entropy=float(-(prob*lp).sum(1).mean()),
                 positive_occupancy=float((z>0).double().mean()),
                 activation_derivative_method='float32 autograd of canonical M.activ; ELU expm1(clamp_max), not ViT F.elu',
                 mean_absolute_activation_derivative=float(derivative.mean()),
                 small_derivative_fraction=float((derivative<.05).double().mean()))
        if model.ln == 'on':
            gamma, beta = model.P[4].double(), model.P[5].double()
            d.update(final_gamma_norm=float(gamma.norm()), final_beta_norm=float(beta.norm()),
                     effective_Wgamma_norm=float((w*gamma).norm()),
                     effective_centered_Wgamma_norm=float(((w-w.mean(0))*gamma).norm()),
                     effective_Wbeta_plus_b_norm=float((w@beta+model.P[3].double()).norm()))
        out['diagnostics'] = d
        finite(model.P, 'model parameters')
        finite([model.opt.state.get(p, {}) for p in model.P], 'Adam')
    finite(out)
    return out


def gradient_record(model):
    saved, stats = {}, {}
    for name, ps in model.groups().items():
        saved[name] = [(p, p.detach().clone(), None if p.grad is None else p.grad.detach().clone()) for p in ps]
        finite([p.grad for p in ps if p.grad is not None], 'gradients.'+name)
        stats[name] = dict(gradient_norm=norm([p.grad for p in ps if p.grad is not None]), parameter_norm=norm(ps))
    return saved, stats


@torch.no_grad()
def finish_record(saved, stats):
    for name, items in saved.items():
        updates = [p-before for p, before, grad in items]
        grads = [grad for p, before, grad in items if grad is not None]
        un, gn = norm(updates), norm(grads)
        dot = sum(float(((p-before).double()*(-grad.double())).sum()) for p, before, grad in items if grad is not None)
        stats[name].update(update_norm=un, update_vs_negative_gradient_cos=dot/(un*gn) if un and gn else None)
    finite(stats)
    return stats


def trajectory(model, x, old, new, order, limit):
    grid = points(limit); curve = [measure(model, x, old, new, 0, True)]; tracks = []
    Y = F.one_hot(new, 10).float()
    for step, idx in enumerate(order[:limit], 1):
        logits = model.forward(x[idx]); loss = loss_for(logits, Y[idx])
        if not bool(torch.isfinite(loss)): raise FloatingPointError(('trajectory', step))
        model.opt.zero_grad(set_to_none=True); loss.backward()
        track = step in (1, 25, 200, limit)
        if track: saved, stats = gradient_record(model)
        model.opt.step()
        if track: tracks.append(dict(step=step, loss=float(loss.detach()), groups=finish_record(saved, stats)))
        if step in grid: curve.append(measure(model, x, old, new, step, step in (25, 200, limit)))
    assert [r['step'] for r in curve] == grid
    finite(model.P, 'final parameters')
    return curve, tracks


def source_file(out, activation, ln, seed):
    return Path(out) / f'source_mlp_{activation}_{ln}_s{seed}.pt'


def stream_data(seed, tasks):
    gl, gb = M.stream('env_labels_0913', seed), M.stream('env_batch_0913', seed)
    labels, orders = {}, {}
    for t in range(1, max(tasks)+2):
        labels[t] = torch.randint(10, (1200,), generator=gl)
        orders[t] = M.order_for(gb, 4000)
    return labels, orders


def small_repeat(model, x, y, order):
    """A deterministic restore/replay check, never part of the source history."""
    start = model.snapshot(); fingerprints = model.fingerprints(); Y = F.one_hot(y, 10).float(); results = []
    for _ in range(2):
        model.restore(start)
        assert model.fingerprints() == fingerprints
        for idx in order[:3]: one_step(model, x[idx], Y[idx])
        results.append(digest(model.snapshot()))
    assert results[0] == results[1], 'Nondeterministic source repeat'
    model.restore(start)
    assert model.fingerprints() == fingerprints
    return dict(updates_per_repeat=3, repeats=2, exact=True, state_and_rng_restored=True)


def make_source(seed, activation, ln, tasks, path):
    started = time.monotonic()
    x, subset = M.load_X(seed); init = initial_params(seed, ln)
    old_path = OLD_STATES / f'states_adam_lr0.001_{activation}_s{seed}.npz'
    # Even LN-on must share this canonical initialization/subset/label stream.
    saved = np.load(old_path)
    assert np.array_equal(subset, saved['subset'])
    assert all(np.array_equal(init[q].numpy(), saved[f'init{q}']) for q in range(4))
    labels, orders = stream_data(seed, tasks)
    assert all(np.array_equal(labels[t].numpy(), saved['y'][t-1]) for t in range(1, max(tasks)+2))
    model = MLP(init, activation, ln)
    repeat = small_repeat(model, x, labels[1], orders[1])
    initial_state = model.snapshot(); bases = {}; history = []; replay_checks = []
    for t in range(1, max(tasks)+1):
        Y = F.one_hot(labels[t], 10).float()
        for idx in orders[t]: one_step(model, x[idx], Y[idx])
        finite(model.P, f'source task{t}')
        acc, ce = evaluate(model, x, labels[t])
        history.append(dict(task=t, acc=acc, ce=ce))
        if ln == 'off':
            errors = [float(np.abs(model.P[q].detach().numpy()-saved[f'end{q}'][t-1]).max()) for q in range(4)]
            if any(v != 0. for v in errors): raise AssertionError(('source replay mismatch', activation, seed, t, errors))
            replay_checks.append(dict(task=t, parameters_bit_exact=True, labels_bit_exact=True, maxabs=0.))
        if t in tasks:
            bases[t] = model.snapshot()
            history[-1]['diagnostics'] = measure(model, x, labels[t], labels[t], 0, True)['diagnostics']
        print(f'SOURCE {activation} LN={ln} s{seed} t{t} acc={acc:.5f}', flush=True)
    source = dict(version=1, study='A', model='mlp', activation=activation, ln=ln, seed=seed,
                  n_examples=1200, tasks=list(tasks), history_steps=4000, initial=initial_state, snapshots=bases,
                  inputs=x, subset=torch.from_numpy(subset.copy()), labels=labels, orders=orders,
                  initial_common_sha256=digest(init[:4]), initial_state_sha256=digest(initial_state),
                  inputs_sha256=digest(x), subset_sha256=digest(subset),
                  labels_sha256={t: digest(v) for t, v in labels.items()},
                  batch_sha256={t: digest(v) for t, v in orders.items()},
                  snapshot_sha256={t: digest(v) for t, v in bases.items()},
                  history=history, replay_checks=replay_checks, deterministic_repeat=repeat,
                  provenance=provenance([old_path, M.DATA/'train-images-idx3-ubyte.gz']),
                  elapsed_s=time.monotonic()-started, complete=True)
    save_state(path, source)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert digest(restored) == digest(source), 'Source serialization mismatch'
    save_json(Path(path).with_suffix('.json'), {k: v for k, v in source.items() if k not in ('initial', 'snapshots', 'inputs', 'subset', 'labels', 'orders')} | dict(source_sha256=sha(path), serialization_exact=True))
    return source


def validate_source(source, seed, activation, ln, tasks):
    assert source['complete'] and source['version'] == 1
    assert (source['seed'], source['activation'], source['ln']) == (seed, activation, ln)
    assert set(tasks) <= set(source['tasks'])
    assert source['provenance']['local_code_hashes'] == code_hashes(), 'Source local execution code changed'
    current_provenance = shared.provenance()
    assert source['provenance']['code_sha256'] == current_provenance['code_sha256'], 'Source shared execution code changed'
    assert source['provenance']['code_hashes'] == current_provenance['code_hashes']
    assert source['provenance']['spec_sha256'] == sha(SPEC), 'Source specification changed'
    for p, old_sha in source['provenance']['sources'].items(): assert sha(p) == old_sha, ('Changed archived source', p)
    assert digest(source['inputs']) == source['inputs_sha256']
    assert digest(source['subset']) == source['subset_sha256']
    assert digest(source['initial']) == source['initial_state_sha256']
    canonical_initial = initial_params(seed, ln)
    assert digest(canonical_initial[:4]) == source['initial_common_sha256']
    assert digest(source['initial']['params_and_adam'][0]) == digest(canonical_initial)
    labels, orders = stream_data(seed, source['tasks'])
    for t in labels:
        assert torch.equal(labels[t], source['labels'][t]) and digest(labels[t]) == source['labels_sha256'][t]
        assert torch.equal(orders[t], source['orders'][t]) and digest(orders[t]) == source['batch_sha256'][t]
    for t in tasks: assert digest(source['snapshots'][t]) == source['snapshot_sha256'][t]
    if ln == 'off':
        assert len(source['replay_checks']) == max(source['tasks'])
        assert all(r['parameters_bit_exact'] and r['labels_bit_exact'] for r in source['replay_checks'])
    finite(source)


@torch.no_grad()
def apply_arm(model, arm, bank, initial_state=None):
    if arm not in ARMS: raise ValueError(arm)
    before = model.fingerprints()
    fork_rng = torch.get_rng_state().clone()
    if arm == 'F0':
        if initial_state is None: raise ValueError('F0 requires canonical initial state')
        model.restore(initial_state)
        # F0 uses the current fork RNG, not an initialization-time RNG.
        # All batches are explicit and this model has no stochastic layers.
        torch.set_rng_state(fork_rng)
    W, b = model.head()
    if arm in ('Z_keep', 'Z_clear', 'F0'):
        W.zero_(); b.zero_()
    else:
        W.copy_(bank['entries'][arm]['W']); b.copy_(bank['entries'][arm]['b'])
    if arm in ('Z_clear', 'F0'):
        model.opt = M.make_opt('adam', model.P, .001)
    else:
        model.opt.state.pop(W, None); model.opt.state.pop(b, None)
    after = model.fingerprints()
    assert not any(model.opt.state.get(p) for p in model.head())
    assert before['rng'] == after['rng']
    if arm != 'F0':
        assert before['body'] == after['body']
    if arm not in ('Z_clear', 'F0'): assert before['body_adam'] == after['body_adam']
    else: assert not any(model.opt.state.get(p) for p in model.body())
    if arm == 'S1': assert before['head'] == after['head']
    return dict(before=before, after=after, head_adam_cleared=True,
                body_adam_cleared=arm in ('Z_clear', 'F0'), fresh_body=arm=='F0')


def legacy_checks(curve, activation, ln, seed, t, arm):
    if ln == 'on': return []
    paths = []
    if activation == 'ELU' and t == 10 and arm in ('S1', 'S01', 'P1', 'P01'):
        paths.append(OLD_SEPARATION / f'mlp_ELU_s{seed}_t10_physical_{arm}_body_keep.json')
    if activation in ('ELU', 'R', 'SILU') and arm in ('S1', 'P1', 'Z_keep'):
        direction = 'P' if arm == 'P1' else 'S'
        policy = 'ZC' if arm == 'Z_keep' else 'KC'
        paths.append(OLD_RESIDUAL / f'mlp_{activation}_S_s{seed}_t{t}_{direction}_tau0_{policy}.json')
    checks = []
    for path in paths:
        old = json.loads(path.read_text()); rows = {r['step']: r for r in old['curve']}
        paired = [(r, rows[r['step']]) for r in curve if r['step'] in rows]
        a = np.asarray([[r['acc'], r['ce']] for r, s in paired]); b = np.asarray([[s['acc'], s['ce']] for r, s in paired])
        exact = bool(np.array_equal(a, b))
        if not exact: raise AssertionError(('Legacy continuation mismatch', str(path), float(np.abs(a-b).max())))
        checks.append(dict(path=str(path), sha256=sha(path), points=len(paired), exact=True, maxabs=0.))
    return checks


def run_source(source, source_path, out, tasks, limit):
    activation, ln, seed = source['activation'], source['ln'], source['seed']
    x = source['inputs']; source_hash = sha(source_path)
    prov = provenance([source_path]); expected = []
    for t in tasks:
        model = MLP(source['initial']['params_and_adam'][0], activation, ln)
        model.restore(source['snapshots'][t]); source_finger = model.fingerprints()
        source_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            h = model.features(x)
            bank = ops.geometry_bank(*model.head(), h.mean(0), seed, 'mlp', t)
            original_mean = (h @ model.P[2].T + model.P[3]).double().mean(0)
        for arm in ARMS:
            name = f'curve_A_mlp_{activation}_{ln}_s{seed}_t{t}_{arm}.json'; expected.append(name)
            path = Path(out)/name
            if path.exists():
                existing = json.loads(path.read_text())
                assert existing['complete'] and existing['source_sha256'] == source_hash
                assert existing['provenance']['code_hashes'] == prov['code_hashes']
                assert existing['provenance']['spec_sha256'] == prov['spec_sha256']
                assert existing['horizon'] == limit and [r['step'] for r in existing['curve']] == points(limit)
                print('RESUME', name, flush=True); continue
            started = time.monotonic(); model.restore(source['snapshots'][t])
            assert model.fingerprints() == source_finger
            checks = apply_arm(model, arm, bank, source['initial'])
            # Every arm receives the corresponding fork RNG, including F0.
            torch.set_rng_state(source_rng)
            checks['after']['rng'] = digest(torch.get_rng_state())
            assert checks['after']['rng'] == source_finger['rng']
            checks['source_state_exact'] = True
            checks['batch_sha256'] = source['batch_sha256'][t+1]
            checks['labels_sha256'] = source['labels_sha256'][t+1]
            checks['canonical_permutation_sha256'] = digest(bank['metadata']['P_permutation'])
            checks['geometry'] = bank['metadata']['checks']
            if arm in ('S1', 'S01', 'P1', 'P01'):
                with torch.no_grad(): now = model.forward(x).double().mean(0)
                err = float((now-original_mean).abs().max())
                tolerance = 2e-5 + 2e-6*float(original_mean.abs().max())
                assert err <= tolerance, ('Mean-logit compensation', arm, err, tolerance)
                checks['mean_logits'] = dict(maxabs=err, tolerance=tolerance, preserved=True)
            else:
                assert all(torch.count_nonzero(p) == 0 for p in model.head())
                checks['head_exact_zero'] = True
            curve, tracks = trajectory(model, x, source['labels'][t], source['labels'][t+1], source['orders'][t+1], limit)
            checks['legacy'] = legacy_checks(curve, activation, ln, seed, t, arm)
            obj = dict(study='A', model='mlp', activation=activation, source_activation=activation,
                       ln=ln, seed=seed, t_end=t, arm=arm, horizon=limit, n_examples=1200,
                       source_id=f'mlp_{activation}_{ln}_s{seed}_t{t}', source_sha256=source_hash,
                       snapshot_sha256=source['snapshot_sha256'][t], initial_state_sha256=source['initial_state_sha256'],
                       source_provenance=source['provenance'], provenance=prov, checks=checks,
                       curve=curve, tracks=tracks, elapsed_s=time.monotonic()-started, complete=True)
            save_json(path, obj)
            print(f'DONE {name} acc={curve[-1]["acc"]:.6f} sec={obj["elapsed_s"]:.2f}', flush=True)
    save_json(Path(out)/f'complete_mlp_{activation}_{ln}_s{seed}.json',
              dict(complete=True, expected_files=expected, source_sha256=source_hash, horizon=limit,
                   provenance=prov, files={name: sha(Path(out)/name) for name in expected}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--activation', choices=ACTIVATIONS, required=True)
    ap.add_argument('--ln', choices=('off', 'on'), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--tasks', default='1,10')
    ap.add_argument('--source-only', action='store_true')
    ap.add_argument('--reuse-source', action='store_true')
    ap.add_argument('--source-path', help='Optional already-generated source .pt outside output directory')
    args = ap.parse_args()
    if args.seed not in range(10): raise ValueError('Registered seeds0..9')
    if not 1 <= args.steps <= 4000: raise ValueError('Continuation horizon outside1..4000')
    tasks = tuple(sorted(set(int(t) for t in args.tasks.split(','))))
    if not tasks or not set(tasks) <= {1, 10}: raise ValueError('Registered task ends1,10')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    path = Path(args.source_path) if args.source_path else source_file(args.out, args.activation, args.ln, args.seed)
    if args.reuse_source:
        source = torch.load(path, map_location='cpu', weights_only=False)
    else:
        if path.exists(): raise FileExistsError(f'{path}; use --reuse-source to validate and reuse')
        source = make_source(args.seed, args.activation, args.ln, tasks, path)
    validate_source(source, args.seed, args.activation, args.ln, tasks)
    if not args.source_only: run_source(source, path, args.out, tasks, args.steps)


if __name__ == '__main__':
    main()
