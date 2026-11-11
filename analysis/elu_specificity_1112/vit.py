"""Study B: genuine current R/ELU in frozen ReLU-trained ViT histories.

No real data or model is loaded at import. Every trajectory is reconstructed from
the same archived source, and the original source hash remains shared by F0.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from analysis.loading_bridge_1111 import dynamic as D
from analysis.p_residual_loading_1111 import run as R
from analysis.readout_separation_1111 import ops
from analysis.joudaki_vit_battle_0919.model import (
    BattleActivation, activations, train_forward,
)
from analysis.joudaki_vit_battle_0919.run import get_rng, set_rng
from analysis.vit_long_1024 import common as C
from . import common as shared

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs/spec_elu_specificity_1112.md'
ARMS = ('S1', 'S01', 'P1', 'P01', 'Z_keep', 'Z_clear', 'F0')
T_END = 30
EVAL_CONVENTION = (
    'Native prior ViT convention: eager genuine-activation CLS features in '
    '128-example chunks; active head logits and full-set metrics in float64.'
)


def finite_tree(value, path='root'):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, path + '.' + str(key))
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            finite_tree(item, path + '.' + str(key))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError(path)


def save_json(path, value):
    finite_tree(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, prefix=path.name + '.', suffix='.tmp',
        delete=False,
    ) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write('\n')
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def points(limit):
    return sorted({0, 1, 5, 25, 50, 100, 199, 200, 201, limit,
                   *range(0, limit + 1, 25)} & set(range(limit + 1)))


def replace_activations(model, activation, seed):
    if activation not in ('R', 'ELU'):
        raise ValueError(activation)
    before = [(name, id(p)) for name, p in model.named_parameters()]
    for index in range(model.depth):
        layer = model.layers[f'block_{index}'].layers['mlp'].layers
        layer['act'] = BattleActivation(
            activation, layer['fc1'].out_features, seed, index,
        ).to(model.cls_token.device)
    after = [(name, id(p)) for name, p in model.named_parameters()]
    assert before == after, 'Activation replacement changed trainable parameters'
    return activation_checks(model, activation)


def derivative(z, activation):
    if activation == 'R':
        # Native BattleActivation R uses clamp; autograd chooses 1 at z == 0.
        return (z >= 0).to(z.dtype)
    if activation == 'ELU':
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z))
    raise ValueError(activation)


def activation_checks(model, activation):
    checks = []
    for index, module in enumerate(activations(model)):
        assert isinstance(module, BattleActivation)
        assert module.arm == activation and module.ref.name == activation
        assert module.V is None and not list(module.parameters())
        z = torch.tensor([-30., -4., -1., 0., .25, 3.],
                         device=model.cls_token.device, requires_grad=True)
        value = module(z)
        expected = torch.clamp(z, min=0.) if activation == 'R' else F.elu(z, alpha=1.)
        actual_grad, = torch.autograd.grad(value.sum(), z)
        assert torch.equal(value, expected)
        assert torch.allclose(actual_grad, derivative(z, activation), rtol=2e-7, atol=0.)
        checks.append(dict(layer=index, module=type(module).__name__,
                           activation=activation, forward_exact=True,
                           genuine_autograd_derivative=True, added_parameters=0,
                           zero_derivative=float(actual_grad[3]),
                           implementation=('torch.clamp(z,min=0)' if activation == 'R'
                                           else 'torch.nn.functional.elu(z,alpha=1,inplace=False)'),
                           numerical_tail=('Native clamp autograd uses derivative 1 at exactly zero.'
                                           if activation == 'R' else
                                           'F.elu backward uses exp(z) on the negative branch; '
                                           'unlike the canonical MLP expm1 expression, it does not '
                                           'round the derivative to zero near z=-17 in float32.')))
    assert len(checks) == model.depth
    return checks


def native_rebuild_check(model, x, y, rows, seed):
    """R sham is exactly the native eager R function and full gradient."""
    before = R.digest(model.state_dict())
    original = model(x)[:, rows]
    params = list(model.parameters())
    gradients = torch.autograd.grad(F.cross_entropy(original, y), params)
    original = original.detach()
    replace_activations(model, 'R', seed)
    rebuilt = model(x)[:, rows]
    sham_gradients = torch.autograd.grad(F.cross_entropy(rebuilt, y), params)
    assert torch.equal(original, rebuilt)
    assert all(torch.equal(a, b) for a, b in zip(gradients, sham_gradients))
    assert before == R.digest(model.state_dict())
    return dict(output_exact=True, all_parameter_gradients_exact=True,
                parameters_and_buffers_unchanged=True)


def norm(values):
    items = [p.detach().double().square().sum() for p in values]
    return float(torch.stack(items).sum().sqrt()) if items else 0.


def parameter_groups(box):
    W, b, _ = box.head()
    final = box.model.layers['norm']
    if isinstance(final, torch.nn.LayerNorm):
        return dict(
            body=list(box.body()),
            trunk=[p for p in box.body() if p is not final.weight and p is not final.bias],
            final_gamma=[final.weight], final_beta=[final.bias], head=[W, b],
        )
    return dict(body=list(box.body()), trunk=list(box.body()),
                final_gamma=[], final_beta=[], head=[W, b])


def gradient_record(box):
    saved, report = {}, {}
    for name, parameters in parameter_groups(box).items():
        saved[name] = [(p, p.detach().clone(),
                        None if p.grad is None else p.grad.detach().clone())
                       for p in parameters]
        report[name] = dict(
            parameter_norm=norm(parameters),
            gradient_norm=norm([p.grad for p in parameters if p.grad is not None]),
            n_parameters=sum(p.numel() for p in parameters),
        )
    return saved, report


@torch.no_grad()
def finish_gradient_record(box, saved, report):
    for name, items in saved.items():
        updates = [p - old for p, old, _ in items]
        gradients = [g for _, _, g in items if g is not None]
        un, gn = norm(updates), norm(gradients)
        dot = sum(float(((p - old).double() * -g.double()).sum())
                  for p, old, g in items if g is not None)
        states = [box.opt.state.get(p, {}) for p, _, _ in items]
        moments = [s['exp_avg'] for s in states if 'exp_avg' in s]
        variances = [s['exp_avg_sq'] for s in states if 'exp_avg_sq' in s]
        report[name].update(
            update_norm=un,
            update_vs_negative_gradient_cos=dot / (un * gn) if un > 0 and gn > 0 else None,
            adam_m_norm=norm(moments),
            adam_v_rms=(sum(float(v.double().sum()) for v in variances)
                        / sum(v.numel() for v in variances)) ** .5 if variances else 0.,
        )
    finite_tree(report)
    return report


@torch.no_grad()
def feature_batch(model, x, with_preln=False):
    layers = model.layers
    h = layers['patch_embed'](x)
    h = torch.cat((model.cls_token.expand(h.shape[0], -1, -1), h), 1) + model.pos_embed
    h = layers['pos_drop'](h)
    for index in range(model.depth):
        h = layers[f'block_{index}'](h)
    preln = h[:, 0]
    out = layers['norm'](h)[:, 0]
    return (out, preln) if with_preln else out


@torch.no_grad()
def measure(box, step, detailed=False):
    h = box.features()
    W, b, rows = box.head()
    out = D.head_stats(h, W[rows], b[rows], box.old, box.new)
    z = h.double() @ W[rows].double().T + b[rows].double()
    diagnostic = dict(
        activation=box.activation,
        feature_rms=float(h.double().square().mean().sqrt()),
        head_norm=float(W[rows].double().norm()),
        head_mean_norm=float(W[rows].double().mean(0).norm()),
        head_bias_norm=float(b[rows].double().norm()),
        logit_rms=float(z.square().mean().sqrt()),
        logit_mean_norm=float(z.mean(0).norm()),
        old_ce=float(F.cross_entropy(z, box.old)),
        **{key: value for key, value in out.items() if key not in ('acc', 'ce')},
    )
    final = box.model.layers['norm']
    if isinstance(final, torch.nn.LayerNorm):
        diagnostic.update(
            final_gamma_norm=float(final.weight.double().norm()),
            final_beta_norm=float(final.bias.double().norm()),
            effective_Wgamma_norm=float((W[rows].double() * final.weight.double()).norm()),
            effective_centered_Wgamma_norm=float(
                ((W[rows].double() - W[rows].double().mean(0)) * final.weight.double()).norm()),
            effective_Wbeta_plus_b_norm=float(
                (W[rows].double() @ final.bias.double() + b[rows].double()).norm()),
        )
    if detailed:
        was_training = box.model.training
        box.model.eval()
        _, traces = C.trace(box.model, box.X[:128])
        _, preln = feature_batch(box.model, box.X[:128], with_preln=True)
        diagnostic.update(
            diagnostic_sample_size=min(128, len(box.X)),
            pre_final_norm_mean=float(preln.double().mean()),
            pre_final_norm_std=float(preln.double().std(unbiased=False)),
            layers=[],
        )
        for index, (_, preactivation) in enumerate(traces):
            z0 = preactivation.flatten(0, 1).double()
            dphi = derivative(preactivation.flatten(0, 1), box.activation).double()
            diagnostic['layers'].append(dict(
                layer=index, activation=box.activation,
                preactivation_mean=float(z0.mean()),
                preactivation_std=float(z0.std(unbiased=False)),
                positive_occupancy=float((z0 > 0).double().mean()),
                mean_abs_derivative=float(dphi.abs().mean()),
                small_derivative_fraction=float((dphi.abs() < .05).double().mean()),
                per_unit_mean_median=float(z0.mean(0).median()),
                per_unit_std_median=float(z0.std(0, unbiased=False).median()),
            ))
        box.model.train(was_training)
    result = dict(step=int(step), acc=out['acc'], ce=out['ce'], diagnostics=diagnostic)
    finite_tree(result)
    return result


class ViT:
    def __init__(self, seed, source_arm, activation, engine):
        if source_arm not in ('S', 'SL') or activation not in ('R', 'ELU'):
            raise ValueError((source_arm, activation))
        assert C.PRODUCTION_FILE.name == 'production_2blk_rlfix.json'
        C.set_determinism(1)
        self.C, self.seed, self.t = C, seed, T_END
        self.source_arm, self.activation, self.engine = source_arm, activation, engine
        self.n, self.eval_every = 2500, 25
        self.model = C.build_arm_model(source_arm, seed, 'cuda', 0., 2, 8.)
        self.initial = copy.deepcopy(self.model.state_dict())
        directory = D.RAW_V / source_arm / f'seed{seed}' / 'ckpt'
        self.sources = [directory / 'model_t30.pt', directory / 'adam_t30.pt']
        self.source_hashes = {str(path): D.sha(path) for path in self.sources}
        self.source_id = f'vit_{source_arm}_s{seed}_t30_R_trained'
        self.source_sha256 = R.digest(self.source_hashes)
        self.archived = torch.load(self.sources[0], map_location='cuda', weights_only=False)
        self.adam = torch.load(self.sources[1], map_location='cuda', weights_only=False)
        assert self.adam['task'] == T_END
        from analysis.joudaki_vit_battle_0919.data import PairedBatches
        data, plan = C.task_dataset(C.image_source('real'), seed, 31, 10, 250, augment=False)
        assert len(data) == self.n and not data.augment
        self.X = C.eval_tensor(data.images, 'cuda')
        self.old = C.task_labels(self.n, seed, T_END, 10).cuda()
        self.new = data.labels.cuda()
        self.rows = torch.tensor(C.loss_classes(source_arm, seed, 31, 10, 250), device='cuda')
        self.order = [torch.tensor([i for i, _ in batch], device='cuda')
                      for batch in PairedBatches(self.n, 128, 1500, seed, 31)]
        assert list(dict.fromkeys(len(batch) for batch in self.order)) == [128, 68]
        self.model.load_state_dict(self.archived)
        self.opt = C.build_optimizer(self.model)
        self.opt.load_state_dict(copy.deepcopy(self.adam['optimizer']))
        set_rng(self.adam['rng'])
        self.model.train()
        idx = self.order[0]
        self.checks = dict(
            native_R_rebuild=native_rebuild_check(
                self.model, self.X[idx], self.new[idx], self.rows, seed),
            data_plan=plan, data_paths_sha256=R.digest(data.paths),
            input_tensor_sha256=R.digest(self.X),
            labels_old_sha256=R.digest(self.old), labels_new_sha256=R.digest(self.new),
            batch_order_sha256=R.digest(self.order),
            initial_state_sha256=R.digest(self.initial),
            source_state_sha256=R.digest(self.model.state_dict()),
            original_fingerprints=R.fingerprints(self),
        )
        # Both treatments use the identical R-sham rebuild before the registered
        # R/ELU replacement. Activation attributes are intentionally not state_dict.
        self.activation = 'R'
        self.original_measurement = measure(self, 0, detailed=True)
        self.activation = activation
        self.checks['activation_modules'] = replace_activations(self.model, activation, seed)
        self.base = copy.deepcopy(self.model.state_dict())
        self.prepared_measurement = measure(self, 0, detailed=True)
        set_rng(self.adam['rng'])
        self.model.train()
        self.source_snapshot = R.snapshot(self)
        self.checks['prepared_fingerprints'] = R.fingerprints(self)
        assert self.checks['prepared_fingerprints'] == self.checks['original_fingerprints']
        self.checks['eval_convention'] = EVAL_CONVENTION
        self.compiled = train_forward(self.model, engine)

    def parameters(self):
        return list(self.model.parameters())

    def head(self):
        head = self.model.layers['out']
        return head.weight, head.bias, self.rows

    def body(self):
        W, b, _ = self.head()
        return [p for p in self.parameters() if p is not W and p is not b]

    def forward(self, x):
        return self.compiled(x)[:, self.rows]

    @torch.no_grad()
    def features(self):
        was_training = self.model.training
        self.model.eval()
        result = torch.cat([feature_batch(self.model, x) for x in self.X.split(128)])
        self.model.train(was_training)
        return result

    def reset(self, fresh=False):
        R.restore(self, self.source_snapshot)
        if fresh:
            self.model.load_state_dict(self.initial)
            self.opt.state.clear()
        self.opt.zero_grad(set_to_none=True)
        assert all(module.arm == self.activation for module in activations(self.model))

    def batches(self, limit):
        for idx in self.order[:limit]:
            yield self.X[idx], self.new[idx]


@torch.no_grad()
def prepare_arm(box, arm, bank):
    if arm not in ARMS:
        raise ValueError(arm)
    box.reset(fresh=arm == 'F0')
    before = R.fingerprints(box)
    W, b, rows = box.head()
    inactive = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
    inactive[rows] = False
    inactive_before = (W[inactive].clone(), b[inactive].clone())
    if arm in ('Z_keep', 'Z_clear', 'F0'):
        W.zero_()
        b.zero_()
    else:
        W[rows] = bank['entries'][arm]['W']
        b[rows] = bank['entries'][arm]['b']
        assert torch.equal(W[inactive], inactive_before[0])
        assert torch.equal(b[inactive], inactive_before[1])
    box.opt.state.pop(W, None)
    box.opt.state.pop(b, None)
    if arm in ('Z_clear', 'F0'):
        box.opt.state.clear()
    after = R.fingerprints(box)
    assert before['body'] == after['body']
    if arm not in ('Z_clear', 'F0'):
        assert before['body_adam'] == after['body_adam']
    assert not box.opt.state.get(W) and not box.opt.state.get(b)
    if arm in ('Z_clear', 'F0'):
        assert not box.opt.state
    if arm == 'F0':
        assert R.digest(box.model.state_dict()) != R.digest(box.archived)
    return dict(before=before, after=after, activation=box.activation,
                source_or_fresh='fresh_initial_body' if arm == 'F0' else 'learned_source',
                body_unchanged_by_head_policy=True, head_adam_empty=True,
                all_adam_empty=arm in ('Z_clear', 'F0'),
                inactive_head_rows='zeroed' if arm.startswith('Z') or arm == 'F0' else 'unchanged')


def warm_engine(box):
    if box.engine != 'compile':
        return dict(batch_sizes=[], optimizer_steps=0, state_and_rng_restored=True)
    state = R.snapshot(box)
    fingerprint = R.fingerprints(box)
    shapes = []
    for idx in box.order:
        size = len(idx)
        if size in shapes:
            continue
        shapes.append(size)
        box.opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(box.forward(box.X[idx]), box.new[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('Warmup loss')
        loss.backward()
        if len(shapes) == 2:
            break
    assert shapes == [128, 68]
    R.restore(box, state)
    box.opt.zero_grad(set_to_none=True)
    assert R.fingerprints(box) == fingerprint
    assert R.digest(R.snapshot(box)[2]) == R.digest(state[2])
    return dict(batch_sizes=shapes, optimizer_steps=0, state_and_rng_restored=True)


def engine_check(box):
    """Validate the selected compiled genuine activation against native eager."""
    idx = box.order[0]
    parameters = box.parameters()
    results = []
    for forward in (lambda x: box.model(x)[:, box.rows], box.forward):
        logits = forward(box.X[idx])
        gradients = torch.autograd.grad(F.cross_entropy(logits, box.new[idx]), parameters)
        results.append((logits.detach(), gradients))
    checks = {}
    pairs = [('logits', results[1][0], results[0][0], False)]
    pairs += [(name, b, a, True) for (name, _), a, b in zip(
        box.model.named_parameters(), results[0][1], results[1][1])]
    for name, current, reference, gradient in pairs:
        a, b = current.detach().double(), reference.detach().double()
        delta = a - b
        err, l2 = float(delta.abs().max()), float(delta.norm())
        refmax, refnorm = float(b.abs().max()), float(b.norm())
        atol = 2e-4 if gradient else 1e-3
        passed = (err <= atol + 3e-4 * refmax
                  and l2 <= 3e-4 * refnorm + 2e-6 * a.numel() ** .5)
        checks[name] = dict(
            passed=passed, maxabs=err, relative_l2=l2 / refnorm if refnorm > 0 else None,
            maxabs_atol=atol, maxabs_rtol=3e-4, l2_rtol=3e-4,
            l2_atol_per_element=2e-6,
        )
        if not passed:
            raise AssertionError(('native_vs_selected_engine', name, checks[name]))
    return checks


def trajectory(box, limit):
    wanted = points(limit)
    curve = [measure(box, 0, detailed=True)]
    tracks = []
    for step, (x, y) in enumerate(box.batches(limit), 1):
        loss = F.cross_entropy(box.forward(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'loss at update {step}')
        box.opt.zero_grad(set_to_none=True)
        loss.backward()
        track = step in (1, 25, 200, limit)
        if track:
            saved, gradients = gradient_record(box)
        box.opt.step()
        if track:
            tracks.append(dict(step=step, loss=float(loss.detach()),
                               groups=finish_gradient_record(box, saved, gradients)))
        if step in wanted:
            curve.append(measure(box, step, detailed=step in (25, 200, limit)))
    assert [row['step'] for row in curve] == wanted
    finite_tree(curve)
    return curve, tracks


def provenance(box, args):
    files = [
        Path(__file__), Path(D.__file__), Path(R.__file__), Path(ops.__file__), Path(C.__file__),
        ROOT / 'analysis/joudaki_vit_battle_0919/model.py',
        ROOT / 'analysis/joudaki_vit_battle_0919/data.py',
        ROOT / 'analysis/joudaki_vit_battle_0919/run.py',
        ROOT / 'src/rlcifar_mlp_battle_0918.py',
        ROOT / 'analysis/loading_bridge_1111/static_probe.py',
        C.PRODUCTION_FILE,
    ]
    result = shared.provenance(box.sources)
    result.update(dict(
        git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        script_sha256=D.sha(__file__), spec_sha256=D.sha(SPEC),
        helpers={str(path.relative_to(ROOT)): D.sha(path) for path in files},
        sources=box.source_hashes, torch=torch.__version__, numpy=np.__version__,
        args=vars(args), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        evaluation_convention=EVAL_CONVENTION,
        loss='cross_entropy(all_200_logits[:, fixed_active_10_rows], new_task31_labels)',
    ))
    return result


def validate_curve(document, box, args, arm, prov):
    assert document['complete'] and document['study'] == 'B'
    assert (document['activation'], document['source_arm'], document['seed'], document['arm']) == (
        box.activation, box.source_arm, box.seed, arm)
    assert document['source_sha256'] == box.source_sha256
    assert document['horizon'] == args.steps and document['n_examples'] == 2500
    for key in ('script_sha256', 'spec_sha256', 'helpers', 'sources', 'code_hashes', 'code_sha256'):
        assert document['provenance'][key] == prov[key], key
    assert document['provenance']['args']['engine'] == args.engine
    assert [row['step'] for row in document['curve']] == points(args.steps)
    finite_tree(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-arm', choices=['S', 'SL'], required=True)
    parser.add_argument('--activation', choices=['R', 'ELU'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--engine', choices=['compile', 'eager'], default='compile')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1500:
        parser.error('--steps must be in 1..1500')
    selected = args.arms.split(',')
    if len(set(selected)) != len(selected) or not set(selected) <= set(ARMS):
        parser.error('--arms must be distinct registered arm names')
    setup = time.monotonic()
    box = ViT(args.seed, args.source_arm, args.activation, args.engine)
    prov = provenance(box, args)
    W, b, rows = box.head()
    bank = ops.geometry_bank(W[rows], b[rows], box.features().mean(0),
                             box.seed, 'vit', T_END)
    meta = dict(
        study='B', model='vit', source_activation='R', activation=box.activation,
        source_arm=box.source_arm, ln='on' if box.source_arm == 'SL' else 'off',
        seed=box.seed, t_end=T_END, source_id=box.source_id,
        source_sha256=box.source_sha256, provenance=prov, checks=box.checks,
        original_R_before_intervention=box.original_measurement,
        current_activation_before_head_intervention=box.prepared_measurement,
        head_bank=bank['metadata'], setup_elapsed_s=time.monotonic() - setup,
        complete=True,
    )
    out = Path(args.out)
    name = f'B_vit_{box.source_arm}_{box.activation}_s{box.seed}_t30'
    save_json(out / f'source_{name}.json', meta)
    print('SOURCE', name, f'{meta["setup_elapsed_s"]:.2f}s', flush=True)
    if args.source_only:
        return
    for arm in selected:
        path = out / f'curve_{name}_{arm}.json'
        if args.reuse and path.exists():
            validate_curve(json.loads(path.read_text()), box, args, arm, prov)
            print('REUSE', path.name, flush=True)
            continue
        start = time.monotonic()
        fork = prepare_arm(box, arm, bank)
        warmup = warm_engine(box)
        selected_engine = engine_check(box)
        curve, tracks = trajectory(box, args.steps)
        result = dict(
            study='B', model='vit', activation=box.activation, source_activation='R',
            source_arm=box.source_arm, ln='on' if box.source_arm == 'SL' else 'off',
            seed=box.seed, t_end=T_END, arm=arm, horizon=args.steps, n_examples=2500,
            source_id=box.source_id, source_sha256=box.source_sha256,
            prepared_fork_hashes=fork,
            provenance=prov,
            checks=dict(source=box.checks, head_bank=bank['metadata']['checks'],
                        fork=fork, warmup=warmup, selected_engine=selected_engine),
            curve=curve, tracks=tracks, elapsed_s=time.monotonic() - start,
            complete=True,
        )
        validate_curve(result, box, args, arm, prov)
        save_json(path, result)
        print('DONE', path.name, f'{result["elapsed_s"]:.2f}s',
              f'acc={curve[-1]["acc"]:.6f}', flush=True)
    print('COMPLETE', name, 'curves', len(selected), flush=True)


if __name__ == '__main__':
    main()

