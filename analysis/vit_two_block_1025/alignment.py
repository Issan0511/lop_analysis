"""Norm-matched readout interventions and layerwise fixed-metric/actual Adam moves."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import common as T
torch, np, F = T.torch, T.np, T.F
MODULE_PATH = T.ROOT / 'analysis/vit_phases_1025/measure.py'
spec = importlib.util.spec_from_file_location('two_block_projection', MODULE_PATH)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def hashes():
    paths = [Path(__file__), MODULE_PATH, T.ROOT / 'specs/spec_vit_two_block_alignment_1025.md']
    return dict(T.own_hashes(), **{str(p.relative_to(T.ROOT)): T.sha256(p) for p in paths})


@torch.no_grad()
def signals(model, old, x, y, classes, seed, task, start):
    h, hx = M.features(model, old).double(), M.features(model, x).double()
    w = model.layers['out'].weight[classes].double()
    b = model.layers['out'].bias[classes].double()
    vc, bc = w - w.mean(0), b - b.mean()
    q = F.one_hot(y, len(classes)).double()
    alignment = (q - 1 / len(classes)).T @ h
    load = alignment * (vc * alignment).sum() / alignment.square().sum()
    remain = vc - load
    assert abs(float((load * remain).sum())) <= 1e-10 * float(vc.square().sum())
    heads = dict(switch=vc, half=remain + .5 * load, zero=remain,
                 norm_unload=remain * vc.norm() / remain.norm())
    for j in range(4):
        gen = torch.Generator().manual_seed(T.C.stream_seed('two_block_alignment', seed, task, start, j))
        perm = torch.randperm(len(q), generator=gen).to(q.device)
        v = (q[perm] - 1 / len(classes)).T @ h
        for direction in (remain, load):
            v -= (v * direction).sum() / direction.square().sum() * direction
        v *= load.norm() / v.norm()
        heads[f'perm{j}'] = remain + v
        assert abs(float(heads[f'perm{j}'].norm() / vc.norm()) - 1) < 1e-8
        assert abs(float((heads[f'perm{j}'] * load).sum() / load.square().sum())) < 1e-8
    data, info = {}, []
    for name, v in heads.items():
        p = (hx @ v.T + bc).softmax(1)
        data[name] = ((p - 1 / len(classes)) @ v / len(x)).float()
        info.append(dict(variant=name, norm_ratio=float(v.norm() / vc.norm()),
                         alignment_coefficient=float((v * load).sum() / load.square().sum()),
                         pmax=float(p.max(1).values.mean()),
                         old_ce=float(F.cross_entropy(h @ v.T + bc, y))))
    p = (hx @ vc.T + bc).softmax(1)
    data['load'] = ((p - 1 / len(classes)) @ load / len(x)).float()
    data['remainder'] = ((p - 1 / len(classes)) @ remain / len(x)).float()
    # Ordinary fit is always measured on old images; added only to the old arm.
    if x.data_ptr() == old.data_ptr():
        data['fit'] = ((p - q) @ vc / len(x)).float()
    return data, heads, info


def backward(model, x, signal):
    params = dict(model.named_parameters())
    fields = {name: {k: torch.zeros_like(p) for k, p in params.items()} for name in signal}
    for lo in range(0, len(x), 32):
        h = M.body(model, x[lo:lo + 32])[0]
        for j, (name, value) in enumerate(signal.items()):
            grads = torch.autograd.grad(h, tuple(params.values()), grad_outputs=value[lo:lo + len(h)],
                                       retain_graph=j + 1 < len(signal), allow_unused=True)
            for (key, _), g in zip(params.items(), grads):
                if g is not None:
                    fields[name][key].add_(g.detach())
    return fields


@torch.no_grad()
def denominators(model, optimizer):
    result = {}
    for name, p in model.named_parameters():
        s = optimizer.state[p]
        result[name] = (s['exp_avg_sq'] / (1 - .999 ** float(s['step']))).sqrt() + 1e-8
    return result


def direct_gradient_gate(model, optimizer, x, classes, field):
    optimizer.zero_grad(set_to_none=True)
    for xx in x.split(32):
        logits = model(xx).index_select(1, classes)
        loss = (logits.logsumexp(1) - logits.mean(1)).sum() / len(x)
        loss.backward()
    error, scale = 0., 0.
    for key, p in model.named_parameters():
        if key.startswith('layers.out.'):
            continue
        error = max(error, float((p.grad - field[key]).abs().max()))
        scale = max(scale, float(p.grad.abs().max()))
    assert error < 1e-7 + .002 * scale, (error, scale)
    optimizer.zero_grad(set_to_none=True)
    return dict(max_abs_error=error, scale=scale)


def run(args):
    T.init()
    head, dirty = T.git_state()
    assert not dirty, dirty
    source_hashes = hashes()
    if args.gate:
        path = T.ARCHIVE / 'gate/continuous/checkpoint.pt'
    else:
        gate = json.loads((T.ARCHIVE / 'gate/alignment.json').read_text())
        assert gate['status'] == 'passed' and gate['source_hashes'] == source_hashes
        assert args.seed in T.SEEDS and args.task in (10, 60) and args.step in (100, 500)
        path = T.ARCHIVE / f'runs/seed{args.seed}/ckpt/t{args.task:02d}_s{args.step:03d}.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['source_hashes'] == T.own_hashes()
    seed, task, start = state['config']['seed'], state['task'], state['step']
    out = args.output or T.ARCHIVE / 'alignment' / f'seed{seed}_t{task:02d}_s{start:03d}'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), f'Already complete: {out}'
    model = T.build(seed).eval()
    model.load_state_dict(state['model'])
    optimizer = T.C.build_optimizer(model)
    optimizer.load_state_dict(state['optimizer'])
    initial_digest = T.digest_tree(model.state_dict())
    source = T.C.image_source(state['config']['data'])
    old, _ = T.C.task_dataset(source, seed, task, 10, 25, augment=False)
    nxt, _ = T.C.task_dataset(source, seed, task + 1, 10, 25, augment=False)
    oldx, nextx = T.C.eval_tensor(old.images, 'cuda'), T.C.eval_tensor(nxt.images, 'cuda')
    y = old.labels.to('cuda')
    classes = torch.tensor(T.C.loss_classes('SL', seed, task, 10, 25), device='cuda')
    raw = M.probe_fields(model, oldx)
    den = denominators(model, optimizer)
    arrays = {f'initial_L{i + 1}_{name}': value.cpu().numpy()
              for i, entry in enumerate(raw) for name, value in entry.items()}
    checks, heads_rows, position_rows, finite_rows = {}, [], [], []
    cfg = dict(seed=seed, task=task, start=start, gate=args.gate, checkpoint=str(path),
               checkpoint_sha256=T.sha256(path), source_hashes=source_hashes)
    T.atomic_json(out / 'provenance.json', dict(config=cfg, git_hash=head, started=time.time()))
    for arm, x in [('old', oldx), ('next', nextx)]:
        signal, heads, hi = signals(model, oldx, x, y, classes, seed, task, start)
        heads_rows.extend(dict(dataset=arm, **r) for r in hi)
        gs = backward(model, x, signal)
        if args.gate:
            checks[f'{arm}_direct_gradient'] = direct_gradient_gate(model, optimizer, x, classes, gs['switch'])
        for metric in ('SGD', 'Adam_fixed'):
            tangent = {name: {k: g if metric == 'SGD' else g / den[k] for k, g in fields.items()}
                       for name, fields in gs.items()}
            full, local, check = M.project(model, oldx, raw, tangent, gate=args.gate and arm == 'old')
            checks[f'{arm}_{metric}'] = check
            for route, result in [('full', full), ('local', local)]:
                for name, moments in result.items():
                    for moment, value in moments.items():
                        delta = -1e-4 * value.detach().cpu().numpy()
                        arrays[f'{arm}_{metric}_{route}_{name}_{moment}'] = delta
                        for layer in (1, 2):
                            valid = raw[layer - 1]['valid'].cpu().numpy()
                            a = delta[layer - 1][valid]
                            position_rows.append(dict(dataset=arm, metric=metric, route=route, signal=name,
                                moment=moment, layer=layer, mean=float(a.mean()), median=float(np.median(a)),
                                negative_fraction=float((a < 0).mean())))
        assert T.digest_tree(model.state_dict()) == initial_digest
        for name in list(heads) + (['fit'] if arm == 'old' else []):
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(T.cpu_tree(state['optimizer']))
            with torch.no_grad():
                if name != 'fit':
                    original = model.layers['out'].weight[classes]
                    model.layers['out'].weight[classes] = (heads[name] + original.double().mean(0)).float()
            optimizer.zero_grad(set_to_none=True)
            for xx, yy in zip(x.split(32), y.split(32)):
                logits = model(xx).index_select(1, classes)
                loss = F.cross_entropy(logits, yy, reduction='sum') if name == 'fit' else (logits.logsumexp(1) - logits.mean(1)).sum()
                (loss / len(x)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            after = M.probe_fields(model, oldx)
            for layer, (before, now) in enumerate(zip(raw, after), 1):
                valid = (before['valid'] & now['valid']).cpu().numpy()
                for moment in ('mu', 'sd', 'r'):
                    delta = (now[moment] - before[moment]).cpu().numpy()
                    arrays[f'{arm}_finite_{name}_L{layer}_{moment}'] = delta
                    finite_rows.append(dict(dataset=arm, variant=name, layer=layer, moment=moment,
                        mean=float(delta[valid].mean()), median=float(np.median(delta[valid])),
                        negative_fraction=float((delta[valid] < 0).mean())))
        model.load_state_dict(state['model'])
        assert T.digest_tree(model.state_dict()) == initial_digest
        print(json.dumps(dict(completed_dataset=arm, seed=seed, task=task, start=start)), flush=True)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    for name, rows in [('positions', position_rows), ('finite', finite_rows), ('heads', heads_rows)]:
        T.write_rows(out / f'{name}.csv', rows)
    T.atomic_json(out / 'checks.json', checks)
    files = ['arrays.npz', 'positions.csv', 'finite.csv', 'heads.csv', 'checks.json', 'provenance.json']
    T.atomic_json(out / 'manifest.json', [dict(path=p, bytes=(out / p).stat().st_size, sha256=T.sha256(out / p)) for p in files])
    T.atomic_json(out / 'complete.json', dict(status='complete', config=cfg, restored_initial_state=True, checks=checks))
    if args.gate:
        T.atomic_json(T.ARCHIVE / 'gate/alignment.json', dict(status='passed', source_hashes=source_hashes,
            git_hash=head, checks=checks, restored_initial_state=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task', type=int, default=10)
    ap.add_argument('--step', type=int, default=500)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--output', type=Path)
    run(ap.parse_args())
