"""One-step backward-gate interventions at saved ordinary-learning states."""
import argparse
import json
import time
from pathlib import Path
import common as T
from t5_model import LocalFFN, fixed_metric

torch, np, F = T.torch, T.np, T.F
LR = 1e-4


def position(local):
    u = local.u[:, 0].double()
    w, b = local.fc1.weight.double(), local.fc1.bias.double()
    z = F.linear(u, w, b)
    mu, sd = z.mean(0), z.std(0, unbiased=True)
    return mu, sd, mu / (sd + 1e-12), sd > 1e-8


def derivatives(local):
    u = local.u[:, 0].double()
    w = local.fc1.weight.detach().double().requires_grad_()
    b = local.fc1.bias.detach().double().requires_grad_()
    z = F.linear(u, w, b)
    mu, sd = z.mean(0), z.std(0, unbiased=True)
    valid = sd > 1e-8
    q = 1 / (sd + 1e-12)
    ds = ((z - mu).T @ (u - u.mean(0))) / ((len(u) - 1) * sd[:, None].clamp_min(1e-20))
    rw = q[:, None] * u.mean(0) - (mu * q.square())[:, None] * ds
    rb = q
    gw, gb = torch.autograd.grad((mu * q)[valid].sum(), (w, b))
    error = max(float((gw[valid] - rw[valid]).abs().max()), float((gb[valid] - rb[valid]).abs().max()))
    assert error < 1e-9, error
    return rw.detach(), rb.detach(), valid.detach(), error


@torch.no_grad()
def fit(local, y):
    logits = torch.cat([local.forward(idx)[0] for idx in torch.arange(len(y), device=y.device).split(32)])
    return float(F.cross_entropy(logits, y)), float((logits.argmax(1) == y).double().mean())


def fields(local, y):
    cs, gs = [], []
    direct = [torch.zeros_like(local.fc1.weight), torch.zeros_like(local.fc1.bias)]
    for idx in torch.arange(len(y), device=y.device).split(32):
        logits, z, a = local.forward(idx)
        loss = F.cross_entropy(logits, y[idx], reduction='sum')
        c, dw, db = torch.autograd.grad(loss, (a, local.fc1.weight, local.fc1.bias))
        cs.append(c.detach()); gs.append((z.detach() >= 0).to(c.dtype))
        direct[0].add_(dw / len(y)); direct[1].add_(db / len(y))
    c, g = torch.cat(cs), torch.cat(gs)
    flat_u = local.u.reshape(-1, local.u.shape[-1])
    def grad(signal):
        flat = signal.reshape(-1, signal.shape[-1])
        return (flat.T @ flat_u / len(y), flat.sum(0) / len(y))
    result = {'relu': grad(c * g), 'all_open': grad(c),
              'rate_gate': grad(c * g.mean(0, keepdim=True)),
              'no_image_pairing': grad(c.mean(0, keepdim=True) * g),
              'positive_request': grad(c.clamp_min(0) * g),
              'negative_request': grad(c.clamp_max(0) * g)}
    error = max(float((x - d).abs().max()) for x, d in zip(result['relu'], direct))
    scale = max(float(d.abs().max()) for d in direct)
    assert error < 1e-7 + .002 * scale, (error, scale)
    identity = max(float((result['positive_request'][j] + result['negative_request'][j] - result['relu'][j]).abs().max()) for j in (0, 1))
    assert identity < 1e-7 + .002 * scale, (identity, scale)
    norm = lambda pair: torch.sqrt(sum(p.double().square().sum() for p in pair))
    ref = norm(result['relu'])
    norms = {name: float(norm(v)) for name, v in result.items()}
    for name in ('all_open', 'rate_gate', 'no_image_pairing'):
        assert norms[name] > 0, name
        result[name + '_norm'] = tuple(p * (ref / norms[name]).to(p.dtype) for p in result[name])
    facts = dict(autograd_max_abs_error=error, autograd_scale=scale, sign_additivity_error=identity,
                 positive_request_fraction=float((c[g.bool()] > 0).double().mean()),
                 gate_open_fraction=float(g.double().mean()), gradient_norms=norms)
    return result, facts


def summarize(value, valid=None):
    v = value.detach().double()
    if valid is not None:
        v = v[valid]
    return dict(mean=float(v.mean()), median=float(v.median()), down_fraction=float((v < 0).double().mean()))


def run(seed, task, start, layer, output):
    checkpoint = T.ARCHIVE / f'runs/seed{seed}/ckpt/t{task:02d}_s{start:03d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['source_hashes'] == T.own_hashes()
    model = T.build(seed).eval()
    model.load_state_dict(state['model'])
    optimizer = T.C.build_optimizer(model)
    optimizer.load_state_dict(state['optimizer'])
    data, _ = T.C.task_dataset(T.C.image_source(state['config']['data']), seed, task, 10, 25, augment=False)
    x, y = T.C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
    classes = torch.tensor(T.C.loss_classes('SL', seed, task, 10, 25), device='cuda')
    local = LocalFFN(model, layer, x, classes)
    assert local.verify_cache(x)
    initial, frozen = local.state(), local.frozen_digest()
    optimizer_initial = T.cpu_tree(optimizer.state_dict())
    metric = fixed_metric(optimizer, local.fc1)
    mu0, sd0, r0, valid0 = [v.detach() for v in position(local)]
    rw, rb, valid, derivative_error = derivatives(local)
    assert torch.equal(valid, valid0)
    field, checks = fields(local, y)
    checks['r_derivative_max_abs_error'] = derivative_error
    ce0, acc0 = fit(local, y)
    ident = dict(seed=seed, task=task, start=start, layer=layer)
    rows, arrays = [], {}
    for name, (gw, gb) in field.items():
        arrays[f'{name}_gw'] = gw.cpu().numpy()
        arrays[f'{name}_gb'] = gb.cpu().numpy()
        for mode in ('sgd', 'fixed'):
            dw, db = (torch.ones_like(gw), torch.ones_like(gb)) if mode == 'sgd' else metric
            uw, ub = -LR * dw.double() * gw.double(), -LR * db.double() * gb.double()
            dm = uw @ local.mean + ub
            dr = (uw * rw).sum(1) + ub * rb
            row = dict(**ident, arm=name, mode=mode, kind='linear', ce0=ce0, accuracy0=acc0)
            row.update({f'mu_{k}':v for k,v in summarize(dm).items()})
            row.update({f'r_{k}':v for k,v in summarize(dr, valid).items()})
            rows.append(row)
            arrays[f'{name}_{mode}_dmu'] = dm.cpu().numpy()
            arrays[f'{name}_{mode}_dr'] = dr.cpu().numpy()
        if name in ('positive_request', 'negative_request'):
            continue
        for mode in ('fixed', 'adam'):
            local.restore(initial)
            optimizer.load_state_dict(T.cpu_tree(optimizer_initial))
            optimizer.zero_grad(set_to_none=True)
            if mode == 'fixed':
                with torch.no_grad():
                    for p, d, gradient in zip((local.fc1.weight, local.fc1.bias), metric, (gw, gb)):
                        p.add_(d * gradient, alpha=-LR)
            else:
                local.fc1.weight.grad, local.fc1.bias.grad = gw.clone(), gb.clone()
                optimizer.step()
            with torch.no_grad():
                mu1, sd1, r1, valid1 = position(local)
                use = valid0 & valid1
                ce1, acc1 = fit(local, y)
                row = dict(**ident, arm=name, mode=mode, kind='finite', ce0=ce0, accuracy0=acc0,
                           ce1=ce1, accuracy1=acc1, ce_delta=ce1-ce0)
                row.update({f'mu_{k}':v for k,v in summarize(mu1-mu0).items()})
                row.update({f'r_{k}':v for k,v in summarize(r1-r0, use).items()})
                rows.append(row)
                arrays[f'{name}_{mode}_finite_dmu'] = (mu1-mu0).cpu().numpy()
                arrays[f'{name}_{mode}_finite_dr'] = (r1-r0).cpu().numpy()
                arrays[f'{name}_{mode}_valid'] = use.cpu().numpy()
                if name == 'relu':
                    ref = T.ARCHIVE / 't5' / f'seed{seed}_t{task:02d}_s{start:03d}_L{layer}_{mode}'
                    with np.load(ref / 'measurements.npz') as d:
                        err = float(np.abs(r1.cpu().numpy()[use.cpu().numpy()] - d['s1_r'][0][use.cpu().numpy()]).max())
                    assert err < 2e-5, (ident, mode, err)
                    checks[f'{mode}_existing_full_s1_r_error'] = err
            assert local.frozen_digest() == frozen
    local.restore(initial)
    assert local.frozen_digest() == frozen
    folder = output / f'seed{seed}_t{task:02d}_s{start:03d}_L{layer}'
    folder.mkdir(parents=True, exist_ok=False)
    arrays.update(mu0=mu0.cpu().numpy(), sd0=sd0.cpu().numpy(), valid=valid.cpu().numpy())
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    T.write_rows(folder / 'rows.csv', [{k:row.get(k) for k in columns} for row in rows])
    T.atomic_json(folder / 'complete.json', dict(status='complete', **ident, checks=checks,
        checkpoint=str(checkpoint), checkpoint_sha256=T.sha256(checkpoint),
        frozen_unchanged=True, cache_forward_bit_equal=True,
        source_hashes={str(p.relative_to(T.ROOT)): T.sha256(p) for p in
          (Path(__file__), T.ROOT/'specs/spec_vit_two_block_ordinary_drive_1025.md')},
        git_hash=T.git_state()[0], arrays_sha256=T.sha256(folder/'arrays.npz'), updated=time.time()))
    print(json.dumps(dict(completed=ident, checks=checks)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one', action='store_true')
    ap.add_argument('--output', type=Path, default=T.ARCHIVE/'ordinary_drive')
    args = ap.parse_args()
    T.init()
    assert not T.git_state()[1]
    states = [(s,t,k,l) for t in (10,60) for k in (100,500) for s in T.SEEDS for l in (1,2)]
    if args.one:
        states = states[:1]
    for ident in states:
        path = args.output / f'seed{ident[0]}_t{ident[1]:02d}_s{ident[2]:03d}_L{ident[3]}'
        if (path/'complete.json').exists():
            continue
        run(*ident, args.output)


if __name__ == '__main__':
    main()
