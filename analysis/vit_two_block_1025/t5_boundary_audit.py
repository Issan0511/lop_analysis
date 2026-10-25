"""Diagnose strict-positive gate versus the original clamp derivative at zero.

Read-only replay of saved replicas; does not train or overwrite measurements.
"""
import argparse
import json
from pathlib import Path
import common as T
from t5_model import LocalFFN, fixed_metric
from t5_ledger import kernel
torch = T.torch


def run(case, steps):
    T.init()
    cfg = json.loads((case / 'provenance.json').read_text())['config']
    state = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    assert T.sha256(Path(cfg['checkpoint'])) == cfg['checkpoint_sha256']
    model = T.build(cfg['seed']).eval()
    model.load_state_dict(state['model'])
    optimizer = T.C.build_optimizer(model)
    optimizer.load_state_dict(state['optimizer'])
    data, _ = T.C.task_dataset(T.C.image_source(state['config']['data']), cfg['seed'], cfg['task'], 10, 25, augment=False)
    x, y = T.C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
    classes = torch.tensor(T.C.loss_classes('SL', cfg['seed'], cfg['task'], 10, 25), device='cuda')
    local = LocalFFN(model, cfg['layer'], x, classes)
    dw, db = [m.double() for m in fixed_metric(optimizer, local.fc1)]
    records = []
    sources = []
    for replica in range(cfg['replicas']):
        p = case / f'replica_{replica:03d}.pt'
        record = torch.load(p, map_location='cpu', weights_only=False)
        assert record['config'] == cfg and record['replica'] == replica
        records.append(record)
        sources.append(dict(path=str(p), sha256=T.sha256(p)))
    out = T.ARCHIVE / 'gate/t5_boundary_audit' / case.name
    out.mkdir(parents=True, exist_ok=True)
    result, arrays = [], {}
    for step in steps:
        strict = torch.zeros(local.fc1.out_features, device='cuda', dtype=torch.float64)
        boundary, direct = torch.zeros_like(strict), torch.zeros_like(strict)
        count, max_dz_error, examples = 0, 0., []
        scale = -cfg['lr'] / len(y) / cfg['replicas']
        for lo in range(0, len(y), 32):
            idx = torch.arange(lo, min(lo + 32, len(y)), device='cuda')
            k = kernel(local.u.index_select(0, idx), local.mean, dw, db)
            for replica, record in enumerate(records):
                local.restore(record['states'][step])
                logits, z, a = local.forward(idx)
                loss = T.F.cross_entropy(logits, y.index_select(0, idx), reduction='sum')
                c, dz, gw, gb = torch.autograd.grad(loss, (a, z, local.fc1.weight, local.fc1.bias))
                expected_dz = c * (z >= 0)
                max_dz_error = max(max_dz_error, float((expected_dz - dz).abs().max()))
                zero = z == 0
                count += int(zero.sum())
                for pos in zero.nonzero()[:max(0, 12 - len(examples))]:
                    n, token, unit = map(int, pos)
                    examples.append(dict(replica=replica, image=lo + n, token=token, unit=unit,
                                         c=float(c[n, token, unit]), dz=float(dz[n, token, unit])))
                strict.add_(scale * (k * c.double() * (z > 0)).sum((0, 1)))
                boundary.add_(scale * (k * c.double() * zero).sum((0, 1)))
                direct.add_(scale * ((dw * gw.double() * local.mean).sum(1) + db * gb.double()))
        row = dict(step=step, strict_error=float((strict - direct).abs().max()),
            corrected_error=float((strict + boundary - direct).abs().max()),
            boundary_mean=float(boundary.mean()), boundary_max_abs=float(boundary.abs().max()),
            direct_mean=float(direct.mean()), direct_max_abs=float(direct.abs().max()),
            exact_zero_elements=count, all_elements=len(y)*local.u.shape[1]*local.fc1.out_features*cfg['replicas'],
            derivative_error=max_dz_error, examples=examples)
        assert row['derivative_error'] == 0
        assert row['corrected_error'] < max(1e-8, 2e-4 * row['direct_max_abs'])
        result.append(row)
        for name, value in [('strict', strict), ('boundary', boundary), ('direct', direct)]:
            arrays[f's{step}_{name}'] = value.cpu().numpy()
        print(json.dumps(row), flush=True)
    T.np.savez_compressed(out / 'arrays.npz', **arrays)
    T.atomic_json(out / 'audit.json', dict(case=str(case), config=cfg, steps=result,
        source_sha256=T.sha256(Path(__file__)), replicas=sources))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    run(args.case, args.steps)
