"""Direct residual/FFN path norms; no parameter updates."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import tarfile
import time

from evaluate import sha, write_csv, write_json, metrics, TASKS, ARMS
import numpy as np
import torch
from torch.nn import functional as F


def paired_vectors(skip, branch):
    """Per-image descriptors. Norm shares are not additive contributions to a sum."""
    a, b = skip.detach().double().flatten(1), branch.detach().double().flatten(1)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    denominator = na + nb
    nan = torch.full_like(na, float('nan'))
    values = dict(skip_norm=na, ffn_norm=nb,
                  ffn_norm_share=torch.where(denominator > 0, nb / denominator, nan),
                  cosine=torch.where(na * nb > 0, (a * b).sum(1) / (na * nb), nan),
                  sum_to_norm_sum=torch.where(denominator > 0, (a + b).norm(dim=1) / denominator, nan))
    return {k: v.cpu().numpy() for k, v in values.items()}


def trace(model, x):
    layers = model.layers
    h = layers['patch_embed'](x)
    h = torch.cat((model.cls_token.expand(len(x), -1, -1), h), dim=1)
    h = layers['pos_drop'](h + model.pos_embed)
    inputs, branches, outputs = [], [], []
    for i in range(model.depth):
        block = layers[f'block_{i}'].layers
        r = h + block['attn'](block['norm1'](h))
        branch = block['mlp'](block['norm2'](r))
        h = r + branch
        inputs.append(r)
        branches.append(branch)
        outputs.append(h)
    return layers['out'](layers['norm'](h)[:, 0]), inputs, branches, outputs


def measure(model, x, y, classes, batch_size):
    params = list(model.named_parameters())
    sums = [torch.zeros_like(p) for _, p in params]
    chunks, logits_parts = {}, []
    versions = [p._version for _, p in params]
    check = dict(gradient_decomposition_max_abs_error=0., final_patch_gradient_max_abs=0.)
    for start in range(0, len(x), batch_size):
        logits, inputs, branches, outputs = trace(model, x[start:start + batch_size])
        logits = logits[:, classes]
        logits_parts.append(logits.detach().cpu().numpy())
        # Sum avoids batch-size scaling of each image's input gradient.
        loss = F.cross_entropy(logits, y[start:start + batch_size], reduction='sum')
        grads = torch.autograd.grad(loss, outputs + inputs + [p for _, p in params], retain_graph=True)
        depth = model.depth
        for acc, grad in zip(sums, grads[2 * depth:]):
            acc.add_(grad.detach())
        for i, (r, branch, gs, total) in enumerate(zip(inputs, branches, grads[:depth], grads[depth:2 * depth])):
            gf = torch.autograd.grad(branch, r, grad_outputs=gs, retain_graph=i < depth - 1)[0]
            error = float((gs + gf - total).abs().max())
            check['gradient_decomposition_max_abs_error'] = max(check['gradient_decomposition_max_abs_error'], error)
            torch.testing.assert_close(gs + gf, total, rtol=1e-4, atol=1e-7)
            for token, selection in (('all', slice(None)), ('cls', 0)):
                for kind, skip, ffn in (('output', r[:, selection], branch[:, selection]),
                                        ('gradient', gs[:, selection], gf[:, selection])):
                    for key, values in paired_vectors(skip, ffn).items():
                        chunks.setdefault(f'layer{i + 1}_{token}_{kind}_{key}', []).append(values)
            if i == depth - 1:
                check['final_patch_gradient_max_abs'] = max(check['final_patch_gradient_max_abs'], float(gs[:, 1:].abs().max()))
        del grads, logits, inputs, branches, outputs, loss
    assert check['final_patch_gradient_max_abs'] == 0.
    assert versions == [p._version for _, p in params]
    energy, counts = {}, {}
    for (name, _), grad in zip(params, sums):
        group = 'other'
        for i in range(model.depth):
            root = f'layers.block_{i}.layers.'
            if name.startswith(root + 'mlp.') or name.startswith(root + 'norm2.'):
                group = f'ffn{i + 1}'
        energy[group] = energy.get(group, 0.) + float((grad.double() / len(x)).square().sum())
        counts[group] = counts.get(group, 0) + grad.numel()
    total_energy = sum(energy.values())
    parameter_rows = [dict(group=key, parameters=counts[key], gradient_squared_norm=value,
                           gradient_rms=(value / counts[key]) ** .5,
                           gradient_squared_norm_share=value / total_energy if total_energy > 0 else None)
                      for key, value in energy.items()]
    return {k: np.concatenate(v) for k, v in chunks.items()}, np.concatenate(logits_parts), parameter_rows, check


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--raw-output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    assert Path(C.__file__).resolve().is_relative_to(args.source_root.resolve())
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.35)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    source = C.RealImages()
    rows, parameter_rows, validations = [], [], []
    began = time.monotonic()
    for seed in range(5):
        machine = 'vn' if seed < 3 else 'il'
        archive = args.archive_root / f'vit_long_1024_{machine}_raw_seed{seed}.tar'
        with tarfile.open(archive, 'r:') as tar:
            def read(arm, name):
                return tar.extractfile(f'vit_long_1024/runs/{arm}/seed{seed}/{name}').read()
            for arm in ARMS:
                prov = json.loads(read(arm, 'provenance.json'))
                for name, digest in prov['source_hashes'].items():
                    assert sha((args.source_root / name).read_bytes()) == digest
                manifest = {r['path']: r for r in json.loads(read(arm, 'manifest.json'))}
                cfg = prov['config']
                model = make_model('R', seed, 'cuda', cfg['model_config']).eval()
                intask = np.load(io.BytesIO(read(arm, 'intask.npz')))
                for task in TASKS:
                    data, _ = C.task_dataset(source, seed, task, cfg['n_labels'], cfg['images_per_class'], augment=False)
                    classes = C.loss_classes(arm, seed, task, cfg['n_labels'], cfg['images_per_class'])
                    x, y = C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
                    task_index = int(np.where(intask['tasks'] == task)[0][0])
                    for state, step, prefix in (('mid', 100, 'mid'), ('end', 500, 'model')):
                        name = f'ckpt/{prefix}_t{task:02d}.pt'
                        payload = read(arm, name)
                        assert sha(payload) == manifest[name]['sha256']
                        model.load_state_dict(torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True), strict=True)
                        del payload
                        raw, logits, param, check = measure(model, x, y, classes, args.batch_size)
                        baseline = metrics(logits, data.labels.numpy(), logits)
                        record_index = step // cfg['eval_every']
                        assert abs(baseline['accuracy'] - float(intask['mem_correct'][task_index, record_index]) / len(data)) < 1e-10
                        check['baseline_loss_abs_error'] = abs(baseline['loss'] - float(intask['mem_loss'][task_index, record_index]) / len(data))
                        assert check['baseline_loss_abs_error'] < 2e-4
                        ident = dict(arm=arm, seed=seed, task=task, state=state, step=step)
                        validations.append(dict(**ident, **check, parameters_unchanged=True))
                        parameter_rows += [dict(**ident, **row) for row in param]
                        for layer in range(1, model.depth + 1):
                            for token in ('all', 'cls'):
                                for kind in ('output', 'gradient'):
                                    record = dict(**ident, layer=layer, token=token, kind=kind)
                                    for metric in ('skip_norm', 'ffn_norm', 'ffn_norm_share', 'cosine', 'sum_to_norm_sum'):
                                        values = raw[f'layer{layer}_{token}_{kind}_{metric}']
                                        finite = values[np.isfinite(values)]
                                        record[metric + '_median'] = float(np.median(finite)) if len(finite) else None
                                        record[metric + '_finite_n'] = len(finite)
                                    rows.append(record)
                        np.savez_compressed(args.raw_output / f'{arm}_seed{seed}_t{task:02d}_{state}.npz', **raw)
                        last = next(r for r in rows[-24:] if r['layer'] == 6 and r['token'] == 'cls' and r['kind'] == 'gradient')
                        print(json.dumps(dict(**ident, final_ffn_gradient_norm_share=last['ffn_norm_share_median'],
                                              elapsed=round(time.monotonic() - began, 1))), flush=True)
                    del x, y
                del model
                torch.cuda.empty_cache()
        write_csv(args.output / 'path_ratios.csv', rows)
        write_csv(args.output / 'parameter_gradients.csv', parameter_rows)
        write_json(args.output / 'gradient_validation.json', validations)
    write_json(args.output / 'gradient_run.json', dict(status='complete', checkpoints=len(validations),
                 source_module=str(C.__file__), batch_size=args.batch_size, elapsed_seconds=time.monotonic() - began))


if __name__ == '__main__':
    main()
