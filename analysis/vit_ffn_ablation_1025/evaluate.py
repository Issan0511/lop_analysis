"""Frozen-checkpoint FFN ablations; see specs/spec_vit_ffn_ablation_1025.md."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

ALPHAS = (0., .25, .5, .75, .9, 1.)
TASKS = (1, 10, 30, 60)
ARMS = ('R0L', 'SL')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(logits, labels, baseline):
    z, b = torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(baseline, dtype=torch.float64)
    y = torch.as_tensor(labels)
    centered = z - z.mean(1, keepdim=True)
    bc = b - b.mean(1, keepdim=True)
    delta = (centered - bc).square().mean().sqrt()
    lp, lb = z.log_softmax(1), b.log_softmax(1)
    loss, base_loss = F.cross_entropy(z, y), F.cross_entropy(b, y)
    acc, base_acc = (z.argmax(1) == y).double().mean(), (b.argmax(1) == y).double().mean()
    return dict(accuracy=float(acc), loss=float(loss), accuracy_delta_pp=float(100 * (acc - base_acc)),
                loss_delta=float(loss - base_loss), prediction_flip_rate=float((z.argmax(1) != b.argmax(1)).double().mean()),
                centered_logit_delta_rms=float(delta), centered_logit_relative_rms=float(delta / bc.square().mean().sqrt().clamp_min(1e-30)),
                kl_full_to_intervention=float((lb.exp() * (lb - lp)).sum(1).mean()))


@torch.inference_mode()
def evaluate(model, x, classes, batch_size):
    """Cache the prefix; apply final norm/head separately for each intervention."""
    outputs = {f'ffn_{a:g}': [] for a in ALPHAS}
    outputs['block_0'] = []
    branch_ratios = []
    checks = dict(forward_max_abs_error=0., ffn_hook_max_abs_error=0., block_hook_max_abs_error=0.)
    versions = [p._version for p in model.parameters()]
    layers = model.layers
    last = layers[f'block_{model.depth - 1}']
    readout = lambda h: layers['out'](layers['norm'](h))[:, classes]
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        h = layers['patch_embed'](xb)
        h = torch.cat((model.cls_token.expand(len(xb), -1, -1), h), dim=1)
        h = layers['pos_drop'](h + model.pos_embed)
        for i in range(model.depth - 1):
            h = layers[f'block_{i}'](h)
        block_input = h
        last_layers = last.layers
        residual = h + last_layers['attn'](last_layers['norm1'](h))
        branch = last_layers['mlp'](last_layers['norm2'](residual))
        branch_ratios.append((branch[:, 0].norm(dim=1) / residual[:, 0].norm(dim=1).clamp_min(1e-30)).cpu().numpy())
        local = {f'ffn_{a:g}': readout(residual[:, 0] + a * branch[:, 0]) for a in ALPHAS}
        local['block_0'] = readout(block_input[:, 0])
        if start == 0:
            ordinary = model(xb)[:, classes]
            hook = last_layers['mlp'].register_forward_hook(lambda module, args, output: torch.zeros_like(output))
            try:
                ffn_zero = model(xb)[:, classes]
            finally:
                hook.remove()
            hook = last.register_forward_hook(lambda module, args, output: args[0])
            try:
                block_zero = model(xb)[:, classes]
            finally:
                hook.remove()
            for key, actual, expected in (
                ('forward_max_abs_error', local['ffn_1'], ordinary),
                ('ffn_hook_max_abs_error', local['ffn_0'], ffn_zero),
                ('block_hook_max_abs_error', local['block_0'], block_zero)):
                checks[key] = float((actual - expected).abs().max())
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        for key, logits in local.items():
            outputs[key].append(logits.cpu().numpy())
    assert versions == [p._version for p in model.parameters()], 'parameters changed during inference'
    checks['parameters_unchanged'] = True
    return {k: np.concatenate(v) for k, v in outputs.items()}, np.concatenate(branch_ratios), checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--raw-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    for folder in (args.output, args.raw_output):
        folder.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.35)
    source = C.RealImages()
    rows, validations, provenance, data_records = [], [], [], []
    began = time.monotonic()
    for seed in args.seeds:
        machine = 'vn' if seed < 3 else 'il'
        archive = args.archive_root / f'vit_long_1024_{machine}_raw_seed{seed}.tar'
        with tarfile.open(archive, 'r:') as tar:
            def read(arm, name):
                return tar.extractfile(f'vit_long_1024/runs/{arm}/seed{seed}/{name}').read()
            for arm in ARMS:
                prov = json.loads(read(arm, 'provenance.json'))
                source_checks = {name: sha((args.source_root / name).read_bytes()) == digest
                                 for name, digest in prov['source_hashes'].items()}
                assert all(source_checks.values()), {k: v for k, v in source_checks.items() if not v}
                manifest = {r['path']: r for r in json.loads(read(arm, 'manifest.json'))}
                task_rows = {int(r['task']): r for r in csv.DictReader(io.StringIO(read(arm, 'per_task.csv').decode()))}
                intask = np.load(io.BytesIO(read(arm, 'intask.npz')))
                cfg = prov['config']
                assert cfg['model_config']['depth'] == 6 and cfg['ln'] and cfg['dropout'] == 0 and cfg['augment'] == 'off'
                model = make_model('R', seed, 'cuda', cfg['model_config']).eval()
                for task in TASKS:
                    data, plan = C.task_dataset(source, seed, task, cfg['n_labels'], cfg['images_per_class'], augment=False)
                    recorded = task_rows[task]
                    assert plan['classes'] == json.loads(recorded['classes'])
                    classes = C.loss_classes(arm, seed, task, cfg['n_labels'], cfg['images_per_class'])
                    assert classes == json.loads(recorded['loss_rows'])
                    x, labels = C.eval_tensor(data.images, 'cuda'), data.labels.numpy()
                    data_records.append(dict(arm=arm, seed=seed, task=task, n=len(labels), plan=plan,
                                             loss_rows=classes, image_bytes_sha256=sha(data.images.tobytes()),
                                             labels_sha256=sha(labels.tobytes()), paths=data.paths))
                    task_index = int(np.where(intask['tasks'] == task)[0][0])
                    for state, step, prefix in (('mid', 100, 'mid'), ('end', 500, 'model')):
                        name = f'ckpt/{prefix}_t{task:02d}.pt'
                        payload = read(arm, name)
                        digest = sha(payload)
                        assert digest == manifest[name]['sha256'] and len(payload) == manifest[name]['bytes']
                        state_dict = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                        model.load_state_dict(state_dict, strict=True)
                        del payload, state_dict
                        outputs, norms, sanity = evaluate(model, x, classes, args.batch_size)
                        base = outputs['ffn_1']
                        base_metrics = metrics(base, labels, base)
                        record_index = step // cfg['eval_every']
                        recorded_acc = float(intask['mem_correct'][task_index, record_index]) / len(labels)
                        recorded_loss = float(intask['mem_loss'][task_index, record_index]) / len(labels)
                        assert abs(base_metrics['accuracy'] - recorded_acc) < 1e-10
                        assert abs(base_metrics['loss'] - recorded_loss) < 2e-4, (arm, seed, task, step, base_metrics['loss'], recorded_loss)
                        ident = dict(arm=arm, seed=seed, task=task, state=state, step=step)
                        validations.append(dict(**ident, checkpoint_sha256_ok=True, source_hashes_ok=True,
                                                recorded_accuracy=recorded_acc, recomputed_accuracy=base_metrics['accuracy'],
                                                recorded_loss=recorded_loss, recomputed_loss=base_metrics['loss'],
                                                loss_abs_error=abs(base_metrics['loss'] - recorded_loss), **sanity))
                        provenance.append(dict(**ident, archive=str(archive), member=f'vit_long_1024/runs/{arm}/seed{seed}/{name}',
                                               checkpoint_sha256=digest, training_git_hash=prov['git_hash']))
                        for key, logits in outputs.items():
                            intervention, alpha = key.split('_')
                            rows.append(dict(**ident, intervention=intervention, alpha=float(alpha),
                                             branch_to_residual_norm_mean=float(norms.mean()),
                                             branch_to_residual_norm_median=float(np.median(norms)),
                                             **metrics(logits, labels, base)))
                        np.savez_compressed(args.raw_output / f'{arm}_seed{seed}_t{task:02d}_{state}.npz',
                                            labels=labels, branch_to_residual_norm=norms, **outputs)
                        print(json.dumps(dict(**ident, base_acc=base_metrics['accuracy'],
                                              removed_acc=rows[-7]['accuracy'], elapsed=round(time.monotonic() - began, 1))), flush=True)
                    del x
                del model
                torch.cuda.empty_cache()
        write_csv(args.output / 'per_checkpoint.csv', rows)
        write_json(args.output / 'validation.json', validations)
        write_json(args.output / 'checkpoint_provenance.json', provenance)
        write_json(args.raw_output / 'dataset_records.json', data_records)
    write_json(args.output / 'run.json', dict(seeds=args.seeds, tasks=TASKS, arms=ARMS, alphas=ALPHAS,
                                            source_root=str(args.source_root), torch=torch.__version__,
                                            gpu=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic() - began,
                                            batch_size=args.batch_size, checkpoints=len(validations), status='complete'))


if __name__ == '__main__':
    main()
