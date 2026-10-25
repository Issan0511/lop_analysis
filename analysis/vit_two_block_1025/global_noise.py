"""Paired full-model Adam continuations, iid sampling versus full-batch."""
import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path
import common as T
torch, np, F = T.torch, T.np, T.F
START_STEPS = (20, 100, 500)


def hashes():
    paths = [Path(__file__), T.ROOT / 'specs/spec_vit_two_block_global_noise_1025.md']
    return dict(T.own_hashes(), **{str(p.relative_to(T.ROOT)): T.sha256(p) for p in paths})


def gradient(model, optimizer, x, y, classes, indices):
    optimizer.zero_grad(set_to_none=True)
    for idx in indices.split(32):
        logits = model(x.index_select(0, idx)).index_select(1, classes)
        loss = F.cross_entropy(logits, y.index_select(0, idx), reduction='sum') / len(indices)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)


def summarize(records, points, out):
    rows, arrays = [], {}
    for step in points:
        for layer in (1, 2):
            key = f's{step}_L{layer}'
            for field in ('mu', 'sd', 'r', 'valid', 'opened'):
                arrays[f'{key}_{field}'] = np.stack([r['arrays'][f'{key}_{field}'] for r in records])
            fit = [next(x for x in r['rows'] if x['step'] == step and x['layer'] == layer) for r in records]
            for field in ('ce', 'accuracy'):
                arrays[f'{key}_{field}'] = np.array([x[field] for x in fit])
            if step == 0:
                continue
            valid = (arrays[f's0_L{layer}_valid'] & arrays[f'{key}_valid']).all(0)
            row = dict(step=step, layer=layer, common_valid_units=int(valid.sum()))
            for field in ('mu', 'r'):
                delta = arrays[f'{key}_{field}'] - arrays[f's0_L{layer}_{field}']
                if field == 'r':
                    delta = delta[:, valid]
                per_replica = delta.mean(1)
                full, reps = per_replica[0], per_replica[1:]
                row.update({f'{field}_full_delta': float(full), f'{field}_replica_delta': float(reps.mean()),
                    f'{field}_noise_effect': float(reps.mean() - full),
                    f'{field}_noise_effect_se': float(reps.std(ddof=1) / np.sqrt(len(reps))),
                    f'{field}_replica_negative_fraction': float((reps < 0).mean())})
            row.update(ce_full=fit[0]['ce'], ce_replica=float(np.mean([x['ce'] for x in fit[1:]])),
                       accuracy_full=fit[0]['accuracy'], accuracy_replica=float(np.mean([x['accuracy'] for x in fit[1:]])))
            rows.append(row)
    T.write_rows(out / 'summary.csv', rows)
    np.savez_compressed(out / 'measurements.npz', **arrays)
    return rows


def run(args):
    T.init()
    head, dirty = T.git_state()
    assert not dirty, dirty
    source_hashes = hashes()
    if args.gate:
        path = T.ARCHIVE / 'gate/continuous/checkpoint.pt'
        replicas, steps, points = 3, 3, (0, 1, 3)
    else:
        gate = json.loads((T.ARCHIVE / 'gate/global_noise.json').read_text())
        assert gate['status'] == 'passed' and gate['source_hashes'] == source_hashes
        assert args.seed in T.SEEDS and args.task in (10, 60) and args.step in START_STEPS
        path = T.ARCHIVE / f'runs/seed{args.seed}/ckpt/t{args.task:02d}_s{args.step:03d}.pt'
        replicas, steps, points = 32, 50, (0, 1, 5, 20, 50)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['source_hashes'] == T.own_hashes()
    seed, task, start = state['config']['seed'], state['task'], state['step']
    cfg = dict(seed=seed, task=task, start=start, replicas=replicas, steps=steps, points=list(points),
               batch=128, lr=1e-4, gate=args.gate, checkpoint=str(path), checkpoint_sha256=T.sha256(path), source_hashes=source_hashes)
    out = args.output or T.ARCHIVE / 'global_noise' / f'seed{seed}_t{task:02d}_s{start:03d}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'provenance.json').exists():
        assert json.loads((out / 'provenance.json').read_text())['config'] == cfg
    else:
        T.atomic_json(out / 'provenance.json', dict(config=cfg, git_hash=head, started=time.time()))
    source = T.C.image_source(state['config']['data'])
    data, plan = T.C.task_dataset(source, seed, task, 10, 25, augment=False)
    x, y = T.C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
    classes = torch.tensor(T.C.loss_classes('SL', seed, task, 10, 25), device='cuda')
    model = T.build(seed).eval()
    model.load_state_dict(state['model'])
    optimizer = T.C.build_optimizer(model)
    optimizer.load_state_dict(state['optimizer'])
    initial_digest = T.digest_tree(model.state_dict())
    allidx = torch.arange(len(y), device='cuda')
    checks = {}
    if args.gate:
        rng = T.digest_tree(T.get_rng())
        T.observe(model, x, classes, y)
        assert initial_digest == T.digest_tree(model.state_dict()) and rng == T.digest_tree(T.get_rng())
        gradient(model, optimizer, x, y, classes, allidx)
        micro = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).index_select(1, classes), y).backward()
        error = max(float((p.grad - micro[k]).abs().max()) for k, p in model.named_parameters())
        scale = max(float(p.grad.abs().max()) for p in model.parameters())
        assert error < 1e-7 + .002 * scale, (error, scale)
        checks = dict(observer_state_rng_unchanged=True, gradient_max_abs_error=error, gradient_scale=scale)
    T.atomic_json(out / 'input.json', dict(plan=plan, images_sha256=T.array_sha(data.images),
        labels_sha256=T.array_sha(data.labels.numpy()), paths=data.paths))
    records, manifest = [], []
    for replica in range(-1, replicas):
        file = out / ('full.pt' if replica == -1 else f'replica_{replica:03d}.pt')
        if file.exists():
            record = torch.load(file, map_location='cpu', weights_only=False)
            assert record['config'] == cfg and record['replica'] == replica and record['initial_digest'] == initial_digest
        else:
            if shutil.disk_usage(out).free < 25 * 1024 ** 3:
                raise RuntimeError('Free disk below 25 GiB')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(T.cpu_tree(state['optimizer']))
            assert initial_digest == T.digest_tree(model.state_dict())
            gen = torch.Generator().manual_seed(T.C.stream_seed('t5_replica', seed, task, start, 1, replica))
            rows, arrays, indices_saved = [], {}, []
            def observe(step):
                rr, aa = T.observe(model, x, classes, y)
                rows.extend(dict(step=step, **r) for r in rr)
                arrays.update({f's{step}_{k}': v for k, v in aa.items()})
            before = time.monotonic()
            observe(0)
            for j in range(1, steps + 1):
                idx = torch.arange(len(y)) if replica == -1 else torch.randint(len(y), (128,), generator=gen)
                indices_saved.append(idx)
                gradient(model, optimizer, x, y, classes, idx.to('cuda'))
                optimizer.step()
                if j in points:
                    observe(j)
            non_fc1_changed = [k for k, p in model.named_parameters()
                               if '.fc1.' not in k and not torch.equal(p.detach().cpu(), state['model'][k])]
            assert non_fc1_changed
            record = dict(config=cfg, replica=replica, rows=rows, arrays=arrays,
                initial_digest=initial_digest, indices_sha256=T.digest_tree(indices_saved),
                non_fc1_changed=non_fc1_changed, model=T.cpu_tree(model.state_dict()),
                optimizer=T.cpu_tree(optimizer.state_dict()), seconds=time.monotonic() - before)
            T.atomic_save(file, record)
            print(json.dumps(dict(replica_complete=replica, seconds=record['seconds'])), flush=True)
        # Analysis only needs moments, avoiding all final optimizer states in RAM.
        records.append(dict(rows=record['rows'], arrays=record['arrays']))
        manifest.append(dict(path=file.name, bytes=file.stat().st_size, sha256=T.sha256(file)))
        T.atomic_json(out / 'status.json', dict(status='running', completed_replicas=replica + 2, updated=time.time()))
    summarize(records, points, out)
    for name in ('summary.csv', 'measurements.npz', 'input.json', 'provenance.json'):
        file = out / name
        manifest.append(dict(path=name, bytes=file.stat().st_size, sha256=T.sha256(file)))
    T.atomic_json(out / 'manifest.json', manifest)
    result = dict(status='complete', config=cfg, checks=checks, updated=time.time())
    T.atomic_json(out / 'complete.json', result)
    T.atomic_json(out / 'status.json', result)
    if args.gate:
        T.atomic_json(T.ARCHIVE / 'gate/global_noise_first_pass.json', result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task', type=int, default=10)
    ap.add_argument('--step', type=int, default=100)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--output', type=Path)
    run(ap.parse_args())
