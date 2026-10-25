"""Registered fixed-input replica experiment; completed replicas can be resumed."""
import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path
import common as T
from t5_model import LocalFFN, fixed_metric, update
from t5_ledger import ReplicaMoments, kernel, components
torch, np, F = T.torch, T.np, T.F


def source_hashes():
    paths = [Path(__file__).with_name(n) for n in ('t5_run.py', 't5_model.py', 't5_ledger.py', 't5_checks.py')]
    paths += [T.ROOT / 'specs/spec_vit_two_block_t5_1025.md']
    return dict(T.own_hashes(), **{str(p.relative_to(T.ROOT)): T.sha256(p) for p in paths})


def raw_mu(local):
    return local.fc1.weight.double() @ local.mean + local.fc1.bias.double()


def evaluate(local, optimizer, metric, y, records, points, out, lr, gate):
    """The first record is deterministic full-batch; only the rest enter covariance."""
    n, replicas = len(y), len(records) - 1
    w = local.fc1.out_features
    rows, arrays, checks = [], {}, []
    dw, db = [d.double() for d in metric]
    for step in points:
        mu = torch.zeros((len(records), w), dtype=torch.float64, device=y.device)
        sq, opened = torch.zeros_like(mu), torch.zeros_like(mu)
        ces = torch.zeros(len(records), dtype=torch.float64, device=y.device)
        correct = torch.zeros_like(ces)
        ledger = {name: torch.zeros(w, dtype=torch.float64, device=y.device)
                  for name in ('total', 'mean_field', 'gate_covariance', 'identity_error')}
        direct, gatevar, selfcov = torch.zeros_like(dw[:, 0]), torch.zeros_like(dw[:, 0]), torch.zeros_like(dw[:, 0])
        halves = [torch.zeros_like(direct), torch.zeros_like(direct)]
        for lo in range(0, n, 32):
            idx = torch.arange(lo, min(n, lo + 32), device=y.device)
            stats = ReplicaMoments()
            groups = [ReplicaMoments(), ReplicaMoments()]
            k = kernel(local.u.index_select(0, idx), local.mean, dw, db)
            for r, record in enumerate(records):
                local.restore(record['states'][step])
                logits, z, a = local.forward(idx)
                loss = F.cross_entropy(logits, y.index_select(0, idx), reduction='sum')
                if r > 0:
                    c, gw, gb = torch.autograd.grad(loss, (a, local.fc1.weight, local.fc1.bias))
                    stats.add(z, c)
                    group = int(r - 1 >= replicas // 2)
                    groups[group].add(z, c)
                    direct.add_(-lr / n / replicas * ((dw * gw.double() * local.mean).sum(1) + db * gb.double()))
                with torch.no_grad():
                    cls = z[:, 0].double()
                    mu[r].add_(cls.sum(0)); sq[r].add_(cls.square().sum(0))
                    opened[r].add_((cls > 0).sum(0))
                    ces[r].add_(loss.detach().double())
                    correct[r].add_((logits.argmax(1) == y.index_select(0, idx)).sum())
            part = components(stats, k, lr, n)
            for name in ledger:
                ledger[name].add_(part[name])
            values = stats.values()
            gatevar.add_(values['gate_variance'].sum((0, 1)) / (n * local.u.shape[1]))
            selfcov.add_(values['activation_gate_covariance'].sum((0, 1)) / (n * local.u.shape[1]))
            for j, group in enumerate(groups):
                halves[j].add_(components(group, k, lr, n)['gate_covariance'])
        mu /= n
        sd = ((sq - n * mu.square()) / (n - 1)).clamp_min(0).sqrt()
        rvalues = mu / (sd + 1e-12)
        valid = sd > 1e-8
        error = (ledger['total'] - direct).abs().max()
        tolerance = max(1e-8, 2e-4 * float(direct.abs().max()))
        assert float(error) <= tolerance, (step, float(error), tolerance)
        assert float(ledger['identity_error'].abs().max()) < 1e-11
        if step == 0:
            assert float(ledger['gate_covariance'].abs().max()) < 1e-11
        check = dict(step=step, direct_max_abs_error=float(error), tolerance=tolerance,
                     identity_max_abs_error=float(ledger['identity_error'].abs().max()))
        checks.append(check)
        quantities = dict(mu=mu, sd=sd, r=rvalues, valid=valid, opened=opened / n, ce=ces / n,
                          accuracy=correct / n, direct=direct, gate_variance=gatevar,
                          activation_gate_covariance=selfcov, covariance_half0=halves[0],
                          covariance_half1=halves[1], **ledger)
        arrays.update({f's{step}_{name}': value.detach().cpu().numpy() for name, value in quantities.items()})
        row = dict(full_ce=float(ces[0] / n), replica_ce=float(ces[1:].mean() / n),
                   full_accuracy=float(correct[0] / n), replica_accuracy=float(correct[1:].mean() / n),
                   full_mu=float(mu[0].mean()), replica_mu=float(mu[1:].mean()),
                   full_r=float(rvalues[0][valid[0]].mean()), replica_r=float(rvalues[1:][valid[1:]].mean()),
                   expected_fixed_mu=float(ledger['total'].mean()), mean_field=float(ledger['mean_field'].mean()),
                   gate_covariance=float(ledger['gate_covariance'].mean()),
                   covariance_half0=float(halves[0].mean()), covariance_half1=float(halves[1].mean()),
                   crossing_fraction=float((gatevar > 0).double().mean()), **check)
        rows.append(row)
        T.atomic_json(out / 'status.json', dict(status='analyzing', step=step, updated=time.time()))
        print(json.dumps(dict(measured=step, gate=gate, **row)), flush=True)
    np.savez_compressed(out / 'measurements.npz', **arrays)
    T.write_rows(out / 'summary.csv', rows)
    return checks


def run(args):
    T.init()
    head, dirty = T.git_state()
    assert not dirty, dirty
    hashes = source_hashes()
    if args.gate:
        checkpoint = T.ARCHIVE / 'gate/continuous/checkpoint.pt'
        replicas, steps, batch, points = 3, 3, 8, (0, 1, 3)
    else:
        assert args.seed in T.SEEDS and args.task in (10, 60) and args.step in (100, 500)
        gate = json.loads((T.ARCHIVE / 'gate/t5.json').read_text())
        assert gate['status'] == 'passed' and gate['source_hashes'] == hashes
        checkpoint = T.ARCHIVE / f'runs/seed{args.seed}/ckpt/t{args.task:02d}_s{args.step:03d}.pt'
        replicas, steps, batch, points = 32, 50, 128, (0, 1, 5, 20, 50)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['source_hashes'] == T.own_hashes()
    seed, task, start = state['config']['seed'], state['task'], state['step']
    if not args.gate:
        assert (seed, task, start) == (args.seed, args.task, args.step)
    out = args.output or T.ARCHIVE / 't5' / f'seed{seed}_t{task:02d}_s{start:03d}_L{args.layer}_{args.mode}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = dict(seed=seed, task=task, start=start, layer=args.layer, mode=args.mode, replicas=replicas,
               steps=steps, batch=batch, points=list(points), lr=1e-4, gate=args.gate,
               checkpoint=str(checkpoint), checkpoint_sha256=T.sha256(checkpoint), source_hashes=hashes)
    if (out / 'provenance.json').exists():
        assert json.loads((out / 'provenance.json').read_text())['config'] == cfg
    else:
        T.atomic_json(out / 'provenance.json', dict(config=cfg, git_hash=head, started=time.time()))
    model = T.build(seed)
    model.load_state_dict(state['model'])
    optimizer = T.C.build_optimizer(model)
    optimizer.load_state_dict(state['optimizer'])
    data, plan = T.C.task_dataset(T.C.image_source(state['config']['data']), seed, task, 10, 25, augment=False)
    x, y = T.C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
    classes = torch.tensor(T.C.loss_classes('SL', seed, task, 10, 25), device='cuda')
    local = LocalFFN(model, args.layer, x, classes)
    initial, frozen = local.state(), local.frozen_digest()
    metric = fixed_metric(optimizer, local.fc1)
    assert local.verify_cache(x)
    T.atomic_json(out / 'input.json', dict(plan=plan, paths=data.paths, images_sha256=T.array_sha(data.images),
        labels_sha256=T.array_sha(data.labels.numpy()), frozen_digest=frozen,
        u_sha256=T.digest_tree(local.u), h_sha256=T.digest_tree(local.h), prefix_bit_equal=True))
    records, manifest = [], []
    for replica in range(-1, replicas):
        path = out / ('full.pt' if replica == -1 else f'replica_{replica:03d}.pt')
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=False)
            assert record['config'] == cfg and record['replica'] == replica
            assert T.digest_tree(record['states'][0]) == T.digest_tree(initial)
            assert sorted(record['states']) == list(points)
        else:
            if shutil.disk_usage(out).free < 25 * 1024 ** 3:
                raise RuntimeError('Free disk below 25 GiB')
            local.restore(initial)
            optimizer.load_state_dict(T.cpu_tree(state['optimizer']))
            gen = torch.Generator().manual_seed(T.C.stream_seed('t5_replica', seed, task, start, args.layer, replica))
            snapshots, indices_saved = {0: local.state()}, []
            begin = time.monotonic()
            for j in range(1, steps + 1):
                indices = torch.arange(len(y)) if replica == -1 else torch.randint(len(y), (batch,), generator=gen)
                indices_saved.append(indices)
                update(local, optimizer, metric, indices.to('cuda'), y, args.mode, cfg['lr'])
                if j in points:
                    snapshots[j] = local.state()
            assert local.frozen_digest() == frozen
            assert local.verify_cache(x)
            record = dict(config=cfg, replica=replica, states=snapshots,
                indices_sha256=T.digest_tree(indices_saved), seconds=time.monotonic() - begin,
                frozen_unchanged=True, prefix_bit_equal_after_updates=True,
                final_optimizer=[T.cpu_tree(optimizer.state[p]) for p in (local.fc1.weight, local.fc1.bias)]
                                if args.mode == 'adam' else None)
            T.atomic_save(path, record)
            print(json.dumps(dict(replica_complete=replica, mode=args.mode, layer=args.layer,
                                  seconds=record['seconds'])), flush=True)
        records.append(record)
        manifest.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=T.sha256(path)))
        T.atomic_json(out / 'status.json', dict(status='training', completed_replicas=replica + 2,
                                               total=replicas + 1, updated=time.time()))
    checks = evaluate(local, optimizer, metric, y, records, points, out, cfg['lr'], args.gate)
    assert local.frozen_digest() == frozen
    for name in ('measurements.npz', 'summary.csv', 'input.json'):
        path = out / name
        manifest.append(dict(path=name, bytes=path.stat().st_size, sha256=T.sha256(path)))
    T.atomic_json(out / 'manifest.json', manifest)
    result = dict(status='complete', config=cfg, checks=checks, frozen_unchanged=True,
                  prefix_bit_equal=True, updated=time.time())
    T.atomic_json(out / 'complete.json', result)
    T.atomic_json(out / 'status.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task', type=int, default=10)
    ap.add_argument('--step', type=int, default=500)
    ap.add_argument('--layer', type=int, choices=(1, 2), required=True)
    ap.add_argument('--mode', choices=('fixed', 'adam'), required=True)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--output', type=Path)
    run(ap.parse_args())
