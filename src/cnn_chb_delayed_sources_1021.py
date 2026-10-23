"""CPU-only qualification of the fixed, archived CHf task-30 sources.

No CIFAR forward pass, optimization, or CUDA initialization belongs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

DEFAULT_SOURCE = Path('/home/issan/Projects/obsidian-research-data/cnn_chprime_bias_1018/host_gpu_recovery_1018/results/cnn_chprime_bias_1018')
DEFAULT_MANIFEST = Path('/home/issan/Projects/claude/wt/cnn_chprime_bias_1018/results/cnn_chprime_bias_1018/backup_manifest.json')
SEEDS = list(range(10, 20))
STATE_SHA = {
    'CHf': '936b3e285a979f97097234dd76c816fa24417e773f5ce2efbf19c6f6c8d0c4c3',
}
DATA_SHA = {'cifar-10-python.tar.gz': '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def tensor_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))))


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_sources(source_root=None, manifest_path=None):
    """Return full outer source states and a compact JSON-serializable manifest."""
    root = Path(source_root or DEFAULT_SOURCE)
    mp = Path(manifest_path or DEFAULT_MANIFEST)
    original = json.loads(mp.read_text())
    require(isinstance(original, list), 'Source archive manifest must be a list')
    by_backup = {}
    for item in original:
        by_backup.setdefault(item['backup'], []).append(item)
    files = []

    def verified(path):
        entries = by_backup.get(str(path), [])
        require(len(entries) == 1, f'Expected exactly one archive entry: {path}')
        entry = entries[0]
        digest = sha_file(path)
        require(path.stat().st_size == entry['bytes'] and digest == entry['sha256'],
                f'Archive bytes/SHA mismatch: {path}')
        files.append({k: entry[k] for k in ('source', 'backup', 'bytes', 'sha256')})
        return digest

    states, arms, provenances, streams = {}, {}, {}, {}
    expected_config = {'seeds': SEEDS, 'tasks': 30, 'epochs': 400, 'batch': 16,
                       'steps_per_task': 30000, 'lr': .001, 'kappa': .2, 'R': 10,
                       'tf32_cudnn': False, 'tf32_matmul': False}
    for arm in STATE_SHA:
        folder = root / arm
        state_path = folder / 'state.pt'
        require(verified(state_path) == STATE_SHA[arm], f'Unregistered source state: {arm}')
        outer = torch.load(state_path, map_location='cpu', weights_only=False)
        require(outer['task'] == 30 and not outer['resume'], f'Not a complete t30 boundary: {arm}')
        config = outer['config']
        for key, value in expected_config.items():
            require(config[key] == value, f'{arm} config mismatch: {key}')
        require(config['arm'] == arm and config['data_sha256'] == DATA_SHA, f'{arm} source/data identity mismatch')
        st = outer['state']
        require(st['tc'] == 900000 and st['kappa'] == .2 and st['enabled'] == (arm == 'CHf_B'),
                f'{arm} optimizer/projection state mismatch')
        require(st['bad'].shape == (10,) and not st['bad'].any(), f'{arm} contains failed slots')
        for key in ('P', 'm', 'v'):
            require(len(st[key]) == 10, f'{arm}: expected 10 parameter tensors in {key}')
            for j, value in enumerate(st[key]):
                require(value.shape[0] == 10 and bool(torch.isfinite(value).all()), f'{arm}/{key}/{j}: invalid tensor')
                if key == 'v':
                    require(bool((value >= 0).all()), f'{arm}: negative Adam variance')
        for key, values in st['act'].items():
            for j, value in enumerate(values):
                require(value.shape[0] == 10, f'{arm}/act/{key}/{j}: wrong slots')
                # ReLU's unused shape parameter fixA is intentionally NaN.
                if key != 'fixA':
                    require(bool(torch.isfinite(value).all()), f'{arm}/act/{key}/{j}: invalid tensor')
        for j, seed in enumerate(SEEDS):
            path = folder / 'ckpt' / f'{arm}_seed{seed}_t30.pt'
            verified(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            require(checkpoint['seed'] == seed and checkpoint['arm'] == arm
                    and checkpoint['tc'] == st['tc'] and checkpoint['kappa'] == st['kappa']
                    and checkpoint['enabled'] == st['enabled'], f'Checkpoint identity mismatch: {path}')
            for key in ('P', 'm', 'v'):
                require(len(checkpoint[key]) == len(st[key]), f'Checkpoint tensor count: {path}/{key}')
                require(all(tensor_equal(a[j], b) for a, b in zip(st[key], checkpoint[key])),
                        f'Checkpoint state mismatch: {path}/{key}')
            require(set(checkpoint['act']) == set(st['act']), f'Checkpoint activation keys: {path}')
            for key, values in st['act'].items():
                require(len(values) == len(checkpoint['act'][key])
                        and all(tensor_equal(a[j], b) for a, b in zip(values, checkpoint['act'][key])),
                        f'Checkpoint activation mismatch: {path}/{key}')
        provenance_path = folder / 'provenance.json'
        verified(provenance_path)
        provenance = json.loads(provenance_path.read_text())
        require(provenance['status'] == 'COMPLETE' and provenance['config'] == config,
                f'{arm}: incomplete or inconsistent provenance')
        require(provenance['torch'] == '2.13.0+cu130', f'{arm}: unexpected source PyTorch')
        stream_path = folder / 'stream_t30.json'
        verified(stream_path)
        stream = json.loads(stream_path.read_text())
        hashes = {key: {str(seed): tensor_sha(st[key][seed]) for seed in SEEDS}
                  for key in ('g_lab', 'g_batch')}
        require(hashes == provenance['final_streams'] == stream['streams_after'],
                f'{arm}: saved RNG differs from provenance/stream trace')
        require(len(stream['batch_order_hashes']) == 400, f'{arm}: incomplete t30 batch trace')
        table_path = folder / 'per_task.csv'
        verified(table_path)
        with table_path.open() as f:
            rows = list(csv.DictReader(f))
        keys = [(int(r['seed']), int(r['task'])) for r in rows]
        expected_keys = {(s, t) for s in SEEDS for t in range(1, 31)}
        require(len(keys) == 300 and set(keys) == expected_keys, f'{arm}: missing/duplicate source rows')
        require(all(r['arm'] == arm and r['diverged'].strip().lower() in ('false', '0', '0.0')
                    for r in rows), f'{arm}: failed/mixed source rows')
        require(len(outer['rows']) == 300, f'{arm}: incomplete checkpoint evaluation history')
        for s in SEEDS:
            subset = provenance['subsets'][str(s)]
            require(len(subset) == len(set(subset)) == 1200 and all(0 <= n < 50000 for n in subset),
                    f'{arm}: invalid fixed subset for seed {s}')
        states[arm] = outer
        provenances[arm] = provenance
        streams[arm] = stream
        arms[arm] = {'state_path': str(state_path), 'state_sha256': STATE_SHA[arm],
                     'provenance_path': str(provenance_path), 'per_task_path': str(table_path),
                     'stream_t30_path': str(stream_path), 'config': config,
                     'source_git_hash': provenance['git_hash'], 'checkpoint_count': 10,
                     'per_task_rows': 300, 'streams': hashes,
                     'subsets_sha256': json_sha(provenance['subsets'])}
    manifest = {'status': 'PASS', 'source_root': str(root), 'seed_order': SEEDS,
                'archive_manifest': {'path': str(mp), 'sha256': sha_file(mp), 'entries': len(original)},
                'files': files, 'arms': arms, 'data_sha256': DATA_SHA,
                'streams_preserved': True, 'subsets_qualified': True,
                'qualification_scope': 'CPU archive/state/checkpoint checks only; no new forward or training'}
    return states, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    _, report = audit_sources(args.source_root, args.manifest_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: 1 source state, 10 checkpoints, complete streams; {len(report['files'])} archived files verified")
