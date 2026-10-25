"""Audit the full registered data matrix before the final scientific report.

An incomplete matrix exits before expensive raw-file checks. Passing this data
audit is not a claim that scientific interpretation or Git integration is done.
"""
import json
from collections import OrderedDict
from pathlib import Path
import subprocess
import time
import common as T
import alignment
import global_noise
import t5_run
from t5_compat import legacy_sources, training_compatible
np, torch = T.np, T.torch
OUTPUT = T.ROOT / 'results/vit_two_block_1025'


def read(path):
    return json.loads(path.read_text())


def expected_cases():
    result = {}
    for kind, starts in (('alignment', (100, 500)), ('global_noise', (20, 100, 500))):
        for seed in T.SEEDS:
            for task in (10, 60):
                for start in starts:
                    result[f'{kind}/seed{seed}_t{task:02d}_s{start:03d}'] = dict(seed=seed, task=task, start=start)
    for seed in T.SEEDS:
        for task in (10, 60):
            for start in (100, 500):
                for layer in (1, 2):
                    for mode in ('fixed', 'adam'):
                        result[f't5/seed{seed}_t{task:02d}_s{start:03d}_L{layer}_{mode}'] = dict(
                            seed=seed, task=task, start=start, layer=layer, mode=mode)
    assert len(result) == 130
    return result


def run():
    OUTPUT.mkdir(exist_ok=True)
    hashes = dict(alignment=alignment.hashes(), global_noise=global_noise.hashes(), t5=t5_run.source_hashes())
    expected = expected_cases()
    missing, finished = [], dict(baseline=0, alignment=0, global_noise=0, t5=0)
    for seed in T.SEEDS:
        path = T.ARCHIVE / f'runs/seed{seed}/stage_complete.json'
        if not path.exists() or read(path).get('through_task', 0) < 60:
            missing.append(f'runs/seed{seed}: through_task=60')
        else:
            assert read(path)['status'] == 'complete'
            finished['baseline'] += 1
    for rel, fields in expected.items():
        marker = T.ARCHIVE / rel / 'complete.json'
        if not marker.exists():
            missing.append(rel)
            continue
        result = read(marker)
        cfg, kind = result['config'], rel.split('/')[0]
        if kind == 't5' and legacy_sources(cfg['source_hashes']):
            missing.append(rel + ': boundary re-evaluation required')
            continue
        assert result['status'] == 'complete'
        assert cfg['source_hashes'] == hashes[kind], rel
        assert all(cfg[k] == v for k, v in fields.items()), rel
        assert cfg['gate'] is False
        finished[kind] += 1
    for kind in ('alignment', 'global_noise', 't5'):
        observed = {str(p.relative_to(T.ARCHIVE)) for p in (T.ARCHIVE / kind).glob('seed*') if p.is_dir()}
        assert observed <= set(expected), sorted(observed - set(expected))
    report = dict(status='incomplete', checked_at=time.time(), completed=finished,
                  expected=dict(baseline=5, alignment=20, global_noise=30, t5=80), missing=missing,
                  source_sha256=T.sha256(Path(__file__)),
                  scope='Data coverage and integrity only; interpretation and Git integration are separate.')
    if missing:
        T.atomic_json(OUTPUT / 'completion_audit.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'missing'}), flush=True)
        return 2

    # Controllers write complete only after every expected child has succeeded.
    for kind in ('', 'alignment', 'global_noise', 't5'):
        assert read(T.ARCHIVE / kind / 'controller_status.json')['status'] == 'complete', kind
    original = T.original_reference()
    gate = read(T.ARCHIVE / 'gate/basic.json')
    assert gate['status'] == 'passed' and gate['source_hashes'] == T.own_hashes()
    assert gate['model_depth'] == 2 and gate['trace_logits_and_preactivations_bit_equal']
    assert gate['recording_state_rng_unchanged']
    resume = read(T.ARCHIVE / 'gate/resume.json')
    assert resume['status'] == 'passed' and resume['engine'] == 'compile'
    assert set(resume['bit_equal']) == {'model', 'optimizer', 'rng', 'rows', 'data_records', 'all_moments'}
    for kind in hashes:
        gate = read(T.ARCHIVE / f'gate/{kind}.json')
        assert gate['status'] == 'passed' and gate['source_hashes'] == hashes[kind]
    tg = read(T.ARCHIVE / 'gate/t5.json')
    assert tg['math']['exact_zero_derivative_tested'] and tg['reread_replicas_unchanged'] and tg['reread_measurements_bit_equal']
    assert {(x['layer'], x['mode']) for x in tg['cases']} == {(l, m) for l in (1, 2) for m in ('fixed', 'adam')}
    ag = read(T.ARCHIVE / 'gate/alignment.json')
    assert ag['restored_initial_state'] and ag['checks']['body_forward_bit_equal']
    gg = read(T.ARCHIVE / 'gate/global_noise.json')
    assert all(gg[k] for k in ('identical_initial_observations', 'non_fc1_updated', 'reread_replicas_unchanged', 'reread_measurements_bit_equal'))

    verified, commits = {}, set()
    def verify(path, expected_hash=None, expected_bytes=None):
        path = path.resolve()
        if path not in verified:
            verified[path] = dict(bytes=path.stat().st_size, sha256=T.sha256(path))
        info = verified[path]
        if expected_hash is not None:
            assert info['sha256'] == expected_hash, path
        if expected_bytes is not None:
            assert info['bytes'] == expected_bytes, path
        return info

    def manifest(root, name='manifest.json'):
        entries = read(root / name)
        paths = [x['path'] for x in entries]
        assert len(paths) == len(set(paths)), root
        for item in entries:
            path = root / item['path']
            assert path.resolve().is_relative_to(root.resolve()), path
            verify(path, item['sha256'], item['bytes'])
        return set(paths)

    for seed in T.SEEDS:
        root = T.ARCHIVE / f'runs/seed{seed}'
        prov = read(root / 'provenance.json')
        assert prov['source_hashes'] == T.own_hashes()
        commits.add(prov['git_hash'])
        files = manifest(root)
        required = {f'moments/t{task:02d}.npz' for task in range(1, 61)}
        required |= {f'ckpt/t{task:02d}_s{step:03d}.pt' for task in range(1, 61) for step in T.snapshot_steps(task)}
        assert files == required, (seed, files ^ required)
        checkpoint = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=False)
        commits.add(checkpoint['git_hash'])
        assert checkpoint['task'] == 60 and checkpoint['step'] == 500
        assert checkpoint['source_hashes'] == T.own_hashes()
        assert checkpoint['original_source_hashes'] == original['source_hashes']
        assert [x['task'] for x in checkpoint['data_records']] == list(range(1, 61))
        assert checkpoint['config']['model_config'] == T.CONFIG and checkpoint['config']['seed'] == seed
        for task in range(1, 61):
            with np.load(root / f'moments/t{task:02d}.npz') as data:
                for step in (0, 1, 5, 20, *range(25, 501, 25)):
                    for pop in ('train', 'validation'):
                        for layer in (1, 2):
                            for field in ('mu', 'sd', 'r', 'valid', 'opened'):
                                a = data[f's{step}_{pop}_L{layer}_{field}']
                                assert a.shape == (1536,) and np.isfinite(a).all()
        del checkpoint

    for rel in expected:
        root, kind = T.ARCHIVE / rel, rel.split('/')[0]
        result = read(root / 'complete.json')
        cfg = result['config']
        checkpoint = T.ARCHIVE / f"runs/seed{cfg['seed']}/ckpt/t{cfg['task']:02d}_s{cfg['start']:03d}.pt"
        assert Path(cfg['checkpoint']) == checkpoint
        verify(checkpoint, cfg['checkpoint_sha256'])
        prov = read(root / 'provenance.json')
        assert prov['config'] == cfg
        commits.add(prov['git_hash'])
        files = manifest(root)
        if kind == 'alignment':
            assert result['restored_initial_state']
            assert files == {'arrays.npz', 'positions.csv', 'finite.csv', 'heads.csv', 'checks.json', 'provenance.json'}
            continue
        assert (cfg['replicas'], cfg['steps'], cfg['batch'], cfg['lr'], cfg['points']) == (32, 50, 128, 1e-4, [0, 1, 5, 20, 50])
        raw = {'full.pt'} | {f'replica_{i:03d}.pt' for i in range(32)}
        assert {x for x in files if x.endswith('.pt')} == raw
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        commits.add(state['git_hash'])
        if kind == 't5':
            prefix = f"layers.block_{cfg['layer'] - 1}.layers.mlp.layers.fc1."
            initial = T.digest_tree(dict(weight=state['model'][prefix + 'weight'], bias=state['model'][prefix + 'bias']))
            frozen = T.digest_tree({k: v for k, v in state['model'].items() if not k.startswith(prefix)})
            assert read(root / 'input.json')['frozen_digest'] == frozen
        else:
            # cpu_tree serializes state_dict as a plain dict; the runtime digest
            # was computed from the model's OrderedDict, including its type.
            initial = T.digest_tree(OrderedDict(state['model']))
        for replica in range(-1, 32):
            name = 'full.pt' if replica == -1 else f'replica_{replica:03d}.pt'
            record = torch.load(root / name, map_location='cpu', weights_only=False)
            assert record['replica'] == replica
            assert training_compatible(record['config'], cfg) if kind == 't5' else record['config'] == cfg
            layer = cfg['layer'] if kind == 't5' else 1
            gen = torch.Generator().manual_seed(T.C.stream_seed('t5_replica', cfg['seed'], cfg['task'], cfg['start'], layer, replica))
            indices = [torch.arange(250) if replica == -1 else torch.randint(250, (128,), generator=gen) for _ in range(50)]
            assert T.digest_tree(indices) == record['indices_sha256'], (rel, replica)
            if kind == 't5':
                assert sorted(record['states']) == cfg['points']
                digest = T.digest_tree(record['states'][0])
                assert record['frozen_unchanged'] and record['prefix_bit_equal_after_updates']
                assert all(torch.isfinite(v).all() for snapshot in record['states'].values() for v in snapshot.values())
            else:
                digest = record['initial_digest']
                assert record['non_fc1_changed']
                assert record['model'].keys() == state['model'].keys()
                changed = {k for k, v in record['model'].items()
                           if isinstance(v, torch.Tensor) and '.fc1.' not in k and not torch.equal(v, state['model'][k])}
                assert changed == set(record['non_fc1_changed'])
                for key, value in record['model'].items():
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all()
                    else:
                        assert T.digest_tree(value) == T.digest_tree(state['model'][key])
            assert digest == initial
            del record
        del state
        if kind == 't5':
            assert result['frozen_unchanged'] and result['prefix_bit_equal']
            assert [c['step'] for c in result['checks']] == cfg['points']
            assert all(c['direct_max_abs_error'] < c['tolerance'] and c['identity_max_abs_error'] < 1e-11 for c in result['checks'])
            backup = root / 'before_clamp_boundary_fix'
            if backup.exists():
                manifest(backup, 'preservation_manifest.json')
                commits.add(read(backup / 'provenance.json')['git_hash'])
        with np.load(root / 'measurements.npz') as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
        print(json.dumps(dict(verified_case=rel)), flush=True)

    for commit in commits:
        subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=T.ROOT, check=True)
    correction = read(OUTPUT / 't5_correction_inputs.json')
    assert correction['legacy_conditions'] == correction['legacy_revalidated'] == 14
    for entry in correction['inputs']:
        verify(Path(entry['original_arrays']), entry['original_sha256'])
        verify(Path(entry['corrected_arrays']), entry['corrected_sha256'])
    # Everything was created in its permanent archive, including old measurements
    # and failed-run diagnostics. Inventory those too, rather than deleting them.
    inventory = []
    for path in sorted(T.ARCHIVE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            info = verify(path)
            inventory.append(dict(source=str(path), backup=str(path), **info, created_in_archive=True))
    T.atomic_json(OUTPUT / 'backup_manifest.json', inventory)
    report.update(status='passed_data_audit', verified_files=len(verified),
                  archive_files=len(inventory), archive_bytes=sum(x['bytes'] for x in inventory),
                  provenance_commits=sorted(commits), finished_at=time.time())
    T.atomic_json(OUTPUT / 'completion_audit.json', report)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
