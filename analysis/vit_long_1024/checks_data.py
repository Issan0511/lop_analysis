"""Data checks of vit_long_1024 (spec §8): S-data, S-stream, S-arm1, S-smoke, S-speed.

    python -m analysis.vit_long_1024.checks_data
        -> results/vit_long_1024/checks/checks_data.json (+ checks_data_<host>.json), dataset.json

Two layouts are checked. The default layout (5 labels x 50 images, augmentation, dropout 0.1)
is the factorial N cell's: S-stream compares it bit for bit with randlabel's task stream and
S-smoke records its online accuracy against the N cell's .481/.712/.806. The production layout
(production.json) gets S-stream's loading checks, S-arm1 (R0 and S identical on task 1) and
S-speed. Runs on white-san and again on the rented GPU (remote/run_remote.sh); a passing record
from another host is the reference for the cross-machine records.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

import analysis.randlabel_control_joudaki_vit_0919 as randlabel
from analysis.joudaki_vit_battle_0919.run import atomic_json

from . import common as C
from . import judge as J

PY = os.environ.get('VIT_LONG_PY', sys.executable)
MANIFEST = Path(os.environ.get('VIT_LONG_MANIFEST', str(C.DATA_ROOT.parent / 'tiny-imagenet-200.manifest_1024.sha256')))
ARCHIVES = {   # spec §10.2; values given by the parent on 2026-10-24
    'kaggle_akash2sharma': dict(file='tiny-imagenet-200.kaggle_akash2sharma.zip', bytes=497536564,
                                sha256='05fafc211a347868093af2133504de066329059df16c2d362d068a73859f466e',
                                source='https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet',
                                year=2019, used=True),
    'kaggle_blackfox20092006': dict(file='tiny-imagenet-200.kaggle_blackfox20092006.zip', bytes=246604677,
                                    sha256='7734d6773fcbd8911876539c191b1c9faccbe12f5457ca9b4c92774a90bb76bd',
                                    source='Kaggle blackfox20092006/tiny-imagenetstanford-cs231n', year=2025,
                                    used=False),
}
OFFICIAL_ZIP_SHA256 = '6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'
N_CELL_ONLINE = (.481, .712, .806)        # factorial spec §5, seed 0 tasks 1-3 (another machine)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def check_data():
    root = C.DATA_ROOT
    out = dict(root=str(root), manifest=str(MANIFEST))
    train_dirs = sorted(p for p in (root / 'train').iterdir() if p.is_dir())
    counts = dict(train_classes=len(train_dirs),
                  train_images=sum(len(list((d / 'images').glob('*.JPEG'))) for d in train_dirs),
                  train_classes_with_500=sum(len(list((d / 'images').glob('*.JPEG'))) == 500 for d in train_dirs),
                  val_images=len(list((root / 'val/images').glob('*.JPEG'))),
                  val_annotations=len((root / 'val/val_annotations.txt').read_text().splitlines()),
                  wnids=len((root / 'wnids.txt').read_text().split()))
    out['counts'] = counts
    counts_ok = counts == dict(train_classes=200, train_images=100000, train_classes_with_500=200, val_images=10000,
                               val_annotations=10000, wnids=200)
    lines = [line.split(None, 1) for line in MANIFEST.read_text().splitlines() if line.strip()]
    listed = {path.strip(): digest for digest, path in lines}
    mismatched = [p for p, d in listed.items() if not (root / p).is_file() or file_sha256(root / p) != d]
    on_disk = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
               and p.relative_to(root).parts[0] != 'test'}
    out.update(manifest_lines=len(lines), manifest_sha256=file_sha256(MANIFEST), manifest_mismatched=mismatched[:20],
               manifest_mismatched_count=len(mismatched), files_not_in_manifest=sorted(on_disk - set(listed))[:20],
               files_missing=sorted(set(listed) - on_disk)[:20],
               test_files=sum(1 for p in (root / 'test').rglob('*') if p.is_file()) if (root / 'test').exists() else 0)
    manifest_ok = len(lines) == 110203 and not mismatched and on_disk == set(listed)
    archives = {}
    for name, spec in ARCHIVES.items():
        path = Path(os.environ.get('VIT_LONG_ARCHIVE', '')) if spec['used'] and os.environ.get('VIT_LONG_ARCHIVE') \
            else root.parent / spec['file']
        entry = dict(path=str(path), present=path.exists(), expected_bytes=spec['bytes'], expected_sha256=spec['sha256'],
                     source=spec['source'], year=spec['year'], used=spec['used'])
        if path.exists():
            entry.update(bytes=path.stat().st_size, sha256=file_sha256(path))
            entry['match'] = entry['bytes'] == spec['bytes'] and entry['sha256'] == spec['sha256']
        archives[name] = entry
    out['archives'] = archives
    # Names and CRC-32 of the used (outer) set against the second copy, read from the zip directories.
    a, b = archives['kaggle_akash2sharma'], archives['kaggle_blackfox20092006']
    if a['present'] and b['present']:
        nested = 'tiny-imagenet-200/tiny-imagenet-200/'
        crc_a = {i.filename: i.CRC for i in zipfile.ZipFile(a['path']).infolist()
                 if not i.is_dir() and not i.filename.startswith(nested)}
        crc_b = {i.filename: i.CRC for i in zipfile.ZipFile(b['path']).infolist() if not i.is_dir()}
        out['second_copy_names_and_crc32_equal'] = crc_a == crc_b
        out['second_copy_files'] = len(crc_b)
    out['pass_'] = bool(counts_ok and manifest_ok)
    return out


def dataset_record(data):
    used = data['archives']['kaggle_akash2sharma']
    return dict(
        root=data['root'], source='Kaggle akash2sharma/tiny-imagenet (2019), not the official cs231n zip',
        source_url=ARCHIVES['kaggle_akash2sharma']['source'],
        why='cs231n.stanford.edu refused connections on 443 and 80 on 2026-10-24; Issa chose the Kaggle copy',
        archive=dict(file=ARCHIVES['kaggle_akash2sharma']['file'], bytes=ARCHIVES['kaggle_akash2sharma']['bytes'],
                     sha256=ARCHIVES['kaggle_akash2sharma']['sha256'], verified_here=used.get('match')),
        extracted='outer tiny-imagenet-200/ only; the nested duplicate tiny-imagenet-200/tiny-imagenet-200/ '
                  '(names and CRC-32 all equal) was not extracted',
        manifest=dict(path=data['manifest'], lines=data['manifest_lines'], sha256=data['manifest_sha256'],
                      covers='train, val, wnids.txt, words.txt (not test/)'),
        counts=data['counts'],
        second_copy=dict(source=ARCHIVES['kaggle_blackfox20092006']['source'], year=2025,
                         archive_bytes=ARCHIVES['kaggle_blackfox20092006']['bytes'],
                         archive_sha256=ARCHIVES['kaggle_blackfox20092006']['sha256'], files=120203,
                         reported_by_parent='file names and CRC-32 equal for all files; class n01443537 (501 files) '
                                            'also equal by sha256',
                         names_and_crc32_equal_verified_here=data.get('second_copy_names_and_crc32_equal')),
        caveat='Neither copy can be checked against the official zip sha256 ' + OFFICIAL_ZIP_SHA256 +
               '. Two independent copies have the same content, but whether these are the images of the '
               '0919 N cell cannot be confirmed.',
        verified_on=socket.gethostname(), time=time.time())


def check_stream(reference):
    out, ok = {}, True
    for name, layout in (('default', dict(n_labels=C.N_LABELS, images_per_class=C.PER_CLASS)),
                         ('production', dict(n_labels=PROD['n_labels'], images_per_class=PROD['images_per_class']))):
        part = check_stream_layout(layout, reference.get('S_stream', {}).get(name) if reference else None)
        out[name] = part
        ok &= part['pass_']
    out['pass_'] = bool(ok)
    return out


def check_stream_layout(layout, reference_part):
    source = C.RealImages()
    n_labels, per_class = layout['n_labels'], layout['images_per_class']
    registered = (n_labels, per_class) == (C.N_LABELS, C.PER_CLASS)
    n_images = n_labels * per_class
    out = dict(seeds=list(C.SEEDS), layout=layout, compared_with_randlabel=registered)
    hashes, ok, problems = {}, True, []
    randlabel.DATA = C.DATA_ROOT            # the function reads its module global; same files, possibly moved
    for seed in C.SEEDS:
        seen = set()
        for task in range(1, 62):
            data, plan = C.task_dataset(source, seed, task, n_labels, per_class)
            if len(data) != n_images or data.images.shape != (n_images, 64, 64, 3):
                ok = False
                problems.append(f'seed{seed} task{task}: shape {data.images.shape}')
            if set(data.paths) & seen:
                ok = False
                problems.append(f'seed{seed} task{task}: repeated paths')
            seen |= set(data.paths)
            hashes[f'{seed}:{task}'] = C.array_sha256(data.images) + ':' + C.array_sha256(data.labels.numpy())
            if registered and task <= 40:
                ref, _ = randlabel.make_task_dataset(seed, task, 'cpu')
                same = (np.array_equal(ref.images, data.images) and torch.equal(ref.labels, data.labels)
                        and [str(Path(p).relative_to(C.DATA_ROOT)) for p in ref.paths] == data.paths)
                if not same:
                    ok = False
                    problems.append(f'seed{seed} task{task}: differs from randlabel make_task_dataset')
        names = C.probe_names(source, seed)
        probe = source.val(names)
        ok &= names == C.probe_names(source, seed) and probe.shape == (64, 64, 64, 3)
        hashes[f'{seed}:probe'] = C.array_sha256(probe)
    out.update(pass_=bool(ok), problems=problems[:20], image_hashes=hashes)
    if reference_part and reference_part.get('image_hashes'):
        # Record only: every production run happens on one machine, so its own S-stream decides.
        out['cross_machine_equal'] = reference_part['image_hashes'] == hashes
    return out


PROD = C.production_layout()
PROD_ARGS = ['--n-labels', str(PROD['n_labels']), '--images-per-class', str(PROD['images_per_class']),
             '--augment', PROD['augment'], '--dropout', str(PROD['dropout']), '--steps', str(PROD['steps'])]


def run_job(out, arm, seed, *extra):
    command = [PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--seed', str(seed), '--smoke',
               '--data-kind', 'real', '--out', str(out), *map(str, extra)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'console.log').open('a') as log:
        log.write(' '.join(command) + '\n')
        log.flush()
        subprocess.run(command, cwd=C.REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return J.read_csv(out / 'per_task.csv')


def intask(out):
    with np.load(out / 'intask.npz') as f:
        return {k: f[k] for k in f.files}


def model_sha(out, task):
    return C.state_sha256(torch.load(out / f'ckpt/model_t{task:02d}.pt', map_location='cpu', weights_only=False))


def check_runs(root, reference):
    res = {}
    # Production layout: S-arm1 and S-speed.
    r0_dir, s_dir = root / 'R0_production', root / 'S_production'
    r0p = run_job(r0_dir, 'R0', 0, '--tasks', 3, *PROD_ARGS)
    s = run_job(s_dir, 'S', 0, '--tasks', 2, *PROD_ARGS)
    a, b = intask(r0_dir), intask(s_dir)
    keys = ('online_correct', 'online_loss', 'batch_size', 'mem_correct', 'mem_loss', 'up_norm_med')
    task1 = all(np.array_equal(a[k][0], b[k][0]) for k in keys)
    w1 = model_sha(r0_dir, 1) == model_sha(s_dir, 1)
    task2 = not np.array_equal(a['online_loss'][1], b['online_loss'][1])
    s_head = float(s[1]['head_norm_start']) > 0 and float(s[1]['head_full_norm_start']) > 0
    res['S_arm1'] = dict(pass_=task1 and w1 and task2 and s_head, task1_records_equal=task1, task1_weights_equal=w1,
                         task2_differs=task2, S_task2_head_nonzero_at_u0=s_head, production=PROD,
                         online=dict(R0=[float(r['online_acc']) for r in r0p], S=[float(r['online_acc']) for r in s]),
                         mem_acc_end=dict(R0=[float(r['mem_acc_end']) for r in r0p], S=[float(r['mem_acc_end']) for r in s]),
                         T90=dict(R0=[int(float(r['T90'])) for r in r0p], S=[int(float(r['T90'])) for r in s]))
    res['S_speed'] = speed(r0p, PROD['steps'])
    # Default layout (the N cell's): S-smoke against .481/.712/.806, and eager against compile.
    d_dir, eager_dir = root / 'R0_default', root / 'R0_default_eager'
    r0 = run_job(d_dir, 'R0', 0, '--tasks', 3, '--steps', 500)
    eager = run_job(eager_dir, 'R0', 0, '--tasks', 3, '--steps', 500, '--engine', 'eager', '--save-mode', 'minimal')
    online = [float(r['online_acc']) for r in r0]
    smoke = dict(online=online, mem_acc_end=[float(r['mem_acc_end']) for r in r0], T90=[int(float(r['T90'])) for r in r0],
                 n_cell_online=list(N_CELL_ONLINE), minus_n_cell=[o - n for o, n in zip(online, N_CELL_ONLINE)],
                 eager_online=[float(r['online_acc']) for r in eager],
                 eager_minus_compile=[float(e['online_acc']) - o for e, o in zip(eager, online)],
                 task1_weights_sha256=model_sha(d_dir, 1), pass_=True, layout='default (5 labels x 50, augment on, dropout 0.1)',
                 note='record only: the N cell ran on another machine, and eager differs from compile by rounding')
    if reference and reference.get('S_smoke'):
        smoke['cross_machine_reference'] = reference.get('host')
        smoke['cross_machine_online_equal'] = reference['S_smoke'].get('online') == online
        smoke['cross_machine_task1_weights_equal'] = reference['S_smoke'].get('task1_weights_sha256') == smoke['task1_weights_sha256']
    res['S_smoke'] = smoke
    return res


def speed(rows, steps):
    later = [r for r in rows if int(r['task']) >= 2]
    per_task = {k: float(np.mean([float(r[k]) for r in later])) for k in ('train_seconds', 'record_seconds', 'other_seconds', 'seconds')}
    task1 = float(rows[0]['seconds'])
    prod = C.production()
    runs = len(prod['seeds']) * len(C.ARMS)
    run_hours = (prod['tasks'] * per_task['seconds'] + task1 - per_task['seconds']) / 3600
    fresh_hours = len(prod['seeds']) * len(prod['fresh_tasks']) * task1 / 3600
    return dict(pass_=True, ms_per_update_training=1000 * per_task['train_seconds'] / steps,
                ms_per_update_with_records=1000 * (per_task['train_seconds'] + per_task['record_seconds']) / steps,
                ms_per_update_all=1000 * per_task['seconds'] / steps, seconds_per_task=per_task,
                task1_seconds_with_compile=task1, hours_per_continual_run=run_hours, runs=runs,
                hours_continual_runs=runs * run_hours, hours_fresh=fresh_hours,
                hours_total=runs * run_hours + fresh_hours, production=PROD,
                gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
                basis='tasks 2-3 of the R0 seed 0 run in the production layout; fresh tasks priced as task 1 with compile')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--skip-stream', action='store_true', help='debug only; the record then cannot pass')
    args = parser.parse_args()
    C.set_determinism()
    host = socket.gethostname()
    checks_dir = C.RESULTS / 'checks'
    checks_dir.mkdir(parents=True, exist_ok=True)
    reference = None
    for path in sorted(checks_dir.glob('checks_data_*.json')):
        record = json.loads(path.read_text())
        if record.get('host') != host and record.get('all_pass'):
            reference = record
    root = Path(args.root) if args.root else C.RAW / 'checks' / f'data_{int(time.time())}'
    root.mkdir(parents=True, exist_ok=True)
    results = dict(started=time.time(), root=str(root), host=host, torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), python=platform.python_version(),
                   production=PROD, reference_host=reference.get('host') if reference else None)
    steps = [('S_data', check_data)]
    if not args.skip_stream:
        steps.append(('S_stream', lambda: check_stream(reference)))
    steps.append(('runs', lambda: check_runs(root, reference)))
    for name, fn in steps:
        began = time.time()
        try:
            value = fn()
        except Exception:
            value = dict(pass_=False, error=traceback.format_exc())
        if name == 'runs' and 'error' not in value:
            results.update(value)
        else:
            results[name] = value
        if name == 'S_data' and value.get('pass_'):
            atomic_json(checks_dir / 'dataset.json', dataset_record(value))
        brief = {k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in value.items()} if name in ('runs', 'S_stream') else \
            {k: v for k, v in value.items() if k != 'image_hashes'}
        print(json.dumps({name: brief, 'seconds': round(time.time() - began, 1)}, default=str)[:3000], flush=True)
    required = ('S_data', 'S_stream', 'S_arm1')
    results['passes'] = {k: results[k]['pass_'] for k in required + ('S_smoke', 'S_speed') if k in results}
    results['all_pass'] = all(results.get(k, {}).get('pass_') for k in required)
    results['source_hashes_at_validation'] = C.source_hashes()
    results['finished'] = time.time()
    clean = json.loads(json.dumps(results, default=str).replace('NaN', 'null'))
    atomic_json(checks_dir / 'checks_data.json', clean)
    atomic_json(checks_dir / f'checks_data_{host}.json', clean)
    print(json.dumps(dict(all_pass=results['all_pass'], passes=results['passes'],
                          speed={k: results.get('S_speed', {}).get(k) for k in ('ms_per_update_training', 'ms_per_update_all',
                                                                                  'hours_continual_runs', 'hours_total')})), flush=True)


if __name__ == '__main__':
    main()
