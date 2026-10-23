"""Shared identities and immutable artifact checks for the NEW RC 200-task run.

The old F51 run is historical evidence and is never read as this run's data.
No function in this module trains or falls back to another device/protocol.
"""
from __future__ import annotations
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

RUN = 'redo_chb_rlcifar_1018'
BASE = '094489c30bcdf196946179ba07ed0e8cadd3a53f'
PREREG_COMMIT = '09d175aa9a86dd8d70c2c8542991b245f754c7bb'
INITIAL_PREREG_COMMIT = 'de8a437fef35f7942023fc581e81f4e6cbbad5bd'
SPEC = ROOT / ('specs/spec_' + RUN + '.md')
COMMON_SPEC = ROOT / 'specs/spec_six_runs_common_1018.md'
OUT = ROOT / 'results' / RUN
ARMS = ('none', 'redo', 'CHB', 'both')
SEEDS = list(range(10, 20))
CHECK_SEEDS = list(range(100, 110))
EXPECTED_EVENTS = 3846
PROTOCOL = dict(run_id=RUN, arms=list(ARMS), seeds=SEEDS, conds=['raw'], R=10,
                n_tasks=200, epochs=400, steps_per_epoch=75, steps_per_task=30000,
                total_updates=6000000, batch=16, n_images=1200, dims=[3072, 100, 100, 10],
                device='cuda', dtype='float32', lr=.001, lam=.1713, beta=.01,
                adam_betas=[.9, .999], adam_eps=1e-8, threads=2,
                tf32_matmul=False, tf32_cudnn=False, deterministic=True,
                redo_tau=.1, redo_period=1560, redo_probe_batches=4, redo_probe_images=64,
                redo_events_per_arm=3846, redo_stream='b5_redo_reinit',
                redo_order='native Adam then B then EMA then one-shot recycle outside graph',
                retain=['bias moments', 'Adam global time', 'hidden EMA'],
                score='actual delivered features after H', late_window=[181, 200], early_window=[31, 50])
PREDICTION = dict(primary='CHB_POLICY_CARRIES', redo_shortfall=True, numeric_probability=None,
                  attribution='Issa 2026-10-18 02:36:16 JST revision formalized by Codex',
                  known_reference='old CHB/none results informed design; these are not new predictive successes')
GPU_LOCK = Path('/tmp/lop_analysis_gpu.lock')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    """Typed tree hash includes raw tensor bytes, preserving signed-zero differences."""
    h = hashlib.sha256()
    def feed(x):
        if isinstance(x, torch.Tensor):
            q = x.detach().cpu().contiguous()
            h.update(repr(('tensor', str(q.dtype), tuple(q.shape))).encode())
            h.update(q.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(repr(('array', str(x.dtype), tuple(x.shape))).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b'dict{')
            for k in sorted(x, key=lambda a: (type(a).__name__, repr(a))):
                feed(k); feed(x[k])
            h.update(b'}')
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b'[')
            for y in x: feed(y)
            h.update(b']')
        else:
            h.update((type(x).__name__ + ':' + repr(x) + ';').encode())
    feed(value)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def import_closure(initial):
    """Deterministic repository Python closure, independent of entry-point imports."""
    seen, pending = set(), list(initial)
    while pending:
        path = Path(pending.pop()).resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(ROOT):
            continue
        seen.add(path)
        if path.suffix != '.py':
            continue
        for parent in path.parents:
            if not parent.is_relative_to(ROOT):
                break
            initializer = parent/'__init__.py'
            if initializer.is_file() and initializer not in seen:
                pending.append(initializer)
            if parent == ROOT:
                break
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    package = list(path.parent.relative_to(ROOT).parts)
                    assert node.level <= len(package), 'relative import escaped repository package'
                    base = package[:len(package)-node.level+1]
                    prefix = '.'.join(base + ([node.module] if node.module else []))
                if prefix:
                    modules = [prefix]
                    modules += [prefix + '.' + v.name for v in node.names if v.name != '*']
            for name in modules:
                if name.split('.')[0] not in ('src', 'analysis'):
                    continue
                p = ROOT.joinpath(*name.split('.'))
                for candidate in (p.with_suffix('.py'), p / '__init__.py'):
                    if candidate.is_file() and candidate not in seen:
                        pending.append(candidate)
    return seen


def source_hashes():
    own = [ROOT / ('src/' + RUN + '.py'), SPEC, COMMON_SPEC]
    own += [ROOT/'analysis'/RUN/(name+'.py') for name in ('common','checks','diagnostics','gpu_checks','launch','report')]
    own += sorted((ROOT / 'analysis' / RUN).glob('*.py'))
    required = [ROOT / p for p in ('src/relu_doors_0919.py', 'src/baselines_cifar5p1_1008.py')]
    assert all(p.is_file() for p in own + required), 'implementation/source/spec missing'
    paths = import_closure(own + required)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def registration():
    blob = subprocess.check_output(['git', 'show', PREREG_COMMIT + ':' + str(SPEC.relative_to(ROOT))], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == sha(SPEC), 'preregistered spec changed'
    for ancestor in (BASE, PREREG_COMMIT):
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, 'HEAD'],
                              cwd=ROOT, capture_output=True).returncode == 0, 'missing registered ancestor'
    return dict(run_id=RUN, base_commit=BASE, prereg_commit=PREREG_COMMIT,
                initial_prereg_commit=INITIAL_PREREG_COMMIT,
                spec_sha256=sha(SPEC), common_spec_sha256=sha(COMMON_SPEC), protocol=PROTOCOL,
                prediction=PREDICTION)


def safe_environment():
    avail = torch.cuda.is_available()
    smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True) if shutil.which('nvidia-smi') else None
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda_build=torch.version.cuda,
                cuda_available=avail, nvidia_devices=sorted(str(p) for p in Path('/dev').glob('nvidia*')),
                nvidia_smi_returncode=None if smi is None else smi.returncode,
                nvidia_smi_output=None if smi is None else (smi.stdout + smi.stderr).strip(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                deterministic=torch.are_deterministic_algorithms_enabled(), threads=torch.get_num_threads(),
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def verify_hash_record(record, *, scope=None):
    assert record.get('all_pass') is True, 'proof did not pass'
    assert record.get('source_sha256') == source_hashes(), 'proof source closure is stale'
    if scope is not None:
        assert record.get('scope') == scope, 'proof scope mismatch'


def relative_file(root, name):
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts, 'unsafe manifest relative path'
    p = Path(root) / rel
    assert p.resolve().is_relative_to(Path(root).resolve()), 'manifest path escaped run directory'
    return p


def require_run_namespace(out):
    path=Path(out).resolve()
    assert path==OUT.resolve() or path.is_relative_to(OUT.resolve()), 'refuse writes outside NEW run namespace'
    return path


def manifest(root, *, exclude=()):
    root = Path(root)
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(root.rglob('*')) if p.is_file() and str(p.relative_to(root)) not in exclude}


def verify_manifest(root, files):
    assert isinstance(files, dict) and files, 'empty artifact manifest'
    for name, item in files.items():
        p = relative_file(root, name)
        assert p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], ('artifact changed', name)


def proof_identity(record):
    return dict(registration=registration(), source_sha256=source_hashes(),
                protocol=PROTOCOL, data=record)
