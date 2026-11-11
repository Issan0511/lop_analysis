"""Independent, CPU-only byte/provenance/paired-stream audit; no model execution.

Use only after completion handoff. --execution-commit is an external identity
anchor, rather than a value inferred from the files being audited. Tensor files
are trusted local experiment checkpoints and are loaded only on CPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch

from . import analyze as A

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/issan/Projects/obsidian-research-data')
MNIST = Path('/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz')
EMPTY = hashlib.sha256(b'').hexdigest()
EXEC_FILES = (
    'analysis/elu_specificity_1112/common.py', 'analysis/elu_specificity_1112/mlp.py',
    'analysis/elu_specificity_1112/vit.py', 'analysis/loading_bridge_1111/dynamic.py',
    'analysis/loading_bridge_1111/static_probe.py', 'analysis/p_residual_loading_1111/run.py',
    'analysis/readout_separation_1111/ops.py', 'analysis/readout_separation_1111/run.py',
    'src/stage1_ctrl_1111.py', 'src/rlcifar_mlp_battle_0918.py', 'src/pmnist_0905.py',
    'src/pmnist_rlcifar_0907.py', 'src/pmnist_rlmnist_0906.py',
    'analysis/vit_long_1024/common.py', 'analysis/vit_long_1024/production_2blk_rlfix.json',
    'analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=A.no_duplicate_keys,
                      parse_constant=A.reject_constant)


def digest(value):
    """Reimplement MLP/readout tensor digest, without importing model runners."""
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode()); add(x[key])
        elif isinstance(x, (list, tuple)):
            for child in x: add(child)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def shared_digest(value):
    """common.py's typed digest; used here only for scalar code-hash mappings."""
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, dict):
            for key in sorted(x, key=str): add(key); add(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for child in x: add(child)
        else:
            h.update(type(x).__name__.encode()); h.update(repr(x).encode()); h.update(b'\0')
    add(value)
    return h.hexdigest()


def tensor_finite(tree):
    if torch.is_tensor(tree):
        need(tree.device.type == 'cpu', 'Audit loaded a non-CPU tensor')
        need(bool(torch.isfinite(tree).all()), 'Nonfinite source tensor')
    elif isinstance(tree, np.ndarray):
        need(bool(np.isfinite(tree).all()), 'Nonfinite source array')
    elif isinstance(tree, dict):
        for child in tree.values(): tensor_finite(child)
    elif isinstance(tree, (list, tuple)):
        for child in tree: tensor_finite(child)
    elif isinstance(tree, (float, np.floating)):
        need(math.isfinite(tree), 'Nonfinite source scalar')


def canonical_path(value):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts, f'Not an exact absolute path: {value}')
    return p


class Resolver:
    """Resolve only exact manifest entries and validate bytes before using them."""
    def __init__(self, manifests=()):
        self.mapping, self.cache, self.verified = {}, {}, {}
        for filename in manifests:
            doc = load_json(filename)
            source_root, backup_root = canonical_path(doc['source_root']), canonical_path(doc['backup_root'])
            for item in doc['files']:
                source, backup = canonical_path(item['source']), canonical_path(item['backup'])
                need(source.is_relative_to(source_root), 'Archive source outside source_root')
                need(backup == backup_root/source.relative_to(source_root), 'Archive destination is not exact relative mapping')
                need(type(item['bytes']) is int and item['bytes'] >= 0, 'Invalid archive byte count')
                need(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'Invalid archive hash')
                entry = dict(source=str(source), backup=str(backup), bytes=item['bytes'], sha256=item['sha256'])
                prior = self.mapping.setdefault(str(source), entry)
                need(prior == entry, 'Conflicting archive mapping')
        destinations = [e['backup'] for e in self.mapping.values()]
        need(len(destinations) == len(set(destinations)), 'Aliased archive destinations')

    def file_hash(self, path):
        path = Path(path)
        need(path.is_file(), f'Missing referenced file: {path}')
        before = path.stat()
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        key = str(path)
        if key not in self.cache or self.cache[key][0] != stamp:
            value = A.sha256(path)
            after = path.stat()
            need(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 f'File changed while hashing: {path}')
            self.cache[key] = (stamp, value)
        return self.cache[key][1]

    def resolve(self, original):
        original = canonical_path(original)
        item = self.mapping.get(str(original))
        if item is not None:
            backup = Path(item['backup'])
            need(backup.is_file(), f'Missing archive backup: {backup}')
            need(backup.stat().st_size == item['bytes'] and self.file_hash(backup) == item['sha256'],
                 f'Corrupt archive backup: {backup}')
            if original.exists():
                need(original.stat().st_size == item['bytes'] and self.file_hash(original) == item['sha256'],
                     f'Original/archive disagreement: {original}')
            return original if original.exists() else backup
        need(original.is_file(), f'No exact archive mapping for missing file: {original}')
        return original

    def verify(self, original, expected):
        need(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Malformed expected SHA256')
        path = self.resolve(original)
        actual = self.file_hash(path)
        need(actual == expected, f'Source bytes do not match recorded SHA256: {original}')
        prior = self.verified.setdefault(str(original), expected)
        need(prior == expected, f'Conflicting source SHA256: {original}')
        return path

    def audit_manifest(self):
        for source in self.mapping: self.resolve(source)


class CommitAnchor:
    def __init__(self, root, commit):
        self.root = Path(root)
        self.commit = self.git('rev-parse', '--verify', f'{commit}^{{commit}}').decode().strip()
        paths = self.git('ls-tree', '-r', '--name-only', self.commit).decode().splitlines()
        inventory = set(EXEC_FILES) | {p for p in paths if
            p.startswith('analysis/joudaki_vit_battle_0919/') and p.endswith('.py')}
        need(inventory <= set(paths), 'Frozen commit lacks required execution modules')
        self.code = {p: hashlib.sha256(self.blob(p)).hexdigest() for p in sorted(inventory)}
        self.spec = hashlib.sha256(self.blob('specs/spec_elu_specificity_1112.md')).hexdigest()
        self.code_digest = shared_digest(self.code)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.PIPE)

    def blob(self, relative):
        p = Path(relative)
        need(not p.is_absolute() and '..' not in p.parts, 'Unsafe code inventory path')
        return self.git('show', f'{self.commit}:{relative}')

    def provenance(self, prov, resolver):
        need(prov['execution_git_hash'] == self.commit, 'Execution commit differs from external anchor')
        need(prov.get('git_hash', self.commit) == self.commit, 'Conflicting execution commit alias')
        need(prov['code_hashes'] == self.code, 'Execution file inventory/hash differs from frozen commit')
        need(prov['code_sha256'] == self.code_digest, 'Invalid combined execution-code digest')
        need(prov['spec_sha256'] == self.spec, 'Specification differs from frozen commit')
        for name in ('local_code_hashes', 'helpers'):
            for path, value in prov.get(name, {}).items():
                need(hashlib.sha256(self.blob(path)).hexdigest() == value, f'Helper hash mismatch: {path}')
        sources = prov['sources']
        need(isinstance(sources, dict) and bool(sources), 'Empty provenance sources')
        for path, expected in sources.items(): resolver.verify(path, expected)


def private_generator(key, modulo=False):
    raw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    value = raw % (2**63-1) if modulo else raw & (2**63-1)
    return torch.Generator().manual_seed(value)


@lru_cache(maxsize=10)
def mlp_streams(seed):
    # The historical expression multiplies by reciprocal, not divides: retain
    # the exact float32 evaluation order when regenerating reference weights.
    g = private_generator(f'pmnist_0905|init|{seed}')
    initial = []
    for din, dout in ((784, 100), (100, 10)):
        initial += [(torch.rand(dout, din, generator=g)*2-1)*(1/math.sqrt(din)),
                    (torch.rand(dout, generator=g)*2-1)*(1/math.sqrt(din))]
    gl = private_generator(f'pmnist_0905|env_labels_0913|{seed}')
    gb = private_generator(f'pmnist_0905|env_batch_0913|{seed}')
    labels, orders = {}, {}
    for task in range(1, 12):
        labels[task] = torch.randint(10, (1200,), generator=gl)
        orders[task] = torch.stack([torch.randperm(1200, generator=gb) for _ in range(54)])
        orders[task] = orders[task].flatten()[:64000].reshape(4000, 16)
    return initial, labels, orders


@lru_cache(maxsize=1)
def mnist_images(path):
    raw = gzip.open(path, 'rb').read()
    need(raw[:4] == bytes((0, 0, 8, 3)), 'Wrong MNIST IDX type/dimensions')
    shape = tuple(int.from_bytes(raw[i:i+4], 'big') for i in (4, 8, 12))
    need(shape == (60000, 28, 28), 'Wrong MNIST image shape')
    data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    need(data.size == 60000*784, 'Truncated MNIST IDX')
    return data.reshape(60000, 784)


def snapshot_fingerprints(snapshot):
    params, state = snapshot['params_and_adam']
    body_ix = [0, 1] + list(range(4, len(params)))
    return dict(body=digest([params[i] for i in body_ix]), head=digest(params[2:4]),
        body_adam=digest([state[i] or {} for i in body_ix]),
        head_adam=digest([state[i] or {} for i in (2, 3)]), rng=digest(snapshot['torch_rng']))


def check_snapshot(snapshot, ln, steps):
    params, state = snapshot['params_and_adam']
    shapes = [(100, 784), (100,), (10, 100), (10,)] + ([(100,), (100,)] if ln == 'on' else [])
    need(len(params) == len(shapes) and len(state) == len(shapes), 'Wrong MLP parameter/state count')
    for p, slot, shape in zip(params, state, shapes):
        need(tuple(p.shape) == shape and p.dtype == torch.float32, 'Wrong MLP parameter shape/dtype')
        if steps == 0:
            need(not slot, 'Initial Adam is not empty')
        else:
            need(set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Incomplete/noncanonical Adam state')
            need(float(slot['step']) == steps, 'Wrong history Adam step count')
            need(tuple(slot['exp_avg'].shape) == shape and tuple(slot['exp_avg_sq'].shape) == shape,
                 'Wrong history Adam moment shape')
            need(bool((slot['exp_avg_sq'] >= 0).all()), 'Negative second Adam moment')
    tensor_finite(snapshot)


def equal_tensors(a, b, message):
    need(a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b), message)


def audit_mlp_source(source, meta, original, source_hash, resolver, anchor, data_root=DATA, mnist=MNIST):
    seed, act, ln = source['seed'], source['activation'], source['ln']
    need(source['complete'] is True and source['study'] == 'A' and source['model'] == 'mlp', 'Wrong MLP source identity')
    need(seed in range(10) and act in A.ACTS_A and ln in ('off', 'on'), 'Invalid MLP source factor')
    need(Path(original).name == f'source_mlp_{act}_{ln}_s{seed}.pt', 'Spoofed MLP source filename/identity')
    need(source['tasks'] == [1, 10] and source['history_steps'] == 4000 and source['n_examples'] == 1200,
         'Incomplete/wrong MLP source history')
    tensor_finite(source)
    projected = {k: v for k, v in source.items() if k not in ('initial', 'snapshots', 'inputs', 'subset', 'labels', 'orders')}
    projected.update(source_sha256=source_hash, serialization_exact=True)
    # JSON changes integer dict keys to strings; compare after canonical JSON conversion.
    need(json.loads(json.dumps(projected)) == meta, 'Source JSON sidecar does not match source PT')
    anchor.provenance(source['provenance'], resolver)
    old_path = data_root/'stage1_ctrl_1111/results/stage1_ctrl_1111/raw'/f'states_adam_lr0.001_{act}_s{seed}.npz'
    need(set(source['provenance']['sources']) == {str(old_path), str(mnist)}, 'Wrong MLP historical/MNIST source paths')
    old = np.load(resolver.resolve(old_path), allow_pickle=False)
    initial, labels, orders = mlp_streams(seed)
    check_snapshot(source['initial'], ln, 0)
    for i, expected in enumerate(initial):
        equal_tensors(source['initial']['params_and_adam'][0][i], expected, 'Initial common weights not canonical')
        need(np.array_equal(old[f'init{i}'], expected.numpy()), 'Archived NPZ has different initialization')
    if ln == 'on':
        params = source['initial']['params_and_adam'][0]
        need(torch.equal(params[4], torch.ones(100)) and torch.equal(params[5], torch.zeros(100)), 'Wrong initial LN affine')
    need(digest(initial) == source['initial_common_sha256'], 'Initial common digest spoof')
    need(digest(source['initial']) == source['initial_state_sha256'], 'Initial state digest spoof')
    subset = torch.randperm(60000, generator=private_generator(f'pmnist_0905|rl_subset|{seed}'))[:1200]
    equal_tensors(source['subset'], subset, 'Different canonical MNIST subset')
    need(np.array_equal(old['subset'], subset.numpy()), 'Archived NPZ subset mismatch')
    pixels = mnist_images(str(resolver.resolve(mnist)))
    inputs = torch.tensor(pixels[subset.numpy()].astype(np.float32)/255, dtype=torch.float32)
    equal_tensors(source['inputs'], inputs, 'Saved inputs differ from source MNIST bytes')
    need(digest(inputs) == source['inputs_sha256'] and digest(subset) == source['subset_sha256'], 'Input/subset digest spoof')
    need(set(source['labels']) == set(range(1, 12)) and set(source['orders']) == set(range(1, 12)), 'Missing task stream')
    for t in range(1, 12):
        equal_tensors(source['labels'][t], labels[t], f'Wrong canonical labels task{t}')
        equal_tensors(source['orders'][t], orders[t], f'Wrong canonical batches task{t}')
        need(np.array_equal(old['y'][t-1], labels[t].numpy()), 'Archived label stream differs')
        need(source['labels_sha256'][t] == digest(labels[t]) and source['batch_sha256'][t] == digest(orders[t]),
             'Task-stream digest spoof')
    need(set(source['snapshots']) == {1, 10}, 'Wrong saved source ages')
    for t, snapshot in source['snapshots'].items():
        check_snapshot(snapshot, ln, t*4000)
        need(source['snapshot_sha256'][t] == digest(snapshot), 'Snapshot digest spoof')
        if ln == 'off':
            for i, p in enumerate(snapshot['params_and_adam'][0]):
                need(np.array_equal(p.numpy(), old[f'end{i}'][t-1]), 'No-LN saved weights differ from archived replay')
    old.close()
    need([q['task'] for q in source['history']] == list(range(1, 11)), 'Incomplete source history records')
    repeat = source['deterministic_repeat']
    need(repeat == dict(updates_per_repeat=3, repeats=2, exact=True, state_and_rng_restored=True), 'Failed repeat/restore check')
    if ln == 'off':
        need([q['task'] for q in source['replay_checks']] == list(range(1, 11)), 'Missing replay checks')
        for q in source['replay_checks']:
            need(q['parameters_bit_exact'] is True and q['labels_bit_exact'] is True and q['maxabs'] == 0,
                 'Failed no-LN history replay check')
    else:
        need(source['replay_checks'] == [], 'LN-on source incorrectly claims archived replay')
    return dict(source=source, original=str(original), hash=source_hash,
                fingerprints={t: snapshot_fingerprints(s) for t, s in source['snapshots'].items()},
                initial_fingerprints=snapshot_fingerprints(source['initial']))


def check_geometry(checks):
    need(checks['P_exact_column_multisets'] is True, 'P does not preserve column multisets')
    need(0 <= checks['O_orthogonal_maxabs'] < 1e-12 and 0 <= checks['O_ones_maxabs'] < 1e-12, 'Orthogonal control failed')
    need(0 <= checks['O_gram_relative_error'] < 2e-6, 'Orthogonal Gram invariant failed')
    for arm in ('S1', 'S01', 'P1', 'P01', 'O1', 'O01'):
        need(0 <= checks[arm+'_centered_norm_relative_error'] < 2e-6, 'Readout centered norm invariant failed')
        need(checks[arm+'_mean_logits_maxabs'] >= 0, 'Invalid mean-logit residual')


def permutation(seed, study, age):
    key = (f'stage1_ctrl_1111|arm|P|adam|{age+1}|{seed}' if study == 'A' else
           f'loading_bridge_1111|P|ViT|SL|{seed}|{age}')
    g = private_generator(key)
    return torch.stack([torch.randperm(10, generator=g) for _ in range(100 if study == 'A' else 384)], 1)


def audit_mlp_curve(r, info, resolver, data_root=DATA):
    s, c, arm, age = info['source'], r['checks'], r['arm'], r['t_end']
    need((r['activation'], r['ln'], r['seed']) == (s['activation'], s['ln'], s['seed']), 'Curve/source factors differ')
    need(r['source_id'] == f"mlp_{s['activation']}_{s['ln']}_s{s['seed']}_t{age}", 'Spoofed source ID')
    need(r['source_sha256'] == info['hash'] and r['snapshot_sha256'] == s['snapshot_sha256'][age], 'Wrong curve source/snapshot hash')
    need(r['initial_state_sha256'] == s['initial_state_sha256'], 'Wrong fresh-body initial source')
    need(r['source_provenance'] == s['provenance'], 'Curve/source provenance mismatch')
    need(c['before'] == info['fingerprints'][age], 'Curve before fingerprint differs from source tensors')
    need(c['source_state_exact'] is True and c['head_adam_cleared'] is True, 'Failed source/head-memory check')
    need(c['body_adam_cleared'] is (arm in ('Z_clear', 'F0')) and c['fresh_body'] is (arm == 'F0'), 'Wrong body-memory/fresh policy')
    after = c['after']
    need(after['head_adam'] == EMPTY, 'Head Adam was retained')
    need(after['rng'] == c['before']['rng'], 'Fork RNG differs')
    want_body = info['initial_fingerprints']['body'] if arm == 'F0' else c['before']['body']
    need(after['body'] == want_body, 'Wrong learned/fresh body at fork')
    need(after['body_adam'] == (EMPTY if arm in ('Z_clear', 'F0') else c['before']['body_adam']), 'Wrong retained/cleared body Adam')
    if arm == 'S1': need(after['head'] == c['before']['head'], 'S1 is not an exact head no-op')
    if arm.startswith('Z') or arm == 'F0':
        need(c['head_exact_zero'] is True, 'Zero-head flag failed')
        params = s['initial']['params_and_adam'][0]
        need(after['head'] == digest([torch.zeros_like(params[2]), torch.zeros_like(params[3])]), 'Head-zero fingerprint is not actually zero')
    else:
        q = c['mean_logits']
        need(q['preserved'] is True and 0 <= q['maxabs'] <= q['tolerance'], 'Mean-logit compensation check failed')
    need(c['batch_sha256'] == s['batch_sha256'][age+1] and c['labels_sha256'] == s['labels_sha256'][age+1], 'Curve uses a different task stream')
    need(c['canonical_permutation_sha256'] == digest(permutation(r['seed'], 'A', age).tolist()), 'Noncanonical P permutation')
    check_geometry(c['geometry'])
    # Verify the legacy comparison against bytes, not the stored exact=True flag.
    expected_paths = set()
    if r['ln'] == 'off' and r['activation'] == 'ELU' and age == 10 and arm in ('S1','S01','P1','P01'):
        expected_paths.add(str(data_root/'readout_separation_1111/results/readout_separation_1111/raw'/
            f"mlp_ELU_s{r['seed']}_t10_physical_{arm}_body_keep.json"))
    if r['ln'] == 'off' and r['activation'] in ('ELU','R','SILU') and arm in ('S1','P1','Z_keep'):
        direction, policy = ('P' if arm == 'P1' else 'S'), ('ZC' if arm == 'Z_keep' else 'KC')
        expected_paths.add(str(data_root/'p_residual_loading_1111/results/p_residual_loading_1111/raw'/
            f"mlp_{r['activation']}_S_s{r['seed']}_t{age}_{direction}_tau0_{policy}.json"))
    need(len(c['legacy']) == len(expected_paths) and {q['path'] for q in c['legacy']} == expected_paths,
         'Missing/extra/wrong registered legacy source')
    for legacy in c['legacy']:
        old = load_json(resolver.verify(legacy['path'], legacy['sha256']))
        rows = {q['step']: q for q in old['curve']}
        pairs = [(q, rows[q['step']]) for q in r['curve'] if q['step'] in rows]
        need(len(pairs) == legacy['points'] and len(pairs) > 1, 'Wrong legacy paired-point count')
        need(legacy['exact'] is True and legacy['maxabs'] == 0, 'Failed legacy exact check')
        need(all(q['acc'] == old['acc'] and q['ce'] == old['ce'] for q, old in pairs), 'Legacy curve values differ')


@lru_cache(maxsize=5)
def vit_streams(seed):
    labels = {t: torch.randint(0, 10, (2500,), generator=private_generator(f'randlabel|{seed}|{t}', True))
              for t in (30, 31)}
    g = private_generator(f'batch|{seed}|31', True)
    batches = []
    while len(batches) < 1500:
        order = torch.randperm(2500, generator=g)
        batches.extend(order[i:i+128].clone() for i in range(0, 2500, 128))
    order = list(range(200))
    random.Random(int.from_bytes(hashlib.sha256(f'cycle|{seed}|5'.encode()).digest()[:8], 'little') % (2**63-1)).shuffle(order)
    plan = dict(task=31, cycle=5, position=0, n_labels=10, fixed=True, classes=order[:10], first=250, count=250)
    return dict(labels_old_sha256=digest(labels[30]), labels_new_sha256=digest(labels[31]),
                batch_order_sha256=digest(batches[:1500]), data_plan=plan)


def vit_state_fingerprints(model, adam):
    need(adam['task'] == 30, 'ViT Adam checkpoint has wrong task')
    optim = adam['optimizer']
    need(len(optim['param_groups']) == 1, 'Unexpected ViT optimizer parameter groups')
    ids = optim['param_groups'][0]['params']
    # BattleActivation serializes {'rng': None} as _extra_state even for
    # non-stochastic R/ELU. These records are part of state_dict/source hashes,
    # but not parameters and therefore have no optimizer parameter ID.
    extras = {f'layers.block_{i}.layers.mlp.layers.act._extra_state' for i in range(2)}
    for name, value in model.items():
        if not torch.is_tensor(value):
            need(name in extras and value == {'rng': None}, 'Unexpected ViT nonparameter extra state')
    names = [name for name, value in model.items() if torch.is_tensor(value)]
    need(len(names) == len(ids), 'State dict has unaccounted buffers/parameter ordering')
    for name, i in zip(names, ids):
        slot = optim['state'].get(i)
        if slot:
            need(tuple(slot['exp_avg'].shape) == tuple(model[name].shape) and
                 tuple(slot['exp_avg_sq'].shape) == tuple(model[name].shape), 'ViT Adam parameter order/shape mismatch')
    need('layers.out.weight' in model and 'layers.out.bias' in model, 'Missing ViT head tensors')
    head_names = ['layers.out.weight', 'layers.out.bias']
    head_ids = [ids[names.index(n)] for n in head_names]
    body_ids = [i for i, name in zip(ids, names) if name not in head_names]
    body = [model[name] for name in names if name not in head_names]
    head = [model[n] for n in head_names]
    need(tuple(head[0].shape) == (200, 384) and tuple(head[1].shape) == (200,), 'Wrong ViT head architecture')
    return dict(body=digest(body), head=digest(head),
        body_adam=digest([optim['state'].get(i, {}) for i in body_ids]),
        head_adam=digest([optim['state'].get(i, {}) for i in head_ids]))


def audit_vit_source(meta, resolver, anchor, data_root=DATA):
    seed, source_arm, activation = meta['seed'], meta['source_arm'], meta['activation']
    need(seed in range(10252010, 10252015) and source_arm in ('S','SL') and activation in ('R','ELU'), 'Wrong ViT factors')
    need(meta['complete'] is True and meta['study'] == 'B' and meta['model'] == 'vit' and meta['t_end'] == 30,
         'Wrong ViT source identity')
    need(meta['ln'] == ('on' if source_arm == 'SL' else 'off') and meta['source_activation'] == 'R', 'Wrong source LN/activation')
    need(meta['source_id'] == f'vit_{source_arm}_s{seed}_t30_R_trained', 'Spoofed ViT source ID')
    anchor.provenance(meta['provenance'], resolver)
    directory = data_root/'vit_long_1024_2blk_rlfix/runs'/source_arm/f'seed{seed}'/'ckpt'
    mp, ap = directory/'model_t30.pt', directory/'adam_t30.pt'
    sources = meta['provenance']['sources']
    need(set(sources) == {str(mp), str(ap)}, 'ViT source paths do not identify its registered arm/seed/task')
    need(meta['source_sha256'] == digest(sources), 'Combined ViT source identity hash is spoofed')
    model = torch.load(resolver.resolve(mp), map_location='cpu', weights_only=False)
    adam = torch.load(resolver.resolve(ap), map_location='cpu', weights_only=False)
    tensor_finite(model); tensor_finite(adam)
    need({k for k,v in model.items() if not torch.is_tensor(v)} ==
         {f'layers.block_{i}.layers.mlp.layers.act._extra_state' for i in range(2)},
         'Missing/unexpected registered activation extra-state records')
    c = meta['checks']
    need(c['source_state_sha256'] == digest(model), 'Source model tensor digest differs')
    fingerprints = vit_state_fingerprints(model, adam)
    need(c['original_fingerprints'] == fingerprints and c['prepared_fingerprints'] == fingerprints,
         'ViT original/prepared fingerprints differ from checkpoint bytes')
    for key, value in vit_streams(seed).items(): need(c[key] == value, f'Wrong canonical ViT {key}')
    need(c['native_R_rebuild'] == dict(output_exact=True, all_parameter_gradients_exact=True,
        parameters_and_buffers_unchanged=True), 'Native ReLU sham failed')
    need(len(c['activation_modules']) == 2, 'Wrong activation-module count')
    for i, module in enumerate(c['activation_modules']):
        need(module['layer'] == i and module['activation'] == activation and module['added_parameters'] == 0,
             'Wrong genuine activation configuration')
        need(module['module'] == 'BattleActivation' and module['zero_derivative'] == 1,
             'Wrong genuine activation module/boundary derivative')
        need(module['implementation'] == ('torch.clamp(z,min=0)' if activation == 'R' else
             'torch.nn.functional.elu(z,alpha=1,inplace=False)'), 'Wrong current-activation numerical implementation')
        need(module['forward_exact'] is True and module['genuine_autograd_derivative'] is True,
             'Activation forward/derivative check failed')
    need(meta['head_bank']['P_permutation'] == permutation(seed, 'B', 30).tolist(), 'Noncanonical ViT permutation')
    check_geometry(meta['head_bank']['checks'])
    zero_head_hash = digest([torch.zeros_like(model['layers.out.weight']), torch.zeros_like(model['layers.out.bias'])])
    return dict(meta=meta, fingerprints=fingerprints, zero_head_hash=zero_head_hash)


def audit_vit_curve(r, info):
    meta, c, arm = info['meta'], r['checks'], r['arm']
    for k in ('study','model','activation','source_activation','source_arm','ln','seed','t_end','source_id','source_sha256'):
        need(r[k] == meta[k], f'Curve/ViT source metadata differs: {k}')
    need(r['provenance'] == meta['provenance'], 'ViT curve/source provenance mismatch')
    need(c['source'] == meta['checks'] and c['head_bank'] == meta['head_bank']['checks'], 'Curve source/head-bank checks differ')
    fork = c['fork']
    need(r['prepared_fork_hashes'] == fork, 'Fork metadata copies differ')
    need(fork['activation'] == r['activation'] and fork['body_unchanged_by_head_policy'] is True, 'Wrong fork activation/body policy')
    need(fork['head_adam_empty'] is True and fork['all_adam_empty'] is (arm in ('Z_clear','F0')), 'Wrong fork memory policy')
    need(fork['source_or_fresh'] == ('fresh_initial_body' if arm == 'F0' else 'learned_source'), 'Wrong fresh/source policy')
    before, after = fork['before'], fork['after']
    if arm != 'F0': need(before == info['fingerprints'], 'ViT fork is not the original saved model/Adam')
    else:
        need(before['head_adam'] == EMPTY and before['body_adam'] == EMPTY, 'F0 Adam is not fresh')
        need(before['body'] != info['fingerprints']['body'], 'F0 reused the learned body')
    need(after['body'] == before['body'] and after['head_adam'] == EMPTY, 'Wrong body/head-Adam after intervention')
    need(after['body_adam'] == (EMPTY if arm in ('Z_clear','F0') else before['body_adam']), 'Wrong ViT body memory after intervention')
    if arm == 'S1': need(after['head'] == before['head'], 'ViT S1 head is not unchanged')
    zero = arm in ('Z_keep','Z_clear','F0')
    need(fork['inactive_head_rows'] == ('zeroed' if zero else 'unchanged'), 'Wrong inactive-head policy')
    if zero: need(after['head'] == info['zero_head_hash'], 'ViT head-zero fingerprint is not zero')
    warm = c['warmup']
    need(warm['optimizer_steps'] == 0 and warm['state_and_rng_restored'] is True, 'Compile warmup changed state/RNG')
    engine = r['provenance']['args']['engine']
    need(engine == 'eager', 'Final main requires unanimous native-eager ViT execution')
    need(warm['batch_sizes'] == ([128,68] if engine == 'compile' else []), 'Wrong compile shape order')
    need(bool(c['selected_engine']) and 'logits' in c['selected_engine'], 'Missing native/selected-engine check')
    for q in c['selected_engine'].values():
        need(q['passed'] is True, 'Native/selected-engine check failed')
        need(q['maxabs'] == 0 and q['relative_l2'] in (0, None), 'Native-eager repeated computation is not exact')


def check_measurement_contract(r):
    A.validate_record(r)
    need(isinstance(r['tracks'], list) and [q['step'] for q in r['tracks']] == [1,25,200,r['horizon']], 'Missing real-update tracking points')
    for q in r['tracks']:
        groups = q['groups']
        need({'body','trunk','head'} <= groups.keys(), 'Missing body/trunk/head update groups')
        for group in groups.values():
            for metric in ('gradient_norm','update_norm','parameter_norm'):
                need(isinstance(group[metric], (int,float)) and group[metric] >= 0, 'Invalid gradient/update norm')
    for at in (0,25,200,r['horizon']):
        q = next(q for q in r['curve'] if q['step'] == at)
        need(isinstance(q.get('diagnostics'), dict) and bool(q['diagnostics']), 'Missing registered state diagnostics')


def same_group(store, key, value, message):
    previous = store.setdefault(key, value)
    need(previous == value, message)


def run_audit(raw, out, execution_commit, archive_maps=(), root=ROOT):
    started = time.process_time()
    torch.set_num_threads(1)
    resolver = Resolver(archive_maps)
    resolver.audit_manifest()
    anchor = CommitAnchor(root, execution_commit)
    records = A.load_records(raw)
    validation = A.validate_design(records)
    source_groups = defaultdict(list)
    for r in records:
        check_measurement_contract(r)
        source_groups[(r['study'],r['activation'],r['ln'],r['seed'])].append(r)
    common_a, common_b, current_b, fresh_b = {}, {}, {}, {}
    total_legacy = 0
    for (study, act, ln, seed), curves in sorted(source_groups.items()):
        if study == 'A':
            references = {tuple(r['provenance']['sources'].items()) for r in curves}
            need(len(references) == 1 and len(next(iter(references))) == 1, 'A curves do not reference one source PT')
            original, expected = next(iter(references))[0]
            source_path = resolver.verify(original, expected)
            source = torch.load(source_path, map_location='cpu', weights_only=False)
            meta_path = resolver.resolve(Path(original).with_suffix('.json'))
            info = audit_mlp_source(source, load_json(meta_path), original, expected, resolver, anchor)
            common = {k: source[k] for k in ('initial_common_sha256','inputs_sha256','subset_sha256','labels_sha256','batch_sha256')}
            same_group(common_a, seed, common, 'A activation/LN conditions use different init/input/label/batch streams')
            for r in curves:
                anchor.provenance(r['provenance'], resolver)
                audit_mlp_curve(r, info, resolver)
                total_legacy += len(r['checks']['legacy'])
        else:
            sample = curves[0]
            name = f"source_B_vit_{sample['source_arm']}_{act}_s{seed}_t30.json"
            meta_path = Path(sample['_path']).with_name(name)
            need(meta_path.is_file(), 'Missing ViT source metadata beside curves')
            info = audit_vit_source(load_json(meta_path), resolver, anchor)
            c = info['meta']['checks']
            shared = {k:c[k] for k in ('data_paths_sha256','input_tensor_sha256','labels_old_sha256','labels_new_sha256','batch_order_sha256','data_plan')}
            same_group(common_b, seed, shared, 'B source LN/current-activation conditions use different data/task streams')
            immutable = {k:c[k] for k in ('initial_state_sha256','source_state_sha256','original_fingerprints','prepared_fingerprints')}
            same_group(current_b, (ln,seed), immutable, 'B current activation comparison changes saved state/initialization')
            for r in curves:
                audit_vit_curve(r, info)
                if r['arm'] == 'F0':
                    same_group(fresh_b, (ln,seed), r['checks']['fork']['before'], 'B F0 initial state differs across current activations')
    need(len(common_a) == 10 and len(common_b) == 5 and len(current_b) == 10 and len(fresh_b) == 10,
         'Missing paired source regimes')
    # 40 readout-separation comparisons + 180 p-residual comparisons.
    need(total_legacy == 220, 'Missing registered legacy continuation controls')
    result = dict(complete=True, curves=validation['actual'], counts=validation['counts'],
        execution_git_hash=anchor.commit, spec_sha256=anchor.spec, code_sha256=anchor.code_digest,
        A_sources_verified=80, B_source_metadata_verified=20, B_original_source_pairs=10,
        legacy_comparisons_verified=total_legacy, unique_file_hashes_verified=len(resolver.verified),
        archive_entries_verified=len(resolver.mapping),
        A_common_init_input_label_batch_verified=True, A_saved_noLN_task1_task10_weights_verified=True,
        B_original_model_Adam_bytes_and_fingerprints_verified=True,
        B_current_activation_common_source_verified=True, all_curve_checks_verified=True,
        limitations=[
            'No model forward or training is replayed. Intermediate no-LN task2..9 exactness is checked from runner records; only saved task1/task10 tensors are independently compared.',
            'B image tensor/path hashes are checked across all paired conditions; original JPEG-to-tensor bytes are not reconstructed.',
            'B canonical fresh-model initialization is not rebuilt; initial/F0 fingerprints are checked across R/ELU and fresh-Adam/zero-head semantics against saved evidence.',
            'Native/compiled numerical-check success flags are audited, but forward/backward numerical comparisons are not rerun.'],
        source_files=dict(sorted(resolver.verified.items())), audit_sha256=A.sha256(__file__),
        audit_cpu_seconds=time.process_time()-started)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.atomic_text(out, json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--execution-commit', required=True)
    p.add_argument('--archive-map', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, default=ROOT/'results/elu_specificity_1112/integrity_validation.json')
    args = p.parse_args()
    report = run_audit(args.raw, args.out, args.execution_commit, args.archive_map)
    print(json.dumps({k:v for k,v in report.items() if k != 'source_files'}, indent=2))


if __name__ == '__main__': main()
