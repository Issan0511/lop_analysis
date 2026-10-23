#!/usr/bin/env python3
"""Delayed natural B recovery from the complete degraded CHf task-30 state."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import cnn_center_feat_1011 as CF
from src import cnn_center_relu_1011 as CR
from src import rlcifar_cnn_0908 as CN
from src import sna_cnn_cause_1009 as E
from src import pmnist_rlcifar_0907 as RC
from src import pmnist_0905 as H
from src.cnn_response_center_1021 import (cpu, sha_file, tree_hash, tensor_equal,
    save_json, write_csv, locked, sign_test, effect_direction)

from src import cnn_response_chb_1021 as S
from src import cnn_chprime_bias_1018 as PARENT

EXPERIMENT = 'cnn_chb_delayed_1021'
SPEC_PATH = 'specs/spec_cnn_chb_delayed_1021.md'
SEEDS = list(range(10, 20))
MAIN_ARMS = ('CH_continue', 'CH_addB')
DIAG_STEPS = (0, 1, 100, 750, 3000, 15000, 30000)
KAPPA = .2
DEFAULT_SOURCE = S.DEFAULT_SOURCE
DEFAULT_MANIFEST = S.DEFAULT_MANIFEST
forward = S.forward  # field=None dispatches directly to CF.forward; never constructs a field.
configure_runtime = S.configure_runtime
atomic_npz = S.atomic_npz

class Data:
    def __init__(self, seeds=SEEDS, device=torch.device('cpu'), source_root=DEFAULT_SOURCE,
                 manifest_path=DEFAULT_MANIFEST, data_dir=None):
        from src.cnn_chb_delayed_sources_1021 import audit_sources, json_sha
        states, metadata = audit_sources(source_root=source_root, manifest_path=manifest_path)
        self.seeds, self.device = list(seeds), torch.device(device)
        if self.seeds != SEEDS:
            raise ValueError('The registered real-data bundle is seeds 10–19 in that order')
        self.sources = {a: s.get('state', s) for a, s in states.items()}
        self.source_meta = metadata
        if data_dir is not None:
            RC.DATA_DIR = Path(data_dir)
        self.cifar = RC.Cifar10()
        self.center = CR.train_channel_mean(self.cifar).to(self.device)
        self.ids = torch.stack([CN.subset_idx(s) for s in self.seeds])
        subset_digest = json_sha({str(s): ids.tolist() for s, ids in zip(self.seeds, self.ids)})
        for arm in ('CHf',):
            if subset_digest != metadata['arms'][arm]['subsets_sha256']:
                raise RuntimeError(f'Runtime global image IDs differ from archived parent: {arm}')
        self.X = torch.stack([CN.images(self.cifar, ids, self.device) for ids in self.ids])
        self.X.sub_(self.center.reshape(1, 1, 3, 1, 1))
        source_data = metadata.get('data_sha256')
        if source_data is not None and source_data != self.cifar.sha256:
            raise RuntimeError('CIFAR SHA differs from source run')
        self.identity = {'seeds': self.seeds, 'sources': metadata, 'data_sha256': self.cifar.sha256,
                         'global_image_ids_sha256': tree_hash(self.ids), 'center': cpu(self.center).tolist()}

    def inputs(self, arm):
        return self.X


class Engine(S.Engine):
    def __init__(self, data, name, graph=True):
        if name not in MAIN_ARMS:
            raise ValueError(name)
        super().__init__(data, 'CH_nat', graph=graph)
        self.name = name
        self.enabled = name == 'CH_addB'
        # Changing this policy flag does not project or change any initial tensor.
        self.update_trace = torch.zeros(CN.STEPS_PER_EPOCH, self.R, 5, 7, device=self.device)

    def step(self, j):
        idx = self.order[:, j*CN.BATCH:(j+1)*CN.BATCH]
        xb, yb = self.X[self.ar, idx], self.Y[self.ar, idx]
        refs = self.refs_ids(idx)
        out, _, _ = forward(self.P, xb, self.act, self.field, refs)
        lossv = F.cross_entropy(out[8].reshape(-1, CN.N_CLASSES), yb.reshape(-1), reduction='none').view(self.R, -1).mean(1)
        hit = (out[8].detach().argmax(-1) == yb).float().mean(1)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.bad.logical_or_(~torch.isfinite(lossv))
            for gr in grads:
                self.bad.logical_or_(~torch.isfinite(gr).flatten(1).all(1))
            before = [p.detach().clone() for p in self.P]
            self.tc.add_(1)
            c1 = (1-torch.pow(self.b1, self.tc)).float()
            c2 = (1-torch.pow(self.b2, self.tc)).float()
            for p, gr, mi, vi, last in zip(self.P, grads, self.m, self.v, self.last_grads):
                last.copy_(gr)
                mi.mul_(.9).add_(gr, alpha=1-.9)
                vi.mul_(.999).addcmul_(gr, gr, value=1-.999)
                p.sub_(self.lr*(mi/c1)/((vi/c2).sqrt()+1e-8))
            self.act.update_actual([out[2*l].detach() for l in range(4)], [out[2*l+1].detach() for l in range(4)])
            adam_bias = [self.P[2*l+1].detach().clone() for l in range(5)]
            if self.enabled:
                rows = PARENT.project_bias(self.P, xb, self.act)
                for l, row in enumerate(rows):
                    for key in self.trace[l]:
                        self.trace[l][key][j].copy_(row[key])
                    self.bad.logical_or_(~row['finite'])
            for l in range(5):
                wp, bp = self.P[2*l:2*l+2]
                w0, b0 = before[2*l:2*l+2]
                denominator = (self.v[2*l]/c2).sqrt()+1e-8
                self.update_trace[j, :, l, 0].copy_((wp-w0).flatten(1).norm(dim=1)/w0.flatten(1).norm(dim=1).clamp_min(1e-30))
                self.update_trace[j, :, l, 1].copy_((bp-b0).square().mean(1).sqrt())
                self.update_trace[j, :, l, 2].copy_(denominator.flatten(1).mean(1))
                self.update_trace[j, :, l, 3].copy_((grads[2*l]/denominator).flatten(1).square().mean(1).sqrt())
                self.update_trace[j, :, l, 4].copy_(grads[2*l].flatten(1).square().mean(1).sqrt())
                self.update_trace[j, :, l, 5].copy_((adam_bias[l]-b0).square().mean(1).sqrt())
                self.update_trace[j, :, l, 6].copy_((adam_bias[l]-b0).norm(dim=1)/b0.norm(dim=1).clamp_min(1e-30))
            for value in (*self.P, *self.m, *self.v, *self.act.m, *self.act.V):
                self.bad.logical_or_(~torch.isfinite(value).flatten(1).all(1))
            self.acc_steps[:, j].copy_(torch.where(self.bad, torch.zeros_like(hit), hit))
            self.ce_steps[:, j].copy_(torch.where(self.bad, torch.full_like(lossv, float('inf')), lossv))
            self.acc_ep.add_(self.acc_steps[:, j])


@torch.no_grad()
def diagnose(B, update, chunk=CN.BATCH):
    rows, units, arrays = S.diagnose(B, update, chunk)
    # Keep natural quantities only; no transplanted/effective fields exist here.
    irrelevant = ('gate_agreement', 'delta_', 'activity_offset', 'fixed_anchor_offset')
    rows = [{k.replace('_raw_', '_'): v for k,v in row.items()
             if not any(k.startswith(tag+'_effective_') for tag in CN.SITES) and not any(t in k for t in irrelevant)} for row in rows]
    units = [{k:v for k,v in u.items() if not any(t in k for t in irrelevant)}
             for u in units if u['kind'] == 'raw']
    arrays = {k.replace('_raw_', '_'):v for k,v in arrays.items()
              if not any(k.startswith(tag+'_effective_') for tag in CN.SITES) and not any(t in k for t in irrelevant)}
    for l,tag in enumerate(CN.SITES):
        sd = arrays[f'{tag}_sd']
        width = np.sqrt((sd**2).mean(axis=-1)) if l < 2 else sd
        bias = cpu(B.P[2*l+1]).numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = bias/width
        arrays[f'{tag}_image_width'] = width
        arrays[f'{tag}_bias_over_width'] = ratio
        for r in range(B.R):
            rows[r][f'{tag}_image_width_mean'] = float(width[r].mean())
            rows[r][f'{tag}_bias_over_width_mean'] = float(ratio[r].mean())
            rows[r][f'{tag}_bias_over_width_abs_mean'] = float(np.abs(ratio[r]).mean())
    return rows, units, arrays

def source_hashes():
    """Every loaded repository Python dependency plus verification sources/spec."""
    files = {Path(__file__), ROOT/SPEC_PATH}
    for mod in tuple(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if path:
            p = Path(path).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT):
                files.add(p)
    for pattern in ('tests/test_cnn_chb_delayed_1021.py', 'analysis/cnn_chb_delayed_1021/*.py',
                    'src/cnn_chb_delayed_sources_1021.py', 'src/cnn_chprime_bias_1018.py'):
        files.update(ROOT.glob(pattern))
    return {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(files) if p.exists()}


def numerical_context():
    result = S.numerical_context()
    result.pop('anchor_arithmetic')
    result['forward'] = 'native_CF.forward_float32_no_artificial_field'
    return result


def guard_release(release):
    """Fail closed on drafts, unpushed release, dirty code, or source drift."""
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], text=True, capture_output=True, check=True).stdout.strip()
    commit = git('rev-parse', '--verify', release+'^{commit}')
    git('merge-base', '--is-ancestor', commit, 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=all', '--', 'src', 'specs',
                'tests/test_cnn_chb_delayed_1021.py', 'analysis/cnn_chb_delayed_1021')
    if dirty:
        raise RuntimeError('Uncommitted implementation/spec invalidates preregistration:\n'+dirty)
    if not git('for-each-ref', '--format=%(refname)', '--contains='+commit, 'refs/remotes/origin'):
        raise RuntimeError('Preregistration commit has no pushed origin tracking reference')
    current = (ROOT/SPEC_PATH).read_text()
    registered = git('show', commit+':'+SPEC_PATH)
    if '登録状態: REGISTERED' not in current.splitlines() or current.strip() != registered.strip():
        raise RuntimeError('A matching REGISTERED spec is required before any real-data forward')
    hashes = source_hashes()
    for path, digest in hashes.items():
        result = subprocess.run(['git', '-C', str(ROOT), 'show', commit+':'+path], capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != digest:
            raise RuntimeError(f'Implementation differs from preregistered commit: {path}')
    return {'registered_spec_commit': commit, 'implementation_commit': commit,
            'current_head': git('rev-parse', 'HEAD'), 'source_sha256': hashes}


def identity(data, steps=30000, graph=True):
    return {'experiment': EXPERIMENT, 'data': data.identity, 'steps': steps, 'graph': graph,
            'code_sha256': source_hashes(), 'numerical_context': numerical_context(),
            'batch': CN.BATCH, 'lr': .001, 'kappa': KAPPA, 'activity_dtype': 'float32', 'B_width_dtype': 'float64',
            'first_B_application': 'after_first_Adam_and_EMA',
            'primary_window': [1, 3000], 'secondary_windows': {'post_transient': [1001, 3000], 'all': [1, 30000], 'last': [27001, 30000]}}


def preflight_contract(data, graph, release_info):
    return {'contract_version': 1, 'registered_spec_commit': release_info['registered_spec_commit'],
            'implementation_commit': release_info['implementation_commit'],
            'identity': identity(data, 30000, graph)}


@torch.no_grad()
def preflight_anchor(B, chunk=CN.BATCH):
    same = True
    for lo in range(0, B.X.shape[1], chunk):
        ids = torch.arange(lo, min(lo+chunk, B.X.shape[1]), device=B.device)[None].expand(B.R, -1)
        actual = B.forward_ids(ids)[0]
        natural = CF.forward(B.P, B.X[B.ar, ids], B.act)
        same &= tree_hash(actual) == tree_hash(natural)
    return {'pass': same, 'field_is_none': B.field is None, 'initial_projection': False}


def run_arm(data, arm, out, steps=30000, graph=True, stop_file=None):
    if steps % CN.STEPS_PER_EPOCH:
        raise ValueError('Complete epoch boundaries are required for resume')
    armout = Path(out)/arm
    armout.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoints', 'diagnostics', 'projection'):
        (armout/name).mkdir(exist_ok=True)
    ident = identity(data, steps, graph) | {'arm': arm}
    B = Engine(data, arm, graph=graph)
    statepath = armout/'state.pt'
    acc, ces = [np.zeros((B.R, steps), dtype=np.float32) for _ in range(2)]
    updates = np.zeros((steps, B.R, 5, 7), dtype=np.float32)
    fires = np.zeros((steps, B.R, 4), dtype=np.float32)
    epochrows, diagrows, unitrows, streams = [], [], [], []
    start_epoch = 0

    def diagnostic(k):
        dr, ur, arrays = diagnose(B, k)
        diagrows.extend(dr); unitrows.extend(ur)
        atomic_npz(armout/'diagnostics'/f'update{k:05d}.npz', **arrays)
        E.save_atomic(B.snapshot(), armout/'checkpoints'/f'update{k:05d}.pt')
        write_csv(armout/'diagnostics.csv', diagrows)
        write_csv(armout/'unit_diagnostics.csv', unitrows)

    def save_state(epoch):
        E.save_atomic({'identity': ident, 'engine': B.snapshot(), 'epoch': epoch,
                       'acc': acc, 'ce': ces, 'updates': updates, 'fires': fires,
                       'epochrows': epochrows, 'diagrows': diagrows, 'unitrows': unitrows,
                       'streams': streams}, statepath)

    if statepath.exists():
        saved = torch.load(statepath, map_location='cpu', weights_only=False)
        if saved['identity'] != ident:
            raise RuntimeError(f'Resume identity differs: {statepath}')
        B.restore(saved['engine']); start_epoch = saved['epoch']
        if not 0 <= start_epoch <= steps//CN.STEPS_PER_EPOCH:
            raise RuntimeError('Invalid saved epoch')
        if float(B.tc) != 900000+start_epoch*CN.STEPS_PER_EPOCH:
            raise RuntimeError('Saved optimizer step and epoch disagree')
        acc, ces, updates, fires = [saved[k] for k in ('acc', 'ce', 'updates', 'fires')]
        epochrows, diagrows, unitrows, streams = [saved[k] for k in ('epochrows', 'diagrows', 'unitrows', 'streams')]
    else:
        B.new_labels()
        # Parent counters count lifetime EMA moves; at task switch the parent resets
        # only this observational counter, preserving EMA/V and optimizer state.
        B.act.hupd.zero_()
        anchor = preflight_anchor(B)
        save_json(armout/'anchor.json', anchor)
        if not anchor['pass']:
            raise RuntimeError(f'Initial all-layer anchor fails: {arm}')
        diagnostic(0)
        save_state(0)
    provenance_path = armout/'provenance.json'
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    git_hash = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    provenance = {'identity': ident, 'git_hash': git_hash, 'command': sys.argv,
                  'started_utc': previous.get('started_utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
                  'resume_epochs': previous.get('resume_epochs', [])+[start_epoch],
                  'anchor_sha256': B.field.cache_hash if B.field is not None else None,
                  'label_sha256': tree_hash(B.Y), 'status': 'RUNNING',
                  'update_trace_fields': ['delta_w_relative_norm', 'delta_b_rms', 'adam_denom_weight_mean',
                                          'grad_over_adam_denom_weight_rms', 'grad_weight_rms',
                                          'adam_delta_bias_rms', 'adam_delta_bias_relative_norm']}
    save_json(provenance_path, provenance)
    started = time.monotonic()
    for epoch in range(start_epoch, steps//CN.STEPS_PER_EPOCH):
        if stop_file is not None and Path(stop_file).exists():
            print(f'{arm}: stop file before epoch {epoch+1}; complete epoch checkpoint retained', flush=True)
            break
        lo, hi = epoch*CN.STEPS_PER_EPOCH, (epoch+1)*CN.STEPS_PER_EPOCH
        interior = [k for k in DIAG_STEPS if lo < k < hi]
        wall = time.monotonic()
        if interior:
            B.new_order(); B.acc_ep.zero_()
            for j in range(CN.STEPS_PER_EPOCH):
                B.step(j)
                if lo+j+1 in interior:
                    diagnostic(lo+j+1)
        else:
            B.run_epoch()
        acc[:, lo:hi] = cpu(B.acc_steps).numpy()
        ces[:, lo:hi] = cpu(B.ce_steps).numpy()
        updates[lo:hi] = cpu(B.update_trace).numpy()
        for l in range(4):
            fires[lo:hi, :, l] = cpu(B.trace[l]['hit'].float().mean(-1)).numpy()
        if B.enabled:
            atomic_npz(armout/'projection'/f'epoch{epoch+1:03d}.npz', order=cpu(B.order).numpy(),
                       **{f'{CN.SITES[l]}_{key}': cpu(value).numpy()
                          for l, row in enumerate(B.trace) for key, value in row.items()})
        streams.append({'epoch': epoch+1, 'order_sha256': tree_hash(B.order),
                        'batch_generator_sha256': tree_hash({s:g.get_state() for s,g in B.g_batch.items()})})
        for r, seed in enumerate(data.seeds):
            epochrows.append({'arm': arm, 'seed': seed, 'epoch': epoch+1, 'update': hi,
                              'online_acc': float(acc[r, lo:hi].mean(dtype=np.float64)),
                              'online_ce': float(ces[r, lo:hi].mean(dtype=np.float64)), 'failed': bool(B.bad[r])})
        if hi in DIAG_STEPS or hi == steps:
            diagnostic(hi)
        save_state(epoch+1)
        write_csv(armout/'epochs.csv', epochrows)
        save_json(armout/'streams.json', {'labels_sha256': tree_hash(B.Y), 'epochs': streams})
        print(f'{arm} epoch {epoch+1:03d}/{steps//75} update {hi:05d} online={acc[:,lo:hi].mean():.6f} wall={time.monotonic()-wall:.2f}s', flush=True)
    completed = len(epochrows) == B.R*(steps//CN.STEPS_PER_EPOCH)
    provenance.update(status='COMPLETE' if completed else 'STOPPED', wall_this_process_s=time.monotonic()-started)
    if completed:
        atomic_npz(armout/'batch_curves.npz', seeds=np.array(data.seeds), accuracy=acc, ce=ces,
                   updates=updates, B_fire_fraction=fires)
        summaries = []
        windows = {'first3000': slice(0, min(3000, steps)), 'steps1001_3000': slice(min(1000,steps-1), min(3000,steps)),
                   'all': slice(None), 'last3000': slice(max(0, steps-3000), steps)}
        for r, seed in enumerate(data.seeds):
            row = {'arm': arm, 'seed': seed, 'failed': bool(B.bad[r])}
            for name, window in windows.items():
                row[name] = float(acc[r, window].mean(dtype=np.float64))
                row['ce_'+name] = float(ces[r, window].mean(dtype=np.float64))
            summaries.append(row)
        write_csv(armout/'summary.csv', summaries)
        save_json(armout/'done.json', {'identity': ident, 'summary_sha256': sha_file(armout/'summary.csv'),
                                      'curves_sha256': sha_file(armout/'batch_curves.npz'),
                                      'final_state_sha256': sha_file(statepath),
                                      'labels_sha256': tree_hash(B.Y), 'streams_sha256': tree_hash(streams),
                                      'failed_seeds': [s for r,s in enumerate(data.seeds) if bool(B.bad[r])]})
    save_json(provenance_path, provenance)
    del B
    if data.device.type == 'cuda':
        torch.cuda.empty_cache()
    return completed


def evaluate_primary(levels):
    """First3000 accuracy and CE: two-sided signs, Holm family of two."""
    bootstrap=np.random.default_rng(1021).integers(0,10,size=(20000,10))
    tests={}
    for metric in ('accuracy','ce'):
        base=np.asarray(levels['CH_continue'][metric],dtype=np.float64)
        treated=np.asarray(levels['CH_addB'][metric],dtype=np.float64)
        finite=bool(np.isfinite(base).all() and np.isfinite(treated).all())
        if metric=='accuracy': benefit=treated-base
        else:
            # Failed CE is +inf. If both fail, the pair is a sign-test tie.
            benefit=np.zeros(10)
            mask=np.isfinite(base)&np.isfinite(treated)
            benefit[mask]=base[mask]-treated[mask]
            benefit[np.isposinf(base)&np.isfinite(treated)]=np.inf
            benefit[np.isfinite(base)&np.isposinf(treated)]=-np.inf
        tests[metric]=sign_test(benefit)|{
            'mean_benefit':float(benefit.mean()) if finite else None,
            'ci95_seed_bootstrap':np.percentile(benefit[bootstrap].mean(1),[2.5,97.5]).tolist() if finite else None,
            'finite':finite, 'benefit_signs':np.sign(benefit).astype(int).tolist()}
    previous=0.
    for rank,key in enumerate(sorted(tests,key=lambda k:tests[k]['p_two_sided'])):
        t=tests[key];p=min(1.,max(previous,(2-rank)*t['p_two_sided']));previous=p
        t['p_holm']=p
        if not t['finite']: verdict='INCONCLUSIVE_NONFINITE'
        elif p>.05: verdict='INCONCLUSIVE'
        elif t['mean_benefit']>0 and t['n_positive']>t['n_negative']: verdict='IMPROVES'
        elif t['mean_benefit']<0 and t['n_negative']>t['n_positive']: verdict='WORSENS'
        else: verdict='HETEROGENEOUS'
        t['verdict']=verdict
    return tests


def summarize(out):
    out=Path(out)
    levels={}
    rows=[]
    for arm in MAIN_ARMS:
        with (out/arm/'summary.csv').open() as f: table=list(csv.DictReader(f))
        if [int(r['seed']) for r in table]!=SEEDS: raise RuntimeError('Incomplete seeds')
        done=json.loads((out/arm/'done.json').read_text())
        if sha_file(out/arm/'summary.csv')!=done['summary_sha256']: raise RuntimeError('Summary hash mismatch')
        levels[arm]={'accuracy':[float(r['first3000']) for r in table], 'ce':[float(r['ce_first3000']) for r in table]}
        rows.extend(table)
    result={'experiment':EXPERIMENT,'tests':evaluate_primary(levels), 'seeds':SEEDS,
            'family':'first3000 accuracy and CE; two-sided exact sign tests; Holm 2',
            'scope':'late natural B addition; improvement over degraded CH, not full restoration or mediation'}
    save_json(out/'verdict.json',result)
    write_csv(out/'all_seed_summary.csv',rows)
    print(json.dumps(result,indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run','summarize'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--manifest-path', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--data-dir')
    p.add_argument('--seeds', default='10-19')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--arms', default=','.join(MAIN_ARMS))
    p.add_argument('--device', default='cuda')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--checks', type=Path)
    p.add_argument('--release')
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--gpu-lock', type=Path, default=Path('/tmp/lop_analysis_gpu.lock'))
    return p


def main():
    args = parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summarize':
        summarize(args.out)
        return
    if not args.release:
        raise SystemExit('Real data require --release <committed and pushed preregistration>')
    release = guard_release(args.release)
    if args.steps != 30000 or E.parse_seeds(args.seeds) != SEEDS or args.device != 'cuda':
        raise SystemExit('Registered run fixes CUDA, all seeds10–19, and 30,000 updates')
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or not set(arms) <= set(MAIN_ARMS):
        raise SystemExit('Only unique preregistered arms may run')
    with locked(args.out/'.run.lock'), locked(args.gpu_lock):
        device = configure_runtime(args.device)
        data = Data(SEEDS, device, args.source_root, args.manifest_path, args.data_dir)
        checkpath = args.checks or args.out/'checks'/'preflight.json'
        checked = json.loads(checkpath.read_text())
        if checked.get('all_pass') is not True:
            raise RuntimeError('A successful matching real GPU preflight is required')
        contract = preflight_contract(data, not args.eager, release)
        if checked.get('contract') != contract:
            raise RuntimeError('Preflight source/spec/runtime/stream identity differs; repeat checks')
        for arm in arms:
            current = guard_release(args.release)
            if preflight_contract(data, not args.eager, current) != contract:
                raise RuntimeError('Implementation changed between arms')
            if not run_arm(data, arm, args.out, args.steps, not args.eager, args.stop_file):
                break


if __name__ == '__main__':
    main()
