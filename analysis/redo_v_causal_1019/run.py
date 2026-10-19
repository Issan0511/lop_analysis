#!/usr/bin/env python3
"""Paired, first-reset L2 readout intervention on the unchanged 5+1 ReDo engine.

Training arithmetic and streams come from src/baselines_cifar5p1_1008.py.
Readout and centering interventions occur after its first L2 outgoing reset
(tc=1560); gauge controls transform the same cohort at task3 step50. The
original ReDo moment-reset policy is retained. Centering uses a fixed input
origin in a forward proxy; gauge controls also transform Adam moments.
See specs/spec_redo_v_causal_1019.md for the experiment and interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import baselines_cifar5p1_1008 as S
from src import cifar5p1_mlp_0920 as C
from src import rlcifar_mlp_battle_0918 as B
from src import pmnist_0905 as H

RUN = 'redo_v_causal_1019'
OUT = ROOT / 'results' / RUN
DATA = Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz')
REFERENCE = Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019/raw')
EXPECTED_SHA = 'c66760ba7c55af952adc76a5432edf3a6b765b86887e0fea3732fab8ae693049'
POINTS = {1, 2, 5, 10, 25, 50, 100, 200, 400, 780}
ARMS = {'zero': 0.0, 'keep': None, 'rand01': 0.1, 'rand03': 0.3, 'rand10': 1.0,
        'center': 0.0, 'center_random': 0.0, 'gauge4': 0.0, 'gauge025': 0.0}
ZERO_READOUT_ARMS = {'zero', 'center', 'center_random', 'gauge4', 'gauge025'}


def cpu(x):
    return x.detach().cpu().numpy().copy()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def independent_directions(seed, classes, units):
    """Class-centered unit-norm columns; no Torch/global/training RNG use."""
    digest = hashlib.sha256(f'{RUN}|outgoing_direction|{seed}'.encode()).digest()
    rng_seed = int.from_bytes(digest[:16], 'little')
    rng = np.random.default_rng(rng_seed)
    z = rng.standard_normal((classes, units))
    z -= z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=0, keepdims=True)
    return z.astype(np.float32)


def replacement_columns(old, seeds, arm):
    """Return all columns; the caller writes only the preregistered mask."""
    if arm in ZERO_READOUT_ARMS:
        return torch.zeros_like(old)
    if arm == 'keep':
        return old.clone()
    directions = torch.as_tensor(np.stack([
        independent_directions(s, old.shape[1], old.shape[2]) for s in seeds
    ]), dtype=old.dtype, device=old.device)
    centered = old - old.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    return directions * norm * ARMS[arm]


class CenteredForward:
    """A fixed per-unit input origin, represented by an effective bias.

    Offsets are allocated before graph capture and updated in place exactly
    once. The original B module is never modified. For non-centering arms,
    original training and evaluation functions are called without wrappers.
    """
    def __init__(self, slots, units, device):
        self.offsets = torch.zeros((slots, units, units), device=device)

    def effective(self, P):
        return [P[0], P[1], P[2], P[3] - (P[2] * self.offsets).sum(-1), P[4], P[5]]

    def __call__(self, P, X, act, train=False):
        return B.forward(self.effective(P), X, act, train=train)


def instrumented(callback, centered=None):
    original = Path(S.__file__).read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA:
        raise RuntimeError('Original engine hash changed; review source injection before running')
    src = inspect.getsource(S.run)

    def insert(old, new):
        nonlocal src
        if src.count(old) != 1:
            raise RuntimeError(f'instrumentation anchor count {src.count(old)}: {old[:100]}')
        src = src.replace(old, new)

    insert('        grads = torch.autograd.grad(lossv.sum(), P)\n',
           '        grads = torch.autograd.grad(lossv.sum(), P)\n        _causal_grads[:] = grads\n')
    insert('        units = [[torch.nonzero(masks[l][r]).flatten() for r in range(R)] for l in range(2)]\n',
           '        units = [[torch.nonzero(masks[l][r]).flatten() for r in range(R)] for l in range(2)]\n'
           '        diagnostic("reset_pre", P=P, m=adam_m, v=adam_v, masks=masks, tc=n, t=t, act=act)\n')
    insert('                    adam_v[2 * l + 2][r, :, idx] = 0.0\n        for r in range(R):',
           '                    adam_v[2 * l + 2][r, :, idx] = 0.0\n'
           '            if l == 1:\n'
           '                diagnostic("reset_outgoing_2", P=P, m=adam_m, v=adam_v, masks=masks, tc=n, t=t, act=act)\n'
           '        for r in range(R):')
    insert('        steps = batches.shape[1]\n        pending = False\n',
           '        steps = batches.shape[1]\n'
           '        if phase == "train":\n'
           '            diagnostic("task_start", P=P, m=adam_m, v=adam_v, batches=batches, tc=tc0, t=t, act=act)\n'
           '        pending = False\n')
    insert('            act.begin_step(R, C.BATCH, device)\n            if cg is not None:',
           '            if phase == "train" and t == 3 and j + 1 in _causal_points:\n'
           '                diagnostic("before_step", P=P, m=adam_m, v=adam_v, tc=tc, t=t, j=j, idx=batches[:, j], act=act)\n'
           '            act.begin_step(R, C.BATCH, device)\n            if cg is not None:')
    insert('            if hook is not None:\n                hook("post_step",',
           '            if phase == "train" and t == 3 and j + 1 in _causal_points:\n'
           '                diagnostic("after_step", P=P, m=adam_m, v=adam_v, tc=tc, t=t, j=j, grads=_causal_grads, inv_c1=inv_c1, inv_c2=inv_c2, acc_sum=acc_sum, act=act)\n'
           '            if hook is not None:\n                hook("post_step",')
    insert('    for t in range(1, n_tasks + 1):\n',
           '    diagnostic("init", P=P, m=adam_m, v=adam_v, tc=0, t=1, act=act)\n'
           '    for t in range(1, n_tasks + 1):\n')
    insert('            if observer is not None:\n',
           '            diagnostic("task_end", P=P, m=adam_m, v=adam_v, tc=tc, t=t, Xt=Xt, Yt=Yt, acc_sum=acc_sum, act=act)\n'
           '            if observer is not None:\n')
    env = dict(S.__dict__, diagnostic=callback, _causal_grads=[], _causal_points=POINTS)
    if centered is not None:
        bproxy = types.SimpleNamespace(**vars(B))
        bproxy.forward = centered
        cproxy = types.SimpleNamespace(**vars(C))
        # Wrapping the original metric functions with effective parameters
        # preserves their arithmetic and avoids changing their module globals.
        cproxy.evaluate = lambda P, *a, **kw: C.evaluate(centered.effective(P), *a, **kw)
        cproxy.accuracy = lambda P, *a, **kw: C.accuracy(centered.effective(P), *a, **kw)
        env.update(B=bproxy, C=cproxy,
                   extra_stats=lambda P, *a, **kw: S.extra_stats(centered.effective(P), *a, **kw),
                   dead_frac_l2=lambda P, *a, **kw: S.dead_frac_l2(centered.effective(P), *a, **kw))
    exec(compile(src, '<paired-first-readout-reset-instrumentation>', 'exec'), env)
    return env['run'], src


def unit_summary(z):
    p = (z > 0).float().mean(dim=1)
    a = z.clamp_min(0)
    return {k: cpu(v) for k, v in dict(
        p=p, mean=z.mean(dim=1), sd=z.std(dim=1, correction=0),
        activation_rms=a.square().mean(dim=1).sqrt(),
        activation_sd=a.std(dim=1, correction=0), activation_mean=a.mean(dim=1),
        gate_diversity=p * (1 - p), near99=(p >= .99).float(),
        fullopen=(p == 1).float(), alloff=(p == 0).float()).items()}


def ce_decomposition(z2, logits, W3, labels):
    """Per-input CE upward drive on a fixed probe, in float64.

    demand_i = g_i v_i^T(y-p_minus_i), brake_i = g_i v_i^T(p-p_minus_i).
    Their difference equals -dL_example/dz_i, with no batch-size scaling.
    Brakes are nonnegative for ReLU, up to floating-point roundoff.
    Chunking over units avoids an R*N*U*C dense permanent array.
    """
    z2, logits, W3 = z2.double(), logits.double(), W3.double()
    p = logits.softmax(dim=-1)
    y = F.one_hot(labels, num_classes=logits.shape[-1]).double()
    demand = torch.empty_like(z2)
    brake = torch.empty_like(z2)
    direct = torch.einsum('rnc,rcu->rnu', y - p, W3) * (z2 > 0)
    for lo in range(0, z2.shape[-1], 10):
        hi = min(lo + 10, z2.shape[-1])
        vv = W3[:, :, lo:hi].transpose(1, 2)
        aa = z2[:, :, lo:hi].clamp_min(0)
        logits_minus = logits[:, :, None, :] - aa[:, :, :, None] * vv[:, None, :, :]
        pm = logits_minus.softmax(dim=-1)
        gate = (z2[:, :, lo:hi] > 0)
        demand[:, :, lo:hi] = ((y[:, :, None, :] - pm) * vv[:, None, :, :]).sum(-1) * gate
        brake[:, :, lo:hi] = ((p[:, :, None, :] - pm) * vv[:, None, :, :]).sum(-1) * gate
    result = {}
    for name, x in (('demand', demand), ('brake', brake), ('upward', direct)):
        result[name + '_mean'] = cpu(x.mean(1))
        result[name + '_rms'] = cpu(x.square().mean(1).sqrt())
    result['closure_max'] = cpu((demand - brake - direct).abs().amax(1))
    result['brake_min'] = cpu(brake.amin(1))
    return result


class Trace:
    def __init__(self, cifar, seeds, device, dest, arm, decomposition=True, centered=None):
        self.dest, self.seeds, self.device, self.arm = dest, seeds, device, arm
        self.decomposition = decomposition
        self.centered = centered
        self.forward = centered if centered is not None else B.forward
        dest.mkdir(parents=True, exist_ok=True)
        self.X = cifar.inputs('train', 'std', device)
        self.Y = cifar.train_y.to(device)
        self.rows = C.class_rows(cifar.train_y)
        self.plans = [C.task_plan(s) for s in seeds]
        gi = torch.cat([r[:10] for r in self.rows]).to(device)
        self.gi, self.gX, self.gY = cpu(gi), self.X[gi], self.Y[gi]
        self.idx3 = self.task_indices(3)
        self.X3, self.Y3 = self.X[self.idx3], self.Y[self.idx3]
        jj = torch.linspace(0, self.idx3.shape[1] - 1, 256).long().to(device)
        self.probe3_idx = self.idx3[:, jj]
        self.probe3_X, self.probe3_Y = self.X[self.probe3_idx], self.Y[self.probe3_idx]
        self.events, self.values, self.seed_rows, self.deltas = [], [], [], []
        self.fingerprints, self.batches, self.shock_rows, self.shock_logits = {}, {}, [], {}
        self.cohort = np.zeros((len(seeds), C.HIDDEN), dtype=bool)
        self.reset_count = np.zeros_like(self.cohort, dtype=np.int32)
        self.old_W3 = None
        self.intervention_applied = False
        self.coordinate_checks = []
        self.gauge_applied = False
        self.online_segments = []
        self.task3_hit50 = None
        self._before = None
        np.savez_compressed(dest / 'probe_indices.npz', global_indices=self.gi,
                            task3_indices=cpu(self.idx3), task3_gradient_indices=cpu(self.probe3_idx))

    def task_indices(self, task):
        return torch.stack([torch.cat([self.rows[c] for c in plan[task-1][1]])
                            for plan in self.plans]).to(self.device)

    def global_forward(self, P, act):
        return self.forward(P, self.gX.unsqueeze(0).expand(len(self.seeds), -1, -1), act, train=False)

    def fingerprint(self, label, state, save_state=False):
        arrays = {f'{family}{i}': cpu(x) for family in ('P', 'm', 'v')
                  for i, x in enumerate(state[family])}
        self.fingerprints[label] = {
            'all': {key: array_hash(a) for key, a in arrays.items()},
            'per_seed': {str(seed): {key: array_hash(a[r]) for key, a in arrays.items()}
                         for r, seed in enumerate(self.seeds)}}
        if save_state:
            np.savez_compressed(self.dest / f'{label}_state.npz', **arrays)

    def shocks(self, label, P, act, task):
        cur = self.task_indices(task)
        jj = torch.linspace(0, cur.shape[1] - 1, 256).long().to(self.device)
        cur = cur[:, jj]
        scopes = {
            'global': (self.gX.unsqueeze(0).expand(len(self.seeds), -1, -1),
                       self.gY.unsqueeze(0).expand(len(self.seeds), -1)),
            'upcoming_task3': (self.probe3_X, self.probe3_Y),
            f'current_task{task}': (self.X[cur], self.Y[cur]),
        }
        for scope, (xx, yy) in scopes.items():
            logits = self.forward(P, xx, act, train=False)[-1]
            self.shock_logits[f'{label}_{scope}'] = cpu(logits)
            # Float64 CE is diagnostic only; training remains float32 unchanged.
            loss = F.cross_entropy(logits.double().reshape(-1, C.N_CLASSES), yy.reshape(-1),
                                   reduction='none').view(len(self.seeds), -1).mean(1)
            hit = (logits.argmax(-1) == yy).float().mean(1)
            for r, seed in enumerate(self.seeds):
                row = dict(state=label, scope=scope, seed=seed, n_inputs=int(xx.shape[1]),
                           loss=float(loss[r]), accuracy=float(hit[r]),
                           logit_rms=float(logits[r].double().square().mean().sqrt()))
                for ref in ('reset_pre', 'zero_post'):
                    key = f'{ref}_{scope}'
                    if key in self.shock_logits:
                        other = torch.as_tensor(self.shock_logits[key][r], device=logits.device,
                                                dtype=torch.float64)
                        diff = logits[r].double() - other
                        row[f'logit_diff_rms_from_{ref}'] = float(diff.square().mean().sqrt())
                        row[f'logit_diff_max_from_{ref}'] = float(diff.abs().max())
                self.shock_rows.append(row)

    def coordinate_observations(self, P, act):
        """Forward checks on both fixed probes; detached copies before mutation."""
        result = {}
        for name, xx in (
            ('global', self.gX.unsqueeze(0).expand(len(self.seeds), -1, -1)),
            ('task3', self.X3)):
            _, _, z, _, logits = self.forward(P, xx, act, train=False)
            result[name] = (z.detach().clone(), logits.detach().clone())
        return result

    def check_coordinate(self, before, after, name, preactivation_expected_scale=1.0):
        for scope in before:
            z0, l0 = before[scope]
            z1, l1 = after[scope]
            logit_error = (l1 - l0).abs().amax(dim=(1, 2))
            gate_changed = ((z0 > 0) != (z1 > 0)).float().sum(dim=(1, 2))
            z_expected = z0.clone()
            mask = torch.as_tensor(self.cohort, device=z0.device)[:, None, :].expand_as(z0)
            z_expected[mask] *= preactivation_expected_scale
            z_error = (z1 - z_expected).abs().amax(dim=(1, 2))
            for r, seed in enumerate(self.seeds):
                self.coordinate_checks.append(dict(intervention=name, scope=scope, seed=seed,
                    logit_max_error=float(logit_error[r]), preactivation_max_error=float(z_error[r]),
                    gate_sign_changed=int(gate_changed[r])))
            if float(logit_error.max()) > 1e-5 or float(z_error.max()) > 1e-5 or bool(gate_changed.any()):
                raise RuntimeError(f'Forward-preserving intervention failed admission: {name}/{scope}; '
                                   f'logits={float(logit_error.max())}, z={float(z_error.max())}, '
                                   f'gate_changes={int(gate_changed.sum())}')

    def apply_centering(self, s):
        P, act = s['P'], s['act']
        before = self.coordinate_observations(P, act)
        hmean = self.global_forward(P, act)[1].mean(1)
        if self.arm == 'center_random':
            vectors = []
            for seed in self.seeds:
                digest = hashlib.sha256(f'{RUN}|input_origin_direction|{seed}'.encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:16], 'little'))
                a = rng.standard_normal(C.HIDDEN)
                vectors.append(a / np.linalg.norm(a))
            direction = torch.as_tensor(np.stack(vectors), dtype=hmean.dtype, device=hmean.device)
            origin = direction * torch.linalg.vector_norm(hmean, dim=1, keepdim=True)
        else:
            origin = hmean
        mask = torch.as_tensor(self.cohort, device=P[2].device)
        self.centered.offsets.copy_(origin[:, None, :] * mask[:, :, None])
        P[3].add_((P[2] * self.centered.offsets).sum(-1))
        after = self.coordinate_observations(P, act)
        self.check_coordinate(before, after, self.arm)
        np.savez_compressed(self.dest / 'center_origin.npz', origin=cpu(origin),
                            empirical_hmean=cpu(hmean), offsets=cpu(self.centered.offsets))

    def apply_gauge(self, s):
        if self.gauge_applied:
            raise RuntimeError('Gauge transformation was already applied')
        P, act = s['P'], s['act']
        c = 4.0 if self.arm == 'gauge4' else 0.25
        before = self.coordinate_observations(P, act)
        self.fingerprint('gauge_pre', s)
        self.shocks('gauge_pre', P, act, s['t'])
        rowmask = torch.as_tensor(self.cohort, device=P[2].device)
        wm = rowmask[:, :, None].expand_as(P[2])
        vm = rowmask[:, None, :].expand_as(P[4])
        P[2][wm] /= c
        P[3][rowmask] /= c
        P[4][vm] *= c
        # A gradient covector transforms inversely to parameter scaling.
        s['m'][2][wm] *= c
        s['m'][3][rowmask] *= c
        s['v'][2][wm] *= c * c
        s['v'][3][rowmask] *= c * c
        s['m'][4][vm] /= c
        s['v'][4][vm] /= c * c
        after = self.coordinate_observations(P, act)
        self.check_coordinate(before, after, self.arm, preactivation_expected_scale=1 / c)
        self.fingerprint('gauge_post', s)
        self.shocks('gauge_post', P, act, s['t'])
        self.gauge_applied = True
        self.record('gauge_post', s)
        self.save()

    @torch.no_grad()
    def __call__(self, event, **s):
        P, act, task, tc = s['P'], s['act'], s['t'], s['tc']
        if event == 'task_start':
            batch = cpu(s['batches'])
            self.batches[str(task)] = {str(seed): array_hash(batch[r])
                                       for r, seed in enumerate(self.seeds)}
            if task == 3:
                np.savez_compressed(self.dest / 'task3_batches.npz', batches=batch)
            return
        if event == 'reset_pre':
            masks = np.stack([cpu(m) for m in s['masks']], axis=1)
            self.reset_count += masks[:, 1].astype(np.int32)
            if tc != 1560:
                return
            if self.old_W3 is not None:
                raise RuntimeError('The first intervention reset was encountered twice')
            self.cohort = masks[:, 1].astype(bool)
            self.old_W3 = P[4].detach().clone()
            self.first_masks = masks.copy()
            self.fingerprint('reset_pre', s, save_state=self.arm == 'zero')
            self.shocks('reset_pre', P, act, task)
        if event == 'reset_outgoing_2':
            if tc != 1560:
                return
            if self.old_W3 is None or self.intervention_applied:
                raise RuntimeError('Missing first-reset state or repeated intervention')
            self.fingerprint('zero_post', s)
            self.shocks('zero_post', P, act, task)
            self.record('zero_post', s)
            replacement = replacement_columns(self.old_W3, self.seeds, self.arm)
            # Snapshot only the tensors the intervention is permitted to change.
            before_W3 = P[4].detach().clone()
            old_m, old_v = s['m'][4].clone(), s['v'][4].clone()
            mask = torch.as_tensor(self.cohort, device=P[4].device)[:, None, :].expand_as(P[4])
            if self.arm not in ZERO_READOUT_ARMS:
                P[4][mask] = replacement[mask]
            if not torch.equal(P[4][~mask], before_W3[~mask]):
                raise RuntimeError('The intervention modified an unselected outgoing column')
            if not torch.equal(s['m'][4], old_m) or not torch.equal(s['v'][4], old_v):
                raise RuntimeError('The intervention changed Adam moments')
            if torch.count_nonzero(s['m'][4][mask]) or torch.count_nonzero(s['v'][4][mask]):
                raise RuntimeError('Original selected outgoing moments were not zero')
            if self.centered is not None:
                self.apply_centering(s)
            self.intervention_applied = True
            self.fingerprint('intervention_post', s)
            self.shocks('intervention_post', P, act, task)
            np.savez_compressed(self.dest / 'cohort.npz', masks=self.first_masks,
                                old_W3=cpu(self.old_W3), replacement_W3=cpu(replacement),
                                direction=np.stack([independent_directions(seed, P[4].shape[1], P[4].shape[2])
                                                    for seed in self.seeds]))
            self.record('reset_post', s)
            self.save()
            return
        if event == 'before_step':
            _, h1, z2, _, _ = self.global_forward(P, act)
            self._before = dict(W=P[2].clone(), b=P[3].clone(), mu=h1.mean(1),
                                mean=z2.mean(1), momentW=s['m'][2].clone(), momentb=s['m'][3].clone())
            return
        if event == 'init':
            self.fingerprint('init', s)
        self.record(event, s)
        if event == 'after_step' and s['j'] + 1 == 50:
            self.fingerprint('task3_step50', s)
            self.task3_hit50 = cpu(s['acc_sum'])
        if event == 'after_step' and s['j'] + 1 == 50 and self.arm == 'zero':
            torch.save({**{key: [q.detach().cpu().clone() for q in s[key]] for key in ('P', 'm', 'v')},
                        'seeds': self.seeds, 'tasks': 2, 'task': 3, 'step_in_task': 50, 'tc': tc,
                        'first_mask': torch.from_numpy(self.cohort.copy())},
                       self.dest.parent / 'task3_step50.pt')
        if event == 'after_step' and s['j'] + 1 == 50 and self.arm in ('gauge4', 'gauge025'):
            self.apply_gauge(s)
        if event == 'task_end':
            if task == 3:
                hits = cpu(s['acc_sum'])
                self.online_segments = [dict(seed=seed, online_steps1_50=float(self.task3_hit50[r]) / 50,
                    online_steps51_780=float(hits[r] - self.task3_hit50[r]) / 730,
                    online_steps1_780=float(hits[r]) / 780) for r, seed in enumerate(self.seeds)]
            self.save()

    def record(self, event, s):
        P, act, task, tc = s['P'], s['act'], s['t'], s['tc']
        step = s['j'] + 1 if 'j' in s else (0 if event == 'init' else 780)
        meta = dict(index=len(self.events), event=event, task=task, tc=tc, step=step)
        _, h1, z2, _, _ = self.global_forward(P, act)
        data = {'global_' + k: v for k, v in unit_summary(z2).items()}
        if task <= 3:
            z3probe = self.forward(P, self.X3, act, train=False)[2]
            data.update({'task3_' + k: v for k, v in unit_summary(z3probe).items()})
        if event == 'task_end':
            zcur = self.forward(P, s['Xt'], act, train=False)[2]
            data.update({'current_' + k: v for k, v in unit_summary(zcur).items()})
        centered = P[4] - P[4].mean(dim=1, keepdim=True)
        data['v_centered_norm'] = cpu(torch.linalg.vector_norm(centered, dim=1))
        data['v_raw_norm'] = cpu(torch.linalg.vector_norm(P[4], dim=1))
        data['incoming_norm'] = cpu(torch.linalg.vector_norm(P[2], dim=2))
        data['bias'] = cpu(P[3])
        data['effective_bias'] = cpu(self.centered.effective(P)[3] if self.centered is not None else P[3])
        hmean = h1.mean(1)
        position_input = hmean[:, None, :]
        if self.centered is not None:
            position_input = position_input - self.centered.offsets
        data['global_center_leak'] = cpu((P[2] * position_input).sum(-1))
        data['global_weight_mean_cosine'] = cpu(
            (P[2] * hmean[:, None, :]).sum(-1) /
            (torch.linalg.vector_norm(P[2], dim=2) * torch.linalg.vector_norm(hmean, dim=1)[:, None]).clamp_min(1e-30))
        data['first_cohort'] = self.cohort.copy()
        data['reset_count'] = self.reset_count.copy()
        if self.decomposition and (event in ('zero_post', 'reset_post') or
                                   (event == 'after_step' and step in {1, 2, 5, 10, 25, 50, 100, 780})):
            _, _, zp, _, lp = self.forward(P, self.probe3_X, act, train=False)
            data.update({'ce_' + k: v for k, v in ce_decomposition(zp, lp, P[4], self.probe3_Y).items()})
        if event == 'after_step':
            before = self._before
            position_input = before['mu'][:, None, :]
            if self.centered is not None:
                position_input = position_input - self.centered.offsets
            own_w = ((P[2] - before['W']) * position_input).sum(-1)
            own_b = P[3] - before['b']
            upstream = torch.einsum('rij,rj->ri', P[2], h1.mean(1) - before['mu'])
            actual = z2.mean(1) - before['mean']
            gW, gb = s['grads'][2], s['grads'][3]
            c1, c2 = s['inv_c1'], s['inv_c2']
            dw = (s['v'][2] * c2).sqrt() + 1e-8
            db = (s['v'][3] * c2).sqrt() + 1e-8
            current_w = -C.LR * (.1 * gW * c1) / dw
            current_b = -C.LR * (.1 * gb * c1) / db
            memory_w = -C.LR * (.9 * before['momentW'] * c1) / dw
            memory_b = -C.LR * (.9 * before['momentb'] * c1) / db
            current = (current_w * position_input).sum(-1) + current_b
            memory = (memory_w * position_input).sum(-1) + memory_b
            self.deltas.append(dict(event=meta['index'], task=task, tc=tc, step=step,
                own_w=cpu(own_w), own_b=cpu(own_b), upstream=cpu(upstream), actual=cpu(actual),
                adam_current=cpu(current), adam_memory=cpu(memory),
                sgd_direction=cpu(-((gW * position_input).sum(-1) + gb)),
                bias_gradient=cpu(gb), closure=cpu(actual - own_w - own_b - upstream),
                adam_closure=cpu(own_w + own_b - current - memory)))
            self._before = None
        self.events.append(meta)
        self.values.append(data)
        for r, seed in enumerate(self.seeds):
            cohorts = {'all': np.ones(C.HIDDEN, dtype=bool)}
            if self.old_W3 is not None:
                cohorts.update(first_reset=self.cohort[r], survivor=~self.cohort[r],
                               first_reset_same_generation=self.cohort[r] & (self.reset_count[r] == 1))
            for label, mask in cohorts.items():
                row = dict(**meta, seed=seed, arm=self.arm, cohort=label, n_units=int(mask.sum()))
                if 'acc_sum' in s:
                    row['online_accuracy_prefix'] = float(s['acc_sum'][r]) / step
                if mask.any():
                    for key, value in data.items():
                        if key in ('first_cohort', 'reset_count'):
                            continue
                        row[key + '_mean'] = float(np.mean(value[r, mask]))
                        row[key + '_median'] = float(np.median(value[r, mask]))
                self.seed_rows.append(row)

    def save(self):
        (self.dest / 'events.json').write_text(json.dumps(dict(seeds=self.seeds, arm=self.arm,
            events=self.events, intervention_applied=self.intervention_applied,
            gauge_applied=self.gauge_applied,
            cohort_definition='L2 slots selected at the first tc=1560 reset; reset_count distinguishes later recycling',
            sd_definition='population standard deviation (torch correction=0)',
            ce_definition='per-example CE, fixed task3 256-image probe, float64 diagnostic'), indent=2))
        keys = sorted(set().union(*(d.keys() for d in self.values)))
        arrays = {}
        for key in keys:
            ref = next(d[key] for d in self.values if key in d)
            arrays[key] = np.stack([d.get(key, np.full(ref.shape, np.nan)) for d in self.values])
        np.savez_compressed(self.dest / 'units.npz', **arrays)
        if self.deltas:
            np.savez_compressed(self.dest / 'deltas.npz',
                                **{k: np.stack([d[k] for d in self.deltas]) for k in self.deltas[0]})
        H.write_csv(self.dest / 'per_seed.csv', self.seed_rows)
        H.write_csv(self.dest / 'shock.csv', self.shock_rows)
        H.write_csv(self.dest / 'online_segments.csv', self.online_segments)
        if self.shock_logits:
            np.savez_compressed(self.dest / 'shock_logits.npz', **self.shock_logits)
        (self.dest / 'fingerprints.json').write_text(json.dumps(self.fingerprints, indent=2))
        (self.dest / 'batch_hashes.json').write_text(json.dumps(self.batches, indent=2))
        (self.dest / 'coordinate_checks.json').write_text(json.dumps(self.coordinate_checks, indent=2))


def compare_csv_prefix(actual, reference, max_task):
    with Path(actual).open() as f:
        aa = [row for row in csv.DictReader(f) if int(row['task']) <= max_task]
    with Path(reference).open() as f:
        bb = {(row['seed'], row['task']): row for row in csv.DictReader(f)}
    differences = []
    for row in aa:
        ref = bb[(row['seed'], row['task'])]
        for key, value in row.items():
            if key not in ref or value == ref[key]:
                continue
            try:
                x, y = float(value), float(ref[key])
                if np.isnan(x) and np.isnan(y):
                    continue
            except ValueError:
                x, y = value, ref[key]
            differences.append(dict(seed=row['seed'], task=row['task'], metric=key,
                                    actual=x, reference=y))
    return dict(rows=len(aa), max_task=max_task, exact=not differences,
                n_different=len(differences), differences=differences)


def self_check():
    """CPU synthetic checks only: intervention, RNG isolation and CE identity."""
    old = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7) / 100
    old = torch.sin(old * 5)
    initial_rng = torch.get_rng_state().clone()
    target_norm = torch.linalg.vector_norm(old - old.mean(1, keepdim=True), dim=1)
    replacement = {arm: replacement_columns(old, [2, 7], arm) for arm in ARMS}
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert torch.equal(replacement['keep'], old)
    assert torch.count_nonzero(replacement['zero']) == 0
    for arm in ('rand01', 'rand03', 'rand10'):
        value = replacement[arm]
        assert torch.allclose(value.mean(1), torch.zeros_like(value.mean(1)), atol=2e-8)
        assert torch.allclose(torch.linalg.vector_norm(value, dim=1), ARMS[arm] * target_norm,
                              rtol=1e-6, atol=1e-7)
    assert torch.allclose(replacement['rand03'], 3 * replacement['rand01'], rtol=2e-6, atol=1e-7)
    gen = torch.Generator().manual_seed(5519)
    z = torch.randn((2, 11, 7), generator=gen, dtype=torch.float64, requires_grad=True)
    W = torch.randn((2, 5, 7), generator=gen, dtype=torch.float64)
    base = torch.randn((2, 11, 5), generator=gen, dtype=torch.float64)
    logits = base + torch.einsum('rnu,rcu->rnc', z.relu(), W)
    y = torch.randint(0, 5, (2, 11), generator=gen)
    loss = F.cross_entropy(logits.reshape(-1, 5), y.reshape(-1), reduction='sum')
    grad = torch.autograd.grad(loss, z)[0]
    result = ce_decomposition(z.detach(), logits.detach(), W, y)
    error = float(np.abs(result['upward_mean'] + cpu(grad).mean(1)).max())
    assert error < 1e-12
    assert result['closure_max'].max() < 1e-12
    assert result['brake_min'].min() >= -1e-12
    _, source = instrumented(lambda *args, **kwargs: None)
    # The centered parametrization preserves predictions and changes dL/dW2
    # by exactly -dL/db2 times its fixed input-origin vector.
    pp = [torch.randn(shape, generator=gen, dtype=torch.float64, requires_grad=True)
          for shape in ((2, 7, 3), (2, 7), (2, 7, 7), (2, 7), (2, 5, 7), (2, 5))]
    xx = torch.randn((2, 11, 3), generator=gen, dtype=torch.float64)
    act = C.make_act('R')
    original = B.forward(pp, xx, act)[-1]
    center = CenteredForward(2, 7, torch.device('cpu'))
    center.offsets = torch.randn((2, 7, 7), generator=gen, dtype=torch.float64)
    shifted = [q.detach().clone().requires_grad_(True) for q in pp]
    with torch.no_grad():
        shifted[3].add_((shifted[2] * center.offsets).sum(-1))
    converted = center(shifted, xx, act)[-1]
    center_error = float((converted - original).detach().abs().max())
    assert center_error < 1e-12
    go = torch.autograd.grad(original.square().sum(), pp)
    gc = torch.autograd.grad(converted.square().sum(), shifted)
    gradient_error = float((gc[2] - (go[2] - go[3][:, :, None] * center.offsets)).abs().max())
    assert gradient_error < 1e-10
    _, _ = instrumented(lambda *args, **kwargs: None, centered=center)
    return dict(status='passed', synthetic_ce_autograd_mean_max_error=error,
                ce_closure_max=float(result['closure_max'].max()),
                minimum_brake=float(result['brake_min'].min()),
                centered_prediction_max_error=center_error,
                centered_gradient_transform_max_error=gradient_error,
                source_injection_sha=hashlib.sha256(source.encode()).hexdigest(),
                checks=['original source hash', 'source injection compilation',
                        'class-centered matched norms', 'shared random directions',
                        'training RNG isolation', 'CE decomposition versus autograd',
                        'centered prediction identity', 'centered incoming-gradient transform'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=tuple(ARMS), default='zero')
    ap.add_argument('--group', type=int, choices=(0, 10), default=0)
    ap.add_argument('--tasks', type=int, default=3)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--plain-check', action='store_true', help='also replay unchanged engine and compare final tensors; zero arm only')
    ap.add_argument('--no-decomposition', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if not 3 <= args.tasks <= 30:
        ap.error('--tasks must be between 3 and 30')
    if args.plain_check and args.arm != 'zero':
        ap.error('--plain-check is meaningful only for --arm zero')
    if args.arm in ('center', 'center_random') and args.tasks != 3:
        ap.error('Centering arms are specified for the first 3 tasks only')
    seeds = list(range(args.group, args.group + 10))
    dest = args.out or OUT / 'raw' / f'{args.arm}_s{args.group}-{args.group+9}'
    if (dest / 'environment.json').exists() or (dest / 'replay/per_task.csv').exists():
        raise RuntimeError(f'Refusing to overwrite an existing training output: {dest}')
    device = H.setup('cuda')
    cifar = C.Cifar100(DATA)
    cfg = S.make_cfg('redo', tau='0.1', period=1560)
    centered = CenteredForward(len(seeds), C.HIDDEN, device) if args.arm in ('center', 'center_random') else None
    trace = Trace(cifar, seeds, device, dest / 'trace', args.arm, not args.no_decomposition, centered=centered)
    run, source = instrumented(trace, centered=centered)
    (dest / 'instrumented_source.py').write_text(source)
    final = {}
    provenance = run(cfg, seeds, args.tasks, device, dest / 'replay', cifar=cifar,
                     fresh=False, graph=True, final=final)
    trace.save()
    if not trace.intervention_applied:
        raise RuntimeError('The registered tc=1560 intervention was not encountered')
    np.savez_compressed(dest / 'final_state.npz', **{
        f'{family}{i}': cpu(q) for family in ('P', 'm', 'v') for i, q in enumerate(final[family])})
    torch.save({**{key: [q.detach().cpu().clone() for q in final[key]] for key in ('P', 'm', 'v')},
                'seeds': seeds, 'tasks': args.tasks, 'task': args.tasks,
                'step_in_task': C.STEPS_PER_TASK, 'tc': args.tasks * C.STEPS_PER_TASK,
                'first_mask': torch.from_numpy(trace.cohort.copy()),
                'center_offsets': None if centered is None else centered.offsets.detach().cpu().clone()},
               dest / 'final.pt')
    reference = REFERENCE / f'redo_s{args.group}-{args.group+9}' / 'replay/per_task.csv'
    prefix = min(args.tasks, 30) if args.arm == 'zero' else 2
    comparison = compare_csv_prefix(dest / 'replay/per_task.csv', reference, prefix)
    (dest / 'reproduction.json').write_text(json.dumps(comparison, indent=2))
    plain_result = None
    if args.plain_check:
        other = {}
        S.run(cfg, seeds, args.tasks, device, dest / 'plain', cifar=cifar,
              fresh=False, graph=True, final=other)
        plain_result = dict(
            tensors={key: all(torch.equal(x, y) for x, y in zip(final[key], other[key]))
                     for key in ('P', 'm', 'v')},
            per_task=compare_csv_prefix(dest / 'replay/per_task.csv', dest / 'plain/per_task.csv', args.tasks))
        (dest / 'plain_check.json').write_text(json.dumps(plain_result, indent=2))
    environment = dict(
        run_id=RUN, arm=args.arm, intervention_only_tc=1560, intervention_only_layer=2,
        gauge_tc=1610 if args.arm in ('gauge4', 'gauge025') else None,
        centered_coordinate=args.arm in ('center', 'center_random'),
        seeds=seeds, tasks=args.tasks, graph=True, torch=torch.__version__, cuda=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        threads=torch.get_num_threads(),
        git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        engine_sha=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
        instrumentation_sha=hashlib.sha256(source.encode()).hexdigest(),
        runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), data_sha=cifar.sha256,
        independent_direction_rng='NumPy PCG64; SHA256(run_id|outgoing_direction|seed) first128bits little-endian',
        moment_policy='All original ReDo resets retained. Centering retains original moments. '
                      'Gauge transforms first moments by inverse parameter scaling, second moments by its square; clock retained.',
        original=provenance)
    (dest / 'environment.json').write_text(json.dumps(environment, indent=2))
    print('REPRODUCTION', json.dumps({k: v for k, v in comparison.items() if k != 'differences'}), flush=True)
    if not comparison['exact']:
        raise RuntimeError('Paired baseline prefix differs; see reproduction.json')
    if plain_result is not None and (not all(plain_result['tensors'].values()) or not plain_result['per_task']['exact']):
        raise RuntimeError('Read-only instrumentation did not reproduce the plain engine exactly')


if __name__ == '__main__':
    main()
