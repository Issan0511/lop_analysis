#!/usr/bin/env python3
"""Preregistered CH′/CH′+B four-layer activation-response exchange.

Only the run CLI reads real data; it requires a committed, pushed preregistration
and a source-bound successful GPU preflight. Importing this module is CPU-safe.
"""
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

EXPERIMENT = 'cnn_response_chb_1021'
SPEC_PATH = 'specs/spec_cnn_response_chb_1021.md'
SEEDS = list(range(10, 20))
MAIN_ARMS = ('CH_self', 'CH_swap', 'CHB_self', 'CHB_swap')
DIAG_STEPS = (0, 100, 750, 3000, 15000, 30000)
KAPPA = .2
DEFAULT_SOURCE = Path('/home/issan/Projects/obsidian-research-data/cnn_chprime_bias_1018/host_gpu_recovery_1018/results/cnn_chprime_bias_1018')
DEFAULT_MANIFEST = Path('/home/issan/Projects/claude/wt/cnn_chprime_bias_1018/results/cnn_chprime_bias_1018/backup_manifest.json')


def phi(z):
    return torch.clamp(z, min=0.)


def dphi(z):
    """Native clamp derivative, including the derivative 1 at exactly zero."""
    return (z >= 0).to(z.dtype)


class AnchorAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frozen, diff):
        return torch.where(diff == 0, frozen, frozen + diff)

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def stable_relu_diff(zd, delta):
    """clamp(zd+delta)-clamp(zd), avoiding cancellation in the positive branch.

    Selection includes zero on the positive branch, preserving torch.clamp's
    native derivative 1 there. References are detached by Field, never optimized.
    """
    ze = zd + delta
    return torch.where(zd >= 0, torch.where(ze >= 0, delta, -zd), phi(ze))


class ResponseDoors(CF.FeatDoors):
    """Raw-z variance recorder and EMA of the *actual* artificial activity."""
    @torch.no_grad()
    def update_actual(self, zs, activities):
        for l, z in enumerate(zs):
            self.V[l].mul_(1-self.beta).add_(self.beta * self.batch_var(z, l))
        if self.h_on:
            for l, a in enumerate(activities):
                bm = F.max_pool2d(a, CN.POOL, CN.POOL).mean(0).view(self.R, -1) if l < 2 else a.mean(1)
                old = self.m[l].clone()
                self.m[l].mul_(1-self.beta_h).add_(bm, alpha=self.beta_h)
                self.hupd[:, l].add_((self.m[l] != old).any(1).to(self.hupd.dtype))


def frozen_act(act):
    result = ResponseDoors(act.R, 'feat', act.m[0].device, beta=act.beta, beta_h=act.beta_h)
    result.load_state(act.state())
    return result


class Field:
    """Four frozen *natural* receiver/donor maps with same image/channel indices.

    Parameters and centering states are copied on construction. Production uses
    cached maps computed in the registered batch size, so anchor values are fixed
    across training and diagnostic chunk sizes. Small synthetic checks may call
    anchors(xb) directly without constructing a cache.
    """
    def __init__(self, receiver_P, receiver_act, donor_P, donor_act):
        self.P0 = [p.detach().clone() for p in receiver_P]
        self.PD = [p.detach().clone() for p in donor_P]
        self.act0, self.actD = frozen_act(receiver_act), frozen_act(donor_act)
        self.cache = None
        self.cache_hash = None

    @torch.no_grad()
    def anchors(self, xb):
        ro = CF.forward(self.P0, xb, self.act0)
        do = CF.forward(self.PD, xb, self.actD)
        return ([ro[2*l].detach() for l in range(4)], [do[2*l].detach() for l in range(4)])

    @staticmethod
    def apply(z, z0, zd):
        # All arms use f64 affine-offset intermediates; activity remains native
        # f32. This prevents self-field subtract/add roundoff changing pool
        # winners and accumulating into a different trajectory. No self bypass.
        z0, zd = z0.detach().double(), zd.detach().double()
        delta = z.double() - z0
        ze = zd + delta
        activity = AnchorAdd.apply(phi(z0), stable_relu_diff(zd, delta)).to(z.dtype)
        return activity, ze, dphi(ze).to(z.dtype)

    @torch.no_grad()
    def build_cache(self, X, image_ids=None, chunk=CN.BATCH):
        R, N = X.shape[:2]
        maps = [[], []]
        for lo in range(0, N, chunk):
            refs = self.anchors(X[:, lo:lo+chunk])
            for kind in range(2):
                maps[kind].append([image_layout(z, l < 2, R).contiguous() for l, z in enumerate(refs[kind])])
        self.cache = [[torch.cat([batch[l] for batch in maps[k]], dim=1) for l in range(4)] for k in range(2)]
        self.image_ids = None if image_ids is None else image_ids.detach().clone()
        self.cache_hash = tree_hash({'maps': self.cache, 'image_ids': self.image_ids, 'chunk': chunk})
        self.cache_chunk = chunk

    def cache_fingerprint(self):
        return tree_hash({'maps': self.cache, 'image_ids': self.image_ids, 'chunk': self.cache_chunk})

    def refs_ids(self, ids):
        if self.cache is None:
            raise RuntimeError('Frozen map cache is not initialized')
        R, B = ids.shape
        ar = torch.arange(R, device=ids.device)[:, None]
        refs = []
        for maps in self.cache:
            zs = []
            for l, z in enumerate(maps):
                v = z[ar, ids]
                if l < 2:
                    side = 32 if l == 0 else 16
                    v = v.permute(1, 0, 2, 3).reshape(B, R*CN.CHANNELS, side, side)
                zs.append(v)
            refs.append(zs)
        return tuple(refs)

    def state(self):
        return cpu({'P0': self.P0, 'PD': self.PD, 'act0': self.act0.state(), 'actD': self.actD.state(),
                    'cache_hash': self.cache_hash, 'cache_chunk': getattr(self, 'cache_chunk', None),
                    'image_ids': getattr(self, 'image_ids', None)})

    def load_state(self, st):
        if self.cache is not None and self.cache_fingerprint() != self.cache_hash:
            raise RuntimeError('Frozen field cache was mutated')
        if tree_hash(self.state()) != tree_hash(st):
            raise RuntimeError('Frozen field state differs; refusing a different intervention on resume')


def forward(P, xb, act, field=None, refs=None, with_inputs=False):
    """Return (out, effective_zs, native_effective_derivatives[, layer_inputs]).

    out follows CF.forward's nine-tensor convention and contains raw z and the
    actual uncentered activities. Centering is always the current receiver EMA.
    """
    if field is None:
        out, inputs = CF.forward(P, xb, act, with_inputs=True)
        result = (out, [out[2*l] for l in range(4)], [dphi(out[2*l]) for l in range(4)])
        return (*result, inputs) if with_inputs else result
    if refs is None:
        refs = field.anchors(xb)
    z0s, zds = refs
    R, B = xb.shape[:2]
    value = xb.transpose(0, 1).reshape(B, R*3, 32, 32)
    out, eff, deriv, inputs = [], [], [], []
    for l in range(4):
        inputs.append(value)
        w, b = P[2*l:2*l+2]
        if l < 2:
            z = F.conv2d(value, w.reshape(R*CN.CHANNELS, -1, CN.KERNEL, CN.KERNEL),
                         b.reshape(-1), padding=CN.PAD, groups=R)
        else:
            z = torch.baddbmm(b[:, None, :], value, w.transpose(1, 2))
        a, ze, de = field.apply(z, z0s[l], zds[l])
        out.extend((z, a)); eff.append(ze); deriv.append(de)
        if l < 2:
            value = F.max_pool2d(a, CN.POOL, CN.POOL) - act.sub_map(l, act.m)
            if l == 1:
                value = value.reshape(B, R, CN.FLAT).transpose(0, 1)
        else:
            value = a - act.m[l][:, None, :]
    out.append(torch.baddbmm(P[9][:, None, :], value, P[8].transpose(1, 2)))
    result = (tuple(out), eff, deriv)
    return (*result, tuple(inputs)) if with_inputs else result


def image_layout(z, conv, R):
    return z.reshape(z.shape[0], R, CN.CHANNELS, -1).permute(1, 0, 2, 3) if conv else z


def image_width(q, conv, R):
    variance = image_layout(q, conv, R).double().var(dim=1, correction=0)
    return variance.mean(-1).sqrt() if conv else variance.sqrt()


@torch.no_grad()
def clip_bias(b, width, kappa=KAPPA):
    cap = kappa * width
    before = b.clone()
    hit = b.double().abs() > cap
    b.copy_(torch.where(hit, b.double().clamp(-cap, cap).to(b.dtype), b))
    return before, hit


@torch.no_grad()
def project_bias(P, xb, act, field=None, refs=None, kappa=KAPPA):
    """Adam -> actual-activity EMA -> sequential biasless-q width projection.

    Corrected upstream biases and current artificial activities generate each
    next input. Raw q widths, never effective z widths, define the clipping cap.
    """
    if field is not None and refs is None:
        refs = field.anchors(xb)
    R, B = xb.shape[:2]
    value = xb.transpose(0, 1).reshape(B, R*3, 32, 32)
    trace = []
    for l in range(4):
        w, b = P[2*l:2*l+2]
        q = (F.conv2d(value, w.reshape(R*CN.CHANNELS, -1, CN.KERNEL, CN.KERNEL), None,
                      padding=CN.PAD, groups=R) if l < 2 else torch.bmm(value, w.transpose(1, 2)))
        width = image_width(q, l < 2, R)
        before, hit = clip_bias(b, width, kappa)
        good = torch.isfinite(image_layout(q, l < 2, R)).flatten(1).all(1)
        good = good & torch.isfinite(width).all(1) & torch.isfinite(before).all(1) & torch.isfinite(b).all(1)
        trace.append(dict(s=width, before=before, after=b.clone(), hit=hit, finite=good))
        z = q + (b.reshape(1, R*CN.CHANNELS, 1, 1) if l < 2 else b[:, None, :])
        a = phi(z) if field is None else field.apply(z, refs[0][l], refs[1][l])[0]
        if l < 2:
            value = F.max_pool2d(a, CN.POOL, CN.POOL) - act.sub_map(l, act.m)
            if l == 1:
                value = value.reshape(B, R, CN.FLAT).transpose(0, 1)
        else:
            value = a - act.m[l][:, None, :]
    return trace


class Data:
    def __init__(self, seeds=SEEDS, device=torch.device('cpu'), source_root=DEFAULT_SOURCE,
                 manifest_path=DEFAULT_MANIFEST, data_dir=None):
        from src.cnn_response_chb_sources_1021 import audit_sources, json_sha
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
        for arm in ('CHf', 'CHf_B'):
            if subset_digest != metadata['arms'][arm]['subsets_sha256']:
                raise RuntimeError(f'Runtime global image IDs differ from archived parent: {arm}')
        self.X = torch.stack([CN.images(self.cifar, ids, self.device) for ids in self.ids])
        self.X.sub_(self.center.reshape(1, 1, 3, 1, 1))
        for key in ('g_lab', 'g_batch'):
            for s in self.seeds:
                if not torch.equal(self.sources['CHf'][key][s], self.sources['CHf_B'][key][s]):
                    raise RuntimeError(f'Parent task streams disagree: {key}/{s}')
        source_data = metadata.get('data_sha256')
        if source_data is not None and source_data != self.cifar.sha256:
            raise RuntimeError('CIFAR SHA differs from source run')
        self.identity = {'seeds': self.seeds, 'sources': metadata, 'data_sha256': self.cifar.sha256,
                         'global_image_ids_sha256': tree_hash(self.ids), 'center': cpu(self.center).tolist()}

    def inputs(self, arm):
        return self.X


class Engine(CF.Bundle):
    """Complete deterministic bundle; constructing from synthetic Data needs no CIFAR."""
    def __init__(self, data, name, graph=True):
        if name not in MAIN_ARMS + ('CH_nat', 'CHB_nat'):
            raise ValueError(name)
        self.name, self.data = name, data
        self.receiver = 'CHf_B' if name.startswith('CHB_') else 'CHf'
        self.donor = self.receiver if name.endswith(('_self', '_nat')) else ('CHf' if self.receiver == 'CHf_B' else 'CHf_B')
        self.enabled = self.receiver == 'CHf_B'
        self.device = torch.device(data.device)
        self.slots = [(self.receiver, s) for s in data.seeds]
        self.useeds = list(data.seeds)
        self.R, self.lr, self.epochs = len(data.seeds), .001, 400
        self.arm = self.receiver
        self.X = data.inputs(self.receiver)
        source = data.sources[self.receiver]
        self.P = [p.to(self.device).detach().clone().requires_grad_(True) for p in source['P']]
        self.m, self.v = [[q.to(self.device).clone() for q in source[key]] for key in ('m', 'v')]
        self.act = ResponseDoors(self.R, 'feat', self.device)
        self.tc = torch.zeros((), dtype=torch.float64, device=self.device)
        self.bad = torch.zeros(self.R, dtype=torch.bool, device=self.device)
        self.g_lab = {s: H.stream('rlcc_labels', s) for s in self.useeds}
        self.g_batch = {s: H.stream('rlcc_batch', s) for s in self.useeds}
        self.Y = torch.zeros(self.R, self.X.shape[1], dtype=torch.long, device=self.device)
        self.order = torch.zeros_like(self.Y)
        self.ar = torch.arange(self.R, device=self.device)[:, None]
        self.acc_ep = torch.zeros(self.R, device=self.device)
        self.b1 = torch.tensor(.9, dtype=torch.float64, device=self.device)
        self.b2 = torch.tensor(.999, dtype=torch.float64, device=self.device)
        self.graph, self.cg, self.upd_mask = bool(graph and self.device.type == 'cuda'), None, None
        E.Bundle.load_full_state(self, source)
        self.field = None
        if not name.endswith('_nat'):
            ds = data.sources[self.donor]
            da = ResponseDoors(self.R, 'feat', self.device)
            da.load_state({k: [t.to(self.device) for t in vs] for k, vs in ds['act'].items()})
            self.field = Field(self.P, self.act, [p.to(self.device) for p in ds['P']], da)
            self.field.build_cache(self.X, getattr(data, 'ids', None))
        self.acc_steps = torch.zeros(self.R, CN.STEPS_PER_EPOCH, device=self.device)
        self.ce_steps = torch.zeros_like(self.acc_steps)
        self.trace = [{key: torch.zeros(CN.STEPS_PER_EPOCH, self.R, width, dtype=dtype, device=self.device)
                       for key, dtype in (('s', torch.float64), ('before', torch.float32),
                                          ('after', torch.float32), ('hit', torch.bool))} for width in CN.WIDTHS]
        # Per-step layer summaries are retained in checkpoints and the curve archive.
        self.update_trace = torch.zeros(CN.STEPS_PER_EPOCH, self.R, 5, 5, device=self.device)
        self.last_grads = [torch.zeros_like(p) for p in self.P]

    def refs_ids(self, ids):
        return None if self.field is None else self.field.refs_ids(ids)

    def forward_ids(self, ids):
        return forward(self.P, self.X[self.ar, ids], self.act, self.field, self.refs_ids(ids))

    def new_order(self):
        perms = {s: torch.randperm(self.X.shape[1], generator=self.g_batch[s]) for s in self.useeds}
        self.order.copy_(torch.stack([perms[s] for s in self.useeds]))

    def new_labels(self):
        # Real N=1200 exactly preserves CN.task_labels; synthetic smaller data are supported.
        labs = {s: torch.randint(CN.N_CLASSES, (self.X.shape[1],), generator=self.g_lab[s]) for s in self.useeds}
        self.Y.copy_(torch.stack([labs[s] for s in self.useeds]))

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
            if self.enabled:
                rows = project_bias(self.P, xb, self.act, self.field, refs)
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
            for value in (*self.P, *self.m, *self.v, *self.act.m, *self.act.V):
                self.bad.logical_or_(~torch.isfinite(value).flatten(1).all(1))
            self.acc_steps[:, j].copy_(torch.where(self.bad, torch.zeros_like(hit), hit))
            self.ce_steps[:, j].copy_(torch.where(self.bad, torch.full_like(lossv, float('inf')), lossv))
            self.acc_ep.add_(self.acc_steps[:, j])

    def _state_tensors(self):
        return (*E.Bundle._state_tensors(self), self.acc_steps, self.ce_steps, self.update_trace,
                *self.last_grads, *(v for row in self.trace for v in row.values()))

    def snapshot(self):
        return cpu({'state': self.full_state(), 'Y': self.Y, 'order': self.order,
                    'acc_steps': self.acc_steps, 'ce_steps': self.ce_steps, 'acc_ep': self.acc_ep,
                    'update_trace': self.update_trace, 'last_grads': self.last_grads, 'trace': self.trace,
                    'field': None if self.field is None else self.field.state(), 'name': self.name})

    def restore(self, st):
        if st['name'] != self.name:
            raise RuntimeError('Cannot restore a different experimental arm')
        if (self.field is None) != (st['field'] is None):
            raise RuntimeError('Frozen field mode differs')
        if self.field is not None:
            self.field.load_state(st['field'])
        self.load_full_state(st['state'])
        with torch.no_grad():
            for key in ('Y', 'order', 'acc_steps', 'ce_steps', 'acc_ep', 'update_trace'):
                getattr(self, key).copy_(st[key])
            for a, b in zip(self.last_grads, st['last_grads']):
                a.copy_(b)
            for dst, src in zip(self.trace, st['trace']):
                for key in dst:
                    dst[key].copy_(src[key])

    def full_state(self):
        return E.Bundle.full_state(self) | {'enabled': self.enabled, 'kappa': KAPPA}

    def load_full_state(self, st):
        if st.get('enabled') != self.enabled or st.get('kappa') != KAPPA:
            raise RuntimeError('B prescription differs on restore')
        E.Bundle.load_full_state(self, st)


class Moments:
    """Per-position image-axis population moments via parallel Welford."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, value):
        x = value.double()
        n, mean = x.shape[1], x.mean(1)
        m2 = (x-mean[:, None]).square().sum(1)
        if not self.n:
            self.n, self.mean, self.m2 = n, mean, m2
        else:
            delta = mean-self.mean
            self.m2 += m2 + delta.square()*(self.n*n/(self.n+n))
            self.mean += delta*(n/(self.n+n))
            self.n += n

    def var(self):
        return self.m2/self.n


def runtime_rng_state(device):
    ns = np.random.get_state()
    result = {'torch_cpu': torch.get_rng_state(),
              'numpy': (ns[0], torch.from_numpy(ns[1].copy()), *ns[2:])}
    if torch.device(device).type == 'cuda':
        result['torch_cuda'] = torch.cuda.get_rng_state_all()
    return result


@torch.no_grad()
def diagnose(B, update, chunk=CN.BATCH):
    """Full fixed set, raw/effective per-position diagnostics, immutable state/RNG.

    Returns scalar rows, per-channel/unit CSV rows, and full positional NPZ arrays.
    Gate agreement concerns activation derivatives only, never entire Jacobians.
    """
    before = tree_hash(B.snapshot())
    rng_before = tree_hash(runtime_rng_state(B.device))
    cache_before = B.field.cache_fingerprint() if B.field is not None else None
    if B.field is not None and cache_before != B.field.cache_hash:
        raise RuntimeError('Frozen anchor cache differs from its initialization')
    R, N, dev = B.R, B.X.shape[1], B.device
    stats = {(l, kind): {key: Moments() for key in ('z', 'd', 'positive', 'zero', 'delta', 'offset', 'fixed_offset', 'agree_receiver', 'agree_donor')}
             for l in range(4) for kind in ('raw', 'effective')}
    pool = {(l, kind): Moments() for l in range(2) for kind in ('raw', 'effective')}
    grams = [torch.zeros(R, w, w, dtype=torch.float64, device=dev) for w in CN.WIDTHS]
    correct = torch.zeros(R, dtype=torch.float64, device=dev)
    ce = torch.zeros_like(correct)
    for start in range(0, N, chunk):
        ids = torch.arange(start, min(start+chunk, N), device=dev)[None, :].expand(R, -1)
        n = ids.shape[1]
        out, effective, derivatives = B.forward_ids(ids)
        refs = B.refs_ids(ids)
        if refs is None:
            refs = ([out[2*l] for l in range(4)], [out[2*l] for l in range(4)])
        labels = B.Y[B.ar, ids]
        correct += (out[8].argmax(-1) == labels).sum(1)
        ce += F.cross_entropy(out[8].reshape(-1, CN.N_CLASSES), labels.reshape(-1), reduction='none').view(R, n).double().sum(1)
        for l in range(4):
            raw, a = out[2*l:2*l+2]
            delta, offset = raw-refs[0][l], a-phi(raw)
            fixed_offset = phi(refs[0][l])-phi(refs[1][l])
            for kind, z, d in (('raw', raw, dphi(raw)), ('effective', effective[l], derivatives[l])):
                record = {'z': z, 'd': d, 'positive': (z > 0).float(), 'zero': (z == 0).float(),
                          'delta': delta, 'offset': offset, 'fixed_offset': fixed_offset,
                          'agree_receiver': (d == dphi(refs[0][l])).float(),
                          'agree_donor': (d == dphi(refs[1][l])).float()}
                for key, value in record.items():
                    stats[l, kind][key].add(image_layout(value, l < 2, R))
                if l < 2:
                    winners = F.max_pool2d(a, CN.POOL, CN.POOL, return_indices=True)[1]
                    dp = d.flatten(2).gather(2, winners.flatten(2))
                    pool[l, kind].add(dp.reshape(n, R, CN.CHANNELS, -1).permute(1, 0, 2, 3))
            ar = image_layout(a, l < 2, R)
            ar = ar.permute(0, 1, 3, 2).reshape(R, -1, CN.CHANNELS) if l < 2 else ar
            grams[l] += ar.double().transpose(1, 2) @ ar.double()
    rows = [{'arm': B.name, 'seed': seed, 'update': update,
             'memo_acc': 0. if bool(B.bad[r]) else float(correct[r]/N),
             'memo_ce': float('inf') if bool(B.bad[r]) else float(ce[r]/N),
             'failed': bool(B.bad[r])} for r, seed in enumerate(B.data.seeds)]
    arrays, units = {}, []
    bc2 = (1-torch.pow(B.b2, B.tc)).float()
    for l, tag in enumerate(CN.SITES):
        for kind in ('raw', 'effective'):
            sm = stats[l, kind]
            mu, sd = sm['z'].mean, sm['z'].var().clamp_min(0).sqrt()
            metrics = {'m': mu, 'sd': sd, 'r': mu/sd, 'p': sm['positive'].mean,
                       'native_mean_d': sm['d'].mean, 'native_var_d': sm['d'].var(),
                       'exact_z0': sm['zero'].mean, 'zero_d': 1-sm['d'].mean,
                       'gate_agreement_receiver': sm['agree_receiver'].mean,
                       'gate_agreement_donor': sm['agree_donor'].mean,
                       'delta_mean': sm['delta'].mean,
                       'delta_rms': (sm['delta'].var()+sm['delta'].mean.square()).sqrt(),
                       'activity_offset_mean': sm['offset'].mean,
                       'activity_offset_rms': (sm['offset'].var()+sm['offset'].mean.square()).sqrt(),
                       'fixed_anchor_offset_mean': sm['fixed_offset'].mean,
                       'fixed_anchor_offset_rms': (sm['fixed_offset'].var()+sm['fixed_offset'].mean.square()).sqrt()}
            if l < 2:
                metrics.update(pool_mean_d=pool[l, kind].mean, pool_var_d=pool[l, kind].var())
            for key, value in metrics.items():
                arrays[f'{tag}_{kind}_{key}'] = value.cpu().numpy()
                per_unit = value.mean(-1) if l < 2 else value
                for r in range(R):
                    rows[r][f'{tag}_{kind}_{key}_mean'] = float(value[r].mean())
                    rows[r][f'{tag}_{kind}_{key}_median'] = float(value[r].median())
            for r, seed in enumerate(B.data.seeds):
                rows[r][f'{tag}_{kind}_p_below_005_fraction'] = float((metrics['p'][r] < .005).double().mean())
                for unit in range(CN.WIDTHS[l]):
                    units.append({'arm': B.name, 'seed': seed, 'update': update, 'site': tag,
                                  'kind': kind, 'unit': unit,
                                  **{key: float(value[r, unit].mean()) for key, value in metrics.items()}})
        arrays[f'{tag}_ema'] = cpu(B.act.m[l]).numpy()
        arrays[f'{tag}_V_raw'] = cpu(B.act.V[l]).numpy()
        w, b = B.P[2*l:2*l+2]
        denominator = (B.v[2*l]/bc2).sqrt()+1e-8
        arrays[f'{tag}_bias'] = cpu(b).numpy()
        arrays[f'{tag}_w_norm'] = cpu(w.flatten(2).norm(dim=2)).numpy()
        arrays[f'{tag}_adam_denom_unit_mean'] = cpu(denominator.flatten(2).mean(2)).numpy()
        for r in range(R):
            gram = grams[l][r].cpu()
            rank = float('nan')
            if torch.isfinite(gram).all():
                sv = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
                if sv.sum() > 0:
                    p = sv/sv.sum()
                    rank = float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum()))
                else:
                    rank = 0.
            rows[r][f'{tag}_activity_effective_rank'] = rank
            rows[r][f'{tag}_weight_norm'] = float(w[r].norm())
            rows[r][f'{tag}_bias_rms'] = float(b[r].square().mean().sqrt())
            rows[r][f'{tag}_adam_denom_mean'] = float(denominator[r].mean())
            rows[r][f'{tag}_B_fire_fraction_latest75_updates'] = float(B.trace[l]['hit'][:, r].float().mean()) if B.enabled and update else 0.
    cache_after = B.field.cache_fingerprint() if B.field is not None else None
    if tree_hash(B.snapshot()) != before or tree_hash(runtime_rng_state(B.device)) != rng_before or cache_before != cache_after:
        raise RuntimeError('Diagnostic changed mutable training state, frozen maps, or RNG')
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
    for pattern in ('tests/test_cnn_response_chb_1021.py', 'analysis/cnn_response_chb_1021/*.py',
                    'src/cnn_response_chb_sources_1021.py', 'src/cnn_chprime_bias_1018.py'):
        files.update(ROOT.glob(pattern))
    return {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(files) if p.exists()}


def configure_runtime(device='cuda'):
    """Called only after preregistration and exclusive GPU/output locks."""
    dev = H.setup(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if dev.type == 'cuda':
        if torch.__version__ != '2.13.0+cu130':
            raise RuntimeError(f'Registered runtime requires torch 2.13.0+cu130, got {torch.__version__}')
        from torch.autograd.graph import set_override_stale_capture_stream
        set_override_stale_capture_stream(True)
    return dev


def numerical_context():
    return {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'override_stale_capture_stream': True,
            'anchor_arithmetic': 'float64_offset_then_float32_activity', 'host': platform.node()}


def guard_release(release):
    """Fail closed on drafts, unpushed release, dirty code, or source drift."""
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], text=True, capture_output=True, check=True).stdout.strip()
    commit = git('rev-parse', '--verify', release+'^{commit}')
    git('merge-base', '--is-ancestor', commit, 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=all', '--', 'src', 'specs',
                'tests/test_cnn_response_chb_1021.py', 'analysis/cnn_response_chb_1021')
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
            'batch': CN.BATCH, 'lr': .001, 'kappa': KAPPA, 'anchor_chunk': CN.BATCH, 'frozen_map_dtype': 'float32',
            'field_intermediate_dtype': 'float64', 'activity_dtype': 'float32',
            'primary_window': [1, 3000], 'secondary_windows': {'post_transient': [1001, 3000], 'all': [1, 30000], 'last': [27001, 30000]}}


def preflight_contract(data, graph, release_info):
    return {'contract_version': 1, 'registered_spec_commit': release_info['registered_spec_commit'],
            'implementation_commit': release_info['implementation_commit'],
            'identity': identity(data, 30000, graph)}


@torch.no_grad()
def preflight_anchor(B, chunk=CN.BATCH):
    before = tree_hash(B.snapshot())
    maximum, pool_equal, centered_equal, ce_max = [0.]*9, True, True, 0.
    values_equal, finite = True, True
    for lo in range(0, B.X.shape[1], chunk):
        ids = torch.arange(lo, min(lo+chunk, B.X.shape[1]), device=B.device)[None, :].expand(B.R, -1)
        xb = B.X[B.ar, ids]
        out, _, _, inputs = forward(B.P, xb, B.act, B.field, B.refs_ids(ids), with_inputs=True)
        natural, ni = CF.forward(B.P, xb, B.act, with_inputs=True)
        for k, (a, b) in enumerate(zip(out, natural)):
            finite &= bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            values_equal &= torch.equal(a, b)
            difference = float((a-b).abs().max())
            maximum[k] = max(maximum[k], difference) if math.isfinite(difference) else float('inf')
        centered_equal &= all(torch.equal(a, b) for a, b in zip(inputs, ni))
        for k in (1, 3):
            pool_equal &= torch.equal(F.max_pool2d(out[k], 2, 2, return_indices=True)[1],
                                      F.max_pool2d(natural[k], 2, 2, return_indices=True)[1])
        y = B.Y[B.ar, ids].reshape(-1)
        ca = F.cross_entropy(out[8].reshape(-1, CN.N_CLASSES), y)
        cb = F.cross_entropy(natural[8].reshape(-1, CN.N_CLASSES), y)
        finite &= bool(torch.isfinite(ca)) and bool(torch.isfinite(cb))
        difference = float((ca-cb).abs())
        ce_max = max(ce_max, difference) if math.isfinite(difference) else float('inf')
    if tree_hash(B.snapshot()) != before:
        raise RuntimeError('Anchor check mutated state')
    return {'feature_max_abs': maximum, 'pool_winners_equal': pool_equal,
            'centered_inputs_equal': centered_equal, 'ce_max_abs': ce_max,
            'values_equal': values_equal, 'all_finite': finite,
            'pass': finite and values_equal and max(maximum) == 0. and pool_equal and centered_equal and ce_max == 0.}


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


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
    updates = np.zeros((steps, B.R, 5, 5), dtype=np.float32)
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
                                          'grad_over_adam_denom_weight_rms', 'grad_weight_rms']}
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


def summarize(out):
    out = Path(out)
    tables, done = {}, {}
    for arm in MAIN_ARMS:
        done[arm] = json.loads((out/arm/'done.json').read_text())
        provenance = json.loads((out/arm/'provenance.json').read_text())
        if provenance.get('status') != 'COMPLETE' or provenance.get('identity') != done[arm]['identity']:
            raise RuntimeError(f'Incomplete or mismatched provenance: {arm}')
        if done[arm]['final_state_sha256'] != sha_file(out/arm/'state.pt'):
            raise RuntimeError(f'Final state artifact changed: {arm}')
        if done[arm]['identity']['steps'] != 30000 or done[arm]['identity']['data']['seeds'] != SEEDS:
            raise RuntimeError('Primary inference requires all 10 seeds and 30,000 updates')
        if done[arm]['summary_sha256'] != sha_file(out/arm/'summary.csv'):
            raise RuntimeError(f'Summary artifact changed: {arm}')
        if done[arm]['curves_sha256'] != sha_file(out/arm/'batch_curves.npz'):
            raise RuntimeError(f'Batch curve artifact changed: {arm}')
        with (out/arm/'summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(SEEDS) or sorted(int(r['seed']) for r in rows) != SEEDS:
            raise RuntimeError('Missing or duplicate preregistered seed')
        tables[arm] = {int(r['seed']): r for r in rows}
        with np.load(out/arm/'batch_curves.npz') as curves:
            if curves['accuracy'].shape != (10, 30000) or curves['seeds'].tolist() != SEEDS:
                raise RuntimeError('Malformed batch curves')
            if not np.isfinite(curves['accuracy']).all():
                raise RuntimeError('Failure policy requires accuracy zero, never NaN')
            for r, seed in enumerate(SEEDS):
                if float(tables[arm][seed]['first3000']) != float(curves['accuracy'][r, :3000].mean(dtype=np.float64)):
                    raise RuntimeError('Primary summary disagrees with saved batch curve')
    base = {k:v for k,v in done[MAIN_ARMS[0]]['identity'].items() if k != 'arm'}
    if any({k:v for k,v in done[a]['identity'].items() if k != 'arm'} != base for a in MAIN_ARMS):
        raise RuntimeError('Arm identities disagree')
    if len({done[a]['labels_sha256'] for a in MAIN_ARMS}) != 1 or len({done[a]['streams_sha256'] for a in MAIN_ARMS}) != 1:
        raise RuntimeError('Paired label/batch streams disagree')
    pairs = (('rescue', 'CH_swap', 'CH_self'), ('reverse', 'CHB_self', 'CHB_swap'))
    rng = np.random.default_rng(1021)
    boots = rng.integers(0, 10, size=(20000, 10))
    tests, pairrows = {}, []
    for label, plus, minus in pairs:
        differences = np.array([float(tables[plus][s]['first3000'])-float(tables[minus][s]['first3000']) for s in SEEDS])
        tests[label] = sign_test(differences) | {'mean_difference': float(differences.mean()),
            'ci95_seed_bootstrap': np.percentile(differences[boots].mean(1), [2.5,97.5]).tolist(),
            'seed_differences': dict(zip(map(str, SEEDS), map(float, differences)))}
        pairrows.extend({'comparison': label, 'seed': s, 'plus_arm': plus, 'minus_arm': minus,
                         'plus_first3000': tables[plus][s]['first3000'], 'minus_first3000': tables[minus][s]['first3000'],
                         'difference': float(d)} for s,d in zip(SEEDS, differences))
    previous = 0.
    for rank, key in enumerate(sorted(tests, key=lambda k: tests[k]['p_two_sided'])):
        test = tests[key]
        adjusted = min(1., max(previous, (2-rank)*test['p_two_sided']))
        previous = adjusted
        test.update(p_holm=adjusted, significant=adjusted <= .05)
        test.update(effect_direction(test['mean_difference'], test['n_positive'], test['n_negative']))
    supported = sum(t['significant'] and t['direction'] == 'predicted' for t in tests.values())
    opposite = any(t['significant'] and t['direction'] == 'opposite' for t in tests.values())
    heterogeneous = any(t['significant'] and t['direction'] == 'heterogeneous' for t in tests.values())
    verdict = ('HETEROGENEOUS' if heterogeneous else 'OPPOSITE' if opposite else
               'BIDIRECTIONAL' if supported == 2 else 'ONE_DIRECTION' if supported == 1 else 'NONE')
    secondary = {}
    for label, plus, minus in pairs:
        secondary[label] = {}
        for metric in ('steps1001_3000', 'all', 'last3000'):
            d = np.array([float(tables[plus][s][metric])-float(tables[minus][s][metric]) for s in SEEDS])
            secondary[label][metric] = {'mean_difference': float(d.mean()),
                                       'ci95_seed_bootstrap': np.percentile(d[boots].mean(1), [2.5,97.5]).tolist(),
                                       'seed_differences': dict(zip(map(str,SEEDS), map(float,d)))}
    result = {'primary': verdict, 'tests': tests, 'alpha': .05, 'family': 'two primary directions',
              'seeds': SEEDS, 'secondary': secondary,
              'interpretation_scope': 'activation derivative gates; not entire Jacobians, natural mediation, or long-term prevention'}
    save_json(out/'verdict.json', result)
    write_csv(out/'paired_primary.csv', pairrows)
    write_csv(out/'all_seed_summary.csv', [tables[a][s] for a in MAIN_ARMS for s in SEEDS])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'summarize'))
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
