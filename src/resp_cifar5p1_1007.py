#!/usr/bin/env python3
"""resp_cifar5p1_1007 -- S4's frozen response-field swap on the 5+1 CIFAR x MLP box (ReLU/std).

Registered in specs/spec_resp_cifar5p1_1007.md (commit 4ed6a3d5) before this file existed.

The host is `src/cifar5p1_mlp_0920.py` (unmodified, imported).  `Engine` re-expresses the
host's `run()` loop with explicit state -- the same stacked forward, the same element-wise
Adam step in the same order, the same CUDA-graph capture with a rolled-back warmup, the same
batch stream and the same global Adam counter `tc` -- so that branch points can be saved and
restored and a frozen layer-l response field can be swapped in:

    a_l = phi(z0_l) + [phi(z_l + d_l) - phi(z0_l + d_l)],   phi(z) = clamp(z, min=0)

z0 is recomputed from the frozen branch parameters P0 on the current minibatch with the same
ops; the bracket is exactly 0 at the branch, so the arm starts bit-identical to the natural
net.  d is a table over global CIFAR-100 train rows, NaN outside the continuation task's
images (a wrong lookup makes the loss non-finite).  The natural arms call the host forward.

    python -m src.resp_cifar5p1_1007 --out results/resp_cifar5p1_1007          # main run
    python -m src.resp_cifar5p1_1007 --check-mode --seeds 100-109 --out <dir>  # checks only
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src import cifar5p1_mlp_0920 as C
from src.cifar_interventions_0920 import cpu, tree_hash, save_pt, AnchorAdd
from analysis.cifar_ledger_0920.replay import sha, put, save_npz, environment, clean

B = C.B
H = C.H
RUN = 'resp_cifar5p1_1007'
REGISTRATION = '4ed6a3d5'
DATA = Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz')
RECORD = ROOT / 'results/cifar5p1_mlp_0920/R_std_lr0.0001'
LR, B1, B2, EPS = 1e-4, 0.9, 0.999, 1e-8
STEPS = C.STEPS_PER_TASK            # 780
BATCH = C.BATCH                     # 32
NCLS = C.N_CLASSES                  # 100
N_TRAIN = 50000
TH, TC = 2, 28                      # registered branch points (end of task)
PREFIX_TASKS = 29
SAVE_TASKS = (2, 3, 28, 29)
DIAG_STEPS = (78, 390, 780)
# name -> (branch task, field layer, donor task, Adam reset)
ARMS = {
    'N_h': (TH, None, None, False), 'N_c': (TC, None, None, False),
    'N_hr': (TH, None, None, True), 'N_cr': (TC, None, None, True),
    'R_ch': (TC, 2, TH, False), 'R_chr': (TC, 2, TH, True),
    'S_hcr': (TH, 2, TC, True), 'S_hc': (TH, 2, TC, False),
    'S_hcL1r': (TH, 1, TC, True)}
DISPLAY = {'N_h': 'N_h', 'N_c': 'N_c', 'N_hr': 'N_h r', 'N_cr': 'N_c r', 'R_ch': 'R_c←h',
           'R_chr': 'R_c←h r', 'S_hcr': 'S_h←c r', 'S_hc': 'S_h←c', 'S_hcL1r': 'S_h←c_L1 r'}
HOST_FLOAT64_COLS = ('eff_rank_l1', 'eff_rank_l2')   # spec §1.2: float64 GEMM/eigvalsh diagnostics
GPU_LOCK = Path('/tmp/lop_analysis_gpu.lock')


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def jnorm(x):
    """JSON-normal form (tuples -> lists, numpy -> python) so saved and live identities compare."""
    return json.loads(json.dumps(clean(x)))


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def phi(z):
    """The host's ReLU (`B.ReLU.phi`): torch.clamp(z, min=0.0)."""
    return torch.clamp(z, min=0.0)


def gtrain(z):
    """Training derivative of phi at z: native autograd of clamp (1 for z >= 0, 0 for z < 0)."""
    with torch.enable_grad():
        q = z.detach().requires_grad_(True)
        return torch.autograd.grad(phi(q).sum(), q)[0].detach()


def setup():
    dev = H.setup('cuda')
    torch.set_num_threads(2)
    assert torch.__version__ == '2.13.0+cu130', torch.__version__
    assert not torch.backends.cuda.matmul.allow_tf32, 'TF32 is outside the registered rounding model'
    assert torch.get_float32_matmul_precision() == 'highest'
    return dev


def write_csv(path, rows):
    """The host's CSV writer (`H.write_csv`): floats as %.10g, column order of first appearance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    H.write_csv(path, rows)


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return float('nan')


def gpu_free_gb():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    return float(p.stdout.split()[0]) / 1024


def other_gpu_python():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    busy = []
    for line in p.stdout.splitlines():
        if not line.strip().isdigit():
            continue
        pid = int(line)
        if pid == os.getpid():
            continue
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            continue
        if 'python' in cmd:
            busy.append((pid, cmd[:200]))
    return busy


@contextlib.contextmanager
def gpu_lock(wait_s=6 * 3600, poll=15, log=print):
    """Shared lab GPU lock (blocking), then wait until no other python process holds the GPU,
    the GPU has room for this run (measured peak 2.4 GB + context; 5 GB asked) and RAM keeps
    6 GB free beyond this run's ~3 GB peak.  One GPU process at a time (spec §8)."""
    with GPU_LOCK.open('a') as lock:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.time() - t0 < wait_s, 'GPU lock wait exceeded'
                time.sleep(poll)
        waited = 0
        while True:
            busy = other_gpu_python()
            free = mem_available_gb()
            gfree = gpu_free_gb()
            if not busy and free >= 10.0 and gfree >= 5.0:
                break
            assert time.time() - t0 < wait_s, ('GPU/RAM wait exceeded', busy, free, gfree)
            if waited % 8 == 0:
                log(f'waiting: other GPU python {busy}, MemAvailable {free:.1f} GB, GPU free {gfree:.1f} GB')
            waited += 1
            time.sleep(poll)
        yield


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Data:
    """CIFAR-100 train/test on the device exactly as the host builds them (cond std)."""

    def __init__(self, device, path=DATA):
        self.device = device
        self.cifar = C.Cifar100(Path(path))
        self.X = self.cifar.inputs('train', 'std', device)
        self.Y = self.cifar.train_y.to(device)
        self.Xte = self.cifar.inputs('test', 'std', device)
        self.Yte = self.cifar.test_y.to(device)
        self.tr_rows = C.class_rows(self.cifar.train_y)
        self.te_rows = C.class_rows(self.cifar.test_y)
        self.sha256 = self.cifar.sha256

    def task_rows(self, seeds, t):
        """Per slot, the task's global train rows in the host's order (CPU LongTensors)."""
        return [torch.cat([self.tr_rows[q] for q in C.task_plan(s)[t - 1][1]]) for s in seeds]

    def test_rows(self, seeds, t):
        return [torch.cat([self.te_rows[q] for q in C.task_plan(s)[t - 1][1]]) for s in seeds]


# --------------------------------------------------------------------------
# the frozen response field
# --------------------------------------------------------------------------

class Field:
    """Layer-l response field: a = phi(z0) + [phi(z + d) - phi(z0 + d)] with frozen P0 and d.

    `d` is (R, 50000, 100) over global CIFAR-100 train rows, NaN outside the continuation task.
    """

    def __init__(self, P0, layer, table):
        self.P0 = [p.detach().clone() for p in P0]
        self.layer = int(layer)
        self.d = table
        self.act = B.ReLU()

    def __call__(self, P, xb, ids):
        with torch.no_grad():
            z01, a01, z02, a02, _ = B.forward(self.P0, xb, self.act)
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        if self.layer == 1:
            a1 = AnchorAdd.apply(a01, phi(z1 + d) - phi(z01 + d))
        else:
            a1 = phi(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        if self.layer == 2:
            a2 = AnchorAdd.apply(a02, phi(z2 + d) - phi(z02 + d))
        else:
            a2 = phi(z2)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def eval_z(P, X, layer):
    """Layer-l preactivation of the natural net on X (R, N, 3072): one stacked eval forward."""
    with torch.no_grad():
        return B.forward(P, X, B.ReLU(), train=False)[2 * (layer - 1)]


def build_field(branch, donor, layer, rows, data):
    """d[r, id, i] = float32(float64(z_src) - float64(z_br)) on the task images, table over rows.

    Returns (table (R,50000,100) with NaN outside the task, ids (R,N), d_local (R,N,100))."""
    dev = data.device
    ids = torch.stack(rows).to(dev)
    X = data.X[ids]
    zb = eval_z([q.to(dev) for q in branch['P']], X, layer)
    zs = eval_z([q.to(dev) for q in donor['P']], X, layer)
    d = (zs.double() - zb.double()).float()
    R = ids.shape[0]
    table = torch.full((R, N_TRAIN, zb.shape[-1]), float('nan'), device=dev)
    table[torch.arange(R, device=dev)[:, None], ids] = d
    del X
    return table, ids, d


# --------------------------------------------------------------------------
# the engine (host semantics, explicit state)
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, seeds, data, state=None, field=None, graph=True):
        self.seeds = list(seeds)
        self.R = len(self.seeds)
        self.data = data
        self.device = data.device
        self.field = field
        self.graph_enabled = graph
        self.act = C.make_act('R')
        assert isinstance(self.act, B.ReLU)
        init = [q.detach() for s in self.seeds for q in C.init_params('R', s, self.device, C.HIDDEN)]
        self.P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
        self.P = [q.requires_grad_(True) for q in self.P]
        self.P_init = [q.detach().clone() for q in self.P]
        self.m = [torch.zeros_like(q) for q in self.P]
        self.v = [torch.zeros_like(q) for q in self.P]
        self.g_batch = {s: H.stream('c51_batch', s) for s in self.seeds}
        self.tc = 0
        self.task = 0
        dev = self.device
        self.static_idx = torch.zeros(self.R, BATCH, dtype=torch.long, device=dev)
        self.inv_c1 = torch.zeros((), device=dev)
        self.inv_c2 = torch.zeros((), device=dev)
        self.step_t = torch.zeros((), dtype=torch.long, device=dev)
        self.acc_sum = torch.zeros(self.R, device=dev)
        self.ce_sum = torch.zeros(self.R, device=dev)
        self.bad_step = torch.full((self.R,), -1, dtype=torch.long, device=dev)
        self.last_hit = torch.zeros(self.R, device=dev)
        self.ar = torch.arange(self.R, device=dev)[:, None]
        self.cg = None
        if state is not None:
            self.restore(state)
        self.capture()

    # -- forward / one step (the host's `step`, plus a CE accumulator) --
    def forward(self, xb, ids):
        if self.field is None:
            return B.forward(self.P, xb, self.act, train=True)
        return self.field(self.P, xb, ids)

    def step(self):
        xb, yb = self.data.X[self.static_idx], self.data.Y[self.static_idx]
        z1, a1, z2, a2, z3 = self.forward(xb, self.static_idx)
        lossv = F.cross_entropy(z3.reshape(-1, NCLS), yb.reshape(-1),
                                reduction='none').view(self.R, BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        self.acc_sum.add_(hit)
        self.last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.ce_sum.add_(lossv.detach())
            bad = ~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & bad, self.step_t, self.bad_step))
            self.step_t.add_(1)
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(B1).add_(gr, alpha=1 - B1)
                vi.mul_(B2).addcmul_(gr, gr, value=1 - B2)
                p.sub_(LR * (mi * self.inv_c1) / ((vi * self.inv_c2).sqrt() + EPS))

    def mutable(self):
        return [*self.P, *self.m, *self.v, self.acc_sum, self.ce_sum, self.bad_step,
                self.step_t, self.last_hit]

    def capture(self):
        """The host's capture: 3 warmup steps on a side stream, capture one, roll back."""
        if not self.graph_enabled:
            return
        assert self.device.type == 'cuda'
        keep = [q.detach().clone() for q in self.mutable()]
        self.inv_c1.fill_(1.0)
        self.inv_c2.fill_(1.0)
        self.static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cg):
            self.step()
        with torch.no_grad():
            for q, old in zip(self.mutable(), keep):
                q.copy_(old)
        del keep

    # -- state --
    def core(self):
        return dict(seeds=self.seeds, P=self.P, m=self.m, v=self.v, tc=self.tc, task=self.task,
                    g_batch={s: g.get_state() for s, g in self.g_batch.items()})

    def state(self):
        return cpu(self.core())

    def restore(self, st):
        assert list(st['seeds']) == self.seeds, 'slot mismatch'
        with torch.no_grad():
            for name in ('P', 'm', 'v'):
                for dst, src in zip(getattr(self, name), st[name]):
                    dst.copy_(src)
        for s, value in st['g_batch'].items():
            self.g_batch[s].set_state(value)
        self.tc = int(st['tc'])
        self.task = int(st['task'])

    def reset_adam(self):
        with torch.no_grad():
            for q in (*self.m, *self.v):
                q.zero_()
        self.tc = 0

    # -- one task --
    def task_batches(self, t):
        """(rows per slot, (R, 780, 32) global rows): the host's `batch_indices` on this
        engine's own generators, slot order."""
        rows = self.data.task_rows(self.seeds, t)
        batches = torch.stack([C.batch_indices(self.g_batch[s], rows[r], STEPS)
                               for r, s in enumerate(self.seeds)]).to(self.device)
        return rows, batches

    def train_steps(self, batches, diag_at=(), on_diag=None):
        """The host's `train_task`: zero the accumulators, then one Adam step per batch."""
        self.acc_sum.zero_()
        self.ce_sum.zero_()
        self.bad_step.fill_(-1)
        self.step_t.zero_()
        for j in range(batches.shape[1]):
            self.static_idx.copy_(batches[:, j])
            self.tc += 1
            self.inv_c1.fill_(1.0 / (1 - B1 ** self.tc))
            self.inv_c2.fill_(1.0 / (1 - B2 ** self.tc))
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
            if on_diag is not None and (j + 1) in diag_at:
                on_diag(j + 1)
        torch.cuda.synchronize(self.device)

    def finite(self):
        with torch.no_grad():
            ok = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in self.P]).all(0)
            return (ok & (self.bad_step < 0)).cpu()

    def host_rows(self, t, rows, steps=STEPS):
        """The host's per-task row for every slot (`evaluate` + `accuracy`, natural forward)."""
        hard = t % 2 == 1
        R = self.R
        with torch.no_grad():
            Xt = torch.stack([self.data.X[rows[r]] for r in range(R)])
            Yt = torch.stack([self.data.Y[rows[r]] for r in range(R)])
            m = C.evaluate(self.P, Xt, Yt, self.act, list(range(R)))
            del Xt
            te = self.data.test_rows(self.seeds, t)
            test_acc = C.accuracy(self.P, torch.stack([self.data.Xte[q] for q in te]),
                                  torch.stack([self.data.Yte[q] for q in te]), self.act)
        out = []
        for r, s in enumerate(self.seeds):
            out.append({'arm': 'R', 'cond': 'std', 'seed': s, 'slot': r, 'lr': LR, 'task': t,
                        'hard': int(hard), 'n_classes': len(C.task_plan(s)[t - 1][1]),
                        'online_acc': float(self.acc_sum[r]) / steps,
                        'train_acc': m[r]['acc'], 'test_acc': float(test_acc[r]), **m[r]})
        return out

    def fresh(self, batches):
        """The host's fresh control: P <- init, m/v <- 0, tc <- 0, the same task batches."""
        with torch.no_grad():
            for q, v in zip(self.P, self.P_init):
                q.copy_(v)
            for q in (*self.m, *self.v):
                q.zero_()
        self.tc = 0
        self.train_steps(batches)
        return [float(self.acc_sum[r]) / STEPS for r in range(self.R)], self.finite()

    # -- diagnostics on all images of the continuation task --
    def diagnostic(self, ids, Y, update):
        """ids (R, N) global rows on the device, Y (R, N).  No RNG, no state change."""
        R, N = ids.shape
        with torch.no_grad():
            X = self.data.X[ids]
            z1, a1, z2, a2, logits = self.forward(X, ids)
            del X
        rows = [dict(seed=s, update=update) for s in self.seeds]
        unit = {}
        for l, z in ((1, z1), (2, z2)):
            arg = z
            if self.field is not None and self.field.layer == l:
                arg = z + self.field.d[self.ar, ids]
            g = gtrain(arg)
            gd = g.double()
            zz = z.double()
            sums = gd.sum(1)
            sq = gd.square().sum(1)
            neff = torch.where(sq > 0, sums.square() / torch.where(sq > 0, sq, torch.ones_like(sq)),
                               torch.zeros_like(sq))
            W = self.P[2 * l - 2].detach().double()
            bvec = self.P[2 * l - 1].detach().double()
            mu = zz.mean(1)
            sd = zz.std(1, unbiased=False)
            qmask = g.abs() < 1e-6
            zmask = g == 0
            mism = g != (arg > 0).to(g.dtype)
            for name, value in dict(zmean=mu, zsd=sd, gmean=gd.mean(1), q=qmask.double().mean(1),
                                    zero=zmask.double().mean(1), neff=neff, neff_fraction=neff / N,
                                    Wnorm=W.norm(dim=2), bias=bvec, argmean=arg.double().mean(1),
                                    zmean_over_sd=torch.where(sd > 0, mu / torch.where(sd > 0, sd, torch.ones_like(sd)),
                                                              torch.full_like(sd, float('nan')))).items():
                unit[f'{name}_l{l}'] = value.cpu().numpy()
            for r, row in enumerate(rows):
                qc = int(qmask[r].sum())
                zc = int(zmask[r].sum())
                row.update({f'G{l}': float(gd[r].mean()), f'Q{l}': qc / (N * z.shape[-1]),
                            f'Q_count{l}': qc, f'zero_count{l}': zc, f'pair_count{l}': int(g[r].numel()),
                            f'zero{l}': zc / (N * z.shape[-1]),
                            f'dead_unit{l}': float(zmask[r].all(0).double().mean()),
                            f'diag_mismatch_count{l}': int(mism[r].sum()),
                            f'neff{l}': float(neff[r].mean()), f'neff_fraction{l}': float(neff[r].mean() / N),
                            f'neff_zero_units{l}': int((sq[r] == 0).sum()),
                            f'zmean{l}': float(mu[r].mean()), f'zsd{l}': float(sd[r].mean()),
                            f'argmean{l}': float(arg[r].double().mean()),
                            f'Wnorm{l}': float(W[r].norm(dim=1).mean()), f'bias{l}': float(bvec[r].mean())})
        mu2 = a1.double().mean(1)
        W2 = self.P[2].detach().double()
        wn = W2.norm(dim=2)
        mn = mu2.norm(dim=1)
        unit['mu2'] = mu2.cpu().numpy()
        unit['cos_W2_mu2'] = ((W2 * mu2[:, None, :]).sum(2) / (wn * mn[:, None])).cpu().numpy()
        W3 = self.P[4].detach().double()
        for r, row in enumerate(rows):
            row.update(mu2_norm=float(mn[r]), Wnorm3=float(W3[r].norm(dim=1).mean()),
                       bias3=float(self.P[5][r].detach().double().mean()),
                       memo_acc=float((logits[r].argmax(-1) == Y[r]).float().mean()),
                       memo_ce=float(F.cross_entropy(logits[r], Y[r])))
        return rows, unit


def core_state(st):
    return {k: st[k] for k in ('P', 'm', 'v', 'tc', 'task', 'g_batch')}


# --------------------------------------------------------------------------
# natural prefix
# --------------------------------------------------------------------------

def natural_prefix(eng, n_tasks, save=None, rows=None, stop=None, on_task=None):
    """Tasks eng.task+1 .. n_tasks with the host's per-task rows.  `save(t, state, batches)`
    is called at every task end.  Returns (rows, last batches) or None on STOP."""
    rows = [] if rows is None else rows
    batches = None
    for t in range(eng.task + 1, n_tasks + 1):
        if stop is not None and Path(stop).exists():
            return None
        task_rows, batches = eng.task_batches(t)
        eng.train_steps(batches)
        eng.task = t
        assert bool(eng.finite().all()), ('DIVERGED prefix', t)
        rows.extend(eng.host_rows(t, task_rows))
        rows.sort(key=lambda q: (q['slot'], q['task']))          # the host's row order
        if save is not None:
            save(t, eng.state(), batches.cpu(), rows)
        if on_task is not None:
            on_task(t)
    return rows, batches


def fresh_rows(seeds, fresh_values, rows, task=29):
    continual = {r['seed']: r['online_acc'] for r in rows if r['task'] == task}
    return [{'arm': 'R', 'cond': 'std', 'seed': s, 'lr': LR, 'task': task,
             'fresh_online_acc': fresh_values[i], 'continual_online_acc': continual[s],
             'fresh_gap': fresh_values[i] - continual[s]} for i, s in enumerate(seeds)]


def csv_text(rows):
    """The host CSV text (header + lines) of rows, for byte comparison."""
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'x.csv'
        H.write_csv(p, rows)
        return p.read_text()


def compare_record(prefix_rows, fresh, record=RECORD, tasks=range(1, PREFIX_TASKS + 1)):
    """S-host(b): prefix rows vs the committed 0920 record, as the host formats them (%.10g).
    All columns byte-identical except the float64 eff_rank diagnostics (reported)."""
    import csv
    rec = {(int(r['seed']), int(r['task'])): r for r in csv.DictReader((record / 'per_task.csv').open())}
    recf = {int(r['seed']): r for r in csv.DictReader((record / 'fresh_control.csv').open())}

    def fmt(v):
        return '' if v is None else (f'{v:.10g}' if isinstance(v, float) else str(v))
    out = []
    ok = True
    seen = set()
    for row in prefix_rows:
        key = (row['seed'], row['task'])
        if row['task'] not in tasks:
            continue
        seen.add(key)
        ref = rec[key]
        assert set(ref) == set(row), ('column set', key)
        for col in ref:
            mine = fmt(row[col])
            same = mine == ref[col]
            if not same and col not in HOST_FLOAT64_COLS:
                ok = False
            if not same or col in HOST_FLOAT64_COLS:
                rel = abs(float(mine) - float(ref[col])) / max(abs(float(ref[col])), 1e-300) if not same else 0.0
                out.append(dict(seed=key[0], task=key[1], column=col, run=mine, record=ref[col],
                                byte_equal=same, gated=col not in HOST_FLOAT64_COLS, relative_difference=rel))
    assert seen == {(s, t) for s in {r['seed'] for r in prefix_rows} for t in tasks}, 'record coverage'
    for row in fresh:
        ref = recf[row['seed']]
        for col in ('fresh_online_acc', 'continual_online_acc', 'fresh_gap'):
            mine = fmt(row[col])
            same = mine == ref[col]
            ok &= same
            out.append(dict(seed=row['seed'], task=29, column=col, run=mine, record=ref[col], byte_equal=same,
                            gated=True, relative_difference=0.0 if same else abs(float(mine) - float(ref[col]))))
    return ok, out


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def arm_engine(name, states, data, graph=True):
    """Fresh, independent engine for one arm from the saved branch state (and donor state)."""
    branch, layer, donor, reset = ARMS[name]
    st = states[branch]
    field = None
    if layer is not None:
        rows = data.task_rows(st['seeds'], branch + 1)
        table, _, _ = build_field(st, states[donor], layer, rows, data)
        field = Field([p.to(data.device) for p in st['P']], layer, table)
    eng = Engine(st['seeds'], data, state=st, field=field, graph=graph)
    if reset:
        eng.reset_adam()
    return eng


def preflight(name, eng, states, batches):
    """S-branch + S-field on the arm as built (independent references, no residual tolerances).

    Batches: all task images at once (B=N), every one of the 780 training batches, and the
    task images in row order cut into chunks of 32."""
    branch, layer, donor, reset = ARMS[name]
    dev = eng.device
    data = eng.data
    T = branch + 1
    bp = [q.to(dev) for q in states[branch]['P']]
    rows = data.task_rows(eng.seeds, T)
    ids_all = torch.stack(rows).to(dev)
    N = ids_all.shape[1]
    ar = eng.ar
    u32, u64 = 2.0 ** -24, 2.0 ** -53
    eta = float(torch.finfo(torch.float32).tiny)
    report = dict(arm=name, batches=0, branch_bit_exact=True)
    if layer is not None:
        assert eng.field.layer == layer, 'field layer'
        assert tree_hash(eng.field.P0) == tree_hash(bp), 'P0 is not the branch state'
        assert all(p.data_ptr() != q.data_ptr() for p, q in zip(eng.P, eng.field.P0)), 'P0 aliased'
        # Independent rebuild of the field from the saved states; must be bit-identical.
        X = data.X[ids_all]
        zb_eval = eval_z(bp, X, layer)
        zs_eval = eval_z([q.to(dev) for q in states[donor]['P']], X, layer)
        del X
        d_ref = (zs_eval.double() - zb_eval.double()).float()
        assert tree_hash(eng.field.d[ar, ids_all]) == tree_hash(d_ref), 'field values'
        outside = torch.ones(eng.R, N_TRAIN, dtype=torch.bool, device=dev)
        outside[ar, ids_all] = False
        assert bool(torch.isnan(eng.field.d[outside]).all()), 'field defined outside the task'
        del outside
        sd = d_ref.double().std(dim=(1, 2))
        assert bool((sd > 0).all()), 'constant field'
        q_all = zb_eval.double() + d_ref.double() - zs_eval.double()
        pos = torch.full((eng.R, N_TRAIN), -1, dtype=torch.long, device=dev)
        pos[ar, ids_all] = torch.arange(N, device=dev)[None, :].expand(eng.R, -1)
        report.update(field_sd_per_seed=sd.cpu().tolist(), max_abs_q=float(q_all.abs().max()))
        arg_ratio = torch.zeros(eng.R, dtype=torch.float64, device=dev)
        ambiguous = torch.zeros(eng.R, dtype=torch.long, device=dev)
        gate_pairs = torch.zeros(eng.R, dtype=torch.long, device=dev)
        count_slack_max = 0
    gens = [ids_all]
    gens += [batches[:, j] for j in range(batches.shape[1])]
    gens += [ids_all[:, j:j + BATCH] for j in range(0, N, BATCH)]
    for ids in gens:
        xb = data.X[ids]
        with torch.no_grad():
            natural = B.forward(bp, xb, B.ReLU(), train=True)
            actual = eng.forward(xb, ids)
        for a, b in zip(actual, natural):
            assert tree_hash(a) == tree_hash(b), 'branch output'
        y = data.Y[ids]
        ce_a = F.cross_entropy(actual[-1].flatten(0, 1), y.flatten(), reduction='none')
        ce_n = F.cross_entropy(natural[-1].flatten(0, 1), y.flatten(), reduction='none')
        assert tree_hash(ce_a) == tree_hash(ce_n), 'branch CE'
        if layer is not None:
            loc = pos[ar, ids]
            assert bool((loc >= 0).all()), 'batch row outside the task'
            zb_mb = natural[2 * (layer - 1)]
            d = d_ref[ar, loc]
            arg = zb_mb + d
            target = zs_eval[ar, loc].double()
            zbe = zb_eval[ar, loc].double()
            bound = (q_all[ar, loc].abs() + (zb_mb.double() - zbe).abs()
                     + u32 * (zb_mb.double().abs() + d.double().abs()) + eta
                     + 3 * u64 * (zbe.abs() + d.double().abs() + target.abs()))
            err = (arg.double() - target).abs()
            assert bool((err <= bound).all()), 'field argument'
            arg_ratio = torch.maximum(arg_ratio, (err / bound).flatten(1).max(1).values)
            sure = target.abs() > bound
            g_arg = gtrain(arg)
            g_tgt = gtrain(target.float())
            assert bool((g_arg[sure] == g_tgt[sure]).all()), 'field gate'
            ambiguous += (~sure).flatten(1).sum(1)
            gate_pairs += sure.flatten(1).sum(1)
            # The arm's own training gate, read through autograd of its forward: the derivative
            # of sum(a_l) w.r.t. b_l is, per unit, the number of images whose gate is open.
            with torch.enable_grad():
                vals = eng.forward(xb, ids)
                counts = torch.autograd.grad(vals[2 * layer - 1].sum(), eng.P[2 * layer - 1])[0]
            expect = g_tgt.sum(1)
            slack = (~sure).sum(1)
            diff = (counts.double() - expect.double()).abs()
            assert bool((diff <= slack.double()).all()), 'training gate'
            count_slack_max = max(count_slack_max, int(diff.max()))
        report['batches'] += 1
    if layer is not None:
        report.update(argument_error_ratio_per_seed=arg_ratio.cpu().tolist(),
                      argument_error_ratio=float(arg_ratio.max()),
                      ambiguous_pairs_per_seed=ambiguous.cpu().tolist(),
                      gate_checked_pairs_per_seed=gate_pairs.cpu().tolist(),
                      training_gate_count_max_difference=count_slack_max)
    report['all_pass'] = True
    return report


def run_arm(name, states, data, graph=True, preflight_on=True):
    """Build, check and train one arm for one task.  Returns a dict of results (no files)."""
    branch, layer, donor, reset = ARMS[name]
    T = branch + 1
    eng = arm_engine(name, states, data, graph=graph)
    rows_T, batches = eng.task_batches(T)
    ids = torch.stack(rows_T).to(eng.device)
    Y = data.Y[ids]
    rng_before = tree_hash({s: g.get_state() for s, g in eng.g_batch.items()})
    pf = preflight(name, eng, states, batches) if preflight_on else None
    history = []
    r0, u0 = eng.diagnostic(ids, Y, 0)
    history.append((0, r0, u0))
    assert rng_before == tree_hash({s: g.get_state() for s, g in eng.g_batch.items()}), 'diagnostic consumed RNG'

    def on_diag(update):
        rr, uu = eng.diagnostic(ids, Y, update)
        history.append((update, rr, uu))
    eng.train_steps(batches, DIAG_STEPS, on_diag)
    eng.task = T
    finite = eng.finite()
    end = []
    final = {r['seed']: r for r in history[-1][1]}
    label_hash = tree_hash(Y)
    batch_hash = tree_hash(batches)
    host = eng.host_rows(T, rows_T) if layer is None else None
    for r, s in enumerate(eng.seeds):
        row = dict(arm=name, display=DISPLAY[name], seed=s, task=T, branch=branch, layer=layer or 0,
                   donor=donor or 0, reset=int(reset), online_acc=float(eng.acc_sum[r]) / STEPS,
                   online_ce=float(eng.ce_sum[r]) / STEPS, memo_acc=final[s]['memo_acc'],
                   memo_ce=final[s]['memo_ce'], finite=bool(finite[r]), label_hash=label_hash,
                   batch_hash=batch_hash)
        if host is not None:
            row.update({f'host_{k}': v for k, v in host[r].items() if k not in ('arm', 'cond', 'seed', 'slot', 'lr', 'task')})
        end.append(row)
    result = dict(rows=end, history=history, preflight=pf, end_state=eng.state(), host_rows=host,
                  batches=batches.cpu(),
                  field=None if eng.field is None else dict(layer=eng.field.layer, ids=ids.cpu(),
                                                            d=eng.field.d[eng.ar, ids].cpu()))
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def source_hashes():
    files = [ROOT / f'src/{RUN}.py', ROOT / f'specs/spec_{RUN}.md', ROOT / 'src/cifar5p1_mlp_0920.py',
             ROOT / 'src/rlcifar_mlp_battle_0918.py', ROOT / 'src/pmnist_0905.py',
             ROOT / 'src/pmnist_rlcifar_0907.py', ROOT / 'src/cifar_interventions_0920.py',
             ROOT / 'analysis/cifar_ledger_0920/replay.py', ROOT / 'analysis/resp_cifar_ee_0920/stats.py']
    files += sorted((ROOT / f'analysis/{RUN}').glob('*.py'))
    files += sorted((ROOT / f'analysis/{RUN}').glob('*.sh'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def identity(seeds, data, steps=STEPS, check_mode=False):
    return jnorm(dict(run_id=RUN, seeds=list(seeds), steps_per_task=steps, prefix_tasks=PREFIX_TASKS,
                      branches=dict(t_h=TH, t_c=TC), arms=list(ARMS), registration_commit=REGISTRATION,
                      check_mode=bool(check_mode),
                      git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      source_sha256=source_hashes(), environment=environment(), data_sha256=data.sha256,
                      input_hash=tree_hash(data.X), labels_hash=tree_hash(data.Y),
                      class_plans={str(s): C.seed_classes(s) for s in seeds}))


def require_identity(saved, current):
    assert jnorm(saved) == jnorm(current), 'resume identity mismatch'


def mark_done(path, ident, files):
    put(path, dict(identity=ident, files={str(Path(p).name): sha(p) for p in files}, complete=True))


def verify_done(path, ident):
    obj = json.loads(Path(path).read_text())
    require_identity(obj['identity'], ident)
    assert obj['complete'] and obj['files'], 'incomplete marker'
    for name, h in obj['files'].items():
        p = Path(path).parent / name
        assert p.exists() and sha(p) == h, ('corrupt output', name)
    return obj


def serialize_history(arm_dir, history):
    rows, arrays = [], {}
    for update, rr, uu in history:
        rows.extend(rr)
        for key, value in uu.items():
            arrays[f'u{update:04d}_{key}'] = value
    put(arm_dir / 'diagnostics.json', rows)
    save_npz(arm_dir / 'diagnostic_units.npz', arrays)


# --------------------------------------------------------------------------
# the registered run
# --------------------------------------------------------------------------

def run(out, seeds, steps=STEPS, check_mode=False, log=print, stop_hook=None, data=None):
    """Prefix (29 tasks + fresh) -> S-host(b) -> N_h, N_c -> S-prefix -> other arms.
    Returns True when complete, False on STOP.  `stop_hook(stage, item)` is a check-only hook."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert steps == STEPS, 'registered budget is 780 updates per task'
    dev = setup()
    data = Data(dev) if data is None else data          # checks may pass the loaded data
    ident = identity(seeds, data, steps, check_mode)
    if not check_mode:
        assert seeds == list(range(10)), 'registered seeds are 0-9'
        chk = json.loads((ROOT / f'results/_checks_{RUN}/checks.json').read_text())
        assert chk['all_pass'] and chk['source_sha256'] == source_hashes(), 'unverified code'
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'analysis', 'specs'],
                                        cwd=ROOT, text=True)
        assert not dirty, 'commit the implementation before the main run'
        saved_checks = out / 'admission_checks.json'
        if saved_checks.exists():
            assert json.loads(saved_checks.read_text()) == chk
        else:
            put(saved_checks, chk)
    put(out / 'input_manifest.json', dict(data_sha256=ident['data_sha256'], input_hash=ident['input_hash'],
                                          labels_hash=ident['labels_hash'], class_plans=ident['class_plans']))
    start = out / 'provenance_start.json'
    if start.exists():
        require_identity(json.loads(start.read_text())['identity'], ident)
    else:
        import datetime
        put(start, dict(identity=ident, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        started_jst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                        pid=os.getpid(), argv=sys.argv, independent_audit=False,
                        mem_available_gb=mem_available_gb(),
                        dirty_diff=subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'analysis', 'specs'],
                                                           cwd=ROOT, text=True)))
    stop = out / 'STOP'
    t0 = time.time()
    prefix = out / 'prefix'
    prefix.mkdir(exist_ok=True)
    marker = prefix / 'done.json'
    if marker.exists():
        verify_done(marker, ident)
    else:
        active = prefix / 'active.pt'
        if active.exists():
            saved = load_pt(active)
            require_identity(saved['identity'], ident)
            eng = Engine(seeds, data, state=saved['state'])
            rows = saved['rows']
        else:
            eng = Engine(seeds, data)
            rows = []

        def save(t, st, batches, rows_now):
            if t in SAVE_TASKS:
                save_pt(prefix / f't{t:02d}.pt', dict(**st, last_batches=batches))
            save_pt(active, dict(identity=ident, state=st, rows=rows_now))
            put(out / 'status.json', dict(stage='prefix', completed_task=t, wall_s=time.time() - t0))

        def on_task(t):
            if stop_hook is not None:
                stop_hook('prefix', t)
        res = natural_prefix(eng, PREFIX_TASKS, save=save, rows=rows, stop=stop, on_task=on_task)
        if res is None:
            return False
        rows, _ = res
        last = load_pt(prefix / f't{PREFIX_TASKS:02d}.pt')['last_batches'].to(dev)
        fv, ff = eng.fresh(last)
        assert bool(ff.all()), 'DIVERGED fresh control'
        fresh = fresh_rows(seeds, fv, rows, PREFIX_TASKS)
        write_csv(prefix / 'per_task.csv', rows)
        write_csv(prefix / 'fresh_control.csv', fresh)
        put(prefix / 'rows.json', rows)
        put(prefix / 'fresh.json', fresh)
        if not check_mode:
            ok, comparison = compare_record(rows, fresh)
            write_csv(prefix / 'host_record_comparison.csv', comparison)
            put(prefix / 'host_record_check.json', dict(all_pass=ok, gated_mismatches=sum(1 for c in comparison if c['gated'] and not c['byte_equal']),
                                                        eff_rank_mismatches=sum(1 for c in comparison if not c['gated'] and not c['byte_equal']),
                                                        eff_rank_max_relative_difference=max([c['relative_difference'] for c in comparison if not c['gated']] or [0.0])))
            if not ok:
                put(out / 'status.json', dict(stage='CHECK_FAILED', reason='S-host(b) record mismatch'))
                raise SystemExit('CHECK_FAILED: S-host(b) prefix does not reproduce the 0920 record')
        del eng
        gc.collect()
        torch.cuda.empty_cache()
        files = sorted(p for p in prefix.iterdir() if p.name not in ('done.json', 'active.pt') and not p.name.endswith('.tmp'))
        mark_done(marker, ident, files)
        active.unlink(missing_ok=True)
        log(f'prefix complete: {time.time() - t0:.1f}s')
    states = {t: load_pt(prefix / f't{t:02d}.pt') for t in SAVE_TASKS}
    prefix_rows = json.loads((prefix / 'rows.json').read_text())
    integrity = {}
    for name in ARMS:
        if stop.exists():
            return False
        arm = out / 'arms' / name
        arm.mkdir(parents=True, exist_ok=True)
        done = arm / 'done.json'
        if done.exists():
            verify_done(done, ident)
            continue
        res = run_arm(name, states, data)
        put(arm / 'preflight.json', res['preflight'])
        put(arm / 'rows.json', res['rows'])
        write_csv(arm / 'per_task.csv', res['rows'])
        serialize_history(arm, res['history'])
        save_pt(arm / 'end.pt', res['end_state'])
        if res['field'] is not None:
            save_pt(arm / 'field.pt', res['field'])
        assert all(r['finite'] for r in res['rows']), ('DIVERGED arm', name)
        if name in ('N_h', 'N_c'):
            check = prefix_check(name, res, states, prefix_rows)
            put(arm / 'prefix_check.json', check)
            integrity[name] = check
        files = sorted(p for p in arm.iterdir() if p.name != 'done.json' and not p.name.endswith('.tmp'))
        mark_done(done, ident, files)
        put(out / 'status.json', dict(stage='arms', completed_arm=name, wall_s=time.time() - t0))
        log(f'arm {name} complete: {time.time() - t0:.1f}s')
        if stop_hook is not None:
            stop_hook('arm', name)
    # Rebuild the S-prefix evidence from disk (also after a resume).
    for name in ('N_h', 'N_c'):
        integrity[name] = json.loads((out / 'arms' / name / 'prefix_check.json').read_text())
        assert integrity[name]['all_pass'], ('S-prefix', name)
    put(out / 'prefix_checks.json', integrity)
    put(out / 'status.json', dict(stage='completed', arms=len(ARMS), seeds=len(seeds), wall_s=time.time() - t0))
    put(out / 'provenance_end.json', dict(identity=ident, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                          elapsed_this_invocation_s=time.time() - t0,
                                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                          peak_cuda_bytes=torch.cuda.max_memory_allocated(), independent_audit=False))
    return True


def prefix_check(name, res, states, prefix_rows):
    """S-prefix: the natural arm resumed from the branch equals the uninterrupted prefix."""
    branch = ARMS[name][0]
    T = branch + 1
    expected = states[T]
    same_state = tree_hash(core_state(res['end_state'])) == tree_hash(core_state(expected))
    same_batches = tree_hash(res['batches']) == tree_hash(expected['last_batches'])
    lookup = {r['seed']: r for r in prefix_rows if r['task'] == T}
    rows_equal = True
    for host in res['host_rows']:
        ref = lookup[host['seed']]
        for k, v in host.items():
            if ref[k] != v:
                rows_equal = False
    assert same_state, ('natural continuation state mismatch', name)
    assert same_batches, ('natural continuation batches mismatch', name)
    assert rows_equal, ('natural continuation rows mismatch', name)
    return dict(all_pass=True, core_bit_exact=True, batches_bit_exact=True, rows_exact=True, task=T)


def parse_seeds(s):
    return B.parse_ints(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'results' / RUN))
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--steps', type=int, default=STEPS)
    p.add_argument('--check-mode', action='store_true')
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    if a.check_mode:
        assert Path(a.out).resolve() != (ROOT / 'results' / RUN).resolve(), 'check mode must not write the main output'
    with gpu_lock():
        ok = run(Path(a.out), seeds, a.steps, a.check_mode)
    print('completed' if ok else 'STOP acknowledged; boundary state saved.', flush=True)


if __name__ == '__main__':
    main()
