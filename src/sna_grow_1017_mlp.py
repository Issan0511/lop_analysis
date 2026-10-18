#!/usr/bin/env python3
"""sna_grow_1017 (MLP part) -- adaptive Snake that follows only the growth of the unit width.

Registered in specs/spec_sna_grow_1017.md (commit 2bb76f32) before this file existed:
§2 (the rule `grow`), §2.2 (the MLP arm), §3 (K-mlp-rep, K-mlp-grow, K-mlp-graph).

Rule `grow` (spec §2).  Task 1 is the host SNA: alpha = clip(c / sqrt(V), lo, hi), V the EMA
(beta 0.01) of the batch variance of z, c 0.6, lo 0.005, hi 3.0.  At the end of task 1, on both
hidden layers, Vmax <- V (per run and unit) and the run's grow flag is raised.  From then on,
right after every EMA update of V, Vmax <- max(Vmax, V) in place, and alpha = clip(c / sqrt(Vmax),
lo, hi).  The EMA of V itself keeps running as in the host.  Everything else (c, beta, clip, init,
lr, data, label sequence, batch sequence) is the host's.

Engine.  `src/controls_cifar5p1_1008.py` is imported unmodified: its `SnakeEngine` (the host
`cifar5p1_mlp_0920.run()` re-expressed with explicit state: the step, the 3-step warmup + capture
rolled back incl. the activation state, the batch stream, the per-task host rows, the fresh
control).  This file adds, without editing that module:

  * `GrowSnake`, a subclass of the host `SnakeFamily` (`rlcifar_mlp_battle_0918`) whose Vmax and
    grow flags are static tensors (a captured CUDA graph reads them; `load_state` copies in place)
    and are part of `state()` / `load_state()`.  With the flags down, alpha is the host's
    expression on the very same V (`torch.where` only selects), so it is bit-identical to SNA.
  * the arm `SNA_grow` = ('SNA', 'grow'), registered in the controls module's `C2_ARMS`;
  * `GrowEngine(SnakeEngine)` with `start_grow()`: Vmax <- V, flags up, recapture (as `freeze()`
    recaptures for fixT1);
  * `run_arm`, the controls `c2_run_arm` loop plus the switch after task 1.

Per-task rows are the controls' C2 rows (same columns, same %.10g CSV).  Two host statistics are
read on the grow arm as follows: `two_alpha_W_med` is 2 * alpha * sqrt(V) with V the running EMA
(the actual width, as in the host); `alpha_clip_frac` is computed from the variance that drives
alpha (Vmax once grow is on, V before), i.e. it is the fraction of units whose alpha is clipped.
Vmax and the flags after every task go to V_history.npz (`vmax_tXX_lY`, `grow_tXX_lY`).

The fresh control is the host's: P back to init, m/v/tc to 0, the activation state back to its
initial value (V = 1, flags down), i.e. a fresh net that adapts as SNA does within its first task.

    python -m src.sna_grow_1017_mlp checks      # -> results/sna_grow_1017/mlp_checks.json
    python -m src.sna_grow_1017_mlp run         # -> results/sna_grow_1017/mlp/ (seeds 0-9)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
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

from src import controls_cifar5p1_1008 as K        # verified engine (unmodified)
from src.cifar_interventions_0920 import cpu, tree_hash, save_pt
from analysis.cifar_ledger_0920.replay import sha, put, save_npz, environment

RS, C, B, H = K.RS, K.C, K.B, K.H
RUN = 'sna_grow_1017'
REGISTRATION = '2bb76f325dfb087f8c007d8e6eb648dc1ed53246'
ARM = 'SNA_grow'
GROW_AFTER_TASK = 1
OUT_MAIN = ROOT / 'results' / RUN / 'mlp'
CHECKS_JSON = ROOT / 'results' / RUN / 'mlp_checks.json'
SEEDS_MAIN = list(range(10))
SEEDS_CHECK = list(range(100, 110))
N_TASKS, STEPS, LR = K.N_TASKS, K.STEPS, K.LR
SN_C, SN_BETA, SN_LO, SN_HI = K.SN_C, K.SN_BETA, K.SN_LO, K.SN_HI

# the record K-mlp-rep reproduces (controls_cifar5p1_1008, C2 arm SNA, seeds 0-9)
REC_DIR = ROOT / 'results/controls_cifar5p1_1008/c2/SNA'
REC_DONE = REC_DIR / 'done.json'
REC_BACKUP = ROOT / 'results/controls_cifar5p1_1008/backup_manifest.json'
ADMISSION_1008 = ROOT / 'results/controls_cifar5p1_1008/admission_checks.json'

ENGINE_FILES = ['src/controls_cifar5p1_1008.py', 'src/resp_cifar5p1_1007.py', 'src/cifar5p1_mlp_0920.py',
                'src/rlcifar_mlp_battle_0918.py', 'src/pmnist_0905.py', 'src/pmnist_rlcifar_0907.py',
                'src/cifar_interventions_0920.py', 'analysis/cifar_ledger_0920/replay.py']

if K.C2_ARMS.get(ARM, ('SNA', 'grow')) != ('SNA', 'grow'):
    raise RuntimeError('C2_ARMS already maps SNA_grow to something else')
K.C2_ARMS[ARM] = ('SNA', 'grow')                   # register the arm in the controls engine


# --------------------------------------------------------------------------
# the activation
# --------------------------------------------------------------------------

class GrowSnake(B.SnakeFamily):
    """Host SNA (`SnakeFamily('SNA', 'snake')`) plus the rule `grow` (spec §2).

    State per hidden layer l: V[l] (R, n) the host EMA, Vmax[l] (R, n), grow[l] (R, 1) bool.
    All three are allocated once and only ever written in place."""

    def __init__(self, c=SN_C, beta=SN_BETA, lo=SN_LO, hi=SN_HI, widths=(C.HIDDEN, C.HIDDEN)):
        super().__init__('SNA', 'snake', c, beta, lo, hi, widths=widths)
        self.Vmax = None
        self.grow = None

    def init_state(self, R, device, key):
        super().init_state(R, device, key)                                   # V = 1
        self.Vmax = [torch.ones(R, w, device=device) for w in self.widths]
        self.grow = [torch.zeros(R, 1, dtype=torch.bool, device=device) for _ in self.widths]

    def v_drive(self, layer):
        """The variance alpha is computed from: Vmax where the run's flag is up, else V."""
        return torch.where(self.grow[layer], self.Vmax[layer], self.V[layer])

    def alpha(self, layer):
        return (self.c / self.v_drive(layer).sqrt()).clamp(self.lo, self.hi)          # (R, n)

    @torch.no_grad()
    def update(self, z1, z2):
        super().update(z1, z2)                                               # the host EMA of V
        for l in range(len(self.widths)):
            self.Vmax[l].copy_(torch.where(self.grow[l], torch.maximum(self.Vmax[l], self.V[l]), self.Vmax[l]))

    @torch.no_grad()
    def start_grow(self):
        for l in range(len(self.widths)):
            self.Vmax[l].copy_(self.V[l])
            self.grow[l].fill_(True)

    def state(self):
        return {'V': [v.clone() for v in self.V], 'Vmax': [v.clone() for v in self.Vmax],
                'grow': [g.clone() for g in self.grow]}

    def load_state(self, st):
        super().load_state(st)
        for dst, src in zip(self.Vmax, st['Vmax']):
            dst.copy_(src)
        for dst, src in zip(self.grow, st['grow']):
            dst.copy_(src)

    @torch.no_grad()
    def stats(self, r):
        """The host's columns.  `alpha_clip_frac` from the driving variance (see module doc)."""
        out = {}
        for l in (0, 1):
            a = self.alpha(l)[r]; W = self.V[l][r].sqrt(); raw = self.c / self.v_drive(l)[r].sqrt()
            out[f"alpha_med_l{l+1}"] = float(a.median()); out[f"alpha_min_l{l+1}"] = float(a.min())
            out[f"alpha_max_l{l+1}"] = float(a.max())
            out[f"alpha_clip_frac_l{l+1}"] = float(((raw < self.lo) | (raw > self.hi)).float().mean())
            out[f"two_alpha_W_med_l{l+1}"] = float((2 * a * W).median())
        return out


# --------------------------------------------------------------------------
# the engine and the arm loop
# --------------------------------------------------------------------------

class GrowEngine(K.SnakeEngine):
    """`controls_cifar5p1_1008.SnakeEngine` with a `GrowSnake` and the switch `start_grow`."""

    def __init__(self, arm, seeds, data, graph=True, act=None, out_name=None):
        base, mode = K.C2_ARMS[arm]
        assert base == 'SNA' and mode in ('adapt', 'grow'), (arm, base, mode)
        super().__init__(arm, seeds, data, graph=graph, act=act if act is not None else GrowSnake(),
                         out_name=out_name)
        assert self.adapting, 'the EMA of V runs in both modes'
        self.grow_started_after_task = None

    def start_grow(self, recapture=True):
        """Vmax <- V and flags up (end of task 1), then rebuild the captured step."""
        self.act.start_grow()
        self.grow_started_after_task = self.task
        if recapture:
            self.capture()


def act_snapshot(act):
    return {k: [x.detach().cpu().clone() for x in v] for k, v in act.state().items()}


def run_arm(arm, seeds, data, graph=True, n_tasks=N_TASKS, fresh=True, out_name=None, on_task=None,
            act=None, recapture=True, keep_engine=False):
    """The controls `c2_run_arm` loop for SNA (mode adapt = host) or SNA_grow (mode grow)."""
    eng = GrowEngine(arm, seeds, data, graph=graph, act=act, out_name=out_name)
    hist = {}
    t0 = time.time()
    for t in range(1, n_tasks + 1):
        eng.run_task(t)
        if eng.mode == 'grow' and t == GROW_AFTER_TASK:
            eng.start_grow(recapture=recapture)
        hist[t] = act_snapshot(eng.act)
        if on_task is not None:
            on_task(t, eng)
    fresh_rows = eng.fresh_control() if (fresh and n_tasks == N_TASKS) else []
    res = dict(rows=[dict(r) for r in eng.rows], fresh=fresh_rows, hist=hist, diverged=list(eng.diverged),
               end=cpu(dict(P=eng.P, m=eng.m, v=eng.v, tc=eng.tc, act=eng.act.state())),
               seconds=time.time() - t0, grow_started_after_task=eng.grow_started_after_task)
    if keep_engine:
        res['engine'] = eng
    else:
        del eng
        K.free_cuda()
    return res


def hist_npz(hist):
    out = {}
    for t, st in hist.items():
        for l in range(2):
            out[f't{t:02d}_l{l + 1}'] = st['V'][l].numpy()
            out[f'vmax_t{t:02d}_l{l + 1}'] = st['Vmax'][l].numpy()
            out[f'grow_t{t:02d}_l{l + 1}'] = st['grow'][l].numpy()
    return out


# --------------------------------------------------------------------------
# environment, identity
# --------------------------------------------------------------------------

def setup():
    dev = K.setup()                    # use_deterministic_algorithms(True), TF32 off, 2 threads
    return dev


def determinism():
    return dict(use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                torch_num_threads=torch.get_num_threads(),
                OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                graph='one training step captured as a CUDA graph (3 warmup steps rolled back), as the host')


def source_hashes():
    files = [f'src/{RUN}_mlp.py', f'specs/spec_{RUN}.md'] + ENGINE_FILES
    return {p: sha(ROOT / p) for p in files}


def engine_unmodified():
    """The imported engine files against the controls run's admission record (their S-src)."""
    rec = json.loads(ADMISSION_1008.read_text())['source_sha256']
    now = {p: sha(ROOT / p) for p in ENGINE_FILES}
    diff = {p: dict(now=now[p], record=rec.get(p)) for p in ENGINE_FILES if rec.get(p) != now[p]}
    return dict(all_pass=not diff, files=now, mismatches=diff, reference=str(ADMISSION_1008.relative_to(ROOT)))


def identity(seeds, data, n_tasks=N_TASKS, smoke=False):
    return RS.jnorm(dict(run_id=RUN, part='mlp', arm=ARM, base_mode=list(K.C2_ARMS[ARM]), seeds=list(seeds),
                         n_tasks=n_tasks, steps_per_task=STEPS, batch=K.BATCH, lr=LR, c=SN_C, beta=SN_BETA,
                         lo=SN_LO, hi=SN_HI, grow_after_task=GROW_AFTER_TASK, cond='std',
                         registration_commit=REGISTRATION, smoke=bool(smoke),
                         git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         source_sha256=source_hashes(), environment=environment(), determinism=determinism(),
                         data_sha256=data.sha256, input_hash=tree_hash(data.X), labels_hash=tree_hash(data.Y),
                         class_plans={str(s): C.seed_classes(s) for s in seeds}))


@contextlib.contextmanager
def maybe_lock(use, log=print):
    if use:
        with K.gpu_lock(log=log):
            yield
    else:
        yield


# --------------------------------------------------------------------------
# K-mlp-rep (spec §3): mode host through this code == the controls record, bit for bit
# --------------------------------------------------------------------------

def record_lines():
    """The committed SNA per_task.csv (sha256 from its done.json): header and {(seed, task): line}."""
    done = json.loads(REC_DONE.read_text())
    path = REC_DIR / 'per_task.csv'
    assert sha(path) == done['files']['per_task.csv'], 'record per_task.csv differs from its done.json'
    lines = path.read_text().splitlines()
    cols = next(csv.reader([lines[0]]))
    ki, kt = cols.index('seed'), cols.index('task')
    table = {}
    for line in lines[1:]:
        f = next(csv.reader([line]))
        table[(f[ki], f[kt])] = line
    return lines[0], table


def record_V():
    """The backed-up V_history.npz of the SNA record (sha256 from the controls backup manifest)."""
    man = json.loads(REC_BACKUP.read_text())
    rel = 'results/controls_cifar5p1_1008/c2/SNA/V_history.npz'
    rec = [f for f in man['files'] if f['relative'] == rel]
    assert len(rec) == 1, rel
    p = Path(rec[0]['backup'])
    assert p.is_file() and sha(p) == rec[0]['sha256'], ('backup sha256', rel)
    done = json.loads(REC_DONE.read_text())
    assert done['files']['V_history.npz'] == rec[0]['sha256'], 'backup is not the recorded V_history'
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}, str(p)


def compare_rows(rows, tasks, header, table):
    """Byte comparison of the host-formatted lines of `rows` (tasks given) with the record lines."""
    text = K.csv_text([r for r in rows if r['task'] in tasks]).splitlines()
    out = dict(header_equal=text[0] == header, lines=len(text) - 1, line_mismatches=0, column_mismatches=[])
    cols = next(csv.reader([text[0]]))
    rcols = next(csv.reader([header]))
    for line in text[1:]:
        f = next(csv.reader([line]))
        key = (f[cols.index('seed')], f[cols.index('task')])
        ref = table.get(key)
        if ref == line:
            continue
        out['line_mismatches'] += 1
        if ref is None:
            out['column_mismatches'].append(dict(seed=key[0], task=key[1], column='<missing in record>'))
            continue
        g = next(csv.reader([ref]))
        for c, a in zip(cols, f):
            b = g[rcols.index(c)] if c in rcols else '<absent>'
            if a != b:
                out['column_mismatches'].append(dict(seed=key[0], task=key[1], column=c, run=a, record=b))
    out['n_column_mismatches'] = len(out['column_mismatches'])
    out['column_mismatches'] = out['column_mismatches'][:12]
    out['bit_equal'] = bool(out['header_equal'] and out['line_mismatches'] == 0 and out['lines'] > 0)
    return out


class _GrowFromStart(GrowSnake):
    """Mutant: grow on from the first update (Vmax tracked over task 1 too, init Vmax = V = 1)."""

    def init_state(self, R, device, key):
        super().init_state(R, device, key)
        for g in self.grow:
            g.fill_(True)


def check_rep(data, log):
    t0 = time.time()
    header, table = record_lines()
    Vrec, Vpath = record_V()
    seeds = SEEDS_MAIN
    # (1) mode host (arm SNA, GrowSnake with the flags never raised), 2 tasks, the new code path
    res = run_arm('SNA', seeds, data, n_tasks=2, fresh=False, out_name='SNA')
    rows_cmp = compare_rows(res['rows'], (1, 2), header, table)
    v_eq = {f't{t:02d}_l{l + 1}': bool(np.array_equal(res['hist'][t]['V'][l].numpy(), Vrec[f't{t:02d}_l{l + 1}'])
                                        and res['hist'][t]['V'][l].numpy().dtype == Vrec[f't{t:02d}_l{l + 1}'].dtype)
            for t in (1, 2) for l in range(2)}
    flags_down = all(not bool(g.any()) for t in (1, 2) for g in res['hist'][t]['grow'])
    vmax_untouched = all(bool((v == 1).all()) for t in (1, 2) for v in res['hist'][t]['Vmax'])
    host = dict(rows=rows_cmp, V_bit_equal=v_eq, grow_flags_never_raised=flags_down, vmax_untouched=vmax_untouched,
                seconds=res['seconds'])
    host_pass = bool(rows_cmp['bit_equal'] and all(v_eq.values()) and flags_down and vmax_untouched)
    log(f'K-mlp-rep host: rows bit_equal={rows_cmp["bit_equal"]} ({rows_cmp["lines"]} lines), V {v_eq}')

    # (1b, addition: the MLP counterpart of K-cnn-off) SNA_grow's task 1 is SNA's task 1; Vmax <- V
    g = run_arm(ARM, seeds, data, n_tasks=1, fresh=False, out_name='SNA')
    g_cmp = compare_rows(g['rows'], (1,), header, table)
    st = g['hist'][1]
    g_V = all(np.array_equal(st['V'][l].numpy(), Vrec[f't01_l{l + 1}']) for l in range(2))
    g_init = all(torch.equal(st['Vmax'][l], st['V'][l]) for l in range(2))
    g_flags = all(bool(x.all()) for x in st['grow'])
    grow_t1 = dict(rows=g_cmp, V_bit_equal=bool(g_V), vmax_equals_V_after_switch=bool(g_init),
                   grow_flags_up_after_switch=bool(g_flags), switched_after_task=g['grow_started_after_task'])
    grow_pass = bool(g_cmp['bit_equal'] and g_V and g_init and g_flags and g['grow_started_after_task'] == 1)
    log(f'K-mlp-rep SNA_grow task 1: rows bit_equal={g_cmp["bit_equal"]}, V {g_V}, Vmax=V {g_init}')
    del g

    # mutants: the comparison must see a one-float32-ulp change and a rule that differs in task 1
    bad = [dict(r) for r in res['rows']]
    k = next(i for i, r in enumerate(bad) if r['task'] == 2)
    bad[k]['train_acc'] = float(np.nextafter(np.float32(bad[k]['train_acc']), np.float32(-1)))
    m1 = compare_rows(bad, (1, 2), header, table)
    mres = run_arm('SNA', seeds, data, n_tasks=1, fresh=False, out_name='SNA', act=_GrowFromStart())
    m2 = compare_rows(mres['rows'], (1,), header, table)
    del mres, res
    K.free_cuda()
    mutants = {'one_float32_ulp_in_train_acc': dict(detected=not m1['bit_equal'], line_mismatches=m1['line_mismatches']),
               'grow_from_first_update': dict(detected=not m2['bit_equal'], line_mismatches=m2['line_mismatches'],
                                              n_column_mismatches=m2['n_column_mismatches'])}
    mut_pass = all(m['detected'] for m in mutants.values())
    return dict(all_pass=bool(host_pass and grow_pass and mut_pass), host_pass=host_pass, host=host,
                sna_grow_task1_pass=grow_pass, sna_grow_task1=grow_t1, mutants=mutants, mutants_all_detected=mut_pass,
                record=dict(per_task=str((REC_DIR / 'per_task.csv').relative_to(ROOT)), V_history=Vpath,
                            per_task_sha256=sha(REC_DIR / 'per_task.csv')),
                seeds=seeds, tasks=[1, 2], comparison='host-formatted CSV lines (%.10g, all columns incl. eff_rank) '
                'byte-equal to the record; V after each task bit-equal to the backed-up record V_history.npz',
                seconds=time.time() - t0)


# --------------------------------------------------------------------------
# K-mlp-grow (spec §3): synthetic V sequence through GrowSnake
# --------------------------------------------------------------------------

def synthetic_scales(R, n, n_pre, n_post, layer):
    """(n_pre + n_post, R, n) float64 standard deviations of z per update: five unit groups.

    A grows throughout (Vmax follows V); B shrinks after the switch (Vmax stays); C shrinks then
    regrows past its old maximum; D is tiny (alpha clipped at hi); E is huge then shrinks (alpha
    clipped at lo, and stays there although V leaves the clip)."""
    third = n_post // 3
    rank = np.random.default_rng(1017 + layer).permutation(n)
    s = np.ones((n_pre + n_post, R, n))
    for u in range(n):
        k = rank[u] * 100 // n
        if k < 25:
            pre, post = np.linspace(0.5, 2.0, n_pre), np.linspace(2.0, 10.0, n_post)
        elif k < 50:
            pre, post = np.linspace(0.5, 3.0, n_pre), np.linspace(3.0, 0.3, n_post)
        elif k < 75:
            pre = np.linspace(0.5, 3.0, n_pre)
            post = np.concatenate([np.linspace(3.0, 0.5, third), np.linspace(0.5, 8.0, n_post - third)])
        elif k < 90:
            pre = np.full(n_pre, 0.02)
            post = np.concatenate([np.full(third, 0.02), np.linspace(0.02, 0.1, n_post - third)])
        else:
            pre = np.linspace(1.0, 300.0, n_pre)
            post = np.concatenate([np.full(third, 300.0), np.linspace(300.0, 1.0, n_post - third)])
        s[:n_pre, :, u] = pre[:, None]
        s[n_pre:, :, u] = post[:, None]
    return s * (1.0 + 0.25 * np.arange(R))[None, :, None]


def drive_synthetic(act, dev, R=3, n=C.HIDDEN, batch=K.BATCH, n_pre=400, n_post=1200):
    """Feed z = N(0,1) * scale to `act` and to a host SnakeFamily side by side.  Records V, Vmax,
    flags, alpha after every update; the switch after update n_pre."""
    host = B.SnakeFamily('SNA', 'snake', SN_C, SN_BETA, SN_LO, SN_HI, widths=(n, n))
    act.init_state(R, dev, key='K-mlp-grow')
    host.init_state(R, dev, key='K-mlp-grow')
    sc = [torch.tensor(synthetic_scales(R, n, n_pre, n_post, l), dtype=torch.float32) for l in range(2)]
    g = torch.Generator().manual_seed(1017)
    rec = {k: [] for k in ('V', 'Vmax', 'grow', 'alpha', 'hV', 'halpha')}

    def snap():
        rec['V'].append(torch.stack([v.cpu() for v in act.V]))
        rec['Vmax'].append(torch.stack([v.cpu() for v in act.Vmax]))
        rec['grow'].append(torch.stack([x.cpu().expand(R, n) for x in act.grow]))
        rec['alpha'].append(torch.stack([act.alpha(l).cpu() for l in range(2)]))
        rec['hV'].append(torch.stack([v.cpu() for v in host.V]))
        rec['halpha'].append(torch.stack([host.alpha(l).cpu() for l in range(2)]))

    alpha_at_switch = None
    for t in range(n_pre + n_post):
        z = [(torch.randn(R, batch, n, generator=g) * sc[l][t][:, None, :]).to(dev) for l in range(2)]
        act.update(z[0], z[1])
        host.update(z[0], z[1])
        snap()
        if t == n_pre - 1:
            act.start_grow()
            alpha_at_switch = torch.stack([act.alpha(l).cpu() for l in range(2)])
            vmax_at_switch = torch.stack([v.cpu() for v in act.Vmax])
            flags_at_switch = torch.stack([x.cpu() for x in act.grow])
    out = {k: torch.stack(v) for k, v in rec.items()}                       # (T, 2, R, n)
    out.update(alpha_at_switch=alpha_at_switch, vmax_at_switch=vmax_at_switch, flags_at_switch=flags_at_switch,
               n_pre=n_pre, n_post=n_post)
    return out


def judge_grow(d, dev):
    """The K-mlp-grow criteria on a recorded synthetic drive (float32 records, float64 references)."""
    p = d['n_pre']
    V, Vmax, al, hV, hal = d['V'], d['Vmax'], d['alpha'], d['hV'], d['halpha']
    V_act = V[p - 1]
    post_V = V[p:]
    run_max = torch.from_numpy(np.maximum.accumulate(torch.cat([V_act[None], post_V]).numpy(), axis=0))[1:]
    crit = {}
    crit['pre_V_equals_host'] = bool(torch.equal(V[:p], hV[:p]))
    crit['pre_alpha_equals_host'] = bool(torch.equal(al[:p], hal[:p]))
    crit['switch_vmax_equals_V'] = bool(torch.equal(d['vmax_at_switch'], V_act))
    crit['switch_flags_all_up'] = bool(d['flags_at_switch'].all())
    crit['switch_alpha_equals_host_alpha_of_V'] = bool(torch.equal(d['alpha_at_switch'], hal[p - 1]))
    crit['post_V_equals_host'] = bool(torch.equal(post_V, hV[p:]))           # the EMA keeps running
    crit['post_vmax_equals_running_max'] = bool(torch.equal(Vmax[p:], run_max))
    seq = torch.cat([d['alpha_at_switch'][None], al[p:]])
    crit['post_alpha_monotone_nonincreasing'] = bool((seq[1:] <= seq[:-1]).all())
    mirror = (SN_C / run_max.to(dev).sqrt()).clamp(SN_LO, SN_HI).cpu()       # same float32 op order
    crit['post_alpha_bit_equals_float32_formula_of_running_max'] = bool(torch.equal(al[p:], mirror))
    ref64 = (SN_C / run_max.double().sqrt()).clamp(SN_LO, SN_HI)
    rel = ((al[p:].double() - ref64).abs() / ref64)
    rel_max = float(rel.max())
    # float32 alpha = fl(fl(1 / fl(sqrt x)) * fl32(c)) (torch's scalar / tensor is reciprocal() * c), then
    # clamp with fl32(lo): four relative roundings, each <= u = 2^-24, so |rel| <= 4u (derived bound).
    crit['post_alpha_rel_le_4u_vs_float64_clip_c_over_sqrt_max_V'] = bool(rel_max <= REL_BOUND_4U)
    crit['post_alpha_rel_le_1e-7_vs_float64_clip_c_over_sqrt_max_V'] = bool(rel_max <= 1e-7)
    host64 = (SN_C / hV[:p].double().sqrt()).clamp(SN_LO, SN_HI)
    host_rel = ((hal[:p].double() - host64).abs() / host64)
    info = dict(post_alpha_rel_max_vs_float64=rel_max,
                post_alpha_rel_gt_1em7_count=int((rel > 1e-7).sum()), post_alpha_values=int(rel.numel()),
                host_alpha_rel_max_vs_float64_same_formula=float(host_rel.max()),
                host_alpha_rel_gt_1em7_count=int((host_rel > 1e-7).sum()), host_alpha_values=int(host_rel.numel()),
                post_updates=int(post_V.shape[0]),
                post_unit_updates_with_V_below_Vmax=int((post_V < Vmax[p:]).sum()),
                post_unit_updates_where_Vmax_rose=int((Vmax[p:] > torch.cat([d['vmax_at_switch'][None], Vmax[p:-1]])).sum()),
                post_unit_updates_alpha_differs_from_host=int((al[p:] != hal[p:]).sum()),
                end_units_alpha_at_hi=int((al[-1] == SN_HI).sum()), end_units_alpha_at_lo=int((al[-1] == SN_LO).sum()),
                end_units_host_alpha_at_lo=int((hal[-1] == SN_LO).sum()),
                end_units_total=int(al[-1].numel()))
    return crit, info


# Spec §3: thresholds are bit equality, or rel 1e-7 where the op order may differ.  The float32 mirror
# below has the code's op order, so bit equality applies to it.  The float64 reference is a different
# precision, not a different op order: it is gated at the derived bound 4u and the literal rel 1e-7 is
# reported next to the host SNA's own value for the same formula (see check_grow).
REL_BOUND_4U = 4.0 * 2.0 ** -24
GROW_BIT_CRITERIA = ('pre_V_equals_host', 'pre_alpha_equals_host', 'switch_vmax_equals_V', 'switch_flags_all_up',
                     'switch_alpha_equals_host_alpha_of_V', 'post_V_equals_host', 'post_vmax_equals_running_max',
                     'post_alpha_monotone_nonincreasing', 'post_alpha_bit_equals_float32_formula_of_running_max')


class _MutFrozen(GrowSnake):
    """Mutant: Vmax never updated after the switch (the fixT1 rule with a running EMA)."""

    @torch.no_grad()
    def update(self, z1, z2):
        B.SnakeFamily.update(self, z1, z2)


class _MutNoGrow(GrowSnake):
    """Mutant: alpha keeps following V although the flag is up (= SNA)."""

    def v_drive(self, layer):
        return self.V[layer]


class _MutInitOne(GrowSnake):
    """Mutant: Vmax starts from 1 instead of V at the switch."""

    @torch.no_grad()
    def start_grow(self):
        for l in range(len(self.widths)):
            self.Vmax[l].fill_(1.0)
            self.grow[l].fill_(True)


class _MutTrackFromStart(GrowSnake):
    """Mutant: Vmax tracked from the first update (max over task 1 too), not reset at the switch."""

    @torch.no_grad()
    def update(self, z1, z2):
        B.SnakeFamily.update(self, z1, z2)
        for l in range(len(self.widths)):
            self.Vmax[l].copy_(torch.maximum(self.Vmax[l], self.V[l]))

    @torch.no_grad()
    def start_grow(self):
        for g in self.grow:
            g.fill_(True)


class _MutEmaStops(GrowSnake):
    """Mutant: the EMA of V stops at the switch."""

    @torch.no_grad()
    def update(self, z1, z2):
        if bool(self.grow[0].any()):
            return
        super().update(z1, z2)


GROW_MUTANTS = {'vmax_frozen_after_switch': (_MutFrozen, 'post_vmax_equals_running_max'),
                'alpha_follows_V': (_MutNoGrow, 'post_alpha_monotone_nonincreasing'),
                'vmax_initialised_to_1': (_MutInitOne, 'switch_vmax_equals_V'),
                'vmax_tracked_from_first_update': (_MutTrackFromStart, 'switch_vmax_equals_V'),
                'ema_of_V_stops_at_switch': (_MutEmaStops, 'post_V_equals_host')}


def check_grow(dev, log):
    t0 = time.time()
    d = drive_synthetic(GrowSnake(), dev)
    crit, info = judge_grow(d, dev)
    bit_pass = all(crit[k] for k in GROW_BIT_CRITERIA)
    log(f'K-mlp-grow real: {crit}')
    mutants = {}
    for name, (cls, expected) in GROW_MUTANTS.items():
        mc, _ = judge_grow(drive_synthetic(cls(), dev), dev)
        failed = sorted(k for k in GROW_BIT_CRITERIA if not mc[k])
        mutants[name] = dict(expected_failure=expected, failed_criteria=failed, detected=expected in failed)
        log(f'K-mlp-grow mutant {name}: failed {failed}')
    mut_pass = all(m['detected'] for m in mutants.values())
    exercised = (info['post_unit_updates_with_V_below_Vmax'] > 0 and info['post_unit_updates_where_Vmax_rose'] > 0
                 and info['end_units_alpha_at_hi'] > 0 and info['end_units_alpha_at_lo'] > 0)
    rel_ok = crit['post_alpha_rel_le_4u_vs_float64_clip_c_over_sqrt_max_V']
    literal = crit['post_alpha_rel_le_1e-7_vs_float64_clip_c_over_sqrt_max_V']
    return dict(all_pass=bool(bit_pass and mut_pass and exercised and rel_ok), bit_criteria_pass=bit_pass,
                float64_rel_within_4u=rel_ok, rel_bound_4u=REL_BOUND_4U,
                literal_float64_rel_1em7_pass=literal,
                literal_float64_rel_1em7_note=('not gating: float32 c/sqrt(x) itself carries up to 4u of rounding; '
                                               'the host SNA alpha on the same fixture misses rel 1e-7 too '
                                               f'(max {info["host_alpha_rel_max_vs_float64_same_formula"]:.4g}, '
                                               f'{info["host_alpha_rel_gt_1em7_count"]}/{info["host_alpha_values"]} values)'),
                criteria=crit, info=info, fixture_exercises_rule=exercised,
                mutants=mutants, mutants_all_detected=mut_pass,
                fixture=dict(R=3, n=C.HIDDEN, batch=K.BATCH, updates_before_switch=d['n_pre'],
                             updates_after_switch=d['n_post'], z='N(0,1) (CPU generator 1017) x per-unit scale; '
                             'five unit groups per layer (grow / shrink / shrink-then-regrow past the max / '
                             'tiny -> clip hi / huge then shrink -> clip lo), run r scaled by 1 + 0.25 r',
                             reference='independent numpy running max of the recorded float32 V; alpha compared '
                             'bit-for-bit with clamp(c / sqrt(running max)) in float32 (same op order) and '
                             'against float64 clip(c / sqrt(max V)) at rel 4u (literal rel 1e-7 reported)'),
                seconds=time.time() - t0)


# --------------------------------------------------------------------------
# K-mlp-graph (spec §3): grow active, CUDA graph vs eager
# --------------------------------------------------------------------------

def engine_hash(eng):
    return tree_hash(dict(P=eng.P, m=eng.m, v=eng.v, tc=eng.tc, act=eng.act.state(), acc_sum=eng.acc_sum,
                          bad_step=eng.bad_step, last_hit=eng.last_hit, rows=eng.rows))


class _PyFlagGrow(GrowSnake):
    """Mutant: the flag is a Python bool (not a tensor the graph reads)."""

    def init_state(self, R, device, key):
        super().init_state(R, device, key)
        self.py_grow = False

    def v_drive(self, layer):
        return self.Vmax[layer] if self.py_grow else self.V[layer]

    @torch.no_grad()
    def update(self, z1, z2):
        B.SnakeFamily.update(self, z1, z2)
        if self.py_grow:
            for l in range(2):
                self.Vmax[l].copy_(torch.maximum(self.Vmax[l], self.V[l]))

    @torch.no_grad()
    def start_grow(self):
        super().start_grow()
        self.py_grow = True


def graph_vs_eager(data, seeds, tasks, act_cls=GrowSnake, recapture=True, fresh_batches=None):
    """Two engines (graph / eager) in lockstep through `tasks`; grow switched on after task 1."""
    engs = {m: GrowEngine(ARM, seeds, data, graph=(m == 'graph'), act=act_cls(), out_name=ARM)
            for m in ('graph', 'eager')}
    assert engs['graph'].cg is not None and engs['eager'].cg is None
    per_task = []
    for t in tasks:
        for e in engs.values():
            e.run_task(t)
            if t == GROW_AFTER_TASK:
                e.start_grow(recapture=recapture)
        hg, he = engine_hash(engs['graph']), engine_hash(engs['eager'])
        rows_eq = K.csv_text(engs['graph'].rows) == K.csv_text(engs['eager'].rows)
        a = engs['graph'].act
        per_task.append(dict(task=t, state_bit_equal=hg == he, rows_byte_equal=rows_eq,
                             grow_on=bool(all(bool(x.all()) for x in a.grow)),
                             units_V_below_Vmax=int(sum(int((v < m).sum()) for v, m in zip(a.V, a.Vmax))),
                             units_total=int(sum(v.numel() for v in a.V))))
    fresh = None
    if fresh_batches is not None:
        for e in engs.values():
            e.saved_batches = fresh_batches
            e.fresh_control()                          # rows need task 29; the state is compared
        fresh = dict(state_bit_equal=engine_hash(engs['graph']) == engine_hash(engs['eager']),
                     flags_down_after_reset=bool(not any(bool(x.any()) for x in engs['graph'].act.grow)))
    del engs
    K.free_cuda()
    return per_task, fresh


def check_graph(data, log):
    t0 = time.time()
    seeds = SEEDS_CHECK
    tasks = (1, 2, 3)
    rows3 = data.task_rows(seeds, 3)
    fb = torch.stack([C.batch_indices(H.stream('sna_grow_1017_kgraph_fresh', s), rows3[r], STEPS)
                      for r, s in enumerate(seeds)]).to(data.device)
    real, fresh = graph_vs_eager(data, seeds, tasks, fresh_batches=fb)
    log(f'K-mlp-graph real: {real} fresh {fresh}')
    real_pass = all(p['state_bit_equal'] and p['rows_byte_equal'] for p in real) and fresh['state_bit_equal']
    exercised = all(p['grow_on'] and p['units_V_below_Vmax'] > 0 for p in real if p['task'] > GROW_AFTER_TASK)
    # informational: static flags make the recapture unnecessary
    norecap, _ = graph_vs_eager(data, seeds, (1, 2), recapture=False)
    # mutant: a Python flag and no recapture -> the captured step never sees the switch
    mut, _ = graph_vs_eager(data, seeds, (1, 2), act_cls=_PyFlagGrow, recapture=False)
    mut_detected = not all(p['state_bit_equal'] for p in mut)
    log(f'K-mlp-graph no-recapture (static flags): {norecap}; mutant python flag: {mut}')
    return dict(all_pass=bool(real_pass and exercised and mut_detected), real=real, fresh_control=fresh,
                graph_vs_eager_bit_equal=bool(real_pass), grow_exercised=bool(exercised),
                steps_with_grow_active=STEPS * (len(tasks) - GROW_AFTER_TASK),
                static_flags_without_recapture=dict(per_task=norecap,
                                                    bit_equal=all(p['state_bit_equal'] for p in norecap)),
                mutants={'python_flag_no_recapture': dict(per_task=mut, detected=mut_detected)},
                seeds=seeds, tasks=list(tasks),
                compared='P, Adam m/v, tc, V, Vmax, grow flags, acc_sum, bad_step, last_hit and the per-task rows '
                         'after every task; then the host fresh control (flags reset) from the task-3 state',
                seconds=time.time() - t0)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def run_checks(out_json=CHECKS_JSON, lock=False, only=None, log=print):
    t0 = time.time()
    with maybe_lock(lock, log):
        dev = setup()
        res = dict(run_id=RUN, part='mlp', registration_commit=REGISTRATION, source_sha256=source_hashes(),
                   engine_unmodified=engine_unmodified(), determinism=determinism(), environment=environment(),
                   git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        which = only or ('K-mlp-grow', 'K-mlp-rep', 'K-mlp-graph')
        if 'K-mlp-grow' in which:
            res['K-mlp-grow'] = check_grow(dev, log)
            K.free_cuda()
        if 'K-mlp-rep' in which or 'K-mlp-graph' in which:
            data = RS.Data(dev)
            if 'K-mlp-rep' in which:
                res['K-mlp-rep'] = check_rep(data, log)
                K.free_cuda()
            if 'K-mlp-graph' in which:
                res['K-mlp-graph'] = check_graph(data, log)
                K.free_cuda()
            del data
        res['required'] = ['K-mlp-rep', 'K-mlp-grow', 'K-mlp-graph']
        res['all_pass'] = bool(res['engine_unmodified']['all_pass']
                               and all(res.get(k, {}).get('all_pass') is True for k in res['required']))
        res['seconds'] = time.time() - t0
        res['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        res['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
    put(out_json, res)
    log(f'checks all_pass={res["all_pass"]} -> {out_json}')
    return res


def run_main(out=OUT_MAIN, seeds=SEEDS_MAIN, n_tasks=N_TASKS, smoke=False, lock=True, log=print):
    out = Path(out)
    with maybe_lock(lock, log):
        dev = setup()
        data = RS.Data(dev)
        ident = identity(seeds, data, n_tasks, smoke)
        if not smoke:
            assert out.resolve() == OUT_MAIN.resolve() and seeds == SEEDS_MAIN and n_tasks == N_TASKS
            chk = json.loads(CHECKS_JSON.read_text())
            assert chk['all_pass'] and chk['source_sha256'] == source_hashes(), 'unverified code'
            dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *source_hashes()],
                                            cwd=ROOT, text=True)
            assert not dirty, 'commit the implementation before the main run'
        else:
            assert out.resolve() != OUT_MAIN.resolve(), 'a smoke run must not write the main output'
        d = out / ARM
        if (d / 'done.json').exists():
            RS.verify_done(d / 'done.json', ident)
            log(f'{d} already complete')
            return True
        d.mkdir(parents=True, exist_ok=True)
        if not smoke:
            put(out / 'admission_checks.json', chk)
        put(out / 'input_manifest.json', dict(data_sha256=ident['data_sha256'], input_hash=ident['input_hash'],
                                              labels_hash=ident['labels_hash'], class_plans=ident['class_plans']))
        put(out / 'provenance_start.json', dict(identity=ident, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                started_jst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                                                pid=os.getpid(), argv=sys.argv, independent_audit=False,
                                                dirty_diff=subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'specs'],
                                                                                   cwd=ROOT, text=True)))
        t0 = time.time()

        def on_task(t, eng):
            put(out / 'status.json', dict(stage='running', completed_task=t, wall_s=time.time() - t0))
            log(f'[{time.strftime("%T")}] {ARM} task {t:2d}/{n_tasks} done ({time.time() - t0:.1f}s)'
                + (' -- grow on, recaptured' if t == GROW_AFTER_TASK else ''))
        res = run_arm(ARM, seeds, data, n_tasks=n_tasks, fresh=True, on_task=on_task)
        RS.write_csv(d / 'per_task.csv', res['rows'])
        RS.write_csv(d / 'fresh_control.csv', res['fresh'])
        save_npz(d / 'V_history.npz', hist_npz(res['hist']))
        save_pt(d / 'end.pt', res['end'])
        put(d / 'arm_provenance.json', dict(arm=ARM, base='SNA', mode='grow', grow_after_task=GROW_AFTER_TASK,
                                            seconds=res['seconds'], seeds=seeds, c=SN_C, beta=SN_BETA, lo=SN_LO,
                                            hi=SN_HI, lr=LR, steps_per_task=STEPS, n_tasks=n_tasks,
                                            divergences=res['diverged'],
                                            grow_started_after_task=res['grow_started_after_task']))
        files = sorted(p for p in d.iterdir() if p.name != 'done.json' and not p.name.endswith('.tmp'))
        RS.mark_done(d / 'done.json', ident, files)
        put(out / 'status.json', dict(stage='completed' if not res['diverged'] else 'completed_with_divergence',
                                      divergences=res['diverged'], wall_s=time.time() - t0))
        put(out / 'provenance_end.json', dict(identity=ident, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                              seconds=time.time() - t0,
                                              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                              peak_cuda_bytes=torch.cuda.max_memory_allocated(), independent_audit=False))
        log(f'{ARM}: {len(res["rows"])} rows, {len(res["fresh"])} fresh rows, divergences {len(res["diverged"])}, '
            f'{res["seconds"]:.1f}s -> {d}')
        return not res['diverged']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run', 'checks'])
    p.add_argument('--out', default=None)
    p.add_argument('--seeds', default=None)
    p.add_argument('--tasks', type=int, default=N_TASKS)
    p.add_argument('--smoke', action='store_true', help='run: any seeds/out other than the main output')
    p.add_argument('--only', default=None, help='checks: comma-separated subset')
    p.add_argument('--lock', dest='lock', action='store_true', default=None)
    p.add_argument('--no-lock', dest='lock', action='store_false')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.stage == 'checks':
        run_checks(Path(a.out) if a.out else CHECKS_JSON, lock=bool(a.lock),
                   only=tuple(a.only.split(',')) if a.only else None)
    else:
        seeds = B.parse_ints(a.seeds) if a.seeds else SEEDS_MAIN
        ok = run_main(Path(a.out) if a.out else OUT_MAIN, seeds, a.tasks, a.smoke, lock=True if a.lock is None else a.lock)
        print('completed' if ok else 'completed with divergence', flush=True)


if __name__ == '__main__':
    main()
