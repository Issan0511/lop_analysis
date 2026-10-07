#!/usr/bin/env python3
"""controls_cifar5p1_1008 -- two registered controls on the 5+1 CIFAR x MLP box (std, Adam 1e-4).

Registered in specs/spec_controls_cifar5p1_1008.md (commit 2adc8331) before this file existed.

C1  Does the layer-2 response field of the healthy net (task 2 end) restore the degraded net
    (task 28 end) on task 29 because of its learned pattern, or because it opens gates?
    Arms N_c and R_h are `resp_cifar5p1_1007`'s N_c and R_c<-h, run through that module's
    unmodified `run_arm`.  The new field arms (R_fresh, R_match, R_perm, R_rand) use the same
    unmodified `Field` / `AnchorAdd` / `Engine`; only the table d differs (spec §3.2).

C2  Does adaptive Snake / KKT1 win because alpha tracks the unit width?  `SnakeEngine` is the
    host's `cifar5p1_mlp_0920.run()` re-expressed with explicit state for one Snake-family arm,
    with the EMA of V switchable: never (fix0), only in task 1 (fixT1), always (the host).
    SNA_pus divides z by the unit width W = c/alpha and applies the fixed-alpha Snake.

    python -m src.controls_cifar5p1_1008 --out results/controls_cifar5p1_1008          # main run
    python -m src.controls_cifar5p1_1008 --check-mode --seeds 100-109 --out <dir>      # checks only
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
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

from src import resp_cifar5p1_1007 as RS          # verified engine (unmodified)
from src.cifar_interventions_0920 import cpu, tree_hash, save_pt
from analysis.cifar_ledger_0920.replay import sha, put, save_npz, environment, clean

C = RS.C
B = RS.B
H = RS.H
RUN = 'controls_cifar5p1_1008'
REGISTRATION = '2adc8331'
OUT_MAIN = ROOT / 'results' / RUN
LR, B1, B2, EPS = RS.LR, RS.B1, RS.B2, RS.EPS     # 1e-4, 0.9, 0.999, 1e-8
STEPS = RS.STEPS                                 # 780
BATCH = RS.BATCH                                 # 32
N_TRAIN = RS.N_TRAIN
TH, TC = RS.TH, RS.TC                            # 2, 28
N_TASKS = C.N_TASKS                              # 30
LATE = (21, 23, 25, 27, 29)
EARLY = (1, 3, 5, 7, 9)
GPU_LOCK = Path('/tmp/lop_analysis_gpu.lock')
RAM_NEED_GB = 8.5          # measured peak RSS 2.40 GB (spec §1.2(a)) + 6 GB kept free
GPU_NEED_GB = 5.0

# ---- C1 (spec §3) ----
C1_ARMS = ('N_c', 'R_h', 'R_fresh', 'R_match', 'R_perm', 'R_rand')
C1_SAVE = (TH, TH + 1, TC, TC + 1)                # t03 only for the S1-wiring mutant (donor t03)
FRESH_OFFSET = 1000                              # R_fresh donor: init_params('R', 1000 + s)
RESP_ROWS = ROOT / 'results/resp_cifar5p1_1007/arms'
RESP_BACKUP = ROOT / 'results/resp_cifar5p1_1007/backup_manifest.json'
C1_DONOR = {'R_h': 't02', 'R_fresh': f'init{FRESH_OFFSET}+s', 'R_match': 'self+c',
            'R_perm': 't02[perm]', 'R_rand': 't02[shuffle]'}

# ---- C2 (spec §4) ----
C2_ARMS = {'SNA': ('SNA', 'adapt'), 'KKT1': ('KKT1', 'adapt'),
           'SNA_fix0': ('SNA', 'fix0'), 'KKT1_fix0': ('KKT1', 'fix0'),
           'SNA_fixT1': ('SNA', 'fixT1'), 'KKT1_fixT1': ('KKT1', 'fixT1'),
           'SNA_pus': ('SNA', 'pus')}
SN_C, SN_BETA, SN_LO, SN_HI = 0.6, 0.01, 0.005, 3.0
RECORD_0920 = ROOT / 'results/cifar5p1_mlp_0920'
HOST_FLOAT64_COLS = RS.HOST_FLOAT64_COLS          # eff_rank_l1, eff_rank_l2


# --------------------------------------------------------------------------
# environment, lock, identity
# --------------------------------------------------------------------------

def setup():
    return RS.setup()                            # deterministic, TF32 off, torch 2.13.0+cu130


def mem_available_gb():
    return RS.mem_available_gb()


def gpu_free_gb():
    return RS.gpu_free_gb()


@contextlib.contextmanager
def gpu_lock(wait_s=12 * 3600, poll=20, log=print):
    """The shared lab lock (blocking flock), then wait for RAM and GPU room (spec §8)."""
    with GPU_LOCK.open('a') as lock:
        t0 = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        log(f'GPU lock acquired after {time.time() - t0:.0f}s')
        n = 0
        while True:
            free, gfree = mem_available_gb(), gpu_free_gb()
            if free >= RAM_NEED_GB and gfree >= GPU_NEED_GB:
                break
            assert time.time() - t0 < wait_s, ('RAM/GPU wait exceeded', free, gfree)
            if n % 6 == 0:
                log(f'[{time.strftime("%T")}] waiting: MemAvailable {free:.1f} GB (need {RAM_NEED_GB}), '
                    f'GPU free {gfree:.1f} GB; other GPU python {RS.other_gpu_python()}')
            n += 1
            time.sleep(poll)
        log(f'[{time.strftime("%T")}] start: MemAvailable {mem_available_gb():.1f} GB, GPU free {gpu_free_gb():.1f} GB')
        yield


def source_hashes():
    files = [ROOT / f'src/{RUN}.py', ROOT / f'specs/spec_{RUN}.md', ROOT / 'src/resp_cifar5p1_1007.py',
             ROOT / 'src/cifar5p1_mlp_0920.py', ROOT / 'src/rlcifar_mlp_battle_0918.py', ROOT / 'src/pmnist_0905.py',
             ROOT / 'src/pmnist_rlcifar_0907.py', ROOT / 'src/cifar_interventions_0920.py',
             ROOT / 'analysis/cifar_ledger_0920/replay.py', ROOT / 'analysis/resp_cifar_ee_0920/stats.py']
    files += sorted((ROOT / f'analysis/{RUN}').glob('*.py'))
    files += sorted((ROOT / f'analysis/{RUN}').glob('*.sh'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def identity(seeds, data, check_mode=False, steps=STEPS):
    return RS.jnorm(dict(run_id=RUN, seeds=list(seeds), steps_per_task=steps, n_tasks=N_TASKS,
                         c1=dict(branch=TC, donor=TH, arms=list(C1_ARMS), fresh_offset=FRESH_OFFSET),
                         c2=dict(arms=C2_ARMS, c=SN_C, beta=SN_BETA, lo=SN_LO, hi=SN_HI, late=LATE),
                         registration_commit=REGISTRATION, check_mode=bool(check_mode),
                         git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         source_sha256=source_hashes(), environment=environment(), data_sha256=data.sha256,
                         input_hash=tree_hash(data.X), labels_hash=tree_hash(data.Y),
                         class_plans={str(s): C.seed_classes(s) for s in seeds}))


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def done_or_none(path, ident):
    """A completed unit is skipped only when its marker carries the same identity and hashes."""
    if not Path(path).exists():
        return None
    return RS.verify_done(path, ident)


# --------------------------------------------------------------------------
# C1: fields (spec §3.2)
# --------------------------------------------------------------------------

def fresh_donor_params(seeds, dev):
    """The host's init (`init_params('R', ...)`) at seed FRESH_OFFSET + s, stacked like the engine."""
    init = [q.detach() for s in seeds for q in C.init_params('R', FRESH_OFFSET + s, dev, C.HIDDEN)]
    return [torch.stack(init[i::6]).contiguous() for i in range(6)]


def derangement(seed, n=100):
    """First permutation without fixed points drawn from the registered stream (spec §3.2)."""
    g = H.stream('ctrl1008_perm', seed)
    while True:
        p = torch.randperm(n, generator=g)
        if bool((p != torch.arange(n)).all()):
            return p


def shuffle_order(seed, n):
    return torch.randperm(n, generator=H.stream('ctrl1008_rand', seed))


def match_shift(zb_row, K):
    """c = fl32(-(v_K + v_{K+1})/2) over the receiver's values sorted descending (float64)."""
    v = np.sort(zb_row.detach().double().flatten().cpu().numpy())[::-1]
    return np.float32(-(v[K - 1] + v[K]) / 2.0)


def c1_field(name, states, data):
    """(target64, d, ids, info) on task TC+1's images for a field arm.  target64 is the float64
    value the arm's shifted argument should reproduce at the branch; d is float32."""
    assert name in C1_DONOR, name
    dev = data.device
    st = states[TC]
    seeds = list(st['seeds'])
    R = len(seeds)
    rows = data.task_rows(seeds, TC + 1)
    ids = torch.stack(rows).to(dev)
    N = ids.shape[1]
    X = data.X[ids]
    zb = RS.eval_z([q.to(dev) for q in st['P']], X, 2)
    info = {}
    if name != 'R_fresh':
        zs = RS.eval_z([q.to(dev) for q in states[TH]['P']], X, 2)
    if name == 'R_h':
        T = zs
    elif name == 'R_fresh':
        T = RS.eval_z(fresh_donor_params(seeds, dev), X, 2)
    elif name == 'R_perm':
        perms = [derangement(s) for s in seeds]
        T = torch.stack([zs[r][:, perms[r].to(dev)] for r in range(R)])
        info['perm'] = torch.stack(perms)
    elif name == 'R_rand':
        orders = [shuffle_order(s, N * zs.shape[-1]) for s in seeds]
        T = torch.stack([zs[r].reshape(-1)[orders[r].to(dev)].reshape(N, -1) for r in range(R)])
    del X
    if name == 'R_match':
        d_h = (zs.double() - zb.double()).float()
        K = ((zb + d_h) >= 0).flatten(1).sum(1).cpu().tolist()
        shifts = [match_shift(zb[r], K[r]) for r in range(R)]
        c = torch.tensor(np.array(shifts, dtype=np.float32), device=dev)
        d = c[:, None, None].expand_as(zb).contiguous()
        target64 = zb.double() + d.double()
        info.update(K=K, shift=[float(x) for x in shifts],
                    K_achieved=((zb + d) >= 0).flatten(1).sum(1).cpu().tolist())
    else:
        d = (T.double() - zb.double()).float()
        target64 = T.double()
    return target64, d, ids, info


def c1_table(d, ids):
    R = ids.shape[0]
    table = torch.full((R, N_TRAIN, d.shape[-1]), float('nan'), device=d.device)
    table[torch.arange(R, device=d.device)[:, None], ids] = d
    return table


def c1_engine(name, states, data, graph=True, built=None):
    target64, d, ids, info = c1_field(name, states, data) if built is None else built
    st = states[TC]
    field = RS.Field([p.to(data.device) for p in st['P']], 2, c1_table(d, ids))
    eng = RS.Engine(st['seeds'], data, state=st, field=field, graph=graph)
    return eng, (target64, d, ids, info)


def c1_preflight(name, eng, states, batches, built):
    """S1-branch + S1-field (spec §7): resp's preflight with the arm's own target T.

    Bit-exact branch outputs on all task images at once, every training batch and row-order
    chunks of 32; the stored table equals the registered d and is NaN outside the task; the
    shifted argument is within the rounding bound of T; outside the bound the gate equals
    the gate of T; the arm's own training gate counts (autograd of its forward) agree."""
    target64, d_ref, ids_all, info = built
    dev = eng.device
    data = eng.data
    bp = [q.to(dev) for q in states[TC]['P']]
    N = ids_all.shape[1]
    ar = eng.ar
    u32, u64 = 2.0 ** -24, 2.0 ** -53
    eta = float(torch.finfo(torch.float32).tiny)
    report = dict(arm=name, batches=0, branch_bit_exact=True)
    assert eng.field.layer == 2, 'field layer'
    assert tree_hash(eng.field.P0) == tree_hash(bp), 'P0 is not the branch state'
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(eng.P, eng.field.P0)), 'P0 aliased'
    X = data.X[ids_all]
    zb_eval = RS.eval_z(bp, X, 2)
    del X
    assert tree_hash(eng.field.d[ar, ids_all]) == tree_hash(d_ref), 'field values'
    outside = torch.ones(eng.R, N_TRAIN, dtype=torch.bool, device=dev)
    outside[ar, ids_all] = False
    assert bool(torch.isnan(eng.field.d[outside]).all()), 'field defined outside the task'
    del outside
    sd = d_ref.double().std(dim=(1, 2))
    if name == 'R_match':
        assert bool((d_ref == d_ref[:, :1, :1]).all()), 'R_match field is not one constant per seed'
    else:
        assert bool((sd > 0).all()), 'constant field'
    q_all = zb_eval.double() + d_ref.double() - target64
    pos = torch.full((eng.R, N_TRAIN), -1, dtype=torch.long, device=dev)
    pos[ar, ids_all] = torch.arange(N, device=dev)[None, :].expand(eng.R, -1)
    report.update(field_sd_per_seed=sd.cpu().tolist(), max_abs_q=float(q_all.abs().max()))
    arg_ratio = torch.zeros(eng.R, dtype=torch.float64, device=dev)
    ambiguous = torch.zeros(eng.R, dtype=torch.long, device=dev)
    gate_pairs = torch.zeros(eng.R, dtype=torch.long, device=dev)
    count_slack_max = 0
    gens = [ids_all] + [batches[:, j] for j in range(batches.shape[1])]
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
        loc = pos[ar, ids]
        assert bool((loc >= 0).all()), 'batch row outside the task'
        zb_mb = natural[2]
        d = d_ref[ar, loc]
        arg = zb_mb + d
        target = target64[ar, loc]
        zbe = zb_eval[ar, loc].double()
        bound = (q_all[ar, loc].abs() + (zb_mb.double() - zbe).abs()
                 + u32 * (zb_mb.double().abs() + d.double().abs()) + eta
                 + 3 * u64 * (zbe.abs() + d.double().abs() + target.abs()))
        err = (arg.double() - target).abs()
        assert bool((err <= bound).all()), 'field argument'
        arg_ratio = torch.maximum(arg_ratio, (err / bound).flatten(1).max(1).values)
        sure = target.abs() > bound
        g_arg = RS.gtrain(arg)
        g_tgt = RS.gtrain(target.float())
        assert bool((g_arg[sure] == g_tgt[sure]).all()), 'field gate'
        ambiguous += (~sure).flatten(1).sum(1)
        gate_pairs += sure.flatten(1).sum(1)
        with torch.enable_grad():
            vals = eng.forward(xb, ids)
            counts = torch.autograd.grad(vals[3].sum(), eng.P[3])[0]
        expect = g_tgt.sum(1)
        slack = (~sure).sum(1)
        diff = (counts.double() - expect.double()).abs()
        assert bool((diff <= slack.double()).all()), 'training gate'
        count_slack_max = max(count_slack_max, int(diff.max()))
        report['batches'] += 1
    report.update(argument_error_ratio_per_seed=arg_ratio.cpu().tolist(),
                  argument_error_ratio=float(arg_ratio.max()),
                  ambiguous_pairs_per_seed=ambiguous.cpu().tolist(),
                  gate_checked_pairs_per_seed=gate_pairs.cpu().tolist(),
                  training_gate_count_max_difference=count_slack_max,
                  open_pairs_at_branch_per_seed=((zb_eval + d_ref) >= 0).flatten(1).sum(1).cpu().tolist())
    report.update({k: v for k, v in info.items() if k in ('K', 'shift', 'K_achieved')})
    report['all_pass'] = True
    return report


def c1_run_field_arm(name, states, data, graph=True, preflight_on=True, built=None):
    """`resp_cifar5p1_1007.run_arm` for a C1 field arm (same engine, same diagnostics, same rows)."""
    eng, built = c1_engine(name, states, data, graph=graph, built=built)
    T = TC + 1
    rows_T, batches = eng.task_batches(T)
    ids = torch.stack(rows_T).to(eng.device)
    Y = data.Y[ids]
    rng_before = tree_hash({s: g.get_state() for s, g in eng.g_batch.items()})
    pf = c1_preflight(name, eng, states, batches, built) if preflight_on else None
    history = []
    r0, u0 = eng.diagnostic(ids, Y, 0)
    history.append((0, r0, u0))
    assert rng_before == tree_hash({s: g.get_state() for s, g in eng.g_batch.items()}), 'diagnostic consumed RNG'

    def on_diag(update):
        rr, uu = eng.diagnostic(ids, Y, update)
        history.append((update, rr, uu))
    eng.train_steps(batches, RS.DIAG_STEPS, on_diag)
    eng.task = T
    finite = eng.finite()
    final = {r['seed']: r for r in history[-1][1]}
    label_hash = tree_hash(Y)
    batch_hash = tree_hash(batches)
    end = []
    for r, s in enumerate(eng.seeds):
        end.append(dict(arm=name, display=name, seed=s, task=T, branch=TC, layer=2, donor=C1_DONOR[name],
                        reset=0, online_acc=float(eng.acc_sum[r]) / STEPS, online_ce=float(eng.ce_sum[r]) / STEPS,
                        memo_acc=final[s]['memo_acc'], memo_ce=final[s]['memo_ce'], finite=bool(finite[r]),
                        label_hash=label_hash, batch_hash=batch_hash))
    result = dict(rows=end, history=history, preflight=pf, end_state=eng.state(), host_rows=None,
                  batches=batches.cpu(),
                  field=dict(layer=2, ids=ids.cpu(), d=eng.field.d[eng.ar, ids].cpu()),
                  field_info={k: (v.tolist() if torch.is_tensor(v) else v) for k, v in built[3].items()})
    del eng
    free_cuda()
    return result


def c1_run_arm(name, states, data, graph=True, preflight_on=True):
    """N_c and R_h run through resp's own `run_arm`; the four new arms through the field path."""
    if name == 'N_c':
        return RS.run_arm('N_c', states, data, graph=graph, preflight_on=preflight_on)
    if name == 'R_h':
        return RS.run_arm('R_ch', states, data, graph=graph, preflight_on=preflight_on)
    if name not in C1_ARMS:
        raise KeyError(name)
    return c1_run_field_arm(name, states, data, graph=graph, preflight_on=preflight_on)


def resp_rows(name):
    """The committed resp rows of N_c / R_c<-h (spec §3.4 (2))."""
    return json.loads((RESP_ROWS / {'N_c': 'N_c', 'R_h': 'R_ch'}[name] / 'rows.json').read_text())


def reproduction_check(name, rows):
    ref = resp_rows(name)
    mine = RS.jnorm(rows)
    keys = ('seed', 'online_acc', 'online_ce', 'memo_acc', 'memo_ce', 'label_hash', 'batch_hash', 'finite')
    same_keys = all({k: a[k] for k in keys} == {k: b[k] for k in keys} for a, b in zip(mine, ref)) and len(mine) == len(ref)
    same_all = mine == ref
    return dict(arm=name, all_pass=bool(same_keys and same_all), registered_columns_equal=bool(same_keys),
                full_row_equal=bool(same_all), reference=str((RESP_ROWS / {'N_c': 'N_c', 'R_h': 'R_ch'}[name] / 'rows.json').relative_to(ROOT)))


def backup_state_check(states):
    """t02/t28/t29 core states against resp's backed-up prefix (sha256 from its manifest)."""
    man = json.loads(RESP_BACKUP.read_text())
    out = {}
    for t in C1_SAVE:
        rel = f'results/resp_cifar5p1_1007/prefix/t{t:02d}.pt'
        rec = [f for f in man['files'] if f['relative'] == rel]
        assert len(rec) == 1, rel
        p = Path(rec[0]['backup'])
        assert p.is_file() and sha(p) == rec[0]['sha256'], ('backup sha256', rel)
        ref = RS.load_pt(p)
        same = tree_hash(RS.core_state(ref)) == tree_hash(RS.core_state(states[t]))
        same_b = tree_hash(ref['last_batches']) == tree_hash(states[t]['last_batches'])
        out[f't{t:02d}'] = dict(core_bit_exact=bool(same), last_batches_bit_exact=bool(same_b), backup=str(p))
    out['all_pass'] = all(v['core_bit_exact'] and v['last_batches_bit_exact'] for k, v in out.items() if k != 'all_pass')
    return out


# --------------------------------------------------------------------------
# C1 run
# --------------------------------------------------------------------------

def c1_prefix(out, seeds, data, ident, check_mode, log, stop_hook=None):
    """The natural prefix (tasks 1-29 + fresh), atomic: STOP is honoured only at arm boundaries."""
    prefix = out / 'c1' / 'prefix'
    prefix.mkdir(parents=True, exist_ok=True)
    marker = prefix / 'done.json'
    t0 = time.time()
    if not marker.exists():
        eng = RS.Engine(seeds, data)
        rows = []

        def save(t, st, batches, rows_now):
            if t in C1_SAVE:
                save_pt(prefix / f't{t:02d}.pt', dict(**st, last_batches=batches))
            put(out / 'status.json', dict(stage='c1_prefix', completed_task=t, wall_s=time.time() - t0))

        def on_task(t):
            if stop_hook is not None:
                stop_hook('c1_prefix', t)
        rows, _ = RS.natural_prefix(eng, RS.PREFIX_TASKS, save=save, rows=rows, stop=None, on_task=on_task)
        last = RS.load_pt(prefix / f't{TC + 1:02d}.pt')['last_batches'].to(data.device)
        fv, ff = eng.fresh(last)
        assert bool(ff.all()), 'DIVERGED fresh control'
        fresh = RS.fresh_rows(seeds, fv, rows, TC + 1)
        RS.write_csv(prefix / 'per_task.csv', rows)
        RS.write_csv(prefix / 'fresh_control.csv', fresh)
        put(prefix / 'rows.json', rows)
        put(prefix / 'fresh.json', fresh)
        del eng
        free_cuda()
        states = {t: RS.load_pt(prefix / f't{t:02d}.pt') for t in C1_SAVE}
        if not check_mode:
            ok, comparison = RS.compare_record(rows, fresh)
            RS.write_csv(prefix / 'host_record_comparison.csv', comparison)
            bk = backup_state_check(states)
            put(prefix / 'host_record_check.json', dict(
                all_pass=bool(ok and bk['all_pass']), record_all_pass=bool(ok),
                gated_mismatches=sum(1 for c in comparison if c['gated'] and not c['byte_equal']),
                eff_rank_mismatches=sum(1 for c in comparison if not c['gated'] and not c['byte_equal']),
                eff_rank_max_relative_difference=max([c['relative_difference'] for c in comparison if not c['gated']] or [0.0]),
                resp_backup_states=bk))
            if not (ok and bk['all_pass']):
                put(out / 'status.json', dict(stage='CHECK_FAILED', reason='C1 prefix vs 0920 record / resp backup'))
                raise SystemExit('CHECK_FAILED: C1 prefix does not reproduce the 0920 record / resp states')
        files = sorted(p for p in prefix.iterdir() if p.name != 'done.json' and not p.name.endswith('.tmp'))
        RS.mark_done(marker, ident, files)
        log(f'C1 prefix complete: {time.time() - t0:.1f}s')
    else:
        RS.verify_done(marker, ident)
    states = {t: RS.load_pt(prefix / f't{t:02d}.pt') for t in C1_SAVE}
    prefix_rows = json.loads((prefix / 'rows.json').read_text())
    return states, prefix_rows


def c1_run(out, seeds, data, ident, check_mode, log, stop_hook=None):
    stop = out / 'STOP'
    states, prefix_rows = c1_prefix(out, seeds, data, ident, check_mode, log, stop_hook)
    t0 = time.time()
    for name in C1_ARMS:
        if stop.exists():
            return False
        arm = out / 'c1' / 'arms' / name
        arm.mkdir(parents=True, exist_ok=True)
        if done_or_none(arm / 'done.json', ident) is not None:
            continue
        res = c1_run_arm(name, states, data)
        assert all(r['finite'] for r in res['rows']), ('DIVERGED arm', name)
        put(arm / 'preflight.json', res['preflight'])
        put(arm / 'rows.json', res['rows'])
        RS.write_csv(arm / 'per_task.csv', res['rows'])
        RS.serialize_history(arm, res['history'])
        save_pt(arm / 'end.pt', res['end_state'])
        if res['field'] is not None:
            save_pt(arm / 'field.pt', res['field'])
        if res.get('field_info'):
            put(arm / 'field_info.json', res['field_info'])
        if name == 'N_c':
            put(arm / 'prefix_check.json', RS.prefix_check('N_c', res, states, prefix_rows))
        if name in ('N_c', 'R_h') and not check_mode:
            chk = reproduction_check(name, res['rows'])
            put(arm / 'reproduction_check.json', chk)
            if not chk['all_pass']:
                put(out / 'status.json', dict(stage='CHECK_FAILED', reason=f'C1 {name} does not reproduce resp'))
                raise SystemExit(f'CHECK_FAILED: {name} differs from resp_cifar5p1_1007')
        files = sorted(p for p in arm.iterdir() if p.name != 'done.json' and not p.name.endswith('.tmp'))
        RS.mark_done(arm / 'done.json', ident, files)
        put(out / 'status.json', dict(stage='c1_arms', completed_arm=name, wall_s=time.time() - t0))
        log(f'C1 arm {name} complete: {time.time() - t0:.1f}s')
        del res
        free_cuda()
        if stop_hook is not None:
            stop_hook('c1_arm', name)
    return True


# --------------------------------------------------------------------------
# C2: activations and the engine (spec §4)
# --------------------------------------------------------------------------

class PerUnitScaleSnake(B.SnakeFamily):
    """SNA_pus: u = z / W, phi = u + sin(c u)^2 / c, W = c / alpha = clip(sqrt V, c/hi, c/lo).

    V is SNA's EMA of the batch variance of z (inherited `update`, same beta); alpha is SNA's.
    In exact arithmetic phi_pus = phi_SNA / W."""

    def __init__(self, c=SN_C, beta=SN_BETA, lo=SN_LO, hi=SN_HI, widths=(C.HIDDEN, C.HIDDEN)):
        super().__init__('SNA_pus', 'snake', c, beta, lo, hi, widths=widths)

    def width(self, layer):
        return self.c / self.alpha(layer)                                   # (R, n)

    def phi(self, z, layer=0, train=False):
        W = self.width(layer)[:, None, :]
        u = z / W
        return u + torch.sin(self.c * u) ** 2 / self.c

    def dphi(self, z, layer=0):
        W = self.width(layer)[:, None, :]
        u = z / W
        return (1.0 + torch.sin(2.0 * self.c * u)) / W


class FixedAlphaSnake(B.SnakeFamily):
    """Check-only independent form of a frozen arm: alpha is held as a tensor computed once."""

    def __init__(self, base, alpha_layers):
        kind = {'SNA': 'snake', 'KKT1': 'kkt1'}[base]
        super().__init__(base, kind, SN_C, SN_BETA, SN_LO, SN_HI, widths=(C.HIDDEN, C.HIDDEN))
        self.fixed = [a.clone() for a in alpha_layers]

    def alpha(self, layer):
        return self.fixed[layer]

    def update(self, z1, z2):
        raise AssertionError('a fixed-alpha activation has no EMA')


def make_c2_act(arm):
    base, mode = C2_ARMS[arm]
    if mode == 'pus':
        return PerUnitScaleSnake()
    act = C.make_act(base, C.HIDDEN, c=SN_C, beta=SN_BETA, lo=SN_LO, hi=SN_HI)
    assert isinstance(act, B.SnakeFamily) and act.beta == SN_BETA
    return act


class SnakeEngine:
    """`cifar5p1_mlp_0920.run()` for one Snake-family arm with explicit state (spec §4.1).

    The step, the capture (3 warmup steps on a side stream, then one captured step, all rolled
    back incl. the activation state), the batch stream, the per-task host rows and the fresh
    control are the host's.  `adapting` decides whether the captured step contains the EMA of V;
    `freeze()` switches it off and recaptures."""

    def __init__(self, arm, seeds, data, graph=True, act=None, out_name=None, adapting=None):
        self.arm = arm
        self.out_name = out_name or arm
        self.base, self.mode = C2_ARMS[arm]
        self.seeds = list(seeds)
        self.R = len(self.seeds)
        self.data = data
        self.device = dev = data.device
        self.act = act if act is not None else make_c2_act(arm)
        init = [q.detach() for s in self.seeds for q in C.init_params(self.base, s, dev, C.HIDDEN)]
        P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
        self.P = [q.requires_grad_(True) for q in P]
        self.P0 = [q.detach().clone() for q in self.P]
        self.act.init_state(self.R, dev, key=f'{self.base}|{self.seeds}|std')
        self.act0 = self.act.state()
        self.m = [torch.zeros_like(q) for q in self.P]
        self.v = [torch.zeros_like(q) for q in self.P]
        self.static_idx = torch.zeros(self.R, BATCH, dtype=torch.long, device=dev)
        self.inv_c1 = torch.zeros((), device=dev)
        self.inv_c2 = torch.zeros((), device=dev)
        self.step_t = torch.zeros((), dtype=torch.long, device=dev)
        self.acc_sum = torch.zeros(self.R, device=dev)
        self.bad_step = torch.full((self.R,), -1, dtype=torch.long, device=dev)
        self.last_hit = torch.zeros(self.R, device=dev)
        self.alive = torch.ones(self.R, dtype=torch.bool, device=dev)
        self.g_batch = {s: H.stream('c51_batch', s) for s in self.seeds}
        self.plans = [C.task_plan(s) for s in self.seeds]
        self.adapting = (self.mode != 'fix0') if adapting is None else bool(adapting)
        self.graph_enabled = graph and dev.type == 'cuda'
        self.tc = 0
        self.task = 0
        self.rows = []
        self.diverged = []
        self.saved_batches = None
        self.V_frozen = None
        self.cg = None
        self.capture()

    # -- the host's step --
    def step(self):
        P, act = self.P, self.act
        xb, yb = self.data.X[self.static_idx], self.data.Y[self.static_idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                reduction='none').view(self.R, BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        self.acc_sum.add_(hit)
        self.last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & bad, self.step_t, self.bad_step))
            self.step_t.add_(1)
            for p, gr, mi, vi in zip(P, grads, self.m, self.v):
                mi.mul_(B1).add_(gr, alpha=1 - B1)
                vi.mul_(B2).addcmul_(gr, gr, value=1 - B2)
                p.sub_(LR * (mi * self.inv_c1) / ((vi * self.inv_c2).sqrt() + EPS))
            if self.adapting:
                act.update(z1.detach(), z2.detach())

    def mutable(self):
        return [*self.P, *self.m, *self.v, self.acc_sum, self.bad_step, self.step_t, self.last_hit]

    def capture(self):
        if not self.graph_enabled:
            self.cg = None
            return
        if self.cg is not None:
            del self.cg
            self.cg = None
            torch.cuda.synchronize(self.device)
        keep = [q.detach().clone() for q in self.mutable()]
        keep_act = self.act.state()
        self.inv_c1.fill_(1.0)
        self.inv_c2.fill_(1.0)
        self.static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.act.begin_step(self.R, BATCH, self.device)
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        self.act.begin_step(self.R, BATCH, self.device)
        with torch.cuda.graph(cg):
            self.step()
        with torch.no_grad():
            for q, old in zip(self.mutable(), keep):
                q.copy_(old)
        self.act.load_state(keep_act)
        self.cg = cg
        del keep

    def freeze(self):
        """fixT1: stop the EMA from here on (the captured step is rebuilt without it)."""
        self.adapting = False
        self.V_frozen = self.act.state()
        self.capture()

    # -- the host's train_task --
    def train_steps(self, batches, tc0):
        tc = tc0
        self.acc_sum.zero_()
        self.bad_step.fill_(-1)
        self.step_t.zero_()
        for j in range(batches.shape[1]):
            self.static_idx.copy_(batches[:, j])
            tc += 1
            self.inv_c1.fill_(1.0 / (1 - B1 ** tc))
            self.inv_c2.fill_(1.0 / (1 - B2 ** tc))
            self.act.begin_step(self.R, BATCH, self.device)
            if self.cg is not None:
                self.cg.replay()
            else:
                self.step()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return tc

    def run_task(self, t):
        """One task of the host loop: batches, training, divergence bookkeeping, host row."""
        data, R = self.data, self.R
        hard = t % 2 == 1
        steps = STEPS
        rows_t = data.task_rows(self.seeds, t)
        batches = torch.stack([C.batch_indices(self.g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(self.seeds)]).to(self.device)
        if t == N_TASKS - 1:
            self.saved_batches = batches.clone()            # the last hard task (fresh control)
        self.tc = self.train_steps(batches, self.tc)
        self.task = t
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in self.P]).all(0)
            newly = self.alive & ((self.bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                self.diverged.append({'diverged': True, 'task': t, 'seed': self.seeds[r], 'arm': self.out_name,
                                      'cond': 'std', 'step': (t - 1) * STEPS + max(int(self.bad_step[r]), 0)})
                self.rows.append({'arm': self.out_name, 'cond': 'std', 'seed': self.seeds[r], 'slot': r, 'lr': LR,
                                  'task': t, 'hard': int(hard), 'acc': float('nan')})
            self.alive &= ~newly
            live = torch.nonzero(self.alive).flatten().tolist()
            Xt = torch.stack([data.X[rows_t[r]] for r in range(R)])
            Yt = torch.stack([data.Y[rows_t[r]] for r in range(R)])
            m = C.evaluate(self.P, Xt, Yt, self.act, live)
            del Xt
            te = data.test_rows(self.seeds, t)
            test_acc = C.accuracy(self.P, torch.stack([data.Xte[q] for q in te]),
                                  torch.stack([data.Yte[q] for q in te]), self.act)
        for r in live:
            self.rows.append({'arm': self.out_name, 'cond': 'std', 'seed': self.seeds[r], 'slot': r, 'lr': LR,
                              'task': t, 'hard': int(hard), 'n_classes': len(self.plans[r][t - 1][1]),
                              'online_acc': float(self.acc_sum[r]) / steps,
                              'train_acc': m[r]['acc'], 'test_acc': float(test_acc[r]), **m[r]})
        self.rows.sort(key=lambda q: (q['slot'], q['task']))

    def fresh_control(self):
        """The host's fresh control on the last hard task; V per spec §4.1."""
        last_hard = N_TASKS - 1
        continual = {int(q['seed']): q['online_acc'] for q in self.rows
                     if q['task'] == last_hard and 'online_acc' in q}
        with torch.no_grad():
            for q, v in zip(self.P, self.P0):
                q.copy_(v)
            for q in (*self.m, *self.v):
                q.zero_()
        self.act.load_state(self.V_frozen if self.mode == 'fixT1' else self.act0)
        alive_f = torch.ones(self.R, dtype=torch.bool, device=self.device)
        self.train_steps(self.saved_batches, 0)
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in self.P]).all(0)
            alive_f &= (self.bad_step < 0) & finite
        out = []
        for r in range(self.R):
            if not bool(alive_f[r]) or self.seeds[r] not in continual:
                continue
            value = float(self.acc_sum[r]) / STEPS
            out.append({'arm': self.out_name, 'cond': 'std', 'seed': self.seeds[r], 'lr': LR, 'task': last_hard,
                        'fresh_online_acc': value, 'continual_online_acc': continual[self.seeds[r]],
                        'fresh_gap': value - continual[self.seeds[r]]})
        return out

    # -- state (checks) --
    def state(self):
        return cpu(dict(seeds=self.seeds, P=self.P, m=self.m, v=self.v, tc=self.tc, task=self.task,
                        g_batch={s: g.get_state() for s, g in self.g_batch.items()}, act=self.act.state(),
                        rows=self.rows))

    def restore(self, st):
        assert list(st['seeds']) == self.seeds
        with torch.no_grad():
            for name in ('P', 'm', 'v'):
                for dst, src in zip(getattr(self, name), st[name]):
                    dst.copy_(src)
        for s, value in st['g_batch'].items():
            self.g_batch[s].set_state(value)
        self.tc = int(st['tc'])
        self.task = int(st['task'])
        self.rows = [dict(r) for r in st['rows']]
        if hasattr(self.act, 'V') and self.act.V is not None and 'V' in st['act']:
            self.act.load_state(st['act'])


def c2_run_arm(arm, seeds, data, graph=True, n_tasks=N_TASKS, fresh=True, out_name=None, on_task=None):
    """One C2 arm, R = len(seeds) runs stacked.  Returns rows, fresh rows, V after every task, state."""
    eng = SnakeEngine(arm, seeds, data, graph=graph, out_name=out_name)
    V_hist = {}
    t0 = time.time()
    for t in range(1, n_tasks + 1):
        eng.run_task(t)
        V_hist[t] = [v.detach().cpu().clone() for v in eng.act.V]
        if eng.mode == 'fixT1' and t == 1:
            eng.freeze()
        if on_task is not None:
            on_task(t, eng)
    fresh_rows = eng.fresh_control() if (fresh and n_tasks == N_TASKS) else []
    res = dict(rows=[dict(r) for r in eng.rows], fresh=fresh_rows, V=V_hist, diverged=list(eng.diverged),
               end=cpu(dict(P=eng.P, m=eng.m, v=eng.v, tc=eng.tc, act=eng.act.state())),
               seconds=time.time() - t0)
    del eng
    free_cuda()
    return res


def csv_text(rows):
    return RS.csv_text(rows)


def c2_record_check(arm, rows, fresh):
    """S2-host(b): per_task.csv vs the committed 0920 record (eff_rank reported), fresh byte-identical."""
    rec = RECORD_0920 / f'{arm}_std_lr0.0001'
    ref = list(csv.DictReader((rec / 'per_task.csv').open()))
    mine = list(csv.DictReader(csv_text(rows).splitlines()))
    assert len(ref) == len(mine) and list(ref[0]) == list(mine[0]), 'record shape'
    mism, eff = [], []
    for a, b in zip(mine, ref):
        for k in b:
            if a[k] != b[k]:
                (eff if k in HOST_FLOAT64_COLS else mism).append(dict(seed=b['seed'], task=b['task'], column=k,
                                                                      run=a[k], record=b[k]))
    fr = csv_text(fresh) == (rec / 'fresh_control.csv').read_text()
    return dict(arm=arm, record=str(rec.relative_to(ROOT)), rows=len(mine), gated_mismatches=len(mism),
                eff_rank_mismatches=len(eff), fresh_byte_identical=bool(fr), examples=(mism + eff)[:20],
                all_pass=bool(not mism and fr))


def c2_run(out, seeds, data, ident, check_mode, log, stop_hook=None):
    stop = out / 'STOP'
    t0 = time.time()
    for arm in C2_ARMS:
        if stop.exists():
            return False
        d = out / 'c2' / arm
        d.mkdir(parents=True, exist_ok=True)
        if done_or_none(d / 'done.json', ident) is not None:
            continue
        res = c2_run_arm(arm, seeds, data)
        assert not res['diverged'], ('DIVERGED', arm, res['diverged'])
        RS.write_csv(d / 'per_task.csv', res['rows'])
        RS.write_csv(d / 'fresh_control.csv', res['fresh'])
        save_npz(d / 'V_history.npz', {f't{t:02d}_l{l + 1}': v[l].numpy() for t, v in res['V'].items() for l in range(2)})
        save_pt(d / 'end.pt', res['end'])
        put(d / 'arm_provenance.json', dict(arm=arm, base=C2_ARMS[arm][0], mode=C2_ARMS[arm][1], seconds=res['seconds'],
                                            seeds=seeds, c=SN_C, beta=SN_BETA, lo=SN_LO, hi=SN_HI, lr=LR,
                                            steps_per_task=STEPS, n_tasks=N_TASKS))
        if arm in ('SNA', 'KKT1') and not check_mode:
            chk = c2_record_check(arm, res['rows'], res['fresh'])
            put(d / 'record_check.json', chk)
            if not chk['all_pass']:
                put(out / 'status.json', dict(stage='CHECK_FAILED', reason=f'C2 {arm} vs 0920 record'))
                raise SystemExit(f'CHECK_FAILED: {arm} does not reproduce the 0920 record')
        files = sorted(p for p in d.iterdir() if p.name != 'done.json' and not p.name.endswith('.tmp'))
        RS.mark_done(d / 'done.json', ident, files)
        put(out / 'status.json', dict(stage='c2_arms', completed_arm=arm, wall_s=time.time() - t0))
        log(f'C2 arm {arm} complete: {res["seconds"]:.1f}s (total {time.time() - t0:.1f}s)')
        del res
        free_cuda()
        if stop_hook is not None:
            stop_hook('c2_arm', arm)
    return True


# --------------------------------------------------------------------------
# the registered run
# --------------------------------------------------------------------------

def run(out, seeds, check_mode=False, log=print, stop_hook=None, data=None, steps=STEPS, which=('c1', 'c2')):
    """C1 then C2.  Returns True when complete, False on STOP."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert steps == STEPS, 'registered budget is 780 updates per task'
    dev = setup()
    data = RS.Data(dev) if data is None else data
    ident = identity(seeds, data, check_mode, steps)
    if not check_mode:
        assert seeds == list(range(10)), 'registered seeds are 0-9'
        chk = json.loads((ROOT / f'results/_checks_{RUN}/checks.json').read_text())
        assert chk['all_pass'] and chk['source_sha256'] == source_hashes(), 'unverified code'
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'analysis', 'specs'],
                                        cwd=ROOT, text=True)
        assert not dirty, 'commit the implementation before the main run'
        saved = out / 'admission_checks.json'
        if saved.exists():
            assert json.loads(saved.read_text()) == chk
        else:
            put(saved, chk)
    put(out / 'input_manifest.json', dict(data_sha256=ident['data_sha256'], input_hash=ident['input_hash'],
                                          labels_hash=ident['labels_hash'], class_plans=ident['class_plans']))
    start = out / 'provenance_start.json'
    if start.exists():
        RS.require_identity(json.loads(start.read_text())['identity'], ident)
    else:
        put(start, dict(identity=ident, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        started_jst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                        pid=os.getpid(), argv=sys.argv, independent_audit=False,
                        mem_available_gb=mem_available_gb(), threads=torch.get_num_threads(),
                        dirty_diff=subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'analysis', 'specs'],
                                                           cwd=ROOT, text=True)))
    t0 = time.time()
    if 'c1' in which and not c1_run(out, seeds, data, ident, check_mode, log, stop_hook):
        return False
    if 'c2' in which and not c2_run(out, seeds, data, ident, check_mode, log, stop_hook):
        return False
    put(out / 'status.json', dict(stage='completed', which=list(which), seeds=len(seeds), wall_s=time.time() - t0))
    put(out / 'provenance_end.json', dict(identity=ident, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                          elapsed_this_invocation_s=time.time() - t0,
                                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                          peak_cuda_bytes=torch.cuda.max_memory_allocated(), independent_audit=False))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(OUT_MAIN))
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--steps', type=int, default=STEPS)
    p.add_argument('--check-mode', action='store_true')
    a = p.parse_args()
    seeds = B.parse_ints(a.seeds)
    if a.check_mode:
        assert Path(a.out).resolve() != OUT_MAIN.resolve(), 'check mode must not write the main output'
    torch.set_num_threads(2)
    with gpu_lock():
        ok = run(Path(a.out), seeds, a.check_mode, steps=a.steps)
    print('completed' if ok else 'STOP acknowledged; boundary state saved.', flush=True)


if __name__ == '__main__':
    main()
