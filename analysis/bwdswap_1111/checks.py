#!/usr/bin/env python3
"""bwdswap_1111 checks and smoke (specs/BRIEF_bwdswap_1111_impl.md, items 3-4).  One part per process; the GPU
parts take the shared GPU lock and run one at a time.  Results: results/bwdswap_1111/smoke/<part>/checks.json,
merged by `collect` into results/bwdswap_1111/checks.json (the main run's admission reads it).

  RM --seed S   smoke on check seed S (2 epochs/task) + E.1 code equality + unit checks with mutations   (CPU)
  RC            smoke on check seeds 100,101 (2 epochs/task) + E.1 code equality + unit checks          (GPU)
  F51           smoke on check seeds 100,101 (E.1's 780 updates) + E.1 code equality + unit checks     (GPU)
  e1-RM --seed S  main seed S, 80 epochs, N and X only: E.1's records bit for bit (CPU, gating)
  e1-RC         seeds 0-9 stacked, natural prefix task 1: this boot vs E.1's saved state (recorded only);
                gating: the main run's E.1-comparison code, self-tested on E.1 against itself
  e1-F51        seeds 0-9 stacked, natural prefix tasks 1-29: this boot vs E.1's saved states (recorded only);
                gating: the same self-test
  timing-RC     seeds 100-109 stacked: ms/update of N, X, B (with and without the monitor) at R=10
  collect       merge, all_pass over the gating parts, source hashes

Unit checks (per setting and direction, on the smoke's own branch states):
  monitor_teeth      the forward monitor wrapped around X sees 0 differing bits at the branch and > 0 after
                     updates (X's forward departs from the natural net; B's must not)
  monitor_invariance B trained with and without the monitor ends in the same state, bit for bit
  reanchor           after B updates: B's gradients = X's with the frozen reference moved to the current weights
                     (bit for bit); B's outputs = natural; X with the branch reference does differ
  derivative_after   after B updates: B's local derivative = native phi'(z + d), bit for bit and float64 bound
  first_update       (GPU, graph path) X and B after one update: same P, m, v bit for bit; N differs
  mutation_field     B built with the sign-flipped field: gradients no longer equal X's (the identity has teeth)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import bwdswap_1111 as bw              # first repo import: fixes the module set source_hashes() reads

import numpy as np
import torch
import torch.nn.functional as F

CI, RE, RM, F51 = bw.CI, bw.RE, bw.RM, bw.F51
SET, KINDS, DIRS = bw.SETTINGS, bw.KINDS, bw.DIRECTIONS
SMOKE = bw.OUT_MAIN / 'smoke'
CHECK_SEEDS = [100, 101]
SHORT = {'RC': 2, 'RM': 2}                     # epochs per task in the smoke
GATING = ('RM_s100', 'RM_s101', 'RC', 'F51', 'e1-RM_s0', 'e1-RM_s1', 'e1-RC', 'e1-F51')   # e1-RC/F51 gate only
RECORDED = ('timing-RC',)                                                                  # their comparison code


class MemSampler(threading.Thread):
    """Peak Rss, Pss and Uss (private pages: what one more such process adds) of this process."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, False
        self.peak = dict(rss_kb=0, pss_kb=0, uss_kb=0)

    def run(self):
        while not self.stop_flag:
            try:
                v = {}
                for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]:
                    k, rest = line.split(':', 1)
                    v[k] = int(rest.split()[0])
                cur = dict(rss_kb=v['Rss'], pss_kb=v['Pss'], uss_kb=v['Private_Clean'] + v['Private_Dirty'])
                self.peak = {k: max(self.peak[k], cur[k]) for k in cur}
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(self.interval)

    def result(self):
        self.stop_flag = True
        return dict(peak_rss_gib=self.peak['rss_kb'] / 2 ** 20, peak_pss_gib=self.peak['pss_kb'] / 2 ** 20,
                    peak_uss_gib=self.peak['uss_kb'] / 2 ** 20, ru_maxrss_gib=bw.peak_rss_bytes() / 2 ** 30)


def gpu_memory():
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                nvidia_smi_process_mib_now=bw.gpu_process_mib())


def save(part, obj):
    obj['all_pass'] = all_true(obj['gating']) if obj.get('gating') else None     # None: recorded-only part
    obj.update(source_sha256=bw.source_hashes(), boot_id=bw.boot_id(), finished_jst=bw.now_jst())
    bw.put(SMOKE / part / 'checks.json', obj)
    print(json.dumps(dict(part=part, all_pass=obj['all_pass'],
                          failed=[k for k, v in flatten(obj.get('gating', {})).items() if v is not True])), flush=True)
    return obj


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f'{prefix}{k}/'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def all_true(d):
    f = flatten(d)
    return bool(f) and all(v is True for v in f.values())


def read_csv(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def compare_rows(mine, e1, keys, skip=()):
    """Every E.1 column (except `skip`) equal for the rows keyed by `keys`."""
    m = {tuple(r[k] for k in keys): r for r in mine}
    bad = []
    for r in e1:
        k = tuple(r[q] for q in keys)
        if k not in m:
            bad.append(dict(key=k, missing=True))
            continue
        cols = [c for c in r if c not in skip and m[k].get(c) != r[c]]
        if cols:
            bad.append(dict(key=k, columns=cols[:10]))
    return dict(equal=not bad and len(e1) > 0, rows=len(e1), mismatches=bad[:20])


# ==========================================================================
# RM (CPU)
# ==========================================================================

def rm_unit_checks(seed, mnist, epochs):
    prows, units, cks, x, info = RM.run_prefix(seed, mnist, epochs, n_tasks=12, save_at=(2, 10))
    out = {}
    for direction in DIRS:
        spec = SET['RM'][direction]
        ck = cks[spec['branch']]
        sh = RM.build_shift(RM.ARMS[spec['X']], cks)
        d2 = sh[bw.LAYER]
        y = RM.RL.task_labels(RM.gen_from(ck['g_lab']))

        def start():
            params = [p.detach().clone().requires_grad_(True) for p in ck['params']]
            if spec['reset']:
                adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
            else:
                adam = ([q.clone() for q in ck['m']], [q.clone() for q in ck['v']], [ck['tc']])
            return params, adam, RM.gen_from(ck['g_batch'])

        r = {}
        # monitor teeth: X wrapped -> 0 at the branch, > 0 after one epoch of X's own training
        mx = bw.Monitor(RM.make_forward(sh), bw.rm_natural, 'cpu')
        params, adam, gb = start()
        ob = torch.arange(RM.N_IMG)
        RM.forward(params, x[ob], ob, mx)
        at_branch = mx.read()['eager_mismatch']
        RM.train_task(params, adam, x, y, gb, 1, mx)
        r['monitor_teeth'] = bool(at_branch == 0 and mx.read()['eager_mismatch'] > 0)
        # monitor invariance: B with and without the monitor, one epoch
        hashes = []
        for wrap in (True, False):
            params, adam, gb = start()
            f = bw.rm_bwd_forward(sh)
            f = bw.Monitor(f, bw.rm_natural, 'cpu') if wrap else f
            RM.train_task(params, adam, x, y, gb, 1, f)
            hashes.append(RM.SH.state_sha256(params, adam, RM.ACT))
            if wrap:
                mon = f.read()
                r['monitor_counts_B_one_epoch'] = bool(mon['eager_calls'] == RM.SPE and mon['eager_mismatch'] == 0)
                Pk = [p.detach().clone() for p in params]
        r['monitor_invariance'] = hashes[0] == hashes[1]
        # re-anchoring identity and the derivative after B's updates (at P_k)
        ob = torch.randperm(RM.N_IMG, generator=torch.Generator().manual_seed(seed))[:RM.BATCH]
        fwds = dict(B=bw.rm_bwd_forward(sh), Xk=RM.make_forward({'p0': Pk, bw.LAYER: d2}), X0=RM.make_forward(sh))
        res = {}
        for k, f in fwds.items():
            P = [p.clone().requires_grad_(True) for p in Pk]
            vals = RM.forward(P, x[ob], ob, f)
            res[k] = dict(vals=[v.detach() for v in vals], g=bw.local_derivative(vals),
                          grads=torch.autograd.grad(F.cross_entropy(vals[4], y[ob]), P))
        nat = bw.rm_natural(Pk, x[ob])
        r['reanchor_grads'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['grads'], res['Xk']['grads']))
        r['reanchor_outputs_natural'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['vals'], nat)) and \
            all(bw.bits_equal(a, b) for a, b in zip(res['Xk']['vals'], nat))
        r['branch_anchor_X_departs'] = not bw.bits_equal(res['X0']['vals'][4], nat[4])
        z2 = res['B']['vals'][2]
        arg = z2 + d2[ob]
        ref = arg.double().clamp(max=0).exp()
        dv = bw.derivative_report(res['B']['g'], bw.rm_gtrain(arg), bw.rm_gtrain(z2), ref, bw.elu_expm1_bound(ref))
        r['derivative_after'] = bool(dv['bit_equal_native_shifted'] and dv['within_float64_bound'])
        r['derivative_after_detail'] = dv
        # mutation: the sign-flipped field breaks the X = B gradient identity at the branch
        ob = torch.randperm(RM.N_IMG, generator=RM.gen_from(ck['g_batch']))[:RM.BATCH]
        g = {}
        for k, f in dict(X=RM.make_forward(sh), Bflip=bw.rm_bwd_forward({**sh, bw.LAYER: -d2})).items():
            P = [p.detach().clone().requires_grad_(True) for p in ck['params']]
            g[k] = torch.autograd.grad(F.cross_entropy(RM.forward(P, x[ob], ob, f)[4], y[ob]), P)
        r['mutation_field_rejected'] = not all(bw.bits_equal(a, b) for a, b in zip(g['X'], g['Bflip']))
        out[direction] = r
    return out


def part_rm(seed):
    sampler = MemSampler()
    sampler.start()
    epochs = SHORT['RM']
    out = SMOKE / f'RM_s{seed}'
    mnist = bw.rm_setup()
    t0 = time.time()
    runtime = bw.run_rm_seed(seed, out / 'run', epochs, check_mode=True, mnist=mnist)
    t_run = time.time() - t0
    names = [SET['RM'][d][k] for d in DIRS for k in 'NX']
    t1 = time.time()
    RM.run_seed(seed, out / 'e1', mnist, epochs, arms=names, prefix_tasks=12, save_ckpt=False, progress=False)
    t_e1 = time.time() - t1
    eq_arms = compare_rows(read_csv(out / 'run/arms.csv'), read_csv(out / 'e1/arms.csv'), ('arm', 'k'), skip=('sec',))
    eq_prefix = compare_rows(read_csv(out / 'run/prefix.csv'), read_csv(out / 'e1/prefix.csv'), ('task',), skip=('sec',))
    unit = rm_unit_checks(seed, mnist, epochs)
    prov = json.loads((out / 'run/provenance.json').read_text())
    gating = dict(runtime={k: v for k, v in runtime.items()},
                  e1_code_equal=dict(arms=eq_arms['equal'], prefix=eq_prefix['equal']),
                  unit={d: {k: v for k, v in r.items() if not k.endswith('_detail')} for d, r in unit.items()})
    return save(f'RM_s{seed}', dict(
        gating=gating, e1_code_detail=dict(arms=eq_arms, prefix=eq_prefix), unit_detail=unit,
        budget=dict(epochs_per_task=epochs, seed=seed),
        timing=dict(seconds_run=t_run, seconds_e1_code=t_e1, seconds_prefix=prov['seconds_prefix'],
                    arm_seconds=prov['arm_seconds']),
        memory=sampler.result()))


def part_e1_rm(seed):
    sampler = MemSampler()
    sampler.start()
    out = SMOKE / f'e1-RM_s{seed}'
    mnist = bw.rm_setup()
    t0 = time.time()
    runtime = bw.run_rm_seed(seed, out, bw.BUDGET['RM'], check_mode=True, mnist=mnist, kinds=('N', 'X'))
    e1 = json.loads((out / 'e1_check.json').read_text())
    prov = json.loads((out / 'provenance.json').read_text())
    return save(f'e1-RM_s{seed}', dict(
        gating=dict(runtime=runtime, e1_records_bit_equal=bool(e1['all_equal'])),
        e1_detail=e1, timing=dict(seconds_total=time.time() - t0, seconds_prefix=prov['seconds_prefix'],
                                  arm_seconds=prov['arm_seconds']),
        memory=sampler.result()))


# ==========================================================================
# RC (GPU)
# ==========================================================================

def rc_first_update(eng, task):
    """Engine.train_task's opening, up to and including its first update (graph replay)."""
    seeds = eng.seeds
    eng.task, eng.epoch, eng.in_task = task, 0, True
    lab = {s: CI.E.RC.task_labels(eng.g_lab[s]) for s in seeds}
    eng.Y.copy_(torch.stack([lab[s] for s in seeds]).to(eng.device))
    eng.acc_sum.zero_(); eng.ce_sum.zero_(); eng.step_t.zero_(); eng.bad_step.fill_(-1)
    order = torch.stack([torch.randperm(CI.E.N_IMAGES, generator=eng.g_batch[s]) for s in seeds]).to(eng.device)
    eng.one_update(order[:, :CI.E.BATCH])
    torch.cuda.synchronize()
    return [q.detach().clone() for q in (*eng.P, *eng.m, *eng.v)]


def rc_unit_checks(states, X):
    dev = X.device
    out = {}
    for direction in DIRS:
        spec = SET['RC'][direction]
        branch, layer, source, reset = RE.ARMS[spec['X']]
        st = states[branch]
        d = RE.field_from_states(st, states[source], layer, X)
        P0 = [p.to(dev) for p in st['P']]
        T = spec['task']
        R = len(st['seeds'])
        ar = torch.arange(R, device=dev)[:, None]
        r = {}

        def engine(shift, state=st, graph=True):
            e = CI.Engine(st['seeds'], X, state=state, shift=shift, graph=graph)
            if reset and state is st:
                e.reset_adam()
            return e
        # monitor teeth on X
        mx = bw.Monitor(CI.Shift(P0, layer, d), bw.rc_natural, dev)
        e = engine(mx)
        labels, order = bw.rc_task_draws(st, X)
        ids = order[:, :CI.E.BATCH]
        with torch.no_grad():
            e.forward(X[ar, ids], ids)
        at_branch = mx.read()
        e.train_task(T, 1)
        after = mx.read()
        r['monitor_teeth'] = bool(at_branch['eager_mismatch'] == 0 and after['graph_mismatch'] > 0)
        del e
        # monitor invariance (graph path, one epoch), and B's state after it
        ends = []
        for wrap in (True, False):
            sh = bw.BwdShift(P0, layer, d)
            sh = bw.Monitor(sh, bw.rc_natural, dev) if wrap else sh
            e = engine(sh)
            e.train_task(T, 1)
            if wrap:
                mon = sh.read()
                r['monitor_counts_B_one_epoch'] = bool(mon['graph_calls'] == CI.E.STEPS_PER_EPOCH and mon['graph_mismatch'] == 0)
            ends.append(e.state())
            del e
        r['monitor_invariance'] = CI.tree_hash(CI.core_state(ends[0])) == CI.tree_hash(CI.core_state(ends[1]))
        stk = dict(ends[1], shift=None)            # restore() would rebuild a forward-swap Shift from a saved shift
        Pk = [p.to(dev) for p in stk['P']]
        engs = dict(B=CI.Engine(stk['seeds'], X, state=stk, shift=bw.BwdShift(P0, layer, d), graph=False),
                    Xk=CI.Engine(stk['seeds'], X, state=stk, shift=CI.Shift(Pk, layer, d), graph=False),
                    X0=CI.Engine(stk['seeds'], X, state=stk, shift=CI.Shift(P0, layer, d), graph=False))
        ids = torch.stack([torch.randperm(CI.E.N_IMAGES, generator=bw.gen_clone(stk['g_batch'][s]))
                           for s in stk['seeds']]).to(dev)[:, :CI.E.BATCH]
        Y = stk['Y'].to(dev)
        res = {}
        for k, e in engs.items():
            vals = e.forward(X[ar, ids], ids)
            loss = F.cross_entropy(vals[-1].reshape(-1, 10), Y[ar, ids].reshape(-1), reduction='none').view(R, -1).mean(1)
            res[k] = dict(vals=[v.detach() for v in vals], g=bw.local_derivative(vals),
                          grads=torch.autograd.grad(loss.sum(), e.P))
        with torch.no_grad():
            nat = bw.rc_natural(Pk, X[ar, ids])
        r['reanchor_grads'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['grads'], res['Xk']['grads']))
        r['reanchor_outputs_natural'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['vals'], nat)) and \
            all(bw.bits_equal(a, b) for a, b in zip(res['Xk']['vals'], nat))
        r['branch_anchor_X_departs'] = not bw.bits_equal(res['X0']['vals'][4], nat[4])
        z2 = res['B']['vals'][2]
        arg = z2 + d[ar, ids]
        ref = arg.double().clamp(max=0).exp()
        dv = bw.derivative_report(res['B']['g'], CI.gtrain(arg), CI.gtrain(z2), ref, bw.elu_cuda_bound(ref))
        r['derivative_after'] = bool(dv['bit_equal_native_shifted'] and dv['within_float64_bound'])
        r['derivative_after_detail'] = dv
        del engs, res
        # first update through the captured graph: X and B identical, N not
        first = {}
        for k in KINDS:
            e = bw.rc_engine(k, spec, states, X, graph=True, monitor=False)
            first[k] = rc_first_update(e, T)
            del e
        flip = engine(bw.BwdShift(P0, layer, -d))
        first['Bflip'] = rc_first_update(flip, T)
        del flip
        r['first_update_X_B'] = all(bw.bits_equal(a, b) for a, b in zip(first['X'], first['B']))
        r['first_update_N_differs'] = not all(bw.bits_equal(a, b) for a, b in zip(first['N'], first['B']))
        r['mutation_field_rejected'] = not all(bw.bits_equal(a, b) for a, b in zip(first['X'], first['Bflip']))
        out[direction] = r
        torch.cuda.empty_cache()
    return out


def part_rc():
    sampler = MemSampler()
    sampler.start()
    seeds, epochs = CHECK_SEEDS, SHORT['RC']
    out = SMOKE / 'RC'
    with bw.gpu_guard():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        runtime = bw.run_rc(out / 'run', seeds, epochs, check_mode=True)
        t_run = time.time() - t0
        mem_run = gpu_memory()
        t1 = time.time()
        RE.run(out / 'e1', list(seeds), epochs, allow_checks=True)
        t_e1 = time.time() - t1
        eq = {}
        for direction in DIRS:
            for k in 'NX':
                name = SET['RC'][direction][k]
                mine = json.loads((out / 'run/arms' / name / 'rows.json').read_text())
                e1 = json.loads((out / 'e1/arms' / name / 'rows.json').read_text())
                same_rows = len(mine) == len(e1) and all(all(m[c] == o[c] for c in o if c != 'arm') for m, o in zip(mine, e1))
                same_end = CI.tree_hash(CI.core_state(CI.load_pt(out / 'run/raw/arms' / name / 'end.pt'))) == \
                    CI.tree_hash(CI.core_state(CI.load_pt(out / 'e1/arms' / name / 'end.pt')))
                eq[name] = bool(same_rows and same_end)
        mine_p = json.loads((out / 'run/prefix/rows.json').read_text())
        e1_p = json.loads((out / 'e1/prefix/rows.json').read_text())
        eq['prefix_rows'] = len(mine_p) == len(e1_p) and all(m == o for m, o in zip(mine_p, e1_p))
        for t in bw.E1['RC']['states']:
            eq[f'prefix_t{t:02d}'] = CI.tree_hash(CI.core_state(CI.load_pt(out / 'run/raw' / f'prefix_t{t:02d}.pt'))) == \
                CI.tree_hash(CI.core_state(CI.load_pt(out / 'e1/prefix' / f't{t:02d}.pt')))
        dev = CI.setup()
        X, _ = CI.make_inputs(seeds, dev)
        states = {t: CI.load_pt(out / 'run/raw' / f'prefix_t{t:02d}.pt') for t in bw.E1['RC']['states']}
        unit = rc_unit_checks(states, X)
        timing = {name: json.loads((out / 'run/arms' / name / 'timing.json').read_text())
                  for d in DIRS for name in SET['RC'][d].values() if isinstance(name, str) and (out / 'run/arms' / name).exists()}
        return save('RC', dict(
            gating=dict(runtime=runtime, e1_code_equal=eq,
                        unit={d: {k: v for k, v in r.items() if not k.endswith('_detail')} for d, r in unit.items()}),
            unit_detail=unit, budget=dict(seeds=seeds, epochs_per_task=epochs),
            timing=dict(seconds_run=t_run, seconds_e1_code=t_e1, arms=timing,
                        prefix=json.loads((out / 'run/prefix/timing.json').read_text())),
            memory=dict(**sampler.result(), gpu_run=mem_run, gpu_all=gpu_memory())))


def selftest_e1_comparison(setting):
    """The main run's E.1-comparison code (it only runs with the main seeds) on E.1 against itself: everything
    equal; and with one seed's online accuracy nudged, that arm (only) is reported unequal."""
    run = bw.E1[setting]['run']
    res = ROOT / 'results' / run
    fn = bw.rc_e1_comparison if setting == 'RC' else bw.f51_e1_comparison
    states = {t: bw.e1_backup_state(run, f'prefix/t{t:02d}.pt') for t in bw.E1[setting]['states']}
    prefix_rows = json.loads((res / 'prefix/rows.json').read_text())
    names = [SET[setting][d][k] for d in DIRS for k in 'NX']
    out = {}
    for case in ('same', 'nudged'):
        tmp = SMOKE / f'e1-{setting}' / 'selftest' / case
        for name in names:
            rows = json.loads((res / 'arms' / name / 'rows.json').read_text())
            if case == 'nudged' and name == names[0]:
                rows[0]['online_acc'] += 1e-9
            bw.put(tmp / 'arms' / name / 'rows.json', rows)
        r = fn(tmp, states, prefix_rows)
        pr = r['prefix_rows'].get('online_acc_equal', r['prefix_rows'].get('equal_except_eff_rank'))
        out[case] = dict(states=all(r['states'].values()), prefix_rows=bool(pr),
                         arms={k: v['all_equal'] for k, v in r['arms'].items()})
    same, nud = out['same'], out['nudged']
    ok = same['states'] and same['prefix_rows'] and all(same['arms'].values()) and \
        nud['arms'][names[0]] is False and all(nud['arms'][n] for n in names[1:])
    return bool(ok), out


def part_e1_rc():
    """This boot vs E.1 (recorded): the stacked natural prefix's first task on seeds 0-9 at 400 epochs.
    Gating: only the self-test of the main run's E.1-comparison code."""
    sampler = MemSampler()
    sampler.start()
    with bw.gpu_guard():
        dev = CI.setup()
        X, _ = CI.make_inputs(bw.MAIN_SEEDS, dev)
        torch.cuda.reset_peak_memory_stats()
        eng = CI.Engine(bw.MAIN_SEEDS, X)
        t0 = time.time()
        rows, _ = eng.train_task(1, bw.BUDGET['RC'])
        sec = time.time() - t0
        st = eng.state()
        old = bw.e1_backup_state('resp_cifar_ee_0920', 'prefix/t01.pt')
        old_rows = {r['seed']: r for r in json.loads((ROOT / 'results/resp_cifar_ee_0920/prefix/rows.json').read_text())
                    if r['task'] == 1}
        per = [dict(seed=r['seed'], online_new=r['online_acc'], online_e1=old_rows[r['seed']]['online_acc'],
                    row_equal=all(r[k] == old_rows[r['seed']][k] for k in old_rows[r['seed']])) for r in rows]
        updates = bw.BUDGET['RC'] * CI.E.STEPS_PER_EPOCH
        del eng
        torch.cuda.empty_cache()
        st_ok, st_detail = selftest_e1_comparison('RC')
        return save('e1-RC', dict(
            gating=dict(e1_comparison_code=st_ok), selftest_detail=st_detail, recorded=dict(
                boot_id=bw.boot_id(),
                state_t01_equal=CI.tree_hash(CI.core_state(st)) == CI.tree_hash(CI.core_state(old)),
                rows_equal=all(p['row_equal'] for p in per), per_seed=per),
            timing=dict(seconds_task1=sec, updates=updates, ms_per_update_natural_R10=1e3 * sec / updates),
            memory=dict(**sampler.result(), gpu=gpu_memory())))


def part_timing_rc(epochs=8):
    """ms/update at the main stack size (R=10, check seeds 100-109): N, X, B with and without the monitor,
    from a 1-task, 2-epoch prefix (branch t1, donor t0); no diagnostics inside the timed task."""
    sampler = MemSampler()
    sampler.start()
    seeds = list(range(100, 110))
    with bw.gpu_guard():
        dev = CI.setup()
        X, _ = CI.make_inputs(seeds, dev)
        torch.cuda.reset_peak_memory_stats()
        eng = CI.Engine(seeds, X)
        st0 = eng.state()
        eng.train_task(1, 2)
        st1 = eng.state()
        del eng
        d = RE.field_from_states(st1, st0, bw.LAYER, X)
        P1 = [p.to(dev) for p in st1['P']]
        res = {}
        for name, mk in (('N', lambda: None), ('X', lambda: CI.Shift(P1, bw.LAYER, d)),
                         ('B_monitor', lambda: bw.Monitor(bw.BwdShift(P1, bw.LAYER, d), bw.rc_natural, dev)),
                         ('B_plain', lambda: bw.BwdShift(P1, bw.LAYER, d))):
            e = CI.Engine(seeds, X, state=st1, shift=mk(), graph=True)
            torch.cuda.synchronize()
            t0 = time.time()
            e.train_task(2, epochs)
            sec = time.time() - t0
            res[name] = 1e3 * sec / (epochs * CI.E.STEPS_PER_EPOCH)
            del e
            torch.cuda.empty_cache()
        return save('timing-RC', dict(gating={}, recorded=dict(ms_per_update_R10=res, updates_each=epochs * CI.E.STEPS_PER_EPOCH),
                                      memory=dict(**sampler.result(), gpu=gpu_memory())))


# ==========================================================================
# F51 (GPU)
# ==========================================================================

def f51_unit_checks(states, data):
    dev = data.device
    out = {}
    for direction in DIRS:
        spec = SET['F51'][direction]
        branch, layer, donor, reset = F51.ARMS[spec['X']]
        st = states[branch]
        T = branch + 1
        rows = data.task_rows(st['seeds'], T)
        table, _, _ = F51.build_field(st, states[donor], layer, rows, data)
        P0 = [p.to(dev) for p in st['P']]
        R = len(st['seeds'])
        r = {}

        def engine(field, state=st, graph=True):
            e = F51.Engine(st['seeds'], data, state=state, field=field, graph=graph)
            if reset and state is st:
                e.reset_adam()
            return e
        # monitor teeth on X (100 updates of X's own training)
        mx = bw.Monitor(F51.Field(P0, layer, table), bw.f51_natural, dev)
        e = engine(mx)
        rows_T, batches = e.task_batches(T)
        with torch.no_grad():
            e.forward(data.X[batches[:, 0]], batches[:, 0])
        at_branch = mx.read()
        e.train_steps(batches[:, :100])
        r['monitor_teeth'] = bool(at_branch['eager_mismatch'] == 0 and mx.read()['graph_mismatch'] > 0)
        del e
        # monitor invariance, 200 updates
        ends = []
        for wrap in (True, False):
            f = bw.BwdField(P0, layer, table)
            f = bw.Monitor(f, bw.f51_natural, dev) if wrap else f
            e = engine(f)
            rows_T, batches = e.task_batches(T)
            e.train_steps(batches[:, :200])
            if wrap:
                mon = f.read()
                r['monitor_counts_B_200'] = bool(mon['graph_calls'] == 200 and mon['graph_mismatch'] == 0)
            ends.append(e.state())
            del e
        r['monitor_invariance'] = CI.tree_hash(F51.core_state(ends[0])) == CI.tree_hash(F51.core_state(ends[1]))
        stk = ends[1]
        Pk = [p.to(dev) for p in stk['P']]
        engs = dict(B=engine(bw.BwdField(P0, layer, table), stk, graph=False),
                    Xk=engine(F51.Field(Pk, layer, table), stk, graph=False),
                    X0=engine(F51.Field(P0, layer, table), stk, graph=False))
        ids = batches[:, 200]
        res = {}
        for k, e in engs.items():
            vals = e.forward(data.X[ids], ids)
            loss = F.cross_entropy(vals[-1].reshape(-1, F51.NCLS), data.Y[ids].reshape(-1), reduction='none').view(R, -1).mean(1)
            res[k] = dict(vals=[v.detach() for v in vals], g=bw.local_derivative(vals),
                          grads=torch.autograd.grad(loss.sum(), e.P))
        with torch.no_grad():
            nat = bw.f51_natural(Pk, data.X[ids])
        r['reanchor_grads'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['grads'], res['Xk']['grads']))
        r['reanchor_outputs_natural'] = all(bw.bits_equal(a, b) for a, b in zip(res['B']['vals'], nat)) and \
            all(bw.bits_equal(a, b) for a, b in zip(res['Xk']['vals'], nat))
        r['branch_anchor_X_departs'] = not bw.bits_equal(res['X0']['vals'][4], nat[4])
        z2 = res['B']['vals'][2]
        arg = z2 + table[torch.arange(R, device=dev)[:, None], ids]
        ref = (arg.double() >= 0).double()
        dv = bw.derivative_report(res['B']['g'], F51.gtrain(arg), F51.gtrain(z2), ref, torch.zeros_like(ref))
        r['derivative_after'] = bool(dv['bit_equal_native_shifted'] and dv['within_float64_bound'])
        r['derivative_after_detail'] = dv
        del engs, res
        first = {}
        for k in KINDS:
            e = bw.f51_engine(k, spec, states, data, graph=True)
            rows_T, batches = e.task_batches(T)
            e.train_steps(batches[:, :1])
            first[k] = [q.detach().clone() for q in (*e.P, *e.m, *e.v)]
            del e
        table_flip = -table
        e = engine(bw.BwdField(P0, layer, table_flip))
        rows_T, batches = e.task_batches(T)
        e.train_steps(batches[:, :1])
        first['Bflip'] = [q.detach().clone() for q in (*e.P, *e.m, *e.v)]
        del e, table_flip
        r['first_update_X_B'] = all(bw.bits_equal(a, b) for a, b in zip(first['X'], first['B']))
        r['first_update_N_differs'] = not all(bw.bits_equal(a, b) for a, b in zip(first['N'], first['B']))
        r['mutation_field_rejected'] = not all(bw.bits_equal(a, b) for a, b in zip(first['X'], first['Bflip']))
        out[direction] = r
        del table
        torch.cuda.empty_cache()
    return out


def part_f51():
    sampler = MemSampler()
    sampler.start()
    seeds = CHECK_SEEDS
    out = SMOKE / 'F51'
    with bw.gpu_guard():
        dev = F51.setup()
        data = F51.Data(dev)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        runtime = bw.run_f51(out / 'run', seeds, check_mode=True, data=data)
        t_run = time.time() - t0
        mem_run = gpu_memory()
        t1 = time.time()
        F51.run(out / 'e1', list(seeds), check_mode=True, data=data)
        t_e1 = time.time() - t1
        eq = {}
        for direction in DIRS:
            for k in 'NX':
                name = SET['F51'][direction][k]
                mine = json.loads((out / 'run/arms' / name / 'rows.json').read_text())
                e1 = json.loads((out / 'e1/arms' / name / 'rows.json').read_text())
                same_rows = len(mine) == len(e1) and all(all(m[c] == o[c] for c in o) for m, o in zip(mine, e1))
                same_end = CI.tree_hash(F51.core_state(CI.load_pt(out / 'run/raw/arms' / name / 'end.pt'))) == \
                    CI.tree_hash(F51.core_state(CI.load_pt(out / 'e1/arms' / name / 'end.pt')))
                eq[name] = bool(same_rows and same_end)
        mine_p = json.loads((out / 'run/prefix/rows.json').read_text())
        e1_p = json.loads((out / 'e1/prefix/rows.json').read_text())
        eq['prefix_rows'] = len(mine_p) == len(e1_p) and all(m == o for m, o in zip(mine_p, e1_p))
        for t in bw.E1['F51']['states']:
            a = CI.load_pt(out / 'run/raw' / f'prefix_t{t:02d}.pt')
            b = CI.load_pt(out / 'e1/prefix' / f't{t:02d}.pt')
            eq[f'prefix_t{t:02d}'] = CI.tree_hash(F51.core_state(a)) == CI.tree_hash(F51.core_state(b))
        states = {t: CI.load_pt(out / 'run/raw' / f'prefix_t{t:02d}.pt') for t in bw.E1['F51']['states']}
        unit = f51_unit_checks(states, data)
        timing = {name: json.loads((out / 'run/arms' / name / 'timing.json').read_text())
                  for d in DIRS for name in (SET['F51'][d][k] for k in KINDS)}
        return save('F51', dict(
            gating=dict(runtime=runtime, e1_code_equal=eq,
                        unit={d: {k: v for k, v in r.items() if not k.endswith('_detail')} for d, r in unit.items()}),
            unit_detail=unit, budget=dict(seeds=seeds, updates_per_task=F51.STEPS),
            timing=dict(seconds_run=t_run, seconds_e1_code=t_e1, arms=timing,
                        prefix=json.loads((out / 'run/prefix/timing.json').read_text())),
            memory=dict(**sampler.result(), gpu_run=mem_run, gpu_all=gpu_memory())))


def part_e1_f51():
    """This boot vs E.1 (recorded): the stacked natural prefix, tasks 1-29 on seeds 0-9."""
    sampler = MemSampler()
    sampler.start()
    with bw.gpu_guard():
        dev = F51.setup()
        data = F51.Data(dev)
        torch.cuda.reset_peak_memory_stats()
        states, rows, sec = bw.f51_prefix(bw.MAIN_SEEDS, data, print)
        rec = {}
        for t, st in states.items():
            old = bw.e1_backup_state('resp_cifar5p1_1007', f'prefix/t{t:02d}.pt')
            rec[f'state_t{t:02d}_equal'] = CI.tree_hash(F51.core_state(st)) == CI.tree_hash(F51.core_state(old))
        old_rows = {(r['seed'], r['task']): r for r in json.loads((ROOT / 'results/resp_cifar5p1_1007/prefix/rows.json').read_text())}
        diff = [dict(seed=r['seed'], task=r['task'], cols=[k for k in r if k not in F51.HOST_FLOAT64_COLS and r[k] != old_rows[(r['seed'], r['task'])][k]])
                for r in rows]
        diff = [q for q in diff if q['cols']]
        rec.update(boot_id=bw.boot_id(), rows_equal_except_eff_rank=not diff, n_rows_differing=len(diff),
                   first_differing=diff[:5])
        st_ok, st_detail = selftest_e1_comparison('F51')
        return save('e1-F51', dict(gating=dict(e1_comparison_code=st_ok), selftest_detail=st_detail, recorded=rec,
                                   timing=dict(seconds_prefix_29=sec, ms_per_update_with_host_rows=1e3 * sec / (29 * F51.STEPS)),
                                   memory=dict(**sampler.result(), gpu=gpu_memory())))


# ==========================================================================

def collect():
    parts = {}
    for p in GATING + RECORDED:
        f = SMOKE / p / 'checks.json'
        parts[p] = json.loads(f.read_text()) if f.exists() else None
    missing = [p for p in GATING if parts[p] is None]
    current = bw.source_hashes()
    stale = [p for p in GATING + RECORDED if parts[p] is not None and parts[p].get('source_sha256') != current]
    ok = not missing and not [p for p in stale if p in GATING] and all(parts[p]['all_pass'] for p in GATING)
    obj = dict(run=bw.RUN, all_pass=bool(ok), missing=missing, stale_parts=stale, gating_parts=list(GATING),
               recorded_parts=list(RECORDED),
               part_pass={p: (None if parts[p] is None else parts[p]['all_pass']) for p in GATING},
               source_sha256=current, git_hash=bw.git('rev-parse', 'HEAD'),
               boot_id=bw.boot_id(), collected_jst=bw.now_jst(),
               recorded={p: (None if parts[p] is None else parts[p].get('recorded')) for p in RECORDED},
               timing={p: (None if parts[p] is None else parts[p].get('timing')) for p in GATING + RECORDED},
               memory={p: (None if parts[p] is None else parts[p].get('memory')) for p in GATING + RECORDED})
    bw.put(bw.CHECKS, obj)
    print(json.dumps(dict(all_pass=obj['all_pass'], part_pass=obj['part_pass'], missing=missing, stale=stale)), flush=True)
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('part', choices=('RM', 'RC', 'F51', 'e1-RM', 'e1-RC', 'e1-F51', 'timing-RC', 'collect'))
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.part == 'RM':
        part_rm(a.seed)
    elif a.part == 'e1-RM':
        part_e1_rm(a.seed)
    elif a.part == 'RC':
        part_rc()
    elif a.part == 'F51':
        part_f51()
    elif a.part == 'e1-RC':
        part_e1_rc()
    elif a.part == 'e1-F51':
        part_e1_f51()
    elif a.part == 'timing-RC':
        part_timing_rc()
    else:
        collect()


if __name__ == '__main__':
    main()
