#!/usr/bin/env python3
"""controls_cifar5p1_1008 admission checks (spec §7).  Check seeds 100-109 and synthetic inputs only.

Every required check must PASS on the real implementation and every listed mutant must be
REJECTED by the same comparator.  Fail closed; every attempt is kept under
results/_checks_controls_cifar5p1_1008/attempt_<ns>/ (bulky .pt dropped at the end).
"""
from __future__ import annotations

import contextlib
import csv
import gc
import inspect
import json
import math
import resource
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F

import src.controls_cifar5p1_1008 as M
from src.cifar_interventions_0920 import tree_hash, cpu
from analysis.cifar_ledger_0920.replay import put, sha
from analysis.resp_cifar_ee_0920.stats import interval, t_quantile
from analysis.controls_cifar5p1_1008 import verdict as V

RS, C, B, H = M.RS, M.C, M.B, M.H
RUN = M.RUN
OUT = ROOT / f'results/_checks_{RUN}'
SEEDS = list(range(100, 110))
TH, TC = M.TH, M.TC
REQUIRED = ['S-src', 'S1-host', 'S1-prefix', 'S1-wiring', 'S1-branch', 'S1-field', 'S1-isolation', 'S1-graph',
            'S2-host', 'S2-freeze', 'S2-pus', 'S2-graph', 'S-verdict', 'S-resume', 'S-cost/CLI']
MUTANTS = ['prefix_wrong_save',                                                   # S1-host
           'mv_lost', 'tc_reset', 'rng_not_restored',                             # S1-prefix
           'donor_t03', 'adam_reset', 'layer1',                                   # S1-wiring
           'drop_anchor', 'alias_P0',                                             # S1-branch
           'fresh_own_init', 'perm_identity', 'perm_same_all_seeds', 'perm_receiver_side',
           'rand_unshuffled', 'rand_within_unit', 'match_zero', 'match_per_unit', 'match_K_from_fresh',
           'sign_flip', 'other_seed', 'stored_table_edit',                        # S1-field
           'shallow_copy',                                                        # S1-isolation
           'lr', 'time', 'beta', 'fix0_adapting', 'warmup_V',                     # S2-host
           'freeze_t2', 'freeze_t0', 'freeze_reset_V', 'resume_ema_t3',           # S2-freeze
           'pus_no_division', 'pus_divide_V', 'pus_no_clip',                      # S2-pus
           'unpaired', 'no_bonferroni', 'missing_seed', 'missing_arm', 'nonfinite',  # S-verdict
           'fake_done', 'bad_hash', 'different_identity',                         # S-resume
           'wrong_arm', 'wrong_budget']                                           # S-cost/CLI
RESP_CODE = ['src/resp_cifar5p1_1007.py', 'src/cifar5p1_mlp_0920.py', 'src/rlcifar_mlp_battle_0918.py',
             'src/pmnist_0905.py', 'src/pmnist_rlcifar_0907.py', 'src/cifar_interventions_0920.py',
             'analysis/cifar_ledger_0920/replay.py', 'analysis/resp_cifar_ee_0920/stats.py',
             'analysis/resp_cifar5p1_1007/checks.py', 'analysis/resp_cifar5p1_1007/report.py',
             'analysis/resp_cifar5p1_1007/launch.sh']


def reject(fn):
    """The comparator must refuse the mutant (an assertion-type failure), never accept it."""
    try:
        fn()
    except (AssertionError, ValueError, KeyError, SystemExit, FileNotFoundError, RuntimeError):
        return True
    raise AssertionError('mutant was accepted')


def same(a, b):
    assert tree_hash(a) == tree_hash(b), 'not bit identical'


def equal(a, b, what='values'):
    assert a == b, f'{what} differ'


def replace_method(owner, name, old, new, extra=None):
    """Re-exec the source of owner.name with one exact substitution (asserted present once)."""
    fn = getattr(owner, name)
    src = textwrap.dedent(inspect.getsource(fn))
    assert src.count(old) == 1, (name, old, src.count(old))
    g = dict(fn.__globals__)
    if extra:
        g.update(extra)
    ns = {}
    exec(src.replace(old, new), g, ns)
    return ns[name]


@contextlib.contextmanager
def patched(obj, attr, value):
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def c1_fingerprint(res):
    """The numbers a C1 arm produces (names excluded so resp's R_ch and our R_h compare)."""
    rows = [(r['seed'], r['online_acc'], r['online_ce'], r['memo_acc'], r['memo_ce'], r['finite'],
             r['label_hash'], r['batch_hash']) for r in res['rows']]
    hist = [(u, rr, {k: v for k, v in sorted(uu.items())}) for u, rr, uu in res['history']]
    return tree_hash(dict(rows=rows, state=RS.core_state(res['end_state']), hist=hist,
                          field=None if res['field'] is None else res['field']['d']))


def text_wo_arm(text):
    """per_task / fresh CSV text with the arm column blanked (fix0 vs the host's beta=0 'SNA')."""
    rows = list(csv.reader(text.splitlines()))
    i = rows[0].index('arm')
    return '\n'.join(','.join(c if j != i or k == 0 else 'ARM' for j, c in enumerate(r)) for k, r in enumerate(rows))


# --------------------------------------------------------------------------
# S1-field: the independent reference (numpy, CPU float64 sort / permute / shuffle)
# --------------------------------------------------------------------------

def reference_field(name, states, data):
    dev = data.device
    seeds = list(states[TC]['seeds'])
    R = len(seeds)
    rows = data.task_rows(seeds, TC + 1)
    ids = torch.stack(rows).to(dev)
    X = data.X[ids]
    zb = RS.eval_z([q.to(dev) for q in states[TC]['P']], X, 2).cpu().numpy()
    zs = RS.eval_z([q.to(dev) for q in states[TH]['P']], X, 2).cpu().numpy()
    N, n = zb.shape[1], zb.shape[2]
    if name == 'R_fresh':
        own = [q for s in seeds for q in C.init_params('R', s, dev, C.HIDDEN)]
        donor = [q for s in seeds for q in C.init_params('R', 1000 + s, dev, C.HIDDEN)]
        P = [torch.stack(donor[i::6]).contiguous() for i in range(6)]
        assert tree_hash([torch.stack(own[i::6]) for i in range(6)]) != tree_hash(P), 'fresh donor equals own init'
        T = RS.eval_z(P, X, 2).cpu().numpy()
    elif name == 'R_h':
        T = zs
    elif name == 'R_perm':
        T = np.empty_like(zs)
        for r, s in enumerate(seeds):
            g = H.stream('ctrl1008_perm', s)
            while True:
                p = torch.randperm(n, generator=g).numpy()
                if (p != np.arange(n)).all():
                    break
            for i in range(n):
                T[r, :, i] = zs[r, :, p[i]]
    elif name == 'R_rand':
        T = np.empty_like(zs)
        for r, s in enumerate(seeds):
            order = torch.randperm(N * n, generator=H.stream('ctrl1008_rand', s)).numpy()
            T[r] = zs[r].ravel()[order].reshape(N, n)
    del X
    if name == 'R_match':
        d_h = (zs.astype(np.float64) - zb.astype(np.float64)).astype(np.float32)
        K = ((zb + d_h) >= 0).reshape(R, -1).sum(1)
        d = np.empty_like(zb)
        for r in range(R):
            v = np.sort(zb[r].astype(np.float64).ravel())[::-1]
            d[r] = np.float32(-(v[K[r] - 1] + v[K[r]]) / 2.0)
        target = zb.astype(np.float64) + d.astype(np.float64)
    else:
        d = (T.astype(np.float64) - zb.astype(np.float64)).astype(np.float32)
        target = T.astype(np.float64)
    return target, d, ids.cpu().numpy()


def field_matches(built, ref):
    target64, d, ids, _ = built
    rt, rd, rids = ref
    assert np.array_equal(ids.cpu().numpy(), rids), 'field image ids'
    assert d.dtype == torch.float32 and np.array_equal(d.cpu().numpy().view(np.uint32), rd.view(np.uint32)), 'field d'
    assert np.array_equal(target64.cpu().numpy().view(np.uint64), rt.view(np.uint64)), 'field target'


# --------------------------------------------------------------------------
# S-verdict (synthetic)
# --------------------------------------------------------------------------

def verdict_check(mut):
    for p in (.975, .9875):
        q = t_quantile(p, 9)
        xx = np.linspace(0, q, 20001)
        dens = math.gamma(5) / (math.sqrt(9 * math.pi) * math.gamma(4.5)) * (1 + xx * xx / 9) ** -5
        area = (xx[1] - xx[0]) / 3 * (dens[0] + dens[-1] + 4 * dens[1:-1:2].sum() + 2 * dens[2:-1:2].sum())
        assert abs(.5 + area - p) < 2e-12
    noise = np.linspace(-.01, .01, 10)
    ok = dict(complete=True, checks_pass=True, reproduced=True)
    # C1 labels over all sign pairs
    b1 = {a: np.full(10, .6) + noise for a in V.C1_ARMS}
    b1['N_c'] = np.full(10, .4) + noise
    want1 = {('+', '+'): 'LEARNED_FIELD_MATTERS', ('+', '0'): 'PARTIAL', ('+', '-'): 'PARTIAL',
             ('0', '+'): 'PARTIAL', ('-', '+'): 'PARTIAL', ('0', '0'): 'OPEN_COUNT_SUFFICES',
             ('-', '0'): 'OPEN_COUNT_SUFFICES', ('0', '-'): 'OPEN_COUNT_SUFFICES', ('-', '-'): 'OPEN_COUNT_SUFFICES'}
    shift = {'+': -.1, '0': 0.0, '-': +.1}
    for (sf, sm), lab in want1.items():
        E = {k: v.copy() for k, v in b1.items()}
        E['R_fresh'] = E['R_h'] + shift[sf]
        E['R_match'] = E['R_h'] + shift[sm]
        out = V.c1_verdict(E, ok)
        assert out['label'] == lab, (sf, sm, out['label'])
        assert out['primary']['D_fresh']['sign'] == sf and out['primary']['D_match']['sign'] == sm
        if '-' in (sf, sm):
            assert out['annotations'], 'missing CONTROL_EXCEEDS'
    E = {k: v.copy() for k, v in b1.items()}
    E['R_h'] = E['N_c'].copy()
    assert V.c1_verdict(E, ok)['label'] == 'NOT_REPRODUCED'
    assert V.c1_verdict(b1, dict(ok, reproduced=False))['label'] == 'CHECK_FAILED'
    assert V.c1_verdict(b1, dict(ok, complete=False))['label'] == 'INCOMPLETE'
    # C2 labels
    b2 = {a: np.full(10, .66) + noise for a in V.C2_ARMS}
    b2.update(R=np.full(10, .42) + noise, L2I=np.full(10, .665) + noise)
    want2 = {('+', '+'): 'ADAPTIVITY_MATTERS', ('+', '0'): 'PARTIAL', ('0', '+'): 'PARTIAL', ('-', '+'): 'PARTIAL',
             ('0', '0'): 'FIXED_ALPHA_SUFFICES', ('-', '-'): 'FIXED_ALPHA_SUFFICES', ('0', '-'): 'FIXED_ALPHA_SUFFICES'}
    for (sa, sk), lab in want2.items():
        E = {k: v.copy() for k, v in b2.items()}
        E['SNA_fix0'] = E['SNA'] + shift[sa]
        E['KKT1_fix0'] = E['KKT1'] + shift[sk]
        assert V.c2_verdict(E, ok)['label'] == lab, (sa, sk)
    # SD = 0: degenerate point interval, sign by the point
    z = interval(np.zeros(10))
    assert z['sign'] == '0' and z['degenerate_sd'] and z['low'] == z['high'] == 0
    E = {k: v.copy() for k, v in b2.items()}
    E['SNA_fix0'] = E['SNA'] - .03                       # constant difference: SD 0, point +.03
    out = V.c2_verdict(E, ok)
    assert out['primary']['A_SNA']['degenerate_sd'] and out['primary']['A_SNA']['sign'] == '+'
    # endpoint exactly 0 counts as 0
    assert V.label_pair('0', '0', ('a', 'b'), ('X', 'Y', 'Z'))[0] == 'Z'
    # Bonferroni: a fixture whose 95% interval excludes 0 but whose 97.5% interval does not
    spread = np.linspace(-.05, .05, 10)
    found = None
    for m in np.linspace(.0, .05, 501):
        x = m + spread
        if interval(x, .95)['sign'] == '+' and interval(x, .975)['sign'] == '0':
            found = m
            break
    assert found is not None
    E = {k: v.copy() for k, v in b2.items()}
    E['SNA_fix0'] = E['SNA'] - (found + spread)
    out = V.c2_verdict(E, ok)
    assert out['primary']['A_SNA']['sign'] == '0' and out['primary_95']['A_SNA']['sign'] == '+'
    mut['no_bonferroni'] = True
    # unpaired: a paired difference that is clearly + while an unpaired comparison would not be
    pair = np.linspace(.2, .8, 10)
    assert interval((pair + .01) - pair)['sign'] == '+'
    assert interval((pair + .01) - pair[::-1])['sign'] == '0'
    E = {k: v.copy() for k, v in b1.items()}
    E['R_h'], E['R_fresh'] = pair + .01, pair
    assert V.c1_verdict(E | {'N_c': pair - .3}, ok)['primary']['D_fresh']['sign'] == '+'
    mut['unpaired'] = True
    for name, edit in [('missing_seed', lambda E: E.__setitem__('R_fresh', E['R_fresh'][:-1])),
                       ('nonfinite', lambda E: E['R_match'].__setitem__(0, np.nan)),
                       ('missing_arm', lambda E: E.pop('R_match'))]:
        E = {k: v.copy() for k, v in b1.items()}
        edit(E)
        reject(lambda: V.c1_verdict(E, ok))
        mut[name] = True
    # prediction scoring: the multi-class Brier and the binary Brier by hand
    rows = V.score(dict(label='PARTIAL'), dict(label='ADAPTIVITY_MATTERS'))
    r = [x for x in rows if x['item'] == 'C1_label' and x['who'] == 'claude'][0]
    assert abs(r['brier'] - ((.35) ** 2 + (.45 - 1) ** 2 + (.20) ** 2)) < 1e-15 and r['hit'] is True
    r = [x for x in rows if x['item'] == 'C2_label' and x['who'] == 'parent'][0]
    assert abs(r['brier'] - ((.55 - 1) ** 2 + .30 ** 2 + .15 ** 2)) < 1e-15
    return dict(labels_c1=len(want1), labels_c2=len(want2), quantiles='independent Simpson integration',
                bonferroni_fixture_mean=float(found))


# --------------------------------------------------------------------------
# S2-pus (synthetic)
# --------------------------------------------------------------------------

def pus_check(mut, dev):
    gen = torch.Generator().manual_seed(1008)
    R, Bn, n = 2, 64, 7
    z = (torch.randn((R, Bn, n), generator=gen) * 6).to(dev)
    # V spans both clips: sqrt(V) < c/hi = 0.2 and sqrt(V) > c/lo = 120 on some units
    V = torch.tensor([[0.01, 0.5, 1.0, 9.0, 30.0, 20000.0, 2.0], [0.02, 4.0, 0.3, 100.0, 15000.0, 1.5, 0.04]]).to(dev)
    act = M.PerUnitScaleSnake()
    act.init_state(R, dev, 'x')
    sna = B.SnakeFamily('SNA', 'snake', M.SN_C, M.SN_BETA, M.SN_LO, M.SN_HI, widths=(n, n))
    sna.init_state(R, dev, 'x')
    for a in (act, sna):
        a.V = [V.clone(), V.clone()]
    c, lo, hi = M.SN_C, M.SN_LO, M.SN_HI
    W64 = (torch.sqrt(V.double())).clamp(c / hi, c / lo)
    assert bool(((V.double().sqrt() < c / hi) | (V.double().sqrt() > c / lo)).any()), 'fixture misses the clip'
    u64 = z.double() / W64[:, None, :]
    phi64 = u64 + torch.sin(c * u64) ** 2 / c
    dphi64 = (1 + torch.sin(2 * c * u64)) / W64[:, None, :]
    eps32 = 2.0 ** -24
    gam = lambda k: k * eps32 / (1 - k * eps32)
    scale = (u64.abs() + 1 / c + 1.0)                      # sum of |terms| in phi (per element)

    def phi_err(fn):
        zz = z.clone().requires_grad_(True)
        out = fn(zz)
        g = torch.autograd.grad(out.sum(), zz)[0]
        out = out.detach()
        return (out.double() - phi64).abs(), (g.double() - dphi64).abs(), out
    e_phi, e_g, out = phi_err(lambda q: act.phi(q, 0))
    bound_phi = gam(16) * scale * (1 + u64.abs())
    bound_g = gam(16) * (2 + u64.abs() * 2 * c) / W64[:, None, :]
    assert bool((e_phi <= bound_phi).all()), float((e_phi / bound_phi).max())
    assert bool((e_g <= bound_g).all()), float((e_g / bound_g).max())
    assert bool(((act.dphi(z, 0).double() - dphi64).abs() <= bound_g).all())
    # phi_pus * W equals phi_SNA within rounding (the arm is SNA divided by the unit width)
    s_out = sna.phi(z, 0).double()
    rel = ((out.double() * W64[:, None, :]) - s_out).abs()
    bound_rel = gam(24) * (s_out.abs() + z.double().abs() + W64[:, None, :] / c)
    assert bool((rel <= bound_rel).all())
    mutants = {
        'pus_no_division': lambda q: q + torch.sin(c * q) ** 2 / c,
        'pus_divide_V': lambda q: (lambda u: u + torch.sin(c * u) ** 2 / c)(q / act.V[0][:, None, :]),
        'pus_no_clip': lambda q: (lambda u: u + torch.sin(c * u) ** 2 / c)(q / act.V[0].sqrt()[:, None, :])}
    for name, fn in mutants.items():
        e, _, _ = phi_err(fn)
        assert bool((e > bound_phi).any()), name
        mut[name] = True
    return dict(max_phi_ratio=float((e_phi / bound_phi).max()), max_grad_ratio=float((e_g / bound_g).max()),
                clip_in_fixture=True, relation='phi_pus * W = phi_SNA within gamma_24')


# --------------------------------------------------------------------------
# S2-freeze: validator
# --------------------------------------------------------------------------

def freeze_validator(res_fixT1, res_adapt_rows_task1, n_tasks):
    rows1 = [r for r in res_fixT1['rows'] if r['task'] == 1]
    assert M.csv_text(rows1).replace('SNA_fixT1', 'X').replace('KKT1_fixT1', 'X') == \
        M.csv_text(res_adapt_rows_task1).replace('SNA', 'X').replace('KKT1', 'X'), 'task 1 differs from the adaptive arm'
    V1 = res_fixT1['V'][1]
    for t in range(2, n_tasks + 1):
        for l in range(2):
            assert torch.equal(res_fixT1['V'][t][l], V1[l]), ('V moved after the freeze', t, l)
    stats = {(r['seed'], k): r[k] for r in res_fixT1['rows'] if r['task'] == 2 for k in r if k.startswith('alpha_')}
    for r in res_fixT1['rows']:
        if r['task'] >= 2:
            for k in r:
                if k.startswith('alpha_'):
                    assert r[k] == stats[(r['seed'], k)], ('alpha stats moved', r['task'], k)
    return True


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------

def suite():
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = OUT / f'attempt_{time.time_ns()}'
    attempt.mkdir()
    evidence, mut = {}, {}
    started = time.time()
    tested_source = M.source_hashes()

    def record(name, value):
        evidence[name] = dict(all_pass=True, **value)
        put(attempt / 'progress.json', dict(evidence=evidence, mutants=mut))
        print(f'{name} PASS ({time.time() - started:.0f}s)', flush=True)

    try:
        # ---------------- S-src ----------------
        adm = json.loads((ROOT / 'results/resp_cifar5p1_1007/admission_checks.json').read_text())
        assert adm['all_pass']
        cur = {p: sha(ROOT / p) for p in RESP_CODE}
        assert all(adm['source_sha256'][p] == h for p, h in cur.items()), 'verified resp code changed'
        record('S-src', dict(files=cur, reference='results/resp_cifar5p1_1007/admission_checks.json'))

        dev = M.setup()
        record('S-verdict', verdict_check(mut))
        record('S2-pus', pus_check(mut, dev))
        data = RS.Data(dev)

        # ---------------- CLI run (all arms) ----------------
        cli = attempt / 'cli'
        argv = sys.argv
        sys.argv = [str(ROOT / f'src/{RUN}.py'), '--check-mode', '--out', str(cli), '--seeds', '100-109']
        t_cli = time.time()
        try:
            with (attempt / 'cli.log').open('w') as log, contextlib.redirect_stdout(log), \
                    patched(M, 'gpu_lock', lambda **k: contextlib.nullcontext()), \
                    patched(RS, 'Data', lambda dev_, path=RS.DATA: data):
                M.main()
        finally:
            sys.argv = argv
        cli_seconds = time.time() - t_cli
        assert json.loads((cli / 'status.json').read_text())['stage'] == 'completed'
        assert {p.name for p in (cli / 'c1' / 'arms').iterdir()} == set(M.C1_ARMS)
        assert {p.name for p in (cli / 'c2').iterdir()} == set(M.C2_ARMS)
        free()
        states = {t: RS.load_pt(cli / 'c1' / 'prefix' / f't{t:02d}.pt') for t in M.C1_SAVE}
        prefix_rows = json.loads((cli / 'c1' / 'prefix' / 'rows.json').read_text())

        v1s, v2s, _ = V.run_verdict(cli, attempt / 'verdict_smoke', seeds=SEEDS)
        assert v1s['label'] in V.C1_LABELS and v2s['label'] in V.C2_LABELS, (v1s.get('label'), v2s.get('label'))
        evidence_verdict_smoke = dict(c1=v1s['label'], c2=v2s['label'], written=str((attempt / 'verdict_smoke').relative_to(ROOT)))

        # ---------------- S1-host ----------------
        hostdir = attempt / 'host_R'
        with (attempt / 'host_R.log').open('w') as log, contextlib.redirect_stdout(log):
            C.run('R', SEEDS, 'std', 30, dev, hostdir, lr=M.LR, cifar=data.cifar)
        free()
        host_lines = (hostdir / 'per_task.csv').read_text().splitlines()
        ti = host_lines[0].split(',').index('task')
        host29 = [host_lines[0]] + [l for l in host_lines[1:] if int(l.split(',')[ti]) <= 29]
        assert (cli / 'c1/prefix/per_task.csv').read_text().splitlines() == host29, \
            'C1 prefix rows differ from the unmodified host (tasks 1-29, all columns, byte level)'
        assert (cli / 'c1/prefix/fresh_control.csv').read_text() == (hostdir / 'fresh_control.csv').read_text(), 'fresh'
        bad = dict(states)
        bad[TC] = states[TC + 1]                                       # t29 stored under the key t28
        reject(lambda: RS.prefix_check('N_c', RS.run_arm('N_c', bad, data, preflight_on=False), bad, prefix_rows))
        mut['prefix_wrong_save'] = True
        free()
        record('S1-host', dict(prefix_rows_equal_host=True, fresh_byte_identical=True, tasks=29, seeds=SEEDS,
                               columns='all (eff_rank included)', reference='unmodified cifar5p1_mlp_0920.run(R)'))

        # ---------------- S1-prefix ----------------
        pc = json.loads((cli / 'c1/arms/N_c/prefix_check.json').read_text())
        assert pc['all_pass']
        for mname, edit in [('mv_lost', lambda st: [q.zero_() for q in st['m']]),
                            ('tc_reset', lambda st: st.__setitem__('tc', 0)),
                            ('rng_not_restored', lambda st: st.__setitem__(
                                'g_batch', {s: H.stream('c51_batch', s).get_state() for s in SEEDS}))]:
            badst = {t: cpu(s) for t, s in states.items()}
            edit(badst[TC])
            res = RS.run_arm('N_c', badst, data, preflight_on=False)
            reject(lambda: RS.prefix_check('N_c', res, states, prefix_rows))
            mut[mname] = True
            del res
            free()
        record('S1-prefix', dict(cli=pc))

        # ---------------- S1-wiring ----------------
        ref = RS.run_arm('R_ch', states, data, preflight_on=False)
        fp_ref = c1_fingerprint(ref)
        cli_h = json.loads((cli / 'c1/arms/R_h/rows.json').read_text())
        assert RS.jnorm(ref['rows']) == cli_h, 'CLI R_h is not resp run_arm(R_ch)'
        del ref
        free()
        mine = M.c1_run_field_arm('R_h', states, data, preflight_on=True)
        assert c1_fingerprint(mine) == fp_ref, 'field path R_h differs from resp run_arm(R_ch)'
        wiring_pf = mine['preflight']
        del mine
        free()
        bad = dict(states)
        bad[TH] = states[TH + 1]
        fp = c1_fingerprint(M.c1_run_field_arm('R_h', bad, data, preflight_on=False))
        reject(lambda: equal(fp, fp_ref, 'fingerprint'))
        mut['donor_t03'] = True
        free()
        orig_engine = M.c1_engine

        def reset_engine(*a, **k):
            eng, built = orig_engine(*a, **k)
            eng.reset_adam()
            return eng, built
        with patched(M, 'c1_engine', reset_engine):
            fp = c1_fingerprint(M.c1_run_field_arm('R_h', states, data, preflight_on=False))
        reject(lambda: equal(fp, fp_ref, 'fingerprint'))
        mut['adam_reset'] = True
        free()

        def layer1_engine(*a, **k):
            eng, built = orig_engine(*a, **k)
            eng.field.layer = 1
            return eng, built
        with patched(M, 'c1_engine', layer1_engine):
            reject(lambda: M.c1_run_field_arm('R_h', states, data, preflight_on=True))
        mut['layer1'] = True
        free()
        record('S1-wiring', dict(field_path_equals_resp_run_arm=True, preflight=wiring_pf))

        # ---------------- S1-field (reference) and S1-branch ----------------
        field_ev = {}
        built = {}
        for name in ('R_h', 'R_fresh', 'R_match', 'R_perm', 'R_rand'):
            built[name] = M.c1_field(name, states, data)
            field_matches(built[name], reference_field(name, states, data))
            field_ev[name] = {k: (v if not torch.is_tensor(v) else 'tensor') for k, v in built[name][3].items()}
            free()
        # open counts: R_perm / R_rand / R_match open exactly as many targets as R_h
        kh = (built['R_h'][0] >= 0).flatten(1).sum(1)
        for name in ('R_perm', 'R_rand'):
            assert torch.equal((built[name][0] >= 0).flatten(1).sum(1), kh), name
        K = torch.tensor(built['R_match'][3]['K'], device=dev)
        zb = RS.eval_z([q.to(dev) for q in states[TC]['P']], data.X[built['R_h'][2]], 2)
        assert torch.equal(((zb + built['R_h'][1]) >= 0).flatten(1).sum(1), K), 'K is not R_h open count'
        field_ev['R_match_K_minus_achieved'] = (K - torch.tensor(built['R_match'][3]['K_achieved'], device=dev)).cpu().tolist()
        del zb
        pf = {n: json.loads((cli / 'c1/arms' / n / 'preflight.json').read_text()) for n in ('R_fresh', 'R_match', 'R_perm', 'R_rand')}
        assert all(p['all_pass'] and p['batches'] == 1 + M.STEPS + 79 for p in pf.values())
        free()
        # construction mutants: built through the production code, compared with the reference
        refs = {n: reference_field(n, states, data) for n in ('R_fresh', 'R_match', 'R_perm', 'R_rand')}
        free()

        def refuse_build(name, *patches):
            with contextlib.ExitStack() as es:
                for obj, attr, val in patches:
                    es.enter_context(patched(obj, attr, val))
                b = M.c1_field(name, states, data)
            reject(lambda: field_matches(b, refs[name]))
            free()
        refuse_build('R_fresh', (M, 'FRESH_OFFSET', 0))
        mut['fresh_own_init'] = True
        refuse_build('R_perm', (M, 'derangement', lambda seed, n=100: torch.arange(n)))
        mut['perm_identity'] = True
        orig_der = M.derangement
        refuse_build('R_perm', (M, 'derangement', lambda seed, n=100: orig_der(SEEDS[0], n)))
        mut['perm_same_all_seeds'] = True
        fn = replace_method(M, 'c1_field', "T = torch.stack([zs[r][:, perms[r].to(dev)] for r in range(R)])",
                            "T = zs; zb = torch.stack([zb[r][:, perms[r].to(dev)] for r in range(R)])")
        refuse_build('R_perm', (M, 'c1_field', fn))
        mut['perm_receiver_side'] = True
        refuse_build('R_rand', (M, 'shuffle_order', lambda seed, n: torch.arange(n)))
        mut['rand_unshuffled'] = True
        fn = replace_method(M, 'c1_field', "T = torch.stack([zs[r].reshape(-1)[orders[r].to(dev)].reshape(N, -1) for r in range(R)])",
                            "T = torch.stack([zs[r][torch.randperm(N, generator=H.stream('x', seeds[r])).to(dev)] for r in range(R)])")
        refuse_build('R_rand', (M, 'c1_field', fn))
        mut['rand_within_unit'] = True
        refuse_build('R_match', (M, 'match_shift', lambda zb_row, K: np.float32(0.0)))
        mut['match_zero'] = True
        fn = replace_method(M, 'c1_field', "d = c[:, None, None].expand_as(zb).contiguous()",
                            "d = (c[:, None, None] + 0.01 * torch.arange(zb.shape[-1], device=dev)[None, None, :]).expand_as(zb).contiguous()")
        refuse_build('R_match', (M, 'c1_field', fn))
        mut['match_per_unit'] = True
        fn = replace_method(M, 'c1_field', "d_h = (zs.double() - zb.double()).float()",
                            "d_h = (RS.eval_z(fresh_donor_params(seeds, dev), data.X[ids], 2).double() - zb.double()).float()")
        refuse_build('R_match', (M, 'c1_field', fn))
        mut['match_K_from_fresh'] = True
        fn = replace_method(M, 'c1_field', "d = (T.double() - zb.double()).float()", "d = (zb.double() - T.double()).float()")
        refuse_build('R_perm', (M, 'c1_field', fn))
        mut['sign_flip'] = True
        orig_field = M.c1_field

        def rolled(name, st, dd):
            t64, d, ids, info = orig_field(name, st, dd)
            return t64, d.roll(1, 0), ids, info
        refuse_build('R_fresh', (M, 'c1_field', rolled))
        mut['other_seed'] = True
        # stored-table edits and forward faults through the production preflight
        eng, b = M.c1_engine('R_rand', states, data, graph=False)
        _, batches = eng.task_batches(TC + 1)
        eng.field.d[eng.ar, b[2]] = eng.field.d[eng.ar, b[2]].roll(1, 1)
        reject(lambda: M.c1_preflight('R_rand', eng, states, batches, b))
        mut['stored_table_edit'] = True
        del eng
        free()
        for name, old, new in [('drop_anchor', 'a2 = AnchorAdd.apply(a02, phi(z2 + d) - phi(z02 + d))', 'a2 = phi(z2 + d)')]:
            fn = replace_method(RS.Field, '__call__', old, new)
            with patched(RS.Field, '__call__', fn):
                eng, b = M.c1_engine('R_fresh', states, data, graph=False)
                _, batches = eng.task_batches(TC + 1)
                reject(lambda: M.c1_preflight('R_fresh', eng, states, batches, b))
            mut[name] = True
            del eng
            free()
        eng, b = M.c1_engine('R_match', states, data, graph=False)
        _, batches = eng.task_batches(TC + 1)
        eng.field.P0 = eng.P
        reject(lambda: M.c1_preflight('R_match', eng, states, batches, b))
        mut['alias_P0'] = True
        del eng, built, refs
        free()
        record('S1-field', dict(arms={n: {k: v for k, v in p.items() if k not in ('branch_bit_exact',)} for n, p in pf.items()},
                                construction=field_ev, reference='numpy CPU float64 (sort, permute, shuffle, midpoint)'))
        record('S1-branch', dict(arms={n: dict(batches=p['batches'], branch_bit_exact=p['branch_bit_exact']) for n, p in pf.items()},
                                 tested='all task images at once, all 780 training batches, row-order chunks of 32'))

        # ---------------- S1-isolation: reversed arm order ----------------
        before = tree_hash(states)
        for name in reversed(M.C1_ARMS):
            res = M.c1_run_arm(name, states, data, preflight_on=False)      # preflight never changes training
            rows_cli = json.loads((cli / 'c1/arms' / name / 'rows.json').read_text())
            assert RS.jnorm(res['rows']) == rows_cli, ('reversed order rows', name)
            same(RS.core_state(res['end_state']), RS.core_state(RS.load_pt(cli / 'c1/arms' / name / 'end.pt')))
            del res
            free()
        same(before, tree_hash(states))
        keep_w = states[TC]['P'][0].clone()
        shallow = dict(states)
        shallow[TC]['P'][0].add_(1.0)
        reject(lambda: same(before, tree_hash(states)))
        states[TC]['P'][0].copy_(keep_w)
        same(before, tree_hash(states))
        mut['shallow_copy'] = True
        record('S1-isolation', dict(reversed_order_bit_identical=True, states_unchanged=True, arms=list(M.C1_ARMS)))

        # ---------------- S1-graph ----------------
        for name in ('R_fresh', 'R_match'):
            ra = M.c1_run_field_arm(name, states, data, graph=False, preflight_on=False)
            rb = M.c1_run_field_arm(name, states, data, graph=True, preflight_on=False)
            assert c1_fingerprint(ra) == c1_fingerprint(rb), ('eager/graph', name)
            del ra, rb
            free()
        record('S1-graph', dict(arms=['R_fresh', 'R_match'], eager_graph_bit_identical=True))
        del states
        free()

        # ---------------- S2-host ----------------
        host2 = {}
        for name, arm, beta in (('SNA', 'SNA', .01), ('KKT1', 'KKT1', .01), ('SNA_fix0', 'SNA', 0.0), ('KKT1_fix0', 'KKT1', 0.0)):
            d = attempt / f'host_{name}'
            with (attempt / f'host_{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                C.run(arm, SEEDS, 'std', 30, dev, d, lr=M.LR, beta=beta, cifar=data.cifar)
            free()
            a = (cli / 'c2' / name / 'per_task.csv').read_text()
            b = (d / 'per_task.csv').read_text()
            fa = (cli / 'c2' / name / 'fresh_control.csv').read_text()
            fb = (d / 'fresh_control.csv').read_text()
            if name in ('SNA', 'KKT1'):
                assert a == b and fa == fb, ('host', name)
            else:
                assert text_wo_arm(a) == text_wo_arm(b) and text_wo_arm(fa) == text_wo_arm(fb), ('host beta=0', name)
            host2[name] = dict(per_task_identical=True, fresh_identical=True)
        ref_sna = list((attempt / 'host_SNA' / 'per_task.csv').read_text().splitlines())
        ref_fix = list((attempt / 'host_SNA_fix0' / 'per_task.csv').read_text().splitlines())

        def first_tasks(res):
            return [l for l in M.csv_text(res['rows']).splitlines()]

        def host_rows_tasks(lines, n):
            hdr = lines[0].split(',')
            ti = hdr.index('task')
            return [lines[0]] + [l for l in lines[1:] if int(l.split(',')[ti]) <= n]

        def refuse_c2(arm, ref_lines, *patches, adapting=None, n=2):
            with contextlib.ExitStack() as es:
                for obj, attr, val in patches:
                    es.enter_context(patched(obj, attr, val))
                if adapting is None:
                    res = M.c2_run_arm(arm, SEEDS, data, n_tasks=n, fresh=False, out_name='SNA')
                else:
                    eng = M.SnakeEngine(arm, SEEDS, data, out_name='SNA', adapting=adapting)
                    for t in range(1, n + 1):
                        eng.run_task(t)
                    res = dict(rows=eng.rows)
                    del eng
            got = first_tasks(res)
            want = host_rows_tasks(ref_lines, n)
            reject(lambda: equal(text_wo_arm('\n'.join(got)), text_wo_arm('\n'.join(want)), 'rows'))
            free()
        # sanity: the unmutated 2-task SNA run equals the host's first two tasks
        ok2 = first_tasks(M.c2_run_arm('SNA', SEEDS, data, n_tasks=2, fresh=False))
        assert ok2 == host_rows_tasks(ref_sna, 2)
        free()
        refuse_c2('SNA', ref_sna, (M.SnakeEngine, 'step', replace_method(M.SnakeEngine, 'step', 'p.sub_(LR *', 'p.sub_(2 * LR *')))
        mut['lr'] = True
        refuse_c2('SNA', ref_sna, (M.SnakeEngine, 'train_steps', replace_method(M.SnakeEngine, 'train_steps', 'tc += 1', 'tc += 100')))
        mut['time'] = True
        orig_make = M.make_c2_act

        def beta2(arm):
            a = orig_make(arm)
            a.beta = 0.02
            return a
        refuse_c2('SNA', ref_sna, (M, 'make_c2_act', beta2))
        mut['beta'] = True
        refuse_c2('SNA_fix0', ref_fix, adapting=True)
        mut['fix0_adapting'] = True
        refuse_c2('SNA', ref_sna, (M.SnakeEngine, 'capture', replace_method(M.SnakeEngine, 'capture', 'self.act.load_state(keep_act)', 'pass')))
        mut['warmup_V'] = True
        record('S2-host', dict(arms=host2, reference='unmodified cifar5p1_mlp_0920.run (beta=0.01; beta=0 for fix0)',
                               seeds=SEEDS, tasks=30, columns='all (arm name blanked for fix0)'))

        # ---------------- S2-freeze ----------------
        fz = {}
        for arm, base in (('SNA_fixT1', 'SNA'), ('KKT1_fixT1', 'KKT1')):
            rows_cli = list(csv.DictReader((cli / 'c2' / arm / 'per_task.csv').open()))
            base_cli = list(csv.DictReader((cli / 'c2' / base / 'per_task.csv').open()))
            t1a = [{k: v for k, v in r.items() if k != 'arm'} for r in rows_cli if r['task'] == '1']
            t1b = [{k: v for k, v in r.items() if k != 'arm'} for r in base_cli if r['task'] == '1']
            assert t1a == t1b, ('fixT1 task 1 differs from the adaptive arm', arm)
            Vh = dict(np.load(cli / 'c2' / arm / 'V_history.npz'))
            for t in range(2, 31):
                for l in (1, 2):
                    assert np.array_equal(Vh[f't{t:02d}_l{l}'], Vh[f't01_l{l}']), ('V moved', arm, t)
            fz[arm] = dict(task1_identical_to_adaptive=True, V_constant_tasks_2_30=True)
        # independent fixed-alpha implementation: tasks 2-4 from the end-of-task-1 state
        eng = M.SnakeEngine('SNA_fixT1', SEEDS, data)
        eng.run_task(1)
        eng.freeze()
        st1 = eng.state()
        for t in (2, 3, 4):
            eng.run_task(t)
        want = dict(rows=[r for r in eng.rows if r['task'] in (2, 3, 4)], P=cpu(eng.P), m=cpu(eng.m), v=cpu(eng.v), tc=eng.tc)
        alpha_T1 = [eng.act.alpha(l).clone() for l in (0, 1)]
        del eng
        free()
        fa = M.FixedAlphaSnake('SNA', alpha_T1)
        eng = M.SnakeEngine('SNA_fixT1', SEEDS, data, graph=False, act=fa, adapting=False)
        eng.restore(st1)
        for t in (2, 3, 4):
            eng.run_task(t)
        got = dict(rows=[r for r in eng.rows if r['task'] in (2, 3, 4)], P=cpu(eng.P), m=cpu(eng.m), v=cpu(eng.v), tc=eng.tc)
        assert M.csv_text(got['rows']) == M.csv_text(want['rows']), 'fixed-alpha rows'
        same(got['P'], want['P'])
        same([got['m'], got['v'], got['tc']], [want['m'], want['v'], want['tc']])
        del eng
        free()
        adapt_t1 = M.c2_run_arm('SNA', SEEDS, data, n_tasks=1, fresh=False)['rows']
        free()
        good = M.c2_run_arm('SNA_fixT1', SEEDS, data, n_tasks=3, fresh=False)
        freeze_validator(good, adapt_t1, 3)
        free()

        def variant(kind):
            def run_variant(arm, seeds, data_, graph=True, n_tasks=3, fresh=False, out_name=None, on_task=None):
                eng = M.SnakeEngine(arm, seeds, data_, graph=graph, out_name=out_name,
                                    adapting=False if kind == 'freeze_t0' else None)
                Vh = {}
                for t in range(1, n_tasks + 1):
                    if kind == 'resume_ema_t3' and t == 3:
                        eng.adapting = True
                        eng.capture()
                    eng.run_task(t)
                    Vh[t] = [v.detach().cpu().clone() for v in eng.act.V]
                    if kind == 'resume_ema_t3' and t == 3:
                        eng.adapting = False
                        eng.capture()
                    if kind != 'freeze_t0' and t == (2 if kind == 'freeze_t2' else 1) and eng.adapting:
                        eng.freeze()
                        if kind == 'freeze_reset_V':
                            eng.act.load_state(eng.act0)
                res = dict(rows=[dict(r) for r in eng.rows], V=Vh)
                del eng
                return res
            return run_variant
        for kind in ('freeze_t2', 'freeze_t0', 'freeze_reset_V', 'resume_ema_t3'):
            res = variant(kind)('SNA_fixT1', SEEDS, data)
            reject(lambda: freeze_validator(res, adapt_t1, 3))
            mut[kind] = True
            del res
            free()
        record('S2-freeze', dict(cli=fz, fixed_alpha_tasks_2_4_bit_identical=True, validator='task-1 rows, V, alpha stats'))

        # ---------------- S2-graph ----------------
        for arm, n in (('SNA_fixT1', 3), ('SNA_pus', 2)):
            ra = M.c2_run_arm(arm, SEEDS, data, graph=False, n_tasks=n, fresh=False)
            rb = M.c2_run_arm(arm, SEEDS, data, graph=True, n_tasks=n, fresh=False)
            assert M.csv_text(ra['rows']) == M.csv_text(rb['rows']), ('eager/graph rows', arm)
            same(ra['end'], rb['end'])
            same(ra['V'], rb['V'])
            del ra, rb
            free()
        record('S2-graph', dict(arms={'SNA_fixT1': 'tasks 1-3 (incl. the recapture at the freeze)', 'SNA_pus': 'tasks 1-2'},
                                eager_graph_bit_identical=True))

        # ---------------- S-resume ----------------
        rs = attempt / 'resume'
        plan = {'c1_arm': 'R_h', 'c2_arm': 'SNA_fix0'}

        def hook(stage, item):
            if plan.get(stage) == item:
                (rs / 'STOP').touch()
        assert M.run(rs, SEEDS, check_mode=True, stop_hook=hook, data=data) is False
        assert json.loads((rs / 'status.json').read_text())['completed_arm'] == 'R_h'
        (rs / 'STOP').unlink()
        plan['c1_arm'] = None
        assert M.run(rs, SEEDS, check_mode=True, stop_hook=hook, data=data) is False
        assert json.loads((rs / 'status.json').read_text())['completed_arm'] == 'SNA_fix0'
        (rs / 'STOP').unlink()
        plan['c2_arm'] = None
        ident_path = rs / 'provenance_start.json'
        saved_ident = ident_path.read_text()
        obj = json.loads(saved_ident)
        obj['identity']['steps_per_task'] = 781
        put(ident_path, obj)
        reject(lambda: M.run(rs, SEEDS, check_mode=True, data=data))
        ident_path.write_text(saved_ident)
        mut['different_identity'] = True
        assert M.run(rs, SEEDS, check_mode=True, data=data) is True
        for name in M.C1_ARMS:
            assert json.loads((rs / 'c1/arms' / name / 'rows.json').read_text()) == json.loads((cli / 'c1/arms' / name / 'rows.json').read_text()), name
            same(RS.core_state(RS.load_pt(rs / 'c1/arms' / name / 'end.pt')), RS.core_state(RS.load_pt(cli / 'c1/arms' / name / 'end.pt')))
        for name in M.C2_ARMS:
            for f in ('per_task.csv', 'fresh_control.csv'):
                assert (rs / 'c2' / name / f).read_text() == (cli / 'c2' / name / f).read_text(), (name, f)
            same(RS.load_pt(rs / 'c2' / name / 'end.pt'), RS.load_pt(cli / 'c2' / name / 'end.pt'))
        ident = json.loads((rs / 'provenance_start.json').read_text())['identity']
        fake = attempt / 'fake'
        fake.mkdir()
        put(fake / 'done.json', dict(identity=ident, complete=True, files={'absent.pt': 'wrong'}))
        reject(lambda: RS.verify_done(fake / 'done.json', ident))
        mut['fake_done'] = True
        (fake / 'a.txt').write_text('good')
        RS.mark_done(fake / 'done.json', ident, [fake / 'a.txt'])
        RS.verify_done(fake / 'done.json', ident)
        (fake / 'a.txt').write_text('bad')
        reject(lambda: RS.verify_done(fake / 'done.json', ident))
        mut['bad_hash'] = True
        record('S-resume', dict(stops=['after C1 R_h', 'after C2 SNA_fix0'], resumed_equals_uninterrupted=True))
        free()

        # ---------------- S-cost / CLI ----------------
        reject(lambda: M.c1_run_arm('bad_arm', {}, data))
        reject(lambda: M.SnakeEngine('bad_arm', SEEDS, data))
        mut['wrong_arm'] = True
        reject(lambda: M.run(attempt / 'budget', SEEDS, steps=781, check_mode=True, data=data))
        reject(lambda: RS.require_identity(ident, {**ident, 'steps_per_task': 781}))
        mut['wrong_budget'] = True
        prov_end = json.loads((cli / 'provenance_end.json').read_text())
        record('S-cost/CLI', dict(cli_seconds_all_arms=cli_seconds, cli_peak_cuda_bytes=prov_end['peak_cuda_bytes'],
                                  verdict_on_cli_output=evidence_verdict_smoke,
                                  suite_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                  c1_arms=list(M.C1_ARMS), c2_arms=list(M.C2_ARMS), seeds=SEEDS, steps_per_task=M.STEPS))

        assert set(evidence) == set(REQUIRED), set(REQUIRED) - set(evidence)
        assert set(mut) == set(MUTANTS), (set(MUTANTS) - set(mut), set(mut) - set(MUTANTS))
        assert tested_source == M.source_hashes(), 'source changed during the checks'
        result = dict(all_pass=True, source_sha256=tested_source, required=REQUIRED, required_mutants=MUTANTS,
                      evidence=evidence, mutants=mut, attempt=str(attempt.relative_to(ROOT)),
                      seconds=time.time() - started, independent_audit=False,
                      deferred_to_main_run={'C1 prefix': '0920 R record (eff_rank reported) and resp backup states',
                                            'C1 N_c/R_h': 'rows equal resp_cifar5p1_1007 rows.json',
                                            'C1 preflight': 'S1-branch/S1-field on every main field arm',
                                            'C2 SNA/KKT1': '0920 records (eff_rank reported), fresh byte-identical'})
        for p in attempt.rglob('*.pt'):
            p.unlink()
        put(attempt / 'checks.json', result)
        put(OUT / 'checks.json', result)
        print('ALL ADMISSION CHECKS PASS', flush=True)
    except BaseException:
        put(attempt / 'failure.json', dict(evidence=evidence, mutants=mut, traceback=traceback.format_exc(),
                                           source_sha256=M.source_hashes()))
        raise


if __name__ == '__main__':
    torch.set_num_threads(2)
    with M.gpu_lock():
        suite()
