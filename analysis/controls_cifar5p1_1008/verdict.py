#!/usr/bin/env python3
"""controls_cifar5p1_1008 -- registered verdict (spec §3.4, §4.3, §5, §6).

    python -m analysis.controls_cifar5p1_1008.verdict --src results/controls_cifar5p1_1008

Reads only the run directory given by --src (never a default main directory when smoke-testing),
the committed 0920 records for R and R+l2init(1e-3) (C2 secondary), and nothing else.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.resp_cifar_ee_0920.stats import interval          # registered t intervals (n = 10)
from analysis.cifar_ledger_0920.replay import put

RUN = 'controls_cifar5p1_1008'
SEEDS = list(range(10))
C1_ARMS = ('N_c', 'R_h', 'R_fresh', 'R_match', 'R_perm', 'R_rand')
C2_ARMS = ('SNA', 'KKT1', 'SNA_fix0', 'KKT1_fix0', 'SNA_fixT1', 'KKT1_fixT1', 'SNA_pus')
LATE = (21, 23, 25, 27, 29)
EARLY = (1, 3, 5, 7, 9)
RECORDS = {'R': ROOT / 'results/cifar5p1_mlp_0920/R_std_lr0.0001',
           'L2I': ROOT / 'results/cifar5p1_mlp_0920/R_std_lr0.0001_l2init1e-3'}
PRIMARY_LEVEL, SECONDARY_LEVEL = 0.975, 0.95
EQUIV_BAND = 0.005

C1_LABELS = ('OPEN_COUNT_SUFFICES', 'PARTIAL', 'LEARNED_FIELD_MATTERS')
C2_LABELS = ('ADAPTIVITY_MATTERS', 'PARTIAL', 'FIXED_ALPHA_SUFFICES')
PRED = {   # spec §6 (registered before the run)
    'C1_label': {'parent': dict(OPEN_COUNT_SUFFICES=.45, PARTIAL=.35, LEARNED_FIELD_MATTERS=.20),
                 'claude': dict(OPEN_COUNT_SUFFICES=.35, PARTIAL=.45, LEARNED_FIELD_MATTERS=.20)},
    'C2_label': {'parent': dict(ADAPTIVITY_MATTERS=.55, PARTIAL=.30, FIXED_ALPHA_SUFFICES=.15),
                 'claude': dict(ADAPTIVITY_MATTERS=.75, PARTIAL=.17, FIXED_ALPHA_SUFFICES=.08)},
    'C1_fresh_restores': {'parent': .85, 'claude': .92},          # R_fresh - N_c 95% low > 0
    'C1_match_restores': {'claude': .92},                         # R_match - N_c 95% low > 0
    'C1_rand_below_h': {'claude': .65},                           # R_h - R_rand 95% low > 0
    'C1_perm_includes_zero': {'claude': .55},                     # R_h - R_perm 95% contains 0
    'C2_fixT1_equivalent': {'parent': .35, 'claude': .20},        # SNA - SNA_fixT1 95% within +-0.005
    'C2_fixT1_below': {'claude': .45},                            # SNA - SNA_fixT1 95% low > 0
    'C2_fix0_above_R': {'claude': .90},                           # SNA_fix0 - R 95% low > 0
    'C2_l2init_above_fix0': {'claude': .80},                      # L2I - SNA_fix0 95% low > 0
    'C2_pus_below': {'claude': .55},                              # SNA - SNA_pus 95% low > 0
}


# --------------------------------------------------------------------------
# labels (pure functions; S-verdict tests these on fixtures)
# --------------------------------------------------------------------------

def label_pair(sign_a, sign_b, names, labels):
    """Two primary signs -> (label, qualifier, annotations).  '-' never counts toward the first
    label (the control did at least as well), spec §3.4/§4.3."""
    both, partial, neither = labels
    for s in (sign_a, sign_b):
        assert s in ('+', '0', '-'), s
    plus = [n for n, s in zip(names, (sign_a, sign_b)) if s == '+']
    minus = [n for n, s in zip(names, (sign_a, sign_b)) if s == '-']
    if len(plus) == 2:
        lab, qual = both, None
    elif len(plus) == 1:
        lab, qual = partial, f'{plus[0]}_differs'
    else:
        lab, qual = neither, None
    return lab, qual, [f'EXCEEDS:{n}' for n in minus]


def c1_label(s_fresh, s_match):
    lab, qual, ann = label_pair(s_fresh, s_match, ('fresh', 'match'),
                                ('LEARNED_FIELD_MATTERS', 'PARTIAL', 'OPEN_COUNT_SUFFICES'))
    return lab, qual, [a.replace('EXCEEDS:', 'CONTROL_EXCEEDS:R_') for a in ann]


def c2_label(s_sna, s_kkt1):
    lab, qual, ann = label_pair(s_sna, s_kkt1, ('SNA', 'KKT1'),
                                ('ADAPTIVITY_MATTERS', 'PARTIAL', 'FIXED_ALPHA_SUFFICES'))
    return lab, qual, [a.replace('EXCEEDS:', 'FIXED_EXCEEDS:') + '_fix0' for a in ann]


def diff(E, a, b):
    """Paired a - b over the ten seeds; refuses missing arms, missing seeds and non-finite values."""
    x, y = np.asarray(E[a], dtype=np.float64), np.asarray(E[b], dtype=np.float64)
    assert x.shape == (10,) and y.shape == (10,), ('ten seeds per arm', a, b)
    assert np.isfinite(x).all() and np.isfinite(y).all(), ('non-finite', a, b)
    return x - y


def comparison(E, a, b, level):
    dv = diff(E, a, b)
    iv = interval(dv, level)
    return dict(a=a, b=b, level=level, mean=iv['mean'], sd=iv['sd'], low=iv['low'], high=iv['high'],
                sign=iv['sign'], degenerate_sd=iv['degenerate_sd'], n_positive=int((dv > 0).sum()),
                min=float(dv.min()), max=float(dv.max()))


def c1_verdict(E, status):
    """E: arm -> ten task-29 online values (seed order); status: completeness and reproduction gates."""
    out = dict(control='C1', status=dict(status))
    if not status.get('complete') or not status.get('checks_pass'):
        out['label'] = 'INCOMPLETE' if not status.get('complete') else 'CHECK_FAILED'
        return out
    if not status.get('reproduced'):
        out['label'] = 'CHECK_FAILED'
        return out
    for arm in C1_ARMS:
        assert arm in E, ('missing arm', arm)
    p1 = comparison(E, 'R_h', 'N_c', PRIMARY_LEVEL)
    out['P1'] = p1
    if p1['sign'] != '+':
        out['label'] = 'NOT_REPRODUCED'
        return out
    prim = {'D_fresh': comparison(E, 'R_h', 'R_fresh', PRIMARY_LEVEL),
            'D_match': comparison(E, 'R_h', 'R_match', PRIMARY_LEVEL)}
    out['primary'] = prim
    out['primary_95'] = {k: comparison(E, 'R_h', v['b'], SECONDARY_LEVEL) for k, v in prim.items()}
    lab, qual, ann = c1_label(prim['D_fresh']['sign'], prim['D_match']['sign'])
    out.update(label=lab, qualifier=qual, annotations=ann,
               degenerate=[k for k, v in prim.items() if v['degenerate_sd']])
    sec = [('R_h', 'R_perm'), ('R_h', 'R_rand')] + [(a, 'N_c') for a in C1_ARMS if a != 'N_c']
    sec += [(a, 'fresh_t29') for a in C1_ARMS if 'fresh_t29' in E]
    out['secondary'] = [comparison(E, a, b, SECONDARY_LEVEL) for a, b in sec]
    base = float(np.mean(diff(E, 'R_h', 'N_c')))
    out['restoration_share'] = {a: float(np.mean(diff(E, a, 'N_c'))) / base for a in C1_ARMS if a != 'N_c'}
    return out


def c2_verdict(E, status):
    out = dict(control='C2', status=dict(status))
    if not status.get('complete') or not status.get('checks_pass'):
        out['label'] = 'INCOMPLETE' if not status.get('complete') else 'CHECK_FAILED'
        return out
    if not status.get('reproduced'):
        out['label'] = 'CHECK_FAILED'
        return out
    for arm in C2_ARMS:
        assert arm in E, ('missing arm', arm)
    prim = {'A_SNA': comparison(E, 'SNA', 'SNA_fix0', PRIMARY_LEVEL),
            'A_KKT1': comparison(E, 'KKT1', 'KKT1_fix0', PRIMARY_LEVEL)}
    out['primary'] = prim
    out['primary_95'] = {k: comparison(E, v['a'], v['b'], SECONDARY_LEVEL) for k, v in prim.items()}
    lab, qual, ann = c2_label(prim['A_SNA']['sign'], prim['A_KKT1']['sign'])
    out.update(label=lab, qualifier=qual, annotations=ann,
               degenerate=[k for k, v in prim.items() if v['degenerate_sd']])
    sec = [('SNA', 'SNA_fixT1'), ('KKT1', 'KKT1_fixT1'), ('SNA', 'SNA_pus'), ('SNA_pus', 'R')]
    fixes = ('SNA_fix0', 'KKT1_fix0', 'SNA_fixT1', 'KKT1_fixT1')
    sec += [(f, 'R') for f in fixes] + [(f, 'L2I') for f in fixes] + [('L2I', 'SNA_fix0')]
    out['secondary'] = [comparison(E, a, b, SECONDARY_LEVEL) for a, b in sec if a in E and b in E]
    eq = {}
    for a, b in (('SNA', 'SNA_fixT1'), ('KKT1', 'KKT1_fixT1')):
        c = comparison(E, a, b, SECONDARY_LEVEL)
        eq[f'{a}-{b}'] = dict(low=c['low'], high=c['high'],
                              equivalent=bool(c['low'] >= -EQUIV_BAND and c['high'] <= EQUIV_BAND))
    out['fixT1_equivalence_0.005'] = eq
    return out


# --------------------------------------------------------------------------
# predictions (spec §6)
# --------------------------------------------------------------------------

def find(sec, a, b):
    m = [c for c in sec if c['a'] == a and c['b'] == b]
    assert len(m) == 1, (a, b)
    return m[0]


def outcomes(v1, v2):
    """Binary outcomes of the registered prediction items (None when not applicable)."""
    o = {}
    if 'secondary' in v1:
        o['C1_fresh_restores'] = find(v1['secondary'], 'R_fresh', 'N_c')['sign'] == '+'
        o['C1_match_restores'] = find(v1['secondary'], 'R_match', 'N_c')['sign'] == '+'
        o['C1_rand_below_h'] = find(v1['secondary'], 'R_h', 'R_rand')['sign'] == '+'
        o['C1_perm_includes_zero'] = find(v1['secondary'], 'R_h', 'R_perm')['sign'] == '0'
    if 'secondary' in v2:
        o['C2_fixT1_equivalent'] = v2['fixT1_equivalence_0.005']['SNA-SNA_fixT1']['equivalent']
        o['C2_fixT1_below'] = find(v2['secondary'], 'SNA', 'SNA_fixT1')['sign'] == '+'
        o['C2_fix0_above_R'] = find(v2['secondary'], 'SNA_fix0', 'R')['sign'] == '+'
        o['C2_l2init_above_fix0'] = find(v2['secondary'], 'L2I', 'SNA_fix0')['sign'] == '+'
        o['C2_pus_below'] = find(v2['secondary'], 'SNA', 'SNA_pus')['sign'] == '+'
    return o


def score(v1, v2):
    rows = []
    for key, verdict, labels in (('C1_label', v1, C1_LABELS), ('C2_label', v2, C2_LABELS)):
        lab = verdict.get('label')
        for who, dist in PRED[key].items():
            assert abs(sum(dist.values()) - 1) < 1e-12
            if lab not in labels:
                rows.append(dict(item=key, who=who, p=None, outcome=lab, hit=None, brier=None, scored=False))
                continue
            top = max(dist, key=dist.get)
            brier = sum((dist[k] - (1.0 if k == lab else 0.0)) ** 2 for k in labels)
            rows.append(dict(item=key, who=who, p=dist[lab], outcome=lab, top=top, hit=top == lab,
                             brier=brier, scored=True))
    for key, val in outcomes(v1, v2).items():
        for who, p in PRED[key].items():
            rows.append(dict(item=key, who=who, p=p, outcome=bool(val), hit=(p > .5) == bool(val),
                             brier=(p - float(val)) ** 2, scored=True))
    return rows


# --------------------------------------------------------------------------
# reading the run directory
# --------------------------------------------------------------------------

def read_json(p):
    return json.loads(Path(p).read_text())


def load_c1(src, smoke=False):
    E, status, table, per_seed, fields = {}, dict(complete=True, checks_pass=True, reproduced=True), [], [], []
    arms = src / 'c1' / 'arms'
    pre = src / 'c1' / 'prefix'
    for arm in C1_ARMS:
        d = arms / arm
        if not (d / 'done.json').exists():
            status['complete'] = False
            status.setdefault('missing', []).append(arm)
            continue
        rows = sorted(read_json(d / 'rows.json'), key=lambda r: r['seed'])
        if [r['seed'] for r in rows] != SEEDS or not all(r['finite'] for r in rows):
            status['complete'] = False
            continue
        E[arm] = np.array([r['online_acc'] for r in rows])
        diag = read_json(d / 'diagnostics.json')
        by = {}
        for r in diag:
            by.setdefault(r['update'], []).append(r)
        pf = read_json(d / 'preflight.json') if (d / 'preflight.json').exists() else None
        if pf is not None and not pf.get('all_pass'):
            status['checks_pass'] = False
        table.append(dict(arm=arm, task=29, online_mean=float(E[arm].mean()), online_sd=float(E[arm].std(ddof=1)),
                          online_min=float(E[arm].min()), online_max=float(E[arm].max()),
                          memo_mean=float(np.mean([r['memo_acc'] for r in rows])),
                          online_ce_mean=float(np.mean([r['online_ce'] for r in rows])),
                          **{f'G2_u{u}': float(np.mean([r['G2'] for r in by[u]])) for u in (0, 78, 390, 780)},
                          **{f'G1_u{u}': float(np.mean([r['G1'] for r in by[u]])) for u in (0, 780)},
                          dead_unit2_u0=float(np.mean([r['dead_unit2'] for r in by[0]])),
                          dead_unit2_u780=float(np.mean([r['dead_unit2'] for r in by[780]])),
                          neff_fraction2_u780=float(np.mean([r['neff_fraction2'] for r in by[780]]))))
        for r in rows:
            per_seed.append(dict(arm=arm, seed=r['seed'], online_acc=r['online_acc'], online_ce=r['online_ce'],
                                 memo_acc=r['memo_acc'], memo_ce=r['memo_ce'],
                                 G2_u0=[q['G2'] for q in by[0] if q['seed'] == r['seed']][0],
                                 G2_u780=[q['G2'] for q in by[780] if q['seed'] == r['seed']][0]))
        if pf is not None and 'open_pairs_at_branch_per_seed' in pf:
            for i, s in enumerate(SEEDS):
                fields.append(dict(arm=arm, seed=s, open_pairs_at_branch=pf['open_pairs_at_branch_per_seed'][i],
                                   open_fraction_at_branch=pf['open_pairs_at_branch_per_seed'][i] / 250000,
                                   ambiguous_pairs=pf['ambiguous_pairs_per_seed'][i],
                                   argument_error_ratio=pf['argument_error_ratio_per_seed'][i],
                                   field_sd=pf['field_sd_per_seed'][i],
                                   K=pf['K'][i] if 'K' in pf else None,
                                   K_achieved=pf['K_achieved'][i] if 'K_achieved' in pf else None,
                                   shift=pf['shift'][i] if 'shift' in pf else None))
    fresh = sorted(read_json(pre / 'fresh.json'), key=lambda r: r['seed']) if (pre / 'fresh.json').exists() else []
    if [r['seed'] for r in fresh] == SEEDS:
        E['fresh_t29'] = np.array([r['fresh_online_acc'] for r in fresh])
    hr = pre / 'host_record_check.json'
    status['prefix_record'] = read_json(hr) if hr.exists() else None
    if not smoke and (status['prefix_record'] is None or not status['prefix_record']['all_pass']):
        status['checks_pass'] = False
    for arm in (() if smoke else ('N_c', 'R_h')):
        p = arms / arm / 'reproduction_check.json'
        chk = read_json(p) if p.exists() else None
        status[f'reproduction_{arm}'] = chk
        if chk is None or not chk['all_pass']:
            status['reproduced'] = False
    pc = arms / 'N_c' / 'prefix_check.json'
    status['prefix_continuation'] = read_json(pc) if pc.exists() else None
    if status['prefix_continuation'] is None or not status['prefix_continuation']['all_pass']:
        status['checks_pass'] = False
    return E, status, table, per_seed, fields


def late(rows_by_seed, tasks):
    return np.array([np.mean([rows_by_seed[s][t] for t in tasks]) for s in SEEDS])


def read_per_task(path):
    by = {}
    full = {}
    for r in csv.DictReader(Path(path).open()):
        s, t = int(r['seed']), int(r['task'])
        by.setdefault(s, {})[t] = float(r['online_acc'])
        full.setdefault(s, {})[t] = r
    assert sorted(by) == SEEDS and all(sorted(v) == list(range(1, 31)) for v in by.values()), ('30 tasks x 10 seeds', path)
    return by, full


def load_c2(src, records=True):
    E, status, table, per_seed = {}, dict(complete=True, checks_pass=True, reproduced=True), [], []
    for arm in C2_ARMS:
        d = src / 'c2' / arm
        if not (d / 'done.json').exists():
            status['complete'] = False
            status.setdefault('missing', []).append(arm)
            continue
        by, full = read_per_task(d / 'per_task.csv')
        fr = {int(r['seed']): r for r in csv.DictReader((d / 'fresh_control.csv').open())}
        E[arm] = late(by, LATE)
        early = late(by, EARLY)
        f29 = {k: [float(full[s][29][k]) for s in SEEDS] for k in
               ('zsd_l1', 'zsd_l2', 'alpha_med_l1', 'alpha_med_l2', 'two_alpha_W_med_l1', 'two_alpha_W_med_l2',
                'mob_l1', 'mob_l2', 'eff_rank_l1', 'eff_rank_l2', 'w_norm_l1', 'w_norm_l2', 'w_norm_l3', 'test_acc')}
        f1 = {k: [float(full[s][1][k]) for s in SEEDS] for k in ('zsd_l1', 'zsd_l2', 'alpha_med_l1', 'alpha_med_l2')}
        gaps = [float(fr[s]['fresh_gap']) for s in SEEDS] if sorted(fr) == SEEDS else [float('nan')] * 10
        table.append(dict(arm=arm, late_mean=float(E[arm].mean()), late_sd=float(E[arm].std(ddof=1)),
                          late_median=float(np.median(E[arm])), early_mean=float(early.mean()),
                          drop_mean=float((early - E[arm]).mean()), fresh_gap_mean=float(np.mean(gaps)),
                          fresh_gap_positive=int(sum(g > 0 for g in gaps)),
                          **{f'{k}_t29': float(np.median(v)) for k, v in f29.items()},
                          **{f'{k}_t1': float(np.median(v)) for k, v in f1.items()}))
        for i, s in enumerate(SEEDS):
            per_seed.append(dict(arm=arm, seed=s, late=float(E[arm][i]), early=float(early[i]),
                                 fresh_gap=gaps[i], t29_online=by[s][29]))
        if arm in ('SNA', 'KKT1') and records:
            p = d / 'record_check.json'
            chk = read_json(p) if p.exists() else None
            status[f'record_{arm}'] = chk
            if chk is None or not chk['all_pass']:
                status['reproduced'] = False
    for name, rec in (RECORDS.items() if records else ()):
        by, _ = read_per_task(rec / 'per_task.csv')
        E[name] = late(by, LATE)
        table.append(dict(arm=name, record=str(rec.relative_to(ROOT)), late_mean=float(E[name].mean()),
                          late_sd=float(E[name].std(ddof=1)), late_median=float(np.median(E[name])),
                          early_mean=float(late(by, EARLY).mean())))
    return E, status, table, per_seed


def write_csv(path, rows):
    path = Path(path)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.10g}' if isinstance(v, float) else v) for k, v in r.items()})


def f4(x):
    return 'NA' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:+.4f}'


def summary_md(v1, v2, t1, t2, preds, src):
    L = [f'# controls_cifar5p1_1008 — 5+1 CIFAR × MLP の 2 対照（登録 `2adc8331`）', '',
         f'seed 0–9（R=10）、std、Adam 1e−4、780 更新/課題。自動生成（`analysis/controls_cifar5p1_1008/verdict.py --src {src}`）。独立監査なし。', '']
    L += ['## C1: 場の中身か、開いたゲートの数か（受け手 = task 28 末、継続 task 29、Adam 継続、第 2 層）', '',
          f'**登録主判定: {v1.get("label")}**' + (f'（{v1["qualifier"]}）' if v1.get('qualifier') else '')
          + (f' {", ".join(v1["annotations"])}' if v1.get('annotations') else ''), '']
    if 'primary' in v1:
        L += ['| 主比較 | 平均対応差 | 97.5% 下端 | 上端 | 95% 下端 | 上端 | 符号 | 正の seed |', '|---|---:|---:|---:|---:|---:|---|---:|']
        for k, c in v1['primary'].items():
            c95 = v1['primary_95'][k]
            L.append(f'| {k} = E({c["a"]}) − E({c["b"]}) | {f4(c["mean"])} | {f4(c["low"])} | {f4(c["high"])} | {f4(c95["low"])} | {f4(c95["high"])} | {c["sign"]} | {c["n_positive"]}/10 |')
        p1 = v1['P1']
        L += ['', f'適用条件: N_c・R_h の行が resp_cifar5p1_1007 と一致、P1 = E(R_h) − E(N_c) = {f4(p1["mean"])}（97.5% [{f4(p1["low"])}, {f4(p1["high"])}]）。', '']
    L += ['| 腕 | online 平均 | SD | 範囲 | memo | 分岐直後 G2 | 更新 78 の G2 | 終端 G2 | 終端で閉じた第 2 層 unit |', '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in t1:
        L.append(f'| {r["arm"]} | {r["online_mean"]:.4f} | {r["online_sd"]:.4f} | {r["online_min"]:.4f}–{r["online_max"]:.4f} | {r["memo_mean"]:.4f} | {r["G2_u0"]:.4f} | {r["G2_u78"]:.4f} | {r["G2_u780"]:.4f} | {r["dead_unit2_u780"]:.3f} |')
    if 'secondary' in v1:
        L += ['', '| 副比較（95%・REPORT_ONLY） | 平均差 | 下端 | 上端 | 符号 | 正の seed |', '|---|---:|---:|---:|---|---:|']
        for c in v1['secondary']:
            L.append(f'| E({c["a"]}) − E({c["b"]}) | {f4(c["mean"])} | {f4(c["low"])} | {f4(c["high"])} | {c["sign"]} | {c["n_positive"]}/10 |')
        L += ['', '復元の割合 ρ_X = mean(E(X) − E(N_c)) / mean(E(R_h) − E(N_c)): '
              + '、'.join(f'{k} {v:.3f}' for k, v in v1['restoration_share'].items()) + '（報告のみ）。', '']
    L += ['## C2: 適応 α の追随か（後期窓 = hard 21–29 の online）', '',
          f'**登録主判定: {v2.get("label")}**' + (f'（{v2["qualifier"]}）' if v2.get('qualifier') else '')
          + (f' {", ".join(v2["annotations"])}' if v2.get('annotations') else ''), '']
    if 'primary' in v2:
        L += ['| 主比較 | 平均対応差 | 97.5% 下端 | 上端 | 95% 下端 | 上端 | 符号 | 正の seed |', '|---|---:|---:|---:|---:|---:|---|---:|']
        for k, c in v2['primary'].items():
            c95 = v2['primary_95'][k]
            L.append(f'| {k} = E({c["a"]}) − E({c["b"]}) | {f4(c["mean"])} | {f4(c["low"])} | {f4(c["high"])} | {f4(c95["low"])} | {f4(c95["high"])} | {c["sign"]} | {c["n_positive"]}/10 |')
        L.append('')
    L += ['| 腕 | 後期窓 平均 | SD | 中央値 | 早期窓 | 低下 | fresh gap | α 中央値 l1/l2（t29） | zsd l1/l2（t29） | 2αW l1/l2（t29） | mob l2（t29） |',
          '|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|']
    for r in t2:
        if 'record' in r:
            L.append(f'| {r["arm"]}（記録） | {r["late_mean"]:.4f} | {r["late_sd"]:.4f} | {r["late_median"]:.4f} | {r["early_mean"]:.4f} | | | | | | |')
        else:
            L.append(f'| {r["arm"]} | {r["late_mean"]:.4f} | {r["late_sd"]:.4f} | {r["late_median"]:.4f} | {r["early_mean"]:.4f} | {r["drop_mean"]:+.4f} | {r["fresh_gap_mean"]:+.4f} ({r["fresh_gap_positive"]}/10) | {r["alpha_med_l1_t29"]:.3f}/{r["alpha_med_l2_t29"]:.3f} | {r["zsd_l1_t29"]:.2f}/{r["zsd_l2_t29"]:.2f} | {r["two_alpha_W_med_l1_t29"]:.2f}/{r["two_alpha_W_med_l2_t29"]:.2f} | {r["mob_l2_t29"]:.3f} |')
    if 'secondary' in v2:
        L += ['', '| 副比較（95%・REPORT_ONLY） | 平均差 | 下端 | 上端 | 符号 | 正の seed |', '|---|---:|---:|---:|---|---:|']
        for c in v2['secondary']:
            L.append(f'| E({c["a"]}) − E({c["b"]}) | {f4(c["mean"])} | {f4(c["low"])} | {f4(c["high"])} | {c["sign"]} | {c["n_positive"]}/10 |')
        eq = v2['fixT1_equivalence_0.005']
        L += ['', 'fixT1 の同等性（95% 区間 ⊂ ±0.005）: ' + '、'.join(f'{k} [{f4(v["low"])}, {f4(v["high"])}] → {v["equivalent"]}' for k, v in eq.items()) + '。', '']
    L += ['## 予測の採点（spec §6）', '', '| 項目 | 誰 | 確率 | 結果 | 的中 | Brier |', '|---|---|---:|---|---|---:|']
    for r in preds:
        if not r['scored']:
            L.append(f'| {r["item"]} | {r["who"]} | — | {r["outcome"]}（採点なし） | — | — |')
        elif r['item'].endswith('_label'):
            L.append(f'| {r["item"]}（最大確率 {r["top"]}） | {r["who"]} | {r["p"]:.2f} | {r["outcome"]} | {r["hit"]} | {r["brier"]:.4f} |')
        else:
            L.append(f'| {r["item"]} | {r["who"]} | {r["p"]:.2f} | {r["outcome"]} | {r["hit"]} | {r["brier"]:.4f} |')
    L += ['', '## 読みの上限（spec §3.5・§4.4 の再掲）', '',
          'C1 は 1 課題 780 更新の固定場の人工的な介入で、自然な喪失の全媒介や恒久的な救済は示さない。R_fresh は開いた数も z の尺度も揃っていない。'
          '0 は差を示せないで、同等性ではない。C2 の後期窓の差は活性化の機構の判定ではない。副比較・ρ・fresh gap は REPORT_ONLY。', '']
    return '\n'.join(L)


def run_verdict(src, out=None, seeds=tuple(range(10))):
    """Registered verdict on a completed run directory.  Seeds other than 0-9 (check-mode smoke
    tests) skip the 0920 records and the prediction scoring, which exist only for seeds 0-9."""
    global SEEDS
    SEEDS = list(seeds)
    smoke = SEEDS != list(range(10))
    src = Path(src)
    out = Path(out) if out else src
    out.mkdir(parents=True, exist_ok=True)
    st = read_json(src / 'status.json')
    assert st.get('stage') == 'completed', ('run not completed', st)
    assert (src / 'provenance_start.json').exists(), 'missing start provenance'
    assert (src / 'provenance_end.json').exists(), 'missing end provenance'
    E1, s1, t1, ps1, fields = load_c1(src, smoke=smoke)        # check mode: no committed rows/records
    E2, s2, t2, ps2 = load_c2(src, records=not smoke)
    v1 = c1_verdict(E1, s1)
    v2 = c2_verdict(E2, s2)
    preds = [] if smoke else score(v1, v2)
    put(out / 'verdict.json', dict(run_id=RUN, registration='2adc8331', C1=v1, C2=v2, predictions=preds,
                                   source=str(src), seeds=SEEDS, smoke=smoke))
    write_csv(out / 'verdict.csv', [dict(control='C1', label=v1.get('label'), qualifier=v1.get('qualifier'),
                                         annotations=';'.join(v1.get('annotations', []))),
                                    dict(control='C2', label=v2.get('label'), qualifier=v2.get('qualifier'),
                                         annotations=';'.join(v2.get('annotations', [])))])
    paired = []
    for ctl, v in (('C1', v1), ('C2', v2)):
        for k, c in v.get('primary', {}).items():
            paired.append(dict(control=ctl, role='primary', name=k, **c))
        for k, c in v.get('primary_95', {}).items():
            paired.append(dict(control=ctl, role='primary_at_95', name=k, **c))
        for c in v.get('secondary', []):
            paired.append(dict(control=ctl, role='secondary', name=f'{c["a"]}-{c["b"]}', **c))
    write_csv(out / 'paired.csv', paired)
    write_csv(out / 'c1_arm_table.csv', t1)
    write_csv(out / 'c1_per_seed.csv', ps1)
    write_csv(out / 'field_stats.csv', fields)
    write_csv(out / 'c2_arm_table.csv', t2)
    write_csv(out / 'c2_per_seed.csv', ps2)
    write_csv(out / 'predictions.csv', preds)
    (out / 'summary.md').write_text(summary_md(v1, v2, t1, t2, preds, src))
    return v1, v2, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='run directory (results/controls_cifar5p1_1008 for the main run)')
    ap.add_argument('--out', default=None, help='where to write (default: --src)')
    ap.add_argument('--seeds', default='0-9')
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition('-')
    seeds = list(range(int(lo), int(hi or lo) + 1))
    v1, v2, _ = run_verdict(a.src, a.out, seeds)
    print(json.dumps(dict(C1=v1.get('label'), C1_q=v1.get('qualifier'), C2=v2.get('label'), C2_q=v2.get('qualifier')),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
