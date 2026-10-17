#!/usr/bin/env python3
"""sna_grow_1017 -- registered verdict (spec specs/spec_sna_grow_1017.md §4.1, §4.2).

    .venv/bin/python analysis/sna_grow_1017/verdict.py

Reads (read only):
  results/sna_grow_1017/cnn/per_task.csv            Q1 (arms SNA, SNA+grow1; seeds 10-19; tasks 1-30)
  results/sna_grow_1017/cnn_checks.json             §3 K-cnn-* gate
  results/sna_grow_1017/mlp/SNA_grow/               Q2 (per_task.csv, done.json; seeds 0-9)
  results/sna_grow_1017/mlp_checks.json             §3 K-mlp-* gate
  results/controls_cifar5p1_1008/c2_per_seed.csv    Q2 references (SNA, SNA_fixT1, SNA_fix0: column `late`)
  results/controls_cifar5p1_1008/c2/<arm>/per_task.csv   Q2 reference layer statistics at t29
Writes:
  results/sna_grow_1017/verdict.json
  results/sna_grow_1017/verdict_tables.md           (numbers only)

Registered bands (spec §4; not to be changed here):
  Q1  D = mean_s [W21(SNA+grow1)_s - W21(SNA)_s], W21 = mean online_acc over tasks 21-30.
      GROW_RESCUES: D >= 0.0055 and positive seeds >= 8/10.  GROW_NOT: D <= 0.0018.  GROW_PARTIAL: otherwise.
  Q2  M = mean_s [late(SNA_grow)_s - late(SNA)_s], late = mean online_acc over the hard tasks 21-29, seeds 0-9,
      95% t interval (df 9).  NO_LOSS: M >= -0.005 and low >= -0.010.  LOSS: M <= -0.010.  UNRESOLVED: otherwise.
  A part whose main output is absent is NOT_RUN.  A part whose §3 checks file does not say all_pass is CHECK_FAILED.
Everything below the labels is REPORT_ONLY (spec §4.1 副, §4.2 副).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.resp_cifar_ee_0920.stats import interval      # registered t interval (n = 10, df = 9)

RUN = 'sna_grow_1017'
RES = ROOT / 'results' / RUN
CNN_CSV = RES / 'cnn' / 'per_task.csv'
CNN_CHECKS = RES / 'cnn_checks.json'
MLP_DIR = RES / 'mlp' / 'SNA_grow'
MLP_CHECKS = RES / 'mlp_checks.json'
REF_DIR = ROOT / 'results' / 'controls_cifar5p1_1008'
REF_PER_SEED = REF_DIR / 'c2_per_seed.csv'
OUT_JSON = RES / 'verdict.json'
OUT_MD = RES / 'verdict_tables.md'

# ---- registered constants (spec §4) ----
ARM_HOST, ARM_GROW = 'SNA', 'SNA+grow1'
CNN_SEEDS = list(range(10, 20))
CNN_TASKS = list(range(1, 31))
W21 = tuple(range(21, 31))
W11 = tuple(range(11, 21))
SCALE = 0.0073                       # SNAc3 - SNA (bundle A), registered fixed value
RESCUE_D, NOT_D = 0.0055, 0.0018     # 0.75 x 0.0073, 0.25 x 0.0073 (as registered, 4 decimals)
RESCUE_POS = 8
MLP_SEEDS = list(range(10))
LATE_HARD = (21, 23, 25, 27, 29)     # hard tasks in 21-29 (odd tasks are hard)
NO_LOSS_M, NO_LOSS_LOW, LOSS_M = -0.005, -0.010, -0.010
RUNAWAY_CE, RUNAWAY_ZSD_F2 = 100.0, 24.0
SITES = ('c1', 'c2', 'f1', 'f2')
SHOW_TASKS = (1, 2, 3, 5, 10, 15, 20, 21, 25, 30)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def put(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def paired(x, y, level=0.95):
    """x - y over ten paired seeds: mean, SD, SE, t interval, positive count, min, max."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    iv = interval(d, level)
    return dict(mean=iv['mean'], sd=iv['sd'], se=iv['sd'] / math.sqrt(len(d)), low=iv['low'], high=iv['high'],
                level=level, sign=iv['sign'], n_positive=int((d > 0).sum()), n=len(d),
                min=float(d.min()), max=float(d.max()), per_seed=[float(v) for v in d])


# --------------------------------------------------------------------------
# labels (pure functions of the registered quantities)
# --------------------------------------------------------------------------

def q1_label(D, n_pos):
    if D >= RESCUE_D and n_pos >= RESCUE_POS:
        return 'GROW_RESCUES'
    if D <= NOT_D:
        return 'GROW_NOT'
    return 'GROW_PARTIAL'


def q2_label(M, low):
    if M >= NO_LOSS_M and low >= NO_LOSS_LOW:
        return 'NO_LOSS'
    if M <= LOSS_M:
        return 'LOSS'
    return 'UNRESOLVED'


def _selftest():
    assert q1_label(0.0055, 8) == 'GROW_RESCUES'
    assert q1_label(0.0055, 7) == 'GROW_PARTIAL'
    assert q1_label(0.0054, 10) == 'GROW_PARTIAL'
    assert q1_label(0.0018, 10) == 'GROW_NOT'
    assert q1_label(0.0019, 0) == 'GROW_PARTIAL'
    assert q1_label(-0.01, 0) == 'GROW_NOT'
    assert q2_label(-0.005, -0.010) == 'NO_LOSS'
    assert q2_label(-0.004, -0.0101) == 'UNRESOLVED'
    assert q2_label(-0.0051, -0.009) == 'UNRESOLVED'
    assert q2_label(-0.010, -0.02) == 'LOSS'
    assert q2_label(0.003, -0.002) == 'NO_LOSS'


# --------------------------------------------------------------------------
# Q1 (CNN)
# --------------------------------------------------------------------------

def read_cnn():
    rows = list(csv.DictReader(CNN_CSV.open()))
    by = {}
    for r in rows:
        by[(r['arm'], int(r['seed']), int(r['task']))] = r
    return rows, by


def fl(r, k):
    return float(r[k])


def med(v):
    return float(np.median(np.asarray(v, dtype=np.float64)))


def q1(status):
    out = dict(status=status)
    if not CNN_CSV.exists():
        out['label'] = 'NOT_RUN'
        return out, None
    rows, by = read_cnn()
    missing = [(a, s, t) for a in (ARM_HOST, ARM_GROW) for s in CNN_SEEDS for t in CNN_TASKS if (a, s, t) not in by]
    status['complete'] = not missing
    status['n_rows'] = len(rows)
    status['missing'] = missing[:10]
    if missing:
        out['label'] = 'INCOMPLETE'
        return out, by
    vals = [fl(by[(a, s, t)], 'online_acc') for a in (ARM_HOST, ARM_GROW) for s in CNN_SEEDS for t in CNN_TASKS]
    status['all_online_finite'] = bool(np.isfinite(vals).all())
    status['n_diverged_rows'] = sum(by[(a, s, t)]['diverged'] == 'True' for a in (ARM_HOST, ARM_GROW)
                                    for s in CNN_SEEDS for t in CNN_TASKS)
    if not status.get('checks_all_pass'):
        out['label'] = 'CHECK_FAILED'
        return out, by

    def window(arm, tasks):
        return np.array([np.mean([fl(by[(arm, s, t)], 'online_acc') for t in tasks]) for s in CNN_SEEDS])

    w_h, w_g = window(ARM_HOST, W21), window(ARM_GROW, W21)
    p = paired(w_g, w_h)
    D, n_pos = p['mean'], p['n_positive']
    out.update(label=q1_label(D, n_pos), D=D, n_positive=n_pos, n_seeds=len(CNN_SEEDS),
               D_over_scale=D / SCALE,
               bands=dict(GROW_RESCUES=f'D >= {RESCUE_D} and positive seeds >= {RESCUE_POS}/10',
                          GROW_NOT=f'D <= {NOT_D}', GROW_PARTIAL='otherwise', scale=SCALE),
               W21_mean={ARM_HOST: float(w_h.mean()), ARM_GROW: float(w_g.mean())},
               W21_per_seed={ARM_HOST: w_h.tolist(), ARM_GROW: w_g.tolist()},
               D_sd=p['sd'], D_se=p['se'], D_t95_report_only=[p['low'], p['high']],
               D_min=p['min'], D_max=p['max'], D_per_seed=dict(zip(CNN_SEEDS, p['per_seed'])))
    return out, by


def q1_secondary(by):
    sec = {}
    # window t11-20, the same paired difference
    def window(arm, tasks):
        return np.array([np.mean([fl(by[(arm, s, t)], 'online_acc') for t in tasks]) for s in CNN_SEEDS])
    w_h, w_g = window(ARM_HOST, W11), window(ARM_GROW, W11)
    p = paired(w_g, w_h)
    sec['window_t11_20'] = dict(mean_SNA=float(w_h.mean()), mean_grow=float(w_g.mean()), diff=p['mean'],
                                sd=p['sd'], se=p['se'], t95=[p['low'], p['high']], n_positive=p['n_positive'],
                                per_seed=dict(zip(CNN_SEEDS, p['per_seed'])))
    # per-task online mean (for the table)
    sec['online_mean_by_task'] = {a: {t: float(np.mean([fl(by[(a, s, t)], 'online_acc') for s in CNN_SEEDS]))
                                      for t in CNN_TASKS} for a in (ARM_HOST, ARM_GROW)}

    def per_task_stat(key, fn=med, transform=None):
        res = {}
        for a in (ARM_HOST, ARM_GROW):
            res[a] = {}
            for t in CNN_TASKS:
                v = [transform(by[(a, s, t)]) if transform else fl(by[(a, s, t)], key) for s in CNN_SEEDS]
                res[a][t] = fn(v)
        return res

    sec['ep_first99_median'] = per_task_stat('ep_first99')
    sec['ep_first99_mean'] = per_task_stat('ep_first99', fn=lambda v: float(np.mean(v)))
    sec['ep_first99_never_count'] = {a: sum(int(by[(a, s, t)]['ep_first99']) < 0 for s in CNN_SEEDS for t in CNN_TASKS)
                                     for a in (ARM_HOST, ARM_GROW)}
    # task-end 2 alpha W: from alpha and the ACTUAL preactivation width (zsd of the end-of-task evaluation),
    # not from the stored two_alpha_W_med (which uses sqrt of the EMA V; in the grow arm alpha uses V_max)
    tw = {}
    for site in ('c2', 'f1', 'f2'):
        tw[site] = dict(
            unit_median_2alpha_sd=per_task_stat(f'two_alpha_sd_med_{site}'),
            product_2_alphamed_zsdmed=per_task_stat(None, transform=lambda r, s=site: 2 * fl(r, f'alpha_med_{s}') * fl(r, f'zsd_{s}')),
            stored_two_alpha_W_med_from_V=per_task_stat(f'two_alpha_W_med_{site}'),
            alpha_med=per_task_stat(f'alpha_med_{site}'),
            zsd=per_task_stat(f'zsd_{site}'))
        tw[site]['unit_median_2alpha_sd_window21_30'] = {
            a: dict(median_over_seeds_of_window_mean=med([np.mean([fl(by[(a, s, t)], f'two_alpha_sd_med_{site}') for t in W21])
                                                          for s in CNN_SEEDS]),
                    min=float(min(fl(by[(a, s, t)], f'two_alpha_sd_med_{site}') for s in CNN_SEEDS for t in W21)),
                    max=float(max(fl(by[(a, s, t)], f'two_alpha_sd_med_{site}') for s in CNN_SEEDS for t in W21)))
            for a in (ARM_HOST, ARM_GROW)}
    sec['task_end_2alphaW'] = tw
    sec['task_end_2alphaW_note'] = ('unit_median_2alpha_sd = per-run median over units of 2*alpha_j*sd_j(z) (column '
                                    'two_alpha_sd_med; sd from the 1200-image end-of-task evaluation), then median over '
                                    'seeds.  product_2_alphamed_zsdmed = 2 * median_j(alpha) * median_j(sd), median over '
                                    'seeds.  stored_two_alpha_W_med_from_V uses sqrt(EMA V), not V_max, so in the grow '
                                    'arm it is not the alpha-times-width of the trained net (shown for reference).')
    sec['f2_seat'] = per_task_stat('seat_f2')
    sec['f2_mob'] = per_task_stat('mob_f2')
    sec['f2_past_valley'] = per_task_stat('past_valley_f2')
    sec['c1_width_zsd'] = per_task_stat('zsd_c1')
    sec['c1_alpha_med'] = per_task_stat('alpha_med_c1')
    sec['logit_sd'] = per_task_stat('logit_sd')
    sec['switch_ce_median'] = per_task_stat('switch_ce')
    sec['switch_ce_max'] = per_task_stat('switch_ce', fn=lambda v: float(np.max(v)))
    sec['zsd_f2_max'] = per_task_stat('zsd_f2', fn=lambda v: float(np.max(v)))
    sec['task_end_acc_min'] = {a: dict(memo_acc=float(min(fl(by[(a, s, t)], 'memo_acc') for s in CNN_SEEDS for t in CNN_TASKS)))
                               for a in (ARM_HOST, ARM_GROW)}
    # runaway (spec §4.1): switch CE > 100 accompanied by f2 width > 24.  switch_ce of task t is measured on the
    # network at the end of task t-1, so the primary width is zsd_f2 of row t-1 (the same network); the end of
    # task t is shown as a second reading.
    rw = {}
    for a in (ARM_HOST, ARM_GROW):
        ev_prim, ev_alt = [], []
        for s in CNN_SEEDS:
            for t in CNN_TASKS:
                ce = fl(by[(a, s, t)], 'switch_ce')
                if ce <= RUNAWAY_CE:
                    continue
                w_prev = fl(by[(a, s, t - 1)], 'zsd_f2') if t > 1 else float('nan')
                w_end = fl(by[(a, s, t)], 'zsd_f2')
                e = dict(seed=s, task=t, switch_ce=ce, zsd_f2_end_prev_task=w_prev, zsd_f2_end_this_task=w_end,
                         logit_sd_end_prev_task=fl(by[(a, s, t - 1)], 'logit_sd') if t > 1 else None)
                if w_prev > RUNAWAY_ZSD_F2:
                    ev_prim.append(e)
                if w_end > RUNAWAY_ZSD_F2:
                    ev_alt.append(e)
        n_ce = sum(fl(by[(a, s, t)], 'switch_ce') > RUNAWAY_CE for s in CNN_SEEDS for t in CNN_TASKS)
        n_w = sum(fl(by[(a, s, t)], 'zsd_f2') > RUNAWAY_ZSD_F2 for s in CNN_SEEDS for t in CNN_TASKS)
        rw[a] = dict(runaway=bool(ev_prim), n_events=len(ev_prim), seeds=sorted({e['seed'] for e in ev_prim}),
                     events=ev_prim,
                     alt_width_end_of_switched_task=dict(runaway=bool(ev_alt), n_events=len(ev_alt),
                                                         seeds=sorted({e['seed'] for e in ev_alt})),
                     n_rows_switch_ce_gt_100=n_ce, n_rows_zsd_f2_gt_24=n_w,
                     max_switch_ce=float(max(fl(by[(a, s, t)], 'switch_ce') for s in CNN_SEEDS for t in CNN_TASKS)),
                     max_zsd_f2=float(max(fl(by[(a, s, t)], 'zsd_f2') for s in CNN_SEEDS for t in CNN_TASKS)))
    sec['runaway'] = rw
    sec['runaway_definition'] = (f'(seed, task t) with switch_ce(t) > {RUNAWAY_CE:g} and zsd_f2 at the end of task t-1 '
                                 f'(the network that met the new labels) > {RUNAWAY_ZSD_F2:g}; alt: zsd_f2 at the end of task t')
    return sec


# --------------------------------------------------------------------------
# Q2 (MLP)
# --------------------------------------------------------------------------

def read_per_task_mlp(path):
    by = {}
    for r in csv.DictReader(Path(path).open()):
        by[(int(r['seed']), int(r['task']))] = r
    return by


def late_of(by):
    hard = sorted({t for (s, t), r in by.items() if 21 <= t <= 29 and int(r['hard']) == 1})
    assert tuple(hard) == LATE_HARD, ('hard tasks in 21-29', hard)
    return np.array([np.mean([float(by[(s, t)]['online_acc']) for t in LATE_HARD]) for s in MLP_SEEDS])


def read_ref_late():
    ref = {}
    for r in csv.DictReader(REF_PER_SEED.open()):
        ref.setdefault(r['arm'], {})[int(r['seed'])] = float(r['late'])
    return {a: np.array([ref[a][s] for s in MLP_SEEDS]) for a in ('SNA', 'SNA_fixT1', 'SNA_fix0')}


def layer_t29(by):
    out = {}
    for l in (1, 2):
        a = [float(by[(s, 29)][f'alpha_med_l{l}']) for s in MLP_SEEDS]
        z = [float(by[(s, 29)][f'zsd_l{l}']) for s in MLP_SEEDS]
        out[f'l{l}'] = dict(alpha_med=med(a), zsd=med(z),
                            two_alpha_zsd=med([2 * x * y for x, y in zip(a, z)]),
                            stored_two_alpha_W_med_from_V=med([float(by[(s, 29)][f'two_alpha_W_med_l{l}']) for s in MLP_SEEDS]))
    return out


def q2(status):
    out = dict(status=status)
    pt = MLP_DIR / 'per_task.csv'
    if not (MLP_DIR / 'done.json').exists() or not pt.exists():
        status['main_output'] = str(MLP_DIR.relative_to(ROOT)) + ' (no done.json)'
        out['label'] = 'NOT_RUN'
        return out, None
    by = read_per_task_mlp(pt)
    missing = [(s, t) for s in MLP_SEEDS for t in range(1, 31) if (s, t) not in by]
    status['complete'] = not missing
    status['missing'] = missing[:10]
    st = MLP_DIR.parent / 'status.json'
    status['run_status'] = json.loads(st.read_text()) if st.exists() else None
    if missing:
        out['label'] = 'INCOMPLETE'
        return out, by
    if not status.get('checks_all_pass'):
        out['label'] = 'CHECK_FAILED'
        return out, by
    g = late_of(by)
    ref = read_ref_late()
    # the reference column against its own per_task.csv (same late window)
    ref_ok = {}
    for a in ref:
        rb = read_per_task_mlp(REF_DIR / 'c2' / a / 'per_task.csv')
        ref_ok[a] = float(np.max(np.abs(late_of(rb) - ref[a])))
    status['reference_late_recomputed_max_abs_diff'] = ref_ok
    p = paired(g, ref['SNA'])
    M, low = p['mean'], p['low']
    out.update(label=q2_label(M, low), M=M, M_t95=[p['low'], p['high']], M_low=low, M_high=p['high'],
               M_sd=p['sd'], M_se=p['se'], n_positive=p['n_positive'], M_min=p['min'], M_max=p['max'],
               M_per_seed=dict(zip(MLP_SEEDS, p['per_seed'])),
               bands=dict(NO_LOSS=f'M >= {NO_LOSS_M} and 95% low >= {NO_LOSS_LOW}', LOSS=f'M <= {LOSS_M}',
                          UNRESOLVED='otherwise'),
               late_mean=dict(SNA_grow=float(g.mean()), **{a: float(v.mean()) for a, v in ref.items()}),
               late_per_seed=dict(SNA_grow=g.tolist(), **{a: v.tolist() for a, v in ref.items()}))
    return out, by


def q2_secondary(by):
    g = late_of(by)
    ref = read_ref_late()
    sec = {}
    for a in ('SNA_fixT1', 'SNA_fix0'):
        p = paired(g, ref[a])
        sec[f'SNA_grow-{a}'] = dict(mean=p['mean'], t95=[p['low'], p['high']], sd=p['sd'], n_positive=p['n_positive'],
                                   per_seed=dict(zip(MLP_SEEDS, p['per_seed'])))
    lay = {'SNA_grow': layer_t29(by)}
    for a in ('SNA', 'SNA_fixT1', 'SNA_fix0'):
        lay[a] = layer_t29(read_per_task_mlp(REF_DIR / 'c2' / a / 'per_task.csv'))
    sec['layers_t29_median_over_seeds'] = lay
    sec['layers_note'] = ('two_alpha_zsd = median over seeds of 2 * alpha_med * zsd (unit medians from the end-of-task '
                          'evaluation); stored_two_alpha_W_med_from_V uses the EMA V (not V_max / the frozen alpha).')
    return sec


# --------------------------------------------------------------------------
# tables (numbers only)
# --------------------------------------------------------------------------

def f(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'NA'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def tables(v):
    L = ['# sna_grow_1017 — verdict tables', '',
         f'Generated by `analysis/sna_grow_1017/verdict.py` ({v["generated"]}). Spec `specs/spec_sna_grow_1017.md` §4.', '']
    a1 = v['Q1']
    L += ['## Q1 (CNN, seeds 10–19)', '', f'Label: **{a1["label"]}**', '']
    if 'D' in a1:
        L += ['| quantity | value |', '|---|---:|',
              f'| W21 SNA (mean) | {f(a1["W21_mean"][ARM_HOST])} |',
              f'| W21 SNA+grow1 (mean) | {f(a1["W21_mean"][ARM_GROW])} |',
              f'| D | {f(a1["D"], 5, True)} |',
              f'| positive seeds | {a1["n_positive"]}/10 |',
              f'| D / 0.0073 | {f(a1["D_over_scale"], 3)} |',
              f'| SD of paired diff | {f(a1["D_sd"], 5)} |',
              f'| SE | {f(a1["D_se"], 5)} |',
              f'| t 95% (report only) | [{f(a1["D_t95_report_only"][0], 5, True)}, {f(a1["D_t95_report_only"][1], 5, True)}] |',
              f'| min / max | {f(a1["D_min"], 5, True)} / {f(a1["D_max"], 5, True)} |', '',
              '| seed | W21 SNA | W21 SNA+grow1 | diff |', '|---:|---:|---:|---:|']
        for i, s in enumerate(CNN_SEEDS):
            L.append(f'| {s} | {f(a1["W21_per_seed"][ARM_HOST][i])} | {f(a1["W21_per_seed"][ARM_GROW][i])} | '
                     f'{f(a1["D_per_seed"][s], 4, True)} |')
        L.append('')
    sec = v.get('Q1_secondary')
    if sec:
        w = sec['window_t11_20']
        L += ['## Q1 secondary (report only)', '', '### Window t11–20', '',
              '| SNA | SNA+grow1 | diff | t 95% | positive seeds |', '|---:|---:|---:|---|---:|',
              f'| {f(w["mean_SNA"])} | {f(w["mean_grow"])} | {f(w["diff"], 4, True)} | '
              f'[{f(w["t95"][0], 4, True)}, {f(w["t95"][1], 4, True)}] | {w["n_positive"]}/10 |', '']
        hdr = '| quantity | arm | ' + ' | '.join(f't{t}' for t in SHOW_TASKS) + ' |'
        sep = '|---|---|' + '---:|' * len(SHOW_TASKS)
        L += ['### By task (median over seeds unless noted)', '', hdr, sep]

        def row(name, d, dig=3):
            for a in (ARM_HOST, ARM_GROW):
                L.append(f'| {name} | {a} | ' + ' | '.join(f(d[a][t], dig) for t in SHOW_TASKS) + ' |')
        row('online (mean)', sec['online_mean_by_task'], 4)
        row('ep_first99', sec['ep_first99_median'], 1)
        for site in ('c2', 'f1', 'f2'):
            row(f'2α·sd {site} (unit median)', sec['task_end_2alphaW'][site]['unit_median_2alpha_sd'])
            row(f'2·α_med·zsd_med {site}', sec['task_end_2alphaW'][site]['product_2_alphamed_zsdmed'])
            row(f'stored 2αW from V {site}', sec['task_end_2alphaW'][site]['stored_two_alpha_W_med_from_V'])
            row(f'zsd {site}', sec['task_end_2alphaW'][site]['zsd'], 2)
            row(f'α_med {site}', sec['task_end_2alphaW'][site]['alpha_med'], 4)
        row('f2 seat', sec['f2_seat'])
        row('f2 mob', sec['f2_mob'])
        row('f2 past_valley', sec['f2_past_valley'])
        row('c1 width zsd', sec['c1_width_zsd'], 3)
        row('c1 α_med', sec['c1_alpha_med'], 3)
        row('logit_sd', sec['logit_sd'], 2)
        row('switch_ce', sec['switch_ce_median'], 2)
        row('switch_ce (max)', sec['switch_ce_max'], 2)
        row('zsd f2 (max)', sec['zsd_f2_max'], 2)
        L += ['', '2α·sd (unit median), tasks 21–30: median over seeds of the window mean [min, max over seed×task]', '',
              '| site | SNA | SNA+grow1 |', '|---|---|---|']
        for site in ('c2', 'f1', 'f2'):
            q = sec['task_end_2alphaW'][site]['unit_median_2alpha_sd_window21_30']
            L.append(f'| {site} | ' + ' | '.join(f'{f(q[a]["median_over_seeds_of_window_mean"], 3)} '
                                                 f'[{f(q[a]["min"], 3)}, {f(q[a]["max"], 3)}]' for a in (ARM_HOST, ARM_GROW)) + ' |')
        L += ['', f'ep_first99 never reached (rows): ' + ', '.join(f'{a} {n}' for a, n in sec['ep_first99_never_count'].items()), '',
              '### Runaway (switch_ce > 100 with zsd_f2 > 24 at the end of the previous task)', '',
              '| arm | runaway | events | seeds | rows switch_ce>100 | rows zsd_f2>24 | max switch_ce | max zsd_f2 | alt (width end of task t): events / seeds |',
              '|---|---|---:|---|---:|---:|---:|---:|---|']
        for a in (ARM_HOST, ARM_GROW):
            r = sec['runaway'][a]
            L.append(f'| {a} | {r["runaway"]} | {r["n_events"]} | {r["seeds"]} | {r["n_rows_switch_ce_gt_100"]} | '
                     f'{r["n_rows_zsd_f2_gt_24"]} | {f(r["max_switch_ce"], 2)} | {f(r["max_zsd_f2"], 2)} | '
                     f'{r["alt_width_end_of_switched_task"]["n_events"]} / {r["alt_width_end_of_switched_task"]["seeds"]} |')
        ev = [(a, e) for a in (ARM_HOST, ARM_GROW) for e in sec['runaway'][a]['events']]
        if ev:
            L += ['', '| arm | seed | task | switch_ce | zsd_f2 end t−1 | zsd_f2 end t | logit_sd end t−1 |', '|---|---:|---:|---:|---:|---:|---:|']
            for a, e in ev:
                L.append(f'| {a} | {e["seed"]} | {e["task"]} | {f(e["switch_ce"], 2)} | {f(e["zsd_f2_end_prev_task"], 2)} | '
                         f'{f(e["zsd_f2_end_this_task"], 2)} | {f(e["logit_sd_end_prev_task"], 2)} |')
        L.append('')
    a2 = v['Q2']
    L += ['## Q2 (MLP, seeds 0–9, late = hard tasks 21–29)', '', f'Label: **{a2["label"]}**', '']
    if 'M' in a2:
        L += ['| quantity | value |', '|---|---:|',
              f'| late SNA_grow (mean) | {f(a2["late_mean"]["SNA_grow"])} |',
              f'| late SNA (record, mean) | {f(a2["late_mean"]["SNA"])} |',
              f'| M | {f(a2["M"], 5, True)} |',
              f'| t 95% | [{f(a2["M_low"], 5, True)}, {f(a2["M_high"], 5, True)}] |',
              f'| positive seeds | {a2["n_positive"]}/10 |', '']
        s2 = v.get('Q2_secondary') or {}
        if s2:
            L += ['| secondary (report only) | mean | t 95% | positive seeds |', '|---|---:|---|---:|']
            for k in ('SNA_grow-SNA_fixT1', 'SNA_grow-SNA_fix0'):
                c = s2[k]
                L.append(f'| {k} | {f(c["mean"], 4, True)} | [{f(c["t95"][0], 4, True)}, {f(c["t95"][1], 4, True)}] | {c["n_positive"]}/10 |')
            L += ['', '| arm | α_med l1 | α_med l2 | zsd l1 | zsd l2 | 2α·zsd l1 | 2α·zsd l2 | stored 2αW(V) l1 | l2 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for a, d in s2['layers_t29_median_over_seeds'].items():
                L.append(f'| {a} | {f(d["l1"]["alpha_med"])} | {f(d["l2"]["alpha_med"])} | {f(d["l1"]["zsd"], 3)} | '
                         f'{f(d["l2"]["zsd"], 3)} | {f(d["l1"]["two_alpha_zsd"], 3)} | {f(d["l2"]["two_alpha_zsd"], 3)} | '
                         f'{f(d["l1"]["stored_two_alpha_W_med_from_V"], 3)} | {f(d["l2"]["stored_two_alpha_W_med_from_V"], 3)} |')
            L.append('')
    else:
        L += [f'Status: `{json.dumps(clean(a2["status"]), ensure_ascii=False)}`', '']
    mech = v.get('cnn_mechanism')
    if mech:
        L += ['## Mechanism replay (task 20 → 21; see cnn_mechanism.json)', '', mech, '']
    return '\n'.join(L) + '\n'


def mech_table():
    p = RES / 'cnn_mechanism.json'
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m.get('table_md')


def main():
    _selftest()
    st1 = dict(checks_file=str(CNN_CHECKS.relative_to(ROOT)))
    if CNN_CHECKS.exists():
        ck = json.loads(CNN_CHECKS.read_text())
        st1['checks_all_pass'] = bool(ck.get('all_pass'))
        st1['checks_this_file_sha256'] = ck.get('this_file_sha256')
    pv = RES / 'cnn' / 'provenance_sna_grow.json'
    if pv.exists():
        pr = json.loads(pv.read_text())
        st1['run_this_file_sha256'] = pr.get('this_file_sha256')
        st1['run_git_hash'] = pr.get('git_hash')
        st1['run_git_dirty_src'] = pr.get('git_dirty_src')
        st1['run_finished'] = pr.get('finished')
        st1['run_engine_sha256'] = pr.get('engine_sha256')
        st1['checks_and_run_same_source'] = st1.get('checks_this_file_sha256') == st1['run_this_file_sha256']
    if CNN_CSV.exists():
        st1['per_task_sha256'] = sha(CNN_CSV)
    v1, by = q1(st1)
    st2 = dict(checks_file=str(MLP_CHECKS.relative_to(ROOT)))
    if MLP_CHECKS.exists():
        ck = json.loads(MLP_CHECKS.read_text())
        st2['checks_all_pass'] = bool(ck.get('all_pass'))
        st2.update({f'{k}_all_pass': (ck.get(k) or {}).get('all_pass') for k in ('K-mlp-rep', 'K-mlp-grow', 'K-mlp-graph')})
    st2['reference_per_seed_sha256'] = sha(REF_PER_SEED)
    v2, by2 = q2(st2)
    out = dict(run_id=RUN, spec='specs/spec_sna_grow_1017.md', script='analysis/sna_grow_1017/verdict.py',
               generated=time.strftime('%Y-%m-%d %H:%M:%S'), Q1=v1, Q2=v2,
               labels=dict(Q1=v1['label'], Q2=v2['label']))
    if by is not None and 'D' in v1:
        out['Q1_secondary'] = q1_secondary(by)
    if by2 is not None and 'M' in v2:
        out['Q2_secondary'] = q2_secondary(by2)
    out['cnn_mechanism'] = mech_table()
    put(OUT_JSON, out)
    OUT_MD.write_text(tables(out))
    print(json.dumps(dict(Q1=v1['label'], D=v1.get('D'), n_positive=v1.get('n_positive'),
                          Q2=v2['label'], M=v2.get('M'), M_low=v2.get('M_low')), indent=1))
    print(f'-> {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
