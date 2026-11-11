#!/usr/bin/env python3
"""Summarize the registered dynamic experiment after all main cells complete.

Run from the worktree using proj_004_drift/.venv/bin/python. No torch import or
training. Default validation requires the full fixed design; --smoke is an
explicit schema check and writes only dynamic_smoke_* artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'results/loading_bridge_1111'
CELL = ['model', 'activation', 't_end']
IDENT = CELL + ['seed', 'arm']
PAIRS = [('T', 'TG'), ('S', 'G'), ('H10', 'T'), ('P', 'S'), ('Q', 'S')]
REC_PAIRS = [('S', 'B0'), ('P', 'B0'), ('H10', 'B0'), ('TG', 'B0'),
             ('P', 'S'), ('H10', 'S'), ('TG', 'S')]
ARMS = ['S', 'P', 'Q', 'H10', 'T', 'G', 'TG', 'H0']
COLORS = dict(zip(ARMS, ['#222222', '#0072B2', '#009E73', '#D55E00',
                        '#CC79A7', '#E69F00', '#56B4E9', '#777777']))
COLORS['B0'] = '#6F3C87'
LAYER_METRICS = ['m_med', 's_med', 'r_med', 'opening_mean', 'dphi_mean',
                 'upper_max_med', 'upper_over_s_med', 'upper_q99_med',
                 'body_weight_norm']
BOOT = 10000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design():
    expected = set()
    for act in ['R', 'SILU', 'ELU']:
        for seed in range(10):
            for t in [1, 10]:
                expected.update(('mlp', act, t, seed, a) for a in ['B0'] + ARMS)
    for seed in range(10252010, 10252015):
        expected.update(('vit', 'SL', 100, seed, a)
                        for a in ['B0'] + [a for a in ARMS if a != 'Q'])
    return expected


def read_inputs(inputs, smoke=False):
    paths = sorted(p for d in inputs for p in d.glob('*.json'))
    if not paths:
        raise ValueError('No JSON input files; main may still be running.')
    records, seen = [], set()
    for path in paths:
        r = json.loads(path.read_text())
        key = tuple(r[k] for k in IDENT)
        if key in seen:
            raise ValueError(f'Duplicate experimental cell: {key}')
        seen.add(key)
        if r.get('complete') is not True:
            raise ValueError(f'Incomplete file: {path}')
        args = r['provenance']['args']
        if not smoke:
            if 'smoke' in str(path).lower() or args['steps'] != 200:
                raise ValueError(f'Non-main input: {path}')
            if not args['recovery']:
                raise ValueError(f'Main requires recovery enabled: {path}')
            if r['model'] == 'mlp' and r['activation'] == 'ELU' and not args['natural']:
                raise ValueError(f'MLP main requires natural enabled: {path}')
        if r['arm'] != 'B0':
            snaps, track = r['acute']['snapshots'], r['acute']['track']
            n = args['steps']
            if [x['step'] for x in track] != list(range(1, n + 1)):
                raise ValueError(f'Missing/duplicated acute steps: {path}')
            actual = {s['step'] for s in snaps}
            required = {0, n} if smoke else {0, 1, 5, 25, 50, 100, 200}
            if actual != required and (not smoke or not required.issubset(actual)):
                raise ValueError(f'Unexpected snapshot steps: {path}: {actual}')
            if len(actual) != len(snaps):
                raise ValueError(f'Duplicate snapshots: {path}')
        requires_rec = r['arm'] in ['B0', 'S', 'P', 'H10', 'TG']
        if not smoke and requires_rec != ('recovery' in r):
            raise ValueError(f'Unexpected/missing recovery: {path}')
        if 'recovery' in r:
            rec = r['recovery']
            if rec['censored'] != (rec['T90'] is None):
                raise ValueError(f'Inconsistent censoring: {path}')
            curve = rec['curve']
            if curve[-1]['step'] != rec['steps_run']:
                raise ValueError(f'Recovery endpoint mismatch: {path}')
            reached = [s['step'] for s in curve if s['acc'] >= .9]
            if rec['T90'] is not None and rec['T90'] != min(reached):
                raise ValueError(f'T90 mismatch: {path}')
            if rec['T90'] is None and reached:
                raise ValueError(f'Censored run reached threshold: {path}')
            budget = 4000 if r['model'] == 'mlp' else 1500
            if not smoke:
                every = 100 if r['model'] == 'mlp' else 25
                stop = rec['steps_run']
                if not isinstance(stop, int) or not (1 <= stop <= budget) or stop % every:
                    raise ValueError(f'Recovery stopping time outside budget: {path}')
                expected_grid = list(range(0, stop + 1, every))
                if [s['step'] for s in curve] != expected_grid:
                    raise ValueError(f'Missing/duplicated recovery grid point: {path}')
                if not rec['censored'] and rec['T90'] != stop:
                    raise ValueError(f'Recovery continued past T90: {path}')
                if rec['censored'] and stop != budget:
                    raise ValueError(f'Unexpected censor horizon: {path}')
        if not smoke and r['model'] == 'mlp' and r['activation'] == 'ELU' and r['arm'] in ['S', 'Q']:
            if [s['step'] for s in r.get('natural', [])] != [1000, 4000]:
                raise ValueError(f'Missing ELU natural continuation: {path}')
        r['_path'] = str(path)
        records.append(r)
    if not smoke:
        missing, extra = design() - seen, seen - design()
        if missing or extra:
            raise ValueError(f'Main design incomplete: expected 580, found {len(seen)}; '
                             f'missing={len(missing)} extra={len(extra)}; '
                             f'first missing={sorted(missing)[:8]}')
    return records


def numbers(snapshot):
    for name, value in snapshot.items():
        if name not in ['step', 'layers'] and isinstance(value, (int, float)):
            yield -1, name, float(value)
    for i, layer in enumerate(snapshot.get('layers', [])):
        for name, value in layer.items():
            yield i, name, float(value)


def frames(records):
    snaps, tracks, paths, recoveries = [], [], [], []
    for r in records:
        ident = {k: r[k] for k in IDENT}
        baseline = {}
        if 'acute' in r:
            baseline = {(l, m): v for l, m, v in numbers(r['acute']['snapshots'][0])}
        for phase, data in [('acute', r.get('acute', {}).get('snapshots', [])),
                            ('natural', r.get('natural', [])),
                            ('recovery', r.get('recovery', {}).get('curve', []))]:
            for s in data:
                for layer, metric, value in numbers(s):
                    row = dict(ident, phase=phase, step=s['step'], layer=layer,
                               metric=metric, value=value)
                    snaps.append(row)
                    if phase != 'recovery' and (layer, metric) in baseline and (
                            layer >= 0 or metric in ['head_centered_norm', 'feature_centered_rms',
                                                     'feature_mean_norm', 'logit_centered_rms']):
                        snaps.append(dict(row, metric='delta_' + metric,
                                          value=value - baseline[layer, metric]))
        track = r.get('acute', {}).get('track', [])
        tracks.extend(dict(ident, **x) for x in track)
        for first, last in [(1, 25), (26, 200)]:
            part = [x for x in track if first <= x['step'] <= last]
            if len(part) != last - first + 1:
                continue
            vals = dict(body_path=sum(x['d_body'] for x in part),
                        head_path=sum(x['d_head'] for x in part),
                        mean_actual_body_step=np.mean([x['d_body'] for x in part]),
                        mean_raw_minibatch_body_grad=np.mean([x['g_body'] for x in part]),
                        mean_adam_v_rms=np.mean([x['v_body'] for x in part]),
                        peak_actual_body_step=max(x['d_body'] for x in part),
                        peak_raw_minibatch_body_grad=max(x['g_body'] for x in part))
            paths.extend(dict(ident, interval=f'{first}-{last}', metric=k, value=v)
                         for k, v in vals.items())
        if 'recovery' in r:
            rr = r['recovery']; end = rr['curve'][-1]
            recoveries.append(dict(ident, T90=rr['T90'], censored=rr['censored'],
                                   reached=not rr['censored'], steps_run=rr['steps_run'],
                                   final_acc=end['acc'], final_ce=end['ce'],
                                   budget=4000 if r['model'] == 'mlp' else 1500))
    return (pd.DataFrame(snaps), pd.DataFrame(tracks), pd.DataFrame(paths),
            pd.DataFrame(recoveries))


def bootstrap(values, key):
    """Percentile CI for a seed mean; one resampling unit is one paired seed."""
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Nonfinite/empty bootstrap values')
    seed = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], 'little')
    idx = np.random.default_rng(seed).integers(0, len(a), size=(BOOT, len(a)))
    bs = a[idx].mean(axis=1)
    lo, hi = np.quantile(bs, [.025, .975])
    return dict(n=len(a), mean=float(a.mean()), ci_low=float(lo), ci_high=float(hi))


def summarize(df, keys):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, sub in df.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        if sub['seed'].duplicated().any():
            raise ValueError(f'Duplicate seed in summary group {key}')
        rows.append(dict(zip(keys, key), **bootstrap(sub['value'], ('mean', key))))
    return pd.DataFrame(rows)


def paired(df, keys, comparisons=PAIRS):
    rows, per_seed = [], []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for key, sub in df.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        tab = sub.pivot(index='seed', columns='arm', values='value')
        for a, b in comparisons:
            if a not in tab or b not in tab:
                continue
            ab = tab[[a, b]].dropna()
            if ab.empty:
                continue
            meta = dict(zip(keys, key), arm_a=a, arm_b=b, comparison=f'{a}-{b}')
            delta = ab[a] - ab[b]
            rows.append(dict(meta, estimand='mean_paired_difference',
                             **bootstrap(delta, ('diff', key, a, b))))
            safe = ab[b].abs() >= 1e-10
            ratio = ab.loc[safe, a] / ab.loc[safe, b]
            ratio_metrics = {'ce','confidence','expected_feature_grad_rms','body_displacement',
                             'head_centered_norm','feature_centered_rms','feature_mean_norm',
                             'logit_centered_rms','s_med','dphi_mean','opening_mean',
                             'body_weight_norm','body_path','head_path','mean_actual_body_step',
                             'mean_raw_minibatch_body_grad','mean_adam_v_rms','peak_actual_body_step',
                             'peak_raw_minibatch_body_grad','final_acc','final_ce','actual_body_step_update1',
                             'actual_head_step_update1','raw_body_gradient_update1','adam_v_rms_update1'}
            metric = meta.get('metric', '')
            if len(ratio) and metric in ratio_metrics:
                rows.append(dict(meta, estimand='mean_seed_ratio',
                                 **bootstrap(ratio, ('ratio', key, a, b))))
            for seed, p in ab.iterrows():
                den = p[b]
                per_seed.append(dict(meta, seed=int(seed), value_a=p[a], value_b=p[b],
                                     difference=p[a]-p[b], ratio=p[a]/den if abs(den)>=1e-10 and metric in ratio_metrics else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


def recovery_stats(rec):
    if rec.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    long = rec.melt(id_vars=IDENT, value_vars=['reached', 'final_acc', 'final_ce', 'steps_run'],
                    var_name='metric', value_name='value')
    long['value'] = long['value'].astype(float)
    summary = summarize(long, CELL + ['arm', 'metric'])
    pairs, seeds = paired(long, CELL + ['metric'], REC_PAIRS)
    timing = []
    for key, sub in rec.groupby(CELL, sort=True):
        for a, b in REC_PAIRS:
            aa = sub[sub.arm == a].set_index('seed'); bb = sub[sub.arm == b].set_index('seed')
            for seed in sorted(set(aa.index) & set(bb.index)):
                x, y = aa.loc[seed], bb.loc[seed]
                status = ('both_observed' if x.reached and y.reached else
                          'a_censored' if not x.reached and y.reached else
                          'b_censored' if x.reached and not y.reached else 'both_censored')
                diff = x.T90-y.T90 if status == 'both_observed' else np.nan
                lower = x.steps_run-y.T90 if status == 'a_censored' else np.nan
                upper = x.T90-y.steps_run if status == 'b_censored' else np.nan
                timing.append(dict(zip(CELL,key), seed=seed, arm_a=a, arm_b=b,
                                   comparison=f'{a}-{b}', status=status, T90_a=x.T90, T90_b=y.T90,
                                   observed_difference=diff, strict_lower_bound=lower,
                                   strict_upper_bound=upper))
    timing = pd.DataFrame(timing)
    # T90 contrasts are estimated only if every paired seed reaches in both arms.
    # No complete-case effect is promoted to an all-seed result.
    extra = []
    for key, sub in timing.groupby(CELL + ['comparison','arm_a','arm_b'], sort=True):
        if (sub.status == 'both_observed').all():
            extra.append(dict(zip(CELL + ['comparison','arm_a','arm_b'],key), metric='T90',
                              estimand='mean_paired_difference',
                              **bootstrap(sub.observed_difference, ('T90',key))))
    if extra:
        pairs = pd.concat([pairs, pd.DataFrame(extra)], ignore_index=True)
    return summary, pairs, timing


def save_csv(df, path):
    df.to_csv(path, index=False, float_format='%.12g')


def plot_series(ax, data, metric, layer=-1, arms=ARMS):
    sub = data[(data.metric == metric) & (data.layer == layer)]
    for arm in arms:
        s = sub[sub.arm == arm].groupby('step').value.agg(['mean','std','count'])
        if s.empty:
            continue
        err = s['std'].fillna(0) / np.sqrt(s['count'])
        ax.plot(s.index, s['mean'], color=COLORS[arm], lw=1.6, label=arm)
        ax.fill_between(s.index, s['mean']-err, s['mean']+err, color=COLORS[arm], alpha=.10)
    ax.set_title(metric.replace('_', ' '), fontsize=9)
    ax.grid(alpha=.18)


def finish_figure(fig, out, name):
    if name.startswith('dynamic_smoke'):
        fig.text(.5, -.02, 'SMOKE: schema check only; no experimental conclusion',
                 ha='center', color='#9B2226', fontsize=11)
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(snap, paths, rec, out, prefix):
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    acute = snap[snap.phase == 'acute']
    for key, data in acute.groupby(CELL, sort=True):
        fig, axs = plt.subplots(2,3, figsize=(11,6), constrained_layout=True)
        for ax, metric in zip(axs.flat, ['ce','confidence','expected_feature_grad_rms',
                                       'body_displacement','r_med','dphi_mean']):
            layer = 0 if metric in LAYER_METRICS else -1
            plot_series(ax, data, metric, layer)
            ax.set_xlabel('Acute updates')
        axs[0,0].legend(ncol=4, fontsize=8)
        fig.suptitle(f'{key[0].upper()} {key[1]} task {key[2]} switch | mean ± SE across seeds\n'
                     'Expected feature gradient and actual body displacement have different units.',fontsize=11)
        finish_figure(fig,out,f'{prefix}_acute_{key[0]}_{key[1]}_t{key[2]}')
    mlp = acute[acute.model == 'mlp']
    groups = list(mlp.groupby(CELL, sort=True))
    if groups:
        fig, axs = plt.subplots(len(groups),6,figsize=(15,2.15*len(groups)),squeeze=False,
                                constrained_layout=True)
        for row,(key,data) in zip(axs,groups):
            for ax,metric in zip(row,['m_med','s_med','r_med','dphi_mean','upper_over_s_med','body_weight_norm']):
                plot_series(ax,data,metric,0,arms=['S','P','Q','H10','T','G','TG'])
            row[0].set_ylabel(f'{key[1]} task {key[2]}')
        axs[0,0].legend(ncol=4,fontsize=7)
        fig.suptitle('MLP body state | full training set | mean ± SE across seeds')
        finish_figure(fig,out,prefix+'_mlp_state')
    elu = snap[(snap.model=='mlp') & (snap.activation=='ELU') & snap.arm.isin(['S','Q'])
               & snap.phase.isin(['acute','natural'])]
    if not snap[snap.phase=='natural'].empty:
        groups=list(elu.groupby(CELL,sort=True))
        fig,axs=plt.subplots(len(groups),6,figsize=(15,2.7*len(groups)),squeeze=False,
                             constrained_layout=True)
        for row,(key,data) in zip(axs,groups):
            for ax,metric in zip(row,['feature_centered_rms','logit_centered_rms','s_med',
                                     'r_med','dphi_mean','upper_over_s_med']):
                plot_series(ax,data,metric,0 if metric in LAYER_METRICS else -1,arms=['S','Q'])
                ax.set_xscale('symlog',linthresh=25);ax.set_xlabel('New-task updates')
            row[0].set_ylabel(f'ELU task {key[2]}')
        axs[0,0].legend()
        fig.suptitle('ELU natural continuation after a single Q intervention | mean ± SE')
        finish_figure(fig,out,prefix+'_elu_natural')
    if not paths.empty:
        groups=list(paths.groupby(CELL,sort=True));fig,axs=plt.subplots(2,4,figsize=(13,6),
                                                                    constrained_layout=True)
        for ax,(key,data) in zip(axs.flat,groups):
            part=data[data.metric=='body_path']
            for j,interval in enumerate(['1-25','26-200']):
                dd=part[part.interval==interval].groupby('arm').value.agg(['mean','std','count'])
                arms=[a for a in ARMS if a in dd.index];xx=np.arange(len(arms))+(j-.5)*.34
                ax.bar(xx,dd.loc[arms,'mean'],width=.34,label=interval,alpha=.8)
                ax.errorbar(xx,dd.loc[arms,'mean'],yerr=dd.loc[arms,'std'].fillna(0)/np.sqrt(dd.loc[arms,'count']),
                            fmt='none',color='black',lw=.7,capsize=2)
            ax.set_xticks(np.arange(len(arms)),arms,rotation=45);ax.set_title(f'{key[0]} {key[1]} t{key[2]}')
            ax.set_ylabel('Actual body path length')
        for ax in list(axs.flat)[len(groups):]:ax.axis('off')
        axs.flat[0].legend(title='Updates');fig.suptitle('Path length is a sum of actual Adam step norms | mean ± SE')
        finish_figure(fig,out,prefix+'_paths')
    if not rec.empty:
        groups=list(rec.groupby(CELL,sort=True));fig,axs=plt.subplots(2,4,figsize=(13,6),
                                                                  constrained_layout=True)
        arms=['B0','S','P','H10','TG']
        for ax,(key,data) in zip(axs.flat,groups):
            for i,arm in enumerate(arms):
                dd=data[data.arm==arm].sort_values('seed');jitter=np.linspace(-.16,.16,len(dd))
                for (idx,r),off in zip(dd.iterrows(),jitter):
                    val=r.T90 if r.reached else r.budget
                    ax.scatter(i+off,val,color=COLORS[arm],marker='o' if r.reached else '^',s=20)
            ax.set_xticks(range(len(arms)),arms);ax.set_title(f'{key[0]} {key[1]} t{key[2]}')
            ax.set_ylabel('Recovery updates');ax.grid(axis='y',alpha=.18)
        for ax in list(axs.flat)[len(groups):]:ax.axis('off')
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color='black',label='Observed T90'),
                            Line2D([],[],marker='^',ls='',color='black',label='T90 > budget (not an event)')],
                   loc='outside lower center',ncol=2)
        fig.suptitle('Standardized body recovery | each marker is one seed; readout and all Adam state reset')
        finish_figure(fig,out,prefix+'_recovery')



def elu_t10_figure(snap,out,prefix):
    data=snap[(snap.model=='mlp') & (snap.activation=='ELU') & (snap.t_end==10)
              & snap.arm.isin(['S','Q']) & snap.phase.isin(['acute','natural'])]
    if data.empty or data.step.max()<4000:return
    metrics=['m_med','s_med','r_med','dphi_mean','upper_max_med','upper_q99_med',
             'upper_over_s_med','body_weight_norm','head_centered_norm',
             'feature_centered_rms','logit_centered_rms','confidence']
    fig,axs=plt.subplots(3,4,figsize=(13,8),constrained_layout=True)
    for ax,metric in zip(axs.flat,metrics):
        plot_series(ax,data,metric,0 if metric in LAYER_METRICS else -1,arms=['S','Q'])
        ax.set_xscale('symlog',linthresh=25);ax.set_xlabel('New-task updates')
        ax.axvline(200,color='#888888',lw=.7,ls=':')
    axs[0,0].legend(title='Single switch intervention')
    fig.suptitle('ELU task-end 10 → task 11: activity, width/location/tail, scale and head regrowth | n=10, mean ± SE')
    finish_figure(fig,out,prefix+'_elu_t10_state_and_regrowth')


def first_update_figure(pairs, out, prefix):
    if pairs.empty:
        return
    sub=pairs[(pairs.estimand=='mean_seed_ratio') & pairs.metric.isin(
        ['raw_body_gradient_update1','actual_body_step_update1'])]
    cells=sorted({tuple(x) for x in sub[CELL].itertuples(index=False,name=None)})
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    for ax,(a,b,expected) in zip(axs,[('T','TG',10.),('S','G',10.),('H10','T',.1)]):
        for metric,offset,color,label in [('raw_body_gradient_update1',-.10,'#0072B2','Raw body gradient'),
                                        ('actual_body_step_update1',.10,'#D55E00','Actual Adam body step')]:
            part=sub[(sub.arm_a==a)&(sub.arm_b==b)&(sub.metric==metric)]
            for i,key in enumerate(cells):
                row=part[(part.model==key[0])&(part.activation==key[1])&(part.t_end==key[2])]
                if row.empty:continue
                r=row.iloc[0]
                ax.errorbar(i+offset,r['mean'],yerr=[[max(0,r['mean']-r.ci_low)],
                                                   [max(0,r.ci_high-r['mean'])]],
                            fmt='o',color=color,capsize=3,label=label if i==0 else None)
        ax.axhline(1,color='#888888',lw=.8)
        ax.axhline(expected,color='#555555',lw=.9,ls='--')
        ax.set_yscale('log');ax.set_title(f'{a} / {b} at update 1')
        labels=[]
        for c in cells:
            cc=sub[(sub.arm_a==a)&(sub.arm_b==b)&(sub.model==c[0])&
                   (sub.activation==c[1])&(sub.t_end==c[2])].set_index('metric')
            nr=int(cc.loc['raw_body_gradient_update1','n']) if 'raw_body_gradient_update1' in cc.index else 0
            na=int(cc.loc['actual_body_step_update1','n']) if 'actual_body_step_update1' in cc.index else 0
            model='ViT ' if c[0]=='vit' else ''
            labels.append(f'{model}{c[1]} t{c[2]}\nn={nr}/{na}')
        ax.set_xticks(range(len(cells)),labels,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('Mean paired-seed ratio [95% bootstrap CI]');ax.grid(axis='y',alpha=.15)
    axs[0].legend(fontsize=8)
    fig.suptitle('Same initial probability aggregates; raw-gradient factors need not equal Adam-step factors\n'
                 'n = valid raw-gradient / Adam-step seed counts; near-zero denominators excluded')
    finish_figure(fig,out,prefix+'_first_update_ratios')


def fmt(v):
    if pd.isna(v):return 'NA'
    return f'{v:.5g}'


def markdown_table(df, columns):
    if df.empty:return '_No eligible observations._\n'
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in df.iterrows():
        lines.append('| '+' | '.join(fmt(r[c]) if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return '\n'.join(lines)+'\n'


def report(snap_sum,path_sum,snap_pairs,path_pairs,rec,rec_pairs,first_pairs,init,out,prefix,smoke):
    lines=['# Dynamic loading bridge results',
           '**SMOKE SCHEMA CHECK ONLY; no experimental conclusion.**' if smoke else
           'Complete fixed design: 540 MLP files and 40 ViT files; no smoke files included.',
           'Bootstrap: 10,000 paired-seed resamples, percentile 95% intervals; strata are model/activation/task. '
           'The ViT has only 5 seeds. Intervals are descriptive and not multiplicity corrected.',
           'Forward probabilities match at update 0 for T/TG, S/G and H10/T by construction. '
           'The stored CE/confidence/entropy aggregates and expected-gradient factors are checked in '
           'dynamic_initial_match_checks.csv. This does not establish matching probabilities through update 200. '
           'P/Q are single switch interventions; this is not sustained prevention.',
           'Snapshot expected_feature_grad_rms is the per-sample uniform-target gradient at the head input (RMS). '
           'Track g_body is the actual minibatch raw body-parameter gradient (L2). d_body/body_path/body_displacement '
           'are actual Adam parameter movements. Their ratios must not be equated.',
           'MLP layer statistics use all 1,200 training inputs. ViT layer statistics use the first 128 training images '
           'and all tokens; layer 0 is the first FFN probe. CSV files retain every measured layer. '
           'upper_over_s_med is median(max preactivation / s), not (max−median)/s. '
           'It includes location changes; centered tail shape is not separately identified.',
           'CSV delta_* values subtract the same arm’s post-intervention update-0 state. '
           'Ratios exclude denominators with absolute value below 1e-10; n records the eligible seed count.',
           'Trajectory/path figures show mean ± SE; the first-update ratio figure and CSV summaries use '
           'bootstrap 95% intervals. Recovery figures show individual seeds and censor marks.',
           '## Update 1: T/TG raw gradient versus actual body movement',
           'The two columns are means of paired seedwise ratios, each with its own eligible n. '
           'Near-zero raw gradients can coexist with movement from inherited Adam state.']
    ratios=first_pairs[(first_pairs.comparison=='T-TG') &
                       (first_pairs.estimand=='mean_seed_ratio')]
    rows=[]
    for key,sub in ratios.groupby(CELL,sort=True):
        row=dict(zip(CELL,key))
        for metric,name in [('raw_body_gradient_update1','raw_gradient'),
                            ('actual_body_step_update1','Adam_body_step')]:
            selected=sub[sub.metric==metric]
            if selected.empty:continue
            r=selected.iloc[0]
            row[name+'_ratio_95CI']=f"{fmt(r['mean'])} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]"
            row[name+'_n']=int(r.n)
        rows.append(row)
    table=pd.DataFrame(rows)
    lines += [markdown_table(table,list(table.columns)),
              'All registered pairs T/TG, S/G, H10/T, P/S and Q/S remain in dynamic_first_update_paired.csv '
              '(raw body gradient and actual Adam movement reported separately).',
              '## Standardized body recovery',
              'T90 is recorded only when observed on the evaluation grid (MLP every 100; ViT every 25 updates). '
              'Unreached values stay blank, with censor horizon recorded separately. No budget is substituted for T90. '
              'All-seed T90 contrasts receive a finite mean/CI only when both arms reach in every paired seed. '
              'Otherwise dynamic_recovery_t90_pairs.csv records censoring and one-sided pairwise bounds.',
              'Recovery resets the readout and all Adam state. T90 excludes the preceding 200 acute updates '
              '(B0 has no acute phase), so H10/TG below B0 is not a comparison of total learning updates. '
              'It stops at the first observed accuracy ≥0.90. '
              'Final accuracy is measured at each run’s own stopping step, not at a common training horizon; '
              'report it together with steps_run. This probes body state including beneficial new-task learning, '
              'and does not by itself identify irreversible damage. '
              'The ViT S−B0 interval spans zero and does not support a common acute body-damage effect '
              'across seeds. H10/TG below B0 can include useful acute learning of the new task.']
    if not rec.empty:
        vit=rec[rec.model=='vit'].copy()
        if not vit.empty:
            vit['T90_or_bound']=vit.apply(lambda r: f'>{int(r.budget)}' if r.censored else str(int(r.T90)),axis=1)
            vt=vit.pivot(index='seed',columns='arm',values='T90_or_bound').reset_index()
            cols=['seed']+[a for a in ['B0','S','P','H10','TG'] if a in vt]
            lines += ['### ViT: every seed (5 seeds only)', markdown_table(vt,cols)]
            vp=rec_pairs[(rec_pairs.model=='vit') & (rec_pairs.metric=='T90')]
            lines += ['ViT all-seed paired T90 differences (first arm minus second; negative is faster):',
                      markdown_table(vp,['comparison','n','mean','ci_low','ci_high'])]
            vend=vit.groupby('arm').agg(n=('seed','count'),reached=('reached','sum'),
                censored=('censored','sum'),final_accuracy_mean=('final_acc','mean'),
                stopping_step_mean=('steps_run','mean')).reset_index()
            lines += ['Accuracy and stopping time must be read together:',markdown_table(vend,list(vend.columns))]
        late=rec[(rec.model=='mlp') & (rec.t_end==10)]
        if not late.empty:
            lr=late.groupby(['activation','arm']).agg(n=('seed','count'),reached=('reached','sum'),
                censored_at_4000=('censored','sum'),final_accuracy_mean=('final_acc','mean'),
                stopping_step_mean=('steps_run','mean')).reset_index()
            lines += ['### MLP late switch: 90% unreached counts at the 4,000-update limit',
                      markdown_table(lr,list(lr.columns))]
    lines += ['## ELU task-end 10 → task 11: single Q intervention followed to update 4,000',
              'These are arm means, not paired differences. Activity, width, location, upper tail, feature scale, '
              'logit scale and readout regrowth are separate measurements. Q is applied once; later changes '
              'do not establish a fixed-head mechanism. Seed-level values and paired bootstrap intervals '
              'are in dynamic_snapshots.csv and dynamic_snapshot_paired.csv.']
    elu=snap_sum[(snap_sum.model=='mlp') & (snap_sum.activation=='ELU') & (snap_sum.t_end==10) &
        snap_sum.arm.isin(['S','Q']) & snap_sum.step.isin([0,200,1000,4000]) &
        snap_sum.phase.isin(['acute','natural'])]
    metric_groups=[['m_med','s_med','r_med','dphi_mean'],
                   ['upper_max_med','upper_q99_med','upper_over_s_med','body_weight_norm'],
                   ['head_centered_norm','feature_centered_rms','logit_centered_rms']]
    for metrics in metric_groups:
        part=elu[elu.metric.isin(metrics)]
        if part.empty:continue
        table=part.pivot(index=['step','arm','n'],columns='metric',values='mean').reset_index()
        table=table.sort_values(['step','arm'])
        lines += [markdown_table(table,['step','arm','n']+[m for m in metrics if m in table])]
    lines += ['## Full-resolution outputs',
              '- dynamic_snapshots.csv: every arm/seed/step/layer, including natural and recovery curves.\n'
              '- dynamic_track.csv: every actual acute update.\n'
              '- dynamic_paths.csv: complete 1–25 and 26–200 windows only.\n'
              '- dynamic_*_summary.csv: arm means and seed-bootstrap intervals.\n'
              '- dynamic_*_paired.csv: paired differences and mean of valid seedwise ratios.\n'
              '- dynamic_*_paired_seeds.csv: auditable seed-level values and contrasts.\n'
              '- dynamic_recovery.csv / dynamic_recovery_t90_pairs.csv: censor-aware endpoints and pairwise bounds.\n'
              '- dynamic_provenance.json: input hashes, script/spec hashes and producing commits.']
    (out/(prefix+'_summary.md')).write_text('\n\n'.join(lines)+'\n')

def selftest():
    a=bootstrap([2.,2.,2.],('constant',))
    assert a['mean']==a['ci_low']==a['ci_high']==2.
    rows=[]
    for seed in range(3):
        for arm in ['T','TG']:
            rows.append(dict(model='test',activation='X',t_end=1,seed=seed,arm=arm,
                             metric='ce',value=seed+(2 if arm=='T' else 0)))
    pp,sp=paired(pd.DataFrame(rows),CELL+['metric'])
    d=pp[pp.estimand=='mean_paired_difference'].iloc[0]
    assert d['mean']==d.ci_low==d.ci_high==2 and len(sp)==3
    assert pp[pp.estimand=='mean_seed_ratio'].iloc[0]['n']==2
    rr=[]
    for seed in range(2):
        for arm in ['S','B0']:
            cens=arm=='S' and seed==1
            rr.append(dict(model='test',activation='X',t_end=1,seed=seed,arm=arm,T90=None if cens else 100,
                           censored=cens,reached=not cens,steps_run=4000 if cens else 100,
                           final_acc=.5 if cens else .91,final_ce=1.,budget=4000))
    _,pairs,timing=recovery_stats(pd.DataFrame(rr))
    assert not (pairs.metric=='T90').any()
    assert timing[timing.status=='a_censored'].strict_lower_bound.iloc[0]==3900
    assert pd.isna(timing[timing.status=='a_censored'].T90_a.iloc[0])
    print('PASS: paired bootstrap, denominator handling, T90 censor preservation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path)
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args()
    if args.selftest:selftest();return
    if args.smoke and args.out==DEFAULT:
        args.out=DEFAULT/'dynamic_smoke'
    inputs=args.inputs or [DEFAULT/'raw/dynamic_mlp',DEFAULT/'raw/dynamic_vit']
    records=read_inputs(inputs,args.smoke)
    snap,track,paths,rec=frames(records)
    first=track[track.step==1].melt(id_vars=IDENT,
        value_vars=['d_body','d_head','g_body','v_body'],var_name='metric',value_name='value')
    first['metric']=first['metric'].map({'d_body':'actual_body_step_update1',
        'd_head':'actual_head_step_update1','g_body':'raw_body_gradient_update1',
        'v_body':'adam_v_rms_update1'})
    fs=summarize(first,CELL+['arm','metric'])
    fp,fps=paired(first,CELL+['metric'])
    init=[]
    zero=snap[(snap.phase=='acute') & (snap.step==0) & (snap.layer==-1)]
    for key,sub in zero.groupby(CELL+['seed'],sort=True):
        tab=sub.pivot(index='arm',columns='metric',values='value')
        for a,b,factor in [('T','TG',10.),('S','G',10.),('H10','T',.1)]:
            if a not in tab.index or b not in tab.index:continue
            rr=dict(zip(CELL+['seed'],key),arm_a=a,arm_b=b,expected_gradient_factor=factor)
            for metric in ['ce','confidence','entropy']:
                x,y=tab.loc[a,metric],tab.loc[b,metric]
                rr[metric+'_difference']=x-y
                rr[metric+'_matches']=bool(np.isclose(x,y,rtol=2e-5,atol=2e-7))
            x,y=tab.loc[a,'expected_feature_grad_rms'],tab.loc[b,'expected_feature_grad_rms']
            rr['expected_feature_gradient_ratio']=x/y if abs(y)>=1e-10 else np.nan
            rr['gradient_factor_matches']=bool(np.isclose(x,factor*y,rtol=2e-5,atol=2e-7))
            init.append(rr)
    init=pd.DataFrame(init)
    prefix='dynamic_smoke' if args.smoke else 'dynamic'
    args.out.mkdir(parents=True,exist_ok=True)
    # Summaries focus on registered endpoints; all intermediate raw values remain available.
    endpoint=snap[((snap.phase=='acute') & snap.step.isin([0,25,200])) | (snap.phase=='natural')]
    ss=summarize(endpoint,CELL+['arm','phase','step','layer','metric'])
    sp,sps=paired(endpoint,CELL+['phase','step','layer','metric'])
    ps=summarize(paths,CELL+['arm','interval','metric'])
    pp,pps=paired(paths,CELL+['interval','metric'])
    rs,rp,rt=recovery_stats(rec)
    for name,df in [('snapshots',snap),('track',track),('paths',paths),('recovery',rec),
                    ('snapshot_summary',ss),('snapshot_paired',sp),('snapshot_paired_seeds',sps),
                    ('first_update_summary',fs),('first_update_paired',fp),
                    ('first_update_paired_seeds',fps),('initial_match_checks',init),
                    ('path_summary',ps),('path_paired',pp),('path_paired_seeds',pps),
                    ('recovery_summary',rs),('recovery_paired',rp),('recovery_t90_pairs',rt)]:
        save_csv(df,args.out/(prefix+'_'+name+'.csv'))
    report(ss,ps,sp,pp,rec,rp,fp,init,args.out,prefix,args.smoke)
    if not args.no_figures:
        figures(snap,paths,rec,args.out,prefix)
        first_update_figure(fp,args.out,prefix)
        elu_t10_figure(snap,args.out,prefix)
    prov=dict(smoke=args.smoke,n_files=len(records),bootstrap_resamples=BOOT,
              statistic='seed mean; paired difference or mean seedwise ratio',
              numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
              script_sha256=digest(__file__),
              spec_sha256=digest(ROOT/'specs/spec_loading_bridge_1111.md'),
              producing_commits=sorted({r['provenance']['git_hash'] for r in records}),
              producing_script_hashes=sorted({r['provenance']['script_sha256'] for r in records}),
              inputs={r['_path']:digest(r['_path']) for r in records},command=sys.argv)
    (args.out/(prefix+'_provenance.json')).write_text(json.dumps(prov,indent=2)+'\n')
    print(f'Wrote {prefix} summaries for {len(records)} files to {args.out}')


if __name__=='__main__':
    main()
