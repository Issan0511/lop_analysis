#!/usr/bin/env python3
"""Registered factorial summary. No training; deterministic paired-seed inference.

Normalised internal records have model, activation, t_end, history, source,
tau, seed, arm, limit, curve[{step,acc,ce}], complete and provenance.
The runner adapter is deliberately separate from metric/inference code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/p_residual_loading_matplotlib')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/p_residual_loading_1111'
CELL = ['model', 'activation', 't_end', 'history', 'source', 'tau']
IDENT = CELL + ['seed', 'arm']
ARMS = ['KK', 'KC', 'ZK', 'ZC']
BOOT = 10000
MAIN_RUNNER_COMMIT = '0c874fe9fc27e46c695d7c41e55f90d92a03949d'
COEFFICIENTS = {
    'parameter_clear': {'KC': 1, 'ZC': -1},
    'parameter_keep': {'KK': 1, 'ZK': -1},
    'memory_keep': {'KK': 1, 'KC': -1},
    'memory_zero': {'ZK': 1, 'ZC': -1},
    'joint': {'KK': 1, 'ZC': -1},
    'interaction': {'KK': 1, 'KC': -1, 'ZK': -1, 'ZC': 1},
}
FORMULAS = {'parameter_clear': 'KC-ZC', 'parameter_keep': 'KK-ZK',
            'memory_keep': 'KK-KC', 'memory_zero': 'ZK-ZC',
            'joint': 'KK-ZC', 'interaction': 'KK-KC-ZK+ZC'}
METRICS = ['U', 'ce_auc_mean', 'error_end', 'accuracy_end', 'ce_end',
           'T90_first_elapsed', 'T90_sustained_elapsed']
PRIMARY = [('mlp', 'ELU', 10, 'S', 'P', 0),
           ('vit', 'SL', 100, 'S', 'P', 0)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).digest()[:8], 'little')


def bootstrap_indices(n, stratum):
    """One common seed resampling plan for every contrast in a stratum."""
    return np.random.default_rng(stable_seed(stratum)).integers(0, n, (BOOT, n))


def expected_design():
    expected = set()
    base = [('mlp', a, t, s) for a in ('R', 'ELU', 'SILU') for t in (1, 10) for s in range(10)]
    base += [('vit', 'SL', 100, s) for s in range(10252010, 10252015)]
    for model, act, t, seed in base:
        for source, tau in [('S', 0), ('P', 0), ('P', 25)]:
            expected.update((model, act, t, 'S', source, tau, seed, arm) for arm in ARMS)
    expected.update(('mlp', 'ELU', 29, 'Pall', 'P', 0, seed, arm) for seed in range(10) for arm in ARMS)
    assert len(expected) == 820
    return expected


def record_key(r):
    return tuple(r[k] for k in IDENT)


def required_grid(r):
    """Fixed registered evaluation grid, including the measured tau25 suffix."""
    if r.get('_synthetic'):
        return r.get('evaluation_grid')
    tau, limit = r['tau'], r['limit']
    every = 100 if r['model'] == 'mlp' else 25
    return sorted(x for x in {0, 1, 25, 26, limit, *range(0, limit + 1, every)} if tau <= x <= limit)


def validate_record(r):
    if r.get('complete') is not True:
        raise ValueError(f'Incomplete record: {record_key(r)}')
    if r['arm'] not in ARMS or r['source'] not in ('S', 'P') or r['history'] not in ('S', 'Pall'):
        raise ValueError(f'Invalid factor: {record_key(r)}')
    tau, limit = r['tau'], r['limit']
    if not isinstance(tau, int) or not isinstance(limit, int) or limit <= tau:
        raise ValueError('Invalid interval')
    curve = r['curve']
    steps = np.asarray([x['step'] for x in curve], dtype=float)
    acc = np.asarray([x['acc'] for x in curve], dtype=float)
    ce = np.asarray([x['ce'] for x in curve], dtype=float)
    if (len(steps) < 2 or not np.isfinite(steps).all() or
            not np.equal(steps, np.floor(steps)).all() or
            steps[0] != tau or steps[-1] != limit or not (np.diff(steps) > 0).all()):
        raise ValueError(f'Invalid/incomplete time grid: {record_key(r)}')
    if not np.isfinite(acc).all() or not np.isfinite(ce).all() or not ((0 <= acc) & (acc <= 1)).all() or not (ce >= 0).all():
        raise ValueError(f'Invalid accuracy/CE: {record_key(r)}')
    grid = required_grid(r)
    if grid is not None and steps.tolist() != list(grid):
        raise ValueError(f'Grid differs from declared grid: {record_key(r)}')
    return steps, acc, ce


def expand_shared(records):
    """Expand only prespecified exact identities; never add independent seeds."""
    by_key = {}
    for r in records:
        k = record_key(r)
        if k in by_key:
            raise ValueError(f'Duplicate cell: {k}')
        validate_record(r)
        by_key[k] = r
    for k, r in list(by_key.items()):
        if r['history'] == 'S' and r['tau'] == 0 and r['arm'] == 'ZC':
            other = copy.deepcopy(r)
            other['source'] = 'P' if r['source'] == 'S' else 'S'
            other['alias_kind'] = 'shared_tau0_ZC'
            other['alias_of'] = list(k)
            target = record_key(other)
            if target not in by_key:
                by_key[target] = other
            elif other['curve'] != by_key[target]['curve']:
                raise ValueError('S/P tau0 ZC identity violated')
    for k, r in list(by_key.items()):
        if (r['history'], r['source'], r['tau'], r['arm']) == ('S', 'P', 0, 'KK'):
            target = list(k); target[5] = 25; target = tuple(target)
            if r['limit'] <= 25:
                continue
            suffix = [copy.deepcopy(x) for x in r['curve'] if x['step'] >= 25]
            if not suffix or suffix[0]['step'] != 25:
                raise ValueError('tau25 KK reuse requires a measured step25')
            other = copy.deepcopy(r)
            other.update(tau=25, curve=suffix, alias_kind='tau25_KK_suffix', alias_of=list(k))
            if other.get('evaluation_grid') is not None:
                other['evaluation_grid'] = [x for x in other['evaluation_grid'] if x >= 25]
            if target not in by_key:
                by_key[target] = other
            elif suffix != by_key[target]['curve']:
                raise ValueError('tau25 KK suffix identity violated')
    return [by_key[k] for k in sorted(by_key)]


def trajectory_metrics(r):
    steps, acc, ce = validate_record(r)
    tau, limit = r['tau'], r['limit']
    threshold = float(np.float32(.9)) if r['model'] == 'mlp' else .9
    reached = acc >= threshold
    hits = np.flatnonzero(reached)
    # A final isolated crossing qualifies under recorded-sustained, but its
    # observation duration is zero and is reported explicitly.
    persistent = np.flatnonzero(np.logical_and.accumulate(reached[::-1])[::-1])
    first = int(steps[hits[0]]) if len(hits) else None
    sustained = int(steps[persistent[0]]) if len(persistent) else None
    row = {k: r[k] for k in IDENT}
    row.update(limit=limit, duration=limit-tau, n_evaluations=len(steps),
               U=float(np.trapezoid(1-acc, steps)/(limit-tau)),
               ce_auc_raw=float(np.trapezoid(ce, steps)),
               ce_auc_mean=float(np.trapezoid(ce, steps)/(limit-tau)),
               accuracy_end=float(acc[-1]), error_end=float(1-acc[-1]), ce_end=float(ce[-1]),
               T90_accuracy_threshold=threshold,
               T90_first_step=first, T90_first_elapsed=None if first is None else first-tau,
               T90_first_censored=first is None,
               T90_sustained_step=sustained, T90_sustained_elapsed=None if sustained is None else sustained-tau,
               T90_sustained_censored=sustained is None,
               sustained_observation_duration=None if sustained is None else limit-sustained,
               sustained_n_evaluations=0 if sustained is None else int((steps >= sustained).sum()),
               censor_horizon_step=limit, censor_horizon_elapsed=limit-tau,
               raw_path=r.get('_path', ''), raw_sha256=r.get('_sha256', ''),
               alias_kind=r.get('alias_kind', 'direct'),
               alias_of=json.dumps(r.get('alias_of')) if r.get('alias_of') else '')
    return row


def paired_contrasts(seed_metrics):
    out = []
    for key, g in seed_metrics.groupby(CELL, sort=True, dropna=False):
        seeds = sorted(g.seed.unique())
        if set(g.arm) != set(ARMS) or g.duplicated(['seed', 'arm']).any():
            raise ValueError(f'Missing/duplicate factorial arms: {key}')
        if any(set(g[g.arm == arm].seed) != set(seeds) for arm in ARMS):
            raise ValueError(f'Unpaired seed sets: {key}')
        # Source excluded: P and S contrasts/ratios share exactly the same draws.
        plan_key = tuple(key[:4]) + (int(key[5]), tuple(int(s) for s in seeds))
        ix = bootstrap_indices(len(seeds), plan_key)
        for metric in METRICS:
            table = g.pivot(index='seed', columns='arm', values=metric).reindex(seeds)
            for name, coef in COEFFICIENTS.items():
                x = table[list(coef)].to_numpy(float)
                valid = np.isfinite(x).all(1)
                row = dict(zip(CELL, key))
                row.update(kind='factorial', contrast=name, formula=FORMULAS[name], metric=metric,
                           n_seed=len(seeds), n_complete_pairs=int(valid.sum()),
                           estimable=bool(valid.all()), reason='' if valid.all() else 'censored_T90',
                           estimate=None, ci_low=None, ci_high=None)
                if valid.all():
                    delta = x @ np.array(list(coef.values()), dtype=float)
                    draws = delta[ix].mean(1)
                    row.update(estimate=float(delta.mean()), ci_low=float(np.quantile(draws,.025)),
                               ci_high=float(np.quantile(draws,.975)))
                out.append(row)
    return pd.DataFrame(out)


def guarded_ratios(seed_metrics, contrasts):
    """Primary parameter-clear U contrast, P/S; denominator gate is prespecified."""
    rows = []
    base_cols = ['model', 'activation', 't_end', 'history', 'tau']
    for key, g in seed_metrics.groupby(base_cols, sort=True):
        if set(g.source) != {'S', 'P'}:
            continue
        if set(g[g.source=='S'].seed)!=set(g[g.source=='P'].seed):
            raise ValueError(f'Unpaired S/P seed sets for ratio: {key}')
        for source in ('S','P'):
            sub=g[g.source==source]
            if any(set(sub[sub.arm==a].seed)!=set(sub.seed) for a in ARMS):
                raise ValueError(f'Incomplete S/P factorial for ratio: {key}')
        table = g.pivot(index='seed', columns=['source','arm'], values='U').sort_index()
        denom = table['S']['KC'] - table['S']['ZC']
        numer = table['P']['KC'] - table['P']['ZC']
        match = contrasts[(contrasts.kind == 'factorial') & (contrasts.source == 'S') &
                          (contrasts.contrast == 'parameter_clear') & (contrasts.metric == 'U')]
        for c, v in zip(base_cols, key):
            match = match[match[c] == v]
        if len(match) != 1:
            raise ValueError('Missing denominator contrast')
        d = match.iloc[0]
        row = dict(zip(base_cols, key))
        row.update(source='P/S', kind='P_over_S_ratio', contrast='parameter_clear', formula='(KC-ZC)_P/(KC-ZC)_S',
                   metric='U_ratio', n_seed=len(table), n_complete_pairs=len(table),
                   denominator_estimate=float(denom.mean()), denominator_ci_low=d.ci_low,
                   denominator_ci_high=d.ci_high, estimable=False, reason='denominator_CI_lower_not_above_0.01',
                   estimate=None, ci_low=None, ci_high=None)
        if d.estimable and d.ci_low > .01:
            plan_key = tuple(key[:4]) + (int(key[4]), tuple(int(s) for s in table.index))
            ix = bootstrap_indices(len(table), plan_key)
            den = denom.to_numpy()[ix].mean(1)
            num = numer.to_numpy()[ix].mean(1)
            if np.any(den == 0):
                row['reason'] = 'zero_bootstrap_denominator'
            else:
                draw = num / den
                row.update(estimable=True, reason='', estimate=float(numer.mean()/denom.mean()),
                           ci_low=float(np.quantile(draw,.025)), ci_high=float(np.quantile(draw,.975)))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(contrasts, main_complete):
    strata = []
    for key in PRIMARY:
        g = contrasts[(contrasts.kind == 'factorial') & (contrasts.contrast == 'parameter_clear') & (contrasts.metric == 'U')]
        for c,v in zip(CELL,key):
            g = g[g[c] == v]
        row = dict(zip(CELL,key))
        row.update(estimate=None, ci_low=None, ci_high=None, supported=None, practical=None)
        if main_complete and len(g) == 1 and bool(g.iloc[0].estimable):
            x = g.iloc[0]
            row.update(estimate=float(x.estimate), ci_low=float(x.ci_low), ci_high=float(x.ci_high),
                       supported=bool(x.ci_low > 0), practical=bool(x.ci_low > .01))
        strata.append(row)
    both = all(x['supported'] is True for x in strata) if main_complete else None
    practical = all(x['practical'] is True for x in strata) if main_complete else None
    classification = ('not_evaluated' if not main_complete else 'supported' if both else
                      'rejected_for_primary_conditions' if any(x['ci_high'] is not None and x['ci_high'] < 0 for x in strata)
                      else 'unresolved')
    return dict(main_design_complete=main_complete, primary_metric='U', primary_contrast='KC-ZC',
                operational_commonality_verdict=classification,
                strata=strata, both_strata_support_operational_head_dependent_delay=both,
                both_strata_practical_above_0_01=practical,
                interpretation='Operational agreement only; not identity of microscopic mechanism.',
                bootstrap=dict(replicates=BOOT, interval='percentile 95%', unit='paired seed', multiplicity_adjusted=False))


def load_records(paths):
    """Registered runner writes one complete logical trajectory per JSON."""
    records=[]
    for path in sorted(paths):
        r=json.loads(path.read_text())
        if r.get('file') != path.name:
            raise ValueError(f'Runner filename mismatch: {path}')
        r['_path'], r['_sha256'] = str(path), digest(path)
        r['alias_kind'] = ('tau25_KK_suffix' if r['tau']==25 else 'shared_tau0_ZC') if r.get('reused') else 'direct'
        if r.get('reused'): r['alias_of'] = r['reuse_of']
        records.append(r)
    return records


def validate_runner_records(records, smoke=False):
    """Reject mixed versions, missing raw aliases and altered fork identities."""
    expected_spec=digest(ROOT/'specs/spec_p_residual_loading_1111.md')
    expected_runner=digest(ROOT/'analysis/p_residual_loading_1111/run.py')
    files={r['file']:r for r in records}
    if len(files)!=len(records): raise ValueError('Duplicate runner filenames')
    helpers={}; source_hashes={}; runner_hashes=set(); git_hashes=set()
    actual_source_hashes={}
    frozen_smoke_sources={}
    if smoke:
        for folder in {Path(r['_path']).parent for r in records}:
            frozen=folder/'source_run.py'
            if frozen.exists(): frozen_smoke_sources[digest(frozen)]=str(frozen)
    strata={}; checkpoint_sources={}
    for r in records:
        validate_record(r)
        q=r['provenance']
        checkpoint=tuple(r[k] for k in ['model','activation','t_end','history','seed'])
        if checkpoint in checkpoint_sources and checkpoint_sources[checkpoint]!=q['sources']:
            raise ValueError(f'Changed source set within checkpoint: {checkpoint}')
        checkpoint_sources[checkpoint]=q['sources']
        if not smoke and r['model']=='vit' and q['args']['engine']!='compile':
            raise ValueError('Main ViT requires the registered compile engine')
        runner_hashes.add(q['script_sha256'])
        git_hashes.add(q['git_hash'])
        if (not smoke and q['script_sha256']!=expected_runner) or q['spec_sha256']!=expected_spec:
            raise ValueError(f'Incompatible runner/spec hashes: {r["file"]}')
        if smoke and q['script_sha256'] not in frozen_smoke_sources:
            raise ValueError('Smoke runner hash does not match a saved source_run.py')
        for source_path,hash_ in q['sources'].items():
            if source_path in source_hashes and source_hashes[source_path]!=hash_:
                raise ValueError(f'Mixed source version: {source_path}')
            source_hashes[source_path]=hash_
            if source_path not in actual_source_hashes:
                actual_source_hashes[source_path]=digest(source_path)
            if actual_source_hashes[source_path]!=hash_:
                raise ValueError(f'Actual source checksum mismatch: {source_path}')
        for rel,hash_ in q['helpers'].items():
            if rel in helpers and helpers[rel]!=hash_: raise ValueError(f'Mixed helper version: {rel}')
            if rel not in helpers and digest(ROOT/rel)!=hash_: raise ValueError(f'Changed helper: {rel}')
            helpers[rel]=hash_
        if not smoke and r['limit']!=(4000 if r['model']=='mlp' else 1500):
            raise ValueError(f'Non-main horizon: {r["file"]}')
        if r.get('reused') not in (True,False): raise ValueError('Missing reused marker')
        f,p=r['fork_hashes'],r['post_hashes']
        if set(f)!=set(p) or set(f)!={'body','body_adam','head','head_adam'}:
            raise ValueError('Missing state fingerprints')
        if f['body']!=p['body'] or f['body_adam']!=p['body_adam']:
            raise ValueError('Factorial altered body or body Adam')
        if r['arm'][0]=='K' and f['head']!=p['head']: raise ValueError('Keep parameter changed head')
        if r['arm'][1]=='K' and f['head_adam']!=p['head_adam']: raise ValueError('Keep memory changed head Adam')
        if r.get('legacy_eval_check') and not r['legacy_eval_check']['full_eval_exact']:
            raise ValueError('Legacy evaluation comparison failed')
        key=tuple(r[k] for k in CELL+['seed'])
        strata.setdefault(key,[]).append(r)
        if r['reused']:
            ref=files.get(r['reuse_of'])
            if ref is None or ref.get('reused'): raise ValueError('Missing/chained/cyclic reuse target')
            if any(r[k]!=ref[k] for k in ['model','activation','t_end','history','seed','limit']):
                raise ValueError('Reuse crosses source checkpoint or horizon')
            if r['tau']==0:
                if not (r['source']=='P' and r['arm']=='ZC' and ref['source']=='S' and ref['arm']=='ZC' and ref['tau']==0):
                    raise ValueError('Unregistered zero-head alias')
                if r['curve']!=ref['curve'] or r['post_hashes']!=ref['post_hashes']:
                    raise ValueError('S/P ZC identity mismatch')
            elif r['tau']==25:
                if not (r['source']==ref['source']=='P' and r['arm']==ref['arm']=='KK' and ref['tau']==0):
                    raise ValueError('Unregistered suffix alias')
                if r['curve']!=[x for x in ref['curve'] if x['step']>=25]:
                    raise ValueError('KK suffix differs')
            else: raise ValueError('Unregistered reuse time')
    if len(runner_hashes)!=1: raise ValueError('Mixed runner versions')
    if not smoke and git_hashes!={MAIN_RUNNER_COMMIT}:
        raise ValueError(f'Unexpected main runtime commits: {git_hashes}')
    for key,rr in strata.items():
        if len(rr)!=4 or {r['arm'] for r in rr}!=set(ARMS): raise ValueError(f'Incomplete factorial: {key}')
        if any(r['fork_hashes']!=rr[0]['fork_hashes'] for r in rr): raise ValueError('Factorial initial hashes differ')
        if len({r['post_hashes']['head'] for r in rr if r['arm'][0]=='Z'})!=1:
            raise ValueError('Zeroed heads differ')
        if len({r['post_hashes']['head_adam'] for r in rr if r['arm'][1]=='C'})!=1:
            raise ValueError('Cleared head Adam differs')
    if not smoke:
        if {record_key(r) for r in records}!=expected_design(): raise ValueError('Missing/extra main logical records')
        if len(records)!=820 or sum(not r['reused'] for r in records)!=690:
            raise ValueError('Unexpected direct/shared trajectory counts')
    return dict(helper_hashes=helpers, declared_source_hashes=source_hashes,
                actual_source_hashes=actual_source_hashes, source_actual_hashes_verified=True,
                runtime_git_hashes=sorted(git_hashes), runtime_runner_hashes=sorted(runner_hashes),
                frozen_smoke_sources=frozen_smoke_sources, main_expected_commit=MAIN_RUNNER_COMMIT)


def clean_json(x):
    if isinstance(x, dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x


def table_md(df):
    if df.empty: return '(none)'
    cols=list(df.columns)
    def value(v):
        if pd.isna(v): return '—'
        if isinstance(v, (float,np.floating)): return f'{v:.5f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |'] +
                     ['| '+' | '.join(value(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def plot_curves(records, out, smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9, 'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    panels=PRIMARY+[('mlp','ELU',1,'S','P',0),('mlp','ELU',29,'Pall','P',0)]
    fig,axs=plt.subplots(2,4,figsize=(16,7),squeeze=False)
    colors={'KK':'#343a40','KC':'#d97706','ZK':'#7c3aed','ZC':'#007c91'}
    for col,key in enumerate(panels):
        found=[r for r in records if tuple(r[k] for k in CELL)==key]
        for arm in ARMS:
            rr=sorted([r for r in found if r['arm']==arm],key=lambda r:r['seed'])
            if not rr: continue
            grids=[np.array([x['step'] for x in r['curve']]) for r in rr]
            if any(not np.array_equal(grids[0],g) for g in grids): raise ValueError('Unaligned plotting grid')
            for row,metric in enumerate(['acc','ce']):
                y=np.array([[x[metric] for x in r['curve']] for r in rr])
                mu=y.mean(0); sd=y.std(0,ddof=1) if len(y)>1 else np.zeros_like(mu)
                axs[row,col].plot(grids[0],mu,color=colors[arm],label=arm,lw=1.6)
                axs[row,col].fill_between(grids[0],mu-sd,mu+sd,color=colors[arm],alpha=.12,lw=0)
        axs[0,col].set_title(f'{key[0].upper()} {key[1]} | {key[3]} history\nTask end {key[2]} | P at switch')
        if not found:
            for ax in axs[:,col]: ax.text(.5,.5,'Not in this smoke subset',transform=ax.transAxes,ha='center',fontsize=8)
        axs[0,col].axhline(.9,color='#999999',lw=.7,ls=':')
        for row in range(2):
            axs[row,col].set_xlabel('Updates after switch'); axs[row,col].grid(alpha=.15)
            axs[row,col].set_ylabel('Training accuracy' if row==0 else 'Training cross-entropy')
    handles=[Line2D([],[],color=colors[a],label=a,lw=1.6) for a in ARMS]
    fig.legend(handles,ARMS,ncol=4,loc='upper center',bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle(('SMOKE IMPLEMENTATION CHECK | ' if smoke else '')+'P residual: readout parameters × head Adam state',weight='bold',y=.995)
    fig.text(.5,.015,'K/Z: keep/zero readout; K/C: keep/clear head Adam. Lines: seed means; bands: seed SD. Body Adam retained.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.84,bottom=.10,hspace=.32,wspace=.24)
    for ext in ['png','pdf']: fig.savefig(out/f'representative_curves.{ext}',dpi=180)
    plt.close(fig)


def plot_forest(contrasts,out,smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=contrasts[(contrasts.kind=='factorial')&(contrasts.metric=='U')&(contrasts.contrast=='parameter_clear')].sort_values(CELL)
    fig,ax=plt.subplots(figsize=(10,max(4,len(d)*.31+1.6)))
    labels=[]
    for i,r in enumerate(d.itertuples(index=False)):
        key=tuple(getattr(r,k) for k in CELL)
        color='#007c91' if key in PRIMARY else '#555e6c'
        labels.append(f'{r.model.upper()} {r.activation}, end {r.t_end}, {r.history} history, {r.source}, tau {r.tau}')
        ax.hlines(i,r.ci_low,r.ci_high,color=color,lw=1.6)
        ax.plot(r.estimate,i,'o',color=color,ms=5)
    ax.set_yticks(range(len(labels)),labels=labels,fontsize=8)
    ax.invert_yaxis();ax.axvline(0,color='#777777',lw=1)
    ax.axvline(.01,color='#aaaaaa',lw=.8,ls=':')
    ax.set_xlabel('Parameter effect D_C = U(KC) − U(ZC); positive: keeping head increases cost')
    ax.set_title(('SMOKE IMPLEMENTATION CHECK\n' if smoke else '')+'Head parameter contrast with head Adam cleared\nPaired-seed bootstrap 95% intervals; teal marks the two primary strata')
    ax.grid(axis='x',alpha=.15);ax.spines[['right','top']].set_visible(False)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'parameter_contrast_forest.{ext}',dpi=180)
    plt.close(fig)


def write_summary(seed, contrasts, v, records, out, smoke=False):
    primary=contrasts[(contrasts.kind=='factorial') & (contrasts.metric=='U')]
    primary=primary[(primary.source=='P') & (primary.tau==0)]
    cols=CELL+['contrast','estimate','ci_low','ci_high','n_seed']
    lines=['# '+('SMOKE — ' if smoke else '')+'P residual factorial summary', '',
           'U is the trapezoidal time-average of 1−accuracy over [tau, limit]. Every trajectory reaches the fixed horizon; no early stopping or horizon substitution for T90.', '',
           'Positive KC−ZC indicates a delay from keeping head parameters after head Adam is cleared. The KK−ZK contrast retains head Adam. Body Adam is fixed across the four interventions at the branch point.', '',
           'Recorded sustained T90 is the first saved evaluation after which every remaining saved evaluation is ≥90%. An isolated final crossing qualifies with zero observed duration. This does not guarantee continuous or future persistence.', '',
           'MLP accuracy is stored by canonical float32 evaluation; the 90% threshold uses float32(0.9), so an exact 1080/1200 score counts as reaching. ViT uses 0.9. Endpoint accuracy contrasts have the opposite desirability direction to U/error/CE; error_end is also provided.', '',
           'Intervals are deterministic paired-seed bootstrap percentile 95% intervals, 10,000 resamples, without multiplicity adjustment. Aliased records do not increase the seed count.', '',
           f'Logical rows: {len(seed)}. Direct input trajectories: {sum(r.get("alias_kind","direct")=="direct" for r in records)}. Main complete: {v["main_design_complete"]}.', '',
           '## U contrasts after P at tau=0', '',table_md(primary[cols]), '',
           '## Primary registered verdict', '', '```json',json.dumps(clean_json(v),indent=2),'```', '',
           '## First/sustained T90 censoring', '']
    cens=seed.groupby(CELL+['arm'],dropna=False).agg(n_seed=('seed','size'),first_unreached=('T90_first_censored','sum'),sustained_unreached=('T90_sustained_censored','sum'),
        first_elapsed_mean_all=('T90_first_elapsed',lambda x: x.mean() if x.notna().all() else np.nan),
        sustained_elapsed_mean_all=('T90_sustained_elapsed',lambda x: x.mean() if x.notna().all() else np.nan)).reset_index()
    lines += [table_md(cens), '', 'T90 contrasts are blank unless every paired seed has a recorded event in every arm used by that contrast. Unreached values stay null, with censor horizons stored separately.', '',
              '## Guarded ΔP/ΔS ratio', '',table_md(contrasts[contrasts.kind=='P_over_S_ratio'][[c for c in CELL+['estimate','ci_low','ci_high','denominator_ci_low','estimable','reason'] if c in contrasts.columns]]), '',
              'The ratio uses the primary parameter-clear U contrast and is reported only when the S denominator CI lower bound exceeds 0.01. This is not a mediation fraction.', '',
              '## Arm means (all seeds)', '',
              table_md(seed.groupby(CELL+['arm'],dropna=False)[['U','ce_auc_mean','accuracy_end','ce_end']].mean().reset_index()), '',
              'Files: seed_metrics.csv, contrasts.csv, verdict.json, representative_curves.png/.pdf, parameter_contrast_forest.png/.pdf, provenance.json.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def synthetic_checks():
    def rec(acc=(0.,.5,1.), steps=(0,25,100), **kw):
        r=dict(model='mlp',activation='ELU',t_end=10,history='S',source='P',tau=0,seed=0,arm='KK',limit=100,complete=True,_synthetic=True,
               curve=[dict(step=s,acc=a,ce=2*(1-a)) for s,a in zip(steps,acc)])
        return {**r,**kw}
    x=trajectory_metrics(rec())
    assert np.isclose(x['U'],.375) and np.isclose(x['ce_auc_mean'],.75)
    assert x['T90_first_step']==100 and x['sustained_observation_duration']==0
    x=trajectory_metrics(rec(acc=(.95,.8,.95)))
    assert x['T90_first_step']==0 and x['T90_sustained_step']==100
    x=trajectory_metrics(rec(acc=(.95,.8,.85)))
    assert x['T90_first_step']==0 and x['T90_sustained_step'] is None
    x=trajectory_metrics(rec(acc=(.1,.5,.8)))
    assert x['T90_first_elapsed'] is None and x['T90_sustained_censored']
    x=trajectory_metrics(rec(acc=(.1,.5,float(np.float32(.9)))))
    assert x['T90_first_step']==100
    x=trajectory_metrics(rec(acc=(.2,.9),steps=(25,100),tau=25))
    assert np.isclose(x['U'],.45) and x['T90_first_elapsed']==75
    for bad in [rec(limit=101), rec(acc=(0.,float('nan'),1.)),rec(steps=(0,25,25)), rec(complete=False)]:
        try: validate_record(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid curve accepted')
    shared=expand_shared([rec(arm='ZC',source='S'),rec()])
    assert len(shared)==4 and any(r['tau']==25 for r in shared)
    assert len(expected_design())==820
    rows=[]
    for source in ['S','P']:
        for s in range(10):
            for arm,value in {'KK':.6,'KC':.5,'ZK':.4,'ZC':.2}.items():
                r=rec(arm=arm,source=source,seed=s,acc=(1-value-.005*s,)*3)
                rows.append(trajectory_metrics(r))
    frame=pd.DataFrame(rows); ct=paired_contrasts(frame)
    u=ct[(ct.source=='P')&(ct.metric=='U')].set_index('contrast')
    for name,value in {'parameter_clear':.3,'parameter_keep':.2,'memory_keep':.1,'memory_zero':.2,'joint':.4,'interaction':-.1}.items():
        assert np.isclose(u.loc[name,'estimate'],value)
        assert np.isclose(u.loc[name,'ci_low'],value) and np.isclose(u.loc[name,'ci_high'],value)
    assert np.isclose(u.loc['joint','estimate'],u.loc['parameter_clear','estimate']+u.loc['memory_keep','estimate'])
    assert np.isclose(u.loc['interaction','estimate'],u.loc['parameter_keep','estimate']-u.loc['parameter_clear','estimate'])
    assert ct[ct.metric.str.startswith('T90')]['estimate'].isna().all()
    partial=frame.copy();partial['T90_first_elapsed']=100.
    partial.loc[(partial.source=='P')&(partial.seed==0)&(partial.arm=='KK'),'T90_first_elapsed']=np.nan
    pc=paired_contrasts(partial)
    bad=pc[(pc.source=='P')&(pc.metric=='T90_first_elapsed')&(pc.contrast=='memory_keep')].iloc[0]
    good=pc[(pc.source=='P')&(pc.metric=='T90_first_elapsed')&(pc.contrast=='parameter_clear')].iloc[0]
    assert bad.n_complete_pairs==9 and pd.isna(bad.estimate) and good.estimable
    rt=guarded_ratios(frame,ct)
    assert len(rt)==1 and rt.iloc[0].estimable and np.isclose(rt.iloc[0].estimate,1.)
    tiny=frame.copy();tiny.loc[(tiny.source=='S')&(tiny.arm=='KC'),'U']=.205
    guarded=guarded_ratios(tiny,paired_contrasts(tiny))
    assert not guarded.iloc[0].estimable and guarded.iloc[0].estimate is None
    mismatched=frame[~((frame.source=='P')&(frame.seed==0))]
    try: guarded_ratios(mismatched,paired_contrasts(mismatched))
    except ValueError: pass
    else: raise AssertionError('Unpaired S/P ratio accepted')
    assert ct.equals(paired_contrasts(frame))
    assert verdict(ct,False)['both_strata_support_operational_head_dependent_delay'] is None
    print('PASS: interval AUC, irregular grids, first/sustained censoring, alias reuse, paired factorial, ratio guard, deterministic bootstrap.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path,default=[OUT/'raw'])
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: synthetic_checks(); return
    paths=sorted({p.resolve() for entry in args.inputs for p in (entry.glob('*.json') if entry.is_dir() else [entry])})
    if not paths: raise ValueError('No runner records found')
    loaded=load_records(paths)
    validation=validate_runner_records(loaded,args.smoke)
    records=expand_shared(loaded)
    actual={record_key(r) for r in records}
    if not args.smoke and actual != expected_design():
        raise ValueError(f'Incomplete/extra design: missing={len(expected_design()-actual)}, extra={len(actual-expected_design())}')
    # Every within-stratum seed/arm must be measured on the same grid.
    grids={}
    for r in records:
        key=tuple(r[c] for c in CELL)
        grid=tuple(x['step'] for x in r['curve'])
        if key in grids and grids[key]!=grid: raise ValueError(f'Mismatched factorial grid: {key}')
        grids[key]=grid
    seed=pd.DataFrame([trajectory_metrics(r) for r in records])
    ct=paired_contrasts(seed)
    ratios=guarded_ratios(seed,ct)
    ct=pd.concat([ct,ratios],ignore_index=True)
    v=verdict(ct,not args.smoke)
    out=args.out/'smoke_summary' if args.smoke else args.out
    out.mkdir(parents=True,exist_ok=True)
    seed.to_csv(out/'seed_metrics.csv',index=False,na_rep='')
    ct.to_csv(out/'contrasts.csv',index=False,na_rep='')
    (out/'verdict.json').write_text(json.dumps(clean_json(v),indent=2,allow_nan=False)+'\n')
    plot_curves(records,out,args.smoke)
    plot_forest(ct,out,args.smoke)
    write_summary(seed,ct,v,records,out,args.smoke)
    prov=dict(script_sha256=digest(__file__),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              inputs=[dict(path=str(p),sha256=digest(p)) for p in paths],logical_records=len(records),input_records=len(loaded),validation=validation,
              aliased_records=sum(r.get('alias_kind','direct')!='direct' for r in records),numpy=np.__version__,pandas=pd.__version__)
    (out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),logical_rows=len(seed),contrasts=len(ct),main_complete=not args.smoke)))


if __name__=='__main__': main()
