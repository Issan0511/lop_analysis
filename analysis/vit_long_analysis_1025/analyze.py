"""Recorded-output analysis of vit_long_1024; no training or checkpoint updates.

Freeze completed runs once with --capture, then reproduce from the frozen snapshot.
Registered J0/J1/J3/JLN rules follow spec_vit_long_1024.md section 7 and addendum A.
Chunk attribution and cross-architecture comparisons are descriptive post-hoc work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = ['R0', 'S', 'R0L', 'SL']
FRESH = ['fresh', 'freshL']
COLORS = dict(R0='#b65b30', S='#9b386b', R0L='#157a87', SL='#365bc2')
LABELS = dict(R0='No LN / reset', S='No LN / shared', R0L='LN / reset', SL='LN / shared')
SMALL = ['per_task.csv', 'layer_stats.csv', 'intask.npz', 'provenance.json', 'done.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def capture(snapshot, source_repo, raw_2blk):
    snapshot.mkdir(parents=True, exist_ok=False)
    files, inventory = [], []
    roots = {'6blk': source_repo / 'results/vit_long_1024', '2blk': raw_2blk / 'runs'}
    stamp = datetime.now(timezone.utc).isoformat()
    for variant, root in roots.items():
        for arm in ARMS + FRESH:
            for seed in range(5):
                base = root / arm / f'seed{seed}'
                dirs = [base] if arm in ARMS else [base / f'task{t:02d}' for t in (25, 50, 60)]
                for src in dirs:
                    done = src / 'done.json'
                    state = json.loads(done.read_text()) if done.exists() else {}
                    complete = state.get('status') == 'complete'
                    inventory.append(dict(variant=variant, path=str(src.relative_to(root)),
                                          included=complete, done=state))
                    if not complete:
                        continue
                    dst = snapshot / variant / src.relative_to(root)
                    dst.mkdir(parents=True)
                    for name in SMALL:
                        original = src / name
                        target = dst / name
                        shutil.copy2(original, target)
                        files.append(dict(source=str(original), snapshot=str(target.relative_to(snapshot)),
                                          bytes=target.stat().st_size, sha256=digest(target)))
    for name in ['judge.json', 'judge.md']:
        src = roots['6blk'] / name
        dst = snapshot / '6blk' / name
        shutil.copy2(src, dst)
        files.append(dict(source=str(src), snapshot=str(dst.relative_to(snapshot)),
                          bytes=dst.stat().st_size, sha256=digest(dst)))
    meta = snapshot / 'source'
    meta.mkdir()
    for name in ['specs/spec_vit_long_1024.md', 'analysis/vit_long_1024/production.json',
                 'analysis/vit_long_1024/production_2blk.json', 'analysis/vit_long_1024/judge.py',
                 'analysis/vit_long_1024/run.py', 'analysis/vit_long_1024/common.py']:
        src = source_repo / name
        dst = meta / src.name
        shutil.copy2(src, dst)
        files.append(dict(source=str(src), snapshot=str(dst.relative_to(snapshot)),
                          bytes=dst.stat().st_size, sha256=digest(dst)))
    dump(snapshot / 'manifest.json', dict(captured_at_utc=stamp, files=files, inventory=inventory,
                                         note='Only terminal complete runs; ongoing training is untouched.'))


def j1_label(items):
    if len(items) < 4:
        return 'INSUFFICIENT'
    if sum(x['loss_condition'] for x in items) >= 3:
        return 'LOSS'
    if sum(x['established'] for x in items) < 3:
        return 'NOT_JUDGEABLE'
    return 'NO_LOSS'


def jln(a, b):
    for special in ['INSUFFICIENT', 'NOT_JUDGEABLE']:
        if special in (a, b):
            return special
    return {(False, False): 'SHIELD_HOLDS', (False, True): 'PHASE1_BREAKS_SHIELD',
            (True, True): 'NO_SHIELD', (True, False): 'MIXED'}[(a == 'LOSS', b == 'LOSS')]


def summarize(snapshot, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    for item in manifest['files']:
        assert digest(snapshot / item['snapshot']) == item['sha256'], item['snapshot']
    runs, layerframes, taskframes, metrics, freshrows = {}, [], [], [], []
    checks = dict(snapshot_hashes=len(manifest['files']), verified_continual_runs=0,
                  verified_tasks=0, verified_fresh_tasks=0, raw_reconstruction='passed')
    for variant in ['6blk', '2blk']:
        root = snapshot / variant
        for arm in ARMS + FRESH:
            for path in sorted((root / arm).glob('seed*/**/per_task.csv')):
                seed = int(next(x for x in path.parts if x.startswith('seed'))[4:])
                d = pd.read_csv(path)
                ls = pd.read_csv(path.parent / 'layer_stats.csv')
                prov = json.loads((path.parent / 'provenance.json').read_text())
                cfg = prov['config']
                depth = cfg['model_config']['depth']
                assert depth == (6 if variant == '6blk' else 2)
                assert cfg['act'] == 'R' and cfg['steps'] == 500 and cfg['n_labels'] == 10
                assert cfg['model_config']['mlp_ratio'] == (4 if variant == '6blk' else 8)
                assert cfg['images_per_class'] == 25 and cfg['batch'] == 128 and cfg['eval_every'] == 25
                assert cfg['data_kind'] == 'real'
                assert cfg['augment'] == 'off' and cfg['dropout'] == 0 and cfg['lr'] == .0001
                assert cfg['seed'] == seed and cfg['arm'] == arm
                assert cfg['ln'] == (arm in ['R0L', 'SL', 'freshL'])
                assert not prov['dirty'], str(path)
                with np.load(path.parent / 'intask.npz') as z:
                    assert np.array_equal(z['tasks'], d.task.to_numpy())
                    correct, size = z['online_correct'], z['batch_size']
                    np.testing.assert_allclose(correct.sum(1) / size.sum(1), d.online_acc, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(correct[:, :100].sum(1) / size[:, :100].sum(1),
                                               d.online_acc_first100, atol=1e-12, rtol=0)
                    chunks = np.array([correct[:, k:k+100].sum(1) / size[:, k:k+100].sum(1)
                                       for k in range(0, 500, 100)]).T
                    np.testing.assert_allclose(chunks, np.array(d.online_chunks.map(json.loads).tolist()), atol=1e-12)
                    mem = z['mem_correct']
                    np.testing.assert_allclose(mem[:, -1] / 250, d.mem_acc_end, atol=1e-12, rtol=0)
                    for threshold in (50, 75, 90):
                        reach = [next((int(z['u'][k]) for k in range(1, len(z['u']))
                                       if row[k] * 100 >= threshold * 250), 525) for row in mem]
                        assert np.array_equal(reach, d[f'T{threshold}']), str(path)
                    up = z['up_norm_med'].mean(axis=2).copy()
                    u = z['u'].copy()
                if arm in FRESH:
                    assert len(d) == 1
                    checks['verified_fresh_tasks'] += 1
                    freshrows.append(dict(variant=variant, arm=arm, seed=seed, task=int(d.task.iloc[0]),
                                          online_acc=float(d.online_acc.iloc[0]), T90=float(d.T90.iloc[0]),
                                          mem_acc_end=float(d.mem_acc_end.iloc[0])))
                    continue
                assert np.array_equal(d.task.to_numpy(), np.arange(1, 61)), str(path)
                end = ls[ls.state == 'end']
                assert end.groupby('task').size().eq(depth).all() and len(end) == depth * 60
                np.testing.assert_allclose(end.groupby('task').dead005.mean(), d.probe_dead005, atol=1e-12, rtol=0)
                early, late = d[d.task.between(2, 11)], d[d.task.between(41, 60)]
                E, L = float(early.T90.median()), float(late.T90.median())
                N = int((late.T90 > 500).sum())
                D = float(end[end.task == 60].dead005.mean())
                onset = next((t for t in range(6, 61)
                              if d[d.task.between(t-4, t)].online_acc.mean() < .5), None)
                x = dict(variant=variant, arm=arm, seed=seed, depth=depth,
                         train_git=prov['git_hash'], early_online=float(early.online_acc.mean()),
                         late_online=float(late.online_acc.mean()),
                         late_minus_early=float(late.online_acc.mean()-early.online_acc.mean()),
                         T90_early=E, T90_late=L, T90_ratio=L/E, unreached_late=N,
                         established=bool((early.T90 <= 500).sum() >= 6),
                         loss_condition=bool(L >= 2 * E and N >= 11), onset=onset,
                         late_mem=float(late.mem_acc_end.mean()),
                         late_first100=float(late.online_acc_first100.mean()),
                         D60=D, strict_dead60=float(end[end.task == 60].dead0.mean()),
                         final_cls_dead60=float(end[(end.task == 60) & (end.layer == depth)].cls_dead0.iloc[0]),
                         final_cls_mu60=float(end[(end.task == 60) & (end.layer == depth)].cls_mu_med.iloc[0]),
                         final_cls_sigma60=float(end[(end.task == 60) & (end.layer == depth)].cls_sigma_med.iloc[0]),
                         final_cls_r60=float(end[(end.task == 60) & (end.layer == depth)].cls_r_med.iloc[0]),
                         head_norm_late_start=float(late.head_norm_start.mean()),
                         head_norm_late_end=float(late.head_norm_end.mean()))
                for window, mask in [('early', d.task.between(2, 11)), ('late', d.task.between(41, 60))]:
                    indices = np.where(mask)[0]
                    denominator = float((up[indices, -1]-up[indices, 0]).sum())
                    x[f'rho_after100_{window}'] = (float((up[indices, -1]-up[indices, 4]).sum()) / denominator
                                                           if denominator > 0 else None)
                    eligible = [i for i in indices if d.T90.iloc[i] <= 250]
                    x[f'rho_postfit_tasks_{window}'] = len(eligible)
                    if eligible:
                        total = float((up[eligible, -1]-up[eligible, 0]).sum()) / (len(eligible)*500)
                        increment = sum(up[i, -1]-up[i, int(d.T90.iloc[i]//25)] for i in eligible)
                        duration = sum(500-float(d.T90.iloc[i]) for i in eligible)
                        x[f'rho_postfit_{window}'] = float(increment/duration/total) if total > 0 else None
                    else:
                        x[f'rho_postfit_{window}'] = None
                metrics.append(x)
                runs[(variant, arm, seed)] = (d, ls)
                taskframes.append(d.assign(variant=variant))
                layerframes.append(ls.assign(variant=variant, arm=arm, seed=seed))
                checks['verified_continual_runs'] += 1
                checks['verified_tasks'] += len(d)
    m = pd.DataFrame(metrics)
    tasks, layers = pd.concat(taskframes, ignore_index=True), pd.concat(layerframes, ignore_index=True)
    pairs, verdict = [], {}
    for variant in ['6blk', '2blk']:
        verdict[variant] = {'J0': {}, 'J1': {}, 'J3': {}}
        for arm in ARMS:
            items = [x for x in metrics if x['variant'] == variant and x['arm'] == arm]
            verdict[variant]['J1'][arm] = dict(label=j1_label(items), n=len(items),
                                              loss_seeds=sum(x['loss_condition'] for x in items),
                                              established_seeds=sum(x['established'] for x in items))
            verdict[variant]['J3'][arm] = dict(label=('INSUFFICIENT' if len(items)<4 else
                                                     'DEATH' if sum(x['D60'] >= .2 for x in items)>=3 else 'NO_DEATH'),
                                              n=len(items), death_seeds=sum(x['D60'] >= .2 for x in items))
        for condition, reset, shared in [('LN', 'R0L', 'SL'), ('noLN', 'R0', 'S')]:
            localpairs = []
            for seed in range(5):
                if (variant, reset, seed) not in runs or (variant, shared, seed) not in runs:
                    continue
                r, s = runs[(variant, reset, seed)][0], runs[(variant, shared, seed)][0]
                assert r.classes.tolist() == s.classes.tolist()
                for key in ['online_acc', 'T90', 'mem_acc_end', 'probe_dead005']:
                    assert r[key].iloc[0] == s[key].iloc[0], (variant, seed, key)
                row = dict(variant=variant, condition=condition, seed=seed)
                for w, a, b in [('early', 2, 11), ('late', 41, 60)]:
                    rc = np.array(r[r.task.between(a, b)].online_chunks.map(json.loads).tolist())
                    sc = np.array(s[s.task.between(a, b)].online_chunks.map(json.loads).tolist())
                    delta = (sc-rc).mean(0)
                    row[f'delta_{w}'] = float(delta.mean())
                    for k, value in enumerate(delta):
                        row[f'delta_{w}_chunk{k+1}'] = float(value)
                    row[f'first100_share_{w}'] = float(delta[0]/delta.sum()) if delta.sum() else None
                    row[f'first200_share_{w}'] = float(delta[:2].sum()/delta.sum()) if delta.sum() else None
                pairs.append(row)
                localpairs.append(row)
            neg, pos = sum(x['delta_late']<0 for x in localpairs), sum(x['delta_late']>0 for x in localpairs)
            verdict[variant]['J0'][condition] = dict(n=len(localpairs), negative=neg, positive=pos,
                label='INSUFFICIENT' if len(localpairs)<4 else 'SHARED_WORSE' if neg>=4 else
                      'SHARED_BETTER' if pos>=4 else 'MIXED')
        verdict[variant]['JLN'] = jln(verdict[variant]['J1']['R0L']['label'], verdict[variant]['J1']['SL']['label'])
    old = json.loads((snapshot/'6blk/judge.json').read_text())
    for condition in ['LN', 'noLN']:
        assert verdict['6blk']['J0'][condition]['label'] == old['J0'][condition]['label']
        for pair in [x for x in pairs if x['variant']=='6blk' and x['condition']==condition]:
            np.testing.assert_allclose(pair['delta_late'], old['J0'][condition]['seeds'][str(pair['seed'])]['delta_late'], atol=1e-12)
    for arm in ARMS:
        assert verdict['6blk']['J1'][arm]['label'] == old['J1'][arm]['label']
        assert verdict['6blk']['J3'][arm]['label'] == old['J3'][arm]['label']
        for x in [x for x in metrics if x['variant']=='6blk' and x['arm']==arm]:
            a, b = old['J1'][arm]['seeds'][str(x['seed'])], old['J3'][arm]['seeds'][str(x['seed'])]
            assert x['T90_early']==a['E'] and x['T90_late']==a['L'] and x['unreached_late']==a['N_unreached_late']
            np.testing.assert_allclose(x['D60'], b['D'], atol=1e-12)
    assert verdict['6blk']['JLN'] == old['JLN']['label']
    checks['registered_6blk_results'] = 'All J0/J1/J3/JLN labels and J0/J1/J3 per-seed endpoints reproduced'
    # Fresh gaps are computed at the same seed and task, never across machines.
    freshgaps=[]
    for f in freshrows:
        for arm in (['R0','S'] if f['arm']=='fresh' else ['R0L','SL']):
            key=(f['variant'],arm,f['seed'])
            if key not in runs:
                continue
            c=runs[key][0].set_index('task').loc[f['task']]
            freshgaps.append(dict(variant=f['variant'],arm=arm,seed=f['seed'],task=f['task'],
                                  fresh_online=f['online_acc'],continual_online=float(c.online_acc),
                                  gap_fresh_minus_continual=f['online_acc']-float(c.online_acc),
                                  fresh_T90=f['T90'],continual_T90=float(c.T90)))
    pairs = pd.DataFrame(pairs)
    for name, frame in [('per_seed',m),('paired_head_effect',pairs),('task_metrics',tasks),
                        ('layer_metrics',layers),('fresh_comparison',pd.DataFrame(freshgaps))]:
        frame.to_csv(output/f'{name}.csv',index=False)
    dump(output/'verdict.json', verdict)
    dump(output/'validation.json', checks)
    manifest['snapshot_root'] = str(snapshot)
    dump(output/'backup_manifest.json', manifest)
    make_plots(tasks,layers,m,pairs,output)
    return manifest, m, pairs, verdict, checks


def make_plots(tasks,layers,m,pairs,output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
    fig,axes=plt.subplots(3,2,figsize=(12.8,10.4))
    for j,variant in enumerate(['6blk','2blk']):
        title='6 blocks / FFN 1536 — complete' if variant=='6blk' else '2 blocks / FFN 3072 — interim'
        axes[0,j].set_title(title,weight='bold',pad=14)
        for arm in ARMS:
            d=tasks[(tasks.variant==variant)&(tasks.arm==arm)]
            l=layers[(layers.variant==variant)&(layers.arm==arm)&(layers.state=='end')]
            n=d.seed.nunique()
            if not n:continue
            curve=[]
            for seed,s in d.groupby('seed'):
                s=s.sort_values('task').set_index('task')
                sm=s.online_acc.rolling(5,min_periods=5).mean()
                axes[0,j].plot(sm.index,100*sm,color=COLORS[arm],alpha=.22,lw=.8)
                curve.append(sm)
            med=pd.concat(curve,axis=1).median(axis=1)
            axes[0,j].plot(med.index,100*med,color=COLORS[arm],lw=2,label=f'{LABELS[arm]} (n={n})')
            dead=l.groupby(['seed','task']).dead005.mean().unstack('seed')
            for seed in dead:
                axes[1,j].plot(dead.index,dead[seed]*100,color=COLORS[arm],alpha=.22,lw=.8)
            axes[1,j].plot(dead.index,dead.median(axis=1)*100,color=COLORS[arm],lw=2)
        axes[0,j].set_ylim(0,104);axes[1,j].set_ylim(0,100)
        axes[0,j].legend(fontsize=8,loc='lower left')
        axes[1,j].axhline(20,color='#777',ls=':',lw=1)
        axes[0,j].set_ylabel('Online accuracy (%)\n5-task rolling average')
        axes[1,j].set_ylabel('Low-activity units (%)\np < 0.005, averaged over blocks')
        for ax in axes[:2,j]:
            ax.set_xlabel('Task');ax.set_xlim(1,60)
            ax.axvspan(41,60,color='#999',alpha=.08)
        p=pairs[(pairs.variant==variant)&(pairs.condition=='LN')]
        cols=[f'delta_late_chunk{k}' for k in range(1,6)]
        for _,row in p.iterrows():
            axes[2,j].plot(range(1,6),100*row[cols].astype(float).to_numpy(),color=COLORS['SL'],alpha=.25)
        axes[2,j].plot(range(1,6),100*p[cols].median(),color=COLORS['SL'],marker='o',lw=2)
        axes[2,j].axhline(0,color='#666',lw=1)
        axes[2,j].set_xticks(range(1,6),['1–100','101–200','201–300','301–400','401–500'])
        axes[2,j].set_xlabel('Update within each task (tasks 41–60)')
        axes[2,j].set_ylabel('LN: shared minus reset (pp)')
        axes[2,j].set_ylim(-38,4)
    fig.suptitle('ViT: performance, low activity, and where the shared-head cost occurs',fontsize=15,weight='bold')
    fig.text(.5,.012,'Thin lines: seeds. Thick lines: seed medians. Online accuracy is training-stream accuracy, not test accuracy.\n'
             'Each architecture is analyzed separately; depth and FFN width changed together.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.965))
    fig.savefig(output/'overview.png',dpi=175)
    fig.savefig(output/'overview.pdf')
    plt.close(fig)
    # The last FFN influences the loss only through CLS; show that measurement explicitly.
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for j,(metric,label) in enumerate([('cls_mu_med','Last FFN / CLS: median raw mean'),
                                     ('cls_sigma_med','Last FFN / CLS: median standard deviation'),
                                     ('cls_dead0','Last FFN / CLS: probe-zero units (%)')]):
        for arm in ['R0L','SL']:
            d=layers[(layers.variant=='6blk')&(layers.arm==arm)&(layers.layer==6)&(layers.state=='end')]
            p=d.pivot(index='task',columns='seed',values=metric)
            scale=100 if metric=='cls_dead0' else 1
            for col in p: axes[j].plot(p.index,p[col]*scale,color=COLORS[arm],alpha=.2,lw=.8)
            axes[j].plot(p.index,p.median(axis=1)*scale,color=COLORS[arm],lw=2,label=LABELS[arm])
        axes[j].set_title(label,fontsize=10);axes[j].set_xlabel('Task')
        axes[j].set_xlim(1,60);axes[j].legend(fontsize=8)
    fig.suptitle('6-block LN models: negative drift can coexist with successful relearning',weight='bold')
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(output/'last_ffn_cls.png',dpi=175)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capture',action='store_true')
    p.add_argument('--source-repo',type=Path,default=Path('/home/issan/Projects/claude/wt/vit_long_1024'))
    p.add_argument('--raw-2blk',type=Path,default=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk'))
    a=p.parse_args()
    if a.capture:capture(a.snapshot,a.source_repo,a.raw_2blk)
    manifest,m,pairs,v,checks=summarize(a.snapshot,a.out)
    print(json.dumps(dict(snapshot_time=manifest['captured_at_utc'],verdict=v,validation=checks),indent=2))
    print(m.groupby(['variant','arm'])[['early_online','late_online','T90_early','T90_late','D60',
                                       'final_cls_dead60','final_cls_mu60','final_cls_sigma60']].median().to_string())
    print(pairs.to_string(index=False))


if __name__=='__main__':main()
