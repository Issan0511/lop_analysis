"""Complete-design analysis and saved-state audit; no selection of study outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from . import run as R

CONTRASTS=dict(D=('S1','P1'),B_S=('S1','S01'),B_P=('P1','P01'),
               Z_S=('S1','Z_keep'),Z_P=('P1','Z_keep'),M=('Z_keep','Z_clear'),
               H=('N','S1'),R_body=('Z_clear','F0'))

def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)

def estimate(v,draws):
    v=np.asarray(v,dtype=float);boot=v[draws].mean(1)
    lo,hi=np.quantile(boot,[.025,.975]);mean=float(v.mean())
    # Independent multinomial-count calculation of the same resampled means.
    counts=np.stack([(draws==i).sum(1) for i in range(len(v))],1)
    assert np.allclose(boot,counts@v/len(v),atol=1e-12,rtol=1e-12)
    return dict(mean=mean,lo=float(lo),hi=float(hi),
                direction='positive' if lo>0 else 'negative' if hi<0 else 'unresolved',
                interval_within_one_pp=bool(lo>=-1 and hi<=1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();raw=Path(a.raw);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);rows=[];records={};audit=[];source_audit=[];manif=[];history=[];gradient_rows=[]
    code_hashes=set();spec_hashes=set();initial={};shared={};expected=[]
    for act in R.ACTS:
        for seed in range(10):
            sp=raw/f'source_{act}_s{seed}.pt';s=torch.load(sp,weights_only=False);sh=R.sha(sp)
            assert s['complete'] and s['activation']==act and s['seed']==seed
            assert len(s['history'])==30 and set(s['snapshots'])==set(R.AGES)
            assert R.digest(s['inputs'])==s['inputs_sha256']
            current=R.digest(s['initial']['params_and_adam'][0])
            assert current==s['initial_parameters_sha256']
            if seed in initial:assert initial[seed]==current
            initial[seed]=current
            ds=R.digest((s['inputs'],s['subset'],s['labels'],s['orders']))
            if seed in shared:assert shared[seed]==ds
            shared[seed]=ds
            if act in ('ELU','R'):assert all(h['canonical_replay_exact'] for h in s['history'])
            for h in s['history']:
                history.append(dict(activation=act,seed=seed,task=h['task'],correct=h['correct'],acc=h['acc'],ce=h['ce']))
            source_audit.append(dict(activation=act,seed=seed,hash=sh,all30_tasks=True,
                                     canonical_control_replay=act in ('ELU','R'),paired_streams=True))
            model=R.Model(s['initial']['params_and_adam'][0],act)
            for age in R.AGES:
                for arm in R.ARMS:
                    key=(act,seed,age,arm);name=f'{act}_s{seed}_t{age}_{arm}.json';expected.append(name)
                    p=raw/name;r=json.loads(p.read_text());records[key]=r
                    assert r['complete'] and r['source_sha256']==sh
                    assert (r['activation'],r['seed'],r['t_end'],r['arm'])==key
                    assert r['source_snapshot_sha256']==R.digest(s['snapshots'][age])
                    assert r['labels_sha256']==s['labels_sha256'][age+1]
                    assert r['batch_sha256']==s['orders_sha256'][age+1]
                    code_hashes.add(r['provenance']['code_sha256']);spec_hashes.add(r['provenance']['spec_sha256'])
                    steps=np.array([z['step'] for z in r['curve']]);acc=np.array([z['acc'] for z in r['curve']])
                    assert steps.tolist()==R.B.points(4000)
                    correct=np.array([z['correct'] for z in r['curve']]);assert np.allclose(acc,correct/1200,atol=3e-8,rtol=0)
                    assert np.isfinite(acc).all() and np.all((acc>=0)&(acc<=1))
                    ce=np.array([z['ce'] for z in r['curve']]);assert np.isfinite(ce).all()
                    u=float(np.trapezoid(1-acc,steps)/4000)
                    independent=sum((2-acc[j]-acc[j+1])*.5*(steps[j+1]-steps[j]) for j in range(len(steps)-1))/4000
                    assert abs(u-independent)<1e-14
                    early=steps<=200;late=steps>=200;hits=np.flatnonzero(correct>=1080)
                    row=dict(activation=act,seed=seed,t_end=age,arm=arm,U=u,
                             U_acute=float(np.trapezoid(1-acc[early],steps[early])/200),
                             U_late=float(np.trapezoid(1-acc[late],steps[late])/3800),
                             accuracy_end=correct[-1]/1200,ce_end=float(ce[-1]),
                             T90_first=int(steps[hits[0]]) if len(hits) else '',T90_censored=not bool(len(hits)))
                    for pt in (0,25,200,4000):
                        d=next(z for z in r['curve'] if z['step']==pt)['diagnostics']
                        for k,v in d.items():
                            if isinstance(v,(int,float)):row[f'step{pt}_{k}']=v
                    rows.append(row)
                    assert [q['step'] for q in r['tracks']]==[1,25,200,4000]
                    for tr in r['tracks']:
                        gr=dict(activation=act,seed=seed,t_end=age,arm=arm,step=tr['step'],loss=tr['loss'],mean_z_step=tr['mean_z_step'])
                        for group,stats in tr['groups'].items():
                            gr.update({group+'_'+k:v for k,v in stats.items()})
                        gradient_rows.append(gr)
                    ep=raw/r['endpoint_file'];assert R.sha(ep)==r['endpoint_sha256']
                    state=torch.load(ep,weights_only=False);model.restore(state)
                    chk=R.measure(model,s['inputs'],s['labels'][age],s['labels'][age+1],4000,True)
                    target=r['curve'][-1]
                    assert chk['correct']==target['correct']
                    assert chk['ce']==target['ce']
                    assert R.digest(chk['diagnostics'])==R.digest(target['diagnostics'])
                    if arm=='N' and age<30:
                        next_native=s['history'][age]
                        assert next_native['task']==age+1
                        assert next_native['correct']==target['correct'] and next_native['ce']==target['ce']
                    audit.append(dict(activation=act,seed=seed,t_end=age,arm=arm,endpoint_metrics_exact=True,
                                      U_independent_maxabs=abs(u-independent)))
                    manif.append(dict(file=name,sha256=R.sha(p)))
    assert len(rows)==1440 and len(records)==1440 and len(code_hashes)==len(spec_hashes)==1
    assert set(p.name for p in raw.glob('*_s*_t*_*.json'))==set(expected)
    lookup={(r['activation'],r['seed'],r['t_end'],r['arm']):r for r in rows}
    draws=np.random.default_rng(11142026).integers(0,10,(10000,10));contrasts=[]
    for age in R.AGES:
        for act in R.ACTS:
            for name,(b,i) in CONTRASTS.items():
                vals=[100*(lookup[act,s,age,b]['U']-lookup[act,s,age,i]['U']) for s in range(10)]
                contrasts.append(dict(activation=act,t_end=age,contrast=name,primary=age==10,
                                      baseline=b,intervention=i,**estimate(vals,draws)))
    for age in R.AGES:
        for act in ('KKA','KKA23','KKT1','SNA'):
            for name in ('B_P','R_body'):
                b,i=CONTRASTS[name]
                vals=[100*(lookup[act,s,age,b]['U']-lookup[act,s,age,i]['U']-
                           lookup['ELU',s,age,b]['U']+lookup['ELU',s,age,i]['U']) for s in range(10)]
                contrasts.append(dict(activation=act+'-ELU',t_end=age,contrast=name,primary=age==10 and act!='SNA',
                                      baseline='paired activation difference',intervention='',**estimate(vals,draws)))
    means=[]
    for age in R.AGES:
        for act in R.ACTS:
            for arm in R.ARMS:
                rr=[lookup[act,s,age,arm] for s in range(10)]
                means.append(dict(activation=act,t_end=age,arm=arm,n=10,
                    **{k:float(np.mean([q[k] for q in rr])) for k in rr[0] if k not in ('activation','seed','t_end','arm','T90_first','T90_censored')},
                    T90_reached=sum(not q['T90_censored'] for q in rr),
                    T90_mean_if_all_reached=float(np.mean([q['T90_first'] for q in rr])) if all(not q['T90_censored'] for q in rr) else ''))
    csv_write(out/'per_seed.csv',rows);csv_write(out/'contrasts.csv',contrasts);csv_write(out/'arm_means.csv',means)
    csv_write(out/'endpoint_audit.csv',audit)
    csv_write(out/'source_history.csv',history)
    csv_write(out/'gradient_steps.csv',gradient_rows)
    R.save_json(out/'integrity.json',dict(complete=True,curves=len(rows),sources=len(source_audit),endpoints=len(audit),
                source_checks=source_audit,code_sha256=list(code_hashes)[0],spec_sha256=list(spec_hashes)[0],
                all_streams_paired=True,all_initial_parameters_paired=True,all_endpoint_metrics_exact=True,
                all_U_independently_recomputed=True,bootstrap_counts_independently_checked=True,
                native_continuations_match_following_source_task=True))
    R.save_json(out/'curve_manifest.json',dict(files=manif))
    make_report(out,contrasts,means)
    make_figures(out,records,contrasts,means)
    print(json.dumps({'complete':True,'curves':len(rows),'endpoints_checked':len(audit)}))

def make_report(out,contrasts,means):
    lines=['# Matched kunekune readout controls — complete results','',
      '6 activations ×10 paired seeds ×3 source ages ×8 arms =1440 curves. CPU native stage1 arithmetic.',
      'ELU/ReLU histories were replayed exactly, not counted as independent replication. All endpoint states reloaded and checked.',
      'U is period-average fixed-training accuracy deficit. Positive benefit is faster learning. CIs: paired-seed10000 bootstrap, descriptive and unadjusted.',
      'P changes column geometry and predictions as well as class association. Head changes are one-time, followed by full learning.',
      'S01/P01 preserve mean logits by compensating the head bias. Zero-head arms reset both weights and bias, so they are not merely the zero endpoint of that scale intervention.',
      'Small benefits alone do not establish health; compare Z_clear and F0 and absolute U. R_body positive means a residual cost after resetting head and Adam.',
      'No causal mediation shares, universal family guarantee, ViT inference, or sustained-prevention claim.','']
    for age in R.AGES:
        lines += [f'## Task end {age}'+(' (primary)' if age==10 else ' (secondary)'),'',
          '| Activation | P benefit D | Shrink B_S | Shrink after P B_P | Zero after P Z_P | Residual body R_body |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
        for act in R.ACTS:
            vals=[]
            for name in ('D','B_S','B_P','Z_P','R_body'):
                q=next(q for q in contrasts if q['activation']==act and q['t_end']==age and q['contrast']==name)
                vals.append(f"{q['mean']:+.2f} [{q['lo']:+.2f},{q['hi']:+.2f}]")
            lines.append('| '+act+' | '+' | '.join(vals)+' |')
        lines+=['','| Activation | Native U (%) | Zero-head / Adam reset U (%) | Fresh-body U (%) | Source mean abs derivative | Source head centered norm |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for act in R.ACTS:
            dd={q['arm']:q for q in means if q['activation']==act and q['t_end']==age}
            n=dd['N'];lines.append(f"| {act} | {100*n['U']:.2f} | {100*dd['Z_clear']['U']:.2f} | {100*dd['F0']['U']:.2f} | {n['step0_mean_abs_derivative']:.4f} | {n['step0_head_centered_norm']:.3f} |")
        lines.append('')
    lines += ['## Validation and scope','',
      'integrity.json and endpoint_audit.csv document1440 saved-state checks, canonical replay, paired streams and independent U aggregation.',
      'per_seed.csv includes censoring and response diagnostics; contrasts.csv includes all registered contrasts and paired activation differences.',
      'The family variants adapt alpha from variance; treatment differences include that history. Positive response plus positive shrink benefit is coexistence, not mediation.',
      'Mean derivatives and positive occupancy are distinct for oscillatory activations. Do not equate negative preactivation with a dead unit.','']
    lines += ['T90_first is the first observed grid point with at least1080 correct; it is not an exact per-step crossing time. Nonarrival is right-censored, not assigned4000.','']
    (out/'summary.md').write_text('\n'.join(lines))

def make_figures(out,records,contrasts,means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(N='#333333',S1='#777777',S01='#0072B2',P1='#E69F00',P01='#009E73',Z_keep='#CC79A7',Z_clear='#D55E00',F0='#56B4E9')
    for age in (10,30):
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for ax,act in zip(axes.flat,R.ACTS):
            for arm in R.ARMS:
                rr=[records[act,s,age,arm]['curve'] for s in range(10)]
                xx=[z['step'] for z in rr[0]];yy=np.array([[z['correct']/1200 for z in q] for q in rr])*100
                ax.plot(xx,yy.mean(0),label=arm,color=colors[arm],lw=1.5,ls='--' if arm=='F0' else '-')
                ax.fill_between(xx,yy.min(0),yy.max(0),color=colors[arm],alpha=.06)
            ax.set_title(act);ax.grid(alpha=.2);ax.set_ylim(0,102)
        axes[0,0].set_ylabel('Training accuracy (%)');axes[1,0].set_ylabel('Training accuracy (%)')
        for ax in axes[1]:ax.set_xlabel('Next-task updates')
        axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle(f'Task-end {age}: mean of10 paired seeds; faint bands=min/max')
        fig.tight_layout();fig.savefig(out/f'curves_t{age}.png',dpi=180);fig.savefig(out/f'curves_t{age}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,age in zip(axes,(10,30)):
        for j,name in enumerate(('B_P','Z_P','R_body')):
            qq=[next(q for q in contrasts if q['activation']==act and q['t_end']==age and q['contrast']==name) for act in R.ACTS]
            xx=np.arange(len(R.ACTS))+(j-1)*.21;yy=np.array([q['mean'] for q in qq])
            er=np.array([[q['mean']-q['lo'] for q in qq],[q['hi']-q['mean'] for q in qq]])
            ax.errorbar(xx,yy,yerr=er,fmt='o',capsize=3,label=name)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(len(R.ACTS)),R.ACTS,rotation=25)
        ax.set_title(f'Task end {age}');ax.set_ylabel('U difference (percentage points)');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Readout benefits and remaining body cost: descriptive paired95% intervals')
    fig.tight_layout();fig.savefig(out/'readout_benefits.png',dpi=180);fig.savefig(out/'readout_benefits.pdf');plt.close(fig)

if __name__=='__main__':main()
