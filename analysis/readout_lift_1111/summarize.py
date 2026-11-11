#!/usr/bin/env python3
"""Summarize every registered cell, including failed directional criteria."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def md(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def main(raw,out):
    raw=Path(raw);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((raw/'complete.json').read_text());assert len(complete['jobs'])==6
    cells=[raw/f's{s}_t{t}' for t in (10,25) for s in (0,1,2)]
    meta=[json.loads((p/'complete.json').read_text()) for p in cells]
    for k in ('code_sha256','engine_sha256','fields_sha256','spec_sha256','git_hash'):
        assert len({m[k] for m in meta})==1,(k,'source drift')
    data={}
    for k in ('state','demand','source','counterfactual'):
        data[k]=pd.concat([pd.read_csv(p/f'{k}.csv') for p in cells],ignore_index=True)
        data[k].to_csv(out/f'{k}.csv',index=False)
    state=data['state'];ends=state[state.step==4000].copy();ends.to_csv(out/'ends.csv',index=False)
    endpoints=['ce','acc','G','G_change','h_median','h_change','body','body_change','body_sd','open_per_input','fixed_upper_da','cooccur_0','cooccur_200','Vnorm','margin']
    avg=ends.groupby(['task','arm'])[endpoints].mean();avg.to_csv(out/'endpoint_means.csv')
    time=state.groupby(['task','arm','step']).mean(numeric_only=True);time.to_csv(out/'time_means.csv')
    source=data['source'];late=source[source.step.isin([200,500,1000,2000])]
    late=late.groupby(['task','seed','arm','group','source'],as_index=False).dz.mean();late.to_csv(out/'late_sources.csv',index=False)
    verdict=[];contrasts=[]
    def decision(task,name,conditions,detail):
        vals=np.array(conditions,dtype=float)
        verdict.append(dict(task=task,test=name,decision='SUPPORTED' if np.all(vals>0) else 'MIXED_OR_NOT_SUPPORTED',minimum=float(vals.min()),detail=detail))
    def diff(task,a,b,metric,test):
        e=ends[ends.task==task].set_index(['arm','seed']);v=[float(e.loc[(a,s),metric]-e.loc[(b,s),metric]) for s in (0,1,2)]
        for s,x in enumerate(v):contrasts.append(dict(task=task,test=test,arm=a,control=b,metric=metric,seed=s,difference=x))
        return v
    for t in (10,25):
        e=ends[ends.task==t].set_index(['arm','seed'])
        for a in ('QV','V01'):
            vals=[e.loc[(a,s),m] for m in ('h_change','G_change') for s in (0,1,2)]
            decision(t,'P2_fixed_readout_growth_'+a,vals,'absolute median-h and gap changes >0, all3seeds')
        vals=sum([diff(t,'V01','V1',m,'P2_scale') for m in ('G','h_median')],[])
        decision(t,'P2_small_scale_advantage',vals,'V01-V1 gap and median h >0, all3seeds')
        for ctrl in ('V01U','V01R'):
            vals=sum([diff(t,'V01',ctrl,m,'P3_labels') for m in ('G','h_median')],[])
            decision(t,'P3_labels_vs_'+ctrl,vals,'fixed labels exceed control in gap and median h, all3seeds')
        for ctrl in ('V1','V01UM'):
            vals=sum([diff(t,'V01M',ctrl,m,'P4_matched') for m in ('G','h_median')],[])
            decision(t,'P4_matched_vs_'+ctrl,vals,'same body-mean path, gap and median h >control, all3seeds')
        for a in ('Q','QV','V01','V01M'):
            vals=late[(late.task==t)&(late.arm==a)&(late.group=='upper')&(late.source=='label')].sort_values('seed').dz.tolist()
            assert len(vals)==3
            decision(t,'P5_fixed_upper_label_source_'+a,vals,'next-step label-history source, mean across200/500/1000/2000 >0, all3seeds')
    pd.DataFrame(verdict).to_csv(out/'verdict.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    checks=dict(cells=6,arms=len(ends),source_hashes={k:meta[0][k] for k in ('code_sha256','engine_sha256','fields_sha256','spec_sha256','git_hash')},
        repro_maxabs=max(c['repro_maxabs'] or 0 for m in meta for c in m['checks']),
        mean_logit_error=max(c['mean_logit_error'] for m in meta for c in m['checks']),
        fixed_V_error=max(c['frozen_V_error'] for m in meta for c in m['checks']),
        body_matching_error=max(c['max_body_match'] for m in meta for c in m['checks']),
        moment_source_error=float(source.moment_error.max()),step_source_rounding=float(source.step_rounding.max()),
        z_rounding=float(source.z_rounding.max()),gradient_split_error=float(data['demand'].split_error.max()),
        seconds=complete['seconds'],rows={k:len(v) for k,v in data.items()})
    assert checks['repro_maxabs']==checks['fixed_V_error']==0
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    lines=['# Readout lift: registered results','',f"Six source states,54 continuations. Wall seconds: {complete['seconds']:.1f}. Three paired seeds; tasks analyzed separately.",'',
        'All planned outcomes are reported. SUPPORTED means the predeclared sign criterion held in all three seeds, not a population significance claim. No causal mediation percentages.','',
        '## Endpoint means at4000','',md(avg.round(6).reset_index()),'',
        '## Registered decisions','',md(pd.DataFrame(verdict)),'',
        '## Per-seed paired contrasts','',md(pd.DataFrame(contrasts).round(6)),'',
        '## Late next-step source means','',md(late.groupby(['task','arm','group','source']).dz.mean().unstack('source').round(8).reset_index()),'',
        '## Validation','',json.dumps(checks,indent=2),'',
        'Source accounting holds the observed Adam denominator fixed and is not a causal source deletion. Uniform and fresh-label controls change objectives/noise; matched-body controls add an external bias force. Neither is claimed to be a complete mediation analysis. Native ELU derivatives can round to zero in float32.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    # One scientific figure: body trajectory, absolute top, and causal contrasts at both tasks.
    colors={'V1':'#545454','V01':'#2379b5','V01U':'#d48624','V01R':'#9164aa','V01M':'#238b66','V01UM':'#bb4557'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,t in enumerate((10,25)):
        for col,metric in enumerate(('body','h_median')):
            ax=axes[row,col]
            arms=('V1','V01','V01M') if col==0 else ('V1','V01','V01U','V01R','V01M','V01UM')
            for a in arms:
                d=state[(state.task==t)&(state.arm==a)].groupby('step')[metric].agg(['mean','min','max'])
                ax.plot(d.index,d['mean'],label=a,color=colors[a],linestyle='--' if a.endswith('M') else '-',linewidth=1.8)
                ax.fill_between(d.index,d['min'],d['max'],color=colors[a],alpha=.09)
            ax.set_title(f'Task {t}: '+('fixed negative-body mean' if col==0 else 'median unit maximum z'))
            ax.set_xlabel('updates');ax.set_ylabel('preactivation');ax.grid(alpha=.2)
        ax=axes[row,2];pairs=[('V01','V1'),('V01','V01U'),('V01','V01R'),('V01M','V1'),('V01M','V01UM')]
        e=ends[ends.task==t].set_index(['arm','seed'])
        for j,(a,b) in enumerate(pairs):
            vs=np.array([e.loc[(a,s),'G']-e.loc[(b,s),'G'] for s in (0,1,2)])
            ax.scatter(j+np.array([-.12,0,.12]),vs,color=colors[a],s=25)
            ax.plot([j-.2,j+.2],[vs.mean()]*2,color='black',linewidth=2)
        ax.axhline(0,color='grey',linewidth=.8);ax.set_xticks(range(len(pairs)),[f'{a}\nminus {b}' for a,b in pairs],fontsize=8)
        ax.set_title(f'Task {t}: endpoint gap contrasts');ax.set_ylabel('max z minus body mean, difference');ax.grid(axis='y',alpha=.2)
    axes[0,1].legend(ncol=3,fontsize=8)
    fig.suptitle('Small fixed readouts: target-dependent expansion with controlled sinking\nLines: three-seed mean; bands: seed range. M: body matched to V1; U: uniform targets; R: fresh labels.',fontsize=12)
    fig.savefig(out/'readout_lift.png',dpi=170);fig.savefig(out/'readout_lift.pdf');plt.close(fig)
    print(avg[['G_change','h_change','body_change','acc']].round(5).to_string())
    print(pd.DataFrame(verdict)[['task','test','decision','minimum']].to_string(index=False))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--out',required=True);v=a.parse_args();main(v.raw,v.out)
