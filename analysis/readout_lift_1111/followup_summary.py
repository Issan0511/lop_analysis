#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import md

def main(raw,out):
    raw=Path(raw);out=Path(out);complete=json.loads((raw/'complete.json').read_text());assert len(complete['jobs'])==6
    cells=[raw/f's{s}_t{t}' for t in (10,25) for s in (0,1,2)]
    metas=[json.loads((p/'complete.json').read_text()) for p in cells]
    for k in ('code_sha256','helper_sha256','engine_sha256','spec_sha256','git_hash'):assert len({m[k] for m in metas})==1
    state=pd.concat([pd.read_csv(p/'state.csv') for p in cells]);source=pd.concat([pd.read_csv(p/'source.csv') for p in cells])
    state.to_csv(out/'followup_state.csv',index=False);source.to_csv(out/'followup_source.csv',index=False)
    end=state[state.step==4000];end.to_csv(out/'followup_ends.csv',index=False)
    avg=end.groupby(['task','arm']).mean(numeric_only=True);avg.to_csv(out/'followup_endpoint_means.csv')
    verdict=[];contrasts=[]
    for t in (10,25):
        e=end[end.task==t].set_index(['arm','seed']);s=state[(state.task==t)&(state.step==0)].set_index(['arm','seed'])
        stability=[s.loc[('C01U',seed),'uniform_loss']+.01-e.loc[('C01U',seed),'uniform_loss'] for seed in (0,1,2)]
        verdict.append(dict(task=t,test='F1_uniform_stability',decision='PASS' if min(stability)>=0 else 'FAIL',minimum=min(stability)))
        for ctrl,test in (('C1','F2_small_readout'),('C01U','F3_labels_uniform'),('C01R','F3_labels_fresh')):
            vals=[]
            for metric in ('G','h_median'):
                for seed in (0,1,2):
                    v=float(e.loc[('C01',seed),metric]-e.loc[(ctrl,seed),metric]);vals.append(v)
                    contrasts.append(dict(task=t,test=test,control=ctrl,metric=metric,seed=seed,difference=v))
            ok=min(vals)>0 and min(stability)>=0
            verdict.append(dict(task=t,test=test,decision='SUPPORTED' if ok else 'MIXED_OR_NOT_SUPPORTED',minimum=min(vals)))
    pd.DataFrame(verdict).to_csv(out/'followup_verdict.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'followup_contrasts.csv',index=False)
    late=source[source.step.isin([200,500,1000,2000])].groupby(['task','seed','arm','group','source'],as_index=False).dz.mean()
    late.to_csv(out/'followup_late_sources.csv',index=False)
    checks=dict(cells=6,arms=len(end),seconds=complete['seconds'],body_matching_error=max(c['body_error'] for m in metas for c in m['checks']),
        initial_z_error=max(c['initial_z_error'] for m in metas for c in m['checks']),fixed_V_error=max(c['fixed_V_error'] for m in metas for c in m['checks']),
        moment_source_error=float(source.moment_error.max()),step_rounding=float(source.step_rounding.max()),source_hashes={k:metas[0][k] for k in ('code_sha256','helper_sha256','engine_sha256','spec_sha256','git_hash')})
    (out/'followup_checks.json').write_text(json.dumps(checks,indent=2))
    cols=['h_median','h_change','body','body_change','G','G_change','acc','ce','uniform_loss','cooccur_0','positive_activity','open_per_input']
    text=['# Differentiated body-constraint follow-up','',
          'Registered AFTER observing original matched-control instability; original failed P4 is retained. This uses exact constrained gradients and clears W Adam in all four arms. It is not a clean mediator estimate for natural Q.','',
          '## Endpoint means','',md(avg[cols].round(6).reset_index()),'',
          '## Decisions','',md(pd.DataFrame(verdict)),'','## Per-seed contrasts','',md(pd.DataFrame(contrasts).round(6)),'',
          '## Late next-step sources','',md(late.groupby(['task','arm','group','source']).dz.mean().unstack('source').round(8).reset_index()),'',
          '## Checks','',json.dumps(checks,indent=2)]
    (out/'followup_summary.md').write_text('\n'.join(text)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors={'C1':'#555555','C01':'#2379b5','C01U':'#d48624','C01R':'#9164aa'}
    for row,t in enumerate((10,25)):
        for col,metric in enumerate(('h_median','acc')):
            for arm,color in colors.items():
                d=state[(state.task==t)&(state.arm==arm)].groupby('step')[metric].agg(['mean','min','max'])
                ax[row,col].plot(d.index,d['mean'],label=arm,color=color,linewidth=2)
                ax[row,col].fill_between(d.index,d['min'],d['max'],color=color,alpha=.12)
            ax[row,col].set_title(f'Task {t}: '+('median unit maximum z' if col==0 else 'fixed-label training accuracy'))
            ax[row,col].set_xlabel('updates');ax[row,col].grid(alpha=.2)
            if col==1:ax[row,col].set_ylim(0,1)
    ax[0,0].legend(ncol=2,fontsize=9)
    fig.suptitle('Exact-gradient constraint: identical sinking body, different selective growth\nC1: full fixed readout; C01: 0.1 fixed readout; U: uniform targets; R: fresh labels. Three-seed mean/range.',fontsize=12)
    fig.savefig(out/'constrained_lift.png',dpi=170);fig.savefig(out/'constrained_lift.pdf');plt.close(fig)
    print(avg[cols].round(6).to_string());print(pd.DataFrame(verdict).to_string(index=False));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.raw,a.out)
