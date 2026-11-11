#!/usr/bin/env python3
"""Complete-cell aggregation of preregistered demand/update diagnostics."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KINDS=['state','demand','step','source','response','allocation','noise','instant']
ACTS=['R','ELU','SILU','GELU','LR']

def table(df):
    def f(v):
        if isinstance(v,(float,np.floating)):
            if not np.isfinite(v):return 'NA'
            return f'{v:.5g}'
        return str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'---|'*len(df.columns)]+['| '+' | '.join(f(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def select(df,**kw):
    for k,v in kw.items():df=df[df[k].isin(v) if isinstance(v,(list,tuple)) else df[k].eq(v)]
    return df

def grouped(df,keys,cols):return df.groupby(keys,sort=False,dropna=False)[cols].mean().reset_index()

def main(raw,out):
    raw,out=Path(raw),Path(out);out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((raw/'complete.json').read_text());assert len(complete['jobs'])==15
    assert sorted((j['act'],j['seed']) for j in complete['jobs'])==sorted((a,s) for a in ACTS for s in range(3))
    assert all(len(j['baseline_maxabs'])==25 and max(j['baseline_maxabs'])==0 for j in complete['jobs'])
    D={k:pd.concat([pd.read_csv(raw/f'{a}_s{s}'/f'{k}.csv') for a in ACTS for s in range(3)],ignore_index=True) for k in KINDS}
    assert len(D['state'])==2376,len(D['state'])
    assert len(D['instant'])==2700,len(D['instant'])
    for k,d in D.items():d.to_csv(out/f'{k}.csv',index=False)
    st,dm,sp,ins=D['state'],D['demand'],D['step'],D['instant']
    checks=dict(cells=15,baseline_task_end_checks=375,baseline_max_error=0,
        logit_identity_max_error=float(D['allocation'].account_error.max()),
        raw_formula_max_error=float(sp.raw_formula_error.max()),
        source_step_max_rounding_error=float(D['source'].step_account_error.max()),
        native_z_max_rounding_error=float(sp.z_rounding_error.max()),
        demand_change_max_rms_error=float(ins.decomp_error.max()),
        variance_identity_max_error=float(D['noise'].variance_identity_error.max()),
        rows={k:len(v) for k,v in D.items()},wall_seconds=complete['seconds'])
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    # Fixed predictions: no independent-unit tests, report paired seeds explicitly.
    verdict=[]
    def add(p,status,values,meaning):
        verdict.append(dict(prediction=p,status=status,paired_values=json.dumps(values,ensure_ascii=False),interpretation=meaning))
    def score(values):
        return 'supported' if all(values) else 'contradicted' if not any(values) else 'mixed'
    p1=select(dm,act='R',task=25,arm='S',phase='fit',step=0,group='closed_units').sort_values('seed')
    p1g=select(sp,act='R',task=25,arm='S',phase='fit',step=0,optimizer='retained').sort_values('seed')
    p1ok=(p1.increase_fraction.to_numpy()>.05)&(p1g.hidden_raw_rms.to_numpy()==0)
    add('P1',score(p1ok),dict(up_fraction=p1.increase_fraction.tolist(),hidden_raw_rms=p1g.hidden_raw_rms.tolist()),'Activity demand exists while raw hidden gradient is zero.')
    q=select(ins,arm=['S','Q'],labels='new',group='all').pivot(index=['act','task','seed'],columns='arm',values='increase_mag')
    q['ratio']=q.Q/q.S
    p2cells=q.reset_index().groupby(['act','task']).ratio.apply(lambda v:bool((v<1).all()))
    add('P2','supported' if p2cells.any() else 'contradicted',q['ratio'].reset_index().to_dict('records'),'Q/S instantaneous increase-demand magnitude; fixed labels, all input/unit pairs.')
    e=select(st,act='ELU',task=10,phase='fit',step=4000).set_index(['arm','seed'])
    top=(e.loc['QV'].top_median-e.loc['S'].top_median).sort_index()
    opening=(e.loc['QV'].open_fraction-e.loc['S'].open_fraction).sort_index()
    add('P3',score((top>0)|(opening>0)),dict(top_difference=top.tolist(),open_fraction_difference=opening.tolist()),'QV minus S at task10 end; fixed V after Q.')
    p4=select(st,act='R',task=25,arm='Q',phase='fit',step=4000).sort_values('seed')
    add('P4',score(p4.initial_closed_reopened.fillna(1).to_numpy()<=.05),p4.initial_closed_reopened.tolist(),'Initially fully closed ReLU units reopening after one-shot Q; separate from repeated-Q prevention.')
    p5=select(sp,act='ELU',task=[10,25],arm=['Q','QA'],phase='fit',step=0,optimizer='retained').pivot(index=['task','seed'],columns='arm',values='body_update_rms')
    p5['ratio']=p5.QA/p5.Q
    hit=p5.reset_index().groupby('task').ratio.apply(lambda v:bool((v>2).all()))
    add('P5','supported' if hit.any() else ('contradicted' if (p5.ratio<=2).all() else 'mixed'),p5.ratio.reset_index().to_dict('records'),'QA/Q hidden step RMS, reset history only.')
    p6=select(st,task=10,arm='Q',phase='fit',step=[0,4000]).pivot(index=['act','seed'],columns='step',values='Vnorm')
    p6['regrowth']=p6[4000]/p6[0]
    p6d=select(sp,task=10,arm='Q',phase='fit',step=[200,500,1000,2000],optimizer='retained').groupby(['act','seed'])[['V_descent','W_descent']].mean()
    add('P6','descriptive',p6[['regrowth']].join(p6d).reset_index().to_dict('records'),'Readout regrowth and four fixed fit probes; frozen-arm counterfactuals listed in all-arm table.')
    p7=select(D['source'],task=10,phase='fit',step=500,source='open',receiver='zero_deriv')
    add('P7','identity_verified',p7[['act','seed','arm','n','dz_mean','dz_rms']].to_dict('records'),'Source accounting verified. Magnitudes are measured; nonzero receiving movement is not an independent algebraic discovery.')
    pd.DataFrame(verdict).to_csv(out/'verdict.csv',index=False)
    # Tables retain the specified task and time. All summaries average the 3 paired seeds.
    sections=[]
    def section(title,df,filename):
        df.to_csv(out/filename,index=False);sections.append('## '+title+'\n\n'+table(df)+'\n')
    section('Prediction scores',pd.DataFrame(verdict)[['prediction','status','interpretation']],'prediction_scores.csv')
    section('Instantaneous new-label demand, task 10',grouped(select(ins,task=10,arm=['S','Q'],labels='new',group='all'),['act','arm'],['increase_fraction','increase_mag','decrease_mag','Drms','confidence']),'instant_t10.csv')
    section('Closed-unit demand versus raw gradient, ReLU task 25',p1[['seed','n','increase_fraction','increase_mag','transmitted_rms']].reset_index(drop=True).join(p1g[['hidden_raw_rms','body_update_rms']].reset_index(drop=True)),'relu_closed_t25.csv')
    cols=['ce','acc','top_median','sd_median','r_median','units_per_input','closed_unit_fraction','Vnorm','initial_closed_reopened']
    section('Every arm at task 10 end',grouped(select(st,task=10,phase='fit',step=4000),['act','arm'],cols),'ends_t10.csv')
    section('Every arm at task 25 end',grouped(select(st,task=25,phase='fit',step=4000),['act','arm'],cols),'ends_t25.csv')
    section('ELU task 10 fit time course',grouped(select(st,act='ELU',task=10,phase='fit',arm=['S','Q','QV'],step=[0,200,500,1000,4000]),['arm','step'],['ce','acc','zmean','top_median','units_per_input','Vnorm']),'elu_time_t10.csv')
    section('ELU task 10 current open-input demand',grouped(select(dm,act='ELU',task=10,phase='fit',arm=['S','Q','QV'],group='open',step=[0,200,500,1000,4000]),['arm','step'],['n','Dmean','increase_fraction','increase_mag','decrease_mag']),'elu_open_demand.csv')
    section('ELU task 10 one-step allocation on all inputs',grouped(select(D['allocation'],act='ELU',task=10,phase='fit',arm=['S','Q'],kind='one_step',step=[0,200,500,1000,4000]),['arm','step','part'],['descent','logit_rms','exact_ce_decrease']),'elu_allocation.csv')
    section('ELU task 10 sources moving deep receivers',grouped(select(D['source'],act='ELU',task=10,phase='fit',arm=['S','Q'],receiver='deep',step=[0,200,500,1000,4000]),['arm','step','source'],['dz_mean','dz_rms','bias_dz_mean']),'elu_sources.csv')
    section('One-step optimizer contrasts at task 25 start',grouped(select(sp,task=25,phase='fit',step=0,arm=['S','Q','QA']),['act','arm','optimizer'],['hidden_raw_rms','body_update_rms','dz_mean','V_update_rms']),'optimizer_t25.csv')
    section('Reload after the one-shot intervention',grouped(select(st,phase='reload',step=[0,200,4000]),['act','task','arm','step'],['ce','acc','zmean','top_median','Vnorm','units_per_input']),'reload.csv')
    section('ELU V-gradient noise (variance includes covariance)',grouped(select(D['noise'],act='ELU',task=10,phase='fit',arm=['S','Q','QH'],step=[0,500,4000]),['arm','step'],['common_variance','remainder_variance','twice_covariance','total_variance','common_over_total','mean_gradient_rms']),'elu_noise.csv')
    (out/'summary.md').write_text('# demand_allocation_1111: registered diagnostic results\n\n'+
        'Three paired seeds (0,1,2); no unit-as-replicate inference. Entries in tables are seed means; all seed rows are in the corresponding CSVs and verdict.csv. Baseline task ends match prior saved states bit for bit. Q is one-shot except the separately audited existing Qall trajectory. A positive first-order descent is a local loss-reducing contribution, not a long-run causal performance share.\n\n'+
        json.dumps(checks,indent=2)+'\n\n'+'\n'.join(sections))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'S':'#4c566a','Q':'#0072b2','QV':'#d55e00','QF':'#009e73','QA':'#cc79a7'}
    fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    metrics=[('ce','Cross entropy'),('Vnorm','Readout norm'),('top_median','Median unit max z'),('units_per_input','Open units per input'),('zmean','Mean preactivation'),('acc','Training accuracy')]
    dat=select(st,act='ELU',task=10,phase='fit',arm=list(colors))
    for a,(metric,label) in zip(ax.flat,metrics):
        for arm,c in colors.items():
            g=dat[dat.arm.eq(arm)].groupby('step')[metric]
            m=g.mean();a.plot(m.index,m.values,label=arm,color=c);a.fill_between(m.index,g.min(),g.max(),alpha=.12,color=c)
        a.set_xscale('symlog',linthresh=10);a.set_xlabel('Steps on new task');a.set_ylabel(label)
    ax[0,0].legend(ncol=3,fontsize=9)
    fig.suptitle('ELU, task 10: paired branches from the same learned state\nMean and seed range (3 seeds); one-shot Q; bands are not confidence intervals')
    fig.savefig(out/'elu_allocation.png',dpi=180);fig.savefig(out/'elu_allocation.pdf');plt.close(fig)
    print(json.dumps(checks,indent=2));print(table(pd.DataFrame(verdict)[['prediction','status']]))
    print(table(pd.read_csv(out/'ends_t10.csv').query("act == 'ELU'")))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.raw,a.out)
