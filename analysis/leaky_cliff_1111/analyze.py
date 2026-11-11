#!/usr/bin/env python3
"""Rebuild tables and scientific figures. Historical analyses explicitly post hoc."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T975={1:12.7062047364,2:4.30265272975,3:3.18244630528,4:2.77644510520,9:2.26215716285}
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/leaky_cliff_1111'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def interval(v):
    v=np.array(v,dtype=float);n=len(v);mean=v.mean();se=v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    h=T975[n-1]*se if n>1 else np.nan
    return dict(n=n,mean=mean,sd=v.std(ddof=1) if n>1 else np.nan,lo=mean-h,hi=mean+h,min=v.min(),max=v.max())
def history():
    src=ROOT/'results/resp_ee_0917';manifest=json.loads((src/'backup_manifest.json').read_text())
    traces=[];events=[];sources=[]
    for s in range(10):
        p=src/f'runs/s{s}/prefix.csv';df=pd.read_csv(p);sources.append(dict(path=str(p),sha256=sha(p)))
        rel=f'results/resp_ee_0917/runs/s{s}/units.npz'
        item=next(f for f in manifest['files'] if f.get('source_rel')==rel)
        raw=Path(item['backup']);assert sha(raw)==item['sha256'];sources.append(dict(path=str(raw),sha256=item['sha256']))
        u=np.load(raw)
        for l in (1,2):
            g=u[f'prefix_gtrain_l{l}'];pplus=u[f'prefix_pplus_eff_l{l}'];m=u[f'prefix_zbar_l{l}'];sd=u[f'prefix_sigma_l{l}']
            for q in (.1,.5,.9):df[f'L{l}_unit_g_q{int(100*q)}']=np.quantile(g,q,axis=1)
            df[f'L{l}_g_native']=g.mean(1);df[f'L{l}_neff_native']=u[f'prefix_neffT_l{l}'].mean(1)
            df[f'L{l}_one_sided']=(np.minimum(pplus,1-pplus)<.01).mean(1)
            df[f'L{l}_r_median']=np.median(m/sd,axis=1)
        df['next_online_acc']=df.online_acc.shift(-1);df['seed']=s;traces.append(df)
        j=int(np.argmin(np.diff(df.online_acc)))+1;a=df.iloc[j-1];b=df.iloc[j]
        events.append(dict(seed=s,task_before=int(a.task),task_after=int(b.task),online_before=a.online_acc,online_after=b.online_acc,
                           endpoint_before=a.memo_acc,endpoint_after=b.memo_acc,g2_before=a.g2_end,g2_after=b.g2_end,
                           neff2_before=a.L2_neff_native,neff2_after=b.L2_neff_native))
    df=pd.concat(traces);df.to_csv(OUT/'historical_elu_traces_posthoc.csv',index=False);pd.DataFrame(events).to_csv(OUT/'historical_elu_events_posthoc.csv',index=False)
    (OUT/'historical_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for s,g in df.groupby('seed'):
        ax[0].plot(g.task,g.online_acc,color='C0',alpha=.18);ax[0].plot(g.task,g.memo_acc,color='C1',alpha=.18)
        ax[1].plot(g.task,g.g2_end,color='C0',alpha=.18)
        ax[2].plot(g.g2_end,g.next_online_acc,alpha=.4,lw=1)
    av=df.groupby('task').mean(numeric_only=True)
    ax[0].plot(av.index,av.online_acc,label='Online, same task',color='C0',lw=2.5)
    ax[0].plot(av.index,av.memo_acc,label='End of task',color='C1',lw=2.5)
    ax[1].plot(av.index,av.g2_end,label='Mean derivative, layer 2',lw=2.5)
    ax[1].plot(av.index,av.L2_neff_native,label='Derivative participation',lw=2.5,color='C2')
    ax[0].set(xlabel='Task',ylabel='Accuracy',ylim=(0,1.03));ax[0].legend(fontsize=8)
    ax[1].set(xlabel='Task end',ylabel='Response',yscale='log');ax[1].legend(fontsize=8)
    ax[2].set(xlabel='Layer 2 mean derivative at task end',ylabel='NEXT task online accuracy',xscale='log',ylim=(0,1.03))
    fig.suptitle('Existing ELU random-label MNIST: post-hoc reanalysis, 10 seeds (not the remembered CondA run)')
    savefig(fig,'historical_elu')

def natural():
    tasks=[];diags=[];events=[];seeds=[];curves=[]
    for stage in ('discovery','confirmation'):
        for p in sorted((OUT/stage).glob('*/s*/tasks.csv')):
            dr=p.parent;assert (dr/'done.json').exists();meta=json.loads((dr/'provenance.json').read_text())['args'];a=meta['a'];seed=meta['seed']
            df=pd.read_csv(p);dd=pd.read_csv(dr/'diagnostics.csv');cc=pd.read_csv(dr/'curves.csv')
            for tab in (df,dd,cc):tab['stage']=stage;tab['a']=a;tab['seed']=seed
            tasks.append(df);diags.append(dd);curves.append(cc)
            late=df[df.task>=16];early=df[df.task<=5]
            seeds.append(dict(stage=stage,a=a,seed=seed,early_online=early.online_acc.mean(),late_online=late.online_acc.mean(),
                              late_endpoint=late.endpoint_acc.mean(),late_ce=late.endpoint_ce.mean(),late_T90_reached=int(late.T90.notna().sum()),
                              online_drop=early.online_acc.mean()-late.online_acc.mean()))
            j=int(np.argmin(np.diff(df.online_acc)))+1;p0=df.iloc[j-1];p1=df.iloc[j]
            d0=dd[(dd.task==p0.task)&(dd.step==6000)].iloc[0];d1=dd[(dd.task==p1.task)&(dd.step==6000)].iloc[0]
            row=dict(stage=stage,a=a,seed=seed,task_before=int(p0.task),task_after=int(p1.task),online_before=p0.online_acc,online_after=p1.online_acc,
                     endpoint_before=p0.endpoint_acc,endpoint_after=p1.endpoint_acc)
            for l in (1,2):
                for k in ('true_gmean','pplus_unit_mean','one_sided_frac','m_mean','s_mean','r_q50','actual_delta_norm','hidden_jvp_descent'):
                    col=f'L{l}_{k}' if k!='hidden_jvp_descent' else k
                    row[col+'_before']=d0[col];row[col+'_after']=d1[col]
            events.append(row)
    t=pd.concat(tasks);d=pd.concat(diags);c=pd.concat(curves);ss=pd.DataFrame(seeds)
    ss.to_csv(OUT/'natural_seed_summary.csv',index=False);pd.DataFrame(events).to_csv(OUT/'natural_events.csv',index=False)
    rows=[]
    for (stage,a),g in ss.groupby(['stage','a']):
        for metric in ['early_online','late_online','late_endpoint','late_ce','online_drop']:
            rows.append(dict(stage=stage,a=a,metric=metric,**interval(g[metric])))
    pd.DataFrame(rows).to_csv(OUT/'natural_group_summary.csv',index=False)
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained');colors={a:plt.cm.viridis(i/6) for i,a in enumerate(sorted(t[t.stage=='discovery'].a.unique()))}
    for a,g in t[t.stage=='discovery'].groupby('a'):
        color=colors[a];av=g.groupby('task').mean(numeric_only=True)
        for s,gs in g.groupby('seed'):ax[0,0].plot(gs.task,gs.online_acc,color=color,alpha=.18)
        ax[0,0].plot(av.index,av.online_acc,color=color,label=f'a={a:g}',lw=2)
        ax[0,1].plot(av.index,av.endpoint_acc,color=color,lw=2)
        gd=d[(d.stage=='discovery')&(d.a==a)&(d.step==6000)].groupby('task').mean(numeric_only=True)
        ax[1,0].plot(gd.index,gd.L1_true_gmean,color=color);ax[1,1].plot(gd.index,gd.L2_true_gmean,color=color)
        ax[1,2].plot(gd.index,gd.L1_one_sided_frac,color=color,ls='-',label=f'a={a:g} L1')
        ax[1,2].plot(gd.index,gd.L2_one_sided_frac,color=color,ls='--')
    for stage,marker in [('discovery','o'),('confirmation','s')]:
        v=ss[ss.stage==stage]
        for a,g in v.groupby('a'):
            ii=interval(g.late_online);ax[0,2].errorbar(a,ii['mean'],yerr=[[ii['mean']-ii['lo']],[ii['hi']-ii['mean']]],marker=marker,color='C0' if stage=='discovery' else 'C3',capsize=3)
    ax[0,0].legend(fontsize=8,ncol=2);ax[0,0].set(xlabel='Task',ylabel='Online accuracy (6k updates)',ylim=(0,1.03))
    ax[0,1].set(xlabel='Task',ylabel='End accuracy (6k updates)',ylim=(0,1.03))
    ax[0,2].set(xlabel='Negative slope a',ylabel='Late online (tasks 16–20)',xscale='symlog',ylim=(0,1.03));ax[0,2].set_xscale('symlog',linthresh=.003);ax[0,2].set_xlim(-.00015,1.3);ax[0,2].set_xticks([0,.003,.01,.03,.1,.3,1],['0','.003','.01','.03','.1','.3','1'])
    ax[0,2].set_title('circles: discovery n=3; squares: confirmation n=5',fontsize=9)
    for l in (0,1):ax[1,l].set(xlabel='Task end',ylabel=f'Layer {l+1} mean derivative',yscale='log');ax[1,l].axhline(.1,color='gray',lw=.8,ls=':')
    ax[1,2].set(xlabel='Task end',ylabel='Near-one-branch units',ylim=(0,1.03));ax[1,2].set_title('solid: layer 1; dashed: layer 2',fontsize=9)
    fig.suptitle('Random-label MNIST / 784–100–100–10 / Adam 0.001 / 20 tasks; paired seeds')
    savefig(fig,'natural_sweep')
    # Dense trajectory, all discovery seeds: finite task-to-task loss versus changing distribution.
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for a in [.003,.01,.03,.1]:
        for seed in range(3):
            dd=d[(d.stage=='discovery')&(d.a==a)&(d.seed==seed)];tt=(dd.task-1)+dd.step/6000
            color=colors[a]
            ax[0,0].plot(tt,dd.L1_true_gmean,color=color,alpha=.5,label=f'a={a:g}' if seed==0 else None)
            ax[0,1].plot(tt,dd.L2_true_gmean,color=color,alpha=.5)
            ax[0,2].plot(tt,dd.L1_positive_width_unit,color=color,alpha=.5)
            sample=dd[dd.step>0];tt2=(sample.task-1)+sample.step/6000
            ax[1,0].plot(tt2,sample.L1_actual_delta_norm,color=color,alpha=.4)
            ax[1,1].plot(tt2,sample.hidden_jvp_descent,color=color,alpha=.4)
            cc=c[(c.stage=='discovery')&(c.a==a)&(c.seed==seed)];ax[1,2].plot(cc.task-1+cc.step/6000,cc.ce,color=color,alpha=.4)
    for p in ax.flat:p.set_xlabel('Task time (each task = 6,000 updates)')
    ax[0,0].set(ylabel='Layer 1 mean derivative',yscale='log');ax[0,0].legend(fontsize=8)
    ax[0,1].set(ylabel='Layer 2 mean derivative',yscale='log');ax[0,2].set(ylabel='Layer 1 units with p+ > .005')
    ax[1,0].set(ylabel='Actual layer 1 parameter step',yscale='log');ax[1,1].set(ylabel='Hidden J delta task descent');ax[1,1].set_yscale('symlog',linthresh=1e-4)
    ax[1,2].set(ylabel='Full-bank cross-entropy',yscale='log');fig.suptitle('Dense diagnostics: weight motion and useful functional motion are separate')
    savefig(fig,'dense_diagnostics')
    return t,d,c,ss

def branches():
    rows=[];curves=[];diags=[];shadows=[]
    for p in sorted((OUT/'branches').glob('*/s*/tasks.csv')):
        dr=p.parent;assert (dr/'done.json').exists();meta=json.loads((dr/'provenance.json').read_text())['args'];arm=meta['arm'];seed=meta['seed']
        for name,target in [('tasks.csv',rows),('curves.csv',curves),('diagnostics.csv',diags),('scale_shadow.csv',shadows)]:
            df=pd.read_csv(dr/name);df['arm']=arm;df['seed']=seed;target.append(df)
    rr=pd.concat(rows);cc=pd.concat(curves);dd=pd.concat(diags);sh=pd.concat(shadows)
    rr.to_csv(OUT/'branch_seed_summary.csv',index=False)
    ss=[]
    for arm,g in rr.groupby('arm'):
        for key in ['acc_6000','acc_24000','ce_6000','ce_24000','online_acc_first6000']:
            vals=g[key].dropna() if key in g else []
            if len(vals):ss.append(dict(arm=arm,metric=key,**interval(vals)))
    pd.DataFrame(ss).to_csv(OUT/'branch_group_summary.csv',index=False)
    contrasts=[]
    for arm,ref in [('A','N'),('B','N'),('X','N'),('O','N'),('H','O'),('BO','O'),('BO','B'),('Fbad','N'),('Fgood','N'),('Breverse','Ngood'),('Xreverse','Ngood'),('B1','N'),('B2','N')]:
        if arm not in rr.arm.values:continue
        g=rr[rr.arm==arm].set_index('seed');r=rr[rr.arm==ref].set_index('seed')
        for key in ['acc_6000','acc_24000','ce_6000','ce_24000','online_acc_first6000']:
            vals=(g[key]-r[key]).dropna()
            if len(vals):contrasts.append(dict(arm=arm,reference=ref,metric=key,**interval(vals)))
    pd.DataFrame(contrasts).to_csv(OUT/'branch_contrasts.csv',index=False)
    dd[dd.step.isin([0,1,75,750,6000,24000])].to_csv(OUT/'branch_diagnostic_points.csv',index=False)
    sh.to_csv(OUT/'branch_scale_shadow.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    arms=['N','A','B','X','O','H','BO','Fbad','Fgood'];palette={a:plt.cm.tab10(i) for i,a in enumerate(arms)}
    for arm in arms:
        g=cc[cc.arm==arm];av=g.groupby('step').mean(numeric_only=True)
        for seed,s in g.groupby('seed'):ax[0].plot(s.step,s.acc,color=palette[arm],alpha=.12,lw=.7)
        ax[0].plot(av.index,av.acc,color=palette[arm],lw=2,label=arm)
        ax[1].plot(av.index,av.ce,color=palette[arm],lw=2)
    ax[0].set(xlabel='Updates on the SAME new task',ylabel='Full-bank accuracy',ylim=(0,1.03));ax[0].legend(fontsize=8,ncol=3)
    ax[1].set(xlabel='Updates on the SAME new task',ylabel='Cross-entropy',yscale='log')
    for p in ax[:2]:p.axvline(6000,color='gray',ls=':',lw=1)
    for i,arm in enumerate(arms):
        g=rr[rr.arm==arm]
        for off,key,color in [(-.15,'acc_6000','C0'),(.15,'acc_24000','C1')]:
            ii=interval(g[key]);ax[2].errorbar(i+off,ii['mean'],yerr=[[ii['mean']-ii['lo']],[ii['hi']-ii['mean']]],fmt='o',color=color,capsize=2)
    ax[2].set(xticks=range(len(arms)),xticklabels=arms,ylabel='Endpoint accuracy, paired n=5',ylim=(0,1.05));ax[2].set_title('blue: 6k; orange: 24k (95% t intervals)',fontsize=9)
    fig.suptitle('One-time interventions from the SAME task-20 state; forward and optimizer controls')
    savefig(fig,'same_state_branches')
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for arm in ['N','B','X','O','H','BO']:
        g=dd[dd.arm==arm].groupby('step').mean(numeric_only=True);col=palette[arm]
        for l in range(2):ax[0,l].plot(g.index,g[f'L{l+1}_true_gmean'],label=arm,color=col)
        ax[0,2].plot(g.index,g.L1_one_sided_frac,color=col)
        h=g[g.index>0];ax[1,0].plot(h.index,h.L1_actual_delta_norm,color=col)
        ax[1,1].plot(h.index,h.hidden_jvp_descent,color=col);ax[1,2].plot(h.index,h.actual_error_cos,color=col)
    for p in ax.flat:p.set_xscale('symlog',linthresh=75);p.set_xlabel('Updates on task 21');p.set_xticks([0,75,750,6000,24000],['0','75','750','6k','24k'])
    ax[0,0].set(ylabel='TRUE mean derivative, layer 1',yscale='log');ax[0,0].legend(ncol=3,fontsize=8)
    ax[0,1].set(ylabel='TRUE mean derivative, layer 2',yscale='log');ax[0,2].set(ylabel='Layer 1 near-one-branch fraction')
    ax[1,0].set(ylabel='Layer 1 actual parameter step',yscale='log');ax[1,1].set(ylabel='Hidden J delta task descent');ax[1,1].set_yscale('symlog',linthresh=1e-5)
    ax[1,2].set(ylabel='Actual logit-step / error alignment')
    fig.suptitle('What changes after intervention? True response, parameter steps and task-aligned movement')
    savefig(fig,'branch_mechanism')
    return rr,pd.DataFrame(contrasts)

def brief_figure():
    ss=pd.read_csv(OUT/'natural_seed_summary.csv');fig,ax=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
    for metric,label,col in [('late_online','Online accuracy','C0'),('late_endpoint','End accuracy','C1')]:
        agrid=[];v=[];err=[]
        for a,g in ss[ss.stage=='confirmation'].groupby('a'):
            ii=interval(g[metric]);agrid.append(a);v.append(ii['mean']);err.append(ii['hi']-ii['mean'])
        ax[0].errorbar(agrid,v,yerr=err,label=label,color=col,marker='o',capsize=4)
    ax[0].set(xlabel='Negative slope a',ylabel='Accuracy, tasks 16–20',ylim=(0,1.03),xticks=agrid,xticklabels=['.03','.0548','.1'])
    ax[0].legend(fontsize=9);ax[0].set_title('Independent confirmation: 5 paired seeds')
    for arm,label,col in [('N','Natural a=.03','C0'),('B','Backward-only .1','C2'),('X','Anchored .1','C3'),('H','Head + Adam reset','C5'),('Fbad','Fresh a=.03','C7')]:
        dfs=[pd.read_csv(p) for p in (OUT/'branches'/arm).glob('s*/curves.csv')];avg=pd.concat(dfs).groupby('step').mean(numeric_only=True)
        for d in dfs:ax[1].plot(d.step,d.acc,color=col,alpha=.12,lw=.7)
        ax[1].plot(avg.index,avg.acc,label=label,color=col,lw=2)
    ax[1].axvline(6000,color='gray',ls=':',lw=1);ax[1].set(xlabel='Updates on one new task',ylabel='Full-bank accuracy',ylim=(0,1.03),xticks=[0,6000,12000,18000,24000],xticklabels=['0','6k','12k','18k','24k'])
    ax[1].legend(fontsize=8,loc='lower right');ax[1].set_title('Same task-20 state: 5 paired seeds')
    fig.suptitle('Adaptation depends on slope, trained state, and update budget')
    savefig(fig,'leaky_cliff_summary')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--history-only',action='store_true');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    history()
    if a.history_only:return
    assert (OUT/'COMPLETE.json').exists(),'main run incomplete'
    natural();branches();brief_figure()
if __name__=='__main__':main()
