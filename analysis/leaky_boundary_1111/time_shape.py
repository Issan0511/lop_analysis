#!/usr/bin/env python3
"""Post-hoc time-shape analysis, 2026-11-11. No new trajectories or selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'results/leaky_boundary_1111'
def expfit(x,y):
    # Variable projection: solve amplitude analytically, then rate in 1D.
    def loss(b):
        e=np.exp(b*x);a=e@y/(e@e);return np.sum((y-a*e)**2)
    grid=np.linspace(-2,2,2001);j=int(np.argmin([loss(b) for b in grid]));assert 0<j<len(grid)-1
    lo,hi=grid[j-1],grid[j+1]
    for _ in range(70):
        b1=lo+(hi-lo)*.38196601125;b2=lo+(hi-lo)*.61803398875
        if loss(b1)<loss(b2):hi=b2
        else:lo=b1
    b=(lo+hi)/2;e=np.exp(b*x);return np.array([e@y/(e@e),b])
def main():
    n=pd.read_csv(O/'probe_seed_summary.csv');n=n[n.arm=='N'];q=pd.read_csv(O/'quarter_source_clocks.csv');fits=[];coeff={}
    for metric in ['T90','T90_sustained','learning_time']:
        for seed in ['mean',3,4,5,6,7]:
            v=q.groupby('age')[metric].mean() if seed=='mean' else q[q.seed==seed].set_index('age')[metric].sort_index()
            x=v.index.to_numpy()-2;y=v.to_numpy();assert len(x)==5 and np.isfinite(y).all()
            for model in ['linear','exponential']:
                if model=='linear':c=np.polyfit(x,y,1);pred=np.polyval(c,x)
                else:c=expfit(x,y);pred=c[0]*np.exp(c[1]*x)
                sse=((y-pred)**2).sum();sst=((y-y.mean())**2).sum()
                fits.append(dict(metric=metric,seed=seed,model=model,n_points=5,n_parameters=2,coefficient0=c[0],coefficient1=c[1],rmse=np.sqrt(sse/5),r2=1-sse/sst))
                if seed=='mean' and metric=='T90':coeff[model]=c
    f=pd.DataFrame(fits);f.to_csv(O/'time_shape_fits_posthoc.csv',index=False)
    records=[]
    for age,g in n.groupby('age'):
        event=g.T90.notna();reached=int(event.sum());lower=g.T90.where(event,g.steps).mean()
        records.append(dict(age=age,n=len(g),reached=reached,censored=len(g)-reached,mean_exact=g.T90.mean() if event.all() else np.nan,mean_lower_bound=lower,strict_lower_bound=not event.all()))
    a=pd.DataFrame(records);a.to_csv(O/'time_shape_by_age_posthoc.csv',index=False)
    assert int(a.censored.sum())==1
    cens=n[n.T90.isna()][['seed','age','steps','endpoint_acc']];assert cens.iloc[0].seed==5 and cens.iloc[0].age==6
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
    for i,(seed,g) in enumerate(n.groupby('seed')):
        local=g[g.age.between(2,3)].sort_values('age');ax[0].plot(local.age,local.T90,color=f'C{i}',alpha=.24,lw=1)
        h=g[g.age%1==0].sort_values('age');ax[1].plot(h.age,h.T90,color=f'C{i}',marker='o',ms=4,alpha=.7,label=f'seed {seed}')
        z=h[h.T90.isna()];ax[1].scatter(z.age,z.steps,marker='^',color=f'C{i}',s=65)
    local=a[a.age.between(2,3)];xx=np.linspace(0,1,101)
    ax[0].scatter(local.age,local.mean_exact,color='black',zorder=4,label='5-seed mean')
    ax[0].plot(xx+2,np.polyval(coeff['linear'],xx),color='C7',ls='--',label='Linear fit')
    ax[0].plot(xx+2,coeff['exponential'][0]*np.exp(coeff['exponential'][1]*xx),color='C3',ls=':',label='Exponential fit')
    ax[0].set(title='Near the boundary: both fits work',xlabel='Tasks learned before the common probe',ylabel='Updates to 90% accuracy');ax[0].legend(fontsize=8)
    broad=a[a.age%1==0];ok=broad.censored==0;ax[1].plot(broad.loc[ok,'age'],broad.loc[ok,'mean_exact'],color='black',lw=2,label='5-seed mean')
    z=broad[~ok];ax[1].scatter(z.age,z.mean_lower_bound,color='black',marker='^',s=65);ax[1].annotate('mean >13,830',(6,13830),xytext=(4.1,15750),arrowprops=dict(arrowstyle='->'),fontsize=9)
    ax[1].annotate('seed 5 >24,000',(6,24000),xytext=(3.6,24700),arrowprops=dict(arrowstyle='->'),fontsize=9)
    ax[1].axhline(6000,color='gray',ls=':',lw=.8);ax[1].set(title='Longer history: unequal growth and censoring',xlabel='Tasks learned before the common probe',ylabel='Updates to 90% accuracy',ylim=(0,26700));ax[1].legend(fontsize=8,loc='upper left',ncol=2)
    fig.suptitle('Post-hoc shape check: same saved runs, no new training / a=.003 / seeds 3–7')
    for ext in ['png','pdf']:fig.savefig(O/f'time_shape_posthoc.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    provenance=dict(status='posthoc_descriptive',new_training=False,source_commit='529080f9c87df0ce90530aba6d9ce972c0ef4628',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'probe_seed_summary.csv',O/'quarter_source_clocks.csv']},warning='Fits are in-sample on five correlated source positions, not validated growth laws. Mean with censoring is only a strict lower bound, never a mean over successful seeds.')
    (O/'time_shape_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f[(f.seed=='mean')&(f.metric=='T90')].to_string(index=False));print(a.to_string(index=False))
if __name__=='__main__':main()
