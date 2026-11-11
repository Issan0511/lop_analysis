#!/usr/bin/env python3
"""User-requested posthoc product of layer-mean derivatives; no new training."""
import csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'results/leaky_order_1111';OUT=ROOT/'results/leaky_product_1111'
GRID=[.001,.002,.003,.005,.01];SEEDS=list(range(3,8))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rr):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def ranks(x):return np.array([np.sum(x<v)+(np.sum(x==v)+1)/2 for v in x])
def correlation(x,y):return float(np.corrcoef(x,y)[0,1])
def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=[];manifest=[]
    assert json.loads((SRC/'analysis_checks.json').read_text())['all_pass']
    for age in [0,3,6]:
        steps=[0,750] if age<6 else [0,750,1500,3000,6000]
        for a in GRID:
            for seed in SEEDS:
                dr=SRC/'probes'/format(a,'.3f').replace('.','p')/f's{seed}'/f'age{age}'
                for fn in ['curves_t0.csv','summary.csv']:manifest.append(dict(path=str((dr/fn).relative_to(ROOT)),sha256=sha(dr/fn)))
                curve=read(dr/'curves_t0.csv');summary=read(dr/'summary.csv')[0];t=float(summary['T90_sustained'])
                for step in steps:
                    r=next(r for r in curve if int(r['step'])==step);mu1=float(r['L1_gmean']);mu2=float(r['L2_gmean']);product=mu1*mu2
                    assert product>0 and step<t
                    raw.append(dict(age=age,a=a,seed=seed,step=step,mu1=mu1,mu2=mu2,product=product,inverse_product=1/product,T90_sustained=t,T_times_product=t*product))
    fits=[];groups=[];comparisons=[]
    for age,step in sorted(set((r['age'],r['step']) for r in raw)):
        rr=[r for r in raw if r['age']==age and r['step']==step];assert len(rr)==25
        xx=np.log([r['product'] for r in rr]);yy=np.log([r['T90_sustained'] for r in rr]);slope,intercept=np.polyfit(xx,yy,1)
        c=float(np.exp(np.mean(xx+yy)));ratio=np.exp(xx+yy)/c
        rec=dict(age=age,step=step,n_cells=25,n_paired_seeds=5,loglog_slope=float(slope),loglog_intercept=float(intercept),loglog_R2=correlation(xx,yy)**2,
                 spearman_inverse_product_vs_time=correlation(ranks(-xx),ranks(yy)),fixed_inverse_C=c,
                 fixed_inverse_geometric_RMSE_factor=float(np.exp(np.sqrt(np.mean(np.log(ratio)**2)))),actual_over_fixed_prediction_min=float(min(ratio)),actual_over_fixed_prediction_max=float(max(ratio)))
        fits.append(rec)
        for measure in ['mu1','mu2','product']:
            v=np.log([r[measure] for r in rr]);b,k=np.polyfit(v,yy,1)
            comparisons.append(dict(age=age,step=step,measure=measure,loglog_slope=float(b),loglog_R2=correlation(v,yy)**2,spearman_inverse_vs_time=correlation(ranks(-v),ranks(yy))))
        for a in GRID:
            aa=[r for r in rr if r['a']==a]
            groups.append(dict(age=age,step=step,a=a,median_product=float(np.median([r['product'] for r in aa])),
                product_of_layer_medians=float(np.median([r['mu1'] for r in aa])*np.median([r['mu2'] for r in aa])),
                product_min=min(r['product'] for r in aa),product_max=max(r['product'] for r in aa),median_T90_sustained=float(np.median([r['T90_sustained'] for r in aa]))))
    write(OUT/'per_seed_products.csv',raw);write(OUT/'fits.csv',fits);write(OUT/'group_summary.csv',groups);write(OUT/'single_layer_comparison.csv',comparisons)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.4),sharex=True,sharey=True,layout='constrained');colors=plt.get_cmap('viridis')(np.linspace(0,1,5));markers=['o','s','^','D','v']
    for ax,step in zip(axes,[0,750]):
        rr=[r for r in raw if r['age']==6 and r['step']==step];fit=next(r for r in fits if r['age']==6 and r['step']==step)
        for ai,a in enumerate(GRID):
            for si,seed in enumerate(SEEDS):
                r=next(r for r in rr if r['a']==a and r['seed']==seed)
                ax.scatter(r['product'],r['T90_sustained'],c=[colors[ai]],marker=markers[si],s=47,edgecolors='black',linewidths=.3,label=f'a={a:g}' if si==0 else None)
        x=np.geomspace(min(r['product'] for r in rr)*.85,max(r['product'] for r in rr)*1.15,100)
        ax.plot(x,np.exp(fit['loglog_intercept'])*x**fit['loglog_slope'],'k-',lw=1.5,label='descriptive log-log fit')
        ax.plot(x,fit['fixed_inverse_C']/x,'k--',lw=1,label='best fixed inverse law')
        ax.set(xscale='log',yscale='log',xlabel="Product of layer means: mean(phi'1) x mean(phi'2)",title=f'Measured at update {step}\nslope={fit["loglog_slope"]:.2f}, log R2={fit["loglog_R2"]:.2f}')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Updates to sustained90');axes[0].set_ylim(4000,65000);axes[1].legend(fontsize=7,loc='best')
    fig.suptitle('After 6 prior tasks: same 25 runs, different measurement time (posthoc)',fontsize=12)
    fig.savefig(OUT/'product_vs_time_posthoc.png',dpi=180);fig.savefig(OUT/'product_vs_time_posthoc.pdf');plt.close(fig)
    p=next(r for r in fits if r['age']==6 and r['step']==0);q=next(r for r in fits if r['age']==6 and r['step']==750);s=next(r for r in fits if r['age']==3 and r['step']==0)
    lines=['# Product of layer-mean derivatives and adaptation time (posthoc)','',
           'Requested by Issa after the completed leaky_order_1111 run: would layer1 x layer2 be the rough scale? No new training, no new labels, no exclusions. Source age6 is the main comparison from the preceding conversation. The source-state product (step0) and the previously shown step750 values are both reported; neither window is relabeled preregistered.','',
           'For each individual seed and leak, compute P=mu1*mu2, where each mu is the mean activation derivative across1200 inputs x100 units of that layer. Aggregate only after multiplying inside the same seed. The product of two across-seed medians is a different statistic. Time is the beginning of the first four consecutive full-bank >=90% evaluations, unchanged from the registered run.','',
           f'At source age6, step0: T is descriptively proportional to P^({p["loglog_slope"]:.4f}), log-log R2={p["loglog_R2"]:.4f}, Spearman(1/P,T)={p["spearman_inverse_product_vs_time"]:.4f}. A fixed inverse fit gives T~{p["fixed_inverse_C"]:.3f}/P. The geometric residual RMS factor is{p["fixed_inverse_geometric_RMSE_factor"]:.3f}; actual/predicted values range{p["actual_over_fixed_prediction_min"]:.3f}-{p["actual_over_fixed_prediction_max"]:.3f}. This is an in-sample descriptive fit, not a prediction validation.',
           f'At source age3, step0, slope={s["loglog_slope"]:.4f}, R2={s["loglog_R2"]:.4f}, fixed-inverse constant={s["fixed_inverse_C"]:.3f}. Thus the rough within-age inverse relationship is not a common clock across source ages.',
           f'At age6, step750, slope={q["loglog_slope"]:.4f}, R2={q["loglog_R2"]:.4f}, Spearman(1/P,T)={q["spearman_inverse_product_vs_time"]:.4f}. The step0 correspondence does not persist at750. We do not select the more favorable time point as a universal law.','',
           '| a | Median product at source age6, step0 | Median sustained90 time |','|---:|---:|---:|']
    for r in groups:
        if r['age']==6 and r['step']==0:lines.append(f'| {r["a"]:g} | {r["median_product"]:.8f} | {r["median_T90_sustained"]:,.0f} |')
    lines+=['','## Scope and theory','',
            'There are25 cells but only5 paired initialization/data seeds; no iid25 p-value or confidence interval is claimed. Fits for source ages0,3,6 at steps0 and750, and age6 at1500,3000,6000 are saved in fits.csv. All these measurements precede their corresponding T90_sustained. Source age0 has very little variation in P or time and an essentially zero fitted slope; it provides no informative within-age scaling range.',
            'For a column-vector MLP with hidden preactivations z1,z2, D_l=diag(phi\'(z_l)), logits f and loss error e=dL/df, the exact first-preactivation gradient is dL/dz1=D1 W2^T D2 W3^T e. The derivative factors multiply along paths. P=mu1*mu2 is an unweighted scalar approximation: weight magnitudes, signs, correlations of the gates on the same input and path, the error direction, and Adam history are not retained in it. Therefore the earlier layer-by-layer comparison was incomplete; the product is a better motivated candidate, while reciprocal learning time is a further empirical hypothesis.',
            '[Backpropagation and the chain rule: Deep Learning, chapter6](https://www.deeplearningbook.org/contents/mlp.html). The numerical fits above use only the local experiment records, not this reference.','',
            '![Product versus time](product_vs_time_posthoc.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    identity=dict(posthoc=True,new_training=False,source_run='leaky_order_1111',source_result_commit='ec6df62b5125f149c77fd6217e7fd5260bdc2693',analysis_code_sha256=sha(__file__),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs=manifest)
    (OUT/'provenance.json').write_text(json.dumps(identity,indent=2)+'\n')
    (OUT/'checks.json').write_text(json.dumps(dict(all_pass=True,all_products_computed_within_seed=True,all_measurements_before_outcome=True,input_files=len(manifest),records=len(raw),cells_per_window=25),indent=2)+'\n')
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(run='leaky_product_1111',files=[],reason='No new training or untracked raw files; all derived records and input SHA256 manifest are versioned. Source raw remains in leaky_order_1111 archive.'),indent=2)+'\n')
    print(json.dumps(fits,indent=2))
if __name__=='__main__':main()
