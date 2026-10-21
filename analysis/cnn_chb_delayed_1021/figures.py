#!/usr/bin/env python3
"""Descriptive figures from archived curves; does not change preregistered tests."""
from pathlib import Path
import argparse, importlib.util, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RUN='cnn_chb_delayed_1021'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('delayed_report',ROOT/'analysis'/RUN/'result_report.py')
    rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep)
    results=ROOT/'results'/RUN;out=args.out or results/'figures';out.mkdir(parents=True,exist_ok=True)
    report=rep.read_json(results/'report/report.json')
    arms=rep.ARMS;colors=['#4b5563','#0072B2'];labels=['CH continuation','CH + delayed B']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    allcurves={}
    for a,c,label in zip(arms,colors,labels):
        with np.load(rep.resolve(results/a/'batch_curves.npz')) as z:
            allcurves[a]={k:z[k].copy() for k in ('accuracy','ce')}
        for j,k in enumerate(('accuracy','ce')):
            values=allcurves[a][k].reshape(10,400,75).mean(2,dtype=np.float64)
            scale=100 if j==0 else 1
            avg=scale*values.mean(0);se=scale*values.std(0,ddof=1)/np.sqrt(10)
            x=np.arange(1,401)*75
            axes[0,j].plot(x,avg,color=c,label=label,lw=1.4)
            axes[0,j].fill_between(x,np.maximum(1e-12,avg-se),avg+se,color=c,alpha=.15)
    axes[0,0].set(ylabel='Online accuracy (%)',title='A  Entire new task; mean ± seed SE',ylim=(0,100))
    axes[0,1].set(ylabel='Online CE (log scale)',yscale='log',title='B  Loss on the same updates')
    for ax in axes[0]:ax.set_xlabel('Task 31 update');ax.axvline(3000,color='k',ls=':',lw=.8)
    axes[0,0].legend(frameon=False)
    for j,k in enumerate(('accuracy','ce')):
        v=np.stack([allcurves[a][k][:,:3000].mean(1,dtype=np.float64) for a in arms])
        if k=='accuracy':v*=100
        for i,seed in enumerate(rep.SEEDS):
            axes[1,j].plot([0,1],v[:,i],color='#64748b',lw=.8,alpha=.65)
        for i,c in enumerate(colors):axes[1,j].scatter(np.full(10,i),v[i],color=c,s=18,zorder=3)
        t=report['primary'][k]
        axes[1,j].set(xticks=[0,1],xticklabels=labels,xlim=(-.15,1.22),
                      title=f"{'C' if j==0 else 'D'}  First 3,000: {t['verdict']}\n10 paired seeds; Holm p = {t['p_holm']:.6g}",
                      ylabel='Accuracy (%)' if j==0 else 'CE')
    fig.suptitle('Natural B added after degradation: paired task-30 CH states',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(out/f'learning_and_primary.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(13,10),layout='constrained')
    metrics=[('_native_mean_d_mean','Native mean derivative'),('_image_width_mean','Image-axis width'),
             ('_bias_over_width_mean','Signed bias / width'),('_r_mean','Relative position m / sd')]
    steps=np.array(rep.UPDATES)
    for i,l in enumerate(('c1','c2','f1','f2')):
        for j,(key,title) in enumerate(metrics):
            ax=axes[i,j]
            for a,c,label in zip(arms,colors,labels):
                val=[report['diagnostics_seed_means'][a][str(u)][l+key] for u in steps]
                ax.plot(steps,val,'o-',color=c,lw=1,ms=3,label=label)
            ax.set_xscale('symlog',linthresh=1)
            if j==1:ax.set_yscale('log')
            if j==0:ax.set_ylim(0,1)
            if i==0:ax.set_title(title)
            if j==0:ax.set_ylabel(l)
            if i==3:ax.set_xlabel('Task 31 update')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Descriptive full-image diagnostics; means over all 10 seeds',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(out/f'natural_diagnostics.{ext}')
    plt.close(fig)
    files=[]
    for p in sorted(out.iterdir()):
        if p.suffix in ('.png','.pdf'):files.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (out/'figure_manifest.json').write_text(json.dumps({'report_sha256':rep.sha(results/'report/report.json'),'files':files},indent=2)+'\n')
    print('Wrote',len(files),'figure files')

if __name__=='__main__':main()
