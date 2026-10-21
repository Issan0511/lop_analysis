#!/usr/bin/env python3
"""Plot completed, independently audited cap outputs; saved data only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib_cnn_width_cap_1021')
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from result_report import ROOT, resolve_input, ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'results/cnn_width_cap_1021')
    args=p.parse_args();base=args.base
    report=json.loads((base/'report/report.json').read_text())
    assert report['status']=='COMPLETE_AUDITED' and report['primary_matches_registered']
    colors=['#285fa3','#267b63'];labels=['ReLU','ELU (alpha=3.6)']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(11.8,8.8))
    fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.86,hspace=.42,wspace=.28)
    fig.suptitle('CIFAR-100 5+1 CNN: width cap and relearning',x=.08,y=.975,ha='left',fontsize=18,fontweight='bold')
    fig.text(.08,.925,'10 paired seeds per activation · 7,800 updates/task · all four hidden sites capped',color='#626975')
    ax=axs[0,0];ax.axhline(0,color='#777',lw=1)
    all_d=[]
    for i,q in enumerate(report['primary_independent']):
        d=np.array([x['difference'] for x in q['pairs']])*100;all_d.extend(d)
        ax.scatter(i+np.linspace(-.13,.13,10),d,color=colors[i],s=30,alpha=.75)
        ax.scatter(i+.23,d.mean(),marker='D',color=colors[i],s=65)
        ax.text(i,.97,f"{q['positive']}/{10} positive; Holm p={q['p_holm']:.4g}\n{q['verdict']}",
                ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=9,color=colors[i])
    ax.set_ylim(min(all_d)-8,max(all_d)+27)
    ax.set_xticks([0,1],labels);ax.set_xlim(-.4,1.5)
    ax.set_ylabel('Cap minus natural (percentage points)')
    ax.set_title('A  Registered primary: late hard, first 780 updates',loc='left',fontsize=11,fontweight='bold')
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    series={}
    for cond in ('natural','cap'):
        series[cond]={k:[] for k in ('s','dmean','dead')}
        for t in range(1,31):
            with np.load(resolve_input(base/cond/'units'/f'b{t:02d}.npz')) as z:
                for key in series[cond]:
                    a=z[f'probe_f2_{key}'].astype(float).reshape(2,10,100)
                    unit_stat=a.mean(2) if key=='dead' else np.median(a,axis=2)
                    series[cond][key].append(unit_stat.mean(1))
        series[cond]={k:np.array(v) for k,v in series[cond].items()}
    panels=[(axs[0,1],'s','B  Fixed-probe f2 width','Preactivation standard deviation'),
            (axs[1,0],'dmean','C  Fixed-probe f2 response','Mean derivative (unit median)'),
            (axs[1,1],'dead','D  Fixed-probe f2 death','Dead-unit fraction')]
    for ax,key,title,ylabel in panels:
        for cond,style in (('natural','-'),('cap','--')):
            for i,label in enumerate(labels):
                ax.plot(range(1,31),series[cond][key][:,i],style,color=colors[i],lw=2,label=f'{label}, {cond}')
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.set_ylabel(ylabel)
        ax.set_xlabel('Task boundary');ax.set_xlim(1,30);ax.grid(alpha=.17);ax.set_axisbelow(True)
        if key=='dead':ax.set_ylim(-.025,1.025)
    axs[0,1].legend(frameon=False,fontsize=8,ncol=2)
    fig.text(.08,.025,'A: dots are paired seeds; diamonds are means. B–D: means across seeds.\nWidth/response use within-seed unit medians; death is the unit fraction with max derivative < 1e-6. f2 is descriptive; intervention covered all four sites.',fontsize=8.5,color='#626975')
    for ext in ('png','pdf'):
        fig.savefig(base/'report'/f'figure.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
