#!/usr/bin/env python3
"""Plot preregistered frozen-probe ratios; no new fitting or measurements."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'P': '#2776AC', 'RC': '#8654A3', 'RR': '#47816C', 'Q': '#D68120'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results/loading_bridge_1111')
    args = parser.parse_args()
    data = list(csv.DictReader(open(args.results / 'static_ratios.csv')))
    def row(architecture, activation, task, arm, metric):
        return next((r for r in data if (r['architecture'],r['activation'],r['task'],r['arm'],r['metric']) ==
                     (architecture,activation,str(task),arm,metric)), None)
    def draw(ax, x, r, color, marker, label=None):
        if r is None:
            return
        mid,lo,hi=[float(r[k]) for k in ('ratio_of_means','ci_low','ci_high')]
        ax.errorbar(x,mid,yerr=np.array([[mid-lo],[hi-mid]]),color=color,marker=marker,
                    markersize=6.4,capsize=3.2,elinewidth=1.6,linestyle='none',label=label,zorder=4)
    plt.rcParams.update({'font.size':10.5,'axes.titlesize':12,'axes.titleweight':'bold',
                         'axes.labelsize':10.5,'savefig.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'font.family':'DejaVu Sans'})
    fig,axs=plt.subplots(2,2,figsize=(11.2,8.5))
    fig.subplots_adjust(left=.085,right=.98,top=.88,bottom=.16,hspace=.48,wspace=.25)
    fig.suptitle('A changed label assignment does not uniquely determine the remaining drive',
                 x=.085,y=.975,ha='left',fontsize=15,fontweight='bold')
    fig.text(.085,.932,'Frozen checkpoints; paired seed bootstrap 95% intervals. Ratios are intervention mean / S mean.',fontsize=10.4)
    for ax in axs.flat:
        ax.axhline(1,color='#555555',linewidth=1,zorder=1)
        ax.axhline(.5,color='#999999',linewidth=.9,linestyle=(0,(4,3)),zorder=1)
        ax.axhline(0,color='#BBBBBB',linewidth=.7,zorder=1)
        ax.grid(axis='y',color='#E6E6E6',linewidth=.7)
        ax.set_ylabel('Ratio to unchanged head (S)')
        ax.set_axisbelow(True)
    for ax,metric,title in [(axs[0,0],'feature_gradient_norm_mean','A   ViT: feature-gradient norm'),
                            (axs[0,1],'radial_product_mean','B   ViT: radial product with features')]:
        for i,a in enumerate(('P','RC','RR')):
            for task,offset,color,marker in [(1,-.12,'#9B9B9B','s'),(100,.12,'#205C88','o')]:
                draw(ax,i+offset,row('ViT','SL',task,a,metric),color,marker,f'Task {task}' if i==0 else None)
        ax.set_xticks(range(3),['P','RC','RR'])
        ax.set_xlim(-.5,2.5);ax.set_ylim(-.06,1.16);ax.set_title(title,loc='left')
        ax.legend(frameon=False,loc='upper left',fontsize=9.5,ncol=2)
        ax.text(.98,.96,'n = 5 seeds',ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#555555')
    axs[0,0].text(.04,.12,'P/RC retain substantial norm at task 100',transform=axs[0,0].transAxes,fontsize=9.3,color='#205C88')
    axs[0,1].text(.04,.23,'The radial component is much smaller',transform=axs[0,1].transAxes,fontsize=9.3,color='#205C88')
    acts=('R','LR','ELU','SILU','GELU')
    for ax,task,panel in [(axs[1,0],10,'C'),(axs[1,1],30,'D')]:
        for j,a in enumerate(('P','RC','Q')):
            for i,act in enumerate(acts):
                draw(ax,i+(j-1)*.21,row('MLP',act,task,a,'push_signed_mean'),COLORS[a],
                     ('o','s','^')[j],a if i==(1 if task==30 else 0) else None)
        ax.set_xticks(range(5),['ReLU','Leaky\nReLU','ELU','SiLU','GELU'])
        ax.set_xlim(-.55,4.55);ax.set_ylim(-.06,1.85)
        ax.set_title(f'{panel}   MLP: signed preactivation push, task {task}',loc='left')
        ax.legend(frameon=False,loc='upper left',fontsize=9.4,ncol=3)
        ax.text(.98,.96,'n = 10 seeds',ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#555555')
    ax=axs[1,1]
    ax.axvspan(-.45,.4,color='#EEEEEE',zorder=0)
    ax.text(-.02,.71,'S push = 0\n(all seeds)\nratio undefined',ha='center',va='center',fontsize=9,color='#555555',
            bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#BBBBBB',linewidth=.7))
    fig.text(.085,.084,'P: independent class permutations per column.   RC: randomized centered columns.   RR: randomized rows.',fontsize=9.6)
    fig.text(.085,.057,'Q: centered column norms reset to initialization; P/RC/Q preserve sample-mean logits. Dashed line: preregistered 0.5 rule.',fontsize=9.3)
    fig.text(.085,.027,'Head-input gradient norm, radial product and MLP body push are distinct estimands. These frozen probes do not test recovery.',fontsize=9.3,color='#555555')
    for ext in ('png','pdf'):
        path=args.results/f'static_figure.{ext}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__=='__main__':
    main()
