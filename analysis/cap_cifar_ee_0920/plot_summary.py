from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.cifar_interventions_0920 import artifact
p=Path('results/cap_cifar_ee_0920');assert json.loads((p/'status.json').read_text())['stage']=='completed'
rows=list(csv.DictReader((p/'per_task.csv').open()));verdict=json.loads((p/'verdict.json').read_text())
arms=['ref','cap1','cap2','cap12','cap12_bfix'];colors=['#53616e','#b77720','#9052a0','#198076','#245aaf']
fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained');tasks=np.arange(1,51)
for arm,color in zip(arms,colors):
    rr=sorted((r for r in rows if r['arm']==arm),key=lambda r:(int(r['seed']),int(r['task'])))
    assert len(rr)==500
    datasets=[np.array([float(r[k]) for r in rr]).reshape(10,50) for k in ['online_acc','G2']]
    biases=[]
    for t in tasks:
        with np.load(artifact(p/'arms'/arm/f'units_t{t:02d}.npz')) as f:biases.append(f['bias_l2'].mean(1))
    b=np.asarray(biases).T;datasets.append(b-b[:,:1])
    for ax,values in zip(axes,datasets):
        ax.fill_between(tasks,values.min(0),values.max(0),color=color,alpha=.10,lw=0)
        ax.plot(tasks,values.mean(0),color=color,lw=1.8,label=arm)
for ax in axes:
    ax.axvspan(31,50,color='#222222',alpha=.035,zorder=-5);ax.set_xlim(1,50);ax.set_xlabel('Task');ax.grid(alpha=.16)
axes[0].set_ylim(0,1.02);axes[0].set_ylabel('Online accuracy');axes[0].set_title('Learning across tasks');axes[0].legend(fontsize=8)
axes[1].set_ylim(bottom=0);axes[1].set_ylabel('Mean native derivative, layer 2');axes[1].set_title('Layer 2 response')
axes[2].set_ylabel('Mean hidden bias change from task 1');axes[2].set_title('Layer 2 bias movement')
fig.suptitle('CIFAR S5: layerwise growth caps, 50 tasks',fontsize=14)
fig.savefig(p/'summary.png',dpi=180);fig.savefig(p/'summary.pdf')
(p/'figure_notes.md').write_text('Curves are means across 10 seeds; shaded bands show the full seed range, not confidence intervals. The pale right-hand window is the registered task31–50 window. The bias panel averages units within each seed before plotting.\n')
