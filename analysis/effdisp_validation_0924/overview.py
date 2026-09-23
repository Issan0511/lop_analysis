"""Post hoc explanatory view: continued growth, frozen response, active plateau.

Selections are illustrative, not additional tests; all groups remain in the
registered tables. Lines are seed medians; shading is the full five-seed range.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/effdisp_validation_0924'
df = pd.read_csv(OUT/'transitions.csv')
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, constrained_layout=True)
for col, (arm, label) in enumerate([('LR','Leaky ReLU: growth'),('R','ReLU: frozen response'),('SNA03','Adaptive Snake c=0.3: active plateau')]):
    for metric, color, linestyle in [('raw','#888888','--'),('effective','#176fba','-')]:
        sub = df[(df.environment=='mnist_rl') & (df.arm==arm) & (df.metric==metric)]
        g = sub.groupby('task').V_next
        a = g.agg(['median','min','max'])
        axes[0,col].plot(a.index,a['median'],color=color,ls=linestyle,label=metric)
        axes[0,col].fill_between(a.index,a['min'],a['max'],color=color,alpha=.12)
    axes[0,col].set(title=label,yscale='log',ylabel='First-layer squared norm / width')
    sub = df[(df.environment=='mnist_rl') & (df.arm==arm) & (df.metric=='effective')]
    for key,color,linestyle,label_acc in [('train_acc','#19753e','-','task-end'),('online_acc','#aa6900','--','within-task online')]:
        a = sub.groupby('task')[key].agg(['median','min','max'])
        axes[1,col].plot(a.index,a['median'],color=color,ls=linestyle,label=label_acc)
        axes[1,col].fill_between(a.index,a['min'],a['max'],color=color,alpha=.1)
    axes[1,col].axhline(.1,color='#666666',lw=.6,ls=':')
    axes[1,col].set(xlabel='Task',ylabel='Training accuracy',ylim=(0,1.025))
    for ax in axes[:,col]:
        ax.axvspan(41,50,color='#dddddd',alpha=.2)
        ax.grid(alpha=.15)
axes[0,0].legend(fontsize=9)
axes[1,0].legend(fontsize=9)
fig.suptitle('RL-MNIST / Adam / 5 fresh seeds\nLines: seed median; band: seed range; shaded tasks 41–50: registered late window',fontsize=12)
fig.savefig(OUT/'overview_rl.png',dpi=180)
plt.close(fig)
