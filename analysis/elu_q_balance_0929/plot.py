"""Plot the saved audit tables; no fitting or training."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--results', type=Path, required=True)
a=ap.parse_args()
def read(name):
    with (a.results/name).open() as f:
        return list(csv.DictReader(f))
norms=read('long_norm_summary.csv')
states=read('states.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
colors=['#2473ad','#ca603a']
for seed,color in zip(('0','1'),colors):
    r=[x for x in norms if x['seed']==seed and x['population']=='all']
    axes[0].plot([int(x['task']) for x in r],[float(x['ratio_median']) for x in r],
                 marker='o',color=color,label=f'Seed {seed}')
    r=[x for x in states if x['seed']==seed and x['act']=='ELU']
    axes[1].plot([int(x['task']) for x in r],
                 [float(x['Q_l1_after_bias_fit'])/float(x['Q_l1_before']) for x in r],
                 marker='o',color=color,label=f'Seed {seed}')
axes[0].set(title='Actual Adam trajectory',xlabel='Task',ylabel='Median per-unit V / R')
axes[0].set_yscale('log'); axes[0].set_ylim(3e-4,1.6)
axes[0].axhline(1,color='#888888',linestyle='--',linewidth=1)
axes[0].text(112,1.13,'Equal squared norms',fontsize=9,color='#666666')
axes[1].set(title='Change only the output bias',xlabel='Task',ylabel='Sum |Q| after bias fit / before')
axes[1].set_yscale('log'); axes[1].set_ylim(.008,1.6)
axes[1].axhline(1,color='#888888',linestyle='--',linewidth=1)
axes[1].text(20,1.12,'No reduction',fontsize=9,color='#666666')
for ax in axes:
    ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
fig.suptitle('ELU Q audit: stored RL-MNIST states, post-hoc, 2 seeds',fontsize=12)
fig.savefig(a.results/'q_diagnostics.png',dpi=180)
