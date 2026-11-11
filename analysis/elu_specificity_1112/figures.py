"""Full registered recovery curves, including the fresh-body reference."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/elu_specificity_1112_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import analyze as A

POLICIES = {
    'P1': ('P, original scale', '#c4473a', '-'),
    'P01': ('P, scale 0.1', '#087e8b', '-'),
    'Z_keep': ('Zero head, body Adam kept', '#6e58a6', '-'),
    'Z_clear': ('Zero head, fresh Adam', '#d18d00', '-'),
    'F0': ('Fresh body + zero head + fresh Adam', '#343434', '--'),
}

def plot_recovery(records, study, out):
    acts = A.ACTS_A if study == 'A' else ('R', 'ELU')
    age = 10 if study == 'A' else 30
    horizon = A.STUDIES[study]['horizon']
    n = len(A.STUDIES[study]['seeds'])
    fig, axes = plt.subplots(2, len(acts), figsize=(15.6 if study == 'A' else 10.5, 7.8),
                             sharex=True, sharey=True, squeeze=False)
    for row, ln in enumerate(('off', 'on')):
        for col, act in enumerate(acts):
            ax = axes[row, col]
            for arm, (label, color, style) in POLICIES.items():
                selected = sorted([r for r in records if r['study']==study and r['activation']==act
                                   and r['ln']==ln and r['t_end']==age and r['arm']==arm], key=lambda r:r['seed'])
                assert len(selected)==n and [r['seed'] for r in selected]==list(A.STUDIES[study]['seeds'])
                steps = np.array([q['step'] for q in selected[0]['curve']])
                values = np.array([[q['acc'] for q in r['curve']] for r in selected])*100
                ax.plot(steps, values.mean(axis=0), color=color, linestyle=style, lw=1.75, label=label)
            ln_label = 'final LN' if study=='A' else 'all-LN history'
            ax.set(title=f'{act} | {ln_label} {ln}', xlim=(0,horizon), ylim=(0,101))
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Full fixed-set training accuracy (%)')
            if row==1:ax.set_xlabel('Updates on the new task')
    title = ('A: MLP trained activation/LN regimes, after task 10' if study=='A' else
             'B: ViT current activation, same ReLU-trained task-30 sources')
    fig.suptitle(title, fontsize=15, y=.985)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.035), ncol=3, frameon=False, fontsize=10)
    fig.text(.5,.012, f'Mean of {n} paired seeds. One-time head edits; all parameters learn. Fresh body uses paired initialization, labels and batches.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.91, bottom=.19, left=.07, right=.985, wspace=.10, hspace=.27)
    for ext in ('png','pdf'):fig.savefig(Path(out)/f'recovery_{study}.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    records=A.load_records(a.raw);A.validate_design(records)
    Path(a.out).mkdir(parents=True,exist_ok=True)
    for study in A.STUDIES:plot_recovery(records,study,a.out)
    print('Saved full-grid recovery curves for both studies')

if __name__=='__main__':main()
