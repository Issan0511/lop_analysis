"""Descriptive frozen-probe figure; reads seed-level CSV, never runs a model.

Top: per-seed ratios of the SAME norm statistic under backward W*0.1 vs W*1,
with forward bank D1, native Jacobian, and actual labels held fixed (D=S,P).
Bottom: saved norm-matched Adam/trunk cosine versus native for forward and
backward bank D1. No new hypothesis tests, bootstrap intervals, or pooled
sample/unit replication are introduced. Ratio display is post hoc descriptive.
Crossed maps display existing actual/native mean and confidence-interval cells.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/readout_probe_plot_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/readout_separation_1111'
SEEDS = {'mlp': list(range(10)), 'vit': list(range(10252010, 10252015))}
STYLE = {'S': ('#176B9A', 'o'), 'P': ('#B05D27', '^')}
NORMS = ('g_trunk_norm', 'adam_keep_trunk_norm', 'adam_clear_trunk_norm')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(df, model, direction, backward, jacobian='native'):
    rows = df[(df.model == model) & (df.label_channel == 'actual') &
              (df.forward_bank == direction+'1') &
              (df.backward_bank == backward) & (df.jacobian == jacobian)]
    assert len(rows) == len(SEEDS[model]) and set(rows.seed) == set(SEEDS[model])
    assert not rows.seed.duplicated().any() and (rows.n_subset == 128).all()
    return rows.set_index('seed').reindex(SEEDS[model])


def finite(values, label):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite value in {label}; seeds must not be dropped')
    return values


def figure_data(df):
    assert len(df) == 2520, 'Requires all 15 completed frozen source assays'
    ratios, cosines, audit = {}, {}, {}
    for model in SEEDS:
        ratios[model], cosines[model], audit[model] = {}, {}, {}
        modes = ['elu_unity'] if model == 'mlp' else ['ln_remove_radial', 'ln_remove_radial_mean']
        for direction in STYLE:
            native = select(df, model, direction, direction+'1')
            scaled = select(df, model, direction, direction+'01')
            assert native.probability_sha256.equals(scaled.probability_sha256)
            a = finite(scaled[list(NORMS)], 'scaled norms')
            b = finite(native[list(NORMS)], 'native norms')
            assert (a > 0).all() and (b > 0).all(), 'Undefined or nonpositive norm ratio'
            ratios[model][direction] = a/b
            cs = []
            for mode in modes:
                altered = select(df, model, direction, direction+'1', mode)
                assert native.probability_sha256.equals(altered.probability_sha256)
                assert (altered.normmatch_status == 'matched').all()
                assert np.allclose(altered.nm_g_trunk_norm, native.g_trunk_norm,
                                   rtol=2e-5, atol=0.), 'Trunk gradient norm match failed'
                for memory in ('keep', 'clear'):
                    v = finite(altered[f'nm_adam_{memory}_trunk_cos_native'], mode+' cosine')
                    assert ((v >= -1) & (v <= 1)).all()
                    cs.append(v)
            cosines[model][direction] = np.column_stack(cs)
            audit[model][direction] = {
                'seeds': SEEDS[model],
                'ratio_mean': dict(zip(NORMS, ratios[model][direction].mean(0).tolist())),
                'ratio_range': dict(zip(NORMS, np.stack([ratios[model][direction].min(0), ratios[model][direction].max(0)], axis=1).tolist())),
                'normmatched_adam_cosine_mean': dict(zip(
                    [f'{j}:{m}' for j in modes for m in ('keep','clear')],
                    cosines[model][direction].mean(0).tolist())),
            }
    return ratios, cosines, audit


def seed_points(ax, values, connect_blocks):
    """Same seed jitter across categories; lines join only declared blocks."""
    n_categories = next(iter(values.values())).shape[1]
    for direction, data in values.items():
        color, marker = STYLE[direction]
        offset = -.13 if direction == 'S' else .13
        jitter = np.linspace(-.055, .055, len(data))
        x = np.arange(n_categories)[None, :] + offset + jitter[:, None]
        for left, right in connect_blocks:
            for i in range(len(data)):
                ax.plot(x[i, left:right], data[i, left:right], color=color,
                        alpha=.18, linewidth=.65, zorder=1)
        for j in range(n_categories):
            ax.scatter(x[:,j], data[:,j], s=26, color=color, marker=marker,
                       alpha=.85, edgecolors='white', linewidths=.4, zorder=3)
            ax.plot([j+offset-.085, j+offset+.085], [data[:,j].mean()]*2,
                    color=color, linewidth=2.7, solid_capstyle='butt', zorder=4)
    ax.set_xlim(-.5, n_categories-.5)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E2E5E8', linewidth=.6)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color('#A2A8AE')
    ax.tick_params(axis='both', length=3, color='#A2A8AE')


def crossed_figure(args):
    """Display existing estimates only; do not recalculate bootstrap intervals."""
    records = pd.read_csv(args.input)
    estimates = pd.read_csv(args.estimates)
    fullset = pd.read_csv(args.fullset)
    assert len(records) == 2520 and len(fullset) == 90
    banks = ('S1','P1','O1')
    panels = {}
    for model, metric in [('mlp','adam_keep_mean_preact_shift'),('vit','g_trunk_norm')]:
        means, lower, upper = (np.empty((3,3)) for _ in range(3))
        for i, forward in enumerate(banks):
            reference = select(records, model, forward[0], forward)
            for j, backward in enumerate(banks):
                rows = select(records, model, forward[0], backward)
                assert rows.probability_sha256.equals(reference.probability_sha256)
                values = finite(rows[metric], f'{model}:{forward}:{backward}:{metric}')
                q = estimates[(estimates.model==model) & (estimates.label_channel=='actual') &
                              (estimates.jacobian=='native') & (estimates.forward_bank==forward) &
                              (estimates.backward_bank==backward) & (estimates.metric==metric)]
                assert len(q)==1
                q=q.iloc[0]
                assert bool(q.estimable) and q.n_seed==len(SEEDS[model]) and q.n_finite==len(SEEDS[model])
                triple=finite([q.estimate,q.ci_low,q.ci_high], 'saved estimate and CI')
                assert triple[1]<=triple[2]
                assert np.isclose(triple[0],values.mean(),rtol=1e-12,atol=1e-14)
                means[i,j],lower[i,j],upper[i,j]=triple
        # The caption concerns the centered Frobenius norm within each source,
        # not the effective LN-affine map, nor equality between different seeds.
        for seed in SEEDS[model]:
            norms=fullset[(fullset.model==model)&(fullset.seed==seed)&fullset.bank.isin(banks)]
            assert len(norms)==3 and set(norms.bank)==set(banks)
            v=finite(norms.head_centered_norm,'head centered Frobenius norm')
            assert np.allclose(v,v[0],rtol=2e-5,atol=0.)
        panels[model]=dict(metric=metric,mean=means,ci_low=lower,ci_high=upper)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':12,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,2,figsize=(11.8,7.2))
    fig.subplots_adjust(left=.09,right=.94,top=.82,bottom=.32,wspace=.39)
    fig.suptitle('Crossed forward probabilities and backward maps',x=.09,y=.975,
                 ha='left',fontsize=18,fontweight='bold')
    fig.text(.09,.922,'Actual new labels · native Jacobian · 10 ELU-MLP seeds / 5 LN-ViT seeds',
             fontsize=10,color='#3D4853')
    for ax,model in zip(axes,('mlp','vit')):
        p=panels[model]
        factor=1000. if model=='mlp' else 1.
        mean,lo,hi=(p[key]*factor for key in ('mean','ci_low','ci_high'))
        if model=='mlp':
            bound=float(np.ceil(np.abs(mean).max()))
            image=ax.imshow(mean,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0.,vmax=bound))
            ax.set_title('A  ELU-MLP · Adam memory kept\nMean preactivation change (×10⁻³)',loc='left',fontweight='bold',pad=13)
        else:
            bound=float(np.ceil(mean.max()/10.)*10.)
            image=ax.imshow(mean,cmap='YlGnBu',vmin=0.,vmax=bound)
            ax.set_title('B  LN-ViT\nTrunk-gradient L2 norm',loc='left',fontweight='bold',pad=13)
        for i in range(3):
            for j in range(3):
                dark=abs(mean[i,j])/bound>.54 if model=='mlp' else mean[i,j]/bound>.55
                color='white' if dark else '#17232D'
                fmt=lambda x:f'{x:+.2f}' if model=='mlp' else f'{x:.1f}'
                ax.text(j,i-.10,fmt(mean[i,j]),ha='center',va='center',fontsize=13,fontweight='bold',color=color)
                ax.text(j,i+.20,f'[{fmt(lo[i,j])}, {fmt(hi[i,j])}]',ha='center',va='center',fontsize=8.7,color=color)
        ax.set_xticks(range(3),banks);ax.set_yticks(range(3),banks)
        ax.set_xlabel('Backward map',labelpad=8)
        ax.set_ylabel('Forward probability bank',labelpad=9)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.3)
        ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        colorbar=fig.colorbar(image,ax=ax,fraction=.046,pad=.04)
        colorbar.ax.tick_params(labelsize=8.5,length=2)
        colorbar.outline.set_visible(False)
    notes=[
        'Each row reuses every example’s complete probability vector within each seed. Across columns, only the backward map changes.',
        'Cells: existing seed mean [unadjusted descriptive 95% seed-bootstrap CI]. Cell intervals describe means, not between-cell differences.',
        'S: learned map. P: independent class-row permutations by column (changes WᵀW). O: class-space orthogonal map (preserves WᵀW).',
        'S1, P1 and O1 have equal centered head Frobenius norm within each source; this does not imply identical residual-to-trunk maps.',
        'ELU: negative means lower full-set recipient preactivation after one virtual Adam step. Gradients use the fixed 128-example subset.',
        'ViT: mean-loss trunk gradient on those 128 examples. These frozen quantities do not assign causal shares of later learning delay.',
    ]
    for i,note in enumerate(notes):fig.text(.09,.239-i*.033,note,fontsize=8.5,color='#3D4853')
    provenance=dict(inputs={str(p):sha(p) for p in (args.input,args.estimates,args.fullset)},
                    analysis_sha256=sha(__file__),description='Existing frozen actual/native means and intervals; no new tests or confidence intervals.')
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'frozen_crossed_maps.png',dpi=220,metadata={'Description':json.dumps(provenance,sort_keys=True)})
    fig.savefig(args.out/'frozen_crossed_maps.pdf',metadata={'Title':'Crossed frozen readout maps','Author':'readout_separation_1111','Subject':json.dumps(provenance,sort_keys=True)})
    plt.close(fig)
    print(json.dumps({'complete':True,'figure':'frozen_crossed_maps','provenance':provenance,
                      'panels':{m:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for m,p in panels.items()}},indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=OUT/'probe_seed_records.csv')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--estimates', type=Path, default=OUT/'probe_estimates.csv')
    ap.add_argument('--fullset', type=Path, default=OUT/'probe_fullset_seed_metrics.csv')
    ap.add_argument('--figure', choices=('both','separation','crossed'), default='both')
    args = ap.parse_args()
    if args.figure != 'separation':
        crossed_figure(args)
        if args.figure == 'crossed':return
    df = pd.read_csv(args.input)
    ratios, cosines, audit = figure_data(df)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.titlesize':12, 'axes.labelsize':10,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.205,
                        hspace=.62, wspace=.22)
    fig.suptitle('Frozen readout separation', x=.085, y=.979, ha='left',
                 fontsize=19, fontweight='bold')
    fig.text(.085, .936,
             'Actual new labels · fixed 128-example gradients · complete probability vectors matched within each comparison',
             fontsize=10, color='#3D4853')
    handles = [Line2D([],[],color=color,marker=marker,linestyle='',markersize=6,label=label)
               for label,(color,marker) in STYLE.items()]
    fig.legend(handles=handles, labels=['S: learned readout', 'P: columnwise class-row permutation'],
               loc='upper left', bbox_to_anchor=(.078,.919), frameon=False, ncol=2,
               columnspacing=2.4, handletextpad=.5)
    for j, (model, title) in enumerate([('mlp','ELU-MLP · 10 seeds'),('vit','LN-ViT · 5 seeds')]):
        ax = axes[0,j]
        seed_points(ax, ratios[model], [(0,3)])
        ax.set_title(chr(ord('A')+j)+'  '+title, loc='left', fontweight='bold', pad=12)
        ax.set_xticks(range(3), ['Trunk\ngradient', 'Adam\nkeep memory', 'Adam\nclear memory'])
        ax.set_ylim(0,1.075);ax.set_yticks([0,.1,.3,.5,.7,.9,1.])
        ax.axhline(.1,color='#747C84',linestyle=(0,(3,3)),linewidth=.8,zorder=0)
        ax.axhline(1.,color='#747C84',linestyle=(0,(2,3)),linewidth=.8,zorder=0)
        ax.set_ylabel('Norm ratio: backward scale 0.1 / 1')
    for j, model in enumerate(('mlp','vit')):
        ax=axes[1,j]
        blocks=[(0,2)] if model=='mlp' else [(0,2),(2,4)]
        seed_points(ax, cosines[model], blocks)
        title = 'C  ELU derivative set to 1' if model=='mlp' else 'D  Altered final-LN input VJP'
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        labels = ['Adam\nkeep memory','Adam\nclear memory'] if model=='mlp' else [
            'Adam keep', 'Adam clear', 'Adam keep', 'Adam clear']
        ax.set_xticks(range(len(labels)),labels,fontsize=9)
        ax.set_ylim(0,1.045);ax.set_yticks(np.arange(0,1.01,.2))
        ax.axhline(1.,color='#747C84',linestyle=(0,(2,3)),linewidth=.8,zorder=0)
        ax.set_ylabel('Adam-step cosine versus native\n(trunk-gradient norm matched)')
        if model=='vit':
            ax.axvline(1.5,color='#D3D7DB',linewidth=.8,zorder=0)
            ax.text(.5,-.18,'Remove radial',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
            ax.text(2.5,-.18,'Remove radial + mean',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
    notes = [
        'Dots are seeds; bars are seed means; faint lines join the same seed. Top ratios are paired within seed and post hoc descriptive.',
        'Top: ratios of L2 norms at backward banks D01 / D1 (D = S or P), with forward bank D1 and native Jacobian fixed.',
        'Bottom: cosine of virtual Adam trunk steps after matching altered/native trunk-gradient L2 norms; forward/backward bank = S1 or P1.',
        '"Remove" omits the indicated LN subtraction. Forward outputs and final-LN affine gradients stay native; affine parameters are excluded from trunk.',
        'All Adam steps are frozen virtual updates, not observed training-minibatch displacements. No new tests or intervals; no cross-architecture causal comparison.',
    ]
    for i, note in enumerate(notes):fig.text(.085,.139-i*.025,note,fontsize=8.15,color='#3D4853')
    args.out.mkdir(parents=True,exist_ok=True)
    provenance = dict(input_file=str(args.input), input_sha256=sha(args.input),
                      analysis_sha256=sha(__file__), description='Post hoc descriptive seed-level visualization; actual labels only; no additional inference.')
    fig.savefig(args.out/'frozen_separation.png',dpi=220,
                metadata={'Description':json.dumps(provenance,sort_keys=True)})
    fig.savefig(args.out/'frozen_separation.pdf',metadata={
        'Title':'Frozen readout separation','Author':'readout_separation_1111',
        'Subject':json.dumps(provenance,sort_keys=True)})
    plt.close(fig)
    print(json.dumps({'complete':True,'provenance':provenance,'seed_summary':audit},indent=2))

if __name__=='__main__':main()
