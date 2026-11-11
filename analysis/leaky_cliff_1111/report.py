#!/usr/bin/env python3
"""Generate reviewable result tables and a compact report from completed outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import analyze as A
O=A.OUT
def md_table(df):
    cols=list(df.columns);out=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in df.iterrows():out.append('| '+' | '.join(str(row[c]) for c in cols)+' |')
    return '\n'.join(out)
def main():
    integrity=json.loads((O/'integrity.json').read_text());assert integrity['all_pass']
    group=pd.read_csv(O/'natural_group_summary.csv');confirm=pd.read_csv(O/'confirmation_contrasts.csv')
    bg=pd.read_csv(O/'branch_group_summary.csv');bc=pd.read_csv(O/'branch_contrasts.csv');bs=pd.read_csv(O/'branch_seed_summary.csv')
    b6=bg[bg.metric=='acc_6000'].set_index('arm');b24=bg[bg.metric=='acc_24000'].set_index('arm')
    def contrast(arm,ref,metric='acc_6000'):
        return bc[(bc.arm==arm)&(bc.reference==ref)&(bc.metric==metric)].iloc[0]
    def ci(row):return f"{100*row['mean']:.2f} pp [{100*row['lo']:.2f}, {100*row['hi']:.2f}]"
    cg=confirm[(confirm.metric=='late_online')&np.isclose(confirm.a_high,.1)&np.isclose(confirm.a_low,.03)].iloc[0]
    nat=group[group.metric.isin(['late_online','late_endpoint'])].pivot(index=['stage','a'],columns='metric',values='mean').reset_index()
    for k in ['late_online','late_endpoint']:nat[k]=nat[k].map(lambda x:f'{100*x:.2f}%')
    arms=['N','A','B','X','O','H','BO','Fbad','Fgood','Ngood','Breverse','Xreverse']
    labels={'N':'Native continuation .03','A':'Ordinary forward slope .1','B':'Native forward .03, backward .1','X':'Anchored forward .1','O':'All Adam reset','H':'Head zero + all Adam reset','BO':'B + all Adam reset','Fbad':'Fresh .03','Fgood':'Fresh .1','Ngood':'Native continuation .1','Breverse':'Native forward .1, backward .03','Xreverse':'Anchored forward .03'}
    tab=[]
    for a in arms:
        tab.append({'arm':a,'intervention':labels[a],'6k accuracy':f'{b6.loc[a,"mean"]*100:.2f}%','24k accuracy':f'{b24.loc[a,"mean"]*100:.2f}%' if a in b24.index else 'not run (registered 6k)'})
    # This check of the literal 0.1 cutoff was added after seeing the registered result.
    below=[]
    for p in (O/'branches/N').glob('s*/diagnostics.csv'):
        d=pd.read_csv(p);c=pd.read_csv(p.parent/'curves.csv');d=d.merge(c,on=['task','step']);d['seed']=int(p.parent.name[1:])
        q=d[(d.acc>=.9)&(d.L1_true_gmean<.1)&(d.L2_true_gmean<.1)]
        if len(q):below.append(q.iloc[0][['seed','step','acc','ce','L1_true_gmean','L2_true_gmean']].to_dict())
    below=pd.DataFrame(below);below.to_csv(O/'below_point1_recovery_posthoc.csv',index=False)
    gap6=b6.loc['Fbad','mean']-b6.loc['N','mean'];gap24=b24.loc['Fbad','mean']-b24.loc['N','mean'];shrink=1-gap24/gap6
    verdicts=[
        dict(hypothesis='Confirmation of selected a contrast',decision='replicated at independent seeds',evidence=ci(cg),limit='5 paired seeds; fixed Adam lr and 6k budget'),
        dict(hypothesis='Hbudget',decision='criterion met in all 5 seeds',evidence=f'fresh-minus-N gap {gap6:.6f} to {gap24:.6f}; group-mean contraction {shrink:.6f}; N T90 12450–16800',limit='one task21 from task20 state; not every-task24k prevention or arbitrary-task guarantee'),
        dict(hypothesis='Hmoment: O minus N >= .15',decision='criterion not met',evidence=ci(contrast('O','N')),limit='reset all Adam states once; no blanket exclusion of optimizer interactions'),
        dict(hypothesis='Hsignal: B rescue >= .15 and reverse harm <= -.15',decision='joint criterion not met; asymmetric effects',evidence='B-N '+ci(contrast('B','N'))+'; Breverse-Ngood '+ci(contrast('Breverse','Ngood')),limit='positive rescue small, reverse harm large; no mediation percentage'),
        dict(hypothesis='H over O',decision='large head-reset benefit conditional on Adam reset',evidence=ci(contrast('H','O')),limit='body continues training; not frozen-feature capacity or head-only optimizer-retained control'),
        dict(hypothesis='X versus B',decision='anchored forward change gives larger rescue',evidence='X-N '+ci(contrast('X','N'))+' versus B-N '+ci(contrast('B','N')),limit='X has fixed input-dependent offsets and changes future forward; not pure backward rescue'),
        dict(hypothesis='Hmean matched natural means',decision='registered counterexample criterion not met',evidence=f"{integrity['matched_mean_pairs']} pairs within .02 per layer; largest endpoint gap {integrity['matched_mean_largest_endpoint_gap']:.6f} < .20",limit='does not establish a universal cutoff or sufficient scalar description'),
        dict(hypothesis='Literal .1 stopping boundary',decision='post-hoc counterexamples under extended budget',evidence=f'{len(below)}/5 seeds have >90% accuracy with both layer means below .1',limit='descriptive check added after main run; not the registered matched-mean criterion'),
        dict(hypothesis='Hnonlinear / largest derivative best',decision='a1 below a.3 in discovery',evidence='late online .580115 vs .667178; endpoint .856500 vs .981889',limit='a1 changes function class; fixed lr; one-sidedness is not identified mediation'),
        dict(hypothesis='Sharp phase transition',decision='not established',evidence='independent midpoint online .413907 between .356360 and .512025',limit='finite width, sparse slope grid, no hysteresis or size scaling')]
    pd.DataFrame(verdicts).to_csv(O/'verdict.csv',index=False)
    doc=f'''# leaky_cliff_1111 — new random-label MNIST experiment

The main outcome is substantial slowing of relearning at small positive leak, with recovery under a larger budget or a one-time head reset. A universal derivative-mean threshold at 0.1 and an irreversible loss of trainability are not supported for this setting. The remembered CondA run is NOT identified or replicated here.

## Design and completion

Frozen spec: `specs/spec_leaky_cliff_1111.md`, preregistration commit `73338cd8`; training code commit `{integrity['training_git_hash']}`. All 96 registered trajectories completed: 21 discovery, 15 independent confirmation, 60 branches. The conditional B1/B2 follow-up was not triggered because B-N at6k was .06117, below .15. Runtime {integrity['seconds']/60:.2f} minutes, at most5 single-thread CPU workers; no GPU.

Fixed1200 MNIST images per seed; iid random labels change each task; 784–100–100–10; Adam .001; batch16; 6000 updates/task for20 tasks. Discovery a={{0,.003,.01,.03,.1,.3,1}}, seeds0–2. Selected .03/.1 from the largest adjacent positive nonlinear late-online gap; independent seeds3–7 and geometric midpoint .0547723. All init, data subset, labels and minibatch RNG are paired across a. No lr tuning, standardization or weight decay. These are training/memorization accuracies, not held-out generalization.

Online = pre-update minibatch accuracy over all6000 updates. Endpoint = full fixed-bank accuracy exactly at the budget. Late = mean of tasks16–20 within seed. Confidence intervals use paired seed differences and t(df4), not pooled discovery+confirmation or epochs as independent samples. Secondary contrast intervals are descriptive and unadjusted for multiplicity.

## Natural trajectories

{md_table(nat)}

Independent .1 minus .03 late-online contrast: {ci(cg)} (95% paired t interval). Midpoint lies between endpoints, so this grid gives no evidence of a discontinuity. a1 is an affine network represented as three matrix factors, not a single-layer baseline; its performance is below a.3 despite derivative1 everywhere. Per-seed steepest drops and pre/post geometry are in `natural_events.csv`, without seed-averaged event selection. The largest drop in discovery .003 occurs task3→4 for all3 seeds; preceding layer1 means span .0578–.1653, layer2 .1979–.3954. No common .1 boundary follows from that event statistic.

## Same-state task21 branches

All arms from the SAME task20 source for each seed except explicitly fresh arms; task21 labels and minibatch stream paired. Both budget columns train on the SAME new labels. Each reset is ONCE. Reverse arms use the independently trained .1 source and were registered for6k only.

{md_table(pd.DataFrame(tab))}

N reaches90% in all5 seeds after12450–16800 updates (median14400); fresh .03 takes1350–1425 (median1425). The fresh .03 minus N endpoint gap contracts {shrink*100:.2f}% from6k to24k (95.7–100% contraction per seed). N24k is97.92–100% across seeds. This is direct evidence for budget-sensitive slowing, not for a permanent capacity loss in these states. It does not show that allocating24k to EVERY task prevents future deterioration.

B keeps the natural .03 forward at every CURRENT parameter value but uses .1 in backward. X preserves initial hidden activations and logits using frozen per-input offsets, then evolves with a different .1 forward. B and X start with the same gradient/Adam step, while their true forward responses differ. X's offsets also change the function class; it is not an ordinary leaky network or a pure-backward intervention. A changes the initial forward, so its gain mixes response and initial-function changes.

B-N at6k: {ci(contrast('B','N'))}; X-N: {ci(contrast('X','N'))}; O-N: {ci(contrast('O','N'))}; H-O: {ci(contrast('H','O'))}. Reverse B-Ngood: {ci(contrast('Breverse','Ngood'))}. Thus larger backward slope has a modest positive recovery effect and a larger reverse impairment; the preregistered symmetric signal criterion is NOT met. Optimizer reset alone does not rescue; head reset given optimizer reset does. H continues to train the body, so this is not a frozen-representation capacity test, and no head-only/no-Adam-reset arm was run.

T90 means FIRST crossing, not sustained performance: for example fresh .1 seed4 crosses90% early but is80.67% exactly at6k. Some long curves fluctuate. `censored_times.csv` preserves right censoring: all15 reverse/source6k runs fail to reach90% within that budget; they are not assigned infinite time.

## Derivatives, parameter motion and useful response

For each unit, p=P_x(z>0), mean(g)=a+(1-a)p, mean(g²)=a²+(1-a²)p, Var_x(g)=(1-a)²p(1-p). Means are inputs first, then units within each layer; true-forward and surrogate-backward derivatives are separate columns. At task20 discovery a.03, near-one-branch fractions average83.3%/84.0% in layers1/2; a.1 gives43.3%/24.0%; a.3 gives2.0%/.67%. These correlations do not identify causal mediation. Positive leak is not literal dead-unit absorption. ReLU's zero first-layer response is a distinct endpoint.

N6k mean derivatives are .0540/.0613 and its actual layer1 parameter-step norm is .0531; B .05297/.06557 and step .05183. Parameters still move. The recorded true J delta, realized centered-logit displacement, error alignment and full-bank CE decrease distinguish useful motion from weight motion. There is no single ordering of the sampled functional projections that explains whole-run speed: samples differ in residual error and phase. In X's first step, mean linear-predicted CE decrease .08355 coexists with realized CE INCREASE .01727, so a local linear predictor must not be treated as exact finite-step improvement.

At N6k, layer2/head epsilon-dominated fractions are0; almost all layer1 epsilon-dominated coordinates correspond to pixels that are identically black in the fixed input bank. After excluding those, layer1 fractions are at most0.153%. This does not support broad epsilon-dominated stopping here. Actual floating-point parameter displacement is measured separately from intended Adam displacement.

At the first task21 gradient, multiplying all gradients by .01 leaves layer1 update norms at .9919 of baseline when the entire Adam history is consistently scaled, and .9954 with fresh moments, but .5978 with stale moments. This is a shadow algebraic comparison, not an SGD training run, nor an intervention equivalent to changing a. The exact history-scaling identity is Delta=-eta*mhat/(sqrt(vhat)+eps/c); only appropriate history and negligible eps permit cancellation.

Registered matched-natural-mean criterion: {integrity['matched_mean_pairs']} matched-task/seed pairs have both means within .02; maximum endpoint gap is18.33pp, below the preregistered20pp cutoff. Report that non-result. A separate POST-HOC check finds {len(below)} seeds recovering above90% while BOTH means remain below.1:

{md_table(below.round(6))}

Derivative participation n_eff=(mean g)²/mean(g²) has minimum4a/(1+a)² and tends1 for both all-negative and all-positive units when a>0. For a.1 its minimum is.33058, so the historical ELU calibration.119 cannot be transported literally. The high-participation all-negative limit is locally affine on the observed inputs, not high nonlinear expressivity. See `analysis/leaky_cliff_1111/interpretation.md` for derivation, conditional Gaussian drift equation, primary sources and corrections to older notes.

## Historical reanalysis and limits

`historical_elu_*` is POST HOC and a different experiment from the remembered CondA. Existing10-seed random-label ELU data at task6 show mean layer2 derivative .0975, endpoint accuracy .9277, but same-task online .5552 and next-task online .3516. Nine seeds have their largest online drop6→7; one4→5. This explains how endpoint fit and subsequent adaptation can separate; it does not identify the user's original figure or prove a universal .1 threshold. Source arrays are verified against the older backup manifest.

No thermodynamic phase transition, bifurcation, hysteresis, universal optimal a, permanent capacity ceiling, eigenvalue collapse, or per-task preventive reset claim is established. Conclusions are for finite100-wide Adam networks at the registered lr, data size, labels and budgets. Frozen-kernel exponential error-mode decay is only an analogy for budget dependence, not a fitted model of these changing nonlinear trajectories. Capacity would require a separate, precisely defined test.

## Reproduce and inspect

`python analysis/leaky_cliff_1111/analyze.py`, then `python analysis/leaky_cliff_1111/audit.py`, then `python analysis/leaky_cliff_1111/report.py` rebuild tables, figures and this report. Dependencies: Python3.12, torch2.13, numpy, pandas, matplotlib; no scipy. Main run: `python analysis/leaky_cliff_1111/launch.py` from a clean tracked analysis/code tree. Run output is resumable only with matching spec/code hashes. For a genuinely fresh rerun use a clean result destination; the launcher otherwise reuses completed trajectories.

Raw states/arrays are archived under `/home/issan/Projects/obsidian-research-data/leaky_cliff_1111/`; `backup_manifest.json` maps original relative paths to sha256-verified backups. The audit resolves archived checkpoints transparently. Historical arrays retain their original backup destinations. The integrity audit verifies96 completions, finite recorded accuracy/CE, code/spec/data identities, paired task20 RNG, analytic mean identity and bitwise B/X initial-forward equality. Pre-run gates verify native leaky backward, float64 finite-difference JVP (max error1.52e-12), host Adam arithmetic (exact), B/X initial gradients and state/RNG restart.

Key files: `leaky_cliff_summary.png/pdf`, `natural_sweep.png/pdf`, `same_state_branches.png/pdf`, `dense_diagnostics.png/pdf`, `branch_mechanism.png/pdf`, `unit_distribution.png/pdf`, `historical_elu.png/pdf`; all seed tables, registered contrasts, event/distribution tables, `verdict.csv`, `integrity.json`, and each run's provenance, full learning curves and logs.
'''
    (O/'summary.md').write_text(doc)
    print('summary, verdict, and post-hoc counterexample table written')
if __name__=='__main__':main()
