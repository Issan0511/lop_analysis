# leaky_cliff_1111 — new random-label MNIST experiment

The main outcome is substantial slowing of relearning at small positive leak, with recovery under a larger budget or a one-time head reset. A universal derivative-mean threshold at 0.1 and an irreversible loss of trainability are not supported for this setting. The remembered CondA run is NOT identified or replicated here.

## Design and completion

Frozen spec: `specs/spec_leaky_cliff_1111.md`, preregistration commit `73338cd8`; training code commit `18b4681fec66d42617da7814ae625852d3ca108d`. All 96 registered trajectories completed: 21 discovery, 15 independent confirmation, 60 branches. The conditional B1/B2 follow-up was not triggered because B-N at6k was .06117, below .15. Runtime 12.50 minutes, at most5 single-thread CPU workers; no GPU.

Fixed1200 MNIST images per seed; iid random labels change each task; 784–100–100–10; Adam .001; batch16; 6000 updates/task for20 tasks. Discovery a={0,.003,.01,.03,.1,.3,1}, seeds0–2. Selected .03/.1 from the largest adjacent positive nonlinear late-online gap; independent seeds3–7 and geometric midpoint .0547723. All init, data subset, labels and minibatch RNG are paired across a. No lr tuning, standardization or weight decay. These are training/memorization accuracies, not held-out generalization.

Online = pre-update minibatch accuracy over all6000 updates. Endpoint = full fixed-bank accuracy exactly at the budget. Late = mean of tasks16–20 within seed. Confidence intervals use paired seed differences and t(df4), not pooled discovery+confirmation or epochs as independent samples. Secondary contrast intervals are descriptive and unadjusted for multiplicity.

## Natural trajectories

| stage | a | late_endpoint | late_online |
| --- | --- | --- | --- |
| confirmation | 0.03 | 56.61% | 35.64% |
| confirmation | 0.0547722557505166 | 66.74% | 41.39% |
| confirmation | 0.1 | 83.41% | 51.20% |
| discovery | 0.0 | 11.19% | 10.78% |
| discovery | 0.003 | 30.58% | 20.95% |
| discovery | 0.01 | 37.02% | 24.34% |
| discovery | 0.03 | 54.34% | 34.37% |
| discovery | 0.1 | 84.24% | 51.82% |
| discovery | 0.3 | 98.19% | 66.72% |
| discovery | 1.0 | 85.65% | 58.01% |

Independent .1 minus .03 late-online contrast: 15.57 pp [12.53, 18.60] (95% paired t interval). Midpoint lies between endpoints, so this grid gives no evidence of a discontinuity. a1 is an affine network represented as three matrix factors, not a single-layer baseline; its performance is below a.3 despite derivative1 everywhere. Per-seed steepest drops and pre/post geometry are in `natural_events.csv`, without seed-averaged event selection. The largest drop in discovery .003 occurs task3→4 for all3 seeds; preceding layer1 means span .0578–.1653, layer2 .1979–.3954. No common .1 boundary follows from that event statistic.

## Same-state task21 branches

All arms from the SAME task20 source for each seed except explicitly fresh arms; task21 labels and minibatch stream paired. Both budget columns train on the SAME new labels. Each reset is ONCE. Reverse arms use the independently trained .1 source and were registered for6k only.

| arm | intervention | 6k accuracy | 24k accuracy |
| --- | --- | --- | --- |
| N | Native continuation .03 | 54.37% | 99.38% |
| A | Ordinary forward slope .1 | 65.33% | 99.48% |
| B | Native forward .03, backward .1 | 60.48% | 99.67% |
| X | Anchored forward .1 | 90.48% | 98.52% |
| O | All Adam reset | 53.72% | 99.15% |
| H | Head zero + all Adam reset | 97.35% | 100.00% |
| BO | B + all Adam reset | 56.67% | 99.52% |
| Fbad | Fresh .03 | 100.00% | 100.00% |
| Fgood | Fresh .1 | 96.13% | 100.00% |
| Ngood | Native continuation .1 | 83.68% | not run (registered 6k) |
| Breverse | Native forward .1, backward .03 | 64.23% | not run (registered 6k) |
| Xreverse | Anchored forward .03 | 63.42% | not run (registered 6k) |

N reaches90% in all5 seeds after12450–16800 updates (median14400); fresh .03 takes1350–1425 (median1425). The fresh .03 minus N endpoint gap contracts 98.65% from6k to24k (95.7–100% contraction per seed). N24k is97.92–100% across seeds. This is direct evidence for budget-sensitive slowing, not for a permanent capacity loss in these states. It does not show that allocating24k to EVERY task prevents future deterioration.

B keeps the natural .03 forward at every CURRENT parameter value but uses .1 in backward. X preserves initial hidden activations and logits using frozen per-input offsets, then evolves with a different .1 forward. B and X start with the same gradient/Adam step, while their true forward responses differ. X's offsets also change the function class; it is not an ordinary leaky network or a pure-backward intervention. A changes the initial forward, so its gain mixes response and initial-function changes.

B-N at6k: 6.12 pp [3.67, 8.56]; X-N: 36.12 pp [32.23, 40.01]; O-N: -0.65 pp [-2.15, 0.85]; H-O: 43.63 pp [39.86, 47.41]. Reverse B-Ngood: -19.45 pp [-21.05, -17.85]. Thus larger backward slope has a modest positive recovery effect and a larger reverse impairment; the preregistered symmetric signal criterion is NOT met. Optimizer reset alone does not rescue; head reset given optimizer reset does. H continues to train the body, so this is not a frozen-representation capacity test, and no head-only/no-Adam-reset arm was run.

T90 means FIRST crossing, not sustained performance: for example fresh .1 seed4 crosses90% early but is80.67% exactly at6k. Some long curves fluctuate. `censored_times.csv` preserves right censoring: all15 reverse/source6k runs fail to reach90% within that budget; they are not assigned infinite time.

## Derivatives, parameter motion and useful response

For each unit, p=P_x(z>0), mean(g)=a+(1-a)p, mean(g²)=a²+(1-a²)p, Var_x(g)=(1-a)²p(1-p). Means are inputs first, then units within each layer; true-forward and surrogate-backward derivatives are separate columns. At task20 discovery a.03, near-one-branch fractions average83.3%/84.0% in layers1/2; a.1 gives43.3%/24.0%; a.3 gives2.0%/.67%. These correlations do not identify causal mediation. Positive leak is not literal dead-unit absorption. ReLU's zero first-layer response is a distinct endpoint.

N6k mean derivatives are .0540/.0613 and its actual layer1 parameter-step norm is .0531; B .05297/.06557 and step .05183. Parameters still move. The recorded true J delta, realized centered-logit displacement, error alignment and full-bank CE decrease distinguish useful motion from weight motion. There is no single ordering of the sampled functional projections that explains whole-run speed: samples differ in residual error and phase. In X's first step, mean linear-predicted CE decrease .08355 coexists with realized CE INCREASE .01727, so a local linear predictor must not be treated as exact finite-step improvement.

At N6k, layer2/head epsilon-dominated fractions are0; almost all layer1 epsilon-dominated coordinates correspond to pixels that are identically black in the fixed input bank. After excluding those, layer1 fractions are at most0.153%. This does not support broad epsilon-dominated stopping here. Actual floating-point parameter displacement is measured separately from intended Adam displacement.

At the first task21 gradient, multiplying all gradients by .01 leaves layer1 update norms at .9919 of baseline when the entire Adam history is consistently scaled, and .9954 with fresh moments, but .5978 with stale moments. This is a shadow algebraic comparison, not an SGD training run, nor an intervention equivalent to changing a. The exact history-scaling identity is Delta=-eta*mhat/(sqrt(vhat)+eps/c); only appropriate history and negligible eps permit cancellation.

Registered matched-natural-mean criterion: 29 matched-task/seed pairs have both means within .02; maximum endpoint gap is18.33pp, below the preregistered20pp cutoff. Report that non-result. A separate POST-HOC check finds 3 seeds recovering above90% while BOTH means remain below.1:

| seed | step | acc | ce | L1_true_gmean | L2_true_gmean |
| --- | --- | --- | --- | --- | --- |
| 3.0 | 18000.0 | 0.920833 | 0.35873 | 0.059884 | 0.062341 |
| 5.0 | 17250.0 | 0.911667 | 0.357711 | 0.067402 | 0.067927 |
| 6.0 | 14250.0 | 0.915 | 0.345868 | 0.066092 | 0.083245 |

Derivative participation n_eff=(mean g)²/mean(g²) has minimum4a/(1+a)² and tends1 for both all-negative and all-positive units when a>0. For a.1 its minimum is.33058, so the historical ELU calibration.119 cannot be transported literally. The high-participation all-negative limit is locally affine on the observed inputs, not high nonlinear expressivity. See `analysis/leaky_cliff_1111/interpretation.md` for derivation, conditional Gaussian drift equation, primary sources and corrections to older notes.

## Historical reanalysis and limits

`historical_elu_*` is POST HOC and a different experiment from the remembered CondA. Existing10-seed random-label ELU data at task6 show mean layer2 derivative .0975, endpoint accuracy .9277, but same-task online .5552 and next-task online .3516. Nine seeds have their largest online drop6→7; one4→5. This explains how endpoint fit and subsequent adaptation can separate; it does not identify the user's original figure or prove a universal .1 threshold. Source arrays are verified against the older backup manifest.

No thermodynamic phase transition, bifurcation, hysteresis, universal optimal a, permanent capacity ceiling, eigenvalue collapse, or per-task preventive reset claim is established. Conclusions are for finite100-wide Adam networks at the registered lr, data size, labels and budgets. Frozen-kernel exponential error-mode decay is only an analogy for budget dependence, not a fitted model of these changing nonlinear trajectories. Capacity would require a separate, precisely defined test.

## Reproduce and inspect

`python analysis/leaky_cliff_1111/analyze.py`, then `python analysis/leaky_cliff_1111/audit.py`, then `python analysis/leaky_cliff_1111/report.py` rebuild tables, figures and this report. Dependencies: Python3.12, torch2.13, numpy, pandas, matplotlib; no scipy. Main run: `python analysis/leaky_cliff_1111/launch.py` from a clean tracked analysis/code tree. Run output is resumable only with matching spec/code hashes. For a genuinely fresh rerun use a clean result destination; the launcher otherwise reuses completed trajectories.

Raw states/arrays are archived under `/home/issan/Projects/obsidian-research-data/leaky_cliff_1111/`; `backup_manifest.json` maps original relative paths to sha256-verified backups. The audit resolves archived checkpoints transparently. Historical arrays retain their original backup destinations. The integrity audit verifies96 completions, finite recorded accuracy/CE, code/spec/data identities, paired task20 RNG, analytic mean identity and bitwise B/X initial-forward equality. Pre-run gates verify native leaky backward, float64 finite-difference JVP (max error1.52e-12), host Adam arithmetic (exact), B/X initial gradients and state/RNG restart.

Key files: `leaky_cliff_summary.png/pdf`, `natural_sweep.png/pdf`, `same_state_branches.png/pdf`, `dense_diagnostics.png/pdf`, `branch_mechanism.png/pdf`, `unit_distribution.png/pdf`, `historical_elu.png/pdf`; all seed tables, registered contrasts, event/distribution tables, `verdict.csv`, `integrity.json`, and each run's provenance, full learning curves and logs.
