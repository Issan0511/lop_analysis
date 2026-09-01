# lr_a1_0901 summary

## Verdict

- P1: **INCONCLUSIVE_WIDE**
- P2: **A_WITHOUT_B_HARMLESS_MULTILAYER**
- Scope: condA, width 100, two hidden layers, 5M steps, leaky-ReLU a=0.1.
- Pairing covers initialization, teacher, and input realization only; trajectories diverge after step 1.
- `strict_dead` and `submerged_frac` were not used in the verdict.

## Primary endpoints

- P1 layer-2 dose (late_t451_500, seed median): 6.38423 [6.03999, 6.54635].
- P1 registered decision band: [5.35, 6.35], centered on the closed-form prediction 5.85.
- P1 lambda-corrected reading (report only): **A_CLOSED_FORM_MATCH** against [5.65, 6.65], centered on 6.15.
- L2_A1 layer-2 dose (late_t451_500, seed median): 7.18266 [6.8847, 7.63523].
- P1 raw paired delta vs L2_A1 (late_t451_500, report only): -0.70306 [-1.5166, -0.385174], not used in the registered P1 decision. The registered leaky band does not contain the L2_A1 point estimate.
- P2 raw unfit (late_t451_500, seed median of mean unfit): 0.00262106.
- P2 paired delta log10(mean unfit) vs L2_A1 (late_t451_500): -0.974208 [-1.2008, -0.669348].
- E_A1 reference delta (late_t451_500, recomputed): -0.927471 [-1.09766, -0.695994].
- P2 sign condition (CI upper < 0): PASS.
- P2 descriptive effect-size reading vs registered E_A1 interval [-1.098, -0.696]: WITHIN_E_A1_INTERVAL. Values below the interval retain the harmless label because they indicate stronger improvement.
- The E_A1 interval is descriptive; the registered primary condition is the CI sign.

## Prediction registration and correspondence

The numerical predictions were proposed in the draft first and then approved by Issa; they are not independent Issa predictions.
| endpoint | preregistered prediction | observed window | observed |
|---|---:|---|---:|
| layer-2 dose | 5.9 | late_t451_500 | 6.38423 |
| layer-2 submerged fraction | about 0.60 | step5m_t500 | 0.84 |
| mean unfit | <= 0.005 | late_t451_500 | 0.00262106 |
If a prediction misses, the preregistered suspected cause is: 全然わからない.

## P3 (REPORT_ONLY)

- Layer-2 submerged fraction (step5m_t500, seed median): 0.84.
- Both submerged counts and fractions are in verdict.csv; unit-level mobility and s_i=M_i+B_i are in s_distribution.csv.
- Task-end, per-layer boundary snapshots are in layer_stats.csv.

## Sanity

- Preflight: **PASS**
- Final pairing: **PASS**
- Mask check: 50 task-end points per seed in late_t451_500 and 500 task-end points per seed overall.
- Floor: 1e-23 inherited from mlp2_phase1_0829 / elu_swamp_0830; not recalibrated.

## Citation limits

Do not generalize beyond condA, width 100, two hidden layers, and the 5M horizon. A 0/10 event count is not evidence that an event is impossible; its one-sided 95% upper bound is 0.2589.
