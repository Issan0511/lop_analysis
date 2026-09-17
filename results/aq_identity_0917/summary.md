# CondA A/Q identity audit

Verdict: **FAIL**

Registered audit on previously studied checkpoints. 4 leaky arms × 4 checkpoints × 10 seeds × 32 input support points; 100 units, three fixed-state learning rates, two dtypes. Also 256-step fixed-task continuations from step 200000 for each arm/dtype.

Full W, row-centered W, and CondA's free five coordinates are distinct. Results below use row centering unless stated otherwise.

| dtype | fixed-state unit updates | max absolute budget error (incl. continuations) | max error / rounding bound | resolved signs | sign mismatches |
|---|---:|---:|---:|---:|---:|
| float64 | 1536000 | 1.86114891e-14 | 0.023494 | 1536000 | 0 |
| float32 | 1536000 | 2.83980464e-06 | 0.00585862 | 447382 | 0 |

## What was checked

Expanded residual/readout/gate/input formulas were compared with an independent autograd network; A, Q, S+H-B-C, production handwritten gradients, actual optimizer displacement, radial/orthogonal geometry, and accumulated short-segment budgets were checked separately. The 32-sample Gram formula was compared with an independently differentiated full-batch loss.

Total failed scalar checks: 100. Mutation controls: {'missing_MSE_factor_2': 1535985, 'full_z_instead_of_centered_z': 1535991, 'uncentered_input_in_Q': 1529441, 'mean_sample_Q_instead_of_batch_Q': 15984, 'missing_H_on_offset': 384000}.

## Pre-update branches (float64, fixed-state probes)

Counts pool checkpoint/seed/sample/unit/lr probe events, not independent experimental replicates or longitudinal unit fates.

| branch | events | significant shrink | significant growth | sign mismatch |
|---|---:|---:|---:|---:|
| positive | 393405 | 199570 | 193835 | 0 |
| negative | 1142595 | 574403 | 568192 | 0 |

## Short continuation budgets (row-centered, sum over saved seeds/units)

Newly sampled 256-step fixed-task continuations, not original-stream replays or complete tasks.

| arm | dtype | linear contribution | quadratic contribution | observed endpoint change | max unit discrepancy |
|---|---|---:|---:|---:|---:|
| LR_a0p1_q0 | float64 | 0.313464224 | 0.00864628154 | 0.322110505 | 1.09773e-14 |
| LR_a0p1_q0 | float32 | 0.313461502 | 0.00864628585 | 0.322095286 | 5.91242e-06 |
| LR_a0p3_q0 | float64 | 0.5825162 | 0.0832100224 | 0.665726222 | 1.83564e-14 |
| LR_a0p3_q0 | float32 | 0.58252004 | 0.0832100379 | 0.665709654 | 8.06074e-06 |
| LR_a0p7_q0 | float64 | 0.624811499 | 2.83097954 | 3.45579104 | 1.86795e-14 |
| LR_a0p7_q0 | float32 | 0.624811126 | 2.83097916 | 3.45584199 | 8.77153e-06 |
| LR_a0p3_qp05 | float64 | 1.67130627 | 0.608526517 | 2.27983278 | 2.39262e-14 |
| LR_a0p3_qp05 | float32 | 1.67130588 | 0.608526456 | 2.27990074 | 1.73564e-05 |

## Limits

- PASS validates the identity and implementation mapping, not a general branch-specific contraction mechanism, long-run learning-rate dependence, or LoP causality.
- Float32 updates round; changes below the registered bound are unresolved, not sign matches. Float64 diagnostics on stored native weights avoid extra float32 norm-reduction cancellation.
- Full-batch Q is squared mean gradient, not mean squared sample gradient. Expected batch-1 Q uses the latter.
- Positive z does not imply positive centered zc. Replacing zc by z is a rejected control.
- Every learning-rate probe starts from the same checkpoint; this is not an equal-horizon learning-rate training sweep.

Reproduce: OMP_NUM_THREADS=1 python3 analysis/aq_identity_0917/verify.py
Complete per-unit arrays and input hashes are indexed by backup_manifest.json and provenance.json.

## Diagnostic follow-up (post-result, completed 2026-09-18)

The original aggregate **FAIL is retained**. All 100 failed scalar comparisons were reproduced. They came from float32 forward evaluation with elementwise sum versus the production einsum reduction, not from an incorrect A/Q identity. When autograd uses the production forward order, all comparisons pass at the unchanged tolerance; the production W gradient and autograd W gradient are identical. The two original forward paths differed in output by up to 3.09944153e-06; they had zero activation-gate disagreements. These are counts of scalar comparisons, not independent samples. The diagnosis was specified in a separate post-result addendum.

### Individual float64 identities

| check | scalar comparisons | maximum absolute error | failures |
|---|---:|---:|---:|
| row_centered/A_vs_autograd | 1536000 | 1.5187851e-13 | 0 |
| row_centered/Q_vs_autograd | 1536000 | 2.84217094e-13 | 0 |
| row_centered/A_SHBC | 1536000 | 5.32907052e-15 | 0 |
| batch32/Q_vs_autograd | 16000 | 1.59872116e-14 | 0 |
| row_centered/norm_budget | 2560000 | 1.86114891e-14 | 0 |
| row_centered/cumulative_budget | 4000 | 2.39261735e-14 | 0 |

### Float32 numerical resolution (fixed-state probes)

| dtype | unresolved changes | total | max relative error over resolved updates | p99 relative error over resolved updates |
|---|---:|---:|---:|---:|
| float64 | 0 | 1536000 | 0.00281525 | 1.32296e-07 |
| float32 | 1088618 | 1536000 | 0.00487853 | 0.00162391 |

Relative error here divides by abs(linear contribution)+quadratic contribution, not the potentially cancelling net change. Float32 can suppress a mathematically nonzero tiny update entirely; those cases must not be interpreted as verified signs.

The production forward, gradient and update methods and the selected leaky activation branches are AST-identical to the original training commit 0ba13e8. See source_equivalence.json.

Diagnostic reproduction: python3 analysis/aq_identity_0917/diagnose.py

Report/figure reproduction: python3 analysis/aq_identity_0917/report.py; python3 analysis/aq_identity_0917/plot.py
