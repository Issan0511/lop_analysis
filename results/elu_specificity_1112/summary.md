# ELU specificity: complete registered analysis

Validated 1260 full curves: Study A 1120 (10 paired seeds), Study B 140 (5 paired seeds).

U is trapezoidal period-average fixed-training accuracy deficit. Positive B_S/B_P means the physical scale reduction improves average training accuracy; positive Z_S/Z_P means zero-head/body-memory-keep improves it. These are distinct interventions. Positive body_utility=U(Z_clear)-U(F0) means the learned body performs worse than the fresh-body reference under zero head and fresh Adam; it does not measure irreducible capacity.

Each bootstrap draw resamples a whole seed cluster; the same 10000 draws are reused across arms, ages, LN levels and current activations within a study. Percentile 95% intervals are descriptive and unadjusted. Signed support requires the interval to lie entirely above/below zero; zero crossing is unresolved, not equivalence. The 1 pp reference is not an equivalence test.

Study A identifies total activation/final-affine-LN training-regime effects including learned state formation. Study B identifies a persistent current-activation switch on ReLU-trained ViT histories, including compatibility with retained optimizer memory. Its all-five-LN history treatment differs from MLP final LN. No architecture-only effect, additive causal share, or exclusion of ELU saturation follows from these comparisons.

U_acute covers [0,200], U_late covers [200,end]; Head edits and Adam resets are applied once at the fork; all parameters then learn jointly. Only the chosen activation/LN configuration persists, with no step200 release. U uses the entire 4000/1500-update horizon, and all endpoints share that horizon. CE area is the unnormalized integral; ce_auc_mean divides it by the horizon. T90 is the first saved observation ≥90%; sustained T90 requires all later saved observations to remain ≥90% and may have zero follow-up duration. Missing T90 is censored, never replaced by the budget; a censored arm mean/paired contrast is not estimated from a selected subset.

## Primary U interactions (percentage points)

### Study A

| kind | contrast | activation | ln | t_end | estimate_pp_95ci | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| activation | B_P:ELU-R | ELU-R | off | 10 | +17.946 [+16.739, +19.421] | positive |
| activation | B_P:ELU-LR | ELU-LR | off | 10 | +26.231 [+24.857, +28.023] | positive |
| activation | B_P:ELU-SILU | ELU-SILU | off | 10 | +18.304 [+16.836, +19.984] | positive |
| activation | B_P:ELU-R | ELU-R | on | 10 | +9.967 [+8.963, +10.982] | positive |
| activation | B_P:ELU-LR | ELU-LR | on | 10 | +2.826 [+1.370, +4.201] | positive |
| activation | B_P:ELU-SILU | ELU-SILU | on | 10 | -0.537 [-1.888, +0.792] | unresolved |
| LN | B_P:LNoff-LNon | ELU | off-on | 10 | +7.684 [+6.014, +9.535] | positive |
| LN | B_P:LNoff-LNon | R | off-on | 10 | -0.295 [-1.415, +0.853] | unresolved |
| LN | B_P:LNoff-LNon | LR | off-on | 10 | -15.722 [-16.521, -14.877] | negative |
| LN | B_P:LNoff-LNon | SILU | off-on | 10 | -11.158 [-12.457, -9.730] | negative |
| activation_by_LN | B_P:(ELU-R)x(off-on) | ELU-R | off-on | 10 | +7.979 [+6.025, +10.026] | positive |
| activation_by_LN | B_P:(ELU-LR)x(off-on) | ELU-LR | off-on | 10 | +23.406 [+21.510, +25.375] | positive |
| activation_by_LN | B_P:(ELU-SILU)x(off-on) | ELU-SILU | off-on | 10 | +18.841 [+16.746, +21.095] | positive |

### Study B

| kind | contrast | activation | ln | t_end | estimate_pp_95ci | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| current_activation | B_P:ELU-R | ELU-R | off | 30 | +4.294 [-13.927, +16.216] | unresolved |
| current_activation | B_P:ELU-R | ELU-R | on | 30 | -23.175 [-26.316, -19.921] | negative |
| current_activation_by_LN_history | B_P:(ELU-R)x(off-on) | ELU-R | off-on | 30 | +27.469 [+11.975, +38.657] | positive |

## Absolute zero-head/fresh-body performance

| study | activation | ln | t_end | arm | U_pp | endpoint_accuracy_pct | T90_first_reached | T90_sustained_reached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | ELU | off | 1 | F0 | 32.610 | 96.908 | 10/10 | 10/10 |
| A | ELU | off | 1 | Z_clear | 25.806 | 98.350 | 10/10 | 10/10 |
| A | ELU | off | 1 | Z_keep | 36.991 | 96.408 | 10/10 | 10/10 |
| A | ELU | off | 10 | F0 | 32.684 | 96.542 | 10/10 | 10/10 |
| A | ELU | off | 10 | Z_clear | 24.366 | 98.792 | 10/10 | 10/10 |
| A | ELU | off | 10 | Z_keep | 36.986 | 96.392 | 10/10 | 10/10 |
| A | ELU | on | 1 | F0 | 24.800 | 98.750 | 10/10 | 10/10 |
| A | ELU | on | 1 | Z_clear | 15.108 | 99.800 | 10/10 | 10/10 |
| A | ELU | on | 1 | Z_keep | 19.245 | 99.725 | 10/10 | 10/10 |
| A | ELU | on | 10 | F0 | 24.914 | 99.017 | 10/10 | 10/10 |
| A | ELU | on | 10 | Z_clear | 13.751 | 100.000 | 10/10 | 10/10 |
| A | ELU | on | 10 | Z_keep | 17.213 | 99.983 | 10/10 | 10/10 |
| A | LR | off | 1 | F0 | 24.916 | 98.875 | 10/10 | 10/10 |
| A | LR | off | 1 | Z_clear | 21.984 | 98.983 | 10/10 | 10/10 |
| A | LR | off | 1 | Z_keep | 30.558 | 97.983 | 10/10 | 10/10 |
| A | LR | off | 10 | F0 | 25.167 | 98.808 | 10/10 | 10/10 |
| A | LR | off | 10 | Z_clear | 24.543 | 98.858 | 10/10 | 10/10 |
| A | LR | off | 10 | Z_keep | 59.739 | 76.150 | 0/10 | 0/10 |
| A | LR | on | 1 | F0 | 14.945 | 99.992 | 10/10 | 10/10 |
| A | LR | on | 1 | Z_clear | 11.028 | 100.000 | 10/10 | 10/10 |
| A | LR | on | 1 | Z_keep | 11.386 | 100.000 | 10/10 | 10/10 |
| A | LR | on | 10 | F0 | 14.995 | 99.992 | 10/10 | 10/10 |
| A | LR | on | 10 | Z_clear | 16.860 | 99.992 | 10/10 | 10/10 |
| A | LR | on | 10 | Z_keep | 27.834 | 99.850 | 10/10 | 10/10 |
| A | R | off | 1 | F0 | 27.195 | 98.408 | 10/10 | 10/10 |
| A | R | off | 1 | Z_clear | 23.022 | 98.867 | 10/10 | 10/10 |
| A | R | off | 1 | Z_keep | 32.616 | 97.575 | 10/10 | 10/10 |
| A | R | off | 10 | F0 | 27.171 | 98.367 | 10/10 | 10/10 |
| A | R | off | 10 | Z_clear | 83.666 | 19.925 | 0/10 | 0/10 |
| A | R | off | 10 | Z_keep | 85.429 | 16.958 | 0/10 | 0/10 |
| A | R | on | 1 | F0 | 15.028 | 100.000 | 10/10 | 10/10 |
| A | R | on | 1 | Z_clear | 12.651 | 98.667 | 10/10 | 10/10 |
| A | R | on | 1 | Z_keep | 13.470 | 99.900 | 10/10 | 10/10 |
| A | R | on | 10 | F0 | 15.074 | 99.658 | 10/10 | 10/10 |
| A | R | on | 10 | Z_clear | 73.635 | 32.292 | 0/10 | 0/10 |
| A | R | on | 10 | Z_keep | 76.611 | 33.425 | 0/10 | 0/10 |
| A | SILU | off | 1 | F0 | 28.561 | 97.933 | 10/10 | 10/10 |
| A | SILU | off | 1 | Z_clear | 21.646 | 98.900 | 10/10 | 10/10 |
| A | SILU | off | 1 | Z_keep | 34.304 | 97.192 | 10/10 | 10/10 |
| A | SILU | off | 10 | F0 | 28.518 | 97.717 | 10/10 | 10/10 |
| A | SILU | off | 10 | Z_clear | 38.051 | 88.267 | 1/10 | 1/10 |
| A | SILU | off | 10 | Z_keep | 50.268 | 79.808 | 0/10 | 0/10 |
| A | SILU | on | 1 | F0 | 16.034 | 99.950 | 10/10 | 10/10 |
| A | SILU | on | 1 | Z_clear | 7.403 | 100.000 | 10/10 | 10/10 |
| A | SILU | on | 1 | Z_keep | 8.248 | 100.000 | 10/10 | 10/10 |
| A | SILU | on | 10 | F0 | 16.252 | 99.233 | 10/10 | 10/10 |
| A | SILU | on | 10 | Z_clear | 12.196 | 100.000 | 10/10 | 10/10 |
| A | SILU | on | 10 | Z_keep | 15.251 | 100.000 | 10/10 | 10/10 |
| B | ELU | off | 30 | F0 | 58.149 | 85.704 | 1/5 | 1/5 |
| B | ELU | off | 30 | Z_clear | 83.848 | 17.752 | 0/5 | 0/5 |
| B | ELU | off | 30 | Z_keep | 83.012 | 23.056 | 0/5 | 0/5 |
| B | ELU | on | 30 | F0 | 38.670 | 99.712 | 5/5 | 5/5 |
| B | ELU | on | 30 | Z_clear | 13.428 | 100.000 | 5/5 | 5/5 |
| B | ELU | on | 30 | Z_keep | 15.569 | 100.000 | 5/5 | 5/5 |
| B | R | off | 30 | F0 | 54.969 | 91.456 | 3/5 | 3/5 |
| B | R | off | 30 | Z_clear | 21.575 | 99.792 | 5/5 | 5/5 |
| B | R | off | 30 | Z_keep | 44.005 | 84.088 | 4/5 | 3/5 |
| B | R | on | 30 | F0 | 22.641 | 100.000 | 5/5 | 5/5 |
| B | R | on | 30 | Z_clear | 8.296 | 100.000 | 5/5 | 5/5 |
| B | R | on | 30 | Z_keep | 9.339 | 100.000 | 5/5 | 5/5 |

All registered estimates and explicit linear coefficients are in contrasts.csv; seed_contrasts.csv exposes each paired difference. Arm means, censor counts and intervals are in arm_means.csv. seed_metrics.csv retains curve hashes and numeric diagnostics. validation.json records completeness and provenance consistency; it does not replace the independent source/archive byte-hash audit.
