# CondA dimension and mean-channel diagnostic — post hoc

This uses the completed CondA snapshots only. It adds no training and changes no preregistered P1–P6 criterion or verdict. All values below are medians of five seed-level values; `transition.csv`, `seed_windows.csv`, and `paired_seed.csv` retain every value and source snapshot path.

## Late window (tasks 81–100)

D and R are square roots of centered Q and V. The raw columns use Σ=I. B uses -2 sum X/sum Q; c and rho also use sums, never averages of per-task ratios.

| arm | m | r | conditional_trace | mean_D_eff | mean_R_eff | c_eff_summed | rho_eff_summed | B_eff_summed | mean_D_raw | mean_R_raw | raw_to_effective_R | mean_D_mean_parameter | mean_D_mean_environment | mean_parameter_sq_over_Q_eff | mean_start_mse | mean_end_mse | mean_mse_gain | mean_mse_gain_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 | 20 | 5 | 1.25 | 0.935 | 11.5 | -0.00413 | 0.085 | 0.0972 | 2.3 | 27 | 2.34 | 4.33 | 3.61 | 19.7 | 3.15 | 0.00813 | 3.13 | 0.996 |
| k1_r10 | 20 | 10 | 2.5 | 1.46 | 14 | -0.0322 | 0.105 | 0.594 | 3.22 | 31.8 | 2.27 | 5.17 | 4.57 | 10.9 | 1.83 | 0.269 | 1.57 | 0.823 |
| k1_sna06 | 20 | 5 | 1.25 | 1.61 | 27.6 | 0.0239 | 0.064 | -0.641 | 4.13 | 58.5 | 2.11 | 9.62 | 4.66 | 34.3 | 3.04 | 0.0173 | 3.02 | 0.992 |
| m40_r10_k1 | 40 | 10 | 2.5 | 1.09 | 7.72 | -0.0664 | 0.147 | 0.813 | 2.39 | 20 | 2.62 | 3.95 | 2.27 | 12.2 | 0.3 | 0.0379 | 0.274 | 0.852 |
| m40_r10_k1_sna06 | 40 | 10 | 2.5 | 1.2 | 3.74 | -0.157 | 0.315 | 1.06 | 2.56 | 13.5 | 3.64 | 3.27 | 1.93 | 8.01 | 0.203 | 0.0789 | 0.124 | 0.597 |
| m40_r10_k2 | 40 | 10 | 2.5 | 1.38 | 9.18 | -0.0633 | 0.146 | 0.669 | 3.01 | 22.2 | 2.42 | 5.12 | 2.98 | 13.8 | 0.376 | 0.0688 | 0.323 | 0.771 |
| m40_r10_k2_sna06 | 40 | 10 | 2.5 | 1.4 | 4.76 | -0.197 | 0.312 | 0.97 | 3.05 | 14.2 | 2.99 | 4.25 | 2.47 | 8.53 | 0.303 | 0.0905 | 0.217 | 0.658 |
| m40_r5_k1 | 40 | 5 | 1.25 | 0.926 | 9.34 | -0.0243 | 0.109 | 0.537 | 2.23 | 22.9 | 2.45 | 4.94 | 2.25 | 28.8 | 0.403 | 0.00221 | 0.401 | 0.993 |

## Seed-indexed comparisons, late window

Ratios are calculated within each seed and then medianed. m20–m40 rows match only seed indices: changing input dimension also changes the teacher and initialization. They are not pure dimension-causal estimates.

| comparison | reference_m | other_m | reference_rank | other_rank | ratio_mean_D_eff | ratio_mean_R_eff | ratio_mean_D_raw | ratio_mean_R_raw | ratio_raw_to_effective_R | ratio_mean_D_mean_parameter | ratio_mean_D_mean_environment | difference_B_eff_summed | difference_mean_mse_gain_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m20_r10_to_m40_r10_k1 | 20 | 40 | 10 | 10 | 0.762 | 0.565 | 0.753 | 0.652 | 1.15 | 0.73 | 0.496 | 0.258 | 0.0401 |
| m20_r10_to_m40_r10_k2 | 20 | 40 | 10 | 10 | 0.882 | 0.624 | 0.853 | 0.674 | 1.08 | 0.884 | 0.664 | -0.0961 | -0.0525 |
| m20_r5_leaky_to_SNA06 | 20 | 20 | 5 | 5 | 1.71 | 2.39 | 1.8 | 2.16 | 0.903 | 2.22 | 1.27 | -1.05 | -0.00368 |
| m20_r5_to_m40_r5 | 20 | 40 | 5 | 5 | 0.979 | 0.823 | 0.993 | 0.856 | 1.04 | 1.2 | 0.631 | 0.189 | -0.00272 |
| m40_r10_k1_leaky_to_SNA06 | 40 | 40 | 10 | 10 | 1.09 | 0.454 | 1.08 | 0.648 | 1.42 | 0.845 | 0.833 | -0.0142 | -0.266 |
| m40_r10_k1_to_k2 | 40 | 40 | 10 | 10 | 1.27 | 1.17 | 1.29 | 1.1 | 0.939 | 1.28 | 1.37 | -0.56 | -0.0637 |
| m40_r10_k2_leaky_to_SNA06 | 40 | 40 | 10 | 10 | 1.08 | 0.495 | 1.05 | 0.633 | 1.25 | 0.861 | 0.813 | 0.243 | -0.131 |

## Early and late paired trajectories

The two 20-transition windows show whether a ratio is persistent over this observed horizon; they do not establish an asymptotic trend.

| comparison | window | ratio_mean_D_eff | ratio_mean_R_eff | ratio_mean_D_raw | ratio_mean_R_raw | difference_mean_mse_gain_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| m20_r10_to_m40_r10_k1 | early | 0.8 | 0.695 | 0.802 | 0.88 | 0.0809 |
| m20_r10_to_m40_r10_k1 | late | 0.762 | 0.565 | 0.753 | 0.652 | 0.0401 |
| m20_r10_to_m40_r10_k2 | early | 0.88 | 0.656 | 0.883 | 0.862 | 0.0802 |
| m20_r10_to_m40_r10_k2 | late | 0.882 | 0.624 | 0.853 | 0.674 | -0.0525 |
| m20_r5_leaky_to_SNA06 | early | 1.61 | 1.53 | 1.74 | 1.33 | -0.00975 |
| m20_r5_leaky_to_SNA06 | late | 1.71 | 2.39 | 1.8 | 2.16 | -0.00368 |
| m20_r5_to_m40_r5 | early | 0.936 | 0.76 | 0.926 | 0.919 | -0.00302 |
| m20_r5_to_m40_r5 | late | 0.979 | 0.823 | 0.993 | 0.856 | -0.00272 |
| m40_r10_k1_leaky_to_SNA06 | early | 1.01 | 0.693 | 1 | 0.892 | -0.179 |
| m40_r10_k1_leaky_to_SNA06 | late | 1.09 | 0.454 | 1.08 | 0.648 | -0.266 |
| m40_r10_k1_to_k2 | early | 1.13 | 0.997 | 1.15 | 0.994 | 0.044 |
| m40_r10_k1_to_k2 | late | 1.27 | 1.17 | 1.29 | 1.1 | -0.0637 |
| m40_r10_k2_leaky_to_SNA06 | early | 1.01 | 0.762 | 1.01 | 0.903 | -0.139 |
| m40_r10_k2_leaky_to_SNA06 | late | 1.08 | 0.495 | 1.05 | 0.633 | -0.131 |

## Mean channel and interpretation

For each transition, the parameter-induced mean change is ||D μ_prev + Δb||². The environment change ||W_next(μ_next−μ_prev)||² and their cross term are separate columns; their sum is the full mean-preactivation shift. This prevents treating persistent-bit directions invisible to centered Σ as harmless weight.
A low sum Q/mean V or STOPPED label is a relative-update diagnostic, not proof that the learner cannot improve. Task-start and task-end support MSE and their signed gain are reported beside displacement. MSE differences across m20/m40 may reflect different target functions and initializations.
The raw/centered algebraic identities and ledger cross-check establish bookkeeping only; they are not independent evidence for a fixed point, causal independence, or asymptotic stationarity.

## Audit

Rows: 4000 transitions; maximum |centered identity residual| = 2.77e-13, maximum |raw identity residual| = 1.09e-12, maximum |mean decomposition residual| = 1.71e-13.
Maximum relative difference from `partial_0053/transitions.csv` = 6.69e-15.
Source: `/home/issan/Projects/obsidian-research-data/effdisp_validation_0924/raw/conda`; previous and next snapshot paths are recorded per transition.
