# ELU depth x width factorial results

CPU Permuted MNIST; ELU1/leaky .1; 3 paired seeds; common task20 branches.
Mean depth is the actual per-unit training-input mean, fixed through bias.
Centered row norms fixed at 5/10; row means frozen at task20. This changes the
causal regime. Width effects at fixed mean may still act through saturation.

## Cell means

| Activation | Cell | Late accuracy % | L points | Late train CE | Late test CE | Gate<.05 |
|---|---|---:|---:|---:|---:|---:|
| ELU1 | n5_d1 | 92.495 | -0.175 | 0.1706 | 0.2402 | 0.151 |
| ELU1 | n10_d1 | 91.471 | -0.317 | 0.2001 | 0.2728 | 0.254 |
| ELU1 | n5_d4 | 90.693 | 0.113 | 0.2492 | 0.3105 | 0.696 |
| ELU1 | n10_d4 | 89.554 | 0.151 | 0.2748 | 0.3393 | 0.634 |
| ELU1 | ref | 90.218 | 1.587 | 0.2652 | 0.3121 | 0.579 |
| LR | n5_d1 | 92.220 | -0.129 | 0.1759 | 0.2513 | 0.000 |
| LR | n10_d1 | 90.719 | -0.268 | 0.2270 | 0.3033 | 0.000 |
| LR | n5_d4 | 89.919 | 0.269 | 0.2919 | 0.3310 | 0.000 |
| LR | n10_d4 | 89.500 | -0.093 | 0.2837 | 0.3503 | 0.000 |
| LR | ref | 89.797 | 1.646 | 0.2814 | 0.3308 | 0.000 |

## Registered accuracy contrasts

W_harm=accuracy(norm5)-accuracy(norm10); H=W_harm(ELU)-W_harm(leaky).
Positive means high width hurts more (H positive means ELU is hurt more).
Intervals are paired-seed t95% with df2; +/-0.5pt practical-equivalence margin.

| Contrast | Mean pt | 95% CI | Verdict |
|---|---:|---|---|
| W_harm_ELU1_d1 | 1.024 | [0.684, 1.364] | descriptive |
| W_harm_ELU1_d4 | 1.139 | [0.715, 1.564] | descriptive |
| W_depth_interaction_ELU1 | 0.116 | [-0.392, 0.624] | descriptive |
| Depth_harm_ELU1_n5 | 1.802 | [1.245, 2.359] | descriptive |
| Depth_harm_ELU1_n10 | 1.917 | [1.459, 2.376] | descriptive |
| W_harm_LR_d1 | 1.501 | [1.293, 1.710] | descriptive |
| W_harm_LR_d4 | 0.419 | [0.116, 0.722] | descriptive |
| W_depth_interaction_LR | -1.082 | [-1.399, -0.766] | descriptive |
| Depth_harm_LR_n5 | 2.301 | [2.106, 2.497] | descriptive |
| Depth_harm_LR_n10 | 1.219 | [0.970, 1.468] | descriptive |
| H_ELU_minus_LR_d1 | -0.478 | [-0.993, 0.038] | INCONCLUSIVE |
| H_ELU_minus_LR_d4 | 0.720 | [0.058, 1.382] | INCONCLUSIVE |
| J_three_way | 1.198 | [0.453, 1.943] | INCONCLUSIVE |

## Validation and limits

All branches restored identical per-activation/seed prefix parameters, Adam and RNG.
Frozen-host one-task exact equality, derivative, projection mutations and measurement
neutrality passed. All task datasets and probes are common across continuation arms.
See each raw provenance for per-update invariant maxima and committed reference checks.
Only three independent seeds; narrow claims require CI within the registered bounds.
Uniform per-unit means and frozen rowmeans differ from natural ELU populations and
old global dclamp. This factorial does not separately identify saturation mediation.
Training-probe CE is descriptive; held-out test CE/accuracy use 2048 fixed test examples.
All seed values, absolute early/late levels, adaptation curves and initial projection
shock are saved; no absence claim is inferred from nonsignificance.
