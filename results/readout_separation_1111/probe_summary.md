# Frozen-state separation

All backward-map and transmission comparisons reuse the whole forward probability vector for each fixed forward bank. Actual next labels and uniform-label expectation are distinct channels. Parameter gradients come from128fixed examples; MLP position shifts concern the full1200-example recipient input mean.

Intervals are paired-seed descriptive95% bootstrap intervals. Norm differences do not form an additive decomposition of delay.

## Native physical banks, actual new labels

| model | forward_bank | metric | value_95ci |
| --- | --- | --- | --- |
| mlp | O01 | g_feature_unreduced_mean_example_norm | 0.521037 [0.517552,0.524388] |
| mlp | O01 | g_trunk_norm | 0.109569 [0.103464,0.116628] |
| mlp | O01 | adam_keep_trunk_norm | 0.0427349 [0.0392059,0.0476354] |
| mlp | O01 | gd_push_mean | 0.00759423 [0.00569363,0.00976395] |
| mlp | O01 | adam_keep_mean_preact_shift | 0.000697209 [0.000356451,0.00104197] |
| mlp | O1 | g_feature_unreduced_mean_example_norm | 5.98821 [5.94164,6.04649] |
| mlp | O1 | g_trunk_norm | 1.53247 [1.46395,1.60407] |
| mlp | O1 | adam_keep_trunk_norm | 0.0951345 [0.0885228,0.101566] |
| mlp | O1 | gd_push_mean | 0.320548 [0.299802,0.341942] |
| mlp | O1 | adam_keep_mean_preact_shift | -0.00655759 [-0.00743897,-0.00562953] |
| mlp | P01 | g_feature_unreduced_mean_example_norm | 0.521393 [0.51767,0.525006] |
| mlp | P01 | g_trunk_norm | 0.109153 [0.104281,0.11403] |
| mlp | P01 | adam_keep_trunk_norm | 0.0427878 [0.0394268,0.0474003] |
| mlp | P01 | gd_push_mean | 0.00866768 [0.00700366,0.0102053] |
| mlp | P01 | adam_keep_mean_preact_shift | 0.000653893 [0.000293118,0.00100759] |
| mlp | P1 | g_feature_unreduced_mean_example_norm | 6.13927 [6.00385,6.2762] |
| mlp | P1 | g_trunk_norm | 1.59196 [1.53465,1.65146] |
| mlp | P1 | adam_keep_trunk_norm | 0.0959358 [0.0897025,0.10303] |
| mlp | P1 | gd_push_mean | 0.388773 [0.356979,0.418684] |
| mlp | P1 | adam_keep_mean_preact_shift | -0.00832365 [-0.00941828,-0.00716834] |
| mlp | S01 | g_feature_unreduced_mean_example_norm | 0.519932 [0.515942,0.523632] |
| mlp | S01 | g_trunk_norm | 0.107283 [0.100316,0.113801] |
| mlp | S01 | adam_keep_trunk_norm | 0.0430939 [0.0395137,0.0480743] |
| mlp | S01 | gd_push_mean | 0.00681447 [0.00483262,0.008864] |
| mlp | S01 | adam_keep_mean_preact_shift | 0.000700433 [0.000361192,0.00102948] |
| mlp | S1 | g_feature_unreduced_mean_example_norm | 6.13803 [6.06274,6.21208] |
| mlp | S1 | g_trunk_norm | 1.46474 [1.37895,1.55258] |
| mlp | S1 | adam_keep_trunk_norm | 0.0975687 [0.0909583,0.105049] |
| mlp | S1 | gd_push_mean | 0.295854 [0.272158,0.31878] |
| mlp | S1 | adam_keep_mean_preact_shift | -0.00652003 [-0.00725878,-0.00579927] |
| vit | O01 | g_feature_unreduced_mean_example_norm | 0.151166 [0.13705,0.164994] |
| vit | O01 | g_trunk_norm | 8.48369 [6.50362,10.3723] |
| vit | O01 | adam_keep_trunk_norm | 0.189067 [0.15723,0.220903] |
| vit | O01 | gd_push_mean | not estimable |
| vit | O01 | adam_keep_mean_preact_shift | not estimable |
| vit | O1 | g_feature_unreduced_mean_example_norm | 1.78244 [1.65254,1.92239] |
| vit | O1 | g_trunk_norm | 71.4755 [56.0485,90.8675] |
| vit | O1 | adam_keep_trunk_norm | 0.53081 [0.510963,0.549937] |
| vit | O1 | gd_push_mean | not estimable |
| vit | O1 | adam_keep_mean_preact_shift | not estimable |
| vit | P01 | g_feature_unreduced_mean_example_norm | 0.155763 [0.14324,0.168285] |
| vit | P01 | g_trunk_norm | 1.82309 [1.22685,2.41933] |
| vit | P01 | adam_keep_trunk_norm | 0.132662 [0.106974,0.151148] |
| vit | P01 | gd_push_mean | not estimable |
| vit | P01 | adam_keep_mean_preact_shift | not estimable |
| vit | P1 | g_feature_unreduced_mean_example_norm | 1.56429 [1.45924,1.66934] |
| vit | P1 | g_trunk_norm | 14.6387 [10.0039,19.2734] |
| vit | P1 | adam_keep_trunk_norm | 0.341131 [0.301337,0.366877] |
| vit | P1 | gd_push_mean | not estimable |
| vit | P1 | adam_keep_mean_preact_shift | not estimable |
| vit | S01 | g_feature_unreduced_mean_example_norm | 0.147124 [0.139207,0.154768] |
| vit | S01 | g_trunk_norm | 6.24179 [5.12952,7.36586] |
| vit | S01 | adam_keep_trunk_norm | 0.166974 [0.132295,0.195754] |
| vit | S01 | gd_push_mean | not estimable |
| vit | S01 | adam_keep_mean_preact_shift | not estimable |
| vit | S1 | g_feature_unreduced_mean_example_norm | 1.73494 [1.63893,1.82734] |
| vit | S1 | g_trunk_norm | 59.8258 [48.3431,72.6348] |
| vit | S1 | adam_keep_trunk_norm | 0.481801 [0.448712,0.514569] |
| vit | S1 | gd_push_mean | not estimable |
| vit | S1 | adam_keep_mean_preact_shift | not estimable |

Full pointwise-matched results: probe_seed_records.csv/probe_estimates.csv/probe_contrasts.csv. Geometry and effective LN-affine scales: probe_fullset_seed_metrics.csv. Gauge and virtual Adam checks: probe_validation.json and raw probe JSON.
