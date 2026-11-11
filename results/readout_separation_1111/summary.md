# Readout separation: complete paired contrasts

Primary metric U is the period-average fixed-training accuracy deficit. Positive baseline-minus-intervention U means the intervention improves average accuracy. Bootstrap paired seeds,10000 replicates,descriptive unadjusted95% intervals. No additive mediation shares.

Validated280 curves: ELU10seeds,ViT5seeds. Physicalscale/head direction, body-memory, initial200update surrogate policies, and Jacobian policies are distinct interventions.

Transmission contrasts use altered-Jacobian minus native, so positive U is harm. Parameterization uses physical0.1 minus TG, so positive U favors TG. Interaction signs follow the explicit coefficients in contrasts.csv.

U_acute/U_late are secondary temporal partitions;200pre is used for the acute partition,200post for late. They are not causal mediators.

## Primary U contrasts

### mlp

| kind | contrast | effect_pp_95ci | verdict | n_seed |
| --- | --- | --- | --- | --- |
| scale | S_keep | +19.461 [+18.629,+20.646] | positive | 10.0 |
| scale | S_clear | +29.973 [+29.215,+31.027] | positive | 10.0 |
| scale | P_keep | +19.479 [+18.354,+20.786] | positive | 10.0 |
| scale | P_clear | +29.361 [+28.253,+30.647] | positive | 10.0 |
| scale | O_keep | +19.799 [+18.991,+20.590] | positive | 10.0 |
| direction | S-P_c1_keep | -0.764 [-1.778,+0.195] | unresolved | 10.0 |
| direction | S-P_c0.1_keep | -0.746 [-1.376,-0.110] | negative | 10.0 |
| interaction | scale_by_P_keep | -0.018 [-1.033,+0.962] | unresolved | 10.0 |
| direction | S-P_c1_clear | -0.014 [-1.087,+0.985] | unresolved | 10.0 |
| direction | S-P_c0.1_clear | -0.627 [-1.239,-0.075] | negative | 10.0 |
| interaction | scale_by_P_clear | +0.612 [-0.352,+1.618] | unresolved | 10.0 |
| direction | S-O_c1_keep | -0.116 [-1.091,+1.064] | unresolved | 10.0 |
| direction | S-O_c0.1_keep | +0.222 [-0.373,+0.847] | unresolved | 10.0 |
| interaction | scale_by_O_keep | -0.338 [-1.211,+0.613] | unresolved | 10.0 |
| memory | S_c1 | -4.356 [-4.686,-4.016] | negative | 10.0 |
| memory | S_c0.1 | +6.156 [+5.755,+6.595] | positive | 10.0 |
| interaction | scale_by_memory_S | -10.512 [-10.671,-10.331] | negative | 10.0 |
| memory | P_c1 | -3.607 [-3.945,-3.259] | negative | 10.0 |
| memory | P_c0.1 | +6.275 [+5.920,+6.666] | positive | 10.0 |
| interaction | scale_by_memory_P | -9.882 [-10.256,-9.600] | negative | 10.0 |
| interaction | direction_by_memory_c1 | -0.750 [-1.124,-0.415] | negative | 10.0 |
| interaction | direction_by_memory_c0.1 | -0.120 [-0.481,+0.228] | unresolved | 10.0 |
| forward | S_gain1 | +10.120 [+9.455,+10.789] | positive | 10.0 |
| forward | S_gain0.1 | +0.920 [+0.725,+1.127] | positive | 10.0 |
| body_gain | S_cf1 | +2.372 [+2.076,+2.698] | positive | 10.0 |
| body_gain | S_cf0.1 | -6.828 [-7.230,-6.380] | negative | 10.0 |
| interaction | forward_by_gain_S | +9.200 [+8.610,+9.808] | positive | 10.0 |
| parameterization | physical01_minus_TG_S | -16.169 [-17.144,-15.340] | negative | 10.0 |
| forward | P_gain1 | +8.850 [+8.479,+9.231] | positive | 10.0 |
| forward | P_gain0.1 | +0.411 [+0.229,+0.597] | positive | 10.0 |
| body_gain | P_cf1 | +2.349 [+2.035,+2.599] | positive | 10.0 |
| body_gain | P_cf0.1 | -6.089 [-6.462,-5.725] | negative | 10.0 |
| interaction | forward_by_gain_P | +8.439 [+8.060,+8.838] | positive | 10.0 |
| parameterization | physical01_minus_TG_P | -16.718 [-17.902,-15.684] | negative | 10.0 |
| transmission | elu_unity_c1 | +11.302 [+10.827,+11.831] | positive | 10.0 |
| transmission | elu_unity_c0.1 | +2.148 [+1.946,+2.367] | positive | 10.0 |
| interaction | scale_by_elu_unity | +9.154 [+8.605,+9.772] | positive | 10.0 |

### vit

| kind | contrast | effect_pp_95ci | verdict | n_seed |
| --- | --- | --- | --- | --- |
| scale | S_keep | +6.797 [+2.149,+11.466] | positive | 5.0 |
| scale | S_clear | +15.344 [+12.075,+19.343] | positive | 5.0 |
| scale | P_keep | -1.424 [-3.753,+1.021] | unresolved | 5.0 |
| scale | P_clear | +0.884 [-0.156,+2.047] | unresolved | 5.0 |
| scale | O_keep | +7.283 [+0.461,+14.140] | positive | 5.0 |
| direction | S-P_c1_keep | +9.178 [+5.292,+14.442] | positive | 5.0 |
| direction | S-P_c0.1_keep | +0.957 [-1.885,+4.169] | unresolved | 5.0 |
| interaction | scale_by_P_keep | +8.222 [+2.707,+14.460] | positive | 5.0 |
| direction | S-P_c1_clear | +15.954 [+12.854,+20.151] | positive | 5.0 |
| direction | S-P_c0.1_clear | +1.494 [+0.162,+2.831] | positive | 5.0 |
| interaction | scale_by_P_clear | +14.460 [+10.063,+19.090] | positive | 5.0 |
| direction | S-O_c1_keep | -0.028 [-6.562,+3.750] | unresolved | 5.0 |
| direction | S-O_c0.1_keep | +0.457 [-4.894,+5.147] | unresolved | 5.0 |
| interaction | scale_by_O_keep | -0.486 [-11.311,+7.155] | unresolved | 5.0 |
| memory | S_c1 | -5.990 [-6.601,-5.222] | negative | 5.0 |
| memory | S_c0.1 | +2.557 [+0.367,+4.768] | positive | 5.0 |
| interaction | scale_by_memory_S | -8.547 [-10.784,-6.569] | negative | 5.0 |
| memory | P_c1 | +0.786 [-2.334,+4.297] | unresolved | 5.0 |
| memory | P_c0.1 | +3.094 [+1.675,+4.865] | positive | 5.0 |
| interaction | scale_by_memory_P | -2.308 [-4.238,+0.372] | unresolved | 5.0 |
| interaction | direction_by_memory_c1 | -6.775 [-9.728,-3.935] | negative | 5.0 |
| interaction | direction_by_memory_c0.1 | -0.537 [-3.243,+2.451] | unresolved | 5.0 |
| forward | S_gain1 | +1.535 [-5.558,+7.315] | unresolved | 5.0 |
| forward | S_gain0.1 | -9.275 [-12.647,-5.032] | negative | 5.0 |
| body_gain | S_cf1 | +7.977 [+6.274,+9.491] | positive | 5.0 |
| body_gain | S_cf0.1 | -2.834 [-6.264,+1.574] | unresolved | 5.0 |
| interaction | forward_by_gain_S | +10.810 [+6.113,+14.282] | positive | 5.0 |
| parameterization | physical01_minus_TG_S | -8.096 [-10.983,-6.034] | negative | 5.0 |
| forward | P_gain1 | -21.001 [-28.412,-12.109] | negative | 5.0 |
| forward | P_gain0.1 | -22.464 [-33.156,-13.488] | negative | 5.0 |
| body_gain | P_cf1 | -2.440 [-4.714,-0.217] | negative | 5.0 |
| body_gain | P_cf0.1 | -3.903 [-10.642,+2.837] | unresolved | 5.0 |
| interaction | forward_by_gain_P | +1.462 [-3.991,+8.522] | unresolved | 5.0 |
| parameterization | physical01_minus_TG_P | -23.479 [-32.413,-16.103] | negative | 5.0 |
| transmission | ln_remove_radial_c1 | +10.853 [+8.826,+12.880] | positive | 5.0 |
| transmission | ln_remove_radial_c0.1 | +4.355 [-2.035,+12.355] | unresolved | 5.0 |
| interaction | scale_by_ln_remove_radial | +6.498 [-2.838,+12.698] | unresolved | 5.0 |
| transmission | ln_remove_radial_mean_c1 | +10.680 [+8.668,+12.737] | positive | 5.0 |
| transmission | ln_remove_radial_mean_c0.1 | +4.209 [-2.009,+11.519] | unresolved | 5.0 |
| interaction | scale_by_ln_remove_radial_mean | +6.472 [-2.125,+12.523] | unresolved | 5.0 |

## Arm endpoints and censoring

- mlp jacobian_S01_elu_unity: U=0.407753; end accuracy=0.939417; T90 reached=10/10; sustained=10/10.

- mlp jacobian_S1_elu_unity: U=0.693900; end accuracy=0.504083; T90 reached=0/10; sustained=0/10.

- mlp physical_O01_body_keep: U=0.384050; end accuracy=0.950500; T90 reached=10/10; sustained=10/10.

- mlp physical_O1_body_keep: U=0.582042; end accuracy=0.702000; T90 reached=0/10; sustained=0/10.

- mlp physical_P01_body_clear: U=0.330981; end accuracy=0.969833; T90 reached=10/10; sustained=10/10.

- mlp physical_P01_body_keep: U=0.393733; end accuracy=0.945667; T90 reached=10/10; sustained=10/10.

- mlp physical_P1_body_clear: U=0.624587; end accuracy=0.662000; T90 reached=0/10; sustained=0/10.

- mlp physical_P1_body_keep: U=0.588521; end accuracy=0.699917; T90 reached=0/10; sustained=0/10.

- mlp physical_S01_body_clear: U=0.324715; end accuracy=0.973083; T90 reached=10/10; sustained=10/10.

- mlp physical_S01_body_keep: U=0.386272; end accuracy=0.951167; T90 reached=10/10; sustained=10/10.

- mlp physical_S1_body_clear: U=0.624445; end accuracy=0.672750; T90 reached=0/10; sustained=0/10.

- mlp physical_S1_body_keep: U=0.580881; end accuracy=0.716833; T90 reached=0/10; sustained=0/10.

- mlp surrogate_P_G: U=0.565027; end accuracy=0.735167; T90 reached=0/10; sustained=0/10.

- mlp surrogate_P_T: U=0.500021; end accuracy=0.777583; T90 reached=0/10; sustained=0/10.

- mlp surrogate_P_TG: U=0.560914; end accuracy=0.733750; T90 reached=0/10; sustained=0/10.

- mlp surrogate_S_G: U=0.557163; end accuracy=0.753333; T90 reached=0/10; sustained=0/10.

- mlp surrogate_S_T: U=0.479679; end accuracy=0.794833; T90 reached=0/10; sustained=0/10.

- mlp surrogate_S_TG: U=0.547962; end accuracy=0.757917; T90 reached=0/10; sustained=0/10.

- vit jacobian_S01_ln_remove_radial: U=0.322604; end accuracy=0.996160; T90 reached=5/5; sustained=5/5.

- vit jacobian_S01_ln_remove_radial_mean: U=0.321144; end accuracy=0.995760; T90 reached=5/5; sustained=5/5.

- vit jacobian_S1_ln_remove_radial: U=0.455556; end accuracy=0.932080; T90 reached=5/5; sustained=4/5.

- vit jacobian_S1_ln_remove_radial_mean: U=0.453830; end accuracy=0.937520; T90 reached=5/5; sustained=4/5.

- vit physical_O01_body_keep: U=0.274482; end accuracy=0.999440; T90 reached=5/5; sustained=5/5.

- vit physical_O1_body_keep: U=0.347312; end accuracy=0.954560; T90 reached=4/5; sustained=4/5.

- vit physical_P01_body_clear: U=0.238550; end accuracy=0.996240; T90 reached=5/5; sustained=5/5.

- vit physical_P01_body_keep: U=0.269488; end accuracy=0.998720; T90 reached=5/5; sustained=5/5.

- vit physical_P1_body_clear: U=0.247387; end accuracy=0.978880; T90 reached=5/5; sustained=4/5.

- vit physical_P1_body_keep: U=0.255243; end accuracy=1.000000; T90 reached=5/5; sustained=5/5.

- vit physical_S01_body_clear: U=0.253486; end accuracy=0.974880; T90 reached=5/5; sustained=5/5.

- vit physical_S01_body_keep: U=0.279057; end accuracy=0.981920; T90 reached=5/5; sustained=5/5.

- vit physical_S1_body_clear: U=0.406925; end accuracy=0.940960; T90 reached=5/5; sustained=4/5.

- vit physical_S1_body_keep: U=0.347027; end accuracy=0.996560; T90 reached=5/5; sustained=5/5.

- vit surrogate_P_G: U=0.279646; end accuracy=1.000000; T90 reached=5/5; sustained=5/5.

- vit surrogate_P_T: U=0.465256; end accuracy=0.981920; T90 reached=5/5; sustained=5/5.

- vit surrogate_P_TG: U=0.504282; end accuracy=0.912080; T90 reached=4/5; sustained=4/5.

- vit surrogate_S_G: U=0.267262; end accuracy=1.000000; T90 reached=5/5; sustained=5/5.

- vit surrogate_S_T: U=0.331677; end accuracy=0.963600; T90 reached=5/5; sustained=4/5.

- vit surrogate_S_TG: U=0.360015; end accuracy=0.976720; T90 reached=5/5; sustained=4/5.
