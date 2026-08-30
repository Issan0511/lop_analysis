# gate_dose_0830 summary

## Verdict

- Main: **GATE_LOAD_BEARING**
- Numeric divergence: none
- Claim strength: observed through 5M steps only; 0/10 one-sided 95% upper bound is 0.2589.
- Pairing removes init/teacher/input-realization variance; activation trajectories diverge after step 1.

## Endpoints

| arm | act | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | submerged frac 5M |
|---|---|---:|---:|---:|---:|---:|
| R_off | relu | 9/10 | 10/10 | -0.674832 | -0.09297 | 0.9915 |
| R_933 | relu | 3/10 | 10/10 | -1.60124 | -0.215445 | 0.9815 |
| R_1216 | relu | 10/10 | 10/10 | -0.649239 | -0.274662 | 0.9855 |
| E_off | elu | 0/10 | 0/10 | -2.16413 | -1.97038 | 0.4545 |
| E_933 | elu | 0/10 | 0/10 | -2.50305 | -2.75984 | 0.3835 |
| E_1216 | elu | 0/10 | 0/10 | -2.2543 | -2.72561 | 0.359 |
| LR_off | leaky | 0/10 | 0/10 | -3.24512 | -2.27546 | 0.63 |
| LR_933 | leaky | 0/10 | 0/10 | -2.2272 | -2.60579 | 0.668 |
| LR_1216 | leaky | 0/10 | 0/10 | -2.20796 | -2.65235 | 0.6415 |

All six non-ReLU arms submerged < 0.05: **False**.
Submergence and strict-dead counts are REPORT_ONLY and were not used in the verdict.

## P3/P4 paired level contrasts at 5M

| endpoint | contrast | median delta log10 U | percentile 95% CI | studentized 95% CI | degenerate |
|---|---|---:|---:|---:|---:|
| P3 | E_off_minus_R_off | -1.81486 | [-2.65693, -1.60731] | [-3.10354, 6.81198e+13] | 1 |
| P3 | E_933_minus_R_933 | -2.63909 | [-2.9844, -2.22505] | [-2.90561e+13, 2.51828e+13] | 1 |
| P3 | E_1216_minus_R_1216 | -2.51636 | [-3.27874, -2.12419] | [-3.45547e+13, 4.25078e+13] | 1 |
| P3 | LR_off_minus_R_off | -2.11441 | [-2.93256, -1.75802] | [-2.29562, 3.85783] | 1 |
| P3 | LR_933_minus_R_933 | -2.34001 | [-3.17918, -1.802] | [-2.36107, 2.43029e+11] | 1 |
| P3 | LR_1216_minus_R_1216 | -2.38562 | [-2.8983, -1.97015] | [-2.4152, -2.09626] | 1 |
| P4 | E_1216_minus_E_933 | 0.0537058 | [-0.71051, 0.55862] | [-0.154005, 0.125533] | 1 |
| P4 | LR_1216_minus_LR_933 | -0.0217357 | [-0.749135, 0.933644] | [-0.0731407, -0.0132616] | 1 |

Jump J over the two fixed in-band doses (9.33 -> 12.16): relu=0.0195557, elu=0.0537058, leaky=0.0217357

## Q2 (REPORT_ONLY)

| arm | status | beta | beta percentile 95% CI | scaling | rho | drift/noise |
|---|---|---:|---:|---|---:|---|
| E_off | OK | 0.601288 | [0.512335, 0.766417] | SCALING_MISMATCH | 0.0031946 | NOISE_DOMINATED |
| E_933 | OK | 0.437794 | [0.29343, 0.456384] | SCALING_MISMATCH | 0.00260821 | NOISE_DOMINATED |
| E_1216 | OK | 0.406409 | [0.338488, 0.541213] | SCALING_MISMATCH | 0.00385608 | NOISE_DOMINATED |
| LR_off | OK | -0.219094 | [-0.270132, -0.0694432] | SCALING_MISMATCH | 0.0773231 | NOISE_DOMINATED |
| LR_933 | OK | -0.16987 | [-0.238581, -0.0554275] | SCALING_MISMATCH | 0.030893 | NOISE_DOMINATED |
| LR_1216 | OK | -0.103001 | [-0.258359, -0.0180825] | SCALING_MISMATCH | 0.0331246 | NOISE_DOMINATED |

## Sanity

- S0prime: **PASS**
- S_pair: **PASS**
- S_pair_final: **PASS**
- S_dose: **PASS**
- S_dose_final: **PASS**
- S_grad: **PASS**
- S_elu_limit: **PASS**
- S_leaky_limit: **PASS**
- S_submerge: **PASS**
- S_tautology: **PASS**
- S6_floor_inherited: **PASS**
