# bshare_posthoc_0903 v3 (registered = 0, posthoc_not_preregistered)

source: `results/gate_dose_0830`  window: record 0 → first later record with `p_hat == 0`

`share` = median(Δb)/(median(Δ(w·µ'))+median(Δb)) over units, then median over seeds (C1's 取り方). `unit share` = median of the per-unit ratios. `share_ff` = the same ratio of medians with both routes accumulated only over flip-free intervals (µ' constant, so no reorientation term).

| arm | seeds | share | diff | unit share | diff | share_ff | diff_ff | predicted | Δ(w·µ') | Δb | t_sub |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E_1216 | 10/10 | 0.1073 | 0.0097 | 0.0932 | -0.0044 | 0.0975 | -0.0000 | 0.0976 | -1.6038 | -0.1809 | 178500 |
| E_933 | 10/10 | 0.1537 | -0.0015 | 0.1396 | -0.0156 | 0.1414 | -0.0138 | 0.1552 | -1.0475 | -0.1808 | 141000 |
| E_off | 10/10 | 0.1058 | 0.0004 | 0.0978 | -0.0076 | 0.1054 | 0.0000 | 0.1054 | -1.3110 | -0.1711 | 166000 |
| LR_1216 | 10/10 | 0.0948 | -0.0028 | 0.0962 | -0.0014 | 0.1050 | 0.0074 | 0.0976 | -1.6193 | -0.1604 | 34000 |
| LR_933 | 10/10 | 0.1374 | -0.0178 | 0.1412 | -0.0140 | 0.1600 | 0.0048 | 0.1552 | -1.2772 | -0.2010 | 61000 |
| LR_off | 10/10 | 0.1008 | -0.0105 | 0.1018 | -0.0095 | 0.1128 | 0.0015 | 0.1113 | -1.4817 | -0.1764 | 37500 |
| R_1216 | 10/10 | 0.0973 | -0.0003 | 0.0954 | -0.0022 | 0.1070 | 0.0094 | 0.0976 | -1.4832 | -0.1560 | 38250 |
| R_933 | 10/10 | 0.1407 | -0.0145 | 0.1416 | -0.0136 | 0.1715 | 0.0163 | 0.1552 | -1.2375 | -0.1974 | 68500 |
| R_off | 10/10 | 0.1030 | -0.0076 | 0.0988 | -0.0117 | 0.1148 | 0.0042 | 0.1105 | -1.4436 | -0.1737 | 41000 |

## diagnostics (what the headline number hides)

| arm | units/seed | submerged at start | no_descent | share incl. no_descent | routes same sign | share outside [0,1] | ff share outside [0,1] | ff keeps Δb | ff keeps Δ(w·µ') | pred (mean-of-recip) | window (records) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E_1216 | 73 | 234/1000 | 42/1000 | 0.1019 | 0.866 | 0.134 | 0.110 | 0.473 | 0.420 | 0.0976 | 178 |
| E_933 | 78 | 180/1000 | 46/1000 | 0.1528 | 0.818 | 0.182 | 0.084 | 0.481 | 0.427 | 0.1552 | 141 |
| E_off | 74 | 222/1000 | 43/1000 | 0.1019 | 0.853 | 0.147 | 0.132 | 0.520 | 0.410 | 0.1082 | 166 |
| LR_1216 | 78 | 234/1000 | 0/1000 | 0.0948 | 0.987 | 0.013 | 0.119 | 0.413 | 0.351 | 0.0976 | 34 |
| LR_933 | 83 | 180/1000 | 0/1000 | 0.1374 | 0.974 | 0.026 | 0.123 | 0.286 | 0.215 | 0.1552 | 61 |
| LR_off | 80 | 222/1000 | 0/1000 | 0.1008 | 0.975 | 0.025 | 0.120 | 0.396 | 0.305 | 0.1115 | 38 |
| R_1216 | 78 | 234/1000 | 0/1000 | 0.0973 | 0.975 | 0.025 | 0.121 | 0.373 | 0.329 | 0.0976 | 38 |
| R_933 | 83 | 180/1000 | 0/1000 | 0.1407 | 0.950 | 0.050 | 0.100 | 0.277 | 0.206 | 0.1552 | 68 |
| R_off | 80 | 222/1000 | 0/1000 | 0.1030 | 0.980 | 0.020 | 0.131 | 0.359 | 0.301 | 0.1115 | 41 |

No verdict label: the prediction was not frozen before the 12.16 comparison; treat as descriptive (`registered = 0`). A registered PASSed rule for the same functional form exists (命題リスト Q19 C2, `results/mu_titration_0823/dose_response.csv`: 0.10137 [0.09642, 0.10744] at ‖µ‖ = 3.0414 and 0.13356 [0.12820, 0.13856] at 2.3333) — compare against that, not against nothing.
