# bshare_posthoc_0903 v2 (registered = 0, posthoc_not_preregistered)

source: `results/gate_dose_0830`  window: record 1 → first later record with `p_hat == 0`

`share` = median(Δb)/(median(Δ(w·µ'))+median(Δb)) over units, then median over seeds (C1's 取り方). `unit share` = median of the per-unit ratios. `share_ff` = the same ratio of medians with both routes accumulated only over flip-free intervals (µ' constant, so no reorientation term).

| arm | seeds | share | diff | unit share | diff | share_ff | diff_ff | predicted | Δ(w·µ') | Δb | t_sub |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E_1216 | 10/10 | 0.1045 | 0.0069 | 0.0910 | -0.0066 | 0.0967 | -0.0008 | 0.0976 | -1.0892 | -0.1119 | 166000 |
| E_933 | 10/10 | 0.1580 | 0.0028 | 0.1468 | -0.0084 | 0.1639 | 0.0087 | 0.1552 | -0.6890 | -0.1376 | 141000 |
| E_off | 10/10 | 0.1191 | 0.0129 | 0.0934 | -0.0128 | 0.1064 | 0.0002 | 0.1063 | -0.8197 | -0.1150 | 161000 |
| LR_1216 | 10/10 | 0.0895 | -0.0081 | 0.0966 | -0.0010 | 0.1094 | 0.0118 | 0.0976 | -1.2381 | -0.1144 | 41000 |
| LR_933 | 10/10 | 0.1352 | -0.0200 | 0.1315 | -0.0237 | 0.1877 | 0.0324 | 0.1552 | -1.0898 | -0.1703 | 61000 |
| LR_off | 10/10 | 0.0960 | -0.0152 | 0.0977 | -0.0135 | 0.1206 | 0.0094 | 0.1112 | -1.1649 | -0.1344 | 44000 |
| R_1216 | 10/10 | 0.0916 | -0.0060 | 0.0894 | -0.0082 | 0.1257 | 0.0281 | 0.0976 | -1.1952 | -0.1130 | 43500 |
| R_933 | 10/10 | 0.1245 | -0.0307 | 0.1370 | -0.0182 | 0.1516 | -0.0036 | 0.1552 | -1.0779 | -0.1563 | 71000 |
| R_off | 10/10 | 0.0980 | -0.0124 | 0.0969 | -0.0136 | 0.1513 | 0.0409 | 0.1105 | -1.1950 | -0.1276 | 51250 |

## diagnostics (what the headline number hides)

| arm | units/seed | submerged at start | no_descent | share incl. no_descent | routes same sign | share outside [0,1] | pred (mean-of-recip) | window (records) |
|---|---|---|---|---|---|---|---|---|
| E_1216 | 82 | 128/1000 | 57/1000 | 0.1184 | 0.778 | 0.222 | 0.0976 | 165 |
| E_933 | 85 | 102/1000 | 57/1000 | 0.1682 | 0.744 | 0.256 | 0.1552 | 140 |
| E_off | 82 | 124/1000 | 54/1000 | 0.1314 | 0.750 | 0.250 | 0.1081 | 160 |
| LR_1216 | 74 | 260/1000 | 0/1000 | 0.0895 | 0.964 | 0.036 | 0.0976 | 40 |
| LR_933 | 81 | 199/1000 | 0/1000 | 0.1352 | 0.963 | 0.037 | 0.1552 | 60 |
| LR_off | 78 | 241/1000 | 0/1000 | 0.0960 | 0.962 | 0.038 | 0.1114 | 43 |
| R_1216 | 74 | 264/1000 | 0/1000 | 0.0916 | 0.950 | 0.050 | 0.0976 | 42 |
| R_933 | 80 | 203/1000 | 0/1000 | 0.1245 | 0.941 | 0.059 | 0.1552 | 70 |
| R_off | 77 | 246/1000 | 0/1000 | 0.0980 | 0.954 | 0.046 | 0.1114 | 50 |

No verdict label: the prediction was not frozen before the 12.16 comparison; treat as descriptive (`registered = 0`). A registered PASSed rule for the same functional form exists (命題リスト Q19 C2, `results/mu_titration_0823/dose_response.csv`: 0.10137 [0.09642, 0.10744] at ‖µ‖ = 3.0414 and 0.13356 [0.12820, 0.13856] at 2.3333) — compare against that, not against nothing.
