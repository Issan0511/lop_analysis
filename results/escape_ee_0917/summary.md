# escape_ee_0917 — 判定（t2 の網の逃げ場を塞いで respdyn の主腕を回し直す）

> 自動生成: `analysis/escape_ee_0917/verdict.py`。spec: `specs/spec_escape_ee_0917.md`。

- HEAD `52c99b7a29dfef683a44ed60dc404df3adf384c7`・run の commit ['52c99b7a29dfef683a44ed60dc404df3adf384c7']・有効 seed 10（無効: なし）
- (A) True・(B) R_none > 0: True・(C) 逃げが減った dclimb > 0: False

## 判定

| 問い | ラベル |
|---|---|
| primary | **ESCAPE_NOT_MANIPULATED** |
| climb_tracks | **CLIMB_UNRESOLVED** |
| fixed_field | **FIX_REMAINDER_REDUCED** |
| impairment_flag | **NO_IMPAIRMENT_FLAG** |

登りと E の順位相関（LADDER 5 腕・seed ごと）: 平均 +0.230、正 7・負 3、符号検定 p = 0.344。seed 別: -0.10, +0.30, +0.20, +0.20, +1.00, -0.60, -0.80, +0.90, +0.30, +0.90

## 量（seed 内の差）

| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |
|---|---|---|---|---|---|---|---|
| dR | 1 | 0.975 | +0.0460 | +0.0177 | [+0.0309, +0.0610] | + | 10/10 |
| R_c12 | 1 | 0.975 | +0.0660 | +0.0295 | [+0.0409, +0.0911] | + | 10/10 |
| R_none | 1 | 0.95 | +0.1120 | +0.0438 | [+0.0806, +0.1433] | + | 10/10 |
| R_c12 | 1 | 0.95 | +0.0660 | +0.0295 | [+0.0449, +0.0872] | + | 10/10 |
| dR | 1 | 0.95 | +0.0460 | +0.0177 | [+0.0333, +0.0586] | + | 10/10 |
| R_c1 | 1 | 0.95 | +0.0879 | +0.0355 | [+0.0625, +0.1133] | + | 10/10 |
| R_c2 | 1 | 0.95 | +0.0954 | +0.0399 | [+0.0668, +0.1239] | + | 10/10 |
| R_c12b | 1 | 0.95 | +0.0630 | +0.0280 | [+0.0429, +0.0830] | + | 10/10 |
| R_frz | 1 | 0.95 | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 | 0/10 |
| dR_b | 1 | 0.95 | +0.0031 | +0.0021 | [+0.0015, +0.0046] | + | 9/10 |
| dclimb | 1 | 0.95 | +0.5419 | +1.4496 | [-0.4951, +1.5789] | 0 | 5/10 |
| I | 1 | 0.95 | -0.0168 | +0.0158 | [-0.0281, -0.0055] | - | 1/10 |
| dR_fix | 1 | 0.95 | +0.0742 | +0.0176 | [+0.0617, +0.0868] | + | 10/10 |
| R_none | 2 | 0.95 | +0.0812 | +0.0218 | [+0.0656, +0.0968] | + | 10/10 |
| R_c12 | 2 | 0.95 | +0.0453 | +0.0098 | [+0.0383, +0.0523] | + | 10/10 |
| dR | 2 | 0.95 | +0.0359 | +0.0161 | [+0.0244, +0.0474] | + | 10/10 |
| dclimb | 2 | 0.95 | -4.7399 | +1.7120 | [-5.9646, -3.5152] | - | 0/10 |
| I | 2 | 0.95 | -0.0934 | +0.0458 | [-0.1262, -0.0606] | - | 0/10 |
| dR_fix | 2 | 0.95 | +0.0362 | +0.0152 | [+0.0254, +0.0471] | + | 10/10 |

## 腕ごとの E と網自身の登り（10 seed 平均）

| 腕 | E 継続 1 | E 継続 2 | 登り 継続 1 | 登り 継続 2 |
|---|---|---|---|---|
| S2dyn_10r | 0.3237 | 0.3062 | +3.500 | -4.407 |
| S2dyn_c1 | 0.2996 | 0.2932 | +2.730 | -0.548 |
| S2dyn_c2 | 0.3070 | 0.3012 | +3.765 | -1.832 |
| S2dyn_c12 | 0.2777 | 0.2703 | +2.958 | +0.333 |
| S2dyn_c12b | 0.2747 | 0.2688 | +2.475 | +0.038 |
| S2dyn_frz | 0.2117 | 0.2250 | +0.000 | +0.000 |
| S2u30r | 0.2117 | 0.2250 | -0.000 | +0.000 |
| S2u30r_c12 | 0.2117 | 0.2250 | -0.000 | +0.000 |
| S2_10r | 0.4644 | 0.4624 | +4.948 | -5.860 |
| S2_10r_c12 | 0.3902 | 0.4261 | +4.772 | +1.884 |
| N2r | 0.7239 | 0.7477 | -0.798 | -2.592 |
| N2r_c12 | 0.7407 | 0.8411 | -0.886 | -0.595 |

## 報告（継続 1 タスク目・seed 平均）

| 腕 | 量 | 値 |
|---|---|---|
| S2dyn_10r | w1norm_end | 6.645 |
| S2dyn_10r | w2norm_end | 2.607 |
| S2dyn_10r | b2_mean_end | 0.08371 |
| S2dyn_10r | probe_zero2_mean | 0.9516 |
| S2dyn_10r | nbar_neffT2 | 0.009506 |
| S2dyn_10r | probe_effbar2_750 | -65.41 |
| S2dyn_10r | probe_effbar2_last | -56.27 |
| S2dyn_10r | probe_zarm2_750 | 0.4542 |
| S2dyn_10r | rows_w2 | 0 |
| S2dyn_10r | rows_perp | 0 |
| S2dyn_10r | restored | 0 |
| S2dyn_c1 | w1norm_end | 4.432 |
| S2dyn_c1 | w2norm_end | 2.721 |
| S2dyn_c1 | b2_mean_end | 0.09619 |
| S2dyn_c1 | probe_zero2_mean | 0.9601 |
| S2dyn_c1 | nbar_neffT2 | 0.007365 |
| S2dyn_c1 | probe_effbar2_750 | -65.49 |
| S2dyn_c1 | probe_effbar2_last | -57.05 |
| S2dyn_c1 | probe_zarm2_750 | 0.3648 |
| S2dyn_c1 | rows_w2 | 0 |
| S2dyn_c1 | rows_perp | 4.353e+05 |
| S2dyn_c1 | restored | 0 |
| S2dyn_c2 | w1norm_end | 7.224 |
| S2dyn_c2 | w2norm_end | 1.934 |
| S2dyn_c2 | b2_mean_end | 0.09628 |
| S2dyn_c2 | probe_zero2_mean | 0.947 |
| S2dyn_c2 | nbar_neffT2 | 0.009392 |
| S2dyn_c2 | probe_effbar2_750 | -65.41 |
| S2dyn_c2 | probe_effbar2_last | -56.01 |
| S2dyn_c2 | probe_zarm2_750 | 0.4491 |
| S2dyn_c2 | rows_w2 | 1.566e+05 |
| S2dyn_c2 | rows_perp | 0 |
| S2dyn_c2 | restored | 0 |
| S2dyn_c12 | w1norm_end | 4.435 |
| S2dyn_c12 | w2norm_end | 1.934 |
| S2dyn_c12 | b2_mean_end | 0.1321 |
| S2dyn_c12 | probe_zero2_mean | 0.9611 |
| S2dyn_c12 | nbar_neffT2 | 0.006357 |
| S2dyn_c12 | probe_effbar2_750 | -65.51 |
| S2dyn_c12 | probe_effbar2_last | -56.82 |
| S2dyn_c12 | probe_zarm2_750 | 0.3465 |
| S2dyn_c12 | rows_w2 | 1.786e+05 |
| S2dyn_c12 | rows_perp | 4.746e+05 |
| S2dyn_c12 | restored | 0 |
| S2dyn_c12b | w1norm_end | 4.436 |
| S2dyn_c12b | w2norm_end | 1.935 |
| S2dyn_c12b | b2_mean_end | 0.03221 |
| S2dyn_c12b | probe_zero2_mean | 0.9633 |
| S2dyn_c12b | nbar_neffT2 | 0.005958 |
| S2dyn_c12b | probe_effbar2_750 | -65.53 |
| S2dyn_c12b | probe_effbar2_last | -57.3 |
| S2dyn_c12b | probe_zarm2_750 | 0.3281 |
| S2dyn_c12b | rows_w2 | 1.815e+05 |
| S2dyn_c12b | rows_perp | 4.751e+05 |
| S2dyn_c12b | restored | 1.04e+06 |
| S2dyn_frz | w1norm_end | 4.441 |
| S2dyn_frz | w2norm_end | 1.949 |
| S2dyn_frz | b2_mean_end | 0.03221 |
| S2dyn_frz | probe_zero2_mean | 0.9772 |
| S2dyn_frz | nbar_neffT2 | 0.00279 |
| S2dyn_frz | probe_effbar2_750 | -65.97 |
| S2dyn_frz | probe_effbar2_last | -59.78 |
| S2dyn_frz | probe_zarm2_750 | -0.1145 |
| S2dyn_frz | rows_w2 | 0 |
| S2dyn_frz | rows_perp | 0 |
| S2dyn_frz | restored | 3.763e+08 |
| S2u30r | w1norm_end | 4.441 |
| S2u30r | w2norm_end | 1.949 |
| S2u30r | b2_mean_end | 0.03221 |
| S2u30r | probe_zero2_mean | 1 |
| S2u30r | nbar_neffT2 | 9.259e-08 |
| S2u30r | probe_effbar2_750 | -30.11 |
| S2u30r | probe_effbar2_last | -30.11 |
| S2u30r | probe_zarm2_750 | -0.1146 |
| S2u30r | rows_w2 | 0 |
| S2u30r | rows_perp | 0 |
| S2u30r | restored | 0 |
| S2u30r_c12 | w1norm_end | 4.441 |
| S2u30r_c12 | w2norm_end | 1.949 |
| S2u30r_c12 | b2_mean_end | 0.03221 |
| S2u30r_c12 | probe_zero2_mean | 1 |
| S2u30r_c12 | nbar_neffT2 | 9.259e-08 |
| S2u30r_c12 | probe_effbar2_750 | -30.11 |
| S2u30r_c12 | probe_effbar2_last | -30.11 |
| S2u30r_c12 | probe_zarm2_750 | -0.1145 |
| S2u30r_c12 | rows_w2 | 0 |
| S2u30r_c12 | rows_perp | 100.6 |
| S2u30r_c12 | restored | 0 |
| S2_10r | w1norm_end | 6.972 |
| S2_10r | w2norm_end | 2.909 |
| S2_10r | b2_mean_end | 0.1093 |
| S2_10r | probe_zero2_mean | 0.8881 |
| S2_10r | nbar_neffT2 | 0.02432 |
| S2_10r | probe_effbar2_750 | -49.53 |
| S2_10r | probe_effbar2_last | -45.92 |
| S2_10r | probe_zarm2_750 | 1.223 |
| S2_10r | rows_w2 | 0 |
| S2_10r | rows_perp | 0 |
| S2_10r | restored | 0 |
| S2_10r_c12 | w1norm_end | 4.435 |
| S2_10r_c12 | w2norm_end | 1.933 |
| S2_10r_c12 | b2_mean_end | 0.1899 |
| S2_10r_c12 | probe_zero2_mean | 0.903 |
| S2_10r_c12 | nbar_neffT2 | 0.01741 |
| S2_10r_c12 | probe_effbar2_750 | -49.97 |
| S2_10r_c12 | probe_effbar2_last | -46.1 |
| S2_10r_c12 | probe_zarm2_750 | 0.7814 |
| S2_10r_c12 | rows_w2 | 3.023e+05 |
| S2_10r_c12 | rows_perp | 5.42e+05 |
| S2_10r_c12 | restored | 0 |
| N2r | w1norm_end | 5.219 |
| N2r | w2norm_end | 2.242 |
| N2r | b2_mean_end | 0.02848 |
| N2r | probe_zero2_mean | 0.0001928 |
| N2r | nbar_neffT2 | 0.498 |
| N2r | probe_effbar2_750 | -4.101 |
| N2r | probe_effbar2_last | -0.912 |
| N2r | probe_zarm2_750 | -4.101 |
| N2r | rows_w2 | 0 |
| N2r | rows_perp | 0 |
| N2r | restored | 0 |
| N2r_c12 | w1norm_end | 4.431 |
| N2r_c12 | w2norm_end | 1.948 |
| N2r_c12 | b2_mean_end | -0.04296 |
| N2r_c12 | probe_zero2_mean | 0 |
| N2r_c12 | nbar_neffT2 | 0.5005 |
| N2r_c12 | probe_effbar2_750 | -3.701 |
| N2r_c12 | probe_effbar2_last | -1 |
| N2r_c12 | probe_zarm2_750 | -3.701 |
| N2r_c12 | rows_w2 | 3.487e+05 |
| N2r_c12 | rows_perp | 3.575e+05 |
| N2r_c12 | restored | 0 |
