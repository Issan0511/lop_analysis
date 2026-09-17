# escneff_ee_0917 — 判定（escape_ee_0917 を seed 10–19 で・操作の確認は n̄_eff）

> 自動生成: `analysis/escneff_ee_0917/verdict.py`。spec: `specs/spec_escneff_ee_0917.md`。

- HEAD `5ca60841ec2f6175a2333b69bee5d2463af00816`・run の commit ['5ca60841ec2f6175a2333b69bee5d2463af00816']・有効 seed 10（無効: なし）
- (A) True・(B) R_none > 0: True・(C) 微分を運ぶ画像数が減った dneff > 0: True

## 判定

| 問い | ラベル |
|---|---|
| primary | **REMAINDER_REDUCED_BY_HOLD** |
| neff_tracks | **NEFF_TRACKS** |
| slope | **SLOPE_POSITIVE** |
| fixed_field | **FIX_REMAINDER_REDUCED** |
| impairment_flag | **NO_IMPAIRMENT_FLAG** |

n̄_eff と E の順位相関（LADDER 5 腕・seed ごと）: 平均 +0.900、正 10・負 0、符号検定 p = 0.00195。seed 別: +0.90, +0.90, +1.00, +1.00, +0.90, +0.70, +0.90, +0.90, +0.90, +0.90
平均の登りと E の順位相関（報告）: 平均 +0.020。seed 別: +0.80, -0.90, +0.30, +0.40, +0.60, -0.90, -0.90, -0.50, +0.50, +0.80
seed 間の dneff と dR の相関: r = +0.920（t = +6.63、df 8、p = 0.000163）

## 量（seed 内の差）

| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |
|---|---|---|---|---|---|---|---|
| dR | 1 | 0.975 | +0.0514 | +0.0157 | [+0.0381, +0.0647] | + | 10/10 |
| R_c12 | 1 | 0.975 | +0.0766 | +0.0159 | [+0.0632, +0.0901] | + | 10/10 |
| R_none | 1 | 0.95 | +0.1281 | +0.0198 | [+0.1139, +0.1422] | + | 10/10 |
| R_c12 | 1 | 0.95 | +0.0766 | +0.0159 | [+0.0653, +0.0880] | + | 10/10 |
| dR | 1 | 0.95 | +0.0514 | +0.0157 | [+0.0402, +0.0626] | + | 10/10 |
| R_c1 | 1 | 0.95 | +0.1039 | +0.0177 | [+0.0913, +0.1166] | + | 10/10 |
| R_c2 | 1 | 0.95 | +0.1095 | +0.0145 | [+0.0991, +0.1198] | + | 10/10 |
| R_c12b | 1 | 0.95 | +0.0740 | +0.0166 | [+0.0621, +0.0859] | + | 10/10 |
| R_frz | 1 | 0.95 | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0 | 0/10 |
| dR_b | 1 | 0.95 | +0.0026 | +0.0022 | [+0.0011, +0.0042] | + | 9/10 |
| dneff | 1 | 0.95 | +0.0033 | +0.0015 | [+0.0022, +0.0044] | + | 10/10 |
| dclimb | 1 | 0.95 | -0.4252 | +0.7501 | [-0.9618, +0.1114] | 0 | 4/10 |
| I | 1 | 0.95 | -0.0225 | +0.0126 | [-0.0315, -0.0134] | - | 0/10 |
| dR_fix | 1 | 0.95 | +0.0834 | +0.0167 | [+0.0715, +0.0953] | + | 10/10 |
| R_none | 2 | 0.95 | +0.0798 | +0.0274 | [+0.0601, +0.0994] | + | 10/10 |
| R_c12 | 2 | 0.95 | +0.0563 | +0.0186 | [+0.0430, +0.0697] | + | 10/10 |
| dR | 2 | 0.95 | +0.0234 | +0.0157 | [+0.0122, +0.0346] | + | 9/10 |
| dneff | 2 | 0.95 | +0.0016 | +0.0012 | [+0.0007, +0.0024] | + | 9/10 |
| dclimb | 2 | 0.95 | -5.2243 | +1.9007 | [-6.5839, -3.8646] | - | 0/10 |
| I | 2 | 0.95 | -0.1022 | +0.0416 | [-0.1319, -0.0724] | - | 0/10 |
| dR_fix | 2 | 0.95 | +0.0442 | +0.0150 | [+0.0335, +0.0550] | + | 10/10 |

## 腕ごとの E・n̄_eff・網自身の登り（10 seed 平均）

| 腕 | E 継続 1 | E 継続 2 | n̄_eff 継続 1 | n̄_eff 継続 2 | 登り 継続 1 | 登り 継続 2 |
|---|---|---|---|---|---|---|
| S2dyn_10r | 0.3475 | 0.3098 | 0.01078 | 0.00814 | +2.715 | -4.432 |
| S2dyn_c1 | 0.3234 | 0.3055 | 0.00867 | 0.00744 | +2.626 | -0.862 |
| S2dyn_c2 | 0.3289 | 0.3124 | 0.01083 | 0.00941 | +3.628 | -1.384 |
| S2dyn_c12 | 0.2961 | 0.2864 | 0.00746 | 0.00658 | +3.140 | +0.792 |
| S2dyn_c12b | 0.2935 | 0.2813 | 0.00701 | 0.00581 | +2.489 | +0.318 |
| S2dyn_frz | 0.2195 | 0.2300 | 0.00312 | 0.00208 | +0.000 | +0.000 |
| S2u30r | 0.2195 | 0.2300 | 0.00000 | 0.00000 | -0.000 | +0.000 |
| S2u30r_c12 | 0.2195 | 0.2300 | 0.00000 | 0.00000 | -0.000 | +0.000 |
| S2_10r | 0.4730 | 0.4840 | 0.02538 | 0.02391 | +5.528 | -5.239 |
| S2_10r_c12 | 0.3896 | 0.4398 | 0.01801 | 0.02095 | +5.038 | +2.242 |
| N2r | 0.7128 | 0.7456 | 0.44761 | 0.27061 | -1.317 | -2.651 |
| N2r_c12 | 0.7352 | 0.8478 | 0.47942 | 0.43361 | -0.881 | -0.613 |

## 報告（継続 1 タスク目・seed 平均）

| 腕 | 量 | 値 |
|---|---|---|
| S2dyn_10r | w1norm_end | 6.797 |
| S2dyn_10r | w2norm_end | 2.681 |
| S2dyn_10r | b2_mean_end | 0.08263 |
| S2dyn_10r | probe_zero2_mean | 0.9483 |
| S2dyn_10r | nbar_neffT2 | 0.01078 |
| S2dyn_10r | probe_effbar2_750 | -64.9 |
| S2dyn_10r | probe_effbar2_last | -55.97 |
| S2dyn_10r | probe_zarm2_750 | 0.6331 |
| S2dyn_10r | rows_w2 | 0 |
| S2dyn_10r | rows_perp | 0 |
| S2dyn_10r | restored | 0 |
| S2dyn_c1 | w1norm_end | 4.491 |
| S2dyn_c1 | w2norm_end | 2.809 |
| S2dyn_c1 | b2_mean_end | 0.09648 |
| S2dyn_c1 | probe_zero2_mean | 0.9556 |
| S2dyn_c1 | nbar_neffT2 | 0.008668 |
| S2dyn_c1 | probe_effbar2_750 | -65.05 |
| S2dyn_c1 | probe_effbar2_last | -56.06 |
| S2dyn_c1 | probe_zarm2_750 | 0.4877 |
| S2dyn_c1 | rows_w2 | 0 |
| S2dyn_c1 | rows_perp | 4.501e+05 |
| S2dyn_c1 | restored | 0 |
| S2dyn_c2 | w1norm_end | 7.42 |
| S2dyn_c2 | w2norm_end | 1.964 |
| S2dyn_c2 | b2_mean_end | 0.09862 |
| S2dyn_c2 | probe_zero2_mean | 0.9415 |
| S2dyn_c2 | nbar_neffT2 | 0.01083 |
| S2dyn_c2 | probe_effbar2_750 | -64.94 |
| S2dyn_c2 | probe_effbar2_last | -55.06 |
| S2dyn_c2 | probe_zarm2_750 | 0.5995 |
| S2dyn_c2 | rows_w2 | 1.682e+05 |
| S2dyn_c2 | rows_perp | 0 |
| S2dyn_c2 | restored | 0 |
| S2dyn_c12 | w1norm_end | 4.494 |
| S2dyn_c12 | w2norm_end | 1.964 |
| S2dyn_c12 | b2_mean_end | 0.1371 |
| S2dyn_c12 | probe_zero2_mean | 0.9561 |
| S2dyn_c12 | nbar_neffT2 | 0.007459 |
| S2dyn_c12 | probe_effbar2_750 | -65.06 |
| S2dyn_c12 | probe_effbar2_last | -55.55 |
| S2dyn_c12 | probe_zarm2_750 | 0.4737 |
| S2dyn_c12 | rows_w2 | 1.906e+05 |
| S2dyn_c12 | rows_perp | 4.886e+05 |
| S2dyn_c12 | restored | 0 |
| S2dyn_c12b | w1norm_end | 4.495 |
| S2dyn_c12b | w2norm_end | 1.964 |
| S2dyn_c12b | b2_mean_end | 0.03074 |
| S2dyn_c12b | probe_zero2_mean | 0.9585 |
| S2dyn_c12b | nbar_neffT2 | 0.007014 |
| S2dyn_c12b | probe_effbar2_750 | -65.08 |
| S2dyn_c12b | probe_effbar2_last | -56.2 |
| S2dyn_c12b | probe_zarm2_750 | 0.4544 |
| S2dyn_c12b | rows_w2 | 1.941e+05 |
| S2dyn_c12b | rows_perp | 4.884e+05 |
| S2dyn_c12b | restored | 1.059e+06 |
| S2dyn_frz | w1norm_end | 4.499 |
| S2dyn_frz | w2norm_end | 1.979 |
| S2dyn_frz | b2_mean_end | 0.03074 |
| S2dyn_frz | probe_zero2_mean | 0.9736 |
| S2dyn_frz | nbar_neffT2 | 0.003123 |
| S2dyn_frz | probe_effbar2_750 | -65.67 |
| S2dyn_frz | probe_effbar2_last | -58.69 |
| S2dyn_frz | probe_zarm2_750 | -0.1354 |
| S2dyn_frz | rows_w2 | 0 |
| S2dyn_frz | rows_perp | 0 |
| S2dyn_frz | restored | 3.845e+08 |
| S2u30r | w1norm_end | 4.499 |
| S2u30r | w2norm_end | 1.979 |
| S2u30r | b2_mean_end | 0.03074 |
| S2u30r | probe_zero2_mean | 1 |
| S2u30r | nbar_neffT2 | 9.259e-08 |
| S2u30r | probe_effbar2_750 | -30.14 |
| S2u30r | probe_effbar2_last | -30.14 |
| S2u30r | probe_zarm2_750 | -0.1355 |
| S2u30r | rows_w2 | 0 |
| S2u30r | rows_perp | 0 |
| S2u30r | restored | 0 |
| S2u30r_c12 | w1norm_end | 4.499 |
| S2u30r_c12 | w2norm_end | 1.979 |
| S2u30r_c12 | b2_mean_end | 0.03074 |
| S2u30r_c12 | probe_zero2_mean | 1 |
| S2u30r_c12 | nbar_neffT2 | 1.852e-07 |
| S2u30r_c12 | probe_effbar2_750 | -30.14 |
| S2u30r_c12 | probe_effbar2_last | -30.14 |
| S2u30r_c12 | probe_zarm2_750 | -0.1355 |
| S2u30r_c12 | rows_w2 | 0 |
| S2u30r_c12 | rows_perp | 169.7 |
| S2u30r_c12 | restored | 0 |
| S2_10r | w1norm_end | 7.053 |
| S2_10r | w2norm_end | 2.954 |
| S2_10r | b2_mean_end | 0.1138 |
| S2_10r | probe_zero2_mean | 0.8848 |
| S2_10r | nbar_neffT2 | 0.02538 |
| S2_10r | probe_effbar2_750 | -49.25 |
| S2_10r | probe_effbar2_last | -45.28 |
| S2_10r | probe_zarm2_750 | 1.427 |
| S2_10r | rows_w2 | 0 |
| S2_10r | rows_perp | 0 |
| S2_10r | restored | 0 |
| S2_10r_c12 | w1norm_end | 4.495 |
| S2_10r_c12 | w2norm_end | 1.96 |
| S2_10r_c12 | b2_mean_end | 0.2015 |
| S2_10r_c12 | probe_zero2_mean | 0.902 |
| S2_10r_c12 | nbar_neffT2 | 0.01801 |
| S2_10r_c12 | probe_effbar2_750 | -49.88 |
| S2_10r_c12 | probe_effbar2_last | -45.77 |
| S2_10r_c12 | probe_zarm2_750 | 0.7901 |
| S2_10r_c12 | rows_w2 | 3.001e+05 |
| S2_10r_c12 | rows_perp | 5.415e+05 |
| S2_10r_c12 | restored | 0 |
| N2r | w1norm_end | 5.338 |
| N2r | w2norm_end | 2.305 |
| N2r | b2_mean_end | 0.02367 |
| N2r | probe_zero2_mean | 0.0002832 |
| N2r | nbar_neffT2 | 0.4476 |
| N2r | probe_effbar2_750 | -4.475 |
| N2r | probe_effbar2_last | -1.452 |
| N2r | probe_zarm2_750 | -4.475 |
| N2r | rows_w2 | 0 |
| N2r | rows_perp | 0 |
| N2r | restored | 0 |
| N2r_c12 | w1norm_end | 4.49 |
| N2r_c12 | w2norm_end | 1.979 |
| N2r_c12 | b2_mean_end | -0.04868 |
| N2r_c12 | probe_zero2_mean | 7.87e-06 |
| N2r_c12 | nbar_neffT2 | 0.4794 |
| N2r_c12 | probe_effbar2_750 | -3.777 |
| N2r_c12 | probe_effbar2_last | -1.016 |
| N2r_c12 | probe_zarm2_750 | -3.777 |
| N2r_c12 | rows_w2 | 3.399e+05 |
| N2r_c12 | rows_perp | 3.504e+05 |
| N2r_c12 | restored | 0 |
