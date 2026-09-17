# swap_ee_0917 — 判定（cap12 の網と ref の網で第2層の場を入れ替える）

> 自動生成: `analysis/swap_ee_0917/verdict.py`。spec: `specs/spec_swap_ee_0917.md`。

- HEAD `c1940966ce9114a527f48621e9a70bfe4dd75d2e`・run の commit ['c1940966ce9114a527f48621e9a70bfe4dd75d2e']・有効 seed 10（無効: なし）
- (A) 有効 seed ≥ 8: **True**・(B) TE +0.6395 [+0.6239, +0.6551]・R_T2 +0.1120 [+0.0806, +0.1433] → **True**

## 判定

| 問い | ラベル |
|---|---|
| Q1 | **REMAINDER_IN_BOTH** |
| Q2 | **BOTH_WAYS** |
| Q3_P_C | **GROWTH_HELPS** |
| Q3_P_R | **GROWTH_COSTS** |
| Q3_P_N | **GROWTH_COSTS** |
| Q3_P_F | **GROWTH_COSTS** |

## 量（seed 内の差）

| 量 | 継続 | 水準 | 平均 | SD | 区間 | 符号 | 正の seed |
|---|---|---|---|---|---|---|---|
| R_C | 1 | 0.975 | +0.0373 | +0.0186 | [+0.0215, +0.0531] | + | 10/10 |
| D | 1 | 0.975 | +0.0156 | +0.0322 | [-0.0118, +0.0429] | 0 | 7/10 |
| R_R | 1 | 0.975 | +0.0218 | +0.0206 | [+0.0043, +0.0392] | + | 9/10 |
| R_T2 | 1 | 0.975 | +0.1120 | +0.0438 | [+0.0748, +0.1492] | + | 10/10 |
| R_C_minus_R_T2 | 1 | 0.975 | -0.0747 | +0.0263 | [-0.0970, -0.0524] | - | 0/10 |
| R_C | 1 | 0.95 | +0.0373 | +0.0186 | [+0.0240, +0.0506] | + | 10/10 |
| D | 1 | 0.95 | +0.0156 | +0.0322 | [-0.0075, +0.0386] | 0 | 7/10 |
| R_R | 1 | 0.95 | +0.0218 | +0.0206 | [+0.0070, +0.0365] | + | 9/10 |
| R_T2 | 1 | 0.95 | +0.1120 | +0.0438 | [+0.0806, +0.1433] | + | 10/10 |
| R_C_minus_R_T2 | 1 | 0.95 | -0.0747 | +0.0263 | [-0.0935, -0.0559] | - | 0/10 |
| TE | 1 | 0.95 | +0.6395 | +0.0218 | [+0.6239, +0.6551] | + | 10/10 |
| L_S | 1 | 0.95 | +0.6109 | +0.0208 | [+0.5960, +0.6258] | + | 10/10 |
| L_S_fix | 1 | 0.95 | +0.5286 | +0.0477 | [+0.4945, +0.5627] | + | 10/10 |
| G_R | 1 | 0.95 | +0.6768 | +0.0221 | [+0.6610, +0.6927] | + | 10/10 |
| G_R_fix | 1 | 0.95 | +0.7497 | +0.0254 | [+0.7315, +0.7679] | + | 10/10 |
| M_C | 1 | 0.95 | +0.0823 | +0.0310 | [+0.0601, +0.1044] | + | 10/10 |
| P_C | 1 | 0.95 | -0.0129 | +0.0052 | [-0.0166, -0.0091] | - | 0/10 |
| P_R | 1 | 0.95 | +0.0088 | +0.0027 | [+0.0068, +0.0108] | + | 10/10 |
| P_N | 1 | 0.95 | +0.0529 | +0.0211 | [+0.0378, +0.0680] | + | 10/10 |
| P_F | 1 | 0.95 | +0.0577 | +0.0107 | [+0.0501, +0.0654] | + | 10/10 |
| R_C | 2 | 0.95 | +0.0344 | +0.0113 | [+0.0263, +0.0425] | + | 10/10 |
| D | 2 | 0.95 | +0.0798 | +0.0302 | [+0.0582, +0.1014] | + | 10/10 |
| TE | 2 | 0.95 | +0.7278 | +0.0117 | [+0.7194, +0.7361] | + | 10/10 |
| L_S | 2 | 0.95 | +0.5921 | +0.0082 | [+0.5862, +0.5980] | + | 10/10 |
| G_R | 2 | 0.95 | +0.5727 | +0.0408 | [+0.5435, +0.6019] | + | 10/10 |
| P_C | 2 | 0.95 | -0.0239 | +0.0123 | [-0.0327, -0.0151] | - | 0/10 |
| P_R | 2 | 0.95 | +0.0900 | +0.0288 | [+0.0694, +0.1106] | + | 10/10 |
| P_N | 2 | 0.95 | +0.0617 | +0.0194 | [+0.0478, +0.0756] | + | 10/10 |
| P_F | 2 | 0.95 | +0.3697 | +0.0901 | [+0.3053, +0.4342] | + | 10/10 |

比（TE に対する・報告）: rho_sink_dyn 0.955、rho_sink_fix 0.827、rho_restore_dyn 1.058、rho_restore_fix 1.172

## 腕ごとの E（10 seed 平均）

| 腕 | 継続 1 | 継続 2 |
|---|---|---|
| NC_r | 0.8082 | 0.8591 |
| NC_free_r | 0.7505 | 0.4893 |
| SC_fix_r | 0.2796 | 0.3923 |
| SC_dyn_r | 0.1973 | 0.2670 |
| SC_dyn_free_r | 0.2102 | 0.2909 |
| SC_u30_r | 0.1600 | 0.2325 |
| NR_r | 0.1687 | 0.1313 |
| NR_cap_r | 0.2216 | 0.1930 |
| RR_fix_r | 0.9184 | 0.7962 |
| RR_dyn_r | 0.8455 | 0.7040 |
| RR_dyn_cap_r | 0.8543 | 0.7940 |
| RR_u30_r | 0.1469 | 0.1767 |
| S2dyn_10r | 0.3237 | 0.3062 |
| S2u30r | 0.2117 | 0.2250 |

## 報告（継続 1 タスク目・seed 平均）

| 腕 | 量 | 値 |
|---|---|---|
| NC_r | probe_zarm2_first | -2.381 |
| NC_r | probe_zarm2_750 | -5.515 |
| NC_r | probe_zarm2_last | -2.579 |
| NC_r | probe_dmean2_first | 0 |
| NC_r | probe_dmean2_750 | 0 |
| NC_r | probe_dmean2_last | 0 |
| NC_r | probe_effbar2_first | -2.381 |
| NC_r | probe_effbar2_750 | -5.515 |
| NC_r | probe_effbar2_last | -2.579 |
| NC_r | nbar_neffT2 | 0.3203 |
| NC_r | probe_zero2_mean | 0.0007194 |
| NC_r | w2norm_end | 1.613 |
| NC_r | exc_w2_own | 1.772e-07 |
| NC_r | exc_v_own | 4.548e-07 |
| NC_r | exc_q_own | 6.901e-09 |
| NC_r | zero2_start | 0.0001042 |
| NC_r | zero2_end | 0.0001567 |
| NC_r | mu2_end | 5.809 |
| NC_r | memo_acc_end | 0.9984 |
| NC_free_r | probe_zarm2_first | -2.381 |
| NC_free_r | probe_zarm2_750 | -8.155 |
| NC_free_r | probe_zarm2_last | -4.765 |
| NC_free_r | probe_dmean2_first | 0 |
| NC_free_r | probe_dmean2_750 | 0 |
| NC_free_r | probe_dmean2_last | 0 |
| NC_free_r | probe_effbar2_first | -2.381 |
| NC_free_r | probe_effbar2_750 | -8.155 |
| NC_free_r | probe_effbar2_last | -4.765 |
| NC_free_r | nbar_neffT2 | 0.2077 |
| NC_free_r | probe_zero2_mean | 0.01134 |
| NC_free_r | w2norm_end | 1.97 |
| NC_free_r | exc_w2_own | 0.6989 |
| NC_free_r | exc_v_own | 1.595 |
| NC_free_r | exc_q_own | 0.2503 |
| NC_free_r | zero2_start | 0.0001042 |
| NC_free_r | zero2_end | 0.01022 |
| NC_free_r | mu2_end | 10.51 |
| NC_free_r | memo_acc_end | 1 |
| SC_fix_r | probe_zarm2_first | -2.381 |
| SC_fix_r | probe_zarm2_750 | -2.941 |
| SC_fix_r | probe_zarm2_last | -0.628 |
| SC_fix_r | probe_dmean2_first | -48.49 |
| SC_fix_r | probe_dmean2_750 | -48.49 |
| SC_fix_r | probe_dmean2_last | -48.49 |
| SC_fix_r | probe_effbar2_first | -50.87 |
| SC_fix_r | probe_effbar2_750 | -51.43 |
| SC_fix_r | probe_effbar2_last | -49.12 |
| SC_fix_r | nbar_neffT2 | 0.009162 |
| SC_fix_r | probe_zero2_mean | 0.9322 |
| SC_fix_r | w2norm_end | 1.602 |
| SC_fix_r | exc_w2_own | 1.723e-07 |
| SC_fix_r | exc_v_own | 4.292e-07 |
| SC_fix_r | exc_q_own | 3.539e-08 |
| SC_fix_r | zero2_start | 0.9392 |
| SC_fix_r | zero2_end | 0.9194 |
| SC_fix_r | mu2_end | 11.43 |
| SC_fix_r | memo_acc_end | 0.4402 |
| SC_dyn_r | probe_zarm2_first | -2.381 |
| SC_dyn_r | probe_zarm2_750 | -2.466 |
| SC_dyn_r | probe_zarm2_last | -1.38 |
| SC_dyn_r | probe_dmean2_first | -48.49 |
| SC_dyn_r | probe_dmean2_750 | -63.59 |
| SC_dyn_r | probe_dmean2_last | -57.39 |
| SC_dyn_r | probe_effbar2_first | -50.87 |
| SC_dyn_r | probe_effbar2_750 | -66.06 |
| SC_dyn_r | probe_effbar2_last | -58.77 |
| SC_dyn_r | nbar_neffT2 | 0.003898 |
| SC_dyn_r | probe_zero2_mean | 0.9722 |
| SC_dyn_r | w2norm_end | 1.599 |
| SC_dyn_r | exc_w2_own | 1.645e-07 |
| SC_dyn_r | exc_v_own | 4.391e-07 |
| SC_dyn_r | exc_q_own | 3.708e-08 |
| SC_dyn_r | zero2_start | 0.9392 |
| SC_dyn_r | zero2_end | 0.956 |
| SC_dyn_r | mu2_end | 9.086 |
| SC_dyn_r | memo_acc_end | 0.3532 |
| SC_dyn_free_r | probe_zarm2_first | -2.381 |
| SC_dyn_free_r | probe_zarm2_750 | -2.866 |
| SC_dyn_free_r | probe_zarm2_last | -0.7501 |
| SC_dyn_free_r | probe_dmean2_first | -48.49 |
| SC_dyn_free_r | probe_dmean2_750 | -63.59 |
| SC_dyn_free_r | probe_dmean2_last | -57.39 |
| SC_dyn_free_r | probe_effbar2_first | -50.87 |
| SC_dyn_free_r | probe_effbar2_750 | -66.46 |
| SC_dyn_free_r | probe_effbar2_last | -58.14 |
| SC_dyn_free_r | nbar_neffT2 | 0.004621 |
| SC_dyn_free_r | probe_zero2_mean | 0.9691 |
| SC_dyn_free_r | w2norm_end | 1.995 |
| SC_dyn_free_r | exc_w2_own | 2.261 |
| SC_dyn_free_r | exc_v_own | 2.127 |
| SC_dyn_free_r | exc_q_own | 0.777 |
| SC_dyn_free_r | zero2_start | 0.9392 |
| SC_dyn_free_r | zero2_end | 0.9484 |
| SC_dyn_free_r | mu2_end | 16.99 |
| SC_dyn_free_r | memo_acc_end | 0.4064 |
| SC_u30_r | probe_zarm2_first | -2.381 |
| SC_u30_r | probe_zarm2_750 | -2.381 |
| SC_u30_r | probe_zarm2_last | -2.381 |
| SC_u30_r | probe_dmean2_first | -30 |
| SC_u30_r | probe_dmean2_750 | -30 |
| SC_u30_r | probe_dmean2_last | -30 |
| SC_u30_r | probe_effbar2_first | -32.38 |
| SC_u30_r | probe_effbar2_750 | -32.38 |
| SC_u30_r | probe_effbar2_last | -32.38 |
| SC_u30_r | nbar_neffT2 | 0 |
| SC_u30_r | probe_zero2_mean | 1 |
| SC_u30_r | w2norm_end | 1.613 |
| SC_u30_r | exc_w2_own | 1.125e-07 |
| SC_u30_r | exc_v_own | 3.015e-07 |
| SC_u30_r | exc_q_own | -1.895e-06 |
| SC_u30_r | zero2_start | 1 |
| SC_u30_r | zero2_end | 1 |
| SC_u30_r | mu2_end | 5.509 |
| SC_u30_r | memo_acc_end | 0.2547 |
| NR_r | probe_zarm2_first | -50.87 |
| NR_r | probe_zarm2_750 | -61.97 |
| NR_r | probe_zarm2_last | -54.27 |
| NR_r | probe_dmean2_first | 0 |
| NR_r | probe_dmean2_750 | 0 |
| NR_r | probe_dmean2_last | 0 |
| NR_r | probe_effbar2_first | -50.87 |
| NR_r | probe_effbar2_750 | -61.97 |
| NR_r | probe_effbar2_last | -54.27 |
| NR_r | nbar_neffT2 | 0.004257 |
| NR_r | probe_zero2_mean | 0.9631 |
| NR_r | w2norm_end | 3.969 |
| NR_r | exc_w2_own | 0.7472 |
| NR_r | exc_v_own | 1.18 |
| NR_r | exc_q_own | 1.021 |
| NR_r | zero2_start | 0.9392 |
| NR_r | zero2_end | 0.9436 |
| NR_r | mu2_end | 76.77 |
| NR_r | memo_acc_end | 0.2692 |
| NR_cap_r | probe_zarm2_first | -50.87 |
| NR_cap_r | probe_zarm2_750 | -51.64 |
| NR_cap_r | probe_zarm2_last | -41.98 |
| NR_cap_r | probe_dmean2_first | 0 |
| NR_cap_r | probe_dmean2_750 | 0 |
| NR_cap_r | probe_dmean2_last | 0 |
| NR_cap_r | probe_effbar2_first | -50.87 |
| NR_cap_r | probe_effbar2_750 | -51.64 |
| NR_cap_r | probe_effbar2_last | -41.98 |
| NR_cap_r | nbar_neffT2 | 0.008051 |
| NR_cap_r | probe_zero2_mean | 0.9251 |
| NR_cap_r | w2norm_end | 3.866 |
| NR_cap_r | exc_w2_own | 3.294e-07 |
| NR_cap_r | exc_v_own | 1.06e-06 |
| NR_cap_r | exc_q_own | -0.001095 |
| NR_cap_r | zero2_start | 0.9392 |
| NR_cap_r | zero2_end | 0.8744 |
| NR_cap_r | mu2_end | 63.71 |
| NR_cap_r | memo_acc_end | 0.3835 |
| RR_fix_r | probe_zarm2_first | -50.87 |
| RR_fix_r | probe_zarm2_750 | -51.06 |
| RR_fix_r | probe_zarm2_last | -50.22 |
| RR_fix_r | probe_dmean2_first | 48.49 |
| RR_fix_r | probe_dmean2_750 | 48.49 |
| RR_fix_r | probe_dmean2_last | 48.49 |
| RR_fix_r | probe_effbar2_first | -2.381 |
| RR_fix_r | probe_effbar2_750 | -2.573 |
| RR_fix_r | probe_effbar2_last | -1.73 |
| RR_fix_r | nbar_neffT2 | 0.4629 |
| RR_fix_r | probe_zero2_mean | 0.0006555 |
| RR_fix_r | w2norm_end | 3.931 |
| RR_fix_r | exc_w2_own | 0.1991 |
| RR_fix_r | exc_v_own | 0.5905 |
| RR_fix_r | exc_q_own | 0.3009 |
| RR_fix_r | zero2_start | 0.0001042 |
| RR_fix_r | zero2_end | 0.0009 |
| RR_fix_r | mu2_end | 69.01 |
| RR_fix_r | memo_acc_end | 0.9993 |
| RR_dyn_r | probe_zarm2_first | -50.87 |
| RR_dyn_r | probe_zarm2_750 | -51.37 |
| RR_dyn_r | probe_zarm2_last | -51.28 |
| RR_dyn_r | probe_dmean2_first | 48.49 |
| RR_dyn_r | probe_dmean2_750 | 46.08 |
| RR_dyn_r | probe_dmean2_last | 48.53 |
| RR_dyn_r | probe_effbar2_first | -2.381 |
| RR_dyn_r | probe_effbar2_750 | -5.289 |
| RR_dyn_r | probe_effbar2_last | -2.751 |
| RR_dyn_r | nbar_neffT2 | 0.3738 |
| RR_dyn_r | probe_zero2_mean | 0.006122 |
| RR_dyn_r | w2norm_end | 3.979 |
| RR_dyn_r | exc_w2_own | 0.3203 |
| RR_dyn_r | exc_v_own | 0.7687 |
| RR_dyn_r | exc_q_own | 0.3757 |
| RR_dyn_r | zero2_start | 0.0001042 |
| RR_dyn_r | zero2_end | 0.009958 |
| RR_dyn_r | mu2_end | 69.84 |
| RR_dyn_r | memo_acc_end | 0.9949 |
| RR_dyn_cap_r | probe_zarm2_first | -50.87 |
| RR_dyn_cap_r | probe_zarm2_750 | -50.9 |
| RR_dyn_cap_r | probe_zarm2_last | -50.87 |
| RR_dyn_cap_r | probe_dmean2_first | 48.49 |
| RR_dyn_cap_r | probe_dmean2_750 | 46.08 |
| RR_dyn_cap_r | probe_dmean2_last | 48.53 |
| RR_dyn_cap_r | probe_effbar2_first | -2.381 |
| RR_dyn_cap_r | probe_effbar2_750 | -4.822 |
| RR_dyn_cap_r | probe_effbar2_last | -2.338 |
| RR_dyn_cap_r | nbar_neffT2 | 0.3901 |
| RR_dyn_cap_r | probe_zero2_mean | 0.003892 |
| RR_dyn_cap_r | w2norm_end | 3.872 |
| RR_dyn_cap_r | exc_w2_own | 2.377e-07 |
| RR_dyn_cap_r | exc_v_own | 1.013e-06 |
| RR_dyn_cap_r | exc_q_own | -0.003881 |
| RR_dyn_cap_r | zero2_start | 0.0001042 |
| RR_dyn_cap_r | zero2_end | 0.004738 |
| RR_dyn_cap_r | mu2_end | 68.17 |
| RR_dyn_cap_r | memo_acc_end | 1 |
| RR_u30_r | probe_zarm2_first | -50.87 |
| RR_u30_r | probe_zarm2_750 | -53.48 |
| RR_u30_r | probe_zarm2_last | -43.82 |
| RR_u30_r | probe_dmean2_first | -30 |
| RR_u30_r | probe_dmean2_750 | -30 |
| RR_u30_r | probe_dmean2_last | -30 |
| RR_u30_r | probe_effbar2_first | -80.87 |
| RR_u30_r | probe_effbar2_750 | -83.48 |
| RR_u30_r | probe_effbar2_last | -73.82 |
| RR_u30_r | nbar_neffT2 | 0.001986 |
| RR_u30_r | probe_zero2_mean | 0.9929 |
| RR_u30_r | w2norm_end | 3.994 |
| RR_u30_r | exc_w2_own | 1.513 |
| RR_u30_r | exc_v_own | 1.568 |
| RR_u30_r | exc_q_own | 0.9536 |
| RR_u30_r | zero2_start | 0.9986 |
| RR_u30_r | zero2_end | 0.9808 |
| RR_u30_r | mu2_end | 70.89 |
| RR_u30_r | memo_acc_end | 0.2354 |
| S2dyn_10r | probe_zarm2_first | -0.1145 |
| S2dyn_10r | probe_zarm2_last | 3.386 |
| S2dyn_10r | probe_dmean2_first | -50.76 |
| S2dyn_10r | probe_dmean2_last | -59.66 |
| S2dyn_10r | probe_effbar2_first | -50.87 |
| S2dyn_10r | probe_effbar2_last | -56.27 |
| S2dyn_10r | nbar_neffT2 | 0.009506 |
| S2dyn_10r | probe_zero2_mean | 0.9516 |
| S2dyn_10r | zero2_start | 0.9392 |
| S2dyn_10r | zero2_end | 0.9164 |
| S2dyn_10r | mu2_end | 25.44 |
| S2dyn_10r | memo_acc_end | 0.5766 |
| S2u30r | probe_zarm2_first | -0.1145 |
| S2u30r | probe_zarm2_last | -0.1146 |
| S2u30r | probe_dmean2_first | -30 |
| S2u30r | probe_dmean2_last | -30 |
| S2u30r | probe_effbar2_first | -30.11 |
| S2u30r | probe_effbar2_last | -30.11 |
| S2u30r | nbar_neffT2 | 9.259e-08 |
| S2u30r | probe_zero2_mean | 1 |
| S2u30r | zero2_start | 1 |
| S2u30r | zero2_end | 1 |
| S2u30r | mu2_end | 5.546 |
| S2u30r | memo_acc_end | 0.2832 |
