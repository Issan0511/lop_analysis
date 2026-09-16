# mucap_el_0916 — 判定（第1層の平均画像方向成分と直交成分の上限・ELU→leaky・150 タスク）

> 自動生成: `analysis/mucap_el_0916/verdict.py`。spec: `specs/spec_mucap_el_0916.md`。数値は `verdict.csv`・`paired.csv`・`secondary.csv` から。

## 0. 実行したもの

- 集計時の HEAD: `01524e08f81b40dba765cfdb3fb110faa4c2c71f`・run の commit: ['01524e08f81b40dba765cfdb3fb110faa4c2c71f']
- shard: 40/40・欠損 0・有効 seed 10（無効: なし）

## 1. 適用条件（spec §5.2）

- (A) 有効 seed 10 ≥ 8: **True**
- (B) ref の現象（95%・上端 < 0）: **True** — E1 -0.7083 [-0.7267, -0.6898]、E2 -2.1235 [-2.1932, -2.0537]
- (C) 上限が主窓の全タスクで書いた: cap_par **True**、cap_perp **True**、cap_both **True**
- IMPAIRED（読みのフラグ）: cap_par 0/10 seed → **False**、cap_perp 0/10 seed → **False**、cap_both 0/10 seed → **False**

## 2. 判定

| 比較 | 水準 | ラベル |
|---|---|---|
| **主: cap_both − ref** | 97.5% | **BOTH_HELD** |
| cap_par − ref | 95% | BOTH_HELD |
| cap_perp − ref | 95% | BOTH_HELD |
| cap_both − ref（参考） | 95% | BOTH_HELD |

## 3. 腕間差（主窓 task 51–150・seed 内対応差）

| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |
|---|---|---|---|---|---|---|
| cap_par | E1 | 0.975 | +0.4273 | +0.0070 | [+0.4213, +0.4332] | + |
| cap_par | E2 | 0.975 | +1.8238 | +0.0326 | [+1.7961, +1.8514] | + |
| cap_par | E1 | 0.95 | +0.4273 | +0.0070 | [+0.4223, +0.4323] | + |
| cap_par | E2 | 0.95 | +1.8238 | +0.0326 | [+1.8005, +1.8471] | + |
| cap_perp | E1 | 0.975 | +0.2891 | +0.0306 | [+0.2631, +0.3151] | + |
| cap_perp | E2 | 0.975 | +0.9673 | +0.1067 | [+0.8767, +1.0579] | + |
| cap_perp | E1 | 0.95 | +0.2891 | +0.0306 | [+0.2672, +0.3110] | + |
| cap_perp | E2 | 0.95 | +0.9673 | +0.1067 | [+0.8910, +1.0437] | + |
| cap_both | E1 | 0.975 | +0.2950 | +0.0098 | [+0.2867, +0.3033] | + |
| cap_both | E2 | 0.975 | +0.8903 | +0.0450 | [+0.8521, +0.9285] | + |
| cap_both | E1 | 0.95 | +0.2950 | +0.0098 | [+0.2880, +0.3020] | + |
| cap_both | E2 | 0.95 | +0.8903 | +0.0450 | [+0.8581, +0.9225] | + |

交互作用 δ_both − δ_par − δ_perp（95%）: E1 -0.4214 [-0.4409, -0.4019]、E2 -1.9008 [-1.9704, -1.8311]

### 3.1 seed 別の差

| 腕 | endpoint | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cap_par | E1 | +0.420 | +0.426 | +0.417 | +0.429 | +0.425 | +0.429 | +0.441 | +0.432 | +0.432 | +0.422 |
| cap_par | E2 | +1.784 | +1.834 | +1.805 | +1.841 | +1.783 | +1.795 | +1.877 | +1.812 | +1.858 | +1.849 |
| cap_perp | E1 | +0.303 | +0.289 | +0.231 | +0.297 | +0.296 | +0.338 | +0.251 | +0.274 | +0.314 | +0.299 |
| cap_perp | E2 | +1.014 | +0.965 | +0.770 | +1.018 | +0.962 | +1.108 | +0.842 | +0.884 | +1.078 | +1.032 |
| cap_both | E1 | +0.296 | +0.288 | +0.274 | +0.293 | +0.294 | +0.298 | +0.309 | +0.300 | +0.306 | +0.292 |
| cap_both | E2 | +0.903 | +0.855 | +0.834 | +0.919 | +0.827 | +0.890 | +0.966 | +0.871 | +0.943 | +0.895 |

## 4. 副 endpoint（判定に使わない）

| 腕 | 量 | 窓 | 値 | 差 [95%] |
|---|---|---|---|---|
| ref | delta_pplus | 51-150 | -0.4878 |  |
| ref | delta_pplus | 2-10 | -0.3269 |  |
| ref | delta_pplus | 11-50 | -0.4702 |  |
| ref | delta_E1_l2 | 51-150 | -0.4882 |  |
| ref | delta_E1_l2 | 2-10 | -0.3219 |  |
| ref | delta_E1_l2 | 11-50 | -0.4719 |  |
| ref | delta_E2_l2 | 51-150 | -1.5525 |  |
| ref | delta_E2_l2 | 2-10 | -1.2134 |  |
| ref | delta_E2_l2 | 11-50 | -2.0155 |  |
| ref | delta_E1 | 51-150 | -0.7083 |  |
| ref | delta_E1 | 2-10 | -0.4654 |  |
| ref | delta_E1 | 11-50 | -0.6820 |  |
| ref | delta_E2 | 51-150 | -2.1235 |  |
| ref | delta_E2 | 2-10 | -1.0309 |  |
| ref | delta_E2 | 11-50 | -1.8958 |  |
| ref | absorb_frac_absorbed | 1-150 | +0.0160 |  |
| ref | absorb_median_absorb_task | 1-150 | +100.1111 |  |
| ref | absorb_n_censored | 1-150 | +98.4000 |  |
| ref | absorb_frac_neg_at_end | 1-150 | +0.0080 |  |
| ref | absorb_frac_ever_neg | 1-150 | +0.7690 |  |
| ref | absorb_mean_reentries | 1-150 | +1.8310 |  |
| ref | level_U_l1 | t1 | +3.8255 |  |
| ref | level_K_l1 | t1 | +3.1862 |  |
| ref | level_sigma_l1 | t1 | +1.2222 |  |
| ref | level_zbar_l1 | t1 | -0.0179 |  |
| ref | level_b1 | t1 | +0.0111 |  |
| ref | level_q_l1 | t1 | -0.0049 |  |
| ref | level_v_norm_l1 | t1 | +3.3894 |  |
| ref | level_eps_frac_l1 | t1 | +0.2106 |  |
| ref | level_U_l1 | t10 | +9.1925 |  |
| ref | level_K_l1 | t10 | +3.1679 |  |
| ref | level_sigma_l1 | t10 | +5.4139 |  |
| ref | level_zbar_l1 | t10 | -7.5795 |  |
| ref | level_b1 | t10 | -0.0685 |  |
| ref | level_q_l1 | t10 | -1.2660 |  |
| ref | level_v_norm_l1 | t10 | +11.0902 |  |
| ref | level_eps_frac_l1 | t10 | +0.2392 |  |
| ref | level_U_l1 | t50 | +18.7236 |  |
| ref | level_K_l1 | t50 | +3.3945 |  |
| ref | level_sigma_l1 | t50 | +15.4580 |  |
| ref | level_zbar_l1 | t50 | -32.9141 |  |
| ref | level_b1 | t50 | -0.9817 |  |
| ref | level_q_l1 | t50 | -5.3822 |  |
| ref | level_v_norm_l1 | t50 | +28.1310 |  |
| ref | level_eps_frac_l1 | t50 | +0.3645 |  |
| ref | level_U_l1 | t100 | +28.9070 |  |
| ref | level_K_l1 | t100 | +3.5081 |  |
| ref | level_sigma_l1 | t100 | +22.3388 |  |
| ref | level_zbar_l1 | t100 | -48.2630 |  |
| ref | level_b1 | t100 | -1.7985 |  |
| ref | level_q_l1 | t100 | -7.8315 |  |
| ref | level_v_norm_l1 | t100 | +38.3185 |  |
| ref | level_eps_frac_l1 | t100 | +0.4042 |  |
| ref | level_U_l1 | t150 | +33.2465 |  |
| ref | level_K_l1 | t150 | +3.5045 |  |
| ref | level_sigma_l1 | t150 | +26.7082 |  |
| ref | level_zbar_l1 | t150 | -58.9718 |  |
| ref | level_b1 | t150 | -2.5057 |  |
| ref | level_q_l1 | t150 | -9.5189 |  |
| ref | level_v_norm_l1 | t150 | +44.9631 |  |
| ref | level_eps_frac_l1 | t150 | +0.4202 |  |
| ref | ledger_adam_q_align | 51-150 | +63.4864 |  |
| ref | ledger_adam_q_sq | 51-150 | +2.1834 |  |
| ref | ledger_proj_q_align | 51-150 | +0.0000 |  |
| ref | ledger_proj_q_sq | 51-150 | +0.0000 |  |
| ref | ledger_adam_v2_align | 51-150 | +1230.5095 |  |
| ref | ledger_adam_v2_sq | 51-150 | +7.1520 |  |
| ref | ledger_proj_v2_align | 51-150 | +0.0000 |  |
| ref | ledger_proj_v2_sq | 51-150 | +0.0000 |  |
| ref | ledger_adam_m_align | 51-150 | +0.0121 |  |
| ref | ledger_adam_m_sq | 51-150 | +0.0017 |  |
| ref | ledger_proj_m_align | 51-150 | +0.0000 |  |
| ref | ledger_proj_m_sq | 51-150 | +0.0000 |  |
| ref | ledger_adam_wt2_align | 51-150 | +1284.5281 |  |
| ref | ledger_adam_wt2_sq | 51-150 | +8.0164 |  |
| ref | ledger_proj_wt2_align | 51-150 | +0.0000 |  |
| ref | ledger_proj_wt2_sq | 51-150 | +0.0000 |  |
| cap_par | delta_pplus | 51-150 | -0.0959 | +0.3920 [+0.3870, +0.3969] |
| cap_par | delta_pplus | 2-10 | +0.0132 | +0.3401 [+0.3304, +0.3499] |
| cap_par | delta_pplus | 11-50 | -0.0295 | +0.4407 [+0.4339, +0.4474] |
| cap_par | delta_E1_l2 | 51-150 | -0.4866 | +0.0016 [+0.0011, +0.0021] |
| cap_par | delta_E1_l2 | 2-10 | -0.4036 | -0.0817 [-0.0914, -0.0719] |
| cap_par | delta_E1_l2 | 11-50 | -0.4865 | -0.0146 [-0.0164, -0.0128] |
| cap_par | delta_E2_l2 | 51-150 | -1.9147 | -0.3622 [-0.4097, -0.3147] |
| cap_par | delta_E2_l2 | 2-10 | -1.5285 | -0.3152 [-0.3661, -0.2643] |
| cap_par | delta_E2_l2 | 11-50 | -2.0679 | -0.0524 [-0.1007, -0.0041] |
| cap_par | delta_E1 | 51-150 | -0.2810 | +0.4273 [+0.4223, +0.4323] |
| cap_par | delta_E1 | 2-10 | -0.0812 | +0.3842 [+0.3731, +0.3953] |
| cap_par | delta_E1 | 11-50 | -0.1842 | +0.4978 [+0.4909, +0.5047] |
| cap_par | delta_E2 | 51-150 | -0.2997 | +1.8238 [+1.8005, +1.8471] |
| cap_par | delta_E2 | 2-10 | +0.0073 | +1.0381 [+1.0039, +1.0723] |
| cap_par | delta_E2 | 11-50 | -0.1167 | +1.7792 [+1.7510, +1.8073] |
| cap_par | absorb_frac_absorbed | 1-150 | +0.0000 |  |
| cap_par | absorb_median_absorb_task | 1-150 | nan |  |
| cap_par | absorb_n_censored | 1-150 | +100.0000 |  |
| cap_par | absorb_frac_neg_at_end | 1-150 | +0.0000 |  |
| cap_par | absorb_frac_ever_neg | 1-150 | +0.0000 |  |
| cap_par | absorb_mean_reentries | 1-150 | +0.0000 |  |
| cap_par | level_U_l1 | t1 | +3.8255 |  |
| cap_par | level_K_l1 | t1 | +3.1862 |  |
| cap_par | level_sigma_l1 | t1 | +1.2222 |  |
| cap_par | level_zbar_l1 | t1 | -0.0179 |  |
| cap_par | level_b1 | t1 | +0.0111 |  |
| cap_par | level_q_l1 | t1 | -0.0049 |  |
| cap_par | level_v_norm_l1 | t1 | +3.3894 |  |
| cap_par | level_eps_frac_l1 | t1 | +0.2106 |  |
| cap_par | level_U_l1 | t10 | +10.8014 |  |
| cap_par | level_K_l1 | t10 | +2.9059 |  |
| cap_par | level_sigma_l1 | t10 | +3.8301 |  |
| cap_par | level_zbar_l1 | t10 | -0.0801 |  |
| cap_par | level_b1 | t10 | +0.0286 |  |
| cap_par | level_q_l1 | t10 | -0.0183 |  |
| cap_par | level_v_norm_l1 | t10 | +9.2633 |  |
| cap_par | level_eps_frac_l1 | t10 | +0.2131 |  |
| cap_par | level_U_l1 | t50 | +20.9771 |  |
| cap_par | level_K_l1 | t50 | +2.8686 |  |
| cap_par | level_sigma_l1 | t50 | +8.1728 |  |
| cap_par | level_zbar_l1 | t50 | -1.6744 |  |
| cap_par | level_b1 | t50 | -1.6403 |  |
| cap_par | level_q_l1 | t50 | -0.0058 |  |
| cap_par | level_v_norm_l1 | t50 | +23.1473 |  |
| cap_par | level_eps_frac_l1 | t50 | +0.2266 |  |
| cap_par | level_U_l1 | t100 | +28.7665 |  |
| cap_par | level_K_l1 | t100 | +2.7288 |  |
| cap_par | level_sigma_l1 | t100 | +12.5128 |  |
| cap_par | level_zbar_l1 | t100 | -4.0795 |  |
| cap_par | level_b1 | t100 | -4.0781 |  |
| cap_par | level_q_l1 | t100 | -0.0002 |  |
| cap_par | level_v_norm_l1 | t100 | +35.5596 |  |
| cap_par | level_eps_frac_l1 | t100 | +0.2385 |  |
| cap_par | level_U_l1 | t150 | +33.9745 |  |
| cap_par | level_K_l1 | t150 | +2.6456 |  |
| cap_par | level_sigma_l1 | t150 | +16.1682 |  |
| cap_par | level_zbar_l1 | t150 | -7.1393 |  |
| cap_par | level_b1 | t150 | -7.1661 |  |
| cap_par | level_q_l1 | t150 | +0.0045 |  |
| cap_par | level_v_norm_l1 | t150 | +44.9975 |  |
| cap_par | level_eps_frac_l1 | t150 | +0.2492 |  |
| cap_par | ledger_adam_q_align | 51-150 | +31.3251 |  |
| cap_par | ledger_adam_q_sq | 51-150 | +4.2580 |  |
| cap_par | ledger_proj_q_align | 51-150 | -38.0792 |  |
| cap_par | ledger_proj_q_sq | 51-150 | +2.4965 |  |
| cap_par | ledger_adam_v2_align | 51-150 | +1491.8594 |  |
| cap_par | ledger_adam_v2_sq | 51-150 | +11.8769 |  |
| cap_par | ledger_proj_v2_align | 51-150 | +0.0163 |  |
| cap_par | ledger_proj_v2_sq | 51-150 | +0.0000 |  |
| cap_par | ledger_adam_m_align | 51-150 | +0.1205 |  |
| cap_par | ledger_adam_m_sq | 51-150 | +0.0030 |  |
| cap_par | ledger_proj_m_align | 51-150 | -0.1139 |  |
| cap_par | ledger_proj_m_sq | 51-150 | +0.0012 |  |
| cap_par | ledger_adam_wt2_align | 51-150 | +1428.7160 |  |
| cap_par | ledger_adam_wt2_sq | 51-150 | +13.8096 |  |
| cap_par | ledger_proj_wt2_align | 51-150 | +51.2333 |  |
| cap_par | ledger_proj_wt2_sq | 51-150 | +1.5402 |  |
| cap_perp | delta_pplus | 51-150 | -0.3457 | +0.1422 [+0.1223, +0.1621] |
| cap_perp | delta_pplus | 2-10 | -0.3161 | +0.0107 [+0.0066, +0.0149] |
| cap_perp | delta_pplus | 11-50 | -0.3700 | +0.1002 [+0.0925, +0.1079] |
| cap_perp | delta_E1_l2 | 51-150 | -0.3234 | +0.1648 [+0.1406, +0.1891] |
| cap_perp | delta_E1_l2 | 2-10 | -0.2281 | +0.0938 [+0.0809, +0.1067] |
| cap_perp | delta_E1_l2 | 11-50 | -0.3357 | +0.1361 [+0.1095, +0.1627] |
| cap_perp | delta_E2_l2 | 51-150 | -1.3871 | +0.1654 [-0.0214, +0.3521] |
| cap_perp | delta_E2_l2 | 2-10 | -0.9180 | +0.2954 [+0.2317, +0.3590] |
| cap_perp | delta_E2_l2 | 11-50 | -1.5468 | +0.4687 [+0.3425, +0.5949] |
| cap_perp | delta_E1 | 51-150 | -0.4191 | +0.2891 [+0.2672, +0.3110] |
| cap_perp | delta_E1 | 2-10 | -0.3549 | +0.1105 [+0.1047, +0.1162] |
| cap_perp | delta_E1 | 11-50 | -0.4265 | +0.2554 [+0.2468, +0.2641] |
| cap_perp | delta_E2 | 51-150 | -1.1561 | +0.9673 [+0.8910, +1.0437] |
| cap_perp | delta_E2 | 2-10 | -0.9708 | +0.0601 [+0.0450, +0.0751] |
| cap_perp | delta_E2 | 11-50 | -1.2012 | +0.6946 [+0.6631, +0.7261] |
| cap_perp | absorb_frac_absorbed | 1-150 | +0.0000 |  |
| cap_perp | absorb_median_absorb_task | 1-150 | nan |  |
| cap_perp | absorb_n_censored | 1-150 | +100.0000 |  |
| cap_perp | absorb_frac_neg_at_end | 1-150 | +0.0140 |  |
| cap_perp | absorb_frac_ever_neg | 1-150 | +0.6310 |  |
| cap_perp | absorb_mean_reentries | 1-150 | +1.5290 |  |
| cap_perp | level_U_l1 | t1 | +3.8255 |  |
| cap_perp | level_K_l1 | t1 | +3.1862 |  |
| cap_perp | level_sigma_l1 | t1 | +1.2222 |  |
| cap_perp | level_zbar_l1 | t1 | -0.0179 |  |
| cap_perp | level_b1 | t1 | +0.0111 |  |
| cap_perp | level_q_l1 | t1 | -0.0049 |  |
| cap_perp | level_v_norm_l1 | t1 | +3.3894 |  |
| cap_perp | level_eps_frac_l1 | t1 | +0.2106 |  |
| cap_perp | level_U_l1 | t10 | +3.0804 |  |
| cap_perp | level_K_l1 | t10 | +3.0696 |  |
| cap_perp | level_sigma_l1 | t10 | +1.7205 |  |
| cap_perp | level_zbar_l1 | t10 | -2.1050 |  |
| cap_perp | level_b1 | t10 | +0.0853 |  |
| cap_perp | level_q_l1 | t10 | -0.3692 |  |
| cap_perp | level_v_norm_l1 | t10 | +3.3888 |  |
| cap_perp | level_eps_frac_l1 | t10 | +0.2112 |  |
| cap_perp | level_U_l1 | t50 | +2.9190 |  |
| cap_perp | level_K_l1 | t50 | +3.0585 |  |
| cap_perp | level_sigma_l1 | t50 | +1.7230 |  |
| cap_perp | level_zbar_l1 | t50 | -2.2412 |  |
| cap_perp | level_b1 | t50 | +0.2471 |  |
| cap_perp | level_q_l1 | t50 | -0.4195 |  |
| cap_perp | level_v_norm_l1 | t50 | +3.3885 |  |
| cap_perp | level_eps_frac_l1 | t50 | +0.2114 |  |
| cap_perp | level_U_l1 | t100 | +2.9761 |  |
| cap_perp | level_K_l1 | t100 | +3.0974 |  |
| cap_perp | level_sigma_l1 | t100 | +1.7365 |  |
| cap_perp | level_zbar_l1 | t100 | -2.2674 |  |
| cap_perp | level_b1 | t100 | +0.3818 |  |
| cap_perp | level_q_l1 | t100 | -0.4466 |  |
| cap_perp | level_v_norm_l1 | t100 | +3.3885 |  |
| cap_perp | level_eps_frac_l1 | t100 | +0.2112 |  |
| cap_perp | level_U_l1 | t150 | +3.0729 |  |
| cap_perp | level_K_l1 | t150 | +3.1000 |  |
| cap_perp | level_sigma_l1 | t150 | +1.7284 |  |
| cap_perp | level_zbar_l1 | t150 | -2.1451 |  |
| cap_perp | level_b1 | t150 | +0.4920 |  |
| cap_perp | level_q_l1 | t150 | -0.4445 |  |
| cap_perp | level_v_norm_l1 | t150 | +3.3887 |  |
| cap_perp | level_eps_frac_l1 | t150 | +0.2115 |  |
| cap_perp | ledger_adam_q_align | 51-150 | -2.9788 |  |
| cap_perp | ledger_adam_q_sq | 51-150 | +3.0262 |  |
| cap_perp | ledger_proj_q_align | 51-150 | +0.0000 |  |
| cap_perp | ledger_proj_q_sq | 51-150 | +0.0000 |  |
| cap_perp | ledger_adam_v2_align | 51-150 | +831.7284 |  |
| cap_perp | ledger_adam_v2_sq | 51-150 | +11.3192 |  |
| cap_perp | ledger_proj_v2_align | 51-150 | -843.1507 |  |
| cap_perp | ledger_proj_v2_sq | 51-150 | +0.1038 |  |
| cap_perp | ledger_adam_m_align | 51-150 | +0.0027 |  |
| cap_perp | ledger_adam_m_sq | 51-150 | +0.0022 |  |
| cap_perp | ledger_proj_m_align | 51-150 | -0.0049 |  |
| cap_perp | ledger_proj_m_sq | 51-150 | +0.0000 |  |
| cap_perp | ledger_adam_wt2_align | 51-150 | +826.6278 |  |
| cap_perp | ledger_adam_wt2_sq | 51-150 | +12.6072 |  |
| cap_perp | ledger_proj_wt2_align | 51-150 | -839.2724 |  |
| cap_perp | ledger_proj_wt2_sq | 51-150 | +0.1032 |  |
| cap_both | delta_pplus | 51-150 | -0.3788 | +0.1090 [+0.1042, +0.1139] |
| cap_both | delta_pplus | 2-10 | -0.0038 | +0.3231 [+0.3106, +0.3355] |
| cap_both | delta_pplus | 11-50 | -0.1882 | +0.2820 [+0.2680, +0.2959] |
| cap_both | delta_E1_l2 | 51-150 | -0.4113 | +0.0769 [+0.0673, +0.0864] |
| cap_both | delta_E1_l2 | 2-10 | -0.3672 | -0.0453 [-0.0541, -0.0366] |
| cap_both | delta_E1_l2 | 11-50 | -0.4602 | +0.0116 [+0.0070, +0.0163] |
| cap_both | delta_E2_l2 | 51-150 | -1.6878 | -0.1353 [-0.2300, -0.0406] |
| cap_both | delta_E2_l2 | 2-10 | -1.2952 | -0.0818 [-0.1281, -0.0355] |
| cap_both | delta_E2_l2 | 11-50 | -1.8675 | +0.1480 [+0.0972, +0.1988] |
| cap_both | delta_E1 | 51-150 | -0.4133 | +0.2950 [+0.2880, +0.3020] |
| cap_both | delta_E1 | 2-10 | -0.0273 | +0.4381 [+0.4248, +0.4514] |
| cap_both | delta_E1 | 11-50 | -0.2078 | +0.4741 [+0.4612, +0.4871] |
| cap_both | delta_E2 | 51-150 | -1.2332 | +0.8903 [+0.8581, +0.9225] |
| cap_both | delta_E2 | 2-10 | -0.0388 | +0.9920 [+0.9533, +1.0307] |
| cap_both | delta_E2 | 11-50 | -0.5814 | +1.3144 [+1.2659, +1.3629] |
| cap_both | absorb_frac_absorbed | 1-150 | +0.0000 |  |
| cap_both | absorb_median_absorb_task | 1-150 | nan |  |
| cap_both | absorb_n_censored | 1-150 | +100.0000 |  |
| cap_both | absorb_frac_neg_at_end | 1-150 | +0.0000 |  |
| cap_both | absorb_frac_ever_neg | 1-150 | +0.0090 |  |
| cap_both | absorb_mean_reentries | 1-150 | +0.0130 |  |
| cap_both | level_U_l1 | t1 | +3.8255 |  |
| cap_both | level_K_l1 | t1 | +3.1862 |  |
| cap_both | level_sigma_l1 | t1 | +1.2222 |  |
| cap_both | level_zbar_l1 | t1 | -0.0179 |  |
| cap_both | level_b1 | t1 | +0.0111 |  |
| cap_both | level_q_l1 | t1 | -0.0049 |  |
| cap_both | level_v_norm_l1 | t1 | +3.3894 |  |
| cap_both | level_eps_frac_l1 | t1 | +0.2106 |  |
| cap_both | level_U_l1 | t10 | +4.0120 |  |
| cap_both | level_K_l1 | t10 | +2.8309 |  |
| cap_both | level_sigma_l1 | t10 | +1.5246 |  |
| cap_both | level_zbar_l1 | t10 | -0.1997 |  |
| cap_both | level_b1 | t10 | -0.0750 |  |
| cap_both | level_q_l1 | t10 | -0.0210 |  |
| cap_both | level_v_norm_l1 | t10 | +3.3892 |  |
| cap_both | level_eps_frac_l1 | t10 | +0.2106 |  |
| cap_both | level_U_l1 | t50 | +3.1285 |  |
| cap_both | level_K_l1 | t50 | +3.0730 |  |
| cap_both | level_sigma_l1 | t50 | +1.4830 |  |
| cap_both | level_zbar_l1 | t50 | -1.3576 |  |
| cap_both | level_b1 | t50 | -1.3975 |  |
| cap_both | level_q_l1 | t50 | +0.0067 |  |
| cap_both | level_v_norm_l1 | t50 | +3.3892 |  |
| cap_both | level_eps_frac_l1 | t50 | +0.2107 |  |
| cap_both | level_U_l1 | t100 | +2.8322 |  |
| cap_both | level_K_l1 | t100 | +3.2151 |  |
| cap_both | level_sigma_l1 | t100 | +1.4795 |  |
| cap_both | level_zbar_l1 | t100 | -1.8761 |  |
| cap_both | level_b1 | t100 | -1.9122 |  |
| cap_both | level_q_l1 | t100 | +0.0061 |  |
| cap_both | level_v_norm_l1 | t100 | +3.3889 |  |
| cap_both | level_eps_frac_l1 | t100 | +0.2107 |  |
| cap_both | level_U_l1 | t150 | +2.9047 |  |
| cap_both | level_K_l1 | t150 | +3.2238 |  |
| cap_both | level_sigma_l1 | t150 | +1.4944 |  |
| cap_both | level_zbar_l1 | t150 | -1.8620 |  |
| cap_both | level_b1 | t150 | -1.9334 |  |
| cap_both | level_q_l1 | t150 | +0.0120 |  |
| cap_both | level_v_norm_l1 | t150 | +3.3891 |  |
| cap_both | level_eps_frac_l1 | t150 | +0.2108 |  |
| cap_both | ledger_adam_q_align | 51-150 | +18.5908 |  |
| cap_both | ledger_adam_q_sq | 51-150 | +3.8235 |  |
| cap_both | ledger_proj_q_align | 51-150 | -24.3729 |  |
| cap_both | ledger_proj_q_sq | 51-150 | +1.9588 |  |
| cap_both | ledger_adam_v2_align | 51-150 | +618.9972 |  |
| cap_both | ledger_adam_v2_sq | 51-150 | +11.1626 |  |
| cap_both | ledger_proj_v2_align | 51-150 | -630.2180 |  |
| cap_both | ledger_proj_v2_sq | 51-150 | +0.0577 |  |
| cap_both | ledger_adam_m_align | 51-150 | +0.0255 |  |
| cap_both | ledger_adam_m_sq | 51-150 | +0.0027 |  |
| cap_both | ledger_proj_m_align | 51-150 | -0.0291 |  |
| cap_both | ledger_proj_m_sq | 51-150 | +0.0010 |  |
| cap_both | ledger_adam_wt2_align | 51-150 | +617.6335 |  |
| cap_both | ledger_adam_wt2_sq | 51-150 | +12.8621 |  |
| cap_both | ledger_proj_wt2_align | 51-150 | -631.7893 |  |
| cap_both | ledger_proj_wt2_sq | 51-150 | +1.2611 |  |

## 5. 読み方（spec §5.4 のまま）

- UNRESOLVED は効果なしではない。単独腕が両方 0 でも「両方必要」とは書かない。
- $U$ の水準差は「上端が残った」の証拠にしない。吸収は §4 の absorb_* だけで読む。
- cap_par は平均画像チャンネルを初期化の大きさに閉じ込める操作として読む。
- IMPAIRED の腕では「可塑性を守った」と書かない。
