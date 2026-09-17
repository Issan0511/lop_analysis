# l2cap_ee_0917 — 判定（第2層の行ノルム上限・ELU→ELU・100 タスク）

> 自動生成: `analysis/l2cap_ee_0917/verdict.py`。spec: `specs/spec_l2cap_ee_0917.md`。数値は `verdict.csv`・`paired.csv`・`timing.csv`・`secondary.csv` から。

## 0. 実行したもの

- 集計時の HEAD: `866c0458bd97f8754f4d01cee55134b699ee2a8f`・run の commit: ['866c0458bd97f8754f4d01cee55134b699ee2a8f']
- shard: 50/40・欠損 0・有効 seed 10（無効: なし）

## 1. 適用条件（spec §5.2）

- (A) 有効 seed 10 ≥ 8: **True**
- (B) ref が主窓で床: 10/10 seed（要 ≥ 8）、ref の第2層 ΔE2 -0.7762 [-0.7975, -0.7549] → **True**
- (C) 上限が task 2–10 の全タスクで書いた: cap1 **True**、cap2 **True**、cap12 **True**、cap12_bfix **True**
- IMPAIRED（読みのフラグ）: cap1 0/10 seed → **False**、cap2 0/10 seed → **False**、cap12 0/10 seed → **False**、cap12_bfix 0/10 seed → **False**
- 主窓の床の状態: ref FLOORED、cap1 FLOORED、cap2 SPLIT、cap12 ALIVE、cap12_bfix ALIVE

## 2. 判定

| 比較 | 水準 | ラベル | 時間（T_half） |
|---|---|---|---|
| **主: cap12 − ref** | 97.5% | **RESCUED** | LATER |
| cap12 − ref（参考） | 95% | RESCUED | |
| cap1 − ref | 95% | COLLAPSED | LATER |
| cap2 − ref | 95% | SPLIT | LATER |
| cap12_bfix − ref | 95% | RESCUED | LATER |

## 3. 腕間差（主窓 task 51–100・seed 内対応差）

| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |
|---|---|---|---|---|---|---|
| cap1 | E1 | 0.975 | +0.0007 | +0.0009 | [-0.0000, +0.0015] | 0 |
| cap1 | E2 | 0.975 | +0.0000 | +0.0001 | [-0.0000, +0.0001] | 0 |
| cap1 | E1 | 0.95 | +0.0007 | +0.0009 | [+0.0001, +0.0014] | + |
| cap1 | E2 | 0.95 | +0.0000 | +0.0001 | [+0.0000, +0.0001] | + |
| cap2 | E1 | 0.975 | +0.0097 | +0.0073 | [+0.0035, +0.0159] | + |
| cap2 | E2 | 0.975 | +0.0010 | +0.0008 | [+0.0003, +0.0016] | + |
| cap2 | E1 | 0.95 | +0.0097 | +0.0073 | [+0.0045, +0.0149] | + |
| cap2 | E2 | 0.95 | +0.0010 | +0.0008 | [+0.0004, +0.0015] | + |
| cap12 | E1 | 0.975 | +0.7347 | +0.0080 | [+0.7279, +0.7415] | + |
| cap12 | E2 | 0.975 | +0.2104 | +0.0118 | [+0.2004, +0.2204] | + |
| cap12 | E1 | 0.95 | +0.7347 | +0.0080 | [+0.7290, +0.7404] | + |
| cap12 | E2 | 0.95 | +0.2104 | +0.0118 | [+0.2020, +0.2188] | + |
| cap12_bfix | E1 | 0.975 | +0.7476 | +0.0060 | [+0.7425, +0.7527] | + |
| cap12_bfix | E2 | 0.975 | +0.4528 | +0.0127 | [+0.4421, +0.4636] | + |
| cap12_bfix | E1 | 0.95 | +0.7476 | +0.0060 | [+0.7433, +0.7519] | + |
| cap12_bfix | E2 | 0.95 | +0.4528 | +0.0127 | [+0.4438, +0.4619] | + |

交互作用 δ_cap12 − δ_cap1 − δ_cap2（95%）: E1 +0.7243 [+0.7165, +0.7320]、E2 +0.2094 [+0.2009, +0.2178]

輸送（主窓の第2層 Δz̄₂ と Δb₂ の ref との差・95%）: cap1|zbar_l2 +76.270 [+63.751, +88.789]、cap1|b2 -0.476 [-0.520, -0.432]、cap2|zbar_l2 +9.470 [-5.559, +24.499]、cap2|b2 -0.187 [-0.213, -0.161]、cap12|zbar_l2 +149.585 [+137.616, +161.554]、cap12|b2 -0.980 [-1.041, -0.919]、cap12_bfix|zbar_l2 +151.103 [+139.150, +163.055]、cap12_bfix|b2 +0.164 [+0.140, +0.187]

### 3.1 seed 別の差と床

| 腕 | endpoint | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cap1 | E1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 | +0.001 | +0.002 | +0.002 | +0.000 |
| cap1 | E2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cap2 | E1 | +0.002 | +0.010 | +0.005 | +0.016 | +0.005 | +0.014 | +0.024 | +0.000 | +0.010 | +0.011 |
| cap2 | E2 | +0.000 | +0.001 | +0.000 | +0.002 | +0.000 | +0.001 | +0.003 | +0.000 | +0.001 | +0.001 |
| cap12 | E1 | +0.735 | +0.731 | +0.721 | +0.725 | +0.736 | +0.747 | +0.730 | +0.738 | +0.743 | +0.739 |
| cap12 | E2 | +0.217 | +0.202 | +0.189 | +0.207 | +0.206 | +0.227 | +0.202 | +0.212 | +0.225 | +0.217 |
| cap12_bfix | E1 | +0.745 | +0.750 | +0.736 | +0.749 | +0.754 | +0.758 | +0.743 | +0.746 | +0.749 | +0.747 |
| cap12_bfix | E2 | +0.450 | +0.437 | +0.451 | +0.449 | +0.440 | +0.464 | +0.470 | +0.456 | +0.438 | +0.472 |
| ref | 主窓 online（閾値） | 0.100 (0.116) 床 | 0.100 (0.116) 床 | 0.101 (0.116) 床 | 0.100 (0.117) 床 | 0.099 (0.114) 床 | 0.101 (0.118) 床 | 0.100 (0.115) 床 | 0.100 (0.116) 床 | 0.100 (0.116) 床 | 0.100 (0.116) 床 |
| cap1 | 主窓 online（閾値） | 0.100 (0.116) 床 | 0.101 (0.116) 床 | 0.101 (0.116) 床 | 0.100 (0.117) 床 | 0.100 (0.114) 床 | 0.101 (0.118) 床 | 0.101 (0.115) 床 | 0.102 (0.116) 床 | 0.102 (0.116) 床 | 0.100 (0.116) 床 |
| cap2 | 主窓 online（閾値） | 0.102 (0.116) 床 | 0.111 (0.116) 床 | 0.105 (0.116) 床 | 0.116 (0.117) 床 | 0.104 (0.114) 床 | 0.115 (0.118) 床 | 0.124 (0.115) | 0.100 (0.116) 床 | 0.110 (0.116) 床 | 0.111 (0.116) 床 |
| cap12 | 主窓 online（閾値） | 0.835 (0.116) | 0.831 (0.116) | 0.822 (0.116) | 0.825 (0.117) | 0.835 (0.114) | 0.848 (0.118) | 0.830 (0.115) | 0.838 (0.116) | 0.844 (0.116) | 0.839 (0.116) |
| cap12_bfix | 主窓 online（閾値） | 0.845 (0.116) | 0.850 (0.116) | 0.837 (0.116) | 0.850 (0.117) | 0.852 (0.114) | 0.859 (0.118) | 0.842 (0.115) | 0.846 (0.116) | 0.849 (0.116) | 0.847 (0.116) |

### 3.2 T_half（適合率 F が初めて 1/2 を下回るタスク・— は 100 まで未到達）

| 腕 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 早/遅/同 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | 7 | 7 | 7 | 7 | 7 | 5 | 8 | 7 | 7 | 8 | | |
| cap1 | 14 | 15 | 14 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 0/10/0 | 0.00195 |
| cap2 | 11 | 12 | 11 | 11 | 11 | 8 | 13 | 13 | 11 | 11 | 0/10/0 | 0.00195 |
| cap12 | — | — | — | — | — | — | — | — | — | — | 0/10/0 | 0.00195 |
| cap12_bfix | — | — | — | — | — | — | — | — | — | — | 0/10/0 | 0.00195 |

### 3.3 第2層のユニット平均 z̄₂ が初めて ln 2^−24 = -16.64 を下回るタスク（報告のみ・— は未到達）

| 腕 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ref | 6 | 7 | 7 | 7 | 7 | 5 | 8 | 7 | 7 | 7 |
| cap1 | 13 | 15 | 14 | 14 | 16 | 13 | 16 | 16 | 17 | 14 |
| cap2 | 11 | 12 | 11 | 12 | 11 | 10 | 14 | 13 | 11 | 12 |
| cap12 | — | — | — | — | — | — | — | — | — | — |
| cap12_bfix | — | — | — | — | — | — | — | — | — | — |

## 4. 副 endpoint（判定に使わない）

| 腕 | 量 | 窓 | 値 | 差 [95%] |
|---|---|---|---|---|
| ref | delta_E1 | 51-100 | -0.6965 |  |
| ref | delta_E1 | 2-10 | -0.2979 |  |
| ref | delta_E1 | 11-50 | -0.6915 |  |
| ref | delta_E2 | 51-100 | -0.7762 |  |
| ref | delta_E2 | 2-10 | -0.5769 |  |
| ref | delta_E2 | 11-50 | -0.7759 |  |
| ref | level_g2_exp | 51-100 | +0.0000 |  |
| ref | level_g2_exp | 2-10 | +0.1994 |  |
| ref | level_g2_exp | 11-50 | +0.0004 |  |
| ref | level_dtrain0_l2 | 51-100 | +1.0000 |  |
| ref | level_dtrain0_l2 | 2-10 | +0.4221 |  |
| ref | level_dtrain0_l2 | 11-50 | +0.9971 |  |
| ref | level_dtrain_l1 | 51-100 | +0.3030 |  |
| ref | level_dtrain_l1 | 2-10 | +0.4103 |  |
| ref | level_dtrain_l1 | 11-50 | +0.3052 |  |
| ref | level_g1 | 51-100 | +0.3030 |  |
| ref | level_g1 | 2-10 | +0.4103 |  |
| ref | level_g1 | 11-50 | +0.3052 |  |
| ref | level_pplus_l2 | 51-100 | +0.0000 |  |
| ref | level_pplus_l2 | 2-10 | +0.1482 |  |
| ref | level_pplus_l2 | 11-50 | +0.0003 |  |
| ref | level_zbar_l2 | 51-100 | -152.7370 |  |
| ref | level_zbar_l2 | 2-10 | -19.4753 |  |
| ref | level_zbar_l2 | 11-50 | -125.0749 |  |
| ref | level_d_l2 | 51-100 | -3.3850 |  |
| ref | level_d_l2 | 2-10 | -1.4724 |  |
| ref | level_d_l2 | 11-50 | -3.2749 |  |
| ref | level_sigma_l2 | 51-100 | +45.2695 |  |
| ref | level_sigma_l2 | 2-10 | +9.6162 |  |
| ref | level_sigma_l2 | 11-50 | +37.9524 |  |
| ref | level_d_l1 | 51-100 | -0.6797 |  |
| ref | level_d_l1 | 2-10 | -0.4988 |  |
| ref | level_d_l1 | 11-50 | -0.6743 |  |
| ref | level_sigma_l1 | 51-100 | +8.0867 |  |
| ref | level_sigma_l1 | 2-10 | +4.0655 |  |
| ref | level_sigma_l1 | 11-50 | +7.5186 |  |
| ref | level_v_norm_l1 | 51-100 | +12.3488 |  |
| ref | level_v_norm_l1 | 2-10 | +7.9785 |  |
| ref | level_v_norm_l1 | 11-50 | +11.9108 |  |
| ref | level_q_l1 | 51-100 | -0.2734 |  |
| ref | level_q_l1 | 2-10 | -0.3228 |  |
| ref | level_q_l1 | 11-50 | -0.3953 |  |
| ref | level_b2 | 51-100 | -0.1273 |  |
| ref | level_b2 | 2-10 | -0.0485 |  |
| ref | level_b2 | 11-50 | -0.1219 |  |
| ref | level_row_norm_l2 | 51-100 | +4.1105 |  |
| ref | level_row_norm_l2 | 2-10 | +3.1332 |  |
| ref | level_row_norm_l2 | 11-50 | +4.0767 |  |
| ref | level_frac_q_pos | 51-100 | +0.2339 |  |
| ref | level_frac_always_on | 51-100 | +0.1228 |  |
| ref | level_mu2sq_share_q_pos | 51-100 | +0.9986 |  |
| ref | level_mu2sq_share_always_on | 51-100 | +0.9720 |  |
| ref | level_frac_q_pos | 2-10 | +0.2013 |  |
| ref | level_frac_always_on | 2-10 | +0.0247 |  |
| ref | level_mu2sq_share_q_pos | 2-10 | +0.7991 |  |
| ref | level_mu2sq_share_always_on | 2-10 | +0.3472 |  |
| ref | level_frac_q_pos | 11-50 | +0.2239 |  |
| ref | level_frac_always_on | 11-50 | +0.1132 |  |
| ref | level_mu2sq_share_q_pos | 11-50 | +0.9979 |  |
| ref | level_mu2sq_share_always_on | 11-50 | +0.9628 |  |
| ref | level_q2mu2 | 51-100 | -152.6096 |  |
| ref | level_q2mu2 | 2-10 | -19.4268 |  |
| ref | level_q2mu2 | 11-50 | -124.9530 |  |
| ref | level_mu2_norm | 51-100 | +202.5087 |  |
| ref | level_mu2_norm | 2-10 | +31.3179 |  |
| ref | level_mu2_norm | 11-50 | +166.0628 |  |
| ref | level_mu2_proj_sd | 51-100 | +59.9310 |  |
| ref | level_mu2_proj_sd | 2-10 | +10.1299 |  |
| ref | level_mu2_proj_sd | 11-50 | +49.8351 |  |
| ref | level_s2_mean | 51-100 | +658.6383 |  |
| ref | level_s2_mean | 2-10 | +98.6793 |  |
| ref | level_s2_mean | 11-50 | +530.0209 |  |
| ref | level_s2_sd | 51-100 | +201.1650 |  |
| ref | level_s2_sd | 2-10 | +38.3611 |  |
| ref | level_s2_sd | 11-50 | +165.1132 |  |
| ref | level_memo_acc | 51-100 | +0.1062 |  |
| ref | level_memo_acc | 2-10 | +0.7471 |  |
| ref | level_memo_acc | 11-50 | +0.1169 |  |
| ref | level_online_acc | 51-100 | +0.1001 |  |
| ref | level_online_acc | 2-10 | +0.4987 |  |
| ref | level_online_acc | 11-50 | +0.1051 |  |
| ref | absorb_l1_frac_absorbed | 1-100 | +0.0120 |  |
| ref | absorb_l1_median_absorb_task | 1-100 | +31.2857 |  |
| ref | absorb_l1_frac_neg_at_end | 1-100 | +0.0070 |  |
| ref | absorb_l1_frac_ever_neg | 1-100 | +0.0820 |  |
| ref | absorb_l2_frac_absorbed | 1-100 | +1.0000 |  |
| ref | absorb_l2_median_absorb_task | 1-100 | +11.9500 |  |
| ref | absorb_l2_frac_neg_at_end | 1-100 | +1.0000 |  |
| ref | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| ref | t_half_median_observed | 2-100 | +7.0000 |  |
| ref | seeds_at_floor | 51-100 | +10.0000 |  |
| cap1 | delta_E1 | 51-100 | -0.6957 | +0.0007 [+0.0001, +0.0014] (10/10 正) |
| cap1 | delta_E1 | 2-10 | -0.0148 | +0.2831 [+0.2594, +0.3068] (10/10 正) |
| cap1 | delta_E1 | 11-50 | -0.6000 | +0.0915 [+0.0777, +0.1053] (10/10 正) |
| cap1 | delta_E2 | 51-100 | -0.7762 | +0.0000 [+0.0000, +0.0001] (8/10 正) |
| cap1 | delta_E2 | 2-10 | -0.5257 | +0.0511 [+0.0338, +0.0684] (10/10 正) |
| cap1 | delta_E2 | 11-50 | -0.7637 | +0.0122 [+0.0101, +0.0142] (10/10 正) |
| cap1 | level_g2_exp | 51-100 | +0.0000 |  |
| cap1 | level_g2_exp | 2-10 | +0.2505 |  |
| cap1 | level_g2_exp | 11-50 | +0.0125 |  |
| cap1 | level_dtrain0_l2 | 51-100 | +0.9996 |  |
| cap1 | level_dtrain0_l2 | 2-10 | +0.0683 |  |
| cap1 | level_dtrain0_l2 | 11-50 | +0.8612 |  |
| cap1 | level_dtrain_l1 | 51-100 | +0.5508 |  |
| cap1 | level_dtrain_l1 | 2-10 | +0.7128 |  |
| cap1 | level_dtrain_l1 | 11-50 | +0.6301 |  |
| cap1 | level_g1 | 51-100 | +0.5508 |  |
| cap1 | level_g1 | 2-10 | +0.7128 |  |
| cap1 | level_g1 | 11-50 | +0.6301 |  |
| cap1 | level_pplus_l2 | 51-100 | +0.0000 |  |
| cap1 | level_pplus_l2 | 2-10 | +0.1820 |  |
| cap1 | level_pplus_l2 | 11-50 | +0.0099 |  |
| cap1 | level_zbar_l2 | 51-100 | -76.4672 |  |
| cap1 | level_zbar_l2 | 2-10 | -5.6594 |  |
| cap1 | level_zbar_l2 | 11-50 | -47.2479 |  |
| cap1 | level_d_l2 | 51-100 | -5.6004 |  |
| cap1 | level_d_l2 | 2-10 | -0.9652 |  |
| cap1 | level_d_l2 | 11-50 | -3.3973 |  |
| cap1 | level_sigma_l2 | 51-100 | +14.3526 |  |
| cap1 | level_sigma_l2 | 2-10 | +5.2959 |  |
| cap1 | level_sigma_l2 | 11-50 | +13.9031 |  |
| cap1 | level_d_l1 | 51-100 | -0.0384 |  |
| cap1 | level_d_l1 | 2-10 | +0.0105 |  |
| cap1 | level_d_l1 | 11-50 | +0.0513 |  |
| cap1 | level_sigma_l1 | 51-100 | +2.0173 |  |
| cap1 | level_sigma_l1 | 2-10 | +1.4611 |  |
| cap1 | level_sigma_l1 | 11-50 | +1.7314 |  |
| cap1 | level_v_norm_l1 | 51-100 | +3.3537 |  |
| cap1 | level_v_norm_l1 | 2-10 | +3.4075 |  |
| cap1 | level_v_norm_l1 | 11-50 | +3.3942 |  |
| cap1 | level_q_l1 | 51-100 | -0.0165 |  |
| cap1 | level_q_l1 | 2-10 | +0.0021 |  |
| cap1 | level_q_l1 | 11-50 | -0.0130 |  |
| cap1 | level_b2 | 51-100 | -0.6032 |  |
| cap1 | level_b2 | 2-10 | -0.2792 |  |
| cap1 | level_b2 | 11-50 | -0.5678 |  |
| cap1 | level_row_norm_l2 | 51-100 | +6.9822 |  |
| cap1 | level_row_norm_l2 | 2-10 | +3.7731 |  |
| cap1 | level_row_norm_l2 | 11-50 | +6.7392 |  |
| cap1 | level_frac_q_pos | 51-100 | +0.4583 |  |
| cap1 | level_frac_always_on | 51-100 | +0.0965 |  |
| cap1 | level_mu2sq_share_q_pos | 51-100 | +0.7727 |  |
| cap1 | level_mu2sq_share_always_on | 51-100 | +0.4763 |  |
| cap1 | level_frac_q_pos | 2-10 | +0.5406 |  |
| cap1 | level_frac_always_on | 2-10 | +0.0000 |  |
| cap1 | level_mu2sq_share_q_pos | 2-10 | +0.7142 |  |
| cap1 | level_mu2sq_share_always_on | 2-10 | +0.0000 |  |
| cap1 | level_frac_q_pos | 11-50 | +0.4478 |  |
| cap1 | level_frac_always_on | 11-50 | +0.0295 |  |
| cap1 | level_mu2sq_share_q_pos | 11-50 | +0.6597 |  |
| cap1 | level_mu2sq_share_always_on | 11-50 | +0.1447 |  |
| cap1 | level_q2mu2 | 51-100 | -75.8640 |  |
| cap1 | level_q2mu2 | 2-10 | -5.3802 |  |
| cap1 | level_q2mu2 | 11-50 | -46.6801 |  |
| cap1 | level_mu2_norm | 51-100 | +29.5145 |  |
| cap1 | level_mu2_norm | 2-10 | +5.0969 |  |
| cap1 | level_mu2_norm | 11-50 | +18.2306 |  |
| cap1 | level_mu2_proj_sd | 51-100 | +5.2413 |  |
| cap1 | level_mu2_proj_sd | 2-10 | +2.2409 |  |
| cap1 | level_mu2_proj_sd | 11-50 | +4.8745 |  |
| cap1 | level_s2_mean | 51-100 | +141.6686 |  |
| cap1 | level_s2_mean | 2-10 | +32.4279 |  |
| cap1 | level_s2_mean | 11-50 | +90.3229 |  |
| cap1 | level_s2_sd | 51-100 | +33.2153 |  |
| cap1 | level_s2_sd | 2-10 | +14.5697 |  |
| cap1 | level_s2_sd | 11-50 | +26.1767 |  |
| cap1 | level_memo_acc | 51-100 | +0.1077 |  |
| cap1 | level_memo_acc | 2-10 | +0.9892 |  |
| cap1 | level_memo_acc | 11-50 | +0.2872 |  |
| cap1 | level_online_acc | 51-100 | +0.1009 |  |
| cap1 | level_online_acc | 2-10 | +0.7818 |  |
| cap1 | level_online_acc | 11-50 | +0.1966 |  |
| cap1 | absorb_l1_frac_absorbed | 1-100 | +0.0150 |  |
| cap1 | absorb_l1_median_absorb_task | 1-100 | +75.6667 |  |
| cap1 | absorb_l1_frac_neg_at_end | 1-100 | +0.0150 |  |
| cap1 | absorb_l1_frac_ever_neg | 1-100 | +0.0970 |  |
| cap1 | absorb_l2_frac_absorbed | 1-100 | +1.0000 |  |
| cap1 | absorb_l2_median_absorb_task | 1-100 | +29.4500 |  |
| cap1 | absorb_l2_frac_neg_at_end | 1-100 | +0.9970 |  |
| cap1 | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| cap1 | t_half_median_observed | 2-100 | +14.5000 |  |
| cap1 | seeds_at_floor | 51-100 | +10.0000 |  |
| cap2 | delta_E1 | 51-100 | -0.6868 | +0.0097 [+0.0045, +0.0149] (10/10 正) |
| cap2 | delta_E1 | 2-10 | -0.1058 | +0.1921 [+0.1802, +0.2040] (10/10 正) |
| cap2 | delta_E1 | 11-50 | -0.6206 | +0.0709 [+0.0633, +0.0784] (10/10 正) |
| cap2 | delta_E2 | 51-100 | -0.7753 | +0.0010 [+0.0004, +0.0015] (10/10 正) |
| cap2 | delta_E2 | 2-10 | -0.4453 | +0.1315 [+0.1193, +0.1438] (10/10 正) |
| cap2 | delta_E2 | 11-50 | -0.7644 | +0.0115 [+0.0101, +0.0128] (10/10 正) |
| cap2 | level_g2_exp | 51-100 | +0.0010 |  |
| cap2 | level_g2_exp | 2-10 | +0.3309 |  |
| cap2 | level_g2_exp | 11-50 | +0.0118 |  |
| cap2 | level_dtrain0_l2 | 51-100 | +0.9906 |  |
| cap2 | level_dtrain0_l2 | 2-10 | +0.0958 |  |
| cap2 | level_dtrain0_l2 | 11-50 | +0.8703 |  |
| cap2 | level_dtrain_l1 | 51-100 | +0.3734 |  |
| cap2 | level_dtrain_l1 | 2-10 | +0.4715 |  |
| cap2 | level_dtrain_l1 | 11-50 | +0.4847 |  |
| cap2 | level_g1 | 51-100 | +0.3734 |  |
| cap2 | level_g1 | 2-10 | +0.4715 |  |
| cap2 | level_g1 | 11-50 | +0.4847 |  |
| cap2 | level_pplus_l2 | 51-100 | +0.0008 |  |
| cap2 | level_pplus_l2 | 2-10 | +0.2387 |  |
| cap2 | level_pplus_l2 | 11-50 | +0.0093 |  |
| cap2 | level_zbar_l2 | 51-100 | -143.2668 |  |
| cap2 | level_zbar_l2 | 2-10 | -5.4856 |  |
| cap2 | level_zbar_l2 | 11-50 | -56.9331 |  |
| cap2 | level_d_l2 | 51-100 | -2.8485 |  |
| cap2 | level_d_l2 | 2-10 | -0.8573 |  |
| cap2 | level_d_l2 | 11-50 | -2.3534 |  |
| cap2 | level_sigma_l2 | 51-100 | +49.7556 |  |
| cap2 | level_sigma_l2 | 2-10 | +4.6015 |  |
| cap2 | level_sigma_l2 | 11-50 | +23.0224 |  |
| cap2 | level_d_l1 | 51-100 | -0.3060 |  |
| cap2 | level_d_l1 | 2-10 | -0.3486 |  |
| cap2 | level_d_l1 | 11-50 | -0.0441 |  |
| cap2 | level_sigma_l1 | 51-100 | +14.3336 |  |
| cap2 | level_sigma_l1 | 2-10 | +4.2171 |  |
| cap2 | level_sigma_l1 | 11-50 | +10.1932 |  |
| cap2 | level_v_norm_l1 | 51-100 | +24.9306 |  |
| cap2 | level_v_norm_l1 | 2-10 | +8.6158 |  |
| cap2 | level_v_norm_l1 | 11-50 | +19.1290 |  |
| cap2 | level_q_l1 | 51-100 | +0.0421 |  |
| cap2 | level_q_l1 | 2-10 | -0.2234 |  |
| cap2 | level_q_l1 | 11-50 | -0.1432 |  |
| cap2 | level_b2 | 51-100 | -0.3145 |  |
| cap2 | level_b2 | 2-10 | -0.0390 |  |
| cap2 | level_b2 | 11-50 | -0.2306 |  |
| cap2 | level_row_norm_l2 | 51-100 | +1.5575 |  |
| cap2 | level_row_norm_l2 | 2-10 | +1.6124 |  |
| cap2 | level_row_norm_l2 | 11-50 | +1.5846 |  |
| cap2 | level_frac_q_pos | 51-100 | +0.2579 |  |
| cap2 | level_frac_always_on | 51-100 | +0.1147 |  |
| cap2 | level_mu2sq_share_q_pos | 51-100 | +0.9981 |  |
| cap2 | level_mu2sq_share_always_on | 51-100 | +0.9552 |  |
| cap2 | level_frac_q_pos | 2-10 | +0.2542 |  |
| cap2 | level_frac_always_on | 2-10 | +0.0009 |  |
| cap2 | level_mu2sq_share_q_pos | 2-10 | +0.7495 |  |
| cap2 | level_mu2sq_share_always_on | 2-10 | +0.0249 |  |
| cap2 | level_frac_q_pos | 11-50 | +0.3640 |  |
| cap2 | level_frac_always_on | 11-50 | +0.0519 |  |
| cap2 | level_mu2sq_share_q_pos | 11-50 | +0.9858 |  |
| cap2 | level_mu2sq_share_always_on | 11-50 | +0.5093 |  |
| cap2 | level_q2mu2 | 51-100 | -142.9523 |  |
| cap2 | level_q2mu2 | 2-10 | -5.4466 |  |
| cap2 | level_q2mu2 | 11-50 | -56.7025 |  |
| cap2 | level_mu2_norm | 51-100 | +280.3235 |  |
| cap2 | level_mu2_norm | 2-10 | +18.6147 |  |
| cap2 | level_mu2_norm | 11-50 | +111.1812 |  |
| cap2 | level_mu2_proj_sd | 51-100 | +95.5383 |  |
| cap2 | level_mu2_proj_sd | 2-10 | +6.3884 |  |
| cap2 | level_mu2_proj_sd | 11-50 | +41.7888 |  |
| cap2 | level_s2_mean | 51-100 | +1075.5332 |  |
| cap2 | level_s2_mean | 2-10 | +94.8678 |  |
| cap2 | level_s2_mean | 11-50 | +556.4897 |  |
| cap2 | level_s2_sd | 51-100 | +346.2127 |  |
| cap2 | level_s2_sd | 2-10 | +36.7968 |  |
| cap2 | level_s2_sd | 11-50 | +183.1719 |  |
| cap2 | level_memo_acc | 51-100 | +0.1289 |  |
| cap2 | level_memo_acc | 2-10 | +0.9847 |  |
| cap2 | level_memo_acc | 11-50 | +0.3097 |  |
| cap2 | level_online_acc | 51-100 | +0.1098 |  |
| cap2 | level_online_acc | 2-10 | +0.6908 |  |
| cap2 | level_online_acc | 11-50 | +0.1760 |  |
| cap2 | absorb_l1_frac_absorbed | 1-100 | +0.0000 |  |
| cap2 | absorb_l1_median_absorb_task | 1-100 | nan |  |
| cap2 | absorb_l1_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap2 | absorb_l1_frac_ever_neg | 1-100 | +0.0080 |  |
| cap2 | absorb_l2_frac_absorbed | 1-100 | +0.9930 |  |
| cap2 | absorb_l2_median_absorb_task | 1-100 | +37.7500 |  |
| cap2 | absorb_l2_frac_neg_at_end | 1-100 | +0.9470 |  |
| cap2 | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| cap2 | t_half_median_observed | 2-100 | +11.0000 |  |
| cap2 | seeds_at_floor | 51-100 | +9.0000 |  |
| cap12 | delta_E1 | 51-100 | +0.0382 | +0.7347 [+0.7290, +0.7404] (10/10 正) |
| cap12 | delta_E1 | 2-10 | +0.0652 | +0.3631 [+0.3317, +0.3945] (10/10 正) |
| cap12 | delta_E1 | 11-50 | +0.0502 | +0.7417 [+0.7368, +0.7467] (10/10 正) |
| cap12 | delta_E2 | 51-100 | -0.5659 | +0.2104 [+0.2020, +0.2188] (10/10 正) |
| cap12 | delta_E2 | 2-10 | -0.3565 | +0.2204 [+0.2000, +0.2408] (10/10 正) |
| cap12 | delta_E2 | 11-50 | -0.5296 | +0.2463 [+0.2393, +0.2533] (10/10 正) |
| cap12 | level_g2_exp | 51-100 | +0.2104 |  |
| cap12 | level_g2_exp | 2-10 | +0.4198 |  |
| cap12 | level_g2_exp | 11-50 | +0.2467 |  |
| cap12 | level_dtrain0_l2 | 51-100 | +0.0000 |  |
| cap12 | level_dtrain0_l2 | 2-10 | +0.0000 |  |
| cap12 | level_dtrain0_l2 | 11-50 | +0.0001 |  |
| cap12 | level_dtrain_l1 | 51-100 | +0.7824 |  |
| cap12 | level_dtrain_l1 | 2-10 | +0.7212 |  |
| cap12 | level_dtrain_l1 | 11-50 | +0.7768 |  |
| cap12 | level_g1 | 51-100 | +0.7824 |  |
| cap12 | level_g1 | 2-10 | +0.7212 |  |
| cap12 | level_g1 | 11-50 | +0.7768 |  |
| cap12 | level_pplus_l2 | 51-100 | +0.0957 |  |
| cap12 | level_pplus_l2 | 2-10 | +0.2612 |  |
| cap12 | level_pplus_l2 | 11-50 | +0.1234 |  |
| cap12 | level_zbar_l2 | 51-100 | -3.1520 |  |
| cap12 | level_zbar_l2 | 2-10 | -1.6069 |  |
| cap12 | level_zbar_l2 | 11-50 | -2.8784 |  |
| cap12 | level_d_l2 | 51-100 | -1.2999 |  |
| cap12 | level_d_l2 | 2-10 | -0.6607 |  |
| cap12 | level_d_l2 | 11-50 | -1.1616 |  |
| cap12 | level_sigma_l2 | 51-100 | +2.4231 |  |
| cap12 | level_sigma_l2 | 2-10 | +2.3212 |  |
| cap12 | level_sigma_l2 | 11-50 | +2.4690 |  |
| cap12 | level_d_l1 | 51-100 | +0.2046 |  |
| cap12 | level_d_l1 | 2-10 | +0.0111 |  |
| cap12 | level_d_l1 | 11-50 | +0.2107 |  |
| cap12 | level_sigma_l1 | 51-100 | +1.3202 |  |
| cap12 | level_sigma_l1 | 2-10 | +1.4419 |  |
| cap12 | level_sigma_l1 | 11-50 | +1.3708 |  |
| cap12 | level_v_norm_l1 | 51-100 | +3.4076 |  |
| cap12 | level_v_norm_l1 | 2-10 | +3.4077 |  |
| cap12 | level_v_norm_l1 | 11-50 | +3.4077 |  |
| cap12 | level_q_l1 | 51-100 | -0.0415 |  |
| cap12 | level_q_l1 | 2-10 | -0.0009 |  |
| cap12 | level_q_l1 | 11-50 | -0.0167 |  |
| cap12 | level_b2 | 51-100 | -1.1075 |  |
| cap12 | level_b2 | 2-10 | -0.2561 |  |
| cap12 | level_b2 | 11-50 | -0.6684 |  |
| cap12 | level_row_norm_l2 | 51-100 | +1.6125 |  |
| cap12 | level_row_norm_l2 | 2-10 | +1.6129 |  |
| cap12 | level_row_norm_l2 | 11-50 | +1.6129 |  |
| cap12 | level_frac_q_pos | 51-100 | +0.2353 |  |
| cap12 | level_frac_always_on | 51-100 | +0.0000 |  |
| cap12 | level_mu2sq_share_q_pos | 51-100 | +0.3038 |  |
| cap12 | level_mu2sq_share_always_on | 51-100 | +0.0000 |  |
| cap12 | level_frac_q_pos | 2-10 | +0.5234 |  |
| cap12 | level_frac_always_on | 2-10 | +0.0000 |  |
| cap12 | level_mu2sq_share_q_pos | 2-10 | +0.6576 |  |
| cap12 | level_mu2sq_share_always_on | 2-10 | +0.0000 |  |
| cap12 | level_frac_q_pos | 11-50 | +0.3712 |  |
| cap12 | level_frac_always_on | 11-50 | +0.0000 |  |
| cap12 | level_mu2sq_share_q_pos | 11-50 | +0.3190 |  |
| cap12 | level_mu2sq_share_always_on | 11-50 | +0.0000 |  |
| cap12 | level_q2mu2 | 51-100 | -2.0445 |  |
| cap12 | level_q2mu2 | 2-10 | -1.3508 |  |
| cap12 | level_q2mu2 | 11-50 | -2.2099 |  |
| cap12 | level_mu2_norm | 51-100 | +4.9652 |  |
| cap12 | level_mu2_norm | 2-10 | +4.2498 |  |
| cap12 | level_mu2_norm | 11-50 | +5.4946 |  |
| cap12 | level_mu2_proj_sd | 51-100 | +1.9576 |  |
| cap12 | level_mu2_proj_sd | 2-10 | +2.0544 |  |
| cap12 | level_mu2_proj_sd | 11-50 | +2.0685 |  |
| cap12 | level_s2_mean | 51-100 | +44.7993 |  |
| cap12 | level_s2_mean | 2-10 | +30.5636 |  |
| cap12 | level_s2_mean | 11-50 | +47.2209 |  |
| cap12 | level_s2_sd | 51-100 | +17.4740 |  |
| cap12 | level_s2_sd | 2-10 | +13.9901 |  |
| cap12 | level_s2_sd | 11-50 | +17.3414 |  |
| cap12 | level_memo_acc | 51-100 | +0.9944 |  |
| cap12 | level_memo_acc | 2-10 | +0.9934 |  |
| cap12 | level_memo_acc | 11-50 | +0.9957 |  |
| cap12 | level_online_acc | 51-100 | +0.8348 |  |
| cap12 | level_online_acc | 2-10 | +0.8618 |  |
| cap12 | level_online_acc | 11-50 | +0.8468 |  |
| cap12 | absorb_l1_frac_absorbed | 1-100 | +0.0000 |  |
| cap12 | absorb_l1_median_absorb_task | 1-100 | nan |  |
| cap12 | absorb_l1_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap12 | absorb_l1_frac_ever_neg | 1-100 | +0.0000 |  |
| cap12 | absorb_l2_frac_absorbed | 1-100 | +0.0000 |  |
| cap12 | absorb_l2_median_absorb_task | 1-100 | nan |  |
| cap12 | absorb_l2_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap12 | absorb_l2_frac_ever_neg | 1-100 | +0.0680 |  |
| cap12 | t_half_median_observed | 2-100 | nan |  |
| cap12 | seeds_at_floor | 51-100 | +0.0000 |  |
| cap12_bfix | delta_E1 | 51-100 | +0.0511 | +0.7476 [+0.7433, +0.7519] (10/10 正) |
| cap12_bfix | delta_E1 | 2-10 | +0.0639 | +0.3618 [+0.3291, +0.3946] (10/10 正) |
| cap12_bfix | delta_E1 | 11-50 | +0.0572 | +0.7487 [+0.7437, +0.7538] (10/10 正) |
| cap12_bfix | delta_E2 | 51-100 | -0.3234 | +0.4528 [+0.4438, +0.4619] (10/10 正) |
| cap12_bfix | delta_E2 | 2-10 | -0.2526 | +0.3242 [+0.3066, +0.3419] (10/10 正) |
| cap12_bfix | delta_E2 | 11-50 | -0.3052 | +0.4707 [+0.4659, +0.4755] (10/10 正) |
| cap12_bfix | level_g2_exp | 51-100 | +0.4528 |  |
| cap12_bfix | level_g2_exp | 2-10 | +0.5236 |  |
| cap12_bfix | level_g2_exp | 11-50 | +0.4710 |  |
| cap12_bfix | level_dtrain0_l2 | 51-100 | +0.0002 |  |
| cap12_bfix | level_dtrain0_l2 | 2-10 | +0.0001 |  |
| cap12_bfix | level_dtrain0_l2 | 11-50 | +0.0002 |  |
| cap12_bfix | level_dtrain_l1 | 51-100 | +0.7365 |  |
| cap12_bfix | level_dtrain_l1 | 2-10 | +0.7100 |  |
| cap12_bfix | level_dtrain_l1 | 11-50 | +0.7302 |  |
| cap12_bfix | level_g1 | 51-100 | +0.7365 |  |
| cap12_bfix | level_g1 | 2-10 | +0.7100 |  |
| cap12_bfix | level_g1 | 11-50 | +0.7302 |  |
| cap12_bfix | level_pplus_l2 | 51-100 | +0.3173 |  |
| cap12_bfix | level_pplus_l2 | 2-10 | +0.3719 |  |
| cap12_bfix | level_pplus_l2 | 11-50 | +0.3350 |  |
| cap12_bfix | level_zbar_l2 | 51-100 | -1.6342 |  |
| cap12_bfix | level_zbar_l2 | 2-10 | -1.0453 |  |
| cap12_bfix | level_zbar_l2 | 11-50 | -1.5126 |  |
| cap12_bfix | level_d_l2 | 51-100 | -0.5704 |  |
| cap12_bfix | level_d_l2 | 2-10 | -0.3935 |  |
| cap12_bfix | level_d_l2 | 11-50 | -0.5235 |  |
| cap12_bfix | level_sigma_l2 | 51-100 | +2.7524 |  |
| cap12_bfix | level_sigma_l2 | 2-10 | +2.4778 |  |
| cap12_bfix | level_sigma_l2 | 11-50 | +2.7626 |  |
| cap12_bfix | level_d_l1 | 51-100 | +0.0386 |  |
| cap12_bfix | level_d_l1 | 2-10 | -0.0366 |  |
| cap12_bfix | level_d_l1 | 11-50 | +0.0266 |  |
| cap12_bfix | level_sigma_l1 | 51-100 | +1.3474 |  |
| cap12_bfix | level_sigma_l1 | 2-10 | +1.4252 |  |
| cap12_bfix | level_sigma_l1 | 11-50 | +1.3742 |  |
| cap12_bfix | level_v_norm_l1 | 51-100 | +3.4076 |  |
| cap12_bfix | level_v_norm_l1 | 2-10 | +3.4076 |  |
| cap12_bfix | level_v_norm_l1 | 11-50 | +3.4076 |  |
| cap12_bfix | level_q_l1 | 51-100 | +0.0098 |  |
| cap12_bfix | level_q_l1 | 2-10 | -0.0094 |  |
| cap12_bfix | level_q_l1 | 11-50 | +0.0074 |  |
| cap12_bfix | level_b2 | 51-100 | +0.0363 |  |
| cap12_bfix | level_b2 | 2-10 | +0.0363 |  |
| cap12_bfix | level_b2 | 11-50 | +0.0363 |  |
| cap12_bfix | level_row_norm_l2 | 51-100 | +1.6108 |  |
| cap12_bfix | level_row_norm_l2 | 2-10 | +1.6122 |  |
| cap12_bfix | level_row_norm_l2 | 11-50 | +1.6113 |  |
| cap12_bfix | level_frac_q_pos | 51-100 | +0.6605 |  |
| cap12_bfix | level_frac_always_on | 51-100 | +0.0000 |  |
| cap12_bfix | level_mu2sq_share_q_pos | 51-100 | +0.9252 |  |
| cap12_bfix | level_mu2sq_share_always_on | 51-100 | +0.0000 |  |
| cap12_bfix | level_frac_q_pos | 2-10 | +0.5050 |  |
| cap12_bfix | level_frac_always_on | 2-10 | +0.0000 |  |
| cap12_bfix | level_mu2sq_share_q_pos | 2-10 | +0.8426 |  |
| cap12_bfix | level_mu2sq_share_always_on | 2-10 | +0.0000 |  |
| cap12_bfix | level_frac_q_pos | 11-50 | +0.6410 |  |
| cap12_bfix | level_frac_always_on | 11-50 | +0.0000 |  |
| cap12_bfix | level_mu2sq_share_q_pos | 11-50 | +0.9179 |  |
| cap12_bfix | level_mu2sq_share_always_on | 11-50 | +0.0000 |  |
| cap12_bfix | level_q2mu2 | 51-100 | -1.6705 |  |
| cap12_bfix | level_q2mu2 | 2-10 | -1.0815 |  |
| cap12_bfix | level_q2mu2 | 11-50 | -1.5489 |  |
| cap12_bfix | level_mu2_norm | 51-100 | +4.1500 |  |
| cap12_bfix | level_mu2_norm | 2-10 | +3.5352 |  |
| cap12_bfix | level_mu2_norm | 11-50 | +4.1224 |  |
| cap12_bfix | level_mu2_proj_sd | 51-100 | +4.3783 |  |
| cap12_bfix | level_mu2_proj_sd | 2-10 | +3.2662 |  |
| cap12_bfix | level_mu2_proj_sd | 11-50 | +4.5168 |  |
| cap12_bfix | level_s2_mean | 51-100 | +30.5357 |  |
| cap12_bfix | level_s2_mean | 2-10 | +25.3325 |  |
| cap12_bfix | level_s2_mean | 11-50 | +30.2789 |  |
| cap12_bfix | level_s2_sd | 51-100 | +34.0845 |  |
| cap12_bfix | level_s2_sd | 2-10 | +24.5025 |  |
| cap12_bfix | level_s2_sd | 11-50 | +35.1445 |  |
| cap12_bfix | level_memo_acc | 51-100 | +0.9921 |  |
| cap12_bfix | level_memo_acc | 2-10 | +0.9889 |  |
| cap12_bfix | level_memo_acc | 11-50 | +0.9921 |  |
| cap12_bfix | level_online_acc | 51-100 | +0.8477 |  |
| cap12_bfix | level_online_acc | 2-10 | +0.8605 |  |
| cap12_bfix | level_online_acc | 11-50 | +0.8538 |  |
| cap12_bfix | absorb_l1_frac_absorbed | 1-100 | +0.0000 |  |
| cap12_bfix | absorb_l1_median_absorb_task | 1-100 | nan |  |
| cap12_bfix | absorb_l1_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap12_bfix | absorb_l1_frac_ever_neg | 1-100 | +0.0000 |  |
| cap12_bfix | absorb_l2_frac_absorbed | 1-100 | +0.0000 |  |
| cap12_bfix | absorb_l2_median_absorb_task | 1-100 | nan |  |
| cap12_bfix | absorb_l2_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap12_bfix | absorb_l2_frac_ever_neg | 1-100 | +0.0000 |  |
| cap12_bfix | t_half_median_observed | 2-100 | nan |  |
| cap12_bfix | seeds_at_floor | 51-100 | +0.0000 |  |

## 5. 読み方（spec §5.4 のまま）

- COLLAPSED の腕では第1層も下流の停止で凍るので、第1層の水準を機構の比較に使わない。
- cap12 が COLLAPSED なら、z̄₂ = q₂‖µ₂‖ + b₂ の分解と cap12_bfix、d₂ と −c₂ で担い手を読む（§5.4）。
- RESCUED は「この箱の 100 タスクで床に落ちなかった」であり、無限時間の非崩壊ではない。
- IMPAIRED の腕では「可塑性を守った」と書かない。
- 機構（‖µ₂‖・c₂・z̄₂）は報告のみ。腕の順位の一致は因果の証明ではない。
