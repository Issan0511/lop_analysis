# mucap_ee_0917 — 判定（第1層の成分上限・ELU→ELU・100 タスク）

> 自動生成: `analysis/mucap_ee_0917/verdict.py`。spec: `specs/spec_mucap_ee_0917.md`。数値は `verdict.csv`・`paired.csv`・`timing.csv`・`secondary.csv` から。

## 0. 実行したもの

- 集計時の HEAD: `18d3ef08f61c8a389460a51011fe4d1fa89274e3`・run の commit: ['18d3ef08f61c8a389460a51011fe4d1fa89274e3']
- shard: 40/40・欠損 0・有効 seed 10（無効: なし）

## 1. 適用条件（spec §5.2）

- (A) 有効 seed 10 ≥ 8: **True**
- (B) ref が主窓で床: 10/10 seed（要 ≥ 8）、ref の第2層 ΔE2 -0.7762 [-0.7975, -0.7549] → **True**
- (C) 上限が task 2–10 の全タスクで書いた: cap_par **True**、cap_perp **True**、cap_both **True**
- IMPAIRED（読みのフラグ）: cap_par 0/10 seed → **False**、cap_perp 0/10 seed → **False**、cap_both 0/10 seed → **False**
- 主窓の床の状態: ref FLOORED、cap_par FLOORED、cap_perp FLOORED、cap_both FLOORED

## 2. 判定

| 比較 | 水準 | ラベル | 時間（T_half） |
|---|---|---|---|
| **主: cap_perp − ref** | 97.5% | **COLLAPSED** | LATER |
| cap_perp − ref（参考） | 95% | COLLAPSED | |
| cap_par − ref | 95% | COLLAPSED | LATER |
| cap_both − ref | 95% | COLLAPSED | LATER |

## 3. 腕間差（主窓 task 51–100・seed 内対応差）

| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |
|---|---|---|---|---|---|---|
| cap_par | E1 | 0.975 | +0.0014 | +0.0012 | [+0.0004, +0.0024] | + |
| cap_par | E2 | 0.975 | +0.0001 | +0.0000 | [+0.0000, +0.0001] | + |
| cap_par | E1 | 0.95 | +0.0014 | +0.0012 | [+0.0006, +0.0023] | + |
| cap_par | E2 | 0.95 | +0.0001 | +0.0000 | [+0.0000, +0.0001] | + |
| cap_perp | E1 | 0.975 | +0.0006 | +0.0006 | [+0.0001, +0.0012] | + |
| cap_perp | E2 | 0.975 | +0.0000 | +0.0000 | [+0.0000, +0.0000] | + |
| cap_perp | E1 | 0.95 | +0.0006 | +0.0006 | [+0.0002, +0.0011] | + |
| cap_perp | E2 | 0.95 | +0.0000 | +0.0000 | [+0.0000, +0.0000] | + |
| cap_both | E1 | 0.975 | +0.0007 | +0.0009 | [-0.0000, +0.0015] | 0 |
| cap_both | E2 | 0.975 | +0.0000 | +0.0001 | [-0.0000, +0.0001] | 0 |
| cap_both | E1 | 0.95 | +0.0007 | +0.0009 | [+0.0001, +0.0014] | + |
| cap_both | E2 | 0.95 | +0.0000 | +0.0001 | [+0.0000, +0.0001] | + |

交互作用 δ_both − δ_par − δ_perp（95%）: E1 -0.0013 [-0.0025, -0.0001]、E2 -0.0000 [-0.0001, +0.0000]

### 3.1 seed 別の差と床

| 腕 | endpoint | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cap_par | E1 | +0.001 | +0.004 | -0.000 | +0.000 | +0.003 | +0.002 | +0.001 | +0.001 | +0.001 | +0.002 |
| cap_par | E2 | +0.000 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cap_perp | E1 | +0.000 | +0.001 | -0.000 | +0.001 | +0.001 | +0.001 | +0.002 | +0.000 | +0.000 | +0.000 |
| cap_perp | E2 | +0.000 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cap_both | E1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 | +0.001 | +0.002 | +0.002 | +0.000 |
| cap_both | E2 | +0.000 | +0.000 | +0.000 | -0.000 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| ref | 主窓 online（閾値） | 0.100 (0.116) 床 | 0.100 (0.116) 床 | 0.101 (0.116) 床 | 0.100 (0.117) 床 | 0.099 (0.114) 床 | 0.101 (0.118) 床 | 0.100 (0.115) 床 | 0.100 (0.116) 床 | 0.100 (0.116) 床 | 0.100 (0.116) 床 |
| cap_par | 主窓 online（閾値） | 0.101 (0.116) 床 | 0.104 (0.116) 床 | 0.101 (0.116) 床 | 0.101 (0.117) 床 | 0.102 (0.114) 床 | 0.103 (0.118) 床 | 0.101 (0.115) 床 | 0.101 (0.116) 床 | 0.101 (0.116) 床 | 0.102 (0.116) 床 |
| cap_perp | 主窓 online（閾値） | 0.100 (0.116) 床 | 0.101 (0.116) 床 | 0.100 (0.116) 床 | 0.101 (0.117) 床 | 0.100 (0.114) 床 | 0.102 (0.118) 床 | 0.102 (0.115) 床 | 0.100 (0.116) 床 | 0.100 (0.116) 床 | 0.101 (0.116) 床 |
| cap_both | 主窓 online（閾値） | 0.100 (0.116) 床 | 0.101 (0.116) 床 | 0.101 (0.116) 床 | 0.100 (0.117) 床 | 0.100 (0.114) 床 | 0.101 (0.118) 床 | 0.101 (0.115) 床 | 0.102 (0.116) 床 | 0.102 (0.116) 床 | 0.100 (0.116) 床 |

### 3.2 T_half（適合率 F が初めて 1/2 を下回るタスク・— は 100 まで未到達）

| 腕 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 早/遅/同 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | 7 | 7 | 7 | 7 | 7 | 5 | 8 | 7 | 7 | 8 | | |
| cap_par | 9 | 11 | 9 | 9 | 12 | 9 | 12 | 11 | 11 | 10 | 0/10/0 | 0.00195 |
| cap_perp | 13 | 15 | 14 | 13 | 14 | 12 | 16 | 16 | 12 | 14 | 0/10/0 | 0.00195 |
| cap_both | 14 | 15 | 14 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 0/10/0 | 0.00195 |

## 4. 副 endpoint（判定に使わない）

| 腕 | 量 | 窓 | 値 | 差 [95%] |
|---|---|---|---|---|
| ref | delta_E1 | 51-100 | -0.6965 |  |
| ref | delta_E1 | 2-10 | -0.2979 |  |
| ref | delta_E1 | 11-50 | -0.6915 |  |
| ref | delta_E2 | 51-100 | -0.7762 |  |
| ref | delta_E2 | 2-10 | -0.5769 |  |
| ref | delta_E2 | 11-50 | -0.7759 |  |
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
| cap_par | delta_E1 | 51-100 | -0.6951 | +0.0014 [+0.0006, +0.0023] (9/10 正) |
| cap_par | delta_E1 | 2-10 | -0.1044 | +0.1935 [+0.1669, +0.2201] (10/10 正) |
| cap_par | delta_E1 | 11-50 | -0.6387 | +0.0528 [+0.0360, +0.0695] (10/10 正) |
| cap_par | delta_E2 | 51-100 | -0.7762 | +0.0001 [+0.0000, +0.0001] (9/10 正) |
| cap_par | delta_E2 | 2-10 | -0.5539 | +0.0229 [-0.0062, +0.0521] (7/10 正) |
| cap_par | delta_E2 | 11-50 | -0.7676 | +0.0083 [+0.0036, +0.0130] (10/10 正) |
| cap_par | level_g1 | 51-100 | +0.5907 |  |
| cap_par | level_g1 | 2-10 | +0.6355 |  |
| cap_par | level_g1 | 11-50 | +0.6091 |  |
| cap_par | level_pplus_l2 | 51-100 | +0.0001 |  |
| cap_par | level_pplus_l2 | 2-10 | +0.1790 |  |
| cap_par | level_pplus_l2 | 11-50 | +0.0077 |  |
| cap_par | level_zbar_l2 | 51-100 | -147.9666 |  |
| cap_par | level_zbar_l2 | 2-10 | -14.7716 |  |
| cap_par | level_zbar_l2 | 11-50 | -104.5670 |  |
| cap_par | level_d_l2 | 51-100 | -2.8647 |  |
| cap_par | level_d_l2 | 2-10 | -0.9228 |  |
| cap_par | level_d_l2 | 11-50 | -2.0801 |  |
| cap_par | level_sigma_l2 | 51-100 | +52.8715 |  |
| cap_par | level_sigma_l2 | 2-10 | +13.2202 |  |
| cap_par | level_sigma_l2 | 11-50 | +50.2010 |  |
| cap_par | level_d_l1 | 51-100 | +0.0929 |  |
| cap_par | level_d_l1 | 2-10 | +0.0196 |  |
| cap_par | level_d_l1 | 11-50 | +0.1124 |  |
| cap_par | level_sigma_l1 | 51-100 | +8.2149 |  |
| cap_par | level_sigma_l1 | 2-10 | +3.3677 |  |
| cap_par | level_sigma_l1 | 11-50 | +6.9829 |  |
| cap_par | level_v_norm_l1 | 51-100 | +14.7386 |  |
| cap_par | level_v_norm_l1 | 2-10 | +7.6130 |  |
| cap_par | level_v_norm_l1 | 11-50 | +13.2155 |  |
| cap_par | level_q_l1 | 51-100 | +0.0000 |  |
| cap_par | level_q_l1 | 2-10 | +0.0063 |  |
| cap_par | level_q_l1 | 11-50 | -0.0011 |  |
| cap_par | level_b2 | 51-100 | -1.0052 |  |
| cap_par | level_b2 | 2-10 | -0.3754 |  |
| cap_par | level_b2 | 11-50 | -0.9411 |  |
| cap_par | level_row_norm_l2 | 51-100 | +5.5649 |  |
| cap_par | level_row_norm_l2 | 2-10 | +3.3267 |  |
| cap_par | level_row_norm_l2 | 11-50 | +5.2475 |  |
| cap_par | level_frac_q_pos | 51-100 | +0.5274 |  |
| cap_par | level_frac_always_on | 51-100 | +0.0000 |  |
| cap_par | level_mu2sq_share_q_pos | 51-100 | +0.6855 |  |
| cap_par | level_mu2sq_share_always_on | 51-100 | +0.0000 |  |
| cap_par | level_frac_q_pos | 2-10 | +0.5551 |  |
| cap_par | level_frac_always_on | 2-10 | +0.0000 |  |
| cap_par | level_mu2sq_share_q_pos | 2-10 | +0.6529 |  |
| cap_par | level_mu2sq_share_always_on | 2-10 | +0.0000 |  |
| cap_par | level_frac_q_pos | 11-50 | +0.4936 |  |
| cap_par | level_frac_always_on | 11-50 | +0.0000 |  |
| cap_par | level_mu2sq_share_q_pos | 11-50 | +0.5397 |  |
| cap_par | level_mu2sq_share_always_on | 11-50 | +0.0000 |  |
| cap_par | level_mu2_norm | 51-100 | +57.1864 |  |
| cap_par | level_mu2_norm | 2-10 | +12.0169 |  |
| cap_par | level_mu2_norm | 11-50 | +42.2209 |  |
| cap_par | level_mu2_proj_sd | 51-100 | +20.6960 |  |
| cap_par | level_mu2_proj_sd | 2-10 | +7.4764 |  |
| cap_par | level_mu2_proj_sd | 11-50 | +21.4297 |  |
| cap_par | level_s2_mean | 51-100 | +448.4865 |  |
| cap_par | level_s2_mean | 2-10 | +105.3280 |  |
| cap_par | level_s2_mean | 11-50 | +343.3926 |  |
| cap_par | level_s2_sd | 51-100 | +164.0452 |  |
| cap_par | level_s2_sd | 2-10 | +53.9829 |  |
| cap_par | level_s2_sd | 11-50 | +149.8660 |  |
| cap_par | level_memo_acc | 51-100 | +0.1085 |  |
| cap_par | level_memo_acc | 2-10 | +0.8789 |  |
| cap_par | level_memo_acc | 11-50 | +0.1983 |  |
| cap_par | level_online_acc | 51-100 | +0.1015 |  |
| cap_par | level_online_acc | 2-10 | +0.6922 |  |
| cap_par | level_online_acc | 11-50 | +0.1579 |  |
| cap_par | absorb_l1_frac_absorbed | 1-100 | +0.0000 |  |
| cap_par | absorb_l1_median_absorb_task | 1-100 | nan |  |
| cap_par | absorb_l1_frac_neg_at_end | 1-100 | +0.0000 |  |
| cap_par | absorb_l1_frac_ever_neg | 1-100 | +0.0000 |  |
| cap_par | absorb_l2_frac_absorbed | 1-100 | +1.0000 |  |
| cap_par | absorb_l2_median_absorb_task | 1-100 | +31.4000 |  |
| cap_par | absorb_l2_frac_neg_at_end | 1-100 | +0.9970 |  |
| cap_par | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| cap_par | t_half_median_observed | 2-100 | +10.5000 |  |
| cap_par | seeds_at_floor | 51-100 | +10.0000 |  |
| cap_perp | delta_E1 | 51-100 | -0.6958 | +0.0006 [+0.0002, +0.0011] (9/10 正) |
| cap_perp | delta_E1 | 2-10 | -0.0691 | +0.2288 [+0.2104, +0.2472] (10/10 正) |
| cap_perp | delta_E1 | 11-50 | -0.5955 | +0.0960 [+0.0842, +0.1078] (10/10 正) |
| cap_perp | delta_E2 | 51-100 | -0.7762 | +0.0000 [+0.0000, +0.0000] (9/10 正) |
| cap_perp | delta_E2 | 2-10 | -0.3873 | +0.1895 [+0.1742, +0.2049] (10/10 正) |
| cap_perp | delta_E2 | 11-50 | -0.7595 | +0.0164 [+0.0140, +0.0187] (10/10 正) |
| cap_perp | level_g1 | 51-100 | +0.2958 |  |
| cap_perp | level_g1 | 2-10 | +0.4830 |  |
| cap_perp | level_g1 | 11-50 | +0.4226 |  |
| cap_perp | level_pplus_l2 | 51-100 | +0.0000 |  |
| cap_perp | level_pplus_l2 | 2-10 | +0.2720 |  |
| cap_perp | level_pplus_l2 | 11-50 | +0.0127 |  |
| cap_perp | level_zbar_l2 | 51-100 | -128.6593 |  |
| cap_perp | level_zbar_l2 | 2-10 | -3.2606 |  |
| cap_perp | level_zbar_l2 | 11-50 | -54.1886 |  |
| cap_perp | level_d_l2 | 51-100 | -3.5245 |  |
| cap_perp | level_d_l2 | 2-10 | -0.8025 |  |
| cap_perp | level_d_l2 | 11-50 | -2.8873 |  |
| cap_perp | level_sigma_l2 | 51-100 | +36.7511 |  |
| cap_perp | level_sigma_l2 | 2-10 | +3.0256 |  |
| cap_perp | level_sigma_l2 | 11-50 | +17.1819 |  |
| cap_perp | level_d_l1 | 51-100 | -1.0863 |  |
| cap_perp | level_d_l1 | 2-10 | -0.5984 |  |
| cap_perp | level_d_l1 | 11-50 | -0.5962 |  |
| cap_perp | level_sigma_l1 | 51-100 | +4.2614 |  |
| cap_perp | level_sigma_l1 | 2-10 | +1.6740 |  |
| cap_perp | level_sigma_l1 | 11-50 | +2.6301 |  |
| cap_perp | level_v_norm_l1 | 51-100 | +3.3255 |  |
| cap_perp | level_v_norm_l1 | 2-10 | +3.4075 |  |
| cap_perp | level_v_norm_l1 | 11-50 | +3.3951 |  |
| cap_perp | level_q_l1 | 51-100 | +0.1045 |  |
| cap_perp | level_q_l1 | 2-10 | -0.1983 |  |
| cap_perp | level_q_l1 | 11-50 | -0.0607 |  |
| cap_perp | level_b2 | 51-100 | +0.0482 |  |
| cap_perp | level_b2 | 2-10 | +0.0275 |  |
| cap_perp | level_b2 | 11-50 | +0.0293 |  |
| cap_perp | level_row_norm_l2 | 51-100 | +6.3847 |  |
| cap_perp | level_row_norm_l2 | 2-10 | +3.0469 |  |
| cap_perp | level_row_norm_l2 | 11-50 | +5.9254 |  |
| cap_perp | level_frac_q_pos | 51-100 | +0.2543 |  |
| cap_perp | level_frac_always_on | 51-100 | +0.1645 |  |
| cap_perp | level_mu2sq_share_q_pos | 51-100 | +0.9982 |  |
| cap_perp | level_mu2sq_share_always_on | 51-100 | +0.9941 |  |
| cap_perp | level_frac_q_pos | 2-10 | +0.1711 |  |
| cap_perp | level_frac_always_on | 2-10 | +0.0060 |  |
| cap_perp | level_mu2sq_share_q_pos | 2-10 | +0.5316 |  |
| cap_perp | level_mu2sq_share_always_on | 2-10 | +0.1364 |  |
| cap_perp | level_frac_q_pos | 11-50 | +0.2527 |  |
| cap_perp | level_frac_always_on | 11-50 | +0.1042 |  |
| cap_perp | level_mu2sq_share_q_pos | 11-50 | +0.9879 |  |
| cap_perp | level_mu2sq_share_always_on | 11-50 | +0.9128 |  |
| cap_perp | level_mu2_norm | 51-100 | +196.0163 |  |
| cap_perp | level_mu2_norm | 2-10 | +8.5222 |  |
| cap_perp | level_mu2_norm | 11-50 | +83.5267 |  |
| cap_perp | level_mu2_proj_sd | 51-100 | +57.1664 |  |
| cap_perp | level_mu2_proj_sd | 2-10 | +2.5734 |  |
| cap_perp | level_mu2_proj_sd | 11-50 | +24.9659 |  |
| cap_perp | level_s2_mean | 51-100 | +553.8524 |  |
| cap_perp | level_s2_mean | 2-10 | -9.9767 |  |
| cap_perp | level_s2_mean | 11-50 | +211.9500 |  |
| cap_perp | level_s2_sd | 51-100 | +178.4607 |  |
| cap_perp | level_s2_sd | 2-10 | +10.6458 |  |
| cap_perp | level_s2_sd | 11-50 | +71.5205 |  |
| cap_perp | level_memo_acc | 51-100 | +0.1072 |  |
| cap_perp | level_memo_acc | 2-10 | +0.9906 |  |
| cap_perp | level_memo_acc | 11-50 | +0.3449 |  |
| cap_perp | level_online_acc | 51-100 | +0.1008 |  |
| cap_perp | level_online_acc | 2-10 | +0.7275 |  |
| cap_perp | level_online_acc | 11-50 | +0.2011 |  |
| cap_perp | absorb_l1_frac_absorbed | 1-100 | +0.2190 |  |
| cap_perp | absorb_l1_median_absorb_task | 1-100 | +61.8500 |  |
| cap_perp | absorb_l1_frac_neg_at_end | 1-100 | +0.1280 |  |
| cap_perp | absorb_l1_frac_ever_neg | 1-100 | +0.6300 |  |
| cap_perp | absorb_l2_frac_absorbed | 1-100 | +1.0000 |  |
| cap_perp | absorb_l2_median_absorb_task | 1-100 | +30.9000 |  |
| cap_perp | absorb_l2_frac_neg_at_end | 1-100 | +1.0000 |  |
| cap_perp | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| cap_perp | t_half_median_observed | 2-100 | +14.0000 |  |
| cap_perp | seeds_at_floor | 51-100 | +10.0000 |  |
| cap_both | delta_E1 | 51-100 | -0.6957 | +0.0007 [+0.0001, +0.0014] (10/10 正) |
| cap_both | delta_E1 | 2-10 | -0.0148 | +0.2831 [+0.2594, +0.3068] (10/10 正) |
| cap_both | delta_E1 | 11-50 | -0.6000 | +0.0915 [+0.0777, +0.1053] (10/10 正) |
| cap_both | delta_E2 | 51-100 | -0.7762 | +0.0000 [+0.0000, +0.0001] (8/10 正) |
| cap_both | delta_E2 | 2-10 | -0.5257 | +0.0511 [+0.0338, +0.0684] (10/10 正) |
| cap_both | delta_E2 | 11-50 | -0.7637 | +0.0122 [+0.0101, +0.0142] (10/10 正) |
| cap_both | level_g1 | 51-100 | +0.5508 |  |
| cap_both | level_g1 | 2-10 | +0.7128 |  |
| cap_both | level_g1 | 11-50 | +0.6301 |  |
| cap_both | level_pplus_l2 | 51-100 | +0.0000 |  |
| cap_both | level_pplus_l2 | 2-10 | +0.1820 |  |
| cap_both | level_pplus_l2 | 11-50 | +0.0099 |  |
| cap_both | level_zbar_l2 | 51-100 | -76.4672 |  |
| cap_both | level_zbar_l2 | 2-10 | -5.6594 |  |
| cap_both | level_zbar_l2 | 11-50 | -47.2479 |  |
| cap_both | level_d_l2 | 51-100 | -5.6004 |  |
| cap_both | level_d_l2 | 2-10 | -0.9652 |  |
| cap_both | level_d_l2 | 11-50 | -3.3973 |  |
| cap_both | level_sigma_l2 | 51-100 | +14.3526 |  |
| cap_both | level_sigma_l2 | 2-10 | +5.2959 |  |
| cap_both | level_sigma_l2 | 11-50 | +13.9031 |  |
| cap_both | level_d_l1 | 51-100 | -0.0384 |  |
| cap_both | level_d_l1 | 2-10 | +0.0105 |  |
| cap_both | level_d_l1 | 11-50 | +0.0513 |  |
| cap_both | level_sigma_l1 | 51-100 | +2.0173 |  |
| cap_both | level_sigma_l1 | 2-10 | +1.4611 |  |
| cap_both | level_sigma_l1 | 11-50 | +1.7314 |  |
| cap_both | level_v_norm_l1 | 51-100 | +3.3537 |  |
| cap_both | level_v_norm_l1 | 2-10 | +3.4075 |  |
| cap_both | level_v_norm_l1 | 11-50 | +3.3942 |  |
| cap_both | level_q_l1 | 51-100 | -0.0165 |  |
| cap_both | level_q_l1 | 2-10 | +0.0021 |  |
| cap_both | level_q_l1 | 11-50 | -0.0130 |  |
| cap_both | level_b2 | 51-100 | -0.6032 |  |
| cap_both | level_b2 | 2-10 | -0.2792 |  |
| cap_both | level_b2 | 11-50 | -0.5678 |  |
| cap_both | level_row_norm_l2 | 51-100 | +6.9822 |  |
| cap_both | level_row_norm_l2 | 2-10 | +3.7731 |  |
| cap_both | level_row_norm_l2 | 11-50 | +6.7392 |  |
| cap_both | level_frac_q_pos | 51-100 | +0.4583 |  |
| cap_both | level_frac_always_on | 51-100 | +0.0965 |  |
| cap_both | level_mu2sq_share_q_pos | 51-100 | +0.7727 |  |
| cap_both | level_mu2sq_share_always_on | 51-100 | +0.4763 |  |
| cap_both | level_frac_q_pos | 2-10 | +0.5406 |  |
| cap_both | level_frac_always_on | 2-10 | +0.0000 |  |
| cap_both | level_mu2sq_share_q_pos | 2-10 | +0.7142 |  |
| cap_both | level_mu2sq_share_always_on | 2-10 | +0.0000 |  |
| cap_both | level_frac_q_pos | 11-50 | +0.4478 |  |
| cap_both | level_frac_always_on | 11-50 | +0.0295 |  |
| cap_both | level_mu2sq_share_q_pos | 11-50 | +0.6597 |  |
| cap_both | level_mu2sq_share_always_on | 11-50 | +0.1447 |  |
| cap_both | level_mu2_norm | 51-100 | +29.5145 |  |
| cap_both | level_mu2_norm | 2-10 | +5.0969 |  |
| cap_both | level_mu2_norm | 11-50 | +18.2306 |  |
| cap_both | level_mu2_proj_sd | 51-100 | +5.2413 |  |
| cap_both | level_mu2_proj_sd | 2-10 | +2.2409 |  |
| cap_both | level_mu2_proj_sd | 11-50 | +4.8745 |  |
| cap_both | level_s2_mean | 51-100 | +141.6686 |  |
| cap_both | level_s2_mean | 2-10 | +32.4279 |  |
| cap_both | level_s2_mean | 11-50 | +90.3229 |  |
| cap_both | level_s2_sd | 51-100 | +33.2153 |  |
| cap_both | level_s2_sd | 2-10 | +14.5697 |  |
| cap_both | level_s2_sd | 11-50 | +26.1767 |  |
| cap_both | level_memo_acc | 51-100 | +0.1077 |  |
| cap_both | level_memo_acc | 2-10 | +0.9892 |  |
| cap_both | level_memo_acc | 11-50 | +0.2872 |  |
| cap_both | level_online_acc | 51-100 | +0.1009 |  |
| cap_both | level_online_acc | 2-10 | +0.7818 |  |
| cap_both | level_online_acc | 11-50 | +0.1966 |  |
| cap_both | absorb_l1_frac_absorbed | 1-100 | +0.0150 |  |
| cap_both | absorb_l1_median_absorb_task | 1-100 | +75.6667 |  |
| cap_both | absorb_l1_frac_neg_at_end | 1-100 | +0.0150 |  |
| cap_both | absorb_l1_frac_ever_neg | 1-100 | +0.0970 |  |
| cap_both | absorb_l2_frac_absorbed | 1-100 | +1.0000 |  |
| cap_both | absorb_l2_median_absorb_task | 1-100 | +29.4500 |  |
| cap_both | absorb_l2_frac_neg_at_end | 1-100 | +0.9970 |  |
| cap_both | absorb_l2_frac_ever_neg | 1-100 | +1.0000 |  |
| cap_both | t_half_median_observed | 2-100 | +14.5000 |  |
| cap_both | seeds_at_floor | 51-100 | +10.0000 |  |

## 5. 読み方（spec §5.4 のまま）

- COLLAPSED の腕では第1層も下流の停止で凍るので、第1層の水準を機構の比較に使わない。
- RESCUED は「この箱の 100 タスクで床に落ちなかった」であり、無限時間の非崩壊ではない。
- IMPAIRED の腕では「可塑性を守った」と書かない。
- 機構（‖µ₂‖・c₂・z̄₂）は報告のみ。腕の順位の一致は因果の証明ではない。
