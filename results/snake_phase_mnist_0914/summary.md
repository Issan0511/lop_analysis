# snake_phase_mnist_0914 summary

spec: `specs/spec_snake_phase_mnist_0914.md`（事前登録 aa78461）。数値はすべて verdict.csv / verdict.json と同じ計算から出力。単位は pt（精度 ×100）、CT1 は log。
解釈（実装時に固定・本走の結果を見る前）: M1 のラベル（書き方の規則と STRUCTURE）は層 1 で判定し層 2 は併記。CI は符号反転検定の反転（二分法 1e−4 pt、外側の端）。Holm は登録の m を固定。

分解能マージン h（基準対 N06−LIN・LR−LIN のみ）: D +0.201・A_late +0.124・A_base +0.158・Gap +0.154

## 可検定性

| | n | D_pair 平均 | 95% CI | 判定 |
|---|---:|---:|---|---|
| TESTABLE_FIXED | 20 | +2.044 | [+1.954, +2.133] | 成立 |
| TESTABLE_LEAKY | 20 | +1.783 | [+1.710, +1.859] | 成立 |

## E1 時間劣化（確認的見出し）・E2 水準・E3 fresh gap

| 族 | 対比 | E1 D（mean t16–30 − mean t101–120） | 95% CI | p_Holm | E1 | ΔA_late (t101–120) | E2 late | ΔA_base (t16–30) | E2 base | ΔGap (t101–120) | E3 | 修飾 | §6.4 |
|---|---|---:|---|---:|---|---:|---|---:|---|---:|---|---|---|
| F1 | C1 | -0.408 (n=20) | [-0.497, -0.318] | +0.000 | LESS_DECLINE | -0.573 | LOWER | -0.980 | LOWER | -0.901 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F1 | C2 | -0.240 (n=20) | [-0.316, -0.164] | +0.000 | LESS_DECLINE | -0.772 | LOWER | -1.013 | LOWER | -0.661 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F1 | C3 | -0.461 (n=20) | [-0.551, -0.368] | +0.000 | LESS_DECLINE | -0.623 | LOWER | -1.084 | LOWER | -0.197 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F1 | C4 | -0.177 (n=20) | [-0.270, -0.083] | +0.003 | LESS_DECLINE | -0.159 | LOWER | -0.336 | LOWER | -1.115 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F1 | C5 | +0.057 (n=20) | [-0.050, +0.165] | +0.277 | EQUIVALENT | -0.279 | LOWER | -0.222 | LOWER | +0.049 | EQUIVALENT | FRESH_LEVEL_CONFOUNDED | LEVEL_ONLY |
| F1 | C6 | -0.259 (n=20) | [-0.335, -0.185] | +0.000 | LESS_DECLINE | -0.555 | LOWER | -0.813 | LOWER | -0.604 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F1x | I_P | +0.231 (n=20) | [+0.103, +0.356] | +0.004 | NON_ADDITIVE | +0.209 | NON_ADDITIVE | +0.440 | NON_ADDITIVE | +0.411 | NON_ADDITIVE |  |  |
| F1x | I_V | -0.039 (n=20) | [-0.175, +0.101] | +0.565 | ADDITIVE | +0.062 | ADDITIVE_INCONCLUSIVE | +0.023 | ADDITIVE | -0.107 | ADDITIVE_INCONCLUSIVE |  |  |
| F1x | B_P | -0.599 (n=20) | [-0.706, -0.491] | +0.000 | LESS_DECLINE | -0.801 | LOWER | -1.400 | LOWER | -0.866 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED |  |
| F2 | C7 | +0.482 (n=20) | [+0.349, +0.616] | +0.000 | MORE_DECLINE | -1.968 | LOWER | -1.486 | LOWER | +0.826 | GAP_LARGER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_MORE |
| F2 | C8 | -0.460 (n=20) | [-0.670, -0.252] | +0.000 | LESS_DECLINE | -2.387 | LOWER | -2.848 | LOWER | -0.031 | INCONCLUSIVE | FRESH_LEVEL_CONFOUNDED | INCONCLUSIVE |
| F2 | C7s | +0.650 (n=20) | [+0.544, +0.755] | +0.000 | MORE_DECLINE | -2.007 | LOWER | -1.358 | LOWER | +2.152 | GAP_LARGER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_MORE |
| F2 | C8s | -0.100 (n=20) | [-0.276, +0.076] | +0.249 | INCONCLUSIVE | -2.061 | LOWER | -2.160 | LOWER | +1.846 | GAP_LARGER | FRESH_LEVEL_CONFOUNDED | LEVEL_DIFF_DECLINE_INCONCLUSIVE |
| F3 | C9 | -1.125 (n=20) | [-1.209, -1.041] | +0.000 | LESS_DECLINE | +0.479 | HIGHER | -0.646 | LOWER | -1.072 | GAP_SMALLER | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_LESS |
| F3 | C9n | -0.781 (n=20) | [-0.903, -0.660] | +0.000 | LESS_DECLINE | -2.277 | LOWER | -3.058 | LOWER | -0.128 | INCONCLUSIVE | FRESH_LEVEL_CONFOUNDED | INCONCLUSIVE |

## §6.6 答え

- peak: `NON_ADDITIVE`
- valley: `ORIGIN`
- structure_C7s: `STRUCTURE`
- structure_C8s: `STRUCTURE_INCONCLUSIVE`

## M1 位相の効き目（Φ = 2|sin(Δθ/2)|·medianᵢ Aᵢ − IQRᵢ ḡᵢ^ref、窓内タスク平均）

| 対・層・窓 | n | Φ 平均 | 95% CI | ラベル |
|---|---:|---:|---|---|
| C1_L1_base | 20 | +0.356 | [+0.349, +0.364] | PHASE_ACTIVE |
| C1_L1_late | 20 | +0.118 | [+0.115, +0.120] | PHASE_ACTIVE |
| C1_L2_base | 20 | +0.047 | [+0.041, +0.053] | PHASE_ACTIVE |
| C1_L2_late | 20 | +0.045 | [+0.042, +0.047] | PHASE_ACTIVE |
| C2_L1_base | 20 | +0.486 | [+0.479, +0.492] | PHASE_ACTIVE |
| C2_L1_late | 20 | +0.158 | [+0.155, +0.161] | PHASE_ACTIVE |
| C2_L2_base | 20 | +0.158 | [+0.147, +0.170] | PHASE_ACTIVE |
| C2_L2_late | 20 | +0.046 | [+0.044, +0.048] | PHASE_ACTIVE |
| C7_L1_base | 20 | +0.526 | [+0.518, +0.534] | PHASE_ACTIVE |
| C7_L1_late | 20 | +0.560 | [+0.555, +0.565] | PHASE_ACTIVE |
| C7_L2_base | 20 | +0.441 | [+0.431, +0.450] | PHASE_ACTIVE |
| C7_L2_late | 20 | +0.503 | [+0.495, +0.511] | PHASE_ACTIVE |
| C8_L1_base | 20 | +0.546 | [+0.539, +0.553] | PHASE_ACTIVE |
| C8_L1_late | 20 | +0.573 | [+0.568, +0.578] | PHASE_ACTIVE |
| C8_L2_base | 20 | +0.481 | [+0.472, +0.490] | PHASE_ACTIVE |
| C8_L2_late | 20 | +0.544 | [+0.536, +0.552] | PHASE_ACTIVE |
| C7s_L1_base | 20 | +0.488 | [+0.477, +0.498] | PHASE_ACTIVE |
| C7s_L1_late | 20 | +0.551 | [+0.544, +0.559] | PHASE_ACTIVE |
| C7s_L2_base | 20 | +0.424 | [+0.411, +0.436] | PHASE_ACTIVE |
| C7s_L2_late | 20 | +0.502 | [+0.495, +0.509] | PHASE_ACTIVE |
| C8s_L1_base | 20 | +0.541 | [+0.532, +0.550] | PHASE_ACTIVE |
| C8s_L1_late | 20 | +0.546 | [+0.540, +0.552] | PHASE_ACTIVE |
| C8s_L2_base | 20 | +0.448 | [+0.436, +0.460] | PHASE_ACTIVE |
| C8s_L2_late | 20 | +0.548 | [+0.541, +0.555] | PHASE_ACTIVE |

## CT1 成長（ΔlogN、t101–120、マージン ±0.09877）

| 対比・層 | n | ΔlogN | 95% CI | p_Holm | ラベル |
|---|---:|---:|---|---:|---|
| C1_L1 | 20 | +0.120 | [+0.117, +0.122] | +0.000 | GROWTH_ENHANCED |
| C2_L1 | 20 | +0.084 | [+0.081, +0.087] | +0.000 | GROWTH_ENHANCED_BELOW_CONDA_SCALE |
| C7_L1 | 20 | +0.183 | [+0.181, +0.185] | +0.000 | GROWTH_ENHANCED |
| C8_L1 | 20 | +0.215 | [+0.202, +0.228] | +0.000 | GROWTH_ENHANCED |
| C1_L2 | 20 | +0.039 | [+0.032, +0.047] | +0.000 | GROWTH_ENHANCED_BELOW_CONDA_SCALE |
| C2_L2 | 20 | +0.190 | [+0.184, +0.197] | +0.000 | GROWTH_ENHANCED |
| C7_L2 | 20 | +0.138 | [+0.130, +0.147] | +0.000 | GROWTH_ENHANCED |
| C8_L2 | 20 | +0.312 | [+0.288, +0.336] | +0.000 | GROWTH_ENHANCED |

## REPORT（ラベルなし）

### 腕ごと

| 腕 | n | L 互換（中央値 t16–20 − 中央値 t101–120、seed 中央値） | D 平均 | A_late 平均 | Gap 平均 | 層1 log成長比 SD（t120/t0） | 層1 strict 縮小割合 | Spearman(c1,c120) | logN1 late | logN2 late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LIN | 20 | +0.347 | +0.226 | +88.706 | +0.721 | +0.041 | +0.000 | +0.038 | +1.930 | -0.105 |
| LR | 20 | +2.102 | +2.009 | +89.965 | +1.119 | +0.040 | +0.000 | -0.020 | +2.092 | +0.906 |
| LR_qKp | 20 | +1.082 | +0.884 | +90.444 | +0.047 | +0.038 | +0.000 | +0.125 | +2.097 | +0.780 |
| LR_qKpn | 20 | +1.495 | +1.229 | +87.688 | +0.992 | +0.070 | +0.000 | +0.126 | +2.354 | +1.051 |
| N06 | 20 | +2.412 | +2.271 | +89.491 | +2.256 | +0.035 | +0.000 | -0.039 | +1.749 | +0.087 |
| P06 | 20 | +2.040 | +1.863 | +88.918 | +1.355 | +0.032 | +0.000 | +0.033 | +1.869 | +0.126 |
| P06c | 20 | +2.130 | +1.809 | +88.868 | +2.059 | +0.029 | +0.000 | +0.034 | +1.883 | +0.148 |
| P06c_k1 | 20 | +1.600 | +1.211 | +88.066 | +1.193 | +0.042 | +0.000 | +0.044 | +1.968 | +0.252 |
| P06i | 20 | +2.280 | +2.093 | +89.332 | +1.140 | +0.038 | +0.000 | +0.022 | +1.786 | +0.091 |
| SNA | 20 | +0.705 | +0.730 | +91.495 | +0.203 | +0.037 | +0.000 | -0.022 | +1.789 | +0.292 |
| SNAP | 20 | +1.450 | +1.212 | +89.528 | +1.029 | +0.026 | +0.000 | -0.042 | +1.972 | +0.431 |
| SNAV | 20 | +0.447 | +0.270 | +89.108 | +0.172 | +0.082 | +0.000 | +0.140 | +2.004 | +0.604 |
| SNAi_P | 20 | +0.735 | +0.563 | +91.535 | -1.124 | +0.036 | +0.000 | -0.015 | +1.819 | +0.318 |
| SNAi_V | 20 | +0.385 | +0.370 | +91.169 | -1.675 | +0.034 | +0.000 | +0.076 | +1.824 | +0.275 |
| V06 | 20 | +2.455 | +2.030 | +88.718 | +1.594 | +0.035 | +0.000 | +0.024 | +1.833 | +0.277 |
| V06c | 20 | +2.485 | +2.328 | +89.211 | +2.305 | +0.036 | +0.000 | -0.015 | +1.780 | +0.140 |
| V06i | 20 | +2.258 | +2.012 | +88.936 | +1.652 | +0.035 | +0.000 | +0.005 | +1.812 | +0.210 |

### 主効果・分解対比の ΔlogN・r_a・起点の過渡（REPORT、95% CI）

- main_S_peak_D: -0.062 [-0.118, -0.005] (n=20)
- main_K_peak_D: -0.346 [-0.418, -0.274] (n=20)
- main_S_peak_Alate: -0.054 [-0.101, -0.008] (n=20)
- main_K_peak_Alate: -0.518 [-0.561, -0.477] (n=20)
- main_S_valley_D: -0.278 [-0.342, -0.215] (n=20)
- main_K_valley_D: +0.038 [-0.023, +0.100] (n=20)
- main_S_valley_Alate: -0.524 [-0.566, -0.482] (n=20)
- main_K_valley_Alate: -0.249 [-0.277, -0.221] (n=20)
- dlogN_C3_L1: +0.133 [+0.130, +0.136] (n=20)
- dlogN_C3_L2: +0.061 [+0.053, +0.069] (n=20)
- dlogN_C4_L1: +0.037 [+0.034, +0.040] (n=20)
- dlogN_C4_L2: +0.004 [-0.005, +0.013] (n=20)
- dlogN_C5_L1: +0.030 [+0.027, +0.033] (n=20)
- dlogN_C5_L2: +0.053 [+0.047, +0.060] (n=20)
- dlogN_C6_L1: +0.063 [+0.060, +0.066] (n=20)
- dlogN_C6_L2: +0.123 [+0.115, +0.130] (n=20)
- r_a_P06c_L1_t1: +0.524 [+0.504, +0.543] (n=20)
- r_a_P06c_L1_base: +0.203 [+0.178, +0.227] (n=20)
- r_a_P06c_L1_late: -0.005 [-0.025, +0.016] (n=20)
- r_a_P06c_L2_t1: +0.887 [+0.868, +0.906] (n=20)
- r_a_P06c_L2_base: +0.373 [+0.346, +0.400] (n=20)
- r_a_P06c_L2_late: +0.291 [+0.271, +0.311] (n=20)
- r_a_P06c_k1_L1_t1: +1.157 [+1.149, +1.164] (n=20)
- r_a_P06c_k1_L1_base: +0.534 [+0.513, +0.555] (n=20)
- r_a_P06c_k1_L1_late: +0.176 [+0.165, +0.188] (n=20)
- r_a_P06c_k1_L2_t1: +0.702 [+0.668, +0.736] (n=20)
- r_a_P06c_k1_L2_base: -0.036 [-0.063, -0.009] (n=20)
- r_a_P06c_k1_L2_late: +0.089 [+0.062, +0.116] (n=20)
- r_a_V06c_L1_t1: +1.012 [+0.979, +1.045] (n=20)
- r_a_V06c_L1_base: +0.688 [+0.633, +0.743] (n=20)
- r_a_V06c_L1_late: +0.308 [+0.252, +0.364] (n=20)
- r_a_V06c_L2_t1: +1.374 [+1.337, +1.411] (n=20)
- r_a_V06c_L2_base: +0.739 [+0.672, +0.806] (n=20)
- r_a_V06c_L2_late: +0.245 [+0.119, +0.370] (n=20)
- r_a_LR_qKp_L1_t1: +0.621 [+0.606, +0.635] (n=20)
- r_a_LR_qKp_L1_base: +0.157 [+0.143, +0.172] (n=20)
- r_a_LR_qKp_L1_late: +0.143 [+0.119, +0.167] (n=20)
- r_a_LR_qKp_L2_t1: +0.915 [+0.903, +0.928] (n=20)
- r_a_LR_qKp_L2_base: +0.741 [+0.730, +0.753] (n=20)
- r_a_LR_qKp_L2_late: +0.734 [+0.714, +0.753] (n=20)
- r_a_LR_qKpn_L1_t1: +1.478 [+1.459, +1.496] (n=20)
- r_a_LR_qKpn_L1_base: +1.106 [+1.089, +1.123] (n=20)
- r_a_LR_qKpn_L1_late: +0.776 [+0.759, +0.794] (n=20)
- r_a_LR_qKpn_L2_t1: +1.049 [+1.028, +1.069] (n=20)
- r_a_LR_qKpn_L2_base: +0.239 [+0.216, +0.262] (n=20)
- r_a_LR_qKpn_L2_late: -0.042 [-0.058, -0.027] (n=20)
- transient_e_P06i-N06: +0.041 [+0.014, +0.068] (n=20)
- transient_e_P06-P06c: +0.471 [+0.369, +0.572] (n=20)
- transient_e_V06i-N06: -0.295 [-0.333, -0.256] (n=20)
- transient_e_V06-V06c: -0.248 [-0.290, -0.206] (n=20)

腕間の幅の差（CT1・logN）は W 病理の証拠ではない（spec §6.7・§14）。EQUIVALENT は「単走のタスク標本の分解能より小さい」の意味。
