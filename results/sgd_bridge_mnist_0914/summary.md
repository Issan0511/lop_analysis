# sgd_bridge_mnist_0914 summary

spec: `specs/spec_sgd_bridge_mnist_0914.md`。箱 B（0914 と同一）の optimizer だけを plain SGD（lr 0.02・2,500 更新/タスク）に替えた 5 腕 × seed 0–9。ΔlogN は log（t101–120 の二乗平均平方根の中心化ノルム）、E は pt。

## X：optimizer × 位相（見出し）— ΔlogN(SGD) − ΔlogN(Adam 0914)、seed 対

| 対比・層 | n | X | 95% CI | p_Holm | ラベル | 修飾 | ΔlogN Adam（seed 0–9） | Adam の登録ラベル（0914, 20 seed） | SGD の G ラベル |
|---|---:|---:|---|---:|---|---|---:|---|---|
| C1_L1 | 10 | -0.083 | [-0.089, -0.077] | +0.008 | X_NEGATIVE |  | +0.118 | GROWTH_ENHANCED | GROWTH_ENHANCED_BELOW_CONDA_SCALE |
| C1_L2 | 10 | -0.240 | [-0.254, -0.226] | +0.008 | X_NEGATIVE | SIGN_REVERSED | +0.035 | GROWTH_ENHANCED_BELOW_CONDA_SCALE | GROWTH_SUPPRESSED |
| C2_L1 | 3 | -1.142 | [-inf, inf] | +0.500 | INCOMPLETE |  | +0.084 | GROWTH_ENHANCED_BELOW_CONDA_SCALE | INCOMPLETE |
| C2_L2 | 3 | -0.788 | [-inf, inf] | +0.500 | INCOMPLETE |  | +0.187 | GROWTH_ENHANCED | INCOMPLETE |

## G：SGD 下の成長差（ΔlogN、t101–120、マージン ±0.09877）

| 対比・層 | n | ΔlogN | 95% CI | p_Holm | ラベル |
|---|---:|---:|---|---:|---|
| C1_L1 | 10 | +0.035 | [+0.031, +0.039] | +0.008 | GROWTH_ENHANCED_BELOW_CONDA_SCALE |
| C1_L2 | 10 | -0.204 | [-0.212, -0.196] | +0.008 | GROWTH_SUPPRESSED |
| C2_L1 | 3 | -1.059 | [-inf, inf] | +0.500 | INCOMPLETE |
| C2_L2 | 3 | -0.601 | [-inf, inf] | +0.500 | INCOMPLETE |

## F：個体の分岐（strict 縮小の割合 f、単峰近似からの超過 e）

| 腕・層 | n | f 平均 | f 95% CI | e 平均 | e 95% CI | ラベル |
|---|---:|---:|---|---:|---|---|
| N06_L1 | 10 | +0.000 | [-0.000, +0.000] | +0.000 | [-0.000, +0.000] | FORK_ABSENT |
| N06_L2 | 10 | +0.000 | [-0.000, +0.000] | -0.000 | [-0.000, +0.000] | FORK_ABSENT |
| P06_L1 | 10 | +0.000 | [-0.000, +0.000] | +0.000 | [-0.000, +0.000] | FORK_ABSENT |
| P06_L2 | 10 | +0.000 | [-0.000, +0.000] | -0.000 | [-0.000, +0.000] | FORK_ABSENT |
| V06_L1 | 3 | +0.060 | [-inf, inf] | -0.222 | [-inf, inf] | FORK_INCONCLUSIVE |
| V06_L2 | 3 | +0.093 | [-inf, inf] | -0.061 | [-inf, inf] | FORK_INCONCLUSIVE |
| LIN_L1 | 10 | +0.000 | [-0.000, +0.000] | +0.000 | [-0.000, +0.000] | FORK_ABSENT |
| LIN_L2 | 10 | +0.000 | [-0.000, +0.000] | -0.000 | [-0.001, -0.000] | FORK_ABSENT |
| LR_L1 | 10 | +0.000 | [-0.000, +0.000] | -0.000 | [-0.000, +0.000] | FORK_ABSENT |
| LR_L2 | 10 | +0.000 | [-0.000, +0.000] | -0.000 | [-0.000, +0.000] | FORK_ABSENT |

## E：時間劣化・水準・fresh gap（族 FS、m=2）

TESTABLE_FIXED（D_pair N06−LIN）: +0.754 [+0.655, +0.849] → 成立。分解能 h: D +0.133・A_late +0.090・Gap +0.123

| 対比 | E1 D（t16–30 − t101–120） | 95% CI | E1 | ΔA_late | E2 late | ΔA_base | ΔGap | E3 | ΔA_fresh | 修飾 | §6.4 |
|---|---:|---|---|---:|---|---:|---:|---|---:|---|---|
| C1 | +0.191 | [+0.051, +0.332] | MORE_DECLINE | -0.957 | LOWER | -0.766 | +0.215 | GAP_LARGER | -0.741 | FRESH_LEVEL_CONFOUNDED | TEMPORAL_LOP_MORE |
| C2 | -64.488 | [-inf, inf] | INCOMPLETE | -18.416 | INCOMPLETE | -82.903 | -62.263 | INCOMPLETE | -80.679 |  | None |

## REPORT

| 腕 | n | D | A_late | Gap | 層1 成長倍率 | 層2 成長倍率 | logN1 − Adam | logN2 − Adam | log成長比 SD 層1 | 層2 | 最小倍率 層1 | 層2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N06 | 10 | +0.926 | +93.303 | -1.142 | +4.867 | +2.359 | -0.717 | +0.214 | +0.058 | +0.117 | +3.662 | +1.407 |
| P06 | 10 | +1.117 | +92.346 | -0.927 | +5.040 | +1.923 | -0.800 | -0.026 | +0.037 | +0.092 | +4.074 | +1.329 |
| V06 | 3 | -63.523 | +74.873 | -63.523 | +1.688 | +1.300 | -1.858 | -0.561 | +0.582 | +0.299 | +0.991 | +0.961 |
| LIN | 10 | +0.173 | +90.194 | +0.352 | +4.273 | +1.397 | -1.028 | -0.115 | +0.062 | +0.097 | +3.647 | +1.041 |
| LR | 10 | +1.401 | +92.399 | -1.637 | +5.290 | +3.214 | -0.977 | -0.295 | +0.261 | +0.281 | +1.496 | +1.153 |

- M1_C1_L1_base: +0.899 [+0.872, +0.924] PHASE_ACTIVE
- M1_C1_L1_late: +0.686 [+0.677, +0.694] PHASE_ACTIVE
- M1_C1_L2_base: +0.144 [+0.120, +0.168] PHASE_ACTIVE
- M1_C1_L2_late: +0.080 [+0.066, +0.094] PHASE_ACTIVE
- M1_C2_L1_base: +1.162 [-inf, inf] PHASE_BORDERLINE
- M1_C2_L1_late: +1.263 [-inf, inf] PHASE_BORDERLINE
- M1_C2_L2_base: +1.206 [-inf, inf] PHASE_BORDERLINE
- M1_C2_L2_late: +0.954 [-inf, inf] PHASE_BORDERLINE

X は optimizer と更新数（625→2,500）を合わせて替えた効果。腕間の幅の差は W 病理の証拠ではない。
