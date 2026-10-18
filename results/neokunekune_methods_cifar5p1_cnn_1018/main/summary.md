# neokunekune_methods_cifar5p1_cnn_1018 — 判定

束の状態: A `COMPLETED`、B `COMPLETED`、C `COMPLETED`。
完走していない腕: なし。

## 主量（後期 hard t=21,23,25,27,29 の online、seed 平均）と選ばれた設定

| 順位 | 腕 | 主量 (%) | 全 hard (%) | 早期 − 後期 (pt) | 最初 100 更新 (%) | test 後期 (%) | 設定 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | ELU | 83.99 | 82.67 | -3.24 | 54.60 | 77.13 | `ELU_a3.6_lr0.0003` |
| 2 | SNAfrz1 | 80.79 | 78.93 | -4.49 | 50.99 | 78.22 | `SNAfrz1_lr0.0003` |
| 3 | SNA | 80.01 | 77.88 | -4.51 | 50.85 | 77.98 | `SNA_lr0.0003` |
| 4 | SN06 | 79.88 | 78.74 | -2.59 | 48.07 | 78.05 | `SN06_lr0.0003` |
| 5 | NEO | 79.42 | 77.53 | -3.33 | 51.09 | 76.80 | `NEO_lr0.001` |
| 6 | KKT1 | 78.92 | 77.10 | -3.61 | 49.45 | 77.36 | `KKT1_lr0.0003` |
| 7 | LK | 78.39 | 76.47 | -4.25 | 53.93 | 77.17 | `LK_a0.8_lr0.001` |
| 8 | RSL | 78.23 | 76.52 | -3.91 | 52.58 | 76.65 | `RSL_c0.5_p0.5_l0.673_u2.673_lr0.001` |
| 9 | SN3 | 75.98 | 73.92 | -4.27 | 51.32 | 76.59 | `SN3_lr0.001` |
| 10 | L2I | 71.94 | 70.79 | -2.18 | 42.27 | 77.28 | `L2I_lam0.01_lr0.001` |
| 11 | R | 48.38 | 51.69 | +8.53 | 2.71 | 65.66 | `R_lr0.0003` |
| 12 | ReDo | 47.12 | 44.82 | +1.56 | 21.76 | 62.86 | `ReDo_t0_F1560_lr0.0003` |

## 主の族: NEO − X（Bonferroni 8、99.375% 区間）

| 比較 | 平均 (pt) | 区間 (pt) | 正の seed | 判定 | 0.5 pt の非劣性 |
|---|---:|---|---:|---|---|
| NEO_minus_SN06 | -0.45 | [-2.14, +0.88] | 5/10 | `INCONCLUSIVE` | 満たさない |
| NEO_minus_SN3 | +3.44 | [+2.16, +4.57] | 10/10 | `BETTER_THAN_SN3` | 満たす |
| NEO_minus_KKT1 | +0.50 | [-1.13, +1.84] | 7/10 | `INCONCLUSIVE` | 満たさない |
| NEO_minus_ELU | -4.56 | [-6.26, -3.34] | 0/10 | `WORSE_THAN_ELU` | 満たさない |
| NEO_minus_LK | +1.03 | [-0.50, +2.33] | 8/10 | `INCONCLUSIVE` | 満たさない |
| NEO_minus_RSL | +1.20 | [-0.40, +2.41] | 8/10 | `INCONCLUSIVE` | 満たす |
| NEO_minus_ReDo | +32.30 | [+29.01, +36.09] | 10/10 | `BETTER_THAN_ReDo` | 満たす |
| NEO_minus_L2I | +7.48 | [+6.02, +8.68] | 10/10 | `BETTER_THAN_L2I` | 満たす |

## 副の族: SNAfrz1 − X（Bonferroni 8、99.375% 区間）

| 比較 | 平均 (pt) | 区間 (pt) | 正の seed | 判定 | 0.5 pt の非劣性 |
|---|---:|---|---:|---|---|
| SNAfrz1_minus_SN06 | +0.91 | [+0.10, +1.67] | 8/10 | `BETTER_THAN_SN06` | 満たす |
| SNAfrz1_minus_SN3 | +4.81 | [+4.12, +5.30] | 10/10 | `BETTER_THAN_SN3` | 満たす |
| SNAfrz1_minus_KKT1 | +1.87 | [+1.26, +2.63] | 10/10 | `BETTER_THAN_KKT1` | 満たす |
| SNAfrz1_minus_ELU | -3.20 | [-4.01, -2.39] | 0/10 | `WORSE_THAN_ELU` | 満たさない |
| SNAfrz1_minus_LK | +2.40 | [+1.51, +3.19] | 10/10 | `BETTER_THAN_LK` | 満たす |
| SNAfrz1_minus_RSL | +2.56 | [+1.60, +3.49] | 10/10 | `BETTER_THAN_RSL` | 満たす |
| SNAfrz1_minus_ReDo | +33.67 | [+30.54, +37.05] | 10/10 | `BETTER_THAN_ReDo` | 満たす |
| SNAfrz1_minus_L2I | +8.85 | [+7.79, +9.94] | 10/10 | `BETTER_THAN_L2I` | 満たす |

選んだ値が格子の端だった腕（`EDGE`）: NEO（lr=0.001）、SN3（lr=0.001）、ELU（a=3.6 (two-value grid)）、LK（lr=0.001・a=0.8）、RSL（lr=0.001）、L2I（lr=0.001）。格子は広げていない。

S-nochange-main: 適用外（bundle A did not select lr 1e-4 for all four arms）。

主量は online（更新前）精度。test 精度・fresh・幅は別の読みで、確認的判定ではない。数値の全量は `verdict.json`。
