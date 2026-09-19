# relu_doors_0919 -- ReLU が沈む道を 1 本ずつ塞ぐ

入力は全腕 raw、活性化は全腕 ReLU、seed 0-9、50 タスク。`ref` は `rlcifar_mlp_battle_0918` の `R` raw の再利用。

## 窓（t31-50 の online、seed 中央値）

| 腕 | 窓 | seed の範囲 | 窓 >= 0.5 の seed | 低下 | t50 の第2層ゲート厳密0 |
|---|---|---|---|---|---|
| `ref` | 0.1132 | 0.1120-0.1156 | **0/10** | +0.0032 | n/a |
| `C` | 0.1133 | 0.1120-0.1156 | **0/10** | +0.3917 | 1.000 |
| `CH` | 0.9867 | 0.9861-0.9878 | **10/10** | -0.0100 | 0.566 |
| `CHB` | 0.9854 | 0.9834-0.9884 | **10/10** | -0.0099 | 0.484 |
| `CHB0` | 0.9868 | 0.9836-0.9894 | **10/10** | -0.0097 | 0.491 |

## 登録判定

- **M**: `RESCUED` — rescued_seeds 10, median_window 0.9854
- **N**: `NEED_CH` — rescued_seeds {'ref': 0, 'C': 0, 'CH': 10, 'CHB': 10, 'CHB0': 10}
- **D1**: `TIE` — median -0.0000, pos 1, n 6, p 0.2188
- **D2**: `H_HELPS` — median 0.8739, pos 10, n 10, p 0.0020
- **D3**: `TIE` — median -0.0015, pos 3, n 10, p 0.3438
- **D4**: `TIE` — median 0.0008, pos 8, n 10, p 0.1094
- **R1**: `L1_SAVED` — median_dead_frac_l1_t1 0.0000, ref_t1 0.9850
- **R2**: `L2_HELD` — median_sink_ratio_l2_t50 -0.1907
- **B_ROUTE**: `BIAS_TAKES_OVER` — C_median 0.0019, CH_median 0.1811, median 0.1791, pos 10, n 10, p 0.0020
- **SKEW_ROUTE**: `NO_SKEW_ROUTE` — median_onesided_frac_l1_t50 {'CHB': 0.0, 'CHB0': 0.0}
- **ZBAR1**: `PINNED` — max_abs_zbar_l1 {'CHB': 4.720052311e-05, 'CHB0': 2.441406286e-05}

- P14 の反例（ゲート 1.00 到達後に窓 > 0.5 に戻った走）: {'ref': 0, 'C': 0, 'CH': 0, 'CHB': 0, 'CHB0': 0}

## 予測の採点

**Claude 8/11, Brier 0.176**

| key | 主張 | p | 的中 |
|---|---|---|---|
| P1 | M == RESCUED | 0.65 | yes |
| P2 | C alone collapses | 0.85 | yes |
| P5 | CH collapses (b2 takes the door over) | 0.60 | no |
| P6 | B_ROUTE == BIAS_TAKES_OVER | 0.60 | yes |
| P7 | R1 == L1_SAVED | 0.80 | yes |
| P8 | R2 == L2_HELD | 0.45 | yes |
| P9 | SKEW_ROUTE == SKEW_TAKES_OVER | 0.20 | no |
| P10 | D4 == TIE (removal is no better than the derived WD) | 0.55 | yes |
| P11 | N == NEED_CHB | 0.50 | no |
| P13 | if CHB is rescued, its window beats leaky 0.1 raw (0.726) | 0.35 | yes |
| P14 | no run recovers after gate_zero_frac_l2 reaches 1.00 | 0.90 | yes |

**Issa 4/4**

| key | 主張 | 的中 |
|---|---|---|
| I1 | M == RESCUED | yes |
| I2 | N in (NEED_CH, NEED_CHB) | yes |
| I3 | B_ROUTE == BIAS_TAKES_OVER | yes |
| I4 | SKEW_ROUTE == NO_SKEW_ROUTE | yes |
