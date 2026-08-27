# M2: Delta-L 群と開口量の動的変化

- **M2_dynamics: HIGH_LESS_PUSHED**
- **M2_baseline: INCONCLUSIVE_BASELINE**
- **M2_combined: DYNAMIC_DIFFERENCE_HIGH_LESS_PUSHED**

## 主結果

- A_deltaS = +0.155695 [+0.146237, +0.180084]
- A_S0 = +0.042295 [+0.031337, +0.058090]
- raw D_deltaS = +0.0771192594 [+0.0616342927, +0.0859033159]
- raw D_S0 = +0.00604811925 [+0.00357620034, +0.00704168274]

A は同じ作業6幾何セル内の優越確率から0.5を引いた量。等価域は +/-0.05。

## 固定集合と再生

- risk exposure: 15,582
- valid cells: 2,839
- low/high: 5,023 / 4,431
- replay implementation: `374e60b`
- 元走の最終 complete-state hash と全既存 landmark 列: **完全一致**
- M2-S1..S9: **PASS**

## 解釈上限

- 同じ軌道の読み取り専用再計装であり、独立 replication ではない。
- 固定幾何セル内の観察的関連であり、Delta-L の因果効果ではない。
- 作業6 PROTECTIVE と r-swap SPECIFIC の判定を差し替えない。
