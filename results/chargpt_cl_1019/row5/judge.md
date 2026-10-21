# 追補 9 の判定（2 ブロックで L0 の応答だけを操作する介入）

## 主：確認的な族（課題 91〜100 の seed 内平均 E1000、両側 exact 符号検定、Holm）

- C3self − betaFA：正の seed 0/0（非有限 10）、p = 1.0000、Holm 後 1.0000、平均 +nan（区間 +nan〜+nan）、中央 +nan → **効果を検出せず**
  - seed ごとの差: +nan +nan +nan +nan +nan +nan +nan +nan +nan +nan
- C3self − Fswap：正の seed 0/0（非有限 10）、p = 1.0000、Holm 後 1.0000、平均 +nan（区間 +nan〜+nan）、中央 +nan → **効果を検出せず**
  - seed ごとの差: +nan +nan +nan +nan +nan +nan +nan +nan +nan +nan
- C3self − Fswapscale：正の seed 0/0（非有限 10）、p = 1.0000、Holm 後 1.0000、平均 +nan（区間 +nan〜+nan）、中央 +nan → **効果を検出せず**
  - seed ごとの差: +nan +nan +nan +nan +nan +nan +nan +nan +nan +nan
- Frev − capself：正の seed 0/0（非有限 10）、p = 1.0000、Holm 後 1.0000、平均 +nan（区間 +nan〜+nan）、中央 +nan → **効果を検出せず**
  - seed ごとの差: +nan +nan +nan +nan +nan +nan +nan +nan +nan +nan

| seed | C3self | capself | betaFA | Fswap | Frev | Fswapscale | betaF |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | nan | nan | nan | nan | nan | nan | nan |
| 1 | nan | nan | nan | nan | nan | nan | nan |
| 2 | nan | 0.136 | nan | nan | nan | nan | nan |
| 3 | nan | nan | nan | nan | nan | nan | nan |
| 4 | nan | nan | nan | nan | nan | nan | nan |
| 5 | nan | nan | nan | nan | nan | nan | nan |
| 6 | nan | nan | nan | nan | nan | nan | nan |
| 7 | nan | nan | nan | nan | nan | nan | nan |
| 8 | nan | nan | nan | nan | nan | nan | nan |
| 9 | nan | nan | nan | nan | nan | nan | nan |

## 操作の成立

- main β-FA：D が空でない seed 0 のうち、検証集合で D の unit の過半が p*/2 以上で開いた seed 0。D の大きさ []。D の実効の開く割合の平均 []
- side β-FA：D が空でない seed 0 のうち、検証集合で D の unit の過半が p*/2 以上で開いた seed 0。D の大きさ []。D の実効の開く割合の平均 []

## 副（記述のみ）

### main（課題 91〜100、10 seed の中央値）

| 腕 | T*（上限 10,000） | 未到達率 | E1000 | ELast | 最初の課題の E1000 | 有限 |
| -- | -- | -- | -- | -- | -- | -- |
| capself | 764 | 0.00 | 0.136 | 0.082 | 0.130 | 1/1 |

### side（課題 51〜60、10 seed の中央値）

| 腕 | T*（上限 10,000） | 未到達率 | E1000 | ELast | 最初の課題の E1000 | 有限 |
| -- | -- | -- | -- | -- | -- | -- |

- 対照間差の回復比 G = (C3self − 腕) / (C3self − capself)（媒介率ではない）の中央値：betaFA nan、Fswap nan、Fswapscale nan
- F-swap − F-swap-scale の直接差：平均 +nan・中央 +nan（正の seed 0/10）

## ELR（9.C、別の確認的比較）

- seed 10/10 がそろった。3 相すべてで C3-2 より良い seed 10、すべて悪い seed 0 → **ELR_RESCUES**
  - T015：ELR − C3-2 の中央 -8832.450（正 0/10、p = 0.0020）、ELR − ReLUcap-2 の中央 +147.200
  - E1000：ELR − C3-2 の中央 -1.125（正 0/10、p = 0.0020）、ELR − ReLUcap-2 の中央 +0.010
  - ELast：ELR − C3-2 の中央 -0.274（正 0/10、p = 0.0020）、ELR − ReLUcap-2 の中央 -0.018
- 操作の成立：回転が C3-2 の 10 倍以上 10/10、L0 の出力の RMS が C3-2 の ±20% 以内 0/10
- 課題 100 の L0 の死：[0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]（0.1 以下 10/10）
- 文字どおりの形の診断（seed 0）：L0 の ‖w‖ の中央値の、課題 1 末に対する比 = t2:1.9 t3:4.48 t4:10.9 t5:22.7 t6:46.7 t7:97.6 t8:194 t9:397 t10:766 t11:1.53e+03 t12:2.99e+03 t13:5.82e+03 t14:1.15e+04 t15:2.06e+04 t16:3.22e+04 t17:4.82e+04 t18:6.09e+04 t19:7.35e+04 t20:8.35e+04