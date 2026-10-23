# chargpt_seed10_1022 の判定（judge.py）

- 作成 2026-10-23 13:38:06 +0900・script sha256 193f8eb67209ad73…・spec sha256 72b9e065df166c42…
- 規則・閾値・窓は spec のまま。予測は spec の表の Opus 5.5 の欄を採点しただけ。「未完」= 出力が揃っていない単位がある（その部分のラベルは出さない）。

## ラベル

| 部分 | ラベル | 数 | 符号検定の p（両側 exact） | spec |
| --- | --- | --- | --- | --- |
| A1 検算 | PASS（700/700 で κ > 0・破れ 0） | κ > 0 700/700・破れ 0 | — | :48 |
| A2（主） | LOAD_ALIGNS | s_early > s_init 10/10（+10 −0 0:0） | 0.0020 | :49-50 |
| A3 | INIT_PUSH_COMPARABLE | r_mag ≥ 0.1 10/10 | — | :52 |
| B（C3-2 L1 計量 D 早期） | OTHER_UNITS_CARRY | ③a 10・①+② 0・③b 0 / 10 | — | :81-84 |
| C1（lnex1） | FEEDBACK_NEEDED | 生き返らない 5/6（対象外 3・未完 0） | 窓 S+1〜E: 0.0312（+6 −0） | :127-131・:241-242 |
| C2（gamma03f） | READOUT_SHRINK_REVIVES | 26 末 ≤ 0.5 10/10（資格外 0・未完 0） | — | :138・:243 |
| D（SiLU） | D1 NO_ESCAPE・D2 not SINK_MOVES・D3 LOSS_REPLICATES | D1 2/10・D2 1/2・D3 9/10 | — | :170-174 |
| D（GELU） | D1 未実施・D2 未実施・D3 未実施 | 出力なし | — | :170-174 |
| D（SiLUcap） | D1 未実施・D2 未実施・D3 未実施 | 出力なし | — | :170-174 |

## 記録

### A（C3-2 の L1、seed ごと）

| seed | n_live init | s_init | 帯 ±（内） | s_early | s 39〜50 | s 89〜100 | r_mag | r_mag（生きた unit） | full s early | fixed-next s early |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3/seed0 | 512 | 0.469 | 0.066（True） | 0.825 | 0.522 | 0.561 | 1.811 | 1.811 | 1.000 | 0.665 |
| C3/seed1 | 512 | 0.514 | 0.066（True） | 0.835 | 0.522 | 0.518 | 1.902 | 1.902 | 1.000 | 0.655 |
| C3/seed2 | 512 | 0.490 | 0.066（True） | 0.867 | 0.547 | 0.599 | 1.708 | 1.708 | 1.000 | 0.689 |
| C3/seed3 | 512 | 0.504 | 0.066（True） | 0.850 | 0.503 | 0.665 | 1.871 | 1.871 | 1.000 | 0.688 |
| C3/seed4 | 512 | 0.482 | 0.066（True） | 0.840 | 0.502 | 0.576 | 1.996 | 1.996 | 1.000 | 0.664 |
| C3/seed5 | 512 | 0.465 | 0.066（True） | 0.888 | 0.477 | 0.494 | 2.063 | 2.063 | 0.992 | 0.675 |
| C3/seed6 | 512 | 0.494 | 0.066（True） | 0.863 | 0.502 | 0.510 | 1.243 | 1.243 | 1.000 | 0.699 |
| C3/seed7 | 512 | 0.477 | 0.066（True） | 0.850 | 0.486 | 0.460 | 1.890 | 1.890 | 1.000 | 0.682 |
| C3/seed8 | 512 | 0.498 | 0.066（True） | 0.837 | 0.555 | 0.501 | 1.823 | 1.823 | 1.000 | 0.654 |
| C3/seed9 | 512 | 0.482 | 0.066（True） | 0.875 | 0.531 | 0.459 | 1.533 | 1.533 | 1.000 | 0.677 |

- 帯に入る seed 10/10・課題 89〜100 の s の中央値 < 0.6 の seed 9/10。未完の単位: なし。
- ReLUcap-2（L1）と L0: ReLUcap/seed0 s_init 0.469 s_early 0.795 / L0 s_init 0.514 s_early 0.375; ReLUcap/seed1 s_init 0.514 s_early 0.783 / L0 s_init 0.508 s_early 0.348; ReLUcap/seed2 s_init 0.490 s_early 0.855 / L0 s_init 0.510 s_early 0.371; ReLUcap/seed3 s_init 0.504 s_early 0.837 / L0 s_init 0.496 s_early 0.377; ReLUcap/seed4 s_init 0.482 s_early 0.811 / L0 s_init 0.494 s_early 0.334; ReLUcap/seed5 s_init 0.465 s_early 0.775 / L0 s_init 0.525 s_early 0.268; ReLUcap/seed6 s_init 0.494 s_early 0.801 / L0 s_init 0.484 s_early 0.348; ReLUcap/seed7 s_init 0.477 s_early 0.766 / L0 s_init 0.504 s_early 0.375; ReLUcap/seed8 s_init 0.498 s_early 0.811 / L0 s_init 0.475 s_early 0.377; ReLUcap/seed9 s_init 0.482 s_early 0.783 / L0 s_init 0.477 s_early 0.359
- cos²・Q（C3-2、課題 1・10・25・50・100）: C3/seed0 cos² 0.800/0.435/0.460/0.323/0.217; C3/seed1 cos² 0.800/0.591/0.449/0.392/0.002; C3/seed2 cos² 0.784/0.607/0.471/0.346/0.044; C3/seed3 cos² 0.773/0.604/0.463/0.340/0.162; C3/seed4 cos² 0.812/0.593/0.481/0.380/0.287; C3/seed5 cos² 0.786/0.589/0.494/0.424/0.245; C3/seed6 cos² 0.784/0.608/0.483/0.334/0.180; C3/seed7 cos² 0.794/0.599/0.467/0.297/0.241; C3/seed8 cos² 0.818/0.567/0.467/0.360/0.334; C3/seed9 cos² 0.812/0.591/0.457/0.310/0.241

### B（seed ごとの沈む unit の割合の中央値 ③a / ①+② / ③b と最大、ρ の中央値、①②の符号の一致）

- C3: 揃った seed 10・欠け 0・検査落ち 0
  - seed0 L1_D_early: 0.76 / 0.04 / 0.14 → ③a・ρ 0.13・一致 0.95
  - seed0 L1_A_early: 0.73 / 0.03 / 0.16 → ③a・ρ 0.10・一致 0.95
  - seed0 L1_D_mid: 0.01 / 0.00 / 0.98 → ③b・ρ 0.01・一致 0.81
  - seed0 L1_D_late: 0.99 / 0.01 / 0.00 → ③a・ρ 0.06・一致 0.93
  - seed0 L0_D_early: 0.62 / 0.13 / 0.24 → ③a・ρ 0.36・一致 0.96
  - seed0 L1_D_early_alive: 0.76 / 0.05 / 0.14 → ③a・ρ 0.13・一致 0.95
  - seed1 L1_D_early: 0.77 / 0.04 / 0.17 → ③a・ρ 0.08・一致 0.97
  - seed1 L1_A_early: 0.72 / 0.04 / 0.22 → ③a・ρ 0.07・一致 0.96
  - seed1 L1_D_mid: -0.31 / 0.01 / 1.31 → ③b・ρ 0.06・一致 0.90
  - seed1 L1_D_late: 0.55 / 0.03 / 0.41 → ③a・ρ 0.20・一致 0.91
  - seed1 L0_D_early: 0.95 / -0.14 / 0.21 → ③a・ρ 0.27・一致 0.96
  - seed1 L1_D_early_alive: 0.77 / 0.04 / 0.17 → ③a・ρ 0.08・一致 0.97
  - seed2 L1_D_early: 0.77 / 0.04 / 0.17 → ③a・ρ 0.09・一致 0.95
  - seed2 L1_A_early: 0.69 / 0.03 / 0.27 → ③a・ρ 0.07・一致 0.95
  - seed2 L1_D_mid: -0.01 / 0.00 / 1.01 → ③b・ρ 0.01・一致 0.93
  - seed2 L1_D_late: 4.06 / 0.77 / -3.79 → ③a・ρ 0.01・一致 0.72
  - seed2 L0_D_early: 0.98 / -0.18 / 0.17 → ③a・ρ 0.20・一致 0.97
  - seed2 L1_D_early_alive: 0.77 / 0.04 / 0.17 → ③a・ρ 0.09・一致 0.95
  - seed3 L1_D_early: 0.76 / 0.05 / 0.18 → ③a・ρ 0.10・一致 0.95
  - seed3 L1_A_early: 0.65 / 0.04 / 0.28 → ③a・ρ 0.08・一致 0.94
  - seed3 L1_D_mid: 0.06 / 0.00 / 0.94 → ③b・ρ 0.01・一致 0.94
  - seed3 L1_D_late: 2.50 / 0.27 / -1.77 → ③a・ρ 0.02・一致 0.53
  - seed3 L0_D_early: 0.76 / 0.13 / 0.15 → ③a・ρ 0.27・一致 0.96
  - seed3 L1_D_early_alive: 0.76 / 0.05 / 0.18 → ③a・ρ 0.10・一致 0.95
  - seed4 L1_D_early: 0.68 / 0.05 / 0.25 → ③a・ρ 0.13・一致 0.96
  - seed4 L1_A_early: 0.69 / 0.06 / 0.22 → ③a・ρ 0.11・一致 0.96
  - seed4 L1_D_mid: -0.05 / 0.01 / 1.05 → ③b・ρ 0.01・一致 0.71
  - seed4 L1_D_late: 0.91 / 0.04 / -0.07 → ③a・ρ 0.15・一致 0.94
  - seed4 L0_D_early: 0.58 / 0.01 / 0.40 → ③a・ρ 0.33・一致 0.95
  - seed4 L1_D_early_alive: 0.68 / 0.05 / 0.25 → ③a・ρ 0.13・一致 0.96
  - seed5 L1_D_early: 0.69 / 0.04 / 0.23 → ③a・ρ 0.16・一致 0.96
  - seed5 L1_A_early: 0.72 / 0.05 / 0.19 → ③a・ρ 0.17・一致 0.96
  - seed5 L1_D_mid: 0.70 / 0.13 / 0.17 → ③a・ρ 0.32・一致 0.95
  - seed5 L1_D_late: 3.09 / 0.21 / -2.21 → ③a・ρ 0.17・一致 0.89
  - seed5 L0_D_early: 0.76 / -0.02 / 0.24 → ③a・ρ 0.39・一致 0.94
  - seed5 L1_D_early_alive: 0.69 / 0.04 / 0.23 → ③a・ρ 0.16・一致 0.96
  - seed6 L1_D_early: 0.77 / 0.06 / 0.16 → ③a・ρ 0.15・一致 0.96
  - seed6 L1_A_early: 0.77 / 0.05 / 0.13 → ③a・ρ 0.15・一致 0.96
  - seed6 L1_D_mid: -0.12 / 0.00 / 1.12 → ③b・ρ 0.01・一致 0.83
  - seed6 L1_D_late: 0.97 / 0.02 / 0.01 → ③a・ρ 0.18・一致 0.93
  - seed6 L0_D_early: 0.57 / 0.14 / 0.15 → ③a・ρ 0.40・一致 0.96
  - seed6 L1_D_early_alive: 0.77 / 0.06 / 0.16 → ③a・ρ 0.15・一致 0.96
  - seed7 L1_D_early: 0.75 / 0.04 / 0.17 → ③a・ρ 0.10・一致 0.95
  - seed7 L1_A_early: 0.70 / 0.04 / 0.26 → ③a・ρ 0.08・一致 0.95
  - seed7 L1_D_mid: 0.08 / 0.00 / -0.07 → ③a・ρ 0.06・一致 0.93
  - seed7 L1_D_late: 0.85 / 0.00 / 0.15 → ③a・ρ 0.21・一致 0.83
  - seed7 L0_D_early: 0.93 / -0.06 / 0.14 → ③a・ρ 0.25・一致 0.96
  - seed7 L1_D_early_alive: 0.75 / 0.05 / 0.17 → ③a・ρ 0.10・一致 0.95
  - seed8 L1_D_early: 0.67 / 0.05 / 0.26 → ③a・ρ 0.12・一致 0.96
  - seed8 L1_A_early: 0.69 / 0.05 / 0.25 → ③a・ρ 0.10・一致 0.95
  - seed8 L1_D_mid: 0.05 / 0.00 / 0.95 → ③b・ρ 0.01・一致 0.88
  - seed8 L1_D_late: 0.01 / 0.05 / 0.95 → ③b・ρ 0.13・一致 0.88
  - seed8 L0_D_early: 0.71 / -0.08 / 0.12 → ③a・ρ 0.32・一致 0.95
  - seed8 L1_D_early_alive: 0.67 / 0.05 / 0.26 → ③a・ρ 0.12・一致 0.96
  - seed9 L1_D_early: 0.73 / 0.04 / 0.20 → ③a・ρ 0.08・一致 0.96
  - seed9 L1_A_early: 0.66 / 0.03 / 0.28 → ③a・ρ 0.06・一致 0.95
  - seed9 L1_D_mid: 0.08 / 0.00 / 0.92 → ③b・ρ 0.01・一致 0.89
  - seed9 L1_D_late: 1.13 / 0.01 / -0.10 → ③a・ρ 0.12・一致 0.80
  - seed9 L0_D_early: 1.12 / -0.35 / 0.22 → ③a・ρ 0.30・一致 0.96
  - seed9 L1_D_early_alive: 0.73 / 0.04 / 0.20 → ③a・ρ 0.08・一致 0.96
- ReLUcap: 揃った seed 10・欠け 0・検査落ち 0（記録のラベル OTHER_UNITS_CARRY）
  - seed0 L1_D_early: 0.77 / 0.09 / 0.12 → ③a・ρ 0.22・一致 0.95
  - seed0 L1_A_early: 0.76 / 0.08 / 0.11 → ③a・ρ 0.23・一致 0.95
  - seed0 L1_D_mid: 0.21 / 0.43 / 0.07 → ①+②・ρ 0.74・一致 0.95
  - seed0 L1_D_late: 0.75 / 0.17 / 0.07 → ③a・ρ 0.45・一致 0.88
  - seed0 L0_D_early: 0.10 / 0.46 / 0.16 → ①+②・ρ 0.85・一致 0.96
  - seed0 L1_D_early_alive: 0.77 / 0.09 / 0.12 → ③a・ρ 0.22・一致 0.95
  - seed1 L1_D_early: 0.77 / 0.06 / 0.15 → ③a・ρ 0.12・一致 0.96
  - seed1 L1_A_early: 0.76 / 0.07 / 0.13 → ③a・ρ 0.13・一致 0.95
  - seed1 L1_D_mid: 0.33 / 0.02 / 0.65 → ③b・ρ 0.15・一致 0.93
  - seed1 L1_D_late: 0.88 / 0.21 / -0.01 → ③a・ρ 0.30・一致 0.89
  - seed1 L0_D_early: 0.15 / 0.37 / 0.33 → ①+②・ρ 0.62・一致 0.96
  - seed1 L1_D_early_alive: 0.77 / 0.07 / 0.15 → ③a・ρ 0.12・一致 0.96
  - seed2 L1_D_early: 0.84 / 0.07 / 0.08 → ③a・ρ 0.17・一致 0.96
  - seed2 L1_A_early: 0.76 / 0.08 / 0.14 → ③a・ρ 0.18・一致 0.96
  - seed2 L1_D_mid: 0.18 / 0.02 / 0.81 → ③b・ρ 0.32・一致 0.92
  - seed2 L1_D_late: 0.76 / 0.22 / 0.02 → ③a・ρ 0.29・一致 0.85
  - seed2 L0_D_early: 0.20 / 0.53 / 0.25 → ①+②・ρ 0.44・一致 0.96
  - seed2 L1_D_early_alive: 0.84 / 0.08 / 0.08 → ③a・ρ 0.17・一致 0.96
  - seed3 L1_D_early: 0.71 / 0.07 / 0.21 → ③a・ρ 0.14・一致 0.95
  - seed3 L1_A_early: 0.68 / 0.05 / 0.27 → ③a・ρ 0.12・一致 0.95
  - seed3 L1_D_mid: 0.22 / 0.04 / 0.74 → ③b・ρ 0.43・一致 0.95
  - seed3 L1_D_late: 0.82 / 0.16 / 0.02 → ③a・ρ 0.43・一致 0.91
  - seed3 L0_D_early: 0.34 / 0.49 / 0.26 → ①+②・ρ 0.51・一致 0.96
  - seed3 L1_D_early_alive: 0.69 / 0.07 / 0.20 → ③a・ρ 0.14・一致 0.95
  - seed4 L1_D_early: 0.70 / 0.06 / 0.22 → ③a・ρ 0.17・一致 0.96
  - seed4 L1_A_early: 0.69 / 0.08 / 0.19 → ③a・ρ 0.20・一致 0.95
  - seed4 L1_D_mid: 0.16 / 0.01 / 0.84 → ③b・ρ 0.19・一致 0.92
  - seed4 L1_D_late: 0.81 / 0.17 / 0.03 → ③a・ρ 0.43・一致 0.89
  - seed4 L0_D_early: -0.05 / 0.52 / 0.55 → ③b・ρ 0.62・一致 0.96
  - seed4 L1_D_early_alive: 0.70 / 0.06 / 0.22 → ③a・ρ 0.17・一致 0.96
  - seed5 L1_D_early: 0.71 / 0.07 / 0.19 → ③a・ρ 0.21・一致 0.96
  - seed5 L1_A_early: 0.72 / 0.11 / 0.13 → ③a・ρ 0.27・一致 0.95
  - seed5 L1_D_mid: 0.35 / 0.04 / 0.64 → ③b・ρ 0.33・一致 0.95
  - seed5 L1_D_late: 0.78 / 0.14 / 0.09 → ③a・ρ 0.47・一致 0.91
  - seed5 L0_D_early: 0.19 / 0.06 / 0.44 → ③b・ρ 0.61・一致 0.95
  - seed5 L1_D_early_alive: 0.71 / 0.07 / 0.19 → ③a・ρ 0.21・一致 0.96
  - seed6 L1_D_early: 0.79 / 0.08 / 0.13 → ③a・ρ 0.17・一致 0.95
  - seed6 L1_A_early: 0.79 / 0.09 / 0.10 → ③a・ρ 0.19・一致 0.95
  - seed6 L1_D_mid: 0.13 / 0.02 / 0.85 → ③b・ρ 0.32・一致 0.96
  - seed6 L1_D_late: 0.71 / 0.17 / 0.09 → ③a・ρ 0.34・一致 0.91
  - seed6 L0_D_early: 0.10 / 0.47 / 0.40 → ①+②・ρ 0.66・一致 0.96
  - seed6 L1_D_early_alive: 0.79 / 0.08 / 0.13 → ③a・ρ 0.17・一致 0.95
  - seed7 L1_D_early: 0.76 / 0.08 / 0.16 → ③a・ρ 0.22・一致 0.96
  - seed7 L1_A_early: 0.74 / 0.10 / 0.16 → ③a・ρ 0.22・一致 0.95
  - seed7 L1_D_mid: 0.30 / 0.03 / 0.67 → ③b・ρ 0.21・一致 0.90
  - seed7 L1_D_late: 0.81 / 0.18 / 0.02 → ③a・ρ 0.44・一致 0.91
  - seed7 L0_D_early: -0.23 / 0.84 / 0.31 → ①+②・ρ 0.61・一致 0.96
  - seed7 L1_D_early_alive: 0.76 / 0.08 / 0.16 → ③a・ρ 0.22・一致 0.96
  - seed8 L1_D_early: 0.70 / 0.07 / 0.22 → ③a・ρ 0.16・一致 0.96
  - seed8 L1_A_early: 0.71 / 0.07 / 0.18 → ③a・ρ 0.19・一致 0.95
  - seed8 L1_D_mid: 0.38 / 0.03 / 0.60 → ③b・ρ 0.27・一致 0.94
  - seed8 L1_D_late: 0.76 / 0.24 / 0.00 → ③a・ρ 0.59・一致 0.93
  - seed8 L0_D_early: 0.18 / 0.34 / 0.48 → ③b・ρ 0.57・一致 0.96
  - seed8 L1_D_early_alive: 0.70 / 0.07 / 0.22 → ③a・ρ 0.16・一致 0.96
  - seed9 L1_D_early: 0.78 / 0.07 / 0.14 → ③a・ρ 0.15・一致 0.94
  - seed9 L1_A_early: 0.73 / 0.07 / 0.18 → ③a・ρ 0.15・一致 0.94
  - seed9 L1_D_mid: 0.12 / 0.02 / 0.86 → ③b・ρ 0.56・一致 0.94
  - seed9 L1_D_late: 0.82 / 0.16 / 0.01 → ③a・ρ 0.37・一致 0.91
  - seed9 L0_D_early: -0.03 / 0.40 / 0.34 → ①+②・ρ 0.58・一致 0.96
  - seed9 L1_D_early_alive: 0.78 / 0.07 / 0.14 → ③a・ρ 0.15・一致 0.94

### C

- C1: 未完の単位 0
  - lnex1: 生き返らない 5/6・ラベル（同じ規則）FEEDBACK_NEEDED・符号検定 窓 S+1〜E p 0.0312・判定窓 p 0.0312
  - nocommon: 生き返らない 6/6・ラベル（同じ規則）FEEDBACK_NEEDED・符号検定 窓 S+1〜E p 0.0312・判定窓 p 0.0312
  - seed0 S 50 E 64；lnex1: control が窓で生き返らない；nocommon: control が窓で生き返らない；control と元の差: online 最大 1.0160・L1 死 最大 0.7969・ビット一致 False
  - seed1 S 43 E 59；lnex1: R' 48・生き返らない True・差 0.760；nocommon: R' 48・生き返らない True・差 0.750；control と元の差: online 最大 0.4878・L1 死 最大 0.5000・ビット一致 False
  - seed2 S 50 E 93；lnex1: R' 55・生き返らない False・差 0.326；nocommon: R' 55・生き返らない True・差 0.504；control と元の差: online 最大 2.0210・L1 死 最大 0.8750・ビット一致 False
  - seed4 S 50 E 65；lnex1: R' 57・生き返らない True・差 0.580；nocommon: R' 57・生き返らない True・差 0.628；control と元の差: online 最大 1.2286・L1 死 最大 0.2031・ビット一致 False
  - seed5 S 25 E 46；lnex1: control が窓で生き返らない；nocommon: control が窓で生き返らない；control と元の差: online 最大 1.8605・L1 死 最大 0.9551・ビット一致 False
  - seed6 S 50 E 77；lnex1: R' 66・生き返らない True・差 0.435；nocommon: R' 66・生き返らない True・差 0.429；control と元の差: online 最大 10.8092・L1 死 最大 0.6816・ビット一致 False
  - seed7 S 41 E 53；lnex1: R' 44・生き返らない True・差 0.704；nocommon: R' 44・生き返らない True・差 0.787；control と元の差: online 最大 0.7011・L1 死 最大 0.1328・ビット一致 False
  - seed8 S 50 E 74；lnex1: R' 73・生き返らない True・差 0.064；nocommon: R' 73・生き返らない True・差 0.054；control と元の差: online 最大 0.6192・L1 死 最大 0.8438・ビット一致 False
  - seed9 S 50 E 66；lnex1: control が窓で生き返らない；nocommon: control が窓で生き返らない；control と元の差: online 最大 1.5560・L1 死 最大 0.9238・ビット一致 False
- C3（記録だけ・ReLUcap-2）: 未完の単位 0
  - lnex1: 生き返らない 9/10・ラベル（同じ規則）FEEDBACK_NEEDED・符号検定 窓 S+1〜E p 0.0020・判定窓 p 0.0020
  - seed0 S 25 E 50；lnex1: R' 43・生き返らない False・差 0.265；control と元の差: online 最大 0.7538・L1 死 最大 0.4297・ビット一致 False
  - seed1 S 25 E 68；lnex1: R' 60・生き返らない True・差 0.240；control と元の差: online 最大 0.6366・L1 死 最大 0.1680・ビット一致 False
  - seed2 S 45 E 59；lnex1: R' 49・生き返らない True・差 0.775；control と元の差: online 最大 0.3397・L1 死 最大 0.1211・ビット一致 False
  - seed3 S 40 E 58；lnex1: R' 46・生き返らない True・差 0.710；control と元の差: online 最大 0.5739・L1 死 最大 0.6504・ビット一致 False
  - seed4 S 25 E 60；lnex1: R' 55・生き返らない True・差 0.188；control と元の差: online 最大 0.5497・L1 死 最大 0.6406・ビット一致 False
  - seed5 S 41 E 59；lnex1: R' 52・生き返らない True・差 0.459；control と元の差: online 最大 0.3628・L1 死 最大 0.3672・ビット一致 False
  - seed6 S 40 E 60；lnex1: R' 51・生き返らない True・差 0.474；control と元の差: online 最大 0.2741・L1 死 最大 0.0723・ビット一致 False
  - seed7 S 50 E 69；lnex1: R' 60・生き返らない True・差 0.508；control と元の差: online 最大 0.5084・L1 死 最大 0.2246・ビット一致 False
  - seed8 S 42 E 59；lnex1: R' 49・生き返らない True・差 0.585；control と元の差: online 最大 0.3051・L1 死 最大 0.2402・ビット一致 False
  - seed9 S 25 E 58；lnex1: R' 55・生き返らない True・差 0.182；control と元の差: online 最大 0.5987・L1 死 最大 0.7910・ビット一致 False
- C2: 未完の単位 0・gamma03（副）READOUT_SHRINK_REVIVES（10/10）・gamma03f で再び死ぬ seed 1（10 seed のうち）/ 資格の中 1
  - seed0: control 26 末 0.977（資格 True）・gamma03f 26 末 0.016・27〜31 0.01/0.00/0.00/0.00/0.00
  - seed1: control 26 末 0.967（資格 True）・gamma03f 26 末 0.020・27〜31 0.00/0.00/0.00/0.00/0.00
  - seed2: control 26 末 0.973（資格 True）・gamma03f 26 末 0.010・27〜31 0.36/0.59/0.74/0.85/0.91
  - seed3: control 26 末 0.975（資格 True）・gamma03f 26 末 0.025・27〜31 0.00/0.00/0.00/0.00/0.00
  - seed4: control 26 末 0.971（資格 True）・gamma03f 26 末 0.008・27〜31 0.00/0.00/0.00/0.00/0.00
  - seed5: control 26 末 0.996（資格 True）・gamma03f 26 末 0.010・27〜31 0.39/0.07/0.28/0.06/0.20
  - seed6: control 26 末 0.975（資格 True）・gamma03f 26 末 0.021・27〜31 0.00/0.00/0.00/0.00/0.00
  - seed7: control 26 末 0.990（資格 True）・gamma03f 26 末 0.020・27〜31 0.34/0.00/0.00/0.00/0.00
  - seed8: control 26 末 0.967（資格 True）・gamma03f 26 末 0.018・27〜31 0.00/0.00/0.00/0.00/0.00
  - seed9: control 26 末 0.973（資格 True）・gamma03f 26 末 0.014・27〜31 0.00/0.00/0.00/0.00/0.00

### D

- SiLU: 揃った seed 10・未完 []・D1（読み (2)）2・始まり ≤ 50 の読み 2・D2 判定できない 0・L3 の押しの向き: 未実施（新しい checkpoint に measure.py を当てる測定は judge.py の外）
  - seed0: t_esc None・D2 None・D3 悪化 0.0348 gap 差 0.0373 頭 1.663→2.426 → True（δ 近傍 []）・課題 1 末 0.047
  - seed1: t_esc None・D2 None・D3 悪化 0.0351 gap 差 0.0914 頭 1.800→2.232 → True（δ 近傍 []）・課題 1 末 0.046
  - seed2: t_esc None・D2 None・D3 悪化 0.0408 gap 差 0.0347 頭 1.353→2.317 → True（δ 近傍 []）・課題 1 末 0.054
  - seed3: t_esc 46・D2 {'judgeable': True, 'base': -0.23853646100000003, 'moved': False, 't_move': None, 'n_pre': 5, 'min_after': -0.188050059}・D3 悪化 0.0221 gap 差 0.0207 頭 1.448→2.202 → False（δ 近傍 []）・課題 1 末 0.044
  - seed4: t_esc 40・D2 {'judgeable': True, 'base': -0.360694795, 'moved': True, 't_move': 87, 'n_pre': 5, 'min_after': -2.29914626}・D3 悪化 0.0378 gap 差 0.0456 頭 1.511→2.412 → True（δ 近傍 []）・課題 1 末 0.050
  - seed5: t_esc None・D2 None・D3 悪化 0.0375 gap 差 0.0458 頭 1.630→2.352 → True（δ 近傍 []）・課題 1 末 0.049
  - seed6: t_esc None・D2 None・D3 悪化 0.0382 gap 差 0.0417 頭 1.400→2.293 → True（δ 近傍 []）・課題 1 末 0.051
  - seed7: t_esc None・D2 None・D3 悪化 0.0277 gap 差 0.0374 頭 1.955→2.281 → True（δ 近傍 []）・課題 1 末 0.033
  - seed8: t_esc None・D2 None・D3 悪化 0.0659 gap 差 0.0760 頭 1.371→2.499 → True（δ 近傍 []）・課題 1 末 0.050
  - seed9: t_esc None・D2 None・D3 悪化 0.0650 gap 差 0.0593 頭 1.437→2.665 → True（δ 近傍 []）・課題 1 末 0.046
- GELU: 未実施（出力なし）
- SiLUcap: 未実施（出力なし）

## 予測の採点（Opus 5.5 の欄。数は書き換えていない）

| 問い | spec | 結果 | 結果に付けた確率 | Brier |
| --- | --- | --- | --- | --- |
| A1 | :67 | True | 0.95 | 0.003 |
| A2 | :68 | LOAD_ALIGNS | 0.60 | 0.285 |
| A2rec | :69 | True | 0.40 | 0.360 |
| A3 | :70 | True | 0.70 | 0.090 |
| Arec | :71 | True | 0.50 | 0.250 |
| B | :93 | OTHER_UNITS_CARRY | 0.55 | 0.335 |
| C1 | :157 | FEEDBACK_NEEDED | 0.60 | 0.260 |
| C1rec | :158 | True | 0.50 | 0.250 |
| C2 | :159 | READOUT_SHRINK_REVIVES | 0.70 | 0.140 |
| C2rec | :160 | False | 0.50 | 0.250 |
| D1 | :185 | NO_ESCAPE | 0.10 | 1.235 |
| D2 | :186 | False | 0.60 | 0.160 |
| D3 | :187 | True | 0.45 | 0.303 |

## 読みの選択（spec の文だけで決まらない所）

- **(1) B の「沈む unit」**: 生きた unit に限らない（4 項の和 < 0 の全 unit）。spec §4（spec:79）の「沈む unit は 4 項の和が負の unit」の数を出した写し元の集計 `$CG/analysis/chargpt_cl_1019_quicklook/rho_kappa_answers.py`（a6517244、blob 015e3c5f）の carriers()（37〜46 行）が、alive で絞らずに tot < 0 の unit で share_dn = Σ項/Σ和 を取る（spec §2 の 0.02/0.05/0.84/0.09 はこの値）。rho_split の出力（rho10.py）は unit ごとの 4 項と alive を持ち、集計の定義は持たないので、この集計に合わせた。③b は others_path（others_gamma は ③b の内訳で 4 項の和に入れない。rho_split.py:184 の閉合と同じ）。記録として、生きた unit に限った割合も並べる。
- **(2) D1 の「課題 50 までに 5 課題以上続く」**: 5 課題の続きが課題 50 までに終わる（始まり t で t+4 ≤ 50）と読む。判定の量は固定の一様 probe（layer_stats.csv の probe = uniform、u = 10000、layer = 3）の r_med > +1 かつ phi1_mean ≥ 0.75（spec_chargpt_cl_1019.md:581 の追補 6 §5 の定義。前例 judge_m2.py:138 の「t から t+4 まで全部」の形を一様 probe に移した）。記録として、続きの始まりだけが課題 50 以内の読み（t ≤ 50）の数も出す。
- **(3) D2 の基準**: 基準 = 逃げ始めの課題 t_esc（D1 の続きの始まり）の前の 5 課題（t_esc−5〜t_esc−1。課題 1 より前は無いので、t_esc ≤ 5 なら有る課題だけ）の L2 の r_med（一様 probe）の平均。課題 t_esc〜100 の課題末に r_med ≤ 基準 − 1 があれば「移る」。t_esc = 1 なら基準が無く判定できない（数に入れず、その数を書く）。前例 judge_m2.py:140 は「直前 5 課題の移動平均」で別の形なので使わない（spec:173 は固定の基準）。
- **(4) A の生きた unit の probe**: load10.py の n_alive・own_sink は写し元 measure.py（:121）の定義のまま、課題 t の鎖の probe（init は課題 1 の鎖）で生きた unit を決める。spec:42 は「一様 probe で z > 0 のトークンがある unit」と書くが、spec:39 の「定義は chargpt_load_fold_1021 §2 のまま」と照合（seed 1000 で一致）に合わせた load10.py の出力を、そのまま使う。A2 の記録の帯 0.5 ± 3√(0.25/n_live) の n_live も、この n_alive（init の L1）。一様 probe で数え直すかは主の判断（compute_A の報告の 1）。
- **(5) A の対象の腕と層**: A2・A3 は C3-2 の L1（spec:53 で ReLUcap-2 と L0 は記録だけ）。A1 は 2 腕 × 10 seed × 課題末 35 = 700 状態（spec:48）。r_mag は写し元 report.py:36 と同じく own_abs_median（全 512 unit の |G_own| の中央値）。生きた unit だけの own_abs_median_alive の比は記録。
- **(6) A2 の符号検定**: s_early > s_init を +、< を −、等しいものは除いて両側 exact 符号検定（二項 1/2）。spec:50 の「9/10 で 0.021」と一致する。
- **(7) B の「割合の中央値」と最大**: seed ごとに、課題 k+1 ∈ {2,3,5,10} の各状態で share(③a)、share(①)+share(②)、share(③b) を出し、4 課題の中央値を取って最大の項を決める。rho10 の検査（check_eval.all_pass）が落ちた状態は使わず、その seed は「検査落ち」として未完と同じく数えない。
- **(8) C1 の符号検定の窓**: spec:131 の「窓の L1 の死の平均」の窓は spec:101 の定義（S_k+1〜E_k）とした。記録として判定窓 [R'_k, E_k] の平均の差の検定も出す。対の差は lnex1 − control（どちらも byakko の再生、追補 1）。0 の差は除く。
- **(9) C の「死」の量**: 各課題末の layer_stats.csv の probe = uniform・layer = 1・frac_p_lt005（u = 課題の更新数）。境は spec のまま「≤ 0.5 で生き返り」「> 0.5 で生き返らない」「≥ 0.9 で死」。
- **(10) C2 の記録（再び死ぬ）**: gamma03f が課題 26 末に生き返り（≤ 0.5）、課題 27〜31 末のどこかで ≥ 0.9 になった seed を数え、予測（spec:160）は 10 seed のうちの数で採点する。資格（control ≥ 0.9）の seed の中の数も並べる。
- **(11) C1 の記録の予測（nocommon 8/9）**: 対象の数 n が 9 のときは文どおり 8 以上。n ≠ 9 のときは追補 1 の規則（n − 1 以上）で読み、その旨を書く。
- **(12) D3 の δ**: δ = 3 × 0.0084 = 0.0252（前例 judge_m2.py:16 と同じ。spec:174 の 0.025 はその丸め）。0.025 と 0.0252 の間に入る値があれば記録に書く。量は judge_m2.py:23-33 のまま（online_all の課題 2〜10 と 91〜100 の平均、fresh gap = online_all − fresh の online_all、頭 100 更新 = online.npz の loss[:, :100] − H の平均）。条件 1 の「課題 1 末 ≤ 0.1」は 3 つに入れず記録にする。
- **(13) 予測の採点**: 結果のラベルに付けた確率と Brier（多クラスは Σ(p − 1[結果])²）を書く。結果が表に無いラベル（INSUFFICIENT・未完など）は採点しない。

## 入力

- load: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/load
- rho: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/rho
- revival: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/revival
- silu4: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/silu4
- revival_plan_sha256: 77f4e1bef1b902e6320dace2349b4fa7ffc058ee57dcdc1defd0749d2d22180c
