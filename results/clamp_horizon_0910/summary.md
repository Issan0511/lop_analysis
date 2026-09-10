# clamp_horizon_0910 summary

spec `specs/spec_clamp_horizon_0910.md`（単独 commit `7ec15b7`）。窓は late = t301–400、base = t16–20（共有前置き）。
ρ = (acc_C − acc_ref)/(acc_base − acc_ref) ＝ 参照の累積損失のうちその腕が取り戻した割合。ラベルは可検定な 3 seed が 3/3 一致したときのみ。

## 事前登録の判定

| arm | A_depth | B_width | C_dissoc | D_dissoc | E_dose |
|---|---|---|---|---|---|
| LR | MIXED | MIXED | NOT_TESTABLE_DEPTH_ALSO_STOPPED | NOT_SHOWN | NOT_TESTABLE_NO_WIDTH_EFFECT |
| ELU1 | DEPTH_NO_EFFECT | WIDTH_REMOVES_LOSS | WIDTH_WITHOUT_DEPTH | NOT_SHOWN | DOSE_GRADED |

## 連続量（seed 別）

| arm | rho_dclamp | rho_wclamp | rho_wcap2 | L_ref_pt | acc_base |
|---|---|---|---|---|---|
| LR | +0.179;+0.166;+0.244 | +0.709;+0.775;+0.875 | +0.396;+0.414;+0.559 | 4.65;4.23;4.21 | 0.9255;0.9242;0.9193 |
| ELU1 | +0.092;+0.066;+0.032 | +0.905;+0.894;+0.817 | +0.540;+0.532;+0.512 | 4.00;4.15;4.25 | 0.9218;0.9235;0.9254 |

`zbar_late`（ref/wclamp/dclamp）と `cnorm_ratio_dclamp`、ゲート G3/G4/G5:

| arm | zbar_late | cnorm_ratio_dclamp | g3 | g4 | g5 |
|---|---|---|---|---|---|
| LR | -7.93/-2.78/-2.54;-8.29/-2.90/-2.59;-8.69/-2.91/-2.63 |  | 1111 | 111 | 000 |
| ELU1 | -8.45/-10.58/-2.96;-8.18/-8.35/-2.67;-8.11/-10.64/-2.66 |  | 1111 | 111 | 111 |

## 副測定（late 窓・seed 中央値）

| arm | clamp | acc | slope (pt/100task) | z̄ | σ_inv | star_sd | ‖W̃‖ | p⁺ | ‖W2col‖ | dead_hard | 飽和対 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | ref | 0.8789 | -0.41 | -8.29 | 4.644 | 4.343 | 14.28 | 0.111 | 3.69 | 0 | 0.00 |
| LR | dclamp | 0.8875 | -0.50 | -2.59 | 3.810 | 5.360 | 13.21 | 0.289 | 3.68 | 0 | 0.00 |
| LR | wclamp | 0.9140 | -0.13 | -2.90 | 1.378 | 1.549 | 3.72 | 0.097 | 3.96 | 0 | 0.00 |
| LR | wcap2 | 0.8995 | -0.10 | -4.56 | 2.431 | 2.338 | 7.42 | 0.108 | 3.87 | 0 | 0.00 |
| ELU1 | ref | 0.8820 | -0.59 | -8.18 | 5.648 | 12.453 | 12.94 | 0.235 | 4.34 | 7 | 0.62 |
| ELU1 | dclamp | 0.8847 | -0.56 | -2.67 | 4.384 | 8.578 | 13.25 | 0.389 | 3.86 | 0 | 0.45 |
| ELU1 | wclamp | 0.9180 | +0.07 | -10.58 | 5.782 | 14.631 | 3.70 | 0.121 | 4.03 | 8 | 0.56 |
| ELU1 | wcap2 | 0.9040 | -0.20 | -8.33 | 4.803 | 11.970 | 7.25 | 0.167 | 4.25 | 3 | 0.61 |

検算値は各腕の `*_provenance.json` の `checks`（G1 は t21–100 の committed 再現）。
図: `fig_clamp_horizon.png`。