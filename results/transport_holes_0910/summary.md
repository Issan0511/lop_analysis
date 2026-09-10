# transport_holes_0910 summary

spec `specs/spec_transport_holes_0910.md`（単独 commit）。V9 §11 の穴 1〜3 を箱 B で塞ぐ 4 副走。

## 副走 A — 谷越え型の参照軌道 t1–400

| arm | A1_inversion | A1p_accumulation | A1pp_freezing | A2_slope | A3_width_order |
|---|---|---|---|---|---|
| GELU | INVERSION_ACCUMULATES | PARTIAL | PARTIAL | PARTIAL | WIDTH_ORDER_BREAKS |
| SILU | INVERSION_ACCUMULATES | PARTIAL | ESCAPE_DOES_NOT_FREEZE | PARTIAL | WIDTH_ORDER_BREAKS |

| arm | inv_units_t20 | inv_units_t400 | delta_inv_units | frozen_units_t20 | frozen_units_t400 | delta_frozen_units | dead_hard_t400 | immobile_units_t400 | beyond_frac_t400 | slope_late | cum_loss_pt | cnorm_t400 | zbar_t400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GELU | 67;72;69 | 79;79;79 | +12;+7;+10 | 0;0;0 | 4;5;5 | +4;+5;+5 | 63;55;53 | 1;0;1 | 0.843;0.841;0.827 | -0.82;-0.19;-0.73 | 7.74;7.47;7.83 | 9.05;8.91;9.41 | -7.00;-6.64;-7.04 |
| SILU | 67;70;68 | 84;80;84 | +17;+10;+16 | 0;0;0 | 0;0;0 | +0;+0;+0 | 28;28;24 | 0;0;0 | 0.838;0.812;0.838 | -1.07;+0.12;-0.31 | 7.05;6.81;6.92 | 10.93;11.13;11.05 | -6.17;-6.53;-6.89 |

`inv_units` = プローブ標本の 50% 超で φ′<0 のユニット数（**上限 100 で t20 で既に飽和気味**）。`frozen_units` = 谷の向こうに居てゲートが float32 で 0 に潰れたユニット数（**0 から始まる**・導出が名指しする終状態）。`dead_hard` は符号付きゲートの counter なので谷越え型では反転を死と読み違える。`immobile_units` は |φ′| で測った本当の不動。

## 副走 B・C — クランプ

| arm | A_depth | B_width | C_dissoc | D_dissoc | E_dose | FG_counter | joint |
|---|---|---|---|---|---|---|---|
| R | DEPTH_NO_EFFECT | WIDTH_REMOVES_LOSS | WIDTH_WITHOUT_DEPTH | NOT_SHOWN | DOSE_GRADED | PARTIAL | BOTH_RIGHT |
| GELU | DEPTH_PARTIAL | WIDTH_REMOVES_LOSS | NOT_TESTABLE_DEPTH_ALSO_STOPPED | NOT_SHOWN | DOSE_PARTIAL | PARTIAL | UNDECIDED |
| SILU | NOT_TESTABLE_DEPTH_NOT_HELD | WIDTH_REMOVES_LOSS | WIDTH_WITHOUT_DEPTH | NOT_TESTABLE_DEPTH_NOT_HELD | DOSE_GRADED | NOT_TESTABLE_DEPTH_NOT_HELD | NOT_TESTABLE |

| arm | rho_dclamp | rho_wclamp | rho_wcap2 | L_ref_pt | counter_drop_dclamp | g0 | g3 | g4 | g5 |
|---|---|---|---|---|---|---|---|---|---|
| R | -0.034;+0.002;-0.037 | +0.646;+0.820;+0.735 | +0.475;+0.616;+0.594 | 4.97;4.74;4.64 | +13;+8;+3 | 111 | 1111 | 111 | 111 |
| GELU | +0.278;+0.273;+0.249 | +0.667;+0.732;+0.673 | +0.256;+0.226;+0.234 | 7.74;7.47;7.83 | +16;+12;+14 | 111 | 1111 | 111 | 000 |
| SILU | +0.045;-0.044;+0.072 | +0.782;+0.761;+0.857 | +0.354;+0.289;+0.373 | 7.05;6.81;6.92 | +4;+3;+4 | 111 | 1111 | 001 | 111 |

## 副走 D — 駆動源の符号

| arm | D_sign | inverted_force | shoulder_force | neg_force_total | pos_force_total | gate_shift_control | trajectory_maxabs |
|---|---|---|---|---|---|---|---|
| R | RELU_ZERO | +0 | +0 | +0 | -3343 | +1.679e-08 | 0.0 |
| LR | SIGN_SINKS | +1984 | +0 | +1984 | -420.3 | +1984 | 0.0 |
| ELU1 | SIGN_MIXED | -161.6 | +0 | -161.6 | -1156 | -161.6 | 0.0 |
| GELU | VALLEY_SIGN_HOLDS | -4841 | +195.7 | -4645 | +3304 | -4645 | 0.0 |
| SILU | VALLEY_SIGN_HOLDS | -2903 | -326.4 | -3229 | +1751 | -3229 | 0.0 |

| arm | neg_by_window_phase | beyond_force |
|---|---|---|
| R | +0;+0;+0;+0 | +0;+0;+0;+0 |
| LR | +680;+264;+516;+525 | +0;+0;+0;+0 |
| ELU1 | +199;+1.59e+03;-659;-1.3e+03 | +0;+0;+0;+0 |
| GELU | -515;-949;-631;-2.55e+03 | -456;-1.33e+03;-634;-2.42e+03 |
| SILU | -158;-400;-599;-2.07e+03 | -486;-394;-363;-1.66e+03 |

負側の Σφ′eS が正なら沈み、負なら浮き。ReLU は φ′(z≤0)=0 で厳密に 0（`gate_shift_control` が同じ和をゲート +1e−12 で取ったもので、0 でないことがこの 0 の非空虚性）。
