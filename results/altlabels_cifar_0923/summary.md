# altlabels_cifar_0923 — 2 ラベル交互（ABAB）の第1層 W と幅

`report.py` が書いた。生成 2026-09-23T03:02:23+0900。spec sha256 `37630ef6c338ac46836d90abf4e5f6f5f16ec6021aae88a28cd454d41bd4a3b5`（c2b6d79）。閾値は spec §4 のもの（下の表）。

**数値は seed 中央値 [最小, 最大]（seed の範囲であって信頼区間ではない）。** 判定は中央値の区間で、中央値の区間に入る seed が 80% 未満なら `_MIXED` を付ける。8/10 は整合性の旗で 5% の検定ではない（spec §3）。

## 0. いまあるデータ

| 腕 | seed | 課題 t の最大 | 登録窓（j=14..25 / t27–50）| 状態 |
|---|---|---|---|---|
| LR_aaaa | 10 (0–9) | 50–50 | 完 | 確定 |
| LR_abab | 10 (0–9) | 50–50 | 完 | 確定 |
| LR_abab_stop | 5 (0–4) | 50–50 | 完 | 確定 |
| LR_abc | 10 (0–9) | 12–12 | —（12 課題の腕なので Q1 の窓は無い） | 確定 |
| LR_iid | 10 (0–9) | 50–50 | 完 | 確定 |
| LR_iid_ext | 10 (0–9) | 50–50 | 完 | 確定 |
| LR_iid_stop | 5 (0–4) | 50–50 | 完 | 確定 |
| SNA_abab | 10 (0–9) | 50–50 | 完 | 確定 |
| SNA_iid_ext | 10 (0–9) | 50–50 | 完 | 確定 |

## 1. 登録判定（§4）

| Q | 統計量 | 中央値 | 判定 | 区間ごとの seed 数 | 状態 |
|---|---|---|---|---|---|
| Q1 | rho_mid2 = slope(E_mid2, j=14..25) / same for LR_iid, per seed | 0.5924 | **IID_SCALE_GROWTH** | {'IID_SCALE_GROWTH': 10} | 確定 |
| Q2 | mid2 C2 (arithmetic mean of Frobenius cosines), LR_abab, late window t26-50 | 0.1883 | **PERIODIC** | {'PERIODIC': 10} | 確定 |
| Q3 | r_top = ||sum_{t=6..50} dW_t||^2_top / sum ||dW_t||^2_top, LR_abab | 0.05886 | **LOW_NET_DISPLACEMENT** | {'LOW_NET_DISPLACEMENT': 8, 'HIGH_NET_DISPLACEMENT': 2} | 確定 |
| Q3b | median over late odd t of cos(W_t, W_{t-2})_top, minus the seed's own (c_IID + 1)/2 threshold | 0.01255 | **HIGH_DIRECTIONAL_RETURN** | {'HIGH_DIRECTIONAL_RETURN': 9, 'LOW_DIRECTIONAL_RETURN': 1} | 確定 |
| Q4 | LR_abab_fork t=48: A vs C, hit999 with the spec's censoring | A/C 中央値 0.1452 | **SAVINGS** + STRONG_SAVINGS + FAST_REVISIT | A先 10 / C先 0 / 同 0 / 未確定 0 | 確定 |
| Q5 | sigma_med(50), LR_abab (numpy median over the 100 units of sd(z1_i)) | 55.18 | **LOWER_WIDTH_SIDE** | {'LOWER_WIDTH_SIDE': 10} | 確定 |
| Q6 | N_all(50) of LR_aaaa over LR_iid, same seed | 0.2642 | **CONTINUOUS_FIT_LT_80PCT_IID** | {'CONTINUOUS_FIT_LT_80PCT_IID': 10} | 確定 |
| Q7 | mid2 (C1,C2,C3) t4–12 | (-0.1721, -0.1553, 0.4585) | **PERIOD3_PATTERN** | {'PERIOD3_PATTERN': 10} | 確定 |
| Q8 | sigma_med(50), SNA_abab | 24.99 | **LOWER_SNAKE_WIDTH** | {'LOWER_SNAKE_WIDTH': 10} | 確定 |
| Q9 | N_all(50) of LR_abab_stop over LR_iid_stop, same seed (seeds 0-4, recorded only) | 0.4512 | **STOP_N_RATIO_LT_0.5** | {'STOP_N_RATIO_LT_0.5': 5} | 確定 |

副次（登録外の記録）

| 名前 | 統計量 | 中央値 | 判定 |
|---|---|---|---|
| Q1_sub_low | rho_low (registered rule, recorded only) | 0.7965 | IID_SCALE_GROWTH |
| Q1_sub_all | rho_all (registered rule, recorded only) | 0.5574 | IID_SCALE_GROWTH |
| Q1_sub_comp | rho_comp (registered rule, recorded only) | 1.238 | IID_SCALE_GROWTH |
| Q1_sub_top | rho_top (registered rule, recorded only) | 0.3976 | REDUCED_GROWTH_MIXED |
| Q1_sub_mid1 | rho_mid1 (registered rule, recorded only) | 0.4424 | REDUCED_GROWTH_MIXED |
| Q1_sub_comp_absolute | comp complete-cycle slope of LR_abab, per task (threshold 2.1; LR_iid measures ~3.9) | 4.657 | COMP_SLOPE_GT_2.1 |
| EXP_GT_1.5 | beta_all = log-log slope of N_all vs t over t6-50 (descriptive flag) | 0.8341 | EXP_LE_1.5 |
| Q2_sub_low | low C2 (late), threshold 0.059 | 0.2057 | LOW_PERIODIC |
| Q2_sub_top | top C2 (late), threshold 0.13 | 0.4065 | TOP_PERIODIC |
| Q4_sub_t2 | LR_abab_fork t=2: A vs C, hit999 with the spec's censoring | A vs C = 0.7141 | SAVINGS |
| Q4_sub_t10 | LR_abab_fork t=10: A vs C, hit999 with the spec's censoring | A vs C = 0.244 | SAVINGS |
| Q4_sub_t20 | LR_abab_fork t=20: A vs C, hit999 with the spec's censoring | A vs C = 0.1818 | SAVINGS |
| Q4_sub_t30 | LR_abab_fork t=30: A vs C, hit999 with the spec's censoring | A vs C = 0.1623 | SAVINGS |
| Q4_sub_t40 | LR_abab_fork t=40: A vs C, hit999 with the spec's censoring | A vs C = 0.1574 | SAVINGS |
| Q4_sub_iid_next_vs_C | LR_iid_fork t=48: next vs C, hit999 with the spec's censoring | next vs C = 0.9792 | SAVINGS_NOT_ESTABLISHED |
| Q8_sub_rho_mid2_vs_ext | SNA_abab mid2 complete-cycle slope over the archived SNA R=20 IID slope (layout differs; recorded only) | 1.081 | IID_SCALE_GROWTH |

周期の中心の移動と直径の増大（mid2、j=14..25 の傾き、seed 中央値）

| 腕 | ‖M_j‖² の傾き | D_j² の傾き |
|---|---|---|
| LR_abab | 413.5 | -0.05084 |
| LR_iid | 698.2 | 2.244 |

### Q4 の頑健性（同じ判定を hit_full と hit99 でも）

| 基準 | A 先 / C 先 / 同 / 未確定 | 判定 | A/C 中央値（点 / 上界）| A の中央値 |
|---|---|---|---|---|
| hit999（登録） | 10 / 0 / 0 / 0 | SAVINGS + STRONG + FAST | 0.1452 / 0.1452 | 400 |
| hit_full (1200/1200) | 10 / 0 / 0 / 0 | SAVINGS + STRONG + FAST | 0.1447 / 0.1447 | 400 |
| hit99 | 10 / 0 / 0 / 0 | SAVINGS + STRONG + FAST | 0.15 / 0.15 | 300 |

打ち切り: hit999 は 30,000 で右打ち切り。hit_full / hit99 は束自身の最後の評価点で打ち切り（束は全スロットが hit999+500 を過ぎた時点で終わるので 30,000 に届かないことがある）。大小が確定する対だけ数える。

## 2. 予測の採点（§5）

○ = 起きた、× = 起きなかった、△ = まだ判定できない。Issa の列は空のまま（就寝中）。

| # | 述語 | Claude | Codex | 結果 | 実測 |
|---|---|---|---|---|---|
| P1 | Q1 = IID_SCALE_GROWTH | 0.50 | 0.50 | ○ | rho_mid2 median = 0.5924 -> IID_SCALE_GROWTH |
| P2 | Q1 = FASTER_GROWTH | 0.30 | 0.15 | × | rho_mid2 median = 0.5924 -> IID_SCALE_GROWTH |
| P3 | Q1 = REDUCED_GROWTH | 0.20 | 0.35 | × | rho_mid2 median = 0.5924 -> IID_SCALE_GROWTH |
| P4 | EXP_GT_1.5 (beta_all > 1.5) | 0.15 | 0.15 | × | beta_all median = 0.8341 |
| P5 | Q2 = PERIODIC (mid2 C2 >= 0.077) | 0.60 | 0.85 | ○ | mid2 C2 (late) median = 0.1883 |
| P6 | Q3 = LOW_NET_DISPLACEMENT (r_top < 0.0636) | 0.70 | 0.70 | ○ | r_top median = 0.05886 |
| P7 | Q3b = HIGH_DIRECTIONAL_RETURN | 0.40 | 0.75 | ○ | median margin over (c_IID+1)/2 = 0.01255 |
| P8 | Q4 = SAVINGS (t48, A < C in >= 8/10) | 0.75 | 0.75 | ○ | A earlier than C in 10/10 seeds |
| P9 | Q4 = STRONG_SAVINGS | 0.35 | 0.25 | ○ | median A/C (upper bound) = 0.1452 |
| P10 | Q5 = LOWER_WIDTH_SIDE (sigma(50) < 60.1) | 0.45 | 0.50 | ○ | sigma_med(50) median = 55.18 |
| P11 | Q6 = CONTINUOUS_FIT_GE_80PCT_IID | 0.45 | 0.45 | × | N_all(50) ratio median = 0.2642 |
| P12 | Q7 = PERIOD3_PATTERN (12 tasks) | 0.50 | 0.80 | ○ | (C1, C2, C3) = (-0.1721, -0.1553, 0.4585) |
| P13 | Q8 = LOWER_SNAKE_WIDTH | 0.35 | 0.55 | ○ | SNA sigma_med(50) median = 24.99 |
| P14 | Q9 N ratio < 0.5 | 0.30 | 0.40 | ○ | stop N ratio median = 0.4512 |
| P15 | Q9 N ratio in [0.5, 1.5] | 0.50 | 0.55 | × | stop N ratio median = 0.4512 |
| P16 | r_top < 0.0636 and N_top(49)/N_top(25) > 1.10 | 0.50 | 0.55 | ○ | r_top = 0.05886, N_top(49)/N_top(25) median = 1.396 |
| P17 | comp complete-cycle slope > 2.1 per task | 0.60 | 0.60 | ○ | comp cycle slope median = 4.657 |

未判定 0/17。* = 窓が未完のままの暫定（0 件）。

決着した 17 件の Brier 得点: Claude 0.2250、Codex 0.1728（小さいほど良い。○ は 12 件）。

## 3. seed ごとの表

**Q1** — rho_mid2 = slope(E_mid2, j=14..25) / same for LR_iid, per seed

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 0.5485 | 0.5584 | 0.543 | 0.6044 | 0.5153 | 0.6422 | 0.6337 | 0.6522 | 0.5804 | 0.6514 | 0.5924 |

**Q2** — mid2 C2 (arithmetic mean of Frobenius cosines), LR_abab, late window t26-50

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 0.1813 | 0.1868 | 0.1798 | 0.1866 | 0.1898 | 0.1918 | 0.1998 | 0.1854 | 0.1955 | 0.1928 | 0.1883 |

**Q3** — r_top = ||sum_{t=6..50} dW_t||^2_top / sum ||dW_t||^2_top, LR_abab

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 0.05877 | 0.05776 | 0.05676 | 0.06257 | 0.06374 | 0.06206 | 0.0639 | 0.05895 | 0.05769 | 0.05813 | 0.05886 |

**Q3b** — median over late odd t of cos(W_t, W_{t-2})_top, minus the seed's own (c_IID + 1)/2 threshold

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 0.007043 | -0.0002069 | 0.006287 | 0.0205 | 0.01192 | 0.02233 | 0.01318 | 0.01137 | 0.01324 | 0.01884 | 0.01255 |

**Q5** — sigma_med(50), LR_abab (numpy median over the 100 units of sd(z1_i))

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 54.14 | 56.33 | 55.59 | 55.02 | 53.07 | 55.28 | 55.57 | 55.87 | 55.09 | 54.63 | 55.18 |

**Q6** — N_all(50) of LR_aaaa over LR_iid, same seed

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 0.2627 | 0.275 | 0.2657 | 0.3123 | 0.21 | 0.2211 | 0.2239 | 0.3293 | 0.2297 | 0.2812 | 0.2642 |

**Q8** — sigma_med(50), SNA_abab

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 中央値 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 値 | 24.92 | 24.62 | 25.05 | 25.37 | 24.9 | 25.27 | 25.49 | 24.79 | 24.53 | 25.4 | 24.99 |

**Q9** — N_all(50) of LR_abab_stop over LR_iid_stop, same seed (seeds 0-4, recorded only)

| seed | 0 | 1 | 2 | 3 | 4 | 中央値 |
|---|---|---|---|---|---|---|
| 値 | 0.4638 | 0.4545 | 0.4427 | 0.4512 | 0.4483 | 0.4512 |

**Q4** — fork t=48、hit999（A / C / 符号）

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 400 | 300 | 400 | 300 | 400 | 400 | 300 | 400 | 400 | 300 |
| C | 2500 | 2600 | 2500 | 2300 | 2400 | 2300 | 2400 | 3300 | 2500 | 2400 |
| A<C | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |

## 4. 使った閾値と規約

| 名前 | 値 | 由来 |
|---|---|---|
| Q1_rho_lo | 0.5 | midpoint of 0 and 1 |
| Q1_rho_hi | 1.5 | operational: 50% faster |
| Q1_comp_abs_slope | 2.1 | midpoint of 0 and the fresh-leak 4.2 per task |
| Q1_beta_flag | 1.5 | descriptive flag EXP_GT_1.5 |
| Q2_mid2_C2 | 0.0774 | midpoint of OU-F -0.005 and periodic mixture +0.16 |
| Q2_low_C2 | 0.059 | sub |
| Q2_top_C2 | 0.13 | sub |
| Q3_r_top | 0.0636 | midpoint of the exact return 1/45 and the IID 0.105 |
| Q4_seeds | 8 | consistency flag out of 10 (one-sided sign test p = 0.055) |
| Q4_ratio | 0.5 | operational |
| Q4_fast | 1800 | the evaluation point after a fresh net's 1767 |
| Q5_sigma50 | 60.08 | midpoint of the fixed-return 20.3 and the IID 82.5 |
| Q5_sigma25_record | 47.7 | record only (64.3 is an extrapolated proxy) |
| Q6_N_ratio | 0.8 | midpoint of the Adam-only null 1.09 and the PMNIST transplant 0.52 |
| Q7_C3 | 0.08 | mixture (-0.08,-0.08,+0.16) vs pure return (-0.5,-0.5,+1) |
| Q7_C2 | 0.08 | same |
| Q8_sigma50 | 29.36 | midpoint of the fixed-reuse 20.7 and the IID equilibrium 36 |
| Q9_lo | 0.5 | operational |
| Q9_hi | 1.5 | operational |
| P16_Ntop_ratio | 1.1 | N_top(49) / N_top(25) |

解析側の規約（`ledger_provenance.json` にも入っている）

- `sigma_med`: numpy median (midpoint) of sd(z1_i) over the 100 units, from W1
- `sigma_med_lower`: torch's lower-middle median, only to compare with trace sig_med
- `trace_tc`: REPAIRED: tc_at_task_end[t] - task_steps + step (the engine writes the task's final tc on every row)
- `hit99/hit999/hit_full`: recomputed from the trace counts (>=1188 / >=1199 / =1200), step > 0 only; per_task.csv's copies kept alongside
- `correct_at_hit999_plus_500`: recomputed and flagged when that eval point was never reached (the engine's acc_at_hit_plus_500 falls back to the final count)
- `fork_censoring`: hit999 at 30,000; hit99 / hit_full at the bundle's own last step

帳簿の恒等式（S7）: ΔN = d + e の最大相対残差 1.78e-15（許容 1e-06）、帯の和 4.41e-16、V の 4 帯の和 4.42e-16、V_mu = 0.0、V_comp = 0.0、σ と走の trace の差 1.7e-05。

合成検査（`synthetic_checks.json`）: all_pass = True。

## 5. 読み方の注意

- どの判定も **50 課題内の分類**で、有界性や漸近次数の判定ではない（spec §0）。
- 「同じ」の主張には同等性の議論が要る。差が出ないことでは足りない（spec §3）。
- 外部参照（`LR_iid_ext` / `SNA_iid_ext`）は親の R=20 の走で、配置が違うので bit 一致は仮定していない。登録の対照は同配置の `LR_iid`。
- fork 点 t = 48 だけが登録で、他の fork 点は独立標本として合算しない（spec §4 Q4）。
