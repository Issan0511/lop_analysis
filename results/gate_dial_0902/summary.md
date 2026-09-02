# gate_dial_0902 summary (stage all)

## Verdict

- **V1（標準点の位置）: CAPACITY_UNDEFINED** — S_b1_1216=CAPACITY_UNDEFINED, G_b1_1216=present
- V1 で発症した腕: G_b1_1216
- **V1 は S-cap 除外の帰結として `CAPACITY_UNDEFINED`**: S_b1_1216 は early 窓でフィットしておらず、絶対閾値 0.05 に対して発症が**定義されない**（spec §6 の S-cap。`width5_gate_0901` と同型）。登録された 4 ラベルのどれでもない。

| family | V2 | 当たっていた行 | 梯子（軟→硬） | n_onset(5M) | 落とした腕 |
|---|---|---|---|---|---|
| leaky | **MONOTONE_TOWARD_RELU** | MONOTONE_TOWARD_RELU | LR_1216 → LR_a0p01_1216 → LR_a0p001_1216 → LR_a0p0001_1216 → LR_a0p00001_1216 → R_1216 | 0, 10, 10, 10, 10, 10 | — |
| elu | **MONOTONE_TOWARD_RELU** | MONOTONE_TOWARD_RELU | E_1216 → E_a0p1_1216 → E_a0p01_1216 → E_a0p001_1216 → R_1216 | 0, 5, 10, 10, 10 | — |
| silu | **MONOTONE_TOWARD_RELU** | MONOTONE_TOWARD_RELU | S_b3_1216 → S_b10_1216 → R_1216 | 10, 10, 10 | S_b0p3_1216(capacity_undefined), S_b1_1216(capacity_undefined) |
| gelu | **MONOTONE_TOWARD_RELU** | MONOTONE_TOWARD_RELU | G_b1_1216 → G_b3_1216 → R_1216 | 10, 10, 10 | G_b0p3_1216(capacity_undefined) |

- Numeric divergence: none
- CAPACITY_UNDEFINED: G_b0p3_1216, S_b0p3_1216, S_b1_1216

### 引用上の注意（spec §8）

- 0/10 は「5M までに観測しなかった」（片側 95% 上限 p<=0.2589）。「起きない」と書かない。
- **対照 `R_1216` / `LR_1216` / `E_1216` は別走 `gate_dose_0830` の committed 値であり、
  同一走の腕ではない。** ペアリングは init・教師・入力実現までで、軌道は step 1 以降で分岐する。
- **用量 1 点（12.16）の主張である。** 引くときは必ず用量を添える。
- **u_fr・谷底は閉形式 + K=1 の代入で、実験出力ではない。** 「凍結深さは 13.8」と
  測定値のように書かない（dial_table.csv の *_registered / *_numeric とも同じ格）。
- `layer1_mob` は ReLU・leaky では p_hat の一次関数。「可動度を測った」と書けるのは
  ELU・SiLU・GELU 腕だけ。
- 対照の m⁻ は **5M チェックポイントの 1 点**（窓 `final_step5000000`）であって
  末尾窓 491-500 の量ではない。窓を落として引かない。
- `beyond_valley` / `frozen` は**位置**であって病理ではない。
- V3 からラベルを作らない。「硬さはスカラーか」は裁定であって判定ではない。
- SiLU/GELU の beta -> inf は数学的極限であって、`R_1216` は SiLU/GELU 族の腕ではない。
- Q2（increments.csv）は `SCALING_MISMATCH` の履歴があるので主張に使わない。

## Endpoints (5M)

| arm | family | dial | onset 1M | onset 5M | median log10 U 1M | median log10 U 5M | source |
|---|---|---:|---:|---:|---:|---:|---|
| S_b1_1216 | silu | 1 | 1/10 | 10/10 | -1.71585 | -0.75298 | this run |
| G_b1_1216 | gelu | 1 | 3/10 | 10/10 | -1.71838 | -0.00301615 | this run |
| LR_a0p01_1216 | leaky | 0.01 | 10/10 | 10/10 | -0.561378 | -0.405637 | this run |
| LR_a0p001_1216 | leaky | 0.001 | 10/10 | 10/10 | -0.551 | -0.0712567 | this run |
| LR_a0p0001_1216 | leaky | 0.0001 | 10/10 | 10/10 | -0.492966 | -0.247254 | this run |
| E_a0p1_1216 | elu | 0.1 | 0/10 | 5/10 | -1.73578 | -1.32614 | this run |
| E_a0p01_1216 | elu | 0.01 | 10/10 | 10/10 | -0.926856 | -0.524654 | this run |
| S_b0p3_1216 | silu | 0.3 | 10/10 | 10/10 | -0.798815 | -0.674779 | this run |
| S_b3_1216 | silu | 3 | 6/10 | 10/10 | -1.08332 | -9.02165e-06 | this run |
| G_b0p3_1216 | gelu | 0.3 | 6/10 | 6/10 | -1.2778 | -1.21727 | this run |
| G_b3_1216 | gelu | 3 | 9/10 | 10/10 | -0.322285 | -2.61001e-06 | this run |
| LR_a0p00001_1216 | leaky | 1e-05 | 10/10 | 10/10 | -0.571884 | -0.23369 | this run |
| E_a0p001_1216 | elu | 0.001 | 9/10 | 10/10 | -0.595037 | -0.344607 | this run |
| S_b10_1216 | silu | 10 | 10/10 | 10/10 | -0.554655 | -0.000574538 | this run |
| R_1216 | relu | 0 | 10/10 | 10/10 | -0.649239 | -0.274662 | gate_dose_0830 (別走) |
| LR_1216 | leaky | 0.1 | 0/10 | 0/10 | -2.20796 | -2.65235 | gate_dose_0830 (別走) |
| E_1216 | elu | 1 | 0/10 | 0/10 | -2.2543 | -2.72561 | gate_dose_0830 (別走) |

## §5.4 V3 硬さ表（ラベルを置かない）

| arm | family | dial | u* | u_fr | m⁻ | m⁻ 窓 | 沈下率 | 谷率 | 凍結率 | n_onset 5M | k* 中央値 | median log10 U 5M |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| S_b1_1216 | silu | 1 | 1.278 | 16.56 | -3.68e-05 | late_tasks_5m | 0.9995 | 0.979 | 0.5 | 10 | 10 | -0.753 |
| G_b1_1216 | gelu | 1 | 0.7518 | 5.394 | -3.992e-15 | late_tasks_5m | 1 | 0.9995 | 1 | 10 | 92 | -0.003016 |
| LR_a0p01_1216 | leaky | 0.01 | — | — | 0.01 | late_tasks_5m | 0.9815 | — | 0 | 10 | 30 | -0.4056 |
| LR_a0p001_1216 | leaky | 0.001 | — | — | 0.001 | late_tasks_5m | 0.9955 | — | 0 | 10 | 30 | -0.07126 |
| LR_a0p0001_1216 | leaky | 0.0001 | — | — | 0.0001 | late_tasks_5m | 0.9805 | — | 0 | 10 | 28 | -0.2473 |
| E_a0p1_1216 | elu | 0.1 | — | 11.51 | 0.0007403 | late_tasks_5m | 0.8385 | — | 0.0005 | 5 | 118 | -1.326 |
| E_a0p01_1216 | elu | 0.01 | — | 9.21 | 0.0001142 | late_tasks_5m | 0.9205 | — | 0.0335 | 10 | 35.5 | -0.5247 |
| S_b0p3_1216 | silu | 0.3 | 4.262 | 55.2 | -0.01374 | late_tasks_5m | 0.7905 | 0.759 | 0 | 10 | 10 | -0.6748 |
| S_b3_1216 | silu | 3 | 0.4262 | 5.52 | -2.311e-09 | late_tasks_5m | 1 | 1 | 0.997 | 10 | 92 | -9.022e-06 |
| G_b0p3_1216 | gelu | 0.3 | 2.506 | 17.98 | -0.0003611 | late_tasks_5m | 0.9935 | 0.951 | 0.657 | 6 | 10 | -1.217 |
| G_b3_1216 | gelu | 3 | 0.2506 | 1.798 | -5.231e-22 | late_tasks_5m | 1 | 1 | 1 | 10 | 56 | -2.61e-06 |
| LR_a0p00001_1216 | leaky | 1e-05 | — | — | 1e-05 | late_tasks_5m | 0.9865 | — | 0 | 10 | 28 | -0.2337 |
| E_a0p001_1216 | elu | 0.001 | — | 6.908 | 1.859e-05 | late_tasks_5m | 0.9675 | — | 0.1675 | 10 | 28.5 | -0.3446 |
| S_b10_1216 | silu | 10 | 0.1278 | 1.656 | -4.991e-12 | late_tasks_5m | 1 | 0.998 | 1 | 10 | 45.5 | -0.0005745 |
| R_1216 | relu | 0 | 0 | 0 | 0 | final_step5000000 | 0.9855 | — | 0.994 | 10 | 28 | -0.2747 |
| LR_1216 | leaky | 0.1 | — | — | 0.1 | final_step5000000 | 0.6415 | — | 0 | 0 | 500 | -2.652 |
| E_1216 | elu | 1 | — | 13.82 | 0.007829 | final_step5000000 | 0.359 | — | 0 | 0 | 10 | -2.726 |

Spearman（予測子 対 median log10 U 5M・**ラベル無し**）

| scope | group | predictor | n | rho |
|---|---|---|---:|---:|
| pool | pool | m_minus | 17 | -0.3064 |
| pool | pool | u_fr | 17 | -0.3921 |
| pool | pool | frozen_plus_valley_frac | 17 | 0.5056 |
| within_family | leaky | m_minus | 5 | -0.7 |
| within_family | leaky | u_fr | 5 | nan |
| within_family | leaky | frozen_plus_valley_frac | 5 | nan |
| within_family | relu | m_minus | 1 | nan |
| within_family | relu | u_fr | 1 | nan |
| within_family | relu | frozen_plus_valley_frac | 1 | nan |
| within_family | silu | m_minus | 4 | 0.6 |
| within_family | silu | u_fr | 4 | -0.6 |
| within_family | silu | frozen_plus_valley_frac | 4 | 0.7379 |
| within_family | elu | m_minus | 4 | -1 |
| within_family | elu | u_fr | 4 | -1 |
| within_family | elu | frozen_plus_valley_frac | 4 | 1 |
| within_family | gelu | m_minus | 3 | 1 |
| within_family | gelu | u_fr | 3 | -1 |
| within_family | gelu | frozen_plus_valley_frac | 3 | 0.866 |

## S-mask（spec 字義との差）

- 実際の窓の記録点数: **10**（spec §5.3 の字義は 100）
- the host's _window_indices keeps only step %% task_period == 0, so a 10-task window has 10 records, not the 100 the spec's S-mask text asserts. The spec's own requirement that U^(10)_100 and U^(10)_500 equal the 1M/5M U_k, and the fact that the committed controls' U_k were built that way, both force the task-ends-only reading. A spec addendum is needed.

## Sanity

- S1_omp: **PASS**
- S_dial: **PASS**
- S_fd: **PASS**
- S_num: **PASS**
- S_limit_smooth: **PASS**
- S_elu_limit: **PASS**
- S_ref: **PASS**
- S_mob: **PASS**
- S_log_b: **PASS**
- S_pair: **PASS**
- S_dose: **PASS**
- S_taut: **PASS**
- S_cap: **FAIL**
- S_mask: **PASS**
- S6_floor_inherited: **PASS**
- S_CI_degeneracy: **PASS**
