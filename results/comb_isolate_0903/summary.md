# comb_isolate_0903 — 櫛の分離・段 A（1 層・用量 12.16・5M）

spec: `specs/spec_comb_isolate_0903.md` / 事前登録 commit で凍結。数値の引用は `verdict.csv` と本ファイルからのみ。

**★ §5.1 の Issa 事前予測は §7.2（Claude）と逐語で同一で、独立の予言ではない。**結果を引くときは「起草側の予測（Issa 承認）」と 1 行で書く（`preregistration.prediction_provenance`・引用禁止 B の `lr_a1_0901` 先例）。

## S-cover（§6 の各項目 → 実装の対応先）

| §6 の項目 | 実装 | 出力 | 段 A で付くか |
| --- | --- | --- | --- |
| V5 井戸 1 個は救うか | _v5_label | verdict.csv:V5 | ○ |
| V6 水準の帰属 | _v6_label（EXACT_FIT の有無だけ） | verdict.csv:V6 | ○ |
| V7 深さ 2 | src/comb_mlp2_0903.py | results/comb_mlp2_0903/ | ×（段 B） |
| EXACT_FIT ガード | _exact_fit / _s_guard | verdict.csv:EXACT_FIT | ○ |
| E1 発症数 | _onset_stats | verdict.csv:n_onset_* | ○ |
| E2 水準 P3'/P5' | _contrast | layer_stats.csv | ○ |
| E3 発症時刻 k* | _onset_times / _kaplan_meier | onset_times.csv / onset_km.csv | ○ |
| at_well・in_band・frozen | _unit_tail | position_table.csv | ○ |
| 全ユニット span と |v| | _unit_tail | position_table.csv | ○ |
| 深さ十分位 | _unit_tail | depth_hist.csv | ○ |
| C1 の再現 | 未実装（走後の別解析） | — | × |

**★ 未実装 1 件**: §6 副次の「C1 の再現」は REPORT_ONLY で判定には入らないが、spec が「本走で `src/` に置く」と書いているので**未了である**（`weird_act_0903` から持ち越し）。

## 判定

| 判定 | ラベル | 腕 |
| --- | --- | --- |
| V5 | INCONCLUSIVE_EXACT_FIT | CB1f / CB1l 対 R_1216 / RB_dpi |
| V6 | LEVEL_FROM_SINGLE_LOBE | CB_a1（committed）対 CB1l / CB1f |
| V7 | — | STAGE_B (comb_mlp2_0903) |

**V5・V6・V7 は互いに独立の判定で、1 つに畳まない。**「ゲートで非線形性を買う代償」は §1 の**動機**であってラベルではない（spec §9）。

## `EXACT_FIT`（1M 窓 U の seed 中央値 ≤ 1e−8）

| 腕 | 1M 窓 U 中央値 | EXACT_FIT |
| --- | --- | --- |
| `CB1f_a1_1216` | 1.6458e-10 | **立つ** |
| `CB1l_a1_1216` | 1.1744e-09 | **立つ** |
| `RB_dpi_1216` | 2.9852e-03 | 立たない |
| `CB_a1_1216`（committed） | 1.5035e-13 | **立つ** |
| `R_1216`（committed） | 2.2430e-01 | 立たない |
| `LR_1216`（committed） | 6.2444e-03 | 立たない |
| `E_1216`（committed） | 5.5750e-03 | 立たない |
| `S_b1_1216`（committed） | 1.9496e-02 | 立たない |
| `RB_d2_1216`（committed） | 1.0330e-02 | 立たない |
| `RB_d4_1216`（committed） | 1.9930e-03 | 立たない |

**`EXACT_FIT` の腕の水準差を機構として引かない**（spec §9）。$n_{\rm onset}$ は引ける。

## 腕（末尾窓 = タスク 491–500 のタスク終端 10 点）

| 腕 | 活性化 | S-cap | n_onset 1M | n_onset 5M | median log10 U (5M) | 沈下率 | span 中央値（全ユニット） | at_well | in_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CB1f_a1_1216` | comb1_flat | OK | 0/10 | 10/10 | -0.2009 | 0.9965 | 6.709 | 0.0025 | — |
| `CB1l_a1_1216` | comb1_leaky | OK | 0/10 | 0/10 | -7.3564 | 0.91 | 5.852 | 0.0465 | — |
| `RB_dpi_1216` | band_leaky_dpi | OK | 0/10 | 0/10 | -2.1182 | 0.903 | 7.854 | — | 0.01 |

**`SN`（Snake）は負側に壁が無いので `submerged` を定義しない**（spec §4）。

## 数値発散（spec §6）

- **`SN_a1_1216` は `NUMERIC_DIVERGENCE`**。最初の発散 step **1,000**・発症 seed 1/10（seed [3]）。登録どおり当該腕だけを落とした。`SN` は錨なので判定腕ではない

## 水準の対比（対照は**別走の committed 値**・同一走の腕ではない）

| 鍵 | 腕 | 種別 | 窓 | 相手 | 点推定 | percentile CI | 等価判定 | 符号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `P3':CB1f_a1_1216:5M` | `CB1f_a1_1216` | P3prime | 5M | `R_1216` | +0.1248 | [-0.074, +0.202] | INCONCLUSIVE_WIDE | 3:7 |
| `P3':CB1f_a1_1216:1M` | `CB1f_a1_1216` | P3prime | 1M | `R_1216` | -9.2001 | [-10.730, -7.818] | BELOW_SOFT | 10:0 |
| `P5':CB1f_a1_1216:E_1216` | `CB1f_a1_1216` | P5prime | 5M | `E_1216` | +2.6412 | [+2.187, +3.334] | SHORT_OF_SOFT | 0:10 |
| `P5':CB1f_a1_1216:LR_1216` | `CB1f_a1_1216` | P5prime | 5M | `LR_1216` | +2.4991 | [+1.961, +2.901] | SHORT_OF_SOFT | 0:10 |
| `P3':CB1l_a1_1216:5M` | `CB1l_a1_1216` | P3prime | 5M | `R_1216` | -6.9840 | [-8.416, -6.258] | BELOW_SOFT | 10:0 |
| `P3':CB1l_a1_1216:1M` | `CB1l_a1_1216` | P3prime | 1M | `R_1216` | -9.1003 | [-11.137, -6.875] | BELOW_SOFT | 10:0 |
| `P5':CB1l_a1_1216:E_1216` | `CB1l_a1_1216` | P5prime | 5M | `E_1216` | -4.6743 | [-5.838, -2.663] | BELOW_SOFT | 10:0 |
| `P5':CB1l_a1_1216:LR_1216` | `CB1l_a1_1216` | P5prime | 5M | `LR_1216` | -4.5045 | [-6.637, -3.336] | BELOW_SOFT | 10:0 |
| `P3':RB_dpi_1216:5M` | `RB_dpi_1216` | P3prime | 5M | `R_1216` | -1.9190 | [-2.235, -1.761] | BELOW_SOFT | 10:0 |
| `P3':RB_dpi_1216:1M` | `RB_dpi_1216` | P3prime | 1M | `R_1216` | -1.8903 | [-2.191, -1.335] | BELOW_SOFT | 10:0 |
| `P5':RB_dpi_1216:LR_1216` | `RB_dpi_1216` | P5prime | 5M | `LR_1216` | +0.4115 | [-0.104, +0.811] | INCONCLUSIVE_WIDE | 3:7 |
| `V5:CB1f-R_1216` | `CB1f_a1_1216` | V5 | 5M | `R_1216` | +0.1248 | [-0.074, +0.202] | INCONCLUSIVE_WIDE | 3:7 |
| `V5:CB1l-RB_dpi_1216` | `CB1l_a1_1216` | V5 | 5M | `RB_dpi_1216` | -5.1475 | [-6.122, -4.017] | BELOW_SOFT | 10:0 |
| `CB1l-CB_a1_1216` | `CB1l_a1_1216` | multilobe | 5M | `CB_a1_1216` | +5.2130 | [+3.816, +6.055] | SHORT_OF_SOFT | 0:10 |

### 対照の出所

- `R_1216`: results/gate_dose_0830 / `verdict.csv`
- `LR_1216`: results/gate_dose_0830 / `verdict.csv`
- `E_1216`: results/gate_dose_0830 / `verdict.csv`
- `S_b1_1216`: results/gate_dial_0902 / `verdict.csv`
- `CB_a1_1216`: results/weird_act_0903 / `verdict.csv`
- `RB_d2_1216`: results/weird_act_0903 / `verdict.csv`
- `RB_d4_1216`: results/weird_act_0903 / `verdict.csv`

## 引用上の注意

- 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。「起きない」と書かない
- **用量 1 点（12.16）・1 層・5M・float32 の主張である。** 引くときは用量を添える
- 4 族すべて本走のための合成活性化。`SN`（Snake）は先行があるが**推奨として引かない**
- **`EXACT_FIT` の腕の水準差を機構として引かない**
- **§5.1 の予測は独立の予言ではない**（起草側の値を Issa が承認したもの）
- 段 C（実ベンチ）は本 spec の外。段 A・B の結果から段 C を自動起案しない
