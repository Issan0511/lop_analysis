# centered 腕の死因分解（事後解析）

> **格の自己申告（最重要）。** 本 spec は**事前登録ではない**。起草者（Claude）は起草前に、下記 E1–E6 に対応する数値をチャット内で `results/mlp2_phase1_0829/logs/*.npz` から観察している。したがって本 spec は `mlp2_centering_delay_posthoc_0830` と同格の「**事後解析の登録**」であり、判定ラベルは引用時に必ず事後の格で運ぶ。判定閾値は観察値から離した位置に置いたが、forking path の risk は残る。

> **新規学習・checkpoint 再計算なし。** 入力は commit 済み `mlp2_phase1_0829` のログだけ。

## 結論

- E1: **WALL_INVARIANT**
- E2: `L1w100_A1` **MU_CHANNEL_ALIVE** / `L2_A1` **MU_CHANNEL_ALIVE**
- E3: **BOUNDARY_CARRIES_DESCENT**
- E4: **CENTERING_REDUCES_BUT_NOT_REMOVES** / raw bias: **BIAS_DESCENT_WORSENED_BY_CENTERING**
- E5: **ABSORPTION_BROKEN_BY_EMA**

## E1 壁の位置

| arm | onset β | 95% CI |
|---|---:|---:|
| `L1w100_A1` | -2.262 | [-2.281, -2.247] |
| `L2_A1` | -2.236 | [-2.255, -2.218] |
| `L2_none` | -2.227 | [-2.276, -2.204] |

3対比の CI がすべて ±0.15 内なら `WALL_INVARIANT`。

## E2 チャネル

| arm | ρ_M | 95% CI | verdict |
|---|---:|---:|---|
| `L1w100_A1` | 0.6321 | [0.5593, 0.7396] | `MU_CHANNEL_ALIVE` |
| `L2_A1` | 0.719 | [0.6618, 0.8048] | `MU_CHANNEL_ALIVE` |

登録式は step 0 からの総和なので、EMA 初期化前の大きな M を含む。`M≈0` は centered 層の構成上ほぼ恒真だが、step 0 は例外である。step 10,000 起点の未登録感度分析は `verdict.csv` の `E2_SENSITIVITY_REPORT_ONLY` に併記し、E2 判定を差し替えない。

## E3 降下の局在

| arm | Δβ boundary | 95% CI | Δβ internal | 95% CI |
|---|---:|---:|---:|---:|
| `L1w100_A1` | -4.297 | [-4.59, -3.905] | 1.933 | [1.79, 2.19] |
| `L2_A1` | -2.19 | [-2.493, -1.912] | 0.5395 | [0.1232, 0.7964] |
| `L2_none` | -4.666 | [-4.99, -4.286] | 1.114 | [0.9024, 1.286] |
| `L2_Aall` | 28.71 | [11.94, 40.39] | -12.35 | [-19.13, -5.083] |

> **中央値は加法的でない。** `med(Δβ_boundary) + med(Δβ_internal)` と `med(Δβ_total)` は一致を要求しない。三量は別々に報告し、中央値の和で検算しない。

## E4 centering の境界効果

| paired contrast | point | 95% CI | verdict |
|---|---:|---:|---|
| Δβ boundary (`A1-none`) | 2.442 | [1.824, 2.831] | `CENTERING_REDUCES_BUT_NOT_REMOVES` |
| Δb_raw boundary (`A1-none`) | -0.1697 | [-0.2617, -0.08679] | `BIAS_DESCENT_WORSENED_BY_CENTERING` |

## E5 タスク内復活

| arm | layer | total | seed counts |
|---|---:|---:|---|
| `L1w100_A1` | 1 | 25723 | `[2817, 2697, 2464, 2742, 2505, 2392, 2512, 2580, 2272, 2742]` |
| `L2_A1` | 1 | 28094 | `[2928, 2893, 2304, 2548, 3404, 2370, 2548, 3034, 2730, 3335]` |
| `L2_A1` | 2 | 19786 | `[2176, 2212, 1761, 1510, 1901, 2201, 1933, 1824, 2128, 2140]` |
| `L2_Aall` | 1 | 3185 | `[253, 408, 280, 302, 510, 266, 173, 271, 391, 331]` |
| `L2_Aall` | 2 | 44044 | `[3518, 4449, 4147, 3503, 5195, 4033, 4083, 5474, 4576, 5066]` |
| `L2_none` | 1 | 0 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |
| `L2_none` | 2 | 9570 | `[1036, 927, 1105, 872, 1151, 819, 899, 1169, 830, 762]` |

件数は初期化遷移 step 0→1,000 を除く。その除外件数は `verdict.csv` の note に保存した。

centered 腕で `strict_dead` を『吸収した』とは書かない。EMA により入力がタスク内でも動き、復活が観測される。

## Sanity

| check | status |
|---|---|
| `S1` | **FAIL** |
| `S2` | **PASS** |
| `S3` | **PASS** |
| `S4` | **PASS** |
| `S5` | **PASS** |
| `S6` | **PASS** |

S1 の登録済み `-√5` 十分条件は layer 2 で FAIL。layer 2 の入力次元・支持は layer 1 の5次元 hypercube と異なるためで、E1–E4 が使う layer 1 の S1 は全件 PASS。layer 2 は E5 の moving-input 対照としてのみ用いる。

## E6 とスコープ

E6 の死亡・復活・churn・直近100タスク連続死率は `verdict.csv` に全腕・全層を記録した。`strict_dead` は当該タスク支持域上のラベルであり、不可逆な unit-ID の死ではない。

対象は condA・w100・T=10^4・batch=1・lr=0.01・center_alpha=0.01・10 seed・5M に限る。他幅、他T、condB、他最適化器へ外挿しない。`L1w100_A1` と `L2_none` は unpaired。因果的な腕間対比は `L2_A1` vs `L2_none` のみ。全数値は事後であり、Phase 1 の事前登録判定を上書きしない。
