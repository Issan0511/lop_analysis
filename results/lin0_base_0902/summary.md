# lin0_base_0902 summary

## Registered verdict

- G-base: **BASELINE_CONSTRUCTION_MATERIAL**
- 事前予測 B1 = `BASELINE_CONSTRUCTION_IMMATERIAL` → **外れ**。**ただし盲の予言ではない**（`LIN5` 版の結果を見たあとに立てた条件付き予測。spec §1・§6）
- **`PHENOMENON3_NOT_REPRODUCED` は上書きしていない。** それは `LIN5` を相手にした登録判定であり、本走はその外部妥当性を測るもの（spec §5）
- `LIN0` は隠れ層ゼロの単層線形回帰（原典 `MyLinear` 対応）、`LIN5` は leaky($a$=1.0)・隠れ 5 ユニット。**別物である**
- **1M 窓の格は非対称**: `LIN0` 相手は事前登録、`LIN5` 相手は事後登録（`7d77a90`）

## G0' 対 `LIN0`（Clopper–Pearson 95%）

| 腕 | 窓 | k | n | 除外 | 同値 | CP95 | ラベル |
|---|---|---:|---:|---:|---:|---|---|
| `R5` | task 491-500 | **18** | 20 | 0 | 0 | [0.6830, 0.9877] | **R5_ABOVE_LINEAR** |
| `LR5` | task 491-500 | **0** | 20 | 0 | 0 | [0.0000, 0.1684] | **LR5_BELOW_LINEAR** |
| `E5` | task 491-500 | **2** | 20 | 0 | 0 | [0.0123, 0.3170] | **E5_BELOW_LINEAR** |
| `R5` | task 91-100 | **10** | 20 | 0 | 0 | [0.2720, 0.7280] | **R5_NOT_SEPARATED_TIGHT** |
| `LR5` | task 91-100 | **0** | 20 | 0 | 0 | [0.0000, 0.1684] | **LR5_BELOW_LINEAR** |
| `E5` | task 91-100 | **0** | 20 | 0 | 0 | [0.0000, 0.1684] | **E5_BELOW_LINEAR** |

## G-base（6 ラベルの一致）

| 腕 | 窓 | `LIN5` 版 | `LIN0` 版 | 一致 |
|---|---|---|---|---|
| `R5` | task 491-500 | R5_INCONCLUSIVE_WIDE | R5_ABOVE_LINEAR | **NO** |
| `LR5` | task 491-500 | LR5_BELOW_LINEAR | LR5_BELOW_LINEAR | YES |
| `E5` | task 491-500 | E5_BELOW_LINEAR | E5_BELOW_LINEAR | YES |
| `R5` | task 91-100 | R5_NOT_SEPARATED_TIGHT | R5_NOT_SEPARATED_TIGHT | YES |
| `LR5` | task 91-100 | LR5_BELOW_LINEAR | LR5_BELOW_LINEAR | YES |
| `E5` | task 91-100 | E5_BELOW_LINEAR | E5_BELOW_LINEAR | YES |

動いたラベル: **R5@5m**

## 水準（報告のみ）

| 腕 | 窓 | median log10 U | 対 `LIN0` 対応差の中央値 | 完全崩壊 k/n | CP95 |
|---|---|---:|---:|---:|---|
| `LIN0` | task 491-500 | -0.3066 | +0.0000 | 0/20 | [0.0000, 0.1684] |
| `LIN0` | task 91-100 | -0.2717 | +0.0000 | 0/20 | [0.0000, 0.1684] |
| `LIN0_lr03` | task 491-500 | -0.2888 | +0.0202 | 0/20 | [0.0000, 0.1684] |
| `LIN0_lr03` | task 91-100 | -0.2541 | +0.0217 | 0/20 | [0.0000, 0.1684] |
| `LIN5` | task 491-500 | -0.2891 | +0.0118 | 0/20 | [0.0000, 0.1684] |
| `LIN5` | task 91-100 | -0.2625 | +0.0094 | 0/20 | [0.0000, 0.1684] |

`LIN0_lr03`（lr 0.03・原典 step_size 先頭値）は**報告のみで判定に入れない**。

## Sanity

- **preflight**: PASS (report)

**S0′ は fresh clone では回せない。** 参照 npz（親走・`gate_dose_0830` とも）が `.gitignore` でローカルのみだからである。replay 側の値は `preflight.json` に記録して commit した（spec §4）。

## 引用上の注意

- **`LIN0` を「原典の Linear ベースライン」と呼ぶときは、step size と環境（`flip_one` の有無）が原典の図と一致している保証はないことを併記する**
- **`LIN` 系を「線形ネットワーク」と呼ぶときは実装を併記する**（`LIN0` = 単層／`LIN5` = leaky($a$=1.0) の隠れ 5）
- **完全崩壊カウントと $k'$ を「LoP の発症率」と呼ばない**
- **`LIN0_lr03` の数値を判定に使わない**
- スコープは condA・$T=10^4$・batch 1・seed 20・5M。`LIN0` は lr 0.01、`LIN0_lr03` は lr 0.03
