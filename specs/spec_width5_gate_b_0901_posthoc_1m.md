# spec_width5_gate_b_0901_posthoc_1m: 1M 窓の符号検定と、非分離の内訳（事後登録）

proj_004 / 作成 2026-09-02 / 対象リポジトリ: `lop_analysis` / run id: `width5_gate_b_0901_posthoc_1m`
対象データ: `results/width5_gate_b_0901/verdict.csv`（committed・`2a62803`）
出所: vault `HANDOFF_LIN0_と1M窓_0901` W1 / 親: `specs/spec_width5_gate_b_0901.md`（凍結済み・`7c41210`）

> **状態: 解析手順登録・未実装・未実行。** この commit では本 spec だけを追加する。
> 集計コード・結果ディレクトリ・数値表は作らない。

---

## 0. 格の宣言（**ここを間違えない**）

**本解析の数値は spec 凍結前にチャットで算出済みである。事前登録ではない。**

2026-09-01 のチャット `P3_P2_0901` で、`verdict.csv` の `U_1m_seed_values` を `ast.literal_eval` で読み、`scipy.stats.beta` の Clopper–Pearson で $k_{R5}$=9/20 ほかを算出した。その値は vault `原典条件照合_0901` §5 と `HANDOFF_LIN0_と1M窓_0901` W1-3 / W1-4 に既に書かれている。**本 spec が固定するのは、その計算を repo の committed 出力として再現し、格を明示して保存する手順だけ**である。格は一貫して **post-hoc reanalysis registration**（`analysis_grade = registered_posthoc_not_preregistered`）とする。

**緩和材料は 1 つだけある。** 1M 窓（`task 91–100`）は親 spec §2 で**窓として事前に定義されており**（`phase1.window_1m_tasks`）、`verdict.csv` の `U_1m_seed_values` は本走時点で既に出力されていた。付け替えたのは**窓であって判定規則ではない**。**それでも事前登録とは呼ばない。**

**独立確認・事前登録効果とは呼ばない。** 出力する全行に `registered = 0` を立てる。

---

## 1. 問い

親 spec の G0（対 `LIN5` の seed 別符号検定・Clopper–Pearson 95%）は**末尾窓 `task 491–500`（5M）だけ**を登録した。ところが Dohare 図 B.10 の **x 軸は 1M までしか描かれていない**（`plots/online_performance.py` の `xticks=[0, 500000, 1000000]`・一次資料・vault `原典条件照合_0901` §3-1）。**当方の末尾窓は図に描かれた範囲の 5 倍先**である。

図と同じホライズンで同じ判定規則を当てるとラベルが変わるか、を事後に登録する。

あわせて、`R5` が末尾窓で `INCONCLUSIVE_WIDE` に落ちた**内訳**を 3 件記録する（§3）。

---

## 2. 入力（**これ以外を読まない**）

| ファイル | sha256 |
|---|---|
| `results/width5_gate_b_0901/verdict.csv` | `a9a89b32e4cf6dd2c46a8d65c282d31b61363957f99f873453463f1e9e0a09d3` |
| `results/width5_gate_b_0901/provenance.json` | `e9c845e66b619b7246544ee49e1776f4349e1b9ab0b784e5b8195d8e345c8bc0` |

**入力ハッシュが一致しなければ実行を中止する**（`SanityError`）。

**新しい走・再計装・生ログの読み出しはしない。** 使う列は `U_1m_seed_values` / `U_5m_seed_values`（腕別・seed 0–19 の生 `unfit`）だけである。

> `results/width5_gate_b_0901/logs/*.npz` は **repo に未添付**（1.0GB・ローカル保持）。本解析が `verdict.csv` だけを入力にするのは、**fresh clone で完全に再計算できる範囲に留める**ためでもある。

---

## 3. 登録する量（**すべて `registered = 0`**）

### P1（主・1M 窓の符号検定）

判定規則は親 spec §5 G0 と**同一**で、窓だけ `task 91–100` に付け替える。

$k_A$ = 1M 窓で腕 $A \in \{R5, LR5, E5\}$ の `unfit` が同 seed の `LIN5` より大きい seed 数。比較は生 `unfit` の厳密な `A > LIN5`。同値は $k_A$ に数えないが有限な試行として $n$ に残す。どちらかが非有限の seed だけ当該腕の比較から除外し、その実効 $n$ で Clopper–Pearson 95% を再計算する（親 spec の凍結時規則をそのまま運ぶ）。

ラベルも親 spec と同一:

- CI 下限 > 0.5 → `{A}_ABOVE_LINEAR`
- CI 上限 < 0.5 → `{A}_BELOW_LINEAR`
- CI が 0.5 を含み、かつ CI ⊆ [0.20, 0.80] → `{A}_NOT_SEPARATED_TIGHT`
- CI が 0.5 を含み、帯からはみ出す → `{A}_INCONCLUSIVE_WIDE`

**主判定は出さない。** 親 spec の `PHENOMENON3_*` は末尾窓で確定済み（`PHENOMENON3_NOT_REPRODUCED`）であり、**事後の窓で主判定を上書きしない**。1M 窓の出力は 3 腕のラベルまでとする。

### P2（付随・非分離の内訳 3 件）

1. **`LIN5` の 1M → 5M 変化**: seed 内差 `U_5m − U_1m` の中央値、悪化した seed 数、腕中央値の差。**「`LIN5` も 5M で劣化する」という記述が支持されるか**を判定する
2. **`R5` の二峰性**: 末尾窓で完全崩壊した seed（下記 3）と、しなかった seed に分け、それぞれ何 seed が `LIN5` より上かを出す
3. **完全崩壊カウント**: `unfit` ≥ 0.999 の seed 数と Clopper–Pearson 95% を **8 腕すべて**で出す。**0.999 は較正の要る閾値ではなく、残差分散が信号分散に等しい縮退点**（`unfit` = residual_var / signal_var の定義上の 1.0 に対する有限精度マージン）である

### 同値・欠測

P1・P2 とも、非有限値が現れた場合は当該 seed を落とし実効 $n$ を報告する。**落ちた seed 数を `summary.md` に明示する。**

---

## 4. 出力

`results/width5_gate_b_0901/posthoc_1m/` に出す。**親走の成果物（`results/width5_gate_b_0901/` 直下）は 1 バイトも書き換えない。**

| ファイル | 内容 |
|---|---|
| `verdict.csv` | P1 の 3 腕 × 1M/5M 窓 + P2-3 の 8 腕。全行 `registered = 0` |
| `bimodality.csv` | P2-1・P2-2 |
| `summary.md` | 上記の表と、§0 の格宣言・§5 の引用注意 |
| `provenance.json` | 入力ハッシュ・`analysis_grade`・出力ハッシュ |

`summary.md` の冒頭に **「1M 窓は事後登録。事前登録は末尾窓のみ」** を書く。

---

## 5. 引用上の注意

- **1M 窓は事後登録。** 親 spec が事前登録したのは末尾窓のみ
- **1M 窓のラベルで `PHENOMENON3_NOT_REPRODUCED` を上書きしない**（§3 P1）
- **完全崩壊カウントを「LoP の発症率」と呼ばない。** 測っているのは縮退点への到達であって、絶対的な機能劣化の定義ではない
- **`LIN5` を「線形ベースライン」と書くときは leaky($a$=1.0)・隠れ 5 ユニットの実装であることを併記する。** 原典の Linear は隠れ層ゼロの単層線形回帰（vault `原典条件照合_0901` §2）で**別物**であり、その差は `LIN0` 腕（同 handoff W2）が入るまで閉じない
- **1M 窓を「原典の図と同じ条件」と書かない。** 揃うのはホライズンだけで、ビン幅・走の本数・step size・ベースライン構成は揃っていない（`原典条件照合_0901` §6）
- スコープは condA・$T=10^4$・batch 1・lr 0.01・seed 20

---

## 6. 実行前 sanity

- **S-input**: §2 の 2 ファイルの sha256 が一致すること。不一致なら中止
- **S-reproduce**: P1 の 5M 窓の $k_A$ とラベルが、親走の `verdict.csv` に記録済みの `k_above_LIN5_5m` / `sign_status` と**完全一致**すること。**同じ規則を同じ窓に当てて同じ値が出ることの確認**であり、これが合わなければ実装を疑う。不一致なら中止
- **S-known**: P1 の 1M 窓の $k_A$ が handoff W1-3 の値（`R5` 9 / `LR5` 0 / `E5` 0）と一致すること。**不一致なら実装を疑う**（handoff W1-3 の指示）。ゲートではなく照合として `summary.md` に PASS/FAIL を出す
