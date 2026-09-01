# spec_lin0_base_0902: 原典どおりの線形ベースライン `LIN0`（隠れ層なし）で G0 を引き直す

proj_004 / 作成 2026-09-02 / 対象リポジトリ: `lop_analysis` / run id: `lin0_base_0902`
出所: vault `HANDOFF_LIN0_と1M窓_0901` W2 / 親: `specs/spec_width5_gate_b_0901.md`（凍結済み・`7c41210`）
関連: `specs/spec_width5_gate_b_0901_posthoc_1m.md`（1M 窓の事後登録・`cd46027`）

> **状態: 凍結**（2026-09-02。§6 の事前予測 B1 は vault `35c7d49` で記入済み。§7 の裁定 4 点は handoff の起草側所見を採用）。この commit では本 spec だけを追加する。

---

## 0. なぜ引き直すか

`width5_gate_b_0901` の G0 は**すべて `LIN5` を比較相手にしている**。ところが原典の Linear ベースラインは**隠れ層ゼロの単層線形回帰**である（`lop/nets/linear.py` の `MyLinear = nn.Linear(input_size, 1)`、`cfg/flip_one/linear.json` の `agent: "linear"` / `num_features: 0`。一次資料・vault `原典条件照合_0901` §2）。

当方の `LIN5` は **leaky($a$=1.0) の隠れ 5 ユニット付き**で、関数クラスは同じ（$x$ のアフィン関数）だが **$x \mapsto v^\top(Wx+b)+c$ のボトルネック経由**という別のパラメータ化である。最適化の力学も初期化スケールも違う。**同じベースラインではない。**

**比較相手を原典どおりに作り直して、6 個のラベルが動くかを見る。**

---

## 1. 格の宣言

- **§5 の G0′ と G-base は事前登録である。** `LIN0` は本 spec 凍結時点で一度も走っていない
- **ただし §6 の予測 B1 は盲の予言ではない。** `LIN5` 版の結果（`2a62803`）を見たあとに立てた条件付き予測である。当たっても「知った上で立てた予測が当たった」までである。`summary.md` にこの区別を書く
- **1M 窓を G0′ に含めるのは本 spec が初めての事前登録である。** `LIN5` 版の 1M 窓は事後（`cd46027`）だが、`LIN0` は未走なので、`LIN0` を相手にした 1M 窓の判定は事前登録として成立する。**この非対称を `summary.md` に明記する**
- §7 の裁定 4 点は handoff の起草側所見をそのまま採用したもので、Issa の独立裁定ではない

---

## 2. 設計

### 2.1 ネット定義（原典 `MyLinear` 対応）

$$\hat y = a^\top x + c,\qquad a \in \mathbb R^{d},\ c \in \mathbb R,\ d = 20$$

- **隠れ層なし・活性化なし。** $d$ は condA の `m` = 20
- 初期化: 原典は `nn.init.kaiming_uniform_(fc1.weight, nonlinearity='linear')` ＋ `bias.data.fill_(0.0)`。PyTorch の該当規則は gain=1 の一様分布 $U(-b_a, b_a)$、$b_a=\sqrt{3/d}$。**`envs.kaiming_mlp_params` の出力層規則（`bv = sqrt(3.0/h)`）を $h \to d$ に読み替えたもの**である
- $c$ は 0 初期化
- R 系列でベクトル化する
- 勾配は既存と同じ**係数 2 の二乗誤差**: $\partial L/\partial a = 2\delta x$、$\partial L/\partial c = 2\delta$（$\delta = \hat y - y$）。`grads_centered_elu` の `d2 = 2.0 * delta` と同じ規約

### 2.2 ★ 乱数の割り当て（**ここが本 spec の最大のリスク**）

`train.make_gens` は基底 `SEED_BASE["A"] + width + offset` から **`init` / `input` / `teacher` / `eval` / `method` / `noise` の 6 本を別々に切る**。したがって:

- **`width = 5` を渡す**（裁定 1）。基底が `width5_gate_b_0901` の w5 4 腕と一致する
- **`init` の消費本数が変わっても `input` / `teacher` / `eval` は動かない。** 生成器が別だからである。`LIN0` は `init` から $a$ を 1 回引くだけで、`W`/`b`/`v`/`c` を引く `VecMLPL` と消費量が違うが、**入力ストリーム・教師・eval 集合は bit 一致する**
- **既存腕の setup 経路には一切触れない。** `LIN0` は独自の setup を持つ

**この主張は §4 の S-share と S0′ で実測確認する。論証で済ませない。**

### 2.3 腕

| 腕 | 構成 | lr | 役割 |
|---|---|---|---|
| **`LIN0`** | 隠れ層なし線形回帰 | **0.01** | **新しい主ベースライン**（他腕と lr を揃える） |
| `LIN0_lr03` | 同上 | **0.03** | 原典対応（`cfg/flip_one/linear.json` の step_size 先頭値。プロットは `setting_idx=0` で取る）。**報告のみ** |

- seed 0–19・5M・condA・$T=10^4$・batch 1・CPU
- **`generator_offset = 202609011921` 据え置き**（裁定 3）。既存腕と実現を共有するために必須
- bootstrap seed も `202609011921` 据え置き。G0′ は Clopper–Pearson なので bootstrap を使わない
- **既存 8 腕は再走しない。** `R5` / `LR5` / `E5` の per-seed `unfit` は committed の `results/width5_gate_b_0901/verdict.csv` から読む
- `lr` は既存 `arm_runs` が `common.lr_main` 一律なので、**腕ごとの `lr` を config で明示し、setup 後に `st["lr"]` を上書きする**。上書きした値は `config_used.yaml` と `provenance.json` に出す

### 2.4 記録

`LIN0` には隠れユニットが無いので、P-3 の 3 軸（$s_i$・$m_i$・$R_c$）は**定義されない**。記録するのは 32 点厳密支持上の run 量だけとする:

`unfit` = `residual_var / signal_var`、`eval_loss_exact`、`signal_var`、`residual_var`。定義は親と同一（`residual = yhat − y_teacher`、`var` は `unbiased=False`）。probe は 1000 step ごと。

---

## 3. 入力（committed からのみ読む）

| ファイル | sha256 |
|---|---|
| `results/width5_gate_b_0901/verdict.csv` | `a9a89b32e4cf6dd2c46a8d65c282d31b61363957f99f873453463f1e9e0a09d3` |
| `results/width5_gate_b_0901/provenance.json` | `e9c845e66b619b7246544ee49e1776f4349e1b9ab0b784e5b8195d8e345c8bc0` |

**不一致なら中止。** 親の生ログ npz は repo に未添付（ローカルのみ）なので、**判定に使う比較相手の値は `verdict.csv` からだけ取る**。

---

## 4. 実行前 sanity（**FAIL は本走禁止**、S-mech-na を除く）

- **S-share（新規・gate）**: `LIN0` の step 0 の**厳密支持 $X$（32×20）と教師出力 $y$ が、`LIN5` の step 0 のそれと bit 一致**すること。G0′ の対応比較はこの一致の上にしか立たない。`flip_state` と env の状態ハッシュも併せて記録する
- **S0′（gate・2 本立て）**: §2.2 の乱数消費の懸念を潰すのが主目的である。**親走の S0′（`gate_dose_0830` 参照）は w100 腕しか replay していない**ので、w5 腕については新しい参照が要る。
  - **S0′-w5**: `LIN0` を足した実装で `R5` / `LR5` / `E5` / `LIN5` を **`generator_offset = 202609011921`・seed 0–19・30,000 step** で replay し、probe step 0…30,000 の **`unfit`・`eval_loss_exact`・`layer1_p_hat` が `results/width5_gate_b_0901/logs/*.npz` の同 step と bit 一致**すること。**参照 npz は repo に未添付（ローカル 1.0GB）なので、照合前に committed の親 `provenance.json` の `output_sha256` と全数突き合わせて、参照が commit された走のものであることを確認する。** replay 側の per-seed 値は `s0prime.json` に書き出して commit し、fresh clone でも replay を回して突き合わせられるようにする
  - **S0′-w100**: 親走と同じ replay（`R100` / `E100` / `LR100`・seed 0–9・`generator_offset = 0`・30,000 step・参照 `results/gate_dose_0830`）を回し、`2ccf6be` / 親 preflight と同じく `differences` が空であること
  - **★ どちらの参照 npz も repo には無い**（`gate_dose_0830/logs/` も `.gitignore` 済みでローカルのみ）。**S0′ は fresh clone では回せない。** これは本走で新たに作った制約ではなく repo の既存条件だが、`summary.md` に明記する。replay 側の値を commit することで、少なくとも「同じマシンで再実行すれば同じ値が出る」ところまでは残す
- **S-lin0（gate）**: `LIN0` の出力が入力の厳密なアフィン関数であること（2 点で線形性を相対 1e-12 以内で確認）。かつ**隠れ層に相当するテンソルが存在しない**こと（`net` が `Ws` / `bs` / `v` を持たない）
- **S-init（gate）**: `LIN0` の初期 $a$ が $U(-\sqrt{3/20}, +\sqrt{3/20})$ に整合すること。**下限・上限が区間内に収まり、20×20=400 個の分位点（0.1/0.25/0.5/0.75/0.9）が理論値と 0.05 以内**で一致すること。$c$ は厳密に 0
- **S-floor（gate）**: 未フィット率の床は 1 層系 `1e-16` を `gate_dose_0830` から継承（親と同じ値であることの確認）
- **S-mech-na（ゲートではない）**: `LIN0` に隠れユニットが無いことを記録し、`mechanism.csv` の 3 軸列を**空にする**。恒真な列を埋めない（`現在地` 穴7 の裁定と同型）

---

## 5. 事前登録判定

実験単位は seed（n=20）。**Clopper–Pearson 95%**（bootstrap 不要）。

### G0′（主）

腕 $A \in \{R5, LR5, E5\}$・窓 $w \in \{$末尾 `task 491–500`, 1M `task 91–100`$\}$ ごとに独立に:

$k'_{A,w}$ = 窓 $w$ で腕 $A$ の `unfit` が同 seed の **`LIN0`** より大きい seed 数。

比較は生 `unfit` の厳密な `A > LIN0`。同値は $k'$ に数えないが有限な試行として $n$ に残す。どちらかが非有限の seed だけ当該比較から除外し、その実効 $n$ で Clopper–Pearson を再計算する（親 spec の凍結時規則をそのまま運ぶ）。

ラベルは親 spec と同一:

- CI 下限 > 0.5 → `{A}_ABOVE_LINEAR`
- CI 上限 < 0.5 → `{A}_BELOW_LINEAR`
- CI が 0.5 を含み、かつ CI ⊆ [0.20, 0.80] → `{A}_NOT_SEPARATED_TIGHT`
- CI が 0.5 を含み、帯からはみ出す → `{A}_INCONCLUSIVE_WIDE`

**到達可能性は親 spec §5 のまま**: `ABOVE` に $k\ge15$、`BELOW` に $k\le5$、`NOT_SEPARATED_TIGHT` は $k\in\{9,10,11\}$ でのみ到達可能。

### G-base（帰趨・**単独の主判定**）

6 個のラベル（3 腕 × 2 窓）が `LIN5` 版と**全部一致**するか。`LIN5` 版の 6 ラベルは:

| 腕 | 末尾窓 | 1M 窓 |
|---|---|---|
| `R5` | `R5_INCONCLUSIVE_WIDE` | `R5_NOT_SEPARATED_TIGHT` |
| `LR5` | `LR5_BELOW_LINEAR` | `LR5_BELOW_LINEAR` |
| `E5` | `E5_BELOW_LINEAR` | `E5_BELOW_LINEAR` |

（末尾窓は事前登録 `2a62803`、1M 窓は事後登録 `7d77a90`）

- 6 個とも一致 → **`BASELINE_CONSTRUCTION_IMMATERIAL`**。`原典条件照合_0901` §2 の差は判定に効かず、同 §5 の読み替えが確定する
- 1 個でも変わる → **`BASELINE_CONSTRUCTION_MATERIAL`**。**`width5_gate_b_0901` の G0 は原典比較には使えない**ことになり、結果ノートの帰趨を書き換える

**`PHENOMENON3_NOT_REPRODUCED` は上書きしない。** それは `LIN5` を相手にした登録判定であり、本走はその**外部妥当性**を測るものである。G-base が `MATERIAL` でも、親の判定が「間違いだった」ことにはならない（比較相手が違うだけである）。

### 報告のみ

- `LIN0` と `LIN5` の水準差（末尾窓・1M 窓の $\log_{10}U$ 中央値と対応差の中央値）
- `LIN0` の完全崩壊カウント（`unfit` ≥ 0.999）
- `LIN0_lr03` の同一量すべて（**判定には一切入れない**）

---

## 6. 事前予測

> **★ 格の限定。** `LIN5` 版の結果を見たあとに立てた条件付き予測である。盲の予言ではない。

| # | 量 | 起草側候補値 | **Issa 記入**（vault `35c7d49`） |
|---|---|---|---|
| **B1** | G-base の帰趨 | **`BASELINE_CONSTRUCTION_IMMATERIAL`**（全一致する） | `BASELINE_CONSTRUCTION_IMMATERIAL` |

**候補値の根拠（未登録）**: `LIN5` と `LIN0` は同じ関数クラスで、`LIN5` の末尾水準（$\log_{10}U$ 中央値 −0.289）は `LIN100`（−0.260）とほぼ同じ。幅を 5 → 100 に変えても線形腕の水準がほぼ動かないので、隠れ層を外しても動かないと予想する。**外れた場合は「ボトルネックのパラメータ化そのものが水準を決めていた」ことになり、それはそれで報告価値がある。**

---

## 7. 裁定（handoff の起草側所見を採用）

| # | 決めたこと |
|---|---|
| 1 | `LIN0` の「幅」引数は **`5`**。既存 4 腕と実現を共有できる。S0′ と S-share 必須 |
| 2 | `LIN0_lr03` を**入れる**。1 腕ぶんで原典の step size 0.03 を潰せる |
| 3 | `generator_offset` / bootstrap seed は **`202609011921` 据え置き** |
| 4 | 主ベースラインは、G-base が `IMMATERIAL` なら `LIN5` と**併記**、`MATERIAL` なら `LIN0` に**切り替え** |

---

## 8. コスト

`LIN0` はパラメータ 21 個で、律速は 1000 step ごとの 32 点厳密評価。**2 腕 × 20 seed × 5M で 30 分以内**の見込み。

---

## 9. 引用上の注意

- **`LIN0` を「原典の Linear ベースライン」と呼ぶときは、step size と環境（`flip_one` の有無）が原典の図と一致している保証はないことを併記する**（`原典条件照合_0901` §3-4）
- **`LIN` 系を「線形ネットワーク」と呼ぶときは実装を併記する**（`LIN0` = 単層／`LIN5` = leaky($a$=1.0) の隠れ 5）
- **完全崩壊カウントを「LoP の発症率」と呼ばない**
- **`k'` を「LoP の発症率」と呼ばない。** 測っているのは線形ベースラインとの相対位置である
- **`LIN0_lr03` の数値を判定に使わない。** 報告のみ
- **G-base の帰趨で `PHENOMENON3_NOT_REPRODUCED` を上書きしない**（§5）
- **1M 窓の非対称に注意**: `LIN0` 相手は事前登録、`LIN5` 相手は事後登録である
- スコープは condA・$T=10^4$・batch 1・seed 20・5M。`LIN0` は lr 0.01、`LIN0_lr03` は lr 0.03

---

## 10. 本 spec が前提にしている未登録の判断

- **§2.2 の乱数割り当ての主張**は `train.make_gens` の実装からの論証である。**S-share と S0′ が実測で裏を取るまで登録量として引かない**
- 原典 `MyLinear` の初期化を PyTorch の `kaiming_uniform_(nonlinearity='linear')` 規則へ翻訳した部分は**一次資料の読み替え**であり、原典の実行時の値と突き合わせたわけではない
- `LIN0` が「原典の Linear と同じもの」だと言えるのは**ネット構造と初期化則まで**で、環境・ホライズン・走の本数・ビン幅は依然として揃っていない（`原典条件照合_0901` §6）
