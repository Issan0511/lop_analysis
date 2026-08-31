# centered 腕の死因分解（事後解析の登録・走なし）

親: [[容量不足レジームの機構_0830]] §10・§11 / 関連: [[吸収不等式の命題化_0824]]（Q26 系1）・[[マージン恒等式と消灯点_0822]] §4.2・[[中心主張v4.2草案_0830]] 柱1
状態: **spec（事後解析の登録）** / 作成 2026-08-31 / run id: `centered_death_posthoc_0831`

> **格の自己申告（最重要）。** 本 spec は**事前登録ではない**。起草者（Claude）は起草前に、下記 E1–E6 に対応する数値をチャット内で `results/mlp2_phase1_0829/logs/*.npz` から観察している。したがって本 spec は `mlp2_centering_delay_posthoc_0830` と同格の「**事後解析の登録**」であり、判定ラベルは引用時に必ず事後の格で運ぶ。判定閾値は観察値から離した位置に置いたが、forking path の risk は残る。**この段落を summary.md に転記すること。**

> **新規学習・checkpoint 再計算なし。** 入力は `results/mlp2_phase1_0829/` の commit 済み成果物のみ。

---

## 0. 問い

`L1w100_A1`（condA・w100・隠れ1層・第1層 centering・5M）で layer 1 の `strict_dead` が 5M で約 0.47 に達する。**µ を消したはずの腕で、何が壁まで運んでいるのか。**

[[容量不足レジームの機構_0830]] §11.1 は「centering は治癒ではなく遅延」までを 5M で確定させたが、**遅延の残りを誰が担っているか**は開いている。本 spec はそれを既存ログだけで座標に分解する。

## 1. データ

- `results/mlp2_phase1_0829/logs/{arm}_seed{s}.npz`、`s = 0..9`
- 腕: `L1w100_A1`（隠れ1層×100・第1層 centering）／`L2_A1`（隠れ2層・第1層のみ centering）／`L2_none`（隠れ2層・無介入）。`L2_Aall` は REPORT_ONLY で併記のみ
- 層: 特記なき限り layer 1。`L2_*` の layer 2 は E5 の対照でのみ使う
- 記録格子: step 0 … 5,000,000 の 1000 刻み（5001 点）。`lop_every = 1000`、`task_period = 10000`
- `L2_A2` は `NUMERIC_DIVERGENCE` のため全 endpoint から除外する

### 1.1 派生量（`src/mlp2_phase1.py` の定義に従う）

| 記号 | 定義 | npz 列 |
|---|---|---|
| $\sigma_{u,t}$ | $\sqrt{\mathbb E[(W_u\cdot(x-\mu))^2]}$（ゲートマージンの分母） | `layer{L}_denom` |
| $M_{u,t}$ | $(W_u\cdot\mu_t)/\sigma_{u,t}$ | `layer{L}_M` |
| $B_{u,t}$ | $b_u/\sigma_{u,t}$ | `layer{L}_B` |
| $\beta_{u,t}$ | $M+B = (W_u\cdot\mu_t+b_u)/\sigma_{u,t}$ | 導出 |
| $b^{\rm raw}_{u,t}$ | $B\cdot\sigma$（生のバイアス） | 導出 |
| dead | $\hat p_{u,t}=0$ | `layer{L}_p_hat` |

**遷移の分類**（$t\to t+1$、`step[t+1]` で判定）:

- **境界窓** `step % 10000 == 1000` … flip が変わった 1000 step を含む遷移。500 個
- **内部** それ以外。4500 個

分類の正しさは S3 で検査する。

### 1.2 欠測の規約

`denom < sigma_tol` のユニットは `M`・`B` が NaN。**NaN 記録が全記録の 1% を超えるユニットは E1–E4 から除外し、除外数を腕・seed ごとに `verdict.csv` に載せる。** 集計は `nansum` / `nanmedian`。

### 1.3 不確かさ

seed クラスタ bootstrap、$B=10{,}000$、**percentile**（[[運用ルール]] §5 教訓⑬に従い studentized は既定で退化とみなす。退化検出は実装し、退化したら percentile に落ちた旨を記録する）。bootstrap seed は **`20260829` を継承**（`20260830` は `generator_offset` として既用。日付として流用しない）。

---

## 2. Endpoint と判定基準（**実行前に凍結**）

### E1 主判定A — 壁は動いていない

各ユニットについて、最終記録で dead なら「**最後の死の入り口**」= 最終の連続 dead 区間の先頭 index を取る。index 0 で始まる区間は除外（開始前が観測できない）。その時点の $\beta$ を $\beta_{\rm onset}$ とする。

- 統計量: 腕ごとの seed 別中央値 → seed クラスタ bootstrap
- 判定: 3 腕（`L1w100_A1` L1 / `L2_A1` L1 / `L2_none` L1）の**全対比**の 95% CI が $[-0.15,+0.15]$ に収まれば **`WALL_INVARIANT`**、一つでも外れれば **`WALL_MOVES`**
- 併記（REPORT_ONLY）: 中央値が理論域 $-\kappa\in[-\sqrt5,-1]$ の近傍にあるか。$\kappa := \lVert w_{\rm free}\rVert_1/\lVert w_{\rm free}\rVert_2$（[[吸収不等式の命題化_0824]] 命題1）

> **`WALL_INVARIANT` が出たときだけ**「µ は壁の位置ではなく到達を支配する」と書いてよい。

### E2 主判定B — 殺し屋のチャネル交代

centered 腕（`L1w100_A1`・`L2_A1` の layer 1）について、ユニットごとに

$$\Delta\beta = \sum_t \Delta B_t + \sum_t \Delta M_t$$

- 統計量: $\rho_M := \lvert\sum\Delta M\rvert / \lvert\Delta\beta\rvert$ の seed 別中央値
- 判定: 95% CI の**上限 < 0.10** なら **`BIAS_CHANNEL_DOMINANT`**、下限 > 0.30 なら **`MU_CHANNEL_ALIVE`**、それ以外は **`CHANNEL_MIXED`**
- 併記: 生の $b$ と $\sigma$ の 5M 倍率（`step=10000` を起点。step 0 は初期化直後で $\sigma$ が別レジーム）

### E3 主判定C — 降下の局在（本 spec の中心）

ユニットごとに加法分解:

$$\Delta\beta^{\rm bnd}_u := \sum_{t\in\text{境界窓}}\Delta\beta_{u,t},\qquad \Delta\beta^{\rm int}_u := \sum_{t\in\text{内部}}\Delta\beta_{u,t}$$

- 統計量: 腕ごと seed 別中央値 → bootstrap。$\Delta b^{\rm raw}$ と $\Delta\log\sigma$ についても同じ分解を出す
- 判定（`L1w100_A1` L1 が主）:
  - CI$(\Delta\beta^{\rm bnd})$ の上限 < 0 **かつ** CI$(\Delta\beta^{\rm int})$ の下限 > 0 → **`BOUNDARY_CARRIES_DESCENT`**（内部は回復側）
  - 前半のみ成立、かつ $\lvert\text{med}\,\Delta\beta^{\rm bnd}\rvert > \lvert\text{med}\,\Delta\beta^{\rm int}\rvert$ → **`BOUNDARY_DOMINANT`**
  - それ以外 → **`NOT_LOCALIZED`**
- 併記: 1 遷移あたり率（境界 = 総和/500、内部 = 総和/4500）と符号
- 併記: 最終死の onset が境界窓に落ちる割合（一様なら 10%）

> **中央値は加法的でない。** $\text{med}(\Delta\beta^{\rm bnd})+\text{med}(\Delta\beta^{\rm int})\ne\text{med}(\Delta\beta)$。三つとも別々に報告し、和で検算しない。**この注意を summary.md に書く。**

### E4 主判定D — centering は境界を守れているか

`L2_A1` と `L2_none` の layer 1 は init・教師・入力列・flip 軌道が bit 一致（A は乱数を消費しない: `intervention.consumes_rng: false`）。**ユニット index で対応づけた paired 対比**を取る。

- 統計量: $\Delta\beta^{\rm bnd}(\text{A1}) - \Delta\beta^{\rm bnd}(\text{none})$、および $\Delta b^{\rm raw,bnd}$ の同じ差
- 判定:
  - $\Delta\beta$ 側の差の CI 下限 > 0（centered のほうが降下が浅い）**かつ** centered 自身の CI 上限 < 0 → **`CENTERING_REDUCES_BUT_NOT_REMOVES`**
  - centered 自身の CI が 0 を含む → **`CENTERING_REMOVES_BOUNDARY_DESCENT`**
  - 差の CI が 0 を含む → **`CENTERING_NO_BOUNDARY_EFFECT`**
- **別立てで** $\Delta b^{\rm raw,bnd}$ の差の符号を判定: 差の CI 上限 < 0 なら **`BIAS_DESCENT_WORSENED_BY_CENTERING`**

> S-pair（[[多層Phase1_spec_0829]] の対応づけ検査）が PASS していることを引用して paired を正当化する。**S-pair が PASS でなければ E4 を unpaired に落とし、その旨を記録する。**

### E5 主判定E — タスク内吸収不変性は成立しているか

「タスク内復活」= 同一 flip の遷移で dead → alive。[[吸収不等式の命題化_0824]] 系1 は**タスク内では復活 0** を定理として要求する。

- 統計量: 腕・層ごとの件数（全 seed 合計、および seed 別）
- 判定:
  - `L2_none` layer 1 が**全 seed で厳密に 0** かつ centered 腕（`L1w100_A1`・`L2_A1` の layer 1）が全 seed で > 0 → **`ABSORPTION_BROKEN_BY_EMA`**
  - `L2_none` layer 1 が 0 でない → **`THEOREM_VIOLATED`**（この場合は測定側の欠陥を疑い、E1–E4 の判定を保留して原因を先に潰す）
  - centered 腕が 0 → **`ABSORPTION_HOLDS_UNDER_CENTERING`**
- **対照（必須）**: `L2_none` の **layer 2** の件数も出す。layer 2 は入力（第1層の活性）がタスク内で動くので > 0 が期待される。これにより「破れの原因は centering そのものではなく**入力が動くこと**」であることを示す

### E6 REPORT_ONLY — ラベルの信頼性

- 死亡遷移数 / 復活遷移数 / churn 比
- 最終記録で dead のユニットのうち、**直近 100 タスク（1000 記録）を通じて連続 dead** の割合
- 目的: `strict_dead` の水準を引くときに、どれだけが芯でどれだけが flicker かを添えるため

---

## 3. サニティ（すべて PASS が必要。FAIL なら該当 endpoint を出さない）

| id | 内容 | 基準 |
|---|---|---|
| **S1** | 吸収不等式の両端。$\hat p=0 \Rightarrow \beta\le-1$、および $\beta\le-\sqrt5 \Rightarrow \hat p=0$（$\kappa\in[1,\sqrt5]$ から厳密） | 違反 **0 件**（NaN 除く。全腕・全層・全記録） |
| **S2** | npz と `layer_stats.csv` の整合。`task_end` 行に対応する記録点で `median_M`・`median_B`・`strict_dead` が一致 | 相対誤差 < 1e-9、`strict_dead` は完全一致 |
| **S3** | 境界窓の定義。`flip_state` の変化が `step % 10000 == 0` の遷移でのみ起き、seed あたり 499 回 | 完全一致 |
| **S4** | `layer{L}_strict_dead == (p_hat == 0).sum(axis=1)` | 完全一致 |
| **S5** | 恒真チェック（[[運用ルール]] §5 教訓②・⑩）。**centered 層では $M\approx0$ が構成上ほぼ恒真**なので、E2 の $\rho_M$ は「小さいこと」を主張する側であって発見ではない。この旨を verdict に注記する | 注記の存在 |
| **S6** | 除外ユニット（§1.2）の数が全腕で全体の 5% 未満 | 5% 以上なら E1–E4 を保留 |

---

## 4. 出力

`results/centered_death_posthoc_0831/` に

- `verdict.csv` … endpoint・腕・層・点推定・CI・判定ラベル・判定基底
- `summary.md` … §格の自己申告（冒頭）、E1–E6 の表、中央値の非加法性の注意、スコープ
- `unit_decomposition.csv` … ユニット単位の $\Delta\beta^{\rm bnd}$・$\Delta\beta^{\rm int}$・$\Delta b^{\rm raw}$・$\Delta\log\sigma$・onset step・onset $\beta$
- `provenance.json` … 入力 npz の sha256、`mlp2_phase1_0829` の commit、実装 commit

実装は `src/centered_death_posthoc.py`。**spec を単独で先に commit してから実装を commit する**（[[現在地]] 運用メモ 8/30）。

---

## 5. スコープと引用上の注意

- condA・w100・$T=10^4$・batch=1・lr=0.01・`center_alpha=0.01`・10 seed・5M。**他の幅・他の $T$・condB・他の最適化器へ外挿しない**
- `L1w100_A1` は `L2_none` と **unpaired**（`mlp2_phase1_0829/summary.md` に明記）。腕間の因果比較には **`L2_A1` vs `L2_none`** を使う。`L1w100_A1` は「隠れ1層での水準」を示す記述統計
- 本 spec の全数値は**事後**。`mlp2_phase1_0829` の事前登録判定（`A_EFFECTIVE` ほか）を上書きしない
- **`strict_dead` は当該タスク支持域に対するラベルであり、不可逆な unit-ID の死ではない**（E6 が定量化する）

### 引用禁止に足す候補（E5 が `ABSORPTION_BROKEN_BY_EMA` を出した場合）

> **centered 腕の `strict_dead` を「吸収した」と書かない。** EMA の `running_mean` が凍結ユニットの入力を動かし続けるため、[[吸収不等式の命題化_0824]] 系1 の不変性はタスク内でも成立していない。水準を引くときは E6 の連続死率を併記する。

## 6. 本 spec が答えないこと

- 境界窓での降下が **EMA の遅れ**（$1/\alpha=100$ step だけ µ が復活する）によるのか **切替ショック**（残差 $\delta$ が大きく、$b$ への自己税は片側符号）によるのか。記録格子が 1000 step なので原理的に分離できない。境界窓の記録時点では EMA は既に再収束している → [[オラクル中心化_spec_0831]]・[[境界窓細粒度記録_spec_0831]]
- 自己税と交差モーメントの分離（力の列 `F_self`・`F_rest` は本走のログに無い）
- $\sigma$ が境界で縮む機構
