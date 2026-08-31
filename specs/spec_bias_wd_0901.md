# bias_wd_0901: condA・centered × bias 専用 weight decay

状態: **事前登録・未実行** / 作成: 2026-09-01 / run id: `bias_wd_0901`

親: `HANDOFF「bias 専用 weight decay（B → A → 本走）」` §5
前段: 段階 B = `docs/lit_bias_wd_0901.md`（commit `96599e4`）、
段階 A = `results/bias_wd_pilot_0901/`（commit `274d888`）
参照: `mlp2_phase1_0829` の `L1w100_A1` / `L2_Aall`、`centered_freeze_0901`

---

## 1. 問い

centered regime では bias $b$ が唯一残ったオフセット自由度で、これが下にも上にも
暴走している。$b$ **だけ**に weight decay を掛けて復元力を与えると、死・飽和・
機能劣化が同時に止まり、しかも frozen 腕の静的コスト（`unfit` 約 50 倍）を
払わずに済むか。

### 1.1 なぜ $b$ だけか（壁の閉形式）

condA・タスク内・32 パターン厳密サポートで、ユニット $i$ の消灯は厳密に

$$\hat p_i = 0 \iff \beta_i + \kappa_i \le 0,\qquad
\beta_i := \frac{\bar z_i}{\sigma_i} = \frac{w_i\cdot\mu + b_i}{\sigma_i},\qquad
\kappa_i := \frac{\max_p z_{p,i} - \bar z_i}{\sigma_i}$$

第1層では自由座標が 0/1 の 5 ビットなので
$\kappa_i = \lVert w_{i,\rm free}\rVert_1 / \lVert w_{i,\rm free}\rVert_2 \in [1,\sqrt5]$。
EMA 中心化（`center_alpha=0.01`）を掛けると task 末では
$w\cdot\mu \approx 0$ になるので、$\beta$ を動かせるのは実質 $b$ だけである
（実測: `mlp2_phase1_0829` の committed logs で
median $\beta$ と median $b/\sigma$ が第1層・第2層とも 3 桁一致する）。

centered ではもう一つ良い性質がある。台帳（flip 15 座標と $b$ が 1 自由度に
縮退する構造）の逃げ道が塞がっている。$w_{\rm flip}$ が勾配のほぼ零方向なので、
$b$ を減衰させても実効バイアスが flip 側へ記帳を移せない。**std 腕にはこの
逃げ道があるので、本走は centered 限定である。**

### 1.2 既に測ってある両端

| 腕 | dead（B10） | `unfit`（B10） | `mean(log10 unfit)` B10−B02 |
|---|---|---|---|
| `L1w100_A1`（$\lambda=0$） | 0.46394 | 0.0039701 | +0.1863 |
| `centered_freeze_0901`（$\lambda=\infty$ 相当） | 0.000000 | 0.228112 | — |

dead と `unfit` は `results/centered_freeze_0901/verdict.csv`。劣化 +0.1863 は
`results/mlp2_phase1_0829/logs/L1w100_A1_seed{0..9}.npz` から §5.5 の集計規約で
本 spec が計算したもの。**HANDOFF §2.4 は同じ欄に +0.1329 と書いているが、
そこでは「事後計算」とだけあって算出手順が示されていない。本 spec は上の
+0.1863 を採り、判定 (c) は「`W1_none` 自身の同一手順による劣化」との対応差で
行うので、この不一致は判定に影響しない**（§5.2 (c)）。

frozen の dead=0 は代数的に強制されており、証拠として引いてはいけない（§7-1）。

---

## 2. 設計

- condA: `m=20`, `f=15`, teacher width 100, $\beta=0.7$, $T=10{,}000$, batch 1
- learner: ReLU、plain SGD、`lr=0.01`、Kaiming uniform 初期化
- centering: `center_alpha=0.01` の層入力 EMA 中心化。教師は生入力を見る
- seed 0–9、5,000,000 step（= task 500）、CPU、`OMP_NUM_THREADS=1`
- checkpoint は step 0 / 5M
- 走の宿主は `mlp2_phase1_0829` の腕実行経路（`setup_arm_p1` / `train_arm_p1` /
  `forward_centered` / `exact_layer_record_p1`）をそのまま使う。`wd_b` は
  `VecMLPL.set_weight_decay_b` で構築後に差し込む（乱数も状態も消費しない）

### 2.1 更新式

$$b \leftarrow b - \eta\,(g_b + \lambda b)$$

**decoupled ではなく素の L2 勾配**である。掛けるのは**隠れ層 bias $b$ のみ**で、
$W$・$v$・出力バイアス $c$ には掛けない。2 層腕では両層の $b$ に掛ける。
`freeze_bias=true` と `wd_b>0` の同時指定はエラー。

分岐を置かず常に `gb + wd_b*b` を計算する。$\lambda=0$ の腕が WD コード経路を
通したうえで無 WD 実装と bit 一致することを検査可能にするためである（§6 S1）。

### 2.2 腕

$\lambda$ グリッドは段階 A（`results/bias_wd_pilot_0901/grid_selection.json`）が
凍結済み規則で決めた値をそのまま写す。**本 spec では選び直さない。**

| 腕名 | hidden | centered_layers | $\lambda$ | 役割 |
|---|---|---|---|---|
| `W1_none` | [100] | [1] | 0 | 対照（既存 `L1w100_A1` と bit 一致） |
| `W1_main` | [100] | [1] | **1e-3** | **主判定** |
| `W1_sub1` | [100] | [1] | 1e-4 | 用量反応 REPORT_ONLY |
| `W1_sub2` | [100] | [1] | 1e-2 | 用量反応 REPORT_ONLY |
| `W1_sub3` | [100] | [1] | 1e-1 | 用量反応 REPORT_ONLY・S5 |
| `W2_Aall_none` | [100,100] | [1,2] | 0 | 対照（既存 `L2_Aall` と bit 一致） |
| `W2_Aall_main` | [100,100] | [1,2] | **1e-3** | **副判定 W2** |
| `W2_Aall_sub` | [100,100] | [1,2] | 1e-1 | 用量反応 REPORT_ONLY |

- **`W2_Aall_sub` に副 3 水準のうち最も強い 1e-1 を割り当てる理由**: W2 の病理は
  上方暴走で、その大きさ（alive 中央 $b/\sigma$ が B02 +0.17 → B10 +12.85）は
  W1 の下方暴走（−0.75 → −0.91）より 1 桁以上大きい。復元力の平衡深さは
  $|b^\star|\approx|g_b|/\lambda$ なので、情報が出るのは強い側である。
  併せて深さ2 での S5 恒真ガードを兼ねる
- 見積もり: 1 層 22 分 × 5 腕 + 2 層 38 分 × 3 腕 ≈ 3.7 時間（逐次）。
  腕ごとに独立プロセスで並列実行してよい（`OMP_NUM_THREADS=1`・軌道は不変）

---

## 3. 記録

task 末（10,000 step ごと、step 0 を含む 501 点）に 32 パターン厳密列挙を行い、
層ごとに次を記録する。非有限ガードは 1,000 step ごと（§6 S4）。

| 量 | 定義 |
|---|---|
| `strict_dead_frac` | $\hat p = 0$ のユニット割合 |
| `b_median_alive` | alive ユニットの生の $b$ の中央値 |
| `B_median_alive` | alive の $b/\sigma$ の中央値（HANDOFF §2.5 の「$b$ 項」） |
| `beta_median_alive` | alive の $\beta=(w\cdot\mu+b)/\sigma$ の中央値 |
| `kappa_median_alive` | alive の $\kappa$ の中央値 |
| `wall_frac` | alive の $\lvert\beta\rvert/\kappa$ の中央値 |
| `margin_median_alive` | alive の $\kappa\sigma - \lvert b\rvert$ の中央値（W3） |
| `p_hat_median_alive`, `p_hat_thin_frac`, `p_hat_sat_frac` | alive の $\hat p$ 中央値、$\hat p\le 8/32$ 率、$\hat p\ge 30/32$ 率 |
| `eff_rank`, `eff_rank_W`, `w_norm_median`, `wcos_mean` | 凍結済み `exact_layer_record_p1` の値をそのまま |
| `unfit` | 32 点厳密サポートでの `residual_var / signal_var` |

---

## 4. 窓とブロック

- ブロックは **50 task 刻み**。ブロック $k$ は task $50(k-1)+1 \ldots 50k$
- **B02 = task 51–100**（$k=2$）、**B10 = task 451–500**（$k=10$）
- 実験単位は seed。ブロック内 50 個の task 末を seed 内で平均して一値にする

---

## 5. 事前登録判定

### 5.1 主判定（`W1_main`, $\lambda$=1e-3, B10）

3 条件の連言。**判定に使う $\lambda$ は主 1 水準のみ**（多重比較を避ける）。

| 条件 | 内容 | しきい |
|---|---|---|
| (a) 死の抑制 | `strict_dead_frac` の seed 平均 | $\le 0.232$（= `W1_none` の 0.46394 の半分） |
| (b) 静的水準の保持 | `mean(log10 unfit)` の対応差 `W1_main − W1_none` の **95% CI 上端** | $< +0.10$（同値マージン $\Delta$、= 1.26 倍） |
| (c) 劣化の抑制 | `mean(log10 unfit)` の B10−B02 の対応差 `(main の劣化) − (none の劣化)` の **95% CI 上端** | $< 0$ |

### 5.2 判定名（決定木・この順で評価する）

HANDOFF §5.2 の 5 つの名前は重なりがあるので、次の決定木で一意にする。

1. **(a) が偽** → `NO_EFFECT`
2. **(a) 真・(b) 偽** → `PAYS_STATIC_COST`。さらに (c) も偽なら
   `dead_only_flag = 1` を立てる（HANDOFF の `DEAD_ONLY` に対応する状態）
3. **(a) 真・(b) 真・(c) 偽** → `LEVEL_ONLY_NO_KINETICS`
4. **3 条件すべて真** → `BIAS_WD_PROTECTS`

(a) は絶対しきい 0.232 で判定し、`W1_none` に対する比も evidence 欄に併記する。
10 seed のうち何本が 0.232 を下回ったかを Clopper–Pearson 95% CI つきで出す。

### 5.3 副 3 水準

`W1_sub1/2/3` は **REPORT_ONLY**。用量反応（dead・`mean(log10 unfit)`・劣化）の
記述だけを出し、主判定には使わない。

### 5.4 副判定

| ID | 内容 | 判定名 |
|---|---|---|
| **W2** | `W2_Aall_main` の**第1層 活性 `eff_rank`** の B10 が、(i) `W2_Aall_none` の B10 より有意に高い（対応差 95% CI 下端 > 0）かつ (ii) seed 中央値が `W2_Aall_none` の **B02 水準の 70% 以上** | `SATURATION_PREVENTED` / `NOT_PREVENTED` |
| **W3** | `W1_main` の保護マージン $\kappa\sigma-\lvert b\rvert$（第1層 alive 中央値）の B02→B10 変化 | CI 下端>0 → `MARGIN_WIDENS` / CI 上端<0 → `MARGIN_NARROWS` / それ以外 `FLAT`。`W1_none` の同一対比も併記 |
| **W4** | `W2_Aall_*` 3 腕の alive $\hat p\ge30/32$ 率 | REPORT_ONLY |

W2 の比較基準は**再走した `W2_Aall_none`** である（S0 により既存 `L2_Aall` と
bit 一致するはずで、その committed logs から計算すると B02 eff_rank = 10.9413、
B10 = 3.1370、したがってしきいは $0.70\times10.9413 = 7.659$ になる見込み）。
`W2_Aall_sub` は REPORT_ONLY で、W2 判定には使わない。

### 5.5 集計規約

- 主判定に使うのは **`mean(log10 unfit)`**: seed ごとに、ブロック内 50 個の
  task 末の $\log_{10}$ を平均する。`log10(mean unfit)` も出力するが**判定には
  使わない**（混同しない）
- **床は系ごとに違う**。深さ1系 `1e-16`（`dose_const_5m_0830` の S6 較正を継承。
  同じ 1 層・幅100・condA 系で `two_summation_orders` により較正済み）、
  深さ2系 `1e-23`（`mlp2_phase1_0829` の S6 較正を継承）。**本走では再較正しない。**
  各腕の床にかかった点の割合を `block_levels.csv` の `floor_frac` に持つ
- CI は seed 水準の **paired percentile bootstrap**、`B = 20000`、
  `bootstrap_seed = 20260902`（`20260830` は `generator_offset`、`20260901` は
  `centered_freeze_0901` の bootstrap seed で既用のため流用しない）
- **studentized はこの repo では Phase 0b 以降ほぼ全行で退化する**ので、
  percentile を主とする。studentized も併算し、退化検出
  （`degenerate_se_tol=1e-15`、`degenerate_frac_max=0.01`、
  `degenerate_width_ratio_max=100`）の結果を `ci_degenerate` 列に出す
- 二値割合の CI は Clopper–Pearson

---

## 6. 実行前・実行中サニティ

| ID | 内容 | 失敗時 |
|---|---|---|
| **S0** | `W1_none` を既存 `L1w100_A1` に、`W2_Aall_none` を既存 `L2_Aall` に対して 30k step・1k 格子で replay し、`unfit`・`eval_loss_exact`・各層の `strict_dead_frac` が一致 | 本走禁止 |
| **S1** | $\lambda=0$ が **WD コード経路を通しても無 WD 実装と bit 一致**（$b$ にしか WD が掛かっていないことの確認）。同一状態から 1 step 進めて全パラメータの bitwise 一致を見る | 本走禁止 |
| **S2** | 同一状態から $\lambda=0$ と $\lambda>0$ で 1 step 進め、(i) $W$・$v$・$c$ が bitwise 一致、(ii) $b$ の差が厳密に $-\eta\lambda b_{\rm before}$、(iii) `nets.py` の `sgd_step_layers` 内で `wd_b` を参照する更新行が `self.bs[i]` の 1 行だけ | 本走禁止 |
| **S3** | 32 点サポートの壁恒等式（$\hat p=0 \iff \beta+\kappa\le0$）、$\hat p$ の 1/32 量子化、第1層 $\kappa$ の閉形式一致、凍結済み `exact_layer_record_p1` との独立実装一致（許容 `1e-10`）、全主指標の有限性 | 本走禁止 |
| **S4** | **数値発散ガード**。`probe_every=1000` で非有限を検出したら当該腕を停止し `arm_status/<arm>.json` に記録して他腕は続行する（前例: `L2_A2` が step 141,000・seed 7 で発散） | 当該腕のみ停止 |
| **S5** | **恒真ガード**。最大 $\lambda$（`W1_sub3`, 1e-1）で alive 中央 $b$ が 0 に漸近し frozen 腕（dead 0・`unfit` 0.228）に接近することを確認する。**この収束は予測の確認であって証拠ではない**旨を `summary.md` に明記する | 記録のみ |

S0/S1/S2 は本走の前にゲートとして実行し、`results/_gate_bias_wd_0901/` に
JSON で保存する。保存済みゲートが PASS でなければ本走は起動しない。

---

## 7. 引いてはいけない線

1. **高 $\lambda$ 端で dead が 0 になることを証拠にしない。** $b\equiv0$ かつ
   centered なら $\beta = w\cdot\mu_{\rm res}/\sigma$ で、実測の
   $\lVert\mu_{\rm res}\rVert \le 0.167$ に対し壁は
   $\kappa \ge 1$ なので task 末に消灯できない。これは観測ではなく恒等式の帰結
2. **「WD が LoP を治す」と一般に書かない。** スコープは condA・centered・
   幅100・$T=10^4$・batch=1・lr=0.01・plain SGD・5M step に限る
3. **std 腕へ外挿しない。** std には台帳の逃げ道があり、同じ介入が別の意味になる
4. **新規性を「WD が効く」に置かない。** 段階 B（`docs/lit_bias_wd_0901.md`）の
   通り、先行（Dohare et al.）の L2 腕は SGD/Adam/AdamW/PPO の 4 経路すべてが
   `.parameters()` 全体に単一 $\lambda$ を掛けており、bias を除外した L2 も
   bias だけの L2 も存在しない。加えて先行の全パラメータ L2 は
   **dead を 13%→23% に上げ effective rank を 28→11 に下げている**（Fig 4b の
   目視読み）。したがって主張は「**$b$ だけの減衰で足りるか**」に限定し、
   (a) が通った場合は「L2 一般の再現」ではなく「全パラメータ L2 との乖離」
   として報告する
5. **`strict_dead` の低下を機能改善と読み替えない。** `centered_freeze_0901` で
   dead 0.464→0 と `unfit` 0.0040→0.2281 が逆向きに動いた前例がある。
   (a) と (b)(c) は独立に報告する
6. **パイロット（`results/bias_wd_pilot_0901/`）の数値を結果として引用しない。**
   $\lambda$ グリッドの由来としてのみ参照する
7. **$\lambda$ グリッドは段階 A で決まっている。結果を見て選び直さない。**
   段階 A では事前登録の絶対目標が対照腕の天井を超えていて到達不能だったため、
   目標値を $\lambda=0$ に対する相対値で読み替えた。その逸脱の事実・理由・
   絶対で読んだ場合の選択は `results/bias_wd_pilot_0901/summary.md` に記録済み

---

## 8. 成果物

```
specs/spec_bias_wd_0901.md      ← 単独で先に commit（本ファイル）
configs/bias_wd_0901.yaml
src/bias_wd_0901.py
results/bias_wd_0901/
    verdict.csv  summary.md  paired_endpoints.csv  task_end_metrics.csv
    block_levels.csv  run_sanity.json  provenance.json  fig_bias_wd.png
results/_gate_bias_wd_0901/     ← S0 / S1 / S2 のゲート記録
```

commit は **spec 単独 → config+実装 → 結果** の 3 段。各段で
`git ls-remote origin refs/heads/main` により push を確認する。
