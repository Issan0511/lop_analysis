# Random Label MNIST で L2-Init の利点を「初期ノルムへ戻す」と「初期方向へ戻す」に分解する — Shell 正則化を R と SNA に並べる（8 腕 × 10 seed）

親: `results/pmnist_rlmnist_0906/`（vault [[RandomLabelMNIST結果_0906]]・同じ箱の 7 腕）／指示書 `~/PROMPT_shell_l2_rlmnist_0913.md`（本文＋追記。**食い違いは追記が優先**） / 状態: **事前登録（実装前・本走前）** / 作成 2026-09-13 / 実行: GCP VM `claude-shell0913`（c2d-standard-32・CPU のみ）

> **run id: `shell_l2_rlmnist_0913`**。新しい runner `src/shell_l2_rlmnist_0913.py` を置く。宿主 `src/pmnist_0905.py` と 0906 runner `src/pmnist_rlmnist_0906.py` は **1 バイトも変えずに import** する（`33a0cab` の blob と sha256 一致を検査する）。
> 順序: **本 spec を commit・push** → 実装＋検査（各検査に mutation）＋煙試験＋再現性 → 実装を commit・push → 本走 → 集計 → 結果を commit・push。
> **過去の結果（0906・CUDA）との bit 一致は要求しない。** 全 8 腕をこの VM・この `.venv`・同じ seed・同じデータ列で新しく走らせ、0906 の数値は主判定に一切混ぜない（summary に参考として別枠で書くだけ）。

---

## 0. 一行

L2-Init の罰則は **λ‖p − p0‖² = λ(‖p‖ − ‖p0‖)² + 2λ‖p‖‖p0‖(1 − cos(p, p0))** と、**半径の項**と**角度の項**に分かれる。半径の項だけを残した **Shell 正則化 R_shell = λ Σ_p (‖p‖ − ‖p0‖)²** を、既存の L2・L2-Init と同じ係数規約・同じパラメータ集合で走らせ、「L2-Init の利得は初期半径の維持で説明できるのか、初期方向へのアンカーに追加の価値があるのか」を、**活性化（R・SNA）ごとに独立に**判定する。

## 1. なぜ

| 正則化 | 何を引き戻すか | 方向の拘束 |
|---|---|---|
| `l2` | 半径も方向も 0 へ（p → 0） | なし（0 に向きは無い） |
| `l2init` | p を p0 へ（半径も方向も初期値へ） | **あり** |
| `shell` | 半径だけを ‖p0‖ へ | **なし** |

- `shell` − `l2init` は、上の恒等式の**角度の項（初期方向アンカー）**の有無だけを変える対比。
- `shell` − `l2` は、方向の拘束が無い 2 つのうち「半径を ‖p0‖ に保つ」か「0 へ縮める」かの対比。
- `l2init` − `l2` は、0906 で見えた L2-Init の利得そのもの（参考: 0906・CUDA で R は +0.0167、SNA は +0.0025。**主判定には使わない**）。

## 2. 設計

### 2.1 箱（0906 の Random Label MNIST を変更せず使う）

- **データ**: MNIST 訓練 60,000 から seed ごとに 1 度だけ 1200 枚を無作為抽出（`rl_subset` stream・層化しない）。全タスクで同じ 1200 枚
- **タスク**: 各タスクで 1200 枚に一様乱数ラベル ∈ {0..9} を付け直す（`rl_labels` stream）。**50 タスク**
- **訓練**: **400 epoch/タスク**・**batch 16**（75 step/epoch・30,000 step/タスク）・epoch ごとに順序を引き直す（`rl_batch` stream）・cross-entropy
- **最適化**: **Adam lr=1e−3**（β=(0.9, 0.999)・ε=1e−8・0906 の手書き実装）・moment はタスクを跨いで保持・重みもリセットしない
- **ネット**: 784–100–100–10・init は宿主 `init_params`（U(±1/√fan_in)・`init` stream）
- **seed 0–9**。seed は 1200 枚・ラベル列・batch 列・init を決め、活性化にも正則化にも依存しない
- **指標**: `online_acc` = タスク内 30,000 バッチの**更新前**バッチ精度の平均（0906 と同一定義）
- 実行環境: `device=cpu`・`torch.set_num_threads(1)`・`OMP_NUM_THREADS=MKL_NUM_THREADS=1`・`torch.use_deterministic_algorithms(True)`（宿主 `setup`）

### 2.2 腕（2 活性化 × 4 正則化 = 8 腕 × seed 0–9 = 80 run）

| act | reg | 引数 |
|---|---|---|
| `R`（ReLU） | `none` / `l2` / `l2init` / `shell` | `--act R --reg none｜l2:1e-3｜l2init:1e-3｜shell:1e-3` |
| `SNA`（適応 Snake・c=0.6・β=0.01、0906 と同じ） | `none` / `l2` / `l2init` / `shell` | `--act SNA --c 0.6 --beta 0.01 --reg …` |

### 2.3 正則化の定義（追記 A2・Issa 裁定。本文の λ/2 と「ndim≥2 のみ」は採らない）

- **対象パラメータ集合 P**: 既存 L2/L2-Init が正則化する **6 tensor 全部** {W1, b1, W2, b2, W3, b3}（**bias を含む**）。3 つの正則化で同一
- **λ = 1e−3**（全腕共通）。係数規約は既存どおり **R = λ Σ‖θ − θ*‖²・勾配の係数は 2λ**
- **`l2`**: 0906 の実装をそのまま使う。autograd の後で g_p ← g_p + 2λ·p
- **`l2init`**: 0906 の実装をそのまま使う。g_p ← g_p + 2λ·(p − p0)。p0 は初期化直後の detach 済み複製
- **`shell`**（新規）: tensor ごとに
  - S_p = Σ_i p_i²（float32・学習と同じ dtype）
  - ‖p‖ := √(S_p + ε²)、ε² = **1e−12**（下で導出）
  - r0_p := 同じ関数で初期化直後の p0 から 1 度だけ計算した定数（detach）
  - **T_p = λ·(‖p‖ − r0_p)²**。ただし **p0 ≡ 0（全要素が 0）の tensor は T_p = λ·S_p**（追記 A2 の規則）
  - **R_shell = Σ_p T_p**。**tensor ごとにノルムを取り、全モデルを 1 本のベクトルにまとめない**
  - 勾配は **R_shell を loss として `torch.autograd.grad(R_shell, P)`** で取る（CE の勾配と同じ更新前の値で）。0906 の l2/l2init が勾配を足すのと**同じ位置**（Adam の前）で g_p ← g_p + ∇_p R_shell とする。**hard projection ではない**。比 ‖p0‖/‖p‖ はどこでも直接計算しない
  - 解析形: ∇_p R_shell = 2λ(‖p‖ − r0_p)·p/‖p‖（**各 tensor で p と平行**）。p0 ≡ 0 の tensor では 2λ·p
- **ε² = 1e−12 の導出**（本文「必要最小限の epsilon を spec に固定」）:
  1. **要る理由**: √S の backward は S=0 で 0/0。この VM の torch 2.13 で ε²=0 だと p=0 の勾配が NaN、ε²=1e−12 だと有限（2026-09-13 19:50 に確認）
  2. **最小限・不活性**: float32 で S + ε² が S に丸め戻るのは ε² が S の half-ulp 未満のとき。**S > ε²·2²⁵ = 3.36e−5（‖p‖ > 0.0058）なら常に成り立つ**。初期値の S の最小は 6 tensor × seed 0–9 で **0.0210（b3）**＝閾値の 624 倍。初期化時点で 60 tensor すべて √(S+ε²) と √S が bit 一致することを確認済み（検査 S7 で再確認）
  3. r0 も同じ関数で取るので、**初期時点の penalty と勾配は厳密に 0**（検査 S2）
  4. 本走中に S が 3.36e−5 以下へ落ちたかは layer_metrics の `sq_norm` の最小値で報告する（REPORT）
- **p0 ≡ 0 の規則**は、この箱では bias も U(±1/√fan_in) なので**発火しない**。コードには入れ、検査 S6 で通す

### 2.4 全腕で同一であるべきもの

初期パラメータ・1200 枚・ラベル列・batch 列は seed だけで決まる（role 別 stream）。検査 S1 に加えて、**本走の各 run が init・1200 枚の index・全 50 タスクのラベル・全 20,000 epoch の batch 順の sha256 を provenance に書き、集計で seed ごとに 8 腕の一致を確かめる**（一致しない seed は無効扱い → §4.4 の INCOMPLETE）。

## 3. 測る量

### 3.1 `per_task.csv`（1 行 = act × reg × seed × task）

`act, reg, lam, seed, task, online_acc, memo_acc`、0906 の `evaluate_rl` の全列（`dead_frac_*`・`mob_*`・`zbar_*`・`eff_rank_*`・`w_norm_*`・SNA の α 統計）、`reg_loss`（その腕自身の罰則のタスク末の値・6 tensor の和・`none` は 0）、`pen_l2_total`・`pen_l2init_total`・`pen_shell_total`（全腕で λ=1e−3 として計算した 3 種の罰則の和）。

### 3.2 `layer_metrics.csv`（1 行 = act × reg × seed × task × tensor。task 0 = 初期化直後、1–50 = タスク末）

`tensor`（W1/b1/W2/b2/W3/b3）・`ndim`・`numel`・`sq_norm`（S）・`norm`・`norm0`・**`norm_ratio` = ‖p‖/‖p0‖**・**`cos_w0` = ⟨p, p0⟩/(‖p‖‖p0‖)**・`dist_w0` = ‖p − p0‖・`pen_l2` = λ‖p‖²・`pen_l2init` = λ‖p − p0‖²・`pen_shell` = λ(‖p‖ − ‖p0‖)²（p0 ≡ 0 なら λ‖p‖²）・`pen_angular` = 2λ‖p‖‖p0‖(1 − cos)・**`reg_loss`** = その腕自身の罰則（`none` は 0）。

いずれも float32 の重みから **float64 で計算し直した記録用の値**（ε を含まない厳密ノルム）で、学習には使わない。‖p0‖ = 0 なら `norm_ratio`・`cos_w0` は NaN（この箱では起きない）。

### 3.3 run ごとの `provenance.json`

git hash・argv・hostname・torch/numpy の版・スレッド数・MNIST gz の sha256・1200 枚の index の sha256・**init / ラベル列 / batch 列の sha256**・**最終状態の sha256**（重み・Adam の m, v, t・SNA の V）・壁時計・発散の有無。

## 4. 登録判定（**R と SNA で独立に**出す）

### 4.1 窓と統計

- **窓 = タスク 31–50**。seed ごとに `online_acc` の 20 タスク平均 x(reg, seed) を取る
- 4 腕の成績: seed 間の平均 ± SD・中央値・最小–最大
- **主要対比**（seed 対応差）: **Δ1 = shell − l2**・**Δ2 = shell − l2init**・**Δ3 = l2init − l2**
- 各対比の統計: 平均・**SE** = SD(ddof=1)/√n・**seed bootstrap 95% CI**（percentile・B=10,000・`numpy.random.default_rng([20260913, act 番号, 対比番号])`・seed を復元抽出）・**符号数**（勝ち/n・差が厳密に 0 の seed は除く）・**両側 sign test**（正確二項）
- **同等性帯 δ = 0.005**（accuracy・Δ2 について・本文で事前登録済み）
- n=10 の両側 sign test で p<0.05 になるのは **9/10 以上**（9/10 → 0.0215、8/10 → 0.109）

### 4.2 対比の分類（活性化ごとに α = 0.05）

- **「X > Y」**: ci95_lo > 0 **かつ** p_sign < 0.05。**「X < Y」**: ci95_hi < 0 **かつ** p_sign < 0.05
- **Δ2（shell − l2init）は同等性帯で 4 つに分ける**:
  - `EQUIV`: −δ ≤ ci95_lo **かつ** ci95_hi ≤ +δ（CI が帯の中）
  - `SHELL_ABOVE`: ci95_lo > +δ **かつ** p_sign < 0.05（CI が帯の上に抜ける）
  - `SHELL_BELOW`: ci95_hi < −δ **かつ** p_sign < 0.05（CI が帯の下に抜ける）
  - `AMBIG`: それ以外（CI が帯の端を跨ぐ、または sign test が有意でない）
- **B・C にも帯を使う理由**: 0906 の対応差の SE は 0.0001–0.0006 と小さく、帯の内側の差（例 +0.002）でも「有意」になる。帯を「実用上の差の最小単位」として A・B・C のすべてに一貫して使い、**帯の端を跨ぐ CI は D（曖昧）**とする

### 4.3 norm-match ガード（機構解釈の前提）

- **weight tensor（ndim ≥ 2: W1・W2・W3）だけで判定する**（本文の字面。bias は 6 tensor 中 3 本だが要素数は 210/89,610 = 0.23%）。**bias は REPORT のみ**（同じ比を bias でも計算して併記する）
- M(reg, W) = seed 0–9 × タスク 31–50 のタスク末の `norm_ratio`（200 値）の**中央値**
- **PASS ⟺ 3 本の W すべてで 0.9 ≤ M(shell, W) / M(l2init, W) ≤ 1.1**（端を含む）

### 4.4 結論ラベル（上から順に当て、最初に当たったものを採る）

| 順 | 条件 | ラベル（理由コード） |
|---|---|---|
| 0 | その活性化の 40 run のどれかが欠損・発散・窓内に非有限、または §2.4 の in-run 同一性が崩れた seed がある | **D**（`INCOMPLETE`）。使える seed での統計は REPORT |
| 1 | norm-match ガード FAIL | **D**（`NORM_MATCH_FAIL`） |
| 2 | Δ2 = `SHELL_BELOW` | **B** |
| 3 | Δ2 = `SHELL_ABOVE` | **C** |
| 4 | Δ2 = `AMBIG` | **D**（`CI_AMBIGUOUS`） |
| 5 | Δ2 = `EQUIV` **かつ** Δ1 が「shell > l2」**かつ** Δ3 の ci95_lo ≥ **2δ = 0.010** **かつ** Δ3 の p_sign < 0.05 | **A** |
| 6 | それ以外（Δ2 は `EQUIV` だが 5 の残りを満たさない） | **D**（`EQUIV_GAIN_UNRESOLVED`） |

- **A**「初期方向への回帰は主要因ではなく、初期半径の維持でL2-Initの利得の大部分を説明できる」
- **B**「初期方向へのアンカーに、半径制御を超える追加価値がある」
- **C**「初期方向へのアンカーは新タスク適応を妨げている」
- **D**「方向効果は未同定。性能比較のみ確定」

**この表で決めた解釈 2 点（結果を見る前に固定）**:
1. **ガードは A も縛る。** 本文の A の行には「norm-match PASS」が書かれていないが、ガードの段落（「条件を外れた場合は性能比較のみ報告し、方向効果の因果結論を出さない」）と D の条件（「norm-match FAIL」）に従う。A の文も方向効果についての結論だから
2. **A には「L2-Init の利得 Δ3 が帯の 2 倍以上」を課す。** 「利得の**大部分**」を「半分超」と読む。Δ2 が `EQUIV` なら l2init − shell ≤ δ なので、shell が取り戻す割合は (shell − l2)/Δ3 ≥ 1 − δ/Δ3 で、これが 1/2 以上になるのは Δ3 ≥ 2δ のとき。Δ3 がそれより小さいと、帯が広すぎて「大部分」を言えない（→ D）。点推定の割合 f = Δ1/Δ3 と bootstrap CI は REPORT

### 4.5 2 つの活性化の扱いと多重比較

- **R と SNA を並列の主判定とし、どちらも主にしない**（それぞれが A–D のラベルを 1 つ持つ）。問いの答えが活性化で違いうること（ReLU は死んだユニットの救命が効き、SNA は正則化が容量税になった・0906）が、この実験の見どころの一部だから
- **活性化ごとに 1 族・α = 0.05**。族の中では、B・C は 1 本の 95% CI の両裾で互いに排他、A は条件の積（intersection–union）なので、ラベルを付けること自体では誤りの率は膨らまない
- **活性化を跨ぐ補正はラベル自体にはかけない。** その代わり、**「R でも SNA でも〜」という活性化を跨ぐ文は、2 つのラベルが同じで、しかも両方が Bonferroni（2 族・α = 0.025: 97.5% bootstrap CI と p_sign < 0.025 を全条件で使う）でも同じラベルになるときだけ**書いてよい。Bonferroni 版のラベルは同じ規則で計算し `verdict.csv` の `label_bonf` に書く（n=10 では p<0.025 でも 9/10 で届く）
- `none` 腕を含む対比と診断量はすべて REPORT_ONLY でラベルを持たない

### 4.6 REPORT_ONLY

`none` 腕と `none` との対比・`memo_acc`・4 腕の tensor ごとの `norm_ratio`／`cos_w0`／罰則の窓内中央値・bias で計算したガード比・L2-Init 罰則の半径成分（`pen_shell`）と角度成分（`pen_angular`）への分解・取り戻し割合 f・`sq_norm` の最小値（ε の不活性）・0906（CUDA）の同名 6 腕の窓平均（**主判定に使わない参考**）・壁時計。

## 5. 検査（**すべて PASS し、すべての mutation が FAIL を出す**ことが本走の条件）

共通: この VM・この `.venv`・CPU 1 スレッド。**mutation** は、新しい runner（または集計スクリプト）のソースを文字列置換し（置換元がちょうど 1 回現れることを assert）、別モジュールとして読み込んで（CLI 検査はスクリプトとして走らせて）同じ検査を当て、**FAIL が出ることを確かめる**。`checks.json` には各検査の値・PASS・閾値の導出と、各 mutation の内容・その検査結果・検出の有無を書く。**失敗した検査の閾値を緩めて通すことはしない**（閾値は下の導出で先に固定する）。

float の許容幅は「正しい実装が取りうる丸め誤差の上限」から導く。記号: ε_d = dtype d の machine epsilon、N = tensor の要素数、k = 勾配の各要素に掛かる丸め演算の数。

| 検査 | 何を確かめるか | 閾値とその導出 | mutation（これで FAIL すべき） |
|---|---|---|---|
| **S1 `S-identical`**（必須 1） | 8 腕 × seed {0,1} × 3 タスク × 2 epoch で、run 内で捕まえた init・1200 枚の index・各タスクのラベル・各 epoch の batch 順の sha256 が **8 腕すべてで一致**し、かつ `H.stream`/`H.init_params`/`RL.subset_idx`/`RL.task_labels` から独立に作り直した値とも一致 | bit 一致（主張が bit 一致だから） | M1a shell 腕だけ batch stream を 1 回余分に引く／M1b l2 腕の init の 1 要素を 1 ulp ずらす／M1c SNA 腕だけラベル stream を 1 タスク分進める／M1d l2init 腕だけ 1200 枚を seed+1 から引く |
| **S2 `S-shell-zero-init`**（必須 2） | seed 0–9 × {R, SNA} の shell 腕で、タスク 1 の step 0（更新前）に runner 内で捕まえた R_shell とその勾配 6 本が**厳密に 0** | 厳密に 0（r0 は同じ関数・同じ bit から計算されるので差は丸め無しで 0） | M2 r0/p0 を seed+1 の init から取る |
| **S3 `S-shell-grad-parallel`**（必須 3＋A2） | (a) float64 の試験状態（seed 0–2 の init を tensor ごとに半径比 1.25 と 0.8 を交互にかけ、直交成分を足して回したもの）で、`shell_penalty` の autograd 勾配が **6 tensor すべてで p と平行**（p_i ≠ 0 の成分で g_i/p_i のばらつき ≤ tol_par、p_i = 0 では g_i = 0）かつ**係数が tensor ごとの解析値 2λ(‖p‖ − r0)/‖p‖ と一致**（math.fsum による float64 の独立計算と比べ ≤ tol_coef）。(b) 実際の shell run（R・SNA・seed 0・2 タスク × 2 epoch）の最終 step の生の正則化勾配（float32）が 6 tensor すべてで平行、かつ捕まえた重みから `shell_penalty` で計算し直した勾配と bit 一致 | **tol_par = k·ε_d(1 + k·ε_d) + 4ε_64、k = 1**: 勾配の各要素は「上流のスカラー × p_i」の丸め 1 回（p*p の backward の g·p_i が 2 回同値で出て和は厳密に 2 倍）。スカラー側の誤差は全要素に共通なので平行性を崩さない。比は float64 で取る。**tol_coef = ((N+1)/2 + 1)·ε_64·((‖p‖ + r0)/\|‖p‖ − r0\| + 1) + 8ε_64**: 和 S の相対誤差 ≤ N·ε/2、差 ‖p‖ − r0 の打ち消しで (‖p‖+r0)/\|‖p‖−r0\| 倍に増える | M3a shell の勾配を l2init のもの 2λ(p − p0) に差し替え（(a)(b) の平行性で FAIL）／M3b ノルムを全 tensor を連結した 1 本で取る（(a) の係数で FAIL）／M3c 罰則を λ/2 規約にする（(a) の係数で FAIL） |
| **S4 `S-l2-unchanged`**（必須 4） | (R, SNA) × (none, l2:1e-3, l2init:1e-3) × seed 0 × 3 タスク × 3 epoch で、新 runner と**無改変の 0906 `RL.run_one`** の (i) 各タスク末の重みの sha256 が一致、(ii) 0906 が書く per_task の全指標列が完全一致。加えて `src/pmnist_0905.py`・`src/pmnist_rlmnist_0906.py` の sha256 が `33a0cab` の blob と一致 | bit 一致 | M4a 新 runner の l2 係数 2.0 → 1.0／M4b l2init の p0 を clone せず重みそのものを参照（差が常に 0） |
| **S5 `S-repro`**（必須 5） | 同じ VM・同じ `.venv` で shell 腕（R・SNA）seed 0・2 タスク × 5 epoch を**別プロセスで 2 回** → `per_task.csv`・`layer_metrics.csv` が byte 一致、最終状態 sha256 が一致 | byte/bit 一致 | M5 batch stream の seed を `time.time_ns()` から取る |
| **S6 `S-zero0-rule`**（A2） | p0 ≡ 0 の合成 tensor で T = λ·Σp²（値）と勾配 2λ·p（S3 と同じ平行性・係数の基準）、p = 0 でも値・勾配が有限。p0 ≠ 0 の tensor で p = 0 のときも勾配が有限 | 値: 相対 ≤ (N+2)·ε_64（和の丸め）。有限性は厳密 | M6a p0 ≡ 0 の項を捨てる（T=0）／M6b ε² = 0（p=0 で NaN） |
| **S7 `S-eps-inert`** | seed 0–9 の init の 6 tensor すべてで float32 の √(S + ε²) が √S と bit 一致 | bit 一致（§2.3 の導出） | M7 ε² = 1e−4 |
| **S8 `S-shell-wiring`** | shell 腕（R・seed 0・1 タスク × 1 epoch = 75 step）を、宿主の forward・CE・`shell_penalty`・手書き Adam で**独立に再生**し、75 step 後の重みが runner と bit 一致。かつ none 腕の重みとは異なる（正則化が実際に効いている） | bit 一致／不一致 | M8a runner が shell の勾配を足さない／M8b 符号を逆にして引く |
| **S9 `S-log`** | 4 正則化（R・seed 0・2 タスク × 2 epoch）で、runner が書く `layer_metrics` と per_task の罰則列が、各タスク末（と init）に捕まえた重みから numpy と math.fsum で独立に計算した値と一致 | 量ごとに float64 の 2 経路の丸め上限: ノルム・比・距離・L2 系罰則は相対 ≤ c·N·ε_64 + 8ε_64（c = 和の本数）、cos は 3N·ε_64 + 8ε_64、`pen_shell` は打ち消しを含むので絶対 ≤ λ(2\|‖p‖−‖p0‖\|·e + e²)、e = (N·ε_64 + 4ε_64)(‖p‖+‖p0‖) | M9a `norm_ratio` を逆数にする／M9b `pen_l2init` を λ/2 にする／M9c `cos_w0` を隣の tensor の p0 と取る |
| **S10 `S-verdict-synthetic`** | 集計スクリプトに、答えを決めて作った合成 shard を与える: A・B・C・D(`NORM_MATCH_FAIL`)・D(`CI_AMBIGUOUS`)・D(`EQUIV_GAIN_UNRESOLVED`)・D(`INCOMPLETE`) → ラベルと理由コードがすべて一致 | 一致 | M10a 窓を 30–49 にずらす／M10b Δ2 の向きを l2init − shell にする／M10c ガードを bias で判定する／M10d 帯を 0.05 にする |

**煙試験 `S-smoke`**: CLI で 8 腕 × seed {0,1} × 2 タスク × 3 epoch を並列に走らせ、集計スクリプトを **`--src results/_smoke_shell_l2_rlmnist_0913/runs` と明示して**当てる → 6 つの出力ファイルが揃う・per_task 32 行・layer_metrics 288 行・`online_acc` とノルム列に NaN なし・shell の task 0 の `reg_loss` = 0・窓が無いので両活性化とも D(`INCOMPLETE`)。**本走ディレクトリには集計を当てない。**

**費用 `S-cost`**（計測・合否は計画のゲートのみ）: 新 runner を 16 本同時（8 腕 × 2）・各 1 タスク × 100 epoch で走らせ、腕ごとの ms/step から 80 run × 16 並列の完了時刻を見積もる。**見積もりの完了が 2026-09-14 13:00 JST（VM 削除の 6 時間前）より後なら、本走を始めずに報告して止まる。**

実装前の計測（0906 runner・この VM・2026-09-13 19:40）: SNA+l2init 1 本だけで **1.15 ms/step**、同時本数ごとの 1 本あたりの時間は 8 本 ×1.00・**16 本 ×1.02**・24 本 ×1.37・32 本 ×2.09（1 本 790 MB）。**→ 並列は 16 本**（物理 16 コア。24 本は処理量が 11% 増えるだけでばらつきが大きい）。

## 6. 実行計画

- **16 本同時・各 1 スレッド**。キューは seed 0 の 8 腕 → seed 1 の 8 腕 → … の順（途中で止めても腕が揃った seed が残る）。shard 名 `{act}_{reg}_s{seed}` は**明示的に列挙**し、ログ名も平坦にする
- 投入直後に `date` と `pgrep -af shell_l2_rlmnist_0913.py` で本数を数える。止めるときは PID を見て `kill <pid>`（`pkill -f` は使わない）
- **途中退避**: 30 分ごとに、完走した shard（`provenance.json` があるもの）だけをパス明示で commit・push する。**走っている間は集計を当てず、窓の値も読まない**
- 見積もり: 1 run ≈ 1.5M step × 1.2 ms ≈ 30 分 → 80 run / 16 並列 ≈ **2.5 時間**
- **VM は 2026-09-14 19:11 JST 頃にディスクごと削除される。** 本走が 2026-09-14 15:00 JST を過ぎても終わらなければ、止めて完走分を push し、判定は出さずに報告する

## 7. 出力

- `results/shell_l2_rlmnist_0913/`: **`per_task.csv`・`layer_metrics.csv`・`verdict.csv`・`summary.md`・`provenance.json`・`checks.json`**、`runs/<shard>/`（run ごとの生出力）、`logs/`
- `results/_checks_shell_l2_rlmnist_0913/`（検査の小さな run）・`results/_smoke_shell_l2_rlmnist_0913/`（煙試験）
- コード: `src/shell_l2_rlmnist_0913.py`・`analysis/shell_l2_rlmnist_0913/{checks.py, verdict.py, launch.sh, autopush.sh}`
- `summary.md` には必ず: 実際に使った commit hash と環境、活性化ごとの 4 腕の窓成績、対応差、norm-match ガードの成否、A–D のどれか 1 つの限定付き結論（理由コードつき）

## 8. 予測（Claude・2026-09-13・実装前）

**起草側の予測であり独立の予言ではない。0906 の結果（CUDA）を見たうえで立てている。**

| | R | SNA |
|---|---|---|
| `none` | 崩落（窓 < 0.2）95% | 4 腕で最高 85% |
| Δ3 = l2init − l2 の ci95_lo ≥ 0.010（利得が帯で解像できる） | 80% | 10%（0906 は +0.0025） |
| ガード PASS | 70%（外れるなら W3） | 75% |
| ラベル | **B 45%**／A 30%／D 20%／C 5% | **D 55%**（主に `EQUIV_GAIN_UNRESOLVED`）／C 25%／B 15%／A 5% |

- R: ReLU の可塑性喪失はユニットの死（重みの向きと bias の漂流）を含み、半径だけの拘束では向きの漂流を戻せない、と読む。ただし 0906 の `R+l2` の ‖w‖ 比 0.92 対 `R+l2init` 1.00 のように、L2 の不利が「半径を縮める」ことにあるなら shell は l2init に届く（A）
- SNA: 死ぬユニットが無いので救命は要らず、正則化は容量税になる（0906: 素の SNA が最高）。半径だけの拘束は税が小さく shell ≥ l2init（60%）。ただし Δ2 が帯 ±0.005 を抜けるほどの差になるかは五分以下

**Issa**: （記入する場合は本走の集計より前に）

## 9. 引用制限

- 結論 A–D は**この箱の中だけ**の話: Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・50 タスク・CPU。λ を振っていない
- Shell は **tensor ごとの半径**の拘束。ユニット（行）ごとのノルムや、hard projection（normalize-and-project 型）について書かない
- **ガード FAIL のときは方向効果について何も書かない**（性能の比較だけ）
- 活性化を跨ぐ一般化は §4.5 の条件を満たすときだけ
- この VM（CPU）の数値と 0906（CUDA）の数値を同じ走として並べない。一致は統計的な水準でしか期待しない

## 10. Log（登録後の変更は日時と「結果を見る前／後」を付けてここに追記する）

- 2026-09-13 19:59 JST: 初版を登録（実装前・本走前）
- 2026-09-13 20:21 JST（実装中・本走前・本走の結果は存在しない）: §5 の mutation を実装で具体化した。**閾値・判定規則は変えていない。**
  - **M3a** は runner のループで「shell の勾配の代わりに 2λ(p − p0) を足す」置換にした。関数単体を見る (a) には現れないので、**(b)（実走の生勾配の平行性と、関数の勾配との bit 一致）で検出される**。表の「(a)(b) の平行性で FAIL」を「(b) で FAIL」に読み替える
  - (a) に当たる非平行の壊し方として **M3a′（ノルムを tensor 全体でなく行ごとに取る）を追加**した
  - **M9c** は「b2 を隣の b1 の p0 に対して記録する」（形が同じ隣の tensor は b1 だけ）
  - S-cost の 1 run の見積もりは「16 本同時に測った 1 タスク × 100 epoch の壁時計 × 4 × 50 ＋ 起動 5 秒」（1 タスク分の評価も 4 倍に数える保守側）。キューは 16 枠・launch と同じ順で模擬する
- 2026-09-13 20:24 JST（本走前）: 検査一式の 1 回目（20:21–20:24）で **S-smoke だけ FAIL**。理由は検査側の数え間違いで、shell の task 0 の行数を「12」と期待していたが、正しくは 2 活性化 × 2 seed × 6 tensor = **24**。その回の記録でも 24 行すべてが厳密に 0（`results/_checks_shell_l2_rlmnist_0913/checks_run1_FAILED_smoke_rowcount_miscount.json` に保存）。期待行数を 24 に直し、「全行が厳密に 0」の条件は変えずに一式を最初から走らせ直す。S1–S10 と S-cost はこの回もすべて PASS・mutation 全検出
- 2026-09-13 20:30 JST（本走中・結果は誰も見ていない）: 上の行の時刻は、実装 commit `4fc2bba` では時計を見ずに「20:30」と書いていた。`checks.json` の started_at/finished_at（1 回目 20:21:27–20:23:58、2 回目 20:24:47–20:27:17 JST）に合わせて「20:24」に直した。検査 2 回目は all_pass（mutation 24/24 検出・S-cost の見積もり 2.78 h）。**本走は 2026-09-13 20:28:27 JST に `4fc2bba` から 16 並列で投入**
