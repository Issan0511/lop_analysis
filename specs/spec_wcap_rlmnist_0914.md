# タスクをまたぐ W の蓄積を止めれば時間劣化は消えるか — Random Label MNIST で隠れ層のユニット別中心化ノルムに上限を置く（8 腕 × 10 seed）

親: vault [[セッション引き継ぎ_0914_W病理とL2系]]・[[L2Initの利点を分解するShell正則化_結果_0913]]・[[RandomLabelMNIST結果_0906]]・[[幅の規制は可塑性を買うか_0910]]・[[回る速さが幅の正体か_結果_0912]] / 状態: **事前登録（実装前・本走前）** / 作成 2026-09-14 JST / チャット: `W病理とL2系_0914` / 実行: white-san（CPU のみ）

> **run id: `wcap_rlmnist_0914`**。新しい runner `src/wcap_rlmnist_0914.py` を置く。宿主 `src/pmnist_0905.py`・0906 runner `src/pmnist_rlmnist_0906.py`・0913 runner `src/shell_l2_rlmnist_0913.py` は **1 バイトも変えずに import** する（blob の sha256 を検査する）。
> 順序: **本 spec と §1 の事後集計を commit・push** → 実装＋検査（各検査に変異対照）＋煙試験＋費用測定 → 実装を commit・push → 本走 → 集計 → 結果を commit・push。
> **過去の結果（0906 CUDA・0913 GCP）との bit 一致は要求しない。** 比較する腕はすべて white-san で同じ seed・同じデータ列で再走し、過去の数値は主判定に混ぜない。

---

## 0. 一行と問い

**問い**: LoP の時間的病理は、隠れ層のユニット別中心化重みノルム ‖W̃_i‖ の成長なのか。W を十分に規制すれば、時間劣化そのものは消えるのか。
**L2・L2-Init・上限腕の絶対精度の同等性は判定しない。** 判定するのは、各腕が参照の時間劣化（fresh gap）を何割取り除いたかである。

- `W̃_i = W_i − m_i·1`（m_i = 行 i の入力方向の平均）。W1（100×784）と W2（100×100）の各行について測る。
- **上限腕は 2 本**:
  - `capT1`: task 1 は参照と同一。task 2 以降、各ユニットの ‖W̃_i‖ をその run 自身の task 1 終了時の値で頭打ちにする＝**タスクをまたぐ蓄積だけ**を止める。
  - `cap2`: step 0 から ‖W̃_i‖ ≤ 2·‖W̃_i(0)‖＝**L2 腕と同程度の小ささ**を、L2 の他の作用（bias・W3 の縮小、方向を問わない原点への引き戻し）なしで与える。

## 1. なぜ（既存データの事後集計・REPORT_ONLY・登録判定ではない）

出所: `results/wcap_rlmnist_0914/prereg_posthoc/report.md`（`analysis/wcap_rlmnist_0914/prereg_posthoc.py`・本 spec と同じ commit）。単位 pt・[ ] は seed bootstrap 95% CI。
A(t) = タスク t の `online_acc`、G = A(1) − A(31–50)、D = A(2–6) − A(31–50)、P = A(11–20) − A(41–50)。

| 腕 | A(1) | A(31–50) | G（fresh gap） | P（進行性） | ‖W1_i‖ 中央値（非中心化）t1→t50 |
|---|---|---|---|---|---|
| R none（0913） | 97.01 | 12.22 | +84.80 [+83.02, +85.75] | −0.13 | 3.43 → 4.60（task 3 前後に死亡で崩落） |
| R l2（0913） | 94.98 | 94.10 | +0.88 [+0.73, +1.06] | −0.09 | 1.24 → 1.19 |
| R l2init（0913） | 95.33 | 95.79 | −0.45 [−0.53, −0.37] | +0.02 | 1.42 → 1.33 |
| R shell（0913） | 95.83 | 94.08 | +1.75 [+1.65, +1.86] | +0.00 | 1.36 → 1.27 |
| **LR（0906）** | 96.96 | 80.64 | **+16.32 [+16.04, +16.59]** | **+1.74 [+1.46, +1.99]** | **3.50 → 60.61** |
| LR l2（0906） | 94.85 | 94.93 | −0.09 [−0.18, +0.02] | +0.04 | 1.28 → 1.21 |
| SNA none（0913） | 97.27 | 98.59 | −1.32 [−1.43, −1.22] | −0.03 | 3.16 → 17.18 |

読み（事後）:
1. **task 1 はその手法の fresh 学習そのもの**である（init・Adam 0 から、同じ 30,000 更新）。Random Label MNIST のタスクは交換可能なので、後期に fresh probe を足しても task 1 と同じ分布を引き直すだけになる。task 1 の seed 間 SD は 0.08–0.20 pt。→ **fresh probe の追加走は行わず、G を fresh gap とする。**
2. 画像 1200 枚が全タスクで固定なので、系列には正の転移が入る（SNA の全腕と R l2init で G < 0）。**G は「転移の利益 − 可塑性の損失」の正味**で、P・D と並べて読む。
3. R の l2init − l2 = +1.69 pt（窓 31–50）のうち、fresh 分は +0.36、**時間分は +1.33**。ただし正則化腕の時間劣化は task 2–8 で起きて止まり、11–50 の傾きは全腕 ±0.1 pt/10 タスク以内。
4. R の参照は task 2–3 で死亡により崩落し、タスクをまたぐ W1 の伸びは ×1.34 しかない（伸びの大半は task 1 の中）。→ ReLU では「タスクをまたぐ蓄積を止める」操作が弱く、`R_capT1` は置かない（§5 のガードが事前データの時点で落ちる）。
5. **leaky（LR）は死なずに進行性に劣化し（P = +1.74）、W1 がタスクをまたいで ×17 伸びる。mob1 は task 5 以後 0.11 前後で横ばいなのに精度は落ち続ける。** 問いの表現型そのものなので主対象にする。
6. 既存の `w_norm_l*` は**非中心化**の行ノルムの中央値で、中心化ノルムは既存 80 走のどこにも無い（重みも保存されていない）。本走で初めて記録する。

## 2. 設計

### 2.1 箱（0906/0913 の Random Label MNIST を変更せず使う）

- **データ**: MNIST 訓練 60,000 から seed ごとに 1 度だけ 1200 枚（`rl_subset` stream・層化しない）。全タスクで同じ 1200 枚。画素は [0, 1]（/255）
- **タスク**: 各タスクで 1200 枚に一様乱数ラベル ∈ {0..9}（`rl_labels` stream）。**50 タスク**
- **訓練**: **400 epoch/タスク**・**batch 16**（75 step/epoch・30,000 step/タスク）・epoch ごとに順序を引き直す（`rl_batch` stream）・cross-entropy
- **最適化**: **Adam lr=1e−3**（β=(0.9, 0.999)・ε=1e−8・0906 の手書き実装）・moment はタスクを跨いで保持・重みもリセットしない
- **ネット**: 784–100–100–10・init は宿主 `init_params`（U(±1/√fan_in)・`init` stream）
- **seed 0–9**。seed が 1200 枚・ラベル列・batch 列・init を決め、活性化にも腕にも依存しない
- **指標**: `online_acc` = タスク内 30,000 バッチの**更新前**バッチ精度の平均（0906 と同一定義）
- 実行環境: white-san（i7-14700K・Raptor Lake・AVX2）・`device=cpu`・`torch.set_num_threads(1)`・`OMP_NUM_THREADS=MKL_NUM_THREADS=1`・`torch.use_deterministic_algorithms(True)`（宿主 `setup`）

### 2.2 腕（8 腕 × seed 0–9 = 80 run）

| ブロック | act | arm | 介入 |
|---|---|---|---|
| **主** | `LR`（leaky, 負側傾き 0.1・宿主 `ARMS["LR"]`） | `ref` | なし |
| 主 | LR | `l2` | 0906/0913 の L2 をそのまま（λ=1e−3・6 tensor・g ← g + 2λp） |
| 主 | LR | `l2init` | 0906/0913 の L2-Init をそのまま（λ=1e−3・g ← g + 2λ(p − p0)） |
| 主 | LR | **`capT1`** | §2.3。r_i = その run の task 1 終了時の ‖W̃_i‖。task 2 の最初の更新から有効 |
| 主 | LR | **`cap2`** | §2.3。r_i = 2·‖W̃_i(0)‖。task 1 の最初の更新から有効 |
| 副 | `R`（ReLU） | `ref` | なし |
| 副 | R | `l2` | 同上 |
| 副 | R | **`cap2`** | 同上 |

### 2.3 上限（射影）の定義

- **対象**: W1 と W2 の各行 i（W3・全 bias・行平均 m_i・Adam の moment と step 数には触れない）
- **いつ**: 各 step で、Adam がパラメータを更新した**直後**（CE 勾配 → Adam 更新 → 射影 → 次の step の forward）。有効期間は §2.2
- **式**（float32・学習と同じ dtype、行ごと）:
  - m_i = mean_j W_ij、W̃_i = W_i − m_i、n_i = ‖W̃_i‖₂（`torch.linalg.vector_norm`）
  - **n_i > r_i の行だけ** W_i ← m_i + W̃_i · (r_i / n_i)。**n_i ≤ r_i の行は 1 bit も書き換えない**
  - 結果: ‖W̃_i‖ は r_i（float32 の丸めの範囲）、m_i と W̃_i の向きは不変
- **r_i の取り方**: capT1 は task 1 の最終更新の後に上と同じ関数で n_i を計算して固定（detach・以後不変）。cap2 は init 直後に n_i(0) を計算して 2 倍
- **上限であって等号への固定ではない**。r_i 未満のユニットは自由に縮み・伸びる
- **行平均と bias を自由に残すのは設計上の選択**: 明度チャネル m_i·Σx と b_i による沈下は止めない（大きさと沈下を分けるため）

### 2.4 全腕で同一であるべきもの

初期パラメータ・1200 枚・ラベル列・batch 列は seed だけで決まる。各 run の provenance に init・1200 枚の index・全 50 タスクのラベル・全 20,000 epoch の batch 順の sha256 を書き、集計で seed ごとに 8 腕の一致を確かめる（一致しない seed はその seed を全腕から除く）。
加えて **capT1 と ref（LR）の task 1 終了時の状態の sha256**（パラメータ＋Adam moment）を provenance に書き、一致を集計で報告する（REPORT・判定の前提にはしない。配線そのものは検査 S3 がプロセス内で保証する）。

## 3. 測る量

### 3.1 `per_task.csv`（1 行 = act × arm × seed × task）

`act, arm, lam, seed, task, online_acc, memo_acc`、0906 の `evaluate_rl` の全列（`dead_frac_*`・`zeroout_*`・`zbar_*`・`zsd_*`・`zbar_min_*`・`mob_*`・`eff_rank_*`・`w_norm_*`）、`reg_loss`（l2/l2init のみ・その腕の罰則）。
新規（l ∈ {l1, l2}＝W1, W2。float64 で float32 の重みから再計算）:
- `wt_q10_l, wt_med_l, wt_q90_l, wt_max_l`: ‖W̃_i‖ の分位点
- `wt_ratio0_med_l`: median_i ‖W̃_i‖/‖W̃_i(0)‖
- `rowmean_abs_med_l`: median_i |m_i|、`b_med_l`: median_i b_i
- `exceed_t1_frac_l`: ‖W̃_i‖ > r^{T1}_i·(1 + 1e−4) の割合（r^{T1} はその run 自身の task 1 終了時の値・全腕で計算・task 1 は 0）
- `exceed_2x0_frac_l`: ‖W̃_i‖ > 2‖W̃_i(0)‖·(1 + 1e−4) の割合（全腕）
- `bind_frac_l`: 上限腕のみ、‖W̃_i‖ ≥ r_i·(1 − 1e−4) の割合（他の腕は空）
- `proj_rows_l`: そのタスク中に書き換えた (step, 行) の数、`proj_removed_l`: その Σ(n_i − r_i)（上限腕のみ）

### 3.2 `layer_metrics.csv`

0913 の `layer_rows`（6 tensor の norm・norm0・norm_ratio・cos_w0・dist_w0・罰則 3 種）をそのまま使い、`arm` 列を足す。task 0 = init 直後、1–50 = タスク末。

### 3.3 `units.npz`（run ごと・判定には使わない・後の解析用）

形 (51, 100) の float32（t = 0..50）: `wt_l1, wt_l2, rowmean_l1, rowmean_l2, b1, b2, zbar_l1, zbar_l2, mob_l1, mob_l2`（z̄・mob は 1200 枚上のユニット別平均。t=0 も計算する）。形 (100,): `r_t1_l1, r_t1_l2`（全腕）、`r_cap_l1, r_cap_l2`（上限腕のみ・他は NaN）。

## 4. endpoint（すべて seed ごとに計算してから seed 間で要約）

- A(a–b) = タスク a..b の `online_acc` の平均
- **G = A(1) − A(31–50)**（fresh gap）
- **除去率 ρ_m = 1 − mean_s(G_m,s) / mean_s(G_ref,s)**（平均の比。ref は同じ act の `ref`）。CI は seed 対応の bootstrap（seed index を復元抽出 10,000 回・rng seed 20260914、同じ index を分子と分母に使う）の percentile。95% を主、97.5% を Bonferroni 版として併記
- 副（REPORT）: P = A(11–20) − A(41–50) と π_m = 1 − mean(P_m)/mean(P_ref)、D = A(2–6) − A(31–50)、タスク 11–50 の OLS 傾き（pt/10 タスク）、水準 A(31–50)、`memo_acc`、ρ の seed 別の値
- 表示用（REPORT）: タスク 1, 2, 3, 5, 10, 20, 30, 40, 50 の軌道、‖W̃_i‖ 中央値の t0/t1/t10/t50、上限腕の `bind_frac`、窓 31–50 の `wt_med_l1` の l2 対 cap2

## 5. 判定

### 5.1 帯の算術

- **帯 0.9 / 0.1**: LR 参照（0906）で mean P / mean G = **0.1064**（§1 の出所）。
  - 残り 10% の fresh gap は、参照自身の進行性の成分と同じ大きさ。**ρ ≥ 0.9 は、速い落ち込みに加えて進行性の成分もほぼ取り除いたときにだけ起きる。**
  - **ρ ≤ 0.1 は、取り除いた量が進行性の成分 1 本分を超えない**ことを意味する。
- **操作の強さのガード（capT1）**:
  - γ = median_s [ median_i ‖W̃1_i‖(t=50) / median_i ‖W̃1_i‖(t=1) ]（LR `ref`）。**γ ≥ 2 を要る。**
  - 根拠: 回る速さ ω ≈ lr/‖W̃_i‖ の法則（`turn_rate_0912`・`band_omega_0912`）では、ω が 1 octave 変わると損失は約 1.0 pt 変わる。伸びが 1 octave 未満なら、上限が ω 経由で戻せる損失は約 1 pt ＝ G_ref の約 6% で、帯 0.1 を下回る。そうなると `NOT_LEVER` と「操作が弱い」を区別できない。0906 の非中心化値では ×17（約 4.1 octave）。
- **操作の強さのガード（cap2）**:
  - κ = median_s [ median_i ‖W̃1_i‖ の窓 31–50 平均 / median_i 2‖W̃1_i(0)‖ ]（同じ act の `ref`）。**κ ≥ 2 を要る**（根拠は γ と同じ）。
- **検出力**: 0906 の LR で sd_s(G) = 0.47 pt、G_ref = 16.3 pt なので、ρ の SE は約 0.01。判定を分けるのは帯の意味で、標本誤差ではない。

### 5.2 ラベル

**Q1（主）LR `capT1`**:

| 条件（上から順に適用） | ラベル |
|---|---|
| 有効 seed（8 腕の stream 一致・発散なし・50 タスク完走）が 8 未満 | `INCOMPLETE` |
| γ < 2 | `WEAK_MANIPULATION` |
| ρ の 95% CI 下限 ≥ 0.9 | **`ACCUMULATION_SUFFICIENT`** |
| 上限 ≤ 0.1 | **`ACCUMULATION_NOT_LEVER`** |
| 下限 > 0.1 かつ上限 < 0.9 | **`ACCUMULATION_PARTIAL`** |
| それ以外（CI が帯の境界をまたぐ） | `UNRESOLVED` |

**Q2（副）LR `cap2` 対 `l2`**:
- Δ = ρ_cap2 − ρ_l2 = (mean G_l2 − mean G_cap2) / mean G_ref（seed 対応 bootstrap）
- 有効 seed < 8 → `INCOMPLETE`、κ < 2 → `WEAK_MANIPULATION`
- CI ⊂ [−0.1, +0.1] → **`SIZE_EXPLAINS_L2`**、上限 < −0.1 → **`L2_BEYOND_SIZE`**、下限 > +0.1 → **`CAP_BEYOND_L2`**、それ以外 → `UNRESOLVED`
- ρ_cap2 自体にも Q1 と同じ帯で `SUFFICIENT / PARTIAL / NOT_LEVER / UNRESOLVED` を付けて併記する

**Q3（副）R `cap2`**:
- ρ_Rcap2（ref は R `ref`）に同じ帯を当てる: `SIZE_RESCUES_RELU`（下限 ≥ 0.9）/ `SIZE_PARTIAL` / `SIZE_NOT_LEVER`（上限 ≤ 0.1）/ `UNRESOLVED`
- ガードは κ_R ≥ 2、有効 seed < 8 → `INCOMPLETE`
- R では G_ref ≈ 85 pt なので、帯 0.9 は「残りの fresh gap が約 8.5 pt 以下＝崩落していない」を意味し、「L2 並み」は意味しない。L2 並みかは ρ_Rcap2 − ρ_Rl2 を REPORT で見る

**REPORT（判定なし）**: ρ_l2init・ρ_l2（両 act）、LR の l2init − l2 の窓 31–50 の差の fresh 分と時間分への分解、π（P の除去率）、上限腕の水準の税（A(1) の ref との差）。

### 5.3 読み方（登録）

| 結果 | 書いてよいこと |
|---|---|
| Q1 SUFFICIENT | この箱（leaky・RL MNIST・50 タスク）では、隠れ層のユニット別中心化ノルムのタスクをまたぐ蓄積を止めるだけで、時間劣化はほぼ消える＝蓄積は十分なレバー |
| Q1 PARTIAL | 蓄積は時間劣化の ρ 割を担う。残りは task 1 終了時にすでにある状態か、自由に残した経路（行平均・bias・W3・Adam）にある |
| Q1 NOT_LEVER | 蓄積を止めても時間劣化は残る＝この箱の LoP は ‖W̃‖ の蓄積では説明できない |
| Q2 SIZE_EXPLAINS_L2 | L2 の LoP 除去は、ユニット別の大きさの規制だけで再現できる |
| Q2 L2_BEYOND_SIZE | L2 は大きさの規制以上のことをしている（bias・W3 の縮小、原点への引き戻し）。「L2 が W を保つから LoP が起きない」は不十分 |
| Q3 SIZE_RESCUES_RELU | ReLU の崩落も、大きさの規制だけで防げる（行平均と bias は自由のまま） |

限定: Random Label MNIST・784–100–100–10・Adam lr=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。**SNA・他の λ・他の箱・長い地平線には外挿しない。** G は転移を含む正味である。

## 6. 検査（実装後・本走前。各検査は変異対照が FAIL することも確かめる。許容は導出から決め、値を見て決めない）

| 検査 | 内容 | 許容の導出 | 変異対照（すべて検出されること） |
|---|---|---|---|
| S1 S-identical | 8 腕 × seed {0,1} × 3 タスク × 2 epoch で init・1200 枚・ラベル・batch 順が全腕で bit 一致し、宿主 stream から独立に再生成した値とも一致 | bit 一致 | capT1 が task 1 の後に batch 順を 1 回余分に引く／cap2 の init を 1 ulp 動かす／R の腕だけラベル stream を 1 タスク進める |
| S2 S-projection | 射影関数を、上限の上と下が混ざった乱数行列と、LR ref の 1 タスク後の実重みに当てる。上限以下の行は bit 不変、射影した行は \|‖W̃'‖/r − 1\| ≤ 1e−4、\|Δm_i\| ≤ 4·ε₃₂·max\|W_i\|、1 − cos(W̃', W̃) ≤ 1e−6、書き換え行数と Σ(n − r) は float64 の独立な数え直しと一致 | ノルムの相対誤差は d·ε₃₂（d=784 で 9.3e−5）以下なので 1e−4。平均は 3 回の丸め＋余裕で 4ε。向きの丸めは θ ≈ ε√d ≈ 3e−6 rad、1 − cos ≈ 5e−12 なので 1e−6 は十分に広い | 非中心化の射影（W·r/‖W‖）／全行を r に合わせる（上限以下も伸ばす）／係数 r/n²（n ≫ 1 の行で検査）／射影した行に 1% の接線成分を足す |
| S3 S-wiring | runner の実走から (task, step) の直前状態を捕まえ、forward・勾配・Adam・射影を検査側で独立に再現して、直後のパラメータと moment が bit 一致。capT1 は task 1 で書き換え 0・task 2 以降で > 0、cap2 は task 1 から > 0 | bit 一致・行数は整数 | 射影を Adam 更新の前に置く／W3 も射影する／capT1 を task 1 から有効にする（r は init から）／cap2 を 1 step おきにしか当てない／W2 を射影しない |
| S4 S-baseline | (LR, R) × (ref, l2, l2init) × seed 0 × 3 タスク × 3 epoch で、新 runner と無改変の 0913 runner がタスク末の重み bit 一致・共通の per_task 列一致。宿主・0906 runner・0913 runner の sha256 がそれぞれ `33a0cab`・`33a0cab`・`4fc2bba` の blob と一致 | bit 一致 | l2/l2init の係数 2→1／l2init のアンカーが生の重みを参照（clone なし）／ref の腕にも cap2 の射影が掛かる |
| S5 S-repro | LR capT1 と R cap2・seed 0・3 タスク × 3 epoch を別プロセスで 2 回。per_task.csv・layer_metrics.csv・units.npz・最終状態 sha256 が一致。**REPORT**: 同じ走を P コア（cpu 0）と E コア（cpu 27）に固定して比べる（プラットフォームの性質・判定しない） | byte/bit 一致 | batch stream を time.time_ns() で種付け |
| S6 S-log | 新規ログ列と units.npz を、debug で捕まえたタスク末の重みから float64 で独立に再計算した値と比べる（相対 1e−12・割合は完全一致） | float64 再計算どうしなので 64·ε₆₄ ≈ 1.4e−14 に余裕を持たせて 1e−12 | 非中心化ノルムを記録／ratio0 の分母を tensor 全体のノルムにする／exceed_t1 の基準を init にする |
| S7 S-verdict | 合成 shard（8 腕 × 10 seed × 50 タスク）で、SUFFICIENT・PARTIAL・NOT_LEVER・UNRESOLVED・WEAK_MANIPULATION・INCOMPLETE・stream 不一致の seed 除外、Q2 の 4 ラベル、Q3 のラベルが期待どおり | ラベルと理由コードの一致 | 窓を 41–50 にする／ρ = G_m/G_ref（1 − を落とす）／γ の分母を t0 にする／帯 0.9 を 0.8 にする |
| S-smoke | 8 腕 × seed {0,1} × 2 タスク × 3 epoch を CLI で並列に回し、集計を `--src` で煙試験ディレクトリに向ける。ファイル・行数・有限値・全ラベル INCOMPLETE | — | — |
| S-cost | 16 プロセス同時（各腕 2 本）× 1 タスク × 100 epoch で ms/step と peak RSS を実測。並列数 N = min(16, ⌊(MemAvailable − 4 GiB) / (1.2 × peak RSS)⌋)。**N < 4 なら本走を止める** | 4 GiB はデスクトップの予備（swap は既に 6/8 GiB 使用中なので当てにしない）。1.2 は RSS の走行長依存の余裕。N < 4 では 80 run が 1 日に収まらない | — |

**本走の起動条件**: `checks.json` が all_pass、コードが commit・push 済み、S-cost の N ≥ 4。
**走を殺す assert は出力を書いた後に置く。** 発散（非有限の loss・重み）した run はそのタスクで打ち切って記録し、救済しない。

## 7. 事前予測（Claude・実装前）

| 項目 | 予測 |
|---|---|
| Q1 ラベル | SUFFICIENT 35% / **PARTIAL 50%** / NOT_LEVER 10% / その他（UNRESOLVED・ガード・INCOMPLETE）5% |
| ρ_capT1 の点推定 | 0.70（80% 区間 0.40–0.95） |
| γ ≥ 2（capT1 のガード通過） | 95% |
| Q2 ラベル | **SIZE_EXPLAINS_L2 45%** / L2_BEYOND_SIZE 40% / CAP_BEYOND_L2 5% / UNRESOLVED 10% |
| ρ_cap2（LR）の点推定 | 0.85 |
| Q3 ラベル | SIZE_RESCUES_RELU 35% / SIZE_PARTIAL 35% / SIZE_NOT_LEVER 25% / その他 5% |
| LR l2init の G ≤ 0（ρ_l2init ≥ 1） | 70% |
| capT1 と ref の task 1 終了状態の sha256 が 10/10 seed で一致 | 85%（P コアと E コアの差で崩れる可能性） |

根拠:
- 箱 B の wclamp は ρ が ReLU 0.65–0.82・谷越え型 0.67–0.86・ELU 0.82–0.91。
- 一方、LR では mob の崩落（0.40 → 0.16）が task 2 に起きており、そのとき非中心化 W は 3.5 → 5.6 しか伸びていない。task 1 終了時の状態が決める部分は capT1 では消えない。
- cap2 は行平均と bias を自由に残すので、沈下の経路が残る。

**Issa の予測**: 未記入（本 spec 作成時点で依頼なし）。

## 8. 費用と実行

- 0913 の実績は 1 run 33.4 分（中央値・c2d 1 スレッド）。80 run ≈ 45 core-h。
- 並列数は S-cost の N で決める（コア数では決めない）。
- 起動は `nohup setsid bash analysis/wcap_rlmnist_0914/launch.sh`。キューは seed 順（seed 0 の 8 腕 → seed 1 …）で、途中で止まっても seed 単位で揃う。
- 完了した shard（`provenance.json` あり）は飛ばす。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- 出力: `results/wcap_rlmnist_0914/{per_task.csv, layer_metrics.csv, verdict.csv, summary.md, provenance.json, checks.json, runs/*/}`。units.npz は `git add -f` で commit する（80 run で十数 MB）。
- vault: `測定/Wの蓄積を止めればLoPは消えるか_結果_0914.md`、[[現在地]] に 1 行、本 spec の写しを `spec/実行済み/` へ。
- 結果の commit 後、git 外の生ファイルを `~/Projects/obsidian-research-data/wcap_rlmnist_0914/` に退避して manifest を commit し、main に入れ、worktree とブランチを消す。
