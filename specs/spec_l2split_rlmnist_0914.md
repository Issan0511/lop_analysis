# L2 を「中心化した W̃ の減衰」と「それ以外の減衰」に割る — Random Label MNIST の 2×2（R・LR × ref/l2/l2wt/l2rest × 10 seed）

親: vault [[Wの蓄積を止めれば時間劣化は消えるか_結果_0914]]（`wcap_rlmnist_0914`・同じ箱）・[[セッション引き継ぎ_0914_W病理とL2系]] / 状態: **事前登録（実装前・本走前）** / 作成 2026-09-14 JST / チャット: `W病理とL2系_0914` / 実行: white-san（CPU のみ）

> **run id: `l2split_rlmnist_0914`**。新しい runner `src/l2split_rlmnist_0914.py` を置く。宿主 `src/pmnist_0905.py`・0906 runner・0913 runner・**wcap runner `src/wcap_rlmnist_0914.py`** は 1 バイトも変えずに import する（blob の sha256 を検査）。
> 順序: **本 spec を commit・push** → 実装＋検査（各検査に変異対照）＋煙試験＋費用測定 → 実装を commit・push → 本走 → 集計 → 結果を commit・push。
> Issa の選択（2026-09-14 09:30）: 構成は 2×2、ref と l2 は bit 一致確認つきで再利用、spec 登録から全自動。

---

## 0. 一行と問い

`wcap_rlmnist_0914` の**硬い上限は副作用が大きかった**。
- fresh の税 −6 pt。
- R で task 1 のうちに第 1 層の 57% が死亡。
- 規模が W3 へ逃げた。

そのため Q3（R cap2 = `SIZE_NOT_LEVER`）は、「大きさが効かない」のか「射影が壊した」のかを区別できない。

本 spec は **L2 の罰則を同じ λ のまま厳密に 2 つに割り、soft な減衰で**次を問う。

- **l2wt**: 隠れ層の中心化した W̃ だけを減衰させる（行平均・bias・W3 は自由）と、L2 と同じように時間劣化は消えるか。
- **l2rest**: 残りの部分（隠れ層の行平均・全 bias・W3・b3）だけを減衰させると消えるか。

L2 と L2-Init、上限腕との絶対精度の同等性は判定しない。判定するのは、参照の時間劣化（fresh gap）の除去率である。

## 1. 分解

隠れ層の行 i について W_i = W̃_i + m_i·1（m_i = mean_j W_ij）で、W̃_i ⊥ 1 なので ‖W_i‖² = ‖W̃_i‖² + d_l·m_i²（d_1 = 784、d_2 = 100）。
したがって 0906 の L2（λ=1e−3・6 tensor）は正確に 2 つに分かれる。

λ Σ_θ ‖θ‖² = **λ Σ_{l=1,2} Σ_i ‖W̃_i‖²**（l2wt） + **λ [ Σ_{l=1,2} Σ_i d_l·m_i² + ‖b1‖² + ‖b2‖² + ‖W3‖² + ‖b3‖² ]**（l2rest）

勾配（Adam の前で CE の勾配に足す。0906 の l2 と同じ位置・結合型）:

| 腕 | W1・W2 に足す | b1・b2・W3・b3 に足す |
|---|---|---|
| `l2`（既存） | 2λ·W | 2λ·p |
| **`l2wt`** | **2λ·(W − m·1)**（行ごとに平均を引いた W） | なし |
| **`l2rest`** | **2λ·m·1**（行平均を行全体に） | **2λ·p** |

- l2wt と l2rest の追加勾配の和は、l2 の追加勾配 2λθ と一致する（float32 の丸めの範囲）。
- {ref, l2wt, l2rest, l2} は「W̃ の減衰 on/off × 残りの減衰 on/off」の 2×2 になる。
- Adam は非線形なので効果は足し算にならない。そこを 2×2 で見る。

## 2. 設計

### 2.1 箱（`wcap_rlmnist_0914` と同一）

Random Label MNIST（seed ごとに固定した 1200 枚・タスクごとに一様乱数ラベル・50 タスク）・784–100–100–10・宿主 init・Adam lr 1e−3（β=(0.9, 0.999)・ε=1e−8・moment はタスクを跨いで保持）・batch 16・400 epoch/task（30,000 step）・`online_acc` = 更新前バッチ精度のタスク平均・seed 0–9・**λ = 1e−3**・white-san・CPU・1 スレッド・`torch.use_deterministic_algorithms(True)`。

### 2.2 腕（2 活性化 × 4 = 8 腕 × seed 0–9）

| act | arm | 用意の仕方 |
|---|---|---|
| **R**（ReLU・主） | `ref` / `l2` | **`wcap_rlmnist_0914` の committed run（`results/wcap_rlmnist_0914/runs/`）を再利用**（§2.3 の条件つき） |
| R | **`l2wt`** / **`l2rest`** | 新規 20 run |
| **LR**（leaky 0.1・副） | `ref` / `l2` | 再利用 |
| LR | **`l2wt`** / **`l2rest`** | 新規 20 run |

### 2.3 再利用の条件（登録）

1. **検査 S4**: 新 runner の `ref`・`l2` が、無改変の wcap runner と (R, LR) × seed 0 × 3 タスク × 3 epoch で、タスク末の重みと共通の per_task 列が bit 一致する。
2. **全長の照合**: 本走の最初に、新 runner で `R_ref_s0`・`R_l2_s0`・`LR_ref_s0`・`LR_l2_s0` を 50 タスク × 400 epoch 通して再走する。provenance の `final_state_sha256`・`task1_end_state_sha256` と、per_task.csv の `online_acc` 列が wcap の committed shard と完全一致すること。
3. 1 と 2 がすべて一致したとき `REUSE_OK` とし、ref・l2 は wcap の 10 seed を使う。**1 つでも不一致なら `REUSE_MISMATCH` とし、ref・l2 の残り 36 run を新 runner で再走してから**、それを使う（起動スクリプトが自動で行う）。

### 2.4 全腕で同一であるべきもの

init・1200 枚・ラベル列・batch 列の sha256 を provenance に書き、集計で seed ごとに 8 腕（再利用分を含む）の一致を確かめる。一致しない seed は全腕から除く。

## 3. 測る量

- `per_task.csv`: wcap runner と同じ列。`online_acc`・`memo_acc`・`evaluate_rl` の全列・‖W̃_i‖ の分位点・行平均・bias・exceed 系を、wcap の `unit_arrays`・`task_fields` をそのまま呼んで書く。上限列（`bind_frac`・`proj_*`）は空。
- 新規列:
  - `pen_wt_total` = λΣ‖W̃_i‖²、`pen_rest_total` = λ[Σ d·m² + ‖b1‖² + ‖b2‖² + ‖W3‖² + ‖b3‖²]。全腕で λ=1e−3 として float64 で計算する。
  - `reg_loss` = その腕自身の罰則（ref 0・l2 は 0913 と同じ・l2wt = pen_wt・l2rest = pen_rest）。
- `layer_metrics.csv`（0913 の layer_rows・`arm` 列つき）と `units.npz`（wcap と同じ配列）。

## 4. endpoint（wcap と同じ定義・seed ごとに計算して seed 間で要約）

- A(a–b) = タスク a..b の `online_acc` の平均。**G = A(1) − A(31–50)**。
- **ρ_m = 1 − mean_s G_m,s / mean_s G_ref,s**（同じ act の ref）。seed index を復元抽出 10,000 回（rng 20260914・分子と分母に同じ index）の percentile CI。95% を主、97.5% を Bonferroni 版として併記する。
- REPORT: P = A(11–20) − A(41–50)、D = A(2–6) − A(31–50)、傾き 11–50、水準 A(31–50)、`memo_acc`、ρ_l2、軌道、状態量（‖W̃‖・mob・z̄・dead・|m|・b・‖W3‖）、wcap の cap2・capT1 との並記。

## 5. 判定

### 5.1 帯・ガードの算術

- **帯 0.9 / 0.1**（wcap と同じ）:
  - LR: 参照の P/G ≈ 0.11（wcap の LR ref で +1.89/+16.53 = 0.114）なので、残り 10% は参照自身の進行性の成分 1 本分。
  - R: G_ref ≈ 84.5 pt なので、0.9 は残りの fresh gap ≲ 8.5 pt ＝崩落していないことを意味する。
- **操作の強さのガード**（1 octave の ω で損失 ≈ 1 pt ≈ LR の G_ref の 6% ＜ 帯 0.1、の算術。wcap と同じ根拠）:
  - l2wt: κ_wt = median_s [ 窓 31–50 の median‖W̃1_i‖（ref） / 同（l2wt） ] ≥ 2。
  - l2rest: κ_rest = median_s [ max( 窓 31–50 の ‖W3‖（ref）/‖W3‖（l2rest）、窓 31–50 の ‖b1‖（ref）/‖b1‖（l2rest） ) ] ≥ 2。
  - 落ちたら `WEAK_MANIPULATION`。
- **副作用ガード**（wcap の教訓）:
  - (i) mean_s A(1)_arm ≥ mean_s A(1)_l2 − 1.0 pt。1.0 は、soft な 3 正則化の A(1) の開きの最大から取る（0913 R: l2 94.98 / l2init 95.33 / shell 95.83 → 0.85 pt、wcap LR: l2 94.83 / l2init 95.05 → 0.22 pt）。
  - (ii) R のみ: median_s dead1(t=1)_arm ≤ median_s dead1(t=1)_l2 + 0.1。0.1 は 100 ユニットの割合の二項 SD（p≈0.24 で 0.043）の 2.4 倍。
  - 落ちたら、帯のラベルの後ろに `+HARMFUL` を付ける。付いた腕の NOT_LEVER・PARTIAL を「大きさ（または残り）はレバーでない」と読むことを禁じる。

### 5.2 ラベル（act ごと・arm ∈ {l2wt, l2rest} ごと・上から適用）

| 条件 | ラベル |
|---|---|
| 有効 seed（8 腕の stream 一致・発散なし・50 タスク完走・§2.3 の再利用が成立または再走済み）< 8 | `INCOMPLETE` |
| κ < 2 | `WEAK_MANIPULATION` |
| ρ の 95% CI 下限 ≥ 0.9 | `SUFFICIENT` |
| 上限 ≤ 0.1 | `NOT_LEVER` |
| 下限 > 0.1 かつ上限 < 0.9 | `PARTIAL` |
| それ以外 | `UNRESOLVED` |

副作用ガードに落ちた腕は `+HARMFUL` を付ける（例 `NOT_LEVER+HARMFUL`）。

**2×2 の型（act ごと・登録）**: l2wt と l2rest のラベル（`+HARMFUL` を除いた本体）から決める。

| l2wt | l2rest | 型 | 書いてよいこと |
|---|---|---|---|
| SUFFICIENT | NOT_LEVER | **`WT_SUFFICES`** | L2 の LoP 除去は、中心化した W̃ の soft な減衰で足りる。R なら、wcap Q3 の失敗は硬い射影の副作用だった |
| NOT_LEVER | SUFFICIENT | **`REST_SUFFICES`** | L2 の LoP 除去は、行平均・bias・W3 の減衰から来る。W̃ の大きさからではない |
| SUFFICIENT | SUFFICIENT | **`EITHER_SUFFICES`** | どちらでも足りる（冗長） |
| NOT_LEVER | NOT_LEVER | **`BOTH_NEEDED`** | 片方ずつでは効かず、両方そろって効く（ρ_l2 ≥ 0.9 のとき） |
| それ以外（PARTIAL・UNRESOLVED・WEAK・INCOMPLETE を含む） | | **`MIXED`** | 型は付けず、ρ を並べて読む |

**主の問いは R の型**（wcap Q3 の読みを決める）。LR の型は副。活性化をまたぐ文は、両方の型が一致したときだけ書く。

### 5.3 限定

Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。λ を変えれば分解の比重は変わりうる。SNA・他の箱・長い地平線へ外挿しない。G は画像固定による正の転移を含む正味である。

## 6. 検査（実装後・本走前。各検査は変異対照が FAIL することも確かめる）

| 検査 | 内容 | 許容の導出 | 変異対照 |
|---|---|---|---|
| S1 S-identical | 6 腕（ref・l2・l2wt・l2rest × act）× seed {0,1} × 3 タスク × 2 epoch で init・画像・ラベル・batch 順が bit 一致し、宿主 stream の独立な再生成とも一致 | bit 一致 | l2wt が batch 順を 1 回余分に引く／l2rest の init を 1 ulp 動かす |
| S2 S-grad | 乱数の 6 tensor と LR ref の 1 タスク後の実重みで、runner の追加勾配関数を確かめる。(i) l2wt の W 部分の行和 \|Σ_j g_ij\| ≤ d·τ_i、(ii) l2rest の W 部分が行内で完全に同じ値、(iii) 要素ごとに \|g^wt + g^rest − 2λθ\| ≤ τ_i（bias・W3・b3 は完全一致）、(iv) l2wt が bias・W3・b3 に何も足さない（完全一致で 0）、(v) float64 で独立に計算した 2λ(W − m)・2λm と要素ごとに ≤ τ_i | τ_i = 4·ε₃₂·2λ·max_j\|W_ij\|。平均を引く・足す・係数を掛ける丸めは要素ごとに高々 3ε₃₂·max\|W_i\| 程度（余裕を見て 4）。行和はそれを d 要素分足した上限。(ii)・(iv) と bias 部分は同じ float を配るだけなので完全一致 | l2wt で行平均を引き忘れる（2λW）／平均を列方向（dim=0）で取る／l2rest で W3 を落とす／係数 2λ → λ |
| S3 S-wiring | runner の実走から (task, step) の直前状態を捕まえ、forward・CE・autograd・追加勾配（検査側で独立に実装）・Adam を再現し、直後の params と moment が bit 一致。R l2wt と LR l2rest で各 2 step | bit 一致 | 追加勾配を Adam の後にパラメータへ直接足す／l2wt を W3 にも掛ける／l2rest が bias を落とす／l2wt と l2rest の分岐を入れ替える |
| S4 S-baseline | (R, LR) × (ref, l2) × seed 0 × 3 タスク × 3 epoch で、新 runner と無改変の wcap runner が各タスク末の重みと per_task の共通列で bit 一致。宿主・0906・0913・wcap runner の sha256 が `33a0cab`・`33a0cab`・`4fc2bba`・`0b8ef53` の blob と一致 | bit 一致 | l2 の係数 2→1／ref に l2wt の減衰が掛かる |
| S5 S-repro | R l2wt と LR l2rest・seed 0・3 タスク × 3 epoch を別プロセスで 2 回。CSV・units.npz の配列・最終状態 sha256 が一致 | byte/bit 一致 | batch stream を time.time_ns() で種付け |
| S6 S-log | `pen_wt_total`・`pen_rest_total` を、捕まえたタスク末の重みから float64 で独立に再計算した値と比べる（相対 1e−12）。あわせて pen_wt ＋ pen_rest ＝ pen_l2_total（相対 1e−12・全腕） | float64 どうし（64ε₆₄ ≈ 1.4e−14） | pen_wt を非中心化で計算／pen_rest から b3 を落とす |
| S7 S-verdict | 合成 shard（新規 4 腕 × 10 seed ＋ 再利用 4 腕 × 10 seed）で、全ラベル、5 つの型、`+HARMFUL`、WEAK、INCOMPLETE、stream 不一致の seed 除外、REUSE_MISMATCH 時に再走分を使うこと、が期待どおり | ラベル・型・推定値の一致（1e−9） | 窓 41–50／ρ の 1 − を落とす／副作用ガードの不等号を逆にする／再利用の判定を task 1 の sha だけで行う |
| S-smoke | 全 8 腕 × seed {0,1} × 2 タスク × 3 epoch を CLI で 8 本ずつ回し、再利用元にも煙試験の ref・l2 を置いて、集計を `--src`・`--reuse-src` で煙試験ディレクトリに向ける。ファイル・行数・有限値・全ラベル INCOMPLETE | — | — |
| S-cost | 8 本同時（新規 4 腕 × 2 act）× 1 タスク × 100 epoch で ms/step と peak RSS を実測する。並列数 N = min(16, ⌊(MemAvailable − 4 GiB) / (1.2 × peak RSS)⌋)。**N < 4 なら本走を待つ**（止めずに、空くまで 60 秒ごとに見る）。見込みは、wcap の実測（R 系 1 run 60–96 分・LR の L2 系約 56 分）から、R 系と L2 系の新規腕に 3 倍を掛ける | 4 GiB はデスクトップの予備。3 倍は wcap で task 1 の測定を実測が上回った比（R ref 21 → 67 分・LR l2 35 → 56 分） | — |

**本走の起動条件**: `checks.json` が all_pass、コードが commit・push 済み、空きメモリから N ≥ 4。
**各 run の起動直前にも** MemAvailable ≥ 4 GiB + 1.2 × peak RSS を確かめ、足りなければ 30 秒ずつ待つ（別セッションの走と共存するため）。

## 7. 事前予測（Claude・実装前）

| 項目 | 予測 |
|---|---|
| R の型 | WT_SUFFICES 20% / **REST_SUFFICES 40%** / EITHER_SUFFICES 10% / BOTH_NEEDED 10% / MIXED 20% |
| R l2wt | SUFFICIENT 30% / PARTIAL 25% / NOT_LEVER 45% |
| R l2rest | SUFFICIENT 50% / PARTIAL 25% / NOT_LEVER 25% |
| LR の型 | **WT_SUFFICES 35%** / REST_SUFFICES 5% / EITHER_SUFFICES 15% / BOTH_NEEDED 5% / MIXED 40% |
| LR l2wt | SUFFICIENT 55% / PARTIAL 40% / NOT_LEVER 5% |
| LR l2rest | SUFFICIENT 15% / PARTIAL 50% / NOT_LEVER 35% |
| l2wt の副作用ガード通過（R・LR とも） | 85% |
| REUSE_OK | 90% |

根拠:
- R の崩落は bias と行平均による沈下が運ぶ（ref の z̄1 −9.3、l2 の z̄1 −0.19）。
- LR の劣化は W̃ の ×17 の伸びが主（wcap Q1 で蓄積停止が 79% を消した）。
- ただし l2wt は W3 と bias を自由に残すので、wcap で見えた W3 への逃げが soft でも起きうる。

**Issa の予測**: 未記入（登録時点で無し）。

## 8. 費用と実行

- 新規 40 run ＋ 照合 4 run。wcap の実測で見積もると、R 系は 1 run 60–96 分、LR の L2 系は約 56 分で、約 45 core-h。
- 並列数は §6 の N（別セッションの走が終わるのを待つことがある）。
- 起動は `nohup setsid bash analysis/l2split_rlmnist_0914/launch.sh`。
  - 照合 4 run を先頭に置き、残りは seed 順に並べる。
  - 全 run の後に `reuse.py` で照合し、不一致なら ref・l2 の残り 36 run を追加で回す。
- 完了した shard は飛ばす。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- 出力: `results/l2split_rlmnist_0914/{per_task.csv, layer_metrics.csv, verdict.csv, summary.md, provenance.json, checks.json, reuse.json, runs/*/, logs/}`。units.npz は `git add -f`。
- vault: `測定/Lの分解で救命の出どころを分ける_結果_0914.md`、[[現在地]] に 1 行、本 spec の写しを `spec/実行済み/` へ。
- 結果の commit 後、git 外ファイルを退避して manifest を commit し、main に入れ、worktree とブランチを消す。
