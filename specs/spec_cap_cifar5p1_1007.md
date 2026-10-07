# cap_cifar5p1_1007 — 5+1 CIFAR × MLP（R・std・lr 1e−4）で W1・W2 の行ノルムを task 1 終端値に止める（4 腕 × seed 0–9）

状態: **事前登録**（この文書の commit が登録。cap 腕の値はまだ 1 つも無い。ref は宿主 `cifar5p1_mlp_0920` の `R_std_lr0.0001` の再走で、その記録は読んである — §1.3 に開示）
作成: 2026-10-07 / 起草: Claude（親 Fable の依頼）/ run: `cap_cifar5p1_1007` / branch: `claude/cap_cifar5p1_1007` / worktree: `wt/cap_cifar5p1_1007`（origin/main `91ec9f2` から）
目的: V12 表 5「証拠表（柱 × 箱）」の柱「W 増大 → 進行」× 箱「5+1 CIFAR × MLP（実ラベルの箱）」のセル（現在「未」）を、背骨で登録済みの S5 [`cap_cifar_ee_0920`](spec_cap_cifar_ee_0920.md)（両層の行ノルム上限）と RL-MNIST の [`l2cap_ee_0917`](spec_l2cap_ee_0917.md)（`cap_row_norm_`）と同じ形の介入で事前登録して走らせ、「登録」に格上げする。
箱: [`spec_cifar5p1_mlp_0920.md`](spec_cifar5p1_mlp_0920.md) §2（箱）・§3（指標・fresh 対照）。エンジンの元: `src/cifar5p1_mlp_0920.py` の `run()`（無改変）。
実装（これから作る）: `src/cap_cifar5p1_1007.py`・`analysis/cap_cifar5p1_1007/{checks.py,verdict.py,launch.sh}`・出力 `results/cap_cifar5p1_1007/`。既存の `src/*.py` は 1 文字も変えない。

## 0. 一行と問い

**実ラベルの 5+1 CIFAR × MLP（ReLU・std・Adam lr 1e−4）で、各 Adam 更新の直後に W1・W2 の各行のノルムを task 1 終端値以下に射影すると、後期の水準（E1）が上がり、新品の網との差（E2 = fresh gap）が縮むか。**

- 主比較は **cap12 − ref**（両層の上限）。cap1（W1 だけ）・cap2（W2 だけ）は副比較で、層別の寄与と交互作用を記述する。
- 乱数ラベルの箱（S5・l2cap）と違い、この箱は床に落ちない（R の後期窓 .424）。救済は **水準と喪失の seed 内対応差** で定義し、S5 の「床 0/10」の型は使わない。

## 1. 出所・既知のこと（登録の前に見たもの）

### 1.1 箱の既知値（宿主 `results/cifar5p1_mlp_0920/R_std_lr0.0001/`、seed 0–9）

- 後期窓（hard 21–29 = t 21,23,25,27,29 の online 平均）: seed 中央値 **.4241**（平均 .4366・SD .0284）。早期窓（hard 1–9）中央値 .615。
- fresh gap（t29）: 中央値 **+.228**、10/10 正（.166〜.258）。
- t29 の seed 中央値: 死 l2 **.38**・mob l2 **.002**・eff_rank l2 **9.1**・z̄ l2 **−8.82**。t1 では死 l2 .005・z̄ l2 +1.66・zsd l2 2.79。
- 行ノルムの seed 中央値（`w_norm_l*` は各行のノルムの中央値）: W1 **.712（t1）→ 1.068（t29）**、W2 **.598 → .705**、W3 → 3.15。初期値は 3 層とも U(±1/√fan_in) なので行ノルムの期待値は √(1/3) = .577。**W2 は task 1 終端で初期値から 3.5% しか動いていない**（W1 は 23%）。
- 同じ箱の既知の介入: L2 Init（λ 1e−2）で gap +.228 → −.010・後期窓 .658（§13.5）、扉 H（ユニット平均の差し引き）で z̄ l2 −8.82 → +0.64・死 .38 → .000・mob .002 → .61・eff_rank l2 9.1 → 13.5・後期窓 .55（§16.6(d)）。**L2 Init は θ₀ への引き戻しで、ノルムだけを止める本介入と同じ介入ではない。** 扉 H も別物。

### 1.2 型にした介入（S5・l2cap）

- S5（RL-CIFAR・ELU/std・50 課題）: cap12 RESCUED（主窓 online .101 → .860）、cap1・cap2 は SPLIT。l2cap（RL-MNIST・ELU→ELU）: cap12 で崩壊を防いだ。どちらも ELU で、ELU の float32 の訓練の微分は z < −16.64 で厳密に 0（**絶対的な床**）。
- 本箱は ReLU。ReLU は正の斉次で、W1・W2 を一様に縮めても z₂ が一様に縮むだけで、どのユニットが死んでいるかは変わらない（死は z₂ の **相対位置** で決まる）。一方 Adam の 1 歩の大きさは重みの尺度に依らないので、ノルムを止めると 1 歩あたりの相対的な動き（実効的な学習率）は保たれる。どちらが勝つかは分からない。これが本走の問い。
- 先行: `wcap_rlmnist_0914` では RL-MNIST の ReLU に対する「2×初期値の硬い上限」は崩落を防げなかった（task 1 で死 57%。上限が初期値基準で、本 spec の task 1 終端基準とは違う）。leaky では task 1 終端基準の上限で fresh gap の 79% が消えた。上限腕では W3 が育つ（規模の逃げ）ことも報告されている → §6 で W3 を記述する。

### 1.3 登録の前に見たもの（開示）

- 宿主の ref の記録（per_task.csv・fresh_control.csv）を読み、行ノルム・z̄ l2・死・gap の軌跡を確かめた（§1.1）。**本走の ref は同じ網の再走（§7 S-nochange で一致を示す）なので、ref だけで決まる量（§5 (B)、§6 (i)(ii)）は登録の前に答えが分かっている。** そのため §8 の私の予測ではこれらを採点しない（親の予測は書かれたとおり採点する）。
- cap 腕は seed 0–9 では 1 歩も走らせていない。検査（§7）の cap 腕は検査用 seed（100–109）と短縮走だけで回す。

### 1.4 環境の差（登録の前に見つけたこと・S-nochange の設計に効く）

- 宿主の無改変の `run()` を今の環境（`.venv` の torch **2.13.0+cu130**・`--threads 2`・CUDA graph）で `R/std/lr 1e−4/seed 0–9/30 課題` 回すと、**fresh_control.csv はバイト一致**、per_task.csv は **eff_rank_l2 の 111/300 行だけ**が違い（最大相対 7.8e−8）、他の 26 列は全行バイト一致した。
- 宿主の provenance の torch は **2.13.0+cu126**。宿主の S-nochange（0920）はスレッド 2 で committed 列とバイト一致を確かめているので、スレッド数ではなく **ライブラリのビルドの差**。今の環境でスレッド 1・8 にすると eff_rank_l2 の違う行は 239・242 行（＋eff_rank_l1 が 1 行）に増え、どのスレッド数でも committed は再現しない（2 が最も近い）。違う 111 行はすべて第 2 層に死んだユニットがある行（活性の行列 A に厳密に 0 の列 → Gram 行列に厳密な 0 固有値 → 計算された値は丸めそのもの）で、その丸めが eff_rank の 1e−8 の桁に出る。
- したがって「ref が既存列と bit 一致」は §7 S-nochange の 2 段で示す: **(a) 今の環境の宿主の無改変コードとファイルごとバイト一致**（写したエンジンが宿主と同じ計算であることの証明）、**(b) committed の列とは eff_rank 以外の全列がバイト一致、eff_rank_l1・l2 は算術から導いた帯（§7.1）の中**。スレッド数は宿主の CLI 既定と同じ 2 に固定する。

## 2. 設計

### 2.1 箱（宿主と同一・変えない）

CIFAR-100（sha256 `85cd44d0…77a7` を確認してから走る）、hard（5 クラス × 500 枚）と easy（1 クラス 500 枚）が交代する 30 課題（task 1 が hard）、780 更新/課題・batch 32、3072–100–100–100・ReLU（宿主の腕 `R`）、入力 `std`、Adam lr 1e−4・β (0.9, 0.999)・ε 1e−8・WD なし（moment と更新番号は課題をまたいで保持）、seed 0–9、R = 10 を 1 プロセスに束ねる（slot = seed 昇順）、CUDA graph、fresh 対照あり。乱数の役割（`c51_classes`・`c51_batch`・`init`）も宿主のまま。

### 2.2 エンジン

`src/cap_cifar5p1_1007.py` に宿主の `run()` を写し、R/std に要らない分岐（iv・扉 C・nan_slot）だけを落として、上限を足す。データ・課題列・バッチ・初期化・評価（`evaluate`・`accuracy`）・活性化・束ね forward・CSV 書き出しは宿主（`cifar5p1_mlp_0920`・`rlcifar_mlp_battle_0918`・`pmnist_0905`）から import する。

- 射影は `step()` の中、宿主の Adam 更新（6 テンソル）と `act.update`・`post_update`（R ではどちらも何もしない）の後に置く。CUDA graph はプロセスにつき 1 回 capture し、宿主の warmup → 状態の巻き戻し（`keep`）を写す。巻き戻す対象に上限の計数器を足す。
- 半径は graph の中から参照される静的テンソル `rad[l]`（R × 行 × 1）。**task 1 の間は +∞**（`n̂ > +∞` は偽なので `torch.where` が W 自身を選び、1 bit も書かない）。task 1 の最後の更新の後、task 2 の最初の更新の前に、その時点の行ノルムを `rad[l].copy_()` で書き込む（graph の参照先はそのまま）。fresh 対照の前に `+∞` に戻す。
- ref 腕の `step()` には射影の演算が 1 つも入らない（宿主と同じ kernel 列）。cap 腕の task 1 は射影が書かないので ref の task 1 と bit 一致するはず（§7 S-radius で確かめる）。

### 2.3 腕（4 腕 × seed 0–9 = 40 run、R = 10 で 4 プロセス、GPU で直列）

| 腕 | W1 の行ノルム上限 | W2 の行ノルム上限 | 格 |
|---|---|---|---|
| ref | なし | なし | 共通参照（宿主の R/std と同じ網） |
| cap1 | task 1 終端値 | なし | 副比較（95%） |
| cap2 | なし | task 1 終端値 | 副比較（95%） |
| **cap12** | task 1 終端値 | task 1 終端値 | **主比較（97.5%）** |

### 2.4 半径と射影（`cap_row_norm_` の型）

- 半径 `r_{l,s,i} = ‖W_l[s,i,:]‖`（float32、`torch.linalg.vector_norm(W, dim=2, keepdim=True)`）を **その run 自身の task 1 の 780 回目の更新の直後** に一度だけ計算し、以後変えない。値は `radii.npz` に保存し sha256 を provenance に書く。行全体（W1 は 3072 次元、W2 は 100 次元）の L2 ノルム。seed・行ごとに別の半径（平均しない）。
- **task 2 の最初の更新から**、各 Adam 更新の直後に、対象層ごとに
  `n̂ = vector_norm(W, dim=2)`（float32、半径と同じ呼び出し）・`over = n̂ > r`・`W ← where(over, W·(r/n̂), W)`。
  超えた行だけ向きを保ってノルムを半径へ写す。**上限以下の行（半径と等しい行を含む）は 1 bit も書かない。小さい行を膨らませない。** ゼロ行は `0 > r` が偽なので割り算に届かない（分母は `where(over, n̂, 1)`）。
- 順序: 宿主の Adam 更新（W1・b1・W2・b2・W3・b3）→ W1 の射影 → W2 の射影。**b1・b2・W3・b3・Adam の m と v・更新番号は触らない。** RNG を進めない。
- 記録（課題ごと・slot ごと、課題内の 780 更新の和）: `rows_w1`・`rows_w2`（射影が書いた (更新, 行) の数）、`rem_w1`・`rem_w2`（除去したノルム Σ(n̂ − r)⁺、float64）。
- 監視（全更新・GPU 上で累積）: 射影後の行ノルムを float64 で測り、§7.2 の行ごとの上界を超えた行の数 `viol_w*` と、超過の最大 `excess_w*`（(‖w‖₆₄/r − 1)/eps32）。本走で `viol` が 1 つでもあれば CHECK_FAILED。

### 2.5 fresh 対照（全腕で宿主と同じ手順・上限なし）

宿主どおり、t29 の **まさにそのバッチ列** で、初期値に戻した網（Adam の moment も 0、更新番号 0）を 780 更新学習し、`fresh_gap = fresh(t29) − continual(t29)`。**上限は fresh 網に掛けない**（fresh 対照の前に半径を +∞ に戻す）。fresh 網は初期化から 780 更新だけ学ぶ網で、「task 1 終端の半径」は fresh 網にとって意味を持たないため。結果として fresh(t29) は 4 腕で bit 共通になる（§7 S-fresh・§5 (A) で確かめる）。

### 2.6 出力（腕ごとに `results/cap_cifar5p1_1007/<腕>/`）

- `per_task.csv`: 宿主と同じ列・同じ書式（`arm` 列は活性化名 `R` のまま。腕名はディレクトリと `cap_task.csv` と provenance に入る）。ref はこのファイルが宿主とバイト一致する。
- `fresh_control.csv`: 宿主と同じ。
- `cap_task.csv`（新規）: `cap_arm, seed, slot, task, hard, rows_w1, rem_w1, rows_w2, rem_w2, viol_w1, viol_w2, excess_w1, excess_w2, ratio_max_l1, ratio_med_l1, ratio_max_l2, ratio_med_l2, mu2_norm, b1_mean, b2_mean`。`ratio_*` は課題終端の行ノルム ÷ task 1 終端の行ノルムの最大・中央値（ref でも記録）。`mu2_norm` は課題終端でその課題の訓練画像に対する第 1 層出力 a1 の画像平均のベクトルのノルム（‖µ₂‖、§6 (iv)、REPORT_ONLY）。
- `provenance.json`（宿主の項目 ＋ 上限の項目・spec・登録 commit・checks.json の sha256）、`radii.npz`。

## 3. 登録する読み出し

すべて seed 内の対応差。`online_acc` は宿主の定義（課題内 780 バッチの更新前の当たり率の平均）。

- **E0（早期の水準）**: hard 1–9（t = 1,3,5,7,9）の online 平均。δ0 = E0_arm − E0_ref。t1 は全腕で共通（上限は task 2 から）なので δ0 の 1/5 は構成上 0。
- **E1（後期の水準）**: hard 21–29（t = 21,23,25,27,29）の online 平均。δ1 = E1_arm − E1_ref。
- **E2（喪失）**: fresh gap = fresh(t29) − continual(t29)（`fresh_control.csv` の `fresh_gap`）。δ2 = gap_arm − gap_ref（**負なら喪失が減った**）。
- **読みの注意（登録する）**: fresh(t29) は全腕で bit 共通なので、δ2 = continual_ref(t29) − continual_arm(t29)、つまり **E2 の対応差は t29 の水準差の符号反転** に等しい。E1 と E2 は独立な 2 量ではない（E1 は 5 課題の平均、E2 は t29 の 1 課題）。E2 は「新品の網からの距離」として読めるが、腕間の差としては t29 の水準差である。

## 4. 判定

### 4.1 区間と符号

n = 10、df = 9、標本 SD s、`mean(δ) ± t_{9, 1−α/2} · s/√10`。t の分位点は登録済みの `analysis/resp_cifar_ee_0920/stats.py` の `t_quantile`（連分数による t 分布、無改変で import）。

- **cap12 の δ1・δ2: 97.5% 両側区間**（2 endpoint の Bonferroni、t_{9,0.9875} ≈ 2.685）。cap1・cap2 の δ1・δ2: 95%（t_{9,0.975} ≈ 2.262）。cap12 の 95% も併記する（判定には使わない）。
- 符号: 下端 > 0 で `+`、上端 < 0 で `−`、それ以外（端点がちょうど 0 を含む）は `0`。`0` は「示せない」であって同等ではない。
- SD = 0 のときは区間を平均の 1 点とし（t で割らない）、`DEGENERATE_SD` を併記する。符号は同じ規則（平均 > 0 で +、< 0 で −、= 0 で 0）。

### 4.2 ラベル（腕ごと）

| δ1 | δ2 | ラベル |
|---|---|---|
| + | − | **RESCUED**（水準が上がり喪失が減った） |
| + | 0 | LEVEL_ONLY |
| + | + | LEVEL_UP_GAP_UP（記述） |
| 0 | − | GAP_ONLY |
| 0 | 0 | NO_EFFECT |
| 0 | + | **GAP_UP**（記述。親の表に無い組み合わせ。表を網羅にするため私が足した） |
| − | 任意 | **WORSE** |

主結論は cap12 のラベル。cap1・cap2 は副ラベル。

### 4.3 IMPAIRED（読みのフラグ・ラベルを変えない）

cap 腕ごとに δ0 の **95%** 区間の上端 < 0 なら `IMPAIRED`（上限で最初から学べなくなった腕を区別する）。ラベルと併記し、IMPAIRED の腕を「可塑性を保った」とは書かない。

### 4.4 交互作用（REPORT_ONLY）

seed ごとに `δ_cap12 − δ_cap1 − δ_cap2`（= Δcap12 − Δcap1 − Δcap2 + Δref）を E1・E2 について 95% 区間で報告する。単独腕の非検出を「両方が必要」と言い換えない。

## 5. 適用条件

- **(A) 完全性と同一性**: 4 腕 × seed 0–9 × 30 課題の行と 4 腕 × 10 seed の fresh 行が揃い、`online_acc`・`fresh_gap` が有限、発散 0。ref が §7 S-nochange の意味で既存列と一致（本走の ref の per_task.csv が検査の ref とバイト一致し、検査の ref は (a)(b) を満たす）。さらに **全腕の task 1 の per_task 行が ref とバイト一致**（上限は task 2 から）、**fresh_online_acc が全腕で bit 共通かつ宿主の committed と一致**、`viol_w*` がすべて 0。欠けは `INCOMPLETE`、一致・監視の破れは `CHECK_FAILED`（どちらも科学ラベルを出さない。欠けた seed を除いて n を縮めない）。
- **(B) 喪失の再現**: ref の fresh gap（10 個）の 95% t 区間の下端 > 0。満たさなければ全腕 `NOT_REPRODUCED`。（§1.3 のとおり、宿主の記録から成り立つことは分かっている。）
- **(C) 上限が書いたこと**: cap 腕ごと・対象層ごとに、**task 2–30 の各課題で、10 seed の和の `rows_w*` > 0**。1 課題でも 0 の対象層があれば、その腕は `INAPPLICABLE`（上限が働いた腕の比較ではない）。seed ごとに「書かなかった課題の数」も記述する。

## 6. 副（登録する記述・ラベルにしない）

- **(i)** ref で ‖W2‖（`w_norm_l2`）が t1 → t29 で増えた seed の数（10 中）。同じく `w_norm_l1`。（W が伸びる箱か。§1.3 のとおり宿主の記録で既知。）
- **(ii)** ref で `zbar_l2` が t1 → t29 で負へ動いた seed の数。（既知。）
- **(iii)** 各腕の `dead_frac_l2`・`mob_l2`・`eff_rank_l2`・`zbar_l2` の後期窓（hard 21–29 の課題終端値の平均）の seed 中央値。扉 H の値（.000 / .61 / 13.5 / +0.64、いずれも t29 の seed 中央値）と並べるため、**t29 の seed 中央値も併記**する。
- **(iv)** ‖µ₂‖（`mu2_norm`、§2.6）の t1・t29 と後期窓の seed 中央値、腕ごと。宿主の per_task には相当する量が無いので新たに記録した。REPORT_ONLY。
- **(v)** W3 への規模の逃げ: `w_norm_l3` の t29 の seed 中央値、腕ごと。上限の強さ: `rows_w*`・`rem_w*` の課題別の和、`ratio_*`、`b1_mean`・`b2_mean` の t1 → t29。
- いずれも記述で、媒介や因果は言わない。

## 7. 検査（登録の後・本走の前に全部 PASS させ、列挙した変異で全部 FAIL させる）

`analysis/cap_cifar5p1_1007/checks.py`。各検査は本物のモジュールで PASS し、変異（ソースのちょうど 1 か所の置換をして読み込んだモジュール）で FAIL しなければならない。結果は `results/cap_cifar5p1_1007/checks.json`（`all_pass` は全検査 PASS かつ全変異検出）。検査の走の出力は git の外（セッションの scratchpad）。cap 腕は検査用 seed（100–109）だけで回す。閾値は固定値・固定倍率にせず、下の算術から導く。

| 検査 | 要求 | 変異（FAIL しなければならない） |
|---|---|---|
| **S-nochange** | (a) 本エンジンの ref（seed 0–9・30 課題・graph・threads 2）の per_task.csv・fresh_control.csv が、同じプロセス条件で回した **宿主の無改変 `run()` の出力とファイルごとバイト一致**。(b) committed の `R_std_lr0.0001` と、eff_rank 以外の全列がバイト一致・fresh_control.csv がバイト一致・eff_rank_l1/l2 は全行が §7.1 の帯の中（帯の幅と実際のずれも記録）。 | ref に W2 の上限が入る／lr 1.0001e−4／Adam の更新番号を課題ごとに 0 に戻す（いずれも短縮走で宿主の同条件の出力と比べる） |
| **S-cap** | 合成と実走（cap12・seed 100–102・短縮走・eager・更新ごとの hook）で: 上限以下の行は bit 不変（半径と等しい行を含む）、超えた行は §7.2 の上界の中で半径に乗り向きを保つ、射影は W1・W2 以外（b・W3・m・v）を 1 bit も変えない、`rows`・`rem` の計数が独立に数えた値と一致（rem は §7.2 の丸めの幅）、二度目の適用で書かれる行のノルムの変化が上界の中、どこかの行が実際に書かれた（非空虚） | 半径を初期値の行ノルムにする／bias も射影する／小さい行も半径へ膨らませる／`>` を `>=` にする（半径と等しい行を書く） |
| **S-radius** | 半径が、task 1 の最後の更新の直後の W から独立に計算した行ノルムと bit 一致。task 1 の全更新で射影は書かず半径は +∞、task 2 の最初の更新で半径が有効。cap 腕の task 1 の per_task 行が ref とバイト一致 | 半径を task 2 終端で取り task 3 から掛ける／半径は task 1 終端だが掛け始めが task 3 |
| **S-graph** | cap12・seed 100–109（R = 10）・4 課題 × 100 更新で、graph と eager の per_task.csv・cap_task.csv・fresh_control.csv がバイト一致、最終の P・m・v が bit 一致 | 半径を `copy_` でなく別テンソルの再束縛で入れる（graph が古い +∞ を見続ける）／graph の再生時だけ static_idx を 1 バッチずらす／warmup の巻き戻しから m を落とす |
| **S-fresh** | 短縮走で、ref・cap12 の fresh_control.csv が **宿主の無改変 `run()` の同条件の出力とバイト一致**、fresh の全更新で半径が +∞・書いた行 0 | 半径を +∞ に戻さず fresh 網にも上限を掛ける／fresh の前に Adam の moment を 0 に戻さない |
| **S-verdict** | 合成 fixture で全ラベル（7 種）・端点ちょうど 0 → 符号 0・SD 0（3 通り）・IMPAIRED・Bonferroni（95% では + だが 97.5% では 0 の fixture で cap12 だけ 0）・(B) 不成立 → NOT_REPRODUCED・(C) 不成立 → INAPPLICABLE・t 分位点が表の値（2.2622・2.6850）と 1e−4 で一致 | Bonferroni を外す／窓を hard 19–27 にする／対応差でなく腕ごとの平均の差の区間にする／端点 0 を + にする（`>=`）／IMPAIRED を 97.5% や平均で判定する |
| **S-CLI** | CLI（`python -m src.cap_cifar5p1_1007 run`）の出力が同じ引数の関数呼び出しとバイト一致。既定値（seed 0–9・30 課題・lr 1e−4・threads 2）を確認。登録先への出力は checks.json の all_pass・ソースの sha256 一致・git clean でなければ拒否 | CLI が `--steps-hard` を無視する／既定の課題数が 30 でない |

### 7.1 eff_rank の帯（S-nochange (b)・算術から）

eff_rank = exp(H(p))、p_k = √max(λ_k,0)/Σ√max(λ_j,0)、λ は G = AᵀA（A は課題の画像に対する活性 N × 100、float32 → float64）の固有値。committed（環境 1）と本走（環境 2）の計算値はともに、厳密な固有値 λ から

`|λ̂_k − λ_k| ≤ d = γ_N‖|A|ᵀ|A|‖_F + p(n)·u·‖Ĝ‖₂`（u = 2⁻⁵³、γ_N = Nu/(1−Nu)、p(n) = n = 100）

の中にある（第 1 項は float64 の内積の誤差、第 2 項は対称固有値問題の後退誤差。LAPACK の手引きの実用的な見積もりは p(n) = 1 なので n は保守側）。Weyl の不等式から両環境の固有値の差は 2d 以下。そこで本走の λ̂ を中心に λ ∈ [λ̂ − 2d, λ̂ + 2d] の箱を取り、H = log S − T/S（S = Σs_k、T = Σ s_k log s_k、s_k ∈ [√max(λ̂_k−2d,0), √max(λ̂_k+2d,0)]）を区間演算で上下から抑えて eff_rank の範囲を出す。これに両方の CSV の 10 桁の印字の丸め（相対 5e−10 ずつ）を足したものを帯とする。帯が空虚でないことは、隣の課題の committed 値を入れると外れること（変異）で示す。

### 7.2 射影の上界（S-cap と本走の監視・算術から）

長さ d の行 w、厳密なノルム n、float32 で計算したノルム n̂ = n(1+θ)（θ は行ごとに float64 で実測）。書かれる行は `fl(w · fl(r/n̂))` なので、比と積の丸め（各 u₃₂ = 2⁻²⁴）から

`‖w'‖ / r ∈ [(1−u₃₂)²/(1+θ), (1+u₃₂)²/(1+θ)]`

に float64 の検査側の誤差 γ⁽⁶⁴⁾_{d+1} と、下位正規化への切り捨て √d·η₃₂（η₃₂ = float32 の最小正規数）/r を足したものを行ごとの上界とする。書かれない行は n̂ ≤ r なので ‖w‖ ≤ r/(1+θ) で、同じ上界に入る。向き: 各要素の相対誤差 ≤ u₃₂ なので角度 ≤ u₃₂、`1 − cos ≤ u₃₂²/2 + 2γ⁽⁶⁴⁾_{2d}`。除去量 rem: 同じ float32 の n̂ と r から float64 で作る和で、和の順序の差だけが残るので `|Δ| ≤ γ⁽⁶⁴⁾_{100}·Σ(n̂−r)⁺`。

親の文面の「4·eps32 の相対誤差」は l2cap（d = 100）の値で、d = 3072 の float32 の和の誤差 θ は先験的に 3·eps32 以下とは限らない（最悪 γ₃₀₇₂ ≈ 1.8e−4）。そこで **合否は上の行ごとの上界** で決め、実測の最大の相対誤差を eps32 の単位で併記し、4·eps32 に収まったかを報告する。

## 8. 予測（どの cap 腕の値も見る前・この commit で固定）

採点: 2 値の命題は真偽と binary Brier (p−o)²。cap12 のラベルの分布は、実際のラベルに付けた確率と多値 Brier Σ(p_k − o_k)²（7 ラベル）。NOT_REPRODUCED・INAPPLICABLE・INCOMPLETE・CHECK_FAILED のときは理由付きで未採点。

### 8.1 親（Fable）の事前確率（依頼文から転記）

| 命題 | 確率 |
|---|---:|
| cap12 → RESCUED / LEVEL_ONLY / GAP_ONLY / NO_EFFECT / WORSE | .45 / .15 / .15 / .15 / .10 |
| cap2 の δ1 の平均 > cap1 の δ1 の平均（この箱で死ぬのは第 2 層） | .60 |
| ref で ‖W2‖ が t1 → t29 で 10/10 seed 増える | .80 |
| IMPAIRED が cap12 に付く（lr 1e−4・780 更新では task 1 終端の行ノルムが初期値に近く、上限が学習を縛る可能性） | .30 |

### 8.2 私（Claude）の確率

| 命題 | 確率 |
|---|---:|
| cap12 → RESCUED / LEVEL_ONLY / GAP_ONLY / NO_EFFECT / WORSE / LEVEL_UP_GAP_UP / GAP_UP | .35 / .12 / .05 / .25 / .20 / .01 / .02 |
| cap2 の δ1 の平均 > cap1 の δ1 の平均 | .35 |
| IMPAIRED が cap12 に付く | .15 |
| cap1 の δ1 の符号が +（95%） | .45 |
| cap2 の δ1 の符号が +（95%） | .30 |
| (C) が cap1・cap2・cap12 の 3 腕すべてで成り立つ | .85 |
| cap12 の dead_frac_l2 の後期窓の seed 中央値 < ref の値 | .60 |
| cap12 の w_norm_l3 の t29 の seed 中央値 > ref の値（W3 への逃げ） | .70 |
| E1 の交互作用の 95% 区間が 0 を含む | .55 |
| ref で ‖W2‖ が 10/10 で増える | 既知（宿主の記録で 10/10）・採点しない |

理由（短く）: W1 は 23% → 50% 増の成長を止められ、W2 は task 1 終端でほぼ初期値なので上限は初期値付近に掛かる。ノルムを止めると Adam の相対的な歩幅が保たれる（可塑性に有利）一方、ReLU の死は z₂ の相対位置で決まり尺度では決まらない（§1.2）、向きは自由なまま、W3 は自由。RL-MNIST の ReLU では硬い上限が効かなかった記録がある。W1 の成長の方が大きいので、cap2 より cap1 の方が効くと読む。早期窓は上限がほとんど掛からないうちなので IMPAIRED は低め。

## 9. 実行

- 起動は `analysis/cap_cifar5p1_1007/launch.sh`。checks.json が all_pass、ソースの sha256 が checks.json と一致、`src`・`analysis`・`specs` が git clean、HEAD が origin に push 済み、でなければ止まる。腕の順は **ref → cap1 → cap2 → cap12**（直列、GPU のプロセスは常に 1 本）。各腕は共有 GPU lock `/tmp/lop_analysis_gpu.lock` を flock で取ってから走る。`--threads 2`、空きメモリ 6 GB 未満なら起動しない。1 腕 30 秒前後。途中の成績で腕の順・実行の有無を変えない。
- 集計は `analysis/cap_cifar5p1_1007/verdict.py`（verdict.json・paired.csv・per_seed.csv・secondary.csv・predictions.csv）。結果ノートは `results/cap_cifar5p1_1007/summary.md`。

## 10. 読みの上限

- この箱・この半径（task 1 終端値）・この予算（30 課題）・R/std/lr 1e−4 に限る。上限はノルムだけを止め、向き・bias・W3・Adam の履歴は自由。
- RESCUED は「ノルムの成長を止めると水準と喪失が改善する」までで、W 増大が喪失の唯一の原因だとも、媒介の割合も言わない。L2 Init の結果とは介入が違うので、効果量を並べて「同じ機構」とは言わない。
- NO_EFFECT は「この大きさの対応差を示せない」で、上限が無効だという証明ではない。WORSE は「この半径の硬い上限が水準を下げた」までで、W 増大が有益だとは言わない（RL-MNIST の記録のとおり、硬い射影の副作用でありうる）。
- 床が無い箱なので「救済」は相対的な改善。fresh 網の水準を超える・下回るの読みは §3 の注意のとおり t29 の水準差。

## 11. 出力と片付け

- git に入れる: spec、`src/cap_cifar5p1_1007.py`、`analysis/cap_cifar5p1_1007/`、`results/cap_cifar5p1_1007/`（4 腕の CSV・provenance・radii.npz、checks.json、verdict の出力、summary.md、起動ログ）。
- CLAUDE.md §4: git の外のファイル（`__pycache__` 以外。`data` の symlink は共有データへのリンクなので辿らず、移さない）を `~/Projects/obsidian-research-data/cap_cifar5p1_1007/` に移し、`results/cap_cifar5p1_1007/backup_manifest.json`（source・backup・bytes・sha256）を commit。`git fetch origin && git merge origin/main && git push origin HEAD:main`。worktree とブランチは親が確かめるまで消さない。
