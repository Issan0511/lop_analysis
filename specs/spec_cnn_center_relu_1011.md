# cnn_center_relu_1011 — 標準の CNN（ReLU）でも、入力と中間層の中心化は喪失を防ぐか

状態: **事前登録**（この commit が登録。扉の腕の走はまだ 1 本も無く、扉のコードもまだ書いていない）
作成: 2026-10-11 13:30 JST / 起草: Claude（Opus 5.5・サブエージェント）/ 依頼: 親（Fable）。Issa「走らせたい」
run id: `cnn_center_relu_1011` / worktree `~/Projects/claude/wt/cnn_center_relu_1011` / branch `claude/cnn_center_relu_1011`
親: `specs/spec_cnn_center_1010.md`（C の定義・エンジンの継ぎ方）、`specs/spec_sna_cnn_cause_1009.md`（箱・束ねエンジン）、`specs/spec_relu_doors_0919.md`（扉・梯子・0.5 の線・符号検定）、`specs/spec_doors_rlmnist_1007.md`（扉 H の定義・床・適用条件・変異の作り方）、`specs/spec_relu_doors_h_0919.md`（9/10 の算術）
実装（予定・登録の後に書く）: `src/cnn_center_relu_1011.py`（扉つきの束ねエンジン・診断・CLI）、`analysis/cnn_center_relu_1011/{checks.py,verdict.py,figures.py,launch.sh}`

## 0. 一行と問い

V12 の第一の柱（中心化の扉）は MLP の登録である: RL-CIFAR × ReLU MLP で、ref は課題 1 を学べず、C は課題 1 を学ぶが落ち、CH は 50 課題保つ。H だけ・CS・LN は救わない（V12 §3.3・表 2・付録 E）。これが標準の CNN（Kumar らの CNN・ReLU）に移るかを、同じ扉 C・H・CH を事前登録で回して決める。狙う文は「標準の CNN（ReLU）でも、入力と中間層の中心化は喪失を防ぐ」。

- **Q1（主）**: CH は ReLU の CNN の機能の喪失を防ぐか（主窓 t21–30）。
- **Q2**: C だけ、H だけではどうか（梯子 N: C → H → CH）。
- **Q3**: MLP と同じ並び（ref 死・C 一時・CH 保つ）が出るか（要約 `DOORS_*`）。
- **Q4（副判定）**: どの層が先に機能を失うか。H は入力層で、C は深い層で落ちるか（MLP の `LAYER_CORRESPONDENCE` の移植）。
- **Q5（機構・報告のみ）**: 扉の下で各層の位置は µ 経路と b 経路のどちらで動くか（cnn_center_1010 の分解を 4 層に）。

## 1. 既知（この spec の前に見たもの。すべて登録外の既存データ）

### 1.1 ReLU の CNN は課題 3 前後で死ぬ（出所）

- **宿主の走** `results/rlcifar_cnn_0908/R`（seed 0–9、50 課題、宿主の 1 本ずつのエンジン、git 外・proj_004_drift にだけある）。online の seed 中央値は t1 0.954 → t2 0.943 → t3 0.653 → t4 0.330 → t5 0.220 → t10 0.119 → t30 0.114。t21–30 の窓は 10/10 seed が 0.5 未満、9/10 seed が §5.1 の床以下（seed 7 だけ 0.177）。
- 層ごと（seed 中央値。mob = チャネル／ユニットごとの開いた割合 P(z>0) の中央値）:

| t | 1 | 2 | 3 | 5 | 10 | 30 |
|---|---|---|---|---|---|---|
| c1 mob / z̄ / 死亡率 | 0.186 / −0.52 / 0.09 | 0.175 / −0.77 / 0.13 | 0.076 / −1.58 / 0.13 | 0.033 / −2.89 / 0.13 | 0.014 / −4.70 / 0.13 | 0.006 / −5.78 / 0.16 |
| c2 mob / z̄ / 死亡率 | 0.128 / −3.5 / 0.06 | **0.008** / −19.6 / 0.19 | 0.008 / −27.3 / 0.13 | 0.001 / −58.6 / 0.16 | 0.000 / −148.8 / 0.44 | 0.000 / −177.1 / 0.78 |
| f1 mob / 死亡率 | 0.208 / 0.30 | 0.065 / 0.26 | 0.113 / 0.01 | 0.014 / 0.14 | 0.002 / 0.38 | 0.000 / 0.82 |
| f2 mob / 死亡率 | 0.804 / 0.03 | 0.473 / 0.00 | 0.206 / 0.00 | 0.023 / 0.10 | 0.000 / 0.61 | 0.000 / 1.00 |

  c2 の開いた割合が t2 に 1% を割り（z̄ −3.5 → −19.6）、続いて f1・f2 が閉じる。MLP の C の腕（課題 1 を学んでから第 2 層が沈んで死ぬ）と同じ型に見える。入力層 c1 も生の入力の平均の経路で沈む（z̄ −0.5 → −5.8）が、死亡率は 0.16 に留まる。
- **束ねエンジンの走** `results/cnn_drive_verify_1009/R`（seed 10 だけ、R = 1、3 課題）: online 0.954 → 0.754 → 0.123（課題 3 で死ぬ。V12 付録 L.1 の「ReLU は seed 10 の課題 1〜3 だけ回した（課題 3 で死ぬ）」はこれ）。

### 1.2 leaky の CNN と入力の中心化（cnn_center_1010・cnn_drive_verify_1009、seed 10–19、30 課題、同じエンジン）

- LR（leaky 0.1・生の入力）: online 0.965（t1）→ 0.889（t30）。c1 の z>0 の割合 0.29 → 0.01（t30）、c1 の沈降の 94% は µ 経路（重みの平均パッチ方向）。
- LRc（LR の入力から CIFAR-10 訓練集合のチャネル平均を引く）: 0.976 → 0.943、低下 0.077 → 0.031（10/10）。c1 の沈降は 0.34 倍に減り全部 b 経路へ移る。c2 は両腕とも 100% のチャネルが沈み、LRc のほうが深い（−5.8 対 −4.3）。c2・fc のゲートは両腕でほぼ同じ。

### 1.3 MLP の扉（V12 §3.3・表 2・付録 E、`relu_doors_0919`・`relu_doors_h_ref_0920`）

窓 t31–50（seed 中央値）: ref 0.1132・C 0.1133・H 0.1133・CH 0.9867・CS 0.1132・LN 0.2062。ref は t1 末に第 1 層が死に（0.985）、C は第 2 層が沈んで死に、H は t1 を途中まで学んでから第 1 層が死ぬ（0.935）。登録 N = `NEED_CH`、D2（CH − C）`H_HELPS` 10/10、S2 は `H_ALONE_FAILS`・`C_NEEDED`・`LAYER_CORRESPONDENCE`。乱数ラベル MNIST × ELU（`doors_rlmnist_1007`）では N = `NEED_H`。5+1 CIFAR では H 単独が 10/10 で勝つ。

### 1.4 床の較正（宿主の seed 0–9 で計算した。登録外）

課題ごとの最多ラベルの割合（`major_frac`）は 0.112–0.116（10 課題平均）、10 課題の SD は 0.002–0.005。§5.1 の床は seed ごとに 0.113–0.120 になる。

## 2. 箱（既存の CNN の走と同一）

- Kumar らの CNN: conv 5×5 3→16（pad 2）→ φ → maxpool 2 → conv 5×5 16→16（pad 2）→ φ → maxpool 2 → fc 1024→100 → φ → fc 100→100 → φ → fc 100→10、全層 bias あり、宿主の初期化（`rlcifar_cnn_0908.init_params`）。
- RL-CIFAR: seed ごとに CIFAR-10 訓練集合の固定 1,200 枚（宿主の `rlcc_subset` の流れ）、/255 の [0, 1] 入力、課題ごとに画像ごとの一様乱数ラベル（`rlcc_labels`）、標本順は `rlcc_batch`。
- batch 16、400 epoch = 30,000 更新/課題、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8（moment は課題をまたいで保持）、WD なし、float32（cuDNN の TF32 は既存の走と同じ既定値のまま）。**30 課題**（sna_cnn_cause_1009 の束 B〜D・cnn_center_1010 と同じ）。seed 10–19。
- **活性化は全 4 か所で ReLU**（宿主の "R": `clamp(z, min=0)`。診断のゲートは `z > 0`）。
- エンジン: `sna_cnn_cause_1009` の束ねエンジン（grouped conv ＋ bmm ＋ 1 epoch = 1 CUDA graph、1 プロセス = 1 腕 × seed 10–19 = R 10）。活性化の差し替えは `cnn_drive_verify_1009_lr` の `BundleLeaky`（傾き 0）と同じ interface で、扉を足した新しいファイル `src/cnn_center_relu_1011.py` に書く。既存の `src/*.py` は改変しない。**扉を切った ref の経路は、既存の束ねエンジンの ReLU（`BundleLR` の腕 "R"）と bit 一致しなければならない**（§6 K-same・K-repro）。

## 3. 扉と腕

### 3.1 扉

| 扉 | 定義 |
|---|---|
| **C** | x ← x − μ。μ は CIFAR-10 訓練集合 5 万枚のチャネル平均 (0.4914, 0.4822, 0.4465)（`cnn_center_1010` の LRc と同じ定義・同じ演算。束を作るときに一度だけ引き、学習の全バッチ・online・memo・診断に同じ x を使う）。**割らない**。seed の 1,200 枚の平均との差が残る（平均パッチのノルム 3.76〜3.84 → 0.015〜0.050。cnn_center_1010 の K-center）。 |
| **H** | 4 つの活性化の出力から、**次の層が受け取る量の平均**を引く。conv（c1・c2）: P_l = maxpool(ReLU(z_l)) − m_l、m_l ∈ R¹⁶ は**チャネルごと**（画像と位置で平均）。fc（f1・f2）: a_l = ReLU(z_l) − m_l、m_l ∈ R¹⁰⁰ は**ユニットごと**（画像で平均）。m_l ← (1−β) m_l + β·mean_batch(中心化前の出力)、**β = 0.01**、初期値 0、課題の境界でリセットしない。演算は `m.mul_(1-β).add_(bm, alpha=β)`。m は勾配を通さない定数で、Adam の対象でもない。更新は各 Adam 更新の直後で、bm はその更新の勾配を出した順伝播（更新前の重み）の中心化前の出力から取る。**online・memo・診断・切替の CE は、すべてその時点の m を使う**（訓練と評価で同じ値）。ゲート（ReLU の微分）は変えない（∂a/∂z = 1[z>0]、m は定数のずれ）。分散正規化も affine も無い。MLP の H（V12 §10.2・doors_rlmnist §2.2）の移植。 |

H の CNN への移し方（開示）:
- conv の m は **pool の後**の平均（次の層が実際に受け取る量の平均）とする。max-pool は定数のずれと可換（maxpool(u − c) = maxpool(u) − c）なので、ReLU の出力から m を引いてから pool しても値は同じ。pool の前の平均を引くと、pool の勝者の上乗せ（窓の max ≥ 窓の平均）が次の層の入力の DC として残り、MLP の H の目的（次の層の入力の平均を ≈ 0 にする。doors_rlmnist §1.2）を満たさない。
- チャネルごとの中心化は、f1 の入力（16 チャネル × 8×8 位置 = 1,024 特徴）の**位置ごとの**平均（チャネルの平均マップが位置で違う分）を消さない。MLP の H は特徴ごとに引くので、ここは MLP と違う。f1 の入力に残る DC は `r_in_f1`（§4）で測る。
- C の c1・H の c2 では、入力が中心化されているので conv の zero padding は平均で埋めることになる（cnn_center_1010 と同じ。縁のパッチの扱いが生の入力と違う）。

### 3.2 腕（主 4 腕 × seed 10–19 = 40 run、対照 1 run）

| 腕 | C | H | seed | 役割 |
|---|---|---|---|---|
| ref | − | − | 10–19 | 素の ReLU（§1.1 で課題 3 前後に死ぬ） |
| C | ✓ | − | 10–19 | 入力の中心化だけ |
| H | − | ✓ | 10–19 | 中間層の中心化だけ（MLP では救わなかった） |
| **CH** | ✓ | ✓ | 10–19 | **主**。両方 |
| Cr | 乱数の定数 | − | 10 | 変異対照（§3.3） |

扉は課題 1 の最初の更新から網の構造として入れる（MLP の扉と同じ）。各腕は自分の束（R = 10）で走るので、腕どうしの比較は seed 対応で、bit の比較ではない。

### 3.3 変異対照（1 seed）

- **Cr（中心化でなく定数シフト）**: C の μ の代わりに、中心化しない定数 r を引く。r の作り方: u ~ N(0, I₃) を `pmnist_0905.stream("cnn_center_relu_1011_cr", 0)` から float64 で引き、μ̂ に直交する成分 v = (u − (u·μ̂)μ̂)/‖·‖ を取り、**r = μ − ‖μ‖ v**。シフト後の入力のチャネル平均は ≈ ‖μ‖ v（**生の入力と同じ大きさ 0.82 の DC が、生の DC と直交する向きに残る**）。シフトの大きさは ‖r‖ = √2 ‖μ‖ = 1.16 で、C のシフト 0.82 より大きい。µ̂ の成分はすべて正なので、v は符号の混じった向きになる。実装の前に計算した値: v = (0.2866, 0.4772, −0.8307)、r = (0.2562, 0.0905, 1.1282)、μ − r = (0.2352, 0.3916, −0.6817)（検査で float32 の正確な値を記録する）。生の入力と同じ大きさの DC を残すので、**Cr が ref と同じ側に出て C と違えば、C の効きはシフトでなく DC を消すこと（中心化）から来る**と読む。
- **H の m を評価時にだけ外す（装置の確認）**: H と CH の全 seed の全課題末に、m = 0 で評価をもう一度行い `memo_acc_m0`・`fit_ce_m0` を記録する（読むだけで、学習の状態には触れない。§6 K-evaloff）。依頼の「1 seed」を全 seed にしたのは、学習を増やさず費用がほぼ 0 だから。訓練が m で調整した網から m を外すと、次の層の入力が一斉に +m ずれる。memo がどれだけ落ちるかが、網が扉にどれだけ依存しているかの量になる。同じ変異（評価で m を使わない）を §6 の K-H-eval が検出できることも確かめる。

## 4. 測る量

課題末ごと・run ごとに `per_task.csv`（エンジンの列 ＋ 下の列）。課題 0（初期値）は `t0.csv` に別に書く。

- `online_acc`（30,000 更新の更新前の argmax の当たり率の平均）、`memo_acc`（課題末に 1,200 枚）、`switch_ce`・`switch_acc`、`ep_first99` など（エンジンの列）。
- `major_frac`: その課題の 1,200 個のラベルのうち最多のクラスの割合（床に使う）。
- 層 l ∈ {c1, c2, f1, f2} ごと（conv はチャネル、fc はユニットを単位にする。チャネルの量は全画像 × 全位置）:
  - **位置**: `zbar_l`（z̄_j の単位中央値。エンジンの列）、`zbar_mean_l`（単位平均）。
  - **幅**: `zsd_l`（sd_j の単位中央値。エンジンの列）。
  - **開いた割合**: `mob_l`（P(z>0) の単位中央値。エンジンの列）、`open_mean_l`（全要素の P(z>0)）。
  - **死亡率**: `dead_frac_l` = 全 1,200 枚・全位置で z ≤ 0 の単位の割合（宿主の `dead_frac` と同じ定義: 診断のゲート `z>0` の最大が 1e−6 未満）。
  - **入力の DC**: `r_in_l` = ‖その層の入力の平均（conv はパッチ、zero padding 込み）‖ / rms ‖偏差‖。
  - **µ 経路と b 経路**: z̄_j = ⟨W_j, m_in,l⟩ + b_j（線形性の恒等式。m_in,l はその層の入力の平均パッチ／平均ベクトル）。`wm_med_l`・`b_med_l`（各項の単位中央値）、`ident_res_l`（恒等式の残差の最大。§6 K-ident の上界と比べる）。
  - `eff_rank_l`・`w_norm_*`（エンジンの列）。
- H の腕: `m_norm_l`・`m_mean_l`（扉の状態）、`h_upd_l`（その課題の更新のうち m_l が bit で変わった更新の数。30,000 が満点）、`memo_acc_m0`・`fit_ce_m0`（§3.3）。
- `units.npz`（git に入れない）: 課題 0〜30 の各 run・各単位の z̄_j・sd_j・開いた割合・死亡・b_j・⟨W_j, m_in⟩、各層の m_in、扉の m。
- チェックポイント: t1・2・3・5・10・20・30 の重み・Adam・扉の状態（git に入れない）。

派生量（verdict が計算する）:
- **主窓** W = t21–30 の online の平均（seed ごと）。副窓 t1–5・t6–10・t11–20 も報告する。低下 = t1–5 の平均 − t26–30 の平均。
- **適合率** q(t) = (A(t) − 0.1)/(A(1) − 0.1)（A は online、A(1) は**その run 自身の**課題 1）。**網の機能の低下の時刻 T_f** = q(t) < 1/2 になる最初の t ≥ 2（V12 §4.0。31 で打ち切り）。
- **層の機能の低下の時刻 T_d,l** = `mob_l` < 0.01 になる最初の t ≥ 1（V12 §4.0。31 で打ち切り）。**最初に落ちる層** = T_d,l が最小の層の集合（同着は全部）。全層が打ち切りなら「なし」。
- **µ 経路と b 経路の分解**（cnn_center_1010 §3 の分解を 4 層に）: t0 → t の Δz̄_j = Δ⟨W_j, m_in⟩（µ 経路）＋ Δb_j（b 経路）。c2・f1・f2 では µ 経路をさらに ⟨ΔW, m₀⟩（自分の重み）＋ ⟨W₀, Δm⟩（上流）＋ ⟨ΔW, Δm⟩（交差）に割る（チェックポイントの課題で）。

## 5. 判定

### 5.1 seed ごと

- **KEPT**: W ≥ 0.5。0.5 は relu_doors_0919 の救命の操作的定義をそのまま継ぐ（V12 表 2 と同じ線）。算術の対応: CNN の課題 1 の水準 A(1) ≈ 0.954（§1.1）では、W = 0.5 は適合率 q = (0.5 − 0.1)/(0.954 − 0.1) = 0.47 で、V12 §4.0 の「機能の低下」（q < 1/2）の線にほぼ重なる。
- **AT_FLOOR**: W ≤ F_s + 3 s_s/√10。F_s と s_s は、その seed の t21–30 の `major_frac` の平均と標本 SD（doors_rlmnist_1007 §4.1 の床を 10 課題の窓に移したもの）。

### 5.2 腕のラベル（RESCUED ／ PARTIAL ／ FAIL）

有効な seed（30 課題を完走し、online と memo が有限）の数を n とする。n < 8 なら INAPPLICABLE。k*(n) は、両側の正確な二項検定で「n 本中 k 本以上が片側」の p が 0.05 を下回る最小の k（n = 10 で 9: 2·(1+10)/2¹⁰ = 0.0215。8 本では 0.109。n = 9 で 8、n = 8 で 8）。

- **RESCUED**: KEPT の seed が k* 本以上。
- **FAIL**: KEPT の seed が n − k* 本以下（n = 10 で 1 本以下。relu_doors の COLLAPSED）。FAIL の内訳として、AT_FLOOR が k* 本以上なら `FAIL_FLOOR`、それ以外は `FAIL_ABOVE_FLOOR`（機能は失ったが当て推量の床の上に残る）。
- **PARTIAL**: それ以外（n = 10 で KEPT が 2〜8 本）。

### 5.3 対応のある符号検定（主窓 W、seed 対応、同値を除く、両側の正確な二項検定）

D_C = C − ref、D_H = H − ref、D_CH = CH − ref、D_{H|C} = CH − C、D_{C|H} = CH − H。p < 0.05 かつ逆向きの seed が 1 本以下なら `HELPS`（負の向きなら `HURTS`）、それ以外は `TIE`（relu_doors §4 と同じ規則。`TIE` は「示せない」で、同等の意味ではない）。

### 5.4 適用条件

- **(A)** §6 の検査が all_pass。主 4 腕の有効 seed が各 8 以上。
- **(B)** ref が FAIL。満たさなければ、N・DOORS・LAYER は `NOT_REPRODUCED`（腕のラベルと符号検定は報告する）。
- **(C)** 扉が働いた記録: H と CH の全 seed・全課題・全層で `h_upd_l` = 30,000、またはその課題末にその層の `dead_frac_l` = 1（層が全部死ぬと出力が 0 になり、m は 0 へ減衰して下位桁で止まる。そこは扉の故障ではない）。C と CH は、全 seed で入力の平均パッチのノルムが生の入力の 0.1 倍未満。満たさない腕は INAPPLICABLE。

### 5.5 梯子 N（順序は C → H → CH で固定）

C が RESCUED なら `NEED_C`、そうでなく H が RESCUED なら `NEED_H`、そうでなく CH が RESCUED なら `NEED_CH`、どれでもなければ `NONE_RESCUED`。PARTIAL は数えない。

### 5.6 要約（Q3）

- **`DOORS_TRANSFER`**（MLP と同じ並び）: 次の 4 つが全部成り立つ。
  1. ref が FAIL（ref 死。(B) と同じ）。
  2. C が RESCUED でなく、かつ C が課題 1 を学ぶ（課題 1 の online ≥ 0.5 の seed が k* 本以上）（C 一時）。
  3. CH が RESCUED（CH 保つ）。
  4. D_{H|C}（CH − C）が `HELPS`（C に H を足すことが効く。MLP の D2 `H_HELPS` の移植）。
- **`DOORS_FAIL`**: CH が FAIL。
- **`DOORS_PARTIAL`**: それ以外（CH が PARTIAL、または CH は RESCUED だが 1・2・4 のどれかが崩れる。例えば C だけで RESCUED（並びが MLP と違う））。
- (A)/(C) を満たさなければ `INAPPLICABLE`、(B) を満たさなければ `NOT_REPRODUCED`。

### 5.7 H 単独（MLP では救わなかった）

M_H = H の腕のラベル。MLP との対応は、FAIL なら `H_LIKE_MLP`、RESCUED なら `H_DIFFERS_FROM_MLP`、PARTIAL なら `H_PARTIAL`。

### 5.8 層の対応（Q4。(B) が必要）

**`LAYER_CORRESPONDENCE`**: 次の 2 つが両方成り立つ。それ以外は `LAYER_CORRESPONDENCE_NOT_SHOWN`（どれが外れたかを書く）。
1. H の「最初に落ちる層」に c1 が入る seed が k* 本以上（H は入力層で落ちる）。
2. C の「最初に落ちる層」が {c2, f1, f2} の中だけにある seed が k* 本以上（C は深い層で落ちる）。

MLP（relu_doors_h_ref_0920）の「H は第 1 層で、C は第 2 層で死ぬ」の移植。H が落ちなければ 1 は成り立たない。

### 5.9 フラグ（ラベルを変えない）

- **`RESIDUAL_LOSS`**（CH）: 低下（t1–5 − t26–30）が正の seed について符号検定（同値を除く）で p < 0.05 かつ負の seed が 1 本以下。救っても喪失が残っていることを示す。
- **`T1_LOWER`**（C・H・CH それぞれ）: 課題 1 の online が ref より低い向きに §5.3 の規則で `HURTS`。扉が課題 1 の学習を遅くしたことを示す。

### 5.10 変異対照のラベル

- **Cr**（seed 10 の 1 本）: 窓 t1–5・t6–10・t11–20・t21–30 のうち、ref と C の 10 seed の範囲 [min, max] が重ならない窓を「分ける窓」とする。分ける窓のすべてで Cr が ref の範囲に入れば `CONST_SHIFT_NO_EFFECT`、すべてで C の範囲に入れば `CONST_SHIFT_ACTS_LIKE_C`、分ける窓が無ければ `CR_UNINFORMATIVE`（C が ref と区別できないので、対照の意味が無い）、それ以外は `CR_UNRESOLVED`。
- **評価だけ m を外す**（H・CH）: t30 の memo_acc − memo_acc_m0 の符号検定（§5.3 の規則）で正なら `EVALOFF_DEPENDS`（網が扉の m に依存している）、それ以外は `EVALOFF_NO_DEPENDENCE`。

### 5.11 報告のみ（判定に使わない）

- T_f の腕ごとの中央値と、ref との seed 対応の符号検定（`LATER` ／ `EARLIER` ／ `NO_TIMING_DIFF`）。
- 層ごとの位置・幅・開いた割合・死亡率・r_in の課題列（t1・2・3・5・10・20・30）、T_d,l、最初に落ちる層の分布。
- µ 経路と b 経路の分解（§4）。b 経路の取り分 |Δb|/(|Δb| + |Δ⟨W, m⟩|) の単位中央値。
- 主窓の水準を、同じ seed・同じエンジンの leaky の LR（cnn_drive_verify_1009）と LRc（cnn_center_1010）と並べる（活性化が違うので比較は記述だけ）。
- memo_acc と memo_acc_m0 の課題列。

### 5.12 読み方（登録）

- **`DOORS_TRANSFER`** → V12 の要旨に「標準の CNN（ReLU）でも、入力と中間層の中心化は喪失を防ぐ（主窓 ref x → CH y、10 seed）」と書ける。C 単独が救わないこと、CH − C の符号検定を併記する。`RESIDUAL_LOSS` が立てば「崩壊は防ぐが、低下は残る」と書く。
- **`DOORS_PARTIAL`** → 何が崩れたかを書く。C だけで救うなら「CNN では入力の中心化で足りる」で、並びは MLP と違う。CH が PARTIAL なら「一部の seed だけ救う」。
- **`DOORS_FAIL`** → 中心化は ReLU の CNN の崩壊を防がない。§5.11 の分解で、扉の下で位置を運んだのが b 経路か、f1 の入力に残る位置ごとの DC か、別の経路かを書く。
- 「救う」は「この箱の 30 課題の主窓で、10 seed 中 9 本以上が online 0.5 以上」の意味に限る。無限時間の非崩壊を意味しない。
- すべて「Kumar らの CNN・RL-CIFAR・ReLU・seed 10–19・30 課題・扉は課題 1 から・β = 0.01」に限る。

## 6. 検査（この spec の commit の後、本走の前に実行する。許容は導出から決め、値を見て決めない）

`analysis/cnn_center_relu_1011/checks.py` → `results/cnn_center_relu_1011/checks.json`。各検査は本物のコードで pass し、列挙した変異で fail しなければならない。変異は対象ファイルの文字列の 1 回置換で作る（doors_rlmnist_1007 の `load`）。

| 検査 | 要求 | 変異（検出すべき） |
|---|---|---|
| **K-same** | ref の束（R = 3）が `cnn_drive_verify_1009_lr.BundleLR`（腕 "R"）と 2 epoch（eager）で全状態 bit 一致 | ref でも H が働く／ReLU を leaky に |
| **K-repro** | ref の腕を R = 1・seed 10・3 課題で回すと、`results/cnn_drive_verify_1009/R/per_task.csv` の共通の全列と `curves.npy` が bit 一致（既存の ReLU の CNN の走の再現） | — |
| **K-center** | C の入力テンソルが `cnn_center_1010.BundleLRc` の入力テンソルと seed 10–19 で bit 一致（LRc と同じ前処理で活性化だけ違う）。平均パッチのノルムが生の 0.1 倍未満。Cr の r が §3.3 の式どおりで、シフト後のチャネル平均のノルムが生の ±5% 以内（seed の部分集合の平均と訓練集合の平均の差の分） | C が平均で割る／C が別の定数を引く |
| **K-relu** | φ が `clamp(z, min=0)` と bit 一致、診断のゲートが `z>0`。訓練の微分（autograd）が `z>0` で 1・`z<0` で 0 | φ を leaky 0.01 に |
| **K-host** | 1 step の損失と全勾配が、宿主の `forward_cnn`（腕 "R"）と、C は中心化した画像で、一致（損失の相対差 1e−5・勾配の相対差 1e−4。cnn_center_1010 の K-host と同じ許容）。H と CH は、独立に書いた扉つきの 1 run の順伝播（m を定数で与える）と同じ許容で一致 | φ の中で中心化（φ(z − m)）／pool の前で m を引かず f1 だけ引く |
| **K-H-ema** | 捕捉した更新で、m の更新が「中心化前の出力（conv は pool の後）のバッチ平均を β = 0.01 で追う」と bit 一致。bm を出した z が更新前の重みの順伝播の z と bit 一致。課題の境界（新しいラベル）で m が変わらない | 中心化後を追う／β = 0.1／pool の前の平均を追う／境界で m をリセット／Adam の後の重みで bm を取る |
| **K-H-grad** | m は勾配を持たず、全勾配が「m を定数とした独立の再計算」と bit 一致 | 勾配を通す（値は同じで、勾配だけバッチ平均を通る） |
| **K-H-eval** | 課題末の memo・診断・切替の CE が、その時点の m での再計算と bit 一致し、m = 0 では一致しない | 評価で m を使わない |
| **K-H-first** | H の最初の更新が ref の最初の更新と bit 一致（m = 0 の順伝播なので）、その直後の m は非零、3 更新後は ref と違う。CH と C も同じ | m の初期値が非零 |
| **K-evaloff** | m = 0 の評価（`memo_acc_m0`）は読むだけ: 前後で全状態・m・乱数の流れが bit 一致、2 epoch の軌道がこの評価の有無で bit 一致 | m = 0 の評価が m をその場で 0 にする |
| **K-ident** | 課題末の診断で z̄_j = ⟨W_j, m_in⟩ + b_j が、TF32 と float32 の丸めから導いた上界 (2⁻¹⁰ + γ_{K+1})·(⟨\|W_j\|, mean\|p\|⟩ + \|b_j\|)（K は内積の長さ、γ_k = k·2⁻²⁴/(1 − k·2⁻²⁴)）以内で成り立つ（ref・CH、全 4 層） | 平均パッチの並びを転置／H の腕で m_in を中心化前の入力から取る |
| **K-graph** | H と CH の束で、CUDA graph と eager の 2 epoch が m・h_upd を含めて bit 一致 | — |
| **K-indep** | 束の中の run 1 の重み・入力・m を乱すと、run 0・2 は bit で変わらず run 1 は変わる | — |
| **K-resume** | 2 課題（2 epoch/課題）を通しで回すのと、1 課題で止めて再開するのとで per_task・units・最終状態が bit 一致 | — |
| **K-verdict** | 合成の per_task で、設計どおりのラベル（腕の 3 値と FAIL の内訳・符号検定・N の 4 値・DOORS の 3 値・NOT_REPRODUCED・INAPPLICABLE・LAYER・フラグ・Cr・EVALOFF）が出る | k* の取り違え／0.5 の線を > に／床の √10 を √30 に／主窓の課題範囲／TRANSFER の条件 4 を落とす／梯子の順序／LAYER の c1 の判定／(C) の死んだ層の免除を落とす |
| **K-cost** | 1 腕の束（R = 10、CH）の 1 epoch の秒数（単独と、主 4 腕の同時起動）と GPU メモリの峰。§8 の同時本数と所要時間をこれで決める | — |

**許容の導出**: bit 一致の検査は許容 0。K-host の 1e−5・1e−4 は cnn_center_1010・cnn_drive_verify_1009 と同じ（束ねの bmm と宿主の mm の float32 の丸めの差、1 step で 1e−7 程度）。K-ident の上界は、conv の入力と重みが TF32 に丸められうること（各 2⁻¹¹ の相対誤差、積で 2⁻¹⁰）と、K 項の float32 の和の誤差 γ_K から来る（float64 の集計の丸めは無視できる大きさ）。

## 7. 事前予測（扉の腕を 1 本も回す前・扉のコードを書く前。この commit で固定する）

### 7.1 Claude（Opus 5.5、2026-10-11 13:30 JST、この spec の起草者）

読み: 宿主の ReLU の CNN は、MLP の C の腕と同じ型で死ぬ（課題 1 を学んでから c2 が t2 に閉じ、続いて f1・f2）。入力の中心化 C は c1 の µ 経路だけを塞ぎ、c2 の入力（ReLU の出力は非負で DC が大きい）には触らないので、救わないと見る（LRc では c2 がむしろ深く沈んだ）。H は c2・f1・f2 の µ 経路を塞ぐが、c1 は生の入力の µ 経路で沈み続ける（宿主の ref でも c1 の開いた割合は t30 で 0.006、leaky の LR では 0.01）。深い層が生きている分 c1 に勾配が届き続けるので、H 単独は c1 で落ちる側を本命にする。CH は全層の µ 経路を塞ぐが、チャネルごとの中心化は f1 の入力の位置ごとの DC を残すので、MLP ほど確実ではないと見る。

略号: 他 = INAPPLICABLE・NOT_REPRODUCED をまとめた桶。

| # | 項目 | Claude | Issa | Fable |
|---|---|---|---|---|
| L1 | ref のラベル | FAIL .93 ／ PARTIAL .04 ／ RESCUED .01 ／ 他 .02 | | |
| L2 | C のラベル | FAIL .72 ／ PARTIAL .16 ／ RESCUED .10 ／ 他 .02 | | FAIL .70 ／ PARTIAL .22 ／ RESCUED .08 |
| L3 | H のラベル（M_H） | FAIL .40 ／ PARTIAL .30 ／ RESCUED .28 ／ 他 .02 | | FAIL .45 ／ PARTIAL .30 ／ RESCUED .25 |
| L4 | **CH のラベル（主）** | **RESCUED .64** ／ PARTIAL .20 ／ FAIL .14 ／ 他 .02 | | RESCUED .60 ／ PARTIAL .25 ／ FAIL .15 |
| L5 | 梯子 N | NEED_CH .36 ／ NONE_RESCUED .30 ／ NEED_H .24 ／ NEED_C .08 ／ 他 .02 | | NEED_CH .40 ／ NONE_RESCUED .30 ／ NEED_H .20 ／ その他 .10 |
| L6 | **要約** | **DOORS_TRANSFER .50** ／ DOORS_PARTIAL .30 ／ DOORS_FAIL .13 ／ 他 .07 | | DOORS_TRANSFER .45 ／ DOORS_PARTIAL .35 ／ DOORS_FAIL .12 ／ NOT_REPRODUCED .08 |
| L7 | LAYER（§5.8） | CORRESPONDENCE .30 ／ NOT_SHOWN .65 ／ NOT_REPRODUCED .05 | | CORRESPONDENCE .30（他 .70） |
| L8 | Cr（§5.10） | CR_UNINFORMATIVE .50 ／ CONST_SHIFT_NO_EFFECT .30 ／ CR_UNRESOLVED .10 ／ CONST_SHIFT_ACTS_LIKE_C .10 | | |

| # | 主張（二値） | Claude | Issa | Fable |
|---|---|---|---|---|
| B1 | ref の FAIL の内訳が `FAIL_FLOOR`（ref が FAIL のときだけ採点） | .60 | | |
| B2 | ref の「最初に落ちる層」に c2 が入る seed が 9 本以上 | .55 | | |
| B3 | C の T_f が ref より `LATER`（報告のみの符号検定） | .25 | | |
| B4 | H の t30 の c1 の `mob`（seed 中央値）< 0.01 | .50 | | |
| B5 | CH の主窓 online（seed 中央値）≥ 0.90 | .50 | | |
| B6 | CH に `RESIDUAL_LOSS` が立つ | .70 | | |
| B7 | CH の低下（seed 中央値）が LRc の低下（leaky・同じ seed、0.031）より小さい | .35 | | |
| B8 | CH に `T1_LOWER` が立つ | .12 | | |
| B9 | CH が `EVALOFF_DEPENDS` | .90 | | |
| B10 | CH の t0 → t30 の b 経路の取り分（単位中央値の seed 中央値）が c2 と f2 の両方で 0.5 超 | .75 | | |
| B11 | CH の同じ取り分が f1 で 0.5 超（f1 の入力には位置ごとの DC が残る） | .45 | | |
| B12 | ref の c2 の t0 → t30 の Δz̄（単位中央値の seed 中央値）< −50 | .80 | | |

### 7.2 採点の規則（登録）

- ラベルの項目（L1〜L8）は、実現したラベルに付けた確率、最頻ラベルの的中、多クラスの Brier 得点 Σ_k (p_k − 1[k = 実現])² を報告する。「他」は 1 つの桶で、名前の無いラベルの確率は 0。
- 二値の項目（B1〜B12）は、真偽と binary Brier (p − o)² を報告する。条件つきの項目（B1）は条件が成り立たなければ「未採点（理由）」。
- Issa と Fable の列は空欄で登録する。走る前に記入されたものだけを採点する。
- **Fable の列の記入**（2026-10-11 13:57 JST、親から受け取った値をそのまま写した。本走の前で、扉の腕の結果は 1 つも無い。親が予測を書いた時点で見ていたのは、この spec・§11 の追補・私の送った要約（腕・判定・Claude の予測・GPU・試走の時間）まで）。Fable の「その他」「他」は、名前の無いラベル全部をまとめた 1 つの桶として採点する。L1・L8 と二値の項目は記入なし。

## 8. 費用と実行

- 1 run の費用の目安: cnn_center_1010 の LRc（leaky、R = 10、30 課題）は単独の GPU で 1,530 秒、`cuda_max_mem_mb` 436。扉 H は 1 更新あたり 4 本の EMA と 4 本の引き算、課題末の診断が 1 回増える。K-cost の実測で置き換える。
- **同時本数は GPU メモリから算術で決める**: n = ⌊(M_free − M_keep) / (1.25 × M_proc)⌋。M_free は起動直前の `nvidia-smi` の空き、M_proc は K-cost で測った 1 プロセスの峰（`nvidia-smi` のプロセスごとの使用量。CUDA の context を含む）、M_keep は同時に GPU を使う別 agent の `adamw_dose2_1011` の分（起動前に `nvidia-smi` で実測した使用量と、その spec に書かれた予定の和）に 2 GiB の余裕を足したもの。n が 5（主 4 腕 ＋ Cr）以上なら全部を同時に、未満なら主 4 腕を先に、Cr を最後に回す。各プロセスに `--mem-gb` で上限を置く（ollama がモデルを読み込んでも評価の峰を確保できるように、エンジンは起動時に評価の峰を予約する）。起動前に `nvidia-smi` と、マシン全体の proj_004_drift の python の本数（`pgrep -c -f 'proj_004_drift/\.venv/bin/python'`。operatingsis の python は数えない。pgrep -f の自己一致に注意し、`[p]roj_004_drift` の形で数える）を見る。決めた n と所要時間の見込みは、本走の前に §11 の追補に書く。
- 起動は、実装の commit・検査の all_pass・HEAD の push・親（Fable）の go・台帳 `PREDICTIONS.md` への登録の後。`nohup setsid` で切り離し、ログは `results/cnn_center_relu_1011/logs/<arm>.log`（ログ名に `/` を含めない）。
- 走っている途中の効果量を見て設計を変えない。監視は完了課題の数・失敗・ログの末尾の課題番号と秒数だけを見る（エンジンのログは腕の online の平均を毎課題印字するので、ログの末尾は `tail -c` で課題番号と秒数の部分だけを読む）。
- python は `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- 走: `results/cnn_center_relu_1011/<arm>/`（`per_task.csv`・`t0.csv`・`curves.npy`・`provenance.json` は git に入れる。`units.npz`・`ckpt/`・`state.pt` は入れない）。
- 判定: `analysis/cnn_center_relu_1011/verdict.py` → `results/cnn_center_relu_1011/{verdict.json,summary.md,tables/*.csv}`、図 `figures.py` → `results/cnn_center_relu_1011/fig_*.png`。
- 片付け: git 外のファイル（`units.npz`・`ckpt/`・`state.pt`・ログ・検査の作業場所）を `~/Projects/obsidian-research-data/cnn_center_relu_1011_1011/` へ移し、`results/cnn_center_relu_1011/backup_manifest.json`（source・backup・bytes・sha256）を commit する。`data` の symlink は辿らない。main に merge・push してから worktree とブランチを消す。
- vault: `可塑性喪失/測定/` に結果ノート、`ハブ/走の履歴.md` と `ハブ/論点マップ.md` に 1 行ずつ。V12 は触らない。

## 10. 開示

- §1 の数値（宿主の ReLU の走・LR・LRc・MLP の扉）は全部この spec の前に読んだ。とくに宿主の ReLU の走の層ごとの値（§1.1）を見てから予測を書いた。
- 床の較正（§1.4）は宿主の seed 0–9 のラベルの流れから計算した（本走の seed 10–19 のラベルは見ていない。ラベルの流れは腕によらず seed だけで決まるので、本走の後に同じ計算をする）。
- Cr の v の概数（§3.3）は、r の作り方を決めるために流れから 1 回計算した。最初に考えた作り方（r = μ + ‖μ‖u。残る DC の大きさは生と同じで向きは乱数）では、この流れの u がたまたま −μ̂ に近く（u·μ̂ = −0.95）、残る DC が生とほぼ同じ向き（cos 0.95）になって対照の意味を失った。そこで向きを µ̂ に直交させる作り方に替えた。値を見て作り方を替えたことを開示する（精度は何も見ていない）。
- 扉 H の「pool の後の平均」と「チャネルごと」は、依頼文の「各 conv 層の活性化はチャネルごと…出力の平均を画像方向の EMA で引く」を、MLP の H の目的（次の層の入力の平均を 0 にする）に合わせて読んだもの（§3.1）。

## 11. 追補 1 — 実装・検査・試走（2026-10-11 13:45 JST、本走の前。判定・窓・予測は変えていない）

- 実装 commit `0a5c0873`。その上で `analysis/cnn_center_relu_1011/checks.py` の全実行が **all_pass**（16 項目 ＋ K-cost、変異 27/27 を検出、例外で落ちた変異は 0）。`results/cnn_center_relu_1011/checks.json`（`git_dirty_code` = false）。
  - **K-repro**: CLI の ref・R = 1・seed 10・3 課題が `results/cnn_drive_verify_1009/R` を、per_task の共通 57 列（文字列）と `curves.npy`（bit）で再現した。既存の ReLU の CNN の走との一致。
  - **K-same**: ref の束（R = 3）が `BundleLR`（腕 "R"）と 2 epoch で全状態・V・evaluate・switch_eval まで bit 一致。
  - **K-center**: C の入力テンソルは `cnn_center_1010` の LRc の入力と seed 10–19 で bit 一致。平均パッチのノルム 3.76〜3.84 → 0.015〜0.050。Cr の r = (0.2562, 0.0905, 1.1282)、シフト後のチャネル平均のノルムは生の 0.989〜1.012 倍、生の平均との cos は −0.009〜+0.016。
  - **K-host**: 1 step の損失の相対差 0、勾配の相対差 ≤ 1.8e−7（ref・C は宿主の forward_cnn、H・CH は独立に書いた扉つきの 1 run の順伝播）。
  - **K-ident**: 恒等式の残差は上界の最大 0.033 倍。
  - **K-relu**: 訓練の微分は z > 0 で 1、z < 0 で 0、z = 0 ちょうどで 1（`clamp` の規約）。診断のゲートは z > 0（宿主の `dead_frac` の定義）。
- **実装時の明確化**（設計は変えていない）:
  1. ログには精度を印字しない（課題番号・秒数・腕名だけ）。§8 の「tail -c で読む」より強い形にした。
  2. K-center の Cr の許容は、部分集合の平均と訓練集合の平均の差 d から導いた 2|d|/(|μ| − |d|)（seed ごとに 0.003〜0.029）にした。§6 の 5% より狭く、5% も同時に確かめる。
  3. K-H-eval は memo（数え上げで偶然一致しうる）でなく、課題末の fit CE と切替の CE（連続量）で比べた。
  4. **K-loop** を足した: `run` の課題の回し方（eager）が手での再生と bit 一致し、m が課題の境界で引き継がれ、h_upd が 1 課題の更新数に等しい。変異「run の中で境界に m をリセット」を検出する。
  5. 診断の chunk は 10（c2 の展開した平均パッチのメモリの峰を抑えるため）。
- **見たもの（開示）**: 実装中の smoke test で、CH の seed 10–11 を課題 1 の 3 epoch だけ回した値（memo 0.165、f1 の入力の r_in 0.52（EMA の立ち上がり中）、m のノルム）を見た。検査は seed 10–13 の 2 epoch（K-loop・K-resume は 2 課題 × 2 epoch）を走らせるが、私が読んだのは pass/fail と上に書いた検査の量だけで、扉の腕の 1 課題を通した精度は 1 つも見ていない。K-cost は未登録の seed 100–109 で、精度を計算しない。§7 の予測は登録 commit `fdb97c2b` のまま。
- **K-cost**（GPU を他に使う走が無い状態）: CH の束（R = 10）は単独で 0.125 秒/epoch（30 課題で 25 分）。主 4 腕を同時に走らせると各 0.49〜0.52 秒/epoch（各 99〜104 分）で、GPU は 1 本で計算に飽和し、同時に走らせても全体の処理量は変わらない。1 プロセスの峰は `nvidia-smi` で 1,160〜1,170 MiB（torch の確保の峰 513 MiB）。
- **同時本数**（§8 の式）: M_proc = 1,170 MiB。起動直前に M_free と、別 agent の `adamw_dose2_1011` の実測の使用量を `nvidia-smi` で取り、その spec §8 の予定（A1 は 3 本、A2 は短い 2 本）を足して M_keep を決める（`launch.sh <M_keep>` が式を計算し `logs/launch_record.txt` に残す）。現時点の空き 14,505 MiB で、別 agent の 3 本を各 1,200 MiB と見積もると M_keep = 3,600 + 2,048 = 5,648 MiB、n = ⌊(14,505 − 5,648) / (1.25 × 1,170)⌋ = 6 ≥ 5 なので、主 4 腕と Cr を同時に起動する。GPU は飽和しているので、所要時間は全体の仕事量で決まる: 私の主 4 腕で約 100 分、Cr（R = 1）で約 7 分（K-repro の 3 課題 46 秒から）、別 agent の A1 が約 57 分（その spec §8）。同時なら終わりまで約 2.5〜3 時間の見込み。

## 12. 追補 2 — 起動直前の資源の再計算の手順・Fable の予測・片付けの置き場所（2026-10-11 14:00 JST、本走の前。判定・窓・予測は変えていない）

親から: 本走は、Codex の監査（`specs/review_codex_cnn_center_relu_1011.md`）を受けて親が go を出すまで起動しない。監査の対象は `fdb97c2b`〜`3f635551`。この追補とそれに伴う `verdict.py` の採点の追加・`launch.sh` の書き直しは、その後の commit。エンジン `src/cnn_center_relu_1011.py` は検査 commit `0a5c0873` から変えていない（sha256 `a7517aa5…`）。

### 12.1 Fable の予測

§7.1 の Fable の列に、親から受け取った値をそのまま入れた（§7.2 の注記）。`verdict.py` は Claude と同じ規則で Fable の L2〜L7 も採点する。

### 12.2 RAM の実測（GPU のメモリより先に効く）

CH の束（R = 10）の 1 プロセスは、定常の VmRSS が 2,345 MiB、読み込み時の峰（VmHWM）が 3,395 MiB（C・CH・Cr は訓練集合のチャネル平均を float64 で一度計算する）。測った時点の MemAvailable は約 19,000 MiB。§8 の GPU のメモリの式だけだと 5 本同時になるが、RAM で数えると 5 本は 6 GiB の余裕を割る。GPU は 1 本で計算に飽和する（§11 の K-cost）ので、同時本数を減らしても全体の所要時間は変わらない。そこで §8 の「主 4 腕と Cr を同時に起動する」をやめ、**同時 2 本までの順番待ち**にする（2 本目は課題末の評価のあいだも GPU を埋めるため。残りの GPU と RAM は別 agent の `adamw_dose2_1011` に残す）。順番は CH → ref → C → H → Cr（主腕を先に回し、壊れ方があれば早く見つかるように）。

### 12.3 起動直前（と、順番待ちの各腕の起動直前）の再計算の手順

`analysis/cnn_center_relu_1011/launch.sh [MAX_CONC=2]` を `nohup setsid` で切り離して起動する。各腕を起動する直前に、その時点のマシンから次を計算し直し、全部満たすまで 60 秒ごとに待つ。決定はすべて `results/cnn_center_relu_1011/logs/launch_record.txt` に残る。

1. **本数**: この走のプロセス（`pgrep -f "[s]rc/cnn_center_relu_1011\.py run --arm <腕> --seeds"`。自己一致を避ける括弧の形）が MAX_CONC 本未満。同じ腕のプロセスが生きていれば起動しない（重複の防止）。`provenance.json` のある腕は完了として飛ばし、`state.pt` だけある腕は再開する。
2. **GPU**: `nvidia-smi --query-gpu=memory.free` − 2,048 MiB ≥ 1.25 × 1,170 MiB（K-cost の 1 本の峰、CUDA の context 込み）。別 agent のプロセスや ollama がメモリを使っていれば、その分はこの空きにすでに入っている。
3. **RAM**: MemAvailable − 6,144 MiB − 1,050 MiB（読み込み時の峰の上乗せ）≥ 1.2 × 2,350 MiB（定常の RSS）。
4. 記録: 起動ごとに、時刻・HEAD と作業木の汚れ・生きている本数・GPU の空き・MemAvailable・両方の余裕・マシン全体の proj_004_drift の python の本数（`pgrep -c -f '[p]roj_004_drift/\.venv/bin/python'`。operatingsis の python は数えない）・`nvidia-smi` のプロセスごとの使用量。
5. 起動の後は 90 秒待ってから次の予算を読む（読み込みの峰は 10 秒ほどで過ぎる）。
6. 起動の後の確認（私が手で行う）: 1〜2 分後に `pgrep` でプロセスが生きていること、`nvidia-smi` に載っていること、最初の課題の後にログに `task  1` の行（課題番号・秒数・腕名だけ）が出ること。
7. 見積もり: 私の仕事は GPU を単独で使って主 4 腕 約 100 分 ＋ Cr 約 7 分。別 agent の A1（その spec §8 で約 57 分）と重なれば、全体で約 2.7 時間（GPU は飽和しているので、時間は全体の仕事量で決まる）。

`DRY=1 bash launch.sh` は起動せずに判断だけを記録する（`launch_record_dry.txt`）。13:47 の試しでは、5 腕とも余裕あり（GPU 10,981 MiB・RAM 8,997 MiB）で起動の判断になった。

### 12.4 片付けの置き場所の訂正

§9 の `~/Projects/obsidian-research-data/cnn_center_relu_1011_1011/` は誤記で、CLAUDE.md §4 の `<実験名>_<MMDD>` は `cnn_center_relu_1011`（実験名 cnn_center_relu、cnn_center_1010 の退避先 `obsidian-research-data/cnn_center_1010/` と同じ形）。`analysis/cnn_center_relu_1011/backup.py` と `verdict.py` はこちらを使う。

## 13. 追補 3 — Codex の本走前監査を受けた修正（2026-10-11 14:40 JST、本走の前。扉の腕の結果は 1 つも無い）

監査: `specs/review_codex_cnn_center_relu_1011.md`（Codex、判定「直してから」、high 2・medium 4・low 2）。指摘ごとの扱い:

### 13.1 要約 DOORS の定義（high 1・2）— 主判定を差し替え、登録時の規則も残す

§5.6 の規則（登録 `fdb97c2b`）は、あり得るラベルの組で日本語の意味と矛盾した: H 単独が RESCUED（N = `NEED_H`）でも `DOORS_TRANSFER` になり、C か H が RESCUED で CH だけ FAIL でも `DOORS_FAIL` になる。主判定を次に替える（`verdict.py` の `doors_summary`）。

- **`DOORS_TRANSFER`**（MLP と同じ並びそのもの）: ref が FAIL ∧ **C が FAIL**（課題 1 は学ぶ: 課題 1 の online ≥ 0.5 が k* 本以上）∧ **H が FAIL** ∧ CH が RESCUED ∧ D_{H|C}（CH − C）が `HELPS`。MLP の表 2 は ref・C・H がすべて 0/10（FAIL）で CH が 10/10 だった。監査の最小の修正は「H が FAIL を足す」だったが、同じ理由（PARTIAL は MLP と同じ並びではない）で C も「RESCUED でない」から「FAIL」に揃えた。
- **`DOORS_FAIL`**: C・H・CH が**すべて** FAIL（中心化のどの形も崩壊を防がない）。
- **`DOORS_PARTIAL`**: それ以外（どれかの扉が救うか部分的に効くが、MLP の並びではない。どの腕が救ったかは N と腕のラベルで読む）。
- `NOT_REPRODUCED`・`INAPPLICABLE` は変えない。

登録時の規則は `DOORS_v1` として同じ verdict に出し、§7.1 の L6（Claude・Fable・Codex の確率）は `DOORS_v1` に対して採点する。新しい主判定には L6b を足す（Claude、14:40、本走の前）: **DOORS_TRANSFER .19 ／ DOORS_PARTIAL .64 ／ DOORS_FAIL .10 ／ 他 .07**（L1〜L4 の確率から: TRANSFER は C・H・CH の同時の条件で、H が FAIL の確率 .40 が効く）。Fable・Codex の L6b は、本走の前に受け取れば記入する。

§5.12 の読みも直す: `DOORS_TRANSFER` だけが「MLP と同じ並びが CNN でも出た」。CH が RESCUED なら、`DOORS_PARTIAL` でも「入力と中間層の中心化（CH）は崩壊を防ぐ」は言え、並びの違い（例: H だけで足りる = `NEED_H`）を併記する。`DOORS_FAIL` のときだけ「この形の中心化は ReLU の CNN の崩壊を防がない」と書く。

### 13.2 入力と provenance の関門（medium）

`verdict.py` に関門を足した。どれかが崩れれば (A) が成り立たず、ラベルは `INAPPLICABLE`（理由つき）。
- 表: 各腕の `arm` 列が腕名だけ、seed がちょうど 10–19、各 seed の課題がちょうど 1〜30 を 1 回ずつ、必要な列（online・memo・major_frac・mob・dead_frac、H の腕は h_upd と memo_acc_m0）がそろって有限。Cr は seed 10 だけで同じ検査。
- provenance: 各腕の `provenance.json` の arm・slots・課題数 30・epoch 400・lr 1e−3・β_H 0.01 が登録どおり、走った木が汚れていない、データの sha256 が全腕で同じ、**走った commit のエンジン（`src/cnn_center_relu_1011.py`）の sha256 が、検査を回した commit のエンジンと同じ**。`checks.json` が all_pass で、検査の木も汚れていない。
- C の扉 (C) は、静的な K-center に加えて、走そのものの `units.npz` の課題 0 の c1 の平均パッチのノルムが ref の 0.1 倍未満であることを seed ごとに確かめる。

### 13.3 そのほかの修正

- Cr（medium）: seed 10 が 30 課題を有効に完走し、4 つの窓が有限なときだけラベルを出す。そうでなければ `INAPPLICABLE`。`CONST_SHIFT_NO_EFFECT` の読みは「この直交 1 方向・1 seed で ref の側」に限る（一般の定数シフトについては言わない）。
- EVALOFF（medium）: `EVALOFF_NO_DEPENDENCE` を **`EVALOFF_NO_DETECTED_DEPENDENCE`** に改名する。`EVALOFF_DEPENDS` は「t30 の memo の関数が評価時の m に依存する」で、m が訓練の中で救命を生んだという因果の証拠ではない。
- µ 経路の内訳（medium）: §4 の自分・上流・交差の分解を `verdict.py` の `three_way_table` で出す（報告のみ）。K-mech が実際の形式の走（K-resume の CH、2 課題）で独立の再計算と 1e−12 で一致し、恒等式の残差 4e−16、変異 2/2 を検出することを確かめた。H の腕では m が t0 で 0、課題 1 に立ち上がるので、起点 t1 の分解も並べる（報告のみ。登録の B10・B11・B12 は起点 t0 のまま採点）。
- B2（low）: 登録どおり「9 本以上」で採点し、有効な seed が 10 本でなければ未採点にする。
- T_f: A(1) ≤ 0.1 の seed は適合率の分母が正でないので、T_f を未定義（NaN）にする（以前の実装は 1 を返していた）。
- 言葉（medium・low）: C は「その seed の 1,200 枚の特徴ごとの平均」を引く MLP の C の厳密な移植ではなく、`cnn_center_1010` の「訓練集合の色のチャネル平均」を引く CNN 用の C。H は MLP の H（特徴ごと）を CNN の畳み込みの対称性に合わせて**チャネル平均にした版**で、位置ごとの厳密な移植ではない（f1 の入力に位置ごとの DC が残る）。結果の主張は「CNN 用の C・H」に限る。§3.1 の「∂a/∂z = 1[z>0]」は、正確には「m の減算は微分を変えない。訓練の実装（`torch.clamp`）の微分は z > 0 と z = 0 で 1、z < 0 で 0。診断のゲートは z > 0」。
- `launch.sh` の安全係数の整数の丸めを切り上げにした（1.25 × 1,170 → 1,463 MiB）。

### 13.4 検査

K-verdict に、監査の矛盾 3 例（H が救って CH も救う／C が救って CH は落ちる／H が救って CH は落ちる）、全 972 の組合せ（腕のラベル 3⁴ × D_{H|C} 3 × C が課題 1 を学ぶか 2 × 適用条件 2）での不変条件（TRANSFER ⇒ C・H が FAIL で N = `NEED_CH`、FAIL ⇒ 全扉の腕が FAIL で N = `NONE_RESCUED`、単独の扉が救えば TRANSFER でも FAIL でもない、ほか）、表の関門（腕名の取り違え・seed 20 の混入・課題の重複・欠落・NaN）、T_f の未定義を足した。K-mech（µ 経路の表）と K-gate（provenance の関門: 登録どおりは通し、epoch・エンジン・汚れ・slots・データの食い違いを落とす）を足した。変異は新しく 10 本（TRANSFER が H を見ない・FAIL が C と H を見ない・Cr の valid を見ない・seed 集合と重複の検査を落とす・EVALOFF の名前・T_f・µ 経路の取り違え 2・エンジンの照合と汚れの検査を落とす）。全部の検査は、修正を commit した後、本走の直前に起動 commit の上でもう一度回す。

### 13.5 見たもの（開示）

報告の表のコードを試すため、K-loop・K-resume の検査の出力（H と CH、seed 10–11、2 課題 × 2 epoch）から µ 経路の表と層の表を出して見た（例: H の c1 の相対位置が −0.42 → −1.84、c2 の入力の平均のノルムが m の立ち上がりで 3.4 → 1.2〜1.8）。1 課題 30,000 更新の 0.5% の長さの走で、扉の腕の精度は見ていない。§7 の Claude の予測（L1〜L8・B1〜B12）は登録 commit のまま変えていない。L6b はこの追補で初めて置く。
