# cnn_center_feat_1011 — ReLU の CNN で、次層に渡る全特徴（チャネル × 位置）の中心化は喪失を防ぐか（P181 の追試）

状態: **事前登録**（この commit が登録。H′ の扉のコードはまだ書いていない。H′・CH′・対照の走は 1 本も無い）
作成: 2026-10-11 17:20 JST / 起草: Claude（Opus 5.5・サブエージェント）/ 依頼: 親（Fable）
run id: `cnn_center_feat_1011` / worktree `~/Projects/claude/wt/cnn_center_feat_1011` / branch `claude/cnn_center_feat_1011`（origin/main `b9ae5a26` から）
親: `specs/spec_cnn_center_relu_1011.md`（P181。箱・エンジン・C・H・判定の規則・検査の形をそのまま継ぐ）、`specs/review_codex_cnn_center_relu_1011.md`（P181 の監査。要約の矛盾・関門の指摘を先回りで反映する）
実装（予定・この登録の後に書く）: `src/cnn_center_feat_1011.py`（P181 のエンジン `src/cnn_center_relu_1011.py` を import して H′ を足す。既存の `src/*.py` は改変しない）、`analysis/cnn_center_feat_1011/{checks.py,verdict.py,figures.py,cost.py,launch.sh,backup.py}`

## 0. 一行と問い

P181（`cnn_center_relu_1011`）で、CNN 用の中心化 CH（C = 入力の色のチャネル平均、H = conv の pool 出力はチャネルごと・fc はユニットごとの EMA 平均）は ReLU の CNN の崩壊を遅らせた（T_f 中央値 3 → 10.5、10/10）が、30 課題の主窓では防がなかった（事後 FAIL）。事後の機構は、チャネルごとの中心化が f1 の入力（c2 の pool 出力 16 チャネル × 8×8 位置 = 1,024 特徴）の**位置ごとの平均**を残し、f1 がそれを通って µ 経路（交差項）で沈むことだった。MLP の扉 H（V12 §3.3・§10.2）は「次の層に渡る各スカラー特徴を中心化する」ので、CNN での厳密な対応は **(チャネル × 位置) ごと**の中心化（H′）になる。これは P181 で未検証。

- **Q1（主）**: CH′（C ＋ H′）は ReLU の CNN の機能の喪失を防ぐか（主窓 t21–30）。P181 の CH（チャネルだけ）は防がなかった。
- **Q2**: H′ 単独ではどうか（MLP の H 単独は救わなかった）。
- **Q3**: CH′ は CH より、H′ は H より、機能の低下を遅らせるか（T_f の符号検定）。
- **Q4（対照）**: 効くとすれば、それは「その位置の平均を消すこと」から来るか（位置をシャッフルした平均を引く対照 Hs）。
- **Q5（事後・記述）**: どの層が先に落ちるか。f1 が生き残るか、b 経路で c1・c2・f2 が落ちるか。

狙う文（成り立てば）: 「標準の CNN（ReLU）でも、各隠れ層の入力の全特徴の中心化（＋入力の色の中心化）は喪失を防ぐ。チャネル平均だけの中心化は遅らせるが防がない」。成り立たなければ「CNN では中心化は扉にならない（b 経路の沈降が残る）」と書く。

## 1. 既知（この spec の前に見たもの。すべて登録外の既存データ）

### 1.1 P181 の結果（`results/cnn_center_relu_1011/summary.md`・`verdict.json`・`tables/seeds_*.csv`、main `b9ae5a26`）

- 箱は §2 と同じ。主窓（t21–30 の online の seed 中央値）: ref 0.1124・C 0.1124・H 0.1124・CH 0.1154（範囲 0.103〜0.268、床の上 4 seed）。KEPT（≥ 0.5）はどの腕も 0/10。
- 登録判定: ref・C は `FAIL`（FAIL_FLOOR）。H・CH は §5.4 (C) の扉の記録が成り立たず `INAPPLICABLE`（§5.4 で穴を説明する）。事後（(C) を使わない）: 4 腕とも `FAIL`（CH は FAIL_ABOVE_FLOOR）、DOORS `DOORS_FAIL`。
- T_f（適合率 (A(t) − 0.1)/(A(1) − 0.1) が 1/2 を割る最初の t ≥ 2）の中央値: ref 3・C 4.5・H 2・CH 10.5。CH は ref より `LATER`（10/10、p 0.002）、H は `EARLIER`（9/10、p 0.021）。
- 最初に落ちる層（開いた割合の単位中央値が 1% を割る最初の課題）: ref は c2 が 9/10、C は c2 が 10/10、**H は f1 が 10/10**、**CH は c1 が 5/10・f1 が 5/10**（seed 12・13・16・17・19 で c1 が t4〜6 に先に落ちる。seed 13 は f1・f2 が 30 課題生き残り、c1・c2 が閉じて T_f 17）。
- t30 の死亡率（seed 中央値）: CH は c1 .28・c2 .41・f1 .99・f2 .95。H は f1・f2 が 1.00、c2 は 0。
- 機構（事後）: CH は c1・c2・f2 を b 経路で沈め（t0 → t30 の b の取り分 1.00・0.98・1.00）、f1 を µ 経路で沈める（取り分 0.00、交差項 −95〜−338）。f1 の入力の DC の比 `r_in_f1`（‖入力の平均‖ / rms 偏差）は CH の t1 で 0.162（seed 中央値）、ref 0.553・C 0.514・H 0.335。H は課題 1 から前活性の尺度が大きく育つ（t1 の logit の SD の seed 中央値: ref 14.7・C 15.0・**H 2,019・CH 626**）。
- 扉の記録の不足（P181 §15）: h_upd が 30,000 に届かない（seed, 課題, 層）が H で 7・CH で 66、すべて 95〜99% が死んだ f1・f2 で、その seed の T_f の後。死んだ単位の m は float32 の非正規数の不動点 7.0×10⁻⁴⁴（0.99·m が m に丸め戻る）で止まり、残る 1〜5 単位は全画像で開いていた。課題 1 の h_upd は H・CH の全 seed・全層で 30,000。

### 1.2 MLP の扉（V12 §3.3・表 2・付録 E。P181 spec §1.3 の要約）

RL-CIFAR × ReLU MLP の窓 t31–50: ref 0.113・C 0.113・H 0.113・CH 0.987。H 単独は第 1 層が生の入力の平均の経路で死ぬ。MLP の H は**各特徴（隠れユニット）の平均**を引く。

## 2. 箱（P181 と同一）

Kumar らの CNN（conv 5×5 3→16 pad 2 → ReLU → maxpool 2 → conv 5×5 16→16 pad 2 → ReLU → maxpool 2 → fc 1024→100 → ReLU → fc 100→100 → ReLU → fc 100→10、全層 bias あり、宿主の初期化）、RL-CIFAR（seed ごとに CIFAR-10 訓練集合の固定 1,200 枚、/255 の入力、課題ごとに乱数ラベル）、batch 16、400 epoch = 30,000 更新/課題、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8（moment は課題をまたいで保持）、WD なし、float32、**30 課題**、seed 10–19。活性化は全 4 か所で ReLU（`clamp(z, min=0)`、診断のゲートは `z > 0`）。エンジンは P181 の `src/cnn_center_relu_1011.py`（`sna_cnn_cause_1009` の束ねエンジン、1 プロセス = 1 腕 × R 本の束、1 epoch = 1 CUDA graph）。`--mem-gb 2.4`。

## 3. 扉と腕

### 3.1 扉

| 扉 | 定義 |
|---|---|
| **C** | P181 と同一（x ← x − μ、μ は CIFAR-10 訓練集合 5 万枚の色のチャネル平均。束を作るときに一度だけ引く）。 |
| **H**（P181） | conv（c1・c2）は pool の後の出力からチャネルごとの EMA 平均 m ∈ R¹⁶ を引く。fc（f1・f2）はユニットごとの m ∈ R¹⁰⁰。 |
| **H′**（この spec） | **conv（c1・c2）は pool の後の出力から (チャネル × 位置) ごとの EMA 平均を引く**: P_l = maxpool(ReLU(z_l)) − m_l、m_l ∈ R^{16×s×s}（c1 の後 s = 16 で 4,096 特徴、c2 の後 s = 8 で 1,024 特徴 = f1 の入力の全特徴）。bm = 画像（バッチの 16 枚）だけで平均した中心化前の pool 出力（位置では平均しない）。**fc（f1・f2）はユニットごと（H と同じ）**。m ← (1−β) m + β·bm、**β = 0.01**、初期値 0、課題の境界でリセットしない、演算は `m.mul_(1-β).add_(bm, alpha=β)`、m は勾配を通さない定数で Adam の対象でない、更新は各 Adam 更新の直後で bm はその更新の勾配を出した順伝播（更新前の重み）から取る、online・memo・診断・切替の CE はすべてその時点の m を使う（訓練と評価で同じ値）。ゲートは変えない。分散正規化も affine も無い。H との違いは conv の m の形だけ。 |
| **Hs**（対照） | m は H′ と同じ（(チャネル × 位置) ごとの EMA で、同じ規則で更新する）。次の層に渡すときだけ、**位置をシャッフルした m を引く**: P_l[c, p] = maxpool(ReLU(z_l))[c, p] − m_l[c, π_l(p)]。π_l は位置（c1 の後 256、c2 の後 64）の固定の置換で、**不動点の無い置換（derangement）**。全チャネル・全 run で共通。`pmnist_0905.stream("cnn_center_feat_1011_shuffle", s)`（s = 16・8）から `torch.randperm` を引き、不動点があれば同じ流れから引き直す（引いた回数を記録する）。fc は H′ と同じ。 |

H′ と Hs の関係（開示）: Hs が引く地図のチャネル平均は m のチャネル平均と同じ（置換は位置の平均を変えない）ので、Hs はチャネルの平均を H と同じく消し、位置ごとの平均は消さない（その位置の平均の代わりに別の位置の平均を引く）。定常で m が位置ごとの平均 µ[c, p] に等しいとき、Hs の後の位置ごとの残り δ[c, p] − δ[c, π(p)] は、δ と π(δ) が無相関なら H の残り δ の約 √2 倍になる。したがって Hs は「H′ と同じ装置・同じ EMA・同じ大きさの引き算で、位置の対応だけを壊したもの」で、H′ の効きが「その位置の平均を消すこと」から来るなら Hs は H の側に出る。

主張の範囲（開示。P181 の監査 4 の限定を継ぐ）: CH′ は「隠れ層の入力（c2・f1・f2・f3 が読む量）の全特徴の中心化 ＋ 入力の色のチャネル平均の中心化」である。入力層 c1 の入力は位置ごとには中心化しない（C は P181 と同じ。平均パッチのノルムは生の 0.013 倍以下だが、平均画像の位置ごとの模様は残る）。MLP の C（その seed の 1,200 枚の特徴ごとの平均）の厳密な移植ではない。結果の主張は「隠れ層の全特徴の中心化（H′）と色の中心化（C）」に限る。

### 3.2 腕

| 腕 | C | 扉 | seed | 役割 |
|---|---|---|---|---|
| **CHf**（= CH′） | ✓ | H′ | 10–19 | **主**。新しく走る |
| **Hf**（= H′） | − | H′ | 10–19 | H′ 単独。新しく走る |
| **Hs** | − | Hs | 10 | 変異対照（指示どおり H′ を基準にする）。新しく走る |
| CHs | ✓ | Hs | 10 | 変異対照（CH′ を基準にする）。**提案。親の go で回すかを決める**（回さなければラベルは `NOT_RUN`） |
| ref・C・H・CH | P181 | P181 | 10–19 | **P181 の走を再利用**（§6 の K-reuse で bit 照合。新しくは走らない） |

扉は課題 1 の最初の更新から入れる。各腕は自分の束（R = 10、対照は R = 1）で走るので、腕どうしの比較は seed 対応で、bit の比較ではない。再利用の正当性は、(i) このファイルのエンジンで ref・C・H・CH を走らせると P181 のエンジンと bit で同じ軌道になること、(ii) いまのマシンで P181 の走の課題 1 が bit で再現されること（per_task・t0・curves・課題 1 末のチェックポイント）、(iii) git の P181 の per_task.csv が `b9ae5a26` の blob と同じであること、の 3 つで確かめる（§6 K-same181・K-reuse、§5.4 (A)）。

## 4. 測る量

P181 §4 のすべて（`per_task.csv`・`t0.csv`・`curves.npy`・`units.npz`・チェックポイント t1・2・3・5・10・20・30、H 系の腕は `m_norm_l`・`m_mean_l`・`h_upd_l`・`memo_acc_m0`・`fit_ce_m0`）に、次を足す。

- **pool 出力の位置ごとの平均（扉の前後）**: c2 の入力（c1 の pool 出力、16 × 16×16）と f1 の入力（c2 の pool 出力、16 × 8×8）について、課題末の 1,200 枚で
  - µ_pre[c, p] = 中心化前の pool 出力の画像平均、µ_post[c, p] = 次の層が実際に受け取る量（µ_pre − 引いた地図）の画像平均。
  - 分解: µ[c, p] = µ̄_c ＋ δ[c, p]（µ̄_c は位置の平均、δ は位置ごとの残り）。
  - 列（l = c2・f1、その層の入力について）: `pdc_pre_l` = ‖δ_pre‖ / σ、`pdc_post_l` = ‖δ_post‖ / σ（**位置ごとの平均の大きさ、扉の前と後**）、`cdc_pre_l`・`cdc_post_l` = ‖µ̄ を位置へ広げたもの‖ / σ（チャネルの部分）、`dc_pre_l`・`dc_post_l` = ‖µ‖ / σ、`prms_l` = σ = √(E_i ‖P_i − µ_pre‖²)（画像の rms 偏差。定数の引き算で変わらない）。比は尺度によらない（H 系では尺度が大きく育つので、比で読む）。
  - `units.npz` に µ_pre・µ_post（c2 の入力 4,096、f1 の入力 1,024）と扉の m（conv は H′・Hs で 4,096・1,024）。
- 雑音の床（導出。判定には使わない）: m が定常な平均を追う場合、EMA の残りは 16 枚のバッチ平均の標本の揺れから来て、各特徴で σ_f·√(β/(2−β))/√16 = 0.0177 σ_f。したがって H′ の `pdc_post_f1` は定常で ≈ 0.018（＋課題の終わりの平均の漂流の遅れ）、H では `pdc_post` = `pdc_pre`（チャネル平均の引き算は δ を変えない）、Hs では定常で ≈ √2 × `pdc_pre`。
- P181 の再利用の腕（ref・C・H・CH）の同じ量は、P181 の `units.npz`（`obsidian-research-data/cnn_center_relu_1011/`、`backup_manifest.json` の sha256 で照合）の `m_in_f1`（= µ_post）から δ_post を、`r_in_f1` から σ を出す（H・CH では δ_pre = δ_post、ref・C では扉が無いので pre = post）。c2 の入力は P181 では µ が無いので出さない。

派生量（verdict が計算する）は P181 §4 と同じ: 主窓 W = t21–30 の online の平均（seed ごと）、副窓 t1–5・t6–10・t11–20、低下 = t1–5 − t26–30、適合率と **T_f**（q(t) < 1/2 になる最初の t ≥ 2。31 で打ち切り、A(1) ≤ 0.1 なら未定義）、**T_d,l**（`mob_l` < 0.01 になる最初の t ≥ 1）、最初に落ちる層、µ 経路と b 経路の分解（t0 → t の Δz̄ = Δ⟨W, m_in⟩ ＋ Δb、自分・上流・交差の内訳）。

## 5. 判定

### 5.1 seed ごと・5.2 腕のラベル（P181 §5.1・§5.2 と同一）

- **KEPT**: W ≥ 0.5。**AT_FLOOR**: W ≤ F_s + 3 s_s/√10（F_s・s_s はその seed の t21–30 の `major_frac` の平均と標本 SD）。
- 有効な seed（30 課題を完走し、online と memo が有限で、diverged が立たない）の数 n < 8 なら INAPPLICABLE。k*(n) は両側の正確な二項検定で p < 0.05 の最小の k（n = 10 で 9、9 で 8、8 で 8）。**RESCUED**: KEPT ≥ k*。**FAIL**: KEPT ≤ n − k*（AT_FLOOR ≥ k* なら `FAIL_FLOOR`、他は `FAIL_ABOVE_FLOOR`）。**PARTIAL**: それ以外。
- ラベルを出す腕: Hf・CHf（主）と、対比のための再利用の腕 ref・C・H・CH（P181 の per_task.csv から、この spec の規則（§5.4 の (C′) を含む）で計算し直す。P181 の登録ラベルは変えない）。

### 5.3 対応のある符号検定（主窓 W、seed 対応、同値を除く、両側の正確な二項検定。P181 §5.3 の規則）

D_{CHf|CH} = CHf − CH、D_{Hf|H} = Hf − H、D_{C|Hf} = CHf − Hf、D_CHf = CHf − ref、D_Hf = Hf − ref。p < 0.05 かつ逆向きの seed が 1 本以下なら `HELPS`（負の向きなら `HURTS`）、それ以外は `TIE`（「示せない」で、同等の意味ではない）。

### 5.4 適用条件

- **(A)** §6 の検査が all_pass（K-reuse を含む）。入力と provenance の関門（P181 §13.2 と同じ形）: 新しい腕の表は腕名・seed（主 10–19、対照 10）・課題 1〜30 を 1 回ずつ・必要な列が有限、provenance は arm・slots・課題数 30・epoch 400・lr 1e−3・β_H 0.01・扉の型（H′／Hs）・汚れていない木・全腕で同じデータの sha256・**走った commit のエンジン（`src/cnn_center_feat_1011.py` と、それが import する `src/cnn_center_relu_1011.py`）の sha256 が検査の commit と同じ**。再利用の腕は、作業木の `results/cnn_center_relu_1011/<腕>/per_task.csv` が `b9ae5a26` の blob と同じ（git の blob hash）で、P181 の `checks.json` が all_pass。主 2 腕の有効 seed が各 8 以上。
- **(B)（対比）** 再利用の ref と CH が、この spec の規則で `FAIL`。満たさなければ FEATURE_DOOR は `NOT_REPRODUCED`（腕のラベルと符号検定は報告する）。P181 の値（KEPT 0/10）から成り立つ見込みだが、規則として置く。
- **(C′) 扉の記録（P181 §5.4 (C) の穴を直したもの）**: Hf・CHf（と対比で使う H・CH）の全 seed・全層で、**課題 1 の `h_upd_l` ≥ 1**（EMA が課題 1 の中で更新された）。課題 2 以降の記録は条件にしない。C の扉は、CHf の全 seed で走の `units.npz` の課題 0 の c1 の平均パッチのノルムが Hf（生の入力・同じ seed）の 0.1 倍未満。満たさない腕は INAPPLICABLE。
  - **P181 の穴（明記）**: P181 の (C) は「全 seed・全課題・全層で h_upd = 30,000、またはその層の死亡率 = 1」を求めた。95〜99% が死んだ fc 層では、死んだ単位の m が float32 の非正規数の不動点 7.0×10⁻⁴⁴ に止まり（0.99·m が m に丸め戻る）、生き残った 1〜5 単位は全画像で開いた常時オンになって、層の m のどの要素も変わらない更新が出た。扉は仕様どおりだった（P181 の K-H-ema・K-loop）のに、免除が「全部死んだ層」だけだったため H・CH が INAPPLICABLE になった。死んだ後の固定点は扉の故障ではないので、今回は INAPPLICABLE の理由にしない。h_upd の全課題の記録と、満点に届かない（課題, 層）の一覧（死亡率・生きた単位の数・死んだ単位の |m| の最大・T_f との前後）は報告する。

### 5.5 要約 FEATURE_DOOR（主判定、Q1）

順に評価する。
1. (A) か (C′) が Hf・CHf で崩れれば `INAPPLICABLE`。
2. (B) が崩れれば `NOT_REPRODUCED`。
3. CHf が RESCUED（P181 の CH は (B) で FAIL）→ **`FEATURE_CENTERING_SUFFICES`**。
4. CHf が FAIL かつ Hf が FAIL → **`CENTERING_NOT_A_DOOR_IN_CNN`**。
5. それ以外 → **`FEATURE_CENTERING_PARTIAL`**（CHf が PARTIAL、または CHf は FAIL だが Hf が RESCUED か PARTIAL）。

依頼の規則（「CH′ FAIL → `CENTERING_NOT_A_DOOR_IN_CNN`」）に、4 で **Hf も FAIL** を足した（開示）。P181 の監査の high 2（単独の扉が救うのに組合せの FAIL を「中心化は防がない」と要約する矛盾）を先回りで塞ぐため。CHf が FAIL で Hf が救うなら、「全特徴の中心化は扉でない」は偽なので PARTIAL に送り、どの腕が救ったかを本文に書く。

### 5.6 H′ 単独（Q2）

M_Hf = Hf の腕のラベル。MLP（H 単独は救わない）との対応: FAIL → `HF_LIKE_MLP`、RESCUED → `HF_DIFFERS_FROM_MLP`、PARTIAL → `HF_PARTIAL`。

### 5.7 機能の低下の時刻（Q3、登録）

T_f の腕ごとの中央値（報告）と、seed 対応の符号検定（§5.3 の規則、同値を除く。T_f = 31 は打ち切りで、両方 31 なら同値）: **CHf 対 CH、CHf 対 ref、Hf 対 H、Hf 対 ref**。後ろの腕より遅ければ `LATER`（§5.3 の `HELPS` に当たる向き）、早ければ `EARLIER`、他は `NO_TIMING_DIFF`。T_f が未定義（A(1) ≤ 0.1）の seed はその対から除く。

### 5.8 フラグ（ラベルを変えない。P181 §5.9 と同じ規則）

- `RESIDUAL_LOSS`（CHf・Hf）: 低下（t1–5 − t26–30）の符号検定で正が `HELPS`。
- `T1_LOWER`（CHf・Hf）: 課題 1 の online が ref より低い向きに `HURTS`。
- `EVALOFF`（CHf・Hf）: t30 の memo_acc − memo_acc_m0 の符号検定で正なら `EVALOFF_DEPENDS`、他は `EVALOFF_NO_DETECTED_DEPENDENCE`（t30 の memo の関数が評価時の m に依存するという意味で、訓練中の因果の証拠ではない）。

### 5.9 変異対照のラベル（Q4）

- **Hs**（seed 10 の 1 本）: 窓 t1–5・t6–10・t11–20・t21–30 のうち、Hf と H（P181）の 10 seed の範囲 [min, max] が重ならない窓を「分ける窓」とする。分ける窓のすべてで Hs が H の範囲に入れば **`SHUFFLE_ACTS_LIKE_H`**（H′ の効きは位置の対応から来る。装置が「その位置の平均」を消していることの結果の側の確認）、すべてで Hf の範囲に入れば `SHUFFLE_ACTS_LIKE_HF`（位置の対応は効きに要らない）、分ける窓が無ければ `SHUF_UNINFORMATIVE`（Hf が H と区別できない）、それ以外は `SHUF_UNRESOLVED`。Hs が 30 課題を有効に完走しなければ `INAPPLICABLE`。
- **CHs**（回すとき）: 同じ規則を CHf と CH（P181）で。`SHUFFLE_ACTS_LIKE_CH`／`SHUFFLE_ACTS_LIKE_CHF`／`SHUF_UNINFORMATIVE`／`SHUF_UNRESOLVED`、回さなければ `NOT_RUN`。
- 読みの限定: 1 seed・1 つの置換。`ACTS_LIKE_H` は「この置換の Hs が H の側を通った」で、位置の対応の一般の必要性ではない。

### 5.10 報告のみ（判定に使わない）

- **装置の読み出し**: 課題末ごとの `pdc_pre_f1`・`pdc_post_f1`・その比（扉が位置ごとの平均を消した割合）、c2 の入力の同じ量。Hf・CHf・Hs（新しい走）と H・CH・ref・C（P181 の `units.npz` から）を並べる。
- 層ごとの位置・幅・開いた割合・死亡率・r_in の課題列、T_d,l、最初に落ちる層の分布（Q5: f1 が生き残るか、どの層が先に落ちるか）。
- µ 経路と b 経路の分解、µ 経路の自分・上流・交差（P181 §4 と同じ）。
- t1 の logit の SD と各層の幅（尺度の育ち）。
- h_upd の全課題の記録と満点に届かない（課題, 層）の一覧（§5.4）。
- memo_acc と memo_acc_m0 の課題列。

### 5.11 読み方（登録）

- **`FEATURE_CENTERING_SUFFICES`** → 「標準の CNN（ReLU）でも、隠れ層の入力の全特徴（pool 出力のチャネル × 位置、fc のユニット）の中心化と入力の色の中心化は、30 課題の主窓で喪失を防ぐ（CHf 主窓 x、10 seed 中 k 本が 0.5 以上）。チャネル平均だけの中心化（P181 の CH）は遅らせるが防がない」と書ける。D_{CHf|CH} と T_f の符号検定、Hs のラベル、`RESIDUAL_LOSS` を併記する。Hs が `SHUFFLE_ACTS_LIKE_H` でなければ「位置の対応が効く」とは書かない。
- **`FEATURE_CENTERING_PARTIAL`** → どれが崩れたか（CHf が PARTIAL か、Hf だけが救うか）を書く。CHf が PARTIAL なら「一部の seed だけ防ぐ」。
- **`CENTERING_NOT_A_DOOR_IN_CNN`** → 「この箱では、全特徴の中心化でも ReLU の CNN の喪失は防げない」。§5.10 の分解で、残った沈降が b 経路か（どの層か）、別の経路かを書く。T_f の符号検定で CHf が CH より `LATER` なら「位置の平均の経路を塞ぐと遅れるが防がない」と書く。
- 「防ぐ」は「この箱の 30 課題の主窓で、10 seed 中 9 本以上が online 0.5 以上」に限る。すべて「Kumar らの CNN・RL-CIFAR・ReLU・seed 10–19・30 課題・扉は課題 1 から・β = 0.01」に限る。

## 6. 検査（実装の後、本走の前。許容は導出から決め、値を見て決めない）

`analysis/cnn_center_feat_1011/checks.py` → `results/cnn_center_feat_1011/checks.json`。各検査は本物のコードで pass し、列挙した変異（対象ファイルの文字列の 1 回置換、P181 の `load`）で fail しなければならない。P181 の検査のうち H′ の装置に関わるものは H′ 用に書き直し、エンジンが同じ部分（ReLU・C・宿主との一致）は K-same181 と K-reuse で P181 の検査に帰着させる。

| 検査 | 要求 | 変異（検出すべき） |
|---|---|---|
| **K-same181** | このファイルのエンジンの ref・C・H・CH の束（R = 3）が P181 のエンジンの同じ腕の束と、2 epoch（eager）で全状態・V・m・h_upd・evaluate・switch_eval・診断の共通列まで bit 一致 | H の conv の m を位置ごとにする／ReLU を leaky に |
| **K-reuse** | このファイルの CLI で ref・C・H・CH を R = 10・seed 10–19・1 課題（CUDA graph、本走と同じ）回すと、P181 の per_task.csv の課題 1 の行（共通の全列、文字列）・t0.csv（共通の全列）・curves.npy の課題 1（bit）と一致し、課題 1 末のチェックポイント（全 seed の重み・Adam の m と v・扉の m と V）が P181 の退避先の `<腕>_seed<s>_t01.pt`（P181 の `backup_manifest.json` の sha256 で照合したもの）と bit 一致。git の P181 の per_task.csv が `b9ae5a26` の blob と同じ | — |
| **K-Hf-pos** | H′ が (チャネル × 位置) ごと: (a) bm が独立に計算した「maxpool(ReLU(z)) の画像だけの平均」と bit 一致し、形が (R, 16·s·s)。(b) forward が次の層に渡す量（c2 と f1 の入力）が、独立に書いた P − m（m を (R, 16, s, s) に並べたもの）と bit 一致。(c) z の pool 窓を位置で並べ替えると、H′ の bm は同じ並べ替えを受け（bit）値が変わり、P181 の H の bm は変わらない（float64 で 1e−6 の相対差以内。和の順の丸め）。(d) 3 更新後の m がチャネルの中で位置によって違う | H′ がチャネル平均を引く（H に戻す）／引く地図の y と x を転置／bm を位置でも平均 |
| **K-Hf-ema** | 捕捉した更新で m の更新が「中心化前の pool 出力（conv は位置ごと）と fc 出力のバッチ平均を β = 0.01 で追う」と bit 一致。bm を出した z が更新前の重みの順伝播の z と bit 一致。課題の境界で m が変わらない。h_upd が更新数を数える | 中心化後を追う／β = 0.1／pool の前の平均を追う／境界で m をリセット／Adam の後の重みで bm を取る |
| **K-Hf-grad** | m は勾配を持たず、全勾配が「m を定数とした独立の再計算」と bit 一致 | 勾配を通す（値は同じで勾配だけバッチ平均を通る） |
| **K-Hf-eval** | 課題末の fit CE と切替の CE がその時点の m での再計算と bit 一致し、m = 0 では一致しない | 評価で m を使わない |
| **K-Hf-first** | Hf の最初の更新が ref の最初の更新と bit 一致（CHf と C も）、直後の m は非零、3 更新後は違う | m の初期値が非零 |
| **K-evaloff** | m = 0 の評価は読むだけ（Hf・状態・m・乱数の流れ・2 epoch の軌道が不変） | m = 0 の評価が m を 0 にする |
| **K-shuf** | Hs: π が流れからの式どおりの derangement（不動点 0、引き直しの回数を記録）。次の層に渡す量が P − m[:, :, π] と bit 一致。m の更新は H′ と同じ（同じ入力で H′ の束と m が bit 一致）。引いた地図のチャネル平均が m のチャネル平均と float64 で一致（1e−6 の相対差以内） | π を当てない（Hs = H′）／π を EMA の側に当てる／π に不動点を許す |
| **K-host** | 1 step の損失と全勾配が、独立に書いた扉つきの 1 run の宿主の形の順伝播（m を定数で与える。Hf・CHf・Hs）と一致（損失の相対差 1e−5・勾配の相対差 1e−4。P181 の K-host と同じ許容） | φ の中で中心化（φ(z − m)）／conv の m を引かない |
| **K-pdc** | 課題末の診断の新しい列（`pdc_*`・`cdc_*`・`dc_*`・`prms_*`）が、捕捉した網での独立の float64 の再計算と相対 1e−9 で一致（float32 の入力からの float64 の集計どうし）。µ_post = µ_pre − 引いた地図（float64 で 1e−6）。H では `pdc_post` = `pdc_pre`（同じ許容）。m を全 1,200 枚の正確な平均にした H′ では `pdc_post_f1` が float32 の引き算の丸めの上界 2⁻²³·max\|P\|·√1024/σ 以内 | pre と post の取り違え／位置の部分からチャネル平均を引かない |
| **K-ident** | z̄_j = ⟨W_j, m_in⟩ + b_j が TF32 と float32 の丸めの上界内（Hf・CHf・Hs、全 4 層。P181 と同じ上界） | 平均パッチの並びを転置／m_in を中心化前の入力から取る |
| **K-graph** | Hf・CHf・Hs の束で CUDA graph と eager が 2 epoch で m・h_upd を含めて bit 一致 | — |
| **K-indep** | CHf の束の run 1 の重み・入力・m を乱すと run 0・2 は bit で不変、run 1 は変わる | — |
| **K-loop** | `run`（eager）が課題の手での再生と bit 一致、m が課題の境界を越えて引き継がれる、h_upd = 150/課題（Hf） | run の中で境界に m をリセット |
| **K-resume** | CHf の 2 課題（2 epoch/課題）を通しで回すのと 1 課題で止めて再開するのとで per_task・t0・curves・units・状態が一致 | — |
| **K-verdict** | 合成の per_task で設計どおりのラベル（腕の 3 値と FAIL の内訳・符号検定・FEATURE_DOOR の 3 値と INAPPLICABLE・NOT_REPRODUCED・M_Hf・T_f の符号検定・フラグ・Hs／CHs の 4 値と INAPPLICABLE・NOT_RUN・(C′) の課題 1 だけの規則）。FEATURE_DOOR を全組合せ（Hf・CHf・ref・CH の 3⁴ × 条件 2 × 対比 2）で不変条件（SUFFICES ⇒ CHf RESCUED、NOT_A_DOOR ⇒ CHf と Hf が FAIL、Hf が RESCUED なら NOT_A_DOOR でない、ほか）。表の関門（腕名・seed・課題の重複と欠落・NaN） | k* の取り違え／0.5 の線を > に／床の √10 を √30 に／主窓の課題範囲／NOT_A_DOOR が Hf を見ない／(C′) が全課題を見る／(C′) が課題 1 を見ない／Hs の比較の腕を取り違え（Hf と CH）／T_f の向きの取り違え／seed 集合の検査を落とす |
| **K-mech** | 報告の表（µ 経路の自分・上流・交差、起点 t0・t1 の分解、位置ごとの DC の表）が独立の再計算と一致（実際の形式の走、K-resume の CHf） | 自分と上流の取り違え／起点 t1 の分解が t0 から引く |
| **K-gate** | provenance の関門: 登録どおりなら通し、epoch・エンジン（新旧 2 ファイル）・汚れ・slots・データ・扉の型の食い違いと、再利用の per_task.csv の blob の食い違いを落とす | エンジンの sha256 を照合しない／汚れた木の走を通す／再利用の blob を照合しない |
| **K-cost** | CHf の束（R = 10）の 1 epoch の秒数（単独と Hf との同時）、GPU メモリの峰（nvidia-smi）、VmRSS（seed 100–109、精度は読まない） | — |

**許容の導出**: bit 一致の検査は許容 0。K-host の 1e−5・1e−4 は P181・cnn_center_1010 と同じ（束ねの bmm と宿主の mm の float32 の丸めの差、1 step で 1e−7 程度）。K-Hf-pos (c)・K-shuf の 1e−6 は float32 の和の順の違い（K ≤ 1,200 項、γ_K ≈ 7×10⁻⁵ の相対上界のうち実際の典型の √K·2⁻²⁴ ≈ 2×10⁻⁶ の桁）から置き、float64 で集計して比べる。K-pdc の 1e−9 は float32 の同じ値の float64 での和の順の違い（N·2⁻⁵³ ≈ 10⁻¹³ に余裕）。K-ident は P181 と同じ上界。

## 7. 事前予測（H′ のコードを書く前・H′ の走を 1 本も回す前。この commit で固定する）

### 7.1 Claude（Opus 5.5、2026-10-11 17:20 JST、この spec の起草者）

読み: P181 の CH の失敗には 2 つの道があった。(1) f1 が位置ごとの DC を通って µ 経路で沈む（seed 10・11・14・15・18 で f1 が先に落ちる）。(2) c1 が b 経路で沈む（seed 12・13・16・17・19 で c1 が t4〜6 に先に落ちる。seed 13 は f1・f2 が生きたまま T_f 17）。H′ は (1) を塞ぐが (2) には直接は触らない。ただし (2) の押しが、H 系で課題 1 から育つ前活性の尺度（t1 の logit の SD: CH 626 対 ref 15）に由来するなら、H′ で尺度の育ちが止まれば (2) も弱まる。尺度の育ちが位置ごとの DC の正の帰還（c2 の DC が f1 の重みに乗り、その重みが c2 を育てる）から来るのか、全層が生きたまま Adam の一定の歩幅で重みが育つこと（W 増大）から来るのかは分からない。後者なら CHf でも尺度は育ち、b 経路の沈降は残る。H′ 単独は c1 が生の入力の µ 経路で沈む（P181 の H で c1 の Δz̄ −68、MLP の H 単独は第 1 層で死ぬ）ので、救わない側を本命にする。

略号: 他 = INAPPLICABLE・NOT_REPRODUCED（・NOT_RUN）をまとめた桶。

| # | 項目 | Claude | Fable | Codex |
|---|---|---|---|---|
| L1 | **CHf のラベル（主）** | RESCUED .30 ／ PARTIAL .30 ／ FAIL .37 ／ 他 .03 | RESCUED .50 ／ PARTIAL .25 ／ FAIL .25 | （監査の後に転記） |
| L2 | Hf のラベル（M_Hf） | RESCUED .10 ／ PARTIAL .20 ／ FAIL .67 ／ 他 .03 | RESCUED .30 ／ PARTIAL .30 ／ FAIL .40 | |
| L3 | **FEATURE_DOOR** | SUFFICES .30 ／ PARTIAL .33 ／ NOT_A_DOOR .34 ／ 他 .03 | SUFFICES .45 ／ PARTIAL .25 ／ NOT_A_DOOR .30 | |
| L4 | T_f: CHf 対 CH | LATER .62 ／ NO_TIMING_DIFF .35 ／ EARLIER .03 | | |
| L5 | T_f: CHf 対 ref | LATER .85 ／ NO_TIMING_DIFF .14 ／ EARLIER .01 | | |
| L6 | T_f: Hf 対 H | LATER .62 ／ NO_TIMING_DIFF .38 ／ EARLIER .00 | | |
| L7 | Hs（§5.9） | ACTS_LIKE_H .45 ／ UNINFORMATIVE .30 ／ UNRESOLVED .20 ／ ACTS_LIKE_HF .05 | | |
| L8 | CHs（§5.9、回したときだけ採点） | ACTS_LIKE_CH .40 ／ UNINFORMATIVE .30 ／ UNRESOLVED .25 ／ ACTS_LIKE_CHF .05 | | |

Fable の列は親の依頼文の値をそのまま写した（2026-10-11、H′ の走の前。L3 の NOT_A_DOOR は §5.5 の 4 の定義（Hf も FAIL）で採点する。親の値は「CH′ FAIL → NOT_A_DOOR」の定義に対して書かれた）。

| # | 主張（二値） | Claude |
|---|---|---|
| B1 | CHf の t30 の f1 の死亡率（seed 中央値）< 0.5（f1 が生き残る） | .45 |
| B2 | CHf の「最初に落ちる層」に c1 が入る seed が 5 本以上（有効 seed が 10 本のときだけ採点） | .35 |
| B3 | CHf の t1 の logit の SD（seed 中央値）が ref（P181）の t1 の 10 倍超（尺度の育ちが残る） | .50 |
| B4 | CHf の t1 の `pdc_post_f1`（seed 中央値）< 0.05（装置が f1 の入力の位置ごとの平均を消す。雑音の床 0.018） | .85 |
| B5 | CHf の主窓 online（seed 中央値）≥ 0.90 | .15 |
| B6 | CHf に `RESIDUAL_LOSS` が立つ | .85 |
| B7 | CHf の t0 → t30 の b 経路の取り分（単位中央値の seed 中央値）が 4 層すべてで 0.5 超 | .70 |
| B8 | Hf の「最初に落ちる層」に c1 が入る seed が 9 本以上（有効 seed が 10 本のときだけ採点） | .25 |
| B9 | CHf が `EVALOFF_DEPENDS` | .85 |
| B10 | CHf に `T1_LOWER` が立つ | .10 |

### 7.2 採点の規則（P181 §7.2 と同じ）

ラベルの項目（L1〜L8）は、実現したラベルに付けた確率、最頻ラベルの的中、多クラスの Brier 得点 Σ_k (p_k − 1[k = 実現])²（名前の無いラベルは「他」の桶、桶の無い列ではそのラベルの確率 0）。二値（B1〜B10）は真偽と binary Brier。条件つきの項目は条件が成り立たなければ「未採点（理由）」。L8 は CHs を回したときだけ採点する。Codex の列は監査の後に、Codex が書いた値をそのまま転記する（何を見た後に書いたかを併記する）。走る前に記入されたものだけを採点する。

## 8. 費用と実行

- 1 本の費用: P181 の K-cost で CH の束（R = 10）は単独 0.125 秒/epoch（30 課題で 25 分）、GPU は 1 本で計算に飽和。P181 の本走は別 agent（`adamw_dose2_1011`）と GPU を分けて 1 課題 100〜195 秒だった。H′ の m は conv で 4,096 + 1,024 要素/run 増えるだけで、計算はほぼ同じ。K-cost で置き換える。
- **同時本数**（P181 §12 の手順）: GPU 1 本で飽和するので、同時 2 本まで（2 本目は課題末の評価の間も GPU を埋めるため）。各腕の起動直前に、GPU の空き − 2,048 MiB ≥ 1.25 × M_proc（K-cost の nvidia-smi の峰）、MemAvailable − 6,144 MiB − （読み込み時の峰の上乗せ）≥ 1.2 × 定常の RSS（K-cost で測る）を満たすまで 60 秒ごとに待つ。順番は CHf → Hf → Hs（→ CHs）。`launch.sh` が判断を `results/cnn_center_feat_1011/logs/launch_record.txt` に残す。起動の前に `nvidia-smi` と、マシン全体の proj_004_drift の python の本数（`pgrep -c -f '[p]roj_004_drift/\.venv/bin/python'`）を見る。
- 起動は、実装の commit・検査の all_pass・HEAD の push・親（Fable）の go（Codex の監査の後）・台帳 `PREDICTIONS.md` への登録（flock、続番）の後。`nohup setsid` で切り離し、ログは `results/cnn_center_feat_1011/logs/<arm>.log`（ログには精度を印字しない。課題番号・秒数・腕名だけ）。
- 走の途中で効果量を見て設計を変えない。監視は完了課題の数・失敗・ログの末尾の課題番号と秒数だけ。
- python は `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- 走: `results/cnn_center_feat_1011/<arm>/`（`per_task.csv`・`t0.csv`・`curves.npy`・`provenance.json` は git に入れる。`units.npz`・`ckpt/`・`state.pt` は入れない）。
- 判定: `analysis/cnn_center_feat_1011/verdict.py` → `results/cnn_center_feat_1011/{verdict.json,summary.md,tables/*.csv}`、図 `figures.py` → `results/cnn_center_feat_1011/fig_*.png`。
- 片付け: git 外のファイル（`units.npz`・`ckpt/`・`state.pt`・ログ・検査の作業場所）を `~/Projects/obsidian-research-data/cnn_center_feat_1011/` へ移し、`results/cnn_center_feat_1011/backup_manifest.json`（source・backup・bytes・sha256）を commit する。symlink は辿らない。verdict は退避先からも読めるようにする。main に merge・push してから worktree とブランチを消す。
- vault: `可塑性喪失/測定/` に結果ノート、`ハブ/走の履歴.md` と `ハブ/論点マップ.md` に 1 行ずつ（`git pull --ff-only` の後）。V12 は触らない。

## 10. 開示

- §1 の数値は全部この spec の前に読んだ。P181 の `summary.md`・`verdict.json`・spec・vault の結果ノートに加え、`tables/seeds_CH.csv`・`seeds_H.csv`（seed ごとの T_d・最初に落ちる層・t30 の開いた割合と死亡率）と、P181 の per_task.csv の課題 1 の `logit_sd`・`r_in_f1`・`h_upd` の seed 中央値を見てから予測を書いた。
- 雑音の床 0.0177 は β と batch から計算した（データは見ていない）。
- §5.5 の 4 に「Hf も FAIL」を足したのは依頼からの変更（理由は §5.5）。CHs は依頼に無い提案（§3.2）。
- Hs の置換の作り方（同じ流れから derangement を引き直す）は、置換を 1 本も引く前に決めた。

## 11. 追補 1 — 実装・検査・費用（2026-10-11 18:05 JST、本走の前。判定・窓・予測は変えていない）

- 実装 commit `d8696db9`（17:40）。その上で `analysis/cnn_center_feat_1011/checks.py` の全実行が **all_pass**（20 項目、変異 41/41 を検出、例外で落ちた変異は 1（K-Hf-pos の「H に戻す」は m の形の食い違いで落ちる。検出として数える。P181 の `run_check` と同じ扱い））、`results/cnn_center_feat_1011/checks.json`（`git_dirty_code` = false、17:40〜18:03）。
  - **K-reuse**: このエンジンの CLI で ref・C・H・CH を R = 10・seed 10–19・1 課題（CUDA graph）回すと、P181 の per_task.csv の課題 1 の 10 行（共通の 91 列（ref・C）・105 列（H・CH）、文字列）・t0.csv（83・97 列）・curves.npy の課題 1（bit）・**全 40 本の課題 1 末のチェックポイント**（重み・Adam の m と v・扉の m・V・h_upd・tc）が bit 一致した。チェックポイントは P181 の退避先のもので、P181 の `backup_manifest.json` の sha256 と照合した。git の P181 の per_task.csv 4 本は `b9ae5a26` の blob。所要 883 秒。
  - **K-same181**: このエンジンの ref・C・H・CH の束（R = 3）が P181 のエンジンと 2 epoch で全状態・V・m・h_upd・evaluate・switch_eval・診断の共通の列と配列まで bit 一致。
  - **K-Hf-pos**: H′ の bm は (R, 4,096)・(R, 1,024) で独立の計算と bit 一致、次の層の入力は P − m（(R, 16, s, s) に並べた m）と bit 一致、z の pool 窓を並べ替えると H′ の bm は同じ並べ替えを bit で受け、P181 の H の bm の変化は相対 5.6×10⁻⁸ 以下（和の順の丸め）。
  - **K-pdc**: 新しい列は独立の float64 の再計算と相対 3.5×10⁻¹⁵ 以下で一致。m を 1,200 枚の正確な平均にした H′ で `pdc_post` は丸めの上界の 0.024 倍。
  - **K-host**: 1 step の損失の相対差 0、勾配の相対差 ≤ 1.7×10⁻⁷（Hf・CHf・Hs）。**K-ident**: 恒等式の残差は上界の最大 0.058 倍。
- **実装時の明確化**（設計は変えていない）:
  1. `FeatDoors.update` は P181 の `ReLUDoors.update` の写し（変異をこのファイルに当てるため）。チャネル平均の腕で P181 と bit 同じことは K-same181・K-reuse で確かめた。
  2. K-pdc の上界は導出をやり直して **3·2⁻²⁴·max|P|·√F/σ** にした（§6 の 2⁻²³ = 2·2⁻²⁴ は引き算の丸め 2·2⁻²⁴·max|P| だけで、m 自身の float32 の丸め 2⁻²⁴·max|P| を落としていた）。H の「前 = 後」も同じ上界で比べた（§6 の「同じ許容」の 1e−6 は µ_post の float64 の比較の許容で、δ は float32 の引き算の値から来るので丸めの上界のほうが正しい）。値を見て決めたものではない。
  3. K-mech の位置ごとの DC の表は、units.npz の float32 の µ からの再計算と 2·2⁻²⁴·‖µ‖/σ ＋ 1e−9（保存の丸めと 10 桁の CSV）で比べた。
  4. K-shuf は登録の辺（16・8、どちらも 1 回目で derangement）に加えて辺 2・3・4（1 回目に不動点が出やすい。引き直しは 2・1・2 回）でも関数を独立の実装と比べ、「不動点を許す」変異を検出できるようにした。K-Hf-ema・K-Hf-first は Hs も含む（EMA の側は置換しない）。
  5. K-gate に扉の型（`h_mode`）と再利用の blob の照合を足した。K-verdict は 19 場面・46 の期待値・FEATURE_DOOR の 162 組合せ・変異 11。
  6. `launch.sh` の予算の定数を K-cost の実測で埋めた（M_PROC 1,180・RSS 2,310・RSS_LOAD 1,040 MiB、切り上げ）。エンジン（`src/`）は検査の commit から変えない。provenance の関門は走った commit と検査の commit のエンジン 2 ファイルの sha256 を比べる。
- **見たもの（開示）**: 実装中の smoke test（seed 10–11 で 3 更新、扉の m の形と、立ち上がり中の `pdc_*` の値）と、検査の内部の量（seed 10–13、2 epoch または 2 課題 × 2 epoch）。K-reuse は P181 の腕の課題 1 を再現しただけで、私が読んだのは一致の真偽と列数。K-cost は未登録の seed 100–109 で精度を計算しない。H′ の腕の 1 課題を通した精度は 1 つも見ていない。§7 の予測は登録 commit `b1b28854` のまま。
- **K-cost**（別 agent の `adamw_dose2_1011` の 3 本が GPU を使っている状態。GPU は 100% で飽和）: CHf の束（R = 10）は単独 0.488 秒/epoch（30 課題で 98 分）、CHf と Hf の同時で各 0.642 秒/epoch（各 128 分、並行）。課題末の診断 5〜6 秒。1 プロセスの峰は nvidia-smi で 1,174 MiB、定常の VmRSS 2,294〜2,305 MiB、読み込み時の VmHWM 3,333 MiB。
- **同時本数と見込み**: §8 の式で、起動直前に GPU の空き 11,766 MiB・MemAvailable 14,671 MiB（GPU の余裕 8,243、RAM の余裕 4,715 MiB）。主 2 腕（CHf・Hf）を同時に起動し、どちらかが終わったら Hs（親が認めれば続けて CHs）。別 agent の 3 本が走り続ければ主 2 腕で約 2 時間 10 分、対照で 20〜30 分。別 agent が終われば短くなる。

## 12. 追補 2 — 親の決定と、起動直前の資源の再計算の手順・順番（2026-10-11 18:20 JST、本走の前。判定・窓・予測は変えていない）

### 12.1 親（Fable）の決定（18:15 ごろ受け取った）

- **CHs（C ＋ シャッフル、seed 10）を回す**（§3.2 の提案を採用）。L8 は採点の対象になる。
- **§5.5 の 4 に「Hf も FAIL」を足した変更を採用**する。
- 本走は Codex の短い監査（`specs/review_codex_cnn_center_feat_1011.md`）の後、親の go を受けてから起動する。

### 12.2 順番

**CHf・Hf（同時）→ Hs → CHs**。主 2 腕を先に同時に回し、どちらかが終わって枠が空いたら Hs、次に CHs を入れる（`launch.sh` は登録の順に、空いた枠へ 1 本ずつ入れる）。`WITH_CHS=1 bash analysis/cnn_center_feat_1011/launch.sh 2` で CHs を待ち行列の最後に足す（監査中のファイルを変えないため、`launch.sh` の既定の待ち行列は書き換えない）。

### 12.3 起動直前（と、各腕の起動直前）の再計算の手順（P181 §12.3 と同じ形）

`nohup setsid env WITH_CHS=1 bash analysis/cnn_center_feat_1011/launch.sh 2 > results/cnn_center_feat_1011/logs/launcher.log 2>&1 < /dev/null &` で切り離して起動する。各腕を起動する直前に、その時点のマシンから次を計算し直し、全部満たすまで 60 秒ごとに待つ。決定はすべて `results/cnn_center_feat_1011/logs/launch_record.txt` に残る。

1. **本数**: この走のプロセス（`pgrep -f "[s]rc/cnn_center_feat_1011\.py run --arm <腕> --seeds"`。自己一致を避ける括弧の形。`--arm Hf` は `--arm CHf` に含まれない）が 2 本未満。同じ腕のプロセスが生きていれば起動しない。`provenance.json` のある腕は完了として飛ばし、`state.pt` だけある腕は再開する。
2. **GPU**: `nvidia-smi --query-gpu=memory.free` − 2,048 MiB ≥ 1.25 × 1,180 MiB（= 1,475 MiB。K-cost の 1 本の峰 1,174 MiB の切り上げ、CUDA の context 込み）。別 agent（`adamw_dose2_1011`）や ollama が使う分は、この空きにすでに入っている。
3. **RAM**: MemAvailable − 6,144 MiB − 1,040 MiB（読み込み時の峰の上乗せ、VmHWM 3,333 − 定常 2,305 の切り上げ）≥ 1.2 × 2,310 MiB（定常の VmRSS の切り上げ）。つまり MemAvailable ≥ 9,956 MiB。swap は条件に入れない（SwapFree は他のアプリの古いページで 0.65 GiB しか無く、swap の条件を置くと起動が何時間も止まる。6 GiB の余裕で受ける）。
4. **記録**: 起動ごとに時刻・HEAD と作業木の汚れ・生きている本数・GPU の空き・MemAvailable・両方の余裕・マシン全体の proj_004_drift の python の本数（`pgrep -c -f '[p]roj_004_drift/\.venv/bin/python'`）・`nvidia-smi` のプロセスごとの使用量。
5. 起動の後は 90 秒待ってから次の予算を読む（読み込みの峰は 10 秒ほどで過ぎる）。
6. 起動の後の確認（私が手で行う）: 1〜2 分後に `pgrep` でプロセスが生きていること、`nvidia-smi` に載っていること、最初の課題の後にログに `task  1` の行（課題番号・秒数・腕名だけ）が出ること。
7. 18:03 の試し（`DRY=1`）では、GPU の空き 11,766 MiB・MemAvailable 14,671 MiB で、GPU の余裕 8,243・RAM の余裕 4,715 MiB、主 2 腕は起動の判断になった（3 本目は枠の上限 2 で待つ）。
8. **見込み**: 別 agent の 3 本が GPU に居る状態の K-cost で、CHf・Hf の同時は各 0.642 秒/epoch ＋ 課題末の診断 6 秒 → 各約 131 分（並行）。Hs・CHs（R = 1）は P181 の Cr（R = 1）の単独 15 秒/課題から、GPU を分けて 1 本 15〜25 分。全体で約 2.5 時間（別 agent が終われば短くなる）。
