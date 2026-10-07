# baselines_cifar5p1_1008 — 5+1 CIFAR × MLP（R・std・lr 1e−4）に CBP・Shrink & Perturb・ReDo を掛け、L2 Init・SNA・KKT1・R と同じ指標で並べる

状態: **事前登録**（この文書の commit が登録。3 手法の値はどの seed でもまだ 1 つも無い）
作成: 2026-10-08 / 起草: Claude（Opus 5.5、親 Claude の依頼）/ run: `baselines_cifar5p1_1008` / branch: `claude/baselines_cifar5p1_1008` / worktree: `wt/baselines_cifar5p1_1008`（origin/main `49ca6ade` から）
目的: 論文 V12 で標準手法との比較が L2 Init だけなので、この箱の原典（Kumar et al. 2024）が並べた再初期化・正則化系の 3 手法 **CBP（continual backprop）・Shrink & Perturb（S&P）・ReDo** を、ハイパーパラメータを較正 seed だけで選んでから評価 seed で走らせ、既存の L2 Init・適応 Snake（SNA）・くねくね（KKT1）・ReLU（R）と同じ指標で並べる。あわせて ReDo を「死んだ unit を戻すと成績が戻るか」の介入として読む。
箱: [`spec_cifar5p1_mlp_0920.md`](spec_cifar5p1_mlp_0920.md) §2（箱）・§3（指標・fresh 対照）・§15（追補 3: seed 10–19 と同等性の判定の型）。エンジンの元: `src/cifar5p1_mlp_0920.py` の `run()`（無改変）。
実装（これから作る）: `src/baselines_cifar5p1_1008.py`・`analysis/baselines_cifar5p1_1008/{checks.py,verdict.py,launch.sh}`・出力 `results/baselines_cifar5p1_1008/`。既存の `src/*.py` は 1 文字も変えない（新しいファイルから import する）。

## 0. 一行と問い

**実ラベルの 5+1 CIFAR × MLP（ReLU・std・Adam lr 1e−4）で、CBP・S&P・ReDo（いずれも両隠れ層）は R の可塑性の喪失をどこまで減らし、L2 Init（λ 1e−3）・SNA・KKT1 と比べてどこに着地するか。**

- 主判定は評価 seed 0–19（20 seed）の後期窓（hard 21–29 の online）の対応差で、追補 3 と同じ型（符号検定・順序統計の区間・±0.005 の同等性帯）。
- ハイパーパラメータは較正 seed 100–109 だけで、文献の既定値を中心にした小さな格子から後期窓の平均で 1 つずつ選び、**選んだ値と較正の表を commit してから** 評価 seed を開く。
- 副: ReDo を「死んだ unit を戻す介入」として、戻した unit の数・死 l2・後期窓の関係を読む。S&P の縮めが第 2 層の相対位置 z̄/sd を動かすかを読む（前の走 `cap_cifar5p1_1007` では行ノルムの上限は相対位置を変えず効かなかった）。

## 1. 出所・既知のこと（登録の前に見たもの）

### 1.1 箱の既知値（committed の宿主の記録、seed 0–19。私がこの依頼で読み直した値）

後期窓 = hard 21–29（t = 21,23,25,27,29）の online の平均。20 seed（`results/cifar5p1_mlp_0920/` の seed 0–9 のセルと `*_s10-19` のセル）。

| 腕 | 後期窓 中央値（0–9 / 10–19 / 20 seed） | 平均 | fresh gap 中央値 | 死 l2 | mob l2 | eff_rank l2 | z̄ l2 | zsd l2 |
|---|---|---|---|---|---|---|---|---|
| R | .4241 / .4440 / .4324 | .4362 | +.228 | .336 | .003 | 9.4 | −8.53 | 5.49 |
| SNA | .6710 / .6567 / .6606 | .6599 | +.041 | .000 | .802 | 49.3 | −2.21 | 5.68 |
| KKT1 | .6667 / .6627 / .6637 | .6614 | +.036 | .000 | .799 | 48.5 | −2.26 | 5.55 |
| R+l2init(1e−3) | .6649 / .6542 / .6612 | .6587 | −.009 | .000 | .642 | 36.8 | +1.06 | 2.96 |
| R+l2init(1e−2) | .6577 / .6504 / .6548 | .6533 | −.010 | .000 | .816 | 25.6 | +1.63 | 1.95 |

（死 l2 以降は後期窓の課題終端値の平均の seed 中央値。R の fresh(t29) の seed 中央値 .640。）
20 seed の対応差（後期窓）: SNA − KKT1 −.0027 [−.0042, +.0001]、SNA − l2init(1e−3) +.0009 [−.0020, +.0042]、KKT1 − l2init(1e−3) +.0023 [−.0005, +.0037]、SNA − R +.2213（20/20）。区間は [v(6), v(15)]。
R の死 l2 は課題の画像で測るので hard と easy で揺れる（hard の課題終端で t1 .01 → t29 .38、easy では .46–.52）。死ぬのは第 2 層だけ（第 1 層の死は全課題で 0）。

### 1.2 文献と参照実装（手法の出所）

- **Kumar, Marklund & Van Roy (2024, CoLLAs; arXiv 2308.11958 v3)** — この箱の原典。§5 と付録 A.1.3・A.3・表 4 を読んだ。
  - 格子（Adam）: S&P は縮め p ∈ {1e−2, 1e−3, 1e−4, 1e−5}・雑音 σ ∈ {1e−2, 1e−3, 1e−4, 1e−5}、CBP は置換率 r ∈ {1e−1, …, 1e−6}（成熟 100・効用の減衰 0.99 は Dohare の値）、ReDo は周期 1・2・5 課題 × 閾値 {0, 0.01, 0.1}。
  - **表 4（5+1 CIFAR・Adam で選ばれた値）**: S&P α = 1e−3・p = 1 − 1e−4・σ = 1e−2、CBP α = 1e−3・r = 1e−4、ReDo α = 1e−3・周期 1560・閾値 0、L2 Init α = 1e−3・λ = 1e−2、Baseline α = 1e−4。
  - S&P の雑音: 付録 A.3 によれば、Ash & Adams (2020) の提案どおり雑音 ϵ は初期化と同じ分布から引いて σ 倍する（層の幅と種類に雑音の大きさを合わせるため）。更新は θ ← p(θ − α∇L) + σϵ（更新の後に縮めて足す）。初期化は全腕 PyTorch 既定 U(±1/√fan_in)（重みも bias も）。
  - CBP は公開の GitHub 実装（Dohare のもの）を使い、成熟 100・減衰 0.99。
  - 図 2（CNN・Adam の総平均 online）: L2 Init .67・L2 .55・ReDo .53・CBP .52・Layer Norm .51・S&P .45・Concat ReLU .37・Baseline .37。
  - **注意**: 3 手法の選ばれた α は 1e−3。本 spec は箱の登録セル（lr 1e−4）を変えない（R・SNA・KKT1・l2init と同じ lr で並べるため）。lr を手法ごとに選ばないことは §10 の限定に書く。
- **Dohare et al. (2024, Nature 632: 768–774; 前刷 arXiv 2306.13812)** — CBP の定義。手元の前刷（`obsidian-research-data/elu_theory_0929/wf2/literature/papers/dohare_2306.13812.txt`）の Algorithm 1（ρ・η・m の例 10⁻⁴・0.99・100）と Algorithm 3（Adam: 置換した重みの moment を 0、更新番号も 0）。効用は Nature 版の **contribution utility**: u ← η u + (1 − η)|h| Σ_k |w_out|。
  - 参照実装: `github.com/shibhansh/loss-of-plasticity` の `lop/algos/gnt.py`（`GnT`）と `lop/utils/AdamGnT.py`（2026-10-08 に取得して読んだ）。GnT: 毎更新 age += 1、u ← η u + (1−η)·mean_k|W_next[k,i]|·mean_b|h_i|、偏り補正 û = u/(1 − η^age)、age > m が対象、置換数 ρ·n_eligible（`accumulate=True` なら端数を繰り越す）、û の小さいものから置換、入力の重み ← U(±bound)（`init='default'` で bound = √(1/fan_in)）、bias ← 0、出力の重み ← 0、u・age ← 0。AdamGnT: 要素ごとの更新番号を持ち、置換した要素の exp_avg・exp_avg_sq・step を 0 に戻す。
- **Sokar et al. (2023, ICML; arXiv 2302.12902)** — ReDo の定義。手元の本文（同フォルダ `sokar2023_2302.12902.txt`）: 層の平均で正規化した活性の大きさ s_i = E|h_i| / ((1/H)Σ_k E|h_k|)、s_i ≤ τ を τ-休眠とし、周期 F ごとに入力の重みを元の初期化分布から引き直し、出力の重みを 0 にする。表 1: 周期 1000・τ = 0.025（既定）/ 0.1、スコアを見積もるミニバッチ 64。
  - 参照実装: `github.com/google/dopamine` の `dopamine/labs/redo/weight_recyclers.py`（`NeuronRecycler`、同日取得）。スコアは `mean|h| / (mean(score) + 1e−9)`、休眠の bias を 0 に、入力の重みを初期化分布で引き直してから出力の重みを 0 に、**Adam の mu・nu は入力・出力の重みの該当分だけ 0**（1 次元の bias の moment は触らない）、更新番号（optax の count）はそのまま。
- repo の過去の実装（`src/train.py` の `apply_method()`・`src/cbp_harm.py`）は 1 隠れ層・SGD の箱用（S&P の雑音は素の等方ガウスで初期化の尺度に合わせない）なので、定義の確認にだけ使い、コードは使わない。

### 1.3 登録の前に見たもの（開示）

- §1.1 の比較腕（R・SNA・KKT1・l2init）の記録は全部読んである。本走の R 腕（`none`）は宿主の R と同じ網の再走（§8 S-nochange で一致を示す）なので、R だけで決まる量（§5.5 (B)、R の副の値）は登録の前に答えが分かっている。私の予測ではそれらを採点しない。
- 3 手法は **どの seed でも 1 歩も走らせていない**。検査（§8）は検査・較正用の seed（100–109）と短縮走だけで回す。評価 seed 0–19 で手法を走らせるのは §3 の選択を commit した後。
- 宿主の無改変 `run()` を今の環境で R・seed 100–109・2 課題だけ走らせ、所要（1.70 ms/更新、RSS 1.8 GB）を測った（値は読んでいない較正 seed の 2 課題で、選択にも判定にも使わない）。

### 1.4 環境

`.venv` の torch 2.13.0+cu130。committed の記録は 2.13.0+cu126。`cap_cifar5p1_1007` §1.4 の実測で、宿主の無改変 `run()` は今の環境で R の committed 記録を eff_rank 以外の全列でバイト一致に再現し（fresh_control.csv もバイト一致）、eff_rank_l2 だけが死んだ unit のある行で最大相対 7.8e−8 ずれる（CPU の固有値計算の丸め）。本 spec の判定は eff_rank を使わない。比較腕（SNA・KKT1・l2init）が今の環境で再現することは §8 S-host-repro で確かめる（再現しなければ本走に進まず親に返す）。スレッド数は宿主の CLI 既定の 2 に固定する。

## 2. 設計

### 2.1 箱（宿主と同一・変えない）

CIFAR-100（sha256 `85cd44d0…`）、hard（5 クラス × 500 枚）と easy（1 クラス 500 枚）が交代する 30 課題（task 1 が hard）、780 更新/課題・batch 32、3072–100–100–100・ReLU（宿主の腕 `R`）、入力 `std`、Adam lr 1e−4・β (0.9, 0.999)・ε 1e−8・WD なし（moment と更新番号は課題をまたいで保持）、R = 10 を 1 プロセスに束ねる（slot = seed 昇順）、CUDA graph、fresh 対照あり。乱数の役割（`c51_classes`・`c51_batch`・`init`）も宿主のまま。

### 2.2 エンジン

`src/baselines_cifar5p1_1008.py` に宿主の `run()` を写し、R/std に要らない分岐（iv・扉・nan_slot）を落として手法を足す（`cap_cifar5p1_1007` と同じ作り）。データ・課題列・バッチ・初期化・評価（`evaluate`・`accuracy`）・活性化・束ね forward・CSV 書き出しは宿主（`cifar5p1_mlp_0920`・`rlcifar_mlp_battle_0918`・`pmnist_0905`）から import する。

- 腕（method）: `none`（宿主の R と同じ演算列。手法の演算を 1 つも入れない）・`snp`・`cbp`・`redo`。
- 捕捉した 1 更新（CUDA graph）は宿主の forward・backward・Adam。`cbp` だけは graph の中に age と効用の更新、置換された要素の更新番号の補正（§2.4）を足す。手法の書き込み（S&P の縮めと雑音・CBP の置換・ReDo の再初期化）は graph の外で、再生の直後に eager に行う（数の決まらない選択と乱数を graph に入れないため）。graph が読む静的テンソルは常に `copy_`・添字代入でその場で書き換え、再束縛しない。

### 2.3 Shrink & Perturb（`snp`、ハイパーパラメータ ε・σ）

- 毎更新、宿主の Adam 更新の直後に、6 テンソル（W1, b1, W2, b2, W3, b3 の順）すべてに `p ← (1 − ε)·p + σ·ζ`、ζ は要素ごとに独立に U(−β_l, β_l)、**β_l = 1/√fan_in(l)**（fan_in = 3072, 100, 100。W も b も同じ β_l）＝宿主の初期化の分布（Kumar 付録 A.3・Ash & Adams）。演算は `p.mul_(1 − ε).add_(ζ, alpha=σ)`、ζ = (rand·2 − 1)·β_l（float32）。
- Adam の m・v・更新番号は触らない。
- 乱数: プロセスに 1 つの CUDA generator、種は sha256(`baselines_cifar5p1_1008|snp_noise|<seed の並び>`) の先頭 8 byte。同じ seed 群なら (ε, σ) の格子のすべてのセルで同じ ζ の列（共通乱数）。束ね（R = 10・slot = seed 昇順）が決まれば再現する。
- 平衡の目安（記述）: 縮めと雑音だけの過程の定常 SD は初期化の SD の σ/√(2ε − ε²) 倍。

### 2.4 CBP（`cbp`、ハイパーパラメータ ρ、成熟 m = 100、減衰 η = 0.99）

Dohare（Nature 2024）の contribution utility と GnT（`accumulate=True`）・AdamGnT に従う。隠れ層 l ∈ {1, 2}（各 100 unit）。

- **毎更新（graph の中、Adam 更新の後）**: age_l += 1。u_l ← η·u_l + (1 − η)·c_l、c_{l,i} = (mean_k |W_{l+1}[k, i]|)·(mean_b |h_{l,b,i}|)（W_{l+1} は更新後、h はこの更新の forward の値＝GnT と同じ時点）。
- **置換数（CPU、厳密な有理数）**: n_elig,l = #{i : age_{l,i} > m}（全 slot で同じ。age の帳簿は CPU でも持ち、GPU の age と一致することを検査する）。acc_l += ρ·n_elig,l（ρ は登録した 10 進表記の `Fraction`）、k_l = ⌊acc_l⌋、acc_l −= k_l。GnT の既定 `accumulate=False`（1 未満のとき乱数で 1 本）は slot ごとに本数が変わり大域乱数が要るので使わない。
- **置換（k_l ≥ 1 の更新だけ、graph の外）**: û_l = u_l/(1 − η^{age_l})。slot ごとに、対象（age > m）のうち û の小さい k_l 本（同値は unit の番号が小さい方。CPU で float32 の値を安定ソート）。2 層とも同じ状態から選んでから、第 1 層 → 第 2 層の順に（GnT の順）:
  - W_l[i, :] ← U(±1/√fan_in(l))（箱の初期化分布＝GnT `init='default'`）。乱数は slot の seed ごとの CPU stream `H.stream("b5_cbp_reinit", seed)`、unit 番号の昇順。
  - b_l[i] ← 0（GnT）。W_{l+1}[:, i] ← 0。u_{l,i} ← 0、age_{l,i} ← 0。
  - Adam: W_l[i, :]・b_l[i]・W_{l+1}[:, i] の m・v ← 0、**要素ごとの更新番号 ← 0**（AdamGnT）。
- **要素ごとの更新番号**: 各 unit の最後の置換の大域更新番号 L（0 = 一度も置換されていない）を持ち、要素の更新番号 s = t − L（W2[j, i] は max(L₂[j], L₁[i])、W3[:, j] は L₂[j]、W1[i, :]・b1[i] は L₁[i]、b2[j] は L₂[j]）。置換された要素の偏り補正は 1/(1 − β₁^s)・1/(1 − β₂^s)（float32、デバイス上）。**一度も置換されていない要素は宿主の scalar をそのまま使う**ので宿主の更新と bit 一致。更新式は宿主の m̂/(√v̂ + ε) のまま（AdamGnT は PyTorch 形 lr·√bc₂/bc₁·m/(√v + ε) で、ε の尺度だけが違う）。
- GnT の「置換した unit の平均の寄与を次層の bias へ移す」は参照実装では無演算（`test_features` で平均の活性を 0 にしてから `gen_new_features` が読む）なので実装しない。
- 置換が初めて起きる更新の前は、`cbp` 腕のパラメータが `none` 腕と bit 一致する（§8 S-cbp）。

### 2.5 ReDo（`redo`、ハイパーパラメータ τ・F）

Sokar Algorithm 1 と Dopamine の `NeuronRecycler` に従う。

- 大域更新番号 n（走の最初から 1 始まり、課題をまたいで続く）が F の倍数の更新の直後に検査する。
- **スコアの標本**: 直近の 64 枚＝更新 n−1 と n のバッチ（Sokar 表 1 の 64。どちらも学習の流れの中の画像で、特権的なデータは使わない）。その時点のパラメータで forward。
- s_{l,i} = mean_x |h_{l,i}(x)| / (mean_k mean_x |h_{l,k}(x)| + 1e−9)、**休眠 ⇔ s ≤ τ**。2 層の休眠を同じ forward から決める。
- 休眠の unit について、まず両層の入力側（W_l[i, :] ← U(±1/√fan_in(l))、乱数は `H.stream("b5_redo_reinit", seed)`・unit 番号の昇順。b_l[i] ← 0）、次に両層の出力側（W_{l+1}[:, i] ← 0）（参照実装の順。両層とも休眠なら W2[j, i] は 0 で終わる）。Adam: W_l[i, :] と W_{l+1}[:, i] の m・v ← 0。**bias の moment と大域の更新番号は変えない**（参照実装どおり）。
- n が課題の最後の更新のとき（F = 780・1560 では毎回そう）は、その課題の課題終端の評価（per_task の行）を **再初期化の前** の網で取り、評価の後で再初期化する（学習の軌道はどちらの順でも同じ）。再初期化の後の死 l2 を `dead_post_l2` として記録する。
- F = 780 は毎課題の終端、F = 1560 は偶数課題（easy）の終端、F = 100 は課題の途中にも当たる（780k は 100 の倍数でないことがあるが、検査が課題の第 1 更新に当たることはない: 100m ≡ 1 (mod 780) は解なし）。

### 2.6 fresh 対照

- **主（`fresh_control.csv`・宿主と同じ手順・手法なし）**: t29 のまさにそのバッチ列で、初期値 P0 に戻した網（Adam の m・v ← 0・更新番号 0・手法なし）を 780 更新学ぶ。**全腕で同じ網になり R の fresh と bit 一致**（§8 S-fresh・§5.5 (A)）。`fresh_gap = fresh(t29) − continual(t29)`。
  - 読みの注意（登録する）: fresh(t29) が全腕で共通なので、腕どうしの gap の対応差は t29 の水準差の符号反転に等しい。後期窓（5 課題の平均）と独立な量ではない。
- **副（`fresh_self.csv`・手法つき・REPORT_ONLY）**: 同じく P0・Adam 0・更新番号 0 から、手法を掛けたまま 780 更新（CBP の u・age・acc・L は 0 に、ReDo の検査は fresh の更新番号で数え直す（最後の更新の後の検査は online に影響しないので行わない）、S&P の generator は続きを使う）。`gap_self = fresh_self(t29) − continual(t29)`。

### 2.7 記録と出力（走ごとに 1 ディレクトリ）

- `per_task.csv`・`fresh_control.csv`: 宿主と同じ列・同じ書式（`arm` 列は活性化名 `R` のまま）。`none` 腕はこの 2 つが宿主とバイト一致する。
- `extra_task.csv`（新規・全腕）: `method, config, seed, slot, task, hard, relpos_l1, relpos_l2, relpos_excl_l1, relpos_excl_l2, zbar_over_zsd_l2, n_reset_l1, n_reset_l2, dead_post_l2, w_fro_l1, w_fro_l2, w_fro_l3, b_absmean_l1, b_absmean_l2`。
  - **相対位置**: 課題終端でその課題の訓練画像に対する前活性 z_{l,i} の平均 z̄_i と SD sd_i（`torch.std`＝宿主の zsd と同じ不偏 SD）、r_i = z̄_i / sd_i（sd_i > 0 の unit）、`relpos_l` = r_i の unit の中央値（torch の下側中央値＝宿主の z̄ と同じ）。`relpos_excl` は sd_i = 0 で除いた unit の数。`zbar_over_zsd_l2` は宿主の列 `zbar_l2 / zsd_l2`（中央値の比、比較腕にも計算できる版）。
  - `n_reset_l*`: その課題の間に置換（CBP）・再初期化（ReDo、課題終端の検査を含む）した unit の数。`dead_post_l2`: ReDo が課題終端で再初期化した課題だけ、再初期化の後の死 l2（宿主の定義）。他は空。
- `fresh_self.csv`（手法腕）: `fresh_self_online_acc, continual_online_acc, gap_self`。
- `redo_checks.csv`（ReDo）: 検査ごと・slot ごとの `n, task, step_in_task, n_dormant_l1, n_dormant_l2`。
- `provenance.json`: 宿主の項目 ＋ 手法の設定・乱数の種・ソースの sha256・spec・admission（checks.json の sha256、選択の sha256）。

## 3. 較正（seed 100–109 だけ）

### 3.1 格子（文献の既定値を中心に・事前登録）

| 手法 | 格子 | 中心（文献） | 理由 |
|---|---|---|---|
| S&P | ε ∈ {1e−5, 1e−4, 1e−3} × σ ∈ {1e−4, 1e−3, 1e−2}（9 セル） | ε 1e−4・σ 1e−2（Kumar 表 4） | 依頼文の例は σ ∈ {1e−5, 1e−4} だったが、ζ を初期化の分布に合わせる定義（依頼文と Kumar 付録 A.3）では σ ≤ 1e−4 の雑音は定常でも初期化の SD の 1% 以下（σ/√(2ε)）で S&P が縮めだけになる。Kumar が 5+1 CIFAR で選んだ σ = 1e−2 を上端に、依頼文の例の上側 1e−4 を下端に置いた。σ を 1e−2 より上に広げないのは Kumar の格子の上端だからで、本箱の lr は Kumar の選んだ 1e−3 の 1/10（同じ σ でも学習の歩幅に対する雑音が 10 倍）なので最適はむしろ下に寄ると読む。選ばれた σ が 1e−2（端）なら §6 で `EDGE_SELECTED` と書く。 |
| CBP | ρ ∈ {1e−5, 1e−4, 1e−3}（3 セル）、m = 100・η = 0.99 固定 | ρ 1e−4（Kumar 表 4・Dohare の例） | 依頼文どおり。100 unit・成熟 100 なら置換は課題あたり層ごとに約 0.78 / 7.8 / 78 本。 |
| ReDo | τ ∈ {0, 0.025, 0.1} × F ∈ {100, 780, 1560}（9 セル） | τ 0・F 1560（Kumar 表 4）、τ 0.025（Sokar 既定） | 依頼文の例（τ {0, 0.025, 0.1} × F {課題ごと = 780, 100 更新}）に Kumar の選んだ F = 1560（2 課題）を足した。 |

加えて `none` を seed 100–109 で 1 本（較正の参照・選択には使わない）。計 22 走（1 走 = R 10 の 1 プロセス、30 課題、両 fresh つき）。

### 3.2 選択規則

- 手法ごとに、セルの点数 = **seed 100–109 の後期窓の平均**（10 seed の算術平均）。1 seed でも発散したセルは候補から外す。点数最大のセルを選ぶ。点数が float64 で厳密に等しいときは表の並び（下）で先のもの。
  - S&P の並び: (1e−4, 1e−2), (1e−4, 1e−3), (1e−4, 1e−4), (1e−5, 1e−2), (1e−5, 1e−3), (1e−5, 1e−4), (1e−3, 1e−2), (1e−3, 1e−3), (1e−3, 1e−4)。
  - CBP の並び: 1e−4, 1e−5, 1e−3。
  - ReDo の並び: (0, 1560), (0, 780), (0, 100), (0.025, 1560), (0.025, 780), (0.025, 100), (0.1, 1560), (0.1, 780), (0.1, 100)。
- 選択は `analysis/baselines_cifar5p1_1008/verdict.py select` が行い、`results/baselines_cifar5p1_1008/calib/selected.json` と `calib_table.csv`（全セルの点数・中央値・早期窓・gap・死 l2・mob l2・eff_rank l2・relpos l2・置換数）を書く。
- **この 2 つと較正の全出力を commit してから** 評価 seed を開く。本走の起動は commit 済みの `selected.json` と一致する設定でなければ拒否する（§9）。

## 4. 本走

選ばれた 3 設定 × {seed 0–9, seed 10–19}（R = 10 ずつ、宿主と同じ束ね）＋ `none` × {0–9, 10–19} = 8 走。比較腕（SNA・KKT1・R+l2init(1e−3)）は committed の記録（`results/cifar5p1_mlp_0920/` の seed 0–9 と `*_s10-19` のセル）を使う（今の環境で再現することを §8 S-host-repro で確かめる）。

## 5. 登録する判定

### 5.1 窓

seed ごとに: **後期窓** = t ∈ {21, 23, 25, 27, 29} の `online_acc` の平均、早期窓 = t ∈ {1, 3, 5, 7, 9}、t29 = t29 の `online_acc`、**gap** = `fresh_control.csv` の `fresh_gap`（手法なしの fresh、§2.6）。

### 5.2 統計（追補 3 の型）

対応差 d_s = (A の値) − (B の値)、s ∈ seed 0–19（n = 20）。v(1) ≤ … ≤ v(20) を昇順。

- 中央値 = numpy の median（v(10) と v(11) の平均）。
- 符号検定: 両側・正確・0 の差は落とす（宿主 `sign()` と同じ）。
- 区間: **[v(6), v(15)]**（中央値の分布によらない区間、被覆 95.9%）。
- ラベル（排他）: `A_WINS`（p < .05 かつ 中央値 > 0）／`B_WINS`（p < .05 かつ 中央値 < 0）／`EQUIVALENT_WITHIN_0.005`（p ≥ .05 かつ −0.005 ≤ v(6) かつ v(15) ≤ +0.005、閉区間）／`UNRESOLVED`（それ以外）。

### 5.3 判定表（x ∈ {SNP, CBP, REDO}、A = x）

| ID | 比較 | 量 | ラベル |
|---|---|---|---|
| **J1-x** | x − R（本走の `none`） | 後期窓 | 4 ラベル |
| **J1g-x** | gap_x − gap_R | fresh gap（= t29 の水準差の符号反転） | `GAP_REDUCED`（p < .05 かつ中央値 < 0）／`GAP_INCREASED`（p < .05 かつ中央値 > 0）／`GAP_NOT_SHOWN` |
| **J2-x** | x − SNA | 後期窓 | 4 ラベル |
| **J3-x** | x − KKT1 | 後期窓 | 4 ラベル |
| **J4-x** | x − R+l2init(1e−3) | 後期窓 | 4 ラベル |

- 主ラベルは上の 15 本。各手法の「喪失が残るか」として gap_x そのもの（中央値・正の seed 数・符号検定）も併記する（REPORT）。
- 比較腕の gap は各腕自身の fresh（手法つき）で取られているので、J2–J4 では gap を比べない。

### 5.4 感度（REPORT_ONLY・ラベルを変えない）

- 15 本の p に Holm をかけた `p_holm` と、それで決めたラベル `label_holm` を併記し、主ラベルと違えば明記する。
- seed 0–9 と 10–19 に分けた中央値・区間 [v(2), v(9)]（n = 10、被覆 97.9%）・符号検定を併記する（向きが両群で揃うか）。

### 5.5 適用条件

- **(A) 完全性と同一性**: 8 走 × 10 seed × 30 課題の行と fresh 行が揃い、`online_acc`・`fresh_gap` が有限、発散 0。`none` の per_task.csv は committed の R（seed 0–9・10–19）と eff_rank 以外の全列がバイト一致、fresh_control.csv はバイト一致（eff_rank は判定に使わず、帯の中にあることは §8 S-nochange で seed 0–9 について確かめる。本走では最大相対差を記録する）。各手法腕の fresh_control.csv は同じ seed 群の `none` とバイト一致。比較腕の記録が 20 seed 揃う。破れは `CHECK_FAILED`（ラベルを出さない）。手法腕で 1 seed でも発散したら、その手法のラベルは `INCOMPLETE`（seed を除いて n を縮めない）とし、発散そのものを結果として書く。
- **(B) 喪失の再現**: R（`none`）の gap が 20 seed の符号検定で p < .05 かつ中央値 > 0。満たさなければ全ラベル `NOT_REPRODUCED`。（既知: 20/20 正。）
- **(C) 手法が書いたこと（フラグ・ラベルは出す）**: CBP は両層の置換の総数 > 0、ReDo は第 2 層の再初期化の総数（20 seed の和）> 0。満たさなければ `INACTIVE` を併記する（登録どおり掛けた結果なのでラベルは出す）。S&P は構成上つねに書く。

## 6. 副（REPORT_ONLY・ラベルにしない）

- **6.1 機構の読み出し**: 各腕（none・3 手法・比較腕）の死 l2・mob l2・eff_rank l2・z̄ l2・zsd l2 の後期窓（課題終端値の 5 課題平均）の seed 中央値と t29 の seed 中央値。本エンジンの腕は relpos l2、w_fro l1–l3 も。
- **6.2 S&P の相対位置**: 後期窓の relpos_l2 の対応差（S&P − R）の中央値・[v(6), v(15)]・符号検定。同じことを `zbar_over_zsd_l2` でも。前の走（行ノルムの上限は相対位置を変えなかった）と並べる。
- **6.3 ReDo: 死を戻すと成績はどこまで戻るか**
  - seed ごと: 再初期化した unit の数（l1・l2、走全体と後期窓の課題）、死 l2（後期窓・再初期化の前）、`dead_post_l2`、後期窓、gap。
  - 戻りの割合: median_s(ReDo_s − R_s) / median_s(SNA_s − R_s)、同じく l2init(1e−3) に対して。Δ死 l2（ReDo − R）と Δ後期窓の対応差の中央値・区間。
  - 用量反応: 較正の ReDo 9 セルと none（seed 100–109）で、再初期化の数・死 l2・後期窓の平均を並べる（記述）。
  - **読みの言い方（登録する。結果を見て言い換えない）**: (i) ReDo の死 l2（後期窓・20 seed の中央値）≤ 0.02 かつ (ii) J2-REDO と J4-REDO がともに `EQUIVALENT_WITHIN_0.005` か `A_WINS` なら「死を戻すと成績は SNA・L2 Init の水準まで戻る」。(i) が成り立ち J2-REDO か J4-REDO が `B_WINS` なら「死を戻すだけでは戻らない（残りは死以外）」。(i) が成り立たなければ「この ReDo は死を除けなかった」で、介入としての読みはしない。それ以外は「決まらない」。
- **6.4 CBP**: 置換の総数（層ごと・課題ごと）、死 l2、age の分布（記述）。
- **6.5 fresh_self**: 各手法の gap_self の中央値と正の数（手法つきの新品の網との差）。
- **6.6 較正**: 選ばれたセルが格子の端か（`EDGE_SELECTED`）、none（seed 100–109）に対する各セルの差。

## 7. 予測（3 手法のどの値も見る前・この commit で固定）

採点: 2 値の命題は真偽と Brier (p − o)²。多値は Σ_k (p_k − o_k)²。(A) が破れたもの・`INCOMPLETE` の手法の命題は理由付きで未採点。

### 7.1 命題の定義（採点の仕方を先に固定）

| # | 命題 | 判定に使う量 |
|---|---|---|
| P1-x | x は R に後期窓で勝つ | J1-x が `A_WINS` |
| P2a | ReDo は死 l2 をほぼ 0 にする | ReDo の死 l2（後期窓の課題終端値の 5 課題平均、再初期化の前）の 20 seed 中央値 ≤ 0.02（= 100 unit 中 2 本、R の .336 の約 6%） |
| P2b | ReDo の後期窓は SNA に届かない | 対応差 SNA − ReDo（後期窓）の 20 seed 中央値 > +0.005 |
| P3 | CBP は SNA に勝つ | J2-CBP が `A_WINS`（「同等か負ける」はその否定） |
| P4 | S&P は第 2 層の相対位置を R から 0.3 以上は変えない | 後期窓の relpos_l2 の対応差（S&P − R）の 20 seed 中央値の絶対値 < 0.3 |
| P5 | 3 手法のどれかが SNA に勝つ | J2-SNP・J2-CBP・J2-REDO のどれかが `A_WINS` |
| P6-x | x − SNA のラベル（4 値） | J2-x |

### 7.2 親（Claude）の事前確率（依頼文から転記）

| # | 確率 |
|---|---:|
| P1-SNP / P1-CBP / P1-REDO | .85 / .85 / .85 |
| P2a | .80 |
| P2b | .55 |
| P3 | .15 |
| P4 | .60 |
| P5 | .25 |

### 7.3 私（Claude Opus 5.5）の確率

| # | 確率 |
|---|---:|
| P1-SNP / P1-CBP / P1-REDO | .85 / .93 / .95 |
| P2a | .55 |
| P2b | .70 |
| P3 | .05 |
| P4 | .35 |
| P5 | .10 |
| P6-SNP（A_WINS / B_WINS / EQUIV / UNRESOLVED） | .03 / .85 / .03 / .09 |
| P6-CBP | .05 / .75 / .08 / .12 |
| P6-REDO | .05 / .70 / .10 / .15 |

理由（短く）: この箱の R の負けは第 2 層のゲートが閉じること（mob .003・死 .34）で、L2 Init と SNA はそれを止めて .66 に届く。ReDo と CBP は閉じた unit を引き直すので R には大きく勝つが、引き直した unit は次の課題の押しで再び沈みうるうえ、出力の重みを 0 から育て直す分だけ遅れる。Kumar（CNN）でも CBP .52・ReDo .53 は L2 Init .67 に遠い。S&P は lr 1e−4 では雑音が学習の歩幅に比べて大きく、縮めは ReLU の斉次性で相対位置をほとんど動かさない（前の走の行ノルム上限と同じ理屈）。ただし S&P が効くならその分だけ相対位置が戻るはずなので、P4 は親より低く置いた。P2a は、F = 780・1560 だと課題の中で死ぬ分が課題終端の測定に残るので 0.02 以下まで下がるかは五分よりやや上。

## 8. 検査（実装の後・較正の前に全部 PASS させ、列挙した変異で全部 FAIL させる）

`analysis/baselines_cifar5p1_1008/checks.py`。各検査は本物のモジュールで PASS し、変異（ソースのちょうど 1 か所の置換で読み込んだモジュール、または設定の変異）で FAIL しなければならない。結果は `results/baselines_cifar5p1_1008/checks.json`（`all_pass` = 全検査 PASS かつ全変異を検出）。検査の走の出力は git の外（セッションの scratch）。手法腕は検査用 seed（100–109）と短縮走だけで回す。閾値は固定値・固定倍率にせず、算術か標本分布から導く。

| 検査 | 要求 | 変異（FAIL しなければならない） |
|---|---|---|
| **S-nochange** | (a) 本エンジンの `none`（seed 0–9・30 課題・graph・threads 2）の per_task.csv・fresh_control.csv が、同じ条件の宿主の無改変 `run()` の出力とファイルごとバイト一致。(b) committed の `R_std_lr0.0001` と eff_rank 以外の全列がバイト一致・fresh_control.csv がバイト一致・eff_rank は帯（`cap_cifar5p1_1007` §7.1 と同じ算術: 固有値の誤差 2d の箱からの区間演算＋印字の丸め）の中、帯は隣の課題の値を拒否する。(c) 短縮走でも (a) が成り立つ | `none` に極小の縮め（ε = 1e−7）が入る／lr × 1.0001／Adam の更新番号を課題ごとに 0 に戻す |
| **S-host-repro** | 宿主の無改変 `run()` を今の環境で SNA・KKT1・R+l2init(1e−3)（seed 0–9 と 10–19）で回し、committed と eff_rank 以外の全列・fresh_control.csv がバイト一致（比較腕の記録が環境の差で動いていない） | 照合先を別のセル（l2init 1e−2）にする／seed 群を入れ替えて照合する |
| **S-snp** | 1 更新ずつの hook で: Adam 直後の P と S&P 後の P の関係が、同じ generator の状態から独立に引き直した ζ による `(1−ε)p + σζ` と bit 一致（6 テンソルすべて）、ζ の値域が [−β_l, β_l]、ζ の平均・分散が U(±β_l) の値から標本数で決まる帯（平均 ±4·β/√(3N)、分散 ±4·SD(ζ²)/√N）の中、m・v・更新番号は不変、`fresh_control` は `none` と同じ | bias を縮めない／ζ を U でなく N(0, β²) にする／β を fan_out から作る／S&P を Adam の前に掛ける／σ を掛け忘れる |
| **S-cbp** | 短縮走（seed 100–102・eager・ρ 1e−3・更新ごとの hook）で、hook の状態から独立に再計算した: age・効用（float64 の再計算との差が float32 の積和の丸めの帯の中）・置換数（Fraction の帳簿、全 slot で同じ）・選ばれた unit（û の安定ソート）・置換後の W_l 行が値域内で新しい・b_l = 0・W_{l+1} 列 = 0・該当要素の m・v = 0・u・age = 0・それ以外は bit 不変・置換された要素の次の更新が更新番号 s の Adam の式と一致（float64 の再計算との差が丸めの帯の中）・置換されていない要素は宿主の式と bit 一致・最初の置換の前は `none` と bit 一致 | 効用に入力側の重みを使う／成熟を無視する／出力の重みを 0 にしない／出力側の moment を 0 にしない／要素の更新番号を戻さない（大域の偏り補正）／端数を繰り越さない／効用の大きい方を選ぶ |
| **S-redo** | 短縮走（seed 100–102・eager・F 50、τ 0.1 と τ 0 の 2 通り、hook）で: スコアが直近 64 枚から参照式どおり（独立再計算と一致）、休眠集合 = {s ≤ τ}、入力行が値域内で新しい・b = 0・出力列 = 0・入力行と出力列の m・v = 0・**bias の m・v と大域の更新番号は不変**・それ以外 bit 不変。τ = 0 では再初期化の前後でスコアの 64 枚に対する logits が一致（Sokar の不変性）。課題終端の検査では per_task の行が再初期化の前の網の値 | スコアを層平均で割らない／標本を直近 32 枚にする／出力の重みを 0 にしない／moment を 0 にしない／bias の moment も 0 にする／周期を 1 ずらす（n mod F = 1）／課題終端の再初期化を評価の前に行う |
| **S-graph** | 3 手法それぞれ（S&P ε 1e−3・σ 1e−2、CBP ρ 1e−3、ReDo τ 0.1・F 100）、seed 100–109・3 課題 × 200 更新で、graph と eager の全出力（per_task・extra_task・fresh_control・fresh_self・redo_checks）がバイト一致、最終の P・m・v が bit 一致 | CBP の L を `copy_` でなく再束縛で書く／graph の再生時だけ更新番号のデバイス値を書き忘れる／graph の再生時だけ static_idx を 1 バッチずらす |
| **S-fresh** | 短縮走で、3 手法の fresh_control.csv が `none` と、`none` が宿主とバイト一致。fresh_self の開始時に P = P0・m = v = 0・手法の状態が初期値（hook） | 手法なしの fresh で手法を掛けたままにする／fresh_self の前に CBP の age を戻さない／fresh_self の前に Adam の moment を 0 にしない |
| **S-select** | 合成の較正表で: 平均最大のセルを選ぶ・発散したセルを除く・同点は表の並び・登録外のセルや seed 100–109 以外の行を拒否・格子の欠けを拒否 | 中央値で選ぶ／早期窓で選ぶ／発散セルを除かない |
| **S-verdict** | 合成 fixture で 4 ラベルと gap の 3 ラベル・区間の順序統計（n 20 で v(6), v(15)、n 10 で v(2), v(9)）・符号検定の正確な p（例 15/20 → .0414、14/20 → .1153）・帯の端（±0.005 ちょうどは内）・0 の差を落とす・Holm・予測の採点（P2a の 0.02 ちょうどは真） | 区間を [v(5), v(16)] にする／帯を ±0.01 にする／対応差でなく中央値の差にする／窓を hard 19–27 にする／片側 p にする |
| **S-CLI** | CLI（`python -m src.baselines_cifar5p1_1008 run`）の出力が同じ引数の関数呼び出しとバイト一致。既定値（30 課題・lr 1e−4・threads 2・m 100・η 0.99）。登録先（`results/baselines_cifar5p1_1008/calib|main`）への出力は checks.json の all_pass・ソースの sha256 一致・git clean・（main は）commit 済みの selected.json と一致、でなければ拒否 | CLI が `--steps-hard` を無視する／既定の課題数が 30 でない |

## 9. 実行・資源

- 起動は `analysis/baselines_cifar5p1_1008/launch.sh calib` → 選択と commit → `launch.sh main`。checks.json が all_pass かつソースの sha256 が一致、`src`・`analysis`・`specs` が git clean、でなければ止まる。main は commit 済みの `selected.json` が要る。
- GPU のプロセスは常に 1 本（腕は直列）。各走は共有の GPU lock `/tmp/lop_analysis_gpu.lock` を flock で取ってから走る。起動の前に GPU の空きが 8.5 GiB 以上（1 走 約 2 GiB を引いても 6 GiB 以上残る）・RAM の MemAvailable が 6 GiB 以上でなければ待つ（30 分で諦めて止まる）。`--threads 2`。provenance.json のある走は飛ばす（再起動しても安全）。途中の成績で順番・実行の有無を変えない。
- 順: calib は none → S&P 9 → CBP 3 → ReDo 9。main は none s0-9 → none s10-19 → snp → cbp → redo（各 s0-9 → s10-19）。
- 集計は `analysis/baselines_cifar5p1_1008/verdict.py judge`（verdict.json・paired.csv・per_seed.csv・secondary.csv・predictions.csv）。結果ノートは `results/baselines_cifar5p1_1008/summary.md`。

## 10. 読みの上限

- この箱・30 課題・R/std・**lr 1e−4 固定**に限る。Kumar は 3 手法とも lr 1e−3 を選んでいる（CNN）。本 spec は比較腕と同じ lr で並べるために lr を手法ごとに選ばないので、「手法が lr ごと最適化されたら」の比較ではない。格子は小さい（S&P 9・CBP 3・ReDo 9）。
- 較正は seed 100–109 の 10 seed で、選択の雑音は残る（選ばれたセルが真の最良とは限らない）。評価 seed は選択に使っていないので、判定は選択に対して held-out。
- 同等性は ±0.005 の帯と 20 seed の順序統計の区間で決める。`UNRESOLVED` は「決まらない」で、同等でも差でもない。`A_WINS`/`B_WINS` は 20 seed の対応差の向きで、効果の大きさは区間で読む。
- ReDo の読み（§6.3）は「この ReDo（選ばれた τ・F・スコアの標本）で死を戻したとき」に限る。死を戻す他の方法（扉 H・l2init）とは別の介入で、効果量を並べて同じ機構とは言わない。
- S&P の相対位置の読みは記述で、因果は言わない。

## 11. 出力と片付け

- git に入れる: spec、`src/baselines_cifar5p1_1008.py`、`analysis/baselines_cifar5p1_1008/`、`results/baselines_cifar5p1_1008/`（checks.json、calib/ と main/ の全走の CSV・provenance、calib_table.csv・selected.json、verdict の出力、summary.md、起動ログ）。
- CLAUDE.md §4: git の外のファイル（`__pycache__` 以外。`data` の symlink は共有データなので辿らず移さない）と検査の作業場所の出力を `~/Projects/obsidian-research-data/baselines_cifar5p1_1008/` に移し、`results/baselines_cifar5p1_1008/backup_manifest.json`（source・backup・bytes・sha256）を commit。`git fetch origin && git merge origin/main && git push origin HEAD:main`（断られたら fetch からやり直す）。worktree とブランチは親が確かめるまで消さない。vault には書かない。
