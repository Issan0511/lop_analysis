# adamw_dose2_1011 — WD を強く言うための 2 つの穴: RL-CIFAR × ELU→ELU で「救う λ」を規則で事前に当てる（A1）と、5+1 CIFAR × ReLU で効くのは「引く先」か「Adam の正規化」か（A2）

状態: **事前登録**（本ファイルと `analysis/adamw_dose2_1011/rule.py` の commit が登録。A1 の AdamW の腕も A2 の coupled L2 の腕も 1 歩も走らせていない。実装もまだ無い）
作成: 2026-10-11 13:30 JST / 起草: Claude（Opus 5.5、サブエージェント）/ 依頼: 親（Fable）。Issa の指示は「走らせたい」。本走の前に親の確認を取る（§8）。
目的: V12 の要旨で WD を強く言うための穴を 2 つ埋める。**(A1)** 救う λ を、P180（`adamw_dose_1010`、MNIST ELU）とは別の箱で、**走る前に規則で予測**できることを示す。**(A2)** ReLU で効くのが「引く先（初期値か 0 か）」か「Adam の正規化を通るか（coupled か decoupled か）」かを分ける。
親 spec: `specs/spec_cap_cifar_ee_0920.md`（A1 の箱・主窓・床・E1/E2・ラベル・IMPAIRED_EARLY をそのまま使う）、`specs/spec_adamw_dose_1010.md`（P180。decoupled の実装、DOSE・EQUILIBRIUM・WIDTH・DEPTH、A2 の判定と記録の再利用の照合）、`specs/spec_cifar5p1_mlp_0920.md` §13（L2 Init）。
実装（予定・登録の後に書く）: `src/adamw_dose2_1011.py`（A1 のエンジン。`src/cifar_interventions_0920.Engine` の子クラス）・`src/adamw_dose2_1011_c51.py`（A2 のエンジン。`src/adamw_dose_1010_c51.py` の写しに coupled を足す）・`src/adamw_dose2_1011_run.py`（CLI）・`analysis/adamw_dose2_1011/{checks.py, verdict.py, launch.sh, backup.py}`。**`analysis/adamw_dose2_1011/rule.py` は規則の算術で、本 spec と同じ commit で登録する。** 既存の `src/*.py` は 1 文字も変えない。
branch: `claude/adamw_dose2_1011`（worktree `~/Projects/claude/wt/adamw_dose2_1011`、origin/main `43e0521b` から）

## 0. 一行と問い

- **Q1（A1・主）**: RL-CIFAR × ELU→ELU・std（`cap_cifar_ee_0920` の箱。cap12 が `RESCUED` 85.98% 対 ref 10.09% を出した箱）で、AdamW（decoupled・全パラメータ）の 3 水準 λ ∈ {0.03, 0.1, 1} の結果を、P180 の規則（釣り合いの幅 × 射影の広がり = 幅 s、ref の相対位置 r、深さ m = r·s を訓練の微分の零点と比べる）で走る前に当てられるか（`RULE_PREDICTED`）。λ は、規則が「救わない（幅は止まるが深い）」「際どい」「救う」を 1 つずつ出すように選んだ（§2.3）。
- **Q2（A1）**: 各 λ は cap_cifar_ee の登録規則で何と判定されるか（`RESCUED` など）。主窓の online は λ とともに上がるか（`DOSE_*`）、行ノルムは回転平衡に落ち着くか（`EQUILIBRIUM_*`）、幅は止まるか（`WIDTH_*`）、第 2 層は零点の上にいるか（`DEPTH_*`）。
- **Q3（A2・主）**: 5+1 CIFAR × MLP × ReLU（std・lr 10⁻⁴・30 課題 × 780 更新）で、**coupled L2（勾配に 2λθ を足して Adam に通す・0 へ引く・全パラメータ）**は、L2 Init（同じ λ = 10⁻² で初期値へ引く coupled）並みに救うか、R 並みか（`RESCUED` / `PARTIAL` / `NO_RESCUE`）。読みのラベル: L2 Init 並み → 「Adam の正規化が効く」（`NORM_MATTERS`）、R 並み → 「引く先が効く」（`TARGET_MATTERS`）、中間 → 「両方」（`BOTH`）。

## 1. 既知（登録の前に見たもの。予測の根拠であり、当たっても独立な予測の成功としては数えない）

### 1.1 A1 の箱（cap_cifar_ee_0920）の記録

committed の `results/cap_cifar_ee_0920/per_task.csv` と、退避先 `~/Projects/obsidian-research-data/cap_cifar_ee_0920/` の unit ごとの配列（`prefix/units.npz`・`arms/<腕>/units_tXX.npz`）から。幅 s・行ノルム・相対位置 r = z̄₂/σ₂ は unit の中央値を 10 seed で平均した。主窓は t31–50。

| 腕・時点 | online | G2 | Z（第 2 層の対で訓練の微分が厳密に 0 の割合） | s1 | ‖w1‖ | ρ1 = s1/‖w1‖ | s2 | ‖w2‖ | f = (s2/‖w2‖)/s1 | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ref t1（全腕の共通の接頭部） | .983 | .353 | .0001 | 16.5 | 3.68 | 4.48 | 12.8 | 1.01 | 0.76 | −0.55 |
| ref t2 | .919 | .066 | .404 | 49.9 | 9.70 | 5.13 | 76.7 | 1.87 | 0.84 | −1.30 |
| ref t3 | .404 | .018 | .813 | 78.6 | 12.8 | 6.14 | 160 | 2.16 | 0.96 | −1.55 |
| ref t5 | .136 | .001 | .984 | 129 | 15.5 | 8.37 | 316 | 2.22 | 1.12 | −1.84 |
| ref 主窓 | .101 | 1.4e−5 | .9985 | 191 | 20.1 | 9.47 | 482 | 2.20 | 1.15 | −1.97 |
| cap12 主窓（RESCUED） | .860 | .132 | .0003 | 20.2 | 3.67 | 5.50 | 14.1 | 1.01 | 0.70 | −1.09 |
| cap12_bfix 主窓（RESCUED） | .789 | .094 | .0018 | 18.4 | 3.67 | 5.02 | 16.6 | 1.01 | 0.89 | −1.23 |
| cap1 主窓（SPLIT） | .119 | .0009 | .982 | 37.3 | 3.63 | 10.4 | 176 | 4.98 | 0.96 | −2.22 |
| cap2 主窓（SPLIT） | .251 | .021 | .872 | 170 | 35.9 | 4.92 | 493 | 0.99 | 3.10 | −1.34 |

- **(a) この箱の訓練の微分の零点は z₀ = −103.97 で、−16.64 ではない。** この箱は torch 2.13 の native `F.elu` を autograd で微分する（cap_cifar_ee の spec §1）。その負側の微分は e^z で、float32 で厳密に 0 になるのは e^z が下にあふれる z < ln 2⁻¹⁵⁰ = −103.97 だけである（CUDA と CPU で確認。subnormal は落とさない環境: cap_cifar_ee の provenance）。P180 の −16.64 は MNIST の宿主の「出力 + 1」形の微分の零点で（`src/rlcifar_mlp_battle_0918.py` の `ELU` の説明と `elu-expm1-float32-zero-grad` の記憶）、この箱には当てはまらない。**出力のほうは z ≤ −16.982 で float32 の −1 に張り付く**（CUDA の `F.elu` を二分法で調べた。CPU では −16.636）。したがってこの箱では、親の「零点 −16.64（と、その手前で起きる機能低下）」は、**零点 z₀ = −103.97**（勾配が厳密に 0）と、**その手前の機能低下の線 z_f = −16.98**（出力が定数 −1、微分 e^z < 4.2×10⁻⁸ で Adam の ε = 10⁻⁸ の桁）の 2 本に分かれる。
- **(b) 正規近似の厳密 0 の割合 Z = Φ(z₀/s2 − r) は、z₀ = −103.97 で記録と合い、−16.64 では合わない。** unit 中央値の s2・r の seed ごとの値から（`rule.py --bands` の line_check）:

| 腕・時点 | 実測 Z | Φ(z₀/s2 − r)、z₀ = −103.97 | 同、−16.64 |
|---|---|---|---|
| ref t2 / t3 / t5 / t50 | .404 / .813 / .984 / .9986 | .475 / .812 / .934 / .960 | .860 / .924 / .963 / .973 |
| cap1 t3 / t5 / t10 / t50 | .168 / .492 / .785 / .9965 | .165 / .561 / .800 / .962 | .833 / .901 / .942 / .988 |
| cap2 t3 / t5 / t10 / t50 | .174 / .460 / .875 / .863 | .165 / .548 / .855 / .857 | .823 / .868 / .922 / .891 |
| cap12 t2 / t10 / t50 | .0013 / .0005 / .0002 | 1×10⁻⁶ / 4×10⁻⁸ / 5×10⁻¹⁰ | .514 / .526 / .448 |

  z₀ = −103.97 での差は飽和の近く（実測 > .98）を除いて 0.09 以内。−16.64 では cap12 で 0.45〜0.53 対 0.0002〜0.001 と外れる。cap12 は第 2 層の対の約半分で訓練の微分が 10⁻⁶ 未満（Q2 = .474、z < −13.8）にいながら RESCUED（.860）だった。**この箱で online の崩れに付いて行くのは Z（零点 z₀ の下の割合）である**（ref t2 → t3 → t4: Z .40 → .81 → .96、online .92 → .40 → .19。cap1・cap2 も同じ並び）。
- **(c) 第 2 層の射影の広がりは第 1 層の幅とともに伸びる。** ρ2 = s2/‖w2‖ を f·s1 と書くと、f は 0.67（cap12 t50）〜1.16（ref t40）で、s1 が 16.5 → 195 と 12 倍に伸びる間ほぼ一定。ρ1 は 4.48（ref t1）〜9.61（ref の後期）。両層とも自由な ref、両層とも止めた cap12・cap12_bfix の全状態（148 状態）での範囲で、片方の層だけを止めた cap1・cap2 の後期（ρ1 は最大 11.0、f は最大 3.4）は入れない（AdamW は両層を縮める。§2.3）。
- **(d) ref は全 seed で T½ = 3**（課題 3 の適合率 (A(t) − 0.1)/(A(1) − 0.1) < 1/2）。ref の生きている間（t1–t3）の幅の傾き（unit 中央値、seed 平均）は s1 +31.0/課題（seed 28.3〜35.0）、s2 +73.4/課題（63.6〜89.2）。主窓の ref の床の状態は FLOORED（10/10）、ΔE2 の区間の上端は負（cap_cifar_ee の (B) は成立）。
- **(e) 箱の η**: Adam の lr = 1×10⁻³（`src/rlcifar_mlp_battle_0918.py` の `LR`、`Engine.step` の `p.sub_(1e-3 * ...)`）、β = (0.9, 0.999)、ε = 10⁻⁸。provenance の環境: torch 2.13.0+cu130、RTX 5060 Ti、TF32 off、決定論、threads 2、`cublas :4096:8`、subnormal を保つ。ドライバは記録時 580.173.02、現在 580.178.04（S0 で bit 一致を確かめる）。

### 1.2 P180（adamw_dose_1010）で分かったこと（規則の出所）

- 行ノルムは回転平衡 η√(d/(2(1 − c))) の **×1.6〜2.3** に釣り合った（λ = 10⁻¹: 第 1 層 ×1.76・第 2 層 ×1.61、3×10⁻²: ×2.25・×1.97、10⁻²: ×1.84・×1.31）。
- **WD が救うのは、止まった幅が零点より浅いときだけ**。λ = 3×10⁻² は幅が止まった（σ2 12.3）が、r2 = σ2/‖w2‖ が第 1 層の幅とともに 4.8 まで伸びて深さ −22 になり、第 2 層の対の 65% が零点 −16.64 の下で `RESCUED_FUNCTION_ONLY`。λ = 10⁻¹ は σ2 0.60・d2 +0.59 で厳密 0 が 0% の `RESCUED`。見積もりの中心（σ2 2.3）が外れた理由が、第 2 層の射影の広がりを第 1 層の幅から伸ばさなかったこと。本 spec の規則はそれを入れる（§2.3 の f）。
- decoupled では死は吸収的でない（勾配が 0 の重みも毎更新 c 倍され 0 へ縮む）。λ = 3×10⁻² は崩れかけてから戻る「戻り」が 8/10 seed。
- 箱 2（5+1 ReLU）の AdamW λ = 5・10 は `NO_RESCUE`・`WORSE`・`IMPAIRED`（後期窓 .363・.354 対 R .437・L2 Init .656）。ゲートは動いた（相対位置 −1.57 → −0.54・−0.38、開いた対 .14 → .43・.46）が、階数 eff_rank l2 は 9.6 → 4.2・3.6、W2 の行ノルムは .69 → .07・.05。

### 1.3 A2 の箱の記録と、coupled と decoupled の強さの対応（較正 seed での測定）

- **R/std/lr 1e−4**（committed `results/cifar5p1_mlp_0920/R_std_lr0.0001/`、seed 0–9）: 後期窓（hard 21–29）平均 .4366、早期窓 .6091、死 l2 .342（= 全閉 l2 .342）、eff_rank l2 9.57、mob l2 .003。**L2 Init λ 1e−2**（`R_std_lr0.0001_l2init1e-2/`）: 後期窓 .6561、死 l2 0、eff_rank l2 26.3、mob l2 .823。L2 Init − R は 10/10 seed で正、中点 (R + L)/2 の平均 .546。L2 Init は宿主の `iv="l2init:1e-2"` で、勾配に 2λ(θ − θ₀) を足して Adam に通す（coupled、初期値へ）。宿主には `iv="l2:<λ>"`（勾配に 2λθ、0 へ）の経路もあるが、この箱では一度も走っていない（`results/` を検索して確かめた）。
- **AdamW（P180）**: adamw5 .3631・adamw10 .3542（後期窓）、早期窓 .4819・.4532、死 l2 .273・.393、eff_rank l2 4.16・3.65。
- **l2split_rlmnist_0914**: RL-MNIST の ReLU で coupled L2（λ 10⁻³、0 へ）は時間劣化の 98.9% を除いた（P180 §5 で元の CSV から確認）。Kumar et al. 2024 の CNN 版 5+1 CIFAR では 0 への L2 が .55（Baseline .37・L2 Init .67）。
- **coupled と decoupled の強さの対応（較正）**: coupled では、損失の勾配が L2 の項より大きい座標で、L2 の項の寄与は η·2λθ/√v̂ になる。decoupled の ηλ_W θ と揃えるには **λ_W = 2λ/√v̂**。R を較正 seed 100–109 で 30 課題走らせ（P180 のエンジンの λ = 0 の経路、同じ走を 2 回、per_task はバイト一致）、hard 21–29 の課題終端の √v̂（バイアス補正済み）の**全 6 テンソルを合わせた中央値**は **1.99×10⁻³**（seed 平均 1.86×10⁻³〜2.22×10⁻³、課題ごと 1.96〜2.02×10⁻³。全 327,500 個のうち W1 が 307,200 個）。テンソルごとの中央値は W1 2.0×10⁻³・W2 6.4×10⁻³・W3 3.2×10⁻⁴・b 2〜4×10⁻³。したがって **coupled λ = 10⁻² は λ_W = 10.0（adamw10）、λ = 5×10⁻³ は λ_W = 5.0（adamw5）と揃う**（全パラメータの中央値の座標で）。W2 だけで見ると λ_W は 3.1・1.6、W3 では 62・31（勾配の小さい座標ほど coupled は強く引く。これが Adam の正規化の効き方そのもの）。

### 1.4 登録の前にしたこと（開示）

- cap_cifar_ee_0920 の記録と退避先の unit ごとの配列の集計（§1.1 の表、`rule.py --bands`）。学習なし。
- `F.elu` の float32 の閾値の確認（CUDA・CPU。§1.1 (a)）。学習なし。
- 速さの測定: 無改変の `cifar_interventions_0920.Engine`（WD なし）を **較正 seed 100–109** の task 1 で 20〜60 epoch、1・2・3 本同時に走らせた（1 本 0.777 ms/更新、2 本同時で各 1.64、3 本で各 2.46。GPU は 1 本で計算に飽和し、同時に走らせても全体の処理量は変わらない。1 本あたり確保 474 MiB）。
- A2 の較正: R（λ = 0）を **較正 seed 100–109** で 30 課題 × 2 回（§1.3）。後期窓は R の seed 100–109 の値（.419 前後）を見た。
- **A1 の AdamW の腕と A2 の coupled L2 の腕は、どちらの箱でも 1 歩も走らせていない。** 登録 seed 0–9 の新しい走は無い。

## 2. 設計

### 2.1 AdamW の入れ方（A1。P180 §2.1 と同じ）

- 各更新で、宿主どおりに順伝播と勾配を計算した後、6 テンソル（W1・b1・W2・b2・W3・b3）のそれぞれに **p ← fl₃₂(p·c)、c = fl₃₂(1 − ηλ)** を掛け（`p.mul_(wd_c)`、`wd_c` は CUDA graph の捕捉の後に値を書き込む float32 の 0 次元テンソル）、続けて宿主の Adam の更新（m・v の更新と `p.sub_(1e-3 * (m * inv1) / ((v * inv2).sqrt() + 1e-8))`）を 1 文字も変えずに行う。θ_{t+1} = c·θ_t − η·m̂_t/(√v̂_t + ε) で、torch.optim.AdamW と同じ式（S-torch で丸めの帯の中で示す）。**勾配には何も足さない**。λ = 0 では減衰の演算を 1 つも入れない（宿主の step と演算ごとに同じ。S0）。
- テンソルごとの装置上のカウンタ（`wd_steps`）を graph の中で減衰と一緒に増やす。
- **減衰は task 2 の最初の更新から**入れる（cap_cifar_ee の上限と同じ分岐点。§2.2）。Adam の moment と更新番号は課題をまたいで保持する（宿主どおり）。

### 2.2 A1 の箱と腕

- **箱**は cap_cifar_ee_0920 の ref で、何も変えない: RL-CIFAR-10（課題ごとに一様乱数ラベル）、seed ごとに固定の 1200 枚、std 入力、3072–100–100–10、両隠れ層 ELU(α = 1)（native `F.elu`）、Adam η = 10⁻³・β (0.9, 0.999)・ε 10⁻⁸、batch 16、400 epoch × 75 = 30,000 更新/課題、50 課題、seed 0–9 を R = 10 で束ねる、CUDA graph、threads 2。乱数の流れは宿主（`rlc_labels`・`rlc_batch`）。
- **分岐**: 3 つの AdamW の腕は、cap_cifar_ee_0920 の共通の接頭部（task 1 終端の状態 `prefix/t01.pt`、退避先）から、cap の腕と同じやり方（`Engine(seeds, X, state=t01)`）で分岐し、task 2–50 を走る。ラベルと標本順の流れは接頭部の生成器の状態から続くので、全腕で ref と同じになる（S-branch）。
- **ref は記録を再利用する**（committed の per_task.csv の ref の行と、退避先の unit ごとの配列）。再利用の条件は S0（§6）: (a) 本エンジンの λ = 0 の経路で seed 0–9 の task 1 を最初から走らせ、状態の sha256 が `prefix/t01.pt` と、行が `prefix/rows.json` と、unit の配列が `prefix/units.npz` と一致する。(b) `t01.pt` から λ = 0 で task 2–10 を走らせ、各課題の行と unit の配列が ref の記録と、t10 の状態の sha256 が `arms/ref/t10.pt` と一致する。**S0 が成り立たなければ、本走の前に spec へ追補を書き、ref の task 2–50 を本エンジンの λ = 0 で走り直して比較に使う**（接頭部も走り直したものにする）。
- **腕**（3 腕 × seed 0–9、task 2–50）:

| 腕 | λ | c = fl₃₂(1 − ηλ) | 1 − c | τ = 1/(1 − c)（更新） | τ（課題） | τ/走（task 2–50 = 1,470,000 更新） | ‖w1‖*（d 3072） | ‖w2‖*（d 100） | 規則 |
|---|---|---|---|---|---|---|---|---|---|
| wd3e-2 | 0.03 | 0x1.fffc12p−1 | 2.9981×10⁻⁵ | 33,354 | 1.11 | 0.0227 | 7.158 | 1.291 | **救わない**（幅は止まるが深い） |
| wd1e-1 | 0.1 | 0x1.fff2e4p−1 | 1.0002×10⁻⁴ | 9,998 | 0.333 | 0.0068 | 3.919 | 0.707 | **際どい**（二値の当て: 救わない） |
| wd1e0 | 1 | 0x1.ff7ceep−1 | 9.9999×10⁻⁴ | 1,000 | 0.0333 | 0.00068 | 1.239 | 0.224 | **救う** |

- 3 水準とも τ ≤ 走/10（親の条件）。λ = 0.1 は P180 で MNIST を救った値で、規則はこの箱では際どい（救わない側）と言う。

### 2.3 規則（登録。`analysis/adamw_dose2_1011/rule.py`）

親の規則を、この箱の既知の量で算術にした。

1. **釣り合いの行ノルム**: ‖w_l‖* = η√(d_l/(2(1 − c)))（≈ √(ηd/2λ)、Kosson 2024 の回転平衡）。d1 = 3072（std の CIFAR に定数の画素は無い）、d2 = 100。実際の行ノルムは κ_l·‖w_l‖*、**κ ∈ [1.6, 2.3]**（P180 の実測、層ごと）。
2. **幅 = 釣り合いの行ノルム × その層の射影の広がり**: s1 = ρ1·‖w1‖、s2 = ρ2·‖w2‖、**ρ2 = f·s1**（第 2 層の射影の広がりは第 1 層の幅に比例して伸びる。§1.1 (c)）。**ρ1 ∈ [4.479, 9.609]、f ∈ [0.667, 1.160]**（§1.1 (c) の 148 状態の範囲）。まとめて s2 = K·‖w1‖*·‖w2‖*、K = κ1κ2ρ1f ∈ [7.65, 59.0]（中心は幾何平均 21.2）。
3. **深さ**: m = r·s2、**r ∈ [−1.95, −1.85]**（ref の実測。V12 表 12 の ELU std 第 2 層 z̄₂/σ₂、中心 −1.90）。
4. **線**: 零点 **z₀ = −103.97**（§1.1 (a)）。予測する量は第 2 層の対で訓練の微分が厳密に 0 の割合 **Z = Φ(z₀/s2 − r)**（§1.1 (b) で記録に合うことを確かめた形）。機能低下の線 z_f = −16.98 の下の割合 F = Φ(z_f/s2 − r) も報告する（判定には使わない）。
5. **3 つの角**: 浅い角（K の下端・r = −1.85）、中心（K の幾何平均・r = −1.90）、深い角（K の上端・r = −1.95）。
6. **分類**: **救う**（SAVES）= 深い角でも Z < 0.1。**救わない**（FAILS）= 浅い角でも Z ≥ 0.1。**際どい**（BORDERLINE）= その間。**二値の当て**: 中心の Z < 0.1 なら救う、そうでなければ救わない。0.1 は §5.6 の DEPTH_ABOVE_LINE（P180 の DEPTH と同じ 1/10）と同じ値で、「当て」と「結果」を同じ尺度で比べるため。

| 腕 | 浅い角: s2 / m / Z | 中心: s2 / m / Z / F | 深い角: s2 / m / Z | 分類 | 二値の当て |
|---|---|---|---|---|---|
| wd3e-2 | 70.7 / −131 / .648 | 196 / −373 / .915 / .97 | 545 / −1063 / .961 | 救わない | 救わない |
| wd1e-1 | 21.2 / −39 / .001 | 58.8 / −112 / .553 / .95 | 163 / −319 / .906 | 際どい | 救わない |
| wd1e0 | 2.1 / −3.9 / 0 | 5.9 / −11.2 / 0 / .16 | 16.3 / −31.9 / 0 | 救う | 救う |

（K の角の値は rule.py の定数から。中心で m = r·s2 が零点 z₀ と重なるのは λ ≈ 0.108 で、wd1e-1 の中心の深さ −112 は零点のすぐ下にある。救う側の端（深い角の Z = 0.1）は λ ≈ 0.51、救わない側の端（浅い角の Z = 0.1）は λ ≈ 0.064、二値の当ての境（中心の Z = 0.1）は λ ≈ 0.18。）

- 規則が当てていないこと: 相対位置 r が WD の下でどこに落ち着くかは導けていない（P180 では λ = 10⁻¹ で +0.59 と ref の −3.39 から大きく浅くなった。この箱の cap12 は −1.08）。親の指定どおり ref の r を使うので、r が浅くなれば規則は深めに外れる。崩れた後に WD が重みを縮めて戻す「戻り」の往復（P180 §2.4、§1.2）も規則の外で、救わない側の λ では Z が釣り合いの値より小さく出うる。

### 2.4 A2 の箱と腕

- **箱**は cifar5p1_mlp_0920 の R/std/lr 1e−4 で、何も変えない（P180 §2.3 と同じ: CIFAR-100、30 課題（奇数 hard 5 クラス × 500 枚、偶数 easy 1 クラス × 500 枚、クラスは再訪しない）、780 更新/課題・batch 32、3072–100–100–100・ReLU、Adam η 10⁻⁴・β (0.9, 0.999)・ε 10⁻⁸、moment と更新番号は課題をまたいで保持、seed 0–9 を R = 10 で束ねる、CUDA graph、threads 2、t29 の fresh 対照）。
- **エンジン**: `src/adamw_dose2_1011_c51.py` に P180 の `adamw_dose_1010_c51.run` を写し、腕の種類（mode）を足す。**coupled**（主）: 宿主の `iv="l2"` と同じ演算 `gr = gr.add(p.detach(), alpha=2.0 * lam)` を勾配の後・m と v の更新の前に置く（全 6 テンソル）。λ = 0 の経路（`nochange`）は宿主の R と演算ごとに同じ。検査専用の mode として **decoupled**（P180 の AdamW と同じ `p.mul_(wd_c)`）と **after**（変異対照: L2 の項を Adam の後に `p.sub_(p_old, alpha=lr * 2 * lam)` で足す）を持つ。テンソルごとの装置上のカウンタ `l2_steps` を graph の中で増やす。
- **腕**（新規、seed 0–9）:

| 腕 | 種類 | λ | 揃う AdamW（§1.3: λ_W = 2λ/1.99×10⁻³） | 役割 |
|---|---|---|---|---|
| **l2c1e-2** | coupled、0 へ | 1×10⁻² | adamw10（λ_W = 10.0） | **主**: L2 Init と同じ λ・同じ coupled、引く先だけが違う |
| l2c5e-3 | coupled、0 へ | 5×10⁻³ | adamw5（λ_W = 5.0） | 副: もう 1 つの揃った強さ |

- **比べるもの**（committed の記録を再利用）: R と L2 Init（λ 1e−2）は P180 と同じ照合（S-nochange・S-host-repro）の上で、AdamW の adamw5・adamw10 は `results/adamw_dose_1010/c51/` の記録（報告の比較のみ）。
- **2×2 の読み**: 引く先 {初期値, 0} × 通し方 {coupled, decoupled}。L2 Init = (初期値, coupled)、AdamW = (0, decoupled) は既知。本走は (0, coupled)。l2c1e-2 と L2 Init は λ も通し方も同じで引く先だけが違う。l2c1e-2 と adamw10、l2c5e-3 と adamw5 は引く先も強さ（全パラメータの中央値の座標で）も同じで通し方だけが違う。
- **fresh**: 宿主の慣習どおり、coupled の腕は同じ λ の coupled L2 で t29 のバッチ列を 780 更新する（fresh_self、P = 初期値・m = v = 0・更新番号 0）。共通 fresh は R の fresh_control。どちらも報告のみ。

## 3. 測る量

### 3.1 A1

- **per_task.csv**（腕ごと、task 2–50）: 宿主（`Engine.diagnostic` と `train_task`）の列をそのまま（online・memo・CE・major_frac・両層の G・Q・厳密 0 の割合 `zero{l}`・全画像で厳密 0 の unit の割合・zmean・zsd・Wnorm・bias・neff・mu2_norm・label/order の hash・finite）。ref の task 1 の行は共通。
- **units_tXX.npz**（git の外）: 宿主の unit ごとの配列（zmean・zsd・gmean・q・zero・neff・Wnorm・bias・argmean・zmean_over_sd、mu2・cos_W2_mu2）に W3 の行ノルムを足す。
- **wd_task.csv**（新規、課題・seed ごと）: `lam`・`c`（16 進）、テンソルごとの `wd_steps`、`wd_c_ok`（graph が読む wd_c が fl₃₂(1 − ηλ) と bit で同じ）、W1–W3 の行ノルムの中央値、b1–b3 のノルム、第 2 層の対の割合（`p2` = z2 > 0、`fline2` = z2 ≤ z_f、`zline2` = 訓練の微分が厳密に 0）、第 1 層の同じ割合。
- 状態の checkpoint（t10・t30・t50 と再開用。git の外）。

### 3.2 A2

- 宿主の per_task.csv（online・train/test 精度、層ごとの dead_frac（死）・zeroout（全閉）・zbar・zsd・zbar_min・mob・eff_rank、w_norm）、fresh_self.csv、**l2_task.csv**（新規: `l2_steps`、b1–b3 のノルム、W1–W3 の Frobenius ノルムと行ノルムの中央値、`relpos_l2`、`p2`（第 2 層の開いた対の割合）、`dead_frac_l2`）。

## 4. endpoint

- **A1**（cap_cifar_ee §4.1 をそのまま）: 主窓 W = t31–50。A_s(t) = online、G_s(t) = 第 2 層の native の微分の全画像・全 unit の算術平均（per_task の `G2`）。**E1 = mean_W A − A_s(1)、E2 = mean_W G − G_s(1)**（task 1 は全腕共通の接頭部）。δ = E_arm − E_ref（seed 内の対応差）。
- **A2**（P180 §4.2）: 後期窓 = hard 21–29 の online の平均（判定）、早期窓 = hard 1–9（IMPAIRED）。

## 5. 判定

### 5.1 A1: λ ごとのラベル（cap_cifar_ee §5.1〜5.5 をそのまま）

- **適用条件**: (A) 3 腕 × 10 seed × task 2–50 が揃い、有限で、全腕の label_hash・order_hash が ref と一致し、全検査が PASS。欠けは `INCOMPLETE`、誤実装は `CHECK_FAILED`、非有限は `DIVERGED`（seed を落として n を縮めない）。(B) ref の主窓の AT_FLOOR が 9/10 以上、かつ ref の E2 の 95% 対応区間の上端 < 0（記録で成立）。満たさなければ全ラベル `NOT_REPRODUCED`。**(C) WD が働いた記録**（新規）: 各 λ 腕・各 seed・task 2–50 の全課題で 6 テンソルとも `wd_steps` = 30,000、かつ `wd_c_ok` = 1。満たさない腕は `INAPPLICABLE`。
- **床**: seed ごとに主窓の最多クラス率 F の平均 F̄_s と SD sF_s から U_s = F̄_s + t₍₁₉,₀.₉₅₎·sF_s/√20。主窓の online の平均 ≤ U_s なら AT_FLOOR。腕は 10/10 床で FLOORED、0/10 で ALIVE、他は SPLIT。
- **区間**: E1・E2 のそれぞれで δ の平均 ± t₍₉, 1−0.05/4₎·s/√10（**97.5%、2 endpoint の Bonferroni**。3 つの λ はそれぞれ規則の当ての対象なので、どれも cap_cifar_ee の主の腕と同じ水準にする）。95% も報告。下端 > 0 で +、上端 < 0 で −、他は 0。SD = 0 は同一点区間と `DEGENERATE_SD`。
- **ラベル**: FLOORED → `COLLAPSED`、SPLIT → `SPLIT`、ALIVE で (+, +) → **`RESCUED`**、(+, 0/−) → `RESCUED_FUNCTION_ONLY`、(0/−, 任意) → `ALIVE_UNRESOLVED`。3 値: RESCUED → RESCUED、SPLIT・RFO・AU → PARTIAL、COLLAPSED → FAIL。
- **フラグ** `IMPAIRED_EARLY`（cap_cifar_ee §5.4）: 全 seed で ref の task 2 の online R_s が F_s(2) より上のとき、腕の task 2 の online が H_s = (R_s + F_s(2))/2 を下回る seed が 6/10 以上。ラベルは変えない。
- **報告のみ**: T½（cap_cifar_ee §5.5、ref との符号検定で LATER / EARLIER / NO_TIMING_DIFF / TIMING_UNDETERMINED）、「戻り」の事象（P180: F(t) < 1/2 が 3 課題以上続いた後に F(t) ≥ 1/2 の課題が来る）の数。

### 5.2 A1 の主判定 `RULE_PREDICTED`（新規）

- **結果の二値**: **救った**（SAVED）= ラベルが `RESCUED`（97.5%）**かつ** `DEPTH_ABOVE_LINE`（§5.6）。それ以外は救わなかった（NOT_SAVED）。
- **一致**: 二値の当て（§2.3 の表）が SAVES で SAVED、または FAILS で NOT_SAVED。
- **`RULE_HOLDS`**（3/3 一致）／**`RULE_PARTIAL`**（2/3）／**`RULE_FAILS`**（1/3 以下）。どれかの腕が INAPPLICABLE・INCOMPLETE・CHECK_FAILED・NOT_REPRODUCED なら `RULE_INAPPLICABLE`。
- **結果の側に DEPTH を入れる理由（算術）**: この箱の `RESCUED` は「全 seed が床の上 ＋ E1・E2 が ref より良い」で、ref が床（G2 = 1.4×10⁻⁵）にいるので、第 2 層の対の 87% が零点の下にいた cap2（主窓 .251、Z .872）でも、床に落ちた seed が 1 つ無ければ `RESCUED` になりえた（実際は 1/10 が床で SPLIT）。decoupled では崩れた重みも縮んで戻るので（§1.2）、深い λ でも床に張り付かずに漂う見込みが高く、ラベルだけでは「零点の上で救った」と「半分死にながら床の上に残った」を分けられない。規則は深さ（Z）の文なので、結果も同じ尺度（Z < 0.1）を入れて比べる。ラベルだけの一致数（`RULE_PREDICTED_LABEL_ONLY`: 当てが SAVES ⇔ RESCUED）も報告する。
- **報告のみ（規則の因子ごとの当たり外れ）**: 各 λ で、主窓の実測の κ1・κ2（行ノルムの unit 中央値 / ‖w‖*）、ρ1、f、r、s2、m、Z、F が §2.3 の帯に入ったか。3 値の分類（救う／際どい／救わない）と 3 値の結果（SAVED／Z が [0.1, 0.5) の NOT_SAVED／Z ≥ 0.5 の NOT_SAVED）の対応表。

### 5.3 DOSE（P180 §5.2）

- 水準の順: L0 = ref（λ = 0）< 0.03 < 0.1 < 1。seed ごとの主窓の online の水準（窓内の課題平均）。隣り合う 3 段のそれぞれで、seed 内の差に正確な両側の符号検定（同値は除く）、3 段で Holm（α = 0.05）。段ごとに UP／DOWN／FLAT。**`DOSE_MONOTONE`**（DOWN が無く UP が 1 つ以上）／`DOSE_PEAKED`（ある UP の段より上に DOWN）／`DOSE_FLAT`（全段 FLAT）／`DOSE_OTHER`。

### 5.4 EQUILIBRIUM（P180 §5.3。3 水準とも τ ≤ 走/10 なので全部で評価）

- seed ごとに、主窓の課題終端の行ノルムの unit 中央値（`Wnorm_l1`・`Wnorm_l2`）を窓で平均し、‖w‖* = η√(d/(2(1 − c)))（d = 3072・100）との比 ρ_eq を取る。log₂ρ_eq の seed 平均の 95% t 区間で、MATCH（区間が [−1, +1] の中 = ×1/2〜×2）／OFF_HIGH（下端 > +1）／OFF_LOW（上端 < −1）／UNRESOLVED。両層 MATCH で **`EQUILIBRIUM_MATCH`**、他は `EQUILIBRIUM_OFF`。

### 5.5 WIDTH（P180 §5.4 を、この箱の ref の生きている間で）

- s = 課題終端の `zsd` の unit 中央値。seed ごとに主窓の s の最小二乗の傾きを、**同じ seed の ref の t1–t3 の傾き**（§1.1 (d)。T½ = 3 なので生きている間は t1–t3）で割った κ_s。
- 腕が ALIVE のときだけ評価（FLOORED → `WIDTH_FROZEN`、SPLIT → `WIDTH_SPLIT`）。層ごとに κ の seed 平均の 95% t 区間で STOPPED（区間が (−0.1, +0.1) の中）／GROWING（下端 > 0.1）／SHRINKING（上端 < −0.1）／UNRESOLVED。両層 STOPPED で **`WIDTH_STOPPED`**、他は `WIDTH_NOT_STOPPED`。
- 0.1 の算術: 3 水準の ‖w‖² の緩和の時間定数は τ/2 ≤ 0.56 課題で、t31 までに残る緩和は e^{−29/0.56} ≈ 0。釣り合いでの期待の傾きは 0、止まっていなければ生きている間の成長率（κ ~ 1）の桁。

### 5.6 DEPTH_ABOVE_LINE（P180 §5.5 の DEPTH を零点の線で。3 水準とも評価）

- Z_λ,s = 主窓の課題終端の `zero2`（第 2 層の対で訓練の微分が厳密に 0 の割合）の窓平均。ζ_s = Z_λ,s / Z_ref,s（ref の主窓は .9977〜.9991）。ζ の seed 平均の 95% t 区間で **`DEPTH_ABOVE_LINE`**（上端 < 1/10）／`DEPTH_AT_LINE`（下端 > 1/10）／`DEPTH_UNRESOLVED`。
- 1/10 は P180 の DEPTH と同じ値（対の 9 割以上が勾配を運ぶ）。

### 5.7 A2

- seed s ごとに後期窓で a_s = W_s − R_s、c_s = W_s − M_s、M_s = (R_s + L_s)/2（L = L2 Init λ 1e−2。閾値は L2 Init − R の差の中点で、L − R は 10/10 seed で正なので算術で決まる）。正確な両側の符号検定、2 腕で a の族と c の族のそれぞれに Holm（α = 0.05）。
  - **`RESCUED`**: c が有意に正。**`PARTIAL`**: RESCUED でなく a が有意に正。**`NO_RESCUE`**: それ以外（a が有意に負なら副フラグ `WORSE`）。
  - **`IMPAIRED`**: 早期窓の W − R の 95% t 区間の上端 < 0。
- **読みのラベル**（主判定ではない。腕ごと、主は l2c1e-2）: d_s = W_s − L_s の正確な両側の符号検定（α = 0.05、Holm なし）で「L2 Init より有意に下」を決め、**`NORM_MATTERS`**（RESCUED かつ L2 Init より有意に下でない ＝ L2 Init 並み → Adam の正規化が効く）／**`TARGET_MATTERS`**（NO_RESCUE ＝ R 並み → 引く先が効く）／**`BOTH`**（それ以外: PARTIAL、または RESCUED で L2 Init より有意に下）。
- **箱のまとめ**: どちらかが RESCUED → `L2C_RESCUES_RELU`、どちらも RESCUED でなくどちらかが PARTIAL → `L2C_PARTIAL_RELU`、他 → `L2C_NO_RESCUE_RELU`。
- **適用条件**: (A) 2 腕 × 10 seed × 30 課題と fresh_self の行が揃い有限、S-nochange・S-host-repro が成立、t29 のバッチ列の sha256 が全腕で一致（欠けは INCOMPLETE、照合の破れは CHECK_FAILED）。(B) R の fresh gap の 95% t 区間の下端 > 0（既知で成立）。(C) 各腕・各 seed・各課題で 6 テンソルとも `l2_steps` = 780。
- **報告のみ**: 揃った AdamW との差（l2c1e-2 − adamw10、l2c5e-3 − adamw5 の後期窓、符号検定）、死 l2・全閉 l2・p2・relpos l2・eff_rank l2・mob l2・行ノルム・bias のノルムの後期窓の seed 中央値と R との対応差、fresh gap（self・共通）。

### 5.8 読み方（登録）

- **`RULE_HOLDS`** → V12 に「救う λ は、釣り合いの幅 × 射影の広がりと相対位置から、走る前に当てられる（MNIST と RL-CIFAR の 2 箱）。救う境は箱で 10 倍違い（MNIST は 0.03〜0.1 の間、RL-CIFAR は 0.1〜1 の間）、射影の広がり（CIFAR の入力の相関で ρ1 ≈ 4.5〜9.6）が決める」と書ける。零点がこの箱では −103.97 であることを併記する。
- `RULE_PARTIAL` で外れが際どい腕だけ → 「明らかな側は当たり、境の位置は規則の帯の中でしか言えない」。外れが救う／救わない腕 → 外れた因子（§5.2 の因子表）を名指しして書く。
- wd1e-1（P180 で救った λ）が救わなかった → 「同じ λ でも箱が違えば救わない」と書ける（規則の予言どおりなら一緒に）。
- ラベルが RESCUED でも DEPTH_ABOVE_LINE でない腕は「零点の上で救った」と書かず、「第 2 層の Z 割が零点の下にいながら床の上に残った」と書く。
- `IMPAIRED_EARLY` の腕では「可塑性を守った」と書かない。
- A2: `NORM_MATTERS` → 「ReLU では引く先より、L2 を Adam の正規化に通すことが効く（decoupled の AdamW は同じ強さで救わない）」。`TARGET_MATTERS` → 「0 へ引く L2 は coupled でも救わず、初期値へ引くことが効く」。`BOTH` → 両方を書き、中点との位置を数で添える。l2c5e-3 が主と違う読みになれば強さの依存として併記する。

## 6. 検査（本 spec の commit の後、本走の前に実行する。許容は導出から決め、値を見て決めない）

`analysis/adamw_dose2_1011/checks.py` → `results/adamw_dose2_1011/checks.json`。各検査は本物のコードで PASS し、列挙した変異（対象ファイルの文字列の一回置換）で FAIL しなければならない。検査の走は検査用 seed（A1 は 100–109、A2 は 100–109）・短縮走・合成入力だけ。例外は S0（A1 の λ = 0 の経路、登録 seed 0–9、task 1–10）・S-nochange と S-host-repro（A2 の λ = 0 の経路と無改変の宿主、登録 seed 0–9）。

| 検査 | 要求 | 変異（FAIL しなければならない） |
|---|---|---|
| **S0** A1 の λ = 0 の bit 一致（依頼の検査） | §2.2 の (a)(b): task 1 の状態の sha256・行・unit の配列が接頭部と、task 2–10 の行・unit の配列と t10 の状態の sha256 が ref の記録と一致 | λ = 0 の腕に λ = 10⁻³ の減衰が入る／Adam の ε が 10⁻⁷／診断が学習の乱数を進める |
| **S-wd** decoupled の単体テスト（A1、合成） | P180 の S-wd と同じ (i) 勾配 0・moment 0 の座標が 1 歩で fl₃₂(p·c)、k 歩で k 回の反復と bit 一致、(ii) 勾配のある座標で「c 倍してから宿主の Adam」を独立に組んだものと bit 一致・カウンタ = k、(iii) coupled（勾配に λθ）は (i) と一致しない | coupled／減衰を Adam の後／bias に掛けない／W3 だけ／c = 1 − λ／λ が 2 倍 |
| **S-torch** torch.optim.AdamW との同値（A1） | seed 100–109 の 2 課題 × 100 更新の実際の勾配列で、本実装と torch.optim.AdamW（weight_decay = 1、foreach = False）の差が P180 の帯 Σ_t [21·u·η|ŝ_t| + 2u|p_t|] の中、second moment は全更新で bit 一致。torch.optim.Adam(weight_decay = 1)（coupled）は帯の外 | 減衰を Adam の後／勾配に λθ |
| **S-branch** 分岐（A1） | 3 腕と λ = 0 の経路の分岐元の状態の sha256 が `t01.pt` と一致、task 2 の label_hash・order_hash が ref の記録と一致、λ 腕の最初の更新が fl(fl(p₀·c) − a)（a は λ = 0 の経路の同じ更新の Adam の項）と bit 一致、task 2 終端の状態は腕ごとに異なる | 減衰が task 3 から／ラベルの流れを腕ごとに初期化 |
| **S-ctrl** 変異対照（A1、依頼の検査） | seed 100（R = 1）の task 1（WD なし）から、ref・ro1（W3・b3 だけ λ = 1）・wd1e0（全テンソル λ = 1）を task 2–4 で走らせ、両層で s_ro(t4) > √(s_ref(t4)·s_wd(t4))（幾何的な中点）かつ s_wd(t4) < s_ref(t4)。見るのは幅と PASS/FAIL だけで online は読まない | ro1 が全テンソルに掛かる |
| **S-graph** （A1） | wd1e0、seed 100–109、task 2 を 4 epoch: graph と eager で行・wd_task・最終の P・m・v が bit 一致、カウンタ = 300 | wd_c を graph の外で差し替える（graph は 1 を読み続ける） |
| **S-resume** （A1） | wd1e0、seed 100–109: epoch 境界の checkpoint からの再開が中断なしと bit 一致 | 再開で wd_c を戻さない |
| **S-nochange** （A2、P180 と同じ） | λ = 0 の経路（seed 0–9・30 課題・graph・fresh）の per_task.csv と fresh_control.csv が無改変の宿主とバイト一致、committed の R と eff_rank 以外の全列でバイト一致（eff_rank は cap_cifar5p1_1007 §7.1 の帯） | λ = 0 の腕に λ = 10⁻² の coupled が入る／lr × 1.0001／Adam の更新番号を課題ごとに戻す |
| **S-host-repro** （A2） | 無改変の宿主 `run('R', iv='l2init:1e-2', seeds 0–9)` が committed の L2 Init を eff_rank 以外の全列で再現 | 照合先を l2init 1e−3 に／seed 10–19 に |
| **S-coupled-host** （A2・coupled の実装） | 本エンジンの l2c1e-2 と l2c5e-3（seed 100–109、30 課題、graph、fresh）の per_task.csv が、無改変の宿主 `run('R', iv='l2:<λ>')` の per_task.csv とバイト一致、fresh_self の値が宿主の fresh_control の値と一致 | 2λθ を λθ に／L2 を b に掛けない／L2 の項を m だけに入れる |
| **S-mut** 変異対照（A2、依頼の検査） | seed 100（R = 1）の 2 課題（1,560 更新）の実際の勾配列を流し、(i) after（L2 を Adam の後）と decoupled（λ_W = 2λ）の差が帯 Σ_t 6u(|p_t| + η|ŝ_t|) の中、(ii) coupled と decoupled の差は要素の 99% 以上で帯の外。加えて (iii) after(10⁻²) と decoupled(2×10⁻²) を seed 100 で 30 課題走らせ、後期窓と行ノルムを R（較正の走の seed 100）と並べて報告（判定なし） | after の L2 を Adam の前に戻す（= coupled）／after で更新後の p を使う |
| **S-graph-c51**・**S-fresh-c51** （A2） | P180 の S-graph・S-fresh を coupled に: graph と eager がバイト一致、fresh_self の開始で P = 初期値・m = v = 0・更新番号 0、l2_steps = 更新の数 | L2 の λ を graph の外で書き換える／fresh_self で L2 を切る |
| **S-verdict** 判定 | 合成の shard で、A1 の全ラベル（5 ラベル・3 値・IMPAIRED_EARLY・RULE の 4・DOSE の 4・EQUILIBRIUM の層 4 と水準 2・WIDTH の層 4 と水準 4・DEPTH の 3）と A2 の全ラベル（3・WORSE・IMPAIRED・読み 3・まとめ 3）と INAPPLICABLE・NOT_REPRODUCED・INCOMPLETE が設計どおりに出る。t 分位点が公表表と一致し、n = 10 の符号検定の p が正確な二項の値と一致 | RULE の結果から DEPTH を外す／床の t 分位点を 3 に／Bonferroni を外す／E1 を腕自身の task 2 から測る／WIDTH の ref の傾きを主窓に／中点を R に |
| **S-cli** | 各腕の出力が登録した対応の in-process の走と一致、腕名と λ の対応が登録どおり、既定値（A1: task 2–50・400 epoch、A2: 30 課題・lr 1e−4・threads 2）、provenance（spec のパス・登録 commit・checks.json の sha256）。登録先への出力は checks.json の all_pass・ソースの sha256 の一致・git clean でなければ拒否 | λ の対応の取り違え／既定 30 課題 |
| **S-cost** | A1: wd1e0 の短縮走で ms/更新・GPU メモリ（nvidia-smi の差分と torch の最大確保）・RSS を 1 本で測る。A2: l2c1e-2 の短縮走で同じ | — |

**許容の導出**: bit 一致の検査は許容 0。S-torch の帯は P180 §6・§13 の 2 と同じ式（n_op = 21、u = 2⁻²⁴）。S-mut の帯: after と decoupled は同じ m・v（同じ勾配列）から同じ ŝ を作り、違うのは減衰と引き算の丸めだけ（after: p − ηŝ と 2ηλ·p_old とその引き算の 3 回、decoupled: p·c と − ηŝ の 2 回、c の丸め 1 回で計 6 回）なので |Δp_K| ≤ Σ_t 6u(|p_t| + η|ŝ_t|)。coupled との差は 1 歩あたり η·2λ|p|(1/√v̂ − 1) の桁で（√v̂ ≈ 2×10⁻³ なら 1 − 2ηλ の decoupled の約 500 倍の引き）、1,560 歩で帯の 10³ 倍以上になる見込み（S-mut の出力に比を書く）。S-ctrl の閾値は ref と全テンソルの λ 腕の幅の幾何的な中点で、値を見て決めない。検査で設計の誤りが見つかった場合は、本走の前に本 spec へ追補として書き、commit してから走る。

## 7. 予測（新しい腕を 1 本も回す前、実装を書く前。この commit で固定する）

Claude の列は本 spec の起草者（Opus 5.5、2026-10-11 13:30 JST）。この系統で私は P180 で「止まった幅の深さ」を見誤って 3×10⁻² を RESCUED .74 と外した（第 2 層の射影の広がりを伸ばさなかった）。本 spec の規則はそこを入れたが、相対位置 r が WD の下でどこに落ち着くかは導けていない。A2 は P180 の AdamW の結果と較正の √v̂ を見ているので独立な予測ではない。Issa と Fable の列は空欄（走る前に記入する）。

略号: RFO = RESCUED_FUNCTION_ONLY、AU = ALIVE_UNRESOLVED、他 = INAPPLICABLE・NOT_REPRODUCED・CHECK_FAILED・INCOMPLETE・DIVERGED をまとめた桶。

### 7.1 ラベルの項目

| # | 項目 | Claude | Issa | Fable |
|---|---|---|---|---|
| A1-1 | **RULE_PREDICTED** | **HOLDS .55** ／ PARTIAL .38 ／ FAILS .05 ／ INAPPLICABLE .02 | | |
| A1-2 | wd3e-2 のラベル（97.5%） | RESCUED .28 ／ SPLIT .28 ／ COLLAPSED .22 ／ RFO .12 ／ AU .05 ／ 他 .05 | | |
| A1-3 | wd1e-1 のラベル | RESCUED .55 ／ SPLIT .18 ／ RFO .12 ／ COLLAPSED .08 ／ AU .03 ／ 他 .04 | | |
| A1-4 | wd1e0 のラベル | RESCUED .85 ／ SPLIT .04 ／ COLLAPSED .03 ／ RFO .03 ／ AU .02 ／ 他 .03 | | |
| A1-5 | DOSE | MONOTONE .75 ／ PEAKED .12 ／ OTHER .11 ／ FLAT .02 | | |
| A1-6 | EQUILIBRIUM（wd1e0） | MATCH .40 ／ OFF .60（層別: MATCH .55・OFF_HIGH .35・OFF_LOW .03・UNRESOLVED .07） | | |
| A1-7 | EQUILIBRIUM（wd1e-1） | MATCH .35 ／ OFF .65（層別: MATCH .50・OFF_HIGH .35・OFF_LOW .05・UNRESOLVED .10） | | |
| A1-8 | EQUILIBRIUM（wd3e-2） | MATCH .30 ／ OFF .70（層別: MATCH .45・OFF_HIGH .20・OFF_LOW .20・UNRESOLVED .15） | | |
| A1-9 | WIDTH（wd1e0） | STOPPED .80 ／ NOT_STOPPED .10 ／ FROZEN・SPLIT .10 | | |
| A1-10 | WIDTH（wd1e-1） | STOPPED .55 ／ NOT_STOPPED .15 ／ FROZEN・SPLIT .30 | | |
| A1-11 | DEPTH（wd1e0 ／ wd1e-1 ／ wd3e-2） | ABOVE_LINE .85 ／ .25 ／ .05（残りは AT_LINE と UNRESOLVED に 3:1） | | |
| A2-1 | l2c1e-2 のラベル | NO_RESCUE .43 ／ PARTIAL .33 ／ RESCUED .22 ／ 他 .02 | | |
| A2-2 | l2c5e-3 のラベル | NO_RESCUE .42 ／ PARTIAL .36 ／ RESCUED .20 ／ 他 .02 | | |
| A2-3 | **読みのラベル（l2c1e-2）** | **TARGET_MATTERS .41** ／ BOTH .37 ／ NORM_MATTERS .20 ／ 他 .02 | | |
| A2-4 | A2 のまとめ | L2C_NO_RESCUE_RELU .38 ／ L2C_PARTIAL_RELU .33 ／ L2C_RESCUES_RELU .27 ／ 他 .02 | | |

### 7.2 二値の項目

| # | 主張 | Claude | Issa | Fable |
|---|---|---|---|---|
| S1 | A1 の (B) が成り立つ | .99 | | |
| S2 | S0 が成り立つ（ref の記録をこの環境で bit で再現） | .85 | | |
| S3 | A1 の (C) が 3 腕とも成り立つ | .97 | | |
| S4 | IMPAIRED_EARLY がどの腕にも立たない | .80 | | |
| S5 | wd1e0 の主窓の online（seed 平均）≥ 0.70 | .60 | | |
| S6 | wd1e-1 の主窓の Z が [0.1, 0.9] | .60 | | |
| S7 | wd3e-2 に「戻り」の事象が 1 seed 以上 | .55 | | |
| S8 | wd1e0 の主窓の s2（unit 中央値・seed 平均）が規則の帯 [2.1, 16.3] の中 | .65 | | |
| S9 | wd1e-1 の主窓の s2 が [21.2, 163] の中 | .70 | | |
| S10 | wd1e0 の主窓の r2（unit 中央値の z̄₂/σ₂・seed 平均）が −1.85 より浅い | .75 | | |
| S11 | wd1e-1 の第 1 層の κ1（主窓の行ノルム / ‖w1‖*）が [1.6, 2.3] の中 | .45 | | |
| S12 | RULE_PREDICTED_LABEL_ONLY（ラベルだけの一致）が 3/3 | .30 | | |
| C5 | l2c1e-2 に WORSE が立つ | .30 | | |
| C6 | l2c1e-2 に IMPAIRED が立つ | .55 | | |
| C7 | l2c1e-2 の後期窓の死 l2 が R より低い（seed 内の正確な符号検定で p < 0.05） | .75 | | |
| C8 | l2c1e-2 の eff_rank l2 が R より低い（同） | .55 | | |
| C9 | l2c1e-2 の後期窓が adamw10 より高い（同） | .60 | | |
| C10 | l2c5e-3 の後期窓が adamw5 より高い（同） | .55 | | |
| C11 | l2c1e-2 の後期窓が L2 Init より低い（同） | .85 | | |

### 7.3 採点の規則（登録。P180 §7.3 と同じ）

- ラベルの項目は、実現したラベルに付けた確率、最頻ラベルの的中、多クラスの Brier 得点 Σ_k(p_k − 1[k = 実現])²。「他」は 1 つの桶。名前を書かれていないラベルの確率は 0。
- 二値の項目は真偽と binary Brier (p − o)²。評価できない項目（WIDTH は ALIVE のときだけ、など）は「未採点（理由）」。
- S1・S2 は記録から答えがほぼ分かっている。A2 の項目は P180 の AdamW の結果と較正を見ているので、独立な予測の成功として数えない。

## 8. 予算と実行（時間の見積もりは書かない）

- **親の確認**: 検査が all_pass になったら、本走の前に親（Fable）へ λ・規則の当て・A2 の λ・判定ラベル・予測・GPU の空き・走の本数を送り、返事（go か修正）を待つ。返事が来るまで本走を起動しない。go の後、本走の直前に台帳 `~/Projects/obsidian-research-data/push_direction_proof_0930/PREDICTIONS.md` に P 番号（直近の続番、flock で追記）で登録する。
- **A1（GPU）**: 3 腕 × R = 10 × task 2–50。1 本 0.78 ms/更新（§1.4。診断を除く）。GPU は 1 本で計算に飽和するので、同時に走らせても全体の処理量は変わらない。**同時本数は GPU メモリから算術で決める**: N = min(3, ⌊(16,311 MiB − 起動直前の nvidia-smi の使用量 − 2,048 MiB) / (1.2 × S-cost の 1 本の使用量)⌋)（2,048 MiB はデスクトップの揺れと CNN の走（別の agent、`cnn_center_relu_1011`）の起動の余地）。起動直前の使用量には CNN の走の分が入る。加えて RAM: MemAvailable − 6 GiB ≥ 1.2 × N × S-cost の RSS。マシン全体の重い python（`pgrep -c -f '\.venv/bin/python'` と proj_004_drift の .venv の本数を両方数える。自己一致に注意）を起動前に記録する。各腕は `nohup setsid` で切り離し、ログは `results/adamw_dose2_1011/logs/<腕>.log`。STOP ファイルで epoch 境界に checkpoint して止まる。
- **A2（GPU）**: 2 腕 × 20 秒程度。S-nochange・S-host-repro・S-coupled-host・S-mut は検査の段で走る。A1 と同時に走らせてよい（GPU のメモリの算術に含める）。
- 起動は、実装の commit・全検査の all_pass・HEAD の push・親の go・台帳の登録を確かめてから。走っている途中の効果量を見て、設計・腕・予測を変えない。監視は完走した課題の番号・失敗の数・ログの末尾だけを見る。
- python は `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`（3.12.3・torch 2.13.0+cu130）。worktree の `data` は `~/Projects/claude/proj_004_drift/data` への symlink（`/data` は .gitignore 済み。片付けで辿らない）。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- A1: `results/adamw_dose2_1011/rlcifar/<腕>/`（per_task.csv・wd_task.csv・provenance_*.json・done.json は git。units_tXX.npz・checkpoint・ログは git の外）。S0 の走は `results/adamw_dose2_1011/rlcifar/_s0/`。
- A2: `results/adamw_dose2_1011/c51/<腕>/`（per_task.csv・fresh_self.csv・l2_task.csv・provenance.json）。照合の走は `c51/_nochange/`・`_hostrepro/`・`_coupledhost/`・`_mut/`。
- 集約と判定: `analysis/adamw_dose2_1011/verdict.py` → `results/adamw_dose2_1011/{summary.md, verdict.json, verdict.csv, rule.csv, paired.csv, dose.csv, equilibrium.csv, width.csv, depth.csv, c51_paired.csv, secondary.csv, predictions.csv}`、検査は `checks.json`。
- git の外のファイル（units・checkpoint・検査の作業場所 `results/_checks_adamw_dose2_1011/`・大きいログ）は `~/Projects/obsidian-research-data/adamw_dose2_1011/` へ移し、`results/adamw_dose2_1011/backup_manifest.json`（source・backup・bytes・sha256）を commit する。`data` の symlink は辿らない。
- 結果を commit した日のうちに origin/main へ merge して push し、worktree とブランチを消す（親の依頼で、片付けまで）。vault `可塑性喪失/測定/` に結果ノート、走の履歴と論点マップに 1 行。V12 は触らない。

## 10. 言わないこと・読みの上限

- **AdamW が可塑性の喪失全体を防ぐとは言わない。** A1 は RL-CIFAR × ELU→ELU・std・この 3 水準・task 2 からの減衰・50 課題、A2 は 5+1 CIFAR × ReLU・std・lr 1e−4・30 課題・この 2 水準。
- 規則は P180 の κ（×1.6〜2.3）と、この箱の既知の状態から取った射影の広がりと ref の相対位置を使う。**相対位置を WD の下で導いたのではない**。`RULE_HOLDS` でも「r は ref の値で十分だった」までで、r の導出は言わない。
- 零点 −103.97 と機能低下の線 −16.98 は、この箱の実装（native `F.elu`・CUDA・subnormal を保つ）の性質で、他の実装や箱に広げない。
- online は学ぶ速さで、記憶の容量ではない。RESCUED は相対の規則（共通の task 1 からの低下を ref と比べる）。
- A2 の読みは λ ∈ {5×10⁻³, 10⁻²} の coupled L2 についてで、「L2 一般」「WD 一般」には広げない。強さの揃え方（全パラメータの √v̂ の中央値）は 1 つの選び方で、層ごとには揃っていない（§1.3）。
- 中心化（扉 C・H）との比較、SiLU・GELU は範囲外。

## 11. 依頼文からの逸脱（登録時点で開示）

1. **零点**: 依頼は「ELU の float32 の訓練の微分の零点 z = −16.64」。この箱の native `F.elu` の訓練の微分の零点は −103.97 で、−16.64 は MNIST の宿主の値（§1.1 (a)(b)）。規則の零点は −103.97 を使い、依頼の「その手前で起きる機能低下」は出力が −1 に張り付く線 −16.98（CUDA）として報告する。−16.64（または −16.98）を零点にした場合、λ = 1 は「際どい」（深さ −3.9〜−32）、救う λ は 2 以上になる。
2. **RULE_PREDICTED の結果の側**: 依頼は「各 λ の事前予測（救う／救わない）が結果と一致した数」。結果を「RESCUED かつ DEPTH_ABOVE_LINE」にした（§5.2 の算術）。ラベルだけの一致数も報告する。
3. **際どい腕の二値の当て**: 3 値の分類（救う／際どい／救わない）を二値の採点に入れるため、際どい腕は中心の Z で二値に当てる（wd1e-1 は救わない）。
4. **分岐点**: P180 は減衰を task 1 から入れた。本 spec は cap_cifar_ee の上限と同じく task 2 から入れる（cap_cifar_ee の登録規則の E1・E2・IMPAIRED_EARLY が共通の task 1 を前提にするため、ref の記録を bit で再利用できるため、cap12 と同じ分岐点で比べられるため）。3 水準とも τ ≤ 1.1 課題で、主窓（t31–50）には分岐点の影響が残らない。
5. **A2 の第 2 の λ**: 依頼は「可能なら P180 の AdamW と実効減衰 ηλ が揃う値」。coupled の実効の減衰は座標ごとの √v̂ で割られるので、全パラメータの √v̂ の中央値（較正 seed、§1.3）で揃えた。主の λ = 10⁻²（L2 Init と同じ）がちょうど adamw10 と揃い、もう 1 つを adamw5 と揃う 5×10⁻³ にした。
6. **A1 の Bonferroni**: cap_cifar_ee は主の腕（cap12）だけ 97.5%、副の腕は 95%。本 spec は 3 水準とも規則の当ての対象なので、3 つとも 97.5%（各腕の 2 endpoint）にした。3 水準をまたぐ補正はしない（RULE は腕ごとの一致の数で、全部が当たる `RULE_HOLDS` は各腕の主張の積集合）。
7. **WIDTH の基準の傾き**: P180 は ref の t1–10。この箱の ref は T½ = 3 なので t1–t3。
8. **変異対照**: A1 は依頼どおり 1 seed（seed 100）で読み出しだけの WD。A2 は依頼の「L2 の勾配を Adam の後に足す（＝decoupled に等しくなること）」を、同じ勾配列での丸めの帯の中の一致（判定）と 1 seed の 30 課題の走（報告）で行う。

## 12. 登録予定の判定ラベル一覧（台帳には、親の go の後・本走の直前に登録する）

**A1**（λ ∈ {0.03, 0.1, 1}、どれも 97.5%）
- 主: `RULE_HOLDS` ／ `RULE_PARTIAL` ／ `RULE_FAILS` ／ `RULE_INAPPLICABLE`（報告: `RULE_PREDICTED_LABEL_ONLY` の数、因子ごとの帯の当たり）
- λ ごと: `RESCUED` ／ `RESCUED_FUNCTION_ONLY` ／ `ALIVE_UNRESOLVED` ／ `SPLIT` ／ `COLLAPSED` ／ `INAPPLICABLE` ／ `NOT_REPRODUCED` ／ `INCOMPLETE` ／ `CHECK_FAILED` ／ `DIVERGED`（3 値: `RESCUED` ／ `PARTIAL` ／ `FAIL`）、フラグ `IMPAIRED_EARLY`
- `DOSE_MONOTONE` ／ `DOSE_PEAKED` ／ `DOSE_FLAT` ／ `DOSE_OTHER`
- λ ごと: `EQUILIBRIUM_MATCH` ／ `EQUILIBRIUM_OFF`（層ごと `MATCH` ／ `OFF_HIGH` ／ `OFF_LOW` ／ `UNRESOLVED`）
- λ ごと: `WIDTH_STOPPED` ／ `WIDTH_NOT_STOPPED` ／ `WIDTH_FROZEN` ／ `WIDTH_SPLIT`（層ごと `STOPPED` ／ `GROWING` ／ `SHRINKING` ／ `UNRESOLVED`）
- λ ごと: `DEPTH_ABOVE_LINE` ／ `DEPTH_AT_LINE` ／ `DEPTH_UNRESOLVED`

**A2**（coupled L2、λ ∈ {10⁻², 5×10⁻³}）
- 腕ごと: `RESCUED` ／ `PARTIAL` ／ `NO_RESCUE`、副フラグ `WORSE`、フラグ `IMPAIRED`
- 読み（主は l2c1e-2）: `NORM_MATTERS` ／ `TARGET_MATTERS` ／ `BOTH`
- まとめ: `L2C_RESCUES_RELU` ／ `L2C_PARTIAL_RELU` ／ `L2C_NO_RESCUE_RELU`
- 適用外: `INCOMPLETE` ／ `CHECK_FAILED` ／ `NOT_REPRODUCED` ／ `INAPPLICABLE`

**予測の項目**: §7 の A1-1〜A1-11、A2-1〜A2-4、S1〜S12、C5〜C11。Issa と Fable の列は空欄。
