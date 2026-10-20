# kappa_reload_cnn_1020 — CNN の最終隠れ層で「装填」を測る（κ_k・切替の押し・Keep・初期化の押し・戻り）

状態: **登録（この commit）。計算コード・検査・本計算はこの commit より後。** 事後解析（保存済みの checkpoint を読むだけ）。新しい学習は、保存済みの Adam 状態からの再生（元の走と bit 一致する再現）だけ。

作成: 2026-10-20 18:50 JST / 設計・判断: Claude（Fable 5.1、この worktree のセッション）/ 実装・計算: Opus（サブエージェント）/ 独立の導出・批評・監査: Codex gpt-6-astra（Astra）。
run: `kappa_reload_cnn_1020` / branch: `claude/kappa_reload_cnn_1020` / worktree: `wt/kappa_reload_cnn_1020` / 起点 `origin/main = b7808018`。
親: V13 機構論文の Fable セッション。依頼主: Issa（「CNN の 0. 装填を別セッションで、Astra＋Fable で」）。

## 0. 一行

1 隠れ層 RL-MNIST で測った「学んだ読み出し κ が課題の切替ごとに押しを装填し直し、押しは沈める側、初期化の押しとは桁が違う」（[[kappa_reload_1020]]・[[reload_dial_1020]]）を、RL-CIFAR CNN の最終隠れ層（fc 100→100 の出力）で **同じ定義のまま** 測り、V13 の部品の表の「CNN × 装填」の欄を埋める。合わせて、conv 層と中間 fc 層でも押しの大きさが課題を重ねても保たれるか（Keep）を押しの大きさの比で見る。

## 1. 既知（登録前に見たもの。再発見しない）

- MLP（`results/kappa_reload_1020/summary.md`・`results/reload_dial_1020/summary.md`）: κ_k ≥ 0.83κ₁（全 10 走）、課題 2 以降の κ の 82〜95% は特徴の側、期待の押しは生きた unit の 67〜100% で沈める側、自己形との符号一致 90〜100%（leaky は課題 4 以降 40〜52%）、実際の 200 更新で 82〜100% が沈む、正側への戻りは 1,000 unit 中 4、初期化（乱数読み出し）の押しは切替の 1/100 以下で向きも半々、押し ≈ 0.2 × (log K − L)、Adam の最初の 1 歩は 52〜58% しか沈まない（4,000 更新/課題の運動量）、κ の下界 κ ≥ [N(log K − L) − B]/‖F‖² は 328 状態で破れ 0、κ/κ_lb 2.6〜3.5。
- CNN（`results/cnn_drive_verify_1009/summary.md`・`results/c1_direction_1010/summary.md`）: c1 の切替の押し G は自己形に従わない（定数モード＝下流の押しの透過が LR では浮かせる側 0.86〜0.95 で支配）。LR の c1 の沈める側は t1 0.50 → t10 0.10 → t30 0.40、c2 は t10 以降 1.00。Snake の c2 は谷の則（seat = 2αz̄ の AUC 0.94）。最初の Adam 1 歩の有意な反転は c1 ≤ 1.7%・c2 3〜4%（Adam の運動量はほぼ消えている）。R は課題 3 で死ぬ（死後 G ≡ 0）。再走 t1→t2 は保存済み t02 と 40 走すべて bit 一致（K7）。
- 見ていないもの: f1・f2（最終隠れ層）のどんな量も、この箱では一度も計算されていない。κ も同じ。

## 2. 箱（読むだけ）

- 網（宿主 `src/rlcifar_cnn_0908.py`）: conv5×5 3→16（pad 2）→ φ → maxpool2 → conv5×5 16→16 → φ → maxpool2 → flatten 1024 → fc 1024→100 → φ → fc 100→100 → φ → head 100→10。全 bias あり。サイトは c1 c2 f1 f2。**最終隠れ層 = f2 の出力 a4 = φ(z4)（100 unit）、読み出し = head（W5 10×100、b5）**。
- 1,200 枚（seed ごとの固定部分集合、[0,1]、正規化なし）、課題ごとに画像ごと独立の一様乱数ラベル、batch 16、400 epoch = 30,000 更新/課題、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8、bias 補正は全体 step tc。
- 腕と保存済みの状態（課題末の P・Adam の m/v・tc・活性化の状態）:
  - **LR**（leaky 0.1）seed 10〜19、t ∈ {1,2,3,5,10,20,30}: `~/Projects/obsidian-research-data/cnn_drive_verify_1009/results/cnn_drive_verify_1009/LR/ckpt/LR_seed<s>_t<tt>.pt`（70 個）。
  - **R**（ReLU）seed 10、t ∈ {1,2,3}: 同 `R/ckpt/`（3 個。課題 3 で死ぬ参照）。
  - **適応 Snake 4 腕** SNA（c = 0.6 全層）・SNAc3（c = 3 全層）・CV06FC3（conv 0.6・fc 3）・CV3FC06（conv 3・fc 0.6）、seed 10〜19、t ∈ {1,2,3,5,10,20,30,40,50}: `~/Projects/obsidian-research-data/sna_cnn_cause_1009/results/sna_cnn_cause_1009/A/ckpt/<arm>_seed<s>_t<tt>.pt`（360 個）。α = clip(c/√V, 0.005, 3) は checkpoint の V・ada・fixA・cval から復元（`src/cnn_drive_verify_1009.py` の `State` と同じ）。
  - **初期値 t = 0**: `CN.init_params(seed)`（乱数 head）。
- 再生のエンジン: `src/sna_cnn_cause_1009.py` の `Bundle`（Snake）と `src/cnn_drive_verify_1009_lr.py` の `BundleLR`（LR・R）。checkpoint の読み込みと stream の進め方は `src/cnn_drive_verify_1009.py` の `replay` と同じ。既存コードの `SNA_ROOT`（消えた worktree）は、本 worktree の `src/` と上の退避先を指すように **本 run のモジュール側で** 差し替える（既存ファイルは編集しない）。
- 精度: 事後の量は float64（GPU、cuDNN TF32 off）。再生は訓練と同じ float32 のエンジン（bit 一致のため）。再生の記録（平均前活性）は float64 で積む。

## 3. 定義（最終隠れ層 f2 は MLP の `analysis/kappa_reload_1020/measure.py` のまま。記号は V12 付録 C.3）

課題 k の末の状態（checkpoint t = k）で、旧ラベル y_n（課題 k）と新ラベル（課題 k+1、一様 iid）について:

- 入力 x_n = a3_n = φ(z3_n) ∈ R^100（f1 の出力）、x̄ = mean_n x_n、**明るさ K_n = x̄·x_n + 1**。
- 前活性 z4_n = W4 x_n + b4、**位置 m_i = mean_n z4_in = W4[i]·x̄ + b4[i]**、幅 s_i = sd_n z4_in、r_i = m_i/s_i、p⁺_i = mean_n 1[z4_in > 0]。傾き g_in = φ′(z4_in)、出力 φ_in = a4_in。
- logit ℓ_n = W5 a4_n + b5、π_n = softmax(ℓ_n)。読み出し v = W5、**クラスで中心化した読み出し v^c = W5 − mean_c W5**（unit ごと）。b5^c = b5 − mean b5。
- **F = Σ_n (y_n − 1/K) ⊗ a4_n ∈ R^{K×H}**（K = 10、H = 100、y_n は旧ラベルの one-hot）。**κ_k = ⟨v^c, F⟩/‖F‖²**（最小二乗 head ≈ v⁰ + κF の κ）。unit ごとの κ_i = Σ_c v^c_ci F_ci / Σ_c F_ci²。学んだ読み出しの大きさ ‖κF‖/√H。現課題の構造で説明できる割合 ⟨v^c,F⟩²/(‖v^c‖²‖F‖²)。
- **分解**（参照状態 ref = 一つ前の保存境界。t1 → 初期値、t2 → t1、t3 → t2、t5 → t3（2 課題分）、t10 → t5（5）、t20 → t10（10）、t30 → t20（10）、t40 → t30、t50 → t40）: F_ref = Σ_n (y_n − 1/K) ⊗ a4^ref_n（**現在の**旧ラベル y_n と参照状態の特徴）。κ = κ_chance + κ_feat + κ_read、κ_chance = ⟨v^c_ref, F_ref⟩/‖F‖²、κ_feat = ⟨v^c_ref, F − F_ref⟩/‖F‖²、κ_read = ⟨v^c − v^c_ref, F⟩/‖F‖² [恒等式]。参照が現在のラベルと独立なので κ_chance は偶然の分。span が 1 課題でない境界はそのことを表に書く。
- **補題の検査**: A = ⟨v^c, F⟩ = Σ_n [(ℓ_{n,y_n} − ℓ̄_n) − b5^c_{y_n}] [恒等式]、A ≥ N(log K − L) − B（Jensen、L = 旧ラベルの平均 CE、B = Σ_c (n_c − N/K) b5^c_c）、κ_lb = [N(log K − L) − B]/‖F‖²。恒等式の相対誤差、破れ（κ < κ_lb − 1e−12）の数、κ/κ_lb。
- **期待の押し（自分の重みを通る分、MLP の S_exact）**: S_exact_i = Σ_n K_n g_in ⟨v_i^c, π_n⟩。全バッチ GD の一歩の期待は E[Δm_i] = −(η/N) S_exact_i、**S > 0 が沈める側**。G_own_i = S_exact_i/N = x̄·∂L_u/∂W4[i] + ∂L_u/∂b4[i]（L_u = (1/N)Σ_n[logsumexp ℓ_n − mean_c ℓ_nc]、一様ラベルの期待は厳密）。
- **自己形**: S_self_i = κ_k (1 − 1/K) Σ_n K_n g_in φ_in（H1 の読み出し模型 v = v⁰ + κF の対角項）。旧ラベルの one-hot を当てた形 S_oh_i = Σ_n K_n g_in v^c_{i,y_n}。
- **全パラメータを通る押し（drive_verify の G）**: G_full_i = ⟨∇_θ m_i, ∇_θ L_u⟩（θ = 10 テンソル全部、Euclid 計量）。SGD の一歩の一次の期待は −η G_full。**G_up = G_full − G_own**（上流 W1〜W3・b1〜b3 が x̄ を動かす分）。c1 は G_up = 0。
- 他の層: f1 は x = h2（pool 後の c2 の 1024 次元）、m_i = mean_n z3_in、G_own・G_full を同じ形で（κ・S_self は head だけの量なので f1 には無い）。c1・c2 は m_j = 画像 × 位置の平均（zero-pad 込み、drive_verify と同じ）、G_full は同じ量（JVP で 4 層 232 unit を一度に出し、c1・c2 は保存済みの `results/cnn_drive_verify_1009/LR/measure/*.npy` の `G_full` と照合する）。G_own は conv では ∂m_j/∂W2[j] = 平均パッチ（zero-pad 込み）、∂m_j/∂b2[j] = 1。
- **生きた unit**: z > 0 の画像（× 位置）が 1 つ以上（MLP と同じ。leaky・Snake は常に勾配が流れるが基準は揃える。R の死んだ unit は G ≡ 0 で除く）。割合は生きた unit で数える。
- **実際の切替（再生）**: checkpoint t から課題 t+1 をエンジンで再生（実ラベル・実 batch 順、Adam の m/v・tc・α の状態を引き継ぐ）。step s ∈ {0,1,2,5,10,20,50,100,200} で 4 層の m（232）・p⁺・s を記録。Δm_s = m(s) − m(0)。沈める側 = Δm_200 < 0。最初の 1 歩 = Δm_1 < 0。−G_full（f2 は −G_own も）との符号一致。**任意（GPU の空きで）**: 課題の残り（s ∈ {750, 7500, 15000, 30000}）も記録し、課題の中の戻り（Σ Δm_rest / Σ|Δm_200|、沈んだ unit）を出す。LR は全境界、Snake は t ∈ {1, 10, 30} から。
- **初期化**: t = 0（乱数 head）で同じ量（κ は定義できるが偶然の値。S_exact・G_full・G_own）と、初期値から課題 1 の最初の 200 更新の再生。
- **Keep**: 層ごとに med_i |G_full_i|（生きた unit）の課題 t の値 / 課題 1 の値。f2 は |S_exact|/N でも。κ_k/κ₁ と並べる。
- **戻り**: (g1) 保存境界の列（t ∈ {1,2,3,5,10,20,30(,40,50)}）で、ある境界で p⁺ < 1/2（または r < 0）だった unit が後の境界で p⁺ ≥ 1/2（r ≥ 0）に戻った数（層ごと・腕ごと・seed 合計）。(g2) 再生の課題の中の戻り（任意の分）。

## 4. 測るもの（対象: LR 70 ＋ R 3 ＋ Snake 360 ＝ 433 状態 ＋ 初期値 50）

(a) κ_k、κ_k/κ₁（seed ごと）、分解、‖κF‖/√H、説明できる割合、κ_i の四分位と負の割合、旧ラベルの正答率と正解クラスの予測確率。
(b) 補題: 恒等式の相対誤差の最大、破れの数（全腕・全境界）、κ/κ_lb の四分位。
(c) 期待の押し: 層ごとの G_full > 0 の割合、f1・f2・c2 の G_own > 0 の割合と G_up の分担（|G_up|/(|G_own|+|G_up|) の中央値）。f2 の S_self と S_exact の符号一致（生きた unit、両方非零）、S_self/S_exact の四分位、S_oh と S_exact の一致、Spearman。Snake は S_self > 0 の割合も（谷の下では負になりうる）。
(d) 再生: 層ごとの Δm_200 < 0 の割合、Δm_1 < 0 の割合、−G_full（f2 は −G_own も）との符号一致、沈み −Δm_200 の中央値と 沈み/幅。
(e) Keep: 層ごとの med|G_full| の比（t / t1）。f2 の med|S_exact|/N の比。
(f) 初期化: med|S_exact|（f2、t = 0）/ med|S_exact|（t = 1）、層ごとの med|G_full| の比、t = 0 の沈める側の割合（期待・200 更新）。
(g) 戻り: (g1) 境界の列での戻った unit の数。(g2) 任意。
(h) **回さない**（提案だけ）: ダイヤル（課題 1 の精度 20/50/80/95% の checkpoint から 5 通りの新ラベルへ 200 更新、LR 1〜2 seed）。新しい GPU の走なので (a)〜(g) のあとに親に提案し、Issa の了承で回す。5+1 CNN（実ラベル）は対象外（一様ラベルの期待が使えない）。

## 5. 検査（本計算の前に全部 PASS）

- **K1 閉じた式 = autograd**: G_own（x̄·∂L_u/∂W4[i] + ∂L_u/∂b4[i]）が S_exact_i/N と相対 1e−10 で一致（f2、f1 も同じ形）。
- **K2 G_full の照合**: JVP で出した c1・c2 の G_full が、保存済みの `results/cnn_drive_verify_1009/LR/measure/measure_LR_s10_t01.npy` などの `G_full` と相対 1e−8 で一致（LR 2 状態、Snake 1 状態は `results/cnn_drive_verify_1009/measure/` があれば）。
- **K3 恒等式**: A の 2 つの式の相対誤差 ≤ 1e−10（全状態で記録し、最大を報告）。
- **K4 分解**: κ_chance + κ_feat + κ_read = κ（相対 1e−10）。
- **K5 再生の忠実度**: LR seed 10〜19 の t1 → t2 の再生の末の P が保存済み t02 と bit 一致（Snake は SNA の t1 → t2 で同じ）。これは drive_verify K7 の再現で、PASS しなければ再生の値は使わない。
- **K6 符号の約束**: 1 状態で、小さな η の全バッチ GD の実ステップ（新ラベル 1 本）の Δm_i の期待（ラベル 20 本の平均）と −η G_full_i の符号一致が生きた unit の 90% 以上（一次の近似の検算。しきい値は雑音から: ラベルの平均の SE の 2 倍以内なら「不定」として除く）。
- **K7 変異対照**: v^c を乱数に置き換えた状態で κ ≈ 0（|κ| < κ_lb の正の値の 1/10）になり、S_exact の沈める側が 0.5 ± 0.1 に落ちること（装置が「沈める側 ≥ 0.8」を状態によらず出していないことの確認。[[threshold-derivation-rule]] の変異対照）。

## 6. 判定の帯

- 割合は seed ごとに出し、10 seed の平均と SD、t 分布（自由度 9）の 95% 区間で書く。腕 × 境界の表は 10 seed の平均（SD）。
- κ_k/κ₁ は seed ごとの値で「≥ 0.8 の seed の数 / 10」を境界ごとに数える。
- 符号一致・沈める側は「生きた unit かつ |G| > 数値の床（K2 の差の最大）」で数える。
- 「初期化の押し／切替の押し」は中央値の比（seed ごとに比を取り、10 seed の中央値）。

## 7. 予測（計算前に登録。P1〜P7 は親 Fable の予測。Issa の予測は「覆ることは無い」＝ MLP と同じ印が全部出る。このセッションの Fable が直した所は ★ で示し、元の文も残す）

- **P1** κ_k/κ₁ ≥ 0.8 が全境界で成り立つ（LR 10/10 seed、Snake 4 腕は各境界 8/10 以上）。
- **P2** 補題の破れ 0。★ κ/κ_lb は **2 以上**（元: 2〜5）。直した理由: 30,000 更新/課題で memo 1.0 まで当てはめるので logit の広がりが MLP（4,000 更新）より大きく、Jensen の隙間 logsumexp − ℓ̄ − log K が開いて κ/κ_lb は 5 を超えうる。上限は置かない。
- **P3** 期待の押し G_full の沈める側: 最終隠れ層 f2 ≥ 0.8、f1 ≥ 0.7、c2 は leaky ≥ 0.7・Snake は谷の則、c1 は LR で ≤ 0.5（既知の浮き）。★ 注: f1 は隠れ層に読まれる層なので定数モードの競合（c1_direction_1010）で c1 のように浮きうる。このセッションの Fable の確率は f1 ≥ 0.7 に 0.5。親の予測はそのまま採点する。
- **P4** 実際の 200 更新の沈める側は f2 で ≥ 0.8。★ 最初の 1 歩の沈める側は **≥ 0.7**（元: 50〜65%）。直した理由: MLP の 52〜58% は 4,000 更新/課題で残っていた前課題の運動量による。この箱は 30,000 更新で課題末の勾配がほぼ 0 になり、drive_verify で最初の Adam 1 歩の有意な反転は c1 ≤ 1.7%・c2 3〜4% と、Adam の 1 歩は −G にほぼ従った。元の帯（50〜65%）も併記して採点する。
- **P5** 初期化の押し／切替の押し ≤ 1/30（f2 の |S_exact| の中央値の比）。
- **P6** 戻り（g1）は unit の 1% 以下（層ごと、全腕）。
- **P7**（ダイヤルを回したら）単調・滑らか・閾値なし、押し／下がり幅は checkpoint 間で 3 倍以内。
- **P8**（Snake の判定行）: Snake 4 腕で κ_k/κ₁ ≥ 0.8（P1）かつ f2 の G_own の符号が S_self の符号と 0.8 以上一致し、S_self < 0 の unit では G_own も浮かせる側が多数 → 「装填は活性化によらず戻り、向きは活性化で決まる」。
- **P9**（Keep）: f2 の med|G_full| の比（t / t1）は全境界で 0.3 以上（押しの大きさは桁では落ちない）。c1 の比は LR で 1 を割る境界がある（既知の浮きと符号の割れ）。

## 8. Astra（Codex gpt-6-astra）の役割

(i) この spec の批評（計算前、`analysis/kappa_reload_cnn_1020/codex_critique.md`）。(ii) 独立の導出ノート（`codex_derivation.md`）: κ の下界は線形 head と CE だけで成り立ち構造によらないこと、conv 層では読み出しが入力依存になるので κ が定義できず何に置き換わるか。(iii) 測定コードの監査（符号の約束、K_n の定義、ラベル期待、Adam の再生、`codex_audit.md`）。(iv) 最終の要約文の審査（言い過ぎの型: 固定状態の帳簿を因果で書かない、分担と符号を混ぜない、`codex_review.md`）。Astra の答えは Codex の発言で Issa の指示ではない。

## 9. 実行の約束

- GPU（RTX 5060 Ti）を使う前に `nvidia-smi` で他の python が無いことを見る。1 プロセス ≤ 4 GB。他のプロセスは殺さない。
- コード: `src/kappa_reload_cnn_1020.py`（事後の量と再生）、`analysis/kappa_reload_cnn_1020/`（集計・Astra のノート）。結果: `results/kappa_reload_cnn_1020/`（rows.json・replay の npz は raw/・summary.md）。
- 既存ファイル（`src/cnn_drive_verify_1009*.py`・`src/sna_cnn_cause_1009.py`・宿主）は編集しない。
- 片付け（CLAUDE.md §4）: git 外の生データは `~/Projects/obsidian-research-data/kappa_reload_cnn_1020/` へ manifest つきで退避 → origin/main を merge → push → worktree と branch を消す。
