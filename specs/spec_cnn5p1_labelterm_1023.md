# spec: 実ラベルの切替の押しを乱数ラベルの項とラベル項に分ける（cnn5p1_labelterm_1023）

状態: **登録**（判定・閾値・予測は実装の前に commit）/ 2026-10-23 / 起案: Claude（Fable、V13 のセッション）/ 実装・実行: Astra（codex exec、gpt-6-astra）/ 親: `specs/spec_kappa_cnn5p1_1022.md`（保存状態と順伝播の再現）、`results/cifar5p1_cnn_budget_1019/`（保存 checkpoint）

## 1. 問い

V13 §3.1 の仮定 H1 は「新ラベルが網の状態から独立」を使う。実ラベルの 5+1 CIFAR CNN の課題の切替で、最終隠れ層 FC2 の unit への 1 歩の期待の押し（自分の重みと bias を通る経路）

- S_i^real = (1/N) Σ_n K_n φ′_in ⟨v_i, π_n − y_n⟩（実ラベル）
- S_i^rand = (1/N) Σ_n K_n φ′_in ⟨v_i, π_n − ȳ⟩（ラベルを経験分布 ȳ で置いた乱数ラベルの模型）
- L_i = (1/N) Σ_n K_n φ′_in ⟨v_i, y_n − ȳ⟩（ラベル項）

は恒等的に S^real = S^rand − L を満たす。乱数ラベルの模型 S^rand が実ラベルの押し S^real の向きを当てるか（J1）、ラベル項は小さいか（J2）、未見クラスの head の行は押しをどちらへ動かすか（J3）。

## 2. 定義

- **状態**: `cifar5p1_cnn_budget_1019` の保存 checkpoint。4 腕（ReLU・ELU α=3.6・KKT1・SNAfrz1）× 3 予算（780・2,340・7,800 更新/課題）× 10 seed、hard 課題末の境界 b ∈ {1, 9, 19, 29}。checkpoint の在処は `/home/issan/Projects/obsidian-research-data/cifar5p1_cnn_budget_1019/results/cifar5p1_cnn_budget_1019/b{780,2340,7800}/checkpoints/bNN.pt`（repo の `results/cifar5p1_cnn_budget_1019/backup_manifest.json` に sha256）。読み込み・腕と seed の取り出し・順伝播・活性化（KKT1/SNAfrz1 の α_s は保存値）は `src/kappa_cnn5p1_1022.py` の実装を再利用する。崩壊した ReLU の状態（train 精度 0、f2 が全死）は kappa_cnn5p1_1022 と同じ規則で除き、数を併記する。
- **次の課題**: (i) easy = 課題 b+1（1 クラス、500 枚）、(ii) hard = 課題 b+2（5 クラス、2,500 枚）。b = 29 は easy（課題 30）のみ。課題のクラスと画像は `results/cifar5p1_cnn_budget_1019/b*/task_plan.json` と `src/cifar5p1_cnn_budget_1019.py` の課題生成に従う（seed ごとのクラス順）。入力の前処理（std）も訓練と同じ。
- **記号**（画像 n、unit i は FC2 の 100 unit）: a_n = FC2 への入力（FC1 の活性化後、100 次元）、ā = 次の課題の全画像での a_n の平均、K_n = ⟨a_n, ā⟩ + 1、z_in = w_i·a_n + b_i、φ′_in = φ′(z_in)（訓練の autograd と同じ微分）、ℓ_n = W φ_n + c（head、100 クラス）、π_n = softmax(ℓ_n)、v_i = W[:, i]（unit i の読み出し列）、y_n = one-hot（100 次元）、ȳ = 次の課題の全画像でのラベルの平均（hard では 5 クラス各 1/5、easy では 1 クラスの one-hot）、N = 次の課題の画像数。
- **符号**: 位置 m_i = w_i·ā + b_i の SGD 1 歩の変化は Δm_i = −η S_i^real（交差エントロピー、全画像の平均勾配、自分の重みと bias の経路のみ）。S > 0 が沈める側。⟨v_i, ·⟩ の相手はいずれも成分の和が 0 なので、v_i の中心化は不要（中心化しても同じ値になることを検査で確かめる）。
- **補助**: S_i^unif = (1/N) Σ_n K_n φ′_in ⟨v_i, π_n − 1/C⟩（C = 100、H1 の字面どおりの一様）。class_term_i = S_i^rand − S_i^unif = (1/N) Σ_n K_n φ′_in ⟨v_i, 1/C − ȳ⟩（easy では未見クラス c の行 1 本の効果）。
- **生きた unit**: 次の課題の画像で Σ_n K_n |φ′_in| > 0 の unit（それ以外は S が恒等的に 0）。生存数を併記する。

## 3. 量

状態 = 腕 × 予算 × seed × 境界 × 次課題種別（easy／hard）。各状態で生きた unit について:

- agree_rand = sign(S^real) = sign(S^rand) の割合。agree_unif = sign(S^real) = sign(S^unif) の割合。
- ratio_L = median_i |L_i| / |S_i^rand|（S_i^rand = 0 の unit は除く）。
- sink_real, sink_rand, sink_unif = S > 0 の割合。
- L_sink = L_i < 0 の割合（S^real = S^rand − L なので、L < 0 はラベル項が沈める側に足す）。
- easy では L ≡ 0（agree_rand ≡ 1）。easy の主量は class_sink = class_term_i > 0 の割合と ratio_class = median |class_term| / |S^unif|。
- 集約: 腕 × 予算 × 境界ごとに seed 間の中央値と範囲。判定は腕 × 予算の 9 マス（ReLU 以外）で、境界を seed 内で平均してから seed 中央値を取る。ReLU は記録。FC1（f1）についても同じ量を記録として出す（a_n は FC1 への入力、押しは自分の重み経路のみ）。

## 4. 判定（実装の前に固定）

- **J1 RAND_PREDICTS_SIGN**: hard 切替で agree_rand の seed 中央値 ≥ 0.85 のマスが 9 マス中 8 以上。満たさなければ RAND_MISSES_SIGN。
- **J2 LABEL_TERM_SMALL**: hard 切替で ratio_L の seed 中央値 ≤ 0.30 のマスが 9 マス中 8 以上。満たさなければ LABEL_TERM_LARGE。
- **J3 UNSEEN_ROWS_SINK**: easy 切替で class_sink の seed 中央値 ≥ 0.75 のマスが 9 マス中 8 以上。≤ 0.25 が 8 以上なら UNSEEN_ROWS_LIFT。どちらでもなければ MIXED。
- 記録（判定外）: agree_unif（hard）、L_sink（hard）、sink_real と sink_rand の差、ReLU の全量、FC1 の全量、境界ごとの推移。

## 5. 予測（実装の前、Fable、確率）

J1 ○ 0.60。J2 ○ 0.50。J3 ○ 0.60（未見クラスの行は負例としてだけ押されており平均より低いので、class_term > 0）。L_sink は 0.5 前後（ラベル項の向きに偏りなし）0.5。Issa の予測はこの下に Issa が書く。

## 6. 検査（必須）

- (a) 順伝播の再現: kappa_cnn5p1_1022 と同じ状態・画像集合で train 精度と f2 の φ′ の中央値が一致（相対 1e−6 以内）。
- (b) 恒等式 S^real = S^rand − L、S^rand = S^unif + class_term の相対誤差 < 1e−10（全状態）。
- (c) 1 状態（ELU 780 seed 2000 境界 9、hard）で、autograd により交差エントロピーの全画像平均の勾配から Δm_i = −η(∇_{w_i}L·ā + ∇_{b_i}L) を直接計算し、−S_i^real と一致（相対 1e−6）。
- (d) 置換対照: hard 切替で y_n を画像の間で置換（5 回、seed 固定）すると L の中央値が 0 付近になり、agree_rand が本物より高いか同等であること（本物の L が偶然でないことの確認）。
- (e) v_i を中心化しても S が変わらない（相対 1e−12）。

## 7. 言わないこと

固定状態の 1 歩の期待勾配で、Adam の分母は使わない。実際の 1 更新ではない。上流の項（a_n の変化）は含まない。課題の中の押しではなく、切替直後の 1 歩だけ。実ラベルが乱数ラベルと同じだとは言わない。言えるのは「切替の押しの向きを乱数ラベルの模型が当てるか、ラベル項の大きさと向き」まで。

## 8. 置き場

spec: `specs/spec_cnn5p1_labelterm_1023.md`。code: `src/cnn5p1_labelterm_1023.py`（計測）、`src/cnn5p1_labelterm_1023_checks.py`（検査）、`analysis/cnn5p1_labelterm_1023/`（集計・判定・codex_runs/ のプロンプトと最終報告）。results: `results/cnn5p1_labelterm_1023/`（rows.csv = 状態 × 層の全量、unit 別は `unit_rows.parquet` または npz、tables.md、verdict.json、checks.json、summary.md、provenance.json に commit・checkpoint の sha256・実行時刻）。checkpoint は obsidian-research-data から読むだけで repo に入れない。git に入れない生成物があれば終了時に列挙する。
