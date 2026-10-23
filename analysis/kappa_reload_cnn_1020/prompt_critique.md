あなたは Codex gpt-6-astra。役割は「計算前の spec の批評」。この worktree（/home/issan/Projects/claude/wt/kappa_reload_cnn_1020）は git の作業ツリーで、spec は `specs/spec_kappa_reload_cnn_1020.md`。読むだけで、コードやデータは変更しない。答えは **日本語** で、`analysis/kappa_reload_cnn_1020/codex_critique.md` に書く（このファイルだけ作成してよい）。

## 背景
1 隠れ層 RL-MNIST で「学んだ読み出し κ が課題の切替ごとに押しを装填し直す」ことを測った先行の解析（`analysis/kappa_reload_1020/measure.py`、`results/kappa_reload_1020/summary.md`、`results/reload_dial_1020/summary.md`）を、RL-CIFAR CNN の最終隠れ層で同じ定義のまま測る。CNN 側の既知は `results/cnn_drive_verify_1009/summary.md` と `results/c1_direction_1010/summary.md`（c1 の押しは定数モードで浮く）。記号は V12 付録 C.3 の流儀（位置 m = w·x̄ + b、明るさ K_n = x̄·x_n + 1、自己形 κ(1−1/K)Σ K_n φ′φ、S > 0 が沈める側）。

## 頼むこと（計算はまだ始めていない。spec を直すための批評）
1. 定義の誤り・曖昧さ: §3 の各定義（κ、分解、補題、S_exact、G_own、G_full、G_up、自己形、生きた unit、再生の記録）に、MLP の定義と食い違う点、CNN に持ち込むときに意味が変わる点（特に G_own と G_full の二本立て、上流項 G_up の扱い、f1 の「入力」が pool 後の特徴であること、conv の平均パッチ）があるか。
2. 検査 K1〜K7 の抜け: 装置が「状態によらず同じ答え」を出す危険を潰せているか。追加すべき変異対照・帰無基準があれば具体的に（しきい値は雑音から導く流儀で）。
3. 予測 P1〜P9 の反証可能性: 文面で採点できるか。採点できない予測は、どう書き直せば採点できるか。親（Fable）の元の予測を Fable がこのセッションで直した 2 件（P2 の上限撤廃、P4 の最初の 1 歩）の理由が通るか。
4. 分解の参照状態（§3「分解」）: span が 1 課題でない境界（t5 は t3 から、t10 は t5 から …）で κ_chance・κ_feat・κ_read の意味が保たれるか。保たれないなら、何と呼び替えるべきか。
5. 補題（κ ≥ [N(log K − L) − B]/‖F‖²）: この箱（CNN、Adam、30,000 更新、memo 1.0）で成り立つための条件が spec に書かれているか。足りない条件があれば書く。
6. 言い過ぎの型の予防: 最終の要約で避けるべき表現（固定状態の帳簿を因果で書く、分担と符号を混ぜる、など）を、この spec の測定に即して列挙する。
7. その他、計算の前に直すべきこと（計算量・数値精度・記録の粒度・seed の扱い）。

出力の形: 見出しごとに「問題 → 直し方（spec のどの文をどう変えるか）」。重要度の高い順。一文一主張。推測は推測と書く。
