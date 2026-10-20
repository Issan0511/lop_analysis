あなたは Codex gpt-6-astra。役割は「測定コードの独立監査」。この worktree（/home/issan/Projects/claude/wt/kappa_reload_cnn_1020）は git の作業ツリーで、読むだけ（コード・データ・結果は変更しない）。答えは **日本語** で `analysis/kappa_reload_cnn_1020/codex_audit.md` に書く（このファイルだけ作成してよい）。

## 対象
- spec: `specs/spec_kappa_reload_cnn_1020.md`（§3 定義、§5 検査、追補 1 の A1.1〜A1.3 が正本。あなたの計算前批評 `analysis/kappa_reload_cnn_1020/codex_critique.md` と導出ノート `codex_derivation.md` を受けて直した版）。
- 測定コード: `src/kappa_reload_cnn_1020.py`（事後の量・検査・再生）。集計: `analysis/kappa_reload_cnn_1020/summarize.py`。
- 検査の結果: `results/kappa_reload_cnn_1020/checks.json`（と `checks_*.json`、`floors.json`）。行: `results/kappa_reload_cnn_1020/rows.json`。表: `results/kappa_reload_cnn_1020/tables_kappa.md`・`tables_push.md`。採点: `verdict.json`。
- 依存（読むだけ）: `src/cnn_drive_verify_1009.py`・`src/cnn_drive_verify_1009_lr.py`・`src/sna_cnn_cause_1009.py`・`src/rlcifar_cnn_0908.py`・`analysis/sna_cnn_cause_1009/posthoc_lib.py`・MLP 側の正本 `analysis/kappa_reload_1020/measure.py`・`analysis/reload_dial_1020/analyze.py`。

## 監査の観点（コードを読んで、spec の定義との食い違いを探す）
1. **符号の約束**: G > 0・S > 0 が沈める側（期待の SGD 一歩で m が下がる）になっているか。JVP の接線（∇L_u）の向き、−η の符号、Δm = m(s) − m(0) の向き、「沈んだ」= Δm < 0、一致 = sign(−G) と sign(Δm)。
2. **K_n の定義**: f2 は x = a3 = φ(z3)（f1 の出力）で K_n = x̄·x_n + 1、f1 は x = flatten(maxpool(φ(z2)))、conv は zero-pad 込みの入力パッチの画像 × 位置平均 ū。「+1」が bias 座標から来ていること。平均の分母（画像数、画像 × 出力位置数）に zero-pad のゼロ位置を数えていないか。
3. **ラベルの期待**: L_u = (1/N)Σ_n[logsumexp ℓ_n − mean_c ℓ_nc] が「画像ごと独立の一様ラベル」の期待として厳密か。旧ラベル（κ・F・L・B 用）と新ラベル（L_u 用）を混ぜていないか。`posthoc_lib.labels_at(seed, t)` が課題 t のラベルを返すこと（off-by-one）。
4. **κ・F・A・B・L・D・J**: v^c の中心化の軸（クラス方向）、F = Σ_n (y_n − 1/K) ⊗ a4_n の形、A の 2 式、B = Σ_c (n_c − N/K) b^c_c、J ≥ 0、κ_lb = D/‖F‖²。射影帳簿（κ_chance・κ_feat・κ_read）の参照状態の取り方（t1 → 初期値、t2 → t1、…、t5 → t3 …）と span の記録。
5. **自己形と帳簿**: S_self = κ(1 − 1/K)Σ K_n g φ、S_oh = Σ K_n g v^c_{y_n}、S_proj = Σ K_n g (κF)_{y_n}、三段の帳簿の定義どおりか。
6. **G_full・G_own・G_up**: JVP の接線が全 1,200 枚で集計した一つの ∇L_u か（chunk ごとの内積の和ではないか）。α が固定されているか（Snake の α は checkpoint の V から復元した値を forward に固定しているか）。G_up がブロックごとの内積から独立に出ているか。head のブロックが 0 か。
7. **Adam の再生**: checkpoint の m/v/tc/act を正しく積み、stream（ラベル・batch 順）を課題 t の末まで進めてから課題 t+1 を回しているか。記録の forward が状態（V・乱数）を更新しないか。記録 step {0,1,2,5,10,20,50,75,100,150,200} が epoch 境界で正しく拾われているか。bit 一致の検査の比較対象（P・m・v・tc・V・ada・fixA・cval）。Snake の 40 束と 10 束の扱い。
8. **生きた unit・数値床・集計**: alive = 正側入力あり、conv は画像 × 位置。床は層ごと。seed の集約（平均・SD・t 区間・中央値）が spec の採点規則 A1.3 どおりか。空集合の扱い。
9. **検査の判定の変更**: Opus が自分で直した 2 つの判定（K6 は Richardson の極限、K4 のラベル置換で κ の変化を要求するのは特徴が画像で変わる状態だけ）が妥当か。
10. **採点（verdict.json）**: A1.3 の表どおりに P1〜P9 が機械的に採点されているか。特に P1（t ≥ 2、LR 10/10、Snake 腕ごと 8/10）、P3（seed 平均の点推定、全境界）、P4（1 歩 ≥ 0.7・200 更新 ≥ 0.8・旧帯）、P5（seed 中央値 ≤ 1/30）、P6（一意な seed-unit / 全 ≤ 1%）、P8、P9。

## 出力の形
「重大（結果の符号や採点を変えうる）」「中（数値を変えうるが結論は変えない）」「軽（表記・冗長）」に分け、それぞれ「どのファイルの何行目 → 何が spec と違う → 直し方」。問題が無い観点は「確認した・問題なし」と一行。推測は推測と書く。あなた自身で小さな再計算（Python で rows.json や raw/ の npz を読んで検算する程度）をしてもよいが、結果ファイルを書き換えない。
