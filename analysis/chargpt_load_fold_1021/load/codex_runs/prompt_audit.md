あなたは独立の監査役です。日本語で答えてください。作業ディレクトリは `/home/issan/Projects/claude/wt/chargpt_load_fold_1021`。**書いてよいのは `analysis/chargpt_load_fold_1021/load/codex_audit.md` の 1 ファイルだけ**です。他は読むだけ。commit しない。

# 監査の対象
別の Codex（あなたではない回）が実装・計算した (A) 装填の事後解析です。
- 登録: `specs/spec_chargpt_load_fold_1021.md` §2・§4（計算前に commit、8122b8b1）。予測 P0-1〜P0-7。
- コード: `analysis/chargpt_load_fold_1021/load/measure.py`。
- 結果: `results/chargpt_load_fold_1021/load/`（checks.json・checks.log・table_head.csv・table_units.csv・summary_load.md・units_*.npz・provenance.json・audit.json）。
- 定義の正本: `src/kappa_reload_cnn_1020.py`、`results/kappa_reload_cnn_1020/summary.md`、`analysis/kappa_reload_cnn_1020/codex_derivation.md`。
- 文字 GPT のモデルとデータ（読むだけ）: `/home/issan/Projects/claude/wt/chargpt_cl_1019/src/chargpt_cl_1019.py`・`chargpt_cl_1019_kune.py`、`results/chargpt_cl_1019/quicklook/`。

# 確かめてほしいこと（それぞれ「問題なし／問題あり（重さ: 致命・要修正・注記で足りる）」と根拠の行番号）
1. head の量: κ = A/‖F‖²、cos²、Q、帰無（クラス置換）、補題 κ_lb = N(log V − L)/‖F‖²（bias なし）が正本の定義と一致するか。F の作り方（one-hot − 1/V と最終 LN の出力 h）と、課題 k の鎖の probe・次の文字の対応（ずれが無いか）。
2. 埋め込みと head の共有の扱い（読み込み後に同じテンソルか、G_full の JVP で共有パラメータが一度だけ数えられているか）。
3. L_u = mean_n[logsumexp f_n − mean_c f_nc] と、その勾配の向き。G_own = ȳ·∇_{w_j} L_u、G_full = ⟨∇_θ m_j, ∇_θ L_u⟩（JVP）の符号の約束（G > 0 が沈める側）と、m_j の定義（どの probe の token 平均か）。
4. 「生きた unit」の定義と、沈める側の割合の分母。
5. stage（init・end・u100）のラベルと、表の「課題 1〜10」「課題 39〜50」の集計に u100 が混ざっていないか。
6. 初期化比（init / 課題 1 末の |G_own| の中央値）の計算と、その解釈で言い過ぎが無いか（初期の予測が一様から離れている、head の初期の尺度が小さくない、など）。
7. P0-7（実際の次の課題の 1 実現との比較）の計算。
8. 予測 P0-1〜P0-7 の採点が、登録した文言どおりか（甘すぎ・厳しすぎ）。
9. summary_load.md の言い過ぎ（固定状態の帳簿を因果で書いていないか、分担と符号を混ぜていないか、1 seed を一般化していないか）。
10. spec と実装の食い違い（例: spec は特徴次元 512 と書いたが実際は 128）が結論に効くか。

最後に「結論を変える問題があるか」を 1 行で。codex_audit.md に全部書き、最終回答にも要点を書いてください。
