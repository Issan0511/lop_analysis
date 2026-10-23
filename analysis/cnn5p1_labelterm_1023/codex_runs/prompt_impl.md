あなたはこの repo（lop_analysis、worktree /home/issan/Projects/claude/wt/cnn5p1_labelterm_1023、ブランチ codex/cnn5p1_labelterm_1023）で、登録済みの spec に従って保存状態の解析を実装・実行し、結果をまとめる担当です。新しい学習は一切しません。

## まず読むもの
1. `specs/spec_cnn5p1_labelterm_1023.md`（この解析の spec。判定 J1〜J3・予測・検査 (a)〜(e) は固定済み。変えない）
2. `specs/spec_kappa_cnn5p1_1022.md` と `src/kappa_cnn5p1_1022.py`（同じ checkpoint から腕・seed を取り出し、訓練時の順伝播を再現し、FC2 の特徴・φ′・head を読む実装。これを import するか関数を再利用する。順伝播の再現の検査もここにある）
3. `src/cifar5p1_cnn_budget_1019.py` と `results/cifar5p1_cnn_budget_1019/b*/task_plan.json`・`frozen_alpha.json`・`provenance.json`（課題のクラス順・画像・前処理・活性化の α_s）
4. `results/kappa_cnn5p1_1022/summary.md`（状態の除外規則と数の例）
5. repo 直上の `/home/issan/Projects/claude/CLAUDE.md`（worktree とブランチの規則）

## やること
- `src/cnn5p1_labelterm_1023.py`: spec §2〜3 の量を全状態（4 腕 × 3 予算 × 10 seed × 境界 1・9・19・29 × 次課題 easy/hard、層は FC2 主・FC1 記録）で計算し、`results/cnn5p1_labelterm_1023/rows.csv`（状態 × 層の集約量）と unit 別の配列（npz）を書く。GPU（RTX 5060 Ti、16 GB、いま空き）を使ってよい。checkpoint は `/home/issan/Projects/obsidian-research-data/cifar5p1_cnn_budget_1019/results/cifar5p1_cnn_budget_1019/b{780,2340,7800}/checkpoints/` から読むだけで、repo にコピーしない。
- `src/cnn5p1_labelterm_1023_checks.py`: spec §6 の検査 (a)〜(e) を実装して `results/cnn5p1_labelterm_1023/checks.json` に PASS/FAIL と数値を書く。(a) と (c) を最初に 1 状態で通してから全状態を回す。通らなければ止めて、原因と試したことを最終報告に書く（結果をでっち上げない）。
- `analysis/cnn5p1_labelterm_1023/summarize.py`: 集約（腕 × 予算 × 境界の seed 中央値と範囲、判定用の 9 マス）、`tables.md`、`verdict.json`（J1〜J3 と記録量）、`summary.md`（日本語。一行の結論、判定の表、窓ごとの表、検査の結果、言わないこと、spec からの逸脱があればその一覧）、`provenance.json`（この worktree の commit、checkpoint の sha256、実行時刻、python/torch の版）。
- commit はこの worktree のブランチにだけ、意味の単位で行う（spec は commit 済み 8c4ecf05）。push はしない。他の worktree・`/home/issan/Projects/claude/proj_004_drift`・vault には触らない。git に入れない生成物（大きな npz など）は最終報告に列挙する。
- このプロンプトは `analysis/cnn5p1_labelterm_1023/codex_runs/prompt_impl.md` に置いてある。最終報告は `--output-last-message` で `codex_runs/final_impl.md` に入る。

## 守ること
- spec の判定・閾値・窓・予測を結果を見て変えない。計算の定義を spec から変えざるを得ないときは、変更点を `summary.md` の冒頭と最終報告に明記する。
- 符号規約: S > 0 が沈める側（Δm_i = −η S_i^real）。検査 (c) で autograd と突き合わせる。
- 「生きた unit」は spec §2 の定義。生存数を併記する。
- 同じ保存状態の再集計を独立な追試と数えない。10 seed は seed 間の範囲と中央値で報告する。
- 新しい学習・再生・API ワーカー・サブエージェントは起動しない。

## 最終報告（日本語、最後のメッセージ）
判定 J1〜J3 の結果と数（9 マス中いくつ）、hard 切替の agree_rand・ratio_L・L_sink・sink_real/sink_rand と easy 切替の class_sink・ratio_class（腕 × 予算の seed 中央値と範囲の表）、ReLU と FC1 の記録の要点、検査 (a)〜(e) の PASS/FAIL と数値、除外した状態の数、spec からの逸脱、commit hash の一覧、git に入れなかった生成物。
