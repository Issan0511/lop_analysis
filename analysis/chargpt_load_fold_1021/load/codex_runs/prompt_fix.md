監査の指摘を反映してください。日本語で報告してください。作業ディレクトリは `/home/issan/Projects/claude/wt/chargpt_load_fold_1021`。commit しない。書いてよいのは `analysis/chargpt_load_fold_1021/load/`（codex_audit.md は書き換えない）と `results/chargpt_load_fold_1021/load/` だけ。`/home/issan/Projects/claude/wt/chargpt_cl_1019` は読むだけ。Python と環境は前回と同じ（CPU、`CUDA_VISIBLE_DEVICES=""`、`PYTHONDONTWRITEBYTECODE=1`、`/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`）。

# 監査の報告
`analysis/chargpt_load_fold_1021/load/codex_audit.md`（独立の監査役が書いた）。主要結論は維持、直すのは次の 3 点。

1. **P0-7 の主比較を作り直す**（監査 §7 の修正案どおり）。各状態 k で、位置 m_k（課題 k の鎖の probe の ȳ_k）と生きた unit の集合 A_k を固定し、(a) 登録どおりの一様損失 L_u の勾配と、(b) 実際の次の課題（課題 k+1 の鎖の probe と次の文字、1 実現）の CE の勾配を、どちらも同じ m_k・A_k に当てて G_own の沈める側の割合を比べる。(b) の CE の位置は元の学習と同じ範囲（`chargpt_cl_1019.py` の学習の CE の位置。監査によれば先頭を除く 127 位置）にする。G_full についても同じ m_k への JVP で同じ比較を出す（重ければ代表状態だけでもよいが、その場合は採点欄で「P0-7 は own で採点、full は代表状態のみ」と限定を書く）。今の差分（m と A が課題 k+1 のもの）は補助の列として残す。全腕・全状態で計算し直し、P0-7 を採り直す（登録の原文から一意に決まる採点と、定義を補った再解析を分けて書く）。
2. **評価記録の検査の許容**: 「登録の丸め基準（.9g の丸め幅だけ）では 0/8、float32 の評価差を足した基準で 8/8」と分けて checks.json と summary_load.md に書く。**許容の変更は Issa（利用者）の承認ではなく、親の Claude の指示による**ので、provenance.json と summary_load.md の「利用者が承認」の記述を「親（Claude、V13 セッション）の指示で変更」に直す。
3. **source HEAD**: 計算の時点で chargpt の worktree の HEAD は 6c80d308 だった（spec の f71be184 から進んだ）。対象のソースの差は更新 100 の Adam 保存の 1 行だけ、という監査の確認を provenance と summary_load.md に書く。

summary_load.md の表と採点を更新し、最終回答には変わった数（P0-7 の新しい表と採点）を書いてください。
