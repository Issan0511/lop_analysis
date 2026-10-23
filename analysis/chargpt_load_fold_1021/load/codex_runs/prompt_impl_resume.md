前回の続きです（前回は「Selected model is at capacity」で途中終了しました）。日本語で報告してください。作業ディレクトリは `/home/issan/Projects/claude/wt/chargpt_load_fold_1021`。commit はしないでください。

# 状態
- 依頼の全文は前回と同じ: `specs/spec_chargpt_load_fold_1021.md` の §2（(A) 装填）と §4 の (A) の検査を実装・計算し、予測 P0-1〜P0-7 を採点する。予測は書き換えない。§3（(B)）は親が済ませたので触らない（`results/chargpt_load_fold_1021/fold/` には書かない）。
- あなたが書いた `analysis/chargpt_load_fold_1021/load/measure.py` があり、`results/chargpt_load_fold_1021/load/checks.log` では C3・SiLU・GELU の検査が `recorded_eval_rounding` 以外すべて通っている。
- `recorded_eval_rounding` の不一致（C3 で絶対 4.2e−8、相対 8.5e−7）は、記録が float32 で計算・丸められたことで説明できる大きさ。許容を float32 の丸めに合う値（例えば相対 2e−6）に直し、その根拠（記録を作ったコードの dtype を `wt/chargpt_cl_1019/src/chargpt_cl_1019.py` で確かめる）を checks の出力に書いてください。許容を緩めるのはこの 1 件だけ。

# やること
1. 許容を直して `measure.py checks` を全腕で回し、全部通ったら `measure.py measure` を回す（CPU、`CUDA_VISIBLE_DEVICES=""`、`PYTHONDONTWRITEBYTECODE=1`、`/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`）。
2. `results/chargpt_load_fold_1021/load/` に checks.json（checks.log から作る）、table_head.csv、table_units.csv、units_*.npz、provenance.json、summary_load.md（日本語。定義の式・検査・表・P0-1〜P0-7 の採点（○・✗・判定不能と理由）・言えること／言えないこと。固定状態の帳簿を因果で書かない。数は表へ、一文一主張）。
3. 書いてよい場所は `analysis/chargpt_load_fold_1021/load/` と `results/chargpt_load_fold_1021/load/` だけ。`/home/issan/Projects/claude/wt/chargpt_cl_1019` は読むだけ。

最終回答には summary_load.md の要点（head の表、unit の表、採点）を書いてください。
