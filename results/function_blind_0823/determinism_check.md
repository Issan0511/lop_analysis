# function_blind_0823 決定性照合

- 実装 commit: `2179d12`
- 正本: `results/function_blind_0823/`
- 再実行先: `/tmp/function_blind_verify.pJIMlN/`
- 実行条件: `OMP_NUM_THREADS=1`、同一入力・同一コマンド
- 照合対象: 両出力の全 CSV 15 ファイル
- 方法: 相対パス順に並べた各 CSV の SHA-256 を `diff -u` で比較
- 結果: **PASS（差分 0、exit code 0）**

`meta.json` と `summary.md` は実測経過時間を含むため byte 一致の対象外。判定値・曝露表・最適化系列を持つ CSV は全て byte 一致した。
