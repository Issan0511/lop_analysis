# ceiling_t0b_C_0829（無効・seed 衝突）

`results/ratchet_log_0829c_invalid_seedcollision_b7f6963/` を入力に走らせた解析。
入力が腕E と bit 一致していたため、`verdict.csv` の数値は
`results/ceiling_t0b_E_0828/verdict.csv` と完全に同一である。**事前登録腕の結果として
読まないこと。** 判定語が `H_POSITIVE`（`EXPLORATORY_` 接頭辞なし）になっているのは
腕C として実行したためであって、独立な検証ではない。

このとき B9 は通過していた。初版の B9 が「`--seeds` と `--outdir` が存在する」しか
見ておらず、そのフラグが乱数系列を変えるかを見ていなかったからである。B9 は
spec §12 の 4 行目で強化した。

正しい腕C は `results/ceiling_t0b_C_0829/`。
