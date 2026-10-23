# chargpt_latepush_1023

固定状態の読み出し勾配を、元の softmax 誤差を固定して分ける。学習・optimizer の更新は行わない。
判定と数値検査の定義は `specs/spec_chargpt_latepush_1023.md`（実装前の登録 commit: `f90fa9fa`）。

## 入力

- `~/Projects/obsidian-research-data/chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar`
- 束の manifest は同じフォルダの `bundle_manifest.json`。束と展開後の全ファイルの SHA256 を検査してから利用。
- worktree 直下に展開し、`results/chargpt_cl_1019/M2/{C3,ReLUcap}/seed{0..9}/ckpts/` を読む。
- 元モデルの src は main に未統合なので、`source_manifest.json` が示す git blob を `source/src/` に保存してある（無改変）。他の worktree を必要としない。
- κ・cos²・probe・alive 等は `analysis/chargpt_seed10_1022/load10.py` を直接 import。CPU float64、8 threads。

## 実行

worktree 直下で、torch と numpy の入った Python を使う。実行環境は結果の `provenance.json` を参照。

```sh
python -B analysis/chargpt_latepush_1023/measure.py gate
python -B analysis/chargpt_latepush_1023/measure.py run
python -B analysis/chargpt_latepush_1023/judge.py
```

`gate` は seed 0 の両腕・全36状態・両層を元 CSV と照合し、同時に unit ごとの full を元の勾配から再計算して照合。
失敗すると `run` へ進めない。seed×腕ごとに CSV と完了 JSON を書き、再実行は検証済みの完了部分を再利用する。
主判定の課題末は35点。init を含む36点の相関は補助として保存する。

## 出力

`results/chargpt_latepush_1023/` の `summary.md` と `judge.md` が結果の入口。
`states.csv` は init も含めた1440層行、`windows.csv` は seed 内の窓中央値、`correlations.csv` は seed ごとの順位相関。
`raw/*.npz` は各状態・層・unit の alive と full/load/res の押しで、終了後に専用の研究データ置き場へ退避する。
退避先・bytes・SHA256 と、束から展開して削除した checkpoint の記録は `backup_manifest.json` を参照。

head のパラメータを差し替えて softmax を再計算する介入とは異なる。
元状態の CE の body 勾配に流れる `e @ v` を同じ関数で3通りに計算し、unit ごとの加法性を全状態で確認する。
