# hsweep_rlmnist_0924 — 隠れ幅 h を振って「容量が余っているから削らない」を因果で決める

spec `specs/spec_hsweep_rlmnist_0924.md`（登録 commit c7aa07c）、予測 `PREDICTIONS_claude.md`（走る前、sha256 は `SHA256_registered.txt`）と `PREDICTIONS_codex_raw.txt`（gpt-6-astra、走の起動 11:47 の後 11:54 到着、sandbox は scratchpad）。
走: 784-h-h-10、h ∈ {8,16,32,64,100,300}、leaky .1、Adam 1e-3、固定 1,200 枚・毎課題乱数ラベル、50 課題 × 30,000 更新、seed 200–204、CPU float32、runner commit 63746e0（`launch_git_hash.txt`）。11:47 起動、13:13 完走、失敗 0。

**判定 LOSS_SET**（`verdict.json`）: 当てはまる腕 h = 64/100/300 で f_10 = .556/.566/.599（h=100 ± .1 の中）、B ∈ [.889, .934]、P2 は全腕 0/5。当てはまらない腕（h=32: .866、16: .588、8: .358）でも f_10 は .51〜.54。h=8 は読み出しがユニット空間の全部を読む（read_share 1.0）が半分残る。

- `summary.md` 群の中央値、`per_seed.csv` seed 別、`transitions.csv` 課題ごとの Σ 計量の帳簿（V, Q, X, c, ρ）、`prediction_scores.{csv,json}`（Brier: Claude .208、Codex .217、共通 10 件）。
- `analysis/hsweep_rlmnist_0924/`: `launch.sh`（第 1 launcher、8 並列）、`launch_remaining.sh`（12:00 に足した第 2 launcher、6 並列。第 1 の残り列は STOP で止めた）、`analyze.py`（h=8 の直交補が空の場合の NaN 処理を本走中に追加。判定の定義は不変）、`score.py`、`archive_raw.py`。
- 逸脱: 最初の起動（11:47）は runner の登録検査で 30 本とも即失敗・出力なし → 検査を直して起動し直し（63746e0）。
- 生データ（state_NNN.npz・input_bank・task inputs・ログ）は `~/Projects/obsidian-research-data/hsweep_rlmnist_0924/`、`backup_manifest.json` に sha256。各系列の `metadata.json`・`status.json`・`per_task.json` は repo に残す。
