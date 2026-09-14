# snake_phase_mnist_0914 追補 1：起動規則の swap 条件（運用のみ・判定規則は不変）

状態: 本走の途中（2026-09-14 08:5x JST）・結果（対比の値）は未読 / 起草: Claude / 親: `specs/spec_snake_phase_mnist_0914.md`（aa78461）§8

## 何が起きたか

§8 の「起動の直前に … SwapFree < RSS_peak なら起動しない」が、07:35–08:38 の約 1 時間、新規 shard の起動を完全に止めた。この間 MemAvailable は 16–20 GiB あり（`scratchpad/memlog.txt`・`results/snake_phase_mnist_0914/launch_log.txt`）、SwapFree は 0.68–1.07 GiB で RSS_peak 1.096 GiB（`resources.json`）を下回っていた。swap の使用 7.2 GiB は長時間走っている他のアプリのページで、swapoff をしない限り戻らない。したがって SwapFree だけでは「いまメモリが逼迫しているか」を表さず、実メモリに十分な余裕があっても起動を止め続ける。08:47 時点で完了 118/340。

## 変更

起動の swap 条件を **「SwapFree < RSS_peak かつ MemAvailable < 2·RSS_peak + ΔM_desk」** に限定する。右側の境界は §8 の見張りの復帰境界と同じ算術（止めた shard の一時確保 1 回分と境界 1 の余白）。MemAvailable がこの境界以上なら、新しい shard は swap に押し出されずに収まる。

変えないもの: 並列数の式 P、見張りの SIGSTOP/SIGTERM/SIGCONT の境界、shard 表、判定規則（§6–§7）、腕・seed・予測。学習の軌道は起動の順序と時刻に依存しない（腕も時刻も RNG に入らない）ので、既に完了した 118 走はそのまま使う。

## 検査

`launch.py --selftest`（S15b）に 2 ケースを追加: MemAvailable 2.2 GiB・SwapFree 0.5 GiB（RSS_peak 1 GiB・ΔM 0.5 GiB）で起動しない／MemAvailable 13 GiB・SwapFree 0.5 GiB で起動する。変異 ignore_swap（swap 条件を外す）は前者で検出される。all_pass。
