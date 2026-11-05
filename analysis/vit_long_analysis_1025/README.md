# vit_long_analysis_1025

Recorded-output analysis requested on 2026-10-25. The 6-block `vit_long_1024`
experiment is complete; the 2-block addendum is a frozen interim snapshot.
This analysis does not launch training or implement the checkpoint-based J5 calculation.

Run `analyze.py --snapshot <frozen snapshot> --out <new output directory>` with
NumPy, pandas and matplotlib. The snapshot is archived at
`/home/issan/Projects/obsidian-research-data/vit_long_analysis_1025/snapshot`.
Every source file is checked against `manifest.json` before aggregation.

`--capture` creates a new snapshot, refusing to overwrite an existing one. It
copies only completed runs. A different capture is a new analysis time point;
do not replace the present interim result silently. Source paths can be set
with `--source-repo` and `--raw-2blk`.

Checks reconstruct online accuracy, 100-update chunks, T50/T75/T90, and final
memorisation from the per-update arrays; verify probe averages, task sequences,
paired task-1 identity, and run configurations; and independently reproduce the
original 6-block registered judgement labels and per-seed endpoints.

Numerical outputs and figures are reproducible. The Japanese interpretation in
`results/vit_long_analysis_1025/summary.md` is maintained separately.

`plot_normalized_position.py --source results/vit_long_analysis_1025 --out <dir>`
plots mean / standard deviation for the LN arms, using the recorded median of
per-unit ratios. It checks all plotted values against the archived run CSVs,
then produces final-FFN and all-FFN CLS plots, seed/task/window tables, and a
source-hash check record. The 2-block selection remains the same interim three
seeds. It does not reconstruct a ratio from separately aggregated means and SDs.
