# ViT final-FFN ablation, 2026-10-25

Post hoc intervention on existing `vit_long_1024` checkpoints. The protocol was committed in `99b2cf5b` before evaluating the interventions. See `specs/spec_vit_ffn_ablation_1025.md` and `results/vit_ffn_ablation_1025/summary.md`.

The evaluator uses the original source from training commit `6164a5ea61b873959d9442653005c30cb548bc22`, frozen under `/home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025/source/`. Every provenance source hash and selected checkpoint hash is checked. No training or optimizer steps are performed. Source archives, raw logits, labels, image hashes and console output are outside the worktree, listed in `backup_manifest.json`. Original checkpoints remain in the existing per-seed tar archives, identified by `checkpoint_provenance.json`.

From the repository root:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/vit_ffn_ablation_1025/evaluate.py \
  --source-root /home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025/source \
  --archive-root /home/issan/Projects/obsidian-research-data/vit_long_1024/_gigafile_dl \
  --raw-output /home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025/raw \
  --output results/vit_ffn_ablation_1025
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/vit_ffn_ablation_1025/summarize.py \
  --results results/vit_ffn_ablation_1025 \
  --archive /home/issan/Projects/obsidian-research-data/vit_ffn_ablation_1025
```

`per_checkpoint.csv` has seven interventions for each of 80 checkpoints: six final-FFN multipliers, plus removal of the entire last block. `aggregate.csv` gives seed medians and ranges. These are task memorization measurements on 250 images per task, not generalization accuracy. Baseline replay and independent hook interventions are checked before computing the ablation comparisons. Accuracy loss under removal establishes a contribution to current predictions; it does not quantify gradients, Adam updates, or the cause of preactivation sinking.
