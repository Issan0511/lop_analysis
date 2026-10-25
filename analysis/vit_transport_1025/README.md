# ViT transport follow-up (2026-10-25)

The initial specification/evaluator was committed at `f7636441`. The gate identified a float32/float64 reconstruction mismatch, not a changed training trajectory; `validation_notes.md` records the diagnosis and `d9f2661e` fixes it without loosening the registered threshold. The input-set check was separately registered at `06b00f0b` before running it. All runs use the five original seeds, task 10/60 end, and original archived model/source hashes.

From the repository root, using `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`:

```bash
python analysis/vit_transport_1025/measure.py --mode carriers
python analysis/vit_transport_1025/measure.py --mode replay
python analysis/vit_transport_1025/measure.py --mode freeze --tasks 60
python analysis/vit_transport_1025/probes.py
python analysis/vit_transport_1025/report.py
```

Default raw/source paths are permanent paths in `obsidian-research-data`. The gate uses `--mode gate --seeds 0 --tasks 10` and separate `--output` and `--raw-output` paths. Original archives are read in place; nothing in the original experiment is changed.

- `carriers.csv`, `ratios.csv`: projections of three full-body signals through disjoint parameter groups into all six FFN positions; SGD and fixed saved Adam denominator. The latter excludes momentum.
- `replay.csv`: two actual 50-step mini-batch Adam trajectories per initial state. Each finite step is divided into own/upstream and mean/width contributions; these sum to the observed change. The original 100->500 interval is a separate post-hoc product decomposition in `original_100_500_channels.csv`.
- `freeze.csv`: independent 50-step trajectories with the selected fc1 or its upstream parameters fixed. Momentum cannot move frozen parameters because their gradients are `None`.
- `probe_states.csv`, `probe_drifts.csv`: forward-only original 100/500 checkpoints on current training, next-task, and fixed validation images. These do not manipulate input variance itself.
- `*_checks.json`, `*_provenance.json`, `*_run.json`: validation, original sources, completeness, and execution commits. `backup_manifest.json` also covers unsuccessful gates, preserving the debugging record.

The theory in `theory.md` distinguishes logit-space contraction/expansion from the signed transport into a hidden unit. Additive bookkeeping checks validate the calculation; the freeze interventions provide separate evidence about the update route. None of these values is a uniquely defined long-term causal percentage.
