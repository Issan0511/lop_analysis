# Direct P-residual loading factorial

Spec: `specs/spec_p_residual_loading_1111.md` (359d0453). Frozen training implementation: 0c874fe9fc27e46c695d7c41e55f90d92a03949d. Training source/helper/spec and input archive hashes are embedded in every raw JSON. Later summary/plot commits do not change this training implementation.

Use the project Python `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python` from this repository at the saved result commit. Summary validation intentionally rejects changed training helpers/spec/source archives. Reproduce from that commit if later main changes these files.

```sh
python -m analysis.p_residual_loading_1111.run --synthetic --out /tmp/p_residual_synthetic
python -m analysis.p_residual_loading_1111.summarize --self-test
python -m analysis.p_residual_loading_1111.verify_summary
python -m analysis.p_residual_loading_1111.summarize --inputs /home/issan/Projects/obsidian-research-data/p_residual_loading_1111/results/p_residual_loading_1111/raw
python -m analysis.p_residual_loading_1111.report
```

The report falls back to the archived raw directory after worktree cleanup; `--raw` overrides it. The summary writes the registered CSVs, plots and verdict. `mlp_analysis.md` provides the detailed MLP-only interpretation, formed before inspecting ViT results. Main counts are820 logical records,690 direct trajectories,130 exact shared identities (65 S/P-ZC and65 KK suffixes); shared curves do not create independent seeds.

To train the full fixed design anew, use `python -m analysis.p_residual_loading_1111.launch` only after checking the spec and resources. It runs four single-thread CPU workers plus one GPU worker. This is separate from the read-only analysis above and need not be repeated for a summary.

All ignored/untracked experiment observations, smoke source and logs are archived by `archive.py --execute` only after source/results are committed. `results/p_residual_loading_1111/backup_manifest.json` maps each original path to the verified archived path and SHA256. The archive command is restricted to the named isolated worktree. It does not delete other experiments or branches.
