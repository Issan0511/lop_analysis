# readout_lift_1111

This follows the user's request to test why small readouts can produce activity growth while sinking persists. It is separate from the concurrent readout_separation_1111 experiment, which studies early scale/direction/optimizer effects and ViT. This experiment focuses on selective ELU growth, persistent labels, and a controlled negative-body trajectory.

Specification commit: `2edd133a`. Implementation and pre-main clarification: `6fe11a240072786ab322f78bbe9ed0062c398441`. Trained code is `src/readout_lift_1111.py`; helpers are the unchanged `stage1_ctrl_1111.py` and `demand_allocation_1111.py`. Every cell records source hashes and uses archived pre-task10/25 checkpoints, seeds0,1,2. Source state reconstruction is unnecessary because complete Adam states are saved. S/Q/QV endpoints are checked bitwise against the previous run.

Use an isolated worktree following `/home/issan/Projects/claude/CLAUDE.md`. Python is `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`. Inputs are the MNIST cache there and `/home/issan/Projects/obsidian-research-data/demand_allocation_1111/results/demand_allocation_1111/raw/ELU_s*/baseline_before_t*.pt`.

```bash
python src/readout_lift_1111.py --synthetic --out /tmp/lift_synthetic.json
python src/readout_lift_1111.py --smoke --workers 1 --out /tmp/lift_smoke_new
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/readout_lift_1111.py --workers 2 --out results/readout_lift_1111/raw
python analysis/readout_lift_1111/summarize.py --raw results/readout_lift_1111/raw --out results/readout_lift_1111
```

Outputs must be new directories; existing cells are never overwritten. Main has54 continuations. Job-level completion files contain checks, hashes, and elapsed time. State/source/counterfactual CSVs are separated; `ends.csv`, `endpoint_means.csv`, `contrasts.csv`, `late_sources.csv`, `summary.md`, `verdict.csv`, and `checks.json` are the reviewer-facing tables. `readout_lift.png/.pdf` is a static scientific figure. Uniform-target/fresh-label curves are evaluated on the fixed hard labels and are not expected to learn those labels.

`G` is the mean unit maximum minus the mean over that unit's initially deep input group, using only units with >=30 initially deep inputs. Its change can grow even if the maximum decreases, so absolute maximum, activity, and opening are always reported alongside. Fixed upper groups are initial top5%, current top groups are reselected. Native float32 ELU numerical saturation is preserved.

Matched arms project hidden bias AFTER each ordinary Adam update so the initially deep body's mean equals V1 at every step; Adam moments are retained. This is an external force and does not match the entire distribution or imply a mediator percentage. The endpoint virtual probe holds the final reference constant. Label/uniform/old-moment source accounting uses the observed denominator; source deletions are the separate trained objective controls.

`audit_old.py` is explicitly post-hoc, uses September vfreeze tasks151-200, and is not pooled with registered seed/task pairs. Archive ignored raw and logs with a SHA256 manifest before cleaning up the worktree. No manuscript changes.

## Registered follow-up after the failed original matched control

Original V01UM was unstable and original P4 versus that control failed. Addendum `98cde383` and exact-gradient implementation `c9d2bbc6` were committed before follow-up outcomes. Differentiate through beta(W)=m_ref-W*xbar_B; clear W Adam in all four arms. The output bias retains its history. This is a separate controlled model, not a replacement of original results.

```bash
python src/readout_lift_constrained_1111.py --synthetic --out /tmp/lift_constrained_synthetic.json
python src/readout_lift_constrained_1111.py --smoke --workers 1 --refs results/readout_lift_1111/raw --out /tmp/lift_constrained_smoke_new
python src/readout_lift_constrained_1111.py --workers 2 --refs results/readout_lift_1111/raw --out results/readout_lift_1111/followup_raw
python analysis/readout_lift_1111/followup_summary.py --raw results/readout_lift_1111/followup_raw --out results/readout_lift_1111
python analysis/readout_lift_1111/selectivity.py --raw results/readout_lift_1111/raw --follow results/readout_lift_1111/followup_raw --out results/readout_lift_1111
python analysis/readout_lift_1111/mechanism_tables.py --out results/readout_lift_1111
python analysis/readout_lift_1111/verify_outputs.py --out results/readout_lift_1111
```

The selectivity and200->500 window diagnostics are post-hoc and marked accordingly. `interpretation.md` gives the final causal scope; `followup_summary.md` preserves all follow-up comparisons. Both static figures were visually inspected. A missing optional tabulate package was handled by a local Markdown table renderer; no dependency or training change was needed.

After archival, restore raw/aggregate files by the exact relative paths in `backup_manifest.json`, or point the raw arguments to the archive directories. The constrained run consumes the original V1_trace.npz references. Source checkpoints remain in the separate demand_allocation_1111 archive and are never modified.
