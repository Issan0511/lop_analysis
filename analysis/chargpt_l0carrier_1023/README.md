# L0 saved-state carrier ledger

Spec: `specs/spec_chargpt_l0carrier_1023.md`, registered before measurement.
No training, optimizer, parameter updates, workers or subagents.

```bash
python analysis/chargpt_l0carrier_1023/measure.py gate
python analysis/chargpt_l0carrier_1023/measure.py run
python analysis/chargpt_l0carrier_1023/report.py
python analysis/chargpt_l0carrier_1023/archive.py
```

The model loader and probes are imported from the previous loadsign/load10
analysis. End checkpoints are read directly from the existing seed10 tar;
u100 checkpoints and reference CSVs are read from the original chargpt_cl_1019
worktree, without modifying it. The archive manifest preserves verified copies
of all borrowed inputs. If the original worktree is later absent, its M2 input
tree can be reconstructed from `backup_manifest.json`'s `input_restore_root`,
or that root can be assigned to `measure.SOURCE` when importing the analysis.

There are 700 saved models, 1400 endpoint state/probe rows, 1360 endpoint
pair/probe ledger rows, and a supplementary 220 u100 states / 440 head-body pairs.
The cohort is fixed by any positive activation on the task10 uniform probe.
It is reused for the chain probe and never filtered by later survival.

`states.csv` has all-unit and cohort medians. `ledger.csv` contains each exact
endpoint pair; `windows.csv` first sums per unit and only then takes medians.
`cohort_pair_median_*` is a separately labelled temporal-median sensitivity.
The median columns do not add. Mean columns do add, and the unit-level arrays
retain the complete additive ledger. `u100_states.csv`, `phases.csv`, and
`phase_windows.csv` are supplementary; they do not replace the registered chain.

The uniform input sequences are fixed across checkpoints. Chain probes change
with the current task, so their differences also include distribution changes.
All standard deviations use ddof=1; covariance PR is scale-invariant. The gate
reproduces the legacy lower-middle sd/mu medians separately from the regular
two-middle-average medians used in the new analysis.

With no bias, r does not explicitly depend on the row norm. A synthetic positive
row rescaling checks exact cancellation of the stretch and width terms. A large
negative stretch contribution alone is therefore not a causal explanation of
relative sinking. Endpoints separated by many tasks also cannot identify the
within-task learning path; inserting checkpoints changes the allocation.
