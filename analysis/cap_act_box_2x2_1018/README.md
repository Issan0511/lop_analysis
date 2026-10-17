# cap_act_box_2x2_1018

This implements the registered cap activation × box experiment on the actual F51 and RC engines. The source contains an explicit copy of `cap_cifar5p1_1007.run`, with native activation selection, graph-readable layer learning rates, real-displacement measurement, and the cap12b difference. RC subclasses `cifar_interventions_0920.Engine`; initialization, random streams, Adam arithmetic and online scoring are inherited. The CPU checks use marked synthetic check fixtures with the real input/hidden/output dimensions. Production has no reduced-task, reduced-seed or CPU fallback option.

The default execution reruns all four ref/cap12 cells (including the two historically observed cells). Historical CSVs are not reused without qualification, and this runner does not expose an unchecked reuse flag. This costs eight evaluation arms plus the two F51/ReLU separation arms, in addition to ref/cap12 calibration. The old cells are explicitly described as known-result reruns, not new independent predictive successes.

The paper-derived forecast is `ACTIVATION_SPLITS` (common spec §8). It is separate from the observed verdict. No numeric probability or accuracy prediction was invented.

## Commands

Run commands from this experiment's worktree, using the existing project interpreter:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.cap_act_box_2x2_1018.checks
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.cap_act_box_2x2_1018 preflight --integration-audit /tmp/issa_six_specs_1018/integration_audit.json
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.cap_act_box_2x2_1018 calibrate --integration-audit /tmp/issa_six_specs_1018/integration_audit.json
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.cap_act_box_2x2_1018 run --integration-audit /tmp/issa_six_specs_1018/integration_audit.json
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.cap_act_box_2x2_1018 analyze
```

`checks` runs the CPU S suite and then the actual CUDA R=10 graph/rollback/native-host checks when CUDA exists. `cpu_checks_pass` and `all_pass` are deliberately separate: a CUDA-unavailable check is `BLOCKED_CUDA`, never `PASS`. Production refuses missing/stale/failed checks, absent CUDA or data, dirty source/spec trees, or an incomplete independent G0 audit. The canonical audit must contain the root auditor identity, source hashes, verified main ancestry, actual host reproduction and mutation-check proof. A static review alone is not sufficient. The GPU exclusive lock is inherited from the native RC tooling.

`calibrate` fixes q from seeds 100–109, not evaluation seeds. q is the median over calibration seeds of the cap12/ref per-task relative real-displacement RMS for each hidden weight layer. It can exceed one and is never clipped. `lr_schedule.json` contains the source closure and calibration CSV hashes, and is independently recomputed before evaluation. Both calibration arms must be COMPLETED with full per-task/cap/fresh coverage, intact source/data/stream evidence and verified t1 radii; cap-check-failed arms are rejected even if their U rows exist. Task 1 is lr=1e−4 for all parameters; afterward the two hidden W/b pairs receive their layer schedule, and the output pair stays at lr=1e−4. No row cap executes in `lr_only`.

`run` keeps R=10, seeds 0–9, F51 30 × 780 updates, RC 50 × 30,000 updates, and the native pipelines. `cap12b` derives each coefficient from W only and applies it to W and the corresponding bias. It does not project a concatenated (w,b) norm or touch moments/output tensors. Per-task output includes the actual layer lrs and U statistics; unit arrays include native derivatives, population m/s/r, zero-width signs, projection counts/removal and pre/post maximum norms. Initial states, endpoint snapshots, radii, task image/label/batch hashes, source closure and data hashes are saved. Diagnostics do not advance the training streams.

Existing arm output is refused to prevent accidental overwrite. RC checkpoints contain the native full training state; F51 snapshots are endpoint diagnostic snapshots and are not advertised as exact resumable states. If interrupted, preserve the partial arm directory and rerun that arm from its beginning under the identical committed sources; do not splice a checkpoint with missing stream/Adam-time state. Automatic arm-level recovery is intentionally not claimed.

`analyze` requires full seed/task coverage, an immutable snapshot of execution admission, unchanged calibration/source identity, actual archive/input-content hashes plus label/batch pairing, and task-1 no-cap equivalence. It uses the eight simultaneous Bonferroni intervals and the separate four-contrast separation family exactly as registered. Ordinary 95% intervals, all ten seed differences, sign counts and the report-only continuous interaction are retained. `UNRESOLVED` is never converted to `NO_RESCUE`. Both separation labels can coexist. Realized step-matching mismatch is reported; calibration does not imply matching directions or evaluation trajectories.

## Current environment

CPU S checks and mutation negatives were executed; full CUDA checks and production are blocked because torch reports no usable CUDA device. `checks.json` therefore records `all_pass: false`. `preflight.json` and `verdict.json` record the full-run admission attempt without assigning a scientific verdict. No accuracy data or successful rescue was fabricated. The registered reference/evaluation/calibration workloads have not run.

Raw checkpoints and unit arrays should be archived under the project `CLAUDE.md` rule before worktree cleanup. This branch must be integrated by the root only after independent review; implementation and check results are locally committed, while publishing is pending the root's network resolution.
