# CH/CHB performance replication (1011)

Run these commands from this experiment's worktree. The unchanged historical engine
`src/relu_doors_0919.py` supplies all training behavior. The wrapper only supplies
the dataset directory, registered settings, serial execution, provenance and a
task-boundary stop callback.

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.ch_chb_performance_1011.launch plan
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.ch_chb_performance_1011.launch check
```

`plan` and `check` do not query or initialize the GPU. Checks use synthetic images
and validation seed 100001 on CPU; these are implementation checks, not experiment
results. The check record includes hashes of the spec, runner, analysis and engine
dependencies. Commit and push the final specification and implementation before
the run. A source/spec change requires repeating the checks.

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.ch_chb_performance_1011.launch run --data-dir /home/issan/Projects/claude/proj_004_drift/data/cifar10 --poll-seconds 300
```

The runner records the actual launch commit and archive hash, waits for the shared
`/tmp/lop_analysis_gpu.lock` **and** for external Python CUDA compute processes to
finish, then runs CH and CHB sequentially. It ignores known desktop GPU clients;
it does not terminate any process. GPU checks while waiting are at least 300 s
apart. The same-run lock rejects a second launcher. Existing unrelated outputs
or an existing launch without `--resume` are rejected. The `checks/` and `logs/`
directories may exist before the initial launch.

To stop at the next task boundary, create
`results/ch_chb_performance_1011/STOP`. Remove that file and use the same run command
with `--resume` to continue **only this run's** checkpoint. The runner verifies
the original settings, source hashes, dataset, task count, optimizer step and
complete checkpoint/CSV history before resuming. A failure before the first
checkpoint must be inspected; the runner does not silently overwrite it.

The launcher automatically validates and analyzes both completed arms. Rebuild
the report separately with:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.ch_chb_performance_1011.report
```

Outputs: `summary.md`, `paired_ci.json`, `per_seed.csv`, `performance.png`,
`paired_performance.png`, `diagnostics.png`, plus each arm's full original
per-task CSV, provenance, checkpoint, histograms and snapshots. Numeric aggregate
outputs keep float64 precision. All ten paired seed differences are shown.

The primary quantity is CHB−CH online accuracy averaged over tasks 181–200.
Seed is the replication unit; the report uses the prespecified paired t interval.
Depth and gate diagnostics do not enter the performance conclusion. The combined
b1/b2 intervention does not identify b2 decay alone.
