# New RC ReDo × CHB run: standalone queue contract

Advisory only. No active controller, service, source, HEAD, status, or lock was changed by this review. The new preregistration is `de8a437fef35f7942023fc581e81f4e6cbbad5bd` in `/home/issan/Projects/claude/wt/redo_chb_rlcifar_1018`.

## Supersession and retained evidence

The old F51 run `redo_position_2x2_1018` is complete at `abd5220b` and is superseded as a primary experiment by the user instruction recorded in the new spec. Preserve its original spec, prediction, COMPLETE status, outcome, artifacts, and hashes. Do not rewrite its scientific status to SUPERSEDED. Record the campaign role separately in the NEW worktree, e.g. a supersession record with `old_run_id`, `old_result_commit`, `superseded_for_primary_claims=true`, `replacement_run_id=redo_chb_rlcifar_1018`, `future_rerun=false`, and the instruction timestamp. The new preregistration already states the no-rerun requirement.

There is no existing machine-enforced SUPERSEDED registry. Future launchers must explicitly honor that record. Do not invoke the old run's production CLI: even a completed invocation rewrites admission/intervention files and executes fresh control again. A per-run STOP is not a supersession guard; global `host_continuation/STOP` pauses unrelated work.

## Existing host queues stay immutable

G0 worktree: `/home/issan/Projects/claude/wt/f51_integration_audit_1018`, reviewed HEAD `1c5494016c2ccf8239cae3262dc1a944021492c2`.

- `analysis/f51_integration_audit_1018/continue_gpu.py` hardcodes six run ids. Its `--runs` choices cannot accept the new run. `--independent-queue` does not remove that restriction. A fresh state path is mandatory; existing state paths are rejected.
- `/run/user/1000/systemd/transient/six-gpu-1018b.service` explicitly selects the five other runs; old F51 run1 is already excluded.
- `analysis/f51_integration_audit_1018/finish_gpu_queue.py` handles only CHB and CNN. It waits for prior controller/recovery locks. Do not add the new experiment to its in-memory plan or modify its source.
- `analysis/f51_integration_audit_1018/retain_gpu_outputs.py` handles the existing registered plan. It will not automatically retain this new experiment.
- Current writer-owned state files are under `results/f51_integration_audit_1018/host_continuation/`: `state_after_repairs.json`, `state_cap_retry.json`, `state_nap_readout.json`, `completion_state.json`, `retention_status.json`. Treat them as evidence, not an injection interface.

The existing `source_identity()` binds `HEAD:src`, `HEAD:analysis`, and `HEAD:specs` for queued runs and rejects dirty scientific source. Do not edit those trees in an active worktree or change an active source/HEAD to add this run. The new worktree is independent.

## New-run lifecycle and admission

Before implementation, an honest new-run record may say `PENDING_IMPLEMENTATION`; it is not an executable GPU queue item. The new launcher must support the following semantic transitions without fabricated PASS values:

`PENDING_IMPLEMENTATION` → `PENDING_CPU_REVIEW` → `PENDING_GPU_RELEASE` → `READY` → `WAITING_FOR_SHARED_GPU` → `RUNNING` → `COMPLETE` / `STOPPED_AT_CHECKPOINT` / `DIVERGED` / `CHECK_FAILED`.

Actual status names may follow the authored CLI, but these distinctions must remain explicit. No launch command is given here because the CLI was still being authored when this contract was written. Use only its reviewed, real CLI after it exists.

1. Commit the preregistered engine, diagnostics, launcher, numerical checks, and readout. Bind all imported source hashes, spec hash, source commit, fixed native box, seed/arm order, and exact configuration.
2. Obtain independent source/CPU approval and real CUDA R=10 native/graph/warmup/cadence checks. A missing/unavailable CUDA device or busy GPU records a blocked state; Work must not bypass its device isolation.
3. GPU checks and training must use the SAME existing shared lock path, `/tmp/lop_analysis_gpu.lock`. Exactly one layer owns it. If the new CLI acquires the lock internally, do not wrap it in another independent `flock` on the same path. If a check has no internal lock, the standalone controller must hold the shared lock while that child runs. Preserve existing desktop-versus-competing-training admission rather than weakening it.
4. Never overlap current long runs. Use a dedicated new-run queue/state owned by the NEW worktree; it may wait on the shared GPU lock, or remain pending until the current owner confirms the existing finish queue drained. Do not create/delete another queue's STOP file, kill a process, stop a service, steal a lock, or rewrite another state. Locks and current source-bound admission must be rechecked after waiting.
5. Only enqueue the registered four-arm sequence `none → redo → CHB → both` after the source-bound real GPU release exists. Preserve R=10, seeds10–19,200 tasks,30000 updates/task. Release checks are implementation evidence, never shortened substitutes for production. No F51 G0 requirement should be invented for this RC/raw box; the required native RC/CHB parity and CUDA release are separate gates.
6. Freeze scientific source and, if the checkpoint identity includes it, the active HEAD for the full run. Diagnostics cannot consume label/batch/reinit RNG. Resume only verified matching checkpoints including cadence, recent four batches, reinit generators, reset generations, Adam state, EMA, and source/data identity. Do not overwrite mismatched or incomplete outputs.
7. Completed evidence must include exact coverage, hash-bound outputs/checkpoints, real GPU release, independent review, data/subset/stream pairing, and the registered seven-interval analysis. Do not convert a scientific prediction failure into an implementation failure.
8. Retain this experiment's raw outputs separately under its own run id before cleanup, with SHA256 verification and compact result commits after active training is finished. Existing G0 retention does not cover the new run. No GitHub publishing is authorized by this contract; the prior automatic review rejection remains in force.

## Author's planned CLI contract (02:44 JST; verify after code is ready)

Agent1 reports the intended entry point is `python -m analysis.redo_chb_rlcifar_1018.launch {preflight,run} --out results/redo_chb_rlcifar_1018 [--arm next|none|redo|CHB|both]`. The launcher itself owns one nonblocking `/tmp/lop_analysis_gpu.lock`; an outer flock wrapper must not be added. This paragraph records an API plan, not evidence the CLI/release has passed.

Planned admission records are `cpu_checks.json` with scope `CPU_SYNTHETIC_ONLY`, `independent_review.json` with scope `CPU_SOURCE_REVIEW`, and `gpu_release.json` with scope `CUDA_REAL_DATA_RELEASE`. All must match the deterministic imported-source closure; the GPU release must include the fixed registration/data and check seeds100–109 plus actual native-none/CHB, graph/eager, and warmup evidence. The reviewed implementation is the authority for the final field names and exact invocation.
