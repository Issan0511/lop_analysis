# New run 1: RC/raw/ReLU, periodic ReDo × CHB

This is a new registered campaign, `redo_chb_rlcifar_1018`, based on main
`094489c3`. The initial preregistration is `de8a437f`; generation definitions were
clarified before new evaluation in `09d175aa`. The prior F51 trial remains
preserved: it completed at `abd5220b`, its result was MIXED, and its
POSITION_CARRIES prediction was not supported. These programs neither rerun nor
write into that old trial.

The production protocol is fixed: CIFAR10, seed-specific fixed1200 images, raw,
3072–100–100–10, CUDA float32, R10/seeds10–19, batch16, Adam1e−3, 200 tasks ×
400 epochs ×75 updates. Arms run sequentially none→redo→CHB→both, all fresh from
task1. CHB is the inherited C/H/B package (lambda .1713, beta .01). Periodic ReDo
uses tau .1, F1560 absolute updates, after native Adam→B→EMA, on the latest four
batch16 image-id arrays. It scores actual features delivered after H, retains
EMA, bias moments and the Adam clock, and follows the original incoming-rows
then outgoing-columns recycler. This is an F51 policy transfer without RC
tuning. It is not a bias-only intervention or a universal claim about ReDo.

Primary readout uses seed-first online means over tasks181–200 and seven paired
Student intervals with Bonferroni m7, df9. CHB_POLICY_CARRIES / REDO_POLICY_CARRIES
are the policy labels; ADDITIVE is an independent flag. REDO_SHORTFALL requires
lower(CHB−ReDo)>.005. No absolute .99/.5 success criterion was added. The adopted
directional prediction is CHB_POLICY_CARRIES plus REDO_SHORTFALL; no numerical
probabilities or precision estimates were supplied by Issa.

From the new worktree:

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
$PY -m analysis.redo_chb_rlcifar_1018.checks
$PY -m analysis.redo_chb_rlcifar_1018.gpu_checks
$PY -m analysis.redo_chb_rlcifar_1018.launch preflight
$PY -m analysis.redo_chb_rlcifar_1018.launch run --arm all
$PY -m analysis.redo_chb_rlcifar_1018.report
```

CPU checks use analytic inputs, seeds100–101, small trainable affine maps and
explicitly reduced synthetic fixture dimensions. They execute the actual
native/adapted functions and inherited recycler. They are never production
results. CUDA release checks are a separate script using real CIFAR and R10
implementation seeds100–109. Neither script creates independent source approval.

Before production, all sources must be committed and the source-bound
`cpu_checks.json`, `independent_review.json` and `gpu_release.json` must pass.
The review must identify an independent reviewer and use scope
`CPU_SOURCE_REVIEW`; the GPU proof uses `CUDA_REAL_DATA_RELEASE`. Missing CUDA,
stale hashes, missing proofs and lock contention are stored as execution states,
never scientific failures. A GPU-less launch does not load evaluation images or
train a shortened CPU substitute.

The launcher and standalone GPU checker each acquire the same shared
`/tmp/lop_analysis_gpu.lock` once. A queue wrapper must **not** acquire another
flock around either CLI. No existing host queue/controller/state/G0 is modified.
An external scheduler can use `launch run --arm next`; the new CLI validates
completed prior arms and executes only the next registered full arm. It exits2
on a busy lock for a later retry. It does not start a background scheduler itself.

Each task checkpoint has a separately persisted source/data/config-bound SHA
envelope before the expensive task observer. A crash between atomic checkpoint
replacement and its envelope fails closed; an unbound checkpoint is never
blessed on resume. Native optimizer/EMA/RNG, latest four batches, reinit RNG,
event clock, reset generations and stream-audit state are restored together.
Task-sidecar recovery after a bound checkpoint does not advance learning RNG.

The StreamAudit hashes actual labels and every complete CPU-generated epoch
permutation. Image/subset order, archive bytes/resolved path, raw input and
initial parameters are bound in provenance; all four arm streams must agree
before scientific labels. Every task and event-pre/event-post saves complete
unit metrics on the fixed population, strict/native gate distinctions, zero
width/infinite ratios and generation lineage. Replaced-live generations are
censored; event-associated deaths are separate from between-event observations.

A `STOP` file requests a task-boundary stop. Completed arm markers bind all
artifacts, including checkpoint envelopes, diagnostics and stream hashes.
`report --verify-only` recomputes a completed admitted result read-only, without
requiring a currently attached GPU. Naked CSVs, missing cells, finite CSV edits,
unqualified checkpoints and foreign run namespaces cannot gain registered
performance labels. Source changes require fresh checks/review/release; old
outputs are not silently reinterpreted under a new implementation.
