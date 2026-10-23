# CHB decomposition — registered run 3

This implementation runs the three new preventive interventions from task1,
with the unmodified `relu_doors_0919` engine's CUDA/R10/raw/ReLU/200-task box.
The original pre-implementation registration is `5c741be8`; the shared spec §8
records the authorized paper-derived prediction `BIAS_STOP_SUFFICES` without
inventing Issa's subjective probabilities or numerical forecasts.

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.chb_decompose_200_1018.launch check
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.chb_decompose_200_1018.launch plan
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.chb_decompose_200_1018.launch run
```

Use `run --resume` only for this run's existing launch manifest/checkpoints.
All commands run from this repository root. The production command has no
CPU fallback, seed/task reduction or lambda override. CUDA absence writes
`execution_status.json`, `provenance.json`, `verdict.json` and `summary.md`
with `BLOCKED_CUDA`, and exits 78 without producing a scientific result.
GPU workload admission uses the shared `/tmp/lop_analysis_gpu.lock` and an
external compute-process check. Coordinate with the root scheduler before
starting it. `STOP` stops at task boundaries after saving the checkpoint.

The extension compiles the hash-locked host `run()` with uniquely checked
local substitutions. It never rewrites the host file or changes its module
globals. `CH_bfreeze` sets b1/b2 `requires_grad=False`, excludes them from
`autograd.grad` and Adam updates, and retains only permanently zero moment
buffers. `CH_wWD` decays W1/W2; `CH_bWD` decays b1/b2 after Adam. W3/b3 are
free. Native CH/CHB follow byte-identical CPU fixture trajectories.

Historical CH/CHB eligibility is recorded against the complete original
seed10–19/t1–200 CSVs, source manifests and numerical environment. Eligibility
is never reported as natural reproduction. The launcher conservatively
reruns CH and CHB from task1 in the current environment, followed by the
three new arms; it never mixes unqualified historical means. This costs two
extra baseline arms but requires no modified scientific condition.

Diagnostics store every hidden unit's b/m/s/r/native derivative mean and
population variance/death/W norm every task, and preserve signed nonfinite
r values with explicit counts when s=0. Native float32 forwards feed
float64 reductions; s uses correction=1 as in the host, derivative variance
uses correction=0. The first-layer mean identity has an a priori gamma-n
float32 dot-product bound. Adam candidate, WD displacement, total actual
displacement and relative actual displacement are measured at every update.
Observers use no training RNG and do not mutate P/m/v/EMA. CUDA graph warmup
accumulations are reset before task1. Checkpoints include the accumulator
and previous RNG state, permitting exact sidecar recovery after a crash
between the host checkpoint write and diagnostic writes.

`checks.json.cpu_all_pass` describes only synthetic CPU checks. Its overall
`all_pass` remains false while CUDA checks and registered production remain
pending. CUDA checks repeat host parity, graph rollback, frozen-bias and
resume checks at production tensor shapes/R10 on calibration seed100–109.
CPU fixtures are never treated as evaluation-seed production results.

The report requires all five arms × ten seeds × 200 unique tasks, matching
checkpoint performance and clocks, paired initial states/label/batch hashes,
all 200 per-unit diagnostic files per arm, and complete diagnostics. It
computes all seven registered Bonferroni intervals with df9, independently
of the hypothesis outcome. POS/NI are strict at their lower bounds and EQ
includes its endpoints; overlapping patterns yield MIXED. Bias sufficiency
is a claim about these preventive interventions, and WD benefit does not
identify effective learning rate as its sole pathway.
