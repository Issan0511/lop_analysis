# CNN CH′ + width-relative bias projection (1018)

Implementation is complete; no registered training result exists. Production is
CUDA only, R=10, seeds10–19, 30 tasks × 400 epochs × 75 updates, Adam1e−3. κ=.2
is fixed. The adopted paper-derived prediction is RESCUED (common spec §8).

The source subclasses P183 without editing historical engines. Training follows
P183 forward/backward → Adam → EMA. One additional bias-free measurement forward
then clips c1/c2/f1/f2 sequentially to ±κ times the current minibatch image-axis
width (conv RMS across positions). Width uses float64 aggregation; q is computed
without bias. Downstream uses q plus corrected bias. W, readout bias, Adam moments
and time are untouched. Native activation remains P183 `torch.clamp(min=0)`;
its actual derivative at zero is one. Conventional gate z>0 is separately named.

Probe updates1/10/100/1000 require eager epochs1/2/14 of each task. The remaining
397 epochs retain the parent epoch CUDA graph. Shapes/R/slot order are unchanged.
CUDA graph/eager equality and graph warmup rewind are an explicit GPU release
gate, unexecuted on the current machine. Numeric divergence stops at the first
host synchronization (each update in probe epochs; otherwise epoch boundary).

All per-update channel candidates/widths/post-bias and batch orders are written
in compressed epoch NPZs. The uncompressed trace payload alone is35,496,000,000
bytes (33.06GiB); production reserves twice this for traces/diagnostics/checkpoints.
The B-disabled fresh CHf route instead reserves8GiB. The fixed1200 image probe
stores full per-position image-axis m/s/r and native derivative moments, distinct
from inherited P183 image×position sample SD. NPZ retains explicit 0/0 NaN and
nonzero/0 infinity. Projection summaries count zero widths separately.

CPU synthetic checks are meaningful implementation checks, not reduced training.
They cover the P183 disabled path, independent NumPy64 sequential projection,
zero/tiny widths, spatial mean confound, nonzero moments, EMA order/lifetime,
logger/resume invariance, and primary label boundaries, with specified mutations.
CUDA checks, independent review and performance remain separate and pending.

P183 CHf artifacts (CSV/provenance, units, state,70 checkpoints) were checked by
hash. P183 records cudnn TF32=true; this registration fixes TF32=false. Therefore
P183 is not reused as the paired control. A fresh CHf run is implemented and
required for the paired secondary comparison. The main RESCUED/FAIL/PARTIAL
threshold remains evaluable after the complete primary arm alone.

Exact commands from this worktree (PY points to the existing runtime):

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
$PY analysis/cnn_chprime_bias_1018/checks.py
$PY src/cnn_chprime_bias_1018.py preflight --out results/cnn_chprime_bias_1018/CHf_B
$PY analysis/cnn_chprime_bias_1018/gpu_checks.py
$PY src/cnn_chprime_bias_1018.py run --arm CHf --out results/cnn_chprime_bias_1018/CHf
$PY src/cnn_chprime_bias_1018.py run --arm CHf_B --out results/cnn_chprime_bias_1018/CHf_B
$PY analysis/cnn_chprime_bias_1018/verdict.py
```

The independent reviewer creates `results/cnn_chprime_bias_1018/independent_review.json`
with `{ "approved": true, "reviewer": "<reviewer identity>", "sources": <source_hashes()> }`
only after review. GPU checks generate `gpu_release.json`, never by manual
substitution. Both source-bound CPU checks and the GPU release are required to
train; a blocked CUDA preflight exits2 and creates no accuracy rows. Rerunning the
same production CLI resumes only complete-task checkpoints and rejects any
changed source/configuration/κ. Partial task files are overwritten from the last
complete task, never treated as complete. Fresh CHf may be scheduled before or
after CHf_B; it does not gate the primary threshold result.
