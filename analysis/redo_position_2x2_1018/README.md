# ReDo × live-unit physical bias shift

Preregistration: `fd2e6682`; seed 0–9, R=10, CUDA float32, 29-task natural prefix,
branch at t28 end, score all 780 pre-update minibatch accuracies on t29.
The four arms share the class, batch, label, ReDo-mask and replacement-draw plan.
The physical b2 shift uses natural-pre-intervention L in both shift arms, the
t1 native gate-count target on t29 images, and an exact ordered-float32 search.

CPU validation (synthetic inputs only):

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.redo_position_2x2_1018.checks
```

Reproducible production launch, from the repository root:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.redo_position_2x2_1018 run --integration-audit /tmp/issa_six_specs_1018/integration_audit.json
```

The launch always checks CUDA, registered torch, the current source hashes,
CPU synthetic checks, independent G0 identity and explicit reproduction/mutation
booleans. Missing G0 or missing CUDA exits with status 2 before loading CIFAR or
training. `preflight` runs those same admission checks without training.
The GPU lock is nonblocking so scheduling remains explicit. A `STOP` file in
the output directory requests a stop at the next prefix-task or arm boundary.
Rerunning resumes from a hash-bound task checkpoint or a completed arm; it never
changes the seed count or falls back to CPU.

`checks.json` distinguishes executable synthetic assertions/mutation negatives
from the CUDA reproduction that still requires the independent G0 audit.
Existing recycler compatibility is tested by executing its unchanged nested
functions extracted through Python AST, with synthetic captured locals.
These checks produce no empirical support for the paper's prediction.

Outputs include prefix/initial and t1/t28/t29 checkpoints, shared intervention
plan, per-arm post-intervention/end checkpoints, per-task rows, native/strict
gate diagnostics and unit arrays, fresh control, seven paired simultaneous
intervals and ordinary 95% intervals, provenance and an output hash manifest.
`DEAD_CARRIES` denotes this thresholded ReDo operation because tau=0.1 also
selects some live units; the two mechanisms need not partition units.
