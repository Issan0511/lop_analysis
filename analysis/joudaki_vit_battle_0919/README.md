# Joudaki ViT / Tiny ImageNet activation battle

Specification: `specs/spec_joudaki_vit_battle_0919.md` and its pre-run addenda.
This experiment compares 13 FFN activations in the original Joudaki ViT. It uses
40 disjoint five-class tasks, 500 updates per task, Adam 1e-4, no weight decay,
float32, and paired seeds 0–9. No main-run results have yet been inspected.

The dataset is shared at
`/home/issan/Projects/obsidian-research-data/datasets/tiny-imagenet-200`.
Raw outputs go directly to
`/home/issan/Projects/obsidian-research-data/joudaki_vit_battle_0919`.
Do not move or remove the shared dataset during worktree cleanup.

Run from the repository root, with the shared Python runtime:

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
$PY -m analysis.joudaki_vit_battle_0919.checks
$PY -m analysis.joudaki_vit_battle_0919.launch
```

`launch.py` is exclusive via a filesystem lock. It advances one GPU job at a
time, retries process failures at most twice, resumes only committed task
boundaries, and generates the final table/figure when all 130 runs terminate.
Writing `STOP` in the raw-output root pauses at a task boundary. Remove it and
run the launcher again to resume. Do not change Python sources during a run;
source/config mismatches deliberately prevent resume and mixed reports.

The registered main accuracy is restricted to the current five classes.
`online_global_acc`, `train_global_acc` and `val_global_acc` also record the
upstream implementation's argmax over all 200 outputs. The loss is masked to
the active classes in both implementations. Classification labels retain true
class semantics; this is not random-label CIFAR.

The upstream source and configs are pinned under `upstream/`. Pillow
augmentations follow the upstream distributions, with independent deterministic
sample streams for paired comparisons. Private RSL generators do not consume
the model's dropout stream. The six adaptive FFN modules compute per-channel
variance across both images and all tokens, including CLS, and freeze EMA in
evaluation.

`model_tNN.pt` includes all parameters and activation state. `preact_tNN.npz`
contains per-layer preactivations for 16 fixed validation images, their indices
and paths. `checkpoint.pt` additionally contains Adam state, RNG state, metrics,
and the manifest. The diagnostics' derivative is computed by autograd through
the actual activation, rather than an approximate zero-output surrogate.

The report refuses incomplete production runs, smoke runs, altered training
budgets, wrong seed identities, class-order mismatches, and mixed source hashes.
Paired sign-test labels reproduce the earlier battle's convention, while all
14 distinct comparisons also report Holm-adjusted p values. A non-significant
comparison does not prove equality or establish a mechanism.

Production uses `torch.compile(fullgraph=True)` plus fused Adam, float32 with
TF32 disabled. `fallback_random=True` retains ATen dropout draws. A private
copy of Ubuntu libpython3.12-dev headers lives under the raw-output runtime
folder; `train_forward` adds its include paths without modifying the system.
Compilation is limited to four CPU workers. Adaptive variance is computed in
the forward, but its EMA is updated only after backward/Adam. Updating EMA
inside the compiled forward failed gradient-equivalence checks and is not used.

Additional preflight commands:

```bash
$PY -m analysis.joudaki_vit_battle_0919.check_speed
$PY -m analysis.joudaki_vit_battle_0919.checks --engine compile
$PY -m analysis.joudaki_vit_battle_0919.check_report
```

No half precision, TF32, batch-size changes, reduced seed counts, or shorter
production horizons are used by this optimization. Fused arithmetic changes
rounding order, so the compiled trajectory need not equal the eager trajectory.
Exact continuation is tested within the chosen engine.
