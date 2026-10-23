# F51 cap: fresh learning cost versus history interaction

Status: post hoc follow-up fixed before collecting the new fresh-cap outcomes, 2026-10-18 JST. Issa requested this after inspecting the original run 2 results. This is not a new prospective confirmation of the original hypothesis. Original spec, source, verdict and raw outputs remain unchanged; this supplement belongs to the same cap experiment branch.

## Question and fixed comparison

The original `fresh_control.csv` always disables cap, including under cap12. It cannot establish whether cap itself hurts fresh learning. Add method-matched fresh controls for F51/std, ELU and ReLU, seeds 0–9, tasks 21, 23, 25, 27, 29. These five tasks are the original primary late hard-task window. Each fresh task independently starts from the original seed's P0, zero Adam moments and counter, and initial activation state. Use the identical original task classes, labels and exact 780 × 32 batch indices, lr 1e-4, architecture, float32 arithmetic, CUDA R=10 batch shape and update order. Original continual results are reused as dependent observations, not counted as new replications.

Fresh arms: ref (no projection) and cap12 (W1/W2 only, after every Adam update including the first). Cap12 uses the **same per-unit radii measured at the original task-1 end** for that seed and activation. Do not estimate a new radius on the fresh task. Do not project P0 before the first forward pass. Biases, W3/b3 and Adam moments are not projected. Both fresh arms use identical P0 and examples. No hyperparameter search, threshold adaptation, extra task budget, or new seed selection.

Primary outcome is online accuracy, averaged within each seed across the five tasks before inference. Let F0/Fc be fresh ref/cap and C0/Cc be the original continual ref/cap. Report all four means and paired contrasts in percentage points:

- fresh cap effect Df = Fc − F0 (negative means worse fresh learning);
- continual cap effect Dc = Cc − C0;
- additional history interaction I = Dc − Df;
- matched gaps G0 = F0 − C0 and Gc = Fc − Cc (so I = G0 − Gc).

Report paired Student-t intervals (df=9) with Bonferroni family-wise 95% coverage across the six contrasts {Df, Dc, I} × {ELU, ReLU}. The five tasks within a seed are not independent samples. Also report all per-seed and per-task results. End-of-task train/test accuracy and projection activity are descriptive diagnostics.

If Df < 0 with an interval below zero, cap harms fresh learning at this fixed radius. If I < 0 with an interval below zero, continual learning has additional harm beyond that fresh cost. Neither a non-significant I nor similar point estimates proves equal effects; only an entire interval within ±0.5 percentage points supports practical equivalence. Positive I means a more favorable cap effect under history. Do not assert a unique biological/geometric mechanism or generalization outside these settings. This interaction includes the history of the optimizer and parameters.

## Integrity and execution

Pin original evidence at git `14ad56af` using a committed input manifest. Verify source closure and every reused input against original sealed output manifests. Require paired original initial states/radii/streams, reconstructed batch hashes, and unchanged data hashes. Keep the original training-step and task-loop kernels identical in an explicit follow-up engine copy, checked by AST. Verify CPU synthetic replay against the original host and cap writes/invariants; on CUDA verify graph/eager equality for both activations and both arms. As a production gate, require task-29 no-cap fresh online accuracies to match the old fresh controls at their CSV precision for all 20 activation/seed pairs. A failed gate stops analysis and is retained; no result-dependent tolerance widening.

All new outputs go under `results/cap_act_box_2x2_1018/posthoc_fresh/`. Preserve initial/final states, radii, stream hashes, source/runtime provenance, checks, per-task/per-seed tables and paired intervals. CUDA uses the shared GPU flock, after the active CNN arm releases it; no concurrent training. Run 3 remains paused. Hash-verify a separate raw backup. No original output or provenance is rewritten.
