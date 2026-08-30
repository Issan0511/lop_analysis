# Switch shock and initial random-feature coverage (post-hoc; no training)

## Verdict

- **Candidate 6: signature supported, mechanism not identified.** Positive `dead_frac` changes are concentrated in the first observed 1,000 updates after a task switch, and the concentration fold is larger at w5. The metric is a task-support label without unit identity, so this does not prove that physical trajectories crossed an absorbing boundary during that bin.
- **Candidate 7: redundancy supported, nearest-direction story not isolated.** With initial `W,b` frozen, the literal step-0 w100 feature matrix spans all 32 support points and a fitted `v` interpolates every seed. w5 does not. Under ideal per-task centering across all 100 tasks, w100 still has much lower readout error; random five-feature subsets of w100 fall back near w5.

## Candidate 6 — switch-aligned label increases

The existing probe grid is 1,000 steps. Rows at 10,000 multiples are pre-switch; the row at offset 1,000 is after the first 1,000 updates on the new task. Task 1 is excluded, leaving 99 post-switch and 891 within-task comparison intervals per seed.

| width | positive change / post bin | positive change / interior bin | concentration fold [seed-cluster 95% CI] | positive-event rate post / interior |
|---|---:|---:|---:|---:|
| w5 | 0.016970 | 0.003277 | 5.18 [4.42, 5.96] | 0.079 / 0.016 |
| w100 | 0.009758 | 0.004321 | 2.26 [2, 2.63] | 0.453 / 0.310 |

The concentration-fold ratio w5/w100 is 2.29 [1.84, 2.79]. Every seed has post/interior fold >1. This supports the proposed timing signature and its stronger relative concentration at w5.

**Limit:** `dead_frac` can fall as well as rise and can change when the task support changes. Positive aggregate changes hide simultaneous unit-level entries/exits. The current files therefore cannot distinguish boundary reclassification from irreversible unit motion, nor localize an event within the 1,000-step bin.

## Candidate 7 — fixed initial features, least-squares `v`

### Literal centered-arm state at step 0

At step 0 the centered arm has `running_mean=0`, so learner input is raw input. The 32 support points are enumerated exactly; `W,b` are the saved initial tensors and only `v` is fitted (no intercept in the primary result).

| width | median NMSE | ranks by seed | result |
|---|---:|---|---|
| w5 | 0.926262 | [3, 5, 4, 5, 3] | substantial residual |
| w100 | 3.77e-29 | [32, 32, 32, 32, 32] | all five seeds interpolate (<1e-20 NMSE) |

This is direct evidence that the initial w100 bank already contains a readout span sufficient for the first task, whereas w5 must change hidden features to reach a comparable fit.

### All 100 tasks under ideal task-wise centering

This sensitivity removes each task's exact input mean before applying the same frozen initial `W,b`. It approximates the steady centered geometry without replaying learning.

| feature bank | median NMSE | IQR | median rank |
|---|---:|---:|---:|
| w5 all 5 | 0.988832 | 0.789626–1.321118 | 5 |
| w100 all 100 | 0.186476 | 0.121758–0.276838 | 21 |
| w100 random 5 (median of 100/case) | 0.865294 | — | ≤5 |

w5/w100-all error fold: 5.3 [4.68, 6.46]. Random-5/w100-all fold: 4.64 [4.33, 5.3]. w5/random-5 fold: 1.14 [0.947, 1.3].

The subset control shows that most of the advantage comes from having a redundant 100-feature span, not from a different per-feature initialization law. It does **not** specifically establish that one initial direction is geometrically close to a unique required direction; the least-squares test identifies span/coverage, which also includes ordinary dimensionality.

## Reproducibility and scope

- No SGD step, optimizer update, or new training run was executed.
- Step-0 `W,b,flip_state` reconstruction and the 1M final `flip_state` replay match exactly for both widths.
- Width groups use different width-dependent generators in the original experiment, so w5 versus w100 is an unpaired five-seed comparison. Bootstrap resamples seed trajectories and retains all 100 tasks within a selected seed.
- All claims are post-hoc and limited to `center_selfcov_0814`, condA, centered input, teacher width 100, T=10,000.
