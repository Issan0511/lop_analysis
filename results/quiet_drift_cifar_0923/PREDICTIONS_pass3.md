# quiet_drift_cifar_0923 pass 3 — does Adam's second-moment memory pace the quiet drift?

Claude (Opus 5.5), 2026-09-23.  Fixed after pass 1 (all seven bundles read) and before pass 2's
results and before any pass-3 replay beyond a 300-step switch test.

Pass 1, post hoc: in every bundle sqrt(v) decays at the β2 memory rate (τ 1,999–2,002 steps) and
the loss at τ 1,785–1,982 (switch bundles); the per-task quiet drift = (loss e-folds from the
post-fit level to the float32 floor, ≈ 11–12 in all switch bundles) × (‖W1‖² growth per e-fold:
I_C 216, A_own 14).  Reading: the loss is locked to Adam's memory rate and the step is whatever
achieves it, so the invariant is the growth per e-fold, not the growth per step.

## Test

Replay I_C and A_own exactly as pass 1 up to step 6,000 (every slot is past hit999 + 500 by
then), then switch Adam's β2 from 0.999 to 0.99 (memory 10× shorter) or 0.9999 (10× longer) for
the rest of the task.  Four bundles: I_C_b99, I_C_b9999, A_own_b99, A_own_b9999.  Baseline = pass
1's I_C and A_own over the same window.

Window per slot: step 6,000 → the first burst episode's first interval (pass 1's definition:
a run of probes with correct < 1199 whose minimum is < 1190) or 30,000.  With L = mean residual
(float64): e-folds = ln(L(6000) / L(end)); τ_L = window length / e-folds; growth per e-fold =
(‖W1‖²(end) − ‖W1‖²(6000)) / e-folds; drift per step = growth / window length.  Seed medians.

## Predictions (Claude)

| # | predicate | P |
|---|---|---|
| E1 | τ_L moves toward the lock in all four: τ_L(0.9999)/τ_L(0.999) > √10 and τ_L(0.99)/τ_L(0.999) < 1/√10, both bundles | 0.65 |
| E2 | growth per e-fold stays within a factor √10 of its β2 = 0.999 value in all four | 0.50 |
| E3 | the I_C / A_own ratio of growth per e-fold stays within a factor √10 of its β2 = 0.999 value, for both β2 | 0.60 |

√10 is the log midpoint between "no change" and the 10× the lock (E1) or a per-step pacing (E2)
would give.
