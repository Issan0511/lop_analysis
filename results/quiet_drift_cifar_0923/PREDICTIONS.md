# quiet_drift_cifar_0923 — predictions, fixed before the 30,000-step replays

Claude (Opus 5.5), 2026-09-23.  Issa asked: "IID の漂流が 10 倍大きい理由を調べて".  The
predictions below are written after two 300-step validation replays (which only checked that the
replay equals the main run's trace; no probe quantity was read) and before any full replay.

## Setup

`analysis/quiet_drift_cifar_0923/replay_probe.py` replays task 49 from `ckpts/t48.pt` of
altlabels_cifar_0923's LR_iid or LR_abab (R = 10 = seeds 0–9, std, 30,000 steps, the engine's
update op for op, deterministic mode).  Seven bundles:

| bundle | net | labels | what it is |
|---|---|---|---|
| I_next | iid | its own task 49 (fresh) | reproduces the main run |
| I_C | iid | fork C of t48 (fresh) | same C as A_C |
| I_rev | iid | task 47's labels (seen once, two tasks ago) | lag-2 revisit, like A |
| I_same | iid | task 48's labels again (no switch) | |
| A_own | abab | A (its own task 49, the 25th visit) | reproduces the main run |
| A_C | abab | fork C of t48 (fresh) | same C as I_C |
| A_same | abab | B again (no switch) | |

All bundles see the same minibatch order per seed (the batch streams of the two nets are identical
at t48).

## Definitions

- hit999 H: first probe (every 100 steps, step > 0) with ≥ 1199/1200 right.
- Rest phase: probes from H + 500 to 30,000.  Intervals (t−100, t].
- Burst episode: a maximal run of consecutive rest-phase probes with correct < 1199 whose minimum
  is < 1190.  Burst intervals: those ending at a probe of the episode, plus the one ending at the
  first probe after it.  Everything else in the rest phase is quiet.
- **Quiet drift Q**: the sum of Δ‖W1‖² over the quiet intervals.  Per seed; the bundle's value is
  the median over the 10 seeds.  (Knife-edge dips that never go below 1190 stay quiet.)
- Pre-burst quiet window: H + 500 → the first burst's first interval (or 30,000).
- In that window, with W_h = W1 at H + 500, W_0 = W1 at the task start, F = W_h − W_0 (the fit's
  displacement) and D = W1 − W_h: Δ‖W1‖² = 2⟨W_h, D⟩ + ‖D‖² = 2⟨W_0, D⟩ + 2⟨F, D⟩ + ‖D‖².
- Coherence over the fixed 1,000-step grid: κ = ‖Σu‖² / (1000 Σ‖u‖²), u = the W1 Adam update.
  Distance travelled in 1,000 steps = 1000 · √κ · rms‖u‖.
- K_eff = (Σ r_n)² / Σ r_n², r_n = 1 − p_y(n) in float64 from the float32 logits, all 1200 images.

## Verdicts

- **V1 (label or network).** Per seed λ = [Q(A_C) − Q(A_own)] / [Q(I_C) − Q(A_own)] (seed-paired;
  a seed with a non-positive denominator is excluded and reported).  Median λ ≥ 2/3 →
  `LABEL_DOMINANT`; ≤ 1/3 → `NETWORK_DOMINANT`; otherwise `MIXED`.  (Equal thirds of [0, 1].)
- **V2 (a once-seen label).** ρ_rev = Q(I_rev) / Q(I_C) per seed, median.  < 1/2 →
  `REVISIT_HALVES`; else `REVISIT_LESS_THAN_HALVES`.
- **V3 (no switch).** ρ_same_I = Q(I_same) / Q(I_C), ρ_same_A = Q(A_same) / Q(A_C).  Both < 1/2 →
  `NO_SWITCH_SMALL`; else `NO_SWITCH_NOT_SMALL`.
- **V4 (where the pre-burst drift points, I_C).** Median over seeds of 2⟨F, D⟩ versus 2⟨W_0, D⟩ at
  the end of the pre-burst window: ⟨F, D⟩ > ⟨W_0, D⟩ → `FIT_DIRECTION`, else `OLD_WEIGHTS`.
  The share ‖D‖² / Δ‖W1‖² is recorded.
- **V5 (coherence or step size, I_C vs A_own).** Over the 1,000-step windows inside each slot's
  pre-burst quiet window: the log ratio (I_C / A_own, seed medians) of √κ versus that of rms‖u‖.
  Larger |log ratio| of √κ → `COHERENCE`; of rms‖u‖ → `STEP_SIZE`.  The ratios of ‖W1‖ and of the
  cosine between Σu and W1 are recorded alongside.
- **V6 (what tracks the drift rate).** Over all 70 (bundle, seed) pre-burst windows: Spearman of the
  drift rate (Δ‖W1‖² per step) with the window's median log K_eff versus with its median log CE.
  |ρ_S(K_eff)| > |ρ_S(CE)| → `KEFF_BETTER`, else `CE_BETTER`.
- Check: Q(I_C) / Q(I_next) within [1/2, 2] (two fresh draws are alike); I_next and A_own
  reproduce the main traces of task 49 bit for bit at every probe.

## Predictions (Claude)

| # | predicate | P |
|---|---|---|
| P1 | V1 = LABEL_DOMINANT | 0.40 |
| P2 | V1 = MIXED | 0.45 |
| P3 | V1 = NETWORK_DOMINANT | 0.15 |
| P4 | V2 = REVISIT_HALVES | 0.35 |
| P5 | V3 = NO_SWITCH_SMALL | 0.70 |
| P6 | V4 = FIT_DIRECTION | 0.60 |
| P7 | V5 = COHERENCE | 0.55 |
| P8 | V6 = KEFF_BETTER | 0.60 |
| P9 | the check Q(I_C)/Q(I_next) in [1/2, 2] | 0.90 |

Reasoning in one line each: the gap grows with task number in the iid run (quiet 593 at t3 →
2,480 at t40) while every iid task is equally new, so the network state must matter (hence MIXED
as the mode); a fresh fit leaves many images at low margin, so the batch gradient keeps one
direction longer (coherence, K_eff); 0922 found the post-fit Adam amplifies the fit (fit direction).
