# quiet_drift_cifar_0923 pass 2 — predictions, fixed before the diagnostic replays

Claude (Opus 5.5), 2026-09-23.  Written after pass 1's first three bundles (I_next, I_C, A_own)
were read and BEFORE the other four (I_rev, I_same, A_C, A_same) were read.  Pass 1 found,
post hoc: the quiet-drift gap (I_C / A_own ≈ 17x per step) is carried by the Adam step size
(W1 rms step 8x) and its outward alignment (cos with W1 3.4x), not by coherence (√κ 0.5x);
in both nets sqrt(v) decays at exactly the β2 memory rate (τ = 2,000 steps) and the loss at
τ ≈ 1,800, with the active images' margins rising at the same ~0.54 per 1,000 steps.

## Definitions (per slot; medians over the probes from hit999 + 2,500 to the last probe before
the first burst episode of pass 1's definition)

- c_l = <G_l, u_l> / L: layer l's first-order share of the log-loss drop per step
  (G_l = full-batch gradient of the mean CE in float64, u_l = the Adam update, L = mean CE).
- c_tot = Σ_l c_l over W1, b1, W2, b2, W3, b3.  share_W1 = c_W1 / c_tot.
- ||u_W1|| = c_W1 / ((||G_W1|| / L) · cos(G_W1, u_W1)), and ||G_W1|| / L = C_img · (Σ_n ||g_n|| / L)
  with C_img = ||Σ_n g_n|| / Σ_n ||g_n|| (g_n = image n's W1 gradient).
- So log(||u_W1|| ratio) = log(c_tot ratio) + log(share_W1 ratio) − log(C_img ratio)
  − log((Σ||g_n||/L) ratio) − log(cos ratio), ratios I_C over A_own of the bundle medians.

## Predictions (Claude)

| # | predicate | P |
|---|---|---|
| D1 | C_img(I_C) < C_img(A_own) | 0.70 |
| D2 | of the four terms (share_W1, C_img, Σ||g_n||/L, cos), the C_img term has the largest magnitude | 0.35 |
| D3 | c_tot is within (1/4000, 1/1000) per step in every bundle whose window has ≥ 20 probes | 0.70 |
| D4 | A_C's W1 step (pass 1 rms, same window) is closer in log to I_C's than to A_own's | 0.60 |
