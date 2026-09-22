# quiet_drift_cifar_0923 pass 4 — does the full gradient turn over within the momentum's window?

Claude (Opus 5.5), 2026-09-23.  Fixed after passes 1–3 were read and before any pass-4 replay.

Post hoc so far: Adam's W1 step is aimed at the full-batch loss gradient G with cos 0.10–0.16
after a fresh fit and 0.36–0.39 in deep states; the whole loss of aim is in the momentum
(cos(G, m) = cos(G, u)); the minibatch noise-to-signal ratio (112–182, alike in all bundles)
predicts cos(G, m) ≈ 0.31–0.38 through β1 = 0.9 averaging, which matches the deep states and is 3×
too high for the fresh ones.  Candidate: after a fresh fit G itself turns over within the ~10
steps the momentum averages (the many shallow, near-tied images reorder), so m is stale.
Arithmetic: if cos(G_t, G_{t−k}) = e^{−k/κ}, the β1-weighted average Σ 0.1·0.9^k e^{−k/κ} must be
≈ 0.1/0.34 ≈ 0.3 to close the gap, i.e. κ ≈ 3 steps and cos at lag 10 ≈ 0.05.

## Test

Replay I_C, A_C (fresh) and A_own, I_same (deep) as in pass 2.  Every 1,000 steps from 6,000 to
22,000, compute the float64 full-batch W1 gradient G at 21 consecutive steps t0 … t0+20 (read only).
Per slot and burst: cos(G_{t0}, G_{t0+k}) for k = 1…20; per bundle the median over slots and bursts
(bursts inside the slot's pre-burst quiet window only).

## Predictions (Claude)

| # | predicate | P |
|---|---|---|
| F1 | fresh (I_C and A_C): median cos(G_t, G_{t+10}) < 0.5; deep (A_own and I_same): > 0.8 | 0.45 |
| F2 | the fresh bundles' lag-10 cosine is below the deep bundles' in every one of the four pairings | 0.75 |

Threshold 0.5: midpoint between the staleness requirement (≈ 0.05) and no turnover (≈ 1); 0.8 for
the deep side, where no deficit needs explaining.
