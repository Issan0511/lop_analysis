# capacity_regime_0830 — post-hoc validation

> Existing final checkpoints only; no new online learning run. Static ReLU fits are
> achievable upper bounds from multistart Adam, not certified global minima.

## Candidate 1 — native width-5 approximation floor

**REJECTED.** On the five exact final w5-centered tasks, the best achieved width-5
MSE has median 3.78195e-18 and maximum 0.00781251, versus observed median 0.718424.  A width-31 constructive interpolant has maximum
absolute error 0, hence width 100 has exact
zero approximation error on the 32-point support.

## Candidate 2 — absolute alive count / effective capacity

**PARTIALLY SUPPORTED.** Final alive counts are [0, 1, 2, 3, 1] (mean 1.4).  Dead-unit removal changes
no prediction on this support, so these counts are the exact active feature counts here.
However, the observed loss exceeds the best achieved network of the same alive width by
[0.0033874760952699, 0.0775293793026603, 0.3382508495735027, 0.2506726782798068, 0.0420430752236891]; count alone is insufficient, especially for
the 2- and 3-alive runs.  Feature placement/conditioning remains part of the failure.

## Candidate 3 — large residual changes the force-field phase

**RESIDUAL PREMISE CONFIRMED; PREDICTED SIGNATURE NOT SUPPORTED.**

| width | mean alive | mean MSE | median unfit | median cancellation | median gradient SNR |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.4 | 0.92165 | 0.573658 | 0.918067 | 0.00903104 |
| 100 | 70.6 | 0.0044999 | 0.00182904 | 0.837537 | 0.0452457 |

The w5-centered residual is large, but its self/rest cancellation is not weaker and its
gradient SNR is smaller than w100-centered.  Thus large residual alone does not imply a
drift-dominated, function-seeing phase in these final static snapshots.  A time-resolved
w5 logger would still be required to test the exact temporal rho used in the pillar.

## Overall

The supported *proximate description* is an **acquired effective-capacity collapse**,
not native width-5 inexpressivity: width 5 can fit the current task, but only 0–3 active
features remain and, when 2–3 survive, their geometry is materially worse than an
alive-matched static oracle.  This does not yet explain why centering fails to prevent
that collapse at width 5.  The mechanism therefore remains unresolved, and Candidate 3
should not be promoted without a dedicated temporal test.
