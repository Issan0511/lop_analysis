# mlp2_phase0_0829 summary

## G0

**LOP_ABSENT** — dU=0.00608679, 95% bootstrap-t CI [-0.00149124, 91.9444], late unfit=0.00608679 (threshold=0.05).

Depths are not treated as paired; G0 is an L2 within-run time comparison.

## Final task-end wall coordinate

| arm | layer | median seed median(M) | seed IQR | max NA units |
|---|---:|---:|---:|---:|
| L1 | 1 | -2.49723 | [-2.75774, -2.25234] | 0 |
| L2 | 1 | -2.12889 | [-2.31821, -1.64674] | 0 |
| L2 | 2 | -1.30727 | [-1.51074, -1.00863] | 0 |

## Task-end trend

Each seed contributes Spearman(task, median M); the reported point is the median over seeds with a studentized bootstrap interval.

| arm | layer | median rho | 95% bootstrap-t CI | increase |
|---|---:|---:|---:|---:|
| L1 | 1 | -0.548927 | [-3.64216e+13, 6.00328e+12] | NO |
| L2 | 1 | -0.648447 | [-6.49331e+11, 3.73889e+11] | NO |
| L2 | 2 | -0.245449 | [-3.28228e+13, 1.35559e+14] | NO |

## Sanity

- S0 legacy identity: **PASS**
- S1/S2 exact identities: **PASS**
- S3 OMP threads: **PASS**

All layer statistics use the exact 32-pattern support. M and B are static wall-condition coordinates, not normalized dynamical variables.
