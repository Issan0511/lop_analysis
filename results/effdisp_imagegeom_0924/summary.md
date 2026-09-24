# CIFAR image geometry and effective displacement

Analysis status: **TERMINAL** (50 complete, 0 DIVERGED of 50 registered seed series).

DIVERGED seeds remain nonpasses for 50-task P1–P4. A fully observed finite early or late registered window before failure remains usable for its window comparison; missing windows are invalid. No shortened horizon is treated as 50 tasks.

## Terminal failures

(no rows)

First-layer V/Q/X use float64 centered projections of the saved 1200-image bank; no ambient covariance matrix is formed.
P1–P4 use the earlier fixed-covariance implementation and thresholds. Identity residuals audit arithmetic and are not evidence for the hypothesis.
STOPPED and LOW_RESPONSE remain distinct from an active plateau. P3 uses tasks 6–25 for calibration and 26–50 for held-out prediction.
dup64 and dup64_half preserve information; gray32 and avg16 also change information and task difficulty.
Raw01 inputs are not the earlier standardized CIFAR condition. All geometries use the same global learning rates, including the first layer.

## Registered P1–P4

| geometry | optimizer | question | verdict | n_pass | seed_median | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avg16 | adam | P1 | PASS | 5 | 0.965 | 0.942 | 0.99 |
| avg16 | adam | P2 | FAIL | 1 | 0.463 | 0.144 | 0.498 |
| avg16 | adam | P3 | FAIL | 0 | 0.219 | 0.204 | 0.224 |
| avg16 | adam | P4 | NOT_ESTABLISHED | 1 | 0.382 | 0.315 | 0.642 |
| avg16 | sgd | P1 | PASS | 5 | 0.964 | 0.924 | 1.02 |
| avg16 | sgd | P2 | PASS | 4 | 0.114 | -0.0907 | 0.241 |
| avg16 | sgd | P3 | FAIL | 0 | 0.101 | 0.0659 | 0.13 |
| avg16 | sgd | P4 | NOT_ESTABLISHED | 3 | 0.217 | 0.0918 | 0.494 |
| dup64 | adam | P1 | PASS | 5 | 0.968 | 0.964 | 0.981 |
| dup64 | adam | P2 | FAIL | 0 | 0.335 | 0.273 | 0.547 |
| dup64 | adam | P3 | FAIL | 0 | 0.225 | 0.221 | 0.247 |
| dup64 | adam | P4 | NOT_ESTABLISHED | 0 | 0.457 | 0.244 | 0.499 |
| dup64 | sgd | P1 | PASS | 5 | 0.968 | 0.953 | 0.995 |
| dup64 | sgd | P2 | FAIL | 1 | 0.235 | 0.163 | 0.438 |
| dup64 | sgd | P3 | FAIL | 0 | 0.117 | 0.0815 | 0.125 |
| dup64 | sgd | P4 | NOT_ESTABLISHED | 1 | 0.416 | 0.275 | 0.54 |
| dup64_half | adam | P1 | PASS | 5 | 0.96 | 0.937 | 0.985 |
| dup64_half | adam | P2 | FAIL | 0 | 0.435 | 0.28 | 0.702 |
| dup64_half | adam | P3 | FAIL | 0 | 0.23 | 0.222 | 0.249 |
| dup64_half | adam | P4 | NOT_ESTABLISHED | 0 | 0.401 | 0.108 | 0.542 |
| dup64_half | sgd | P1 | PASS | 5 | 0.952 | 0.942 | 1.04 |
| dup64_half | sgd | P2 | FAIL | 3 | 0.195 | -0.177 | 0.406 |
| dup64_half | sgd | P3 | FAIL | 2 | 0.0361 | 0.0262 | 0.066 |
| dup64_half | sgd | P4 | NOT_ESTABLISHED | 3 | 0.382 | 0.165 | 0.75 |
| gray32 | adam | P1 | PASS | 5 | 0.967 | 0.96 | 0.975 |
| gray32 | adam | P2 | FAIL | 0 | 0.49 | 0.205 | 0.623 |
| gray32 | adam | P3 | FAIL | 0 | 0.188 | 0.178 | 0.193 |
| gray32 | adam | P4 | NOT_ESTABLISHED | 0 | 0.354 | 0.237 | 0.671 |
| gray32 | sgd | P1 | PASS | 5 | 1.01 | 0.952 | 1.05 |
| gray32 | sgd | P2 | FAIL | 3 | 0.0212 | -0.363 | 0.299 |
| gray32 | sgd | P3 | FAIL | 0 | 0.0745 | 0.0301 | 0.0807 |
| gray32 | sgd | P4 | NOT_ESTABLISHED | 3 | 0.373 | 0.0718 | 0.741 |
| rgb32 | adam | P1 | PASS | 5 | 0.961 | 0.952 | 0.975 |
| rgb32 | adam | P2 | FAIL | 0 | 0.471 | 0.285 | 0.618 |
| rgb32 | adam | P3 | FAIL | 0 | 0.231 | 0.206 | 0.235 |
| rgb32 | adam | P4 | NOT_ESTABLISHED | 0 | 0.403 | 0.255 | 0.567 |
| rgb32 | sgd | P1 | PASS | 5 | 0.966 | 0.956 | 1.01 |
| rgb32 | sgd | P2 | PASS | 4 | 0.086 | -0.189 | 0.322 |
| rgb32 | sgd | P3 | UNINFORMATIVE | 0 | 0.0456 | 0.0163 | 0.0537 |
| rgb32 | sgd | P4 | SEPARATED | 4 | 0.502 | 0.295 | 0.783 |

## Registered E1–E5

| question | verdict | n_pass | criterion |
| --- | --- | --- | --- |
| E1 | PASS | 5 | late halfdup/SGD R and d_eff ratios both in [0.8,1.2] |
| E2 | PASS | 5 | early halfdup/Adam d_eff ratio >1.2 |
| E3 | PASS | 5 | early rawdup d_eff ratio >1, independently in 4/5 seeds per optimizer |
| E4 | REPORT_ONLY | nan | gray32/avg16 direction unspecified; rank and performance reported |
| E5 | FAIL | 0 | all four late c geometry comparisons pass in each optimizer |

## Output tables

`transitions.csv` has observed endpoint tasks and source snapshot paths. `seed_summary.csv` has all 50 seeds, including explicit DIVERGED rows; `closure.csv` contains only complete held-out predictions. `windows.csv` marks each registered window's status and availability. `contrasts.csv` retains invalid pairs as missing with NaN rather than dropping their seeds; `paired_groups.csv` and `groups.csv` bootstrap valid seed-level values (5000 draws, seed 924). `covstats.csv` and `spectrum.csv` report rank, participation ratio, and covariance spectrum. `source_manifest.csv` hashes all raw inputs.

If P2 fails, reported R is an observed finite-window width rather than an established fixed-point height.
