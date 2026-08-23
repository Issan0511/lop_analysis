# mu_titration_0823 analysis

Overall preregistered verdict: **FULL_PASS**.

This summary uses measured `mu_norm` as dose. `center_alpha` is an EMA update rate, not a partial-subtraction fraction.

## Sanity and provenance

- Source S1--S5 and S7 passed for all eight arms; a failed arm is never silently dropped.
- S6a passed: step-0 reproducibility hashes, logged step-0 statistics, and complete flip trajectories agree across arms.
- S6b passed: alpha=0 and alpha=.01 common columns are bit-equal to the preregistered endpoint references.
- Specification: `specs/spec_mu_titration_0823.md`; post-hoc S3 addenda (in governing order): `specs/spec_mu_titration_0823_addendum.md`, `specs/spec_mu_titration_0823_addendum2.md`; config: `configs/mu_titration_0823.yaml`.
- All arms carry the same ordered addenda path/SHA list, and the sweep commit is a Git ancestor of the clean analysis commit.
- All bootstrap estimates use one shared set of seed-bundle weights (B=10,000, RNG seed 20260823).

## Verdicts

| ID | status | estimate | 95% CI | detail |
|---|---|---:|---:|---|
| C0 | **PASS** | 41.35 | NA | dose_ratio=41.3495; distinct_dose=8; finite_theta=6 |
| C1 | **PASS** | -0.2764 | [-0.3515, -0.2359] | median seed Spearman=0.974679; order violations=0/15 (0); nonmonotonic=False |
| W1 | **PASS** | -0.8613 | [-0.8729, -0.8498] | low-dose NA arms=2; exact-censored=True; q slope=0.138742; denominator=-1 |
| C2 | **PASS** | 0.8107 | [0.7644, 0.8556] | median seed Spearman=1 |
| P1 | **REPORT_ONLY** | 0.229 | [0.2074, 0.2511] | final strict_dead vs measured bulk mu_norm; does not override C1/C2 |
| OVERALL | **FULL_PASS** | NA | NA | C0=PASS; C1=PASS; W1=PASS; C2=PASS |

## Bulk dose response

| alpha | mu_norm | theta_med | wall log | bias share | strict_dead |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.041 | -0.15 | -1.634 | 0.1014 | 0.926 |
| 1e-06 | 2.333 | -0.2 | -1.571 | 0.1336 | 0.556 |
| 3e-06 | 1.878 | -0.2 | -1.636 | 0.1614 | 0.316 |
| 1e-05 | 1.521 | -0.25 | -1.519 | 0.2188 | 0.302 |
| 3e-05 | 1.112 | -0.35 | -1.193 | 0.3127 | 0.19 |
| 0.0001 | 0.6265 | -0.5 | -0.7361 | 0.4887 | 0.105 |
| 0.0003 | 0.2237 | NA | 0.1287 | 0.6518 | 0.134 |
| 0.01 | 0.07355 | NA | 1.513 | 0.883 | 0.245 |

## Scope and interpretation

- Primary scope is `bulk`: 1,000-step grid points more than 100 steps from scheduled boundaries (901 points).
- Secondary outputs cover realized-boundary offsets -100..+100, all-recorded, bulk time halves, and fixed phase offset +5000.
- `theta_all`, final `near_off`, and final `dead_0.05` are secondary/reporting quantities. Unqualified `dead` is not used.
- A missing theta is not called absence of a wall; `left_censored_exact` records domain-left-censoring when median exact `cos_crit < -1`.
- Wall and bias-field paths are intermediate mechanisms, not independent causal mediation proportions for `strict_dead`.
- No inference is extended to condB or other widths, periods, batches, or learning rates.

## Files

`arm_manifest.csv`, `raw_sha256.csv`, `gate_curve.csv`, `theta_estimates.csv`, `dose_response.csv`, `path_decomposition.csv`, `per_seed_metrics.csv`, `verdict.csv`, `analysis_meta.json`, `determinism_check.md`, and `figures/`.
