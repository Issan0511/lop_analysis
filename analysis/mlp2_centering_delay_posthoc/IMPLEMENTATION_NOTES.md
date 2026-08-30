# Implementation notes

The registered specification remains frozen at
`specs/spec_mlp2_centering_delay_posthoc_0830.md`.  The following choices fill
implementation details that the specification did not completely enumerate;
they do not change P1--P3.

## P4 unenumerated states

The specification defines retained layer-1 protection as an A1-minus-none B10
CI upper bound below -0.10, but names only three narrative outcomes.  The code
uses the following exhaustive mapping:

- layer-1 CI lower bound above +0.05: `A1_LAYER1_PARADOX`;
- P1 catch-up plus retained layer-1 protection: `LAYER2_LOCALIZED_CATCHUP`;
- P1 catch-up plus layer-1 equivalence within +/-0.05:
  `GLOBAL_CATCHUP`;
- every other state: `INCONCLUSIVE_LAYER_LOCALIZATION`.

P4 never overrides P1 or P2.

## Requested Aall function check

Registered P3 is morphological and therefore cannot by itself answer whether
`L2_Aall` retains evaluation performance.  Two explicitly report-only rows are
added under `P3F_REPORT_ONLY`:

1. B10 paired Aall-minus-none mean-log10-unfit gap;
2. paired within-Aall B10-minus-B02 mean-log10-unfit change.

They use the registered functional margin (+/-0.1 log10 unit), shared paired
seed bootstrap draws, and do not alter P1--P4.

The output also contains `RECONCILIATION_REPORT_ONLY` rows using
`log10(mean unfit)`, the transform order from the original Phase 1 report.
Those rows verify why that report and the new registered `mean(log10 unfit)`
endpoint can differ; they never enter a post-hoc verdict.

## Floating-point boundary

Equivalence comparisons use a `1e-12` numerical tolerance.  This treats a CSV
result such as `0.050000000000000044` as the registered exact boundary 0.05.
The tolerance is many orders of magnitude below one width-100 unit (0.01 in
`strict_dead_frac`) and is used only for decision-boundary comparison.
