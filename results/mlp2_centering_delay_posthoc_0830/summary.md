# MLP2 5M centering-delay post-hoc reanalysis

> Existing `mlp2_phase1_0829` logs only; no new training or checkpoint replay.
> This is a post-hoc reanalysis registration, not a preregistration or independent replication.

## Main verdicts

- **P1 morphology:** `NO_EARLY_MORPHOLOGICAL_ADVANTAGE`; catch-up timing: `CATCHUP_BY_5M_SINGLE_BLOCK`.
- **P2 morphology + function:** `INCONCLUSIVE_FUNCTIONAL_KINETICS`.
- **P3 Aall low state:** `AALL_RELATIVE_ONLY`.
- **P4 localization:** `INCONCLUSIVE_LAYER_LOCALIZATION`.

## Registered components

| component | paired median gap [95% percentile CI] | interpretation |
|---|---:|---|
| A1−none, layer 2, B02 strict_dead_frac | 0.1221 [0.0624, 0.1799] | early protection if CI upper < 0 |
| A1−none, layer 2, B10 strict_dead_frac | 0.0284 [-0.0066, 0.05] | equivalent only if CI within ±0.05 |
| B10 gap−B02 gap | -0.101 [-0.1367, -0.0662] | closure if CI lower > 0 |
| A1−none, B10 log10(unfit) | -0.4718 [-0.9677, 0.3904] | negative means functional benefit |
| Aall−none, layer 2, B10 strict_dead_frac | -0.3573 [-0.4177, -0.2807] | P3 relative component |
| Aall−A1, layer 2, B10 strict_dead_frac | -0.3711 [-0.4493, -0.2816] | P3 relative component |
| Aall layer 2 B10 absolute strict_dead_frac | 0.5366 [0.4573, 0.6369] | absolute-low cutoff 0.25 |
| A1−none, layer 1, B10 strict_dead_frac | -0.4145 [-0.4769, -0.3534] | P4 negative control |

## Aall evaluation check (requested, REPORT_ONLY)

The registered P3 is morphological.  The following functional checks answer the requested
Aall evaluation question without changing P1–P4 after registration.

- B10 Aall−none log10(unfit): **-0.4824 [-0.9315, 0.3418]** — `INCONCLUSIVE_AALL_FUNCTION`.
- Aall B10−B02 log10(unfit): **0.3827 [0.3392, 0.4646]** — `AALL_FUNCTION_DETERIORATED_BY_5M`.

## Reconciliation with the original Phase 1 summary

The original report used `log10(mean unfit)`; this post-hoc spec registers
`mean(log10 unfit)`.  The former weights rare high-unfit tasks much more strongly.
It is reproduced here only to verify that the source data and transform order agree:

- Phase-1 transform, A1−none B10: **-0.9154 [-1.188, -0.3728]**.
- Phase-1 transform, Aall−none B10: **-1.191 [-1.446, -0.9856]**.
- These rows do not replace P2 or P3F and do not change any verdict.

## B10 levels (tasks 451–500)

`geometric unfit = 10^(mean log10 unfit)`; the table reports the median seed level.

| arm | layer-2 strict_dead_frac | mean log10(unfit) | geometric unfit |
|---|---:|---:|---:|
| L2_none | 0.8791 | -1.8389 | 0.014492 |
| L2_A1 | 0.9068 | -2.3003 | 0.005009 |
| L2_Aall | 0.5366 | -2.1899 | 0.0064585 |

## Limits

- `strict_dead` is a current-task support label, not permanent unit-ID death or an absorption time.
- Morphological catch-up does not imply functional catch-up; P1 and P2 remain separate.
- P3F is explicitly report-only because the registered P3 did not include an Aall functional verdict.
- Conclusions are limited to paired condA, depth 2, width 100, T=10,000, batch 1, lr=0.01, and 5M steps.
