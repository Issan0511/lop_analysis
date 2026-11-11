# P residual factorial summary

U is the trapezoidal time-average of 1−accuracy over [tau, limit]. Every trajectory reaches the fixed horizon; no early stopping or horizon substitution for T90.

Positive KC−ZC indicates a delay from keeping head parameters after head Adam is cleared. The KK−ZK contrast retains head Adam. Body Adam is fixed across the four interventions at the branch point.

Recorded sustained T90 is the first saved evaluation after which every remaining saved evaluation is ≥90%. An isolated final crossing qualifies with zero observed duration. This does not guarantee continuous or future persistence.

MLP accuracy is stored by canonical float32 evaluation; the 90% threshold uses float32(0.9), so an exact 1080/1200 score counts as reaching. ViT uses 0.9. Endpoint accuracy contrasts have the opposite desirability direction to U/error/CE; error_end is also provided.

Intervals are deterministic paired-seed bootstrap percentile 95% intervals, 10,000 resamples, without multiplicity adjustment. Aliased records do not increase the seed count.

Logical rows: 820. Direct input trajectories: 690. Main complete: True.

## U contrasts after P at tau=0

| model | activation | t_end | history | source | tau | contrast | estimate | ci_low | ci_high | n_seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | ELU | 1 | S | P | 0 | parameter_clear | -0.21925 | -0.22243 | -0.21560 | 10 |
| mlp | ELU | 1 | S | P | 0 | parameter_keep | -0.17150 | -0.17531 | -0.16744 | 10 |
| mlp | ELU | 1 | S | P | 0 | memory_keep | -0.00064 | -0.00112 | -0.00020 | 10 |
| mlp | ELU | 1 | S | P | 0 | memory_zero | -0.04840 | -0.04953 | -0.04717 | 10 |
| mlp | ELU | 1 | S | P | 0 | joint | -0.21989 | -0.22293 | -0.21645 | 10 |
| mlp | ELU | 1 | S | P | 0 | interaction | 0.04775 | 0.04668 | 0.04884 | 10 |
| mlp | ELU | 10 | S | P | 0 | parameter_clear | 0.21866 | 0.20676 | 0.23125 | 10 |
| mlp | ELU | 10 | S | P | 0 | parameter_keep | 0.23061 | 0.21849 | 0.24354 | 10 |
| mlp | ELU | 10 | S | P | 0 | memory_keep | -0.00482 | -0.00528 | -0.00435 | 10 |
| mlp | ELU | 10 | S | P | 0 | memory_zero | -0.01677 | -0.01784 | -0.01568 | 10 |
| mlp | ELU | 10 | S | P | 0 | joint | 0.21385 | 0.20217 | 0.22614 | 10 |
| mlp | ELU | 10 | S | P | 0 | interaction | 0.01195 | 0.01099 | 0.01284 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | parameter_clear | 0.19765 | 0.18441 | 0.20964 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | parameter_keep | 0.19813 | 0.18473 | 0.21034 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | memory_keep | -0.00157 | -0.00187 | -0.00124 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | memory_zero | -0.00206 | -0.00287 | -0.00118 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | joint | 0.19608 | 0.18292 | 0.20798 | 10 |
| mlp | ELU | 29 | Pall | P | 0 | interaction | 0.00049 | -0.00038 | 0.00130 | 10 |
| mlp | R | 1 | S | P | 0 | parameter_clear | -0.16177 | -0.16939 | -0.15425 | 10 |
| mlp | R | 1 | S | P | 0 | parameter_keep | -0.10411 | -0.11175 | -0.09633 | 10 |
| mlp | R | 1 | S | P | 0 | memory_keep | -0.00336 | -0.00441 | -0.00234 | 10 |
| mlp | R | 1 | S | P | 0 | memory_zero | -0.06103 | -0.06232 | -0.05976 | 10 |
| mlp | R | 1 | S | P | 0 | joint | -0.16513 | -0.17251 | -0.15781 | 10 |
| mlp | R | 1 | S | P | 0 | interaction | 0.05766 | 0.05646 | 0.05906 | 10 |
| mlp | R | 10 | S | P | 0 | parameter_clear | 0.02797 | 0.01742 | 0.04123 | 10 |
| mlp | R | 10 | S | P | 0 | parameter_keep | 0.02816 | 0.01692 | 0.04200 | 10 |
| mlp | R | 10 | S | P | 0 | memory_keep | -0.00059 | -0.00123 | -0.00020 | 10 |
| mlp | R | 10 | S | P | 0 | memory_zero | -0.00077 | -0.00291 | 0.00133 | 10 |
| mlp | R | 10 | S | P | 0 | joint | 0.02739 | 0.01720 | 0.04034 | 10 |
| mlp | R | 10 | S | P | 0 | interaction | 0.00018 | -0.00170 | 0.00200 | 10 |
| mlp | SILU | 1 | S | P | 0 | parameter_clear | -0.20431 | -0.20933 | -0.19961 | 10 |
| mlp | SILU | 1 | S | P | 0 | parameter_keep | -0.14702 | -0.15220 | -0.14212 | 10 |
| mlp | SILU | 1 | S | P | 0 | memory_keep | -0.00076 | -0.00125 | -0.00029 | 10 |
| mlp | SILU | 1 | S | P | 0 | memory_zero | -0.05805 | -0.05903 | -0.05717 | 10 |
| mlp | SILU | 1 | S | P | 0 | joint | -0.20507 | -0.20987 | -0.20039 | 10 |
| mlp | SILU | 1 | S | P | 0 | interaction | 0.05729 | 0.05660 | 0.05803 | 10 |
| mlp | SILU | 10 | S | P | 0 | parameter_clear | 0.03479 | 0.02279 | 0.04618 | 10 |
| mlp | SILU | 10 | S | P | 0 | parameter_keep | 0.05582 | 0.04457 | 0.06781 | 10 |
| mlp | SILU | 10 | S | P | 0 | memory_keep | -0.00304 | -0.00349 | -0.00248 | 10 |
| mlp | SILU | 10 | S | P | 0 | memory_zero | -0.02406 | -0.02830 | -0.01983 | 10 |
| mlp | SILU | 10 | S | P | 0 | joint | 0.03176 | 0.01961 | 0.04336 | 10 |
| mlp | SILU | 10 | S | P | 0 | interaction | 0.02102 | 0.01660 | 0.02564 | 10 |
| vit | SL | 100 | S | P | 0 | parameter_clear | 0.02289 | -0.01752 | 0.06792 | 5 |
| vit | SL | 100 | S | P | 0 | parameter_keep | 0.06517 | 0.01962 | 0.11073 | 5 |
| vit | SL | 100 | S | P | 0 | memory_keep | 0.01133 | -0.00198 | 0.02464 | 5 |
| vit | SL | 100 | S | P | 0 | memory_zero | -0.03095 | -0.03979 | -0.02181 | 5 |
| vit | SL | 100 | S | P | 0 | joint | 0.03422 | -0.00472 | 0.07316 | 5 |
| vit | SL | 100 | S | P | 0 | interaction | 0.04228 | 0.02267 | 0.06190 | 5 |

## Primary registered verdict

```json
{
  "main_design_complete": true,
  "primary_metric": "U",
  "primary_contrast": "KC-ZC",
  "operational_commonality_verdict": "unresolved",
  "strata": [
    {
      "model": "mlp",
      "activation": "ELU",
      "t_end": 10,
      "history": "S",
      "source": "P",
      "tau": 0,
      "estimate": 0.21866452047927307,
      "ci_low": 0.2067633189329575,
      "ci_high": 0.23125197159524308,
      "supported": true,
      "practical": true
    },
    {
      "model": "vit",
      "activation": "SL",
      "t_end": 100,
      "history": "S",
      "source": "P",
      "tau": 0,
      "estimate": 0.022890666666666698,
      "ci_low": -0.017519786666666613,
      "ci_high": 0.06791690666666671,
      "supported": false,
      "practical": false
    }
  ],
  "both_strata_support_operational_head_dependent_delay": false,
  "both_strata_practical_above_0_01": false,
  "interpretation": "Operational agreement only; not identity of microscopic mechanism.",
  "bootstrap": {
    "replicates": 10000,
    "interval": "percentile 95%",
    "unit": "paired seed",
    "multiplicity_adjusted": false
  }
}
```

## First/sustained T90 censoring

| model | activation | t_end | history | source | tau | arm | n_seed | first_unreached | sustained_unreached | first_elapsed_mean_all | sustained_elapsed_mean_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | ELU | 1 | S | P | 0 | KC | 10 | 0 | 0 | 1550.00000 | 1550.00000 |
| mlp | ELU | 1 | S | P | 0 | KK | 10 | 0 | 0 | 1530.00000 | 1530.00000 |
| mlp | ELU | 1 | S | P | 0 | ZC | 10 | 0 | 0 | 3140.00000 | 3190.00000 |
| mlp | ELU | 1 | S | P | 0 | ZK | 10 | 0 | 0 | 2860.00000 | 2860.00000 |
| mlp | ELU | 1 | S | P | 25 | KC | 10 | 0 | 0 | 1525.00000 | 1525.00000 |
| mlp | ELU | 1 | S | P | 25 | KK | 10 | 0 | 0 | 1505.00000 | 1505.00000 |
| mlp | ELU | 1 | S | P | 25 | ZC | 10 | 0 | 0 | 3115.00000 | 3135.00000 |
| mlp | ELU | 1 | S | P | 25 | ZK | 10 | 0 | 0 | 2895.00000 | 2895.00000 |
| mlp | ELU | 1 | S | S | 0 | KC | 10 | 0 | 0 | 2520.00000 | 2520.00000 |
| mlp | ELU | 1 | S | S | 0 | KK | 10 | 0 | 0 | 2490.00000 | 2510.00000 |
| mlp | ELU | 1 | S | S | 0 | ZC | 10 | 0 | 0 | 3140.00000 | 3190.00000 |
| mlp | ELU | 1 | S | S | 0 | ZK | 10 | 0 | 0 | 2840.00000 | 2860.00000 |
| mlp | ELU | 10 | S | P | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | P | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | P | 0 | ZC | 10 | 0 | 0 | 3220.00000 | 3220.00000 |
| mlp | ELU | 10 | S | P | 0 | ZK | 10 | 0 | 0 | 3160.00000 | 3160.00000 |
| mlp | ELU | 10 | S | P | 25 | KC | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | P | 25 | KK | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | P | 25 | ZC | 10 | 5 | 5 | — | — |
| mlp | ELU | 10 | S | P | 25 | ZK | 10 | 4 | 4 | — | — |
| mlp | ELU | 10 | S | S | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | S | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | ELU | 10 | S | S | 0 | ZC | 10 | 0 | 0 | 3220.00000 | 3220.00000 |
| mlp | ELU | 10 | S | S | 0 | ZK | 10 | 0 | 0 | 3160.00000 | 3160.00000 |
| mlp | ELU | 29 | Pall | P | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | ELU | 29 | Pall | P | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | ELU | 29 | Pall | P | 0 | ZC | 10 | 10 | 10 | — | — |
| mlp | ELU | 29 | Pall | P | 0 | ZK | 10 | 10 | 10 | — | — |
| mlp | R | 1 | S | P | 0 | KC | 10 | 0 | 0 | 1680.00000 | 1680.00000 |
| mlp | R | 1 | S | P | 0 | KK | 10 | 0 | 0 | 1630.00000 | 1630.00000 |
| mlp | R | 1 | S | P | 0 | ZC | 10 | 0 | 0 | 2800.00000 | 2800.00000 |
| mlp | R | 1 | S | P | 0 | ZK | 10 | 0 | 0 | 2400.00000 | 2400.00000 |
| mlp | R | 1 | S | P | 25 | KC | 10 | 0 | 0 | 1625.00000 | 1625.00000 |
| mlp | R | 1 | S | P | 25 | KK | 10 | 0 | 0 | 1605.00000 | 1605.00000 |
| mlp | R | 1 | S | P | 25 | ZC | 10 | 0 | 0 | 2645.00000 | 2645.00000 |
| mlp | R | 1 | S | P | 25 | ZK | 10 | 0 | 0 | 2495.00000 | 2495.00000 |
| mlp | R | 1 | S | S | 0 | KC | 10 | 0 | 0 | 2960.00000 | 2960.00000 |
| mlp | R | 1 | S | S | 0 | KK | 10 | 0 | 0 | 2850.00000 | 2850.00000 |
| mlp | R | 1 | S | S | 0 | ZC | 10 | 0 | 0 | 2800.00000 | 2800.00000 |
| mlp | R | 1 | S | S | 0 | ZK | 10 | 0 | 0 | 2390.00000 | 2390.00000 |
| mlp | R | 10 | S | P | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 0 | ZC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 0 | ZK | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 25 | KC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 25 | KK | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 25 | ZC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | P | 25 | ZK | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | S | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | S | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | S | 0 | ZC | 10 | 10 | 10 | — | — |
| mlp | R | 10 | S | S | 0 | ZK | 10 | 10 | 10 | — | — |
| mlp | SILU | 1 | S | P | 0 | KC | 10 | 0 | 0 | 1380.00000 | 1400.00000 |
| mlp | SILU | 1 | S | P | 0 | KK | 10 | 0 | 0 | 1370.00000 | 1370.00000 |
| mlp | SILU | 1 | S | P | 0 | ZC | 10 | 0 | 0 | 2850.00000 | 2900.00000 |
| mlp | SILU | 1 | S | P | 0 | ZK | 10 | 0 | 0 | 2480.00000 | 2500.00000 |
| mlp | SILU | 1 | S | P | 25 | KC | 10 | 0 | 0 | 1345.00000 | 1345.00000 |
| mlp | SILU | 1 | S | P | 25 | KK | 10 | 0 | 0 | 1345.00000 | 1345.00000 |
| mlp | SILU | 1 | S | P | 25 | ZC | 10 | 0 | 0 | 2655.00000 | 2655.00000 |
| mlp | SILU | 1 | S | P | 25 | ZK | 10 | 0 | 0 | 2515.00000 | 2515.00000 |
| mlp | SILU | 1 | S | S | 0 | KC | 10 | 0 | 0 | 2000.00000 | 2000.00000 |
| mlp | SILU | 1 | S | S | 0 | KK | 10 | 0 | 0 | 2020.00000 | 2020.00000 |
| mlp | SILU | 1 | S | S | 0 | ZC | 10 | 0 | 0 | 2850.00000 | 2900.00000 |
| mlp | SILU | 1 | S | S | 0 | ZK | 10 | 0 | 0 | 2490.00000 | 2510.00000 |
| mlp | SILU | 10 | S | P | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 0 | ZC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 0 | ZK | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 25 | KC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 25 | KK | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 25 | ZC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | P | 25 | ZK | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | S | 0 | KC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | S | 0 | KK | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | S | 0 | ZC | 10 | 10 | 10 | — | — |
| mlp | SILU | 10 | S | S | 0 | ZK | 10 | 10 | 10 | — | — |
| vit | SL | 100 | S | P | 0 | KC | 5 | 0 | 0 | 705.00000 | 750.00000 |
| vit | SL | 100 | S | P | 0 | KK | 5 | 0 | 0 | 710.00000 | 755.00000 |
| vit | SL | 100 | S | P | 0 | ZC | 5 | 0 | 0 | 675.00000 | 780.00000 |
| vit | SL | 100 | S | P | 0 | ZK | 5 | 0 | 0 | 605.00000 | 640.00000 |
| vit | SL | 100 | S | P | 25 | KC | 5 | 0 | 0 | 700.00000 | 765.00000 |
| vit | SL | 100 | S | P | 25 | KK | 5 | 0 | 0 | 685.00000 | 730.00000 |
| vit | SL | 100 | S | P | 25 | ZC | 5 | 0 | 0 | 970.00000 | 1040.00000 |
| vit | SL | 100 | S | P | 25 | ZK | 5 | 0 | 0 | 955.00000 | 990.00000 |
| vit | SL | 100 | S | S | 0 | KC | 5 | 0 | 0 | 1010.00000 | 1100.00000 |
| vit | SL | 100 | S | S | 0 | KK | 5 | 0 | 0 | 955.00000 | 1185.00000 |
| vit | SL | 100 | S | S | 0 | ZC | 5 | 0 | 0 | 675.00000 | 780.00000 |
| vit | SL | 100 | S | S | 0 | ZK | 5 | 0 | 0 | 605.00000 | 770.00000 |

T90 contrasts are blank unless every paired seed has a recorded event in every arm used by that contrast. Unreached values stay null, with censor horizons stored separately.

## Guarded ΔP/ΔS ratio

| model | activation | t_end | history | source | tau | estimate | ci_low | ci_high | denominator_ci_low | estimable | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | ELU | 1 | S | P/S | 0 | — | — | — | -0.09383 | False | denominator_CI_lower_not_above_0.01 |
| mlp | ELU | 10 | S | P/S | 0 | 1.03609 | 0.98969 | 1.08638 | 0.20089 | True |  |
| mlp | R | 1 | S | P/S | 0 | — | — | — | -0.02651 | False | denominator_CI_lower_not_above_0.01 |
| mlp | R | 10 | S | P/S | 0 | 0.95984 | 0.91744 | 1.01045 | 0.01806 | True |  |
| mlp | SILU | 1 | S | P/S | 0 | — | — | — | -0.13183 | False | denominator_CI_lower_not_above_0.01 |
| mlp | SILU | 10 | S | P/S | 0 | 0.65429 | 0.55978 | 0.77253 | 0.03432 | True |  |
| vit | SL | 100 | S | P/S | 0 | 0.19959 | -0.15637 | 0.54423 | 0.08015 | True |  |

The ratio uses the primary parameter-clear U contrast and is reported only when the S denominator CI lower bound exceeds 0.01. This is not a mediation fraction.

## Arm means (all seeds)

| model | activation | t_end | history | source | tau | arm | U | ce_auc_mean | accuracy_end | ce_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | ELU | 1 | S | P | 0 | KC | 0.15076 | 0.60407 | 0.99192 | 0.08530 |
| mlp | ELU | 1 | S | P | 0 | KK | 0.15011 | 0.60028 | 0.99208 | 0.08427 |
| mlp | ELU | 1 | S | P | 0 | ZC | 0.37001 | 1.19308 | 0.96408 | 0.26199 |
| mlp | ELU | 1 | S | P | 0 | ZK | 0.32161 | 1.06466 | 0.97267 | 0.20952 |
| mlp | ELU | 1 | S | P | 25 | KC | 0.14625 | 0.58218 | 0.99175 | 0.08543 |
| mlp | ELU | 1 | S | P | 25 | KK | 0.14554 | 0.57809 | 0.99208 | 0.08427 |
| mlp | ELU | 1 | S | P | 25 | ZC | 0.37765 | 1.21879 | 0.96350 | 0.26573 |
| mlp | ELU | 1 | S | P | 25 | ZK | 0.33604 | 1.10922 | 0.97117 | 0.22109 |
| mlp | ELU | 1 | S | S | 0 | KC | 0.28252 | 0.95833 | 0.98558 | 0.13491 |
| mlp | ELU | 1 | S | S | 0 | KK | 0.28403 | 0.95873 | 0.98542 | 0.13576 |
| mlp | ELU | 1 | S | S | 0 | ZC | 0.37001 | 1.19308 | 0.96408 | 0.26199 |
| mlp | ELU | 1 | S | S | 0 | ZK | 0.32196 | 1.06508 | 0.97292 | 0.20957 |
| mlp | ELU | 10 | S | P | 0 | KC | 0.58851 | 1.60259 | 0.69992 | 0.85552 |
| mlp | ELU | 10 | S | P | 0 | KK | 0.58370 | 1.59167 | 0.70242 | 0.84301 |
| mlp | ELU | 10 | S | P | 0 | ZC | 0.36985 | 1.17687 | 0.96392 | 0.25809 |
| mlp | ELU | 10 | S | P | 0 | ZK | 0.35308 | 1.12720 | 0.96517 | 0.24603 |
| mlp | ELU | 10 | S | P | 25 | KC | 0.58569 | 1.57647 | 0.70017 | 0.85351 |
| mlp | ELU | 10 | S | P | 25 | KK | 0.58174 | 1.56755 | 0.70242 | 0.84301 |
| mlp | ELU | 10 | S | P | 25 | ZC | 0.47240 | 1.42318 | 0.90500 | 0.44933 |
| mlp | ELU | 10 | S | P | 25 | ZK | 0.45660 | 1.37990 | 0.90992 | 0.42936 |
| mlp | ELU | 10 | S | S | 0 | KC | 0.58090 | 1.58001 | 0.71683 | 0.80510 |
| mlp | ELU | 10 | S | S | 0 | KK | 0.57548 | 1.56850 | 0.72175 | 0.79056 |
| mlp | ELU | 10 | S | S | 0 | ZC | 0.36985 | 1.17687 | 0.96392 | 0.25809 |
| mlp | ELU | 10 | S | S | 0 | ZK | 0.35248 | 1.12668 | 0.96533 | 0.24587 |
| mlp | ELU | 29 | Pall | P | 0 | KC | 0.76788 | 2.00191 | 0.34933 | 1.63824 |
| mlp | ELU | 29 | Pall | P | 0 | KK | 0.76631 | 1.99826 | 0.35042 | 1.63276 |
| mlp | ELU | 29 | Pall | P | 0 | ZC | 0.57023 | 1.60197 | 0.70908 | 0.89146 |
| mlp | ELU | 29 | Pall | P | 0 | ZK | 0.56817 | 1.59321 | 0.70433 | 0.90097 |
| mlp | R | 1 | S | P | 0 | KC | 0.16444 | 0.67705 | 0.99742 | 0.08921 |
| mlp | R | 1 | S | P | 0 | KK | 0.16108 | 0.66656 | 0.99708 | 0.08799 |
| mlp | R | 1 | S | P | 0 | ZC | 0.32622 | 1.10386 | 0.97575 | 0.21629 |
| mlp | R | 1 | S | P | 0 | ZK | 0.26519 | 0.93575 | 0.98492 | 0.15813 |
| mlp | R | 1 | S | P | 25 | KC | 0.15633 | 0.63545 | 0.99800 | 0.08315 |
| mlp | R | 1 | S | P | 25 | KK | 0.15658 | 0.64163 | 0.99708 | 0.08799 |
| mlp | R | 1 | S | P | 25 | ZC | 0.31515 | 1.07941 | 0.98300 | 0.18479 |
| mlp | R | 1 | S | P | 25 | ZK | 0.29331 | 1.01984 | 0.98550 | 0.16846 |
| mlp | R | 1 | S | S | 0 | KC | 0.30917 | 1.03160 | 0.97317 | 0.20855 |
| mlp | R | 1 | S | S | 0 | KK | 0.30178 | 1.02127 | 0.97667 | 0.20436 |
| mlp | R | 1 | S | S | 0 | ZC | 0.32622 | 1.10386 | 0.97575 | 0.21629 |
| mlp | R | 1 | S | S | 0 | ZK | 0.26497 | 0.93561 | 0.98392 | 0.15736 |
| mlp | R | 10 | S | P | 0 | KC | 0.88226 | 2.28858 | 0.12225 | 2.27600 |
| mlp | R | 10 | S | P | 0 | KK | 0.88167 | 2.28767 | 0.12275 | 2.27449 |
| mlp | R | 10 | S | P | 0 | ZC | 0.85428 | 2.20706 | 0.16958 | 2.11347 |
| mlp | R | 10 | S | P | 0 | ZK | 0.85351 | 2.20409 | 0.17233 | 2.11511 |
| mlp | R | 10 | S | P | 25 | KC | 0.88209 | 2.28779 | 0.12208 | 2.27657 |
| mlp | R | 10 | S | P | 25 | KK | 0.88155 | 2.28676 | 0.12275 | 2.27449 |
| mlp | R | 10 | S | P | 25 | ZC | 0.85899 | 2.22270 | 0.16492 | 2.13739 |
| mlp | R | 10 | S | P | 25 | ZK | 0.85925 | 2.22185 | 0.16367 | 2.14556 |
| mlp | R | 10 | S | S | 0 | KC | 0.88343 | 2.29238 | 0.11983 | 2.28111 |
| mlp | R | 10 | S | S | 0 | KK | 0.88318 | 2.29178 | 0.11983 | 2.28076 |
| mlp | R | 10 | S | S | 0 | ZC | 0.85428 | 2.20706 | 0.16958 | 2.11347 |
| mlp | R | 10 | S | S | 0 | ZK | 0.85345 | 2.20165 | 0.17308 | 2.10975 |
| mlp | SILU | 1 | S | P | 0 | KC | 0.13879 | 0.56318 | 0.99500 | 0.06064 |
| mlp | SILU | 1 | S | P | 0 | KK | 0.13803 | 0.56075 | 0.99525 | 0.06103 |
| mlp | SILU | 1 | S | P | 0 | ZC | 0.34309 | 1.12029 | 0.97192 | 0.20564 |
| mlp | SILU | 1 | S | P | 0 | ZK | 0.28505 | 0.96312 | 0.97842 | 0.15508 |
| mlp | SILU | 1 | S | P | 25 | KC | 0.13278 | 0.53040 | 0.99525 | 0.05859 |
| mlp | SILU | 1 | S | P | 25 | KK | 0.13336 | 0.53388 | 0.99525 | 0.06103 |
| mlp | SILU | 1 | S | P | 25 | ZC | 0.31971 | 1.06707 | 0.97567 | 0.17468 |
| mlp | SILU | 1 | S | P | 25 | ZK | 0.29342 | 0.99038 | 0.97967 | 0.15136 |
| mlp | SILU | 1 | S | S | 0 | KC | 0.21764 | 0.79923 | 0.99517 | 0.07011 |
| mlp | SILU | 1 | S | S | 0 | KK | 0.21976 | 0.80473 | 0.99442 | 0.07394 |
| mlp | SILU | 1 | S | S | 0 | ZC | 0.34309 | 1.12029 | 0.97192 | 0.20564 |
| mlp | SILU | 1 | S | S | 0 | ZK | 0.28503 | 0.96328 | 0.97825 | 0.15463 |
| mlp | SILU | 10 | S | P | 0 | KC | 0.53753 | 1.59136 | 0.64417 | 1.16950 |
| mlp | SILU | 10 | S | P | 0 | KK | 0.53450 | 1.58360 | 0.64575 | 1.16366 |
| mlp | SILU | 10 | S | P | 0 | ZC | 0.50274 | 1.50123 | 0.79808 | 0.75926 |
| mlp | SILU | 10 | S | P | 0 | ZK | 0.47868 | 1.44057 | 0.79442 | 0.76958 |
| mlp | SILU | 10 | S | P | 25 | KC | 0.53412 | 1.57565 | 0.64542 | 1.16545 |
| mlp | SILU | 10 | S | P | 25 | KK | 0.53226 | 1.57075 | 0.64575 | 1.16366 |
| mlp | SILU | 10 | S | P | 25 | ZC | 0.56986 | 1.67414 | 0.70492 | 1.01821 |
| mlp | SILU | 10 | S | P | 25 | ZK | 0.54813 | 1.62637 | 0.71358 | 1.01881 |
| mlp | SILU | 10 | S | S | 0 | KC | 0.55592 | 1.60636 | 0.62850 | 1.19306 |
| mlp | SILU | 10 | S | S | 0 | KK | 0.55259 | 1.59901 | 0.63233 | 1.18654 |
| mlp | SILU | 10 | S | S | 0 | ZC | 0.50274 | 1.50123 | 0.79808 | 0.75926 |
| mlp | SILU | 10 | S | S | 0 | ZK | 0.47765 | 1.44008 | 0.79767 | 0.77351 |
| vit | SL | 100 | S | P | 0 | KC | 0.25530 | 0.74811 | 1.00000 | 0.00620 |
| vit | SL | 100 | S | P | 0 | KK | 0.26663 | 0.77773 | 0.99912 | 0.01251 |
| vit | SL | 100 | S | P | 0 | ZC | 0.23241 | 0.71331 | 0.99888 | 0.02032 |
| vit | SL | 100 | S | P | 0 | ZK | 0.20145 | 0.61528 | 0.99984 | 0.00926 |
| vit | SL | 100 | S | P | 25 | KC | 0.25950 | 0.74762 | 0.99976 | 0.00800 |
| vit | SL | 100 | S | P | 25 | KK | 0.25604 | 0.73239 | 0.99912 | 0.01251 |
| vit | SL | 100 | S | P | 25 | ZC | 0.40127 | 1.10723 | 0.97856 | 0.10510 |
| vit | SL | 100 | S | P | 25 | ZK | 0.38856 | 1.06573 | 0.98264 | 0.08444 |
| vit | SL | 100 | S | S | 0 | KC | 0.34709 | 1.15012 | 0.99656 | 0.03376 |
| vit | SL | 100 | S | S | 0 | KK | 0.35352 | 1.15716 | 0.98904 | 0.05032 |
| vit | SL | 100 | S | S | 0 | ZC | 0.23241 | 0.71331 | 0.99888 | 0.02032 |
| vit | SL | 100 | S | S | 0 | ZK | 0.20444 | 0.62277 | 0.99976 | 0.01144 |

Files: seed_metrics.csv, contrasts.csv, verdict.json, representative_curves.png/.pdf, parameter_contrast_forest.png/.pdf, provenance.json.
