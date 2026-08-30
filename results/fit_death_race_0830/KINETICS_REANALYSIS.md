# 1-layer, 1M-step kinetics reanalysis

## Scope

Post-hoc analysis of the committed 1-layer CondA logs at width 5 and width
100.  No network was trained or updated.  Normalized MSE (NMSE) is
`eval_loss / Var(y)`, where `Var(y)` was recovered by deterministic replay of
the input/teacher stream and checked against the saved 1M checkpoint state.

## Main result

Centering delays entry into high-`dead_frac` states, but does not stop the
positive death trend within the 1M-step observation window.  At width 100 the
remaining trend is slow enough that it is dissociated from evaluation loss;
at width 5 the centered arm approaches the already-high error level of the
standard arm before the run ends.

| arm | mean dead_frac, task 1 -> 50 -> 100 | median OLS slope, tasks 1-50 | median OLS slope, tasks 51-100 | task-end NMSE, tasks 10-30 -> 80-100 |
|---|---:|---:|---:|---:|
| w5 standard | 0.400 -> 0.760 -> 0.800 | 0.007827 | 0.000807 | 0.511 -> 0.530 |
| w5 centered | 0.000 -> 0.560 -> 0.720 | 0.011208 | 0.004437 | 0.335 -> 0.494 |
| w100 centered | 0.000 -> 0.202 -> 0.294 | 0.003548 | 0.000948 | 0.002145 -> 0.002116 |

The late centered slope is 4.68 times larger at w5 than at w100.  The
independent seed-cluster bootstrap 95% interval for the w5-minus-w100 median
slope difference is `[0.000896, 0.006399]` (`B=20,000`; five seeds per arm).

## Width-5 standard arm: ceiling and reversibility

The first crossing of a `dead_frac` threshold is not an absorption time.
`dead_frac` is evaluated against the current task support, so it can decrease
when the task changes.

| threshold | first-crossing median | median first sampled time after which the run stays above the threshold |
|---|---:|---:|
| dead_frac >= 0.6 | 13k | 184k |
| dead_frac >= 0.8 | 161k | 943k |

Across all 1,001 probes per seed, the standard arm occupies `dead_frac >= 0.6`
for 89.4% of probes and `dead_frac >= 0.8` for 63.5%.  Therefore the previous
"98% of the run above the ceiling" wording, obtained by treating the median
first crossing as persistent, is too strong.  The direct occupancy estimate
still shows that most of the run is spent in a high-death regime.

For `dead_frac >= 0.6`, the first crossings by standard-arm seed are 11k,
161k, 0, 13k, and 98k.  The corresponding centered-arm crossings are 311k,
101k, 51k, 243k, and 511k.  The ratio of arm medians is 18.7, but this is a
descriptive first-crossing ratio.  Accounting for reversibility with a
persistent-crossing definition reduces the representative delay to roughly
2.8 times (184k versus about 511k, with right censoring in the centered arm).

## Interpretation

1. **Strong fit-induced absorption is rejected.** Fast fitting does not create
   an observed state in which `dead_frac` stops increasing.
2. **A rate effect is supported.** Centering delays high-death occupancy at w5,
   while the centered late trend is substantially slower at w100.
3. **The w5 final snapshot hides an earlier benefit.** Standard w5 is already
   near NMSE 0.5 in the early window.  Centered w5 starts lower but rises from
   0.335 to 0.494, approaching the standard-arm ceiling by 1M.
4. **Death and function are dissociated at w100.** Centered w100 increases from
   mean `dead_frac=0.202` to `0.294` between tasks 50 and 100 while task-end
   NMSE remains essentially unchanged.
5. **Long-run failure at w100 is not established.** The 1M logs cannot
   distinguish a slow march beyond the window from an asymptote near the
   observed level.

## Design limitations

- `dead_frac` is a task-dependent support label, not unit-ID survival or an
  irreversible death event.
- Runs carrying the same displayed seed in the standard and centered arms do
  not share a paired initialization, teacher, and input stream.  Arm contrasts
  and crossing-time ratios are therefore unpaired descriptive statistics.
- These logs establish kinetics and association, not causal attribution to a
  particular residual route such as the bias-coordinate path.

## Sources

- `results/center_selfcov_0814/lop_metrics_A_w5.csv`
- `results/center_selfcov_0814/lop_metrics_A_w100.csv`
- `results/center_selfcov_0814/ckpts/A_w5_step1000000.pt`
- `results/center_selfcov_0814/ckpts/A_w100_step1000000.pt`
- `results/fit_death_race_0830/task_end_series.csv`
- `results/fit_death_race_0830/dead_slope_by_seed.csv`
- `results/fit_death_race_0830/summary.json`
