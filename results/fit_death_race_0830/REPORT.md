# Fit–death race validation (post-hoc, no training)

## Verdict

- **Strong form rejected:** fitting does not create an absorbing state in which `dead_frac` stops.
- **Weak rate form supported descriptively:** the late fractional-death trend is slower at w100, but remains positive.
- The CSV alone cannot establish causal direction or bistability; `dead_frac` is a task-dependent support label, not unit identity survival.

## Decisive numbers

- w100: all task-1 raw eval losses are at most 0.005145; mean dead_frac moves 0.000 -> 0.202 -> 0.294 at tasks 1, 50, 100.
- Across all 100 tasks, median task-end normalized MSE is 0.00211 at w100 versus 0.400 at w5. Task switches reheat the residual: at the first 1k probe the medians are 0.0147 and 0.488, respectively.
- Late median OLS slope (tasks 51-100): w5=0.004437, w100=0.000948 dead_frac/task; ratio=4.68x.
- Difference w5-w100=0.003489, independent seed-cluster bootstrap 95% CI [0.000896, 0.006399] (B=20,000).

## Seed ordering

Early fit is the median normalized MSE over tasks 1-10; normalization is eval_loss / exact Var(y).

| width | seed | early normalized MSE | final dead_frac |
|---:|---:|---:|---:|
| 5 | 0 | 0.25837 | 1.000 |
| 5 | 1 | 0.21671 | 0.800 |
| 5 | 2 | 0.384392 | 0.600 |
| 5 | 3 | 0.0997399 | 0.400 |
| 5 | 4 | 0.183293 | 0.800 |
| 100 | 0 | 0.00202176 | 0.190 |
| 100 | 1 | 0.00278809 | 0.380 |
| 100 | 2 | 0.00116887 | 0.270 |
| 100 | 3 | 0.00243611 | 0.340 |
| 100 | 4 | 0.00174953 | 0.290 |

- Spearman(early error, final dead_frac): w5=0.359, w100=0.700 (n=5 each; descriptive only).
- The proposed seed-3 versus seed-0 ordering holds for w5. It is not monotone over all five seeds: seed 4 has the smallest task-1 normalized error (0.0286) but ends at dead_frac=0.8, and seed 2 has the worst early-10-task error but ends at 0.6.
- Median seed-wise Spearman(first-1k NMSE, same-task net dead change): w5=0.013, w100=0.095; using the task mean gives w5=0.042, w100=0.090. These weak associations are exploratory and non-causal.

## Replay validation

For each width, only the input generator was advanced in 10,000-step segments. After 100 segments, `env.t` and the full final `flip_state` matched the saved 1M checkpoint exactly. No network training or update was executed.
