# Primary selected-unit recovery (descriptive registered secondary readout)

**DESCRIPTIVE ONLY.** This reads the completed primary job; it adds no intervention, inferential test, causal claim, or hypothesis-significance analysis. Co-occurrence of gate opening and learning differences does not establish that one caused the other.

Exact scope: RL, checkpoint T=20, layer 2; the same 20 `target20` IDs per seed from `selections.json`; branches A/B20/C20/D20; future tasks 21-25; registered steps 0, 75, 375, 1500, 3000, 6000. Sink means q>=.95.

`dWin_interval_norm`, `db_interval_abs`, and `dWout_interval_norm` in the CSV are actual changes between adjacent measurement points. Table movement values sum those five interval norms within each task, then average over selected IDs; all-five values additionally average tasks 21-25. They are path lengths, not signed or net displacement.

## Selected target20 unit state and movement

Each state value is the mean over the 20 selected IDs. `resunk` is the count out of 20 at the task endpoint. Values are shown as first future task / mean across all five future tasks.

| seed | branch | q step0 | q endpoint | resunk | zmean step0 | zmean endpoint | gate step0 | gate endpoint | dWin path | db path | dWout path |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A | 1 / 0.9995 | 1 / 0.9995 | 20 / 20.0 | -107.6 / -106.3 | -108.6 / -107.1 | 4.167e-05 / 0.0003784 | 7.674e-10 / 0.0004284 | 0.01933 / 0.09375 | 0.001795 / 0.006963 | 0.0346 / 0.04415 |
| 0 | B20 | 1 / 1 | 1 / 1 | 20 / 20.0 | -107.6 / -106.5 | -104.8 / -108.1 | 4.167e-05 / 1.637e-05 | 1.35e-07 / 1.281e-05 | 0 / 0 | 0 / 0 | 0 / 0 |
| 0 | C20 | 0.4445 / 0.8684 | 0.9298 / 0.979 | 4 / 16.6 | -1 / -107 | -75.29 / -145.9 | 0.5289 / 0.1239 | 0.06235 / 0.01861 | 1.499 / 0.8747 | 0.09196 / 0.0558 | 0.4079 / 0.1777 |
| 0 | D20 | 0.4445 / 0.8622 | 0.9442 / 0.9712 | 10 / 17.2 | -1 / -79.36 | -81.5 / -105.8 | 0.5289 / 0.129 | 0.04891 / 0.02497 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1 | A | 0.9995 / 0.9997 | 0.9993 / 0.9998 | 20 / 20.0 | -105.7 / -115.9 | -112.4 / -120.4 | 0.0003873 / 0.0002626 | 0.0007157 / 0.0001935 | 0.1676 / 0.1121 | 0.0127 / 0.00922 | 0.05315 / 0.04531 |
| 1 | B20 | 0.9995 / 0.9996 | 0.9993 / 0.9997 | 20 / 20.0 | -105.7 / -114.3 | -112 / -118.2 | 0.0003873 / 0.000302 | 0.0005296 / 0.0002822 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1 | C20 | 0.4442 / 0.8759 | 0.9556 / 0.9868 | 11 / 18.2 | -1 / -114.3 | -88.8 / -156.8 | 0.529 / 0.1172 | 0.03927 / 0.01168 | 1.318 / 0.7642 | 0.08383 / 0.05171 | 0.3388 / 0.1437 |
| 1 | D20 | 0.4442 / 0.869 | 0.9503 / 0.9792 | 9 / 17.6 | -1 / -85.84 | -82.61 / -113.9 | 0.529 / 0.1228 | 0.04278 / 0.01785 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | A | 0.9999 / 0.9999 | 1 / 1 | 20 / 20.0 | -107.1 / -113.3 | -113.7 / -116.7 | 0.0001262 / 6.694e-05 | 1.108e-07 / 5.004e-05 | 0.0734 / 0.06283 | 0.007251 / 0.005287 | 0.04029 / 0.04204 |
| 2 | B20 | 0.9999 / 0.9999 | 1 / 0.9999 | 20 / 20.0 | -107.1 / -110.6 | -112.8 / -111.3 | 0.0001262 / 5.588e-05 | 2.627e-07 / 3.124e-05 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | C20 | 0.4392 / 0.8668 | 0.9344 / 0.9787 | 8 / 17.0 | -1 / -104 | -80.71 / -143.5 | 0.5323 / 0.1253 | 0.05897 / 0.01908 | 1.733 / 0.931 | 0.1138 / 0.058 | 0.4862 / 0.1964 |
| 2 | D20 | 0.4392 / 0.8727 | 0.9662 / 0.9839 | 14 / 18.6 | -1 / -94.31 | -97.99 / -124.1 | 0.5323 / 0.1195 | 0.02929 / 0.01382 | 0 / 0 | 0 / 0 | 0 / 0 |

## Dose endpoint learning

Endpoint is step 6000. Accuracy remains a fraction. First is task 21; all-five is the arithmetic mean across tasks 21-25.

| seed | branch | CE first | accuracy first | CE all-five mean | accuracy all-five mean |
|---:|---|---:|---:|---:|---:|
| 0 | C5 | 2.16838 | 0.157500 | 2.26024 | 0.122167 |
| 0 | D5 | 2.1679 | 0.152500 | 2.26804 | 0.116500 |
| 0 | C10 | 2.12743 | 0.169167 | 2.24371 | 0.128333 |
| 0 | D10 | 2.0888 | 0.190000 | 2.22571 | 0.134500 |
| 0 | C20 | 2.03154 | 0.215000 | 2.20989 | 0.143000 |
| 0 | D20 | 2.04682 | 0.205833 | 2.16682 | 0.156500 |
| 1 | C5 | 2.29595 | 0.125833 | 2.29649 | 0.118167 |
| 1 | D5 | 2.25976 | 0.137500 | 2.28708 | 0.123167 |
| 1 | C10 | 2.20411 | 0.160000 | 2.27544 | 0.126833 |
| 1 | D10 | 2.211 | 0.155833 | 2.27014 | 0.129667 |
| 1 | C20 | 2.12655 | 0.192500 | 2.23849 | 0.143333 |
| 1 | D20 | 2.08928 | 0.200000 | 2.19911 | 0.148333 |
| 2 | C5 | 2.27346 | 0.122500 | 2.28553 | 0.119333 |
| 2 | D5 | 2.26934 | 0.106667 | 2.29684 | 0.110333 |
| 2 | C10 | 2.19978 | 0.150000 | 2.27639 | 0.122167 |
| 2 | D10 | 2.25656 | 0.129167 | 2.28702 | 0.116500 |
| 2 | C20 | 2.03275 | 0.195000 | 2.20586 | 0.144833 |
| 2 | D20 | 2.09653 | 0.165000 | 2.20629 | 0.141000 |

## Post-hoc first-task estimator check

This post-hoc check does not change the registered primary verdict. `online_ce` is cumulative minibatch CE through update 6000; the probe AUC uses full-1200 probe CE at six points. They are distinct estimators. `0-75 probe area` is only the first trapezoid contribution after division by 6000.

| seed | branch | online CE through 6000 | 0-75 probe area contribution |
|---:|---|---:|---:|
| 0 | A | 2.318788 | 0.028995 |
| 0 | B20 | 2.313663 | 0.028993 |
| 0 | C20 | 2.222394 | 0.246308 |
| 0 | D20 | 2.613459 | 0.272265 |
| 1 | A | 2.295611 | 0.033484 |
| 1 | B20 | 2.303421 | 0.033568 |
| 1 | C20 | 2.247745 | 0.253983 |
| 1 | D20 | 2.437278 | 0.269933 |
| 2 | A | 2.314479 | 0.030833 |
| 2 | B20 | 2.313834 | 0.030898 |
| 2 | C20 | 2.265637 | 0.346254 |
| 2 | D20 | 2.879254 | 0.395306 |

Mean first-task online CE across seeds is 2.309626 for A and 2.245259 for C20: the online estimator improves descriptively even though the registered six-point probe AUC is worsened by the large initial-shock contribution.

Across the registered task endpoints, A/B20 remain almost entirely sunk. The lifted C20/D20 units are opened at step 0, but most or all are again at q>=.95 by step 6000. On task 21 and in the five-task endpoint averages, C20 and D20 each have lower CE and higher accuracy than A and B20 in all three seeds. Thus improved fresh-task learning accompanies the transient gate opening descriptively. C20 versus D20 learning is mixed across seeds, while only C20 permits selected layer-2 incident-parameter movement, so the improvement does not consistently track that updating.

D20 does not hold the selected units preactivation-fixed: its selected layer-2 incoming rows, biases, and outgoing columns have zero direct movement, but layer-1 parameters and activations still evolve. The CSV therefore includes mean layer-1 interval movement alongside each selected-unit record. These co-occurrences are not a mediation or causal decomposition. Gate opening also creates a large immediate forward shock, so endpoint or trajectory improvement cannot be attributed solely to gate state or parameter updating. A unit may open and re-sink between recorded endpoints.

## Sources

- `selections.json` SHA256: `4e561bae77f004679f2c81ac6b8b13e832a637c802371adc1213dbc34fc119f7`
- `units.npz` SHA256: `d93ed9ad9d33709bb2523dabaf24819828b72f94b3b2bac48df50ee7fe54dd21`
- `rows.csv` SHA256: `c5fbdaf7ffe84b88943a1fb213f6c2b9adad816ef08296f5b480b0f57b771ea5`
- Source directory: `/home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913/RL_t20_l2`
