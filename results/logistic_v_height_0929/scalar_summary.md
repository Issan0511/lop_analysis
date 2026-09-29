# Frozen-v scalar logistic audit

Replicates the prior three-point toy, then audits broad controls. These are scalar mechanistic models, not RL-MNIST experiments.
All entries are float64; log-domain Adam avoids arithmetic underflow. stopped_large_update means an update exceeded 1e6 or was nonfinite; it is an operational stopping rule, not a divergence theorem.

## Prior example and optimizer controls at 4000 steps

| optimizer | epsilon | v | z | final step / eta | loss | status |
|---|---:|---:|---:|---:|---:|---|
| gd | 0 | 0.1 | 0.19900359 | 0.0495026 | 0.683247 | ok |
| adam | 0 | 0.1 | 3.8719256 | 0.948914 | 0.518175 | ok |
| adam | 1e-08 | 0.1 | 3.8719248 | 0.948914 | 0.518175 | ok |
| no_first | 0 | 0.1 | 3.8701502 | 0.948459 | 0.518247 | ok |
| no_first | 1e-08 | 0.1 | 3.8701494 | 0.948459 | 0.518247 | ok |
| no_second | 0 | 0.1 | 4.0019728 | 1.00054 | 0.512936 | ok |
| no_second | 1e-08 | 0.1 | 4.0019719 | 1.00054 | 0.512936 | ok |
| instant | 0 | 0.1 | 4 | 1 | 0.513015 | ok |
| instant | 1e-08 | 0.1 | 3.9999991 | 1 | 0.513015 | ok |
| equal_memory | 0 | 0.1 | 3.9999995 | 1 | 0.513015 | ok |
| equal_memory | 1e-08 | 0.1 | 3.9999986 | 1 | 0.513015 | ok |
| gd | 0 | 1 | 1.3066495 | 0.213084 | 0.239588 | ok |
| adam | 0 | 1 | 2.6922672 | 0.495376 | 0.0655323 | ok |
| adam | 1e-08 | 1 | 2.6922671 | 0.495376 | 0.0655323 | ok |
| no_first | 0 | 1 | 2.6846176 | 0.494767 | 0.0660193 | ok |
| no_first | 1e-08 | 1 | 2.6846175 | 0.494767 | 0.0660193 | ok |
| no_second | 0 | 1 | 4.0303251 | 1.00901 | 0.0176125 | ok |
| no_second | 1e-08 | 1 | 4.0303245 | 1.00901 | 0.0176125 | ok |
| instant | 0 | 1 | 4 | 1 | 0.0181499 | ok |
| instant | 1e-08 | 1 | 3.9999994 | 0.999999 | 0.0181499 | ok |
| equal_memory | 0 | 1 | 3.9998709 | 0.999956 | 0.0181523 | ok |
| equal_memory | 1e-08 | 1 | 3.9998703 | 0.999955 | 0.0181523 | ok |
| gd | 0 | 10 | 0.59796079 | 0.0252407 | 0.00252662 | ok |
| adam | 0 | 10 | 0.6556814 | 0.0619668 | 0.0014194 | ok |
| adam | 1e-08 | 10 | 0.6556814 | 0.0619668 | 0.0014194 | ok |
| no_first | 0 | 10 | 0.65264677 | 0.0621353 | 0.0014631 | ok |
| no_first | 1e-08 | 10 | 0.65264677 | 0.0621353 | 0.0014631 | ok |
| no_second | 0 | 10 | 4.4374194 | 1.11189 | 5.3522e-20 | ok |
| no_second | 1e-08 | 10 | 2.3648213 | 0.0511959 | 5.36675e-11 | ok |
| instant | 0 | 10 | 4 | 1 | 4.24835e-18 | ok |
| instant | 1e-08 | 10 | 2.3525922 | 0.0572116 | 6.06487e-11 | ok |
| equal_memory | 0 | 10 | 3.9790484 | 0.994557 | 5.23856e-18 | ok |
| equal_memory | 1e-08 | 10 | 2.35354 | 0.0569796 | 6.00766e-11 | ok |

## Independent implementation and analytic checks

- PyTorch comparison max endpoint error: 3.078e-13.
- Gradient finite-difference max error: 1.512e-09.
- Exact effective-gain matched logit endpoint range: 3.317e-13.
- beta1<=beta2 Adam max step/eta: 1 (bound 1).
- GD finite-optimum error max: 7.105e-15.

## Same-state readout intervention, retained Adam moments

| target probability | gain | z before | z after 4000 | change |
|---|---:|---:|---:|---:|
| 1 | 0.1 | 2.6922671 | 5.2825689 | 2.5903018 |
| 1 | 1 | 2.6922671 | 4.4968958 | 1.8046287 |
| 1 | 10 | 2.6922671 | 2.6967483 | 0.0044812148 |
| 0.8 | 0.1 | 1.3855355 | 4.7179804 | 3.3324449 |
| 0.8 | 1 | 1.3855355 | 1.3862944 | 0.00075888646 |
| 0.8 | 10 | 1.3855355 | 0.13862944 | -1.246906 |

The separable target=1 model never moves z downward. The target=.8 finite-optimum control can move either way; its unique margin is log(4). Neither changes normalized shape of the symmetric two-point input.

## Long-time check (exponential tracking ansatz is not a convergence theorem)

| v | epsilon | t | observed delta margin | epsilon=0 ansatz | exp(m)/t divided by eta*v^2/eps |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0 | 200000 | 9.0573442e-05 | 9.0573441e-05 |  |
| 0.1 | 1e-08 | 200000 | 3.5667703e-05 | 9.0573441e-05 | 0.08829879301080706 |
| 0.1 | 0.0001 | 200000 | 8.967047e-06 | 9.0573441e-05 | 0.5071388655092324 |
| 1 | 0 | 200000 | 0.00041483873 | 0.00041483873 |  |
| 1 | 1e-08 | 200000 | 6.5124925e-06 | 0.00041483873 | 0.762772109396097 |
| 1 | 0.0001 | 200000 | 5.6942536e-06 | 0.00041483873 | 0.8730988308879084 |
| 10 | 0 | 200000 | 0.00049901624 | 0.00049901624 |  |
| 10 | 1e-08 | 200000 | 6.1776996e-06 | 0.00049901624 | 0.8089096253300974 |
| 10 | 0.0001 | 200000 | 5.5457146e-06 | 0.00049901624 | 0.9011442572904443 |
| 0.1 | 0 | 1000000 | 9.0573441e-05 | 9.0573441e-05 |  |
| 0.1 | 1e-08 | 1000000 | 1.2668296e-06 | 9.0573441e-05 | 0.7793694143360121 |
| 0.1 | 0.0001 | 1000000 | 1.1239094e-06 | 9.0573441e-05 | 0.8797401915461265 |
| 1 | 0 | 1000000 | 0.00041483873 | 0.00041483873 |  |
| 1 | 1e-08 | 1000000 | 1.0507208e-06 | 0.00041483873 | 0.9507365571755507 |
| 1 | 0.0001 | 1000000 | 1.0267946e-06 | 0.00041483873 | 0.9729135082258671 |
| 10 | 0 | 1000000 | 0.00049901624 | 0.00049901624 |  |
| 10 | 1e-08 | 1000000 | 1.0398143e-06 | 0.00049901624 | 0.961620103083758 |
| 10 | 0.0001 | 1000000 | 1.0202361e-06 | 0.00049901624 | 0.9800751414941813 |

## Limitations

- Endpoint ordering is not bidirectional response to an intervention.
- At fixed loss, inverse-v height is a conditional identity; different v runs have different losses at equal steps.
- Scalar symmetric points always have max/s=(max-median_midpoint)/s=1; no standardized tail mechanism exists here.
- The actual frozen-W2 intervention changes softmax logits and preserves optimizer history. Its v is a vector norm, not a signed scalar mean.
- No_second can be unstable; finite epsilon and positive moment memory change extreme/long-time limits.
- Parameter sweeps are sensitivity checks, not independent statistical replications.
