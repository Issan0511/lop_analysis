# ELU Q audit — post-hoc numerical output

Stored Adam states; gradients/rates below are Euclidean gradient-flow diagnostics with eta=1.
Each seed/phase has four task states (10,20,30,40). No independent-unit statistical claim.

| act | seed | phase | population | n | median Q | Q positive | median Sr | median Sv | Qcomp/Q L1 | Q/DC sign agree |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ELU | 0 | 200 | all | 400 | -4.17803e-06 | 0.500 | -0.00210865 | -0.00481652 | 0.03642 | 0.985 |
| ELU | 0 | 200 | alive | 373 | -0.000636915 | 0.496 | -0.00263368 | -0.00525112 | 0.03826 | 0.984 |
| ELU | 0 | bias_fit | all | 400 | -0.000147518 | 0.305 | 0.00211524 | 0.00159632 | 1 | 0.527 |
| ELU | 0 | bias_fit | alive | 384 | -0.000175578 | 0.310 | 0.00227104 | 0.00171312 | 1 | 0.526 |
| ELU | 0 | end | all | 400 | -0.00162213 | 0.480 | 0.00201341 | 0.00243977 | 0.04426 | 0.993 |
| ELU | 0 | end | alive | 384 | -0.00151886 | 0.482 | 0.00220866 | 0.00273981 | 0.04569 | 0.992 |
| ELU | 0 | sw | all | 400 | -0.0109106 | 0.430 | -0.0450748 | -0.0598725 | 0.2786 | 0.930 |
| ELU | 0 | sw | alive | 383 | -0.0105481 | 0.431 | -0.0490926 | -0.0634973 | 0.2898 | 0.927 |
| ELU | 1 | 200 | all | 400 | -0.00117564 | 0.482 | -0.00258411 | -0.00707007 | 0.05343 | 0.980 |
| ELU | 1 | 200 | alive | 368 | -0.0023896 | 0.473 | -0.00312447 | -0.00825602 | 0.0575 | 0.978 |
| ELU | 1 | bias_fit | all | 400 | -0.000144247 | 0.282 | 0.00243233 | 0.00183004 | 1 | 0.515 |
| ELU | 1 | bias_fit | alive | 372 | -0.000210625 | 0.280 | 0.0027358 | 0.00205288 | 1 | 0.516 |
| ELU | 1 | end | all | 400 | -0.00407345 | 0.453 | 0.0025276 | -0.000293143 | 0.04475 | 0.973 |
| ELU | 1 | end | alive | 372 | -0.00394461 | 0.457 | 0.00274269 | 0.000693469 | 0.04794 | 0.970 |
| ELU | 1 | sw | all | 400 | -0.0101542 | 0.407 | -0.0498575 | -0.061534 | 0.2865 | 0.892 |
| ELU | 1 | sw | alive | 382 | -0.0110808 | 0.401 | -0.0514324 | -0.0663708 | 0.2994 | 0.887 |
| LR | 0 | 200 | all | 400 | 0 | 0.000 | -0.0159625 | -0.0159625 | 2.391e+31 | 0.000 |
| LR | 0 | 200 | alive | 345 | 0 | 0.000 | -0.0126444 | -0.0126444 | 2.088e+31 | 0.000 |
| LR | 0 | end | all | 400 | 0 | 0.000 | 0.00743332 | 0.00743332 | 1.61e+31 | 0.000 |
| LR | 0 | end | alive | 355 | 0 | 0.000 | 0.00789097 | 0.00789097 | 1.431e+31 | 0.000 |
| LR | 0 | sw | all | 400 | 0 | 0.000 | -0.0380177 | -0.0380177 | 1.739e+31 | 0.000 |
| LR | 0 | sw | alive | 346 | 0 | 0.000 | -0.0398044 | -0.0398044 | 1.542e+31 | 0.000 |
| LR | 1 | 200 | all | 400 | 0 | 0.000 | -0.017617 | -0.017617 | 2.837e+31 | 0.000 |
| LR | 1 | 200 | alive | 345 | 0 | 0.000 | -0.0205718 | -0.0205718 | 2.461e+31 | 0.000 |
| LR | 1 | end | all | 400 | 0 | 0.000 | 0.00931636 | 0.00931636 | 1.671e+31 | 0.000 |
| LR | 1 | end | alive | 348 | 0 | 0.000 | 0.00975028 | 0.00975028 | 1.443e+31 | 0.000 |
| LR | 1 | sw | all | 400 | 0 | 0.000 | -0.039588 | -0.039588 | 1.582e+31 | 0.000 |
| LR | 1 | sw | alive | 351 | 0 | 0.000 | -0.042148 | -0.042148 | 1.372e+31 | 0.000 |

The Qcomp/Q and Q/DC columns are meaningful here only for ELU; LR is the Q=0 control.

## Actual Adam norm trajectory (all units; median of per-unit ratios)

| seed | task | R | V | V/R | top |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 126.51 | 2.6294 | 0.0210039 | 8.3577 |
| 0 | 20 | 250.73 | 2.6594 | 0.0108007 | 8.1737 |
| 0 | 50 | 576.68 | 1.7084 | 0.00308491 | 10.221 |
| 0 | 100 | 1120.9 | 0.60048 | 0.000533703 | 20.106 |
| 0 | 200 | 2260.7 | 4.2045 | 0.00186404 | 34.982 |
| 1 | 10 | 122.61 | 2.5764 | 0.0212922 | 7.7332 |
| 1 | 20 | 248.99 | 2.7126 | 0.0106426 | 8.9869 |
| 1 | 50 | 561.06 | 1.7927 | 0.00323901 | 11.791 |
| 1 | 100 | 1085.4 | 0.59819 | 0.000547873 | 21.58 |
| 1 | 200 | 2250.4 | 4.4169 | 0.00196981 | 38.002 |

## Output-bias-only fit (input and readout weights held fixed)

| seed | task | sum abs Q before | after | after/before | Q sign flips | CE before | CE after |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 2.29741 | 0.156253 | 0.06801 | 0.55 | 0.728132 | 0.719898 |
| 0 | 20 | 2.67355 | 0.114285 | 0.04275 | 0.56 | 1.368473 | 1.362719 |
| 0 | 30 | 3.5216 | 0.0527465 | 0.01498 | 0.51 | 1.632578 | 1.622970 |
| 0 | 40 | 2.99395 | 0.0415762 | 0.01389 | 0.48 | 1.705732 | 1.698351 |
| 1 | 10 | 2.09064 | 0.173033 | 0.08277 | 0.41 | 0.814238 | 0.805566 |
| 1 | 20 | 2.41569 | 0.130883 | 0.05418 | 0.47 | 1.268511 | 1.262376 |
| 1 | 30 | 4.19712 | 0.0763179 | 0.01818 | 0.55 | 1.607978 | 1.592803 |
| 1 | 40 | 2.8779 | 0.0553895 | 0.01925 | 0.43 | 1.677923 | 1.671068 |

## Endpoint gradient versus the following ten-task interval (descriptive only)

| seed | observations | R increases | V increases | ratio increases | Q vs delta B sign | raw qdot vs delta q sign |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 300 | 1.000 | 0.257 | 0.000 | 0.527 | 0.483 |
| 1 | 300 | 1.000 | 0.283 | 0.000 | 0.547 | 0.493 |

## Checks

Maximum Sv-Sr-Q residual: 8.88e-16.
Maximum ELU finite-difference relative L1 error: 6.95e-08.
Maximum LR finite-difference absolute error: 1.11e-11.

## Limits

A local gradient at a stored state is not an integrated Adam update. Endpoint norm differences are actual saved-trajectory differences, but intermediate optimizer moments are missing.
Output-bias fitting and compensated rescaling are local diagnostic interventions at fixed hidden/readout weights, not natural trajectory decompositions.
The end-state norm join matches saved preactivations bit-for-bit at tasks 10 and 30, and per-unit maxima at all four tasks.
Mathematical ELU derivatives are used; the float32 expm1 training implementation has an exact-zero negative-tail derivative.
See interpretation.md for the derivation, selected comparisons, and implications.
