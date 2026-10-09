# Independent audit: shuffled singleton finite reference verifier

2026-10-09. Audited the repository's `analysis/cnn_drive_1009/verify_adam_singleton.py` and saved `results/cnn_drive_1009/adam_singleton.json`, together with the analytical singleton reference. No repository edits or training runs were made. The torch verifier was read rather than rerun. The stationary output-bias fractions were independently recomputed using a different exact-rational pair-probability formula.

**Verdict: PASS. No blocking error found.** The finite-history enumeration, infinite stationary bias calculation, normalized spectral inequality, and normalized capacity calculation are correctly separated. The tiny-scale floating checks are not presented as practical learning-rate or success-probability certificates.

## 1. Reference scaling and raw-coordinate accounting

The code reuses the disclosed smaller native witness: RGB12-by-12, Conv5 widths2/2, both pools, FC3/FC3, binary head, all 331 raw parameters. It scales the last hidden FC weights/biases and the output weights by t=1e-18, and output biases by t². Earlier layers and biases retain their values. This is exactly the analytical two-group scaling.

The final two raw coordinates are the two output biases, in both the full and repacked literal-self networks. Their half-contrast sensitivities remain +/-1/2 and their directional Jacobian derivatives are zero. All remaining sensitivities have the required t or t² order. Excluding only those last two coordinates from the small-sensitivity estimate is therefore correct.

The source does not incorrectly scale the derivative of a freely trainable last-FC coordinate by t². It differentiates the scaled raw state first, then normalizes the resulting independent raw Jacobian by t. This distinction is necessary for the argument.

The saved rank checks are on normalized quantities: the hidden matrix divided by t has smallest singular value about 6.27e-4, and the non-output-bias half-contrast Jacobian divided by t has smallest singular value about 4.65e-4. The raw first-layer sensitivities remain positive but can be as small as 5.45e-40. There is no underflow-to-zero in the saved values. Strict positive activations and full/self pool winners are preserved; pooling precedes the scaled last FC, so its gaps remain of ordinary size.

## 2. The 64 labels enumerate a finite corrected history, not stationarity

`finite_history_linearization` fixes two valid six-step tasks, with two permutation epochs per task. Its six independent label variables are indexed by `(task,image)`, so the 64 assignments represent all binary labels for those two tasks. Each label is correctly reused on the two appearances of its image within the task. The selected orders are valid permutations in every three-step epoch.

The a and b arrays are the normalized geometric weights at the twelfth optimizer step, using denominators 1-beta1^12 and 1-beta2^12. Thus they correspond to finite zero-initialized, bias-corrected moments. They do not include an infinite negative-time stationary history and do not sum responses over all task phases.

The grouping calculation forms

    A_n = sum_task A_task,n,
    C_n = sum_task A_task,n B_task,n,
    sigma_j² = sum_history b_l s_(image_l,j)².

Differentiating the expected finite-history quotient with respect to the three logit contrasts at zero gives exactly

    L_jn = s_nj [A_n/(sigma_j+epsilon)
               -s_nj² C_n/{sigma_j(sigma_j+epsilon)²}].

The autograd comparison holds s fixed, as required by this response identity; the separate fact that the zero-logit expectation vanishes for every s supplies the raw-parameter chain rule in the theoretical construction. The saved maximum relative coordinate error is about 5.50e-16. The positive lower coefficient check uses the correct epsilon A_n/(sigma+epsilon)² bound.

The docstring, comments, and JSON accurately call this a finite normalized-EMA identity. It is neither a stationary Monte Carlo measurement nor a long-time optimizer experiment.

## 3. Independent exact audit of the H=6 stationary bias coefficient

For an output-bias coordinate, |s|=1/2. At the zero-logit reference its squared singleton gradients are constant, so the denominator is exactly schedule- and label-independent. The remaining stationary covariance depends only on repeated occurrences of the same `(task,image)` label.

The script enumerates all 6²=36 equally likely pairs of independent epoch permutations. Its `omega_finite(r)` is the expected grouped first/second moment product from the current task's first r steps, using unnormalized stationary-EMA weights. For complete past tasks, the product decays by (beta1 beta2)^H per task. Therefore

    omega_r = omega_current(r)
              +(beta1 beta2)^r omega_current(H)
                                      /[1-(beta1 beta2)^H]

is the correct infinite stationary phase value.

I checked all six fractions by an independent exact rational calculation that does not enumerate schedules. For positions i,j in the two epochs, the same-image probability is 1 when i=j, 0 for distinct positions in the same epoch, and 1/3 for positions in different independent epochs. Summing these probabilities against the first/second EMA powers reproduces `omega_current(r)`, after which the geometric past-task term is added. Every resulting fraction matched the saved JSON exactly.

The two output-bias rows together contribute

    J_B L_B = c_bias * ones(3,3),
    c_bias = 2s² sum_(r=1)^6
              [epsilon+s(1-omega_r)]/[3(epsilon+s)²].

With s=1/2 and epsilon=1e-8, the independently reproduced exact coefficient is

    1119038125942036605 / 560555577977778002
      = 1.9963018296580013... > 0.

The factors 1/N, the six-phase sum, and the two opposite raw bias coordinates are all correct. This is an exact rational stationary block calculation, distinct from the preceding finite 12-step normalized history.

## 4. The stability inequality separates the unscaled bias mode

Write R for the actual non-output-bias half-contrast Jacobian and Rhat=R/t. Let gamma=max|R_nj|. The analytical response bound gives

    ||R(L_R-c0 R^T)||_op
       <= c0 (2gamma/epsilon) ||R||_op ||R||_F,
    c0=H/(N epsilon).

The normalized relative comparison used by the code is therefore exactly

    e = [2gamma/epsilon] ||Rhat||_op ||Rhat||_F
                                  /sigma_min(Rhat)².

The t² factors in the numerator and the positive Gram gap cancel; this is not an absolute error being compared with an unrelated tiny signal. The output-bias block is excluded from gamma and Rhat and contributes the separate, exact positive common-mode matrix from Section 3.

The saved e is about 5.38e-5, below .5. Consequently the stated lower bound

    c0 t² sigma_min(Rhat)²(1-e)
      = approximately 4.33e-35

for the symmetric part of the full normal matrix is the correct analytical inequality evaluated at the witness. The verifier does not attempt to recover a 1e-35 eigenvalue by diagonalizing a floating matrix whose unscaled bias common mode is O(1), which would be numerically unreliable.

The positive ray coefficient bound is also valid: for each sample n, the code sums the coordinatewise lower bounds for L_jn against the nonnegative mean weights u_j, giving a positive lower bound for (L^T u)_n. Squaring and summing then bounds ||L^T u||² from below. The saved value about 7.15e-56 is derived from the stationary coefficient inequality, not from interpreting the finite-history derivative as the stationary response.

## 5. Capacity normalization preserves every raw block

Let J_R and J_R' be the all-logit raw Jacobian and its mean-direction derivative after deleting only the final two output-bias columns. The code sets

    Jhat=J_R/t,  Jhat'=J_R'/t,
    K_b=J_b J_b^T,
    Khat=Jhat Jhat^T,
    Khat'=Jhat' Jhat^T+Jhat Jhat'^T.

Hence, exactly in real arithmetic,

    K=K_b+t² Khat,
    K'=t² Khat',
    C_lambda/t² = .5 tr[(I+K_b+t² Khat)^(-1) Khat']

for the verifier's ridge lambda=1. `Khat` already contains the additional t²-order contribution of earlier raw layers; no earlier block is dropped. Output-bias K_b remains unscaled, and its zero directional derivative is correctly omitted only from K'. The literal-self calculation uses its own repacked Jacobian and the retained output biases.

The saved normalized slopes are approximately 2.74765 for full and 2.01236 for literal self, giving unnormalized slopes of order 1e-36. These signs match the analytical leading-block construction. At this scale floating addition may round the tiny t² Khat denominator correction away; that does not alter the algebraic normalization, but it reinforces that the output is a float64 consistency check rather than an exact real-arithmetic or interval certificate. The analytic normalized continuity argument supplies nonemptiness independently of such rounding.

## 6. Interpretation and scope

The saved result correctly distinguishes four different items:

1. a finite 64-label, 12-step quotient derivative identity;
2. an infinite stationary output-bias coefficient computed exactly with rational arithmetic;
3. a scale-normalized finite bound sufficient for stable normal linearization;
4. full/self raw-capacity signs evaluated through normalized matrices.

No training trajectory or success probability is measured. No practical scalar learning-rate threshold or numerical endpoint-cone radius is certified. The reference scale 1e-18 and the correspondingly small normal/mean margins are explicitly disclosed. Apart from the rational bias calculation, reported numerical margins remain float64 evaluations accompanying the analytical existence proof.

Within those limits, the verifier and saved result faithfully check the intended singleton construction, including all raw biases and the separation of the unscaled bias mode. No changes are required to the reviewed files for mathematical consistency.
