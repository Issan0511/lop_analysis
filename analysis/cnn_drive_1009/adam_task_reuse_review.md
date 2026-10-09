# Independent audit: ordinary Adam with taskwise reused labels

Reviewed 2026-10-09:

- `/tmp/cnn_adam_task_reuse_1009.md`
- `/tmp/cnn_adam_task_reuse_verify_1009.py`
- `/tmp/cnn_adam_task_reuse_verify_1009.json`

Review result: **PASS, no blocking mathematical or interval-arithmetic issue found.** This was a read-only source/proof/output audit; I did not repeat the Torch run and did not edit the proof, verifier, or repository. The valid conclusion is a finite, conditional second-task mean-decrease certificate on a positive-probability reachable Adam history. It is not an invariant long-time result or a certificate for the RL-CIFAR training run.

## 1. Probability space, label reuse, and old-history carry

The conditioning is correctly placed at the task boundary. The new complete assignment is independent and uniform conditional on the entire old history. Inside the new task the same assignment is reused, so within-task gradients are allowed to depend on that already revealed assignment. No proof step incorrectly re-averages those gradients as fresh labels.

The frozen reference preserves all label dependence: each assignment is used in every occurrence of its images, and its reference moments are nonlinear functions of those same labels. It is not Adam applied to an expected gradient.

The verifier starts with zero moments once, performs two genuine old-task updates with assignment `(0,1)`, then deep-copies the complete resulting state, including `theta`, `m`, `v`, and global step `t=2`, into each new-label branch. Both reference and actual new-task steps use global bias-correction steps 3, 4, and 5. Moments are not reset. All four elements of `{0,1}^2` are enumerated and weighted by 1/4. This is precisely the ordinary iid-label law for two images, including the unbalanced `(0,0)` and `(1,1)` assignments. The previous assignment `(0,1)` has probability 1/4 under the declared old task law.

The ten-coordinate parameter update uses a gradient computed before any coordinates move. The coordinate loop then changes all ten parameters and their moments, so it is a simultaneous raw-parameter Adam step, not coordinate descent. The reference updates its moments with parameter updates disabled.

## 2. Universal movement bound and moment-error theorem

For a zero-initialized Adam history of length t, weighted Cauchy-Schwarz gives

    |mhat_t|/sqrt(vhat_t)
      <= [(1-beta1)/(1-beta1^t)]
         sqrt[(1-beta2^t)/(1-beta2) * sum_(i=0)^(t-1) (beta1^2/beta2)^i].

The first prefactor is at most 1, the numerator in the square root is at most 1, and the geometric sum is at most `1/(1-beta1^2/beta2)`. This proves the stated, conservative bound K whenever beta1^2<beta2. Adding epsilon decreases the ratio. Appending arbitrary frozen-reference gradients to the actual old history preserves applicability of this bound; the appended sequence does not need to be a realizable moving network trajectory for this numerical history inequality.

The box radius uses the infinity norm, so there is no omitted factor involving the number of parameters. For the actual state, `rho_r=K sum_(k<=r) eta_k` is valid coordinatewise.

The first-moment difference is linear and the old moments cancel. The second-moment argument is also correct: write corrected RMS as the Euclidean norm of a vector containing the common old-history coordinate and the weighted new gradient coordinates. Reverse triangle inequality yields C without differentiating sqrt or requiring a positive old variance. This is valid even if the target gradient or variance is zero.

The quotient identity

    q-q0 = [(mhat-mhat0)+q0(sigma0-sigma)]/(sigma+epsilon)

is exact. The denominator lower bound `max(0,sigma0-C)+epsilon` is valid and positive. Keeping the label-dependent q0 and sigma0 inside E_task avoids a false cancellation between the gradient and the Adam denominator.

The pathwise error and its boundary expectation therefore imply the stated certificate `E[D|boundary] >= Psi-E_bar`. The old first moment need not have a favorable sign. In the optional reciprocal-denominator proxy formula, the old signed moment contribution is indeed retained explicitly, and no independence among frozen gradients from different within-task steps is needed.

## 3. Geometry constants for the concrete CNN

The true mean input is 9/8, including all four spatial locations of both images; the mean observable is therefore `(9/8) w0`. The pooled spatial vectors are `(1,2)` and `(2,1)`. Positive Conv weights make every ReLU active and preserve each pool's unique winner.

The universal bound is checked rationally to be less than 73. Across all five actual updates, every raw coordinate stays within `.00365` of initialization. Equivalently, the new-task boundary is at most `.00146` from initialization, and its required radius `.00219` box remains within `.00365`. Hence the entire new-task geometry box, not just the computed four trajectories, lies in the same ReLU/MaxPool branch. In particular, weights remain positive and below .73, while head magnitudes remain below .16.

For one image, writing Q=3 and xmax=2, the column-l1 and row-l1 logit-Jacobian bounds are respectively

    max(2 V Q, W xmax) = 1.46,
    2(V+W)Q = 5.34.

The softmax covariance has absolute row sums at most 1/2, so the covariance Hessian term has row-sum bound `1.46*5.34/2=3.8982`.

The logit Hessian term is essential because the logits are bilinear in the jointly trained Conv/head parameters. For a Conv-coordinate row its bound is `sum_c |p_c-y_c| Q <= 2Q=6`; for a head-coordinate row the bound is at most xmax=2. Thus the uniform full CE Hessian row-sum bound is

    3.8982 + 6 = 9.8982 = 49491/5000 < 10.

A mean over the two images does not increase this bound. Consequently the stated infinity-norm gradient Lipschitz constant 10 is valid uniformly over all assignments and throughout the required box. The script includes the bilinear logit Hessian contribution and checks the exact rational bound.

## 4. Outward rational interval arithmetic

All operations used to certify inequalities have rational inputs and rational/integer implementations. The JSON float fields and optional Torch branch are diagnostic only and do not feed back into certification.

- `floor_grid` uses exact integer floor division, including negative inputs. `ceil_grid(x)=-floor_grid(-x)` is an outward upper rounding rule.
- Addition, negation, subtraction, and all four multiplication endpoint products enclose the exact operations.
- Reciprocal endpoints `(1/hi,1/lo)` are correctly ordered on both strictly positive and strictly negative intervals. Division asserts the divisor interval excludes zero.
- Squaring explicitly handles an interval crossing zero.
- For nonnegative rational x, `isqrt(floor(x*SCALE^2))/SCALE` is a valid lower sqrt bound. Increasing the upper result by one grid unit when its square is too small gives a valid upper bound. The code additionally checks both exact squared inequalities.
- For nonnegative x<=2, the 70-term exponential Taylor partial sum is a lower bound. Its first omitted term is the term of order 71, and every subsequent term ratio is at most x/72. The geometric-tail upper bound is therefore valid. Negative endpoints are handled by taking the reciprocal of the positive exponential enclosure. Monotonicity of exp justifies combining endpoint bounds.
- The two-class probability `exp(d)/(1+exp(d))` is enclosed even though dependency between its numerator and denominator is ignored; that only enlarges the interval.
- Reusing a single interval enclosure for the exact boundary state in reference and actual branches is sound. The loss of correlations only enlarges the enclosures.

The verifier computes A exactly as a rational and uses the outward upper endpoint for C. It uses the lower endpoint of sigma0 and the upper bound of |q0| in the movement error. This always increases the error bound. Four reference intervals are averaged with outward arithmetic; the error average remains exact rational arithmetic. The certified lower bound is the reference mean's lower endpoint minus that upper error.

One presentation detail: the claimed `Psi-E_bar > 1.966e-5` is valid but slightly weaker than the JSON's actual lower endpoint `1.9764796842378992e-5`. This is not an error or a required correction.

## 5. Raw network, capacity-self connection, and limitations

The algebraic forward and gradient formulas match the declared shared Conv1x1 -> ReLU -> MaxPool(1x2) -> free spatial head within the certified branch. Their gradient is mean cross entropy, and all ten independent raw parameters are represented. This concrete witness has no trained biases; the general theorem permits them when u and the geometry bounds include them. It does not silently establish a biased-network numerical witness.

Channel permutation symmetry is preserved exactly from the declared initialization and zero initial moments: corresponding raw gradients, first moments, second moments, and Adam updates match. Therefore the boundary equal-channel identity used to connect CE to the logit ray is valid on the actual old-task history. The opposite class-head rows are also preserved by two-class CE and coordinatewise Adam symmetry. The small box keeps the class contrast nonzero.

For both the full and literal channel-isolated models, the Conv Jacobian Gram is independent of w0 in this branch, while the head Gram has derivative `2 w0 mu_X [(X X^T) tensor I_2]`. The latter is positive definite here: X has rows `(1,2)` and `(2,1)`, so X X^T has eigenvalues 1 and 9. Thus both original logdet-capacity derivatives are strictly positive for every positive ridge. The actual three-step conditional Adam certificate establishes mean motion in their negative-gradient direction; CE is not being renamed as capacity-self.

The negative actual task decrease for assignment `(0,0)` and its negative later first moments are correctly retained in the expectation. Hence the witness is not a disguised assumption of pointwise positive update or momentum sign.

The martingale paragraph is conditional on having certificates along later actual task boundaries. The proof does not establish that persistence, infinite cumulative margin, unrestricted learning-rate validity, long-time sinking, or positivity of the large RL-CIFAR reference sum. The draft clearly retains these limitations. Ordinary Adam is used without AMSGrad or weight decay; alternative variants need separate formulas.
