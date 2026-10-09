# Independent review: native N=1200 reference

2026-10-09. Reviewed the final corrected version of `analysis/cnn_drive_1009/adam_1200_reference.md` in `/home/issan/Projects/claude/wt/cnn_drive_1009`. Repository files were not changed. No learning experiment was run.

**Verdict: PASS for the stated reference construction and its conditional connection to the existing stochastic theorem.** The previously identified head-common-component clarification is fixed: the reference raw head rows are explicitly `(+d,-d)`, and both rows are multiplied by `t`. A sufficiently small independently varied common head can subsequently be admitted by continuity at the finite positive reference. The exact scaling identities need not hold throughout that open neighborhood.

## 1. Native first-Conv minor and strict branches

The 16 centers in `{3,11,19,27}^2` are available after two center-convolution/Pool2 stages. Their 5-by-5 patches lie inside the 32-by-32 image and are mutually disjoint. Three color planes therefore give exactly 1200 distinct input coordinates. At the selector reference, the contrast backpropagates to channel `i` only at its assigned center, with coefficient `q_i>0`. Consequently

`partial_(W1_(i,c,a,b)) z = q_i x_(c,r_i+a,c_i+b)`

gives an invertible diagonal input derivative minor after ordering its coordinates. Shared convolution weights do not invalidate this calculation: the selector makes the backpropagated coefficient vanish at the other spatial positions.

The perturbation hierarchy correctly handles a subtlety in the literal-self network. At the initial diagonal Conv2 selector, deleting the other first channels would leave some Conv2 channels spatially constant, so strict self-pool winners do not follow by continuity from that limit. Adding positive **center** mixing first makes every second channel receive a positive multiple of the retained first-channel feature. This restores strict self winners. Choosing off-center coefficients still smaller than that finite margin then preserves both full and self winners. Positive FC perturbations preserve the pools and expose every raw path. The full minor remains invertible by continuity on the already strict full branch. No differentiability of the self network at the initial tied selector is needed.

## 2. Gradient-feature level and rank

On a fixed branch, `G(x)` is linear in the input, with no intercept: it is the derivative with respect to first-layer weights. Hidden-bias contributions to the logit are independent of the input. Thus the image-dependent contrast is `w^T G(x)` and `beta=M^T w>0` is its selected-pixel gradient.

For `T=(I-11^T/N)diag(1/beta)`, both `T beta=0` and `1^T T=0` hold. The simplex images therefore have a common contrast and exactly the original input mean. Positivity and all fixed full/self branches persist for sufficiently small nonzero `delta`.

The row-rank argument for `G_N=1 g0^T+delta T M^T` is correct: a left null vector first has zero sum after multiplication by `w`; invertibility of `M` then puts it in the left nullspace of `T`, namely `span(1)`, so it is zero. This proves gradient-feature rank 1200 without claiming hidden-feature rank greater than 100. Relative sensitivity closeness can be made arbitrarily tight while keeping nonzero `delta`; the singular-value margin is allowed to shrink with `delta`.

## 3. Independent raw scaling and normal margin

Scaling first-Conv weights and every hidden bias by `t`, both opposite raw head rows by `t`, and the canceling output-bias contrast by `t^2` makes all hidden activations scale by `t`. Remaining hidden weights stay fixed. Raw differentiation gives

`R(t)=[t J1,t^2 J2]`,

where `J1` contains first-Conv weights, all hidden biases and raw head weights. The first-Conv submatrix is the rank-1200 `G_N`. This uses independent raw coordinates, rather than a tied scale parameter. The resulting sufficient bound

`Sym(JL) >= c0 t^2 [lambda1-C_B t Gamma C^2/epsilon] I`

is consistent with the prior relative-response estimate, provided the separate output-bias common-mode coefficient is nonnegative. The finite positive choice of `t` is nonempty. The output-bias sensitivity remains unscaled and is correctly excluded from the small-sensitivity argument.

## 4. Full and literal-self capacity

With zero reference common head, all raw full-logit Jacobian blocks have the claimed powers, giving

`K=K_bias+t^2 K1+t^4 K2`, `K_bias=11^T tensor I2`.

For the actual additive first-filter mean direction `u`, first-Conv and hidden-bias Jacobians have zero directional derivative on the branch. The head-weight block has Jacobian `t h0` and directional derivative of order one. Other hidden-weight Jacobians have order `t^2` and directional derivative of order `t`. Hence

`D_u K=t K1'+t^3 K2'`

is correct. The same accounting applies after literal deletion of the other first channels and their second-layer input columns; self full-row-rank is not required.

At coincident images, the leading derivative is `11^T tensor [2<h0,D_u h0>I2]`, with a strictly positive inner product for each model. The fixed-ridge resolvent therefore gives

`lim_(t->0+) C_lambda(t,0)/t = 2N<h0,D_u h0>/(lambda+N)>0`.

Joint continuity in the coefficient formulas permits a fixed small nonzero image simplex, followed by a finite positive scale meeting both capacity signs and the normal bound. This is a positive-branch coefficient extension; it does not differentiate an actual zero-scale ReLU network. No scale-independent unnormalized capacity margin is claimed.

## 5. N=1200, B=16, E=400 regularity and remaining scope

With a new independent permutation in each epoch, the probability that every batch in a task is label balanced is exactly

`p_N r^E`, where `p_N=binom(1200,600)/2^1200` and `r=binom(16,8)^75/binom(1200,600)`.

The labels must first contain 600 of each sign; conditional on that event, each epoch has probability `r`, independently over its permutation. Sensitivity closeness makes every unbalanced batch simultaneously give a nonzero gradient lower bound for every raw coordinate. The task event can therefore be used in the inverse-RMS tail bound.

An independent standard-library exact rational comparison was performed during this audit:

`0 < p_N < 1` and `r < (999/1000)^300`.

It passed. For orientation, `p_N` is approximately `0.0230281`, `r` approximately `4.17001e-52`, and the right-hand side approximately `0.740707`. Thus, for every finite positive `E`,

`p_N r^E < beta2^(75 E * 8/2)` for `beta2=.999`,

including `E=400`, `H=30000`. The exact comparison, rather than these rounded displays, supports the inequality. Reusing one fixed partition would not justify the exponent `r^E`; the note correctly requires fresh epoch permutations.

The note correctly leaves the H=30000 output-bias sign to a companion proof and does not substitute the former H=4/H=6 certificate. The first-batch-only tail estimate is also explicitly excluded. The construction supports the binary, sufficiently small decaying-rate theorem once those companion components are supplied. It does not establish the actual ten-class constant-rate RL-CIFAR trajectory, nor assert a practical size for its learning-rate or initialization margins.
