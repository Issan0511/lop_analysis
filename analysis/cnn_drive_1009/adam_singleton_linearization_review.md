# Independent audit: singleton Adam linearization and native reference

2026-10-09. Read-only review of `analysis/cnn_drive_1009/adam_singleton_linearization.md`, `adam_singleton_reference.md`, and their use in `adam_singleton_longtime.md`. No numerical tests were rerun and no repository files were edited.

**Verdict: PASS for the mathematical argument.** No blocking algebraic error or missing limit-order assumption was found. One terminology correction is noted below. The formulas agree with my independently derived `/tmp/cnn_adam_minibatch_geometry_1009.md`.

## 1. Exact reused-label correction

At zero contrast, conditioning on the complete image schedule makes the stationary squared-gradient sum label-independent. First- and second-moment weights must be grouped by the same task/image pair. With these grouped weights, the independent signs obey `E[Y_q Y_p]=1_{q=p}` and give

    L_jn = s_nj sum_r E[A_n/d - s_nj^2 Gamma_n/(sigma_j d^2)],
    d=sigma_j+epsilon.

The correction has the stated minus sign and cubic sensitivity before factoring out `s_nj`. Repeated visits within one task correctly contribute cross-products inside `A_q B_q`; no iid-per-optimizer-step substitution occurs. There is no extra `1/N` for a singleton loss. The factor `H/N` comes from averaging the schedule, not changing the loss normalization.

The inequality `s_nj^2 Gamma_n <= A_n sigma_j^2` yields the exact positive lower coefficient `epsilon A_n/d^2`. The relative error and the looser bound `2 gamma_j/epsilon` follow. These coordinate signs alone do not establish normal stability; both notes correctly retain the additional matrix condition.

## 2. Output biases and shuffle symmetry

For the two independent output biases, sensitivities are `+1/2` and `-1/2`, so their RMS equals `1/2` without approximation. Image-exchangeable scheduling makes the expected coefficient independent of the image. The two raw contributions add to `(c_b/2) 11^T`, not `c_b 11^T`.

The displayed formula

    c_b=(H/N)[epsilon+(1/2)(1-omega)]/(epsilon+1/2)^2

is consistent with the exact grouped-weight correction. Both notes explicitly reserve this symmetric common-image contribution for an exchangeable schedule law. Balanced coverage alone gives the phase-summed numerator weight `H/N` in a stationary task law, but is not used to claim the bias correction is image-independent. No deterministic asymmetric order is silently admitted.

## 3. Relative spectral stability and the native finite scale

From `L_R=c0 R^T+E_R`, the Frobenius error gives

    lambda_min(Sym(JL))
      >=c0[sigma_min(R)^2-(2 gamma_R/epsilon)||R||op||R||F].

This controls the full small-image-separation spectral gap and does not compare against a nonvanishing gap inherited from coincident images. Dropping the exact PSD output-bias block is conservative and valid.

The native raw groups are disjoint: `R(t)=[t J1,t^2 J2]`. The final hidden weights and biases, and the free output-head weights, have order `t`; every earlier raw hidden parameter has order `t^2`. Differentiation is in independent raw coordinates, not through the initialization amplitude. The free head block contains `+H0/2,-H0/2`, making `J1` full row rank. Thus

    sigma_min(R(t))^2>=t^2 lambda1,
    ||R(t)||op||R(t)||F<=t^2 C^2,
    gamma_R<=t Gamma.

The stated finite interval `0<t<epsilon lambda1/(4 Gamma C^2)` produces at least the claimed factor-two normal margin. The linearization note's more general Weyl bound for `tJ1+t^2J2` is also valid, though less sharp than exploiting the disjoint columns.

At each chosen strictly positive t, positivity of all raw sensitivities gives a per-singleton gradient/RMS floor near zero output. Smoothness and differentiation under the infinite-history expectation follow from that floor and the summable EMA weights. The reference t=0 is never used as a differentiable physical ReLU state.

## 4. Original full/self capacity and the two limits

For the full raw logit Jacobian, rather than only the contrast Jacobian, the three raw groups give exactly

    K=K_bout+t^2 K1+t^4 K2,
    D_u K=t^2 K1'+t^4 K2',
    K_bout=11^T tensor I2.

Output biases are included. Scaling the entire final hidden affine layer, including its bias, preserves these orders in the literal self architecture even when its earlier biases prevent a simple full/self output rescaling identity.

At three coincident images, `K1'=11^T tensor k1'`. Positive fixed-branch hidden polynomials make the final-hidden-FC contribution nonnegative, while the free head block contributes `2<h,D_u h>I2` strictly positively. Therefore the coefficient limit is correctly

    Ctilde_lambda(0,0)=3/[2(lambda+3)] tr(k1')>0.

For fixed positive ridge, the normalized trace has joint continuity in image separation and scale. A small rectangle can therefore be chosen first, followed by a fixed distinct full-rank image triple and a finite positive scale satisfying the normal-stability certificate. This ordering avoids the former rank-collapse/two-limit obstruction. The restriction to a prescribed ridge, or a compact ridge interval bounded away from zero, is correctly stated; no uniform neighborhood for all ridge values is inferred.

## 5. Integrated endpoint and actual-process use

At zero output, complementing every task label proves `Rbar(0,S)=0` for every sensitivity array. Hence no sensitivity-Hessian term survives in `D_theta Fbar=LJ`. The Hadamard factorization `Fbar=V z` and continuity preserve a positive symmetric part of `JV` in a small tube. Symmetry of `JL` itself is unnecessary.

The integrated ray `v=L L^T u` lies in `range L`, so the endpoint projection `Q=L(JL)^(-1)J` obeys `Qv=v` and the mean derivative equals `||L^T u||^2>0`. The finite Taylor cone is consequently noncircular. The integrated stochastic proof invokes the phase-dependent taskwise averaging theorem, not its separate full-batch sign lemma. Its nested-tube bootstrap reserves a within-task displacement before using cumulative identities, and its convergent energy remainder plus positive dissipation forces the output energy to zero. The probability statement remains per initial point, and actual deterministic power rates have infinite total mass.

## 6. Minor wording correction and scope

In `adam_singleton_reference.md` section 8, replace **“rank-three hidden/logit features”** with **“rank-three hidden feature matrix and half-contrast Jacobian.”** The reference logits vanish, and the three-by-two logit-value matrix cannot have rank three. The proof uses the feature matrix and the three-by-P Jacobian, whose ranks are correctly established elsewhere.

The audited result remains binary, locally constructed, and based on a finite but potentially extremely small initialization scale and sufficiently small decaying rates. No practical RL-CIFAR trajectory, arbitrary ten-class data, constant rate, gate death, or unconditional expected displacement on exceptional exit paths is implied. The finite verifier's response derivative is correctly labeled a finite-history check, not a stationary or infinite-time numerical certificate.
