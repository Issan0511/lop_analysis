# Shuffled singleton minibatches: actual long-time Adam self direction in a native CNN

2026-10-09. This removes full-batch updating from the preceding three-image construction. It retains binary output, specially constructed images, a local active/winner region, and sufficiently small power-decaying rates. The new argument handles independently shuffled epochs and the same labels reused within a task. It establishes a [local stationary linearization](adam_singleton_linearization.md) and its stable endpoint, rather than assuming that the full-batch coordinate-sign theorem also applies to minibatches.

## 1. Precise theorem

Fix a finite number of epochs E>=2, H=3E updates per task, ridge lambda>0, epsilon>0, beta1=.9, beta2=.999, and 1/2<p<=1. There exist three distinct positive images and a nonempty open subset of the entire independent raw-parameter space of the native CNN

    RGB32x32 -> Conv5/pad2(16), ReLU, MaxPool2
    -> Conv5/pad2(16), ReLU, MaxPool2
    -> FC100, ReLU -> FC100, ReLU -> binary FC2,

with every ordinary bias present, having the following property. Each task independently draws one uniform binary label for each image. In each epoch an independent uniform permutation of the three image indices is drawn. The network takes one ordinary Adam step on each image in that order, with its task label reused in subsequent epochs. All raw parameters are updated; moments and the global bias-correction counter continue across tasks and start from zero only once.

For every prescribed alpha in (0,1), a sufficiently small eta0>0 makes the actual process with

    eta_(k,r)=delta_k=eta0/(k+1)^p, r=1,...,H,

satisfy, with probability at least 1-alpha for each initial point in a fixed smaller initial slice:

- all raw parameters converge to a finite limit and all three limiting class contrasts vanish;
- the selected first-Conv spatial/data mean has a strictly negative net displacement with a uniform finite lower margin;
- the original full and literal-self all-raw NTK capacities have positive mean-direction derivatives at all iterates and the limit, for the specified ridge;
- all required ReLU/Pool, rank, sensitivity and capacity margins persist, including intermediate optimizer steps;
- actual rates have divergent sum and convergent squared sum.

The same constants and rate bound can serve the initial slice, but the probability statement is per initial point. It does not assert one noise event controlling an uncountable set of initial states. No moment reset, projection, refit, frozen block or state-dependent damping is used. The construction keeps the hidden ReLUs positive; it does not prove gate death or unconditional expected displacement after failed exits.

## 2. Exact singleton response at uniform outputs

Let z_n=(f_n,+-f_n,-)/2, J=D_theta z, and s_nj=J_nj. A singleton raw gradient is

    g_j=s_nj[tanh(z_n)-Y_(task,n)].

At z=0, its square is s_nj^2, independent of the label. For one frozen stationary phase, group EMA weights by the pair (task q, image n): A_qn for the first moment and B_qn for the second. Both arrays sum to one. Define

    a_n=sum_q A_qn, b_n=sum_q B_qn,
    c_n=sum_q A_qn B_qn,
    sigma_j^2=sum_n b_n s_nj^2.

Conditional on the complete schedule history, averaging the labels after differentiating the Adam quotient gives the exact phase-summed output response

    L_jn=s_nj sum_(r=1)^H E_schedule[
       a_n/(sigma_j+epsilon)
       -s_nj^2 c_n/{sigma_j(sigma_j+epsilon)^2}].

Label correlations survive precisely when two entries refer to the same task and image. Thus the formula includes repeated labels and all old moments. It is not the independent-new-label-per-step response.

Since c_n<=a_n b_n and s_nj^2 b_n<=sigma_j^2, every bracket is at least `epsilon a_n/(sigma_j+epsilon)^2`. The exchangeable shuffle law gives E[a_n]=1/3 at each phase. With gamma_j=max_n|s_nj| and c0=H/(3 epsilon),

    L_jn=s_nj d_jn,
    c0/(1+gamma_j/epsilon)^2 <= d_jn <= c0,
    |L_jn-c0 s_nj| <=2c0(gamma_j/epsilon)|s_nj|.

These are statements about the linearization at uniform outputs. In general d_jn depends on both the image and coordinate; the full stationary field is not asserted to be a diagonal positive matrix times the population gradient.

The stationary field vanishes at z=0 for every sensitivity array, by complementing all task labels. Therefore its raw derivative is exactly `D Fbar(theta_*)=L J`. Derivatives of the sensitivities contribute zero there.

## 3. Nonempty stable reference, including the unscaled output biases

Use the three-image native reference with strict branches, positive hidden paths, a full-row-rank feature matrix, and a common positive head-contrast level canceled by its initial output bias. The [reference construction](adam_singleton_reference.md) multiplies the last hidden FC weight matrix and bias by t>0, the output-head contrast by t, and the canceling output-bias contrast by t^2. These are initial values only; all blocks subsequently train freely.

The last features become t H0. The non-output-bias half-contrast Jacobian R has the exact disjoint-block form

    R(t)=[t J1, t^2 J2].

The free head-weight block makes J1 full row rank. Output-bias sensitivities remain +/-1/2 and cannot be called small compared with epsilon. Shuffle symmetry instead makes their exact contribution to the normal matrix a positive common-image term `beta_b 11^T`.

Writing L_R=c0 R^T+E_R gives

    J L=beta_b 11^T+c0 R R^T+R E_R,
    ||R E_R||_op<=2c0(gamma_R/epsilon)||R||_op||R||_F,

where gamma_R=max|R|. Thus the finite, current-reference inequality

    2(gamma_R/epsilon)||R||_op||R||_F
           < sigma_min(R)^2                                      (3.1)

guarantees Sym(JL)>0. It compares the error with the complete three-output spectral margin, including the small modes that distinguish nearby images.

This condition is nonempty. If Gamma bounds the entries of J1,J2 and C^2=||J1||_F^2+||J2||_F^2, then for t<=1,

    gamma_R<=t Gamma,
    ||R||_op||R||_F<=t^2 C^2,
    sigma_min(R)^2>=t^2 lambda1,
    lambda1=lambda_min(J1 J1^T)>0.

Any `0<t<min(1,epsilon lambda1/(4 Gamma C^2))` gives a strict normal margin at least `c0 t^2 lambda1/2`. Unlike a generic absolute small-gradient argument, the retained rank margin is explicitly compared with the error. Only non-output-bias coordinates use this approximation.

The original capacity signs also survive at the same reference. For either full or literal-self network,

    K=K_bias+t^2 K1+t^4 K2,
    D_u K=t^2 K1'+t^4 K2',

where `K_bias=11^T tensor I2`. Dividing the capacity slope by t^2 gives a continuous function at t=0. At coincident images its limit is strictly positive because the free head-weight block contributes `2<h,D_u h>I2`; the last-hidden-FC block has a nonnegative contribution. First fix a sufficiently small nonzero image separation preserving the rank-three features, then choose t sufficiently small for (3.1) and both full/self slopes. The [reference note](adam_singleton_reference.md) makes this two-parameter continuity argument explicit. It does not assume an unscaled capacity margin stays uniformly positive as t tends to zero.

Every raw sensitivity is nonzero for fixed t>0. In a small neighborhood, each singleton gradient therefore has a positive absolute floor for both labels. Together with the strict CNN branch this makes the frozen stationary field smooth, including all its required derivatives.

## 4. Smooth endpoint with a nonsymmetric stable normal matrix

Treat the frozen response as a smooth function Rbar(z,S) of independent outputs and sensitivities. Since Rbar(0,S)=0,

    Fbar(theta)=V(theta) z(theta),
    V(theta)=integral_0^1 D_z Rbar(q z(theta),S(theta)) dq,
    V(theta_*)=L.

Thus A(theta)=J(theta)V(theta) has positive definite symmetric part on a smaller compact neighborhood of the stable reference. In the averaged ODE `theta_dot=-Fbar`,

    z_dot=-A(theta)z,
    d||z||^2/dt=-2 z^T A(theta)z <=-2a||z||^2

for a fixed a>0. The whole parameter speed is O(||z||), so a finite path-length estimate and a reserved boundary distance prove local containment and convergence to M={z=0}.

The [endpoint proof](adam_singleton_geometry.md) adapts the previous quantitative variational construction to this nonsymmetric A. In coordinates (y,z), the system is `y_dot=-B(y,z)z`, `z_dot=-A(y,z)z`. For compact derivative bounds A1,B0,B1, a small normal radius delta satisfying A1 delta<=a/2 gives a stable normal variational block with rate nu=a/2. Tangential coupling is O(delta exp(-a t)). The explicit integral absorption

    (delta/a)[B1+(B0+B1 delta)A1/nu] <=1/2

bounds the first variation; the second forcing contains either z or a decaying normal first variation. The first and second flow derivatives converge uniformly. This proves a C2 endpoint pi with finite derivative bounds, rather than assuming a stable foliation.

Its identities and reference derivative are

    pi|M=id, Dpi Fbar=0,
    Dpi(theta_*)=I-Q, Q=L(JL)^(-1)J.

The linear formula follows by solving the normal equation with `exp[-(JL)t]`. Positive symmetric part is sufficient; JL need not be symmetric. Q is a projection and QL=L.

## 5. An exact positive mean ray and a finite open cone

Let u be the true mean-input-patch/bias direction of the selected first Conv. Its supported entries and sensitivities s_nj are positive. The singleton coefficient bound in section 2 yields `L^T u>0` componentwise. Choose the current-reference normal ray

    v_*=L L^T u.

Then Qv_*=v_* and

    b_* = u^T Qv_* = ||L^T u||^2 >0.

This uses the actual stationary singleton response. The ray can change several raw parameter blocks. No claim that it moves only the selected filter is needed for a statement about that filter's eventual affine mean.

Let ell=u^T Q and let C_m bound the entry-sum norm of the Hessian of `m-m o pi` on a convex reference ball. Choose q>0 and 0<r_L<r_U such that

    q<b_*/(4||ell||_1),
    M_v=||v_*||_infinity+q,
    r_U<b_*/(2 C_m M_v^2),

omitting the last condition when C_m=0 and shrinking r_U for the chart radius. For `theta0=theta_*+r(v_*+e)`, r in (r_L,r_U), ||e||_infinity<q, Taylor's theorem gives

    m(theta0)-m(pi(theta0))>b_* r/2>=Delta,
    Delta=b_* r_L/2>0.

This union of open balls permits independent perturbations of every raw weight and bias. The endpoint sign is derived from L,J,u and a finite remainder. It is not imposed as an assumption about the future stochastic trajectory.

## 6. Actual moving Adam with shuffled reused-label tasks

Use the existing [taskwise averaging](adam_task_averaging.md) and [C2-observable averaging](adam_observable_averaging.md), sections 1-5. The task innovation now contains both its three labels and its E permutations. Its H phase gradients differ. The general averaging theorem permits this; the separate full-batch sign lemma is not invoked.

For all ambient coordinates of pi and the observable E_z=||z||^2, the theorem gives convergent remainders and exact cumulative task-boundary identities

    pi(theta_K)=pi(theta0)+R_K^pi,
    E_z(theta_K)=E_z(theta0)
       -sum_(k<K)delta_k [2 z^T A z](theta_k)+R_K^z.

The simultaneous all-times error budgets can be made arbitrarily small with eta0, with probability at least 1-alpha. They include retained moments, initial zeros, global bias correction, changing parameters, all reused-label correlations, schedule randomness and Taylor errors. Each observable gets its own Poisson/martingale decomposition.

Choose endpoint coordinate selection T so that `(T pi,z)` is a local chart and ||T||_infinity=1. Form nested compact chart tubes with endpoint radii rho0<rho1<rho2 and squared-output radii e0<e1<e2. The largest tube is compactly contained in the common region of strict routing, nonzero sensitivities, full row rank, Sym(JV)>=aI, and both capacity signs. Put the initial cone inside the smallest tube. Let d_* be the positive infinity-distance from the middle tube to the complement of the largest tube. Require

    e_pi<min(rho1-rho0,Delta/(2||u||_1)),
    e_z<e1-e0,
    H*73*eta0<d_*.

These budgets have strictly positive solutions. Starting at a task boundary in the middle tube, the universal raw Adam bound keeps every next phase and the next boundary inside the largest tube. The cumulative identities then return that boundary to the middle tube: pi stays within e_pi of its original value and E_z increases by at most e_z because its averaged drift is nonnegative. Induction proves permanent containment. Smooth extensions used in the averaging argument therefore coincide with the actual CNN forever on this event.

The nonnegative dissipation sum is finite and E_z converges. Its lower drift bound `2a E_z` and `sum delta_k=infinity` force E_z->0. Every pi component converges, so the chart gives convergence of the entire parameter vector. Intermediate phases have vanishing movement and converge too. Finally,

    m(theta0)-m(theta_infinity)
       >=Delta-||u||_1 e_pi>Delta/2.

The proof uses no estimate of total absolute stochastic path length and no sign restriction on individual updates. The rate is deterministic, so `sum eta=H sum delta=infinity` and `sum eta^2=H sum delta^2<infinity` without an additional damping argument.

## 7. Finite check and remaining limits

[verify_adam_singleton.py](verify_adam_singleton.py) and its [saved result](../../results/cnn_drive_1009/adam_singleton.json) check the construction with 331 independent raw parameters, RGB12x12, Conv5(2)/Conv5(2), FC3/3, all biases, three distinct images, two shuffled epochs and H=6. The final-layer/head initialization scale is 1e-18, with epsilon=1e-8. This very small scale is an existence witness, not a proposed practical initialization.

The spectral error ratio in (3.1) is approximately 5.38e-5, strictly below one. The corresponding positive lower normal margin is about 4.33e-35. Full/self original capacity slopes divided by the layer scale squared are about 2.74765 and 2.01236. A finite 12-step history enumerating all 64 reused-label histories verifies the singleton response derivative, with maximum coordinate-relative error below 5.50e-16. That finite-history calculation is not a stationary estimate.

The output-bias contribution for the infinite stationary shuffled schedule is separately computed with exact rational grouped moments; its common-mode coefficient is approximately 1.99630. The other numerical margins are float64 checks, while mathematical nonemptiness follows from the scaling and continuity proof. No numerical probability-certified learning rate or long-time success rate is reported.

The added scope is shuffled singleton minibatches with genuine within-task reuse and moving all-raw Adam. Binary output, three specially constructed nearby images, local strict routing, small nonbias sensitivities relative to epsilon, and sufficiently small decaying rates remain material restrictions. The result does not establish batch16, ten-class, constant-rate RL-CIFAR, arbitrary CIFAR images, negative mean or gate death. No other activation function is introduced.

Independent audits passed: [exact linearization and reference](adam_singleton_linearization_review.md), [finite verifier and independent rational bias covariance](adam_singleton_reference_review.md), and [nonsymmetric endpoint and integrated actual-process theorem](adam_singleton_longtime_review.md). The reference's rank terminology was clarified to refer to hidden features and the half-contrast Jacobian; the equilibrium logits themselves are zero.
