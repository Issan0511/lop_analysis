# Independent audit: open native CNN family and actual long-time Adam

2026-10-09. Read-only audit of:

- `adam_open_geometry.md`
- `adam_observable_averaging.md`
- the supporting `adam_task_averaging.md`
- their connection to `adam_native_bias_reference.md`.

**Verdict: PASS. No blocking mathematical defect found in the combined local, high-probability, actual-Adam long-time claim, under the stated N=1 binary/full-batch/fixed-finite-H/power-decaying-step restrictions.** No numerical run was performed for this audit. The guarantee is strict net sinking of the chosen first-Conv mean and parameter convergence to a uniform-output equilibrium; it is not stepwise sinking, feature collapse, or an RL-CIFAR trajectory theorem.

## 1. Smooth frozen field through the equilibrium is established

The task-grouped EMA weights in geometry Section 2 are correct: the current task contributes r phases and every past task contributes H phases. With one binary label per task, each fixed-coordinate quotient is exactly

    s(mu-M_Y) / [|s| sqrt(1+mu^2-2mu R_Y)+epsilon].

The nonzero sensitivity reserve and |mu|<1 make the RMS uniformly positive on the local neighborhood. Consequently all finite-order parameter derivatives of the integrand are uniformly bounded and expectation can be differentiated. The coefficient's removable value at mu=0 is correctly computed as

    [epsilon+|s|(1-omega_r)]/(epsilon+|s|)^2,
    omega_r = sum_l A_l B_l.

The positive diagonal representation of the block-averaged field, its smoothness, and the positive lower bound on A=Dz·V are therefore proved, including at z=0. They do not rely on dividing by a vanishing mean gradient without justification. Task reuse is handled by independent task blocks, not by pretending the phases carry fresh labels.

## 2. Smooth endpoint map and full-dimensional initial region

W=V/A is a smooth field with Dz·W=1. Along its flow, z increases exactly by the flow time. The definition pi(theta)=Psi_{-z(theta)}(theta) is consequently valid in a sufficiently small reference neighborhood. Smooth dependence on initial state and time proves smoothness of pi, including across z=0.

The flow group identity gives Dpi·W=Dpi·V=0. The endpoint mean margin is the integral of Dm·W along the flow segment; its lower bound c_m z(theta) has the correct sign and integration limits.

Because the level set z=0 is regular, the map (p,z)↦Psi_z(p), with p on that level set, has local inverse (pi,z). An open relative set of p and an open nonzero interval of z therefore produce an open subset of the entire raw parameter space. This does not impose weight equalities or a tied-amplitude manifold. An arbitrary sufficiently small common head component is also allowed. Its subsequent dynamical constancy follows from opposite softmax gradients and equal optimizer initialization, rather than from freezing a raw parameter.

## 3. Observable averaging correctly controls nonlinear pi and z²

The observable proof applies the Poisson construction to the new reward Dphi(theta)·F(theta,X); it does not multiply a pre-existing convergent vector error by a moving derivative. This is the essential justified step.

The derivative bounds A_i and B_i give the stated reward Lipschitz/memory constants, and the task-level Taylor remainder is bounded by B_i B² delta_k²/2. The single task displacement already includes H phases, so there is no missing additional first-order phase term. The base tracking bound includes moving raw parameters, missing initial prehistory, global bias correction, and the extra within-task factor H.

The Poisson corrector parameter modulus x log(1/x), the weighted summation-by-parts bound, and the square-summable martingale series are sufficient for almost-sure convergence of each observable remainder. Power steps with exponent p>1/2 make both sum delta_k² and sum delta_k² log(1/delta_k) finite. The displayed simultaneous probability budget follows from coordinatewise martingale maximal bounds and a union bound; no independence among observables is assumed.

The base-history decomposition is adapted correctly: theta_k is known before task innovation Xi_k. The current task's label is not conditionally averaged again during its reused phases.

## 4. Localization and no exit include every optimizer phase

Use smooth coefficient/observable extensions that agree on a neighborhood of the closure of the outer tube. Such extensions exist because the flow chart and fixed CNN branch are available on a slightly larger neighborhood. The globally bounded auxiliary process is only a coupling device; it does not modify the actual Adam algorithm before exit.

On the simultaneous observable-error event, every task boundary that is still in the outer chart satisfies

    ||pi(theta_k)-theta_*||_infinity < r1,
    |z(theta_k)| < Z1,

and hence belongs to the middle tube. The middle tube is compactly contained in the outer tube, so its positive distance d_* to the complement is legitimate. From such a boundary the universal Adam quotient bound gives at most H K delta0 of motion over the entire next task. The strict inequality H K delta0<d_* rules out both an intermediate-phase exit and an endpoint jump across the boundary. This supports an ordinary induction/first-exit argument, without presuming stability of the original CNN.

The observable identities continue to apply at each next task boundary because all its phases stayed in the agreement region. This avoids a potential circularity in using a task-boundary barrier alone. Once nonexit is proved, the auxiliary and original processes coincide forever on the probability event.

## 5. Convergent remainders give actual parameter convergence

On the no-exit event, let S_K be the accumulated nonnegative drift in the z² identity. Since z²≥0 and the remainder converges, S_K is bounded above and monotone, hence converges. The identity then makes z_k² converge. Its limit must be zero: a positive limit would force S_K to diverge by the uniform lower drift bound and sum delta_k=infinity.

The pi identity and its convergent remainder imply pi(theta_k)→p_infinity. The chart inverse theta_k=Psi_{z_k}(pi(theta_k)) then gives theta_k→p_infinity, including the tangential directions along the equilibrium manifold. The bound on within-task motion tends to zero, so convergence holds at every optimizer step, not merely task boundaries. This is stronger than a conclusion that the distance to the equilibrium set tends to zero.

## 6. Strict mean margin and learning-rate divergence

The initial flow slice has z(theta0)>z_L>0, giving the uniform deterministic margin

    m(theta0)-m(pi(theta0)) >= c_m z_L.

The coordinatewise terminal pi error bounds the mean error by ||u||_1 epsilon_pi because the chosen first-Conv preactivation mean is exactly affine. The stated strict budget therefore yields

    m(theta_infinity) < m(theta0)-c_m z_L/2.

No bound on the total absolute variation of the stochastic parameter path is used or needed. Actual stochastic increments and z may change sign during convergence.

For the specified deterministic rate eta_(k,r)=delta0(k+1)^(-p), repeated H times, the actual total rate is H sum delta_k=infinity, while H sum delta_k²<infinity. Unlike an amplitude-dependent damping scheme, there is no missing lower-bound argument for the learning-rate factor.

The constants can be chosen uniformly on a compact closure strictly inside the initial open slice, and then apply to its nonempty interior. The correct probability quantifier is: **one common choice of constants and delta0 gives success probability at least 1-alpha for each allowed initial state**. This does not establish one common label-history event on which all uncountably many initial states succeed simultaneously. The latter stronger statement is neither needed nor proved; the integrated main statement should keep the former interpretation explicit.

## 7. Connection to the native all-bias capacity reference

The native reference construction supplies exactly the geometry's missing architectural conditions for a 32-by-32 positive image, two Conv5/pad2 layers of widths 16/16, FC100/100, and a binary head. Every raw sensitivity is nonzero: interior selected paths cover every padded spatial coefficient, all FC inputs/outgoing paths are positive, and output-bias derivatives are +/-1/2. Positive hidden biases are permitted. Independent raw perturbations preserve these strict conditions.

At head rows +/-alpha d with d>0, the N=1 full NTK has the exact all-raw formula

    K=(||h||²+1)I_2+alpha²||grad_hidden(d·h)||² vv^T.

Its derivative in the actual first-mean direction is positive definite, uniformly as alpha tends to zero, because the free output-weight block contributes 2<h,D_u h>I_2. The remaining scalar coefficient is nonnegative by the positive fixed-branch polynomial structure, and all hidden bias Jacobian blocks are included. The literal self network obeys the same formula using its own features and hidden parameters. Both strict capacity certificates persist in a sufficiently small raw open neighborhood, which can be intersected with the flow tubes.

Changing the initial output-bias contrast to cancel the full z does not change either NTK or its derivative. A fixed alpha>0 gives the sensitivity/head-contrast reserve required for smoothness and the endpoint mean margin. The capacity sign survives alpha=0, but the nonzero hidden sensitivities do not; the theorem correctly uses alpha>0. The literal self network's CE contrast need not vanish and its CE drift is not used. What agrees is the actual full-network mean decrease and the decreasing-mean direction of the original full/literal-self capacities.

## 8. Final scope

The combined proof gives a nonempty full-dimensional open family of native, all-bias CNN initial states for which ordinary all-raw Adam, with retained moments and H-fold reused iid uniform binary labels, converges to zero output contrast and has a strict terminal first-Conv mean decrease with probability at least 1-alpha, for sufficiently small positive power-decaying steps.

The restrictions N=1, binary labels, full-batch finite H reuse, a local fixed-routing region, and small decaying learning rates remain material. The constants may be extremely conservative for small epsilon or large H/parameter dimension. Hidden units remain active in the constructed region; the theorem is not about ReLU death. Within these stated limits, the three companion arguments connect without a missing stability, endpoint, or capacity-self step.
