# Open full-parameter binary CNN family: scalar-output flow geometry for long-time Adam

2026-10-09. This note supplies the geometry and endpoint argument for a full-parameter open-set extension. The [observable theorem](adam_observable_averaging.md) supplies generalized stochastic averaging for the observables used below; [the integrated theorem](adam_open_longtime.md) includes the native reference and capacity-self connection. No parameter is tied, projected, reset, or frozen by the proposed optimizer. The endpoint map called pi is only an analysis map; the algorithm never applies it.

The stronger local construction uses one positive input image, a finite shared Conv/ReLU/MaxPool network with a free binary head and all ordinary biases, and one independently uniform binary label per task. That label is reused in H full-batch updates. Every raw weight and bias is trained with ordinary Adam, common beta1=.9, beta2=.999, epsilon>0, and a deterministic taskwise decaying scalar learning rate. The claim is strict long-time net decrease of a selected first-Conv mean on a high-probability no-exit event. It is not collapse of every feature, and does not assert every stochastic step has the same sign.

## 1. Noncircular local construction

For binary logits f_+(theta), f_-(theta), define the scalar contrast

    z(theta) = [f_+(theta)-f_-(theta)]/2,
    s(theta) = grad_theta z(theta).

For Y in {-1,+1}, cross entropy, up to the irrelevant common-logit component, is

    loss(theta,Y)=log(2 cosh z(theta))-Y z(theta),
    g_j(theta,Y)=s_j(theta)[tanh z(theta)-Y].          (1.1)

Choose a reference state theta_* with the following finite, checkable properties:

1. Hidden weights and head contrast entries `(V_+-V_-)/2` are strictly positive. Hidden preactivations are strictly positive and all MaxPool winners are unique. The chosen input and routing give every raw Conv/FC weight at least one selected path with a strictly positive incoming activation. Valid convolutions on a strictly positive image are a simple way to ensure this, including every finite spatial-kernel offset.
2. All ordinary biases are trainable. Hidden biases can be chosen to ensure the strict activation margins. Adjust only the **initial value** of the output-bias contrast so that z(theta_*)=0. This cancellation does not change hidden activations, hidden Jacobians, or any NTK block. The output bias remains freely trained afterward.
3. Every raw sensitivity s_j(theta_*) is nonzero. In particular, sensitivities of hidden weights and hidden biases are positive, while those of the two output-head rows have opposite signs. Output-bias sensitivities are +/-1/2.

These are conditions on an explicitly constructible state, not on a future update sign. They hold in a neighborhood by strictness and continuity. Different hidden channels and their spatial maps need not be proportional. All raw parameters may vary independently in this neighborhood; the common binary-head component is not constrained at initialization.

The actual CE gradients of the two head rows/biases are opposite. With zero initial Adam moments and equal optimizer hyperparameters their updates are opposite too. Thus their common component is dynamically constant as a consequence of softmax invariance. It is not a frozen parameter imposed on the optimizer, and allowing arbitrary initial common components is consistent with a full-dimensional open parameter set.

Fix a convex compact neighborhood O lying strictly inside these gate/winner conditions. Shrink it if needed to have, for constants known from O,

    |z(theta)|<=Z_*,     |tanh z(theta)|<=m_*<1,
    0<sigma_j<=|s_j(theta)|<=S_j,

with each s_j having its fixed reference sign. The reference state is interior. The logit contrast and sensitivities are smooth finite polynomials on this branch. If literal channel-isolated capacity is also used, its own gates/winners and strict capacity condition must be included in the definition of O; that is a separate local certificate.

## 2. Smooth stationary Adam field at z=0

At a fixed state, the same scalar task label Y_k is used in all H frozen gradients. At phase r=1,...,H, group the stationary moment weights by task:

    A_0=1-beta1^r,
    A_l=beta1^[r+(l-1)H](1-beta1^H), l>=1,
    B_0=1-beta2^r,
    B_l=beta2^[r+(l-1)H](1-beta2^H), l>=1.

Both positive sequences sum to one. Put

    M_Y=sum_l A_l Y_(k-l),     R_Y=sum_l B_l Y_(k-l),
    mu=tanh z.

For one raw coordinate with sensitivity s!=0, its frozen stationary quotient is exactly

    q_r^*(s,mu)
      =s(mu-M_Y)/[|s| sqrt(1+mu^2-2mu R_Y)+epsilon].  (2.1)

Because |mu|<=m_*<1, the expression under the square root is at least `(1-m_*)^2`. Every frozen coordinate gradient also has absolute value at least `sigma_j(1-m_*)`. Consequently the stationary RMS is uniformly bounded away from zero throughout O. This is a substantive advantage of the one-image binary construction; for multiple images a weighted label sum can vanish, so this particular smoothness argument must not simply be reused.

The integrand and all finite-order parameter derivatives are uniformly bounded on O. Differentiation under the expectation is therefore legitimate. Complementing all past labels shows that E q_r^*(s,mu) is odd in mu. Thus

    E q_r^*(s,mu)=s mu c_r(s,mu)                     (2.2)

has a smooth extension at mu=0. The extension is explicit. Let

    omega_r=sum_l A_l B_l
      =(1-beta1^r)(1-beta2^r)
        +(beta1 beta2)^r (1-beta1^H)(1-beta2^H)
                              /[1-(beta1 beta2)^H].

Then

    c_r(s,0)=[epsilon+|s|(1-omega_r)]/(epsilon+|s|)^2>0.  (2.3)

To obtain (2.3), differentiate (2.1) at mu=0 and use E[M_Y R_Y]=omega_r. The absolute value of s is smooth on each nonzero-sign coordinate neighborhood.

The task-block odd-monotonicity argument gives throughout O

    epsilon/[epsilon+S_j(1+m_*)]^2
       <= c_r(s_j,mu) <=1/epsilon.                  (2.4)

This includes the removable value at mu=0 by continuity. Summing phases, the stationary averaged **block** field for a constant scalar learning-rate factor h=1 is

    Fbar(theta)=tanh z(theta) V(theta),
    V_j(theta)=s_j(theta) sum_(r=1)^H c_r(s_j(theta),tanh z(theta)).  (2.5)

V is smooth, its j-th sign is the fixed sign of s_j, and

    A(theta):=Dz(theta) V(theta)
             =sum_j s_j(theta)^2 sum_r c_r(...) >0. (2.6)

This is an exact coordinatewise-positive metric representation of the stationary averaged Adam field, not Adam applied to an expected gradient. Task reuse and retained stationary moments are included in (2.1).

A useful entirely geometric uniform bound is obtained by putting

    c_min = min_j epsilon/[epsilon+S_j(1+m_*)]^2,
    c_max = 1/epsilon.

Then

    H c_min sum_j sigma_j^2 <= A(theta)
                             <= H c_max sum_j S_j^2.  (2.7)

Finite initialization and the actual moving moment history are not presumed stationary; they are handled by the separate averaging theorem.

## 3. The mean direction and an explicitly positive endpoint margin

Let m(theta) be the true first-Conv mean preactivation of a selected channel, averaged over all spatial locations of the fixed image. It is affine in that channel's augmented filter, including its trainable bias:

    m(theta)=u^T theta + constant.

The nonzero entries of u are the nonnegative mean input-patch entries and the bias coordinate 1. In the construction, all corresponding sensitivities s_j are positive. Therefore

    Dm V = sum_j u_j V_j >0                         (3.1)

throughout a sufficiently small O. This strict sign follows from the verified local cone and is not assumed for a future path outside O.

Normalize the smooth field by its effect on z:

    W(theta)=V(theta)/A(theta).

Then

    Dz W=1,
    ||W||_infinity<=M<infinity,
    Dm W>=c_m>0                                    (3.2)

on O, with finite derivative bounds. One permissible explicit lower bound is

    c_m = [c_min sum_(j in target) u_j sigma_j]
          /[c_max sum_j S_j^2] >0.                 (3.3)

All constants depend on the finite reference neighborhood and known optimizer parameters. They do not depend on an actual future label realization.

## 4. Smooth endpoint map from an ordinary local flow

Let Psi_tau(theta) be the local flow solving

    d Psi_tau(theta)/d tau=W(Psi_tau(theta)),
    Psi_0(theta)=theta.

As long as it remains in O,

    z(Psi_tau(theta))=z(theta)+tau.                  (4.1)

If theta has distance at least d_O from the boundary of O and `M |z(theta)|<d_O`, this flow is defined for all tau between 0 and -z(theta). Define its endpoint

    pi(theta)=Psi_[-z(theta)](theta).                (4.2)

Then z(pi(theta))=0. Standard smooth dependence of a finite-time smooth ODE gives a smooth pi on the corresponding open neighborhood, with bounded first and second derivatives on every compact sub-neighborhood. This invokes only the constructed vector field, not the unknown Adam trajectory.

The flow group identity and (4.1) imply

    pi(Psi_tau(theta))=pi(theta),
    Dpi W=0,
    Dpi V=0,
    Dpi Fbar=0.                                   (4.3)

Thus pi records which local zero-contrast state the averaged trajectory approaches. It is not an orthogonal projection and is not an optimizer action.

For z(theta)>0, the affine mean's endpoint decrease satisfies

    m(theta)-m(pi(theta))
      = integral_[-z(theta)]^0 DmW(Psi_tau(theta)) d tau
      >= c_m z(theta)>0.                           (4.4)

The averaged ODE is theta_dot=-tanh(z) V. Its scalar contrast obeys

    z_dot=-A(theta) tanh z.

Starting with z>0, its contrast stays positive and approaches zero. Its parameter curve is exactly the W-flow from z(theta) down to zero; only its time parameter changes. This is why the endpoint map has a strict first-mean margin even though the limiting hidden features are generally nonzero.

## 5. Full-dimensional initial slice and nested flow tubes

For a concrete nonempty local setup, choose a ball `B_infinity(theta_*,R)` contained in the strict branch neighborhood O. Use a compact small part P_2 of the regular level surface `z=0` around theta_*. Regularity follows already from the nonzero output-bias sensitivity. Its radius may be chosen less than R/4. Choose Z_2>0 small enough that `M Z_2<R/4`. The W-flow of every p in P_2 for |z|<=Z_2 stays in O.

The map

    T(p,z)=Psi_z(p)

is a smooth local coordinate map: its inverse is `(pi(theta), z(theta))`, by (4.1)-(4.3). Its image of the relative interior of P_2 times `(-Z_2,Z_2)` is open in the entire raw parameter space, not just in a symmetric submanifold.

For clarity, let P(r) denote a sufficiently small relative ball of radius r about theta_* in the level surface z=0, using the ambient infinity distance. Choose

    0<r0<r1<r2,
    0<z_L<z_U<Z_1<Z_2,

inside the flow-chart construction. Define

    T_outer={Psi_z(p): p in P(r2), |z|<Z_2},
    T_middle={Psi_z(p): p in closure P(r1), |z|<=Z_1},
    T_initial={Psi_z(p): p in P(r0), z_L<z<z_U}.

Shrink the radii, if necessary, so the middle tube is compactly contained in the outer chart. Then

    d_* = dist_infinity(T_middle, complement T_outer)>0.  (5.1)

These are sets computed from a reference ODE and parameter margins. The initial set is nonempty and full-dimensional. No equality among Conv weights, head entries, or biases is imposed. All independently perturbed raw coordinates are allowed as long as they lie in this open slice. Every theta0 in it has the uniform deterministic endpoint margin

    m(theta0)-m(pi(theta0)) >= c_m z_L.              (5.2)

## 6. Precisely what is required from observable averaging

Use actual ordinary Adam with taskwise learning rate

    eta_(k,r)=delta_k=delta0(k+1)^(-p),
    1/2<p<=1,

at all H phases, with zero initial moments once and no task-boundary resets. More generally a common smooth positive scalar h can be used if it is bounded below on the tube; h cancels in W and does not change the endpoint geometry. Taking h=1 is sufficient here.

The needed companion result is generalized averaging for the finitely many observables given by coordinates of pi and by z^2. Up to compact localization it must provide

    pi(theta_(KH)) = pi(theta0)+R_K^pi,              (6.1)

    z(theta_(KH))^2
      =z(theta0)^2-sum_(k<K)delta_k L_z(theta_(kH))+R_K^z,
    L_z(theta)=2 z(theta) tanh z(theta) A(theta),     (6.2)

where both remainder processes converge almost surely. Since `Dpi Fbar=0`, no unaccounted drift is omitted from (6.1). For the compact tube,

    L_z(theta)>=c_z z(theta)^2,
    c_z=2 A_min [tanh Z_2/Z_2]>0.                   (6.3)

It must also provide, for any fixed failure probability alpha>0 and sufficiently small positive delta0, the simultaneous event

    sup_K ||R_K^pi||_infinity <= epsilon_pi,
    sup_K |R_K^z| <= epsilon_z                      (6.4)

with probability at least 1-alpha. A union bound with alpha/2 allocated to each observable group suffices. Finite-coordinate vector pi is allowed. Taylor terms from the nonlinear observables are included in the remainder; they are not silently discarded.

The branch maps and coefficients may be smoothly extended outside an outer compact neighborhood as a coupling device, allowing (6.4) to be stated globally for an auxiliary process. The actual Adam algorithm is not extended, projected, or clipped. Agreement is used only until an exit, and the following argument rules out that exit on (6.4). The required observable averaging and its explicit constants are owned by the companion proof; no stationarity assumption on actual moments is made here.

## 7. Finite budgets ensuring no exit at every optimizer phase

Choose the deterministic budgets to satisfy

    epsilon_pi < r1-r0,
    z_U^2+epsilon_z < Z_1^2,
    epsilon_pi < c_m z_L/(2 ||u||_1).              (7.1)

Choose delta0 positive and small enough for the averaging event (6.4) and also

    H K delta0 < d_*.                              (7.2)

All right sides are strictly positive known quantities, so these requirements are nonempty once the companion error budgets tend to zero with delta0. They can hold uniformly for every initial theta0 in a slightly smaller compact initial slice; hence this is an open-family assertion, not a single specially initialized point.

Before a first outer-chart exit, (6.1) gives

    ||pi(theta_(KH))-theta_*||_infinity
       <r0+epsilon_pi<r1,

and (6.2), using nonnegative L_z, gives

    |z(theta_(KH))|<=sqrt(z_U^2+epsilon_z)<Z_1.

Thus every preceding task boundary lies in the middle tube. The universal Adam bound gives `||theta_(t+1)-theta_t||_infinity<=K delta_k`, regardless of the realized gradient or its sign. Throughout the following task the total parameter displacement from that boundary is at most H K delta0<d_*. It cannot leave the outer tube, even at an intermediate optimizer phase or by jumping across its boundary. This contradicts any first exit.

Therefore the auxiliary and actual full raw-parameter Adam processes coincide forever on the event (6.4). All gates, winners, nonzero sensitivity bounds, and local geometry then follow from membership in the constructed tube. They were not assumed along an unknown future trajectory.

## 8. Limit and strict actual mean decrease

On the no-exit event, intersected with the probability-one event that the remainders converge, the nonnegative sums in (6.2) are bounded and hence converge. Therefore z(theta_(KH))^2 converges. If its limit were positive, (6.3) and sum delta_k=infinity would contradict boundedness of those sums. Consequently

    z(theta_t)->0.

Equation (6.1) gives pi(theta_(KH))->p_infinity in the inner part of the level surface. Since

    theta_(KH)=Psi_[z(theta_(KH))](pi(theta_(KH))),

smoothness of the flow implies theta_(KH)->p_infinity. Within-task displacements vanish as delta_k->0, so every optimizer-step parameter sequence converges to the same theta_infinity=p_infinity. This limit has zero contrast and generally positive, nonzero hidden activations.

From (6.4),

    ||theta_infinity-pi(theta0)||_infinity<=epsilon_pi.

The selected first-Conv mean is affine, so

    m(theta_infinity)
      <=m(pi(theta0))+||u||_1 epsilon_pi
      <m(theta0)-c_m z_L/2.                         (8.1)

This is strict long-time **net** sinking for every initial state in the open slice, on an event of probability at least 1-alpha. Realized z and finite Adam increments may change sign; the proof does not require a positive first moment or one-step sinking throughout.

Because the actual rate here is simply delta_k repeated H times,

    sum_t eta_t = H sum_k delta_k = infinity,
    sum_t eta_t^2 = H sum_k delta_k^2 < infinity.

No additional near-zero amplitude argument is needed, and no hidden unit or weight is frozen.

## 9. Relation to original capacity-self and scope

The geometric result uses the actual binary CE gradient and its stationary-averaged Adam field. It does not rename this field as a self-capacity derivative. If a separate native-CNN certificate supplies strict positive derivatives of the original full and literal channel-isolated NTK capacities throughout O in the chosen first-mean direction, the actual net decrease (8.1) agrees with their negative-gradient direction. Those strict capacity conditions can be intersected with the present strict local geometry. Output-bias cancellation changes the logits but not the NTK or its parameter derivatives, so constructing z(theta_*)=0 does not itself destroy such a capacity certificate.

The isolated self network need not have its CE contrast canceled by the retained output bias. Its CE direction is not used in this argument; only its original capacity derivative is compared.

The main restrictions are one image per task, independently uniform binary labels, full-batch H-fold reuse, a local fixed-routing tube, and sufficiently small power-decaying ordinary-Adam steps. The existence of a smooth endpoint map and the mean margin are proved here without assuming future sign or containment. The stochastic theorem depends on the companion observable-averaging result in exactly the form (6.1)-(6.4). This note does not claim that arbitrary multi-image or ten-class RL-CIFAR trajectories satisfy these conditions.
