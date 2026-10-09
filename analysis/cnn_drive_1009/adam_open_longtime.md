# Native all-bias CNN: an open-set, actual-Adam long-time self-direction theorem

2026-10-09. This integrates the [native reference](adam_native_bias_reference.md), [flow geometry](adam_open_geometry.md), and [observable averaging](adam_observable_averaging.md). It removes exact equal-weight initialization, 1x1-only kernels, absent biases, and the amplitude-dependent learning-rate factor from the preceding long-time Adam construction. It retains one fixed image, binary labels, a local active/winner region, and a sufficiently small power-decaying learning rate.

## 1. Precise result

Consider a fixed strictly positive RGB image and a CNN with two native 5x5 shared convolutions, ReLU and MaxPool, two hidden FC/ReLU layers, and a free binary output head. Every ordinary bias is present and every raw weight/bias is trained. The structural dimensions may be the RL-CIFAR hidden architecture, namely RGB32x32, Conv16/16, FC100/100; only the binary output size and the one-image task distribution differ in this construction.

There exists a nonempty **open subset of the entire raw parameter space**, with nonproportional Conv channel maps allowed, such that the following holds. For every prescribed alpha in (0,1), one can choose eta0>0, uniformly for an initial slice compactly contained in that open set, so that ordinary Adam with

    beta1=.9, beta2=.999, epsilon>0,
    eta_(k,r)=delta_k=eta0/(k+1)^p,   1/2<p<=1,

has the conclusions below with probability at least 1-alpha. Each task draws a new independent uniform binary label, uses that same label for H updates (any fixed finite H), and carries Adam moments and global bias-correction time into the next task. Moments start at zero only once, at the beginning of the whole process.

On the success event:

1. All raw parameters converge to a finite limiting state theta_infinity whose two logits agree.
2. The selected first-Conv mean obeys a uniform strict bound

       m(theta_infinity) <= m(theta0)-c_m z_L/2 < m(theta0).

3. At every finite iterate and at the limit, the original full and literal channel-isolated NTK capacity derivatives are strictly positive in the mean-increasing direction. The actual cumulative mean decrease therefore agrees with descent in the original capacity self direction.
4. The strict ReLU and MaxPool routing margins persist. Hidden activations remain positive; this theorem does not claim neuron death or collapse of the hidden representation.
5. The actual rates satisfy sum_t eta_t=infinity and sum_t eta_t^2<infinity. There is no projection, head refit, parameter freezing, moment reset, or state-dependent damping in the actual optimizer.

The probability statement holds for each initial point with the same deterministic constants/rate bound on a fixed initial slice. It does not assert simultaneous success for every point of an uncountable initial set on the same label path. Individual Adam steps and the realized logit contrast may change sign. An unconditional expected terminal displacement, including the failure event, is not inferred.

## 2. Why the assumptions are nonempty and do not assume the answer

Choose strictly positive hidden weights and small positive hidden biases. A positive image increasing in both spatial coordinates, initially center-only Conv kernels followed by sufficiently small positive off-center coefficients, gives strict ReLU/pool margins. Interior selected paths ensure every raw logit-contrast sensitivity is nonzero. Both hidden FC matrices are positive. The construction works with all 25 Conv offsets nonzero and admits independent perturbations of every raw coordinate.

Choose a positive output contrast vector d and head rows +/-d at a reference. Write the last hidden vector as h, and set the initial output-bias contrast to -d^T h. This produces a reference theta_* with equal logits while preserving positive hidden sensitivities. The output bias is freely trained after initialization. An open positive-contrast slice near theta_* supplies the actual initial states; exact equality of head rows, filters, or biases is not imposed on that slice.

With z=(f_+-f_-)/2 and s=grad z, binary CE has the exact raw-coordinate form

    g_j(theta,Y)=s_j(theta)[tanh z(theta)-Y],  Y in {-1,+1}.

The target mean m=u^T theta is affine. Its nonzero u coordinates are the actual mean input patch and its bias coordinate 1. They are positive, and the corresponding s_j are strictly positive in the reference neighborhood. All these conditions concern current images, weights, routing and derivatives at a constructed reference. Future containment and the final update sign are proved below, rather than assumed.

## 3. The original capacity self survives output-bias cancellation

At the reference head +/-d, let psi=d^T h and v=(1,-1). With xi denoting **all** hidden raw parameters, including biases, the full raw NTK satisfies

    K=(||h||^2+1)I_2+||grad_xi psi||^2 vv^T,
    D_u K=2<h,D_u h>I_2
             +2<grad_xi psi,D_u grad_xi psi> vv^T.

Inside the strict active/pool branch, h is a polynomial with nonnegative coefficients in positive hidden weights and nonnegative biases. Thus the second inner product is nonnegative, while the first is strictly positive. Consequently D_u K is positive definite.

Delete the other first-Conv channels and their matching second-Conv input columns, keeping all remaining values and all biases without refitting. The same formulas apply to this literal self model using its own hidden vector and raw coordinates, and give a positive definite derivative there too. Output-bias **values** appear in neither derivative.

Both signs persist in a full-dimensional neighborhood by continuity, or by the finite Jacobian-distance bound in the reference note. Hence for every finite lambda>0,

    D_u [1/2 log det(lambda I+K_full)]>0,
    D_u [1/2 log det(lambda I+K_self)]>0.

The retained output bias generally does not cancel the self network's logits. The self network's CE gradient may therefore have a different sign; that gradient is not the definition of the original capacity self and is not used in this theorem.

## 4. The stationary field and its endpoint are smooth local geometry

At any fixed state in a small neighborhood, every raw sensitivity s_j is nonzero and |tanh z| is bounded strictly below one. The frozen gradients, repeated H times per independent task, therefore have a positive lower bound in absolute value for each coordinate. Grouping EMA weights by task gives the exact phase-stationary field

    Fbar(theta)=tanh z(theta) V(theta),
    V_j=s_j sum_(r=1)^H c_r(s_j,tanh z),
    epsilon/(epsilon+G)^2 <= c_r <=1/epsilon.

The coefficients are smooth through z=0. They retain numerator/denominator correlation; this is not Adam applied to the expected gradient. Their exact removable value is

    c_r(s,0)=[epsilon+|s|(1-omega_r)]/(epsilon+|s|)^2,

where omega_r is the inner product of the first/second moment weights after grouping by task. The companion geometry note gives its closed form.

Define A=Dz V and W=V/A. Output-bias sensitivities +/-1/2 give a positive uniform lower bound for A, while the positive target-coordinate cone gives Dm W>=c_m>0. Since Dz W=1, the smooth W-flow Psi satisfies z(Psi_t theta)=z(theta)+t. Define the local endpoint map

    pi(theta)=Psi_[-z(theta)](theta).

It lies on {z=0}, satisfies Dpi V=0, and obeys

    m(theta)-m(pi(theta)) >= c_m z(theta)  for z(theta)>0.

This map is a mathematical description of the averaged flow. The optimizer never applies it. Existence and its derivative bounds are obtained from an ordinary finite-time smooth ODE on a known compact neighborhood, independently of the future Adam trajectory.

## 5. Actual history, nonlinear observables, and permanent containment

Apply observable averaging to all ambient components of pi and to z^2. With theta_k denoting task boundaries, it gives convergent remainder processes and the exact cumulative identities

    pi(theta_K)=pi(theta0)+R_K^pi,
    z(theta_K)^2=z(theta0)^2
       -sum_(k<K)delta_k [2z(theta_k)tanh z(theta_k) A(theta_k)]
       +R_K^z.

For any positive budgets e_pi,e_z and failure probability alpha, a sufficiently small positive eta0 makes the simultaneous bounds

    sup_K ||R_K^pi||_infinity<=e_pi,
    sup_K |R_K^z|<=e_z

hold with probability at least 1-alpha. The bounds include moment initialization, global bias correction, label reuse, the changing network, and Taylor remainders of the nonlinear observables. The Poisson construction is applied to the new rewards Dphi F; an old convergent remainder is not simply multiplied by a varying Dphi.

Construct nested flow tubes using coordinates (pi,z). The initial slice has pi-radius r0 and z_L<z<z_U. A compact middle tube has radius r1>r0 and |z|<=Z1>z_U. It lies strictly inside an outer smooth/routing/capacity tube, at positive distance d_* from its complement. Choose

    e_pi<min(r1-r0, c_m z_L/(2||u||_1)),
    e_z<Z1^2-z_U^2,
    H K eta0<d_*,   K=73.

All right sides are known strictly positive geometric quantities. The stochastic bounds tend to zero with eta0, so the requirements have positive solutions and do not assume containment.

The observable identities keep every task boundary in the middle tube: pi has no averaged drift, and the z^2 drift is nonnegative. The universal raw Adam bound limits every intervening task's displacement to H K eta0, preventing an intermediate-step exit too. This induction proves permanent containment on the error event. Any auxiliary coefficient/observable extension used to establish the bounds therefore agrees forever with the actual unmodified CNN on that event.

The nonnegative z^2 drift sum is then bounded. Remainder convergence makes z^2 converge; a positive limit would contradict sum delta_k=infinity. Thus z->0. The pi coordinates converge, so the whole parameter vector converges to theta_infinity=pi_infinity. Finally

    m(theta0)-m(theta_infinity)
       >=c_m z_L-||u||_1 e_pi>c_m z_L/2.

No estimate of total absolute stochastic path length is used. It need not be finite.

## 6. Finite derivative and probability constants

The observable theorem states all bounds in terms of finite gradient/Lipschitz constants, task length, optimizer parameters, and first/second derivatives of its observables. These constants can be constructed from the local flow without referencing an unknown future trajectory.

For example, suppose all flow segments used by pi stay in a region with

    ||W||_infinity<=M, ||DW||_(infinity operator)<=L,
    max_i sum_(j,l)|partial_j partial_l W_i|<=B,
    |z|<=Z, ||Dz||_1<=S, sum_(j,l)|partial_j partial_l z|<=T.

Flow derivative bounds give, componentwise,

    ||Dpi_i||_1 <= exp(LZ)+M S,
    sum|D^2 pi_i| <= Z B exp(2LZ)+2L exp(LZ)S+L M S^2+M T.

For z^2 the corresponding bounds are 2ZS and 2S^2+2ZT. Smooth cutoff extensions on a slightly larger chart add explicit product-rule terms from the cutoff derivatives and bounded observable values. Centering pi_i by the fixed reference coordinate makes those value bounds smaller. The observable averaging theorem then supplies a finite, dimension-dependent error threshold tending to zero with eta0.

This proves positive-rate nonemptiness analytically. The finite example below has not been assigned a numerical long-time probability-certified rate; no ordinary training rate or practical convergence time is claimed from it.

## 7. Native all-parameter identity check

[verify_adam_open_native.py](verify_adam_open_native.py) uses RGB12x12, two native Conv5/Pool blocks, two hidden FC/ReLU layers, and all biases: 331 independent raw parameters. Its channels are independently weighted and spatially nonproportional. It constructs a positive reference, cancels the output contrast by initial bias values, and perturbs all raw coordinates, including the common class-head component.

The saved [results](../../results/cnn_drive_1009/adam_open_native.json) check the full and literal-self raw NTK formulas, both binary-label CE gradient factorizations, strict routing and sensitivity margins, and the exact task-phase linearization formula. The full/self reference derivative lower bounds are approximately 6.58536 and 4.20821; finite perturbation error bounds are approximately 1.08e-4 and 8.96e-5. The largest raw-formula discrepancy is below 3.56e-15.

The local endpoint mean-loss coefficient is about 2.01556 per unit small positive logit contrast. This is a derivative of the averaged endpoint geometry, not a measured terminal displacement. The verifier averages task phases in its reported normal coefficient; using the block-sum clock multiplies that coefficient by H and leaves the endpoint ratio unchanged. Its finite numerical margins are float64 checks, while nonemptiness of the open family follows from the analytic strict reference and continuity arguments.

## 8. Remaining boundary

The result covers full raw asymmetry in an open initial set, native finite spatial kernels, hidden FC layers, all biases, shared MaxPool backpropagation, ordinary retained Adam moments, and actual infinite task repetition. It does not impose a sign on every realized update or an invariant equal-weight family.

It remains a one-image, binary-label, local fixed-routing, decaying-rate theorem. It does not prove the long-time behavior of many-image, ten-class, shuffled-minibatch, constant-rate RL-CIFAR training, or a negative bulk mean, finite-time gate death, or unconditional expectation after including failed exits. No other activation function is introduced.

Independent audits: [integrated geometry and actual-process argument](adam_open_longtime_review.md), [observable averaging](adam_observable_averaging_review.md), and [native all-bias reference](adam_native_bias_reference_review.md). All three passed within the stated scope.
