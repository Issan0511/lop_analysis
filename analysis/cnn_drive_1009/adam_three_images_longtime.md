# Three distinct images: native all-raw CNN and actual Adam long-time self direction

2026-10-09. This extends the [one-image theorem](adam_open_longtime.md) to three distinct images with three independent feature/output directions. It combines the [native reference](adam_three_images_reference.md), [finite mean-loss cone](adam_three_images_geometry.md), [vector-normal endpoint lemma](adam_three_images_equilibrium.md), and the existing [observable averaging theorem](adam_observable_averaging.md). ReLU, binary labels, full-batch task reuse and sufficiently small power-decaying rates remain explicit conditions.

## 1. Result

Fix a finite ridge lambda>0, a finite task length H>=1, and 1/2<p<=1. There exist three pairwise distinct, strictly positive RGB images and a nonempty full-dimensional open set of initial raw weights/biases for a native two-Conv5/ReLU/MaxPool, two-FC/ReLU, binary-head CNN with the following property.

The hidden architectural dimensions can be those of RL-CIFAR: RGB32x32, Conv16/16, FC100/100. All raw weights and all ordinary biases are freely trained. The image feature matrix and the logit half-contrast Jacobian both have row rank three throughout a local neighborhood. Channels overlap on all images and may be spatially nonproportional; no equality constraint ties trainable parameters.

For every alpha in (0,1), a sufficiently small positive eta0, common to a fixed initial slice compactly contained in that open set, makes ordinary Adam with beta1=.9, beta2=.999, epsilon>0, and

    eta_(k,r)=delta_k=eta0/(k+1)^p,  r=1,...,H,

satisfy the following with probability at least 1-alpha for each initial point. Each task draws three independent uniform binary labels, keeps that assignment for H full-batch updates, and carries Adam moments and global bias-correction time across tasks. Moments start at zero once. On the success event:

1. Every raw parameter converges to a finite limit; all three limiting class contrasts are zero.
2. The selected first-Conv mean has a uniform strictly negative net displacement: `m(theta_infinity)<=m(theta0)-Delta/2`, where Delta>0 is derived below from a fixed reference and a finite open cone.
3. The original full and literal channel-isolated all-raw NTK capacities both have positive derivatives in the true mean-increasing direction, for the specified ridge, at all iterates and the limit. Thus the actual net mean change agrees with descent in the original capacity self direction.
4. Strict ReLU/MaxPool margins remain valid, as do the rank, sensitivity and capacity margins. Hidden activations remain positive. This is not a neuron-death theorem.
5. Actual rates have divergent sum and convergent squared sum. The optimizer uses no projection, refit, frozen block, state-dependent damping or moment reset.

The statement is per initial point with common constants, not a single noise event guaranteeing every point of an uncountable initial set. It does not assert unconditional expected displacement after including failed exits or a fixed sign for every individual update.

## 2. A nonempty reference with three independent images

On a strict active/winner branch the last hidden vector is affine in input: `h(x)=Ax+c`. Positive native Conv kernels and positive FC layers can be chosen with rank A>=3: three distinct surviving spatial cells give an independent three-column minor, and sufficiently small positive remaining weights preserve it. Every raw hidden parameter has a positive path. Both full and literal-self networks have strict branches.

For a positive output contrast d, choose independent input perturbations v1,v2 satisfying `d^T A v_i=0`. The reference note gives explicit sparse formulas using three independent input columns. Set

    x1=x0+tau v1, x2=x0+tau v2, x3=x0-tau(v1+v2).

For sufficiently small fixed tau>0 the images are positive and distinct, with exactly the original mean input x0. Their hidden feature rows have rank three, while d^T h(x_n) is the same positive number for every image. A single common initial output-bias contrast cancels this number; consequently all three half-contrasts z_n are zero at a reference theta_*. The free head-weight block ensures `rank J_*=3`, where J=D_theta z. Biases are then trained normally.

For base-image sensitivities s_j*!=0, choose tau so that `|J_nj-s_j*|<|s_j*|/6`. Since the sum of three binary signs cannot be zero, at equilibrium every one of the eight label assignments satisfies `|g_j|>|s_j*|/6` for every raw coordinate. A small state neighborhood retains `|g_j|>|s_j*|/12`. This is compatible with rank three for every nonzero sufficiently small tau.

The target mean is the actual spatial/data average of the selected first-Conv preactivation: `m(theta)=u^T theta`, including its trainable bias. Since the three inputs average exactly to x0, u is unchanged from the base-image direction. Its supported entries are positive; the corresponding columns of J are positive.

At identical inputs the six-logit kernel and derivative are `11^T tensor K0` and `11^T tensor K0'`. The single-image positive reference gives K0' positive definite, both in full and literal-self models. At fixed lambda the capacity derivative is

    (3/2) trace[(lambda I2+3K0)^(-1) K0'] >0.

The inverse-ridge trace is continuous in the three inputs and all raw parameters. Both signs therefore survive a sufficiently small positive tau and a full-dimensional parameter neighborhood. This does not assert that the perturbed six-dimensional K' is positive definite or that one neighborhood works for every lambda approaching zero.

## 3. Actual stationary Adam field at a frozen state

Binary mean CE has the exact full-batch raw gradient

    g(theta,Y)=J(theta)^T[tanh z(theta)-Y]/3.

Its centered coordinate noise is symmetric under Y -> -Y. Group the infinite EMA weights by independent tasks; the H within-task copies remain one random object. The [task-block sign theorem](adam_task_averaging.md) yields

    Fbar(theta)=D(theta) J(theta)^T tanh z(theta)/3,

where D is a strictly positive diagonal matrix including the sum of H stationary phase coefficients. This retains correlation between Adam numerator and denominator.

To justify smoothness, for each coordinate first regard its mean mu and sensitivity row as independent arguments of the frozen expected quotient R(mu,s). The eight-assignment gradient floor remains uniform along the interpolation from mu=0 to its local value. Every history RMS is therefore bounded away from zero. Uniformly bounded derivatives permit differentiation under expectation. Symmetry makes R odd in mu, so

    R(mu,s)/mu = integral_0^1 partial_mu R(t mu,s) dt

extends smoothly through mu=0. The positive lower bound from the sign theorem persists there. Hence D is smooth with finite derivative bounds on the compact neighborhood. It is a fixed-state history expectation, not a coefficient inferred from the future path.

Let `K=J D J^T`. Full row rank gives K>=kappa I3 for some kappa>0 on a sufficiently small compact neighborhood.

## 4. Vector endpoint and a derived positive mean margin

For the averaged ODE `theta_dot=-Fbar(theta)`, use

    E(z)=sum_n log cosh(z_n).

Its derivative is exactly `-tanh(z)^T K tanh(z)/3`. On a small compact contrast region, this dominates a positive multiple of E. Thus z decays exponentially and the ODE speed is integrable. A finite path-length bound proves local containment and convergence to the regular equilibrium manifold M={z=0}.

The vector-normal endpoint lemma proves that the limit map pi is C2 on a smaller neighborhood. The proof uses local coordinates (a,z), a uniformly stable normal variational block, exponentially integrable tangential coupling, and uniform convergence of the first and second flow derivatives. It does not assume stochastic containment. The endpoint identities are

    pi|M=id,   Dpi(theta) Fbar(theta)=0,
    Dpi(theta_*)=I-Q_*,
    Q_*=D_* J_*^T (J_*D_*J_*^T)^(-1) J_*.

There is no assertion that Dpi D J^T vanishes throughout the neighborhood; in several normal dimensions that stronger identity is unnecessary.

Choose the initial direction `v_*=D_*u`. It changes only the target filter/bias block, weighted by the reference Adam metric. The linearized endpoint mean loss in this direction is

    b_* = u^T Q_* v_*
        = (J_*D_*u)^T (J_*D_*J_*^T)^(-1) (J_*D_*u) >0.

Strictness follows from positive target sensitivities and positive u on its support. This quadratic form is the deciding multi-image sign argument; it includes the coupling between all three outputs.

For `L_m=m-m o pi`, set ell=u^T Q_* and let C_m bound the entry-sum norm of its Hessian on a convex reference ball. Choose

    0<q<b_*/(4||ell||_1), M_v=||v_*||_infinity+q,
    0<t_L<t_U<b_*/(2 C_m M_v^2),

with the last condition omitted when C_m=0, and make the reference ball large enough to contain all `theta_*+t(v_*+e)` with t<=t_U, ||e||_infinity<=q. Equivalently shrink t_U so this ball is inside the proved smooth neighborhood. Taylor's theorem gives, for every such initial point with t in (t_L,t_U),

    m(theta0)-m(pi(theta0)) > b_* t/2 >= Delta,
    Delta=b_* t_L/2>0.

The cone is a union of open balls in the entire raw parameter space. Initial mean loss is derived from the ray and a finite Hessian bound, rather than assumed from an unknown terminal state. All constants refer to a known reference and local derivatives.

## 5. Transfer to the actual moving Adam process

Apply [observable averaging](adam_observable_averaging.md), sections 1-5, to all P ambient components of pi and to E(z). The generic theorem is C2-observable based; its one-dimensional application need not be reused. It gives convergent remainders and exact identities at task boundaries:

    pi(theta_K)=pi(theta0)+R_K^pi,
    E(z(theta_K))=E(z(theta0))
      -sum_(k<K) delta_k [tanh(z)^T K tanh(z)/3](theta_k)+R_K^E.

For any positive budgets e_pi,e_E, choosing eta0 sufficiently small makes `sup_K||R_K^pi||_infinity<=e_pi` and `sup_K|R_K^E|<=e_E` hold jointly with probability at least 1-alpha. The bounds include old moments, initial zeros, global bias correction, task reuse, all moving parameters and nonlinear-observable Taylor errors. Poisson averaging is applied to each new observable reward; no convergent error series is multiplied by an uncontrolled variable coefficient.

Here is an explicit nonexit construction. Choose a linear coordinate selection R on the equilibrium manifold such that `(a,z)=(R pi(theta),z(theta))` is a local chart; the derivative formulas above guarantee one exists. Use nested chart tubes

    T_i={||a-a_*||_infinity<=r_i, E(z)<=e_i},
    r_0<r_1<r_2, e_0<e_1<e_2,

whose inverse images are compact in the common smooth/routing/rank/capacity neighborhood. Shrink the initial cone to lie in the interior of T_0. Set `d_*=dist_infinity(T_1,complement int(T_2))>0`. Coordinate selection has infinity operator norm one. Require

    e_pi<min(r_1-r_0, Delta/(2||u||_1)),
    e_E<e_1-e_0,
    H K_Adam eta0<d_*,  K_Adam=73.

For analysis, smoothly extend gradients and observables outside a slightly larger tube; inside T_2 they agree with the CNN. Starting in T_1, the universal raw Adam bound keeps every intermediate phase and the next boundary inside T_2. The two observable identities then put that next boundary back in T_1: a changes by at most e_pi from its initial value, and E never exceeds its initial value plus e_E. Induction proves nonexit forever on the error event. Thus the auxiliary process coincides with the actual unmodified optimizer on that event.

Since the E dissipation is nonnegative and its remainder converges, its weighted dissipation sum is finite and E has a limit. A positive limit would contradict `sum delta_k=infinity` and the uniform positive lower bound on dissipation away from zero. Thus z -> 0. Every pi coordinate converges, so the local chart gives convergence of all parameters, including intermediate phases because their movements are O(delta_k). Finally

    m(theta0)-m(theta_infinity)
      >=Delta-||u||_1 e_pi>Delta/2.

This proves actual net sinking with retained moments and a moving network. Finite derivative constants for pi come from the endpoint lemma, and those for E o z are bounded by the first two derivatives of z on the fixed chart. Together with the existing observable constants they give a positive eta0 for each prescribed alpha. No numerical probability-certified eta0 or practical training time is claimed here.

## 6. Finite native identity check and limits

[verify_adam_three_images.py](verify_adam_three_images.py) constructs 331 independent raw parameters with RGB12x12, native Conv5/Pool blocks, FC3/3 and all biases. The [saved result](../../results/cnn_drive_1009/adam_three_images.json) checks rank-three hidden features and logit Jacobian, the exactly preserved mean-input direction, all eight binary CE gradient factorizations, positive raw gradient floors, strict full/self routing, and a finite capacity continuity bound at lambda=1.

The smallest feature/Jacobian singular values are approximately 6.27e-4 and 5.24e-4. The minimum raw gradient over all coordinates and all eight assignments is about 1.82e-4. Full/self capacity lower margins after the explicit continuity error are approximately 4.77744 and 4.75482; the largest CE formula error is 5.56e-17. These are float64 identities/margins supporting the analytic construction, not rigorous interval arithmetic or a measurement of long-time success probability.

The verifier also checks the weighted-projection quadratic-form identity for a freely chosen positive diagonal metric. That metric is explicitly not an estimate of the stationary Adam D_*; the identity holds analytically for every positive diagonal metric. The actual-Adam cone in the theorem uses its own frozen-history D_*.

The one-image restriction is removed in a rank-three construction. The theorem still assumes three specially constructed nearby images, binary output, full-batch reuse, local strict routing and sufficiently small decaying rates. It does not establish arbitrary CIFAR image data, ten classes, shuffled minibatches, constant learning rates, negative mean, or gate death. No other activation function is introduced.

Independent audits: [native reference, verifier and finite cone](adam_three_images_reference_review.md), and [quantitative C2 endpoint plus integrated actual-Adam theorem](adam_three_images_longtime_review.md). Both passed within this scope.
