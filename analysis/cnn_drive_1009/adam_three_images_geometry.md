# Three-image binary Adam: a noncircular finite cone of strict endpoint sinking

2026-10-09. This note proves the local geometry needed to extend the one-image equilibrium-tube argument to N=3 distinct nearby images. The actual stochastic-averaging theorem is a separate companion result. The key conclusions are a smooth averaged-flow endpoint map, its weighted normal projection at equilibrium, and an explicit finite open cone on which the actual first-Conv mean loses a strictly positive amount before reaching that endpoint. The direction is constructed from current Jacobians and stationary Adam coefficients, not from an assumed final sink sign.

All parameters below are independent raw weights and biases. No optimizer projection, parameter tying, moment reset, or freezing is introduced.

## 1. Assumptions that are finite and local

Fix N=3 images and binary half-contrasts

    z(theta)=(z_1(theta),z_2(theta),z_3(theta))^T,
    J(theta)=D_theta z(theta),                      (N by P).

Each task draws Y in {-1,+1}^3 with independent uniform entries and reuses the same assignment for H full-batch updates. Mean binary CE has raw gradient

    g(theta,Y)=N^(-1) J(theta)^T [tanh z(theta)-Y].  (1.1)

The componentwise tanh is understood. Assume a constructive reference theta_* and a compact local branch region O with:

1. `z(theta_*)=0` for all three images and `rank J(theta_*)=3`.
2. Every full and, if needed, literal-self ReLU/MaxPool branch has strict margins in O. Thus the finite network maps and their derivatives are smooth there.
3. Every one of the finitely many 8 raw gradient vectors at theta_* has every coordinate bounded away from zero. By shrinking O this remains true on O. This condition ensures smooth stationary Adam coefficients, not a desired drift sign.
4. The first-Conv target mean is the fixed affine observable `m(theta)=u^T theta+constant`, where u is the augmented patch mean averaged over the three images. Its target-filter entries are nonnegative, including bias entry 1. The relevant columns of J are strictly positive at the reference. This is supplied, for example, by positive hidden paths and positive head contrast.

Full row rank and strict inequalities are open conditions. No equality between raw Conv weights or channels is imposed.

### 1.1 Why the 8-label gradient floor is nonempty near repeated images

At three identical copies of a one-image zero-contrast reference, suppose each raw one-image sensitivity s_j is nonzero. Then each row of J equals s^T and

    g_j=-(s_j/3)(Y_1+Y_2+Y_3).

Because the sum of three signs is odd, `|g_j|>=|s_j|/3`. For nearby distinct images, a sufficient reference condition is

    sum_(n=1)^3 |J_(n,j)-s_j| < |s_j|/2.

It gives `|(J^T Y)_j|>=|s_j|/2` for all 8 assignments and therefore a gradient floor `|s_j|/6` at z=0. A further small state neighborhood preserves a positive floor even after adding the population mean gradient. Thus odd N=3 avoids the exact zero-label-sum obstruction present for identical even-size batches.

The repeated-image state has rank-one J, but a separate native-CNN construction can choose distinct nearby images with rank-three J while retaining this strict gradient-floor condition. The geometry theorem requires and directly checks that rank; it does not infer it solely from image distinctness.

## 2. Smooth positive diagonal stationary field

Write the population raw gradient as

    mu(theta)=N^(-1)J(theta)^T tanh z(theta),
    xi(theta,Y)=-N^(-1)J(theta)^T Y.

For fixed theta the distribution of xi is centrally symmetric because Y and -Y have the same iid binary law. In each raw coordinate the repeated frozen gradient is `mu_j+xi_j`, and the independent objects are whole task assignments, not their H copies.

The grouped-EMA odd-monotonicity result gives the phase-summed stationary averaged Adam field

    Fbar(theta)=D(theta) mu(theta)
               =N^(-1)D(theta)J(theta)^T tanh z(theta),   (2.1)

where D is a positive diagonal matrix. It includes the sum over the H task phases; using a phase average instead changes only the time scale. Uniformly bounded gradients give positive lower and upper bounds on every diagonal entry.

The gradient floor in Section 1 bounds every frozen stationary RMS away from zero. Holding the noise coefficients J fixed and temporarily regarding mu_j as an independent scalar, the expected quotient is smooth and odd in mu_j. Dividing by mu_j therefore has a smooth removable extension at mu_j=0. Differentiation under expectation is justified by the uniform floor and bounded smooth raw derivatives. Thus D(theta) is smooth on a sufficiently small O, including at z=0. Cross-coordinate noise dependence is irrelevant to this coordinatewise conclusion.

Set

    K(theta)=J(theta)D(theta)J(theta)^T.

Shrinking O around the full-row-rank reference gives

    K(theta)>=kappa I_N,  kappa>0,
    ||D(theta)J(theta)^T||_(2 to infinity)<=B<infinity.  (2.2)

All constants can be bounded from this fixed compact region. They are not assumptions about the unknown future Adam path.

## 3. The correct normal Lyapunov observable

The averaged ODE is

    theta_dot=-Fbar(theta),
    z_dot=-N^(-1)K(theta)tanh z.                    (3.1)

For N>1, positive definiteness of K alone does not justify a global claim that `z^T K tanh z` is nonnegative. Accordingly, use

    E(z)=sum_(n=1)^N log cosh(z_n).                 (3.2)

Its exact dissipation is

    E_dot=-N^(-1)tanh(z)^T K(theta)tanh(z)
          <=-(kappa/N)||tanh z||_2^2.              (3.3)

On a local region with `||z||_infinity<=Z`, put

    l_Z=sech(Z)^2>0,   b_Z=tanh(Z)/Z>0.

Then

    (l_Z/2)||z||_2^2 <= E(z) <= (1/2)||z||_2^2,
    ||tanh z||_2 >= b_Z ||z||_2.

As long as the path stays in O,

    E(z(t))<=E(z(0)) exp[-2 lambda t],
    ||z(t)||_2<=C_z ||z(0)||_2 exp[-lambda t],
    lambda=kappa b_Z^2/N,  C_z=l_Z^(-1/2).          (3.4)

The speed satisfies

    ||theta_dot||_infinity <= (B/N)||z||_2.

Hence its full path length is at most

    integral_0^infinity ||theta_dot||_infinity dt
       <=C_path ||z(0)||_2,
    C_path=B C_z/(N lambda).                       (3.5)

If an initial state is at distance at least d from the boundary of O and `C_path||z(0)||_2<d`, a first-exit argument makes (3.4)-(3.5) valid for all time. The trajectory converges to an endpoint on the regular equilibrium manifold

    M={theta:z(theta)=0}.

This proves averaged local containment and convergence from finite geometric bounds. It does not assume them along a future trajectory.

## 4. Smooth endpoint map and its linearization

Define pi(theta) as the limit of the averaged ODE starting at theta in a sufficiently small neighborhood U of theta_*. The path-length argument proves existence. The map is C2 after shrinking U, as can be seen directly from the stable-normal equations below; no integrability of a chosen normal distribution is assumed.

### 4.1 Smoothness justification

Full row rank of J supplies local smooth coordinates `(x,z)`, with x in R^(P-N). In these coordinates the vector field has the form

    x_dot=b(x,z),
    z_dot=-N^(-1)K(x,z)tanh z,

where b(x,0)=0 and the whole equilibrium set is z=0. All coefficients have bounded derivatives of every needed finite order on a small compact chart. In particular, tangential derivatives of the vector field vanish at z=0 and are O(||z||). The normal linearization at z=0 is `-K(x,0)/N`, uniformly negative definite.

Shrink the normal chart so the symmetric part of the normal variational block remains bounded above by a negative constant. The variational equations then have these properties along a base trajectory:

- Their tangent-to-tangent and tangent-to-normal forcing coefficients are O(||z(t)||), hence exponentially integrable by (3.4).
- Their normal-to-tangent coefficients are bounded.
- Their homogeneous normal block is uniformly exponentially stable.

Variation of constants and the exponential bound on z give a bounded tangent first variation and exponentially decaying normal first variation. One elementary way to close the bound is to choose the normal chart small enough that the integral feedback coefficient from tangent variation through the normal equation and back is below 1/2; all coefficients and integrals are finite known bounds from the chart. The tangent derivative then has an integrable time derivative and converges uniformly on a smaller compact chart.

For second variations, terms with two tangent directions are O(||z(t)||), because the vector field vanishes identically on z=0. Mixed terms contain an exponentially decaying normal first variation; terms with two normal directions decay likewise. The inhomogeneous second-variation forcing is therefore exponentially integrable. The same stable-block argument yields uniformly convergent tangent second variations and decaying normal second variations. Uniform convergence of the flow, its first derivatives, and its second derivatives gives a C2 endpoint pi on a smaller neighborhood. The CNN branch and the gradient-floor stationary field are smooth enough for these derivative bounds.

Thus the following C2 constants are finite and can be bounded from ordinary flow-variation inequalities on a compact reference chart. Their existence is not predicated on stochastic stability.

### 4.2 Endpoint identities

The ODE semigroup gives

    pi|_M=id,
    pi(flow_t(theta))=pi(theta),
    Dpi(theta)Fbar(theta)=0.                        (4.1)

For N>1 it is generally unnecessary, and potentially false, to assert `Dpi(theta)D(theta)J(theta)^T=0` throughout U. Only (4.1) is required for observable averaging. Different normal vector fields need not commute; pi is the endpoint of the specified averaged ODE, not the result of an arbitrary sequence of coordinate-wise logit corrections.

At theta_*, write D_*=D(theta_*), J_*=J(theta_*), K_*=J_*D_*J_*^T. The linearized ODE is

    h_dot=-N^(-1)D_*J_*^T J_* h.

Its exact solution has

    J_*h(t)=exp[-K_*t/N]J_*h(0),
    h(t)=h(0)-D_*J_*^T K_*^(-1)
                 [I-exp(-K_*t/N)]J_*h(0).

Consequently

    Dpi(theta_*)=I-Q_*,
    Q_*=D_*J_*^T K_*^(-1)J_*.                      (4.2)

Q_* is a projection onto `range(D_*J_*^T)` along `ker J_*`. It need not be an orthogonal Euclidean projection.

## 5. A natural noncircular positive-loss ray

Let the target mean-gradient u be fixed as in Section 1. Define the initial direction

    v_*=D_*u.                                     (5.1)

It raises only the selected first-filter/bias coordinates when u has its usual nonnegative support. D_* is computed from the stationary frozen reference, before any new labels or trajectory are observed.

Let

    r_*=J_*D_*u,
    ell_*=u^T Q_*,
    b_*=ell_*v_*=r_*^T K_*^(-1) r_*.              (5.2)

The positive target sensitivities imply every component of r_* is positive; in particular r_*!=0. Since K_*>0,

    b_*>0.                                        (5.3)

Define the endpoint mean loss

    L_m(theta)=m(theta)-m(pi(theta)).

At the reference L_m(theta_*)=0 and

    D L_m(theta_*)=u^T Q_*=ell_*.

Therefore v_* has a strictly positive **derived** endpoint-loss derivative b_*. Neither the sign of the future displacement nor its final value was assumed.

An alternative valid normal ray is `D_*J_*^T J_*D_*u`: it lies in range(D_*J_*^T), is annihilated by Dpi(theta_*), and has positive mean pairing `||J_*D_*u||_2^2`. The simpler v_*=D_*u is preferred because it is the target mean direction weighted by the known Adam metric. It need not be purely normal; its tangential endpoint motion is correctly included by Q_*.

## 6. Explicit finite cone and endpoint margin

On a convex small ball about theta_* contained in U, bound the scalar Hessian by

    C_m >= sup_theta sum_(i,j)|partial_i partial_j L_m(theta)|<infinity.

This follows from C2 smoothness of pi and the fixed affine mean. Put

    L_ell=||ell_*||_1>0.

Choose a positive cone radius q satisfying

    q < b_*/(4 L_ell),
    M_v=||v_*||_infinity+q.

Choose t_U>0 so that the ball of radius t_U M_v stays in the C2 chart and

    t_U < b_*/(2 C_m M_v^2)                         (6.1)

when C_m>0; if C_m=0 this latter restriction is unnecessary. For any t in (0,t_U) and ||e||_infinity<q, set

    theta0=theta_*+t(v_*+e).                       (6.2)

Taylor's theorem gives

    L_m(theta0)
       >=t[b_*-L_ell q]-(C_m/2)t^2 M_v^2
       > (b_*/2)t >0.                             (6.3)

The strict finite inequalities derive the final sign from a known reference ray and a bounded error. They do not impose `L_m>0` as an input condition.

Choose 0<t_L<t_U. The union of the open balls (6.2) over t in (t_L,t_U) is a nonempty full-dimensional open subset of raw parameter space. It permits independent perturbations of every Conv coefficient, bias, FC parameter, and both head rows, including parameters outside the support of u. Its uniform deterministic endpoint margin is

    L_m(theta0)>Delta_*:=b_* t_L/2.                (6.4)

All chart/branch requirements can be enforced by shrinking t_U and q; they impose only finite distances from the constructive reference. The cone is therefore finite, robust, and noncircular.

## 7. Nested tube and the actual stochastic transfer

Because the combined coordinates `(pi,z)` have full rank at theta_* by (4.2) and rank J_*=N, they form a local chart after restricting pi to coordinates on M. Indeed a tangent vector annihilated by J_* lies in ker J_*, on which Dpi is the identity; the remaining normal part is seen by J_*.

Choose nested compact flow-coordinate tubes T_initial inside T_middle inside T_outer, all contained in the simultaneous branch/gradient-floor/capacity neighborhood. The initial cone (6.2) can be made to lie in T_initial. Let

    d_*=dist_infinity(T_middle,complement T_outer)>0.

Apply the companion observable-averaging theorem to every ambient coordinate of pi and to the smooth normal energy E(z)=sum log cosh z. It gives, up to localization, convergent remainders with simultaneous uniform small budgets:

    pi(theta_(KH))=pi(theta0)+R_K^pi,

    E(z(theta_(KH)))=E(z(theta0))
       -sum_(k<K)delta_k A_E(theta_(kH))+R_K^E,

    A_E(theta)=N^(-1)tanh(z)^T K(theta)tanh(z)>=0.    (7.1)

Choose the probability budgets so that the pi displacement and the E upper bound keep all task boundaries in T_middle. For example, if T_outer has `||z||_infinity<=Z_2`, its energy bound implies

    ||z||_2^2<=2[E_initial,max+epsilon_E]/sech(Z_2)^2,

which may be required to lie below the middle tube's squared normal radius. Use the common scalar rate delta_k with a power exponent in (1/2,1], and impose the intermediate-step margin

    H K_Adam delta0 < d_*.                         (7.2)

Then the same first-exit argument as in the one-image case prevents an exit at every optimizer phase. This is where branch and sensitivity preservation are proved for the actual trajectory; they are not assumed for all future times.

On the simultaneous event, convergence of the remainders and nonnegative dissipation imply E converges and its drift sum is finite. The lower bound in (3.3) on the compact tube rules out any positive E limit when sum delta_k diverges. Thus z tends to zero. Every pi coordinate converges, and the local chart gives convergence of the full parameter vector to theta_infinity on M, including intermediate optimizer phases.

If the uniform coordinatewise pi error is epsilon_pi, then

    |m(theta_infinity)-m(pi(theta0))|
       <=||u||_1 epsilon_pi.

Choose

    epsilon_pi < Delta_*/(2||u||_1)
               =b_*t_L/(4||u||_1).                (7.3)

The actual final mean decrease is then strictly at least Delta_*/2=b_*t_L/4. The error thresholds from observable averaging tend to zero with delta0, so these strictly positive budgets admit a sufficiently small positive rate. The actual rates have divergent sum and convergent squared sum because they are delta_k repeated H times, with no vanishing state multiplier required.

This is a conditional high-probability pathwise result. Individual z coordinates, momenta, and hidden-mean increments may change sign. No all-time same-sign condition is used.

## 8. Original full/self capacity sign for nearby distinct images

Capacity sign remains an independent geometric certificate. At three repeated copies of a one-image positive reference, the all-raw kernel and its target-mean derivative are

    K_rep=(1_3 1_3^T) tensor K_one,
    K'_rep=(1_3 1_3^T) tensor K'_one.

If the one-image derivative is nonzero PSD, then for each prescribed finite ridge lambda>0,

    trace[(lambda I+K_rep)^(-1) K'_rep]>0.

The same applies to the literal channel-isolated network with its unchanged head and biases. Raw Jacobians, directional Jacobians, actual patch-mean u, and inverse-ridge traces vary continuously with the input images and parameters while full/self routing margins persist. Therefore sufficiently nearby distinct images retain both strict capacity signs for the chosen lambda. Uniform continuity gives a common neighborhood for lambda in any fixed compact interval bounded away from zero and infinity.

This continuity argument alone does **not** prove a single neighborhood valid for every lambda>0: the repeated-image derivative is rank deficient in image space, and image perturbations need not preserve its PSD property. A stronger separate finite-kernel certificate is needed for such a uniform-in-all-ridge claim. For the original capacity at a specified ridge, strict trace continuity is sufficient and noncircular.

The native reference construction is responsible for simultaneously obtaining equal zero contrast at all three nearby images and rank-three J. Once it does, intersect its strict full/self capacity neighborhood with the strict branch/gradient-floor region O used here. The positive-ray and finite-cone proof then applies without redefining CE or the averaged field as a new self term.

## 9. Scope

This note replaces the special equal-amplitude invariant line by a full-dimensional local cone for three independent binary labels, with exact H-fold task reuse and a smooth positive stationary Adam metric. The mechanism is local attraction to the zero-contrast equilibrium manifold and a quantitatively positive loss of the original first-Conv mean along the stable endpoint map. Features and parameters generally remain nonzero.

It does not prove arbitrary many-class Adam signs, constant-rate convergence, negative bulk preactivation, gate death, or the actual RL-CIFAR trajectory. Smooth coefficient floors, full-row-rank J, and sufficient capacity margins must be verified at the native reference. These are finite current-state conditions; future sign or containment is derived rather than assumed.
