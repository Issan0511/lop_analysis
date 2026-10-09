# Three-image binary Adam: an elementary C2 equilibrium projection

2026-10-09. This note constructs the smooth endpoint projection needed for a multi-image equilibrium tube. It treats N=3 (the argument works for any fixed finite N) and gives explicit variational estimates rather than invoking an unproved stable-foliation theorem. Construction of the native CNN reference, its full/self capacity signs, and the nested stochastic tube are companion tasks.

## 1. Frozen taskwise Adam field and its smooth positive diagonal factor

Let theta contain every raw weight and bias, z(theta)=(z_1,z_2,z_3) the binary half-logit contrasts, and J(theta)=D z(theta), an N-by-P matrix. One full batch uses labels Y_n independently uniform in {-1,1}, held fixed for H optimizer steps per task. The frozen raw coordinate gradient is

    G_j(theta,Y)=mu_j(theta)-s_j(theta)^T Y/N,
    mu_j(theta)=[J(theta)^T tanh z(theta)]_j/N,
    s_j(theta)=(partial_j z_1,...,partial_j z_N).

The noise is exactly symmetric under complementing all labels. Repeating a full-batch gradient H times is handled by grouping stationary Adam EMA weights by independent tasks. The stationary phase sum therefore has the form

    Fbar(theta)=D(theta) J(theta)^T tanh z(theta)/N,    (1.1)

where D is diagonal and positive. D includes the sum over H phases; the averaged ODE clock is one base learning-rate unit per task.

For clarity, smoothness at mu_j=0 is proved in independent variables rather than asserted by dividing a composite function by zero. Let R_j(mu,s) be the stationary phase-summed response to the scalar task gradient `mu-s^T Y/N`. Suppose on a compact neighborhood all signed sums satisfy

    min_{Y in {-1,1}^N}|s^T Y/N|>=b>0,
    |mu|<=b/2.                                     (1.2)

The same lower bound b/2 holds along every segment t mu, 0<=t<=1. Every stationary second-moment RMS is then at least b/2. All finite derivatives of the EMA quotient in (mu,s) are uniformly bounded: the first and second EMA weights sum to one, the label support is finite, and the quotient denominators stay separated from zero. Differentiation under expectation is legitimate. Thus R_j is smooth.

Complementing the entire label history gives `R_j(-mu,s)=-R_j(mu,s)` and `R_j(0,s)=0`. Define

    c_j(mu,s)=integral_0^1 partial_mu R_j(t mu,s) dt.  (1.3)

Then R_j(mu,s)=mu c_j(mu,s), with a smooth c_j through mu=0. The independent-task odd-monotone quotient lemma supplies the positive bounds

    H epsilon/(epsilon+G_max)^2<=c_j<=H/epsilon,     (1.4)

where G_max bounds `|mu|+max_Y|s^T Y/N|` over the interpolation region. Equation (1.3) is the definition of the removable value; positivity follows from (1.4) and continuity, not from claiming each partial_mu R is pointwise positive for arbitrary parameters. Substituting the smooth functions mu_j(theta),s_j(theta) gives D_jj(theta)=c_j(mu_j(theta),s_j(theta)).

Condition (1.2) is substantive. For three sufficiently similar nonzero sensitivities, the odd number of signed terms yields a gap; three exactly identical images would, however, fail the separate Jacobian-rank condition below. Uniform noise separation and full row rank must both be checked at the reference.

## 2. Local deterministic hypotheses

Fix theta_* with z(theta_*)=0. On a neighborhood assume:

1. z is C3 and D is C2. The strict CNN branch and (1.2) actually provide higher smoothness.
2. J has full row rank N=3.
3. D is diagonal positive with uniform lower and upper bounds on the chosen compact neighborhood.

Consider the averaged ODE

    dot theta=-D(theta)J(theta)^T tanh z(theta)/N.     (2.1)

The zero-output set M={theta:z(theta)=0} is a regular submanifold. The aim is to prove directly that nearby solutions converge to M, that their endpoint pi(theta) is C2, and that pi is constant along (2.1).

Output biases alone give only a rank-one common-image direction. Full row rank for three images is an additional geometric requirement; it is not inferred from the presence of output biases.

## 3. Local coordinates and explicit coefficient bounds

Choose a constant (P-N)-by-P matrix R with orthonormal rows spanning ker J(theta_*). The map

    T(theta)=(y,z)=(R(theta-theta_*),z(theta))

is a local C3 coordinate chart. At theta_* its smallest singular value is

    s_T,0=min{1,sigma_min(J(theta_*))}>0.

Shrinking a convex parameter neighborhood until `||J(theta)-J(theta_*)||<s_T,0/2` makes the chart uniformly invertible and locally injective; its inverse chi has bounded first and second derivatives. One may choose a compact coordinate box inside its image.

In these coordinates, put

    A(y,z)=J D J^T/N,
    B(y,z)=R D J^T/N,
    q(z)=tanh z.

The ODE becomes

    dot y=-B(y,z)q(z),
    dot z=-A(y,z)q(z).                              (3.1)

All matrix norms below are Euclidean operator norms. On a compact outer coordinate box take explicit finite bounds

    A>=a I, a>0;   ||A||<=A_0;   ||B||<=B_0,
    ||D A||<=A_1;  ||D B||<=B_1,
    ||D^2 A||<=A_2; ||D^2 B||<=B_2.                (3.2)

Derivatives are with respect to the combined state (y,z); second derivatives use the bilinear operator norm. These constants are computable from the local J,D and chart bounds.

Choose a normal radius delta>0 and set

    lambda=3a/4,       nu=a/2,
    Q_0=B_0+B_1 delta,
    kappa=(delta/lambda)[B_1+Q_0 A_1/nu].

Require

    A_0 delta^2/3<=a/4,
    A_0 delta^2+A_1 delta<=a/2,
    kappa<=1/2.                                    (3.3)

All hold for sufficiently small positive delta. Also require `B_0 delta/lambda` to be smaller than the reserved distance of the initial y-set to the outer y-boundary. No sign assumption on a future stochastic update is involved.

## 4. Exponential normal decay and finite tangential displacement

For ||z||<=delta,

    ||tanh z-z||<=||z||^3/3,
    ||D(tanh z)-I||<=||z||^2.

The first inequality follows by integrating `|tanh'(x)-1|=tanh(x)^2<=x^2` coordinatewise. Hence along (3.1),

    (1/2)d||z||^2/dt
       <=-[a-A_0 delta^2/3]||z||^2
       <=-lambda||z||^2.

Therefore

    ||z(t)||<=||z(0)|| exp(-lambda t),
    integral_0^infinity ||dot y||dt
        <=B_0||z(0)||/lambda.                      (4.1)

A first-exit argument now keeps the solution inside the outer coordinate box: z cannot leave its small normal ball, and the total y displacement is below the reserved tangential margin. Consequently y(t) has a limit y_infinity, with a uniform exponential tail. This proves existence of a local endpoint without assuming trajectory boundedness.

## 5. First variational equation: bounded tangent derivatives, decaying normal derivatives

For an initial variation v=(v_y,v_z), let (Y(t),Z(t)) be the derivative of the flow. Its equation is

    dot Y=P(t)Y+Q(t)Z,
    dot Z=C(t)Y+S(t)Z,                             (5.1)

with the following bounds, directly obtained by differentiating (3.1):

    ||P(t)||<=B_1 delta exp(-lambda t),
    ||Q(t)||<=Q_0,
    ||C(t)||<=A_1 delta exp(-lambda t),
    symmetric_part(S(t))<=-nu I.                   (5.2)

For the final inequality, `S=-A Dq-(D_z A)[.]q`; its difference from -A has norm at most A_0 delta^2+A_1 delta, bounded by a/2 in (3.3). Thus its time-dependent fundamental solution has norm at most exp[-nu(t-s)].

For a finite horizon define Y_sup=sup_t||Y(t)||. Variation of constants gives

    integral ||Z(t)||dt
       <=||v_z||/nu+A_1 delta Y_sup/(nu lambda).

Integrating the tangent equation gives

    Y_sup<=||v_y||+(Q_0/nu)||v_z||+kappa Y_sup.

The kappa<=1/2 condition therefore yields the uniform bounds

    Y_sup<=C_Y||v||,
    C_Y=2(1+Q_0/nu),

    ||Z(t)||<=C_Z exp(-nu t)||v||,
    C_Z=1+A_1 delta C_Y/(lambda-nu).                (5.3)

The same constants hold on arbitrarily long finite horizons and hence on [0,infinity). Moreover

    ||dot Y(t)||
       <=[B_1 delta C_Y+Q_0 C_Z]exp(-nu t)||v||.    (5.4)

Thus the tangent derivative has a uniform limit and the normal derivative tends to zero. This is the first part of the C2 endpoint construction, not just a trajectory-level estimate.

## 6. Second variational equation and an explicit Hessian bound

For two initial directions v,w, write (Y_2,Z_2) for the second derivative of the flow. Its initial value is zero in the (y,z) coordinates. It satisfies the same homogeneous system (5.1), plus quadratic forcing (f_y,f_z).

For a matrix coefficient C equal to B or A, the second derivative of `-C(y,z)q(z)` is

    -D^2C[v,w]q
    -DC[v]Dq w_z-DC[w]Dq v_z
    -C D^2q[v_z,w_z].                              (6.1)

Each term contains either the decaying base z or a decaying first normal variation. Put C_V=C_Y+C_Z and define

    K_B=B_2 delta C_V^2+2B_1 C_V C_Z+2B_0 C_Z^2,
    K_A=A_2 delta C_V^2+2A_1 C_V C_Z+2A_0 C_Z^2.

Using (4.1), (5.3), ||Dq||<=1 and ||D^2q||<=2 gives

    ||f_y(t)||<=K_B exp(-nu t)||v||||w||,
    ||f_z(t)||<=K_A exp(-nu t)||v||||w||.           (6.2)

Repeating the integral absorption from section 5 yields

    sup_t||Y_2(t)||<=C_Y2||v||||w||,
    C_Y2=2[K_B/nu+Q_0 K_A/nu^2].                   (6.3)

Indeed the normal forcing has total integral at most K_A/nu, and its stable convolution adds at most K_A/nu^2 to the normal L1 norm. The same kappa is absorbed; there is no new unverified stability assumption.

Pointwise variation of constants gives

    ||Z_2(t)||<=C_Z2 exp(-nu t/2)||v||||w||,
    C_Z2=A_1 delta C_Y2/(lambda-nu)+2K_A/nu.        (6.4)

Here `t exp(-nu t)<= (2/nu)exp(-nu t/2)` bounds the possible resonant convolution. Finally,

    ||dot Y_2(t)||
       <=[B_1 delta C_Y2+Q_0 C_Z2+K_B]
                           exp(-nu t/2)||v||||w||. (6.5)

Thus every second tangent derivative converges uniformly, and every second normal derivative tends uniformly to zero. The estimates are uniform over a compactly contained set of initial points. Finite-time flow differentiability follows directly from the C2 vector field and its variational equations; uniform convergence of the flow, its first derivatives and its second derivatives proves that its endpoint map is C2. No stable-foliation theorem is being used as a substitute for these estimates.

## 7. Endpoint projection, invariance and finite derivative constants

In coordinates define

    Pi(y,z)=(y_infinity(y,z),0),
    pi(theta)=chi(Pi(T(theta))).                   (7.1)

Then pi is C2, its image is M, and pi|_M=id. The flow semigroup property gives

    pi(flow_t(theta))=pi(theta),
    Dpi(theta)Fbar(theta)=0.                        (7.2)

The coordinates of Pi have first derivative operator bound C_Y and Hessian bilinear bound C_Y2. To translate this into raw-parameter bounds, let

    ||DT||<=C_T1,  ||D^2T||<=C_T2,
    ||Dchi||<=C_chi1, ||D^2chi||<=C_chi2.

One can use `C_chi1<=1/s_T` and `C_chi2<=C_T2/s_T^3`, where s_T is a certified lower singular bound for DT on the chart. Chain rule gives

    ||Dpi||<=C_chi1 C_Y C_T1,

    ||D^2pi||<=C_chi2(C_Y C_T1)^2
       +C_chi1[C_Y2 C_T1^2+C_Y C_T2].             (7.3)

These are finite, explicitly constructed bounds. For the earlier observable theorem, a conservative conversion is `||grad pi_i||_1<=sqrt(P)||Dpi||` and `sum_{j,l}|partial_j partial_l pi_i|<=P^2||D^2pi||`. Smooth cutoff costs can then be added by the product-rule bounds in the observable companion.

The set of nearby points with a given pi value is a local stable leaf. This assertion also follows elementarily: in (y,z) coordinates, the derivative of `(y_infinity(y,z),z)` at z=0 is block triangular with identity tangent and normal diagonal blocks, so it is a local coordinate map. Its leaf z-coordinates contract under the ODE while its endpoint coordinate is constant.

## 8. Exact linearized endpoint formula

At an equilibrium theta_* let

    J_*=J(theta_*), D_*=D(theta_*),
    B_*=D_* J_*^T,
    A_*=J_* D_* J_*^T.

The derivative of Fbar at theta_* is B_*J_*/N. Derivatives of D and J are multiplied by tanh z=0 and disappear from this linearization. For an initial linear perturbation v,

    delta theta(t)
      =v-B_*A_*^{-1}[I-exp(-A_*t/N)]J_*v.

Taking the uniform first-derivative limit established above gives

    boxed: Dpi(theta_*)=I-B_*A_*^{-1}J_*.           (8.1)

This is a projection with image ker J_* and kernel range(D_*J_*^T). It is generally not the Euclidean orthogonal projection.

Consequently, for theta_0=theta_*+v with v small,

    theta_0-pi(theta_0)
       =D_*J_*^T(J_*D_*J_*^T)^{-1} z(theta_0)
          +O(||v||^2).                             (8.2)

The coefficient is independent of the common factor 1/N or a common rescaling of the ODE clock. The O(||v||^2) remainder has a finite bound obtained from (7.3) and the Hessian of z.

More explicitly, if H_pi bounds ||D^2pi|| and H_z bounds ||D^2z|| on the segment from theta_* to theta_0, the norm of the remainder in (8.2) is at most

    (1/2)[H_pi+||B_*A_*^{-1}|| H_z]||v||^2.        (8.2a)

For a fixed affine first-Conv mean m(theta)=u^Ttheta+constant, an initial output ray z_0=t e has leading sink coefficient

    t u^T D_*J_*^T(J_*D_*J_*^T)^{-1}e.             (8.3)

Its sign must be checked; positivity of individual entries of J does not make the inverse Gram matrix entrywise positive. A useful noncircular choice, when e_*=J_*D_*u is nonzero, is the raw initial ray v=t D_*u. Then

    u^T D_*J_*^T(J_*D_*J_*^T)^{-1}e_*
       =e_*^T(J_*D_*J_*^T)^{-1}e_*>0.             (8.4)

This supplies a strictly positive leading mean decrease. For a sufficiently small fixed t>0 the quadratic remainder cannot reverse it, and continuity gives an open neighborhood of such initial states. Reference construction and capacity margins must still be verified independently.

For a fully finite margin, write Q=e_*^T A_*^{-1}e_*>0. Taylor expansion of pi along v=tD_*u gives

    m(theta_*+tD_*u)-m(pi(theta_*+tD_*u))
       >=tQ-(1/2)H_pi||u|| t^2||D_*u||^2.          (8.5)

Hence `0<t<Q/[H_pi||u||||D_*u||^2]`, together with the chart-radius restriction, gives a margin at least tQ/2. If the denominator vanishes, the quadratic term is zero and only the chart restriction is needed. This inequality selects a nonempty positive initial ray from fixed reference geometry; it does not assume a desired sign along the later trajectory.

## 9. Population-risk observable and the actual-Adam handoff

Use the nonnegative normal population risk

    Phi(theta)=(1/N)sum_n log cosh z_n(theta).

Its drift under the averaged descent ODE is

    DPhi Fbar
      =(1/N^2)tanh(z)^T J D J^T tanh(z)
      =(1/N)tanh(z)^T A tanh(z)>=0.                (9.1)

On a compact tube with ||z||<=delta and A>=aI, put `k_delta=tanh(delta)/delta`, with value one at zero. Since `Phi<=||z||^2/(2N)`,

    DPhi Fbar>=2a k_delta^2 Phi.                   (9.2)

Apply the proved C2-observable averaging theorem to every component of pi and to Phi. It yields convergent remainder processes with a simultaneous uniform high-probability budget tending to zero with the base learning-rate scale. Before exit from the certified tube,

    pi(theta_K)=pi(theta_0)+R_K^pi,

    Phi(theta_K)=Phi(theta_0)
         -sum_{k<K}delta_k DPhi(theta_k)Fbar(theta_k)+R_K^Phi.

If the separate nested-tube argument proves nonexit on that error event, (9.2) and remainder convergence imply Phi(theta_K)->0 and hence z(theta_K)->0. Every pi component converges as well. The finite ODE endpoint-path estimate gives `||theta-pi(theta)||<=C||z(theta)||` on the tube, so the full parameter sequence converges to pi_infinity in M. The within-task displacement bound tends to zero, transferring convergence to all optimizer steps.

If the initial mean-to-endpoint margin in (8.3) or (8.4) exceeds twice the pi-error contribution `||u||_1 sup||R^pi||_infinity`, the actual limiting first-Conv mean remains strictly below its initial value. This is a conditional consequence of the observable error event and nonexit, not an assumption about the realized Adam update signs.

## 10. Scope and possible degeneracies

- Rank(J)=3 is essential. Exactly identical images give only a scalar normal direction; the displayed inverse Gram formula then does not apply.
- Near-identical but distinct images can satisfy both the odd signed-sum noise gap and full row rank. The smallest singular value may be tiny, making the normal radius and step-size certificates correspondingly conservative. No uniform limit as the images become identical is asserted.
- Output biases do not by themselves provide all three normal directions.
- This note does not assume that the diagonal stationary coefficients coincide across coordinates, nor that moments are reset between tasks.
- The construction remains binary, full-batch, finite-H task reuse and ReLU-only. Native CNN reference geometry, capacity-self positivity and the full stochastic noexit event are separate checks.
