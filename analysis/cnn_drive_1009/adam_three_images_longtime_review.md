# Independent audit: quantitative C2 endpoint and the three-image Adam theorem

2026-10-09. Read-only review of:

- `/tmp/cnn_adam_multi_equilibrium_1009.md`, including the explicit finite remainder added to Section 8;
- `/tmp/cnn_adam_multi_geometry_1009.md`, including its finite open cone and capacity-continuity scope;
- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/adam_three_images_longtime.md`.

**Result: PASS. No blocking issue found in the quantitative endpoint lemma or its integration with the previously reviewed observable averaging theorem.** No experiment was rerun and no repository/source file was changed. Native reference construction and the finite raw verifier are companion evidence; this review concentrates on smoothness, the quantitative variational proof, the mean cone, and the infinite-time bootstrap.

## 1. Smooth positive diagonal field at the same reference point

The proof correctly separates two logically different requirements: full row rank of the three-image contrast Jacobian and a nonzero raw-gradient floor for all eight label assignments. Output biases alone cannot supply row rank three, and the text does not claim that they can.

For each raw coordinate, the independent-variable response `R(mu,s)` uses gradients `mu-s^T Y/N`. The condition `min_Y |s^T Y/N|>=b` and `|mu|<=b/2` applies along every interpolation t mu, not just at the final composite state. Consequently every frozen RMS is bounded below by b/2. The EMA coefficient sums, finite label support, and compact smooth geometry bound every required derivative uniformly, justifying differentiation under expectation.

Complementing the entire iid task-label history gives exact oddness in the independent variable mu. Therefore

    R(mu,s)=mu integral_0^1 partial_mu R(t mu,s) dt

gives a smooth removable quotient at zero. The positivity of that quotient follows from the already proved shifted-symmetric task-block sign bound and continuity. The proof does not incorrectly assert that a derivative of the full coupled expectation is pointwise positive without argument.

Substituting the actual mu_j(theta) and s_j(theta) defines one smooth positive diagonal D(theta) at each state. Thus `Fbar=D J^T tanh(z)/N` holds with the same J and D used later in the Gram matrix and endpoint formula. The dependencies of D on all images, head coordinates, and old frozen history are not omitted.

For three nearby copies of a reference sensitivity s_j*, the coordinatewise deviation `<|s_j*|/6` gives total signed-sum error `<|s_j*|/2`. The odd sum of three signs has magnitude at least one, so the raw equilibrium gradient magnitude exceeds |s_j*|/6. A further sufficiently small state neighborhood can enforce the interpolation floor. This establishes nonemptiness of the floor condition without confusing exactly repeated inputs with rank-three inputs.

## 2. Coordinate chart and the deterministic normal decay

The coordinate matrix R in the quantitative lemma has orthonormal rows spanning ker J_*. The rows of J_* are orthogonal to that subspace, so the derivative of the chart `T(theta)=(R(theta-theta_*),z(theta))` has smallest singular value `min(1,sigma_min J_*)`. The stated small derivative perturbation preserves a uniform inverse bound. On a convex reference neighborhood, comparing T with its reference linear map also gives local injectivity; a compact coordinate box can be selected inside its image.

The coordinate ODE is exactly

    y_dot=-B(y,z)tanh z,
    z_dot=-A(y,z)tanh z,
    A=J D J^T/N,  B=R D J^T/N.

The first small-radius inequality `A0 delta^2/3<=a/4` is sufficient for the declared normal decay rate lambda=3a/4. In detail, the cubic tanh remainder contributes at most `(A0/3)||z||^4`, whereas `z^T A z>=a||z||^2`. The estimate is restricted to the chosen small normal ball; no unjustified global positivity of `z^T A tanh z` is assumed.

The tangent speed is at most B0||z||, so its integrated displacement is bounded by B0||z0||/lambda. Reserving more than this distance to the outer tangent boundary makes the first-exit argument valid. The normal radius cannot be crossed because its norm decreases. Hence the deterministic trajectory and endpoint are obtained from finite local constants rather than assumed future containment.

## 3. First variational constants

Differentiating the coordinate ODE gives the block bounds in Section 5:

- `||P||<=B1 delta exp(-lambda t)`;
- `||Q||<=Q0=B0+B1 delta`;
- `||C||<=A1 delta exp(-lambda t)`;
- the symmetric part of S is bounded above by -nu I with nu=a/2.

For the final item, S differs from -A by norm at most `A0 delta^2+A1 delta`. The second small-radius inequality therefore gives a stable time-dependent normal fundamental solution with norm at most exp[-nu(t-s)]. This controls the noncommuting matrix evolution without replacing it by a scalar equality.

Integrating its variation-of-constants bound yields

    integral ||Z|| <= ||v_z||/nu
                      +A1 delta Y_sup/(nu lambda).

Substitution into the tangent integral equation produces precisely

    kappa=(delta/lambda)[B1+Q0 A1/nu].

The condition kappa<=1/2 gives the stated conservative `C_Y=2(1+Q0/nu)`. Since lambda-nu=a/4>0, the pointwise convolution gives the stated `C_Z=1+A1 delta C_Y/(lambda-nu)`. The derivative of Y is exponentially integrable with the claimed coefficient. These bounds are uniform on all finite horizons and hence on the full time interval.

## 4. Second variations and uniform C2 convergence

The displayed second derivative of `-C(y,z)tanh z` contains all four types of terms: the second derivative of C times tanh z, two mixed coefficient/normal derivatives, and C times the second tanh derivative. No term involving the parameter dependence of A or B is missing.

The constants

    K_C=C2 delta(C_Y+C_Z)^2
         +2C1(C_Y+C_Z)C_Z+2C0 C_Z^2

correctly bound the corresponding quadratic forcing by exp(-nu t)||v||||w||. The operator bound ||D² tanh||<=2 is valid for the componentwise map in Euclidean norm. Terms containing the base z decay at least at lambda, and those with two normal variations decay at least at 2nu; replacing these by the slower nu rate is conservative.

The second-variation initial state is zero in the (y,z) coordinates. The stable normal convolution gives L1 forcing contribution K_A/nu². Repeating the same kappa absorption therefore yields exactly

    C_Y2=2[K_B/nu+Q0 K_A/nu²].

The pointwise normal bound has one possible resonant term `K_A t exp(-nu t)`. The inequality `t exp(-nu t)<=2 exp(-nu t/2)/nu` proves the given C_Z2. The resulting derivative of Y2 is exponentially integrable with the stated coefficient.

Thus the flow, its first derivatives, and its second derivatives converge uniformly on compactly contained initial sets. Standard calculus for uniform derivative limits proves the endpoint map is C2. This is a complete quantitative argument; it does not rely on an unproved stable-foliation assertion. The smooth CNN branch and stationary noise floor provide the required coefficient regularity.

## 5. Raw endpoint derivative bounds and exact linearization

The inverse-chart estimates

    ||Dchi||<=1/s_T,
    ||D²chi||<=C_T2/s_T³

follow from the inverse derivative formulas. Applying the chain rule to `pi=chi o Pi o T` gives the displayed first- and second-derivative bounds. The conversions `sqrt(P)||Dpi||` for a coordinate gradient l1 norm and `P²||D²pi||` for a Hessian entry-sum norm are valid conservative bounds for the observable theorem.

At equilibrium, differentiating Fbar eliminates derivatives of D and J because tanh z=0. Solving the remaining linear system gives

    Dpi_*=I-D_*J_*^T(J_*D_*J_*^T)^(-1)J_*.

The solution and its limit use the same positive diagonal D_* as the stationary field. The formula is a generally nonorthogonal projection onto ker J_* along range(D_*J_*^T). The finite remainder bound in (8.2a), including the Hessian of z, follows by two Taylor expansions and is correct.

The positive mean ray `v=D_*u` is valid even though it is not necessarily purely normal. Its leading mean loss is

    (J_*D_*u)^T(J_*D_*J_*^T)^(-1)(J_*D_*u)>0.

The positive target sensitivities and u ensure J_*D_*u is nonzero. The exact-ray quadratic bound (8.5) and the integrated theorem's finite cone bound both have the correct factors. Positivity of an inverse Gram's individual entries is neither assumed nor needed.

For several outputs, only `Dpi Fbar=0` is asserted throughout the neighborhood. The stronger `Dpi D J^T=0` would require more than the endpoint construction and is appropriately not used.

## 6. Normal-risk observable and parameter convergence

The unscaled energy `E=sum log cosh z` used in the integrated theorem has averaged dissipation

    DE Fbar=tanh(z)^T(J D J^T)tanh(z)/N>=0.

The quantitative endpoint note instead uses its 1/N-normalized version and consistently includes the extra factor 1/N. Its lower bound `DPhi Fbar>=2a k_delta² Phi` is correct for A=JDJ^T/N>=aI. The distinction is only a common normalization and is handled consistently.

Observable averaging is applied anew to each coordinate of pi and to the chosen risk. It includes their Taylor errors and applies the Poisson decomposition to their own varying-gradient rewards. The finite derivative estimates above supply the required constants.

On nonexit, the risk remainder converges and the accumulated dissipation is nonnegative. This makes the dissipation sum finite and the risk convergent. A positive risk limit would force a positive eventual drift and contradict divergent sum delta_k. Therefore z tends to zero. Every pi component converges as well. The local (pi,z) chart, or the uniform endpoint path-length estimate, then gives convergence of the entire raw parameter vector, not just convergence of logits. Intermediate phases converge because their displacement from the boundary is O(delta_k).

## 7. Audit of the integrated nested-tube bootstrap

A coordinate-selection matrix R with infinity operator norm one can indeed be chosen so that R restricted to the tangent space ker J_* has full rank. A basis matrix for that tangent space has a nonsingular row minor; selecting those ambient coordinates supplies R. Together with the projection formula, `(R pi,z)` is a local coordinate chart. This R is a separate convenient choice from the orthonormal chart R used in the quantitative variational proof; no identity between them is needed.

The integrated theorem defines closed nested tubes using endpoint-coordinate radii and risk sublevels. They can be chosen compactly inside the common smooth/routing/rank/capacity region, with T1 contained in the interior of T2. Thus

    d_*=dist_infinity(T1,complement int T2)>0.

On the simultaneous error event, the proof proceeds without circular containment:

1. The initial cone lies inside T0, hence T1.
2. From a boundary in T1, the universal raw Adam bound and `H K_Adam eta0<d_*` keep every next intermediate phase and the next boundary inside T2.
3. Until that next boundary, coefficient/observable extensions agree with the actual CNN, the pi drift is zero, and risk dissipation is nonnegative.
4. The cumulative observable identities bound the endpoint-coordinate change from the **original** initial point by e_pi and the risk increase from that same initial point by e_E. Since these budgets are smaller than the T0-to-T1 gaps, the next boundary is back in T1.
5. Induction excludes an exit for every task and every optimizer phase.

The induction does not restart a probability estimate each task and does not accumulate an error budget once per task. It uses one all-times remainder event. There is no future boundedness or all-time same-sign update assumption.

The uniform initial mean-to-endpoint margin Delta and `||u||_1 e_pi<Delta/2` then give the actual final mean decrease. The statement is correctly per initial point with common constants, not one noise event claimed to control an uncountable collection of initial states simultaneously.

## 8. Capacity, learning-rate, and scope checks

The three-copy capacity trace is correctly `(3/2)tr[(lambda I2+3K0)^(-1)K0']`. It is strictly positive for a nonzero PSD one-image derivative. Continuity in nearby inputs and parameters preserves both full and literal-self capacity signs at the **specified** positive ridge. The integrated theorem does not claim a neighborhood uniform over all ridge values tending to zero, nor PSD of the perturbed six-dimensional derivative.

The target mean uses the true data/spatial augmented patch mean. The constructed three inputs average exactly to the base image, so that vector u is unchanged. Positivity of its supported entries and corresponding J columns supplies the strict ray coefficient.

Actual learning rates are delta_k repeated H times, without state damping. Their sum diverges and squared sum converges for the declared exponent. Moments start at zero once and then carry through every iid-label task. All raw weights and biases are updated; the common binary-head component may remain constant by the loss's symmetry, which is not an imposed frozen block.

The final result is high-probability strict long-time net mean decrease in a local, full-dimensional open family with three specially constructed nearby images, binary output, full-batch H-fold label reuse, strict routing, and sufficiently small decaying rates. It is not unconditional expected sinking on exit paths, arbitrary CIFAR-data convergence, ten-class Adam, constant-rate training, negative mean, or gate death. These limits are preserved in the integrated text.

No blocking correction is required for the reviewed claim.
