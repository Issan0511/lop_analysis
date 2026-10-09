# Singleton shuffled minibatches: a finite stable Adam construction and a positive mean cone

2026-10-09. This note extends the local binary Adam mechanism from full-batch reuse to singleton minibatches. It gives an exact frozen linearization with reused labels, explains why entrywise sign alone is insufficient, and supplies a nonempty native all-bias scaling construction whose normal linearization is strictly stable. Its endpoint map then gives a finite full-dimensional cone of strict mean decrease. The actual stochastic transfer uses the already proved general taskwise observable averaging theorem, which permits phase-dependent minibatches.

No raw parameter is tied, frozen, reset, or projected. The scaling below selects initial parameter values only. The final admitted initial set is open in all raw coordinates.

## 1. Task law and frozen singleton gradients

There are N=3 images, binary half-contrasts z_n(theta), and contrast Jacobian J_(n,j)=partial_j z_n. Each task draws independent uniform labels Y_(k,n) in {-1,+1}. Each of E passes visits all N images once in a shuffled order; H=NE is fixed and finite. The same labels are reused for all passes of that task. Schedules are independent of labels and tasks, and their law is invariant under every permutation of the image names. Uniform independent shuffles in each pass meet this condition. Reusing one uniform shuffle within a task also meets the condition; independence between phase image indices is not required.

For the sampled image n, the raw gradient coordinate is

    g_j=s_(n,j)[tanh z_n-Y_n].                       (1.1)

There is no 1/N factor in a singleton loss. The factor 1/N appearing later comes from schedule exchangeability.

At a reference theta_* all z_n are zero. Assume a strict branch, every s_(n,j)!=0, and uniform nonzero sensitivity margins on a compact neighborhood. Then for small contrasts every possible singleton gradient has a positive absolute floor. Frozen stationary RMS values are consequently bounded away from zero, and the frozen expected Adam field is smooth in the independent variables z and the sensitivity matrix S=J. This statement includes every raw bias.

Let Fbar be the **sum** of stationary expected quotients over the H task phases. Define

    L=D_z Fbar(0,J_*),                              (P by N).

The linearization of the parameter field at equilibrium is L J_*. Derivatives with respect to S do not contribute, because flipping all past task labels gives Fbar(0,S)=0 for every S. This last identity remains true for shuffled schedules and reused labels.

## 2. Exact reused-label linearization

Fix a phase and condition on the complete schedule history. For one raw coordinate j, abbreviate s_n=s_(n,j). Group the first-/second-EMA weights by historical task q and image n:

    A_(q,n)>=0, B_(q,n)>=0,
    sum_(q,n) A_(q,n)=sum_(q,n) B_(q,n)=1.

Each A_(q,n), B_(q,n) sums all occurrences of that image in that task, including only already-observed phases of the current task. This grouping retains the repeated label. Put

    a_n=sum_q A_(q,n),
    b_n=sum_q B_(q,n),
    c_n=sum_q A_(q,n)B_(q,n),
    sigma^2=sum_n b_n s_n^2,
    d=sigma+epsilon.

At zero contrast the squared gradient is s_n^2, so sigma is schedule-dependent but label-independent. Differentiating the exact stationary quotient and averaging over the independent signs Y_(q,n) gives the phase derivative

    L_(j,n)^phase
      = E_schedule{ s_n [a_n/d-s_n^2 c_n/(sigma d^2)] }.       (2.1)

Indeed, the first numerator derivative is a_n s_n. The denominator derivative contains `-s_n^2 sum_q B_(q,n)Y_(q,n)/sigma`; pairing it with the zero-contrast numerator leaves only the same task/image covariance c_n. This is not an iid-per-step replacement.

Since all weights are nonnegative,

    c_n<=a_n b_n,
    b_n s_n^2<=sigma^2.

Consequently, pathwise in the schedule,

    a_n/d-s_n^2 c_n/(sigma d^2)
       >=a_n epsilon/d^2>0.                       (2.2)

Each image has positive total historical weight. Thus the phase derivative has the sign of s_n. Summing phases gives the same sign for L_(j,n).

These signs alone do not prove stability of J L: the image/coordinate coefficients need not factor as a common positive diagonal matrix times J^T. A nonsymmetric inverse or a rank-deficient continuity argument cannot be assigned a favorable sign solely from entrywise positivity. The following construction resolves that issue quantitatively.

## 3. A uniform small-sensitivity approximation, with finite error

Let S_j=max_n |s_(n,j)| for one coordinate. Exchangeability of the schedule implies E[a_n]=1/N at every phase. From (2.1),

    |L_(j,n)^phase-s_(n,j)/(N epsilon)|
        <=2 |s_(n,j)| S_j/(N epsilon^2).           (3.1)

To check this bound, subtract the leading numerator term `s_n a_n/epsilon`. Its absolute denominator error is at most `|s_n|a_n sigma/epsilon^2`. The covariance term is also at most that quantity because `s_n^2 c_n<=a_n sigma^2`. Since sigma<=S_j, taking E[a_n]=1/N proves (3.1).

The bound holds for every fixed finite H and every exchangeable task schedule; it does not require a favorable contrast-mode Taylor coefficient at coincident images.

Separate the two raw output biases from all other raw coordinates R. Their sensitivity columns are exactly `+1/2 * 1_N` and `-1/2 * 1_N`. Exchangeability makes their respective L rows `+ell_b * 1_N^T` and `-ell_b * 1_N^T`, with ell_b>0 by (2.2). Their contribution to J L is therefore

    J_bias L_bias=ell_b 1_N 1_N^T>=0.               (3.2)

No small-output-bias-sensitivity assumption is needed; their fixed magnitude is handled exactly.

Let J_R contain every other raw sensitivity column and let L_R contain the matching rows. Summing H phases, write

    L_R=H J_R^T/(N epsilon)+E_R,
    ||E_R||_F <= [2H S/(N epsilon^2)] ||J_R||_F,
    S=max_(n,j in R)|J_(n,j)|.                     (3.3)

It follows that

    A_*:=J L
      =ell_b 1_N1_N^T+H J_RJ_R^T/(N epsilon)+J_R E_R,

    lambda_min(sym A_*)
      >=H sigma_min(J_R)^2/(N epsilon)
         -[2H S/(N epsilon^2)]||J_R||_op||J_R||_F.  (3.4)

A directly checkable, noncircular sufficient condition is therefore

    2(S/epsilon)||J_R||_op||J_R||_F
         <sigma_min(J_R)^2.                        (3.5)

This certifies a strictly positive symmetric part, stronger than merely positive-real-part eigenvalues. It will be enforced at a finite parameter state below.

## 4. Native all-bias scaling with a full-rank normalized limit

Start from a native positive hidden CNN with strict ReLU/MaxPool routing, at least one last hidden affine/ReLU layer, and a free binary output head. All ordinary biases are included. Construct three distinct nearby images for which the unscaled last-hidden feature matrix H_0 has row rank three and a strictly positive head contrast d has equal `d^T h_(0,n)` across the images. The native three-image reference already supplies such a configuration.

Choose a finite number 0<tau<=1 and set initial values as follows:

- Multiply the last hidden affine layer's weights **and bias** by tau.
- Set the output-head contrast to tau d; use antisymmetric output rows as a convenient reference.
- Set the output-bias contrast to cancel the common output value, which is of order tau^2.
- Keep all earlier hidden initial weights/biases unchanged.

Positive ReLU homogeneity makes the last-hidden vector exactly `h_n=tau h_(0,n)` for every tau>0. The earlier routing is unchanged, and the last hidden ReLU remains strictly active. Every parameter is still a free raw coordinate; none of these scale relations is enforced after initialization.

The non-output-bias raw sensitivities split into disjoint column blocks:

    J_R(tau)=[tau J_a, tau^2 J_b].                  (4.1)

The tau block contains output-head weights and last-hidden affine weights/biases. The tau^2 block contains earlier hidden parameters. In particular, the free output-head block contains `+tau H_0/2` and `-tau H_0/2`, so J_a has row rank three. This is essential: the normalized rank does not collapse as tau tends to zero.

Let

    nu=sigma_min(J_a)^2>0,
    S_1=max|[J_a,J_b]|,
    B_1=||[J_a,J_b]||_op,
    F_1=||[J_a,J_b]||_F.

For 0<tau<=1,

    S(tau)<=tau S_1,
    ||J_R(tau)||_op<=tau B_1,
    ||J_R(tau)||_F<=tau F_1,
    sigma_min(J_R(tau))^2>=tau^2 nu.               (4.2)

Thus every finite choice

    0<tau<min(1, epsilon nu/[4 S_1 B_1 F_1])        (4.3)

satisfies (3.5) with at least a factor-two margin, and gives

    sym(J L)>= H tau^2 nu/(2N epsilon) I_N>0.      (4.4)

If a denominator constant happens to vanish, the nonzero rank already rules out the problematic zero product here. All constants are from a fixed finite unscaled reference. This is an explicit nonempty finite interval, not a theorem only at tau=0. For tiny epsilon and nearly dependent images it can be extremely conservative, but every admitted tau is positive.

Every raw coordinate has a nonzero singleton gradient floor at each such fixed tau. Earlier hidden coordinates can have sensitivity of order tau^2 and thus small floors, but not zero floors. They remain trained. Strict sensitivity, rank, branch, and stability margins persist in a full-dimensional open neighborhood of the chosen finite state.

### 4.1 Compatibility with original full/self capacity signs

For a prescribed finite ridge lambda>0, compatibility can be established without a circular choice of image spacing and tau. On the reference branch, the full raw logit Jacobian has disjoint nonbias blocks of orders tau and tau^2, while the output-bias block is constant. Hence

    K(tau)=K_bias+tau^2 K_2+tau^4 K_4,
    D_u K(tau)=tau^2 K_2'+tau^4 K_4'.               (4.5)

The same holds for the literal channel-isolated architecture. At identical images, the positive one-image polynomial construction gives a strictly positive ridge trace for K_2': its free output-weight contribution alone has a strictly positive scalar-image derivative, and the hidden mixed-derivative contribution is nonnegative.

For tau>0 the capacity slope divided by tau^2 is

    (1/2)tr{[lambda I+K_bias+tau^2K_2+tau^4K_4]^(-1)
                    [K_2'+tau^2K_4']}.

Its algebraic branch expression has a continuous limit at tau=0, with strictly positive value at the identical-image reference. This is a coefficient limit, not a claim about ReLU differentiability at a physically zero last layer. Therefore sufficiently nearby distinct full-rank images and all sufficiently small positive tau retain both full and literal-self capacity signs for the fixed ridge. First choose such a fixed nonzero image separation; then choose tau also satisfying (4.3).

No common neighborhood for every ridge lambda>0 is inferred from this argument. The independently verified finite trace/continuity certificate may be used instead at the actual chosen state.

## 5. Smooth nonlinear field and nonsymmetric normal stability

In independent variables (z,S), singleton gradients have the form `s_n(tanh z_n-Y_n)`. With all |s_nj| bounded below and z small, every possible gradient has an absolute floor. The schedule-conditioned stationary RMS then has a uniform lower bound, and all finite derivatives of the frozen expected quotient are uniformly bounded. The label/schedule law is fixed and finite per task. Consequently the phase-summed response R(z,S) is smooth.

At z=0, complementing every label in every task negates the whole numerator and leaves the denominator unchanged. Thus

    R(0,S)=0 for every S.

The vector Hadamard identity gives an exact smooth local factorization

    Fbar(theta)=C(theta) z(theta),
    C(theta)=integral_0^1 D_z R(t z(theta),J(theta)) dt,
    C(theta_*)=L.                                  (5.1)

This is not the diagonal full-batch formula. In general its columns and image couplings do not coincide with `D J^T`.

By (4.4) and continuity, on a small compact neighborhood

    sym[J(theta)C(theta)]>=a I_N, a>0.             (5.2)

The averaged ODE obeys

    theta_dot=-C(theta)z,
    z_dot=-A(theta)z,  A(theta)=J(theta)C(theta).

Although A is generally nonsymmetric, (5.2) gives directly

    (1/2)d||z||_2^2/dt<=-a||z||_2^2.               (5.3)

Thus z decays exponentially, and the parameter speed is O(||z||). A finite path-length and first-exit argument gives a local equilibrium endpoint pi(theta) on M={z=0}.

## 6. Elementary C2 endpoint extension to nonsymmetric A

The [earlier quantitative vector-normal lemma](adam_three_images_equilibrium.md) applies with the following explicit replacements. Use a smooth coordinate chart `(y,z)` and write

    y_dot=-B(y,z)z,
    z_dot=-A(y,z)z,
    sym A>=aI.

Assume finite coefficient bounds A_0,B_0,A_1,B_1,A_2,B_2 as in that lemma. The base normal decay rate is now lambda=a, because q(z)=z exactly. There is no cubic tanh remainder.

Choose a small normal radius delta so that

    A_1 delta<=a/2,
    nu=a/2,
    Q_0=B_0+B_1 delta,
    kappa=(delta/lambda)[B_1+Q_0 A_1/nu]<=1/2.       (6.1)

The first variational normal block is `S=-A-(D_z A)[.]z`. Its symmetric part is at most -nu I; symmetry of A itself is unnecessary. The first-variation estimates are therefore unchanged in form:

    C_Y=2(1+Q_0/nu),
    C_Z=1+A_1 delta C_Y/(lambda-nu).

For the second variations, `Dq=I` and `D²q=0`. With C_V=C_Y+C_Z one may take

    K_B=B_2 delta C_V^2+2B_1 C_V C_Z,
    K_A=A_2 delta C_V^2+2A_1 C_V C_Z,
    C_Y2=2[K_B/nu+Q_0 K_A/nu^2],
    C_Z2=A_1 delta C_Y2/(lambda-nu)+2K_A/nu.         (6.2)

The same integrable derivative bounds prove uniform convergence of the flow and its first two derivatives. Thus pi is C2 with the same raw chart/chain-rule derivative bounds. Its semigroup identity gives `Dpi Fbar=0`; no assertion of integrability of all columns of C is needed.

At equilibrium set A_*=J_*L. Linearizing the parameter ODE gives `delta theta_dot=-L J_* delta theta`. Since sym A_*>0,

    delta theta(t)=v-L A_*^(-1)[I-exp(-A_* t)]J_*v,
    Dpi(theta_*)=I-Q_*,
    Q_*=L A_*^(-1)J_*.                            (6.3)

Q_* is a projection onto range L along ker J_*. Its range has dimension N because J_*L is invertible. Neither A_* nor Q_* needs to be symmetric.

## 7. A positive mean ray without a nonsymmetric inverse-sign claim

Let the actual first-Conv mean have its fixed augmented-patch gradient u. Its supported entries are nonnegative and the corresponding sensitivities s_(n,j) are strictly positive. By (2.2), every supported L_(j,n) is strictly positive. Hence

    r_*=L^T u!=0,
    v_*=L L^T u,                                  (7.1)

is a nonzero current-reference direction. It lies in range L. Therefore

    Q_*v_*=v_*,
    Dpi(theta_*)v_*=0,
    u^T Q_*v_*=u^T v_*=||L^T u||_2^2=:b_*>0.      (7.2)

The direction can change all raw initial coordinates. It is constructed solely from the frozen reference linearization and the true mean observable. It does not assume a favorable realized momentum, a desired final displacement, or positivity of entries of A_*^(-1).

Let `L_m(theta)=m(theta)-m(pi(theta))` and ell=u^T Q_*. On a convex smooth reference ball, let C_m bound the entry-sum norm of its Hessian. Choose

    0<q<b_*/(4||ell||_1),
    M_v=||v_*||_infinity+q,
    0<t_L<t_U<b_*/(2 C_m M_v^2),                   (7.3)

with the last restriction omitted if C_m=0 and with the usual chart-radius restriction. For every

    theta0=theta_*+t(v_*+e),
    t_L<t<t_U,  ||e||_infinity<q,

Taylor's theorem gives

    m(theta0)-m(pi(theta0))>b_*t/2
                          >=Delta:=b_*t_L/2>0.     (7.4)

This is a finite full-dimensional open cone. Every raw weight and bias can vary independently within its perturbation balls. The desired endpoint sign is a conclusion of a squared norm and a finite remainder bound.

## 8. Actual shuffled-Adam transfer

The generic taskwise observable averaging theorem already permits phase-dependent gradients and a task innovation containing both the fixed label assignment and the shuffled schedule. Its exact full-batch stationary sign theorem is not used here. Instead, Sections 2–7 supply the required local field and geometry.

Use all ambient components of pi and the normal energy

    E(theta)=||z(theta)||_2^2/2.

Their averaged drifts satisfy

    Dpi Fbar=0,
    DE Fbar=z^T[J C]z>=a||z||_2^2=2a E.             (8.1)

Observable averaging gives convergent remainders and one simultaneous uniform high-probability budget. Nested compact tubes in endpoint coordinates and E, with a reserved within-task distance, then give the same nonexit bootstrap as in the three-image full-batch theorem:

- a boundary lies in the middle tube;
- `H K_Adam delta0` is smaller than the distance to the outer tube's complement, so every phase and the next boundary remain in the outer tube;
- the cumulative pi and energy identities put the next boundary back in the middle tube.

On this event E tends to zero, every pi coordinate converges, and the complete raw parameter vector converges to M. Choose the pi error smaller than `Delta/(2||u||_1)` to obtain actual strict limiting first-Conv mean decrease at least Delta/2. The initial geometric margins and averaging thresholds are uniform on a compact slice of the open cone.

The actual scalar learning rate may be `delta_k=delta0(k+1)^(-p)` repeated across each task, with 1/2<p<=1. Its sum diverges and its squared sum converges. Moments and global bias correction carry through all shuffled epochs and tasks; only the initial moments are zero. No state-dependent damping is needed in this local bounded construction.

## 9. What this does and does not resolve

This gives a nonempty finite all-bias native-CNN route for true singleton shuffled minibatches, independently uniform binary labels assigned once per task, finite H-fold reuse, retained ordinary Adam moments, and a full-dimensional open initial set. It resolves the rank-collapse obstruction by controlling the normalized nonbias sensitivity matrix and treating the non-small output-bias block exactly.

It does not claim that every positive sensitivity geometry yields stable J L. Exchangeable shuffling is used essentially to obtain E[a_n]=1/N and the exact common output-bias block. The construction can require very small head/last-hidden scales and very small learning rates. All chosen values are finite and positive; practical training rates or times are not certified by this argument.

The result remains binary, local, and based on specially constructed nearby images and strict routing. General multiclass Adam, arbitrary CIFAR data, changing gates, constant learning rates, negative mean, and neuron death remain outside its scope. A high-probability pathwise final sink is not an unconditional expectation statement about the exceptional exit paths.
