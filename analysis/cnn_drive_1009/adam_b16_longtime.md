# Batch 16 with new epoch partitions: a local actual-Adam long-time theorem

2026-10-09. This removes the singleton-batch restriction from the preceding native CNN construction. Every epoch draws a new uniform permutation and regroups images into batches of sixteen. Labels are reused within each task, and Adam retains its moments across tasks. The result remains a local existence theorem with binary outputs, specially constructed images, two epochs per task, and sufficiently small decaying rates. It is not the standard many-image, ten-class, 400-epoch, constant-rate RL-CIFAR theorem.

## 1. Precise statement

Fix N=32 or N=48, B=16, E=2, H=EN/B (respectively 4 or 6), a finite ridge lambda>0, beta1=.9, beta2=.999, epsilon=1e-8, and 1/2<p<=1. There exist N distinct positive images and a full-dimensional open initial subset of the independent raw parameters of

    RGB32x32 -> Conv5/pad2(16), ReLU, MaxPool2
    -> Conv5/pad2(16), ReLU, MaxPool2
    -> FC100, ReLU -> FC100, ReLU -> binary FC2,

with all ordinary biases present, with the following property. At each task, independently assign each image a uniform binary label. Independently shuffle all images in each of the two epochs and divide each order into N/16 batches. Use the mean cross entropy of the sixteen examples. Update every raw parameter with ordinary Adam, initially zero moments, retained moments between tasks, and the continuing global bias-correction counter. Use the deterministic rate

    eta_(k,r)=delta_k=eta0/(k+1)^p,  r=1,...,H.

For any prescribed alpha in (0,1), a sufficiently small eta0>0 gives, for each initial point in a fixed smaller open initial slice, probability at least 1-alpha of all the following:

- every parameter converges to a finite limit, with all N class contrasts tending to zero;
- the actual spatial/data average of the selected first Conv preactivation has strictly negative net displacement, bounded away from zero uniformly on the initial slice;
- both the original full and literal-self raw-NTK capacities have positive derivatives in that mean direction, for the fixed ridge, at every iterate and the limit;
- the required ReLU and MaxPool branches, sensitivity, rank, stability and capacity margins hold at every optimizer step;
- the learning rates have divergent sum and convergent squared sum.

The constants are common to that initial slice; the probability guarantee is per initial point, not one event for uncountably many starting states. ReLUs remain active. The theorem does not assert a negative final mean, gate death, update-by-update monotonicity, monotonicity of the capacity value under all raw updates, or unconditional expected displacement including exits. It uses no moment reset, projection, frozen block, weight tying, exact head refit or state-dependent damping.

## 2. The stationary field and the new batch difficulty

Put z_n=(f_(n,+)-f_(n,-))/2 and J=D_theta z. For independent output and sensitivity variables (z,S), a raw batch gradient is

    g_j=(1/B) sum_(n in batch) s_nj [tanh(z_n)-Y_n].

Let Fbar be the sum, over H phases, of the infinite-history stationary expected Adam quotient at a frozen state. Whole task innovations consist of all N labels and both permutations. Different tasks are independent; gradients within one task are generally dependent. Complementing every historical label negates the numerator at z=0 and preserves the denominator, so Rbar(0,S)=0 for every S.

Unlike singleton batches, a balanced batch can have zero gradient. A deterministic RMS floor is false. Two separate arguments replace it: a first-derivative bound that also works at RMS zero, and an inverse-moment proof of smoothness of the stationary expectation. Neither deletes balanced batches.

## 3. The non-output-bias response and the exact bias sign

At a zero-output reference write L=D_z Rbar(0,J), a P-by-N matrix. The [reference proof](adam_b16_reference.md), sections 1-2, derives

    |L_jn-c0 s_nj| <= c0 C_B (gamma_j/epsilon)|s_nj|,
    c0=H/(N epsilon),  C_B=1+sqrt(N/B),
    gamma_j=max_n |s_nj|.                                      (3.1)

For one stationary history, q=m/(sigma+epsilon) obeys

    |Dq-Dm/epsilon| <= (gamma_j/epsilon^2)(|Dm|+||Dg||_(l2(b))).

At sigma=0, m=0 and the derivative is Dm/epsilon, with a quadratic remainder. For the z_n derivative, the numerator and RMS-history derivative have respective factors `(s_nj/B) A_n` and `|s_nj| sqrt(B_n)/B`, where E A_n=E B_n=B/N. Jensen gives the factor C_B. This keeps both the mean-loss factor 1/B and sample inclusion B/N, without any independence between gradient numerator and denominator.

The two raw output-bias sensitivities are +/-1/2. Their positivity is proved separately, not by treating them as small compared with epsilon. Let X be the mean label of a batch of sixteen, and put

    A=EMA_beta1(X), B0=EMA_beta2(X), V=EMA_beta2(X^2), sigma=sqrt(V).

At any stationary phase the response to a common contrast displacement is

    Gamma_r=E[1/(sigma+2epsilon)
                 -A B0/{sigma(sigma+2epsilon)^2}].             (3.2)

The [bias proof](adam_b16_bias.md) retains all within-task dependence. Grouping EMA weights by independent tasks gives, with T_i=(1-beta_i^H)/(1+beta_i^H),

    E A^2 <= T1/16, E B0^2 <= T2/16,
    P(V<1/64) <= exp[-9/(92 T2)].

The latter uses E X^2=1/16 and E X^4=46/4096 and bounds each task's second moment before multiplying independent task Laplace transforms. The partial current task is included. Weighted Cauchy gives |A|<=sqrt(370/7) sigma<8 sigma. Splitting (3.2) at V=1/64 yields

    Gamma_r > 1/(1/4+2epsilon)
                  -32 sqrt(T1 T2)-exp[-9/(92 T2)]/epsilon.

Exact rational bounds prove Gamma_r>29/10 at every phase for both H=4 and H=6. Exchangeability of the true reshuffle law therefore gives exactly

    J_bias L_bias = beta_b 11^T,
    beta_b=(sum_r Gamma_r)/N >29H/(10N)>0.                     (3.3)

The two +/-1/2 Jacobian columns cancel the apparent missing factor of two. Bounds and normalizations are checked by [exact arithmetic](verify_adam_b16_certificates.py). This positive bias certificate is not asserted for arbitrary task duration.

## 4. Nonempty native reference and a finite stable normal matrix

The [native construction](adam_b16_reference.md), sections 3-5 and 7, provides a strict positive branch whose final hidden input map h(x)=Ax+c has rank at least N. The two MaxPools leave 64 selected spatial input directions in RGB32x32. Positive near-selector FC rows use N of them; the remaining entries can all be made strictly positive while preserving the nonzero minor and strict full/self branches.

For independent selected input columns b_i=Ae_i and a positive head contrast d, put beta_i=d^T b_i>0 and

    T=(I-11^T/N) diag(1/beta_i),
    x_n=x0+delta sum_i T_ni e_i.

Since T beta=0 and 1^T T=0, these images share the head-contrast level and preserve the mean input. For every sufficiently small delta>0 they are distinct, positive, and on the same strict full/self branches; their feature matrix has row rank N. A common output bias cancels all contrasts. Shrinking delta also makes every sensitivity relatively close to its nonzero base-image value.

Scale the last hidden FC weights and biases by t>0, the output-head contrast by t, and its canceling bias contrast by t^2. These are initial values only. The non-output-bias Jacobian has disjoint raw column groups

    R(t)=[t J1,t^2 J2],  lambda1=lambda_min(J1 J1^T)>0.

The free output-head block supplies full row rank to J1. If Gamma bounds entries of J1,J2 and C^2=||J1||_F^2+||J2||_F^2, (3.1)-(3.3) imply

    Sym(JL) >= c0 t^2[lambda1-C_B t Gamma C^2/epsilon] I.

Thus any finite

    0<t<min(1, epsilon lambda1/(2 C_B Gamma C^2))               (4.1)

gives a strict positive normal margin. More generally, the finite reference test is

    C_B(max|R|/epsilon)||R||op||R||F < sigma_min(R)^2.          (4.2)

Conditioning of the distinct feature rows is part of this inequality. No uniform gap in the coincident-image limit is claimed.

For either full or literal-self original capacity, with every retained raw bias included,

    K=K_bias+t^2 K1(delta)+t^4 K2(delta),
    D_u K=t^2 K1'(delta)+t^4 K2'(delta),
    K_bias=11^T tensor I2.

The slope divided by t^2 extends continuously to t=0 as a positive-branch coefficient formula. At coincident images the free head block contributes `2<h,D_u h>I2>0` and the last-hidden-FC contribution is positive semidefinite. The resulting trace is strictly positive for fixed lambda. Joint continuity first permits one fixed small nonzero delta, retaining rank N, and then a sufficiently small positive t satisfying (4.1) and both capacity signs. This does not differentiate the actual ReLU at t=0. Every strict condition persists on a full raw neighborhood of the finite reference.

## 5. C3 of the expected field without a gradient floor

The [regularity proof](adam_b16_regularity.md) supplies a uniform probabilistic substitute for the floor. For each raw coordinate choose c_j!=0 and a compact product neighborhood of (z,S) such that

    |s_nj/c_j-1|<=e_j, |tanh z_n|<=mu,
    e_j+(1+e_j)mu<=1/B.

The image construction and a sufficiently small neighborhood give this condition for all raw coordinates, including negative output-row sensitivities and ordinary biases. In any batch with unequal label counts, |g_j|>=|c_j|/B simultaneously for every point of this neighborhood. The probability that the first batch of a task fails this test is p_B=6435/32768. Different tasks provide independent trials.

The last such good completed task gives a uniform random positive lower bound on every coordinate's stationary RMS. If its backward task index is T and the current phase is r, then

    sigma_j >= (|c_j|/B) sqrt(1-beta2) beta2^((TH+r-1)/2).

T is geometric, so E sup sigma_j^(-q)<infinity whenever p_B<beta2^(Hq/2). The exact certificate verifies q=8 for H=4,6. An optional sharper whole-task calculation gives failure probability

    p_N [binom(B,B/2)^(N/B)/binom(N,N/2)]^E,
    p_N=binom(N,N/2)/2^N,

and proves this inverse moment for any finite E; that stronger smoothness fact does not extend the separate bias sign to arbitrary E.

Representing sigma as the norm of the weighted gradient history gives bounded first derivatives and second/third derivative bounds involving at most sigma^(-1)/sigma^(-2). For fixed epsilon>0, three quotient differentiations yield

    ||D^3 q|| <= C0+C1/sigma+C2/sigma^2.

The explicit constants and the uniform compact domination are in the companion note. Thus the inverse-second moment suffices; the inverse-eighth moment already proved is more than enough. Dominated differentiation and continuity establish a C3 stationary expectation in independent (z,S). No finite realized optimizer map at RMS zero is asserted to be C3.

## 6. Endpoint, positive mean cone, and actual Adam

Since Rbar is C3 and Rbar(0,S)=0, Hadamard's formula gives

    Fbar(theta)=V(theta) z(theta),
    V(theta)=integral_0^1 D_z Rbar(q z(theta),J(theta)) dq,
    V(theta_*)=L,

with V C2. In a small compact neighborhood, Sym(JV)>=aI for some a>0. Therefore the averaged ODE has `z_dot=-JV z` and `d||z||^2/dt<=-2a||z||^2`, while full parameter speed is O(||z||).

Sections 5-6 of the [nonsymmetric endpoint proof](adam_singleton_geometry.md) apply in arbitrary finite output dimension: their hypotheses are C2 coefficients, positive symmetric normal part, and finite local derivative bounds, not the singleton gradient formula. The normal variational decay and finite-radius absorption produce a C2 local endpoint pi with

    pi|{z=0}=id, Dpi Fbar=0,
    Dpi(theta_*)=I-Q, Q=L(JL)^(-1)J.

Let u be the actual fixed mean-input-patch/bias direction for the selected first filter. Its supported coordinates and sensitivities are positive. The stronger choice (4.1) makes the relative error in (3.1) less than 1/2, hence L^T u is componentwise positive. The derived current-reference ray

    v=LL^T u, Qv=v, u^T Qv=||L^T u||^2>0

produces, by a finite C2 Taylor remainder, a full-dimensional open cone of initial points with

    m(theta0)-m(pi(theta0))>=Delta>0, m(theta)=u^T theta.

The cone is defined from the stationary field and finite derivative bounds; it does not assume the eventual stochastic sink sign. Its perturbations include every raw coordinate. See the explicit cone inequalities in section 5 of [the singleton integrated theorem](adam_singleton_longtime.md), replacing that theorem's L,J,u by the batch16 values above.

The general [taskwise averaging theorem](adam_task_averaging.md) allows phase-dependent batch gradients and an iid innovation containing labels and both permutations. The [observable version](adam_observable_averaging.md) is applied separately to every coordinate of pi and to E_z=||z||^2. It yields convergent cumulative remainders. As long as preceding task boundaries and intermediate steps are in the outer tube, the exact task-boundary identities reduce to

    pi(theta_K)=pi(theta0)+R_K^pi,
    E_z(theta_K)=E_z(theta0)
       -sum_(k<K) delta_k 2 z(theta_k)^T J(theta_k)V(theta_k)z(theta_k)
       +R_K^z.

For sufficiently small eta0, the probability that all remainder suprema satisfy prescribed positive budgets is at least 1-alpha. This includes initial zero moments, continuing global bias correction, all earlier task memory, changing parameters, and within-task reuse. The tracking estimates use the RMS norm inequality and epsilon>0 and remain valid when actual finite-history second moments are zero. Smoothness of the realized Adam update is not required. The global auxiliary identities retain their observable drift terms; Dpi Fbar=0 and the nonnegative energy drift are used only at preceding boundaries inside the outer tube, one induction step at a time.

Choose an ambient coordinate-selection matrix T with P-N rows and ||T||_infinity=1 such that (T pi,z) is a local chart. Choose nested compact tubes with endpoint radii rho0<rho1<rho2 and squared-output radii e0<e1<e2, with the largest tube inside all strict conditions above. Put the initial cone inside the smallest tube. For positive distance d_* from the middle tube to the complement of the largest, choose budgets

    e_pi<min(rho1-rho0, Delta/(2||u||_1)),
    e_z<e1-e0, H*73*eta0<d_*.

The ordinary raw Adam step bound keeps every intermediate phase in the outer tube. The cumulative identities and nonnegative averaged z^2 drift return each task boundary to the middle tube. Induction proves no exit on this event, so the smooth extensions used in averaging coincide with the actual CNN. E_z converges and its nonnegative dissipation sum is finite; since sum delta_k diverges and the drift is at least 2a E_z, its limit is zero. All pi coordinates converge, so the full parameter vector converges in the chart. Intermediate steps converge to the same limit. Finally,

    m(theta0)-m(theta_infinity)>=Delta-||u||_1 e_pi>Delta/2.

This proves the statement without bounding total absolute stochastic path length or requiring same-sign individual updates.

## 7. Verification and scope

[Exact rational certificates](../../results/cnn_drive_1009/adam_b16_certificates.json) check all constants for the output-bias and inverse-moment arguments. [The native verifier](verify_adam_b16_native.py) separately checks an RGB24x24, Conv5(2)/Conv5(2), FC32/32 witness with 3,712 free raw parameters, N32 distinct images, B16, E2 and every bias. It verifies the mean direction, rank, strict routing, relative spectral inequality, both original capacity slopes and the raw CE formula on sixteen actual batches from four fixed label vectors and two separately shuffled epochs. It includes balanced batches. This finite check does not enumerate 2^32 label assignments, estimate a stationary response, train a long trajectory, or compute a probability-certified learning rate. Mathematical nonemptiness comes from the native construction and strict inequalities; float64 margins are accompanying identity checks.

The genuinely new scope is batch size sixteen with fresh partitions each epoch and all raw moving Adam parameters. Binary classification, selected nearby images, short tasks, a very small finite layer scale, strict local routing, and sufficiently small decaying rates remain essential restrictions of this proof. Other activations are not studied. Standard RL-CIFAR with arbitrary images, ten classes, 400 epochs per task and constant learning rate remains unresolved.

Regarding the role of ReLU: its positive branch is exactly the identity, and that branch is preserved throughout the successful trajectories. Its fixed derivative and positive scaling make the affine input construction and layer-scale formulas particularly direct. The sign still requires the CNN Jacobian, the actual stationary Adam response and the derived endpoint cone; it is not a general consequence of choosing ReLU. The negative-input cutoff is never reached in this construction. Thus this proves a strict drop between positive means in the stated family, not a ReLU death mechanism or a comparison with another activation.

Independent audits passed for the [bias response and batch relative bound](adam_b16_bias_review.md), [expected-field regularity](adam_b16_regularity_review.md), [native reference/verifier](adam_b16_native_review.md), and [integrated endpoint and actual-process theorem](adam_b16_longtime_review.md).
