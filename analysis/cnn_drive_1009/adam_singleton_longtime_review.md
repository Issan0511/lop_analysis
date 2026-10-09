# Independent mathematical review: shuffled singleton Adam long-time theorem

Date: 2026-10-09. Reviewer: cnn_sign_theorem agent. Result: **PASS; no blocking mathematical issue found in the reviewed scope.**

Reviewed:

- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/adam_singleton_longtime.md`, sections 1–7.
- `/tmp/cnn_adam_minibatch_geometry_1009.md`, especially sections 5–8.
- The integrated singleton reference, and the quantitative multi-output endpoint and taskwise observable-averaging lemmas used by the proof.

This was a mathematical audit. I did not rerun the numerical verifier or treat its small-model checks as a proof of the native-width construction. The numerical values in section 7 remain the responsibility of the independent verifier audit.

## 1. Exact reused-label linearization

The phase derivative in section 2 is correct. At zero logits, conditional on the full image schedule, the RMS is label-independent. Aggregating EMA weights by `(task, image)` is essential: the covariance term is the product of the **aggregated** first-/second-moment weights, summed over those independent label groups. It includes cross-products between repeated occurrences of one task label. The displayed formula correctly keeps these terms.

The derivative of the RMS contributes with a minus sign to the quotient derivative. The final bracket is

    a_n/(sigma+epsilon)
      -s_nj^2 c_n/[sigma(sigma+epsilon)^2].

The bounds `c_n<=a_n b_n` and `s_nj^2 b_n<=sigma^2` imply the stated lower coefficient bound. Schedule exchangeability gives `E a_n=1/3` at every phase, so summing H phases introduces `H/3`, with no missing singleton-loss factor.

The field is identically zero at z=0 for every sensitivity array. Thus sensitivity-derivative terms vanish there, and the full raw derivative is `LJ`; the normal derivative is `JL`. No Hessian-of-logit term is missing.

## 2. Finite normal-spectrum certificate

The estimate

    |L_jn-c0 s_nj| <= 2c0(gamma_j/epsilon)|s_nj|

follows directly by bounding the numerator-denominator discrepancy and the label-covariance correction separately. Their respective errors are each at most `|s_nj| a_n sigma/epsilon^2` before averaging.

Output-bias sensitivities are exactly +1/2 and -1/2. Their constant magnitudes make their RMS exactly 1/2, while shuffle exchangeability makes each expected response row constant across images. The two bias columns therefore contribute a positive scalar times `11^T`, with the stated factor. The proof correctly avoids approximating these large sensitivities by an epsilon-dominated formula.

For the remaining columns, the Frobenius estimate and submultiplicativity give

    lambda_min(Sym JL)
      >= c0[sigma_min(R)^2
             -2(gamma_R/epsilon)||R||op ||R||F].

This proves the advertised strict symmetric-part margin. It is stronger than a positive-real-part eigenvalue test and is sufficient for every contraction argument that follows.

The raw scaling uses disjoint parameter column groups, so `R=[tJ1,t^2J2]` implies `RR^T=t^2J1J1^T+t^4J2J2^T`. Consequently the rank lower bound does not need a cancellation-prone Weyl estimate. The explicit threshold in section 3 leaves at least a factor-two margin, and its image-conditioning factor lambda1 is retained. There is no hidden passage from rank-one coincident images to a uniformly conditioned rank-three limit.

The capacity extension distinguishes image separation from final-layer scale, divides the capacity slope by t^2, and uses continuity of the finite-ridge expression. This avoids assuming an unscaled positive capacity margin persists uniformly as t tends to zero. At a fixed finite reference, strict full/self capacities can be included in the outer neighborhood used later.

## 3. Smoothness and the nonsymmetric C2 endpoint lemma

At a fixed t>0, all raw singleton sensitivities are nonzero in the reference construction. Shrinking to a compact strict-routing neighborhood gives a positive absolute gradient floor for each coordinate, for both labels and all sampled images. Thus the frozen RMS has a positive floor, and differentiating its stationary quotient under the expectation is valid at every finite order needed here. The independently parameterized response in `(z,S)` is important: its vanishing at z=0 yields the smooth Hadamard factorization `Fbar=V(theta)z` without dividing a vector field by individual vanishing outputs.

The geometry note's replacement of the previous fullbatch lemma is valid. In coordinates `(y,z)` the equations are

    y_dot=-B(y,z)z,
    z_dot=-A(y,z)z,
    Sym A>=aI.

Therefore the base normal decay rate is lambda=a. The normal first-variation block is

    S=-A-(D_z A)[.]z.

Its symmetric part is at most `-(a-A1 delta)I`, so `A1 delta<=a/2` supplies the fundamental-solution bound with nu=a/2 even though A is not symmetric. The tangent and normal coupling bounds are exactly those in the earlier integral-absorption argument. The displayed kappa condition absorbs the tangent supremum on every finite horizon and then uniformly on the infinite horizon.

For second variations, differentiating `-C(y,z)z` produces

    -D^2C[v,w]z -DC[v]w_z -DC[w]v_z.

There is no `C D^2(tanh z)` term because q=z. Every forcing term contains either the decaying base z or a decaying normal first variation. The geometry note's KA, KB, CY2, and CZ2 are therefore valid conservative constants. The stable convolution and the resonant `t exp(-nu t)` estimate give integrable second-derivative tails. Uniform convergence of the flow and its first two derivatives supplies a C2 endpoint, including the stated chart-chain-rule bounds.

The equilibrium derivative follows from integrating `exp[-(JL)t]`:

    Dpi=I-L(JL)^(-1)J.

Positive symmetric part ensures exponential decay and invertibility; symmetry itself is not required. The resulting Q is a projection onto range L along ker J.

## 4. Exact positive mean ray and finite open cone

The true first-Conv affine mean has fixed raw gradient u. At the constructed reference, its supported entries and sensitivities are positive; the exact coefficient lower bound makes `L^T u` componentwise positive. Hence

    v*=L L^T u,
    Qv*=v*,
    u^TQv*=||L^T u||^2>0.

This is valid for nonsymmetric JL and avoids assigning a sign to its inverse. The ray may change many raw blocks, consistently with the claimed full-dimensional initial set.

The finite cone calculation is correct with the stated norms. Its linear loss is at least `3b*r/4`, and its quadratic remainder is smaller than `b*r/4`, giving the claimed `b*r/2` lower bound. The union over a finite positive interval of open infinity-balls is open in the entire raw parameter space. A compactly contained slice permits one collection of constants and one rate threshold to serve all admitted initial points. The probability statement remains pointwise in the initial state, as the theorem explicitly says.

## 5. Actual Adam transfer and permanent containment

The task innovation consists of the three fresh labels and all E permutations. These innovations are iid across tasks, while phase gradients are deterministic functions of the task innovation and current state. This fits the general averaging theorem; it does not fit an iid-new-label-per-optimizer-step theorem. The proof uses the former and correctly avoids the latter.

The observables are all ambient pi components and `E_z=||z||^2`. Each receives its own frozen reward and Poisson/martingale decomposition. The already proved observable theorem supplies convergent remainder series and a simultaneous uniform probability budget that tends to zero with eta0. Its Taylor costs are finite because the local endpoint is C2; bounded smooth/Lipschitz extensions are available outside the outer compact tube. The finite history, global bias correction, and carried moments are included in the tracking bound.

The nested-tube induction handles intermediate phases, not only task boundaries. If the boundary is in the middle tube, `H*73*eta0<d_*` keeps all H updates and the next boundary in the outer tube. The cumulative identities then put the next boundary back in the middle tube. The energy drift is nonnegative throughout that outer tube and endpoint coordinates move by at most their uniform error budget. This proves permanent agreement between the auxiliary extended process and the actual unprojected CNN on the good event.

Convergence is also valid. The nonnegative cumulative dissipation is bounded because E_z is nonnegative and its remainder converges. Thus E_z converges. If its limit were positive, the lower drift `2a E_z` and divergent sum of delta_k would make dissipation diverge. Hence z tends to zero. Every pi component converges, and the local endpoint chart implies convergence of all raw parameters. The phase displacement bound tends to zero, so the full optimizer-step sequence has the same limit. The final affine-mean inequality uses the correct infinity/one-norm duality.

## 6. Scope and nonblocking clarifications

No additional hypothesis is needed beyond the explicitly chosen compact geometry and the cited averaging conditions. For a fully standalone reading, the outer compact tube should be understood as lying wholly inside **all** strict routing, nonzero-sensitivity, full-rank, symmetric-part, and capacity-margin neighborhoods, with the smooth extension agreeing on a slightly larger neighborhood. This is available by shrinking around the fixed positive-scale reference and is already the intended construction; it is not a new invariant-trajectory assumption.

The theorem's restrictions matter: binary outputs, three specifically constructed nearby distinct images, finite fixed reuse H, image-exchangeable shuffled scheduling, a very small but positive scale relative to epsilon, sufficiently small deterministic power-decaying rates, and a local open initial set. It proves a high-probability finite net sink in the affine mean, not negative mean, gate death, monotonicity of every update, arbitrary initialization, or an unconditional expectation after exit. Those limits are stated correctly.

No change to repository files was made during this review.
