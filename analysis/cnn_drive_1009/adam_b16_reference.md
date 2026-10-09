# Native batch-16 reference with a fresh shuffle every epoch

2026-10-09. This note constructs native all-bias CNN references for N=32 or N=48 distinct positive images, batch size B=16, and E=2 independently shuffled epochs per task. Thus H=4 or H=6 actual Adam updates reuse one iid binary label assignment. It proves a relative non-output-bias linear-response bound, including zero-RMS histories; a full-rank native reference; an explicit small-parameter stability condition; a noncircular mean-sinking ray; and the original full/literal-self capacity signs. The stationary output-bias sign and the C3/endpoint stochastic transfer are separate companion results and are identified as such below.

## 1. Model and what is averaged

Each task independently assigns uniform binary labels Y_n in {-1,+1} to the N images. Each epoch draws a new independent uniform permutation and regroups it into N/B consecutive disjoint batches of size B=16. The same labels are reused in both epochs. All schedules are independent of the label values. Every batch loss is the mean of its B per-image cross entropies.

For half-contrasts z_n=(f_{n,+}-f_{n,-})/2, let J_nj=partial_j z_n. At a zero-logit reference write s_nj=J_nj. For a batch I the raw coordinate gradient is

    g_j(z,S;I,Y)=(1/B) sum_(n in I) s_nj[tanh(z_n)-Y_n].  (1.1)

The stationary Adam field Fbar sums the expected quotients over all H task phases, retaining the full moment history. Labels are not redrawn between batches or epochs. Positive epsilon is fixed. All raw weights and biases train; there are no resets, ties, projections, or frozen coordinates.

## 2. A relative quotient derivative bound, even when RMS is zero

Fix one raw coordinate and one stationary history of batches/labels. Let a_l,b_l be the normalized first- and second-moment weights over past optimizer steps; each sequence sums to one. At z=0 put

    m=sum_l a_l g_l,  sigma=(sum_l b_l g_l²)^(1/2),
    gamma=max_n |s_n|.

Every batch average obeys |g_l|<=gamma, so |m|<=gamma and sigma<=gamma. For any perturbation of the logits,

    |Dq-Dm/epsilon|
      <= (gamma/epsilon²)(|Dm|+||Dg||_(l2(b))),     (2.1)
    q=m/(sigma+epsilon).

For sigma>0 this follows from

    Dq=Dm/(sigma+epsilon)-m Dsigma/(sigma+epsilon)²,
    |Dsigma|<=||Dg||_(l2(b)).

At sigma=0 every history gradient is zero, hence m=0, and q has first derivative Dm/epsilon. Its residual along a finite-dimensional perturbation is quadratic because both m and sigma are first-order small. Thus (2.1) holds there as well. No inverse-RMS moment assumption and no universal K_beta bound are needed for this first-derivative estimate. C2/C3 regularity is a different issue.

For a change in only z_n, define

    A_n=sum_l a_l 1{n in batch_l},
    B_n=sum_l b_l 1{n in batch_l}.

Then Dm=(s_n/B)A_n and ||Dg||_(l2(b))=|s_n|sqrt(B_n)/B. Each historical phase has sample-inclusion probability p=B/N. Consequently

    E A_n=p,  E B_n=p,  E sqrt(B_n)<=sqrt(p).

Summing (2.1) over the H phases gives the actual stationary logit-response matrix L, of shape P by N:

    |L_jn-c0 s_nj|
       <=c0 rho_j |s_nj|,
    c0=H/(N epsilon),
    rho_j=C_B gamma_j/epsilon,
    C_B=1+sqrt(N/B),  gamma_j=max_n|s_nj|.          (2.2)

The factor 1/B from each mean batch loss and the inclusion probability B/N are both included. For E=2, c0=E/(B epsilon)=1/(8 epsilon), whether N=32 or N=48. Here C_B is 1+sqrt(2) or 1+sqrt(3), respectively.

The estimate applies to reused labels and regrouped batches: it uses neither independence among history gradients nor independence between numerator and denominator. Differentiation under expectation at first order is justified by the deterministic bounds above. Central label complementation makes the field zero on z=0 for every sensitivity array. Therefore its raw-parameter linearization at an equilibrium is

    D Fbar(theta_*)=L J.                           (2.3)

In particular, (2.2) is a relative bound on each response entry. It is not an absolute epsilon-dominance error asserted to outweigh an arbitrarily small normal drift.

## 3. Native rank-N affine input map

Use the actual-size hidden architecture

    RGB32×32 → Conv5/pad2, 3→16, bias → ReLU → MaxPool2
    → Conv5/pad2, 16→16, bias → ReLU → MaxPool2
    → flatten1024 → FC100, bias → ReLU
    → FC100, bias → ReLU → binary FC2, bias.

Choose a positive strictly increasing base image x0 as in the previous reference. Temporarily use positive center-offset convolution weights, zero off-center weights, and positive biases. Both pools have strict bottom-right winners. The final 8-by-8 spatial grid has 64 independent selected original pixels (4r+3,4c+3), so at least N=32 or N=48 independent input directions are available in one color plane.

Select N of these spatial coordinates in the first N rows of FC1, and select those N outputs in the first N rows of FC2. All remaining FC rows and all hidden biases can be positive. The corresponding N-by-N derivative minor from the selected input pixels to the last hidden outputs is positive diagonal. Replace the zero off-center convolution weights and the zero selector-row FC entries by sufficiently small positive numbers. Strict full/self routing and the nonzero minor persist. Every spatial offset is now used, every channel overlaps, and all hidden FC weights and biases are positive. The 100-unit widths accommodate both choices of N.

On a common input branch the last hidden map is affine:

    h(x)=Ax+c,  rank A>=N.

Let e_1,...,e_N be the selected pixel directions, with independent b_i=Ae_i. Fix d>0 and beta_i=d^T b_i>0. A symmetric level-simplex perturbation is

    T=(I_N-1_N1_N^T/N) diag(1/beta_1,...,1/beta_N),
    x_n=x0+delta sum_i T_ni e_i,  1<=n<=N.         (3.1)

Here T beta=0, 1_N^T T=0, and rank T=N-1. For sufficiently small delta>0 all N images remain strictly positive, below one, and inside the same strict full/self routing neighborhoods. They are genuinely distinct and share the full spatial support. Their mean input is exactly x0, so the actual first-filter mean direction u, including its bias coefficient 1, is unchanged. The symmetric construction avoids concentrating the sum of all perturbations in a single last image.

Their last-hidden rows have the same positive d-level and row rank N. In detail, writing B_A=[b_1,...,b_N], the feature matrix is H0=1_N h(x0)^T+delta T B_A^T. A relation a^T H0=0 first gives sum a_n=0 after multiplying by d. Because B_A has full column rank, it then gives a^T T=0. The left nullspace of T is span(1_N), so a=0. A common output-bias contrast therefore cancels the logits on all images while preserving full row rank. An alternative construction with N-1 independent level rays and a final negative sum is also valid; the symmetric version is better balanced for finite witnesses.

This construction can also make each sensitivity close to its nonzero base-image value c_j:

    |s_nj-c_j|<|c_j|/(2B)  for every n,j.           (3.2)

There are finitely many raw coordinates, and each base sensitivity is nonzero by positive interior paths and free positive hidden/head coefficients. Shrinking delta preserves row rank for every nonzero delta while enforcing (3.2).

## 4. Raw layer scaling and a finite normal-stability inequality

Scale the last hidden FC weights/bias by t>0 and the output-head contrast by t, with output-bias contrast scaled by t² to retain z=0. Every other reference parameter is unchanged and remains trainable. Then the non-output-bias raw Jacobian has disjoint column groups

    R(t)=[t J1,t² J2],                             (4.1)

where J1 includes free output-head and last-hidden-FC coordinates, and J2 contains all earlier coordinates. The output-head block alone contains (H0/2,-H0/2), so

    lambda1=lambda_min(J1 J1^T)>0.

Let Gamma bound the entries of J1 and J2, and let C²=||J1||_F²+||J2||_F². For 0<t<=1,

    gamma_nonbias<=t Gamma,
    ||R||_op ||R||_F<=t² C²,
    lambda_min(R R^T)>=t² lambda1.                (4.2)

Equation (3.2) is preserved by this scaling: a raw sensitivity and its base-image value are multiplied by the same t or t² within each coordinate. The selected first-filter mean coordinates lie in the t² group.

### 4.1 Explicit separation of the output-bias block

The two output-bias sensitivities remain +/-1/2 for every image. They cannot satisfy an all-raw epsilon-dominance condition. Sample-exchangeability of the new-shuffle schedule law makes their response contribution a scalar common-mode matrix:

    J_B L_B=b_B 1_N1_N^T.                         (4.3)

The sign b_B>=0 is supplied by the separate stationary output-bias result. It is **not** inferred from (2.2), which is too weak for a sensitivity of 1/2 when epsilon is small. Nonnegativity is sufficient for the following reference argument; the companion may prove a strictly positive coefficient.

For the remaining coordinates, E_R=L_R-c0 R^T satisfies

    ||R E_R||_op
       <=c0 [C_B t Gamma/epsilon] t² C².

Therefore, once (4.3) has its nonnegative sign,

    Sym(JL)>=c0 t²[lambda1-C_B t Gamma C²/epsilon] I_N.

The explicit choice

    0<t<min{1, epsilon lambda1/(2 C_B Gamma C²)}    (4.4)

gives

    Sym(JL)>=(c0/2)t² lambda1 I_N>0.               (4.5)

Thus all N normal directions are strictly stable in the averaged linearization. Conditioning of the N distinct feature rows is included in lambda1; no bound uniform in the coincident-image limit is claimed. Every fixed nonzero admissible image separation has lambda1>0, so (4.4) permits a strictly positive t.

## 5. The actual mean has a positive normal endpoint ray

At (4.4), the response relative error in every non-output-bias coordinate is less than 1/2: lambda1<=C² and the stronger spectral inequality implies this entrywise reserve. Every raw sensitivity on the support of u is positive. Equation (2.2) consequently gives

    (L^T u)_n
       >=c0(1-rho_target) sum_(j in target) u_j s_nj>0,
    rho_target<=C_B t² Gamma_target/epsilon.

Here L is the actual stationary batch-16 response, not an arbitrary diagonal metric. Since JL is invertible, its normal projection is

    Q=L(JL)^(-1)J,  QL=L.

The exact ray

    v=L L^T u

satisfies

    Qv=v,
    u^T Qv=||L^T u||²>0.                         (5.1)

When the companion regularity/stable-endpoint argument supplies a C2 endpoint map with derivative I-Q, the usual finite Taylor cone around this ray has strictly positive endpoint loss of the actual affine first-filter mean. This uses a current-state response and a derived margin, rather than assuming a future sink sign. The ray may involve all raw blocks.

## 6. Regularity input for the zero-RMS histories

Equation (2.1) establishes first differentiability at a zero RMS, but it must not be used to claim pointwise C2/C3 there. The batch-16 reference supplies the following uniform small-ball input for the separate expectation-regularity proof.

At zero logits, for a batch whose 16 labels are not evenly split, |sum Y_n|>=2. Under (3.2), every raw coordinate then satisfies

    |g_j| >= (2/B)|c_j|-max_n|s_nj-c_j|
            > |c_j|/B.

This is a simultaneous good-batch event for all raw coordinates. Its failure probability for the first batch of a task is exactly

    q_bad=binom(16,8)/2^16 approximately .19638.

Different tasks provide independent good-task trials. A recent good task contributes a known positive term to the stationary second moment. If K is the number of successive bad tasks since such a contribution, then P(K>=k)<=q_bad^k, while the RMS decays no faster than a fixed positive constant times beta2^(H(k+1)/2). Thus a uniform inverse-RMS moment of order r is finite whenever

    q_bad < beta2^(r H/2).                        (6.1)

For beta2=.999, H=4 or H=6, this holds in particular for r=8. Positive constants may be very small after t-scaling but remain finite. Small parameter/logit perturbations preserve a reduced good-batch margin, since the population tanh term and sensitivity deviations are continuous and can be kept below a fixed fraction of the displayed strict bound. These are finite current-state neighborhood conditions.

This makes the reference compatible with the companion derivative-under-expectation argument; the latter must still prove precisely which inverse moment controls its C3 quotient derivatives. The first-derivative estimate and the high-order regularity argument have not been conflated. Stronger whole-task small-ball estimates are possible but unnecessary for E=2.

## 7. Capacity for N images and both scaling limits

Fix any prescribed finite ridge lambda>0. The zero-sum image construction keeps the true mean direction u fixed while delta varies. For either full or literal-self architecture, the complete raw logit NTK has exact scale decomposition

    K(t,delta)=K_bout+t² K1(delta)+t⁴ K2(delta),
    K'(t,delta)=t² K1'(delta)+t⁴ K2'(delta),
    K_bout=1_N1_N^T tensor I_2.                   (7.1)

Every raw bias is included. The last-hidden-FC derivatives are order t because they differentiate independent raw coordinates; earlier blocks are order t². Output-bias derivatives are constant and their mean derivative is zero. The same identities hold for the literal isolated-channel model, with its own retained positive features and biases and no head refitting.

The original capacity slope divided by t² extends continuously to t=0:

    Ctilde_lambda(t,delta)
      =.5 tr[(lambda I+K_bout+t²K1+t⁴K2)^(-1)
                           (K1'+t²K2')].          (7.2)

At delta=0 all N images coincide. The one-image leading derivative obeys

    k1' >= 2<h_base,D_u h_base> I_2 >0,

because the free output-head block supplies this strict term and positive-polynomial mixed derivatives make the final-hidden-FC contribution nonnegative. Therefore

    Ctilde_lambda(0,0)
      =N/[2(lambda+N)] tr(k1')>0.                 (7.3)

This holds independently for full and literal self. Joint continuity gives a rectangle of small image separations and small t on which both normalized slopes remain positive. Choose one sufficiently small delta>0 first, retaining feature row rank N and (3.2), then choose t small enough for both that rectangle and (4.4). These requirements have a nonempty intersection. The original unnormalized slopes are positive and of order t²; a scale-independent capacity margin is not asserted.

At the resulting fixed finite reference, rank, routing, the strict capacity traces, and normal stability persist in a sufficiently small full-dimensional raw parameter neighborhood. The fixed ridge is essential to this continuity argument; no single neighborhood for all lambda approaching zero is claimed.

## 8. What this completes and what remains external

This note provides the native N=32/48 all-bias architecture, distinct full-support images, common zero logits, rank-N raw Jacobian, the new-shuffle mean-gradient factors, a relative nonbias response bound including zero RMS, finite normal-stability and mean-ray conditions, and compatible original full/self capacity signs.

Two external components must be joined explicitly:

1. the [output-bias stationary theorem](adam_b16_bias.md) giving b_B>=0 in (4.3), with the actual regrouped schedule and retained moments;
2. the [expectation-regularity proof](adam_b16_regularity.md) and [integrated stable-endpoint/tube argument](adam_b16_longtime.md) converting these local conditions into the actual stochastic Adam long-time statement.

Neither is silently replaced by full-batch sign reasoning or by the finite-history quotient bound. ReLU is unchanged. Very small nonbias sensitivities relative to epsilon, local fixed routing, binary labels, finite E=2, and sufficiently small decaying learning rates remain substantive scope restrictions. This is a reference construction and conditional local proof component, not an RL-CIFAR trajectory measurement.
