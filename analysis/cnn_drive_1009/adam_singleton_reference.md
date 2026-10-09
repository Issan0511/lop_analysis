# Native all-bias reference for shuffled singleton minibatches

2026-10-09. Constructive local result for three distinct positive images, binary labels, singleton minibatches, finite shuffled task reuse, and ordinary Adam. The output-bias sensitivities are treated exactly, rather than incorrectly being declared small compared with epsilon. The construction gives a stable normal linearization, an explicit noncircular mean-sinking ray, and the original full/literal-self capacity signs. The stochastic endpoint/tube transfer uses the separate averaging and geometry arguments.

## 1. Schedule scope

There are N=3 fixed images. Each task independently draws the three uniform binary labels once and reuses them for E epochs, H=3E optimizer updates. Each update uses one image. The task schedule is independent of label values and has balanced coverage: each image occurs E times. Assume its law is invariant under a common permutation of the three image indices. The usual independent uniform `randperm(3)` in each epoch satisfies this for every finite E. Arbitrary dependence among the epoch permutations is also allowed if this permutation symmetry and balanced coverage hold. Different task innovations, consisting of labels and schedule, are iid.

No particular realized shuffle sequence is required. However, permutation symmetry is a condition on the schedule law being averaged, not on every individual realization. A fixed asymmetric balanced sequence is not automatically covered by the exact output-bias calculation below. Standard beta1=.9, beta2=.999, fixed epsilon>0, retained moments, and global bias correction are compatible with the construction; no reset or projection is introduced.

## 2. Native images and the two-group raw sensitivity scaling

Start with the [native three-image reference](adam_three_images_reference.md): RGB32-by-32, Conv5/pad2 widths16/16, both ReLU/MaxPool layers, FC100/ReLU, FC100/ReLU, free binary head, and every ordinary bias. The three positive images have common d-level in the last hidden feature, full-row-rank feature matrix H0, strict full/self routing, positive hidden paths, and the same mean input as a base image. Here d>0 is the reference head-contrast vector.

Keep all earlier hidden parameters fixed in the reference construction, but multiply the final hidden FC weight matrix and its bias by t>0. Multiply the output-head contrast by t as well:

    W_last(t)=t W_last(1),  b_last(t)=t b_last(1),
    V_+(t)=t d,  V_-(t)=-t d.

Set the output-bias contrast to minus t² times the common original d-level. This gives z_n=0 for all three images. Every raw parameter remains independently trainable after initialization. Earlier hidden biases retain their positive values; final hidden biases are positive and of order t; output-bias contrast is of order t².

The last hidden features are exactly h_n(t)=t h_n(1), with all gates and pool winners unchanged. Thus H(t)=tH0 has row rank three for every t>0. The full raw half-contrast Jacobian J(t), of shape 3 by P, splits into:

- output-bias columns J_B=(+1/2,-1/2), constant across images and independent of t;
- group R1: free output-head weights and final-hidden-FC weights/biases, with raw sensitivities exactly t times their values at t=1;
- group R2: all earlier Conv/FC weights/biases, with raw sensitivities exactly t² times their values at t=1.

Write R(t)=J_non-output-bias(t). Its two disjoint groups are

    R(t)=[t J1, t² J2],                            (2.1)

up to a fixed column ordering. The raw mean-direction support lies in R2. This accounting is essential: differentiating f=t²f_base with respect to a tied amplitude would not give the correct raw Jacobian. The free output-head block in J1 is (H0/2,-H0/2), so

    lambda1 := lambda_min(J1 J1^T)>0.              (2.2)

For 0<t<=1, let finite reference constants satisfy

    Gamma >= max(|J1|,|J2|) entrywise,
    C² = ||J1||_F²+||J2||_F².

Then

    max|R(t)| <= t Gamma,
    ||R(t)||_op ||R(t)||_F <= t² C²,
    lambda_min(R(t)R(t)^T) >= t² lambda1.          (2.3)

All raw sensitivities are nonzero at every fixed t>0. In particular, first-Conv mean sensitivities are small but strictly positive, of order t². The three images remain distinct, full-support, and overlapping; they are not made identical by this parameter scaling.

## 3. Exact stationary singleton linear response

At a zero-logit reference, let s_nj=J_nj. A singleton gradient of coordinate j on sample n with its reused task label is

    g_j=s_nj[tanh(z_n)-Y_n].

For one stationary phase, group the first- and second-moment weights by the pair (old task, image). Denote them A_{q,n} and B_{q,n}; both arrays sum to one. Put

    A_n=sum_q A_{q,n},  B_n=sum_q B_{q,n},
    C_n=sum_q A_{q,n} B_{q,n},
    sigma_j²=sum_n s_nj² B_n.

At z=0, sigma_j does not depend on the labels, because each singleton gradient square is s_nj². Label correlations in the first derivative occur precisely when two history entries use the same task and image. Differentiating the quotient and averaging the labels therefore gives the phase-summed response L of shape P by N:

    L_jn = s_nj sum_phases E_schedule[
          A_n/(sigma_j+epsilon)
          -s_nj² C_n/{sigma_j(sigma_j+epsilon)²}].  (3.1)

There is no additional 1/N outside this expression: these are singleton losses summed over H phases. The balanced schedule supplies the sample-average factor below.

The stationary expected field is zero on the entire local manifold z=0 by simultaneous complementation of all task labels. Its derivative with respect to sensitivity changes alone is therefore zero on that manifold. Consequently its full raw-parameter linearization is

    D Fbar(theta_*)=L J.                           (3.2)

At a fixed positive t the per-step singleton gradient has a positive absolute floor: every |s_nj| is nonzero and |tanh z_n| is bounded below one in a small neighborhood. Every stationary RMS has the same positive floor. The stationary field and L are smooth there. This does not require the odd-three full-batch noise-floor argument.

### 3.1 Positive coefficients and a relative small-sensitivity bound

The grouped weights satisfy C_n<=A_n B_n and s_nj²B_n<=sigma_j². Hence the coefficient in brackets in (3.1) lies between

    epsilon A_n/(sigma_j+epsilon)²
    and A_n/epsilon.

Balanced coverage and the stationary phase sum give

    sum_phases E[A_n]=H/N.

This last identity follows by phase-averaging the normalized first-moment convolution: every past task contains the same E occurrences of image n. It does not require a new label within an epoch.

For gamma_j=max_n|s_nj| and c0=H/(N epsilon),

    L_jn=s_nj a_jn,
    c0(1+gamma_j/epsilon)^(-2)<=a_jn<=c0.           (3.3)

In particular, L_jn has the sign of s_nj. More quantitatively,

    |L_jn-c0 s_nj|
       <=c0 rho_j |s_nj|,
    rho_j=1-(1+gamma_j/epsilon)^(-2)
           <=2 gamma_j/epsilon.                  (3.4)

This is a relative entry bound on the linear response, rather than an absolute error asserted to dominate a vanishing normal drift.

## 4. Output biases are exact, not epsilon dominated

For either output-bias coordinate, |s_nj|=sigma_b=1/2 for every image. Permutation symmetry of the shuffle law implies that the expected response coefficient is identical for every image. The two bias response rows are therefore

    L_b+,n=+c_b/2,  L_b-,n=-c_b/2,

where

    0 < (H/N) epsilon/(epsilon+1/2)² <= c_b <= c0.

For example, if omega=(1/H)sum_phases E[sum_(q,n) A_(q,n)B_(q,n)], then

    c_b=(H/N)[epsilon+(1/2)(1-omega)]/(epsilon+1/2)².

Thus their exact contribution to the normal linearization is

    J_B L_B=(c_b/2) 1_N 1_N^T >=0.                (4.1)

The fixed raw bias sensitivity 1/2 may be enormously larger than epsilon; it has not been scaled, approximated, removed, or frozen. Its contribution is a positive common-mode matrix. Sample-exchangeable shuffling is what makes it symmetric in the sample coordinates.

## 5. Explicit nonempty normal-stability condition

For non-output-bias coordinates define E_R=L_R-c0 R^T. Equations (3.4) and (2.3) give

    ||E_R||_F <= c0 rho ||R||_F,
    ||R E_R||_op <= c0 rho ||R||_op ||R||_F,
    rho <=2t Gamma/epsilon.

The exact normal matrix is

    B(t):=J L
      =(c_b/2)11^T+c0 R R^T+R E_R.               (5.1)

It need not be symmetric. Its symmetric part nevertheless satisfies

    Sym B(t) >= c0 t²[lambda1-2t Gamma C²/epsilon] I_N.

Consequently the explicit choice

    0<t<min{1, epsilon lambda1/(4 Gamma C²)}        (5.2)

guarantees

    Sym B(t) >= (c0/2)t² lambda1 I_N >0.           (5.3)

This proves a stable full-rank normal linearization: under the averaged descent flow, the linearized logit dynamics are dot z=-B(t)z. The perturbation is O(t³/epsilon²) while the guaranteed normal margin is O(t²/epsilon); the factor lambda1 captures the conditioning of the three distinct images. The normal gap may be very small, but (5.2) admits a strictly positive finite t for every fixed admissible image construction and every fixed epsilon>0.

Only the non-output-bias coordinates use the epsilon-dominated estimate. Using a bound based on the maximum of all raw sensitivities, which includes 1/2, would not establish (5.2).

## 6. A mean-sinking ray using the actual singleton response

This section uses the actual stationary response L from (3.1), not an arbitrary diagonal metric and not Adam applied to an averaged gradient. At the reference, B=JL is invertible by (5.3). The endpoint normal projection supplied by a smooth locally stable equilibrium geometry is

    Q=L (JL)^(-1) J.

It obeys Q²=Q and QL=L. Let u be the actual first-Conv mean gradient. Its supported raw sensitivities are positive, and (3.3) therefore gives

    L^T u >0 componentwise.

Choose the initial normal ray

    v=L L^T u.                                    (6.1)

Then

    Qv=v,
    u^T Qv=u^T v=||L^T u||_2²>0.                  (6.2)

This is an exact positive endpoint-mean derivative, with no delicate comparison to a diagonal-SGD projection. It gives a finite open cone of strict endpoint mean loss by the same C2 endpoint/Taylor argument used previously, once local stable-normal geometry is supplied. The ray generally changes multiple parameter blocks; it does not claim that a target-filter-only ray works for arbitrary singleton Adam response. The reference capacity direction remains the original u, and the conclusion concerns the net change of its affine mean observable.

## 7. Original capacity survives the joint image/parameter limit

The image separation parameter and the scale t must not be conflated. Let delta denote the image perturbation width in the three-image construction, with delta=0 giving three copies of the base image. The symmetric three-image construction keeps u exactly fixed for all delta. Fix a prescribed ridge lambda>0.

For either the full or literal-self network, the independent raw logit-Jacobian blocks have exact t-orders:

- output-bias Jacobian is constant in t;
- output-head and final-hidden-FC Jacobians are t times their unscaled blocks;
- all earlier hidden Jacobians are t² times their unscaled blocks.

These identities also hold in a small mean-direction perturbation, since that perturbation changes only the first-filter augmented block. Thus the entire raw NTK and its mean derivative have the exact form

    K(t,delta)=K_bout+t² K1(delta)+t⁴ K2(delta),
    K'(t,delta)=t² K1'(delta)+t⁴ K2'(delta),         (7.1)
    K_bout=11^T tensor I_2.

The last-FC raw derivative is of order t even though its reference weight has also been scaled by t; all trainable bias blocks are included. In the literal self architecture, retained earlier biases may destroy a simple full/self rescaling relation, but the t-orders above still hold because the final hidden affine layer and its bias are scaled together.

Define the continuously extended normalized capacity slope

    Ctilde_lambda(t,delta)
      =(1/2)tr[(lambda I+K_bout+t²K1+t⁴K2)^(-1)
                          (K1'+t²K2')].           (7.2)

For t>0 this is exactly the original capacity slope divided by t². The positive ridge makes (7.2) continuous at t=0 as well as delta=0.

At delta=0, the leading derivative is

    K1'(0)=11^T tensor k1',
    k1' >= 2<h_base,D_u h_base> I_2 >0.            (7.3)

The free output-head block supplies the displayed strict positive term. The final-hidden-FC block contributes a nonnegative scalar multiple of vv^T, because positive fixed-branch polynomial coefficients give nonnegative first and mixed derivatives. Hence

    Ctilde_lambda(0,0)
      =3/[2(lambda+3)] tr(k1')>0.                 (7.4)

The self network has the same argument with its own positive hidden features. By joint continuity, there are delta_cap>0 and t_cap>0 such that both normalized full/self slopes remain strictly positive throughout a sufficiently small rectangle 0<=delta<delta_cap, 0<=t<t_cap.

Choose one fixed delta>0 within that rectangle, small enough also for full-row-rank features, strict routing, and the positive path conditions from the three-image construction. Then choose t satisfying both (5.2) and t<t_cap. All constraints permit a strictly positive t. Both original capacity slopes are positive for this fixed finite native reference and remain so on a sufficiently small full raw-parameter neighborhood. They scale like t²; no scale-independent lower bound on the unnormalized slope is claimed.

This resolves the two-limit issue: a capacity certificate at an arbitrary fixed unscaled reference was not silently assumed to persist down to t=0. The scaled leading raw blocks and the jointly continuous normalized slope establish that persistence explicitly. A fixed finite lambda, or a compact interval bounded away from zero, is the asserted ridge scope.

## 8. Result and remaining transfer

For every finite E>=1, every fixed epsilon>0, and the sample-exchangeable balanced shuffle laws above, the native all-bias three-image construction admits a sufficiently small strictly positive t with:

- distinct positive overlapping images, a rank-three hidden feature matrix, and a rank-three half-contrast Jacobian;
- zero output contrast on all images and nonzero per-step raw sensitivities;
- a smooth stationary singleton-Adam field with a strictly stable normal linearization;
- an exact positive endpoint-mean ray v=L L^T u;
- strict full and literal-self original NTK capacity directions at the prescribed ridge.

All raw parameters, including output biases, train under the actual optimizer. Small positive t, strict routing, stable normal spectrum, sensitivity floors, and capacity traces persist in an open parameter neighborhood. The actual stochastic theorem still uses the companion taskwise observable averaging and stable-endpoint tube argument, with sufficiently small decaying scalar learning rates and moments carried across tasks.

The construction does not establish the same result for a deterministic asymmetric schedule law, arbitrary batch sizes, ten classes, standard constant learning rates, or an actual RL-CIFAR training trajectory. The very small nonbias sensitivities relative to epsilon are a substantive sufficient condition, and the resulting finite constants may be conservative. ReLU is unchanged and remains active in the local family.
