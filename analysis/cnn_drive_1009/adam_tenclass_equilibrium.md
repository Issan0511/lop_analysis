# Ten uniform classes: the uniform-output Adam field need not vanish

2026-10-09. Fixed-state stationary-response analysis, with standard Adam beta1=.9, beta2=.999, epsilon>0, iid uniform ten-class labels, and retained moments. The principal counterexample uses the native positive-branch deep ReLU/MaxPool CNN with all raw parameters free. A separate intermittent-sensitivity construction retains batch size sixteen and genuine taskwise label reuse. No training experiment is used.

**Main conclusion.** The binary identity `Fbar(0,S)=0` cannot be transferred to ten classes from uniform labels and uniform predictions alone. There are two distinct obstacles: a nonzero common-class raw head/bias drift that is only a softmax gauge motion, and a realizable hidden drift that actually moves predictions away from uniformity. Paired centered head rows can cancel the latter, but do not generally make the entire raw-parameter field zero.

## 1. What uniform cross entropy does establish

For C=10, define centered class sensitivities for image n and hidden raw coordinate j by

    a_(n,c,j)=partial_j f_(n,c)-(1/C)sum_d partial_j f_(n,d).

At uniform outputs p_(n,c)=1/C, the single-example raw gradient is

    g_(n,j)=-a_(n,Y_n,j),
    E_Y g_(n,j)=0.                                 (1.1)

For a mean batch loss it is the average of these gradients. This proves a zero expected SGD gradient. It does not prove

    E[m_j/(sqrt(v_j)+epsilon)]=0.

For binary labels the two centered values are opposites. For ten labels, their sum is zero but their multiset need not be centrally symmetric. Adam's denominator distinguishes different squared magnitudes, so centering does not suffice.

All stationary expectations below retain the same gradient history in m and v. None replaces v by its mean or applies Adam after averaging the labels.

## 2. Exact two-point stationary lemma, including task reuse

Let q in (1/2,1), and let independent task variables be

    xi_k=q-I_k, I_k~Bernoulli(q).

The values are q with probability 1-q and -(1-q) with probability q. Thus E xi=0 but the two magnitudes differ. At a fixed stationary phase, suppose a raw gradient stream has the form

    g_t=s d_t xi_(task(t)),  s>0,

where d_t is a schedule-dependent indicator in {0,1}, independent of labels. It may be identically one. Group first- and second-moment weights over occurrences in the same task:

    A_k=sum_(lags in task k) alpha_l d_(t-l),
    B_k=sum_(lags in task k) beta_l d_(t-l),
    alpha_l=(1-beta1)beta1^l,
    beta_l=(1-beta2)beta2^l.

Then A_k,B_k>=0, their sums are at most one, and the exact stationary quotient is

    Q=[sum_k A_k xi_k]/[sqrt(sum_k B_k xi_k^2)+e],
    e=epsilon/s.                                  (2.1)

Condition on the complete schedule and all xi except xi_k. Writing C0 for the other squared-moment terms and h(v)=1/(sqrt(v)+e),

    E[xi_k h(C0+B_k xi_k^2)|others]
       =q(1-q)[h(C0+B_k q^2)-h(C0+B_k(1-q)^2)]
       <=0,                                       (2.2)

strictly if B_k>0. Summing with A_k proves E Q<0 whenever at least one task has A_k B_k>0 almost surely. Reused labels are grouped into a single independent task variable before conditioning. This lemma does not require independent optimizer-step gradients, nor normalized sums of the active weights.

A finite negative margin is

    E[Q|schedule] <= -c(q,e) sum_k A_k B_k,
    c(q,e)=(1-q)(2q-1)/[2(q+e)^2].                 (2.3)

Indeed the entire squared-moment sum is at most q^2, so `-h'(v)>=1/[2q(q+e)^2]` on the relevant interval. Absolute convergence and epsilon>0 justify interchanging the numerator sum and expectation.

When d_t=1 and each task repeats its label for H phases, the phase-r overlap is exactly

    omega_r=(1-beta1^r)(1-beta2^r)
       +(beta1 beta2)^r (1-beta1^H)(1-beta2^H)
                         /[1-(beta1 beta2)^H]>0.   (2.4)

For H=1, omega=(1-beta1)(1-beta2)/(1-beta1 beta2).

## 3. A native all-raw deep CNN at exactly uniform logits

Take one strictly positive RGB image and the native hidden architecture

    Conv5/pad2(16), ReLU, MaxPool2,
    Conv5/pad2(16), ReLU, MaxPool2,
    FC100, ReLU, FC100, ReLU,

with every ordinary hidden bias. Choose positive hidden weights/biases, a strictly active ReLU branch, and unique MaxPool winners. Positive center-dominant kernels and a monotone spatial image provide such a branch; a small positive perturbation of all kernel entries preserves it. Every shared kernel entry has a positive path from a valid winning patch through the free positive FC layers.

Let h(theta_hidden)>0 be the final feature and choose a strictly positive vector w. Define

    psi(theta_hidden)=w^T h(theta_hidden),
    d_j=partial_j psi>0

for every hidden raw coordinate j at the reference. Channel features need not be proportional. The strict positivity follows from positive paths, and applies to the true shared convolutional parameters and biases.

Assign the centered class coefficients

    a_c=2/5 for six classes,
    a_c=-3/5 for the remaining four.               (3.1)

They sum to zero. Set the independently trainable output-head rows and output biases to the reference values

    V_c=a_c w,
    b_c=-a_c psi(theta_hidden,*).                  (3.2)

Every logit is exactly zero. The bias in (3.2) is an initial value only; no parameter is subsequently tied, frozen, or refit.

For a hidden raw coordinate,

    g_j=d_j[(3/5)-I{Y is one of the six classes}].  (3.3)

The favored indicator is Bernoulli(3/5) under truly uniform ten-class labels. Each expected raw CE gradient is zero. Applying (2.2) with s=d_j shows, for every stationary phase and any finite H-fold task reuse,

    E[Adam quotient_j]<0                           (3.4)

for every hidden coordinate. This includes the entire selected first-filter augmented block. For its true mean direction u>=0,

    u^T Fbar_hidden<0,
    expected Adam mean displacement=-eta u^T Fbar_hidden>0.

Thus the hidden mean moves upward at this uniform-prediction reference, even though the ordinary CE mean gradient is zero.

For a coordinate normalized to d_j=1, H=1 and epsilon=1e-8, the exact rational bound (2.3) is less than `-0.000110120027162941`. This is a stationary finite-beta result, not a fresh-first-step sign calculation.

### 3.1 The hidden drift is not merely a softmax gauge motion

For each output-head entry and output bias, the fixed-state gradient laws are respectively

    g_(V_c,l)=h_l[1/10-I{Y=c}],
    g_(b_c)=1/10-I{Y=c}.

Uniform labels make their stationary expected quotients identical across c, for each l. Consequently the expected head/bias changes contribute only a common shift to all ten logits on this image.

The hidden contribution is different. In the averaged descent ODE, the centered logits have derivative

    d/dt[f_c-(1/10)sum_d f_d]
       =-a_c sum_(hidden j) d_j Fbar_j.             (3.5)

The sum on the right is strictly negative before its minus sign, by (3.4). Since a is nonzero, (3.5) is nonzero. Therefore the uniform-prediction manifold itself is not invariant at this reference, even after quotienting out common-class logits.

### 3.2 Compatibility with the original capacity/self direction

At the same one-image reference, the full independent-raw logit NTK is exactly

    K=(||h||^2+1) I_10+||grad_hidden psi||^2 a a^T.

For the actual first-filter mean direction u,

    D_u K=2<h,D_u h>I_10
       +2<grad psi,D_u grad psi> a a^T.             (3.6)

The first coefficient is strictly positive. On the positive fixed branch, h and psi have nonnegative polynomial coefficients in the hidden parameters and input, so the mixed-derivative coefficient is nonnegative. Hence D_u K is positive definite and the original ridge-capacity derivative is strictly positive for every fixed lambda>0. The literal isolated-channel architecture has the same argument with its retained positive hidden features and biases.

Thus a positive original full/self capacity direction does not remove the multiclass stationary-Adam hidden bias. This is a fixed-state obstruction, not a proof of an indefinitely reversed moving-CNN trajectory.

### 3.3 Robustness

The stationary quotient is locally Lipschitz in a uniformly bounded gradient history, by the RMS norm inequality and epsilon>0. The strict hidden margin therefore survives sufficiently small perturbations. With one image, arbitrary small changes of hidden/head parameters can be accompanied by small free output-bias adjustments that preserve equal logits. Nonzero hidden/normal drift persists on a relatively open part of that uniform-output manifold. The obstruction is not confined to exactly repeated six-versus-four head rows.

## 4. Batch sixteen with independent labels and reshuffled reuse

A batch-size restriction does not remove the two-point obstruction if a target coordinate has small effective image support. At a uniform-output fixed state, suppose a target first-Conv input-color coefficient has sensitivity kappa>0 on one image and zero on all other N-1 images. Then its true mean-batch gradient is

    g_t=(kappa/16) I{target image in batch_t} xi_task,
    xi_task=3/5-I{target label is favored}.          (4.1)

All N labels are still mutually independent and uniform; the other labels simply have zero derivative in this coordinate. The target occurs once each epoch under fresh reshuffling. Conditional on the schedule, (2.1)-(2.3) apply with s=kappa/16 and grouped task weights. Therefore the stationary expected hidden quotient is strictly negative at every phase, including phases whose current batch omits the target.

For E=2 and H=4 or 6, the previous complete task contains a target visit at lag at most 2H-1. Hence uniformly over all schedules,

    sum_k A_k B_k
      >=(1-beta1)(1-beta2)(beta1 beta2)^(2H-1).      (4.2)

For kappa=1 and epsilon=1e-8, the exact rational negative margins from (2.3)-(4.2) are at least

    H=4: 5.27731773222e-6,
    H=6: 3.44861913230e-6.                         (4.3)

These use genuine task-label reuse and fresh partitions, not correlated labels across the sixteen images.

### 4.1 Shared-Conv realization and positive-input perturbations

A minimal transparent realization uses shared Conv1x1 -> ReLU -> MaxPool2 with two overlapping channels, independently trainable biases b1=b2=1/10 at the reference, and spatial pattern rho=(1,.8,.6,.4). The two filter rows over RGB are

    W1=(2,1,1), W2=(2,3/2,3/2).

An image has RGB coefficients c=(R,G,B)>=0 with R+G+B=1, multiplied by rho at its four positions. Both channels are strictly active and have unique top-left pooled winners. With h=(W1 c+1/10,W2 c+1/10) and w=(1,-2),

    psi=w^T h=-21/10

for every such image. The ten-class head V_c=a_c w and output bias b_c=(21/10)a_c give exactly zero logits on every image. Both channels contribute; the second has a nonzero opposing readout.

Take one red image c=(1,0,0) and N-1 different green/blue mixtures with R=0. The target first-filter red coefficient has `partial psi/partial W1,R=R`, giving (4.1) with kappa=1. The image labels are independent even if some feature rows coincide.

This example need not rely on zero color inputs. Give the target image c=(1-2delta,delta,delta), and every other image R=delta with positive different green/blue mixtures summing to 1-delta. Uniform logits remain exact because psi=-21/10 on the entire color simplex. The target gradient history changes by at most

    (3/5)(1+1/16)delta=(51/80)delta

at every optimizer step. For standard stationary Adam the norm bound K0<8 gives a response change less than

    [9*(51/80)/epsilon]delta.                     (4.4)

At delta=1e-16 and epsilon=1e-8 this is 5.7375e-8, less than half of either margin in (4.3). These comparisons were checked with exact Fraction arithmetic. Thus positive RGB images, all active ReLUs, unique spatial pooling winners, B16, fresh reshuffles, and truly iid ten-class labels can still have a nonzero stationary hidden field at exactly uniform logits. No training was run.

The prototype can be embedded in wider/deeper native fixed-branch CNNs through center-supported convolutions and full-rank affine feature maps; sufficiently small free parameter perturbations preserve the strict hidden response. To preserve all equal logits during such an embedding, one must retain the common-level constraints rather than assume one bias cancels unequal image outputs. On a shared fixed branch, input images rho*c give h(c)=M c+b. Choosing a free head vector w with `M^T w=k 1` makes psi constant on the color simplex. The prototype has this property and positive target derivative; additional full-rank hidden coordinates provide free head directions to retain it. The principal complete native all-raw/normal-drift construction remains section 3, while this batch-16 construction proves a hidden-field obstruction; it does not assert that every other hidden coordinate or the total first-filter mean has the same sign.

## 5. Class permutation symmetry is equivariance, not cancellation at an arbitrary state

Permuting class rows of the parameters and permuting labels together preserves the CE/Adam law. Thus the expected field is equivariant under class permutations. Uniform outputs do not, however, make the parameter state invariant under those permutations: (3.2) has unequal class-head rows compensated by unequal output biases.

At a state fixed by every class permutation, all head rows are equal, so hidden label gradients vanish samplewise. This is a much stronger and largely degenerate condition. A symmetry of the data law does not imply an odd hidden gradient distribution at a fixed, nonsymmetric parameter state.

There is also a genuine gauge issue. For a one-image uniform-output model, every raw output-bias coordinate has the centered two-point gradient `p-I`, with p=1/10. Applying the sign-reversed version of section 2 gives a strictly positive stationary quotient for every class bias. For H=1 and epsilon=1e-8, a rational lower bound is `4.89422348384e-5`. Equal expected shifts of all class biases do not change softmax probabilities, but do make the full raw field nonzero.

The same is true for free head entries: their expected rows agree classwise. Adding a common vector to all head rows or a common scalar to all output biases leaves CE gradients unchanged, so the quotient dynamics modulo these common-class translations is well-defined. Convergence in that quotient is a different conclusion from convergence of every raw parameter. In the one-image paired construction below, the averaged raw gauge modes keep moving even when all predictive contrasts remain exactly uniform.

## 6. Structural cancellation conditions that do survive ten classes

A sufficient coordinatewise condition at uniform outputs is the following. For each hidden raw coordinate j and image n, there is a class permutation pi_(n,j) such that

    a_(n,pi_(n,j)(c),j)=-a_(n,c,j) for every c.     (6.1)

Transform each reused task label Y_n once by this same permutation for that image. The iid uniform task-label law is unchanged, and every batch gradient in coordinate j changes sign throughout the whole task. Apply the transformation to every historical task: m_j changes sign and v_j does not. Hence the expected stationary Adam quotient for coordinate j is exactly zero, for any batch size, label-independent schedule, finite task reuse, betas, and positive epsilon.

The transformation may depend on j because a coordinatewise expectation can be paired separately. A stronger and simpler native condition is a single five-pair involution of the centered class-head rows:

    V_(pi(c))-Vbar=-(V_c-Vbar),
    Vbar=(1/10)sum_c V_c.                         (6.2)

For hidden parameters, (6.2) implies (6.1) for every image and raw coordinate simultaneously. It allows nonzero, multidirectional class contrasts and arbitrarily many overlapping CNN channels. At a uniform-output reference, hidden Fbar is then zero. Expected free head/bias rows are class-common, so the expected predictive contrast field is zero as well.

This is a nonempty structural cancellation mechanism, but it is a condition on head geometry, not a consequence of uniform labels. It is not open in arbitrary free head parameters. Small generic asymmetric perturbations can restore a constant hidden bias at uniform prediction, as section 3 demonstrates. Nor does (6.2) make the raw common-class head/bias field vanish. A proof based on it should formulate an equilibrium modulo gauge and separately handle any raw common-mode drift.

## 7. Consequences for extending the binary long-time theorem

The binary construction used both `Fbar(0,S)=0` and a normally attracting zero-output manifold to build an endpoint map for all raw parameters. For C=10:

1. Zero expected CE gradient is not enough to obtain that field identity.
2. Class-permutation invariance of the label law is not enough at a fixed head state.
3. Pure gauge drift must be distinguished from hidden drift and from normal predictive drift.
4. The native counterexample has strictly nonzero normal predictive drift, so quotienting out common logits does not solve the generic obstruction.
5. A paired-head reference supplies an actual cancellation condition in the predictive quotient, but requires a new normal-spectrum analysis and a treatment of gauge motion; it does not inherit the existing all-raw convergence theorem automatically.

These results do not rule out a different multiclass positive long-time construction or a shifted stationary manifold. They do rule out substituting C=10 into the binary proof while retaining uniform predictions as an equilibrium merely because fresh labels are uniform. Every counterexample here is a frozen-state stationary-field statement; the resulting indefinitely moving CNN trajectory would require a separate analysis.

The displayed rational margins are reproduced in [verify_adam_tenclass.py](verify_adam_tenclass.py) and [its saved certificate](../../results/cnn_drive_1009/adam_tenclass.json). No sampling or training is involved.
