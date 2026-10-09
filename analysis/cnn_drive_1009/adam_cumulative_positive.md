# Moving CNN / Adam: a rigorous cumulative self-direction bridge

2026-10-09. This document provides identities and sufficient conditions on the **actual moving Adam history**. It does not replace that history by gradients at a fixed parameter, assume all momentum components have the desired sign, or transfer an SGD trajectory theorem to Adam without checking the additional terms.

## 1. Actual first-Conv mean and Adam recursion

Consider any CNN architecture and any joint training of its other parameters. Inputs to the first Conv are fixed. Let theta_t in R^d denote a chosen first-Conv parameter block, including its bias if that bias is trained. Let u be its fixed mean augmented input patch, so its mean preactivation is

  zbar_t = u^T theta_t.

Let F_{t-1} contain all past parameters, labels, batches and optimizer state. The current stochastic gradient is g_t and its actual conditional expectation is

  mu_t=E[g_t|F_{t-1}].

This expectation must use the current moving network. Previously observed task labels cannot silently be averaged as if freshly independent. Write Adam's first moment as b_t to avoid confusion with zbar:

  b_t=beta b_{t-1}+(1-beta)g_t, beta=beta_1 in (0,1),
  v_t=beta_2 v_{t-1}+(1-beta_2)g_t^{odot 2},
  bhat_t=b_t/(1-beta^t), vhat_t=v_t/(1-beta_2^t),
  theta_t=theta_{t-1}-eta_t bhat_t odot d_t,
  d_{t,j}=1/(sqrt(vhat_{t,j})+epsilon).

Assume epsilon>0. Initial moments are zero unless explicitly stated otherwise. eta_t>0 is predictable. Define

  w_t = eta_t/(1-beta^t) (u odot d_t),
  D_T = zbar_0-zbar_T = sum_{t=1}^T w_t^T b_t.

D_T>0 is the actual cumulative mean-preactivation decrease, not a hypothetical capacity derivative.

## 2. Two exact summation-by-parts identities

### 2.1 Using the actual Adam weights

For arbitrary realized gradients and weights,

  D_T = sum_{t=1}^T w_t^T g_t + B_T(w),
  B_T(w) = beta/(1-beta) [
      w_1^T b_0 - w_T^T b_T
      +sum_{t=1}^{T-1}(w_{t+1}-w_t)^T b_t].

Proof: g_t=b_t+beta/(1-beta) * (b_t-b_{t-1}), followed by finite summation by parts. No stochastic assumption is used.

Let bar_w_t be a predictable vector and let s_t>0 be a predictable scalar. Splitting the first term gives the exact decomposition

  D_T = S_T + M_T + R_pred,T + R_current,T + B_T(w),
  S_T = sum_t s_t u^T mu_t,
  M_T = sum_t bar_w_t^T(g_t-mu_t),
  R_pred,T = sum_t (bar_w_t-s_t u)^T mu_t,
  R_current,T = sum_t (w_t-bar_w_t)^T g_t.

M_T is a martingale. R_current,T is generally **not** a martingale: the current Adam denominator depends on g_t. Dropping its conditional bias would contradict the stationary Adam counterexample.

### 2.2 Splitting the denominator before summation by parts

Often a cleaner identity uses only the predictable scalar proxy s_t u:

  boxed: D_T = S_T + M_T^0 + R_den,T + B_T^0,

where

  M_T^0=sum_t s_t u^T(g_t-mu_t),
  R_den,T=sum_t (w_t-s_t u)^T b_t,
  B_T^0=beta/(1-beta) [
      s_1 u^T b_0-s_T u^T b_T
      +sum_{t=1}^{T-1}(s_{t+1}-s_t)u^T b_t].

This follows by first writing w_t=s_t u+(w_t-s_t u), then applying summation by parts only to the proxy part. It avoids requiring small total variation of the **actual** noisy denominator. The price is a denominator residual multiplied by the actual momentum. That residual is bounded in norm below; its sign is never assumed.

These identities hold for every finite T, including random terminal times by evaluating the pathwise equality at the terminal index.

### 2.3 What ordinary long-time averaging does remove

For plain, unweighted momentum the identity simplifies to

  (1/T)sum_{t<=T}(b_t-g_t)
    = beta/[(1-beta)T] * (b_0-b_T).

Thus bounded moments make the difference O(1/T), regardless of the signs of individual moments. The same limiting average holds for bias-corrected bhat_t: if ||b_t||<=G, then

  sum_{t>=1}||bhat_t-b_t||
    <=G sum_{t>=1} beta^t/(1-beta^t)<infinity.

So the intuition that a first-moment reversal can wash out in a long-time average is correct for momentum alone. It does not dispose of Adam's denominator. Even in a stationary scalar model, E[b_t d_t] contains the persistent covariance between the momentum and reciprocal denominator. For a fixed beta_2 and iid gradient squares of nonzero variance, the stationary variance is

  Var(v_t)=(1-beta_2)/(1+beta_2) Var(g_t^2)>0.

The second-moment estimate uses a fixed exponential horizon; observing the process for longer does not make each v_t an increasingly accurate full-history average. Time averaging can converge to an expectation containing that covariance. In the general weighted identity, the weight-variation and denominator residuals can therefore be of the same order as the cumulative self signal. Their negligibility needs a bound, rather than following merely from T tending to infinity.

## 3. A finite-horizon certificate without a momentum-sign assumption

Define observable upper bounds

  E_den,T=sum_{t<=T} ||w_t-s_t u||_2 ||b_t||_2,

  E_mom,T=beta/(1-beta) ||u||_2 [
      s_1||b_0||_2+s_T||b_T||_2
      +sum_{t<T}|s_{t+1}-s_t| ||b_t||_2].

Then, pathwise,

  boxed: D_T >= S_T + M_T^0 - E_den,T - E_mom,T.

All momentum vectors may change signs. The bounds use their norms and the proxy's variation, not the desired sign of Adam's update.

Suppose a separate gradient-geometry theorem gives

  u^T mu_t >= Gamma_t >=0

on the current states. The source of Gamma_t must be stated: e.g. a valid current-state F2/F3 fitted-head certificate, a joint-training stability theorem, or the structural CE family in section 6. Put L_T=sum_{t<=T}s_t Gamma_t. If

  E_den,T+E_mom,T <= (1-delta)L_T, delta>0,

then D_T>=delta L_T+M_T^0. Thus cumulative sinking follows once the martingale fluctuation is smaller than the remaining positive margin.

The condition concerns the current feature/gradient geometry, second-moment history, moment norms and their variation. It does not assume the sign of the final Adam displacement. It can fail, and the stationary counterexample shows that failure cannot be ruled out solely by beta_2 being close to one.

## 4. How to express and check the denominator conditions

Choose a predictable scalar d_ref,t>0 and

  s_t=eta_t d_ref,t/(1-beta^t).

If the actual coordinate denominators satisfy

  max_j |d_{t,j}/d_ref,t -1| <= delta_t,

then

  E_den,T <= sum_t s_t delta_t ||u||_2 ||b_t||_2.

This explicitly measures both anisotropy across coordinates and the current-gradient effect on the denominator. It is not enough to replace d_t by E[d_t|past] and ignore the difference.

For t>=2 the debiased second moment satisfies exactly

  vhat_t=(1-a_t)vhat_{t-1}+a_t g_t^{odot2},
  a_t=(1-beta_2)/(1-beta_2^t).

For a predictable scalar q_t>0, take d_ref,t=1/(sqrt(q_t)+epsilon). Then

  |vhat_{t,j}-q_t|
    <=(1-a_t)|vhat_{t-1,j}-q_t|+a_t|g_{t,j}^2-q_t|,

and the exact reciprocal-root identity is

  |d(v)/d(q)-1|
    = |v-q| / [(sqrt(v)+sqrt(q))(sqrt(v)+epsilon)].

Lower bounds on vhat and q, together with feature-derived bounds on gradient squares, therefore give a quantitative denominator error. If gradients tend to zero, another possible proxy is d_ref=1/epsilon; then delta_t<=max_j sqrt(vhat_{t,j})/(sqrt(vhat_{t,j})+epsilon)->0. Whether its accumulated error is small **relative to the self signal** still needs proof.

The momentum norm itself has the deterministic bound

  ||b_t|| <= beta^t||b_0||+(1-beta)sum_{r=1}^t beta^{t-r}||g_r||.

If ||g_t||<=G, b_0=0 and s_t is nonincreasing, then

  E_mom,T <= [beta/(1-beta)] ||u|| G s_1.

Thus momentum can create finite boundary/variation effects while leaving a diverging positive cumulative signal intact. No coordinatewise momentum sign is needed for this statement.

## 5. Long-time consequences, including finite total sink

### 5.1 Diverging self signal

Let a_T be a deterministic increasing normalization tending to infinity. If

  sum_t E[(s_t u^T(g_t-mu_t))^2]/a_t^2 < infinity,

then M_T^0/a_T->0 almost surely by martingale-series convergence and Kronecker's lemma. Suppose, on the geometry-certificate event,

  liminf S_T/a_T >= s_*>0,
  limsup (E_den,T+E_mom,T)/a_T <= r_*<s_*.

Then

  liminf D_T/a_T >= s_*-r_*>0  almost surely on that event.

This is a positive long-time theorem for the actual moving Adam history. The event's persistence must be established independently, or the process must be stopped at its first geometric exit. The martingale bound applies to the stopped increments as usual. An arbitrary arrangement average is not substituted for conditional sampling along the actual history.

### 5.2 Finite self signal is a different case

The moving-SGD family proved in the companion note has a finite, nonzero total mean decrease. In that regime one must not demand an artificial diverging drift, or infer that martingale noise vanishes relative to the finite signal.

Suppose a stopped version of the process has total martingale variance at most V_infinity, and its geometric certificate provides, for all T>=T_0,

  S_T>=Gamma>0,
  E_den,T+E_mom,T<=r<Gamma.

For any 0<x<Gamma-r, Doob's L2 maximal inequality gives

  P(sup_T |M_T^0|>=x) <= V_infinity/x^2.

Consequently, on the surviving certificate event and outside this martingale-failure event,

  D_T>=Gamma-r-x>0 for every T>=T_0.

The total failure probability includes any separately bounded probability of leaving the geometry region. A useful probability bound requires V_infinity small enough relative to the margin; it is not automatic.

### 5.3 What cannot be transferred from an SGD proof

A theorem such as a_infinity^2<=a_0^2-||B_0 P_U||^2 proved for a specific SGD trajectory is not an input that can simply be assigned to Adam. Adam changes the states, the head, the gradient samples and the relevant geometric quantities. The bridge is valid only if the self-signal certificate and error budget are evaluated or proved **on the Adam trajectory itself**, or if a separate trajectory-comparison theorem supplies that transfer.

## 6. A concrete moving CNN where the conditional self signal survives Adam

This connects the identities to the image/spatial-rank family in [moving_ce_longtime.md](moving_ce_longtime.md), rather than to an unrelated all-zero-label supervised example.

Use L layers of shared 1x1 Conv -> ReLU -> MaxPool, with no hidden or output biases. Write

  W_l=a_l u_l u_{l-1}^T,  a_l>0,
  V_{c,j,s}=B_{c,s} u_{L,j},
  input_n(s)=X_n(s)u_0.

Here every u_l, including the input vector u_0, is the unit vector with all d_l coordinates equal to 1/sqrt(d_l), X_n(s)>0, and selected pool maxima are unique. The pooled spatial vectors x_n can have rank at least two; they need not be proportional across images. Every conv channel responds on every image. The layer outputs remain in the channel-symmetric manifold, and

  f_n=alpha B x_n, alpha=product_l a_l.

At every optimizer step draw fresh uniform class labels (and, optionally, a fresh image minibatch). This is fresh-label-per-step training, not multiple epochs reusing a fixed within-task label assignment.

At the actual current alpha,B, put r_n=B x_n and let u_C be the uniform class distribution. Conditional on the whole past,

  c_t=sum_n pi_n r_n^T(softmax(alpha r_n)-u_C)>=0,
  E[partial L/partial a_l | past]=(alpha/a_l)c_t>=0.

Proof: for each r, h(s)=log sum_c exp(s r_c)-s mean_c r_c is convex, h'(0)=0, and the displayed scalar is h'(alpha). It is strict when some r_n is not class-constant. No old-label independence is imposed on B or the CNN state; only the current new labels are fresh uniform labels.

There is also a quantitative geometry lower bound. If max_{n,c}|alpha r_{n,c}|<=Z and P=I-11^T/C, then

  c_t >= alpha exp(-2Z)/C sum_n pi_n ||P Bx_n||^2.

Indeed integrate r^T(diag(p)-pp^T)r from s=0 to alpha, use p_c>=exp(-2Z)/C along the interval, and minimize the unweighted squared deviation over a scalar mean. This is a representation/logit-spread condition that yields Gamma_t for the general theorem.

For a representative first-layer channel,

  zbar=(mean X)/sqrt(d_1) a_1,
  u^T mu=(mean X)/sqrt(d_1) E[partial L/partial a_1|past].

Thus the scalar signal is exactly proportional to the actual first-Conv mean-preactivation direction.

### 6.1 Ordinary coordinate Adam preserves the channel symmetry

Within each Conv layer, the gradient is g_{a_l} u_l u_{l-1}^T: every entry equals g_{a_l}/sqrt(d_l d_{l-1}). This requires uniform u_0 as well as uniform hidden vectors. Starting from zero optimizer states, all entries within a layer have identical first/second moments and identical coordinate updates. The channel copies of each spatial/class head coefficient likewise have gradient (partial L/partial B_{c,s})/sqrt(d_L), hence receive identical updates. Therefore ordinary coordinate Adam, updating every Conv layer and the complete multiclass head, preserves this channel-symmetric manifold.

B is an arbitrary class-by-spatial matrix here. Coordinate Adam need not preserve class-centering of B, and no such claim is used. A common-logit class component does not affect the softmax, and the structural conditional signal subtracts the uniform class mean explicitly. No hidden or output bias is trained in this construction.

For D_l=d_l d_{l-1}, the scalar amplitude follows

  Delta a_l = -eta_t sqrt(D_l) bhat_{a_l,t}
               /[sqrt(vhat_{a_l,t})+epsilon sqrt(D_l)].

The layer amplitudes need not stay balanced. The SGD Lorentz/energy invariant need not survive. Only the stated channel symmetry is claimed.

### 6.2 Positivity of all amplitudes can be preserved predictably

For beta_1^2<beta_2, weighted Cauchy–Schwarz gives the distribution-free coordinate bound

  |bhat_t|/sqrt(vhat_t)
    <= K_beta := [(1-beta_2)(1-beta_1^2/beta_2)]^(-1/2),

with a zero-over-zero gradient history interpreted as zero. Adding epsilon only reduces the ratio.

For completeness, with j denoting lag, the debiased EMA weights are

  A_j=(1-beta_1)beta_1^j/(1-beta_1^t),
  B_j=(1-beta_2)beta_2^j/(1-beta_2^t).

Cauchy–Schwarz gives bhat_t^2/vhat_t<=sum_{j<t}A_j^2/B_j. This sum equals

  [(1-beta_1)^2/(1-beta_2)]
  [(1-beta_2^t)/(1-beta_1^t)^2]
  sum_{j<t}(beta_1^2/beta_2)^j.

Using (1-beta_1)/(1-beta_1^t)<=1 and 1-beta_2^t<=1 gives exactly K_beta^2 above. The factor (1-beta_1) has been bounded away conservatively, not accidentally omitted. The bound includes bias correction.

Therefore the predictable step cap

  eta_t <= min_l a_{l,t-1}/[2 sqrt(D_l) K_beta]

keeps every a_l strictly positive at every finite step for every label realization. The cap is conservative but nonzero at every finite positive state. This does not by itself supply a strictly positive limiting amplitude or the SGD uniform lower bound.

Therefore the conditional self-sign formula c_t>=0 remains valid on an actual infinite moving Adam trajectory in this finite-width, shared-CNN family. What this alone does **not** prove is that current-denominator bias and accumulated momentum/noise errors are below the self signal. That is the additional certificate in sections 3–5. If the head becomes class-constant, the structural signal can vanish; its positive amplitude cannot simply be assumed to persist.

### 6.3 Two layers: connection to the literal capacity-self direction

For L=2 this structural signal also agrees with the original channel-isolated capacity direction, even when Adam makes a_1 and a_2 unequal. Let X have rows x_n^T, b=vec(BX^T), Q=(XX^T) tensor I_C (with the matching vectorization order), and k=(mean X)u_{1,c}>0. Perturb the target first-layer row along its actual mean-preactivation gradient: W_1(q)=W_1+q(mean X)e_c u_0^T. For K=JJ^T computed from all raw Conv and head parameters, the direct derivatives are

  K'_full(0)=2a_1 k[bb^T+a_2^2 Q],
  K'_self(0)=2a_1 k[bb^T+a_2^2 u_{1,c}^2 Q].

Here the self network keeps only first-layer channel c and the corresponding second-layer input column, without refitting the head. Both derivatives are nonzero positive semidefinite. Therefore the full and literal-self logdet capacities Phi=(1/2)logdet(I+K/lambda), lambda>0, strictly increase in the mean-increasing direction q. Their negative-gradient direction lowers the mean.

On the same actual Adam states, partial_q f=(k/a_1)f, so the conditional raw CE drive is nonnegative in this direction and is strictly positive when Bx_n is not class-constant for some sampled image. At class-constant output, raw CE drive is zero although the capacity derivatives remain positive. This is nonreversal, with strict sign agreement only while class contrast is present. The actual Adam direction still needs the residual bridge; positive capacity derivatives alone do not control its denominator.

## 7. A nonempty positive Adam case for the exactly balanced erasure objective

There is a restricted batch design that proves actual long-term Adam sinking without freezing the network. For every selected image, include one copy with each of the C class labels in the update batch. The order may be randomized, but the full-batch gradient exactly equals the uniform-new-label expectation. This is **balanced label enumeration**, not iid labels for the copies.

Use the same channel-symmetric CNN, train every Conv and multiclass head by ordinary coordinate Adam, and apply the positive-amplitude step cap above. The structural formula gives each realized Conv amplitude gradient g_{a_l,t}>=0, even while the head and all Conv layers change. Starting from zero moment states, nonnegative Conv moments now follow from the recursion; their sign is a derived property of this particular architecture/batch design, not an assumption of the general cumulative theorem.

Each amplitude is nonincreasing and stays positive. If the initial head has a nonclassconstant output on an image included in the first update batch, the first Conv gradient is strictly positive. Thus for all later times

  a_{1,t} <= a_{1,0}-delta_1 <a_{1,0},

where, for D_1=d_1 d_0 and first-step amplitude gradient g_{a_1,1}>0,

  delta_1=eta_1 sqrt(D_1) g_{a_1,1}
             /[g_{a_1,1}+epsilon sqrt(D_1)] >0.

Consequently the actual first-Conv mean has a strictly negative permanent net displacement and a limiting mean, because a_{1,t} decreases within (0,a_{1,0}). This is a finite-width moving CNN with all channels overlapping on every image and with a free spatial/multiclass head, not a fixed-theta gradient driver. The image/spatial rank can exceed one.

This example establishes nonemptiness of a positive long-time Adam conclusion, but its balanced labels remove the label-noise/denominator-correlation difficulty. It does not establish the same result for the ordinary iid-label version, and its channel symmetry is restrictive. The general residual theorem is needed precisely when those special signs are lost.

### 7.1 Minimal check of the actual identity and residual budget

Reproduce with [verify_adam_cumulative_positive.py](verify_adam_cumulative_positive.py); recorded output is [adam_cumulative_positive.json](../../results/cnn_drive_1009/adam_cumulative_positive.json).

A float64 check used two positive, nonproportional 8x8 input images, widths (1,2,2), two shared Conv1x1 -> ReLU -> MaxPool2x2 blocks, a free 3-class by 4-position head, and all three class copies per image. Every Conv weight and head coefficient was updated by actual coordinate Adam, with beta_1=.9,beta_2=.999,epsilon=1e-8 and eta=min(.001,the positivity cap). No output or hidden biases were added.

Over 200 updates, amplitudes moved from (1,1) to (.9376903112,.9113422137); channel-symmetry discrepancy was exactly zero in this computation. For a representative first-Conv mean, the actual sink was .07159685561. Choosing the actual scalar denominator as the predictable proxy is valid here because the balanced full-batch gradient is a deterministic function of the current state. The cumulative identity then gave

  S_T=.09138640457,
  E_mom,T=.02586827734,
  E_den,T=0, M_T^0=0,
  certified lower bound S_T-E_mom,T=.06551812723>0.

The exact displacement identity's floating-point discrepancy was 7.50e-16. This verifies that the norm/variation certificate itself, not only the direct sign argument, is nonempty on an actual moving multi-layer CNN/Adam example. The finite arithmetic check does not replace the symbolic all-time monotonicity proof or establish iid-label behavior.

## 8. Completion boundary

Proved here:

- Exact pathwise cumulative identities for arbitrary actual Adam histories.
- Predictable-proxy martingales and explicit denominator/momentum residual bounds, without assuming momentum keeps one sign.
- Long-time sufficient conditions for both diverging and finite self-signal regimes.
- Preservation of a nonempty moving shared-CNN geometry under actual coordinate Adam, with a structural fresh-label conditional self signal.
- A restricted balanced-label moving CNN with a strict persistent mean decrease under actual Adam.

Not proved here:

- That the residual budget remains below the signal for ordinary iid labels in the moving CNN family.
- That an SGD energy invariant or its finite sink bound holds along Adam's trajectory.
- That fixed labels reused for many optimizer steps admit the same per-step uniform-label conditioning.
- That the full RL-CIFAR architecture with arbitrary channels, output biases, its task schedule and Adam satisfies these conditions.

The stationary negative Adam example and this positive conditional bridge are compatible: the former shows why the denominator residual cannot be omitted; the latter specifies what would suffice to control it on the actual learning history.


Independent mathematical audit: [adam_cumulative_positive_review.md](adam_cumulative_positive_review.md).
