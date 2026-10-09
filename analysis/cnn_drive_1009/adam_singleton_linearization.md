# Singleton minibatches with taskwise label reuse: exact stationary linearization

Date: 2026-10-09. Scope: a frozen-state calculation for ordinary Adam, followed by a checkable normal-stability certificate. The calculation keeps the same random image labels throughout each task and keeps Adam history across tasks. It is the stationary field needed by the existing stochastic-averaging theorem; by itself it is not a theorem about a moving CNN trajectory or an arbitrary RL-CIFAR run.

## 1. Model, conventions, and averaging

There are (N=3) distinct fixed images; the formulas hold for any finite (N). Every task (k\in\mathbb Z) draws independent Rademacher labels (Y_{k,n}\in\{-1,+1\}), independently across (k,n). Each task uses those same labels for (E\ge1) epochs, with (H=NE) singleton updates. The schedules are independent of every label. The positive certificate below uses a uniform random permutation in every epoch, independently across epochs and tasks. More generally, the same result holds when the two-sided schedule law is invariant under every permutation of image identities and each task has (H) phases.

The network is a shared-weight ReLU CNN on a strict routing cell: all required ReLU signs and MaxPool winners have nonzero margins. All raw weights and biases may train. Write its binary logit contrast as

\[
 z_n(\theta)=\frac{f_{+,n}(\theta)-f_{-,n}(\theta)}2,
 \qquad s_{nj}(\theta)=\partial_{\theta_j}z_n(\theta),
 \qquad J=(s_{nj})\in\mathbb R^{N\times P}.
\]

Up to a label-independent additive constant, binary cross-entropy is
\(\ell(z,Y)=\log(2\cosh z)-Yz\). Thus the true singleton gradient is

\[
 g_{t,j}=s_{i_t,j}\bigl(\tanh z_{i_t}-Y_{k(t),i_t}\bigr).
 \tag{1.1}
\]

Use fixed (0<\beta_1,\beta_2<1), in particular (0.9,0.999), and fixed (\epsilon>0), in particular (10^{-8}). The infinite-history frozen Adam response is

\[
 m_{t,j}=\sum_{\ell\ge0}(1-\beta_1)\beta_1^\ell g_{t-\ell,j},
 \quad v_{t,j}=\sum_{\ell\ge0}(1-\beta_2)\beta_2^\ell g_{t-\ell,j}^2,
 \quad q_{t,j}=\frac{m_{t,j}}{\sqrt{v_{t,j}}+\epsilon}.
 \tag{1.2}
\]

Here every gradient in (1.2) is evaluated at one fixed state. Let (R^{(r)}(s,z)=\mathbb E q_r) for phase (r\in\{1,\ldots,H\}), and define the **task-summed** field

\[
 F(\theta)=\sum_{r=1}^H R^{(r)}(s(\theta),z(\theta)).
 \tag{1.3}
\]

An epoch/task learning-rate convention using the phase average instead simply divides (F) and all matrices below by (H). Neither convention changes a sign or stability assertion. Labels are not replaced by their means before the Adam quotient is evaluated.

## 2. Exact conditional response at zero logits

Fix phase (r) and the entire past image schedule. A single reused label is indexed by (q=(k,n)), a **task-image pair**. Group all lags in (1.2) that carry this label:

\[
 A_q=\sum_{\ell:\,(k(t-\ell),i_{t-\ell})=q}(1-\beta_1)\beta_1^\ell,
 \qquad
 B_q=\sum_{\ell:\,(k(t-\ell),i_{t-\ell})=q}(1-\beta_2)\beta_2^\ell.
 \tag{2.1}
\]

The sums range only over already observed positions. They satisfy (A_q,B_q\ge0), (\sum_q A_q=\sum_q B_q=1). Put

\[
 A_n=\sum_{q:i(q)=n}A_q,
 \quad \Gamma_n=\sum_{q:i(q)=n}A_qB_q,
 \quad \sigma_j^2=\sum_q B_qs_{i(q),j}^2.
 \tag{2.2}
\]

All these quantities are functions of the schedule and sensitivities, not of the labels. At (z=0),

\[
 m_j=-\sum_q A_qs_{i(q),j}Y_q,
 \qquad v_j=\sigma_j^2.
 \tag{2.3}
\]

For now assume (\sigma_j>0). Holding (s) fixed while differentiating (z),

\[
 \partial_{z_n}m_j=s_{nj}A_n,
 \qquad
 \partial_{z_n}\sqrt{v_j}
 =-\frac{s_{nj}^2}{\sigma_j}\sum_{q:i(q)=n}B_qY_q.
 \tag{2.4}
\]

The labels (Y_q) are independent across distinct task-image pairs. Their covariance is (\mathbb E(Y_qY_{q'})=\mathbf1_{q=q'}), including the reuse within a pair through the already aggregated weights. Therefore

\[
 \mathbb E\left[m_j\,\partial_{z_n}\sqrt{v_j}
 \mid\text{schedule}\right]
 =\frac{s_{nj}^3}{\sigma_j}\Gamma_n.
 \tag{2.5}
\]

The exact conditional phase Jacobian is consequently

\[
 \boxed{\quad
 L_{jn}^{(r\mid\mathrm{schedule})}
 =\left.\partial_{z_n}\mathbb E[q_{r,j}\mid\mathrm{schedule}]\right|_{z=0}
 =s_{nj}\left[
 \frac{A_n}{\sigma_j+\epsilon}
 -\frac{s_{nj}^2\Gamma_n}{\sigma_j(\sigma_j+\epsilon)^2}
 \right].\quad}
 \tag{2.6}
\]

For the task-summed field, (L_{jn}=\sum_{r=1}^H\mathbb E_{\mathrm{schedule}}L_{jn}^{(r\mid\mathrm{schedule})}). In particular, the reuse correction is a sum of **products of grouped weights** (A_qB_q). Replacing it by the sum of products of individual-lag weights drops cross-products between repeated visits to the same image within one task and gives the wrong linearization.

## 3. Each scalar coefficient is positive; normal stability still needs a matrix condition

Write the bracket in (2.6) as (c_{jn}^{(r)}). Since (s_{nj}^2 B_q\le\sigma_j^2) for each relevant (q),

\[
 0\le s_{nj}^2\Gamma_n\le A_n\sigma_j^2.
\]

It follows that

\[
 \boxed{\qquad
 \frac{\epsilon A_n}{(\sigma_j+\epsilon)^2}
 \le c_{jn}^{(r)}
 \le\frac{A_n}{\sigma_j+\epsilon}.
 \qquad}
 \tag{3.1}
\]

The lower bound is strict whenever (A_n>0). Because (c_{jn}^{(r)}) may depend on both (j) and (n), this fact does **not** imply that (JL) is symmetric, positive definite, or even stable. In particular one cannot import the fullbatch diagonal-preconditioner identity into singleton Adam without an additional argument.

The same scalar inequalities give the useful finite-error bound

\[
 \begin{aligned}
 0\le\frac{A_n}{\epsilon}-c_{jn}^{(r)}
 &=\frac{A_n\sigma_j}{\epsilon(\sigma_j+\epsilon)}
 +\frac{s_{nj}^2\Gamma_n}{\sigma_j(\sigma_j+\epsilon)^2}\\
 &\le\frac{2A_n\sigma_j}{\epsilon^2}
 \le\frac{2A_n s_{\max,j}}{\epsilon^2},
 \qquad s_{\max,j}=\max_n|s_{nj}|.
 \end{aligned}
 \tag{3.2}
\]

This bound uses no asymptotic expansion in (\beta_1,\beta_2,H), or the label-reuse duration.

## 4. Schedule exchangeability and the exact output-bias contribution

Under the exchangeable shuffled schedule law, image relabeling shows that (\mathbb E A_n=1/N) at every phase, since (\sum_n A_n=1). Equations (2.6) and (3.2) therefore imply

\[
 \boxed{\qquad
 \left|L_{jn}-\frac{H}{N\epsilon}s_{nj}\right|
 \le\frac{2H}{N\epsilon^2}|s_{nj}|s_{\max,j}.
 \qquad}
 \tag{4.1}
\]

For the two raw output biases, (s_{n,b_+}=+1/2) and (s_{n,b_-}=-1/2) for every image. Hence (\sigma_{b_+}=\sigma_{b_-}=1/2). Put (D_b=1/2+\epsilon). Exchangeability makes the following coefficient independent of (n):

\[
 \begin{aligned}
 d_b
 &=\sum_{r=1}^H\mathbb E\left[
 \frac{A_n}{D_b}-\frac{\Gamma_n}{2D_b^2}\right]\\
 &=\frac{H}{ND_b}
 -\frac{1}{2ND_b^2}\sum_{r=1}^H\mathbb E\sum_qA_qB_q.
 \end{aligned}
 \tag{4.2}
\]

In the second line, exchangeability was used for (\Gamma_n), which is valid here because its weights do not involve a nonconstant sensitivity row. The strict bounds are

\[
 \frac{H\epsilon}{ND_b^2}\le d_b\le\frac{H}{ND_b},
 \quad L_{b_+,n}=\frac{d_b}{2},
 \quad L_{b_-,n}=-\frac{d_b}{2}.
 \tag{4.3}
\]

Consequently, the two trainable output biases contribute exactly

\[
 J_bL_b=\frac{d_b}{2}\,\mathbf1\mathbf1^T
 =:\beta\mathbf1\mathbf1^T,\qquad\beta>0,
 \tag{4.4}
\]

to the normal matrix. Their sensitivities need not be small compared with (\epsilon). Treating them exactly avoids a false small-gradient assumption on all coordinates.

For a fixed schedule, (2.6), (3.1), and (3.2) remain exact. A fixed schedule is not automatically exchangeable; neither (4.2) nor a common output-bias coefficient should be asserted for an arbitrary balanced order without a separate schedule-symmetry proof. The theorem below deliberately uses shuffled schedules.

## 5. A finite, noncircular normal-stability certificate

Let (R) contain every raw coordinate except the two output biases. Write (J_R\in\mathbb R^{N\times P_R}), and let

\[
 \gamma=\max_{n,j\in R}|s_{nj}|,
 \quad c_0=\frac{H}{N\epsilon},
 \quad L_R=c_0J_R^T+E_R.
\]

Equation (4.1) gives

\[
 \|E_R\|_F\le\frac{2H\gamma}{N\epsilon^2}\|J_R\|_F.
 \tag{5.1}
\]

Thus the **exact**, possibly nonsymmetric, normal matrix is

\[
 M:=JL=\beta\mathbf1\mathbf1^T+c_0J_RJ_R^T+J_RE_R.
 \tag{5.2}
\]

For every (x\in\mathbb R^N), Cauchy--Schwarz and (5.1) imply

\[
 \begin{aligned}
 x^T\operatorname{Sym}(M)x
 &\ge c_0\left[
 \sigma_{\min}(J_R)^2
 -\frac{2\gamma}{\epsilon}\|J_R\|_{\mathrm{op}}\|J_R\|_F
 \right]\|x\|_2^2.
 \end{aligned}
 \tag{5.3}
\]

Here (\sigma_{\min}(J_R)) denotes its smallest row singular value; it is positive precisely when (J_R) has full row rank. The positive rank-one bias term was dropped to get this conservative bound.

**Finite certificate.** If

\[
 \boxed{\qquad
 \frac{2\gamma}{\epsilon}\|J_R\|_{\mathrm{op}}\|J_R\|_F
 <\sigma_{\min}(J_R)^2,
 \qquad}\tag{5.4}
\]

then (\operatorname{Sym}(JL)\succ0). Every quantity in (5.4) is a present-state sensitivity or the actual Adam epsilon; the condition does not assume the desired update sign, an invariant training trajectory, or the conclusion of a long-time theorem. It holds for any finite (H=NE) under the shuffled schedule model and for the unchanged Adam betas. The cancellation of (H/N) in this sufficient condition does not mean that the exact field is independent of reuse: the exact (L) and (d_b) retain (H), the grouping, and both betas.

## 6. Nonempty small-amplitude construction

Suppose a strict-routing zero-output reference CNN has a last hidden FC layer and a binary affine head. Keep all earlier parameters fixed and use the following simultaneous scaling by (t>0):

1. Scale the last hidden FC weights **and hidden biases** by (t).
2. Scale the binary head contrast weights by (t).
3. Scale the output-bias contrast by (t^2).

The common head mode is irrelevant to (z) and may be arbitrary. Positive homogeneity of ReLU gives last hidden features (h_t=t h_1), so (z_t=t^2z_1). Hence a seed satisfying (z_1=0) on all three images stays on that zero-output manifold. This statement requires an actual zero-output seed; a single output-bias scalar cannot cancel three arbitrary unequal contrasts.

On the fixed routing cell, head-weight sensitivities and last-FC weight/bias sensitivities are proportional to (t); earlier sensitivities are proportional to (t^2). Thus

\[
 J_R(t)=tJ_1+t^2J_2.
 \tag{6.1}
\]

More generally, it suffices to have a remainder bounded by (t^2) in the following norms. If the three unscaled last-feature rows are linearly independent, the head-weight columns already give (\operatorname{rank}(J_1)=3). Let

\[
 s=\sigma_{\min}(J_1)>0,
 \quad a=\|J_1\|_{\max}+\|J_2\|_{\max},
 \quad P=\|J_1\|_{\mathrm{op}}+\|J_2\|_{\mathrm{op}},
 \quad Q=\|J_1\|_F+\|J_2\|_F.
\]

For (0<t\le1) and (t\|J_2\|_{\mathrm{op}}\le s/2),

\[
 \gamma(t)\le ta,
 \quad\|J_R(t)\|_{\mathrm{op}}\le tP,
 \quad\|J_R(t)\|_F\le tQ,
 \quad\sigma_{\min}(J_R(t))\ge ts/2.
\]

In particular the explicit choice

\[
 0<t\le1,\qquad
 t\|J_2\|_{\mathrm{op}}\le s/2,\qquad
 t<\frac{\epsilon s^2}{8aPQ}
 \tag{6.2}
\]

implies (5.4). Zero remainder norms impose no additional restriction. This proves nonemptiness for every fixed positive epsilon, including (10^{-8}); it does not change epsilon or reset Adam. The margin may require very small amplitudes and is not asserted for ordinary initialization scales. At a fixed positive (t) with strict inequality, the finite certificate and strict routing persist in an open neighborhood. This neighborhood may shrink as (t\downarrow0).

## 7. Smoothness, full raw linearization, and normal attraction

A convenient sufficient smoothness condition on a compact routing neighborhood is

\[
 |s_{nj}(\theta)|\ge s_{*,j}>0\quad\text{for every }n,j,
 \qquad |\tanh z_n(\theta)|\le r_*<1.
 \tag{7.1}
\]

Then every possible singleton gradient has

\[
 |g_{t,j}|\ge s_{*,j}(1-r_*),\qquad
 \sqrt{v_{t,j}}\ge s_{*,j}(1-r_*).
 \tag{7.2}
\]

All upper gradient derivatives are bounded on a slightly larger compact routing cell. The EMA weights sum to one; differentiating their sums preserves these bounds. Equation (7.2) bounds the derivatives of the square root and quotient uniformly over all infinite label and schedule histories. Dominated convergence consequently makes every phase response (R^{(r)}(s,z)) smooth, and justifies every expectation/derivative interchange above. Arbitrarily high finite orders are available because a fixed ReLU routing cell gives polynomial logits and smooth CE. This is a sufficient condition, not a claim that Adam's square root is smooth at an all-zero gradient history. In columns not satisfying (7.1), a separate RMS-floor argument or a weaker regularity analysis is required.

At (z=0), the expected response is identically zero for **every** sensitivity array in this neighborhood: the simultaneous label flip makes the numerator odd and leaves the denominator unchanged. Therefore

\[
 R^{(r)}(s,0)=0,\quad D_sR^{(r)}(s,0)=0,
 \quad D_\theta F\big|_{z=0}=LJ.
 \tag{7.3}
\]

No Hessian-of-logit term has been omitted: the sensitivity derivatives vanish because of this identity. If (\dot\theta=-F(\theta)), then its normal linearization is

\[
 \dot z=-JLz=-Mz.
 \tag{7.4}
\]

When (5.4) holds at a zero-output reference with margin (\kappa>0), continuity gives a neighborhood in which the symmetric part of the corresponding zero-manifold normal matrix is at least (\kappa/2). The smooth Hadamard formula

\[
 F(\theta)
 =\left[\int_0^1D_z\!\left(\sum_rR^{(r)}\right)(s(\theta),\tau z(\theta))\,d\tau\right]z(\theta)
 =:\mathcal L(\theta)z(\theta)
 \tag{7.5}
\]

extends this positivity to a sufficiently thin tube. Along a solution staying in that tube,

\[
 \frac12\frac{d}{dt}\|z\|_2^2
 =-z^T\operatorname{Sym}(J\mathcal L)z
 \le-\frac{\kappa}{4}\|z\|_2^2.
 \tag{7.6}
\]

Together with full row rank of (J), this supplies the normal contraction input for an elementary smooth-endpoint/foliation construction. The quantitative construction in the companion multi-output endpoint note extends by replacing symmetry of the normal matrix with its uniformly positive symmetric part; the geometry proof must still control tangential movement and the tube boundary. The present note does not silently assume that control.

## 8. Relation to actual Adam and limits of this result

The infinite-history response is the stationary frozen reference. Standard bias-corrected Adam started at a finite time with finite initial moments has the same frozen-state limiting response: first- and second-moment initial-history discrepancies decay geometrically, and a positive RMS floor bounds the induced quotient discrepancy. Moments carried across task boundaries are retained throughout; there is no task reset in this construction.

For a moving network, this fixed-state observation does not suffice. One must use the taskwise stochastic-averaging result with whole-task innovations (the label vector and shuffled schedule), and then a stopped-tube/endpoint argument. The gradient process is not conditionally fresh at each singleton update, and (\mathbb E[m/(\sqrt v+\epsilon)]) has nowhere been replaced by a quotient of expectations.

The positive result here is deliberately local: a computable stationary linearization, an explicit finite sufficient stability inequality, and a nonempty open small-amplitude family supporting that inequality. Capacity-self direction, the chosen mean-preactivation displacement, and high-probability infinite-time nonexit require the separate reference geometry and averaging arguments. Arbitrary fixed schedules, general initializations, nonbinary labels, or practical RL-CIFAR long-time trajectories are not concluded from this note.
