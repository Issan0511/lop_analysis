# Binary CNN: an exact raw-Adam invariant line and the original capacity self term

2026-10-09. Independent derivation by `existing_drive_proof`. This note establishes the architecture, exact raw-parameter recurrence, and full/literal-self NTK signs. The moving-process stochastic averaging and long-time convergence theorem are separate. No numerical experiment is needed for the identities below.

## 1. Scope and explicit images

Use a bias-free CNN with independently trainable parameters:

\[
\mathbb R^{3\times4\times8}
\xrightarrow{\mathrm{Conv}_{1\times1}:3\to4}
\mathrm{ReLU}\to\mathrm{MaxPool}_{2\times2}
\xrightarrow{\mathrm{Conv}_{1\times1}:4\to3}
\mathrm{ReLU}\to\mathrm{MaxPool}_{2\times2}
\to\mathrm{Flatten}_{6}\xrightarrow{\mathrm{Linear}:6\to2}f.
\]

Both pools have stride 2. There are 12 first-convolution weights, 12 second-convolution weights, and 12 head weights: 36 independent raw coordinates. There are no biases, normalization, weight decay, projections, frozen parameters, parameter ties, or moment resets.

For an explicit positive spatial pattern, set

\[
P_{r,c}=\frac{4r+(c\bmod4)+1}{16},
\quad 0\le r<4,\quad0\le c<8.
\]

The three input planes of image \(n\) all equal \(X_n=\rho_nP\), with, for example, \(\rho_1=2/5\) and \(\rho_2=4/5\). Each of the two 4-by-4 tiles has maximum 1 before scaling. Every first-pool window has a unique maximum, and the second pool has a unique maximum in each tile. The two final spatial values are both \(\rho_n\). Images differ, all channels overlap on every image, and strictly positive nonwinner sites are present. The resulting sample feature matrix has rank one; no full-row-rank assertion is made.

The data mean of one input plane is

\[
\bar\mu=\frac1{N\,32}\sum_{n,r,c}X_{n,r,c}>0.
\]

For the displayed two images, \(\bar\mu=(3/5)(17/32)=51/160\).

The finite raw-Adam verifier uses another valid tile, whose mean over the two images is 0.22875. The formulas below apply to both patterns using their own actual mean \(\bar\mu\); all pooled maxima are still \(\rho_n\).

## 2. Why the raw gradients coincide

First consider positive layer amplitudes \(w_1,w_2,b\). Every first-convolution entry equals \(w_1\), every second-convolution entry equals \(w_2\), and every head entry for class \(+\) equals \(b\), while every head entry for class \(-\) equals \(-b\). This describes a state in the full raw parameter space, rather than a tied-parameter implementation.

Each final hidden coordinate on image \(n\) is \(12w_1w_2\rho_n\), so

\[
f_n=(z_n,-z_n),\qquad z_n=72bw_1w_2\rho_n.
\]

Write the class label as \(Y_n\in\{-1,1\}\). Its binary softmax cross entropy is

\[
\ell_n=\log(2\cosh z_n)-Y_nz_n.
\]

For a full batch with uniform sample weights, define

\[
T=\frac1N\sum_n\rho_n\bigl(\tanh z_n-Y_n\bigr).
\]

Differentiating with respect to each raw coordinate separately gives

\[
\partial_{W^{(1)}_{c,j}}L=6bw_2T,
\quad
\partial_{W^{(2)}_{d,c}}L=6bw_1T,
\quad
\partial_{V_{+,d,s}}L=6w_1w_2T,
\quad
\partial_{V_{-,d,s}}L=-6w_1w_2T.
\tag{1}
\]

The factor one half in a single class-head derivative is essential: \(\partial_{f_+}\ell=(\tanh z-Y)/2\). Formula (1) incorporates it. At

\[
w_1=w_2=b=a>0,
\tag{2}
\]

all 24 convolution weights and all six positive-head weights have the same gradient

\[
g(a,Y)=6a^2\frac1N\sum_n\rho_n
\bigl[\tanh(72a^3\rho_n)-Y_n\bigr],
\tag{3}
\]

and all six negative-head weights have gradient \(-g(a,Y)\). This holds for every label assignment, including a label assignment reused for arbitrarily many updates. It also holds for sample weighting or minibatches if the same loss is used for all coordinates, although the frozen-noise statement in Section 4 specifically uses the full fixed batch.

The channel counts are substantive. More generally, for \(L\) positive 1-by-1 convolutions, widths \(d_0,\ldots,d_L\), and \(S\) equal final spatial values per image, write \(P=\prod_{j=0}^{L-1}d_j\). The raw positive-head gradient is

\[
g_b=\tfrac12P\Bigl(\prod_lw_l\Bigr)T,
\]

while a raw entry of layer \(l\) has gradient

\[
g_l=\frac{d_LSbP\prod_jw_j}{d_ld_{l-1}w_l}T.
\]

Consequently the all-raw equal-amplitude line is invariant under identical coordinate optimizers when

\[
d_ld_{l-1}=2d_LS\quad\text{for every }l.
\tag{4}
\]

For two layers, widths \(r\to2S\to r\) satisfy (4). Here \(r=3,S=2\). Generic widths, or a rescaling of amplitudes without matching the coordinate Adam denominators and epsilon, do not establish the same invariant line.

## 3. Exact moving Adam recurrence and pathwise positivity

Initialize every convolution entry and positive-head entry at \(a_0>0\), every negative-head entry at \(-a_0\), and every Adam moment at zero. Use the same \(\beta_1,\beta_2,\epsilon>0\) and scalar learning rate on all coordinates. From (1)–(3), the positive copies have identical first and second moments, and the negative copies have the opposite first moment and identical second moment. Thus ordinary coordinate Adam on all 36 raw parameters preserves (2) exactly while \(a_t>0\):

\[
g_t=g(a_t,Y_t),\quad
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\widehat m_t=\frac{m_t}{1-\beta_1^{t+1}},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^{t+1}},\qquad
Q_t=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},
\qquad a_{t+1}=a_t-\eta_tQ_t.
\tag{5}
\]

Here \(m_{-1}=v_{-1}=0\) and \(t=0,1,\ldots\). Bias correction uses global optimizer time and is not reset at task boundaries. Equation (5) is an exact moving-network reduction; no frozen-state approximation has been used.

For \(\beta_1^2<\beta_2\), a uniform, conservative bound is

\[
|Q_t|\le K_\beta
=\bigl[(1-\beta_2)(1-\beta_1^2/\beta_2)\bigr]^{-1/2}.
\tag{6}
\]

Indeed, express \(\widehat m_t=\sum_j\alpha_jg_{t-j}\) and \(\widehat v_t=\sum_jb_jg_{t-j}^2\), with their normalized geometric weights. Weighted Cauchy–Schwarz bounds the squared ratio by \(\sum_j\alpha_j^2/b_j\). Bounding \((1-\beta_1)/(1-\beta_1^{t+1})\le1\), \(1-\beta_2^{t+1}\le1\), and the geometric sum proves (6). A positive epsilon only decreases the absolute ratio.

For any predictable schedule \(0<\delta_t\le1/2\), a scalar rate

\[
\eta_t=\frac{\delta_ta_t}{K_\beta(1+a_t)^p},\qquad p\ge0,
\tag{7}
\]

satisfies

\[
a_{t+1}\ge a_t(1-\delta_t)>0.
\tag{8}
\]

Hence every finite-time iterate keeps all ReLUs open and all specified pool winners strict. The choice of \(p\) and the decay of \(\delta_t\) needed for a long-time stochastic theorem are additional hypotheses of that theorem. Formula (8) alone proves neither boundedness nor convergence nor a sign for the actual conditional Adam update.

The first-convolution mean of any channel is \(3\bar\mu a_t\), and its total mean change is exactly \(3\bar\mu(a_T-a_0)\). Therefore a separate proof of \(a_t\to0\) yields strict net mean sinking and approach to the ReLU boundary in this family. It does not imply finite-time ReLU death. This is distinct from the earlier unbalanced SGD family with a strictly positive limiting amplitude.

## 4. Fixed-state scalar noise and the scope of averaging

For iid uniform binary labels on the full fixed batch, put

\[
\mu(a)=\frac1N\sum_n\rho_n\tanh(72a^3\rho_n)>0,
\qquad
\xi=-\frac1N\sum_n\rho_nY_n.
\]

Then

\[
g(a,Y)=6a^2[\mu(a)+\xi].
\tag{9}
\]

The law of \(\xi\) is symmetric about zero, by complementing all labels. The values \(\rho_n\) may differ. With taskwise label reuse for a fixed finite \(H\), a frozen state produces independent task-level scalar gradients, each repeated \(H\) times. EMA weights can then be aggregated by independent task blocks at each phase. The actual moving process still has (5), and replacing it by a frozen-state stationary average requires the separate averaging theorem. Arbitrary minibatch schedules do not automatically have the same independent scalar-block structure.

In particular, equality of the raw gradients and the fixed-state symmetry do not assert that momentum has the same sign as the current mean gradient on each realized step. This note does not assume such a sign.

## 5. Full raw NTK and its mean-direction derivative

Fix \(a>0\), let \(\rho=(\rho_1,\ldots,\rho_N)^\top\), \(Z=\rho\rho^\top\), and \(v=(1,-1)^\top\). Stack logits sample first, class second. The unnormalized full raw NTK is \(K=JJ^\top\), where \(J\) differentiates with respect to all independent raw parameters.

Choose one first-layer output channel \(c_*\). Its preactivation mean is

\[
\mathfrak m_{c_*}(\theta)
=\frac1{N\,32}\sum_{n,r,c,j}W^{(1)}_{c_*,j}X_{n,r,c}.
\]

Thus its full parameter gradient has value \(\bar\mu\) in each of the three raw entries \(W^{(1)}_{c_*,j}\) and zero elsewhere. The mean direction \(u=\nabla\mathfrak m_{c_*}\) is parameter independent here. To compute \(D_uK\), perturb those three entries to \(a+q\bar\mu\), keeping every other raw parameter fixed. A sufficiently small interval of \(q\) preserves positive activations and every pool winner.

At \(q=0\):

* Each of the 12 first-convolution Jacobian columns is \(6a^2\rho\otimes v\).
* Each of the 12 second-convolution Jacobian columns is \(6a^2\rho\otimes v\).
* Each of the six head columns of class \(c\) is \(12a^2\rho\otimes e_c\).

Therefore

\[
K_{\rm full}=864a^4 Z\otimes(vv^\top+I_2).
\tag{10}
\]

The directional derivative must include all parameter blocks. Under the perturbation:

* The first-convolution Jacobian columns do not change with \(q\).
* The three second-convolution columns receiving channel \(c_*\) become \(6a(a+q\bar\mu)\rho\otimes v\); the other nine remain fixed.
* Every class-head column becomes \(3a(4a+q\bar\mu)\rho\otimes e_c\).

Consequently

\[
K'_{\rm full}(0)
=216a^3\bar\mu\,Z\otimes(vv^\top+2I_2).
\tag{11}
\]

These are the full raw-parameter kernel and its actual derivative, not the scalar tied-amplitude kernel. In particular the head-block derivative supplies the \(I_2\) term and cannot be discarded.

## 6. Literal isolated-channel capacity self

Delete the other three first-layer channels and their incoming weights, and delete their three corresponding input columns in the second convolution. Retain all three second-layer channels and all twelve head weights at their original values, with no refitting. This gives the literal isolated-channel network. Its 18 raw parameters are three first-convolution entries, three second-convolution entries, and twelve head entries.

Its logits are

\[
f_{{\rm self},n}=18a^3\rho_nv.
\]

At \(q=0\), each of its three first-convolution and three second-convolution Jacobian columns equals \(6a^2\rho\otimes v\). Each of its six head columns for class \(c\) equals \(3a^2\rho\otimes e_c\). Hence

\[
K_{\rm self}
=a^4 Z\otimes(216vv^\top+54I_2).
\tag{12}
\]

Under the same mean perturbation, the first-convolution columns remain fixed, the three second-convolution columns become \(6a(a+q\bar\mu)\rho\otimes v\), and the head columns become \(3a(a+q\bar\mu)\rho\otimes e_c\). Thus

\[
K'_{\rm self}(0)
=a^3\bar\mu\,Z\otimes(216vv^\top+108I_2).
\tag{13}
\]

For either full or self network, define the regularized capacity functional

\[
\mathcal C(\theta)=\tfrac12\log\det(\lambda I+K(\theta)),\qquad \lambda>0.
\]

Equations (11) and (13) are nonzero positive semidefinite matrices. Since \((\lambda I+K)^{-1}\) is positive definite,

\[
D_u\mathcal C
=\tfrac12\operatorname{tr}[(\lambda I+K)^{-1}K']>0.
\tag{14}
\]

The same conclusion holds after a fixed positive normalization of \(K\), or after subtracting a parameter-independent log determinant. Rank one of \(Z\) causes no problem because \(\lambda>0\). This is a sign statement for the original log-determinant capacity self term, using the isolated network literally; it is not a redefinition of self as a CE term.

## 7. Direction agreement and precise limitations

The full-network logits and their mean-direction response are

\[
f_{{\rm full},n}=72a^3\rho_nv,
\qquad R_{{\rm full},n}=18a^2\bar\mu\rho_nv
=\frac{\bar\mu}{4a}f_{{\rm full},n}.
\]

The literal self network has

\[
R_{{\rm self},n}=18a^2\bar\mu\rho_nv
=\frac{\bar\mu}{a}f_{{\rm self},n}.
\]

For fresh uniform labels, the corresponding expected CE directional derivatives are therefore

\[
\mathbb E_Y[D_uL_{\rm full}]
=\frac1N\sum_n18a^2\bar\mu\rho_n\tanh(72a^3\rho_n)>0,
\]
\[
\mathbb E_Y[D_uL_{\rm self}]
=\frac1N\sum_n18a^2\bar\mu\rho_n\tanh(18a^3\rho_n)>0.
\tag{15}
\]

Thus, at every positive state on the invariant line, expected CE descent and descent of the original full/literal-self capacity functional point toward decreasing the target mean. This sign agreement holds even though all channels overlap, all raw parameters are moving, pool nonwinners exist, and other-channel kernel blocks are included.

Equations (14)–(15) are instantaneous directional statements; they are not by themselves a theorem about momentum-conditioned Adam descent or labels reused within a task. Long-time sinking of the actual Adam process requires the separately proved scalar-process result based on (5). The exact equal-amplitude manifold is a restrictive family, and no transverse stability or open-family claim is made. The architecture is bias-free, rank one across images, and uses 1-by-1 convolutions; it is not the all-bias, 5-by-5, multiclass RL-CIFAR architecture. Those limitations do not affect the exact all-raw recurrence or the literal capacity-self calculation established here.
