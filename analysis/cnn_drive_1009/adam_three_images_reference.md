# Three distinct positive images: native all-bias CNN reference

2026-10-09. This constructive note extends the native N=1 reference to N=3 distinct positive images. It establishes a rank-three hidden feature matrix and logit Jacobian, an exact common uniform-output equilibrium, a nonzero gradient-noise floor for all eight label assignments and every raw coordinate, and the original full/literal-self capacity signs for any prescribed fixed ridge lambda>0. It does not supply the vector-normal endpoint geometry or a long-time Adam theorem; those are separate arguments.

## 1. Native network and a hidden input map of rank at least three

Use the full native architecture

    RGB32×32 → Conv5/pad2, 3→16, bias → ReLU → MaxPool2
    → Conv5/pad2, 16→16, bias → ReLU → MaxPool2
    → flatten1024 → FC100, bias → ReLU
    → FC100, bias → ReLU → binary FC2, bias.

All raw parameters are independent and trainable. Hidden weights and biases at the reference are strictly positive. All 25 spatial offsets participate, all channels overlap on all images, and no support separation is used. ReLUs and both full/self pool routings will be strict.

Start with one RGB image whose three planes equal

\[
(x_0)_{r,c}=\frac{2+r+32c}{2048},\qquad0\le r,c<32.
\tag{1}
\]

Its values lie strictly between zero and one. First temporarily set each convolution's center-offset coefficients positive and its other offsets to zero. Both pool layers have unique bottom-right winners. Positive hidden biases preserve this strict ordering, since in this temporary center-only network they introduce only spatial constants. All hidden activations are strictly positive.

Select three different final pool cells, for example cells corresponding to original-image winner pixels (3,3), (3,15), and (19,3). Pick the same one output channel of Conv2 at these three spatial cells. Their derivatives with respect to the red-plane values of those three pixels form a positive diagonal 3-by-3 matrix at this temporary network. They are independent input directions even though channels may initially be proportional.

Choose the first three rows of FC1 to select those three flattened features, and the first three rows of FC2 to select the first three FC1 outputs. Set all remaining FC rows to positive weights and all hidden FC biases positive. Every FC preactivation is positive, including the selector rows. The 3-by-3 input-to-last-hidden derivative minor just identified remains a positive diagonal minor, up to positive scalar factors.

Now replace every zero off-center convolution coefficient and every zero selector-row FC coefficient by sufficiently small positive values. The strict ReLU/pool margins and the nonzero derivative minor persist by continuity. This gives a genuine all-positive, two-Conv5/two-FC network with all spatial offsets used. Sufficiently small independent hidden-weight perturbations can also make channel maps nonproportional. The construction simultaneously preserves strict routing in the literal self network obtained by retaining only the selected first-Conv channel and its matching Conv2 input columns: that reduced center-only reference also has strict bottom-right winners, and there are only finitely many margins to preserve.

On the common fixed input-routing neighborhood of x0, the last hidden feature is an affine map of the flattened RGB input:

\[
h(x)=Ax+c,\qquad h(x)\in\mathbb R^{100},\quad \operatorname{rank}A\ge3.
\tag{2}
\]

This input-affinity is compatible with nonlinear dependence on the network's raw parameters. The matrix A is nonnegative, and the selected three input columns \(b_i=Ae_i\), i=1,2,3, are independent. Here e_i is the coordinate vector for one of the three selected red-plane pixels. Each b_i has a positive path to the last hidden layer. Dense positive FC weights make \(d^\top b_i>0\) for every \(d\in\mathbb R^{100}_{>0}\).

## 2. Exact equal-level image perturbations and rank-three features

Fix d>0 and write

\[
\beta_i=d^\top b_i>0,\qquad
v_1=\frac{e_1}{\beta_1}-\frac{e_3}{\beta_3},\qquad
v_2=\frac{e_2}{\beta_2}-\frac{e_3}{\beta_3}.
\tag{3}
\]

Then

\[
d^\top A v_1=d^\top A v_2=0,
\quad a_1:=Av_1,\ a_2:=Av_2\text{ are independent}.
\tag{4}
\]

Independence follows by expanding a linear combination in the three independent columns b_i. Choose a sufficiently small tau>0 and set

\[
x_1=x_0+\tau v_1,\quad
x_2=x_0+\tau v_2,\quad
x_3=x_0-\tau(v_1+v_2).
\tag{5}
\]

All three images stay strictly inside (0,1) pixelwise and inside the same full/self routing neighborhoods. They are pairwise distinct. All retain the entire positive spatial support; the perturbation uses overlapping ordinary images, not isolated channels or disjoint data supports. The three images have the exact pointwise mean

\[
\tfrac13(x_1+x_2+x_3)=x_0.
\tag{6}
\]

With \(h_0=h(x_0)>0\), their hidden rows are

\[
h_1=h_0+\tau a_1,\quad
h_2=h_0+\tau a_2,\quad
h_3=h_0-\tau(a_1+a_2).
\tag{7}
\]

Every row has the same positive d-level \(\psi_0=d^\top h_0>0\). The 3-by-100 feature matrix H with rows h_n^top has rank three for every sufficiently small nonzero tau. To prove this, suppose \(\sum_n t_n h_n=0\). Multiplication by d gives \(t_1+t_2+t_3=0\). Equation (7) then gives \((t_1-t_3)a_1+(t_2-t_3)a_2=0\), forcing t_1=t_2=t_3=0. Thus full row rank is exact, not inferred from a numerical small singular value.

## 3. Uniform output equilibrium and rank-three logit Jacobian

Choose alpha>0, give the two free output-head rows the reference values +alpha d and -alpha d, and give their free biases the reference values -alpha psi0 and +alpha psi0. For the half-contrasts \(z_n=(f_{n,+}-f_{n,-})/2\),

\[
z_1=z_2=z_3=0.
\tag{8}
\]

Both classes are therefore uniform on all three images, despite their independent hidden features. These equalities specify a constructive equilibrium reference; the raw head rows and biases remain freely trainable and may vary independently in a surrounding open parameter neighborhood.

The logit Jacobian \(J_z=D_\theta(z_1,z_2,z_3)\) contains the free output-head blocks

\[
\tfrac12 H,\quad-\tfrac12 H,
\]

and output-bias columns \(+\tfrac12\mathbf1\), \(-\tfrac12\mathbf1\). Since rank H=3,

\[
\operatorname{rank}J_z=3.
\tag{9}
\]

Consequently the local uniform-output set is a regular codimension-three parameter manifold. The rank persists on a full raw-parameter neighborhood of this reference. Its minimum singular value need not stay uniformly positive as tau tends to zero; tau is fixed at a sufficiently small positive value before constructing such a neighborhood.

## 4. All raw sensitivities and the odd-three noise floor

At the base image x0, let \(s_j^*=\partial_{\theta_j}z(x_0)\) at the chosen raw reference parameters. Every s_j^* is nonzero. All hidden-coordinate sensitivities are positive because every raw Conv coefficient has an interior selected path with a positive incoming pixel/feature, and all FC weights and biases have positive outgoing routes. The positive/negative output rows have sensitivities \(+h_{0,a}/2\), \(-h_{0,a}/2\); the output-bias sensitivities are +/-1/2. This includes every hidden bias as an independent coordinate.

For the three nearby images write \(s_{nj}=\partial_{\theta_j}z_n\). There are finitely many raw parameters, so tau can be chosen small enough to satisfy simultaneously

\[
|s_{nj}-s_j^*|<|s_j^*|/6
\quad\text{for all }n,j.
\tag{10}
\]

For any of the eight iid binary label assignments Y=(Y1,Y2,Y3),

\[
\left|\sum_{n=1}^3s_{nj}Y_n\right|
\ge |s_j^*|\left|\sum_nY_n\right|
       -\sum_n|s_{nj}-s_j^*|
> |s_j^*|/2.
\tag{11}
\]

The crucial fact is oddness: \(|\sum_nY_n|\ge1\) for three signs. At the equilibrium, the full-batch CE raw gradient is

\[
g_j(\theta_*,Y)=-\tfrac13\sum_n s_{nj}Y_n,
\qquad |g_j(\theta_*,Y)|>|s_j^*|/6.
\tag{12}
\]

There is thus no zero-gradient label assignment for any raw coordinate, although the expected gradient is zero. For a finite, explicit neighboring-state version, retain (10) and require

\[
\max_n|\tanh z_n|<1/14.
\tag{13}
\]

The population part satisfies

\[
\left|\tfrac13\sum_n s_{nj}\tanh z_n\right|
<\frac76|s_j^*|\frac1{14}=|s_j^*|/12.
\]

Subtracting this from (11)/3 yields, uniformly over every label assignment,

\[
|g_j(\theta,Y)|>|s_j^*|/12.
\tag{14}
\]

These are open conditions satisfied by the equilibrium reference. Taking the minimum over finitely many j gives a strictly positive common gradient floor. It can be small when alpha is small, but alpha and tau are fixed positive constants for the theorem.

The centered label noise \(-\tfrac13\sum_n s_{nj}Y_n\) is symmetric under complementing all three labels. If a label assignment is reused for H full-batch updates, its frozen scalar gradient repeats H times and independent task assignments still give the required independent block noise. The stationary coordinate RMS is bounded below by (14), including all task phases. Frozen quotients and their expectations are consequently smooth functions of the parameters on a compact sub-neighborhood: the finitely many gradient functions are smooth and every infinite EMA denominator has a uniform positive RMS floor.

For clarity, the resulting averaged field has a positive diagonal representation relative to the population gradient

\[
\nabla\Phi(\theta)
=\tfrac13J_z(\theta)^\top\tanh z(\theta),
\quad \Phi(\theta)=\tfrac13\sum_n\log(2\cosh z_n).
\]

The diagonal coefficients can be taken smooth also where one population-gradient coordinate vanishes. Treat its mean \(\mu_j\) and sensitivity vector \((s_{1j},s_{2j},s_{3j})\) as local independent arguments of the frozen quotient. Complementing all task labels makes its expectation an odd smooth function of \(\mu_j\). The gradient floor remains positive while \(\mu_j\) is interpolated to zero, so division by \(\mu_j\) has the smooth extension given by the integral of the derivative along that interpolation. The independent-task symmetric-noise lemma gives strictly positive upper/lower coefficient bounds, including the extension at zero. Summing the finite H phase fields preserves smoothness and positivity.

The field is not generally a scalar multiple of one common tanh z, and the N=1 one-dimensional endpoint flow must not be reused without its separate vector-normal extension. Regularity (9) and the noise floor supply the reference hypotheses for that extension.

## 5. The actual mean direction is unchanged by the image construction

For a selected first-Conv channel, let m be its mean preactivation over all three images and all original spatial positions. Its raw gradient u consists of the dataset mean input patch, including padding, and the bias coefficient 1.

Patch extraction and averaging are linear in the input, so (6) implies

\[
u_{N=3}=u_{x_0}
\tag{15}
\]

exactly. No surrogate direction is substituted. The full/self capacity derivatives below therefore use exactly the same target direction as in the identical-image limit.

## 6. Original full/self capacity: strict trace at fixed ridge

Fix any prescribed lambda>0. At tau=0 the three images coincide at x0. Let \(K_0,J_0,K_0'\) be the two-logit, one-image all-raw kernel and its derivative in direction u. The native positive-hidden reference establishes

\[
K_0'=2\langle h_0,D_uh_0\rangle I_2
 +2\alpha^2\langle\nabla_\xi(d^\top h_0),
                   D_u\nabla_\xi(d^\top h_0)\rangle vv^\top
\succ0.
\tag{16}
\]

All hidden biases, output weights, and output biases are included in the raw Jacobian. Positive fixed-branch polynomial coefficients make the second scalar nonnegative. The literal self network has the identical formula using its own features and hidden coordinates, and hence also has a strictly positive capacity derivative.

For the three identical images, with sample-first/class-second stacking,

\[
\overline K=\mathbf1\mathbf1^\top\otimes K_0,
\qquad \overline K'=\mathbf1\mathbf1^\top\otimes K_0'.
\]

Diagonalizing the sample factor gives the exact positive trace

\[
\overline C_\lambda
=\tfrac12\operatorname{tr}[(\lambda I_6+\overline K)^{-1}\overline K']
=\tfrac32\operatorname{tr}[(\lambda I_2+3K_0)^{-1}K_0']>0.
\tag{17}
\]

The same calculation holds separately for the literal self architecture. The coincident-image six-by-six kernel is singular, but the fixed positive ridge makes the inverse continuous; no positive-definite K' claim in six dimensions is needed.

For tau>0 the images, raw Jacobians, and mean-direction Jacobian derivatives depend continuously on tau on the strict branches. A finite certificate is

\[
|C_\lambda(\tau)-\overline C_\lambda|
\le\frac{\|K'(\tau)-\overline K'\|_*}{2\lambda}
 +\frac{\|\overline K'\|_*\,\|K(\tau)-\overline K\|_{\rm op}}
        {2\lambda^2}.
\tag{18}
\]

It follows from the inverse resolvent identity and \(\|(\lambda I+K)^{-1}\|_{\rm op}\le1/\lambda\). For sufficiently small tau>0, the right side is less than half the positive value in (17), for both full and self networks. Hence both original all-raw NTK capacity derivatives remain strictly positive. The output-bias cancellation in Section 3 affects neither K nor K'.

This continuity choice of tau is compatible with (10), input/routing strictness, and rank H=3: all upper restrictions allow arbitrarily small positive tau, whereas the row-rank proof works for every such nonzero tau. After fixing one admissible tau, all strict trace, rank, sensitivity, and routing conditions persist in a full-dimensional open raw-parameter neighborhood. The construction does not claim one tau works uniformly for every lambda approaching zero; lambda is fixed first.

## 7. Result and completion boundary

There exist three distinct, everywhere-positive RGB32-by-32 images and a native two-Conv5/FC100/FC100/all-bias binary CNN reference such that:

- their last-hidden feature matrix and full logit Jacobian have row rank three;
- all three logits are exactly uniform after one common initial output-bias cancellation;
- all hidden raw sensitivities are positive and every raw-coordinate gradient is bounded away from zero for all eight label assignments on a local parameter neighborhood;
- the true first-Conv mean direction is unchanged from the base-image construction;
- the original full and literal-self NTK capacities have strictly positive mean-direction derivatives at any preselected finite ridge lambda>0;
- these strict properties hold on an open neighborhood of independently varying raw parameters, with no support splitting or tied-channel requirement.

The reference is an equilibrium construction, so it does not itself specify the positive-side initial cone or prove a positive endpoint mean margin. The codimension-three normal flow, endpoint map, and actual moving-Adam stochastic stability require the companion geometry/averaging results. Rank/sensitivity lower bounds need not be uniform in the coincident-image or zero-head limits. Those limits are used only to establish nonemptiness of a fixed finite reference with strict margins.
