# A native all-bias CNN near binary uniform-output equilibrium

2026-10-09. Bounded constructive result: a nonempty open reference neighborhood for native 5-by-5 convolutions, hidden FC layers, every ordinary bias, the original full/literal-self NTK capacity, and the fixed-state averaged Adam field. A local trajectory theorem is proved for that averaged ODE. No transfer to the actual moving stochastic Adam trajectory is asserted here.

## 1. Architecture and a nonempty strict branch

Use one fixed positive RGB image of size 32 by 32 and the architecture

    Conv5/pad2, 3→16, bias → ReLU → MaxPool2
    Conv5/pad2, 16→16, bias → ReLU → MaxPool2
    flatten1024 → FC100, bias → ReLU
    FC100, bias → ReLU → binary FC2, bias.

All raw parameters, including all biases, are independently trainable. There is no BN, tying, freezing, or moment resetting. N=1 is sufficient for the result; it is a deliberate restriction.

Choose strictly positive hidden weights and sufficiently small strictly positive hidden biases. Strict ReLU and pool margins exist analytically. For example, start with zero hidden biases and identical RGB planes

\[
X_{r,c}=\frac{1+r+32c}{1024},\qquad0\le r,c<32.
\]

Initially take each convolution's center-offset weights positive and all other offsets zero. Both pools then have unique bottom-right winners because the input is strictly increasing in each coordinate, and every hidden preactivation is positive. Replace every zero off-center coefficient by a sufficiently small positive number, and every hidden bias by a sufficiently small positive number. The finite strict margins persist by continuity, in both the full network and the literal self network defined below. Thus all 25 offsets genuinely participate. Both hidden FC weight matrices may have arbitrary strictly positive entries. Further sufficiently small independent perturbations of all hidden weights and biases preserve these properties and can make the spatial channel maps nonproportional. The argument does not require aligned or tied hidden channels. Zero hidden biases are also permitted in the formulas below, but a strictly positive choice puts the reference inside the positive hidden-parameter cone.

Let \(\xi\) denote all hidden raw parameters and \(h(\xi)\in\mathbb R^{100}\) the last hidden vector. It is strictly positive. For a chosen first-Conv channel, let \(m=u^\top\xi\) be its actual spatial mean preactivation. Its nonzero coordinates are the actual augmented patch mean:

\[
u_j=\mathbb E_{\rm spatial}[X_{{\rm patch},j}]>0
\quad(j=1,\ldots,75),\qquad u_{\rm bias}=1.
\tag{1}
\]

The expectation includes padding and all 32-by-32 first-layer output positions. Padding does not make any of these 75 averages zero. Thus \(u\) is fixed and coordinatewise positive on the selected first-filter block \(A\), and zero elsewhere.

Every component of \(h'=D_uh\) is strictly positive. At the first layer, the directional preactivation derivative is at least 1 because of the bias coordinate in (1). Positive kernels, the active ReLUs, selected positive pool paths, and dense positive FC weights propagate a strictly positive derivative to every component of \(h\).

For each individual \(j\in A\), \(\partial_j h\) is nonnegative and has a strictly positive route to every last-hidden coordinate. For a first-filter spatial weight, select an interior final-pool window, the center second-Conv offset, and the corresponding first-pool winner; its first patch lies entirely inside the image. Its input coefficient and every subsequent path coefficient are positive. The bias case is immediate. Therefore for every \(d\in\mathbb R^{100}_{>0}\),

\[
\psi=d^\top h>0,\qquad
\partial_j\psi>0\quad(j\in A).
\tag{2}
\]

The same interior-path argument covers every raw Conv1/Conv2 spatial coefficient, not just the selected mean block. Each FC weight has positive input and a positive outgoing route, and every hidden bias has a positive outgoing route. Consequently \(\partial_j\psi>0\) for every hidden raw coordinate. Every output-head coordinate and output-bias coordinate also has a nonzero derivative of z, with its class-dependent sign. No zero raw-coordinate derivative is needed at the reference.

The literal self network deletes all first-Conv channels other than the selected channel and the matching input-channel columns of Conv2. All remaining hidden layers, their biases, and the output head/bias are retained without refitting. The same construction gives positive \(h_{\rm self}\), \(D_uh_{\rm self}>0\), and strict routing margins. With positive later biases it need not be a simple scalar rescaling of the full network; none of the arguments requires such a rescaling.

## 2. Output cancellation creates a genuine equilibrium with positive derivatives

Choose \(\alpha>0\), place the two output-weight rows at \(+\alpha d\) and \(-\alpha d\), and place output biases at \(+b\) and \(-b\). Define the half-contrast

\[
z(\theta)=\frac{f_+(\theta)-f_-(\theta)}2.
\]

At this reference,

\[
z=\alpha\psi+b,\qquad
\partial_jz=\alpha\partial_j\psi>0\quad(j\in A).
\tag{3}
\]

Set \(b=-\alpha\psi\) to obtain a uniform-output equilibrium \(\theta_*\) with \(z(\theta_*)=0\). Changing only \(b\) to \(-\alpha\psi+z_0\) gives any prescribed small \(z_0>0\), without changing any derivative in (3), any gate, or any NTK. The raw output rows/biases remain independent coordinates: their antisymmetry specifies a reference state, not tied training parameters.

At arbitrary nearby states the logits need not be centered. Binary CE nevertheless depends only on their half-contrast:

\[
L(\theta,Y)=\log(2\cosh z(\theta))-Yz(\theta),
\qquad Y\in\{-1,1\}.
\tag{4}
\]

Positive outgoing contrast supplies (3) geometrically. It does not assume the desired loss-gradient sign. All hidden parameters are allowed to move in the neighborhood.

## 3. Exact full and literal-self capacity formulas, including all bias blocks

For the reference head in Section 2, put \(v=(1,-1)^\top\). Differentiating the two logits with respect to every independent raw parameter gives

\[
K=JJ^\top=(\|h\|^2+1)I_2
  +\alpha^2\|\nabla_\xi\psi\|^2vv^\top.
\tag{5}
\]

The first term includes the free output-weight block \(\|h\|^2I_2\) and the output-bias block \(I_2\). The second includes all convolution/hidden-FC weights and all hidden biases, whether their reference values are positive or zero. No raw Jacobian block is omitted.

Since the mean direction changes only the first-filter augmented block,

\[
K'=D_uK
=2\langle h,h'\rangle I_2
 +2\alpha^2\langle\nabla_\xi\psi,
                   D_u\nabla_\xi\psi\rangle vv^\top.
\tag{6}
\]

Within a fixed active/winner branch, each hidden output is a polynomial with nonnegative coefficients in hidden weights and biases. At positive weights and nonnegative biases, its first derivatives and mixed derivatives in a nonnegative direction are nonnegative. Weight sharing only adds polynomial terms. Hence

\[
\nabla_\xi\psi\ge0,\qquad
D_u\nabla_\xi\psi\ge0.
\]

Together with \(h,h'>0\), equation (6) yields

\[
K'_{\rm full}\succeq\ell_{\rm full} I_2,
\qquad \ell_{\rm full}=2\langle h,D_uh\rangle>0.
\tag{7}
\]

The same exact formulas apply to the literal self network with its own hidden parameter vector and \(h_{\rm self}\):

\[
K'_{\rm self}\succeq\ell_{\rm self}I_2,
\qquad \ell_{\rm self}
=2\langle h_{\rm self},D_uh_{\rm self}\rangle>0.
\tag{8}
\]

Both lower bounds are independent of \(\alpha\) and of the output bias. Thus, for every \(\lambda>0\), the original capacity functional satisfies

\[
D_u\Bigl[\tfrac12\log\det(\lambda I+K)\Bigr]
=\tfrac12\operatorname{tr}[(\lambda I+K)^{-1}K']>0
\tag{9}
\]

for full and literal self networks, even at \(\alpha=0\). At \(\alpha=0\), hidden-parameter Jacobian blocks vanish, but the free output-weight block alone gives the strictly positive derivative. The classifier being uniformly uninformative therefore does not make this capacity derivative vanish.

Continuity gives a full-dimensional raw-parameter neighborhood of each \(\theta_*\) with positive capacity slopes, positive target-block derivatives, and strict gates/winners in both architectures. Explicitly, with matching reference \(J_*,J_*'\), the bound

\[
E_K=2\bigl(\|J'-J_*'\|_F\|J_*\|_F
 +\|J-J_*\|_F\|J_*'\|_F
 +\|J-J_*\|_F\|J'-J_*'\|_F\bigr)
\]

and \(E_{K,{\rm full}}<\ell_{\rm full}\), \(E_{K,{\rm self}}<\ell_{\rm self}\) suffice. Target-block conditions can similarly be certified by \(|\partial_jz-\alpha\partial_j\psi|<\alpha\partial_j\psi/2\). These are norm/coordinate distances to the constructive reference, not the requested drive sign written as an assumption. The neighborhood permits arbitrary small independent changes of every bias, both head rows, every kernel coefficient, and every FC weight.

## 4. All-coordinate Adam anisotropy does not change the mean direction

Hold any state in that neighborhood fixed. With iid uniform binary labels, let \(s_j=\partial_jz\). Its raw gradient is

\[
g_j=s_j(\tanh z-Y).
\tag{10}
\]

For standard Adam with \(\epsilon>0\), independent labels over optimizer steps, and stationary fixed-state EMA moments, the shifted-symmetric scalar lemma gives

\[
F_j(\theta):=\mathbb E\!\left[
\frac{m_j}{\sqrt{v_j}+\epsilon}\right]
=c_j(\theta)s_j\tanh z,
\qquad
\frac{\epsilon}{(\epsilon+G)^2}\le c_j\le\frac1\epsilon,
\tag{11}
\]

provided \(|g_j|\le G\) uniformly over the neighborhood. The sign result does not require the \(c_j\) to coincide across coordinates. A zero \(s_j\tanh z\) simply gives \(F_j=0\), and its coefficient can be chosen within the displayed interval.

For completeness, the scalar lemma follows by writing the EMA ratio as a sum of numerator-weighted terms. Conditional on all other iid gradients, each term has the form \(x/(\sqrt{b x^2+C}+\epsilon)\), an odd increasing function. On the bounded gradient range its derivative lies between \(\epsilon/(\epsilon+G)^2\) and \(1/\epsilon\). Translating a symmetric scalar distribution by its mean proves (11). Correlation between different coordinates does not enter this marginal calculation.

For full-batch labels reused a fixed finite H times, the same statement applies to each stationary task phase after grouping EMA weights by independent task labels. Averaging these phase fields also preserves the positive coefficient bounds. This statement is about the frozen-state phase-stationary field, not the conditional update at a realized momentum state.

Because \(u_j>0\) and \(s_j>0\) for every \(j\in A\),

\[
D_mF:=u^\top F
=\tanh z\sum_{j\in A}u_jc_js_j>0
\quad\text{whenever }z>0.
\tag{12}
\]

Thus arbitrary diagonal anisotropy within (11) cannot reverse the first-Conv mean drift. The averaged descent vector is \(-F\), so it decreases the mean. At \(z=0\) the averaged field is zero while the coordinate derivatives in (3) and the capacity margins (7)–(8) remain positive. For \(z<0\) the mean drift reverses; the positive side is essential.

## 5. A nonempty local tube reaching equilibrium for the averaged ODE

This section supplies trajectory preservation for \(\dot\theta=-F(\theta)\), not for stochastic Adam itself. The fixed-state field is locally Lipschitz here: couple its stationary label histories, use the smooth bounded gradient on the branch, bound EMA first-moment differences by the gradient difference, and bound RMS second-moment differences by the corresponding weighted Euclidean norm. The denominator is at least epsilon, so taking expectations preserves a local Lipschitz bound. The same argument applies to the finite-H phase average. Choose a closed raw-parameter ball \(B_r(\theta_*)\) contained in the simultaneous strict routing/capacity/target-derivative neighborhood. On it let

\[
M\ge\sup\|\nabla z\|,\quad
s_j\ge\underline s_j>0\ (j\in A),\quad
c_-=\frac{\epsilon}{(\epsilon+2M)^2},\quad c_+=1/\epsilon.
\]

Since \(|\tanh z-Y|\le2\), the choice \(G=2M\) in (11) is valid. The two raw output-bias derivatives are always \(+1/2,-1/2\); consequently \(\|\nabla z\|^2\ge1/2\), at every state, without a genericity assumption.

For \(z>0\),

\[
\dot z=-\tanh z\,T(\theta),\qquad
T=\sum_jc_js_j^2\ge c_-/2.
\tag{13}
\]

Moreover the speed divided by the decrease of z obeys

\[
\frac{\|\dot\theta\|}{-\dot z}
=\frac{\|\operatorname{diag}(c_j)\nabla z\|}
       {\nabla z^\top\operatorname{diag}(c_j)\nabla z}
\le B:=\frac{2c_+M}{c_-}.
\tag{14}
\]

Start at the reference hidden weights/head and shift only the two output biases by \(+z_0,-z_0\). If

\[
0<z_0<\min\{r/(4\sqrt2),\ r/(4B)\},
\tag{15}
\]

the initial distance from \(\theta_*\) is \(\sqrt2z_0<r/4\), and (14) bounds the entire subsequent path length by \(Bz_0<r/4\). A first-exit argument therefore keeps the trajectory in \(B_{r/2}(\theta_*)\) for all time. The same argument applies to an open set of nearby initial states satisfying the corresponding strict distance inequality and \(z>0\).

Equation (13) preserves \(z>0\) at finite times and implies \(z(t)\to0\), indeed exponentially with any lower bound for \((\tanh z)/z\) on \([0,z_0]\). The finite path-length estimate gives \(\theta(t)\to\theta_\infty\) with \(z(\theta_\infty)=0\). All target derivative signs and both original capacity margins persist to this limiting equilibrium.

The mean decrease is quantitatively strict. Set \(S_-=\sum_{j\in A}u_j\underline s_j>0\). Dividing its rate by (13) and integrating over z gives

\[
m(\theta_0)-m(\theta_\infty)
=\int_0^{z_0}\frac{\sum_{j\in A}u_jc_js_j}{T}\,dz
\ge\frac{c_-S_-}{c_+M^2}\,z_0>0.
\tag{16}
\]

Thus this construction includes an actual moving all-coordinate averaged trajectory, with no assumed boundedness, and not merely a pointwise sign. To obtain an actual stochastic Adam theorem, one still needs a controlled averaging/noise error and a positive-probability or almost-sure nonexit argument; none is inferred from (16).

## 6. Small-head/small-bias limit and limitations

For a fixed positive hidden network and d, letting \(\alpha\downarrow0\) leaves the capacity lower bounds in (7)–(8) exactly unchanged. To make all bias values tend to zero as well, fix the positive hidden weights and choose the hidden biases as \(\tau b_h\) with \(b_h>0\), letting \(\tau\downarrow0\). Choose these weights from the zero-bias strict-branch construction in Section 1, so that the branch remains strict down to \(\tau=0\). Then \(h(\tau),D_uh(\tau)\) and their self counterparts converge to strictly positive limits. Both capacity lower bounds therefore stay uniformly bounded below by positive constants for small \(\tau\). Set the output-bias contrast to \(b=-\alpha\psi(\tau)+z_0\); it also tends to zero when \(\alpha,z_0\to0\).

Target derivatives \(\partial_jz=\alpha\partial_j\psi(\tau)\) remain positive for every \(\alpha>0\), but tend to zero. Hence the allowed neighborhood for preserving their signs generally shrinks with \(\alpha\), and the strict mean-drift/mean-sink lower bounds are not uniform as \(\alpha\to0\). A concrete simultaneous choice is \(\tau=\alpha\) and \(z_0=c\alpha^2>0\) for a sufficiently small fixed c and sufficiently small \(\alpha\). The hidden-positive-cone and target-derivative radii may then be chosen of order \(\alpha\), while the gate/capacity margins and M have uniform local bounds, so (15) holds. For each positive \(\alpha\), the reference and the admitted ball have strictly positive hidden biases, and all biases tend to zero only in the limiting family.

The output-bias cancellation is for the full network. In the literal self network that same retained bias generally makes its z negative; its CE drift need not point toward decreasing mean. The result needed here is instead that **the actual full-network averaged Adam mean drift agrees with the original literal-self capacity direction**. The two capacity signs are proved by (5)–(9), independently of any self-network CE claim.

This construction uses one image and binary labels; it does not yet establish the analogous diagonal-field formula for a many-image minibatch problem, nor an actual RL-CIFAR trajectory. It does cover native spatial kernels, hidden FC layers, all freely trainable biases, overlapping channels, pool nonwinners, and a full-dimensional neighborhood of untied raw parameters. ReLU remains strictly active, and the equilibrium is uniform output through bias cancellation rather than neuron death.
