# True reshuffled batch-16 Adam: a positive output-bias normal coefficient

2026-10-09. This note proves a strict positive stationary output-bias coefficient for binary batch-16 Adam with labels reused across newly shuffled epochs. The explicit certificates cover \(N=32,E=2,H=4\) and \(N=48,E=2,H=6\), with the actual \(\beta_1=.9,\beta_2=.999,\epsilon=10^{-8}\). No fixed partition, independent-batch-noise replacement, or moment reset is used.

The result is a frozen-state field calculation and a local smoothness lemma. Actual long-time CNN behavior additionally needs the nonbias normal-spectrum, geometry, and stochastic-averaging arguments.

## 1. Exact model and common-mode response

Each task draws independent labels \(Y_{k,n}\in\{-1,+1\}\), once for all \(N\) images. In each of \(E\) epochs, a fresh independent uniform permutation of the images is drawn and split into batches of \(B=16\) distinct images. There are \(H=EN/B\) updates per task. Every batch uses the task's original image labels. Entire task innovations are iid across tasks; the batches within a task are generally dependent, including across reshuffled epochs.

Let \(z_n=(f_{+,n}-f_{-,n})/2\). For the two raw output biases, a batch gradient is

\[
 g_{b_+,t}=\frac12\left[\frac1B\sum_{n\in\mathcal B_t}\tanh z_n-X_t\right],
 \qquad g_{b_-,t}=-g_{b_+,t},
 \qquad X_t=\frac1B\sum_{n\in\mathcal B_t}Y_{k(t),n}.
 \tag{1.1}
\]

At a common contrast \(z_n=z\), put \(\mu=\tanh z\). Since \(\mu'(0)=1\), the derivative in the common-output direction at zero is the derivative with respect to \(\mu\). Define normalized stationary EMA quantities

\[
 a_\ell=(1-\beta_1)\beta_1^\ell,\qquad
 b_\ell=(1-\beta_2)\beta_2^\ell,
\]
\[
 A=\sum_{\ell\ge0}a_\ell X_{t-\ell},\qquad
 B_0=\sum_{\ell\ge0}b_\ell X_{t-\ell},\qquad
 V=\sum_{\ell\ge0}b_\ell X_{t-\ell}^2,\qquad \sigma=\sqrt V,
 \quad e=2\epsilon.
 \tag{1.2}
\]

Here \(B_0\) is an EMA random variable; \(B=16\) remains the batch size. Scaling out the raw bias sensitivity \(1/2\) gives

\[
 q_{b_+}(\mu)
 =\frac{\mu-A}{\sqrt{\sum_\ell b_\ell(\mu-X_{t-\ell})^2}+e},
 \qquad q_{b_-}(\mu)=-q_{b_+}(\mu).
 \tag{1.3}
\]

The exact expected common-mode derivative at stationary phase \(r\) is

\[
 \boxed{\quad
 \Gamma_r
 =\mathbb E\left[
 \frac1{\sigma+e}
 -\frac{A B_0}{\sigma(\sigma+e)^2}
 \right].\quad}
 \tag{1.4}
\]

In (1.4) a zero-\(\sigma\) history uses the derivative value \(1/e\); the product term is assigned zero there. Such infinite histories have probability zero in this model, but this convention also makes the underlying derivative argument explicit.

The second term in (1.4) need not have a favorable pointwise sign or be smaller than the first term pointwise. The proof below controls its expectation with task-level bounds.

## 2. Universal stationary normalization and differentiation

Weighted Cauchy--Schwarz gives, for any real history \(x_\ell\),

\[
 \left|\sum_\ell a_\ell x_\ell\right|
 \le K_0\left(\sum_\ell b_\ell x_\ell^2\right)^{1/2},
 \qquad
 K_0^2=\sum_{\ell\ge0}\frac{a_\ell^2}{b_\ell}
 =\frac{(1-\beta_1)^2}
 {(1-\beta_2)(1-\beta_1^2/\beta_2)}.
 \tag{2.1}
\]

For the standard betas this is the exact rational constant

\[
 K_0^2=\frac{370}{7}<64,\qquad K_0<8.
 \tag{2.2}
\]

This stationary constant includes the factor \(1-\beta_1\). It is distinct from the larger conservative bound used for globally bias-corrected finite-time Adam.

Apply (2.1) to \(x_\ell=\mu-X_{t-\ell}\). The derivative of the RMS where positive has magnitude at most one. Therefore

\[
 |q_{b_+}'(\mu)|\le \frac1e+\frac{K_0}{4e}.
 \tag{2.3}
\]

Indeed, its quotient correction is at most
\(K_0\sigma_\mu/(\sigma_\mu+e)^2\le K_0/(4e)\).
If \(\sigma_\mu=0\), all history entries equal \(\mu\), and the nearby scalar quotient is \(h/(|h|+e)\), whose derivative at \(h=0\) is \(1/e\). Thus the uniform derivative bound also permits differentiation under the expectation by dominated difference quotients. This establishes (1.4) without assuming a deterministic positive RMS floor.

## 3. The only independence used: different tasks

Every label-independent batch of \(B\) distinct images has the same unconditional moments

\[
 r_2:=\mathbb E X_t^2=\frac1B,\qquad
 r_4:=\mathbb E X_t^4=\frac{3B-2}{B^3}.
 \tag{3.1}
\]

For \(B=16\),

\[
 r_2=\frac1{16},\qquad r_4=\frac{46}{4096}.
 \tag{3.2}
\]

These moments hold even after a realized schedule is fixed, because the task image labels are iid. They do not make two batches independent when their reused image labels overlap.

For \(0<\beta<1\), group the geometric EMA weights by historical task. At phase \(r\in\{1,\ldots,H\}\), the current partial task has weight \(1-\beta^r\); previous complete tasks have weights

\[
 \beta^r(1-\beta^H),\
 \beta^{r+H}(1-\beta^H),\
 \beta^{r+2H}(1-\beta^H),\ldots.
\]

Put

\[
 T(\beta,H)=\frac{1-\beta^H}{1+\beta^H}.
 \tag{3.3}
\]

The sum of squared task weights is

\[
 (1-\beta^r)^2+\beta^{2r}T(\beta,H)
 \le T(\beta,H).
 \tag{3.4}
\]

For a direct proof, let \(x=\beta^r\in[\beta^H,1]\). The left side is the convex quadratic \((1-x)^2+x^2T\), whose values at both endpoints equal \(T\). Thus (3.4) is uniform over every task phase, including the partial current task.

Write \(T_i=T(\beta_i,H)\). Within one task, Cauchy--Schwarz gives

\[
 \mathbb E\left(\sum_{t\text{ in task}}w_tX_t\right)^2
 \le r_2\left(\sum_{t\text{ in task}}w_t\right)^2.
\]

Different task contributions have mean zero and are independent. Consequently

\[
 \mathbb E A^2\le r_2T_1,\qquad
 \mathbb E B_0^2\le r_2T_2,\qquad
 \mathbb E|A B_0|\le r_2\sqrt{T_1T_2}.
 \tag{3.5}
\]

The within-task bound allows arbitrary positive correlation; it is not a hidden independent-reshuffled-batch approximation.

## 4. A lower-tail bound for the RMS, preserving reuse

Let \(V_k=\sum_{t\text{ in task }k}b_tX_t^2\), and let
\(W_k=\sum_{t\text{ in task }k}b_t\). Then \(V=\sum_kV_k\), \(\sum_kW_k=1\), and the \(V_k\) are independent across tasks. Jensen/Cauchy--Schwarz within a task gives

\[
 \mathbb EV_k=r_2W_k,\qquad
 \mathbb EV_k^2\le r_4W_k^2.
 \tag{4.1}
\]

For every \(u\ge0\), \(e^{-u}\le1-u+u^2/2\). Therefore, for every \(t\ge0\),

\[
 \mathbb E e^{-tV_k}
 \le 1-tr_2W_k+\frac{t^2r_4W_k^2}{2}
 \le \exp\left[-tr_2W_k+\frac{t^2r_4W_k^2}{2}\right].
\]

Multiplying independent task factors, taking the infinite-history limit, and using (3.4) yields

\[
 \mathbb E e^{-tV}
 \le \exp\left[-tr_2+\frac{t^2r_4T_2}{2}\right].
 \tag{4.2}
\]

The infinite-history passage is valid by monotone convergence of the nonnegative partial sums and bounded convergence of their exponentials. This lower-tail estimate does not invoke a Gaussian approximation or a centered Bernstein bound.

Chernoff optimization gives, for \(0<\ell<r_2\),

\[
 \boxed{\quad
 \mathbb P(V<\ell)
 \le \exp\left[-\frac{(r_2-\ell)^2}{2r_4T_2}\right].
 \quad}
 \tag{4.3}
\]

At the convenient threshold \(\ell=r_2/4=1/64\),

\[
 p_{\rm bad}:=\mathbb P(V<1/64)
 \le \exp\left[-\frac{9}{92T_2}\right].
 \tag{4.4}
\]

## 5. A general finite lower certificate

The positive term in (1.4) obeys

\[
 \mathbb E\frac1{\sigma+e}
 \ge\frac1{\mathbb E\sigma+e}
 \ge\frac1{\sqrt{r_2}+e},
 \tag{5.1}
\]

since \(\mathbb EV=r_2\).
On \(V\ge\ell\), equation (3.5) gives

\[
 \mathbb E\left[
 \frac{|AB_0|}{\sigma(\sigma+e)^2}\,
 \mathbf1_{V\ge\ell}\right]
 \le\frac{r_2\sqrt{T_1T_2}}
 {\sqrt\ell(\sqrt\ell+e)^2}.
 \tag{5.2}
\]

On the remaining histories, \(|A|\le K_0\sigma\) and
\(|B_0|\le\sigma\), so

\[
 \frac{|AB_0|}{\sigma(\sigma+e)^2}
 \le\frac{K_0\sigma}{(\sigma+e)^2}
 \le\frac{K_0}{4e}.
 \tag{5.3}
\]

Combining the bounds yields, uniformly over every phase,

\[
 \boxed{
 \Gamma_r\ge
 \frac1{\sqrt{r_2}+e}
 -\frac{r_2\sqrt{T_1T_2}}{\sqrt\ell(\sqrt\ell+e)^2}
 -\frac{K_0}{4e}
 \exp\left[-\frac{(r_2-\ell)^2}{2r_4T_2}\right].
 }\tag{5.4}
\]

This is valid for every finite \(H\). A strictly positive right side is an explicit sufficient condition, not an assertion that the bound stays positive for all reuse durations.

For the particular \(B=16,\ell=1/64,e=2\epsilon,K_0<8\), a simpler conservative form is

\[
 \boxed{
 \Gamma_r>
 \frac1{1/4+2\epsilon}
 -32\sqrt{T_1T_2}
 -\frac1{\epsilon}\exp\left[-\frac9{92T_2}\right].
 }\tag{5.5}
\]

## 6. Rational certificates at the actual standard constants

Set \(\beta_1=9/10,\beta_2=999/1000,\epsilon=10^{-8}\). All the following comparisons are rational:

| Task length | Bound on \(\sqrt{T_1T_2}\) | Lower bound on \(9/(92T_2)\) | Rational lower bound from (5.5) |
|---|---:|---:|---:|
| \(H=4\) | \(21/1000\) | \(48\) | \(3.3279993247\ldots\) |
| \(H=6\) | \(31/1000\) | \(32\) | \(2.9847166156\ldots\) |

For the first column of bounds, square the stated rational number and compare with \(T_1T_2\). For the tail, the elementary inequality \(e>2\) gives \(e^{-x}<2^{-m}\) whenever \(x>m>0\). Thus the exact lower bounds used in the table are

\[
 \frac1{1/4+2\epsilon}
 -32\frac{21}{1000}-\frac{2^{-48}}{\epsilon}
 \quad(H=4),
\]
\[
 \frac1{1/4+2\epsilon}
 -32\frac{31}{1000}-\frac{2^{-32}}{\epsilon}
 \quad(H=6).
 \tag{6.1}
\]

Both exceed \(29/10\). In particular,

\[
 \boxed{\Gamma_r>\frac{29}{10}>0
 \quad\text{for every phase, for }H=4\text{ or }6.}
 \tag{6.2}
\]

The decimal values are only display values of rational expressions, not floating-point sign certificates. A minimal independent arithmetic check is:

    from fractions import Fraction as F
    b1, b2, eps = F(9,10), F(999,1000), F(1,10**8)
    for H, q, m in [(4,F(21,1000),48), (6,F(31,1000),32)]:
        T1 = (1-b1**H)/(1+b1**H)
        T2 = (1-b2**H)/(1+b2**H)
        assert T1*T2 < q*q
        assert F(9,92)/T2 > m
        lower = 1/(F(1,4)+2*eps) - 32*q - F(1,2**m)/eps
        assert lower > F(29,10)
    assert (1-b1)**2/((1-b2)*(1-b1*b1/b2)) == F(370,7)

These assertions were executed successfully. They prove the coefficients for \(N=32,E=2,H=4\) and \(N=48,E=2,H=6\) under genuinely new permutations and regrouping each epoch.

## 7. Exact normal-matrix bias coefficient and its factors

Let the frozen field sum the expected Adam quotients over \(H\) phases, and let \(L=D_z\overline F\) be its raw-parameter by output linearization at zero contrast. Image exchangeability of the actual shuffle law implies that each output-bias response row is constant across image index \(n\). Taking the uniform-output directional derivative therefore gives exactly

\[
 L_{b_+,n}=\frac1N\sum_{r=1}^H\Gamma_r,
 \qquad
 L_{b_-,n}=-\frac1N\sum_{r=1}^H\Gamma_r.
 \tag{7.1}
\]

The raw logit-contrast Jacobian has bias columns
\(J_{n,b_+}=1/2\), \(J_{n,b_-}=-1/2\). Consequently

\[
 \boxed{
 J_bL_b=\beta_b\,\mathbf1\mathbf1^T,
 \qquad \beta_b=\frac1N\sum_{r=1}^H\Gamma_r
 >\frac{29H}{10N}>0.
 }\tag{7.2}
\]

There is no missing factor of \(1/2\): the two bias-column products are respectively \(+\frac12(\sum\Gamma/N)\) and \(+\frac12(\sum\Gamma/N)\). The normalization \(e=2\epsilon\) in (1.3) already absorbed each raw gradient's magnitude \(1/2\). If the field uses a phase average instead of a phase sum, divide both \(L\) and \(\beta_b\) by \(H\).

This is the common-mode positive-semidefinite block needed by a separate nonbias small-sensitivity proof. Entrywise signs or a positive common mode alone do not establish the other \(N-1\) normal directions.

## 8. Local C3 smoothness despite possible balanced batches

A batch can have exactly eight labels of each sign, so a deterministic nonzero bias-gradient floor is false. Nevertheless the stationary bias field is locally \(C^3\), with finite derivative bounds, for the two explicit task lengths.

Work in a contrast neighborhood satisfying
\[
 \max_n|\tanh z_n|\le1/16.
 \tag{8.1}
\]

From every complete past task, select its first batch in its first epoch. For that batch,
\[
 p_0:=\mathbb P(X=0)=\binom{16}{8}/2^{16}
 =12870/65536<1/5.
\]
These balance events are independent across tasks, regardless of the later reused-label batches. If \(X\ne0\), its magnitude is at least \(1/8\); throughout (8.1) the normalized bias gradient
\(B^{-1}\sum_{n\in\mathcal B}\tanh z_n-X\) then has magnitude at least \(1/16\).

Let \(K\ge0\) count complete past tasks before the nearest selected batch with \(X\ne0\). Then
\(\mathbb P(K=k)=(1-p_0)p_0^k\). The lag of that selected batch is at most \(H(k+2)\), uniformly over the current phase. The normalized stationary RMS therefore satisfies, uniformly throughout (8.1),

\[
 \sigma(\theta)^2
 \ge \frac{1-\beta_2}{256}\,\beta_2^{H(K+2)}.
 \tag{8.2}
\]

For every \(q>0\) such that \(p_0\beta_2^{-Hq/2}<1\),

\[
 \mathbb E\sup_{\theta\text{ in }(8.1)}\sigma(\theta)^{-q}
 \le
 \left(\frac{256}{1-\beta_2}\right)^{q/2}
 \beta_2^{-Hq}
 \frac{1-p_0}{1-p_0\beta_2^{-Hq/2}}
 <\infty.
 \tag{8.3}
\]

In particular \(q=6\) is admissible for \(H=4,6\); this is another rational comparison. Every finite first, second, and third derivative of the normalized gradient is uniformly bounded on a compact contrast neighborhood. Differentiating the RMS and quotient three times is then dominated by a constant times \(1+\sigma^{-5}\); epsilon bounds the outer quotient denominators. Equation (8.3) justifies differentiation under the expectation through order three, and dominated convergence makes those derivatives continuous. Composing with a smooth fixed-routing CNN preserves this \(C^3\) statement for the bias rows.

This argument deliberately uses inverse-moment domination instead of asserting that every realized batch has a nonzero gradient. Higher finite regularity can be checked with the same criterion; \(C^\infty\) is not claimed from this fixed geometric-tail bound.

## 9. Scope

The proofs retain all task-label reuse, newly reshuffled partitions each epoch, partial current-task histories, and all older task moments. The marginal moment and task-group arguments even tolerate arbitrary dependence among the within-task schedules, provided batches are label-independent and contain 16 distinct images; image-exchangeable scheduling is additionally used for the exact matrix form in section 7.

The positive certificate is for the stationary frozen bias response, with standard fixed betas and epsilon. Transferring it to moving actual Adam requires the existing taskwise averaging theorem and a verified local normally attracting reference. This note does not establish nonbias normal stability, capacity-self direction, a global invariant region, practical learning rates, ten-class behavior, or the \(H=30000\) task duration of the original RL-CIFAR run. Equation (5.4) remains a valid finite-\(H\) bound, but its positivity is not asserted at such large \(H\).
