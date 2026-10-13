# A strict hold in the original two-layer capacity kernel with no one-step exit label

This is an exact mathematical counterexample to a universal **one-step** exit
claim. It is not a counterexample to eventual exit after some finite label
sequence at a fixed positive learning rate.

## Model and strict hold

Take five inputs $x_n=e_n\in\mathbb R^5$, three hidden ReLU units, a scalar
readout, and train both layers and both biases. Write the preactivation matrix
(rows = inputs, columns = units) and readout as

$$
Z=\begin{pmatrix}
-4&-4&-4\\
-7&-4&-9\\
2&4&1\\
-8&-3&-9\\
8&5&6
\end{pmatrix},\qquad v=(1,1,1)^T.
$$

This is realized by hidden bias zero and hidden weights $W=Z^T$. The output
bias is zero initially. All three gates have the active set $\{3,5\}$; in
particular the target, unit 3, is open on only $2/5$ of the inputs.

Let $\Theta=I_5+11^T$, $k=\Theta1/5=(6/5)1$, $D_j=\operatorname{diag}(d_j)$,
$F=\operatorname{ReLU}(Z)$, $\lambda=1$, and

$$
\mathcal K=\sum_{j=1}^3v_j^2D_j\Theta D_j+FF^T+11^T,
\quad E=(I_5+\mathcal K)^{-1}.
$$

Use the same uniform-position direction as the source convention:
$\delta z_i=k\,\delta t$. All preactivations are nonzero. Thus the hard-gate
derivative vanishes locally, and the exact full capacity derivative is

$$A_i=(k\odot d_i)^TEf_i.$$

The self derivative uses the identical formula with
$\mathcal K_i=v_i^2D_i\Theta D_i+f_if_i^T+11^T$.
Direct rational matrix inversion gives

$$A_3=-\frac{36}{3377}<0,
\qquad A_{3,\rm self}=\frac{336}{2285}>0.$$

This is a genuine strict hold under the source definition `A <= 0 and self > 0`.
It uses the readout-feature route of the original two-layer kernel; it does
not establish a hold caused by a nonzero smooth-gate derivative.

For a short hand derivation, eliminate the three inactive input coordinates.
The full inverse restricted to active inputs is the inverse of
$\frac14\left(\begin{smallmatrix}113&181\\181&529\end{smallmatrix}\right)$,
whereas the self inverse is the inverse of
$\frac14\left(\begin{smallmatrix}17&29\\29&157\end{smallmatrix}\right)$.
In both cases the derivative vector is $(6/5)(1,1)^T$ and the target feature
vector is $(1,6)^T$.

## MSE: exact enumeration of all labels

For loss $L=\frac1{2N}\sum_n(f_n-y_n)^2$, let $y\in\{-1,+1\}^5$ and
$\eta=1/1000$. With residual $r=f-y$, one full-batch SGD step gives

$$
z'_{nj}=z_{nj}-\frac\eta5v_j\left(d_{nj}r_n+\sum_m d_{mj}r_m\right),
\quad
v'_j=v_j-\frac\eta5\sum_nF_{nj}r_n,
\quad
c'=c-\frac\eta5\sum_nr_n.
$$

`verify_exact.py` enumerates all 32 label vectors, computes the updated
parameters and both capacity derivatives in exact SymPy rational arithmetic,
and asserts both strict inequalities. No floating-point tolerance is used.
Every gate remains in its original chamber and every label remains held.
For presentation only, the resulting derivative ranges are

- full $A_3$: $[-0.01182888121,-0.01167959754]$;
- self $A_{3,\rm self}$: $[0.14798408578,0.14809014146]$.

The exact rational values for every label are saved in
`exact_counterexample.json`.

## Binary cross-entropy: a uniform analytic bound for every label

Keep exactly the same initial network and capacity definition. For mean
binary logistic cross-entropy, $r_n=\operatorname{sigmoid}(f_n)-y_n$ lies in
$[-1,1]$ for every $y_n\in\{0,1\}$. Take the fixed learning rate
$\eta=10^{-6}$. No numerical approximation to sigmoid is needed.

The same update formulas imply, uniformly over all residual vectors in
$[-1,1]^5$,

$$
|\Delta z_{nj}|\le\tfrac35\eta\quad(n\in\{3,5\}),
\qquad
|\Delta z_{nj}|\le\tfrac25\eta\quad(n\notin\{3,5\}),
$$

and $|\Delta v_j|\le(2,9/5,7/5)_j\eta\le2\eta$.
All gates therefore stay fixed. In operator/Frobenius norms,

$$
\|F\|\le\|F\|_F=\sqrt{146}<13,
\quad\|\Delta F\|\le\|\Delta F\|_F
\le\tfrac35\sqrt6\eta<\tfrac32\eta.
$$

Since $\|D\Theta D\|=3$ for their common gate $D$,

$$
\|\Delta\mathcal K\|
\le 3\sum_j|\Delta(v_j^2)|
  +2\|F\|\|\Delta F\|+\|\Delta F\|^2
\le75\eta+\tfrac{153}{4}\eta^2<76\eta.
$$

Both $E$ and $E'$ have norm at most one, because $\lambda=1$ and the
capacity kernel is positive semidefinite. The resolvent identity gives
$\|E'-E\|\le\|\Delta\mathcal K\|<76\eta$.
For $q=k\odot d_3$, $h=f_3$, we have
$\|q\|<2$, $\|h\|=\sqrt{37}<7$, and
$\|\Delta h\|\le(3/5)\sqrt2\eta<\eta$. Hence

$$
|A'_3-A_3|
\le\|q\|(\|E'-E\|\|h\|+\|E'\|\|\Delta h\|)
<1066\eta=0.001066<\frac1{100}<\frac{36}{3377}.
$$

Thus $A'_3<0$ for **all** binary label vectors.

For the self kernel, the same estimates sharpen to
$\|\Delta\mathcal K_3\|\le26\eta+13\eta^2<27\eta$, and therefore
$|A'_{3,\rm self}-A_{3,\rm self}|<380\eta<1/10<336/2285$.
Thus the self derivative remains strictly positive for every label as well.

The prediction errors are nonzero: all initial logits are finite, so binary
sigmoid cross-entropy has nonzero residual on every input. The update is
nonzero, but it cannot exit the strict hold in one step.

## A genuine hold with no first-order outward excitation

Change only the readout to $v=(0,0,0)$, keeping the identical hidden parameters
and both-layer capacity definition. Exact rational inversion now gives

$$
A_3=-\frac{27}{640}<0,
\qquad A_{3,\rm self}=\frac{168}{895}>0.
$$

Nevertheless $J\nabla_\theta A_3=0$ exactly, where $J$ is the vector-output
Jacobian of the scalar logits on all five inputs:

- The hidden-parameter columns of $J$ vanish because all $v_j=0$.
- The capacity depends on $v_j$ through $v_j^2$, so
  $\partial A_3/\partial v_j=0$ at this point.
- The capacity is independent of the value of the output bias, so
  $\partial A_3/\partial c=0$.

Consequently $D A_3[-J^Tr]=0$ for **every** residual vector, and first-order
variance or outward-drift conditions for the coordinate $A_3$ do not follow
automatically from being a genuine hold. This is not a claim that all
higher-order changes vanish: a first SGD step can make $v$ nonzero and affect
$A_3$ at order $\eta^2$.

This variant is not asserted to satisfy a separately imposed assumption that
the previous random-label task has already been fit. It refutes automatic
first-order transversality on the unrestricted hold set, not on a potentially
smaller reachable, task-fitted state set.

It also lies outside any standing condition $\gamma_i>0$ used to normalize
the readout direction. It does not directly refute pointwise transversality
on that restricted set. However, taking all $v_j=\varepsilon>0$ and sending
$\varepsilon$ to zero preserves the strict hold for sufficiently small
$\varepsilon$, while the maximal first-order outward derivative tends to
zero by continuity. Thus positivity alone, without a quantitative lower
bound, cannot supply a uniform positive outward-drive bound.

## What this does and does not settle

- It disproves `for every held state, some label exits in one SGD step`, even
  with the actual two-layer kernel, trainable biases, positive step size,
  nonzero residuals, and an open fraction below one half.
- For any prescribed finite horizon $T$, continuity plus the finite label
  alphabet also gives a sufficiently small positive learning rate for which
  **all** label words of length at most $T$ stay held. This changes the learning
  rate with $T$; it is not an infinite-horizon invariant-set proof.
- We have not proved that this state has no exiting finite label word at the
  specified fixed learning rate. The general finite-word reachability question
  remains separate.
- This counterexample is in the hard-ReLU kernel. Since all relevant initial
  and one-step points have nonzero gate margins and strict hold margins, the
  same existence result extends to sufficiently small fixed softplus smoothing
  by continuity (with a fixed, finite collection of label words). It does not
  assert that the source's empirical convention $\tau=0.1\operatorname{sd}(z)$
  is already in that sufficiently small range.
