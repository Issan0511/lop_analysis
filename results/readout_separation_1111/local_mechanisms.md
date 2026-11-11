# What is separated algebraically

These are fixed-state identities, not conclusions from the new trajectories. The actual-label channel is primary.

## Readout, residual and transmission

For logits $z=Wh+b$ and cross-entropy residual $r=p-y$, the body gradient is $J_{\theta,h}^{T}W^{T}r$. A physical readout shrink changes both $p$ and $W^Tr$, while the evolving head changes the future residual. The crossed frozen assay fixes the whole $p$ vector when changing the backward $W$ map; a scalar entropy match would not suffice.

## ELU and mean preactivation

Write $u_i=A x_i+a$, $h_i=\mathrm{ELU}(u_i)$. On the fixed128-example gradient subset, the unreduced per-example preactivation gradient is $v_i=\mathrm{ELU}'(u_i)\odot W_B^T(p_i-y_i)$. For recipient full-set input mean $\mu_X$,

$$P_j=(\nabla_A L\,\mu_X+\nabla_aL)_j=\frac1{128}\sum_i v_{ij}(x_i^T\mu_X+1).$$

A gradient-descent step changes the recipient mean by $-\eta P_j$. This is not merely the mean of $v_{ij}$. Adam instead changes it by $(\Delta A\mu_X+\Delta a)_j$; its sign and magnitude must be measured from the actual virtual parameter displacement. ELU's derivative is small in the sufficiently negative region, so changing its derivative can alter both magnitude and direction across units. No universal sign of the position shift follows from the readout norm alone.

## Final affine LN

Let $u=x-\bar x\mathbf1$, $s=\sqrt{\overline{u^2}+\epsilon}$, $q=u/s$ and $h=\gamma\odot q+\beta$. If $g=\partial L/\partial h$ and $v=\gamma\odot g$, then

$$\frac{\partial L}{\partial x}=\frac{v-\bar v\mathbf1-q\,\overline{qv}}s.$$

The mean direction is removed; the centered radial direction has eigenvalue $\epsilon/s^3$ in the non-affine normalization Jacobian, so it is not exactly annihilated at positive epsilon. The radial-remove surrogate restores the subtracted radial term. The radial-plus-mean-remove surrogate gives $v/s$. The rest of the trunk Jacobian still acts upstream. Final gamma and beta gradients remain native at a shared state; trajectories need not retain identical affine gradients after diverging.

## The LN gauge is a same-function statement

For positive $c$, $W'=cW$, $\gamma'=\gamma/c$, $\beta'=\beta/c$, $b'=b$ preserves the evaluated active-class logits, $W\operatorname{diag}(\gamma)$, $W\beta+b$, and the pre-LN/trunk gradients. Coordinate gradients transform as $\nabla_{W'}L=\nabla_WL/c$ and $\nabla_{\gamma',\beta'}L=c\nabla_{\gamma,\beta}L$.

This identity does not preserve Adam's trajectory at unchanged learning rates. Even ideally transformed moments and zero Adam epsilon do not automatically produce the required coordinate-rescaled parameter displacements. With the same trunk moments the initial trunk step agrees, but head/affine updates can change subsequent residuals. Raw W norm is therefore insufficient to specify the effective readout map, while the gauge identity by itself does not show that physical readout shrink is harmless.

## What norm matching says

Matching the altered upstream-trunk gradient to the native L2 norm removes one scalar magnitude difference. It leaves its angle, coordinate distribution, optimizer history and resulting Adam displacement free to differ. The frozen diagnostic cannot by itself establish that this angle mediates all later accuracy delay. The dynamic unnormalized Jacobian policies and full-horizon U answer a separate intervention question.

## Limits of each separation

The forward surrogate is not a pure confidence or temperature intervention. It contracts logits toward a fixed class-specific source mean, so it can change the argmax as well as confidence. Its trajectory contrast measures this forward policy over the first 200 updates. Fixed-state crossed comparisons hold the entire example-wise probability vector constant when changing a backward map.

O preserves the class-centered Gram matrix but changes alignment with fixed labels and generally changes probabilities. An S-versus-O effect shows that changing the Gram matrix is not required for that intervention effect; it is not universal removal of every notion of alignment. P and O provide different controls and neither is an alignment-only mediation instrument.

Even the class-centered effective W diag(gamma) norm is not sufficient to determine harm: effective bias, input distribution, directions and the normalization factor 1/s matter. The gauge identity determines equivalent functions, not a sufficient scalar mechanism score.

The frozen virtual Adam update uses the fixed 128-example diagnostic subset. It is not the first actual training minibatch update. Norm matching is performed before Adam, so it does not match Adam displacement norms. Removing both final-LN subtraction terms leaves the factor gamma/s and the unchanged normalized forward pass; it is not an identity layer or a model trained without LN.
