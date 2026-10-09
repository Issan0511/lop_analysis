# Rank 1200 in the native CNN: use the first-Conv raw weights

2026-10-09. This gives a native 32x32 / Conv16 / Conv16 / FC100 / FC100 reference with 1,200 distinct positive images and a rank-1,200 binary contrast Jacobian. It corrects a limitation of the previous last-FC scaling: the hidden feature width of 100 cannot supply rank 1,200. No layer is widened. All raw weights and biases remain independent and trainable. The output-bias sign, expectation regularity, endpoint construction and moving-Adam transfer are separate components.

## 1. Why multiplying the old image count is insufficient

On the all-positive last two ReLU branches, write h4=W4 h3+b4, with h3 in R^100. For binary head contrast d, every last-FC-weight, last-FC-bias and head-weight sensitivity is in the image-row span of [h3,1]. Thus the old leading block J1, obtained by scaling only the last hidden FC and head, has row rank at most 101. Its sufficient full-row-rank condition cannot hold at N=1200. Even choosing 1,200 distinct inputs does not fix this limitation.

The alternative scaling below puts the 16*3*5*5=1200 first-Conv weight coordinates in the leading block. Their gradients need not lie in the span of the final hidden features.

## 2. A first-Conv gradient minor with 1200 independent input pixels

Use the original binary-head native architecture, two Conv5/pad2 layers with 16 channels, two MaxPool2 layers, and FC100/FC100. Choose a positive base RGB image with strict bottom-right pool winners; a positive hierarchical two-pool pattern with sufficiently small positive off-center Conv weights supplies explicit margins. Every hidden bias is positive.

Assign the 16 first-Conv channels to the 16 spatial centers

    (r_i,c_i) in {3,11,19,27} x {3,11,19,27}.

Their 5-by-5 input patches are disjoint and stay inside the image. Including the three color planes gives exactly 1,200 distinct pixels. These centers are final-pool-selected original positions: they have the form (4r+3,4c+3).

Start with a selector reference: second-Conv center weights form a positive diagonal channel map; the first 16 FC1 rows select the respective channel and pooled site; the first 16 FC2 rows select those FC1 rows. Unused units have positive biases. Take raw head rows (+d,-d), with d strictly positive, so the reference common head is zero. First-Conv center and off-center weights can all be positive, with off-center weights sufficiently small for the stated winners. At this reference the backpropagated contrast sensitivity to the first-Conv preactivation is supported, for channel i, only at its assigned center, with a positive coefficient q_i.

For a first-Conv raw weight j=(i,c,a,b), define its sensitivity feature

    G_j(x)=partial_(W1_j) z(x).

On this fixed branch,

    G_(i,c,a,b)(x)=q_i x_(c,r_i+a,c_i+b), a,b in {-2,...,2}.

Consequently the 1200-by-1200 derivative M of these G features with respect to the selected input pixels is diagonal with positive entries, after matching their order. It is invertible.

Now perturb the selector zeros to positive values. Use a hierarchy: positive off-diagonal center mixing in Conv2 first, and still smaller off-center coefficients, so the literal-self network also has strict spatial winners in every retained second-Conv channel. Positive FC perturbations activate every raw path while preserving all hidden positivity. The full and literal-self pool branches can be kept strict, and the minor remains nonzero by continuity. Thus at a fixed finite positive reference, all G features are linear in the input, M is invertible, and every raw contrast sensitivity at the base image is nonzero. This is an existence argument at an ordinary finite reference, not differentiation at the zero coefficients.

## 3. Level-simplex images using gradient features

Write x_P for the 1200 selected input coordinates, w for the corresponding first-Conv weight vector, g0=G(x0)>0, and

    beta = M^T w >0,  Q0=g0^T w>0.

Indeed the fixed-branch contribution depending on the image is w^T G(x); all other hidden-bias contributions are independent of the input. Hence beta is the logit-contrast input derivative on the selected pixels. Set

    T=(I-11^T/N) diag(1/beta_i), N=1200,
    x_n=x0+delta sum_i T_ni e_i.

A common positive scalar normalization of T is harmless. Since T beta=0 and 1^T T=0, the images lie on the same contrast level and have mean exactly x0. For sufficiently small nonzero delta they are positive, below one, distinct, and within all common full/self strict input branches. A common output-bias contrast cancels their logits.

Their first-Conv sensitivity rows are

    G_N=1 g0^T+delta T M^T.

If a^T G_N=0, multiplication by w gives Q0 sum_n a_n=0. Then a^T T M^T=0; invertibility of M gives a^T T=0. The left nullspace of T is span(1), so sum a_n=0 implies a=0. Therefore G_N has row rank 1200. This is a gradient-feature rank, not a claim that a 1200-by-100 hidden matrix has rank 1200.

The input mean and hence the true spatial/data first-filter mean direction u are exactly preserved. Since there are finitely many nonzero base sensitivities, shrinking delta additionally enforces any fixed positive relative sensitivity tolerance across all images. Rank remains 1200 for every nonzero delta in this construction; its smallest singular value is not uniform as delta tends to zero.

## 4. A different raw initialization scale

At this fixed unscaled reference, make the following changes for t>0:

- multiply the first-Conv weight matrix by t;
- multiply every hidden bias, in both Conv and both FC layers, by t;
- multiply both raw output-head rows (+d,-d) by t, preserving their zero reference common component;
- multiply its canceling output-bias contrast by t^2;
- leave all remaining hidden weight matrices at their reference values.

Then every hidden activation equals t times its unscaled value, all pool winners and positive gates are unchanged, and z is t^2 times its unscaled value. These are initial values only; the actual optimizer subsequently updates every raw coordinate freely.

The non-output-bias raw contrast Jacobian splits into disjoint columns

    R(t)=[t J1,t^2 J2].

J1 now includes first-Conv weights, every hidden bias, and head weights. J2 consists of the second-Conv and two hidden-FC weight matrices. The first-Conv submatrix of J1 is exactly G_N, so

    lambda1=lambda_min(J1 J1^T)>0.

All raw sensitivities remain nonzero for each fixed t>0, with the same relative inter-image closeness. The zero common head is used only to construct the reference and its exact full-logit NTK scale formulas; subsequent small independent raw perturbations may add a common head, by continuity of the strict margins. If Gamma bounds the entries of J1,J2 and C^2=||J1||_F^2+||J2||_F^2, then the general batch relative-response bound gives, once the output-bias common mode is nonnegative,

    Sym(JL)>=c0 t^2[lambda1-C_B t Gamma C^2/epsilon] I,
    c0=H/(N epsilon), C_B=1+sqrt(N/B).

Thus a finite positive t less than min(1, epsilon lambda1/(2 C_B Gamma C^2)) gives a strict normal margin. Neither full hidden-feature rank nor a small output-bias sensitivity is assumed.

## 5. Capacity derivatives have a different scale

Keep u as the actual additive raw first-filter mean direction; do not rescale u with the initialization. The full raw logit NTK, for the full or literal-self network separately, has

    K=K_bias+t^2 K1+t^4 K2,
    K_bias=11^T tensor I2.

The derivative powers differ from the previous last-FC construction:

    D_u K=t K1'+t^3 K2'.                                  (5.1)

To check this, the first-Conv and hidden-bias Jacobian blocks are order t but independent of W1 and b1 on the fixed branch, so their u derivatives vanish. The head-weight Jacobian is h=t h0, while D_u h is order one. The remaining hidden-weight Jacobian blocks are order t^2, with u derivative of order t. These are derivatives with respect to independent raw coordinates, not derivatives with respect to the scale t.

For a fixed ridge lambda>0, the original capacity slope divided by t therefore has the continuous positive-branch coefficient extension

    C_lambda/t = .5 tr[(lambda I+K_bias+t^2 K1+t^4 K2)^(-1)
                                (K1'+t^2 K2')].

At delta=0, K1' comes only from the free head-weight block and equals `11^T tensor [2<h0,D_u h0>I2]`. Positive paths give <h0,D_u h0>>0 for both the full and literal-self models. Thus

    lim_(t->0+) C_lambda(t,0)/t
       = 2N <h0,D_u h0>/(lambda+N)>0.

Joint continuity first permits a fixed sufficiently small nonzero delta, retaining rank 1200 and relative sensitivities, and then a sufficiently small positive t meeting the normal inequality and both capacity signs. No actual zero-scale ReLU is differentiated. No fixed lower bound on the unnormalized slopes as t tends to zero is claimed.

## 6. Components still needed for the stochastic theorem

For N1200 and B16, inverse-RMS regularity can use the whole-task event from `adam_b16_regularity.md`: with M=75 batches per epoch and a fresh permutation each epoch, the event that every batch in E epochs is balanced has probability

    p_N r^E,
    p_N=binom(1200,600)/2^1200,
    r=binom(16,8)^75/binom(1200,600).

The exact finite inequality r<(.999)^(75*8/2) suffices for an inverse-eighth moment for every finite E, including E400. This must be checked by the companion arithmetic certificate. The old first-batch-only probability bound is too weak at H30000 and must not be substituted.

A separate long-reuse output-bias proof is required at H30000; the earlier H4/H6 certificate is insufficient. Once that sign and the preceding inverse-moment inequality are supplied, the nonbias bound, C3 field, C2 nonsymmetric endpoint, positive ray LL^T u, full raw initial cone, and taskwise observable transfer apply without changing their dimension-independent arguments. The result still requires binary labels and sufficiently small decaying rates. Ten classes and constant-rate behavior need their own arguments; this construction does not resolve them.
