# Native finite-kernel deep CNN: literal full/self NTK direction on an open set

This note supplies a fixed-state theorem, not a long-time invariant-family theorem. It uses the ordinary ReLU and MaxPool architecture, two genuine shared spatial convolutions (for example 5×5 followed by 5×5), a free multiclass spatial head, and all ordinary biases as trainable raw parameters. It provides an open set of states with nonproportional channel features on which:

1. The original all-parameter NTK logdet capacity increases strictly in the first-Conv channel mean-increasing direction.
2. The literal channel-isolated network has the same strict capacity direction.
3. A fresh uniform-label CE SGD step decreases that first-Conv mean in expectation.

The conditions are checked from architecture, a constructive reference state, its spatial feature Gram matrix, and distances of Jacobians/logits from that state. Neither the sign of the desired derivative nor independence between the trained state and old labels is assumed. A trained state may depend arbitrarily on its past; only the new labels are fresh and uniform conditional on it.

## 1. Definitions and architecture

Fix N images and a CNN

    Conv(k1, shared weights, bias) -> ReLU -> MaxPool
    -> Conv(k2, shared weights, bias) -> ReLU -> MaxPool
    -> flatten -> full C-class linear head with output bias.

Each layer can have any finite number of channels, at least two if channel nonproportionality is desired. The spatial kernels can be 3×3, 5×5, or other finite sizes. Stride and padding are fixed. There is no batch normalization in this theorem. All ReLU preactivations are strictly positive and every pooling winner is unique at the reference state; these conditions will persist in a neighborhood. No parameters are tied during differentiation or actual SGD.

Let x_tilde[n,p] be the first-layer input patch at position p, with an appended 1 for the bias. Let

    mu = mean_(n,p) x_tilde[n,p],   kappa = ||mu|| > 0,
    r = mu/kappa.

For the selected first-Conv channel c, let m_c be its mean preactivation over these same images and spatial positions. Its Euclidean gradient u is mu in that channel's augmented filter coordinates and zero elsewhere. Since the input is fixed, m_c is affine and u is constant. This is the actual spatial mean direction, including bias, not a bias-only surrogate.

Write J = D_theta vec(f) for the Jacobian with respect to every raw trainable parameter. Define

    K = J J^T,
    Phi_lambda = (1/2) log det(I + K/lambda),  lambda > 0,
    J' = D_u J,
    K' = J' J^T + J J'^T.

The literal self network is obtained by deleting every first-Conv output channel except c and deleting the corresponding input-channel columns in the second Conv. The second Conv output channels, all retained biases, and the full original head are kept. The head is not refit. Its J_self includes all parameters remaining in that reduced architecture. Mean direction u_self uses the same mu.

## 2. Constructive native spatial reference

Choose positive unit channel vectors u1 and u2, positive numbers a1,a2, and a strictly positive spatial kernel k2(delta). Choose the first augmented filters and second weights as

    W1_tilde[j,:] = a1 u1[j] r^T,
    W2[i,j,delta] = a2 u2[i] u1[j] k2(delta),
    b2[i] = 0.

The first bias is the last coordinate of W1_tilde and is generally nonzero. The second bias is zero only in value, and its trainable Jacobian block is retained.

Let Z_n be the first scalar field after applying the first pool to r^T x_tilde. Let X_n be the scalar field after convolving Z_n with k2 and applying the second pool. Flatten X_n to S spatial coordinates; write X as the N×S matrix of its rows. Positivity and common positive channel factors give

    H1[n,j,p] = a1 u1[j] Z[n,p],
    H2[n,i,s] = a1 a2 u2[i] X[n,s].

All offsets of k2 participate; this is not a 1×1 convolution written in a larger zero-padded tensor.

Choose any C×S spatial class matrix B such that each column has class sum zero. The head is

    V[class,i,s] = B[class,s] u2[i],   bout = 0.

All these head and output-bias entries remain free trainable parameters. B need not have rank one and need not have a common class coefficient across spatial positions. At the reference,

    f0[n,:] = a1 a2 B X_n,     b = vec((B X_n)_n).

Require X to have full row rank and require at least one B X_n != 0. Full row rank is possible when N <= S and is directly checkable without examining any sign of K'. The numerical witness below uses N=2, S=9.

## 3. Exact full raw-parameter NTK derivative

Perturb only along the actual first-channel mean gradient u by a scalar q. Define

    v(q) = a1 u1 + q kappa e_c,
    k = kappa u1[c] > 0,
    t(q) = u1^T v(q) = a1 + q k.

Within the fixed gate/winner neighborhood,

    H1(q)[n,j,p] = v(q)[j] Z[n,p],
    H2(q)[n,i,s] = a2 t(q) u2[i] X[n,s],
    f(q) = a2 t(q) b,
    R0 := D_u vec(f0) = (k/a1) vec(f0).

Partition J by raw parameter blocks. There are q-independent PSD matrices G1, G2, K_b2, K_bout with

    K_full(q) = a2^2 G1 + ||v(q)||^2 G2
                + K_b2 + a2^2 t(q)^2 Q + K_bout,
    Q = (X X^T) tensor I_C.

The blocks correspond respectively to first-Conv weights and biases; second-Conv weights; second-Conv bias; head weights; output bias. In particular, no bias block is silently removed.

Why the two nonconstant blocks have this form:

- A second-Conv weight derivative for output channel i, input channel j, and offset delta is v(q)[j] u2[i] times a fixed class/spatial vector. Summing its outer product over all i,j,delta yields ||v(q)||^2 G2 because ||u2||=1. G2 is the sum over genuine spatial offsets and is PSD.
- A head weight derivative is a2 t(q) u2[i] X[n,s] in its output class. Summing over all head coordinates yields a2^2 t(q)^2 Q.
- First-Conv Jacobians depend on the second weights, head, and fixed routing, not on the first filters' values. Bias2 and output-bias Jacobians also do not depend on q.

Consequently,

    K'_full(0) = 2 a1 k [G2 + a2^2 Q].                 (3.1)

Since Q is positive definite under full row rank of X,

    K'_full(0) >= ell_full I,
    ell_full = 2 a1 k a2^2 lambda_min(X X^T) > 0.      (3.2)

This identity differentiates all untied raw parameter blocks. Differentiating a common scalar amplitude as if it were the only trainable parameter would produce a different NTK and is not used here.

## 4. Literal channel-isolated self

Delete first-layer channels j != c as specified in Section 1. Positivity and the spatial routing are preserved, since all retained feature maps at the reference differ only by positive scalar factors. In the isolated model,

    v_c(q) = a1 u1[c] + q kappa,
    H2_self(q)[n,i,s] = a2 u1[c] v_c(q) u2[i] X[n,s].

The second-weight Gram block is v_c(q)^2 G2 and the head-weight Gram block is a2^2 u1[c]^2 v_c(q)^2 Q. All other blocks are q-independent. Hence

    K'_self(0) = 2 a1 k [G2 + a2^2 u1[c]^2 Q],         (4.1)
    K'_self(0) >= ell_self I,
    ell_self = u1[c]^2 ell_full > 0.                  (4.2)

This is the original capacity self from the reduced network, not a name assigned to the CE drift. For either network and every finite lambda > 0,

    D_u Phi_lambda
      = (1/2) tr[(lambda I + K)^(-1) K'] > 0.         (4.3)

Thus negative capacity gradient specifies a decreasing-mean direction in both the full and isolated networks. Finite inverse dependence causes no sign problem here because (3.1) and (4.1) are positive definite.

## 5. A finite certificate for an open set of fully untied CNNs

The channel-aligned reference is a device for proving nonemptiness. The final sufficient conditions do not require the actual network to remain channel-aligned.

For either the full or literal-self architecture, compare its actual Jacobians J,J' with the matching reference J0,J0'. Let

    d = ||J-J0||_F,    e = ||J'-J0'||_F,
    E_K = 2[e ||J0||_F + d ||J0'||_F + d e].

Expanding K'-K0' and using ||AB^T||_* <= ||A||_F ||B||_F gives

    ||K'-K0'||_op <= ||K'-K0'||_* <= E_K.             (5.1)

The sufficient condition is

    E_K,full < ell_full,
    E_K,self < ell_self.                              (5.2)

It is a norm-distance condition to a known reference, not the requested sign rewritten as an assumption. It is directly computable using first and directional-second derivatives; in a fixed routing region these are derivatives of finite polynomials in the parameters. It implies K'_full and K'_self are positive definite, simultaneously for every lambda > 0.

J and J' vary continuously within the strictly active, unique-winner region. At the reference d=e=0 and both ell are positive. Therefore (5.2) holds on a nonempty open neighborhood in the entire raw parameter space, including independent biases, every spatial kernel entry, independent channel entries, and a free class/spatial head. The full and self architectures both need their routing neighborhoods checked; the same small full-parameter neighborhood works by continuity and the linear deletion map.

Generic finite perturbations in that neighborhood make channel feature maps nonproportional. Indeed, the explicit witness below has feature-channel matrix rank two in both convolutional layers and strict remaining certification margins. Nonzero channel-rank minors persist on an open neighborhood of that witness. No other channel is set to zero and all channels overlap on all images.

Optional weaker version: when X lacks full row rank, (3.1)/(4.1) remain nonzero PSD if X != 0, and the reference capacity slopes are still positive. For a fixed lambda, the entirely explicit bound

    D_u Phi_lambda >= tr(K0')/[2(lambda+||K||_op)]
                         - ||K'-K0'||_*/(2 lambda)

can certify a neighborhood. Unlike (5.2), this version depends on lambda. The full-row-rank version gives a simpler uniform-in-lambda result.

## 6. Actual conditional fresh-label CE update

Hold the actual trained state fixed and draw new labels independently and uniformly from {1,...,C}. No statement about the distribution of old labels conditional on this state is needed. With mean CE over the N images,

    D_CE = E[D_u L_CE | state]
         = (1/N) <R, p(f)-1/C>,   R = J u.

At the reference R0=(k/a1)f0 and each f0[n] is class-centered. Thus

    D0 = (k/(a1 N)) sum_n <f0[n], softmax(f0[n])-1/C> > 0

as long as one class contrast is nonzero. A structural lower bound is

    D0 >= (k/(a1 N C)) sum_n exp(-range(f0[n])) ||f0[n]||^2.  (6.1)

To prove it, write softmax(f)-softmax(0) as the integral of its covariance Hessian along tf. On class-centered vectors the covariance quadratic form is at least min_c p_c(tf) times squared Euclidean norm, and min_c p_c(tf) >= exp(-range(f))/C.

For a perturbed actual state, using the Euclidean softmax Lipschitz constant 1/2 gives

    |D_CE-D0| <= E_CE,
    E_CE = ||R-R0||_F sqrt((1-1/C)/N)
           + ||R0||_F ||f-f0||_F/(2N).               (6.2)

One can upper-bound ||R-R0||_F further by d ||u||. Either the known reference expression for D0 or its positive structural lower bound (6.1) can be used. The sufficient condition E_CE < D0 is open and holds at the reference. Together with (5.2), it therefore defines a nonempty open set where actual conditional CE descent matches both original full and literal-self capacity directions.

For ordinary SGD updating every parameter simultaneously, the first-Conv preactivation mean is affine in its first-layer weights and bias. Therefore, for any learning rate eta,

    E[m_c(theta-eta grad L)-m_c(theta) | theta]
       = -eta D_CE < 0.                              (6.3)

There is no Taylor remainder in this particular mean observable. The size of eta still matters if one wants the next state to remain in the certified neighborhood. Adam, momentum, label reuse inside a task, arbitrary new-label distributions, and indefinite trajectory preservation are separate claims and are not established by this fixed-state result.

## 7. Concrete finite 5×5 witness and independent raw autograd checks

The reproducible scratch script is `verify_full_ntk_positive.py`; its output is `../../results/cnn_drive_1009/full_ntk_positive.json`. It uses float64, two distinct RGB images of size 12×12, 10 classes, two channels in each convolution, and both spatial kernels of size 5×5 with padding 2. There are 444 independently differentiated trainable raw parameters, including all biases. MaxPool2 is applied after both ReLUs, leaving S=9 spatial positions.

The first images have different smooth spatial patterns plus a small fixed random perturbation. The first spatial filter r is defined from the actual augmented patch mean. The second scalar 5×5 kernel has 25 strictly positive, nonidentical entries. The head B is a full random class-centered 10×9 matrix. No kernel is a padded 1×1 filter.

Reference facts:

- kappa=3.5640088119, k=2.1384052871, a1=1.1, a2=.9, u1=(.6,.8), u2=(.8,.6).
- Eigenvalues of X X^T: .8334957736 and 192.5901706842.
- Minimum preactivations: .7328268912 and .6132640350.
- Minimum MaxPool winner margins: .0157876980 and .0143168551.
- The largest error between the raw autograd K' and (3.1)/(4.1) is 1.71e-13 for either architecture.
- The error in R0=(k/a1)f0 is below 8.89e-16.

Independently perturb all 444 raw coordinates by fixed Gaussian draws with standard deviation 5e-6, including all biases and all head entries. The realized parameter displacement has norm .0001114799. At this non-aligned actual state:

| Quantity | Full CNN | Literal channel-isolated self |
|---|---:|---:|
| Structural reference eigenvalue lower bound ell | 3.17615085 | 1.14341431 |
| Perturbation bound E_K from (5.1) | 1.34000787 | .60541848 |
| Certified actual K' eigenvalue lower bound | 1.83614299 | .53799583 |
| Direct actual minimum eigenvalue, diagnostic only | 3.17596360 | 1.14333267 |
| Capacity derivative at lambda=1, diagnostic only | 29.28446472 | 58.26153458 |

The simultaneous CE certificate is D0=.9722649361, E_CE=.0007684694, so D_CE >= .9714964668 > 0. The direct CE projection is .9722107392. The smallest singular values of the two-channel feature matrices (flattening images and space) are 5.74e-5 in layer 1 and 2.096e-4 in layer 2, so neither layer has proportional channel maps. The reference routing remains unchanged.

The numerical computations verify the algebra and exhibit finite slack. The mathematical sufficient conditions are (5.2), the CE bound, and the strict gate/winner margins, and their nonempty open set follows analytically from Sections 2–6; it is not an inference from the numerical signs alone.

## 8. Scope

This settles a structural fixed-state extension from a symmetric 1×1 family to an open set of native multi-channel finite-kernel deep CNNs with literal capacity-self and actual CE direction. It includes positive-active ReLU/MaxPool and all biases. It does not prove that an arbitrary learned CNN lands in this neighborhood or remains there long-term; it does not prove negative bulk preactivation or gate death. Its reference hidden activations are positive. It also does not equate CE with a logdet objective: their directions are proved separately and then compared.

## 9. Extension to arbitrary finite homogeneous Conv/FC suffixes

The same argument includes additional jointly trainable hidden fully connected/ReLU layers and a final output FC, without adding an activation or treating FC parameters as frozen. In particular, it covers a two-Conv architecture followed by any specified finite number of hidden FC/ReLU layers and a free output FC. The following stronger reference construction also permits nonproportional second-Conv channels already at the reference.

Keep the augmented first filters as in Section 2,

    W1_tilde[j,:] = a1 u1[j] r^T,

but replace the second filters by

    W2[i,j,delta] = u1[j] C[i,delta],

where the finite spatial kernels C[i,delta] are otherwise unrestricted subject to strict gate/winner conditions. They need not be proportional over output channels i. Set every bias after the first Conv to zero in value, while retaining all of these biases as independent trainable coordinates in J. Attach any finite composition of ordinary linear/Conv, ReLU, and MaxPool layers with fixed weights at the reference, followed by a free linear multiclass output head. All weights are independently trainable. Require strict nonzero ReLU preactivations and unique pool winners wherever derivatives are needed. Taking positive C and positive hidden weights is one simple way to obtain positive hidden preactivations.

The second preactivation field under the mean-direction perturbation is

    z2(q) = t(q) U,     t(q)=a1+q k,
    U[n,i,p] = sum_delta C[i,delta] Z[n,p+delta].

Because all subsequent biases have zero value, the suffix is positively homogeneous in its input. In a neighborhood with t(q)>0, every downstream hidden feature and the output logits scale by t(q), and the same gates/winners are selected. If F is the N×D last-hidden feature matrix at input scale t=1, then the actual last-hidden features at the reference are a1 F. The logits satisfy

    f(q) = t(q) f_star,
    R0 = (k/a1) f0.

### 9.1 All raw Jacobian blocks, including trainable biases

On the fixed branch the suffix is a linear map of its input. Its input Jacobian is independent of t. Consequently:

- First-Conv weight and first-bias Jacobians are independent of q.
- A second-Conv weight Jacobian with input channel j is v(q)[j] times a fixed vector. Summing raw outer products over j gives `||v(q)||^2 G_2`, for a q-independent PSD G_2. No proportionality among the C output channels is required.
- Every subsequent weight Jacobian is t(q) times a fixed vector. This follows because its incoming hidden activation scales with t, while the downstream Jacobian has fixed gates and weights. The sum over every such raw Conv/FC/output-head weight coordinate is `t(q)^2 G_suf`, with G_suf PSD.
- Every bias Jacobian after the first Conv is independent of q. A bias perturbation is injected as a constant basis vector at its layer; only the fixed downstream linear Jacobian acts on it. Zero bias values do not mean that their nonzero Jacobian blocks are removed.

Thus for q near zero,

    K_full(q) = K_first_const + ||v(q)||^2 G_2
                + t(q)^2 G_suf + K_bias_const,
    K'_full(0) = 2 a1 k (G_2 + G_suf).                (9.1)

Delete all first-Conv channels except c and their corresponding second-Conv columns, without refitting or otherwise changing the suffix. Its second preactivation is now

    z2_self(q) = u1[c] v_c(q) U.

This is a positive rescaling of the full reference branch, so all strict gate/winner choices agree. Its second-weight Gram is `v_c(q)^2 G_2`, while its suffix-weight Gram is `u1[c]^2 v_c(q)^2 G_suf`. Its first/bias blocks are again q-independent. Therefore

    K'_self(0) = 2 a1 k (G_2 + u1[c]^2 G_suf).        (9.2)

These are derivatives of the entire untied raw parameter NTK, not a reduced scalar parameterization.

### 9.2 Full row rank and nonempty open conditions

The free output-head weight block contributes

    (F F^T) tensor I_C

to G_suf. If F has full row rank, then

    K'_full(0) >= ell_full_new I,
    ell_full_new = 2 a1 k lambda_min(F F^T) > 0,
    K'_self(0) >= u1[c]^2 ell_full_new I.              (9.3)

The same Jacobian perturbation certificate (5.1)–(5.2) therefore gives a nonempty open set in the entire enlarged raw parameter space. In this open set all previously zero-value biases may be nonzero, every Conv/FC weight may vary independently, and first- and later-layer channel feature maps need not be proportional. The CE ray identity and its finite perturbation bound in Section 6 apply without change. One may choose a class-centered output weight matrix and nonzero class contrast to obtain the same explicit reference CE lower bound. Thus adding hidden FC layers does not create a gap between the fixed-state architecture and a Conv/FC classifier of this form.

Full row rank is not lost merely by adding positive-active hidden FC layers of width at least N. An explicit existence argument is useful. Let A be an N×D strictly positive, full-row-rank hidden-feature matrix, and choose N columns I such that A[:,I] is invertible. For the first N output coordinates, set the next FC transpose matrix to the column-selector matrix plus `epsilon * 1_D 1_N^T`. For any sufficiently small positive epsilon, every FC weight is positive and

    A W^T = A[:,I] + epsilon (A 1_D) 1_N^T

remains invertible by continuity of determinant. Its entries are positive, so ReLU acts as the identity with strict margin. Extra output coordinates may use arbitrary positive weight columns. Repeat for each additional hidden FC layer. This constructs a finite, nonempty positive-active FC suffix with full-row-rank last-hidden features; no limiting network or zero-channel construction is used. The native 5×5 witness in Section 7 supplies the starting positive full-row-rank pooled features. Existing strict margins and derivative bounds then supply the open neighborhood.

The extension remains a fixed-state/open-neighborhood theorem. It does not establish that unrestricted Adam or a label-reusing trajectory stays in that neighborhood. Additional FC biases are free in the final open-set result, but their unrestricted nonzero values are handled by the perturbation certificate, not by claiming the exact reference homogeneity identities hold at every such state.


## 10. Explicit nonemptiness for the 32×32 / 16→16 / FC100→100 architecture

The full-row-rank reference need not rely only on the 12×12 numerical witness. Use two 32×32 RGB images: one constant positive image, and the same image with a positive localized bump at pixel (8,8) in one colour channel. For example, background 1/4 and bump 1/8 keep every input in (0,1). The actual mean augmented patch gives a strictly positive first filter direction r. Use 16 positive first-channel coefficients, 16 positive second-channel coefficients, and a strictly positive genuine 5×5 second spatial kernel.

For Conv5(pad2)→Pool2→Conv5(pad2)→Pool2, final coordinate q has one-dimensional receptive support [4q−6,4q+9]. Coordinates q=2 and q=5 have supports [2,17] and [14,29], both entirely inside the image. Take the same row coordinate 2 for both cells. The constant image has equal positive values A at these two final scalar feature coordinates. In the bumped image, the left cell increases strictly by some Δ>0 and the right cell is unchanged. Positive kernels and max pooling propagate a positive change along at least one path in the left receptive field; the right receptive field excludes the bump. With columns ordered (right,left), the corresponding two-image feature minor is AΔ>0.

The initially constant fields have pooling ties. Remove them by perturbing the image pair as I₁→I₁+εS and I₂→I₂−εS. This preserves the image sum, hence the actual mean augmented patch and r exactly. Sufficiently small ε preserves positive inputs and the nonzero feature minor. A generic S avoids the finitely many proper piecewise-affine tie sets: distinct pooling candidates with fixed positive kernels have different spatial supports and do not define identical functions. Thus all required pool winners can be made strict while retaining rank two.

Apply the selector-plus-positive-ε FC construction in §9 to two hidden layers of width100. It preserves strictly positive activations and the two-image row rank. A class-centered free10-class output head with nonzero contrast completes a reference for the exact structural sizes RGB32×32→Conv5(16)→Pool2→Conv5(16)→Pool2→FC100→ReLU→FC100→ReLU→FC10, with every bias trainable. This is an analytic construction of inputs and states satisfying the certificate, not a measurement of the trained CIFAR run.

Independent mathematical review, including the all-raw bias/suffix formulas and this full-size existence argument: [full_ntk_positive_review.md](full_ntk_positive_review.md). No additional numerical training was used for this extension.
