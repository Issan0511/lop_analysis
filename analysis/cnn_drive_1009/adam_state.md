# Statewise CE + Adam: exact decomposition, noncircular certificates, and a batch-16 counterexample

2026-10-09. These are one-switch theorems conditioned on the **entire trained CNN and optimizer history**. No independence of old labels and learned features is assumed. The fresh new labels are independent uniform labels conditional on that state. The main update is standard Adam with finite epsilon, no gradient clipping, no AMSGrad maximum, and no decoupled weight decay; extensions needing extra terms are stated explicitly.

## 1. Exact conditional CE moments

Fix a batch of B inputs, C classes, network parameters θ, logits f_n, probabilities p_n=softmax(f_n), and logit Jacobians J_n. Let fresh labels Y_n be independent and uniform over classes. For the batch-mean CE loss,

  g(Y)=(1/B)Σ_n J_nᵀ(p_n−e_{Y_n}),
  μ=E[g]=(1/B)Σ_n J_nᵀ(p_n−1/C),
  ξ=g−μ.

Writing J_{nc,j}=∂f_nc/∂θ_j, each coordinate has exactly

  σ_j²=Var(g_j)
      =(1/B²)Σ_n[(1/C)Σ_c J_{nc,j}²−((1/C)Σ_c J_{nc,j})²].

All moments are observable from the current probabilities and Jacobians. In particular, this expectation does not claim learned logits/Jacobians are independent of old labels.

The coordinate's complete possible gradient interval is bounded by

  l_j=(1/B)Σ_n[Σ_c p_nc J_nc,j−max_c J_nc,j],
  u_j^g=(1/B)Σ_n[Σ_c p_nc J_nc,j−min_c J_nc,j].

These bounds need no enumeration of C^B label vectors. Repeated dataset indices sharing the same assigned label require a corresponding covariance correction; the independence assumption applies to the label draws, not merely to positions in a batch.

## 2. Adam's exact scalar decomposition

Fix the previous Adam moments m_j^−∈R and v_j^−≥0, and the step number t. Put

  a_j=β₁m_j^−, b=1−β₁,
  c_j=β₂v_j^−, d=1−β₂,
  κ=sqrt(1−β₂^t)/(1−β₁^t)>0,
  ε̃=ε sqrt(1−β₂^t)>0,
  h_j(x)=1/(sqrt(c_j+d x²)+ε̃).

The actual Adam descent vector, excluding the learning rate, is

  A_j(Y)=κ(a_j+b g_j(Y))h_j(g_j(Y)).

For any fixed statistic direction u=∇m(θ),

  E[uᵀA]
   =κΣ_j u_j[(a_j+bμ_j) E h_j(g_j)
                    +b Cov(g_j,h_j(g_j))].

This separates three effects:

- the old first moment a_j;
- positive but coordinate-dependent mean scaling E h_j;
- the covariance introduced because the **current label gradient also changes the denominator**.

Even E[uᵀg]>0 and zero old momentum do not determine the sign of E[uᵀA]. Section 5 gives an exact CE/batch-16 reversal.

## 3. An observable interval/moment certificate

From [l_j,u_j^g], define G_max,j=max(|l_j|,|u_j^g|) and G_min,j=0 if the interval contains zero, otherwise its smaller endpoint magnitude. Since h_j decreases with |x|,

  h_min,j=h_j(G_max,j), h_max,j=h_j(G_min,j),
  w_j=(h_max,j+h_min,j)/2>0,
  ρ_j=(h_max,j−h_min,j)/(h_max,j+h_min,j)<1.

Thus |h_j(g_j)/w_j−1|≤ρ_j for every new-label vector. Let W=diag(w_j). Then

  E[uᵀA] ≥ κ{uᵀW(a+bμ)−E_den},
  E_den=Σ_j |u_j|ρ_jw_j[|a_j+bμ_j|+bσ_j].

Proof. Subtract (a_j+bμ_j)w_j from the coordinate expectation and write the difference as

  (a_j+bμ_j)E[h_j−w_j]
   +b E[(g_j−μ_j)(h_j−w_j)].

Bound the first term by |a_j+bμ_j|ρ_jw_j, and the second by bρ_jw_j E|g_j−μ_j|≤bρ_jw_jσ_j. Sum with |u_j|. □

Consequently, a strictly positive displayed lower bound proves the Adam direction. It depends on the current CNN, old optimizer state, gradient ranges and variances, not on the sign of the actual expected Adam update.

### Separating coordinate distortion from the raw self margin

Suppose a structural CE argument supplies S_self≤uᵀμ with S_self>0. Here S_self denotes a CE mean-descent margin; it is not automatically the literal isolated-unit NTK self term. When deriving it from a channel self shape, that additional identification must be stated. Put w_bar=(w_max+w_min)/2 and ε_W=(w_max−w_min)/(w_max+w_min), over coordinates with u_j≠0. Then

  uᵀWμ≥w_bar[S_self−ε_W||u||₂||μ||₂].

A sufficient condition that displays all three costs separately is

  b w_bar[S_self−ε_W||u||||μ||]
  +Σ_j u_jw_j a_j > E_den.

The momentum term is signed and measurable; it need not be discarded when it helps. This is a sufficient condition only. A failed certificate does not imply the Adam update is reversed.

### A structural CE mean-descent margin source

Center the logits z_n=f_n−mean_c(f_n) and the statistic-induced logit derivative A_n=J_nu likewise. Suppose

  A_n=τ_n z_n+e_n, τ_n≥0.

This is a geometric alignment assumption between the direction's effect and the existing logits, not an assumption about the CE/Adam update sign. For a rectified target whose feature and positive position derivative scale the same class-head vector, τ_n>0 arises directly. Finite other channels may be present; their effect on alignment is included in e_n.

Then

  uᵀμ≥(1/B)Σ_n[τ_n z_nᵀ(p_n−1/C)−||e_n||||p_n−1/C||].

For nonconstant centered z_n,

  z_nᵀ(softmax(z_n)−1/C)>0,

because it equals the integral from 0 to 1 of z_nᵀ[diag(p(t))−p(t)p(t)ᵀ]z_n. A simple explicit lower bound is

  z_nᵀ(p_n−1/C)≥exp(−range(z_n))||z_n||²/C.

The latter follows from the lower bound on each path probability and the weighted-variance characterization of the softmax Hessian. These quantities give an entirely structural S_self to insert above.

## 4. An exact positive theorem from symmetric class coefficients

This complementary theorem works even for fresh Adam with v^−=0, where the interval certificate may be too conservative.

Let real class coefficients a_c be a nonconstant symmetric multiset: each a occurs with its negative equally often. Their mean is zero. At the current CNN state suppose logits have the form

  f_nc=a_c t_n + k_n,
  t_n=qᵀH_n+β>0,
  q≥0,

where k_n is class independent. The output bias a_cβ is allowed and can be trained. Assume all relevant hidden directional derivatives satisfy

  d_nj=∂t_n/∂θ_j≥0.

A CNN with nonnegative hidden weights, nonnegative input/hidden features, ReLU and MaxPool provides this derivative sign at its differentiable states. Channels may overlap on every image and H need not have rank one. Only the current class-head parameterization has the displayed common scalar t_n.

For hidden coordinate j,

  g_j=(1/B)Σ_n d_nj[A(t_n)−a_{Y_n}],
  A(t)=Σ_c a_c exp(a_ct)/Σ_c exp(a_ct).

A(0)=0 and A′(t)=Var_{softmax(ta)}(a)>0, so A(t_n)>0. Thus μ_j>0 whenever at least one d_nj>0. The centered noise −B⁻¹Σ_n d_nj a_{Y_n} has an **exactly symmetric distribution**, including unequal d_nj and arbitrary finite B.

Assume m_j^−≥0 on the coordinates being considered; v_j^− may be any nonnegative value. The function

  F_j(x)=x/(sqrt(c_j+d x²)+ε̃)

is odd and strictly increasing for ε̃>0. For x≠0 its derivative is

  F_j′(x)=[c_j/sqrt(c_j+d x²)+ε̃]/[sqrt(c_j+d x²)+ε̃]²>0,

with the continuous derivative interpretation at c_j=x=0. If ξ_j is symmetric,

  E F_j(μ_j+ξ_j)>E F_j(ξ_j)=0.

Therefore E[A_j]>0 whenever μ_j>0. The old-momentum contribution κ a_j E h_j is nonnegative. If the mean statistic has u_j≥0 and some u_j>0 has μ_j>0, then

  E[uᵀA]>0.

This is an exact finite-C, finite-batch, finite-epsilon theorem, not a replacement of Adam by SGD or by its average denominator. It requires no old-label independence and no head-fit approximation.

The condition is one-step/statewise. A subsequent unconstrained Adam update can destroy the head structure or positivity of hidden weights. However, a strict expectation margin persists in an open neighborhood: with finite label support and ε>0, every term and its finite expectation is continuous on a fixed smooth CNN region. Thus the positive cases are not restricted to an algebraic equality surface of exactly symmetric/rank-one class heads.

### Concrete C=10, B=16 positive case with overlapping channels

Use 8 red and 8 green 2×2 images, each with intensities ρ=(1,.8,.6,.4) in its color plane. A shared 1×1 Conv has two channels with red/green coefficients (1,.1) and (.1,1), zero biases, followed by ReLU and MaxPool2. Its two pooled feature rows are (1,.1) and (.1,1); H has rank two and **both channels respond to every image**.

Take a_c=−4.5,−3.5,...,4.5, q=(.2,.1), β=.05, and output head/bias V_ic=q_i a_c, b_c=βa_c. Then t_red=.26 and t_green=.17. For the first Conv channel's mean preactivation, u=(.35,.35,0,1) in its RGB-weight/bias coordinates.

For fresh Adam, ε=1e−8, exact integer convolution of the label-sum distributions gives:

- red-weight mean CE gradient .1934531377, expected Adam direction .9448385521;
- green-weight mean CE gradient .1338649837, expected Adam direction .8000463664;
- bias mean CE gradient .3273181214, expected Adam direction .9799577138;
- expected actual mean direction .35*.9448385521+.35*.8000463664+.9799577138 = **1.5906674353>0**.

The expectation aggregates all 10^16 label vectors into finite sum distributions, not Monte Carlo samples. The coefficients/counts are exact integers; softmax and finite-epsilon values are evaluated numerically. The sign is established independently by the preceding analytic symmetry theorem.

## 5. Exact Adam reversal at C=10, B=16

Partition classes into 3 favored and 7 other classes. Let their total predicted favored probability be p=.301. Construct a scalar hidden direction whose batch CE derivative is

  g=p−K/16,
  K~Binomial(16,.3).

Its ordinary expected gradient is p−.3=.001>0. For fresh Adam, bias correction gives exactly A=g/(|g|+ε). With ε=1e−8, the complete 17-term binomial expectation is

  E[A]=−.10019160294460545<0.

In the epsilon→0 limit it is 2P(K≤4)−1=−.100191760350191; the finite-epsilon result above is already strictly reversed and was computed in exact Fraction arithmetic. Thus the failure is not a zero-epsilon artifact.

### Real shared CNN embedding

Use 16 identical-pixel but separately labeled 2×2 RGB images with red values ρ=(1,.8,.6,.4). There are two shared Conv1×1 channels. Both have red weight 1, bias h−1, and h>0. ReLU→MaxPool gives the same positive h in each channel, with unique top-left winner and preactivation winner gap .2.

Let v_c=.7 for the 3 favored classes and −.3 for the other 7; both channels have readout vector v. Set

  s=log[(p/(1−p))*(.7/.3)], h=s/2≈.002378691451.

The logits are 2h v=s v and their favored probability is exactly .301. The target channel's self scaling is aligned with these centered logits, and its ordinary expected CE direction is positive. Both channels have nonzero, overlapping contributions; no competitor was set to zero.

For the target's red weight and bias, the gradients both equal g above. Its actual mean-patch direction has weights (.7,1), so

  E[SGD mean direction]=1.7*.001=.0017>0,
  E[Adam mean direction]=1.7*(−.1001916029446)=−.1703257250058<0.

The mean statistic is affine in the first-layer weights, so this is an **actual one-step mean-update reversal**, not just a first-order expansion. The network can be regarded as a fixed trained state with reset optimizer moments. The counterexample disproves any claim based solely on the sign of the CE mean gradient; it does not assert typical trained Adam histories have zero moments.

### A finite nonempty history certificate on the same CNN

Keep exactly the same CNN and label distribution, but set the relevant previous moments to m^−=0,v^−=1, at step t=3, β₁=.9,β₂=.999. The interval certificate gives

  ρ=.0001222426304531,
  lower bound on E[A_j]=1.99232113264e−5>0,
  actual E[A_j]=2.02140884494e−5>0.

Thus the expected actual mean direction is 1.7 times these values and is positive. This is a finite parameter/history regime, not a limiting infinite-v argument.

As an optimizer recurrence, m^−=0,v^−=1 is attainable after two past scalar gradients G,−β₁G, where

  G=[(1−β₂)(β₂+β₁²)]^(−1/2)≈23.51152.

This verifies compatibility with Adam's moment equations; it is not a claim that these two past gradients occur on the particular repeated image batch without changing earlier network states.

## 6. From the directional theorem to the actual mean update

For a first-layer channel mean, m is affine in its own parameters and independent of all downstream parameters. Simultaneously updating every CNN layer and the head with Adam therefore gives exactly

  E[m(θ−ηA)−m(θ)]=−η E[uᵀA].

For a deeper mean statistic, upstream parameter updates also matter and m is nonlinear. If ||∇²m||₂≤L_m on every update segment,

  E[m(θ−ηA)−m(θ)]
   ≤−η E[uᵀA]+(η²L_m/2)E||A||².

A positive directional lower bound Γ proves actual sinking when η<2Γ/(L_m E||A||²). The moments/ranges above give finite bounds on A; a first-layer mean has L_m=0. Claiming an arbitrary-size deep-mean update from the directional sign alone would omit this remainder.

For AdamW add the known decoupled term η λ_w uᵀD_w θ to the descent-direction calculation, where D_w marks decayed coordinates. It can oppose sinking of a negative mean and must not be silently ignored. Gradient clipping or AMSGrad changes the denominator/function and requires its own certificate.

## 7. Verification artifacts and remaining scope

- `/tmp/cnn_adam_sign_check_1009.py`: exact 17-term Fraction expectation for the finite-epsilon reversal, interval/history certificate, and moment-history compatibility.
- `/tmp/cnn_adam_sign_autograd_1009.py`: literal Conv2d→ReLU→MaxPool→10-class logits→CE, all 17 class-count possibilities, actual shared-weight/bias gradients and Adam parameter updates. Maximum gradient-formula discrepancy 1.18e−16; actual first-layer mean reversal −.1703257250063, and positive-history case +3.4363950240e−5.
- `/tmp/cnn_adam_symmetric_check_1009.py`: exact integer label-sum convolution for the C=10,B=16 overlapping-feature positive construction; numerical softmax/Adam expectations.

All are statewise first-switch results. They do not establish that the ordinary RL-CIFAR optimizer state satisfies the certificates, or that any certificate is preserved over multiple tasks. They do demonstrate both a nonempty finite CE+Adam self-direction regime with interacting overlapping channels and a fully realized counterexample to unconditional transfer from SGD's expected sign.
