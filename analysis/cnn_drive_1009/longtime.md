# CNN drive: stopped martingale bridge and an invariant repeated-fit toy family

2026-10-09, independent derivation. This document separates a conditional expectation at one label switch, a time-average theorem, and a concrete repeated-CNN trajectory. It does not prove the actual end-to-end RL-CIFAR Adam trajectory.

## 1. Filtration and the label-independence issue

For the general F1 model, let F_t contain the hidden parameters theta_t and all randomness from previous episodes. Conditional on F_t, draw fresh old labels Y_t with mean zero and covariance E[Y_t Y_t^T|F_t]=nu I. Fit the linear head exactly with ridge lambda>0 while holding the CNN features fixed. Draw independent mean-zero new labels Y'_t, evaluate the new-loss hidden gradient, and update the hidden parameters by SGD. The next episode draws its old labels afresh, independently of its updated features.

This is a precise repeated version of F1. It is not automatically the ordinary schedule in which Y'_t is reused as the old label Y_{t+1}: theta_{t+1} has then already used Y'_t, so conditioning on theta_{t+1} may destroy the old-label covariance/independence required by F1. This gap cannot be patched by saying labels were initially independent. Section 5 below constructs a special CNN in which the conditional sign survives even under ordinary label reuse, by an exact squared-label argument.

Inputs and the upstream representation remain fixed. Let mu_aug be the mean augmented input patch, u=mu_aug, and m_t=mu_aug^T theta_{c,t}. Let Z_t be the directional hidden gradient along u of the new loss. Full SGD on theta_c then gives the exact identity

  m_{t+1}-m_t = -eta_t Z_t.

There is no projection of the update onto u in this identity; it is the dot product of the full target-filter update with the fixed mean patch.

## 2. Geometry certificate and stopping time

At state theta_t define the complete pooled feature matrix H_t, target block H_{c,t}, its directional derivative R_{c,t}, K_t=H_t H_t^T and Q_t=K_t(K_t+lambda I)^(-2). Let Pi_t project onto range(K_t). Put

  c_t = <R_{c,t},H_{c,t}>_F /
        (||Pi_t R_{c,t}||_F ||H_{c,t}||_F),
  alpha_t = (q_max,t+q_min,t)/2,
  epsilon_t = (q_max,t-q_min,t)/(q_max,t+q_min,t),

where q_min,t and q_max,t are the smallest and largest positive eigenvalues of Q_t. The geometry certificate is

  H_{c,t} != 0,
  <R_{c,t},H_{c,t}>_F>0,
  c_t-epsilon_t >= delta_t>0.

Winner/activation margins and any input-geometry conditions used to establish the inner-product sign are part of the certificate. For ReLU and nonnegative augmented patch slopes the inner product is positive whenever the target has a positive response. Uniform bias is not substituted for the mean-patch direction.

The F3 support bound gives, conditionally on F_t,

  mu_t := E[Z_t|F_t]
    >= (nu/N) alpha_t ||Pi_t R_{c,t}||_F ||H_{c,t}||_F delta_t.

Choose a deterministic nonnegative lower schedule ell_t, and require the **observable geometric** inequality

  (nu/N) alpha_t ||Pi_t R_{c,t}||_F ||H_{c,t}||_F delta_t >= ell_t.

Let tau be the first state failing one of these geometric conditions. All these quantities are known before drawing the episode labels, so 1{t<tau} is predictable. This is not the assumption “the actual drive keeps its sign”: it is a kernel-spectrum/feature-angle/amplitude certificate, independently checkable at each visited state.

The F2 diagonal metric certificate can be substituted verbatim: replace the geometric lower bound by (nu/N)(S_D,t-epsilon_D,t N_D,t). Neither theorem by itself proves that tau is infinite. That requires a separate invariant-region argument, such as section 5.

## 3. Stopped martingale theorem

Assume eta_t>0 is deterministic, E[Z_t^2 1{t<tau}] is finite, and

  E[1{t<tau}(Z_t-mu_t)^2] <= sigma_t^2.

Define

  xi_t = 1{t<tau}(Z_t-mu_t),
  M_T = sum_{t<T} eta_t xi_t,
  A_T = sum_{t<T} eta_t,
  L_T = sum_{t<T} eta_t ell_t.

Then M_T is a square-integrable martingale and

  sum_{t<T} eta_t 1{t<tau} Z_t
    >= sum_{t<T} eta_t 1{t<tau} ell_t + M_T.

Equivalently,

  m_{T wedge tau}+sum_{t<T} eta_t 1{t<tau} ell_t

is a supermartingale. Thus the mean sink is an actual stochastic-process statement, not a permutation-average analogy.

### 3a. Weighted time averages

If A_T -> infinity and

  sum_t eta_t^2 sigma_t^2 / A_{t+1}^2 < infinity,

then M_T/A_T ->0 almost surely. Therefore, on {tau=infinity},

  liminf_T [sum_{t<T} eta_t Z_t / A_T]
    >= liminf_T [L_T/A_T].

In particular, if ell_t>=ell>0, the long-run weighted average drive is at least ell>0 almost surely on the nonexit event. If ell_t merely tends to zero, this only guarantees a nonnegative limiting lower average when the right side is zero; it does not establish a strictly positive limiting constant.

Proof of the noise limit: the martingale series sum_t eta_t xi_t/A_{t+1} has a finite total second moment by the displayed summability condition, and therefore converges almost surely. Summation by parts (Kronecker's lemma) yields M_T/A_T->0. Uniformly bounded conditional variance with constant step size satisfies the condition, since the summands are O(1/t^2). Usual eta_t=t^(-a), 0<=a<=1, also works under a uniform variance bound.

Merely requiring each noise variable to have a finite variance is insufficient. A growth/summability bound such as the one above is needed.

### 3b. Resolving a decaying signal

If L_T -> infinity and the stronger signal-relative condition (with the sum begun after L_{t+1} first becomes positive)

  sum_t eta_t^2 sigma_t^2 / L_{t+1}^2 < infinity

holds, then M_T/L_T->0 almost surely. On {tau=infinity},

  sum_{t<T} eta_t Z_t >= (1-o(1)) L_T >0

eventually. This certifies the cumulative direction even when the per-step drive decays and L_T/A_T->0.

If L_T stays finite, no such relative-noise conclusion follows from this argument. In particular, one cannot infer that realized cumulative drive divided by its predictable mean converges to one. The concrete toy below has a finite total sink and requires its additional exact dynamics to establish the pathwise final direction.

### 3c. Finite-time guarantee

For every x>0,

  P(tau>=T and sum_{t<T} eta_t Z_t < L_T-x)
    <= [sum_{t<T} eta_t^2 sigma_t^2]/x^2.

This follows from the stopped martingale second moment and Chebyshev's inequality. With bounded increments one can strengthen it to an exponential martingale concentration bound, but boundedness is not needed for the displayed result.

Bounded one-hot/Rademacher labels, lambda>0, and bounded feature/derivative norms on the stopped region give a directly checkable uniform gradient-variance bound. F1's second-label-moment assumption alone does not automatically give a finite gradient variance for unbounded labels; a fourth moment may be needed because the fitted-label contribution is quadratic.

## 4. Why a permanent positive drift margin may be impossible

If geometry confines m_t to a lower-bounded region while A_T->infinity, a uniform ell>0 plus the time-average theorem is incompatible with tau=infinity: it would force m_t to -infinity. Thus a proof cannot simultaneously assume a permanently bounded feature region, an indefinitely positive absolute drift margin, and an infinite learning horizon without examining their compatibility.

A realistic dying-ReLU family instead has a uniformly positive **angle** margin but a drift amplitude that tends to zero. The long-run average can be zero even though every finite-step conditional drift is in the self direction and each sufficiently long net displacement is downward. Section 5 makes this distinction exact.

## 5. Repeated shared-CNN family with a provable invariant geometry

### 5.1 Actual CNN and images

Use C=3 input colors and C output conv channels. There are M identical images of each color, so N=CM. Take M=3 and L=10 output classes in the concrete example; more generally 1<=M<L. Every image in group i is a 2x2 RGB image with patches

  x_{is}=rho_s e_i,
  rho=(1,.8,.6,.4).

A shared 1x1 Conv, ReLU and a 2x2 MaxPool produce one feature per output channel. An L-dimensional linear readout produces the class output; no output intercept is added in this toy. A trainable common output intercept would couple the groups and needs a different proof.

Initial filters and biases are

  w_ci=A if i=c, and delta otherwise;
  b_c=-B;
  A>B>delta>0.

Each channel responds positively only to its own color group, with h_0=A-B>0, and its winner is the unique rho=1 position. Other channels are active on their own groups at finite strength. For target c, its active fraction is 1/C; taking M=3 gives three active images among nine, rather than only one active example.

The mean augmented patch direction is

  u=((rho_bar/C) 1_C,1),
  k=u^T(e_c,1)=1+rho_bar/C>0.

Thus the target mean-direction derivative is R_c=k 1_{group c}, while H_c=h_c 1_{group c}. This is the actual mean-patch direction, including its nonzero weight component.

### 5.2 Head fit and ordinary reused-label schedule

At each task, fit the L-dimensional linear head exactly to current old centered one-hot labels Y=e_l-1/L, with class l drawn uniformly. Let S_c in R^L be the sum of the M old labels in group c. Then the target readout vector is

  v_c = h_c S_c/(M h_c^2+lambda).

Draw independent new centered one-hot labels and let S'_c be their group sum. Update all conv filters and biases by full-batch SGD on the new squared loss, with the just-fitted readout held fixed. Refit the head to these new labels at the next step, so labels are reused in the ordinary way.

Because only the own group is active, the target conv gradient is proportional to a_c=(e_c,1), and the exact activation recurrence is

  h_c^+ = h_c - (2 eta_t/N) v_c^T(M h_c v_c-S'_c)
        = h_c [1-(2 eta_t/N) M h_c^2 ||S_c||^2/(M h_c^2+lambda)^2
                  +(2 eta_t/N) <S_c,S'_c>/(M h_c^2+lambda)].

Conditional on the entire current state and old labels, S'_c has mean zero. The directional gradient consequently satisfies

  E[Z_{c,t}|current state, old labels]
    = (k M/N) h_c^3 ||S_c||^2/(M h_c^2+lambda)^2 >0.

If n_l is the number of old labels of class l in this group, then

  ||S_c||^2=sum_l n_l^2-M^2/L >= M-M^2/L >0.

For M=3,L=10 this lower bound is 2.1, and ||S_c||^2<=M^2. This is a stronger special-case conditional statement than F1: it survives old-label/feature dependence, because the remaining old-label coefficient is an explicitly nonzero squared norm. New-label averaging does not silently average the current features over old labels. The same proof also covers scalar Rademacher labels with odd M, for which S_c^2>=1.

### 5.3 Positivity and the geometric exit threshold

Let

  H_exit=A+B-2 delta.

While 0<h_c<H_exit for every channel, the same **pooled feature supports across image groups** and MaxPool winners remain valid. Individual nonwinning positions can cross their own ReLU thresholds; this does not change the winner or the pooled output formula. Indeed the update changes only w_cc and b_c, and

  w_cc=(A+B+h_c)/2>0,
  b_c=-(A+B-h_c)/2,
  w_ci=delta for i!=c.

Own winners retain the positive gap (1-rho_s)w_cc. Other-group maximal preactivations are delta+b_c<0 precisely when h_c<H_exit.

If

  0<2 eta_t M^2/N<lambda,

then h_c^+ remains strictly positive for every possible old/new label realization. To see this, ||S_c||,||S'_c||<=M and |<S_c,S'_c>|<=M^2. With x=Mh_c^2/lambda,

  M^3 h_c^2/(Mh_c^2+lambda)^2 + M^2/(Mh_c^2+lambda)
    = (M^2/lambda)(2x+1)/(x+1)^2 <= M^2/lambda.

Thus the multiplicative factor in the update is at least 1-2 eta_t M^2/(N lambda)>0. No positive pooled feature h_c reaches zero in finite mathematical time.

### 5.4 Geometry certification on the whole surviving trajectory

The feature columns have disjoint group supports. The full kernel is block diagonal, with one rank-one block h_c^2 11^T per group. It has rank C, even when M>1 and N>C. Hence the full-space F2 would fail, but the F3 support theorem applies.

For each target, H_c and R_c are exactly positively proportional and belong to range(K). Therefore c_Pi=1 at every finite surviving time. The full Q has positive eigenvalues Mh_c^2/(Mh_c^2+lambda)^2 on its C-dimensional support. Thus epsilon<1=c_Pi for every finite state. The geometry certificate is maintained because of the proved support/winner invariant, not because the desired sign was postulated.

Equivalently, the target feature is itself an eigenvector of the full Q with a positive eigenvalue. Other channels may have any finite positive h values; they need not be weak. This is a restrictive support-separation family, but is nonempty and has strict activation/winner margins. Small perturbations of initial filters preserving these strict inequalities also give the same family of activation supports; exact equality of channel strengths is unnecessary.

### 5.5 Nonexit probability and limiting sink

Stop at the first task tau where some h_c>=H_exit. Before this exit, each h_c is a nonnegative supermartingale under the exact conditional drift above. The stopped sum S_t=sum_c h_{c,t wedge tau} is therefore a nonnegative supermartingale. At an exit S_tau>=H_exit. The maximal inequality gives

  P(tau<infinity) <= C h_0/H_exit.

Whenever C h_0<H_exit, the geometry survives for all time with strictly positive probability at least 1-C h_0/H_exit. This is a proved nonexit bound, not an assumed long-run certificate.

Suppose sum_t eta_t=infinity and the step bound above holds. On {tau=infinity}, every h_c converges to zero almost surely. Proof: the stopped nonnegative supermartingale converges. Its predictable decrements have finite total sum almost surely, since their expected total is at most h_0. If the limit were positive, the drift formula, ||S_c||^2>=M-M^2/L>0 and sum eta_t=infinity would make that total decrement infinite, a contradiction.

The full mean-preactivation displacement is exactly

  m_{c,T}-m_{c,0} = (k/2)(h_{c,T}-h_0).

Consequently, on the nonexit event,

  m_{c,T}-m_{c,0} -> -k h_0/2 <0,
  sum_{t<T} eta_t Z_{c,t} -> k h_0/2 >0.

Every fixed starting time also eventually sees a negative net mean displacement, because its h_c>0 while h_{c,T}->0. Individual updates can point upward, and the instantaneous drift tends to zero. With constant eta the 1/T time average of the gradient converges to zero, not a strictly positive constant. The rigorous long-time result here is eventual net sinking and a positive cumulative drive, with the same direction as the isolated channel at every conditional expectation.

The isolated target head has exactly the same predictions and gradients on its own group in this family, because the other channel columns have disjoint support. Thus “same direction as self” is literal in this construction, not merely the sign of an unweighted contraction.

## 6. Numerical verification of the derived recurrence

Choose C=3 conv/input channels, L=10 output classes, M=3, N=9, A=2.1, B=2, delta=.1, lambda=.1, eta=.005 and rho=(1,.8,.6,.4). Then

  h_0=.1,
  H_exit=3.9,
  2 eta M^2/(N lambda)=.1<1,
  P(nonexit)>=1-3*.1/3.9=12/13 ~=92.3077%,
  k=1.2333333333,
  limiting cumulative drive=k h_0/2=.0616666667.

Using ordinary label reuse, one 50,000-step numerical trajectory gave

  h=(3.82764e-5,.0005360813,.0003846608),
  cumulative drive=(.061643062888856,.061336083182099,.061429459146483),
  maximum h observed=.1001314924,
  minimum observed old-label sum squared norm=2.1,
  cumulative-displacement identity error=4.23e-16.

Separately, for the same C=3,M=3,L=10 family, 200 direct updates of the shared Conv weights/biases, using explicit MaxPool and an exact linear solve for all ten readout outputs, agreed with the activation recurrence to maximum error 4.44e-16. A second independent check used PyTorch float64 Conv2d -> ReLU -> MaxPool, a detached exact fitted head, the complete new-label squared loss, and autograd for every shared Conv weight and bias. Its 200-update maximum discrepancy was also 4.44e-16. These finite computations check algebra/implementation; the infinite-time claims come from the invariant-region and supermartingale proof above.

## 7. Scope and unresolved transfer

- The general stopped theorem converts **conditional** label-mean signs along a trajectory into realized weighted averages under an explicit noise condition. It does not identify a uniform random arrangement average with a training trajectory.
- Geometry persistence must be proved or measured. It is proved only for the support-separated shared-CNN toy above, not for RL-CIFAR saved runs.
- Ordinary label reuse invalidates naive repeated use of F1 in general. It is justified in the toy by an explicit old-label square, not by an independence shortcut.
- A positive angle margin does not imply a positive absolute drift bound. Vanishing feature amplitudes can make the entire drive vanish, as in the toy.
- Multiple conv layers, nonlinear heads, evolving input representations, CE-trained multi-step heads, mini-batch Adam, and jointly adapting all layers remain outside these long-time results.
- The toy has no trainable output intercept. Adding one destroys the disjoint-group inverse structure and needs a new invariant/sign argument.
- These statements establish a rigorous bridge in a specified repeated fitting model. They do not complete the user's broader requested theorem for the actual RL-CIFAR CNN.
