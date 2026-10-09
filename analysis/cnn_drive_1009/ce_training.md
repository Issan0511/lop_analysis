# CNN CE theorem over a nonzero interval of head training

Date: 2026-10-09. Independent derivation by existing_drive_proof. Parent manuscript not edited. This is a finite quantitative extension of the exact one-step head theorem, not a replacement of the original full-CNN/long-time goal.

## 1. Setup

Use the setup of analysis/cnn_drive_1009/proof.md §4: H>=0 is an N-by-A frozen feature matrix, D=partial_q H>=0 is independent of old iid uniform categorical labels, C>=2, and Y_n=e_(c_n)-u. A zero linear head is trained for T fullbatch ordinary CE-GD steps with nonnegative learning rates eta_0,...,eta_(T-1). The loss is mean CE over N images. Let S_k=sum_(j<k)eta_j and S=S_T.

Set

    h=||H||_2, d=||D||_2, c_y=sqrt((C-1)/C),
    B=h c_y/sqrt(N), L=h^2/(2N),
    G=HH^T, R=max_n sum_m G_nm,
    Q=<D,GH>_F.

Assume Q>0, which is guaranteed by D_na H_na>0 somewhere. All norms and Q,R are computed from the actual shared-filter feature matrix; no independent spatial labels, free gate permutations or weak-channel limit is introduced.

The switching gradient is always a partial derivative at fixed trained head, averaged over independent fresh labels and over old labels. Denote it g_T.

## 2. Exact positive margin of the linear-label reference

For any S>0 define V_lin=(S/N)H^T Y. This is the exact one-step construction with effective head step S, so the previous theorem applies for arbitrary finite S.

Its explicit margin is

    g_lin(S) >= a S^2 exp(-r S),
    a=c_y^2 Q/(N^3 C), r=R/N.

Proof: in the softmax lemma, t=(S/N)G_nm. Along the conditional segment Z+t'Y_m, 0<=t'<=t, the logit range is at most (S/N)sum_j G_nj <= Sr, because every centered categorical vector has coordinate range 1. Thus p_min>=exp(-Sr)/C.

For a centered vector y, y^T(diag(p)-pp^T)y=min_b sum_c p_c(y_c-b)^2 >=p_min ||y||^2. Here ||Y_m||^2=c_y^2. Integrate the softmax lemma derivative over t, then sum its nonnegative coefficients D_na H_ma. This gives the bound above.

## 3. Actual multistep CE head stays near that reference

Let

    Phi_H(V)=(1/N)sum_n [logsumexp((HV)_n)-u^T(HV)_n].

Then

    V_(k+1)=V_k + eta_k H^T Y/N - eta_k grad Phi_H(V_k).

Phi_H is convex, grad Phi_H(0)=0, and its gradient is L-Lipschitz in Frobenius norm. The rowwise softmax Hessian has operator norm <=1/2: its absolute row sum is 2p_i(1-p_i)<=1/2. Therefore the composed Hessian norm is <=h^2/(2N).

For 0<=eta_k<=2/L, the map V -> V-eta_k grad Phi_H(V) is nonexpansive. This follows by integrating its PSD Hessian along a segment; the resulting averaged Hessian has spectrum in [0,L]. Consequently, for every old-label realization,

    ||V_k||_F <= B S_k,
    eps_V := ||V_T-V_lin||_F
      <= L B sum_k eta_k S_k
      = (L B/2)(S^2-sum_k eta_k^2)
      <= (L B/2) S^2.

The sharper expression vanishes exactly for one head step. This is a pathwise bound, so old-label averaging cannot invalidate it.

## 4. Switching-gradient stability is quadratic near zero

For fixed H,D define

    g_Y(V)=(1/N)<DV, softmax(HV)-U>_F,

where U has row u. For ||V||_F<=M and ||Delta V||_F<=eps,

    |g_Y(V+Delta V)-g_Y(V)|
       <= (d h/(2N)) eps(2M+eps).

Indeed softmax(HV)-U has Frobenius norm <=h||V||_F/2, and the difference of the two softmax outputs is at most h||Delta V||_F/2. Expand the two factors of the inner product. This stronger O(M eps+eps^2) estimate is needed; the coarse global residual bound would not prove a useful small-time result.

Apply it with M=BS and eps=(LB/2)S^2:

    |g_T-g_lin(S)|
       <= (d h/(2N)) L B^2 S^3 (1+LS/4).

Thus the CE correction is O(S^3) while the positive reference margin is O(S^2).

## 5. An explicit nonzero head-training interval

Define

    S_star=min{ N/R, 2N/h^2,
                8Q/(5 e C d h^5) }.

All three entries are strictly positive when Q>0. For any T>=1 and any nonnegative schedule with 0<S<=S_star,

    g_T >= (1/2) a S^2 exp(-rS) >0.

Proof: S<=1/L implies eta_k<=1/L<2/L, validating the nonexpansiveness step. S<=1/r gives exp(rS)<=e, and 1+LS/4<=5/4. Substituting the definitions into error/margin gives

    error/margin
      <= [C d h^5/(4Q)] S exp(rS)(1+LS/4)
      <=1/2.

This covers arbitrarily many actual CE head steps over a finite accumulated learning time, with a fully explicit structural bound. It is conservative and does not claim the margin fails immediately after S_star.

## 6. Realizable shared-convolution example

Take two images consisting of one two-component input patch each:

    x_1=(4,0), x_2=(0,1).

Use two ReLU convolution filters with zero biases:

    w_1=(1/4,2), w_2=(1/2,1).

Both features are strictly active, giving H=[[1,2],[2,1]]. For target filter 1, increase weights along the empirical mean patch mu=(2,1/2) and bias at unit speed. Then K=(9,3/2), so D=[[9,0],[3/2,0]]. This is a genuine shared-filter mean-input coordinate, not independently varied feature entries.

Here N=C=2, h=3, d=sqrt(83.25), Q=138, R=9, hence

    S_star=0.018317921637110195.

Head GD with eta=0.001 for 18 steps is inside the proved interval. Direct finite-label enumeration/closed recurrences give g_18=0.0013429665043768718>0. Ten steps give 0.0004223336058763751>0.

MaxPool is not necessary to realize the example. It can be included by putting a unique maximizing patch in each pool window, with sufficiently lower other patches. Small perturbations preserve its winner and ReLU margins, giving an open set of realizable examples.

## 7. Perturbations of hidden features, sensitivity and head

The following quantitative comparison permits label-dependent deviations from a reference state, including small signed coordinate sensitivity. Let base H,D be as above and let V_ref be any label-dependent reference with ||V_ref||_F<=M. Suppose, pathwise over all old labels,

    ||H_tilde-H||_2<=eps_H,
    ||D_tilde-D||_2<=eps_D,
    ||V_tilde-V_ref||_F<=eps_V.

D_tilde need not be entrywise nonnegative. The difference of averaged switching coordinate gradients is at most

    E_pert =
      [eps_D (h+eps_H)(M+eps_V)^2
       +d(h+eps_H)eps_V(2M+eps_V)
       +d eps_H M^2] / (2N).

Proof: write g=(1/N)<DV,R(HV)>; expand changes in D, V and R. Use ||R(H_tilde V_tilde)||_F <=(h+eps_H)(M+eps_V)/2 and rowwise softmax 1/2-Lipschitz. No independence of the deviations from labels is used, only the uniform bounds.

Choose V_ref=V_lin, M=BS. If E_pert < a S^2 exp(-rS), the actual perturbed system still has g>0. Alternatively, after a frozen-H multistep theorem has supplied a positive margin, use its actual V_T as the reference and preserve that margin.

This is a finite open-neighborhood extension, not an assumption that the answer's sign already holds. eps_H,eps_D,eps_V are measurable norm distances to a structurally positive reference, and the tolerated signed component in D_tilde is explicit.

### Bound on the head error caused by changing hidden features

Suppose joint old-task training uses feature matrices H_k with ||H_k-H||_2<=eps_H at each head step. Compare its head V_tilde_k to frozen-H CE head V_k, both starting at zero. If eta_k <=4N/(h+eps_H)^2, then the update map at H_k is nonexpansive and

    ||V_tilde_T-V_T||_F
      <= eps_H c_y S/sqrt(N)
         +eps_H(2h+eps_H)B S^2/(4N).

Proof: difference of the label forcing is bounded by eps_H c_y/sqrt(N); difference of the uniform-label CE gradient at the frozen reference V_k is <=eps_H(2h+eps_H)||V_k||_F/(2N). Sum over steps and use ||V_k||_F<=BS_k. If applying this with the previous frozen-H estimate, also impose the frozen-H step condition.

Add this bound to ||V_T-V_lin||_F for eps_V in E_pert. At the switch, separately bound the actual current H_tilde and coordinate Jacobian D_tilde. This allows feature and head to move jointly over a small interval while retaining an independently stated sufficient condition.

For ReLU/MaxPool, a parameter-radius bound on eps_H and eps_D needs positive activation/winner margins or explicit directional derivatives. It does not follow from mere parameter closeness at a kink. A full-CNN proof must bound these constants along actual updates, including the signed tail.

Within a fixed activation/winner region of a finite ReLU CNN, H and D are polynomial functions of the parameters and therefore locally Lipschitz on a compact parameter ball. If explicit local constants are available, a parameter path of length ell gives eps_H<=L_H ell and eps_D<=L_D ell. Positive activation and winner margins can guarantee the path stays in that region. This makes the perturbation conditions hold for a nonzero parameter interval around a structurally positive reference, while still requiring the constants and margins to be checked for the target CNN.

### Independent check of the multistep and perturbation formulas

Script /tmp/cnn_ce_finite_steps_verify_1009.py enumerates all four old-label assignments in the shared-convolution example, runs 18 actual head CE steps, and compares to the linear-label reference. It also makes H_k label-dependent through a diagonal perturbation of operator norm <=1e-4 and gives the switching sensitivity D_tilde a negative entry of norm 1e-4.

All pathwise head-error and switching-gradient-error inequalities passed. Results:

    positive reference margin lower bound =0.0012885351353709793
    combined perturbation error upper bound=0.0002046680635146
    actual expected frozen-H gradient     =0.0013429665043768724
    actual expected perturbed gradient    =0.0013429570538035626.

Individual opposite-label realizations have negative gradients even in this short interval. The proved statement is the old/new-label expectation, as intended. These computations check the algebra and bounds; the imposed feature perturbations are not claimed to be an actual deep-CNN training trajectory.

## 8. Why a finite interval is essential: actual CE counterexample later

The same H=[[1,2],[2,1]] permits exact CE recurrences for a head **without an output bias**. Write the two-class weight difference v=V[:,1]-V[:,2].

- For equal positive labels, v=(a,a) with a_0=0 and a_(k+1)=a_k+3eta/(1+exp(3a_k)). Equal negative labels give its negative.
- For opposite labels (+,-), v=(-b,b) with b_0=0 and b_(k+1)=b_k+eta/(1+exp(b_k)). The other ordering gives its negative.

For a target-column coordinate D=[[K1,0],[K2,0]], the expected switching gradient is exactly

    g_k=[(K1+K2)a_k tanh(3a_k/2)
         -(K1-K2)b_k tanh(b_k/2)]/8.

The genuine shared-filter mean-input example has K1=9,K2=1.5. With eta=0.1, 1,000 ordinary CE head steps give

    a=2.26635024117432, b=4.570827983667447,
    g_1000=-1.2294026704682501.

The head step eta=0.1 is within the global nonexpansiveness bound 2/L=8/9. Old-task CE is 0.0011142119656184328 for equal labels and 0.01029619880191851 for opposite labels; the counterexample is not an unstable-stepsize or unlearned-task artifact.

The isolated target feature has H=(1,2)>0. For any trained scalar two-class head v without output bias and K1,K2>0, its switching gradient is sum_n K_n v tanh(H_n v/2)/(2N)>=0, strictly for v!=0. Thus this is a realizable counterexample to universal late-time self/full direction agreement in this head-without-bias family, with actual finite-step CE training and positive raw inputs. It is not the earlier logdet self definition that includes an output-bias kernel. It does not contradict the short-interval theorem. One can retain a MaxPool layer with unique winners as above.

This explicitly shows why a theorem about arbitrary full-duration CNN training cannot be obtained merely by iterating the exact one-step head result. Additional structural conditions have to exclude this interaction.

## 9. What remains for the original goal

The finite-time theorem and perturbation bounds provide rigorous progress toward CNN learning. They do not prove the full signed-tail RL-CIFAR CNN follows the self direction over its actual 30,000-update tasks or over repeated tasks. Still needed: control of label-dependent feature motion and coordinate sensitivity, nonzero inherited heads, actual optimizer projection, upstream motion in the channel mean, and a condition that excludes the explicit late-time counterexample without restating the desired sign.
