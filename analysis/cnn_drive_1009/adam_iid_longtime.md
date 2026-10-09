# All-raw CNN / iid task labels / ordinary Adam: a long-time self-direction construction

2026-10-09. This theorem concerns a specified, nonempty ReLU CNN family. Its Adam uses the standard coordinatewise first and second moments, positive epsilon, global bias corrections, and **no moment resets**. Its learning rate is explicitly damped and decays by task. It is not a theorem for constant-rate RL-CIFAR Adam.

## 1. Result and limitations

There are two genuinely shared Conv layers, multiple channels responding to every image, multiple MaxPool windows, and a free binary linear head. All 36 raw weights are trained. Each image receives an independent uniform binary label at each task boundary; the same assignment is reused in H full-batch updates, for any fixed finite H.

For any initial amplitude a0>0 and prescribed failure probability alpha in (0,1), a strictly positive initial learning-rate coefficient can be chosen from explicit finite constants so that, with probability at least 1-alpha,

    a_t -> 0,
    mean(first-Conv preactivation) -> 0 < initial mean,
    sum_t eta_t = infinity,   sum_t eta_t^2 < infinity.

Every ReLU stays on its positive branch at every finite time. At the limit all activations vanish. There is no finite-time dead-gate claim and no monotonicity claim for individual stochastic updates. At every finite state on this invariant family, the derivatives of the original full-network NTK capacity and the literal isolated-channel capacity have the same strictly positive sign in the target mean direction. Thus the long-time mean decrease is in the negative of that shared direction.

The preceding SGD theorem used a positive imbalance invariant to keep a_infinity>0. This theorem uses a different, exactly balanced all-raw-amplitude family. The different limiting value must not be attributed to the optimizer change alone.

The restrictions are substantive: 1x1 Conv, no biases, binary classes, rank-one image features, symmetric initialization, full batches, and the stated decaying/damped learning rate. The prior higher-rank / 5x5 / all-bias fixed-state results are different theorems; their assumptions are not silently combined with this one.

## 2. A realizable network and its raw gradients

Use RGB images of size 4x8. All three color planes coincide. Each image is rho_n times a fixed positive pattern made of two identical 4x4 tiles, each having maximum 1 and strict maxima in all relevant 2x2 and two-stage pool windows. For example:

    tile = [[1,.2,.7,.3], [.4,.1,.2,.1],
            [.6,.3,.8,.4], [.2,.1,.5,.2]],
    rho = (.4,.8).

The architecture is

    Conv1x1(3->4), ReLU, MaxPool2,
    Conv1x1(4->3), ReLU, MaxPool2,
    flatten(6), free Linear(6->2).

There are no biases. Initialize all 12 first-Conv and 12 second-Conv entries to a0>0. Initialize the six positive-class head entries to a0 and the six negative-class entries to -a0. No weights are tied in the optimizer.

At a state of this form with amplitude a>0, the two logits are

    f_n = 72 a^3 rho_n (1,-1).

Write labels Y_n in {-1,+1}. The derivative of CE with respect to a positive logit coefficient is tanh(72 a^3 rho_n)-Y_n. **Every individual raw Conv coordinate and every positive-head coordinate** has the same gradient

    g(a,Y) = 6 a^2 N^(-1) sum_n rho_n
                 [tanh(72 a^3 rho_n)-Y_n].                 (2.1)

Negative-head coordinates have gradient -g. This equality follows from the raw chain rule, not differentiation of a tied scalar network (which would introduce a factor of 36).

With identical zero-initialized Adam states, all positive-coordinate first moments coincide, negative-head moments are their negatives, and all second moments coincide. Their coordinatewise updates therefore preserve the stated raw-parameter family exactly, including the head antisymmetry. All channels overlap every image, and nonwinning spatial preactivations also change through the shared weights.

## 3. Exact scalar Adam and a pathwise positive lower barrier

Index optimizer updates by t=0,1,..., with pre-update amplitude a_t. Initialize M_-1=V_-1=0 and use

    M_t = beta1 M_(t-1)+(1-beta1)g(a_t,Y_task),
    V_t = beta2 V_(t-1)+(1-beta2)g(a_t,Y_task)^2,
    q_t = [M_t/(1-beta1^(t+1))]
          /[sqrt(V_t/(1-beta2^(t+1)))+epsilon],
    a_(t+1) = a_t-eta_t q_t.                              (3.1)

Take beta1=.9, beta2=.999 and any fixed epsilon>0, including 1e-8. A weighted Cauchy-Schwarz bound gives, uniformly over every gradient history,

    |q_t| <= [(1-beta2)(1-beta1^2/beta2)]^(-1/2) < 73 = K.

For task k, set

    delta_k = delta0/(k+1)^r,   1/2<r<=1,
    h(a)=a/[K(1+a)],
    eta_t=delta_k h(a_t),       kH<=t<(k+1)H.             (3.2)

Choose delta0<=.01. Then each actual step satisfies

    1-delta_k <= a_(t+1)/a_t <= 1+delta_k,
    |a_(t+1)-a_t| <= delta_k.                            (3.3)

Consequently a_t>0 for every finite t and every label realization. ReLU gates and MaxPool winners are preserved as a consequence of the raw updates, rather than imposed as an unverified trajectory assumption.

## 4. Frozen-state taskwise Adam has strictly positive mean

At any fixed a>0, (2.1) has the form

    g(a,Y_k) = mu(a)+xi_k(a),
    mu(a)=6a^2 N^(-1)sum_n rho_n tanh(72a^3rho_n)>0,
    xi_k(a)=-6a^2 N^(-1)sum_n rho_n Y_(k,n).

The xi_k are independent and symmetric across tasks. Within a task, the identical frozen-state gradient is repeated H times. At each task phase, collect the stationary first- and second-moment weights by task. Their respective positive coefficients c_i,d_i each sum to one. The stationary output is

    Q = [sum_i c_i g_i]/[sqrt(sum_i d_i g_i^2)+epsilon].

Condition on all task gradients except g_i. The function

    x -> x/[sqrt(C+d_i x^2)+epsilon]

is odd and strictly increasing. If |g_i|<=G on the fixed compact state range, its derivative along the pairing interval is at least epsilon/(G+epsilon)^2. Pairing xi_i with -xi_i gives

    E Q >= mu(a) epsilon/(G+epsilon)^2 > 0.               (4.1)

This argument conditions on independent **task blocks**, not on the H correlated copies as if they were fresh labels. It retains numerator/denominator correlation and works with unequal beta1 and beta2. Summing phases gives the averaged block drift

    Fbar(a)=h(a)sum_(phase=1)^H E Q_phase(a),
    Fbar(0)=0,   Fbar(a)>0 for a>0.                      (4.2)

Fbar is continuous on each compact [0,A], including at zero because epsilon>0. The full raw-gradient distribution is used; Adam is not applied to an expected gradient.

## 5. Relating the stationary drift to the actual moving history

The companion [averaging proof](adam_task_averaging.md) supplies the needed quantitative result. With g and h smoothly/Lipschitz extended from [0,A] by clipping their state arguments, the actual moving-state block recursion admits

    a_(KH)=a0-sum_(k<K)delta_k Fbar(a_(kH))+R_K.          (5.1)

Here R_K converges almost surely. For any prescribed alpha>0 it supplies a deterministic, explicitly evaluable bound B(delta0,alpha), made from finite gradient/Lipschitz, EMA-memory, H and epsilon constants, such that

    P(sup_K |R_K|<=B(delta0,alpha)) >= 1-alpha,
    B(delta0,alpha)->0 as delta0->0.                    (5.2)

This includes taskwise label dependence, the nonstationary zero initialization, true global bias correction, movement of the CNN state, and movement of h(a). It follows from a geometrically decaying history comparison and a Poisson/martingale decomposition. No assumption that the actual moments equilibrate at each task is used. Clipping is only a coupling device in the proof; the algorithm in (3.1)-(3.2) is never clipped or reset.

All constants for this model are finite and can be bounded directly. On [0,A], put rho_bar=N^(-1)sum rho_n and rho2_bar=N^(-1)sum rho_n^2. Then

    |g| <= G=12A^2 rho_bar,
    |partial_a g| <= L=24A rho_bar+1296A^4 rho2_bar,
    0<=h<=1/K,   |h'|<=1/K.

The same bounds apply to the clipped extensions. They do not depend on a future trajectory or on the sign of a realized update.

## 6. Nonempty high-probability infinite-time theorem

Fix a0>0, alpha in (0,1), H and r. Put A=2a0. Choose delta0>0 satisfying

    delta0<=.01,
    H delta0 < a0/4,
    B(delta0,alpha) < a0/2.                             (6.1)

Such positive choices exist by (5.2); the conditions involve only known initial data and optimizer constants. In particular, no sign or boundedness property of the future trained trajectory is assumed.

On the event in (5.2), (5.1) and nonnegative averaged drift show that every task boundary before an exit has amplitude below 1.5a0. Inside each task, (3.3) permits an additional upward excursion of at most H delta0<a0/4. A first exit through A=2a0 is therefore impossible. The pathwise positive barrier precludes an exit through zero. This proves that the auxiliary process and actual unmodified CNN/Adam process coincide forever on this event.

On that event, the nonnegative partial sums sum delta_k Fbar(a_(kH)) are bounded above by a0+sup R_K and hence converge. Since R_K converges, (5.1) implies a_(kH) converges. Within-task variations tend to zero, so the entire optimizer-step sequence converges to the same limit. A positive limit would, by continuity and (4.2), bound Fbar away from zero eventually. This contradicts sum delta_k=infinity and the finite nonnegative drift sum. Thus a_t->0.

No statement on the exceptional probability-alpha paths is needed. In fact the zero-probability label sequence that forever chooses the positive class drives a upward; a deterministic finite upper bound for *every* possible label history would be false.

### 6.1 An explicit rational witness for the probability bound

For a0=1/5, A=2/5, H=4, r=3/4, epsilon=10^-8 and alpha=1/10, all bounds can be checked with rational arithmetic. Use K=73 and the memory-decay envelope zeta=1999/2000>sqrt(.999), with rho=zeta^4. The companion constants then give an explicit bound

    B(delta0,1/10) <= C_total delta0,   0<delta0<=1,
    C_total = 6167818970385784578200530438994937565664
              /12424445797146898608389309.

To verify the upper bound without numerically evaluating infinite sums or logarithms, use

    sum_n n^(-3/2)<=3,
    sum_n log(n)n^(-3/2)<=5,
    -log(rho)>=1-rho,
    log_+(x)<=x,
    delta0 log(1/delta0)<=1,
    sqrt(3/alpha)<6.

The first two inequalities follow from the integral bound for a decreasing function and, for log(x)x^(-3/2), the same bound with its single maximum included. The maximum is less than 1 and its integral on [1,infinity) is 4. The remaining replacements only enlarge the explicit error budget.

Taking

    delta0 = 12424445797146898608389309
              /123356379407715691564010608779898751313280
           approximately 1.0071992917432996e-16

gives B<=1/20<a0/2 and satisfies all other conditions. Thus the 90% theorem has a concrete positive parameter value, not merely an asserted small-enough limit. The numerical size is extremely conservative: this certifies mathematical nonemptiness, not a practical training-rate prescription. The theorem does not establish that larger, usual rates satisfy the no-exit bound. The verifier separately records its much larger rate used only for the raw recurrence check.

## 7. The actual learning budget is infinite

This part is pathwise. Let d=1-delta0. From (3.3), a_(t-i)<=a_t d^(-i). Equation (2.1) gives |g_t|<=C a_t^2, C=12rho_bar. Since beta1/d^2<1 under delta0<=.01, the actual bias-corrected first moment satisfies

    |Mhat_t| <= C a_t^2/[1-beta1/d^2].

As the denominator is at least epsilon,

    |q_t| <= C' a_t^2,
    C'=C/[epsilon(1-beta1/d^2)].                         (7.1)

On the proven bounded event,

    |a_(t+1)-a_t|/a_t <= C' A eta_t.

If sum eta_t were finite, the total tail variation of log(a_t) would be finite, since |log(1+x)|<=2|x| for |x|<=1/2. Then a_t would have a strictly positive limit, contradicting section 6. Hence sum eta_t=infinity. Also eta_t<=delta_task/K gives sum eta_t^2<=H K^(-2)sum delta_k^2<infinity.

The sink is therefore not produced by halting after a finite total learning-rate budget.

## 8. Original full/self NTK capacity direction, not a replacement definition

Let mu_patch>0 be the mean pixel value in one input plane, averaged over images and sites. The first-channel mean preactivation is

    zbar=3mu_patch a.

Its raw mean-gradient direction adds q mu_patch to the three weights of one first-Conv filter. Let Z=rho rho^T and v=(1,-1)^T. At q=0, the full raw-parameter NTK and its q derivative are

    K_full=864a^4 Z tensor (vv^T+I_2),
    K_full'=216a^3 mu_patch Z tensor (vv^T+2I_2).        (8.1)

For the literal self model, delete the other three first-Conv channels and their second-Conv input columns. Keep all remaining raw values, including the head, unchanged. Then

    K_self=a^4 Z tensor (216vv^T+54I_2),
    K_self'=a^3 mu_patch Z tensor (216vv^T+108I_2).      (8.2)

These kernels include every remaining trainable Conv and head coordinate. Both derivatives are nonzero positive semidefinite for every a>0. Therefore, for every finite ridge lambda>0,

    partial_q log det(lambda I+K_full)>0,
    partial_q log det(lambda I+K_self)>0.                (8.3)

The rank-one image matrix Z is allowed: positive definiteness of K' is unnecessary for a strictly positive trace against (lambda I+K)^(-1). Consequently the actual long-time first-Conv mean change on the success event equals

    zbar_infinity-zbar_0=-3mu_patch a0<0,

and agrees with descent in the full and literal self capacity directions. The same forward calculation gives R_full=(mu_patch/(4a))f_full and R_self=(mu_patch/a)f_self, so the frozen new-label CE direction agrees as well. Finite actual Adam increments may point either way; the claim is the long-time cumulative result.

The verifier reports the derivative of the unhalved log determinant in (8.3). A capacity convention using one half of that logarithm multiplies each numerical slope by one half and leaves every sign assertion unchanged.

## 9. Verification and what it does not establish

[verify_adam_iid_longtime.py](verify_adam_iid_longtime.py) updates all 36 free raw coordinates with torch.optim.Adam for 60 independently relabeled tasks, each reused in four full-batch updates. All moments and global steps carry across task boundaries. It compares raw gradients, logits, parameters and moments against (2.1)-(3.2), and checks (8.1)-(8.2) by raw Jacobians and their directional derivatives before and after training.

The maximum checked discrepancy is 6.67e-16, from the NTK calculation. The raw amplitude recurrence agrees exactly at float64 precision. These are finite algebra checks. The choice delta0=.01 in that check is **not** claimed to satisfy the conservative probability bound (6.1), and its finite endpoint is not an estimate of the infinite-time limit or of success probability. The infinite-time statement follows from the analytic averaging and no-exit arguments, independently of that finite run.

This construction closes the iid-label/moment-carryover/actual-moving-Adam long-time gap for this specified family. It leaves the full RL-CIFAR architecture, nonproportional image features, biases, ten classes, changing gates and constant-rate Adam unresolved.

The independent [review](adam_iid_longtime_review.md) audits the averaging bound, the intermediate-step no-exit margin, convergence, the infinite actual learning budget, the rational probability certificate, and both raw NTK coefficient formulas. The [raw invariant-line derivation](adam_invariant_line_derivation.md) gives an additional independent chain-rule calculation.
