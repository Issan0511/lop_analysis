# Independent review: actual raw CNN / iid task labels / ordinary Adam long-time theorem

2026-10-09. Reviewed read-only:

- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/adam_iid_longtime.md`
- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/verify_adam_iid_longtime.py`
- `/home/issan/Projects/claude/wt/cnn_drive_1009/results/cnn_drive_1009/adam_iid_longtime.json`
- The companion [averaging proof](adam_task_averaging.md), after its current-task tracking H factor was corrected.

**Result: PASS. No blocking issue found.** I did not rerun the finite Torch experiment or the certificate. The proof, source, constants, and saved output were independently inspected. This is a high-probability infinite-time theorem for the specified symmetric, bias-free binary CNN and its explicit decaying/damped learning rate. It is not a theorem for ordinary constant-rate RL-CIFAR training.

## 1. Real raw network and exact invariant line

Both 1x1 Conv layers have independently trained raw entries. The pattern has positive pixels and unique maxima in every first-stage 2x2 cell; its first pooled tile has entries `(1,.7;.6,.8)` and therefore a unique second-stage maximum. The two final spatial sites have equal selected values but are not pooled together, so that equality creates no MaxPool tie.

With RGB inputs rho_n times this pattern, all first-Conv entries a, all second-Conv entries a, and binary head rows +/-a:

- A selected first-Conv activation is 3a rho_n.
- A selected second-Conv activation is 12a^2 rho_n.
- Each logit sums six head terms, yielding +/-72a^3 rho_n.

Let e_n=tanh(72a^3rho_n)-Y_n. A positive-head raw gradient is `(e_n/2)*12a^2rho_n=6a^2rho_n e_n`. For each second-Conv entry the two spatial positions and opposite head classes give the same value. For each first-Conv RGB entry, the three second-layer channels and two selected spatial positions again give the same value. Averaging images yields equation (2.1). Negative-head raw gradients have the opposite sign. All 36 coordinates are counted; differentiating a tied scalar model would indeed multiply the common coordinate gradient by 36 and is not the optimizer used here.

Zero initial moments, coordinatewise Adam with common beta/epsilon, and one shared scalar learning rate preserve the raw sign symmetry exactly. The code uses all 36 free Torch parameters and changes only the optimizer's scalar learning-rate field; it does not project, tie, refit, freeze, or reset them.

Each task draws two ordinary independent binary labels and reuses that same assignment for H steps. Label assignments are not balanced by construction. Moments and the global bias-correction index carry across all tasks.

## 2. Positivity and stationary task-block drift

The universal Adam quotient bound is valid because beta1^2<beta2 and moments start at zero once. K=73 is a rigorous upper bound. With eta=delta a/[K(1+a)] and delta<=.01,

    1-delta <= a_next/a <= 1+delta.

Thus a remains positive at every finite step on every label path, not merely on the later high-probability event. This preserves the positive ReLU branches and unique pool winners of this exact invariant family.

For frozen a, ordinary independent binary label complementation gives symmetric centered task-gradient noise. Grouping the exponentially weighted first/second moments by task preserves positive weights and independence between task block values. The conditioning/odd-monotonicity proof correctly retains numerator-denominator dependence and the H copies within a task. The gradient envelope `G=12 A^2 average rho` also bounds the interpolation used in the derivative inequality. Consequently Fbar(a)>0 for every a>0 and Fbar is continuous at zero, with Fbar(0)=0.

The statement concerns the stationary frozen block field. It is transferred to the moving process by the separate averaging proof, not by declaring its actual history stationary.

## 3. Averaging and infinite no-exit argument

The corrected averaging proof includes the factor H needed to compare a late current-task gradient with the task-boundary state. Its RMS-history bound is valid at zero variance; its bias-correction error is summable; its whole-task iid-shift Poisson decomposition uses a valid `x log(1/x)` parameter modulus. Thus the asserted convergent cumulative remainder and uniform high-probability error bound apply on the compactly extended scalar process.

The main proof correctly includes an intermediate-step no-exit margin. On the event `sup_K |R_K|<a0/2`, every task boundary preceding an exit satisfies a<1.5a0 because its accumulated averaged drift is nonnegative. An entire task can move upward by at most H delta0<a0/4. Hence no optimizer phase can first cross A=2a0. The pathwise positive barrier prevents a lower exit. The auxiliary clipped-coefficient process and the actual unclipped raw CNN/Adam therefore agree forever on that event.

The event of almost-sure convergence of R has probability one and can be intersected with the preceding event without reducing its claimed probability. On it, the sum of nonnegative drift terms is bounded by `a0+sup R`, hence converges. The decomposition gives a convergent amplitude at task boundaries, and within-task changes vanish. If the limit were positive, continuity and strict positivity of Fbar would force the drift sum to diverge because sum delta_k diverges. Therefore a_t tends to zero.

The proof makes no monotonicity claim for individual stochastic updates and no deterministic upper-bound claim for all possible label histories. It correctly states a high-probability pathwise conclusion; it does not infer an unconditional expectation statement from the success event while ignoring the exit event.

## 4. Actual infinite learning budget

This part is independent of stochastic averaging. The following reproduces the independently derived pathwise lemma.

With d=1-delta0, positivity gives `a_(t-i)<=a_t d^(-i)`. The raw gradient has magnitude at most C a_t^2. Since beta1/d^2<1 for delta0<=.01, the bias-corrected first EMA is bounded by

    C a_t^2/[1-beta1/d^2].

The Adam denominator is at least epsilon, hence q_t=O(a_t^2) pathwise, including all old moment contributions. On the no-exit event a_t<=A, finite sum eta_t would imply finite total tail variation of log a_t and therefore a positive limiting amplitude. This contradicts a_t->0. Thus the actual learning-rate sum diverges. Its squared sum is bounded by `H K^(-2) sum delta_k^2` and converges.

The result is not obtained by exhausting a finite total learning budget. The extra small-step inequality needed by this argument is explicitly verified by the rational certificate.

## 5. Exact rational probability certificate

The certificate function uses `Fraction` for every value establishing an inequality. Float conversion occurs only when constructing explanatory output. I checked the formulas without rerunning the function.

For a0=1/5, A=2/5, rho=(2/5,4/5), the stated geometry bounds are

    G=12 A^2 average rho = 1.152,
    L=24 A average rho +1296 A^4 average rho^2 =19.03104.

The second term in L follows by differentiating tanh(72a^3rho); the first uses `|tanh-Y|<=2`. The clip extension is Lipschitz with the same constants. h_max=L_h=1/K are valid conservative bounds.

The rational zeta=1999/2000 exceeds sqrt(beta2) since zeta^2=.99900025>.999; it also exceeds beta1. Taking rho=zeta^H in all memory constants is valid. The source includes the corrected H in `move0` and matches the companion definitions of tracking, reward Lipschitz, tail influence, and Poisson bounds.

For r=3/4:

- `S2=sum n^(-3/2)<=1+integral_1^infinity x^(-3/2) dx=3`.
- `Slog=sum log(n)n^(-3/2)<=5`: the integrand is decreasing from x=2 onward, its integral over [1,infinity) is 4, and the remaining first nonzero term is less than 1. These conservative elementary bounds justify the constants used in the source.
- `-log(rho)>=1-rho` permits the rational reciprocal bound.
- For 0<delta<=1 and x>0,

      delta^2 log_+(x/delta) <= delta (x+1),

  using `log_+x<=x` and `delta log(1/delta)<=1`. This is exactly the replacement used to make the corrector error linear in delta.
- `sqrt(S2/alpha)<=sqrt(30)<6` justifies the martingale term `2 U0 * 6 * delta` for alpha=1/10.

Thus the resulting exact rational coefficient C_total satisfies `B(delta,.1)<=C_total delta` for the specified range. The chosen

    delta0=min(1/100, a0/(8H), a0/(4 C_total))

is positive and gives error at most a0/4, strictly below the required a0/2. It also gives H delta0<=a0/8<a0/4 and beta1/(1-delta0)^2<1. These comparisons are checked rationally in the source.

The saved certificate reports C_total approximately 4.964260837937849e14, delta0 approximately 1.0071992917432996e-16, and a uniform error upper bound .05=a0/4, certifying success probability at least .9. This extremely conservative analytic rate is distinct from the finite algebra check's delta0=.01. The proof and output explicitly state this distinction. The certificate proves nonemptiness of the admissible choice; it is not a practical-time estimate or an empirical success probability.

## 6. Independent full and literal-self NTK coefficient check

Let v=(1,-1), Z=rho rho^T, and let mu_patch be the true spatial mean in one RGB plane. The target mean direction adds q mu_patch to each of the three weights of one first-Conv filter.

At the full width-4 state:

- The 12 first-Conv raw Jacobian columns are each `6a^2 rho v`, contributing `432a^4 Z tensor vv^T`.
- The 12 second-Conv columns give the same contribution.
- Each class has six head columns `12a^2 rho`, giving `864a^4 Z tensor I_2`.

Thus `K_full=864a^4 Z tensor(vv^T+I_2)`. Under the target mean perturbation, the first-Conv block is constant. The three affected second-Conv input-column parameters contribute derivative `216a^3 mu_patch Z tensor vv^T`, and the head block contributes `432a^3 mu_patch Z tensor I_2`, yielding (8.1).

For the literal self, retaining only one first channel leaves three first-Conv and three second-Conv columns, contributing `216a^4 Z tensor vv^T` in total. The head feature is `3a^2rho`, giving `54a^4 Z tensor I_2`. Their derivatives are respectively `216a^3 mu_patch Z tensor vv^T` and `108a^3 mu_patch Z tensor I_2`, yielding (8.2). Nothing is refit and every remaining raw parameter is included.

Both derivatives are nonzero PSD even though Z has rank one. A positive definite ridge inverse has strictly positive trace against any nonzero PSD matrix, so the original full/self logdet capacity derivatives are strictly positive. The mean limits therefore align with their negative-gradient direction. Direct differentiation also confirms `R_full=(mu_patch/(4a))f_full` and `R_self=(mu_patch/a)f_self`.

## 7. Final scope assessment

The theorem closes the actual-moving-Adam / retained-moments / iid taskwise label-reuse gap for this nonempty, exactly invariant family, with an explicitly admissible positive damping coefficient. All raw weights are trained, channels overlap, MaxPool has multiple genuine windows, gates remain positive at finite times, and the actual learning budget is infinite.

It does not prove negative bulk preactivation, finite-time gate death, arbitrary nonproportional feature behavior, bias dynamics, multiclass stationary sign, changing within-task minibatches, or constant-rate RL-CIFAR convergence. These restrictions are stated in the main theorem and should be preserved in any summary. No further correction is required for the reviewed claim.
