# Constant-rate Adam: what fails at infinite time, and what can still express sinking

2026-10-09. Analytical boundary for the current local long-time theorem. No new learning experiment was run and no activation was changed. The relevant reference is `analysis/cnn_drive_1009/adam_b16_longtime.md`, whose successful trajectories use decaying rates, square-summable stochastic errors, permanent local containment, and convergence of every raw parameter.

The following conclusions apply to ordinary Adam with trainable output biases, finite epsilon>0, fixed positive learning rate eta, zero initial moments, no weight decay/projection/clipping, and infinitely many independent uniform-label tasks. They include N=1200, B=16, C=10, E=400, H=30000 updates per task, with fresh independent epoch permutations and retained moments. The proofs use only the first batch of each new task or specially chosen whole-task label events; they do not substitute fresh labels at every reused-label step.

## 1. A direct impossibility theorem for finite parameter convergence

Choose any output-bias coordinate b_c. For a mean-CE batch of B examples its raw gradient is always

    g_t = mean_batch p_c(theta_t,x) - L_t/B,
    -1 <= g_t <= 1,

where L_t is the number of labels equal to class c. At the first update of each new task, conditional on the entire old optimizer history and the new batch indices, the prediction average is already determined and

    L_t ~ Binomial(B,1/C).

This is valid because the newly assigned labels and the shuffle are independent. Within-task later gradients need not have this conditional law and are not used here.

Choose any k in {0,...,B-1} and define

    a=1/(2B),
    p_* = min(P(L=k),P(L=k+1))>0.

The two possible batch label proportions differ by 1/B, so for any real prediction average at least one is at distance at least a. Thus, at every new-task first step,

    P(|g_t|>=a | old history, batch indices) >= p_*.
                                                        (1.1)

The event occurs infinitely often almost surely. An elementary proof is to condition successively: the probability of avoiding it for the next K task starts is at most (1-p_*)^K. Letting K grow, then taking the countable union over possible last-hit times, proves infinite recurrence.

For the standard B=16,C=10 setting, k=1 gives

    p_* = P(Binomial(16,.1)=2)
         = .274521509459532,
    a=1/32.

For C=2, k=8 gives p_*=.174560546875. These are lower bounds uniform in the current prediction, not approximations assuming the model already outputs uniformly.

Now use the actual Adam recursion

    m_t=beta1 m_(t-1)+(1-beta1)g_t,
    q_t=mhat_t/(sqrt(vhat_t)+epsilon),
    b_(t+1)-b_t=-eta q_t.

Because |g_t|<=1 and moments start at zero, the globally bias-corrected second moment obeys vhat_t<=1 at every time. If b_t had a finite limit, its increments would tend to zero. The bounded denominator would force mhat_t→0, hence m_t→0; the first-moment recurrence would then force g_t→0. This contradicts (1.1).

**Therefore every individual output-bias coordinate fails to converge to a finite value almost surely. In particular, convergence of every raw CNN parameter, as asserted by the decaying-rate theorem, is impossible at a fixed positive learning rate under these assumptions.** This does not depend on the hidden architecture, a ReLU derivative, or an assumed frozen state.

Retaining old Adam moments does not remove the contradiction. It is explicitly included in the recurrence.

## 2. A quantitative nonvanishing-update consequence

If |g_t|>=a, the recurrence implies

    max(|m_t|,|m_(t-1)|)
       >= [(1-beta1)/(1+beta1)] a.

Otherwise the recurrence could not produce such a gradient. Since |mhat_s|>=|m_s| and sqrt(vhat_s)+epsilon<=1+epsilon, at one of those two adjacent updates,

    |b_(s+1)-b_s|
      >= eta [(1-beta1)/(1+beta1)]/[2B(1+epsilon)].  (2.1)

For beta1=.9 and B=16, the lower bound is eta/[608(1+epsilon)]. This occurs infinitely often almost surely. The large task duration H=30000 makes task starts sparse in optimizer time, but does not make their infinite number finite or make these jumps vanish.

Nonvanishing jumps alone would not prove exit from every bounded region: a bounded stochastic process can keep jumping. A separate stronger argument supplies that conclusion for this ordinary-Adam/all-bias setting.

## 3. Permanent containment in any fixed compact parameter tube has probability zero

Fix any compact region K of finite raw parameters and choose a class c. The network has finitely many fixed images and continuous finite logits on K, so softmax never assigns exact probability one. There is a uniform number kappa>0 such that

    p_c(theta,x_n) <= 1-kappa
    for every theta in K and every image n.

The coordinate b_c has finite width W on K. Consider a block of consecutive tasks in which every one of the N labels is c. If the process stays in K throughout this block, then every batch, every epoch, and every shuffle in the block satisfies

    g_t <= -kappa.

Starting from any previous moment with |m|<=1, after ell such updates,

    m <= -kappa+(1+kappa) beta1^ell.

Choose a finite burn length ell0 with

    beta1^ell0 <= kappa/[2(1+kappa)].

Thereafter m<=-kappa/2 for the rest of the forced block. Global bias correction only increases its negative magnitude, and the second-moment denominator remains at most 1+epsilon. Thus each subsequent update has

    b_(t+1)-b_t >= eta kappa/[2(1+epsilon)].

More than

    ell0 + 2(1+epsilon)W/(eta kappa)

updates of this type are incompatible with remaining in K. Choose a finite integer M so that M whole tasks last longer than this bound. The event that all NM labels in those M tasks equal c has probability

    p_force=C^(-NM)>0.

Events on disjoint M-task blocks are independent. If the trajectory has not already left K, every such event forces it to leave. Consequently

    P(all iterates stay in K forever)=0,
    P(stay in K for the next LM whole tasks)
       <=(1-p_force)^L.                            (3.1)

This proves that the permanent fixed compact tube in the current theorem cannot simply be retained after replacing its decaying rate by a constant rate. It also implies that the full raw-parameter trajectory is almost surely unbounded over infinite time: apply (3.1) to the countable sequence of finite boxes [-R,R]^P. This is a statement about the supremum over an infinite path, not about typical finite-time magnitudes or divergence to infinity at every late time.

The rare event is deliberately extreme. With N=1200,C=10 its probability is 10^(-1200M). The resulting estimate can be astronomically uninformative on any practical training horizon. It establishes an infinite-time boundary, not an empirical claim that ordinary runs promptly leave a useful region.

The argument does not rule out a tight limiting distribution or recurrent returns to a bounded region. A stationary process with unbounded support can have arbitrarily large excursions almost surely while its one-time marginal distribution remains stable.

## 4. What these impossibility results do not say

The output-bias obstruction is not a theorem that the selected first-Conv mean cannot decrease or cannot converge. It does not force every hidden coordinate to fluctuate. For example, an exactly inactive ReLU subnetwork can have zero hidden gradients while the output biases continue responding to labels. More generally, a hidden observable can settle or remain systematically low even when the complete parameter vector does not converge.

The results also do not prove that logit statistics, validation performance, activation distributions, or task-boundary averages cannot converge in distribution. Nor do they decide the sign of an actual hidden-mean displacement over a long finite run.

Thus “constant learning rate is impossible” is too broad. The precise incompatible package is **a fixed positive ordinary-Adam rate, endlessly fresh random-label tasks, freely trained output biases, permanent containment in a fixed compact region, and convergence of every raw parameter**. Removing the last two conclusions leaves meaningful questions.

Weight decay, bias constraints, projections, a vanishing learning rate, or only finitely many new tasks would change the hypotheses. No such modification is silently assumed for standard RL-CIFAR here.

## 5. Different meanings of a long-term self direction

Let the target first-Conv mean be the actual affine observable m(theta)=u^T theta+constant for the fixed image dataset. Under constant-rate Adam,

    m(theta_T)-m(theta_0)=-eta sum_(t<T) u^T q_t.   (5.1)

This separates several logically different targets:

1. **A negative net offset after adaptation:** m_T<m_0 over a long finite horizon, or a limiting mean level below its initial level. Full-parameter convergence is unnecessary.
2. **A negative asymptotic velocity:** limsup [m_T-m_0]/T<0, equivalently a strictly positive long-run average u^Tq. This implies continuing unbounded decrease at linear speed and is much stronger than adaptation followed by stabilization.
3. **A lower stationary or time-averaged level:** a limiting distribution or occupation measure has a lower mean/median m, more mass below an activation threshold, or a lower temporal average of m. Its average velocity can be zero.
4. **A local conditional restoring direction:** while the state is in a specified region, the averaged response to fresh tasks points toward a lower mean. This can explain a transient shift without implying that the direction persists outside the region or forever.

The original positive derivative of the capacity-self functional in direction u supplies a direction for comparison. It does not make CE identical to that capacity, does not remove Adam's retained-history dependence, and does not itself choose among the four targets above. The current decaying-rate theorem proves a strict terminal net offset; it does not prove a nonzero asymptotic mean velocity.

## 6. Why a genuine stationary regime cannot have a persistent signed mean velocity

Suppose an autonomous limiting augmented process has a stationary law under which m is integrable. The augmented state must include parameters, first/second moments, and the task phase/current task information. Then stationarity gives

    E_pi[m_(t+1)-m_t]=0.

Under constant eta and integrability of the update this implies

    E_pi[u^T q_t]=0.                              (6.1)

If the stationary process is ergodic, the temporal average has the same value. Even without ergodicity, stationarity and integrability imply m_T/T→0 almost surely by a tail-sum/Borel–Cantelli argument, hence the telescoping average in (5.1) vanishes.

This does not preclude E_pi[m] being lower than m_0, or a negative conditional drift in one part of state space balanced by positive drift elsewhere. It also does not preclude more frequent negative than positive increments if their magnitudes compensate. These are different statements from a strictly negative stationary mean velocity.

Continuing global Adam bias correction makes the exact finite-time process time-inhomogeneous. An invariant-law statement should therefore be formulated for the asymptotically uncorrected augmented dynamics, or as an explicitly proved asymptotic distributional statement for the corrected process. One cannot attach an invariant probability law to a counter that increases forever and call the issue settled.

Existence, uniqueness, recurrence, and integrability of such a law for the full CNN are not proved here. In particular, output/common-mode symmetries and hidden scale directions can matter. Unbounded path excursions from Section 3 alone do not refute a stationary marginal law.

## 7. A finite-horizon constant-rate theorem is available from the existing machinery

The loss of square summability blocks the current infinite-time remainder argument, but its finite-horizon estimates remain useful. Under the same bounded local coefficient/observable hypotheses, use a constant task rate eta for K tasks. Let B=H K_Adam be the task movement bound. For one smooth observable phi_i, retain the previous constants A_i,B_i,U_i,omega_i and the tracking constants. The same decomposition gives a uniform remainder bound over all k<=K with probability at least 1-alpha_i:

    sup_(k<=K)|R_i,k|
      <= A_i[C_init eta zeta/(1-zeta)
                +C_move H K eta²/(1-zeta)]
         +2 U_i eta+K eta omega_i(B eta)
         +(B_i/2) B² K eta²
         +2 U_i eta sqrt(K/alpha_i).              (7.1)

This is the finite sum version of the established tracking, Poisson-corrector, Taylor, and martingale bounds. The notation B here is the movement bound, not the minibatch size. A union bound controls the finite family of endpoint and normal-energy observables. The within-task margin still requires H K_Adam eta<d_*.

For any fixed averaged clock horizon S=K eta, the right side tends to zero as eta→0: it is of order sqrt(S eta) plus S eta log(1/eta) and eta, with fixed probability and geometry constants. Thus, where the same local averaged endpoint geometry is established, one can prove:

**For a prescribed finite clock horizon sufficiently long for the averaged flow to realize a strict mean drop, and any prescribed failure probability, there is a positive constant learning rate small enough that the actual Adam path stays in the certified region through that horizon and has a strict final mean drop with the stated probability.**

This conclusion is constant-rate within the entire finite run. It does not claim one eta works for all horizons S→infinity. The original N1200/C10/H30000 reference geometry is still an independent missing step; (7.1) does not supply it automatically.

For an explicit energy estimate, suppose the local observable recurrence has drift at least aE and a eta<=1, and sup|R_E|<=e_E. Summation by parts gives

    E_K <= (1-a eta)^K E_0+2 e_E.                 (7.2)

Combine this with a small endpoint-coordinate error and a local bound |m(theta)-m(pi(theta))|<=C sqrt(E(theta)). A pre-existing deterministic endpoint margin Delta then remains positive at K when

    ||u||_1 e_pi
       +C sqrt((1-a eta)^K E_0+2e_E) < Delta.

This is a concrete finite constant-rate alternative to requiring all parameters to converge.

## 8. Finite-window time averages and what remains open

Dividing the same finite cumulative-error decomposition by K eta yields, schematically and with the previous explicit constants,

    ||(1/K)sum_(k<K)[actual_block_update-Fbar(theta_k)]||
      <= O(1/K)+O(eta)+omega(B eta)
                   +O(sqrt(1/(K alpha))).         (8.1)

The logarithmic corrector modulus gives omega(B eta)=O(eta log(1/eta)). This holds for a globally bounded extension, or for the actual process up to the certified finite stopping horizon. It can establish a finite-window mean direction if the averaged signal exceeds that bias and fluctuation budget. At fixed eta, the deterministic discrepancy does not vanish merely by sending K to infinity. Replacing this statement by exact moving-process stationarity would require a separate argument.

Potential mathematically appropriate next targets are therefore a quantified finite-time net mean drop, a post-transient time-averaged mean level, a stationary/occupation-measure shift if such a measure is proved to exist, or a hidden-only absorbing/limiting observable. A strict negative infinite-time velocity is a different and stronger hypothesis, and a stationary integrable mean would exclude it.

The current rare-burst/nonconvergence results close the door on carrying over the exact permanent-tube/all-parameter-convergence theorem unchanged. They leave these other formulations open. They neither prove nor disprove the empirical long-term tendency of the first-Conv mean to follow the capacity-self direction under standard constant-rate RL-CIFAR.
