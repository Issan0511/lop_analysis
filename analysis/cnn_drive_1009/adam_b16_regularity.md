# Batch-16 Adam: smooth stationary fields without a deterministic RMS floor

2026-10-09. Scope: binary labels, N=32 or 48 fixed nearby images, batch size B=16, E finite independently shuffled epochs per task, labels reused throughout the task, and ordinary Adam with beta2=.999 and fixed epsilon>0. This note proves local C3 regularity of the frozen stationary response despite a positive probability of zero batch gradients. It does not establish the separate normal-stability or capacity conditions.

No optimizer modification is used. The argument replaces a false deterministic gradient-floor claim by uniform inverse-RMS moment bounds over the stationary history law.

## 1. Structural local condition and one good batch

Let z_n be the binary half-logit contrast and s_nj=partial_j z_n. A batch I of B distinct images has raw gradient

    g_j(theta;I,Y)=(1/B) sum_(n in I) s_nj(theta)[tanh z_n(theta)-Y_n].

Choose a compact parameter neighborhood K inside one strict ReLU/MaxPool branch. For each independent raw coordinate j, choose a fixed nonzero reference sensitivity c_j, which may be negative. Assume, uniformly on K and over all N images,

    |s_nj/c_j-1|<=e_j,
    |tanh z_n|<=mu,
    e_j+(1+e_j)mu<=1/B.                            (1.1)

For example, e_j<=1/(4B) and mu<=1/(4B) suffice. At a positive-path uniform-output reference with nearly identical images, all c_j are nonzero and these conditions hold after shrinking image separation and the parameter neighborhood. Output-bias coordinates satisfy c_j=+1/2 or -1/2 and e_j=0. A small finite layer scale may make c_j very small, but does not invalidate (1.1).

Write barY_I=(1/B)sum_(n in I)Y_n. Directly,

    |g_j+c_j barY_I|
      <=|c_j|[e_j+(1+e_j)mu] <=|c_j|/B.             (1.2)

Because B is even, an unbalanced label count has |barY_I|>=2/B. Therefore the same label-only event implies, simultaneously for every raw coordinate and every theta in K,

    sum_(n in I)Y_n !=0  ==>  |g_j(theta;I,Y)|>=gamma_j,
    gamma_j=|c_j|/B>0.                             (1.3)

This is a structural condition on current sensitivities and outputs, not an assumption about the future parameter trajectory or the desired drift sign. It supplies an event uniform in theta, which is essential for differentiating expectations later.

## 2. Independent task opportunities and exact failure probabilities

Each task independently draws N iid uniform signs once, then draws an independent uniform permutation for each of its E epochs. Each permutation is split into M=N/B batches; H=ME is the number of updates in the task. The labels are retained throughout all E epochs. The whole task, comprising labels and permutations, is independent of other tasks.

### 2.1 Conservative first-batch event

The first batch is balanced with probability

    p_B=binom(B,B/2)/2^B.

For B=16,

    p_B=6435/32768=0.196380615234375.               (2.1)

Thus a good first batch has probability about .804 and gives (1.3) for all coordinates at once. This event is independent across tasks. No union over raw coordinates loses probability.

### 2.2 Stronger whole-task event, valid for every finite E

Call a task good if at least one of its H batches is unbalanced. A task can fail this test only if the total number of positive labels is N/2. Define

    p_N=binom(N,N/2)/2^N,
    r_(N,B)=binom(B,B/2)^M / binom(N,N/2).           (2.2)

Conditional on any fixed globally balanced label assignment, one uniformly shuffled epoch has every batch balanced with probability r_(N,B). To see this, the positions of the N/2 positive labels in the permutation are a uniform subset; exactly binom(B,B/2)^M such subsets balance all M batches. The E independent permutations remain independent conditional on that fixed label assignment. Hence the exact probability of no certified good batch is

    p_fail(E)=p_N r_(N,B)^E.                       (2.3)

This calculation retains label reuse. It does not replace the E epochs by independent label draws. At a nearby unequal-sensitivity state, even a count-balanced batch can have a nonzero gradient, so (2.3) is the probability of failure of the sufficient event, not necessarily the probability that every actual raw gradient vanishes.

For the requested image counts,

    N=32: p_N=300540195/2147483648,
          r=1840410/6678671 = .27556530333654705;
    N=48: p_N=8061900920775/70368744177664,
          r=182200590/2756205443 = .06610559109907396.

For E=2 the failure probabilities are approximately .01062727128 and .000500649794. Good-task indicators are iid across tasks. Within a task, their constituent batch events need not be independent; (2.3) already accounts for that dependence.

## 3. Uniform inverse-RMS moments from the last good task

Fix one stationary phase r in {1,...,H} of the current task. Let b=beta2. The frozen stationary RMS is

    sigma_j(theta)^2=(1-b) sum_(ell>=0) b^ell g_(t-ell,j)(theta)^2.

Inspect only completed past tasks, indexed k=1,2,... backwards from the immediately preceding task. Let T be the first good task in this list. If its certified good update occurs at phase a in {1,...,H}, its lag from the current phase is

    ell=T H+r-a <=T H+r-1.

Therefore, for all theta in K simultaneously,

    sigma_j(theta)
      >=gamma_j sqrt(1-b) b^[(T H+r-1)/2].          (3.1)

With failure probability p, T has P(T=k)=(1-p)p^(k-1). The single random T can be used for every raw coordinate and every theta. In particular sigma_j is strictly positive on the entire compact K almost surely, even though no deterministic lower bound exists.

For any q>0 satisfying

    p b^(-Hq/2)<1,                                (3.2)

the explicit inverse-moment bound is

    E[sup_(theta in K) sigma_j(theta)^(-q)]
      <=gamma_j^(-q)(1-b)^(-q/2) b^[-(r-1)q/2]
         (1-p)b^(-Hq/2) / [1-p b^(-Hq/2)].         (3.3)

This follows by summing the geometric distribution, with no independence assumption between successive optimizer gradients. For simultaneous parameter-coordinate estimates one may replace gamma_j by the positive minimum over the finite raw coordinate set.

Using only the first-batch event, p=p_B. For H=4 and 6, q=8 already satisfies (3.2):

    p_B < .999^16 = .984119441815640...,
    p_B < .999^24 = .976273986583630....

Using the whole-task event, p=p_N r^E and H=ME, a sufficient condition for (3.2), for every finite E>=1, is

    r < b^(M q/2).                                (3.4)

Indeed p b^(-Hq/2)=p_N[r b^(-Mq/2)]^E<1. For beta2=.999 and q=8, both requested image counts satisfy this as exact rational inequalities:

    1840410/6678671 < (999/1000)^8,
    182200590/2756205443 < (999/1000)^12.           (3.5)

The right sides are approximately .992027944070 and .988065780494. Thus the inverse-eighth moment is finite for every finite E, not only E=2. The constants can grow very large with E or with tiny gamma_j; only finiteness and explicit dependence are claimed.

The stronger arbitrary-E statement uses independent uniform permutations in each epoch. The conservative first-batch statement only needs a uniform first batch and iid tasks, but then retains its explicit H-dependent condition (3.2).

## 4. Why inverse RMS squared is enough for C3 when epsilon is fixed

Work in a slightly larger neighborhood on which the same uniform conditions hold, and suppose every possible batch-gradient map has bounded derivatives up to order three:

    sup ||D^a g_j|| <= M_a, a=0,1,2,3.             (4.1)

The derivative norms are multilinear operator norms in the chosen finite-dimensional coordinates. The constants may depend on j; taking maxima over the finitely many coordinates is allowed. In a strict CNN branch, logits are polynomial and CE is smooth, so these bounds hold. In raw parameter coordinates, three gradient derivatives require four logit derivatives, which the fixed branch supplies.

The parameter may also be the independent pair (z,S) used to define the frozen response. In that case the gradient map is directly `B^(-1)sum s_nj(tanh z_n-Y_n)`, and the same bounds apply on a small product neighborhood satisfying (1.1).

View the stationary second-moment history as a Hilbert-space vector

    G(theta)=(sqrt(1-b)b^(ell/2)g_(t-ell,j)(theta))_(ell>=0),
    sigma=||G||_2.

Because the weights sum to one, ||D^a G||<=M_a. The first EMA numerator m likewise has ||D^a m||<=M_a, regardless of beta1 in (0,1). These infinite-history maps are C3: the possible batch-gradient functions form a finite smooth family, and their uniform derivative bounds justify differentiation of the summable weighted histories.

At sigma>0, Hilbert norm differentiation gives the safe bounds

    ||D sigma|| <=M1,
    ||D² sigma|| <=M2+M1²/sigma,
    ||D³ sigma|| <=M3+3M1M2/sigma+6M1³/sigma².       (4.2)

The last coefficient 6 is a conservative operator bound for the third derivative of the norm. Crucially, applying Cauchy--Schwarz before bounding removes the much larger inverse powers suggested by differentiating sqrt(v) and bounding its terms separately.

For the ordinary Adam quotient Q=m/(sigma+epsilon), epsilon>0 implies

    ||D³ Q|| <= C0+C1/sigma+C2/sigma²,              (4.3)

where one explicit choice is

    C0=M3/epsilon+6M1M2/epsilon²+6M1³/epsilon³
       +M0[6M1³/epsilon⁴+6M1M2/epsilon³+M3/epsilon²],
    C1=3M1³/epsilon²
       +M0[6M1³/epsilon³+3M1M2/epsilon²],
    C2=6M0 M1³/epsilon².                           (4.4)

These constants follow by three product/chain differentiations. Lower derivative orders have bounds of the same form or better: DQ is deterministically bounded, while D²Q needs at most a sigma^(-1) term. No relation such as beta1²<beta2 is required for this regularity estimate, because m and its derivatives are bounded directly.

Consequently the inverse-second moment (3.3), and hence the already established inverse-eighth moment, supplies one integrable random bound for the quotient and all its first three derivatives, uniform over theta in K. For almost every history, (3.1) also makes these derivatives continuous on K. Dominated differentiation and dominated continuity prove:

**Regularity conclusion.** Each phase frozen expectation `R^(r)(theta)=E Q^(r)(theta)` is C3 in a neighborhood of K. Its first three derivatives equal the expectations of the corresponding ordinary quotient derivatives, and are uniformly bounded by explicit constants obtained from (3.3)-(4.4). The phase-summed stationary field is C3 as well. In particular its Hessian is locally Lipschitz, with a finite bound from its third derivative.

The expectation is over the stationary product law of whole tasks. This is not a claim about fresh-label conditional expectations given an actual trained parameter state. Old labels and their repeated uses remain in each whole-task innovation.

## 5. Consequences for the endpoint construction

At zero contrasts, complementing every label in the entire history negates every batch gradient and the numerator m while preserving the squared-moment denominator. Thus the frozen response in independent variables obeys

    Rbar(0,S)=0

without a deterministic RMS floor. The preceding theorem justifies the derivatives and the smooth Hadamard representation

    Fbar(theta)=V(theta)z(theta),
    V(theta)=integral_0^1 D_z Rbar(t z(theta),S(theta)) dt.

Since Rbar is C3, V is C2. This is the regularity needed for the quantitative first/second-variation endpoint argument, once a separate positive symmetric-part bound for `J V` has been established. The present note proves regularity, not that matrix bound or the mean-direction sign.

The actual finite-history Adam process can encounter v_j=0, for instance after an initially balanced batch at uniform outputs. Its quotient is still well-defined because epsilon>0. It is neither necessary nor generally correct to claim that every finite-history update map is C3 at such a state. The existing taskwise tracking/observable argument uses the norm inequality for RMS, epsilon>0, and C2 observables; it does not require differentiability of each realized optimizer update. Its finite-initial-history errors are handled separately. The C3 claim here concerns the infinite-history expected field used for the endpoint geometry.

## 6. Limitations and nonempty scope

The sufficient neighborhood requires each raw sensitivity to be nonzero, to have the same sign across the nearby images for that coordinate, and to satisfy a quantitative relative comparability condition. It permits negative head-row sensitivities and includes both ordinary output biases. It does not claim this condition for arbitrary trained CNNs or arbitrary CIFAR images.

Near a positive-path all-bias reference, the conditions are open at every fixed positive initialization scale. A finite collection of distinct sufficiently nearby images can preserve them. Whether those images additionally provide the required full Jacobian row rank, original capacity signs, and stable averaged normal matrix is a separate construction task.

Balanced batches and their positive probability have not been removed by conditioning, balancing labels, changing epsilon, resetting moments, freezing parameters, or modifying Adam. The proof averages their effects together with all other histories and obtains integrable derivatives from the repeated independent task opportunities.
