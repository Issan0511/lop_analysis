# Actual Adam with finite taskwise label reuse: stationary averaging and a uniform error bound

2026-10-09. This note proves the missing averaging step from a frozen-state stationary Adam response to an actually moving network with retained moments. It permits a finite task length H, ordinary iid labels drawn once per task, standard beta_1=.9, beta_2=.999, fixed epsilon>0, and scalar state-dependent learning-rate factors. It does not prove the invariant region or final sinking result for a particular CNN; those are separate uses of this theorem.

The theorem applies on a compact region including a zero-amplitude boundary. It needs no lower bound on the second moment. A stopped-region formulation is given below, so compactness is not silently asserted for the original learning trajectory.

## 1. Model and noncircular geometric conditions

Let theta be all raw trainable parameters, or an exactly invariant scalar parameter a whose raw-coordinate Adam updates coincide. Task k draws an innovation Xi_k independently of past tasks. Xi_k can contain an entire iid binary label assignment and a label-independent within-task batch schedule. Each task lasts a fixed finite H updates. Phase r has gradient g_r(theta,Xi_k). The same labels in Xi_k are reused at all its phases.

Assume globally, or on a compact region followed by the extension in section 8,

    ||g_r(theta,xi)||_infinity <= G,
    ||g_r(theta,xi)-g_r(theta',xi)||_infinity
        <= L ||theta-theta'||_infinity,                 (1.1)

uniformly over phases and innovations. These bounds can come from fixed ReLU/MaxPool branches and bounded raw weights, logit Jacobians and logit Hessians. They concern the gradients, not the desired update sign.

Let h be a scalar function with

    0<=h(theta)<=h_max,
    |h(theta)-h(theta')|<=L_h ||theta-theta'||_infinity.  (1.2)

At each phase, use the actual ordinary Adam moments, including global bias correction, and update

    theta_{k,r}=theta_{k,r-1}
       -delta_k h(theta_{k,r-1}) q_{k,r},
    q_t=[b_t/(1-beta_1^t)]
         /[sqrt(v_t/(1-beta_2^t))+epsilon],
    b_t=beta_1 b_{t-1}+(1-beta_1)g_t,
    v_t=beta_2 v_{t-1}+(1-beta_2)g_t^{odot2}.          (1.3)

Moments start at zero once at global time zero and are never reset at task boundaries. Put theta_k=theta_{k,0}. No weight decay or AMSGrad is included. The deterministic base sequence is

    delta_k=eta_0 (k+1)^(-r),  1/2<r<=1.             (1.4)

The actual learning rate is delta_k h(theta_{k,r-1}); a nonsummable base sequence does not alone imply a nonsummable actual rate if h tends to zero. That point must be checked separately in an application.

For beta_1^2<beta_2, use the universal bias-corrected Adam bound

    ||q_t||_infinity <= K,
    K>=[(1-beta_2)(1-beta_1^2/beta_2)]^(-1/2).        (1.5)

Any deterministic upper bound K may be used, for example K=73 for the standard betas. The decay envelope zeta below may likewise be replaced by any strictly larger number less than one; use that same envelope consistently in rho=zeta^H and the constants.

Consequently one step moves at most h_max K delta_k and a task moves at most B delta_k, where B=H h_max K. The same K bounds every frozen infinite-history Adam quotient.

## 2. Correct frozen object: a phase-dependent stationary history of whole tasks

Extend Xi_k to iid innovations indexed by all integers. For fixed theta, form an infinite gradient history by using g_r(theta,Xi_k) at every phase r of every task k. Use the stationary, uncorrected EMAs of that history and write its phase-r quotient as q_r^*(theta,X_k), where

    X_k=(Xi_k,Xi_{k-1},Xi_{k-2},...).

Define the block reward and its stationary expectation

    F(theta,X_k)=h(theta) sum_{r=1}^H q_r^*(theta,X_k),
    barF(theta)=E[F(theta,X_k)].                     (2.1)

This expectation retains numerator/denominator dependence and the repetition of a label assignment within its task. It is neither Adam applied to an expected gradient nor the stationary response to iid per-step labels.

The main conclusions below are

    sum_{k>=0} delta_k [sum_r h(theta_{k,r-1})q_{k,r}
                       -barF(theta_k)]
    converges almost surely,                        (2.2)

and an explicit high-probability bound for the supremum of all its partial sums. Hence on finite intervals of the clock s_k=sum_{i<k}delta_i, the actual process is an asymptotic pseudotrajectory of

    d theta/ds = -barF(theta).                       (2.3)

The interpretation of (2.2) is convergence of the signed cumulative error; the stochastic noise is not asserted to be absolutely summable. The moving-history tracking error below is absolutely summable.

## 3. Two Lipschitz facts that remain valid at v=0

For two gradient histories with the same weights, let m,sigma and m',sigma' denote their first EMA and the square root of their second EMA. Viewing sigma as the weighted l2 norm of the gradient history gives

    |sigma-sigma'| <= sqrt(sum_i b_i |g_i-g_i'|^2).

Also the exact quotient identity is

    q-q'=[(m-m')+q'(sigma'-sigma)]/(sigma+epsilon).

Thus an infinite-history change of theta satisfies

    ||q^*(theta,X)-q^*(theta',X)||_infinity
       <= L_q ||theta-theta'||_infinity,
    L_q=(1+K)L/epsilon.                              (3.1)

No derivative of sqrt at zero is used. Therefore

    ||F(theta,X)-F(theta',X)||_infinity
       <= L_F ||theta-theta'||_infinity,
    L_F=H(h_max L_q+L_h K),
    ||F||_infinity<=B_F:=H h_max K.                  (3.2)

These bounds include a=0 when the scalar gradient and h are Lipschitz there. In particular, g(a,Y)=C a^2 e(a,Y) with smooth bounded e and h(a)=a/[K(1+a)^p] on [0,A] meet the condition for p>=1.

## 4. Actual moments track the frozen stationary history with summable weighted error

Set zeta=max(beta_1,sqrt(beta_2))<1. Let d_t=delta_k at a step in task k, and T_t=sum_{i=1}^t zeta^{t-i} d_i. Compare the actual history at phase t of task k with the infinite history evaluated at theta_k.

For a positive-time past gradient at step s<=t, the distance from its parameter state to the current task's boundary theta_k is at most

    h_max K [sum_{i=s}^t d_i+(H-1)delta_k].

The extra current-task term is necessary: the most recent gradient may have been evaluated H-1 updates after the boundary. Summing the first-moment weights and applying the l2 triangle inequality to the second-moment history gives, since T_t>=delta_k,

    |m_t-m_t^*| <= L H h_max K T_t+G beta_1^t,
    |sigma_t-sigma_t^*|
       <= L H h_max K T_t+G beta_2^(t/2).           (4.1)

The final terms represent the missing negative-time history at initialization; they are not a moment reset at a later task.

The difference between a zero-initialized uncorrected quotient and its bias-corrected quotient is bounded by

    (G/epsilon)[beta_1^t/(1-beta_1)+K beta_2^t].     (4.2)

For example, sigma_hat<=G and sigma_hat-sigma
<=G[1-sqrt(1-beta_2^t)]<=G beta_2^t. Equations (4.1), (4.2) and the quotient identity give

    ||q_t-q_r^*(theta_k,X_k)||_infinity
       <= C_move0 T_t+C_init0 zeta^t,
    C_move0=L H h_max K(1+K)/epsilon,
    C_init0=(G/epsilon)[1+1/(1-beta_1)+2K].          (4.3)

Accounting also for the moving h within the task, define

    C_move=h_max C_move0+H L_h h_max K^2,
    C_init=h_max C_init0.

Then the absolute weighted sum of the discrepancy between the actual block reward and F(theta_k,X_k) is at most

    B_track = C_init eta_0 zeta/(1-zeta)
              +[C_move H/(1-zeta)] sum_k delta_k^2. (4.4)

Indeed sum_t d_t zeta^t<=eta_0 zeta/(1-zeta), while

    sum_t d_t sum_{i<=t}zeta^{t-i}d_i
        <=[1/(1-zeta)]sum_t d_t^2
        =[H/(1-zeta)]sum_k delta_k^2.

The convolution inequality is Cauchy-Schwarz for each lag. This is a pathwise, uniform-in-time, summable error bound. It already includes changing network parameters, finite initialization, bias correction and the retained full moment history.

## 5. Centered frozen-history noise: a Poisson decomposition

The history chain X_k is the iid shift: append a fresh Xi and retain the old infinite past. Its transition P is independent of theta. Set

    rho=max(beta_1^H,beta_2^(H/2))=zeta^H,
    C_q=2G(1+K)/epsilon,
    V=max{2B_F, H h_max C_q/rho},
    U_0=V/(1-rho).                                 (5.1)

If two histories agree in their m most recent whole tasks, their block rewards differ by at most V rho^m. For m>=1, every phase has at least (m-1)H matching recent gradients, so the first-moment and RMS tails are bounded by 2G beta_1^{(m-1)H} and 2G beta_2^{(m-1)H/2}; section 3 gives the claim. The m=0 case follows from boundedness.

Coupling a prescribed old history and an independent stationary old history with the same future innovations gives

    ||P^n(F(theta,.)-barF(theta))||_infinity
        <=V rho^n.

Therefore the Poisson series

    U_theta=sum_{n>=0}P^n(F(theta,.)-barF(theta)),
    Z_theta=P U_theta

converges uniformly, with ||U_theta||,||Z_theta||<=U_0 and

    F(theta,X_k)-barF(theta)
       =M_k+Z_theta(X_{k-1})-Z_theta(X_k),
    M_k=U_theta(X_k)-P U_theta(X_{k-1}).             (5.2)

For the adapted choice theta=theta_k, M_k is a martingale difference with respect to the whole task history: theta_k is known before Xi_k. Also ||M_k||_infinity<=2U_0. No within-task label is treated as fresh in this step.

### 5.1 Parameter regularity of the corrector

It would be insufficient to assert that the Poisson solution is Lipschitz merely because each summand is. We instead use the explicit uniform modulus

    ||U_theta-U_theta'||, ||Z_theta-Z_theta'||
      <=omega(||theta-theta'||_infinity),
    omega(x)=2 sum_{n>=0} min(L_F x,V rho^n).         (5.3)

For x>0, with b_rho=|log rho| and C_rho=1+1/(1-rho),

    omega(x)<=2L_F x [C_rho+
                  log_+(V/(L_F x))/b_rho].         (5.4)

The zero and L_F=0 cases are interpreted by continuity. This x log(1/x) modulus avoids requiring a second-moment floor or a differentiable RMS at zero.

### 5.2 Weighted telescoping and martingale convergence

Because delta_k is nonincreasing and ||theta_{k+1}-theta_k||<=B delta_k, summation by parts yields, uniformly in N,

    ||sum_{k<N}delta_k[Z_{theta_k}(X_{k-1})
                              -Z_{theta_k}(X_k)]||
      <=2U_0 eta_0+sum_k delta_k omega(B delta_k).   (5.5)

The last sum is finite for (1.4). More precisely, put

    S_2=sum_{k>=0}(k+1)^(-2r),
    S_log=sum_{k>=0}(k+1)^(-2r)log(k+1).

Then

    B_corrector :=2U_0 eta_0
       +2L_F B eta_0^2 [C_rho S_2
          +{S_2 log_+(V/(L_F B eta_0))+r S_log}/b_rho]

is a valid upper bound for (5.5). Moreover the weighted corrector series converges: its terminal boundary term vanishes, while the weight-variation and parameter-change sums converge absolutely.

The martingale series sum_k delta_k M_k converges almost surely and in L2, because each coordinate's total variance is at most 4U_0^2 eta_0^2 S_2. Combining this with section 4 proves (2.2).

The same proof works for other deterministic nonincreasing base sequences if sum delta_k^2[1+log_+(1/delta_k)]<infinity, sum delta_k=infinity and delta_k->0. The ordinary sum-of-squares condition alone is not claimed sufficient for the particular logarithmic-modulus proof used here.

## 6. Uniform high-probability cumulative-error bound

Let d be the number of scalar coordinates controlled. Define the actual cumulative averaging error

    E_N=sum_{k<N}delta_k [sum_r h(theta_{k,r-1})q_{k,r}
                                   -barF(theta_k)].

For any alpha in (0,1), coordinatewise martingale maximal inequalities and a union bound give

    P{sup_N ||E_N||_infinity
        >B_track+B_corrector
             +2U_0 eta_0 sqrt(d S_2/alpha)} <=alpha. (6.1)

For the one-dimensional amplitude line d=1. No independence of the errors across times is assumed.

All constants depend only on the bounded/Lipschitz geometry, H, beta_1,beta_2,epsilon,h and the step exponent. For 0<eta_0<=1,

    B_track+B_corrector=O(eta_0)

with a finite constant independent of eta_0. The only apparently stronger term is eta_0^2 log(1/eta_0), and eta_0 log(1/eta_0)<=1/e. The sharper displayed formulas remain available. Thus the entire high-probability error budget can be made arbitrarily small by choosing eta_0 sufficiently small, while sum delta_k still diverges.

This bound is an absolute cumulative-error bound. If a desired drift margin itself tends to zero, its relative domination or the relevant Lyapunov argument still needs to be established; averaging alone does not supply a uniform positive drift near zero.

## 7. Exact stationary sign for independent task blocks, including H-fold reuse

For this sign theorem, restrict to full-batch reuse: at fixed theta a scalar coordinate has the same gradient G_j(theta,Xi_k) at every phase in task k. Suppose

    G_j=mu_j+xi_j,  xi_j has a distribution symmetric about zero,
    |G_j|<=M_j.

The coordinate noises can be dependent across coordinates. Different task innovations are independent. At any fixed phase, grouping the stationary EMA terms by whole tasks gives

    q_j^* = [sum_{ell>=0} A_ell G_{j,k-ell}]
             /[epsilon+sqrt(sum_{ell>=0} B_ell G_{j,k-ell}^2)],

where A_ell,B_ell are positive deterministic grouped weights and each sequence sums to one. The current task contributes only its observed phases, and older tasks contribute all H occurrences. Thus task reuse changes the weights but not independence between the scalar block values.

Condition on all block values except one. The corresponding numerator term is a positive weight times

    phi(x)=x/[epsilon+sqrt(C+B_ell x^2)], C>=0.

This function is odd and increasing, with

    epsilon/(epsilon+M_j)^2 <=phi'(x)<=1/epsilon

along the interval between xi_j and xi_j+mu_j, whenever the denominator's square root stays at most M_j. That last claim needs a slightly enlarged envelope: it is enough to replace M_j by

    M_j^+=ess sup |xi_j|+|mu_j|.                    (7.1)

Other block values are at their shifted values while the chosen block is moved between zero and its shift, so M_j^+ safely bounds the entire interpolation. Symmetry gives E[phi(xi_j)|other blocks]=0. Summing the positive numerator weights yields, for mu_j>0,

    mu_j epsilon/(epsilon+M_j^+)^2
      <=E[q_j^*]<=mu_j/epsilon.                     (7.2)

For mu_j<0 the same bounds hold after multiplying by its sign; for mu_j=0 the expectation is zero. Thus

    E[q_j^*]=c_j(theta) mu_j,
    epsilon/(epsilon+M_j^+)^2<=c_j(theta)<=1/epsilon. (7.3)

These bounds hold at every phase and hence for the block average. No existing first moment is assumed nonnegative; the stationary old moments are included in the independent-block sum.

For the useful parametrization G_j=gamma_j(theta)[mu_e(theta)+xi], |xi|<=K_xi, take M_j^+=|gamma_j|[|mu_e|+K_xi]. Then the stationary quotient has the sign of gamma_j mu_e and lies between that raw mean times the two positive coefficients in (7.3). If M_j^+/epsilon tends to zero, its ratio to the raw mean divided by epsilon lies in

    [(1+M_j^+/epsilon)^(-2),1],

which gives relative recovery, including a vanishing cubic raw signal. This result uses task-block symmetry, not a generic absolute epsilon-domination approximation.

### 7.1 Why uniform binary cross entropy supplies this symmetry

For a fixed full-batch CNN state, write the mean loss gradient as

    G(theta,Y)=average_n J_n(theta)^T[p_n(theta)-u_2]
              +average_n J_n(theta)^T[u_2-Y_n].

Complementing every independently uniform binary label flips the second vector's sign and preserves its distribution. Therefore every coordinate has symmetric centered noise, for any fixed differentiable CNN state. This holds regardless of cross-coordinate dependence and regardless of how theta depends on old tasks. Frozen tasks use independent assignments.

Consequently the frozen averaged Adam field has the same coordinatewise sign as the uniform-label population gradient. In a channel-symmetric family with equal target-coordinate gradients and equal corresponding stationary moment laws, the first-Conv mean direction inherits the intended sign. The separate CNN geometry theorem must identify that mean direction and its self-capacity interpretation.

This grouped-scalar proof does not apply merely from joint central symmetry when different minibatches within a task yield different scalar gradients from the same assignment. The averaging theorem still applies to that correlated phase process, but its stationary mean then needs a separate sign proof.

## 8. Compact stopped regions, including a=0

For the scalar application on [0,A], extend the gradient by

    g_ext(a,xi)=g(clamp(a,0,A),xi)

and similarly extend h. Bounds G,L,h_max,L_h on [0,A] become global bounds for the extensions. Analyze the extended recursion with the same innovations and moments. It agrees exactly with the original CNN/Adam recursion until the original process exits [0,A]. Apply (6.1) to the extended process and use that event in a separate exit-barrier argument. This does not assert that the original process stays inside the region.

The supremum in (6.1) is over task boundaries. A barrier for all optimizer steps must also reserve the within-task margin H h_max K delta_0, or use a sharper application-specific bound. A task-boundary upper bound alone does not rule out an intermediate-phase exit.

No singularity appears at a=0: epsilon>0, the gradient history is compared through weighted l2 norms, and the parameter corrector needs only the explicit logarithmic modulus. If h(0)=0, zero is absorbing for the parameter update even if an old first moment is temporarily nonzero; the moment itself continues its ordinary recursion.

For a finite-dimensional convex compact parameter region, a nonexpansive projection followed by the gradient map gives the same localization up to norm-conversion constants. No projection is introduced into the original optimizer before exit; it is a device defining the auxiliary coefficients outside the region.

## 9. What has been proved and what remains external

Proved here:

- Actual standard Adam with retained moment histories and finite H-fold label reuse admits the stationary block-averaged ODE limit under explicit bounded/Lipschitz geometry and power-decaying base steps.
- The weighted cumulative averaging error converges almost surely and has an explicit uniform high-probability bound tending to zero with eta_0.
- A zero second moment and a zero-amplitude boundary cause no gap in this argument.
- Full-batch binary-uniform label reuse has an exact stationary coordinate-sign theorem and finite relative bounds, with no iid-per-step substitution.

Still external:

- An invariant region or positive-probability nonexit event for the actual CNN;
- a Lyapunov or barrier argument converting the averaged field and the error event into a specific long-time mean decrease;
- a proof that the actual learning rates remain nonsummable when h tends to zero;
- the general multiclass, arbitrary-minibatch stationary sign, and the full RL-CIFAR long-time conclusion.

No activation other than ReLU is introduced.
