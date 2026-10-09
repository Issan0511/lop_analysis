# Smooth-observable averaging for a full-bias Adam equilibrium tube

2026-10-09. This note extends the proved taskwise Adam averaging theorem to bounded C2 observables, in particular every component of a local equilibrium projection pi and the normal energy z^2. It retains actual full-parameter Adam updates, global bias correction, old moments, and labels reused for H phases. Geometry of the equilibrium tube and construction of a native full-CNN reference are separate companion arguments.

The result below supplies a convergent remainder and a high-probability bound uniform over all task boundaries. It does not replace stochastic updates by an ODE without controlling their cumulative error.

## 1. Base averaging notation

Use the hypotheses and notation of [taskwise Adam averaging](adam_task_averaging.md). Thus the actual update in task k, phase r, is

    theta_{k,r}=theta_{k,r-1}
        -delta_k h(theta_{k,r-1}) q_{k,r},
    delta_k=eta_0(k+1)^(-r), 1/2<r<=1,

where q is the ordinary bias-corrected Adam quotient, moments are never reset at task boundaries, and h is bounded and Lipschitz. Standard deterministic learning rates are the case h=1. The full raw gradient is bounded and Lipschitz on the localized region and is extended outside it as in the base theorem.

Let theta_k denote the task boundary, and set

    A_k=sum_{r=1}^H h(theta_{k,r-1})q_{k,r},
    F(theta,X_k)=h(theta)sum_{r=1}^H q_r^*(theta,X_k),
    barF(theta)=E[F(theta,X_k)].

The stationary reward q_r^* uses the same assignment at each of its H task occurrences. It is not the iid-per-step stationary response. With the base constants,

    ||A_k||_infinity<=B=H h_max K,
    ||F||_infinity<=B_F=H h_max K,
    Lip_theta(F)<=L_F,
    memory_oscillation(F;m)<=V rho^m,
    U_0=V/(1-rho).

The base theorem supplies the pathwise bound

    sum_k delta_k ||A_k-F(theta_k,X_k)||_infinity
        <=B_track,                                  (1.1)

including parameter motion, initialization, global bias correction and full moment history. The corrected expression for B_track includes the within-task factor H in C_move0.

All statements below can be applied to a globally extended auxiliary process, then used up to the original process's first exit from the region where the extensions agree. This does not presume stability of the original CNN.

## 2. Observable hypotheses

Let phi=(phi_1,...,phi_q) consist of q real C2 functions. Suppose the extensions satisfy

    A_i=sup_theta ||grad phi_i(theta)||_1<infinity,
    B_i=sup_theta sum_{j,l}|partial_j partial_l phi_i(theta)|<infinity.

The second condition implies both

    ||grad phi_i(theta)-grad phi_i(theta')||_1
        <=B_i||theta-theta'||_infinity

and a Taylor remainder bounded by `(B_i/2)||Delta theta||_infinity^2`. Such global extensions exist for smooth observables on a compact subset of a slightly larger smooth coordinate neighborhood, using a cutoff. The observable values themselves need not be bounded if these derivative bounds hold; a cutoff can make the values bounded as well.

The frozen scalar reward associated with phi_i is

    F_i^phi(theta,X)=grad phi_i(theta)^T F(theta,X),
    barF_i^phi(theta)=grad phi_i(theta)^T barF(theta).

It has the explicit bounds

    |F_i^phi|<=A_i B_F,
    L_i:=Lip_theta(F_i^phi)<=A_i L_F+B_i B_F,
    memory_oscillation(F_i^phi;m)<=V_i rho^m,
    V_i=A_i V,
    U_i=V_i/(1-rho)=A_i U_0.                       (2.1)

Crucially, this is an application of the Poisson construction to the *new reward*. One cannot multiply an already-convergent vector remainder series by an arbitrary varying grad phi and assume convergence.

## 3. Exact observable expansion and its tracking/Taylor error

One task obeys theta_{k+1}-theta_k=-delta_k A_k. Taylor expansion at its starting point gives

    phi_i(theta_{k+1})-phi_i(theta_k)
       =-delta_k grad phi_i(theta_k)^T A_k+T_{i,k},
    |T_{i,k}|<=(B_i/2)B^2 delta_k^2.                (3.1)

Using (1.1), replacing A_k by F(theta_k,X_k) introduces a scalar error whose total absolute sum is at most A_i B_track. The total absolute Taylor error is at most

    B_Taylor,i=(B_i/2)B^2 sum_k delta_k^2.           (3.2)

This task-level Taylor bound already includes all H intermediate updates; there is no missing extra first-order phase term.

## 4. Poisson corrector for each observable

Apply the iid-task-history shift operator P to the scalar reward in section 2. Its Poisson solution is

    U_{i,theta}=sum_{n>=0}P^n(F_i^phi(theta,.)-barF_i^phi(theta)),
    Z_{i,theta}=P U_{i,theta},
    ||U_{i,theta}||,||Z_{i,theta}||<=U_i.

The corrector's parameter modulus is

    omega_i(x)=2sum_{n>=0}min(L_i x,V_i rho^n).

With b_rho=|log rho| and C_rho=1+1/(1-rho),

    omega_i(x)<=2L_i x[C_rho+log_+(V_i/(L_i x))/b_rho].

Use the continuous zero convention when a constant vanishes. Let

    S_2=sum_{k>=0}(k+1)^(-2r),
    S_log=sum_{k>=0}(k+1)^(-2r)log(k+1).

Both are finite. The weighted corrector telescoping term has the uniform bound

    B_corrector,i=2U_i eta_0
      +2L_i B eta_0^2 [C_rho S_2
          +{S_2 log_+(V_i/(L_i B eta_0))+r S_log}/b_rho].       (4.1)

Its series converges, by vanishing terminal boundary terms, summable step variation, and the summable parameter-modulus terms. The associated task-boundary martingale has increments bounded by 2U_i, and its weighted series has variance at most

    4U_i^2 eta_0^2 S_2.                              (4.2)

The parameter theta_k is known before the task innovation Xi_k, so the martingale property is valid. No within-task reused label is re-averaged as fresh.

## 5. Observable theorem: convergence and simultaneous probability bound

There exist remainder processes R_{i,N}, each converging almost surely, such that for every N

    boxed: phi_i(theta_N)-phi_i(theta_0)
       =-sum_{k<N}delta_k grad phi_i(theta_k)^T barF(theta_k)
          +R_{i,N}.                                (5.1)

The deterministic part of the uniform remainder bound is

    D_i(eta_0)=A_i B_track+B_corrector,i
                    +(B_i/2)B^2 eta_0^2 S_2.       (5.2)

For chosen alpha_i>0 with sum_i alpha_i<=alpha,

    P{for every i and N,
       |R_{i,N}|<=D_i(eta_0)
                    +2U_i eta_0 sqrt(S_2/alpha_i)}>=1-alpha. (5.3)

In particular alpha_i=alpha/q gives simultaneous thresholds

    D_i(eta_0)+2U_i eta_0 sqrt(q S_2/alpha).

There is no independence assumption between different observable remainders; the dimension factor comes from a union bound. Applying the theorem to every ambient component of pi plus z^2 uses q=P+1 for P raw parameters. A local P-1 dimensional chart of the equilibrium manifold can reduce the count to P.

For fixed compact geometry, fixed alpha and 0<eta_0<=1, every threshold tends to zero and is O(eta_0), since eta_0^2 log(1/eta_0)=O(eta_0). These bounds may be very conservative when epsilon is tiny or the raw parameter dimension is large. They nevertheless give a positive, sufficiently small eta_0 for any prescribed strictly positive collection of margins.

Proof of convergence in (5.1): the actual-history tracking and Taylor errors are absolutely summable; the corrected frozen-history series is the sum of an L2-convergent martingale and a convergent weighted telescoping term. This is stronger than a bound on a single finite task, and does not assert absolute summability of the martingale fluctuations.

## 6. Application to a smooth zero-logit equilibrium projection

The following is conditional on the local geometry supplied by the companion proof. For N=1 binary CE, write the logit half-contrast as z(theta) and suppose

    barF(theta)=tanh(z(theta)) V(theta),
    A(theta)=grad z(theta)^T V(theta)>=a_*>0         (6.1)

throughout a compact tube. Let pi map the tube smoothly to M={z=0}, with

    pi|_M=id,
    D pi(theta) V(theta)=0.                         (6.2)

For example, pi is the endpoint at z=0 of the local normalized flow `dtheta/dz=V/A`, whose z coordinate has derivative one. Use only a compact portion on which that construction is smooth and forms valid local coordinates. The output-bias derivatives ensure that grad z is nonzero; bounds on every nonzero raw sensitivity provide smoothness of the stationary field in the proposed reference construction.

For each component of pi, (5.1) reduces, until exit, to

    pi_i(theta_N)-pi_i(theta_0)=R_{pi_i,N}.          (6.3)

For phi=z^2 it gives

    z(theta_N)^2-z(theta_0)^2
       =-sum_{k<N}delta_k 2z(theta_k)tanh(z(theta_k))A(theta_k)
          +R_{z^2,N}.                              (6.4)

On a tube |z|<=Z, the drift in (6.4) is at least

    2a_*[tanh(Z)/Z] z^2,

with the usual value one at Z=0. Thus it is nonnegative and strictly positive away from M.

If a separate nested-tube argument establishes nonexit on the simultaneous error event, then every pi_i converges by (6.3). Equation (6.4), nonnegativity of z^2, and convergence of R_{z^2,N} imply that the nonnegative drift sum is finite and z(theta_N)^2 has a limit. A positive limit would contradict sum delta_k=infinity. Hence z(theta_N)->0. The local coordinate property of (pi,z) then gives convergence of the full parameter vector to

    theta_infinity=pi_infinity in M.

The within-task parameter movement is bounded by B delta_k->0, so convergence holds for all optimizer steps as well as task boundaries.

### 6.1 Explicit derivative bounds for the projection observables

The required observable constants need not be left as unspecified finite numbers. Write W=V/A. Suppose throughout every local flow segment used to define pi that

    ||W||_infinity<=M,
    ||D W||_{infinity->infinity}<=L_W,
    max_i sum_{j,l}|partial_j partial_l W_i|<=B_W,
    |z|<=Z,
    ||D z||_1<=S,
    sum_{j,l}|partial_j partial_l z|<=T.

The flow Jacobian has infinity operator norm at most exp(L_W Z), and each component flow-Hessian has entry sum at most Z B_W exp(2L_W Z). Applying the full chain rule to pi(theta)=Psi_{-z(theta)}(theta) gives

    A_pi_i<=exp(L_W Z)+M S,

    B_pi_i<=Z B_W exp(2L_W Z)
          +2L_W exp(L_W Z) S+L_W M S^2+M T.        (6.5)

The mixed flow/time derivatives contribute the second term; the flow's second time derivative is DW W and gives the third term. The time argument's Hessian gives the final term. The bounds require all these flow segments to stay inside the region on which M,L_W,B_W are certified.

For z^2, the corresponding constants are

    A_z2<=2ZS,
    B_z2<=2S^2+2ZT.                                (6.6)

Cutoff costs can likewise be stated explicitly. If chi equals one on the working tube, has gradient l1 bound A_chi and Hessian entry-sum bound B_chi, and C_i bounds |phi_i-phi_i(theta_*)| on its support, extend by

    phi_i,ext=phi_i(theta_*)+chi[phi_i-phi_i(theta_*)].

Then valid global constants are

    A_i,ext<=A_i+C_i A_chi,
    B_i,ext<=B_i+2A_chi A_i+C_i B_chi.              (6.7)

Centering at theta_* reduces C_i and does not change any derivative or observable increment. Thus the local projection construction, its derivative bounds and an explicit smooth cutoff determine every constant in (5.3).

## 7. Retaining a strictly positive hidden-mean decrease

Let the actual target first-Conv mean be the fixed linear functional m(theta)=u^T theta. Suppose the deterministic averaged-flow endpoint has margin

    Delta(theta_0)=m(theta_0)-m(pi(theta_0))>0.

If the simultaneous pi remainder threshold is at most e_pi in each ambient coordinate, then on the nonexit event

    |m(theta_infinity)-m(pi(theta_0))|
        <=||u||_1 e_pi.

Choosing `||u||_1 e_pi<Delta(theta_0)/2` proves actual long-time mean decrease at least Delta(theta_0)/2. On an open initial region with a uniform positive endpoint margin, one sufficiently small eta_0 works throughout a compactly contained subset.

This uses a terminal displacement, not total variation of the stochastic hidden trajectory. Convergent stochastic iterates can still have infinite accumulated absolute motion. The averaging theorem alone does not justify a claim that the sum of absolute hidden updates is O(z_0^2).

The probability statement is conditional on the initial state and original zero-initialized optimizer history used by the base theorem. Moments remain continuous across all subsequent iid-label tasks. More general finite initial moment states can be included by adjusting the geometrically decaying initialization constants; no moment reset is needed or implied.

## 8. Localization and intermediate-step margin

Extend the smooth coefficients and observables from a slightly larger tube neighborhood using bounded Lipschitz/C2 cutoffs. Apply (5.3) to that globally defined auxiliary process. It matches the original CNN/Adam process up to first exit from the coefficient-agreement region. The inequalities (6.3), (6.4) can then rule out a first task-boundary exit when their margins fit inside a nested tube.

Since the probability bound is indexed by task boundaries, reserve additional geometric room for `B delta_0=H h_max K eta_0` of parameter movement inside a task. Equivalently bound the change in each tube-defining coordinate by its derivative bound times this quantity. A task-boundary barrier alone does not exclude an intermediate-phase exit.

This extension/localization step is a proof device. The original optimizer is not projected or clamped while inside the certified region, and every raw CNN weight and bias continues to train.

## 9. Warning example: zero initial head contrast can move hidden features upward

Consider the scalar binary network z=b+w h, with h>0 a trainable hidden feature, w the output contrast weight and b the output contrast bias. The frozen stationary Adam field has positive diagonal coefficients d_b,d_w,d_h and averaged dynamics

    dot b=-tanh(z)d_b,
    dot w=-tanh(z)d_w h,
    dot h=-tanh(z)d_h w.

These coefficients include the phase average over reused-label tasks. At w=0, the hidden raw gradient vanishes and its continuous coefficient is d_h=H/epsilon; using a per-step rather than per-task clock removes the common H.

Start with w_0=0 and b_0=z_0>0. Initially dot w<0. While z>0, w therefore becomes negative and dot h becomes positive. For small z_0 the local stable-normal argument keeps the solution nearby and drives z to zero, but the hidden endpoint is higher than its starting point.

Let the coefficients at the reference equilibrium (b,w,h)=(0,0,h_0) be d_b^0,d_w^0,d_h^0 and A_0=d_b^0+d_w^0 h_0^2. Using z as the clock gives

    w(z)=[d_w^0 h_0/A_0](z-z_0)+o(z_0),
    h_infinity-h_0
      =[d_h^0 d_w^0 h_0/(2A_0^2)]z_0^2+o(z_0^2)>0.

Thus output-bias relaxation alone does not ensure hidden sinking. A positive output-contrast reserve, large enough compared with the O(z_0) movement of the head, is needed in the proposed positive construction. With a fixed positive reserve the hidden endpoint decrease is generally O(z_0); with a reserve proportional to z_0 and sufficiently large proportionality constant it can be O(z_0^2). The sign and size must come from the endpoint geometry, not from averaging itself.

## 10. Completion boundary

This note proves the required observable averaging extension, including simultaneous dimension-dependent probability bounds, C2 Taylor control, almost-sure remainder convergence and applicability to equilibrium projections and normal Lyapunov functions. It does not independently assert the smooth reference CNN construction, capacity-self sign, positive endpoint margin, or nested-tube nonexit; those are the geometric companion's responsibilities.
