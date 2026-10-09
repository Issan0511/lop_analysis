# Independent audit: local constant-rate finite-time Adam drift

2026-10-09. Mathematical audit of the proposed finite constant-rate implication using the existing taskwise and observable averaging estimates. No repository edits, optimizer changes, or learning experiments were performed.

**Verdict: PASS.** The proposed theorem is valid under the explicit fixed-state drift and local branch hypotheses below. Containment is deterministic for the stated short horizon; it does not assume stochastic stability, a normal equilibrium, or the desired realized displacement. C3 stationary regularity and inverse-RMS moments are unnecessary for this finite drift implication.

## 1. Precise sufficient assumptions

Let all independent raw CNN parameters be `theta`. For the fixed dataset, let the actual selected first-filter preactivation mean be `m(theta)=u^T theta+c`, with fixed `u`.

Each task has a fixed finite H updates and draws an iid whole-task innovation containing its labels and its label-independent batch schedule. Labels may be reused through all H phases; schedules can include newly shuffled/regrouped epochs. Use ordinary Adam with zero initial moments once, retained moments, global bias correction, fixed epsilon>0, no weight decay, and betas satisfying `beta1^2<beta2`. At the standard betas the universal raw coordinate quotient bound is `K_A=73`. Write `M=H K_A`; for H30000, M=2190000.

Assume there is a **fixed finite reference** `theta_*`, independent of the later choice of eta, such that the correctly defined infinite-history, phase-summed stationary field has

`u^T Fbar(theta_*) >= d0 > 0`.

Also assume the full and literal-self original capacity derivatives in direction u are strictly positive there, and both models have the required smooth composite ReLU/MaxPool branches in a neighborhood. Choose a closed supnorm ball of radius r, contained in a slightly larger such neighborhood, on which

`u^T Fbar(theta) >= d=d0/2 > 0`

and both capacity derivatives remain positive. Take starting states with `||theta0-theta_*||_infinity<=r/4`.

Uniform bounds G and L for all phase raw-gradient maps on this ball are needed by the tracking argument. A fixed finite dataset, finite schedule/label alphabet, and the smooth compact branch supply them. The field's required local Lipschitz continuity also follows from the RMS norm inequality, these gradient bounds, and epsilon>0. No per-history lower bound on RMS is required.

This is a conditional implication from a fixed-state stationary drift certificate and capacity margins. It does not independently establish that a proposed reference satisfies those margins, or supply a practical magnitude for d0 or r.

## 2. Every-history containment, including intermediate phases

Choose any fixed

`0<S<r/(4M)`.

For a constant rate `0<eta<S`, put `K=ceil(S/eta)`. Then

`S<=K eta<S+eta<2S`.

Every raw Adam update has supnorm at most `K_A eta`, including the globally bias-corrected initial updates. Therefore the total path length through all H K updates is at most

`M K eta<2MS<r/2`.

Combining this with the starting distance r/4 shows

`||theta_(k,r)-theta_*||_infinity<3r/4`

at every update through task K, **for every label and shuffle history**. The bound is independent of the gradient's sign. In particular there is a remaining radius r/4 between the actual path and the ball boundary. No probabilistic containment or equilibrium argument is invoked.

For formal localization, extend each raw-gradient map by composing it with coordinatewise clipping onto the radius-r cube. This extension is bounded and Lipschitz in supnorm, and agrees with the original map inside the cube. It is an auxiliary coefficient definition only. The universal Adam step bound and the preceding deterministic induction show that its optimizer recursion and the actual CNN recursion agree for all H K steps and every history. No clipping or projection is performed by the actual optimizer. The linear observable m itself can be used globally, with constant gradient u and zero Hessian.

## 3. The stated finite remainder is sufficient

Apply the scalar-reward Poisson construction to `F_m(theta,X)=u^T F(theta,X)`. Let `U_m` and `omega_m` be its finite corrector bound and parameter modulus, respectively, and retain the base theorem's `C_init`, `C_move`, and `zeta=max(beta1,sqrt(beta2))<1`. One may use `U_m=||u||_1 U_0` and the corresponding scalar Lipschitz/memory constants. Near zero, `omega_m(x)=O(x log(1/x))`.

For each fixed starting state, with probability at least `1-alpha`, simultaneously for every `k<=K`, the exact mean-observable identity is

`m(theta_k)-m(theta0) = -eta sum_(j<k) u^T Fbar(theta_j) + R_k`,

with `R_0=0` and

`sup_(k<=K)|R_k|`

`<= ||u||_1 [C_init eta zeta/(1-zeta)`

`                  + C_move H K eta^2/(1-zeta)]`

`   + 2 U_m eta + K eta omega_m(M eta)`

`   + 2 U_m eta sqrt(K/alpha)`.

The first line is the actual-history tracking cost, including initial missing history, global bias correction, within-task motion and retained past moments. The next terms are the finite corrector telescoping boundary and parameter-change costs. The last term follows from the scalar martingale maximal inequality with variance at most `4 U_m^2 K eta^2`. Task innovations, not reused minibatch labels, determine its martingale conditioning.

There is no missing Taylor cost: m is affine, so its second derivative and its task-level Taylor remainder vanish identically. There is no need to take a union bound over every raw parameter because the only stochastic observable controlled here is m. All-coordinate containment was already deterministic.

For fixed S and geometry, all five terms tend to zero as eta tends to zero:

`K eta^2 <= (S+eta) eta`,

`K eta omega_m(M eta) <= (S+eta) omega_m(M eta)`,

`eta sqrt(K/alpha) <= sqrt(eta(S+eta)/alpha)`.

Thus one can choose a positive constant eta, used unchanged throughout the run, such that the remainder bound is at most `dS/2`.

## 4. Strict finite net mean drop

Containment keeps every task boundary inside the positive-drift ball. On the stated probability event,

`m(theta_K)-m(theta0)`

`<= -d K eta + dS/2`

`<= -dS/2 < 0`.

Consequently the original full/self capacity direction stays positive throughout the path, and the realized net decrease of the true mean has that direction. All raw weights and biases continue to update freely. This proves a terminal displacement, not a negative increment at every step or monotonic decrease of the capacity under the complete raw update.

The constants can be chosen uniformly on the radius-r/4 initial set. The probability guarantee is per initial point, with common constants; it does not claim one common-noise event for all uncountably many initial states.

## 5. What is unnecessary, and the exact limits

The proof does not require normal stability, full row rank of the logit Jacobian, a zero-logit equilibrium, a C2 endpoint projection, C3 of the stationary expected field, or inverse-RMS moment estimates. Those conditions are needed by the prior equilibrium/convergence route, whereas this argument needs only a locally certified positive stationary observable drift and bounded/Lipschitz phase gradients. Epsilon positivity and the RMS norm comparison handle zero second moments in finite initialized histories. Complement symmetry of binary labels is not intrinsically needed for this implication; it matters only if used to establish the proposed stationary drift certificate.

S must satisfy the deterministic movement restriction above. At H30000 the crude universal bound can make it very small. The obtained displacement `dS/2` may also be very small. Neither is asserted to match a substantial empirical loss of activity. K can be extremely large when eta is chosen sufficiently small; the theorem supplies an existence threshold, not a verified practical rate or task count.

The reference, r, d, and S are fixed before decreasing eta. The result is not an interchange of small-rate and infinite-horizon limits. It does not apply one fixed eta to every growing horizon, does not prove permanent compact containment or all-parameter convergence, and does not contradict the constant-rate boundary theorem. It supplies no guarantee for rate .001, an actual 50-task run, or a particular CIFAR trajectory unless their required fixed-state and finite-error inequalities are separately established.
