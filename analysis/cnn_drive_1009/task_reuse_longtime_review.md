# Independent review: taskwise label-reuse long-time SGD theorem

2026-10-09. Reviewed `task_reuse_longtime.md` as a mathematical argument. No numerical checks were rerun. Scope is the stated shared 1x1 Conv/ReLU/MaxPool family, with all raw weights trained by the specified damped scalar SGD and fresh assignments only at task boundaries.

**Verdict: PASS. No blocking mathematical error found.** The proof accounts for within-task label/state dependence without assuming that a reused label remains conditionally uniform after its observation. The long-time conclusion is stronger than a frozen-state or finite-task sign certificate, but only under the explicitly specified architecture, symmetry, damping and step schedule.

## 1. Raw-parameter SGD and the positive invariant

The channel-symmetric outer-product family is invariant under raw SGD with a common scalar learning rate. Positive unit vectors may be nonuniform here: SGD follows the same outer-product gradient. The uniform-coordinate restriction required for ordinary coordinate Adam does not apply to this SGD theorem.

With initially equal layer amplitudes, each of the L independently trained Conv matrices has amplitude update `-eta a^(L-1)<B,A>`. There is correctly no additional factor L. When differentiating the reduced state-dependent vector field later, the logits `a^L Bx` do have derivative L with respect to the common state a; section 4 correctly includes that factor in `sqrt(L^2+1)`.

Writing `lambda=eta a^(L-1)` gives exactly

    D^+-D=lambda^2[<B,A>^2-a^2||A||^2].

The lower bound `D^+>=(1-delta_k^2)D` follows from `||A||<=K`, the stated damping and `beta(a)<=1`. The upper bound `D^+<=D` follows from Cauchy-Schwarz. Independently, `2 sqrt(D)||B||<=a^2` gives `|a^+-a|<=delta_k beta(a) a/2<a`, so positivity is not inferred from D alone. Iteration over H steps per task gives the stated deterministic positive infinite-product lower bound. Its explicit exponential lower bound uses a valid logarithm inequality on `[0,1/4]`.

## 2. Global Lipschitz bound is on a genuinely convex domain

The lower-Lorentz domain

    C_d={(a,B): a>=sqrt(d+||B||^2)}

is a convex epigraph. Excluding the nonconvex upper constraint `D<=D_0` from the Lipschitz domain is necessary and is done correctly. The upper constraint is used only for bounds on actual states.

The components of the derivative bound check out:

    ||D A|| <= (R^2/2)sqrt(L^2+1) a^L,
    ||grad s|| <= (1/K)(sqrt(2)/sqrt(d)+p/4),
    ||D(B/a)|| <= sqrt(2)/a,
    ||D T|| <= sqrt(2)K/a+sqrt(2)C_A a^L.

For the product rule, the first term is at most `2/sqrt(d)+p/(2sqrt(2))`. The remaining two terms are at most `sqrt(2)` and `(R/2)sqrt(L^2+1)`, respectively, because `p=L+1` and `sqrt(D)<=a`. Thus the stated `L_F` is a valid global constant on all of C_d. No boundedness of the trajectory is presupposed.

The vector-field bounds are also consistent: globally `||F||<=sqrt(2)`; along actual states `D<=D_0` yields `C_F=sqrt(2) min(1,sqrt(D_0))`.

## 3. The finite-task decomposition retains all label reuse

The pathwise error bound

    ||e_k|| <= L_F C_F H(H-1) delta_k^2/2

follows from `||z_{k,j}-Z_k||<=j delta_k C_F`. This compares the entire damped vector field, including its state-dependent learning-rate factor; it does not freeze that factor while silently updating the other components.

The equality `sum_j F(Z_k;Y_k,q_{k,j})=H F(Z_k;Y_k,pi)` follows from linearity in the batch weights and the exact coverage assumption. The same assignment Y is used in all summands. Only after this equality is formed is Y averaged at the task boundary, where it is independent of the past. The error e_k is bounded pathwise and never assigned martingale mean zero.

For fixed finite H, the `O(H^2 delta_k^2)` errors are summable. The example `delta_k=kappa/[H(k+1)^rho]` correctly keeps the leading task-scale drift nonsummable while making the total within-task error summable.

## 4. Almost-supermartingale argument is not circular

The conditional inequality

    E[a_{k+1}|G_k] <= a_k-H delta_k h(Z_k)+C_H delta_k^2

and the deterministic summable tail produce a nonnegative supermartingale Q_k. Each finite-time a_k is integrable because the actual step increments have a deterministic bound. Thus applying nonnegative-supermartingale convergence does not require the almost-sure boundedness that it is used to prove.

Telescoping gives finite expectation for the nonnegative sum `sum_k H delta_k h(Z_k)`, hence that sum is finite almost surely. Q_k convergence gives `a_k->a_infinity<infinity`; the deterministic lower invariant gives `a_infinity>=sqrt(d)>0`. The vanishing within-task excursion bound transfers this convergence and boundedness to every inner update.

The resulting asymptotic proportionality of raw eta to delta is correct, so the construction does not obtain convergence merely by using a finite total learning-rate budget.

## 5. Barbalat, visible-head extinction and the final bound

h is globally Lipschitz as the expectation of the globally Lipschitz scalar component `-F_a`. Therefore

    |h(Z_{k+1})-h(Z_k)|<=L_F H C_F delta_k.

Together with `sum delta_k h(Z_k)<infinity`, `sum delta_k=infinity` and `delta_k->0`, this implies `h(Z_k)->0`. One explicit equivalent proof is to interpolate h in the clock `s_k=sum_{r<k}delta_r`: the interpolant is Lipschitz, its integral differs from the left-endpoint weighted sum by at most a constant times `sum delta_k^2`, and a nonnegative Lipschitz function with finite integral tends to zero. This confirms the excursion argument in the draft without imposing monotonicity of delta.

The coefficient h/c tends to a strictly positive finite limit because a and D have strictly positive limits. Compactness of the resulting random trajectory, the positive lower bound on a, and class-centering supply a positive path-dependent constant in the covariance bound for c. Hence `Bx_n->0` for every image. The positive Gram eigenvalue on U then gives `BP_U->0`, while every raw head update preserves `B(I-P_U)=B_0(I-P_U)`. This proves the complete head limit, including inner steps.

Finally, orthogonality of P_U and the pathwise monotonicity of D imply

    a_infinity^2 <= a_0^2-||B_0P_U||^2.

Strict initial visible head therefore yields strict final mean-preactivation decrease in every positive homogeneous layer. The proof correctly distinguishes this from gate death: a_infinity remains positive and the ReLU gates remain active.

## 6. Scope and nonblocking presentation observations

The result requires fixed finite H, the same positive coverage weights pi in each task, centered initial head, a positive Lorentz gap, the stated scalar damping and square-summable/nonsummable delta conditions. The manuscript states these requirements and does not transfer the result to standard Adam or a constant learning rate.

The L=2 capacity-self formulas agree with the earlier raw-parameter derivation at every state in the preserved family. They establish the shared long-time direction, not the sign of each reused-label gradient. The manuscript makes that distinction explicitly.

No changes are required for mathematical correctness. For readers, the short interpolated-clock proof above could optionally make the Barbalat step easier to audit, but the existing proof is sufficient.
