# Independent audit: observable averaging and the open full-bias CNN tube

2026-10-09. Read-only review of:

- `adam_observable_averaging.md` (new observable averaging theorem);
- `adam_open_geometry.md` (the independently derived geometry/endpoint construction);
- Sections 3 and 5 of `adam_native_bias_reference.md` (native all-raw capacity and averaged-ODE reference);
- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/verify_adam_open_native.py` and its saved `results/cnn_drive_1009/adam_open_native.json`.

**Result: PASS. No blocking issue found in the stated combination.** No numerical test was rerun and no repository/source file was edited. This audit assumes the corrected base averaging theorem with the H factor in its current-task tracking constant, which was independently checked earlier.

## 1. Observable norms and Taylor control

The observable theorem uses

    A_i=sup ||grad phi_i||_1,
    B_i=sup sum_(j,l) |partial_j partial_l phi_i|.

These norms correctly match the infinity-norm parameter movement bounds. Integrating the Hessian along a segment gives a gradient l1 Lipschitz constant B_i, and the second-order Taylor remainder is at most `(B_i/2)||Delta theta||_infinity^2`. There is no missing factor of the raw parameter dimension.

The exact full-task displacement is `-delta_k A_k`, with `||A_k||_infinity<=H h_max K`. One task-level Taylor expansion therefore includes all intermediate steps and changes of h. It does not omit a first-order phase-motion term. Multiplying the already absolutely summable **tracking** discrepancy by the bounded gradient is legitimate and gives `A_i B_track`; the Taylor series has the stated square-summable bound.

## 2. A new Poisson reward, not a variable multiple of an old remainder

The proof correctly applies stochastic averaging anew to

    F_i^phi(theta,X)=grad phi_i(theta)^T F(theta,X).

Its bounds are exactly

    |F_i^phi|<=A_i B_F,
    Lip_theta(F_i^phi)<=A_i L_F+B_i B_F,
    memory_oscillation<=A_i V rho^m.

Consequently the new Poisson solution has norm at most `U_i=A_i V/(1-rho)` and the same logarithmic parameter modulus with the new constants. This is the necessary treatment of a varying observable gradient. Merely multiplying the base convergent vector-remainder series by a varying gradient would not justify convergence, and the draft explicitly avoids that error.

The weighted corrector telescoping bound and the martingale variance follow from the already checked base proof. The parameter state at each task boundary is measurable before that task's label draw, so the martingale argument is valid with H-fold label reuse. No repeated within-task label is treated as fresh.

## 3. Simultaneous probability event and convergence

The remainder in the observable identity is a sum of an absolutely summable tracking/Taylor contribution, a convergent weighted corrector, and a convergent martingale series. Its almost-sure convergence is therefore justified.

Each scalar observable's martingale variance is at most `4U_i^2 eta0^2 S2`. A maximal inequality followed by a union bound with probabilities alpha_i proves the stated simultaneous all-times event. Independence between coordinates or observable errors is not assumed. For every ambient component of pi plus z^2, q=P+1 is a valid conservative count. The dimension dependence is visible in the formula rather than suppressed.

All deterministic and probabilistic thresholds tend to zero as eta0 tends to zero; the `eta0^2 log(1/eta0)` corrector term is harmless. This yields a strictly positive, sufficiently small step coefficient for any fixed strictly positive geometric margins. It is not an assertion that a practically sized constant-rate learning rate satisfies those margins.

## 4. Smoothness and geometry of the zero-contrast field

For one image and a uniform binary task label, every raw coordinate gradient factors exactly as

    g_j=s_j(theta)[tanh z(theta)-Y].

On the constructed compact neighborhood, each s_j is bounded away from zero and `|tanh z|<1`. Thus the stationary RMS has a uniform positive floor. For the grouped task weights A_l,B_l, the exact quotient is

    s(mu-M_Y)/[|s|sqrt(1+mu^2-2mu R_Y)+epsilon].

It is smooth with uniformly bounded derivatives in the neighborhood. Its expectation is odd in mu. The removable coefficient at mu=0 is

    c_r(s,0)=[epsilon+|s|(1-sum_l A_l B_l)]/(epsilon+|s|)^2>0.

This verifies the needed smooth positive vector field V even at the uniform-output equilibrium. The old realized moment state is not presumed stationary: its discrepancy is handled by the averaging remainder.

The positive hidden weights and positive head contrast give s_j>0 in the target first-filter/bias block. The augmented patch-mean vector u is nonnegative and contains bias coefficient 1. Hence DmV>0 follows from a strict, finite reference cone and persists in a sufficiently small neighborhood. It is not an assumption of the desired future trajectory sign.

The normalized field `W=V/(Dz V)` satisfies DzW=1. Its finite-time flow therefore constructs a smooth endpoint map `pi(theta)=Psi_{-z(theta)}(theta)` under an explicit radius/displacement condition. The identities DpiV=0 and `m(theta)-m(pi(theta))>=c_m z(theta)` follow from the flow group property and integration of DmW. The map is used only as a mathematical observable, never as an optimizer projection.

## 5. Nested-tube no-exit and long-time limit

The geometry note defines a full-dimensional initial slice with `z_L<z<z_U`, a middle compact tube, and a larger outer tube inside the strict routing/capacity region. The construction does not impose equal raw weights, proportional channels, centered common logits, or zero trained biases on the admitted initial states.

On the simultaneous remainder event:

- Dpi Fbar=0 implies the endpoint coordinates remain within epsilon_pi of their initial values at every task boundary.
- The z^2 observable has averaged dissipation `2z tanh z A>=c_z z^2>=0`, so its boundary value never exceeds `z_U^2+epsilon_z` after adding the remainder budget.
- The chosen budgets put every preceding boundary strictly inside the middle tube.
- `H K delta0<dist_infinity(T_middle,complement T_outer)` rules out exit at **every optimizer phase**, including a jump through a boundary between task endpoints.

Thus the localized auxiliary process agrees forever with the original full-raw Adam process on that event. The original optimizer has no clipping, projection, reset, or freezing operation.

Convergence of the z^2 remainder and its nonnegative dissipation sum implies that z^2 has a limit. A positive limit contradicts divergent sum delta_k. Hence z tends to zero. Every pi coordinate converges. Since `(pi,z)` are valid local coordinates, the entire parameter vector converges to an equilibrium on z=0, including all intermediate optimizer steps.

The final mean comparison is exact for the affine first-Conv mean:

    |m(theta_infinity)-m(pi(theta0))|<=||u||_1 epsilon_pi.

Choosing `epsilon_pi<c_m z_L/(2||u||_1)` preserves a strict net decrease of at least `c_m z_L/2`. Actual finite updates or z itself may point either way; the proof needs neither monotonicity nor an always-favorable realized momentum.

With h=1 the actual learning rates are the taskwise base delta_k. Their sum diverges and their squared sum converges automatically. This removes the additional adaptive-rate summability issue from the earlier amplitude-collapse family.

## 6. Independent native full/literal-self NTK check

The native reference note uses head rows +/-alpha d and freely trained output biases. For its positive last-hidden vector h, differentiating every independent raw parameter gives

    K=(||h||^2+1)I_2+alpha^2||grad_hidden(d^T h)||^2 vv^T.

The I_2 term includes both output weights and output biases. The other term includes every Conv/hidden-FC weight and bias. Its first-filter mean-direction derivative is the stated

    K'=2<h,D_u h>I_2
         +2alpha^2<grad psi,D_u grad psi>vv^T.

On a fixed ReLU/MaxPool branch with positive hidden weights and nonnegative biases, hidden outputs are polynomials with nonnegative coefficients. Input and padding coefficients are nonnegative, and fixed pooling selects coordinates without introducing subtraction. Therefore first and mixed derivatives in the nonnegative mean direction are nonnegative, including shared convolution coordinates. The strictly positive `2<h,D_u h>` gives a positive definite K' lower bound.

The same reasoning holds for the literal reduced network with its own hidden vector, biases, and routing. Its output bias is retained without refitting. Output-bias cancellation changes neither K nor K'; the self network's CE contrast need not also be canceled, and no self-CE sign is required.

The reference note's averaged-ODE path-length estimate is also valid: output-bias sensitivities +/-1/2 give `||grad z||^2>=1/2`, and positive diagonal coefficient bounds imply the displayed finite speed/decrease ratio. Small positive initial z keeps the whole averaged path inside the strict neighborhood and gives the stated endpoint margin. The open-family stochastic argument adds the separate observable averaging and no-exit controls rather than inferring them from this ODE fact.

## 7. Saved finite native verification

The source uses two genuine 5x5 shared Conv layers, two hidden FC/ReLU layers, all ordinary biases, and a free binary head, with 331 independent raw parameters. Literal self deletion removes only other first-Conv channels and their matching second-Conv input columns; it retains all subsequent parameters and biases.

The raw Jacobian formulas, mean direction including bias, and gradient factorization are implemented consistently. An independent perturbation affects all coordinates, then the initial output-bias contrast is adjusted to a specified small positive z. This constructs one interior point; openness comes from the strict analytic inequalities, not from pretending this one adjusted point itself is a full-dimensional distribution.

The saved results show:

- Reference full/self K' lower bounds 6.585364805 and 4.208214024.
- Perturbation norm bounds on K' of .000107522 and .000089596, leaving strict positive margins.
- Minimum absolute raw contrast sensitivity .0170934212 and positive target alignment.
- Raw CE factorization errors below 5.6e-17.
- Correct phasewise grouped-weight covariance omega, with a positive stationary linearization and endpoint mean coefficient about 2.015555322.

The code averages phase coefficients for the last numerical linearization rather than summing them. This multiplies both its normal coefficient and mean coefficient by the same common H factor relative to a block clock; the endpoint ratio is unchanged. Its labels correctly present this as a stationary local-geometry check, not an actual long-time stochastic displacement.

These computations support the finite identities and local nonempty reference. They are not an empirical success-rate estimate, nor a computed admissible learning-rate certificate for the open-family infinite-time theorem. The proof derives the existence of a sufficiently small positive rate from finite compact geometric/observable constants.

## 8. Scope to preserve

The combined theorem gives high-probability strict long-time **net** first-Conv mean decrease on a full-dimensional open initial set of native all-bias binary CNN parameters, with ordinary Adam, no moment resets, and iid task labels reused H times. Its endpoint generally retains positive hidden features; uniform output is achieved through parameter adjustment including output bias, not through neuron death.

It is not an unconditional expectation result on the complementary exit event, and it is not a theorem for ten classes, arbitrary many-image minibatches, changing gates, or constant-rate RL-CIFAR training. All-time positive realized drift is neither assumed nor concluded. No blocking correction is needed for the reviewed claim and its stated limitations.
