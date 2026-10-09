# Independent integration review: N1200/B16/E400 Adam

2026-10-09. Reviewed `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/adam_1200_longtime.md`, with `adam_1200_reference.md`, `adam_longreuse_bias.md`, `adam_b16_regularity.md`, `adam_b16_longtime.md`, `adam_task_averaging.md`, `adam_observable_averaging.md`, and `adam_constant_rate_boundary.md`. Read the saved `results/cnn_drive_1009/adam_longreuse_bias.json` certificate. No repository file was changed and no new learning experiment or certificate rerun was performed in this integration audit.

**Verdict: PASS for the stated binary, specially constructed native reference, with its distinct decaying-rate infinite-time and constant-rate finite-horizon conclusions. No blocking defect was found.** In particular, section 5's ceiling convention, cumulative-error energy estimate, and final mean-drop reserve are valid. Neither conclusion establishes the actual ten-class, rate-.001 experiment.

## 1. Reference, long-reuse bias, and regularity join correctly

The target's lines 19–38 use the independently audited first-Conv gradient-feature construction, rather than exceeding the final hidden width with a hidden-feature-rank claim. The fixed nonzero image separation supplies a positive finite `lambda1`; subsequently choosing the small positive scale `t` makes the response error smaller than that actual least normal mode. Both raw head rows are scaled at the zero-common-head reference. Independent common-head perturbations are admitted only afterward by continuity. Exact scale formulas need not hold throughout the resulting open neighborhood.

The long-reuse bias coefficient uses the same half-contrast convention `z=(f_+-f_-)/2`, raw output-bias sensitivities `+/-1/2`, and **phase-summed** stationary field as the integration. Exchangeability gives

`L_(b+,n)=N^(-1) sum_r Gamma_r`, `L_(b-,n)=-N^(-1) sum_r Gamma_r`,

so multiplication by the two Jacobian columns gives exactly

`J_b L_b = [(sum_r Gamma_r)/N] 11^T`.

There is no missing factor of two or H. The certificate's `Gamma_r>209/100` therefore gives `beta_b>30000*209/(1200*100)=209/4`. The note and saved report distinguish this frozen infinite-history statement from a conditional mean at an actual finite initialized step. The full moving-history transfer is supplied separately.

The covariance proof retains the reused-label `1/N` contribution. Its thirteen completed epochs remain independently shuffled conditional on the relevant task label vectors even when crossing task boundaries. The saved finite integer report passes all 1201 conditional-label-count comparisons, the lower-tail bound, and the stated rational bias margins. This integration does not rely on the short H=4/H=6 bias certificate.

The whole-task regularity probability is `p_N r^400`, with `r=binom(16,8)^75/binom(1200,600)`. The saved exact comparison `r<(.999)^300` gives

`p_N r^400 beta2^(-30000*8/2)<1`.

The exponents agree: `300*400=30000*8/2=120000`. Thus the general finite-N geometric-last-good-task argument supplies the required uniform inverse moment even though the regularity note's opening examples name N32/N48. The proof itself depends on finite N through this explicitly replaced probability and finite coordinate minima. The good-batch event is label-only and uniform on a compact product neighborhood of outputs and sensitivities. This justifies expectation differentiation in those independent variables, not merely along the realized CNN parameter manifold. Zero finite-history RMS remains allowed; the tracking proof requires a norm inequality and positive epsilon, not smoothness of each realized update at RMS zero.

## 2. Infinite-time conclusion and the meaning of self direction

The response/regularity inputs give `Fbar=V z`, with `V` C2 and `Sym(JV)>=aI`. This is the same nonsymmetric endpoint geometry used in the finite-N predecessor. At the reference `Q=L(JL)^(-1)J` is a projection and the ray `v=L L^T u` satisfies `Qv=v` and `u^T v=||L^T u||^2>0`.

The entrywise positivity reserve is available from the spectral scale choice: `lambda1<=C^2`, so (2.1) also implies `C_B t Gamma/epsilon<1/2`. Hence the positive supported first-mean sensitivities give strictly positive `L^T u`. A finite Taylor reserve then gives a full-dimensional cone with a uniform endpoint gap on a smaller slice; no eventual stochastic sign is inserted as a premise.

For fixed H30000, the retained-history/Poisson estimates have finite, possibly very large constants. Power rates `eta0/(k+1)^p`, `1/2<p<=1`, satisfy the logarithmically strengthened squared-summability requirement as well as a divergent sum. Here the scalar multiplier is exactly one; no hidden state-dependent factor can make the actual rate summable.

The nested-tube induction uses the general auxiliary observable identities until each previous boundary has been certified inside the region. The `H*73*eta0` margin handles every intermediate optimizer phase. Convergence of the energy remainder implies finite nonnegative dissipation and a limit for the energy; a positive limit would contradict the divergent task-rate sum. Convergent endpoint coordinates and the local chart then give every raw parameter's limit. Intermediate updates share that limit because their maximum movement tends to zero.

The retained capacity margins identify the direction of the strict terminal mean offset. They do not identify CE with the capacity, make each update negative, prove monotonicity of the capacity under all raw updates, or imply gate death. Lines 13–15 and 79–83 state these distinctions accurately.

## 3. Constant-rate finite-horizon error estimate

Lines 87–98 concern one **constant positive rate throughout all H K updates**, with a finite K chosen after fixing the averaged clock horizon S. The finite prefix of the existing decomposition is enough; infinite-time square summability is not used for this assertion.

For one observable, finite tracking costs are bounded by an initialization term `O(eta)` and a movement term `O(K eta^2)`. Corrector boundary terms cost `O(eta)`, parameter changes cost `O(K eta omega(H*73*eta))`, and task Taylor terms cost `O(K eta^2)`. A martingale maximal inequality gives `O(eta sqrt(K/alpha_i))`. Reuse of labels inside the task is already included in the iid whole-task innovation. It does not alter this martingale conditioning.

With `K=ceil(S/eta)`,

`S <= K eta < S+eta`,

and hence the four displayed contributions satisfy, for fixed S, H, geometry, and failure budgets,

- `eta -> 0`;
- `K eta^2 <= (S+eta) eta -> 0`;
- `K eta omega(H*73*eta) <= (S+eta) omega(H*73*eta) -> 0`;
- `eta sqrt(K/alpha_i) <= sqrt(eta(S+eta)/alpha_i) -> 0`.

Here `omega(r)=O(r log(1/r))` near zero. The ceiling therefore causes no loss and does not hide a vanishing rate within a fixed run. Simultaneous positive error budgets can be met with one sufficiently small constant eta. The finite boundary induction uses the same inner/outer margins and `H*73*eta<d_*`; it certifies containment only through K.

## 4. Explicit derivation of the energy bound

Write `E_k=||z(theta_k)||^2` and `d_k=2 z_k^T J_k V_k z_k>=2a E_k`. On the certified prefix, the observable identity gives

`E_(k+1)=E_k-eta d_k+(R_(k+1)-R_k)`.

Take `q=1-2a eta` in `[0,1]`. Since the remainder starts at zero,

`E_K <= q^K E_0 + sum_(k=0)^(K-1) q^(K-1-k)(R_(k+1)-R_k)`.

The weighted sum is exactly

`R_K - (1-q) sum_(j=1)^(K-1) q^(K-1-j) R_j`.

If every prefix satisfies `|R_j|<=e_E`, its upper bound is

`e_E + (1-q^(K-1)) e_E <= 2e_E`.

This proves line 102, including `K=1` and the endpoint case `q=0` (or by the immediate one-step inequality). No bound on the total variation `sum|R_(k+1)-R_k|` is needed or asserted. It would generally be incorrect to replace this cumulative bound by a separately charged error at every task; the summation-by-parts calculation avoids that loss.

Furthermore,

`q^(K/2) <= exp(-a eta K) <= exp(-a S)`.

Thus the ceiling gives at least the intended deterministic contraction. The task field is summed over H phases, so its natural clock is `K eta`; there is no missing H in the energy exponent. H is already present in the field, the normal constant, and the error constants.

## 5. From energy to the finite mean drop

On a compact subchart, the smooth identity `theta=pi(theta)` on `z=0` implies a finite bound

`|m(theta)-m(pi(theta))|<=C_m ||z(theta)||`.

It follows either from the chart's Lipschitz inverse in its z variables, or by integrating that derivative along a chart segment within a slightly larger neighborhood. This is a local bound, used only after containment.

Since `m(theta0)-m(pi(theta0))>=Delta`, the final displacement obeys

`m(theta0)-m(theta_K)`

`>=Delta-||u||_1 e_pi-C_m sqrt(q^K E_0+2e_E)`

`>=Delta-||u||_1 e_pi-C_m exp(-aS)sqrt(E_0)-C_m sqrt(2e_E)`.

The three strict `Delta/8` reserves at line 104 give a bound strictly greater than `5Delta/8`, and therefore greater than `Delta/2`. Initial energy is uniformly bounded on the smaller fixed slice, so S can be chosen once for that slice. Choosing smaller stochastic budgets afterward retains the geometric non-exit budgets as well. The probability union is over finitely many observables, not over an uncountable collection of initial states.

## 6. Constant-rate boundary is used accurately

The boundary theorem concerns infinitely many fresh-label tasks at one fixed positive rate. Its bias-gradient recurrence excludes finite convergence of any individual output-bias coordinate; its all-one-class task bursts exclude permanent containment of the entire raw state in a fixed compact region. Those conclusions apply at both two and ten classes under their stated ordinary-Adam assumptions.

The current finite theorem never passes to K=infinity while keeping eta fixed. It fixes an adequate finite S, then chooses one eta small enough, and proves the result for `K=ceil(S/eta)`. As eta is made smaller across this family of separate runs, K grows; this does not yield a guarantee for all horizons of a single fixed-rate run. The target's line 106 makes that distinction explicitly.

The boundary theorem does not forbid finite hidden-mean sinking, hidden-only limits, or stationary mean-level shifts. Conversely the integration claims none of those stronger stationary or empirical conclusions. It makes no assertion for eta=.001, 50 tasks, arbitrary CIFAR images, or ten output classes. This scope is consistent across the main statement and final limitations.
