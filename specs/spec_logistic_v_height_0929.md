# Frozen-readout logistic height audit — 2026-09-29

## Question and status

User requests extensive numerical and theoretical verification of the scalar logistic explanation for small readout v producing large hidden preactivation, and its limits for ELU's normalized upper tail. This plan is committed before the expanded sweeps. The three-point scalar Adam example (v=0.1,1,10; 4000 updates) was already observed in conversation; reproducing it is a replication, not a new prediction. New synthetic controls are mechanistic tests, not replication of RL-MNIST. No source datasets or existing notes are modified.

## Frozen definitions

Scalar baseline: binary symmetric examples x=y=+/-1, score=v*z*x, loss=softplus(-v*z), fixed v>0, z0=0 unless specified. Equal steps and equal loss are distinct comparisons. Gradients are evaluated before updates; metrics after updates. Float64 is primary. Scalar loss is separable and has no finite minimizer.

Distribution metrics: raw maximum, mean, median, population standard deviation s, max/s, (max-median)/s, centered quantiles, positive fraction. max/s is not invariant to translations; (max-median)/s is invariant to positive affine transformations. Degenerate s is flagged. Unit summaries are descriptive; units are not independent statistical samples.

## A. Scalar numerical audit

- Reproduce prior Adam values with an independent PyTorch implementation; verify gradients against finite differences and GD against its analytic gradient-flow limit with step refinement.
- Sweep v on log grids including 1e-4..1e2, durations 100..200000 updates, initial z=-2,0,2, and learning rates 1e-4,1e-3,1e-2. Compute only bounded selected factorial subsets; record the exact executed configurations.
- Controls: GD; Adam default (.9,.999); no first moment (0,.999); no second-moment memory (.9,0); instantaneous normalization (0,0); equal memory (.9,.9); epsilon=0,1e-8,1e-4. Stable residual/softplus computations and underflow flags required.
- Verify exact scalar Adam reparameterization m=v*z, effective learning rate eta*v and epsilon/v. Test equivalent gain/rate runs against one another, not just endpoint correlations.
- Check loss-matched inverse-height relation as a conditional identity, without treating it as a mechanism for the actual network. Report equal-time counterexamples to universal monotonicity.
- Audit long-time behavior; distinguish numerical evidence/asymptotic ansatz from a convergence theorem. Do not infer a finite equilibrium from a finite training horizon.
- Label any follow-up chosen after these outputs as post-hoc and preserve the original sweep.

## B. Shared-input synthetic models

Independent worker owns a bounded matrix using fixed paired seeds 0..4, readout multipliers .1,1,10, shared initial hidden parameters, float64, Adam default lr=.001, and 4000 updates. Datasets must be generated and recorded before the first fit: Gaussian balanced linearly separable, rare-positive linearly separable, and random binary labels; N=128,D=8. Models: linear logistic, one-unit ELU logistic (train hidden w,b and output c, freeze positive scalar v), and 8-unit ELU logistic (fixed signed readout). All use the same finite input sets per seed. A linear-logistic matched-effective-rate control must be included; bias learning rates must be handled explicitly. Optional leaky control and longer horizon are post-hoc if added after results. Record dataset hashes, model parameters, exact initialization, and losses. Report the shape metrics above, separating max/s changes caused by shifts from changes in standardized tail shape.

No claim that a synthetic effect reproduces the RL-MNIST mechanism. Failure to reproduce normalized-tail inflation is an informative result. New noise is not needed for the primary deterministic runs; no claim that noise is unnecessary in the actual network follows.

## Source/theory-informed supplement (before scalar sweeps)

The saved-data audit established that the actual intervention retains Adam moments and raises or lowers an already-existing upper tail. Add scalar warm-start tests: pretrain v=1 for 4000 updates, then multiply v by .1,1,10 and continue 20000 updates with retained, reset, or scale-adjusted moments. The separable scalar model should have strictly increasing z, so it cannot reproduce a falling preactivation after raising v. This is a limitation test, not a claim of equivalence to RL-MNIST.

Add a finite-optimum logistic control with soft target q=.8 (equivalently contradictory binary labels at the same covariate), whose unique margin is log(4). Verify analytically and numerically that a readout gain change can cause both upward and downward movement here. Use GD for convergence checking and default Adam for finite-horizon response; do not assume constant-rate Adam converges exactly. This extension was chosen from source/theory inspection before viewing the expanded outputs, not from fitting their results.

## Post-hoc constructive geometry test

Chosen AFTER the Gaussian shared-input matrix showed that smaller v does not universally increase standardized tail height. Purpose is a sufficiency/counterexample construction, not fitting RL-MNIST or a confirmatory test of its mechanism. Use four symmetric atoms: +/-A e1 with total mass p and +/-e2 with mass 1-p; labels are the sign of the nonzero coordinate. Loss is p*softplus(-v*A*w1)+(1-p)*softplus(-v*w2). w starts0, bias0, fixed v=.1,1,10. Cross A=1,2,5,10 and p=.5,1/16,1/64 with default Adam and instantaneous normalization, eps0, eta=.001,4000 steps. Record analytic population SD and interpolated median0. Derive independent coordinate reduction, verify one PyTorch case with eps1e-8 and correctly weight-adjusted epsilon, and record input-weight direction. A=1 and instantaneous normalization are negative controls for gain-dependent standardized shape. The constructed model cannot reproduce decreasing raw maxima after increasing v because both coordinates grow monotonically; state that limitation. No additional geometry search is authorized by this supplement.

## C. Existing RL-MNIST records

Read only saved vfreeze results and scripts; audit v definitions, optimizer settings, measurement windows, and availability of parameters, errors, and Adam moments. Where raw data permit, independently reproduce the 2-seed summaries. Separate missing measurements from negative results. This is post-hoc validation of already published observations.

## User-steered theoretical identification supplement

After the numerical matrix, the user explicitly clarified that theoretical mechanism identification, rather than reproducing the phenomenon, is the objective. The synthesis must not treat a constructed example as identification in the actual ELU system. Add exact same-state readout response for multiclass ELU/Adam, conditional sign lemmas for centered-tail growth, the fixed-gate saturated-branch scale symmetry, and its violation by exact ELU. Derive what could distinguish optimizer/transient explanations from stationary objective/branch explanations; state when no sign can be predicted. Independently verify the algebra at seeded arbitrary states, without counting this as actual-network evidence. Document up to three discriminating interventions and missing measurements. Do not run additional endpoint-fitting models or claim that these future interventions were performed.

## Actual-system discriminating intervention supplement (registered before runs)

The source MNIST file and original CPU implementation are available locally, so missing saved optimizer states can be reconstructed rather than treated as a permanent obstacle. Extend the user-steered identification work with one bounded actual-system intervention matrix, not another endpoint-matching toy. This is a new same-protocol run, not automatically a bitwise replay of the historical data.

Reconstruct seeds 0 and 1 through task50 using the source streams, initialization, float32 CPU single thread, flush_denormal=True, 784-100-10 ELU, batch16,4000updates/task,torch Adam .001,(.9,.999),eps1e-8. The source backward uses expm1(z)+1 on the negative branch; preserve this rather than silently replacing it by mathematical exp(z). Compare the reconstructed task50 shape/CE to archived values. Save full parameters, optimizer moments/counters and random generator state. A short implementation/throughput pilot is permitted before the registered runs.

From each seed's same task50 state, run only tasks51-53, sharing inputs, labels and batch order. One common c=1 frozen-readout reference plus c=.1 and10 for each of the following:

- A native: freeze W2 after c scaling, retain all moments,epsilon unchanged.
- B aligned: W1/b1 moments scaled c,c²; their epsilon unchanged.
- C aligned and epsilon matched: as B, but W1/b1 epsilon is c*1e-8.
- D second-moment replay: as C, natural forward residuals and natural first moments; the hidden Adam denominator uses c² times the reference's updated second moment from the same step, including matching bias correction.
- E residual replay: as C, but hidden and output-bias backward use the reference's same-batch residual. This has the strict mathematical null prediction of identical hidden trajectories. E is a verification/counterfactual feedback removal, not ordinary minimization of the arm's own loss.

Readout values remain frozen throughout; output-bias moments/epsilon are unscaled in every arm. D's output bias trains normally, E's output bias receives the reference residual. Preserve optimizer step counters. c=1 arms are algebraically identical and may share the reference execution.

Primary observable: same-unit T=(max−lower_median)/sample_sd over all1200inputs for all100units, and paired contrast median_unit(T[c=.1]−T[c=10]) at each task end. Report both seeds individually. Also record raw top, positive fraction,CE and source switch-alive summaries. Checkpoints at task51 steps0,1,10,100,500,1000,2000,4000 and task52/53 ends; save full states at task50/51/53 ends and max/median input IDs at diagnostics. Do not count units/tasks as independent replicates.

Interpretation is conditional on native A producing the target effect in this short window. B−A tests the intervention's old-moment amplitude mismatch; C−B quantifies the epsilon adjustment; D−C tests gain-specific second-moment feedback while retaining natural residual/first-moment feedback. E−C removes residual feedback after amplitude matching and supplies an exact zero prediction. Report raw contrasts and ratios D/C,B/A only when the denominator contrast exceeds .05; no fitted causal percentages or claims that these effects add. No significance threshold or unique-mechanism claim is inferred from attenuation alone. Residual replay differences from floating-point propagation must be distinguished from failure of the mathematical null, using same-state one-step comparisons if necessary.

The theory module predicts the first post-intervention update from saved state with no fitted parameters, including a common-gradient-only counterfactual, previous-moment/current-gradient projections onto the tail direction, and the sign conditions for selective top-input alignment. It must distinguish an exact first-step prediction from explanation of three-task or long-term effects. The historical tasks151-200 effect is not claimed identified by a short-window test.

For that read-only first-state analysis, use the source lower-middle median and sample SD, evaluate per-input projected contributions (current maximum input, open set,closed set) using the realized Adam denominator, and report that deleting an input would also change the denominator. Primary analytic arithmetic may be float64 with the source float32 preactivations/gates held fixed; report this precision convention and mismatch against native float32 updates. A separately labeled derivative-only comparison to exp(z) measures the difference from the source's quantized expm1(z)+1 backward. No coefficients are fitted and no long-term causal claim is made from these first-step projections.

Stop after this registered two-seed,three-task intervention matrix and necessary implementation checks. Do not expand seeds, horizons or geometry based on desired signs. If resources make reconstruction impractical, preserve the pilot and state the limitation instead of substituting a different system.

## Required output and stopping

Reproducible code, exact scalar and synthetic CSVs, checks/provenance JSON, theory derivation, plots, and a Japanese synthesis stating which prior claims survive. Stop after the registered matrices and necessary correctness checks; any additional experiments must target a concrete unresolved conclusion and be labeled post-hoc. No rerun through 200 MNIST tasks or extension to other datasets is part of this audit.

## Repository and environment

Worktree: wt/logistic_v_height_0929; branch codex/logistic_v_height_0929. Cached origin/main=61e5222. Initial git fetch failed because github.com could not resolve in this restricted environment. Do not bypass network or filesystem restrictions. All new artifacts will be tracked, so no worktree-only raw logs will be discarded. Main integration/push follows CLAUDE.md when available; otherwise retain the committed worktree and report the exact limitation.
