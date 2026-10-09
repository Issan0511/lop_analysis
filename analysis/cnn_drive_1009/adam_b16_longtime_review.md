# Independent audit of the integrated batch-16 long-time theorem

2026-10-09. Read-only audit of `analysis/cnn_drive_1009/adam_b16_longtime.md` and its links to `adam_b16_reference.md`, `adam_b16_bias.md`, `adam_b16_regularity.md`, `adam_singleton_geometry.md`, `adam_task_averaging.md`, and `adam_observable_averaging.md`. No repository edits or numerical reruns were performed.

**Verdict: PASS after two explicit localization clarifications, now present in section 6.** No unresolved mathematical gap was found in the stated local existence theorem. Numerical verifier execution and result integration remain the parent's responsibility; this audit checks the mathematical argument and document references.

## 1. Exact scope and optimizer conventions

The statement correctly fixes binary outputs, N=32 or N=48, B=16, E=2, H=4 or H=6, beta1=.9, beta2=.999, epsilon=1e-8, prescribed finite ridge lambda>0, and a power-decaying deterministic rate with exponent in (1/2,1]. Whole-task innovations contain all image labels and both fresh epoch permutations. Labels are reused within the task and moments/global bias correction carry across tasks. The mean loss normalization and the phase-summed stationary field agree throughout.

The open initial family is in the entire independent raw parameter space, including all biases. Layer scaling specifies only reference initial values; it is not maintained as a constraint during Adam. The probability guarantee is per initial point with uniform constants on a smaller slice, not a single event for uncountably many initial states. No unconditional expectation, gate death, practical rate, or arbitrary ten-class RL-CIFAR claim is made.

## 2. The two batch-specific obstacles are separately resolved

The nonbias linearization uses the general bound with

    c0=H/(N epsilon), C_B=1+sqrt(N/B).

Both the batch-mean 1/B factor and the sample-inclusion B/N factor are present. A zero-RMS history has the first derivative Dm/epsilon with a quadratic residual; the proof does not promote that pointwise statement to C3.

The output biases have sensitivities +/-1/2 and are treated separately by the exact normalized response with 2epsilon. The companion proof establishes Gamma_r>29/10 for every stationary phase at both admitted task lengths, preserving all within-task correlations and the partial current task. Exchangeability gives the correct common-mode contribution

    J_bias L_bias=[sum_r Gamma_r/N] 11^T.

The regularity proof then supplies uniform inverse-RMS moments over a neighborhood of independent (z,S), despite balanced batches. Its sufficient relative-sensitivity condition is compatible with the reference images and finite layer scale. The common good-batch event works simultaneously for every raw coordinate. C3 of the expected field follows from integrable bounds on derivatives through order three; no uniform deterministic RMS floor is asserted. The optional arbitrary-finite-E inverse-moment statement is correctly not used to extend the separate H=4/6 bias-sign certificate.

## 3. Nonempty full-rank native construction and capacity compatibility

The positive near-selector architecture retains at least N independent selected spatial input directions through the two pools and 100-unit FC layers. The level-simplex image construction has `T beta=0`, zero mean perturbation, and rank N-1. Its feature matrix has row rank N because the common positive head level contributes the missing direction. Thus arbitrarily close but distinct admissible image sets exist without asserting a nonvanishing singular-value bound as images coincide.

The raw sensitivity blocks split as `[tJ1,t^2J2]`. The free head columns make J1 full row rank, and the relative error is explicitly compared with its smallest row singular value. The finite positive scale interval in (4.1) gives both a normal spectral margin and an entrywise response reserve below 1/2. The unscaled output-bias block is not subjected to the small-sensitivity estimate.

The original full and literal-self capacity derivatives include all retained raw parameters and biases. Their normalized slopes have a continuous coefficient extension at zero scale for fixed positive ridge. At coincident images the leading free-head contribution is strictly positive, while the last-FC contribution is nonnegative. Joint continuity permits choosing the nonzero image separation first and then the positive layer scale. No differentiability of a physically zero ReLU layer, fixed gap at coincident images, or uniform neighborhood for ridge approaching zero is used.

## 4. Smooth endpoint and derived positive open cone

C3 of Rbar in independent (z,S), together with `Rbar(0,S)=0`, makes the Hadamard coefficient V C2. At the finite reference V=L, and the positive symmetric part of JL extends to JV in a small compact neighborhood. These are exactly the hypotheses of the nonsymmetric endpoint argument; a diagonal full-batch preconditioner identity is unnecessary.

The variational proof in `adam_singleton_geometry.md` section 6 is dimension-independent: replace its normal vector by R^N and the tangent chart by dimension P-N. Its estimates use operator bounds and a positive symmetric part, not N=3 or singleton-specific gradients. Section 5 supplies the general Hadamard/contraction step after its singleton-only smoothness derivation is replaced by the present regularity proof.

The derivative of the endpoint is `I-Q`, with `Q=L(JL)^(-1)J`. The ray `v=LL^T u` satisfies Qv=v and gives the strictly positive mean derivative `||L^T u||^2`. A finite C2 remainder therefore produces an open cone with a uniform endpoint-loss margin on a smaller slice. This is a consequence of the current reference matrices and derivative bounds, not an assumed sign of future motion.

## 5. Averaging, localization, convergence, and final mean loss

The applicable parts of `adam_task_averaging.md` are its general phase-dependent results in sections 1-6 and localization in section 8. Its separate full-batch sign lemma in section 7 is not used. On the local fixed branch, all batch gradients are uniformly bounded and Lipschitz, and h=1. The norm-based RMS tracking remains valid at zero finite-history second moment.

The applicable observable theorem is `adam_observable_averaging.md` sections 1-5. It handles every ambient coordinate of pi and the scalar energy `||z||^2`, with a separate Poisson reward for each observable, finite Taylor bounds, convergent remainders, and a simultaneous probability budget. It does not require the optimizer update itself to be differentiable at RMS zero. Its later one-output application is not being substituted for the present N-dimensional geometry.

Two clarifications requested during this audit are now explicitly present in integrated section 6:

1. T is an ambient coordinate-selection matrix with P-N rows and infinity operator norm one. Such a selection exists because pi restricted to the equilibrium manifold has rank P-N. The budget `e_pi<rho1-rho0` therefore correctly controls the selected endpoint coordinates.
2. Reduced identities using `Dpi Fbar=0` and nonnegative energy drift are used only while the preceding boundaries remain inside the outer tube. The globally extended auxiliary identity retains its general drift term. This makes the induction noncircular.

The bootstrap is valid: a boundary in the middle tube remains inside the outer tube during all H following updates because `H*73*eta0<d_*`; the local cumulative identities then put the next boundary back in the middle tube. Consequently all optimizer phases remain within the strict branch, rank, regularity, stability, and capacity neighborhood on the same event.

The nonnegative dissipation sum is finite and the energy remainder converges, so the output energy has a limit. Its lower drift bound `2a E_z` and the divergent sum of delta_k force that limit to zero. Every pi component converges, and the local chart `(Tpi,z)` then gives convergence of all raw parameters. Intermediate-phase displacement vanishes. Since pi is the identity at the limit, the final mean loss is at least `Delta-||u||_1 e_pi>Delta/2`. Deterministic h=1 rates directly have divergent total mass and square-summable squares.

## 6. References and limitations of this review

The cited reference-proof sections 1-2, 3-5, and 7 contain the indicated estimates and constructions. The nonsymmetric endpoint is in `adam_singleton_geometry.md` sections 5-6, and the finite cone inequalities are correctly cited to `adam_singleton_longtime.md` section 5. All document, verifier, and saved certificate links in the integrated theorem resolved at review time.

The finite native witness is explicitly distinguished from the full-size analytic construction, a stationary estimator, and an infinite-time probability certificate. I did not rerun or independently certify its floating-point margins. The theorem remains an existence result with specially selected nearby images, a possibly very small finite layer scale, short reused-label tasks, and sufficiently small decaying rates. Within these stated restrictions, the companion results supply the required hypotheses and the integrated conclusion follows.
