# Integrator review of the native 1200-image verifier and constant-rate boundary

2026-10-09. The integration author independently read the complete finite verifier written by the geometry agent, read the separate constant-rate boundary proof, checked its finite-error terms against `adam_task_averaging.md` and `adam_observable_averaging.md`, and executed the repository verifiers. This record does not claim an additional agent review that was interrupted by the usage limit.

Subsequent update: that separate constant-rate/ten-class-formula audit was later completed and is saved in [adam_constant_rate_boundary_review.md](adam_constant_rate_boundary_review.md). Its liminf wording suggestion has been applied. The paragraph above records the earlier integration stage.

## Native verifier: scope and identities

`verify_adam_1200_reference.py` uses the exact native hidden widths and all 120434 raw weights/biases, with a binary head. It avoids an enormous N-by-P Jacobian. The first-Conv mixed minor follows from `G_(i,c,a,b)=sum_r q_(i,r) x_(c,r+offset)` on the fixed branch. The gather map and color mask implement its input derivative, with row ordering matching the packed first-Conv weights and the chosen disjoint pixels. Both strict row and column diagonal dominance imply an invertible minor, with inverse 2-norm at most the inverse geometric mean of those two margins.

The level identity is `beta=M^T w1`. The code compares it with the actual input derivative. Its normalized simplex has zero column sum and kills beta, preserving both the true mean direction and the common contrast level. For `G=delta diag(d)M^T+1 r^T`, `d_i=1/(normalization beta_i)` and `r=g0-delta M d/N`, Sherman--Morrison has denominator `(normalization/delta)Q0`, where `Q0=g0^T w1>0`. The recorded inverse norm bound and positive singular-value lower bound follow. The proof is about gradient-feature rank, not an impossible 1200-dimensional hidden-feature rank.

Every raw sensitivity on this positive fixed branch is a signed nonnegative affine function of the input, with fixed sign for its raw coordinate. For |Delta x|<=delta and x0>=xmin>0, its relative variation is at most delta/xmin. This justifies the all-image maximum-entry and Frobenius bounds computed from one base Jacobian. The output biases are omitted from that small-sensitivity calculation. The finite scale makes the spectral error upper ratio one quarter; the positive normal bound additionally uses the separately proved stationary bias coefficient.

Per-channel convolution Lipschitz bounds, together with the worst pixel displacement, control the full/self pool gaps on every image. A global maximum across channels would be too crude for the small literal-self channels; the code correctly uses each channel's own bound. Positive weights, positive inputs, and positive hidden biases keep all hidden ReLUs active. The literal-self model actually deletes the other first-Conv channels and their corresponding second-Conv inputs, without refitting the head.

The scale masks put first-Conv weights, every hidden bias and head weights at order t, the remaining hidden weights at order t^2, and output biases at order one. Directional Jacobian derivatives vanish for first-Conv weights/hidden biases, are order one for head weights, and order t for intermediate weights. The exact raw derivative check and normalized capacity expression therefore implement `K'=tK1'+t^3K2'`, rather than importing the previous scale powers.

The executed check compares these raw identities on exactly three stated images. Its positive capacity values are for those three images; the identical-image N1200 coefficient is a separate continuity input. Full distinct-image N1200 capacity positivity is supplied analytically by choosing the image width in the continuity neighborhood. The numerical witness at its displayed delta/t is not a complete interval certificate of every full-data capacity condition. The output explicitly retains this distinction.

All non-rational numerical margins are float64 checks. Analytic nonemptiness follows from the positive selector perturbation, rank argument and joint continuity, not from unrounded floating-point values alone. The integrated theorem was separately reviewed in `adam_1200_longtime_review.md`.

## Constant-rate boundary: independent checks

At the first batch of each new task, one raw class-bias gradient is `pbar-L/B`, with L binomial conditional on all prior state and the fresh label-independent batch indices. Two adjacent possible counts give a prediction-independent positive lower probability of a gradient with magnitude at least 1/(2B). Iterated conditioning, followed by a countable union over last-hit times, proves infinitely many such gradients almost surely.

If a bias coordinate converged at constant eta, its increments would vanish. The globally bias-corrected second moment is bounded by one, so the first moment would vanish, and its recursion would force every gradient to vanish. This contradicts the recurrent fresh-task gradient event. The adjacent-update lower bound follows from `|g_t|<= (|m_t|+beta1|m_(t-1)|)/(1-beta1)`.

For any compact full-parameter region, finite softmax logits leave a uniform gap below probability one. A finite block of tasks labeling all images with the same class then forces a strictly one-directional bias movement after a finite moment burn-in. Arbitrarily many disjoint blocks give independent positive-probability opportunities; eventual occurrence has probability one. Taking a countable union over increasing boxes excludes bounded full trajectories. This is a rare-event infinite-time statement, not a practical exit-time estimate, not failure of a hidden-only observable to settle, and not a claim that every late parameter norm diverges.

The finite-horizon error terms in the boundary note match the original tracking bound (including its within-task H), observable Taylor bound, Poisson modulus, and martingale maximal bound. The new integrated theorem uses only a finite prefix and was separately audited with `K=ceil(S/eta)`; it does not carry the permanent-containment assertion into constant-rate infinite time. The stationary-level discussion correctly assumes integrability and does not assign an invariant probability measure to an ever-increasing bias-correction counter.

## Arithmetic and integration provenance

The repository's long-reuse verifier preserves the audited integer/ratio proof, changes its default output location, and additionally checks the inverse-eighth-moment criterion `r<(.999)^300`. The latter was also independently checked in `adam_1200_reference_review.md`. The original source/report hash comparison in `adam_longreuse_bias_review.md` refers to the standalone files at the time of that review; the repository run records a new source hash after these integration changes. No original scientific assertion or polynomial comparison was changed.

`verify_adam_tenclass.py` independently reproduces the displayed Fraction formulas in the two ten-class notes, including the raw epsilon convention, factor 1/(NC), hidden/gauge two-point margins, positive-RGB perturbation margin and fresh-task recurrence constant. Its successful execution does not claim a native ten-class training trajectory or a complete multiclass equilibrium theorem.
