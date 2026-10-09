# Independent audit: N=3 finite verifier and endpoint-cone geometry

2026-10-09. Read-only review of:

- `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/verify_adam_three_images.py`
- its saved `results/cnn_drive_1009/adam_three_images.json`
- `/tmp/cnn_adam_multi_geometry_1009.md`
- their agreement with `/tmp/cnn_adam_three_images_reference_1009.md` and the existing observable-averaging proof.

**Verdict: PASS. No blocking algebraic, architectural, or geometric defect found.** This audit read the script/result and proofs; it did not rerun the numerical computation or edit the repository. The numerical metric witness is correctly distinguished from the stationary Adam metric.

## 1. Raw architecture, mean direction, and literal self

The verifier uses a disclosed smaller native architecture: RGB12-by-12, Conv5 widths 2/2, both pools, FC3/FC3, binary head, all 331 raw weights and biases. This is a finite check of the same identities, not a numerical verification of the analytical 32-by-32/16/16/100/100 construction.

`unfold(images,5,padding=2).mean((0,2))` correctly averages the first-layer augmented patch coefficients over samples and all output positions. The bias coefficient is 1, and only the selected first output channel receives this direction. The observed pointwise image-mean equality implies equality of this actual mean direction; the script checks both separately.

The literal self function deletes only the other Conv1 output channels, their biases, and the corresponding Conv2 input columns. It retains both Conv2 output channels and biases, both hidden FC layers and their biases, and the original head and output bias. There is no head refitting or deletion of unrelated bias blocks. Repacking gives the correct reduced raw architecture and differentiates its remaining parameters independently.

All hidden coefficients/biases are positive by construction. With positive images, the use of pre-ReLU values to check the two pool margins is legitimate. The script explicitly compares full and self winner indices with their matching one-image bases. The JSON reports strict pool gaps and positive full hidden preactivations; positivity of the self hidden activations also follows directly from its retained positive weights, biases, and inputs.

## 2. Three-image construction, ranks, and all-label CE gradients

The input-to-hidden Jacobian selects the red-plane pixels (3,3), (7,7), and (11,11), matching the selected 3-by-3 final spatial cells. Its chosen 3-by-3 derivative minor is nonzero. The two perturbation directions divide by d^T A e_i and subtract the third column, exactly as in the analytical construction. Their common positive normalization preserves their equal-level property and independence.

The three perturbations are v1, v2, and -(v1+v2), so the input average and actual mean direction remain unchanged. Strict routing makes the input-to-hidden map affine; consequently the common head contrast is canceled on all three images by the same retained output bias. The script checks this and the positive input range rather than inferring them solely from small perturbation size.

The saved witness has:

- hidden smallest singular value about 6.27e-4;
- half-contrast Jacobian smallest singular value about 5.24e-4;
- maximum absolute half-contrast about 1.39e-17;
- maximum relative raw-sensitivity perturbation about .00768, below 1/6;
- minimum one-image raw sensitivity about 5.46e-4.

The sample/class stacking is correct: flattening `[N,2]` is sample first, class second; reshaping J to `[N,2,P]` and taking `(J[:,0]-J[:,1])/2` yields the Jacobian of the half-contrast. For labels encoded by torch as 0/1, `1-2*y` correctly represents the signed binary labels. The CE identity is

    grad L = J_z^T [tanh(z)-Y]/3,

with both the half-contrast and sample-average factors correct. The script enumerates all eight label assignments and verifies the raw autograd identity. It also checks the odd-three gradient floor and positivity of every target-mean sensitivity pairing. The saved smallest raw gradient over all assignments is about 1.82e-4, comfortably above the displayed one-image minimum divided by six.

## 3. Full/self NTK stacking and resolvent certificate

`j @ j.T` is the full independent-raw-parameter logit NTK, including common-logit/output-bias directions. Its derivative `jp @ j.T + j @ jp.T` uses the actual first-mean direction. No scalar tied-parameter NTK is substituted.

With sample-first/class-second stacking, the identical-image limit is exactly

    K_rep = ones(3,3) tensor K_one,
    K'_rep = ones(3,3) tensor K'_one.

At ridge lambda=1, its capacity slope is

    (3/2) tr[(I_2+3 K_one)^(-1) K'_one].

The script computes this factor correctly, and the actual slope `.5*trace(solve(I+K,K'))` is the matching six-dimensional expression. The continuity error

    ||Delta K'||_*/2 + ||K'_rep||_* ||Delta K||_op/2

is the correct inverse-resolvent bound for lambda=1. The self certificate uses its own self base, its own reduced Jacobian, and the unchanged retained head/bias. There is no full/self base mismatch.

The saved finite lower margins are about 4.7774 (full) and 4.7548 (literal self), versus respective continuity errors .01877 and .00716. This is a fixed-ridge trace certificate. It does not establish PSD of the perturbed six-by-six K' or a single neighborhood valid for every ridge lambda>0; the proof and verifier do not make either claim.

The displayed numerical certificates are float64 evaluations of analytically valid inequalities, not outward-rounded interval arithmetic. They provide a well-separated finite verification. The mathematical nonemptiness proof is supplied by the analytical continuity construction; numerical output should not be described as an independently rigorous rounding-error certificate.

## 4. Arbitrary positive metric is not identified with stationary Adam

The verifier chooses a diagonal metric with entries from .5 to 1.5 and forms

    K = J D J^T,
    Q = D J^T K^(-1) J,
    v = D u,
    r = J D u.

It verifies Q²=Q and

    u^T Q v = r^T K^(-1) r > 0.

These identities are valid for every positive diagonal D and full-row-rank J. The comments, JSON scope, and interpretation field explicitly state that this chosen metric is not an estimate of the stationary Adam D_*. This distinction is correct.

In particular, the numerical coefficient .190890692 is **not** a numerical lower bound for the actual Adam endpoint-cone coefficient. The theoretical Adam statement uses its actual stationary phase-summed D_*, whose positivity and smoothness are proved separately. The proof can establish nonemptiness for that D_* without measuring it, because the algebra holds for every positive diagonal matrix; numerical cone radii, rates, or failure budgets would require additional estimates of D_* and the associated derivative constants.

## 5. Multi-normal averaged geometry is sound

The factorization Fbar=(1/3)D J^T tanh z uses the mean CE population gradient and the phase-summed stationary coefficient. H is already included in D, so no additional H factor is missing from the ODE.

The use of E(z)=sum log cosh(z_n) is essential and correct. Its dissipation is

    -(1/3) tanh(z)^T J D J^T tanh(z),

which is nonpositive for every z. The proof appropriately avoids the generally unjustified sign assertion for z^T K tanh z. The displayed quadratic local bounds, exponential decay rate, and total path-length bound follow with the stated constants. The finite distance-to-boundary condition gives averaged containment by a first-exit argument.

The C2 endpoint construction does not assume integrability of a distribution of possible normal directions. It uses the endpoint of the one specified smooth averaged vector field. In local coordinates `(x,z)`, the normal variational block is uniformly stable after shrinking the chart; derivatives in purely tangent directions vanish at z=0 and are O(||z||). The resulting exponentially integrable coupling and second-variation forcing justify uniform convergence of the first and second flow derivatives on a smaller compact chart. This supplies the C2 endpoint map needed for the observable theorem.

The exact linearization at the equilibrium is

    Dpi = I - D_* J_*^T (J_* D_* J_*^T)^(-1) J_*.

It is a weighted normal projection, generally not a Euclidean orthogonal projection. The note correctly requires only Dpi·Fbar=0 away from equilibrium, not Dpi·D J^T=0 for every normal direction.

## 6. Finite open mean-sinking cone and stochastic transfer

For v_*=D_*u, the endpoint mean-loss derivative is

    b_* = (J_*D_*u)^T (J_*D_*J_*^T)^(-1)(J_*D_*u)>0.

Positive target sensitivities make J_*D_*u nonzero; this does not assume a future sink. The Taylor cone bounds are consistent: the direction perturbation costs less than b_*/4 and the quadratic term less than b_*t/4, leaving more than b_*t/2. The union of the specified open balls is full-dimensional and gives a uniform positive margin when t is bounded below by t_L>0.

The pair `(pi,z)` is a local coordinate system after choosing coordinates on the regular equilibrium manifold. Applying the existing observable-averaging theorem to pi and E, rather than z², is sufficient. The nonnegative energy drift and convergent remainder force E→0 because the actual power-step sum diverges. Convergence of pi then gives convergence of the full parameter vector. The within-task displacement tends to zero and extends convergence to every optimizer step.

Nested tube margins and H K_Adam delta0<d_* give the same valid all-phase first-exit argument as for N=1. The terminal pi error bounds the affine mean error by ||u||_1 epsilon_pi. The prescribed error budget therefore preserves strict net mean loss at least b_*t_L/4. Ordinary deterministic power steps repeated H times have divergent total rate and square-summable squared rate.

As before, the uniform initial-family interpretation is one common choice of constants/rate with success probability at least 1-alpha for each initial point in the admitted family. A common label-history event for all uncountably many initial states is not established or needed.

## 7. Scope and integration verdict

The analytical reference, finite raw verifier, positive stationary Adam metric, vector-normal endpoint geometry, and observable averaging fit together. A native three-image, binary, full-batch H-reuse, all-bias open family is supported under the explicitly small decaying rates and fixed local routing conditions. The proof retains full channel overlap, independently varying raw parameters, and the original capacity-self definition.

The result remains distinct from many-class/general-minibatch RL-CIFAR, constant-rate convergence, gate death, or a measured long-time probability for the numerical witness. No additional defect was found within the stated scope.
