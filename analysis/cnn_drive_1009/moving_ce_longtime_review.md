# Independent review: moving CNN long-time CE/SGD family

2026-10-09. Read-only mathematical audit of `/tmp/cnn_moving_longtime_family_1009.md`, including the L=2 literal-self NTK calculation and its unbalanced-state extension. Also inspected `/tmp/cnn_moving_ce_kernel_1009.py` for what its checks cover; did not repeat numerical tests. The corrected repository copy is `/home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/moving_ce_longtime.md`.

**Verdict:** no blocking mathematical defect found in the stated moving-SGD theorem or in the displayed full/isolated NTK formulas. Two clarification corrections were requested and their application was directly confirmed in the repository copy: the π-weighted CE expectation in §7 and the requirement that the Adam symmetry observation use uniform u₀ as well as uniform hidden u_l.

## 1. Raw trainable-parameter invariant family

The gradient formula is for the independent raw Conv matrices and full raw head tensor. It does not differentiate a single tied a through all layers. At the initialized outer-product family, every forward channel vector is proportional to u_{l−1} and every backward channel vector to u_l. Positive scalar channel coefficients preserve the same MaxPool routing, so summing the spatial contributions produces the stated pooled data vector x_n. This gives

  ∇W_l=(α/a_l)<B,A> u_l u_{l−1}ᵀ,
  ∇V_{c,j,s}=α A_{c,s}u_{L,j}.

Because each u has unit Euclidean norm, a shared raw SGD learning rate produces the stated a_l and B updates without an extra factor L or a channel-width factor. Balanced initial amplitudes remain equal under SGD. The head remains centered because every CE residual has zero class sum.

The final spatial feature matrix has rank(X); the rank-one restriction is in channel direction. A free C×S class/spatial matrix B can therefore have multiple class contrasts and overlap all images/channels, as claimed. The initialization is still an invariant symmetry manifold rather than an open subset of unrestricted CNN parameter space.

## 2. Predictable learning rate, positivity, and balance

K=√2 max_n||x_n|| bounds every possible minibatch gradient A_t independently of the fresh labels. The adaptive η_t depends only on existing a_t,B_t and the predetermined γ_t, so it is predictable when conditioning before the new minibatch/labels.

The exact discrete identity

  D_{t+1}=D_t−λ_t²[a_t²||A_t||²−<B_t,A_t>²]

has both required bounds: Cauchy–Schwarz makes the bracket nonnegative, while it is at most a_t²K². Hence

  (1−γ_t²)D_t≤D_{t+1}≤D_t.

The separate increment estimate |Δa_t|≤γ_t a_t/2 establishes a_{t+1}>0; this is not being inferred merely from a positive squared quantity. The product lower bound d_*>0 follows from Σγ_t²<∞. It keeps every hidden layer positive and all strict MaxPool winners valid forever, for every label realization.

## 3. Conditional CE drift and head convergence

Fresh labels at every optimizer step are exactly what makes E[e_y|history]=uniform valid, even though a_t and B_t depend on all preceding labels. The pairwise softmax identity yields c(a,B)≥0, strictly positive precisely when the centered head is visible on at least one data vector. Thus a_t is a genuine nonnegative supermartingale and converges almost surely to a finite limit.

The head-norm process with its deterministic remaining Σγ² tail is also a nonnegative supermartingale. Its compensator gives Σγ_t c_t<∞ almost surely. Convergence of a_t and ||B_t||<a_t put each realized trajectory in a compact set bounded away from a=0.

The claimed discrete Barbalat step is valid: bounded state increments imply |c_{t+1}−c_t|≤L_*γ_t on each trajectory. Together with γ_t→0, Σγ_t=∞ and Σγ_t c_t<∞, infinitely many positive-height excursions would cost a fixed positive amount of weighted sum each, a contradiction. Therefore c_t→0, rather than merely liminf c_t=0.

On the pathwise compact set, a positive lower bound for all softmax path probabilities gives c_t≥constant×Σπ_n||B_tx_n||². Hence every visible logit vector vanishes. The image-span orthogonal complement is exactly unchanged by the head update. This proves the full head limit B_t→B₀(I−P_U); it is not an inference from head-norm convergence alone.

## 4. Strict final hidden-mean decrease

Combining the monotone D_t limit and the visible/invisible head decomposition gives

  a_∞²≤a₀²−||B₀P_U||².

When the initial visible head is nonzero, this is strictly below a₀², while the deterministic d_* lower bound gives a_∞>0. Every layer/channel mean is a positive constant times a_t^l, so the strict final decrease is pathwise and includes upstream changes at deeper layers.

The draft correctly does not claim that a_t^l is itself a supermartingale for l>1: its nonlinear discrete update has extra terms. It also permits individual upward updates. The conclusion is strict final net decrease, with one-step conditional nonpositive drift directly proved for the first layer.

## 5. Full and literal isolated NTKs at L=2

The q-direction is the actual gradient of the selected first-Conv mean: its raw first-layer row is μ_Xu₀. At a balanced state,

  v₁(q)=au₁+qμ_Xe_c, t(q)=a+qk, k=μ_Xu₁,c.

I independently recovered all three raw Jacobian Gram contributions:

- Conv1: a²bbᵀ, independent of q;
- Conv2: ||v₁(q)||²bbᵀ;
- head: a²t(q)²Q, Q=(XXᵀ)⊗I_C.

Thus K′_full(0)=2ak[bbᵀ+a²Q] is nonzero PSD. This includes downstream trainable blocks; it does not freeze their q dependence.

Deleting every first-layer channel except c, retaining the corresponding raw Conv2 input column and the same head, and recomputing all Jacobians gives

  K′_self(0)=2ak[bbᵀ+a²u₁,c²Q].

It too is nonzero PSD. A positive ridge resolvent therefore makes both logdet capacity derivatives strictly positive. No head refit or relabeling of CE as “self” is involved.

The unbalanced-state extension also checks: replacing the layer amplitudes by a₁,a₂>0 gives K′_full=2a₁k[bbᵀ+a₂²Q] and K′_self=2a₁k[bbᵀ+a₂²u₁,c²Q], while R=(k/a₁)f. This is a statewise capacity/CE fact, not an Adam update theorem.

## 6. CE zero exception and normalization

For visible centered class contrast, R is a positive scalar multiple of the current logits, so the expected new-uniform CE projection has the same strict downward sign as the negative capacity gradient. At a state with Bx_n=0 for every image, that CE projection is zero while the capacity derivatives remain positive because the trainable-head Jacobian still changes. “Does not reverse the self direction” is valid universally in the family; strict sign equality requires visible class contrast.

For a general image distribution π, the CE formula must be Σπ_n R_n·(p_n−u_C). The previously displayed 1/N formula was only the uniform case. This correction is present in the reviewed repository copy.

## 7. Adam observations: correction and limits

The ancillary statement that ordinary coordinate Adam preserves channel symmetry requires **all** u₀,...,u_L to be uniform, or a scalar input dimension. With a nonuniform RGB u₀, first-layer gradients scale with its input-channel entries, and coordinate-specific normalization generally breaks that direction. The repository copy now explicitly includes u₀ in the uniformity condition.

Even with this symmetry, Adam need not preserve equal a_l across layers, and the first-order term in the balance D update no longer cancels. The document correctly leaves the moving long-time Adam theorem unproved. A step-size cap that preserves positive amplitudes is not sufficient to recover the SGD supermartingale or its final-amplitude bound.

## 8. Numerical-check scope and conclusions that must not be overstated

The inspected script differentiates the original raw two-Conv/MaxPool/head network, verifies its closed-form full and isolated NTKs, and compares actual raw SGD against the recurrence. It does not simulate the infinite horizon; that conclusion comes from the proof above. No extra numerical run was needed for this review.

Restrictions that must remain explicit:

- labels are freshly resampled at every optimizer update;
- the optimizer is the specified predictable scalar-rate SGD;
- convolutions are 1×1 and all biases are absent in the moving theorem;
- initialization obeys the exact channel-symmetry manifold;
- literal full/self capacity sign is proved for the first-layer mean in the two-hidden-Conv calculation;
- positive a_∞ and positive hidden activations mean this construction proves mean decrease, **not ReLU gate death or loss of plasticity**;
- it does not establish the actual two-5×5-Conv/two-FC RL-CIFAR task-reuse/Adam trajectory, nor the stability of arbitrary off-manifold perturbations.

Within those stated conditions, this is a genuine moving, jointly updated, overlapping-channel, multiclass/spatial CNN long-time theorem rather than a frozen-state or head-only calculation.
