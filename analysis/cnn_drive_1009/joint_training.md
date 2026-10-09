# Exact extension from a label-independent reference to finite joint hidden/head training

2026-10-09. Independent derivation. This extends F1 to a neighborhood of genuinely jointly trained deep hidden representations. It does not prove that the ordinary RL-CIFAR CE/Adam trajectory remains in that neighborhood, and it does not identify a label-dependent trained representation with an independent reference.

## 1. Full hidden parameterization and the actual statistic

Let θ collect **all hidden parameters** of an arbitrary finite CNN, including every Conv/FC weight and bias before a final linear head V. H(θ) is its N×P final feature matrix; a trained output bias is a column of ones. Define a scalar statistic m(θ), for example the mean preactivation of one convolutional channel over all images/positions. For a deep channel this depends on its upstream parameters too.

At a differentiable state define

u(θ)=∇_θ m(θ),
R(θ)=DH(θ)[u(θ)].

All pathways affected by u are included in R; its nonzero columns need not belong to only one final feature channel. For first-layer mean preactivation, u is simply the mean augmented raw patch in the target filter's coordinates and zero elsewhere. For a deeper mean, u includes upstream derivatives and must not be truncated to the target filter alone.

Given an old-label-dependent endpoint (θ_Y,V_Y), the new label Y′ is independent and centered. For new square loss ||H(θ)V−Y′||_F²/(2N),

E_{Y′}[∇mᵀ∇_θL_new | Y]
=G_Y:=N⁻¹ tr[V_Yᵀ R_Yᵀ H_Y V_Y].

This identity is exact for any training algorithm that produced the endpoint. For first-layer m, simultaneous full-hidden SGD changes it exactly by −ηG_Y after averaging the new label, because m is affine in its filter parameters. For deeper m the finite-step correction is treated below.

## 2. The independence-safe reference

Choose θ₀ **before drawing the current old labels**, or condition on all earlier training history and take θ₀ measurable with respect to that history. Set H₀=H(θ₀), R₀=R(θ₀), K₀=H₀H₀ᵀ, P₀=(K₀+λI)⁻¹, Q₀=K₀P₀². For the *same* old label Y used in actual joint training, define the comparison head

V₀(Y)=H₀ᵀP₀Y.

Assume E[Y]=0 and E[YYᵀ]=νI, conditional on the prior history. F1 applied to this reference gives

M₀=E_Y[N⁻¹tr(V₀ᵀR₀ᵀH₀V₀)]
=(ν/N)tr(R₀ᵀQ₀H₀).

A positive lower bound M_ref≤M₀ can come from F2. If R₀ has several nonzero final feature columns, bounding each column separately and summing avoids weakening the alignment by unused columns.

**Do not apply isotropy to H_Y or R_Y.** They depend on the old label. Every comparison below couples the actual endpoint and reference using the same Y, takes a pointwise error bound, and only then averages Y. No conditional isotropy given θ_Y is assumed.

## 3. Uniform endpoint perturbation theorem

Suppose for every old label in its support,

||H_Y−H₀||₂≤a,
||R_Y−R₀||₂≤b,
||V_Y−V₀(Y)||_F≤e,
||V₀(Y)||_F≤B.

Write h₀=||H₀||₂, r₀=||R₀||₂ and

D_A=b h₀+r₀ a+ab,
E_grad=[(B+e)²D_A+e(2B+e)r₀h₀]/N.

Then

|E_Y[G_Y]−M₀|≤E_grad,
E_Y[G_Y]≥M_ref−E_grad.

Thus **E_grad<M_ref** proves the actual jointly trained endpoint has the same positive mean-gradient direction as the reference, and hence negative expected first-layer-mean update. These assumptions concern representation displacement, Jacobian/statistic displacement, and head fit error. They do not assume the target trained gradient's sign.

Proof. A_Y=R_YᵀH_Y satisfies ||A_Y−A₀||₂≤D_A. Split

tr(V_YᵀA_YV_Y)−tr(V₀ᵀA₀V₀)
=tr[V_Yᵀ(A_Y−A₀)V_Y]
 +tr[(V_Y−V₀)ᵀA₀V_Y]
 +tr[V₀ᵀA₀(V_Y−V₀)].

The three trace bounds give the displayed error for every Y. Average this inequality. Neither A_Y nor A₀ must be symmetric. □

For finite label support, all a,b,e,B can be uniform maxima over that support. A numerical maximum on sampled labels is not a rigorous uniform certificate for unsampled labels. Deterministic bounds from the next sections do not require enumeration.

## 4. A measurable head-stationarity residual controls e

At the actual representation H=H_Y, let

S_Y=(HᵀH+λI)V_Y−HᵀY,
||S_Y||_F≤s.

This is the gradient of the *sum* old head objective ½||HV−Y||²+λ||V||²/2. If a mean loss is used, rescale λ and the residual consistently. It is computable from the endpoint.

Let E₀(Y)=Y−H₀V₀(Y), with ||E₀(Y)||_F≤E. Then the following bound is valid:

e ≤ s/λ + a E/λ + a B/(2sqrt(λ)).

Proof. The head objective is λ-strongly convex, hence ||V_Y−V*(H,Y)||_F≤s/λ. If ΔH=H−H₀, the normal equations give the exact identity

(HᵀH+λI)(V*(H,Y)−V₀)
=ΔHᵀE₀−HᵀΔH V₀.

The first term is bounded using ||(HᵀH+λI)⁻¹||≤1/λ. The second uses

||(HᵀH+λI)⁻¹Hᵀ||₂
=max_j σ_j(H)/(σ_j(H)²+λ)≤1/(2sqrt(λ)).

Combining yields the result. This is substantially sharper than using 1/λ for both terms. □

For bounded labels ||Y||_F≤Y_max, universally E≤Y_max and B≤Y_max/(2sqrt(λ)). Sharper B,E follow from the reference spectrum or exact finite-support maxima. This yields a completely observable certificate using H,R and old head residuals, without knowing the actual target-gradient sign.

## 5. From a displacement bound to a,b

Work on a closed parameter ball around θ₀ that stays inside a differentiable activation/MaxPool region. Suppose

||DH(θ)||_{2→F}≤L,
||DH(θ)−DH(θ₀)||_{2→F}≤J||θ−θ₀||,
||∇m(θ)−u₀||≤M||θ−θ₀||,
u₀=∇m(θ₀).

For ||θ_Y−θ₀||≤δ,

a≤Lδ,
b≤(J||u₀||+LM)δ.

Indeed R_Y−R₀=(DH_Y−DH₀)u₀+DH_Y(u_Y−u₀). These constants can be bounded using layer norms and activation derivatives or interval bounds on the ball. With finite fixed data, ReLU and MaxPool yield polynomial maps on a fixed region; strict activation/winner margins at θ₀ provide a nonzero radius with finite constants. This argument does not pretend a winner-fixed formula holds through an unverified switch.

For first-layer mean m, M=0. For deeper mean M is a Hessian bound and captures the motion of upstream features in the statistic itself.

## 6. Genuine finite joint GD can satisfy the conditions

This proves nonemptiness for an actual all-hidden/head learning process, not merely for arbitrary nearby endpoints. Let all hidden layers use a common positive learning rate α, the head use positive β, and train the old sum objective jointly for T finite steps:

θ_{t+1}=θ_t−α∇_θ[½||H_tV_t−Y||²],
V_{t+1}=V_t−β[(H_tᵀH_t+λI)V_t−H_tᵀY],
θ_init=θ₀, V_init=0.

In a chosen smooth ball assume ||H||₂≤h_bar and ||DH||≤L. Choose β≤1/(h_bar²+λ). Set ρ=1−βλ<1 and

V_bar=h_bar Y_max/λ,
G=L(h_bar V_bar+Y_max)V_bar.

Head contraction bounds ||V_t||_F≤V_bar. Consequently each hidden gradient has norm at most G and ||θ_t−θ₀||≤tαG. The bootstrap condition TαG≤δ keeps the whole old trajectory in the chosen ball.

The ridge-optimal head map is globally Lipschitz in H on bounded labels with the safe constant

L_* = 5Y_max/(4λ).

This follows from the exact normal-equation perturbation identity in section 4, using E≤Y_max and B≤Y_max/(2sqrt(λ)). Hence, defining e_t*=||V_t−V*(H_t,Y)||_F,

e_{t+1}*≤ρ e_t*+L_* L αG,
e_T*≤ρ^T B+L_*LαG(1−ρ^T)/(βλ).

Comparison back to the fixed reference gives

||V_T−V₀(Y)||_F
≤ρ^T B+L_*LαG[T+(1−ρ^T)/(βλ)].

Insert this e and a=LTαG, b=(J||u₀||+LM)TαG into section 3. For any strict reference margin M_ref>0, first choose finite T large enough that ρ^T B makes the e-only error small. Then choose **α strictly positive** and small enough that the remaining terms and the bootstrap condition hold. All hidden layers are being optimized in every step, and there is no zero-learning-rate limit in the resulting theorem. This supplies an open finite regime with a quantitative certificate.

The theorem is a controlled-small-drift result. A very conservative Lipschitz calculation can require a small α. It does not establish that the ordinary RL-CIFAR learning rate lies inside its regime. Endpoint measurements can be much sharper, as the example below shows.

## 7. Finite new-task step for a deeper mean

For a joint hidden SGD update θ⁺=θ−ηg, if ||∇²m||₂≤M along its step segment,

m(θ⁺)−m(θ)≤−η∇mᵀg+(Mη²/2)||g||².

If E||g||²≤G_new² and section 3 gives γ=M_ref−E_grad>0, then

E[m(θ⁺)−m(θ)]≤−ηγ+(Mη²/2)G_new²<0

whenever η<2γ/(M G_new²). For affine first-layer m, M=0 and the mean-update identity is exact at any step size. Simultaneous head updates do not enter m. A history-dependent Adam preconditioner is not covered by this SGD result.

## 8. Concrete nonempty two-Conv joint-training example

Script: `analysis/cnn_drive_1009/verify_joint.py`; output: `results/cnn_drive_1009/joint_verification.json`.

Use the earlier two-image nonnegative red/green square construction and first 5×5 Conv filters yielding pooled features [[1,.1],[.1,1]]. Insert a second **trainable** 1×1 Conv with matrix [[1,.01],[.01,1]], zero biases, then ReLU and a trained linear head plus output bias. The target statistic is the first Conv channel's mean preactivation. Thus the architecture is Conv5×5→ReLU→MaxPool2→Conv1×1→ReLU→linear-with-bias. This is a two-Conv example, not the exact two-5×5-Conv/two-Pool RL-CIFAR architecture.

For each of the 4 old Rademacher label vectors, jointly update both Conv layers and the head for T=100 steps from zero head, λ=.1, hidden learning rate α=5e−6, head rate β=.2. Both Conv weight tensors change by nonzero amounts for every old-label vector. Some bias gradients cancel by symmetry for opposite labels; no layer is frozen. New labels are averaged analytically using their zero mean, equivalently averaging all four new Rademacher vectors.

Uniform endpoint quantities over all four old labels:

- reference exact expected mean gradient M₀ = 1.815824041870713;
- reference F2 columnwise spectral lower bound M_ref = 1.197735121835798;
- a = .003213543029558494;
- b = .0012730454949164693;
- measured actual head displacement = .0037475153531530588;
- stationarity residual bound s = .0001710292325731782;
- section 4 certified e = .013956995663367104;
- section 3 error E_grad = .6130314740065129;
- certified actual joint expected gradient ≥ **.584703647829285 > 0**;
- directly enumerated actual expectation = **1.8158693565660384 > 0**.

The script differentiates the actual two-Conv graph to compute R=DH[∇m], tracks the actual shared-MaxPool winners, and checks its error inequality. Its floating-point computations verify this numerical witness, while the preceding inequalities constitute the general proof. A formally machine-checked real-arithmetic numerical certificate would additionally need rounding enclosures; do not call float64 results exact rational certificates.

## 9. What this resolves and what remains

Resolved at theorem level: old hidden parameters may be label-dependent and jointly trained at positive finite rates; a measurable, noncircular perturbation certificate preserves the expected self direction, including all other channels and downstream features. The proof does not substitute a frozen H_Y for an independent H₀.

Not resolved by this extension alone: unrestricted hidden motion, long-run invariance of the certificate, CE fitting at convergence, Adam moments/preconditioning, the actual 16-image RL-CIFAR training trajectory, or unconditional preservation of a literal single-unit self term in arbitrary deep architectures. For a deep statistic, the reference direction must include all upstream contributions rather than silently relabeling a partial filter-only derivative as the full mean drift.

### Exact nonemptiness of the deep reference, independent of floating-point witness

The inserted 1×1 Conv gives reference feature columns with diagonal entry d=1001/1000 and off-diagonal o=11/100, plus the output-bias column. Both nonzero R columns are constant over the two images, with values 89/8 and 89/800. The eigenvalues of K₀ are

κ₊=(d+o)²+2=3234321/1000000,
κ₋=(d−o)²=793881/1000000.

Write f(t)=t/(t+1/10)². The normalized off-diagonal magnitude of Q₀ is the rational number

ε=[f(κ₋)−f(κ₊)]/[f(κ₋)+f(κ₊)].

Direct rational arithmetic gives ε<11/20=.55. For each nonzero R column, the squared cosine is the rational number

c²=(d+o)²/[2(d²+o²)]>(39/50)²=.78².

Hence F2 has a strict positive margin using exact rational inequalities. This establishes a nonempty smooth neighborhood and, by section 6, a nonempty finite positive-hidden-learning-rate family even without accepting any float64 endpoint certificate as a formal arithmetic proof. The numerical trajectory above then demonstrates a concrete, readily reproducible member with comfortable measured margin.
