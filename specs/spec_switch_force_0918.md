# Task-switch force and mean-input transport — switch_force_0918

Registered 2026-09-18 before new switch evaluations or continuations. This is a prospective protocol on previously studied checkpoints, not a previously unseen-data confirmation. Earlier conversation inspected frozen no-switch gradients and a post-hoc one-task eta ladder. No new results from the present protocol have been inspected.

## Question and scope

Does a CondA task switch recreate a downward weight displacement along the new task's mean input? What produces the force, does it persist to task end, and is its mean-response effect carried by radial or tangential weight updates? Distinguish changing inputs from learning and bias from weights. An algebraic decomposition is not a causal removal intervention.

Use existing zero_attraction_0913 checkpoints: LR_a0p1_q0 and LR_a0p7_q0; steps 200000, 1000000, 5000000 (20, 100, 500 completed tasks); all 10 saved seeds, 100 units. No initialization in the primary analysis. Plain SGD lr=0.005, unhalved scalar MSE, W/b/v/output-bias updated simultaneously, no decay/momentum/projection. CondA changes one of 15 fixed bits; five random bits have exactly 32 equally likely patterns. Teacher stays fixed; its targets on the changed raw inputs can change. The constant-dose input offset is recomputed as in the original code.

## A. Exhaustive switch at fixed parameters

Enumerate every one of the 15 possible bit flips, for every checkpoint and seed. Evaluate all 32 random-bit patterns before any learning. For each paired pattern evaluate (old input, old target), (new input, old target), (old input, new target), (new input, new target). The crossed combinations are diagnostic counterfactuals, not natural tasks. Use the same new mean input as the projection reference across all four combinations. Report the full interaction, and symmetric Shapley allocation to input and target changes; do not attribute an order-dependent sequential difference as unique causality.

Record per unit: instantaneous mean-response drift in W and b; h=2(f-y)v phi'; h_self=2v^2 phi phi'; h_rest=h-h_self; both split by z>0 and z<=0. Also decompose h_total change at the paired pattern level:

    h_new-h_old = 2v [(e_new-e_old) k_old + e_old(k_new-k_old)
                         + (e_new-e_old)(k_new-k_old)].

The analogous response-force decomposition must additionally include the change of the input vector, not just h. Changing targets with x fixed must leave h_self identical. Independently check autograd gradients and frozen actual SGD displacements. Enumerated patterns/flips/units are not independent replicates: aggregate them within seed for summaries.

## B. Paired actual one-task continuations

For each checkpoint, choose one uniform bit flip per saved seed using a fresh torch CPU generator seeded 20260918, matching the preceding post-hoc ladder's flip convention. Generate T=10000 random-bit inputs with the same generator after the flip draw. Run switch and no-switch copies from identical parameters with the exact same random-bit stream, in float64 (primary arithmetic) and native float32 (numerical sensitivity). These are new continuations, not original RNG replay. No additional task change inside the window. Precompute teacher targets for the 32 patterns using the original float32 threshold teacher.

Save windows 1, 10, 100, 1000, 10000 updates and time-course checkpoints including 0, 2, 5, 20, 50, 200, 500, 2000, 5000. At each window report W and b endpoints, exact-support loss, mean response, norm, cosine to fixed new mean, per-unit and seed summaries. No-switch trajectories must also be evaluated against the switch trajectory's fixed new mean (primary paired comparison), and their own mean (descriptive).

Per actual step, decompose W transport relative to fixed reference mu_new:

    delta_s = Delta w dot mu_new;
    radial = ((w dot Delta w)/||w||^2) (w dot mu_new);
    tangent = (Delta w-radial_vector) dot mu_new.

Use each step's pre-update w, not task-start w. The tangent contribution to a linear projection is exact; do not call it an exact finite-angle rotation. Also report the exact endpoint separation s=r c, c=unit(w) dot mu_new, using symmetric allocation length=Delta r*(c0+c1)/2 and direction=Delta c*(r0+r1)/2.

Keep full W, row-centered W, and row-mean channel distinct. Audit centered-norm identity -2 eta A+eta^2 Q against actual centered weights. For the linear mean-response ledger use h_self/h_rest, positive/negative branch, and input decomposition:

    delta_s = -eta h ||mu_new||^2 - eta h ((x-mu_new) dot mu_new).

The second term need not average to zero; for no-switch controls x-mu_new is not zero-mean. Bias has delta_b=-eta h. Record floating-point update residuals separately and close all ledgers against actual stored parameter changes. Record positive/negative *transport* sums (downward/upward motion), not only branches; branch membership does not fix the sign of h. Record signed cosine change, fractions of units moving downward, and radial/tangent contributions; do not pool units as independent observations.

Input-only boundary jump is J_input=w_start dot (mu_new-mu_old), before any weight update. At task end:

    z_end(new)-z_start(old) = J_input + (w_end-w_start) dot mu_new + Delta b.

## Primary endpoints and inference

Primary per-seed endpoint: unit-mean [(w_switch,t-w0) dot mu_new - (w_control,t-w0) dot mu_new], at t=100 and 10000, for six arm/checkpoint cells (12 comparisons). Use two-sided paired Student t intervals with Bonferroni family alpha=.05/12, n=10 seeds. Labels: SWITCH_ADDS_DOWNWARD if upper bound<0; SWITCH_ADDS_UPWARD if lower bound>0; otherwise UNRESOLVED. Report the actual switched displacement too: a negative paired effect need not mean absolute sinking. Secondary estimates use descriptive 95% t intervals and are labeled exploratory; no post-hoc selection of a favorable window.

An early downward effect with no resolved late effect is not persistent sinking evidence. Even a late effect is evidence for one task only, not unbounded long-run drift. Do not assert self dominance merely because self has a large magnitude: rest may cancel it. Do not divide by tiny net effects to infer percentages. Report signed self/rest levels and changes. If float32 and float64 differ materially, retain both; precision agreement is not presumed.

## Verification before interpretation

1. Original source/activation/update equivalence and constant-dose offset; teacher threshold output matches direct original teacher expression; original SCREnv one-bit operation matches chosen flips and random bits.
2. Independent autograd: float64 gradients tolerance 1e-10*(1+abs(expected)); native float32 same forward order tolerance 2e-5*(1+abs(expected)).
3. All algebraic h/source/branch and instantaneous force decompositions close; wrong MSE factor, omitted bias, unadjusted mu boundary, omitted rest, and omitted tangent are rejected when applicable.
4. Float64 actual one-step norm and transport errors are bounded by 64*eps*(1+old magnitude+new magnitude+absolute contributions); sum these bounds over trajectories. Float32 roundoff is retained explicitly, with the same dtype-scaled bounds. Closure of the *actual* update decomposition should be double-precision accurate.
5. Positive/negative branches sum to total; self+rest sums to total; radial+tangent sums to actual transport; boundary+learning equals endpoint z difference; length+direction equals endpoint projection difference.
6. Frozen target-only counterfactual leaves self unchanged; input/target/interaction allocation closes.
7. Use saved endpoints/raw sums for an independent report-time recomputation. Preserve failed checks and diagnostic changes, do not loosen thresholds after viewing results.

## Outputs and resources

Own only analysis/switch_force_0918/, specs/spec_switch_force_0918.md, results/switch_force_0918/. One CPU process with one torch thread, cases may run sequentially; no new long-term training. Raw outputs/scripts/logs are backed up under ~/Projects/obsidian-research-data/switch_force_0918/ with SHA256 manifest. Save checkpoint hashes, code/spec git hashes, environment, random draws, finite checks, full unrounded numbers, summary CSV/Markdown, plot PNG/PDF. Commit and push spec before execution, then code before primary runs; preserve commits. Complete on main, push, remove this worktree/branch, and add a linked vault result note. Do not alter other sessions' files.
