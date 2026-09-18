# orth_reveal_0918: A-task orthogonal increment → B-task exposure → selective counteraction

Registered 2026-09-18 before calculating the new endpoints. User explicitly requests testing this chain, not re-demonstrating post-switch response restoration. The preceding switch_force_0918 protocol did not test this chain.

## Scope and fixed inputs

CondA, one hidden layer 20–100–1, leaky a=.1 and .7, plain SGD lr=.005, unhalved MSE, all W/b/v/c updated together. Reuse the switched Task A trajectories from switch_force_0918 at prior training ages 20,100,500 tasks and all 10 saved seeds. Task A is their 10000-update post-switch task, not the old task before that switch. Use saved start/end weights and exact Task A input support. Both primary float64 and native float32 sensitivity use their own saved A endpoints. These A trajectories were inspected previously; the new decomposition and B outcomes were not. No new historical 5M training.

For each unit, with A-start w0 and A-end w1, define D=w1-w0, alpha=<w0,D>/||w0||², p=alpha*w0, u=D-p. Thus w1=(1+alpha)w0+u and u perpendicular w0. This is a *whole-task* decomposition; it is not a sum of instantaneous perpendicular updates. Treat alpha<0 as erosion of the old direction, but do not assume it occurs. Whole weights are primary. Repeat geometry and tagged-direction diagnostics for row-centered weights as a separately labeled secondary coordinate system. Bias and the row-mean channel are kept distinct.

## A. Is the new component hidden in A and exposed in B?

Enumerate all 15 possible single-bit B task switches from A, with the same five random bits paired across the 32 support patterns. The teacher is fixed, targets on raw inputs and constant-dose offset update as in the original environment. Compute, before B learning:

1. q_A(x)=u·x_A and q_B(x)=u·x_B: mean, positive/negative energy, total RMS over all 32 patterns and units. Also compute the actual full increment D and radial p fields and all cross terms, so a component-wise apparent effect cannot be mistaken for a total effect.
2. Mean-projection change u·(mu_B-mu_A), and its exact addition with the radial term to D·(mu_B-mu_A). Record the signed shift and RMS rather than inferring a negative drift from increased variance.
3. Exact finite functional visibility: phi(z)-phi(z-q), and v[phi(z)-phi(z-q)] at A and B with the same A-end model; these correspond to removing u at fixed other parameters. They are descriptive ablations, not preserving A's function by assumption. Record cells q_B sign × actual initial z_B sign, plus activations created by u: z_B>0 and z_B-q_B<=0.
4. True input-nullspace audit: S_A=span of all 32 A input vectors, SVD cutoff 1e-10 of largest singular value. For D,p,u,w0, record projections into S_A and its nullspace. Check null(D)=null(p)+null(u). In fixed-task plain SGD each nominal update lies in S_A, so D's null component should be only roundoff. A null component of u alone can be canceled by p; never label such a cancellation as newly stored task-invisible weight. Also do the corresponding audit with centered inputs/weights.

Primary visibility endpoint G: for each seed, log(RMS_B(q)/RMS_A(q)), pooling squared projections over units and patterns, and uniformly over all 15 B flips before taking the ratio. Positive means greater raw preactivation visibility in B, not literal invisibility in A. Report the A normalized projection energy and nullspace fractions to distinguish those claims. Functional visibility, mean-only visibility, and centered-coordinate ratios are registered descriptive supplements; do not substitute them for a failed raw-visibility endpoint.

## B. Does B learning preferentially counteract the positive exposure?

Continue one B task for 10000 updates, paired with no-switch (continue A) control from identical A-end W/b/v/c, using identical random five-bit streams. RNG seed20260918: consume the previous A flip draw and its 10000 random-bit draws, then draw B's single flipped bit per seed and B's random-bit stream. Verify against the original SCREnv sampler. All parameters evolve normally. No weights or gradients are removed during these natural trajectories.

Fix the B receiver support and q_B at B start; never reselect receiver groups by a later activation or survival outcome. At times 0,1,10,100,1000,10000 (additional 2,5,20,50,200,500,2000,5000 for curves), record full W/b/v/c endpoints and:

- Tagged Euclidean direction counteraction K=-sum_i<u_i,w_t-w1_i>/sum_i||u_i||². The coefficient of u in w1 is exactly 1; its pooled coefficient becomes 1-K. This tracks a fixed direction in weight space, not an identifiable physical parcel of weights.
- Per-receiver W response delta_s=(w_t-w1)·x_B. For initially positive/negative q_B define C_+=-sum[q_B*delta_s*1(q_B>0)]/sum[q_B²*1(q_B>0)] and C_- analogously. Positive C means response changes oppose the inherited field on that side. Compute W-only and W+b versions; primary is W-only. Also report the four q_B sign × initial full-z_B sign cells and the newly activated receiver subset, with their denominator mass/support. q_B's sign is not the full preactivation's sign.
- Exact per-step attribution of <u,Delta w> to the *sampled unit's current z>0 or z<=0 branch*, plus source event counts. These source sums are distinct from the fixed receiver groups. They determine whether positive-branch updates supply the observed erosion of the tagged direction. Float32 rounding is retained, and actual updates, not only nominal gradients, are attributed.
- A/B mean preactivation and input jump; exact-support loss; full/centered norms. Check that apparent functional suppression is not being silently substituted for a W-direction change.

Primary selective-counteraction endpoint S at10000: (C_+-C_-) in B-switch minus the same quantity in continue-A control, both evaluated on the fixed B receiver support. Positive means B adds preferential opposition to the positive inherited field. The actual C_+ level must also be reported; a positive difference alone does not imply absolute positive-side suppression.

Primary direction-loss endpoint Kdiff at10000: K_B-K_control. Positive means B removes more of the fixed u direction than continued A learning; report K_B itself to distinguish actual removal from reduced growth.

There are 3 primary endpoints × 6 arm/age cells =18 comparisons. Two-sided paired/seed-level Student-t intervals, Bonferroni family alpha=.05 across18, n=10 seeds. Label each endpoint POSITIVE, NEGATIVE, or UNRESOLVED from the adjusted interval. All other summaries use descriptive95% intervals. No pooling units, patterns, or possible flips as independent replicates. If a required denominator vanishes for a seed, retain missingness and mark that primary cell NOT_ESTIMABLE rather than changing groups. Fractions/ratios use pooled numerator and denominator within seed, not a mean of unstable per-unit ratios.

Support for the full proposed chain requires compatible signs of A erosion/orthogonal growth, increased B visibility, and positive-exposure counteraction/direction loss; do not infer the chain from one positive endpoint. An alternative functional-visibility interpretation must be labeled secondary. Neither two tasks nor a source ledger establishes an infinite-time mechanism or causality across architectures/optimizers. In particular, plain SGD input-span restrictions need not hold for coordinate-preconditioned Adam.

## Verification and preservation

Before main execution: synthetic geometry tests with known radial/orthogonal and nullspace components, independent autograd/production update, and original environment/RNG agreement. Negative controls must catch omission of p-u cancellation, confusing mean-zero with all-input-invisible, reversing receiver signs, and relabeling moving gates as fixed receiver groups.

Exact decompositions use float64 tolerance 1e-10*(1+abs(expected)); comparisons involving native float32 production gradients use2e-5*(1+abs(expected)). Nullspace tolerance is reported relative to update norm; float64 nominal-gradient span membership must be within1e-10. Native actual drift outside span is quantified, not silently rounded to zero. Independent uninstrumented replay must match saved parameter arrays bitwise for both precisions. Independently recompute endpoint fields, C±, K, nullspace cancellation and primary intervals from saved arrays. Keep failures and fixes; no threshold relaxation after outcomes.

Own specs/spec_orth_reveal_0918.md, analysis/orth_reveal_0918/, results/orth_reveal_0918/. Back up raw arrays, scripts, logs and parent/checkpoint SHA256 references under ~/Projects/obsidian-research-data/orth_reveal_0918/. Commit/push spec before measurements and code before main. Merge/push main, remove this worktree and branch, and create linked vault result note. Do not alter parent results or other sessions' files. No subagents.
