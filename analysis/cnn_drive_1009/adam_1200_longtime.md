# Native 1200-image, batch-16, 400-epoch Adam: conditional self-direction theorems

2026-10-09. This joins the native rank-1200 construction to the exact long-reuse bias certificate. It reaches the original image count, batch size, epoch count, convolutional/FC widths, full biases, and retained Adam history. The images and initial state are specially constructed, labels remain binary, and the infinite-time result uses sufficiently small decaying rates. A separate finite-horizon result uses a single positive constant rate. Neither is a theorem for the observed ten-class, rate-.001 RL-CIFAR run.

## 1. Infinite-time statement and exact task law

Fix N=1200, B=16, E=400, H=30000, beta1=.9, beta2=.999, epsilon=1e-8, a prescribed ridge lambda>0 and 1/2<p<=1. Use the native RGB32x32 / Conv5(16) / Pool2 / Conv5(16) / Pool2 / FC100 / FC100 / binary-head architecture, with ReLU at all four hidden sites and every ordinary bias.

There exist 1200 distinct positive images and an open subset of the complete independent raw-parameter space such that, for every alpha in (0,1), a sufficiently small eta0>0 gives the following per-initial-point probability guarantee at least 1-alpha on a fixed smaller initial slice. Each task assigns fresh independent uniform binary labels to all images; each of its 400 epochs independently shuffles the full dataset and regroups it into 75 batches of sixteen. Every batch uses mean CE. Ordinary Adam starts with zero moments once, keeps all moments and the global bias-correction count between tasks, and uses

    eta_(k,r)=eta0/(k+1)^p, r=1,...,30000.

On the guaranteed event every raw parameter converges to a finite limit, every binary logit contrast tends to zero, and the actual data/spatial mean of a selected first-Conv preactivation has a uniform strict negative net displacement. The original full and literal-self raw-NTK capacities have positive mean-direction derivatives for the prescribed ridge throughout the path and at its limit. The ReLU/MaxPool branches and all local stability/regularity conditions are retained at every phase. The step sizes have divergent sum and convergent squared sum.

This is a high-probability terminal offset, not update-by-update monotonicity, a persistent negative asymptotic velocity, a negative final mean, or gate death. The mean stays positive in this construction. The shared constant/rate bounds are uniform on the chosen initial slice, but no event controlling every uncountable initial state simultaneously is asserted. Nothing is tied, frozen, projected or refit during training.

## 2. Nonempty reference at the full native image count

The [rank-1200 reference](adam_1200_reference.md) uses the first-Conv weight sensitivities as features. Sixteen separated 5x5 RGB patches provide 1200 independent input pixels; positive near-selector downstream weights make their derivative map to the 1200 first-Conv raw gradients invertible. A mean-preserving simplex within one common contrast level gives 1200 distinct images whose gradient-feature matrix has rank 1200, even though the last hidden feature matrix has only 100 columns.

Scale first-Conv weights, every hidden bias, and both antisymmetric raw head rows by t, and scale the canceling output-bias contrast by t^2. Other hidden weights are unchanged. All hidden activations scale by t. The non-output-bias contrast Jacobian then has the exact raw decomposition

    R=[t J1,t^2 J2],  lambda1=lambda_min(J1 J1^T)>0.

The leading block includes the full-rank first-Conv block. Every raw sensitivity is nonzero and has arbitrarily small relative variation across the nearby images. All ordinary output biases remain trainable, with sensitivities +/-1/2.

For gamma_j=max_n|J_nj|, the general batch response bound is

    |L_jn-c0 J_nj|<=c0 C_B(gamma_j/epsilon)|J_nj|,
    c0=H/(N epsilon), C_B=1+sqrt(75),

where L is the exact phase-summed stationary logit response at uniform predictions. Let Gamma bound entries of J1,J2, and C^2=||J1||_F^2+||J2||_F^2. A nonnegative output-bias common block and

    0<t<min(1, epsilon lambda1/(2 C_B Gamma C^2))              (2.1)

imply Sym(JL)>=c0 t^2 lambda1 I/2>0. This compares the errors with the actual smallest rank-1200 mode. The constants can be small/large but are finite at one fixed nonzero image separation.

The capacity powers here are `K=K_bias+t^2K1+t^4K2` and `D_uK=tK1'+t^3K2'`. At coincident images the leading capacity derivative divided by t is strictly positive, supplied by the free head-weight block. Joint continuity in image separation and t permits the same positive finite reference to meet (2.1) and both original full/self capacity signs. Small independent raw perturbations, including a nonzero common head, then give a full-dimensional initial neighborhood.

## 3. The long-reuse bias and regularity obligations are supplied

The [long-reuse bias proof](adam_longreuse_bias.md) gives, for every stationary phase of the actual N1200/B16/E400 task,

    Gamma_phase>209/100,
    J_bias L_bias=beta_b 11^T, beta_b>209/4>0.                 (3.1)

Its covariance bound includes the reused-label population term 1/N. Its low-RMS bound conditions on all relevant task labels and uses thirteen completed, independently reshuffled epochs. The conditional generating function is bounded by checking all 1201 possible task label counts with exact integer polynomial arithmetic. No independence between reused-label gradients is substituted. The [saved certificate](../../results/cnn_drive_1009/adam_longreuse_bias.json) is regenerated by [the standard-library verifier](verify_adam_longreuse_bias.py).

For expectation regularity, set

    p_N=binom(1200,600)/2^1200,
    r=binom(16,8)^75/binom(1200,600).

All batches in an entire task fail the sufficient unequal-label-count test with probability p_N r^400. The same verifier checks `r<(.999)^300`, so

    p_N r^400 beta2^(-H*8/2)<1.

Thus the [whole-task inverse-RMS argument](adam_b16_regularity.md) supplies a uniform inverse-eighth moment for every raw coordinate on a compact product neighborhood of outputs and sensitivities. Its Hilbert-norm quotient derivatives need only the inverse-second moment to establish C3 of the stationary expected field. Balanced batches and finite-history zero second moments are included. A deterministic gradient floor is not assumed.

## 4. Endpoint and moving-process proof

The rest of [the batch16 integration](adam_b16_longtime.md), sections 5-6, applies in arbitrary finite N once its reference, bias and regularity inputs are replaced by sections 2-3 here. Binary label complementation gives Rbar(0,S)=0. Hence

    Fbar(theta)=V(theta)z(theta), V in C2,
    Sym(JV)>=aI>0

on a fixed compact neighborhood. The averaged dynamics contract z, have integrable full-parameter speed, and admit the C2 endpoint pi proved by the nonsymmetric variational argument. At the reference,

    Dpi=I-Q, Q=L(JL)^(-1)J.

For the true affine first-filter mean m=u^Ttheta, the supported sensitivities are positive and the entrywise error reserve from (2.1) gives L^T u>0. The current-reference ray v=LL^T u has Qv=v and u^TQv=||L^Tu||^2>0. A finite C2 Taylor estimate produces an open raw cone with

    m(theta0)-m(pi(theta0))>=Delta>0.

The general phase-dependent [task averaging](adam_task_averaging.md) and [observable averaging](adam_observable_averaging.md) are applied to the ambient components of pi and to ||z||^2. Their whole-task innovation includes all 1200 labels and all 400 permutations. Retained moments, global bias correction and changing parameters are included in the convergent error. The constants explicitly depend on H; no H-independent small-step threshold is claimed.

With an ambient coordinate-selection chart (T pi,z), ||T||_infinity=1, nested compact tubes and the additional intermediate-step margin H*73*eta0<d_*, the observable error budgets prove non-exit by induction. Reduced identities Dpi Fbar=0 and nonnegative output-energy drift are used only at previous boundaries inside the outer tube. Their globally extended auxiliary identities retain general drift terms. This avoids assuming the future containment to prove itself.

Convergent output-energy remainder and finite nonnegative dissipation force z->0 because the step-size sum diverges. Each pi coordinate converges; the chart gives convergence of all raw coordinates. Finally, an endpoint error budget below Delta/(2||u||_1) yields

    m(theta0)-m(theta_infinity)>Delta/2.

The original capacity-self direction is positive in the retained region, so this strict net mean decrease has the asserted self direction. It does not turn CE into the capacity functional or prove monotonic decrease of the capacity value during arbitrary full-parameter updates.

## 5. A genuinely constant rate on a finite certified horizon

The same nonempty native binary reference gives a separate constant-rate theorem. For each prescribed failure probability and the fixed initial slice above, there exist a finite task-clock horizon S>0 and a sufficiently small positive constant eta such that K=ceil(S/eta) tasks, using that same eta at every one of their H updates, stay in the certified region and satisfy

    m(theta0)-m(theta_K)>Delta/2

with the prescribed high probability. Original full/self capacity direction and strict branches persist through that finite horizon. This assertion concerns one constant rate throughout the run; no decay is hidden in it.

Here is the finite-error argument. For K tasks, the existing tracking/Poisson/observable decomposition has error terms of the form

    C1 eta+C2 K eta^2+C3 K eta omega(H*73*eta)
                       +C4 eta sqrt(K/alpha_i),

where omega(r)=O(r log(1/r)) and the finite constants depend on the fixed local geometry, H and observable. Summing the finite initialization/geometric tails produces C1. For fixed S and K=ceil(S/eta), every term tends to zero as eta decreases. The same simultaneous error budgets and intermediate-step margin prove containment through K, without an infinite-time containment claim.

More explicitly, write E=||z||^2 and take 2a eta<=1. A cumulative energy-error bound sup|R_E|<=e_E gives by summation by parts

    E_K<=(1-2a eta)^K E_0+2e_E.

On the compact chart, |m(theta)-m(pi(theta))|<=C_m sqrt(E(theta)) for a finite C_m. Choose S large enough that `C_m exp(-aS) sqrt(E_0)` is uniformly below Delta/8 on the initial slice, then choose positive error budgets with `||u||_1 e_pi<Delta/8` and `C_m sqrt(2e_E)<Delta/8`. A sufficiently small constant eta meets these budgets and gives the asserted margin, using sqrt(x+y)<=sqrt(x)+sqrt(y).

This does not assert that eta=.001 works, that K equals the experiment's 50 tasks, or that one fixed eta works for every increasing horizon. The [constant-rate boundary theorem](adam_constant_rate_boundary.md) shows that all-raw finite convergence and permanent containment in a fixed compact tube actually fail at constant eta with infinitely many fresh-label tasks. Finite net sinking is a compatible and distinct statement.

## 6. What remains unresolved

The two results above cover binary specially constructed images at the original N/B/E/hidden widths. They preserve full raw freedom, ordinary task reuse and retained moments. The [ten-class equilibrium obstruction](adam_tenclass_equilibrium.md) shows why replacing two output classes by ten is not a formal substitution: uniform CE predictions can have a nonzero stationary hidden Adam field, even after removing the common-logit gauge. Ten-class predictive geometry and actual rate-.001 trajectories remain open. The constant-rate finite-horizon result must not be reported as permanent convergence or as an empirical success-rate measurement.
