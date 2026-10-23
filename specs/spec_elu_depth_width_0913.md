# ELU depth x width x activation factorial (elu_depth_width_0913)

Status: preregistration draft, 2026-09-13. User authorized experiment B.
Question: Does increasing first-layer centered weight norm harm ELU less than leaky
when actual per-unit preactivation means are controlled, and does this width effect
depend on depth? This is a controlled intervention, not a natural-mediation estimate.

## Scope and frozen design
Permuted MNIST, frozen pmnist_boundary_host_0908: 784-100-100-10, cross-entropy,
Adam lr .001, betas (.9,.999), eps 1e-8, batch 16, 10,000 stratified training examples
and 625 updates/task; 120 tasks; no weight decay; CPU float32, seeds 0,1,2.
Activations ELU alpha=1 and leaky slope=.1, in both hidden layers.
Each activation/seed trains an untreated 20-task prefix ONCE; all five continuations
restore exactly the same parameters, Adam states, and permutation/data/batch streams.
The cells are norm 5 or 10 crossed with mean -1 (shallow) or -4 (deep), plus untreated
reference. Six independent activation/seed workers, each one CPU thread, sequential
five-cell continuations. No additional seeds, target tuning, or budget extension after outcomes.

## Intervention
Write each first-layer row W_i = m_i 1 + Wtilde_i. In all four factorial cells,
freeze m_i to its own task-20 value and impose ||Wtilde_i||_2 = 5 or 10.
At each new task compute xbar from that task's actual 10,000 training examples AFTER
permutation (batch ordering does not change the mean). Set
b_i = depth_target - W_i dot xbar, independently for every unit.
Apply both projections before the first evaluation/update of each new task and after
EVERY Adam update. Adam states are not projected or reset; remaining parameters untouched.
Consequently actual per-unit training-input E[z_i] is -1 or -4 regardless of W.
This bias intervention is distinct from old dclamp, which controlled mean row-weight;
rowmean freezing suppresses brightness-channel evolution and changes the causal regime.
It also equalizes mean depth across units; results do not estimate effects of natural
heterogeneous depth distributions. Test-probe means can differ and are reported.
Fixed mean does NOT mean fixed saturation: W can change tails, gates and representation.
The W contrast will be called a width effect at fixed mean, never a non-saturation effect.

## Measurements
For every continuation/task evaluate before projection, immediately after projection
(step 0), and after steps 20, 100, 300, 625 on fixed seed-specific independent probes:
2,048 held-out test examples and 2,048 training examples chosen separately from the
task draw stream (training-set probe is descriptive, not held-out generalization).
Save test accuracy and test CE, training-probe accuracy and CE. Additionally report
full task-dataset train CE at step 625. Save per-task minibatch mean pre-update CE.
First-layer state diagnostics: actual per-unit mean on all training examples (computed
analytically from xbar and independently checked on explicit activations at selected
measurements), probe z mean/std, positive fraction, gate mean, gate<.05 fraction,
unit maximum gate<1e-6, Wtilde norm, row mean, bias, W2 column norm.
Save per-unit arrays at step 0 and 625 and parameter/RNG snapshots at task20 and task120.

## Registered endpoints and contrasts
Early window tasks21-40, late101-120, all-continuation21-120; task-boundary single-point
projection shock is reported separately. Primary endpoints:
(1) late task625 test accuracy (percentage points);
(2) degradation L = early task625 accuracy minus late task625 accuracy (points).
Secondary: same early/late windows for test CE and train CE; adaptation AUC of test CE
over steps0,20,100,300,625 using trapezoidal integral/625; per-task gain
testCE(step0)-testCE(step625), and online minibatch CE. Report absolute levels alongside L.
For each activation, depth, seed: W_harm = accuracy(norm5)-accuracy(norm10).
Depth_harm = accuracy(shallow)-accuracy(deep), separately at each norm.
Interaction I_a = W_harm(deep)-W_harm(shallow).
Activation contrast H = W_harm(ELU)-W_harm(leaky), separately at each depth.
Three-way J = I_ELU-I_leaky. Compute analogous contrasts for L and CE with explicitly
stated sign convention. Seeds are paired across activations; tasks are NOT replicates.
Save individual seed values, mean, sample SD, and two-sided 95% Student-t CI (df2,
critical4.30265273). Small sample means uncertainty may remain wide.
Meaningful equivalence margin = +/-0.5 accuracy points for H and J.
If H upper CI < -0.5, support materially smaller ELU width harm at that depth.
If H CI lies wholly inside [-.5,.5], practical equivalence within this regime.
If H lower CI > .5, support materially larger ELU width harm. Otherwise INCONCLUSIVE.
For J use same material/equivalent/inconclusive rules with direction stated.
Do not conclude no effect from nonsignificance. Secondary outcomes descriptive, no
multiple-endpoint selection or relabeling based on which results are favorable.

## QA and provenance
Before full run: frozen host forward/Adam equivalence, ELU/leaky derivative check,
exact branch parameter/Adam/RNG copy; finite guards; neutral restore reference equality.
Projection asserts every update: norm relative error<=2e-6; mean absolute error<=2e-5;
rowmean absolute error<=2e-7; untouched tail exactly equal; bias changes recorded.
Measure explicit training-input mean cross-check on task1/t20 and all continuation
endpoints (can chunk input), with tolerance2e-5. Test a wrong-bias mutation and wrong
norm target to show invariant checks detect errors. Check probes do not advance task
RNG or mutate parameters. Compare untreated prefix/reference per-unit centered norms
to committed elu_growth_0909 anchors where available (report missing anchors).
Technical failures are fixed and documented without changing targets, endpoints or
allocation; divergence or failed manipulation invalidates that cell (no causal label).
A 2-prefix/2-continuation smoke run may precede full run; it is implementation QA,
not used to choose targets or endpoints. Full 120-task outcomes only after root confirms
preregistration commit and push. Save code/spec/host/data SHA256, runtime versions,
git hash, timings, command, raw CSV/NPZ/checkpoints, summary.md and verdict.csv.

