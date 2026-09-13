# Output-matched ELU response intervention 0913

Status: PROSPECTIVE PREREGISTRATION. User authorized execution on 2026-09-13.
Scientific design/interpretation: root Codex. Implementation, aggregation and independent checks: user-requested GPT-5.6 Sol.
Parent: spec_elu_sunk_rescue_0913, results commit 8ddd987ac13951c6f7323e8f2a2957407423b86b.
Prior results are known and motivated RL/T20/L2 selection. No claim of blind setting selection.
Worktree /home/issan/Projects/claude/elu_response_anchor_0913; branch codex/elu-response-anchor-0913.
Do not run substantive outcomes until this spec AND implementation/technical QA are committed and pushed.

## 1. Question and intervention estimand

Does increasing the response of naturally saturated selected ELU units improve subsequent fresh-task learning when their INITIAL features, logits, weights and optimizer history are held equal?
Separate initial feature offsets from the response function. This does not hold features equal throughout learning: their learned change is an outcome.
The response intervention changes derivative, curvature and saturation trajectory, not merely a scalar gradient multiplier.
A positive result identifies an engineered response intervention effect at this checkpoint, not a natural indirect-effect fraction and not proof that all LoP arises from sinking.
Flat low performance can be sustained lost plasticity. Absence of additional decline is not absence of prior loss.

## 2. Fixed source and cohort

Use original ELU reference RL seeds0,1,2 at end task20.
Source checkpoint: /home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913/prefix/checkpoint_20.pt
Source units and RNG hashes: same prefix/units.npz and prefix/rng_hashes.json.
Source selections: /home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913/RL_t20_l2/selections.json
Verify source file SHA256 against prior raw manifest/provenance and pinned selections. Retain hashes in new provenance.
Reuse exact target20 IDs and ordered deltas20 in each seed. Independently verify these IDs are in the pool q>=.95 at each task16,17,18,19,20. q=P_inputs(phi'(z)<.05).
Do not reselect by future outcomes, substitute other IDs, or enlarge the cohort. All3 sources have exactly20 selected IDs.
The delta_i=max(0,-1-mean_x z20_i) is the previously fixed, old-task input-only displacement; all other delta entries are zero.

MNIST fixed1200 training inputs/seed; 784-100-100-10 MLP; ELU alpha1 both layers.
Same original Adam lr=.001, betas(.9,.999), eps1e-8, batch16,80epochs=6000updates/task.
Preserve all parameters, Adam m/v and t=120000 at branch start. No optimizer reset, L2 or norm projection.
No prefix retraining needed: the previously validated checkpoint is the source.
18 models are evaluated in a new common batch geometry. Exact long-run equality with old33-model runs is not required or claimed; source initialization/state identity and synthetic same-geometry equivalence are required.

## 3. Fixed-anchor differentiable function

For a selected layer2 unit, z_theta(x) is the current unshifted preactivation, z0(x) the preactivation of the fully frozen source network on the same RAW input x.
Let F,K be0 or1:
a_FK(theta,x) = phi(z_theta(x)+K*delta) - phi(z0(x)+K*delta) + phi(z0(x)+F*delta).
Unselected units use ordinary phi(z_theta).
The source prefix through layer2 must be fully fixed, including layer1. Do not calculate anchors from the changing current layer1 activations.
Since RL inputs are fixed, cache source phi(z0) and phi(z0+delta) once on all1200 inputs, without future labels. Keep caches immutable and align by sample indices during batching.
At theta0, a_FK=phi(z0+F*delta). Thus INITIAL features/logits agree within F across K.
Derivative with respect to trainable theta is phi'(z_theta+K*delta) times dz_theta/dtheta. Use true gradients of this differentiable forward, not an STE.
The fixed correction is never refreshed or decayed.
Implement off-diagonal cases as anchor + (current activation - anchor) to reduce cancellation.
A/AF use ordinary phi(z_theta) exactly; D uses phi(z_theta+delta) exactly; these are algebraically equivalent simplifications, not different scientific interventions.
Stored biases remain equal at initialization; delta is a fixed function shift. D corresponds to ordinary bias lift by reparameterizing selected b2'=b2+delta.
No claim that all inputs are open when the mean shifted argument is -1. Fixed delta can re-sink.

## 4. Six branches and task horizon

A: F0K0, all parameters trainable.
B: F0K1, all parameters trainable.
C: F1K0, all parameters trainable.
D: F1K1, all parameters trainable.
AF: F0K0, entire layer1 W1 and b1 frozen.
BF: F0K1, entire layer1 W1 and b1 frozen.

All layer2 and output parameters, including selected incoming rows, b2 entries and outgoing columns, remain trainable in every branch.
Freeze means actual W1/b1 values restored after EVERY Adam step to branch-start values; m/v may continue updating. Do not merely zero gradients.
Same3seeds x6branches =18 continuations.
Task21 is PRIMARY. Tasks22,23,24,25 are PRESPECIFIED durability secondary, each6000updates; no further intervention at switches.
Continue original saved task/RNG streams for label draws and minibatch permutations. Even unused PM permutation streams are advanced identically for reproducibility.
Use exact same fresh labels and minibatch orders across all6 branches within seed.
Verify task21-25 RNG hashes against saved prefix RNG hashes. Do not consume RNG to draw anchors.
No other environments, checkpoints, layers, reverse interventions, ongoing depth control, adaptive retargeting, or update-normalization arms are authorized by this spec.

## 5. Primary and secondary outcomes

Primary per-seed loss L(branch) = arithmetic mean of all6000 pre-update minibatch cross-entropies on task21.
Save EVERY step's CE and accuracy directly, rather than infer small increments by subtracting large cumulative float32 sums. Accumulate reported totals in float64 off-line.
This is a new prospective primary endpoint for this experiment. The older rescue study's six-point full-probe AUC and INCONCLUSIVE verdict remain unchanged.
Primary contrast Delta_K0 = L(A)-L(B); positive means response restoration improves fresh-task learning with identical initial features.
Directional3seed pilot rules:
- all3 Delta_K0 > .01 -> DIRECTIONAL_RESPONSE_SUPPORT
- all3 Delta_K0 < -.01 -> DIRECTIONAL_RESPONSE_HARM
- otherwise -> INCONCLUSIVE
Technical failure -> TECHNICAL_FAILURE, no scientific label. These thresholds are directional pilots, not population significance or equivalence.
Report each seed, mean and paired t95% df2; tasks/units are not independent replicas.

Prespecified secondary task21 contrasts:
Delta_K1=L(C)-L(D) (response effect at lifted initial features);
I=Delta_K1-Delta_K0 (response-by-initial-feature interaction);
Delta_K0_L1fixed=L(AF)-L(BF);
feature contrasts L(A)-L(C) and L(B)-L(D), with explicit INITIAL feature/logit mismatch across F;
total ordinary lift L(A)-L(D).
All secondary contrasts neutral/descriptive; do not reuse primary support labels on them.
Other secondary outcomes:
- step6000 full1200 train/memorization CE and accuracy for each new task;
- full-probe CE trapezoidal AUC normalized by6000 from the FIXED measurement grid;
- mean online CE/accuracy across all5 tasks and task-by-task trajectories;
- first75-update online means for transient mechanism description.
Use percentage points for accuracy differences and retain cost/benefit signs explicitly.
These outcomes measure fitting novel random assignments on training inputs, not unseen-example generalization.
Do not discard the initial intervention shock in F1 versus F0. Within-F logit equality removes this INITIAL confound.

Full1200 probe measurements at steps:
0,1,2,5,10,20,25,50,75,150,375,750,1500,3000,6000
for every task. Early updates may use eager stepping and later25-update graphs; verify equivalence technically.
Log task and literal update step, including both sides of a task switch.

## 6. Mechanism measurements

At the fixed probe grid save all-unit, and explicitly target20-ID, measurements:
- unshifted z2 mean and spread;
- effective trainable argument z2+K*delta mean/spread, response gate mean and q;
- actual corrected layer2 activations, their mean/spread and fraction varying across inputs;
- W1/W2/W3 row/column norms, centered incoming row norms and directions via actual saved parameters;
- interval actual dW/db and input-wise delta z means/SD, corrected activation changes and logit changes.
Freeze does not imply preactivations are fixed when upstream inputs learn; AF/BF control that pathway.
Keep low-gate state distinct from functional contribution; no permanent death label from finite5task observation.

At steps0,1,20,75 per task and at every task boundary save complete p/m/v/t states for independent replay.
At those steps use the NEXT scheduled training minibatch for a non-mutating diagnostic where available (within-task indices step..step+1; step6000 use final minibatch explicitly labelled retrospective diagnostic).
Save raw current gradients, Adam current-gradient numerator part, history numerator part, denominator, predicted update and actual realizable updates with freeze mask. A decomposition with fixed realized denominator is an arithmetic accounting, not counterfactual removal of history.
At minimum retain raw tensors or compact tensor archives allowing independent recomputation; record which probes were evaluated and avoid consuming task RNG.
Report actual data-induced update separately from gate sizes. Do not assume Adam cancels a new gradient scale after an intervention with old m/v.
All norm differences after initialization are outcomes. Do not claim W fixed over training or a W-independent mediated fraction.

## 7. Technical acceptance gates before outcomes

Synthetic tests may run before substantive execution:
1. A/AF function at no-freeze reproduces original base forward/manual gradients/Adam on same geometry. Exact equality targeted, maxabs <=1e-6 required.
2. D forward corresponds to explicit shifted bias; use atol1e-5+rtol1e-5 on synthetic float64/float32 checks; record rounding/reparameterization.
3. Manual gradients of all6 parameters match independent float64 autograd of the actual corrected function for all F/K cells with nonzero anchors: atol1e-7+rtol1e-5. Confirm correction independent of trainable parameters. Finite-difference directional derivative outside ELU kink also checked (relative error<1e-4, report absolute error).
4. F0 and F1 initial pair features/logits agree. On source full1200 require exact equality targeted; accept only float-rounding bounded by atol2e-4+rtol1e-6 and report maxima. Probe and representative16-sample minibatch equality both checked. Tolerances not silently widened.
5. Source p/m/v/t copies are bitwise identical across all branches. Cached source anchors/hash and delta arrays immutable. Target counts20 and past-only validity verified.
6. Synthetic CUDA graph versus eager state and online step-log buffers agree with nonzero optimizer history, maxabs5e-5. Actual AF/BF W1/b1 freeze exact at every measurement, final, and synthetic every step.
7. Synthetic full-feature freeze: within-F K pairs have equal functions and identical readout-only updates. This is a technical null check, not an extra scientific branch.
8. Per-step CE/accuracy buffers contain exactly6000 finite entries/branch/task, reference eager last-step loss matches. Full90 task endpoints and1350 fullprobe rows required (18*5*15).
9. Source subset/order/labels/RNG hashes match, all parameters/moments/logs finite, final t150000. A branch reused as pair baseline is not duplicated as extra evidence.
10. Source-code/spec/data/anchor/probe hashes, PyTorch/device/TF32/determinism and launch commit saved. TF32 disabled; deterministic algorithms enabled; preserve original model/batch gradient scaling.
QA results are tied to source SHA256. Stale QA blocks run. No scientific outcomes before PASS and code/spec commits pushed.
If a technical gate fails, save failure, fix with an explicit pre-run technical amendment when tolerance/semantics change, and preserve old artifact. No post-outcome endpoint changes.

## 8. Limits and data custody

B versus A can show usefulness of restoring this response with identical initial features. It is an engineered checkpoint-specific architecture and may not describe a natural mediator.
C versus A can reflect useful fixed feature offsets. Such a pathway is compatible with sinking-related LoP, not proof against it.
AF/BF test a restricted network; a negative result may reflect loss of necessary upstream adaptation.
If corrected response re-sinks or other bottlenecks persist, failed rescue is not proof of harmless sinking.
Prior RL/T20L2 has nearly all units saturated; no nonsunk matched-group specificity claim.
No task/unit pseudoreplication, no dropping seeds or replacing prior INCONCLUSIVE.

Code: src/elu_response_engine_0913.py, src/elu_response_engine_validate_0913.py, src/elu_response_anchor_0913.py, src/elu_response_report_0913.py
Results: results/elu_response_anchor_0913/
Commit tables, code, figures and source/QA/audit manifests. Raw pt/npz excluded from Git and backed up with SHA256 verification at /home/issan/Projects/obsidian-research-data/elu_response_anchor_0913/files/.
Create Obsidian result under 測定, move executed spec overview to spec/実行済み, preserve older results and current-context limits.
