# Why small readouts lift activity while the negative body still sinks

Experiment: readout_lift_1111. User request 2026-11-11: verify why readout shrink raises unit output even though sinking has not stopped. This is a new diagnostic, informed by demand_allocation_1111 and the September vfreeze/two-group analysis, not a blind prediction of those known results. No manuscript edit. No causal percentage of all plasticity loss.

## Scope and source

Use the archived ELU checkpoints immediately BEFORE tasks 10 and 25, seeds 0,1,2, from demand_allocation_1111. Restore all parameters, Adam moments/steps, labels, and 4000 minibatches. 784-100-10, 1200 fixed raw-MNIST/255 examples, batch16, Adam lr .001, beta .9/.999, eps1e-8, float32 native expm1 ELU, CPU one thread per worker. Hash source checkpoints, input cache, imported engine, specification, and implementation. Fixed training-set accuracy is not test generalization. The six seed/task states are not six independent seeds.

This isolates within-task selective growth at two existing states. It does not assert the same mediator fractions for September tasks151-200. That older run is an explicitly post-hoc audit only.

## Derivation and alternative hypotheses

For ten-class softmax CE, u=(.1,...,.1), e=p-y, D_i=v_i^T e. The exact split is D_i=D^U_i+D^Y_i, with D^U_i=v_i^T(p-u), D^Y_i=v_i^T(u-y). L_U=logsumexp(ell)-mean_c ell is the uniform-target objective; L_Y=(u-y)^T ell is linear and unbounded alone. We therefore will not train a label-only loss to claim a normal optimum. D<0 requests activity growth, and ELU has nonnegative derivative; actual parameter/activation movement also depends on shared inputs and Adam history.

Hypotheses to distinguish: (a) readout regrowth is required for lifting; (b) reduced sinking alone accounts for lifting; (c) persistent input-label associations create selective expansion against a simultaneously sinking body; (d) optimizer memory or unlabeled stochastic motion suffices. Fixed readouts, uniform-target and resampled-label controls, and imposed body trajectories address these separately. Source accounting in an observed Adam denominator is descriptive, not a causal deletion or additive mediation percentage.

## Arms, fixed before new main outcomes

Each state is continued exactly4000 steps in nine arms:

- S: unchanged; Q: canonical Q column-norm reset plus mean-logit-preserving bias compensation; QV: Q with V fixed, output bias still trained. These three reproduce archived endpoints bitwise and provide a link to the earlier experiment.
- V1: original V fixed, output bias trained, hard fixed new labels.
- V01: V'=Vbar+.1*(V-Vbar), b'=b+(V-V')mean(a0), V fixed; hard fixed new labels.
- V01U: same as V01, but uniform soft targets u for BOTH body and output-bias training.
- V01R: same as V01, but fresh independent uniform hard labels at each step/example occurrence. Use one deterministic private [4000,16] label array per source state. Evaluation always uses the original fixed new labels. This removes stable input-label association while retaining stochastic label gradients; its noise is not identical to the hard-label trajectory.
- V01M: V01 hard labels, with an after-step hidden-bias correction that forces each eligible unit's fixed negative-body mean preactivation to the V1 value at that same step.
- V01UM: V01U with the identical V1 body-mean trajectory imposed.

Vbar is the class-row mean of each column. Scaling preserves class-column directions and means, with centered column norms multiplied by.1. Bias compensation preserves full-set mean logits only at step0, not all probabilities. All retained optimizer histories are identical at the fork; frozen V never updates its moments. Adam for hidden and output-bias parameters is otherwise retained, including after a body correction. Matching introduces an explicit external bias force, not an ordinary optimizer or a pure mediator intervention.

For the fixed body of unit i, B_i={n:z_ni at the original fork <= -3}. Eligible units have at least30 such inputs. For other units report exclusion count and do not clamp. Precompute xbar_Bi and track bmean_i(t)=w_i(t)^T xbar_Bi+beta_i(t). V1 produces a reference at EVERY step before any matched arm is run. After each ordinary V01M/V01UM update, add target-current to beta_i. Save the correction trace; verify agreement within 2e-5*max(1,abs(reference)). This keeps the negative body at the same measured depth while allowing shape/selectivity to change. It does not match whole preactivation distributions or gradients.

## Measurements

Full-set evaluation at steps0,1,10,50,200,500,1000,2000,4000. Save per-unit initial and current mean, body mean/SD, maximum h_i, positive fraction, and mean activity. Fixed upper group T_i is the largest60 original z values (5% of1200), tie broken by stable index order. Record fixed-upper z/a, fixed-deep z/a, and current top5% separately. Never treat a reselected group as a fixed population.

Primary shape statistic G(t)=mean_i[h_i(t)-bmean_i(t)] over eligible units, and change from step0. Also report median_i h_i(t), opening fraction, mean_i bmean_i(t), and fixed-upper changes. A larger gap is not by itself proof that absolute output grew: absolute h, positive activity and openings accompany it. Within-unit co-occurrence is fraction of eligible units with body change<-.1 AND h change>+.1, separately0->200,0->4000,200->4000. Report paired-seed differences, all three seed values, and means; no unitwise pseudoreplicated p values.

At each probe state record native full-data D_U,D_Y,D and raw hidden gradients, by all, fixed-body, fixed-upper, current-top5%, current-open, and labels matching the maximal readout class of each unit. Also record logits, entropy, CE, accuracy, class-centered readout norm, confidence, correct-class versus competing logit margins, and alignment mean((a-mean_a)*(-D_Y)).

For exact next-step Adam probes, clone state and use the next scheduled minibatch. Maintain first-moment source histories for uniform and label terms during training, plus the original first moment multiplied by beta1^t. Use the REAL next second moment and bias correction for all sources, and report old-history, uniform-history, label-history, and numerical residual parameter contributions. Map each to fixed receivers via X DeltaW^T+DeltaBeta and to current/fixed upper and body means. For matched arms report external bias projection separately. Check reconstructed parameter movement, preactivation movement, and moment sums. Do not equate finite activity change with phi' times dz when crossing branches.

Frozen-state residual/map counterfactuals at each probe: centered readout factors .1,1,10 relative to the current V, with current-mean bias compensation. Cross native versus changed probabilities with native versus changed V; report D sign/magnitude and split terms in the same groups. This diagnoses whether altered residuals, a pure backward multiplier, or later learned associations explain positive demands. It is not another trained arm and not an assumption that small V increases all upward gradients.

## Registered decisions

P1 reproduction/identity: S,Q,QV endpoints equal old saved weights bitwise; fixed V unchanged; mean-logit compensation, uniform/label gradient decomposition, moment accounting, and body matching meet numerical tolerances. A failure invalidates the affected comparison until fixed and rerun with documented implementation changes.

P2 readout regrowth not necessary: QV and/or V01 has positive absolute median-h change and positive G change at4000 in all3seeds, for each task separately. Otherwise report mixed/absent. Compare V01-V1 for G and h; small-readout selective advantage requires both differences positive in all3seeds. No equivalence from a small/zero difference.

P3 stable-label contribution: at4000, V01 exceeds BOTH V01U and V01R in G and in absolute median h in all3seeds per task. Report effects even if only one control passes. This tests the intervention's necessity in this setting, not sole sufficiency or a noise-free isolation.

P4 sinking-matched expansion: V01M exceeds V1 AND V01UM at4000 in G and median h, all3seeds, with matching verified. Report actual body trajectories and co-occurrence even if body did not sink in a particular window. Support only says retaining shallower body is not necessary under this controlled trajectory; it does not give a mediator fraction in natural Q.

P5 direction/source: at planned late probes200,500,1000,2000, average the actual next-step label-history contribution to fixed-upper z across the four probes within each seed/arm. Report its sign and the uniform/history contributions alongside; a positive value in all3seeds supports a label-driven upward component. Full D and total displacement can have different signs; numerical checks must pass. This is descriptive accounting, not independent intervention evidence.

Practical magnitudes and signs are both published. No outcome-based dropping of tasks/seeds/arms or widening a failed criterion. If a registered condition fails, state that failure. Source/archive audit numbers remain post-hoc. No claim of proving global recovery or a universal 1/V compensation law.

## Validation, runtime, and delivery

Commit this spec before source-state smoke and any new main continuation. Commit implementation before main. Synthetic float64 chain-rule/gradient split and Adam-source tests, native float32 ELU derivative check, fixed-body projection test, and short real smoke <=12steps per path before full run. Main two CPU workers, one thread each, nice10, fixed54 task-equivalent continuations; budget60 wall minutes and2 CPU-core hours, raw<1GB. Poll long jobs no more than every5minutes unless expected to finish sooner or attention is needed. Recover execution failures only under unchanged design and source hashes; document any implementation correction and rerun affected results.

Save code, summaries/verdict, per-seed tables, checks, and one static scientific figure to main. Archive ignored raw/logs with bytes and SHA256 before removing this worktree/branch. Save a Japanese result note and link from the existing demand-allocation note using Obsidian MCP. Do not edit the paper or overwrite concurrent work.

## Pre-main endpoint-probe clarification

No new main continuation outcomes have been inspected. The virtual next-step probe at step4000 uses the first scheduled batch again; for matched arms its external body reference is held at the step4000 V1 value. These virtual probes do not alter trained endpoints and are explicitly flagged post_horizon. Registered P5 only uses200,500,1000,2000. Smoke completed all9 paths for12steps; synthetic gradient-split, Adam-source and projection checks passed. The only post-smoke code change detached two tensors before scalar diagnostic conversion to remove warnings; training math is unchanged.
