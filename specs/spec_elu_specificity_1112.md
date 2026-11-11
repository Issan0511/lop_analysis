# ELU dependence of the readout cost after P

Experiment elu_specificity_1112, requested by Issa on 2026-11-12: after asking whether the difference is simply because of ELU, "しらべめよ" (investigate it). Base main 9f78ec704ef71a261f473569df3bc92526b2da1c. This specification is frozen before any new real-data replay, smoke or intervention.

## Question and identification

Previous readout_separation_1111 compared an ELU MLP without LN at task10 with a ReLU ViT with LN at task100. It separated readout interventions but did not separate activation, trained state and architecture. In the long ViT, SL means shared head + LayerNorm, NOT SiLU (nor the SmoothLeaky activation named SL in the older activation battle). The actual nonlinearities are ReLU.

Primary question: within the same MLP, does the readout-scale benefit after P depend on activation and on final affine LN when each combination is trained from the same initial weights, data and task/batch streams? This identifies the total effect of an activation/LN training regime, including the state it builds. It does not hold the eventual trained representation fixed.

Secondary question: in the actual pretrained ViT, does using a genuine ELU forward AND backward, instead of ReLU at the SAME saved weights, change the response to readout P/shrink? This is a current-function intervention on ReLU-trained histories, not an ELU-trained ViT. Compare existing no-LN and all-LN histories separately. The MLP final-LN factor and ViT all-five-LN factor are not the same treatment. No direct architecture-only effect or additive causal shares will be claimed.

Prior results inform selection of age and interventions. ReLU/SiLU show smaller zero-head benefits but poorer zero-head recovery, so a small removable cost must not be interpreted as a healthy body. Genuine activation training and activation replacement are kept distinct from the previous forward-identical unity-VJP experiment.

## Study A: matched MLP activation x final LN histories

- Canonical stage1_ctrl_1111 engine: MNIST raw/255, fixed1200 images per seed,784-100-10, batch16,4000 updates/task, iid uniform10-class labels renewed each task, Adam lr.001,beta(.9,.999),eps1e-8,weight_decay0,CPU float32,one torch thread per job.
- Seeds0..9, activations ELU(alpha1), R(ReLU), LR(negative slope.1), SILU. Two final-LN levels: off/on. LN is AFTER activation and BEFORE head, h=LayerNorm(phi(W1 x+b1)),100 features,eps1e-5,affine gamma1,beta0 initially. It is not preactivation LN.
- For each seed, all eight models share exactly the canonical initial W1,b1,W2,b2, subset, labels and batch order. LN parameters are additional deterministic parameters and do not consume the initialization stream.
- Train canonical shared-readout histories from scratch through task10, saving task1 and task10 parameters/Adam. Off-LN histories must reproduce archived weights and labels at EVERY task bit-for-bit. Do not silently accept numerical drift. New on-LN histories have no old reference; verify replay, source serialization and deterministic small repeats.
- Save source-state hashes, initial common-weight/subset/label/order hashes and history accuracy/CE. Do not select/exclude states by outcome.
- At both task1 and task10, fork next-task continuations for every registered head intervention below,4000 updates. No early stopping. Same next labels and batches across all activation/LN regimes for the same seed and age.
- Sample count8 regimes*10seeds*2ages*6arms=960 curves. Statistical unit is seed, never unit/example/arm.

## Study B: actual ViT with matched current activation

- Archived runs vit_long_1024_2blk_rlfix/runs, source arms S(no LN) and SL(all five affine LN), task-end30, seeds10252010..10252014. Restore model, optimizer and RNG; hash all source files. Use the original2blocks,embed384,FFN3072,patch8,ReLU-trained models,dropout0,shared10active head rows. Fixed2500 TinyImageNet images and new random labels for task31,original seed-specific data/task stream; batch128 incl last68,Adam1e-4,betas(.9,.999),eps1e-8,WD0.
- Current activation factor R versus ELU(alpha1), replacing BOTH FFN activation modules with mathematically genuine activations and no added trainable parameters. Keep all learned weights and all optimizer memories identical before head interventions. A fresh object/configuration is used so compiled graphs cannot silently reuse the wrong activation. R must reproduce the native model outputs/gradients.
- Recompute full-set features and mean-logit compensation under the chosen current activation. This does not preserve the original ReLU model's function: the change in function is part of the registered acute activation treatment. Record pre-head-treatment old-label/new-label accuracy,CE,logit statistics and feature statistics to expose disruption.
- Six head arms as below,1500 updates,full horizon. No activation release: R or ELU remains in use through the continuation. Current activation and head intervention are crossed from the same original parameter/Adam/RNG state.
- Sample count2 LN histories*5seeds*2current activations*6arms=120 curves. Same-seed S/SL comparisons concern their trained LN regimes; ELU-R comparisons within one history concern the current activation. There is no ELU-trained source and no extrapolation to it.
- Use compiled forward where validated. Warm128 then68 batch shapes without optimizer updates, restore parameters/Adam/RNG, and verify restoration before every fork. Never change evaluation convention to force agreement.

## Head interventions (both studies)

All six arms begin with head Adam cleared. Body Adam is retained except in Z_clear. Body includes final-LN gamma/beta where present. Adam reset means moments and step counter.

1. S1: native readout,body memory keep.
2. S01: native direction,class-centered readout scale.1,body keep.
3. P1: canonical independent class-entry permutation per column,body keep.
4. P01: same P,class-centered scale.1,body keep.
5. Z_keep: head weight AND bias zero,body keep.
6. Z_clear: head weight AND bias zero,all body Adam cleared.

For physical S/P scale c in {1,.1}, W'=Wbar+c*(W_D-Wbar), b'=b+(W-W')mu0, where mu0 is the full-set feature mean before the head intervention under the selected current function. Preserve each class's mean logit at this fork. S1 uses an exact no-op. P uses the same canonical seed/age/architecture permutation as readout_separation_1111, independent of activation and LN. P preserves column multisets, not the full Gram or probabilities. Scaling is a joint parameter/function/learning-response intervention, not confidence-only. The zero-head arms intentionally do not preserve mean logits and are reported separately, not as c=0 on the same physical-scale curve. Inactive ViT head rows remain unchanged for S/P and are zeroed for Z to match prior zero-head controls; only active rows enter the loss.

Zero-head recovery is an optimizer-and-function intervention, not a measure of irreducible capacity. Its poorer recovery can explain a small benefit without proving permanent damage. Report absolute Z curves as well as differences, and never normalize costs by an uncertain "head-free capacity" denominator.

## Measurements and contrasts

Primary U=trapz(1-full-fixed-training-accuracy)/horizon. Positive U(S/P1)-U(S/P01) means scale reduction improves period-average training accuracy, in percentage points after multiplying100. It is not test generalization, time-to-threshold, or a causal percentage of damage.

Evaluate at0,1,5,25,50,100,200,each100MLP/25ViT,endpoint. Preserve canonical native MLP full-set evaluation batching for exact old-control comparison. ViT evaluates all2500 examples with the selected genuine activation using the same numerical engine across arms. Secondary Uacute[0,200],Ulate[200,end],CE area,endpoint accuracy/CE,first and sustained observedT90 (null if not attained; sustained means all subsequent SAVED observations, not a guaranteed duration).

State diagnostics at source and0,25,200,endpoint: centered head norm,head/feature mean and centered RMS,mean/centered logit scale,old-label accuracy,confidence/entropy,hidden preactivation mean/std,positive occupancy,mean absolute actual activation derivative,small-derivative fraction (|phi'|<.05). MLP diagnostics before/after LN distinguish activation from final features. ViT diagnostics perFFN use the actual R/ELU derivative; final gamma/beta and effective Wdiag(gamma) norms if present. Record real body/head update norm and gradient norm at updates1,25,200,end, separating final LN affine where feasible. Finite checks are mandatory; no dropped nonfinite seeds.

Planned per-seed contrasts within each source regime:
- B_S=U(S1)-U(S01), B_P=U(P1)-U(P01).
- D_1=U(S1)-U(P1), D_.1=U(S01)-U(P01), scale-by-P interaction B_S-B_P.
- Z_S=U(S1)-U(Z_keep), Z_P=U(P1)-U(Z_keep), M_Z=U(Z_keep)-U(Z_clear).
- Absolute U/endpoint/T90 of Z_keep and Z_clear, not only removable cost.
Primary StudyA at task10: B_P(ELU)-B_P(R/LR/SILU), separately at each final-LN level; B_P(LNoff)-B_P(LNon) within each activation; activation-by-LN differences of these differences. Other contrasts and task1 are planned secondary; age-by-activation comparison reports learned-history dependence, not a mediator.
Primary StudyB: B_P(currentELU)-B_P(currentR) within S and SL histories; the difference of these activation effects across the two LN histories. S/SL differences of B_P at fixed current activation are secondary. Training-regime LN difference is not an instantaneous LN-only removal.

Paired-seed bootstrap10000,private deterministic streams,percentile95%,descriptive unadjusted intervals,10MLP/5ViT. Publish all contrasts. Signed support when interval is entirely positive/negative; otherwise unresolved. Practical reference1pp but no equivalence inferred from zero crossing. Cross-study absolute pp are not directly comparable effects because datasets/architectures/ages/horizons differ. Do not call an activation-specific pattern an architecture cause. Local derivative dependence and long-run intervention effects are different estimands.

## Validation, provenance, resources and completion

Commit this spec before real-data smoke/replay. Commit implementation before main. Synthetic tests cover LN forward/autograd/parameter grouping,physical-head invariants,zero-head reset semantics,shape order,warmup state restoration,metric sign/pairing/censoring,malformed/incomplete-output rejection. Native no-LN MLP source replay must be bit-exact. Native ELU/noLN task10 S/P scale controls are checked against readout_separation_1111; old R/SILU/ELU S1/P1 and Z_keep controls can also be checked against p_residual_loading_1111 under the same head/body memory and evaluation convention. Do not mix zero-head and mean-logit-preserving-scale outputs.

Smoke uses MLPseed0 and ViTseed10252010,all new execution paths,at most26continuation updates. Timing/compilation overhead is charged. No scientific seed/arm/horizon changes after main begins. Failures are preserved and diagnosed; rerun affected outputs after a genuine code fix with a new frozen implementation hash, never treat failed curves as exclusions. Existing control equality is not relaxed to hide algorithmic errors.

Main fixed counts:960MLP+120ViT=1080 curves. Up to4single-thread CPU jobs and1GPU job. Total cap4GPU worker-hours,8CPU core-hours(including GPU host/compiler children and all smoke/retries),raw/log archive<6GB. Conservative external usage can be upper-bounded but must be labeled. If caps or validation fail, preserve partials,mark incomplete,do not publish a complete signed interpretation. This is a targeted study; no full ELU ViT task100 retraining,cloud spend,manuscript editing,or claimed universal architecture separation.

Monitor long operations at most every5minutes unless expected to finish sooner or an error needs immediate attention. Archive all ignored/untracked raw/logs except __pycache__ with verified SHA256 manifest before worktree removal. Commit code/spec/results,merge latest origin/main and push,verify ancestry,remove only this worktree/branch. Save a new Obsidian result note via MCP. The previous synthesis's unity-VJP wording must not be used to exclude ELU saturation/asymmetry as a causal candidate.
