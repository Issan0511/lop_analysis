# Disentangling readout growth harm in ELU and final-LN ViT
Experiment readout_separation_1111. Issa, 2026-11-11: "ちゃんと分離して".
Base main 0096ab39fa6f63377b655c354282788a52648c61.
This document is committed before new checkpoint reconstruction, smoke, or trained interventions.

## Scope and identification
Prior studies loading_bridge_1111 and p_residual_loading_1111 informed this design. Their comparisons are not new held-out evidence. P preserves column multisets but changes other geometry, bias, probabilities, and gradients. Zeroing W,b is not a pure norm intervention. We will not assign a percentage of total delay to "alignment" or "size", nor infer that the architectural difference is caused solely by LN.

Separate (1) a physical direction-by-scale-by-body-memory factorial, (2) an initial-200-update forward/backward mechanism factorial with head gradients separated from body gain, and (3) same-forward Jacobian interventions, plus exact frozen-state crossed residual/pullback diagnostics. Positive shared behavior is operational evidence, not identity of microscopic mechanism.

No new from-scratch architecture training, Pall history, activation sweep, manuscript edits, or causal claim about why norms originally grew. Sources are the same saved late ELU and SL ViT states, so conclusions are restricted to these states and tasks.

## Fixed sample and data
- ELU one-hidden-layer 784-100-10 MLP, source task-end10, next task11, seeds0..9, full4000 updates. Reconstruct saved body AND Adam by canonical replay. Verify all10 task-end weights and labels per seed bitwise.
- ViT SL,2blocks,final affine LN,source task-end100,next task101,seeds10252010..10252014,full1500 updates. Restore model/Adam/RNG from archived sources.
- Existing helper source and archive paths in analysis/loading_bridge_1111/dynamic.py and results/p_residual_loading_1111/provenance.json. Hash sources, spec, all new code and imported helpers. No excluded seeds or outcome-dependent subsampling.
- Identical deterministic next-task labels and batch sequence across each source-state fork. Head Adam is cleared in every new continuation. "Body keep" includes LN gamma/beta memories. "Body clear" clears all nonhead memories including step counters.
- MLP and ViT update counts are not equivalent units of physical training time. Accuracies are fixed training-set accuracies, not test generalization.

## Physical readout interventions
Let W,b be the active10-class readout, mu0 the full fixed training-set feature mean at the original fork, and Wbar its class-row mean. Define direction D=S,P,O:
- S: original W.
- P: canonical independent class permutation for each column, identical to p_residual_loading_1111. Preserve each column multiset and centered norm.
- O: an orthogonal class-space rotation fixing the all-ones direction. Construct a deterministic orthonormal basis B for the 9-dimensional complement of ones, draw a private seeded9x9 Gaussian, QR with positive diagonal convention to obtain R, Q=11^T/10+B R B^T, and W_O=QW. Preserve W^T W, all column norms, column means, singular values, total norm. Row individual norms and nonlinear softmax probabilities need not be preserved. One pre-fixed O draw per seed; this is a geometry robustness control, not a universal "alignment eraser".
For c in {1,0.1}, use W'=Wbar+c*(W_D-Wbar), b'=b+(W-W')mu0. Preserve per-class full-set mean logits at the fork. W centered-column norms scale exactly by c; the scale comparison preserves the chosen directions. Inactive ViT output rows remain unchanged, all head optimizer states are cleared.
The c=.1 contrast is a bias-compensated weight-scale intervention, not a confidence-only intervention. Norm matching is at the initial fork, not a constraint imposed throughout learning.

Continuation conditions:
- S/P x c=1/.1 x body-memory keep/clear:8.
- O x c=1/.1 x body-memory keep:2.
The two S/P c=1 body-keep controls must reproduce the corresponding previous KC curves (allow documented engine float tolerance for ViT; exact MLP native outputs). Do not silently combine incompatible engines.

## Forward/backward continuation controls
Begin from unscaled S or P physical direction, with head Adam clear and body Adam kept. Define z=W h+b, zbar0 the per-class mean logit at that fork.
Forward policy during first200 updates: z_F=zbar0+cf*(z-zbar0), cf in {1,.1}.
Backward to W,b uses derivative1 through z_F, rather than cf. Backward to body features additionally multiplies by g in {1,.1}. Thus body input gradient =g W^T(p(z_F)-y), while head gradient =(p(z_F)-y)h^T and bias gradient=p(z_F)-y. At a shared state, changing only g leaves predictions AND head gradients identical.
Additional arms per direction: (cf,g)=(.1,1),(1,.1),(.1,.1). The (1,1) comparator is the physical c=1 body-keep arm. Total6 new curves.
At step201 onward, use ordinary forward and backward. No parameter or Adam reset at200. Record an extra pre-release step200 diagnostic; the primary trajectory uses post-release step200 evaluation, with199 and201 also recorded. This exposes any forward-policy discontinuity.
The (.1,.1) surrogate has the same initial p and body gradient as the physical c=.1 arm, but different head parameterization and head-learning response. They are not interchangeable. These interventions isolate instantaneous operators and the consequence of applying them for200 updates; probabilities of diverged trajectories are not claimed to remain pointwise matched.

## Same-forward transmission controls
Use physical S direction and c=1/.1, head Adam clear, body memory kept. Apply only first200 updates and release at201 without changing weights or state:
- ELU: retain exact native forward activations but replace derivative phi'(z) with1. Two additional continuations.
- ViT: retain exact native final-LN forward, gamma gradient and beta gradient, but remove only the radial subtraction from input VJP; separately remove both radial and mean subtractions. Four additional continuations.
Use epsilon in the native sigma and q. These are surrogate Jacobian interventions, not ordinary removal of an LN layer or replacement of an activation. Other LN layers and residual pathways are unchanged. For ViT, record trunk and final gamma/beta gradients/states separately.
A same-forward surrogate may change gradient direction and norm. Frozen probes below include a norm-matched diagnostic. Dynamic arms are deliberately unnormalized and report both magnitude and direction; a worse curve alone does not establish the specific radial mechanism.

Total fixed continuations:18*10 MLP +20*5 ViT =280. All run full fixed horizons; no early stopping at90%. No curves counted as independent replicates merely because multiple arms share a seed.

## Frozen-state crossed assay
Run on every source state before any new continuation. Full-set features and readout stats are descriptive and checked against old sources.
- Select128 fixed training examples using an experiment-private stream keyed by model and seed. Reuse exactly these samples for every operator and memory comparison. Do not use unit/sample count as statistical replication.
- Six physical forward banks=S/P/O x c1/.1, with full-set mu0 compensation. Six backward maps are their W matrices.
- Cross all6 probability banks with all6 backward maps. This matches the entire per-example probability vector when comparing backward maps, not merely CE, confidence, or entropy.
- Treat residuals separately: actual fixed new labels and the expectation under independent uniform new labels. The primary local label channel is actual new labels; expected uniform is mechanism support, not their substitute.
- For each residual/backward-map pair compare native transmission with the above ELU or ViT same-forward Jacobian modes. Report head-input gradient, preactivation/pre-LN gradient, body-parameter gradient, cosine with native, and per-group norms. For MLP report signed mean-preactivation gradient push (positive lowers the mean under gradient descent). For ViT group final gamma,beta separately from the upstream trunk.
- With the SAME calculated body gradient, compute the virtual next Adam update using original memories versus cleared memories; source states remain unchanged. Verify the virtual Adam implementation against torch.optim.Adam on synthetic cases and at least one real parameter group. Report update norm and angle, not just gradient norm. No claim that Adam is generally scale invariant.
- For altered transmission modes additionally rescale the upstream-trunk parameter gradient to match the native trunk gradient norm, keeping final-LN affine gradients separate. Compare cosine and virtual Adam updates. This is a local diagnostic, not a gradient of a newly trained architecture. If native or alternative norm is zero, flag and omit ratio rather than substitute a finite value.
- Save scalar records and small selected arrays, not multi-gigabyte dense parameter gradients. Logical assay size before memory/normmatch expansion: MLP10*6*6*2labels*2Jac=1440; ViT5*6*6*2labels*3Jac=1080.
- ViT gauge sanity: W→.1W, gamma→10gamma, beta→10beta, b unchanged. Verify same logits and upstream/pre-LN gradient (within stated float tolerance); gamma and W coordinate gradients transform. Report W diag(gamma) and W beta+b norms. This is a same-function identity, not an intervention removing all effective readout scale. Raw W norm is not an invariant cause.

## Measurements and registered contrasts
Continuation primary U=trapezoidal integral of(1-full-train-accuracy) over[0,limit]/limit, lower is better. Unit of reported U difference*100 is percentage points averaged over the horizon.
Evaluation0,1,5,25,50,100,199,200,201, each100MLP/25ViT,endpoint. Full training set on every evaluation. Extra pre-release200 stored separately. Layer/state diagnostics at0,25,200pre,200post,endpoint. Surrogate evaluations must actually use the forward policy; do not use the old wrapper-ignoring evaluation helper unchanged.
Secondary: CE area, terminalaccuracy/CE, first and sustained observed T90, censor flag/horizon, centered W norm, effective Wgamma (ViT), entropy, old-task margin, feature RMS, MLP opening/dphi/position, ViT layer opening/position and final gamma/beta norms. Nonfinite values are failures, not dropped seeds.

Prespecified paired-seed contrasts within each architecture:
A. Scale benefit at each S/P/O direction and each available memory setting: U(c1)-U(c.1).
B. Direction effect at each scale/memory: U(S)-U(P), with S-O as geometry-control replication; interaction between scale and direction.
C. Body-memory contribution: keep-clear at each S/P scale; interaction with scale/direction.
D. Forward-only and body-gain-only effects and interaction in each S/P surrogate2x2; physical c.1 versus surrogate(.1,.1).
E. Transmission mode effect versus native, at physical S scales1/.1; scale-by-transmission interaction.
Local results use the corresponding exact matched residual/map/Jacobian contrasts. Only inspect same metric under same label channel and sampling window.

Paired seed bootstrap10000, deterministic private stream, percentile95%, unadjusted descriptive intervals;10MLP/5ViT seeds. Publish estimates/intervals and all planned controls. Positive lower bound>0 supports a signed effect in the sampled condition; upper bound<0 supports the opposite; otherwise unresolved. Practical continuation threshold is .01 U=1pp. No equivalence inferred from a nonsignificant difference. Architectural effects are compared as patterns, not a direct architecture-causal test; task/data/activation/depth differ.
This is a decomposition into identifiable interventions and local maps, not an additive mediation model. We will not normalize the effects into percentages of "all harm". Presence of norm-matched local direction effects does not prove that direction mediates all long-run delay. Absence of immediate push need not imply absence of later adaptation cost.

## Validation, execution, stopping
Commit specification before any real replay/smoke. Commit experiment implementation before main run. No implementation-dependent outcome changes to seeds/arms/metrics after main starts.
Synthetic float64 tests: direction/scale invariants; mean-logit preservation; O Q^TQ=I and Q1=1; manual versus autograd crossed VJPs; surrogate forward/head/body gradient separation; native/custom LN forward and gamma/beta equality; exact LN VJP with epsilon; ELU forward unchanged/derivative1; Adam update formula; gauge identity.
Real smoke outside main outputs: one ELU seed0 and ViT seed10252010, up to26updates, all implementation paths. Validate source replay, standard wrapper/eager/compile agreement, perturbation removal, common initial body/state hashes, deterministic batches, mean-logit invariants, initial matched probabilities/body/head gradients, compatible evaluation. Re-run smoke only for implementation fixes; record all corrections.
Main defaults: four single-thread CPU workers plus one GPU worker. Budget3 GPU-hours total and4 CPU core-hours, raw<6GB. Frozen assay target<=30 GPU-minutes, continuation expected~100 GPU-minutes based on prior timing. Timing-only smoke assesses feasibility; no outcome-based arm omission. If fixed cap/error blocks completion, stop affected job, preserve partials, mark incomplete and do not emit a complete signed synthesis. Recover failed work only under identical frozen code/source hashes, or document and rerun affected outputs after a bug fix. No result-driven repeat attempts.
Monitor at most every5minutes unless an imminent completion/error merits earlier attention. All code, prereg, results, checks and source hashes go to repo main. Archive ignored raw/logs with SHA256 manifest before removing only this worktree. Save a new Obsidian result note via MCP; do not alter manuscript/current-location notes.
