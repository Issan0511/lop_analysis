# ELU reserve freezing 0913: prospective test of a shallow, small-pattern-norm reserve

Status: PRE-REGISTERED DESIGN; no new training outcomes inspected before registration.
Parent: W growth -> plasticity loss / user's 2026-09-13 shallow-small-W reserve hypothesis.
Owner: experiment A. Code: src/elu_reserve_0913.py. Output: results/elu_reserve_0913/.
This tests a particular candidate reserve; an absent candidate or a null selective-freeze effect does not establish that saturation is harmless.

## 1. Fixed design
Activations: established ELU alpha=1 (ELU1) and leaky ReLU slope .1 (LR).
Seeds: 0,1,2. CPU float32, one torch thread per worker; at most four workers.
Generate the established unregularized PermutedMNIST trajectory, with the frozen host architecture, Adam (.001,.9,.999,epsilon=1e-8), 10000 stratified examples and batch16 per task.
Save identical parameter, Adam, and all three task RNG states after tasks20 and100. Each checkpoint independently branches for the following10 fresh permutation tasks,625 updates/task. All branches receive identical permutations, sampled examples and minibatch orders. No readout refit, learning-rate tuning or parameter rescaling.

## 2. Candidate selection before branch training
Use the established fixed512 test-image probe and eight reference permutations (C.refperms). For each first-layer unit record incoming centered pattern norm ||Wtilde_i||, raw norm, bias, row mean, mean/SD of z, and occupancy h_i=P(|z|<=1), pooled equally over the eight permutations. Also save occupancies at .5 and2 descriptively; they never redefine selection.
Define SMALL as the25 smallest centered norms and SHALLOW as the25 largest h_i (stable ties by unit index). TARGET=intersection, with n=|TARGET|. IDs remain fixed throughout all10 continuation tasks.
Record all100 unit diagnostics. Mean z near0 is not used to select, and the gate band is not used.
If n<3, register COHORT_TOO_SMALL / reserve efficacy NOT_IDENTIFIABLE at that checkpoint. Do not expand thresholds, replenish the cohort, or substitute a rank-sum cohort. Execute the planned branches anyway, including empty masks; these are descriptive controls only.

## 3. Four branches, initially identical
NONE: no freeze.
TARGET: freeze actual updates to W1 rows and b1 entries in TARGET.
MATCHED: freeze n distinct non-TARGET units matched to TARGET by current-task |mean-ablation deltaCE_i| and outgoing W2-column norm.
RANDOM: freeze n distinct non-TARGET units sampled without replacement with an independent deterministic seed20260913+1000*seed+checkpoint.
Matching: each of the two features is converted to population percentile ranks, pair cost=sum of squared rank differences; a deterministic rectangular Hungarian implementation chooses the minimum-total-cost assignment (verified against brute-force small problems). Stable unit ordering resolves deterministic ties. Save assignments, costs and feature balance. Matching is based on the checkpoint's just-completed task and cannot use future outcomes.
Compute current-task mean-ablation deltaCE on the fixed512 probe; its activation is replaced with its current-task mean, with no fitting.
Adam moments continue to update by the ordinary rule for every parameter. After calculating the ordinary Adam update, masked W1 rows/b1 entries are restored exactly to their previous values. W2 and later parameters train normally. Masking only gradients is forbidden because it permits momentum drift.
The frozen IDs are not changed when a new task changes which units are shallow.

Secondary branches (predefined, regardless of TARGET existence):
SHALLOW freezes the25 SHALLOW units; SHALLOW_MATCHED freezes25 distinct non-SHALLOW units using the same two-feature minimum-cost matching procedure. These test the broad shallow-occupancy reserve without requiring small centered norm. They never substitute for the primary TARGET hypothesis. Save feature percentile differences and matching costs for both comparisons; describe poor matches explicitly rather than assuming comparability.

## 4. Endpoints and analysis fixed in advance
Selection/ablation uses512 images. The next512 images in the same deterministic test-index shuffle form a disjoint evaluation probe used for CE at steps0,20,100,300,625. At625 evaluate CE/accuracy on all9488 test images excluding the selection512. Candidate-selection images never enter either outcome evaluation.
Primary adaptation endpoint: trapezoidal CE area over the five timepoints divided by625, averaged over all10 continuation tasks. Lower is better. Report intervention minus NONE, and selective excess harm TARGET minus MATCHED, by seed/checkpoint/activation.
Primary hypothesis readout at checkpoint100 for ELU1: TARGET minus MATCHED >1e-5 CE in all3 seeds with all3 cohorts n>=3 -> DIRECTIONAL_PILOT_SUPPORT. Mixed signs -> MIXED. All effects <=1e-5 -> NO_SELECTIVE_RESERVE_EVIDENCE. Any insufficient cohort -> NOT_IDENTIFIABLE, while valid seed effects are retained.
This is an explicit directional pilot criterion, not a significance claim or evidence of zero effect. Report effect magnitudes, every seed, cohort sizes and matching balance; no null-equivalence claim.
Secondary: SHALLOW minus SHALLOW_MATCHED (broad shallow reserve, reported independently of TARGET existence); TARGET minus RANDOM; final CE and accuracy; task1 vs tasks2-10; checkpoint20 vs100; LR comparison. Cross-activation effect differences are descriptive because n may differ. Never infer ELU-specificity from unmatched numbers of frozen units.
Save per-task timepoint outcomes, seed summaries, and raw candidate/match diagnostics. Do not use early-minus-late training accuracy alone as the plasticity endpoint.

## 5. Checks required before substantive interpretation
A. Original uninstrumented training rule and no-mask implementation match exactly on a synthetic short stream and against C.loop over a one-task smoke continuation (same checkpoint, states and batches). Before branching at20/100, compare existing unit_triage_0911 ref per-unit cnorm, row mean, bias, zcur and sdcur exactly to establish the original checkpoint reproduction.
B. Branch parameters, initial logits, Adam state, and task RNG state are identical before the first update. Empty masks reproduce NONE exactly.
C. Every frozen W1 row and bias entry stays bit-identical throughout continuation; a nonzero-mask gradient-only mutation must be detected as drift by the test when momentum is nonzero.
D. All branches use identical saved task permutation/data/order hashes.
E. Candidate counts, set intersection and uniqueness; matched/random equal counts, disjoint from TARGET; IDs fixed over continuation.
F. Mean-ablation with outgoing W2 column zero gives deltaCE exactly0. All measurements have finite outputs. Original data files match T.DATA_SHA.
G. Preregistration commit and spec/code/data/module hashes recorded in provenance. Full training starts only after root supplies the committed and pushed preregistration revision.
Smoke checks may execute before registration but may not produce the registered long-run hypothesis endpoints. Failures get fixed and documented before launching.

