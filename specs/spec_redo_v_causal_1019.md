# ReDo: outgoing reset, positive one-sidedness and adaptation (1019)

Status: prospective protocol; commit before experimental runs. User request: verify with Claude API Fable and write results to Vault. Parent question is the mechanism of ReDo's residual adaptation deficit, not performance tuning.

## Evidence tier and hypotheses

Prior posthoc observations: the first-reset layer2 cohort becomes nearly always open on fixed probes; mean preactivation rises much more than its standard deviation. These are observations, not evidence that outgoing reset alone causes positive one-sidedness or that this one-sidedness mediates poorer performance.

H1: With the same selected units and freshly reset incoming weights, zero outgoing initialization produces more positive one-sidedness than a nonzero outgoing initialization.
H2: Restoring input-dependent gating at an already positive-sided checkpoint improves subsequent adaptation. H2 requires its own intervention; co-movement of accuracy and opening in H1 arms is not mediation evidence.

## A. First reset outgoing intervention

Use original engine src/baselines_cifar5p1_1008.py SHA256 c66760ba7c55af952adc76a5432edf3a6b765b86887e0fea3732fab8ae693049. Dataset path /home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz; SHA256 85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7.

Original 5+1 CIFAR100 real labels, 3072-100-100-100 ReLU MLP, Adam lr1e-4, 780 updates/task, ReDo tau0.1 every1560 updates. Seeds0-19, separate original stacks0-9 and10-19. First3 tasks, no new reset during task3. TF32 matmul false, highest precision, threads2. All interventions occur AFTER the complete first reset at tc1560 and affect only selected L2 outgoing W3 columns. Original W3 moments remain zero; all other values and optimizer state remain unchanged. Bias moments remain as in original implementation.

Arms:
- zero: original ReDo.
- keep: selected pre-reset W3 columns restored, with W3 moments still zero. This tests retention of old direction, not optimizer memory.
- rand01/rand03/rand10: independent class-centered Gaussian outgoing directions shared across these three arms; norms are respectively0.1,0.3,1.0 times each selected unit's OLD class-centered outgoing norm. New RNG stream per seed is separate from training and incoming initialization. Common logit component is removed.

These nonzero arms change initial predictions. Measure immediate upcoming-task CE/logit change; do not claim prediction-matched v-only mediation. The primary contrast is rand10 minus zero; the lower doses and keep are registered diagnostic contrasts. No choice of a winning arm from evaluation seeds for a headline superiority claim.

Primary H1 endpoint: per-seed fraction of fixed first-reset L2 cohort with opening >=0.99 on all2500 task3 training images at step50. Secondary endpoint same at step780. Also report fixed global1000-image probe, mean opening, strict full-open count, mean position, SD width, activation RMS/SD, class-centered outgoing norm, and p(1-p) gate variance. Fixed cohort, never condition on a post-intervention endpoint. Performance: task3 online accuracy averaged over all780 minibatches, train and test accuracy at end. Step1/2/5/10/25/50/100/200/400/780 trajectories.

If available, evaluate the exact gated CE split on fixed256 current-task examples: -dL/dz = gate*[v^T(y-p_minus_i)-v^T(p-p_minus_i)]. Report closure and two terms; these are raw local gradients, not Adam updates or full mean-position drift.

Statistics: seed is unit of replication (n20). Report paired mean differences and percentile95% bootstrap CIs (20,000 resamples, fixed analysis RNG1019). Primary meaningful suppression criterion: upper95%CI of rand10-zero high-open-fraction difference < -0.10. If the interval includes zero, unresolved; if entirely positive, opposite-direction evidence for this contrast. Report exact two-sided sign test as supporting descriptive evidence. Secondary contrasts remain secondary, with no uncorrected global claim from their minimum p value. Failure of one particular random direction/norm arm does not refute every possible v intervention.

Checks: zero replay first3 tasks matches archived baseline rows exactly. All arms match pre-reset state, masks, reset incoming rows, unchanged moments/parameters, and task3 minibatch hashes. No default RNG consumption by diagnostics. No NaN/infinite states. Verify CE split closure if computed.

## B. Forward-preserving change of the mean-input update route

Two additional arms start from the same complete first-reset state as zero. For the fixed selected layer2 cohort, write z_i = w_i dot (h - mu_i) + bprime_i, with bprime_i = old_b_i + w_i dot mu_i at intervention. Other units have mu_i=0. In center, mu_i is the postreset h1 mean on fixed global1000 training images (10/class, no labels used in the mean). In center_random, use an independent Gaussian vector of the same norm, shared within each seed's selected cohort. Fix mu for all subsequent updates. Independent RNG does not touch task or reset streams.

Initial z/logits should match original on global and task3 probes within float32 numerical tolerance; preserve the zero outgoing values and all original optimizer state policy, including retained bias moments. This is a new gradient coordinate system, not a claim of covariantly transporting full Adam history. Incoming weight moments are already zero. Record stored bprime separately from effective bias bprime-w dot mu. Changing upstream mean remains possible; do not claim a hard bound on Adam displacement or guaranteed prevention of full opening.

Primary operation check: same per-seed high99 endpoint at task3 step50; performance: task3 online accuracy. Comparisons center-zero AND center-center_random. Report both paired CIs. A supportive pattern requires high99 reductions (upper95%CI<0) and online accuracy improvements (paired mean at least +0.005 and lower95%CI>0) in both contrasts. If full opening is reduced but performance does not improve, state that the proposed rescue was not demonstrated; do not call every form of positive one-sidedness benign from a nonsignificant result. Because this intervention changes the mean-input route and optimizer geometry, even a supportive pattern does not uniquely identify nonlinear gating as the mediator.

## C. Positive homogeneous scaling sensitivity

Additional arms gauge4 and gauge025 follow zero through task3 update50. Then, for the same first-reset layer2 cohort, W2_i,b2_i are divided by c and outgoing W3_i multiplied by c, c=4 or0.25. This preserves the represented function and gates at that instant. Transform gradient-history moments by the inverse coordinate scaling: m_W,m_b times c; second moments times c squared; m_V divided by c; second moments divided by c squared. Global Adam clock unchanged. Record before/after logits and gates. Verify raw gradient chain rule on a synthetic example before production.

Continue the same730 remaining task3 updates. Report online accuracy over steps51-780, task3 end train/test accuracy, and subsequent gate trajectories. First50 steps must match zero exactly. This tests dependence on the activation/readout parameterization under Adam; it is not a pure v intervention, nor a gate-preserving constraint for the whole future trajectory. A positive effect does not exclude an additional gating pathway; a null does not refute all optimization-geometry mechanisms. Initial gate/function invariance is a required check.

## Scope and Fable audit

This protocol covers the first recycling episode, not a full30-task policy comparison. Task3 has a small descriptive original ReDo-no-reset online deficit, while task29 ReDo outperforms no-reset; the larger long-term shortfall discussed in the paper is relative to stronger activation baselines. Do not conflate these comparisons. Tasks3-8 would cross further resets, so they are deliberately excluded from the matched-cohort primary test. Initial task1/2 full-open counts are a descriptive stage control only, since the h1 representation differs from the mature reset state.

Fable round1 suggested the norm ladder, input-mean reparameterization and gauge control. Parent corrected its proposed Adam moment transformation (moments follow gradient covectors, not parameter scaling), its claimed Adam learning-rate displacement bound, the reset timing, and overly strong null-result interpretations. Round2 accepted these corrections and the limited design. Its further approximate first-step Adam statement is not used: retained global bias correction changes the step multiplier after moment resets. No weight decay is used. Step50 has a known high baseline near99 fraction from the earlier replay and is not chosen using treatment results. A and B are interventions; C and the gradient decomposition are complementary diagnostics. No result alone establishes the complete v-reset -> positive-sidedness -> performance causal chain.

## Reporting

Save spec, source, provenance, all seeds, checks, Fable prompts/responses and conclusions. Report supportive and contradictory results. New results go to 可塑性喪失/測定/ in Vault; parent results notes link to them. Keep theory, observations, interventions and remaining arrows explicit. Archive raw outputs before removing worktree; preserve original source and unrelated worktrees.
