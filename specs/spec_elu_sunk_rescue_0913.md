# ELU sunk-unit accumulation and one-shot rescue 0913
Status: PROSPECTIVE INTERVENTION PREREGISTRATION / retrospective reanalysis explicitly labelled.
Owner: user-requested follow-up to elu_environment_0913; parent: W growth and lost learning capacity.
Created 2026-09-13. Prior results known: RL ELU collapsed by task10, PM ceiling, layer2 low gates. These motivated this follow-up; there is no claim of blind environment selection.
Root controls scientific design; GPT-5.6 Sol assists implementation, data handling and independent audit.

## 1. Questions and scope
Does the stock of persistently low-gate units grow, do those exact units leave that state on new tasks, and does lifting them once restore their contribution to learning new tasks?
A flat low performance level can represent sustained lost plasticity. Early-minus-late decline is NOT the primary rescue endpoint.
Low-gate occupancy is a state diagnostic, not proof of functional death. Finite non-recovery is not mathematical irreversibility.

Original environment protocol unchanged: MNIST1200 fixed training images/seed, MLP784-100-100-10, ELU alpha1 both hidden layers, Adam(.001,.9,.999,1e-8), batch16,80epochs=6000updates/task; 3seeds0,1,2; PM and RandomLabel (RL).
No new activations/environments/budgets. Primary causal setting is RL, layer2, checkpoint20, full dose (up to20 units). Other setting/checkpoint/layer contrasts are prespecified secondary.
Use checkpoints at end tasks10,20,50. Each branches into the next5 tasks,6000updates/task. Each layer studied separately (layer1 or layer2); no joint layer rescue.
Reconstruct the original 24-model batched trajectory (including original leaky/clamp models for exact numerical geometry) with the unchanged original Engine. Save checkpoint params, Adam, counter, RNG states, last inputs/labels/permutation and subset IDs; only ELU reference models branch.
Prefix gates/norms/means must reproduce old units.npz at all50 endpoints, with exact equality targeted and maxabs <=1e-6 accepted only with explanation. Final reference parameters/moments vs old checkpoint.pt <=1e-6.
Original raw location: /home/issan/Projects/claude/elu_reserve_0913/results/elu_environment_0913/.
Original experiment code source commit46becd95b992d1ce27f95a6dc5b9cbdd2ad73f63; worktree root /home/issan/Projects/claude/elu_sunk_rescue_0913.

## 2. Retrospective identity tracking (no causal label)
On old PM/RL ELU/leaky ref/wclamp data, track every unit ID in both layers at tasks1-50. q_i(t)=fraction of the1200 task inputs with phi'(z_i)<.05. Sink-state S_i(t)=[q_i(t)>=.95].
Report per-task occupancy count, entries false->true, exits true->false, and persistent count S true at each of latest5 endpoints (available t>=5).
Verify N(t)=N(t-1)+entries-exits, with N(0)=0 reported only as initialization convention.
For each entry at t<=45, report whether that same ID exits S within next5 task endpoints; use identical follow-up horizon, no censoring late entries into apparent non-recovery. Also report stronger gate recovery q<.5 within5tasks. Recurrent entries allowed but never treated as independent experimental replicas.
Store full unit-state transition table and seed summaries. Low-gate counts for leaky .1 are structurally zero at this gate threshold; do not infer leaky has no LoP. Cross-activation functional conclusions come from actual learning, not this count.
A leave-and-return episode counts as recovered within the finite window even if the unit later sinks again. Task-end data cannot exclude within-task transient recovery.
Any stock/performance association is descriptive. Do not infer causation, or use ever-sunk cumulative count as evidence of accumulation.

## 3. Selection uses past only
For ELU ref at checkpoint T, persistent candidate pool P = IDs with q>=.95 at EVERY endpoint T-4,...,T, in the selected layer.
Use all1200 task inputs for state measurement, without labels in selection. New-task random labels and all future metrics are excluded from selection.
Order P with a dedicated torch generator seeded by SHA256 of 'elu_sunk_rescue_0913|target|seed|env|T|layer'. Primary target = first min(20,|P|) IDs; doses5 and10 are nested prefixes capped at pool size.
If |P|<5: COHORT_TOO_SMALL and rescue mechanism NOT_IDENTIFIABLE; execute descriptive empty/small branches, do not enlarge threshold or substitute another cohort.
For the full-dose control, sample the same number of IDs from all units outside the selected primary target using separate role 'control' generator. This can include other persistent units when the whole layer is saturated. Report that overlap and incoming norm/outgoing norm/current mean distribution; no claim of a nonsunk matched control if overlap exists.
IDs remain fixed across5 continuation tasks. No replenishment and no selection based on future recovery.
Control tests a same-count intervention at other IDs; it is not a perfect propensity match or pure nonsunk comparison.

## 4. One-shot lift and actual-update freeze
For target IDs compute delta_i = max(0, -1 - mean_x z_i) on the checkpoint's just-completed task1200 inputs. Add delta_i once to their bias in the selected layer, immediately before task T+1. No further lift or depth clamp.
All W matrices and all Adam moments/counter are bit-identical before vs after the lift. Only selected bias entries change; no outgoing compensation or optimizer reset.
Dose5/10 use the corresponding prefix deltas. Random-control lift applies the SAME delta multiset to its sampled IDs in order, so magnitude/count are preserved, not the achieved depth. Report its immediate shock; high shock makes specificity weak.
Freeze = block actual updates to selected incoming W rows, their biases, AND outgoing W columns. Other entries update normally. Adam moments continue updating for all parameters. Frozen values are restored to their post-intervention values after each Adam step. This freezes incident parameters, not the unit's input when earlier layers learn.
Masks are fixed, graph-compatible; frozen references reset only at branch start. Do not mask only gradients.
Eleven branches per seed/env/T/layer:
 A: deep, train (no intervention)
 B20: deep, freeze target20
 C20: lifted target20, train
 D20: lifted target20, freeze target20
 C5,D5: lifted dose5, train/freeze
 C10,D10: lifted dose10, train/freeze
 RB20: deep, freeze random control
 RC20: lift random control, train
 RD20: lift random control, freeze random control
There are 2env x3T x2layers x3seeds x11branches=396 continuations, each5tasks. Capped duplicate doses remain explicitly duplicates, not independent evidence.
Compare A vs B20 and C20 vs D20: within each pair initial functions are exactly identical, including Adam and task/data RNG. Lifting DOES change the function across pairs; report that shock instead of pretending to preserve it.

## 5. Outcomes and registered contrasts
Every future task uses identical input/permutation, target labels and minibatch orders across corresponding branches. Continue exact original RNG streams saved at T. Use same subsequent task sequence also for different layers.
Evaluate full1200 current-task train/memorization CE and accuracy at step0,75,375,1500,3000,6000. This measures learning fresh assignments under fixed budget, not RL generalization to unseen image labels.
Save pre-lift/current-task and post-lift/current-task metrics; future-task step0 separately.
Primary AUC: trapezoidal CE area over these six points /6000, on FIRST future task. This is absolute trajectory cost, not a pure speed measure; initial levels are matched within freeze pairs.
Primary restoration interaction R20 = (AUC_D20-AUC_C20) - (AUC_B20-AUC_A). Positive means learning via the incident parameters becomes more useful after lifting.
Also direct rescue G20=AUC_A-AUC_C20. Positive means lift+learning improves new-task trajectory overall.
Registered primary pilot at RL/layer2/T20:
 all3 pools>=5 and all3 R20>.01 CE and all3 G20>.01 CE -> DIRECTIONAL_RESCUE_AND_UPDATE_SUPPORT.
 all3 pools>=5 and all3 G20>.01 but not all R20>.01 -> RESCUE_WITHOUT_SELECTIVE_UPDATE_SUPPORT.
 mixed signs / other -> INCONCLUSIVE; insufficient pool anyseed -> NOT_IDENTIFIABLE.
These are pilot directional thresholds, not equivalence or population significance.
Report each seed, mean and t95% df2 for R,G and primitive contrasts. No task/unit pseudoreplication, no dropping bad seeds/large shocks.
Secondary: all5tasks average AUC and endpoint accuracy/CE, task-specific curves; endpoint restoration interaction with accuracy sign reversed appropriately; dose5/10 R_k=(Dk-Ck)-(B20-A) is NOT a matched dose interaction because deep freeze differs. Therefore report dose-specific Ck-Dk update benefit and A-Ck direct rescue only, and compare these across nested doses descriptively, with exact n.
Random control interaction R_random=(RD20-RC20)-(RB20-A); report R20-R_random, Gtarget-Grandom and overlap of random group with persistent pool.
Report persistent candidates remaining sunk, q, mean z, gate mean, actual incident deltaW/deltab and per-input mean/SD delta z between measurement points; norm trajectories both layers; immediate forward shock.
Restoration that occurs despite frozen target params is a representational/readout pathway, not evidence against lost capacity. Absence of R or G is not proof that sinking is harmless: one-shot lift can fail to sustain recovery, affect useful features, or be blocked by another layer.
After lift, W norms can evolve differently. This identifies total one-shot lift effects and update dependence, NOT a W-independent mediated fraction of natural LoP.

## 6. Technical gates before full intervention run
- Original Engine unchanged and source SHA recorded. New engine empty masks matches original no-clamp Engine p/m/v exactly on synthetic same-shape trajectory with nonzero Adam time.
- Actual frozen incident parameters remain bit-identical to branch-start values; gradient-only masking mutation with momentum must drift.
- Nonfrozen entries and all moments match the corresponding ordinary step.
- Active-freeze graph/eager trajectory maxabs<5e-5 with nonzero counter; counter advances25; empty-mask graph must match.
- Original prefix checks as above; subset/permutation/label/order hashes match old records through50. Fresh tasks51-55 are deterministic continuations of those streams.
- Within A/B20 and C20/D20 (also C5/D5,C10/D10,RC20/RD20) initial params/moments/logits exactly equal. All bias-lift changes confined to designated biases, all W/Adam unchanged.
- Lift reaches selected mean z=-1 within2e-4 for positive deltas on the old task; finite diagnostics; empty pool no-op.
- Branch freeze invariants checked at each measurement and final; all finite, masks/pool counts/nesting/control disjointness checked.
- Models independent, gradients averaged over batch only; sum over models is not averaged again. Prior original Engine gradient validation retained.
- Source/spec/data/RNG hashes, launch commit, all exceptions saved. Prereg and implementation committed and pushed before intervention runs. Synthetic technical smoke allowed first.
- No silent tolerance relaxation. Technical amendments, if needed, registered before rerunning failed substantive work.

## 7. Storage and reporting
specs/spec_elu_sunk_rescue_0913.md
src/elu_sunk_engine_0913.py; src/elu_sunk_rescue_0913.py; src/elu_sunk_tracking_0913.py
results/elu_sunk_rescue_0913/ and results/elu_sunk_tracking_0913/
Tables, figures, code, provenance committed. Raw pt/npz stored both original worktree and verified SHA256 backup at /home/issan/Projects/obsidian-research-data/elu_sunk_rescue_0913/files/ with manifest.
Obsidian result under 可塑性喪失/測定/, spec overview moved to 実行済み after completion. Existing result notes preserved, clarification that flat damaged state can remain LoP carried into new result/current context.
No new unrequested environment or follow-up run is automatically launched from unexpected results.

