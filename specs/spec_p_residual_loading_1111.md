# P後に残るhead依存の適応費用：MLP–ViT直接比較

Preregistration, 2026-11-11. Experiment p_residual_loading_1111. Base 23986ee4342d40441504c3e0e685cc0194d58191. Written before any new trained-cell smoke/main execution.

## Question and scope
Issa asks whether the part of the remaining learning delay after destroying alignment (P) that disappears on removing the readout is of the same kind as ViT loading cost. Test an operational commonality: inherited head parameters can prolong adaptation even after P, with the body and its optimizer state fixed at the fork. A shared intervention response does not establish identical microscopic mechanisms or that all residual delay resides in the head. P independently permutes class entries within each head column and compensates bias to preserve sample mean logits; it destroys a particular alignment, not all possible old information.

Prior-data-informed design: stage1 existing full curves were inspected before registration. Early P outperforms H0 for R/ELU/SiLU. At task-end10 ELU, P ends at .7024 vs H0 .9639; R and SiLU remain censored. Those known contrasts are reproduction/descriptive evidence, not newly held-out confirmation. New information is the head-parameter × head-Adam factorial, delayed forks, matched ViT comparison, and removal after a Pall history. Previous loading_bridge recovery reset ALL Adam and restarted batches; it does not answer this question and is not pooled here.

## Fixed sample and forks
- MLP R/ELU/SILU, seeds0..9, S-history task ends1,10, next tasks2,11: 60 source cells. Reconstruct canonical body AND Adam, verify saved task-end weights/labels exactly.
- ViT SL, final affine LN, two blocks, seeds10252010..10252014, task-end100 to101: 5 cells, original saved model/Adam/RNG, same2500examples and active10class rows.
- Each: S and P at tau=0, four head factorial arms; P after25updates forks again into four arms at tau=25. No S tau25; do not claim a matched S/P temporal comparison.
- MLP ELU Pall history, seeds0..9, end29 to30: P tau0 four arms only. Replay canonical persistent-P tasks1..29 and compare every stored record field exactly; no saved endpoint weights exist for this history, so do not claim an independent bitwise weight reference. This is a representative check, not a sweep over Pall activations/ages.
- 820 logical curves. S/P tau0 ZC coincide exactly (65 shared trajectories). P tau25 KK is the suffix of P tau0 KK (65 more shared), giving690 executed trajectories. Shared curves are identities, not extra replication.

## Interventions
Two letters = head parameters then head optimizer memory. KK: keep/keep; KC: keep/clear; ZK: zero/keep; ZC: zero/clear. Zero both weight and bias, including all ViT output rows. Clear only head Adam state (both parameters, including steps); always retain all body parameters and body Adam state. ZK deliberately retains possibly displaced old moments and is a diagnostic, not a standard reset recommendation. Head includes weight AND bias: claims concern the joint readout parameters.

MLP single P uses canonical stage1 RNG own('arm|P|adam|{next_task}',seed). Pall uses own('persist|P|adam|{next_task}',seed). ViT P uses frozen loading_bridge.static_probe.rng('P',seed,t_end,'ViT','SL'). P permutes W's first/second moments with entries, leaves bias moments, and compensates bias in float32. Preserve column multisets and body state hashes. P-ZC = S-ZC at tau0 must have identical parameter AND optimizer-state hashes, and reuse the common curve. P-ZK differs from S-ZK and must not be shared. At each fork verify body parameter/state hashes unchanged by factorial and the reference fork hash common to four arms.

## Training and recording
MLP4000 total next-task updates, ViT1500. tau25 branches continue the SAME batch sequence starting index25, to the same absolute endpoint; they do not restart. No early stopping at90%. Train loss/optimizer/data follow original engines. MLP CPU single-thread; ViT compiled forward with deterministic settings as frozen loading_bridge helper. CPU replay checks use exact original operations. Runtime source hashes and git commit recorded.

Evaluation: full fixed training set at0,1,25,26, every100MLP/every25ViT, endpoint; for tau25 use suffix points>=25. Record acc, CE, old acc/margin, head centered norm, confidence/entropy and supporting feature summaries. MLP primary acc/CE use canonical float32 evaluate (and can match old eval exactly); ViT uses deterministic full-set feature/head evaluation as frozen helper. Body unit/layer summaries atfork,25,200,endpoint only. Evaluation frequency is common across paired arms. Record finite values and no update side effects. No test/generalization claim.

## Estimands and decision
Primary U = trapezoidal integral of (1−full-train-acc) over [tau,limit], divided by (limit−tau): mean accuracy deficit, lower better. Primary parameter effect D_C=U(KC)−U(ZC), with head Adam identically empty at fork. Also D_K=U(KK)−U(ZK), memory effects U(KK)−U(KC) and U(ZK)−U(ZC), joint U(KK)−U(ZC), interaction D_K−D_C. Positive means removing the indicated inheritance reduces cost. Report normalized CE area and final acc/CE separately. Effect is an intervention contrast, not a causal mediation fraction or a recovered fraction of total old-task damage.

Primary strata are ELU S-history end10 Ptau0 and ViT end100 Ptau0; selection explicitly informed by previous data. Shared positive D_C in both, with each paired-seed bootstrap95% lower bound>0, supports head-parameter-dependent loading cost in this operational sense; lower>.01 in both indicates practical strength (1percentage point averaged over the trajectory). A clearly negative contrast in either rejects that shared pattern for these conditions. Other cases are unresolved. Report D_K and optimizer interactions; a positive joint reset alone is insufficient to assign to head parameters. Report all controls, not only supportive ones. Same signs do not establish equivalence of magnitudes across architectures or a common molecular/gradient mechanism. tau25 and Pall are prespecified extensions/boundaries.

Bootstrap10000 paired seed resamples, deterministic seed, percentile95%; n10MLP/n5ViT. These are descriptive seed variability intervals, no broad population claim or multiplicity-adjusted discovery. Ratio D_P/D_S only if denominator D_S bootstrap lower>.01; otherwise omit as unstable. No cross-architecture update-count equivalence.

T90 first recorded crossing and recorded sustained T90 (first observed point after which every later observation is>=.9), both absolute and elapsed. Censor=null, horizon separate; never replace with4000/1500 as observed. Report counts and full-seed paired means only when every seed in both arms reaches. Sustained at endpoint alone is one-point evidence, report supporting length. Supplementary body markers describe remaining state, not permanent injury.

## Validation and resources
Synthetic toy factorial: keep/clear/zero behavior, body hash invariance, S/P-ZC equality, P moment transport. Smoke one early MLP and one ViT up to50updates, outside main output, only implementation checks; commit this spec before smoke and runner before main. Replay mismatches stop affected cells. Identical helper/spec hashes required on resume; do not mix incompatible outputs.

Fixed design cap: 2GPU hours, 4CPU core-hours, <5GB raw, <=4single-thread CPU workers and1GPU worker. If cap/error blocks completion, report incomplete cells and no complete verdict, seek no outcome-driven seed truncation. Monitor at most once/5min unless near expected completion or an immediate error. Save all raw and code, source hashes, report, plots, backup manifest; merge/push main and remove only this worktree when complete. Record any implementation correction or deviation before affected rerun.
