# Demand, saturation, and readout shrinkage (demand_allocation_1111)

Status: preregistered by the commit containing this file, before new training/diagnostic results.
Date: 2026-11-11 JST. Owner: Codex, hidden-demand investigation.
Parent question: why can activity rise after readout shrinkage, while saturated units fail to recover?
This experiment owns only `src/demand_allocation_1111.py`, `analysis/demand_allocation_1111/`, and `results/demand_allocation_1111/`. It does not edit the manuscript or other experiments.

## Existing evidence and the gap

The 10-seed stage1_ctrl_1111 experiment already measures repeated Q interventions (centered readout column norms restored to initialization with mean-logit preserving bias) and 30-task prevention. Its endpoint data will be reused, explicitly as a post-hoc audit of a preregistered earlier experiment. No new claim of having preregistered those data. The September vfreeze experiment has two seeds, V multiplied by .1/1/10 then frozen at task 51, and windows 151–200. Those results are long-after-intervention results, not instantaneous gradients. The fixed-target two-group model is conditional and does not fix the logits of our experiment.

The missing measurements are input/unit activity demand `D=e V`, the implemented derivative, actual optimizer displacement, source/receiver interference, and allocation between hidden activity and readout. We will reconstruct optimizer histories rather than attach fresh Adam states to saved weights.

## Box and budget

Use the exact `src/stage1_ctrl_1111.py` engine helpers from origin/main 14402c91. CPU float32, 1 thread per process, MNIST raw/255, fixed 1,200 inputs, 784-100-10, random labels per task, batch 16, 4,000 steps/task, Adam lr=.001, betas=(.9,.999), eps=1e-8, no weight decay.
Activations: R, ELU, SILU, GELU, LR(.1). Seeds 0,1,2, paired throughout. Checkpoint ends 1,9,24; new tasks 2,10,25. All baseline end parameters through task 25 must match saved stage1 states bit for bit; if not, halt and document the mismatch before interpreting.
Use existing input/label/order RNG streams. Diagnostics must not advance training RNG or update training optimizer state.
Maximum new training: 15 baseline trajectories to task 25, 225 single-task branches (S,Q,QV,QF,QA), plus 30 QH branches (ELU only at 3 checkpoints, 3 seeds = 9, not 30; 225 includes S baseline duplication), and one next task for S and Q for every state (90 task-equivalents). Total ceiling 720 task-equivalents / 2.88M steps; implementation may reuse already computed S to reduce cost. At most 2 concurrent single-thread processes, nice priority, no GPU/cloud/paid models. A wall-time budget of 90 minutes for the new experiment; if exceeded, report incomplete cells without substituting seeds or selecting endpoints. Store checkpoints, CSV diagnostics, per-cell provenance, all failures. Smoke uses seed 900, 30 steps on synthetic data, not inference data.

## Arms, all forked from the same exact state

- S: unchanged Adam and parameters.
- Q: exact stage1 Q: centered V columns restored to their initial norms; class-common component retained; b_out += (V_old-V_new) mean(a); all moments retained.
- QV: Q then freeze V (gradient None, its optimizer state also held). Output bias still learns. Separates allocation to V from hidden/bias.
- QF: Q then freeze both hidden W and hidden bias. V and output bias learn.
- QA: Q then discard only hidden W/bias Adam states.
- QH (ELU only): Q then discard only V/output-bias Adam states.

Each arm fits exactly one new task. S and Q then fit one further new-label task without another intervention, to measure reload. QV/QF/QA/QH stop after their first task. We explicitly do not equate this with repeated-Q prevention. For instantaneous diagnostics only also test V*.1 with output bias unchanged (Craw) and V*.1 with mean-logit-preserving bias (Cmean). These are not trajectory arms.

## Times, demands, and masks

Measure steps 0,1,10,50,200,500,1000,2000,4000 of the first task. Next task (S/Q): 0,200,4000. Full input set has its actual newly drawn labels; also measure old labels and independent-uniform-label expected demand at step 0. `D_ni = sum_c e_nc V_ci` is the unreduced per-example activity gradient; -D>0 means activity-increase demand, not positive z demand. `G_ni = implemented phi'(z_ni)`, `delta=D*G`. Batch-mean parameter gradients include the 1/B factor.

Current groups: open z>0; band -3<z<=0; deep z<=-3. Additional independent derivative-sign groups G<0, G=0, 0<|G|<1e-6, |G|>=1e-6; unit groups exactly all-input-closed and all-input derivative small. Fixed initial open/band/deep masks follow the same input/unit pairs, never redefine them based on final survival. All masks report counts; empty means/ratios are NA. All-input closed is not synonymous with dead for smooth activations or leaky ReLU. ReLU positive branch has derivative 1 and must not be called saturated.

Outcomes per seed/checkpoint/arm/time:
1. CE, accuracy, residual norm, confidence, logit centered RMS, V centered norm, hidden W/bias norms; z mean/SD/r, max and q99, open counts per unit and per input, exact-closed-unit fraction, derivative-small-unit fraction.
2. Per group: D sign fractions, mean and RMS, mean positive increase/decrease demand, transmitted delta RMS, exact-zero derivative fraction, current and fixed-mask z/activity movements; fixed-negative to positive transitions. Save per-unit arrays as well as pooled summaries. No unit-as-seed inference.
3. At step 0 explicitly decompose D_Q-D_S = e_S (V_Q-V_S) + (e_Q-e_S) V_Q; report norms, cosine, sign changes, not an e-fixed causal trajectory. Fixed-label and uniform-expectation are separate.
4. At each diagnostic time simulate, on a clone only, one update using the next scheduled minibatch (step 4000 uses a fixed first minibatch for a probe with the current label). Compute exact parameter and z displacement; additive hidden contribution X DeltaW^T + Deltab. Compare the current demand with actual dz/da, including opposing signs.
5. Decompose the Adam hidden step with the *actual new denominator held fixed*: old first moment, current open/band/deep gradient sources. This is exact conditional bookkeeping, not a causal deletion experiment because deleting a source would change the denominator. Record source contributions to full-input mean dz and fixed receiver groups, and compare sum to the step including float32 rounding. At step 0 additionally compare fresh-Adam and SGD lr=.001 clones, no learning-rate performance claim.
6. One-step logits: Delta l = Delta a V_old^T + a_old DeltaV^T + Delta a DeltaV^T + Delta b_out. Record signed -mean(e*each contribution), norm, total, exact CE change; cross term explicit. These are local first-order allocations, not percentages of eventual performance. End-state finite counterfactual logits also logged with original and final (a,V,b).
7. At 0,500,4000 estimate V-gradient mean/noise from 64 predetermined minibatches (same across arms; no updates). ELU identity gV = mean[e (a+1)^T] - mean[e] 1^T: record common-component variance, remainder variance, covariance, total variance (do not add variances without covariance). Output-bias gradient is mean[e]. Also record moments and actual V step norm/cosine; large V noise alone does not prove suppressed drift.

Training uses autograd of exact existing activation, including float32 `expm1` derivative floor; separately record analytic float64 derivative on the same float32 z to distinguish mathematical smallness from exact implemented zeros. No silent activation correction.

## Predictions and rejection conditions (3 paired seeds, descriptive)

P1: At task 25, all-input-closed ReLU units still receive both increase and decrease activity demands in S (increase-demand fraction >.05), but raw hidden gradients are exactly zero. Adam residual memory may move them; report it, do not assume immobility from the instantaneous derivative. Reject demand-absent explanation if P1 holds. If no closed units, mark unavailable.
P2: Q does not universally increase instantaneous activity-increase demand. At least one activation/checkpoint has smaller mean positive increase-demand magnitude than S, despite Q improving later activity or performance. Universal instant-amplification explanation is rejected by any paired counterexample in all 3 seeds.
P3: For ELU at task 10, QV ends with greater current top z or more input/unit openings than S in all 3 seeds, while no universal 1/c restoration of logits is required. Failure in any seed marks mixed, reverse in all rejects. This tests available feature/bias response when V cannot regrow.
P4: At task 25 Q fails to revive fully closed ReLU, whereas Q prevention in existing 30-task data retains openings and much better accuracy. If >5% of initially fully closed ReLU units reopen in all 3 seeds, reject the numerical absorption prediction and identify memory-driven crossings.
P5: At task 10 or 25, early hidden update RMS (first probe) in QA exceeds Q by factor >2 for ELU in all 3 seeds; sign is unrestricted. If not, mark mixed/rejected. This tests immediate history suppression only, not the assertion that Adam history explains all LoP.
P6: Q uses both hidden and V changes during fitting; at task 10, QF and QV have different CE/activity/V trajectories from Q, and neither frozen arm must reproduce joint Q. Report all arm contrasts, no preclaimed dominant route. For a stronger proposed explanation that *only* activity compensates, any Q cell with positive V norm regrowth and nonzero V first-order descent in all seeds is a counterexample.
P7: Current-source contributions to closed receiver dz are nonzero even where receiver derivatives are exactly zero, if the same unit has some open inputs. Validate the exact shared-weight identity; this is algebraic verification, not independent evidence for the phenomenon. Report empirical magnitudes without tautological significance tests.

Report each prediction supported / mixed / contradicted / unavailable, with raw paired values. No p-values or bootstrap confidence intervals with n=3; seed min/max and means, same checkpoint and time only. Existing 10-seed long trajectory outputs retain their original inference tier. No stopping based on favorable findings, new selection of seeds, or new unregistered training arms.

## Checks, provenance, delivery

Independent double-precision gradient formulas versus autograd and central differences away from nondifferentiable points; bias/W/V included. Adam manual formula versus torch including seeded moments and eps; gradient scaling with consistent vs stale history. Float32 exact step accounting errors normalized by max(1,parameter scale), threshold 2e-6 (z/logit accounting 2e-5); double algebra tolerance 1e-9. Baseline bit equality required. Diagnoses that fail checks cannot support claims.
Commit spec before execution, then implementation before main execution. Include SHA256 inputs/code, git hash, command, torch version, seeds, output completeness, discrepancies. Reproduce September vfreeze reported numbers/windows using its source aggregator (post-hoc audit); read existing stage1 judge and describe_T with exact definitions. Preserve priors and mark amendments explicitly rather than rewrite this file after results. Save raw outputs to obsidian-research-data/demand_allocation_1111 with backup_manifest; merge/push into main without rewriting history and clean own worktree. Japanese result note in existing vault `測定/` via MCP, include equations/limitations and links. No current-manuscript changes.

## Infrastructure addendum — 2026-11-11 05:40 UTC, before aggregate result inspection

The initiating command returned exit 143; process inspection confirms no remaining demand_allocation process. No Python traceback was seen in the log. Eleven complete jobs have provenance and all 25 baseline checks; GELU seed 2 and LR seed 0 have partial output, LR seeds 1 and 2 have not started. Preserve the two partial directories and original log, retain the eleven complete cells, and rerun exactly the four incomplete cells from their original initial states with the unchanged training source and engine SHA256. Do not replace seeds, change arms, endpoints, predictions, or thresholds. This is an infrastructure recovery, not a response to observed outcome values.

The maximum repeated work is two incomplete 46-task jobs. Raise only the computation ceiling from 720 to 800 task-equivalents (3.2M updates); original 699 plus at most 92 wasted task-equivalents is at most 791. The 90-minute wall budget from original launch and two single-thread workers remain. Record the interrupted status, preserved partial paths, and completion in recovery.json. Detached launch avoids dependence on the ended interactive execution session. Aggregate only after all fifteen complete cells are present.
