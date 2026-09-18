# CondA task switch: mean-input force and actual transport

Protocol: specs/spec_switch_force_0918.md, committed 355529e before these runs. Existing checkpoints and prior post-hoc analyses informed the question. Prospective measurement protocol on previously studied models, not an unseen-data confirmation.

Two leaky arms, 3 trained checkpoints, 10 seeds, 15 exhaustive possible switches, 32 exact support patterns; paired 10000-step switch/no-switch continuations in float64 and float32. MSE, lr=.005. No original RNG replay.

## Registered paired endpoints (float64)

New-task mean held fixed for both trajectories; weight-only response change. Bonferroni two-sided t intervals across 12 comparisons (family alpha .05), n=10 seeds. Negative paired effect alone is not absolute sinking.

| a | prior tasks | updates | switched | control | difference | adjusted interval | label |
|---|---:|---:|---:|---:|---:|---|---|
| LR_a0p1_q0 | 20 | 100 | -0.04071878 | 4.291629e-05 | -0.0407617 | [-0.07078779, -0.0107356] | SWITCH_ADDS_DOWNWARD |
| LR_a0p1_q0 | 20 | 10000 | -0.04540572 | 0.0003551702 | -0.04576089 | [-0.08856024, -0.002961549] | SWITCH_ADDS_DOWNWARD |
| LR_a0p1_q0 | 100 | 100 | -0.01502723 | 0.0001088618 | -0.01513609 | [-0.03867827, 0.00840609] | UNRESOLVED |
| LR_a0p1_q0 | 100 | 10000 | -0.001526245 | 0.001455032 | -0.002981277 | [-0.02700423, 0.02104168] | UNRESOLVED |
| LR_a0p1_q0 | 500 | 100 | -0.02081736 | 0.0003368718 | -0.02115423 | [-0.05142498, 0.009116519] | UNRESOLVED |
| LR_a0p1_q0 | 500 | 10000 | -0.003484645 | 0.003410716 | -0.006895362 | [-0.05771955, 0.04392883] | UNRESOLVED |
| LR_a0p7_q0 | 20 | 100 | -0.0006966362 | 0.0004512239 | -0.00114786 | [-0.01198849, 0.009692768] | UNRESOLVED |
| LR_a0p7_q0 | 20 | 10000 | 0.0007031915 | -0.000170007 | 0.0008731985 | [-0.04652905, 0.04827544] | UNRESOLVED |
| LR_a0p7_q0 | 100 | 100 | -0.005555036 | 0.001141318 | -0.006696353 | [-0.0143917, 0.0009989974] | UNRESOLVED |
| LR_a0p7_q0 | 100 | 10000 | 0.01213925 | -0.004368798 | 0.01650805 | [-0.02198724, 0.05500334] | UNRESOLVED |
| LR_a0p7_q0 | 500 | 100 | -0.007405068 | 0.001114119 | -0.008519186 | [-0.02047419, 0.003435815] | UNRESOLVED |
| LR_a0p7_q0 | 500 | 10000 | -0.0004898831 | -0.001008216 | 0.0005183325 | [-0.03778822, 0.03882488] | UNRESOLVED |

## Frozen switches: expected one-step force, all 15 flips

Each entry averages support patterns and units/flips within seed, then 10 seeds. Projection reference is the new mean for both old and new conditions. Source decomposition is algebraic, not a removal intervention.

| arm | prior tasks | old total | new total | old self | new self | old rest | new rest | input Shapley | target Shapley |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR_a0p1_q0 | 20 | 1.227421e-05 | -0.001066146 | -0.004097212 | -0.004724747 | 0.004109486 | 0.003658601 | -0.0005641816 | -0.0005142388 |
| LR_a0p1_q0 | 100 | -0.000107791 | -0.00083568 | -0.000415469 | -0.0004650606 | 0.000307678 | -0.0003706194 | -0.0002514317 | -0.0004764573 |
| LR_a0p1_q0 | 500 | 1.850333e-05 | -0.0002142695 | -0.0003409239 | -0.0003527546 | 0.0003594272 | 0.0001384851 | -8.353541e-05 | -0.0001492375 |
| LR_a0p7_q0 | 20 | 0.0004972448 | 0.0004824836 | -0.0001081024 | -0.000222361 | 0.0006053472 | 0.0007048446 | -3.923327e-05 | 2.447202e-05 |
| LR_a0p7_q0 | 100 | 0.0002044848 | -9.440464e-05 | -8.343565e-06 | -4.51422e-05 | 0.0002128284 | -4.926245e-05 | -0.0001537745 | -0.000145115 |
| LR_a0p7_q0 | 500 | -0.0001455396 | 0.0001370206 | -3.240147e-05 | -7.668341e-05 | -0.0001131381 | 0.000213704 | 0.0001011587 | 0.0001814016 |

## Full-task switched trajectory: separate input jump, W and bias

| arm | prior tasks | input jump | W learning | b learning | total z change | radial W | tangent W | endpoint length | endpoint direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR_a0p1_q0 | 20 | 0.04935361 | -0.04540572 | -0.004758931 | -0.0008110471 | -0.004110161 | -0.04129556 | -0.001035766 | -0.04436996 |
| LR_a0p1_q0 | 100 | -0.00612502 | -0.001526245 | -0.0003036256 | -0.00795489 | 0.003521127 | -0.005047372 | -0.0007757328 | -0.0007505122 |
| LR_a0p1_q0 | 500 | 0.03693759 | -0.003484645 | -0.000133444 | 0.0333195 | 0.00107125 | -0.004555895 | -0.0005455862 | -0.002939059 |
| LR_a0p7_q0 | 20 | 0.01858364 | 0.0007031915 | -0.0005392477 | 0.01874758 | 0.0004968475 | 0.000206344 | 0.0005724462 | 0.0001307453 |
| LR_a0p7_q0 | 100 | -0.008913407 | 0.01213925 | 0.001661864 | 0.004887712 | 0.005359665 | 0.00677959 | 0.0004752532 | 0.011664 |
| LR_a0p7_q0 | 500 | 0.001066598 | -0.0004898831 | -0.000628651 | -5.193623e-05 | 0.004652221 | -0.005142104 | 0.001044793 | -0.001534676 |

## Full-task source cancellation

| arm | prior tasks | self pos | self neg | rest pos | rest neg | net W | positive-branch total | negative-branch total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR_a0p1_q0 | 20 | -53.93953 | 9.471156 | 53.88975 | -9.466784 | -0.04540572 | -0.04977771 | 0.004371986 |
| LR_a0p1_q0 | 100 | -48.40222 | 40.98265 | 48.38406 | -40.96601 | -0.001526245 | -0.01816789 | 0.01664164 |
| LR_a0p1_q0 | 500 | -41.14067 | 41.97241 | 41.12096 | -41.95619 | -0.003484645 | -0.01970714 | 0.01622249 |
| LR_a0p7_q0 | 20 | -12.91363 | 10.17044 | 12.68514 | -9.941247 | 0.0007031915 | -0.2284934 | 0.2291966 |
| LR_a0p7_q0 | 100 | -16.08949 | 15.30447 | 15.535 | -14.73784 | 0.01213925 | -0.5544942 | 0.5666335 |
| LR_a0p7_q0 | 500 | -14.91537 | 14.55295 | 14.48175 | -14.11982 | -0.0004898831 | -0.4336203 | 0.4331304 |

## Reading of the registered and descriptive results

The adjusted primary test resolves added downward transport only for a=.1 after 20 prior tasks, at both 100 and 10000 updates. Other cells are unresolved, not evidence of zero effect. Native float32 yields the same 12 primary labels.
For a=.1, frozen switching makes the seed-mean force more negative at all three checkpoints. Only the 20- and 100-task new-force descriptive 95% intervals exclude zero; the 500-task interval crosses zero. Input changes and target changes both contribute under symmetric two-factor allocation. The residual/gate interaction is retained; changing the gate alone is not the complete explanation.
The 20-task a=.1 force changes through both a more negative self contribution and a weaker positive rest contribution. By 100 tasks the rest contribution changes sign; by 500 tasks it weakens while self changes little. A single fixed self-force story is insufficient across checkpoints.
The a=.1 20-task switched trajectory has W transport -0.045406, of which stepwise radial is -0.004110 and tangential is -0.041296. Exact symmetric endpoint allocation gives length -0.001036 and direction -0.044370. Only 50.5% of units have negative W transport: the population mean is not a universal per-unit drift.
In that same trajectory the input-only jump is +0.049354 and bias learning is -0.004759. Thus final z relative to the old-task starting mean changes by only -0.000811, despite negative learning transport. Downward adaptation after a changed input distribution is not automatically a downward staircase across task boundaries.
At later a=.1 checkpoints, the mean first falls then substantially recovers within the task. All six cells gain row-centered squared weight norm on average over the switched task, while mean-response transport differs in sign. Norm growth and signed mean transport remain distinct.
The duration of one task and three sampled ages cannot establish a persistent force or explain the complete 500-task drift. No new long-horizon run was performed.

## Verification and limits

Independent endpoint verification: PASS.
The first preflight failed because scalar torch.where branches made diagnostic leak constants float32; production update was correct. Original failed output is retained. Corrected code uses dtype-preserving branches; no thresholds changed. Main checks are in checks_main.json.
Primary inference concerns the added effect of a switch relative to no switch, at 100 and 10000 updates. Other windows, source allocations, branch breakdowns and geometry are descriptive/exploratory. The paired trajectories use one random flip per seed; all-flip robustness is tested only for frozen instantaneous forces. Results do not prove indefinite drift or transfer to CE/Adam.
Tables are seed means, not universal unit behavior. A whole-population mean can be driven by subsets. Tangential projection contributions are exact stepwise ledgers, not pure finite rotations. Endpoint length/direction allocation is symmetric and exact.
Hybrid input/target interventions are frozen diagnostics. Shapley allocations symmetrize two ordering choices, not a unique physical separation. The switch changes input, offset and teacher targets together. The environment-induced mean jump is not a weight update.
Native float32 sensitivity results, including all registered interval labels, are retained in primary.csv. Full unrounded seed/window data and exhaustive-factor estimates are in window_seed.csv and frozen_seed.csv.
