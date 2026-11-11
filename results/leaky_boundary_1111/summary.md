# leaky_boundary_1111 — boundary anatomy and a finite adaptation clock

Completed 2026-11-11. Registered spec `specs/spec_leaky_boundary_1111.md`, spec commit `6d4eed96`, training/spectral code commit `f890ea93e86746e3668ca9f45808f36f4855675e`. Five NEW a=.003 seeds3–7, fixed1200 MNIST images with independently renewed random labels,784–100–100–10, Adam.001,6000 updates/task.70 common-task probes and35 source spectra completed. All numbers below are derived by `analysis/leaky_boundary_1111/report.py`; source tables are named at the point of use.

## Answer to the boundary question

In this new setting, the evidence favors a **gradually lengthening adaptation clock crossing the6000-update task budget**, with a concrete deterioration of input-dependent activation patterns during relearning. It does not support the registered abrupt-slowing criterion. This is stronger than saying merely that more training eventually recovers: the same probe task is used on both sides, the actual path between sources is sampled, and the CE learning curves approximately collapse after time rescaling in all5 seeds. The remembered CondA figure has not been identified; these findings do not establish its mechanism or a universal threshold at mean derivative.1.

Sequential online accuracy fell from69.67% on task3 to45.96% on task4; the largest adjacent drop was task3→4 in all5 seeds (drops0.188–0.298). The relevant SOURCE states are ages2 and3, not end-of-task3 and4. Source quarter states are actual states with their original Adam moments, not weight interpolations. See `prefix_tasks.csv`, `boundary_selection.csv`.

## Same-task clocks and controls

| arm | source | T90 mean | accuracy6k | accuracy24k | accuracy96k |
| --- | --- | --- | --- | --- | --- |
| G | post | 1320 | 99.88% | 100.00% | 100.00% |
| G | pre | 795 | 100.00% | 100.00% | 100.00% |
| L | post | 915 | 99.83% | 100.00% | 98.68% |
| L | pre | 570 | 100.00% | 100.00% | 100.00% |
| N | post | 7905 | 82.22% | 99.62% | 100.00% |
| N | pre | 3585 | 98.12% | 100.00% | 100.00% |

N is the natural network. G fixes each image's original slope masks, with trainable weights. L trains the exact linearized model using the original full Jacobian. All begin with identical forward values, minibatch gradients, actual first parameter update, labels, minibatch order, and Adam history. No head or optimizer reset. G/L are diagnostic surrogate function families; this comparison cannot by itself identify an intervention achievable within the original network. See `probe_seed_summary.csv`, `endpoint_aggregate.csv`, `integrity_audit.json`.

Native T90 mean increased from3585 to7905; the post sources needed6000–10875 updates. Four exceeded the6k budget; one reached90% exactly at6k. All10 selected pre/post native sources reached90% within24k and were100% at96k. The other native source at age6, seed5, did not reach90% within its24000-update horizon (endpoint89.25%); its T90 is right-censored, not an excluded seed. G post reached90% in mean1320 updates; L post in915. The fixed-gate advantage at6k was17.67 percentage points (paired95% t interval7.47–27.86); its extra advantage post versus pre was15.78 points (3.81–27.75). Intervals are descriptive, n=5, and unadjusted for secondary comparisons. See `paired_contrasts.csv`, `control_boundary_interactions.csv`.

Long-run exception: L seed7 post was100% at24k but93.42% at96k. Do not describe every control as remaining at100% or treat reaching90% as permanent stability. Its sustained-T90 was750. We have not diagnosed that later fluctuation. See `probe_seed_summary.csv` and `probes/L/s7/age_3p00/curves_t0.csv`.

| seed | pre_T90 | post_T90 | T90_max_quarter_ratio | normalized_ce_rmse | single_clock_consistent |
| --- | --- | --- | --- | --- | --- |
| 3 | 3075.0 | 6375.0 | 1.2857 | 0.0147 | True |
| 4 | 3600.0 | 8850.0 | 1.3721 | 0.0572 | True |
| 5 | 3600.0 | 10875.0 | 1.4792 | 0.0271 | True |
| 6 | 3600.0 | 7425.0 | 1.2951 | 0.0287 | True |
| 7 | 4050.0 | 6000.0 | 1.1786 | 0.0353 | True |

The registered normalization begins at first CE<=2 and scales the interval to first CE<=.5. RMSE was0.0147–0.0572, below.10 in all5. The interval-ratio consistency condition also passed all5. This is a specific learning window, not a claim that all96000 updates are identical up to a clock. Old-function removal took mean630→1125 updates; further learning took2520→5145. Slowing therefore extends beyond initial removal. Adjacent-quarter T90 ratios never exceeded1.479; no seed met the registered combination of a>=2 quarter jump and>=3 overall jump. The learning-time sharp criterion also passed0/5. See `clock_seed_tests.csv`, `quarter_source_clocks.csv`.

## What changed inside the network?

At the common probe's start, both source groups had zero images lacking all positive L2 units, and source L2 gate-covariance participation was30.39→31.25. Thus the evidence does not show a collapse of this source mask-diversity statistic at the boundary. Its value is not the rank or capacity of the network.

The larger separation appeared **after switching to the new task**. At75 updates, the mean fraction of images with no positive L2 unit was16.90%→67.63%, a paired difference50.73 percentage points (95% t interval9.47–92.00). This fraction rose in4/5 seeds. For those inputs all L2 slopes are the weak leak.003, not zero; output bias and other paths remain, so they are not dead inputs with a zero Jacobian.

**Seed5 is an essential exception:** no image had zero positive L2 units at75 in either source, yet its T90 slowed most (3600→10875). Its mean L2 derivative at75 even increased.0817→.0954. Thus neither the mean derivative nor the fraction with zero positive units is a sufficient phase variable or established necessary cause.

A shared change was reduced input-dependent L2 gating during relearning. At750 updates, mean per-unit variance across input images of the true derivative fell0.045817→0.010578, and the number of distinct L2 sign masks among1200 images fell685.0→206.8. Both fell in5/5 seeds; the pattern-count paired change was-478.2 (95% t interval-621.0–-335.4). These measure gating diversity, not exact expressive dimension. See `paired_dense_points.csv`, `gate_aggregate.csv`, `mechanism_paired_contrasts.csv`.

G's fast learning shows that retaining the source's input-dependent masks avoids much of the observed delay; L shows that the initial response can already fit the new task under that diagnostic model. This supports a cost of changing response during relearning, rather than a complete lack of useful response at the source. It does **not** uniquely prove that the measured mask statistic mediates all slowing: G/L change the subsequent function family, and the dynamics of weights and Adam remain relevant. The strong-path/coverage interpretation is post hoc; the metrics and controls were registered.

## Accuracy threshold and spectral checks

Across source fractions0,.25,.5,.75,1, native6k accuracy means were98.12,96.68,91.80,89.17,82.22%; CE means.0838,.1288,.2618,.3474,.5087; average margin6.018,5.491,4.721,4.277,3.785. These move progressively together. There is no isolated accuracy-only discontinuity in this sampled window. A margin threshold can still affect the visual steepness; we have not assigned it a causal fraction. See `margin_seed_summary.csv`, `margin_cdfs.csv`, `quarter_source_clocks.csv`.

All source tangent products used float64; their gates matched float32 source gates. Time1/time10 demand estimates passed the64/128 criterion for all operators. However, time100 passed only1/35 Euclidean and2/30 Adam-frozen cases; ridge alpha1e-4 passed19/35 and15/30. **Do not use the unconverged slow-tail estimates to claim a rank transition, spectral gap, or absence of one.** Retain them as flagged diagnostic output. The converged early-time responses and exact traces also vary nonmonotonically over source quarters, which differ in within-task phase. Source geometry alone does not explain the monotonic increase in probe T90. Kernel virtual time is not Adam updates. See `spectral_summary.csv`, `spectral_convergence_counts.csv`; the crosses in `spectral_demand.png` flag failures.

## Limits, reproducibility, and earlier interpretation

This design samples one leak, one width, one common probe task per seed, five independent seeds, and a quarter-task source resolution. It does not rule out a narrower crossover, other settings, or phase-transition behavior under width/system-size scaling. In the present window, the registered evidence supports a budget crossing with changing gate dynamics. Do not infer a universal critical mean derivative.

The earlier `leaky_cliff_1111` result used later states and demonstrated recoverability. Its observation alone did not distinguish an abrupt dynamical change from continuous slowing. This experiment supplies the missing boundary alignment and clock tests; the older spec and data are unchanged.

`checks.json`, `spectral_checks.json`, and `integrity_audit.json` passed: full seed100 two-task trajectories and RNG exactly match the old host; all10 main source pairs have exactly equal N/G/L initial forward/gradient/actual update; all70 probes share their seed's labels; all raw logits are finite; G/L used masks remain fixed. No incomplete jobs or excluded seeds. Raw states/logits/operator arrays are moved with SHA256 to `/home/issan/Projects/obsidian-research-data/leaky_boundary_1111/`; resolve via `backup_manifest.json`. Code, spec, tables, logs and figures remain in Git. The analysis reads archived logits through the manifest.
