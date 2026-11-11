# leaky_order_1111: orders of adaptation time across leak .001–.01

Registered 2026-11-11 before new data. User asks how many orders of learning updates are needed BETWEEN a=.001 and .01, rather than one isolated smaller leak. This is a new experiment using the existing deterministic random-label MNIST host; not a reconstruction of the remembered CondA figure. No inverse-a or inverse-mean-derivative law is presupposed.

## Design

Leak grid a={.001,.002,.003,.005,.01}, seeds3–7. Fixed1200 MNIST images per seed, pixels/255,784–100–100–10, both hidden layers use the same constant leak from initialization onward. Adam lr=.001, eps1e-8, batch16, no decay, float32, deterministic CPU and one thread per process. Same initial params, image subset, training label streams and batch streams across a within seed. Six sequential random-label tasks,6000 updates each. Save source states at ages0,3,6 with parameters and all Adam/RNG state.

The a=.003 source states at0,3,6 already exist in leaky_boundary_1111. Reuse them by SHA256-verified manifest, without counting reuse as independent replication. Generate20 new prefixes for the other four leaks. All75 probes (five leaks x three source ages x five seeds) are run with the new validated runner. The original .003 probe trajectories at registered reference ages supply exact arithmetic checks, not additional observations in statistics.

At each source, use the SAME independent boundary_probe_labels and boundary_probe_batch streams from leaky_boundary_1111 within seed. Keep this ONE new task unchanged for the entire probe. Preserve source Adam history, no head reset, no leak change at branching. Comparison across leaks includes the effects of their different learned states; it is not a same-parameter instantaneous leak intervention. Primary source age6, secondary age3, and fresh age0 to distinguish acquired slowing from an inherently slow initialization. These fixed ages are chosen before new outcomes, not event-selected per leak.

## Stopping, thresholds and censoring

Evaluate full-bank train accuracy and CE at step0,1, then every75 updates. Primary time outcome is T90_sustained: beginning of the first four consecutive scheduled evaluations >=90%. Also report first T90, T80, T95, T99, CE thresholds2,1.5,1,.5,.2,.1. These higher thresholds are secondary and may be censored by stopping at90%.

Every probe runs at least24000 updates, even if90% is reached sooner. From then on, stop at the first evaluation with a recorded T90_sustained, or at a maximum1536000 updates (256 times the usual6000). Thus a probe that reaches sustained90% before24k stops at24k; a late recovery stops after the four-point confirmation. Save fixed budget snapshots at6000,24000,96000,384000,1536000 if reached, plus the final state. Do not restart labels, batch order or optimizer at these checkpoints. A later endpoint may fluctuate below90% after an early sustained crossing; report this rather than assuming permanent stability.

Unreached T90 is a RIGHT-CENSORED lower bound, not infinity, and is never dropped from means. Report each seed, event counts at budgets6k/24k/96k/384k/1536k, paired leak ratios when both times are observed, and bounds when one is censored. Give median only if identifiable; otherwise give its lower bound. Preserve range/censoring markers in plots. Accuracy at24k is common to all cells; later endpoints belong to adaptively selected slow runs, so no survivor-only average accuracy curve is called a population average. First-crossing T90 and sustained-T90 remain distinct.

The main answer is descriptive orders of time (10^3,10^4,10^5,10^6 and lower bounds), and the progression across the five leaks at matched age. Plot log leak against log time with per-seed trajectories. Any local log-log slopes or power-law fits are explicitly exploratory, require observed endpoints, and do not establish a universal scaling law or a critical threshold. a is the negative-side slope, not mean activation derivative. No phase-transition verdict follows from finite censoring.

## Diagnostics and verification

At step0,1,every750 and fixed budgets, record per-layer mean true derivative, positive fraction, input variance of derivatives, one-sided fraction, zero-positive-unit input fraction, and preactivation m/s statistics. At selected budgets include gate/feature covariance participation and distinct sign masks. Accuracy/CE are dense even when these diagnostics are sparse. At first and every750 record gradient norms, actual parameter updates, actual centered-logit displacement and CE decrease. These distinguish a small leak from actual gradient/functional dynamics but are secondary to measured hitting times.

Before main, verify seed100 native updates and a two-task prefix against the existing host at a=.001; preserve exact final parameters, moments and RNG. Check interrupted/resumed training with saved RNG, same labels/orders across leaks, and the stop/censor rule with synthetic evaluation traces. Compare a=.003 source/probe outputs against archived leaky_boundary_1111 at available budgets; discrepancies require investigation before combining results. No data-dependent retuning or selective exclusion.

Spec commit/push before main. Save git/code/spec/data hashes, source-manifest hashes, all per-seed curves and final checkpoints. CPU only, at most5 training processes, one thread each, reserve6GiB RAM. No paid workers, subagents, or messages to other chats. Check ongoing runs at most every5min unless completion is expected sooner or an error needs attention. Archive raw with SHA256, merge main without rewriting commits, remove only own worktree, and save a new Obsidian result note. Do not edit manuscript/current-context notes or earlier registered specs.
