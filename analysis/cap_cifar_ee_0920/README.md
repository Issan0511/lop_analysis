# S5 implementation and execution record

Design and predictions were approved together with S4 before either new result (`5705359`). S4 was completed and merged as `0e733e9` before creating this branch. While S4 was running, draft S5 numerical/statistical code was prepared outside the repository; no S5 training or scientific result was inspected.

This experiment runs its own fresh task 1. It clones that state into ref, cap1, cap2, cap12 and cap12_bfix, preserving every optimizer and RNG state. Every arm runs through task 50, in that order, on the same R=10 stack and fixed seeds 0–9. The registered readout is task 31–50.

`CapEngine` extends the already verified original-R engine. Each step completes native Adam first, then projects W1 and W2 as registered, then restores b1/b2 in cap12_bfix. It never projects optimizer moments or W3/b3. The cap formula uses native float32 row norms; every actual post-projection row norm is independently accumulated in float64 and checked against the registered arithmetic bound. Warmup restores all counters. Each task saves hits, actual removed displacement, maximum norm/radius, excess and violation counts, and bias candidate displacement.

Every task also saves the A6 four-term ledger and symmetric norm/direction decomposition in float64, with propagated closure bounds. The discrepancy between native mean z and algebraic W×mean(a)+b is checked separately. Undefined variance ratios stay undefined.

The statistics use fixed arrays of 5×10×50 values. A constant sample has an exact point interval; constant floor windows preserve equality without a spurious mean-rounding residual. No statistical threshold, radius, seed, task window or prediction was changed after S4.

Run admission checks using `analysis/cap_cifar_ee_0920/checks.py` in the shared Python environment. All attempts are retained. Commit/push the verified implementation before `analysis/cap_cifar_ee_0920/launch.sh`. The runner requires exact source hashes, the original input identity, clean implementation and the GPU lock. A STOP file in the output directory checkpoints at an epoch boundary; remove it and use the same command to resume.

Run the explicit-source report only after all five arms finish. Missing/nonfinite/invalid runs do not get a scientific label. Independent audit: none; these are implementer tests and independent numerical reference calculations.

Admission completed: 13 required checks and 43 mutation controls passed. Earlier attempts exposed mistakes in the test harness's generated method inheritance/indentation; those attempts are retained. The zero-radius rule was also made explicit for nonzero subnormal rows. None of these checks used scientific seeds 0–9. The final short-run estimate is 10,442 seconds (~2.9 hours); peak test RSS ~2.87 GB and CUDA allocations ~2.73 GB.

All five arms completed 50 tasks on 2026-09-20 10:52:51 UTC, in 9,832.37 seconds. The registered primary label is RESCUED. See `results/cap_cifar_ee_0920/interpretation.md` for the complete readout and limitations. Source hashes frozen at implementation commit 318c722 were unchanged during the run and report.

From the repository root, regenerate outputs with the shared Python environment:

```bash
python analysis/cap_cifar_ee_0920/report.py --src results/cap_cifar_ee_0920
python analysis/cap_cifar_ee_0920/audit_readout.py
python analysis/cap_cifar_ee_0920/plot_summary.py
python analysis/cap_cifar_ee_0920/supplement.py
```

The audit and plot scripts were prepared under the ignored log directory before outcomes were read, then copied unchanged here. The descriptive supplement was added after outcomes. These scripts do not change registration or training source hashes. Raw-file reads use the archive resolver after cleanup. The audit is an implementer cross-check, not an independent auditor's review.
