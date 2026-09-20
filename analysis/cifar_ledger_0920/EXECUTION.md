# A6 execution

Registration: `f034e9a`; Issa approval and shared prediction contents: `4aec348`.
One process, original 20 slots, original CUDA environment. No optimizer or training updates.

Use the shared runtime from `proj_004_drift/.venv/bin/python`:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/cifar_ledger_0920/checks.py
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/cifar_ledger_0920/replay.py
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/cifar_ledger_0920/report.py --src results/cifar_ledger_0920
```

Commit code after checks, before replay. The replay requires the tested source hashes and clean source status.
It holds `/tmp/lop_analysis_gpu.lock` and refuses other research GPU processes. Desktop graphics contexts are excluded.
Create `results/cifar_ledger_0920/STOP` to stop at a saved-state boundary. Remove it to resume the identical code/input/environment.
The manifest and per-state completion hashes guard resume; report refuses partial output.

`checks.py` uses artificial data, native activations and their autograd derivatives. It does not inspect experiment outcomes.
The initial test attempts found an unavailable optional `threadpoolctl` and an exact-equality assertion on two differently ordered floating sums.
The implementation now sets BLAS environment limits before NumPy import; the assertion uses its arithmetic error bound.
Neither change modifies registered scientific definitions. Additional runtime failures, if any, are kept as separate attempt records.

Replaying verifies input SHA, full state shapes, original initialization, saved hist/float16 preactivations, CSV diagnostics,
activation state, unchanged RNG, direct affine reconstruction, four-term closure and norm/direction closure.
Term calculations use float64 error propagation from the faithful float32 features. Undefined ratios remain undefined.

The scientific results are a registered reanalysis of known trajectories, with no independent audit.

## Before first outcome report

The full replay completed on implementation commit `45b12f2` in about nine minutes; all 306 states passed.
During a separate synthetic review, exactly 60,000 low-response pairs out of 120,000 could produce
`0.5000000000000001` when per-unit fractions were averaged. Before reading any verdicts, the report
was changed to recover and sum integer pair counts before division. This implements the registered
strict-majority boundary and does not change its threshold. A regression fixture reproduces the issue.
Late first crossings, the L2-ledger-when-L1-first annotation, and report-specific provenance were added
in the same pre-report review. Replay code and its scientific arrays remain unchanged.
