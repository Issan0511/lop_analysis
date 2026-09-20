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
