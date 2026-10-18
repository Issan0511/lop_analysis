# Loss-gated Snake — CNN implementation checks

Implementation complete; scientific performance verdict: **not evaluated**.

Five checks passed on CUDA (PyTorch 2.13.0+cu130, RTX 5060 Ti):

- Per-run detached loss gate and variance update formula; high/low/non-finite loss.
- Last-10-epoch median calibration, epsilon floor, missing-controller-state rejection.
- Task-1 CNN training bit-identical to host (parameters, Adam moments, variance).
- Gate enabled after capture: CUDA graph/eager and active-gate checkpoint continuation bit-identical.
- Actual two-task runner: interrupted/resumed state, CSV records and calibration identical; changed settings rejected.

The runner check uses two epochs per task and one seed, solely for bookkeeping.
The initial parity test advanced one batch-order stream an extra time; the corrected test aligns both streams.
Source hashes and command: `checks.json`; complete successful test log: `checks.log`.

Run entry point: `python src/sna_loss_gate_1018.py --seeds 10-19 --tasks 30 --epochs 400`.
This command has **not** been run. Register performance windows/criteria first (see spec).
