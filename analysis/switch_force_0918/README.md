# Reproduction

Read `specs/spec_switch_force_0918.md` first. `run.py` imports the repository's production `VecMLPL`/`SCREnv`; it owns no external checkpoints. It writes raw data to `/home/issan/Projects/obsidian-research-data/switch_force_0918/` and small results to `results/switch_force_0918/`.

Dependencies: Python 3.12, NumPy, PyTorch, SciPy, Matplotlib. The research host has NumPy/PyTorch/Matplotlib; SciPy for reports was installed in the isolated `/tmp/switch_force_report_deps_0918` directory without changing the host packages.

```bash
python3 analysis/switch_force_0918/run.py --preflight
python3 analysis/switch_force_0918/run.py
python3 analysis/switch_force_0918/verify_external.py
PYTHONPATH=/tmp/switch_force_report_deps_0918 python3 analysis/switch_force_0918/report.py
```

Primary data axes: trajectories are `[time, condition*seed, unit, ...]` with switch seeds 0–9, control seeds 10–19. Frozen metrics are `[flipped_bit, hybrid_condition, seed, unit]`, where hybrid conditions are old/old, new-input/old-target, old-input/new-target, new/new. Frozen change-only arrays omit the hybrid axis. Every force/transport uses the *new* input mean as its projection reference, including no-switch controls. `oldmean_projection_change` is separately available.

`h` denotes total gradient, `hs` self, `hr` rest in frozen data. `self_pos`, `self_neg`, `rest_pos`, `rest_neg` are *signed weight-response increments*, not magnitudes or normalized fractions. `cum_` arrays accumulate actual steps, with rounding ledgers kept separately. `input_jump` is the environmental boundary change before any update. `length`/`direction` are exact symmetric endpoint allocations, distinct from cumulative radial/tangent step contributions.

The first preflight failed due to an incorrect diagnostic scalar dtype; its data and checks are retained with `failed_gate_dtype` filenames. Main trajectories were run only after correction and successful preflight. `verify_external.py` replays the registered continuations without instrumentation, using the original environment sampler, and checks saved W/b/v/c arrays for bitwise identity. This is a replay of the *new protocol trajectories*, not a replay of the original historical 5M-step training stream.
