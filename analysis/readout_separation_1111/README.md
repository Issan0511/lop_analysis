# Readout separation, 2026-11-11

Registered design: `specs/spec_readout_separation_1111.md`. This study follows `p_residual_loading_1111`; all new main runs start after committing the implementation.

The question is which effects of a large readout remain after class-feature associations are changed, and whether ELU and final-LN ViT share the same harmful pathway. The preceding P-versus-zero study cannot separate these causes.

## What each contrast identifies

| Contrast | Held fixed at the fork | Changed | Interpretation limit |
|---|---|---|---|
| Physical c1 versus c0.1 within S/P/O | Body, chosen readout directions, mean logits, batches, head-Adam reset | Centered readout weights and compensating bias | Also changes probabilities, body gradients, and relative head learning; not a confidence-only test |
| S versus P | Column multisets and norms, body | Class-feature assignment and other geometry | P is not an alignment-only operator |
| S versus O | Full feature Gram W^T W, singular values, column means, body | Class-space direction and compensated bias | One fixed rotation per seed; does not preserve probabilities |
| Crossed frozen forward/backward banks | Entire per-example probability vector and labels when comparing backward maps | W in W^T(p-y) | Identifies local maps; not a long-run mediation share |
| T/G/TG during updates1–200 | Same initial state and head parameterization | Forward centered-logit scale and/or body-gradient gain | Later trajectories diverge; T also changes the head residual through p |
| ELU derivative=1; LN selected projection terms removed | Exact forward output at a fixed state | Input Jacobian only | A surrogate backward rule, not a retrained architecture without ELU/LN |
| Norm-matched altered Jacobian | Native trunk-gradient norm, forward probability | Trunk-gradient direction | Local diagnostic; does not identify the long-run effect of a pure direction intervention |
| Body Adam keep versus clear | Same physical state, or same frozen gradient | All body optimizer memories and step counters | Distinguish gradient from actual Adam displacement |
| W→.1W, gamma,beta→10gamma,beta | ViT logits and upstream effective map | Coordinate scales | Raw W norm alone is not an invariant cause |

The fixed comparison contains280 full-horizon trajectories (ELU10seeds×18, ViT5seeds×20) and2520 crossed frozen-state records. The statistical unit is the source seed, not individual examples or arms. U is horizon-average fixed-training accuracy deficit. The paired bootstrap intervals are descriptive and unadjusted. Report interventions separately and their interactions; do not normalize effects into additive percentages of total delay.

The uniform-label channel is the exact expected **gradient** under independent uniform labels at the frozen state. Its virtual Adam value applies Adam to that expected gradient; because Adam is nonlinear, it is not the expectation of label-sampled Adam displacements. Actual fixed next labels remain the primary local channel.

## Files

- `ops.py`: physical readout bank, surrogate forward/body gains, forward-identical Jacobians.
- `run.py`: source restoration, common forks, registered measurements, legacy controls and trajectories.
- `probe.py`: fixed128-example crossed VJPs, virtual Adam, norm matching, gauge check.
- `launch.py`: bounded CPU/GPU execution and attempt logs.
- `tests_ops.py`, `probe_tests.py`: synthetic operator tests.
- `summarize.py`, `summarize_probe.py`: paired-seed summaries.
- `verify_summary.py`, `verify_probe_summary.py`, `verify_integrity.py`: metric and provenance validation.

Run these as modules from the repository root with the existing project virtualenv. Full main execution is through the registered launcher after smoke and implementation commit. Archive raw files and attempt logs before removing the experiment worktree; `results/readout_separation_1111/backup_manifest.json` is the durable path map.
