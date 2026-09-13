# Reproduce zero-attraction 0913

Training: /home/issan/Projects/claude/zero_attraction_0913, branch codex/zero-attraction-0913, HEAD 0ba13e8.
Analysis: /home/issan/Projects/claude/zero_attraction_analysis_0913, branch codex/zero-attraction-analysis-0913, code HEAD 422f9dc.
Primary spec/config commit 1689251. Natural-trace amendment a72f455 preceded its implementation.
Python: /home/issan/Projects/claude/proj_004_drift/.venv/bin/python; torch 2.13.0+cu130, CPU float32, one thread.
Use OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.

Raw data backup: /home/issan/Projects/obsidian-research-data/zero_attraction_0913/
Restore training/ to the training worktree's results/zero_attraction_learning_0913/. SHA256 manifests live next to training/ and analysis/.
Analysis scripts use this absolute DATA location. Source tarballs contain the registered code/config/specs.
Do not rerun learning into the completed result directory. Main invocation in a fresh checkout/output context was:
python -m src.zero_attraction_launch_0913 --parallel 12

From the analysis worktree, with the input paths restored:
python -m src.zero_attraction_report_0913
python -m src.zero_attraction_mechanism_0913
python -m src.zero_attraction_natural_trace_0913
python -m src.zero_attraction_controls_0913
python -m src.zero_attraction_synthesis_0913

The natural trace can reuse shards only if its source SHA matches. The probe can reuse the saved partial measurements only when its tested mathematical function ASTs match. On a clean checkout both can recompute from checkpoints and logs.
The primary CI in comparisons.csv uses the same registered draw as verdict.json. Older partial summary is preserved separately.

Completion verification: 190 logs, 76 checkpoints, 76 local probes, 76 algebra controls, and 14 natural replay trajectories. No new claims from unit-level pooling; primary independent unit is seed. All 14 replays matched free weights and readout weights byte for byte.

No claim of replication of the inaccessible lab-PC shifted-Snake implementation. This is the explicit phase formula in the spec. Scalar MSE/SGD evidence only; self-term ablation, equal-initial-output learning, CE/Adam and long-run projection interventions remain later stages.
