Registered run 4: response-field gradient/Adam update controls

Registration: local `34bac645`, remote `fe33a41389c25cf24aad6e8f768fc18093686bcc`.
Prediction: `NOT_LR_ALL_THREE`, paper-derived under the common spec §8. No quantitative Issa prediction is invented.

The implementation imports the original response engines, with a native-loop RM adapter and stacked CUDA RC/F51 subclasses. The two CUDA ELUs and CPU expm1 ELU stay distinct. R swaps the original second-hidden-layer anchored field. N and both LR controls keep the recipient's parameters, moments, Adam count and random streams. LR_G/LR_U multiply the actual per-layer Adam learning rate, including the bias, using fixed un-clipped calibration ratios.

First 256 updates log each W and bias separately: gradient, m, v, bias-corrected moment norms, candidate and actual displacement norms, relative displacement, actual LR and cosine to N. qU and group U use physical P_after−P_before displacement computed after casting both states to float64. Candidate versus actual displacement is checked with a rounding-derived bound. Whole-layer W+b norms and cosines are exported separately. Diagnostics never use or advance training RNG. Prefix and arm checkpoints have SHA256 digests and strict source/config identities. Resume preserves completed prefixes/seeds/arms, rerunning an interrupted arm from the exact branch checkpoint.

The independent CPU checks exercise real host loop parity, nonzero moments, logging invariance, native field gradients, CPU/CUDA-ELU distinction, RMS ratios, explicit zero cases, calibration-only seeds, fail-closed G0, and the 15-interval verdict. Thirteen specified adversarial mutations must be detected. The short timing fixture uses real MNIST and is explicitly CHECK_ONLY, with no shortened production CLI.

Run checks:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/resp_step_controls_1018/checks.py
# On CUDA hardware, also execute real-CIFAR R=10 graph/native parity checks:
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/resp_step_controls_1018/checks.py --cuda
```

After independent review, the standalone CPU RM production entry point runs all 10 calibration and all 10 evaluation seeds with exactly 80 epochs/task, t2 donor/t10 recipient, and both t11/t12 continuations. It is restartable; run it on the same committed source. The full workload is 2.16 million updates. The real-data timing fixture includes all logger work and gives a conservative sequential estimate; prefix updates are faster.

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/resp_step_controls_1018/run_rm.py --out results/resp_step_controls_1018/RM
```

The two phases can also run independently:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.resp_step_controls_1018 --box RM --phase calibrate --out results/resp_step_controls_1018/RM/calibration
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m src.resp_step_controls_1018 --box RM --phase evaluate --schedule results/resp_step_controls_1018/RM/calibration/schedule.json --out results/resp_step_controls_1018/RM/evaluation
```

For RC/F51, replace the box name and destination. Registered CUDA protocol remains R=10, seed order 100–109 for calibration and 0–9 for evaluation, RC 400 epochs/task or F51 780 updates/task. F51 additionally requires `--integration-audit /tmp/issa_six_specs_1018/integration_audit.json`, with independent proof flags and exact host hashes; missing/false proof fails closed. The runner does not substitute CPU or shorten a protocol when CUDA is absent.

Readout:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/resp_step_controls_1018/verdict.py --RM results/resp_step_controls_1018/RM/evaluation/per_task.json --RC results/resp_step_controls_1018/RC/evaluation/per_task.json --F51 results/resp_step_controls_1018/F51/evaluation/per_task.json --out results/resp_step_controls_1018/readout
```

The readout refuses missing/duplicate seed-arm rows, mismatched label/order streams or non-finite results. It reports m=15 Bonferroni paired t intervals, ordinary intervals, all ten seed differences, signs, medians, degenerate-SD flags, box labels and the three-box result. A missing box yields INCOMPLETE, never an all-three conclusion. NOT_LR applies to the two registered constant layerwise schedules only.

Raw gradient caches, checkpoint `.pt` and unit `.npz` files must be retained and archived under the repository's CLAUDE.md policy before the worktree is removed. They should not be committed as large blobs. Source and compact check/blocked-status evidence are versioned.
