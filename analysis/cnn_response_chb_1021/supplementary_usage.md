# CH′ response experiment: supplementary reporting support

The script is standalone and CPU-only (`numpy`, `matplotlib`, standard library). It imports no repository module or Torch. It was first prepared outside the repository while the GPU run was pending. After all four arms completed, this reporting-only script was copied here, validated against the saved results, and used to generate the supplementary artifacts. It does not modify the frozen training code or spec.

Run from the repository root, only after every arm is COMPLETE:

```bash
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
  analysis/cnn_response_chb_1021/supplementary_figures.py \
  --results results/cnn_response_chb_1021 \
  --out results/cnn_response_chb_1021/report/supplement
```

If raw artifacts have already been archived, add `--archive-manifest /absolute/path/to/backup_manifest.json`. The resolver verifies archived bytes and SHA256. The input root should retain its original results-directory identity, even if some raw files were moved.

All inputs are read-only. The script refuses to create outputs unless all four COMPLETE records, final-state hashes, curve hashes, summary hashes, paired streams, all40 seeds, four fixed windows and all registered diagnostics are consistent. Existing generated artifacts in `--out` are replaced by the same named artifacts. Use an output directory dedicated to this script.

Outputs:

- `per_seed_windows.csv`: the four registered accuracy/CE windows, all seeds and failure onset.
- `per_seed_layer_windows.csv`, `per_seed_epoch_layer_updates.csv`: all five layer update/Adam summaries, B fractions and failure status.
- `per_seed_epoch_accuracy.csv`: all40 epoch trajectories.
- `diagnostics_long.csv`, `diagnostic_group_means.csv`, `nonfinite_diagnostic_counts.csv`: all stored scalar diagnostics, all observations retained.
- `initial_manipulation_per_seed.csv`: raw versus effective gate change, initial donor matching and receiver-pool derivatives for every seed/layer.
- `late_response_per_seed.csv`, `per_seed_final_metrics.csv`: late behavior and final diagnostics, preserving NaN/inf.
- Nine figures, each as PNG and PDF: full learning curves; registered windows; raw/effective gates; donor/receiver gate agreement; delta/fixed offset/current displacement; Adam and updates; continuing B; full cross-entropy trajectories; registered cross-entropy windows.
- `READING_GUIDE.md`: precise reductions and interpretation limits.
- `supplement_manifest.json`: script/input/output hashes, library versions and fixed analysis definitions.

Recommended report order: (1) verify manipulation at update0 using all seeds and the registered preflight; (2) present the existing primary verdict and fixed-window plots; (3) inspect late response and reference displacement; (4) use B/Adam plots as diagnostics, without claiming that they identify a separate causal mechanism. Keep fixed anchor offset and current activity displacement separate. Do not treat high gate agreement alone as proof of full-Jacobian transfer or semantic feature alignment.

CSV represents undefined/infinite values as literal `nan`, `inf`, `-inf`. Every mean includes all10 seeds; there is no `nanmean`. Plot gaps are annotated. Head B fields are explicitly not applicable, rather than biological failures.

Pre-run validation used `py_compile`, CLI `--help`, and four in-memory helper checks without reading experimental results. Post-run execution completed with exit0 on all four COMPLETE arms, retaining exact agreement with the registered reduction and all failure/NaN policies. All nine actual-data PNGs were visually inspected for labels, scales, legends, clipping and complete seed display; matching PDFs were generated from the same figure objects. No simulated result figures or training reruns were used.

This script was added to `analysis/cnn_response_chb_1021/` only after every provenance-critical arm had completed. The recorded execution identity remains the preregistered training code; this supplementary script belongs to the subsequent result-report commit.


## Reporting-only exact-reduction correction (2026-10-21)

The first supplementary attempt stopped before producing result artifacts at `CH_swap/seed11/ce_all`. The stored result was `40.989630426147315` (`0x1.47eac35b5b6f2p+5`). Pre-casting its float32 curve to float64 and then reducing produced `40.98963042614732` (`0x1.47eac35b5b6f3p+5`), a one-ULP difference of `7.105427357601002e-15`. Both native float32-source `mean(dtype=np.float64)` and the independent audit's two-dimensional `mean(axis=1, dtype=np.float64)` exactly reproduce the stored result. The report now uses the latter for every registered window and retains exact `np.array_equal` comparisons. No tolerance, experiment code, result, failure rule, or primary test was changed. The first failed log is retained as `results/cnn_response_chb_1021/supplementary_figures_first_attempt.log`.

Two CE-specific figures were added (`08_cross_entropy_all_seeds`, `09_registered_window_cross_entropy`), together with `window_arm_means.csv` and a visible accuracy/CE table in `READING_GUIDE.md`. The accuracy-defined primary rescue must not be described as an improvement of all metrics: CH′ swap has much higher mean CE despite higher accuracy. Every finite CE observation remains present; log-like display uses a stated symlog scale rather than clipping.
