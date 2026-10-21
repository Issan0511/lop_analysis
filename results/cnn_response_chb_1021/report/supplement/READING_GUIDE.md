# Supplementary descriptive diagnostics

These figures and tables were produced only after every registered arm was COMPLETE. They do not modify the primary verdict and introduce no statistical tests or data-dependent windows.

## Reading order

1. `initial_manipulation_per_seed.csv` and `03_raw_effective_gates`: establish whether the effective activation derivative means match the initial donor while receiver raw gates remain unchanged at update0. The CSV includes each seed and layer. A scalar mean alone does not demonstrate semantic feature alignment or the full Jacobian exchange; use the registered preflight for the elementwise intervention checks.
2. `01_learning_all_seeds`, `02_registered_windows`, `08_cross_entropy_all_seeds`, `09_registered_window_cross_entropy` and `window_arm_means.csv`: inspect the entire task, every seed, and the fixed primary/secondary windows. The primary verdict belongs to the registered independent main report. Accuracy improvement and CE worsening may coexist. The registered primary verdict is defined by accuracy and does not assert improvement on all metrics.
3. `04_gate_agreement` and `05_delta_and_offsets`: inspect response movement away from the fixed fields. Gate agreement can be high simply because many units share a closed gate. Fixed anchor offsets, current activity displacement and raw preactivation delta are distinct quantities.
4. `06_adam_and_updates` and `07_bias_projection`: inspect receiver optimization and continued B. Similar update norms do not isolate an effective-step-size causal mechanism.
5. `per_seed_final_metrics.csv` and `late_response_per_seed.csv`: retain all final values, failure status, and layer-level late behavior.

## Reductions and missing values

- Every scientific comparison includes seeds10–19. `mean_all10` and thick figure lines use ordinary arithmetic means; no `nanmean`, failed-seed exclusion, or successful-seed rerun occurs.
- Accuracy tables use fractions0–1; behavior figures use percentages0–100. Accuracy and CE come from all30000 saved pre-update online measurements. Failure policy remains accuracy0/CE+∞ from failure onset onward. The script rejects curves that violate that protocol.
- Diagnostics use the six registered snapshots. The raw/effective mean derivative is1 for native clamp input≥0 (including exactly0), and0 below0; the strict opening fraction is a separate diagnostic available in CSV.
- For convolutional diagnostics, stored scalar means average the per-position image-axis statistics over channels and spatial positions. In particular, the plotted displacement magnitude is the mean of per-position image RMS values, not an RMS across all channels/positions.
- All layers retain their own scales. Activity effective rank follows the original stored channel/unit convention; convolutional rank is not the rank of the entire flattened spatial feature vector.
- At convolutional sites, pooled derivatives are evaluated at the receiver's actual artificial-activity pool winners. They are not derivatives at the donor's own pool winners.
- `activity_offset_rms` is current artificial activity minus native clamp of current raw z. `fixed_anchor_offset_rms` is clamp(initial receiver z) minus clamp(initial donor z). These must not be conflated.
- Update traces are reported as fixed nonoverlapping75-update epoch means. Weight relative update uses the pre-update weight norm. Bias update includes B after Adam where present. Denominator means and RMS(gradient/denominator) are descriptive recorded quantities.
- Head B columns are NaN with `B_applicable=False`; no head bias projection exists. CH′ hidden B fractions are zero and `B_applicable=False`.
- CSV preserves literal `nan`, `inf`, and `-inf`. They remain in means. Figures display nonfinite values as gaps and annotate their counts. `nonfinite_diagnostic_counts.csv` distinguishes undefined ratios from a failed seed; the failure flags are authoritative for the experiment's failure policy.
- The log-like figure axes use fixed, stated symlog thresholds solely for visualization, retaining exact zeros. They do not change exported data or inference.

## Interpretation limits

The intervention concerns activation derivative gates at fixed corresponding image/channel/position indices. It does not exchange weights, head, optimizer history, centering prescription, pool selection, or the full Jacobian. Do not divide the effect by the known natural CH′+B benefit to claim a mediated fraction. The one-task intervention does not establish long-term prevention or identify which of the four layers is necessary.

## Validation

All four completion records, provenance identities, all40 seed rows, label/batch pairing, curves and final-state SHA256 values were checked before output. All four registered windows were independently reconstructed from float32 batch curves with `mean(axis=1,dtype=float64)`, exactly matching the registered independent audit. Pre-casting to float64 before summation is intentionally avoided because NumPy may choose another summation kernel. No torch or GPU library is imported.

Failure onset (seed@update):
- CH_self: none
- CH_swap: none
- CHB_self: none
- CHB_swap: none

## Accuracy and cross-entropy together

| Arm | First3000 accuracy (%) | First3000 CE | Last3000 accuracy (%) | Last3000 CE |
|---|---:|---:|---:|---:|
| CH_self | 14.63562 | 2.38638 | 37.44771 | 1.64464 |
| CH_swap | 58.21854 | 732.47 | 99.20042 | 8.6998 |
| CHB_self | 71.41917 | 7563.08 | 98.58313 | 168.688 |
| CHB_swap | 59.95083 | 14223.8 | 98.74083 | 231.91 |

An accuracy-defined rescue is not an all-metric rescue; retain the CE increase in interpretation.

Initial maximum raw gate mean residual within receiver: 0.0.

Initial maximum swapped effective minus donor raw gate mean residual: 0.0.
