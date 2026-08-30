# Residual-induced force after centering (post-hoc; no training)

## Verdict

- **Cross-moment regeneration is confirmed:** after input centering, E[delta x] remains nonzero and is dominated by Cov(delta, x), not by residual input mean.
- **Causal killing is not established:** force magnitude has no robust 1k-step event association with increases in the aggregate, task-dependent dead_frac label.
- This is analogous to layer-2 mean regeneration, but not mathematically identical: the recreated object is a residual-input cross-moment (mean force), not the input mean itself.

## Clean pre-death checkpoint: step 10k

All centered runs have exact dead_frac=0 here.

| metric | w5 median | w100 median | w5/w100 fold [95% CI] |
|---|---:|---:|---:|
| nmse | 0.0980459 | 0.00154766 | 63.35 [18.32, 139.32] |
| mu_norm | 0.0744373 | 0.0730236 | 1.02 [0.61, 1.50] |
| G_norm | 0.040124 | 0.0127432 | 3.15 [1.44, 5.62] |
| covariance_term_norm | 0.0409634 | 0.0109156 | 3.75 [1.54, 6.10] |
| mean_input_term_norm | 0.00172916 | 0.00196505 | 0.88 [0.66, 8.28] |
| ungated_force_sq_per_unit | 0.00280587 | 2.54798e-05 | 110.12 [26.05, 391.21] |
| gated_force_sq_per_unit | 0.00137602 | 1.60102e-05 | 85.95 [41.70, 409.84] |
| residual_input_coherence | 0.0121843 | 0.0443173 | 0.27 [0.03, 0.88] |

Covariance share ||Cov(delta,x)||/(||Cov||+||Edelta mu||): w5=0.918, w100=0.833.
CSV gradient SNR at the same point: w5=0.0158, w100=0.0572. The mean force is larger per unit at w5 but less dominant relative to per-sample gradient variance. Residual-input coherence is also lower at w5, so the force increase is primarily an absolute residual-scale effect, not stronger normalized directionality.

## Does it predict death?

- At w5, Spearman(step-10k force, final dead_frac): ungated=0.154, gated=0.564 (n=5; descriptive).
- Same-task interval Spearman(previous drift, next 1k net dead change), seed median: w5=-0.003, w100=-0.015.
- Only 2/5 w5 first onsets have a 1k observation in the same task immediately before detection; the other onsets occur at the first probe after a task switch.

The logs therefore confirm the proposed nonzero force before final gate closure, but they do not show that this force points toward the gate-closing boundary. A unit-ID logger must record the signed force projection onto each unit's gate margin immediately before onset.
