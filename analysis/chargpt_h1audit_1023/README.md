# H1 saved-state audit

Preregistered spec: `specs/spec_chargpt_h1audit_1023.md`.
Only checkpoint reads and fixed-state algebra/autograd. No training or optimizer.

```bash
python analysis/chargpt_h1audit_1023/audit.py gate
python analysis/chargpt_h1audit_1023/audit.py run
python analysis/chargpt_h1audit_1023/summarize.py
```

Uses the existing `chargpt_loadsign_1023` loader, the original `load10.py`
data/probe and alive definitions, and the pinned vendored model source.
`gate` verifies the input bundle SHA, all 70 seed-0 matched-probe results,
eight exact gradients, explicit small-tensor input-pair sums, and an
exhaustive independent uniform old-label expectation with frozen features.

`actual = oracle + fit` is an exact fixed-error decomposition. `oracle`
means an externally substituted one-hot old-label error, not a retrained
or improved predictor. `truth` substitutes the task's true conditional
probabilities, including the stationary conditional distribution at the
first token. F and kappa retain their original sampled-label values.

The own-unit loading source and the input-pair diagonal are separate axes.
The latter is the analytic IID old-label expectation only with fixed
features, derivatives, kappa, and geometry. Neither is a causal contribution
to the actual training trajectory. `direct` removes only the final LN's
variance derivative; all forward activations remain unchanged.
