# L0 self-output force

Interpretation correction (2026-10-23): the source-removal criterion does not
test whether the remainder vanishes in the old-label expectation required by
the paper's H1. The local `J_i^T H J_i a_i` source is a different decomposition.
The original claim of support for H1/a cause of L0 sinking is withdrawn; measured
values and preregistered criterion outcomes are retained. See `conclusion.md`.

Saved-state only, no optimizer, no parameter update. Preregistered in
`specs/spec_chargpt_l0force_1023.md` (initial c7561318; full-path extension ba5c94b4).

Use the repository Python environment, with bytecode disabled:

```
python analysis/chargpt_l0force_1023/measure.py gate
python analysis/chargpt_l0force_1023/measure.py run
python analysis/chargpt_l0force_1023/report.py
python analysis/chargpt_l0force_1023/full_self.py gate
python analysis/chargpt_l0force_1023/full_self.py run
python analysis/chargpt_l0force_1023/audit.py
python analysis/chargpt_l0force_1023/full_report.py
python analysis/chargpt_l0force_1023/archive.py
```

`measure.py` closes the direct residual self/other/output terms, final-LN
variance derivative, and both upper residual branches against autograd in 100
states. `full_self.py` uses JVP/VJP to include all downstream/time-mixing paths
for 32 randomly selected alive units in each of 40 states. All other L0 units
remain in the full feature-source total; they are not restricted to the sample.

The source convention is a fixed logarithmic-mean PSD softmax secant and a local
linearization. Its self term is a gradient of a positive quadratic self-output
energy. This does not guarantee a positive position projection: that projection
is evaluated separately. Source removal is arithmetic at a saved point, not a
finite lesion or a training intervention. The constant `f-Ja` is a Taylor
intercept, not a physically independent set of outputs.

The primary error is expected current-task CE (`p-q`), with exact target and
uniform-switch controls. No assertion equates irreducible target entropy with a
nonzero expected training gradient. Frozen Adam denominators and saved moments
are distinct from a new optimizer step.

After cleanup, `full_report.py` locates its arrays through `backup_manifest.json`.
To rerun measurement or the independent vector audit, restore the listed arrays
to their relative paths. The recorded measurement completed without interruption,
with all 60 arrays present and checked before archiving. A subsequent maintenance
fix requires both outputs plus matching checksums and code metadata before a
direct-stage restart can skip a part; missing metadata triggers recomputation.
This does not change the recorded measurement code hashes or its results.
