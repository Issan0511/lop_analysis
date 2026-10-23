# L0 phase-2 saved-state diagnostic

Run from this repository with its existing PyTorch environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_l0phase2_1023/measure.py gate
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_l0phase2_1023/measure.py run
PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_l0phase2_1023/report.py
```

No optimizer is instantiated; no checkpoint is modified. Checkpoints are read
directly from the existing tar bundle. The archived u100 checkpoints and prior
force arrays are also read-only inputs. `gate` verifies the bundle hash and all
20 prior-force array hashes. Restarting `run` requires matching code and output
hashes, and reuses complete states only.

For a feature matrix F with normalized columns A=F diag(1/s), G=A' A,
the leave-one-out residual of identifiable column i is

    P_i F_i = s_i A G^+ e_i / (G^+)_{ii}.

Thus the registered score is

    <u_i,P_i F_i>/||P_i F_i||² = (u_i' A G^+)_i / s_i,

and the residual-energy fraction is `1/(G^+)_{ii}`. A column that participates
in the numerical nullspace is not individually identifiable; its score is NaN.
This is checked against independently computed least-squares projections and
against a duplicated-column example. Null columns are not called positive.

Scalar features are centered over loss positions to remove the constant
nuisance feature. Output-space features keep every downstream Jacobian,
including attention across positions. Position zero is removed from the loss,
not from the activation tangent: its influence on later positions is retained.
Each output vector is centered over classes, which removes the softmax-invariant
direction. The background feature is `center(logits) - sum_i J_i a_i`.

The output-space surrogate freezes downstream J and fits one scalar amplitude
per current unit direction, plus one background amplitude. It does **not** fit
arbitrary down-weight vectors. Residualization gives the independent-label
least-squares ensemble expectation in that local surrogate, not the actual CE
force. Actual CE forces, their gradients through all parameters, and finite
saved-point motions are measured separately. In particular, `switch` means
independent uniform labels on the current-task input probe, not a replay of
the next task's input stream.

`raw/*.npz` contains unit-level scores, Gram/cross matrices for output-space
projections, actual force projections, and the moments needed for saved-point
comparisons. It intentionally omits the large token-by-unit output-feature
matrix, which can be reconstructed from the checkpoint and the code. Final
archive locations and SHA-256 values are in `backup_manifest.json`.

Implementation setup: the environment has no SciPy; Spearman uses exact average
ranks for ties and NumPy correlation. The first import attempt stopped before
measurement, then the independent gate was run successfully. Roundoff handling
was made explicit as registered (relative inner-product tolerance 1e-10), and
the gate was rerun before the 10-seed run. Registered thresholds were unchanged.
