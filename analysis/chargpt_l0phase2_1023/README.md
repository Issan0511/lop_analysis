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

The post-result probes are `validate64.py` (C3 seed0, both registered endpoints),
`contextual.py` (Markov-context covariance) and `headcov.py` (unrestricted final
head, instead of scalar unit amplitudes). Their measurement definitions were
committed in spec sections 7--9 before the respective measurements. These do
not overwrite the primary judgments. In particular, unit-level strength/rank
prediction is an additional empirical test, not a necessary condition of a
theorem about the old-label expectation.

The first contextual run stopped at C3 seed6 task100: explicitly multiplying
`G+ G G+` lost relative precision (1.56e-4, condition number 3.12e8). The rerun
factors the covariance through the dual basis, using QR/SVD above condition
number 1e6; the registered rank tolerance is unchanged. The failing state's
independent factor check then had relative error 4.19e-11, and its QR/SVD iid
score differed from the original Gram score by 2.30e-10. The failed log is
retained. This changes numerical evaluation, not the model or pass thresholds.
