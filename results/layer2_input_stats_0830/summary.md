# Layer-2 input statistics: ordinary vs centering

> Existing exact 32-support logs from `mlp2_phase1_0829`; no new training run.

The layer-2 input is the first hidden ReLU activation.  `L2_A1` centers only
the raw input to layer 1; `L2_Aall` centers both layer inputs.

## Late window (tasks 451–500)

| arm | ||mu2|| | tr Sigma2 | ||mu2||/tr Sigma2 | ||mu2||^2/tr Sigma2 | dose |
|---|---:|---:|---:|---:|---:|
| L2_none | 2.5347 | 9.7607 | 0.28256 | 0.76171 | 8.4809 |
| L2_A1 | 1.4447 | 4.2526 | 0.34752 | 0.51916 | 7.1827 |
| L2_Aall | 0.5655 | 31.0262 | 0.01868 | 0.01214 | 1.0356 |

`dose = sqrt(100) * ||mu2||/sqrt(tr Sigma2)`.

## Interpretation

- Layer-1-only centering (`L2_A1`) reduces the scale-free mean ratio to 0.730x [0.599, 0.919] of ordinary, but does not
  eliminate it: ReLU regenerates a positive downstream mean.
- The literal requested ratio `||mu||/tr Sigma` moves to 1.230x [1.076, 1.511] under `L2_A1`; it can increase because
  it is not scale invariant and `tr Sigma` shrinks faster than `||mu||`.
- Centering the layer-2 input itself (`L2_Aall`) reduces the scale-free ratio to 0.0234x [0.0200, 0.0263] of `L2_A1`.
- A zero-mean Gaussian passed through ReLU has the reference ratio `1/(pi-1) = 0.4669`.  The observed `L2_A1` value 0.5192
  is close, identifying rectification itself as the main source of regenerated mean.
- Therefore ordinary observation centering does not remove the layer-2 mean mechanism;
  it attenuates it.  Direct per-layer centering is required to suppress it.

## Metric caution

Under a rescaling `h -> c h`, `||mu||/tr Sigma` changes as `1/c`.  The
dimensionless `||mu||^2/tr Sigma` (or its square root) is the safer comparison
across arms whose activation scales differ.
