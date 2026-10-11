# CNN self contribution and effective input directions (posthoc, 1011)

Question: after observing c1 variance shrink and mean rise, quantify whether a self component contributes to actual upward Adam updates, then resolve weight changes by input directions.

This is exploratory posthoc work prompted by the original seed0 animation. No causal mechanism is assumed from co-occurrence. The original `rlcifar_cnn_0908/LR` has pooled histograms for all50 tasks and float16 fixed probes at nine tasks; it did not save full model/Adam state. Its c1 geometry can be reconstructed approximately. Actual update attribution instead uses the saved `cnn_drive_verify_1009/LR` seed10–19 checkpoints and explicitly reports that different trajectory. The bundled implementation is not bit-identical to the original sequential host.

## Attribution definition

Keep the full CNN and actual labels. For image n let f be logits, a_j the first-layer activated channel (pool selection included in the downstream Jacobian), A_j=df/da_j, v_j=A_j a_j and Q(f)=integral_0^1 Hessian(logsumexp)(t f)dt. Then p(f)-uniform=Q(f)f.

Decompose dL/da_j into A_j^T Q v_j (local linear CE diagonal-channel self), A_j^T Q(f-v_j) (cross-channel and affine remainder), and A_j^T(uniform-onehot(y)) (actual label term). Record quadrature and floating residuals separately. This is an explicit additive operational definition, not a claim of a unique self mechanism. In particular, the old channel-deleted network gradient and a regression of cotangent on activity are not substituted for this definition.

Replay the first75 updates of task31 from task30 states, initially all10 saved seeds. These are switch-window updates, not a reconstruction of seed0 task40→50. Use the original engine's exact forward, full loss, labels, batch order and Adam formulas. Compare the decomposed replay to an undecomposed reference over the same window. Increase the window only if required to resolve the measured question; distinguish any extension from original saved runs.

Track self, cross, label and inherited first-moment history using the linear Adam first-moment recurrence. All components use the actual full second moment and denominator. Numerical/parameter-rounding residual is explicit. For c1, mean change is exactly (mu,1) dot delta(W,b), so component sums and their time integrals must close to actual observed movement. Report signed contributions before shares, with upward+shrinking channels and other channels separately. Do not infer a late-task or long-run cause from switch-window evidence alone.

## Directions

For original seed0 (and seeds0–9 as a robustness extension if reconstruction validates), reconstruct c1 from512 fixed float16 preactivation probes; compute full1200-image×1024-position padded-patch mean/covariance in float64. Check against independent channel means and saved median norms/std; held-out probes estimate reconstruction error.

In the fixed raw-input covariance eigenbasis, resolve each endpoint change exactly: delta norm squared =2 a delta a+(delta a)^2 and delta variance =sum lambda[2 a delta a+(delta a)^2]. Separate bias and input-mean contributions to mean displacement. Report all16 channels, especially the previously observed channel10 exception; compare20→30,30→40,40→50,20→50 without selecting an interval based on the outcome.

Also define an activation/pool effective-input matrix Psi_j with rows phi'(z_winner)*(x_winner,1). Use a fixed reproducible image subset and a fixed start-state eigensystem per comparison, keeping gate/pool changes separate from parameter movement. Raw covariance directions and Psi directions are different quantities and must not share labels.

## Verification and outputs

Required: gradient closure; Qf identity convergence; native-reference replay comparison; exact c1 mean-change closure; direction-accounting closure; reconstruction fidelity with quantified approximation. No multi-channel sample is treated as an independent seed. Snapshot algebra is not labelled an intervention. Save code, numerical tables, provenance and a readable summary; back up raw binaries with SHA256 before worktree cleanup. Original data are read-only.

## Sequential extension after observing the switch window

The first75-step result had80/160 upward+shrinking channels and61/80 positive self contributions. This motivated a phase-sensitivity extension, not an independent preregistered test: advance the same10 saved runs through task31 and measure its final75 updates (steps29926–30000). Start a fresh component-history ledger at that window; do not treat its history bucket as absence of earlier self effects. This does not extend the original seed0 animation trajectory. Retain both windows regardless of outcome.
