# mu_titration_0823 Phase 0

- Final implementation commit: `dcabc43a75323370b529c38dca767395df6e7bc3`
- Command: `OMP_NUM_THREADS=1 .venv/bin/python -m src.mu_titration --config configs/mu_titration_0823.yaml --selfcheck`
- Result: **PASS** for all eight preregistered alpha arms.
- Cross-alpha reset fingerprint: `a45b608804e0b2487b8a419450f0f273098f04273b0ad1cbaae5ac005616dc3f` (identical for all arms).
- Structural checks: legacy/new logger schema, float32 public values, read-only probe, `M`/`b+M`/`cos_crit` identities, direct `delta_s` closure, `p_hat` quantization, finite statistics, and deterministic 32-support S3 reweighting all passed.
- Every required synthetic S3 negative case failed as intended; extreme z diagnostics did not affect PASS/FAIL, as fixed by the second post-hoc addendum.
- The input-only calibration and alpha grid were frozen in `specs/spec_mu_titration_0823.md` before the formal sweep. No `p_hat`, theta, strict-dead, loss, or dose-response outcome was inspected during Phase 0.
