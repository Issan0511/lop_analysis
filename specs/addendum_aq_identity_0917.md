# aq_identity_0917 diagnostic addendum (post-result, 2026-09-17)

The original registered verdict remains FAIL: 100 scalar float32 comparisons
against an independently ordered forward exceeded the fixed 2e-5+2e-5 relative
tolerance. Float64 checks and both dtypes' actual norm budgets passed. This
addendum is registered after observing that result and before running the
diagnostic below. Do not loosen the original tolerance or overwrite its verdict.

Replay exactly the same frozen states and 256-step continuations in float32.
At each state, compare (a) the original elementwise-sum autograd forward and
(b) an autograd forward whose matrix-product/reduction order matches production.
Autograd, not the handwritten derivative, supplies gradients in both cases.
Record preactivation, output, and gate differences, every original failed
comparison and the corresponding matched-order comparison. Use the original
2e-5*(1+abs(reference)) tolerance. Diagnosis succeeds only if the original 100
failures reproduce and every matched-order comparison passes; otherwise report
unresolved discrepancies. Keep diagnosis and original registered verdict separate.

This tests implementation arithmetic, not any new biological/statistical hypothesis.
