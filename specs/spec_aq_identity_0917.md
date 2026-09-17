# aq_identity_0917 — CondA SGD norm-budget audit

Registered before execution, 2026-09-17. User request: doubt the derived A/Q
identities and independently verify every term against real CondA or PM updates.
Scope: V10 H1/A1 arithmetic and implementation validity, not a new mechanism claim.

## Data and scope

Use existing CondA checkpoints from `zero_attraction_0913/training/ckpts` in the
research-data archive. Arms: LR_a0p1_q0, LR_a0p3_q0, LR_a0p7_q0,
LR_a0p3_qp05. Steps: 0, 200000, 1000000, 5000000. All ten saved seeds and all
32 free-bit support points. No new long training run. These are previously studied
checkpoints; the audit protocol is registered, not independent replication data.
The host is a scalar-output 20-100-1 MLP, unhalved squared loss `(f-y)^2`, plain
batch-1 SGD; W, b, v and output bias all update simultaneously from pre-step values.
The archived input offset is applied before the network, as in the original code.

Audit both native float32 and float64 versions of the saved state. Native uses
the archived float32 input-offset boundary; float64 uses double-precision input
subtraction and inference on the same saved tensors. They are separate worlds,
not assumed to reproduce the same trajectory. Use the existing `VecMLPL`
forward, handwritten gradients and `sgd_step_layers` from this experiment's
committed source. Hash checkpoints and relevant source files in provenance.

## Coordinates and identities

Keep three quantities distinct: (1) whole W, (2) row-centered Wc = W - rowmean(W),
(3) the original CondA free-coordinate block W[...,15:]. The conversation's Wtilde
is (2), not (3). For row centering, xc = x - coordinate_mean(x), zc_i = Wc_i dot xc,
h_i = 2(f-y) v_i phi'(z_i), and Gc_i = h_i xc. All refer to the same pre-step state.

- `A_i = <Wc_i,Gc_i> = h_i zc_i`.
- `Q_i = ||Gc_i||^2 = h_i^2 ||xc||^2`.
- `A_i = S_i + H_i - B_i - C_i`, with u_i=2(f-y)v_i,
  S_i=u_i phi(z_i), H_i=u_i[z_i phi'(z_i)-phi(z_i)],
  B_i=b_i h_i, C_i=rowmean(W_i) sum(x) h_i.
- `D_i = ||Wc_new_i||^2 - ||Wc_old_i||^2 = -2 eta A_i + eta^2 Q_i`.
- Actual-displacement geometry: `D_i = 2<Wc_old_i,dWc_i> + ||dWc_i||^2`.
- Within one row, radial/orthogonal decomposition of dWc has the same budget;
  do not confuse per-row projections with one projection of the whole layer.
- In a multi-step segment, sum the pre-step budgets, including all evolving
  parameters; do not square the sum of gradients as a replacement for sum Q.
- Full-batch check on the 32 support points: A averages, whereas
  Q_batch = ||mean_s Gc_s||^2 = mean_{s,t} h_s h_t <xc_s,xc_t>, per row.
  It is generally not mean_s Q_s. Expected batch-1 Q is mean_s Q_s.

## Independent paths and measurements

1. Evaluate expanded formulas from pre-step residual, v, analytic phi' and inputs.
2. Independently differentiate a separately expressed PyTorch forward and MSE
   with autograd. Compare W, b, v, output-bias gradients to the production path;
   compute A and Q from the autograd gradient without using h/zc formulas.
3. Call the production optimizer step, save W before and after, and compute D
   directly. Promote stored weights to float64 for norm diagnostics to avoid
   extra float32 reductions. Compare to formulas and actual-displacement identity.
4. Fixed-state learning rates 0.0001, 0.001, 0.005. These are one-step probes,
   not learning-rate sweep trajectories. Save per-unit A,Q,D and pre-update z.
5. Continue each arm's step-200000 state for 256 updates at its saved lr (0.005),
   for each dtype. Draw support indices with NumPy PCG64 seed 20260917, one index
   per saved seed per step. Keep the task/teacher/input offset fixed. This is a
   new short continuation, not exact replay of the original RNG stream and not
   a complete task. Compare endpoint norms to the cumulative budget.
6. Positive/negative/zero preactivation groups are diagnostic groupings of
   sample-unit events. Save counts, signed linear contribution, quadratic
   contribution, actual norm change, sign of A and significant shrink/grow
   counts. Do not infer a universal positive/negative-side mechanism from these.

## Acceptance, falsification controls and outputs

Float64 gradient/A/Q checks: absolute tolerance 1e-10 plus relative 1e-10.
Float32 gradient/A/Q checks: absolute 2e-5 plus relative 2e-5 (including differently
ordered reductions). Norm-budget tolerance is cancellation/rounding aware:
`64*eps(dtype)*(1+R_old^2+R_new^2+abs(linear)+quadratic)` per row. Report actual
absolute and term-scaled errors, not only pass/fail. Actual-displacement geometry,
computed in float64, uses 64*eps(float64) times the analogous scale. Significant
sign comparisons require |predicted D| > the norm-budget tolerance; classify
smaller changes as unresolved at this precision instead of calling them a match.
Cumulative tolerance is the sum of local tolerance bounds.

Deliberately wrong formulas must be detected in float64 for at least one
nondegenerate state: omit the MSE factor 2; replace zc with full z; use uncentered
x in Q; omit H on the offset arm; substitute mean sample Q for full-batch Q.
At least one significant shrink and growth event must be covered, with both
positive and negative preactivations represented. If coverage fails, report
INCOMPLETE rather than weakening the criteria after observing results.

PASS requires every correctly scoped identity to pass, zero significant sign
mismatches, and all registered falsification/coverage controls to work. Otherwise
FAIL (or INCOMPLETE for missing input/coverage) and identify the failed link.
Finite-precision discrepancy is recorded separately from a wrong analytic term.
No claim about long-term LoP, norm-growth mechanisms or optimal lr follows from
an algebraic PASS.

Commit `summary.md`, `verdict.csv`, `checks.csv`, `groups.csv`,
`continuations.csv`, `batch_checks.csv`, figures and `provenance.json` under
`results/aq_identity_0917/`. Store complete per-unit arrays and small continuation
snapshots under the external raw-data archive; commit a SHA256 backup manifest.
Report the result to the user and create a bounded result note in Obsidian.
