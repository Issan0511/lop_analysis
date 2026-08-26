# dynrepair_0826 result

This report applies the preregistered dynamic-repair decision rules.

A1 is an oracle intervention using the exact current-task loss; it is not a proposed learning method. A2 is a one-shot reset and does not represent continuous CBP.

## Verdicts

| test | window | estimate | 95% CI | verdict | precision |
|---|---|---:|---:|---|---|
| O-1 | short | -0.137500 | [-0.259895, -0.047862] | PERSISTENT | INSUFFICIENT |
| C-1 | short | +0.012383 | [+0.000254, +0.033546] | INCONCLUSIVE | OK |
| O-1 | long | -0.115159 | [-0.208873, -0.030831] | PERSISTENT | OK |
| C-1 | long | +0.015478 | [-0.064564, +0.068770] | INCONCLUSIVE | OK |
| F-1 | short | +0.276190 | [+0.137037, +0.400000] | DESCRIPTIVE_ONLY | INSUFFICIENT |
| Ch-a | long | +0.971989 | NA | DESCRIPTIVE_ONLY | NOT_APPLICABLE |
| Ch-b-dead | long | -0.041000 | [-0.064000, -0.020975] | INCONCLUSIVE | OK |
| Ch-b-frozen | long | -0.058000 | [-0.090000, -0.030000] | INCONCLUSIVE | OK |
| Ch-1 | long | NA | NA | DESCRIPTIVE_ONLY | NOT_APPLICABLE |

## Guards and sanity

- S2-source: PASS saved posreset S2=PASS; hash mismatches=[]
- S1: PASS float64 oracle mean; abs_error=1.7347234759768071e-18; max_kick_error=3.61e-16; float32 resume mean=0.0091482729587758459
- S2-probe-replay: PASS A0 probe/no-probe final hash differences=[]
- S2-source-log: PASS overlap_steps=201; rows=2010; common_columns=17; differing=[]; keys_ok=True
- S3: PASS max relative error in x^2+r^2=w_norm^2; by_arm={'A0': 4.315692663517959e-16, 'A1': 4.2566261420650875e-16, 'A2': 4.3885394644231556e-16, 'A3': 4.369936175316336e-16, 'A1_lo': 4.270023792663645e-16, 'A1_hi': 4.278474638029714e-16}
- S4: PASS strict_dead iff pre_max<=0 mismatches over finite records; by_arm={'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A1_lo': 0, 'A1_hi': 0}; non_finite_records_skipped=30900
- finite: REPORT raw non-finite values across all arms; by_arm={'A1_hi': {'pre_max': {'n': 30900, 'first_step': 500100}, 'x': {'n': 30900, 'first_step': 500100}, 'r': {'n': 30900, 'first_step': 500100}, 'w_norm': {'n': 30900, 'first_step': 500100}, 'b': {'n': 30900, 'first_step': 500100}, 'v': {'n': 30900, 'first_step': 500100}, 'utility_nmse': {'n': 30900, 'first_step': 500100}, 'eval_nmse': {'n': 309, 'first_step': 500100}}}; descriptive-only arms (A1_lo/A1_hi) are reported, not gated; note p_hat=mean(pre>0) stays finite (=0) where pre is NaN, so pre_max is the divergence indicator, not p_hat
- finite-judgment: PASS non-finite values in the judgment arms ['A0', 'A1', 'A2', 'A3']
- S5: PASS post-kick p_hat>0 rate (tautology check); n_target=915; after oracle optimisation p_hat>0 rate=0.906011; practical (p_hat>=0.05) rate=0.390164; post-oracle by_seed={0: 0.9647058823529412, 1: 0.8877551020408163, 2: 0.9690721649484536, 3: 0.8854166666666666, 4: 0.7938144329896907, 5: 0.9021739130434783, 6: 0.9012345679012346, 7: 0.9578947368421052, 8: 0.9473684210526315, 9: 0.8673469387755102}
- S7: REPORT A2 immediate dead_frac; mean U=0.23407525378257379; cbp_harm_0815 anchors are not on a verified identical setup
- S2-arm-stream: PASS compared 15 shared state keys against A0; mismatches={}; arms with an advanced method generator (A2 only is expected)=['A2']
- S6: PASS direct silencing comparison, n=20
- S8-A0: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.2s
- S8-A1: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.4s
- S8-A2: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.7s
- S8-A3: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.5s
- S8-A1_lo: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.6s
- S8-A1_hi: PASS raw_array_differences=[]; final_state_differences=[]; elapsed_main=71.0s
- G1: CLEAR F-1 valid-cell retained fraction
- G2: TRIGGERED number of imprecise CIs
- G3: TRIGGERED A1 practical revival rate
- G4: TRIGGERED maximum seed-level A3-induced death rate
- analysis-finite: REPORT non-finite values carried into traj/utility as NaN; by_arm={'A1_hi': {'pre_max': 30900, 'x': 30900, 'r': 30900, 'w_norm': 30900, 'b': 30900, 'v': 30900, 'utility_nmse': 30900, 'eval_nmse': 309}}

## Scope

- condA, width 100, period 10,000, batch 1, std encoding, snapshot step 500,000 only.
- F-1 remains observational because functional utility was not randomized.
- A3 matches total bias-space displacement, not unit count or per-unit displacement; G4 decides whether it can be read as an undirected control.
- A1_lo / A1_hi are kick-width sensitivity arms: descriptive only, never used by a verdict.
- source commit: 1183d3a898aa25262bb269fdf30c72a3c0f64860
- Non-finite records were kept, not dropped (§5). non-finite values carried into traj/utility as NaN; by_arm={'A1_hi': {'pre_max': 30900, 'x': 30900, 'r': 30900, 'w_norm': 30900, 'b': 30900, 'v': 30900, 'utility_nmse': 30900, 'eval_nmse': 309}}
