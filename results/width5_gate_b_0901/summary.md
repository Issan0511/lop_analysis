# width5_gate_b_0901 summary

## Registered verdict

- G0: **PHENOMENON3_NOT_REPRODUCED**
- G0 compares each width-5 nonlinear arm with LIN5 seed by seed; it is not an absolute LoP onset rate.
- G0b crossing is a registered secondary endpoint.
- G1 levels, G2 width levels, and all G3 mechanism metrics are REPORT_ONLY.
- Numeric divergence: none

## G0 paired signs (terminal task 491–500)

| arm | k above LIN5 | valid n | CP95 | label |
|---|---:|---:|---|---|
| R5 | 13 | 20 | [0.407811, 0.846091] | **R5_INCONCLUSIVE_WIDE** |
| LR5 | 0 | 20 | [0, 0.168433] | **LR5_BELOW_LINEAR** |
| E5 | 1 | 20 | [0.00126509, 0.248733] | **E5_BELOW_LINEAR** |

## G0b crossing (early window minimum, task 2–11 → terminal window mean, task 491–500)

| arm | early better than LIN5 | crossings | valid n | CP95 | status |
|---|---:|---:|---:|---|---|
| R5 | 19 | 12 | 20 | [0.360543, 0.80881] | R5_CROSSING_DEFINED |
| LR5 | 20 | 0 | 20 | [0, 0.168433] | LR5_CROSSING_DEFINED |
| E5 | 20 | 1 | 20 | [0.00126509, 0.248733] | E5_CROSSING_DEFINED |

## Endpoints

| arm | width | activation | median log10 U 5M | submerged frac 5M |
|---|---:|---|---:|---:|
| R5 | 5 | relu | -0.133634 | 0.88 |
| LR5 | 5 | leaky | -0.418252 | 0.55 |
| E5 | 5 | elu | -0.395672 | 0.6 |
| LIN5 | 5 | linear | -0.289139 | 0 |
| R100 | 100 | relu | -0.310581 | 0.982 |
| LR100 | 100 | leaky | -2.23904 | 0.5795 |
| E100 | 100 | elu | -2.55037 | 0.407 |
| LIN100 | 100 | linear | -0.26035 | 0 |

## G1 paired levels (REPORT_ONLY)

- LR5_minus_LIN5: terminal Pearson = 0.919641; status = REPORT_ONLY
- E5_minus_LIN5: terminal Pearson = 0.771223; status = REPORT_ONLY

## Floor characterization (not a gate)

- R5 early-window per-seed minima: min 0.0608691, median 0.163238, max 0.674422; below 0.05 = 0/20.
- LIN5 early-window per-seed minima: min 0.154853, median 0.318303, max 0.705714; below 0.05 = 0/20.

## LIN5 reference finiteness

- LIN5 early: finite 20/20; nonfinite seeds: none
- LIN5 5M: finite 20/20; nonfinite seeds: none
- R5 vs LIN5 excluded seeds: none
- LR5 vs LIN5 excluded seeds: none
- E5 vs LIN5 excluded seeds: none

## Prediction provenance

- Carry-over predictions retain the status frozen in P2.
- New N1–N3 entries are reported with the provenance stored in config_used.yaml.

## Sanity

- preflight: **PASS**
- final_pairing: **PASS**
