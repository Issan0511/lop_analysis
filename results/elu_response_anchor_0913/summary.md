# ELU response-anchor results

**Preregistered 3-seed pilot.** Seeds are the replicate unit; tasks and units are not replicates. The primary outcome is mean online minibatch CE over all 6,000 updates of task 21.

A=F0K0, B=F0K1, C=F1K0, D=F1K1; AF/BF freeze layer 1. Positive A-B means the K1 response architecture lowers online CE within F0.

## Primary per seed

| seed | A online CE | B online CE | A-B benefit |
|---:|---:|---:|---:|
| 0 | 2.318781 | 0.992598 | 1.326183 |
| 1 | 2.295867 | 0.878437 | 1.417429 |
| 2 | 2.313137 | 0.952278 | 1.360859 |

Mean A-B = 1.368157; t95% CI with df=2 [1.253740, 1.482574]. Pilot label: **DIRECTIONAL_RESPONSE_SUPPORT**. This is a directional pilot threshold, not a population-significance claim.

## Task 21 response contrasts

| seed | C-D | (C-D)-(A-B) | AF-BF |
|---:|---:|---:|---:|
| 0 | 2.642552 | 1.316369 | 0.269274 |
| 1 | 2.213977 | 0.796548 | 0.370216 |
| 2 | 3.028559 | 1.667700 | 0.294636 |
| mean | 2.628363 | 1.260206 | 0.311375 |

All three are registered secondary descriptions and receive no primary support label. AF/BF freeze layer 1, while layer 2 and output parameters remain trainable.

## Endpoint accuracy and durability

| task | A acc % | B acc % | C acc % | D acc % | AF acc % | BF acc % |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 12.111 | 74.889 | 16.250 | 20.056 | 11.750 | 35.639 |
| 22 | 11.778 | 72.500 | 23.528 | 14.750 | 11.556 | 36.056 |
| 23 | 12.833 | 75.278 | 28.528 | 13.833 | 12.306 | 36.333 |
| 24 | 11.111 | 74.083 | 32.194 | 11.556 | 11.278 | 34.444 |
| 25 | 11.361 | 73.361 | 26.750 | 11.000 | 11.556 | 33.944 |

Full-probe AUC, early-75 probe AUC, first-75 online means, endpoint CE/accuracy, taskwise durability, C-D, (C-D)-(A-B), AF-BF, total ordinary lift A-D, and feature effects A-C/B-D are registered secondary descriptions in `contrasts.csv` and `verdict.csv`.

C-D and A-B compare K within a fixed F state. A-C and B-D change F and therefore start from different functions; they are feature effects with an initial-difference label, not pure update effects. `initial_shock.csv` reports those step-0 differences separately.

The intervention identifies an architecture response under matched inputs. Within each F state, corresponding branches have the same full initial function; initial W/Adam state is identical as registered. Output equivalence is initial only. Do not interpret any contrast as a natural mediated percentage.

## Source and validation

- rows.csv SHA256 `fad9b8f8f776d701f2c34406bebf762b5d3882203eb9bd13b2f01927c929e2a9`
- learning.npz SHA256 `be8c1feb08429f67c7e1fa6eb952edadee6342f4442abfe89f4e252856fbe21d`
- provenance status `COMPLETE`; validation status `PASS`
- Exact required grid: 3 seeds x 6 branches x 5 tasks x 15 full-probe steps; dense online shape 5 x 18 x 6000.
