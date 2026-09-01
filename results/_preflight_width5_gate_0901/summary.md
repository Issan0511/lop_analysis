# width5_gate_0901 preflight result

Status: **PREFLIGHT_FAILED_S_CAP**  
Executed: 2026-09-01 (Asia/Tokyo)  
Implementation commit: `240b713`  
Command: `OMP_NUM_THREADS=1 .venv/bin/python -m src.width5_gate_0901 --preflight`

## Registered checks

| Check | Result |
| --- | --- |
| S-omp | PASS |
| S-offset | PASS |
| S-lin | PASS |
| S-mob | PASS |
| S-rank | PASS |
| S-floor | PASS |
| S-initial-pairing | PASS |
| S0′ (30k replay) | PASS |
| **S-cap** | **FAIL** |

## S-cap failure

The frozen rule requires every seed in each calibration arm to attain an early-window (`task 2–11`) minimum `unfit < 0.05`; equivalently, the maximum of the 20 per-seed minima must be below `0.05`.

| Arm | Smallest per-seed minimum | Largest per-seed minimum | Seeds attaining `< 0.05` | Result |
| --- | ---: | ---: | ---: | --- |
| R5 | 0.0608690874 | 0.6744219931 | 0/20 | FAIL |
| LIN5 | 0.1548528769 | 0.7057135806 | 0/20 | FAIL |

The onset threshold is therefore not demonstrated to lie above the registered width-5 attainable floor by this preflight. Per the frozen spec, the 8-arm 5M main run was not started, G0 is invalid, and G1 is not promoted to a primary verdict.

Exact per-seed values and all check records are in `preflight.json`; replay and S-cap raw logs are retained below this directory.
