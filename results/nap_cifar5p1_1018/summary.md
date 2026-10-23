# NaP on F51 — GPU production complete

Full calibration (seed100–109) and evaluation (seed0–9), each 30 tasks, completed on CUDA. Calibration evidence was fixed before evaluation. Fresh current-runtime R/SNA/KKT1 controls passed the unchanged reuse qualification.

## Registered outcome

Primary window: online accuracy on hard tasks 21, 23, 25, 27, 29, averaged within each seed, then over 10 evaluation seeds.

| Method | Mean online accuracy |
|---|---:|
| NaP | 38.507933% |
| R | 43.660737% |
| SNA | 66.477404% |
| KKT1 | 66.388381% |

NaP loses to KKT1 and SNA in all 10 paired seeds. Both registered labels are COMPARATOR_WINS; two-sided exact sign tests with Holm correction give p=0.00390625 for each comparison.
Mean paired differences: NaP−KKT1 -27.880449 percentage points; NaP−SNA -27.969471 percentage points.
NaP−R is report-only: mean difference -5.152804 percentage points; no registered primary significance label is assigned to that comparison.

## Prediction provenance

The original registration and unchanged verdict field say OPEN_COMPARISON for SNA/KKT1. Issa added KKT1 > NaP and SNA > NaP before NaP production in amendment commit 761530e3eb6fd00f0a2c85f6f31aed2f3a0d5818. Both added directions receive the required COMPARATOR_WINS result. The pre-existing direction that NaP improves over R is not borne out by the observed mean.
Amendment: `specs/amendments/spec_nap_cifar5p1_1018_prediction_20261018T020813JST.md`; SHA256 `e432d8558629ec7a56aba649d51c5424deaa1bfed359640aace161f3cf78a5f7`.
KKT1 versus SNA ranking was not tested here. This result is limited to the registered F51/std/ReLU setting, fixed learning rate 1e-4, and the specified NaP implementation; it does not establish a universal or RL ranking.

## Verification

`verdict.json` is the registered result. A separate CSV read verified complete 10-seed × 30-task coverage for all four methods, independently recomputed the specified window means, paired differences and all-negative exact-sign p-values, and matched the saved verdict. The earlier CUDA-blocked summary is preserved as `summary_pre_gpu.md`.
