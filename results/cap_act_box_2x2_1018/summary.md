# cap_act_box_2x2_1018

Implementation is complete on native F51/RC training engines. The paper-derived forecast is ACTIVATION_SPLITS; no empirical scientific verdict is available.

CPU implementation checks passed: native activation/gradient behavior (including zero), native F51 host and inherited cap12 trajectory byte equality at R=10, native RC update/moment equality at R=10, cap timing/radii and bias scaling, moment/output preservation, calibration-seed separation, unclipped learning-rate schedule, paired windows/statistics, actual image content hashes and stream manifests. Calibration admission additionally rejects failed cap status, cap violations, missing fresh controls and stale source/evidence. All 12 specified source/data mutations were detected. CPU fixtures are explicitly synthetic implementation checks, not shortened production experiments.

CUDA graph/R=10 checks remain BLOCKED_CUDA. The registered `run` command was attempted and exited 2 before training because CUDA is unavailable, G0's actual host-reproduction/mutation proof is incomplete, and the full CUDA-inclusive S suite is therefore incomplete. `all_pass` remains false. `preflight.json` records the precise attempted workload and blockers; `verdict.json` assigns no scientific label.

Calibration (seeds 100–109), all four ref/cap12 evaluation cells (seeds 0–9), and the cap12b/lr_only separation arms have not run. Historical cells are scheduled for native reruns rather than unsupported reuse. No task/seed count or device substitution was made.

Commands and recovery limitations are in `analysis/cap_act_box_2x2_1018/README.md`.
