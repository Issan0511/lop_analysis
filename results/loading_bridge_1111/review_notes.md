# loading_bridge_1111 validation and independent review

2026-11-11. Reviews were performed before main-run interpretation; registered main training code was not altered during execution.

- Existing-data audit checked the exact head/bias/Adam interventions and source arrays. Corrections: Qall norm1.739 is a task-start value; old ELU vfreeze approximately−2.3 is a 500-update all-input-median change after long adaptation, not an initial expected gradient. The existing ELU moment/direction/null tests were not rerun as new evidence.
- Pre-main code review corrected transformed old/new margins to include the forward logit gain and strengthened resume compatibility checks. Synthetic forward/backward identities and frozen-probe invariants passed before the main dynamic run. The main dynamic implementation is commit c331ea27.
- Frozen analysis: 28 synthetic checks and 3180 real-data checks passed. MLP float32-feature versus original float64-body diagnostic differences were reported rather than silently equated.
- Independent statistical review checked paired-seed bootstrap and censored T90 handling. The dynamic ratio is the mean of eligible seed ratios; frozen A primarily uses a ratio of means. Censor caps are never substituted as observed T90. Recovery differences receive a bootstrap interval only when all paired seeds reach90%.
- The analysis validator was strengthened to require a complete regular recovery evaluation grid, a within-budget endpoint, observed T90 equal to the endpoint, and censored trajectories reaching the cap. Valid and deliberately malformed fixtures were checked; main raw files were not edited.
- Interpretation review distinguished head-input gradient norm from its radial product, radial product from cosine with the original gradient, MLP preactivation push from Adam displacement, initial probability matching from full-trajectory matching, and acute body recovery from irreversible damage or long-term prevention.
- Existing-data longitudinal checks: task1 S/Qall450 record-field comparisons; ELU task2 Qbranch/Qall90 comparisons; S/Qbranch initial-body20 comparisons all exactly matched. Figures were visually inspected.
- ELU interpretation review confirmed checkpoint task-end indexing (t_end10 means task11 learning), feature centering, and max(z)/s as a standardized upper location rather than a centered-tail shape measure. Width medians alone do not characterize every unit's width changes.

Main output completeness and exact counts are recorded in dynamic_provenance.json and dynamic_summary.md. Raw checkpoint and output hashes are in provenance files and backup_manifest.json. No result is dropped on effect direction or significance.
