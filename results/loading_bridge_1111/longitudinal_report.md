# POSTHOC EXISTING DATA: stage1_ctrl longitudinal audit

No new training. All five activations, seeds 0–9, tasks 1–30. Task 1 exists in every Qall file and exactly matches S for all nine stored record fields (450 checks). The ELU task-2 Qbranch/Qall records also exactly match for all nine fields (90 checks). S/Qbranch initial unit summaries match at tasks 2 and 30 (20 checks).

Each row in longitudinal_seed.csv is a seed/task/arm. Summaries contain seed means and between-seed sample SD, not confidence intervals. All 100 units are included; the registered ReLU eligibility filter is not applied to this post-hoc description. The deficit uses the same seed/task fresh arm.

m and s are each unit's preactivation mean and population SD over 1,200 inputs; the reported m, s, and r=m/s are medians over units, formed before the seed mean. The r statistic is the median of unit-wise ratios, not the ratio of the two medians. Derivative and p+ are means over units and inputs; p+ is the fraction with preactivation > 0. End statistics are after 4,000 updates. Head norm is the Frobenius norm after class-centering each head column.

The stored ELU derivative diagnostic uses the mathematical derivative; it does not directly measure Adam updates or establish causal mediation.

Task 30, means over 10 seeds:

| Activation | Qall norm start | Qall norm end | Qall confidence at start | S confidence at start |
|---|---:|---:|---:|---:|
| Leaky ReLU | 1.7393 | 3.9759 | 0.4138 | 0.4630 |
| ReLU | 1.7393 | 4.5294 | 0.3589 | 0.1158 |
| ELU | 1.7393 | 4.4078 | 0.4478 | 0.3843 |
| SiLU | 1.7393 | 5.2392 | 0.4154 | 0.3123 |
| GELU | 1.7393 | 5.2266 | 0.4034 | 0.1883 |

Qall start norm 1.7393 is the post-intervention task-30 value, not its end value. Norm regrowth is not itself a measurement of label alignment or kappa. Confidence is available only at update 0.

The ELU task-2 Qbranch and Qall curves coincide. At task 30, Qbranch shares the initial body with S, while Qall has a different history before update 0. Its wider body and higher opening/derivative cannot be attributed solely to changes during task 30.

Plots: longitudinal_overview.png/.pdf and longitudinal_readout_history.png/.pdf. The second plot uses a symlog update axis for the single-switch branches.

Source files and SHA256 hashes: longitudinal_provenance.json. Reproduce with the project Python environment and analysis/loading_bridge_1111/longitudinal.py.
