# Dynamic loading bridge results

Complete fixed design: 540 MLP files and 40 ViT files; no smoke files included.

Bootstrap: 10,000 paired-seed resamples, percentile 95% intervals; strata are model/activation/task. The ViT has only 5 seeds. Intervals are descriptive and not multiplicity corrected.

Forward probabilities match at update 0 for T/TG, S/G and H10/T by construction. The stored CE/confidence/entropy aggregates and expected-gradient factors are checked in dynamic_initial_match_checks.csv. This does not establish matching probabilities through update 200. P/Q are single switch interventions; this is not sustained prevention.

Snapshot expected_feature_grad_rms is the per-sample uniform-target gradient at the head input (RMS). Track g_body is the actual minibatch raw body-parameter gradient (L2). d_body/body_path/body_displacement are actual Adam parameter movements. Their ratios must not be equated.

MLP layer statistics use all 1,200 training inputs. ViT layer statistics use the first 128 training images and all tokens; layer 0 is the first FFN probe. CSV files retain every measured layer. upper_over_s_med is median(max preactivation / s), not (max−median)/s. It includes location changes; centered tail shape is not separately identified.

CSV delta_* values subtract the same arm’s post-intervention update-0 state. Ratios exclude denominators with absolute value below 1e-10; n records the eligible seed count.

Trajectory/path figures show mean ± SE; the first-update ratio figure and CSV summaries use bootstrap 95% intervals. Recovery figures show individual seeds and censor marks.

## Update 1: T/TG raw gradient versus actual body movement

The two columns are means of paired seedwise ratios, each with its own eligible n. Near-zero raw gradients can coexist with movement from inherited Adam state.

| model | activation | t_end | raw_gradient_ratio_95CI | raw_gradient_n | Adam_body_step_ratio_95CI | Adam_body_step_n |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | ELU | 1 | 10 [10, 10] | 10 | 2.8045 [2.5404, 3.0949] | 10 |
| mlp | ELU | 10 | 10 [10, 10] | 10 | 2.2065 [2.0139, 2.3866] | 10 |
| mlp | R | 1 | 10 [10, 10] | 10 | 4.187 [3.904, 4.4683] | 10 |
| mlp | R | 10 | 10 [10, 10] | 3 | 1.7016 [1, 2.4564] | 10 |
| mlp | SILU | 1 | 10 [10, 10] | 10 | 3.73 [3.447, 4.0383] | 10 |
| mlp | SILU | 10 | 10 [10, 10] | 10 | 1.7844 [1.6401, 1.9184] | 10 |
| vit | SL | 100 | 10 [10, 10] | 5 | 3.8946 [3.4654, 4.3038] | 5 |


All registered pairs T/TG, S/G, H10/T, P/S and Q/S remain in dynamic_first_update_paired.csv (raw body gradient and actual Adam movement reported separately).

## Standardized body recovery

T90 is recorded only when observed on the evaluation grid (MLP every 100; ViT every 25 updates). Unreached values stay blank, with censor horizon recorded separately. No budget is substituted for T90. All-seed T90 contrasts receive a finite mean/CI only when both arms reach in every paired seed. Otherwise dynamic_recovery_t90_pairs.csv records censoring and one-sided pairwise bounds.

Recovery resets the readout and all Adam state. T90 excludes the preceding 200 acute updates (B0 has no acute phase), so H10/TG below B0 is not a comparison of total learning updates. It stops at the first observed accuracy ≥0.90. Final accuracy is measured at each run’s own stopping step, not at a common training horizon; report it together with steps_run. This probes body state including beneficial new-task learning, and does not by itself identify irreversible damage. The ViT S−B0 interval spans zero and does not support a common acute body-damage effect across seeds. H10/TG below B0 can include useful acute learning of the new task.

### ViT: every seed (5 seeds only)

| seed | B0 | S | P | H10 | TG |
| --- | --- | --- | --- | --- | --- |
| 10252010 | 550 | 650 | 550 | 400 | 450 |
| 10252011 | 600 | 750 | 575 | 575 | 625 |
| 10252012 | 600 | 575 | 650 | 525 | 325 |
| 10252013 | 950 | 775 | 700 | 675 | 600 |
| 10252014 | 500 | 550 | 375 | 275 | 275 |


ViT all-seed paired T90 differences (first arm minus second; negative is faster):

| comparison | n | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| H10-B0 | 5 | -150 | -230 | -70 |
| H10-S | 5 | -170 | -245 | -95 |
| P-B0 | 5 | -70 | -165 | 15 |
| P-S | 5 | -90 | -160 | -5 |
| S-B0 | 5 | 20 | -85 | 110 |
| TG-B0 | 5 | -185 | -295 | -60 |
| TG-S | 5 | -205 | -250 | -160 |


Accuracy and stopping time must be read together:

| arm | n | reached | censored | final_accuracy_mean | stopping_step_mean |
| --- | --- | --- | --- | --- | --- |
| B0 | 5 | 5 | 0 | 0.91184 | 640 |
| H10 | 5 | 5 | 0 | 0.91 | 490 |
| P | 5 | 5 | 0 | 0.9064 | 570 |
| S | 5 | 5 | 0 | 0.91136 | 660 |
| TG | 5 | 5 | 0 | 0.90824 | 455 |


### MLP late switch: 90% unreached counts at the 4,000-update limit

| activation | arm | n | reached | censored_at_4000 | final_accuracy_mean | stopping_step_mean |
| --- | --- | --- | --- | --- | --- | --- |
| ELU | B0 | 10 | 10 | 0 | 0.90708 | 2330 |
| ELU | H10 | 10 | 10 | 0 | 0.90817 | 2330 |
| ELU | P | 10 | 10 | 0 | 0.90708 | 2580 |
| ELU | S | 10 | 10 | 0 | 0.90817 | 2600 |
| ELU | TG | 10 | 10 | 0 | 0.90742 | 2330 |
| R | B0 | 10 | 0 | 10 | 0.19925 | 4000 |
| R | H10 | 10 | 0 | 10 | 0.194 | 4000 |
| R | P | 10 | 0 | 10 | 0.16067 | 4000 |
| R | S | 10 | 0 | 10 | 0.1635 | 4000 |
| R | TG | 10 | 0 | 10 | 0.19567 | 4000 |
| SILU | B0 | 10 | 1 | 9 | 0.88142 | 3980 |
| SILU | H10 | 10 | 0 | 10 | 0.87092 | 4000 |
| SILU | P | 10 | 0 | 10 | 0.75817 | 4000 |
| SILU | S | 10 | 0 | 10 | 0.75217 | 4000 |
| SILU | TG | 10 | 0 | 10 | 0.86425 | 4000 |


## ELU task-end 10 → task 11: single Q intervention followed to update 4,000

These are arm means, not paired differences. Activity, width, location, upper tail, feature scale, logit scale and readout regrowth are separate measurements. Q is applied once; later changes do not establish a fixed-head mechanism. Seed-level values and paired bootstrap intervals are in dynamic_snapshots.csv and dynamic_snapshot_paired.csv.

| step | arm | n | m_med | s_med | r_med | dphi_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Q | 10 | -9.1773 | 5.5486 | -1.6 | 0.075561 |
| 0 | S | 10 | -9.1773 | 5.5486 | -1.6 | 0.075561 |
| 200 | Q | 10 | -9.1973 | 5.5871 | -1.6315 | 0.072474 |
| 200 | S | 10 | -13.354 | 6.0562 | -2.1958 | 0.014936 |
| 1000 | Q | 10 | -7.6605 | 5.4916 | -1.3889 | 0.11802 |
| 1000 | S | 10 | -13.065 | 6.027 | -2.1712 | 0.017536 |
| 4000 | Q | 10 | -5.6939 | 5.8511 | -0.95703 | 0.21572 |
| 4000 | S | 10 | -9.8341 | 5.8267 | -1.6526 | 0.067366 |


| step | arm | n | upper_max_med | upper_q99_med | upper_over_s_med | body_weight_norm |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Q | 10 | 8.1168 | 3.1122 | 1.483 | 112.05 |
| 0 | S | 10 | 8.1168 | 3.1122 | 1.483 | 112.05 |
| 200 | Q | 10 | 7.9936 | 2.9012 | 1.4415 | 112.05 |
| 200 | S | 10 | 3.1547 | -0.92423 | 0.53102 | 111.74 |
| 1000 | Q | 10 | 10.234 | 4.6089 | 1.9041 | 113.76 |
| 1000 | S | 10 | 3.6401 | -0.79898 | 0.60787 | 112.2 |
| 4000 | Q | 10 | 14.269 | 7.797 | 2.4657 | 126.49 |
| 4000 | S | 10 | 8.3179 | 3.0465 | 1.4582 | 118.02 |


| step | arm | n | head_centered_norm | feature_centered_rms | logit_centered_rms |
| --- | --- | --- | --- | --- | --- |
| 0 | Q | 10 | 1.7393 | 0.80726 | 0.50694 |
| 0 | S | 10 | 16.224 | 0.80726 | 3.7036 |
| 200 | Q | 10 | 1.6662 | 0.78406 | 0.40679 |
| 200 | S | 10 | 15.903 | 0.24528 | 1.1462 |
| 1000 | Q | 10 | 2.4934 | 1.1315 | 0.92715 |
| 1000 | S | 10 | 15.884 | 0.30014 | 1.4277 |
| 4000 | Q | 10 | 6.3693 | 1.8772 | 3.2343 |
| 4000 | S | 10 | 16.349 | 0.79435 | 3.707 |


## Full-resolution outputs

- dynamic_snapshots.csv: every arm/seed/step/layer, including natural and recovery curves.
- dynamic_track.csv: every actual acute update.
- dynamic_paths.csv: complete 1–25 and 26–200 windows only.
- dynamic_*_summary.csv: arm means and seed-bootstrap intervals.
- dynamic_*_paired.csv: paired differences and mean of valid seedwise ratios.
- dynamic_*_paired_seeds.csv: auditable seed-level values and contrasts.
- dynamic_recovery.csv / dynamic_recovery_t90_pairs.csv: censor-aware endpoints and pairwise bounds.
- dynamic_provenance.json: input hashes, script/spec hashes and producing commits.
