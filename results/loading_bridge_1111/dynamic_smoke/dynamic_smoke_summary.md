# Dynamic loading bridge results

**SMOKE SCHEMA CHECK ONLY; no experimental conclusion.**

Bootstrap: 10,000 paired-seed resamples, percentile 95% intervals; strata are model/activation/task. The ViT has only 5 seeds. Intervals are descriptive and not multiplicity corrected.

Forward probabilities match at update 0 for T/TG, S/G and H10/T by construction. The stored CE/confidence/entropy aggregates and expected-gradient factors are checked in dynamic_initial_match_checks.csv. This does not establish matching probabilities through update 200. P/Q are single switch interventions; this is not sustained prevention.

Snapshot expected_feature_grad_rms is the per-sample uniform-target gradient at the head input (RMS). Track g_body is the actual minibatch raw body-parameter gradient (L2). d_body/body_path/body_displacement are actual Adam parameter movements. Their ratios must not be equated.

MLP layer statistics use all 1,200 training inputs. ViT layer statistics use the first 128 training images and all tokens; layer 0 is the first FFN probe. CSV files retain every measured layer. upper_over_s_med is median(max preactivation / s), not (max−median)/s.

CSV delta_* values subtract the same arm’s post-intervention update-0 state. Ratios exclude denominators with absolute value below 1e-10; n records the eligible seed count.

Trajectory/path figures show mean ± SE; the first-update ratio figure and CSV summaries use bootstrap 95% intervals. Recovery figures show individual seeds and censor marks.

## Update 1: T/TG raw gradient versus actual body movement

The two columns are means of paired seedwise ratios, each with its own eligible n. Near-zero raw gradients can coexist with movement from inherited Adam state.

| model | activation | t_end | raw_gradient_ratio_95CI | raw_gradient_n | Adam_body_step_ratio_95CI | Adam_body_step_n |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | R | 1 | 10 [10, 10] | 1 | 4.0804 [4.0804, 4.0804] | 1 |
| vit | SL | 100 | 10 [10, 10] | 1 | 3.113 [3.113, 3.113] | 1 |


All registered pairs T/TG, S/G, H10/T, P/S and Q/S remain in dynamic_first_update_paired.csv (raw body gradient and actual Adam movement reported separately).

## Standardized body recovery

T90 is recorded only when observed on the evaluation grid (MLP every 100; ViT every 25 updates). Unreached values stay blank, with censor horizon recorded separately. No budget is substituted for T90. All-seed T90 contrasts receive a finite mean/CI only when both arms reach in every paired seed. Otherwise dynamic_recovery_t90_pairs.csv records censoring and one-sided pairwise bounds.

Recovery resets the readout and all Adam state. It stops at the first observed accuracy ≥0.90. Final accuracy is measured at each run’s own stopping step, not at a common training horizon; report it together with steps_run. This probes body state including beneficial new-task learning, and does not by itself identify irreversible damage.

## ELU task 10: single Q intervention followed to update 4,000

These are arm means, not paired differences. Activity, width, location, upper tail, feature scale, logit scale and readout regrowth are separate measurements. Q is applied once; later changes do not establish a fixed-head mechanism. Seed-level values and paired bootstrap intervals are in dynamic_snapshots.csv and dynamic_snapshot_paired.csv.

## Full-resolution outputs

- dynamic_snapshots.csv: every arm/seed/step/layer, including natural and recovery curves.
- dynamic_track.csv: every actual acute update.
- dynamic_paths.csv: complete 1–25 and 26–200 windows only.
- dynamic_*_summary.csv: arm means and seed-bootstrap intervals.
- dynamic_*_paired.csv: paired differences and mean of valid seedwise ratios.
- dynamic_*_paired_seeds.csv: auditable seed-level values and contrasts.
- dynamic_recovery.csv / dynamic_recovery_t90_pairs.csv: censor-aware endpoints and pairwise bounds.
- dynamic_provenance.json: input hashes, script/spec hashes and producing commits.
