# Layerwise activation chimera RL/PM report

Q1: **FLOOR_IN_DEEP_LAYER**
RL EE was AT_FLOOR; EL was ABOVE_FLOOR and LE was AT_FLOOR.

## RL floor calls

|Arm|Seed 0|Seed 1|Seed 2|Arm rollup|
|---|---|---|---|---|
|EE|AT_FLOOR|AT_FLOOR|AT_FLOOR|AT_FLOOR (all 3 seeds AT_FLOOR)|
|EL|ABOVE_FLOOR|ABOVE_FLOOR|ABOVE_FLOOR|ABOVE_FLOOR (all 3 seeds ABOVE_FLOOR)|
|LE|AT_FLOOR|AT_FLOOR|AT_FLOOR|AT_FLOOR (all 3 seeds AT_FLOOR)|

Q2: **WIDTH_SET_BY_L2**
HIGH/LOW boundary = sqrt(g_EE * g_LL) = 1.24563873.

|Arm|sd2(t2)|sd2(t8)|g|Class|
|---|---:|---:|---:|---|
|EE|1.66399896|12.1072083|1.39202968|HIGH|
|EL|2.4811852|3.6332283|1.0656279|LOW|
|LE|0.785013497|6.03672361|1.40492787|HIGH|
|LL|1.10866344|2.12624836|1.11464279|LOW|

## Q3: PM late online accuracy

Mean across 3 seeds; late window is tasks 41–50.

|Arm|Late online accuracy|
|---|---:|
|EE|0.988470825|
|EL|0.989151726|
|LE|0.98606804|
|LL|0.98486562|

## G1 historical anchors

|Comparison|Overall max abs delta|Seed 0|Seed 1|Seed 2|Flag|
|---|---:|---:|---:|---:|---|
|EE_vs_RL_ELU1_ref|0|0|0|0|OK|
|LL_vs_RL_LR_ref|0|0|0|0|OK|

Anchor flags are diagnostic and do not change Q1 or Q2.

## Interpretation notes

- Q1 arm rollup resolves the preregistration gap as follows: ABOVE_FLOOR requires all 3 seeds individually ABOVE_FLOOR; AT_FLOOR requires all 3 individually AT_FLOOR; any disagreement makes that arm unresolved and forces Q1 UNRESOLVED when the arm is needed.
- Floor F, sample standard deviation (ddof=1), and threshold F + 3*s/sqrt(10) are computed separately for each seed/environment from that run's actual task 41–50 labels; PM calls are reported in levels.csv but do not enter Q1.
- Q2 is descriptive, not a causal test of width causing collapse. PM ceiling behavior is not generalized. Forward activation and backward gate are coupled within each layer, so their separate effects are not identified.
- This scope is 3 seeds, one 80-epoch box; it is distinct from the historical 400-epoch RL setting.
