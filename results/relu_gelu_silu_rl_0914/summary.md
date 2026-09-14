# relu_gelu_silu_rl_0914 report

G0: **PASS** (ELU1 AT_FLOOR, LR ABOVE_FLOOR)

Q1: **UNRESOLVED**

## RL floor calls (late online t41–50)

|Arm|late acc (mean)|seed 0|seed 1|seed 2|rollup|collapse task (per seed)|
|---|---:|---|---|---|---|---|
|LR|0.4882|ABOVE_FLOOR|ABOVE_FLOOR|ABOVE_FLOOR|**ABOVE_FLOOR**|— / — / —|
|ELU1|0.1022|AT_FLOOR|AT_FLOOR|AT_FLOOR|**AT_FLOOR**|23 / 24 / 16|
|R|0.1128|AT_FLOOR|AT_FLOOR|AT_FLOOR|**AT_FLOOR**|42 / 44 / 49|
|GELU|0.2206|ABOVE_FLOOR|ABOVE_FLOOR|ABOVE_FLOOR|**ABOVE_FLOOR**|— / — / —|
|SILU|0.1941|AT_FLOOR|AT_FLOOR|ABOVE_FLOOR|**DISAGREEMENT**|50 / 47 / —|

## PM late online accuracy (capability check, report only)

|Arm|late acc|rollup|
|---|---:|---|
|LR|0.9849|ABOVE_FLOOR|
|ELU1|0.9885|ABOVE_FLOOR|
|R|0.9398|ABOVE_FLOOR|
|GELU|0.9298|ABOVE_FLOOR|
|SILU|0.9666|ABOVE_FLOOR|

## RL layer 2 at t50 (seed mean, report only)

|Arm|tinygate (|φ′|<1e−8)|neggate (φ′<0)|gate mean|median z̄₂|
|---|---:|---:|---:|---:|
|LR|0.000|0.000|0.123|-7.47|
|ELU1|1.000|0.000|3.34e-05|-153.07|
|R|0.997|0.000|0.00333|-0.00|
|GELU|0.002|0.014|0.506|-0.00|
|SILU|0.644|0.554|0.133|-33.65|

## G1 anchors (diagnostic)

|Arm|vs layer_chimera|max abs Δ|seed 0|seed 1|seed 2|flag|
|---|---|---:|---:|---:|---:|---|
|LR|LL|0|0|0|0|BIT_IDENTICAL|
|ELU1|EE|0|0|0|0|BIT_IDENTICAL|
