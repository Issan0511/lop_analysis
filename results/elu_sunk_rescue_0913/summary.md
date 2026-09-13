# One-shot ELU sunk-unit rescue results

Preregistered 3-seed pilot; same1200 images and6000updates per new task. All outcomes below concern fresh-task learning, not just early-minus-late decline.
Primary: RL, layer2, checkpoint20, up to20 persistent units, first subsequent task.
R=(D-C)-(B-A); G=A-C, with A deep/train, B deep/frozen, C lifted/train, D lifted/frozen. Positive is restoration/benefit for cost metrics.
CE area is divided by6000; accuracy contrast units are percentage points. Every timepoint is retained including the immediate lift shock.
Dose primitive Ck-Dk is reported literally as dose_train_minus_freeze: negative cost means permitting updates helps. It is not a matched depth-by-update interaction.

|Environment|Checkpoint|Layer|Pool sizes|R mean [95% CI]|G mean [95% CI]|Pilot label|
|---|---:|---:|---|---|---|---|
|PM|10|1|0,0,0|0.000000 [0.000000,0.000000]|0.000000 [0.000000,0.000000]|SECONDARY_NOT_CLASSIFIED|
|PM|10|2|0,0,0|0.000000 [0.000000,0.000000]|0.000000 [0.000000,0.000000]|SECONDARY_NOT_CLASSIFIED|
|PM|20|1|0,0,0|0.000000 [0.000000,0.000000]|0.000000 [0.000000,0.000000]|SECONDARY_NOT_CLASSIFIED|
|PM|20|2|1,0,0|0.000106 [-0.000350,0.000563]|-0.001108 [-0.005877,0.003661]|SECONDARY_NOT_CLASSIFIED|
|PM|50|1|0,0,0|0.000000 [0.000000,0.000000]|0.000000 [0.000000,0.000000]|SECONDARY_NOT_CLASSIFIED|
|PM|50|2|3,17,10|0.003054 [-0.006454,0.012563]|-0.102515 [-0.239457,0.034427]|SECONDARY_NOT_CLASSIFIED|
|RL|10|1|0,0,0|0.000000 [0.000000,0.000000]|0.000000 [0.000000,0.000000]|SECONDARY_NOT_CLASSIFIED|
|RL|10|2|6,13,5|0.011905 [-0.033694,0.057505]|-0.050142 [-0.081422,-0.018862]|SECONDARY_NOT_CLASSIFIED|
|RL|20|1|3,1,4|-0.011524 [-0.053566,0.030517]|0.027675 [-0.026044,0.081393]|SECONDARY_NOT_CLASSIFIED|
|RL|20|2|100,99,100|0.415130 [-0.185231,1.015490]|-0.121753 [-0.248668,0.005162]|INCONCLUSIVE|
|RL|50|1|11,21,16|-0.013706 [-0.026515,-0.000897]|0.013562 [0.009222,0.017902]|SECONDARY_NOT_CLASSIFIED|
|RL|50|2|100,100,100|0.588461 [0.264690,0.912232]|-0.138389 [-0.479977,0.203199]|SECONDARY_NOT_CLASSIFIED|

## Primary per-seed detail

|Seed|Pool|A CE area|B CE area|C CE area|D CE area|A end accuracy %|C end accuracy %|D end accuracy %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|100|2.305753|2.302273|2.375271|2.783242|11.083|21.500|20.583|
|1|99|2.291061|2.297606|2.415189|2.597046|13.667|19.250|20.000|
|2|100|2.303703|2.306006|2.475317|3.136246|11.583|19.500|16.500|

Limits: retrospective choice of environment/checkpoints from the previous result; prospective intervention comparisons. Finite5-task observation does not prove permanent non-recovery. Bias lifting changes initial predictions across depth conditions; paired train/frozen functions match. Incident-parameter freezing includes incoming rows, bias and outgoing columns, while other parameters still learn. Restored fixed features can help even without target updates, so lack of R is not evidence that capacity was never lost. A failed one-shot rescue is not proof of harmless sinking. W norms may diverge after intervention; this is not a W-independent mediated fraction. Random controls may contain other persistent units. PM ceiling and RL floor remain visible.

See per-job selections.json for exact IDs/deltas and provenance.json for all technical checks, initial shocks and hashes. Raw curves and per-input transport diagnostics are retained.
