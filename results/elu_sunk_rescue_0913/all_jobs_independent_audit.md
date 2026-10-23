# Independent audit of all rescue jobs

Status: **PASS**. All 12 jobs passed the bounded raw-artifact checks.

The independently recomputed primary RL checkpoint-20 layer-2 values are R20 = 0.41145224, 0.17531153, 0.65862484 and G20 = -0.06951719, -0.12412819, -0.17161478. The registered verdict is **INCONCLUSIVE**, matching `verdict.csv`.

Selections, nested doses, control order, continuation hashes, saved pair observables, all selected incident-update arrays, counters, finite arrays, and first-task AUC contrasts were recomputed from saved artifacts. `contrasts.csv` matches exactly. Provenance source hashes also match the current files.

RL boundary continuity was also checked directly: across all six RL jobs, seeds, branches, four boundaries, and saved q/z/gate/weight-norm arrays, task-end step 6000 equals the next task step 0 exactly (overall maxabs 0). Thus partial opening at task 21 persists at task 22 step 0. The plotted steep rise is real movement during task 22 updates 1–75, visually compressed on the 0–30000-update axis; the endpoint and label are correct. PM is excluded from this equality because its input permutation changes at each task.

Runtime assertions were inspected separately: lift confinement, exact full-logit/parameter/moment pair identity, and frozen-reference invariants all report zero error, but the complete intermediate tensors needed to reconstruct those assertions independently were not persisted. Saved per-unit incident updates provide independent evidence that every selected incoming row, bias, and outgoing column remained fixed at every measurement interval.

Jobs with any pool below five:

- PM_t10_l1: 0,0,0
- PM_t10_l2: 0,0,0
- PM_t20_l1: 0,0,0
- PM_t20_l2: 1,0,0
- PM_t50_l1: 0,0,0
- PM_t50_l2: 3,17,10
- RL_t10_l1: 0,0,0
- RL_t20_l1: 3,1,4
