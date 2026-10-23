# Retrospective identity tracking (post-hoc)

Descriptive reanalysis of existing task-endpoint arrays; not a new experiment, causal test, or hypothesis-significance analysis. Recurrent entries are episodes, not independent replicas.

Sink is q >= .95, where q is each unit's fraction of 1,200 task inputs with phi'(z) < .05. Recovery follows the same unit ID through endpoints t+1..t+5. Entries after task 45 are right-censored. Leave-and-return counts as recovery; task-end data cannot exclude within-task transients.

## ELU ref persistent stock (sink at each of latest five endpoints)

Each cell gives the three seed counts in seed order 0, 1, 2.

| env | layer | task 10 | task 20 | task 50 |
|---|---:|---:|---:|---:|
| PM | 1 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| PM | 2 | 0, 0, 0 | 1, 0, 0 | 3, 17, 10 |
| RL | 1 | 0, 0, 0 | 3, 1, 4 | 11, 21, 16 |
| RL | 2 | 6, 13, 5 | 100, 99, 100 | 100, 100, 100 |

## ELU eligible entry episodes by entry-task window (three seeds combined descriptively)

| env | iv | layer | entry tasks | entries | exit q<.95 within 5 | stronger q<.5 within 5 |
|---|---|---:|---:|---:|---:|---:|
| PM | ref | 1 | 1-10 | 50 | 50 (100.0%) | 47 (94.0%) |
| PM | ref | 1 | 11-20 | 118 | 118 (100.0%) | 112 (94.9%) |
| PM | ref | 1 | 21-30 | 171 | 170 (99.4%) | 142 (83.0%) |
| PM | ref | 1 | 31-40 | 216 | 216 (100.0%) | 172 (79.6%) |
| PM | ref | 1 | 41-45 | 105 | 104 (99.0%) | 72 (68.6%) |
| PM | ref | 2 | 1-10 | 58 | 58 (100.0%) | 43 (74.1%) |
| PM | ref | 2 | 11-20 | 240 | 235 (97.9%) | 109 (45.4%) |
| PM | ref | 2 | 21-30 | 374 | 363 (97.1%) | 88 (23.5%) |
| PM | ref | 2 | 31-40 | 419 | 391 (93.3%) | 59 (14.1%) |
| PM | ref | 2 | 41-45 | 229 | 217 (94.8%) | 14 (6.1%) |
| PM | wclamp | 1 | 1-10 | 50 | 50 (100.0%) | 47 (94.0%) |
| PM | wclamp | 1 | 11-20 | 111 | 106 (95.5%) | 103 (92.8%) |
| PM | wclamp | 1 | 21-30 | 124 | 121 (97.6%) | 114 (91.9%) |
| PM | wclamp | 1 | 31-40 | 173 | 169 (97.7%) | 159 (91.9%) |
| PM | wclamp | 1 | 41-45 | 93 | 92 (98.9%) | 88 (94.6%) |
| PM | wclamp | 2 | 1-10 | 58 | 58 (100.0%) | 43 (74.1%) |
| PM | wclamp | 2 | 11-20 | 176 | 173 (98.3%) | 96 (54.5%) |
| PM | wclamp | 2 | 21-30 | 298 | 294 (98.7%) | 152 (51.0%) |
| PM | wclamp | 2 | 31-40 | 296 | 289 (97.6%) | 133 (44.9%) |
| PM | wclamp | 2 | 41-45 | 148 | 144 (97.3%) | 64 (43.2%) |
| RL | ref | 1 | 1-10 | 28 | 28 (100.0%) | 10 (35.7%) |
| RL | ref | 1 | 11-20 | 192 | 169 (88.0%) | 15 (7.8%) |
| RL | ref | 1 | 21-30 | 176 | 131 (74.4%) | 7 (4.0%) |
| RL | ref | 1 | 31-40 | 121 | 85 (70.2%) | 1 (0.8%) |
| RL | ref | 1 | 41-45 | 44 | 21 (47.7%) | 0 (0.0%) |
| RL | ref | 2 | 1-10 | 341 | 69 (20.2%) | 0 (0.0%) |
| RL | ref | 2 | 11-20 | 33 | 1 (3.0%) | 0 (0.0%) |
| RL | ref | 2 | 21-30 | 3 | 0 (0.0%) | 0 (0.0%) |
| RL | ref | 2 | 31-40 | 3 | 0 (0.0%) | 0 (0.0%) |
| RL | ref | 2 | 41-45 | 0 | NA | NA |
| RL | wclamp | 1 | 1-10 | 28 | 28 (100.0%) | 11 (39.3%) |
| RL | wclamp | 1 | 11-20 | 179 | 172 (96.1%) | 16 (8.9%) |
| RL | wclamp | 1 | 21-30 | 229 | 202 (88.2%) | 15 (6.6%) |
| RL | wclamp | 1 | 31-40 | 167 | 147 (88.0%) | 10 (6.0%) |
| RL | wclamp | 1 | 41-45 | 84 | 73 (86.9%) | 3 (3.6%) |
| RL | wclamp | 2 | 1-10 | 341 | 73 (21.4%) | 0 (0.0%) |
| RL | wclamp | 2 | 11-20 | 43 | 1 (2.3%) | 0 (0.0%) |
| RL | wclamp | 2 | 21-30 | 2 | 0 (0.0%) | 0 (0.0%) |
| RL | wclamp | 2 | 31-40 | 2 | 0 (0.0%) | 0 (0.0%) |
| RL | wclamp | 2 | 41-45 | 1 | 0 (0.0%) | 0 (0.0%) |

Leaky .1 low-gate values are structurally zero at this threshold; this does not imply leaky has no LoP. Cross-activation functional conclusions require actual learning outcomes. Ever-sunk cumulative count is not evidence of accumulation. Any stock/performance association is descriptive.
