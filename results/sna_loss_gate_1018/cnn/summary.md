# Loss-gated Snake CNN production — completed

Registered protocol: 3 arms × seed 10–19 × 30 tasks × 400 epochs.
Execution completed with exit code 0. All 900 task records present; task-1 accuracies match across arms.

## Registered results (tasks 21–30 online accuracy; seed means averaged)

| Arm | Accuracy |
| --- | ---: |
| SNA | 95.7584% |
| Task-1 frozen | 96.9430% |
| Loss-gated | 96.9378% |

Primary: **BETTER_THAN_SNA**, paired mean +1.1795 percentage points, 95% bootstrap interval [+0.8483, +1.5516], positive 10/10 seeds.
Freeze comparison: **NONINFERIOR_TO_FREEZE** at the preregistered 0.5-point margin; paired mean −0.0052 points, interval [−0.2259, +0.2521], positive 3/10 seeds. This does not establish superiority to freeze.
Switch CE >100: 0 for every arm. Full registered outputs: `verdict.json`, `per_task.csv`.

## Descriptive readout (posthoc; not a causal mechanism verdict)

Across tasks 2–30, the loss-gated mean g is 0.00004811 in the first 10 epochs and 0.99714 in the final 10. The controller therefore slows adaptation during relearning and resumes near-normal adaptation late; it is not a permanent freeze.
Every seed's reference loss hits the epsilon floor 1e-4. In this box the rule effectively uses that fixed loss threshold; the benefit of calibrating a task-1 loss reference has not been established.
Task-30 f2 spread, averaged across 10 seeds: SNA 16.64, frozen 6.57, loss-gated 33.32. Loss-gated c1 median alpha averaged over seeds is 1.545, so c1 does not stay at the upper clip. Better online accuracy does not establish scale stability or long-horizon safety.
Values and source hashes: `mechanism_posthoc.json`. Raw epoch records and checkpoints are archived via `../backup_manifest.json`.
