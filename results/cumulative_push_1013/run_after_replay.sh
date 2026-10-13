#!/bin/bash
# wait for the three replay arms to finish, then run the first-step expectation and the within-task ledger for each arm
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
WT=/home/issan/Projects/claude/wt/cumulative_push_1013
R=/home/issan/Projects/obsidian-research-data/cumulative_push_1013/replay
cd $WT
until grep -q DONE $R/a0.1/replay.log && grep -q DONE $R/a0.3/replay.log && grep -q DONE $R/a0.5/replay.log; do sleep 20; done
echo "replay done $(date)"
for t in a01 a03 a05; do
  OMP_NUM_THREADS=1 $PY analysis/cumulative_push_1013/adam_first_step.py $t > results/cumulative_push_1013/adam_first_step_$t.log 2>&1 &
done
wait
echo "adam_first_step done $(date)"
for t in a01 a03 a05; do
  OMP_NUM_THREADS=1 $PY analysis/cumulative_push_1013/ledger_replay.py $t > results/cumulative_push_1013/ledger_replay_$t.log 2>&1 &
done
wait
echo "ledger done $(date)"
