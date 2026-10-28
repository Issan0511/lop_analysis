#!/bin/bash
# エージェント A の回し直しの列（Fable、依頼 4 の A 形成則、1028）。GPU の仕事は同時に 1 本。
# 走る前に GPU の空き 3 GB・RAM の空き 3 GB を確かめ、足りなければ 60 秒待つ（エージェント B と同じ GPU を使う）。
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
cd /home/issan/Projects/claude/wt/vit_long_1024 || exit 1
OUT=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0_form
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
LOG=$OUT/fable_queue.log

wait_free() {
  while :; do
    g=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}')
    m=$(free -m | awk '/Mem:/{print $7}')
    if [ "$g" -ge 3000 ] && [ "$m" -ge 3000 ]; then break; fi
    echo "$(date +%T) wait gpu_free=$g ram_avail=$m" >> "$LOG"
    sleep 60
  done
}

run() {
  tag="$1"; shift
  f="$OUT/fable_replay_${tag}.json"
  if [ -f "$f" ] && [ "$(python3 -c "import json;print(json.load(open('$f'))['steps'])")" = "5000" ]; then
    echo "$(date +%T) skip $tag" >> "$LOG"; return
  fi
  wait_free
  echo "$(date +%T) start $tag" >> "$LOG"
  nice -n 10 $PY -u analysis/vit_long_1024/posthoc_2blk/fable_r0form_replay.py "$@" > "$OUT/fable_replay_${tag}.log" 2>&1
  echo "$(date +%T) end $tag rc=$?" >> "$LOG"
}

run R0_t10_keep R0 10
run S_t10_keep S 10
run R0_t30_keep R0 30
run S_t30_keep S 30
run R0_t60_keep R0 60
run R0_t30_zero_m R0 30 --adam zero_m
run R0_t30_zero_mv R0 30 --adam zero_mv
run S_t60_keep S 60
run R0_t40_keep R0 40
run R0_t1_keep R0 1
run S_t1_keep S 1
echo "$(date +%T) ALL_DONE" >> "$LOG"
