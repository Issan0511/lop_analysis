#!/bin/bash
# エージェント A の回し直しの列 3（Fable、依頼 4、1028）。列 2 の ALL_DONE2 を待ってから始める。
# 崩れの間の「共通の押し」Σ Wᵀ(p−y)（画像に依らない head 特徴への勾配）と予測の偏りを 50 更新ごとに記録する取り直し。
# 列 1 の R0 t30・S t30・R0 t60 の keep 走には無い記録なので、同じ条件を _pbar の接尾辞で回す（軌跡は同じになるはず）。
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
cd /home/issan/Projects/claude/wt/vit_long_1024 || exit 1
OUT=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0_form
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
LOG=$OUT/fable_queue3.log

until grep -q ALL_DONE2 "$OUT/fable_queue2.log" 2>/dev/null; do sleep 30; done
echo "$(date +%T) queue2 done, queue3 start" >> "$LOG"

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

run R0_t30_keep_pbar R0 30 --suffix _pbar
run S_t30_keep_pbar S 30 --suffix _pbar
run R0_t60_keep_pbar R0 60 --suffix _pbar
echo "$(date +%T) ALL_DONE3" >> "$LOG"
