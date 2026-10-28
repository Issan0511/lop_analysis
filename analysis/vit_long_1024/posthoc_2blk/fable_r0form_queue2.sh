#!/bin/bash
# エージェント A の回し直しの列 2（Fable、依頼 4、1028）。列 1 の ALL_DONE を待ってから始める。
# 切替相の体の過渡を除く反実仮想（最初の 200 更新は体の lr=0、head だけ動く）: R0 t30・S t30。
# 1028 23:15 に絞った（当初は t10・t60 の反実仮想と R0 t10・S t10 の取り直しも入れていたが、切替相は成長の主因でないと分かったので外した）。
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
cd /home/issan/Projects/claude/wt/vit_long_1024 || exit 1
OUT=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0_form
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
LOG=$OUT/fable_queue2.log

until grep -q ALL_DONE "$OUT/fable_queue.log" 2>/dev/null; do sleep 30; done
echo "$(date +%T) queue1 done, queue2 start" >> "$LOG"

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

run R0_t30_keep_frz200 R0 30 --freeze_body_until 200
run S_t30_keep_frz200 S 30 --freeze_body_until 200
echo "$(date +%T) ALL_DONE2" >> "$LOG"
