#!/usr/bin/env bash
# Registered main run of cnn_center_budget_1020 (spec section 10).
#
#   cd ~/Projects/claude/wt/cnn_center_budget_1020
#   setsid nohup bash analysis/cnn_center_budget_1020/production.sh \
#       > results/cnn_center_budget_1020/logs/production.log 2>&1 < /dev/null &
#
# 1. Waits, without an upper bound, until all of the following hold (polled every 60 s):
#    a. the 5+1 MLP run is over: wt/mlp_center_x10_1020/results/mlp_center_x10_1020/exit_code.txt exists;
#       or its worktree is gone after having existed (cleaned up after the run); or its production.sh
#       process was seen alive and has then been absent on 3 consecutive polls without an exit code
#       (an abnormal end).  The MLP checks / smoke ending is not enough: they end before its main run.
#    b. no python / bash process of the character-GPT run chargpt_cl_1019;
#    c. no other python process on the GPU.
# 2. Runs budget 780, then 7800, then the verdict.
# 3. Writes results/cnn_center_budget_1020/exit_code.txt (0, or the first non-zero stage code) and
#    production_status.json (per-stage codes and times).
set -u
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
R=results/cnn_center_budget_1020
mkdir -p "$R/logs"
MLP_WT=/home/issan/Projects/claude/wt/mlp_center_x10_1020
MLP_EXIT="$MLP_WT/results/mlp_center_x10_1020/exit_code.txt"
MLP_PROD_PAT='^[^ ]*(bash|sh) [^ ]*mlp_center_x10_1020/production\.sh'
GPT_PAT='^[^ ]*(python[0-9.]*|bash|sh) [^ ]*chargpt_cl_1019'
ts() { date '+%F %T %z'; }
status_json() {  # stage code
  printf '{"stage": "%s", "code": %s, "time": "%s"}\n' "$1" "$2" "$(ts)" >> "$R/production_status.jsonl"
}

echo "[$(ts)] production start (pid $$, git $(git rev-parse --short HEAD))"
mlp_seen=0; mlp_wt_seen=0; mlp_absent=0; last_msg=""
while :; do
  mlp_done=0; why=""
  [ -d "$MLP_WT" ] && mlp_wt_seen=1
  if [ -f "$MLP_EXIT" ]; then
    mlp_done=1; why="MLP exit_code.txt = $(head -c 64 "$MLP_EXIT" | tr -d '\n')"
  elif [ "$mlp_wt_seen" = 1 ] && [ ! -d "$MLP_WT" ]; then
    mlp_done=1; why="MLP worktree removed after the run"
  else
    if pgrep -f "$MLP_PROD_PAT" > /dev/null; then
      mlp_seen=1; mlp_absent=0
    elif [ "$mlp_seen" = 1 ]; then
      mlp_absent=$((mlp_absent + 1))
      if [ "$mlp_absent" -ge 3 ]; then mlp_done=1; why="MLP production.sh gone on 3 polls without exit_code"; fi
    fi
  fi
  gpt_busy=0; pgrep -f "$GPT_PAT" > /dev/null && gpt_busy=1
  gpu_py=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -i python | tr '\n' ';')
  if [ "$mlp_done" = 1 ] && [ "$gpt_busy" = 0 ] && [ -z "$gpu_py" ]; then
    echo "[$(ts)] go: $why; chargpt idle; no python on the GPU"
    break
  fi
  msg="waiting: mlp_done=$mlp_done (seen=$mlp_seen absent=$mlp_absent) chargpt_busy=$gpt_busy gpu_python=[$gpu_py]"
  if [ "$msg" != "$last_msg" ]; then echo "[$(ts)] $msg"; last_msg="$msg"; fi
  sleep 60
done

code=0
for b in 780 7800; do
  echo "[$(ts)] budget $b start"
  "$runtime" src/cnn_center_budget_1020.py main --budget "$b" --out "$R/b$b"
  c=$?
  status_json "b$b" "$c"
  echo "[$(ts)] budget $b exit $c"
  if [ "$c" != 0 ] && [ "$code" = 0 ]; then code=$c; fi
done
"$runtime" analysis/cnn_center_budget_1020/verdict.py --root "$R"
c=$?
status_json verdict "$c"
echo "[$(ts)] verdict exit $c"
if [ "$c" != 0 ] && [ "$code" = 0 ]; then code=$c; fi
echo "$code" > "$R/exit_code.txt"
echo "[$(ts)] production end, exit_code $code"
exit "$code"
