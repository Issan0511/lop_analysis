#!/usr/bin/env bash
# chargpt_cl_1019 を別マシンで回す（spec 追補 2、手順は specs/chargpt_cl_1019_remote_runbook.md）。
#   PY=<python> bash src/chargpt_cl_1019_remote.sh prepare          環境の記録・データ列の照合・検査（箱 M）・速さ
#   PY=<python> bash src/chargpt_cl_1019_remote.sh run 5 6 7 8 9    担当 seed の本走を切り離して起動
#   PY=<python> bash src/chargpt_cl_1019_remote.sh status           進み具合
#   bash src/chargpt_cl_1019_remote.sh push 5 6                     完走した seed の記録を commit して push
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
host=$(hostname)
R=results/chargpt_cl_1019
D=$R/remote_$host
mkdir -p "$D"
cmd=${1:-}; shift || true
case "$cmd" in
  prepare)
    {
      echo "host $host  $(date '+%F %T')"
      git rev-parse HEAD
      $PY -c "import sys,torch,numpy;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'numpy',numpy.__version__);print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      free -h | head -2; nproc
    } 2>&1 | tee "$D/env.txt"
    echo "== S-portable（データ列が基準機と同じか）"
    $PY src/chargpt_cl_1019_streams.py verify --expected $R/expected_streams.json --out "$D/streams_verify.json" || { echo "STOP: データ列が基準機と違う"; exit 1; }
    echo "== 検査（箱 M）"
    $PY src/chargpt_cl_1019_checks.py --boxes M --out "$D/checks" > "$D/checks_run.log" 2>&1
    grep -E "^\[(PASS|FAIL)\]" "$D/checks_run.log" | cut -c1-80
    grep -q "^\[FAIL\]" "$D/checks_run.log" && { echo "STOP: 検査が落ちた"; exit 1; }
    echo "== 速さ（C3・2 課題 × 3 通り）"
    $PY src/chargpt_cl_1019_cost.py --boxes M --nm 10000 --alpha 0.01 --nw 0 --arms C3 --out "$D/cost" 2>&1 | grep -v -i warn | tail -12
    echo "PREPARE_OK"
    ;;
  run)
    [ $# -ge 1 ] || { echo "seed を渡す"; exit 1; }
    PY=$PY setsid nohup bash src/chargpt_cl_1019_production.sh "$@" > "$R/production_launch_$host.log" 2>&1 < /dev/null &
    sleep 5; cat "$R/production_launch_$host.log"
    ;;
  status)
    tail -5 "$R/production_launch_$host.log" 2>/dev/null
    for f in $R/M/logs/*.log; do [ -f "$f" ] && echo "$(basename $f): $(grep -c 'task' $f) 行 $(tail -1 $f | cut -c1-70)"; done
    ;;
  push)
    [ $# -ge 1 ] || { echo "seed を渡す"; exit 1; }
    for s in "$@"; do
      for arm in C3 C4 B3 N4; do
        d=$R/M/$arm/seed$s
        [ -f "$d/provenance.json" ] || { echo "未完走: $d（push しない）"; continue; }
        git add "$d" "$R/M/logs/${arm}_seed${s}.log"
      done
    done
    git add "$D" "$R/production_launch_$host.log" 2>/dev/null
    git commit -m "results: chargpt_cl_1019 box M seeds $* on $host" || true
    for i in 1 2 3; do
      git pull --no-rebase --no-edit origin claude/chargpt_cl_1019 && git push origin HEAD:claude/chargpt_cl_1019 && break
      sleep 5
    done
    ;;
  *)
    sed -n 2,7p "$0"; exit 1;;
esac
