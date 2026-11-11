#!/usr/bin/env bash
# bwdswap_1111 main run (brief item 5).  Start it only after the predictions are registered:
#   setsid nohup bash analysis/bwdswap_1111/launch.sh <prediction-registration-commit> \
#       > results/bwdswap_1111/logs/launch.log 2>&1 &
# Gates (the runner re-checks them in each process): results/bwdswap_1111/checks.json is all_pass for the
# current sources; src/bwdswap_1111.py and analysis/bwdswap_1111 are committed; the registration commit is an
# ancestor of HEAD.
# GPU (one card): RC, then F51 -- one process at a time, each taking /tmp/lop_analysis_gpu.lock.  N, X and B of a
# setting run inside one process, so inside one boot; a rerun after a reboot refuses the old output directory
# (boot_id is part of the identity): move it aside and start that setting again.
# CPU: RM seeds 0-9, one process per seed, 1 thread, nice 10.  At most MAX_RM at once (default 10), fitted to
# MemAvailable at launch: RESERVE_GIB (default 6) for the desktop, plus the GPU process's measured peak RSS, plus
# 1.2 x the measured private memory (USS) of an RM process per slot.  Each job also waits for that room before
# it starts.  A seed (or GPU setting) already complete with the same identity is skipped by the runner.
set -euo pipefail
cd "$(dirname "$0")/../.."
PREREG=${1:?usage: launch.sh <prediction registration commit>}
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=results/bwdswap_1111
MAX_RM=${MAX_RM:-10}
RESERVE_GIB=${RESERVE_GIB:-6}
mkdir -p "$OUT/logs" "$OUT/RM"
export BWDSWAP_PREREG="$PREREG"

"$PY" - <<'EOF' || { echo "ABORT: checks.json missing, failed or stale"; exit 1; }
import json, sys
sys.path.insert(0, '.')
from src import bwdswap_1111 as bw
c = json.loads(bw.CHECKS.read_text())
assert c['all_pass'] is True, 'checks.json is not all_pass'
assert c['source_sha256'] == bw.source_hashes(), 'checks.json was made from other sources'
EOF
[ -z "$(git status --porcelain -- src/bwdswap_1111.py analysis/bwdswap_1111)" ] \
  || { echo "ABORT: uncommitted implementation"; exit 1; }
git merge-base --is-ancestor "$PREREG" HEAD || { echo "ABORT: $PREREG is not an ancestor of HEAD"; exit 1; }
[ "$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 1)" = 0 ] || echo "WARN: HEAD is not on the upstream yet"

read -r N NEED_GIB < <("$PY" - "$MAX_RM" "$RESERVE_GIB" <<'EOF'
import json, math, sys
c = json.load(open('results/bwdswap_1111/checks.json'))
mem = c['memory']
uss = max(mem[p]['peak_uss_gib'] for p in ('e1-RM_s0', 'e1-RM_s1', 'RM_s100', 'RM_s101') if mem.get(p))
gpu = max(mem[p]['peak_rss_gib'] for p in ('RC', 'F51', 'e1-RC', 'e1-F51') if mem.get(p))
avail = [int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:')][0] / 2 ** 20
reserve = float(sys.argv[2])
fit = int((avail - reserve - gpu) // (1.2 * uss))
print(max(0, min(int(sys.argv[1]), fit)), math.ceil(reserve + 1.2 * uss))
EOF
)
[ "$N" -ge 1 ] || { echo "ABORT: no RM slot fits in MemAvailable now"; exit 1; }

run_rm() {
  local seed=$1 out="$OUT/RM/s$1" avail
  sleep $(( (seed % N_SLOTS) * 15 ))
  while :; do
    avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$avail" -ge "$NEED_GIB" ] && break
    echo "$(date -Is) RM s$seed waits: MemAvailable ${avail} GiB < ${NEED_GIB} GiB"
    sleep 30
  done
  echo "$(date -Is) start RM s$seed (MemAvailable ${avail} GiB)"
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -u -m src.bwdswap_1111 --setting RM --seed "$seed" \
      --out "$out" > "$OUT/logs/RM_s$seed.log" 2>&1; then
    echo "$(date -Is) done RM s$seed"
  else
    echo "$(date -Is) FAIL RM s$seed (see $OUT/logs/RM_s$seed.log)"
  fi
}
export -f run_rm
export OUT PY NEED_GIB
export N_SLOTS="$N"

echo "$(date -Is) launch: git $(git rev-parse HEAD)  prereg $(git rev-parse "$PREREG")  boot $(cat /proc/sys/kernel/random/boot_id)"
echo "$(date -Is) RM slots $N (each needs ${NEED_GIB} GiB available)  $(free -g | awk '/^Mem:/{print "MemAvailable "$7" GiB"}')"
(
  for setting in RC F51; do
    if "$PY" -u -m src.bwdswap_1111 --setting "$setting" --out "$OUT/$setting" > "$OUT/logs/$setting.log" 2>&1; then
      echo "$(date -Is) done $setting"
    else
      echo "$(date -Is) FAIL $setting (see $OUT/logs/$setting.log)"
    fi
  done
) &
GPU_CHAIN=$!
printf '%s\n' 0 1 2 3 4 5 6 7 8 9 | xargs -P "$N" -I{} bash -c 'run_rm "$1"' _ {}
wait "$GPU_CHAIN"
echo "$(date -Is) all jobs returned"
"$PY" analysis/bwdswap_1111/judge.py --src "$OUT" > "$OUT/logs/judge.log" 2>&1 \
  && echo "$(date -Is) judge: complete ($OUT/summary.md)" || echo "$(date -Is) judge: incomplete (see $OUT/logs/judge.log)"
