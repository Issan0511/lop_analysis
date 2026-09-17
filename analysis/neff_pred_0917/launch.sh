#!/usr/bin/env bash
# neff_pred_0917 -- spec section 7: one GPU process (both engines, 150 tasks) and the CPU box's seeds 10-12
# (51 tasks each, one thread each).
# Refuses to start unless checks.json says all_pass, PREREG_COMMIT is set in the runner, the code, the
# calibration and the spec are committed and HEAD is pushed.  The CPU seeds start 25 s apart and only while
# MemAvailable stays above 6 GiB plus 1.5x the ee CLI's peak RSS (shared desktop).  A shard with
# provenance.json is skipped (relaunch-safe).  Log names are flat.
#   setsid nohup bash analysis/neff_pred_0917/launch.sh > results/neff_pred_0917/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/neff_pred_0917
MOD=src/neff_pred_0917.py
PY=/usr/bin/python3
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
grep -q '^PREREG_COMMIT = "[0-9a-f]\{40\}"' "$MOD" || { echo "ABORT: PREREG_COMMIT not set in $MOD"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/neff_pred_0917 specs/spec_neff_pred_0917.md $OUT/calibration.json)" ] \
  || { echo "ABORT: uncommitted code, calibration or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
mkdir -p "$OUT/gpu" "$OUT/ee" "$OUT/logs"

avail_gib() { awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
# the ee CLI process's own peak RSS (checks S6 wrote it; the S-cost figure is the checks process, CUDA included)
NEED=$($PY -c "import json; p=json.load(open('results/_checks_neff_pred_0917/cli_base/ee/provenance.json')); print(round(6.0 + 1.5*max(p['peak_rss_kb']/2**20, 0.5), 1))")

echo "$(date -Is) launch: git $(git rev-parse HEAD)  avail $(avail_gib) GiB  need per CPU seed $NEED GiB"
if [ -f "$OUT/gpu/provenance.json" ]; then
  echo "$(date -Is) skip gpu (done)"
else
  echo "$(date -Is) start gpu"
  ( nice -n 5 "$PY" -m src.neff_pred_0917 gpu --out "$OUT/gpu" > "$OUT/logs/gpu.log" 2>&1 \
      && echo "$(date -Is) done  gpu" || echo "$(date -Is) FAIL  gpu (exit $?)" ) &
  sleep 60
fi
for seed in 10 11 12; do
  if [ -f "$OUT/ee/s$seed/provenance.json" ]; then echo "$(date -Is) skip ee s$seed (done)"; continue; fi
  while [ "$($PY -c "print(int($(avail_gib) >= $NEED))")" != 1 ]; do
    echo "$(date -Is) wait ee s$seed: avail $(avail_gib) GiB < $NEED"; sleep 60
  done
  echo "$(date -Is) start ee s$seed  avail $(avail_gib) GiB"
  ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -m src.neff_pred_0917 ee --seed "$seed" --threads 1 \
      --out "$OUT/ee/s$seed" > "$OUT/logs/ee_s$seed.log" 2>&1 \
      && echo "$(date -Is) done  ee s$seed" || echo "$(date -Is) FAIL  ee s$seed (exit $?)" ) &
  sleep 25
done
wait
echo "$(date -Is) all jobs returned"
