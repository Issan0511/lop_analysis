#!/usr/bin/env bash
# Second launcher (deviation, 12:00): the first launcher's 8 slots are held by the five h=300 series
# (~2 min/task) so the small-h arms would wait behind them.  A STOP file makes launch.sh skip its
# remaining queue; this script runs every not-yet-started (h, seed) with its own parallelism.
set -euo pipefail
cd "$(dirname "$0")/../.."
P="${P:-6}"
HASH="$(cat results/hsweep_rlmnist_0924/launch_git_hash.txt)"
[ "$HASH" = "$(git rev-parse HEAD)" ] || { echo "HEAD moved since launch"; exit 1; }
OUT=results/hsweep_rlmnist_0924
date -Is > "$OUT/launch2_started.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=()
for h in 100 64 32 16 8; do
  for s in 200 201 202 203 204; do
    tag=$(printf "h%03d_s%d" "$h" "$s")
    [ -e "$OUT/$tag" ] || jobs+=("$h $s")
  done
done
echo "remaining: ${#jobs[@]}"
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -L 1 bash -c '
  h=$0; s=$1; tag=$(printf "h%03d_s%d" "$h" "$s")
  python3 src/hsweep_rlmnist_0924.py --hidden "$h" --seeds "$s" --threads 1 --device cpu \
    --source-git-hash "'"$HASH"'" --out "'"$OUT"'/$tag" > "'"$OUT"'/_run_logs/$tag.log" 2>&1 \
    && echo "done $tag $(date -Is)" || echo "FAILED $tag $(date -Is)"
' | tee -a "$OUT/_run_logs/launch_progress.txt"
date -Is > "$OUT/launch2_finished.txt"
