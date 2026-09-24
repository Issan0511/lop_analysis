#!/usr/bin/env bash
# hsweep_rlmnist_0924: 6 widths x 5 seeds = 30 series on CPU, one process per (h, seed), P parallel.
# Longest arms (h=300) start first. Logs go to results/hsweep_rlmnist_0924/_run_logs/ (no '/' in log names).
set -euo pipefail
cd "$(dirname "$0")/../.."
P="${P:-8}"
HASH="$(git rev-parse HEAD)"
OUT=results/hsweep_rlmnist_0924
mkdir -p "$OUT/_run_logs"
echo "$HASH" > "$OUT/launch_git_hash.txt"
date -Is > "$OUT/launch_started.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=()
for h in 300 100 64 32 16 8; do
  for s in 200 201 202 203 204; do
    jobs+=("$h $s")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$P" -L 1 bash -c '
  h=$0; s=$1; tag=$(printf "h%03d_s%d" "$h" "$s")
  if [ -e STOP ]; then echo "STOP present, skipping $tag"; exit 0; fi
  python3 src/hsweep_rlmnist_0924.py --hidden "$h" --seeds "$s" --threads 1 --device cpu \
    --source-git-hash "'"$HASH"'" --out "'"$OUT"'/$tag" > "'"$OUT"'/_run_logs/$tag.log" 2>&1 \
    && echo "done $tag $(date -Is)" || echo "FAILED $tag $(date -Is)"
' | tee -a "$OUT/_run_logs/launch_progress.txt"
date -Is > "$OUT/launch_finished.txt"
