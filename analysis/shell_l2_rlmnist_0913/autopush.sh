#!/usr/bin/env bash
# shell_l2_rlmnist_0913 -- spec section 6: every 30 min, commit and push the raw outputs of FINISHED shards
# (provenance.json present) with explicit paths.  No aggregation, nothing is read.  Exits after the push
# that contains all 80 shards.  The VM is deleted around 2026-09-14 19:11 JST: unpushed files are lost.
#   nohup setsid bash analysis/shell_l2_rlmnist_0913/autopush.sh > results/shell_l2_rlmnist_0913/logs/autopush.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/shell_l2_rlmnist_0913
BRANCH=exp/shell_l2_rlmnist_0913
INTERVAL=${INTERVAL:-1800}

while true; do
  n=0
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    for act in R SNA; do
      for reg in none l2 l2init shell; do
        name="${act}_${reg}_s${seed}"
        d="$OUT/runs/$name"
        if [ -f "$d/provenance.json" ]; then
          n=$((n + 1))
          git add -- "$d/per_task.csv" "$d/layer_metrics.csv" "$d/provenance.json"
          [ -f "$OUT/logs/$name.log" ] && git add -- "$OUT/logs/$name.log"
        fi
      done
    done
  done
  if ! git diff --cached --quiet; then
    git commit -q -F - <<EOF
[L2Init中間実験_0913] 本走の途中退避: 完走 ${n}/80 run の生出力（未集計・判定前）

analysis/shell_l2_rlmnist_0913/autopush.sh が完走した shard の per_task.csv・layer_metrics.csv・provenance.json・ログだけを
パス明示で commit した。集計はしていない（spec §6）。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SuEJDXnrm7iScm8PjWD68s
EOF
    for i in 1 2 3; do git push -q origin "$BRANCH" && break; sleep 30; done
    echo "$(date -Is) pushed ${n}/80"
  else
    echo "$(date -Is) nothing new (${n}/80)"
  fi
  [ "$n" -ge 80 ] && { echo "$(date -Is) all 80 shards pushed; exiting"; exit 0; }
  sleep "$INTERVAL"
done
