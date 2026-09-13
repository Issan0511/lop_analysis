#!/bin/bash
# transport_holes_0910: sub-run A first, then C and D.
#
# Stage 2 is gated PER ARM, not per stage.  Only the valley arms of C and D anchor
# on A; sub-run D's R arm anchors on the committed leak_ladder_force_posthoc_0910
# and its ELU1 arm on the committed elu_growth_0909, so neither has any reason to
# be cancelled by a GELU failure.  The first version aborted stage 2 wholesale and
# would have thrown away all ten of those runs.
#
# Exit status is what says a run succeeded: clamp_horizon's TIME_CAP assert fires
# AFTER the outputs are written, so a complete-looking directory is not evidence.
set -u
cd /home/i_nakatsuka/Projects/lop_analysis/.claude/worktrees/collective_kick_0908
PY=python3
LOG=results/_logs_transport_holes_0910
mkdir -p "$LOG"
FAIL=0
STAMP=$(date +%H%M%S)

wait_all() {   # wait_all name:pid ...
  local e pid name
  for e in "$@"; do
    pid="${e%%:*}"; name="${e##*:}"
    if wait "$pid"; then echo "OK   $name"; else echo "FAIL $name"; FAIL=1; fi
  done
}

echo "=== stage 1: sub-run A (valley reference trajectories) ==="
PIDS=()
for arm in GELU SILU; do
  for s in 0 1 2; do
    extra=""
    [ "$s" != "0" ] && extra="--no-controls"
    $PY -m src.long_horizon_acts_0910 --arm "$arm" --seed "$s" $extra \
        > "$LOG/A_${arm}_s${s}_$STAMP.log" 2>&1 &
    PIDS+=("$!:A_${arm}_s${s}")
  done
done
wait_all "${PIDS[@]}"

echo "=== stage 2: sub-run C (per arm/seed, only where A produced an anchor) + sub-run D ==="
PIDS=()
for arm in GELU SILU; do
  for s in 0 1 2; do
    if [ -f "results/long_horizon_acts_0910/${arm}_none_s${s}_units.npz" ]; then
      $PY -m src.clamp_horizon_acts_0910 --arm "$arm" --seed "$s" \
          > "$LOG/C_${arm}_s${s}_$STAMP.log" 2>&1 &
      PIDS+=("$!:C_${arm}_s${s}")
    else
      echo "SKIP C_${arm}_s${s} (no anchor from sub-run A)"
    fi
  done
done
# Sub-run B is re-run here under the same module version as A/C/D.  Its first
# execution finished correctly (G1 0.0 over 600 arrays) under commit 742fd41, but
# that predates the 追補 2 fixes, and B takes about as long as C so re-running it
# costs no wall clock.
for s in 0 1 2; do
  $PY -m src.clamp_horizon_acts_0910 --arm R --seed "$s" \
      > "$LOG/B_R_s${s}_$STAMP.log" 2>&1 &
  PIDS+=("$!:B_R_s${s}")
done
# D's R and ELU1 arms have committed anchors and never depend on stage 1.
for arm in R ELU1; do
  $PY -m src.why_down_acts_0910 --arm "$arm" > "$LOG/D_${arm}_$STAMP.log" 2>&1 &
  PIDS+=("$!:D_${arm}")
done
for arm in GELU SILU; do
  if [ -f "results/long_horizon_acts_0910/${arm}_none_s0_units.npz" ]; then
    $PY -m src.why_down_acts_0910 --arm "$arm" > "$LOG/D_${arm}_$STAMP.log" 2>&1 &
    PIDS+=("$!:D_${arm}")
  else
    echo "SKIP D_${arm} (no anchor from sub-run A)"
  fi
done
wait_all "${PIDS[@]}"

echo "=== done (FAIL=$FAIL) ==="
exit "$FAIL"
