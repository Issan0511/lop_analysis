#!/usr/bin/env bash
# J6 の解析（spec_chargpt_detlabel_1023 §6.5）を 1 腕について回す。CPU float64・8 threads・学習なし。
# 使い方: analysis/chargpt_detlabel_1023/j6/run_j6.sh <A4|A3|S2> [--seeds 0 1 ...]
# 再開: 各段の gate は gate の json が無いときだけ回す。run は seed ごとの part（csv・npz・json）が揃った seed を飛ばす。
#       コードを変えたら、その腕の出力（results/chargpt_detlabel_1023/j6/<腕>/）を消してから回し直す（code_sha256 の照合で止まる）。
# 前提: その腕の 10 seed の走が済み、<run_dir>/seed<k>/ckpts/task<t>.pt（課題末 1〜10・25・39〜50・89〜100）と pi.npz がある。
#       pi.npz が無い課題は ChainM で π を作る（α ≤ 1e−3 では 1 連鎖およそ 76〜194 秒、3 段がそれぞれ作り直す）。
set -euo pipefail
ARM=${1:?arm (A4|A3|S2)}; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
DL=$(cd "$HERE/../../.." && pwd)
PY=/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019/.venv/bin/python
export J6_ARM=$ARM OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
OUT=${J6_OUT:-$DL/results/chargpt_detlabel_1023/j6/$ARM}
mkdir -p "$OUT"
cd "$HERE"
log(){ echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }
for stage in measure components loading_sources; do
  case $stage in measure) G=gate.json;; components) G=components_gate.json;; loading_sources) G=loading_sources_gate.json;; esac
  if [ ! -f "$OUT/$G" ]; then log "$stage gate"; $PY $stage.py gate 2>&1 | tee -a "$OUT/run.log"; fi
  log "$stage run $*"; $PY $stage.py run "$@" 2>&1 | tee -a "$OUT/run.log"
done
if [ $# -eq 0 ]; then
  log report; $PY report.py 2>&1 | tee -a "$OUT/run.log"; $PY report_components.py 2>&1 | tee -a "$OUT/run.log"
fi
log done
