#!/usr/bin/env bash
# vit_long_1024 追補 E の byakko の分を、AI なしで回すための 1 本（Claude、1026）。
# 手順書 specs/vit_long_1024_rlfix_e_runbook.md の §1〜§5 を 4 つの命令に畳んである。
#
#   bash analysis/vit_long_1024/remote/byakko.sh setup  <lop_analysis の clone> <出力の置き場> <データの置き場>
#   bash analysis/vit_long_1024/remote/byakko.sh start  <lop_analysis の clone> <出力の置き場> <データの置き場>
#   bash analysis/vit_long_1024/remote/byakko.sh status <lop_analysis の clone> <出力の置き場> <データの置き場>
#   bash analysis/vit_long_1024/remote/byakko.sh stop   <lop_analysis の clone> <出力の置き場> <データの置き場>
#   bash analysis/vit_long_1024/remote/byakko.sh send   <lop_analysis の clone> <出力の置き場> <データの置き場>
#
# setup  = 版の確認・Tiny ImageNet の取得と照合・検査（E3 抜き）。最後に SETUP_OK か SETUP_FAILED と出る。
# start  = 本走を背景で起動（2 本ずつ）。止まったら同じ命令でもう一度（終わった走は飛ばし、途中の走は課題の境から再開）。
# status = 進み具合と、止める規則に当たっていないかを表示。
# stop   = 課題の境で止める（再開は start）。
# send   = 終わった結果を束ねて sha256 を出し、ギガファイル便に上げて URL を表示。
#
# このスクリプトは、worktree（git worktree add で作った claude/vit_long_1024 の checkout）の中から流す。
# <lop_analysis の clone> は .venv（python 3.12・torch 2.13.0+cu130）のある元の clone。
set -uo pipefail
CMD=${1:-help}; CLONE=${2:-}; OUTROOT=${3:-}; DATAROOT=${4:-}
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"          # この worktree
if [ "$CMD" = help ] || [ -z "$CLONE" ] || [ -z "$OUTROOT" ] || [ -z "$DATAROOT" ]; then
  sed -n 2,20p "$0"; exit 1
fi
PY="$CLONE/.venv/bin/python"
export PY
export VIT_LONG_RAW="$OUTROOT/vit_long_1024_2blk_rlfix"
export VIT_LONG_DATA="$DATAROOT/tiny-imagenet-200"
export PYTHONPATH="$HERE" VIT_LONG_PRODUCTION=production_2blk_rlfix_e.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
JOBS="$HERE/analysis/vit_long_1024/jobs_rlfix_e_byakko.txt"
LOG="$VIT_LONG_RAW/run_queue_jobs_rlfix_e_byakko.log"
ZIP_URL="https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet"
ZIP_BYTES=497536564
ZIP_SHA=05fafc211a347868093af2133504de066329059df16c2d362d068a73859f466e
mkdir -p "$VIT_LONG_RAW" "$DATAROOT"
cd "$HERE"

fail() { echo; echo "SETUP_FAILED: $*"; exit 1; }

case "$CMD" in
setup)
  echo "== 1. 版"
  [ -x "$PY" ] || fail "python が無い: $PY"
  "$PY" - <<'PYEOF' || fail "版が合わない（python 3.12・torch 2.13.0+cu130 が要る）"
import sys, torch
v = sys.version_info
assert (v.major, v.minor) == (3, 12), sys.version
assert torch.__version__.startswith('2.13.0+cu130'), torch.__version__
assert torch.cuda.is_available(), 'CUDA が見えない'
print('python', sys.version.split()[0], 'torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))
PYEOF
  "$PY" -c "import PIL, numpy; print('numpy', numpy.__version__, 'pillow', PIL.__version__)" 2>/dev/null || {
    echo "pillow を入れる"; "$PY" -m pip install -q "pillow==12.3.0" || fail "pillow が入らない"; }
  echo "== 2. Tiny ImageNet"
  ZIP="$DATAROOT/tiny-imagenet-200.kaggle_akash2sharma.zip"
  if [ ! -d "$VIT_LONG_DATA/train" ]; then
    if [ ! -f "$ZIP" ]; then
      curl -L --fail --retry 3 -o "$ZIP.part" "$ZIP_URL" || fail "データが取れない"
      mv "$ZIP.part" "$ZIP"
    fi
    [ "$(stat -c %s "$ZIP")" = "$ZIP_BYTES" ] || fail "zip の大きさが違う: $(stat -c %s "$ZIP")（$ZIP_BYTES のはず）"
    [ "$(sha256sum "$ZIP" | cut -d' ' -f1)" = "$ZIP_SHA" ] || fail "zip の sha256 が違う"
    TMP="$DATAROOT/_unzip_vit"; rm -rf "$TMP"; mkdir -p "$TMP"
    unzip -q "$ZIP" -x 'tiny-imagenet-200/tiny-imagenet-200/*' -d "$TMP" || fail "unzip に失敗"
    mv "$TMP/tiny-imagenet-200" "$VIT_LONG_DATA"; rmdir "$TMP"
  fi
  NTR=$(find "$VIT_LONG_DATA/train" -name '*.JPEG' | wc -l); NVA=$(find "$VIT_LONG_DATA/val/images" -name '*.JPEG' | wc -l)
  echo "train $NTR (100000)  val $NVA (10000)"
  [ "$NTR" = 100000 ] && [ "$NVA" = 10000 ] || fail "画像の数が合わない"
  echo "== 3. 検査（E1・E2・E4・E5。E3 は white-san の記録と比べる項目なので飛ばす。10 分ほど）"
  "$PY" -m analysis.vit_long_1024.checks_rlfix_e --skip-e3 2>&1 | grep -v -i "warning" | tail -3
  REC="$HERE/results/vit_long_1024_2blk_rlfix/checks_e_$(hostname).json"
  [ -f "$REC" ] || fail "検査の記録が無い"
  "$PY" -c "import json,sys; r=json.load(open('$REC')); print('検査', {k: r[k]['pass'] for k in ('E1','E2','E3','E4','E5')}, '1 更新', round(r['E2']['SRMS']['ms_per_update'][1],1), 'ms'); sys.exit(0 if r['pass_'] else 1)" || fail "検査が通らない（$REC を Issa に送る）"
  echo; echo "SETUP_OK  次は: bash analysis/vit_long_1024/remote/byakko.sh start $CLONE $OUTROOT $DATAROOT"
  ;;
start)
  REC="$HERE/results/vit_long_1024_2blk_rlfix/checks_e_$(hostname).json"
  [ -f "$REC" ] || { echo "先に setup を流す"; exit 1; }
  rm -f "$VIT_LONG_RAW/STOP"
  if pgrep -f "[r]un_queue.sh" >/dev/null; then echo "もう走っている（status で見る）"; exit 0; fi
  setsid nohup bash analysis/vit_long_1024/run_queue.sh "$JOBS" > "$VIT_LONG_RAW/queue_byakko.out" 2>&1 < /dev/null &
  sleep 5
  echo "起動した。進み具合: bash analysis/vit_long_1024/remote/byakko.sh status $CLONE $OUTROOT $DATAROOT"
  tail -3 "$LOG"
  ;;
status)
  echo "== 手順のログ（最後の 8 行）"; tail -8 "$LOG" 2>/dev/null || echo "まだ無い"
  echo "== 走っている数: $(pgrep -fc "analysis.vit_long_1024.run --arm" 2>/dev/null || echo 0)（2 が正常。0 なら全部終わったか止まった）"
  echo "== 本走の進み（腕 seed: 終わった課題数 / 100）"
  "$PY" - "$VIT_LONG_RAW" "$JOBS" <<'PYEOF'
import csv, json, sys
from pathlib import Path
raw, jobs = Path(sys.argv[1]), Path(sys.argv[2])
warn = []
for line in jobs.read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    parts = line.split()
    arm, seed = parts[0], parts[1]
    d = raw / 'runs' / arm / f'seed{seed}' / (f'task{parts[2]}' if len(parts) > 2 else '')
    f = d / 'per_task.csv'
    if len(parts) > 2:
        state = '完了' if (d / 'done.json').exists() else ('途中' if f.exists() else '未')
        print(f'  新品 {arm} {seed} 課題 {parts[2]}: {state}')
        continue
    if not f.exists():
        print(f'  {arm} {seed}: 未'); continue
    rows = list(csv.DictReader(open(f)))
    n = len(rows); done = (d / 'done.json').exists()
    last = rows[-1] if rows else None
    print(f"  {arm} {seed}: {n} / 100{'（完了）' if done else ''}  直近の T90 {last['T90'] if last else '—'}  1 課題 {float(last['seconds']):.0f} 秒" if last else f'  {arm} {seed}: 0 / 100')
    if arm in ('R0L', 'R0RMS'):
        bad = [r['task'] for r in rows if 2 <= int(r['task']) <= 5 and int(r['T90']) > 5000]
        if bad: warn.append(f'{arm} {seed} の課題 {bad} で 0.9 に届かず')
if warn:
    print('!! 止める規則に当たった:', '; '.join(warn), '→ stop を流して Issa に知らせる')
else:
    print('  止める規則（戻す腕の課題 2〜5）: 問題なし')
PYEOF
  grep -q ALL_DONE "$LOG" 2>/dev/null && echo "== ALL_DONE: 全部終わった。次は send"
  grep "rc=[1-9]" "$LOG" 2>/dev/null && echo "!! 終了コードが 0 でない走がある。その走の console.log の最後を Issa に送る"
  ;;
stop)
  touch "$VIT_LONG_RAW/STOP"; echo "STOP を置いた。走っている課題が終わり次第止まる（数分）。再開は start"
  ;;
send)
  grep -q ALL_DONE "$LOG" 2>/dev/null || { echo "まだ ALL_DONE でない（status で見る）。それでも送るなら、この行を無視して続ける"; }
  cd "$VIT_LONG_RAW"
  cp "$HERE"/results/vit_long_1024_2blk_rlfix/checks_e_*.json . 2>/dev/null
  TAR=vit_rlfix_e_byakko.tar
  tar --exclude='checkpoint.pt' -cf "$TAR" runs run_queue_jobs_rlfix_e_byakko.log checks_e_*.json || { echo "tar に失敗"; exit 1; }
  sha256sum "$TAR" | tee "$TAR.sha256"
  ls -l "$TAR"
  if [ ! -x gfvenv/bin/gfile ]; then
    (command -v uv >/dev/null && uv venv -q gfvenv --python 3.12 && uv pip install -q --python gfvenv/bin/python "gigafile==3.3.1") \
      || ("$PY" -m venv gfvenv && gfvenv/bin/pip install -q "gigafile==3.3.1") || { echo "gigafile が入らない。$TAR を別の方法で送る"; exit 1; }
  fi
  echo "== アップロード（数十分）"
  gfvenv/bin/gfile upload "$TAR" | tee "$TAR.upload.txt"
  echo; echo "上の URL と $TAR.sha256 の中身を Issa に送る"
  ;;
*)
  sed -n 2,20p "$0"; exit 1;;
esac
