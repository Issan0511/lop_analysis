#!/usr/bin/env bash
# On white-san: pack this branch's committed code, the white-san check records and the dataset
# manifest into one tarball for the rented GPU (README.md). Refuses uncommitted changes.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY="${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}"
MANIFEST="${MANIFEST:-$HOME/Projects/obsidian-research-data/datasets/tiny-imagenet-200.manifest_1024.sha256}"
OUT="${1:-/tmp/vit_long_1024_remote.tar.gz}"
cd "$REPO"
if [ -n "$(git status --porcelain -- src analysis specs results/vit_long_1024)" ]; then
  echo "uncommitted changes under src/ analysis/ specs/ results/vit_long_1024/: commit first" >&2
  exit 1
fi
STAGE="$(mktemp -d)"
git archive --format=tar --prefix=lop_analysis/ HEAD \
  analysis/__init__.py analysis/vit_long_1024 analysis/joudaki_vit_battle_0919 \
  analysis/randlabel_control_joudaki_vit_0919.py analysis/factorial_joudaki_vit_0919.py \
  src/__init__.py src/rlcifar_mlp_battle_0918.py src/pmnist_0905.py src/pmnist_rlcifar_0907.py \
  src/pmnist_rlmnist_0906.py specs/spec_vit_long_1024.md \
  results/joudaki_vit_battle_0919/checks_compile.json results/joudaki_vit_battle_0919/dataset.json \
  results/vit_long_1024/checks | tar -x -C "$STAGE"
git rev-parse HEAD > "$STAGE/lop_analysis/SOURCE_COMMIT"
(cd "$STAGE/lop_analysis" && "$PY" -c 'import json; from analysis.vit_long_1024 import common as C; print(json.dumps(C.source_hashes(), indent=1, sort_keys=True))' > SOURCE_HASHES.json)
cp "$MANIFEST" "$STAGE/tiny-imagenet-200.manifest_1024.sha256"
tar -czf "$OUT" -C "$STAGE" lop_analysis tiny-imagenet-200.manifest_1024.sha256
rm -r "$STAGE"
echo "packed $(cat "$REPO/.git" >/dev/null 2>&1; git rev-parse --short HEAD) -> $OUT ($(stat -c %s "$OUT") bytes)"
