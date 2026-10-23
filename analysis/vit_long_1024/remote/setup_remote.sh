#!/usr/bin/env bash
# On the rented GPU (Ubuntu container, root): system packages, a uv venv with white-san's exact
# versions, and Tiny ImageNet from Kaggle's unauthenticated URL, checked against the manifest.
# No cloud API calls. Run from the folder where the tarball was unpacked (README.md).
set -euo pipefail
WORK="${WORK:-$(cd "$(dirname "$0")/../../../.." && pwd)}"      # holds lop_analysis/ and the manifest
REPO="$WORK/lop_analysis"
VENV="${VENV:-$WORK/venv}"
DATA_PARENT="${DATA_PARENT:-$HOME/Projects/obsidian-research-data/datasets}"   # the code's default
URL="https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet"
ZIP_BYTES=497536564
ZIP_SHA256=05fafc211a347868093af2133504de066329059df16c2d362d068a73859f466e
MANIFEST="$WORK/tiny-imagenet-200.manifest_1024.sha256"
echo "== 1. system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential curl ca-certificates unzip tmux procps
apt-get install -y -q python3.12-dev || echo "WARNING: python3.12-dev not installable here; uv's CPython has its own Python.h"
echo "== 2. uv venv (python 3.12.3) and pinned packages"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12.3 "$VENV"
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  -r "$REPO/analysis/vit_long_1024/remote/requirements.txt"
uv pip freeze --python "$VENV/bin/python" > "$WORK/pip_freeze_remote.txt"
grep -v '^#' "$REPO/analysis/vit_long_1024/remote/requirements.txt" | sed 's/+cu130//' | sort > "$WORK/_want.txt"
sed 's/+cu130//' "$WORK/pip_freeze_remote.txt" | sort > "$WORK/_have.txt"
if comm -23 "$WORK/_want.txt" "$WORK/_have.txt" | grep .; then echo "ERROR: pinned versions missing (above)"; exit 1; fi
echo "== 3. Tiny ImageNet (Kaggle akash2sharma/tiny-imagenet; outer copy only)"
mkdir -p "$DATA_PARENT"
ZIP="$DATA_PARENT/tiny-imagenet-200.kaggle_akash2sharma.zip"
if [ ! -f "$ZIP" ]; then
  curl -L --fail --retry 3 -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi
HAVE_BYTES=$(stat -c %s "$ZIP"); HAVE_SHA=$(sha256sum "$ZIP" | cut -d' ' -f1)
echo "archive: $HAVE_BYTES bytes, sha256 $HAVE_SHA"
if [ "$HAVE_BYTES" != "$ZIP_BYTES" ] || [ "$HAVE_SHA" != "$ZIP_SHA256" ]; then
  echo "WARNING: archive differs from white-san's ($ZIP_BYTES, $ZIP_SHA256); the per-file manifest decides"
fi
if [ ! -d "$DATA_PARENT/tiny-imagenet-200/train" ]; then
  TMP="$DATA_PARENT/_unzip_vit_long_1024"
  rm -rf "$TMP"; mkdir "$TMP"
  unzip -q "$ZIP" -x 'tiny-imagenet-200/tiny-imagenet-200/*' -d "$TMP"     # drop the nested duplicate
  mv "$TMP/tiny-imagenet-200" "$DATA_PARENT/tiny-imagenet-200"
  rmdir "$TMP"
fi
cp "$MANIFEST" "$DATA_PARENT/tiny-imagenet-200.manifest_1024.sha256"
X="$DATA_PARENT/tiny-imagenet-200"
echo "train $(find "$X/train" -name '*.JPEG' | wc -l) (100000)  val $(find "$X/val/images" -name '*.JPEG' | wc -l) (10000)" \
     "wnids $(wc -l < "$X/wnids.txt") (200)  val_annotations $(wc -l < "$X/val/val_annotations.txt") (10000)"
(cd "$X" && sha256sum -c --quiet "$DATA_PARENT/tiny-imagenet-200.manifest_1024.sha256") && echo "manifest: all 110203 files match"
echo "== 4. stack"
"$VENV/bin/python" - << 'PYEOF'
import sysconfig, os, torch, numpy, PIL
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, 'pillow', PIL.__version__)
print('gpu', torch.cuda.get_device_name(), 'Python.h', os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))
f = torch.compile(lambda x: (x * 2).sin().sum())
print('compile ok', float(f(torch.ones(8, device='cuda'))))
PYEOF
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "setup done. next: bash $REPO/analysis/vit_long_1024/remote/run_remote.sh"
