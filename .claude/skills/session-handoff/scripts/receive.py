#!/usr/bin/env python3
"""receive.py — 引き継ぎ manifest からギガファイル便を落とし、sha256 を照合して展開する。

    python3 receive.py --manifest manifest.json --results-dir results [--only a,b] [--download-dir DIR]

- データ tar（kind=tar）は `results-dir` に展開する（トップレベルが run 名なので results/<run>/ になる）。
- manifest に `extract_to` があればそこへ（repo 基準の相対パス）。tar のトップが run 名でない束（`logs_tail/` など）用。
- 束（kind=tgz・name が bundle を含む）は `download-dir/<name>/` に展開する。
- 展開済み（results/<run>/arm_status がある等）はサイズ照合だけして飛ばす。
依存: pip install gigafile（モジュール名 gfile）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


DL_SNIPPET = r"""
import sys, io, contextlib
from gfile import GFile
g = GFile(sys.argv[1], progress=False, mute=True)
with contextlib.redirect_stdout(io.StringIO()):
    g.download(output=sys.argv[2])
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--download-dir", default="_handoff_dl")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    only = {s for s in a.only.split(",") if s}
    dl = Path(a.download_dir); dl.mkdir(parents=True, exist_ok=True)
    res = Path(a.results_dir); res.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, rec in man["items"].items():
        if only and name not in only:
            continue
        if not rec.get("url"):
            print(f"[skip] {name}: URL なし"); continue
        fname = Path(rec["file"]).name
        local = dl / fname
        if not (local.exists() and local.stat().st_size == rec["size_bytes"]):
            print(f"[download] {name} {rec['size_bytes']/1e9:.2f} GB ← {rec['url']}", flush=True)
            subprocess.run([sys.executable, "-c", DL_SNIPPET, rec["url"], str(local)], check=True)
        got = sha256_of(local)
        if got != rec["sha256"]:
            print(f"[FAIL] {name}: sha256 不一致 {got[:12]} != {rec['sha256'][:12]}"); bad += 1; continue
        print(f"[ok] {name}: sha256 一致")
        kind = rec.get("kind", "tar")
        if kind == "file":
            continue
        if rec.get("extract_to"):
            # manifest の extract_to（results-dir の親＝repo 基準の相対パス、または絶対パス）
            dest = Path(rec["extract_to"])
            if not dest.is_absolute():
                dest = res.parent / dest
        else:
            dest = (dl / name) if (kind == "tgz" or "bundle" in name) else res
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(local), "-C", str(dest)], check=True)
        print(f"[extract] {name} → {dest}/")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
