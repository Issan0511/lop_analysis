#!/usr/bin/env python3
"""CLAUDE.md §4 step 1: move every file git does not track (untracked or ignored, except __pycache__)
under results/hsweep_rlmnist_0924 to ~/Projects/obsidian-research-data/hsweep_rlmnist_0924/<same relative path>
and write results/hsweep_rlmnist_0924/backup_manifest.json (source, backup, bytes, sha256).
Run after the small result files have been committed, so only raw states/logs remain untracked.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = "results/hsweep_rlmnist_0924"
ARCHIVE = Path.home() / "Projects/obsidian-research-data/hsweep_rlmnist_0924"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = subprocess.run(["git", "status", "--porcelain", "--ignored", "--untracked-files=all", "--", RESULTS],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        code, path = line[:2], line[3:].strip()
        if code.strip() not in ("??", "!!"):
            continue
        if "__pycache__" in path:
            continue
        p = REPO / path
        if p.is_file():
            files.append(path)
    print(f"{len(files)} untracked/ignored files to archive")
    manifest, total = [], 0
    for rel in files:
        src = REPO / rel
        dst = ARCHIVE / Path(rel).relative_to(RESULTS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest, size = sha256(src), src.stat().st_size
        shutil.move(str(src), str(dst))
        manifest.append({"source": rel, "backup": str(dst), "bytes": size, "sha256": digest})
        total += size
    # remove now-empty directories
    for d in sorted((REPO / RESULTS).rglob("*"), key=lambda q: -len(str(q))):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    (REPO / RESULTS / "backup_manifest.json").write_text(json.dumps(
        {"archive": str(ARCHIVE), "n_files": len(manifest), "total_bytes": total, "files": manifest}, indent=1) + "\n")
    print(f"moved {len(manifest)} files, {total / 1e9:.2f} GB -> {ARCHIVE}")


if __name__ == "__main__":
    sys.exit(main())
