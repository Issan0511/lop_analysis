#!/usr/bin/env python3
"""CLAUDE.md §4-1: move every file git does not track (ignored + untracked, except __pycache__
and the `data` symlink) out of this worktree into obsidian-research-data/<name>/ at the same
relative path, and write results/<exp>/backup_manifest.json (source, backup, bytes, sha256).

    python3 analysis/sna_cnn_cause_1009/backup.py --name sna_cnn_cause_1009 [--dry-run]

Symlinks are never followed: a symlink entry is skipped and listed under "skipped" (the 0918
accident moved the shared CIFAR archive through the `data` link).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST_ROOT = Path.home() / "Projects" / "obsidian-research-data"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="<experiment>_<MMDD> folder under obsidian-research-data")
    ap.add_argument("--exp", default=None, help="results/<exp>/ for the manifest (default = --name)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    exp = args.exp or args.name
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--ignored", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    files, skipped = [], []
    for line in out:
        if not (line.startswith("!! ") or line.startswith("?? ")):
            continue
        rel = line[3:].rstrip("/")
        p = ROOT / rel
        if p.is_symlink():
            skipped.append({"path": rel, "why": "symlink (not followed)"})
            continue
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_symlink():
                    skipped.append({"path": str(q.relative_to(ROOT)), "why": "symlink (not followed)"})
                elif q.is_file() and "__pycache__" not in str(q):
                    files.append(q.relative_to(ROOT))
        elif p.is_file():
            files.append(Path(rel))
    dest = DEST_ROOT / args.name
    entries, total = [], 0
    for rel in files:
        src = ROOT / rel
        size = src.stat().st_size
        total += size
        entries.append({"source": str(rel), "backup": str(dest / rel), "bytes": size,
                        "sha256": sha256(src) if not args.dry_run else None})
    print(f"{len(files)} files, {total / 2**20:.1f} MiB -> {dest}  (skipped {len(skipped)})")
    if args.dry_run:
        for e in entries[:40]:
            print(" ", e["source"], e["bytes"])
        return
    for e in entries:
        src, dst = ROOT / e["source"], Path(e["backup"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if sha256(dst) != e["sha256"]:
            raise SystemExit(f"sha256 mismatch after move: {e['source']}")
    manifest = {"name": args.name, "worktree": str(ROOT), "moved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip(),
                "n_files": len(entries), "total_bytes": total, "files": entries, "skipped": skipped}
    mp = ROOT / "results" / exp / "backup_manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"wrote {mp}")


if __name__ == "__main__":
    main()
