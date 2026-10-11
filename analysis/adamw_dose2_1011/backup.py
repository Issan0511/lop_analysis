"""CLAUDE.md section 4: move the git-untracked raw data of adamw_dose2_1011 out of the worktree.

Moves (never follows symlinks; the `data` symlink is never listed) every file under the listed directories that git
does not track to ~/Projects/obsidian-research-data/adamw_dose2_1011/<same relative path>, and writes
results/adamw_dose2_1011/backup_manifest.json (relative, source, backup, bytes, sha256).  Tracked files stay.
Re-running is idempotent (rows already in the manifest are kept).  verdict.py reads the units from the backup the
manifest names.

usage: .venv/bin/python analysis/adamw_dose2_1011/backup.py [--dry]
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DST = Path.home() / "Projects" / "obsidian-research-data" / "adamw_dose2_1011"
DIRS = ["results/adamw_dose2_1011", "results/_checks_adamw_dose2_1011"]
MANIFEST = ROOT / "results" / "adamw_dose2_1011" / "backup_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def untracked(paths):
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--ignored", "--untracked-files=all", "--",
                          *paths], capture_output=True, text=True, check=True).stdout
    return [ROOT / line[3:].strip() for line in out.splitlines() if line[:2] in ("??", "!!")]


def main():
    dry = "--dry" in sys.argv
    cand = [p for p in untracked([d for d in DIRS if (ROOT / d).exists()])
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p != MANIFEST]
    for p in cand:
        assert ROOT.resolve() in p.resolve().parents, f"outside the worktree: {p} -> {p.resolve()}"
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    rows = {r["relative"]: r for r in old["files"]}
    total = 0
    for src in sorted(cand):
        rel = src.relative_to(ROOT)
        dst = DST / rel
        row = {"relative": str(rel), "source": str(src), "backup": str(dst), "bytes": src.stat().st_size,
               "sha256": sha256(src)}
        total += row["bytes"]
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            assert sha256(dst) == row["sha256"], rel
        rows[str(rel)] = row
    man = {"run_id": "adamw_dose2_1011", "moved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dry": dry,
           "archive_root": str(DST),
           "rule": "CLAUDE.md 4: every git-untracked file under " + ", ".join(DIRS) + " (no symlinks, no __pycache__)",
           "n_files": len(rows), "total_bytes": sum(r["bytes"] for r in rows.values()),
           "files": sorted(rows.values(), key=lambda r: r["relative"])}
    if not dry:
        MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"{'would move' if dry else 'moved'} {len(cand)} files, {total / 2**20:.1f} MiB -> {DST}")


if __name__ == "__main__":
    main()
