"""CLAUDE.md section 4: move the git-untracked raw data of c1_direction_1010 out of the worktree.

Moves (never follows symlinks) every file under the listed directories that git does not track to
~/Projects/obsidian-research-data/c1_direction_1010/<same relative path>, and writes
results/c1_direction_1010/backup_manifest.json (source, backup, bytes, sha256).  Tracked files
(checks.json, *_R1.json, tables) stay.  Re-running is idempotent.

usage: .venv/bin/python analysis/c1_direction_1010/backup.py [--dry]
"""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DST = Path.home() / "Projects" / "obsidian-research-data" / "c1_direction_1010"
DIRS = ["results/c1_direction_1010/cnn", "results/c1_direction_1010/cnn_task", "results/c1_direction_1010/replay",
        "results/c1_direction_1010/mlp", "results/c1_direction_1010/logs"]
LOGS = ["analysis/c1_direction_1010/codex_exec.log", "analysis/c1_direction_1010/codex_exec2.log",
        "analysis/c1_direction_1010/codex_exec3.log"]
MANIFEST = ROOT / "results" / "c1_direction_1010" / "backup_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def untracked(paths):
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--ignored", "--untracked-files=all", "--", *paths],
                         capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        if line[:2] in ("??", "!!"):
            files.append(ROOT / line[3:].strip())
    return files


def main():
    dry = "--dry" in sys.argv
    cand = untracked(DIRS + [p for p in LOGS if (ROOT / p).exists()])
    cand = [p for p in cand if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts]
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    rows = {r["source"]: r for r in old["files"]}
    total = 0
    for src in sorted(cand):
        rel = src.relative_to(ROOT)
        dst = DST / rel
        row = {"source": str(rel), "backup": str(dst), "bytes": src.stat().st_size, "sha256": sha256(src)}
        total += row["bytes"]
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            assert sha256(dst) == row["sha256"], rel
        rows[str(rel)] = row
    man = {"experiment": "c1_direction_1010", "moved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dry": dry,
           "n_files": len(rows), "bytes": sum(r["bytes"] for r in rows.values()), "files": sorted(rows.values(), key=lambda r: r["source"])}
    if not dry:
        MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"{'would move' if dry else 'moved'} {len(cand)} files, {total / 2**20:.1f} MiB -> {DST}")


if __name__ == "__main__":
    main()
