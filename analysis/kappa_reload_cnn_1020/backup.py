"""CLAUDE.md section 4: move the git-untracked raw data of kappa_reload_cnn_1020 out of the worktree.

Moves (never follows symlinks) every file under the listed directories that git does not track to
~/Projects/obsidian-research-data/kappa_reload_cnn_1020/<same relative path>, and writes
results/kappa_reload_cnn_1020/backup_manifest.json (source, backup, bytes, sha256).  Tracked files
(tables, rows.json, checks.json, verdict.json, logs) stay.  Re-running is idempotent.

usage: .venv/bin/python analysis/kappa_reload_cnn_1020/backup.py [--dry]
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DST = Path.home() / "Projects" / "obsidian-research-data" / "kappa_reload_cnn_1020"
DIRS = ["results/kappa_reload_cnn_1020", "analysis/kappa_reload_cnn_1020"]
MANIFEST = ROOT / "results" / "kappa_reload_cnn_1020" / "backup_manifest.json"


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
    cand = untracked(DIRS)
    cand = [p for p in cand if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts]
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    rows = {r["source"]: r for r in old["files"]}
    for src in sorted(cand):
        rel = src.relative_to(ROOT)
        dst = DST / rel
        row = {"source": str(rel), "backup": str(dst), "bytes": src.stat().st_size, "sha256": sha256(src)}
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            assert sha256(dst) == row["sha256"], rel
        rows[str(rel)] = row
        print(("DRY " if dry else "moved ") + str(rel), row["bytes"])
    man = {"experiment": "kappa_reload_cnn_1020", "moved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dry": dry,
           "n_files": len(rows), "bytes": sum(r["bytes"] for r in rows.values()),
           "files": sorted(rows.values(), key=lambda r: r["source"])}
    if not dry:
        MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        for d in DIRS:
            for p in sorted((ROOT / d).rglob("*"), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
    print("files", man["n_files"], "bytes", man["bytes"], "dry", dry)


if __name__ == "__main__":
    main()
