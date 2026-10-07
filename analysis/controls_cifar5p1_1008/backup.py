#!/usr/bin/env python3
"""CLAUDE.md §4 step 1: move every git-external file of this worktree (except __pycache__) to
/home/issan/Projects/obsidian-research-data/controls_cifar5p1_1008/<same relative path> and write
results/controls_cifar5p1_1008/backup_manifest.json (relative, source, backup, bytes, sha256).
Copies, verifies the sha256 of the copy, then removes the source.  Never follows symlinks."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = 'controls_cifar5p1_1008'
ARCHIVE = Path('/home/issan/Projects/obsidian-research-data') / RUN


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


out = subprocess.run(['git', 'status', '--porcelain', '--ignored', '--untracked-files=all', '-z'],
                     cwd=ROOT, capture_output=True, text=True, check=True).stdout
paths = []
for entry in out.split('\0'):
    if not entry:
        continue
    code, rel = entry[:2], entry[3:]
    if code not in ('??', '!!') or '__pycache__' in Path(rel).parts:
        continue
    paths.append(rel)
manifest_rel = f'results/{RUN}/backup_manifest.json'
paths = sorted(p for p in paths if p != manifest_rel)
# Only run outputs are archived; code (this script included) is committed.  The first run of
# this script on 2026-10-08 archived itself; it was copied back and its manifest entry removed.
code = [p for p in paths if not p.startswith('results/')]
assert set(code) <= {f'analysis/{RUN}/backup.py'}, ('uncommitted non-result files', code)
paths = [p for p in paths if p.startswith('results/')]
files, total = [], 0
for rel in paths:
    src = ROOT / rel
    assert not src.is_symlink() and src.is_file(), ('not a regular file', rel)
    dst = ARCHIVE / rel
    h, n = sha(src), src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert sha(dst) == h, ('archive holds a different file', rel)
    else:
        shutil.copy2(src, dst)
        assert sha(dst) == h and dst.stat().st_size == n, ('copy mismatch', rel)
    files.append(dict(relative=rel, source=str(src), backup=str(dst), bytes=n, sha256=h))
    total += n
man = dict(run_id=RUN, archive_root=str(ARCHIVE), worktree=str(ROOT), files=files, total_bytes=total, n_files=len(files))
(ROOT / manifest_rel).write_text(json.dumps(man, ensure_ascii=False, indent=1) + '\n')
for f in files:                                   # remove sources only after every copy verified
    Path(f['source']).unlink()
for d in sorted({Path(f['source']).parent for f in files}, key=lambda p: -len(p.parts)):
    while d != ROOT and d.exists() and not any(d.iterdir()):
        d.rmdir()
        d = d.parent
print(f'{len(files)} files, {total / 2**20:.1f} MiB -> {ARCHIVE}')
