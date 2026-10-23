#!/usr/bin/env python3
"""Archive raw unit arrays and remove only verified copies of bundle members."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_latepush_1023'
TRANSFER = Path('/home/issan/Projects/obsidian-research-data/chargpt_seed10_1022/transfer')
BACKUP = Path('/home/issan/Projects/obsidian-research-data/chargpt_latepush_1023')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    # The report must exist before deleting any input copies.
    for name in ['states.csv','summary.md','judge.md','judge.json','provenance.json']:
        assert (OUT/name).is_file(), name
    manifest = json.loads((TRANSFER/'bundle_manifest.json').read_text())
    tar = TRANSFER/manifest['tar_name']
    assert sha(tar) == manifest['tar_sha256']
    tracked = set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    records = []
    # Verify every input before deleting any input. Never remove a tracked file.
    for row in manifest['files']:
        p = ROOT/row['path']
        assert p.is_relative_to(ROOT/'results/chargpt_cl_1019')
        assert row['path'] not in tracked, row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
        records.append(dict(source=str(p), backup=str(tar), bundle_member=row['path'],
                            bytes=row['bytes'], sha256=row['sha256'], action='束から展開して削除'))
    raw_records = []
    provenance = json.loads((OUT/'provenance.json').read_text())
    for row in provenance['raw_files']:
        p = ROOT/row['path']; q = BACKUP/row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
        if q.exists():
            assert q.stat().st_size == row['bytes'] and sha(q) == row['sha256'], str(q)
        raw_records.append(dict(source=str(p), backup=str(q), bytes=row['bytes'],sha256=row['sha256'],action='moved'))
    for row in raw_records:
        p, q = Path(row['source']), Path(row['backup'])
        q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():
            p.unlink()
        else:
            shutil.move(str(p),str(q))
        assert q.stat().st_size == row['bytes'] and sha(q) == row['sha256']
    for row in records:
        Path(row['source']).unlink()
    # Remove empty directories only, restricted to this experiment's extracted tree.
    tree = ROOT/'results/chargpt_cl_1019'
    for p in sorted(tree.rglob('*'), key=lambda p:len(p.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()): p.rmdir()
    if tree.exists() and not any(tree.iterdir()): tree.rmdir()
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(
        bundle=str(tar),bundle_sha256=manifest['tar_sha256'],
        extracted_copies_removed=len(records),extracted_bytes=manifest['total_bytes'],
        raw_files_moved=len(raw_records),raw_bytes=sum(r['bytes'] for r in raw_records),
        files=records+raw_records),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(extracted_copies_removed=len(records),raw_files_moved=len(raw_records),backup=str(BACKUP)),ensure_ascii=False))


if __name__ == '__main__':
    main()
