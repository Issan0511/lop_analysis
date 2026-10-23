#!/usr/bin/env python3
"""Verified archival; only this experiment's generated arrays are removed."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_l0carrier_1023'
DEST=Path('/home/issan/Projects/obsidian-research-data/chargpt_l0carrier_1023')
SOURCE=Path('/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/M2')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def preserve(source,backup,expected):
    assert sha(source)==expected,source
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists():assert sha(backup)==expected,backup
    else:shutil.copy2(source,backup)
    assert sha(backup)==expected
    return dict(source=str(source),backup=str(backup),bytes=source.stat().st_size,sha256=expected)


def main():
    pv=json.loads((OUT/'provenance.json').read_text());assert pv['model_states']==700
    files=[];inputs=[];sources=[]
    for r in pv['borrowed_files']:
        source=Path(r['path']);backup=DEST/'inputs/M2'/source.relative_to(SOURCE)
        row=preserve(source,backup,r['sha256']);row['action']='copied; original other-worktree file retained unchanged';inputs.append(row)
    for r in pv['raw_files']:
        source=ROOT/r['path'];backup=DEST/r['path']
        row=preserve(source,backup,r['sha256']);row['action']='verified copy, then generated local array removed';files.append(row);sources.append(source)
    assert len(files)==20 and len(inputs)==240
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(files=files,borrowed_input_copies=inputs,
        original_end_checkpoints='Read in memory from the original seed10 bundle, no extracted copies.',
        input_restore_root=str(DEST/'inputs/M2'),source_bundle=pv['bundle_path'],source_bundle_sha256=pv['bundle_sha256']),indent=2)+'\n')
    for p in sources:p.unlink()
    (OUT/'raw').rmdir()
    print(f'Archived and SHA256-verified {len(files)} generated arrays ({sum(r["bytes"] for r in files)} bytes) and copied {len(inputs)} read-only borrowed inputs ({sum(r["bytes"] for r in inputs)} bytes).')


if __name__=='__main__':main()
