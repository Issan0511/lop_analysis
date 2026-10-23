#!/usr/bin/env python3
"""Archive section-10 arrays and preserve read-only borrowed ELR inputs."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_loadsign_1023/rotation'
DEST=Path('/home/issan/Projects/obsidian-research-data/chargpt_loadsign_1023')
SOURCE=Path('/home/issan/Projects/claude/wt/chargpt_cl_1019')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4<<20),b''):h.update(data)
    return h.hexdigest()


def copy(source,backup,expected=None):
    actual=sha(source)
    if expected is not None: assert actual==expected,(source,actual,expected)
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists(): assert sha(backup)==actual,backup
    else: shutil.copy2(source,backup)
    assert sha(backup)==actual
    return dict(source=str(source),backup=str(backup),bytes=source.stat().st_size,sha256=actual)


def main():
    assert json.loads((OUT/'provenance.json').read_text())['states']==700
    assert json.loads((OUT/'elr_gate.json').read_text())['passed']
    inventory=json.loads((OUT/'elr_inventory.json').read_text())
    ep=json.loads((OUT/'elr_provenance.json').read_text())
    inputs={}
    for r in inventory['checkpoints']:
        if r['exists']:inputs[Path(r['path'])]=r['sha256']
    for r in ep['input_files']:
        inputs[Path(r['rownorm_path'])]=r['rownorm_sha256']
        folder=SOURCE/f"results/chargpt_cl_1019/row5/elr/elr/seed{r['seed']}"
        for name in ['per_task.csv','layer_stats.csv','provenance.json']:
            inputs[folder/name]=None
    inputs[Path(inventory['source_path'])]=inventory['source_sha256']
    input_rows=[]
    for source,expected in sorted(inputs.items()):
        backup=DEST/'inputs/rotation_elr'/source.relative_to(SOURCE)
        row=copy(source,backup,expected);row['action']='copied; original other-worktree file retained unchanged';input_rows.append(row)
    manifest=OUT/'backup_manifest.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else dict(files=[])
    for r in old['files']: assert sha(Path(r['backup']))==r['sha256']
    rows=old['files'];sources=sorted((OUT/'raw').glob('*.npz'))
    if sources: assert len(sources)==21
    for source in sources:
        backup=DEST/source.relative_to(ROOT)
        row=copy(source,backup);row['action']='verified copy, then generated local array removed';rows.append(row)
    assert len(rows)==21
    manifest.write_text(json.dumps(dict(files=rows,borrowed_input_copies=input_rows,
        input_restore_root=str(DEST/'inputs/rotation_elr'),
        checkpoints='Original seed10 bundle read in memory; no extracted copies. Borrowed ELR inputs copied for provenance; originals untouched.',
        original_manifest_unchanged='../backup_manifest.json'),ensure_ascii=False,indent=2)+'\n')
    for source in sources:source.unlink()
    if (OUT/'raw').exists():(OUT/'raw').rmdir()
    print(f'Archived {len(rows)} arrays ({sum(r["bytes"] for r in rows)} bytes), preserved {len(input_rows)} borrowed inputs ({sum(r["bytes"] for r in input_rows)} bytes).')


if __name__=='__main__':main()
