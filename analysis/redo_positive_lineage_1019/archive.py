"""Move this experiment's ignored artifacts to the prescribed local archive, verify every file."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parents[2]
results=root/'results/redo_positive_lineage_1019'
dest=Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


manifest=json.loads((results/'backup_manifest.json').read_text())['files'] if (results/'backup_manifest.json').exists() else []
def record(row):
    global manifest
    manifest=[old for old in manifest if old['source']!=row['source']]
    manifest.append(row)
for folder in ('raw','checks','logs'):
    source=results/folder
    for src in sorted(source.rglob('*')):
        if not src.is_file():continue
        if src.is_symlink():raise RuntimeError(f'unexpected symlink: {src}')
        dst=dest/src.relative_to(results)
        digest=sha(src); size=src.stat().st_size
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst)!=digest:raise RuntimeError(f'archive collision: {dst}')
            src.unlink()
        else:shutil.move(str(src),str(dst))
        if sha(dst)!=digest:raise RuntimeError(f'archive verification failed: {dst}')
        record(dict(source=str(src),backup=str(dst),bytes=size,sha256=digest))
    if source.exists():shutil.rmtree(source)

exports=dest/'exports'
exports.mkdir(parents=True,exist_ok=True)
for src in sorted(results.iterdir()):
    if not src.is_file() or src.name=='backup_manifest.json':continue
    dst=exports/src.name
    shutil.copy2(src,dst)
    record(dict(source=str(src),backup=str(dst),bytes=src.stat().st_size,sha256=sha(dst),kind='report_copy'))
for sub in ('analysis/redo_positive_lineage_1019','specs/spec_redo_positive_lineage_1019.md'):
    src=root/sub
    if src.is_dir():
        for f in sorted(src.glob('*.py')):
            dst=dest/'sources'/f.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    else:
        dst=dest/'sources'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
info=dict(run_id='redo_positive_lineage_1019',files=manifest,
          total_bytes=sum(r['bytes'] for r in manifest),
          note='Raw, check and log files moved; report copies retained in repo. Every archive SHA verified.')
(results/'backup_manifest.json').write_text(json.dumps(info,indent=2))
(dest/'backup_manifest.json').write_text(json.dumps(info,indent=2))
print(json.dumps(dict(files=len(manifest),total_bytes=info['total_bytes'],destination=str(dest)),indent=2))
