"""Archive the posthoc individual arrays and a readable copy of all outputs."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[2]
out=root/'results/redo_v_opening_1019'
dest=Path('/home/issan/Projects/obsidian-research-data/redo_v_opening_1019')
manifest=[]
for src in sorted((out/'raw').glob('*')):
    if not src.is_file():continue
    dst=dest/'raw'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256(src.read_bytes()).hexdigest()
    if dst.exists():raise RuntimeError(f'archive collision: {dst}')
    size=src.stat().st_size;shutil.move(str(src),str(dst))
    if hashlib.sha256(dst.read_bytes()).hexdigest()!=h:raise RuntimeError('archive mismatch')
    manifest.append(dict(source=str(src),backup=str(dst),bytes=size,sha256=h))
for src in sorted(out.iterdir()):
    if not src.is_file() or src.name=='backup_manifest.json':continue
    dst=dest/'exports'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    manifest.append(dict(source=str(src),backup=str(dst),bytes=src.stat().st_size,
                         sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),kind='report_copy'))
for src in (root/'analysis/redo_v_opening_1019').glob('*.py'):
    dst=dest/'sources'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
log=Path('/tmp/redo_v_opening_1019_final.log')
if log.exists():shutil.copy2(log,dest/'analysis_stdout.log')
text=json.dumps(dict(run_id='redo_v_opening_1019',files=manifest,total_bytes=sum(r['bytes'] for r in manifest)),indent=2)
(out/'backup_manifest.json').write_text(text);(dest/'backup_manifest.json').write_text(text)
print('Archived',len(manifest),'files to',dest)
