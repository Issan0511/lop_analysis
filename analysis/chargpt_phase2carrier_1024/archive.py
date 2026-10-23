#!/usr/bin/env python3
"""Archive only this experiment's generated raw arrays and verify every hash."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_phase2carrier_1024'
BACKUP=Path('/home/issan/Projects/obsidian-research-data/chargpt_phase2carrier_1024')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

assert (OUT/'audit.json').exists()
assert json.loads((OUT/'audit.json').read_text())['states']==90
files=sorted((OUT/'raw').glob('*'))
assert len(files)==180, len(files)
records=[]
for p in files:
    assert p.suffix in ('.npz','.json') and p.name.startswith('C3_seed')
    dest=BACKUP/p.relative_to(ROOT)
    digest=sha(p)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        assert sha(dest)==digest, dest
    else:
        shutil.copy2(p,dest)
    assert p.stat().st_size==dest.stat().st_size and sha(dest)==digest
    records.append(dict(source=str(p),source_relative=str(p.relative_to(ROOT)),backup=str(dest),bytes=p.stat().st_size,sha256=digest))
manifest=dict(experiment='chargpt_phase2carrier_1024',files=records,
              checkpoint_action='束のcheckpointとAdam状態はtarから直接読取。展開・削除なし。',
              generated_files=len(records),generated_bytes=sum(r['bytes'] for r in records),verified=True)
(OUT/'backup_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
for p in files:p.unlink()
(OUT/'raw').rmdir()
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False))
