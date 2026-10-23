#!/usr/bin/env python3
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/chargpt_l0force_1023'
DEST=Path('/home/issan/Projects/obsidian-research-data/chargpt_l0force_1023')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for folder in [OUT/'raw',OUT/'full_self/raw']:
 for p in sorted(folder.glob('*.npz')):
  dest=DEST/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);h=sha(p);shutil.copy2(p,dest);assert sha(dest)==h
  rows.append(dict(source=str(p),backup=str(dest),bytes=p.stat().st_size,sha256=h));p.unlink()
 folder.rmdir()
manifest=dict(files=rows,checkpoint_note='元のseed10束とl0carrierの検証済み退避を読み取り。checkpoint展開/生成なし。元のデータは変更しない。',checkpoints_created=0,training_steps=0)
(OUT/'backup_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('archived',len(rows),'arrays',sum(r['bytes'] for r in rows),'bytes')
