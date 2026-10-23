#!/usr/bin/env python3
"""Verified archival of this experiment's generated per-unit arrays only."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/chargpt_loadsign_1023'
DEST=Path('/home/issan/Projects/obsidian-research-data/chargpt_loadsign_1023')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    rows=[]
    sources=sorted((OUT/'raw').glob('*.npz'))+sorted((OUT/'raw_components').glob('*.npz'))
    for source in sources:
        backup=DEST/source.relative_to(ROOT)
        parts='parts' if source.parent.name=='raw' else 'parts_components'
        expected=json.loads((OUT/parts/source.with_suffix('.json').name).read_text())['raw_sha256']
        assert sha(source)==expected
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists(): assert sha(backup)==expected
        else: shutil.copy2(source,backup)
        assert sha(backup)==expected
        rows.append(dict(source=str(source.relative_to(ROOT)),backup=str(backup),bytes=source.stat().st_size,sha256=expected))
    assert len(rows)==40
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(files=rows,
        checkpoints='Read directly from the existing seed10 tar bundle; no extracted checkpoint copies were created.',
        source_bundle='/home/issan/Projects/obsidian-research-data/chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar'),indent=2)+'\n')
    for row in rows:(ROOT/row['source']).unlink()
    (OUT/'raw').rmdir()
    (OUT/'raw_components').rmdir()
    print(f'Archived and verified {len(rows)} arrays, {sum(r["bytes"] for r in rows)} bytes')


if __name__=='__main__':main()
