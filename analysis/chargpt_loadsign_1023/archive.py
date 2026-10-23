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
    manifest=OUT/'backup_manifest.json'
    rows=json.loads(manifest.read_text())['files'] if manifest.exists() else []
    for row in rows:assert sha(Path(row['backup']))==row['sha256']
    sources=sum([sorted((OUT/d).glob('*.npz')) for d in ['raw','raw_components','raw_loading']],[])
    for source in sources:
        backup=DEST/source.relative_to(ROOT)
        parts={'raw':'parts','raw_components':'parts_components','raw_loading':'parts_loading'}[source.parent.name]
        expected=json.loads((OUT/parts/source.with_suffix('.json').name).read_text())['raw_sha256']
        assert sha(source)==expected
        backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists(): assert sha(backup)==expected
        else: shutil.copy2(source,backup)
        assert sha(backup)==expected
        rows.append(dict(source=str(source.relative_to(ROOT)),backup=str(backup),bytes=source.stat().st_size,sha256=expected))
    assert len(rows)==60
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(files=rows,
        checkpoints='Read directly from the existing seed10 tar bundle; no extracted checkpoint copies were created.',
        source_bundle='/home/issan/Projects/obsidian-research-data/chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar'),indent=2)+'\n')
    for source in sources:source.unlink()
    for d in ['raw','raw_components','raw_loading']:
        if (OUT/d).exists():(OUT/d).rmdir()
    print(f'Archived and verified {len(rows)} arrays, {sum(r["bytes"] for r in rows)} bytes')


if __name__=='__main__':main()
