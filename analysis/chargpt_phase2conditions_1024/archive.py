#!/usr/bin/env python3
"""Archive only this experiment's raw arrays and logs, verifying each copy."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'chargpt_phase2conditions_1024'
DEST = Path('/home/issan/Projects/obsidian-research-data')/NAME
OUT = ROOT/'results'/NAME


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    records = []
    for folder in [OUT/'raw', OUT/'logs']:
        for source in sorted(folder.rglob('*')):
            if not source.is_file():
                continue
            target = DEST/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = sha(source)
            if target.exists():
                assert sha(target) == digest, target
            else:
                shutil.copy2(source, target)
            assert sha(target) == digest
            records.append(dict(source=str(source), backup=str(target), bytes=source.stat().st_size, sha256=digest))
    assert records
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(experiment=NAME, records=records,
        checkpoints='Read-only from original bundle / other worktree; no extracted checkpoint copies.'), ensure_ascii=False, indent=2)+'\n')
    for record in records:
        source = Path(record['source'])
        assert sha(source) == record['sha256']
        source.unlink()
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records), destination=str(DEST))))


if __name__ == '__main__':
    main()
