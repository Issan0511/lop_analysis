#!/usr/bin/env python3
"""Archive only this experiment's raw unit arrays, verify, then remove copies."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_h1audit_1023'
DEST = Path('/home/issan/Projects/obsidian-research-data/chargpt_h1audit_1023')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4<<20), b''): h.update(block)
    return h.hexdigest()


def main():
    paths = sorted((OUT/'raw').glob('*.npz'))
    assert len(paths)==20, len(paths)
    rec = []
    for p in paths:
        dest = DEST/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(p)
        if dest.exists(): assert sha(dest)==digest
        else: shutil.copy2(p, dest)
        assert sha(dest)==digest and dest.stat().st_size==p.stat().st_size
        rec.append(dict(source=str(p.relative_to(ROOT)), backup=str(dest), bytes=p.stat().st_size, sha256=digest))
    manifest = dict(files=rec, count=len(rec), total_bytes=sum(r['bytes'] for r in rec), all_verified=True,
        checkpoint_note='既存bundleをメモリで読み出し。checkpointの展開複製なし、削除対象なし。元のbundleは変更せず。')
    (OUT/'backup_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    for p in paths: p.unlink()
    (OUT/'raw').rmdir()
    print(json.dumps(dict(count=len(rec), total_bytes=manifest['total_bytes'], all_verified=True)))


if __name__=='__main__': main()
