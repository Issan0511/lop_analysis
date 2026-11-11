#!/usr/bin/env python3
"""Archive remaining experiment outputs after committing chosen review artifacts."""
import hashlib, json, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/home/issan/Projects/obsidian-research-data/demand_allocation_1111')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    names=set()
    for flags in [[],['--ignored']]:
        out=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z',*flags],cwd=ROOT)
        names.update(p.decode() for p in out.split(b'\0') if p)
    names=sorted(n for n in names if '__pycache__' not in Path(n).parts)
    assert names
    assert all(n.startswith('results/demand_allocation_1111/') for n in names),names
    entries=[]
    for name in names:
        src=ROOT/name;dst=DEST/name
        assert src.is_file() and not src.is_symlink(),name
        assert not dst.exists(),dst
        dst.parent.mkdir(parents=True,exist_ok=True)
        size=src.stat().st_size;digest=sha(src)
        shutil.copy2(src,dst)
        assert dst.stat().st_size==size and sha(dst)==digest
        entries.append(dict(source=str(src),relative=name,backup=str(dst),bytes=size,sha256=digest))
    # Delete only after every copy has been verified.
    for e in entries:Path(e['source']).unlink()
    manifest=ROOT/'results/demand_allocation_1111/backup_manifest.json'
    manifest.write_text(json.dumps(dict(experiment='demand_allocation_1111',files=entries,total_bytes=sum(e['bytes'] for e in entries)),indent=2))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),manifest=str(manifest))))

if __name__=='__main__':main()
