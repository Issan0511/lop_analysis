#!/usr/bin/env python3
"""Archive git-external outputs with an auditable manifest, after committing summaries."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def archive(run):
    target=Path('/home/issan/Projects/obsidian-research-data')/run
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True)
    assert not status,'Commit all intended code/reports before archiving; ignore raw output explicitly.'
    raw=subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files=[Path(p) for p in raw if p and '__pycache__' not in Path(p).parts]
    assert all(p.parts[0]=='results' for p in files),'Unexpected ignored file outside results: inspect before archiving.'
    records=[]
    for rel in sorted(files):
        source=ROOT/rel;backup=target/rel
        assert source.is_file() and not source.is_symlink(),str(source)
        size=source.stat().st_size;h=digest(source);backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.stat().st_size==size and digest(backup)==h;source.unlink()
        else:shutil.move(source,backup)
        assert backup.stat().st_size==size and digest(backup)==h
        records.append(dict(relative=str(rel),source=str(source),backup=str(backup),bytes=size,sha256=h))
    manifest=ROOT/'results'/run/'backup_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())['files'];lookup={r['relative']:r for r in old}
        for r in records:
            if r['relative'] in lookup:assert lookup[r['relative']]['sha256']==r['sha256']
            lookup[r['relative']]=r
        records=list(lookup.values())
    obj=dict(run_id=run,archive_root=str(target),files=records,total_bytes=sum(r['bytes'] for r in records))
    manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=obj['total_bytes'],manifest=str(manifest))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();archive(a.run)
