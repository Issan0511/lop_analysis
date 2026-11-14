"""Copy raw study outputs, verify every SHA-256, then optionally remove originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RUN='kunekune_readout_1114'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--backup',required=True);ap.add_argument('--move',action='store_true')
    a=ap.parse_args();root=Path(a.root).resolve();backup=Path(a.backup).resolve()
    results=root/'results'/RUN
    assert json.loads((results/'integrity.json').read_text())['complete']
    assert json.loads((results/'launch_complete.json').read_text())['complete']
    files=sorted(p for d in ('raw','logs') for p in (results/d).rglob('*') if p.is_file())
    assert files and all(not p.is_symlink() for p in files)
    rows=[]
    for p in files:
        rel=p.relative_to(root);dest=backup/rel;expected=sha(p)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==expected,('backup collision',str(dest))
        else:shutil.copy2(p,dest)
        assert dest.stat().st_size==p.stat().st_size and sha(dest)==expected
        rows.append(dict(source=str(p),backup=str(dest),relative=str(rel),bytes=p.stat().st_size,sha256=expected))
    manifest=dict(complete=True,all_sha_verified=True,files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows))
    text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    (results/'backup_manifest.json').write_text(text)
    (backup/'backup_manifest.json').write_text(text)
    if a.move:
        for r in rows:
            p=Path(r['source']);assert sha(p)==r['sha256'] and sha(Path(r['backup']))==r['sha256']
            p.unlink()
        for d in ('raw','logs'):
            base=results/d
            for p in sorted(base.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if p.is_dir():p.rmdir()
            base.rmdir()
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))

if __name__=='__main__':main()
