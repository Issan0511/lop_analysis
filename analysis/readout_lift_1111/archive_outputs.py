#!/usr/bin/env python3
"""Archive only this experiment's untracked/ignored outputs, then verify hashes."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/home/issan/Projects/obsidian-research-data/readout_lift_1111')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    inventory=subprocess.check_output(['git','status','--porcelain','--ignored','--untracked-files=all','-z'],cwd=ROOT).decode()
    files=set();other=[]
    for rec in inventory.split('\0'):
        if not rec or rec[:2] not in ('??','!!'):continue
        path=rec[3:]
        if '__pycache__' in Path(path).parts:continue
        if not path.startswith('results/readout_lift_1111/'):
            other.append(path);continue
        p=ROOT/path
        files.update(q for q in p.rglob('*') if q.is_file()) if p.is_dir() else files.add(p)
    assert not other,('untracked outside experiment outputs',other)
    DEST.mkdir(parents=True,exist_ok=False);rows=[]
    for p in sorted(files):
        assert p.is_file() and not p.is_symlink();rel=p.relative_to(ROOT);dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);rows.append(dict(source=str(p),backup=str(dest),relative_path=str(rel),bytes=p.stat().st_size,sha256=sha(p),group='worktree'))
    for folder in ('smoke','followup_smoke'):
        top=Path('/tmp/readout_lift_1111')/folder
        for p in sorted(q for q in top.rglob('*') if q.is_file()):
            dest=DEST/'validation'/folder/p.relative_to(top);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            rows.append(dict(source=str(p),backup=str(dest),relative_path=str(dest.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p),group='validation'))
    for row in rows:
        p=Path(row['backup']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row
    manifest=dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),entries=rows)
    (ROOT/'results/readout_lift_1111/backup_manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in files:p.unlink()
    print(json.dumps(dict(files=manifest['files'],bytes=manifest['bytes'],archive=str(DEST))))

if __name__=='__main__':main()
