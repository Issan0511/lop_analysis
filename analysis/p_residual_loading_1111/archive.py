"""Archive only this experiment's ignored/untracked data after results commit."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/home/issan/Projects/obsidian-research-data/p_residual_loading_1111')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    assert ROOT.name=='p_residual_loading_1111', 'Run from the isolated experiment worktree only'
    raw=subprocess.check_output(['git','status','--porcelain','--ignored','--untracked-files=all','-z'],cwd=ROOT).decode()
    entries=[]
    for record in filter(None,raw.split('\0')):
        code,rel=record[:2],record[3:]
        if code not in ('??','!!'):continue
        p=Path(rel)
        if '__pycache__' in p.parts:continue
        assert p.parts[:2]==('results','p_residual_loading_1111'), ('Commit source/report first; unexpected untracked file',rel)
        src=ROOT/p;assert src.is_file() and not src.is_symlink(),str(src)
        dst=DEST/p;h=sha(src);entries.append(dict(source=str(src),backup=str(dst),relative_path=rel,bytes=src.stat().st_size,sha256=h))
    assert entries,'No data to archive'
    if not args.execute:
        print(json.dumps(dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),destination=str(DEST))));return
    for r in entries:
        src,dst=Path(r['source']),Path(r['backup']);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==r['sha256'],'Existing different backup: '+str(dst)
        else:shutil.copy2(src,dst)
        assert dst.stat().st_size==r['bytes'] and sha(dst)==r['sha256']
    manifest=ROOT/'results/p_residual_loading_1111/backup_manifest.json'
    manifest.write_text(json.dumps(dict(complete=True,files=len(entries),bytes=sum(r['bytes'] for r in entries),archive_root=str(DEST),entries=entries),indent=2)+'\n')
    # Remove sources only after all copies and the durable manifest succeed.
    for r in entries:Path(r['source']).unlink()
    print(json.dumps(dict(complete=True,files=len(entries),bytes=sum(r['bytes'] for r in entries),manifest=str(manifest))))
if __name__=='__main__':main()
