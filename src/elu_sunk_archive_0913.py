"""Archive raw rescue arrays/checkpoints after all registered jobs complete."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/home/issan/Projects/obsidian-research-data/elu_sunk_rescue_0913')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 jobs=list((ROOT/'results/elu_sunk_rescue_0913').glob('*_t*_l*/provenance.json'))
 assert len(jobs)==12 and all(json.loads(p.read_text())['status']=='COMPLETE' for p in jobs)
 files=[]
 for run in ['elu_sunk_rescue_0913','elu_sunk_tracking_0913']:
  for src in sorted((ROOT/'results'/run).rglob('*')):
   if src.is_file() and src.suffix in ('.pt','.npz'):
    rel=src.relative_to(ROOT);dst=DEST/'files'/rel;dst.parent.mkdir(parents=True,exist_ok=True);h=sha(src)
    if dst.exists():assert sha(dst)==h,dst
    else:shutil.copy2(src,dst)
    assert sha(dst)==h
    files.append(dict(repo_path=str(rel),backup_path=str(dst),sha256=h,bytes=src.stat().st_size))
 out=dict(status='SHA256_COPY_VERIFIED',backup_root=str(DEST),scope='Same-host separate directory; raw binaries excluded from Git clone',files=files,total_bytes=sum(r['bytes'] for r in files))
 txt=json.dumps(out,indent=2)+'\n'
 (ROOT/'results/elu_sunk_rescue_0913/raw_manifest.json').write_text(txt);(DEST/'manifest.json').write_text(txt)
 print(json.dumps({k:v for k,v in out.items() if k!='files'}));print('raw_files',len(files))
if __name__=='__main__':main()
