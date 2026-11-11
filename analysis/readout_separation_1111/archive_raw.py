"""Archive this worktree's ignored data with verified hashes before removal."""
import argparse,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=Path('/home/issan/Projects/obsidian-research-data/readout_separation_1111')
DATA=ROOT/'results/readout_separation_1111'
MANIFEST=DATA/'backup_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()

def sync_parent(path):
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_manifest(path, doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',prefix=path.name+'.',suffix='.writing',dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
        json.dump(doc,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.replace(tmp,path);sync_parent(path)
    finally:tmp.unlink(missing_ok=True)

def copy_verified(src,dst,entry):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',prefix=dst.name+'.',suffix='.copying',dir=dst.parent,delete=False) as f:
        tmp=Path(f.name)
        with src.open('rb') as source:shutil.copyfileobj(source,f,2**20)
        f.flush();os.fsync(f.fileno())
    try:
        assert tmp.is_file() and not tmp.is_symlink()
        assert tmp.stat().st_size==entry['bytes'] and sha(tmp)==entry['sha256']
        os.replace(tmp,dst);sync_parent(dst)
    finally:tmp.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    names=subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    paths=[ROOT/p for p in names if p and '__pycache__' not in Path(p).parts]
    assert all(p.is_relative_to(DATA) for p in paths), 'Ignored data outside this experiment; archive separately before cleanup'
    assert all(p.is_file() and not p.is_symlink() for p in paths)
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    if MANIFEST.exists():
        doc=json.loads(MANIFEST.read_text());known={x['source'] for x in doc['files']}
        assert all(str(p) in known for p in paths),'New ignored data after archive; inspect before changing the manifest'
    else:
        entries=[dict(source=str(p),backup=str(DEST/p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
        doc=dict(source_root=str(ROOT),backup_root=str(DEST),files=entries,total_bytes=sum(x['bytes'] for x in entries),
                 created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources_removed=False)
    assert doc['total_bytes']<6*1024**3, 'Registered raw-data budget exceeded'
    print(json.dumps({'files':len(doc['files']),'bytes':doc['total_bytes'],'apply':a.apply}))
    if not a.apply:return
    for e in doc['files']:
        src,dst=Path(e['source']),Path(e['backup'])
        assert src.is_relative_to(DATA) and dst == DEST/src.relative_to(ROOT)
        assert str(src.relative_to(ROOT)) not in tracked, 'Refusing to remove tracked file'
        assert not any(q.is_symlink() for q in (src,*src.parents,dst,*dst.parents)), 'Refusing symlink path'
        assert not src.exists() or src.is_file()
        assert not dst.exists() or dst.is_file()
        if not dst.exists():
            assert src.exists() and src.stat().st_size==e['bytes'] and sha(src)==e['sha256']
            copy_verified(src,dst,e)
        assert dst.stat().st_size==e['bytes'] and sha(dst)==e['sha256']
        if src.exists():assert src.stat().st_size==e['bytes'] and sha(src)==e['sha256']
    atomic_manifest(MANIFEST,doc)
    atomic_manifest(DEST/'backup_manifest.json',doc)
    # All backups are verified and the full path map is durable before removing any source.
    for e in doc['files']:
        src=Path(e['source'])
        if src.exists():src.unlink()
    doc.update(sources_removed=True,all_backups_sha256_verified=True)
    atomic_manifest(MANIFEST,doc)
    atomic_manifest(DEST/'backup_manifest.json',doc)
    print(json.dumps({'complete':True,'manifest':str(MANIFEST)}))

if __name__=='__main__':main()
