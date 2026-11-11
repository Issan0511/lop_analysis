#!/usr/bin/env python3
"""Archive verified experiment state before worktree removal."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN='leaky_order_1111';OUT=ROOT/'results'/RUN
DEST=Path('/home/issan/Projects/obsidian-research-data')/RUN
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['probes']==75
    assert json.loads((OUT/'analysis_checks.json').read_text())['all_pass']
    manifest=[];to_remove=[]
    def save(src,rel,category,remove=False):
        dst=DEST/rel;digest=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==digest,(src,dst)
        else:shutil.copy2(src,dst)
        assert sha(dst)==digest
        manifest.append(dict(source=str(src),source_rel=str(rel),backup=str(dst),bytes=src.stat().st_size,sha256=digest,category=category))
        if remove:to_remove.append(src)
    for src in sorted(OUT.rglob('*')):
        if src.is_file() and src.suffix in ['.pt','.npz','.log']:save(src,src.relative_to(ROOT),'main_raw',True)
    for name in ['leaky_order_checks','leaky_order_old_host','leaky_order_smoke_prefix']:
        base=Path('/tmp')/name
        assert base.exists(),base
        for src in sorted(base.rglob('*')):
            if src.is_file():save(src,Path('validation')/name/src.relative_to(base),'validation_raw')
    provenance=json.loads(next((OUT/'probes').glob('*/*/*/provenance.json')).read_text())
    result=dict(run=RUN,main_training_git=provenance['git_hash'],all_sha256_verified=True,files=manifest)
    (OUT/'backup_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    for src in to_remove:src.unlink()
    print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest),removed_verified_worktree_raw=len(to_remove),training_git=provenance['git_hash'])))
if __name__=='__main__':main()
