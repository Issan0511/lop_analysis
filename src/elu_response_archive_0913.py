#!/usr/bin/env python3
"""Copy raw result artifacts to a separate same-host directory and verify SHA256."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/elu_response_anchor_0913"
BACK=Path("/home/issan/Projects/obsidian-research-data/elu_response_anchor_0913")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    prov=json.loads((OUT/"provenance.json").read_text())
    if prov.get("status")!="COMPLETE":raise RuntimeError("complete scientific run required")
    sources=sorted(p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".pt",".npz"))
    if not sources:raise RuntimeError("no raw files")
    files=[]
    for p in sources:
        rel=p.relative_to(ROOT);dst=BACK/"files"/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        expected=sha(p)
        if not dst.exists():shutil.copy2(p,dst)
        if sha(dst)!=expected:raise RuntimeError(f"backup mismatch (not overwritten): {dst}")
        files.append(dict(repo_path=str(rel),backup_path=str(dst),sha256=expected,bytes=p.stat().st_size))
    manifest=dict(status="SHA256_COPY_VERIFIED",scope="Same-host separate directory; binaries are excluded from Git",backup_root=str(BACK),file_count=len(files),total_bytes=sum(x["bytes"] for x in files),files=files)
    for p in (OUT/"raw_manifest.json",BACK/"manifest.json"):p.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="files"},indent=2))
if __name__=="__main__":main()
