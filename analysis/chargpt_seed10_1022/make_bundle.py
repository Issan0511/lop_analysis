#!/usr/bin/env python3
"""chargpt_seed10_1022: byakko に渡す checkpoint の束（tar、非圧縮）と manifest を white-san で作る。

使い方: python make_bundle.py <claude/chargpt_cl_1019 の checkout> <出力ディレクトリ>
束の中のパスは checkout の直下からの相対（results/chargpt_cl_1019/...）。checkout は読むだけ。
中身は手順書 specs/chargpt_seed10_1022_runbook.md §2 の表のとおり。
"""
import hashlib, json, socket, subprocess, sys, tarfile, time
from pathlib import Path

CG = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
R = CG / "results/chargpt_cl_1019"

files = []
for arm in ("C3", "ReLUcap"):
    for s in range(10):
        d = R / "M2" / arm / f"seed{s}" / "ckpts"
        files += sorted(p for p in d.iterdir() if p.suffix == ".pt" and not p.name.startswith("u100_"))
d = R / "quicklook/M2/C3/seed1000/ckpts"
files += sorted(p for p in d.iterdir()
                if p.suffix == ".pt" and not p.name.startswith("u100_") and not p.name.endswith("_adam.pt"))
d = R / "quicklook/M/C3/seed1000/ckpts"
files += [d / "task1.pt", d / "task1_adam.pt"]
files += [R / "quicklook/rho/out/C3/t2_a_chain.json"]


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


entries = []
tar_path = OUT / "chargpt_seed10_1022_bundle.tar"
with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
    for p in files:
        rel = str(p.relative_to(CG))
        entries.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p)))
        tf.add(p, arcname=rel, recursive=False)

head = subprocess.run(["git", "-C", str(CG), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %z"), host=socket.gethostname(),
    source_checkout_branch="claude/chargpt_cl_1019", source_checkout_head=head,
    n_files=len(entries), total_bytes=sum(e["bytes"] for e in entries),
    tar_name=tar_path.name, tar_bytes=tar_path.stat().st_size, tar_sha256=sha256(tar_path),
    extract_to="claude/chargpt_cl_1019 の checkout の直下（tar -xf <tar> -C $CG）",
    files=entries,
)
(OUT / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
print(f"{len(entries)} files, {manifest['total_bytes']/1e9:.2f} GB -> {tar_path} ({manifest['tar_bytes']/1e9:.2f} GB) sha256 {manifest['tar_sha256']}")
