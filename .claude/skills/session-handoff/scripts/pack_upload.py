#!/usr/bin/env python3
"""pack_upload.py — 引き継ぎ用にパスを tar で束ね、sha256 を取り、ギガファイル便へ上げる。

    python3 pack_upload.py --out OUTDIR [--no-upload] [--resume] ITEM ...

ITEM の書式:
    name=/abs/path[,/abs/path2,...]   → OUTDIR/name.tar（各 path は「親ディレクトリ基準」で入るので、
                                         展開すると path.name がトップレベルに現れる）
    name=@/abs/file                   → 既存ファイルをそのまま上げる（tar しない）
    name=gz:/abs/path[,...]           → OUTDIR/name.tar.gz（小さい束・テキスト向け）
    name=/abs/path[,...]>results/run  → 末尾の `>dir` は受け側の展開先（manifest の extract_to・repo 基準の相対パス）。
                                         tar のトップが run 名でないとき（logs_tail/ など）に使う

出力: OUTDIR/manifest.json（1 項目 1 レコード。built/sha256/size/url/delete_key）と、
標準出力に Markdown 表。アップロードは **1 ファイル 1 プロセス**（gfile の cookie jar を分ける）。
`--resume` は既に built/uploaded の項目を飛ばす。ネットワーク失敗は 2 回まで再試行。

npz は既に deflate 済みなので、データは無圧縮 tar にする（zstd で 2% しか縮まない）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tar(dst: Path, paths: list[Path], gzip: bool) -> None:
    cmd = ["tar", "-cf" if not gzip else "-czf", str(dst)]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        cmd += ["-C", str(p.parent), p.name]
    subprocess.run(cmd, check=True)


UPLOAD_SNIPPET = r"""
import json, sys, io, contextlib
from gfile import GFile
g = GFile(sys.argv[1], progress=False, mute=True, verify=True)
with contextlib.redirect_stdout(io.StringIO()):
    g.upload()
    ok = bool(g.data and "url" in g.data)
    url = g.get_download_page() if ok else None
print(json.dumps({"ok": bool(url), "url": url, "delete_key": (g.data or {}).get("delkey"),
                  "finished_at": (g.data or {}).get("finished_at")}))
"""


def upload(path: Path, tries: int = 3) -> dict:
    last = {}
    for k in range(tries):
        r = subprocess.run([sys.executable, "-c", UPLOAD_SNIPPET, str(path)],
                           capture_output=True, text=True)
        try:
            last = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            last = {"ok": False, "error": (r.stderr or r.stdout)[-800:]}
        if last.get("ok"):
            return last
        time.sleep(30 * (k + 1))
    return last


def parse_item(s: str) -> tuple[str, list[Path], str, str | None]:
    name, _, rhs = s.partition("=")
    if not name or not rhs:
        raise SystemExit(f"bad ITEM: {s}")
    rhs, _, extract_to = rhs.partition(">")
    extract_to = extract_to or None
    if rhs.startswith("@"):
        return name, [Path(rhs[1:]).expanduser().resolve()], "file", extract_to
    kind = "tar"
    if rhs.startswith("gz:"):
        kind, rhs = "tgz", rhs[3:]
    return name, [Path(p).expanduser().resolve() for p in rhs.split(",")], kind, extract_to


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if (a.resume and mpath.exists()) else {"items": {}}

    def save() -> None:
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    for item in a.items:
        name, paths, kind, extract_to = parse_item(item)
        rec = manifest["items"].get(name, {})
        if kind == "file":
            dst = paths[0]
        else:
            dst = out / (f"{name}.tar" if kind == "tar" else f"{name}.tar.gz")
            if not (a.resume and rec.get("built") and dst.exists() and dst.stat().st_size == rec.get("size_bytes")):
                print(f"[pack] {name} ← {[str(p) for p in paths]}", file=sys.stderr, flush=True)
                build_tar(dst, paths, gzip=(kind == "tgz"))
                rec = {}
        if not rec.get("sha256"):
            rec.update(file=str(dst), size_bytes=dst.stat().st_size, sha256=sha256_of(dst), built=True,
                       sources=[str(p) for p in paths], kind=kind)
            manifest["items"][name] = rec
            save()
        if extract_to:
            rec["extract_to"] = extract_to
            manifest["items"][name] = rec
            save()
        if a.no_upload or rec.get("url"):
            continue
        print(f"[upload] {name} ({rec['size_bytes']/1e9:.2f} GB)", file=sys.stderr, flush=True)
        t0 = time.time()
        r = upload(dst)
        rec.update(url=r.get("url"), delete_key=r.get("delete_key"), uploaded_at=r.get("finished_at"),
                   upload_sec=round(time.time() - t0), upload_error=None if r.get("ok") else r)
        manifest["items"][name] = rec
        save()
        print(f"[upload] {name} → {rec.get('url')} ({rec['upload_sec']} s)", file=sys.stderr, flush=True)

    print("| 項目 | 中身 | サイズ | sha256 (先頭 12) | URL |")
    print("|---|---|---|---|---|")
    for name, rec in manifest["items"].items():
        src = ", ".join(Path(s).name for s in rec.get("sources", []))
        print(f"| `{name}` | {src} | {rec['size_bytes']/1e9:.2f} GB | `{rec['sha256'][:12]}` | {rec.get('url') or '（未）'} |")


if __name__ == "__main__":
    main()
