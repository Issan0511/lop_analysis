"""pm_rl_samebox_1108: write results/pm_rl_samebox_1108/backup_manifest.json (CLAUDE.md §4).

The raw data never lived inside the worktree: run.py and ledger.py wrote straight to
~/Projects/obsidian-research-data/pm_rl_samebox_1108/, so each file's source and backup
are the same path.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/issan/Projects/obsidian-research-data/pm_rl_samebox_1108")
OUT = REPO / "results" / "pm_rl_samebox_1108" / "backup_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = []
    total = 0
    for p in sorted(DATA.rglob("*")):
        if p.is_file():
            n = p.stat().st_size
            total += n
            files.append({"source": str(p), "backup": str(p), "bytes": n, "sha256": sha256(p)})
    OUT.write_text(json.dumps({
        "experiment": "pm_rl_samebox_1108",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "生データ（課題末の W1・b1・順列・ラベル・固定 1,200 枚の添字・課題ごとの online・新品・ログ・smoke）は"
                "走のときから obsidian-research-data/pm_rl_samebox_1108/ に直接書いた。"
                "worktree の中に git 外のファイルは __pycache__ 以外ない。",
        "files": files, "total_bytes": total, "n_files": len(files)}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}: {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
