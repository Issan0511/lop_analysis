"""追補 F の回し直しの出力（posthoc_f）の一覧と sha256 を results/.../addendum_f/backup_manifest.json に書く（Fable、1104）。
生データは最初から obsidian-research-data の下にあるので source = backup。
    python -m analysis.vit_long_1024.posthoc_2blk.fable_f_manifest
"""
import hashlib
import json
from pathlib import Path

RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
OUT = Path(__file__).resolve().parents[3] / 'results/vit_long_1024_2blk_rlfix/addendum_f/backup_manifest.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for p in sorted((RAW / 'posthoc_f').iterdir()):
        if p.is_file():
            rows.append(dict(source=str(p), backup=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    OUT.write_text(json.dumps(dict(note='追補 F の回し直し（R1〜R5・J F1・書き手の分解）の出力。生データは obsidian-research-data に置いたまま', files=rows),
                              ensure_ascii=False, indent=1))
    print(len(rows), 'files', sum(r['bytes'] for r in rows) / 2**20, 'MiB ->', OUT)


if __name__ == '__main__':
    main()
