"""既存入力の SHA256 を記録・照合する（Fable エージェント A、依頼 4 の A 形成則、1028）。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_hashes.py record
    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_hashes.py verify
"""
import json
import sys
from pathlib import Path

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import REPO, ROOT, sha

OUT = ROOT / 'posthoc_r0_form'


def inputs():
    files = []
    for arm in ['R0', 'S']:
        files += sorted((ROOT / 'runs' / arm / 'seed10252010/ckpt').glob('*.pt'))
        files += sorted((ROOT / 'runs' / arm / 'seed10252010').glob('*.csv'))
    files += [REPO / 'analysis/vit_long_1024/production_2blk_rlfix.json',
              REPO / 'analysis/vit_long_1024/common.py',
              REPO / 'analysis/vit_long_1024/posthoc_2blk/codex_r0_common.py',
              REPO / 'analysis/vit_long_1024/posthoc_2blk/codex_r0_replay.py',
              REPO / 'analysis/joudaki_vit_battle_0919/data.py',
              REPO / 'analysis/joudaki_vit_battle_0919/model.py']
    return [f for f in files if f.exists()]


def main(mode):
    dest = OUT / 'fable_input_hashes.json'
    if mode == 'record':
        table = {str(f): sha(f) for f in inputs()}
        dest.write_text(json.dumps(table, indent=1))
        print(f'recorded {len(table)} files')
    else:
        old = json.loads(dest.read_text())
        changed = [f for f, h in old.items() if not Path(f).exists() or sha(f) != h]
        (OUT / 'fable_input_verify.json').write_text(json.dumps(dict(checked=len(old), changed=changed), indent=1))
        print(f'checked {len(old)} files, changed {changed}')


if __name__ == '__main__':
    main(sys.argv[1])
