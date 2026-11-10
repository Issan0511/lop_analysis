"""追補 E.3 の記録（RMSNorm の装填の押し、PE6）と追補 G.4 の記録（行と特徴の cos²）を、保存した課題末すべてで出す（Fable、1111）。

道具は追補 D の J5（j5_rlfix.py）の j5_state をそのまま seed を替えて呼ぶ（量の定義は同じ。最後の正規化が RMSNorm の腕は
j5_rlfix.py の 1111 の追記のとおり、平均を引かない）。実際の 1 歩（GPU）は回さない。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_e.json \
        python -m analysis.vit_long_1024.posthoc_2blk.fable_eg_j5 SRMS 10252010 10252011 10252012 10252013 10252014

出力は <raw>/posthoc_j5/j5_<腕>_seed<seed>.json（既にあれば飛ばす）。追補 D の posthoc_j5/j5_<腕>.json（seed 10252010）とは別のファイル。
"""
import json
import os
import sys
import time

os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_e.json')
import torch

from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk import j5_rlfix as J

OUT = J.RUNS.parent / 'posthoc_j5'


def main(arm, seeds):
    torch.set_num_threads(16)
    C.set_determinism(16)
    source = C.image_source('real')
    for seed in seeds:
        out = OUT / f'j5_{arm}_seed{seed}.json'
        if out.exists():
            print('あるので飛ばす:', out.name)
            continue
        J.SEED = seed                                   # j5_rlfix の関数はこのモジュール変数を見る
        data1, _ = C.task_dataset(source, seed, 1, 10, 250, augment=False)
        x = C.eval_tensor(data1.images, 'cpu').double()  # 全課題で同じ 2,500 枚
        ck = J.RUNS / arm / f'seed{seed}' / 'ckpt'
        tasks = sorted(int(p.stem.split('_t')[1]) for p in ck.glob('model_t*.pt'))
        tasks = [t for t in tasks if t >= 1]
        results, t0 = [], time.monotonic()
        for t in tasks:
            res, _ = J.j5_state(arm, t, x, C.task_labels(2500, seed, t, 10))
            results.append(res)
            print(f"{arm} {seed} t{t:3d} κ {res['kappa']:+.4f} cos² {res['cos2']:.3f} 生 {res['alive']:4d}/{res['n_units']}"
                  f" | 沈める側 装填 {res['s_load']:.3f}（直通 {res['s_load_direct']:.3f}・分散微分 {res['s_load_var']:.3f}）"
                  f" 全体 {res['s_full']:.3f} | {time.monotonic() - t0:.0f}s", flush=True)
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(arm=arm, seed=seed, tasks=tasks, results=results), indent=1))
        tmp.rename(out)


if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2:]])
