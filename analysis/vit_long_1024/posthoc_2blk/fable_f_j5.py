"""追補 F の J F1（装填、spec F.5）を S の 5 seed で出す（Fable、1104）。

追補 D の J5 の道具 j5_rlfix.py（seed 10252010 固定）の j5_state と actual_first_step を、seed を替えて呼ぶだけで、量の定義は同じ:
  κ_t = ⟨W^c, F_t⟩/‖F_t‖²（F_t = Σ_m (e_{y_m} − 1/K) ⊗ h_m、課題 t の 2,500 枚・そのラベル、CLS 位置、LN なしは h = t）、cos²、
  補題 ⟨W^c, F_t⟩ ≥ N(log K − L_K) − Σ_c (n_c − N/K) b^c_c、
  s_load = 装填成分 vL = κF を通した押し S_j が沈める側の、生きた unit の割合（CLS 位置）、
  Adam のある課題末では課題 t+1 の最初の batch で実際に 1 歩（勾配と Adam の 1 歩が沈める側の割合、S_load との符号の一致）。
課題末は 1・10・30・60・100（判定は t10・t30: κ > 0 かつ補題成立かつ s_load ≥ 0.56、3/5）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json \
        python -m analysis.vit_long_1024.posthoc_2blk.fable_f_j5 10252010 10252011 10252012 10252013 10252014

出力は <raw>/posthoc_f/j5_S_seed<seed>.json（既にあれば飛ばす）。seed 10252010 は posthoc_j5/j5_S.json（追補 D）と同じ値になるはず。
"""
import json
import os
import sys
import time

os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_f.json')
import torch

from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk import j5_rlfix as J

TASKS = (1, 10, 30, 60, 100)
ARM = 'S'


def main(seeds):
    torch.set_num_threads(8)
    C.set_determinism(8)
    out_dir = C.RAW / 'posthoc_f'
    out_dir.mkdir(parents=True, exist_ok=True)
    source = C.image_source('real')
    for seed in seeds:
        dest = out_dir / f'j5_{ARM}_seed{seed}.json'
        if dest.exists():
            print(f'skip {dest} (exists)', flush=True)
            continue
        J.SEED = seed                                   # j5_rlfix の関数はこのモジュール変数を見る
        data1, _ = C.task_dataset(source, seed, 1, 10, 250, augment=False)
        x = C.eval_tensor(data1.images, 'cpu').double()  # 固定の 2,500 枚（追補 D.1）
        results, t0 = [], time.monotonic()
        for t in TASKS:
            labels_t = C.task_labels(2500, seed, t, 10)
            res, aux = J.j5_state(ARM, t, x, labels_t)
            try:
                res['first_step'] = J.actual_first_step(ARM, t, aux, source)
            except Exception as exc:
                res['first_step'] = dict(error=repr(exc))
            results.append(res)
            fs = res.get('first_step', {})
            print(f"{ARM} seed {seed} t{t:3d} κ {res['kappa']:.3e} cos² {res['cos2']:.3f} 補題 {'○' if res['lemma_holds'] else '×'}"
                  f" 生 {res['alive']:4d}/{res['n_units']} | s_load {res['s_load']:.3f} s_full {res['s_full']:.3f}"
                  + (f" | 1 歩: 勾配 {fs['frac_grad_sinks']:.3f} Adam {fs['frac_adam_sinks']:.3f} 一致 {fs['agree_Sload_adam']:.3f}"
                     if 'frac_grad_sinks' in fs else f" | 1 歩: {fs.get('error', '')}")
                  + f" | {time.monotonic() - t0:.0f}s", flush=True)
            dest.write_text(json.dumps(dict(arm=ARM, seed=seed, tasks=list(TASKS), results=results,
                                            git_hash=C.git_state()[0]), indent=1))


if __name__ == '__main__':
    main([int(s) for s in sys.argv[1:]] or [10252010, 10252011, 10252012, 10252013, 10252014])
