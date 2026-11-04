"""追補 F の R4（R0 の対照、spec F.4）を seed ごとに回す（Fable、1104）。

依頼 4 の道具 fable_r0late_replay.py（seed 10252010 固定、posthoc_r0_late/fable_R0_t60_hl001_{base,br_body,br_newhead1}）を、
seed と出力先を替えて同じ 3 段で呼ぶ:
  base        R0 の t60 末（重み＋Adam、head は 0 に戻す）から head lr 1/100・体 1e-4 で課題 61 を 5,000 更新。暗記率（2,500 枚）が
              初めて 0.9 に届いた評価点で model と Adam を保存する。
  br_body     その状態から体（ブロック 0・注意 1・最終 FFN）を全部凍結して 5,000 更新まで続ける。量は最後 1,000 更新の平均。
  br_newhead1 同じ状態の体に新 head（0 から、lr 1e-4、head の Adam は捨てる）を載せ、体を凍結して 5,000 更新。量は T90。
base が 5,000 更新で 0.9 に届かなければ分岐は回さず、その旨を記す（J FR0 はその seed で不成立）。
一歩ごとの分解の記録は 1 seed のときより粗く（--diag_every 100000、--window 0。学習には影響しない）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json \
        python -m analysis.vit_long_1024.posthoc_2blk.fable_f_r4 10252011 10252012 10252013 10252014

出力は <raw>/posthoc_f/r4_R0_seed<seed>_t60_{base,br_body,br_newhead1}.json（既にあれば飛ばす）。
"""
import argparse
import os
import sys
import time

os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_f.json')

from analysis.vit_long_1024.posthoc_2blk import codex_r0_common as RC
from analysis.vit_long_1024.posthoc_2blk import fable_r0late_replay as RL

TASK = 60


def stage(tag, **kw):
    dest = RL.OUT / f'{tag}.json'
    if dest.exists():
        print(f'skip {tag} (exists)', flush=True)
        return
    ns = argparse.Namespace(arm='R0', task=TASK, tag=tag, head_factor=kw.get('head_factor', .01), body_factor=1.,
                            steps=5000, save_t90=kw.get('save_t90', False), branch=kw.get('branch'),
                            freeze=kw.get('freeze', 'none'), new_head=kw.get('new_head', False),
                            diag_every=100000, window=0)
    t0 = time.monotonic()
    RL.main(ns)
    print(f'{tag} done in {time.monotonic() - t0:.0f}s', flush=True)


def main(seeds):
    for seed in seeds:
        RC.SEED = seed                                  # ckpt()・images()・load_model() が見る
        RL.SEED = seed                                  # ラベル・損失の行・batch 列が見る
        RL.OUT = RC.ROOT / 'posthoc_f'
        RL.OUT.mkdir(parents=True, exist_ok=True)
        base = f'r4_R0_seed{seed}_t{TASK}_base'
        state = RL.OUT / f'{base}_t90state.pt'
        stage(base, head_factor=.01, save_t90=True)
        if not state.exists():
            RL.progress(f'{base}: 5,000 更新で暗記率 0.9 に届かず、分岐（br_body・br_newhead1）は回さない。')
            print(f'{base}: no t90 state, branches skipped', flush=True)
            continue
        stage(f'r4_R0_seed{seed}_t{TASK}_br_body', head_factor=.01, branch=state, freeze='body')
        stage(f'r4_R0_seed{seed}_t{TASK}_br_newhead1', head_factor=1., branch=state, new_head=True)


if __name__ == '__main__':
    main([int(s) for s in sys.argv[1:]] or [10252011, 10252012, 10252013, 10252014])
