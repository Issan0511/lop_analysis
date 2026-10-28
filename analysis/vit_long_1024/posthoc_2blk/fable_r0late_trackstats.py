"""fable_r0late_track の毎更新の記録を集計する（Fable エージェント B、1029）。

一貫性（項の和のノルム / ノルムの和）、群の項どうしの cos、急落の更新での Adam の歩幅・batch の損失、跳ねの統計。
    python fable_r0late_trackstats.py <tag>
"""
import json
import sys
import statistics as st
from pathlib import Path

import numpy as np

OUT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0_late')


def main(tag):
    d = json.load((OUT / f'{tag}.json').open())
    q = d['per_step']
    rec = {r['step']: r for r in d['records']}
    steps = np.array([x['step'] for x in q])
    body = np.array([x['body'] for x in q])
    out = {'tag': tag, 'start': d['start_step'], 'n_steps': len(q)}
    # 跳ね: 体の項の分布
    out['body_rms'] = dict(median=float(np.median(body)), q90=float(np.quantile(body, .9)), q99=float(np.quantile(body, .99)),
                           max=float(body.max()), max_step=int(steps[body.argmax()]))
    # 急落（評価点間で −0.1 以上）とその 25 更新の中の歩幅
    evals = sorted(rec)
    drops = [(s1, rec[s0]['acc'] - rec[s1]['acc']) for s0, s1 in zip(evals, evals[1:]) if rec[s0]['acc'] - rec[s1]['acc'] >= .1]
    rows = []
    for s1, fall in drops:
        win = [x for x in q if s1 - 25 < x['step'] <= s1]
        if not win:
            continue
        peak = max(win, key=lambda x: x['body'])
        rows.append(dict(step=s1, fall=round(fall, 3), body_max=round(peak['body'], 3), at=peak['step'],
                         dtheta_l2={g: peak['dtheta_l2'][g] for g in ('block0', 'attn1', 'ffn1')},
                         dtheta_ratio={g: round(peak['dtheta_l2_over_eta_sqrt_n'][g], 4) for g in ('block0', 'attn1', 'ffn1')},
                         grad_l2={g: round(peak['grad_l2'][g], 3) for g in ('block0', 'attn1', 'ffn1')},
                         batch_loss=round(peak['batch_loss'], 3), batch_loss_max=round(peak['batch_loss_max'], 2),
                         top1_share=round(peak['batch_top1_share'], 3), batch_wrong=peak['batch_wrong'],
                         margin_q10_before=round(peak['margin_q10_before'], 2), flip=round(peak['flip_frac'], 3),
                         dmargin_max=round(peak['dmargin_abs_max'], 2)))
    out['drops'] = rows
    # 平常の歩幅（中央値）
    typ = {}
    for key in ('dtheta_l2', 'dtheta_l2_over_eta_sqrt_n', 'grad_l2'):
        typ[key] = {g: dict(median=float(np.median([x[key][g] for x in q])), q99=float(np.quantile([x[key][g] for x in q], .99)),
                            max=float(max(x[key][g] for x in q))) for g in ('block0', 'attn1', 'ffn1', 'head')}
    typ['batch_loss'] = dict(median=float(np.median([x['batch_loss'] for x in q])), q99=float(np.quantile([x['batch_loss'] for x in q], .99)))
    typ['batch_loss_max'] = dict(median=float(np.median([x['batch_loss_max'] for x in q])), q99=float(np.quantile([x['batch_loss_max'] for x in q], .99)))
    typ['top1_share'] = dict(median=float(np.median([x['batch_top1_share'] for x in q])), q99=float(np.quantile([x['batch_top1_share'] for x in q], .99)))
    out['typical'] = typ
    # 跳ねと歩幅・batch の相関（体の項の上位 1% と残り）
    thr = np.quantile(body, .99)
    hi = [x for x in q if x['body'] >= thr]
    lo = [x for x in q if x['body'] < thr]
    out['spike_vs_rest'] = {k: dict(spike=float(np.median([f(x) for x in hi])), rest=float(np.median([f(x) for x in lo])))
                            for k, f in (('dtheta_l2_block0', lambda x: x['dtheta_l2']['block0']), ('dtheta_l2_attn1', lambda x: x['dtheta_l2']['attn1']),
                                         ('dtheta_l2_ffn1', lambda x: x['dtheta_l2']['ffn1']), ('grad_l2_block0', lambda x: x['grad_l2']['block0']),
                                         ('grad_l2_attn1', lambda x: x['grad_l2']['attn1']), ('batch_loss', lambda x: x['batch_loss']),
                                         ('batch_loss_max', lambda x: x['batch_loss_max']), ('top1_share', lambda x: x['batch_top1_share']),
                                         ('batch_wrong', lambda x: x['batch_wrong']), ('upstream_rms', lambda x: x['upstream_rms']),
                                         ('cos_ffn1_upstream', lambda x: x['cos_ffn1_upstream']), ('flip', lambda x: x['flip_frac']))}
    # 群の cos と打ち消し
    out['group_cos_median'] = {k: float(np.median([x['group_cos'][k] for x in q])) for k in q[0]['group_cos']}
    out['cos_ffn1_upstream'] = dict(median=float(np.median([x['cos_ffn1_upstream'] for x in q])),
                                    frac_negative=float(np.mean([x['cos_ffn1_upstream'] < 0 for x in q])))
    out['cancel'] = dict(upstream_rms_median=float(np.median([x['upstream_rms'] for x in q])),
                         with_ffn1_rms_median=float(np.median([x['upstream_plus_ffn1_rms'] for x in q])),
                         ffn1_rms_median=float(np.median([x['groups_rms']['ffn1'] for x in q])))
    # 一貫性の時系列（分岐から n 更新後）
    coh = []
    for n in (25, 100, 400, 1000, len(q)):
        x = q[min(n, len(q)) - 1]
        coh.append(dict(n=x['cum_n'], coherence={k: round(v, 4) for k, v in x['coherence'].items()},
                        cum_margin_median={k: round(v, 3) for k, v in x['cum_margin_median'].items()},
                        random_walk=round(1 / x['cum_n'] ** .5, 4)))
    out['coherence'] = coh
    (OUT / f'{tag}_stats.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
