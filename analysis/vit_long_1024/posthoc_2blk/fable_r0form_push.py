"""崩れの間の「共通の押し」と予測の偏り（Fable エージェント A、依頼 4 の仮説 9 の検証、1029）。

入力: fable_replay_*_pbar.json / _vectors.npz（50 更新ごとに Σ(p−y)/N、予測クラスの最大割合、共通の押し −Σ Wᵀ(p−y)/N、
head 特徴の平均 mean_h、m2、FFN0 平均 patch 出力、注意 1 平均出力、窓ごとの体の歩幅の和）。
各 50 更新窓で: ‖p̄−ȳ‖、予測の最大割合、共通の押しのノルム、cos(押し, m2)・cos(押し, mean_h)・cos(押し, FFN0 平均 patch 出力)、
体の歩幅の和（窓の 2 つの 25 更新記録の和）、m2 と FFN0 出力の log 増分。窓の種類（崩れ・上昇・平ら）ごとに平均し、
cos(押し, m2) と増分の Spearman を出す。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_push.py
"""
import json
import math

import numpy as np

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import ROOT
from analysis.vit_long_1024.posthoc_2blk.fable_r0form_summary import spearmanr

OUT = ROOT / 'posthoc_r0_form'


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    result = {}
    lines = ['## 崩れの間の共通の押し Σ Wᵀ(p−y) と予測の偏り（50 更新窓、_pbar 走）\n',
             '共通の押し = −Σ_i Wᵀ(p_i − y_i)/N（head 特徴への降下の向きの、画像に依らない成分）。ラベルが釣り合い予測も偏らなければ 0 に近い。\n',
             '| 走 | 窓の種類 | 窓数 | ‖p̄−ȳ‖ | 予測の最大割合 | 押しのノルム | cos(押し, m2) | cos(押し, mean_h) | cos(押し, FFN0 平均) | 体の歩幅の和 | Δlog m2 | Δlog FFN0 patch rms | Δlog FFN0 平均出力 | FFN0 開口の変化 | 最終 FFN の死の変化 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for f in sorted(OUT.glob('fable_replay_*_pbar.json')):
        d = json.loads(f.read_text())
        if d['steps'] < 5000:
            continue
        name = f'{d["arm"]}_t{d["task_end"]}'
        z = np.load(f.with_name(f.stem + '_vectors.npz'))
        steps = z['steps'].tolist()
        rec = {r['step']: r for r in d['records']}
        sd = {s['step']: s for s in d['scale_derivative']}
        rows = []
        for k in range(len(steps) - 1):
            t0, t1 = steps[k], steps[k + 1]
            a0, a1 = rec[t0]['acc'], rec[t1]['acc']
            kind = 'crash' if a1 - a0 <= -.1 else 'rise' if a1 - a0 >= .1 else 'flat'
            push = z['common_push'][k].astype(np.float64)
            # 窓の体の歩幅の和: 記録は 25 更新ごとなので t0+25 と t1 の 2 つの窓を足す（細かい記録の窓も含めて t0 < step ≤ t1 の記録を全部足す）
            bsum = sum(r.get('win_body_step_sum', 0.) for s, r in rec.items() if t0 < s <= t1)
            hsum = sum(r.get('win_head_step_sum', 0.) for s, r in rec.items() if t0 < s <= t1)
            rows.append(dict(step=t0, kind=kind, dacc=a1 - a0, pbar_norm=sd[t0]['pbar_minus_ybar_norm'], pred_max=sd[t0]['pred_max_frac'],
                             push_norm=sd[t0]['common_push_norm'], cos_m2=cos(push, z['m2'][k]), cos_mean_h=cos(push, z['mean_h'][k]),
                             cos_ffn0=cos(push, z['ffn0_patch'][k]), cos_m2_cls=cos(push, z['m2_cls'][k]), body_step=bsum, head_step=hsum,
                             dlog_m2=math.log(rec[t1]['m2_mean_norm'] / rec[t0]['m2_mean_norm']),
                             dlog_ffn0=math.log(rec[t1]['ffn0_patch_rms'] / rec[t0]['ffn0_patch_rms']),
                             dlog_ffn0_mean=math.log(rec[t1]['ffn0_patch_mean_norm'] / rec[t0]['ffn0_patch_mean_norm']),
                             d_open=rec[t1]['ffn0_open_patch'] - rec[t0]['ffn0_open_patch'],
                             d_dead=rec[t1]['z1_dead'] - rec[t0]['z1_dead'],
                             loss=rec[t0]['loss'], acc=a0))
        agg = {}
        for kind in ['crash', 'rise', 'flat', 'all']:
            sel = [r for r in rows if kind == 'all' or r['kind'] == kind]
            if not sel:
                agg[kind] = dict(n=0)
                continue
            m = {key: float(np.mean([r[key] for r in sel])) for key in ['pbar_norm', 'pred_max', 'push_norm', 'cos_m2', 'cos_mean_h', 'cos_ffn0', 'cos_m2_cls', 'body_step', 'head_step', 'dlog_m2', 'dlog_ffn0', 'dlog_ffn0_mean', 'd_open', 'd_dead']}
            m['n'] = len(sel)
            agg[kind] = m
            lines.append(f'| {name} | {kind} | {m["n"]} | {m["pbar_norm"]:.3f} | {m["pred_max"]:.2f} | {m["push_norm"]:.3f} | {m["cos_m2"]:+.2f} | {m["cos_mean_h"]:+.2f} | {m["cos_ffn0"]:+.2f} | {m["body_step"]:.2f} | {m["dlog_m2"]:+.3f} | {m["dlog_ffn0"]:+.3f} | {m["dlog_ffn0_mean"]:+.3f} | {m["d_open"]:+.4f} | {m["d_dead"]:+.3f} |')
        corr = {}
        for xkey in ['cos_m2', 'push_norm', 'body_step', 'pbar_norm', 'pred_max', 'd_open']:
            for ykey in ['dlog_m2', 'dlog_ffn0', 'dlog_ffn0_mean']:
                rho, p = spearmanr([r[xkey] for r in rows], [r[ykey] for r in rows])
                corr[f'{xkey}~{ykey}'] = dict(rho=rho, p=p)
        rho, p = spearmanr([r['body_step'] for r in rows], [r['d_open'] for r in rows])
        corr['body_step~d_open'] = dict(rho=rho, p=p)
        # 押しの半径成分 × 押しの大きさ（m2 方向の押しの量）と増分
        xs = [r['cos_m2'] * r['push_norm'] for r in rows]
        for ykey in ['dlog_m2', 'dlog_ffn0']:
            rho, p = spearmanr(xs, [r[ykey] for r in rows])
            corr[f'radial_push~{ykey}'] = dict(rho=rho, p=p)
        result[name] = dict(rows=rows, agg=agg, corr=corr)
        lines.append(f'\n{name} の Spearman（全 {len(rows)} 窓）: ' + '、'.join(f'{k}: ρ {v["rho"]:+.2f}（p {v["p"]:.3f}）' for k, v in corr.items()) + '\n')
    (OUT / 'fable_push.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / 'fable_push_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
