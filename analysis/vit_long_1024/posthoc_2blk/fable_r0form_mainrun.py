"""本走（compile）の課題ごとの記録から、共通成分の 1 課題の成長と当てはめの長さの関係を出す（Fable エージェント A、依頼 4、1028）。

入力: runs/{R0,S}/seed10252010/per_task.csv（T90・課題末の暗記率など）と layer_stats.csv（課題末の固定 probe の
FFN 入力の共通成分のノルム in_mean_norm、幅 sigma_med、死 dead005、層 1 = FFN0、層 2 = 最終 FFN）。
出力: posthoc_r0_form/fable_mainrun.json。課題 k の log 増分 Δ_k = log in_mean_norm(k) − log in_mean_norm(k−1) と、
課題 k の T90（未到達は 5,000 扱い）・暗記率末・前課題の T90 との Spearman。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_mainrun.py
"""
import csv
import json
import math

import numpy as np

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import ROOT
from analysis.vit_long_1024.posthoc_2blk.fable_r0form_summary import spearmanr

OUT = ROOT / 'posthoc_r0_form'


def read(arm):
    base = ROOT / 'runs' / arm / 'seed10252010'
    per = {int(r['task']): r for r in csv.DictReader((base / 'per_task.csv').open())}
    lay = {}
    for r in csv.DictReader((base / 'layer_stats.csv').open()):
        lay[(int(r['task']), int(r['layer']))] = r
    z = np.load(base / 'intask.npz')
    # 課題の中の 25 更新ごとの暗記率（201 点）。未当てはめの積分 = Σ(1 − 暗記率)、未当てはめの点数 = 暗記率 < 0.9 の点数
    acc = z['mem_correct'] / 2500.
    # 崩れの数 = 50 更新（記録 2 点）で暗記率が 0.1 以上落ちた窓の数（回し直しの「崩れ窓」と同じ定義）
    unfit = {int(t): dict(misfit_integral=float((1 - acc[i]).sum()), unfit_points=int((acc[i] < .9).sum()),
                          misfit_integral_after200=float((1 - acc[i, 8:]).sum()),
                          crash_windows=int(((acc[i, 2:] - acc[i, :-2]) <= -.1).sum()),
                          crash_depth=float(np.clip(-(acc[i, 2:] - acc[i, :-2]), 0, None)[(acc[i, 2:] - acc[i, :-2]) <= -.1].sum()))
             for i, t in enumerate(z['tasks'])}
    return per, lay, unfit


def main():
    result = {}
    lines = []
    for arm in ['R0', 'S']:
        per, lay, unfit = read(arm)
        tasks = sorted(per)
        rows = []
        for k in tasks:
            t90 = float(per[k]['T90'])
            rows.append(dict(task=k, T90=t90 if t90 <= 5000 else None, T90_capped=min(t90, 5000.), acc_end=float(per[k]['mem_acc_end']),
                             misfit_integral=unfit[k]['misfit_integral'], unfit_points=unfit[k]['unfit_points'],
                             misfit_after200=unfit[k]['misfit_integral_after200'],
                             crash_windows=unfit[k]['crash_windows'], crash_depth=unfit[k]['crash_depth'],
                             loss_start=float(per[k]['mem_loss_start']), head_norm_end=float(per[k]['head_norm_end']),
                             m1=float(lay[(k, 1)]['in_mean_norm']), m2=float(lay[(k, 2)]['in_mean_norm']),
                             sigma1=float(lay[(k, 1)]['sigma_med']), sigma2=float(lay[(k, 2)]['sigma_med']),
                             dead1=float(lay[(k, 1)]['dead005']), dead2=float(lay[(k, 2)]['dead005']),
                             mu2=float(lay[(k, 2)]['mu_med']), up2=float(lay[(k, 2)]['up_norm_med'])))
        for i in range(1, len(rows)):
            for key in ['m1', 'm2', 'sigma1', 'sigma2']:
                rows[i][f'dlog_{key}'] = math.log(rows[i][key]) - math.log(rows[i - 1][key])
        rows[0].update({f'dlog_{k}': None for k in ['m1', 'm2', 'sigma1', 'sigma2']})
        stats = {}
        for span, lo, hi in (('all', 2, 100), ('t2_40', 2, 40), ('t7_40', 7, 40), ('t41_100', 41, 100), ('t2_25', 2, 25)):
            sel = [r for r in rows if lo <= r['task'] <= hi]
            for key in ['dlog_m2', 'dlog_m1', 'dlog_sigma2']:
                d = [r[key] for r in sel]
                for xname in ['T90_capped', 'acc_end', 'loss_start', 'misfit_integral', 'unfit_points', 'misfit_after200', 'crash_windows', 'crash_depth']:
                    rho, p = spearmanr([r[xname] for r in sel], d)
                    stats[f'{span}:{key}~{xname}'] = dict(rho=rho, p=p, n=len(sel))
                # 前課題の T90（成長が前の課題の長さに従うなら）
                prev = [[rr for rr in rows if rr['task'] == r['task'] - 1][0]['T90_capped'] for r in sel]
                rho, p = spearmanr(prev, d)
                stats[f'{span}:{key}~prev_T90'] = dict(rho=rho, p=p, n=len(sel))
                stats[f'{span}:{key}:mean'] = float(np.mean(d))
                stats[f'{span}:{key}:mean_reached'] = float(np.mean([v for v, r in zip(d, sel) if r['T90'] is not None])) if any(r['T90'] is not None for r in sel) else None
                stats[f'{span}:{key}:mean_unreached'] = float(np.mean([v for v, r in zip(d, sel) if r['T90'] is None])) if any(r['T90'] is None for r in sel) else None
                stats[f'{span}:{key}:n_unreached'] = sum(r['T90'] is None for r in sel)
        result[arm] = dict(rows=rows, stats=stats)
        lines.append(f'\n### {arm}: 本走の課題ごとの m2（最終 FFN の入力の共通成分）の log 増分と当てはめ\n')
        lines.append('| 課題 | T90 | 未当てはめの積分 Σ(1−暗記率)（25 更新ごと 201 点） | 暗記率末 | 切替時の損失 | m2 | Δlog m2 | m1 | Δlog m1 | σ2 | 死 2 | head ノルム末 |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in rows:
            if r['task'] <= 12 or r['task'] % 5 == 0 or 26 <= r['task'] <= 34:
                d2 = f'{r["dlog_m2"]:+.2f}' if r['dlog_m2'] is not None else '—'
                d1 = f'{r["dlog_m1"]:+.2f}' if r['dlog_m1'] is not None else '—'
                lines.append(f'| {r["task"]} | {r["T90"] if r["T90"] is not None else "未到達"} | {r["misfit_integral"]:.1f} | {r["acc_end"]:.2f} | {r["loss_start"]:.2f} | {r["m2"]:,.0f} | '
                             f'{d2} | {r["m1"]:,.1f} | {d1} | {r["sigma2"]:,.1f} | {r["dead2"]:.2f} | {r["head_norm_end"]:.2f} |')
        lines.append(f'\n{arm} の Spearman（課題 k の Δlog m2 と）:\n')
        lines.append('| 区間 | 相手 | ρ | 並べ替え p | n | Δlog m2 の平均（到達 / 未到達、未到達の数） |')
        lines.append('|---|---|---|---|---|---|')
        for span in ['t2_40', 't7_40', 't41_100', 'all']:
            for xname in ['T90_capped', 'prev_T90', 'misfit_integral', 'crash_windows', 'crash_depth', 'acc_end', 'loss_start']:
                s = stats[f'{span}:dlog_m2~{xname}']
                lines.append(f'| {span} | {xname} | {s["rho"]:+.2f} | {s["p"]:.3f} | {s["n"]} | '
                             f'{stats[f"{span}:dlog_m2:mean_reached"] if stats[f"{span}:dlog_m2:mean_reached"] is None else round(stats[f"{span}:dlog_m2:mean_reached"], 3)} / '
                             f'{stats[f"{span}:dlog_m2:mean_unreached"] if stats[f"{span}:dlog_m2:mean_unreached"] is None else round(stats[f"{span}:dlog_m2:mean_unreached"], 3)}、{stats[f"{span}:dlog_m2:n_unreached"]} |')
    (OUT / 'fable_mainrun.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / 'fable_mainrun_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
