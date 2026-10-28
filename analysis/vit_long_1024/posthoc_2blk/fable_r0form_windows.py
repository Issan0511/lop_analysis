"""平均ベクトルの増分を半径成分と直交成分に割る（Fable エージェント A、依頼 4 の A-7、1028）。

恒等式: ‖a + Δa‖² − ‖a‖² = 2⟨a, Δa⟩ + ‖Δa‖²。半径項 2⟨a,Δa⟩ は向きを持ち、直交項 ‖Δa‖² は常に正（足し算の法則）。
log ‖a‖² の増分 = log(1 + 半径項/‖a‖² + 直交項/‖a‖²)。
課題の中（回し直しの 50 更新窓、fable_replay_*_vectors.npz の FFN0 平均 patch 出力・m2・注意 1 平均 patch 出力）と、
課題をまたぐ保存状態（fable_coherence_vectors.npz）で、窓の種類（崩れ・上昇・平ら、未当てはめ・当てはめ済み）ごとに和を出す。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_windows.py
"""
import json
import math

import numpy as np

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import ROOT

OUT = ROOT / 'posthoc_r0_form'
KEYS = ['ffn0_patch', 'm2', 'attn1_patch', 'attn0_patch']


def decompose(X):
    """X: (n, d)。窓 k の半径項・直交項（‖a_k‖² で割った比）と log‖a‖² の増分。"""
    rows = []
    for k in range(len(X) - 1):
        a, b = X[k], X[k + 1]
        d = b - a
        n2 = float(a @ a)
        radial = float(2 * a @ d) / n2
        ortho = float(d @ d) / n2
        rows.append(dict(radial=radial, ortho=ortho, dlog2=math.log(float(b @ b) / n2),
                         cos=float(a @ d / (np.linalg.norm(a) * np.linalg.norm(d) + 1e-30))))
    return rows


def main():
    result = {'within': {}, 'across': {}}
    lines = ['## 課題の中の 50 更新窓: 平均ベクトルの ‖a‖² の増分の半径項と直交項（‖a‖² 比の和、窓の種類ごと）\n',
             '崩れ窓 = 暗記率が 0.1 以上落ちた窓、上昇窓 = 0.1 以上上がった窓、平ら = それ以外。log‖a‖² の増分 ≈ 半径項 + 直交項（小さいとき）。\n',
             '| 走 | ベクトル | 窓の種類 | 窓数 | Σ 半径項 | Σ 直交項 | Σ Δlog‖a‖² | 半径 cos の平均 |', '|---|---|---|---|---|---|---|---|']
    for f in sorted(OUT.glob('fable_replay_*.json')):
        d = json.loads(f.read_text())
        if d['steps'] < 5000:
            continue
        frz = d.get('freeze_body_until', 0)
        name = f'{d["arm"]}_t{d["task_end"]}_{d["adam"]}' + (f'_frz{frz}' if frz else '')
        z = np.load(f.with_name(f.stem + '_vectors.npz'))
        steps = z['steps'].tolist()
        rec = {r['step']: r for r in d['records']}
        kinds = []
        for k in range(len(steps) - 1):
            a0, a1 = rec[steps[k]]['acc'], rec[steps[k + 1]]['acc']
            kind = 'crash' if a1 - a0 <= -.1 else 'rise' if a1 - a0 >= .1 else 'flat'
            kinds.append(dict(kind=kind, unfit=a0 < .9))
        result['within'][name] = {}
        for key in KEYS:
            rows = decompose(z[key].astype(np.float64))
            agg = {}
            for label, cond in (('crash', lambda kk: kk['kind'] == 'crash'), ('rise', lambda kk: kk['kind'] == 'rise'), ('flat', lambda kk: kk['kind'] == 'flat'),
                                ('unfit', lambda kk: kk['unfit']), ('fit', lambda kk: not kk['unfit']), ('all', lambda kk: True)):
                sel = [r for r, kk in zip(rows, kinds) if cond(kk)]
                agg[label] = dict(n=len(sel), radial=float(sum(r['radial'] for r in sel)), ortho=float(sum(r['ortho'] for r in sel)),
                                  dlog2=float(sum(r['dlog2'] for r in sel)), cos_mean=float(np.mean([r['cos'] for r in sel])) if sel else None)
            result['within'][name][key] = agg
            if key in ('ffn0_patch', 'm2'):
                for label in ('crash', 'rise', 'flat', 'unfit', 'fit', 'all'):
                    g = agg[label]
                    cm = f'{g["cos_mean"]:+.2f}' if g['cos_mean'] is not None else '—'
                    lines.append(f'| {name} | {key} | {label} | {g["n"]} | {g["radial"]:+.3f} | {g["ortho"]:+.3f} | {g["dlog2"]:+.3f} | {cm} |')
    # 課題をまたぐ
    zc = np.load(OUT / 'fable_coherence_vectors.npz')
    lines += ['\n## 課題をまたぐ保存状態: 増分の半径項と直交項\n', '| 腕 | ベクトル | 区間 | 区間数 | Σ 半径項 | Σ 直交項 | Σ Δlog‖a‖² | 半径 cos の平均 | 区間ごとの (半径, 直交) |', '|---|---|---|---|---|---|---|---|---|']
    spans = {'t1_10': list(range(1, 11)), 't10_40': [10, 20, 30, 40], 't40_100': list(range(40, 101, 10)), 't20_40': [20, 30, 40]}
    for arm in ['R0', 'S']:
        for key in ['ffn0_patch', 'm2', 'attn1_patch']:
            for span, ts in spans.items():
                X = np.stack([zc[f'{arm}_t{t}__{key}'] for t in ts]).astype(np.float64)
                rows = decompose(X)
                g = dict(n=len(rows), radial=float(sum(r['radial'] for r in rows)), ortho=float(sum(r['ortho'] for r in rows)),
                         dlog2=float(sum(r['dlog2'] for r in rows)), cos_mean=float(np.mean([r['cos'] for r in rows])), rows=rows)
                result['across'][f'{arm}_{key}_{span}'] = g
                per = ' '.join(f'({r["radial"]:+.2f}, {r["ortho"]:+.2f})' for r in rows)
                lines.append(f'| {arm} | {key} | {span} | {g["n"]} | {g["radial"]:+.2f} | {g["ortho"]:+.2f} | {g["dlog2"]:+.2f} | {g["cos_mean"]:+.2f} | {per} |')
    (OUT / 'fable_windows.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUT / 'fable_windows_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
