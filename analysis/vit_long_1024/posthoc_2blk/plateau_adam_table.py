"""停滞を分ける回し直し（1102、§9）の集計: T90 と停滞の長さの表、記録した時系列の窓ごとの平均の表（Markdown）。

    python analysis/vit_long_1024/posthoc_2blk/plateau_adam_table.py [名前 ...]
"""
import json
import sys
from pathlib import Path

OUT = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_scale'
DEFAULT = ['base', 'base_track', 'head0', 'head0_track', 'head0+adam0', 'adam0at25', 'bodyadam0at25',
           'beta2_0.99', 'head0+beta2_0.99', 'head0at25+adam0at25']
WINDOWS = [(1, 25), (26, 100), (101, 300), (301, 500)]
STEP = 25


def first_below(ml, th):
    for i, v in enumerate(ml):
        if v < th:
            return i * STEP
    return None


def main(names):
    rows, tracks = [], {}
    for n in names:
        f = OUT / f'scale_SL_T100_{n}.json'
        if not f.exists():
            print(f'（{n}: 無し）')
            continue
        j = json.load(open(f))
        ml = j['mem_loss']
        rows.append((n, ml[0], ml[1] if len(ml) > 1 else None, first_below(ml, 2.0), first_below(ml, 1.5), first_below(ml, 1.0),
                     j['T50'], j['T75'], j['T90'], j['status']))
        if j.get('track'):
            tracks[n] = j['track']
    print('| 腕 | 始めの損失 | 25 更新目 | 損失 < 2.0 | < 1.5 | < 1.0 | T50 | T75 | T90 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for n, s0, s25, a, b, c, t50, t75, t90, status in rows:
        f = lambda v: '—' if v is None else f'{v:,}'
        print(f'| {n} | {s0:.1f} | {s25:.2f} | {f(a)} | {f(b)} | {f(c)} | {f(t50)} | {f(t75)} | {f(t90)}{"" if status == "reached90" else " (" + status + ")"} |')
    if not tracks:
        return
    for key, label in [('d_body', '体の 1 更新あたりの変位 ‖Δθ_body‖'), ('s_body', '体の m̂/√v̂ の RMS（sign の歩幅 = 1）'),
                       ('g_body', '体の勾配 ‖g_body‖'), ('v_body', '体の √mean(v)'), ('d_head', 'head（損失の 10 行）の 1 更新あたりの変位'),
                       ('s_head', 'head の m̂/√v̂ の RMS'), ('loss', '学習の損失')]:
        print(f'\n{label}（窓の平均）\n')
        print('| 腕 | ' + ' | '.join(f'{a}〜{b}' for a, b in WINDOWS) + ' |')
        print('|---|' + '---|' * len(WINDOWS))
        for n, t in tracks.items():
            vals = []
            for a, b in WINDOWS:
                xs = [x for s, x in zip(t['step'], t[key]) if a <= s <= b and x == x]
                vals.append(sum(xs) / len(xs) if xs else None)
            fmt = (lambda v: f'{v:.2e}') if key == 'v_body' else (lambda v: f'{v:.3f}')
            print(f'| {n} | ' + ' | '.join('—' if v is None else fmt(v) for v in vals) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT)
