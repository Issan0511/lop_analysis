"""act_chimera_0913 事後解析（未登録）: 負側の効果は φ′（逆向きのゲート）と φ（前向きの値）のどちらを通るか。

箱 B（PMNIST）の分離腕の late 精度（t101–120 の窓平均）と劣化 L = acc(t16–20) − acc(t101–120) を、
単純効果ごとに「全部入り / ゲートだけ / 前向きだけ」に並べる。入力は commit 済みの
results/act_chimera_0913/pmnist/{ARM}_s{seed}_rows.csv だけ。登録した判定（D5a）は verdict.csv が正。

    python3 -m analysis.act_chimera_0913.posthoc_gate_channel
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PM = ROOT / 'results' / 'act_chimera_0913' / 'pmnist'
SEEDS = (0, 1, 2)
# 単純効果: (起点, 全部入り, ゲートだけ, 前向きだけ)
CHANNELS = {
    '近傍 LR → SMAXH': ('LR', 'SMAXH', 'GN', 'FN'),
    '深部 ELU1 → SMAXH': ('ELU1', 'SMAXH', 'GD', 'FD'),
}


def windows(arm, seed):
    d = pd.read_csv(PM / f'{arm}_s{seed}_rows.csv')
    acc = d['acc'] * 100
    base = acc[(d['task'] >= 16) & (d['task'] <= 20)].mean()
    late = acc[(d['task'] >= 101) & (d['task'] <= 120)].mean()
    return late, base - late


def main():
    lines = ['| 単純効果 | 量 | 全部入り | ゲートだけ | 前向きだけ |', '|---|---|---|---|---|']
    for name, (a, full, gate, fwd) in CHANNELS.items():
        for q, idx, sign in (('late 精度の差 [pt]', 0, 1), ('劣化 L の減少 [pt]', 1, -1)):
            cells = []
            for arm in (full, gate, fwd):
                v = [sign * (windows(arm, s)[idx] - windows(a, s)[idx]) for s in SEEDS]
                cells.append(' / '.join(f'{x:+.2f}' for x in v))
            lines.append(f'| {name}（{full}/{gate}/{fwd} − {a}） | {q} | ' + ' | '.join(cells) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
