"""width_aid.py の帳簿を表にする（Claude、1027）。

相対位置 r = μ/σ の変化を、unit ごとに平均 μ の動きと幅 σ の動きに割る（2 つの順序の平均で、和は厳密に Δr）:
    Δr_μ = ½(μ_b − μ_a)(1/σ_a + 1/σ_b)        Δr_σ = ½(μ_a + μ_b)(1/σ_b − 1/σ_a)
幅の動きはさらに、Δlog σ = Δlog‖w‖ + Δlog σ̂（σ̂ = σ/‖w‖、向きと入力の分）の比で Δr_σ を割り振る。
μ < 0 の unit では、幅が伸びると Δr_σ > 0（0 の側へ戻す）になる。これが「幅の応急処置」の量である。

    python analysis/vit_long_1024/posthoc_2blk/width_aid_table.py <npz の置き場> <表.md>
"""
import sys
from pathlib import Path

import numpy as np

M, S, W, B, P = range(5)


def split(a, b):
    """unit ごとの (Δr, Δr_μ, Δr_σ, うち ‖w‖ の分)。a, b は (量, unit)。"""
    ra, rb = a[M] / a[S], b[M] / b[S]
    dm = 0.5 * (b[M] - a[M]) * (1 / a[S] + 1 / b[S])
    ds = 0.5 * (a[M] + b[M]) * (1 / b[S] - 1 / a[S])
    dls, dlw = np.log(b[S] / a[S]), np.log(b[W] / a[W])
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_w = np.where(np.abs(dls) > 1e-12, dlw / dls, 0.)
    return rb - ra, dm, ds, ds * frac_w


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def across_tasks(src, lines):
    lines += ['## 課題末どうし（保存した重み、固定 probe）', '',
              '幅を伸ばして戻すなら、μ < 0 の unit で Δr_σ が Δr_μ の沈みを打ち消しているはずである。'
              '「閉じた側」は区間の始めに μ < 0 の unit。和は unit の平均。', '',
              '| 腕 | 層 | 区間 | median r（始め → 終わり） | mean ‖w‖²（始め → 終わり） | 閉じた側の数 | 閉じた側の Δr | うち μ の動き | うち幅の動き | 幅の動きのうち ‖w‖ |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for arm in ('SL', 'R0L'):
        z = np.load(src / f'tasks_{arm}.npz')
        T, U = list(z['task']), z['units']
        for layer in (0, 1):
            for ta, tb in ((0, 10), (10, 30), (30, 60), (60, 100), (10, 100)):
                a, b = U[T.index(ta), layer], U[T.index(tb), layer]
                dr, dm, ds, dw = split(a, b)
                closed = a[M] < 0
                lines.append(f'| {arm} | {layer + 1} | {ta}→{tb} | {np.median(a[M] / a[S]):+.2f} → {np.median(b[M] / b[S]):+.2f}'
                             f' | {np.mean(a[W] ** 2):.2f} → {np.mean(b[W] ** 2):.2f} | {int(closed.sum())}'
                             f' | {dr[closed].mean():+.3f} | {dm[closed].mean():+.3f} | {ds[closed].mean():+.3f} | {dw[closed].mean():+.3f} |')
    lines.append('')


def within_task(src, lines):
    lines += ['## 課題の中（保存状態＋Adam から 5,000 更新の回し直し、固定 probe）', '',
              'unit ごとに、SNAP のうち r が最も低い更新を谷とし、始め → 谷（沈み）と 谷 → 5,000（戻り）に割る。'
              '集計は谷で沈んだ unit（r_谷 < r_0）の和。「幅の分の割合」= Σ Δr_σ ÷ Σ Δr。'
              '最後の 2 列は説 (A) の検査で、沈みの深さ r_0 − r_谷 と課題の中の ‖w_j‖² の伸びの順位相関、'
              '沈みの深さの上位 1/4 と下位 1/4 の伸びの平均の比。', '',
              '| 腕 | 切替 | 層 | 沈んだ unit | 谷の更新（中央値） | 沈み Σ Δr | うち幅の分 | 戻り Σ Δr | うち μ の分 | うち幅の分（‖w‖ の分） | 戻りのうち幅の割合 | 順位相関（沈み × 伸び） | 伸びの比（深い ÷ 浅い） |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in ('SL', 'R0L'):
        for t in (60, 100):
            f = src / f'replay_{arm}_T{t}.npz'
            if not f.exists():
                continue
            z = np.load(f)
            snap, U = list(z['snap']), z['units']
            for layer in (0, 1):
                X = U[:, layer]                          # (snap, 量, unit)
                r = X[:, M] / X[:, S]
                k = r.argmin(0)                          # unit ごとの谷
                n = X.shape[2]
                trough = X[k, :, np.arange(n)].T         # (量, unit)
                start, end = X[0], X[-1]
                sank = r[k, np.arange(n)] < r[0]
                d1 = split(start, trough)
                d2 = split(trough, end)
                rec = d2[0][sank].sum()
                depth = r[0] - r[k, np.arange(n)]
                grow = end[W] ** 2 - start[W] ** 2
                q1, q3 = np.quantile(depth, [0.25, 0.75])
                ratio = grow[depth >= q3].mean() / grow[depth <= q1].mean()
                lines.append(f'| {arm} | {t}→{t + 1} | {layer + 1} | {int(sank.sum())} / {n} | {int(np.median([snap[i] for i in k[sank]]))}'
                             f' | {d1[0][sank].sum():+.1f} | {d1[2][sank].sum():+.1f} | {rec:+.1f} | {d2[1][sank].sum():+.1f}'
                             f' | {d2[2][sank].sum():+.1f}（{d2[3][sank].sum():+.1f}） | {d2[2][sank].sum() / rec:.3f}'
                             f' | {spearman(depth, grow):+.2f} | {ratio:.2f} |')
    lines.append('')


def main(src, out):
    lines = ['# 「沈んでも幅を伸ばして応急処置する」説の unit ごとの帳簿（width_aid.py）', '']
    across_tasks(src, lines)
    within_task(src, lines)
    Path(out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
