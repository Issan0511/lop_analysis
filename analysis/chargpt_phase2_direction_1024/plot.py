#!/usr/bin/env python3
"""Render saved phase-2/CE push directions; no model loads or training."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATES = [(9, 10000), (39, 10000), (99, 10000), (100, 100), (100, 10000)]
ARMS = ['C3', 'ReLUcap']
METRICS = ['SGD', 'Adam_fixed']
BLUE, ORANGE, GRAY = '#2474a5', '#cb792d', '#d4d9de'
CMAP = ListedColormap([ORANGE, GRAY, BLUE])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ba(x, y):
    pos, neg = y > 0, y < 0
    return float(((x[pos] > 0).mean() + (x[neg] < 0).mean()) / 2) if pos.any() and neg.any() else np.nan


def rho(x, y):
    return float(pd.Series(x).rank(method='average').corr(pd.Series(y).rank(method='average')))


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(out / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def direction_figure(arrays, frame, arm, metric, out):
    fig = plt.figure(figsize=(13, 8.7))
    ax = fig.add_axes([.15, .19, .40, .56])
    ay = fig.add_axes([.66, .19, .29, .56])
    metric_name = 'Adam分母固定' if metric == 'Adam_fixed' else 'SGD方向'
    fig.text(.06, .95, '相2が読む向きと、合計の押しの向き', fontsize=21, weight='bold')
    fig.text(.06, .905, f'{arm}  /  L0  /  課題100末  /  {metric_name}  /  全10 seed・生きた全unit', fontsize=12)
    fig.legend(handles=[Patch(color=BLUE, label='沈む（rを下げる）'), Patch(color=ORANGE, label='浮く（rを上げる）'),
                        Patch(color=GRAY, label='押しがゼロ')], loc='upper left', bbox_to_anchor=(.055, .875),
               ncol=3, frameon=False, fontsize=11)
    ax.set_title('同じunitを上下で比較\n上：相2の予測　下：実測', fontsize=12, pad=13)
    ay.set_title('予測で分けた2群の実測\n実測で沈むunitの割合', fontsize=12, pad=13)
    labels = []
    for seed in range(10):
        x, y = arrays[(arm, seed, 100, 10000, metric)]
        order = np.argsort(x, kind='stable')
        for values, lo, hi in [(x, seed - .29, seed - .025), (y, seed + .025, seed + .29)]:
            ax.imshow(np.sign(values[order])[None, :] + 1, cmap=CMAP, vmin=0, vmax=2,
                      extent=(0, 100, hi, lo), aspect='auto', interpolation='nearest', rasterized=True)
        row = frame[(frame.arm == arm) & (frame.seed == seed) & (frame.task == 100) &
                    (frame.step == 10000) & (frame.metric == metric)].iloc[0]
        neg = row.p_actual_sink_given_pred_float * 100
        pos = row.p_actual_sink_given_pred_sink * 100
        ay.plot([neg, pos], [seed, seed], color='#c2c9ce', lw=1.7, zorder=1)
        ay.scatter([neg, pos], [seed, seed], c=[ORANGE, BLUE], s=49, zorder=3)
        ay.text(neg - 3.8, seed, f'{neg:.0f}%', ha='right', va='center', fontsize=10, color=ORANGE)
        ay.text(pos + 3.8, seed, f'{pos:.0f}%', ha='left', va='center', fontsize=10, color=BLUE)
        labels.append(f'seed {seed}\nn={len(x)} / ゼロ{int(row.n_both_zero)}')
    ax.set(yticks=range(10), yticklabels=labels, xlim=(0, 100), ylim=(9.6, -.6), xticks=[0, 50, 100])
    ax.set_xticklabels(['0%', '50%', '100%'])
    ax.set_xlabel('seed内で予測値の小さい順に並べたunit', fontsize=10, labelpad=9)
    ax.tick_params(axis='y', length=0, pad=13, labelsize=9)
    ay.set(xlim=(-9, 111), ylim=(9.6, -.6), xticks=[0, 25, 50, 75, 100], yticks=[])
    ay.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ay.set_xlabel('橙：浮くと予測した群　青：沈むと予測した群', fontsize=9, labelpad=9)
    ay.grid(axis='x', alpha=.18)
    for a in [ax, ay]:
        a.spines[['top', 'left', 'right']].set_visible(False)
        a.spines['bottom'].set_color('#b6bec5')
    fig.text(.06, .095, '実測：他unit・後段の注意・FFN・LNを含む、L0 up行への切替の合計勾配から算出。', fontsize=10)
    fig.text(.06, .065, '予測の量は合わせていない。灰色の0同士は、右の「沈む／浮く」2群に入れない。nは生きたunit数。', fontsize=10)
    fig.text(.06, .035, '保存状態での局所的な向きの照合。全パラメータの実更新や、長期の沈降の因果を示す図ではない。', fontsize=9, color='#5d6770')
    save(fig, out, f'direction_{arm}_{metric}_t100')


def overview(frame, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for i, arm in enumerate(ARMS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            sub = frame[(frame.arm == arm) & (frame.metric == metric)]
            for k, (task, step) in enumerate(STATES):
                rows = sub[(sub.task == task) & (sub.step == step)].sort_values('seed')
                if rows.empty:
                    ax.text(k, 12, '分母の保存なし', ha='center', fontsize=8, color='#7b858c')
                    continue
                vals = rows.balanced_accuracy_nonzero.to_numpy() * 100
                ax.scatter(k + np.linspace(-.13, .13, len(vals)), vals, c=BLUE, s=25, alpha=.72)
                ax.plot([k - .19, k + .19], [np.median(vals)] * 2, color='#16374e', lw=2)
            ax.axhline(50, ls='--', color='#9ba5ac', lw=1)
            ax.set_title(f'{arm} / ' + ('Adam分母固定' if metric == 'Adam_fixed' else 'SGD方向'), loc='left', fontsize=12)
            ax.set(ylim=(0, 101), xticks=range(5), xticklabels=['9末', '39末', '99末', '100頭\n100更新', '100末'])
            ax.grid(axis='y', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0:
                ax.set_ylabel('両方向の的中率の平均 (%)', fontsize=11)
    fig.suptitle('ゼロ同士の一致を除いて、全保存点・全seedで確認', x=.08, ha='left', fontsize=19, weight='bold')
    fig.text(.08, .91, '1点 = 1 seed　横線 = seed中央値　破線50% = 片方の向きだけを答える場合', fontsize=11)
    fig.text(.08, .045, '実測で沈む群／浮く群それぞれの的中率を平均（balanced accuracy）。予測・実測の片方でも0なら別集計。', fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.85, bottom=.13, hspace=.28, wspace=.13)
    save(fig, out, 'direction_overview_nonzero')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path.home() / 'Projects/obsidian-research-data/chargpt_l0phase2_1023/results/chargpt_l0phase2_1023')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/chargpt_phase2_direction_1024')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    original = pd.read_csv(args.source / 'headcov_states.csv')
    rows, arrays, files, differences = [], {}, {}, []
    for arm in ARMS:
        for seed in range(10):
            for task, step in STATES:
                paths = [args.source / 'raw' / f'{tag}_{arm}_seed{seed}_t{task}_u{step}.npz' for tag in ['headcov', 'scalar']]
                for path in paths:
                    files[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
                with np.load(paths[0]) as pred, np.load(paths[1]) as actual:
                    mask = pred['alive'].astype(bool)
                    assert np.array_equal(mask, actual['alive'])
                    for metric in METRICS:
                        if metric + '_iid' not in pred:
                            continue
                        x, y = pred[metric + '_iid'][mask], actual[metric + '_switch_r'][mask]
                        assert np.isfinite(x).all() and np.isfinite(y).all()
                        arrays[(arm, seed, task, step, metric)] = (x.copy(), y.copy())
                        nz = (x != 0) & (y != 0)
                        xp, yp = x > 0, y > 0
                        legacy_ba = ((xp[yp]).mean() + (~xp[~yp]).mean()) / 2
                        row = dict(arm=arm, seed=seed, task=task, step=step, metric=metric,
                            n_alive=len(x), n_nonzero=int(nz.sum()), n_both_zero=int(((x == 0) & (y == 0)).sum()),
                            n_pred_only_zero=int(((x == 0) & (y != 0)).sum()), n_actual_only_zero=int(((x != 0) & (y == 0)).sum()),
                            agreement_legacy=float((xp == yp).mean()), balanced_accuracy_legacy=float(legacy_ba),
                            agreement_nonzero=float((np.sign(x[nz]) == np.sign(y[nz])).mean()),
                            balanced_accuracy_nonzero=ba(x[nz], y[nz]),
                            rho_all=rho(x, y), rho_nonzero=rho(x[nz], y[nz]),
                            n_pred_sink=int(xp.sum()), n_pred_float=int((x < 0).sum()),
                            p_actual_sink_given_pred_sink=float((y[xp] > 0).mean()),
                            p_actual_sink_given_pred_float=float((y[x < 0] > 0).mean()),
                            p_actual_zero_given_pred_sink=float((y[xp] == 0).mean()),
                            p_actual_zero_given_pred_float=float((y[x < 0] == 0).mean()),
                            pred_mean=float(x.mean()), actual_mean=float(y.mean()))
                        old = original[(original.arm == arm) & (original.seed == seed) & (original.task == task) &
                                       (original.step == step) & (original.metric == metric) & (original.covariance == 'iid')]
                        assert len(old) == 1 and int(old.iloc[0].switch_n) == len(x)
                        for key, oldkey in [('agreement_legacy', 'switch_agreement'), ('balanced_accuracy_legacy', 'switch_balanced_accuracy'), ('rho_all', 'switch_rho')]:
                            diff = abs(row[key] - old.iloc[0][oldkey])
                            assert diff < 1e-12, (arm, seed, task, step, metric, key, diff)
                            differences.append(diff)
                        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / 'directions.csv', index=False)
    for arm in ARMS:
        for metric in METRICS:
            direction_figure(arrays, frame, arm, metric, args.out)
    overview(frame, args.out)
    report = ['# 相2の方向と実測を並べる（保存済み配列の表示）', '',
        '最終head全方向の条件付き相2模型と、独立新課題ラベル平均CEの押しを同じunitで比較。新しい学習・checkpoint更新はなし。', '',
        '## 課題100末、10seedの範囲', '',
        '| 腕 | 分母 | 沈む予測群で実測も沈む | 浮く予測群で実測が沈む | 非ゼロ符号一致 | 非ゼロbalanced accuracy | 非ゼロρ |',
        '|---|---|---|---|---|---|---|']
    def rng(series, percent=False):
        return f'{100*series.min():.1f}–{100*series.max():.1f}%' if percent else f'{series.min():.3f}–{series.max():.3f}'
    for arm in ARMS:
        for metric in METRICS:
            sub = frame[(frame.arm == arm) & (frame.metric == metric) & (frame.task == 100) & (frame.step == 10000)]
            values = [rng(sub[c], c != 'rho_nonzero') for c in ['p_actual_sink_given_pred_sink', 'p_actual_sink_given_pred_float', 'agreement_nonzero', 'balanced_accuracy_nonzero', 'rho_nonzero']]
            report.append('| ' + ' | '.join([arm, metric] + values) + ' |')
    report += ['', '## ゼロの扱いと既報の検査', '',
        '既報は `S>0 / S<=0` の一致率。ゼロ同士も一致とした。今回の図では0を灰色にし、二方向の一致・balanced accuracy・ρは両方が非ゼロのunitでも計算した。旧値を訂正したものではなく、別の問いに対応する集計。', '',
        f'全{len(frame)}組の旧符号一致・balanced accuracy・ρを再計算し、headcov_states.csvとの最大絶対差は {max(differences):.3g}。', '',
        f'ゼロが片側だけのunit数（全状態・metricを通じた延べ数）: 予測だけ {int(frame.n_pred_only_zero.sum())}、実測だけ {int(frame.n_actual_only_zero.sum())}。', '',
        '## 図が示すもの', '',
        '予測値を縦の帯の並べ替えにだけ使い、実測の符号に合わせて係数fitやunit選択はしていない。右側は各seed内の条件付き割合で、seedをプールしていない。overviewの50%は両方向の的中率の平均に対する単一方向回答の水準で、元の符号一致率の帰無値ではない。', '',
        'この対応は観察的な照合であり、相2に固有の因果を特定しない。L0 up行への勾配は他unit・後段を全て含むが、全パラメータを実際に更新したrの変化ではない。入力は現課題の鎖probeで、新課題の入力を再生したものでもない。Adamは分母固定、運動量なし。', '',
        '理論側はHを旧ラベル平均の間固定し、自由な最終headを共同fitする条件付き模型。量の共通係数を省いている。現課題fit中の押しや長期沈降を、この切替方向の一致で説明し終えたとはしない。', '',
        '前提に沿うラベルと最小二乗headを人工的に作って方向を再確認しても、それだけでは実学習で相2が原因だった証拠にならない。原因の識別には、相2の条件に対する事前に予測した応答と、別機構でも同じ応答になる可能性の検討が要る。', '',
        '## 再現と保存', '', 'コード・表示方針: `analysis/chargpt_phase2_direction_1024/`。入力SHAはprovenance.json。図はPNG/PDF。新たな生配列は作らず、親実験の退避済み配列を読み取り専用で利用した。']
    (args.out / 'summary.md').write_text('\n'.join(report) + '\n')
    provenance = dict(parent_commit='6f780068e043520a50bb4b3e5c303420ed88927e',
        display_plan_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256=sha(Path(__file__)), source_summary_sha256=sha(args.source / 'headcov_states.csv'),
        legacy_comparisons=len(differences), max_legacy_difference=max(differences), rows=len(frame),
        training_steps=0, checkpoint_loads=0, inputs=files)
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.out / 'backup_manifest.json').write_text(json.dumps(dict(
        note='All new outputs are committed. No new git-excluded raw data; archived parent inputs read only.',
        parent_manifest=str(args.source / 'backup_manifest.json'), files=[]), indent=2) + '\n')
    print('\n'.join(report[:15]))
    print('legacy max difference', max(differences), 'rows', len(frame))


if __name__ == '__main__':
    main()
