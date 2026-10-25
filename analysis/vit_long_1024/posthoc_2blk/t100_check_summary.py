"""追補 B の記述: (1) 課題 1〜60 が追補 A の seed 0 とビットで一致するか、(2) 4 腕と fresh の要約（Claude、1025）。"""
import csv, numpy as np, torch
from pathlib import Path
D = Path.home() / 'Projects/obsidian-research-data'
A, B = D / 'vit_long_1024_2blk/runs', D / 'vit_long_1024_2blk_t100/runs'
TIMING = {'train_seconds', 'record_seconds', 'other_seconds', 'seconds'}
rd = lambda p: list(csv.DictReader(open(p)))
def same(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
    return x == y
print('(1) 課題 1〜60 の一致（時間の列を除く）')
for arm in ['SL', 'R0L', 'S', 'R0']:
    pa, pb = A / arm / 'seed0', B / arm / 'seed0'
    ta = {int(r['task']): r for r in rd(pa / 'per_task.csv')}; tb = {int(r['task']): r for r in rd(pb / 'per_task.csv')}
    common = sorted(set(ta) & set(tb))
    mis = [(t, c) for t in common for c in ta[t] if c not in TIMING and ta[t][c] != tb[t].get(c)]
    la = {(r['state'], r['task'], r['layer']): r for r in rd(pa / 'layer_stats.csv')}
    lb = {(r['state'], r['task'], r['layer']): r for r in rd(pb / 'layer_stats.csv')}
    lc = sorted(set(la) & set(lb))
    lmis = [(k, c) for k in lc for c in la[k] if la[k][c] != lb[k].get(c)]
    ca = {p.name for p in (pa / 'ckpt').glob('model_t*.pt')}; cb = {p.name for p in (pb / 'ckpt').glob('model_t*.pt')}
    cmis, cn = [], 0
    for n in sorted(ca & cb):
        sa = torch.load(pa / 'ckpt' / n, map_location='cpu', weights_only=False); sb = torch.load(pb / 'ckpt' / n, map_location='cpu', weights_only=False)
        cn += 1
        if not same(sa, sb): cmis.append(n)
    ia, ib = np.load(pa / 'intask.npz'), np.load(pb / 'intask.npz')
    imis = []
    for k in ia.files:
        x, y = ia[k], ib[k]
        n = min(x.shape[0], y.shape[0]) if x.ndim else 0
        if x.ndim and not np.array_equal(x[:n], y[:n], equal_nan=True): imis.append(k)
    print(f" {arm}: 課題 {common[0]}〜{common[-1]}（{len(common)}）per_task 不一致 {len(mis)} | layer_stats {len(lc)} 行 不一致 {len(lmis)} | 保存状態 {cn} 個 不一致 {len(cmis)} | intask 鍵 {len(ia.files)} 不一致 {imis[:5]}"
          + (f" 例 {mis[:3]}" if mis else ''))
print()
print('(2) 4 腕の要約（課題の窓ごと）: online 正解率の平均 | T90 の中央値（未到達 525）| online 損失の平均')
W = [(1, 10), (41, 60), (81, 100)]
def t90(r): return float(r['T90']) if r['T90'] not in ('', 'nan') else 525.0
for arm in ['R0L', 'SL', 'R0', 'S']:
    rows = {int(r['task']): r for r in rd(B / arm / 'seed0' / 'per_task.csv')}
    cells = []
    for a, b in W:
        rr = [rows[t] for t in range(a, b + 1)]
        cells.append(f"{np.mean([float(r['online_acc']) for r in rr]):.3f} | {np.median([t90(r) for r in rr]):.0f} | {np.mean([float(r['online_loss']) for r in rr]):.3f}")
    print(f" {arm}: " + '  ‖  '.join(f'課題 {a}〜{b}: {c}' for (a, b), c in zip(W, cells)))
print()
print('(3) fresh との比較（同じ課題、online 正解率 | T90 | online 損失）')
for fr, arms in [('freshL', ['R0L', 'SL']), ('fresh', ['R0', 'S'])]:
    for t in [25, 50, 75, 100]:
        f = rd(B / fr / 'seed0' / f'task{t:02d}' / 'per_task.csv')[0]
        line = f" {fr} 課題 {t}: {float(f['online_acc']):.3f} | {t90(f):.0f} | {float(f['online_loss']):.3f}"
        for arm in arms:
            r = {int(x['task']): x for x in rd(B / arm / 'seed0' / 'per_task.csv')}[t]
            line += f"  ‖ {arm}: {float(r['online_acc']):.3f} | {t90(r):.0f} | {float(r['online_loss']):.3f}"
        print(line)
print()
print('(4) 課題 10・60・100 末の層ごとの死と相対位置 r（全 token）・CLS の r')
for arm in ['R0L', 'SL', 'R0', 'S']:
    ls = rd(B / arm / 'seed0' / 'layer_stats.csv')
    out = []
    for t in [10, 60, 100]:
        for L in ['1', '2']:
            r = [x for x in ls if x['state'] != 'init' and int(x['task']) == t and x['layer'] == L][0]
            out.append(f"t{t}L{L} 死 {float(r['dead005']):.2f} r {float(r['r_med']):+.2f} CLS {float(r['cls_r_med']):+.2f}")
    print(f" {arm}: " + ' | '.join(out))
