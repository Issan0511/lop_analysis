"""§7 judgements of vit_long_1024 (fixed before the run): J0, J1 and J3, records, prediction scores.

J5 is post-hoc on saved states (analysis/vit_long_1024/j5/, after the run) and is reported
here as pending. Reads per_task.csv / layer_stats.csv / done.json under a results or raw root.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analysis.joudaki_vit_battle_0919.run import atomic_json

from . import common as C

EARLY = tuple(range(2, 12))
LATE = tuple(range(41, 61))
FRESH = C.FRESH_TASKS
PREDICTIONS = [  # (symbol, judgement, predicted label, Fable, Opus 5.5)
    ('P0', 'J0', 'SHARED_WORSE', .65, .55),
    ('P1R', 'J1 R0', 'LOSS', .70, .65),
    ('P1S', 'J1 S', 'LOSS', .80, .70),
    ('P3R', 'J3 R0', 'DEATH', .65, .60),
    ('P3S', 'J3 S', 'DEATH', .75, .70),
    ('P5', 'J5 S', 'LOADING_SINKS', .60, .55),
]


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load(root):
    """{arm: {seed: {'tasks': {t: row}, 'end': {t: [layer rows]}, 'done': dict}}}, fresh: {seed: {t: row}}."""
    root = Path(root)
    runs = {arm: {} for arm in C.ARMS}
    for arm in C.ARMS:
        for seed in C.SEEDS:
            run = root / arm / f'seed{seed}'
            if not (run / 'per_task.csv').exists():
                continue
            tasks = {int(r['task']): r for r in read_csv(run / 'per_task.csv')}
            end = {}
            if (run / 'layer_stats.csv').exists():
                for r in read_csv(run / 'layer_stats.csv'):
                    if r['state'] == 'end':
                        end.setdefault(int(r['task']), []).append(r)
            done = json.loads((run / 'done.json').read_text()) if (run / 'done.json').exists() else {}
            runs[arm][seed] = dict(tasks=tasks, end=end, done=done)
    fresh = {}
    for seed in C.SEEDS:
        for task in FRESH:
            path = root / 'fresh' / f'seed{seed}' / f'task{task:02d}' / 'per_task.csv'
            if path.exists():
                rows = read_csv(path)
                if rows:
                    fresh.setdefault(seed, {})[task] = rows[0]
    return runs, fresh


def complete(run, window):
    return run is not None and all(t in run['tasks'] for t in window)


def online(run, window, key='online_acc'):
    return float(np.mean([num(run['tasks'][t][key]) for t in window]))


def onset(run):
    """First t (6..60) with the mean online of tasks t-4..t below .5; None if it never happens."""
    for t in range(6, C.TASKS + 1):
        window = range(t - 4, t + 1)
        if not all(s in run['tasks'] for s in window):
            return None
        if np.mean([num(run['tasks'][s]['online_acc']) for s in window]) < .5:
            return t
    return None


def j0(runs):
    pairs = {}
    for seed in C.SEEDS:
        r0, s = runs['R0'].get(seed), runs['S'].get(seed)
        entry = dict(onset_R0=onset(r0) if r0 else None, onset_S=onset(s) if s else None)
        if complete(r0, LATE) and complete(s, LATE):
            entry.update(delta_late=online(s, LATE) - online(r0, LATE),
                         late_R0=online(r0, LATE), late_S=online(s, LATE))
        if complete(r0, EARLY) and complete(s, EARLY):
            entry.update(delta_early=online(s, EARLY) - online(r0, EARLY),
                         delta_early_first100=online(s, EARLY, 'online_acc_first100')
                         - online(r0, EARLY, 'online_acc_first100'))
        if complete(r0, LATE) and complete(s, LATE):
            entry['delta_late_first100'] = (online(s, LATE, 'online_acc_first100')
                                            - online(r0, LATE, 'online_acc_first100'))
        pairs[seed] = entry
    deltas = [e['delta_late'] for e in pairs.values() if 'delta_late' in e]
    negative, positive = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
    if len(deltas) < 4:
        label = 'INSUFFICIENT'
    elif negative >= 4:
        label = 'SHARED_WORSE'
    elif positive >= 4:
        label = 'SHARED_BETTER'
    else:
        label = 'MIXED'
    return dict(label=label, usable=len(deltas), negative=negative, positive=positive, seeds=pairs)


def t90(run, task):
    return num(run['tasks'][task]['T90'])


def j1(runs, arm):
    seeds = {}
    for seed, run in runs[arm].items():
        if not (complete(run, EARLY) and complete(run, LATE)):
            seeds[seed] = dict(usable=False)
            continue
        steps = int(float(run['tasks'][EARLY[0]]['steps']))
        unreached = steps + C.EVAL_EVERY                   # 525 for U = 500 (spec §1)
        early = [t90(run, t) for t in EARLY]
        late = [t90(run, t) for t in LATE]
        E, L = float(np.median(early)), float(np.median(late))
        N = int(sum(v >= unreached for v in late))
        reached_early = int(sum(v <= steps for v in early))
        seeds[seed] = dict(usable=True, steps=steps, unreached_value=unreached, E=E, L=L, N_unreached_late=N, ratio=L / E,
                           established=reached_early >= 6, reached_early=reached_early,
                           loss=bool(L >= 2 * E and N >= 11),
                           decades=[dict(tasks=f'{a}-{a + 9}',
                                         median=float(np.median([t90(run, t) for t in range(a, a + 10)])),
                                         unreached=int(sum(t90(run, t) >= unreached for t in range(a, a + 10))))
                                    for a in range(1, C.TASKS, 10) if complete(run, range(a, a + 10))])
    usable = [v for v in seeds.values() if v['usable']]
    if len(usable) < 4:
        label = 'INSUFFICIENT'
    elif sum(v['loss'] for v in usable) >= 3:
        label = 'LOSS'
    elif sum(v['established'] for v in usable) < 3:
        label = 'NOT_JUDGEABLE'
    else:
        label = 'NO_LOSS'
    return dict(label=label, usable=len(usable), loss_seeds=sum(v['loss'] for v in usable),
                established_seeds=sum(v['established'] for v in usable), seeds=seeds)


def j3(runs, arm):
    seeds = {}
    for seed, run in runs[arm].items():
        rows = run['end'].get(C.TASKS)
        if not rows or len(rows) != C.DEPTH:
            seeds[seed] = dict(usable=False)
            continue
        D = float(np.mean([num(r['dead005']) for r in rows]))
        first = next((t for t in sorted(run['end']) if len(run['end'][t]) == C.DEPTH
                      and np.mean([num(r['dead005']) for r in run['end'][t]]) >= .2), None)
        at40 = run['end'].get(40)
        seeds[seed] = dict(usable=True, D=D, death=D >= .2,
                           dead0_mean=float(np.mean([num(r['dead0']) for r in rows])),
                           per_layer=[num(r['dead005']) for r in rows],
                           readout_dead_t60=float(np.mean([num(r['readout_dead']) for r in rows])),
                           readout_dead_t40=float(np.mean([num(r['readout_dead']) for r in at40])) if at40 else None,
                           D_t40=float(np.mean([num(r['dead005']) for r in at40])) if at40 else None,
                           first_task_D_ge_02=first)
    usable = [v for v in seeds.values() if v['usable']]
    if len(usable) < 4:
        label = 'INSUFFICIENT'
    else:
        label = 'DEATH' if sum(v['death'] for v in usable) >= 3 else 'NO_DEATH'
    return dict(label=label, usable=len(usable), death_seeds=sum(v['death'] for v in usable), seeds=seeds)


def fresh_records(runs, fresh):
    rows = []
    for seed, tasks in sorted(fresh.items()):
        for task, f in sorted(tasks.items()):
            entry = dict(seed=seed, task=task, fresh_online=num(f['online_acc']),
                         fresh_online_first100=num(f['online_acc_first100']), fresh_T90=num(f['T90']))
            for arm in C.ARMS:
                run = runs[arm].get(seed)
                if run and task in run['tasks']:
                    row = run['tasks'][task]
                    entry[f'{arm}_online'] = num(row['online_acc'])
                    entry[f'{arm}_T90'] = num(row['T90'])
                    entry[f'{arm}_delta_fresh'] = entry['fresh_online'] - num(row['online_acc'])
                    entry[f'{arm}_delta_fresh_first100'] = (entry['fresh_online_first100']
                                                            - num(row['online_acc_first100']))
            rows.append(entry)
    return rows


def judge(root):
    runs, fresh = load(root)
    result = dict(J0=j0(runs), J1={arm: j1(runs, arm) for arm in C.ARMS}, J3={arm: j3(runs, arm) for arm in C.ARMS},
                  J5=dict(label='PENDING', note='post-hoc on saved states (spec §7.4), not computed here'),
                  fresh=fresh_records(runs, fresh),
                  diverged={arm: {seed: run['done'] for seed, run in runs[arm].items()
                                  if run['done'].get('status') == 'diverged'} for arm in C.ARMS})
    outcome = {'J0': result['J0']['label'], 'J1 R0': result['J1']['R0']['label'], 'J1 S': result['J1']['S']['label'],
               'J3 R0': result['J3']['R0']['label'], 'J3 S': result['J3']['S']['label'], 'J5 S': 'PENDING'}
    result['predictions'] = [dict(symbol=s, judgement=j, predicted=p, fable=f, opus=o, outcome=outcome[j],
                                  hit=None if outcome[j] in ('PENDING', 'INSUFFICIENT') else outcome[j] == p)
                             for s, j, p, f, o in PREDICTIONS]
    return result


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _clean(obj.item())
    return obj


def markdown(result):
    lines = ['# vit_long_1024 判定（spec §7、走る前に固定した規則）', '']
    j0r = result['J0']
    lines += [f"## J0 SHARED_VS_RESET: `{j0r['label']}`（使える対 {j0r['usable']}、負 {j0r['negative']}、正 {j0r['positive']}）", '',
              '| seed | Δ（後期、S − R0） | R0 後期 | S 後期 | 始まり R0 | 始まり S |', '| --- | --- | --- | --- | --- | --- |']
    for seed, e in j0r['seeds'].items():
        fmt = lambda v: '—' if v is None else (f'{v:+.4f}' if isinstance(v, float) else str(v))
        lines.append(f"| {seed} | {fmt(e.get('delta_late'))} | {fmt(e.get('late_R0'))} | {fmt(e.get('late_S'))} | "
                     f"{e['onset_R0'] or 'なし'} | {e['onset_S'] or 'なし'} |")
    for arm in C.ARMS:
        r = result['J1'][arm]
        lines += ['', f"## J1 {arm}: `{r['label']}`（使える seed {r['usable']}、条件を満たす seed {r['loss_seeds']}、"
                      f"学習の成立 {r['established_seeds']}）。未到達は U + 25（U = 500 なら 525）として扱う", '',
                  '| seed | E（課題 2〜11 の T90 中央値） | L（課題 41〜60） | 後期の未到達 | L/E | 条件 |', '| --- | --- | --- | --- | --- | --- |']
        for seed, v in r['seeds'].items():
            if v['usable']:
                lines.append(f"| {seed} | {v['E']:.1f} | {v['L']:.1f} | {v['N_unreached_late']}/20 | {v['ratio']:.2f} | {'○' if v['loss'] else '×'} |")
            else:
                lines.append(f'| {seed} | — | — | — | — | 使えない |')
    for arm in C.ARMS:
        r = result['J3'][arm]
        lines += ['', f"## J3 {arm}: `{r['label']}`（使える seed {r['usable']}、D ≥ 0.2 の seed {r['death_seeds']}）", '',
                  '| seed | D（課題 60 末、p < 0.005、6 層平均） | 厳密な死 | 課題 40 末の D | 初めて 0.2 以上 |', '| --- | --- | --- | --- | --- |']
        for seed, v in r['seeds'].items():
            if v['usable']:
                lines.append(f"| {seed} | {v['D']:.4f} | {v['dead0_mean']:.4f} | "
                             f"{'—' if v['D_t40'] is None else format(v['D_t40'], '.4f')} | "
                             f"{v['first_task_D_ge_02'] or 'なし'} |")
            else:
                lines.append(f'| {seed} | — | — | — | 使えない |')
    lines += ['', '## J5: 事後解析（未計算）', '', '## 予測の採点', '', '| 記号 | 判定 | 予測 | Fable | Opus 5.5 | 結果 | 当たり |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for p in result['predictions']:
        hit = '—' if p['hit'] is None else ('○' if p['hit'] else '×')
        lines.append(f"| {p['symbol']} | {p['judgement']} | {p['predicted']} | {p['fable']} | {p['opus']} | {p['outcome']} | {hit} |")
    if result['fresh']:
        lines += ['', '## fresh との比較（記録）', '', '| seed | 課題 | fresh | R0 | S | Δ_fresh R0 | Δ_fresh S |', '| --- | --- | --- | --- | --- | --- | --- |']
        for e in result['fresh']:
            g = lambda k: '—' if e.get(k) is None or (isinstance(e.get(k), float) and not math.isfinite(e[k])) else f'{e[k]:.4f}'
            lines.append(f"| {e['seed']} | {e['task']} | {g('fresh_online')} | {g('R0_online')} | {g('S_online')} | "
                         f"{g('R0_delta_fresh')} | {g('S_delta_fresh')} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(C.RESULTS))
    parser.add_argument('--dest', default=None, help='default: the root itself and RAW/summary')
    args = parser.parse_args()
    result = _clean(judge(args.root))
    dests = [Path(args.dest)] if args.dest else [Path(args.root), C.RAW / 'summary']
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / 'judge.json', result)
        (dest / 'judge.md').write_text(markdown(result))
    print(markdown(result))


if __name__ == '__main__':
    main()
