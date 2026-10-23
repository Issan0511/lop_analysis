"""§7 judgements of vit_long_1024 (fixed before the run). Main: J0 with LN (SL - R0L), then JLN.
Records: J0 without LN (S - R0), J1 and J3 of the four arms, spikes and breakdowns, fresh, scores.

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
PREDICTIONS = [  # (symbol, judgement, predicted label, Fable, Opus 5.5); spec §3
    ('P0', 'J0 noLN', 'SHARED_WORSE', .65, .55),
    ('P1R', 'J1 R0', 'LOSS', .70, .65),
    ('P1S', 'J1 S', 'LOSS', .80, .70),
    ('P3R', 'J3 R0', 'DEATH', .65, .60),
    ('P3S', 'J3 S', 'DEATH', .75, .70),
    ('P5', 'J5 S', 'LOADING_SINKS', .60, .55),
    ('P0L', 'J0 LN', 'SHARED_WORSE', .60, .55),
    ('P1RL', 'J1 R0L', 'LOSS', .25, .20),
    ('P1SL', 'J1 SL', 'LOSS', .45, .40),
    ('PLN_BREAK', 'JLN', 'PHASE1_BREAKS_SHIELD', .35, .30),
    ('PLN_HOLD', 'JLN', 'SHIELD_HOLDS', .45, .52),
    ('PLN_NONE', 'JLN', 'NO_SHIELD', .15, .12),
    ('PLN_MIXED', 'JLN', 'MIXED', .05, .06),
    ('P5L', 'J5 SL', 'LOADING_SINKS', .55, .45),
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
    fresh = {arm: {} for arm in C.FRESH_ARMS}
    for arm in C.FRESH_ARMS:
        for seed in C.SEEDS:
            for task in FRESH:
                path = root / arm / f'seed{seed}' / f'task{task:02d}' / 'per_task.csv'
                if path.exists():
                    rows = read_csv(path)
                    if rows:
                        fresh[arm].setdefault(seed, {})[task] = rows[0]
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


def j0(runs, reset='R0', shared='S'):
    pairs = {}
    for seed in C.SEEDS:
        r0, s = runs[reset].get(seed), runs[shared].get(seed)
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
    return dict(label=label, arms=f'{shared} - {reset}', usable=len(deltas), negative=negative, positive=positive,
                seeds=pairs)


def jln(j1_reset_ln, j1_shared_ln):
    """JLN (spec §7.6): does the LN shield hold only without the switch push?"""
    a, b = j1_reset_ln['label'], j1_shared_ln['label']
    if 'INSUFFICIENT' in (a, b):
        return 'INSUFFICIENT'
    if 'NOT_JUDGEABLE' in (a, b):
        return 'NOT_JUDGEABLE'
    if a != 'LOSS' and b == 'LOSS':
        return 'PHASE1_BREAKS_SHIELD'
    if a != 'LOSS' and b != 'LOSS':
        return 'SHIELD_HOLDS'
    if a == 'LOSS' and b == 'LOSS':
        return 'NO_SHIELD'
    return 'MIXED'


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


def spikes_of(intask, n_labels, steps, reach_percent=90):
    """Per task (spec §7.5): a spike is a 25-update block (u = 25k .. 25k + 25) after the memorisation
    rate first reaches .9 whose mean online loss exceeds ln K (the loss of a uniform prediction);
    a breakdown is a spike with the end-of-task memorisation rate below .5. reach_percent = 80 is the
    variant recorded alongside (white-san's R0 seed 0 task 3 peaked at 223/250 = .892 and then broke)."""
    limit = math.log(n_labels)
    out = []
    n_images = None
    for i, task in enumerate(intask['tasks']):
        n_images = int(intask['batch_size'][i][:2].sum()) if n_images is None else n_images
        mem = intask['mem_correct'][i]
        reach = next((k for k in range(1, len(mem)) if int(mem[k]) * 100 >= reach_percent * n_images), None)
        spike = False
        if reach is not None:
            u90 = reach * C.EVAL_EVERY
            loss = intask['online_loss'][i]
            spike = any(float(np.mean(loss[u:u + C.EVAL_EVERY])) > limit for u in range(u90, steps, C.EVAL_EVERY))
        out.append(dict(task=int(task), spike=bool(spike), breakdown=bool(spike and int(mem[-1]) * 2 < n_images)))
    return out


def spike_records(root, runs, j1s):
    table = {}
    for arm in C.ARMS:
        for seed in sorted(runs[arm]):
            path = Path(root) / arm / f'seed{seed}' / 'intask.npz'
            if not path.exists():
                continue
            rows = runs[arm][seed]['tasks']
            first = rows[min(rows)]
            n_labels = len(json.loads(first['loss_rows']))
            with np.load(path) as f:
                record = {k: f[k] for k in f.files}
            per_task = spikes_of(record, n_labels, int(float(first['steps'])))
            variant = spikes_of(record, n_labels, int(float(first['steps'])), reach_percent=80)
            spikes = [r['task'] for r in per_task if r['spike']]
            breaks = [r['task'] for r in per_task if r['breakdown']]
            spikes80 = [r['task'] for r in variant if r['spike']]
            breaks80 = [r['task'] for r in variant if r['breakdown']]
            late_unreached = [t for t in LATE if t in rows and num(rows[t]['T90']) > num(rows[t]['steps'])]
            after_break = [t for t in late_unreached if breaks and t > breaks[0]]
            table.setdefault(arm, {})[seed] = dict(
                spike_tasks=len(spikes), first_spike=spikes[0] if spikes else None,
                breakdown_tasks=len(breaks), first_breakdown=breaks[0] if breaks else None,
                late_breakdowns=sum(1 for t in breaks if t in LATE),
                late_unreached=len(late_unreached), late_unreached_after_first_breakdown=len(after_break),
                share_of_loss_after_breakdown=(len(after_break) / len(late_unreached)) if late_unreached else None,
                arm_J1=j1s[arm]['label'], spike_tasks_reach80=len(spikes80),
                first_spike_reach80=spikes80[0] if spikes80 else None, breakdown_tasks_reach80=len(breaks80),
                first_breakdown_reach80=breaks80[0] if breaks80 else None)
    return table


def fresh_records(runs, fresh):
    rows = []
    for fresh_arm, by_seed in fresh.items():
        for seed, tasks in sorted(by_seed.items()):
            for task, f in sorted(tasks.items()):
                rows.append(_fresh_row(runs, fresh_arm, seed, task, f))
    return rows


def _fresh_row(runs, fresh_arm, seed, task, f):
    if True:
        if True:
            entry = dict(fresh_arm=fresh_arm, seed=seed, task=task, fresh_online=num(f['online_acc']),
                         fresh_online_first100=num(f['online_acc_first100']), fresh_T90=num(f['T90']))
            for arm in [a for a in C.ARMS if C.FRESH_FOR[a] == fresh_arm]:
                run = runs[arm].get(seed)
                if run and task in run['tasks']:
                    row = run['tasks'][task]
                    entry[f'{arm}_online'] = num(row['online_acc'])
                    entry[f'{arm}_T90'] = num(row['T90'])
                    entry[f'{arm}_delta_fresh'] = entry['fresh_online'] - num(row['online_acc'])
                    entry[f'{arm}_delta_fresh_first100'] = (entry['fresh_online_first100']
                                                            - num(row['online_acc_first100']))
            return entry


def judge(root):
    runs, fresh = load(root)
    # Main judgement moved to the LN comparison (Issa 1024): J0 = SL - R0L, then JLN.
    J1 = {arm: j1(runs, arm) for arm in C.ARMS}
    result = dict(J0={'LN': j0(runs, 'R0L', 'SL'), 'noLN': j0(runs, 'R0', 'S')}, J1=J1,
                  J3={arm: j3(runs, arm) for arm in C.ARMS}, JLN=dict(label=jln(J1['R0L'], J1['SL'])),
                  J5=dict(label='PENDING', note='post-hoc on saved states (spec §7.4), not computed here'),
                  fresh=fresh_records(runs, fresh), spikes=spike_records(root, runs, J1),
                  diverged={arm: {seed: run['done'] for seed, run in runs[arm].items()
                                  if run['done'].get('status') == 'diverged'} for arm in C.ARMS})
    outcome = {'J0 noLN': result['J0']['noLN']['label'], 'J0 LN': result['J0']['LN']['label'],
               'JLN': result['JLN']['label'], 'J5 S': 'PENDING', 'J5 SL': 'PENDING'}
    outcome.update({f'J1 {arm}': result['J1'][arm]['label'] for arm in C.ARMS})
    outcome.update({f'J3 {arm}': result['J3'][arm]['label'] for arm in C.ARMS})
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
    fmt = lambda v: '—' if v is None else (f'{v:+.4f}' if isinstance(v, float) else str(v))
    for cond, (reset, shared) in (('LN', ('R0L', 'SL')), ('noLN', ('R0', 'S'))):
        j0r = result['J0'][cond]
        role = '主の判定' if cond == 'LN' else '記録'
        lines += [f"## J0 {cond}（{shared} − {reset}、{role}）: `{j0r['label']}`（使える対 {j0r['usable']}、負 {j0r['negative']}、正 {j0r['positive']}）", '',
                  f'| seed | Δ（後期） | {reset} 後期 | {shared} 後期 | 始まり {reset} | 始まり {shared} |', '| --- | --- | --- | --- | --- | --- |']
        for seed, e in j0r['seeds'].items():
            lines.append(f"| {seed} | {fmt(e.get('delta_late'))} | {fmt(e.get('late_R0'))} | {fmt(e.get('late_S'))} | "
                         f"{e['onset_R0'] or 'なし'} | {e['onset_S'] or 'なし'} |")
        lines.append('')
    lines += [f"## JLN（主の判定の次）: `{result['JLN']['label']}`（R0L の J1 = {result['J1']['R0L']['label']}、SL の J1 = {result['J1']['SL']['label']}）"]
    if result.get('spikes'):
        lines += ['', '## 跳ね上がりと崩れ（記録）', '', '| 腕 | seed | 跳ね上がりの課題 | 最初 | 崩れの課題 | 最初 | 後期の未到達 | そのうち最初の崩れの後 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for arm, by_seed in result['spikes'].items():
            for seed, v in by_seed.items():
                lines.append(f"| {arm} | {seed} | {v['spike_tasks']} | {v['first_spike'] or '—'} | {v['breakdown_tasks']} | "
                             f"{v['first_breakdown'] or '—'} | {v['late_unreached']} | {v['late_unreached_after_first_breakdown']} |")
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
        lines += ['', '## fresh との比較（記録）', '', '| fresh | seed | 課題 | fresh の online | 戻す腕 | 共有の腕 | Δ_fresh 戻す | Δ_fresh 共有 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for e in result['fresh']:
            g = lambda k: '—' if e.get(k) is None or (isinstance(e.get(k), float) and not math.isfinite(e[k])) else f'{e[k]:.4f}'
            reset, shared = ('R0', 'S') if e['fresh_arm'] == 'fresh' else ('R0L', 'SL')
            lines.append(f"| {e['fresh_arm']} | {e['seed']} | {e['task']} | {g('fresh_online')} | {g(reset + '_online')} | "
                         f"{g(shared + '_online')} | {g(reset + '_delta_fresh')} | {g(shared + '_delta_fresh')} |")
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
