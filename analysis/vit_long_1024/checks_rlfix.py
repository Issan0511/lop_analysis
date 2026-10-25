"""Checks of addendum D (specs/spec_vit_long_1024.md 追補 D, D.5) before the run.

T1  Under production_2blk_t100.json (no fixed_images) task_plan is unchanged: tasks 1-100 equal
    the plan recorded in addendum B's provenance.json.
T2  Under production_2blk_rlfix.json (its seed) every task has the same ten classes, task 101's, and
    images 251-500 of each class (for seed 0: addendum C's N = 2,500 set), labels follow
    ('randlabel', seed, task) and differ between tasks, and S and R0 use the same ten loss rows.
T3  Smoke runs of run.py (S and R0, 2 tasks x 100 updates, real data): R0 zeroes the head at both
    tasks and S only at task 1, the loss rows are the fixed classes, task 2 starts at ln 10 for R0
    and above it for S; timing per update and per evaluation.

    python -m analysis.vit_long_1024.checks_rlfix
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
RAW = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix'
OUT = REPO / 'results/vit_long_1024_2blk_rlfix'
B_PROVENANCE = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs/S/seed0/provenance.json'


def env(production):
    return dict(os.environ, PYTHONPATH=str(REPO), VIT_LONG_PRODUCTION=production, CUBLAS_WORKSPACE_CONFIG=':4096:8')


def child(production, code):
    out = subprocess.run([PY, '-c', code], cwd=REPO, env=env(production), check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


T1 = r'''
import json
from pathlib import Path
from analysis.vit_long_1024 import common as C
ref = json.loads(Path(%r).read_text())['plan']
bad = [t for t in range(1, 101) if C.task_plan(0, t, 10, 25) != ref[str(t)]]
print(json.dumps(dict(tasks=100, mismatched=bad, fixed_key=C._PRODUCTION.get('fixed_images'))))
''' % str(B_PROVENANCE)

T2 = r'''
import json, torch
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.capacity import capacity_task
from analysis.joudaki_vit_battle_0919.model import stream_seed
seed = C.production()['seeds'][0]
src = C.image_source('real')
plans = [C.task_plan(seed, t, 10, 250) for t in range(1, 101)]
same_plan = all({k: v for k, v in p.items() if k != 'task'} == {k: v for k, v in plans[0].items() if k != 'task'}
                for p in plans)
d1, _ = C.task_dataset(src, seed, 1, 10, 250, augment=False)
d7, _ = C.task_dataset(src, seed, 7, 10, 250, augment=False)
task101 = list(C.cycle_classes(seed, 5, 10)[0])        # task 101 = cycle 5, position 0 (20 tasks per cycle)
expected_paths = [str(q.relative_to(src.root)) for cls in task101 for q in src.train_paths(cls)[250:500]]
labels = [C.task_labels(2500, seed, t, 10) for t in (1, 2, 3)]
expect = [torch.randint(0, 10, (2500,), generator=torch.Generator().manual_seed(stream_seed('randlabel', seed, t)))
          for t in (1, 2, 3)]
cap_equal = None
if seed == 0:
    cap, cap_classes = capacity_task(src, 2500)
    cap_equal = plans[0]['classes'] == cap_classes and d1.paths == cap.paths
print(json.dumps(dict(
    seed=seed, same_plan_all_tasks=same_plan, classes=plans[0]['classes'], first=plans[0]['first'],
    count=plans[0]['count'], classes_equal_task101=plans[0]['classes'] == task101,
    paths_are_images_251_500=d1.paths == expected_paths, equal_to_addendum_C_set=cap_equal,
    images_task1_equal_task7=bool((d1.images == d7.images).all()), n_images=len(d1),
    labels_follow_stream=all(bool(torch.equal(a, b)) for a, b in zip(labels, expect)),
    labels_differ=[float((labels[0] == labels[1]).double().mean()), float((labels[1] == labels[2]).double().mean())],
    task1_labels_equal_dataset=bool(torch.equal(d1.labels, labels[0])),
    loss_rows_S=C.loss_classes('S', seed, 5, 10, 250), loss_rows_R0=C.loss_classes('R0', seed, 5, 10, 250),
    first_epoch_sizes=C.first_epoch_sizes(2500))))
'''


def smoke(arm, seed):
    out = RAW / '_checks' / f'smoke_{arm}'
    if out.exists():
        subprocess.run(['rm', '-rf', str(out)], check=True)
    cmd = [PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--seed', str(seed), '--tasks', '2', '--steps', '100',
           '--smoke', '--out', str(out), '--n-labels', '10', '--images-per-class', '250', '--augment', 'off',
           '--dropout', '0.0', '--depth', '2', '--mlp-ratio', '8.0']
    subprocess.run(cmd, cwd=REPO, env=env('production_2blk_rlfix.json'), check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(open(out / 'per_task.csv')))
    return dict(head_zeroed=[int(r['head_zeroed']) for r in rows], loss_rows=[json.loads(r['loss_rows']) for r in rows],
                classes=[json.loads(r['classes']) for r in rows],
                mem_loss_start=[float(r['mem_loss_start']) for r in rows], T90=[r['T90'] for r in rows],
                ms_per_update=[1000 * float(r['train_seconds']) / 100 for r in rows],
                record_seconds=[float(r['record_seconds']) for r in rows])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = child('production_2blk_t100.json', T1)
    t1['pass'] = not t1['mismatched'] and t1['fixed_key'] is None
    t2 = child('production_2blk_rlfix.json', T2)
    fixed = t2['classes']
    t2['pass'] = (t2['same_plan_all_tasks'] and t2['classes_equal_task101'] and t2['paths_are_images_251_500']
                  and t2['equal_to_addendum_C_set'] in (None, True) and t2['images_task1_equal_task7'] and t2['n_images'] == 2500 and t2['labels_follow_stream']
                  and max(t2['labels_differ']) < .2 and t2['task1_labels_equal_dataset']
                  and t2['loss_rows_S'] == fixed and t2['loss_rows_R0'] == fixed and t2['first'] == 250
                  and t2['count'] == 250 and t2['first_epoch_sizes'] == [128, 68])
    s, r = smoke('S', t2['seed']), smoke('R0', t2['seed'])
    t3 = dict(S=s, R0=r)
    t3['pass'] = (s['head_zeroed'] == [1, 0] and r['head_zeroed'] == [1, 1]
                  and all(rows == fixed for rows in s['loss_rows'] + r['loss_rows'])
                  and all(c == fixed for c in s['classes'] + r['classes'])
                  and abs(r['mem_loss_start'][1] - math.log(10)) < 1e-5
                  # zero logits give ln 10 to float32 rounding; S kept its head, so its task-2 start differs.
                  # 100 smoke updates barely grow the head, so no margin above ln 10 is implied (1025).
                  and s['mem_loss_start'][1] > r['mem_loss_start'][1])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    report = dict(git_hash=head, T1=t1, T2=t2, T3=t3, pass_=t1['pass'] and t2['pass'] and t3['pass'])
    (OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if report['pass_'] else 1)
