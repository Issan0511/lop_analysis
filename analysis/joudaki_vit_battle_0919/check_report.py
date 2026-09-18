"""Synthetic fixture for registered windows and completeness/source guards."""
import csv
import json
from pathlib import Path
import tempfile

from .data import task_classes
from .model import ARM_ORDER
from .report import summarize
from .run import REPO, atomic_json


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'runs'
        for ai, arm in enumerate(ARM_ORDER):
            for seed in range(10):
                out = root / arm / f'seed{seed}'
                out.mkdir(parents=True)
                atomic_json(out / 'done.json', {'status':'complete'})
                atomic_json(out / 'provenance.json', dict(source_hashes={'fixture':'constant'},
                    classes=task_classes(seed), config=dict(arm=arm, seed=seed, smoke=False, tasks=40, steps=500, engine='compile')))
                with (out / 'per_task.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=['task','online_acc','online_global_acc','val_acc','train_acc','steps','samples'])
                    writer.writeheader()
                    for task in range(1, 41):
                        writer.writerow(dict(task=task, online_acc=.9 - ai*.02 - task*.001,
                                             online_global_acc=.85-ai*.02-task*.001,
                                             val_acc=.7, train_acc=.95, steps=500, samples=62500))
        output = Path(temporary) / 'report'
        rows = summarize(root, output)
        sna = next(r for r in rows if r['arm'] == 'SNA')
        assert abs(sna['window_median'] - (.9-.0305)) < 1e-12
        assert abs(sna['drop_median'] - .03) < 1e-12
        verdict = json.loads((output / 'verdict.json').read_text())
        assert verdict['labels']['A'] == 'SNA_TOP'
        assert all(c['p_holm'] >= c['p'] for c in verdict['comparisons'])
        target = root / 'SNA/seed0/provenance.json'
        saved = target.read_text()
        data = json.loads(saved)
        data['source_hashes'] = {'fixture':'changed'}
        atomic_json(target, data)
        try:
            summarize(root, output)
            raise AssertionError('mixed source accepted')
        except ValueError as error:
            assert 'Source hashes differ' in str(error)
        target.write_text(saved)
        (root / 'SNA/seed0/done.json').unlink()
        try:
            summarize(root, output)
            raise AssertionError('incomplete run accepted')
        except ValueError as error:
            assert 'ranking refused' in str(error)
    atomic_json(REPO / 'results/joudaki_vit_battle_0919/report_checks.json',
                dict(windows=True, paired_signs=True, holm=True, mixed_sources_refused=True,
                     incomplete_runs_refused=True, all_pass=True))
    print('report checks PASS')


if __name__ == '__main__':
    main()
