"""R0 後期の持続崩れ: 回し直しを 1 本ずつ順に走らせる（Fable エージェント B、1028）。GPU の仕事は同時に 1 本。"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0_late')
BASE = REPO / 'analysis/vit_long_1024/posthoc_2blk'
PY = '/home/issan/Projects/claude/proj_004_drift/.venv/bin/python'


def run(script, args, tag):
    if (OUT / f'{tag}.json').exists():
        print(datetime.now().isoformat(), tag, 'skip (exists)', flush=True)
        return
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1')
    with (OUT / f'{tag}.log').open('w') as log:
        print(datetime.now().isoformat(), tag, flush=True)
        result = subprocess.run(['nice', '-n', '10', PY, '-u', str(BASE / script), *args], cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{tag} failed: {result.returncode}')


def replay(arm, task, tag, *extra):
    run('fable_r0late_replay.py', [arm, str(task), tag, *extra], tag)


def main():
    only = sys.argv[1:] or ['t60', 't30', 'bound']
    for task in (60, 30):
        if f't{task}' not in only:
            continue
        base = f'fable_R0_t{task}_hl001_base'
        state = OUT / f'{base}_t90state.pt'
        replay('R0', task, base, '--head_factor', '0.01', '--save_t90')
        for fz in ('body', 'head', 'ffn1', 'attn1', 'block0'):
            replay('R0', task, f'fable_R0_t{task}_hl001_br_{fz}', '--head_factor', '0.01', '--branch', str(state), '--freeze', fz)
        replay('R0', task, f'fable_R0_t{task}_hl001_br_bl01', '--head_factor', '0.01', '--body_factor', '0.1', '--branch', str(state))
        replay('R0', task, f'fable_R0_t{task}_hl001_br_newhead1', '--head_factor', '1.0', '--branch', str(state), '--new_head')
        replay('R0', task, f'fable_R0_t{task}_hl001_br_newhead001', '--head_factor', '0.01', '--branch', str(state), '--new_head')
        replay('R0', task, f'fable_R0_t{task}_hl001_bl001', '--head_factor', '0.01', '--body_factor', '0.01')
        if task == 60:
            replay('R0', task, f'fable_R0_t{task}_hl001_bl01', '--head_factor', '0.01', '--body_factor', '0.1')
    if 'bound' in only:
        for task in (60, 30):
            state = OUT / f'fable_R0_t{task}_hl001_base_t90state.pt'
            if state.exists():
                run('fable_r0late_bound.py', [str(state), f'fable_bound_R0_t{task}_t90'], f'fable_bound_R0_t{task}_t90')
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
