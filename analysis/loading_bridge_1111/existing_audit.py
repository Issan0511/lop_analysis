#!/usr/bin/env python3
"""Read-only summaries of archived stage1_ctrl and ELU vfreeze data.

Run with /home/issan/Projects/claude/proj_004_drift/.venv/bin/python.
No training, writes, or imports from mutable experiment modules.
"""
from pathlib import Path
import numpy as np

RAW = Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')
VF = Path('/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/vfreeze/runs_vf')
ACTS = ('LR', 'R', 'ELU', 'SILU', 'GELU')


def dataset(act, seed, arm):
    if arm in ('S', 'Qbranch'):
        return np.load(RAW / f'branch_adam_lr0.001_{act}_s{seed}.npz'), ('Q' if arm == 'Qbranch' else 'S')
    return np.load(RAW / f'persist_{arm}_adam_lr0.001_{act}_s{seed}.npz'), arm + 'all'


def norm_table():
    print('Task30: act,arm,norm_start,norm_end,ratio_of_means,conf0')
    for act in ACTS:
        for arm in ('S', 'P', 'Q', 'H0'):
            rows = []
            for seed in range(10):
                with dataset(act, seed, arm)[0] as d:
                    key = 'S' if arm == 'S' else arm + 'all'
                    rows.append([*d[f'{key}|t30|vnorm'], float(d[f'{key}|t30|conf0'])])
            a = np.mean(rows, 0)
            ratio = a[1] / a[0] if a[0] else np.nan
            print(','.join([act, arm, *(f'{v:.6f}' for v in (a[0], a[1], ratio, a[2]))]))


def elu_trajectories():
    print('ELU: task,arm,step,med_m,med_sd,med_r,mean_abs_dphi,mean_pplus')
    for task in (2, 30):
        for arm in ('S', 'Qbranch', 'Q'):
            rows = []
            for seed in range(10):
                d, key = dataset('ELU', seed, arm)
                with d:
                    units = d[f'{key}|t{task:02d}|units']
                    steps = d[f'{key}|t{task:02d}|upts']
                    assert np.array_equal(steps, [0, 1, 10, 50, 200, 1000, 4000])
                    rows.append([[np.median(u[0]), np.median(u[1]),
                                  np.median(u[0] / np.maximum(u[1], 1e-12)),
                                  u[3].mean(), u[2].mean()] for u in units])
            for step, values in zip(steps, np.mean(rows, 0)):
                print(','.join([str(task), 'Qall' if arm == 'Q' else arm, str(step), *(f'{v:.6f}' for v in values)]))


def immediate_b1():
    print('Immediate B1: act,task,nseed,conf_S,conf_P,conf_Q,push_P_over_S,push_Q_over_S')
    for act in ACTS:
        for task in (2, 30):
            rows = []
            for seed in range(10):
                with np.load(RAW / f'branch_adam_lr0.001_{act}_s{seed}.npz') as d:
                    t = f't{task:02d}'
                    alive = d[f'S|{t}|units'][0, 2] > 0
                    if alive.sum() < 3:
                        continue
                    pushes = [d[f'{a}|{t}|push'][0, alive].sum() for a in ('S', 'P', 'Q')]
                    rows.append([*[float(d[f'{a}|{t}|conf0']) for a in ('S', 'P', 'Q')],
                                 pushes[1] / pushes[0], pushes[2] / pushes[0]])
            values = np.mean(rows, 0) if rows else np.full(5, np.nan)
            print(','.join([act, str(task), str(len(rows)), *(f'{v:.6f}' for v in values)]))


def vfreeze_push():
    print('vfreeze: arm,top_over_sd,width,k,ce,body_push_500,top_push_500')
    for arm in ('EEQ', 'VF0.1', 'VF1', 'VF10'):
        rows = []
        for seed in (0, 1):
            with np.load(VF / f'{arm}_s{seed}.npz') as d:
                for t in range(150, 200):
                    alive = d['sw_k'][t] > 0
                    rows.append([np.median((d['end_top'][t] / d['end_s'][t])[alive]),
                                 np.median(d['end_s'][t][alive]), np.median(d['end_k'][t][alive]),
                                 d['ce_end'][t], np.median((d['mid_body'][t] - d['sw_body'][t])[alive]),
                                 np.median((d['mid_top'][t] - d['sw_top'][t])[alive])])
        print(','.join([arm, *(f'{v:.6f}' for v in np.mean(rows, 0))]))


if __name__ == '__main__':
    norm_table()
    elu_trajectories()
    immediate_b1()
    vfreeze_push()
