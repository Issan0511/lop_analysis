"""Post-hoc direction-only second-moment replay, registered d4bb3c3.

Uses the exact existing task50 checkpoints and source-faithful native runner.
For each F unit, keep its replay-preconditioned proposal direction, match the
actual natural-C joint (W1,b1) step norm, and audit float32 rounding explicitly.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('direction_native_runner', Path(__file__).with_name('real_interventions.py'))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
SOURCE_SHA = R.sha(__file__)
RUNNER_SHA = R.RUNNER_SHA
H = R.H


def load(seed, out):
    with np.load(out / f'real_s{seed}_t50.npz') as d:
        X = torch.tensor(d['X'], dtype=torch.float32)
        P = [torch.tensor(d[name], dtype=torch.float32, requires_grad=True) for name in R.NAMES]
        opt = R.native_optimizer(P)
        for name, param in zip(R.NAMES, P):
            opt.state[param] = {'step': torch.tensor(float(d[f'step_{name}'])), 'exp_avg': torch.tensor(d[f'm_{name}']), 'exp_avg_sq': torch.tensor(d[f'q_{name}'])}
    with np.load(out / f'real_s{seed}_task_plan.npz') as d:
        plans = {}
        for task in (51, 52, 53):
            y = torch.tensor(d[f'y_t{task}'], dtype=torch.int64)
            plans[task] = (y, torch.nn.functional.one_hot(y, R.K).to(torch.float32), torch.tensor(d[f'order_t{task}'], dtype=torch.int64))
    return X, P, opt, plans


def hidden_old(arm):
    return [arm['P'][j].detach().clone() for j in (0, 1)]


@torch.no_grad()
def joint_norm_difference(P, old):
    dw = P[0].double() - old[0].double(); db = P[1].double() - old[1].double()
    return torch.sqrt((dw * dw).sum(1) + db * db)


@torch.no_grad()
def impose_length(arm, old, target):
    P = arm['P']
    dw = P[0].double() - old[0].double(); db = P[1].double() - old[1].double()
    proposed = torch.sqrt((dw * dw).sum(1) + db * db)
    positive = proposed > 0
    factor = torch.zeros_like(proposed)
    factor[positive] = target[positive] / proposed[positive]
    dw.mul_(factor[:, None]); db.mul_(factor)
    ideal_length = torch.sqrt((dw * dw).sum(1) + db * db)
    ideal_w = old[0].double() + dw; ideal_b = old[1].double() + db
    P[0].copy_(ideal_w.float()); P[1].copy_(ideal_b.float())
    actual = joint_norm_difference(P, old)
    rounding_w = P[0].double() - ideal_w; rounding_b = P[1].double() - ideal_b
    rounding_bound = torch.sqrt((rounding_w * rounding_w).sum(1) + rounding_b * rounding_b)
    missing = (~positive) & (target > 0)
    # Reverse triangle inequality: deviation from the ideal norm cannot exceed
    # the vector roundoff norm. No artificial direction is created at zero.
    bound_excess = (actual - ideal_length).abs() - rounding_bound
    assert float(bound_excess.max()) <= 1e-11
    assert bool(torch.all(torch.isfinite(actual)))
    return {key: value.detach().numpy().copy() for key, value in {
        'target_length': target, 'proposal_length': proposed, 'actual_minus_target': actual - target,
        'ideal_minus_target': ideal_length - target, 'rounding_bound': rounding_bound,
        'zero_proposal_nonzero_target': missing, 'zero_target': target == 0,
        'bound_excess': bound_excess}.items()}


def setup(seed, out):
    X, P, opt, plans = load(seed, out)
    ref = R.clone_arm(P, opt, 'REF', 1.)
    C = {gain: R.clone_arm(P, opt, 'C', gain) for gain in (.1, 10.)}
    F = {}
    for gain in (.1, 10.):
        arm = R.clone_arm(P, opt, 'C', gain); arm['mode'] = 'F'; arm['tag'] = arm['tag'].replace('C_', 'F_')
        F[gain] = arm
    return X, P, plans, ref, C, F


def step_all(X, Y, idx, ref, C, F):
    xb, yb = X[idx], Y[idx]
    R.native_step(ref['P'], ref['optim'], xb, yb)
    sq = [ref['optim'].state[ref['P'][j]]['exp_avg_sq'].detach() for j in (0, 1)]
    checks = {}
    for gain in (.1, 10.):
        old_c = hidden_old(C[gain])
        R.native_step(C[gain]['P'], C[gain]['optim'], xb, yb)
        target = joint_norm_difference(C[gain]['P'], old_c)
        old_f = hidden_old(F[gain])
        R.native_step(F[gain]['P'], F[gain]['optim'], xb, yb, replay_second=sq, scale=gain)
        checks[gain] = impose_length(F[gain], old_f, target)
    return checks


def pilot(out):
    X, P, plans, ref, C, F = setup(0, out)
    _, Y, order = plans[51]
    lengths = step_all(X, Y, order[0], ref, C, F)
    checks = {}
    with np.load(out / 'real_s0_t51_step1.npz') as saved:
        for gain in (.1, 10.):
            tag = C[gain]['tag']
            bit_equal = {name: bool(np.array_equal(C[gain]['P'][j].detach().numpy(), saved[f'{tag}__{name}'])) for j, name in enumerate(R.NAMES)}
            assert all(bit_equal.values())
            q = lengths[gain]; valid = ~q['zero_proposal_nonzero_target']
            checks[str(gain)] = {'C_first_step_bit_equal': bit_equal,
                                 'max_ideal_length_error_when_nonzero': float(np.max(abs(q['ideal_minus_target'][valid]))),
                                 'max_actual_length_error': float(np.max(abs(q['actual_minus_target']))),
                                 'max_rounding_bound_excess': float(np.max(q['bound_excess'])),
                                 'zero_proposal_nonzero_target_count': int(q['zero_proposal_nonzero_target'].sum())}
            assert checks[str(gain)]['max_ideal_length_error_when_nonzero'] < 1e-12
    checks.update(source_sha256=SOURCE_SHA, native_runner_sha256=RUNNER_SHA, registration_commit='d4bb3c3')
    (out / 'real_direction_pilot.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks), flush=True)


def run(seed, out):
    start = time.perf_counter()
    X, P0, plans, ref, C, F = setup(seed, out); X64 = X.double()
    arms = [ref, C[.1], C[10.], F[.1], F[10.]]
    length_records = {gain: {} for gain in (.1, 10.)}
    summary, units, contrast = [], [], []
    frozen = {a['tag']: a['P'][2].detach().clone() for a in arms}
    for task in (51, 52, 53):
        _, Y, order = plans[task]
        for it in range(R.STEPS):
            check = step_all(X, Y, order[it], ref, C, F)
            for gain, record in check.items():
                for key, value in record.items(): length_records[gain].setdefault(key, []).append(value)
        byarm = {}
        for arm in arms:
            unit, ce, acc = R.diagnostics(arm['P'], X64, Y.double()); byarm[arm['tag']] = unit
            with torch.no_grad():
                dw = arm['P'][0].double() - P0[0].double(); db = arm['P'][1].double() - P0[1].double()
                displacement = torch.sqrt((dw * dw).sum(1) + db * db).numpy()
            summary.append({'seed': seed, 'task': task, 'step': 4000, 'arm': arm['tag'], 'policy': arm['mode'], 'gain': arm['scale'],
                            'median_T_all': float(np.median(unit['T'])), 'median_raw_top_all': float(np.median(unit['top'])),
                            'median_s_all': float(np.median(unit['s'])), 'mean_pplus_all': float(np.mean(unit['pplus'])),
                            'ce': ce, 'accuracy': acc, 'median_wnorm': float(np.median(unit['wnorm'])),
                            'median_initial_hidden_displacement': float(np.median(displacement)),
                            'max_initial_hidden_displacement': float(np.max(displacement))})
            for i in range(H): units.append({'seed': seed, 'task': task, 'step': 4000, 'arm': arm['tag'], 'unit': i, **{k: int(v[i]) if k.endswith('_id') else float(v[i]) for k, v in unit.items()}})
        for mode in ('C', 'F'):
            d = byarm[f'{mode}_c0p1']['T'] - byarm[f'{mode}_c10p0']['T']
            contrast.append({'seed': seed, 'task': task, 'step': 4000, 'policy': mode, 'paired_median_T_small_minus_large': float(np.median(d)),
                             'paired_mean_T_small_minus_large': float(np.mean(d)), 'fraction_T_small_greater': float(np.mean(d > 0)),
                             'paired_median_raw_top_small_minus_large': float(np.median(byarm[f'{mode}_c0p1']['top'] - byarm[f'{mode}_c10p0']['top']))})
        R.csv_write(out / f'real_direction_s{seed}_summary.csv', summary)
        R.csv_write(out / f'real_direction_s{seed}_units.csv', units)
        R.csv_write(out / f'real_direction_s{seed}_contrasts.csv', contrast)
        print(json.dumps({'event': 'direction_task_complete', 'seed': seed, 'task': task, 'seconds': time.perf_counter() - start,
                          'contrasts': contrast[-2:]}), flush=True)
    states = {'arm_names': np.array([a['tag'] for a in arms]), 'gains': np.array([a['scale'] for a in arms])}
    for arm in arms: states.update({f'{arm["tag"]}__{k}': v for k, v in R.snapshot(arm['P'], arm['optim']).items()})
    np.savez_compressed(out / f'real_direction_s{seed}_t53_states.npz', **states)
    match = {}
    with np.load(out / f'real_s{seed}_t53_states.npz') as old:
        for arm in (ref, C[.1], C[10.]):
            match[arm['tag']] = {key: bool(np.array_equal(value, old[f'{arm["tag"]}__{key}'])) for key, value in R.snapshot(arm['P'], arm['optim']).items()}
            assert all(match[arm['tag']].values())
    checks = []
    for gain, history in length_records.items():
        record = {k: np.stack(v) for k, v in history.items()}
        np.savez_compressed(out / f'real_direction_s{seed}_F_c{str(gain).replace(".", "p")}_lengths.npz', **record)
        target = record['target_length']; actual_error = record['actual_minus_target']; valid = target > 1e-6
        nonzero_proposal = ~record['zero_proposal_nonzero_target']
        checks.append({'seed': seed, 'gain': gain, 'unit_steps': int(target.size),
                       'zero_target_count': int(record['zero_target'].sum()),
                       'zero_proposal_nonzero_target_count': int(record['zero_proposal_nonzero_target'].sum()),
                       'max_ideal_error_when_nonzero': float(np.max(abs(record['ideal_minus_target'][nonzero_proposal]))),
                       'max_actual_length_error': float(np.max(abs(actual_error))),
                       'mean_actual_length_error': float(np.mean(abs(actual_error))),
                       'max_actual_relative_error_target_gt_1e_minus6': float(np.max(abs(actual_error[valid]) / target[valid])),
                       'max_rounding_bound_excess': float(record['bound_excess'].max())})
    assert all(torch.equal(a['P'][2], frozen[a['tag']]) for a in arms)
    R.csv_write(out / f'real_direction_s{seed}_length_checks.csv', checks)
    meta = {'seed': seed, 'seconds': time.perf_counter() - start, 'source_sha256': SOURCE_SHA, 'native_runner_sha256': RUNNER_SHA,
            'registration_commit': 'd4bb3c3', 'reference_and_C_bit_equal_to_original': match,
            'frozen_readouts_unchanged': True, 'all_length_rounding_bounds_verified': True,
            'task50_sha256': R.sha(out / f'real_s{seed}_t50.npz'), 'task_plan_sha256': R.sha(out / f'real_s{seed}_task_plan.npz')}
    (out / f'real_direction_s{seed}_provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'event': 'direction_complete', 'seed': seed, 'seconds': meta['seconds'], 'length_checks': checks}), flush=True)


def summarize(out):
    audit, paired = [], []
    calibration = []
    lines = ['# 二次moment提案の方向と一歩の長さを分ける最終対照', '',
             'post-hoc登録d4bb3c3、実行コードf11cf4d。元task50の同一状態から2seed・task51–53のみ。追加の学習は完了し、この対照後の探索は行わない。', '',
             'FはF自身の状態・自然な残差から一次momentを更新し、二次momentだけ基準c=1の同時刻の値をc²倍して置く。その提案の向きを保ちながら、各unitのjoint (Δw,Δb) のEuclidean長さを、同gain・同stepの自然Cの実際の一歩と揃える。b2は自然なAdam、momentはパラメータ変位の長さ補正では変更しない。既存D軌道の方向を再生する操作ではない。', '',
             '## 同一unitの主対比', '', '| seed | task | C | F |', '|---|---|---|---|']
    geometry = []
    for seed in (0, 1):
        with (out / f'real_direction_s{seed}_contrasts.csv').open() as f: contrasts = list(csv.DictReader(f))
        with (out / f'real_direction_s{seed}_summary.csv').open() as f: summ = list(csv.DictReader(f))
        with (out / f'real_direction_s{seed}_units.csv').open() as f: units = list(csv.DictReader(f))
        for task in (51, 52, 53):
            values = {r['policy']: float(r['paired_median_T_small_minus_large']) for r in contrasts if int(r['task']) == task}
            paired.append({'seed': seed, 'task': task, **values})
            lines.append(f'| {seed} | {task} | {values["C"]:.7g} | {values["F"]:.7g} |')
        geometry.extend(r for r in summ if r['task'] == '53' and r['policy'] in ('C', 'F'))
        exceptional_units = np.zeros(H, dtype=bool)
        for gain in (.1, 10.):
            tag = f'F_c{str(gain).replace(".", "p")}'
            with np.load(out / f'real_direction_s{seed}_{tag}_lengths.npz') as d:
                tar = d['target_length']; err = d['actual_minus_target']; missing = d['zero_proposal_nonzero_target']
                exceptional_units |= np.any(missing, axis=0)
                eligible = (tar > 1e-6) & (~missing)
                rel = abs(err[eligible]) / tar[eligible]
                audit.append({'seed': seed, 'gain': gain, 'unit_steps': int(tar.size), 'zero_proposal_cases': int(missing.sum()),
                              'units_with_zero_proposal': int(np.any(missing, axis=0).sum()),
                              'missing_target_arclength_fraction': float(tar[missing].sum() / tar.sum()),
                              'max_ideal_length_error_nonzero': float(np.max(abs(d['ideal_minus_target'][~missing]))),
                              'max_actual_length_error_nonzero': float(np.max(abs(err[~missing]))),
                              'median_relative_error_eligible': float(np.median(rel)),
                              'p99_relative_error_eligible': float(np.quantile(rel, .99)),
                              'max_relative_error_eligible': float(np.max(rel)),
                              'max_rounding_bound_excess': float(d['bound_excess'].max())})
        keep = ~exceptional_units
        sensitivity = {}
        for mode in ('C', 'F'):
            small = sorted([r for r in units if r['task'] == '53' and r['arm'] == f'{mode}_c0p1'], key=lambda r: int(r['unit']))
            large = sorted([r for r in units if r['task'] == '53' and r['arm'] == f'{mode}_c10p0'], key=lambda r: int(r['unit']))
            diff = np.array([float(s['T']) - float(l['T']) for s, l in zip(small, large)])
            sensitivity[mode] = float(np.median(diff[keep]))
        meta = json.loads((out / f'real_direction_s{seed}_provenance.json').read_text())
        assert all(all(v.values()) for v in meta['reference_and_C_bit_equal_to_original'].values())
        calibration.append(f'\nseed{seed}: REF/Cのtask53全parameter・全moments・step counterは元実行とbit一致。長さを与えられなかったunitは{np.where(exceptional_units)[0].tolist()}。そのunitを両腕から除いた記述的感度解析（{int(keep.sum())}unit）ではC={sensitivity["C"]:.7g}, F={sensitivity["F"]:.7g}。これは主対比に置き換える選択ではない。\n')
        with np.load(out / f'real_direction_s{seed}_t53_states.npz') as states:
            assert all(np.all(np.isfinite(states[k])) for k in states.files if states[k].dtype.kind in 'fiu')
    lines.extend(calibration)
    R.csv_write(out / 'real_direction_paired_summary.csv', paired)
    R.csv_write(out / 'real_direction_length_audit.csv', audit)
    lines.extend(['', '## 自然な尺度からどれだけ離れたか（task53）', '',
                  '| seed | arm | CE | raw top | s | w norm | initial→end joint displacement |',
                  '|---|---|---|---|---|---|---|'])
    for r in geometry:
        lines.append('| ' + ' | '.join(r[k] if k in ('seed', 'arm') else f'{float(r[k]):.6g}' for k in ('seed', 'arm', 'ce', 'median_raw_top_all', 'median_s_all', 'median_wnorm', 'median_initial_hidden_displacement')) + ' |')
    lines.extend(['', 'Dに見られた重みノルム約1万・幅約1000への離脱はFにはない。一方、c=.1のFはCE≈2.26で、Cの約1.69–1.76より当てはめが悪い。したがって終端のTが似ていることを同じ学習状態や同じ機構の証拠にしない。', '',
                  '## 一歩の長さの監査', '',
                  '| seed | gain | zero cases / unit-steps | affected units | missing arc fraction | max ideal error | max actual error excluding zeros | median rel error | p99 rel error |',
                  '|---|---|---|---|---|---|---|---|---|'])
    for r in audit:
        cells = [str(r['seed']), str(r['gain']), f'{r["zero_proposal_cases"]}/{r["unit_steps"]}', str(r['units_with_zero_proposal'])]
        cells += [f'{r[k]:.6g}' for k in ('missing_target_arclength_fraction', 'max_ideal_length_error_nonzero', 'max_actual_length_error_nonzero', 'median_relative_error_eligible', 'p99_relative_error_eligible')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', '相対誤差はtarget>1e−6かつF提案非zeroのunit-step。ごく小さい一歩では最大相対誤差が約.19–.35に達するが、非zeroの絶対誤差は最大約1.9e−6。全stepで理想normとのずれがパラメータcastの丸めベクトルnormで抑えられることを確認した。float32の実現長さが数学的に厳密一致するとは書かない。ゼロ提案では登録どおり新しい方向を作らず停止している。全step・全unitのtarget/proposal/error/roundoff boundはreal_direction_s*_F_*_lengths.npz。', '',
                  '## 識別できた範囲', '',
                  '- unitごとの一歩の長さを自然Cへ戻すと、小vのTが高いという主対比が戻り、Dの符号反転は残らなかった。Dの反転を、二次履歴の方向成分が自然な現象を作る証拠に使えない。',
                  '- これはwの方向が無関係という結果ではない。Tの変化には方向変化が必要で、Fも自然な残差・一次履歴・ゲートを通して方向を変える。さらにFはvに依存するCの一歩の長さをそのまま与えられている。分母のv依存が一歩の長さを通して効く可能性は残る。',
                  '- joint長さの一致は||Δw||単独、wへの径方向／横方向成分、bへの配分を固定しない。これらの寄与を個別に同定した対照ではない。',
                  '- 出力biasは自由に適応し、C/Fの残差軌道・当てはめも違う。介入による媒介経路のフィードバックを含むため、単純な寄与率や唯一の機構を主張しない。',
                  '- 2seed、task51–53の同一初期状態に限定。task151–200や他設定へ外挿しない。追加学習はこれで終了し、残る符号・量を予測する縮約則が未導出であることを結論の境界にする。'])
    (out / 'real_direction_report.md').write_text('\n'.join(lines) + '\n')
    (out / 'real_direction_reporting_provenance.json').write_text(json.dumps({'report_source_sha256': SOURCE_SHA, 'status': 'read-only final summary; no added training',
        'executed_code_commit': 'f11cf4d', 'executed_code_sha_by_seed': {str(s): json.loads((out / f'real_direction_s{s}_provenance.json').read_text())['source_sha256'] for s in (0, 1)}}, indent=2) + '\n')
    print(json.dumps(paired), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--pilot', action='store_true'); p.add_argument('--seed', type=int, choices=(0, 1))
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--out', type=Path, default=Path('results/logistic_v_height_0929')); args = p.parse_args()
    torch.set_num_threads(1); torch.set_flush_denormal(True)
    if args.pilot: pilot(args.out)
    if args.summarize: summarize(args.out)
    if args.seed is not None: run(args.seed, args.out)


if __name__ == '__main__': main()
