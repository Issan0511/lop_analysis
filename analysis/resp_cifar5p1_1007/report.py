#!/usr/bin/env python3
"""resp_cifar5p1_1007 registered readout (spec §5).  Explicit --src; refuses partial runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from src.resp_cifar5p1_1007 import (RUN, ARMS, DISPLAY, ROOT, REGISTRATION, write_csv, put, sha,
                                    verify_done)
from analysis.resp_cifar_ee_0920.stats import interval, label

LABELS = ('RESPONSE_BOTH_WAYS', 'SINK_ONLY', 'RESTORE_ONLY', 'RESPONSE_NOT_SHOWN', 'RESPONSE_REVERSED')
# spec §6 (registered before implementation): parent (Fable) and implementing agent (Claude)
PRED_LABEL = {'fable': {'RESPONSE_BOTH_WAYS': .50, 'SINK_ONLY': .20, 'RESTORE_ONLY': .15,
                        'RESPONSE_NOT_SHOWN': .13, 'RESPONSE_REVERSED': .02},
              'claude': {'RESPONSE_BOTH_WAYS': .70, 'SINK_ONLY': .10, 'RESTORE_ONLY': .12,
                         'RESPONSE_NOT_SHOWN': .05, 'RESPONSE_REVERSED': .03}}
PRED_BINARY = [  # metric, fable probability, claude probability
    ('applicability', .85, .95),
    ('L2_sink_exceeds_L1', .70, .85),
    ('rho1_ge_half', .50, .50),
    ('restored_below_fresh', None, .80)]
SECONDARY = [('restore_reset', 'R_chr', 'N_cr'), ('sink_keep', 'N_h', 'S_hc'),
             ('sink_L1', 'N_hr', 'S_hcL1r'), ('layer_difference', 'S_hcL1r', 'S_hcr')]


def verdict(values, initial_g, fresh=None):
    """values: arm -> 10 online accuracies (seed order); initial_g: {'h': [...], 'c': [...]} = G2
    of the natural branches; fresh: 10 fresh(t29) online accuracies (seed order)."""
    assert set(values) == set(ARMS), 'missing or extra arm'
    for name, v in values.items():
        v = np.asarray(v, dtype=np.float64)
        assert len(v) == 10 and np.isfinite(v).all(), ('incomplete/nonfinite', name)
    assert set(initial_g) == {'h', 'c'}
    for v in initial_g.values():
        assert len(v) == 10 and np.isfinite(np.asarray(v, dtype=np.float64)).all(), 'G incomplete'
    x = {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}
    gap = interval(x['N_h'] - x['N_c'])
    reset_gap = interval(x['N_hr'] - x['N_cr'])
    p1 = interval(x['R_ch'] - x['N_c'], .975)
    p2 = interval(x['N_hr'] - x['S_hcr'], .975)
    g_ok = bool(np.all(np.asarray(initial_g['h'], dtype=np.float64) > np.asarray(initial_g['c'], dtype=np.float64)))
    applicable = bool(gap['low'] > 0 and g_ok)
    result = dict(label=label(p1, p2) if applicable else 'NOT_REPRODUCED', applicable=applicable,
                  natural_gap=gap, natural_G2_all_seeds=g_ok, reset_gap=reset_gap, P1=p1, P2=p2,
                  P1_95=interval(x['R_ch'] - x['N_c']), P2_95=interval(x['N_hr'] - x['S_hcr']),
                  rho1=p1['mean'] / gap['mean'] if gap['low'] > 0 else None,
                  rho2=p2['mean'] / reset_gap['mean'] if reset_gap['low'] > 0 else None)
    result['layer_gap'] = interval(x['S_hcL1r'] - x['S_hcr'])
    for metric, a, b in SECONDARY:
        result[metric] = interval(x[a] - x[b])
    if fresh is not None:
        f = np.asarray(fresh, dtype=np.float64)
        assert len(f) == 10 and np.isfinite(f).all(), 'fresh incomplete'
        result['restored_minus_fresh'] = interval(x['R_ch'] - f)
        result['restored_reset_minus_fresh'] = interval(x['R_chr'] - f)
        result['natural_minus_fresh'] = interval(x['N_c'] - f)
    return result


def score(result):
    """Prediction scoring (spec §6): only when applicable; undefined facts are not scored."""
    rows = []
    app = result['applicable']
    facts = {'applicability': app,
             'L2_sink_exceeds_L1': result['layer_gap']['mean'] > 0 if app else None,
             'rho1_ge_half': (result['rho1'] >= .5) if (app and result['rho1'] is not None) else None,
             'restored_below_fresh': (result['restored_minus_fresh']['high'] < 0)
             if (app and 'restored_minus_fresh' in result) else None}
    for metric, pf, pc in PRED_BINARY:
        hit = facts[metric]
        for who, p in (('fable', pf), ('claude', pc)):
            if p is None:
                continue
            rows.append(dict(metric=metric, who=who, probability=p, outcome=hit,
                             binary_brier=None if hit is None else (p - float(hit)) ** 2))
    for who, probs in PRED_LABEL.items():
        if app:
            top = max(probs, key=probs.get)
            brier = sum((p - float(k == result['label'])) ** 2 for k, p in probs.items())
            rows.append(dict(metric='main_label', who=who, probability=probs[result['label']],
                             outcome=result['label'], top_label=top, top_hit=top == result['label'],
                             multiclass_brier=brier))
        else:
            rows.append(dict(metric='main_label', who=who, probability=None, outcome=result['label'],
                             top_label=max(probs, key=probs.get), top_hit=None, multiclass_brier=None))
    return rows


def load_arm(src, name, ident):
    arm = src / 'arms' / name
    verify_done(arm / 'done.json', ident)
    rows = sorted(json.loads((arm / 'rows.json').read_text()), key=lambda r: r['seed'])
    diag = json.loads((arm / 'diagnostics.json').read_text())
    pf = json.loads((arm / 'preflight.json').read_text())
    assert [r['seed'] for r in rows] == list(range(10)), ('seeds', name)
    assert all(r['arm'] == name and r['finite'] and r['task'] == ARMS[name][0] + 1 for r in rows), ('rows', name)
    assert pf['all_pass'], ('preflight', name)
    return rows, diag, pf


def report(src):
    src = Path(src)
    st = json.loads((src / 'status.json').read_text())
    assert st['stage'] == 'completed', 'partial report'
    ident = json.loads((src / 'provenance_start.json').read_text())['identity']
    assert (src / 'provenance_end.json').exists(), 'missing end provenance'
    assert ident['seeds'] == list(range(10)) and ident['steps_per_task'] == 780 and not ident['check_mode']
    verify_done(src / 'prefix' / 'done.json', ident)
    hostrec = json.loads((src / 'prefix' / 'host_record_check.json').read_text())
    assert hostrec['all_pass'], 'S-host(b) failed'
    pc = json.loads((src / 'prefix_checks.json').read_text())
    assert set(pc) == {'N_h', 'N_c'} and all(p['all_pass'] for p in pc.values()), 'S-prefix'
    checks = json.loads((src / 'admission_checks.json').read_text())
    assert checks['all_pass'] and checks['source_sha256'] == ident['source_sha256'], 'admission checks'
    fresh = sorted(json.loads((src / 'prefix' / 'fresh.json').read_text()), key=lambda r: r['seed'])
    fresh_v = [r['fresh_online_acc'] for r in fresh]
    values, armtable, per_seed, diag_rows, preflights = {}, [], [], [], {}
    initial_g = {}
    for name in ARMS:
        rows, diag, pf = load_arm(src, name, ident)
        preflights[name] = pf
        values[name] = [r['online_acc'] for r in rows]
        d0 = {r['seed']: r for r in diag if r['update'] == 0}
        dend = {r['seed']: r for r in diag if r['update'] == 780}
        for r in rows:
            per_seed.append({**r, 'fresh_t29': fresh_v[r['seed']] if r['task'] == 29 else None,
                             **{f'initial_{k}': v for k, v in d0[r['seed']].items() if k not in ('seed', 'update')},
                             **{f'final_{k}': v for k, v in dend[r['seed']].items() if k not in ('seed', 'update')}})
        for r in diag:
            diag_rows.append(dict(arm=name, display=DISPLAY[name], **r))
        if name in ('N_h', 'N_c'):
            initial_g['h' if name == 'N_h' else 'c'] = [d0[s]['G2'] for s in range(10)]
        v = np.asarray(values[name])
        armtable.append(dict(arm=name, display=DISPLAY[name], task=ARMS[name][0] + 1,
                             online_mean=float(v.mean()), online_sd=float(v.std(ddof=1)),
                             online_min=float(v.min()), online_max=float(v.max()),
                             memo_mean=float(np.mean([r['memo_acc'] for r in rows])),
                             online_ce_mean=float(np.mean([r['online_ce'] for r in rows])),
                             initial_G1=float(np.mean([d0[s]['G1'] for s in range(10)])),
                             initial_G2=float(np.mean([d0[s]['G2'] for s in range(10)])),
                             final_G1=float(np.mean([dend[s]['G1'] for s in range(10)])),
                             final_G2=float(np.mean([dend[s]['G2'] for s in range(10)])),
                             final_dead_unit2=float(np.mean([dend[s]['dead_unit2'] for s in range(10)])),
                             final_neff_fraction2=float(np.mean([dend[s]['neff_fraction2'] for s in range(10)]))))
    result = verdict(values, initial_g, fresh_v)
    result.update(independent_audit=False, n_seeds=10, registration=REGISTRATION, arms=len(ARMS),
                  initial_G2={k: v for k, v in initial_g.items()})
    paired, secondary = [], []
    comparisons = [('P1', 'R_ch', 'N_c', .975, 'PRIMARY'), ('P2', 'N_hr', 'S_hcr', .975, 'PRIMARY'),
                   ('P1_95', 'R_ch', 'N_c', .95, 'REPORT'), ('P2_95', 'N_hr', 'S_hcr', .95, 'REPORT'),
                   ('natural_gap', 'N_h', 'N_c', .95, 'APPLICABILITY'),
                   ('reset_gap', 'N_hr', 'N_cr', .95, 'REPORT_ONLY')]
    comparisons += [(m, a, b, .95, 'REPORT_ONLY') for m, a, b in SECONDARY]
    for metric, a, b, level, tier in comparisons:
        d = np.asarray(values[a]) - np.asarray(values[b])
        secondary.append(dict(metric=metric, arm_a=DISPLAY[a], arm_b=DISPLAY[b], tier=tier, **interval(d, level)))
        for s in range(10):
            paired.append(dict(metric=metric, seed=s, a=values[a][s], b=values[b][s], difference=float(d[s])))
    fresh_cmp = []
    for metric, a in (('restored_minus_fresh', 'R_ch'), ('restored_reset_minus_fresh', 'R_chr'),
                      ('natural_minus_fresh', 'N_c')):
        d = np.asarray(values[a]) - np.asarray(fresh_v)
        secondary.append(dict(metric=metric, arm_a=DISPLAY[a], arm_b='fresh(t29)', tier='REPORT_ONLY',
                              **interval(d, .95)))
        for s in range(10):
            fresh_cmp.append(dict(metric=metric, seed=s, arm=values[a][s], fresh=fresh_v[s], difference=float(d[s])))
            paired.append(dict(metric=metric, seed=s, a=values[a][s], b=fresh_v[s], difference=float(d[s])))
    predictions = score(result)
    prefix_rows = json.loads((src / 'prefix' / 'rows.json').read_text())
    write_csv(src / 'per_seed.csv', per_seed)
    write_csv(src / 'paired.csv', paired)
    write_csv(src / 'secondary.csv', secondary)
    write_csv(src / 'arm_table.csv', armtable)
    write_csv(src / 'fresh_comparison.csv', fresh_cmp)
    write_csv(src / 'diagnostics.csv', diag_rows)
    write_csv(src / 'predictions.csv', predictions)
    write_csv(src / 'per_task.csv', [dict(stage='prefix', **r) for r in prefix_rows]
              + [dict(stage='arm', **r) for r in per_seed])
    write_csv(src / 'branches.csv', [dict(arm=n, display=DISPLAY[n], branch_task=t[0], continuation_task=t[0] + 1,
                                          field_layer=t[1] or 0, donor_task=t[2] or 0, adam_reset=int(t[3]))
                                     for n, t in ARMS.items()])
    (src / 'host_record_comparison.csv').write_text((src / 'prefix' / 'host_record_comparison.csv').read_text())
    write_csv(src / 'verdict.csv', [dict(label=result['label'], applicable=result['applicable'],
                                         P1=result['P1']['mean'], P1_low=result['P1']['low'], P1_high=result['P1']['high'],
                                         P2=result['P2']['mean'], P2_low=result['P2']['low'], P2_high=result['P2']['high'],
                                         rho1=result['rho1'], rho2=result['rho2'])])
    put(src / 'verdict.json', result)
    put(src / 'checks.json', dict(pre_run=checks, host_record=hostrec, prefix=pc, preflight=preflights,
                                  all_pass=True, independent_audit=False))
    summary(src, result, armtable, secondary, predictions, hostrec, preflights, fresh_v)
    put(src / 'report_provenance.json', dict(
        git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / f'analysis/{RUN}').glob('*.py'))},
        input_identity=ident, independent_audit=False))
    print(json.dumps({k: result[k] for k in ('label', 'applicable', 'rho1', 'rho2')}, ensure_ascii=False))
    return result


def fmt(x, n=4):
    return 'undefined' if x is None else f'{x:.{n}f}'


def summary(src, result, armtable, secondary, predictions, hostrec, preflights, fresh_v):
    t = ['# 5+1 CIFAR × MLP（ReLU/std）: 第 2 層の応答の場の交換（S4 の移植）', '',
         f"登録主判定: **{result['label']}**（適用条件 {result['applicable']}）。10 seed、R=10、分岐 t_h = task 2 末 / t_c = task 28 末、"
         '継続 1 課題 × 780 更新（task 3 / task 29、どちらも hard）。登録 commit `' + REGISTRATION + '`。独立監査なし。', '',
         '| 主比較 | 平均対応差 | 97.5% 下端 | 上端 | 95% 下端 | 上端 | 符号 |', '|---|---:|---:|---:|---:|---:|---|']
    for k, k95, name in (('P1', 'P1_95', 'P1 = E(R_c←h) − E(N_c)'), ('P2', 'P2_95', 'P2 = E(N_h r) − E(S_h←c r)')):
        q, q95 = result[k], result[k95]
        t.append(f"| {name} | {q['mean']:+.4f} | {q['low']:+.4f} | {q['high']:+.4f} | {q95['low']:+.4f} | {q95['high']:+.4f} | {q['sign']}{' DEGENERATE_SD' if q['degenerate_sd'] else ''} |")
    g = result['natural_gap']
    t += ['', f"適用条件: 自然差 D = E(N_h) − E(N_c) の平均 {g['mean']:+.4f}、95% 下端 {g['low']:+.4f}；"
          f"全 10 seed で G2(t_h) > G2(t_c): {result['natural_G2_all_seeds']}。",
          f"ρ1 = {fmt(result['rho1'])}、ρ2 = {fmt(result['rho2'])}（報告のみ）。", '',
          '| 腕 | 課題 | online 平均 | SD | seed 範囲 | memo 平均 | 分岐直後 G1 | 分岐直後 G2 | 終端 G2 |',
          '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in armtable:
        t.append(f"| {r['display']} | {r['task']} | {r['online_mean']:.4f} | {r['online_sd']:.4f} | {r['online_min']:.4f}–{r['online_max']:.4f} | "
                 f"{r['memo_mean']:.4f} | {r['initial_G1']:.4f} | {r['initial_G2']:.4f} | {r['final_G2']:.4f} |")
    t += ['', f"fresh(t29)（本走の接頭部で再計算・0920 記録と byte 一致）: 平均 {np.mean(fresh_v):.4f}。", '',
          '| 副比較（REPORT_ONLY） | a | b | 平均差 | 95% 下端 | 上端 | 符号 |', '|---|---|---|---:|---:|---:|---|']
    for r in secondary:
        if r['tier'] in ('REPORT_ONLY', 'APPLICABILITY'):
            t.append(f"| {r['metric']} | {r['arm_a']} | {r['arm_b']} | {r['mean']:+.4f} | {r['low']:+.4f} | {r['high']:+.4f} | {r['sign']} |")
    t += ['', '## 予測の採点（spec §6）', '', '| 項目 | 誰 | 確率 | 結果 | Brier |', '|---|---|---:|---|---:|']
    for r in predictions:
        if r['metric'] == 'main_label':
            t.append(f"| 主ラベル（最大確率 {r['top_label']}） | {r['who']} | {fmt(r['probability'], 2)} | {r['outcome']}（最大確率ラベル的中 {r['top_hit']}） | {fmt(r['multiclass_brier'], 4)} |")
        else:
            t.append(f"| {r['metric']} | {r['who']} | {r['probability']:.2f} | {r['outcome']} | {fmt(r['binary_brier'], 4)} |")
    fr = [p for p in preflights.values() if 'argument_error_ratio' in p]
    t += ['', '## 検査', '',
          f"必須検査と変異は本走の前に all_pass（`admission_checks.json`）。S-host(b): 接頭部 task 1–29 の 0920 記録との照合 gated 不一致 {hostrec['gated_mismatches']}、"
          f"eff_rank の不一致 {hostrec['eff_rank_mismatches']} 件（相対差最大 {hostrec['eff_rank_max_relative_difference']:.3g}、報告のみ）。"
          'S-prefix: N_h / N_c の終端状態・バッチ列・行が接頭部と bit 一致。',
          f"S-branch/S-field（本走の状態で）: 場のある 5 腕で分岐時の出力が bit 一致、引数誤差/上界の最大比 {max(p['argument_error_ratio'] for p in fr):.4f}、"
          f"曖昧帯の pair 数の合計 {sum(sum(p['ambiguous_pairs_per_seed']) for p in fr)}、訓練ゲートの数の差の最大 {max(p['training_gate_count_max_difference'] for p in fr)}。", '',
          '## 読みの上限', '',
          'P1 は劣化側の Adam 履歴を継続、P2 は両腕 reset。P1 と P2 は別の課題（task 29 / task 3）の中の対応差で、絶対量の差を非対称と読まない。',
          '初期出力は自然腕と bit 一致。固定場の人工的な関数変更で、学習が進めば特徴も変わる。自然な喪失の全媒介や恒久的救済は示さない。',
          '5+1 の喪失は適応の速さの喪失で、online は床に落ちない。場は継続課題の全画像（2500 枚）で定義した（S4 の固定 1200 枚とは違う）。',
          '非検出側を効果ゼロ・同等性とは読まない。副比較・fresh 比較・ρ は REPORT_ONLY。科学的定義・分岐・seed・予測は結果の後に変更していない。']
    (src / 'summary.md').write_text('\n'.join(t) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    a = p.parse_args()
    report(Path(a.src))
