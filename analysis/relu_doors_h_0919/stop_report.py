#!/usr/bin/env python3
"""Report a failed reuse prerequisite; never score missing H outcomes."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/relu_doors_h_0919'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    evidence = json.loads((OUT / '_checks/S-reuse.json').read_text())
    failed = evidence['checks']['ref_R10_vs_registered_R20_raw']
    assert evidence['pass'] is False and failed['pass'] is False
    assert not (OUT / 'H').exists(), 'this report is only for a STOP before H training'
    # Independently re-enumerate all differences rather than copying the first 12.
    paths = [OUT / '_checks/runs' / name / 'per_task.csv'
             for name in ('current_ref_R10', 'frozen_ref_R20')]
    frames = []
    for p in paths:
        df = pd.read_csv(p, float_precision='round_trip')
        df = df[(df.cond == 'raw') & df.task.isin([1, 2])].set_index(['seed', 'task']).sort_index()
        assert len(df) == 20 and df.index.is_unique
        frames.append(df)
    differences = []
    for key in frames[0].index:
        for col in failed['columns']:
            if col in ('seed', 'task'):
                continue
            left, right = (float(df.loc[key, col]) for df in frames)
            if left != right and not (pd.isna(left) and pd.isna(right)):
                differences.append({'seed': int(key[0]), 'task': int(key[1]), 'column': col,
                                    'R10': left, 'R20': right, 'difference': left-right})
    assert len(differences) == failed['mismatch_count'] == 193
    pd.DataFrame(differences).to_csv(OUT / 'reuse_differences.csv', index=False)
    labels = {key: 'INAPPLICABLE' for key in ('M_H', 'N_C', 'R_LAYER', 'D_HR')}
    verdict = {'run_id': 'relu_doors_h_0919', 'status': 'STOP', 'labels': labels,
               'reason': evidence['reason'], 'H_main_run': 'NOT_RUN',
               'H_outcomes_generated': False, 'all_required_checks_passed': False,
               'independent_audit': False, 'predictions_scored': False,
               'prereg_commit': evidence['provenance']['prereg_commit'],
               'qualification_commit': evidence['provenance']['git_hash'],
               'compared_values': failed['compared_values'], 'mismatch_count': len(differences),
               'first_difference': differences[0], 'remaining_checks': 'NOT_RUN',
               'reuse_evidence_sha256': digest(OUT / '_checks/S-reuse.json')}
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2, allow_nan=False) + '\n')
    pd.DataFrame([{'key': key, 'label': value} for key,value in labels.items()]).to_csv(OUT / 'verdict.csv', index=False)
    pd.DataFrame([{'person': person, 'key': key, 'prediction': pred,
                   'probability': probability, 'score_status': 'NOT_SCORED', 'hit': None}
                  for person, probs in [('Issa', [None,None,None]), ('Codex', [.8,.8,.7])]
                  for key, pred, probability in zip(('M_H','N_C','R_LAYER'),
                     ('H_ALONE_FAILS','C_NEEDED','LAYER_CORRESPONDENCE'), probs)]).to_csv(OUT / 'prediction_score.csv', index=False)
    refs = ['results/rlcifar_mlp_battle_0918/R', 'results/relu_doors_0919/C', 'results/relu_doors_0919/CH']
    manifest = {'eligible_for_reuse': False, 'reason': verdict['reason'], 'references': []}
    for folder in refs:
        p = ROOT / folder
        prov = json.loads((p / 'provenance.json').read_text())
        manifest['references'].append({'path': folder, 'csv_sha256': digest(p / 'per_task.csv'),
            'provenance_sha256': digest(p / 'provenance.json'), 'launch_commit': prov['git_hash'],
            'data_sha256': prov['data_sha256'], 'subset_sha256': prov['subset_sha256']})
    costs = {}
    for name in ('frozen_ref_R20', 'current_ref_R20', 'current_ref_R10'):
        p = OUT / '_checks/runs' / name
        prov = json.loads((p / 'provenance.json').read_text())
        assert prov['data_sha256'] == manifest['references'][0]['data_sha256']
        assert prov['subset_sha256'] == manifest['references'][0]['subset_sha256']
        assert prov['seeds'] == list(range(10)) and prov['epochs_per_task'] == 400
        costs[name] = {'wall_clock_s': prov['wall_clock_s'], 'R': prov['R'],
                       'per_task_sha256': digest(p / 'per_task.csv')}
    manifest['qualification_runs'] = costs
    (OUT / 'reuse_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    seconds = sum(r['wall_clock_s'] for r in costs.values())
    (OUT / 'summary.md').write_text(f'''# relu_doors_h_0919 — 再利用検査で停止

**判定: INAPPLICABLE。H の本走は未実施。** Issa の予測・GOを記録した登録 commit
`{verdict['prereg_commit']}` の §6.3・§7・§8 に従い、refの再利用資格が不成立のため停止した。
「H単独は救えない」「Cが必要」「層別対応が成立」のどれも、この走からは判定できない。
Issa/Codexの予測は未採点。検査不合格を H_ALONE_FAILS に数えていない。

## 再利用検査

| 比較 | 結果 |
|---|---|
| 登録済みref/C/CH CSVのSHA-256 | 全て一致 |
| ref旧起動commitで再計算したR=20/rawと登録済みCSV（seed0–9、task1–2、400epochs） | 460/460項目一致 |
| 新実装R=20/rawと旧コードR=20/raw | 460/460項目一致 |
| 同じR=20での最終重み・Adam m/v/t・RNG・alive | 19,205,741要素一致 |
| 新実装R=10/rawと旧R=20のraw | **460項目中193項目不一致** |

460項目にはseed・task・lrを含む23列×20行が入る。CSV比較は親と同じ出力精度で保存した
数値の厳密一致で、許容誤差は0。状態比較は保存Tensorの厳密一致。
seed・task対応と比較数を別の報告器で再確認し、全不一致を `reuse_differences.csv` に保存した。
最初の例はseed0、task1のonline精度: R=10が **0.262475**、元のR=20が **0.22545625**。
一括計算の枠数を変更すると軌道が変わったことは確認したが、特定のCUDAカーネルを原因とする
診断までは実施していない。元配置では一致するので、旧結果の破損とは判定していない。

比較器はCSVの1 ULP改変、seedずれ、stdの誤選択を同じ述語で検出した。
C/CHの再計算、C→H変異、および残りの必須検査はこの前提不成立で未実施。
S-off/S-reuse全体のPASSとは主張しない。Hのseed0–9も検査seed200–209も生成していない。

## 実行と開示

- 検査実装commit: `{verdict['qualification_commit']}`。起動時src/analysisはclean。
- 検査開始: 2026-09-19 23:53:51 JST。終了: 同日23:58:40 JST付近、exit=2（登録済みの停止）。
- GPU: RTX 5060 Ti、torch 2.13.0+cu130、float32、CPU threads=2、CUDA graph。
- 3本を直列実行。各2tasks×400epochs。実測学習・保存時間の合計は **{seconds:.3f}秒**。
  frozenR20/newR20/newR10の順であり、新しい50taskの参照本走ではない。
- 起動時provenanceと入力データ・subsetのSHA-256を保存・照合。旧ホスト依存コードも旧commitとbyte一致。
- 最初のbackground起動はプロセスも出力も生成せず終了し、生存確認で検出。
  foregroundの管理sessionからnohupで起動し直したものだけが検査を実行した。学習の再試行ではない。
- 進捗確認で既知refの検査走task1/2の集約精度を読んだ。Hの途中結果は存在せず、判定・予測は変更していない。
- 独立監査なし。単一Codexの実装・自己検査・報告器による再照合。
- 計画上はchecks/report/launcherを実装してから一括検査だったが、費用節約のため、まず再利用前提を
  fail-closedの段階検査として実装・実行した。前提で停止したため、H用の残りの検査・本走launcherは未実装。
  これは実行順の変更で、腕・seed・判定帯・停止規則の変更ではない。

## 次に必要な判断

この登録のまま許容誤差を広げたり、参照を差し替えたりしない。
続行する場合は、refもR=10で新規に走らせる比較を**新しいrun id・別登録**として設計する。
その案の費用と再利用条件を定めるまで、Hの本走は停止。
生データの所在は `backup_manifest.json`。数値と判定の正本は本summaryと `verdict.json`。
''')
    print(json.dumps({'status': verdict['status'], 'labels': labels, 'mismatches': len(differences)}))

if __name__ == '__main__':
    main()
