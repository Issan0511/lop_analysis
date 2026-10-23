#!/usr/bin/env python3
"""Validate completed RM with its original source, then combine with RC/F51.

The CUDA fixture fix changes the check-file hash. RM is not retrained, relabelled,
or given a forged current-source identity. Its original offline verifier runs in
a clean temporary checkout of the saved result commit.
"""
import hashlib
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BASE=Path('/home/issan/Projects/claude')
WT=BASE/'wt/resp_step_controls_1018'
PY=BASE/'proj_004_drift/.venv/bin/python'
HISTORICAL='d3d235d7e0dd3d3f34db75f23273d539c047728a'
OUT=WT/'results/resp_step_controls_1018/readout'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-rm-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='resp-rm-source-1018-') as folder:
        old=Path(folder)/'source'
        subprocess.run(['git','worktree','add','--detach',str(old),HISTORICAL],cwd=WT,check=True)
        try:
            command=[str(PY),'analysis/resp_step_controls_1018/verdict.py',
                     '--RM',str(WT/'results/resp_step_controls_1018/RM/evaluation/per_task.json'),
                     '--out',str(OUT/'rm_historical_verification')]
            subprocess.run(command,cwd=old,check=True)
            audited=json.loads((OUT/'rm_historical_verification/verdict.json').read_text())
            assert audited['scientific_evidence'] is True and audited['boxes']['RM']['status']=='COMPLETE'
            proof=dict(historical_source_commit=HISTORICAL,command=command,
                       verifier_sha256=sha(old/'analysis/resp_step_controls_1018/verdict.py'),
                       source_checker_sha256=sha(old/'analysis/resp_step_controls_1018/checks.py'),
                       rm_per_task_sha256=sha(WT/'results/resp_step_controls_1018/RM/evaluation/per_task.json'),
                       validated_by_original_source=True,
                       reason='Only the CUDA test fixture changed; completed RM remains bound to its original source and admission evidence.')
        finally:
            for cache in old.rglob('__pycache__'):
                shutil.rmtree(cache)
            subprocess.run(['git','worktree','remove',str(old)],cwd=WT,check=True)
    (OUT/'rm_historical_source_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    if args.verify_rm_only:
        print('RM original-source verification PASS');return
    sys.path.insert(0,str(WT))
    V=importlib.import_module('analysis.resp_step_controls_1018.verdict')
    inputs={b:V.load_completed_evaluation(WT/f'results/resp_step_controls_1018/{b}/evaluation/per_task.json',b)
            for b in ('RC','F51')}
    rm_path=WT/'results/resp_step_controls_1018/RM/evaluation/per_task.json'
    assert sha(rm_path)==proof['rm_per_task_sha256']
    inputs['RM']=json.loads(rm_path.read_text())
    result=V.verdict(inputs)
    assert result['boxes']['RM']==audited['boxes']['RM'], 'current and historical RM readouts differ'
    result.update(scientific_evidence=True,rm_historical_verification=proof)
    V.R.put(OUT/'verdict.json',result)
    V.R.put(OUT/'rm_historical_source_proof.json',proof)
    lines=[f"{V.R.RUN}: {result['status']}",f"Overall label: {result['label']}",
           '15 Bonferroni paired t intervals; n=10 seed pairs; equivalence margin 0.005.',
           'RM verified by its original immutable source; RC/F51 verified by current source.','']
    for box,value in result['boxes'].items():
        lines.append(f"{box}: {value['status']} {value.get('label','')}")
        for name,effect in value.get('effects',{}).items():
            lines.append(f"  {name}: {effect['mean']*100:.4f} pt; simultaneous CI [{effect['lo']*100:.4f}, {effect['hi']*100:.4f}] pt")
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(result['status'],result['label'])

if __name__=='__main__':main()
