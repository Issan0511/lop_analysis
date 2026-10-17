#!/usr/bin/env python3
"""Exact registered RM calibration then evaluation, resumable at completed seed/arm checkpoints."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import resp_step_controls_1018 as R
from analysis.resp_step_controls_1018 import verdict as V

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/f'results/{R.RUN}/RM')
    p.add_argument('--checks',type=Path,default=ROOT/f'results/{R.RUN}/checks.json');a=p.parse_args()
    try:
        R.execute('RM','calibrate',a.out/'calibration',a.checks,None)
        rows=R.execute('RM','evaluate',a.out/'evaluation',a.checks,None,a.out/'calibration/schedule.json')
        # Report the observed RM box with registered m=15; unavailable boxes retain INCOMPLETE.
        result=V.verdict({'RM':rows});R.put(a.out/'verdict.json',result)
        (a.out/'summary.md').write_text(f"RM: {result['boxes']['RM'].get('label',result['boxes']['RM']['status'])}\nOverall: {result['status']} (RC and F51 required for the three-box claim).\n")
    except Exception as e:
        R.put(a.out/'status.json',dict(status='STOPPED',reason=str(e)));raise
    R.put(a.out/'status.json',dict(status='COMPLETE_RM_ONLY',overall_verdict_status=result['status']))

if __name__=='__main__':main()
