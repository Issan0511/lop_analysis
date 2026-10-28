"""Serial authorized one-task replays. Each child exit is checked before next."""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0')
BASE=REPO/'analysis/vit_long_1024/posthoc_2blk'


def run(script,args,tag):
    cmd=[sys.executable,'-u',str(BASE/script),*args]
    with (OUT/f'{tag}.log').open('x') as log:
        print(datetime.now().isoformat(),tag,flush=True)
        result=subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{tag} failed: {result.returncode}')


def main():
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    run('codex_r0_probe.py',[],'codex_linear_probes')
    jobs=[('R0',10,'base',True)]
    # Start with the informative collapse boundary and late-state interventions.
    for arm,task in [('R0',30),('R0',60),('R0',40),('R0',20),('S',30),('S',60)]:
        for mode in ['base','lr01','lr001','scale01']:
            jobs.append((arm,task,mode,mode=='base' and arm=='R0' and task in [30,60]))
    for arm,task,mode,full in jobs:
        args=[arm,str(task),mode]
        if full:args+=['--full']
        run('codex_r0_replay.py',args,f'codex_replay_{arm}_t{task}_{mode}')
    print('ALL COMPLETE',flush=True)


if __name__=='__main__':main()
