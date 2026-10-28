"""Run only after the primary serial queue has returned exit code zero."""
import subprocess
import sys
from pathlib import Path

REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0')
BASE=REPO/'analysis/vit_long_1024/posthoc_2blk'


def run(script,args,tag):
    with (OUT/f'{tag}.log').open('x') as log:
        r=subprocess.run([sys.executable,'-u',str(BASE/script),*args],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{tag}: exit {r.returncode}')
    print('finished',tag,flush=True)


def main():
    assert 'ALL COMPLETE' in (OUT/'codex_queue.log').read_text()
    run('codex_r0_radial.py',[],'codex_radial')
    run('codex_r0_ce_self.py',[],'codex_ce_self')
    for task,mode in [(30,'lr01'),(30,'lr001'),(60,'lr001')]:
        run('codex_r0_replay.py',['R0',str(task),mode,'--full','--suffix','_full'],f'codex_replay_R0_t{task}_{mode}_full')
    print('ALL COMPLETE',flush=True)


if __name__=='__main__':main()
