"""Run each authorized replay to completion before starting the next heavy job."""
import os
import subprocess
import sys
from pathlib import Path

OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation')
CASES=[('SL',10,'zero',1),('SL',60,'shared',1),('SL',100,'shared',1),
       ('R0L',10,'zero',1),('R0L',100,'zero',1),('SL',100,'zero',1),('SL',100,'zero',4)]

def main():
    for arm,t,head,temp in CASES:
        tag=f'codex_replay_{arm}_t{t}_{head}_temp{temp}'
        with (OUT/f'{tag}.log').open('x') as log:
            args=[sys.executable,'-m','analysis.vit_long_1024.posthoc_2blk.codex_qkform_replay',arm,str(t),'--head',head,'--temperature',str(temp),'--steps','1500']
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
        print(f'completed {tag}',flush=True)
    with (OUT/'codex_signals.log').open('x') as log:
        subprocess.run([sys.executable,'-m','analysis.vit_long_1024.posthoc_2blk.codex_qkform_signals'],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
    print('completed signals',flush=True)

if __name__=='__main__':main()
