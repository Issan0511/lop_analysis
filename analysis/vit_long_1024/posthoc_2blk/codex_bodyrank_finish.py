"""Sequential completion suite. Run only after earlier GPU sessions have exited.

Each replay is from an existing saved state and contains at most 5000 updates.
All hypotheses for these calls are in the raw-output PROGRESS.md before launch.
"""
from argparse import Namespace
import json
import sys
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import OUT
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_kernel import run as kernel
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_replay import main


def replay(prefix,arm,t,mode='actual',steps=1500,**kwargs):
    p=OUT/f'{prefix}_{mode}_{arm}_t{t}.json'
    if p.exists():
        d=json.loads(p.read_text())
        if d['steps']==steps or (d['T90'] is not None and not kwargs.get('full',False)):
            print('Already complete:',p.name,flush=True);return
        raise RuntimeError(f'Partial output requires inspection: {p}')
    args=dict(arm=arm,task=t,next_task=25 if t==0 else None,mode=mode,steps=steps,micro=64,prefix=prefix,
              fresh_adam=False,full=False,save=[],intervention='none',refresh=0,keep_head=False,geometry=False)
    args.update(kwargs);main(Namespace(**args))


def kernels():
    if not (OUT/'kernel_noscore_SL_t100.json').exists():
        kernel('SL',100,'kernel_noscore',coords=8192,mode='noscore_both')
    for t in (0,100):
        for step in (25,100,500):
            prefix=f'kernel_step{step}'
            p=OUT/f'{prefix}_SL_t{t}.json'
            if p.exists():continue
            cp=OUT/f'replay_SL_t{t}_step{step}.pt'
            if not cp.exists():raise FileNotFoundError(cp)
            kernel('SL',t,prefix,checkpoint=cp,coords=8192)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='canonical':
        for arm,t in [('SL',0),('SL',10),('SL',40),('SL',60),('SL',100),('R0L',10),('R0L',40),('R0L',60),('R0L',100)]:
            if (arm,t)!=('SL',100):replay('canonical',arm,t,micro=128)
            replay('canonical5',arm,t,'tangent',micro=128,refresh=5)
        sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1]=='formation-early':
        from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_operator import main as operators
        operators()
        replay('formation_shared','SL',10,full=True,keep_head=True,geometry=True)
        replay('formation_zero','SL',10,full=True,geometry=True)
        sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1]=='refine-all':
        for arm,t in [('SL',0),('SL',10),('SL',40),('SL',60),('SL',100),('R0L',10),('R0L',40),('R0L',60),('R0L',100)]:
            replay('refresh5',arm,t,'tangent',refresh=5)
        sys.exit(0)
    kernels()
    for t in (0,100):replay('refresh5','SL',t,'tangent',refresh=5)
    replay('q4_temp4','SL',100,intervention='temp4')
    replay('q4_sigma','SL',100,intervention='sigma')
    replay('formation_shared','SL',100,full=True,keep_head=True,geometry=True)
    replay('formation_zero','SL',100,full=True,geometry=True)
    replay('micro128','SL',100,micro=128)
    replay('complete','SL',0,'tangent',steps=5000)
