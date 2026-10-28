"""Frozen features, new zero head, identical next-task labels and batch stream."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C,OUT,SEED,dump,progress,resources,setup


def main():
    setup();start=time.monotonic();results=[]
    dest=OUT/'codex_linear_probes.json'
    if dest.exists():raise FileExistsError(dest)
    for arm,tasks in [('R0',[20,30,40,60]),('S',[30,60])]:
        for task in tasks:
            a=np.load(OUT/f'codex_features_{arm}_t{task}.npz')
            h=torch.from_numpy(a['h']).cuda()
            y=C.task_labels(len(h),SEED,task+1,10).cuda()
            hc=h.double()-h.double().mean(0)
            e=torch.linalg.eigvalsh(hc.T@hc).clamp_min(0)
            pr=float(e.sum().square()/e.square().sum())
            for lr_factor in [1.,.1,.01]:
                w=torch.nn.Parameter(torch.zeros(10,h.shape[1],device='cuda'))
                b=torch.nn.Parameter(torch.zeros(10,device='cuda'))
                opt=torch.optim.Adam([w,b],lr=1e-4*lr_factor,betas=(.9,.999),eps=1e-8,fused=True)
                rec=[];t90=None
                for step,batch in enumerate(PairedBatches(len(h),128,5000,SEED,task+1),1):
                    ix=torch.tensor([i for i,_ in batch],device='cuda')
                    opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(F.linear(h[ix],w,b),y[ix]);loss.backward();opt.step()
                    if step%25==0:
                        with torch.no_grad():
                            logits=F.linear(h,w,b);cn=int((logits.argmax(1)==y).sum())
                            rec.append(dict(step=step,correct=cn,acc=cn/len(h),loss=float(F.cross_entropy(logits,y))))
                        if cn>=2250 and t90 is None:t90=step
                results.append(dict(arm=arm,task_end=task,next_task=task+1,head_lr=1e-4*lr_factor,
                                    head_reset=True,adam='fresh',T90=t90,steps=5000,pr=pr,records=rec,
                                    final_acc=rec[-1]['acc'],max_acc=max(r['acc'] for r in rec)))
                dump(dest,dict(results=results,resources=resources(),seconds=time.monotonic()-start))
                print(f'{arm} t{task} frozen lr{lr_factor}: T90={t90} best={results[-1]["max_acc"]:.4f}',flush=True)
    progress(f'凍結体の新headによるlinear probe完了。{len(results)}条件。codex_linear_probes.json。')


if __name__=='__main__':main()
