"""Freeze only the body; retain exactly the head/Adam start used by replays."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C,OUT,SEED,ckpt,load_model,dump,progress,resources,setup


def main():
    setup();start=time.monotonic();results=[]
    dest=OUT/'codex_frozen_matched.json'
    if dest.exists():raise FileExistsError(dest)
    for arm in ['R0','S']:
        for task in [30,60]:
            arr=np.load(OUT/f'codex_features_{arm}_t{task}.npz')
            h=torch.from_numpy(arr['h']).cuda();y=C.task_labels(len(h),SEED,task+1,10).cuda()
            rows=torch.tensor(C.loss_classes(arm,SEED,task+1,10,250),device='cuda')
            for factor in [1.,.1,.01]:
                model=load_model(arm,task);opt=C.build_optimizer(model)
                opt.load_state_dict(torch.load(ckpt(arm,task,'adam'),map_location='cpu',weights_only=False)['optimizer'])
                if arm=='R0':reset_head(model)
                head=model.layers['out']
                base=dict(opt.param_groups[0]);base['params']=list(head.parameters());base['lr']=1e-4*factor
                opt.param_groups=[base]
                for p in model.parameters():p.requires_grad_(False)
                for p in head.parameters():p.requires_grad_(True)
                rec=[];t90=None
                @torch.no_grad()
                def evaluate(step):
                    logits=F.linear(h,head.weight,head.bias).index_select(1,rows)
                    correct=int((logits.argmax(1)==y).sum())
                    rec.append(dict(step=step,correct=correct,acc=correct/len(h),loss=float(F.cross_entropy(logits,y))))
                    return correct
                evaluate(0)
                for step,batch in enumerate(PairedBatches(len(h),128,5000,SEED,task+1),1):
                    ix=torch.tensor([i for i,_ in batch],device='cuda')
                    opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(F.linear(h[ix],head.weight,head.bias).index_select(1,rows),y[ix])
                    loss.backward();opt.step()
                    if step%25==0:
                        correct=evaluate(step)
                        if correct>=2250 and t90 is None:t90=step
                results.append(dict(arm=arm,task_end=task,next_task=task+1,head_lr=1e-4*factor,
                    reset_head=arm=='R0',loaded_adam=True,steps=5000,T90=t90,records=rec,
                    final_acc=rec[-1]['acc'],max_acc=max(r['acc'] for r in rec)))
                dump(dest,dict(results=results,resources=resources(),seconds=time.monotonic()-start))
                print(f'{arm} t{task} matched frozen lr{factor}: T90={t90} best={results[-1]["max_acc"]:.4f}',flush=True)
                del opt,model
    progress(f'初期headと保存Adamをそろえた凍結体対照12条件が完了。codex_frozen_matched.json。')


if __name__=='__main__':main()
