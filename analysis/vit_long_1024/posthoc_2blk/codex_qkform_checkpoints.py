"""Q1 first: exact six-factor trajectory, static input mixing, and increments."""
import json
import math
import time
import numpy as np
import torch
from analysis.vit_long_1024.posthoc_2blk.codex_qkform_common import *

@torch.no_grad()
def main():
    setup();dest=OUT/'codex_checkpoints.json'
    if dest.exists():raise FileExistsError(dest)
    data,_=C.task_dataset(C.image_source('real'),SEED,11,10,250,augment=False)
    idx=torch.randperm(2500,generator=torch.Generator().manual_seed(0))[:32]
    x=C.eval_tensor(data.images[idx.numpy()],'cuda')
    result=dict(timestamp=stamp(),idx=idx.tolist(),query='all 65, uniform rms over images and heads',states={},increments={},input_mix={})
    start=time.monotonic()
    for arm in ('SL','R0L'):
        model=load_model(arm,10)
        refxi=tokens(model,x).detach();refmom=moment(refxi)
        patch0=model.layers['patch_embed'](x).detach().clone()
        cls0=model.cls_token.detach().clone();pos0=model.pos_embed.detach().clone()
        del model
        previous=None;previous_t=None
        result['states'][arm]={};result['increments'][arm]=[]
        for t in TASKS:
            model=load_model(arm,t)
            rec,p,cm,xi=snapshot(model,x,refmom)
            result['states'][arm][t]=rec
            if previous is not None:
                d=delta_record(previous,p,refmom[0]);d.update(t0=previous_t,t1=t)
                result['increments'][arm].append(d)
            previous=p;previous_t=t
            if t==100:
                patch1=model.layers['patch_embed'](x)
                cls1=model.cls_token;pos1=model.pos_embed
                mix={}
                for a in (0,1):
                    for b in (0,1):
                        patch=patch1 if a else patch0
                        cls=cls1 if b else cls0
                        pos=pos1 if b else pos0
                        z=torch.cat((cls.expand(len(x),-1,-1),patch),1)+pos
                        normed=(z-z.mean(-1,keepdim=True))/(z.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
                        mix[f'{a}{b}']=effective(p['A'],moment(normed)[0])
                lp=.5*math.log(mix['10']*mix['11']/(mix['00']*mix['01']))
                lpos=.5*math.log(mix['01']*mix['11']/(mix['00']*mix['10']))
                result['input_mix'][arm]=dict(values=mix,log_patch=lp,log_pos_cls=lpos)
            del model,xi,cm
            print(f'{arm} t{t}: S={rec["S"]:.5f} fixed={rec["a_fixed"]:.5f} sat={rec["saturation"]:.4f}',flush=True)
        changes={k:math.log(result['states'][arm][100]['factors'][k]/result['states'][arm][10]['factors'][k]) for k in ('P','R','G','T','I','J')}
        result.setdefault('log_changes',{})[arm]=changes
    result['log_difference']={k:result['log_changes']['SL'][k]-result['log_changes']['R0L'][k] for k in result['log_changes']['SL']}
    result['seconds']=time.monotonic()-start;result.update(resource_record())
    write_new(dest,result)
    progress('Q1の全40保存状態の計算が完了した。`codex_checkpoints.json`に六因子、初期毎課題の軌跡、更新の整合、入力の静的入替を保存した。\n\n'+json.dumps(result['log_changes'],ensure_ascii=False))
    print(json.dumps({k:result[k] for k in ('log_changes','log_difference','rss_mb','cuda_peak_mb','seconds')},indent=2))

if __name__=='__main__':main()
