"""Forward-preserving derivative path ablations, without optimizer updates."""
import argparse
import json
import os
import time
import resource
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT, OUT, SEED


def forward(model, x, mode, metadata=False):
    z=model.layers['patch_embed'](x)
    z=torch.cat((model.cls_token.expand(len(x),-1,-1),z),1)+model.pos_embed
    aux={}
    for i in range(2):
        b=model.layers[f'block_{i}'].layers
        a=b['norm1'](z)
        att=b['attn']; B,N,D=a.shape
        q,k,v=att.layers['qkv'](a).reshape(B,N,3,6,64).permute(2,0,3,1,4).unbind(0)
        score=q@k.transpose(-2,-1)*att.scale
        A=score.softmax(-1)
        if mode in (f'noscore{i}','noscore_both'):
            A=A.detach()
        if mode in (f'uniformscore{i}','uniformscore_both'):
            # Same forward values, linearization of uniform softmax in backward.
            U=(score-score.mean(-1,keepdim=True))/N
            A=A.detach()+U-U.detach()
        if metadata:
            s=score[0,:,0].sort(-1,descending=True).values
            aux[f'b{i}.gap']=s[:,0]-s[:,1]
            aux[f'b{i}.score_std']=score[0,:,0].std(-1)
            aux[f'b{i}.Amax']=A[0,:,0].max(-1).values
            aux[f'b{i}.qnorm']=q[0,:,0].norm(dim=-1)
            aux[f'b{i}.knorm']=k[0].norm(dim=-1).mean(-1)
        av=(A@v).transpose(1,2).reshape(B,N,D)
        z=z+att.layers['proj'](av)
        mlp=b['mlp'].layers
        y=mlp['fc2'](mlp['act'](mlp['fc1'](b['norm2'](z))))
        if mode in (f'nomlp{i}','nomlp_both'):
            y=y.detach()
        z=z+y
    aux['sigma']=(z[0,0].var(unbiased=False)+model.layers['norm'].eps).sqrt()
    return model.layers['norm'](z)[:,0],aux


def run(arm,t):
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    model=C.build_arm_model(arm,SEED,'cuda',0.,2,8.)
    model.load_state_dict(torch.load(ROOT/'runs'/arm/f'seed{SEED}/ckpt/model_t{t:02d}.pt',map_location='cpu',weights_only=False))
    model.eval()
    data,_=C.task_dataset(C.image_source('real'),SEED,t+1,10,250,augment=False)
    z=np.load(OUT/f'decompose_{arm}_t{t}_n256.npz')
    U=torch.tensor(z['U'][[0,4]],device='cuda')
    idx=z['idx']
    params=[(n,p) for n,p in model.named_parameters() if not n.startswith('layers.out')]
    modes=['full','noscore0','noscore1','noscore_both','nomlp0','nomlp1','nomlp_both']
    result=np.empty((len(idx),len(modes),2,5))
    info=[]; start=time.monotonic()
    for j,m in enumerate(idx):
        x=C.eval_tensor(data.images[m:m+1],'cuda')
        for k,mode in enumerate(modes):
            h,aux=forward(model,x,mode,metadata=(k==0))
            if k==0:
                info.append({key:v.detach().cpu().tolist() for key,v in aux.items()})
            for d,u in enumerate(U):
                gs=torch.autograd.grad((h[0]*u).sum(),[p for _,p in params],allow_unused=True,retain_graph=d==0)
                g2=[g.square().sum() if g is not None else torch.tensor(0.,device='cuda') for g in gs]
                total=torch.stack(g2).sum()
                # q, k, v contributions across both attention blocks.
                qs=torch.zeros(3,device='cuda')
                for (name,p),g in zip(params,gs):
                    if '.qkv.' in name and g is not None:
                        qs+=g.reshape(3,-1).square().sum(1)
                patch=sum(v for (name,_),v in zip(params,g2) if name.startswith('layers.patch_embed'))
                result[j,k,d]=torch.stack([total,patch,*qs]).detach().cpu().numpy()
    dest=OUT/f'paths_{arm}_t{t}.npz'
    if dest.exists(): raise FileExistsError(dest)
    np.savez_compressed(dest,norms_sq=result,idx=idx,modes=np.array(modes),directions=np.array(['rand0','pc0']))
    dest.with_suffix('.json').write_text(json.dumps(dict(arm=arm,t=t,metadata=info,seconds=time.monotonic()-start,
        rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20)))
    print(f'{arm} {t}: paths done in {time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('task',type=int);a=p.parse_args();run(a.arm,a.task)
