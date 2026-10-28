"""Checkpoint replay or exact full-coordinate frozen body tangent surrogate.

No change to existing checkpoints. No workers, compilation, or full Jacobian.
The frozen-body surrogate keeps the bilinear W J_h delta term and exact Adam.
"""
import argparse
import json
import os
import time
import resource
import types
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, jvp
from analysis.vit_long_1024 import common as C
from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.tangent_rank import cls_features
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT, OUT, SEED, forward_nodes


class Features(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self, x):
        return cls_features(self.model,x)


class ScaleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        scale,=ctx.saved_tensors
        return grad*scale,None


def intervene(model,kind,anchor=1.):
    """Only used in authorized short actual replays, after preregistration."""
    if kind=='none':return
    if kind=='temp4':
        att=model.layers['block_0'].layers['attn']
        def attention(self,x):
            B,N,D=x.shape
            q,k,v=self.layers['qkv'](x).reshape(B,N,3,self.n_heads,D//self.n_heads).permute(2,0,3,1,4).unbind(0)
            a=((q@k.transpose(-2,-1))*self.scale/4).softmax(-1)
            v=(a@v).transpose(1,2).reshape(B,N,D)
            return self.layers['proj'](v)
        att.forward=types.MethodType(attention,att)
    elif kind=='sigma':
        norm=model.layers['norm'];original=norm.forward
        def normalized(self,x):
            y=original(x)
            scale=((x.detach().var(-1,unbiased=False,keepdim=True)+self.eps).sqrt()/anchor).clamp(.1,10)
            return ScaleBackward.apply(y,scale)
        norm.forward=types.MethodType(normalized,norm)
    else:raise ValueError(kind)


def main(args):
    assert 0 < args.steps <= 5000
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    dest=OUT/f'{args.prefix}_{args.mode}_{args.arm}_t{args.task}.json'
    if dest.exists():raise FileExistsError(dest)
    model=C.build_arm_model(args.arm,SEED,'cuda',0.,2,8.)
    model.load_state_dict(torch.load(ROOT/'runs'/args.arm/f'seed{SEED}/ckpt/model_t{args.task:02d}.pt',map_location='cpu',weights_only=False))
    model.eval()
    keep_head=getattr(args,'keep_head',False)
    if not keep_head:reset_head(model)
    intervention=getattr(args,'intervention','none')
    anchor=1.
    if intervention=='sigma':
        old=np.load(OUT/f'decompose_{args.arm}_t{args.task}_n256.npz')
        anchor=float(np.median(old['values'][:,0,old['fields'].tolist().index('final.sigma')]))
    if intervention!='none':
        assert args.mode=='actual'
        intervene(model,intervention,anchor)
    task=getattr(args,'next_task',None) or args.task+1
    data,_=C.task_dataset(C.image_source('real'),SEED,task,10,250,augment=False)
    X=torch.cat([C.eval_tensor(data.images[s:s+32],'cuda') for s in range(0,2500,32)])
    y=data.labels.to('cuda')
    rows=torch.tensor(C.loss_classes(args.arm,SEED,task,10,250),device='cuda')
    if args.mode=='tangent':
        fmodel=Features(model)
        base={n:p.detach().clone() for n,p in fmodel.named_parameters()}
        names=[n for n in base if not n.startswith('model.layers.out')]
        bodybase={n:base[n] for n in names}
        delta={n:nn.Parameter(torch.zeros_like(base[n])) for n in names}
        headw=model.layers['out'].weight
        headb=model.layers['out'].bias
        # Return independent base + tangent body features; head stays nonlinear.
        def featurefun(par,x):
            return functional_call(fmodel,{**base,**par},(x,))
        def forward(x):
            h,dh=jvp(lambda p:featurefun(p,x),(bodybase,),(delta,))
            return F.linear(h+dh,headw,headb).index_select(1,rows)
        params=[delta['model.'+n] if not n.startswith('layers.out') else p for n,p in model.named_parameters()]
    elif args.mode=='headonly':
        for n,p in model.named_parameters():
            if not n.startswith('layers.out'):p.requires_grad_(False)
        def forward(x):return model(x).index_select(1,rows)
        params=list(model.parameters())
    else:
        if args.mode!='actual':raise ValueError(args.mode)
        def forward(x):return model(x).index_select(1,rows)
        params=list(model.parameters())
    optimizer=torch.optim.Adam(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    adamfile=ROOT/'runs'/args.arm/f'seed{SEED}/ckpt/adam_t{args.task:02d}.pt'
    loaded=False
    if not args.fresh_adam and args.task>0 and adamfile.exists():
        state=torch.load(adamfile,map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);del state
        loaded=True
    rec=[]; start=time.monotonic(); first90=None
    per_example=[]
    geometry_records=[]
    @torch.no_grad()
    def geometry(step):
        assert args.mode=='actual' and intervention=='none'
        idx=torch.randperm(2500,generator=torch.Generator().manual_seed(0))[:256].to('cuda')
        store={}
        for s in range(0,256,32):
            _,nodes,_=forward_nodes(model,X[idx[s:s+32]])
            for i in range(2):
                block=model.layers[f'block_{i}'].layers;att=block['attn']
                z=nodes[f'b{i}.n1y'];B,N,D=z.shape
                q=att.layers['qkv'](z)[:,:,:384].reshape(B,N,6,64).permute(0,2,1,3)
                wk=att.layers['qkv'].weight[384:768].reshape(6,64,384)
                aa=torch.einsum('bhnd,hde->bhne',q,wk)*att.scale*block['norm1'].weight
                an=aa.norm(dim=-1)
                sc=nodes[f'b{i}.score'].std(-1,unbiased=False)
                amax=nodes[f'b{i}.A'].max(-1).values
                for key,v in [('effective_cls',an[:,:,0]),('effective_tokens',an[:,:,1:]),('score_cls',sc[:,:,0]),('score_tokens',sc[:,:,1:]),('amax_cls',amax[:,:,0]),('amax_tokens',amax[:,:,1:])]:
                    store.setdefault(f'b{i}.{key}',[]).append(v.flatten())
            sigma=(nodes['final.x'][:,0].var(-1,unbiased=False)+model.layers['norm'].eps).sqrt()
            store.setdefault('final_sigma',[]).append(sigma)
        out={'step':step}
        for key,arr in store.items():
            v=torch.cat(arr).clamp_min(1e-30)
            out[key]=dict(median=float(v.median()),geomean=float(v.log().mean().exp()))
        geometry_records.append(out)
    @torch.no_grad()
    def evaluate(step):
        logits=torch.cat([forward(X[s:s+args.micro]) for s in range(0,2500,args.micro)])
        correct_n=int((logits.argmax(1)==y).sum())
        acc=correct_n/len(y)
        loss=float(F.cross_entropy(logits,y))
        correct=logits.gather(1,y[:,None]).squeeze(1)
        other=logits.clone();other.scatter_(1,y[:,None],-torch.inf)
        margin=correct-other.max(1).values
        per_example.append(margin.cpu().numpy())
        rec.append(dict(step=step,correct=correct_n,acc=acc,loss=loss,head_norm=float(model.layers['out'].weight[rows].norm())))
        print(f'{args.arm} {args.task} {args.mode}: {step} acc={acc:.4f} loss={loss:.5g} elapsed={time.monotonic()-start:.1f}',flush=True)
        return acc
    evaluate(0)
    if getattr(args,'geometry',False):geometry(0)
    for step,batch in enumerate(PairedBatches(2500,128,args.steps,SEED,task),1):
        ix=torch.tensor([m for m,_ in batch],device='cuda')
        optimizer.zero_grad(set_to_none=True)
        for s in range(0,len(ix),args.micro):
            ii=ix[s:s+args.micro]
            loss=F.cross_entropy(forward(X[ii]),y[ii],reduction='sum')/len(ix)
            loss.backward()
        optimizer.step()
        refresh=getattr(args,'refresh',0)
        if args.mode=='tangent' and refresh and step%refresh==0:
            # Move the expansion point in the original parameter coordinates.
            # Adam's moments are unchanged by this translation.
            with torch.no_grad():
                for n in names:
                    base[n].add_(delta[n])
                    delta[n].zero_()
        if args.mode=='actual' and step in args.save:
            torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in model.state_dict().items()},OUT/f'{args.prefix}_{args.arm}_t{args.task}_step{step}.pt')
        if step%25==0 or step==args.steps:
            acc=evaluate(step)
            if getattr(args,'geometry',False) and step in (25,100,500,1000,1500,args.steps):geometry(step)
            if acc>=.9 and first90 is None:first90=step
            checkpoint=dict(arm=args.arm,t=args.task,task=task,mode=args.mode,keep_head=keep_head,geometry=geometry_records,refresh=refresh,intervention=intervention,intervention_anchor=anchor,loaded_adam=loaded,micro=args.micro,
               T90=first90,steps=step,records=rec,seconds=time.monotonic()-start,
               rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20,cuda_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
            if args.mode=='tangent':
                checkpoint['delta_norm']=float(torch.stack([d.detach().square().sum() for d in delta.values()]).sum().sqrt())
            dest.write_text(json.dumps(checkpoint,indent=2))
            if first90 is not None and not args.full:break
    np.savez_compressed(dest.with_suffix('.npz'),margins=np.stack(per_example),labels=y.cpu().numpy())
    print(json.dumps(checkpoint),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('task',type=int)
    p.add_argument('mode',choices=['actual','tangent','headonly'])
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--micro',type=int,default=16)
    p.add_argument('--prefix',default='replay');p.add_argument('--fresh-adam',action='store_true')
    p.add_argument('--full',action='store_true');p.add_argument('--save',type=int,nargs='*',default=[])
    p.add_argument('--next-task',type=int)
    p.add_argument('--intervention',choices=['none','temp4','sigma'],default='none')
    p.add_argument('--refresh',type=int,default=0)
    p.add_argument('--keep-head',action='store_true')
    p.add_argument('--geometry',action='store_true')
    main(p.parse_args())
