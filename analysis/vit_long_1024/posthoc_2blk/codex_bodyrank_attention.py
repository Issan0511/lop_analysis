"""Exact CLS score-scale factorization and backward-weighted attention diagnostics."""
import os
import json
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT,OUT,SEED,forward_nodes


def run(arm,t,prefix='attention'):
    C.set_determinism(2);torch.cuda.set_per_process_memory_fraction(.15)
    model=C.build_arm_model(arm,SEED,'cuda',0.,2,8.)
    model.load_state_dict(torch.load(ROOT/'runs'/arm/f'seed{SEED}/ckpt/model_t{t:02d}.pt',map_location='cpu',weights_only=False));model.eval()
    data,_=C.task_dataset(C.image_source('real'),SEED,t+1,10,250,augment=False)
    old=np.load(OUT/f'decompose_{arm}_t{t}_n256.npz')
    U=torch.tensor(old['U'][[0,4]],device='cuda');idx=old['idx']
    scales=[];bp=[];fields=None;start=time.monotonic()
    with torch.no_grad():
        b=model.layers['block_0'].layers
        const=model.cls_token+model.pos_embed[:,:1]
        query=b['attn'].layers['qkv'](b['norm1'](const))[0,0,:384].reshape(6,64)
        wk=b['attn'].layers['qkv'].weight[384:768].reshape(6,64,384)
        effective=torch.einsum('hd,hde->he',query,wk)*b['attn'].scale*b['norm1'].weight
        norms=effective.norm(dim=1)
        weights={n:float(p.norm()) for n,p in model.named_parameters() if ('qkv' in n or 'norm' in n or n=='layers.out.weight')}
    for m in idx:
        h,nodes,_=forward_nodes(model,C.eval_tensor(data.images[m:m+1],'cuda'))
        with torch.no_grad():
            x=nodes['b0.n1x'][0]
            q=(x-x.mean(-1,keepdim=True))/(x.var(-1,unbiased=False,keepdim=True)+b['norm1'].eps).sqrt()
            pc=q@(effective/norms[:,None]).T
            keyspread=pc.std(0,unbiased=False)
            score=nodes['b0.score'][0,:,0]
            actual=score.std(-1,unbiased=False)
            reconstructed=norms*keyspread
            scales.append(torch.stack([actual,keyspread,reconstructed]).cpu().numpy())
        rec=[]
        allscale={}
        with torch.no_grad():
            for i in range(2):
                block=model.layers[f'block_{i}'].layers
                att=block['attn']
                qq=att.layers['qkv'](nodes[f'b{i}.n1y'])[0,:,:384].reshape(65,6,64).transpose(0,1)
                kk=att.layers['qkv'].weight[384:768].reshape(6,64,384)
                aa=torch.einsum('hnd,hde->hne',qq,kk)*att.scale*block['norm1'].weight
                an=aa.norm(dim=-1)
                ss=nodes[f'b{i}.score'][0].std(-1,unbiased=False)
                allscale[i]=(an,ss,ss/an.clamp_min(1e-30))
        wanted=[nodes[f'b{i}.{key}'] for i in range(2) for key in ('A','score')]
        for d,u in enumerate(U):
            gs=torch.autograd.grad((h[0]*u).sum(),wanted,retain_graph=d==0)
            vals={}
            for i in range(2):
                A=nodes[f'b{i}.A'][0]; ra,rs=gs[2*i][0],gs[2*i+1][0]
                # Each head and query is weighted by its incoming cotangent energy.
                weight=ra.square().sum(-1)
                total=weight.sum().clamp_min(1e-30)
                maxa=A.max(-1).values
                vals[f'b{i}.weighted_saturation']=(weight*(maxa>.99)).sum()/total
                vals[f'b{i}.weighted_maxa']=(weight*maxa).sum()/total
                vals[f'b{i}.cls_cotangent_share']=weight[:,0].sum()/total
                vals[f'b{i}.score_grad_sq']=rs.square().sum()
                vals[f'b{i}.score_cls_share']=rs[:,0].square().sum()/rs.square().sum().clamp_min(1e-30)
                vals[f'b{i}.gate_gain']=rs.norm()/ra.norm().clamp_min(1e-30)
                wscore=rs.square().sum(-1)
                vals[f'b{i}.active_queries']=(wscore.sum()**2)/(wscore.square().sum().clamp_min(1e-30))
                an,ss,ks=allscale[i]
                for name,v in [('effective_norm',an),('score_std',ss),('keyspread',ks)]:
                    vals[f'b{i}.{name}_weighted']=(weight*v.clamp_min(1e-30).log()).sum().div(total).exp()
                    vals[f'b{i}.{name}_uniform']=v.clamp_min(1e-30).log().mean().exp()
            if fields is None:fields=list(vals)
            rec.append(torch.stack(list(vals.values())).detach())
        bp.append(torch.stack(rec).cpu().numpy())
    dest=OUT/f'{prefix}_{arm}_t{t}.npz'
    if dest.exists():raise FileExistsError(dest)
    np.savez_compressed(dest,scales=np.stack(scales),effective_norms=norms.cpu().numpy(),backward=np.stack(bp),fields=np.array(fields),idx=idx)
    dest.with_suffix('.json').write_text(json.dumps(dict(weights=weights,seconds=time.monotonic()-start),indent=2))
    print(arm,t,'effective norms',norms.cpu().tolist(),'seconds',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    for a,t in [('SL',10),('SL',100),('R0L',10),('R0L',100)]:run(a,t)
