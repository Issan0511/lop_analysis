"""Stratified coordinate approximation of a directional Gram, exact row norms.

Retains up to K coordinates per parameter tensor. This is not the T90 predictor:
the predictor uses the full Jacobian implicitly. Validate PR against old exact
checkpoint diagnostics before interpreting intermediate-checkpoint results.
"""
import os
import json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT,OUT,SEED
from analysis.vit_long_1024.posthoc_2blk.tangent_rank import cls_features,gram_stats


def run(arm,t,prefix,checkpoint=None,coords=2048,mode='full'):
    C.set_determinism(2);torch.cuda.set_per_process_memory_fraction(.15)
    model=C.build_arm_model(arm,SEED,'cuda',0.,2,8.)
    cp=checkpoint or ROOT/'runs'/arm/f'seed{SEED}/ckpt/model_t{t:02d}.pt'
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False));model.eval()
    data,_=C.task_dataset(C.image_source('real'),SEED,t+1,10,250,augment=False)
    with torch.no_grad():
        H=torch.cat([cls_features(model,C.eval_tensor(data.images[s:s+32],'cuda')) for s in range(0,2500,32)])
        _,sv,vh=torch.linalg.svd(H-H.mean(0),full_matrices=False)
        gen=torch.Generator().manual_seed(0);idx=torch.randperm(2500,generator=gen)[:256]
        rr=torch.randn(4,384,generator=gen).to('cuda');rr=rr/rr.norm(dim=1,keepdim=True)
        U=torch.stack((rr[0],vh[0]))
    params=[p for n,p in model.named_parameters() if not n.startswith('layers.out')]
    rng=torch.Generator().manual_seed(1028)
    ix=[torch.randperm(p.numel(),generator=rng)[:min(coords,p.numel())].to('cuda') for p in params]
    scales=[(p.numel()/len(ii))**.5 for p,ii in zip(params,ix)]
    count=sum(len(ii) for ii in ix)
    out=dict(arm=arm,t=t,mode=mode,checkpoint=str(cp),coords_per_tensor=coords,n_coords=count,
             feature_pr=float(sv.square().sum().square()/sv.pow(4).sum()))
    for key,u in zip(('rand0','pc0'),U):
        X=torch.empty(256,count,device='cuda');norm2=torch.empty(256,device='cuda')
        for j,m in enumerate(idx.tolist()):
            x=C.eval_tensor(data.images[m:m+1],'cuda')
            if mode=='full':h=cls_features(model,x)
            else:
                from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_paths import forward
                h,_=forward(model,x,mode)
            g=torch.autograd.grad((h[0]*u).sum(),params)
            X[j]=torch.cat([v.flatten()[ii]*c for v,ii,c in zip(g,ix,scales)])
            norm2[j]=torch.stack([v.square().sum() for v in g]).sum()
        raw=X.square().sum(1)
        error=(raw/norm2-1).abs()
        X*=torch.sqrt(norm2/raw)[:,None]
        G=X@X.T
        out[key]=gram_stats(G)
        out[key]['raw_diag_relative_error_median']=float(error.median())
        out[key]['raw_diag_relative_error_max']=float(error.max())
        del X,G
    dest=OUT/f'{prefix}_{arm}_t{t}.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(out,indent=2))
    print(prefix,arm,t,'rand',out['rand0']['pr'],'pc',out['pc0']['pr'],flush=True)


if __name__=='__main__':
    for a,t in [('SL',0),('SL',10),('SL',100),('R0L',100)]:run(a,t,'kernel_validate')
