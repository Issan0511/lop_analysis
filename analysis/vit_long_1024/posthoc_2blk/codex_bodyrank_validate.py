"""Validate the frozen-body surrogate's feature and parameter gradients at delta=0."""
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
from torch.func import functional_call,jvp
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT,OUT,SEED
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_replay import Features
from analysis.vit_long_1024.posthoc_2blk.tangent_rank import cls_features


def validate():
    C.set_determinism(2);torch.cuda.set_per_process_memory_fraction(.15)
    model=C.build_arm_model('SL',SEED,'cuda',0.,2,8.)
    model.load_state_dict(torch.load(ROOT/'runs/SL'/f'seed{SEED}/ckpt/model_t100.pt',map_location='cpu',weights_only=False));model.eval()
    data,_=C.task_dataset(C.image_source('real'),SEED,101,10,250,augment=False)
    X=C.eval_tensor(data.images[:4],'cuda');y=data.labels[:4].to('cuda')
    rows=torch.tensor(C.loss_classes('SL',SEED,101,10,250),device='cuda')
    nativeh=cls_features(model,X)
    native=model.layers['out'](nativeh).index_select(1,rows)
    named=list(model.named_parameters())
    ng=torch.autograd.grad(F.cross_entropy(native,y),[p for _,p in named])
    fmodel=Features(model);base={n:p.detach().clone() for n,p in fmodel.named_parameters()}
    names=[n for n in base if not n.startswith('model.layers.out')]
    body={n:base[n] for n in names}
    delta={n:torch.zeros_like(base[n],requires_grad=True) for n in names}
    def fun(par):return functional_call(fmodel,{**base,**par},(X,))
    h,dh=jvp(fun,(body,),(delta,))
    logits=F.linear(h+dh,model.layers['out'].weight,model.layers['out'].bias).index_select(1,rows)
    params=[delta['model.'+n] if not n.startswith('layers.out') else p for n,p in named]
    gg=torch.autograd.grad(F.cross_entropy(logits,y),params)
    diff=torch.stack([(a-b).square().sum() for a,b in zip(ng,gg)]).sum().sqrt()
    norm=torch.stack([a.square().sum() for a in ng]).sum().sqrt()
    res=dict(feature_max_error=float((h-nativeh).abs().max().detach()),logit_max_error=float((logits-native).abs().max().detach()),gradient_relative_l2=float(diff/norm),nonzero_head_norm=float(model.layers['out'].weight[rows].norm().detach()))
    assert res['feature_max_error']<1e-5 and res['logit_max_error']<1e-5 and res['gradient_relative_l2']<1e-5,res
    (OUT/'verification_tangent.json').write_text(json.dumps(res,indent=2));print(res,flush=True)


if __name__=='__main__':validate()
