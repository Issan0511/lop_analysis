"""Numerical identities and saved-Adam group split, without writing checkpoints."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C,OUT,SEED,ckpt,dump,features,images,load_model,parts,resources,setup


def main():
    setup();result={}
    x=images(train=False)[:8]
    for arm,task in [('R0',10),('R0',60),('S',30)]:
        m=load_model(arm,task,dtype=torch.float64);xx=x.double()
        nd=parts(m,xx,True)
        out=m.layers['out'];rows=C.loss_classes(arm,SEED,task,10,250)
        logits=F.linear(nd['h'],out.weight,out.bias)
        rel=float((logits-m(xx)).norm()/m(xx).norm().clamp_min(1e-30))
        loss=-(logits[:,rows].log_softmax(-1)).mean()
        fc=m.layers['block_1'].layers['mlp'].layers['fc1']
        gw,gb=torch.autograd.grad(loss,[fc.weight,fc.bias])
        with torch.no_grad():
            p=logits[:,rows].softmax(-1)
            U=m.layers['block_1'].layers['mlp'].layers['fc2'].weight
            dz=((p-.1)@out.weight[rows]@U)*(nd['z1'][:,0]>0)
            exact=dz.T@nd['y2'][:,0]/len(x)
            diff=float((exact-gw).norm()/gw.norm().clamp_min(1e-30))
            db=float((dz.mean(0)-gb).abs().max())
        result[f'{arm}{task}']=dict(forward_relative=rel,analytic_grad_relative=diff,bias_grad_max_abs=db)
        assert rel<1e-10 and diff<1e-10
        del m,nd,gw,gb,logits,loss,out,fc
    # Splitting parameter groups at equal learning rates preserves every Adam step.
    m=load_model('R0',30);other=copy.deepcopy(m)
    opts=[C.build_optimizer(mm) for mm in [m,other]]
    for opt in opts:
        st=torch.load(ckpt('R0',30,'adam'),map_location='cpu',weights_only=False)
        opt.load_state_dict(st['optimizer']);del st
    head=other.layers['out'];ids={id(p) for p in head.parameters()}
    base=dict(opts[1].param_groups[0]);params=base.pop('params')
    opts[1].param_groups=[dict(base,params=[p for p in params if id(p) not in ids]),dict(base,params=list(head.parameters()))]
    for mm,opt in zip([m,other],opts):
        with torch.no_grad():mm.layers['out'].weight.zero_();mm.layers['out'].bias.zero_()
        logits=mm(x)[:,C.loss_classes('R0',SEED,31,10,250)]
        loss=F.cross_entropy(logits,torch.arange(8,device='cuda')%10)
        opt.zero_grad();loss.backward();opt.step()
    mx=max(float((a-b).abs().max()) for a,b in zip(m.parameters(),other.parameters()))
    result['adam_equal_lr_group_split_max_abs']=mx
    assert mx==0.
    result['resources']=resources()
    dump(OUT/'codex_verification.json',result)
    print(result)


if __name__=='__main__':main()
