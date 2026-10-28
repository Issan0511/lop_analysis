"""One-task checkpoint replays, head-lr and feature-scale interventions.

No compile workers, no changed input files. Train/eval use the same deterministic
batch stream as production. Full batch128 (last epoch batch68), no micro split.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import (
    C, OUT, SEED, ckpt, dump, features, images, load_model, logit_stats,
    parts, progress, resources, setup, summary)


@torch.no_grad()
def geometry(model, xp):
    arrays={}
    for s in range(0,len(xp),8):
        nd=parts(model,xp[s:s+8])
        for k in ['e','attn0','ffn0','attn1','y2','z1','h1','h']:
            arrays.setdefault(k,[]).append(nd[k].detach())
    a={k:torch.cat(v).double() for k,v in arrays.items()}
    mm=a['y2'].mean((0,1));mn=mm.norm()
    fc=model.layers['block_1'].layers['mlp'].layers['fc1']
    w=fc.weight.detach().double();b=fc.bias.detach().double()
    co=(w@mm)/(w.norm(dim=1)*mn)
    z=a['z1'];zc=z[:,0]
    zf=z.flatten(0,1)
    out=dict(m2_norm=float(mn),h_norm_median=float(a['h'].norm(dim=1).median()),
             lr_l1_median=float(1e-4*a['h'].abs().sum(1).median()),
             h1_cls_norm=float(a['h1'][:,0].norm(dim=1).median()),
             h1_patch_norm=float(a['h1'][:,1:].norm(dim=-1).median()),
             cos_median=float(co.median()),cos_negative=float((co<0).double().mean()),
             zmean=float(z.mean()),z_cls_mean=float(zc.mean()),
             width=float(zf.std(0).median()),r=float((zf.mean(0)/zf.std(0).clamp_min(1e-12)).median()),
             dead=float(((zf>0).double().mean(0)<=.005).double().mean()),
             cls_dead=float(((zc>0).double().mean(0)<=.005).double().mean()),
             cls_alive=int((zc>0).any(0).sum()),
             contrib={k:float(a[k].mean((0,1))@mm/mn) for k in ['e','attn0','ffn0','attn1']})
    return out,dict(w=w.cpu(),b=b.cpu(),m=mm.cpu())


def main(args):
    assert 0<args.steps<=5000
    setup()
    tag=f'codex_replay_{args.arm}_t{args.task}_{args.mode}'+args.suffix
    dest=OUT/f'{tag}.json'
    if dest.exists():raise FileExistsError(dest)
    model=load_model(args.arm,args.task)
    opt=C.build_optimizer(model)
    adamfile=ckpt(args.arm,args.task,'adam')
    loaded=False
    if adamfile.exists():
        st=torch.load(adamfile,map_location='cpu',weights_only=False)
        opt.load_state_dict(st['optimizer']);del st
        loaded=True
    if args.arm=='R0':reset_head(model)
    head=model.layers['out']
    headids={id(p) for p in head.parameters()}
    base=dict(opt.param_groups[0]); allparams=base.pop('params')
    lr_factor={'base':1.,'lr01':.1,'lr001':.01,'scale01':1.}[args.mode]
    scale=.1 if args.mode=='scale01' else 1.
    opt.param_groups=[dict(base,params=[p for p in allparams if id(p) not in headids],lr=1e-4),
                      dict(base,params=list(head.parameters()),lr=1e-4*lr_factor)]
    x=images(); xp=images(train=False)
    y=C.task_labels(len(x),SEED,args.task+1,10).cuda()
    rows=torch.tensor(C.loss_classes(args.arm,SEED,args.task+1,10,250),device='cuda')
    records=[];geoms=[];stepsizes=[];first90=None;start=time.monotonic()
    previous=None
    def forward(xb):
        hh=features(model,xb)
        return F.linear(hh*scale,head.weight,head.bias).index_select(1,rows)
    @torch.no_grad()
    def evaluate(step):
        H=torch.cat([features(model,x[s:s+128]) for s in range(0,len(x),128)])
        logits=F.linear(H*scale,head.weight,head.bias).index_select(1,rows)
        accn=int((logits.argmax(1)==y).sum())
        row=dict(step=step,correct=accn,acc=accn/len(x),loss=float(F.cross_entropy(logits,y)),
                 **logit_stats(H*scale,head,rows))
        records.append(row)
        if step%250==0 or step in [1,25,100]:
            print(f'{tag} step{step} acc={row["acc"]:.4f} loss={row["loss"]:.4g} h={row["h_norm_median"]:.4g} sec={time.monotonic()-start:.1f}',flush=True)
        return row['acc']
    def take_geometry(step):
        nonlocal previous
        g,u=geometry(model,xp);g['step']=step
        if previous is not None:
            dw=u['w']-previous['w'];dm=u['m']-previous['m']
            terms={'weight':dw@previous['m'],'input':previous['w']@dm,'cross':dw@dm,'bias':u['b']-previous['b']}
            delta=u['w']@u['m']+u['b']-previous['w']@previous['m']-previous['b']
            g['ledger']={k:summary(v.numpy()) for k,v in terms.items()}
            g['ledger']['total']=summary(delta.numpy())
            g['ledger_error']=float((delta-sum(terms.values())).abs().max())
        previous=u;geoms.append(g)
    evaluate(0);take_geometry(0)
    diagnostic_steps={1,2,5,10,25,50,100,250,500,1000,2000,3500,5000}
    geometry_steps={1,5,25,100,250,500,1000,2000,3500,5000}
    last=0
    for step,batch in enumerate(PairedBatches(len(x),128,args.steps,SEED,args.task+1),1):
        ix=torch.tensor([i for i,_ in batch],device='cuda')
        opt.zero_grad(set_to_none=True)
        if step in diagnostic_steps:
            with torch.no_grad():
                fixedh=features(model,xp)*scale
                oldw=head.weight.detach()[rows].clone();oldb=head.bias.detach()[rows].clone()
                oldlog=F.linear(fixedh,oldw,oldb)
                rawnorm=0.
        loss=F.cross_entropy(forward(x[ix]),y[ix]);loss.backward()
        if step in diagnostic_steps:
            rawnorm=float(torch.stack([p.grad.square().sum() for p in allparams if p.grad is not None and id(p) not in headids]).sum().sqrt())
        opt.step();last=step
        if step in diagnostic_steps:
            with torch.no_grad():
                dw=head.weight[rows]-oldw;db=head.bias[rows]-oldb
                dhlog=F.linear(fixedh,dw,db)
                newh=features(model,xp)*scale
                bodylog=F.linear(newh-fixedh,head.weight[rows],None)
                total=dhlog+bodylog
                def rms(v):return float((v-v.mean(1,keepdim=True)).square().mean().sqrt())
                cap=fixedh.abs()@dw.abs().T+db.abs()
                stepsizes.append(dict(step=step,head_logit_rms=rms(dhlog),body_logit_rms=rms(bodylog),
                                      total_logit_rms=rms(total),head_logit_max=float(dhlog.abs().max()),
                                      exact_l1_bound_max=float(cap.max()),eta_l1_median=float(1e-4*lr_factor*fixedh.abs().sum(1).median()),
                                      max_head_coordinate_step=float(dw.abs().max()),body_raw_grad_norm=rawnorm,
                                      h_norm_before=float(fixedh.norm(dim=1).median()),h_norm_after=float(newh.norm(dim=1).median()),
                                      head_body_cross=float(((dhlog-dhlog.mean(1,keepdim=True))*(bodylog-bodylog.mean(1,keepdim=True))).mean())))
        if step in geometry_steps:take_geometry(step)
        if step%25==0 or step==args.steps:
            acc=evaluate(step)
            if acc>=.9 and first90 is None:first90=step
            save=dict(arm=args.arm,task_end=args.task,next_task=args.task+1,mode=args.mode,
                      head_lr=1e-4*lr_factor,body_lr=1e-4,feature_scale=scale,
                      loaded_adam=loaded,reset_head=args.arm=='R0',T90=first90,steps=step,
                      records=records,geometry=geoms,step_diagnostics=stepsizes,
                      seconds=time.monotonic()-start,resources=resources(),
                      stopping='full5000' if args.full else '250 updates after first T90 or budget')
            dump(dest,save)
            if first90 is not None and step>=first90+250 and not args.full:break
    if last not in geometry_steps:
        take_geometry(last);save['geometry']=geoms;dump(dest,save)
    progress(f'{tag}完了。更新{last}。T90={first90}。末の暗記率={records[-1]["acc"]:.4f}。保存Adam={loaded}。資源={resources()}。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['R0','S']);p.add_argument('task',type=int)
    p.add_argument('mode',choices=['base','lr01','lr001','scale01'])
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--full',action='store_true')
    p.add_argument('--suffix',default='')
    main(p.parse_args())
