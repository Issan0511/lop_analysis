"""One checkpoint, one task, <=1500 updates; Adam every step, geometry every 25."""
import argparse
import time
from analysis.vit_long_1024.posthoc_2blk.codex_qkform_common import *
from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_replay import intervene

def adam_direction(opt,p):
    st=opt.state[p];pg=opt.param_groups[0]
    t=float(st['step']);b1,b2=pg['betas']
    return st['exp_avg']/(1-b1**t)/(st['exp_avg_sq'].sqrt()/math.sqrt(1-b2**t)+pg['eps'])

def main(args):
    setup();assert 1<=args.steps<=1500
    tag=f'codex_replay_{args.arm}_t{args.task}_{args.head}_temp{args.temperature:g}'
    dest=OUT/f'{tag}.json'
    if dest.exists():raise FileExistsError(dest)
    model=load_model(args.arm,args.task)
    task=args.task+1
    data,_=C.task_dataset(C.image_source('real'),SEED,task,10,250,augment=False)
    X=torch.cat([C.eval_tensor(data.images[s:s+32],'cuda') for s in range(0,2500,32)])
    y=data.labels.to('cuda');rows=torch.tensor(C.loss_classes(args.arm,SEED,task,10,250),device='cuda')
    idx=torch.randperm(2500,generator=torch.Generator().manual_seed(0))[:32].to('cuda');probe=X[idx]
    original_head=model.layers['out'].weight[rows].detach().clone()
    original_basis=row_basis(original_head)
    if args.head=='zero':reset_head(model)
    if args.temperature==4:intervene(model,'temp4')
    optimizer=C.build_optimizer(model)
    adamfile=ckpt(args.arm,args.task,'adam');loaded=False
    if not args.fresh_adam:
        if not adamfile.exists():raise FileNotFoundError(f'{adamfile}; pass --fresh-adam explicitly')
        track(adamfile)
        state=torch.load(adamfile,map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);del state;loaded=True
    with torch.no_grad():
        base,p0,refmom,_=snapshot(model,probe)
        direct=model(probe)
        manual,_,_=attention_forward(model,probe,args.temperature)
        forward_error=norm(direct-manual)/max(norm(direct),1e-100)
        assert forward_error<1e-5,forward_error
    block=model.layers['block_0'].layers
    W=block['attn'].layers['qkv'].weight;gamma=block['norm1'].weight;beta=block['norm1'].bias
    W0=W.detach().clone();g0=gamma.detach().clone()
    previous=p0;previous_W=W.detach().clone()
    records=[];geometry=[];diagnostics=[];step_rows=[];fields=None
    qsums=torch.zeros(3,device='cuda',dtype=torch.float64)
    c2sums=torch.zeros(3,device='cuda',dtype=torch.float64)
    usum=torch.zeros_like(W,dtype=torch.float64)
    start=time.monotonic()
    def evaluate(step):
        with torch.no_grad():
            # Full 2500-image evaluation is the same protocol as the prior replay.
            logits=torch.cat([model(X[s:s+128]).index_select(1,rows) for s in range(0,2500,128)])
            rec=dict(step=step,loss=float(F.cross_entropy(logits,y)),acc=float((logits.argmax(1)==y).float().mean()),head_norm=norm(model.layers['out'].weight[rows]))
            records.append(rec)
        return rec
    evaluate(0);base['step']=0;geometry.append(base)
    for step,batch in enumerate(PairedBatches(2500,128,args.steps,SEED,task),1):
        ii=torch.tensor([i for i,_ in batch],device='cuda')
        optimizer.zero_grad(set_to_none=True)
        # No instrumentation graph on ordinary steps; diagnostic steps use the verified forward.
        capture=step%25==0 or step==1
        if capture:
            logits,nodes,_=attention_forward(model,X[ii],args.temperature)
            nodes['score'].retain_grad();nodes['q'].retain_grad();nodes['k'].retain_grad()
        else:logits=model(X[ii])
        loss=F.cross_entropy(logits.index_select(1,rows),y[ii]);loss.backward()
        with torch.no_grad():
            oldW=W.detach().clone();oldg=gamma.detach().clone()
            rawrad=(oldW*W.grad).reshape(3,-1).double().sum(1)
            lhs=gamma*gamma.grad+beta*beta.grad
            rhs=(W*W.grad).sum(0)
            gauge=(lhs-rhs).norm()/(lhs.norm()+rhs.norm()).clamp_min(1e-30)
            ggmean=gamma.grad.mean().detach().clone()
            ggdot=(gamma*gamma.grad).sum().detach().clone()
            if capture:
                e=nodes['score'].grad.detach().square().flatten(1)
                ktop=max(1,int(math.ceil(e.shape[1]*.05)))
                share=e.topk(ktop,dim=1).values.sum()/e.sum().clamp_min(1e-30)
                vals=dict(step=step,score_top5_energy=float(share),score_top5_image_mean=float((e.topk(ktop,dim=1).values.sum(1)/e.sum(1).clamp_min(1e-30)).mean()),
                          score_grad_energy=float(e.sum()),saturation=float((nodes['prob'].detach().max(-1).values>.99).float().mean()))
                for key in ('q','k'):
                    z=nodes[key].grad.detach().permute(0,2,1,3).reshape(-1,D)
                    cov=z.T@z
                    vals[f'{key}_signal_pr']=float(cov.trace().square()/cov.square().sum().clamp_min(1e-30))
                if step%100==0 or step==1:
                    vals['gradient_Q_pr']=pr(W.grad[:D]);vals['gradient_K_pr']=pr(W.grad[D:2*D])
                diagnostics.append(vals)
        optimizer.step()
        with torch.no_grad():
            u=adam_direction(optimizer,W);ug=adam_direction(optimizer,gamma)
            delta=W-oldW
            step_sq=delta.reshape(3,-1).double().square().sum(1)
            radial=2*(oldW.double()*delta.double()).reshape(3,-1).sum(1)
            c2=u.reshape(3,-1).double().square().mean(1)
            c2sums+=c2;qsums+=step_sq;usum+=u.double()
            vals={'step':step,'batch_size':len(ii),'batch_loss':loss.detach(),'gauge_relative_error':gauge,
                  'gamma_gradient_mean':ggmean,'gamma_gradient_radial':ggdot,'gamma_u_mean':ug.mean(),
                  'gamma_update_mean':(gamma-oldg).mean(),'gamma_update_radial':2*(oldg*(gamma-oldg)).sum(),
                  'gamma_step_sq':(gamma-oldg).square().sum()}
            for j,key in enumerate(('q','k','v')):
                vals.update({f'{key}_c':c2[j].sqrt(),f'{key}_step_sq':step_sq[j],f'{key}_radial':radial[j],f'{key}_raw_radial':rawrad[j]})
            vals['adam_update_error']=((delta+1e-4*u).square().sum()/(1e-8*u.square().sum()).clamp_min(1e-30)).sqrt()
            if fields is None:fields=list(vals)
            step_rows.append(torch.stack([torch.as_tensor(vals[k],device='cuda',dtype=torch.float64) for k in fields]).cpu().numpy())
        if capture:del nodes
        del logits,loss,oldW,oldg,u,ug,delta
        if step%25==0 or step==args.steps:
            ev=evaluate(step)
            with torch.no_grad():
                geo,p,_,_=snapshot(model,probe,refmom);geo['step']=step
                diff=delta_record(previous,p,refmom[0],ranks=(step%100==0 or step==args.steps))
                geo['delta25']=diff
                if step in (100,500,1500,args.steps):
                    geo['delta_start']=delta_record(p0,p,refmom[0])
                    current=model.layers['out'].weight[rows].detach()
                    basis=row_basis(current)
                    geo['head_to_original']=basis_overlap(original_basis,basis) if original_basis.shape[1] and basis.shape[1] else None
                    # Head arrays and q/k endpoint are small and sufficient for paired subspace checks.
                    np.savez_compressed(OUT/f'{tag}_step{step}_small.npz',head=current.cpu().numpy(),Q=p['Q'].cpu().numpy(),K=p['K'].cpu().numpy(),gamma=p['g'].cpu().numpy())
                dw=W.detach()-W0
                net=dw.reshape(3,-1).double().square().sum(1)
                geo['walk']={key:dict(net_norm=math.sqrt(float(net[j])),independent_norm=math.sqrt(float(qsums[j])),
                                      coherence=float(net[j]/qsums[j].clamp_min(1e-30)),c_rms=math.sqrt(float(c2sums[j]/step))) for j,key in enumerate(('q','k','v'))}
                geo['integrated_adam_relative_error']=norm(dw.double()+1e-4*usum)/max(norm(dw),1e-100)
                geometry.append(geo);previous=p
            result=dict(timestamp=stamp(),arm=args.arm,task_end=args.task,next_task=task,head=args.head,temperature=args.temperature,
                        loaded_adam=loaded,fresh_adam=args.fresh_adam,batch=128,steps=step,query_probe_images=32,
                        full_eval_images=2500,manual_forward_error=forward_error,records=records,geometry=geometry,diagnostics=diagnostics,
                        seconds=time.monotonic()-start,complete=step==args.steps,**resource_record())
            dest.write_text(json.dumps(result,indent=2,allow_nan=False))
            np.savez_compressed(dest.with_suffix('.npz'),values=np.stack(step_rows),fields=np.array(fields),idx=idx.cpu().numpy(),original_head=original_head.cpu().numpy())
            if step%100==0:
                print(f'{tag}: step={step} loss={ev["loss"]:.5g} S={geo["S"]:.5g} cQ={geo["walk"]["q"]["c_rms"]:.4g} time={time.monotonic()-start:.1f}s',flush=True)
    progress(f'{tag}が{args.steps}更新で終了した。保存Adam={loaded}。経過{result["seconds"]:.1f}秒。RSS={result["rss_mb"]:.0f}MiB。GPU割当={result["cuda_peak_mb"]:.0f}MiB。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['SL','R0L']);p.add_argument('task',type=int)
    p.add_argument('--head',choices=['shared','zero'],required=True);p.add_argument('--temperature',type=float,choices=[1.,4.],default=1.)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--fresh-adam',action='store_true')
    main(p.parse_args())
