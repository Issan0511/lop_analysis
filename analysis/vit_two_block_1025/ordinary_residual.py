"""Hold forward fixed and separate residual magnitude from wrong-class direction."""
import json
import time
from pathlib import Path
import common as T
from ordinary_drive import position, derivatives, fit, summarize
from t5_model import LocalFFN, fixed_metric

torch, np, F = T.torch, T.np, T.F
LR = 1e-4


def residual_fields(local, y):
    n = len(y)
    with torch.no_grad():
        logits = torch.cat([local.forward(idx)[0] for idx in torch.arange(n,device=y.device).split(32)])
        p = logits.double().softmax(1)
        wrong = p.clone(); wrong[torch.arange(n),y] = 0
        eps = wrong.sum(1)
        assert bool((eps > 0).all())
        stable = wrong.clone(); stable[torch.arange(n),y] = -eps
        uniform = eps[:,None].expand_as(p).clone()/(p.shape[1]-1)
        uniform[torch.arange(n),y] = -eps
        signals = dict(stable=stable, equal_error=stable*(eps.mean()/eps)[:,None],
                       uniform_wrong=uniform, both=uniform*(eps.mean()/eps)[:,None])
    fields = {k:[torch.zeros_like(local.fc1.weight),torch.zeros_like(local.fc1.bias)] for k in ['native',*signals]}
    for idx in torch.arange(n, device=y.device).split(32):
        f, _, _ = local.forward(idx)
        grads = torch.autograd.grad(F.cross_entropy(f,y[idx],reduction='sum'),
                                   (local.fc1.weight,local.fc1.bias),retain_graph=True)
        for a,b in zip(fields['native'],grads): a.add_(b/n)
        for j,(name,e) in enumerate(signals.items()):
            grads=torch.autograd.grad(f,(local.fc1.weight,local.fc1.bias),grad_outputs=e[idx].float(),retain_graph=j+1<len(signals))
            for a,b in zip(fields[name],grads): a.add_(b/n)
    error=max(float((a-b).abs().max()) for a,b in zip(fields['native'],fields['stable']))
    scale=max(float(a.abs().max()) for a in fields['native'])
    assert error < 1e-7+.002*scale, (error,scale)
    norm=lambda pair:torch.sqrt(sum(p.double().square().sum() for p in pair))
    norms={name:float(norm(v)) for name,v in fields.items()}
    normalized={name:tuple(p*(norms['stable']/norms[name]) for p in value)
                for name,value in fields.items()}
    return fields,normalized,dict(stable_native_error=error,native_scale=scale,
             epsilon_min=float(eps.min()),epsilon_max=float(eps.max()),epsilon_mean=float(eps.mean()),
             gradient_norms=norms),dict(epsilon=eps.cpu().numpy(),logits=logits.cpu().numpy())


def run(seed, task, start, layer):
    ident=dict(seed=seed,task=task,start=start,layer=layer)
    folder=T.ARCHIVE/'ordinary_residual'/f'seed{seed}_t{task:02d}_s{start:03d}_L{layer}'
    if (folder/'complete.json').exists(): return
    checkpoint=T.ARCHIVE/f'runs/seed{seed}/ckpt/t{task:02d}_s{start:03d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['source_hashes']==T.own_hashes()
    model=T.build(seed).eval();model.load_state_dict(state['model'])
    optimizer=T.C.build_optimizer(model);optimizer.load_state_dict(state['optimizer'])
    data,_=T.C.task_dataset(T.C.image_source(state['config']['data']),seed,task,10,25,augment=False)
    x,y=T.C.eval_tensor(data.images,'cuda'),data.labels.to('cuda')
    classes=torch.tensor(T.C.loss_classes('SL',seed,task,10,25),device='cuda')
    local=LocalFFN(model,layer,x,classes)
    assert local.verify_cache(x)
    initial,frozen=local.state(),local.frozen_digest()
    saved_optimizer=T.cpu_tree(optimizer.state_dict())
    metric=fixed_metric(optimizer,local.fc1)
    mu0,sd0,r0,valid0=[v.detach() for v in position(local)]
    rw,rb,valid,derr=derivatives(local)
    fields,normalized,checks,arrays=residual_fields(local,y)
    checks['position_derivative_error']=derr
    ce0,acc0=fit(local,y)
    rows=[]
    for name,pair in fields.items():
        for mode in ('sgd','fixed'):
            dw,db=(torch.ones_like(pair[0]),torch.ones_like(pair[1])) if mode=='sgd' else metric
            uw,ub=[-LR*d.double()*p.double() for d,p in zip((dw,db),pair)]
            dm=uw@local.mean+ub;dr=(uw*rw).sum(1)+ub*rb
            row=dict(**ident,arm=name,mode=mode,kind='linear',ce0=ce0,accuracy0=acc0)
            row.update({f'mu_{k}':v for k,v in summarize(dm).items()})
            row.update({f'r_{k}':v for k,v in summarize(dr,valid).items()})
            rows.append(row)
            arrays[f'{name}_{mode}_dmu']=dm.cpu().numpy();arrays[f'{name}_{mode}_dr']=dr.cpu().numpy()
        for mode in ('fixed','adam'):
            local.restore(initial);optimizer.load_state_dict(T.cpu_tree(saved_optimizer));optimizer.zero_grad(set_to_none=True)
            gradients=normalized[name]
            if mode=='fixed':
                with torch.no_grad():
                    for p,d,g in zip((local.fc1.weight,local.fc1.bias),metric,gradients):p.add_(d*g,alpha=-LR)
            else:
                local.fc1.weight.grad,local.fc1.bias.grad=[g.clone() for g in gradients]
                optimizer.step()
            with torch.no_grad():
                mu1,sd1,r1,valid1=position(local);ce1,acc1=fit(local,y)
                use=valid0&valid1
                row=dict(**ident,arm=name,mode=mode,kind='finite_norm',ce0=ce0,accuracy0=acc0,
                         ce1=ce1,accuracy1=acc1,ce_delta=ce1-ce0)
                row.update({f'mu_{k}':v for k,v in summarize(mu1-mu0).items()})
                row.update({f'r_{k}':v for k,v in summarize(r1-r0,use).items()})
                rows.append(row)
                arrays[f'{name}_{mode}_finite_dmu']=(mu1-mu0).cpu().numpy()
                arrays[f'{name}_{mode}_finite_dr']=(r1-r0).cpu().numpy()
            assert local.frozen_digest()==frozen
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'arrays.npz',**arrays)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    T.write_rows(folder/'rows.csv',[{k:row.get(k) for k in columns} for row in rows])
    paths=[Path(__file__),Path(__file__).with_name('ordinary_drive.py'),
           T.ROOT/'specs/spec_vit_two_block_residual_direction_1025.md']
    T.atomic_json(folder/'complete.json',dict(status='complete',**ident,checks=checks,
        checkpoint=str(checkpoint),checkpoint_sha256=T.sha256(checkpoint),frozen_unchanged=True,
        source_hashes={str(p.relative_to(T.ROOT)):T.sha256(p) for p in paths},
        git_hash=T.git_state()[0],arrays_sha256=T.sha256(folder/'arrays.npz'),updated=time.time()))
    print(json.dumps(dict(completed=ident,stable_native_error=checks['stable_native_error'])),flush=True)


if __name__=='__main__':
    T.init();assert not T.git_state()[1]
    for task in (10,60):
        for start in (100,500):
            for seed in T.SEEDS:
                for layer in (1,2):run(seed,task,start,layer)
