"""Separate the shared-FFN update coming from CLS and patch tokens."""
import json
import time
from pathlib import Path
import common as T
from t5_model import LocalFFN, fixed_metric
from ordinary_drive import derivatives, fit, summarize

torch,np,F=T.torch,T.np,T.F
LR=1e-4


@torch.no_grad()
def populations(local):
    stats={name:[] for name in ('cls','patch')}
    for u in local.u.split(32):
        z=local.fc1(u).double()
        stats['cls'].append(z[:,0]);stats['patch'].append(z[:,1:].reshape(-1,z.shape[-1]))
    result={}
    for name,parts in stats.items():
        z=torch.cat(parts);mu=z.mean(0);sd=z.std(0,unbiased=True)
        result[name]=dict(mu=mu,sd=sd,r=mu/(sd+1e-12),valid=sd>1e-8)
    return result


def token_fields(local,y):
    fields={name:[torch.zeros_like(local.fc1.weight),torch.zeros_like(local.fc1.bias)] for name in ('all','cls_only','patch_only')}
    for idx in torch.arange(len(y),device=y.device).split(32):
        logits,z,a=local.forward(idx)
        dz,gw,gb=torch.autograd.grad(F.cross_entropy(logits,y[idx],reduction='sum'),(z,local.fc1.weight,local.fc1.bias))
        fields['all'][0].add_(gw/len(y));fields['all'][1].add_(gb/len(y))
        for name,sl in [('cls_only',slice(0,1)),('patch_only',slice(1,None))]:
            signal=dz[:,sl].reshape(-1,dz.shape[-1]);u=local.u[idx,sl].reshape(-1,local.u.shape[-1])
            fields[name][0].add_(signal.T@u/len(y));fields[name][1].add_(signal.sum(0)/len(y))
    error=max(float((fields['all'][j]-fields['cls_only'][j]-fields['patch_only'][j]).abs().max()) for j in (0,1))
    scale=max(float(g.abs().max()) for g in fields['all'])
    assert error<1e-7+.002*scale,(error,scale)
    norms={name:float(torch.sqrt(sum(g.double().square().sum() for g in value))) for name,value in fields.items()}
    if local.layer==2:assert norms['patch_only']==0,norms
    return fields,dict(additivity_error=error,scale=scale,norms=norms)


def run(seed,task,start,layer):
    ident=dict(seed=seed,task=task,start=start,layer=layer)
    folder=T.ARCHIVE/'ordinary_tokens'/f'seed{seed}_t{task:02d}_s{start:03d}_L{layer}'
    if (folder/'complete.json').exists():return
    checkpoint=T.ARCHIVE/f'runs/seed{seed}/ckpt/t{task:02d}_s{start:03d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['source_hashes']==T.own_hashes()
    model=T.build(seed).eval();model.load_state_dict(state['model'])
    optimizer=T.C.build_optimizer(model);optimizer.load_state_dict(state['optimizer'])
    data,_=T.C.task_dataset(T.C.image_source(state['config']['data']),seed,task,10,25,augment=False)
    x,y=T.C.eval_tensor(data.images,'cuda'),data.labels.to('cuda')
    classes=torch.tensor(T.C.loss_classes('SL',seed,task,10,25),device='cuda')
    local=LocalFFN(model,layer,x,classes);assert local.verify_cache(x)
    initial,frozen=local.state(),local.frozen_digest()
    saved_optimizer=T.cpu_tree(optimizer.state_dict());metric=fixed_metric(optimizer,local.fc1)
    before=populations(local);rw,rb,valid,derr=derivatives(local)
    fields,checks=token_fields(local,y);checks['derivative_error']=derr
    ce0,acc0=fit(local,y)
    rows=[];arrays={}
    for name,pair in fields.items():
        for mode in ('sgd','fixed'):
            dmtr=(torch.ones_like(pair[0]),torch.ones_like(pair[1])) if mode=='sgd' else metric
            uw,ub=[-LR*d.double()*p.double() for d,p in zip(dmtr,pair)]
            dm=uw@local.mean+ub;dr=(uw*rw).sum(1)+ub*rb
            row=dict(**ident,arm=name,mode=mode,kind='linear',population='cls',ce0=ce0,accuracy0=acc0)
            row.update({f'mu_{k}':v for k,v in summarize(dm).items()});row.update({f'r_{k}':v for k,v in summarize(dr,valid).items()})
            rows.append(row);arrays[f'{name}_{mode}_dmu']=dm.cpu().numpy();arrays[f'{name}_{mode}_dr']=dr.cpu().numpy()
        if checks['norms'][name]==0:continue
        gradients=[g*(checks['norms']['all']/checks['norms'][name]) for g in pair]
        for mode in ('fixed','adam'):
            local.restore(initial);optimizer.load_state_dict(T.cpu_tree(saved_optimizer));optimizer.zero_grad(set_to_none=True)
            if mode=='fixed':
                with torch.no_grad():
                    for p,d,g in zip((local.fc1.weight,local.fc1.bias),metric,gradients):p.add_(d*g,alpha=-LR)
            else:
                local.fc1.weight.grad,local.fc1.bias.grad=[g.clone() for g in gradients];optimizer.step()
            after=populations(local);ce1,acc1=fit(local,y)
            for population in before:
                v0,v1=before[population],after[population];use=v0['valid']&v1['valid']
                row=dict(**ident,arm=name,mode=mode,kind='finite_norm',population=population,ce0=ce0,accuracy0=acc0,
                         ce1=ce1,accuracy1=acc1,ce_delta=ce1-ce0)
                row.update({f'mu_{k}':v for k,v in summarize(v1['mu']-v0['mu']).items()})
                row.update({f'r_{k}':v for k,v in summarize(v1['r']-v0['r'],use).items()});rows.append(row)
                arrays[f'{name}_{mode}_{population}_finite_dr']=(v1['r']-v0['r']).cpu().numpy()
            assert local.frozen_digest()==frozen
    for mode in ('fixed','adam'):
        ref=T.ARCHIVE/'t5'/f'seed{seed}_t{task:02d}_s{start:03d}_L{layer}_{mode}'
        record=torch.load(ref/'full.pt',map_location='cpu',weights_only=False)
        assert T.digest_tree(record['states'][0])==T.digest_tree(initial)
        local.restore(record['states'][50]);after=populations(local)
        with np.load(ref/'measurements.npz') as d:
            err=float(np.abs(after['cls']['r'].cpu().numpy()-d['s50_r'][0]).max())
        assert err<2e-5,(ident,mode,err)
        checks[f'{mode}_existing_s50_error']=err
        for population in before:
            v0,v1=before[population],after[population];use=v0['valid']&v1['valid']
            row=dict(**ident,arm='existing_full_50',mode=mode,kind='saved_50',population=population,ce0=ce0,accuracy0=acc0)
            row.update({f'mu_{k}':v for k,v in summarize(v1['mu']-v0['mu']).items()})
            row.update({f'r_{k}':v for k,v in summarize(v1['r']-v0['r'],use).items()});rows.append(row)
        assert local.frozen_digest()==frozen
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'arrays.npz',**arrays)
    columns=list(dict.fromkeys(k for row in rows for k in row));T.write_rows(folder/'rows.csv',[{k:row.get(k) for k in columns} for row in rows])
    paths=[Path(__file__),Path(__file__).with_name('ordinary_drive.py'),T.ROOT/'specs/spec_vit_two_block_token_drive_1025.md']
    T.atomic_json(folder/'complete.json',dict(status='complete',**ident,checks=checks,checkpoint=str(checkpoint),
        checkpoint_sha256=T.sha256(checkpoint),frozen_unchanged=True,
        source_hashes={str(p.relative_to(T.ROOT)):T.sha256(p) for p in paths},
        git_hash=T.git_state()[0],arrays_sha256=T.sha256(folder/'arrays.npz'),updated=time.time()))
    print(json.dumps(dict(completed=ident,checks=checks)),flush=True)


if __name__=='__main__':
    T.init();assert not T.git_state()[1]
    for task in (10,60):
        for start in (100,500):
            for seed in T.SEEDS:
                for layer in (1,2):run(seed,task,start,layer)
