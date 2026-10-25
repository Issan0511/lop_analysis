"""Image-dependent gradient covariance and full-batch continuation controls."""
import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import measure as X

M,torch,np,pd,F,ROOT=X.M,X.torch,X.np,X.pd,X.F,X.ROOT


def covariance(m,x,y,xp,classes,opt,seed,task,stage,rawdir):
    checks,rows,comparisons,steps,arrays={},{},[],[],{}
    rows=[]
    raw=M.probe_fields(m,xp)
    with torch.no_grad():
        h=M.features(m,x).double()
        w=m.layers['out'].weight[classes].double()
        b=m.layers['out'].bias[classes].double()
        v=w-w.mean(0)
        p=(h@w.T+b).softmax(1)
        q=F.one_hot(y,10).double()
        e=(p-q)@v
        sig=dict(fit=e/len(x),mean=e.mean(0,keepdim=True).expand_as(e)/len(x),
                 covariance=(e-e.mean(0))/len(x),
                 pred_cov=(p-p.mean(0))@v/len(x),target_cov=(q-q.mean(0))@v/len(x))
        for j in range(4):
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(1025425+seed*1000+task*10+j)).to(x.device)
            sig[f'perm{j}']=e[order]/len(x)
    gs=X.backward(m,x,sig)
    saved={k:w.detach().clone() for k,w in m.named_parameters()}
    if opt is not None: state,den,pg=X.adam_fields(m,opt)
    for metric in (['SGD','Adam_fixed'] if opt is not None else ['SGD']):
        projected={}
        for name,gg in gs.items():
            tangent=gg if metric=='SGD' else {k:g/den[k] for k,g in gg.items()}
            values=X.project(m,xp,raw,tangent)
            arrays[f'{name}_{metric}']=values
            projected[name]=values
            for i in range(6):
                valid=raw[i]['valid'].cpu().numpy()
                a=values[i,2][valid]
                rows.append(dict(metric=metric,signal=name,layer=i+1,rms=float(np.sqrt(np.mean(a*a))),
                                 **M.stats(values[i,2],valid)))
        scale=max(np.abs(projected['pred_cov']).max(),np.abs(projected['target_cov']).max(),1e-12)
        for key,diff in [('mean_cov',projected['fit']-projected['mean']-projected['covariance']),
                         ('pred_target',projected['covariance']-projected['pred_cov']+projected['target_cov'])]:
            err=float(np.abs(diff).max()/scale)
            checks[f'{metric}_{key}_error']=err
            assert err<2e-3
        for i in range(6):
            actual=projected['fit'][i,2]
            valid=raw[i]['valid'].cpu().numpy()
            keep=valid & (actual>max(1e-12,1e-6*np.max(np.abs(actual))))
            mean=projected['mean'][i,2]
            erased=float((mean[keep]<=max(1e-12,1e-6*np.max(np.abs(mean)))).mean()) if keep.any() else None
            for name,values in projected.items():
                if name=='fit': continue
                comparisons.append(dict(metric=metric,layer=i+1,prediction=name,mean_erases_sinking_fraction=erased,
                                        **X.compare(values[i,2],actual,valid)))
        if opt is not None and metric=='Adam_fixed':
            prior=np.load(rawdir/f'diagnostic_seed{seed}_t{task}.npz')
            X.checked(checks,'baseline_fit_reproduction',torch.from_numpy(projected['fit']),
                      torch.from_numpy(prior['old_actual_fit_Adam_fixed']))
    if opt is not None:
        for name in ['fit','mean','covariance','perm0','perm1','perm2','perm3']:
            delta=X.adam_delta(gs[name],state,pg)
            with torch.no_grad():
                for k,w in m.named_parameters(): w.copy_(saved[k]+delta[k])
            after=M.probe_fields(m,xp)
            for i,r in enumerate(after):
                dr=(r['r']-raw[i]['r']).cpu().numpy()
                valid=(r['valid']&raw[i]['valid']).cpu().numpy()
                arrays[f'finite_{name}_L{i+1}']=dr
                steps.append(dict(signal=name,layer=i+1,delta_r_mean=float(dr[valid].mean()),
                                  delta_r_median=float(np.median(dr[valid])),decreased_fraction=float((dr[valid]<0).mean())))
            with torch.no_grad():
                for k,w in m.named_parameters(): w.copy_(saved[k])
    np.savez_compressed(rawdir/f'covariance_seed{seed}_t{task}_{stage}.npz',**arrays)
    return rows,comparisons,steps,checks


def fullbatch(m,opt,x,y,xp,classes,seed,task,rawdir):
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,foreach=False)
    optimizer.load_state_dict(copy.deepcopy(opt))
    initial=M.probe_fields(m,xp)
    rows,arrays=[],{}
    for step in range(51):
        if step in (0,1,5,20,50):
            raw=M.probe_fields(m,xp)
            with torch.no_grad():
                h=M.features(m,x)
                ce=float(F.cross_entropy(h@m.layers['out'].weight[classes].T+m.layers['out'].bias[classes],y))
            for i,r in enumerate(raw):
                dr=(r['r']-initial[i]['r']).cpu().numpy()
                valid=(r['valid']&initial[i]['valid']).cpu().numpy()
                arrays[f'step{step}_L{i+1}_delta']=dr
                rows.append(dict(step=step,layer=i+1,ce=ce,delta_r_mean=float(dr[valid].mean()),
                                 delta_r_median=float(np.median(dr[valid])),decreased_fraction=float((dr[valid]<0).mean())))
        if step==50: break
        optimizer.zero_grad(set_to_none=True)
        for lo in range(0,len(x),16):
            loss=F.cross_entropy(m(x[lo:lo+16])[:,classes],y[lo:lo+16],reduction='sum')/len(x)
            loss.backward()
        optimizer.step()
    np.savez_compressed(rawdir/f'fullbatch_seed{seed}_t{task}.npz',**arrays)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['covariance','fullbatch'],required=True)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--archive-root',type=Path,required=True)
    ap.add_argument('--raw-output',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    assert Path(C.__file__).resolve().is_relative_to(args.source_root.resolve())
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.45)
    source=C.RealImages()
    accum={k:[] for k in ('pressures','comparisons','steps','checks','fullbatch','provenance')}
    begin=time.monotonic()
    for seed in range(5):
        archive=args.archive_root/f'vit_long_1024_{"vn" if seed<3 else "il"}_raw_seed{seed}.tar'
        xp=C.eval_tensor(source.val(C.probe_names(source,seed)),'cuda')
        with tarfile.open(archive,'r:') as tar:
            def read(name): return tar.extractfile(f'vit_long_1024/runs/SL/seed{seed}/{name}').read()
            prov=json.loads(read('provenance.json'))
            for name,digest in prov['source_hashes'].items(): assert M.sha((args.source_root/name).read_bytes())==digest
            manifest={r['path']:r for r in json.loads(read('manifest.json'))}
            m=make_model('R',seed,'cuda',prov['config']['model_config']).eval()
            for task in (10,60):
                data,_=C.task_dataset(source,seed,task,10,25,augment=False)
                x,y=C.eval_tensor(data.images,'cuda'),data.labels.to('cuda')
                classes=C.loss_classes('SL',seed,task,10,25)
                for stage in (['mid','end'] if args.mode=='covariance' else ['end']):
                    ident=dict(seed=seed,task=task,stage=stage)
                    name=f'ckpt/{"mid" if stage=="mid" else "model"}_t{task:02d}.pt'
                    payload=read(name)
                    assert M.sha(payload)==manifest[name]['sha256']
                    accum['provenance'].append(dict(**ident,archive=str(archive),member=name,sha256=M.sha(payload)))
                    m.load_state_dict(torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True))
                    opt=None
                    if stage=='end':
                        name=f'ckpt/adam_t{task:02d}.pt'
                        payload=read(name)
                        assert M.sha(payload)==manifest[name]['sha256']
                        opt=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)['optimizer']
                        accum['provenance'].append(dict(**ident,archive=str(archive),member=name,sha256=M.sha(payload)))
                    if args.mode=='covariance':
                        res=covariance(m,x,y,xp,classes,opt,seed,task,stage,args.raw_output)
                        for name,rr in zip(('pressures','comparisons','steps','checks'),res):
                            accum[name].extend(dict(**ident,**r) for r in (rr if isinstance(rr,list) else [rr]))
                    else:
                        rr=fullbatch(m,opt,x,y,xp,classes,seed,task,args.raw_output)
                        accum['fullbatch'].extend(dict(**ident,**r) for r in rr)
                    for name,rr in accum.items():
                        if not rr: continue
                        if name in ('checks','provenance'): M.dump(args.output/f'{args.mode}_{name}.json',rr)
                        else: M.csv_write(args.output/f'{args.mode}_{name}.csv',rr)
                    print(json.dumps(dict(mode=args.mode,**ident,seconds=round(time.monotonic()-begin,1))),flush=True)
                del opt,x,y
            del m
            torch.cuda.empty_cache()
    M.dump(args.output/f'{args.mode}_run.json',dict(status='complete',states=20 if args.mode=='covariance' else 10,
        seconds=time.monotonic()-begin,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))


if __name__=='__main__': main()
