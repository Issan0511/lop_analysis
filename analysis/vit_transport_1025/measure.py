"""Blockwise parameter transport and finite Adam trajectory counterfactuals."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mechanisms', ROOT/'analysis/vit_mechanisms_1025/measure.py')
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
M, torch, np, pd, F = X.M, X.torch, X.np, X.pd, X.F
DATA = Path('/home/issan/Projects/obsidian-research-data')


def group_name(key):
    match = re.match(r'layers\.block_(\d+)\.layers\.(.*)', key)
    if match:
        i, suffix = int(match[1])+1, match[2]
        if suffix.startswith(('attn.', 'norm1.')): part = 'attention'
        elif suffix.startswith('norm2.'): part = 'norm2'
        elif suffix.startswith('mlp.layers.fc1.'): part = 'fc1'
        elif suffix.startswith('mlp.layers.fc2.'): part = 'fc2'
        else: raise ValueError(key)
        return f'L{i}_{part}'
    if key.startswith(('layers.norm.', 'layers.out.')): return 'tail'
    assert key in ('cls_token', 'pos_embed') or key.startswith('layers.patch_embed.'), key
    return 'embedding'


def is_upstream(group, layer):
    if group == 'embedding': return True
    if group == 'tail': return False
    index, part = group[1:].split('_', 1)
    return int(index) < layer or (int(index) == layer and part in ('attention', 'norm2'))


def channel_stats(values, raw, layer):
    valid = raw[layer]['valid'].cpu().numpy()
    mu = raw[layer]['mu'].cpu().numpy()
    sd = raw[layer]['sd'].cpu().numpy()+1e-12
    a = values[layer]
    return dict(n=int(valid.sum()), mean=float(a[2, valid].mean()),
                median=float(np.median(a[2, valid])), rms=float(np.sqrt(np.mean(a[2, valid]**2))),
                sink_fraction=float((a[2, valid]>0).mean()),
                mu_mean=float(a[0, valid].mean()), sd_mean=float(a[1, valid].mean()),
                mean_channel=float((a[0]/sd)[valid].mean()),
                width_channel=float((-mu*a[1]/sd**2)[valid].mean()))


def carrier(m, opt, x, y, xn, xp, classes, seed, task, rawdir):
    raw = M.probe_fields(m, xp)
    with torch.no_grad():
        h, hn = M.features(m, x), M.features(m, xn)
        w, b = m.layers['out'].weight[classes], m.layers['out'].bias[classes]
        vc = w-w.mean(0)
        loading, _, _ = M.loading(h, y, vc)
        p, pn = (h@w.T+b).softmax(1), (hn@w.T+b).softmax(1)
        signals_old = {'fit': (p-F.one_hot(y, 10))@vc/len(x)}
        signals_next = {'switch': (pn-.1)@vc/len(xn), 'load': (pn-.1)@loading/len(xn)}
        ce = float(F.cross_entropy(h@w.T+b, y))
    reference = pd.read_csv(ROOT/'results/vit_phases_1025/states.csv')
    rr = reference.query('arm=="SL" and seed==@seed and task==@task and stage=="end"').iloc[0]
    assert abs(ce-rr.ce)<2e-6
    gradients = X.backward(m, x, signals_old) | X.backward(m, xn, signals_next)
    _, den, _ = X.adam_fields(m, opt)
    groups = {}
    for key, w in m.named_parameters(): groups.setdefault(group_name(key), []).append(key)
    assert len(groups)==26 and sum(map(len, groups.values()))==len(list(m.parameters()))
    rows, ratios, arrays, checks = [], [], {}, {'baseline_ce_abs_error': abs(ce-rr.ce)}
    zeros = {k: torch.zeros_like(w) for k,w in m.named_parameters()}
    for signal, gg in gradients.items():
        for metric in ('SGD', 'Adam_fixed'):
            tangent = gg if metric=='SGD' else {k:g/den[k] for k,g in gg.items()}
            full = X.project(m, xp, raw, tangent)
            projected = {}
            for group, keys in groups.items():
                tg = dict(zeros)
                for k in keys: tg[k] = tangent[k]
                values = X.project(m, xp, raw, tg)
                projected[group] = values
                arrays[f'{signal}_{metric}_{group}'] = values
                for i in range(6):
                    if not is_upstream(group, i+1) and group!=f'L{i+1}_fc1':
                        assert np.abs(values[i]).max()<1e-12, (group, i)
                    rows.append(dict(signal=signal, metric=metric, layer=i+1, group=group,
                                     **channel_stats(values, raw, i)))
            scale = max(max(np.abs(v).max() for v in projected.values()), 1e-12)
            error = float(np.abs(sum(projected.values())-full).max()/scale)
            checks[f'{signal}_{metric}_ledger'] = error
            assert error<2e-3
            arrays[f'{signal}_{metric}_full'] = full
            for i,r in enumerate(raw):
                own = projected[f'L{i+1}_fc1'][i]
                upstream = sum(v[i] for g,v in projected.items() if is_upstream(g, i+1))
                prefix = f'layers.block_{i}.layers.mlp.layers.fc1.'
                gw, gb = tangent[prefix+'weight'].double(), tangent[prefix+'bias'].double()
                dm = gw@r['mean_u']+gb
                ds = (r['cov_uz']*gw).sum(1)/r['sd'].clamp_min(1e-12)
                analytic = torch.stack([dm, ds, dm/(r['sd']+1e-12)-r['mu']*ds/(r['sd']+1e-12)**2]).cpu().numpy()
                error = float(np.abs(own-analytic).max()/max(np.abs(analytic).max(), 1e-12))
                checks[f'{signal}_{metric}_L{i+1}_own'] = error
                assert error<2e-3
                valid = r['valid'].cpu().numpy()
                un, ln = np.linalg.norm(upstream[2, valid]), np.linalg.norm(own[2, valid])
                ratios.append(dict(signal=signal, metric=metric, layer=i+1,
                                   upstream_norm_share=float(un/(un+ln)) if un+ln>0 else None,
                                   full_mean=float(full[i,2,valid].mean()),
                                   own_mean=float(own[2,valid].mean()),
                                   upstream_mean=float(upstream[2,valid].mean())))
            if metric=='Adam_fixed':
                folder = 'adam_raw' if signal=='fit' else 'adam_next_raw'
                with np.load(DATA/'vit_phases_1025'/folder/f'SL_seed{seed}_t{task}.npz') as prior:
                    old = np.array([[prior[f'L{i}_{signal}_full_{mo}'] for mo in ('mu','sd','r')] for i in range(1,7)])
                error = float(np.abs(old-full).max()/max(np.abs(old).max(),1e-12))
                checks[f'{signal}_prior_reproduction'] = error
                assert error<2e-3
    np.savez_compressed(rawdir/f'carriers_seed{seed}_t{task}.npz', **arrays)
    return rows, ratios, checks, groups


@torch.no_grad()
def frame(m, xp, checks):
    inputs={i:[] for i in range(6)}
    hooks=[]
    for i in range(6):
        fc=m.layers[f'block_{i}'].layers['mlp'].layers['fc1']
        hooks.append(fc.register_forward_pre_hook(lambda module,args,i=i: inputs[i].append(args[0].detach())))
    try:
        out = [M.body(m, xx, True) for xx in xp.split(32)]
    finally:
        for h in hooks: h.remove()
    frames = []
    for i in range(6):
        u = torch.cat(inputs[i])
        fc = m.layers[f'block_{i}'].layers['mlp'].layers['fc1']
        w, b = fc.weight.detach().clone(), fc.bias.detach().clone()
        original_z=torch.cat([a[2][i] for a in out]).double()
        z = hybrid_z(u,w,b)
        assert torch.equal(z,original_z)
        error = float((u[:,0].double()@w.double().T+b.double()-original_z).abs().max())
        checks['affine_reconstruction_max_abs'] = max(checks.get('affine_reconstruction_max_abs',0),error)
        assert error<2e-5
        frames.append(dict(u=u,w=w,b=b,z=z,mom=moments(z),actual_mom=moments(original_z)))
    return frames


def hybrid_z(u,w,b):
    # Preserve the original float32 Linear and its full-token batch shape.
    # CLS-only float64 re-evaluation changes r noticeably when SD is tiny.
    return torch.cat([F.linear(xx,w,b)[:,0] for xx in u.split(32)]).double()


def moments(z):
    mu, sd = z.mean(0), z.std(0,unbiased=True)
    assert bool((sd>1e-8).all())
    return torch.stack([mu, sd, mu/(sd+1e-12)])


def edge(a,b):
    ai, bi = 1/(a[1]+1e-12), 1/(b[1]+1e-12)
    return torch.stack([.5*(a[0]-b[0])*(ai+bi), .5*(a[0]+b[0])*(ai-bi)])*-1


def split_step(before, after, checks):
    pieces=[]
    for a,b in zip(before,after):
        aa,bb = a['mom'],b['mom']
        ba = moments(hybrid_z(a['u'],b['w'],b['b']))
        ab = moments(hybrid_z(b['u'],a['w'],a['b']))
        own = .5*(edge(aa,ba)+edge(ab,bb))
        upstream = .5*(edge(aa,ab)+edge(ba,bb))
        vals = torch.cat([own,upstream])
        err = float((vals.sum(0)-(bb[2]-aa[2])).abs().max())
        scale = max(float(vals.abs().max()),1e-12)
        checks['finite_ledger_max_abs'] = max(checks.get('finite_ledger_max_abs',0),err)
        assert err<1e-10+1e-8*scale
        pieces.append(vals)
    return torch.stack(pieces)


def trajectories(m, base, opt, x, y, xp, classes, seed, task, rawdir, interventions=False):
    generator = torch.Generator().manual_seed(1025125+seed*100+task)
    batches=[]
    while len(batches)<50:
        batches.extend(torch.randperm(len(x),generator=generator).split(128))
    newy=torch.randint(10,(len(y),),generator=torch.Generator().manual_seed(1025225+seed*100+task)).to(y.device)
    variants = [(0,'baseline')] if not interventions else [(i,mode) for i in (1,6) for mode in ('freeze_own','freeze_upstream')]
    rows, checks = [], {}
    for label_arm, yy in [('continue',y),('relabel',newy)]:
        for target, variant in variants:
            m.load_state_dict(base)
            frozen=[]
            for k,w in m.named_parameters():
                g=group_name(k)
                freeze = ((variant=='freeze_own' and g==f'L{target}_fc1') or
                          (variant=='freeze_upstream' and is_upstream(g,target)))
                w.requires_grad_(not freeze)
                w.grad=None
                if freeze: frozen.append(k)
            optimizer=torch.optim.Adam(m.parameters(),lr=1e-4,foreach=False)
            optimizer.load_state_dict(copy.deepcopy(opt))
            initial = frame(m,xp,checks)
            previous = initial
            cum = torch.zeros(6,4,1536,device=x.device,dtype=torch.float64)
            all_steps=[]
            arrays={}
            for step in range(51):
                if step:
                    batch=batches[step-1].to(x.device)
                    optimizer.zero_grad(set_to_none=True)
                    for ids in batch.split(16):
                        loss=F.cross_entropy(m(x[ids])[:,classes],yy[ids],reduction='sum')/len(batch)
                        loss.backward()
                    optimizer.step()
                if not interventions:
                    current=initial if not step else frame(m,xp,checks)
                    if step:
                        terms=split_step(previous,current,checks)
                        cum+=terms
                        all_steps.append(terms.cpu().numpy())
                    previous=current
                elif step in (0,1,5,20,50): current=initial if not step else frame(m,xp,checks)
                if step not in (0,1,5,20,50): continue
                with torch.no_grad():
                    h=M.features(m,x)
                    logits=h@m.layers['out'].weight[classes].T+m.layers['out'].bias[classes]
                    ce=float(F.cross_entropy(logits,yy))
                    accuracy=float((logits.argmax(1)==yy).float().mean())
                for i,fr in enumerate(current):
                    delta=fr['mom'][2]-initial[i]['mom'][2]
                    values=delta.cpu().numpy()
                    row=dict(label_arm=label_arm,variant=variant,target=target,step=step,layer=i+1,
                             ce=ce,accuracy=accuracy,delta_mean=float(values.mean()),
                             delta_median=float(np.median(values)),decreased_fraction=float((values<0).mean()),
                             mu_mean=float(fr['mom'][0].mean()),sd_mean=float(fr['mom'][1].mean()))
                    arrays[f'step{step}_L{i+1}_mom']=fr['mom'].cpu().numpy()
                    if not interventions:
                        err=float((cum[i].sum(0)-delta).abs().max())
                        checks['integral_max_abs']=max(checks.get('integral_max_abs',0),err)
                        assert err<1e-9
                        for j,name in enumerate(('own_mean','own_width','upstream_mean','upstream_width')):
                            row[name]=float(cum[i,j].mean())
                        arrays[f'step{step}_L{i+1}_integral']=cum[i].cpu().numpy()
                    rows.append(row)
                if not interventions:
                    with np.load(DATA/'vit_mechanisms_1025/raw'/f'rollout_seed{seed}_t{task}.npz') as prior:
                        error=max(float(np.max(np.abs(current[i]['mom'][2].cpu().numpy()-prior[f'{label_arm}_step{step}_L{i+1}_r']))) for i in range(6))
                    checks[f'{label_arm}_step{step}_prior_r']=error
                    if error>=2e-5:
                        with np.load(DATA/'vit_mechanisms_1025/raw'/f'rollout_seed{seed}_t{task}.npz') as prior:
                            actual_error=max(float(np.max(np.abs(current[i]['actual_mom'][2].cpu().numpy()-prior[f'{label_arm}_step{step}_L{i+1}_r']))) for i in range(6))
                        raise AssertionError(dict(arm=label_arm,step=step,reconstructed_r_error=error,original_float32_r_error=actual_error))
            for k,w in m.named_parameters():
                if k in frozen:
                    assert torch.equal(w.detach().cpu(),base[k]), k
            if not interventions: arrays['step_terms']=np.stack(all_steps)
            mode='freeze' if interventions else 'replay'
            np.savez_compressed(rawdir/f'{mode}_seed{seed}_t{task}_{label_arm}_{variant}_L{target}.npz',**arrays)
            print(json.dumps(dict(mode=mode,seed=seed,task=task,label_arm=label_arm,variant=variant,target=target,status='trajectory_complete')),flush=True)
            del optimizer,initial,previous,current,arrays,all_steps
    for w in m.parameters(): w.requires_grad_(True)
    checks['frozen_parameter_checks']='passed'
    return rows,checks


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('gate','carriers','replay','freeze'),required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    ap.add_argument('--tasks',type=int,nargs='+',default=[10,60])
    ap.add_argument('--source-root',type=Path,default=DATA/'vit_ffn_ablation_1025/source')
    ap.add_argument('--archive-root',type=Path,default=DATA/'vit_long_1024/_gigafile_dl')
    ap.add_argument('--raw-output',type=Path,default=DATA/'vit_transport_1025/raw')
    ap.add_argument('--output',type=Path,default=ROOT/'results/vit_transport_1025')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.raw_output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    assert Path(C.__file__).resolve().is_relative_to(args.source_root.resolve())
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.45)
    source=C.RealImages()
    allrows={}; checks=[]; provenance=[]
    begin=time.monotonic(); completed=0
    for seed in args.seeds:
        archive=args.archive_root/f'vit_long_1024_{"vn" if seed<3 else "il"}_raw_seed{seed}.tar'
        xp=C.eval_tensor(source.val(C.probe_names(source,seed)),'cuda')
        with tarfile.open(archive,'r:') as tar:
            def read(name): return tar.extractfile(f'vit_long_1024/runs/SL/seed{seed}/{name}').read()
            prov=json.loads(read('provenance.json'))
            for name,digest in prov['source_hashes'].items(): assert M.sha((args.source_root/name).read_bytes())==digest
            manifest={r['path']:r for r in json.loads(read('manifest.json'))}
            m=make_model('R',seed,'cuda',prov['config']['model_config']).eval()
            for task in args.tasks:
                ident=dict(seed=seed,task=task)
                loaded={}
                for kind in ('model','adam'):
                    name=f'ckpt/{kind}_t{task:02d}.pt'; payload=read(name)
                    digest=M.sha(payload); assert digest==manifest[name]['sha256']
                    loaded[kind]=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
                    provenance.append(dict(**ident,archive=str(archive),member=name,sha256=digest))
                m.load_state_dict(loaded['model'])
                data,_=C.task_dataset(source,seed,task,10,25,augment=False)
                x,y=C.eval_tensor(data.images,'cuda'),data.labels.to('cuda')
                classes=C.loss_classes('SL',seed,task,10,25)
                if args.mode in ('gate','carriers'):
                    nxt,_=C.task_dataset(source,seed,task+1,10,25,augment=False)
                    xn=C.eval_tensor(nxt.images,'cuda')
                    rr,ratios,cc,groups=carrier(m,loaded['adam']['optimizer'],x,y,xn,xp,classes,seed,task,args.raw_output)
                    for key,values in [('carriers',rr),('ratios',ratios)]:
                        allrows.setdefault(key,[]).extend(dict(**ident,**r) for r in values)
                    checks.append(dict(**ident,mode='carriers',**cc))
                    M.dump(args.output/'parameter_groups.json',groups)
                    del xn
                if args.mode in ('gate','replay','freeze'):
                    rr,cc=trajectories(m,loaded['model'],loaded['adam']['optimizer'],x,y,xp,classes,
                                       seed,task,args.raw_output,interventions=args.mode=='freeze')
                    allrows.setdefault(args.mode,[]).extend(dict(**ident,**r) for r in rr)
                    checks.append(dict(**ident,mode=args.mode,**cc))
                completed+=1
                for key,rr in allrows.items(): M.csv_write(args.output/f'{key}.csv',rr)
                M.dump(args.output/f'{args.mode}_checks.json',checks)
                M.dump(args.output/f'{args.mode}_provenance.json',provenance)
                print(json.dumps(dict(mode=args.mode,**ident,states=completed,seconds=round(time.monotonic()-begin,1))),flush=True)
                del loaded,x,y
            del m
            torch.cuda.empty_cache()
    M.dump(args.output/f'{args.mode}_run.json',dict(status='complete',states=completed,seconds=time.monotonic()-begin,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))


if __name__=='__main__': main()
