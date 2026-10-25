"""Readout interventions, conditional joint-fit audit, and actual Adam continuations."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('phase_base', ROOT / 'analysis/vit_phases_1025/measure.py')
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)
torch, np, pd, F = M.torch, M.np, M.pd, M.F


def rel(a, b):
    return float((a - b).abs().max()) / max(float(a.abs().max()), float(b.abs().max()), 1e-12)


def checked(checks, key, a, b, tol=2e-3):
    checks[key] = rel(a, b)
    assert checks[key] <= tol, (key, checks[key])


def compare(pred, actual, valid):
    keep = valid & (np.abs(actual) > max(1e-12, 1e-6 * np.max(np.abs(actual))))
    a, p = actual[keep], pred[keep]
    ptol = max(1e-12, 1e-6 * np.max(np.abs(pred)))
    pos, neg = a > 0, a < 0
    ba = .5 * ((p[pos] > ptol).mean() + (p[neg] < -ptol).mean()) if pos.sum() >= 10 and neg.sum() >= 10 else None
    rho = pd.Series(p).rank().corr(pd.Series(a).rank()) if len(p) and np.std(p) > 0 else None
    return dict(n=int(keep.sum()), balanced_accuracy=ba, spearman=rho,
                sign_agreement=float(((p > ptol) & pos | (p < -ptol) & neg).mean()) if len(p) else None,
                relative_l2_error=float(np.linalg.norm(p-a) / max(np.linalg.norm(a), 1e-12)))


def backward(m, x, signals):
    params = dict(m.named_parameters())
    gs = {name: {k: torch.zeros_like(w) for k, w in params.items()} for name in signals}
    for start in range(0, len(x), 16):
        h = M.body(m, x[start:start+16])[0]
        for j, (name, sig) in enumerate(signals.items()):
            gg = torch.autograd.grad(h, list(params.values()), grad_outputs=sig[start:start+len(h)].float(),
                                     retain_graph=j+1 < len(signals), allow_unused=True)
            for (key, _), g in zip(params.items(), gg):
                if g is not None:
                    gs[name][key] += g.detach()
    return gs


def project(m, x, raw, tangent):
    wrapper = M.Moments(m, raw)
    params = {f'model.{k}': w.detach() for k, w in m.named_parameters()}
    tangents = {f'model.{k}': g for k, g in tangent.items()}
    dmom = torch.zeros(12, 1536, device=x.device, dtype=torch.float64)
    for xx in x.split(16):
        with torch.no_grad():
            d = torch.func.jvp(lambda p: torch.func.functional_call(wrapper, p, (xx,)), (params,), (tangents,))[1]
        dmom += d.double() * len(xx)/len(x)
    out = []
    for i, r in enumerate(raw):
        dm = dmom[2*i]
        ds = dmom[2*i+1] * len(x)/(len(x)-1)/(2*r['sd'].clamp_min(1e-12))
        dr = dm/(r['sd']+1e-12) - r['mu']*ds/(r['sd']+1e-12).square()
        out.append(torch.stack([dm, ds, dr]))
    return torch.stack(out).cpu().numpy()


def adam_fields(m, opt):
    pg = opt['param_groups'][0]
    assert len(pg['params']) == len(list(m.parameters()))
    states, den = {}, {}
    for index, (key, w) in zip(pg['params'], m.named_parameters()):
        ss = opt['state'][index]
        assert ss['exp_avg_sq'].shape == w.shape
        states[key] = dict(m=ss['exp_avg'].to(w.device), v=ss['exp_avg_sq'].to(w.device), t=float(ss['step']))
        den[key] = (states[key]['v']/(1-pg['betas'][1]**states[key]['t'])).sqrt()+pg['eps']
    return states, den, pg


def adam_delta(gs, state, pg):
    b1, b2 = pg['betas']
    delta = {}
    for k, g in gs.items():
        ss = state[k]
        mm = (b1*ss['m']+(1-b1)*g)/(1-b1**(ss['t']+1))
        vv = (b2*ss['v']+(1-b2)*g.square())/(1-b2**(ss['t']+1))
        delta[k] = -pg['lr']*mm/(vv.sqrt()+pg['eps'])
    return delta


@torch.no_grad()
def feature_signals(h, hn, w, b, y, seed, task, checks):
    h, hn, w, b = (a.double() for a in (h, hn, w, b))
    n, c, dim = len(h), len(w), h.shape[1]
    q = F.one_hot(y, c).double()
    vc, bc = w-w.mean(0), b-b.mean()
    ww = torch.cat([vc.T, bc[None]], 0)
    ha, hna = torch.cat([h, torch.ones_like(h[:, :1])], 1), torch.cat([hn, torch.ones_like(hn[:, :1])], 1)
    target = c*(q-1/c)
    u, sv, vt = torch.linalg.svd(ha, full_matrices=False)
    rank = {tol: int((sv.square() > tol*sv[0].square()).sum()) for tol in (1e-6, 1e-8, 1e-10)}
    k = sv.square() > 1e-8*sv[0].square()
    pinv = (vt[k].T/sv[k]) @ u[:, k].T
    gi = pinv @ pinv.T
    ws = pinv @ target
    z, zs = ha@ww, ha@ws
    p = z.softmax(1)
    old = dict(actual_fit=(p-q)@vc/n, actual_switch=(p-1/c)@vc/n,
               sq_fit=(z-target)@vc/(c*n), sq_switch=z@vc/(c*n),
               ls_fit=(zs-target)@ws[:-1].T/(c*n), ls_switch=zs@ws[:-1].T/(c*n),
               P2_iid=90*ha@gi[:, :dim]/(c*n), LS_refit_CE=(zs.softmax(1)-q)@ws[:-1].T/n)
    old['fit_gap'] = old['sq_fit']-old['ls_fit']
    old['switch_gap'] = old['sq_switch']-old['ls_switch']
    old['ce'] = old['actual_fit']-old['sq_fit']
    old['task_departure'] = old['ls_switch']-old['P2_iid']
    checked(checks, 'feature_ce_ledger', old['actual_switch'], old['sq_switch']+old['ce'], 1e-8)
    checked(checks, 'normal_equation', u[:, k].T @ zs, u[:, k].T @ target, 1e-8)
    # Orthogonal residual times the retained pseudoinverse is exactly zero.
    resid_projector = u[:, k]@u[:, k].T-torch.eye(n, device=h.device, dtype=h.dtype)
    checked(checks, 'ordinary_iid_mean_zero', resid_projector@pinv.T+pinv.T, pinv.T, 1e-8)
    checked(checks, 'covariance_identity', gi, (vt[k].T/sv[k].square())@vt[k], 1e-8)
    load = (q-1/c).T@h
    kap = (vc*load).sum()/load.square().sum()
    load = kap*load
    remain = vc-load
    checked(checks, 'alignment_orthogonality', vc.square().sum(), load.square().sum()+remain.square().sum(), 1e-8)
    heads = dict(base=vc, alpha_half=remain+.5*load, alpha_zero=remain,
                 norm_unload=remain*vc.norm()/remain.norm())
    for j in range(4):
        gen = torch.Generator().manual_seed(1025025+seed*1000+task*10+j)
        perm = torch.randperm(n, generator=gen).to(h.device)
        vv = (q[perm]-1/c).T@h
        for direction in (remain, load):
            vv -= (vv*direction).sum()/direction.square().sum()*direction
        vv *= load.norm()/vv.norm()
        heads[f'perm{j}'] = remain+vv
        assert abs(float(heads[f'perm{j}'].norm()/vc.norm())-1) < 1e-8
        assert abs(float((heads[f'perm{j}']*load).sum()/load.square().sum())) < 1e-8
    new, head_info = {}, []
    for name, vv in heads.items():
        pp = (hn@vv.T+bc).softmax(1)
        new[name] = (pp-1/c)@vv/len(hn)
        head_info.append(dict(variant=name, norm_ratio=float(vv.norm()/vc.norm()),
                              aligned_coefficient=float((vv*load).sum()/load.square().sum()),
                              next_pmax=float(pp.max(1).values.mean()),
                              old_ce=float(F.cross_entropy(h@vv.T+bc, y))))
    new.update(P2_iid=90*hna@gi[:, :dim]/(c*len(hn)),
               ls_switch=(hna@ws)@ws[:-1].T/(c*len(hn)),
               iso=hn/hn.norm(), gram=(hna@(ha.T@ha))[:, :dim])
    new['gram'] /= new['gram'].norm()
    # Record Monte Carlo means of independently refitted heads, without fitting a scale to real gradients.
    gen = torch.Generator().manual_seed(1025325+seed*100+task)
    mco, mcn = [], []
    for half in range(2):
        so, sn = torch.zeros_like(h), torch.zeros_like(hn)
        for _ in range(128):
            yy = F.one_hot(torch.randint(c, (n,), generator=gen).to(h.device), c).double()
            vstar = pinv@(c*(yy-1/c))
            so += (ha@vstar)@vstar[:-1].T/(c*n*128)
            sn += (hna@vstar)@vstar[:-1].T/(c*len(hn)*128)
        mco.append(so)
        mcn.append(sn)
    old['P2_mc'] = (mco[0]+mco[1])/2
    new['P2_mc'] = (mcn[0]+mcn[1])/2
    # Sensitivity keeps the old feature Gram even when evaluation inputs change.
    for tol in (1e-6, 1e-10):
        if rank[tol] == rank[1e-8]:
            continue
        kk = sv.square() > tol*sv[0].square()
        gii = (vt[kk].T/sv[kk].square())@vt[kk]
        old[f'P2_tol{tol:g}'] = 90*ha@gii[:, :dim]/(c*n)
        new[f'P2_tol{tol:g}'] = 90*hna@gii[:, :dim]/(c*len(hn))
    info = dict(rank=rank[1e-8], rank_1e6=rank[1e-6], rank_1e10=rank[1e-10],
                kappa=float(kap), head_alignment_cos2=float(load.square().sum()/vc.square().sum()),
                ce=float(F.cross_entropy(z,y)), accuracy=float((z.argmax(1)==y).double().mean()),
                ls_residual_ratio=float((zs-target).norm()/target.norm()),
                ls_prediction_gap=float((z-zs).norm()/zs.norm()),
                ce_correction_ratio=float(old['ce'].norm()/old['actual_fit'].norm()),
                ls_fit_signal_ratio=float(old['ls_fit'].norm()/old['actual_fit'].norm()),
                mc_old_rel=float((old['P2_mc']-old['P2_iid']).norm()/old['P2_iid'].norm()),
                mc_next_rel=float((new['P2_mc']-new['P2_iid']).norm()/new['P2_iid'].norm()),
                mc_old_split_rel=float((mco[0]-mco[1]).norm()/old['P2_iid'].norm()),
                mc_next_split_rel=float((mcn[0]-mcn[1]).norm()/new['P2_iid'].norm()))
    return old, new, info, head_info


def diagnostic(m, opt, x, y, xn, xp, classes, seed, task, rawdir):
    checks = {}
    raw = M.probe_fields(m, xp)
    state, denominator, pg = adam_fields(m, opt)
    with torch.no_grad():
        oldsig, newsig, info, heads = feature_signals(M.features(m,x), M.features(m,xn),
            m.layers['out'].weight[classes], m.layers['out'].bias[classes], y, seed, task, checks)
    saved = {k:w.detach().clone() for k,w in m.named_parameters()}
    arrays, rows, comparisons, steps = {}, [], [], []
    for dataset, xx, signals in [('old',x,oldsig), ('next',xn,newsig)]:
        gs = backward(m,xx,signals)
        projected = {}
        for name, gg in gs.items():
            for metric in ('SGD','Adam_fixed'):
                tangent = gg if metric=='SGD' else {k:g/denominator[k] for k,g in gg.items()}
                vals = project(m,xp,raw,tangent)
                projected[name,metric] = vals
                arrays[f'{dataset}_{name}_{metric}'] = vals
                for layer in range(6):
                    valid = raw[layer]['valid'].cpu().numpy()
                    for j,moment in enumerate(('mu','sd','r')):
                        rows.append(dict(dataset=dataset,signal=name,metric=metric,layer=layer+1,moment=moment,
                                         **M.stats(vals[layer,j],valid)))
            if dataset=='next' and name in ['base','alpha_half','alpha_zero','norm_unload','perm0','perm1','perm2','perm3']:
                delta = adam_delta(gg,state,pg)
                pred = project(m,xp,raw,delta)
                with torch.no_grad():
                    for k,w in m.named_parameters():
                        w.copy_(saved[k]+delta[k])
                after = M.probe_fields(m,xp)
                for i,r in enumerate(after):
                    actual = (r['r']-raw[i]['r']).cpu().numpy()
                    valid = raw[i]['valid'].cpu().numpy() & r['valid'].cpu().numpy()
                    arrays[f'finite_{name}_L{i+1}'] = actual
                    steps.append(dict(variant=name,layer=i+1,delta_r_mean=float(actual[valid].mean()),
                                      delta_r_median=float(np.median(actual[valid])),decreased_fraction=float((actual[valid]<0).mean()),
                                      jvp_relative_l2=float(np.linalg.norm(actual-pred[i,2])/max(np.linalg.norm(actual),1e-12))))
                with torch.no_grad():
                    for k,w in m.named_parameters():
                        w.copy_(saved[k])
                if seed==0 and task==10 and name=='base':
                    oracle = torch.optim.Adam(m.parameters(),lr=pg['lr'],foreach=False)
                    oracle.load_state_dict(copy.deepcopy(opt))
                    for k,w in m.named_parameters():
                        w.grad=gg[k].clone()
                    oracle.step()
                    error=max(float((w.detach()-saved[k]-delta[k]).abs().max()) for k,w in m.named_parameters())
                    scale=max(float(v.abs().max()) for v in delta.values())
                    checks['adam_formula_absolute_error']=error
                    assert error <= 1e-7+2e-3*scale, (error,scale)
                    with torch.no_grad():
                        for k,w in m.named_parameters():
                            w.copy_(saved[k])
                    m.zero_grad(set_to_none=True)
                    del oracle
                del delta, pred
        if dataset=='old':
            for metric in ('SGD','Adam_fixed'):
                for actual, parts in [('actual_fit',('ls_fit','fit_gap','ce')),('actual_switch',('ls_switch','switch_gap','ce'))]:
                    aa = torch.from_numpy(projected[actual,metric])
                    total = torch.from_numpy(sum(projected[n,metric] for n in parts))
                    # Normalize cancellation errors by the component scale, not the tiny residual CE fit.
                    scale=max(float(np.abs(projected[n,metric]).max()) for n in parts)
                    err=float((aa-total).abs().max())/max(scale,1e-12)
                    checks[f'{actual}_{metric}_ledger']=err
                    assert err<2e-3
        targets = ('actual_fit','actual_switch') if dataset=='old' else ('base',)
        for metric in ('SGD','Adam_fixed'):
            for actual in targets:
                for name in signals:
                    if name==actual: continue
                    for i in range(6):
                        comparisons.append(dict(dataset=dataset,actual=actual,prediction=name,metric=metric,layer=i+1,
                            **compare(projected[name,metric][i,2],projected[actual,metric][i,2],raw[i]['valid'].cpu().numpy())))
        del gs, projected
    # Reproduce the previous next-input baseline (same body, original head).
    oldraw = np.load(Path('/home/issan/Projects/obsidian-research-data/vit_phases_1025/adam_next_raw')/f'SL_seed{seed}_t{task}.npz')
    for i in range(6):
        a=torch.from_numpy(arrays['next_base_Adam_fixed'][i,2])
        b=torch.from_numpy(oldraw[f'L{i+1}_switch_full_r'])
        checked(checks,f'previous_next_L{i+1}',a,b)
    np.savez_compressed(rawdir/f'diagnostic_seed{seed}_t{task}.npz',**arrays)
    return rows, comparisons, steps, info, heads, checks


def rollout(m, base, opt, x, y, xp, classes, seed, task, rawdir):
    rows, arrays = [], {}
    generator = torch.Generator().manual_seed(1025125+seed*100+task)
    batches=[]
    while len(batches)<50:
        order=torch.randperm(len(x),generator=generator)
        batches.extend(order.split(128))
    newy=torch.randint(10,(len(y),),generator=torch.Generator().manual_seed(1025225+seed*100+task)).to(y.device)
    for arm, yy in [('continue',y),('relabel',newy)]:
        m.load_state_dict(base)
        optim=torch.optim.Adam(m.parameters(),lr=1e-4,foreach=False)
        optim.load_state_dict(copy.deepcopy(opt))
        initial=M.probe_fields(m,xp)
        for step in range(51):
            if step in (0,1,5,20,50):
                raw=M.probe_fields(m,xp)
                with torch.no_grad():
                    h=M.features(m,x)
                    logits=h@m.layers['out'].weight[classes].T+m.layers['out'].bias[classes]
                    ce=float(F.cross_entropy(logits,yy))
                for i,r in enumerate(raw):
                    dr=(r['r']-initial[i]['r']).cpu().numpy()
                    valid=(r['valid'] & initial[i]['valid']).cpu().numpy()
                    arrays[f'{arm}_step{step}_L{i+1}_r']=r['r'].cpu().numpy()
                    arrays[f'{arm}_step{step}_L{i+1}_delta']=dr
                    rows.append(dict(arm=arm,step=step,layer=i+1,ce=ce,delta_r_mean=float(dr[valid].mean()),
                                     delta_r_median=float(np.median(dr[valid])),decreased_fraction=float((dr[valid]<0).mean())))
            if step==50: break
            batch=batches[step].to(x.device)
            optim.zero_grad(set_to_none=True)
            for ids in batch.split(16):
                logits=m(x[ids])[:,classes]
                loss=F.cross_entropy(logits,yy[ids],reduction='sum')/len(batch)
                loss.backward()
            optim.step()
        del optim
    np.savez_compressed(rawdir/f'rollout_seed{seed}_t{task}.npz',**arrays)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--raw-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['diagnostic','rollout'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--tasks',type=int,nargs='+',default=[10,60])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.raw_output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    assert Path(C.__file__).resolve().is_relative_to(args.source_root.resolve())
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.45)
    source=C.RealImages()
    begin=time.monotonic()
    accum={k:[] for k in ('pressures','comparisons','finite_steps','states','heads','checks','rollouts','provenance')}
    for seed in args.seeds:
        archive=args.archive_root/f'vit_long_1024_{"vn" if seed<3 else "il"}_raw_seed{seed}.tar'
        xp=C.eval_tensor(source.val(C.probe_names(source,seed)),'cuda')
        with tarfile.open(archive,'r:') as tar:
            def read(name): return tar.extractfile(f'vit_long_1024/runs/SL/seed{seed}/{name}').read()
            provenance=json.loads(read('provenance.json'))
            for name,digest in provenance['source_hashes'].items():
                assert M.sha((args.source_root/name).read_bytes())==digest
            manifest={r['path']:r for r in json.loads(read('manifest.json'))}
            m=make_model('R',seed,'cuda',provenance['config']['model_config']).eval()
            for task in args.tasks:
                ident=dict(seed=seed,task=task)
                loaded={}
                for kind in ('model','adam'):
                    name=f'ckpt/{kind}_t{task:02d}.pt'
                    payload=read(name)
                    digest=M.sha(payload)
                    assert digest==manifest[name]['sha256']
                    loaded[kind]=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
                    accum['provenance'].append(dict(**ident,member=name,sha256=digest,archive=str(archive)))
                m.load_state_dict(loaded['model'])
                data,_=C.task_dataset(source,seed,task,10,25,augment=False)
                x,y=C.eval_tensor(data.images,'cuda'),data.labels.to('cuda')
                classes=C.loss_classes('SL',seed,task,10,25)
                if args.mode=='diagnostic':
                    nxt,_=C.task_dataset(source,seed,task+1,10,25,augment=False)
                    xn=C.eval_tensor(nxt.images,'cuda')
                    result=diagnostic(m,loaded['adam']['optimizer'],x,y,xn,xp,classes,seed,task,args.raw_output)
                    for name,value in zip(('pressures','comparisons','finite_steps','states','heads','checks'),result):
                        accum[name].extend(dict(**ident,**v) for v in (value if isinstance(value,list) else [value]))
                    reference=pd.read_csv(ROOT/'results/vit_phases_1025/states.csv')
                    rr=reference[(reference.arm=='SL')&(reference.seed==seed)&(reference.task==task)&(reference.stage=='end')].iloc[0]
                    assert abs(result[3]['ce']-rr.ce)<2e-6 and abs(result[3]['accuracy']-rr.accuracy)<1e-6
                    del xn
                else:
                    rr=rollout(m,loaded['model'],loaded['adam']['optimizer'],x,y,xp,classes,seed,task,args.raw_output)
                    accum['rollouts'].extend(dict(**ident,**r) for r in rr)
                del loaded,x,y
                for name,rr in accum.items():
                    if rr:
                        if name in ('checks','provenance'):
                            M.dump(args.output/f'{args.mode}_{name}.json',rr)
                        else: M.csv_write(args.output/f'{name}.csv',rr)
                print(json.dumps(dict(mode=args.mode,**ident,seconds=round(time.monotonic()-begin,1))),flush=True)
            del m
            torch.cuda.empty_cache()
    M.dump(args.output/f'{args.mode}_run.json',dict(status='complete',states=len(args.seeds)*len(args.tasks),
        seconds=time.monotonic()-begin,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))


if __name__=='__main__': main()
