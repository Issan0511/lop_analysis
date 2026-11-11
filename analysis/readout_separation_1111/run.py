"""Registered 280-trajectory readout separation. No work occurs at import."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import argparse
import copy
import functools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from analysis.loading_bridge_1111 import dynamic as D
from analysis.p_residual_loading_1111 import run as R
from src import stage1_ctrl_1111 as M
from . import ops

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT/'specs/spec_readout_separation_1111.md'
OLD = Path('/home/issan/Projects/obsidian-research-data/p_residual_loading_1111/results/p_residual_loading_1111/raw')
RELEASE = 200


def finite_tree(value, path='root'):
    if isinstance(value,dict):
        for k,v in value.items():finite_tree(v,path+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for k,v in enumerate(value):finite_tree(v,path+'.'+str(k))
    elif isinstance(value,(float,np.floating)) and not np.isfinite(value):
        raise FloatingPointError(path)


def file_name(model, activation, seed, t, arm):
    return f'{model}_{activation}_s{seed}_t{t}_{arm["id"]}.json'


def evaluation_points(limit, every):
    return sorted({0,1,5,25,50,100,199,200,201,limit,*range(0,limit+1,every)} & set(range(limit+1)))


def policy(arm, step, for_update=False):
    # Updates1..200 use the intervention; primary evaluation200 is post-release.
    active = step <= RELEASE if for_update else step < RELEASE
    if active:
        return arm['cf'],arm['body_gain'],arm['jacobian']
    return 1.,1.,'native'


@torch.no_grad()
def measure(box, step, arm, bank, units=False, pre_release=False):
    cf,gain,jac = (arm['cf'],arm['body_gain'],arm['jacobian']) if pre_release else policy(arm,step)
    h = box.features()
    W,b,rows = box.head();w,bb=W[rows],b[rows]
    # head_stats takes a conventional linear map; this gives the specified
    # centered forward logits and the unscaled-W body pullback.
    if cf == 1.:
        out = D.head_stats(h,w,bb,box.old,box.new,back=gain)
    else:
        out = D.head_stats(h,cf*w.double(),cf*bb.double()+(1-cf)*bank['zbar0'].double(),box.old,box.new,back=gain/cf)
        out['head_centered_norm'] = float((w.double()-w.double().mean(0)).norm())
    if isinstance(box,D.MLP):
        if cf == 1.:
            out['acc'],out['ce'] = M.evaluate(box.P,box.ctx,box.new)
        else:
            z = h @ w.T + bb
            z = bank['zbar0'] + cf*(z-bank['zbar0'])
            out['acc'] = float((z.argmax(1)==box.new).float().mean())
            out['ce'] = float((z.logsumexp(1)-z.gather(1,box.new[:,None]).squeeze(1)).mean())
    out['feature_rms'] = float(h.double().square().mean().sqrt())
    out['cf'],out['body_gain'],out['jacobian'],out['step'] = cf,gain,jac,int(step)
    if not isinstance(box,D.MLP):
        norm=box.model.layers['norm'];gamma,beta=norm.weight,norm.bias
        out.update(final_gamma_norm=float(gamma.norm()),final_beta_norm=float(beta.norm()),
                   effective_Wgamma_norm=float((w.double()*gamma.double()).norm()),
                   effective_Wbeta_plus_b_norm=float((w.double()@beta.double()+bb.double()).norm()))
    if units:
        extra=box.measure('S',units=True)
        for key in ('layers','unit_arrays'):
            if key in extra:out[key]=extra[key]
    finite_tree(out)
    return out


def parameter_groups(box):
    head=list(box.head()[:2]);body=list(box.body())
    if isinstance(box,D.MLP):return dict(trunk=body,head=head)
    ln=box.model.layers['norm']
    return dict(trunk=[p for p in body if p is not ln.weight and p is not ln.bias],
                final_gamma=[ln.weight],final_beta=[ln.bias],head=head)


def norm(ps):
    return float(torch.stack([p.detach().double().square().sum() for p in ps]).sum().sqrt()) if ps else 0.


def gradient_record(box):
    groups=parameter_groups(box);saved={};stats={}
    for name,ps in groups.items():
        saved[name]=[(p,p.detach().clone(),None if p.grad is None else p.grad.detach().clone()) for p in ps]
        stats[name]=dict(gradient_norm=norm([p.grad for p in ps if p.grad is not None]),parameter_norm=norm(ps))
    return saved,stats


@torch.no_grad()
def finish_gradient_record(box,saved,stats):
    for name,items in saved.items():
        updates=[p-old for p,old,g in items];gs=[g for p,old,g in items if g is not None]
        un=norm(updates);gn=norm(gs)
        dot=sum(float(((p-old).double()*(-g.double())).sum()) for p,old,g in items if g is not None)
        ss=[box.opt.state.get(p,{}) for p,old,g in items]
        mm=[s['exp_avg'] for s in ss if 'exp_avg' in s];vv=[s['exp_avg_sq'] for s in ss if 'exp_avg_sq' in s]
        stats[name].update(update_norm=un,update_vs_negative_gradient_cos=dot/(un*gn) if un>0 and gn>0 else None,
                           adam_m_norm=norm(mm),adam_v_rms=(sum(float(v.double().sum()) for v in vv)/sum(v.numel() for v in vv))**.5 if vv else 0.)
    finite_tree(stats)
    return stats


def custom_callable(box,arm,bank,engine):
    fn=functools.partial(ops.forward,box,cf=arm['cf'],body_gain=arm['body_gain'],jacobian=arm['jacobian'],zbar0=bank['zbar0'])
    if engine=='compile' and not isinstance(box,D.MLP) and arm['family']!='physical':
        # Fixed registered controls and two batch shapes exceed Dynamo's small
        # default per-code-object specialization budget. No dynamic arm selection.
        import torch._dynamo.config as config
        if hasattr(config,'recompile_limit'):config.recompile_limit=64
        else:config.cache_size_limit=64
        fn=torch.compile(fn,fullgraph=True)
    return fn


def warm_engine(box,fn,engine):
    """Prime each registered batch shape without updates, then restore state/RNG.

    The former KC arm ran after KK, so its native compiled graph had already
    seen both 128 and 68 samples. Automatic dynamic-shape specialization can
    change float32 kernels even at the earlier shape. This same ordered warmup
    prevents the first arm from inheriting a different compiler history.
    """
    if isinstance(box,D.MLP) or engine!='compile':return None
    sp=R.snapshot(box);finger=R.fingerprints(box);shapes=[]
    for idx in box.order:
        size=len(idx)
        if size in shapes:continue
        shapes.append(size)
        box.opt.zero_grad(set_to_none=True)
        loss=D.loss_for(box,fn(box.X[idx]),box.new[idx])
        if not torch.isfinite(loss):raise FloatingPointError('Warmup loss')
        loss.backward()
    R.restore(box,sp);box.opt.zero_grad(set_to_none=True)
    assert R.fingerprints(box)==finger
    assert R.digest(R.snapshot(box)[2])==R.digest(sp[2])
    return dict(batch_sizes=shapes,optimizer_steps=0,state_and_rng_restored=True)


def trajectory(box,arm,bank,limit,engine,warmup_checks=None):
    points=evaluation_points(limit,box.eval_every)
    fn=custom_callable(box,arm,bank,engine)
    warmed=warm_engine(box,fn,engine)
    if warmup_checks is not None:warmup_checks.update(warmed or {})
    rec=[measure(box,0,arm,bank,True)];pre=None;tracks=[]
    for step,(x,y) in enumerate(box.batches(limit),1):
        logits=fn(x) if step<=RELEASE else box.forward(x)
        loss=D.loss_for(box,logits,y)
        if not torch.isfinite(loss):raise FloatingPointError(f'loss step {step}')
        box.opt.zero_grad(set_to_none=True);loss.backward()
        tracked=step in (1,25,199,200,201,limit)
        if tracked:saved,stats=gradient_record(box)
        box.opt.step()
        if tracked:
            tracks.append(dict(step=step,groups=finish_gradient_record(box,saved,stats)))
        if step==RELEASE:
            pre=measure(box,step,arm,bank,True,pre_release=True)
        if step in points:
            rec.append(measure(box,step,arm,bank,step in (25,200,limit)))
    assert [r['step'] for r in rec]==points
    return rec,pre,tracks


def legacy_check(box,arm,curve,model,act,t):
    if arm['family']!='physical' or arm['scale']!=1. or arm['body_memory']!='keep' or arm['direction'] not in ('S','P'):
        return None
    p=OLD/f'{model}_{act}_S_s{box.seed}_t{t}_{arm["direction"]}_tau0_KC.json'
    previous=json.loads(p.read_text())
    old={r['step']:r for r in previous['curve']}
    pairs=[(r,old[r['step']]) for r in curve if r['step'] in old]
    a=np.asarray([[r['acc'],r['ce']] for r,s in pairs]);b=np.asarray([[s['acc'],s['ce']] for r,s in pairs])
    exact=bool(np.array_equal(a,b));err=float(np.max(np.abs(a-b)))
    ok=exact if model=='mlp' else bool(np.allclose(a,b,rtol=2e-5,atol=2e-5))
    if not ok:raise AssertionError(('legacy KC mismatch',str(p),err))
    return dict(path=str(p),sha256=D.sha(p),points=len(pairs),exact=exact,maxabs=err,rtol=0. if model=='mlp' else 2e-5,atol=0. if model=='mlp' else 2e-5)


def assert_close(label,a,b,checks,rtol=2e-5,atol=2e-5):
    a,b=a.detach(),b.detach()
    err=float((a-b).abs().max()) if a.numel() else 0.
    checks[label]=dict(maxabs=err,rtol=rtol,atol=atol,passed=bool(torch.allclose(a,b,rtol=rtol,atol=atol)))
    if not checks[label]['passed']:raise AssertionError((label,err))


def assert_engine_close(label,a,b,checks,gradient=False):
    # Native eager vs native compiled differs at late ViT checkpoints even
    # without any intervention. Separate that roundoff from same-engine tests.
    a,b=a.detach().double(),b.detach().double();delta=a-b
    dn=float(delta.norm());bn=float(b.norm());err=float(delta.abs().max()) if a.numel() else 0.
    maxref=float(b.abs().max()) if b.numel() else 0.
    atol=2e-4 if gradient else 1e-3
    ok=err<=atol+3e-4*maxref and dn<=3e-4*bn+2e-6*a.numel()**.5
    checks[label]=dict(passed=ok,maxabs=err,relative_l2=dn/bn if bn>0 else None,
                       maxabs_atol=atol,maxabs_rtol=3e-4,l2_rtol=3e-4,l2_atol_per_element=2e-6)
    if not ok:raise AssertionError((label,checks[label]))


def smoke_controls(box,bank,engine):
    """Real-source operator validation without optimizer steps; restore afterward."""
    sp=R.snapshot(box);finger=R.fingerprints(box);checks={}
    x,y=next(box.batches(1));W,b,rows=box.head()
    compiled_native=box.forward(x)
    native=box.forward(x) if isinstance(box,D.MLP) else box.model(x)[:,rows]
    if not isinstance(box,D.MLP):assert_engine_close('native_compile_vs_eager',compiled_native,native,checks)
    h=ops.features(box,x)
    direct=(h@W.T+b)[:,rows] if isinstance(box,D.MLP) else torch.nn.functional.linear(h,W,b)[:,rows]
    assert_close('native_features_forward',direct,native,checks,rtol=0.,atol=0.)
    for jac in (('elu_unity',) if isinstance(box,D.MLP) else ('ln_remove_radial','ln_remove_radial_mean')):
        other=ops.forward(box,x,jacobian=jac)
        assert_close(jac+'_same_forward',other,native,checks,rtol=0.,atol=0.)
    # At a fixed state, changing only body gain preserves head gradients.
    for cf in (1.,.1):
        allgrads=[];outs=[]
        for gain in (1.,.1):
            box.opt.zero_grad(set_to_none=True)
            z=ops.forward(box,x,cf=cf,body_gain=gain,zbar0=bank['zbar0'],force_eager=True)
            D.loss_for(box,z,y).backward()
            allgrads.append([None if p.grad is None else p.grad.detach().clone() for p in box.parameters()]);outs.append(z.detach())
        assert_close(f'cf{cf}_bodygain_output',outs[0],outs[1],checks)
        for i,(p,g0,g1) in enumerate(zip(box.parameters(),*allgrads)):
            if g0 is None and g1 is None:continue
            target=g0 if p is W or p is b else .1*g0
            assert_close(f'cf{cf}_bodygain_gradient_{i}',g1,target,checks,rtol=2e-4,atol=2e-5)
    if engine=='compile' and not isinstance(box,D.MLP):
        for arm in ops.arm_plan('vit'):
            if arm['family']=='physical' or arm['scale']==.1 or arm['direction']=='P':continue
            fn=custom_callable(box,arm,bank,engine)
            grads=[];outs=[]
            for f in (functools.partial(ops.forward,box,cf=arm['cf'],body_gain=arm['body_gain'],jacobian=arm['jacobian'],zbar0=bank['zbar0']),fn):
                box.opt.zero_grad(set_to_none=True);z=f(x);D.loss_for(box,z,y).backward()
                outs.append(z.detach());grads.append([None if p.grad is None else p.grad.detach().clone() for p in box.parameters()])
            assert_engine_close(arm['id']+'_compiled_output',outs[1],outs[0],checks)
            for i,(a,bg) in enumerate(zip(grads[0],grads[1])):
                if a is None and bg is None:continue
                assert_engine_close(arm['id']+f'_compiled_gradient_{i}',bg,a,checks,gradient=True)
    assert_close('no_hook_leak',box.forward(x),compiled_native,checks,rtol=0.,atol=0.)
    R.restore(box,sp);box.opt.zero_grad(set_to_none=True)
    assert R.fingerprints(box)==finger
    return checks


def provenance(box,args,plan):
    helpers=[Path(D.__file__),Path(R.__file__),Path(M.__file__),Path(ops.A.__file__)]
    helpers+=[ROOT/'analysis/readout_separation_1111'/name for name in ('ops.py','run.py','probe.py','launch.py')]
    helpers+= [ROOT/'analysis/vit_long_1024/posthoc_2blk/tangent_rank.py']
    hashes={str(p.relative_to(ROOT)):D.sha(p) for p in helpers}
    if args.model=='vit':hashes.update(box.C.source_hashes())
    sources=list(getattr(box,'sources',[getattr(box,'source',None)]))
    source_hashes={str(p):D.sha(p) for p in sources if p is not None}
    old=json.loads((ROOT/'results/p_residual_loading_1111/provenance.json').read_text())['validation']['actual_source_hashes']
    for p,v in source_hashes.items():assert old[p]==v,('changed source',p)
    batches=box.orders[box.t+1] if isinstance(box,D.MLP) else box.order
    return dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                spec_sha256=D.sha(SPEC),code_hashes=hashes,sources=source_hashes,args=vars(args),
                torch=torch.__version__,numpy=np.__version__,batch_sha256=R.digest(batches),labels_sha256=R.digest(box.new),
                inputs_sha256=R.digest(box.X),replay_checks=getattr(box,'replay_checks',[]),
                expected_arms=[a['id'] for a in plan],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def save_state(path,state):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(state,temp);temp.replace(path)


def run_source(box,args,t):
    start=time.monotonic();box.reset(t)
    act='ELU' if args.model=='mlp' else 'SL';plan=ops.arm_plan(args.model)
    limit=args.limit if args.smoke else box.recovery_limit
    if limit<=0:limit=min(26,box.recovery_limit) if args.smoke else box.recovery_limit
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    prov=provenance(box,args,plan)
    prov['expected_files']=[file_name(args.model,act,box.seed,t,a) for a in plan]
    present=[]
    for name in prov['expected_files']:
        p=outdir/name
        if p.exists():
            d=json.loads(p.read_text());q=d['provenance']
            assert d['complete'] and d['limit']==limit and q['spec_sha256']==prov['spec_sha256']
            assert q['code_hashes']==prov['code_hashes'] and q['sources']==prov['sources']
            assert q['batch_sha256']==prov['batch_sha256'] and q['args']['engine']==args.engine
            present.append(name)
    native=R.snapshot(box);fork=R.fingerprints(box);bank=ops.make_head_bank(box,t)
    stem=f'source_{args.model}_{act}_s{box.seed}_t{t}'
    statepath=outdir/(stem+'.pt')
    metapath=outdir/(stem+'.json')
    if statepath.exists() or metapath.exists():
        assert statepath.exists() and metapath.exists(), 'Incomplete source-state artifact; preserve and inspect'
        metadata=json.loads(metapath.read_text())
        assert metadata['state_sha256']==D.sha(statepath)
        assert metadata['metadata']['native_fingerprints']==fork
        for key in ('spec_sha256','code_hashes','sources','batch_sha256','labels_sha256','inputs_sha256'):
            assert metadata['provenance'][key]==prov[key], ('Changed source-state provenance',key)
    else:
        save_state(statepath,dict(snapshot=native,model=args.model,seed=box.seed,t_end=t,provenance=prov,fingerprints=fork))
        metadata=dict(metadata=bank['metadata'],mu0=bank['mu0'].cpu().tolist(),zbar0=bank['zbar0'].cpu().tolist(),
                      rows=bank['rows'].cpu().tolist(),state_file=str(statepath),state_sha256=D.sha(statepath),provenance=prov)
        D.save(metapath,metadata)
    prov['source_state_file']=str(statepath);prov['source_state_sha256']=metadata['state_sha256']
    prov['source_metadata_file']=str(metapath);prov['source_metadata_sha256']=D.sha(metapath)
    native_warmup=warm_engine(box,box.forward,args.engine)
    prov['native_engine_warmup']=native_warmup
    smoke_checks=smoke_controls(box,bank,args.engine) if args.smoke else None
    if args.smoke:D.save(outdir/(stem+'_smoke_checks.json'),smoke_checks)
    if args.with_probe:
        from .probe import assay
        probe_dir=Path(args.probe_out) if args.probe_out else ROOT/'results/readout_separation_1111'/('smoke_probe' if args.smoke else 'probes')
        before_probe=R.fingerprints(box)
        result=assay(box,t,probe_dir,bank=bank)
        assert result['complete'] and R.fingerprints(box)==before_probe==fork
        prov['probe_file']=str(probe_dir/f'probe_{args.model}_{act}_s{box.seed}_t{t}.json')
        prov['probe_sha256']=D.sha(prov['probe_file'])
        print('PROBE',args.model,box.seed,len(result['records']),f"{result['elapsed_s']:.2f}s",flush=True)
    for name in present:
        existing=json.loads((outdir/name).read_text())['provenance']
        for key in ('source_state_sha256','source_metadata_sha256'):
            assert existing[key]==prov[key], ('Changed saved source artifact',key)
        if args.with_probe:
            assert existing.get('probe_sha256')==prov['probe_sha256'], 'Changed/missing linked probe'
    for arm in plan:
        name=file_name(args.model,act,box.seed,t,arm)
        if name in present:continue
        R.restore(box,native);assert R.fingerprints(box)==fork
        ops.apply_physical(box,bank['entries'][arm['bank']],arm['body_memory']=='clear')
        post=R.fingerprints(box)
        assert post['body']==fork['body']
        if arm['body_memory']=='keep':assert post['body_adam']==fork['body_adam']
        assert all(not box.opt.state.get(p) for p in box.head()[:2])
        if arm['body_memory']=='clear':assert all(not box.opt.state.get(p) for p in box.body())
        st=time.monotonic()
        warmup={}
        curve,pre,tracks=trajectory(box,arm,bank,limit,args.engine,warmup_checks=warmup)
        check=legacy_check(box,arm,curve,args.model,act,t)
        result=dict(model=args.model,activation=act,seed=box.seed,t_end=t,arm=arm,limit=limit,curve=curve,pre_release=pre,
                    update_diagnostics=tracks,fork_hashes=fork,post_hashes=post,checks=dict(legacy_KC=check,geometry=bank['metadata']['checks'],engine_warmup=warmup),
                    provenance=prov,complete=True,elapsed_s=time.monotonic()-st,file=name,smoke=bool(args.smoke))
        finite_tree(result);D.save(outdir/name,result)
        print('DONE',name,f'{result["elapsed_s"]:.2f}s',flush=True)
    D.save(outdir/(stem+'_complete.json'),dict(complete=True,expected_files=prov['expected_files'],elapsed_s=time.monotonic()-start,provenance=prov))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['mlp','vit'],default='mlp')
    ap.add_argument('--seeds');ap.add_argument('--engine',choices=['eager','compile'],default='compile')
    ap.add_argument('--out');ap.add_argument('--smoke',action='store_true');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--probe-out',help='Explicit smoke probe subdirectory; main uses the registered probes directory')
    ap.add_argument('--with-probe',action='store_true',help='Also run the full frozen assay in smoke; always enabled for main')
    args=ap.parse_args()
    args.with_probe=args.with_probe or not args.smoke
    if not args.smoke and args.probe_out:raise ValueError('--probe-out is smoke-only')
    if not args.smoke and args.limit:raise ValueError('--limit is restricted to --smoke')
    if args.smoke and not 0<=args.limit<=26:raise ValueError('Registered real smoke is at most 26 updates')
    seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else ([0] if args.smoke and args.model=='mlp' else [10252010] if args.smoke else list(range(10)) if args.model=='mlp' else list(range(10252010,10252015)))
    allowed={0} if args.smoke and args.model=='mlp' else {10252010} if args.smoke else set(range(10)) if args.model=='mlp' else set(range(10252010,10252015))
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<=allowed:raise ValueError('Unregistered seeds')
    args.out=args.out or str(ROOT/'results/readout_separation_1111'/('smoke_'+args.model if args.smoke else 'raw'))
    for seed in seeds:
        st=time.monotonic()
        box=D.MLP(seed,'ELU',[10]) if args.model=='mlp' else D.ViT(seed,args)
        print('SETUP',args.model,seed,f'{time.monotonic()-st:.2f}s',flush=True)
        run_source(box,args,10 if args.model=='mlp' else 100)
        del box
        if args.model=='vit':torch.cuda.empty_cache()

if __name__=='__main__':main()
