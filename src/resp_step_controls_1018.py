#!/usr/bin/env python3
"""Native response-field restoration with preregistered gradient/Adam-step LR controls.

No production protocol shortening switch is exposed. RM uses its registered CPU
expm1 ELU and / bias correction; RC and F51 use their original stacked CUDA
engines and multiplication by reciprocal bias correction. Logs are read-only
pre-update autograd probes, tested against the actual native update.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import resp_ee_0917 as RM
from src import cifar_interventions_0920 as RC
from src import resp_cifar5p1_1007 as F51
from src import resp_cifar_ee_0920 as RESP_RC

RUN = 'resp_step_controls_1018'
PREREG_COMMIT = '34bac645'
REMOTE_PREREG_COMMIT = 'fe33a41389c25cf24aad6e8f768fc18093686bcc'
SPEC = ROOT / 'specs/spec_resp_step_controls_1018.md'
COMMON = ROOT / 'specs/spec_six_runs_common_1018.md'
PARAMS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
BOXES = ('RC', 'RM', 'F51')
K = 256
CAL_SEEDS = list(range(100, 110))
EVAL_SEEDS = list(range(10))
ARMS = ('N', 'R', 'LR_G', 'LR_U')
DATA_ROOT = Path('/home/issan/Projects/claude/proj_004_drift/data')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def put(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(p)


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def source_hashes():
    files = {SPEC, COMMON, Path(__file__)}
    files.update((ROOT / f'analysis/{RUN}').glob('*.py'))
    for mod in list(sys.modules.values()):
        file = getattr(mod, '__file__', None)
        if file and str(file).endswith('.py'):
            p = Path(file).resolve()
            if p.is_relative_to(ROOT) and p.is_file(): files.add(p)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def identity(box, phase):
    return dict(run=RUN, box=box, phase=phase, prereg_commit=PREREG_COMMIT,
                remote_prereg_commit=REMOTE_PREREG_COMMIT, git_hash=git('rev-parse', 'HEAD'), source_sha256=source_hashes(),
                torch=torch.__version__, python=sys.version, platform=platform.platform(),
                cpu_capability=torch.backends.cpu.get_cpu_capability(),
                threads=torch.get_num_threads(), flush_denormal=RM._flush_is_on(),
                base_lr=1e-4 if box == 'F51' else 1e-3, K=K,
                steps_per_task={'RC':30000, 'RM':6000, 'F51':780}[box],
                seeds=CAL_SEEDS if phase == 'calibrate' else EVAL_SEEDS,
                prediction='NOT_LR_ALL_THREE (paper-derived; no numerical probability supplied)')


def require_g0(path):
    if not path or not Path(path).is_file(): raise RuntimeError('BLOCKED_G0: integration audit missing')
    a = json.loads(Path(path).read_text())
    flags = ('all_pass', 'main_ancestor_verified', 'host_reproduction_pass', 'mutation_checks_pass')
    if a.get('schema_version') != 1 or not all(a.get(k) is True for k in flags):
        raise RuntimeError('BLOCKED_G0: integration audit incomplete or failed')
    if a.get('independent_auditor') != 'Codex /root (independent of six implementers)':
        raise RuntimeError('BLOCKED_G0: independent audit identity missing')
    files = ('src/cifar5p1_mlp_0920.py', 'src/resp_cifar5p1_1007.py')
    for file in files:
        if a.get('source_sha256', {}).get(file) != sha(ROOT/file):
            raise RuntimeError(f'BLOCKED_G0: audit source mismatch {file}')
    if subprocess.run(['git','merge-base','--is-ancestor',a['audited_commit'],'HEAD'],cwd=ROOT).returncode:
        raise RuntimeError('BLOCKED_G0: audited commit is not ancestor')
    return sha(path)


def admission(box, checks_path, audit):
    if box != 'RM' and not torch.cuda.is_available(): raise RuntimeError('BLOCKED_CUDA: registered CUDA device unavailable')
    if box == 'F51': require_g0(audit)
    chk = json.loads(Path(checks_path).read_text())
    if chk.get('source_sha256') != source_hashes(): raise RuntimeError('CHECK_FAILED: stale check source hashes')
    key = 'cpu_pass' if box == 'RM' else 'cuda_pass'
    if chk.get(key) is not True: raise RuntimeError(f'CHECK_FAILED: {key} required')
    dirty = git('status','--porcelain','--','src',f'analysis/{RUN}','specs')
    if dirty: raise RuntimeError('CHECK_FAILED: commit implementation before production')


def native_update(P, grads, m, v, tc, rates, style, invs=None):
    """Exact operation order of each imported host, including CPU division."""
    with torch.no_grad():
        c1, c2 = 1 - .9**tc, 1 - .999**tc
        for i, (p, g, mi, vi) in enumerate(zip(P, grads, m, v)):
            mi.mul_(.9).add_(g, alpha=1-.9)
            vi.mul_(.999).addcmul_(g, g, value=1-.999)
            if style == 'RM': upd = rates[i//2] * (mi/c1) / ((vi/c2).sqrt() + 1e-8)
            else: upd = rates[i//2] * (mi*invs[0]) / ((vi*invs[1]).sqrt() + 1e-8)
            p.sub_(upd)


@torch.no_grad()
def probe_begin(P, grads, m, v, tc, rates, style, invs=None):
    """No mutation or RNG; independent candidate m/v/update before native Adam."""
    out=[]; c1,c2=1-.9**tc,1-.999**tc
    for i,(p,g,mi,vi) in enumerate(zip(P,grads,m,v)):
        mm=mi.clone().mul_(.9).add_(g,alpha=1-.9)
        vv=vi.clone().mul_(.999).addcmul_(g,g,value=1-.999)
        if style=='RM': mh,vh=mm/c1,vv/c2
        else: mh,vh=mm*invs[0],vv*invs[1]
        upd=rates[i//2]*mh/(vh.sqrt()+1e-8)
        out.append(dict(p=p.detach().clone(),g=g.detach().clone(),m=mm,v=vv,mhat=mh,vhat=vh,upd=upd))
    return out


@torch.no_grad()
def probe_finish(P, m, v, capture, seeds, arm, task, step, rates, baseline=None):
    """Return scalar traces per seed/parameter, with a rounding-derived actual-delta check."""
    rows=[]
    for i,(p,mi,vi,c) in enumerate(zip(P,m,v,capture)):
        assert torch.equal(mi,c['m']) and torch.equal(vi,c['v']), 'S-log: moment candidate mismatch'
        delta=c['p'].double()-p.double()
        eps=torch.finfo(p.dtype).eps/2; tiny=torch.finfo(p.dtype).tiny
        bound=eps*(c['p'].double().abs()+c['upd'].double().abs())+tiny
        assert bool(((delta-c['upd'].double()).abs()<=bound).all()), 'S-log: candidate/actual displacement outside rounding bound'
        stacked=(p.ndim==3 if i%2==0 else p.ndim==2)
        for slot,seed in enumerate(seeds):
            def take(x): return x[slot] if stacked else x
            def norm(x): return float(take(x).double().norm())
            pn=norm(c['p']); un=norm(c['upd']); physical=norm(delta)
            row=dict(seed=seed,arm=arm,task=task,step=step,param=PARAMS[i],layer=i//2+1,
                     grad_norm=norm(c['g']),m_norm=norm(mi),v_norm=norm(vi),
                     mhat_norm=norm(c['mhat']),vhat_norm=norm(c['vhat']),
                     adam_candidate_norm=un,actual_delta_norm=physical,
                     param_norm=pn,relative_update=None if pn==0 else physical/pn,candidate_relative_update=None if pn==0 else un/pn,
                     zero_param_norm=pn==0,lr=rates[i//2],tc=None)
            if baseline is not None:
                a,b=take(c['g']).double().flatten(),take(baseline[i]).double().flatten()
                den=float(a.norm()*b.norm())
                row['baseline_grad_norm']=float(b.norm())
                row['cos_grad_N']=None if den==0 else float(torch.dot(a,b))/den
                row['cos_grad_undefined']=den==0
            rows.append(row)
    return rows


def group_rows(rows):
    by={}
    for row in rows:by.setdefault((row['seed'],row['arm'],row['task'],row['step'],row['layer']),[]).append(row)
    result=[]
    for (seed,arm,task,step,layer), rr in sorted(by.items()):
        assert len(rr)==2 and {r['param'] for r in rr}==set(PARAMS[(layer-1)*2:layer*2]),'INCOMPLETE W+b group'
        norm=lambda key:math.sqrt(sum(r[key]**2 for r in rr))
        pn=norm('param_norm');un=norm('actual_delta_norm');gn=norm('grad_norm')
        row=dict(seed=seed,arm=arm,task=task,step=step,layer=layer,
                 G=gn,U=un,relative_U=None if pn==0 else un/pn,zero_param_norm=pn==0,
                 W_grad_norm=next(r['grad_norm'] for r in rr if r['param'].startswith('W')),
                 lr=rr[0]['lr'])
        if 'baseline_grad_norm' in rr[0]:
            bn=norm('baseline_grad_norm');dot=sum((r['cos_grad_N'] or 0)*r['grad_norm']*r['baseline_grad_norm'] for r in rr)
            row['cos_grad_N']=None if gn*bn==0 else dot/(gn*bn)
        result.append(row)
    return result


def q_from_rows(rows, seeds=CAL_SEEDS, k=K):
    """RMS of grouped W,b norms, per-seed R/N then median; no clipping."""
    expected={(s,a,j,p) for s in seeds for a in ('N','R') for j in range(1,k+1) for p in PARAMS}
    keys=[(int(r['seed']),r['arm'],int(r['step']),r['param']) for r in rows if r['arm'] in ('N','R')]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('INCOMPLETE calibration trace')
    by={key:r for key,r in zip(keys,[r for r in rows if r['arm'] in ('N','R')])}
    ratios={}; zeros=[]; per_seed=[]
    for metric,col in (('qG','grad_norm'),('qU','actual_delta_norm')):
        ratios[metric]=[]
        for l in range(3):
            rr=[]
            for s in seeds:
                vals={a: math.sqrt(sum(float(by[s,a,j,p][col])**2 for j in range(1,k+1)
                                       for p in PARAMS[2*l:2*l+2])/k) for a in ('N','R')}
                n,r=vals['N'],vals['R']
                if n==0 or r==0:
                    if n==r==0: q=1.;zeros.append(dict(metric=metric,layer=l+1,seed=s))
                    else: raise ValueError(f'CALIBRATION_UNDEFINED: {metric} layer{l+1} seed{s} one-sided zero')
                else:q=r/n
                if not math.isfinite(q):raise ValueError('CALIBRATION_UNDEFINED: nonfinite ratio')
                rr.append(q);per_seed.append(dict(metric=metric,layer=l+1,seed=s,N=n,R=r,ratio=q))
            ratios[metric].append(float(np.median(np.asarray(rr,dtype=np.float64))))
    return dict(**ratios,zero_over_zero=zeros,per_seed=per_seed,K=k,seeds=list(seeds),method='RMS W+b; per-seed R/N; median seeds')


def load_schedule(path,box,dataset_signature=None):
    obj=json.loads(Path(path).read_text())
    assert obj['run']==RUN and obj['box']==box and obj['seeds']==CAL_SEEDS and obj['K']==K, 'S-schedule: calibration identity'
    for metric in ('qG','qU'):
        assert len(obj[metric])==3 and all(math.isfinite(v) and v>0 for v in obj[metric])
    assert 'identity' in obj,'S-schedule: missing calibration identity'
    ci=obj['identity']
    assert ci['source_sha256']==source_hashes(), 'S-schedule: calibration source differs from evaluation'
    assert ci['phase']=='calibrate' and ci['box']==box and ci['seeds']==CAL_SEEDS and ci['K']==K, 'S-schedule: source cohort identity'
    assert 'dataset_signature' in ci,'S-schedule: dataset identity missing'
    if dataset_signature is not None:
        assert ci['dataset_signature']==dataset_signature,'S-schedule: changed dataset between calibration and evaluation'
    parent=Path(path).parent
    for rel,h in obj['trace_sha256'].items(): assert sha(parent/rel)==h, 'S-schedule: trace hash mismatch'
    rows=[]
    for rel in obj['trace_sha256']:
        rows.extend(json.loads((parent/rel).read_text()))
    check=q_from_rows(rows)
    assert obj['qG']==check['qG'] and obj['qU']==check['qU'], 'S-schedule: multipliers differ from locked calibration traces'
    return obj


class RCEngine(RC.Engine):
    def __init__(self,*args,rates=(1e-3,)*3,**kwargs): self.rates=tuple(rates);super().__init__(*args,**kwargs)
    def step(self):
        xb,yb=self.X[self.ar,self.ids],self.Y[self.ar,self.ids]
        vals=self.forward(xb,self.ids);z3=vals[-1]
        lossv=F.cross_entropy(z3.reshape(-1,RC.E.N_CLASSES),yb.reshape(-1),reduction='none').view(self.R,RC.E.BATCH).mean(1)
        hit=(z3.detach().argmax(-1)==yb).float().mean(1)
        self.acc_sum.add_(hit);self.last_hit.copy_(hit)
        grads=torch.autograd.grad(lossv.sum(),self.P)
        with torch.no_grad():
            self.ce_sum.add_(lossv.detach());bad=~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step<0)&bad,self.step_t,self.bad_step));self.step_t.add_(1)
        native_update(self.P,grads,self.m,self.v,self.tc,self.rates,'RC',(self.inv1,self.inv2))
        self.last_values=tuple(q.detach() for q in vals)


class F51Engine(F51.Engine):
    def __init__(self,*args,rates=(1e-4,)*3,**kwargs): self.rates=tuple(rates);super().__init__(*args,**kwargs)
    def step(self):
        xb,yb=self.data.X[self.static_idx],self.data.Y[self.static_idx]
        z3=self.forward(xb,self.static_idx)[-1]
        lossv=F.cross_entropy(z3.reshape(-1,F51.NCLS),yb.reshape(-1),reduction='none').view(self.R,F51.BATCH).mean(1)
        hit=(z3.detach().argmax(-1)==yb).float().mean(1)
        self.acc_sum.add_(hit);self.last_hit.copy_(hit)
        grads=torch.autograd.grad(lossv.sum(),self.P)
        with torch.no_grad():
            self.ce_sum.add_(lossv.detach());bad=~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step<0)&bad,self.step_t,self.bad_step));self.step_t.add_(1)
        native_update(self.P,grads,self.m,self.v,self.tc,self.rates,'F51',(self.inv_c1,self.inv_c2))


def rm_train(params,adam,x,y,g_batch,epochs,fwd=None,rates=(1e-3,)*3,trace=None,seed=0,arm='N',task=11,cache=None):
    """Registered CPU loop, exact order/sample stream, optional read-only logs first K."""
    spt=RM.SPE*epochs;ce=torch.empty(spt);acc=torch.empty(spt);stp=torch.empty(spt,3);epsf={}
    m,v,tc=adam;chain=''
    for ep in range(epochs):
        order=torch.randperm(RM.N_IMG,generator=g_batch).to(x.device)
        chain=hashlib.sha256(chain.encode()+order.cpu().numpy().tobytes()).hexdigest()
        xs,ys=x[order],y[order]
        for j in range(RM.SPE):
            step=ep*RM.SPE+j+1;ob=order[j*RM.BATCH:(j+1)*RM.BATCH]
            xb,yb=xs[j*RM.BATCH:(j+1)*RM.BATCH],ys[j*RM.BATCH:(j+1)*RM.BATCH]
            vals=RM.forward(params,xb,ob,fwd);loss=F.cross_entropy(vals[-1],yb)
            if not bool(torch.isfinite(loss)):raise RuntimeError(f'DIVERGED: RM seed{seed} arm{arm} task{task} step{step}')
            ce[step-1]=loss.detach();acc[step-1]=(vals[-1].detach().argmax(1)==yb).float().mean()
            grads=torch.autograd.grad(loss,params)
            record=trace is not None and step<=K
            cap=probe_begin(params,grads,m,v,tc[0]+1,rates,'RM') if record else None
            tc[0]+=1
            # Store the exact native candidate for inherited step diagnostics (RM uses RMS, not norm).
            with torch.no_grad():
                c1,c2=1-.9**tc[0],1-.999**tc[0]
                for i,(p,g,mi,vi) in enumerate(zip(params,grads,m,v)):
                    mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999)
                    upd=rates[i//2]*(mi/c1)/((vi/c2).sqrt()+1e-8);p-=upd
                    if i%2==0:stp[step-1,i//2]=upd.square().mean().sqrt()
                if step in RM.eps_points(spt):epsf[step]=[float(((v[i]/c2).sqrt()<1e-8).double().mean()) for i in RM.WEIGHTS]
            if record:
                baseline=None
                if cache is not None:
                    path=Path(cache)/f'g{step:04d}.pt'
                    if arm=='N':RC.save_pt(path,[c['g'] for c in cap])
                    else:baseline=torch.load(path,map_location='cpu',weights_only=False)
                logged=probe_finish(params,m,v,cap,[seed],arm,task,step,rates,baseline)
                for row in logged:row['tc']=tc[0]
                trace.extend(logged)
    return ce,acc,stp,epsf,chain


def rm_prefix(seed,mnist,out,ident):
    ckpath=out/'prefix.pt'
    if ckpath.exists():
        assert sha(ckpath)==json.loads((out/'prefix_digest.json').read_text())['sha256'],'prefix checkpoint checksum'
        z=torch.load(ckpath,map_location='cpu',weights_only=False);assert z['identity']==ident
        return z['cks'],mnist.train_x[RM.RL.subset_idx(seed)],z['rows']
    rows,units,cks,x,info=RM.run_prefix(seed,mnist,epochs=80,n_tasks=12,save_at=(2,10,11,12),progress=True)
    RC.save_pt(ckpath,dict(identity=ident,cks=cks,rows=rows,info=info))
    put(out/'prefix_digest.json',dict(sha256=sha(ckpath)))
    put(out/'prefix_rows.json',rows)
    return cks,x,rows


def rm_seed(seed,mnist,out,ident,phase,schedule):
    out.mkdir(parents=True,exist_ok=True)
    cks,x,prefix=rm_prefix(seed,mnist,out,ident)
    allrows=[];traces=[]
    names=('N','R') if phase=='calibrate' else ARMS
    for arm in names:
        armout=out/arm;armout.mkdir(exist_ok=True);done=armout/'done.json'
        if done.exists():
            z=json.loads(done.read_text());assert z['identity']==ident
            for f,h in z['files'].items():assert sha(armout/f)==h
            allrows.extend(json.loads((armout/'rows.json').read_text()));traces.extend(json.loads((armout/'trace.json').read_text()));continue
        ck=cks[10];params=[p.clone().requires_grad_(True) for p in ck['params']]
        adam=([p.clone() for p in ck['m']],[p.clone() for p in ck['v']],[ck['tc']])
        init_hash=RM.SH.state_sha256(params,adam,RM.ACT)
        assert init_hash==ck['state_sha256'], 'branch clone state mismatch'
        gl,gb=RM.gen_from(ck['g_lab']),RM.gen_from(ck['g_batch'])
        sh=RM.build_shift(RM.ARMS['R2_10'],cks) if arm=='R' else None;fwd=RM.make_forward(sh)
        with torch.no_grad():assert torch.equal(RM.forward(params,x,torch.arange(1200),fwd)[-1],RM.forward(params,x,None,None)[-1]),'S-host/field: branch logit'
        mult=[1.]*3 if arm in ('N','R') else schedule['qG' if arm=='LR_G' else 'qU']
        rates=tuple(1e-3*q for q in mult);rows=[];trace=[];units={}
        for task in (11,12):
            y=RM.RL.task_labels(gl);labhash=RC.tree_hash(y)
            ce,acc,stp,epsf,chain=rm_train(params,adam,x,y,gb,80,fwd,rates,trace if task==11 else None,seed,arm,task,out/'gradient_cache')
            row=dict(seed=seed,arm=arm,task=task,**RM.task_row(ce,acc,stp,epsf,y,6000),
                     label_hash=labhash,order_hash=chain,initial_state_sha256=init_hash,
                     state_sha256=RM.SH.state_sha256(params,adam,RM.ACT),rates=list(rates))
            u,lg=RM.unit_point(params,x,fwd,sh)
            row['memo_acc']=float((lg.argmax(1)==y).float().mean());rows.append(row)
            for key,value in u.items():units[f't{task}_{key}']=value
            if arm=='N':assert row['state_sha256']==cks[task]['state_sha256'], 'S-host: N is not natural continuation'
            RC.save_pt(armout/f't{task}.pt',dict(params=params,m=adam[0],v=adam[1],tc=adam[2][0],g_lab=gl.get_state(),g_batch=gb.get_state(),shift=sh))
            if not row['finite']:raise RuntimeError(f'DIVERGED: RM seed{seed} arm{arm} task{task}')
        put(armout/'rows.json',rows);put(armout/'trace.json',trace);np.savez_compressed(armout/'units.npz',**units)
        put(done,dict(identity=ident,files={p.name:sha(p) for p in armout.iterdir() if p.name!='done.json'}))
        allrows.extend(rows);traces.extend(trace)
        print(f'RM seed={seed} arm={arm} complete',flush=True)
    return allrows,traces


def cuda_probe(eng,box,ids,arm,task,step,cache):
    """Same minibatch shape as captured native step; no accumulators or RNG touched."""
    if box=='RC':xb,yb=eng.X[eng.ar,ids],eng.Y[eng.ar,ids];ncls=10;invs=(eng.inv1,eng.inv2)
    else:xb,yb=eng.data.X[ids],eng.data.Y[ids];ncls=100;invs=(eng.inv_c1,eng.inv_c2)
    loss=F.cross_entropy(eng.forward(xb,ids)[-1].reshape(-1,ncls),yb.reshape(-1),reduction='none').view(eng.R,-1).mean(1).sum()
    if not bool(torch.isfinite(loss)):raise RuntimeError(f'DIVERGED: {box} arm{arm} task{task} step{step}')
    grads=torch.autograd.grad(loss,eng.P)
    invs[0].fill_(1/(1-.9**(eng.tc+1)));invs[1].fill_(1/(1-.999**(eng.tc+1)))
    capture=probe_begin(eng.P,grads,eng.m,eng.v,eng.tc+1,eng.rates,box,invs)
    path=Path(cache)/f'g{step:04d}.pt';base=None
    if arm=='N':RC.save_pt(path,[c['g'] for c in capture])
    else:base=[v.to(eng.device) for v in torch.load(path,map_location='cpu',weights_only=False)]
    return capture,base


def cuda_cohort(box,out,ident,phase,schedule):
    seeds=ident['seeds'];out.mkdir(parents=True,exist_ok=True)
    if box=='RC':
        dev=RC.setup();X,data=RC.make_inputs(seeds,dev);eng=RC.Engine(seeds,X)
        natural=RC.Engine;branch=10;donor=1;end=11
    else:
        dev=F51.setup();data=F51.Data(dev);eng=F51.Engine(seeds,data)
        natural=F51.Engine;branch=28;donor=2;end=29
    bind_manifest(out/'cuda_input_manifest.json',dict(data_sha256=data.sha256,
        input_hash=RC.tree_hash(X) if box=='RC' else RC.tree_hash(data.X),
        subset={str(s):RC.E.RC.subset_idx(s).tolist() for s in seeds} if box=='RC' else
        {str(s):F51.C.task_plan(s) for s in seeds}))
    states={};prefix=[]
    # Each completed task is an exact restart checkpoint. No partial task is reused.
    latest=out/'prefix'/'latest.pt'
    if latest.exists():
        assert sha(latest)==json.loads((out/'prefix'/'digest.json').read_text())['latest.pt'],'prefix checksum'
        z=torch.load(latest,map_location='cpu',weights_only=False);assert z['identity']==ident
        eng=natural(seeds,X if box=='RC' else data,state=z['state']);prefix=z['rows']
    for task in range(eng.task+1,end+1):
        if box=='RC':rows,units=eng.train_task(task,400)
        else:
            taskrows,batches=eng.task_batches(task);eng.train_steps(batches);eng.task=task
            rows=eng.host_rows(task,taskrows);units={}
        prefix.extend(rows)
        if task in (donor,branch,end):RC.save_pt(out/'prefix'/f't{task:02d}.pt',eng.state())
        RC.save_pt(latest,dict(identity=ident,state=eng.state(),rows=prefix));put(out/'prefix'/'rows.json',prefix)
        put(out/'prefix'/'digest.json',{p.name:sha(p) for p in (out/'prefix').glob('*.pt')})
        print(f'{box} prefix t{task}/{end}',flush=True)
    for name,h in json.loads((out/'prefix'/'digest.json').read_text()).items():assert sha(out/'prefix'/name)==h,'prefix state checksum'
    for t in (donor,branch,end):states[t]=torch.load(out/'prefix'/f't{t:02d}.pt',map_location='cpu',weights_only=False)
    del eng
    allrows=[];traces=[];names=('N','R') if phase=='calibrate' else ARMS
    for arm in names:
        dest=out/arm;dest.mkdir(exist_ok=True);marker=dest/'done.json'
        if marker.exists():
            z=json.loads(marker.read_text());assert z['identity']==ident
            for f,h in z['files'].items():assert sha(dest/f)==h
            allrows.extend(json.loads((dest/'rows.json').read_text()));traces.extend(json.loads((dest/'trace.json').read_text()));continue
        st=states[branch];mult=[1.]*3 if arm in ('N','R') else schedule['qG' if arm=='LR_G' else 'qU'];lr=ident['base_lr'];rates=[lr*q for q in mult]
        if box=='RC':
            shift=None if arm!='R' else RC.Shift([q.to(dev) for q in st['P']],2,RESP_RC.field_from_states(st,states[donor],2,X))
            eng=RCEngine(seeds,X,state=st,shift=shift,rates=rates)
            fullids=torch.arange(1200,device=dev)[None,:].expand(10,-1)
            with torch.no_grad():assert torch.equal(eng.forward(X,fullids)[-1],RC.E.forward(eng.P,X,eng.act)[-1]),'S-host/field branch logits'
        else:
            rowsids=data.task_rows(seeds,end);table,ids,d=F51.build_field(st,states[donor],2,rowsids,data)
            field=F51.Field([q.to(dev) for q in st['P']],2,table) if arm=='R' else None
            eng=F51Engine(seeds,data,state=st,field=field,rates=rates)
            with torch.no_grad():assert torch.equal(eng.forward(data.X[ids],ids)[-1],F51.B.forward(eng.P,data.X[ids],eng.act)[-1]),'S-host/field branch logits'
        trace=[];cache=out/'gradient_cache'
        if box=='RC':
            original=eng.one_update
            def logged(ids):
                step=int(eng.step_t)+1
                rec=cuda_probe(eng,box,ids,arm,end,step,cache) if step<=K else None
                original(ids)
                if rec is not None:
                    rr=probe_finish(eng.P,eng.m,eng.v,rec[0],seeds,arm,end,step,rates,rec[1])
                    for row in rr:row['tc']=eng.tc
                    trace.extend(rr)
            eng.one_update=logged
            rows,units=eng.train_task(end,400)
        else:
            taskrows,batches=eng.task_batches(end)
            eng.acc_sum.zero_();eng.ce_sum.zero_();eng.bad_step.fill_(-1);eng.step_t.zero_()
            for j in range(780):
                ids=batches[:,j];step=j+1
                rec=cuda_probe(eng,box,ids,arm,end,step,cache) if step<=K else None
                eng.static_idx.copy_(ids);eng.tc+=1;eng.inv_c1.fill_(1/(1-.9**eng.tc));eng.inv_c2.fill_(1/(1-.999**eng.tc));eng.cg.replay()
                if rec is not None:
                    rr=probe_finish(eng.P,eng.m,eng.v,rec[0],seeds,arm,end,step,rates,rec[1])
                    for row in rr:row['tc']=eng.tc
                    trace.extend(rr)
            eng.task=end;torch.cuda.synchronize();rows=eng.host_rows(end,taskrows)
            ids=torch.stack(taskrows).to(dev);_,units=eng.diagnostic(ids,data.Y[ids],780)
            for row,finite in zip(rows,eng.finite().tolist()):row['finite']=finite
        for slot,row in enumerate(rows):
            row['arm']=arm
            row['initial_state_sha256']=RC.tree_hash(st)
            row['rates']=rates
            if box=='F51':
                row['label_hash']=RC.tree_hash(data.Y[batches[slot]])
                row['order_hash']=RC.tree_hash(batches[slot])
        if arm=='N':
            if box=='RC':assert RC.tree_hash(RC.core_state(eng.state()))==RC.tree_hash(RC.core_state(states[end])),'S-host: natural continuation'
            else:assert RC.tree_hash(eng.state())==RC.tree_hash(states[end]),'S-host: natural continuation'
        if not all(r['finite'] for r in rows):raise RuntimeError(f'DIVERGED: {box} {arm}')
        put(dest/'rows.json',rows);put(dest/'trace.json',trace);RC.save_pt(dest/'end.pt',eng.state());np.savez_compressed(dest/'units.npz',**units)
        put(marker,dict(identity=ident,files={p.name:sha(p) for p in dest.iterdir() if p.name!='done.json'}))
        allrows.extend(rows);traces.extend(trace);del eng;torch.cuda.empty_cache()
    return allrows,traces


def bind_manifest(path,obj):
    """Existing data descriptions are evidence, never overwritten on resume."""
    if Path(path).exists():
        assert json.loads(Path(path).read_text())==json.loads(json.dumps(obj)), 'DATA_IDENTITY_MISMATCH: existing data manifest'
    else:put(path,obj)


def describe_data(box,seeds):
    mnist=None
    if box=='RM':
        RM.H.DATA_DIR=DATA_ROOT/'mnist';RM.H.setup('cpu');mnist=RM.H.Mnist(torch.device('cpu'))
        signature=dict(archive_sha256=mnist.sha256,train_tensor_sha256=RC.tree_hash(mnist.train_x),
                       transform='native float32 /255, flat 784',loader_source_sha256=sha(ROOT/'src/pmnist_0905.py'))
        subsets={str(s):RM.RL.subset_idx(s).tolist() for s in seeds}
    elif box=='RC':
        RC.E.RC.DATA_DIR=RC.DATA;data=RC.E.RC.Cifar10()
        signature=dict(archive_sha256=data.sha256,raw_train_sha256=RC.tree_hash(data.train_u8),
                       transform='native RC slot_inputs std',loader_source_sha256=sha(ROOT/'src/pmnist_rlcifar_0907.py'),
                       transform_source_sha256=sha(ROOT/'src/rlcifar_mlp_battle_0918.py'))
        subsets={str(s):RC.E.RC.subset_idx(s).tolist() for s in seeds}
    else:
        data=F51.C.Cifar100(F51.DATA)
        signature=dict(archive_sha256=data.sha256,raw_train_sha256=RC.tree_hash(data.train_u8),
                       train_labels_sha256=RC.tree_hash(data.train_y),transform='native F51 inputs train std',
                       loader_source_sha256=sha(ROOT/'src/cifar5p1_mlp_0920.py'))
        subsets={str(s):json.loads(json.dumps(F51.C.task_plan(s))) for s in seeds}
    manifest=dict(dataset_signature=signature,subsets_or_class_plans=subsets,
                  subset_or_plan_sha256=hashlib.sha256(json.dumps(subsets,sort_keys=True).encode()).hexdigest())
    return manifest,mnist


def worker_setup():
    torch.set_num_threads(1);torch.set_flush_denormal(True);RM.H.setup('cpu')


def rm_seed_job(seed,out,ident,phase,schedule):
    """Spawn target: each seed owns its data, streams, checkpoints and one CPU thread."""
    worker_setup()
    manifest,mnist=describe_data('RM',[seed])
    assert manifest['dataset_signature']==ident['dataset_signature'],'DATA_IDENTITY_MISMATCH: spawned worker data'
    rows,traces=rm_seed(seed,mnist,Path(out),ident,phase,schedule)
    return seed,rows,traces


def order_seed_shards(shards,seeds):
    got=[int(s[0]) for s in shards]
    assert len(got)==len(set(got)) and set(got)==set(seeds),'INCOMPLETE: duplicate or missing seed shard'
    by={s[0]:s for s in shards}
    return [by[s] for s in seeds]


def combine_rm_shards(shards,seeds,phase):
    rows=[];traces=[]
    arms=('N','R') if phase=='calibrate' else ARMS
    for seed,rr,tt in order_seed_shards(shards,seeds):
        rk=[(r['seed'],r['arm'],r['task']) for r in rr]
        expected={(seed,a,t) for a in arms for t in (11,12)}
        assert len(rk)==len(set(rk)) and set(rk)==expected,'INCOMPLETE: RM arm/task shard'
        tk=[(r['seed'],r['arm'],r['step'],r['param']) for r in tt]
        expected={(seed,a,k,p) for a in arms for k in range(1,K+1) for p in PARAMS}
        assert len(tk)==len(set(tk)) and set(tk)==expected,'INCOMPLETE: RM trace shard'
        rows.extend(rr);traces.extend(tt)
    return rows,traces


def worker_equivalence_fixture(seed):
    """CHECK_ONLY: one genuine native-shape epoch; not a production performance CLI."""
    worker_setup();g=torch.Generator().manual_seed(seed)
    x=torch.randn(1200,784,generator=g);y=RM.RL.task_labels(RM.H.stream('rl_labels',seed))
    p=RM.H.init_params(seed,torch.device('cpu'))
    adam=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0]);gb=RM.H.stream('rl_batch',seed)
    ce,acc,stp,epsf,chain=rm_train(p,adam,x,y,gb,1)
    return seed,RC.tree_hash((p,adam,gb.get_state(),ce,acc)),chain


def _execute(box,phase,out,checks,audit,schedule_path=None,workers=1):
    assert 1<=workers<=4,'workers must be 1..4; each worker uses one CPU thread'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    try:admission(box,checks,audit)
    except (RuntimeError,FileNotFoundError) as e:
        put(out/'status.json',dict(status=str(e).split(':')[0],reason=str(e)));raise
    torch.set_num_threads(1 if box=='RM' else 2);torch.set_flush_denormal(True)
    seeds=CAL_SEEDS if phase=='calibrate' else EVAL_SEEDS
    data_manifest,mnist=describe_data(box,seeds)
    if box=='RM':
        canonical=json.loads(Path(checks).read_text())['checks']['S-cost']['detail']['real_data_sha256']
        assert data_manifest['dataset_signature']['archive_sha256']==canonical,'DATA_IDENTITY_MISMATCH: current MNIST differs from verified check data'
    ident=identity(box,phase);ident['dataset_signature']=data_manifest['dataset_signature']
    ident['subset_or_plan_sha256']=data_manifest['subset_or_plan_sha256']
    ident['admission_checks_sha256']=sha(checks)
    if box=='F51':ident['integration_audit_sha256']=require_g0(audit)
    schedule=None
    if phase=='evaluate':
        if schedule_path is None:raise RuntimeError('CALIBRATION_REQUIRED: --schedule')
        schedule=load_schedule(schedule_path,box,ident['dataset_signature']);ident['schedule_sha256']=sha(schedule_path)
    bind_manifest(out/'data_manifest.json',data_manifest)
    saved=out/'provenance.json'
    if saved.exists():assert json.loads(saved.read_text())==ident,'resume identity mismatch'
    else:put(saved,ident)
    (out/'spec.md').write_text(SPEC.read_text());(out/'common.md').write_text(COMMON.read_text())
    t0=time.time();allrows=[];traces=[]
    if box=='RM':
        shards=[]
        if workers==1:
            for seed in ident['seeds']:
                rows,logs=rm_seed(seed,mnist,out/f's{seed}',ident,phase,schedule);shards.append((seed,rows,logs))
                put(out/'status.json',dict(status='RUNNING',completed_seed=seed,completed_seeds=[s[0] for s in shards],workers=workers,elapsed_s=time.time()-t0))
        else:
            import concurrent.futures
            import multiprocessing
            del mnist
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(rm_seed_job,seed,str(out/f's{seed}'),ident,phase,schedule) for seed in ident['seeds']]
                for future in concurrent.futures.as_completed(futures):
                    shard=future.result();shards.append(shard)
                    put(out/'status.json',dict(status='RUNNING',completed_seed=shard[0],completed_seeds=sorted(s[0] for s in shards),workers=workers,elapsed_s=time.time()-t0))
        allrows,traces=combine_rm_shards(shards,ident['seeds'],phase)
    else:
        with F51.gpu_lock():allrows,traces=cuda_cohort(box,out,ident,phase,schedule)
    put(out/'per_task.json',allrows);write_csv(out/'per_task.csv',allrows)
    put(out/'trace.json',traces);write_csv(out/'trace.csv',traces)
    write_csv(out/'group_trace.csv',group_rows(traces))
    if phase=='calibrate':
        q=q_from_rows(traces);put(out/'schedule.json',dict(run=RUN,box=box,**q,trace_sha256={'trace.json':sha(out/'trace.json')},identity=ident))
    put(out/'status.json',dict(status='COMPLETE',phase=phase,box=box,elapsed_s=time.time()-t0))
    return allrows


def semantic_status(error):
    status=str(error).split(':')[0]
    known=('BLOCKED_CUDA','BLOCKED_G0','CHECK_FAILED','CALIBRATION_REQUIRED','CALIBRATION_UNDEFINED','INCOMPLETE','DIVERGED','DATA_IDENTITY_MISMATCH')
    return status if status in known else 'CHECK_FAILED'


def execute(box,phase,out,checks,audit,schedule_path=None,workers=1):
    try:return _execute(box,phase,out,checks,audit,schedule_path,workers)
    except Exception as e:
        put(Path(out)/'status.json',dict(status=semantic_status(e),reason=str(e),phase=phase,box=box))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--box',choices=BOXES,required=True);p.add_argument('--phase',choices=('calibrate','evaluate'),required=True)
    p.add_argument('--out',required=True);p.add_argument('--schedule')
    p.add_argument('--checks',default=str(ROOT/f'results/{RUN}/checks.json'))
    p.add_argument('--integration-audit');p.add_argument('--workers',type=int,default=1)
    a=p.parse_args(argv)
    try:execute(a.box,a.phase,a.out,a.checks,a.integration_audit,a.schedule,a.workers)
    except (RuntimeError,FileNotFoundError,AssertionError,ValueError) as e:
        status=str(e).split(':')[0]
        if status not in ('BLOCKED_CUDA','BLOCKED_G0','CHECK_FAILED','CALIBRATION_REQUIRED','CALIBRATION_UNDEFINED','INCOMPLETE','DIVERGED','DATA_IDENTITY_MISMATCH'):status='CHECK_FAILED'
        put(Path(a.out)/'status.json',dict(status=status,reason=str(e)))
        print(str(e),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
