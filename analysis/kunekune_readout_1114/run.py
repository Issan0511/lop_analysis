"""Preregistered matched kunekune readout controls; native CPU arithmetic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from src import stage1_ctrl_1111 as M
from src.rlcifar_mlp_battle_0918 import SnakeFamily
from analysis.elu_specificity_1112 import mlp as B
from analysis.readout_separation_1111 import ops

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT/'specs/spec_kunekune_readout_1114.md'
ACTS = ('KKA','KKA23','KKT1','SNA','ELU','R')
KINDS = dict(KKA='kk',KKA23='kk23',KKT1='kkt1',SNA='snake')
ARMS = ('N','S1','S01','P1','P01','Z_keep','Z_clear','F0')
AGES = (1,10,30)
save_json, save_state, digest, sha = B.save_json, B.save_state, B.digest, B.sha

def provenance():
    paths = [Path(__file__),Path(M.__file__),Path(B.__file__),Path(ops.__file__),
             ROOT/'src/rlcifar_mlp_battle_0918.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    return dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                spec_sha256=sha(SPEC),code_hashes=hashes,code_sha256=digest(hashes),
                torch=torch.__version__,numpy=np.__version__,cpu_threads=torch.get_num_threads())

class Model(B.MLP):
    def __init__(self,params,activation):
        assert activation in ACTS
        self.activation,self.ln=activation,'off'
        self.P=M.new_params(params); self.opt=M.make_opt('adam',self.P,.001)
        self.family=None
        if activation in KINDS:
            self.family=SnakeFamily(activation,KINDS[activation],widths=(params[1].numel(),))
            self.family.init_state(1,'cpu',None)

    def phi(self,z):
        return M.activ(z,self.activation) if self.family is None else self.family.phi(z[None])[0]

    def features(self,x,auxiliary=False):
        z=x@self.P[0].T+self.P[1]; a=self.phi(z)
        return (a,z,a) if auxiliary else a

    @torch.no_grad()
    def update_ema(self,z):
        if self.family is not None:
            self.family.V[0].mul_(.99).add_(.01*z.detach().var(0,unbiased=False)[None])

    def snapshot(self):
        s=super().snapshot()
        s['ema']=None if self.family is None else self.family.state()
        return s

    def restore(self,state):
        super().restore(state)
        if self.family is not None:self.family.load_state(state['ema'])
        else:assert state['ema'] is None

    def fingerprints(self):
        f=super().fingerprints()
        f['activation_ema']=digest(None if self.family is None else self.family.state())
        return f

def one_step(model,x,Y,track=False):
    a,z,_=model.features(x,True)
    logits=a@model.P[2].T+model.P[3]
    loss=B.loss_for(logits,Y)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite training loss')
    model.opt.zero_grad(set_to_none=True);loss.backward()
    if track:
        saved,stats=B.gradient_record(model)
        old_w=model.P[0].detach().clone();old_b=model.P[1].detach().clone()
    model.opt.step();model.update_ema(z)
    if not track:return float(loss.detach())
    return float(loss.detach()),B.finish_record(saved,stats),model.P[0].detach()-old_w,model.P[1].detach()-old_b

def derivative(model,z):
    with torch.enable_grad():
        q=z.detach().requires_grad_(True);a=model.phi(q)
        d,=torch.autograd.grad(a,q,torch.ones_like(a))
    return d.detach()

@torch.no_grad()
def measure(model,x,old,new,step,diagnostics=False):
    h,z,_=model.features(x,True);logits=h@model.P[2].T+model.P[3]
    B.finite(logits,'logits')
    correct=int((logits.argmax(1)==new).sum())
    r=dict(step=int(step),acc=float((logits.argmax(1)==new).float().mean()),correct=correct,
           ce=float((logits.logsumexp(1)-logits.gather(1,new[:,None]).squeeze(1)).mean()))
    if diagnostics:
        zd=z.double();hd=h.double();mu=zd.mean(0);sd=zd.std(0,unbiased=False)
        g=derivative(model,z).double();v=model.P[2].double();vc=v-v.mean(0)
        lp=logits.double().log_softmax(1);p=lp.exp();D=p@vc-vc[new]
        kn=x.double()@x.double().mean(0)+1
        push=(kn[:,None]*g*(p@vc)).mean(0)
        d=dict(m_mean=float(mu.mean()),m_median=float(mu.median()),sd_median=float(sd.median()),
               r_median=float((mu/sd.clamp_min(1e-15)).median()),
               mean_abs_derivative=float(g.abs().mean()),small_derivative_fraction=float((g.abs()<.05).double().mean()),
               derivative_input_variance=float(g.var(0,unbiased=False).mean()),
               activity_centered_rms=float((hd-hd.mean(0)).square().mean().sqrt()),
               head_centered_norm=float(vc.norm()),head_norm=float(v.norm()),
               logit_class_centered_rms=float((logits.double()-logits.double().mean(1,keepdim=True)).square().mean().sqrt()),
               confidence=float(p.max(1).values.mean()),entropy=float(-(p*lp).sum(1).mean()),
               positive_occupancy=float((z>0).double().mean()),old_accuracy=float((logits.argmax(1)==old).double().mean()),
               increase_demand_fraction=float((D<0).double().mean()),increase_demand_magnitude=float((-D).clamp_min(0).mean()),
               uniform_sgd_mean_z_step=-.001*float(push.mean()),
               unit_m=mu.tolist(),unit_sd=sd.tolist(),unit_mean_abs_derivative=g.abs().mean(0).tolist(),
               unit_uniform_push=push.tolist())
        if model.family is not None:
            aa=model.family.alpha(0).double();ema=model.family.V[0].double()
            d.update(alpha_median=float(aa.median()),alpha_min=float(aa.min()),alpha_max=float(aa.max()),
                     alpha_clip_fraction=float(((aa<=.005)|(aa>=3.)).double().mean()),ema=ema[0].tolist())
        r['diagnostics']=d
    B.finite(r);return r

def trajectory(model,x,old,new,order,steps=4000):
    grid=set(B.points(steps));rows=[measure(model,x,old,new,0,True)];tracks=[]
    Y=F.one_hot(new,10).float();xbar=x.double().mean(0)
    for step,idx in enumerate(order[:steps],1):
        tr=step in (1,25,200,steps)
        out=one_step(model,x[idx],Y[idx],tr)
        if tr:
            loss,groups,dw,db=out
            tracks.append(dict(step=step,loss=loss,groups=groups,
                               mean_z_step=float((dw.double()@xbar+db.double()).mean())))
        if step in grid:rows.append(measure(model,x,old,new,step,step in (25,200,steps)))
    return rows,tracks

def apply_arm(model,arm,bank,initial):
    before=model.fingerprints()
    if arm=='N':
        checks=dict(before=before,after=before,native=True)
    else:
        checks=B.apply_arm(model,arm,bank,initial)
    after=model.fingerprints()
    if arm!='F0':assert before['activation_ema']==after['activation_ema']
    else:assert digest(model.snapshot()['ema'])==digest(initial['ema'])
    checks['activation_state_correct']=True
    return checks

def source(seed,act,out):
    path=out/f'source_{act}_s{seed}.pt';prov=provenance()
    if path.exists():
        s=torch.load(path,weights_only=False)
        assert s['provenance']['code_sha256']==prov['code_sha256']
        assert s['provenance']['spec_sha256']==prov['spec_sha256']
        return s,path
    x,subset=M.load_X(seed);labels,orders=B.stream_data(seed,AGES)
    model=Model(M.initial(seed,(784,100,10)),act);initial=model.snapshot()
    saved=np.load(B.OLD_STATES/f'states_adam_lr0.001_{act}_s{seed}.npz') if act in ('ELU','R') else None
    if saved is not None:
        assert np.array_equal(subset,saved['subset'])
        assert all(np.array_equal(q.detach().numpy(),saved[f'init{i}']) for i,q in enumerate(model.P))
    snapshots={};history=[];start=0;resume=out/f'resume_{act}_s{seed}.pt'
    if resume.exists():
        rr=torch.load(resume,weights_only=False)
        assert rr['code_sha256']==prov['code_sha256'] and rr['spec_sha256']==prov['spec_sha256']
        model.restore(rr['state']);snapshots=rr['snapshots'];history=rr['history'];start=rr['task']
    for t in range(start+1,31):
        Y=F.one_hot(labels[t],10).float()
        for idx in orders[t]:one_step(model,x[idx],Y[idx])
        B.finite(model.snapshot())
        rec=measure(model,x,labels[t],labels[t],4000,t in AGES);rec['task']=t
        if saved is not None:
            assert np.array_equal(labels[t].numpy(),saved['y'][t-1])
            errs=[float(np.abs(model.P[i].detach().numpy()-saved[f'end{i}'][t-1]).max()) for i in range(4)]
            if any(errs):raise AssertionError(('canonical source mismatch',act,seed,t,errs))
            rec['canonical_replay_exact']=True
        history.append(rec)
        if t in AGES:snapshots[t]=model.snapshot()
        save_state(resume,dict(task=t,state=model.snapshot(),snapshots=snapshots,history=history,
                               code_sha256=prov['code_sha256'],spec_sha256=prov['spec_sha256']))
        save_json(out/f'progress_{act}_s{seed}.json',dict(phase='source',task=t,activation=act,seed=seed))
        print('SOURCE',act,seed,t,flush=True)
    result=dict(activation=act,seed=seed,initial=initial,snapshots=snapshots,history=history,inputs=x,
                subset=subset,labels=labels,orders=orders,provenance=prov,
                data_sha256=sha(M.DATA/'train-images-idx3-ubyte.gz'),inputs_sha256=digest(x),
                labels_sha256={t:digest(y) for t,y in labels.items()},
                orders_sha256={t:digest(q) for t,q in orders.items()},
                initial_parameters_sha256=digest(initial['params_and_adam'][0]),complete=True)
    if saved is not None:result['archived_control_sha256']=sha(B.OLD_STATES/f'states_adam_lr0.001_{act}_s{seed}.npz')
    save_state(path,result)
    assert digest(torch.load(path,weights_only=False))==digest(result)
    return result,path

def run(seed,act,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    s,path=source(seed,act,out);x=s['inputs'];source_sha=sha(path);prov=provenance();files=[]
    model=Model(s['initial']['params_and_adam'][0],act)
    for age in AGES:
        model.restore(s['snapshots'][age])
        with torch.no_grad():
            mu=model.features(x).mean(0);bank=ops.geometry_bank(*model.head(),mu,seed,'mlp',age)
            oldmean=model.forward(x).double().mean(0)
        for arm in ARMS:
            stem=f'{act}_s{seed}_t{age}_{arm}';dest=out/(stem+'.json');end=out/(stem+'.pt');files.append(dest.name)
            if dest.exists():
                old=json.loads(dest.read_text());assert old['complete'] and old['source_sha256']==source_sha
                assert old['provenance']['code_sha256']==prov['code_sha256'] and old['provenance']['spec_sha256']==prov['spec_sha256']
                assert end.exists() and sha(end)==old['endpoint_sha256']
                continue
            model.restore(s['snapshots'][age]);checks=apply_arm(model,arm,bank,s['initial'])
            if arm in ('S1','S01','P1','P01','N'):
                with torch.no_grad():err=float((model.forward(x).double().mean(0)-oldmean).abs().max())
                tol=2e-5+2e-6*float(oldmean.abs().max());assert err<=tol,('mean logits',stem,err,tol)
                checks['mean_logit_maxabs']=err
            checks['geometry']=bank['metadata']['checks'];checks['permutation_sha256']=digest(bank['metadata']['P_permutation'])
            rows,tracks=trajectory(model,x,s['labels'][age],s['labels'][age+1],s['orders'][age+1])
            checks['legacy_continuations']=B.legacy_checks(rows,act,'off',seed,age,arm) if age in (1,10) and arm!='N' else []
            save_state(end,model.snapshot())
            record=dict(activation=act,seed=seed,t_end=age,arm=arm,horizon=4000,n_examples=1200,
                        source_sha256=source_sha,source_file=path.name,source_snapshot_sha256=digest(s['snapshots'][age]),
                        labels_sha256=s['labels_sha256'][age+1],batch_sha256=s['orders_sha256'][age+1],
                        provenance=prov,checks=checks,curve=rows,tracks=tracks,endpoint_file=end.name,
                        endpoint_sha256=sha(end),complete=True)
            save_json(dest,record);save_json(out/f'progress_{act}_s{seed}.json',dict(phase='fork',age=age,arm=arm,seed=seed,activation=act))
            print('DONE',stem,flush=True)
    save_json(out/f'complete_{act}_s{seed}.json',dict(complete=True,files=files,hashes={f:sha(out/f) for f in files},
               elapsed_s=time.monotonic()-started,source_sha256=source_sha,provenance=prov))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--activation',choices=ACTS,required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();assert a.seed in range(10);run(a.seed,a.activation,a.out)
