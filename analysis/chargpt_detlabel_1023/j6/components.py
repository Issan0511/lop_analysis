#!/usr/bin/env python3
"""Exploratory, exact softmax secant decomposition of the direct signal.

J6（spec_chargpt_detlabel_1023 §6.5）の第二段階。写し元:
  analysis/chargpt_loadsign_1023/components.py  git blob 68f97baadf1c22d51ee55600894ad0a13a6bcfc0（commit 3baa567f）
分解（apply_h・measure の中の計算と検算）は変えない。変えた所:
  1. checkpoint の読み元（inputs.load、模型の腕は M.MODEL_ARM）。2. 鎖 probe を腕の α で（inputs.data）。
  4. 腕は J6_ARM の 1 つ（M.LABEL）。main は 1 腕について --seeds/--tasks を回し、完走した seed を飛ばす。
  第一段階の値との一致の確認（stage1_error）はそのまま残す（同じ腕の OUT/raw を読む）。
"""
import argparse
import csv
import json
import subprocess
import time
import measure as M

A,C,torch,np=M.A,M.C,M.torch,M.np
OUT=M.OUT


def apply_h(d,z):
    return d*(z-(d*z).sum(-1,keepdim=True)/d.sum(-1,keepdim=True))


@torch.no_grad()
def measure(inputs,arm,seed,task):
    model=inputs.load(M.MODEL_ARM,seed,task)
    x,labels,_,_=inputs.data(seed,task)
    ybar,alive,_,vc,vl,_=M.task_fields(model,x,labels)
    mlp=model.h[1].mlp; U=mlp.down.weight
    Uc=U-U.mean(0,keepdim=True); gamma=model.nf.weight
    QL=vl@(gamma[:,None]*Uc)
    xu=A.uniform(seed); N=xu.numel()
    acc={f'{e}_{part}':torch.zeros(C.FF,dtype=A.DT)
         for e in ['original','loaderror'] for part in ['self','other','residual','direct','self_absK']}
    max_error=0.; max_sum_error=0.
    for _,xx in A.chunks(xu):
        cap=[]; h=model.body(xx,cap).reshape(-1,C.D)
        r=cap[2].reshape(-1,C.D); z=cap[3].reshape(-1,C.FF)
        a=mlp.phi(z); gate=mlp.dphi(z); y=model.h[1].n2(r)
        ffn=mlp.down(a); t=r+ffn
        sigma=(M.cent(t).square().mean(-1,keepdim=True)+model.nf.eps).sqrt()
        k=y@ybar
        for name,vf in [('original',vc),('loaderror',vl)]:
            f=h@vf.T; p=f.softmax(-1); err=p-1/C.V
            alpha=f.logsumexp(-1,keepdim=True)-np.log(C.V)
            delta=f-alpha
            # Exact smooth positive ratio; series avoids cancellation at zero.
            safe=torch.where(delta.abs()<1e-7,torch.ones_like(delta),delta)
            d=torch.where(delta.abs()<1e-7,(1+delta/2+delta.square()/6+delta.pow(3)/24)/C.V,
                          torch.expm1(delta)/(C.V*safe))
            assert bool((d>0).all())
            herr=apply_h(d,f)
            max_error=max(max_error,M.relmax(herr-err,err))
            QF=vf@(gamma[:,None]*Uc)
            self_coeff=d@(QL*QF)-(d@QL)*(d@QF)/d.sum(-1,keepdim=True)
            self_term=a*self_coeff/sigma.square()
            rz=(M.cent(r)*gamma)@vf.T
            fz=(M.cent(ffn)*gamma)@vf.T
            resid_term=(apply_h(d,rz)@QL)/sigma.square()
            ffn_term=(apply_h(d,fz)@QL)/sigma.square()
            direct=(err@QL)/sigma
            max_sum_error=max(max_sum_error,M.relmax(ffn_term+resid_term-direct,direct,ffn_term,resid_term))
            parts=dict(self=self_term,other=ffn_term-self_term,residual=resid_term,direct=direct)
            for part,v in parts.items():acc[f'{name}_{part}']+=(k[:,None]*gate*v).sum(0)/N
            acc[f'{name}_self_absK']+=(k.abs()[:,None]*gate*self_term).sum(0)/N
    assert max_error<=1e-10 and max_sum_error<=1e-10,(max_error,max_sum_error)
    assert float(acc['loaderror_self_absK'].min())>=-1e-12*float(acc['loaderror_self_absK'].abs().max())
    prior=np.load(OUT/'raw'/f'{arm}_seed{seed}.npz')
    raw_err=0.
    row=dict(arm=arm,seed=seed,task=task,n_alive=int(alive.sum()),softmax_identity_error=max_error,
             token_sum_error=max_sum_error)
    for e in ['original','loaderror']:
        direct=acc[f'{e}_direct'];self_=acc[f'{e}_self'];other=acc[f'{e}_other'];resid=acc[f'{e}_residual']
        before=torch.from_numpy(prior[f't{task}_{e}_direct_K'])
        raw_err=max(raw_err,M.relmax(direct-before,before))
        assert M.relmax(direct-self_-other-resid,direct,self_,other,resid)<=1e-10
        for part in ['self','other','residual','direct','self_absK']:
            v=acc[f'{e}_{part}'];row[f's_{e}_{part}']=A.sink(v,alive)
            row[f'abs_{e}_{part}']=M.median(v[alive].abs())
        den=self_[alive].abs()+other[alive].abs()+resid[alive].abs(); nz=den>0
        row[f'{e}_self_abs_share']=M.median(self_[alive][nz].abs()/den[nz])
        row[f'{e}_other_abs_share']=M.median(other[alive][nz].abs()/den[nz])
        row[f'{e}_residual_abs_share']=M.median(resid[alive][nz].abs()/den[nz])
        row[f'{e}_share_missing']=int((~nz).sum())
        row[f'{e}_self_positive_direct_nonpositive']=float(((self_[alive]>0)&(direct[alive]<=0)).double().mean())
        row[f's_{e}_self_plus_other']=A.sink(self_+other,alive)
        row[f's_{e}_self_plus_residual']=A.sink(self_+resid,alive)
    assert raw_err<=1e-10,(arm,seed,task,raw_err)
    row['stage1_error']=raw_err
    acc['alive']=alive
    return row,acc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run'])
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    ap.add_argument('--tasks',type=int,nargs='+',default=list(A.TASKS));args=ap.parse_args()
    inp=M.Inputs();arm=M.LABEL
    if args.mode=='gate':
        rr=[measure(inp,arm,0,t)[0] for t in M.GATE_TASKS]
        M.dump(OUT/'components_gate.json',dict(passed=True,rows=rr,code_sha256=M.sha(__file__)))
        print('component gate passed');return
    g=json.loads((OUT/'components_gate.json').read_text());assert g['passed'] and g['code_sha256']==M.sha(__file__)
    rows=[];start=time.monotonic()
    for seed in args.seeds:
        p=OUT/'parts_components'/f'{arm}_seed{seed}.csv';q=OUT/'raw_components'/f'{arm}_seed{seed}.npz'
        if p.with_suffix('.json').exists():
            d=json.loads(p.with_suffix('.json').read_text())
            assert d['code_sha256']==M.sha(__file__) and d['sha256']==M.sha(p) and d['raw_sha256']==M.sha(q) and d['tasks']==args.tasks
            rows.extend(list(csv.DictReader(p.open())));continue
        ts=time.monotonic();rr=[];raw={}
        for t in args.tasks:
            r,arrays=measure(inp,arm,seed,t);rr.append(r)
            raw.update({f't{t}_{k}':v.numpy() for k,v in arrays.items()})
        M.write_csv(p,rr)
        q.parent.mkdir(exist_ok=True)
        np.savez_compressed(q,**raw)
        M.dump(p.with_suffix('.json'),dict(sha256=M.sha(p),raw_sha256=M.sha(q),code_sha256=M.sha(__file__),seconds=time.monotonic()-ts,tasks=args.tasks))
        rows.extend(rr);print(f'components {arm} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        inp.clear()
    M.write_csv(OUT/'components_states.csv',rows)
    M.dump(OUT/'components_provenance.json',dict(script_sha256=M.sha(__file__),seconds=time.monotonic()-start,training_steps=0,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip(),
        spec_sha256=M.sha(M.ROOT/'specs/spec_chargpt_detlabel_1023.md'),rows=len(rows),arm=arm,model_arm=M.MODEL_ARM,alpha=M.ALPHA,
        max_stage1_error=max(float(r['stage1_error']) for r in rows),max_softmax_error=max(float(r['softmax_identity_error']) for r in rows),
        max_token_sum_error=max(float(r['token_sum_error']) for r in rows)))


if __name__=='__main__':main()
