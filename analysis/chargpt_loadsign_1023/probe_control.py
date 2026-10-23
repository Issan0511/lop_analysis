#!/usr/bin/env python3
"""Matched current-task inputs: switch expectation vs original-task CE."""
import argparse,csv,json,subprocess,time
import measure as M
from report import WINDOWS
A,C,torch,np=M.A,M.C,M.torch,M.np
OUT=M.OUT


@torch.no_grad()
def calculate(model,x,y):
    hs=[];rs=[];zs=[];ys=[]
    for _,xx in A.chunks(x):
        cap=[];hs.append(model.body(xx,cap).reshape(-1,C.D))
        rs.append(cap[2].reshape(-1,C.D));zs.append(cap[3].reshape(-1,C.FF))
        ys.append(model.h[1].n2(cap[2]).reshape(-1,C.D))
    h,r,z,xx=map(torch.cat,[hs,rs,zs,ys]);ybar=xx.mean(0);alive=(z>0).any(0)
    target=torch.nn.functional.one_hot(y.reshape(-1),C.V).to(A.DT)
    vc=model.head.weight-model.head.weight.mean(0);F=(target-1/C.V).T@h
    kappa=(vc*F).sum()/F.square().sum();vl=kappa*F
    logits=model.head(h);p=logits.softmax(-1);gate=model.h[1].mlp.dphi(z)
    t=r+model.h[1].mlp.down(model.h[1].mlp.phi(z));sig=(M.cent(t).square().mean(-1,keepdim=True)+model.nf.eps).sqrt();q=M.cent(t)/sig
    k=xx@ybar;U=model.h[1].mlp.down.weight;push={}
    for errname,err in [('switch',p-1/C.V),('old',p-target)]:
        for vn,v in [('full',vc),('load',vl)]:
            b=model.nf.weight*(err@v);back=(M.cent(b)-q*(q*b).mean(-1,keepdim=True))/sig
            push[f'{errname}_{vn}']=(k[:,None]*gate*(back@U)).mean(0)
    row=dict(n_alive=int(alive.sum()),kappa=float(kappa),accuracy=float((p.argmax(-1)==y.reshape(-1)).double().mean()),
             mean_p_true=float((p*target).sum(-1).mean()),mean_p_max=float(p.max(-1).values.mean()),
             entropy=float(-(p*p.clamp_min(1e-300).log()).sum(-1).mean()),CE=float(A.ce(logits,y.reshape(-1))))
    for name,v in push.items():row['s_'+name]=A.sink(v,alive);row['abs_'+name]=M.median(v[alive].abs())
    row['old_over_switch_full_abs']=row['abs_old_full']/row['abs_switch_full']
    return row,push,ybar,vl


def state(inp,arm,seed,t,gate=False):
    model=inp.load(arm,seed,t);x,y,_,H=A.data(seed,t)
    row,push,ybar,vl=calculate(model,x,y)
    ref=inp.ref[arm,seed,t]
    assert row['n_alive']==int(ref['n_alive']) and A.rel(row['kappa'],ref['kappa'])<=1e-10
    err=0.
    if gate:
        gs=M.P.fixed_error_gradients(model,x,[vl])[0][1]
        old=A.gradient(model,x,y)['h.1.mlp.up.weight']
        new=A.gradient(model,x)['h.1.mlp.up.weight']
        for name,g in [('switch_load',gs),('old_full',old),('switch_full',new)]:
            err=max(err,M.relmax(push[name]-(g*ybar).sum(1),push[name]))
        assert err<=1e-10
    return dict(arm=arm,seed=seed,task=t,H_true=H,uniform_s_load=float(ref['s_load']),uniform_s_full=float(ref['s_full']),
                gate_error=err if gate else None,**row)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args();inp=M.Inputs()
    if args.mode=='gate':
        rows=[state(inp,a,0,t,True) for a in A.ARMS for t in [1,9,50,100]]
        M.dump(OUT/'probe_control_gate.json',dict(passed=True,rows=rows,max_error=max(r['gate_error'] for r in rows),code_sha256=M.sha(__file__)));print('probe control gate passed');return
    g=json.loads((OUT/'probe_control_gate.json').read_text());assert g['passed'] and g['code_sha256']==M.sha(__file__)
    rows=[];start=time.monotonic()
    for seed in range(10):
        for arm in A.ARMS:
            ts=time.monotonic();rr=[state(inp,arm,seed,t) for t in A.TASKS]
            M.write_csv(OUT/'parts_probe'/f'{arm}_seed{seed}.csv',rr);rows.extend(rr)
            print(f'probe {arm} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        A.CACHE.clear();A.UCACHE.clear()
    M.write_csv(OUT/'probe_control_states.csv',rows)
    fields=[k for k in rows[0] if k not in ['arm','seed','task','gate_error']];wins=[]
    for a in A.ARMS:
        for s in range(10):
            for w,tasks in WINDOWS.items():
                rr=[r for r in rows if r['arm']==a and r['seed']==s and r['task'] in tasks]
                wins.append(dict(arm=a,seed=s,window=w,**{k:float(np.median([r[k] for r in rr])) for k in fields}))
    M.write_csv(OUT/'probe_control_windows.csv',wins)
    judge={a:sum(r['s_switch_load']>=.75 and r['s_switch_load']-r['uniform_s_load']>=.2 for r in wins if r['arm']==a and r['window']=='late') for a in A.ARMS}
    M.dump(OUT/'probe_control_judge.json',dict(exploratory=True,n_restored=judge))
    M.dump(OUT/'probe_control_provenance.json',dict(code_sha256=M.sha(__file__),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip(),
        spec_sha256=M.sha(M.ROOT/'specs/spec_chargpt_loadsign_1023.md'),rows=len(rows),training_steps=0,seconds=time.monotonic()-start))


if __name__=='__main__':main()
