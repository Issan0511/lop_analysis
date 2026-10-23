#!/usr/bin/env python3
"""Post-result verification of the surprising early negative P2 correlation.

No predictor/threshold changes: finite-difference the actual CE and capacity
along six fixed unit-row r directions at task ends 1, 4, 6 (C3 seed0).
"""
import json
from pathlib import Path
import measure as E

torch,np,A,C,M,OUT=E.torch,E.np,E.A,E.C,E.M,E.OUT
IDS=[0,73,146,219,292,365]


@torch.no_grad()
def losses(m,x,rank):
    h=torch.cat([m.body(xx)[:,1:].reshape(-1,C.D) for _,xx in A.chunks(x)])
    eig=torch.linalg.eigvalsh(h.T@h)
    assert int((eig>eig[-1]*1e-10).sum())==rank
    f=m.head(h)
    return dict(P1=float(h.square().sum()/2),P2=float(eig[-rank:].log().sum()/2),
                switch=float((f.logsumexp(-1)-f.mean(-1)).mean()))


def main():
    inp=E.Inputs();rows=[]
    for task in [1,4,6]:
        m=inp.load('C3',0,task);raw=E.fields(m,0)[0]
        x=A.data(0,task)[0];w=m.h[0].mlp.up.weight;original=w.detach().clone()
        with np.load(OUT/'raw'/f'C3_seed0_task{task}.npz') as data:
            for eps in [1e-5,1e-6]:
                for unit in IDS:
                    direction=raw['kr'][unit]/raw['kr'][unit].norm()
                    vals=[]
                    for sign in [-1,1]:
                        with torch.no_grad():w[unit].copy_(original[unit]+sign*eps*direction)
                        vals.append(losses(m,x,127))
                    with torch.no_grad():w[unit].copy_(original[unit])
                    for name in ['P1','P2','switch']:
                        fd=(vals[1][name]-vals[0][name])/(2*eps)
                        expected=float(data['L0_SGD_r_'+name][unit])/float(raw['kr'][unit].norm())
                        rows.append(dict(task=task,unit=unit,epsilon=eps,predictor=name,fd=fd,expected=expected,
                                         sign_equal=bool(np.sign(fd)==np.sign(expected)),absolute=abs(fd-expected)))
        assert torch.equal(w.detach(),original)
    E.csvwrite(OUT/'early_fd_audit.csv',rows)
    checks=[]
    for task in [1,4,6]:
        for name in ['P1','P2','switch']:
            rr=[r for r in rows if r['task']==task and r['predictor']==name and r['epsilon']==1e-6]
            scale=max(abs(r['expected']) for r in rr)
            error=max(r['absolute'] for r in rr)/scale
            assert error<1e-4,(task,name,error)
            assert all(r['sign_equal'] for r in rr),(task,name,rr)
            checks.append(dict(task=task,predictor=name,relative_max_error=error,all_signs_match=True))
    E.dump(OUT/'early_fd_audit.json',dict(passed=True,units=IDS,checks=checks,post_result=True,
           code_sha256=M.sha(__file__),training_steps=0,parameters_restored=True))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
