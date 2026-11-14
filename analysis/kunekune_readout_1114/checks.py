"""Preproduction synthetic, state and operator checks; no production outcomes."""
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from . import run as R

def check():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    g=torch.Generator().manual_seed(9001);x=torch.randn(32,784,generator=g)
    y=torch.randint(10,(32,),generator=g);Y=F.one_hot(y,10).float();results={}
    for act in R.ACTS:
        m=R.Model(R.M.initial(9001,(784,100,10)),act)
        if m.family is not None:
            m.family.V[0].copy_(torch.linspace(.05,30,100)[None])
            z=torch.linspace(-40,40,101)[:,None].expand(-1,100).double().clone().requires_grad_(True)
            a=m.family.alpha(0).double();theta=2*a*z
            core=z+(1-torch.cos(theta))/(2*a)
            if act=='SNA':ref=core
            elif act in ('KKA','KKA23'):
                ref=torch.where(theta< -1.5*math.pi,2*z+(3*math.pi/4+.5)/a,
                                torch.where(theta>math.pi/2,2*z+(-math.pi/4+.5)/a,core))
                if act=='KKA23':ref=ref*(2/3)
            else:
                ref=torch.where(theta< -2*math.pi,z,torch.where(theta>math.pi,z+1/a,core))
            actual=m.phi(z);assert torch.allclose(actual,ref,atol=2e-5,rtol=2e-6)
            da,=torch.autograd.grad(actual.sum(),z,retain_graph=True)
            dr,=torch.autograd.grad(ref.sum(),z)
            assert torch.allclose(da,dr,atol=2e-6,rtol=2e-6)
            assert torch.allclose(da,m.family.dphi(z[None])[0],atol=2e-6,rtol=2e-6)
            # Independent finite differences away from joins, in double precision.
            zz=torch.randn(7,100,generator=g,dtype=torch.float64)
            dd=R.derivative(m,zz);fd=(m.phi(zz+1e-5)-m.phi(zz-1e-5))/(2e-5)
            assert torch.allclose(dd,fd,atol=5e-6,rtol=5e-5)
        state=m.snapshot();ema0=R.digest(state['ema'])
        R.measure(m,x,y,y,0,True)
        assert R.digest(m.snapshot()['ema'])==ema0
        with torch.no_grad():zbefore=m.features(x,True)[1].clone()
        R.one_step(m,x,Y)
        if m.family is not None:
            expected=state['ema']['V'][0]*.99+.01*zbefore.var(0,unbiased=False)[None]
            assert torch.equal(expected,m.family.V[0])
        after=R.digest(m.snapshot());m.restore(state);R.one_step(m,x,Y)
        assert R.digest(m.snapshot())==after
        state=m.snapshot();initial=R.Model(R.M.initial(9001,(784,100,10)),act).snapshot()
        with torch.no_grad():bank=R.ops.geometry_bank(*m.head(),m.features(x).mean(0),9001,'mlp',10)
        for arm in R.ARMS:
            m.restore(state);before=m.fingerprints();R.apply_arm(m,arm,bank,initial)
            if arm!='F0':assert m.fingerprints()['body']==before['body']
            if arm=='N':assert m.fingerprints()==before
            if arm in ('S1','S01','P1','P01','Z_keep'):
                assert m.fingerprints()['body_adam']==before['body_adam']
                assert all(not m.opt.state.get(p) for p in m.head())
            if arm in ('Z_clear','F0'):assert all(not m.opt.state.get(p) for p in m.P)
        # Canonical arithmetic and optimizer parity on synthetic data.
        if act in ('ELU','R'):
            old=R.B.MLP(R.M.initial(9001,(784,100,10)),act,'off')
            new=R.Model(R.M.initial(9001,(784,100,10)),act)
            for _ in range(15):R.B.one_step(old,x,Y);R.one_step(new,x,Y)
            assert R.digest(old.snapshot()['params_and_adam'])==R.digest(new.snapshot()['params_and_adam'])
        # Timing only; synthetic seed9001, no study selection.
        m.restore(state);start=time.monotonic()
        for _ in range(200):R.one_step(m,x[:16],Y[:16])
        results[act]=dict(ok=True,synthetic_seconds_per_update=(time.monotonic()-start)/200)
    R.save_json(R.ROOT/'results/kunekune_readout_1114/preflight.json',dict(results=results,production_outcomes_used=False,provenance=R.provenance()))
    print(json.dumps(results,indent=2))

if __name__=='__main__':check()
