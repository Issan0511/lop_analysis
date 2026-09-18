"""Uninstrumented original sampler/SGD replay of all new B/control trajectories."""
import json,subprocess,platform,hashlib
from pathlib import Path
import numpy as np
import torch
import run as r

def main():
    bitwise=[];parents=[]
    for arm in r.ARMS:
      for age in r.AGES:
        cp=r.base.load(arm,age)
        p=r.base.CPROOT/f'{arm}_step{age}.pt';parents.append(dict(path=str(p),sha256=r.base.sha(p)))
        for name in ['float64','float32']:
            dtype=getattr(torch,name);task=r.prepare(arm,age,dtype)
            parent=r.PARENT/f'{arm}_{age}_{name}_trajectory.npz';parents.append(dict(path=str(parent),sha256=r.base.sha(parent)))
            data=np.load(r.RAW/f'{arm}_{age}_{name}_B.npz')
            gen=torch.Generator().manual_seed(20260918)
            torch.randint(0,15,(10,),generator=gen);torch.randint(0,2,(10000,10,5),generator=gen)
            env=r.base.SCREnv(10,20,15,torch.full((10,),10000),torch.Generator().manual_seed(0),'cpu')
            env.load_state(task['ca']['env']);env.gen=gen
            net=r.base.new_net(task['ca'],dtype,2);lrs=torch.full((20,),r.LR,dtype=dtype)
            xall=torch.cat([task['xb'],task['xa']],1);yall=torch.cat([task['yb'],task['ya']],1)
            snapshots={int(t):i for i,t in enumerate(data['time'])};ids=torch.arange(20)
            # New B initial states have an independent autograd comparison too.
            cb=dict(task['ca'],env=dict(flip_state=task['flip_B'],t=age+10000))
            r.base.autodiff_check(cb,dtype)
            for t in range(10000):
                raw=env.step();ix=(raw[:,15:].long()*torch.tensor([16,8,4,2,1])).sum(-1)
                if not np.array_equal(ix.numpy(),data['B_support_index'][t]):raise AssertionError('RNG mismatch')
                if t==0:r.base.check('B_flip_replay',raw[:,:15],task['flip_B'],0.)
                xx=xall[ix.repeat(2),ids];yy=yall[ix.repeat(2),ids]
                z,a,f=net.forward(xx);gg=net.grads(xx,z,a,f-yy);net.sgd_step(lrs,*gg)
                if t+1 in snapshots:
                    i=snapshots[t+1]
                    for k,v in [('W',net.W),('b',net.b),('v',net.v),('cout',net.c)]:
                        ok=np.array_equal(v.double().numpy(),data[k][i]);bitwise.append(dict(arm=arm,age=age,dtype=name,time=t+1,param=k,equal=ok))
                        if not ok:raise AssertionError('replay not bitwise equal')
            print(arm,age,name,'bitwise replay passed',flush=True)
    checks=dict(checks=r.base.CHECKS,bitwise_comparisons=len(bitwise),all_bitwise_equal=all(x['equal'] for x in bitwise),all_pass=all(x['failed']==0 for x in r.base.CHECKS.values()))
    (r.OUT/'external_verification.json').write_text(json.dumps(checks,indent=2));r.base.writecsv(r.OUT/'bitwise_replay.csv',bitwise)
    provenance=dict(runner_git=json.loads((r.OUT/'checks_main.json').read_text())['git_hash'],verification_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip(),spec_sha256=r.base.sha(r.ROOT/'specs/spec_orth_reveal_0918.md'),parents=parents,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',torch_threads=1)
    (r.OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if not checks['all_pass']:raise SystemExit(1)
    print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
