#!/usr/bin/env python3
"""Independent reverse-mode spot check of the new next-task full JVP."""
import json
import numpy as np
import torch
import measure as M
from reanalyze_p07 import next_gradient

def main():
    records=[]
    # Audit section 7 published these own deltas independently.
    for arm,t,expected in [('C3',50,-0.217213115),('SiLU',3,-0.136986301),('GELU',3,-0.175781250)]:
        m=M.load('M',arm,t); x,_,_,_=M.data(t); xn,yn,_,_=M.data(t+1)
        gn=next_gradient(m,xn,yn)
        with np.load(M.OUT/f'units_M_{arm}_t{t}.npz') as z:
            alive=z['p07_fixed_alive_last']
            delta=float((z['p07_fixed_next_own_last'][alive]>0).mean()-(z['p07_uniform_own_last'][alive]>0).mean())
            assert abs(delta-expected)<5.1e-10
            # Choose the live unit with the largest absolute new full response.
            full=z['p07_fixed_next_full_last']
            j=int(np.where(alive)[0][np.argmax(np.abs(full[alive]))])
            expected_full=float(full[j])
        # Reverse-mode scalar derivative, independent of torch.func.jvp.
        params=dict(m.named_parameters()); dot=0.
        for _,xx in M.chunks(x):
            cap=[]; m.body(xx,cap)
            grads=torch.autograd.grad(cap[-1][...,j].mean(),tuple(params.values()),allow_unused=True)
            dot+=sum(float((g*gn[n]).sum()) for (n,_),g in zip(params.items(),grads) if g is not None)*len(xx)/len(x)
        relative=abs(dot-expected_full)/max(abs(dot),abs(expected_full),1e-300)
        assert relative<1e-10
        records.append(dict(arm=f'M/{arm}',task=t,audit_own_delta=expected,recomputed_own_delta=delta,unit=j,next_full_jvp=expected_full,next_full_reverse=dot,relative_error=relative))
    result=dict(all_pass=True,scope='3 audit counterexample states; one live unit per state for reverse-mode next full vs JVP',checks=records)
    (M.OUT/'p07_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
