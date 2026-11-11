#!/usr/bin/env python3
"""Post-run integrity checks. No additional research trajectories."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analyze import raw as resolve_raw
from src import leaky_boundary_1111 as B
import torch
import torch.nn.functional as F
O=ROOT/'results/leaky_boundary_1111'
def main():
    complete=json.loads((O/'COMPLETE.json').read_text());assert complete['prefixes']==5 and complete['probes']==70 and complete['spectra']==35
    hashes={}; counts={};first_checks=[]
    for mode,pat,n in [('prefix','prefix/s*/provenance.json',5),('probe','probes/*/s*/age_*/provenance.json',70),('spectrum','spectra/s*/age_*/provenance.json',35)]:
        pp=list(O.glob(pat));assert len(pp)==n;counts[mode]=n
        for p in pp:
            j=json.loads(p.read_text());assert (p.parent/'done.json').exists()
            assert j['git_hash']=='f890ea93e86746e3668ca9f45808f36f4855675e'
            assert j['spec_sha256']==B.sha(ROOT/'specs/spec_leaky_boundary_1111.md')
            hashes.setdefault(mode,set()).add(j['code_sha256'])
    assert len(hashes['prefix'])==len(hashes['probe'])==len(hashes['spectrum'])==1
    assert next(iter(hashes['prefix']))==B.sha(ROOT/'src/leaky_boundary_1111.py')
    curve_points=0;all_labels={};all_sources={}
    for p in O.glob('probes/*/s*/age_*/summary.csv'):
        sm=pd.read_csv(p).iloc[0];d=pd.read_csv(p.parent/'curves_t0.csv');seed=int(sm.seed)
        assert len(d)==int(sm.steps)//75+2 and d.step.iloc[-1]==sm.steps and d.step.is_unique
        assert np.isfinite(d[['acc','ce','L1_pplus','L2_pplus','L1_gmean','L2_gmean']]).all().all();curve_points+=len(d)
        ck=json.loads((p.parent/'branch_check.json').read_text());assert ck['initial_forward_equal']
        all_labels.setdefault(seed,set()).add(ck['probe_labels_sha256']);all_sources.setdefault((seed,sm.age),set()).add(ck['source_sha256'])
        source=O/'prefix'/f's{seed}'/f'state_{B.tag(sm.age)}.pt';assert B.sha(resolve_raw(source))==ck['source_sha256']
        raws=list(p.parent.glob('logits_*.npz'))
        if not raws:
            rel=str(p.parent.relative_to(ROOT))+'/'
            raws=[Path(f['backup']) for f in json.loads((O/'backup_manifest.json').read_text())['files'] if f['source_rel'].startswith(rel+'logits_') and f['source_rel'].endswith('.npz')]
        assert len(raws)==(6 if sm.steps==96000 else 4)
        for raw in raws:
            z=np.load(raw);assert np.isfinite(z['logits']).all();assert hashlib.sha256(z['labels'].tobytes()).hexdigest()==ck['probe_labels_sha256']
        if sm.arm in ['G','L']:
            for l in [1,2]:
                for metric in ['pplus','gmean','zero_positive_input']:
                    v=d[f'L{l}_{metric}'];assert np.max(np.abs(v-v.iloc[0]))<1e-12
    assert all(len(v)==1 for v in all_labels.values());assert all(len(v)==1 for v in all_sources.values())
    for seed in range(3,8):
        x=B.C.get_x(seed);y=B.probe_labels(seed);order=torch.randperm(len(x),generator=B.H.stream('boundary_probe_batch',seed));ids=order[:16]
        for age in [2.,3.]:
            ck=torch.load(resolve_raw(O/'prefix'/f's{seed}'/f'state_{B.tag(age)}.pt'),weights_only=False);grads={};delta={};initial={}
            for arm in ['N','G','L']:
                net=B.Model(ck['params'],x,arm);initial[arm]=net.forward(x[ids],ids)[2];grads[arm]=torch.autograd.grad(F.cross_entropy(initial[arm],y[ids]),net.ps)
                B.C.adam_step(net.ps,grads[arm],B.copy_adm(ck));delta[arm]=[p.detach()-q for p,q in zip(net.ps,ck['params'])]
            forward=max(float((initial['N']-initial[a]).abs().max()) for a in ['G','L'])
            ge=max(float((u-v).abs().max()) for a in ['G','L'] for u,v in zip(grads['N'],grads[a]))
            de=max(float((u-v).abs().max()) for a in ['G','L'] for u,v in zip(delta['N'],delta[a]))
            assert forward==ge==de==0,(seed,age,forward,ge,de)
            first_checks.append(dict(seed=seed,age=age,forward_maxabs=forward,gradient_maxabs=ge,actual_delta_maxabs=de))
    host={};a=Path('/tmp/leaky_boundary_smoke_prefix_final/state_2p00.pt');b=Path('/tmp/leaky_boundary_reference_old_host/state_t2.pt')
    if a.exists() and b.exists():
        aa=torch.load(a,weights_only=False);bb=torch.load(b,weights_only=False)
        for key in ['params','m','v']:host[key]=all(torch.equal(u,v) for u,v in zip(aa[key],bb[key]))
        for key in ['g_lab','g_batch']:host[key]=torch.equal(aa[key],bb[key])
        host['tc']=aa['tc']==bb['tc'];assert all(host.values())
        host['note']='seed100: 2 tasks x6000 updates; exact final state and RNG match old runner'
    else:
        previous=json.loads((O/'integrity_audit.json').read_text());host=previous['full_host_replication'];assert all(host[k] for k in ['params','m','v','g_lab','g_batch','tc'])
    result=dict(all_pass=True,counts=counts,curve_points=curve_points,git_hash='f890ea93e86746e3668ca9f45808f36f4855675e',source_and_labels_verified=True,finite_curves_and_raw_logits=True,frozen_masks_verified=True,initial_main_source_checks=first_checks,full_host_replication=host)
    (O/'integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
