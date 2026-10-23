#!/usr/bin/env python3
"""Preregistered L0 endpoint ledger. Forward passes only, no optimizer."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('loadsign',ROOT/'analysis/chargpt_loadsign_1023/measure.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
A,C,torch,np=M.A,M.C,M.torch,M.np
OUT=ROOT/'results/chargpt_l0carrier_1023'
SOURCE=Path('/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/M2')
WINDOWS={'early1_10':(1,10),'bridge10_25':(10,25),'bridge25_39':(25,39),
         'middle39_50':(39,50),'bridge50_89':(50,89),'late89_100':(89,100),'cumulative39_100':(39,100)}
STATE_KEYS=['w_norm','c','sigma_hat','mu','s','r','A','B','b','p']
RAW_KEYS=STATE_KEYS+['m','alive']
SOURCES=['stretch','rotate','w','input','cross','bias']
TERMS=[f'{scale}_{s}' for scale in ['dmu','dr'] for s in SOURCES]+['dmu_total','dr_width','dr_total']


def dump(path,value):M.dump(path,value)
def med(x):return float(torch.quantile(x,.5))
def sha(p):return M.sha(p)


def check(delta,*scales):
    absolute=float(delta.abs().max());relative=M.relmax(delta,*scales)
    assert absolute<=1e-10,(absolute,relative)
    return absolute,relative


class Inputs(M.Inputs):
    def __init__(self):
        super().__init__();self.stats={};self.files={}

    def references(self,arm,seed):
        if (arm,seed) not in self.stats:
            p=SOURCE/arm/f'seed{seed}/layer_stats.csv'
            self.files[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,kind='reference_csv')
            self.stats[arm,seed]={(int(r['task']),int(r['u']),r['probe']):r for r in csv.DictReader(p.open()) if r['layer']=='0' and int(r['u']) in [100,10000]}
        return self.stats[arm,seed]

    def local(self,arm,seed,task):
        p=SOURCE/arm/f'seed{seed}/ckpts/u100_task{task}.pt'
        data=p.read_bytes();self.files[str(p)]=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),kind='u100_checkpoint')
        C.L=2;m=C.GPT(arm,'M');m.load_state_dict(torch.load(io.BytesIO(data),map_location='cpu',weights_only=True));m.eval().double()
        for b in m.h:b.mlp.track_v=False
        return m


@torch.no_grad()
def fields(model,x):
    block=model.h[0];W=block.mlp.up.weight.detach();bias=block.mlp.up.bias
    assert bias is None and block.n2.bias is None
    b=torch.zeros(C.FF,dtype=A.DT);xs=[];zs=[]
    for _,xx in A.chunks(x):
        cap=[];model.body(xx,cap)
        xs.append(block.n2(cap[0]).reshape(-1,C.D));zs.append(cap[1].reshape(-1,C.FF))
    xs,z=torch.cat(xs),torch.cat(zs);m=xs.mean(0);wn=W.norm(dim=1);mn=m.norm()
    assert (wn>0).all() and mn>0
    wh=W/wn[:,None];proj=xs@wh.T;sh=proj.std(0);s=wn*sh
    assert (sh>0).all() and torch.isfinite(s).all()
    mu=W@m+b;c=(wh@m)/mn;r=mu/s;aa=mn*c/sh;bb=b/s
    mu_z=z.mean(0);sd_z=z.std(0)
    identities={}
    for name,delta,scale in [('forward',z-xs@W.T-b,z),('mu',mu-mu_z,mu_z),('sd',s-sd_z,sd_z),('r',r-aa-bb,r)]:
        ab,re=check(delta,scale);identities[f'{name}_identity_abs']=ab;identities[f'{name}_identity_rel']=re
    xc=xs-m;cov=xc.T@xc/(len(xs)-1);trace=cov.trace();pr=trace.square()/cov.square().sum()
    raw=dict(W=W.clone(),wh=wh,w_norm=wn,m=m,c=c,sigma_hat=sh,mu=mu,s=s,r=r,A=aa,B=bb,b=b,
             alive=(z>0).any(0),p=(z>0).double().mean(0))
    scalars=dict(n_tokens=len(xs),m_norm=float(mn),m_norm_over_sqrt_d=float(mn/C.D**.5),cov_pr=float(pr),cov_trace=float(trace),
                 n_alive=int(raw['alive'].sum()),bias_present=False,zero_norm=0,zero_sd=0,
                 legacy_r_med=med(mu_z/(sd_z+1e-12)),legacy_sd_med=float(sd_z.median()),legacy_mu_med=float(mu_z.median()),
                 dead=float((~raw['alive']).double().mean()),**identities)
    return scalars,raw


def stats(raw,cohort):
    result=dict(n_cohort=int(cohort.sum()))
    for group,mask in [('all',torch.ones(C.FF,dtype=torch.bool)),('cohort',cohort)]:
        for key in STATE_KEYS:result[f'{group}_{key}_med']=med(raw[key][mask])
        result[f'{group}_frac_p_lt005']=float((raw['p'][mask]<.005).double().mean())
        result[f'{group}_dead']=float((~raw['alive'][mask]).double().mean())
    return result


def reference_check(inp,arm,seed,task,probe,step,row):
    ref=inp.references(arm,seed)[task,step,probe];out=[]
    for name,key in [('r_med','legacy_r_med'),('sd_med','legacy_sd_med'),('mu_med','legacy_mu_med'),('dead','dead')]:
        expected=float(ref[name]);got=row[key];tol=1e-3*abs(expected) if expected else 1e-6
        error=abs(got-expected)
        out.append(dict(metric=name,actual=got,expected=expected,absolute_error=error,tolerance=tol,passed=error<=tol))
    return dict(arm=arm,seed=seed,task=task,probe=probe,step=step,passed=all(c['passed'] for c in out),checks=out)


def ledger(old,new):
    dw=new['W']-old['W'];dm=new['m']-old['m']
    raw=dict(dmu_w=dw@old['m'],dmu_input=old['W']@dm,dmu_cross=dw@dm,dmu_bias=new['b']-old['b'],
             dmu_stretch=(new['w_norm']-old['w_norm'])*(old['wh']@old['m']),
             dmu_rotate=new['w_norm']*((new['wh']-old['wh'])@old['m']),dmu_total=new['mu']-old['mu'])
    for name in SOURCES:raw[f'dr_{name}']=raw[f'dmu_{name}']/new['s']
    raw['dr_width']=-old['mu']*(new['s']-old['s'])/(old['s']*new['s']);raw['dr_total']=new['r']-old['r']
    ids={}
    for name,delta,scale in [('mu',raw['dmu_total']-sum(raw[f'dmu_{n}'] for n in ['w','input','cross','bias']),raw['dmu_total']),
        ('w',raw['dmu_w']-raw['dmu_stretch']-raw['dmu_rotate'],raw['dmu_w']),
        ('r',raw['dr_total']-sum(raw[f'dr_{n}'] for n in ['w','input','cross','bias','width']),raw['dr_total'])]:
        ab,re=check(delta,scale);ids[f'{name}_ledger_abs']=ab;ids[f'{name}_ledger_rel']=re
    return raw,ids


def ledger_stats(raw,cohort):
    row={}
    for group,mask in [('all',torch.ones(C.FF,dtype=torch.bool)),('cohort',cohort)]:
        for key in TERMS:
            row[f'{group}_{key}_med']=med(raw[key][mask]);row[f'{group}_{key}_mean']=float(raw[key][mask].mean())
        row[f'{group}_abs_cumulative_dr_med']=med(raw['dr_total'][mask].abs())
        if 'total_variation' in raw:row[f'{group}_total_variation_med']=med(raw['total_variation'][mask])
    return row


def scale_check(raw):
    factor=torch.linspace(.5,2,C.FF,dtype=A.DT);new={k:v.clone() for k,v in raw.items()}
    new.update(W=raw['W']*factor[:,None],w_norm=raw['w_norm']*factor,mu=raw['mu']*factor,s=raw['s']*factor)
    terms,ids=ledger(raw,new);err=float((terms['dr_stretch']+terms['dr_width']).abs().max())
    assert err<1e-10 and torch.count_nonzero(terms['dr_total'])==0
    return dict(passed=True,stretch_width_cancel_abs=err,**ids)


def gate(inp):
    assert sha(M.BUNDLE)==M.BUNDLE_SHA
    refs=[];rows=[];raws={};rounded=[]
    known={('C3',0,10):dict(r_med=-.4293,sd_med=2.4617,mu_med=-1.0724,dead=0),
           ('C3',0,50):dict(r_med=-1.5437,sd_med=8.4266,dead=.0566),
           ('C3',0,100):dict(r_med=-3.4058,sd_med=20.5484,mu_med=-69.5471,dead=.6191),
           ('C3',4,100):dict(r_med=-1.0957,sd_med=13.7017,dead=.0039),
           ('ReLUcap',0,100):dict(r_med=-.7577,sd_med=.3215,dead=0)}
    for (arm,seed,task),values in known.items():
        model=inp.load(arm,seed,task)
        for probe,x in [('uniform',A.uniform(seed)),('chain',A.data(seed,task)[0])]:
            row,raw=fields(model,x);rows.append(dict(arm=arm,seed=seed,task=task,probe=probe,**row))
            refs.append(reference_check(inp,arm,seed,task,probe,10000,row));raws[arm,seed,task,probe]=raw
            if probe=='uniform':
                for key,expected in values.items():
                    got=row['dead' if key=='dead' else f'legacy_{key}'];tol=abs(expected)*1e-3+5e-5
                    rounded.append(dict(arm=arm,seed=seed,task=task,metric=key,actual=got,expected=expected,tolerance=tol,passed=abs(got-expected)<=tol))
    _,led=ledger(raws['C3',0,50,'uniform'],raws['C3',0,100,'uniform'])
    passed=all(r['passed'] for r in refs+rounded)
    dump(OUT/'gate.json',dict(passed=passed,code_sha256=sha(__file__),references=refs,rounded_K1=rounded,rows=rows,
         ledger=led,scale_check=scale_check(raws['C3',0,10,'uniform'])))
    assert passed,'K1 or reference mismatch: stop before full run'
    print('PASS: 5 K1 states, both probes, all identities and pure row-scale cancellation',flush=True)


def run(inp):
    g=json.loads((OUT/'gate.json').read_text());assert g['passed'] and g['code_sha256']==sha(__file__)
    start=time.monotonic();states=[];ledgers=[];windows=[];urows=[];phases=[];phase_windows=[];refs=[];sums=[]
    for arm in A.ARMS:
        for seed in range(10):
            ts=time.monotonic();_,ten=fields(inp.load(arm,seed,10),A.uniform(seed));cohort=ten['alive'];assert cohort.any()
            arrays={'cohort':cohort.numpy()};by_probe={p:{} for p in ['uniform','chain']};terms_by_probe={p:{} for p in by_probe}
            local_states=[];local_ledgers=[]
            for task in A.TASKS:
                model=inp.load(arm,seed,task)
                for probe,x in [('uniform',A.uniform(seed)),('chain',A.data(seed,task)[0])]:
                    row,raw=fields(model,x);ref=reference_check(inp,arm,seed,task,probe,10000,row);refs.append(ref)
                    if not ref['passed']:dump(OUT/'failed_reference.json',ref);raise RuntimeError('Reference mismatch')
                    row=dict(arm=arm,seed=seed,task=task,step=10000,probe=probe,**row,**stats(raw,cohort));local_states.append(row)
                    arrays.update({f't{task}_{probe}_{k}':raw[k].numpy() for k in RAW_KEYS})
                    if by_probe[probe]:
                        previous=max(by_probe[probe]);lr,ids=ledger(by_probe[probe][previous],raw);terms_by_probe[probe][previous,task]=lr
                        local_ledgers.append(dict(arm=arm,seed=seed,probe=probe,old_task=previous,new_task=task,task_gap=task-previous,**ids,**ledger_stats(lr,cohort)))
                        arrays.update({f'p{previous}_{task}_{probe}_{k}':v.numpy() for k,v in lr.items()})
                    by_probe[probe][task]=raw
            for probe,allterms in terms_by_probe.items():
                for name,(lo,hi) in WINDOWS.items():
                    chosen=[v for (a,b),v in allterms.items() if a>=lo and b<=hi]
                    agg={k:sum(v[k] for v in chosen) for k in TERMS};agg['total_variation']=sum(v['dr_total'].abs() for v in chosen)
                    err=float((agg['dr_total']-(by_probe[probe][hi]['r']-by_probe[probe][lo]['r'])).abs().max());assert err<1e-10;sums.append(err)
                    wr=dict(arm=arm,seed=seed,probe=probe,window=name,old_task=lo,new_task=hi,n_pairs=len(chosen),task_gap=hi-lo,n_cohort=int(cohort.sum()),telescope_abs=err,**ledger_stats(agg,cohort))
                    for k in TERMS:wr[f'cohort_pair_median_{k}']=float(np.median([med(v[k][cohort]) for v in chosen]))
                    windows.append(wr)
            localph=[]
            for task in range(90,101):
                model=inp.local(arm,seed,task);row,raw=fields(model,A.uniform(seed));ref=reference_check(inp,arm,seed,task,'uniform',100,row);refs.append(ref)
                if not ref['passed']:dump(OUT/'failed_reference.json',ref);raise RuntimeError('u100 reference mismatch')
                urows.append(dict(arm=arm,seed=seed,task=task,step=100,probe='uniform',**row,**stats(raw,cohort)))
                arrays.update({f'u100_t{task}_{k}':raw[k].numpy() for k in RAW_KEYS})
                for phase,old,new in [('head',by_probe['uniform'][task-1],raw),('body',raw,by_probe['uniform'][task])]:
                    lr,ids=ledger(old,new);localph.append((phase,task,lr))
                    phases.append(dict(arm=arm,seed=seed,task=task,phase=phase,n_updates=100 if phase=='head' else 9900,**ids,**ledger_stats(lr,cohort)))
                    arrays.update({f'{phase}_t{task}_{k}':v.numpy() for k,v in lr.items()})
            for phase in ['head','body']:
                chosen=[v for ph,t,v in localph if ph==phase];agg={k:sum(v[k] for v in chosen) for k in TERMS}
                agg['total_variation']=sum(v['dr_total'].abs() for v in chosen)
                phase_windows.append(dict(arm=arm,seed=seed,phase=phase,n_pairs=len(chosen),n_cohort=int(cohort.sum()),**ledger_stats(agg,cohort)))
            for task in range(90,101):
                pair=[v for ph,t,v in localph if t==task]
                assert float((pair[0]['dr_total']+pair[1]['dr_total']-terms_by_probe['uniform'][task-1,task]['dr_total']).abs().max())<1e-10
            states.extend(local_states);ledgers.extend(local_ledgers)
            M.write_csv(OUT/'parts'/f'{arm}_seed{seed}_states.csv',local_states)
            p=OUT/'raw'/f'{arm}_seed{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays)
            print(f'{arm} seed{seed}: 35 endpoints x2 probes +11 u100, cohort {int(cohort.sum())}, {time.monotonic()-ts:.1f}s',flush=True)
            A.CACHE.clear();A.UCACHE.clear()
    for name,rows in [('states',states),('ledger',ledgers),('windows',windows),('u100_states',urows),('phases',phases),('phase_windows',phase_windows)]:M.write_csv(OUT/f'{name}.csv',rows)
    dump(OUT/'reference_checks.json',dict(passed=all(r['passed'] for r in refs),n=len(refs),rows=refs))
    dump(OUT/'checks.json',dict(reference_passed=True,n_reference_checks=len(refs),max_telescope_abs=max(sums),
        max_identity_abs=max(r[k] for r in states+urows for k in r if k.endswith('_identity_abs')),
        max_ledger_abs=max(r[k] for r in ledgers+phases for k in r if k.endswith('_ledger_abs')),head_body_telescope_passed=True))
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256=sha(__file__),
        spec_sha256=sha(ROOT/'specs/spec_chargpt_l0carrier_1023.md'),bundle_path=str(M.BUNDLE),bundle_sha256=M.BUNDLE_SHA,
        model_states=700,state_probe_rows=len(states),u100_states=len(urows),ledger_pairs=len(ledgers),phase_pairs=len(phases),
        training_steps=0,optimizer_created=False,checkpoint_extracted=False,device='cpu',dtype='float64',threads=torch.get_num_threads(),seconds=time.monotonic()-start,
        borrowed_files=list(inp.files.values()),raw_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((OUT/'raw').glob('*.npz'))]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    inp=Inputs()
    (gate if args.mode=='gate' else run)(inp)


if __name__=='__main__':main()
