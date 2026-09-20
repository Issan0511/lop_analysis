"""A5 synthetic independent references and full CPU test-seed100:101 pipeline."""
import argparse,copy,itertools,json,math,resource,time
from pathlib import Path
from statistics import NormalDist
import numpy as np
import torch
from analysis.initgeom_cifar_0920 import geometry as G,stats as S,run as R,report as Q
IDS=('A5-source-seed','A5-r','A5-dial','A5-sign','A5-normal','A5-forward','A5-permute','A5-C-counterexample','A5-band','A5-verdict','A5-resume-manifest','A5-cost')
MUTATIONS={
 'A5-source-seed':('seed_shift','reinitialize','zero_bias','Gaussian_init'),
 'A5-r':('sample_variance','squared_mu','inverse_r'),
 'A5-dial':('std_as_C','scale_whole_X','duplicate_raw'),
 'A5-sign':('threshold_1188','zeros_positive','exclude_ambiguous'),
 'A5-normal':('q196','missing_tail_factor','bias_in_main'),
 'A5-forward':('X_in_layer2','wrong_bias','r1_in_layer2'),
 'A5-permute':('X_only_permutation','misaligned_columns'),
 'A5-C-counterexample':('centered_f_zero',),
 'A5-band':('mean_SE','family_missing','raw_duplicate'),
 'A5-verdict':('hide_miss','partial_seed_success','monotone_as_required'),
 'A5-resume-manifest':('duplicate_seed','foreign_identity','partial_report','overwrite'),
 'A5-cost':('init_only','all_conditions_in_RAM')}


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError,KeyError,FileNotFoundError):return
    raise AssertionError('mutant survived')


def validate_checks(d):
    assert d['status']=='PASS' and d['source_sha256']==R.sources()
    assert set(d['checks'])==set(IDS) and all(x['pass'] is True for x in d['checks'].values())
    assert d['mutants']=={k:list(v) for k,v in MUTATIONS.items()}
    assert d['test_seeds']==[100,101] and d['resume_exact'] and d['scientific_seeds_used'] is False


def fixtures():
    torch.set_num_threads(2);torch.manual_seed(128)
    X=torch.tensor([[1.,2.,4.],[3.,4.,8.],[-1.,0.,-2.],[1.,6.,6.]],dtype=torch.double)
    g=G.geometry(X);mu=[math.fsum(float(x) for x in X[:,j])/4 for j in range(3)]
    trace=math.fsum((float(X[i,j])-mu[j])**2 for i in range(4) for j in range(3))/4
    norm=math.sqrt(math.fsum(x*x for x in mu));assert abs(g['trSigma']-trace)<=g['trSigma_error']
    assert abs(g['r']-norm/math.sqrt(trace))<=G.gamma(100)*g['r']
    for wrong in (norm/math.sqrt(trace*4/3),norm**2/math.sqrt(trace),math.sqrt(trace)/norm):assert wrong!=g['r']
    assert G.geometry(torch.zeros(4,3,dtype=torch.double))['status']=='UNDEFINED_INPUT'
    constant=G.geometry(torch.ones(4,3,dtype=torch.double));assert constant['p']==1 and constant['status']=='DEGENERATE_INPUT'
    assert S.probability(0)==0 and S.probability(math.inf)==1
    assert abs(NormalDist().inv_cdf(.99)-S.Q99)<2e-15
    assert abs(S.probability(2)-2*(1-NormalDist().cdf(S.Q99/2)))<4e-16
    for wrong in (S.probability(2,1.96),S.probability(2)/2,S.probability(math.sqrt(5))):assert wrong!=S.probability(2)
    # Analytic normal population: 99th percentile threshold, not random sample fitting.
    assert NormalDist(2.4,1).cdf(0)<.01 and NormalDist(2.2,1).cdf(0)>.01
    # Integer, zero and uncertain-unit controls (all units remain in denominator).
    z=torch.ones(1200,4,dtype=torch.double);z[:12,0]=-1;z[:11,1]=-1;z[:,2]=0;z[:,3]=-1
    a=G.signs(z,torch.zeros_like(z));assert a['one_sided'].tolist()==[False,True,False,True] and a['threshold']==1189
    assert a['f']==.5 and a['old_min_fraction']==.75 and a['all_zero_units']==1
    err=torch.zeros_like(z);err[:,1]=2;amb=G.signs(z,err);assert amb['f_lower']==.25 and amb['f_upper']==.5 and amb['units']==4
    assert .5!=3/4 and amb['f_lower']!=1/3
    # Centered distribution with 1199 positive observations is an explicit counterexample.
    skew=torch.ones(1200,1,dtype=torch.double);skew[-1]=-1199
    assert skew.mean()==0 and G.signs(skew,torch.zeros_like(skew))['f']==1
    # Independent scalar dot, host double forward, and covariance-preserving translation.
    raw=torch.rand(1200,3072);p32=R.E.H.init_params(100,torch.device('cpu'),R.E.DIMS);P=[p.detach().double() for p in p32]
    gen=R.E.H.stream('init',100);manual=[]
    for a,b in zip(R.E.DIMS[:-1],R.E.DIMS[1:]):
        lim=1/math.sqrt(a);manual.extend([(torch.rand((b,a),generator=gen)*2-1)*lim,(torch.rand(b,generator=gen)*2-1)*lim])
    assert all(torch.equal(x,y) for x,y in zip(p32,manual))
    assert R.tree_hash(p32)!=R.tree_hash(R.E.H.init_params(101,torch.device('cpu'),R.E.DIMS))
    assert all(p32[i].abs().sum()>0 for i in (1,3,5))
    assert R.tree_hash(p32)!=R.tree_hash([torch.randn_like(x) for x in p32])
    base=G.geometry(raw.double(),rank=False)
    for c in S.CONDITIONS:
        x,xe=G.transformed(raw,c)
        if c not in ('raw','std'):
            b=G.geometry(x,rank=False);gg={'C':0,'gamma025':.25,'gamma050':.5,'gamma075':.75,'gamma150':1.5,'gamma200':2.}[c]
            assert abs(b['trSigma']-base['trSigma'])<=b['trSigma_error']+base['trSigma_error']
            assert abs(b['r']-gg*base['r'])<b['r_upper']-b['r_lower']+G.gamma(100)*base['r']
    assert torch.equal(G.transformed(raw,'gamma100')[0],raw.double())
    assert not torch.equal(G.transformed(raw,'C')[0],G.transformed(raw,'std')[0])
    scaled=G.geometry(raw.double()*2,rank=False);assert abs(scaled['trSigma']-base['trSigma'])>base['trSigma_error']
    x,xe=G.transformed(raw,'raw');z1,e1=G.affine(x,P[0],P[1],xe);h=z1.clamp(min=0);z2,e2=G.affine(h,P[2],P[3],e1)
    ref=R.E.H.forward(P,x,R.E.make_act('C'));assert torch.equal(z1,ref[0]) and torch.equal(z2,ref[2])
    for i,j in ((0,0),(617,73),(1199,99)):
        scalar=math.fsum(float(x[i,k])*float(P[0][j,k]) for k in range(3072))+float(P[1][j])
        assert abs(scalar-float(z1[i,j]))<=float(e1[i,j])
    native=R.E.H.forward(p32,raw,R.E.make_act('C'))
    native_bound=(2*3072+3)*2**-24/(1-(2*3072+3)*2**-24)*(x.abs()@P[0].abs().T+P[1].abs())
    assert ((native[0].double()-z1).abs()<=native_bound).all()
    assert not torch.equal(z2,G.affine(h,P[2],P[1])[0])
    rejected(lambda:G.affine(x,P[2],P[3]))
    assert G.geometry(h,rank=False)['r']!=base['r']
    row=torch.randperm(1200);col=torch.randperm(3072)
    zp,ep=G.affine(x[row],P[0],P[1]);assert torch.equal(G.signs(zp,ep)['positive'],G.signs(z1,e1)['positive'])
    zc,ec=G.affine(x[:,col],P[0][:,col],P[1]);assert ((zc-z1).abs()<=ec+e1).all()
    wrong=G.affine(x[:,col],P[0],P[1])[0];assert not torch.allclose(wrong,z1)
    # Exhaustive n=4, units=3 distribution, independently using combinatorial PMFs.
    ps=[.125,.25,.5,.75];cdf=S.order_cdf(ps,3);exact=np.zeros((2,4));mass=0
    for ks in itertools.product(range(4),repeat=4):
        prob=math.prod(math.comb(3,k)*p**k*(1-p)**(3-k) for k,p in zip(ks,ps));mass+=prob;ordered=sorted(ks)
        for j in range(2):
            for k in range(4):exact[j,k]+=prob*int(ordered[j+1]<=k)
    assert abs(mass-1)<1e-14 and np.max(np.abs(cdf-exact))<1e-14
    for p,k in ((0,0),(1,1)):
        b=S.median_band([p]*20);assert b['low']==b['high']==k
    b=S.median_band([.2]*20);assert b!=S.median_band([.2]*20,alpha=.05)
    assert len(S.CONDITIONS)==8 and 'gamma100' not in S.CONDITIONS
    assert S.median_band([.2]*20)['low']!=.2-1.96*math.sqrt(.2*.8/(100*20))
    labels=['PREDICTED','OFF_HIGH','OFF_LOW','NUMERIC_UNRESOLVED','UNDEFINED','DIVERGED']
    observations=[(.2,.3),(.5,.5),(.1,.1),(.1,.3),(None,None),(float('nan'),float('nan'))]
    for (lo,hi),label in zip(observations,labels):assert S.condition_label(lo,hi,{'low':.2,'high':.4})==label
    family={c:'PREDICTED' for c in S.CONDITIONS};assert S.family(family,1)=='L1_ALL_COMPATIBLE'
    family['raw']='OFF_HIGH';assert S.family(family,1)=='L1_MODEL_MISS';family.pop('C');assert S.family(family,1)=='INCOMPLETE'
    assert S.monotone([(0,0)]*6)=='MONOTONE_SAMPLE_MEDIANS'
    assert S.monotone([(1,1)]+[(0,0)]*5)=='NONMONOTONE_SAMPLE_MEDIANS'
    assert S.monotone([(0,1)]*6)=='NUMERIC_UNRESOLVED'
    assert S.paired_interval([0]*20)['sign']=='0';rejected(lambda:S.paired_interval([0]*19))
    rejected(lambda:Q.evaluate([]))
    return dict(exhaustive_outcomes=4**4,forward_scalar_units=3,centered_counterexample=True,independent_init=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--fixtures-only',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();f=fixtures();R.put(out/'fixtures.json',dict(status='PASS',details=f,source_sha256=R.sources()))
    if a.fixtures_only:print('A5 fixtures PASS; full test-seed pipeline NOT RUN');return
    R.run(out/'continuous');records=Q.validate(out/'continuous',production=False)
    assert len(records)==2*8*2
    first=R.run(out/'resumed',stop_after=1);assert first['status']=='STOPPED'
    rejected(lambda:Q.validate(out/'resumed',production=False))
    R.run(out/'resumed',resume=True);resumed=Q.validate(out/'resumed',production=False)
    assert records==resumed,'responses differ after seed-boundary resume'
    for seed in (100,101):
        c=json.loads((out/f'continuous/seed{seed}/complete.json').read_text());r=json.loads((out/f'resumed/seed{seed}/complete.json').read_text())
        assert c==r,'all parameters, input predictions, raw arrays and checksums must match'
    marker=out/'continuous/complete.json';orig=marker.read_text();m=json.loads(orig)
    for which in ('duplicate','foreign','partial'):
        bad=copy.deepcopy(m)
        if which=='duplicate':bad['seeds']=[100,100]
        elif which=='foreign':bad['identity']['run_id']='foreign'
        else:bad['seed_markers'].pop('101')
        R.put(marker,bad);rejected(lambda:Q.validate(out/'continuous',production=False));marker.write_text(orig)
    existing=out/'continuous/seed100/prediction_manifest.json';rejected(lambda:R.put(existing,{},immutable=True))
    rejected(lambda:Q.validate(out/'continuous',production=True))
    costs=json.loads((out/'continuous/cost.json').read_text());assert len(costs)==2 and all(c['conditions']==8 and c['layers']==2 and c['images']==1200 and c['bias0'] for c in costs)
    assert all(c['bytes']>8*1200*100*8 for c in costs),'initialization-only cost is not acceptable'
    # Resource evidence is from the actual sequential whole pipeline, not multiplying init time.
    assert 'for c in S.CONDITIONS:' in Path(R.__file__).read_text()
    result=dict(status='PASS',source_sha256=R.sources(),checks={i:{'pass':True} for i in IDS},mutants={k:list(v) for k,v in MUTATIONS.items()},test_seeds=[100,101],resume_exact=True,scientific_seeds_used=False,fixtures=f,cost=costs,seconds=time.monotonic()-t0,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    validate_checks(result);R.put(out/'checks.json',result);print('A5 PASS: fixtures, test-seed full pipeline, exact seed-boundary resume and completeness guards')
if __name__=='__main__':main()
