"""A5 CPU-only, one-seed-at-a-time run; predictions are persisted before responses."""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import numpy as np
import torch
from src import relu_doors_0919 as E
from src.cifar_interventions_0920 import tree_hash,save_pt,DATA
from analysis.cifar_ledger_0920.replay import sha,save_npz
from analysis.initgeom_cifar_0920 import geometry as G,stats as S
RUN='initgeom_cifar_0920'


def put(path,x,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(G.serial(x),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists() and immutable:assert path.read_text()==content,'refusing altered prior output';return
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(content);tmp.replace(path)


def sources():
    paths=['specs/spec_initgeom_cifar_0920.md','src/relu_doors_0919.py','src/pmnist_0905.py','src/pmnist_rlcifar_0907.py','src/pmnist_rlmnist_0906.py','src/cifar_interventions_0920.py','src/rlcifar_mlp_battle_0918.py','analysis/cifar_ledger_0920/replay.py','analysis/cifar_ledger_0920/ledger.py','analysis/resp_cifar_ee_0920/stats.py']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'analysis'/RUN).glob('*.py'))]
    return {p:sha(ROOT/p) for p in paths}


def seed_audit():
    """Read seed metadata only; never load initialization or outcome arrays."""
    paths=subprocess.check_output(['rg','--files','results','-g','*provenance*.json','-g','*config*.json','-g','*input_manifest*.json'],cwd=ROOT,text=True).splitlines()
    inspected=[];overlaps=[]
    def seed_fields(o):
        found=[]
        if isinstance(o,dict):
            for k,v in o.items():
                if k=='seed' and isinstance(v,int):found.append(v)
                elif k=='seeds' and isinstance(v,list):found.extend(x for x in v if isinstance(x,int))
                elif isinstance(v,(dict,list)):found.extend(seed_fields(v))
        elif isinstance(o,list):
            for v in o:found.extend(seed_fields(v))
        return found
    for rel in paths:
        if not any(k in rel.lower() for k in ('cifar','relu_doors')) or RUN in rel:continue
        p=ROOT/rel
        try:obj=json.loads(p.read_text())
        except (json.JSONDecodeError,UnicodeError):continue
        seeds=sorted(set(seed_fields(obj)));overlap=sorted(set(seeds)&set(range(20,40)))
        inspected.append(dict(path=rel,sha256=sha(p),seeds=seeds))
        if overlap:overlaps.append(dict(path=rel,seeds=overlap))
    return dict(status='CLEAR' if not overlaps else 'OVERLAP_REQUIRES_PREREG_APPENDIX',scope='local CIFAR/relu_doors provenance/config/input metadata',inspected=inspected,overlaps=overlaps)


def verify_files(out,record):
    assert record['files'] and len(record['files'])==len({f['path'] for f in record['files']})
    for f in record['files']:
        rel=Path(f['path']);assert not rel.is_absolute() and '..' not in rel.parts
        p=Path(out)/rel;assert p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'file integrity'


def seed_run(out,seed,cifar,ident):
    out=Path(out);base=out/f'seed{seed}';base.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic();raw32=cifar.images(E.RC.subset_idx(seed),torch.device('cpu'))
    assert raw32.shape==(1200,3072)
    stats={};rawrank=None
    # No W is initialized/read in this phase. Persist all L1 input-only predictions first.
    for c in S.CONDITIONS:
        x,xe=G.transformed(raw32,c)
        g=G.geometry(x,rank=c in ('raw','std'),rank_reference=rawrank)
        if c=='raw':rawrank=g['effective_rank'];rawmu=g['mu'];rawtr=g['trSigma']
        if c not in ('raw','std'):
            centered=x-x.mean(0);reference=raw32.double()-raw32.double().mean(0)
            bound=G.gamma(2408)*(x.abs()+raw32.double().abs()+rawmu.abs()[None,:]*4)+xe
            assert ((centered-reference).abs()<=bound).all(),'dial covariance changed'
            g['effective_rank_reference']='raw covariance; centered-array equality checked within rounding bound'
        stats[c]=G.serial(g)
        del x,xe
    put(base/'prediction_manifest.json',dict(seed=seed,conditions=list(S.CONDITIONS),alias={'gamma100':'raw'},layer=1,input_sha256=tree_hash(raw32),subset_sha256=tree_hash(E.RC.subset_idx(seed)),source_sha256=ident['source_sha256'],statistics=stats),immutable=True)
    P32=E.H.init_params(seed,torch.device('cpu'),E.DIMS)
    P=[p.detach().double() for p in P32]
    init_sha=tree_hash(P32);save_pt(base/'initial.pt',dict(P=P32,seed=seed,subset=E.RC.subset_idx(seed),input_sha256=tree_hash(raw32)))
    records=[]
    for c in S.CONDITIONS:
        x,xe=G.transformed(raw32,c)
        z1,ze1=G.affine(x,P[0],P[1],xe);h=z1.clamp(min=0)
        g2=G.geometry(h)
        put(base/f'{c}_layer2_prediction.json',dict(seed=seed,condition=c,conditional_on='measured native-bias layer1 activations',statistics=g2,P_sha256=init_sha,source_sha256=ident['source_sha256']),immutable=True)
        z2,ze2=G.affine(h,P[2],P[3],ze1)
        d1=G.diagnostic(x,P[0],P[1],z1,ze1,stats[c]);d2=G.diagnostic(h,P[2],P[3],z2,ze2,g2)
        z10,e10=G.affine(x,P[0],torch.zeros_like(P[1]),xe);h0=z10.clamp(min=0)
        g20=G.geometry(h0);z20,e20=G.affine(h0,P[2],torch.zeros_like(P[3]),e10)
        b1=G.diagnostic(x,P[0],torch.zeros_like(P[1]),z10,e10,stats[c]);b2=G.diagnostic(h0,P[2],torch.zeros_like(P[3]),z20,e20,g20)
        for layer,d,bias0 in ((1,d1,b1),(2,d2,b2)):
            obj=dict(seed=seed,condition=c,layer=layer,main=d,bias0=bias0,P_sha256=init_sha,source_sha256=ident['source_sha256'])
            path=base/f'{c}_layer{layer}.json';put(path,obj,immutable=True);records.append(str(path.relative_to(out)))
        save_npz(base/f'{c}_raw.npz',dict(z1=z1.numpy(),z2=z2.numpy(),error1=ze1.numpy(),error2=ze2.numpy(),bias0_z1=z10.numpy(),bias0_z2=z20.numpy()))
        del x,xe,z1,ze1,z2,ze2,h,z10,h0,z20,d1,d2,b1,b2
    files=[]
    for p in sorted(base.iterdir()):
        if p.name=='complete.json':continue
        assert p.is_file();files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    complete=dict(seed=seed,identity=ident,files=files,records=records,P_sha256=init_sha)
    put(base/'complete.json',complete,immutable=True)
    return complete,dict(seed=seed,seconds=time.monotonic()-t0,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,bytes=sum(f['bytes'] for f in files),conditions=8,layers=2,bias0=True,images=1200)


def run(out,mode='check',resume=False,stop_after=None):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);E.RC.DATA_DIR=DATA
    assert mode in ('check','production');seeds=[100,101] if mode=='check' else list(range(20,40))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cifar=E.RC.Cifar10()
    ident=dict(run_id=RUN,mode=mode,seeds=seeds,conditions=list(S.CONDITIONS),alias={'gamma100':'raw'},layers=[1,2],source_sha256=sources(),data_sha256=cifar.sha256,geometry_dtype='float64',host_init_dtype='float32',images=1200,units=100,device='cpu',threads=2)
    manifest=out/'input_manifest.json';state=out/'state.json'
    if manifest.exists():
        assert resume and json.loads(manifest.read_text())==ident,'resume identity or overwrite'
        current=json.loads(state.read_text()) if state.exists() else dict(identity=ident,seeds=[],cost=[])
        assert current['identity']==ident
    else:
        assert not any(out.iterdir()),'nonempty output without manifest'
        audit=seed_audit()
        if mode=='production':assert audit['status']=='CLEAR','used seed; preregister before changing seeds'
        put(out/'seed_metadata_audit.json',audit)
        put(manifest,ident,immutable=True)
        put(out/'provenance_start.json',dict(identity=ident,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),python=sys.version,torch=torch.__version__,numpy=np.__version__),immutable=True)
        current=dict(identity=ident,seeds=[],cost=[]);put(state,current)
    assert current['seeds']==seeds[:len(current['seeds'])],'non-prefix resume'
    for seed in current['seeds']:
        rec=json.loads((out/f'seed{seed}/complete.json').read_text());assert rec['identity']==ident;verify_files(out,rec)
    done=0
    for seed in seeds[len(current['seeds']):]:
        rec,cost=seed_run(out,seed,cifar,ident);verify_files(out,rec)
        current['seeds'].append(seed);current['cost'].append(cost);put(state,current);done+=1
        print(f'completed check={mode=="check"} seed {seed}; responses saved',flush=True)
        if (out/'STOP').exists() or (stop_after and done>=stop_after):return dict(status='STOPPED',seeds=current['seeds'])
    marker=dict(identity=ident,seeds=current['seeds'],state_sha256=sha(state),seed_markers={str(s):sha(out/f'seed{s}/complete.json') for s in seeds})
    put(out/'complete.json',marker);put(out/'cost.json',current['cost']);put(out/'provenance_end.json',dict(status='COMPLETE',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return dict(status='COMPLETE',seeds=seeds)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['check','production'],default='check');p.add_argument('--resume',action='store_true');p.add_argument('--production-go',action='store_true');p.add_argument('--checks');a=p.parse_args()
    if a.mode=='production':
        assert a.production_go and a.checks,'separate explicit production GO required'
        from analysis.initgeom_cifar_0920.checks import validate_checks
        validate_checks(json.loads(Path(a.checks).read_text()))
    print(run(a.out,a.mode,a.resume))
if __name__=='__main__':main()
