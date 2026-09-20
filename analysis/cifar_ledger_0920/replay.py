#!/usr/bin/env python3
"""Read-only, original-R CUDA replay of saved CIFAR networks. Never trains."""
from __future__ import annotations
import argparse, contextlib, csv, fcntl, hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
from src import rlcifar_mlp_battle_0918 as E
from analysis.cifar_ledger_0920 import ledger as L

RUN='cifar_ledger_0920'
ARMS=('ELU','GELU','SILU','R','KKT1','LR')
PARAMS=('W1','b1','W2','b2','W3','b3')
SLOTS=[(s,c) for s in range(10) for c in ('raw','std')]
RAW=Path('/home/issan/Projects/obsidian-research-data/rlcifar_mlp_battle_0918/results/rlcifar_mlp_battle_0918')
PARENT=ROOT/'results/rlcifar_mlp_battle_0918'
OUT=ROOT/'results'/RUN
APPROVAL='4aec348'
SPEC=ROOT/'specs/spec_cifar_ledger_0920.md'
DATA=Path('/home/issan/Projects/claude/proj_004_drift/data/cifar10')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def put(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


def read_npz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def output_artifact(out,relative):
    """Resolve an archived output through the committed backup manifest."""
    out=Path(out);relative=Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    direct=out/relative
    if direct.exists():return direct
    manifest=json.loads((out/'backup_manifest.json').read_text())
    wanted=str(Path('results')/out.name/relative)
    matches=[x for x in manifest['files'] if x.get('relative')==wanted]
    assert len(matches)==1,('archived artifact',wanted)
    target=Path(matches[0]['backup'])
    assert target.is_file() and target.stat().st_size==matches[0]['bytes']
    return target


def save_npz(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(p)


def records(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def validate_rows(rr):
    keys=[(int(x['seed']),x['cond'],int(x['task'])) for x in rr]
    expected={(s,c,t) for s,c in SLOTS for t in range(1,51)}
    assert len(keys)==1000 and set(keys)==expected,'CSV completeness'


def existing_shard(p,git_hash,input_sha):
    marker=p.with_suffix('.json')
    if not marker.exists():return None
    old=json.loads(marker.read_text())
    assert p.exists() and sha(p)==old['sha256'],'resume shard mismatch'
    assert old['git_hash']==git_hash and old['input_manifest_sha256']==input_sha,'resume identity mismatch'
    return old


def stopped(out):
    return (out/'STOP').exists()


def setup():
    E.RC.DATA_DIR=DATA
    dev=E.H.setup('cuda')
    torch.set_num_threads(2)
    assert torch.__version__=='2.13.0+cu130',torch.__version__
    assert not torch.backends.cuda.matmul.allow_tf32,'TF32 is outside registered roundoff model'
    assert torch.get_float32_matmul_precision()=='highest'
    return dev


@contextlib.contextmanager
def exclusive():
    with Path('/tmp/lop_analysis_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in p.stdout.splitlines():
            if not line.strip().isdigit():continue
            pid=int(line)
            if pid==os.getpid():continue
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:continue
            assert '--type=gpu-process' in cmd or cmd.startswith('/usr/libexec/gnome-remote-desktop-daemon'),f'GPU busy: {pid} {cmd}'
        yield


def environment():
    return dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip(),
                tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),
                deterministic=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),
                cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                cpu_smallest_subnormal_preserved=bool(torch.tensor(np.nextafter(np.float32(0),np.float32(1))).item()!=0),
                cuda_smallest_subnormal_preserved=bool((torch.tensor(np.nextafter(np.float32(0),np.float32(1)),device='cuda')*1).item()!=0))


def verify_source():
    paths=['src/rlcifar_mlp_battle_0918.py','src/pmnist_0905.py','src/pmnist_rlcifar_0907.py','src/pmnist_rlmnist_0906.py']
    evidence={}
    for ref in ('e7069ac','2fa57c3'):
        d=subprocess.run(['git','diff',ref,'HEAD','--',*paths],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        evidence[ref]=d
    return dict(source_sha256={p:sha(ROOT/p) for p in paths},diffs=evidence)


def validate_inputs():
    manifest=json.loads((PARENT/'backup_manifest.json').read_text())
    lookup={x['backup']:x for x in manifest['files']}
    entries=[];provs={}
    for arm in ARMS:
        prov=json.loads((PARENT/arm/'provenance.json').read_text());provs[arm]=prov
        assert [(x['seed'],x['cond']) for x in prov['slots']]==SLOTS
        assert prov['R']==20 and prov['n_tasks']==50 and prov['device']=='cuda'
        rr=records(PARENT/arm/'per_task.csv')
        validate_rows(rr)
        actual=set((RAW/arm/'snap').glob('*/*.npz'))
        want={E.snapshot_path(RAW/arm,arm,c,s,t) for s,c in SLOTS for t in range(51)}
        assert actual==want,(arm,'snapshot completeness')
        hist={RAW/arm/'hist'/f'{arm}_{c}_seed{s}.npz' for s,c in SLOTS}
        assert hist==set((RAW/arm/'hist').glob('*.npz'))
        for p in sorted(want|hist):
            item=lookup[str(p)]
            assert p.stat().st_size==item['bytes'],p
            digest=sha(p);assert digest==item['sha256'],p
            entries.append(dict(source=str(p),bytes=item['bytes'],sha256=digest,role='snapshot' if p in want else 'hist'))
        for p in [PARENT/arm/'per_task.csv',PARENT/arm/'provenance.json']:
            entries.append(dict(source=str(p),bytes=p.stat().st_size,sha256=sha(p),role=p.name))
    digest=sha(DATA/E.RC.ARCHIVE)
    assert all(p['data_sha256'][E.RC.ARCHIVE]==digest for p in provs.values())
    entries.append(dict(source=str(DATA/E.RC.ARCHIVE),bytes=(DATA/E.RC.ARCHIVE).stat().st_size,sha256=digest,role='dataset'))
    return dict(files=entries,provenances=provs,total_snapshots=6120,total_hist=120,all_pass=True)


def load_state(arm,t,dev):
    ds=[read_npz(E.snapshot_path(RAW/arm,arm,c,s,t)) for s,c in SLOTS]
    shapes=((100,3072),(100,),(100,100),(100,),(10,100),(10,))
    for d in ds:
        for k,shape in zip(PARAMS,shapes):
            assert d[k].shape==shape and d[k].dtype==np.float32 and np.isfinite(d[k]).all(),(arm,t,k)
        if arm=='KKT1':
            for k in ('V1','V2'):assert d[k].shape==(100,) and d[k].dtype==np.float32 and np.isfinite(d[k]).all() and (d[k]>0).all()
        if t:
            for k in ('z1','z2'):assert d[k].shape==(1200,100) and d[k].dtype==np.float16 and np.isfinite(d[k]).all()
    P=[torch.stack([torch.from_numpy(d[k]) for d in ds]).to(dev) for k in PARAMS]
    act=E.make_act(arm);act.init_state(20,dev,'replay')
    if act.adaptive:act.V=[torch.stack([torch.from_numpy(d[k]) for d in ds]).to(dev) for k in ('V1','V2')]
    return ds,P,act


def gtrain(act,z,layer):
    with torch.enable_grad():
        zz=z.detach().clone().requires_grad_(True)
        return torch.autograd.grad(act.phi(zz,layer,train=True).sum(),zz)[0].detach()


def reduced_state(z,h,w,b,g,dg):
    """One slot/layer: CPU float64 accounting on faithful float32 features."""
    z=np.asarray(z,dtype=np.float64);h=np.asarray(h,dtype=np.float64)
    w=w.astype(np.float64);b=b.astype(np.float64)
    mu=L.mean_ball(h,0);m=L.mean_ball(z,0)
    centered=L.Ball(z)-m
    sd=((centered*centered).sum(0)/len(z)).sqrt()
    malg=L.dot(L.Ball(w),mu)+L.Ball(b)
    a=abs(w)@abs(h).mean(0)+abs(b)
    # Direct independent matrix evaluation; the ledger never defines its own reference.
    direct=(h@w.T+b).mean(0)
    direct_bound=L.gamma(2*h.shape[1]+2)*a+L.gamma(len(h)+2)*a+malg.e
    assert np.all(abs(direct-malg.v)<=direct_bound),'direct affine closure'
    replay_bound=L.gamma(2*h.shape[1]+2,np.finfo(np.float32).eps/2)*a+malg.e+m.e
    # CUDA may flush each subnormal product. Its absolute contribution is bounded.
    replay_bound+=(2*h.shape[1]+2)*np.finfo(np.float32).tiny
    assert np.all(abs(m.v-malg.v)<=replay_bound),'float32 affine bound'
    wr=np.linalg.norm(w,axis=1);mr=np.linalg.norm(mu.v)
    with np.errstate(divide='ignore',invalid='ignore'):cos=(w@mu.v)/(wr*mr)
    out=dict(m=m.v,m_error=m.e,sd=sd.v,sd_error=sd.e,m_alg=malg.v,m_alg_error=malg.e,
             zmin=z.min(0),zmax=z.max(0),positive=(z>0).mean(0),w_norm=wr,bias=b,
             mu_norm=np.full(100,mr),cos=cos,replay_residual=m.v-malg.v,replay_bound=replay_bound,
             direct_residual=direct-malg.v,direct_bound=direct_bound)
    for prefix,v in [('train',g),('diag',dg)]:
        out.update({prefix+'_'+k:val for k,val in L.response(v).items()})
    out['derivative_max_difference']=abs(g.astype(np.float64)-dg.astype(np.float64)).max(0)
    out['derivative_zero_disagreement']=((g==0)!=(dg==0)).mean(0)
    return out,dict(w=w,b=b,mu=mu.v,mu_error=mu.e)


def snap_check(arm,t,zs,logits,Y,act,ds,hists,rec):
    evidence=dict(task=t,mean_max_abs_error=0.,mean_max_error_over_bound=0.,z16_exact=True,csv_exact=True,alpha_exact=True)
    if t==0:
        for r,(seed,cond) in enumerate(SLOTS):
            initial=E.H.init_params(seed,torch.device('cpu'),E.DIMS)
            assert all(np.array_equal(ds[r][k],initial[j].detach().numpy()) for j,k in enumerate(PARAMS)),'initial state'
        evidence['initial_exact']=True
        return evidence
    for li,z in enumerate(zs):
        k=li+1;dg=act.dphi(z,li)
        for r,(s,c) in enumerate(SLOTS):
            zr=z[r];m=zr.mean(0).cpu().numpy();ref=hists[r][f'm{k}'][t-1]
            assert hists[r][f'm{k}'].shape==(50,100)
            diff=abs(m.astype(float)-ref.astype(float));scale=max(abs(m).max(),abs(ref).max());bound=2*np.finfo(np.float32).eps*scale
            assert np.all(diff<=bound),(arm,t,r,k,'hist mean',float(diff.max()),float(bound))
            evidence['mean_max_abs_error']=max(evidence['mean_max_abs_error'],float(diff.max()))
            evidence['mean_max_error_over_bound']=max(evidence['mean_max_error_over_bound'],float(diff.max()/bound) if bound else 0.)
            assert np.array_equal(zr.cpu().numpy().astype(np.float16),ds[r][f'z{k}']),(arm,t,r,k,'z16')
            row=rec[(s,c,t)]
            got={f'zbar_l{k}':float(zr.mean(0).median()),f'zsd_l{k}':float(zr.std(0).median()),
                 f'dead_frac_l{k}':float((dg[r].abs().amax(0)<E.DEAD_TOL).float().mean()),
                 f'mob_l{k}':float(dg[r].mean(0).median())}
            if li==0:got['memo_acc']=float((logits[r].argmax(-1)==Y[r]).float().mean())
            assert all(f'{v:.10g}'==f'{float(row[key]):.10g}' for key,v in got.items()),(arm,t,r,k,'csv',got)
            if act.adaptive:
                assert np.array_equal(act.alpha(li)[r].cpu().numpy(),hists[r][f'alpha{k}'][t-1]),'alpha'
    return evidence


def compute(arm,t,X,Xcpu,dev,hists,rec,Y,previous=None):
    start=time.time();ds,P,act=load_state(arm,t,dev)
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    before=[q.clone() for q in P]+([q.clone() for q in act.V] if act.adaptive else [])
    with torch.no_grad():z1,a1,z2,a2,logits=E.forward(P,X,act,False)
    ev=snap_check(arm,t,(z1,z2),logits,Y,act,ds,hists,rec)
    values=[];states=[]
    for li,(z,h) in enumerate(((z1,X),(z2,a1))):
        gt=gtrain(act,z,li)
        with torch.no_grad():
            gd=act.dphi(z,li)
            assert torch.equal(act.phi(z,li,True),act.phi(z,li,False)),'train/eval mode'
        zn=z.cpu().numpy();hn=Xcpu if li==0 else h.cpu().numpy();gn=gt.cpu().numpy();dn=gd.cpu().numpy()
        layer_vals=[];layer_states=[]
        for r in range(20):
            v,st=reduced_state(zn[r],hn[r],ds[r][f'W{li+1}'],ds[r][f'b{li+1}'],gn[r],dn[r])
            layer_vals.append(v);layer_states.append(st)
        values.append(layer_vals);states.append(layer_states)
    shard={k:np.stack([[values[l][r][k] for l in range(2)] for r in range(20)]) for k in values[0][0]}
    shard['mu2']=np.stack([s['mu'] for s in states[1]])
    shard['online']=np.array([float(rec[(s,c,t)]['online_acc']) if t else np.nan for s,c in SLOTS])
    if previous is not None:
        comps=[[L.components(previous[l][r],states[l][r]) for l in range(2)] for r in range(20)]
        for k in comps[0][0]:
            shard['ledger_'+k]=np.stack([[comps[r][l][k].v for l in range(2)] for r in range(20)])
            shard['ledger_'+k+'_error']=np.stack([[comps[r][l][k].e for l in range(2)] for r in range(20)])
        assert np.all(abs(shard['ledger_closure'])<=shard['ledger_closure_error']),'four-term closure'
        assert np.all(shard['ledger_upstream'][:,0]==0) and np.all(shard['ledger_cross'][:,0]==0),'fixed layer1 inputs'
        for base in ('self','upstream'):
            residual=shard['ledger_'+base+'_growth']+shard['ledger_'+base+'_rotation']-shard['ledger_'+base]
            bound=shard['ledger_'+base+'_growth_error']+shard['ledger_'+base+'_rotation_error']+shard['ledger_'+base+'_error']+L.gamma(3)*(abs(shard['ledger_'+base+'_growth'])+abs(shard['ledger_'+base+'_rotation'])+abs(shard['ledger_'+base]))
            valid=np.isfinite(residual)&np.isfinite(bound)
            assert np.all(abs(residual[valid])<=bound[valid]),'direction closure'
        ev['closure_max_abs']=float(abs(shard['ledger_closure']).max())
        ev['closure_max_error_ratio']=float((abs(shard['ledger_closure'])/shard['ledger_closure_error']).max())
    after=P+(act.V if act.adaptive else [])
    assert all(torch.equal(a,b) for a,b in zip(before,after)),'state mutation'
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()),'RNG mutation'
    ev.update(all_pass=True,elapsed_s=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              peak_cuda_bytes=torch.cuda.max_memory_allocated())
    return shard,states,ev


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    setup();dev=torch.device('cuda')
    git=E.git_state();assert not git['dirty_src_analysis'],'commit implementation before replay'
    env=environment();spec_hash=sha(SPEC)
    startfile=out/'provenance.json'
    if startfile.exists():
        prov=json.loads(startfile.read_text())
        assert prov['git_hash']==git['git_hash'] and prov['spec_sha256']==spec_hash and prov['environment']==env,'resume provenance mismatch'
    else:
        prov=dict(run_id=RUN,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),started_epoch=time.time(),
                  **git,approval_commit=APPROVAL,spec_sha256=spec_hash,command=sys.argv,environment=env,source=verify_source(),independent_audit=False)
        put(startfile,prov)
    synthetic=json.loads((ROOT/'results/_checks_cifar_ledger_0920/checks.json').read_text())
    assert synthetic['all_pass']
    for p,digest in synthetic['source_sha256'].items():assert sha(ROOT/p)==digest,('untested source',p)
    inputs=validate_inputs();put(out/'input_manifest.json',inputs)
    print('Input hashes verified: 6120 snapshots, 120 histograms',flush=True)
    cifar=E.RC.Cifar10()
    X=torch.stack([E.slot_inputs(cifar,s,c,dev) for s,c in SLOTS]);Xcpu=X.cpu().numpy()
    ids={str(s):E.RC.subset_idx(s).numpy() for s in range(10)}
    input_ids={s:dict(ordered_sha256=hashlib.sha256(v.tobytes()).hexdigest(),sorted_sha256=hashlib.sha256(np.sort(v).tobytes()).hexdigest()) for s,v in ids.items()}
    assert all(p['subset_sha256'][s]==v['sorted_sha256'] for p in inputs['provenances'].values() for s,v in input_ids.items())
    put(out/'image_order.json',dict(subsets=input_ids,X_sha256=hashlib.sha256(Xcpu.tobytes()).hexdigest(),slots=SLOTS))
    evidence=[]
    for arm in ARMS:
        rec={(int(x['seed']),x['cond'],int(x['task'])):x for x in records(PARENT/arm/'per_task.csv')}
        hists=[read_npz(RAW/arm/'hist'/f'{arm}_{c}_seed{s}.npz') for s,c in SLOTS]
        gens=[E.H.stream('rlc_labels',s) for s,c in SLOTS]
        previous=None
        for t in range(51):
            if stopped(out):
                put(out/'status.json',dict(status='stopped',arm=arm,task=t));return 75
            Y=torch.stack([E.RC.task_labels(g) for g in gens]).to(dev) if t else None
            p=out/'shards'/f'{arm}_t{t:02d}.npz';marker=p.with_suffix('.json')
            old=existing_shard(p,git['git_hash'],sha(out/'input_manifest.json'))
            if old is not None:
                evidence.append(old['checks'])
                # Only the state immediately before an unfinished task is needed.
                nxt=out/'shards'/f'{arm}_t{t+1:02d}.json'
                if t<50 and not nxt.exists():_,previous,_=compute(arm,t,X,Xcpu,dev,hists,rec,Y,None)
                continue
            shard,previous,ev=compute(arm,t,X,Xcpu,dev,hists,rec,Y,previous)
            save_npz(p,shard)
            put(marker,dict(sha256=sha(p),git_hash=git['git_hash'],input_manifest_sha256=sha(out/'input_manifest.json'),checks=ev))
            evidence.append(ev)
            put(out/'status.json',dict(status='running',arm=arm,task=t,completed_states=len(evidence)))
            if t in (0,1,10,50):print(f'{arm} t{t:02d}: replay checks passed; {ev["elapsed_s"]:.2f}s',flush=True)
    put(out/'checks.json',dict(all_pass=all(e['all_pass'] for e in evidence),synthetic=synthetic,states=evidence,
                             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'analysis'/RUN).glob('*.py'))},
                             independent_audit=False))
    put(out/'status.json',dict(status='completed',completed_states=len(evidence),wall_clock_s=time.time()-prov['started_epoch']))
    print('All 306 replay states complete; report may now run.',flush=True)
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args()
    try:
        with exclusive():rc=run(args.out)
    except BaseException as exc:
        put(args.out/'failure.json',dict(error=repr(exc),time=time.time()));raise
    sys.exit(rc)
