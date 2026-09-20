#!/usr/bin/env python3
"""Serial, exclusive launch; startup provenance and task-boundary STOP."""
import argparse, contextlib, fcntl, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src import ch_chb_200_0919 as S
LOCK=Path('/tmp/lop_analysis_gpu.lock')

def compute_pids():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[]
    for x in p.stdout.splitlines():
        if not x.strip().isdigit():continue
        pid=int(x)
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:continue
        # Desktop C+G contexts are not competing research jobs.
        if cmd.startswith('/usr/libexec/gnome-remote-desktop-daemon') or '--type=gpu-process' in cmd:continue
        pids.append(pid)
    return pids

def assert_exclusive(pids=None):
    pids=compute_pids() if pids is None else pids
    assert not [p for p in pids if p!=os.getpid()],f'Other GPU compute PIDs: {pids}'

@contextlib.contextmanager
def exclusive():
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_exclusive()
        yield

def output_guard(out):assert out.resolve()==S.OUT.resolve(),'wrong output root'
def log_path(arm,basename=None):
    name=basename or f'{S.RUN}_{arm}.log'
    assert Path(name).name==name and '/' not in name and name not in ('','.','..'),'log basename'
    return S.OUT/'logs'/name

def start_record(arm):
    gs=S.E.git_state();assert gs['git_hash']!='unknown' and not gs['dirty_src_analysis'],'source must be committed'
    parent=json.loads((S.OUT/'inputs'/arm/'provenance.json').read_text())
    return dict(run_id=S.RUN,**gs,approval_commit=S.APPROVAL,time=time.time(),pid=os.getpid(),arm=arm,
        spec_sha256=S.sha(S.SPEC),registration_sha256=S.sha(S.OUT/'registration.json'),
        engine_sha256=S.sha(Path(S.E.__file__)),observer_sha256=S.sha(Path(S.__file__)),
        parent_provenance=parent,parent_checkpoint_sha256=S.EXPECTED[arm],threads=S.torch.get_num_threads(),
        torch=S.torch.__version__,command=sys.argv,independent_audit=False,
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True,check=True).stdout,
        environment={'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})

def arm_run(arm):
    out=S.OUT/arm;ck=out/'ckpt.pt';assert ck.is_file(),'missing continuation checkpoint'
    if (S.OUT/'STOP').exists():return 75
    expected=S.torch.load(ck,map_location='cpu',weights_only=False)
    S.validate_checkpoint(expected,arm,list(range(10)),400,expected['t'])
    assert 50<=expected['t']<=200
    S.validate_history(out,arm,list(range(10)),expected['t'])
    if expected['t']==200:return 0
    if expected['t']==50:assert S.sha(ck)==S.EXPECTED[arm]
    startup=start_record(arm)
    startup['resume_checkpoint_sha256']=S.sha(ck);startup['resume_state_sha256']=S.digest(S.state(expected));startup['resume_task']=expected['t']
    S.put(out/'starts'/f"{time.time_ns()}.json",startup);S.put(out/'provenance.json',dict(startup,status='running'))
    cifar=S.E.RC.Cifar10()
    assert cifar.sha256==startup['parent_provenance']['data_sha256']
    dev=S.E.H.setup('cuda')
    if expected['t']==50 and 'omega_l1' not in S.readcsv(out/'per_task.csv')[0]:S.enrich_prefix(arm,cifar,dev)
    observer=S.Observer(out,arm,expected,list(range(10)),400,stop=S.OUT/'STOP')
    prov=S.E.run(arm,list(range(10)),['raw'],200,400,dev,out,lam=S.LAM[arm],checkpoint=True,resume=True,
        lifecycle=observer,cifar=cifar,progress=lambda _:None)
    completed=S.torch.load(ck,map_location='cpu',weights_only=False)['t']
    S.put(out/'engine_provenance.json',prov)
    S.put(out/'provenance.json',dict(startup,status='completed' if completed==200 else 'stopped',completed_task=completed,
        continuation_engine=prov,wall_clock_s=time.time()-startup['time']))
    S.put(out/'heartbeat.json',dict(status='completed' if completed==200 else 'stopped',task=completed,pid=os.getpid(),time=time.time()))
    return 0 if completed==200 else 75

def launch():
    output_guard(S.OUT)
    reg=json.loads((S.OUT/'registration.json').read_text());assert reg==S.registration()
    checks=json.loads((S.OUT/'checks/checks.json').read_text());assert checks['all_pass']
    hashes=checks['tested_source_sha256']
    for file,h in hashes.items():assert S.sha(S.ROOT/file)==h,f'untested source: {file}'
    S.OUT.joinpath('logs').mkdir(exist_ok=True)
    with exclusive():
        S.put(S.OUT/'launcher.json',dict(pid=os.getpid(),time=time.time(),status='running',poll_interval_s=3600))
        for arm in S.LAM:
            if (S.OUT/'STOP').exists():return 75
            assert_exclusive()
            log=log_path(arm)
            with log.open('a',buffering=1) as f:
                child=subprocess.Popen([sys.executable,__file__,'--arm',arm],cwd=S.ROOT,stdout=f,stderr=subprocess.STDOUT)
                S.put(S.OUT/'launcher.json',dict(pid=os.getpid(),child_pid=child.pid,arm=arm,time=time.time(),status='running',log=str(log)))
                rc=child.wait()
            if rc:
                S.put(S.OUT/'launcher.json',dict(pid=os.getpid(),status='stopped' if rc==75 else 'failed',arm=arm,returncode=rc,time=time.time()))
                return rc
        from analysis.ch_chb_200_0919.report import report
        report(S.OUT)
        S.put(S.OUT/'launcher.json',dict(pid=os.getpid(),status='completed',time=time.time()))
    return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=list(S.LAM));a=ap.parse_args()
    S.torch.set_num_threads(2)
    try:
        rc=arm_run(a.arm) if a.arm else launch()
    except BaseException as exc:
        target=S.OUT/a.arm if a.arm else S.OUT
        S.put(target/'failure.json',dict(type=type(exc).__name__,message=str(exc),time=time.time(),pid=os.getpid()))
        raise
    sys.exit(rc)
