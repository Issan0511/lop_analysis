"""Cumulative, process-group-accounted execution of the registered study."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, save_json, sha

GPU_CAP = 4 * 3600.
CPU_CAP = 8 * 3600.
DEST = ROOT / 'results/elu_specificity_1112'


def process_group_cpu(groups):
    """Own and reaped-descendant CPU, without counting a live child twice."""
    values = {g:0. for g in groups}
    ticks = os.sysconf('SC_CLK_TCK')
    for item in Path('/proc').iterdir():
        if not item.name.isdigit(): continue
        try:
            tail = (item/'stat').read_text().rsplit(') ',1)[1].split()
            pgid = int(tail[2])
            if pgid in values:
                values[pgid] += sum(int(tail[i]) for i in (11,12,13,14)) / ticks
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):
            continue
    return values


def stop_all(active):
    for job in active:
        try: os.killpg(job['process'].pid, signal.SIGTERM)
        except ProcessLookupError: pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and any(j['process'].poll() is None for j in active):
        time.sleep(.05)
    for job in active:
        # A dead GNU time wrapper does not guarantee that its descendants exited.
        try: os.killpg(job['process'].pid, signal.SIGKILL)
        except ProcessLookupError: pass
        job['process'].wait()


def usage(path):
    if not path.exists(): return None
    for line in reversed(path.read_text().splitlines()):
        try:
            value=json.loads(line)
            if all(k in value for k in ('user_s','system_s','wall_s','exit_status')):
                if all(math.isfinite(value[k]) and value[k]>=0 for k in ('user_s','system_s','wall_s')): return value
        except (json.JSONDecodeError,TypeError,ValueError): pass
    return None


def plan(only, smoke, out, vit_engine="eager"):
    jobs=[]
    if only!='vit':
        for seed in ([0] if smoke else range(10)):
            for act in ('ELU','R','LR','SILU'):
                for ln in ('off','on'):
                    name=f'mlp_{act}_{ln}_s{seed}'
                    args=['--seed',str(seed),'--activation',act,'--ln',ln,'--out',str(out),'--steps','26' if smoke else '4000']
                    jobs.append(dict(name=name,device='cpu',module='analysis.elu_specificity_1112.mlp',args=args))
    if only!='mlp':
        for seed in ([10252010] if smoke else range(10252010,10252015)):
            for source in ('S','SL'):
                for act in ('R','ELU'):
                    name=f'vit_{source}_{act}_s{seed}'
                    args=['--seed',str(seed),'--source-arm',source,'--activation',act,'--out',str(out),'--steps','26' if smoke else '1500','--engine',vit_engine]
                    jobs.append(dict(name=name,device='gpu',module='analysis.elu_specificity_1112.vit',args=args))
    return jobs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--only',choices=['all','mlp','vit'],default='all')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--external-cpu-seconds',type=float,required=True)
    ap.add_argument('--external-gpu-seconds',type=float,required=True)
    ap.add_argument('--cpu-workers',type=int,default=4)
    ap.add_argument('--vit-engine',choices=['eager','compile'],default='eager')
    a=ap.parse_args()
    assert 1<=a.cpu_workers<=4
    assert all(math.isfinite(v) and v>=0 for v in (a.external_cpu_seconds,a.external_gpu_seconds))
    DEST.mkdir(parents=True,exist_ok=True)
    lock=os.open(DEST,os.O_RDONLY);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oldpaths=sorted(DEST.glob('launch_attempt_*.json'))
    old=[json.loads(p.read_text()) for p in oldpaths]
    if any(not d['accounting_complete'] for d in old): raise RuntimeError('Incomplete prior accounting; reconcile before another attempt')
    if any(a.external_cpu_seconds<d['external_cpu_seconds'] or a.external_gpu_seconds<d['external_gpu_seconds'] for d in old):
        raise ValueError('External accounted use cannot decrease')
    prior_cpu=a.external_cpu_seconds+sum(d['measured_cpu_seconds'] for d in old)
    prior_gpu=a.external_gpu_seconds+sum(d['gpu_worker_seconds'] for d in old)
    if prior_cpu>=CPU_CAP or (a.only!='mlp' and prior_gpu>=GPU_CAP): raise RuntimeError('Cumulative budget exhausted')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())
    logs=DEST/'logs'/stamp;logs.mkdir(parents=True,exist_ok=False)
    out=DEST/('smoke' if a.smoke else 'raw')/stamp;out.mkdir(parents=True,exist_ok=False)
    pending=plan(a.only,a.smoke,out,a.vit_engine)
    manifest=DEST/f'launch_attempt_{stamp}.json'
    record=dict(complete=False,accounting_complete=False,attempt=stamp,smoke=a.smoke,only=a.only,
        vit_engine=a.vit_engine,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),output_root=str(out),logs=str(logs),
        execution_git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        launcher_sha256=sha(__file__),external_cpu_seconds=a.external_cpu_seconds,external_gpu_seconds=a.external_gpu_seconds,
        prior_cpu_seconds=prior_cpu,prior_gpu_seconds=prior_gpu,cpu_cap_seconds=CPU_CAP,gpu_cap_seconds=GPU_CAP,
        cpu_accounting='GNU time user+system per subprocess incl reaped children; watchdog sums own and reaped-child /proc CPU in managed groups. External use may be a documented upper bound.',
        planned_jobs=[j['name'] for j in pending],jobs=[],previous_attempts=[str(p) for p in oldpaths])
    save_json(manifest,record)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    active=[];started=time.monotonic();stop_reason=None
    def completed(job, forced=False):
        job['handle'].close();res=usage(job['resources']);cpu=None if res is None else res['user_s']+res['system_s']
        value={k:v for k,v in job.items() if k not in ('handle','process','started','resources')}
        value.update(returncode=job['process'].returncode,elapsed_s=time.monotonic()-job['started'],
            resources=str(job['resources']),resource_usage=res,total_cpu_seconds=cpu,
            accounting_complete=res is not None and not forced,forced_stop=forced)
        record['jobs'].append(value)
        print(json.dumps({'job':value['name'],'returncode':value['returncode'],'elapsed_s':value['elapsed_s'],'cpu_seconds':cpu}),flush=True)
        return value
    try:
        while pending or active:
            for job in list(active):
                if job['process'].poll() is not None:
                    value=completed(job);active.remove(job)
                    if value['returncode']!=0 or not value['accounting_complete']:
                        stop_reason='job failure or incomplete accounting: '+value['name']
            measured=sum(j['total_cpu_seconds'] or 0. for j in record['jobs'])
            gpu=sum(j['elapsed_s'] for j in record['jobs'] if j['device']=='gpu')
            current=process_group_cpu({j['process'].pid for j in active})
            cpu_use=prior_cpu+measured+sum(current.values())
            gpu_use=prior_gpu+gpu+sum(time.monotonic()-j['started'] for j in active if j['device']=='gpu')
            record.update(elapsed_s=time.monotonic()-started,observed_cumulative_cpu_seconds=cpu_use,
                observed_cumulative_gpu_seconds=gpu_use,active_jobs=[j['name'] for j in active])
            capped = cpu_use>=CPU_CAP-15 or gpu_use>=GPU_CAP-10
            if capped:
                stop_reason='cumulative resource cap'
            if capped:
                record['stop_reason']=stop_reason;record['not_started']=[j['name'] for j in pending]
                pending.clear();stop_all(active)
                for job in active:completed(job,forced=True)
                active.clear();break
            if stop_reason:
                record['stop_reason']=stop_reason
                if pending:
                    record['not_started']=[j['name'] for j in pending]
                    pending.clear()
                save_json(manifest,record)
                if active:time.sleep(2)
                continue
            counts={d:sum(j['device']==d for j in active) for d in ('cpu','gpu')}
            for job in list(pending):
                cap=a.cpu_workers if job['device']=='cpu' else 1
                if counts[job['device']]>=cap: continue
                log=logs/(job['name']+'.log');resources=logs/(job['name']+'.resources.json')
                command=[sys.executable,'-m',job['module'],*job['args']]
                timed=['/usr/bin/time','--format','{"user_s":%U,"system_s":%S,"wall_s":%e,"exit_status":%x}','--output',str(resources),*command]
                handle=log.open('x');st=time.monotonic()
                try: process=subprocess.Popen(timed,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:handle.close();raise
                active.append(dict(name=job['name'],device=job['device'],command=command,log=str(log),
                    resources=resources,handle=handle,process=process,started=st))
                pending.remove(job);counts[job['device']]+=1
            save_json(manifest,record)
            if pending or active:time.sleep(2)
    except BaseException:
        stop_all(active)
        for job in active:completed(job,forced=True)
        record['stop_reason']='launcher exception';save_json(manifest,record);raise
    accounted=all(j['accounting_complete'] for j in record['jobs'])
    cpu=sum(j['total_cpu_seconds'] for j in record['jobs']) if accounted else None
    gpu=sum(j['elapsed_s'] for j in record['jobs'] if j['device']=='gpu')
    record.update(accounting_complete=accounted,measured_cpu_seconds=cpu,gpu_worker_seconds=gpu,
        complete=not stop_reason and accounted and len(record['jobs'])==len(record['planned_jobs']) and all(j['returncode']==0 for j in record['jobs']),
        cumulative_cpu_seconds=prior_cpu+cpu if accounted else None,cumulative_gpu_seconds=prior_gpu+gpu,
        elapsed_s=time.monotonic()-started,active_jobs=[])
    save_json(manifest,record);print('ATTEMPT_MANIFEST',manifest,flush=True)
    if not record['complete']:raise SystemExit('Incomplete attempt; preserve outputs and reconcile before resume')

if __name__=='__main__':main()
