"""Four CPU workers, one GPU worker, attempt-preserving cumulative resource caps."""
import argparse
import concurrent.futures as cf
import json
import math
import fcntl
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
GPU_CAP=10800.
CPU_CAP=14400.


def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def remaining_budget(attempts,external_gpu,external_cpu):
    if not all(math.isfinite(x) and x>=0 for x in (external_gpu,external_cpu)):raise ValueError('External resource accounting must be finite and nonnegative')
    gpu=cpu=0.
    for old in attempts:
        if not old.get('accounting_complete'):
            raise RuntimeError('Prior attempt has incomplete resource accounting; inspect before resume')
        if external_gpu<old['external_gpu_seconds'] or external_cpu<old['external_cpu_seconds']:
            raise ValueError('External usage cannot decrease across attempts')
        for job in old['jobs']:
            if job['device']=='gpu':gpu+=job['elapsed_s']
            else:cpu+=job['elapsed_s']
    return GPU_CAP-external_gpu-gpu,CPU_CAP-external_cpu-cpu



def stop_group(process, grace=5.):
    """Stop all descendants even if the GNU time wrapper exits on SIGTERM."""
    pgid=process.pid
    def send(sig):
        try:os.killpg(pgid,sig);return True
        except ProcessLookupError:return False
    alive=send(signal.SIGTERM);deadline=time.monotonic()+grace
    while alive and time.monotonic()<deadline:
        process.poll()  # Reap wrapper; that alone does not imply group exit.
        time.sleep(min(.05,max(0.,deadline-time.monotonic())))
        alive=send(0)
    forced=alive and send(signal.SIGKILL)
    process.wait()
    return bool(forced)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['all','mlp','vit'],default='all')
    ap.add_argument('--prior-gpu-seconds',type=float,required=True,help='All GPU smoke/diagnostic worker-wall time outside this launcher, conservatively accounted')
    ap.add_argument('--prior-cpu-seconds',type=float,required=True,help='MLP smoke worker-wall time outside this launcher; this is NOT total CPU usage')
    ap.add_argument('--prior-total-cpu-seconds',type=float,required=True,help='Total external user+system CPU usage, or a documented conservative upper bound, including GPU host/compilation')
    ap.add_argument('--gpu-budget-seconds',type=float,help='Optional smaller cap on this attempt, never replenishes the cumulative budget')
    args=ap.parse_args()
    dest=ROOT/'results/readout_separation_1111';dest.mkdir(parents=True,exist_ok=True)
    lock_fd=os.open(dest,os.O_RDONLY)
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_paths=sorted(dest.glob('launch_attempt_*.json'))
    old=[json.loads(p.read_text()) for p in old_paths]
    gpu_left,cpu_left=remaining_budget(old,args.prior_gpu_seconds,args.prior_cpu_seconds)
    if not math.isfinite(args.prior_total_cpu_seconds) or args.prior_total_cpu_seconds<0:raise ValueError('Invalid prior total CPU usage')
    for attempt in old:
        if args.prior_total_cpu_seconds<attempt['external_total_cpu_seconds']:raise ValueError('External total CPU usage cannot decrease')
    measured_cpu_before=sum(j['total_cpu_seconds'] for a in old for j in a['jobs'])
    if measured_cpu_before+args.prior_total_cpu_seconds>=CPU_CAP:raise RuntimeError('Total CPU budget exhausted')
    if args.gpu_budget_seconds is not None:
        if not 0<args.gpu_budget_seconds<=gpu_left:raise ValueError('Requested GPU budget exceeds remaining cumulative cap')
        gpu_left=args.gpu_budget_seconds
    if args.only!='mlp' and gpu_left<=0:raise RuntimeError('GPU budget exhausted')
    if args.only!='vit' and cpu_left<=0:raise RuntimeError('CPU budget exhausted')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())
    logs=dest/'logs'/('attempt_'+stamp);logs.mkdir(parents=True,exist_ok=False)
    manifest=dest/('launch_attempt_'+stamp+'.json')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cpu=[] if args.only=='vit' else [('mlp_ELU_s'+str(seed),['--model','mlp','--seeds',str(seed)]) for seed in range(10)]
    gpu=[] if args.only=='mlp' else [('vit_SL_all',['--model','vit','--seeds','10252010,10252011,10252012,10252013,10252014'])]
    started=time.monotonic();results=[]
    record=dict(complete=False,accounting_complete=False,attempt=stamp,only=args.only,jobs=results,
                external_gpu_seconds=args.prior_gpu_seconds,external_cpu_seconds=args.prior_cpu_seconds,
                external_total_cpu_seconds=args.prior_total_cpu_seconds,total_cpu_seconds_before=measured_cpu_before+args.prior_total_cpu_seconds,
                cpu_accounting='Independent GNU time user+system per subprocess; GPU host and reaped compiler children included. external_total_cpu_seconds is caller-accounted prior usage.',
                remaining_gpu_seconds_before=gpu_left,remaining_mlp_worker_wall_seconds_before=cpu_left,
                previous_attempts=[str(p) for p in old_paths],logs=str(logs),gpu_cap_seconds=GPU_CAP,total_cpu_cap_seconds=CPU_CAP,mlp_worker_wall_cap_seconds=CPU_CAP,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(manifest,record)
    def execute(job,is_gpu):
        name,extra=job
        # Four workers for <=remainingCPU/4 wall seconds bound their sum even
        # before any worker completes. The GPU worker includes the frozen assays.
        deadline=started+(gpu_left if is_gpu else cpu_left/4.)
        remaining=deadline-time.monotonic()-5.  # Reserve timeout termination grace within the wall cap.
        command=[sys.executable,'-m','analysis.readout_separation_1111.run',*extra,'--out',str(dest/'raw')]
        st=time.monotonic();resource_path=logs/(name+'.resources.json')
        result=dict(job=name,device='gpu' if is_gpu else 'cpu',command=command,log=str(logs/(name+'.log')),resource_file=str(resource_path))
        if remaining<=0:
            result.update(returncode=None,timed_out=True,not_started=True,elapsed_s=0.,total_cpu_seconds=0.,cpu_accounting_complete=True)
            return result
        fmt='{"user_s":%U,"system_s":%S,"elapsed_s":%e,"exit_status":%x}'
        timed_command=['/usr/bin/time','--format',fmt,'--output',str(resource_path),*command]
        with (logs/(name+'.log')).open('x') as f:
            try:
                process=subprocess.Popen(timed_command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    returncode=process.wait(timeout=remaining)
                    result.update(returncode=returncode)
                except subprocess.TimeoutExpired:
                    forced=stop_group(process,grace=5.)
                    result.update(returncode=None,timed_out=True,forced_group_kill=forced)
            except Exception as exc:
                result.update(returncode=None,error=repr(exc))
        result['elapsed_s']=time.monotonic()-st
        resource=None
        if resource_path.exists():
            for line in reversed(resource_path.read_text().splitlines()):
                try:
                    candidate=json.loads(line)
                    if all(k in candidate for k in ('user_s','system_s','elapsed_s','exit_status')):
                        resource=candidate;break
                except json.JSONDecodeError:pass
        valid=not result.get('forced_group_kill',False) and resource is not None and all(math.isfinite(resource[k]) and resource[k]>=0 for k in ('user_s','system_s','elapsed_s'))
        result.update(cpu_accounting_complete=valid,resource_usage=resource,
                      total_cpu_seconds=resource['user_s']+resource['system_s'] if valid else None)
        print(json.dumps(result),flush=True)
        return result
    with cf.ThreadPoolExecutor(max_workers=4) as cp,cf.ThreadPoolExecutor(max_workers=1) as gp:
        futures=[cp.submit(execute,j,False) for j in cpu]+[gp.submit(execute,j,True) for j in gpu]
        for f in cf.as_completed(futures):
            # An infrastructure exception leaves accounting_complete false;
            # subsequent launches must stop rather than grant a fresh budget.
            results.append(f.result());record['elapsed_s']=time.monotonic()-started
            atomic_json(manifest,record)
    accounted=all(r.get('cpu_accounting_complete') for r in results)
    total_cpu=sum(r['total_cpu_seconds'] for r in results) if accounted else None
    cumulative_cpu=measured_cpu_before+args.prior_total_cpu_seconds+total_cpu if accounted else None
    record.update(complete=accounted and cumulative_cpu<=CPU_CAP and all(r.get('returncode')==0 for r in results),accounting_complete=accounted,
                  total_cpu_seconds=total_cpu,cumulative_total_cpu_seconds=cumulative_cpu,total_cpu_budget_exceeded=cumulative_cpu>CPU_CAP if accounted else None,
                  elapsed_s=time.monotonic()-started,
                  gpu_worker_seconds=sum(r['elapsed_s'] for r in results if r['device']=='gpu'),
                  mlp_worker_wall_seconds=sum(r['elapsed_s'] for r in results if r['device']=='cpu'))
    atomic_json(manifest,record)
    print('ATTEMPT_MANIFEST',str(manifest),flush=True)
    if not record['complete']:raise SystemExit('Incomplete attempt; preserve partials, use the cumulative ledger for any authorized resume')

if __name__=='__main__':main()
