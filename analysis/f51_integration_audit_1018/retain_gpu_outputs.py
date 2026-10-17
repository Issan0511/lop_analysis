#!/usr/bin/env python3
"""After the serial controller exits, retain raw outputs and commit compact evidence.

Waits on the controller's lock (no polling). Never moves live data, deletes a
worktree, publishes remotely, changes a scientific verdict, or reruns training.
"""
from datetime import datetime, timezone
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'results/f51_integration_audit_1018/host_continuation'
BASE=Path('/home/issan/Projects/claude/wt')
BACKUP=Path('/home/issan/Projects/obsidian-research-data')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def put(path,data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    report=dict(status='WAITING_FOR_CONTROLLER',started_at=datetime.now(timezone.utc).isoformat(),runs={})
    put(STATE/'retention_status.json',report)
    with ExitStack() as stack:
        for name in ('completion.lock','controller.lock','state_nap_readout.lock','state_cap_retry.lock'):
            lock=stack.enter_context((STATE/name).open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX)
        states=[json.loads((STATE/name).read_text()) for name in
                ('state_after_repairs.json','state_nap_readout.json','state_cap_retry.json','completion_state.json')]
        state=states[0]
        # A killed controller with a RUNNING state may still have a child. Do not
        # snapshot files whose completion/stop was not acknowledged by the owner.
        if any(s['status']=='RUNNING' for s in states):
            report['status']='NEEDS_EXPLICIT_STOP_VERIFICATION';put(STATE/'retention_status.json',report);return 2
        report['controller_statuses']=[s['status'] for s in states];report['status']='RETAINING'
        put(STATE/'retention_status.json',report)
        for run in state['plan']:
            wt=BASE/run;folder=wt/'results'/run
            try:
                policy_path=STATE/'run_policy.json'
                paused=(json.loads(policy_path.read_text()).get('paused_runs',{}).get(run)
                        if policy_path.exists() else None)
                files=[p for p in folder.rglob('*') if p.is_file() and p.name!='backup_manifest.json']
                needed=sum(p.stat().st_size for p in files)
                if shutil.disk_usage(BACKUP).free < needed + 10*2**30:
                    raise RuntimeError(f'insufficient backup space: need {needed} bytes plus 10 GiB reserve')
                destination=BACKUP/run/(paused['retention_snapshot'] if paused else 'host_gpu_1018')
                entries=[]
                for source in files:
                    target=destination/source.relative_to(wt);target.parent.mkdir(parents=True,exist_ok=True)
                    digest=sha(source)
                    if target.exists():
                        if sha(target)!=digest:raise ValueError('refusing to overwrite a different backup: '+str(target))
                    else:
                        temporary=target.with_name(target.name+'.copying')
                        shutil.copy2(source,temporary)
                        if sha(temporary)!=digest:raise ValueError('backup digest mismatch')
                        temporary.replace(target)
                    entries.append(dict(source=str(source),backup=str(target),bytes=source.stat().st_size,sha256=digest))
                put(folder/'backup_manifest.json',entries)
                if paused:
                    # The checkpoint manifest binds HEAD. Preserve it until the
                    # user resumes or closes this run; back up without a commit.
                    report['runs'][run]=dict(backup_verified=True,files=len(entries),bytes=needed,
                        status='PAUSED_BY_USER',commit_deferred='Preserve checkpoint-bound HEAD',
                        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip())
                    put(STATE/'retention_status.json',report);continue
                selected=[str(p.relative_to(wt)) for p in folder.rglob('*') if p.is_file()
                          and p.suffix in ('.json','.csv','.md','.txt') and p.stat().st_size < 10*2**20]
                if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=wt):
                    raise ValueError('unrelated staged files; raw backup complete but commit deferred')
                subprocess.run(['git','add','--',*selected],cwd=wt,check=True)
                if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode:
                    subprocess.run(['git','commit','-m',f'Retain GPU execution artifacts and raw hashes for {run}'],cwd=wt,check=True)
                stages=[s for controller in states for s in controller['stages'] if s['run']==run]
                report['runs'][run]=dict(backup_verified=True,files=len(entries),bytes=needed,
                    scientific_execution_stages=stages,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip())
            except Exception as exc:
                report['runs'][run]=dict(error=repr(exc))
            put(STATE/'retention_status.json',report)
        report['status']='NEEDS_ATTENTION' if any('error' in v for v in report['runs'].values()) else 'RETAINED_LOCALLY'
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        put(STATE/'retention_status.json',report)
        return 1 if report['status']=='NEEDS_ATTENTION' else 0

if __name__=='__main__':sys.exit(main())
