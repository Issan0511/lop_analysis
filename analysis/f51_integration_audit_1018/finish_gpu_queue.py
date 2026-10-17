#!/usr/bin/env python3
"""Finish runs 3/6 after all earlier explicit GPU queues have drained.

Their original nonblocking admission may refuse a queued recovery job. Resume
only their registered checkpoints; do not repeat a completed scientific run.
"""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback
import continue_gpu as Q

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def main():
    os.environ['SIX_GPU_QUEUE']='1'
    state=dict(status='WAITING_FOR_PRIOR_QUEUES',plan={},stages=[])
    destination=Q.OUT/'completion_state.json'
    with ExitStack() as stack:
        for name in ('completion.lock','controller.lock','state_nap_readout.lock','state_cap_retry.lock'):
            lock=stack.enter_context((Q.OUT/name).open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX)
            if name=='completion.lock':Q.put(destination,state)
        preceding=[load(Q.OUT/name) for name in ('state_after_repairs.json','state_nap_readout.json','state_cap_retry.json')]
        if any(not s or s.get('status')=='RUNNING' for s in preceding):
            state['status']='NEEDS_EXPLICIT_STOP_VERIFICATION';Q.put(destination,state);return 2
        chb=Q.RUNS[4];cnn=Q.RUNS[5]
        state['status']='RUNNING';Q.put(destination,state)
        for run in (chb,cnn):
            wt=Q.BASE/'wt'/run;root=wt/'results'/run
            Q.source_identity(wt)
            if run==chb:
                commands=[] if load(root/'execution_status.json').get('status')=='COMPLETE' else [
                    ('resume',dict(argv=[Q.PY,'-m','analysis.'+run+'.launch','run','--resume'],lock=False))]
                commands.append(('report',dict(argv=[Q.PY,'-m','analysis.'+run+'.launch','report'],lock=False)))
            else:
                commands=[]
                for name,spec in Q.steps(run):
                    if name=='cpu_checks' and load(root/'checks.json').get('all_pass') is True:continue
                    if name=='gpu_checks' and load(root/'gpu_release.json').get('all_pass') is True:continue
                    if name in ('CHf','CHf_B') and load(root/name/'provenance.json').get('status')=='COMPLETE':continue
                    commands.append((name,spec))
            state['plan'][run]=commands;Q.put(destination,state)
            for name,spec in commands:
                stage=dict(run=run,name=name,status='RUNNING',started_at=Q.utc())
                state['stages'].append(stage);state['current']=dict(run=run,stage=name)
                path=Q.OUT/'logs'/f'finish_{run}_{name}.log';stage['log']=str(path);Q.put(destination,state)
                try:
                    with path.open('x',buffering=1) as log:Q.command(wt,spec['argv'],log,spec.get('lock',False))
                    stage.update(status='COMMAND_SUCCEEDED',finished_at=Q.utc())
                except Exception as exc:
                    stage.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
                    Q.put(destination,state);break
                Q.put(destination,state)
        state['status']='NEEDS_ATTENTION' if any(x['status']=='FAILED' for x in state['stages']) else 'COMMANDS_COMPLETED'
        state.pop('current',None);state['finished_at']=Q.utc();Q.put(destination,state)
        return int(state['status']=='NEEDS_ATTENTION')

if __name__=='__main__':sys.exit(main())
