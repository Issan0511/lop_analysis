#!/usr/bin/env python3
"""Durable serial execution of the six registered CLIs; no scientific shortcuts.

Each independent run stops at its first failing command. Other runs can continue.
There is no automatic retry/resume of partially completed scientific outputs.
No network publication, cleanup, source edits or performance-based selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

BASE = Path('/home/issan/Projects/claude')
PY = str(BASE / 'proj_004_drift/.venv/bin/python')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/f51_integration_audit_1018/host_continuation'
AUDIT = ROOT / 'results/f51_integration_audit_1018/gpu_host_same_runtime/integration_audit.json'
RUNS = ['redo_position_2x2_1018', 'nap_cifar5p1_1018', 'cap_act_box_2x2_1018',
        'resp_step_controls_1018', 'chb_decompose_200_1018', 'cnn_chprime_bias_1018']

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def put(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)

def command(wt, args, log, gpu_lock=False, check=True):
    if gpu_lock:
        args = ['flock', '-n', '/tmp/lop_analysis_gpu.lock', *args]
    print(utc(), wt.name, args, flush=True)
    log.write('\n' + json.dumps(dict(at=utc(), cwd=str(wt), argv=args)) + '\n')
    log.flush()
    completed = subprocess.run(args, cwd=wt, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'PYTHONUNBUFFERED':'1'})
    if check:
        completed.check_returncode()
    return completed.returncode

def freeze(wt, folder, message, log):
    paths = sorted(str(p.relative_to(wt)) for p in folder.rglob('*')
                   if p.is_file() and p.suffix in ('.json', '.csv', '.md'))
    if not paths:
        raise ValueError('no compact artifacts to freeze')
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=wt):
        raise ValueError('unrelated staged files; refusing mixed commit')
    command(wt, ['git','add','--',*paths], log)
    command(wt, ['git','commit','-m',message], log)

def calibration(wt, log):
    selected = wt / 'results/nap_cifar5p1_1018/calib/selected.json'
    data = json.loads(selected.read_text())
    for name, digest in data['calibration_artifacts_sha256'].items():
        assert sha(wt/name) == digest, 'calibration artifact changed'
    freeze(wt, selected.parent, 'Freeze registered NaP calibration before evaluation', log)

def comparators(wt, log):
    """Qualify old artifacts, or generate full same-runtime fresh comparators."""
    result = wt / 'results/nap_cifar5p1_1018'
    for arm in ('R','SNA','KKT1'):
        q = result / f'reuse_{arm}.json'
        old = wt / f'results/cifar5p1_mlp_0920/{arm}_std_lr0.0001'
        base = [PY,'-m','analysis.nap_cifar5p1_1018.verdict','qualify','--arm',arm,
                '--integration-audit',str(AUDIT)]
        rc = command(wt, [*base,'--directory',str(old),'--out',str(q)],log,check=False)
        if rc == 0:
            continue
        evidence = json.loads(q.read_text())
        assert evidence['errors'] and all(e.startswith('HISTORICAL_EQUIVALENCE_AUDIT_REQUIRED:')
                                         for e in evidence['errors']), evidence['errors']
        shutil.copy2(q, result / f'reuse_{arm}_historical_rejected.json')
        fresh = result / 'current_runtime_comparators' / arm
        if fresh.exists():
            raise ValueError('comparator destination exists; explicit recovery required')
        command(wt,[PY,'src/cifar5p1_mlp_0920.py','run','--arm',arm,'--seeds','0-9',
                    '--cond','std','--tasks','30','--lr','0.0001','--device','cuda',
                    '--threads','2','--out',str(fresh)],log,gpu_lock=True)
        freeze(wt,fresh,f'Fix current-runtime {arm} comparator for registered NaP comparison',log)
        command(wt,[*base,'--directory',str(fresh),'--out',str(q)],log)

def steps(run):
    def mod(module,*args,lock=False): return dict(argv=[PY,'-m',module,*args],lock=lock)
    def file(name,*args,lock=False): return dict(argv=[PY,name,*args],lock=lock)
    audit = ['--integration-audit',str(AUDIT)]
    if run == RUNS[0]:
        return [('run',mod('src.'+run,'run',*audit)),('analyze',mod('src.'+run,'analyze'))]
    if run == RUNS[1]:
        root='results/'+run
        return [('cuda_checks',mod('analysis.'+run+'.checks','--device','cuda',lock=True)),
                ('calibrate',mod('src.'+run,'run','--phase','calib',*audit)),
                ('select',mod('src.'+run,'select')),('freeze_calibration',dict(action='calibration')),
                ('evaluate',mod('src.'+run,'run','--phase','eval',*audit)),
                ('comparators',dict(action='comparators')),
                ('verdict',mod('analysis.'+run+'.verdict','verdict',
                    '--r-qualification',root+'/reuse_R.json','--sna-qualification',root+'/reuse_SNA.json',
                    '--kkt1-qualification',root+'/reuse_KKT1.json'))]
    if run == RUNS[2]:
        return [('cuda_checks',mod('analysis.'+run+'.checks',lock=True)),
                ('preflight',mod('src.'+run,'preflight',*audit)),
                ('calibrate',mod('src.'+run,'calibrate',*audit)),
                ('run',mod('src.'+run,'run',*audit)),('analyze',mod('src.'+run,'analyze'))]
    if run == RUNS[3]:
        items=[('cuda_checks',file('analysis/'+run+'/checks.py','--cuda',lock=True))]
        for box in ('RC','F51'):
            root='results/'+run+'/'+box
            extra=audit if box=='F51' else []
            items += [(box+'_calibrate',mod('src.'+run,'--box',box,'--phase','calibrate',
                         '--out',root+'/calibration',*extra)),
                      (box+'_evaluate',mod('src.'+run,'--box',box,'--phase','evaluate',
                         '--schedule',root+'/calibration/schedule.json','--out',root+'/evaluation',*extra))]
        items.append(('verdict',file(str(Path(__file__).with_name('combined_resp_verdict.py')))))
        return items
    if run == RUNS[4]:
        return [(s,mod('analysis.'+run+'.launch',s)) for s in ('check','plan','run')]
    if run == RUNS[5]:
        return [('cpu_checks',file('analysis/'+run+'/checks.py')),
                ('gpu_checks',file('analysis/'+run+'/gpu_checks.py',lock=True)),
                *[(a,file('src/'+run+'.py','run','--arm',a,'--out','results/'+run+'/'+a,lock=True))
                  for a in ('CHf','CHf_B')],
                ('verdict',file('analysis/'+run+'/verdict.py'))]
    raise ValueError(run)

def source_identity(wt):
    if subprocess.check_output(['git','status','--porcelain','--','src','analysis','specs'],cwd=wt):
        raise ValueError('uncommitted scientific source: '+str(wt))
    return subprocess.check_output(['git','rev-parse','HEAD:src','HEAD:analysis','HEAD:specs'],cwd=wt,text=True)

def verify_audit():
    data=json.loads(AUDIT.read_text())
    for key in ('all_pass','host_reproduction_pass','mutation_checks_pass','main_ancestor_verified'):
        assert data[key] is True, key
    for check in data['checks']:
        assert check['pass_'] is True
        if check['name']=='cuda_main_iv_doors':
            assert set(check['evidence'])=={'main','iv','doors'}
            for e in check['evidence'].values():
                assert e['returncode']==0 and e['all_pass'] is True
                assert sha(e['checks_json'])==e['checks_sha256']
    for name,digest in data['source_sha256'].items():
        assert sha(BASE/'proj_004_drift'/name)==digest, name
    assert data['historical_current_runtime']['all_pass'] is True
    return sha(AUDIT)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--runs',nargs='+',choices=RUNS,default=RUNS)
    parser.add_argument('--state',type=Path,default=OUT/'state.json')
    parser.add_argument('--resume-chb',action='store_true',
                        help='Resume the already validated CHB launch manifest without rewriting its CPU check evidence')
    args=parser.parse_args()
    plan={r:steps(r) for r in args.runs}
    if args.resume_chb and RUNS[4] in plan:
        plan[RUNS[4]]=[('resume',dict(argv=[PY,'-m','analysis.'+RUNS[4]+'.launch','run','--resume'],lock=False))]
    if args.mode=='plan':
        print(json.dumps(plan,indent=2));return 0
    OUT.mkdir(exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():
            raise ValueError('state exists; explicit review and separate state path required')
        audit_sha=verify_audit()
        canonical=Path('/tmp/issa_six_specs_1018/integration_audit.json')
        canonical.parent.mkdir(exist_ok=True)
        shutil.copyfile(AUDIT,canonical)
        identities={r:source_identity(BASE/'wt'/r) for r in args.runs}
        state=dict(status='RUNNING',pid=os.getpid(),started_at=utc(),audit_sha256=audit_sha,
                   source_identity=identities,plan=plan,stages=[])
        put(args.state,state)
        for run, items in plan.items():
            wt=BASE/'wt'/run
            for name, spec in items:
                if (OUT/'STOP').exists():
                    state.update(status='STOPPED_BETWEEN_COMMANDS',finished_at=utc())
                    put(args.state,state);return 75
                stage=dict(run=run,name=name,status='RUNNING',started_at=utc())
                state['stages'].append(stage);state['current']=dict(run=run,stage=name)
                put(args.state,state)
                try:
                    assert source_identity(wt)==identities[run], 'source tree changed while queued'
                    assert sha(AUDIT)==audit_sha, 'G0 changed while queued'
                    logfile=OUT/'logs'/f'{run}_{name}_{os.getpid()}.log'
                    logfile.parent.mkdir(exist_ok=True)
                    stage['log']=str(logfile)
                    put(args.state,state)
                    with logfile.open('x',buffering=1) as log:
                        if spec.get('action')=='calibration':calibration(wt,log)
                        elif spec.get('action')=='comparators':comparators(wt,log)
                        else:command(wt,spec['argv'],log,spec.get('lock',False))
                    stage.update(status='COMMAND_SUCCEEDED',finished_at=utc())
                except Exception as exc:
                    stage.update(status='FAILED',finished_at=utc(),error=repr(exc),traceback=traceback.format_exc())
                    put(args.state,state)
                    print(run,name,'FAILED',repr(exc),flush=True)
                    if isinstance(exc,subprocess.CalledProcessError) and exc.returncode==75:
                        stage['status']='STOPPED';state['status']='STOPPED_AT_CHECKPOINT'
                        put(args.state,state);return 75
                    break
                put(args.state,state)
        state['status']='NEEDS_ATTENTION' if any(s['status']=='FAILED' for s in state['stages']) else 'COMMANDS_COMPLETED'
        state['finished_at']=utc();state.pop('current',None);put(args.state,state)
        return 1 if state['status']=='NEEDS_ATTENTION' else 0

if __name__=='__main__':
    sys.exit(main())
