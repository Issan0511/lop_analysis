"""Serial launcher. STOP blocks starts/resumes; checks and source hashes fail closed."""
from common import *
import fcntl, argparse, time
EXPECTED={
'S-off':{'C_on','labels_shift','adam_time'},
'S-reuse':{a+'_'+m for a in ('C','CH') for m in ('csv_ulp','seed','cond','C_to_H')},
'S-H-route':{'C_on','skip_m1','skip_m2','eval_m0','zero_b1'},
'S-H-EMA':{'beta0','centered','pre_forward','stop_m1','stop_m2'},
'S-grad':{'m_in_P','batchmean_grad'},
'S-resume':{p+str(l) for p in ('missing','skipload','zero') for l in (1,2)}|{'rng','wrong_config'},
'S-graph':{'skip_restore'},
'S-eval':{'no_H','post_update','swap_layers','ratio_medians','skip_m1','skip_m2'},
'S-diverge':{'nan_success','stop_all'},
'S-verdict':{'boundary_ge','median_difference','tie_redundant','drop_missing'},
'S-CLI':{'old_runid','end_git','STOP','old_output'},
'S-collect':{'missing','empty','stale'},
}
def qualified(parts,include_collect=True):
 expected=REQUIRED if include_collect else REQUIRED-{'S-collect'}
 if set(parts)!=expected:return False
 for name,d in parts.items():
  if d.get('name')!=name or not d.get('pass') or not d.get('positive') or d.get('comparisons',0)<=0:return False
  muts=d.get('mutants_detected',{})
  if set(muts)!=EXPECTED[name] or not all(muts.values()):return False
  p=d.get('provenance',{})
  if p.get('prereg_commit')!=PREREG or p.get('dirty_src_analysis'):return False
  sources=p.get('source_sha256',{})
  if not sources or any(not (ROOT/k).is_file() or sha(ROOT/k)!=v for k,v in sources.items()):return False
 return True

def guard(base=BASE,parts=None):
 if (base/'_launch/STOP').exists():raise RuntimeError('STOP file present')
 parts=parts if parts is not None else {n:json.loads((CHECK/(n+'.json')).read_text()) for n in REQUIRED}
 if not qualified(parts):raise RuntimeError('mandatory checks/source qualification incomplete')
 if E.git_state()['dirty_src_analysis']:raise RuntimeError('dirty implementation')
 return parts

def command(arm):
 return [sys.executable,str(ROOT/'src/relu_doors_0919.py'),'run','--arm',arm,'--seeds','0-9','--conds','raw',
 '--tasks','50','--epochs','400','--beta','.01','--run-id',RUN,'--prereg-commit',PREREG,'--out',str(BASE/arm)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');a=ap.parse_args()
 parts=guard()
 if a.validate_only:print('qualified');return
 launch=BASE/'_launch';launch.mkdir(parents=True,exist_ok=True)
 # Shared lock is held for the whole two-arm run, not just process creation.
 lock=open('/tmp/lop_analysis_gpu_launch.lock','a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan={'run_id':RUN,'prereg_commit':PREREG,**E.git_state(),'started_at':datetime.now(timezone.utc).isoformat(),
 'commands':{arm:command(arm) for arm in ('ref','H')},'checks_sha256':{n:sha(CHECK/(n+'.json')) for n in REQUIRED}}
 write(launch/'plan.json',plan)
 for arm in ('ref','H'):
  guard()
  with (launch/(arm+'_raw_seeds0-9.log')).open('a') as log:
   child=subprocess.Popen(command(arm),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(launch/'active.json',{'arm':arm,'pid':child.pid,'started_at':datetime.now(timezone.utc).isoformat()})
   rc=child.wait()
  write(launch/(arm+'_exit.json'),{'arm':arm,'exit_code':rc,'ended_at':datetime.now(timezone.utc).isoformat()})
  if rc:raise SystemExit(rc)
 write(launch/'complete.json',{'status':'FINISHED','ended_at':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':main()
