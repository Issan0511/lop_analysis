from pathlib import Path
import hashlib, json, subprocess, sys, os, importlib.util
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch, numpy as np, pandas as pd
from src import relu_doors_0919 as E
RUN='relu_doors_h_ref_0920'
BASE=ROOT/'results'/RUN
CHECK=BASE/'_checks'
PREREG='66285ba79ab8be8c030e797a4379bdf94707baa6'
SEEDS=list(range(200,210))
REQUIRED={'S-off','S-reuse','S-H-route','S-H-EMA','S-grad','S-resume','S-graph','S-eval','S-diverge','S-verdict','S-CLI','S-collect'}
HASHES={'C':'7de7284dc8c292a032f750f94cb9f51a8bf0eb1a982529704bd5e2a7dbbfd1dd','CH':'3259e819ec8ad1d06b3bc861f4daa77121dc2acd8d2646422cde9cbaf52467f1'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,default=str));tmp.replace(p)
def provenance(files):
 return {'run_id':RUN,'prereg_commit':PREREG,'started_at':datetime.now(timezone.utc).isoformat(),**E.git_state(),
 'git_status':subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True).splitlines(),
 'source_sha256':{str(Path(p).relative_to(ROOT)):sha(p) for p in [ROOT/'src/relu_doors_0919.py',Path(__file__),*files]},
 'torch':torch.__version__,'threads':torch.get_num_threads(),'argv':sys.argv}
def part(name,positive,mutants,count,details,prov):
 result={'name':name,'pass':bool(positive and count>0 and mutants and all(mutants.values())),
 'positive':bool(positive),'comparisons':int(count),'mutants_detected':mutants,'details':details,'provenance':prov}
 write(CHECK/(name+'.json'),result);print(name,'PASS' if result['pass'] else 'FAIL',flush=True)
 return result['pass']
def frozen(commit,filename):
 for dep in ('src/pmnist_0905.py','src/pmnist_rlcifar_0907.py'):
  assert subprocess.check_output(['git','show',f'{commit}:{dep}'],cwd=ROOT)==(ROOT/dep).read_bytes(),dep
 code=subprocess.check_output(['git','show',f'{commit}:src/{filename}.py'],cwd=ROOT)
 p=CHECK/'frozen'/f'{filename}_{commit}.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(code)
 return module(p)
def module(p):
 sp=importlib.util.spec_from_file_location('test_'+sha(p)[:12],p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def mutated(name,replacements):
 code=(ROOT/'src/relu_doors_0919.py').read_text()
 for before,after in replacements:
  assert code.count(before)==1,(name,before,code.count(before));code=code.replace(before,after)
 p=CHECK/'mutants'/(name+'.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(code)
 write(p.with_suffix('.json'),{'name':name,'source_sha256':sha(ROOT/'src/relu_doors_0919.py'),'mutant_sha256':sha(p),'replacements':replacements,'count':len(replacements)})
 return module(p)
def df(p): return pd.read_csv(Path(p)/'per_task.csv',float_precision='round_trip').sort_values(['seed','task']).reset_index(drop=True)
def equal_rows(a,b):
 cols=[c for c in a.select_dtypes(include='number').columns if c in b and c!='slot']
 if len(a)!=len(b) or not len(a) or not cols:return False,0
 x,y=a[cols].to_numpy(float),b[cols].to_numpy(float)
 return bool(np.array_equal(x,y,equal_nan=True)),x.size
def equal_tree(x,y):
 if torch.is_tensor(x):return torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y)
 if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(equal_tree(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return isinstance(y,(list,tuple)) and len(x)==len(y) and all(equal_tree(a,b) for a,b in zip(x,y))
 return x==y
STATE_KEYS=('P','m','v','tc','g_lab','g_batch','alive')
def state(p):return torch.load(Path(p)/'ckpt.pt',map_location='cpu',weights_only=False)
def equal_state(a,b,act=True):
 keys=STATE_KEYS+(('act_state',) if act else ())
 return all(equal_tree(a[k],b[k]) for k in keys)
def run(mod,arm,tag,*,epochs=2,tasks=2,seeds=None,**kw):
 dest=CHECK/'runs'/tag
 if not kw.get('resume'):assert not dest.exists(),str(dest)
 mod.run(arm,SEEDS if seeds is None else seeds,['raw'],tasks,epochs,torch.device('cuda'),dest,
  cifar=CIFAR,checkpoint=True,progress=lambda msg:None,**kw)
 torch.cuda.empty_cache()
 return dest
CIFAR=None
