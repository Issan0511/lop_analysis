"""Early replay and unchanged paired transport instrumentation."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from src import boundary_early_source_0908 as E
from src import boundary_transport_0908 as T
ROOT=E.ROOT
OUT=ROOT/'results/boundary_early_0908'
def compare(a,b,path=''):
 if torch.is_tensor(a):
  assert torch.equal(a,b),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 else:assert a==b,(path,a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--arm');a=ap.parse_args()
 torch.set_num_threads(1);E.H.setup('cpu');mnist=E.H.Mnist(torch.device('cpu'))
 E.OUT.mkdir(parents=True,exist_ok=True);T.OUT=OUT/'transport';T.OUT.mkdir(parents=True,exist_ok=True)
 if a.preflight:E.preflight(mnist);return
 oldsource=T.G.SOURCE;T.G.SOURCE=E.OUT
 for iv in ['none','l2']:
  for seed in range(3):
   prefix=f'{a.arm}_{iv}_s{seed}'
   assert not (E.OUT/(prefix+'_provenance.json')).exists()
   E.run(a.arm,seed,iv,mnist)
   old=torch.load(oldsource/(prefix+'_task1.pt'),weights_only=False,map_location='cpu')
   new=torch.load(E.OUT/(prefix+'_task1.pt'),weights_only=False,map_location='cpu')
   compare(old,new)
   print('TASK1 EXACT',prefix,flush=True)
   T.run(a.arm,iv,seed,mnist)
   # Restore early-task reference scope and record both the reused instrumentation and this spec.
   f=T.OUT/(prefix+'_provenance.json');meta=json.loads(f.read_text())
   meta.update(early_spec_sha256=T.G.sha(ROOT/'specs/spec_boundary_early_0908.md'),
               wrapper_sha256=T.G.sha(Path(__file__)),task1_original_CPU_exact=True,
               task_range=[2,20],source_generator_sha256=T.G.sha(Path(E.__file__)))
   f.write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
