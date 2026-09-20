"""Independent final cross-check using CSV+Decimal, without importing the report/engine."""
import csv,json,statistics,hashlib
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/relu_doors_h_ref_0920'
D=Decimal
u64=D(2)**-53
gamma=lambda n:(D(n)*u64)/(1-D(n)*u64)
max_error=D(0)
v=json.loads((BASE/'verdict.json').read_text());manifest=json.loads((BASE/'reuse_manifest.json').read_text())
windows={};tables={}
for arm,info in manifest['sources'].items():
 path=ROOT/info['path']/'per_task.csv'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==info['csv_sha256']
 rows=list(csv.DictReader(path.open()));assert len(rows)==500
 keys={(int(r['seed']),int(r['task'])) for r in rows}
 assert keys=={(s,t) for s in range(10) for t in range(1,51)}
 assert all(r['cond']=='raw' for r in rows)
 tables[arm]=rows;windows[arm]=[]
 for seed in range(10):
  vals=[D(r['online_acc']) for r in rows if int(r['seed'])==seed and 31<=int(r['task'])<=50]
  assert len(vals)==20 and all(x.is_finite() for x in vals)
  w=sum(vals)/20;windows[arm].append(w)
  # 20-term float64 mean: summation and division bounded by gamma20.
  err=abs(D.from_float(v['windows'][arm][seed])-w);max_error=max(max_error,err)
  assert err<=gamma(20)*max(abs(x) for x in vals)
 err=abs(D.from_float(v['medians'][arm])-statistics.median(windows[arm]));max_error=max(max_error,err)
 # Median is nonexpansive in sup norm, followed by two operations for the even-n midpoint.
 assert err<=gamma(22)*max(abs(D(r['online_acc'])) for r in rows)
k={a:sum(x>=D('.5') for x in vals) for a,vals in windows.items()};assert k==v['K']
diffs=sorted(h-c for h,c in zip(windows['H'],windows['CH']))
assert diffs[8]<-D('0.0007606370276951454')
theta=D('0.49249997734999995')
def count(arm,task,key,fn):return sum(fn(D(r[key])) for r in tables[arm] if int(r['task'])==task)
layer={
 'H_l1_dies':count('H',1,'dead_frac_l1',lambda x:x>theta)>=9,
 'C_l1_saved':count('C',1,'dead_frac_l1',lambda x:x<theta)>=9,
 'C_l2_dies':count('C',50,'gate_zero_frac_l2',lambda x:x==1)>=9,
 'CH_l2_not_all_zero':count('CH',50,'gate_zero_frac_l2',lambda x:x<1)>=9}
assert layer==v['layer_conditions'] and all(layer.values())
assert k['ref']==k['H']==k['C']==0 and k['CH']==10
assert v['labels']['M_H']=='H_ALONE_FAILS' and v['labels']['N_C']=='C_NEEDED' and v['labels']['R_LAYER']=='LAYER_CORRESPONDENCE'
result={'pass':True,'method':'independent csv.DictReader + Decimal; float64 mean gamma20 and midpoint gamma22 bound; labels compared exactly','max_float64_vs_decimal_error':str(max_error),
 'rows_per_arm':500,'windows_per_arm':10,'rescued_seeds':k,'layer_conditions':layer,'independent_audit':False,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
