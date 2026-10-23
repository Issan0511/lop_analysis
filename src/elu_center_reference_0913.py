from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/home/issan/Projects/claude/proj_004_drift/results/p3_extend_0902/logs")
OUT=ROOT/"results/unit_fates_0913/elu_reference"
def write(p,rows):
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True);flux=[];co=[];checkpoints=[];sources=[]
 for seed in range(10):
  p=DATA/f"E_1216_seed{seed}.npz";sources.append(dict(path=str(p),sha256=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()))
  with np.load(p) as z:
   assert str(z['activation'])=='elu' and float(z['act_alpha'])==1. and int(z['task_period'])==10000
   ix=np.searchsorted(z['step'],np.arange(1501)*10000);assert np.array_equal(z['step'][ix],np.arange(1501)*10000)
   mu=z['layer1_zbar'][ix].astype(float);w=z['layer1_w_norm'][ix].astype(float);v=z['layer1_v_unit'][ix].astype(float)
  assert mu.shape==w.shape==v.shape==(1501,100) and all(np.isfinite(a).all() for a in [mu,w,v])
  for eps in [.05,.1,.2]:
   state=np.abs(mu)<=eps;n=state.sum(-1);ins=(~state[:-1]&state[1:]).sum(-1);outs=(state[:-1]&~state[1:]).sum(-1)
   assert np.array_equal(np.diff(n),ins-outs)
   for t in range(1501):flux.append(dict(seed=seed,eps=eps,task=t,n=int(n[t]),inflow=int(ins[t-1]) if t else 0,outflow=int(outs[t-1]) if t else 0))
   for t in [0,1,5,20,100,500,1500]:checkpoints.append(dict(seed=seed,eps=eps,task=t,center_n=int(n[t]),below_minus1=int((mu[t]<-1).sum()),below_minus5=int((mu[t]<-5).sum())))
   for start in [20,100]:
    sel=state[start];count=int(sel.sum())
    for end in [500,1500]:
     co.append(dict(seed=seed,eps=eps,start=start,end=end,n=count,end_inside=int(state[end,sel].sum()),end_below_minus1=int((mu[end,sel]<-1).sum()),end_below_minus5=int((mu[end,sel]<-5).sum()),never_left=int(state[start:end+1,sel].all(0).sum()),mean_full_W_growth=float((w[end,sel]/w[start,sel]).mean()) if count else None,mean_abs_v_start=float(abs(v[start,sel]).mean()) if count else None,mean_abs_v_end=float(abs(v[end,sel]).mean()) if count else None))
 write(OUT/'flux.csv',flux);write(OUT/'checkpoints.csv',checkpoints);write(OUT/'cohorts.csv',co)
 (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',n_logs=10,n_units=1000,task_endpoints=1501,flow_identity='EXACT',source_files=sources,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Available CondA ELU reference; original animation not identified. Full W only, not free W.'),indent=2))
 out=['# 既存CondA ELU参照の中心群（元動画は未特定）','','ELU alpha1・用量12.16・lr.01・各100unit・10seed、中心=平均zが±.1。人数はseed平均。新Snakeとlrが異なる。','', '|task|中心数|mean z < -1|mean z < -5|','|---|---:|---:|---:|']
 for t in [0,1,5,20,100,500,1500]:
  rr=[x for x in checkpoints if x['task']==t and x['eps']==.1]
  out.append(f"|{t}|"+'|'.join(f"{np.mean([x[k] for x in rr]):.2f}" for k in ['center_n','below_minus1','below_minus5'])+'|')
 out+=['','中心個数の減少は終点IDの負側移動と別集計。閾値や区間を変えた値はcsv参照。元の動画との同一性は確認していない。']
 (OUT/'summary.md').write_text('\n'.join(out)+'\n');print('ELU REFERENCE PASS')
if __name__=='__main__':main()
