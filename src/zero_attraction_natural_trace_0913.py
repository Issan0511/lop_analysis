"""Reconstruct input RNG and replay registered task21/task101 SGD ledgers."""
from pathlib import Path
import copy,csv,hashlib,json,time
import numpy as np
import torch
from src import edge_law_0905 as e
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/home/issan/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913")
OUT=ROOT/"results/zero_attraction_learning_0913/natural_trace"
NAMES=["SN_normal_q0","SN_peak_q0","SN_valley_q0","LR_a0p1_q0","LR_a0p1_qm05","LR_a0p1_qp05","LIN_q0"]
MILESTONES={1,20,200,1000,10000}
def arr(x):return x.detach().cpu().numpy()
def check(key,a,b,checks,atol=1e-7,rtol=1e-8):
 error=float((a-b).abs().max());checks[key]=max(checks.get(key,0.),error)
 torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
def stats(st):
 raw=e.full_support_ro(st["env"]).double()
 x=raw-(st["layer_means"][0].double()[None] if st["centered_layers"][0] else 0)
 z=torch.einsum("rhd,prd->prh",st["net"].W.double(),x)+st["net"].b.double()
 return z.mean(0),z.var(0,unbiased=False)
def restored(base,cp):
 st=copy.deepcopy(base)
 assert st["env"].t==cp["env"]["t"]
 assert torch.equal(st["env"].flip_state,cp["env"]["flip_state"])
 for k,v in cp["teacher"].items():assert torch.equal(getattr(st["teacher"],k),v),k
 for k,v in cp["net"].items():st["net"].params()[k].copy_(v)
 st["net"].set_activation(cp["activation"],cp["act_alpha"],"alpha_exp")
 st["activation"]=cp["activation"];st["act_alpha"]=cp["act_alpha"];st["arm"]=cp["arm"]
 st["running_mean"]=cp["running_mean"].clone()
 st["layer_means"]=[v.clone() if v is not None else None for v in cp["layer_means"]]
 st["centered_layers"]=cp["centered_layers"][:];st["runs"]=copy.deepcopy(cp["runs"])
 return st
def trace(st,cp):
 net=st["net"];eta=float(cp["runs"][0]["lr"]);assert eta==.005
 u0=net.W[:,:,15:].double().clone();muold,varold=stats(st)
 Is=torch.zeros_like(net.v,dtype=torch.float64);Ir=Is.clone();Iq=Is.clone();K=Is.clone();Q=Is.clone()
 disp=torch.zeros_like(u0);checks={};records=[];raw={}
 jump=None
 for step in range(1,10001):
  u=net.W[:,:,15:].double().clone()
  x=st["env"].step();y=st["teacher"](x)
  inputs,pres,acts,yh=e.forward_gate(st,x)
  if step==1:
   startmean,startvar=stats(st);jump=startmean-muold
  grads=e.grads_centered_elu(net,inputs,pres,acts,yh-y)
  v=net.v.double();phi=acts[0].double();gate=net.act_grad(pres[0],acts[0]).double()
  delta=(yh-y).double();xf=inputs[0][:,15:].double()
  ds=-eta*(2*v*v*phi*gate)[:,:,None]*xf[:,None,:]
  dr=-eta*(2*v*(delta[:,None]-v*phi)*gate)[:,:,None]*xf[:,None,:]
  net.sgd_step_layers(st["lr"],*grads)
  unew=net.W[:,:,15:].double();du=unew-u;dq=du-ds-dr
  Is-=(u*ds).sum(-1);Ir-=(u*dr).sum(-1);Iq-=(u*dq).sum(-1)
  K+=(disp*du).sum(-1);Q+=du.square().sum(-1);disp+=du
  if step in MILESTONES:
   G=unew.square().sum(-1)-u0.square().sum(-1);D2=disp.square().sum(-1)
   cND=-(u0*disp).sum(-1);I=Is+Ir+Iq
   check("G_ledger",G,Q-2*I,checks);check("D2_path",D2,Q+2*K,checks);check("cND_path",cND,I+K,checks)
   check("displacement",disp,unew-u0,checks)
   n0=u0.norm(dim=-1);D=D2.sqrt();ND=n0*D
   valid=(n0>1e-12)&(D>1e-12);c=torch.full_like(ND,float("nan"));c[valid]=cND[valid]/ND[valid]
   mean,var=stats(st)
   values=dict(N=n0,D=D,c=c,cND=cND,I_self=Is,I_rest=Ir,I_round=Iq,K=K,Q=Q,G=G,
    R_self=-2*Is,R_rest=-2*Ir,R_round=-2*Iq,mean_boundary_jump=jump,mean_parameter_move=mean-startmean,
    variance_initial=varold,variance_after_boundary=startvar,variance_current=var)
   raw[str(step)]={k:arr(vv).copy() for k,vv in values.items()}
   for ri,run in enumerate(cp["runs"]):
    for ui in range(net.h):
     row={"arm":cp["arm"],"task":cp["step"]//10000+1,"seed":run["seed"],"unit":ui,"updates":step}
     row.update({k:float(vv[ri,ui]) for k,vv in values.items()});records.append(row)
 # Compare final replay endpoint against the originally trained trajectory.
 exact=True
 for ri,run in enumerate(cp["runs"]):
  with np.load(DATA/"logs"/f"{cp['arm']}_seed{run['seed']}.npz") as z:
   target=cp["step"]+10000
   ix=np.flatnonzero(z["layer1_w_free_step"]==target);assert len(ix)==1
   saved=torch.as_tensor(z["layer1_w_free"][int(ix[0])],dtype=torch.float64)
   exact=exact and np.array_equal(arr(net.W[ri,:,15:]),z["layer1_w_free"][int(ix[0])])
   check("replay_free",net.W[ri,:,15:].double(),saved,checks,atol=1e-6,rtol=1e-6)
   k=int(np.flatnonzero(z["step"]==target)[0])
   for key,val in [("layer1_v_unit",net.v[ri].double()),("layer1_w_norm",net.W[ri].double().norm(dim=-1)),("layer1_zbar",stats(st)[0][ri])]:
    check("replay_"+key,val,torch.as_tensor(z[key][k],dtype=torch.float64),checks,atol=5e-5,rtol=1e-5)
 return records,checks,exact,raw
def writecsv(path,rows):
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 e.CONFIG=ROOT/"configs/zero_attraction_learning_0913.yaml";e._TABLE=None
 cfg=e.build_cfg();base=e.setup_arm_dial(cfg,e._arm(cfg,"SN_normal_q0"),"cpu")
 allchecks={};allrows=[];started=time.monotonic()
 for start in [200000,1000000]:
  while base["env"].t<start:base["env"].step()
  print("RNG reconstructed",start,flush=True)
  for name in NAMES:
   path=DATA/"ckpts"/f"{name}_step{start}.pt";cp=torch.load(path,map_location="cpu",weights_only=True)
   st=restored(base,cp);rows,checks,exact,raw=trace(st,cp)
   allrows.extend(rows);allchecks[f"{name}_{start}"]={"max_abs_errors":checks,"free_weights_byte_exact":exact}
   np.savez_compressed(OUT/f"{name}_task{start//10000+1}.npz",**{f"{s}_{k}":v for s,vs in raw.items() for k,v in vs.items()})
   print("TRACE PASS",name,start,"byte_exact",exact,flush=True)
 writecsv(OUT/"unit_ledger.csv",allrows)
 summary=[]
 for name in NAMES:
  for task in [21,101]:
   for updates in sorted(MILESTONES):
    for seed in range(10):
     rr=[r for r in allrows if r["arm"]==name and r["task"]==task and r["updates"]==updates and r["seed"]==seed]
     assert len(rr)==100
     sr={"arm":name,"task":task,"updates":updates,"seed":seed,"n_units":100}
     for k in rr[0]:
      if k not in sr and k not in ["unit"]:sr[k]=float(np.mean([r[k] for r in rr])) if k!="c" else float(np.nanmean([r[k] for r in rr]))
     sr["undefined_c_count"]=int(sum(not np.isfinite(r["c"]) for r in rr));summary.append(sr)
 writecsv(OUT/"seed_ledger.csv",summary)
 (OUT/"verification.json").write_text(json.dumps({"status":"PASS","checks":allchecks,"elapsed_seconds":time.monotonic()-started},indent=2))
 out=["# 自己項からtask侵食角まで：追加解析 0913","","task21/101、各10000更新の自然軌道再生。全unit・10seedの平均。","cND=I_self+I_rest+I_round+K。G=Q-2(I_self+I_rest+I_round)。","大きなself/restが相殺する場合、各項の大きさだけで原因を決めない。","",
 "|arm|task|I_self|I_rest|I_round|K|cND|Q|G|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for name in NAMES:
  for task in [21,101]:
   rr=[r for r in summary if r["arm"]==name and r["task"]==task and r["updates"]==10000]
   out.append(f"|{name}|{task}|"+"|".join(f"{np.mean([r[k] for r in rr]):.7g}" for k in ["I_self","I_rest","I_round","K","cND","Q","G"])+"|")
 out+=["","帳簿の分解でありself除去介入ではない。plainSGD/MSEの2境界であり、Adam/CEへ一般化しない。"]
 (OUT/"summary.md").write_text("\n".join(out)+"\n")
if __name__=="__main__":main()
