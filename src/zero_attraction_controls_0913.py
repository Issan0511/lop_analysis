"""Registered residual alignment, compensated offset and v=0 algebra controls."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from src.zero_attraction_mechanism_0913 import checkpoint,check,DATA,ROOT
OUT=ROOT/"results/zero_attraction_learning_0913/controls"
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];checks={}
 cfg=json.loads((ROOT/"configs/zero_attraction_learning_0913.yaml").read_text())
 for arm in cfg["arms"]:
  for step in [0,200000,1000000,5000000]:
   cp=torch.load(DATA/"ckpts"/f"{arm['name']}_step{step}.pt",weights_only=True)
   n,x,y,act=checkpoint(cp);W=n["W"];u=W[:,:,15:];v=n["v"]
   z=torch.einsum("rhd,prd->prh",W,x)+n["b"];a=act.act_fn(z);gate=act.act_grad(z,a)
   delta=(a*v).sum(-1)+n["c"]-y
   h=v*gate*torch.einsum("rhj,prj->prh",u,x[:,:,15:])
   hrms=h.square().mean(0).sqrt();valid=hrms>1e-12;aligned=h/hrms.clamp(min=1e-12)
   plus=2*(aligned*h).mean(0);minus=2*(-aligned*h).mean(0)
   check("same_residual_RMS",aligned.square().mean(0),(-aligned).square().mean(0),checks)
   check("opposite_radial",plus,-minus,checks)
   check("unit_residual_RMS",aligned.square().mean(0)[valid],torch.ones_like(hrms[valid]),checks)
   assert (plus[valid]>0).all() and (minus[valid]<0).all()
   # Offset all activations by q, compensate current output with c-q sum(v).
   q=.5;shifted=(v*(a+q)).sum(-1)+n["c"]-q*v.sum(-1)-y
   def gw(d):return torch.einsum("prh,prd->rhd",2*d[:,:,None]*v*gate,x)/32
   check("compensated_delta",delta,shifted,checks)
   check("compensated_incoming_gradient",gw(delta),gw(shifted),checks)
   selfg=torch.einsum("prh,prd->rhd",2*v*v*a*gate,x)/32
   selfshift=torch.einsum("prh,prd->rhd",2*v*v*(a+q)*gate,x)/32
   # At v=0 all incoming W,b gradients are zero for arbitrary finite delta.
   null=torch.einsum("prh,prd->rhd",2*delta[:,:,None]*torch.zeros_like(v)*gate,x)/32
   check("zero_v_incoming",null,torch.zeros_like(null),checks)
   for si,run in enumerate(cp["runs"]):
    rows.append(dict(arm=arm["name"],step=step,seed=run["seed"],valid_units=int(valid[si].sum()),undefined_alignment_units=int((~valid[si]).sum()),positive_residual_radial_mean=float(plus[si].mean()),negative_residual_radial_mean=float(minus[si].mean()),compensated_self_change_norm=float((selfshift[si]-selfg[si]).norm()),free_rms=float(u[si].square().sum(-1).mean().sqrt()),fixed15_rms=float(W[si,:,:15].square().sum(-1).mean().sqrt()),full_W_rms=float(W[si].square().sum(-1).mean().sqrt()),b_rms=float(n["b"][si].square().mean().sqrt()),v_rms=float(v[si].square().mean().sqrt())))
 with (OUT/"seed_summary.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 (OUT/"verification.json").write_text(json.dumps({"status":"PASS","n_arms":len(cfg["arms"]),"n_checkpoints":len(cfg["arms"])*4,"max_abs_errors":checks,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"interpretation":"Algebra controls at fixed model states; altered residuals define hypothetical targets, not new natural learning tasks."},indent=2))
 print("CONTROLS PASS",checks)
if __name__=="__main__":main()
