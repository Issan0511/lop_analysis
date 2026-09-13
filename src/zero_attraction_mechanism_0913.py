"""Registered fixed-checkpoint local perturbations; scalar MSE loss E[delta^2]."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json
import numpy as np
import torch
from src.nets import VecMLPL
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/home/issan/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913")
RESULTS=ROOT/"results/zero_attraction_learning_0913"
def arr(x):return x.detach().cpu().numpy()
def writecsv(p,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def check(key,a,b,checks,atol=1e-8,rtol=1e-8):
 err=float((a-b).abs().max());checks[key]=max(checks.get(key,0),err)
 torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
def checkpoint(cp):
 n={k:v.double() for k,v in cp["net"].items()}
 R,H,D=n["W"].shape
 bits=torch.tensor(list(itertools.product([0.,1.],repeat=5)),dtype=torch.float64)
 raw=torch.cat([cp["env"]["flip_state"].double()[None].expand(32,-1,-1),bits[:,None].expand(-1,R,-1)],-1)
 x=raw-(cp["layer_means"][0].double()[None] if cp["centered_layers"][0] else 0)
 t=cp["teacher"]
 pre=torch.einsum("rhd,prd->prh",t["W"].double(),raw)+t["b"].double()
 y=((pre>=t["tau"].double()).double()*t["v"].double()).sum(-1)+t["cout"].double()
 act=VecMLPL(1,[1],1,torch.Generator().manual_seed(0),"cpu",act=cp["activation"],act_alpha=cp["act_alpha"])
 return n,x,y,act
def root(act):
 if float(act.act_fn(torch.tensor(0.,dtype=torch.float64)))==0.:return 0.
 l,h=-100.,100.
 for _ in range(90):
  m=(l+h)/2
  if act.act_fn(torch.tensor(m,dtype=torch.float64))>0:h=m
  else:l=m
 return (l+h)/2
def measure(path,check_only=False):
 cp=torch.load(path,map_location="cpu",weights_only=True);n,x,y,act=checkpoint(cp)
 z=torch.einsum("rhd,prd->prh",n["W"],x)+n["b"]
 a=act.act_fn(z);g=act.act_grad(z,a);o=n["v"]*a
 pred=o.sum(-1)+n["c"];delta=pred-y
 other=delta[:,:,None]-o
 mu=x.mean(0);xi=x[:,:,15:]-mu[:,15:]
 rho=1+torch.einsum("prd,rd->pr",x,mu)
 u=n["W"][:,:,15:];radius=u.norm(dim=-1);direction=u/radius.clamp(min=1e-12)[:,:,None]
 feature=torch.einsum("rhj,prj->prh",direction,xi)
 actual_feature=torch.einsum("rhj,prj->prh",direction,x[:,:,15:])
 def forces(zz,dd):
  aa=act.act_fn(zz);gg=act.act_grad(zz,aa);vv=n["v"]
  dp=dd*gg
  bias=-2*vv*dp.mean(0)
  mean=-2*vv*(dp*rho[:,:,None]).mean(0)
  radial=-2*vv*(dp*actual_feature).mean(0)
  tangent=-2*vv*(dp*feature).mean(0)
  self_mean=-2*vv**2*(aa*gg*rho[:,:,None]).mean(0)
  rest_mean=-2*vv*(other*gg*rho[:,:,None]).mean(0)
  self_tan=-2*vv**2*(aa*gg*feature).mean(0)
  rest_tan=-2*vv*(other*gg*feature).mean(0)
  return dict(bias=bias,mean=mean,radial=radial,tangent=tangent,self_mean=self_mean,rest_mean=rest_mean,self_tangent=self_tan,rest_tangent=rest_tan)
 checks={}
 if not check_only:
  for ri,run in enumerate(cp["runs"]):
   with np.load(DATA/"logs"/f"{cp['arm']}_seed{run['seed']}.npz") as log:
    ix=np.flatnonzero(log["step"]==cp["step"]);assert len(ix)==1
    k=int(ix[0])
    for key,val in [("layer1_zbar",z[:,ri].mean(0)),("layer1_zmin",z[:,ri].min(0).values),("layer1_zmax",z[:,ri].max(0).values),("eval_loss_exact",delta[:,ri].square().mean())]:
     target=torch.as_tensor(log[key][k],dtype=torch.float64)
     check("baseline_"+key,val,target,checks,atol=3e-5,rtol=1e-4)
 base=forces(z,delta[:,:,None].expand_as(z))
 check("mean_self_rest",base["mean"],base["self_mean"]+base["rest_mean"],checks)
 check("tangent_self_rest",base["tangent"],base["self_tangent"]+base["rest_tangent"],checks)
 err=2*delta[:,:,None]*n["v"]*g
 gw=torch.einsum("prh,prd->rhd",err,x)/32;gb=err.mean(0)
 check("actual_mean",base["mean"],-gb-torch.einsum("rhd,rd->rh",gw,mu),checks)
 check("actual_radial",base["radial"],-(direction*gw[:,:,15:]).sum(-1),checks)
 # Independent network autograd.
 ww=n["W"].clone().requires_grad_();bb=n["b"].clone().requires_grad_()
 zz=torch.einsum("rhd,prd->prh",ww,x)+bb
 dd=(act.act_fn(zz)*n["v"]).sum(-1)+n["c"]-y
 dd.square().mean(0).sum().backward()
 check("autograd_W",ww.grad,gw,checks);check("autograd_b",bb.grad,gb,checks)
 eta=float(cp["runs"][0]["lr"])
 sample_g=err[:,:,:,None]*x[:,:,None,15:]
 R=-2*eta*(u*gw[:,:,15:]).sum(-1)
 Q=eta**2*sample_g.square().sum(-1).mean(0)
 observed=((u[None]-eta*sample_g).square().sum(-1)-u.square().sum(-1)[None]).mean(0)
 check("expected_SGD_norm_identity",observed,R+Q,checks,atol=1e-9,rtol=1e-8)
 # Independent per-unit worlds for each intervention.
 records=[];z0=root(act)
 for kind in ["mean","width"]:
  for eps in [.01,.05,.1]:
   f=[]
   for sign in [-1,1]:
    displacement=sign*eps if kind=="mean" else sign*eps*radius[None]*feature
    zp=z+displacement;ap=act.act_fn(zp);dd=delta[:,:,None]+n["v"]*(ap-a)
    ff=forces(zp,dd)
    check("perturbed_mean_self_rest",ff["mean"],ff["self_mean"]+ff["rest_mean"],checks)
    check("perturbed_tangent_self_rest",ff["tangent"],ff["self_tangent"]+ff["rest_tangent"],checks)
    f.append(ff)
   denom=2*eps if kind=="mean" else 2*eps*radius.clamp(min=1e-12)
   keys=["mean","self_mean","rest_mean"] if kind=="mean" else ["tangent","self_tangent","rest_tangent"]
   kappa={k:-(f[1][k]-f[0][k])/denom for k in keys}
   for ri,run in enumerate(cp["runs"]):
    for ui in range(u.shape[1]):
     row={"arm":cp["arm"],"step":cp["step"],"task":cp["step"]//10000,"seed":run["seed"],"unit":ui,"kind":kind,"eps":eps,
      "radius":float(radius[ri,ui]),"root":z0,"mean_z":float(z[:,ri,ui].mean()),"v":float(n["v"][ri,ui]),
      "effective_v2_gate2":float(n["v"][ri,ui]**2*g[:,ri,ui].square().mean()),
      "output_variance":float(o[:,ri,ui].var(unbiased=False)),
      "mean_force":float(base["mean"][ri,ui]),"radial_force":float(base["radial"][ri,ui]),
      "tangent_force":float(base["tangent"][ri,ui]),"self_tangent_force":float(base["self_tangent"][ri,ui]),
      "rest_tangent_force":float(base["rest_tangent"][ri,ui]),
      "expected_SGD_R":float(R[ri,ui]),"expected_SGD_Q":float(Q[ri,ui]),"expected_SGD_G":float((R+Q)[ri,ui]),
      "radius_valid":bool(radius[ri,ui]>1e-8),
      "kappa_total":float(kappa[keys[0]][ri,ui]),"kappa_self":float(kappa[keys[1]][ri,ui]),"kappa_rest":float(kappa[keys[2]][ri,ui])}
     records.append(row)
 # Autograd derivative along independent mean-preserving width changes.
 ee=torch.zeros_like(radius,requires_grad=True)
 zt=z+ee[None]*radius[None]*feature
 dt=delta[:,:,None]+n["v"]*(act.act_fn(zt)-a)
 gg=torch.autograd.grad(dt.square().mean(0).sum(),ee)[0]
 check("width_tangent_autograd",gg,-radius*base["tangent"],checks)
 # Explicit finite parameter change preserves mean.
 up=u*1.05;bp=n["b"]-.05*(u*mu[:,None,15:]).sum(-1)
 wwide=n["W"].clone();wwide[:,:,15:]=up
 zwide=torch.einsum("rhd,prd->prh",wwide,x)+bp
 check("width_preserves_mean",zwide.mean(0),z.mean(0),checks)
 check("width_displacement",zwide,z+.05*radius[None]*feature,checks)
 return records,checks
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--check-only",action="store_true");ap.add_argument("--available",action="store_true");args=ap.parse_args()
 out=RESULTS/("mechanism_partial" if args.available else "mechanism");out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/"configs/zero_attraction_learning_0913.yaml").read_text())
 allchecks={};seedrows=[]
 previous=RESULTS/"mechanism_partial"
 prevchecks=json.loads((previous/"verification.json").read_text()).get("checks",{}) if (previous/"verification.json").exists() and not args.available and not args.check_only else {}
 prevrows=list(csv.DictReader((previous/"seed_summary.csv").open())) if prevchecks else []
 for arm in cfg["arms"]:
  if args.available and not (DATA/"arm_status"/f"{arm['name']}_done.json").exists():continue
  for step in [0,200000,1000000,5000000]:
   if args.check_only and (arm["name"]!="SN_peak_q0" or step!=0):continue
   p=DATA/"ckpts"/f"{arm['name']}_step{step}.pt"
   key=f"{arm['name']}_{step}"
   if key in prevchecks:
    import shutil
    src=previous/f"{arm['name']}_step{step}_units.csv"
    shutil.copy2(src,out/src.name)
    seedrows.extend({k:(v if k in ["arm","kind"] else (None if v=="" else float(v))) for k,v in row.items()} for row in prevrows if row["arm"]==arm["name"] and int(row["step"])==step)
    allchecks[key]=prevchecks[key]
    print("REUSE VERIFIED PROBE",arm["name"],step,flush=True)
    continue
   rows,checks=measure(p,check_only=args.check_only)
   if not args.check_only:
    writecsv(out/f"{arm['name']}_step{step}_units.csv",rows)
    for seed in range(10):
     for kind in ["mean","width"]:
      for eps in [.01,.05,.1]:
       rr=[r for r in rows if r["seed"]==seed and r["kind"]==kind and r["eps"]==eps and (kind=="mean" or r["radius_valid"])]
       sr={"arm":arm["name"],"step":step,"seed":seed,"kind":kind,"eps":eps,"n_units":len(rr)}
       for key in ["kappa_total","kappa_self","kappa_rest","radial_force","tangent_force","self_tangent_force","rest_tangent_force","effective_v2_gate2","output_variance","expected_SGD_R","expected_SGD_Q","expected_SGD_G"]:
        sr[key+"_mean"]=float(np.mean([r[key] for r in rr])) if rr else None
       for key in ["kappa_total","kappa_self","kappa_rest"]:
        sr[key+"_positive_fraction"]=float(np.mean([r[key]>0 for r in rr])) if rr else None
       sr["expected_SGD_shrink_fraction"]=float(np.mean([r["expected_SGD_G"]<0 for r in rr])) if rr else None
       seedrows.append(sr)
   allchecks[f"{arm['name']}_{step}"]=checks
   print("PROBE PASS",arm["name"],step,flush=True)
 if seedrows:writecsv(out/"seed_summary.csv",seedrows)
 (out/("check_only.json" if args.check_only else "verification.json")).write_text(json.dumps({"status":"PASS","checks":allchecks},indent=2))
if __name__=="__main__":main()
