"""G0/G1/G2/M1 checks, including the three registered mutation controls."""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); os.environ.setdefault("OMP_NUM_THREADS","1")
from src import fb_width_seat_0915 as F

SKIP={"arm","run_id","state_hash_final","init_hook","init_hook_arg","batch_mode",
      "layer1_branch_step","layer1_w_flip","layer1_w_flip_norm","layer1_n_band",
      "layer1_k_on","layer1_unit_all_negative","projection_zero_free","projection_zero_flip"}
SKIP.update({"numeric_divergence","divergence_events"})

def same(a,b,mask=None):
    A=np.load(a); B=np.load(b); bad=[]
    missing=(set(B.files)^set(A.files))-SKIP
    bad.extend("missing:"+k for k in sorted(missing))
    for k in sorted(set(A.files)&set(B.files)-SKIP):
        if mask is not None and k in ("state_hash_1m","state_hash_final"): continue
        x,y=A[k],B[k]
        if mask is not None and x.ndim:
            if x.shape[0]==len(A["step"]): x=x[mask]; y=y[mask]
            elif k=="layer1_w_free" and "layer1_w_free_step" in A:
                m=A["layer1_w_free_step"]<=A["step"][mask][-1]; x=x[m]; y=y[m]
            elif k.startswith("layer1_m_") and "layer1_moment_step" in A:
                m=A["layer1_moment_step"]<=A["step"][mask][-1]; x=x[m]; y=y[m]
        if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes(): bad.append(k)
    return bad

def smoke(out):
    out=Path(out); shutil.rmtree(out,ignore_errors=True)
    host=out/"host"; F.run_single_arm("LRoff0_1216",40000,host,[0])
    direct=out/"host_direct"; cfg0=F.build_cfg(); cfg0["common"]["seeds"]=[0]; st0=F.setup_arm_dial(cfg0,F._arm(cfg0,"LRoff0_1216"),"cpu")
    probes=list(range(0,40001,1000)); rec0=F.E.EdgeRecorder(probes,st0); F.E.train_arm_edge(st0,rec0,probes,40000,direct,[]); F.E.write_arm_logs_edge(direct,"LRoff0_1216",st0,rec0)
    arms=["LRwf21_1216","FB21LRoff0_1216","FB21LRwf21_1216","LRwi21_1216","FB21LRwi21_1216"]
    for a in arms: F.run_single_arm(a,40000,out/"normal",[0],switch_step=20000)
    base=host/"logs/LRoff0_1216_seed0.npz"; checks={}
    # S-noop is the host function identity plus a complete numeric-log comparison.
    nb=same(base,direct/"logs/LRoff0_1216_seed0.npz"); checks["S_noop"]={"pass":not nb,"bad":nb}
    for a in arms:
        p=out/"normal/logs"/f"{a}_seed0.npz"; z=np.load(p); mask=z["step"]<=20000
        checks.setdefault("S_prefix",{})[a]=same(base,p,mask)
        q=np.load(base); checks.setdefault("S_flip",{})[a]=([] if np.array_equal(q["flip_state"],z["flip_state"]) else ["flip_state"])
    z=np.load(out/"normal/logs/LRwf21_1216_seed0.npz"); i=np.where(z["layer1_w_free_step"]==20000)[0][0]
    n=np.linalg.norm(z["layer1_w_free"][i],axis=-1); later=np.linalg.norm(z["layer1_w_free"][i:],axis=-1)
    checks["S_clamp"]={"max_rel":float(np.max(np.abs(later/n-1))),"pass":float(np.max(np.abs(later/n-1)))<=1e-6}
    cfg=F.build_cfg(); cfg["common"]["seeds"]=[0]; st=F.setup_arm_dial(cfg,F._arm(cfg,"LRwf21_1216"),"cpu")
    nf=F._norm(st["net"].Ws[0],F.FREE).clone(); before=[st["net"].Ws[0][...,F.FLIP].clone(),st["net"].bs[0].clone(),st["net"].v.clone(),st["net"].c.clone(),st["running_mean"].clone()]
    import torch
    with torch.no_grad(): st["net"].Ws[0][...,F.FREE].mul_(1.01)
    F.project_rows(st,nf); after=[st["net"].Ws[0][...,F.FLIP],st["net"].bs[0],st["net"].v,st["net"].c,st["running_mean"]]
    side=all(torch.equal(a,b) for a,b in zip(before,after)); rel=float(torch.max(torch.abs(F._norm(st["net"].Ws[0],F.FREE)/nf-1)))
    checks["S_clamp_unit"]={"pass":side and rel<=1e-6,"side_effect_free":side,"max_rel":rel}
    # Divergence mutation: host detector fires, only bad seed is quarantined.
    cfgd=F.build_cfg(); cfgd["common"]["seeds"]=[0,1]; sd=F.setup_arm_dial(cfgd,F._arm(cfgd,"LRwf21_1216"),"cpu"); rd=F.BranchRecorder([0],sd)
    good0=sd["net"].Ws[0][0].clone()
    with torch.no_grad(): sd["net"].Ws[0][1,0,0]=float("nan")
    rd(sd,0); div_ok=(sd.get("divergent_seed_indices")=={1} and torch.equal(good0,sd["net"].Ws[0][0]) and torch.isfinite(sd["net"].Ws[0][1]).all())
    checks["S_divergence_seed_isolation"]={"pass":bool(div_ok),"indices":sorted(sd.get("divergent_seed_indices",set()))}
    # Mutations each get their own short run and must violate its target invariant.
    for mut in ("project_flip","moving_reference","late_switch"):
        d=out/("mut_"+mut); F.run_single_arm("LRwf21_1216",40000,d,[0],mutate=mut,switch_step=20000)
        zm=np.load(d/"logs/LRwf21_1216_seed0.npz")
        if mut=="project_flip": detected=zm["layer1_w_flip"].tobytes()!=z["layer1_w_flip"].tobytes()
        elif mut=="moving_reference": detected=float(np.max(np.abs(np.linalg.norm(zm["layer1_w_free"][i:],axis=-1)/n-1)))>1e-6
        else:
            j=np.where(zm["layer1_w_free_step"]==30000)[0][0]; detected=float(np.max(np.abs(np.linalg.norm(zm["layer1_w_free"][j],axis=-1)/n-1)))>1e-6
        checks["mutation_"+mut]={"detected":bool(detected)}
    ok=(checks["S_noop"]["pass"] and checks["S_clamp"]["pass"] and checks["S_clamp_unit"]["pass"] and checks["S_divergence_seed_isolation"]["pass"] and all(not v for v in checks["S_prefix"].values())
        and all(not v for v in checks["S_flip"].values()) and all(checks["mutation_"+m]["detected"] for m in ("project_flip","moving_reference","late_switch")))
    res={"pass":bool(ok),"checks":checks}; (out/"g0.json").write_text(json.dumps(res,indent=2)); return res

def full(out):
    out=Path(out); ref=Path("/home/issan/Projects/obsidian-research-data/act_offset_review_0908/full/logs"); arms=list(F.table()); detail={}; ok=True
    for s in range(10):
        a=out/"logs"/f"LRoff0_1216_seed{s}.npz"; b=ref/f"LRoff0_1216_seed{s}.npz"
        raw_bad = same(a,b) if b.exists() else ["REFERENCE_MISSING"]
        detail[f"G1_raw_mismatch_s{s}"] = raw_bad
        # The sole pre-results technical correction is missing lr_used metadata.
        # Preserve the raw comparison; every other difference still fails G1.
        detail[f"G1_measurement_s{s}"] = [k for k in raw_bad if k != "lr_used"]
        if b.exists():
            A=np.load(a); B=np.load(b); measured_ok=not detail[f"G1_measurement_s{s}"]
            state_ok=bool(np.array_equal(A["state_hash_final"],B["state_hash_final"]))
            lr_ok=bool(np.array_equal(A["lr_used"],B["lr_used"],equal_nan=True))
            detail[f"G1_state_hash_s{s}"]=[] if state_ok else ["state_hash_final"]
            detail[f"G1_raw_metadata_s{s}"]={"lr_used_equal":lr_ok,"run_lr_used":float(A["lr_used"]),"reference_lr_used":float(B["lr_used"]),"classification":("metadata_only" if measured_ok and state_ok and not lr_ok else "not_metadata_only")}
    for arm in arms[1:]:
        for s in range(10):
            p=out/"logs"/f"{arm}_seed{s}.npz"; q=out/"logs"/f"LRoff0_1216_seed{s}.npz"; z=np.load(p); mask=z["step"]<=200000
            detail[f"G2_{arm}_s{s}"]=same(q,p,mask)
    for arm in ("LRwf21_1216","FB21LRwf21_1216","LRwi21_1216","FB21LRwi21_1216"):
        mx=0.; zeros=0
        for s in range(10):
            z=np.load(out/"logs"/f"{arm}_seed{s}.npz");
            if bool(z.get("numeric_divergence",False)): continue
            W=z["layer1_w_free"]; i=np.where(z["layer1_w_free_step"]==200000)[0][0]; n=np.linalg.norm(W[i],axis=-1)
            mx=max(mx,float(np.max(np.abs(np.linalg.norm(W[i:],axis=-1)/n-1)))); zeros+=int(z["projection_zero_free"])
        detail["M1_"+arm]={"max_rel":mx,"zero_skips":zeros,"pass":mx<=1e-5}
        if "wi21" in arm:
            mxfi=0.
            for s in range(10):
                z=np.load(out/"logs"/f"{arm}_seed{s}.npz");
                if bool(z.get("numeric_divergence",False)): continue
                W=z["layer1_w_flip"]; st=z["layer1_branch_step"]; i=np.where(st==200000)[0][0]; nfi=np.linalg.norm(W[i],axis=-1); mxfi=max(mxfi,float(np.max(np.abs(np.linalg.norm(W[i:],axis=-1)/nfi-1))))
            detail["M1_"+arm]["flip_max_rel"]=mxfi; detail["M1_"+arm]["pass"] &= mxfi<=1e-5
    res={"pass":all(not v for k,v in detail.items() if k.startswith("G1_measurement_") or k.startswith("G1_state_hash_") or k.startswith("G2_")) and all(v["pass"] for k,v in detail.items() if k.startswith("M1_")),"detail":detail,"g1_reference":str(ref),"g1_raw_pass":all(not v for k,v in detail.items() if k.startswith("G1_raw_mismatch_") or k.startswith("G1_state_hash_"))}
    (out/"checks.json").write_text(json.dumps(res,indent=2)); return res

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--full",action="store_true"); a=p.parse_args(); print(json.dumps(full(a.out) if a.full else smoke(a.out),indent=2))
