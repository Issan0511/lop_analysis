from pathlib import Path
import csv,json,math,statistics,hashlib
import numpy as np
import torch
root=Path("/home/issan/Projects/claude/elu_reserve_0913")
out=root/"results/elu_depth_width_0913"
raw=out/"raw"
provs=sorted(raw.glob("*_provenance.json")); assert len(provs)==6
vals={};summary=[];maxerr={"norm":0.,"mean":0.,"rowmean":0.}
hashes=set()
for pp in provs:
    p=json.loads(pp.read_text());a=p["activation"];s=p["seed"];tag=f"{a}_s{s}"
    assert p["checks"]["anchor"]["count"]==120 and p["checks"]["anchor"]["maxabs"]==0
    assert p["threads"]==1 and p["tasks"]==120 and p["prefix"]==20
    hashes.add((p["spec_sha256"],p["code_sha256"]))
    log=(raw/f"{tag}.log").read_text().splitlines()
    assert log[-1].startswith("FINISHED "),tag
    assert sum(line.startswith("CELL_DONE ") for line in log)==5
    assert not any("Traceback" in line for line in log),tag
    rows=list(csv.DictReader((raw/f"{tag}_rows.csv").open()))
    assert len(rows)==3020,(tag,len(rows))
    for c in ("n5_d1","n10_d1","n5_d4","n10_d4","ref"):
        r=[r for r in rows if r["cell"]==c]
        assert len(r)==600
        for task in range(21,121):
            rr=[x for x in r if int(x["task"])==task]
            assert sorted(int(x["step"]) for x in rr)==[-1,0,20,100,300,625]
        for window,lo,hi in (("early",21,40),("late",101,120)):
            rr=[x for x in r if lo<=int(x["task"])<=hi and int(x["step"])==625]
            vals[a,s,c,window]=statistics.mean(float(x["test_acc"]) for x in rr)
    snap=torch.load(raw/f"{tag}_task20.pt",weights_only=False)
    mbase=snap["p"][0].double().mean(1).numpy()
    with np.load(raw/f"{tag}_units.npz") as z:
        for c,n,d in (("n5_d1",5.,-1.),("n10_d1",10.,-1.),("n5_d4",5.,-4.),("n10_d4",10.,-4.)):
            assert p["branch_checks"][c]["projections"]==62600
            for t in range(21,121):
                for step in (0,625):
                    prefix=f"{c}_t{t}_s{step}_"
                    v={"norm":float(np.max(np.abs(z[prefix+"cnorm_i"]/n-1))),
                       "mean":float(np.max(np.abs(z[prefix+"train_zmean_i"]-d))),
                       "rowmean":float(np.max(np.abs(z[prefix+"rowmean_i"]-mbase)))}
                    for k,e in v.items():maxerr[k]=max(maxerr[k],e)
    summary.append(dict(activation=a,seed=s,wall_seconds=p["wall_seconds"],anchor_count=120,anchor_maxabs=0.))
assert len(hashes)==1
assert maxerr["norm"]<2e-6 and maxerr["mean"]<2e-5 and maxerr["rowmean"]<2e-7
verdict=list(csv.DictReader((out/"verdict.csv").open()))
worst=0.;records=[]
for depth in ("d1","d4"):
    v=[]
    for s in range(3):
        ew=vals["ELU1",s,f"n5_{depth}","late"]-vals["ELU1",s,f"n10_{depth}","late"]
        lw=vals["LR",s,f"n5_{depth}","late"]-vals["LR",s,f"n10_{depth}","late"]
        v.append(ew-lw)
    r=next(r for r in verdict if r["metric"]=="late_test_acc" and r["contrast"]==f"H_ELU_minus_LR_{depth}")
    m=statistics.mean(v);h=4.30265273*statistics.stdev(v)/math.sqrt(3)
    err=max(abs(float(r["mean"])-m),abs(float(r["ci_low"])-(m-h)),abs(float(r["ci_high"])-(m+h)))
    worst=max(worst,err)
    records.append(dict(depth=depth,seed_values=v,mean=m,ci=[m-h,m+h],maxdiff=err))
assert worst<1e-10
audit=dict(status="PASS",independent="stdlib CSV endpoint/CI recomputation plus direct NPZ/state invariants",
           workers=summary,raw_rows=18120,unit_record_invariant_max=maxerr,
           H_recomputation=records,CI_maxdiff=worst,
           code_spec_hash_consistency=True,
           limitations=["Only 3 independent seeds","Within-activation common checkpoint; activation baselines differ",
                        "Controlled mean depth and frozen rowmeans; width can still change saturation"])
(out/"audit.json").write_text(json.dumps(audit,indent=2))
execution=dict(remote_parent_pid=1610836,ssh_transport_exit_code=1,
               ssh_transport_message="Connection reset after remote process started",
               remote_processes_continued_after_disconnect=True,
               worker_exit_codes="Not directly observed after SSH disconnect",
               completion_evidence="Six FINISHED log records and six complete provenance files; five CELL_DONE per worker; no traceback",
               report_exit_code=0,audit_exit_code=0,worker_wall_seconds=[p["wall_seconds"] for p in summary],
               restarted=False)
(out/"execution_status.json").write_text(json.dumps(execution,indent=2))
print(json.dumps(audit,indent=2))
