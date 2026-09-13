#!/usr/bin/env python3
"""Retrospective/post-hoc unit identity tracking from committed spec section 2."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T, U, CUT, STRONG = 50, 100, .95, .5
COLORS = {"ELU1": "#d66535", "LR": "#3273a8"}

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def dump_csv(p, rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def key(m): return (m["seed"],m["env"],m["act"],m["iv"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    paths=[a.source/x for x in ("units.npz","provenance.json","rows.csv")]
    prov=json.loads(paths[1].read_text()); models=prov["models"]
    assert len(models)==24 and len({key(m) for m in models})==24
    q={}
    with np.load(paths[0]) as z:
        for mi,m in enumerate(models):
            for layer in (1,2): q[key(m)+(layer,)]=np.stack([z[f"lowgate_l{layer}_t{t}"][mi] for t in range(1,T+1)])
    expected={(s,e,x,v,l) for s in range(3) for e in ("PM","RL") for x in ("ELU1","LR") for v in ("ref","wclamp") for l in (1,2)}
    finite=all(np.isfinite(x).all() for x in q.values()); shapes=all(x.shape==(T,U) for x in q.values()); bounds=all(((x>=0)&(x<=1)).all() for x in q.values())
    assert set(q)==expected and finite and shapes and bounds
    transitions=[]; events=[]; counts=[]; summaries=[]; grouped=defaultdict(list); balance=True; eid=0
    for k in sorted(q):
        seed,env,act,iv,layer=k; x=q[k]; sink=x>=CUT; prev=np.zeros(U,bool); recent=[]; local=[]
        for ti in range(T):
            task=ti+1; state=sink[ti]; ent=state&~prev; ext=~state&prev; recent.append(state)
            n,ne,nx=map(int,(state.sum(),ent.sum(),ext.sum())); balance &= n==int(prev.sum())+ne-nx
            counts.append(dict(seed=seed,env=env,act=act,iv=iv,layer=layer,task=task,occupancy=n,entries=ne,exits=nx,persistent_latest5=int(np.logical_and.reduce(recent[-5:]).sum()) if task>=5 else ""))
            for u in range(U):
                transitions.append(dict(seed=seed,env=env,act=act,iv=iv,layer=layer,unit_id=u,task=task,q_lowgate=float(x[ti,u]),sink=int(state[u]),entry=int(ent[u]),exit=int(ext[u]),persistent_latest5=int(all(y[u] for y in recent[-5:])) if task>=5 else ""))
            for u in np.flatnonzero(ent):
                eid+=1; eligible=task<=45; fs=sink[ti+1:ti+6,u] if eligible else np.array([],bool); fq=x[ti+1:ti+6,u] if eligible else np.array([])
                eo=np.flatnonzero(~fs)+1; so=np.flatnonzero(fq<STRONG)+1
                ev=dict(event_id=eid,seed=seed,env=env,act=act,iv=iv,layer=layer,unit_id=int(u),entry_task=task,entry_q=float(x[ti,u]),followup_eligible=int(eligible),censor_reason="" if eligible else "entry_task_gt_45",exit_within5=int(eo.size>0) if eligible else "",first_exit_task=task+int(eo[0]) if eo.size else "",q_lt_0_5_within5=int(so.size>0) if eligible else "",first_q_lt_0_5_task=task+int(so[0]) if so.size else "")
                events.append(ev); local.append(ev); grouped[k].append(ev)
            prev=state
        el=[e for e in local if e["followup_eligible"]]; nr=sum(int(e["exit_within5"]) for e in el); ns=sum(int(e["q_lt_0_5_within5"]) for e in el)
        summaries.append(dict(seed=seed,env=env,act=act,iv=iv,layer=layer,occupancy_t10=int(sink[9].sum()),occupancy_t20=int(sink[19].sum()),occupancy_t50=int(sink[49].sum()),total_entries=len(local),eligible_entries=len(el),censored_entries=len(local)-len(el),exit_within5_count=nr,exit_within5_fraction=nr/len(el) if el else "",q_lt_0_5_within5_count=ns,q_lt_0_5_within5_fraction=ns/len(el) if el else ""))
    leaky_zero=all(np.count_nonzero(x)==0 for k,x in q.items() if k[2]=="LR")
    assert len(transitions)==240000 and balance and leaky_zero
    for name,rows in (("unit_transitions.csv",transitions),("entry_events.csv",events),("task_counts.csv",counts),("seed_summary.csv",summaries)): dump_csv(a.out/name,rows)

    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True,sharex=True); im=None
    for r,env in enumerate(("PM","RL")):
        for c,l in enumerate((1,2)):
            im=axs[r,c].imshow(q[(0,env,"ELU1","ref",l)].T,aspect="auto",origin="lower",vmin=0,vmax=1,extent=(.5,50.5,-.5,99.5),cmap="magma")
            axs[r,c].set(title=f"{env}, layer {l}",xlabel="task endpoint",ylabel="stable unit ID")
    fig.colorbar(im,ax=axs,label="q = fraction with derivative < .05",shrink=.85); fig.suptitle("RETROSPECTIVE / POST-HOC — ELU ref identity tracks (seed 0; display only)"); fig.savefig(a.out/"identity_heatmaps_elu_ref_seed0.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True,sharex=True,sharey=True)
    for r,env in enumerate(("PM","RL")):
        for c,l in enumerate((1,2)):
            ax=axs[r,c]
            for act,name in (("ELU1","ELU"),("LR","leaky .1")):
                for iv,ls in (("ref","-"),("wclamp","--")):
                    cs=np.stack([(q[(s,env,act,iv,l)]>=CUT).sum(1) for s in range(3)]); ax.plot(range(1,51),cs.mean(0),color=COLORS[act],ls=ls,lw=2,label=f"{name} {iv}"); ax.fill_between(range(1,51),cs.min(0),cs.max(0),color=COLORS[act],alpha=.1)
            ax.set(title=f"{env}, layer {l}",xlabel="task endpoint",ylabel="sink units / 100"); ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8); fig.suptitle("RETROSPECTIVE / POST-HOC — sink occupancy (mean of 3 seeds; band = seed range)"); fig.savefig(a.out/"sink_count_curves_mean3seed.png",dpi=180); plt.close(fig)

    lines=["# Retrospective identity tracking (post-hoc)","","Descriptive reanalysis of existing task-endpoint arrays; not a new experiment, causal test, or hypothesis-significance analysis. Recurrent entries are episodes, not independent replicas.","","Sink is q >= .95, where q is each unit's fraction of 1,200 task inputs with phi'(z) < .05. Recovery follows the same unit ID through endpoints t+1..t+5. Entries after task 45 are right-censored. Leave-and-return counts as recovery; task-end data cannot exclude within-task transients.","","## ELU ref persistent stock (sink at each of latest five endpoints)","","Each cell gives the three seed counts in seed order 0, 1, 2.","","| env | layer | task 10 | task 20 | task 50 |","|---|---:|---:|---:|---:|"]
    count_lookup={(int(r["seed"]),r["env"],r["act"],r["iv"],int(r["layer"]),int(r["task"])):r for r in counts}
    for env in ("PM","RL"):
        for l in (1,2):
            vals=[]
            for t in (10,20,50): vals.append(", ".join(str(count_lookup[(s,env,"ELU1","ref",l,t)]["persistent_latest5"]) for s in range(3)))
            lines.append(f"| {env} | {l} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ["","## ELU eligible entry episodes by entry-task window (three seeds combined descriptively)","","| env | iv | layer | entry tasks | entries | exit q<.95 within 5 | stronger q<.5 within 5 |","|---|---|---:|---:|---:|---:|---:|"]
    for env in ("PM","RL"):
      for iv in ("ref","wclamp"):
       for l in (1,2):
        evs=[e for s in range(3) for e in grouped[(s,env,"ELU1",iv,l)] if e["followup_eligible"]]
        for lo,hi in ((1,10),(11,20),(21,30),(31,40),(41,45)):
            z=[e for e in evs if lo<=e["entry_task"]<=hi]; n=len(z); nr=sum(int(e["exit_within5"]) for e in z); ns=sum(int(e["q_lt_0_5_within5"]) for e in z)
            lines.append(f"| {env} | {iv} | {l} | {lo}-{hi} | {n} | {nr} ({nr/n:.1%}) | {ns} ({ns/n:.1%}) |" if n else f"| {env} | {iv} | {l} | {lo}-{hi} | 0 | NA | NA |")
    lines += ["","Leaky .1 low-gate values are structurally zero at this threshold; this does not imply leaky has no LoP. Cross-activation functional conclusions require actual learning outcomes. Ever-sunk cumulative count is not evidence of accumulation. Any stock/performance association is descriptive."]
    (a.out/"summary.md").write_text("\n".join(lines)+"\n")
    # Independent outcome audit: reconstruct each emitted event directly from its
    # stable unit-ID slice rather than trusting the event-building intermediates.
    direct_ok=True; eligible_checked=0; censored_checked=0
    for e in events:
        ek=(int(e["seed"]),e["env"],e["act"],e["iv"],int(e["layer"])); ti=int(e["entry_task"])-1; u=int(e["unit_id"])
        if int(e["followup_eligible"]):
            eligible_checked+=1; direct=q[ek][ti+1:ti+6,u]
            direct_ok &= len(direct)==5 and int(np.any(direct<CUT))==int(e["exit_within5"]) and int(np.any(direct<STRONG))==int(e["q_lt_0_5_within5"])
        else:
            censored_checked+=1; direct_ok &= int(e["entry_task"])>45 and e["exit_within5"]=="" and e["q_lt_0_5_within5"]==""
    assert direct_ok
    audit=dict(analysis_label="RETROSPECTIVE/POSTHOC reanalysis; descriptive only",source_files={p.name:{"path":str(p.resolve()),"sha256":digest(p)} for p in paths},models=len(models),expected_model_keys=len(expected),layers=[1,2],tasks=[1,50],units_per_layer=U,unit_transition_rows=len(transitions),entry_event_rows=len(events),finite_arrays=finite,array_shapes_valid=shapes,q_bounds_valid=bounds,stock_identity_balance_valid=balance,indexing_valid=set(q)==expected,indexing_definition="NPZ model axis matches provenance.models; unit_id is unchanged array column 0..99; task t uses lowgate_l{layer}_t{t}",initialization_convention="N(0)=0 only; task-1 sinks are false-to-true entries",followup="entry task <=45; endpoints t+1..t+5; later entries right-censored",direct_event_slice_audit={"valid":bool(direct_ok),"eligible_events_checked":eligible_checked,"right_censored_events_checked":censored_checked,"method":"For every emitted event, independently slice original q[model, layer][entry_task:entry_task+5, unit_id]; compare any q<.95 and any q<.5. Verify late-event labels are blank."},leaky_lowgate_structurally_zero=leaky_zero,no_pseudoreplication="seed_summary retains one row per seed/env/act/iv/layer; episode summaries are descriptive",generated_files=["unit_transitions.csv","entry_events.csv","task_counts.csv","seed_summary.csv","summary.md","identity_heatmaps_elu_ref_seed0.png","sink_count_curves_mean3seed.png"])
    (a.out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
if __name__=="__main__": main()
