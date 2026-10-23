"""Read-only full-population diagnostics and reset-generation lineage.

Lineage uses strict max(z)>0 over the fixed population as its primary alive
definition; native clamp-autograd status is recorded separately. First death
is observed/censored, never inferred continuously between recorded states.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def digest(obj):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy(); add(a)
        elif isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b"{")
            for k in sorted(x,key=lambda v:(type(v).__name__,str(v))): add(k); add(x[k])
            h.update(b"}")
        elif isinstance(x,(tuple,list)):
            h.update(b"[")
            for y in x:add(y)
            h.update(b"]")
        else:h.update((type(x).__name__+":"+repr(x)+";").encode())
    add(obj);return h.hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_name(path.name+".tmp")
    with temp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def _versions(P,act):
    extra=[]
    if getattr(act,"m",None) is not None:extra+=act.m
    if getattr(act,"v",None) is not None:extra+=act.v
    return tuple(p._version for p in list(P)+extra)


@torch.no_grad()
def unit_metrics(P,X,act):
    """Native float32 forward; float64 moments, s correction1, g variance0."""
    from src import relu_doors_0919 as E
    version=_versions(P,act)
    z1,h1,z2,h2,_=E.forward(P,X,act,train=False)
    out={}
    for li,(z,h) in enumerate(((z1,h1),(z2,h2)),1):
        if not torch.isfinite(z).all() or not torch.isfinite(h).all():
            raise ValueError("DIVERGED: nonfinite full-population diagnostic forward")
        with torch.enable_grad():
            q=z.detach().requires_grad_(True)
            value=act.phi(q,li-1,train=False)
            native=torch.autograd.grad(value,q,torch.ones_like(value))[0].detach().double()
        zd=z.double(); strict=(z>0).double(); pre=z.clamp(min=0)
        m,s=zd.mean(1),zd.std(1,correction=1)
        r=m/s  # Keep +/-inf and NaN with explicit counts; no epsilon.
        actual_abs=h.abs().mean(1)
        pre_abs=pre.abs().mean(1)
        actual_score=actual_abs/(actual_abs.mean(1,keepdim=True)+1e-9)
        pre_score=pre_abs/(pre_abs.mean(1,keepdim=True)+1e-9)
        data=dict(m=m,s=s,r=r,b=P[2*li-1].detach().double(),
                  strict_g_mean=strict.mean(1),strict_g_var=strict.var(1,correction=0),
                  native_g_mean=native.mean(1),native_g_var=native.var(1,correction=0),
                  strict_dead=~(z.amax(1)>0),native_dead=(native.amax(1)==0),
                  actual_abs_activity=actual_abs,preH_abs_activity=pre_abs,
                  actual_activity_score=actual_score,preH_activity_score=pre_score,
                  w_norm=P[2*li-2].detach().double().norm(dim=2))
        if getattr(act,"m",None) is not None:data["ema"]=act.m[li-1].detach().double()
        else:data["ema"]=torch.zeros_like(m)
        out.update({f"l{li}_{k}":v.cpu().numpy() for k,v in data.items()})
    if version!=_versions(P,act):raise ValueError("CHECK_FAILED: diagnostic mutated native tensors")
    return out


class Diagnostics:
    def __init__(self,out,arm,seeds):
        self.out,self.arm,self.seeds=Path(out),arm,list(seeds)
        self.completed_tasks=[];self.rows=[];self.files={}
        self.generation=self.start=self.started_alive=self.first_death=self.reason=None
        self.native_started_alive=self.native_first_death=None

    def initialize(self,P):
        if self.generation is not None:return
        shapes=[tuple(P[i].shape) for i in (1,3)]
        self.generation=[np.zeros(s,np.int64) for s in shapes]
        self.start=[np.zeros(s,np.int64) for s in shapes]
        self.started_alive=[np.zeros(s,bool) for s in shapes]
        self.native_started_alive=[np.zeros(s,bool) for s in shapes]
        self.first_death=[np.full(s,-1,np.int64) for s in shapes]
        self.native_first_death=[np.full(s,-1,np.int64) for s in shapes]
        self.reason=[np.zeros(s,np.int8) for s in shapes]

    def state(self):
        return dict(schema_version=1,completed_tasks=self.completed_tasks,rows=self.rows,files=self.files,
                    generation=self.generation,start=self.start,started_alive=self.started_alive,
                    native_started_alive=self.native_started_alive,first_death=self.first_death,
                    native_first_death=self.native_first_death,reason=self.reason)

    def load_state(self,st):
        if st.get("schema_version")!=1:raise ValueError("CHECK_FAILED: diagnostic checkpoint schema")
        for key in ("completed_tasks","rows","files","generation","start","started_alive",
                    "native_started_alive","first_death","native_first_death","reason"):
            setattr(self,key,st[key])
        # Only checkpointed files count. Later files from an interrupted task
        # are overwritten deterministically when that task is replayed.
        for rel,expected in self.files.items():
            p=Path(rel)
            if p.is_absolute() or ".." in p.parts or sha(self.out/p)!=expected:
                raise ValueError("CHECK_FAILED: diagnostic file/checkpoint disagreement")

    def _observe(self,tc,data,event_associated=False):
        for li in range(2):
            dead=data[f"l{li+1}_strict_dead"]
            new=(self.generation[li]>0)&self.started_alive[li]&(self.first_death[li]<0)&dead
            self.first_death[li][new]=tc
            self.reason[li][new]=2 if event_associated else 1
            native_dead=data[f"l{li+1}_native_dead"]
            new_native=(self.generation[li]>0)&self.native_started_alive[li]&(self.native_first_death[li]<0)&native_dead
            self.native_first_death[li][new_native]=tc

    def _lineage(self):
        out={}
        for li in range(2):
            for name,value in (("generation",self.generation),("generation_start",self.start),
                               ("generation_started_alive",self.started_alive),
                               ("generation_native_started_alive",self.native_started_alive),
                               ("first_observed_death",self.first_death),
                               ("first_observed_native_death",self.native_first_death),
                               ("first_death_reason",self.reason)):
                out[f"l{li+1}_{name}"]=value[li].copy()
        return out

    def _save(self,rel,data):
        path=self.out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+".tmp")
        with temp.open("wb") as f:np.savez_compressed(f,**data)
        temp.replace(path);self.files[rel]=sha(path)

    def before_recycle(self,index,tc,task,P,X,act,masks,scores,probe_ids):
        data=unit_metrics(P,X,act)
        self._observe(tc,data,event_associated=False)
        data.update(self._lineage())
        for li in range(2):
            data[f"l{li+1}_reset_mask"]=masks[li].numpy().copy()
            data[f"l{li+1}_probe_actual_score"]=scores[li].detach().cpu().numpy()
        data["probe_ids"]=probe_ids.numpy().copy()
        return data

    def after_recycle(self,index,tc,task,P,X,act,masks,scores,probe_ids,before):
        after=unit_metrics(P,X,act)
        from src import relu_doors_0919 as E
        ids=probe_ids.to(X.device)
        probe=X[torch.arange(X.shape[0],device=X.device)[:,None],ids]
        probe_values=E.forward(P,probe,act,train=False)
        post_scores=[]
        for h in (probe_values[1],probe_values[3]):
            a=h.abs().mean(1);post_scores.append((a/(a.mean(1,keepdim=True)+1e-9)).detach().cpu().numpy())
        for li in range(2):
            mask=masks[li].numpy()
            dead=after[f"l{li+1}_strict_dead"]
            native_dead=after[f"l{li+1}_native_dead"]
            # Every new generation is evaluated after BOTH layers' full
            # incoming-then-outgoing transaction, not after an individual row.
            self.generation[li][mask]+=1
            self.start[li][mask]=tc
            self.started_alive[li][mask]=~dead[mask]
            self.native_started_alive[li][mask]=~native_dead[mask]
            self.first_death[li][mask]=-1
            self.native_first_death[li][mask]=-1
            self.reason[li][mask]=0
            after[f"l{li+1}_reset_mask"]=mask.copy()
            pre_dead=before[f"l{li+1}_strict_dead"]
            pre_native_dead=before[f"l{li+1}_native_dead"]
            after[f"l{li+1}_revived_dead_unit"]=mask&pre_dead&~dead
            after[f"l{li+1}_failed_revival"]=mask&pre_dead&dead
            after[f"l{li+1}_new_generation_immediately_dead"]=mask&dead
            after[f"l{li+1}_native_revived_dead_unit"]=mask&pre_native_dead&~native_dead
            after[f"l{li+1}_native_failed_revival"]=mask&pre_native_dead&native_dead
            # This effect also applies to original (generation0) units and is
            # independent of recycled-generation lifetime bookkeeping.
            after[f"l{li+1}_unselected_event_death"]=(~mask)&(~pre_dead)&dead
            after[f"l{li+1}_native_unselected_event_death"]=(~mask)&(~pre_native_dead)&native_dead
            after[f"l{li+1}_previous_generation_replaced_while_alive"]=(mask&~pre_dead&
                (before[f"l{li+1}_generation"]>0)&(before[f"l{li+1}_first_observed_death"]<0))
            after[f"l{li+1}_probe_actual_score"]=post_scores[li]
        # Unselected generations can die because of this intervention. Record
        # that observation as event-associated, distinct from between-event.
        self._observe(tc,after,event_associated=True)
        after.update(self._lineage())
        combined={"index":np.asarray(index),"tc":np.asarray(tc),"task":np.asarray(task),
                  "probe_ids":probe_ids.numpy().copy(),
                  **{"pre_"+k:v for k,v in before.items()},**{"post_"+k:v for k,v in after.items()}}
        self._save(f"event_units/e{index:05d}_k{tc:07d}.npz",combined)
        summaries=[]
        for r,seed in enumerate(self.seeds):
            row=dict(arm=self.arm,seed=seed,event=index,tc=tc,task=task,probe_images=int(probe_ids.shape[1]),
                     probe_sha256=digest(probe_ids[r]))
            for li in (1,2):
                for name,value in (("reset_count",after[f"l{li}_reset_mask"]),
                                   ("strict_dead_pre",before[f"l{li}_strict_dead"]),
                                   ("strict_dead_post",after[f"l{li}_strict_dead"]),
                                   ("native_dead_pre",before[f"l{li}_native_dead"]),
                                   ("native_dead_post",after[f"l{li}_native_dead"]),
                                   ("revived",after[f"l{li}_revived_dead_unit"]),
                                   ("failed_revival",after[f"l{li}_failed_revival"]),
                                   ("immediately_dead_new_generation",after[f"l{li}_new_generation_immediately_dead"]),
                                   ("unselected_event_associated_death",after[f"l{li}_unselected_event_death"]),
                                   ("native_unselected_event_associated_death",after[f"l{li}_native_unselected_event_death"]),
                                   ("replaced_while_alive",after[f"l{li}_previous_generation_replaced_while_alive"])):
                    row[f"l{li}_{name}"]=int(value[r].sum())
            summaries.append(row)
        put(self.out/"diagnostic_manifest.json",self.metadata())
        return summaries

    def task_end(self,task,tc,P,X,act):
        if task in self.completed_tasks:return
        data=unit_metrics(P,X,act)
        self._observe(tc,data,event_associated=False)
        data.update(self._lineage())
        for li in (1,2):data[f"l{li}_reset_mask"]=np.zeros_like(data[f"l{li}_strict_dead"])
        data.update(task=np.asarray(task),tc=np.asarray(tc))
        self._save(f"diagnostic_units/t{task:03d}.npz",data)
        for r,seed in enumerate(self.seeds):
            row=dict(arm=self.arm,seed=seed,task=task,tc=tc)
            for li in (1,2):
                for k in ("m","s","r","b","actual_abs_activity","preH_abs_activity","actual_activity_score",
                          "preH_activity_score","strict_g_mean","strict_g_var","native_g_mean","native_g_var","w_norm"):
                    arr=data[f"l{li}_{k}"][r];finite=arr[np.isfinite(arr)]
                    row[f"l{li}_{k}_mean_finite"]=float(finite.mean()) if finite.size else ""
                    row[f"l{li}_{k}_median_finite"]=float(np.median(finite)) if finite.size else ""
                    row[f"l{li}_{k}_nonfinite"]=int((~np.isfinite(arr)).sum())
                sd=data[f"l{li}_s"][r];rr=data[f"l{li}_r"][r]
                row.update({f"l{li}_s_zero":int((sd==0).sum()),f"l{li}_r_posinf":int(np.isposinf(rr).sum()),
                            f"l{li}_r_neginf":int(np.isneginf(rr).sum()),f"l{li}_r_nan":int(np.isnan(rr).sum()),
                            f"l{li}_strict_dead":int(data[f"l{li}_strict_dead"][r].sum()),
                            f"l{li}_native_dead":int(data[f"l{li}_native_dead"][r].sum()),
                            f"l{li}_max_generation":int(self.generation[li-1][r].max()),
                            f"l{li}_current_generation_redeath_observed":int((self.first_death[li-1][r]>=0).sum())})
            self.rows.append(row)
        self.completed_tasks.append(task)
        write_csv(self.out/"diagnostics.csv",self.rows)
        put(self.out/"diagnostic_manifest.json",self.metadata())

    def metadata(self):
        return dict(files=self.files,completed_tasks=self.completed_tasks,
                    s_correction=1,native_derivative_variance_correction=0,
                    reduction_dtype="float64 from native float32 forward",population="fixed complete1200 images/seed; synthetic checks declare smaller shapes",
                    death_definition="primary strict max(z)>0 alive; native clamp autograd status separate",
                    reason_codes={"0":"no observed re-death","1":"between-events observed death; causal drift not inferred",
                                  "2":"event-associated observed death of unselected generation"},
                    censoring="Replaced-while-alive generation is censored, not a re-death. New generations are evaluated after full two-layer recycler.",
                    zero_width="Raw r signed infinities/NaN in NPZ, explicit counts and named finite-only summaries in CSV")
