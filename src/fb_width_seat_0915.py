"""Preregistered branch runner for ``fb_width_seat_0915``.

The no-hook parent uses the host loop unchanged.  Branch arms share the online
prefix and switch, after recording step 200000, to optional exact-support GD
and/or row-wise input-weight norm projection.
"""
from __future__ import annotations

import argparse, copy, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch

from .common import ROOT, load_config
from . import edge_law_0905 as E
from .gate_dial_0902 import CONFIG as HOST_CONFIG, SanityError, _arm, setup_arm_dial
from .gate_dose import forward_gate, save_checkpoint_gate
from .elu_swamp import grads_centered_elu
from .ratchet_log import full_support_ro
from .mlp2_phase0 import require_omp
from .mlp2_phase1 import NumericDivergenceError

EXPERIMENT = "fb_width_seat_0915"
CONFIG = Path(ROOT) / "configs/fb_width_seat_0915.yaml"
FREE = slice(15, 20); FLIP = slice(0, 15); PERIOD = 10_000


def registered(path=None): return load_config(str(path or CONFIG))


def table(cfg=None):
    return {str(r["name"]): copy.deepcopy(r) for r in (cfg or registered())["arms"]}


def build_cfg(path=None):
    reg = registered(path); c = copy.deepcopy(load_config(str(HOST_CONFIG)))
    ov = reg["common_overrides"]
    c["common"].update(lr_main=float(ov["lr_main"]), seeds=list(ov["seeds"]),
                       generator_offset=int(ov["generator_offset"]))
    c["sanity"]["omp_num_threads"] = int(ov["omp_num_threads"])
    for name, val in reg["activation"].items(): c["activation"][name] = dict(val)
    templ = reg["arm_template"]
    for r in reg["arms"]:
        c["arms"].append(dict(templ, name=r["name"], family=r["family"],
                              activation=r["activation"], dial=r["dial"], u_fr=r.get("u_fr")))
    c["common"]["lop_every"] = int(reg["record"]["every"])
    return c


def _norm(W, sl): return torch.linalg.vector_norm(W[..., sl], dim=-1)


def project_rows(st, ref_free, ref_flip=None, *, mutate=None):
    """Project rows in-place; return zero-denominator counts and side-effect evidence."""
    W = st["net"].Ws[0]
    zf = torch.zeros(W.shape[0],dtype=torch.int64,device=W.device); zi = zf.clone()
    with torch.no_grad():
        cur = _norm(W, FREE)
        if mutate == "moving_reference": ref_free = cur.clone()
        good = cur != 0; zf = (~good).sum(dim=1)
        Wf = W[..., FREE]
        Wf[good] *= (ref_free[good] / cur[good])[..., None]
        if ref_flip is not None or mutate == "project_flip":
            target = ref_flip
            curi = _norm(W, FLIP); goodi = curi != 0; zi = (~goodi).sum(dim=1)
            Wi = W[..., FLIP]; Wi[goodi] *= (target[goodi] / curi[goodi])[..., None]
    return zf, zi


def support_metrics(st):
    with torch.no_grad():
        X = full_support_ro(st["env"]).double()
        mean = st["layer_means"][0].double()
        z = torch.einsum("rhd,prd->prh", st["net"].Ws[0].double(), X-mean[None]) + st["net"].bs[0].double()
        return dict(k_on=(z > 0).sum(0), n_band=(z.abs() < .3).sum(0),
                    unit_all_negative=(z.amax(0) < 0))


class BranchRecorder(E.EdgeRecorder):
    def __init__(self, steps, st):
        super().__init__(steps, st)
        ends = self.steps % PERIOD == 0
        self.branch_steps = self.steps[ends].astype(np.int64)
        self.branch_index = {int(x): i for i,x in enumerate(self.branch_steps)}
        n,R,H = len(self.branch_steps), st["R"], st["hidden"][0]
        self.branch = {k: np.empty((n,R,H), np.float32) for k in
                       ("w_flip_norm","n_band","k_on","unit_all_negative")}
        self.w_flip = np.empty((n,R,H,15), np.float32)
    def __call__(self, st, step):
        try:
            super().__call__(st, step)
        except NumericDivergenceError as exc:
            # Seeds are independent along tensor dimension 0.  Preserve the host
            # detector, quarantine only nonfinite seed slices, and let finite
            # streams continue unchanged (spec: exclude divergent seeds).
            bad=set()
            tensors=list(st["net"].Ws)+list(st["net"].bs)+[st["net"].v,st["net"].c]
            for t in tensors:
                flat=torch.isfinite(t).reshape(t.shape[0],-1).all(1)
                bad.update(torch.where(~flat)[0].cpu().tolist())
            if not bad: raise
            st.setdefault("divergent_seed_indices",set()).update(bad)
            st.setdefault("divergence_events",[]).append(dict(exc.event))
            with torch.no_grad():
                for t in tensors:
                    for ri in bad: t[ri].zero_()
            super().__call__(st, step)
        j = self.branch_index.get(int(step))
        if j is None: return
        m = support_metrics(st); W = st["net"].Ws[0]
        self.branch["w_flip_norm"][j] = _norm(W, FLIP).cpu().numpy()
        for k in ("n_band","k_on","unit_all_negative"):
            self.branch[k][j] = m[k].cpu().numpy().astype(np.float32)
        self.w_flip[j] = W[..., FLIP].detach().cpu().numpy()


def write_logs(out, arm, st, rec):
    paths = E.write_arm_logs_edge(out, arm, st, rec)
    for ri,p in enumerate(paths):
        with np.load(p, allow_pickle=False) as z: payload={k:z[k] for k in z.files}
        payload["layer1_branch_step"] = rec.branch_steps
        payload["layer1_w_flip"] = rec.w_flip[:,ri]
        for k,v in rec.branch.items(): payload["layer1_"+k] = v[:,ri]
        payload["projection_zero_free"] = np.int64(st.get("projection_zero_free",np.zeros(st["R"],int))[ri])
        payload["projection_zero_flip"] = np.int64(st.get("projection_zero_flip",np.zeros(st["R"],int))[ri])
        payload["numeric_divergence"] = np.bool_(ri in st.get("divergent_seed_indices",set()))
        payload["divergence_events"] = np.array(json.dumps(st.get("divergence_events",[]),sort_keys=True))
        np.savez_compressed(p, **payload)
    return paths


def train_branch(st, rec, probes, total, out, checkpoints, hook, mutate=None):
    switch = int(hook["switch_step"]); full = bool(hook["full_batch"]); clamp=hook["clamp"]
    if mutate == "late_switch": switch += PERIOD
    ps=set(map(int,probes)); cs=set(map(int,checkpoints)); net,env,teacher=st["net"],st["env"],st["teacher"]
    ref_free=ref_flip=None; started=time.time()
    st["projection_zero_free"]=np.zeros(st["R"],dtype=np.int64); st["projection_zero_flip"]=np.zeros(st["R"],dtype=np.int64)
    for step in range(total):
        if step in cs: save_checkpoint_gate(st, arm=st["arm"], step=step, outdir=out)
        if step in ps: rec(st,step)
        # The reference is the recorded, pre-env.step state at the branch boundary.
        if step == switch:
            ref_free=_norm(net.Ws[0],FREE).detach().clone(); ref_flip=_norm(net.Ws[0],FLIP).detach().clone()
        x=env.step()
        if step >= switch and full:
            X=full_support_ro(env); Y=teacher(X)
            ins,pres,acts,yhat=E.forward_gate_batch(st,X)
            grads=E.grads_centered_elu_batch(net,ins,pres,acts,yhat-Y)
        else:
            y=teacher(x); ins,pres,acts,yhat=forward_gate(st,x)
            grads=grads_centered_elu(net,ins,pres,acts,yhat-y)
        net.sgd_step_layers(st["lr"],*grads)
        if step >= switch and clamp != "none":
            a,b=project_rows(st,ref_free,ref_flip if (clamp=="free_flip" or mutate=="project_flip") else None,mutate=mutate)
            st["projection_zero_free"]+=a.cpu().numpy(); st["projection_zero_flip"]+=b.cpu().numpy()
    if total in ps: rec(st,total)
    if total in cs: save_checkpoint_gate(st,st["arm"],total,out)
    return time.time()-started


def run_single_arm(arm, steps=None, outdir=None, seeds=None, mutate=None, switch_step=None):
    reg=registered(); row=table(reg)[arm]; cfg=build_cfg(); require_omp(cfg)
    use=list(seeds if seeds is not None else cfg["common"]["seeds"])
    if steps is None and use != list(cfg["common"]["seeds"]): raise SanityError("registered run requires all seeds")
    total=int(steps or row["total_steps"]); out=Path(outdir or Path(ROOT)/reg["output"]["dir"]); out.mkdir(parents=True,exist_ok=True)
    cfg["common"]["seeds"]=use; st=setup_arm_dial(cfg,_arm(cfg,arm),"cpu")
    every=int(reg["record"]["every"]); probes=list(range(0,total+1,every));
    if probes[-1] != total: probes.append(total)
    rec=BranchRecorder(probes,st); cps=[int(x) for x in row.get("checkpoints",[]) if int(x)<=total]
    t=time.time()
    if row.get("hook") is None:
        elapsed=E.train_arm_edge(st,rec,probes,total,out,cps)
    else:
        hook=dict(row["hook"])
        if switch_step is not None: hook["switch_step"] = int(switch_step)
        elapsed=train_branch(st,rec,probes,total,out,cps,hook,mutate)
    write_logs(out,arm,st,rec)
    status=dict(status="COMPLETE",arm=arm,seeds=use,total_steps=total,elapsed_sec=elapsed,
                wall_sec=time.time()-t,projection_zero_free=np.asarray(st.get("projection_zero_free",[])).tolist(),
                projection_zero_flip=np.asarray(st.get("projection_zero_flip",[])).tolist(),divergent_seed_indices=sorted(st.get("divergent_seed_indices",set())),git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip())
    p=out/"arm_status"/f"{arm}_done.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(status,indent=2))
    return status


def main():
    p=argparse.ArgumentParser(); p.add_argument("--arm",required=True); p.add_argument("--steps",type=int); p.add_argument("--outdir",type=Path); p.add_argument("--mutate"); p.add_argument("--switch-step",type=int)
    a=p.parse_args(); print(json.dumps(run_single_arm(a.arm,a.steps,a.outdir,mutate=a.mutate,switch_step=a.switch_step),indent=2))
if __name__ == "__main__": main()
