"""Native random-label CIFAR ReDo x CHB engine.

The unchanged, hash-locked ReLUDoors run function supplies the entire training
and CUDA graph path. Local, count-checked substitutions add an eager real-step
hook and checkpoint its state. No historical module globals are changed.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import torch
from src import relu_doors_0919 as E

RUN = "redo_chb_rlcifar_1018"
ARMS = ("none", "redo", "CHB", "both")
LAM, TAU, PERIOD, BETA = .1713, .1, 1560, .01
HOST_SHA256 = "2478c8339decaee3aacb90ac7aa87f8cff6b49d2da30833c6e3dbea0dfa82da4"
ROOT = Path(__file__).resolve().parents[1]
DOORS = {"none":dict(c=False,h=False,b="none"), "redo":dict(c=False,h=False,b="none"),
         "CHB":dict(c=True,h=True,b="wd"), "both":dict(c=True,h=True,b="wd")}


def source_identity():
    """Content hashes, including the preregistration, without a mutable HEAD pin."""
    from analysis.redo_chb_rlcifar_1018.diagnostics import sha
    paths={Path(inspect.getsourcefile(m)).resolve() for m in (E,E.H,E.RC,E.RC.RL)}
    paths.update((Path(__file__).resolve(),ROOT/"analysis/redo_chb_rlcifar_1018/diagnostics.py",
                  ROOT/"specs/spec_redo_chb_rlcifar_1018.md",ROOT/"specs/spec_six_runs_common_1018.md"))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def verify_checkpoint_file(ck):
    """Verify independently saved bytes and source BEFORE deserializing P/RNG."""
    from analysis.redo_chb_rlcifar_1018.diagnostics import sha
    ck=Path(ck)
    try:
        record=json.loads((ck.parent/"checkpoint_digest.json").read_text())
    except (OSError,ValueError) as exc:
        raise ValueError("CHECK_FAILED: checkpoint has no valid integrity envelope") from exc
    if (record.get("schema_version")!=1 or record.get("checkpoint")!=ck.name or
        record.get("bytes")!=ck.stat().st_size or record.get("sha256")!=sha(ck) or
        record.get("source_identity")!=source_identity()):
        raise ValueError("CHECK_FAILED: checkpoint integrity/source envelope mismatch")
    return record


def make_act(arm, lam=0., beta=BETA):
    if arm not in ARMS:
        raise ValueError("unknown registered arm")
    native = "CHB" if arm in ("CHB", "both") else "ref"
    act = E.ReLUDoors(native, lam if native == "CHB" else 0., beta)
    act.name = act.arm = arm
    return act


@torch.no_grad()
def score_masks(P, probe_X, act, tau=TAU):
    """Actual post-centering hidden outputs, before any recycling, both layers."""
    values = E.forward(P, probe_X, act, train=False)
    scores, masks = [], []
    for h in (values[1], values[3]):
        s = h.abs().mean(1)
        s = s / (s.mean(1, keepdim=True) + 1e-9)
        if not torch.isfinite(s).all():
            raise ValueError("DIVERGED: nonfinite ReDo activity score")
        scores.append(s.detach().clone())
        masks.append((s <= tau).cpu())
    return scores, masks


@torch.no_grad()
def recycle(P, m, v, masks, g_reinit, seeds, generations=None):
    """Existing b5 recycler semantics; bias moments/global clock/EMA untouched.

    Every layer's incoming rows/biases are reset before ANY outgoing column.
    This order matters at W2 rows/columns selected by both layer masks.
    """
    if len(P) != 6 or len(m) != 6 or len(v) != 6 or len(masks) != 2:
        raise ValueError("incomplete recycler state")
    for layer in range(2):
        wi, bi = 2*layer, 2*layer+1
        if tuple(masks[layer].shape) != tuple(P[bi].shape):
            raise ValueError("invalid recycler mask shape")
        for r, seed in enumerate(seeds):
            ids = torch.nonzero(masks[layer][r]).flatten().to(P[wi].device)
            n, fan_in = len(ids), P[wi].shape[-1]
            # Even the zero-row draw follows the inherited generator semantics.
            draw = ((torch.rand((n,fan_in),generator=g_reinit[seed],dtype=torch.float32)*2-1)
                    * (1/math.sqrt(fan_in))).to(P[wi].device)
            P[wi][r,ids,:] = draw
            P[bi][r,ids] = 0
            m[wi][r,ids,:] = 0
            v[wi][r,ids,:] = 0
    for layer in range(2):
        wi = 2*layer+2
        for r,_seed in enumerate(seeds):
            ids = torch.nonzero(masks[layer][r]).flatten().to(P[wi].device)
            P[wi][r,:,ids] = 0
            m[wi][r,:,ids] = 0
            v[wi][r,:,ids] = 0
    if generations is not None:
        for layer in range(2):
            generations[layer].add_(masks[layer].to(generations[layer].device,dtype=torch.int64))
    return masks


class Controller:
    """Only real completed updates reach this controller; never graph warmup."""
    def __init__(self, arm, seeds, conds, out, device, external=None, settings=None):
        from analysis.redo_chb_rlcifar_1018.diagnostics import Diagnostics
        if conds != ["raw"]:
            raise ValueError("registered engine requires exactly one raw condition")
        self.arm, self.seeds, self.out = arm, list(seeds), Path(out)
        self.device, self.external = device, external
        settings = settings or {}
        self.period, self.tau = int(settings.get("period",PERIOD)), float(settings.get("tau",TAU))
        if self.period < 4 or not 0 <= self.tau <= 1:
            raise ValueError("invalid ReDo controller setting")
        self.enabled = arm in ("redo","both")
        self.g_reinit = {s:E.H.stream("b5_redo_reinit",s) for s in self.seeds}
        self.ring = None
        self.cursor = self.count = self.last_tc = self.event_count = 0
        self.observer = Diagnostics(self.out,arm,self.seeds)
        self.events = []

    def checkpoint_state(self):
        return dict(schema_version=1,arm=self.arm,seeds=self.seeds,period=self.period,tau=self.tau,
                    enabled=self.enabled,probe_ring=self.ring.clone(),probe_cursor=self.cursor,
                    probe_count=self.count,last_tc=self.last_tc,event_count=self.event_count,
                    reinit_rng={s:g.get_state() for s,g in self.g_reinit.items()},
                    events=self.events,diagnostics=self.observer.state(),identity=self.identity,
                    external_state=self.external.state() if self.external is not None and hasattr(self.external,"state") else None)

    def verify_checkpoint_file(self,ck):
        self.checkpoint_envelope=verify_checkpoint_file(ck)
        expected=dict(arm=self.arm,seeds=self.seeds,period=self.period,tau=self.tau,enabled=self.enabled)
        if self.checkpoint_envelope.get("controller_config")!=expected:
            raise ValueError("CHECK_FAILED: checkpoint envelope controller configuration")

    def checkpoint_saved(self,ck,c):
        from analysis.redo_chb_rlcifar_1018.diagnostics import put,sha,digest
        # This immediately follows the host's atomic checkpoint replacement,
        # before expensive task sidecars. A crash in this small gap leaves an
        # unbound checkpoint that is refused, never silently blessed on resume.
        if source_identity()!=self.identity["source_sha256"]:
            raise ValueError("CHECK_FAILED: source/spec changed during native trajectory")
        put(Path(ck).parent/"checkpoint_digest.json",dict(schema_version=1,checkpoint=Path(ck).name,
            bytes=Path(ck).stat().st_size,sha256=sha(ck),source_identity=source_identity(),
            identity=self.identity,identity_sha256=digest(self.identity),task=c["t"],tc=c["tc"],
            controller_config=dict(arm=self.arm,seeds=self.seeds,period=self.period,tau=self.tau,enabled=self.enabled)))

    def _restore(self, extension, c):
        from analysis.redo_chb_rlcifar_1018.diagnostics import digest
        expected = dict(schema_version=1,arm=self.arm,seeds=self.seeds,period=self.period,tau=self.tau,enabled=self.enabled)
        if any(extension.get(k) != v for k,v in expected.items()):
            raise ValueError("CHECK_FAILED: checkpoint extension configuration")
        if extension.get("identity") != self.identity:
            raise ValueError("CHECK_FAILED: checkpoint source/data/input identity")
        if (self.checkpoint_envelope["identity"]!=self.identity or
            self.checkpoint_envelope["identity_sha256"]!=digest(self.identity) or
            self.checkpoint_envelope["task"]!=c["st"]["t"] or self.checkpoint_envelope["tc"]!=c["tc"]):
            raise ValueError("CHECK_FAILED: native checkpoint/envelope identity disagreement")
        if extension["last_tc"] != c["tc"]:
            raise ValueError("CHECK_FAILED: ReDo/native checkpoint clock disagreement")
        self.ring = extension["probe_ring"].clone()
        if tuple(self.ring.shape) != (len(self.seeds),4,c["BATCH"] if "BATCH" in c else c["static_idx"].shape[1]):
            raise ValueError("CHECK_FAILED: probe ring shape")
        self.cursor, self.count = extension["probe_cursor"], extension["probe_count"]
        if self.cursor not in range(4) or not 0 <= self.count <= 4:
            raise ValueError("CHECK_FAILED: probe ring position")
        self.last_tc, self.event_count = extension["last_tc"],extension["event_count"]
        if set(extension["reinit_rng"]) != set(self.seeds):
            raise ValueError("CHECK_FAILED: reinit generator identities")
        for s,g in self.g_reinit.items(): g.set_state(extension["reinit_rng"][s])
        self.events = extension["events"]
        self.observer.load_state(extension["diagnostics"])
        if extension.get("external_state") is not None:
            if self.external is None or not hasattr(self.external,"load_state"):
                raise ValueError("CHECK_FAILED: external checkpoint state has no matching observer")
            self.external.load_state(extension["external_state"])
        old = c["st"]
        keys = ("P","m","v","tc","act_state","g_lab","g_batch","alive")
        actual = dict(P=c["P"],m=c["adam_m"],v=c["adam_v"],tc=c["tc"],act_state=c["act"].state(),
                      g_lab={s:g.get_state() for s,g in c["g_lab"].items()},
                      g_batch={s:g.get_state() for s,g in c["g_batch"].items()},alive=c["alive"])
        if digest(actual) != digest({k:old[k] for k in keys}):
            raise ValueError("CHECK_FAILED: native checkpoint continuity before draw")

    def __call__(self,event,c):
        if event == "ready":
            from analysis.redo_chb_rlcifar_1018.diagnostics import digest,put,sha
            self.ring = torch.zeros(len(self.seeds),4,c["static_idx"].shape[1],dtype=torch.int64)
            self.identity=dict(source_sha256=source_identity(),
                data_sha256=c["cifar"].sha256,input_sha256=digest(c["X"]),
                arm=self.arm,seeds=self.seeds,period=self.period,tau=self.tau,
                parameter_shapes=[list(p.shape) for p in c["P"]],input_shape=list(c["X"].shape))
            self.observer.initialize(c["P"])
            if c["resume"] and c.get("st") is not None:
                ext = c["st"].get("extension_state")
                if ext is None: raise ValueError("CHECK_FAILED: missing ReDo checkpoint extension")
                self._restore(ext,c)
                # Recover the task sidecar if interruption occurred after the
                # host checkpoint but before its task_end observer callback.
                if c["st"]["t"] not in self.observer.completed_tasks:
                    self.observer.task_end(c["st"]["t"],c["tc"],c["P"],c["X"],c["act"])
            put(self.out/"continuity.json",dict(native_state_sha256=digest(dict(P=c["P"],m=c["adam_m"],v=c["adam_v"],
                tc=c["tc"],act_state=c["act"].state(),g_lab={s:g.get_state() for s,g in c["g_lab"].items()},
                g_batch={s:g.get_state() for s,g in c["g_batch"].items()})),
                task_first=c["t_first"],probe_sha256=digest(self.ring),reinit_rng_sha256=digest({s:g.get_state() for s,g in self.g_reinit.items()}),
                input_sha256=digest(c["X"]),data_sha256=c["cifar"].sha256))
        elif event == "task_end":
            if c["diverged"]:
                raise ValueError("DIVERGED: native host saved divergent checkpoint")
            self.observer.task_end(c["t"],c["tc"],c["P"],c["X"],c["act"])
        if self.external is not None:
            return bool(self.external(event,c))
        return False

    @torch.no_grad()
    def real_step(self,c):
        from analysis.redo_chb_rlcifar_1018.diagnostics import digest,put,write_csv
        tc = c["tc"]
        if tc != self.last_tc + 1:
            raise ValueError("CHECK_FAILED: real-step clock discontinuity")
        self.last_tc = tc
        ids = c["ORD_cpu"][:,c["j"]*self.ring.shape[2]:(c["j"]+1)*self.ring.shape[2]]
        self.ring[:,self.cursor,:].copy_(ids)
        self.cursor = (self.cursor+1)%4
        self.count = min(4,self.count+1)
        if not self.enabled or tc % self.period:
            return
        if self.count != 4:
            raise ValueError("CHECK_FAILED: scheduled ReDo event lacks four training batches")
        if not all(torch.isfinite(p).all() for p in c["P"]):
            raise ValueError("DIVERGED: nonfinite parameters at recycle event")
        order = list(range(self.cursor,4))+list(range(self.cursor))
        probe_ids = self.ring[:,order,:].reshape(len(self.seeds),-1).to(c["X"].device)
        probe_X = c["X"][torch.arange(len(self.seeds),device=c["X"].device)[:,None],probe_ids]
        scores,masks = score_masks(c["P"],probe_X,c["act"],self.tau)
        self.event_count += 1
        before = self.observer.before_recycle(self.event_count,tc,c["t"],c["P"],c["X"],c["act"],masks,scores,probe_ids.cpu())
        nonparam = digest(dict(tc=tc,bias_m=[c["adam_m"][i] for i in (1,3,5)],
                              bias_v=[c["adam_v"][i] for i in (1,3,5)],act=c["act"].state(),
                              gl={s:g.get_state() for s,g in c["g_lab"].items()},gb={s:g.get_state() for s,g in c["g_batch"].items()}))
        event_context = dict(c,redo_event=dict(index=self.event_count,tc=tc,masks=masks,scores=scores,probe_ids=probe_ids))
        if self.external is not None: self.external("redo_pre",event_context)
        recycle(c["P"],c["adam_m"],c["adam_v"],masks,self.g_reinit,self.seeds)
        after_nonparam = digest(dict(tc=tc,bias_m=[c["adam_m"][i] for i in (1,3,5)],
                              bias_v=[c["adam_v"][i] for i in (1,3,5)],act=c["act"].state(),
                              gl={s:g.get_state() for s,g in c["g_lab"].items()},gb={s:g.get_state() for s,g in c["g_batch"].items()}))
        if nonparam != after_nonparam:
            raise ValueError("CHECK_FAILED: ReDo changed bias moments, EMA, clock or training RNG")
        summary = self.observer.after_recycle(self.event_count,tc,c["t"],c["P"],c["X"],c["act"],masks,scores,probe_ids.cpu(),before)
        self.events.extend(summary)
        write_csv(self.out/"redo_events.csv",self.events)
        put(self.out/"reinit_rng.json",dict(tc=tc,event_count=self.event_count,sha256=digest({s:g.get_state() for s,g in self.g_reinit.items()})))
        if self.external is not None: self.external("redo_post",event_context)


def build_runner(overrides=None, *, native=False):
    """Clone host globals locally; overrides are only for synthetic verification.

    native=True compiles the untouched host function into the same local
    namespace, enabling meaningful alias-for-alias no-ReDo comparison.
    """
    path = inspect.getsourcefile(E)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != HOST_SHA256:
        raise RuntimeError("host source changed; independent requalification required")
    source = inspect.getsource(E.run)
    env = dict(E.__dict__,make_act=make_act,DOORS=DOORS,Controller=Controller,CONTROL_SETTINGS={})
    if overrides: env.update(overrides)
    if not native:
        changes = [
            ('act = make_act(arm, lam, beta)',
             'act = make_act(arm, lam, beta)\n'
             '    lifecycle = Controller(arm,seeds,conds,out,device,external=lifecycle,settings=CONTROL_SETTINGS)'),
            ('ORD = torch.stack([order[s] for s, cd in slots]).to(device)  # (R, 1200)',
             'ORD_cpu = torch.stack([order[s] for s, cd in slots])\n'
             '            ORD = ORD_cpu.to(device)  # same native stack and transfer\n'
             '            lifecycle("epoch_order",locals())'),
            ('Ydev.copy_(Y)', 'Ydev.copy_(Y)\n        lifecycle("task_labels",locals())'),
            ('if debug is not None:\n                    debug.setdefault("online", []).append(last_hit.cpu().clone())',
             'lifecycle.real_step(locals())\n'
             '                if debug is not None:\n                    debug.setdefault("online", []).append(last_hit.cpu().clone())'),
            ('"rsl_state": act.gen.get_state() if act.stochastic else None,',
             '"extension_state": lifecycle.checkpoint_state(),\n'
             '                        "rsl_state": act.gen.get_state() if act.stochastic else None,'),
            ('st = torch.load(ck, map_location="cpu", weights_only=False)   # generator states must stay on the cpu',
             'lifecycle.verify_checkpoint_file(ck)\n'
             '        st = torch.load(ck, map_location="cpu", weights_only=False)   # generator states must stay on the cpu'),
            ('os.replace(tmp, ck)', 'os.replace(tmp, ck)\n            lifecycle.checkpoint_saved(ck,locals())'),
        ]
        for old,new in changes:
            if source.count(old) != 1:
                raise RuntimeError("native adapter anchor missing or ambiguous")
            source = source.replace(old,new)
    exec(compile(source,__file__+(":native" if native else ":adapted"),"exec"),env)
    runner = env["run"]
    runner.adapted_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return runner


run = build_runner()
ADAPTED_RUN_SHA256 = run.adapted_source_sha256
