"""Registered CHB decomposition: a guarded extension of the unchanged ReLUDoors host.

The small adapter is compiled from the exact, hash-locked host run function. This
keeps graph warmup/rollback, RNG, evaluation, checkpointing and task order in the
host, while exposing the two changes it has no hook for: frozen optimizer inputs
and update diagnostics. No existing module or module globals are modified.
"""
from __future__ import annotations

import hashlib
import inspect
import torch
from src import relu_doors_0919 as E

HOST_SHA256 = "2478c8339decaee3aacb90ac7aa87f8cff6b49d2da30833c6e3dbea0dfa82da4"
LAM = .1713
NEW_ARMS = ("CH_bfreeze", "CH_wWD", "CH_bWD")
ARMS = ("CH", "CHB", *NEW_ARMS)
DOORS = {**E.DOORS, **{a: dict(c=True, h=True, b="none") for a in NEW_ARMS}}


def frozen_indices(arm):
    return (1, 3) if arm == "CH_bfreeze" else ()


def wd_indices(arm):
    return {"CH_wWD": (0, 2), "CH_bWD": (1, 3)}.get(arm, ())


class Decomposition(E.ReLUDoors):
    def __init__(self, arm, lam=LAM, beta=.01, widths=(100, 100)):
        if arm not in NEW_ARMS or lam != LAM:
            raise ValueError("unregistered intervention or lambda")
        super().__init__("CH", lam, beta, widths)
        self.name = self.arm = arm

    def post_update(self, P, lr):
        for i in wd_indices(self.arm):
            P[i].mul_(1. - lr * self.lam)


def make_act(arm, lam=0., beta=.01):
    if arm in NEW_ARMS:
        return Decomposition(arm, lam, beta)
    return E.make_act(arm, lam, beta)


def build_runner():
    path = inspect.getsourcefile(E)
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != HOST_SHA256:
        raise RuntimeError("host source changed: adapter requires independent requalification")
    source = inspect.getsource(E.run)
    changes = [
        ('P = [q.contiguous().requires_grad_(True) for q in P]',
         'P = [q.contiguous().requires_grad_(i not in frozen_indices(arm)) for i, q in enumerate(P)]'),
        ('grads = torch.autograd.grad(lossv.sum(), P)',
         'free = [i for i in range(len(P)) if i not in frozen_indices(arm)]\n'
         '        free_grads = torch.autograd.grad(lossv.sum(), [P[i] for i in free])\n'
         '        grads = [None] * len(P)\n'
         '        for i, gr in zip(free, free_grads): grads[i] = gr'),
        ('for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):',
         'for pi, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):\n'
         '                if gr is None: continue'),
        ('p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))',
         'if lifecycle is not None and hasattr(lifecycle, "parameter_step"):\n'
         '                    lifecycle.parameter_step(pi, p, mi, vi, inv_c1, inv_c2, lr, arm)\n'
         '                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))'),
        ('act.post_update(P, lr)                 # door B: b1 held at 0, b2 decayed or zeroed',
         'act.post_update(P, lr)                 # exact registered post-Adam policy'),
        ('"rsl_state": act.gen.get_state() if act.stochastic else None,',
         '"extension_state": lifecycle.checkpoint_state() if lifecycle is not None and hasattr(lifecycle, "checkpoint_state") else None,\n'
         '                        "rsl_state": act.gen.get_state() if act.stochastic else None,'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError("host adapter anchor is not unique")
        source = source.replace(old, new)
    env = dict(E.__dict__, make_act=make_act, DOORS=DOORS,
               frozen_indices=frozen_indices)
    exec(compile(source, __file__ + ":guarded_host_run", "exec"), env)
    return env["run"], hashlib.sha256(source.encode()).hexdigest()


run, ADAPTED_RUN_SHA256 = build_runner()
