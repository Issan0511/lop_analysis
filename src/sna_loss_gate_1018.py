#!/usr/bin/env python3
"""RC CNN Snake: slow variance adaptation when current training CE exceeds task-1 CE.

See specs/spec_sna_loss_gate_1018.md. Example (performance run, not a check):
  python src/sna_loss_gate_1018.py --seeds 10-19 --tasks 30 --epochs 400
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import sna_cnn_cause_1009 as E
import numpy as np
import torch

EXPERIMENT = "sna_loss_gate_1018"
ARMS = ("SNA", "SNAfrz1", "SNA_lossgate")
EPS = 1e-4


class LossGateSnake(E.BundleSnake):
    EXTRA = ("enabled", "loss_ref", "last_loss", "gate", "loss_sum", "gate_sum")

    def __init__(self, specs, device, **kwargs):
        super().__init__(specs, device, **kwargs)
        # Lists keep the engine's checkpoint/run-slicing format. Every tensor is static:
        # enabling the rule after capture does not leave stale pointers in the graph.
        for key in self.EXTRA:
            value = 1.0 if key in ("loss_ref", "gate") else 0.0
            setattr(self, key, [torch.full((self.R, 1), value, device=device)])

    @torch.no_grad()
    def observe(self, loss):
        loss = loss.detach().reshape(self.R, 1)
        safe = torch.where(torch.isfinite(loss), loss.clamp_min(EPS), torch.ones_like(loss))
        ratio = (self.loss_ref[0] / safe).clamp(max=1.0)
        g = torch.where(self.enabled[0].bool(), ratio, torch.ones_like(ratio))
        g = torch.where(torch.isfinite(loss), g, torch.zeros_like(g))
        self.last_loss[0].copy_(loss)
        self.gate[0].copy_(g)
        self.loss_sum[0].add_(loss)
        self.gate_sum[0].add_(g)

    @torch.no_grad()
    def update(self, zs):
        rate = self.beta * self.gate[0]
        for l, z in enumerate(zs):
            var = self.batch_var(z, l)
            # Keep normal/task-1 paths in the host's exact operation order.
            normal = self.V[l].clone().mul_(1 - self.beta).add_(self.beta * var)
            gated = self.V[l] * (1 - rate) + rate * var
            self.V[l].copy_(torch.where(self.enabled[0].bool(), gated, normal))

    @torch.no_grad()
    def calibrate(self, epoch_losses, runs):
        values = torch.as_tensor(np.asarray(epoch_losses), device=self.V[0].device,
                                 dtype=self.V[0].dtype)
        if values.ndim != 2 or values.shape[1] != self.R or not values.shape[0]:
            raise ValueError("calibration needs [epoch, run] mean training CE")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid calibration CE")
        ref = values[-10:].quantile(0.5, dim=0).clamp_min(EPS)
        for r in runs:
            self.loss_ref[0][r, 0].copy_(ref[r])
            self.enabled[0][r, 0] = 1

    def state(self):
        return super().state() | {k: [t.clone() for t in getattr(self, k)] for k in self.EXTRA}

    @torch.no_grad()
    def load_state(self, st):
        expected = set(super().state()) | set(self.EXTRA)
        if set(st) != expected:
            raise ValueError("loss-gate checkpoint keys do not match")
        for k in expected:
            dsts = getattr(self, k)
            if len(st[k]) != len(dsts) or any(d.shape != s.shape for d, s in zip(dsts, st[k])):
                raise ValueError(f"loss-gate checkpoint shape mismatch: {k}")
        super().load_state(st)
        for k in self.EXTRA:
            for dst, src in zip(getattr(self, k), st[k]):
                dst.copy_(src)


class LossGateBundle(E.Bundle):
    def __init__(self, slots, cifar, device, **kwargs):
        if not slots or any(a not in ARMS for a, _ in slots) or len(set(slots)) != len(slots):
            raise ValueError("use distinct (arm, seed) slots from " + str(ARMS))
        host_slots = [("SNA" if a == "SNA_lossgate" else a, s) for a, s in slots]
        super().__init__(host_slots, cifar, device, **kwargs)
        self.slots = slots
        act = LossGateSnake(self.specs, device, lo=self.act.lo, hi=self.act.hi, beta=self.act.beta)
        E.BundleSnake.load_state(act, self.act.state())
        self.act = act

    def observe_loss(self, lossv):
        self.act.observe(lossv)

    def epoch_body(self):
        self.act.loss_sum[0].zero_()
        self.act.gate_sum[0].zero_()
        super().epoch_body()

    def full_state(self):
        # The host leaves labels/order out because it resumes at task boundaries. Include
        # them here as well so an epoch-boundary checkpoint within a task is self-contained.
        return super().full_state() | {"Y": self.Y.cpu().clone(), "order": self.order.cpu().clone()}

    def load_full_state(self, st):
        super().load_full_state(st)
        self.Y.copy_(st["Y"])
        self.order.copy_(st["order"])


def run(slots, out, tasks, epochs, device, lr=1e-3, graph=True, resume=True, log=print):
    if tasks < 1 or epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("positive tasks, epochs and finite lr required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    cifar = E.RC.Cifar10()
    b = LossGateBundle(slots, cifar, device, lr=lr, epochs=epochs, graph=graph)
    meta = dict(schema=1, run_id=EXPERIMENT, slots=slots, tasks=tasks, epochs=epochs,
                lr=lr, graph=graph, epsilon=EPS, reference_epochs=10,
                data_sha256=cifar.sha256, beta=b.act.beta, lo=b.act.lo, hi=b.act.hi)
    path = out / "state.pt"
    rows, history, first, gits = [], [], 1, [E.git_state()]
    if path.exists():
        if not resume:
            raise ValueError("output already has a state; choose a fresh output")
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["meta"] != meta:
            raise ValueError("checkpoint configuration mismatch")
        b.load_full_state(st["state"])
        rows, history, first = st["rows"], st["history"], st["task"] + 1
        gits = st["gits"] + gits
    elif any(out.iterdir()) and any(p.is_file() for p in out.iterdir()):
        raise ValueError("refusing to overwrite an output without a resumable state")
    prov = meta | dict(gits=gits, torch=torch.__version__, device=str(device),
                       status="RUNNING", scientific_verdict=None)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    for task in range(first, tasks + 1):
        b.new_labels()
        sce, sacc = E.switch_eval(b.P, b.X, b.Y, b.act)
        accs, losses = [], []
        for epoch in range(epochs):
            b.run_epoch()
            mean_loss = (b.act.loss_sum[0][:, 0] / E.SPE).cpu().numpy().copy()
            mean_gate = (b.act.gate_sum[0][:, 0] / E.SPE).cpu().numpy().copy()
            accs.append((b.acc_ep / E.SPE).cpu().numpy().copy())
            losses.append(mean_loss)
            for r, (arm, seed) in enumerate(slots):
                history.append(dict(arm=arm, seed=seed, task=task, epoch=epoch + 1,
                                    mean_ce=float(mean_loss[r]), mean_gate=float(mean_gate[r]),
                                    last_gate=float(b.act.gate[0][r, 0]),
                                    loss_ref=float(b.act.loss_ref[0][r, 0]),
                                    gate_enabled=bool(b.act.enabled[0][r, 0])))
            if b.bad.any() or not np.isfinite(mean_loss).all():
                E.save_atomic(dict(meta=meta, state=b.full_state(), task=task, epoch=epoch + 1),
                              out / "failed_state.pt")
                E.H.write_csv(out / "per_epoch.csv", history)
                prov["status"] = "FAILED_NONFINITE"
                (out / "provenance.json").write_text(json.dumps(prov, indent=2))
                raise FloatingPointError("non-finite training state; saved failed_state.pt")
        ev = E.evaluate(b.P, b.X, b.Y, b.act)
        a = np.asarray(accs)
        for r, (arm, seed) in enumerate(slots):
            rows.append(dict(arm=arm, seed=seed, task=task, online_acc=float(a[:, r].mean()),
                             switch_ce=float(sce[r]), switch_acc=float(sacc[r]),
                             **ev[r]))
        if task == 1:
            b.act.calibrate(losses, [r for r, (arm, _) in enumerate(slots) if arm == "SNA_lossgate"])
            b.act.freeze([r for r, (arm, _) in enumerate(slots) if arm == "SNAfrz1"])
        if task in E.CKPT_TASKS or task == tasks:
            for r, (arm, seed) in enumerate(slots):
                E.save_atomic(b.run_state(r), out / "ckpt" / f"{arm}_seed{seed}_t{task:02d}.pt")
        E.H.write_csv(out / "per_task.csv", rows)
        E.H.write_csv(out / "per_epoch.csv", history)
        E.save_atomic(dict(meta=meta, state=copy.deepcopy(b.full_state()), rows=rows,
                          history=history, task=task, gits=gits), path)
        log(f"task {task}/{tasks}: mean online {a.mean():.5f}", flush=True)
    prov["status"] = "COMPLETED"
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(ARMS), choices=None)
    p.add_argument("--seeds", default="10-19")
    p.add_argument("--tasks", type=int, default=30)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results" / EXPERIMENT / "cnn")
    a = p.parse_args()
    slots = [(arm, s) for arm in a.arms.split(",") for s in E.parse_seeds(a.seeds)]
    run(slots, a.out, a.tasks, a.epochs, E.H.setup(a.device), a.lr, not a.no_graph)


if __name__ == "__main__":
    main()
