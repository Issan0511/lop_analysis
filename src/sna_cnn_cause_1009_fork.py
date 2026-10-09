#!/usr/bin/env python3
"""Fork runs of sna_cnn_cause_1009 from a checkpoint and continue them under another arm.

    python3 src/sna_cnn_cause_1009_fork.py --src results/sna_cnn_cause_1009/A --at 30 \
        --forks SNA:SNAc3,SNAc3:SNA,SNA:SNA,SNAc3:SNAc3 --tasks 10 --out results/sna_cnn_cause_1009/F30

`--forks old:new[@mod],...`: every seed's run of arm `old`, as it stood at the end of task `--at`
(parameters, Adam moments and step count, V), continues for `--tasks` more tasks with the
activation settings of arm `new` (its c / fixed alpha per site; V is carried over, so an
adaptive site's alpha is c_new / sqrt(V) from the first step).  Labels and batch orders
continue the seed's own streams, so old:old reproduces the source run's next tasks up to the
engine's float differences (S-fork in the summary).

Modifiers applied at the fork point (after loading, before the first continued step):
  @f3x<k>   readout weight and bias times k: every logit times k, the argmax unchanged
  @frz      alpha of every site frozen at its fork-point value (c_new / sqrt(V), clipped) for
            the whole continuation: the state is kept, only the tracking of V is switched off
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlcifar_0907 as RC        # noqa: E402
from src import rlcifar_cnn_0908 as CN           # noqa: E402
from src import sna_cnn_cause_1009 as E          # noqa: E402


def fork_bundle(src: Path, at: int, pairs: list[tuple[str, str]], seeds: list[int], device,
                cifar, epochs: int = 400) -> tuple[E.Bundle, list[tuple[str, str, int]]]:
    slots, meta = [], []
    for old, new in pairs:
        new, _, mod = new.partition("@")
        for s in seeds:
            slots.append((new, s))
            meta.append((old, new + (f"@{mod}" if mod else ""), s))
    B = E.Bundle(slots, cifar, device, epochs=epochs, graph=True)
    with torch.no_grad():
        tcs = set()
        for r, (old, new, s) in enumerate(meta):
            st = torch.load(src / "ckpt" / f"{old}_seed{s}_t{at:02d}.pt", map_location="cpu",
                            weights_only=False)
            for dst, srcl in ((B.P, st["P"]), (B.m, st["m"]), (B.v, st["v"])):
                for q, x in zip(dst, srcl):
                    q[r].copy_(x)
            for l in range(4):
                B.act.V[l][r].copy_(st["act"]["V"][l])
            tcs.add(st["tc"])
            mod = new.partition("@")[2]
            if mod.startswith("f3x"):
                k = float(mod[3:])
                B.P[8][r].mul_(k)
                B.P[9][r].mul_(k)
            elif mod == "frz":
                B.act.freeze([r])
            elif mod:
                raise SystemExit(f"unknown fork modifier {mod!r}")
        if len(tcs) != 1:
            raise SystemExit(f"step counts differ across the forked runs: {tcs}")
        B.tc.fill_(tcs.pop())
    # streams: labels drawn `at` times, batch orders `at * epochs` times
    # the source ran 400 epochs per task, whatever `epochs` the fork continues with
    for s in B.useeds:
        for _ in range(at):
            CN.task_labels(B.g_lab[s])
        for _ in range(at * 400):
            torch.randperm(CN.N_IMAGES, generator=B.g_batch[s])
    return B, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--at", type=int, required=True)
    ap.add_argument("--forks", required=True)
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400, help="per continued task (400 = the protocol)")
    ap.add_argument("--trace", default=None,
                    help="comma list of epochs (0 = task start) at which to write trace.csv rows: "
                         "per-layer move, feature drift, and the fit with the task-start conv")
    args = ap.parse_args()
    device = H.setup("cuda")
    cifar = RC.Cifar10()
    pairs = [tuple(p.split(":")) for p in args.forks.split(",")]
    for o, n in pairs:
        E.parse_arm(o); E.parse_arm(n.partition("@")[0])
    seeds = E.parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    B, meta = fork_bundle(Path(args.src), args.at, pairs, seeds, device, cifar, epochs=args.epochs)
    E.evaluate(B.P, B.X, B.Y, B.act)          # reserve the evaluation's memory (see E.run)
    E.switch_eval(B.P, B.X, B.Y, B.act)
    R, epochs = B.R, B.epochs
    trace_at = sorted({int(x) for x in args.trace.split(",")}) if args.trace else []
    trows = []
    g_pr = torch.Generator().manual_seed(4242)
    probe = torch.randperm(CN.N_IMAGES, generator=g_pr)[:300].to(device)
    rows = []
    curves = np.zeros((R, args.tasks, epochs), dtype=np.float32)
    gits = [E.git_state()]
    for k in range(args.tasks):
        t = args.at + 1 + k
        tt = time.time()
        B.new_labels()
        sce, sacc = E.switch_eval(B.P, B.X, B.Y, B.act)
        P0 = [B.P[i].detach().clone() for i in range(10)]
        V0 = [v.clone() for v in B.act.V]
        h0 = None

        @torch.no_grad()
        def trace(e: int) -> None:
            """Read only; the probe goes through in chunks of 50 images (memory)."""
            nonlocal h0
            n_pr = probe.numel()
            hit = torch.zeros(R, device=device)
            hit_old = torch.zeros(R, device=device)
            ce_s = torch.zeros(R, device=device)
            hs = []
            gsum = {2: torch.zeros(R, CN.HIDDEN, device=device), 3: torch.zeros(R, CN.HIDDEN, device=device)}
            zsum = {2: torch.zeros(R, CN.HIDDEN, device=device), 3: torch.zeros(R, CN.HIDDEN, device=device)}
            z2sum = {2: torch.zeros(R, CN.HIDDEN, device=device), 3: torch.zeros(R, CN.HIDDEN, device=device)}
            Vc = [B.act.V[l].clone() for l in (0, 1)]
            for i0 in range(0, n_pr, 50):
                pi = probe[i0:i0 + 50]
                Xp, Yp = B.X[:, pi], B.Y[:, pi]
                o = E.forward(B.P, Xp, B.act)
                b = pi.numel()
                hs.append(torch.nn.functional.max_pool2d(o[3], 2, 2).reshape(b, R, -1).transpose(0, 1))
                hit += (o[8].argmax(-1) == Yp).float().sum(1)
                ce_s += torch.nn.functional.cross_entropy(o[8].reshape(-1, 10), Yp.reshape(-1),
                                                          reduction="none").view(R, -1).sum(1)
                for l in (2, 3):
                    z = o[2 * l]
                    gsum[l] += B.act.dphi(z, l).sum(1)
                    zsum[l] += z.sum(1)
                    z2sum[l] += (z * z).sum(1)
                # current fc on the task-start conv (conv weights and conv alpha state of epoch 0)
                for l in (0, 1):
                    B.act.V[l].copy_(V0[l])
                om = E.forward(P0[:4] + list(B.P[4:]), Xp, B.act)
                for l in (0, 1):
                    B.act.V[l].copy_(Vc[l])
                hit_old += (om[8].argmax(-1) == Yp).float().sum(1)
            h = torch.cat(hs, 1)
            if h0 is None:
                h0 = h.clone()
            dh = (h - h0).flatten(1).norm(dim=1) / h0.flatten(1).norm(dim=1)
            mv = {tag: ((B.P[2 * i] - P0[2 * i]).flatten(1).norm(dim=1)
                        / P0[2 * i].flatten(1).norm(dim=1)) for i, tag in enumerate(CN.WEIGHT_TAGS)}
            gates = {}
            for l, tag in ((2, "f1"), (3, "f2")):
                gates[f"gate_{tag}"] = (gsum[l] / n_pr).mean(1)
                # cpu: median(dim) is not deterministic on CUDA
                gates[f"seat_{tag}"] = (2 * B.act.alpha(l) * zsum[l] / n_pr).cpu().median(1).values
                sd = (z2sum[l] / n_pr - (zsum[l] / n_pr) ** 2).clamp_min(0).sqrt()
                gates[f"spread_{tag}"] = (2 * B.act.alpha(l) * sd).cpu().median(1).values
            for r, (old, new, s_) in enumerate(meta):
                trows.append({"fork": f"{old}>{new}", "seed": s_, "task": t, "epoch": e,
                              "acc": float(hit[r]) / n_pr, "acc_oldconv": float(hit_old[r]) / n_pr,
                              "ce": float(ce_s[r]) / n_pr, "feat_drift": float(dh[r]),
                              **{f"move_{k}": float(v[r]) for k, v in mv.items()},
                              **{k: float(v[r]) for k, v in gates.items()}})

        ep_acc = torch.zeros(R, epochs, device=device)
        if 0 in trace_at:
            trace(0)
        for e in range(epochs):
            B.run_epoch()
            ep_acc[:, e].copy_(B.acc_ep)
            if e + 1 in trace_at:
                trace(e + 1)
        ep_acc /= CN.STEPS_PER_EPOCH
        curves[:, k] = ep_acc.cpu().numpy()
        online = ep_acc.double().mean(1)
        ev = E.evaluate(B.P, B.X, B.Y, B.act)
        with torch.no_grad():                          # relative move of each weight tensor over the task
            rel = {tag: ((B.P[2 * i] - P0[2 * i]).flatten(1).norm(dim=1)
                         / P0[2 * i].flatten(1).norm(dim=1)).cpu() for i, tag in enumerate(CN.WEIGHT_TAGS)}
        for r, (old, new, s) in enumerate(meta):
            cur = curves[r, k]
            hit99 = np.nonzero(cur >= 0.99)[0]
            rows.append({"old": old, "new": new, "fork": f"{old}>{new}", "seed": s, "task": t,
                         "online_acc": float(online[r]), "switch_ce": float(sce[r]),
                         "switch_acc": float(sacc[r]),
                         "ep_first99": int(hit99[0]) + 1 if hit99.size else -1,
                         **{f"move_{k}": float(v[r]) for k, v in rel.items()}, **ev[r]})
        H.write_csv(out / "per_task.csv", rows)
        if trows:
            H.write_csv(out / "trace.csv", trows)
        np.save(out / "curves.npy", curves)
        m = {}
        for r, (old, new, s) in enumerate(meta):
            m.setdefault(f"{old}>{new}", []).append(float(online[r]))
        print(f"[{time.strftime('%T')}] task {t:2d} {time.time() - tt:6.1f}s  "
              + "  ".join(f"{a} {np.mean(v):.4f}" for a, v in m.items()), flush=True)
    import json
    (out / "provenance.json").write_text(json.dumps(
        {"run_id": E.EXPERIMENT + "_fork", "src": args.src, "at": args.at, "forks": pairs,
         "seeds": seeds, "tasks": args.tasks, "gits": gits, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name()}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
