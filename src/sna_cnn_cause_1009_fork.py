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
    rows = []
    curves = np.zeros((R, args.tasks, epochs), dtype=np.float32)
    gits = [E.git_state()]
    for k in range(args.tasks):
        t = args.at + 1 + k
        tt = time.time()
        B.new_labels()
        sce, sacc = E.switch_eval(B.P, B.X, B.Y, B.act)
        P0 = [B.P[i].detach().clone() for i in range(10)]
        ep_acc = torch.zeros(R, epochs, device=device)
        for e in range(epochs):
            B.run_epoch()
            ep_acc[:, e].copy_(B.acc_ep)
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
