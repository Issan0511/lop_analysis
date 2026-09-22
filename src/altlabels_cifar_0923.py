#!/usr/bin/env python3
"""altlabels_cifar_0923 -- 2 labels alternating (ABAB) on RL-CIFAR x MLP (spec_altlabels_cifar_0923.md).

The engine is the parent's, `src/rlcifar_mlp_battle_0918.py` run(); this file is only the CLI and
the fork stage.  With `--schedule iid` and no trace/ckpt/stop flag the engine is bit for bit the
parent's (check S1a), so nothing in the battle's results changes meaning.

    # the GPU arms (R = 10: seeds 0-9 x std, one CUDA graph replayed, trace every 100 steps)
    python3 src/altlabels_cifar_0923.py run --arm LR --schedule abab --tasks 50 \
        --hit-every 100 --keep-ckpts --out results/altlabels_cifar_0923/LR_abab

    # a stop chain: seeds run one after another in this process, R = 1 and eager each,
    # every task ending 500 steps after the eval that first sees 1199/1200 right.  cuda, not
    # cpu: check S5b shows cpu float32 moves images across the threshold and the chain drifts.
    python3 src/altlabels_cifar_0923.py chain --arm LR --schedule iid --seeds 0-4 --tasks 50 \
        --device cuda --threads 2 --stop 0.999,500 --out results/altlabels_cifar_0923/LR_iid_stop

    # forks: from each ckpts/t<NN>.pt one bundle per branch, in the main run's own R = 10 layout
    # (--stop defaults to 0.999,500 and implies --hit-every 100; a fork without them is refused)
    python3 src/altlabels_cifar_0923.py fork --src results/altlabels_cifar_0923/LR_abab \
        --t 2,10,20,30,40,48 --branches A,B,C --out results/altlabels_cifar_0923/LR_abab_fork
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402  the engine
from src import pmnist_0905 as H                       # noqa: E402  streams, device setup
from src import pmnist_rlcifar_0907 as RC              # noqa: E402  data layer, task_labels

EXPERIMENT = "altlabels_cifar_0923"
PARENT = B.EXPERIMENT
OUT_ROOT = H.REPO / "results" / EXPERIMENT
# branch C: one stream per fork point, first draw, so (seed, t) -> C is fixed whichever fork
# points are actually run (and independent of the run's own rlc_labels)
FORK_STREAM = "rlc_labels_fork_t{t}"
BRANCHES = ("A", "B", "C", "next")


def fork_labels(seed: int, t: int) -> torch.Tensor:
    """Branch C at fork point t: the first draw of rlc_labels_fork_t<t> for that seed."""
    return RC.task_labels(H.stream(FORK_STREAM.format(t=t), seed))


def iid_labels(seed: int, task: int) -> torch.Tensor:
    """The iid stream's `task`-th draw (branch "next" of an iid run: its own task t+1)."""
    g = H.stream("rlc_labels", seed)
    for _ in range(task):
        y = RC.task_labels(g)
    return y


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# fork (spec §1.5): one bundle per (fork point, branch), in the main run's own layout
# --------------------------------------------------------------------------

def rel(p: Path) -> str:
    """Repo-relative where it can be, absolute otherwise (--src may point anywhere)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(H.REPO.resolve()))
    except ValueError:
        return str(p)


def check_cached(o: Path, fp: dict, stop, hit_every: int, device) -> bool:
    """Is the finished bundle in `o` the one this request asks for?

    A directory with a provenance.json is only reused when every part of the configuration
    that could change its numbers matches -- source run, fork point, branch, parent checkpoint
    digest, labelling digest, stop rule, trace grid, device.  A mismatch is refused, never
    silently regenerated and never silently reused under this request's provenance.
    """
    p = o / "provenance.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    got = {**{k: d.get("fork", {}).get(k) for k in fp},
           "stop": tuple(d.get("stop") or ()), "hit_every": d.get("hit_every"),
           "device": (d.get("device") or "").split(":")[0]}
    want = {**fp, "stop": tuple(stop or ()), "hit_every": hit_every,
            "device": device.type}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise SystemExit(f"{o} already holds a bundle built with another configuration "
                         f"{bad} (got, wanted); move it aside or point --out elsewhere")
    if not (o / "per_task.csv").exists():
        raise SystemExit(f"{o} has a provenance.json but no per_task.csv; it is incomplete")
    return True


def do_fork(a, device) -> None:
    if not a.stop or not a.hit_every:
        raise SystemExit("fork needs --stop ACC,EXTRA and --hit-every N: without them a bundle "
                         "runs the full 30,000 steps and records no hits or stop snapshots")
    src = Path(a.src).resolve()
    prov = json.loads((src / "provenance.json").read_text())
    slots = [(q["seed"], q["cond"]) for q in prov["slots"]]
    seeds, conds = prov["seeds"], prov["conds"]
    lab = np.load(src / "labels.npz") if (src / "labels.npz").exists() else None
    lseeds = list(lab["seeds"]) if lab is not None else []
    ts = B.parse_ints(a.t)
    branches = [q.strip() for q in a.branches.split(",") if q.strip()]
    for br in branches:
        if br not in BRANCHES:
            raise SystemExit(f"unknown branch {br!r}; known: {','.join(BRANCHES)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cifar = RC.Cifar10()
    stop = B.parse_stop(a.stop)
    # keep the rows of fork points / branches this call is not regenerating, so running the
    # branches in separate invocations still leaves one complete forks.csv
    want = {(str(t), br) for t in ts for br in branches}
    rows: list[dict] = ([q for q in csv.DictReader(open(out / "forks.csv"))
                         if (q["t"], q["branch"]) not in want]
                        if (out / "forks.csv").exists() else [])
    t_start = time.time()
    for t in ts:
        ckpt = src / "ckpts" / f"t{t:02d}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing {ckpt} (the main run needs --keep-ckpts)")
        sha = sha256_file(ckpt)
        for br in branches:
            fixed = {}
            for s in seeds:
                if br == "C":
                    fixed[s] = [fork_labels(s, t)]
                elif br == "next":
                    fixed[s] = [iid_labels(s, t + 1)]
                else:
                    if lab is None or br not in lab.files:
                        raise SystemExit(f"{src}/labels.npz has no labelling {br}")
                    fixed[s] = [torch.from_numpy(lab[br][lseeds.index(s)].astype(np.int64))]
            name = f"t{t:02d}_{br}"
            o = out / "_bundles" / name
            fp = {"src": str(src), "t": t, "branch": br, "parent_sha256": sha,
                  "labels_sha256": B.labels_sha256(fixed)}
            done = check_cached(o, fp, stop, a.hit_every, device)
            if not done:
                shutil.rmtree(o, ignore_errors=True)
                B.run(prov["arm"], seeds, conds, 1, prov["epochs_per_task"], device, o,
                      lr=prov["lr"], c=prov["sna_c"], beta=prov["sna_beta"],
                      lo=prov["alpha_lo"], hi=prov["alpha_hi"], cifar=cifar,
                      graph=not a.no_graph, snapshots=True, checkpoint=False,
                      schedule="aaaa", labels_fixed=fixed, hit_every=a.hit_every, stop=stop,
                      stop_snap=f"fork_t{t:02d}_{br}_seed{{seed}}_stop.npz",
                      restore={"path": ckpt}, run_id=EXPERIMENT,
                      extra_prov={"fork": {"src": str(src), "t": t, "branch": br,
                                           "parent_ckpt": str(ckpt), "parent_sha256": sha,
                                           "label_stream": (FORK_STREAM.format(t=t) if br == "C"
                                                            else f"labels.npz:{br}" if br in "ABC"
                                                            else f"rlc_labels draw {t + 1}"),
                                           "labels_sha256": fp["labels_sha256"]}})
            # the parent hash reported for these rows comes from the bundle that was actually
            # trained, never from this request (a reused bundle may predate it)
            bprov = json.loads((o / "provenance.json").read_text())
            bsha = bprov["fork"]["parent_sha256"]
            (out / "snap").mkdir(parents=True, exist_ok=True)
            for p in sorted((o / "snap").glob("fork_*_stop.npz")):   # each slot's own stop point
                shutil.copyfile(p, out / "snap" / p.name)
            by_seed = {int(q["seed"]): q for q in csv.DictReader(open(o / "per_task.csv"))}
            for s, cd in slots:
                q = by_seed.get(s, {})
                # a slot can be dead in the parent checkpoint or diverge inside the bundle: it
                # gets a row with its status and empty snapshot paths, and never stops the export
                status = ("diverged" if (not q or q.get("memo_acc") in (None, "")) else "alive")
                end_src = B.snapshot_path(o, prov["arm"], cd, s, 1)
                end_rel = ""
                if end_src.exists():
                    end_rel = f"snap/fork_t{t:02d}_{br}_seed{s}_end.npz"
                    shutil.copyfile(end_src, out / end_rel)
                stop_rel = f"snap/fork_t{t:02d}_{br}_seed{s}_stop.npz"
                if not (out / stop_rel).exists():
                    stop_rel = ""                        # never reached hit999 + 500
                rows.append({"t": t, "branch": br, "seed": s, "cond": cd, "status": status,
                             "hit99": q.get("hit99", ""), "hit999": q.get("hit999", ""),
                             "stop_step": q.get("stop_step", ""),
                             "bundle_steps": q.get("steps", ""),
                             "correct_at_stop": q.get("correct_at_hit_plus_500", ""),
                             "stop_observed": q.get("hit_plus_500_observed", ""),
                             "min_correct_after_hit": q.get("min_correct_after_hit", ""),
                             "memo_acc": q.get("memo_acc", ""), "online_acc": q.get("online_acc", ""),
                             "tc": q.get("tc", ""), "parent_ckpt": rel(ckpt),
                             "parent_sha256": bsha, "reused": int(bool(done)),
                             "stop_snap": stop_rel, "end_snap": end_rel})
            H.write_csv(out / "forks.csv", rows)
            print(f"[{time.strftime('%T')}] fork {name}: bundle {rows[-1]['bundle_steps']} steps, "
                  f"{sum(1 for q in rows[-len(slots):] if q['status'] != 'alive')} not alive"
                  f"{' (reused)' if done else ''} "
                  f"({(time.time() - t_start) / 60:.1f} min)", flush=True)
    (out / "provenance.json").write_text(json.dumps(
        {"run_id": EXPERIMENT, "stage": "fork", "parent": PARENT, **B.git_state(),
         "src": str(src), "src_run_id": prov.get("run_id"), "src_schedule": prov.get("schedule"),
         "arm": prov["arm"], "seeds": seeds, "conds": conds, "slots": prov["slots"],
         "t": ts, "branches": branches, "schedule": "fork",
         "hit_every": a.hit_every, "keep_ckpts": False,
         "stop": list(stop) if stop else None,
         "labels_sha256": prov.get("labels_sha256"),
         "fork_label_stream": FORK_STREAM, "perm_consumption_rule": B.PERM_RULE,
         "epochs_per_task": prov["epochs_per_task"], "lr": prov["lr"],
         "device": str(device), "torch": torch.__version__,
         "n_rows": len(rows), "wall_clock_s": time.time() - t_start}, indent=2))
    print(f"wrote {out}/forks.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")


# --------------------------------------------------------------------------

def do_chain(a, device) -> None:
    """The stop chains: R = 1 per seed, one seed after another inside this one process."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = B.parse_stop(a.stop)
    seeds = B.parse_ints(a.seeds)
    cifar = RC.Cifar10()
    t0 = time.time()
    done = []
    want = {"arm": a.arm, "schedule": a.schedule, "n_tasks": a.tasks,
            "epochs_per_task": a.epochs, "hit_every": a.hit_every,
            "stop": list(stop) if stop else None, "conds": a.conds.split(",")}
    for s in seeds:
        o = out / f"seed{s}"
        if (o / "provenance.json").exists():
            # a finished seed is only kept when it was run under this configuration, so a
            # chain whose settings changed can never mix old seeds with new ones
            d = json.loads((o / "provenance.json").read_text())
            bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
            if (d.get("device") or "").split(":")[0] != device.type:
                bad["device"] = (d.get("device"), device.type)
            if bad:
                raise SystemExit(f"{o} was run with another configuration {bad} (got, wanted); "
                                 f"move it aside or point --out elsewhere")
            done.append(s)
            continue
        B.run(a.arm, [s], a.conds.split(","), a.tasks, a.epochs, device, o,
              c=a.c, beta=a.beta, lo=a.alpha_lo, hi=a.alpha_hi, cifar=cifar,
              checkpoint=True, resume=not a.no_resume, graph=False,
              schedule=a.schedule, hit_every=a.hit_every, keep_ckpts=a.keep_ckpts,
              stop=stop, run_id=EXPERIMENT)
        done.append(s)
    (out / "provenance.json").write_text(json.dumps(
        {"run_id": EXPERIMENT, "stage": "chain", "parent": PARENT, **B.git_state(),
         "arm": a.arm, "schedule": a.schedule, "seeds": seeds, "conds": a.conds.split(","),
         "n_tasks": a.tasks, "epochs_per_task": a.epochs, "hit_every": a.hit_every,
         "keep_ckpts": a.keep_ckpts, "stop": list(stop) if stop else None,
         "labels_sha256": {str(s): json.loads((out / f"seed{s}" / "provenance.json").read_text())
                           .get("labels_sha256") for s in done},
         "R_per_seed": 1, "engine": "eager, R = 1, one seed at a time",
         "perm_consumption_rule": B.PERM_RULE, "device": str(device), "torch": torch.__version__,
         "seed_dirs": [f"seed{s}" for s in done],
         "wall_clock_s": time.time() - t0}, indent=2))
    print(f"wrote {out}/provenance.json ({len(done)} seeds, {(time.time() - t0) / 60:.1f} min)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "chain", "fork"])
    ap.add_argument("--arm", default="LR", help="LR | SNA | any battle arm")
    ap.add_argument("--schedule", default="iid", choices=list(B.SCHEDULES))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--conds", default="std")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hit-every", type=int, default=0,
                    help="trace eval every N steps -> trace/*.npz and hit99/hit999 (0 = off)")
    ap.add_argument("--keep-ckpts", action="store_true", help="also write ckpts/t<NN>.pt per task")
    ap.add_argument("--stop", default=None, metavar="ACC,EXTRA",
                    help="a slot stops EXTRA steps after the eval that first has ACC right; the "
                         "task ends when every slot has passed (run: R = 1 only)")
    ap.add_argument("--src", default=None, help="fork: the main run's directory")
    ap.add_argument("--t", default="2,10,20,30,40,48", help="fork: the checkpoints to fork from")
    ap.add_argument("--branches", default="A,B,C")
    ap.add_argument("--c", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--alpha-lo", type=float, default=0.005)
    ap.add_argument("--alpha-hi", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="default results/altlabels_cifar_0923/<arm>_<schedule>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    if a.stage == "fork" and not a.stop:    # a fork without the stop rule is never what is wanted
        a.stop = "0.999,500"
        print(f"fork: --stop defaults to {a.stop}", flush=True)
    if a.stop and not a.hit_every:      # the stop rule lives on the trace grid (spec §1.4)
        a.hit_every = 100
        print(f"--stop {a.stop} implies --hit-every {a.hit_every}", flush=True)
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    if a.stage == "fork":
        if not a.src or not a.out:
            raise SystemExit("fork needs --src and --out")
        return do_fork(a, device)
    if a.stage == "chain":
        if not a.out:
            raise SystemExit("chain needs --out")
        if not a.stop:
            raise SystemExit("chain is for the stop chains; pass --stop ACC,EXTRA")
        return do_chain(a, device)
    out = Path(a.out) if a.out else OUT_ROOT / f"{a.arm}_{a.schedule}"
    stop = B.parse_stop(a.stop)
    if stop is not None and len(B.parse_ints(a.seeds)) * len(a.conds.split(",")) != 1:
        raise SystemExit("run --stop wants R == 1 (one seed, one cond); use the chain stage for "
                         "several seeds, or fork for a lockstep bundle")
    B.run(a.arm, B.parse_ints(a.seeds), a.conds.split(","), a.tasks, a.epochs, device, out,
          c=a.c, beta=a.beta, lo=a.alpha_lo, hi=a.alpha_hi,
          checkpoint=True, resume=not a.no_resume,
          graph=(not a.no_graph) and stop is None,
          schedule=a.schedule, hit_every=a.hit_every, keep_ckpts=a.keep_ckpts, stop=stop,
          run_id=EXPERIMENT)


if __name__ == "__main__":
    main()
