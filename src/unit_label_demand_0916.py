"""Unmodified parent training; save states for preregistered unit-label diagnostics."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import relu_gelu_silu_rl_0914 as B
from src import l1_push_split_0915 as P
from src import perm_dial_rl_0915 as DIAL

NAME = "unit_label_demand_0916"
ROOT = B.ROOT
OUT = ROOT / "results" / NAME
TASKS = 10
STOPS = (0, 25, 100, 300, 1000, 6000)
MODELS = P.MODELS
RID = [i for i, m in enumerate(MODELS) if m["env"] == "RL"]
RMODELS = [MODELS[i] for i in RID]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


@torch.no_grad()
def save_state(e, path, task, step):
    state = {name: [q[RID].cpu().clone() for q in vals]
             for name, vals in (("p", e.p), ("m", e.m), ("v", e.v))}
    state.update(t=e.t.cpu().clone(), task=task, step=step, models=RMODELS)
    torch.save(state, path)


def check_observer():
    a = B.Engine(RMODELS, "cpu")
    b = B.Engine(RMODELS, "cpu")
    g = torch.Generator().manual_seed(719)
    x = torch.rand(len(RMODELS), 16, 784, generator=g)
    y = F.one_hot(torch.randint(10, (len(RMODELS), 16), generator=g), 10).float()
    a.step(x, y)
    b.step(x, y)
    before = a.snapshot()
    # Exactly the observation operations, without an unnecessary filesystem write.
    observed = [q.cpu().clone() for q in a.p + a.m + a.v + [a.t]]
    B.forward(a.p, x, a.A)
    assert all(torch.equal(q, r) for q, r in zip(before, a.snapshot()))
    a.step(x, y)
    b.step(x, y)
    assert all(torch.equal(q, r) for q, r in zip(a.snapshot(), b.snapshot()))
    assert len(observed) == 19
    return {"state_unchanged": True, "next_step_bit_identical": True}


def train():
    from src.unit_label_demand_0916_report import selftest
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    assert not list(raw.glob("state_*.pt")), "Refusing to overwrite previous training states"
    dirty = git("status", "--porcelain", "--untracked-files=no")
    assert not dirty, f"Commit the preregistration and implementation first: {dirty}"
    g0 = {"parent": B.selftest(), "diagnostic": selftest(), "observer": check_observer()}
    (OUT / "selftest.json").write_text(json.dumps(g0, indent=2))
    print("G0 PASS", flush=True)
    e = B.Engine(MODELS, "cuda")
    seeds = range(3)
    ax = B.read_idx(B.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    ay = B.read_idx(B.DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
    subset = {s: torch.randperm(len(ax), generator=B.stream("rl_subset", s))[:B.N].numpy() for s in seeds}
    xs = {s: torch.tensor(ax[subset[s]], device=e.device) for s in seeds}
    ys = {s: torch.tensor(ay[subset[s]], device=e.device) for s in seeds}
    glabel = {s: B.stream("env_labels_0913", s) for s in seeds}
    labels = torch.stack([torch.stack([torch.randint(10, (B.N,), generator=glabel[s]) for s in seeds])
                          for _ in range(TASKS)])
    torch.save(dict(x=torch.stack([xs[s].cpu() for s in seeds]), labels=labels,
                    subset={s: torch.tensor(v) for s, v in subset.items()}, models=RMODELS), raw / "inputs.pt")
    gperm = {s: B.stream("env_perm_0913", s) for s in seeds}
    gbatch = {s: B.stream("env_batch_0913", s) for s in seeds}
    gpart = {s: B.stream("partial_perm_0915", s) for s in seeds}
    save_state(e, raw / "initial.pt", 2, -1)
    e.capture()
    rows, unitout, orders_out = [], {}, {}
    started = time.monotonic()
    for task in range(1, TASKS + 1):
        epochs = -(-(B.STEPS * B.BATCH) // B.N)
        orders = {s: torch.stack([torch.randperm(B.N, generator=gbatch[s]) for _ in range(epochs)])
                  .reshape(-1)[:B.STEPS * B.BATCH].reshape(B.STEPS, B.BATCH) for s in seeds}
        for s in seeds:
            orders_out[f"t{task}_s{s}"] = orders[s].numpy().astype(np.int16)
        perm = {s: torch.randperm(784, generator=gperm[s]).to(e.device) for s in seeds}
        lab = {s: labels[task - 1, s].to(e.device) for s in seeds}
        idx = {}
        for s in seeds:
            o = torch.randperm(784, generator=gpart[s])
            r = torch.rand(784, generator=gpart[s])
            for k in (107, 784):
                idx[s, k] = DIAL.partial_index(o, r, k).to(e.device)
        orderstack = torch.stack([orders[m["seed"]] for m in MODELS], 1).to(e.device)
        floors = {}
        for j, model in enumerate(MODELS):
            s = model["seed"]
            if model["env"] == "RL":
                e.cx[j].copy_(xs[s]); yy = lab[s]
            elif model["env"] == "PM":
                e.cx[j].copy_(xs[s][:, perm[s]]); yy = ys[s]
            else:
                e.cx[j].copy_(xs[s][:, idx[s, int(model["env"][1:])]]); yy = lab[s]
            e.cy[j].copy_(F.one_hot(yy, 10))
            floors[j] = float(torch.bincount(yy, minlength=10).max()) / B.N
        e.acc.zero_(); e.ce.zero_()
        if task >= 2:
            save_state(e, raw / f"state_t{task:02d}_s0000.pt", task, 0)
        for start in range(0, B.STEPS, B.BLOCK):
            e.indices.copy_(orderstack[start:start + B.BLOCK])
            e.replay_block()
            end = start + B.BLOCK
            if task >= 2 and end in STOPS:
                save_state(e, raw / f"state_t{task:02d}_s{end:04d}.pt", task, end)
        met, units = e.evaluate()
        acc = (e.acc / B.STEPS).cpu().numpy()
        ce = (e.ce / B.STEPS).cpu().numpy()
        assert np.isfinite(ce).all()
        for j, model in enumerate(MODELS):
            rows.append(dict(**model, task=task, floor_acc=floors[j], online_acc=float(acc[j]),
                             online_ce=float(ce[j]), **{k: float(v[j]) for k, v in met.items()}))
        for k, v in units.items():
            unitout[f"{k}_t{task}"] = v
        print(f"TASK {task}/{TASKS} elapsed={time.monotonic()-started:.1f}s", flush=True)
    B.csvwrite(OUT / "rows.csv", rows)
    np.savez_compressed(raw / "training_units.npz", **unitout)
    np.savez_compressed(raw / "batch_orders.npz", **orders_out)
    old = B.read_rows(ROOT / "results/l1_push_split_0915/rows.csv")
    oldmap = {(r["seed"], r["env"], r["act"], r["task"]): r for r in old if r["task"] <= TASKS}
    diffs = {k: max(abs(r[k] - oldmap[r["seed"], r["env"], r["act"], r["task"]][k]) for r in rows)
             for k in ("online_acc", "online_ce", "train_ce")}
    with np.load(ROOT / "results/l1_push_split_0915/units.npz") as reference:
        diffs["cnorm_l1"] = max(float(np.max(np.abs(unitout[f"cnorm_l1_t{t}"] - reference[f"cnorm_l1_t{t}"])))
                                 for t in range(1, TASKS + 1))
    guard = dict(max_abs_differences=diffs, verdict="BIT_IDENTICAL" if max(diffs.values()) == 0 else "TRAJECTORY_DIFFERS")
    (OUT / "trajectory_check.json").write_text(json.dumps(guard, indent=2))
    provenance = dict(git_hash=git("rev-parse", "HEAD"), spec_sha256=B.sha(ROOT / f"specs/spec_{NAME}.md"),
                      code_sha256=B.sha(__file__), parent_code_sha256=B.sha(B.__file__),
                      models=MODELS, measured_models=RMODELS, tasks=TASKS, stops=STOPS,
                      data_sha256={f: B.sha(B.DATA / f) for f in B.DATA_FILES},
                      subset_sha256={s: B.arrsha(v) for s, v in subset.items()},
                      torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                      dtype="float32", tf32=False, deterministic=True, wall_seconds=time.monotonic()-started)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("G1 " + json.dumps(guard), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.selftest:
        from src.unit_label_demand_0916_report import selftest
        print(json.dumps(dict(diagnostic=selftest(), observer=check_observer()), indent=2))
    else:
        train()
