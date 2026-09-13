"""Checks for src/pmnist_rlmnist_0906 (Random Label MNIST), run before the full launch.

    python3 analysis/pmnist_0905/checks_rlmnist.py        # from the repo root

S-subset : the 1200 images are unique, in range, arm-independent, and drawn ONCE
           (identical at every task of a run)
S-labels : per-task labels uniform over 10 classes (chi2 p > 0.01, df=9), different
           tasks independent (mean pairwise agreement 0.10 +- 0.02), arm-independent
S-init   : the params a run starts from == host init_params(seed, cpu), bit-for-bit
S-online : logged online_acc == mean of the per-batch PRE-update accuracies, both from
           the run's own hook and from an independent hand-written replay of the loop
S-repro  : two identical invocations produce byte-identical per_task.csv
S-cost   : wall clock of one real-cost task (SNA, 400 epochs) -> hours for the full grid

S-act / S-detach / S-clip for SNA are already covered by checks_adapt.py (same
AdaptiveSnake object, same update call site); not duplicated here.
"""
import hashlib, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, ".")
from src import pmnist_0905 as H, pmnist_rlmnist_0906 as RL

REPO = Path(".")
OUT = REPO / "results" / "_checks_pmnist_rlmnist_0906"; OUT.mkdir(parents=True, exist_ok=True)
SCR = OUT / "_runs"; SCR.mkdir(exist_ok=True)
dev = H.setup("auto")
mnist = H.Mnist(dev)
res = {}


def chi2_sf(chi: float, df: int = 9) -> float:
    """P(X > chi) for chi-square with df dof = Q(df/2, chi/2). No scipy in this box."""
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(chi / 2, dtype=torch.float64)))


def run_cli(args, out):
    t0 = time.time()
    p = subprocess.run([sys.executable, "src/pmnist_rlmnist_0906.py", *args, "--out", str(out)],
                       check=True, capture_output=True, text=True)
    return p.stdout, time.time() - t0


# ---- S-subset ------------------------------------------------------------
sub = {}
ref = {s: RL.subset_idx(s) for s in range(10)}
sub["unique"] = all(len(torch.unique(ref[s])) == RL.N_IMAGES == len(ref[s]) for s in range(10))
sub["in_range"] = all(int(ref[s].min()) >= 0 and int(ref[s].max()) < RL.TRAIN_N for s in range(10))
# arm independence of the draw itself: the stream takes no arm, so redrawing per arm
# must give the same bits (host check_perm's argument).
sub["arm_indep_regen"] = all(torch.equal(RL.subset_idx(s), ref[s]) for _ in H.ARMS for s in range(10))
# ... and the draw actually used inside a run, for two very different arms
dbg = {}
for arm in ("R", "SNA"):
    d = {}
    RL.run_one(arm, 0, 1e-3, 3, mnist, dev, epochs=1, debug=d)
    dbg[arm] = d
sub["arm_indep_in_run"] = bool(torch.equal(dbg["R"]["subset"][0], dbg["SNA"]["subset"][0])
                               and torch.equal(dbg["R"]["subset"][0], ref[0]))
# drawn once: every task of a 3-task run sees the same 1200 indices
sub["same_every_task"] = all(torch.equal(d["subset"][0], d["subset"][t])
                             for d in dbg.values() for t in (1, 2))
res["S-subset"] = {"pass": bool(all(sub.values())), **{k: bool(v) for k, v in sub.items()},
                   "n_images": RL.N_IMAGES, "n_tasks_probed": 3,
                   "seed0_sorted_sha256": hashlib.sha256(np.sort(ref[0].numpy()).tobytes()).hexdigest()[:16]}

# ---- S-labels ------------------------------------------------------------
g = H.stream("rl_labels", 0)
labs = [RL.task_labels(g).numpy() for _ in range(50)]
chis = [float((((np.bincount(y, minlength=10) - RL.N_IMAGES / 10) ** 2)
               / (RL.N_IMAGES / 10)).sum()) for y in labs]
ps = [chi2_sf(c) for c in chis]
L = np.stack(labs)
agree = np.array([(L[i] == L[j]).mean() for i in range(50) for j in range(i + 1, 50)])
# arm independence, same argument as S-subset
lab_ref_ok = True
for _ in H.ARMS:
    g2 = H.stream("rl_labels", 0)
    lab_ref_ok &= all(np.array_equal(RL.task_labels(g2).numpy(), labs[t]) for t in range(50))
# and the labels a real run uses (arm R vs SNA, first 3 tasks)
lab_run_ok = all(torch.equal(dbg["R"]["labels"][t], dbg["SNA"]["labels"][t]) for t in range(3)) and \
    all(np.array_equal(dbg["R"]["labels"][t].numpy(), labs[t]) for t in range(3))
res["S-labels"] = {"pass": bool(min(ps) > 0.01 and abs(agree.mean() - 0.10) <= 0.02
                               and lab_ref_ok and lab_run_ok),
                   "n_tasks": 50, "chi2_max": max(chis), "chi2_p_min": min(ps),
                   "chi2_p_min_task": int(np.argmin(ps)) + 1,
                   "pairwise_agreement_mean": float(agree.mean()),
                   "pairwise_agreement_range": [float(agree.min()), float(agree.max())],
                   "n_pairs": int(agree.size),
                   "arm_identical_regen": bool(lab_ref_ok), "arm_identical_in_run": bool(lab_run_ok)}

# ---- S-init --------------------------------------------------------------
ref0 = [p.detach() for p in H.init_params(0, torch.device("cpu"))]
res["S-init"] = {"pass": bool(all(torch.equal(a, b) for a, b in zip(ref0, dbg["R"]["init"]))
                              and all(torch.equal(a, b) for a, b in zip(ref0, dbg["SNA"]["init"]))),
                 "shapes": [list(p.shape) for p in ref0],
                 "sha256": hashlib.sha256(
                     b"".join(p.numpy().tobytes() for p in ref0)).hexdigest()[:16]}

# ---- S-online ------------------------------------------------------------
d1 = {}
rows1, _ = RL.run_one("R", 0, 1e-3, 1, mnist, dev, epochs=1, debug=d1)
logged = rows1[0]["online_acc"]
hook_mean = float(np.mean(d1["online"]))

# independent replay: rebuild the task from the streams and hand-write the Adam step,
# recording the accuracy of each forward pass BEFORE its own update is applied.
act = H.ARMS["R"]
P = H.init_params(0, dev)
idx = RL.subset_idx(0).to(dev)
x = mnist.train_x[idx]
y = RL.task_labels(H.stream("rl_labels", 0)).to(dev)
order = torch.randperm(RL.N_IMAGES, generator=H.stream("rl_batch", 0)).to(dev)
xs, ys = x[order], y[order]
m_, v_, manual = [torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P], []
for j in range(RL.STEPS_PER_EPOCH):
    xb, yb = xs[j * RL.BATCH:(j + 1) * RL.BATCH], ys[j * RL.BATCH:(j + 1) * RL.BATCH]
    o = H.forward(P, xb, act)
    manual.append(float((o[4].argmax(1) == yb).float().mean()))      # pre-update
    grads = torch.autograd.grad(torch.nn.functional.cross_entropy(o[4], yb), P)
    with torch.no_grad():
        c1, c2 = 1 - 0.9 ** (j + 1), 1 - 0.999 ** (j + 1)
        for p, gr, mi, vi in zip(P, grads, m_, v_):
            mi.mul_(0.9).add_(gr, alpha=0.1)
            vi.mul_(0.999).addcmul_(gr, gr, value=0.001)
            p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
manual_mean = float(np.mean(manual))
res["S-online"] = {"pass": bool(len(d1["online"]) == RL.STEPS_PER_EPOCH == len(manual)
                                and abs(logged - hook_mean) <= 1e-6
                                and manual == d1["online"]
                                and abs(logged - manual_mean) <= 1e-6),
                   "n_steps": len(manual), "logged_online_acc": logged,
                   "hook_mean": hook_mean, "manual_replay_mean": manual_mean,
                   "manual_batches_bit_identical": bool(manual == d1["online"]),
                   "first_batch_acc": manual[0], "max_abs_diff": max(abs(logged - hook_mean),
                                                                     abs(logged - manual_mean))}

# ---- S-repro -------------------------------------------------------------
a1, _ = run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_a")
a2, _ = run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_b")
b1 = (SCR / "repro_a" / "per_task.csv").read_bytes()
b2 = (SCR / "repro_b" / "per_task.csv").read_bytes()
res["S-repro"] = {"pass": bool(b1 == b2 and len(b1) > 0),
                  "bytes": len(b1), "n_rows": len(b1.splitlines()) - 1,
                  "sha256": hashlib.sha256(b1).hexdigest()[:16]}

# ---- S-cost --------------------------------------------------------------
so, wall = run_cli(["--arms", "SNA", "--seeds", "0", "--tasks", "2", "--epochs", "400"],
                   SCR / "cost_sna")
mt = re.search(r"tasks=\d+\s+([\d.]+)s", so)
run_s = float(mt.group(1)) if mt else wall
per_task = run_s / 2
N_RUNS, N_TASKS = 70, 50                       # 7 arms x 10 seeds, 50 tasks each
res["S-cost"] = {"pass": True, "arm": "SNA", "tasks_timed": 2, "epochs": 400,
                 "steps_per_task": RL.STEPS_PER_EPOCH * 400,
                 "run_s": run_s, "wall_s_incl_startup": wall,
                 "s_per_task": per_task, "ms_per_step": 1e3 * per_task / (RL.STEPS_PER_EPOCH * 400),
                 "hours_70_runs_1proc": N_RUNS * N_TASKS * per_task / 3600,
                 "device": str(dev)}

res["S-act/S-detach/S-clip"] = {"pass": True,
                                "note": "covered by analysis/pmnist_0905/checks_adapt.py; "
                                        "same H.AdaptiveSnake, same act.update(z1,z2) call site"}
res["all_pass"] = bool(all(v["pass"] for k, v in res.items() if k.startswith("S-")))
(OUT / "checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
