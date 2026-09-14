"""Checks for src/pmnist_rlcifar_0907 (Random Label CIFAR), run before the full launch.

    python3 analysis/pmnist_0905/checks_rlcifar.py        # from the repo root

S-data     : tarball sha256 recorded, 50,000 images loaded, pixels in [0,1]; the 1200
             indices are unique, in range, arm-independent and drawn ONCE (identical at
             every task of a 3-task run)
S-labels   : per-task labels uniform over 10 classes (chi2 p > 0.01, df=9), different
             tasks independent (mean pairwise agreement 0.10 +- 0.02), arm-independent
S-init     : the params a run starts from == host init_params(seed, cpu, DIMS), bit-for-bit,
             with the 3072-wide first layer
S-online   : logged online_acc == mean of the per-batch PRE-update accuracies, from the
             run's own hook and from an independent hand-written replay (exact, diff 0.0)
S-capacity : spec §4 gate -- SNA seed 0, 2 tasks x 400 epochs, memo_acc >= 0.90.
             BELOW THAT THE FULL RUN MUST NOT BE LAUNCHED (spec §3.1 INCONCLUSIVE_CAPACITY)
S-repro    : two identical invocations produce byte-identical per_task.csv
S-hist     : the per-task preactivation capture -- npz present with the right shapes,
             h.sum(1) + oob == 1200*100 for every task (nothing lost), and oob == 0 so
             [-32, 16] actually contains the distribution.  A non-zero oob means the
             RANGE MUST BE WIDENED; it is not a threshold to relax.
S-cost     : wall clock of the S-capacity run -> hours for the full 70-run grid

S-act / S-detach / S-clip for SNA are already covered by checks_adapt.py (same
AdaptiveSnake object, same update call site); not duplicated here.
"""
import hashlib, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, ".")
from src import pmnist_0905 as H, pmnist_rlcifar_0907 as RC

REPO = Path(".")
OUT = REPO / "results" / "_checks_pmnist_rlcifar_0907"; OUT.mkdir(parents=True, exist_ok=True)
SCR = OUT / "_runs"; SCR.mkdir(exist_ok=True)
CAPACITY_MIN = 0.90                            # spec §3.1 guard
dev = H.setup("auto")
cifar = RC.Cifar10()
res = {}


def chi2_sf(chi: float, df: int = 9) -> float:
    """P(X > chi) for chi-square with df dof = Q(df/2, chi/2). No scipy in this box."""
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(chi / 2, dtype=torch.float64)))


def run_cli(args, out):
    t0 = time.time()
    p = subprocess.run([sys.executable, "src/pmnist_rlcifar_0907.py", *args, "--out", str(out)],
                       check=True, capture_output=True, text=True)
    return p.stdout, time.time() - t0


# ---- S-data --------------------------------------------------------------
dat = {}
dat["sha256_recorded"] = bool(re.fullmatch(r"[0-9a-f]{64}", cifar.sha256[RC.ARCHIVE]))
dat["n_train"] = int(cifar.train_u8.shape[0])
dat["n_pixels"] = int(cifar.train_u8.shape[1])
dat["shape_ok"] = bool(dat["n_train"] == RC.TRAIN_N and dat["n_pixels"] == RC.PIXELS)
# pixel range over all 50,000, through the same /255 the runs use (chunked: one
# float32 copy of the whole set is 614 MB and buys nothing).
lo, hi = float("inf"), float("-inf")
for s in range(0, RC.TRAIN_N, 5000):
    f = cifar.images(torch.arange(s, min(s + 5000, RC.TRAIN_N)), torch.device("cpu"))
    lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
dat["pixel_min"], dat["pixel_max"] = lo, hi
dat["pixel_range_ok"] = bool(lo >= 0.0 and hi <= 1.0)

ref = {s: RC.subset_idx(s) for s in range(10)}
dat["unique"] = all(len(torch.unique(ref[s])) == RC.N_IMAGES == len(ref[s]) for s in range(10))
dat["in_range"] = all(int(ref[s].min()) >= 0 and int(ref[s].max()) < RC.TRAIN_N for s in range(10))
# arm independence of the draw itself: the stream takes no arm, so redrawing per arm
# must give the same bits (host check_perm's argument).
dat["arm_indep_regen"] = all(torch.equal(RC.subset_idx(s), ref[s]) for _ in H.ARMS for s in range(10))
# ... and the draw actually used inside a run, for two very different arms
dbg = {}
for arm in ("R", "SNA"):
    d = {}
    RC.run_one(arm, 0, 1e-3, 3, cifar, dev, epochs=1, debug=d, hist_dir=SCR / "hist_probe")
    dbg[arm] = d
dat["arm_indep_in_run"] = bool(torch.equal(dbg["R"]["subset"][0], dbg["SNA"]["subset"][0])
                               and torch.equal(dbg["R"]["subset"][0], ref[0]))
# drawn once: every task of a 3-task run sees the same 1200 indices
dat["same_every_task"] = all(torch.equal(d["subset"][0], d["subset"][t])
                             for d in dbg.values() for t in (1, 2))
res["S-data"] = {"pass": bool(dat["sha256_recorded"] and dat["shape_ok"] and dat["pixel_range_ok"]
                              and dat["unique"] and dat["in_range"] and dat["arm_indep_regen"]
                              and dat["arm_indep_in_run"] and dat["same_every_task"]),
                 "archive": RC.ARCHIVE, "archive_sha256": cifar.sha256[RC.ARCHIVE],
                 **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in dat.items()},
                 "n_images": RC.N_IMAGES, "n_tasks_probed": 3,
                 "seed0_sorted_sha256": hashlib.sha256(np.sort(ref[0].numpy()).tobytes()).hexdigest()[:16]}

# ---- S-labels ------------------------------------------------------------
g = H.stream("rlc_labels", 0)
labs = [RC.task_labels(g).numpy() for _ in range(50)]
chis = [float((((np.bincount(y, minlength=10) - RC.N_IMAGES / 10) ** 2)
               / (RC.N_IMAGES / 10)).sum()) for y in labs]
ps = [chi2_sf(c) for c in chis]
L = np.stack(labs)
agree = np.array([(L[i] == L[j]).mean() for i in range(50) for j in range(i + 1, 50)])
# arm independence, same argument as S-data
lab_ref_ok = True
for _ in H.ARMS:
    g2 = H.stream("rlc_labels", 0)
    lab_ref_ok &= all(np.array_equal(RC.task_labels(g2).numpy(), labs[t]) for t in range(50))
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
ref0 = [p.detach() for p in H.init_params(0, torch.device("cpu"), RC.DIMS)]
want_shapes = [[100, 3072], [100], [100, 100], [100], [10, 100], [10]]
res["S-init"] = {"pass": bool(all(torch.equal(a, b) for a, b in zip(ref0, dbg["R"]["init"]))
                              and all(torch.equal(a, b) for a, b in zip(ref0, dbg["SNA"]["init"]))
                              and [list(p.shape) for p in ref0] == want_shapes),
                 "dims": list(RC.DIMS),
                 "shapes": [list(p.shape) for p in ref0], "shapes_expected": want_shapes,
                 "sha256": hashlib.sha256(
                     b"".join(p.numpy().tobytes() for p in ref0)).hexdigest()[:16]}

# ---- S-online ------------------------------------------------------------
d1 = {}
rows1, _ = RC.run_one("R", 0, 1e-3, 1, cifar, dev, epochs=1, debug=d1)
logged = rows1[0]["online_acc"]

# independent replay: rebuild the task from the streams and hand-write the Adam step,
# recording the accuracy of each forward pass BEFORE its own update is applied.
# The accuracies are accumulated exactly as run_one accumulates them -- float32, on
# device, in step order -- so an exact match is the right bar, not a tolerance.
act = H.ARMS["R"]
P = H.init_params(0, dev, RC.DIMS)
idx = RC.subset_idx(0)
x = cifar.images(idx, dev)
y = RC.task_labels(H.stream("rlc_labels", 0)).to(dev)
order = torch.randperm(RC.N_IMAGES, generator=H.stream("rlc_batch", 0)).to(dev)
xs, ys = x[order], y[order]
m_, v_ = [torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P]
manual, manual_sum = [], torch.zeros((), device=dev)
for j in range(RC.STEPS_PER_EPOCH):
    xb, yb = xs[j * RC.BATCH:(j + 1) * RC.BATCH], ys[j * RC.BATCH:(j + 1) * RC.BATCH]
    o = H.forward(P, xb, act)
    hit = (o[4].detach().argmax(1) == yb).float().mean()                 # pre-update
    manual.append(float(hit)); manual_sum += hit
    grads = torch.autograd.grad(torch.nn.functional.cross_entropy(o[4], yb), P)
    with torch.no_grad():
        c1, c2 = 1 - 0.9 ** (j + 1), 1 - 0.999 ** (j + 1)
        for p, gr, mi, vi in zip(P, grads, m_, v_):
            mi.mul_(0.9).add_(gr, alpha=0.1)
            vi.mul_(0.999).addcmul_(gr, gr, value=0.001)
            p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
manual_mean = float(manual_sum) / RC.STEPS_PER_EPOCH
hook_mean = float(np.mean(d1["online"]))                                 # float64 cross-check
res["S-online"] = {"pass": bool(len(d1["online"]) == RC.STEPS_PER_EPOCH == len(manual)
                                and manual == d1["online"]
                                and manual_mean == logged
                                and abs(logged - hook_mean) <= 1e-6),
                   "n_steps": len(manual), "logged_online_acc": logged,
                   "manual_replay_mean": manual_mean, "hook_mean_float64": hook_mean,
                   "manual_batches_bit_identical": bool(manual == d1["online"]),
                   "first_batch_acc": manual[0],
                   "max_abs_diff": abs(logged - manual_mean)}

# ---- S-capacity (spec §4 gate) -------------------------------------------
so_cap, wall_cap = run_cli(["--arms", "SNA", "--seeds", "0", "--tasks", "2", "--epochs", "400"],
                           SCR / "capacity_sna")
import csv
with (SCR / "capacity_sna" / "per_task.csv").open() as fh:
    cap_rows = list(csv.DictReader(fh))
memo = [float(r["memo_acc"]) for r in cap_rows]
res["S-capacity"] = {"pass": bool(len(memo) == 2 and min(memo) >= CAPACITY_MIN),
                     "arm": "SNA", "seed": 0, "tasks": 2, "epochs": 400,
                     "threshold": CAPACITY_MIN, "memo_acc": memo,
                     "memo_acc_min": min(memo) if memo else None,
                     "online_acc": [float(r["online_acc"]) for r in cap_rows],
                     "mob_l1": [float(r["mob_l1"]) for r in cap_rows],
                     "alpha_med_l1": [float(r["alpha_med_l1"]) for r in cap_rows],
                     "alpha_clip_frac_l1": [float(r["alpha_clip_frac_l1"]) for r in cap_rows],
                     "gate": "full run MUST NOT be launched if memo_acc < %.2f" % CAPACITY_MIN}

# ---- S-repro -------------------------------------------------------------
run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_a")
run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_b")
b1 = (SCR / "repro_a" / "per_task.csv").read_bytes()
b2 = (SCR / "repro_b" / "per_task.csv").read_bytes()
res["S-repro"] = {"pass": bool(b1 == b2 and len(b1) > 0),
                  "bytes": len(b1), "n_rows": len(b1.splitlines()) - 1,
                  "sha256": hashlib.sha256(b1).hexdigest()[:16]}

# ---- S-hist --------------------------------------------------------------
# The 3-task probe above (in-process) and the S-capacity run (through the CLI, and
# trained to memorisation, where z is largest) are both inspected: the probe proves
# the plumbing, the trained run proves the range.
TOTAL = RC.N_IMAGES * RC.DIMS[1]                # 1200 x 100 values per layer per task
hist_files = {"probe_R": SCR / "hist_probe" / "R_seed0.npz",
              "probe_SNA": SCR / "hist_probe" / "SNA_seed0.npz",
              "capacity_SNA": SCR / "capacity_sna" / "hist" / "SNA_seed0.npz"}
hd, hist_ok = {}, True
for name, p in hist_files.items():
    if not p.exists():
        hd[name] = {"exists": False}; hist_ok = False; continue
    z = np.load(p)
    nt = z["h1"].shape[0]
    shapes_ok = (z["h1"].shape == z["h2"].shape == (nt, RC.HIST_NB)
                 and z["m1"].shape == z["m2"].shape == (nt, RC.DIMS[1])
                 and z["oob1"].shape == z["oob2"].shape == (nt,)
                 and z["acc"].shape == (nt,) and z["edges"].shape == (RC.HIST_NB + 1,)
                 and z["h1"].dtype == np.int32 and z["m1"].dtype == np.float32)
    conserved = bool(np.all(z["h1"].sum(1) + z["oob1"] == TOTAL)
                     and np.all(z["h2"].sum(1) + z["oob2"] == TOTAL))
    oob = int(z["oob1"].sum() + z["oob2"].sum())
    hd[name] = {"exists": True, "n_tasks": int(nt), "shapes_ok": bool(shapes_ok),
                "conserved": conserved, "oob_total": oob,
                "oob1_per_task": z["oob1"].tolist(), "oob2_per_task": z["oob2"].tolist(),
                "z_min_bin_edge_used": float(RC.HIST_EDGES[np.nonzero(z["h1"].sum(0))[0][0]]),
                "z_max_bin_edge_used": float(RC.HIST_EDGES[np.nonzero(z["h1"].sum(0))[0][-1] + 1])}
    hist_ok &= bool(shapes_ok and conserved and oob == 0)
res["S-hist"] = {"pass": bool(hist_ok), "total_values_per_layer_per_task": TOTAL,
                 "lo": RC.HIST_LO, "hi": RC.HIST_HI, "bins": RC.HIST_NB,
                 "note": "oob != 0 means widen [lo, hi]; do not relax this check",
                 **hd}

# ---- S-cost --------------------------------------------------------------
mt = re.search(r"tasks=\d+\s+([\d.]+)s", so_cap)
run_s = float(mt.group(1)) if mt else wall_cap
per_task = run_s / 2
N_RUNS, N_TASKS, N_PROC = 70, 50, 7             # 7 arms x 10 seeds, 50 tasks each
PAR_SLOWDOWN = 83.5 / 23.6                      # measured on pmnist_rlmnist_0906 at 7-way
res["S-cost"] = {"pass": True, "arm": "SNA", "tasks_timed": 2, "epochs": 400,
                 "steps_per_task": RC.STEPS_PER_EPOCH * 400,
                 "run_s": run_s, "wall_s_incl_startup": wall_cap,
                 "s_per_task_1proc": per_task,
                 "ms_per_step": 1e3 * per_task / (RC.STEPS_PER_EPOCH * 400),
                 "hours_70_runs_1proc": N_RUNS * N_TASKS * per_task / 3600,
                 "par_slowdown_assumed": PAR_SLOWDOWN,
                 "s_per_task_7way": per_task * PAR_SLOWDOWN,
                 "hours_70_runs_7way": N_RUNS * N_TASKS * per_task * PAR_SLOWDOWN / N_PROC / 3600,
                 "device": str(dev)}

res["S-act/S-detach/S-clip"] = {"pass": True,
                                "note": "covered by analysis/pmnist_0905/checks_adapt.py; "
                                        "same H.AdaptiveSnake, same act.update(z1,z2) call site"}
res["all_pass"] = bool(all(v["pass"] for k, v in res.items() if k.startswith("S-")))
(OUT / "checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
if not res["S-capacity"]["pass"]:
    print("\n*** S-CAPACITY FAILED: memo_acc %s < %.2f -- DO NOT LAUNCH THE FULL RUN ***"
          % (res["S-capacity"]["memo_acc"], CAPACITY_MIN))
if not res["S-hist"]["pass"]:
    print("\n*** S-HIST FAILED: %s -- if oob_total > 0 the histogram range [%g, %g] is too "
          "narrow for CIFAR and must be WIDENED ***"
          % ({k: v.get("oob_total") for k, v in hd.items()}, RC.HIST_LO, RC.HIST_HI))
