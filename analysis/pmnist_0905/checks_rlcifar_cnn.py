"""Checks for src/rlcifar_cnn_0908 (Random Label CIFAR on Kumar's CNN), spec §5.

    python3 analysis/pmnist_0905/checks_rlcifar_cnn.py        # from the repo root

S-data        : tarball sha256 recorded, 50,000 images loaded, images (1200,3,32,32) in
                [0,1]; the 1200 indices unique, in range, arm-independent and drawn ONCE
S-labels      : labels uniform -- pooled chi2 over all 60,000 (df=9) plus a calibration
                test that the 50 per-task p-values are themselves U(0,1); tasks
                independent (agreement 0.10 +- 0.02); arm-identical.  All by hand, no
                scipy.  See the block itself for why the spec's literal per-task
                criterion is not used as the gate (it fails 39.5% of the time on a
                perfect generator)
S-init       : shapes (16,3,5,5)(16,)(16,16,5,5)(16,)(100,1024)(100,)(100,100)(100,)(10,100)(10,),
                bound U(+-1/sqrt(fan_in)) with conv fan_in = 25*C_in, bit-identical across arms
S-shape       : (N,16,32,32) -> pool (N,16,16,16) -> (N,16,16,16) -> pool (N,16,8,8)
                -> 1024 -> 100 -> 100 -> 10
S-equivariance: rolling the input one pixel along W rolls conv1's post-activation by the
                same pixel.  THE check that alpha is per channel and shared across space;
                a per-position alpha is run as a negative control and must fail it
S-pool-commute: pool(phi(conv x)) == phi(pool(conv x)) for R, LR, SNA (phi monotone)
S-act         : channel-wise phi' vs autograd, conv and fc sites, on a grid that includes
                every zero of 1+sin(2 a z)
S-ema-off     : ChannelSnake(beta=0, c) == the host's fixed Snake at alpha=c, bit-for-bit
                (this is what `* a.reciprocal()` instead of `/ a` buys)
S-detach      : alpha carries no grad; a backward through phi leaves V untouched
S-clip        : alpha inside [lo, hi] at every task of a real run; clip fraction reported
S-online      : logged online_acc == mean of the per-batch PRE-update accuracies, from the
                run's own hook and from an independent hand-written replay (max diff 0.0)
S-capacity    : spec §5 GATE -- SNA seed 0, 2 tasks x 400 epochs, memo_acc >= 0.90.
                BELOW THAT THE FULL RUN MUST NOT BE LAUNCHED
S-hist        : npz shapes/dtypes, h.sum + oob == N*C*H*W per site, oob == 0 on the smoke
                and on the memorised capacity run; and the fixed-grid raw samples are read
                at the coordinates the npz says they are (recomputed from the run's own
                end-of-run weights).  A non-zero oob means the RANGE MUST BE WIDENED
S-div         : a divergent run stops at the bad task, writes the NaN row, records the step
                in provenance and keeps only the histograms it did produce
S-repro       : two identical invocations produce byte-identical per_task.csv
S-cost        : wall clock of the capacity run at 1 process, plus a MEASURED 7-way
                slowdown (7 concurrent 400-epoch tasks) and the per-process GPU
                high-water mark -> hours and memory for the 70-run grid
"""
import csv, hashlib, json, math, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src import pmnist_0905 as H, rlcifar_cnn_0908 as M

REPO = Path(".")
OUT = REPO / "results" / "_checks_rlcifar_cnn_0908"; OUT.mkdir(parents=True, exist_ok=True)
SCR = OUT / "_runs"; SCR.mkdir(exist_ok=True)
CAPACITY_MIN = 0.90                            # spec §5 gate
dev = H.setup("auto")
cifar = M.RC.Cifar10()
res = {}


def chi2_sf(chi: float, df: int = 9) -> float:
    """P(X > chi) for chi-square with df dof = Q(df/2, chi/2). No scipy in this box."""
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(chi / 2, dtype=torch.float64)))


def run_cli(args, out):
    t0 = time.time()
    p = subprocess.run([sys.executable, "src/rlcifar_cnn_0908.py", *args, "--out", str(out)],
                       capture_output=True, text=True)
    return p, time.time() - t0


# ---- S-data --------------------------------------------------------------
dat = {}
dat["sha256_recorded"] = bool(re.fullmatch(r"[0-9a-f]{64}", cifar.sha256[M.RC.ARCHIVE]))
dat["n_train"] = int(cifar.train_u8.shape[0])
dat["shape_ok"] = bool(dat["n_train"] == M.TRAIN_N and cifar.train_u8.shape[1] == M.RC.PIXELS)
ref = {s: M.subset_idx(s) for s in range(10)}
img0 = M.images(cifar, ref[0], torch.device("cpu"))
dat["image_shape"] = list(img0.shape)
dat["image_shape_ok"] = bool(tuple(img0.shape) == (M.N_IMAGES, *M.IMG))
lo, hi = float("inf"), float("-inf")
for s in range(0, M.TRAIN_N, 5000):            # whole set, chunked: one float32 copy is 614 MB
    f = cifar.images(torch.arange(s, min(s + 5000, M.TRAIN_N)), torch.device("cpu"))
    lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
dat["pixel_min"], dat["pixel_max"] = lo, hi
dat["pixel_range_ok"] = bool(lo >= 0.0 and hi <= 1.0)
dat["unique"] = all(len(torch.unique(ref[s])) == M.N_IMAGES == len(ref[s]) for s in range(10))
dat["in_range"] = all(int(ref[s].min()) >= 0 and int(ref[s].max()) < M.TRAIN_N for s in range(10))
# arm independence of the draw itself: the stream takes no arm, so redrawing per arm
# must give the same bits.
dat["arm_indep_regen"] = all(torch.equal(M.subset_idx(s), ref[s]) for _ in H.ARMS for s in range(10))
# ... and the draw actually used inside a run, for two very different arms
dbg = {}
for arm in ("R", "SNA"):
    d = {}
    M.run_one(arm, 0, 1e-3, 3, cifar, dev, epochs=1, debug=d, hist_dir=SCR / "hist_probe")
    dbg[arm] = d
dat["arm_indep_in_run"] = bool(torch.equal(dbg["R"]["subset"][0], dbg["SNA"]["subset"][0])
                               and torch.equal(dbg["R"]["subset"][0], ref[0]))
dat["same_every_task"] = all(torch.equal(d["subset"][0], d["subset"][t])
                             for d in dbg.values() for t in (1, 2))
# The reshape really does produce (C,H,W).  Checked structurally, not by repeating the
# reshape: in a correctly laid out natural image, neighbours along W are strongly
# correlated.  Reading the same bytes as (H,W,C) drops that to 0.75 while leaving the
# cross-plane correlation high, so the row correlation is the discriminating statistic.
def _corr(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


dat["corr_adjacent_w"] = _corr(img0[:, :, :, :-1], img0[:, :, :, 1:])
dat["corr_adjacent_h"] = _corr(img0[:, :, :-1, :], img0[:, :, 1:, :])
dat["corr_cross_plane"] = _corr(img0[:, 0], img0[:, 1])
dat["plane_layout_ok"] = bool(dat["corr_adjacent_w"] > 0.85 and dat["corr_adjacent_h"] > 0.85
                              and dat["corr_cross_plane"] > 0.70)
res["S-data"] = {"pass": bool(dat["sha256_recorded"] and dat["shape_ok"] and dat["image_shape_ok"]
                              and dat["pixel_range_ok"] and dat["unique"] and dat["in_range"]
                              and dat["arm_indep_regen"] and dat["arm_indep_in_run"]
                              and dat["same_every_task"] and dat["plane_layout_ok"]),
                 "archive": M.RC.ARCHIVE, "archive_sha256": cifar.sha256[M.RC.ARCHIVE],
                 **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in dat.items()},
                 "n_images": M.N_IMAGES, "n_tasks_probed": 3,
                 "seed0_sorted_sha256": hashlib.sha256(np.sort(ref[0].numpy()).tobytes()).hexdigest()[:16]}

# ---- S-labels ------------------------------------------------------------
# DEVIATION from the literal spec §5 wording, stated in the report.  "labels uniform
# (chi2 p > 0.01)" read as "every one of the 50 tasks has p > 0.01" is a mis-sized test:
# it rejects a PERFECT generator with probability 1 - 0.99^50 = 39.5%.  Seed 0 duly has
# one task at p = 0.0083 while the pooled counts over all 60,000 labels sit at p = 0.50.
# The per-task minimum is still reported; the pass condition is the correctly sized pair
# at the same alpha:
#   * pooled -- one chi2 over all 50 x 1200 labels, p > 0.01
#   * calibration -- the 50 per-task p-values are themselves uniform: the number below
#     0.01 inside the exact binomial band (P(X >= observed) > 0.001), and a two-sided
#     Kolmogorov-Smirnov test of the p-values against U(0,1) with p > 0.01
# Both are computed by hand; there is no scipy in this box.
def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def ks_uniform_p(vals) -> tuple:
    """Two-sided KS of `vals` against U(0,1): (D, asymptotic p) by the standard series."""
    n = len(vals); s = sorted(vals)
    d = max(max((i + 1) / n - v, v - i / n) for i, v in enumerate(s))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return d, 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))


g = H.stream("rlcc_labels", 0)
labs = [M.task_labels(g).numpy() for _ in range(50)]
chis = [float((((np.bincount(y, minlength=10) - M.N_IMAGES / 10) ** 2)
               / (M.N_IMAGES / 10)).sum()) for y in labs]
ps = [chi2_sf(c) for c in chis]
pooled = np.bincount(np.concatenate(labs), minlength=10)
exp_p = 50 * M.N_IMAGES / 10
chi_pooled = float((((pooled - exp_p) ** 2) / exp_p).sum())
p_pooled = chi2_sf(chi_pooled)
n_below = int(sum(p < 0.01 for p in ps))
p_count = binom_sf(n_below, 50, 0.01) if n_below else 1.0
ks_d, ks_p = ks_uniform_p(ps)
L = np.stack(labs)
agree = np.array([(L[i] == L[j]).mean() for i in range(50) for j in range(i + 1, 50)])
lab_ref_ok = True
for _ in H.ARMS:
    g2 = H.stream("rlcc_labels", 0)
    lab_ref_ok &= all(np.array_equal(M.task_labels(g2).numpy(), labs[t]) for t in range(50))
lab_run_ok = all(torch.equal(dbg["R"]["labels"][t], dbg["SNA"]["labels"][t]) for t in range(3)) and \
    all(np.array_equal(dbg["R"]["labels"][t].numpy(), labs[t]) for t in range(3))
res["S-labels"] = {"pass": bool(p_pooled > 0.01 and p_count > 0.001 and ks_p > 0.01
                               and abs(agree.mean() - 0.10) <= 0.02
                               and lab_ref_ok and lab_run_ok),
                   "n_tasks": 50, "n_labels_total": int(L.size),
                   "pooled_counts": pooled.tolist(), "pooled_chi2": chi_pooled,
                   "pooled_chi2_p": p_pooled,
                   "per_task_p_below_0.01": n_below,
                   "per_task_p_below_0.01_binom_sf": p_count,
                   "per_task_p_ks_D": ks_d, "per_task_p_ks_p": ks_p,
                   "chi2_max": max(chis), "chi2_p_min": min(ps),
                   "chi2_p_min_task": int(np.argmin(ps)) + 1,
                   "literal_spec_criterion_all_50_p_gt_0.01": bool(min(ps) > 0.01),
                   "literal_criterion_false_failure_rate": 1 - 0.99 ** 50,
                   "pairwise_agreement_mean": float(agree.mean()),
                   "pairwise_agreement_range": [float(agree.min()), float(agree.max())],
                   "n_pairs": int(agree.size),
                   "arm_identical_regen": bool(lab_ref_ok), "arm_identical_in_run": bool(lab_run_ok)}

# ---- S-init --------------------------------------------------------------
ref0 = [p.detach() for p in M.init_params(0, torch.device("cpu"))]
want_shapes = [[16, 3, 5, 5], [16], [16, 16, 5, 5], [16], [100, 1024], [100],
               [100, 100], [100], [10, 100], [10]]
bounds_ok = all(float(p.abs().max()) <= 1.0 / math.sqrt(f) + 1e-12
                for p, f in zip(ref0, [f for f in M.FAN_IN for _ in (0, 1)]))
res["S-init"] = {"pass": bool(all(torch.equal(a, b) for a, b in zip(ref0, dbg["R"]["init"]))
                              and all(torch.equal(a, b) for a, b in zip(ref0, dbg["SNA"]["init"]))
                              and [list(p.shape) for p in ref0] == want_shapes and bounds_ok),
                 "shapes": [list(p.shape) for p in ref0], "shapes_expected": want_shapes,
                 "fan_in": list(M.FAN_IN), "bounds_respected": bool(bounds_ok),
                 "arm_identical": bool(all(torch.equal(a, b) for a, b in zip(dbg["R"]["init"],
                                                                            dbg["SNA"]["init"]))),
                 "sha256": hashlib.sha256(
                     b"".join(p.numpy().tobytes() for p in ref0)).hexdigest()[:16]}

# ---- S-shape -------------------------------------------------------------
# detached: these blocks only read the net, and a graph-carrying output cannot be
# handed to numpy for the exactness comparisons below.
P = [q.detach() for q in M.init_params(0, dev)]
sna = M.ChannelSnake(0.6, 0.01, dev)
xs4 = M.images(cifar, ref[0][:8], dev)
o = M.forward_cnn(P, xs4, sna)
Wc1 = P[0]
p1 = F.max_pool2d(o[1], M.POOL, M.POOL)
p2 = F.max_pool2d(o[3], M.POOL, M.POOL)
got = {"z_c1": list(o[0].shape), "pool1": list(p1.shape), "z_c2": list(o[2].shape),
       "pool2": list(p2.shape), "flat": [8, int(p2.flatten(1).shape[1])],
       "z_f1": list(o[4].shape), "z_f2": list(o[6].shape), "logits": list(o[8].shape)}
want = {"z_c1": [8, 16, 32, 32], "pool1": [8, 16, 16, 16], "z_c2": [8, 16, 16, 16],
        "pool2": [8, 16, 8, 8], "flat": [8, 1024], "z_f1": [8, 100], "z_f2": [8, 100],
        "logits": [8, 10]}
res["S-shape"] = {"pass": bool(got == want), "got": got, "expected": want,
                  "act_order": "conv -> phi -> pool", "sites": list(M.SITES)}

# ---- S-equivariance (new) ------------------------------------------------
# alpha is per channel and shared across space, so phi commutes with a spatial roll.
# Zero padding is not itself circular-equivariant, so the zero-padded form is compared
# on the interior columns only (2..29 -> 3..30, where no kernel window touches a pad);
# the circular-padded form is exact over the whole field and isolates phi.
class _PosSnake(M.ChannelSnake):
    """Negative control: alpha varies with the column index.  Must FAIL the test."""
    def _bcast(self, layer, a):
        if not self.is_conv[layer]:
            return a[None, :]
        w = torch.arange(32, device=a.device, dtype=a.dtype)
        return a[None, :, None, None] * (1.0 + 0.05 * w)[None, None, None, :]


def equivar(act, shift=1):
    x = M.images(cifar, ref[0][:4], dev)
    xr = torch.roll(x, shift, dims=3)
    def head(inp, circ):
        z = F.conv2d(F.pad(inp, (M.PAD,) * 4, mode="circular"), Wc1, P[1]) if circ \
            else F.conv2d(inp, Wc1, P[1], padding=M.PAD)
        return act.phi(z, 0) if isinstance(act, M.ChannelSnake) else act.phi(z)
    a, ar = head(x, False), head(xr, False)
    W = x.shape[3]
    # out_r[w] == out[w-shift] only where NEITHER window touches a zero pad and the roll
    # has not wrapped: w in [PAD+shift, W-1-PAD].
    zero_pad = float((ar[..., M.PAD + shift:W - M.PAD]
                      - a[..., M.PAD:W - M.PAD - shift]).abs().max())
    ac, acr = head(x, True), head(xr, True)
    circ = float((acr - torch.roll(ac, shift, dims=3)).abs().max())
    scale = float(a.abs().max())
    return {"zero_pad_interior_max_abs_diff": zero_pad, "circular_full_max_abs_diff": circ,
            "activation_scale": scale}


eq = {arm: equivar(sna if arm == "SNA" else H.ARMS[arm]) for arm in ("SNA", "R", "LR")}
neg = equivar(_PosSnake(0.6, 0.01, dev))
TOL = 1e-5
eq_pass = all(v["zero_pad_interior_max_abs_diff"] <= TOL and v["circular_full_max_abs_diff"] <= TOL
              for v in eq.values())
neg_fails = bool(neg["circular_full_max_abs_diff"] > TOL)
res["S-equivariance"] = {"pass": bool(eq_pass and neg_fails), "tol": TOL, "shift_px": 1,
                         "per_arm": eq,
                         "negative_control_per_position_alpha": neg,
                         "negative_control_fails_as_required": neg_fails,
                         "note": "zero padding is not circular-equivariant, so that variant is "
                                 "compared only where no kernel window touches a pad and the roll "
                                 "has not wrapped (out cols 3..29 vs 2..28); the circular-padded "
                                 "variant is exact over the whole field"}

# ---- S-pool-commute (new) ------------------------------------------------
pc = {}
for arm in ("R", "LR", "SNA"):
    act = sna if arm == "SNA" else H.ARMS[arm]
    x = M.images(cifar, ref[0][:16], dev)
    z = F.conv2d(x, Wc1, P[1], padding=M.PAD)
    phi = (lambda t: act.phi(t, 0)) if isinstance(act, M.ChannelSnake) else act.phi
    lhs = F.max_pool2d(phi(z), M.POOL, M.POOL)          # conv -> phi -> pool (the run's order)
    rhs = phi(F.max_pool2d(z, M.POOL, M.POOL))          # conv -> pool -> phi
    l, r = lhs.cpu().numpy(), rhs.cpu().numpy()
    pc[arm] = {"allclose_atol0": bool(np.allclose(l, r, atol=0)),
               "bit_identical": bool(np.array_equal(l, r)),
               "max_abs_diff": float(np.abs(l - r).max())}
res["S-pool-commute"] = {"pass": bool(all(v["allclose_atol0"] for v in pc.values())),
                         "per_arm": pc, "criterion": "np.allclose(atol=0)"}

# ---- S-act (channel-wise phi' vs autograd) --------------------------------
eps = float(np.finfo(np.float64).eps)
act_d = {}
for li, tag in ((0, "c1"), (2, "f1")):
    s = M.ChannelSnake(0.6, 0.01, dev)
    torch.manual_seed(0)
    s.V[li] = torch.rand(M.WIDTHS[li], device=dev) * 8 + 0.05     # spread of W across channels
    alpha = s.alpha(li).double()
    zs = torch.linspace(-12, 12, 20001, dtype=torch.float64, device=dev)
    grid = [zs]
    for k in range(-60, 61):                                      # zeros of 1+sin(2 a z)
        grid.append(((-math.pi / 2 + 2 * math.pi * k) / (2 * alpha)).flatten())
    z = torch.cat(grid); z = z[z.abs() <= 12]
    C = M.WIDTHS[li]
    Z = z[:, None].expand(-1, C).clone()
    Z = (Z[:, :, None, None] if M.IS_CONV[li] else Z).requires_grad_(True)
    a = s._bcast(li, alpha)
    phi = Z + torch.sin(a * Z) ** 2 / a
    (gr,) = torch.autograd.grad(phi.sum(), Z)
    ana = s.dphi(Z.detach(), li)
    worst = float((gr - ana).abs().max())
    mask = ana.abs() > 1e-9
    rel = bool(np.allclose(gr[mask].cpu().numpy(), ana[mask].cpu().numpy(), atol=0.0))
    act_d[tag] = {"pass": bool(rel and worst <= 4 * eps), "rel_allclose_atol0": rel,
                  "max_abs_dev_ulp": worst / eps, "n_points": int(z.numel()), "n_channels": C,
                  "conv": bool(M.IS_CONV[li])}
res["S-act"] = {"pass": bool(all(v["pass"] for v in act_d.values())), "per_site": act_d}

# ---- S-ema-off (the `* a.reciprocal()` op order) --------------------------
frozen = M.ChannelSnake(0.6, 0.0, dev)
fixed = H.ARMS["SN06"]
x = M.images(cifar, ref[0][:32], dev)
oa = M.forward_cnn(P, x, frozen)
ob = M.forward_cnn(P, x, fixed)
res["S-ema-off"] = {"pass": bool(all(torch.equal(u, v) for u, v in zip(oa, ob))),
                    "max_abs_logit_diff": float((oa[8] - ob[8]).abs().max()),
                    "note": "beta=0 freezes V at 1, so alpha == c; bit-identity is what "
                            "`* a.reciprocal()` rather than `/ a` buys"}

# ---- S-detach ------------------------------------------------------------
s2 = M.ChannelSnake(0.6, 0.01, dev)
V_before = [v.clone() for v in s2.V]
zz = torch.randn(8, 16, 32, 32, device=dev, requires_grad=True)
s2.phi(zz, 0).sum().backward()
zf = torch.randn(8, 100, device=dev, requires_grad=True)
s2.phi(zf, 2).sum().backward()
res["S-detach"] = {"pass": bool(all(not v.requires_grad for v in s2.V)
                                and all(torch.equal(u, v) for u, v in zip(V_before, s2.V))
                                and zz.grad is not None and zf.grad is not None),
                   "V_requires_grad": [bool(v.requires_grad) for v in s2.V],
                   "V_unchanged": bool(all(torch.equal(u, v) for u, v in zip(V_before, s2.V)))}

# ---- S-capacity (spec §5 GATE) -------------------------------------------
p_cap, wall_cap = run_cli(["--arms", "SNA", "--seeds", "0", "--tasks", "2", "--epochs", "400"],
                          SCR / "capacity_sna")
cap_rows = list(csv.DictReader((SCR / "capacity_sna" / "per_task.csv").open())) \
    if p_cap.returncode == 0 else []
memo = [float(r["memo_acc"]) for r in cap_rows]
res["S-capacity"] = {"pass": bool(len(memo) == 2 and min(memo) >= CAPACITY_MIN),
                     "arm": "SNA", "seed": 0, "tasks": 2, "epochs": 400,
                     "threshold": CAPACITY_MIN, "memo_acc": memo,
                     "memo_acc_min": min(memo) if memo else None,
                     "online_acc": [float(r["online_acc"]) for r in cap_rows],
                     "mob_c1": [float(r["mob_c1"]) for r in cap_rows],
                     "mob_pool_c1": [float(r["mob_pool_c1"]) for r in cap_rows],
                     "mob_f2": [float(r["mob_f2"]) for r in cap_rows],
                     "alpha_med_c1": [float(r["alpha_med_c1"]) for r in cap_rows],
                     "alpha_clip_frac_c1": [float(r["alpha_clip_frac_c1"]) for r in cap_rows],
                     "stderr": p_cap.stderr[-400:] if p_cap.returncode else "",
                     "gate": "full run MUST NOT be launched if memo_acc < %.2f" % CAPACITY_MIN}

# ---- S-clip --------------------------------------------------------------
clip_ok = all(float(r[f"alpha_min_{t}"]) >= 0.005 - 1e-9 and float(r[f"alpha_max_{t}"]) <= 3.0 + 1e-9
              for r in cap_rows for t in M.SITES)
res["S-clip"] = {"pass": bool(clip_ok and cap_rows),
                 "lo": 0.005, "hi": 3.0,
                 "alpha_med_last": {t: float(cap_rows[-1][f"alpha_med_{t}"]) for t in M.SITES} if cap_rows else {},
                 "alpha_clip_frac_last": {t: float(cap_rows[-1][f"alpha_clip_frac_{t}"]) for t in M.SITES} if cap_rows else {},
                 "two_alpha_W_med_last": {t: float(cap_rows[-1][f"two_alpha_W_med_{t}"]) for t in M.SITES} if cap_rows else {}}

# ---- S-online ------------------------------------------------------------
d1 = {}
rows1, _ = M.run_one("R", 0, 1e-3, 1, cifar, dev, epochs=1, debug=d1)
logged = rows1[0]["online_acc"]
# independent replay: rebuild the task from the streams and hand-write the Adam step,
# recording the accuracy of each forward pass BEFORE its own update is applied.  The
# accuracies are accumulated exactly as run_one accumulates them -- float32, on device,
# in step order -- so an exact match is the right bar, not a tolerance.
act = H.ARMS["R"]
Pm = M.init_params(0, dev)
xm = M.images(cifar, M.subset_idx(0), dev)
ym = M.task_labels(H.stream("rlcc_labels", 0)).to(dev)
order = torch.randperm(M.N_IMAGES, generator=H.stream("rlcc_batch", 0)).to(dev)
xsm, ysm = xm[order], ym[order]
m_, v_ = [torch.zeros_like(q) for q in Pm], [torch.zeros_like(q) for q in Pm]
manual, manual_sum = [], torch.zeros((), device=dev)
for j in range(M.STEPS_PER_EPOCH):
    xb, yb = xsm[j * M.BATCH:(j + 1) * M.BATCH], ysm[j * M.BATCH:(j + 1) * M.BATCH]
    ou = M.forward_cnn(Pm, xb, act)
    hit = (ou[8].detach().argmax(1) == yb).float().mean()                # pre-update
    manual.append(float(hit)); manual_sum += hit
    grads = torch.autograd.grad(torch.nn.functional.cross_entropy(ou[8], yb), Pm)
    with torch.no_grad():
        c1, c2 = 1 - 0.9 ** (j + 1), 1 - 0.999 ** (j + 1)
        for p, gr, mi, vi in zip(Pm, grads, m_, v_):
            mi.mul_(0.9).add_(gr, alpha=0.1)
            vi.mul_(0.999).addcmul_(gr, gr, value=0.001)
            p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
manual_mean = float(manual_sum) / M.STEPS_PER_EPOCH
hook_mean = float(np.mean(d1["online"]))                                 # float64 cross-check
res["S-online"] = {"pass": bool(len(d1["online"]) == M.STEPS_PER_EPOCH == len(manual)
                                and manual == d1["online"] and manual_mean == logged
                                and abs(logged - hook_mean) <= 1e-6),
                   "n_steps": len(manual), "logged_online_acc": logged,
                   "manual_replay_mean": manual_mean, "hook_mean_float64": hook_mean,
                   "manual_batches_bit_identical": bool(manual == d1["online"]),
                   "first_batch_acc": manual[0],
                   "max_abs_diff": abs(logged - manual_mean)}

# ---- S-hist --------------------------------------------------------------
# Three sources: the two in-process 3-task probes (plumbing, both arms) and the capacity
# run, which goes through the CLI and is trained to memorisation -- that is where z is
# largest, so it is the one that really tests the range.
TOT = {t: M.N_IMAGES * M.WIDTHS[i] * (32 * 32 if t == "c1" else 16 * 16 if t == "c2" else 1)
       for i, t in enumerate(M.SITES)}
hist_files = {"probe_R": SCR / "hist_probe" / "R_seed0.npz",
              "probe_SNA": SCR / "hist_probe" / "SNA_seed0.npz",
              "capacity_SNA": SCR / "capacity_sna" / "hist" / "SNA_seed0.npz"}
hd, hist_ok = {}, True
for name, p in hist_files.items():
    if not p.exists():
        hd[name] = {"exists": False}; hist_ok = False; continue
    z = np.load(p)
    nt = z["h_c1"].shape[0]
    nk = z["ckpt_tasks"].shape[0]
    shapes_ok = all(z[f"h_{t}"].shape == (nt, M.HIST_NB) and z[f"m_{t}"].shape == (nt, M.WIDTHS[i])
                    and z[f"oob_{t}"].shape == (nt,) and z[f"h_{t}"].dtype == np.int32
                    and z[f"m_{t}"].dtype == np.float32
                    and z[f"s_{t}"].shape == (nk, M.SAMPLE_N, M.WIDTHS[i])
                    and z[f"s_{t}"].dtype == np.float16
                    for i, t in enumerate(M.SITES))
    shapes_ok &= (z["acc"].shape == (nt,) and z["edges"].shape == (M.HIST_NB + 1,)
                  and z["img_idx"].shape == (M.SAMPLE_N,) and z["pos_idx"].shape == (M.SAMPLE_N, 2))
    conserved = all(np.all(z[f"h_{t}"].sum(1) + z[f"oob_{t}"] == TOT[t]) for t in M.SITES)
    oob = int(sum(int(z[f"oob_{t}"].sum()) for t in M.SITES))
    used = np.nonzero(z["h_f2"].sum(0))[0]
    hd[name] = {"exists": True, "n_tasks": int(nt), "n_ckpt": int(nk),
                "ckpt_tasks": z["ckpt_tasks"].tolist(),
                "shapes_ok": bool(shapes_ok), "conserved": bool(conserved), "oob_total": oob,
                "oob_per_site": {t: int(z[f"oob_{t}"].sum()) for t in M.SITES},
                "z_f2_bin_edge_span_used": [float(M.HIST_EDGES[used[0]]),
                                            float(M.HIST_EDGES[used[-1] + 1])]}
    hist_ok &= bool(shapes_ok and conserved and oob == 0)

# the coordinate check: recompute the sampled preactivations from the probe run's own
# end-of-run weights and compare with the last checkpoint stored in its npz.  A silent
# index mix-up would make every later scatter plot wrong, and nothing else would notice.
# The R arm is used: its phi is stateless, so a forward from the run's end-of-run weights
# reproduces exactly the z the npz was written from (SNA would also need its EMA state).
gridd = dbg["R"]["grid"]
coord = {}
zr = np.load(hist_files["probe_R"])
zs_end = M.forward_cnn(dbg["R"]["params_end"], M.images(cifar, ref[0], dev), H.ARMS["R"])[0:8:2]
im = gridd["img_idx"].to(dev); pos = gridd["pos_idx"].to(dev); pos2 = gridd["pos_idx2"].to(dev)
for i, t in enumerate(M.SITES):
    zt = zs_end[i]
    want_v = (zt[im, :, pos[:, 0], pos[:, 1]] if i == 0 else
              zt[im, :, pos2[:, 0], pos2[:, 1]] if i == 1 else zt[im, :])
    got_v = torch.from_numpy(zr[f"s_{t}"][-1].astype(np.float32)).to(dev)
    d16 = float((got_v - want_v).abs().max())
    # float16 has ~3 decimal digits; the bar is the rounding of the value itself
    tolv = float((want_v.abs().max() * 2.0 ** -10).clamp_min(1e-3))
    coord[t] = {"max_abs_diff": d16, "tol_float16": tolv, "ok": bool(d16 <= tolv)}
coord_ok = all(v["ok"] for v in coord.values()) and \
    np.array_equal(zr["img_idx"], gridd["img_idx"].numpy()) and \
    np.array_equal(zr["pos_idx"], gridd["pos_idx"].numpy())
hist_ok &= bool(coord_ok)
res["S-hist"] = {"pass": bool(hist_ok), "values_per_task_per_site": TOT,
                 "lo": M.HIST_LO, "hi": M.HIST_HI, "bins": M.HIST_NB,
                 "bin_width": (M.HIST_HI - M.HIST_LO) / M.HIST_NB,
                 "sample_n": M.SAMPLE_N, "sample_ckpt": list(M.SAMPLE_CKPT),
                 "sample_coordinates_verified": bool(coord_ok), "sample_coord_per_site": coord,
                 "note": "oob != 0 means widen [lo, hi]; do not relax this check",
                 **hd}

# ---- S-div ---------------------------------------------------------------
# (a) through the CLI, from a real blow-up: lr 1e6 makes the loss non-finite inside
#     task 1, so nothing completes and no npz is written.
p_div, _ = run_cli(["--arms", "R", "--seeds", "0", "--tasks", "3", "--epochs", "1",
                    "--lrs", "1e6"], SCR / "div")
drows = list(csv.DictReader((SCR / "div" / "per_task.csv").open()))
dprov = json.loads((SCR / "div" / "provenance.json").read_text())
last_nan = bool(drows) and drows[-1].get("acc", "") == "nan" and drows[-1].get("online_acc", "") == ""
nhist = int(np.load(SCR / "div" / "hist" / "R_seed0.npz")["h_c1"].shape[0]) \
    if (SCR / "div" / "hist" / "R_seed0.npz").exists() else 0
cli_ok = bool(p_div.returncode == 0 and dprov["divergences"] and last_nan
              and len(drows) == int(dprov["divergences"][0]["task"]) and nhist == len(drows) - 1)

# (b) the truncation path proper.  The blow-up above is instant at every lr that
#     produces one, so task 2 is failed deliberately instead: the logits of every
#     training forward from task 2 onward are made non-finite.  One completed task
#     must survive in the csv AND in the npz.
real_fwd, seen = M.forward_cnn, [0]


def _poisoned(params, xx, aa):
    o = real_fwd(params, xx, aa)
    if xx.shape[0] == M.BATCH:
        seen[0] += 1
        if seen[0] > M.STEPS_PER_EPOCH:                # task 1 is 75 training steps
            return (*o[:8], o[8] * float("inf"))
    return o


M.forward_cnn = _poisoned
try:
    trows, tdiv = M.run_one("R", 0, 1e-3, 3, cifar, dev, epochs=1,
                            hist_dir=SCR / "div_trunc" / "hist")
finally:
    M.forward_cnn = real_fwd
tn = int(np.load(SCR / "div_trunc" / "hist" / "R_seed0.npz")["h_c1"].shape[0])
trunc_ok = bool(tdiv["diverged"] and tdiv["task"] == 2 and len(trows) == 2
                and trows[-1]["acc"] != trows[-1]["acc"] and tn == 1
                and "memo_acc" in trows[0])
res["S-div"] = {"pass": bool(cli_ok and trunc_ok),
                "cli": {"pass": cli_ok, "lr": 1e6, "n_rows": len(drows),
                        "diverged_at_task": dprov["divergences"][0]["task"] if dprov["divergences"] else None,
                        "diverged_at_step": dprov["divergences"][0]["step"] if dprov["divergences"] else None,
                        "last_row_is_nan": last_nan, "n_hist_rows_kept": nhist},
                "truncation": {"pass": trunc_ok, "forced_at_task": tdiv["task"],
                               "n_rows": len(trows), "n_hist_rows_kept": tn,
                               "how": "logits forced non-finite from task 2's first training step"},
                "note": "rows truncated at the bad task, NaN row written, step recorded in "
                        "provenance, histograms kept only for the tasks that completed"}

# ---- S-repro -------------------------------------------------------------
run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_a")
run_cli(["--arms", "R", "--seeds", "0", "--tasks", "2", "--epochs", "2"], SCR / "repro_b")
b1 = (SCR / "repro_a" / "per_task.csv").read_bytes()
b2 = (SCR / "repro_b" / "per_task.csv").read_bytes()
res["S-repro"] = {"pass": bool(b1 == b2 and len(b1) > 0),
                  "bytes": len(b1), "n_rows": len(b1.splitlines()) - 1,
                  "sha256": hashlib.sha256(b1).hexdigest()[:16]}

# ---- S-cost --------------------------------------------------------------
mt = re.search(r"tasks=\d+\s+([\d.]+)s", p_cap.stdout)
run_s = float(mt.group(1)) if mt else wall_cap
per_task = run_s / 2
N_RUNS, N_TASKS, N_PROC = 70, 50, 7             # 7 arms x 10 seeds, 50 tasks each
# The 7-way slowdown is MEASURED here rather than carried over from the MLP box: this
# net is launch-bound (a 30,000-step task is ~120 tiny kernels per step), so it does not
# share the MLP's contention profile.  Seven concurrent processes, one 400-epoch task
# each; the slowest is the one that sets the wall clock of a real 7-way launch.
par = [subprocess.Popen([sys.executable, "src/rlcifar_cnn_0908.py", "--arms", "SNA",
                         "--seeds", str(i), "--tasks", "1", "--epochs", "400",
                         "--out", str(SCR / f"par{i}")],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
       for i in range(N_PROC)]
t_par = time.time()
par_out = [p.communicate()[0] for p in par]
par_wall = time.time() - t_par
par_task = [float(m.group(1)) for m in (re.search(r"tasks=\d+\s+([\d.]+)s", o) for o in par_out) if m]
par_s = max(par_task) if par_task else par_wall
slowdown = par_s / per_task
mem = [json.loads((SCR / f"par{i}" / "provenance.json").read_text())["cuda_max_mem_mb"]
       for i in range(N_PROC) if (SCR / f"par{i}" / "provenance.json").exists()]
res["S-cost"] = {"pass": True, "arm": "SNA", "tasks_timed": 2, "epochs": 400,
                 "steps_per_task": M.STEPS_PER_EPOCH * 400,
                 "run_s": run_s, "wall_s_incl_startup": wall_cap,
                 "s_per_task_1proc": per_task,
                 "ms_per_step": 1e3 * per_task / (M.STEPS_PER_EPOCH * 400),
                 "hours_70_runs_1proc": N_RUNS * N_TASKS * per_task / 3600,
                 "par_procs": N_PROC, "par_wall_s": par_wall,
                 "par_s_per_task_measured": par_s, "par_slowdown_measured": slowdown,
                 "par_throughput_gain_vs_1proc": N_PROC / slowdown,
                 "hours_70_runs_7way": N_RUNS * N_TASKS * par_s / N_PROC / 3600,
                 # stage 0 is 4 lambda x 3 seeds x 10 tasks.  Only the 1-process figure is
                 # measured; the 4-way bound reuses the 7-way slowdown, which overstates
                 # contention at 4 processes, so it is an upper bound (and, since the
                 # slowdown exceeds 4, it says 4-way buys nothing here).
                 "hours_stage0_1proc": 4 * 3 * 10 * per_task / 3600,
                 "hours_stage0_4way_upper_bound": 4 * 3 * 10 * per_task * slowdown / 4 / 3600,
                 "cuda_max_mem_mb_per_proc": mem,
                 "cuda_max_mem_mb_7way_total": sum(mem) if mem else None,
                 "device": str(dev)}

res["all_pass"] = bool(all(v["pass"] for k, v in res.items() if k.startswith("S-")))
(OUT / "checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
if not res["S-capacity"]["pass"]:
    print("\n*** S-CAPACITY FAILED: memo_acc %s < %.2f -- DO NOT LAUNCH THE FULL RUN ***"
          % (res["S-capacity"]["memo_acc"], CAPACITY_MIN))
if not res["S-hist"]["pass"]:
    print("\n*** S-HIST FAILED: %s -- if oob_total > 0 the histogram range [%g, %g] is too "
          "narrow and must be WIDENED ***"
          % ({k: v.get("oob_total") for k, v in hd.items()}, M.HIST_LO, M.HIST_HI))
