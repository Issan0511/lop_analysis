"""TASK V (cobirth_seeds_1013, verifier, 1014): extra adversarial checks on the verifier's own counts (verify/vcounts_<tag>.npz).
(a) out-of-sample break-even: per target, n*(p) = mu + s sqrt((1-p)/p) from one window, frequency of N >= n* in the other window,
    both directions (T 11-70 -> 71-100 as in ELU_REPORT, and the reverse T 41-100 -> 11-40), p = 0.14 and 0.08.  In-sample this frequency
    can never exceed the Cantelli value for a fixed n* (the empirical distribution satisfies Cantelli) [exact], so only out-of-sample is a test.
(b) ELU E1 vs E2 sign check of pooled P(N >= 2) and P(N >= 3) against null 1 (independent circular shift, R = 300, own seed 4242).
(c) a0.1 seed 0, all 91 states (nbeta_time files): time-mean of the state-matched per-(i,T) Cantelli per target, its max and Sigma_i.
(d) per-window mean of N (trend) for every condition."""
import os, json, glob, re, math
import numpy as np
V = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013"
TAGS = ["a01s0", "a03s0", "a05s0", "a01s1", "a03s1", "a05s1", "a003s0", "a003s1", "a02s0", "a02s1", "ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2"]
out = {}
def series(N, i, lo, hi):
    r = np.arange(lo - 1, hi); v = N[r, i]; return v[v >= 0].astype(float)
for tag in TAGS:
    N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]; H = N.shape[1]; o = {}
    for name, (fit, test) in {"fwd_11-70_to_71-100": ((11, 70), (71, 100)), "bwd_41-100_to_11-40": ((41, 100), (11, 40))}.items():
        for p in (0.14, 0.08):
            ev = tot = 0; tg_over = 0; nt = 0
            for i in range(H):
                a = series(N, i, *fit); b = series(N, i, *test)
                if len(a) < 15 or len(b) < 15: continue
                ns = a.mean() + a.std(ddof=1) * math.sqrt((1 - p) / p); e = int((b >= ns).sum()); ev += e; tot += len(b); nt += 1; tg_over += int(e / len(b) > p)
            o[f"{name}_p{p:g}"] = dict(pooled_freq=ev / tot, n_targets=nt, targets_freq_gt_p=tg_over)
    win = []
    for lo, hi in ((11, 40), (41, 70), (71, 100)):
        v = N[lo - 1:hi]; win.append(float(v[v >= 0].mean()))
    o["window_mean_N"] = win
    out[tag] = o
    print(tag, "trend", [round(x, 3) for x in win], "| out-of-sample:", {k: (round(v["pooled_freq"], 4), v["targets_freq_gt_p"]) for k, v in o.items() if k != "window_mean_N"})
# (b) ELU sign check of P(N>=2), P(N>=3) vs null 1
rng = np.random.default_rng(4242)
for tag in ["ELU_s0_E1", "ELU_s0_E2", "ELU_s1_E1", "ELU_s1_E2", "a01s0"]:
    z = np.load({"a01s0": f"{V}/../cobirth_1013/m1/counts_a01.npz"}.get(tag, f"{V}/elu/counts_{tag}.npz")); isb = z["isb"].astype(np.int8); N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]
    mats = []
    for i in range(N.shape[1]):
        rows = np.arange(10, 100); rows = rows[N[rows, i] >= 0]
        if len(rows) >= 30: mats.append(np.delete(isb[rows, i, :], i, axis=1))
    obs = np.concatenate([M.sum(1) for M in mats]); o2, o3 = (obs >= 2).mean(), (obs >= 3).mean(); R = 300; g2 = np.empty(R); g3 = np.empty(R)
    for r in range(R):
        allv = []
        for M in mats:
            n, m = M.shape; off = rng.integers(0, n, size=m); idx = (np.arange(n)[:, None] - off[None, :]) % n; allv.append(np.take_along_axis(M, idx, axis=0).sum(1))
        allv = np.concatenate(allv); g2[r] = (allv >= 2).mean(); g3[r] = (allv >= 3).mean()
    out.setdefault("null1_tail_sign", {})[tag] = dict(P2_obs=float(o2), P2_null=[float(np.quantile(g2, .025)), float(np.quantile(g2, .975))], P3_obs=float(o3), P3_null=[float(np.quantile(g3, .025)), float(np.quantile(g3, .975))])
    print(tag, "P(N>=2) obs %.4f null1 [%.4f, %.4f] | P(N>=3) obs %.4f null1 [%.4f, %.4f]" % (o2, *np.quantile(g2, [.025, .975]), o3, *np.quantile(g3, [.025, .975])))
# (c) a0.1 seed 0 time-mean of the state-matched Cantelli
st = json.load(open(f"{V}/verify/vstats_a01s0.json")); mu = dict(zip(st["per_target"]["i"], st["per_target"]["mu"])); var = dict(zip(st["per_target"]["i"], st["per_target"]["var"]))
N = np.load(f"{V}/verify/vcounts_a01s0.npz")["N"]; NB = {}
for f in glob.glob(f"{V}/nbeta_time/grid/p172c_a01_T*_real.json") + glob.glob(f"{V}/nbeta_time/full/p172c_a01_T*_real_slim.json"):
    T = int(re.search(r"_T(\d+)_", f).group(1)); NB[T] = {int(r["i"]): float(r["n_b"]) for r in json.load(open(f))["real"]}
tm = {}
for i in mu:
    cs = []
    for T in range(11, 101):
        if N[T - 1, i] < 0: continue
        nb = NB[T][i]; cs.append(var[i] / (var[i] + (nb - mu[i]) ** 2) if nb > mu[i] else 1.0)
    tm[i] = float(np.mean(cs))
imax = max(tm, key=tm.get); out["a01s0_time_mean_cantelli"] = dict(max=tm[imax], target=imax, sigma_i=float(sum(tm.values())))
print("a01s0 time-mean state-matched Cantelli: max %.4f (u%d) Sigma_i %.4f" % (tm[imax], imax, sum(tm.values())))
json.dump(out, open(f"{V}/verify/verify_extra.json", "w"), indent=1)
