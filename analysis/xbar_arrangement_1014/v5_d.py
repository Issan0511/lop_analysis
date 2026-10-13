"""V5 (verifier): spot-check agent D's claim 'in judged states, inputs with K2 <= 0 have f percentile >= 0.994' (REPORT_D §D2 summary)
using agent D's own loader (d_brightness.inputs / act_spec / hard_fns; snapshots t<t>.npz) but own code for f and the percentiles.
LR raw, seeds 0, 1, 2, all t = 1..50 snapshots: list every judged state with a K2 <= 0 input; for each such input:
f = fraction of alive layer-2 units open; percentile (a) average-rank (D's definition), (b) share of inputs with f strictly below, (c) share with f <= (ties counted)."""
import os, sys, json
sys.dont_write_bytecode = True
os.environ.setdefault("D_TLIST", "0")
import numpy as np
import d_brightness as D
from v_common import RES

def avg_rank(a):
    order = np.argsort(a, kind="mergesort"); sa = a[order]; new = np.r_[True, sa[1:] != sa[:-1]]; gid = np.cumsum(new) - 1
    cnt = np.bincount(gid); avg = np.cumsum(cnt) - (cnt - 1) / 2.0; r = np.empty(len(a)); r[order] = avg[gid]; return r

def main(arm="LR", cond="raw"):
    phi, _ = D.hard_fns(D.act_spec(arm)); rows = []; n_states_judged = 0
    for seed in (0, 1, 2):
        X = D.inputs(seed, cond); N = X.shape[0]
        for t in range(1, 51):
            f = os.path.join(D.SNAP, arm, "snap", f"{arm}_{cond}_seed{seed}", f"t{t:02d}.npz")
            if not os.path.exists(f): continue
            d = np.load(f); W1, b1, W2, b2 = (d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))
            h1 = phi(X @ W1.T + b1); Z = h1 @ W2.T + b2; Xt = np.c_[h1, np.ones(N)]; xbar = Xt.mean(0); K2 = Xt @ xbar
            Open = Z > 0; alive = np.flatnonzero(Open.sum(0) > 0); p2 = Open.mean(0)
            judged = (int((h1.std(0) > 0).sum()) > 3) and not (np.median(p2) < 0.01) and alive.size > 0
            n_states_judged += int(judged)
            le0 = np.flatnonzero(K2 <= 0)
            if not judged or le0.size == 0: continue
            fr = Open[:, alive].mean(1); pa = (avg_rank(fr) - 1) / (N - 1)
            for n in le0:
                rows.append(dict(seed=seed, t=t, n=int(n), K2=float(K2[n]), Kbar2=float(xbar @ xbar), f=float(fr[n]), pct_avgrank=float(pa[n]),
                                 share_strictly_below=float((fr < fr[n]).mean()), share_le=float((fr <= fr[n]).mean()), n_ties=int((fr == fr[n]).sum()),
                                 is_argmax=bool(fr[n] == fr.max()), n_alive=int(alive.size), N=N))
    out = dict(arm=arm, cond=cond, n_states_judged=n_states_judged, n_le0_inputs=len(rows), n_states_with_le0=len({(r["seed"], r["t"]) for r in rows}),
               min_pct_avgrank=min(r["pct_avgrank"] for r in rows), min_share_strictly_below=min(r["share_strictly_below"] for r in rows), rows=rows)
    json.dump(out, open(os.path.join(RES, "v5_d_LRraw.json"), "w"), indent=1)
    for r in rows: print(r)
    print({k: v for k, v in out.items() if k != "rows"})

if __name__ == "__main__":
    main()
