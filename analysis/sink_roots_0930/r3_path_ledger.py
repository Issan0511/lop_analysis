#!/usr/bin/env python3
"""Round 3 R1p_cifar_path_ledger (spec_sink_roots_0930_round3.md): the endpoint and path ledgers of layer 2's mean
pre-activation m_i = w_i . mu + b_i in RL-CIFAR ELU/std, tasks 1-3 (records every 100 updates).

Per unit i and window [a, b] (each task, and the whole window tasks 1-3):
  endpoint:  dm = (dw . mu_a + db)  [self]  +  w_a . dmu  [upstream]  +  dw . dmu  [cross]
  path (100-update steps s):  self_path = sum_s (dw_s . mu_{s-1} + db_s),  up_path = sum_s w_{s-1} . dmu_s,
                              resid = sum_s dw_s . dmu_s      (identity: self_path + up_path + resid = dm)
  cross to self = self_path - (dw . mu_a + db) = sum_{r<s} dw_s . dmu_r;  share_self = that / (dw . dmu)
  f_i(s) = (w_i(s) - w_i(a)) . dw_i / |dw_i|^2,  g(s) = (mu(s) - mu(a)) . dmu / |dmu|^2,  int g df = sum_s g_{s-1} (f_s - f_{s-1})
  cos_w = cos(w(mid) - w(a), w(b) - w(mid)), cos_mu likewise (mid = the window's middle record)
Registered test (the request): on unit-windows with cos_w >= 0.8 and cos_mu >= 0.8, |share_self - int g df| <= 0.2.
Descriptive: |dw|/|w_a|, cos(dw, dmu), cos(w_a, dmu), the cross / upstream ratio; units = all 100 (and those open at a)."""
import glob, json, re
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round3/R1p")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def window(W, bb, mu, ia, ib):
    """W: [n, H, D] (rows = units), bb: [n, H], mu: [n, D]; records ia..ib inclusive."""
    W = W[ia:ib + 1].astype(np.float64); bb = bb[ia:ib + 1].astype(np.float64); mu = mu[ia:ib + 1]
    dW = W[-1] - W[0]; db = bb[-1] - bb[0]; dmu = mu[-1] - mu[0]
    m = np.einsum("nhd,nd->nh", W, mu) + bb
    dm = m[-1] - m[0]
    e_self = dW @ mu[0] + db; e_up = W[0] @ dmu; e_cross = dW @ dmu
    dWs = np.diff(W, axis=0); dbs = np.diff(bb, axis=0); dmus = np.diff(mu, axis=0)
    p_self = np.einsum("shd,sd->h", dWs, mu[:-1]) + dbs.sum(0)
    p_up = np.einsum("shd,sd->h", W[:-1], dmus)
    p_res = np.einsum("shd,sd->h", dWs, dmus)
    to_self = p_self - e_self
    share = to_self / np.where(np.abs(e_cross) > 1e-12, e_cross, np.nan)
    f = np.einsum("nhd,hd->nh", W - W[0], dW) / (np.einsum("hd,hd->h", dW, dW)[None] + 1e-300)
    g = (mu - mu[0]) @ dmu / (dmu @ dmu + 1e-300)
    igdf = (g[:-1, None] * np.diff(f, axis=0)).sum(0)
    mid = (len(W) - 1) // 2
    cos_w = np.array([cosv(W[mid, h] - W[0, h], W[-1, h] - W[mid, h]) for h in range(W.shape[1])])
    cos_mu = cosv(mu[mid] - mu[0], mu[-1] - mu[mid])
    return dict(dm=dm, e_self=e_self, e_up=e_up, e_cross=e_cross, p_self=p_self, p_up=p_up, p_res=p_res,
                closure=float(np.max(np.abs(p_self + p_up + p_res - dm))), share=share, igdf=igdf, cos_w=cos_w, cos_mu=cos_mu,
                rel_dw=np.linalg.norm(dW, axis=1) / (np.linalg.norm(W[0], axis=1) + 1e-300),
                cos_dw_dmu=np.array([cosv(dW[h], dmu) for h in range(W.shape[1])]),
                cos_w0_dmu=np.array([cosv(W[0, h], dmu) for h in range(W.shape[1])]),
                f_med=np.median(f, axis=1), g=g)


def main():
    L = ["seed window | n units | closure | cross / dm (median) | cross / upstream (median) | share_self median [q25, q75] | int g df median | "
         "cos_mu | cos_w median | test n (both cos >= 0.8) within 0.2 | |dw|/|w0| median | cos(dw,dmu) med | cos(w0,dmu) med"]
    out = {}
    agg = {"task": [], "all": []}
    for fn in sorted(glob.glob(str(RAW / "ledger_ELU_std_s*.npz"))):
        seed = int(re.search(r"_s(\d+)\.npz", fn).group(1))
        d = np.load(fn); task, W, bb, mu, k2 = d["task"], d["W2"], d["b2"], d["mu2"], d["k2"]
        wins = []
        for t in sorted(set(task.tolist())):
            idx = np.where(task == t)[0]; wins.append((f"task {t}", idx[0], idx[-1]))
        wins.append(("tasks 1-3", 0, len(task) - 1))
        for name, ia, ib in wins:
            r = window(W, bb, mu, ia, ib)
            ok = (r["cos_w"] >= 0.8) & (r["cos_mu"] >= 0.8) & np.isfinite(r["share"])
            within = np.abs(r["share"] - r["igdf"]) <= 0.2
            q = np.nanquantile(r["share"], [0.25, 0.5, 0.75])
            L.append(f"s{seed} {name:9s} | 100 | {r['closure']:.1e} | {np.median(r['e_cross'] / r['dm']):+.2f} | {np.median(r['e_cross'] / r['e_up']):+.2f} | "
                     f"{q[1]:+.2f} [{q[0]:+.2f}, {q[2]:+.2f}] | {np.median(r['igdf']):+.2f} | {r['cos_mu']:+.2f} | {np.median(r['cos_w']):+.2f} | "
                     f"{ok.sum()} {np.mean(within[ok]) if ok.any() else float('nan'):.2f} | {np.median(r['rel_dw']):.2f} | {np.median(r['cos_dw_dmu']):+.2f} | {np.median(r['cos_w0_dmu']):+.2f}")
            out[f"s{seed}_{name}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            agg["all" if name == "tasks 1-3" else "task"].append((ok.sum(), within[ok].sum() if ok.any() else 0))
    for key, v in agg.items():
        n = sum(a for a, _ in v); w = sum(b for _, b in v)
        L.append(f"\nregistered test ({'each task' if key == 'task' else 'tasks 1-3 window'}): unit-windows with both cos >= 0.8: {n}; "
                 f"|share - int g df| <= 0.2 in {w} ({w / n if n else float('nan'):.2f})")
    txt = "\n".join(L); print(txt)
    (RES / "round3_R1p_path_ledger.txt").write_text(txt + "\n")
    (RES / "round3_R1p_path_ledger.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
