"""xbar_arrangement_1014 / agent D: brightness K2 in h1 space (layer 2 of the 3-layer RL-CIFAR battle nets) vs occupancy,
bridges, overturned units and hubs.  Read-only on obsidian-research-data (snapshots, P176 json); nothing there is written.

Loader and layer-2 model copied from P176 (stage16_1008_transplant/p176_l2.py, p176_core.py), numpy only:
  X = RC.Cifar10().images(RC.subset_idx(seed)) (std: standardize), float64;  z1 = X W1^T + b1;  h1 = phi(z1) (hard leaky 0.1 / ReLU / ELU);
  x~_n = (h1_n, 1);  xbar = mean_n x~_n;  K2_n = x~_n . xbar (= mu2.h1_n + 1);  Kbar2 = xbar.xbar (= mean_n K2_n, identity).
  layer-2 unit k: w~_k = (W2_k, b2_k);  z_kn = w~_k . x~_n;  m_k = w~_k . xbar (= mean_n z_kn);  u_kn = z_kn - m_k K2_n / Kbar2 (sum_n u_kn = 0).
  open: z_kn > 0 (P176's Open, all arms);  alive: open on >= 1 input;  f_n = fraction of alive units open on n.
  gate derivative Dp (P176 gate_parts): LR/R smooth gate tau = 0.1 sd_n(z) + 1e-12, ELU exact (e^z on z <= 0, 0 above).
  threshold inputs T_i = top 3 of e' = K2 * Dp_i (np.argsort(-e)[:3], as P176 bridges());  bridge j of target i:
  j != i, k_j > 0, |O_j & T_i| >= 2 and |O_j & O_i| >= ceil(|O_i|/2)  (P158/P161 loose criterion, P176 bridges()).
  Extra (not P176): the K2-free threshold version T^c_i = top 3 of Dp_i (K2 const) with the same bridge rule.
  P176 verdict: t >= 1, not input-lost (h1 units with sd > 0 <= 3), not dead (median p2 < 1%), >= 1 alive unit.
usage: d_brightness.py <arm> [<arm> ...]     arm in LR, ELU, R   -> results/xbar_arrangement_1014/d_bright_<arm>.json
"""
import os, sys, json, time, math
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
sys.dont_write_bytecode = True
import numpy as np, torch
torch.set_num_threads(2)
ROOT = "/home/issan/Projects/claude/proj_004_drift"; sys.path.insert(0, os.path.join(ROOT, "src"))
import pmnist_rlcifar_0907 as RC
from rlcifar_mlp_battle_0918 import standardize
OBS = "/home/issan/Projects/obsidian-research-data"
SNAP = os.path.join(OBS, "rlcifar_mlp_battle_0918/results/rlcifar_mlp_battle_0918")
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
TLIST = [int(x) for x in os.environ["D_TLIST"].split(",")] if os.environ.get("D_TLIST") else list(range(51))
SEEDS = [int(x) for x in os.environ["D_SEEDS"].split(",")] if os.environ.get("D_SEEDS") else [0, 1, 2]
CONDS = os.environ["D_CONDS"].split(",") if os.environ.get("D_CONDS") else ["raw", "std"]
_cifar = None


# ---------------- copied from p176_core.py (numpy) ----------------
def act_spec(arm):
    return {"LR": "LR:0.1", "R": "LR:0", "ELU": "ELU"}[arm]


def hard_fns(spec):
    if spec.startswith("LR:"):
        a = float(spec[3:])
        return (lambda z: np.where(z > 0, z, a * z)), (lambda z: np.where(z > 0, 1.0, a))
    if spec == "ELU":
        return (lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))), (lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0.0))))
    raise ValueError(spec)


def gate_parts(Z, spec):
    if spec.startswith("LR:"):
        a = float(spec[3:]); tau = 0.1 * Z.std(0) + 1e-12
        s = 0.5 * (1 + np.tanh(0.5 * Z / tau[None, :])); D = a + (1 - a) * s; Dp = (1 - a) * s * (1 - s) / tau[None, :]
        return D, Dp
    if spec == "ELU":
        ez = np.exp(np.minimum(Z, 0.0))
        return np.where(Z > 0, 1.0, ez), np.where(Z > 0, 0.0, ez)
    raise ValueError(spec)


def bridges_T(Z, score, alive_idx):
    """P176 bridges() with the threshold score given explicitly (score = K2*Dp reproduces P176).  Returns per-target T_i, bridges."""
    Open = Z > 0; Of = Open.astype(np.float64); kj = Open.sum(0); H = Z.shape[1]; res = []
    for i in alive_idx:
        e = score[:, i]; Ti = np.argsort(-e)[:3]; Oi = Open[:, i]; mi = int(Oi.sum()); need = (mi + 1) // 2
        cntT = Open[Ti, :].sum(0); cntO = Of[Oi, :].sum(0); ok = (np.arange(H) != i) & (kj > 0)
        br = ok & (cntT >= 2) & (cntO >= need)
        res.append((int(i), Ti, np.where(br)[0]))
    return res


# ---------------- small helpers ----------------
def rankdata(a):
    a = np.asarray(a, float); n = len(a); order = np.argsort(a, kind="mergesort"); sa = a[order]
    new = np.r_[True, sa[1:] != sa[:-1]]; gid = np.cumsum(new) - 1
    cnt = np.bincount(gid); avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = np.empty(n); r[order] = avg[gid]; return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean(); den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def F(x):
    return None if x is None or not np.isfinite(x) else float(x)


def inputs(seed, cond):
    global _cifar
    if _cifar is None: _cifar = RC.Cifar10()
    x = _cifar.images(RC.subset_idx(seed), torch.device("cpu"))
    if cond == "std": x = standardize(x)
    return x.numpy().astype(np.float64)


def hist8(idx, octv):
    return np.bincount(octv[np.asarray(idx, int)], minlength=8).astype(int).tolist() if len(idx) else [0] * 8


def bridge_block(res, K2, octv, pct, Open):
    """D3 numbers for one bridge listing."""
    ev = []; uniq = set(); thr_all = []
    for i, Ti, br in res:
        thr_all.extend(int(n) for n in Ti)
        for j in br:
            for n in Ti:
                if Open[n, j]: ev.append(int(n)); uniq.add((i, int(n)))
    un = [n for (_, n) in uniq]
    ev = np.array(ev, int); un = np.array(un, int); thr_all = np.array(thr_all, int)
    return dict(n_pairs=int(sum(len(b) for _, _, b in res)), n_bridge=[int(len(b)) for _, _, b in res],
                ev_n=int(ev.size), ev_oct=hist8(ev, octv), ev_le0=int((K2[ev] <= 0).sum()) if ev.size else 0, ev_pct_med=F(np.median(pct[ev])) if ev.size else None,
                un_n=int(un.size), un_oct=hist8(un, octv), un_le0=int((K2[un] <= 0).sum()) if un.size else 0,
                thr_n=int(thr_all.size), thr_oct=hist8(thr_all, octv), thr_le0=int((K2[thr_all] <= 0).sum()) if thr_all.size else 0,
                thr_pct_med=F(np.median(pct[thr_all])) if thr_all.size else None)


def hub_block(sel, K2, octv, pct):
    sel = np.asarray(sel, int)
    return dict(n=int(sel.size), oct=hist8(sel, octv), le0=int((K2[sel] <= 0).sum()) if sel.size else 0,
                pct_med=F(np.median(pct[sel])) if sel.size else None, K2_med=F(np.median(K2[sel])) if sel.size else None)


def one_state(arm, cond, seed, t, X, keep_units):
    spec = act_spec(arm); phi, _ = hard_fns(spec)
    f = os.path.join(SNAP, arm, "snap", f"{arm}_{cond}_seed{seed}", f"t{t:02d}.npz")
    if not os.path.exists(f): return None
    d = np.load(f); W1, b1, W2, b2 = (d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))
    N = X.shape[0]
    z1 = X @ W1.T + b1; h1 = phi(z1); Z = h1 @ W2.T + b2
    chk_z2 = float(np.median(np.abs(Z - d["z2"].astype(np.float64)) / (np.abs(d["z2"].astype(np.float64)) + 1e-3))) if "z2" in d.files else None
    Xt = np.concatenate([h1, np.ones((N, 1))], 1); xbar = Xt.mean(0); K2 = Xt @ xbar; Kbar = float(xbar @ xbar)
    Wt = np.concatenate([W2, b2[:, None]], 1); m = Wt @ xbar; U = Z - np.outer(K2 / Kbar, m)
    S = dict(t=t, seed=seed, chk=dict(z2_vs_stored_relmed=chk_z2, Kbar_minus_meanK2=float(Kbar - K2.mean()),
                                      m_minus_meanz_maxabs=float(np.abs(m - Z.mean(0)).max()), u_mean_maxabs=float(np.abs(U.mean(0)).max()),
                                      z_scale=float(np.abs(Z).max())))
    Open = Z > 0; k2 = Open.sum(0); alive = np.where(k2 > 0)[0]; p2 = Open.mean(0)
    n_h1var = int((h1.std(0) > 0).sum()); input_lost = n_h1var <= 3; dead = bool(np.median(p2) < 0.01)
    S.update(n_h1var=n_h1var, input_lost=input_lost, dead=dead, n_alive=int(alive.size), p2_med=float(np.median(p2)),
             verdict=bool(t >= 1 and not input_lost and not dead and alive.size > 0))
    # ---- D1
    edges = np.quantile(K2, np.arange(1, 8) / 8.0); octv = np.searchsorted(edges, K2, side="right"); pct = (rankdata(K2) - 1) / max(N - 1, 1)
    le0 = np.where(K2 <= 0)[0]
    S["D1"] = dict(min=float(K2.min()), max=float(K2.max()), mean=float(K2.mean()), sd=float(K2.std()), Kbar=Kbar, median=float(np.median(K2)),
                   n_le0=int(le0.size), frac_le0=float(le0.size / N), n_lt02=int((K2 < 0.2 * Kbar).sum()), frac_lt02=float((K2 < 0.2 * Kbar).mean()),
                   le0_idx=[int(n) for n in le0], le0_K2=[float(K2[n]) for n in le0], oct_count=np.bincount(octv, minlength=8).tolist(),
                   n_unique_K2=int(np.unique(np.round(K2, 12)).size), mu2_norm=float(np.linalg.norm(xbar[:-1])))
    # ---- D5 (hub sets) and D6 do not need alive units except D6-alive
    nrm = np.sqrt(((h1 - xbar[:-1]) ** 2).sum(1)); n1 = (z1 > 0).sum(1).astype(float)
    top_norm5 = np.argsort(-nrm, kind="stable")[: int(round(0.05 * N))]; top_norm1 = np.argsort(-nrm, kind="stable")[: int(round(0.01 * N))]
    q05, q95 = np.quantile(n1, 0.05), np.quantile(n1, 0.95)
    l1low = np.where(n1 <= q05)[0]; l1high = np.where(n1 >= q95)[0]
    S["D5"] = dict(norm5=hub_block(top_norm5, K2, octv, pct), norm1=hub_block(top_norm1, K2, octv, pct),
                   l1low5=dict(hub_block(l1low, K2, octv, pct), q=float(q05)), l1high5=dict(hub_block(l1high, K2, octv, pct), q=float(q95)),
                   sp_K2_norm=spearman(K2, nrm), sp_K2_n1=spearman(K2, n1), n1_med=float(np.median(n1)), n1_max=float(n1.max()),
                   n1_at_le0=[float(n1[n]) for n in le0], norm_pct_at_le0=[float((rankdata(nrm)[n] - 1) / max(N - 1, 1)) for n in le0])
    # ---- D6 (mean unit), all H units and alive units
    def d6(idx):
        if len(idx) == 0: return None
        zbar = Z[:, idx].mean(1); mbar = float(m[idx].mean()); ubar = U[:, idx].mean(1)
        idr = float(np.abs(zbar - (mbar * K2 / Kbar + ubar)).max()); pos = zbar > 0; kp = K2 > 0
        ratio = ubar[kp] / (abs(mbar) * K2[kp] / Kbar) if (abs(mbar) > 0 and kp.any()) else np.array([np.nan])
        return dict(mbar=mbar, n_pos=int(pos.sum()), n_pos_K2le0=int((pos & ~kp).sum()), n_pos_K2gt0=int((pos & kp).sum()), n_K2le0=int((~kp).sum()),
                    n_K2le0_pos=int((pos & ~kp).sum()), max_ratio_K2gt0=F(np.nanmax(ratio)), identity_maxabs=idr, max_zbar=float(zbar.max()),
                    zbar_at_le0=[float(zbar[n]) for n in le0], ubar_sd=float(ubar.std()))
    S["D6"] = dict(all=d6(np.arange(Z.shape[1])), alive=d6(alive))
    if alive.size == 0:
        return S
    # ---- D2
    fr = Open[:, alive].mean(1)
    W3 = d["W3"].astype(np.float64); W3c = W3 - W3.mean(0, keepdims=True); g = (W3c * W3c).sum(0)
    fw = (Open.astype(np.float64) * g[None, :]).sum(1) / (g.sum() + 1e-300)                 # P161 strength-weighted (P176's f)
    fmax = fr.max(); tie = np.where(fr == fmax)[0]; am = int(np.argmax(fr)); ge = np.where(fr >= 0.5)[0]
    fpct = (rankdata(fr) - 1) / max(N - 1, 1)
    S["D2"] = dict(sp_f_K2=spearman(fr, K2), oct_f=[F(fr[octv == o].mean()) if (octv == o).any() else None for o in range(8)],
                   f_max=float(fmax), f_mean=float(fr.mean()), n_ge_half=int(ge.size), ge_half_le0=int((K2[ge] <= 0).sum()), ge_half_oct0=int((octv[ge] == 0).sum()),
                   argmax_K2=float(K2[am]), argmax_pct=float(pct[am]), argmax_oct=int(octv[am]), n_tie=int(tie.size), tie_oct0=int((octv[tie] == 0).sum()),
                   tie_le0=int((K2[tie] <= 0).sum()), f_at_le0=[float(fr[n]) for n in le0], fpct_at_le0=[float(fpct[n]) for n in le0],
                   o_at_le0=[int(Open[n, alive].sum()) for n in le0],
                   fw_sp_K2=spearman(fw, K2), fw_max=float(fw.max()), fw_n_ge_half=int((fw >= 0.5).sum()), fw_ge_half_le0=int((K2[fw >= 0.5] <= 0).sum()),
                   fw_argmax_pct=float(pct[int(np.argmax(fw))]))
    # position term vs pattern at the K2 <= 0 inputs and at argmax f:  z_kn = m_k K2_n / Kbar2 + u_kn  (alive units)
    def decomp(n):
        pos_term = m[alive] * K2[n] / Kbar; u = U[n, alive]; z = Z[n, alive]
        return dict(n=int(n), K2=float(K2[n]), n_open=int((z > 0).sum()), n_u_pos=int((u > 0).sum()), n_open_u_le0=int(((z > 0) & (u <= 0)).sum()),
                    n_closed_u_pos=int(((z <= 0) & (u > 0)).sum()), n_posterm_pos=int((pos_term > 0).sum()), posterm_med=float(np.median(pos_term)),
                    u_med=float(np.median(u)), z_med=float(np.median(z)), n_open_typ=int((u + m[alive] > 0).sum()))
    S["D2"]["decomp_le0"] = [decomp(n) for n in le0]; S["D2"]["decomp_argmax"] = decomp(am)
    fu = (U[:, alive] > 0).mean(1); S["D2"]["fu_sp_K2"] = spearman(fu, K2); S["D2"]["fu_n_ge_half"] = int((fu >= 0.5).sum())
    fupct = (rankdata(fu) - 1) / max(N - 1, 1); S["D2"]["fupct_at_le0"] = [float(fupct[n]) for n in le0]
    # o_n top 1% (P178 diag "most-open inputs")
    o = Open[:, alive].sum(1).astype(float); top_o = np.argsort(-o)[: int(round(0.01 * N))]
    S["D5"]["open2_top1"] = hub_block(top_o, K2, octv, pct); S["D5"]["sp_o_norm"] = spearman(o, nrm); S["D5"]["sp_o_n1"] = spearman(o, n1)
    # ---- D3 bridges (P176 thresholds) and K2-free thresholds
    _, Dp = gate_parts(Z, spec)
    resP = bridges_T(Z, K2[:, None] * Dp, alive); resC = bridges_T(Z, Dp, alive)
    S["D3"] = dict(P176=bridge_block(resP, K2, octv, pct, Open), K2free=bridge_block(resC, K2, octv, pct, Open))
    # bridge events at K2 <= 0 inputs: number of (target, bridge) pairs whose covered threshold inputs include a K2 <= 0 input
    S["D3"]["P176"]["pairs_with_le0"] = int(sum(1 for i, Ti, br in resP for j in br if any(Open[n, j] and K2[n] <= 0 for n in Ti)))
    S["D3"]["K2free"]["pairs_with_le0"] = int(sum(1 for i, Ti, br in resC for j in br if any(Open[n, j] and K2[n] <= 0 for n in Ti)))
    S["units"] = dict(unit=[int(i) for i in alive], n_bridge=S["D3"]["P176"]["n_bridge"])
    if keep_units:
        S["units"].update(p=[float(p2[i]) for i in alive], T=[[int(n) for n in Ti] for _, Ti, _ in resP], T_K2=[[float(K2[n]) for n in Ti] for _, Ti, _ in resP],
                          T_pct=[[float(pct[n]) for n in Ti] for _, Ti, _ in resP], T_oct=[[int(octv[n]) for n in Ti] for _, Ti, _ in resP],
                          Tc=[[int(n) for n in Ti] for _, Ti, _ in resC], Tc_K2=[[float(K2[n]) for n in Ti] for _, Ti, _ in resC],
                          Tc_pct=[[float(pct[n]) for n in Ti] for _, Ti, _ in resC], Tc_oct=[[int(octv[n]) for n in Ti] for _, Ti, _ in resC],
                          n_bridge_c=S["D3"]["K2free"]["n_bridge"],
                          br_ev_K2=[[float(K2[n]) for j in br for n in Ti if Open[n, j]] for _, Ti, br in resP],
                          m=[float(m[i]) for i in alive], e_pos=[int((K2 * Dp[:, i] > 0).sum()) for i in alive],
                          open_le0=[[bool(Open[n, i]) for n in le0] for i in alive])
    return S


def run(arm):
    out = dict(arm=arm, spec=act_spec(arm), tlist=TLIST, runs={})
    for cond in CONDS:
        for seed in SEEDS:
            ts = time.time(); X = inputs(seed, cond); R = {}
            for t in TLIST:
                S = one_state(arm, cond, seed, t, X, keep_units=True)
                if S is not None: R[t] = S
            out["runs"][f"{cond}_s{seed}"] = R
            le = [(t, S["D1"]["n_le0"], round(S["D1"]["min"], 2)) for t, S in R.items() if S["D1"]["n_le0"] > 0]
            print(f"{arm} {cond} s{seed}: {len(R)} states, K2<=0 states {len(le)} {le[:12]} ({time.time() - ts:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(OUT, f"d_bright_{arm}.json"), "w"))
    print(f"== {arm} done", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]: run(a)
