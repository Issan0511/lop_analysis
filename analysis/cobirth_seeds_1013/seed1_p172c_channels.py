"""P172c (agent_bound, 1007, resumed): lower bound on the number of lifting units with BOTH channels of the bottom.
Bottom (lemma 0, both layers):  dhat_in = sum_{k!=i} w_ik d_kn  +  sum_k w^r_ik f_kn  +  w^1_i,
   w_ik = g_k <k_i,k_k>_G / kap'_in  (gate channel, k open or closed),   w^r_ik = <k_i, f_k>_G / kap'_in  (readout channel, any input; k = i is the unit's own feature),
   w^1_i = <k_i, 1>_G / kap'_in  (ones term).   The script p172_bound.py lumps  w_r = sum_k w^r f + w^1.
Per unit k != i, per input n:  Delta_k(n) = w_ik d_kn + w^r_ik f_kn  (the unit's lift of the bottom; three parts: open gate, closed gate, readout).
Identity:  A^g_i = NEED_i - sum_{k!=i} L_k,   NEED_i = sum_n c_in (d_in - w^1_in - w^r_ii f_in),   L_k = sum_n c_in Delta_k(n),  c_in > 0.
Count bounds (three bookkeepings):  beta: all three parts per unit, NEED as above;  alpha': closed-gate parts moved into NEED;  gamma: closed-gate AND readout moved into NEED (= p172_bound's n*_c).
Static caps per (i, k, n) [exact]:  |w_ik d_kn| <= sqrt(g_k/g_i) R_i R_k / (1 - R_i^2),   |w^r_ik f_kn| <= R_i |f_k|_G |f_kn| / (sqrt(g_i Th_nn) (1 - R_i^2)),  |f_k|_G <= 1,
   and l2:  sum_{k!=i} (g_i/g_k) w_ik^2 + g_i Th_nn sum_k (w^r_ik)^2 <= R_i^2/(1 - R_i^2)   (from Gram_G({sqrt(g) k}, {f}) <= diag(Th_nn I, I)).
usage: p172c_channels.py <a01|a03|a05> <task> [real|helpers|scale|all]

[cobirth_seeds_1013 seed1 copy, 2026-10-14] Generalized copy of
  push_direction_proof_0930/review/stage4_1002/block_1003/stage14_1007_h4/agent_bound/p172c_channels.py
  The computation (parts / analyze / self_term / target / summarize) is unchanged. Only the I/O was generalized:
    --npz PATH   saved weights (absolute, or relative to ROOT); default: the seed-0 file of --tag (original behaviour)
    --a SLOPE    leaky slope a; default: from --tag (original behaviour)
    --tag TAG    label used in the output file name (default a01/a03/a05 positional arg)
    --task t     stoch/p161/p170/p172c convention: state = end0[t-2]  (t = T + 1 in the M1 convention)
    --T T        M1 convention (cobirth_1013): state = end0[T-1]; converted to t = T + 1
    --out DIR    output directory (default: the folder of this script)
    --mode real|helpers|scale|all  (default real)
  Threads: OMP/OPENBLAS/MKL = 1 (the original set 2).  Output file: <out>/p172c_<tag>_t<t>_<mode>.json (t in the stoch convention).
  Positional legacy form still works: p172c_channels.py a03 100 real  (= seed 0 a0.3, t 100 = T 99).
  usage (new): seed1_p172c_channels.py --npz review/opus_lrslope/out/LRa0.3_s1.npz --a 0.3 --tag a03s1 --T 99 --out DIR"""
import os, sys, json, time, glob
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"   # cobirth_seeds_1013: 1 thread (original: 2)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
S10 = os.path.abspath(os.path.join(HERE, "..", "..", "stage10_1005"))
sys.path.insert(0, ROOT); from common import load_X
sp = lambda x: np.logaddexp(0, x)
def parts(Xt, W, a, tau):
    Z = Xt @ W.T; s = 0.5 * (1 + np.tanh(0.5 * Z / tau[None, :])); D = a + (1 - a) * s; Dp = (1 - a) * s * (1 - s) / tau[None, :]
    F = a * Z + (1 - a) * tau[None, :] * sp(Z / tau[None, :]); return Z, s, D, Dp, F
# ---- cobirth_seeds_1013: generalized argument parsing (computation below unchanged) ----
_DEF = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
_argv = sys.argv[1:]; _kw = {}; _pos = []; _k = 0
while _k < len(_argv):
    if _argv[_k].startswith("--"): _kw[_argv[_k][2:]] = _argv[_k + 1]; _k += 2
    else: _pos.append(_argv[_k]); _k += 1
tag = _kw.get("tag", _pos[0] if _pos else None)
if "T" in _kw: task = int(_kw["T"]) + 1          # M1 convention T -> stoch t = T + 1 (state end0[T-1] = end0[t-2])
elif "task" in _kw: task = int(_kw["task"])
else: task = int(_pos[1])
mode = _kw.get("mode", _pos[2] if len(_pos) > 2 else "real")
src = _kw.get("npz", _DEF[tag][0] if tag in _DEF else None); a = float(_kw["a"]) if "a" in _kw else _DEF[tag][1]
OUTDIR = _kw.get("out", HERE); os.makedirs(OUTDIR, exist_ok=True)
assert src is not None, "need --npz (or a seed-0 tag a01/a03/a05)"
dd = np.load(src if os.path.isabs(src) else os.path.join(ROOT, src)); X = load_X(dd["subset"]); N = X.shape[0]; Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kv = Xt @ xbar; Th = Xt @ Xt.T; one = np.ones((N, N)); thr = (1 + a) / 2
W1, b1, W2 = dd["end0"][task - 2].astype(np.float64), dd["end1"][task - 2].astype(np.float64), dd["end2"][task - 2].astype(np.float64)
W0 = np.concatenate([W1, b1[:, None]], 1); Vc = W2.T - W2.T.mean(1, keepdims=True); g0 = (Vc * Vc).sum(1)
def q(x, qs=(0.1, 0.5, 0.9)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return [float(np.quantile(x, v)) for v in qs] if x.size else [float("nan")] * len(qs)
def r3(x): return [round(float(v), 4) for v in x]
def analyze(W, g, tau, lam=None):
    Z, s, D, Dp, F = parts(Xt, W, a, tau); H = W.shape[0]; R = F @ F.T + one
    full = (D * g[None, :]) @ D.T * Th + R
    if lam is None: lam = 1e-4 * float(np.linalg.eigvalsh(full)[-1])
    E = np.linalg.inv(full + lam * np.eye(N)); Enn = np.diag(E).copy(); dPhi = g[None, :] * ((Th * E) @ D)
    Ar = np.einsum("nk,nk->k", Kv[:, None] * D, E @ F); ep = Kv[:, None] * Dp
    dhat = np.zeros((N, H)); kap = np.zeros((N, H)); wr = np.zeros((N, H)); W1v = np.zeros((N, H)); nk2 = np.zeros((N, H)); nf2 = np.zeros((N, H))
    Wall = np.zeros((N, H, H), np.float32); Wr = np.zeros((N, H, H), np.float32); lmaxG = np.zeros(N); lmaxX = np.zeros(N); sg = np.sqrt(g); wr_chk = 0.0
    for n in range(N):
        Kf = D * Th[:, n][:, None]; Kf[n] = 0.0; EK = E @ Kf; GK = EK - np.outer(E[:, n], EK[n]) / E[n, n]; Mn = Kf.T @ GK
        Ff = F.copy(); Ff[n] = 0.0; EF = E @ Ff; GF = EF - np.outer(E[:, n], EF[n]) / E[n, n]; Vn = Kf.T @ GF; NF = Ff.T @ GF
        ov = np.ones(N); ov[n] = 0.0; Eo = E @ ov; Go = Eo - E[:, n] * (Eo[n] / E[n, n]); o = Kf.T @ Go
        rf = R[:, n].copy(); rf[n] = 0.0; Er = E @ rf; Gr = Er - E[:, n] * (Er[n] / E[n, n]); vn = Kf.T @ Gr
        kp = Th[n, n] - g * np.diag(Mn); Wn = (g[None, :] * Mn) / kp[:, None]; np.fill_diagonal(Wn, 0.0); Wrn = Vn / kp[:, None]; w1n = o / kp
        wrn = (Vn @ F[n] + o) / kp; wr_chk = max(wr_chk, float(np.abs(wrn - vn / kp).max() / max(np.abs(vn / kp).max(), 1e-300)))
        dhat[n] = Wn @ D[n] + wrn; kap[n] = kp; wr[n] = wrn; Wall[n] = Wn; Wr[n] = Wrn; W1v[n] = w1n; nk2[n] = np.diag(Mn); nf2[n] = np.diag(NF)
        Gam = sg[:, None] * Mn * sg[None, :]; lmaxG[n] = float(np.linalg.eigvalsh(Gam)[-1] / Th[n, n])
        B = (sg[:, None] * Vn) / np.sqrt(Th[n, n]); Xg = np.block([[Gam / Th[n, n], B], [B.T, NF]]); lmaxX[n] = float(np.linalg.eigvalsh(Xg)[-1])
    chk = g[None, :] * kap * Enn[:, None] * (D - dhat); err = float(np.abs(chk - dPhi).max() / np.abs(dPhi).max()); Ag = (ep * dPhi).sum(0)
    return dict(Z=Z, s=s, D=D, Dp=Dp, F=F, ep=ep, dPhi=dPhi, Ar=Ar, Ag=Ag, A=Ag + Ar, dhat=dhat, kap=kap, wr=wr, W1v=W1v, nk2=nk2, nf2=nf2, Wall=Wall, Wr=Wr, lam=lam, Enn=Enn, g=g, tau=tau, err=err, wr_chk=wr_chk,
                lmaxG=lmaxG, lmaxX=lmaxX, alive=(Z > 0).any(0))
def self_term(S, i):
    g, lam = S["g"], S["lam"]; d = S["D"][:, i]; f = S["F"][:, i]
    Ki = g[i] * (d[:, None] * Th * d[None, :]) + np.outer(f, f) + 1.0; Ei = np.linalg.inv(Ki + lam * np.eye(N)); return float(g[i] * (Kv * S["Dp"][:, i]) @ ((Th * Ei) @ d) + (Kv * d) @ (Ei @ f))
def pos(x): return np.maximum(x, 0.0)
def target(S, i, helpers=None, ntop=3):
    g = S["g"]; H = len(g); D = S["D"]; F = S["F"]; s = S["s"]; ep = S["ep"][:, i]; kap = S["kap"]; Enn = S["Enn"]
    c = ep * g[i] * kap[:, i] * Enn; Csum = float(c.sum()); cn = c / Csum
    Wi = S["Wall"][:, i, :].astype(np.float64); Wri = S["Wr"][:, i, :].astype(np.float64); w1 = S["W1v"][:, i]
    Gk = Wi * D; Rk = Wri * F; own = Rk[:, i].copy(); Rk[:, i] = 0.0; Gk[:, i] = 0.0
    opn = s > 0.5; opn[:, i] = False; cls = ~opn; cls[:, i] = False
    need_n = D[:, i] - w1 - own                                                           # beta need per input
    Lopen = (cn[:, None] * Gk * opn).sum(0); Lcls = (cn[:, None] * Gk * cls).sum(0); Lr = (cn[:, None] * Rk).sum(0); L = Lopen + Lcls + Lr
    need_b = float((cn * need_n).sum()); need_a = need_b - float(Lcls.sum()); need_g = need_a - float(Lr.sum())
    Ag_direct = float((ep * S["dPhi"][:, i]).sum()); ident = float(abs((need_b - L.sum()) * Csum - Ag_direct) / max(abs(Ag_direct), 1e-300))
    La = Lopen + Lr; Lg = Lopen
    def nb(need, Lv): m = float(pos(Lv).max()); return float(need / m) if (need > 0 and m > 0) else (float("inf") if need > 0 else 0.0)
    PR = lambda v: float(pos(v).sum() ** 2 / max((pos(v) ** 2).sum(), 1e-300))
    covL = (cn[:, None] * ((Gk * opn + Rk) > 0)).sum(0); cov = (cn[:, None] * opn).sum(0)
    tops = [int(n) for n in np.argsort(-ep)[:ntop]]; cs = np.sort(c)[::-1]; J90 = int(np.searchsorted(np.cumsum(cs) / Csum, 0.9) + 1)
    rows = []
    for n in tops:
        Th_n = Th[n, n]; Ri2 = g[i] * S["nk2"][n, i] / Th_n; Rk2 = g * S["nk2"][n] / Th_n; ok = opn[n]; ck = cls[n]
        dG = Gk[n].copy(); dR = Rk[n].copy(); Delta = dG + dR; Dalpha = dG * ok + dR; Dgam = dG * ok
        wb = float(need_n[n]); wa = wb - float(dG[ck].sum()); wg = wa - float(dR.sum())
        Mik = Wi[n] * kap[n, i] / g; cosg = Mik / np.sqrt(S["nk2"][n, i] * np.maximum(S["nk2"][n], 1e-300)); cosg[i] = 0.0
        Vik = Wri[n] * kap[n, i]; cosf = Vik / np.sqrt(S["nk2"][n, i] * np.maximum(S["nf2"][n], 1e-300)); cosf_i = float(cosf[i]); cosf[i] = 0.0
        capg = np.sqrt(g / g[i]) * np.sqrt(Ri2 * Rk2) / (1 - Ri2); capg[i] = np.inf
        capr = np.sqrt(Ri2) * np.sqrt(np.maximum(S["nf2"][n], 0)) * np.abs(F[n]) / (np.sqrt(g[i] * Th_n) * (1 - Ri2)); capr_i = float(capr[i]); capr[i] = np.inf
        viol_g = float((np.abs(dG) / capg).max()); viol_r = float((np.abs(dR) / capr).max())
        kb = int(np.argmax(Delta)); ka = int(np.argmax(Dalpha)); kg = int(np.argmax(Dgam)) if ok.any() else -1
        l2_lhs = float(((g[i] / g) * Wi[n] ** 2).sum() + g[i] * Th_n * (Wri[n] ** 2).sum()); l2_rhs = float(Ri2 / (1 - Ri2))
        fbar2 = float((F[n] ** 2).max() / (g[i] * Th_n)); rho_g = float(g.max() / g[i]); n_l2 = float(wb ** 2 * (1 - Ri2) / (2 * max(rho_g, fbar2) * Ri2)) if wb > 0 else 0.0
        hl = None
        if helpers is not None:
            hl = dict(sum_open=float((dG * ok)[helpers].sum()), sum_closed=float((dG * ck)[helpers].sum()), sum_r=float(dR[helpers].sum()), max_delta=float(Delta[helpers].max()),
                      cosg_max=float(np.abs(cosg[helpers]).max()), cosg_med=float(np.median(np.abs(cosg[helpers]))), cosf_max=float(np.abs(cosf[helpers]).max()), cosf_med=float(np.median(np.abs(cosf[helpers]))),
                      capg_over_actual=float((capg[helpers] / np.maximum(dG[helpers], 1e-300)).min()), capr_over_actual=float((capr[helpers] / np.maximum(np.abs(dR[helpers]), 1e-300)).min()), n_open=int(ok[helpers].sum()))
        rows.append(dict(n=n, s=float(s[n, i]), d=float(D[n, i]), e_share=float(ep[n] / ep.sum()), c_share=float(cn[n]), w1=float(w1[n]), own_r=float(own[n]), cosf_own=cosf_i, capr_own=capr_i,
                         need_b=wb, need_a=wa, need_g=wg, sup_open=float(dG[ok].sum()), sup_closed=float(dG[ck].sum()), sup_r=float(dR.sum()), sup_r_pos=float(pos(dR).sum()), n_open=int(ok.sum()),
                         max_delta_b=float(Delta.max()), max_delta_a=float(Dalpha.max()), max_delta_g=float(Dgam.max()) if ok.any() else 0.0,
                         n_b=nb(wb, Delta), n_a=nb(wa, Dalpha), n_g=nb(wg, Dgam), PR_b=PR(Delta), PR_a=PR(Dalpha), kb=kb, kb_open=bool(ok[kb]), kb_share_r=float(dR[kb] / Delta[kb]) if Delta[kb] > 0 else float("nan"),
                         Ri=float(np.sqrt(Ri2)), Rk_b=float(np.sqrt(Rk2[kb])), rho_g=rho_g, fbar2=fbar2, cosg_max=float(np.abs(cosg).max()), cosg_max_open=float(np.abs(cosg[ok]).max()) if ok.any() else 0.0, cosf_max=float(np.abs(cosf).max()),
                         capg_kb=float(capg[kb]), capr_kb=float(capr[kb]), capg_over_dG_kb=float(capg[kb] / dG[kb]) if dG[kb] > 0 else float("nan"), capr_over_dR_kb=float(capr[kb] / abs(dR[kb])) if dR[kb] != 0 else float("nan"),
                         cap_b_over_actual=float((capg[kb] + capr[kb]) / Delta[kb]) if Delta[kb] > 0 else float("nan"), viol_g=viol_g, viol_r=viol_r, l2_lhs=l2_lhs, l2_rhs=l2_rhs, l2_tight=l2_lhs / l2_rhs, n_l2=n_l2, lifted=bool(S["dhat"][n, i] > D[n, i]), helpers=hl))
    top = [int(k) for k in np.argsort(-L)[:10]]
    rec = dict(i=int(i), A=float(S["A"][i]), Ag=Ag_direct, Ar=float(S["Ar"][i]), Csum=Csum, ident_err=ident, need_b=need_b, need_a=need_a, need_g=need_g, sup_b=float(L.sum()), sup_open=float(Lopen.sum()), sup_closed=float(Lcls.sum()), sup_r=float(Lr.sum()), sup_r_pos=float(pos(Lr).sum()),
               max_L_b=float(pos(L).max()), max_L_a=float(pos(La).max()), max_L_g=float(pos(Lg).max()), n_b=nb(need_b, L), n_a=nb(need_a, La), n_g=nb(need_g, Lg), PR_b=PR(L), PR_a=PR(La), PR_g=PR(Lg), nL_pos=int((L > 0).sum()), nLr_pos=int((Lr > 0).sum()),
               covL_max=float(covL.max()), cov_max=float(cov.max()), J90=J90, top10=[dict(k=int(k), L=float(L[k]), open=float(Lopen[k]), closed=float(Lcls[k]), r=float(Lr[k]), covL=float(covL[k]), cov=float(cov[k])) for k in top], rows=rows,
               fac_bound=float(need_g * (1 - np.mean([r["Ri"] for r in rows]) ** 2) / (max(cov.max(), 1e-12) * np.sqrt(g[S["alive"]].max() / g[i]) * max(max(r["cosg_max"] for r in rows), 1e-12) * np.mean([r["Ri"] for r in rows]))))
    if helpers is not None:
        hL = L[helpers]; rec["helpers"] = dict(sum_L=float(hL.sum()), sum_open=float(Lopen[helpers].sum()), sum_closed=float(Lcls[helpers].sum()), sum_r=float(Lr[helpers].sum()), max_L=float(hL.max()), n_h=float(need_b / hL.max()) if hL.max() > 0 else float("inf"),
                              L=[float(x) for x in hL], r_share=[float(Lr[h] / L[h]) if L[h] != 0 else float("nan") for h in helpers], orig_sum_L=float(L[:helpers[0]].sum()), covL=[float(covL[h]) for h in helpers], cov=[float(cov[h]) for h in helpers])
    return rec
t0 = time.time(); Z0 = Xt @ W0.T; tau0 = 0.1 * Z0.std(0) + 1e-12
S = analyze(W0, g0, tau0); alive = S["alive"]; g_med = float(np.median(g0[alive])); H0 = W0.shape[0]; lam0 = S["lam"]
print(f"{tag} t{task}: alive {alive.sum()} lam {lam0:.3g} identity err {S['err']:.1e} w_r split check {S['wr_chk']:.1e} | Gram check: max_n lmax(Gamma)/Th_nn {S['lmaxG'].max():.4f}, extended (gates+features) {S['lmaxX'].max():.4f} (must be < 1) | g_med {g_med:.3g} rho_g(alive) {g0[alive].max() / g0[alive].min():.2f} [{time.time() - t0:.0f}s]", flush=True)
out = dict(tag=tag, task=task, a=a, lam=lam0, g_med=g_med, alive=int(alive.sum()), lmaxG_max=float(S["lmaxG"].max()), lmaxX_max=float(S["lmaxX"].max()), real=[], helpers=[], scale=[])
def summarize(T_):
    col = lambda k: np.array([t[k] for t in T_], float); rows = [r for t in T_ for r in t["rows"]]; rc = lambda k: np.array([r[k] for r in rows], float)
    fin = lambda v: v[np.isfinite(v)]
    print(f"  identity A^g = NEED_b - sum L: max rel err {col('ident_err').max():.1e} | caps: max |w d|/cap_g {rc('viol_g').max():.4f} max |w^r f|/cap_r {rc('viol_r').max():.4f} (must be <= 1) | l2 lhs/rhs q {r3(q(rc('l2_tight')))} max {rc('l2_tight').max():.4f}")
    print(f"  UNIT: need_b q {r3(q(col('need_b')))} need_a q {r3(q(col('need_a')))} need_g q {r3(q(col('need_g')))} | supply_b q {r3(q(col('sup_b')))} [open {r3(q(col('sup_open')))} closed {r3(q(col('sup_closed')))} readout {r3(q(col('sup_r')))}] | readout share of positive supply q {r3(q(col('sup_r_pos') / (col('sup_open') + col('sup_closed') + col('sup_r_pos'))))}")
    print(f"  UNIT count bounds: n_b (all parts) q {r3(q(col('n_b')))} min {fin(col('n_b')).min():.1f} | n_a (closed in need) q {r3(q(col('n_a')))} min {fin(col('n_a')).min():.1f} | n_g (gate route only) q {r3(q(col('n_g')))} min {fin(col('n_g')).min():.1f} | PR_b q {r3(q(col('PR_b')))} PR_a q {r3(q(col('PR_a')))} | units with L>0 q {r3(q(col('nL_pos')))} | covL max q {r3(q(col('covL_max')))} cov max q {r3(q(col('cov_max')))} | J90 q {r3(q(col('J90')))}")
    print(f"  UNIT factorized bound need_g(1-R^2)/(cov_max sqrt(rho) kappa R) q {r3(q(col('fac_bound')))} min {col('fac_bound').min():.1f}")
    print(f"  TOP-3 inputs: need_b q {r3(q(rc('need_b')))} need_a q {r3(q(rc('need_a')))} need_g q {r3(q(rc('need_g')))} | w^1 q {r3(q(rc('w1')))} own readout q {r3(q(rc('own_r')))} | supply at n: open {r3(q(rc('sup_open')))} closed {r3(q(rc('sup_closed')))} readout {r3(q(rc('sup_r')))} (pos {r3(q(rc('sup_r_pos')))}) | n_open q {r3(q(rc('n_open')))} | lifted {int(rc('lifted').sum())}/{len(rows)}")
    print(f"  TOP-3 max Delta: b q {r3(q(rc('max_delta_b')))} a q {r3(q(rc('max_delta_a')))} g q {r3(q(rc('max_delta_g')))} | argmax(b) is open {int(rc('kb_open').sum())}/{len(rows)}, its readout share q {r3(q(rc('kb_share_r')))}")
    print(f"  TOP-3 count bounds: n_b q {r3(q(rc('n_b')))} min {fin(rc('n_b')).min():.2f} inf {int(np.isinf(rc('n_b')).sum())} | n_a q {r3(q(rc('n_a')))} min {fin(rc('n_a')).min():.2f} inf {int(np.isinf(rc('n_a')).sum())} | n_g q {r3(q(rc('n_g')))} min {fin(rc('n_g')).min():.2f} inf {int(np.isinf(rc('n_g')).sum())} | PR_b q {r3(q(rc('PR_b')))} | n_l2 (static l2) q {r3(q(rc('n_l2')))} max {rc('n_l2').max():.3f}")
    print(f"  TOP-3 geometry: R_i q {r3(q(rc('Ri')))} max {rc('Ri').max():.3f} | rho_g q {r3(q(rc('rho_g')))} | fbar^2/(g_i Th) q {r3(q(rc('fbar2')))} | |cos_G| gate max q {r3(q(rc('cosg_max')))} max {rc('cosg_max').max():.3f} (open only q {r3(q(rc('cosg_max_open')))}) | |cos_G| feature max q {r3(q(rc('cosf_max')))} max {rc('cosf_max').max():.3f} | own feature cos q {r3(q(rc('cosf_own')))}")
    print(f"  TOP-3 cap looseness at argmax unit: cap_g/(w d) q {r3(q(rc('capg_over_dG_kb')))} min {np.nanmin(rc('capg_over_dG_kb')):.2f} | cap_r/|w^r f| q {r3(q(rc('capr_over_dR_kb')))} min {np.nanmin(rc('capr_over_dR_kb')):.2f} | (cap_g+cap_r)/Delta q {r3(q(rc('cap_b_over_actual')))} min {np.nanmin(rc('cap_b_over_actual')):.2f}")
if mode in ("real", "all"):
    for i in np.where(alive)[0]:
        r = target(S, int(i)); r["self"] = self_term(S, int(i)); out["real"].append(r)
    T_ = out["real"]; print(f"REAL {tag} t{task}: {len(T_)} alive targets, A<=0 {sum(1 for t in T_ if t['A'] <= 0)} A^g<=0 {sum(1 for t in T_ if t['Ag'] <= 0)} | A/self q {r3(q([t['A'] / t['self'] for t in T_]))} [{time.time() - t0:.0f}s]"); summarize(T_)
if mode in ("helpers", "all"):
    files = sorted(glob.glob(os.path.join(S10, f"p155_helpers_{tag}_{os.environ.get('HELPERS_GLOB', 'g1_u*_k10')}.npz"))); tau_h = float(np.median(tau0[alive]))
    for f in files:
        i = int(os.path.basename(f).split("_u")[1].split("_k")[0]); z = np.load(f); Wh = z["Wh"]; gh = z["gh"]; k = Wh.shape[0]
        Wa = np.concatenate([W0, Wh], 0); ga = np.concatenate([g0, gh]); ta = np.concatenate([tau0, np.full(k, tau_h)])
        base = target(S, i); base["self"] = self_term(S, i); Sa = analyze(Wa, ga, ta, lam=lam0); helpers = list(range(H0, H0 + k)); r = target(Sa, i, helpers=helpers); r["self"] = self_term(Sa, i)
        hz = r["helpers"]; hr = [x["helpers"] for x in r["rows"]]
        rec = dict(i=i, k=k, before=base, after=r, lmaxG_max=float(Sa["lmaxG"].max()), lmaxX_max=float(Sa["lmaxX"].max()), gh=[float(x) for x in gh], gh_over_gi=float(gh.max() / g0[i]), g_i=float(g0[i])); out["helpers"].append(rec)
        print(f"HELPERS {tag} target {i} + {k}: A {base['A']:+.3g} -> {r['A']:+.3g} (A^g {base['Ag']:+.3g} -> {r['Ag']:+.3g}, A^r {base['Ar']:+.3g} -> {r['Ar']:+.3g}; gate-route hold {r['Ag'] <= 0}) | need_b {base['need_b']:.3f} -> {r['need_b']:.3f} (w^1+own at top-1: {base['rows'][0]['w1'] + base['rows'][0]['own_r']:.3f} -> {r['rows'][0]['w1'] + r['rows'][0]['own_r']:.3f}) | supply_b {base['sup_b']:.3f} -> {r['sup_b']:.3f}: orig units {hz['orig_sum_L']:.3f}, helpers {hz['sum_L']:.3f} = open {hz['sum_open']:.3f} + closed {hz['sum_closed']:.3f} + readout {hz['sum_r']:.3f} | helpers L {r3(sorted(hz['L'], reverse=True))} max {hz['max_L']:.4f} -> n_h = need_b/max {hz['n_h']:.1f}; n_b(all) {r['n_b']:.1f} (before {base['n_b']:.1f}) n_a {r['n_a']:.1f} (before {base['n_a']:.1f}) n_g {r['n_g']:.1f} (before {base['n_g']:.1f}) | helper covL {r3(hz['covL'])} cov {r3(hz['cov'])} | TOP-3: need_b {r3([x['need_b'] for x in r['rows']])} helpers' Delta sum open {r3([h['sum_open'] for h in hr])} closed {r3([h['sum_closed'] for h in hr])} readout {r3([h['sum_r'] for h in hr])} max {r3([h['max_delta'] for h in hr])} n_open {[h['n_open'] for h in hr]} | helpers |cos_G| gate max {max(h['cosg_max'] for h in hr):.3f} med {np.median([h['cosg_med'] for h in hr]):.3f}, feature max {max(h['cosf_max'] for h in hr):.3f} med {np.median([h['cosf_med'] for h in hr]):.3f} | cap/actual min: gate {min(h['capg_over_actual'] for h in hr):.2f} readout {min(h['capr_over_actual'] for h in hr):.2f} | R_i top3 {r3(base['Ri_top'] if 'Ri_top' in base else [x['Ri'] for x in base['rows']])} -> {r3([x['Ri'] for x in r['rows']])} | Gram {Sa['lmaxX'].max():.4f} ident {r['ident_err']:.1e} [{time.time() - t0:.0f}s]", flush=True)
if mode in ("scale", "all"):
    A0 = S["A"]; selfA = np.array([self_term(S, int(i)) if alive[i] else np.nan for i in range(H0)]); ratio = A0 / selfA
    targets = [int(i) for i in np.argsort(np.where(alive & (A0 > 0) & (selfA > 0), ratio, np.inf))[:5]]
    for i in targets:
        base = target(S, i); base["self"] = float(selfA[i]); rec = dict(i=i, factors=[1.0], recs=[base])
        line = f"SCALE {tag} target {i} (A/self {ratio[i]:.3f}): x1 A {base['A']:+.3g} need_b {base['need_b']:.3f} supply_b {base['sup_b']:.3f} maxL_b {base['max_L_b']:.4f} n_b {base['n_b']:.1f} n_g {base['n_g']:.1f} R_i {np.mean([x['Ri'] for x in base['rows']]):.3f} cos_g {max(x['cosg_max'] for x in base['rows']):.3f} cos_f {max(x['cosf_max'] for x in base['rows']):.3f} maxDelta(top1) {base['rows'][0]['max_delta_b']:.4f}"
        for fac in (10.0, 100.0):
            gs = g0 * fac; gs[i] = g0[i]; Ss = analyze(W0, gs, tau0, lam=lam0); r = target(Ss, i); r["self"] = self_term(Ss, i); rec["factors"].append(fac); rec["recs"].append(r)
            line += f" | x{fac:g}: A {r['A']:+.3g} (self {r['self']:+.3g}) need_b {r['need_b']:.3f} supply_b {r['sup_b']:.3f} [open {r['sup_open']:.3f} closed {r['sup_closed']:.3f} readout {r['sup_r']:.3f}] maxL_b {r['max_L_b']:.4f} (x{r['max_L_b'] / base['max_L_b']:.2f} vs sqrt {np.sqrt(fac):.2f}) n_b {r['n_b']:.1f} n_g {r['n_g']:.1f} R_i {np.mean([x['Ri'] for x in r['rows']]):.3f} cos_g {max(x['cosg_max'] for x in r['rows']):.3f} cos_f {max(x['cosf_max'] for x in r['rows']):.3f} maxDelta(top1) {r['rows'][0]['max_delta_b']:.4f} (x{r['rows'][0]['max_delta_b'] / max(base['rows'][0]['max_delta_b'], 1e-300):.2f}) Gram {Ss['lmaxX'].max():.3f}"
        out["scale"].append(rec); print(line + f" [{time.time() - t0:.0f}s]", flush=True)
out["src"] = src; out["state_note"] = f"state end0[{task - 2}] = stoch t {task} = M1 T {task - 1}"
json.dump(out, open(os.path.join(OUTDIR, f"p172c_{tag}_t{task}_{mode}{os.environ.get('OUT_SUFFIX', '')}.json"), "w"), indent=1); print("done", f"{time.time() - t0:.0f}s")
