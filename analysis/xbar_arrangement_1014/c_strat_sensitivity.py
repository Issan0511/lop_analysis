"""C extra (xbar_arrangement_1014): resolution sensitivity of the stratified expectation (exact closed form only, no MC).
Strata = equal-count bins of K (Kv = x~ . x~bar) with S in {2, 5, 10, 20, 40, 120}, and deciles of |x~|^2 (diag Theta) instead of K.  S = N would give E_strat = actual (gap closed = 1 by construction).
Same state construction as c_strat_expect.py (P149 kernel).  usage: c_strat_sensitivity.py <a01|a03|a05> [seed=0]"""
import os, sys, json, time
NT = os.environ.get("C_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = NT
import numpy as np
T0 = time.time()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_strat_expect.py")).read()
exec(src.split("print(f\"[{time.time() - T0:.0f}s] {tag} s{SEED}: kernel inverted")[0])          # state, kernel, E4 (identical code path)
Ebar = sum(E4[c, :, c, :] for c in range(Kp)); Ip = np.eye(Xt.shape[1])
def binning(key, S):
    order = np.argsort(key, kind="stable"); dec = np.empty(N, int); dec[order] = np.arange(N) * S // N; return dec
def strat_exact(M, x, y, dec, S):
    Bm = np.zeros((N, S)); Bm[np.arange(N), dec] = 1.0; ns = Bm.sum(0)
    Ms = Bm.T @ M @ Bm; tr = np.bincount(dec, weights=np.diag(M), minlength=S); Sx = Bm.T @ x; Sy = Bm.T @ y; Sxy = Bm.T @ (x * y)
    xb = Sx / ns; yb = Sy / ns; O = Ms.copy(); np.fill_diagonal(O, 0.0)
    return float(tr @ (Sxy / ns) + (np.diag(Ms) - tr) @ ((Sx * Sy - Sxy) / (ns * (ns - 1))) + xb @ O @ yb)
def lemma(M, x, y):
    n = len(x); mud = np.trace(M) / n; tot = M.sum(); muo = (tot - np.trace(M)) / (n * (n - 1)); cov = np.mean(x * y) - x.mean() * y.mean()
    return n * (mud - muo) * cov + tot * x.mean() * y.mean()
SCHEMES = [(f"K{S}", Kv, S) for S in (2, 5, 10, 20, 40, 120)] + [("xn10", np.diag(Th).copy(), 10)]
bins = {nm: (binning(key, S), S) for nm, key, S in SCHEMES}
rows = []
for i in np.where((Z > 0).any(0))[0]:
    d = D[:, i]; e = Kv * Dp[:, i]; f = F[:, i]; u = Kv * d; P = np.einsum("a,b,anbm->nm", V[i], V[i], E4)
    Mf = Th * P; Y = d[:, None] * Xt; PY = P @ Y; G = Y.T @ PY; PB = P + PY @ np.linalg.solve(Ip - G, PY.T); Mb = Th * PB
    r = dict(unit=int(i), Ag=float(e @ Mf @ d), Aw=float(e @ Mb @ d), Ar=float(u @ Ebar @ f), Ef=float(lemma(Mf, e, d)), Eb=float(lemma(Mb, e, d)), Er=float(lemma(Ebar, u, f)))
    for nm, (dec, S) in bins.items(): r[f"f_{nm}"] = strat_exact(Mf, e, d, dec, S); r[f"b_{nm}"] = strat_exact(Mb, e, d, dec, S); r[f"r_{nm}"] = strat_exact(Ebar, u, f, dec, S)
    rows.append(r)
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
corr = float(np.corrcoef(Kv, np.diag(Th))[0, 1])
json.dump(dict(meta=dict(tag=tag, seed=SEED, schemes=[s[0] for s in SCHEMES], corr_K_xnorm2=corr, elapsed_s=time.time() - T0), rows=rows), open(os.path.join(OUT, f"c_strat_sensitivity_{tag}_s{SEED}.json"), "w"))
f_ = lambda k: np.array([r[k] for r in rows])
print(f"{tag} s{SEED}: units {len(rows)} | corr(K, |x~|^2) {corr:.3f}")
for p, A, E in (("f", "Ag", "Ef"), ("b", "Aw", "Eb"), ("r", "Ar", "Er")):
    out = []
    for nm, _, _ in SCHEMES:
        Es = f_(f"{p}_{nm}"); den = f_(E) - f_(A); ok = den > 0; gap = (f_(E) - Es)[ok] / den[ok]
        out.append(f"{nm}: gap {np.median(gap):.2f} a/Es {np.median(f_(A) / Es):.2f} Es>0 {int((Es > 0).sum())} a<Es {int((f_(A) < Es).sum())}")
    print(f"  {dict(f='full', b='probe', r='readout')[p]}: " + " | ".join(out))
print(f"[{time.time() - T0:.0f}s] done", flush=True)
