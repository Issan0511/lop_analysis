"""C2 (xbar_arrangement_1014): isotropic-direction null for the gate push (and readout part), M fixed.
Per alive unit i: actual z_i = Xt w_i, m_i = mean_n z_in, K_n = x~_n . xbar, Kbar = xbar . xbar (= mean_n K_n), s_perp_i = sd_n(z_in - m_i K_n / Kbar).
Draw g ~ N(0, I_785), g <- g - xbar (xbar.g)/Kbar, u = Xt g rescaled to sd_n(u) = s_perp_i, z_iso = m_i K/Kbar + u.
e'_iso = K * Dp(z_iso), d_iso = D(z_iso), f_iso = F(z_iso) with the unit's own tau_i (= 0.1 sd_n(z_i), as in the P149 script).
A^g,iso = e'_iso^T M d_iso for M = full (Theta o gamma P_i) and probe (Theta o gamma P^B_i); readout A^r,iso = (K * d_iso)^T Ebar f_iso. 150 draws per unit (rng 20261014 + 10*seed + tag index).
Kernel/state construction = c_strat_expect.py (P149) prefix.  usage: c2_iso_expect.py <a01|a03|a05> [seed=0]"""
import os, sys, json, time
NT = os.environ.get("C_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = NT
import numpy as np
T0 = time.time()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_strat_expect.py")).read()
exec(src.split("print(f\"[{time.time() - T0:.0f}s] {tag} s{SEED}: kernel inverted")[0])          # state, kernel, E4 (identical code path to P149)
print(f"[{time.time() - T0:.0f}s] {tag} s{SEED}: kernel inverted, lam={lam:.6g}", flush=True)
Ebar = sum(E4[c, :, c, :] for c in range(Kp)); Ip = np.eye(Xt.shape[1]); NDRAW = 150
def lemma(M, x, y):
    n = len(x); mud = np.trace(M) / n; tot = M.sum(); muo = (tot - np.trace(M)) / (n * (n - 1)); cov = np.mean(x * y) - x.mean() * y.mean()
    return n * (mud - muo) * cov + tot * x.mean() * y.mean()
def parts1(z, t):                                  # same formulas as p128 `parts`, one unit, z may be (N,) or (N, k)
    s = 0.5 * (1 + np.tanh(0.5 * z / t)); return a + (1 - a) * s, (1 - a) * s * (1 - s) / t, a * z + (1 - a) * t * np.logaddexp(0, z / t)
xbar = Xt.mean(0); Kbar = float(xbar @ xbar); assert abs(Kbar - Kv.mean()) < 1e-9 * Kbar
rng_i = np.random.default_rng(20261014 + 10 * SEED + {"a01": 1, "a03": 3, "a05": 5}[tag])
def q(x, p): return float(np.percentile(x, p))
rows = []; alive_units = np.where((Z > 0).any(0))[0]
for j, i in enumerate(alive_units):
    z = Z[:, i]; t = tau[i]; d = D[:, i]; e = Kv * Dp[:, i]; f = F[:, i]; u = Kv * d
    D1, Dp1, F1 = parts1(z, t); assert np.allclose(D1, d, rtol=0, atol=1e-12) and np.allclose(Dp1, Dp[:, i], rtol=1e-10, atol=1e-12) and np.allclose(F1, f, rtol=1e-10, atol=1e-10)
    P = np.einsum("a,b,anbm->nm", V[i], V[i], E4); Mf = Th * P
    Y = d[:, None] * Xt; PY = P @ Y; G = Y.T @ PY; PB = P + PY @ np.linalg.solve(Ip - G, PY.T); Mb = Th * PB
    Ag = float(e @ Mf @ d); Aw = float(e @ Mb @ d); Ar = float(u @ Ebar @ f); Ef = lemma(Mf, e, d); Eb = lemma(Mb, e, d); Er = lemma(Ebar, u, f)
    m = float(z.mean()); zperp = z - m * Kv / Kbar; sperp = float(zperp.std())
    Gd = rng_i.normal(size=(NDRAW, Xt.shape[1])); Gd -= np.outer(Gd @ xbar, xbar) / Kbar; U = Xt @ Gd.T                # (N, k), column means 0
    U *= sperp / U.std(0, keepdims=True); Zi = m * Kv[:, None] / Kbar + U
    Di, Dpi, Fi = parts1(Zi, t); Ei = Kv[:, None] * Dpi; Ui = Kv[:, None] * Di
    vf = np.einsum("nk,nk->k", Ei, Mf @ Di); vb = np.einsum("nk,nk->k", Ei, Mb @ Di); vr = np.einsum("nk,nk->k", Ui, Ebar @ Fi)
    p_iso = (Zi > 0).mean(0); p_act = float((z > 0).mean())
    r = dict(unit=int(i), p_act=p_act, p_iso_mean=float(p_iso.mean()), p_iso_med=float(np.median(p_iso)), p_iso_zero=float((p_iso == 0).mean()), m=m, s_perp=sperp, tau=float(t),
             mass_act=float((d - a).sum()), mass_iso_mean=float((Di - a).sum(0).mean()), emass_act=float(e.sum()), emass_iso_mean=float(Ei.sum(0).mean()))
    for k, A, E, v in (("f", Ag, Ef, vf), ("b", Aw, Eb, vb), ("r", Ar, Er, vr)):
        r.update({f"A_{k}": A, f"Efull_{k}": float(E), f"Eiso_{k}": float(v.mean()), f"sdiso_{k}": float(v.std()), f"q025_{k}": q(v, 2.5), f"q975_{k}": q(v, 97.5),
                  f"rank_{k}": float((v < A).mean())})
    rows.append(r)
    if j % 20 == 0: print(f"[{time.time() - T0:.0f}s] unit {j + 1}/{len(alive_units)}", flush=True)
OUT = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
json.dump(dict(meta=dict(tag=tag, seed=SEED, a=a, N=N, lam=lam, Kbar=Kbar, ndraw=NDRAW, threads=NT, script="analysis/xbar_arrangement_1014/c2_iso_expect.py", elapsed_s=time.time() - T0), rows=rows),
          open(os.path.join(OUT, f"c2_iso_expect_{tag}_s{SEED}.json"), "w"))
f_ = lambda k: np.array([r[k] for r in rows]); n = len(rows)
for k, nm in (("f", "full"), ("b", "probe"), ("r", "readout")):
    A = f_(f"A_{k}"); E0 = f_(f"Efull_{k}"); Ei = f_(f"Eiso_{k}"); si = f_(f"sdiso_{k}"); zi = (A - Ei) / si; den = E0 - A; ok = den > 0; gap = (E0 - Ei)[ok] / den[ok]; ra = A / Ei
    print(f"  {nm}: actual/E_iso med {np.median(ra):.2f} [{q(ra, 10):.2f}, {q(ra, 90):.2f}] | z_iso med {np.median(zi):+.1f} [{q(zi, 10):+.1f}, {q(zi, 90):+.1f}] | actual<E_iso {int((A < Ei).sum())}/{n} | gap closed med {np.median(gap):+.2f} (den<=0 {int((~ok).sum())}) | E_iso>0 {int((Ei > 0).sum())}/{n} | |z|<=2 {int((np.abs(zi) <= 2).sum())}/{n}")
pr = f_("p_iso_mean") / f_("p_act"); print(f"  p_iso/p_act med {np.median(pr):.2f} [{q(pr, 10):.2f}, {q(pr, 90):.2f}] | p_act med {np.median(f_('p_act')):.4f} | p_iso med {np.median(f_('p_iso_mean')):.4f} | draws with p_iso=0 (mean frac) {f_('p_iso_zero').mean():.3f}")
print(f"[{time.time() - T0:.0f}s] done", flush=True)
