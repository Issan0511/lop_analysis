#!/usr/bin/env python3
"""Exact checks of the accounting algebra in README.md (analysis/bridge_time_average_1013).
1 eq.(2)=>(3)  2 eq.(4)=>(5)  3 Adam summation by parts eq.(7)  4 tower identity eq.(6)  5 CE split
L_Y = L_U - G_Y (sec.5). Inequalities are judged only in exact arithmetic (Fraction / sympy). Floats
appear only in 3 (torch Adam run) and 5 (torch autograd), where max-abs residuals are compared with
n_ops*eps64*scale. Every check has a mutation control that must FAIL (so a pass is not vacuous).
Run: OMP_NUM_THREADS=1 <venv>/bin/python verify_accounting.py  -> verify_accounting.json"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools, json, random, time
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
T0, rng, EPS = time.time(), random.Random(20261013), np.finfo(np.float64).eps
OUT, R = Path(__file__).with_name("verify_accounting.json"), {}
K, Al = sp.symbols("kappa alpha", positive=True)
rq = lambda lo, hi, den=12: Fr(rng.randint(lo * den, hi * den), den)   # random rational, grid 1/den
P3 = (Fr(1, 2), Fr(1, 3), Fr(1, 6))      # 3-path ensemble in which (2) holds only for the means

def kappa_alpha(kmax):                       # 0 <= alpha < kappa <= kmax
    k, a = rq(0, kmax), rq(0, 1)
    return (k, a) if a < k else kappa_alpha(kmax)

def bounds(k, a, s, x0, xT):                 # the two right-hand sides of (3) (x = u) or (5) (x = V)
    return (1 - a / k) * sum(s) - (x0 - xT) / k, (1 - a / k) * sum(s) - x0 / k

def split_mean(x):                           # nonnegative per-path values whose P3-mean is exactly x
    w = [rq(0, 2) + Fr(1, 12) for _ in P3]
    return [x * wi / sum(p * wj for p, wj in zip(P3, w)) for wi in w]

# ---------- 1/2 numeric in exact Fractions; E = identity: A_t = S_t - U_t, u_t = U_t ----------
def numeric(eq):
    c = dict.fromkeys(["trials", "viol_first", "viol_second", "tight_trials", "tight_not_equal",
                       "mutation_kappa_dropped_violations"]
                      + (["ensemble_viol", "ensemble_paths", "ensemble_paths_violating_eq2"] if eq == 2 else []), 0)
    for T in (1, 2, 3, 5, 10, 40):
        for i in range(400):
            k, a = kappa_alpha(1 if eq == 2 else 2)      # (4) needs only 0 <= alpha < kappa
            tight, s = i % 4 == 0, [rq(0, 3) for _ in range(T)]   # tight: slack 0 in (2)/(4)
            if eq == 2:                                  # u_{t+1} in [0, (1-k)u_t + a s_t]
                u = x = [rq(0, 5)]
                for t in range(T):
                    u.append(((1 - k) * u[t] + a * s[t]) * (1 if tight else 1 - rq(0, 1)))
            else:                                        # u_t <= (V_t + a s_t)/k keeps V_{t+1} >= 0
                u, x = [], [rq(0, 5)]
                for t in range(T):
                    room = x[t] + a * s[t]
                    u.append(room / k * rq(0, 1))
                    x.append((room - k * u[t]) * (1 if tight else 1 - rq(0, 1)))  # dV = a s - k u - slack
            SA = sum(s[t] - u[t] for t in range(T))
            B1, B2 = bounds(k, a, s, x[0], x[T])
            c["trials"] += 1; c["viol_first"] += SA < B1; c["viol_second"] += B1 < B2
            c["tight_trials"] += tight; c["tight_not_equal"] += bool(tight and SA != B1)
            c["mutation_kappa_dropped_violations"] += SA < (1 - a) * sum(s) - (x[0] - x[T])
            if eq == 2:
                Us, Ss = [split_mean(v) for v in u], [split_mean(v) for v in s]
                c["ensemble_viol"] += sum(p * (Ss[t][j] - Us[t][j]) for t in range(T) for j, p in enumerate(P3)) < B1
                c["ensemble_paths"] += len(P3)
                c["ensemble_paths_violating_eq2"] += sum(any(
                    Us[t + 1][j] > (1 - k) * Us[t][j] + a * Ss[t][j] for t in range(T)) for j in range(len(P3)))
    return c

# ---------- 1/2 symbolic: general-T telescoping + exact certificates for T = 1..6 ----------
t_, T_ = sp.symbols("t T", integer=True, positive=True)
uI, sI = sp.IndexedBase("u"), sp.IndexedBase("s")
z = lambda x: sp.simplify(x) == 0
general_T = dict(
    summand_is_telescoping_plus_linear=z((1 - K) * uI[t_] + Al * sI[t_] - uI[t_ + 1]
                                         - (uI[t_] - uI[t_ + 1] + Al * sI[t_] - K * uI[t_])),
    telescoped_sum_equals_u0_minus_uT=sp.Sum(uI[t_] - uI[t_ + 1], (t_, 0, T_ - 1)).doit() == uI[0] - uI[T_])
sym1, sym2 = {}, {}
for T in range(1, 7):
    s, u, e, V = (sp.symbols(f"{c}0:{T + n}") for c, n in (("s", 0), ("u", 1), ("e", 0), ("V", 1)))
    SA = sum(s[t] - u[t] for t in range(T))
    red2 = Al * sum(s) + u[0] - u[T] - K * sum(u[:T])   # (3) <=> kappa sum u <= alpha sum s + u0 - uT
    red4 = Al * sum(s) + V[0] - V[T] - K * sum(u[:T])
    u_sl = [u[0]]
    for t in range(T):
        u_sl.append((1 - K) * u_sl[t] + Al * s[t] - e[t])   # (2) with slack e_t >= 0
    (B1, B2), (C1, C2) = bounds(K, Al, s, u[0], u[T]), bounds(K, Al, s, V[0], V[T])
    sym1[T] = dict(summed_eq2_equals_reduced=z(sum((1 - K) * u[t] + Al * s[t] - u[t + 1] for t in range(T)) - red2),
                   kappa_x_first_gap_equals_reduced=z(K * (SA - B1) - red2),
                   second_gap_equals_uT_over_kappa=z(B1 - B2 - u[T] / K),
                   reduced_equals_sum_of_slacks=z(red2.subs(dict(zip(u, u_sl))) - sum(e)))
    sym2[T] = dict(summed_eq4_equals_reduced=z(sum(Al * s[t] - K * u[t] - (V[t + 1] - V[t]) for t in range(T)) - red4),
                   kappa_x_first_gap_equals_reduced=z(K * (SA - C1) - red4),
                   second_gap_equals_VT_over_kappa=z(C1 - C2 - V[T] / K),
                   eq2_is_eq4_with_V_eq_u=all(z((1 - K) * u[t] + Al * s[t] - u[t + 1]
                                                - (Al * s[t] - K * u[t] - (u[t + 1] - u[t]))) for t in range(T)))
for eq, key, sym in ((2, "check1_eq3_from_eq2", sym1), (4, "check2_eq5_from_eq4", sym2)):
    c = numeric(eq)
    ok = c["viol_first"] == c["viol_second"] == c["tight_not_equal"] == c.get("ensemble_viol", 0) == 0
    ok &= all(all(d.values()) for d in sym.values()) and (eq == 4 or all(general_T.values()))
    R[key] = dict(pass_=bool(ok), numeric_exact_fraction=c, symbolic_small_T=sym,
                  mutation_failed_as_expected=c["mutation_kappa_dropped_violations"] > 0)
R["check1_eq3_from_eq2"]["symbolic_general_T"] = general_T

# ---------- 3. eq.(7): Adam discrete summation by parts ----------
b, ac = sp.symbols("beta a")
VARIANTS = {  # name -> (sign of the a.m_0 term, which a multiplies m_0, flipped a-difference, coefficient)
    "printed": (-1, "1", False, b / (1 - b)), "mut_m0_sign_flipped": (+1, "1", False, b / (1 - b)),
    "mut_m0_uses_aT": (-1, "T", False, b / (1 - b)), "mut_adiff_flipped": (-1, "1", True, b / (1 - b)),
    "mut_coef_beta": (-1, "1", False, b)}

def rhs7(a, g, m, T, v):                     # RHS of (7) exactly as printed (or a mutated variant)
    sgn, idx, flip, coef = VARIANTS[v]
    bracket = a[T] * m[T] + sgn * a[1 if idx == "1" else T] * m[0] + sum(
        ((a[k + 1] - a[k]) if flip else (a[k] - a[k + 1])) * m[k] for k in range(1, T))
    return sum(a[k] * g[k] for k in range(1, T + 1)) - coef * bracket, bracket

fails7, proof_step, fixed_a = {v: [] for v in VARIANTS}, True, True
for T in range(1, 7):
    a, g, m = [None, *sp.symbols(f"a1:{T + 1}")], [None, *sp.symbols(f"g1:{T + 1}")], [sp.Symbol("m0")]
    for k in range(1, T + 1):
        m.append(b * m[k - 1] + (1 - b) * g[k])
    lhs = sum(a[k] * m[k] for k in range(1, T + 1))
    for v in VARIANTS:
        if sp.cancel(lhs - rhs7(a, g, m, T, v)[0]) != 0:
            fails7[v].append(T)
    proof_step &= all(sp.cancel(g[k] - (m[k] + b * (m[k] - m[k - 1]) / (1 - b))) == 0 for k in range(1, T + 1))
    fixed_a &= z(rhs7(a, g, m, T, "printed")[1].subs({a[k]: ac for k in range(1, T + 1)}) - ac * (m[T] - m[0]))

def adam_run(G, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    p = torch.zeros(G.shape[1], dtype=torch.float64, requires_grad=True)
    opt, th = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps), [np.zeros(G.shape[1])]
    for gk in G:
        p.grad = torch.from_numpy(gk.copy()); opt.step(); th.append(p.detach().numpy().copy())
    m, v, M, A = np.zeros(G.shape[1]), np.zeros(G.shape[1]), [np.zeros(G.shape[1])], [None]
    for k, gk in enumerate(G, 1):                   # a_k = lr / ((1 - b1^k)(sqrt(vhat_k) + eps))
        m, v = b1 * m + (1 - b1) * gk, b2 * v + (1 - b2) * gk ** 2
        M.append(m); A.append(lr / ((1 - b1 ** k) * (np.sqrt(v / (1 - b2 ** k)) + eps)))
    return np.array(th), M, A

rs = np.random.default_rng(1013)
G = (0.3 + rs.standard_normal((600, 6))) * np.exp(rs.standard_normal((600, 1)))   # time-varying scale
G[:, 0] = 0.900001 - rs.binomial(16, 0.9, 600) / 16       # finite-label gradient of README sec.6
th, M, A = adam_run(G)
num7, cb, ok_num7 = {}, 0.9 / (1 - 0.9), True
for k0, T in ((0, 600), (200, 300)):                     # window from init (m_0 = 0) / inherited m_0
    ks = range(k0 + 1, k0 + T + 1)
    lhs = sum(A[k] * M[k] for k in ks)
    br = A[k0 + T] * M[k0 + T] - A[k0 + 1] * M[k0] + sum((A[k] - A[k + 1]) * M[k] for k in range(k0 + 1, k0 + T))
    rhs = sum(A[k] * G[k - 1] for k in ks) - cb * br
    scale = sum(np.abs(A[k] * G[k - 1]) + (1 + 2 * cb) * np.abs(A[k] * M[k]) for k in ks) + cb * np.abs(A[k0 + 1] * M[k0])
    d = dict(max_abs_residual_eq7=float(np.max(np.abs(lhs - rhs))), tol=float(np.max(4 * T * EPS * scale)),
             max_abs_torch_displacement_minus_sum_a_m=float(np.max(np.abs(th[k0] - th[k0 + T] - lhs))),
             tol_disp=float(4 * T * EPS * np.max(np.abs(th))), max_abs_sum_a_m=float(np.max(np.abs(lhs))),
             max_abs_m0=float(np.max(np.abs(M[k0]))),
             mut_m0_sign_flipped_residual=float(np.max(np.abs(lhs - (rhs - 2 * cb * A[k0 + 1] * M[k0])))))
    ok_num7 &= d["max_abs_residual_eq7"] <= d["tol"] and d["max_abs_torch_displacement_minus_sum_a_m"] <= d["tol_disp"]
    num7[f"window_k0={k0}_T={T}"] = d
eq7_exact, w2 = fails7["printed"] == [], num7["window_k0=200_T=300"]
R["check3_eq7_adam_summation_by_parts"] = dict(
    pass_=bool(eq7_exact and proof_step and fixed_a and ok_num7), printed_form_exact_for_T_1_to_6=eq7_exact,
    symbolic_T_values_where_LHS_minus_RHS_nonzero=fails7, proof_step_g_substitution_ok=proof_step,
    fixed_a_bracket_is_a_times_mT_minus_m0=fixed_a, numeric_float64_vs_torch_adam=num7,
    mutation_failed_as_expected=bool(all(fails7[v] for v in VARIANTS if v != "printed")
                                     and w2["mut_m0_sign_flipped_residual"] > w2["tol"]))

# ---------- 4. eq.(6): E[q0 - qT] = sum_t E b_t, b_t = E[D_t | F_t], F_t = sigma(X_0..X_t) ----------
q4, thr, S4 = [Fr(3), Fr(-1), Fr(5, 2), Fr(0)], Fr(1, 2), range(4)

def rand_dist():
    w = [rng.randint(1, 9) for _ in S4]
    return {x: Fr(wx, sum(w)) for x, wx in enumerate(w)}

mu4, P4, hist_table = rand_dist(), [rand_dist() for _ in S4], {}
kernels = {"markov": lambda h: P4[h[-1]],
           "history_dependent": lambda h: hist_table[h] if h in hist_table else hist_table.setdefault(h, rand_dist())}

def tower(T, kn):
    paths = []

    def rec(h, p):                                   # enumerate every path with its exact probability
        if len(h) == T + 1:
            return paths.append((p, h))
        for x, px in kernels[kn](h).items():
            rec(h + (x,), p * px)
    for x0, p0 in mu4.items():
        rec((x0,), p0)
    b = []                       # b_t by definition: mean of D_t over the paths sharing X_0..X_t
    for t in range(T):
        num, den = {}, {}
        for p, h in paths:
            num[h[:t + 1]] = num.get(h[:t + 1], 0) + p * (q4[h[t]] - q4[h[t + 1]])
            den[h[:t + 1]] = den.get(h[:t + 1], 0) + p
        b.append({key: num[key] / den[key] for key in num})
    E = lambda f: sum(p * f(h) for p, h in paths)
    Eb = [E(lambda h, t=t: b[t][h[:t + 1]]) for t in range(T)]
    o = dict(n_paths=len(paths), total_prob=E(lambda h: 1), lhs=E(lambda h: q4[h[0]] - q4[h[T]]),
             sum_Eb=sum(Eb), mut_off_by_one=sum(Eb[:-1]))
    tau = lambda h: next((t for t in range(T + 1) if q4[h[t]] < thr), T + 1)       # stopping time
    tau_bad = lambda h: next((t for t in range(T) if q4[h[t + 1]] < thr), T + 1)   # peeks one step ahead
    for name, tf in (("stopped", tau), ("mut_nonstopping", tau_bad)):
        o[name] = [E(lambda h: q4[h[0]] - q4[h[min(T, tf(h))]]),
                   sum(E(lambda h, t=t: b[t][h[:t + 1]] * (t < tf(h))) for t in range(T))]
    o["ok"] = o["total_prob"] == 1 and o["lhs"] == o["sum_Eb"] and o["stopped"][0] == o["stopped"][1]
    o["mutations_fail"] = o["mut_off_by_one"] != o["lhs"] and o["mut_nonstopping"][0] != o["mut_nonstopping"][1]
    if kn == "markov":           # cross-check b_t = q - Pq; mutation: uniform "reference" law of X_t
        drift = [q4[x] - sum(P4[x][y] * q4[y] for y in S4) for x in S4]
        o["b_equals_q_minus_Pq"] = all(b[t][key] == drift[key[-1]] for t in range(T) for key in b[t])
        o["mut_uniform_reference_average"] = T * sum(drift) / 4
        o["ok"] &= o["b_equals_q_minus_Pq"]
        o["mutations_fail"] &= o["mut_uniform_reference_average"] != o["lhs"]
    return o

tw = {f"{kn}_T={T}": tower(T, kn) for kn in kernels for T in (1, 2, 3, 4)}
R["check4_eq6_tower_identity"] = dict(pass_=all(o["ok"] for o in tw.values()), cases=tw,
                                      mutation_failed_as_expected=all(tw[f"{kn}_T=3"]["mutations_fail"] for kn in kernels))

# ---------- 5. CE split with one-hot labels: exact on symbolic logits, then torch autograd ----------
N, C, Din, H = 3, 3, 5, 4
labels = list(itertools.product(range(C), repeat=N))                       # all 27 label vectors
fs = [[sp.Symbol(f"f{n}{c}") for c in range(C)] for n in range(N)]
ys = [[sp.Symbol(f"y{n}{c}") for c in range(C)] for n in range(N)]
lse = [sp.log(sum(sp.exp(x) for x in fs[n])) for n in range(N)]
LU_s = sum(lse[n] - sum(fs[n]) / C for n in range(N)) / N
G_aff = sum((ys[n][c] - sp.Rational(1, C)) * fs[n][c] for n in range(N) for c in range(C)) / N
ok_split = ok_aff = True
avgL = avgG = 0
for Y in labels:
    LY_s, GY_s = (sum(lse[n] - fs[n][Y[n]] for n in range(N)) / N, sum(fs[n][Y[n]] - sum(fs[n]) / C for n in range(N)) / N)
    ok_split &= sp.expand(LY_s - (LU_s - GY_s)) == 0
    ok_aff &= sp.expand(GY_s - G_aff.subs({ys[n][c]: int(Y[n] == c) for n in range(N) for c in range(C)})) == 0
    avgL, avgG = avgL + LY_s / len(labels), avgG + GY_s / len(labels)
exact5 = dict(split_identity_all_27_labels=ok_split, G_equals_affine_form_in_onehot_all_27=ok_aff,
              uniform_label_avg_of_G_identically_zero=sp.expand(avgG) == 0,
              uniform_label_avg_of_L_equals_LU=sp.expand(avgL - LU_s) == 0)
gen = torch.Generator().manual_seed(1013)
X = torch.randn(N, Din, generator=gen, dtype=torch.float64)
design = np.array([[1.0] + [float(Y[n] == c) for n in range(N) for c in range(C)] for Y in labels])
w_nonuni = np.array([np.prod([(0.5, 1 / 3, 1 / 6)[y] for y in Y]) for Y in labels])   # mutation weights
diag = [i for i, Y in enumerate(labels) if len(set(Y)) == 1]   # Y1=Y2=Y3: dependent, marginally uniform
t5, upd = {}, lambda key, val: t5.__setitem__(key, max(t5.get(key, 0.0), float(val)))
fit_res = lambda y: np.abs(design @ np.linalg.lstsq(design, y, rcond=None)[0] - y).max()
for trial in range(5):
    theta = [torch.randn(*sh, generator=gen, dtype=torch.float64).requires_grad_() for sh in ((H, Din), (H,), (C, H), (C,))]
    W1, b1, W2, b2 = theta
    f = F.leaky_relu(X @ W1.T + b1, 0.1) @ W2.T + b2                         # N x C logits
    flat = lambda L: torch.cat([q.reshape(-1) for q in torch.autograd.grad(L, theta, retain_graph=True)]).numpy()
    LU = (torch.logsumexp(f, 1) - f.mean(1)).mean()
    gU, gL, gG, Gv = flat(LU), [], [], []
    for Y in labels:
        Yt = torch.tensor(Y)
        LY, GY = F.cross_entropy(f, Yt), (f[torch.arange(N), Yt] - f.mean(1)).mean()
        upd("a_max_abs_LY_minus_LU_plus_GY", abs(LY - (LU - GY)).item())
        gL.append(flat(LY)); gG.append(flat(GY)); Gv.append(GY.item())
    gL, gG, Gv = np.array(gL), np.array(gG), np.array(Gv)
    upd("b_max_abs_avg_gradLY_minus_gradLU", np.abs(gL.mean(0) - gU).max())
    upd("b_max_abs_avg_gradGY_uniform", np.abs(gG.mean(0)).max())
    upd("b_info_max_abs_avg_gradGY_diag_labels", np.abs(gG[diag].mean(0)).max())
    upd("c_max_abs_affine_fit_residual_GY", fit_res(Gv))
    upd("mut_b_max_abs_avg_gradGY_nonuniform_labels", np.abs(w_nonuni @ gG).max())
    upd("mut_c_max_abs_affine_fit_residual_GY_squared", fit_res(Gv ** 2))
    upd("scale_max_abs_grad_or_logit", max(np.abs(gL).max(), np.abs(gG).max(), f.abs().max().item()))
tol5 = 64 * len(labels) * EPS * max(1.0, t5["scale_max_abs_grad_or_logit"])
R["check5_CE_split"] = dict(
    pass_=bool(all(exact5.values()) and all(t5[k] <= tol5 for k in t5 if k[:2] in ("a_", "b_", "c_"))),
    exact_sympy_on_logits=exact5, torch_float64=t5, tol=tol5,
    mutation_failed_as_expected=bool(min(t5["mut_b_max_abs_avg_gradGY_nonuniform_labels"],
                                         t5["mut_c_max_abs_affine_fit_residual_GY_squared"]) > tol5))

# ---------- summary ----------
checks = [k for k in R if k.startswith("check")]
R |= dict(all_pass=all(R[k]["pass_"] for k in checks), all_mutations_fail=all(R[k]["mutation_failed_as_expected"] for k in checks),
          eq7_as_printed_exactly_correct=eq7_exact, runtime_s=round(time.time() - T0, 1),
          discrepancies=[] if eq7_exact else [f"eq.(7) as printed fails for T in {fails7['printed']}"])
conv = lambda o: str(o) if isinstance(o, Fr) else bool(o) if isinstance(o, (np.bool_, bool)) else float(o)
OUT.write_text(json.dumps(R, indent=1, default=conv, ensure_ascii=False))
for k in checks:
    print(f"{k:38s} pass={R[k]['pass_']}  mutation_fails={R[k]['mutation_failed_as_expected']}")
print("eq.(7) printed form exact (sympy, T=1..6):", eq7_exact, "| T where each variant fails:", fails7)
for w, d in num7.items():
    print(f"  Adam {w}: |res7|={d['max_abs_residual_eq7']:.2e} (tol {d['tol']:.1e}), "
          f"|torch disp - sum a m|={d['max_abs_torch_displacement_minus_sum_a_m']:.2e} (tol {d['tol_disp']:.1e})")
print("check4 T=3 lhs == sum E b_t:", {kn: tw[f"{kn}_T=3"]["lhs"] == tw[f"{kn}_T=3"]["sum_Eb"] for kn in kernels})
print("check5:", {k: f"{v:.2e}" for k, v in t5.items()}, f"tol={tol5:.1e}")
print(f"all_pass={R['all_pass']} all_mutations_fail={R['all_mutations_fail']} runtime={R['runtime_s']}s -> {OUT.name}")
