"""1 層の保存状態で、記憶のフィルタの式（読み出しの NTK）が実際の押しをどこまで出すか。訓練はしない（前向き計算と固有分解だけ）。
模型: 特徴 Φ = φ(z)（切替時点の活動）を固定し、読み出し＋出力 bias を二乗誤差の勾配流で τ だけ学習（ラベル独立な初期値）。
  E[S^oh_i] = (1-1/K) b_i^T F_τ φ_i、F_τ = Θ^+(I - e^{-τΘ})、b_i = K_n φ'(z_in)
  P(S^oh_i > 0) = ∫_0^{π-arccos ρ} sin^{K-2} / ∫_0^π sin^{K-2}、ρ_i = cos(b_i, F_τ φ_i)
Θ は 2 通り: plain Θ0 = ΦΦ^T + 11^T、Adam 版 ΘD = Φ diag(D) Φ^T + d_c 11^T（D = 保存 v̂ の 1/√v̂ を k で平均）。
τ の自然な値: 1 課題 T=4000 更新・lr η で τ_nat = ηT/N（二乗誤差の等価）。CE の初期（p≈u）は 1/K 倍。
使い方: python3 filter_real.py <tag> <glob> [<glob> ...]
"""
import os, sys, glob, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from common_f import load_state, activ, gate

def p_pos_table(K):
    th = np.linspace(0, math.pi, 100001)
    f = np.sin(th) ** (K - 2) if K > 2 else np.ones_like(th)
    c = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(th))]); c /= c[-1]
    return lambda rho: np.interp(math.pi - np.arccos(np.clip(rho, -1, 1)), th, c)

def spearman(a, b):
    if len(a) < 3: return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

ETA = 1e-3
FACS = np.logspace(-4, 6, 41)            # τ λ1 の格子（λ1 = Θ の最大固有値）

def analyze(path, extra_taus=None):   # sink_roots_0930 round 2: extra_taus = {name: tau}
    st = load_state(path); X, P, act, K = st["X"].numpy(), [q.numpy() for q in st["P"]], st["act"], st["K"]
    L = st["L"]; N = X.shape[0]
    y = st["y_old"].numpy()
    inp = X
    for l in range(L - 1):                                   # 最終隠れ層への入力
        inp = activ(torch.tensor(inp @ P[2 * l].T + P[2 * l + 1]), act).numpy()
    W1, b1, W2, b2 = P[2 * (L - 1)], P[2 * (L - 1) + 1], P[2 * L], P[2 * L + 1]; H = W1.shape[0]
    zt = torch.tensor(inp @ W1.T + b1)
    z = zt.numpy(); a = activ(zt, act).numpy(); g = gate(zt, act).numpy()
    lg = a @ W2.T + b2; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    mu = inp.mean(0); Kn = inp @ mu + 1.0
    V = W2.T; Vc = V - V.mean(1, keepdims=True)
    Bm = Kn[:, None] * g
    S = (Bm * (p @ Vc.T)).sum(0); Soh = (Bm * Vc[:, y].T).sum(0)
    op = z > 0; k = op.sum(0); alive = k > 0
    S_open = (Bm * (p @ Vc.T) * op).sum(0); S_closed = S - S_open
    acc_old = float((lg.argmax(1) == y).mean())
    pp = p_pos_table(K)
    vhat2 = st["vA"][2 * L].numpy() / (1 - 0.999 ** st["step"]); vhat3 = st["vA"][2 * L + 1].numpy() / (1 - 0.999 ** st["step"])
    D = (1.0 / (np.sqrt(vhat2) + 1e-8)).mean(0)            # [H]
    dc = float((1.0 / (np.sqrt(vhat3) + 1e-8)).mean())
    res = dict(name=Path(path).stem, act=act, K=K, acc_old=acc_old, alive=int(alive.sum()),
               S_pos=float((S[alive] > 0).mean()), Soh_pos=float((Soh[alive] > 0).mean()),
               clop=float(np.median(np.abs(S_closed[alive]) / (np.abs(S_open[alive]) + 1e-30))))
    # 網全体の NTK（二乗誤差・クラスについて等方な部分だけ）: 読み出し + 最終隠れ層 (w,b) + 上流の層
    Vc_ = V - V.mean(1, keepdims=True)                        # [H,K]
    th_full = a @ a.T + 1.0
    wv = (Vc_ ** 2).sum(1) / (K - 1)                          # ‖v_i^c‖²/(K-1)
    th_full += ((g * wv) @ g.T) * (inp @ inp.T + 1.0)
    if L == 2:
        g1 = gate(torch.tensor(X @ P[0].T + P[1]), act).numpy()
        hc = np.einsum("ij,ni,ik->njk", P[2], g, Vc_)         # h^c_njk = Σ_i W2_ij g2_ni v^c_ik
        Gm = (g1[:, :, None] * hc).reshape(N, -1) / np.sqrt(K - 1)
        th_full += (Gm @ Gm.T) * (X @ X.T + 1.0)
    for tag, Th in (("plain", a @ a.T + 1.0), ("adamD", (a * D) @ a.T + dc), ("full", th_full)):
        lam, U = np.linalg.eigh(Th); lam = np.clip(lam, 0, None)
        l1 = lam[-1]; l2 = lam[-2]
        Ua = U.T @ a; Ub = U.T @ Bm                           # [N,H]
        tau_nat = ETA * float(np.load(path)["ks"][-1]) / N   # sink_roots_0930 round 2: the state's own task length (was 4000 / 6000)
        rows = []
        taus = list(FACS / l1) + [tau_nat, tau_nat / K, np.inf]
        extra = dict(extra_taus or {})
        for tau in taus + list(extra.values()):
            if np.isinf(tau):
                f = np.where(lam > 1e-9 * l1, 1 / np.where(lam > 1e-9 * l1, lam, 1), 0.0)
            else:
                f = np.where(lam > 1e-12 * l1, -np.expm1(-tau * lam) / np.where(lam > 1e-12 * l1, lam, 1), tau)
            Gf = f[:, None] * Ua                              # U^T F φ
            E = (1 - 1 / K) * (Ub * Gf).sum(0)
            rho = (Ub * Gf).sum(0) / (np.linalg.norm(Ub, axis=0) * np.linalg.norm(Gf, axis=0) + 1e-300)
            Pp = pp(rho)
            G = U @ Gf                                          # F φ in sample space
            Eo = (1 - 1 / K) * (Bm * G * op).sum(0); Ec = E - Eo
            al = alive
            rows.append(dict(tau=tau, taul1=tau * l1, taul2=tau * l2, meanP=float(Pp[al].mean()), Epos=float((E[al] > 0).mean()),
                             rank=spearman(E[al], S[al]), rank_oh=spearman(E[al], Soh[al]), agree=float((np.sign(E[al]) == np.sign(S[al])).mean()),
                             clop=float(np.median(np.abs(Ec[al]) / (np.abs(Eo[al]) + 1e-30))), Ecl_neg=float((Ec[al] < 0).mean())))
        res[tag] = dict(l1N=l1 / N, l2N=l2 / N, rows=rows[:len(taus)], extra=dict(zip(extra.keys(), rows[len(taus):])))
    return res

if __name__ == "__main__":
    t0 = time.monotonic(); out = []
    tag_out = sys.argv[1]
    for pat in sys.argv[2:]:
        for f in sorted(glob.glob(pat)):
            r = analyze(f); out.append(r)
            line = f"{r['name']:22s} acc {r['acc_old']:.2f} alive {r['alive']:3d} | obs S>0 {r['S_pos']:.2f} Soh>0 {r['Soh_pos']:.2f} cl/op {r['clop']:.2f}"
            for tag in ("plain", "full"):
                R = r[tag]; rows = R["rows"]
                raw, nat, natK, inf = rows[0], rows[-3], rows[-2], rows[-1]
                best = max(rows[:-3], key=lambda q: (q["rank"] if not np.isnan(q["rank"]) else -9))
                line += (f" || {tag} λ1/N {R['l1N']:.1f} λ2/N {R['l2N']:.2f} | raw P {raw['meanP']:.2f} rk {raw['rank']:+.2f}"
                         f" | nat(τλ1 {nat['taul1']:.1e}) P {nat['meanP']:.2f} rk {nat['rank']:+.2f} cl/op {nat['clop']:.2f}"
                         f" | nat/K P {natK['meanP']:.2f} rk {natK['rank']:+.2f} | FWL P {inf['meanP']:.2f} rk {inf['rank']:+.2f}"
                         f" | best τλ1 {best['taul1']:.0e} P {best['meanP']:.2f} rk {best['rank']:+.2f}")
            print(line, flush=True)
    np.save(f"out/filter_{tag_out}.npy", np.array(out, dtype=object), allow_pickle=True)
    print(f"({time.monotonic() - t0:.0f}s)")
