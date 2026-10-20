"""幅 σ_i と μ_i の分解（CPU だけ）。z_i = w_i·y, y = γ⊙x̂（LN 後の FFN 入力）。
σ_i² = w_iᵀ Σ_y w_i。等方なら σ_i ≈ ‖w_i‖√(tr Σ_y / d)。実測/等方 = 向きの揃い（alignment）。
μ_i = w_i·ȳ。|μ_i|/σ_i ≤ √(ȳᵀΣ_y⁻¹ȳ)（マハラノビス、Cauchy–Schwarz の恒等的上界）。
"""
import sys, math
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
R = Path("results/chargpt_cl_1019")
RUNS = {"KKT1b3_s1000": (R/"quicklook/M/KKT1b3/seed1000", "KKT1b3", 1000)}
CK = [("init", 1), ("task1", 1), ("task10", 10), ("task25", 25), ("task50", 50)]

def erank(S):
    e = torch.linalg.eigvalsh(S).clamp(min=0); p = e / e.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum())), float(e.max() / e.sum())

for name, (d, arm, seed) in RUNS.items():
    print(f"\n==== {name}")
    probes = {t: C.task_cpu_M(seed, t, 10, 0.01)["probe"] for t in (1, 10, 25, 50)}
    for ck, t in CK:
        m = C.build_model(arm, "M", seed, "cpu")
        m.load_state_dict(torch.load(d/"ckpts"/f"{ck}.pt", map_location="cpu")); m.eval()
        cap = []
        with torch.no_grad():
            m.body(probes[t], cap)
        emb = m.wte.weight.norm(dim=1).mean()
        print(f"-- {ck:6s} (probe of task {t})  ‖emb row‖ mean {float(emb):.2f}")
        for li, blk in enumerate(m.h):
            xin, z = cap[2*li], cap[2*li+1]
            X = xin.reshape(-1, C.D).double(); Z = z.reshape(-1, C.FF).double()
            xhat = torch.nn.functional.layer_norm(xin, (C.D,), eps=blk.n2.eps).reshape(-1, C.D).double()
            gam = blk.n2.weight.double(); y = xhat * gam
            W = blk.mlp.up.weight.double()
            Sig = torch.cov(y.T); ybar = y.mean(0)
            tr_d = float(Sig.trace() / C.D)
            er, top = erank(Sig)
            wn = W.norm(dim=1)
            sd = Z.std(0); mu = Z.mean(0)
            iso = wn * math.sqrt(tr_d)
            align = (sd / iso)
            r = mu / sd
            Sr = Sig + 1e-10 * torch.eye(C.D, dtype=torch.float64)
            mahal = float(torch.sqrt(ybar @ torch.linalg.solve(Sr, ybar)))
            Kc = y @ ybar; cwall = float(Kc.mean() / Kc.std())
            # 残差の流れ: ブロック入力の rms と、共通成分（トークン平均）の割合
            xres = float(X.pow(2).mean().sqrt()); xbar = X.mean(0); share = float(xbar.pow(2).sum() / X.pow(2).mean(0).sum())
            # γ: rms と 上位 8 座標の寄与
            g2 = gam.pow(2); gtop = float(g2.sort(descending=True).values[:8].sum() / g2.sum())
            print(f"  L{li}: σ_med {float(sd.median()):7.2f} = ‖w‖_med {float(wn.median()):5.2f} × √(trΣ/d) {math.sqrt(tr_d):6.2f} × 揃い {float(align.median()):4.2f}"
                  f" | γ rms {float(gam.pow(2).mean().sqrt()):5.2f} (上位8座標 {gtop:.2f}) | y: erank {er:5.1f} top {top:.2f} ‖ȳ‖/√trΣ {float(ybar.norm()/Sig.trace().sqrt()):.2f}"
                  f" | μ/σ med {float(r.median()):+.2f} q10 {float(torch.quantile(r,0.1)):+.2f} | 壁 c {cwall:5.2f} mahal {mahal:5.2f}"
                  f" | 残差 rms {xres:7.2f} 共通成分 {share:.2f} | ‖W_down‖ {float(blk.mlp.down.weight.norm()):6.1f}")
