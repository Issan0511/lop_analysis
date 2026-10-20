"""(1) σ_med(t) の log-log 傾き（課題末・chain probe）。
(2) checkpoint で、損失の尺度微分 ∂L/∂log s（γ_l・W_up,l・W_down,l をまとめて s 倍）を eval 系列の 16 塊で測り、符号の一貫性を見る。
    負 = FFN を大きくすると損失が下がる = 大きくする向きの系統的な押し。
(3) ∂L/∂γ と ∂L/∂W_up を、入力の共通成分（トークン平均 m̂・ȳ）の分と変動の分に割る。
    ∂L/∂γ_j = Σ_n x̂_j(n) u_j(n), u = W_upᵀ ∂L/∂z。共通の分 = N m̂_j ū_j。 ∂L/∂W_up = Σ_n g_z(n) y(n)ᵀ、共通の分 = N ḡ_z ȳᵀ（階数 1）。
    報告: 共通の分の射影 ⟨G_c, G⟩/‖G‖²（1 なら勾配は共通成分が運ぶ）と、尺度微分のうち共通の分。"""
import sys, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
R = Path("results/chargpt_cl_1019")
RUNS = {"KKT1_s1000": (R/"quicklook/M/KKT1/seed1000", "KKT1", 1000), "C3_s0": (R/"crosscheck_white-san/M/C3/seed0", "C3", 0)}
print("== (1) σ_med の伸びの指数（log σ vs log t の傾き）  層 0〜3")
for name, (d, arm, seed) in RUNS.items():
    U = np.load(d/"unit_stats.npz"); sd = np.median(U["chain_sd"], axis=2)   # [task, layer]
    t = np.arange(1, sd.shape[0]+1)
    for lo, hi in ((10, 50), (25, 50), (38, 50)):
        sl = slice(lo-1, hi)
        slopes = [np.polyfit(np.log(t[sl]), np.log(sd[sl, l]), 1)[0] for l in range(C.L)]
        print(f"  {name:11s} 課題 {lo:2d}〜{hi}: " + " ".join(f"{s:+.2f}" for s in slopes))
print("\n== (2)(3) checkpoint の勾配（eval 系列 16 塊）")
for name, (d, arm, seed) in RUNS.items():
    print(f"---- {name}")
    for ck, t in (("task1", 1), ("task10", 10), ("task25", 25), ("task50", 50)):
        m = C.build_model(arm, "M", seed, "cpu")
        m.load_state_dict(torch.load(d/"ckpts"/f"{ck}.pt", map_location="cpu")); m.eval()
        ev = C.task_cpu_M(seed, t, 10, 0.01)["eval_seq"]
        chunks = ev.chunk(16)
        rec = {l: dict(gs_g=[], gs_up=[], gs_dn=[], proj_g=[], proj_up=[], gs_g_c=[], gs_up_c=[]) for l in range(C.L)}
        for ch in chunks:
            m.zero_grad(set_to_none=True)
            cap = []
            x = ch[:, :C.T]
            lg = m(x, cap)
            zs = [cap[2*l+1] for l in range(C.L)]
            for z in zs: z.retain_grad()
            loss = F.cross_entropy(lg[:, 1:].reshape(-1, C.V), ch[:, 2:C.T+1].reshape(-1))
            loss.backward()
            for l, blk in enumerate(m.h):
                g = blk.n2.weight; Wu = blk.mlp.up.weight; Wd = blk.mlp.down.weight
                gs_g = float((g * g.grad).sum()); gs_up = float((Wu * Wu.grad).sum()); gs_dn = float((Wd * Wd.grad).sum())
                xin = cap[2*l]; xhat = F.layer_norm(xin, (C.D,), eps=blk.n2.eps).reshape(-1, C.D).double()
                gz = zs[l].grad.reshape(-1, C.FF).double(); N = gz.shape[0]
                u = gz @ Wu.double()                       # [N, d] = ∂L/∂(γ⊙x̂)
                mhat = xhat.mean(0); ubar = u.mean(0)
                G_g = (xhat * u).sum(0); Gc_g = N * mhat * ubar
                y = xhat * g.double(); ybar = y.mean(0)
                G_up = gz.T @ y; Gc_up = N * torch.outer(gz.mean(0), ybar)
                rec[l]["gs_g"].append(gs_g); rec[l]["gs_up"].append(gs_up); rec[l]["gs_dn"].append(gs_dn)
                rec[l]["proj_g"].append(float((Gc_g @ G_g) / (G_g @ G_g)))
                rec[l]["proj_up"].append(float((Gc_up * G_up).sum() / (G_up * G_up).sum()))
                rec[l]["gs_g_c"].append(float((g.double() * Gc_g).sum())); rec[l]["gs_up_c"].append(float((Wu.double() * Gc_up).sum()))
        for l in range(C.L):
            r = {k: np.array(v) for k, v in rec[l].items()}
            def ms(a): return f"{a.mean():+.2e}±{a.std(ddof=1)/4:.1e} 負{int((a<0).sum()):2d}/16"
            print(f"  {ck:6s} L{l}: ∂L/∂log s  γ {ms(r['gs_g'])} | W_up {ms(r['gs_up'])} | W_down {ms(r['gs_dn'])}"
                  f" | 共通成分の射影 γ {r['proj_g'].mean():+.2f} W_up {r['proj_up'].mean():+.2f}"
                  f" | 尺度微分の共通の分 γ {r['gs_g_c'].mean():+.2e} W_up {r['gs_up_c'].mean():+.2e}")
