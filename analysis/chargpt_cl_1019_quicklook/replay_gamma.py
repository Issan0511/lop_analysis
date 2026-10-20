"""KKT1 seed 1000: 課題 25 の末（重み＋Adam の状態）から 1,000 更新を CPU で再生し、γ・W_up の動径方向の歩を 1 更新ごとに見る。
条件: switch = 課題 26 のデータ / cont = 課題 25 のデータ列の続き（更新 10000〜10999）/ switch_frz = 課題 26・V 凍結。
動径の歩: Σ_j γ_j Δγ_j / ‖γ‖²（相対）。外向き割合: sign(Δγ_j) = sign(γ_j) の座標の割合（雑音なら 0.5）。
雑音だけの ‖γ‖² の増え方の参照: lr²·d / ‖γ‖² per 更新。"""
import sys, math, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
d = Path("results/chargpt_cl_1019/quicklook/M/KKT1/seed1000")
SEED, NU = 1000, 1000
data26 = C.task_cpu_M(SEED, 26, NU, 0.01)["data"]
data25 = C.task_cpu_M(SEED, 25, 10000 + NU, 0.01)["data"][10000:]
def run(cond, data, freeze):
    m = C.build_model("KKT1", "M", SEED, "cpu")
    m.load_state_dict(torch.load(d/"ckpts"/"task25.pt", map_location="cpu")); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=True)
    opt.load_state_dict(torch.load(d/"ckpts"/"task25_adam.pt", map_location="cpu"))
    for g in opt.param_groups: g["lr"] = 1e-3; g["capturable"] = False; g["lr"] = float(g["lr"])
    if freeze:
        for b in m.h: b.mlp.v_rate.fill_(0.0)
    gam = [b.n2.weight for b in m.h]; wup = [b.mlp.up.weight for b in m.h]
    rec = dict(loss=[], g_rad=[[] for _ in range(C.L)], g_out=[[] for _ in range(C.L)], w_rad=[[] for _ in range(C.L)], w_out=[[] for _ in range(C.L)],
               g_rms=[[] for _ in range(C.L)], sig=[[] for _ in range(C.L)], w_rms=[[] for _ in range(C.L)])
    t0 = time.time()
    for u in range(NU):
        x = data[u][:, :C.T].long(); y = data[u][:, 1:C.T + 1].long()
        g0 = [g.detach().clone() for g in gam]; w0 = [w.detach().clone() for w in wup]
        cap = []
        out = m(x, cap)
        loss = F.cross_entropy(out[:, 1:].reshape(-1, C.V), y[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        rec["loss"].append(float(loss))
        with torch.no_grad():
            for l in range(C.L):
                dg = gam[l] - g0[l]; dw = wup[l] - w0[l]
                rec["g_rad"][l].append(float((g0[l] * dg).sum() / (g0[l] * g0[l]).sum()))
                rec["g_out"][l].append(float(((dg.sign() == g0[l].sign()) & (dg != 0)).float().mean()))
                rec["w_rad"][l].append(float((w0[l] * dw).sum() / (w0[l] * w0[l]).sum()))
                rec["w_out"][l].append(float(((dw.sign() == w0[l].sign()) & (dw != 0)).float().mean()))
                rec["g_rms"][l].append(float(gam[l].pow(2).mean().sqrt())); rec["w_rms"][l].append(float(wup[l].norm(dim=1).median()))
                z = cap[2 * l + 1].reshape(-1, C.FF); rec["sig"][l].append(float(z.std(0).median()))
        if u == 0: print(f"  [{cond}] 1 更新 {time.time()-t0:.2f}s", flush=True)
    print(f"== {cond}  損失: 更新 0 {rec['loss'][0]:.3f}, 1〜100 {np.mean(rec['loss'][1:101]):.3f}, 101〜300 {np.mean(rec['loss'][101:301]):.3f}, 301〜1000 {np.mean(rec['loss'][301:]):.3f}")
    for l in range(C.L):
        gr = np.array(rec["g_rad"][l]); go = np.array(rec["g_out"][l]); wr = np.array(rec["w_rad"][l]); wo = np.array(rec["w_out"][l])
        grms = rec["g_rms"][l]; sig = rec["sig"][l]; wrms = rec["w_rms"][l]
        noise = 1e-6 * C.D / (grms[0] ** 2 * C.D)        # lr²d/‖γ‖² = lr²/γ_rms²
        def seg(a, s0, s1): return f"{a[s0:s1].mean():+.1e}"
        def segf(a, s0, s1): return f"{a[s0:s1].mean():.2f}"
        print(f"  L{l}: γ_rms {grms[0]:.2f}→{grms[-1]:.2f}  ‖w‖_med {wrms[0]:.2f}→{wrms[-1]:.2f}  σ_med {sig[0]:.1f}→{sig[-1]:.1f}"
              f" | γ 動径/更新 [0-100 {seg(gr,0,100)} 100-300 {seg(gr,100,300)} 300-1000 {seg(gr,300,1000)}] 雑音参照 {noise:+.1e}"
              f" | γ 外向き割合 [{segf(go,0,100)} {segf(go,100,300)} {segf(go,300,1000)}]"
              f" | W_up 動径/更新 [{seg(wr,0,100)} {seg(wr,100,300)} {seg(wr,300,1000)}] 外向き [{segf(wo,0,100)} {segf(wo,100,300)} {segf(wo,300,1000)}]")
    return rec
R = {}
R["switch"] = run("switch", data26, False)
R["cont"] = run("cont", data25, False)
R["switch_frz"] = run("switch_frz", data26, True)
np.save("/tmp/claude-1000/-home-issan-Projects-claude/67e9d043-d36b-47f1-8820-bf53e97ee76c/scratchpad/replay_gamma.npy", R, allow_pickle=True)
