"""KKT1b3: 共通の書き込みはどこで消えたか。FFN 出力 f = W_down h + b_down の共通成分を W_down h̄ と b_down に分け、相殺を見る。
また残差の共通成分 x̄ の源（emb・attn・ffn）。KKT1（β なし）と並べる。"""
import sys, math
from pathlib import Path
import torch
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
R = Path("results/chargpt_cl_1019")
for name, d, arm in (("KKT1b3", R/"quicklook/M/KKT1b3/seed1000", "KKT1b3"), ("KKT1", R/"quicklook/M/KKT1/seed1000", "KKT1")):
    print(f"\n==== {name}")
    for ck, t in (("task10", 10), ("task25", 25), ("task50", 50)):
        m = C.build_model(arm, "M", 1000, "cpu"); m.load_state_dict(torch.load(d/"ckpts"/f"{ck}.pt", map_location="cpu")); m.eval()
        probe = C.task_cpu_M(1000, t, 10, 0.01)["probe"]
        with torch.no_grad():
            h = m.wte(probe); srcs = [("emb", h.reshape(-1, C.D).double())]
            for li, b in enumerate(m.h):
                a = b.sc * b.attn(b.n1(h)); h = h + a; srcs.append((f"attn{li}", a.reshape(-1, C.D).double()))
                X = h.reshape(-1, C.D).double(); xbar = X.mean(0); xb2 = float(xbar.pow(2).sum())
                parts = sorted(((k, float((v.mean(0) @ xbar) / xb2)) for k, v in srcs), key=lambda kv: -abs(kv[1]))[:3]
                y = b.n2(h); z = b.mlp.up(y); hh = b.mlp.phi(z)
                Wd = b.mlp.down.weight.double(); hbar = hh.reshape(-1, C.FF).double().mean(0)
                wh = Wd @ hbar; bd = b.mlp.down.bias.double() if b.mlp.down.bias is not None else torch.zeros(C.D, dtype=torch.float64)
                fbar = wh + bd
                canc = -float((wh @ bd) / (wh @ wh))
                f = b.mlp.down(hh); h = h + f; srcs.append((f"ffn{li}", f.reshape(-1, C.D).double()))
                sd = z.reshape(-1, C.FF).double().std(0); off = (hbar - z.reshape(-1, C.FF).double().mean(0)) / sd
                print(f"  {ck:6s} L{li}: ‖W_down h̄‖ {float(wh.norm()):8.1f}  ‖b_down‖ {float(bd.norm()):8.1f}  相殺 {canc:+.2f}  → ‖f̄‖ {float(fbar.norm()):8.1f}"
                      f" | オフセット(h̄−μ)/σ 中央 {float(off.median()):.2f} | 残差 ‖x̄‖ {math.sqrt(xb2):8.1f} 源 " + " ".join(f"{k} {v:+.2f}" for k, v in parts))
