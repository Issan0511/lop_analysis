"""残差の流れの共通成分（トークン平均 x̄）を、書き込んだ源ごとに分ける: 埋め込み・各ブロックの注意・各ブロックの FFN。
また Snake の正のオフセット: FFN 隠れ層 h の unit 平均が σ_i/(2c) に比例するか（ReLU は E[z⁺]）。"""
import sys, math
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
R = Path("results/chargpt_cl_1019")
RUNS = {"KKT1_s1000": (R/"quicklook/M/KKT1/seed1000", "KKT1", 1000), "C3_s0": (R/"crosscheck_white-san/M/C3/seed0", "C3", 0)}
for name, (d, arm, seed) in RUNS.items():
    print(f"\n==== {name}")
    for ck, t in (("task10", 10), ("task25", 25), ("task50", 50)):
        m = C.build_model(arm, "M", seed, "cpu")
        m.load_state_dict(torch.load(d/"ckpts"/f"{ck}.pt", map_location="cpu")); m.eval()
        probe = C.task_cpu_M(seed, t, 10, 0.01)["probe"]
        with torch.no_grad():
            h = m.wte(probe)
            srcs = [("emb", h.reshape(-1, C.D).double())]
            for li, b in enumerate(m.h):
                a = b.sc * b.attn(b.n1(h)); h = h + a
                srcs.append((f"attn{li}", a.reshape(-1, C.D).double()))
                # ブロック l の入力 x_l の共通成分を源に分ける（FFN_l の前）
                X = h.reshape(-1, C.D).double(); xbar = X.mean(0); xb2 = float(xbar.pow(2).sum())
                parts = {k: float((v.mean(0) @ xbar) / xb2) for k, v in srcs}   # x̄ への射影の割合（和は 1）
                z = b.mlp.up(b.n2(h)); hh = b.mlp.phi(z); f = b.mlp.down(hh); h = h + f
                srcs.append((f"ffn{li}", f.reshape(-1, C.D).double()))
                Z = z.reshape(-1, C.FF).double(); H = hh.reshape(-1, C.FF).double()
                sd = Z.std(0); mu = Z.mean(0); hbar = H.mean(0)
                off = hbar - mu                              # 活性化が足す平均のオフセット（線形なら 0）
                ratio = (off / sd).median()
                top = sorted(parts.items(), key=lambda kv: -abs(kv[1]))[:4]
                print(f"  {ck:6s} L{li}: x̄ の源 " + " ".join(f"{k} {v:+.2f}" for k, v in top) +
                      f" | ‖x̄‖ {math.sqrt(xb2):8.2f} | h̄−μ の中央 {float(off.median()):8.2f} = {float(ratio):.2f}σ | ‖f̄‖(FFN出力の共通成分) {float(f.reshape(-1,C.D).double().mean(0).norm()):8.2f} ‖ā‖ {float(a.reshape(-1,C.D).double().mean(0).norm()):7.2f}")
