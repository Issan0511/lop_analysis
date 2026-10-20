"""KKT1 s1000: 課題 25→50 の Δμ_i を 自分の重み（Δw·ȳ25）/ 上流（w25·Δȳ）/ 交差（Δw·Δȳ）に割る。ȳ = γ⊙m̂（LN 後の入力のトークン平均）。
同じ probe（課題 50）で両 checkpoint を評価して、データの違いを除く。σ は課題 50 の値で割る。"""
import sys, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
d = Path("results/chargpt_cl_1019/quicklook/M/KKT1/seed1000")
probe = C.task_cpu_M(1000, 50, 10, 0.01)["probe"]
def state(ck):
    m = C.build_model("KKT1", "M", 1000, "cpu"); m.load_state_dict(torch.load(d/"ckpts"/f"{ck}.pt", map_location="cpu")); m.eval()
    cap = []
    with torch.no_grad(): m.body(probe, cap)
    out = []
    for l, blk in enumerate(m.h):
        xin, z = cap[2*l], cap[2*l+1]
        xhat = F.layer_norm(xin, (C.D,), eps=blk.n2.eps).reshape(-1, C.D).double()
        ybar = (xhat * blk.n2.weight.double()).mean(0)
        out.append(dict(W=blk.mlp.up.weight.double().clone(), ybar=ybar, mu=z.reshape(-1, C.FF).double().mean(0), sd=z.reshape(-1, C.FF).double().std(0),
                        ynorm=float(ybar.norm()), gam=float(blk.n2.weight.double().pow(2).mean().sqrt()), mhat=float(xhat.mean(0).norm())))
    return out
A, B = state("task25"), state("task50")
print("層 | μ/σ 中央 25→50（課題50のprobe） | Δμ/σ50 の中央: 自分の重み Δw·ȳ25 / 上流 w25·Δȳ / 交差 | ‖ȳ‖ 25→50 | ‖m̂‖ 25→50 (√d=11.3) | cos(ȳ25,ȳ50)")
for l in range(C.L):
    a, b = A[l], B[l]
    own = (b["W"] - a["W"]) @ a["ybar"]; up = a["W"] @ (b["ybar"] - a["ybar"]); cross = (b["W"] - a["W"]) @ (b["ybar"] - a["ybar"])
    sd = b["sd"]
    tot = (b["mu"] - a["mu"]) / sd
    cos = float(a["ybar"] @ b["ybar"] / (a["ybar"].norm() * b["ybar"].norm()))
    print(f"L{l} | {float((a['mu']/a['sd']).median()):+.2f} → {float((b['mu']/b['sd']).median()):+.2f} | 合計 {float(tot.median()):+.2f}: "
          f"{float((own/sd).median()):+.2f} / {float((up/sd).median()):+.2f} / {float((cross/sd).median()):+.2f} | {a['ynorm']:.1f} → {b['ynorm']:.1f} | {a['mhat']:.2f} → {b['mhat']:.2f} | {cos:+.2f}")
print("\n課題ごと（課題末・chain probe）: 超過損失 | μ/σ 中央 L0..3 | σ 中央 L3 | α 下限張り付き L3 | 曲がりの区間 L3 | H（課題のエントロピー）")
pt = {int(r["task"]): r for r in csv.DictReader(open(d/"per_task.csv"))}
E_ = {}
for r in csv.DictReader(open(d/"layer_stats.csv")):
    if r["probe"] == "chain" and int(r["u"]) == 10000: E_[(int(r["task"]), int(r["layer"]))] = r
for t in list(range(26, 51)):
    print(f"t{t:2d} {float(pt[t]['online_all']):.3f} | " + " ".join(f"{float(E_[(t,l)]['r_med']):+.2f}" for l in range(4)) +
          f" | {float(E_[(t,3)]['two_alpha_sd_med'])/ (2*float(E_[(t,3)]['alpha_med'])):6.1f} | {float(E_[(t,3)]['alpha_clip_frac']):.2f} | {float(E_[(t,3)]['frac_in_curve']):.3f} | H {float(pt[t]['H']):.3f}")
