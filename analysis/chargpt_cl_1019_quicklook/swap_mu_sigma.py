#!/usr/bin/env python3
"""μ/σ の変化を「自分の重み」「上流の平均 ȳ」「上流の分散 Σ_y」に割る入れ替え計算（下見・未登録、1020 夕方の Issa の問い）。
μ_i/σ_i = (w_i·ȳ)/√(w_iᵀΣ_y w_i) = M·cos_i、M = √(ȳᵀΣ_y⁻¹ȳ)（上流だけで決まる上限）。
状態 A（前）・B（後）の組で、w・ȳ・Σ のどれを B に替えると μ/σ の中央値がどれだけ動くかを、同じ probe（後の課題の chain probe）で測る。
  python analysis/chargpt_cl_1019_quicklook/swap_mu_sigma.py --arm SiLU --pairs task1:task10 task10:task25 task25:task50
"""
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C


def state(arm, d, name, probe):
    m = C.build_model(arm, "M", 1000, "cpu"); m.load_state_dict(torch.load(d / "ckpts" / f"{name}.pt", map_location="cpu")); m.eval()
    cap = []
    with torch.no_grad():
        m.body(probe, cap)
    out = []
    for li, blk in enumerate(m.h):
        y = blk.n2(cap[2 * li]).reshape(-1, C.D).double()
        ybar = y.mean(0); Sig = torch.cov(y.T)
        out.append(dict(W=blk.mlp.up.weight.double(), ybar=ybar, Sig=Sig))
    return out


def rmed(W, ybar, Sig):
    return float(((W @ ybar) / torch.sqrt(((W @ Sig) * W).sum(1))).median())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="SiLU"); ap.add_argument("--pairs", nargs="+", default=["task1:task10", "task10:task25", "task25:task50"])
    a = ap.parse_args(); torch.set_num_threads(8)
    d = ROOT / "results/chargpt_cl_1019/quicklook/M" / a.arm / "seed1000"
    for pr in a.pairs:
        an, bn = pr.split(":"); tb = int(bn.replace("task", ""))
        probe = C.task_cpu_M(1000, tb, 10, 0.01)["probe"]
        A, B = state(a.arm, d, an, probe), state(a.arm, d, bn, probe)
        print(f"== {a.arm} {an} → {bn}（probe は課題 {tb}）")
        for li in range(C.L):
            x, y = A[li], B[li]
            M = lambda s: float(torch.sqrt(s["ybar"] @ torch.linalg.solve(s["Sig"] + 1e-10 * torch.eye(C.D, dtype=torch.float64), s["ybar"])))
            share = lambda s: float(s["ybar"].pow(2).sum() / s["Sig"].trace())
            r_AA = rmed(x["W"], x["ybar"], x["Sig"]); r_BB = rmed(y["W"], y["ybar"], y["Sig"])
            r_wB = rmed(y["W"], x["ybar"], x["Sig"])          # 重みだけ B
            r_yB = rmed(x["W"], y["ybar"], x["Sig"])          # 上流の平均だけ B
            r_SB = rmed(x["W"], x["ybar"], y["Sig"])          # 上流の分散だけ B
            r_upB = rmed(x["W"], y["ybar"], y["Sig"])         # 上流（平均と分散）を B、重みは A
            print(f"  L{li}: μ/σ 中央 {r_AA:+.2f} → {r_BB:+.2f} | 重みだけ替える {r_wB:+.2f} | 上流の平均だけ {r_yB:+.2f} | 上流の分散だけ {r_SB:+.2f} | 上流を両方 {r_upB:+.2f}"
                  f" | M {M(x):.2f} → {M(y):.2f} | ‖ȳ‖²/trΣ {share(x):.3f} → {share(y):.3f} | trΣ {float(x['Sig'].trace()):.1f} → {float(y['Sig'].trace()):.1f}")


if __name__ == "__main__":
    main()
