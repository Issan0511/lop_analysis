"""Post-hoc four-atom sufficiency example for a changing standardized tail.

This is a constructed deterministic logistic model, not an RL-MNIST replication.
Run --prepare first to save the bounded design, then --run after prerecording.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np


DESIGN = {
    "status": "post-hoc constructed sufficiency example, chosen after Gaussian synthetic outputs",
    "features": "mass p/2 each at +(A,0), -(A,0); mass (1-p)/2 each at +(0,1), -(0,1)",
    "labels": "sign of the nonzero coordinate, in {-1,+1}",
    "parameters": "w=(0,0), fixed v>0, no hidden/output bias, no noise, full deterministic loss",
    "loss": "p softplus(-v A w1) + (1-p) softplus(-v w2)",
    "A": [1.0, 2.0, 5.0, 10.0], "p": [0.5, 1 / 16, 1 / 64], "v": [0.1, 1.0, 10.0],
    "optimizers": {"adam": [0.9, 0.999], "instantaneous": [0.0, 0.0]},
    "steps": 4000, "learning_rate": 0.001, "epsilon": 0.0,
    "checkpoints": [1, 10, 100, 500, 1000, 2000, 4000],
    "population_sd": "sqrt(p*(A*w1)^2 + (1-p)*w2^2)",
    "body": "symmetry midpoint median=0; no sample-median tie convention",
    "independent_torch_check": {"A": 10.0, "p": 1 / 64, "v": 1.0, "epsilon": 1e-8, "steps": 4000},
    "limits": ["No real-data mechanism identification", "No claim that finite-time level is an equilibrium", "Cannot model downward post-v-increase trajectories because each coordinate grows monotonically", "No task changes or existing Adam history", "All inputs share two coordinates but pairs on different axes do not interact in the gradient"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def log_bias_correction(beta, t):
    return 0.0 if beta == 0 else math.log(-math.expm1(t * math.log(beta)))


def matrix(configs, steps=4000, checkpoints=()):
    """Positive-gradient-magnitude log EMAs; update sign is always positive."""
    A = np.array([c["A"] for c in configs]); p = np.array([c["p"] for c in configs]); v = np.array([c["v"] for c in configs])
    gain = v[:, None] * np.stack([A, np.ones_like(A)], axis=1)
    weight = np.stack([p, 1 - p], axis=1)
    b1 = np.array([c["beta1"] for c in configs])[:, None]
    b2 = np.array([c["beta2"] for c in configs])[:, None]
    eps = np.array([c["epsilon"] for c in configs])[:, None]
    # Selective logs avoid warnings at the intentional beta=0 and epsilon=0 controls.
    lb1 = np.full_like(b1, -np.inf); lb2 = np.full_like(b2, -np.inf); leps = np.full_like(eps, -np.inf)
    np.log(b1, out=lb1, where=b1 > 0); np.log(b2, out=lb2, where=b2 > 0); np.log(eps, out=leps, where=eps > 0)
    w = np.zeros_like(gain); lm = np.full_like(gain, -np.inf); lq = lm.copy()
    records = []
    for t in range(1, steps + 1):
        margin = gain * w
        lg = np.log(weight) + np.log(gain) - np.logaddexp(0.0, margin)
        lm = np.logaddexp(lb1 + lm, np.log1p(-b1) + lg)
        lq = np.logaddexp(lb2 + lq, np.log1p(-b2) + 2 * lg)
        bc1 = np.array([log_bias_correction(float(x), t) for x in b1[:, 0]])[:, None]
        bc2 = np.array([log_bias_correction(float(x), t) for x in b2[:, 0]])[:, None]
        lmh, lqh = lm - bc1, lq - bc2
        ratio = np.exp(lmh - np.logaddexp(0.5 * lqh, leps))
        w += DESIGN["learning_rate"] * ratio
        if t in checkpoints:
            for i, c in enumerate(configs):
                h = np.array([c["A"] * w[i, 0], w[i, 1]])
                s = math.sqrt(c["p"] * h[0] ** 2 + (1 - c["p"]) * h[1] ** 2)
                records.append({**c, "step": t, "w1": float(w[i, 0]), "w2": float(w[i, 1]),
                                "angle_degrees": float(np.degrees(np.arctan2(w[i, 1], w[i, 0]))),
                                "rare_height": float(h[0]), "common_height": float(h[1]), "raw_top": float(h.max()), "population_sd": s,
                                "top_over_s": float(h.max() / s), "centered_tail_over_s": float(h.max() / s),
                                "height_ratio_rare_common": float(h[0] / h[1]), "max_on_rare_axis": bool(h[0] >= h[1]),
                                "loss": float((weight[i] * np.logaddexp(0.0, -gain[i] * w[i])).sum()),
                                "margin_rare": float(gain[i, 0] * w[i, 0]), "margin_common": float(gain[i, 1] * w[i, 1]),
                                "last_step_over_lr_rare": float(ratio[i, 0]), "last_step_over_lr_common": float(ratio[i, 1]),
                                "last_preupdate_log_gradient_abs_rare": float(lg[i, 0]), "last_preupdate_log_gradient_abs_common": float(lg[i, 1]),
                                "last_log_mhat_abs_rare": float(lmh[i, 0]), "last_log_mhat_abs_common": float(lmh[i, 1]),
                                "last_log_rms_rare": float(lqh[i, 0] / 2), "last_log_rms_common": float(lqh[i, 1] / 2),
                                "last_gradient_over_rms_rare": float(np.exp(lg[i, 0] - lqh[i, 0] / 2)),
                                "last_gradient_over_rms_common": float(np.exp(lg[i, 1] - lqh[i, 1] / 2))})
    return w, records


def torch_check():
    import torch
    torch.set_num_threads(1)
    c = DESIGN["independent_torch_check"]
    gain = torch.tensor([c["v"] * c["A"], c["v"]], dtype=torch.float64)
    weight = torch.tensor([c["p"], 1 - c["p"]], dtype=torch.float64)
    w = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    equivalent = [torch.zeros((), dtype=torch.float64, requires_grad=True) for _ in range(2)]
    opt = torch.optim.Adam([w], lr=DESIGN["learning_rate"], betas=(.9, .999), eps=c["epsilon"])
    opts = [torch.optim.Adam([q], lr=DESIGN["learning_rate"], betas=(.9, .999), eps=c["epsilon"] / float(weight[j])) for j, q in enumerate(equivalent)]
    for _ in range(c["steps"]):
        opt.zero_grad(set_to_none=True)
        loss = (weight * torch.nn.functional.softplus(-gain * w)).sum()
        loss.backward(); opt.step()
        for j, q in enumerate(equivalent):
            opts[j].zero_grad(set_to_none=True)
            torch.nn.functional.softplus(-gain[j] * q).backward(); opts[j].step()
    config = {"A": c["A"], "p": c["p"], "v": c["v"], "optimizer": "adam", "beta1": .9, "beta2": .999, "epsilon": c["epsilon"]}
    independent, _ = matrix([config], steps=c["steps"])
    got = w.detach().numpy(); mapped = np.array([float(q.detach()) for q in equivalent])
    out = {"config": c, "torch": torch.__version__, "log_ema_weights": independent[0].tolist(), "torch_weights": got.tolist(),
           "scalar_mapped_weights": mapped.tolist(), "log_ema_max_abs_error": float(abs(independent[0] - got).max()),
           "weighted_epsilon_mapping_max_abs_error": float(abs(mapped - got).max())}
    assert out["log_ema_max_abs_error"] < 1e-11
    assert out["weighted_epsilon_mapping_max_abs_error"] < 1e-11
    return out


def mdtable(rows, keys):
    s = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        s.append("| " + " | ".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys) + " |")
    return "\n".join(s)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--prepare", action="store_true"); p.add_argument("--run", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/logistic_v_height_0929")); args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    design_path = args.out / "anisotropic_design.json"
    if args.prepare:
        design_path.write_text(json.dumps(DESIGN, indent=2) + "\n")
        print("Saved post-hoc bounded design; no fits executed.")
    if not args.run: return
    assert design_path.exists() and json.loads(design_path.read_text()) == DESIGN
    configs = [{"A": A, "p": p0, "v": v, "optimizer": opt, "beta1": betas[0], "beta2": betas[1], "epsilon": DESIGN["epsilon"]}
               for A in DESIGN["A"] for p0 in DESIGN["p"] for v in DESIGN["v"] for opt, betas in DESIGN["optimizers"].items()]
    _, trace = matrix(configs, steps=DESIGN["steps"], checkpoints=DESIGN["checkpoints"])
    end = [r for r in trace if r["step"] == DESIGN["steps"]]
    write_csv(args.out / "anisotropic_trace.csv", trace); write_csv(args.out / "anisotropic_end.csv", end)
    # Primary controls test exact algebraic implications, not an expected direction of results.
    instant_errors = [abs(r["top_over_s"] - r["A"] / math.sqrt(r["p"] * r["A"] ** 2 + 1 - r["p"])) for r in end if r["optimizer"] == "instantaneous"]
    isotropic_errors = [abs(r["top_over_s"] - 1) for r in end if r["A"] == 1]
    p_errors = []
    for A in DESIGN["A"]:
        for v in DESIGN["v"]:
            for opt in DESIGN["optimizers"]:
                x = np.array([[r["w1"], r["w2"]] for r in end if r["A"] == A and r["v"] == v and r["optimizer"] == opt])
                p_errors.append(float(np.max(abs(x - x[0]))))
    tail_order = []
    for A in DESIGN["A"]:
        for mass in DESIGN["p"]:
            if A == 1: continue
            rows = sorted([r for r in end if r["A"] == A and r["p"] == mass and r["optimizer"] == "adam"], key=lambda r: r["v"])
            tail_order.append({"A": A, "p": mass, "tail_by_v": [r["top_over_s"] for r in rows],
                               "strictly_decreases_with_v": all(rows[i]["top_over_s"] > rows[i + 1]["top_over_s"] for i in range(len(rows) - 1))})
    checks = {"instantaneous_formula_max_abs_error": max(instant_errors), "A1_tail_max_abs_error": max(isotropic_errors),
              "epsilon0_weight_cancellation_max_abs_error": max(p_errors), "independent_pytorch": torch_check()}
    checks["descriptive_ordering_not_a_universal_theorem"] = tail_order
    assert max(instant_errors + isotropic_errors + p_errors) < 1e-10
    (args.out / "anisotropic_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (args.out / "anisotropic_provenance.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "source_sha256": sha(__file__), "design_sha256": sha(design_path), "fits": len(configs), "status": DESIGN["status"], "source_data": "none; exact four-atom weighted distribution"}, indent=2) + "\n")
    selected = [r for r in end if r["A"] == 10 and r["p"] == 1 / 64]
    report = r"""# 異方的な4点logistic模型（post-hoc構成例）

Gaussian合成実験の結果を見た後に追加した、構成的な十分性の例である。RL-MNISTの再現でも、その機構の同定でもない。デザインは実行前にanisotropic_design.jsonへ保存した。A=1,2,5,10、p=.5,1/16,1/64、v=.1,1,10、Adam(.9,.999)と瞬時正規化、全72系を4000歩、η=.001、ε=0で計算した。

## 厳密な式

入力は±A e₁（各質量p/2）と±e₂（各質量(1−p)/2）、ラベルは非零座標の符号。バイアスなし、v>0固定、w₀=0。

\[
L=p\log(1+e^{-vAw_1})+(1-p)\log(1+e^{-vw_2}),\qquad
g_j=-\lambda_j a_j/(1+e^{a_jw_j}),
\]

ここでλ=(p,1−p)、a=(vA,v)。Adamの各座標は独立で、λ>0はmにλ、二次momentにλ²として掛かる。したがってε=0ではλが完全に消える。ε>0では、重みなしスカラーlogisticのgain=a_j、epsilon=ε/λ_jに厳密に対応する。別実装のPyTorchでこの対応を確認した。

前活性は±h₁=±Aw₁と±h₂=±w₂、対称な本体=0、population s²=p h₁²+(1−p)h₂²。h₁≥h₂ならR=h₁/h₂と置いて、

\[
T=\frac{z_{\max}}s=\frac{z_{\max}-\mathrm{body}}s
=\frac{R}{\sqrt{pR^2+1-p}},\qquad
\frac{dT}{dR}=\frac{1-p}{(pR^2+1-p)^{3/2}}>0.
\]

つまり標準化した上端を変えるには、wの大きさだけでなく、2方向の育ち方の比Rが変わればよい。A=1では厳密にw₁=w₂でT=1。瞬時正規化では常にw₁=w₂=ηtなのでR=A、T=A/√(pA²+1−p)はvによらない。

Adamではgain=vAの座標とgain=vの座標で残差の減衰が異なり、各々のmoment履歴に入る。したがってw₁/w₂は変わり得る。これは確率的共分散を必要としない、決定論的な方向変化である。ただし、この構成例の2軸は損失が分離し、実ネットワークの入力間競合は含んでいない。

## 数値結果（A=10、p=1/64）

"""
    report += mdtable(selected, ["optimizer", "v", "raw_top", "top_over_s", "height_ratio_rare_common", "angle_degrees", "last_step_over_lr_rare", "last_step_over_lr_common", "loss"])
    report += f"\n\n登録したA>1の9組すべてで、4000歩後の規格化上端はv=.1→1→10で小さくなった（実測{sum(r['strictly_decreases_with_v'] for r in tail_order)}/9）。これは指定した3時点ではなく3つのvの比較であり、任意の時間・gainでの単調性定理ではない。\n"
    report += "\n\n全行と時間経過はanisotropic_end.csv / anisotropic_trace.csv。Adam momentsは正の勾配絶対値をlog空間で保存し、残差のunderflowによる誤停止を防いだ。最後の勾配・moment・一歩は4000歩目の更新前評価、高さと損失は更新後。\n\n"
    report += r"""## この例で言えることと限界

- vの固定値を変えると、残差履歴を通して異なる入力方向の相対的な伸びが変わり、標準化した上端が変わる経路を、非常に小さいlogistic模型で検査できる。何でもスカラーの一様拡大に還元されるわけではない。
- ε=0ではpはwの軌道に影響せず、分布の標準化だけに影響する。稀な高レバレッジ方向を置いたことが、上端を大きくできる幾何学的前提である。pのデータ質量が勾配に効かないのはこの分離・完全正規化模型に特有。
- raw上端と規格化上端の差は同時に検査する。損失はv間で一致しないので「同じlogitを満たすため1/vまで上がった」という説明は使わない。
- 全勾配の符号は一定で両座標が増え続けるため、vを途中で増やした際にraw上端が下がる現象はこの模型では出ない。新しいtask・既存moments・出力bias・ELUの閉側もない。実験のc倍凍結直後の上下両向きの調整を説明したとは言えない。
- 前活性分布は常に対称で、1歩目以降の正側割合は1/2。本体が負側に沈み、正側割合が約.02となるELUの分布は再現していない。
- 有限時間の結果で、有限の高さへの収束ではない。個々の点は線形分離可能で、有限の損失最小点は存在しない。
"""
    (args.out / "anisotropic_report.md").write_text(report)
    print(mdtable(selected, ["optimizer", "v", "raw_top", "top_over_s", "height_ratio_rare_common", "loss"]))


if __name__ == "__main__":
    main()
