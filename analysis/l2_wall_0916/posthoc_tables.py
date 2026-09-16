#!/usr/bin/env python3
"""POSTHOC tables (registered=0) for the vault note on the Part A re-runs of l2_wall_0916.
Reads results/l2_wall_0916/logs/diag_{chimera,ext150}.npz and writes results/l2_wall_0916/posthoc_tables.md."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/l2_wall_0916"
out = []


def load(tag):
    d = np.load(SRC / "logs" / f"diag_{tag}.npz")
    models = json.loads((SRC / f"provenance_{tag}.json").read_text())["models"]
    pts = list(zip(d["task"].tolist(), d["step"].tolist()))
    return d, models, pts


def med(a):
    with np.errstate(all="ignore"):
        return float(np.nanmedian(a))


def rng(vals, f="{:.2f}"):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return (f + "〜" + f).format(min(vals), max(vals)) if len(vals) > 1 else f.format(vals[0])


def const_row(d, i, j):
    return dict(d2=med(d["u_d_l2"][i, j]), c2=float(d["m_in2_c"][i, j]), k2=float(d["m_in2_k"][i, j]),
                rho=med(d["u_rho_l2"][i, j]), bs=med(np.abs(d["u_b_l2"][i, j]) / d["u_sig_l2"][i, j]),
                ab=float(np.mean(d["u_hU_l2"][i, j] < 0)), pmin=float(d["m_in2_pmin"][i, j]),
                cos1=float(d["m_in2_cos1"][i, j]), eshare=float(d["m_in2_eshare"][i, j]),
                gate=float(np.nanmean(d["u_gate_mean_l2"][i, j])), r2=float(d["m_in2_r"][i, j]))


out.append("## A. 第2層の壁定数（タスク終端・中央ユニット・3 seed の範囲）\n")
out.append("| 走 | 組 | t | d₂ | c₂ | κ₂ | ρ_µ | \\|b₂\\|/σ₂ | U₂<0 | min e₂ᵀu | cos(µ₂,1) | µ₂ 方向の分散割合 | 平均 φ′ |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag, sel, t in (("chimera", dict(env="RL", act1="ELU1", act2="ELU1"), 50), ("chimera", dict(env="RL", act1="LR", act2="ELU1"), 50),
                    ("chimera", dict(env="RL", act1="ELU1", act2="LR"), 50), ("chimera", dict(env="RL", act1="LR", act2="LR"), 50),
                    ("chimera", dict(env="PM", act1="ELU1", act2="ELU1"), 50), ("chimera", dict(env="PM", act1="LR", act2="LR"), 50),
                    ("ext150", dict(env="RL", act="ELU1"), 150), ("ext150", dict(env="RL", act="SILU"), 150),
                    ("ext150", dict(env="RL", act="GELU"), 150), ("ext150", dict(env="RL", act="LR"), 150),
                    ("ext150", dict(env="PM", act="GELU"), 150), ("ext150", dict(env="PM", act="SILU"), 150),
                    ("ext150", dict(env="PM", act="ELU1"), 150)):
    d, models, pts = load(tag)
    i = max(k for k, p in enumerate(pts) if p[0] == t)
    rows = [const_row(d, i, j) for j, m in enumerate(models) if all(m[k] == v for k, v in sel.items())]
    col = lambda k, f="{:.2f}": rng([r[k] for r in rows], f)
    name = "→".join([sel.get("act1", sel.get("act")), sel.get("act2", sel.get("act"))])
    out.append(f"| {tag} | {sel['env']} {name} | {t} | {col('d2')} | {col('c2')} | {col('k2')} | {col('rho')} | {col('bs')} | {col('ab')} | {col('pmin', '{:+.1f}')} | {col('cos1')} | {col('eshare')} | {col('gate', '{:.2g}')} |")

out.append("\n## B. 切替衝撃と回復（層別キメラ・RL・seed ごと）\n")
d, models, pts = load("chimera")
for sel in (dict(act1="ELU1", act2="ELU1"), dict(act1="LR", act2="ELU1"), dict(act1="ELU1", act2="LR"), dict(act1="LR", act2="LR")):
    out.append(f"\n**RL {sel['act1']}→{sel['act2']}**（各セル: seed 0 / 1 / 2）\n")
    out.append("| t | 切替 75 更新後の吸収割合 | 75 更新後の平均 φ′ | 75 更新後の ‖µ₂‖ | 終端の吸収割合 | 終端の平均 φ′ | 終端の ‖µ₂‖ | 終端の √ρ−κ/c |")
    out.append("|---|---|---|---|---|---|---|---|")
    js = [j for j, m in enumerate(models) if m["env"] == "RL" and m["act1"] == sel["act1"] and m["act2"] == sel["act2"]]
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50):
        cells = []
        i75 = pts.index((t, 75)) if (t, 75) in pts else None
        ie = max(k for k, p in enumerate(pts) if p[0] == t)
        def f(i, fn, fmt):
            return " / ".join(fmt.format(fn(i, j)) for j in js) if i is not None else "—"
        ab = lambda i, j: float(np.mean(d["u_hU_l2"][i, j] < 0))
        gt = lambda i, j: float(np.nanmean(d["u_gate_mean_l2"][i, j]))
        mu = lambda i, j: float(d["m_in2_r"][i, j])
        cone = lambda i, j: med(np.sqrt(np.clip(d["u_rho_l2"][i, j], 0, None))) - float(d["m_in2_k"][i, j] / d["m_in2_c"][i, j])
        out.append(f"| {t} | {f(i75, ab, '{:.2f}')} | {f(i75, gt, '{:.3f}')} | {f(i75, mu, '{:.1f}')} | {f(ie, ab, '{:.2f}')} | {f(ie, gt, '{:.3f}')} | {f(ie, mu, '{:.1f}')} | {f(ie, cone, '{:+.2f}')} |")
(SRC / "posthoc_tables.md").write_text("# l2_wall_0916 posthoc tables (registered=0)\n\n" + "\n".join(out) + "\n")
print("\n".join(out))
