#!/usr/bin/env python3
"""rlcifar_mlp_battle_0918 -- POST-HOC, NOT REGISTERED.

Three questions Issa asked after reading the result.  None of this is in spec §5;
the labels in verdict.json do not depend on any of it.  seed 0 unless stated.

  fold  : per unit, does it cross its own kink on this data?  min(P(z>0), P(z<0)),
          and the share of units that never cross.  Plus the same quantity at
          initialisation for seeds 0-9 against the closed form 2*Phi(-2.33/r),
          r = |xbar| / rms|x - xbar| (the input's mean-to-spread ratio).
  ratio : R = |w_i| * (per-dim rms spread of the layer's input) / sd(z_i), unit median.
          R = 1 means the row is isotropic w.r.t. its input (checked against a
          random-direction control); R < 1 means it is aligned with the top directions.
  wnorm : the unit row norm over the 50 tasks, as a multiple of the init value
          1/sqrt(3) (exact for U(+-1/sqrt(fan_in))), and the exponent of |w| ~ task^p.

Usage: posthoc_0918.py [fold|ratio|wnorm|all]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import pmnist_rlcifar_0907 as B                                  # noqa: E402

OUT = REPO / "results" / "rlcifar_mlp_battle_0918"
PI = math.pi
INIT_ROW = 1.0 / math.sqrt(3.0)          # U(+-1/sqrt(fan_in)) gives this in every layer
ARMS = ("SNA", "KKA", "KKA23", "KKT1", "LR", "LK03", "LK001", "SL", "RSL", "R", "ELU", "SILU", "GELU")
SNAKE = ("SNA", "KKA", "KKA23", "KKT1")
LEAKY = ("LR", "LK03", "LK001", "SL", "RSL")
COL = {"SNA": "#1f77b4", "KKA": "#ff7f0e", "KKA23": "#2ca02c", "KKT1": "#d62728", "R": "#7f7f7f",
       "LK001": "#9467bd", "LR": "#8c564b", "LK03": "#e377c2", "SL": "#17becf", "RSL": "#bcbd22",
       "ELU": "#aec7e8", "SILU": "#ffbb78", "GELU": "#98df8a"}
for _f in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAGothic", "DejaVu Sans"):
    if any(_f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
MEAN_C = torch.tensor([0.4914, 0.4822, 0.4465]).repeat_interleave(1024)
STD_C = torch.tensor([0.2470, 0.2435, 0.2616]).repeat_interleave(1024)
_CIF = None


def inputs(seed: int, cond: str) -> torch.Tensor:
    global _CIF
    if _CIF is None:
        _CIF = B.Cifar10()
    x = _CIF.images(B.subset_idx(seed), torch.device("cpu"))
    return x if cond == "raw" else (x - MEAN_C) / STD_C


def snap(arm: str, cond: str, seed: int, t: int):
    return np.load(OUT / arm / "snap" / f"{arm}_{cond}_seed{seed}" / f"t{t:02d}.npz")


def alphas(d):
    """The snake family's per-unit alpha = clip(c / sqrt(EMA var), lo, hi)."""
    if "V1" not in d.files:
        return None, None
    return tuple((0.6 / torch.from_numpy(d[k]).sqrt()).clamp(0.005, 3.0) for k in ("V1", "V2"))


def gate(arm: str, z: torch.Tensor, a=None) -> torch.Tensor:
    """phi'(z), the factor the gradient is multiplied by at this unit."""
    if arm in ("LR", "LK03", "LK001"):
        return torch.where(z > 0, 1.0, {"LR": 0.1, "LK03": 0.3, "LK001": 0.01}[arm])
    if arm in ("SL", "RSL"):
        al = 0.1 if arm == "SL" else 0.229            # RSL at eval uses the midpoint
        s = torch.sigmoid(5.0 * z / 3.0)
        return al + (1 - al) * (s + z * s * (1 - s) * 5.0 / 3.0)
    th = 2 * a * z
    g = 1.0 + torch.sin(th)
    if arm == "SNA":
        return g
    if arm in ("KKA", "KKA23"):
        g = torch.where((th < -1.5 * PI) | (th > 0.5 * PI), torch.full_like(g, 2.0), g)
        return g * (2 / 3) if arm == "KKA23" else g
    return torch.where((th < -2 * PI) | (th > PI), torch.ones_like(g), g)       # KKT1


def spread(U: torch.Tensor) -> float:
    return float((U - U.mean(0)).pow(2).mean().sqrt())


# --------------------------------------------------------------------------


def fold(arms=("LR", "LK03", "LK001", "SL", "RSL", "SNA", "KKA", "KKT1")) -> dict:
    out = {}
    print("折れ目 min(p,1-p) と片側ユニット（少数側 <1%）。seed 0")
    print(f"{'arm':6s} {'cond':4s} {'t':>3s} | {'l1 折れ目':>9s} {'片側':>5s} | {'l2 折れ目':>9s} {'片側':>5s}")
    for arm in arms:
        for cond in ("raw", "std"):
            for t in (1, 10, 50):
                d = snap(arm, cond, 0, t)
                a1, a2 = alphas(d)
                r = []
                for z, a in ((torch.from_numpy(d["z1"]).float(), a1),
                             (torch.from_numpy(d["z2"]).float(), a2)):
                    p = (z > 0).float().mean(0)
                    f = torch.minimum(p, 1 - p)
                    g = gate(arm, z, a)
                    r += [float(f.mean()), float((f < 0.01).float().mean()),
                          float((g.std(0) / g.mean(0).abs().clamp(min=1e-9)).median())]
                out[f"{arm}_{cond}_t{t}"] = r
                print(f"{arm:6s} {cond:4s} {t:3d} | {r[0]:9.3f} {r[1]:5.0%} | {r[3]:9.3f} {r[4]:5.0%}")
        print()

    print("\n初期化時（学習 0 step）の片側ユニット、seed 0-9。予測 = 2*Phi(-2.33/r)")
    Phi = lambda u: 0.5 * (1 + math.erf(u / math.sqrt(2)))
    for cond in ("raw", "std"):
        rs, obs = [], []
        for s in range(10):
            x = inputs(s, cond)
            rs.append(float(x.mean(0).norm()) / spread(x) / math.sqrt(x.shape[1]) * math.sqrt(x.shape[1]))
            rs[-1] = float(x.mean(0).norm()) / float((x - x.mean(0)).pow(2).sum(1).mean().sqrt())
            d = snap("LR", cond, s, 0)
            z = x @ torch.from_numpy(d["W1"]).T + torch.from_numpy(d["b1"])
            p = (z > 0).float().mean(0)
            obs.append(float((torch.minimum(p, 1 - p) < 0.01).float().mean()))
        r = float(np.mean(rs))
        out[f"init_{cond}"] = {"r": r, "observed": float(np.mean(obs)), "predicted": 2 * Phi(-2.33 / r)}
        print(f"  {cond}: r = {r:.3f} +- {np.std(rs):.3f}   実測 {np.mean(obs):.1%} +- {np.std(obs):.1%}"
              f"   予測 {2 * Phi(-2.33 / r):.1%}")
    return out


def ratio() -> dict:
    """R = |w_i| * s_input / sd(z_i), unit median, every task."""
    arms = SNAKE + LEAKY
    out = {}
    for arm in arms:
        for cond in ("raw", "std"):
            su1 = spread(inputs(0, cond))
            rows = []
            for t in range(1, 51):
                d = snap(arm, cond, 0, t)
                a1, _ = alphas(d)
                z1 = torch.from_numpy(d["z1"]).float()
                z2 = torch.from_numpy(d["z2"]).float()
                W1 = torch.from_numpy(d["W1"])
                W2 = torch.from_numpy(d["W2"])
                a1v = gate  # noqa: F841  (kept explicit below)
                act1 = _phi(arm, z1, a1)
                rows.append([float((W1.norm(dim=1) * su1 / z1.std(0)).median()),
                             float((W2.norm(dim=1) * spread(act1) / z2.std(0)).median())])
            out[f"{arm}_{cond}"] = np.array(rows)
    # control: random directions at the t50 norms
    ctl = {}
    for cond in ("raw", "std"):
        x = inputs(0, cond); su = spread(x)
        W = torch.from_numpy(snap("LR", cond, 0, 50)["W1"])
        g = torch.Generator().manual_seed(4242)
        Wr = torch.randn(W.shape, generator=g)
        Wr = Wr / Wr.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True)
        ctl[cond] = float((Wr.norm(dim=1) * su / (x @ Wr.T).std(0)).median())
    print(f"対照（向きだけ乱数）: raw {ctl['raw']:.3f}  std {ctl['std']:.3f}   ← R=1 が「等方」")
    for arm in arms:
        for cond in ("raw", "std"):
            r = out[f"{arm}_{cond}"]
            print(f"  {arm:6s} {cond:4s} l1 {r[0,0]:.3f} -> {r[-1,0]:.3f}   l2 {r[0,1]:.3f} -> {r[-1,1]:.3f}")
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 6.4), dpi=110, sharex=True)
    t = np.arange(1, 51)
    for j, cond in enumerate(("raw", "std")):
        for i, L in enumerate((0, 1)):
            a = ax[i, j]
            for arm in arms:
                a.plot(t, out[f"{arm}_{cond}"][:, L], lw=1.7, color=COL[arm],
                       ls="-" if arm in SNAKE else "--", label=arm)
            a.axhline(1.0, color="0.4", lw=1.2, ls=":", zorder=0)
            a.set_ylim(0, 1.35 if L == 1 else 0.75)
            a.grid(alpha=.25)
            a.set_title(f"入力 {cond}・第 {L+1} 層", fontsize=10)
            if i == 1:
                a.set_xlabel("タスク")
            if j == 0:
                a.set_ylabel(f"第 {L+1} 層  R")
    ax[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.suptitle("R = ‖w_i‖ × 入力の幅 ÷ 前活性の幅（ユニット中央値・seed 0）　点線 R=1 は「重みが入力に対して等方」",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "ratio_wx_over_z.png")
    return {k: v.tolist() for k, v in out.items()} | {"control_random_direction": ctl}


def _phi(arm, z, a):
    if arm in ("LR", "LK03", "LK001"):
        s = {"LR": 0.1, "LK03": 0.3, "LK001": 0.01}[arm]
        return torch.where(z > 0, z, s * z)
    if arm in ("SL", "RSL"):
        al = 0.1 if arm == "SL" else 0.229
        return al * z + (1 - al) * z * torch.sigmoid(5.0 * z / 3.0)
    ia = 1.0 / a
    sn = z + torch.sin(a * z) ** 2 * ia
    th = 2 * a * z
    if arm == "SNA":
        return sn
    if arm in ("KKA", "KKA23"):
        o = torch.where(th < -1.5 * PI, 2 * z + 0.75 * PI * ia + 0.5 * ia,
                        torch.where(th > 0.5 * PI, 2 * z - 0.25 * PI * ia + 0.5 * ia, sn))
        return o * (2 / 3) if arm == "KKA23" else o
    return torch.where(th < -2 * PI, z, torch.where(th > PI, z + ia, sn))


def wnorm() -> dict:
    d = pd.concat([pd.read_csv(OUT / a / "per_task.csv", float_precision="round_trip") for a in ARMS])
    out = {}
    print(f"初期のユニット行ノルムは 3 層とも 1/sqrt(3) = {INIT_ROW:.4f}")
    print(f"{'arm':6s} {'cond':4s} | {'l1 t50/init':>11s} {'指数':>5s} | {'l2 t50/init':>11s} {'指数':>5s} |"
          f" {'l3 t50/init':>11s} {'指数':>5s}")
    for arm in ARMS:
        for cond in ("raw", "std"):
            g = (d[(d.arm == arm) & (d.cond == cond)]
                 .groupby("task")[["w_norm_l1", "w_norm_l2", "w_norm_l3"]].median())
            tt = g.index.values
            m = tt >= 5                                     # skip task 1-4, the initial burst
            ex = [float(np.polyfit(np.log(tt[m]), np.log(g[f"w_norm_l{L}"].values[m]), 1)[0])
                  for L in (1, 2, 3)]
            out[f"{arm}_{cond}"] = {"final_over_init": [float(g[f"w_norm_l{L}"].iloc[-1] / INIT_ROW)
                                                        for L in (1, 2, 3)], "exponent": ex}
            print(f"{arm:6s} {cond:4s} |" + "".join(
                f" {g[f'w_norm_l{L}'].iloc[-1]/INIT_ROW:11.0f} {ex[L-1]:5.2f} |" for L in (1, 2, 3)))
        print()
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), dpi=110)
    for j, cond in enumerate(("raw", "std")):
        for i, L in enumerate((1, 2, 3)):
            a = ax[j, i]
            for arm in ARMS:
                g = (d[(d.arm == arm) & (d.cond == cond)].groupby("task")[f"w_norm_l{L}"].median())
                ls = "-" if arm in SNAKE else ("--" if arm in LEAKY else ":")
                a.plot(g.index, g.values / INIT_ROW, lw=1.6, color=COL[arm], ls=ls, label=arm)
            t = np.arange(1, 51)
            a.plot(t, 3.0 * np.sqrt(t), color="k", lw=1.0, ls=(0, (6, 3)), alpha=.45, zorder=0)
            a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=.25, which="both")
            a.set_title(f"入力 {cond}・第 {L} 層", fontsize=10)
            if i == 0:
                a.set_ylabel("ユニット行ノルム ÷ 初期値")
            if j == 1:
                a.set_xlabel("タスク")
    ax[0, 0].legend(fontsize=6.5, ncol=3, loc="upper left")
    ax[0, 2].text(.97, .05, "黒破線 = √t の傾き", transform=ax[0, 2].transAxes,
                  ha="right", fontsize=8, color="0.3")
    fig.suptitle("重みのノルムの時間変化（ユニット行ノルムの中央値 ÷ 初期値 1/√3・10 seed 中央値）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "wnorm_over_tasks.png")
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if what in ("fold", "all"):
        res["fold"] = fold()
    if what in ("ratio", "all"):
        res["ratio"] = ratio()
    if what in ("wnorm", "all"):
        res["wnorm"] = wnorm()
    (OUT / "posthoc_0918.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {OUT}/posthoc_0918.json  (POST-HOC, 登録外)")
