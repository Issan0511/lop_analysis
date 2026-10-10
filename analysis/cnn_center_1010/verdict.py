#!/usr/bin/env python3
"""Registered verdicts of cnn_center_1010 (spec §3-§4): LRc (centred input) against LR (raw input).

    python3 analysis/cnn_center_1010/verdict.py

Reads results/cnn_center_1010/LRc/per_task.csv and the reference
results/cnn_drive_verify_1009/LR/per_task.csv; the t0 rows are `evaluate` on the init parameters with
each arm's own input; per-channel quantities come from the task-end checkpoints (LR's are archived in
obsidian-research-data/cnn_drive_verify_1009).  Writes results/cnn_center_1010/summary.md and verdict.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import cnn_center_1010 as C              # noqa: E402  (installs nothing until install())

E, LRM, CN, H, RC = C.E, C.LRM, C.CN, C.H, C.RC
OUT = ROOT / "results" / "cnn_center_1010"
LR_CKPT = Path.home() / "Projects/obsidian-research-data/cnn_drive_verify_1009/results/cnn_drive_verify_1009/LR/ckpt"
LRC_CKPT = OUT / "LRc" / "ckpt"
SEEDS = list(range(10, 20))
TASKS = (1, 5, 10, 20, 30)


def sign_p(k: int, n: int) -> float:
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@torch.no_grad()
def channel_stats(B, chunk: int = 100) -> dict:
    """Per channel of c1 and c2: mean and sd of z over (N, H, W) for every run of the bundle."""
    R = B.R
    acc = {l: [torch.zeros(R, 16, dtype=torch.float64, device=B.X.device) for _ in range(2)] for l in (0, 1)}
    n = {0: 0, 1: 0}
    for i0 in range(0, CN.N_IMAGES, chunk):
        o = E.forward(B.P, B.X[:, i0:i0 + chunk], B.act)
        for l, z in ((0, o[0]), (1, o[2])):
            b = z.shape[0]
            zr = z.reshape(b, R, 16, -1).double()
            acc[l][0] += zr.sum((0, 3)); acc[l][1] += (zr * zr).sum((0, 3))
            n[l] += b * zr.shape[3]
    out = {}
    for l, tag in ((0, "c1"), (1, "c2")):
        m = acc[l][0] / n[l]
        out[f"zbar_{tag}"] = m.cpu()
        out[f"zsd_{tag}"] = (acc[l][1] / n[l] - m * m).clamp_min(0).sqrt().cpu()
    return out


def load_into(B, files: list[Path]) -> None:
    with torch.no_grad():
        for r, f in enumerate(files):
            st = torch.load(f, map_location="cpu", weights_only=False)
            for q, x in zip(B.P, st["P"]):
                q[r].copy_(x)


def main() -> None:
    device = H.setup("cuda")
    cifar = RC.Cifar10()
    C.install(cifar)
    ref = pd.read_csv(ROOT / "results/cnn_drive_verify_1009/LR/per_task.csv")
    new = pd.read_csv(OUT / "LRc" / "per_task.csv")
    T = int(new.task.max())
    arms = {"LR": (ref, LR_CKPT), "LRc": (new, LRC_CKPT)}

    # t0 (init) per-channel stats and per-task-end stats from the checkpoints
    stats, mpatch, Bs = {}, {}, {}
    for arm in arms:
        B = C.BundleLRc([(arm, s) for s in SEEDS], cifar, device, graph=False)
        Bs[arm] = B
        mpatch[arm] = C.mean_patch(B.X).cpu()                       # (R, 75)
        stats[(arm, 0)] = channel_stats(B) | {"W1": B.P[0].detach().reshape(len(SEEDS), 16, 75).cpu().double(),
                                               "b1": B.P[1].detach().cpu().double()}
        init = [p.detach().clone() for p in B.P]
        for t in TASKS:
            if t > (30 if arm == "LR" else T):
                continue
            files = [arms[arm][1] / f"{arm}_seed{s}_t{t:02d}.pt" for s in SEEDS]
            if not all(f.exists() for f in files):
                continue
            load_into(B, files)
            stats[(arm, t)] = channel_stats(B) | {"W1": B.P[0].detach().reshape(len(SEEDS), 16, 75).cpu().double(),
                                                   "b1": B.P[1].detach().cpu().double()}
        with torch.no_grad():
            for q, x in zip(B.P, init):
                q.copy_(x)
    m_raw = mpatch["LR"]                                            # raw mean patch per seed
    mhat = m_raw / m_raw.norm(dim=1, keepdim=True)

    L = ["# cnn_center_1010 — 入力を中心化した leaky CNN（LRc）対 生の入力（LR）", "",
         f"LRc の完了課題: {T}。seed 10–19。z̄ はチャネル平均（N・H・W）、per_task の zbar はその中央値。", ""]
    # ---- per_task table
    rows = []
    for arm, (d, _) in arms.items():
        g = d.groupby("task")[["online_acc", "zbar_c1", "zsd_c1", "zbar_c2", "zsd_c2", "zbar_f1", "zsd_f1",
                               "zbar_f2", "zsd_f2", "w_norm_c1"]].mean()
        for t in TASKS:
            if t in g.index:
                rows.append({"arm": arm, "task": t, **g.loc[t].to_dict()})
    tab = pd.DataFrame(rows)
    tab["rel_c1"] = tab.zbar_c1 / tab.zsd_c1
    tab["rel_c2"] = tab.zbar_c2 / tab.zsd_c2
    L += ["| 腕 | t | online | z̄_c1 | sd_c1 | r1 | z̄_c2 | sd_c2 | r2 | z̄_f2/sd_f2 | ‖W1‖ 行 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        L.append(f"| {r.arm} | {int(r.task)} | {r.online_acc:.4f} | {r.zbar_c1:+.3f} | {r.zsd_c1:.3f} | {r.rel_c1:+.2f} | "
                 f"{r.zbar_c2:+.3f} | {r.zsd_c2:.3f} | {r.rel_c2:+.2f} | {r.zbar_f2 / r.zsd_f2:+.2f} | {r.w_norm_c1:.3f} |")

    V = {}
    # ---- primary: D1 = zbar_c1(t30) - zbar_c1(t0), zbar_c1 = channel median of channel means
    tend = min(T, 30)
    D1 = {}
    for arm in arms:
        z0 = stats[(arm, 0)]["zbar_c1"].median(dim=1).values.numpy()
        if (arm, tend) in stats:
            zt = stats[(arm, tend)]["zbar_c1"].median(dim=1).values.numpy()
        else:
            d = arms[arm][0]
            zt = d[d.task == tend].set_index("seed").loc[SEEDS, "zbar_c1"].to_numpy()
        D1[arm] = zt - z0
    rho = float(D1["LRc"].mean() / D1["LR"].mean())
    lab = "C1_STOPS" if rho <= 0.25 else "C1_KEEPS" if rho >= 0.75 else "C1_PARTIAL"
    k = int((D1["LRc"] < 0).sum())
    V["Q"] = {"label": lab if tend == 30 else f"(interim t{tend}) " + lab, "rho": rho,
              "D1_LR_mean": float(D1["LR"].mean()), "D1_LR_sd": float(D1["LR"].std(ddof=1)),
              "D1_LRc_mean": float(D1["LRc"].mean()), "D1_LRc_sd": float(D1["LRc"].std(ddof=1)),
              "LRc_seeds_sinking": k, "sign_p": sign_p(k, len(SEEDS))}

    # ---- per-channel: fraction sank, routes
    L += ["", f"チャネルごと（t0 → t{tend}、10 seed × 16 チャネル）:", "",
          "| 腕 | c1 で沈んだ割合 | c1 Δz̄ 中央値 | うち Δb | うち Δ⟨W, m⟩ | W1 の DC 成分 a（m̂ 方向）t0 → t | c2 で沈んだ割合 | c2 Δz̄ 中央値 |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        if (arm, tend) not in stats:
            continue
        s0, s1 = stats[(arm, 0)], stats[(arm, tend)]
        dz1 = s1["zbar_c1"] - s0["zbar_c1"]
        dz2 = s1["zbar_c2"] - s0["zbar_c2"]
        db = s1["b1"] - s0["b1"]
        mw = torch.einsum("rjk,rk->rj", s1["W1"] - s0["W1"], mpatch[arm])        # Δ<W, m> with the arm's m
        a0 = torch.einsum("rjk,rk->rj", s0["W1"], mhat); a1 = torch.einsum("rjk,rk->rj", s1["W1"], mhat)
        f1, f2 = float((dz1 < 0).double().mean()), float((dz2 < 0).double().mean())
        L.append(f"| {arm} | {f1:.3f} | {dz1.median():+.3f} | {db.median():+.3f} | {mw.median():+.3f} | "
                 f"{a0.median():+.3f} → {a1.median():+.3f} | {f2:.3f} | {dz2.median():+.3f} |")
        V[f"channels_{arm}"] = {"c1_sank_frac": f1, "c1_dz_median": float(dz1.median()), "c1_db_median": float(db.median()),
                                "c1_dWm_median": float(mw.median()), "a_dc_t0": float(a0.median()), "a_dc_t": float(a1.median()),
                                "c2_sank_frac": f2, "c2_dz_median": float(dz2.median())}
    if "channels_LRc" in V:
        f2 = V["channels_LRc"]["c2_sank_frac"]
        V["Q_c2"] = {"label": "C2_SINKS" if f2 >= 0.90 else "C2_STOPS" if f2 < 0.75 else "C2_PARTIAL", "frac": f2}

    # ---- relative position and plasticity
    r1 = {}
    for arm, (d, _) in arms.items():
        x = d[d.task == tend].set_index("seed").loc[SEEDS]
        r1[arm] = (x.zbar_c1 / x.zsd_c1).to_numpy()
    V["r1"] = {"LR": float(r1["LR"].mean()), "LRc": float(r1["LRc"].mean()),
               "half_of_LR": bool(abs(r1["LRc"].mean()) <= 0.5 * abs(r1["LR"].mean()))}
    if T >= 30:
        drop = {}
        lev = {}
        for arm, (d, _) in arms.items():
            e = d[d.task <= 5].groupby("seed").online_acc.mean().loc[SEEDS]
            l = d[(d.task >= 26) & (d.task <= 30)].groupby("seed").online_acc.mean().loc[SEEDS]
            drop[arm], lev[arm] = (e - l).to_numpy(), e.to_numpy()
        delta = float(drop["LRc"].mean() - drop["LR"].mean())
        V["Q_plast"] = {"label": "LESS_LOSS" if delta <= -0.01 else "MORE_LOSS" if delta >= 0.01 else "SAME_LOSS",
                        "drop_LR": float(drop["LR"].mean()), "drop_LRc": float(drop["LRc"].mean()), "delta": delta,
                        "level_t1_5_LR": float(lev["LR"].mean()), "level_t1_5_LRc": float(lev["LRc"].mean()),
                        "LRc_less_drop_seeds": int((drop["LRc"] < drop["LR"]).sum())}
    L += ["", "## 判定（spec §4）", ""]
    for k_, v in V.items():
        if "label" in v:
            L.append(f"- **{k_}** = `{v['label']}` — " + json.dumps({a: b for a, b in v.items() if a != 'label'},
                                                                    ensure_ascii=False, default=float))
        else:
            L.append(f"- {k_}: " + json.dumps(v, ensure_ascii=False, default=float))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(V, indent=1, ensure_ascii=False, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
