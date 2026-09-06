# -*- coding: utf-8 -*-
"""smooth_kink_0907 の判定（spec `specs/spec_smooth_kink_0907.md` §4）と診断の表。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.smooth_kink_analyze_0907 \\
        [--logs results/smooth_kink_0907/logs_tail] [--out results/smooth_kink_0907]

主判定は 1 本: 押し下げの深さ d = −(tail 窓の z̄ の ALL ユニット中央値) の、参照からの差
Δd(s) の CI が 0 を除外する**最小の s**（梯子 A = a 0.1）を s* とし、
s* ≤ 0.1 → DELTA_SCALE / s* ≥ 1 → SUPPORT_SCALE / s* = 0.3 → INTERMEDIATE /
どれも 0 を含む → SHARPNESS_IRRELEVANT。梯子 B（a=0.5）は補助。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .act_offset_analyze_0906 import _alive, _load_arm, _tail, ci_excludes_zero
from .offset_grid_analyze_0906 import support_groups, w_ratios
from .common import ROOT, load_config

CFG = Path(ROOT) / "configs" / "smooth_kink_0907.yaml"
T = 10_000
NEEDED = ("step", "seed", "unfit", "layer1_denom", "layer1_zbar", "layer1_zmax",
          "layer1_zmin", "layer1_mob", "layer1_v_unit", "layer1_w_norm",
          "layer1_w_free", "layer1_w_free_step", "layer1_m_dphiddphi", "layer1_m_dphi2")
LIN_V = 0.05
HALF_W = {0.1: 3.79, 0.5: 1.54}     # 既存ログの半幅中央値（spec §1 の表）


def paired_delta(qa: np.ndarray, qb: np.ndarray, rng, nboot: int):
    """Δ(中央値) を seed 単位の対で（同じ seed 添字を両腕に使う）。"""
    S = min(qa.shape[0], qb.shape[0])
    qa, qb = qa[:S], qb[:S]

    def stat(idx):
        a = qa[idx].reshape(-1)
        b = qb[idx].reshape(-1)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        return float(np.median(a) - np.median(b)) if a.size and b.size else np.nan
    point = stat(np.arange(S))
    bs = np.array([stat(rng.integers(0, S, S)) for _ in range(nboot)])
    return point, (float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))


def arm_row(logs: Path, arm: str, a: float, s: float, tail) -> dict | None:
    z = _load_arm(logs, arm, NEEDED)
    if not z:
        return None
    zb = _tail(z, "layer1_zbar", tail)
    g = support_groups(z, tail)
    v = np.abs(_tail(z, "layer1_v_unit", tail))
    mob = _tail(z, "layer1_mob", tail)
    alive = _alive(z, tail)
    row = dict(arm=arm, a=a, s=s, s_over_W=(s / HALF_W[a] if s else 0.0),
               n_seed=len(z), zbar_all=float(np.median(zb)),
               zbar_alive=float(np.median(zb[alive])) if alive.any() else float("nan"),
               depth=-float(np.median(zb)),
               frac_up=float(g["up"].mean()), frac_mid=float(g["mid"].mean()),
               frac_dn=float(g["dn"].mean()),
               mob_med=float(np.median(mob)), v_abs_med=float(np.median(v)),
               lin_rate=float((v < LIN_V).mean()),
               alive_rate=float(alive.mean()),
               unfit_tail=float(np.median([np.mean(x["unfit"][-50:]) for x in z])),
               curv_med=float(np.median(_tail(z, "layer1_m_dphiddphi", tail))),
               dphi2_med=float(np.median(_tail(z, "layer1_m_dphi2", tail))))
    row.update(w_ratios(z))
    row["_zb"] = zb
    return row


def analyze(logs: Path, out: Path) -> dict:
    cfg = load_config(str(CFG))
    an = cfg["analysis"]
    nboot, seed = int(an["boot"]["n"]), int(an["boot"]["seed"])
    rng = np.random.default_rng(seed)
    dial = {r["name"]: float(r["dial"]) for r in cfg["arms"]}
    sval = {k: float(v) for k, v in an["s_values"].items()}
    total = int(cfg["arms"][0]["total_steps"])
    n_task = total // T
    tail = (n_task - 49, n_task)
    not_run = set(an.get("not_run", []))

    rows: dict[str, dict] = {}
    for name in [an["reference"]["a0p1"], an["reference"]["a0p5"]] + an["ladder_a0p1"] + an["ladder_a0p5"]:
        r = arm_row(logs, name, dial[name], sval.get(name, 0.0), tail)
        if r is not None:
            rows[name] = r

    def ladder(names, ref_name):
        ref = rows.get(ref_name)
        out_rows = []
        for n in names:
            if n in not_run or n not in rows:
                out_rows.append(dict(arm=n, s=sval[n], status="NOT_RUN"))
                continue
            d, ci = paired_delta(rows[n]["_zb"], ref["_zb"], rng, nboot)
            out_rows.append(dict(arm=n, s=sval[n], s_over_W=rows[n]["s_over_W"],
                                 status="COMPLETE", dzbar=d, dzbar_ci=list(ci),
                                 ddepth=-d, ddepth_ci=[-ci[1], -ci[0]],
                                 moves=bool(ci_excludes_zero(ci))))
        return out_rows

    lad_a = ladder(an["ladder_a0p1"], an["reference"]["a0p1"])
    lad_b = ladder(an["ladder_a0p5"], an["reference"]["a0p5"])
    moved = [r for r in lad_a if r.get("moves")]
    s_star = min((r["s"] for r in moved), default=None)
    n_div = sum(1 for r in lad_a if r["status"] == "NOT_RUN")
    if n_div >= 3:
        label = "NOT_DETERMINED"
    elif s_star is None:
        label = "SHARPNESS_IRRELEVANT"
    elif s_star <= float(an["labels"]["delta_scale_max_s"]):
        label = "DELTA_SCALE"
    elif s_star >= float(an["labels"]["support_scale_min_s"]):
        label = "SUPPORT_SCALE"
    else:
        label = "INTERMEDIATE"
    s_star_b = min((r["s"] for r in lad_b if r.get("moves")), default=None)

    res = dict(experiment="smooth_kink_0907", tail=list(tail), n_boot=nboot, boot_seed=seed,
               label=label, s_star=s_star, s_star_ladder_b=s_star_b,
               ladder_a0p1=lad_a, ladder_a0p5=lad_b,
               arms=[{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in rows.values()])
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    L = [f"# smooth_kink_0907 判定: **{label}**（s\\* = {s_star}・梯子 B の s\\* = {s_star_b}）", "",
         "## 梯子 A（a=0.1・W≈3.79・参照 SKref_1216）", "",
         "| 腕 | s | s/W | Δ深さ [CI] | 参照と違う |", "|---|---|---|---|---|"]
    for r in lad_a:
        if r["status"] == "NOT_RUN":
            L.append(f"| `{r['arm']}` | {r['s']:g} | — | **NOT_RUN**（発散） | — |")
        else:
            L.append(f"| `{r['arm']}` | {r['s']:g} | {r['s_over_W']:.3f} | "
                     f"{r['ddepth']:+.3f} [{r['ddepth_ci'][0]:+.3f}, {r['ddepth_ci'][1]:+.3f}] | "
                     f"{'**はい**' if r['moves'] else 'いいえ'} |")
    L += ["", "## 梯子 B（a=0.5・W≈1.54・参照 SKa0p5ref_1216）", "",
          "| 腕 | s | s/W | Δ深さ [CI] | 参照と違う |", "|---|---|---|---|---|"]
    for r in lad_b:
        L.append(f"| `{r['arm']}` | {r['s']:g} | {r['s_over_W']:.3f} | "
                 f"{r['ddepth']:+.3f} [{r['ddepth_ci'][0]:+.3f}, {r['ddepth_ci'][1]:+.3f}] | "
                 f"{'**はい**' if r['moves'] else 'いいえ'} |")
    L += ["", "## 腕の表（診断・ラベルにしない）", "",
          "| 腕 | a | s | 深さ | 上 | 跨 | 下 | mob | \\|v\\| | 線形化 | ‖w_free‖比 | E[φ′φ″] | E[φ′²] | unfit |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows.values(), key=lambda x: (x["a"], x["s"])):
        L.append(f"| `{r['arm']}` | {r['a']} | {r['s']:g} | {r['depth']:.3f} | {r['frac_up']:.2f} | "
                 f"{r['frac_mid']:.2f} | {r['frac_dn']:.2f} | {r['mob_med']:.3f} | {r['v_abs_med']:.3f} | "
                 f"{r['lin_rate']:.3f} | {r.get('w_free_ratio', float('nan')):.2f} | "
                 f"{r['curv_med']:+.3g} | {r['dphi2_med']:.3f} | {r['unfit_tail']:.4g} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(ROOT) / "results/smooth_kink_0907/logs_tail"))
    ap.add_argument("--out", default=str(Path(ROOT) / "results/smooth_kink_0907"))
    a = ap.parse_args()
    analyze(Path(a.logs), Path(a.out))


if __name__ == "__main__":
    main()
