# -*- coding: utf-8 -*-
"""offset_grid_0906 の判定（spec `specs/spec_offset_grid_0906.md` §2）と診断の表（§1・§3・Log の読み規則）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.offset_grid_analyze_0906 \\
        [--logs results/offset_grid_0906/logs] [--ref results/act_offset_0906/logs] [--out results/offset_grid_0906]

判定ラベルはブロック 1（`SEAT_TRACKS_WELL`）と 2（`C_VIA_V_ONLY`）だけ。ほかは見る量を表にする。
3 分割は閾値ではなく作動条件: 上 = zmin > 0 ／ 跨ぐ = zmin ≤ 0 ≤ zmax ／ 下 = zmax < 0。
井戸の底: c>0 → −c/a（下の枝）、c<0 → +|c|（上の枝）。
act_offset_0906 の腕（同じマシン・同じ runner・`LRoff0_1216` がバイト一致）を参照として同じ表に載せる。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import edge_law_0905 as E
from .act_offset_analyze_0906 import (LIN_V, NBOOT, RNG_SEED, T, _alive, _load_arm, _tail,
                                      boot_median, ci_excludes_zero, ci_within)
from .common import ROOT, load_config

CFG = Path(ROOT) / "configs" / "offset_grid_0906.yaml"
CFG_REF = Path(ROOT) / "configs" / "act_offset_0906.yaml"
NEEDED = ("step", "seed", "lr_used", "freeze_v", "batch_mode", "unfit",
          "layer1_denom", "layer1_zbar", "layer1_zmax", "layer1_zmin", "layer1_mob",
          "layer1_v_unit", "layer1_w_norm", "layer1_w_free", "layer1_w_free_step")
REF_ARMS = ("LRoffm0p5_1216", "LRoffp0p5_1216", "LRoff0_lr0p00125_1216",
            "LRoffm2_lr0p00125_1216", "LRoffp2_lr0p00125_1216", "Eoffm1_1216", "Eoffp1_1216")
BANDS = dict(dzmax=0.3, dzbar=0.5)


def windows(total_steps: int):
    n = total_steps // T
    tail = (n - 49, n)
    settles = [(round(0.7 * n) - 49, round(0.7 * n)), (round(0.85 * n) - 49, round(0.85 * n)), tail]
    return tail, settles


def well(c: float, a: float):
    return dict(up=-c, dn=(-c / a if a > 0 else float("nan")))


def support_groups(z, tail):
    zn, zx = _tail(z, "layer1_zmin", tail), _tail(z, "layer1_zmax", tail)
    up, dn = zn > 0, zx < 0
    return dict(up=up, mid=~up & ~dn, dn=dn)


def w_ratios(z: list[dict]) -> dict:
    """‖w_free‖（幅）と ‖w_flip‖（位置）の 初期 → 末尾 の中央値比。"""
    f0, f1, p0, p1 = [], [], [], []
    for s in z:
        wf = s["layer1_w_free"].astype(np.float64)
        wn = s["layer1_w_norm"].astype(np.float64)
        a0, a1 = np.linalg.norm(wf[0], axis=-1), np.linalg.norm(wf[-1], axis=-1)
        f0.append(a0); f1.append(a1)
        p0.append(np.sqrt(np.maximum(wn[0] ** 2 - a0 ** 2, 0)))
        p1.append(np.sqrt(np.maximum(wn[-1] ** 2 - a1 ** 2, 0)))
    g = lambda L: float(np.median(np.concatenate(L)))
    return dict(w_free_init=g(f0), w_free_tail=g(f1), w_free_ratio=g(f1) / g(f0),
                w_flip_init=g(p0), w_flip_tail=g(p1), w_flip_ratio=g(p1) / g(p0))


def summarize(name, z, a, c, total_steps, rng) -> dict:
    tail, settles = windows(total_steps)
    zb, zx = _tail(z, "layer1_zbar", tail), _tail(z, "layer1_zmax", tail)
    v, w = _tail(z, "layer1_v_unit", tail), _tail(z, "layer1_w_norm", tail)
    al = _alive(z, tail)
    G = support_groups(z, tail)
    med = {}
    for g, m in G.items():
        med[g] = boot_median(np.where(m, zb, np.nan), rng) if m.any() else (np.nan, (np.nan, np.nan))
    zb_al = boot_median(np.where(al, zb, np.nan), rng)
    zb_all = boot_median(zb, rng)
    settle = [float(np.median(_tail(z, "layer1_zbar", wn)[_alive(z, wn)])) for wn in settles]
    unfit = float(np.mean([np.mean(s["unfit"][-50:]) for s in z]))
    return dict(arm=name, a=a, c=c, lr=float(z[0]["lr_used"]), freeze_v=bool(z[0]["freeze_v"]),
                batch_mode=str(z[0]["batch_mode"]), total_steps=total_steps, tail=list(tail),
                n_seeds=len(z), n_alive=int(al.sum()), death_rate=float(1 - al.mean()),
                zbar_alive=zb_al, zbar_all=zb_all, zmax_med=float(np.median(zx[al])),
                half_med=float(np.median((zx - zb)[al])), v_abs_med=float(np.median(np.abs(v[al]))),
                w_med=float(np.median(w[al])), lin_rate=float(np.mean(np.abs(v) < LIN_V)),
                frac={g: float(m.mean()) for g, m in G.items()}, med=med,
                well=well(c, a), settle_zbar=settle, unfit_tail=unfit, **w_ratios(z),
                _zb=zb, _zx=zx, _al=al, _G=G, _seeds=[int(s["seed"]) for s in z])


def paired(qa, qb, key, rng, mask_extra=None):
    """seed 対応・ユニット対応の差（両腕 ALIVE ＋ 任意のマスク）。"""
    ia = {s: i for i, s in enumerate(qa["_seeds"])}; ib = {s: i for i, s in enumerate(qb["_seeds"])}
    common = sorted(set(ia) & set(ib)); i1 = [ia[s] for s in common]; i2 = [ib[s] for s in common]
    both = qa["_al"][i1] & qb["_al"][i2]
    if mask_extra is not None:
        both &= mask_extra[i1]
    d = np.where(both, qa[key][i1] - qb[key][i2], np.nan)
    pt, ci = boot_median(d, rng)
    return dict(point=pt, ci=ci, n_pairs=int(both.sum()), n_seeds=len(common))


def analyze(logs: Path, ref: Path, out: Path) -> dict:
    cfg = load_config(str(CFG)); table = E.arm_table(cfg)
    ref_table = E.arm_table(load_config(str(CFG_REF)))
    offset = {n: float(cfg["activation"][r["activation"]].get("offset", 0.0)) for n, r in table.items()}
    ref_offset = {n: float(load_config(str(CFG_REF))["activation"][r["activation"]].get("offset", 0.0))
                  for n, r in ref_table.items()}
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []
    Q: dict[str, dict] = {}

    def load(name, logdir, tbl, off):
        z = _load_arm(logdir, name, NEEDED)
        if not z:
            return None
        drop = [int(s["seed"]) for s in z if not np.all(np.isfinite(s["layer1_zbar"]))]
        z = [s for s in z if int(s["seed"]) not in drop]
        q = summarize(name, z, float(tbl[name]["dial"]), off[name], int(tbl[name]["total_steps"]), rng)
        q["dropped"] = drop
        return q

    missing = []
    for name in table:
        q = load(name, logs, table, offset)
        if q is None:
            missing.append(name); continue
        Q[name] = q
    for name in REF_ARMS:
        q = load(name, ref, ref_table, ref_offset)
        if q is not None:
            Q["AO:" + name] = q

    def block_of(n):
        if n.startswith("AO:"): return "参照(act_offset)"
        if n.startswith("LRa0p"): return "1 a×c"
        if "_vf1" in n or n == "LRvf1_1216": return "2 v凍結"
        if "_lr0p00125" in n: return "3 ラダーB"
        if n.startswith("FBLR"): return "4 full-batch"
        if "0p25" in n: return "5 小さいc"
        if n.startswith("Eoff"): return "6 ELU"
        if n.endswith("bp2_1216"): return "7 b+2"
        return "S-null"

    for n, q in Q.items():
        rows.append(dict(section="arm", block=block_of(n), arm=n, a=q["a"], c=q["c"], lr=q["lr"],
                         batch=q["batch_mode"], freeze_v=q["freeze_v"], steps=q["total_steps"],
                         n_seeds=q["n_seeds"], dropped=q["dropped"], death_rate=q["death_rate"],
                         zbar_alive=q["zbar_alive"][0], zbar_alive_ci=q["zbar_alive"][1],
                         zbar_all=q["zbar_all"][0], zmax_med=q["zmax_med"], half_med=q["half_med"],
                         v_abs_med=q["v_abs_med"], w_med=q["w_med"], lin_rate=q["lin_rate"],
                         frac_up=q["frac"]["up"], frac_mid=q["frac"]["mid"], frac_dn=q["frac"]["dn"],
                         med_up=q["med"]["up"][0], med_mid=q["med"]["mid"][0], med_dn=q["med"]["dn"][0],
                         well_up=q["well"]["up"], well_dn=q["well"]["dn"], settle_zbar=q["settle_zbar"],
                         unfit_tail=q["unfit_tail"], w_free_ratio=q["w_free_ratio"],
                         w_flip_ratio=q["w_flip_ratio"], w_free_tail=q["w_free_tail"],
                         w_flip_tail=q["w_flip_tail"]))
    for n in missing:
        rows.append(dict(section="arm", block=block_of(n), arm=n, label="NOT_RUN"))

    # ---- ブロック 1: SEAT_TRACKS_WELL（c=+0.5 の「下」群の z̄ − (−c/a) の CI ⊂ ±0.3）----
    b1 = []
    for a in (0.2, 0.3, 0.5, 0.7):
        n = f"LRa{str(a).replace('.', 'p')}_offp0p5_1216"
        q = Q.get(n)
        if q is None:
            b1.append(dict(a=a, arm=n, label="NOT_RUN")); continue
        dn = q["_G"]["dn"]; target = q["well"]["dn"]
        pt, ci = boot_median(np.where(dn, q["_zb"], np.nan), rng)
        diff = (pt - target, (ci[0] - target, ci[1] - target))
        settle = q["settle_zbar"]; mono = (settle[0] < settle[1] < settle[2]) or (settle[0] > settle[1] > settle[2])
        drift = max(settle) - min(settle); unsettled = bool(mono and drift > (ci[1] - ci[0]))
        b1.append(dict(a=a, arm=n, target=target, dn_frac=q["frac"]["dn"], dn_med=pt, dn_ci=ci,
                       diff=diff[0], diff_ci=diff[1], within=ci_within(diff[1], -0.3, 0.3),
                       unsettled=unsettled, dropped=len(q["dropped"]), up_frac=q["frac"]["up"],
                       mid_frac=q["frac"]["mid"]))
    ok = [r for r in b1 if r.get("label") != "NOT_RUN"]
    nd = (len(ok) < 4 or any(r["dropped"] > 3 for r in ok) or any(r["unsettled"] for r in ok)
          or sum(r["dn_frac"] < 0.2 for r in ok) >= 2)
    k = sum(r["within"] for r in ok)
    if nd:
        lab1 = "NOT_DETERMINED"
    elif k == 4:
        lab1 = "SEAT_TRACKS_WELL"
    elif k >= 2 and all(r["within"] for r in ok if r["a"] < 0.5):
        lab1 = "SEAT_TRACKS_WELL_SHALLOW_ONLY"
    elif k <= 1:
        lab1 = "SEAT_OFF_WELL"
    else:
        lab1 = "INCONCLUSIVE"
    rows.append(dict(section="verdict", block="1 a×c", judgment="§2 ブロック 1", label=lab1, k=k, per_a=b1))

    # ---- ブロック 2: C_VIA_V_ONLY（v 凍結下の Δ(±0.5) 対 LRvf1）----
    b2 = {}
    if "LRvf1_1216" in Q:
        for n in ("LRoffm0p5_vf1_1216", "LRoffp0p5_vf1_1216"):
            if n in Q:
                b2[n] = dict(dzbar=paired(Q[n], Q["LRvf1_1216"], "_zb", rng),
                             dzmax=paired(Q[n], Q["LRvf1_1216"], "_zx", rng))
    unfit_ref = Q["LRoff0_1216"]["unfit_tail"] if "LRoff0_1216" in Q else np.nan
    unfit_vf = Q["LRvf1_1216"]["unfit_tail"] if "LRvf1_1216" in Q else np.nan
    unfit_bad = bool(np.isfinite(unfit_ref) and np.isfinite(unfit_vf) and unfit_vf > 80 * unfit_ref)
    if len(b2) < 2 or unfit_bad or any(len(Q[n]["dropped"]) > 3 for n in b2):
        lab2 = "NOT_DETERMINED"
    else:
        within = all(ci_within(b2[n]["dzbar"]["ci"], -BANDS["dzbar"], BANDS["dzbar"])
                     and ci_within(b2[n]["dzmax"]["ci"], -BANDS["dzmax"], BANDS["dzmax"]) for n in b2)
        excl = any(ci_excludes_zero(b2[n]["dzbar"]["ci"]) or ci_excludes_zero(b2[n]["dzmax"]["ci"]) for n in b2)
        lab2 = "C_VIA_V_ONLY" if within else ("C_NOT_ONLY_VIA_V" if excl else "INCONCLUSIVE")
    rows.append(dict(section="verdict", block="2 v凍結", judgment="§2 ブロック 2", label=lab2,
                     deltas=b2, unfit_ref=unfit_ref, unfit_vf=unfit_vf, unfit_cutoff_failed=unfit_bad))

    # ---- 診断の Δ（各ブロックの c=0 参照に対して）----
    def delta_row(n, ref_n, block):
        if n in Q and ref_n in Q:
            dz, dx = paired(Q[n], Q[ref_n], "_zb", rng), paired(Q[n], Q[ref_n], "_zx", rng)
            rows.append(dict(section="delta", block=block, arm=n, ref=ref_n, c=Q[n]["c"], a=Q[n]["a"],
                             dzbar=dz["point"], dzbar_ci=dz["ci"], dzmax=dx["point"], dzmax_ci=dx["ci"],
                             n_pairs=dz["n_pairs"]))
    for a in ("0p2", "0p3", "0p5", "0p7"):
        for s in ("m0p5", "p0p5"):
            delta_row(f"LRa{a}_off{s}_1216", f"LRa{a}_off0_1216", "1 a×c")
    for n in ("LRoffm0p25_1216", "LRoffp0p25_1216", "AO:LRoffm0p5_1216", "AO:LRoffp0p5_1216"):
        delta_row(n, "LRoff0_1216", "5 小さいc / 参照")
    for n in ("LRoffm1_lr0p00125_1216", "LRoffm0p5_lr0p00125_1216", "LRoffp0p5_lr0p00125_1216",
              "LRoffp1_lr0p00125_1216", "AO:LRoffm2_lr0p00125_1216", "AO:LRoffp2_lr0p00125_1216"):
        delta_row(n, "AO:LRoff0_lr0p00125_1216", "3 ラダーB")
    for n in ("FBLRoffm0p5_1216", "FBLRoffp0p5_1216"):
        delta_row(n, "FBLRoff0_1216", "4 full-batch")
    for n in ("AO:Eoffm1_1216", "AO:Eoffp1_1216"):
        delta_row(n, "Eoff0_1216", "6 ELU")

    # ---- Chen 型か一次か（Log の読み規則）: c=−0.5 の「上」割合を 3 条件で ----
    chen = {}
    for lab, n in (("online lr 0.01", "AO:LRoffm0p5_1216"), ("online lr 0.00125 (雑音 1/8)", "LRoffm0p5_lr0p00125_1216"),
                   ("full-batch lr 0.01 (雑音 0)", "FBLRoffm0p5_1216")):
        if n in Q:
            q = Q[n]
            chen[lab] = dict(arm=n, up_frac=q["frac"]["up"], up_med=q["med"]["up"][0],
                             lin_rate=q["lin_rate"], zbar_all=q["zbar_all"][0])
    rows.append(dict(section="chen", block="Log の読み", arm="c=−0.5 の『上』群の割合", per=chen,
                     rule="一次なら 3 条件で ≈10%、Chen 型（Itô ドリフト ∝ η）なら雑音 0 で ≈0・雑音 1/8 で数 %"))

    result = dict(labels=dict(block1=lab1, block2=lab2), missing=missing, rows=rows)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with (out / "verdict.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader()
        for r in rows:
            wr.writerow({k: (r.get(k) if isinstance(r.get(k), (str, int, float, type(None)))
                             else json.dumps(r.get(k), default=str, ensure_ascii=False)) for k in keys})
    (out / "summary.md").write_text(_markdown(rows, result), encoding="utf-8")
    return result


def _f(x, nd=2):
    if x is None: return "—"
    if isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[1], (tuple, list)):
        return f"{x[0]:.{nd}f} [{x[1][0]:.{nd}f}, {x[1][1]:.{nd}f}]"
    if isinstance(x, float): return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _markdown(rows, result) -> str:
    L = ["# offset_grid_0906 判定と診断（spec §2・§1・Log の読み規則）", "",
         "3 分割 = 上 zmin>0 ／ 跨ぐ ／ 下 zmax<0（ラチェットの作動条件）。井戸の底: 上の枝 −c・下の枝 −c/a。"
         "z̄ の「ALIVE」は denom>0.25、群の中央値と割合は ALL ユニット。CI = seed bootstrap 2000・rng(20260906)。"
         "`AO:` は act_offset_0906 の腕（同じマシン・`LRoff0_1216` がバイト一致）。", ""]
    L += [f"## 登録判定: ブロック 1 **`{result['labels']['block1']}`** ／ ブロック 2 **`{result['labels']['block2']}`**", ""]
    for r in rows:
        if r.get("section") == "verdict" and r["block"].startswith("1"):
            L += ["### ブロック 1 — c=+0.5 の「下」群は −c/a に座るか（k = ±0.3 に内包された腕の数 = " + str(r["k"]) + "）", "",
                  "| a | 下の割合 | 下の z̄ [CI] | 底 −c/a | 差 [CI] | 内包 | 上% | 跨% | 未定着 |", "|---|---|---|---|---|---|---|---|---|"]
            for p in r["per_a"]:
                if p.get("label") == "NOT_RUN": L.append(f"| {p['a']} | NOT_RUN |||||||"); continue
                L.append(f"| {p['a']} | {p['dn_frac']:.2f} | {_f((p['dn_med'], p['dn_ci']))} | {p['target']:+.2f} | "
                         f"{_f((p['diff'], p['diff_ci']))} | {'✔' if p['within'] else '✘'} | {p['up_frac']*100:.0f} | {p['mid_frac']*100:.0f} | {p['unsettled']} |")
            L.append("")
        if r.get("section") == "verdict" and r["block"].startswith("2"):
            L += ["### ブロック 2 — v 凍結下の Δ(c) 対 `LRvf1`（両腕 ALIVE・ユニット対応）", "",
                  "| 腕 | Δz̄ [CI] | Δzmax [CI] | n 対 |", "|---|---|---|---|"]
            for n, d in r["deltas"].items():
                L.append(f"| `{n}` | {_f((d['dzbar']['point'], d['dzbar']['ci']))} | {_f((d['dzmax']['point'], d['dzmax']['ci']))} | {d['dzbar']['n_pairs']} |")
            L += [f"", f"unfit（末尾）: LRvf1 {r['unfit_vf']:.3g} 対 LRoff0 {r['unfit_ref']:.3g}（80 倍足切り {'発動' if r['unfit_cutoff_failed'] else '通過'}）", ""]
    L += ["## 腕ごとの末尾（診断）", "",
          "| ブロック | 腕 | a | c | lr | batch | 生存 | z̄ ALIVE | z̄ ALL | zmax | 半幅 | \\|v\\| | 線形化 | 上% z̄ | 跨% z̄ | 下% z̄ | 底 −c / −c/a | ‖w_free‖ 比 | ‖w_flip‖ 比 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r.get("section") == "arm"], key=lambda r: (r["block"], r.get("a", 0), r.get("c", 0))):
        if r.get("label") == "NOT_RUN":
            L.append(f"| {r['block']} | `{r['arm']}` | NOT_RUN（発散）" + " |" * 17); continue
        g = lambda k: f"{r['frac_'+k]*100:.0f}% {_f(r['med_'+k])}"
        L.append(f"| {r['block']} | `{r['arm']}` | {r['a']} | {r['c']:+g} | {r['lr']:g} | {r['batch']}{' vf' if r['freeze_v'] else ''} | "
                 f"{1-r['death_rate']:.2f} | {_f((r['zbar_alive'], r['zbar_alive_ci']))} | {_f(r['zbar_all'])} | {_f(r['zmax_med'])} | "
                 f"{r['half_med']:.2f} | {r['v_abs_med']:.3f} | {r['lin_rate']:.2f} | {g('up')} | {g('mid')} | {g('dn')} | "
                 f"{r['well_up']:+.2f} / {r['well_dn']:+.2f} | ×{r['w_free_ratio']:.2f} | ×{r['w_flip_ratio']:.2f} |")
    L += ["", "## Δ(c) 対 各ブロックの c=0（ユニット対応・両腕 ALIVE）", "",
          "| ブロック | 腕 | 参照 | a | c | Δz̄ [CI] | Δzmax [CI] | n 対 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("section") != "delta": continue
        L.append(f"| {r['block']} | `{r['arm']}` | `{r['ref']}` | {r['a']} | {r['c']:+g} | {_f((r['dzbar'], r['dzbar_ci']))} | {_f((r['dzmax'], r['dzmax_ci']))} | {r['n_pairs']} |")
    for r in rows:
        if r.get("section") == "chen":
            L += ["", "## 井戸の駆動: 雑音か一次か（Log の読み規則・c=−0.5 の「上」群）", "", f"規則: {r['rule']}", "",
                  "| 条件 | 腕 | 上の割合 | 上の z̄ | 線形化率 | z̄ ALL |", "|---|---|---|---|---|---|"]
            for lab, p in r["per"].items():
                L.append(f"| {lab} | `{p['arm']}` | **{p['up_frac']*100:.1f}%** | {_f(p['up_med'])} | {p['lin_rate']:.2f} | {_f(p['zbar_all'])} |")
    L += ["", "## 引用上の注意", "",
          "1. 登録ラベルはブロック 1・2 だけ。他は診断（予測つき・ラベルなし）。",
          "2. lr 0.00125 の腕（ラダー B）と full-batch は lr 0.01 の腕と水準比較しない。持ち出すのは向きと『上』群の割合まで。",
          "3. 3 分割の割合は ALL ユニットの量。ALIVE 規則は潰れたユニットを落とすので、群の議論には使わない。",
          "4. b+2 の 2 腕は発散（NOT_RUN）。「上から戻らない」とは読まない。"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(ROOT) / "results/offset_grid_0906/logs"))
    ap.add_argument("--ref", default=str(Path(ROOT) / "results/act_offset_0906/logs"))
    ap.add_argument("--out", default=str(Path(ROOT) / "results/offset_grid_0906"))
    a = ap.parse_args()
    r = analyze(Path(a.logs), Path(a.ref), Path(a.out))
    print(json.dumps(dict(labels=r["labels"], missing=r["missing"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
