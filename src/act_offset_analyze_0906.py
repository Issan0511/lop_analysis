# -*- coding: utf-8 -*-
"""act_offset_0906 の登録判定（spec `specs/spec_act_offset_0906.md` §4）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.act_offset_analyze_0906 \\
        [--logs results/act_offset_0906/logs] [--ckpts results/act_offset_0906/ckpts] \\
        [--out results/act_offset_0906]

主量 = **zmax の末尾中央値**（タスク 451–500・ALIVE = `layer1_denom` 末尾平均 > 0.25）、
副量 = z̄ の末尾中央値。Δ(c) = 腕 c − 腕 c=0 の**ユニット対応差**（両腕 ALIVE・活性化は
乱数を消費しないので (seed, unit) が全腕で対応する = edge_law の G5）の中央値、
CI は seed bootstrap 2000 回・percentile 95%・`default_rng(20260906)`。
ラベル（排他・§4 の順）: NOT_DETERMINED → OFFSET_IRRELEVANT → OFFSET_SIGNED → OFFSET_OTHER → INCONCLUSIVE。
REPORT（登録外・併記）: |v|・‖w‖・線形化率（|v|<0.05）・出力バイアス c_out（ckpt 5M の `net.c`）・
ELU 2 腕の対応差・復元場（条件 z̄(t−2task)・変位 z̄(t+1task)−z̄(t)）の零点。
`snake_flip_analyze_0906` に倣い numpy のみ（c_out だけ torch で ckpt を読む）。
logs_tail（`--tail-extract` の縮約）でも同じ窓が読める。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import ROOT, load_config
from .mlp2_phase0b import _window_indices

CFG = Path(ROOT) / "configs" / "act_offset_0906.yaml"
T = 10_000
RNG_SEED = 20260906
NBOOT = 2000
LIN_V = 0.05                      # 線形化（読み出しの消失）の閾値 |v| < 0.05（snake_flip と同じ）


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def _load_arm(logdir: Path, arm: str) -> list[dict]:
    out = []
    for f in sorted(Path(logdir).glob(f"{arm}_seed*.npz")):
        with np.load(f, allow_pickle=True) as z:
            out.append({k: z[k] for k in z.files})
    return sorted(out, key=lambda z: int(z["seed"]))


def _tail(seeds: list[dict], key: str, tasks) -> np.ndarray:
    """(seed, unit) の窓平均（タスク終端記録の平均）。"""
    rows = []
    for z in seeds:
        idx = _window_indices(z["step"], T, list(tasks))
        rows.append(z[key][idx].astype(np.float64).mean(axis=0))
    return np.stack(rows)


def _alive(seeds: list[dict], tasks) -> np.ndarray:
    return _tail(seeds, "layer1_denom", tasks) > 0.25


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def boot_median(values: np.ndarray, rng: np.random.Generator, stat=np.median):
    """seed 単位の復元抽出。`values` は (S,) か (S, h)（NaN はマスク済み・unit をプール）。"""
    S = values.shape[0]

    def pooled(idx):
        v = values[idx]
        v = v.reshape(-1) if v.ndim > 1 else v
        v = v[np.isfinite(v)]
        return float(stat(v)) if v.size else np.nan
    point = pooled(np.arange(S))
    bs = np.array([pooled(rng.integers(0, S, S)) for _ in range(NBOOT)])
    return point, (float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))


def ci_within(ci, lo: float, hi: float) -> bool:
    return (np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] >= lo and ci[1] <= hi)


def ci_excludes_zero(ci) -> bool:
    return np.isfinite(ci[0]) and np.isfinite(ci[1]) and (ci[0] > 0.0 or ci[1] < 0.0)


# ---------------------------------------------------------------------------
# 判定（spec §4）— 純関数
# ---------------------------------------------------------------------------
JUDGED_C = (-2.0, -0.5, 0.5, 2.0)


def offset_label(deltas: dict, bands: dict, not_determined: bool) -> str:
    """`deltas[c] = dict(dzmax=(pt, (lo, hi)), dzbar=(pt, (lo, hi)))`（c は判定 4 腕）。

    - OFFSET_IRRELEVANT（H-slope）: 4 つの c すべてで Δzmax の CI ⊂ ±0.3 かつ Δz̄ の CI ⊂ ±0.5
    - OFFSET_SIGNED（H-mag）: Δz̄(+2) の CI < 0 かつ Δz̄(−2) の CI > 0、かつ |Δz̄| が ±0.5 より ±2 で大きい
    - OFFSET_OTHER（H-v など）: いずれかの c で CI が 0 を外すが SIGNED の型でない
    - NOT_DETERMINED: 発散 3/10 seed 超・未定着・判定腕の欠落
    - それ以外 INCONCLUSIVE
    """
    if not_determined or set(deltas) != set(JUDGED_C):
        return "NOT_DETERMINED"
    wz, wb = float(bands["dzmax_irrelevant"]), float(bands["dzbar_irrelevant"])
    if all(ci_within(deltas[c]["dzmax"][1], -wz, wz)
           and ci_within(deltas[c]["dzbar"][1], -wb, wb) for c in JUDGED_C):
        return "OFFSET_IRRELEVANT"
    p2, m2, p05, m05 = (deltas[2.0]["dzbar"], deltas[-2.0]["dzbar"],
                        deltas[0.5]["dzbar"], deltas[-0.5]["dzbar"])
    signed = (np.isfinite(p2[1][1]) and p2[1][1] < 0.0
              and np.isfinite(m2[1][0]) and m2[1][0] > 0.0
              and abs(p2[0]) > abs(p05[0]) and abs(m2[0]) > abs(m05[0]))
    if signed:
        return "OFFSET_SIGNED"
    if any(ci_excludes_zero(deltas[c]["dzmax"][1]) or ci_excludes_zero(deltas[c]["dzbar"][1])
           for c in JUDGED_C):
        return "OFFSET_OTHER"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# 復元場（REPORT・[[命題1-5_上端則結果_0905]] §10.3 の手続き）
# ---------------------------------------------------------------------------
def restore_field(seeds: list[dict], tau: int = 10, from_record: int = 2000,
                  nb: int = 14, min_bin: int = 300) -> dict:
    """条件 z̄(t−2τ)・変位 z̄(t+τ)−z̄(t)（τ = 1 task = 10 記録）・後半 3M・alive = |v| > 0.05。
    分位ビンの中央値曲線の零点 z*（局所線形）と剛性 −dΔz̄/dz̄。"""
    Z = np.stack([z["layer1_zbar"].astype(np.float64) for z in seeds])      # (S, R, h)
    V = np.stack([np.abs(z["layer1_v_unit"].astype(np.float64)) for z in seeds])
    R = Z.shape[1]
    if R <= from_record + 3 * tau:                                          # 短い走（縮約ログ）では出さない
        return dict(zstar=np.nan, stiffness=np.nan, n=0)
    t = np.arange(2 * tau, R - tau)
    t = t[t >= from_record]
    cond = Z[:, t - 2 * tau, :].ravel()
    dz = (Z[:, t + tau, :] - Z[:, t, :]).ravel()
    alive = (V[:, t, :] > LIN_V).ravel()
    m = alive & np.isfinite(cond) & np.isfinite(dz)
    if m.sum() < 10 * min_bin:
        return dict(zstar=np.nan, stiffness=np.nan, n=int(m.sum()))
    q = np.quantile(cond[m], np.linspace(0, 1, nb + 1))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        s = m & (cond >= lo) & (cond < hi)
        if s.sum() < min_bin:
            continue
        xs.append(float(np.median(cond[s]))); ys.append(float(np.median(dz[s])))
    xs, ys = np.asarray(xs), np.asarray(ys)
    zstar, k = np.nan, np.nan
    for i in range(len(xs) - 1):
        if ys[i] > 0 >= ys[i + 1] or ys[i] >= 0 > ys[i + 1]:
            sl = slice(max(0, i - 1), min(len(xs), i + 3))
            kk, bb = np.polyfit(xs[sl], ys[sl], 1)
            zstar, k = -bb / kk, kk
            break
    return dict(zstar=float(zstar), stiffness=float(-k) if np.isfinite(k) else np.nan,
                n=int(m.sum()), curve_x=xs.tolist(), curve_y=ys.tolist())


# ---------------------------------------------------------------------------
# c_out（REPORT・ckpt 5M の net.c）
# ---------------------------------------------------------------------------
def c_out_from_ckpt(ckpts: Path | None, arm: str, step: int = 5_000_000) -> dict:
    if ckpts is None:
        return dict(c_out=np.nan, sum_v=np.nan, n=0)
    f = Path(ckpts) / f"{arm}_step{step}.pt"
    if not f.exists():
        return dict(c_out=np.nan, sum_v=np.nan, n=0)
    import torch
    d = torch.load(f, map_location="cpu", weights_only=False)
    c = d["net"]["c"].detach().cpu().double().reshape(-1).numpy()          # (S,)
    v = d["net"]["v"].detach().cpu().double().numpy()                        # (S, h)
    return dict(c_out=float(np.median(c)), c_out_seeds=c.tolist(),
                sum_v=float(np.median(v.sum(axis=1))), n=int(c.size))


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def _seed_ids(seeds: list[dict]) -> list[int]:
    return [int(z["seed"]) for z in seeds]


def _pair(a: list[dict], b: list[dict]):
    """seed 番号で対応づけ、両腕に（発散せず）居る seed だけ残す。"""
    ia = {s: i for i, s in enumerate(_seed_ids(a))}
    ib = {s: i for i, s in enumerate(_seed_ids(b))}
    common = sorted(set(ia) & set(ib))
    return common, [ia[s] for s in common], [ib[s] for s in common]


def analyze(logs: Path, out: Path, ckpts: Path | None = None) -> dict:
    cfg = load_config(str(CFG))
    A = cfg["analysis"]
    tail = tuple(A["tail_window_tasks"])
    settles = [tuple(w) for w in A["settle_windows_tasks"]]
    bands = A["bands"]
    ref_arm, judged = str(A["reference_arm"]), [str(a) for a in A["judged_arms"]]
    arms = [str(a["name"]) for a in cfg["arms"]]
    offset = {str(a["name"]): float(cfg["activation"][a["activation"]]["offset"])
              for a in cfg["arms"]}
    family = {str(a["name"]): str(a["family"]) for a in cfg["arms"]}
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []

    raw = {a: _load_arm(logs, a) for a in arms}
    missing = [a for a in arms if not raw[a]]
    dropped = {a: [int(z["seed"]) for z in raw[a]
                   if not np.all(np.isfinite(z["layer1_zbar"]))] for a in arms}
    S = {a: [z for z in raw[a] if int(z["seed"]) not in dropped[a]] for a in arms}

    # ---- 腕ごとの要約（REPORT）----
    Q: dict[str, dict] = {}
    for a in arms:
        if not S[a]:
            rows.append(dict(section="arm", arm=a, c=offset[a], family=family[a],
                             label="NOT_RUN", n_seeds=0, dropped=dropped[a]))
            continue
        z = S[a]
        q = dict(zbar=_tail(z, "layer1_zbar", tail), zmax=_tail(z, "layer1_zmax", tail),
                 zmin=_tail(z, "layer1_zmin", tail), mob=_tail(z, "layer1_mob", tail),
                 v=_tail(z, "layer1_v_unit", tail), w=_tail(z, "layer1_w_norm", tail),
                 alive=_alive(z, tail))
        q["half"] = q["zmax"] - q["zbar"]
        al = q["alive"]
        zmax_pt, zmax_ci = boot_median(np.where(al, q["zmax"], np.nan), rng)
        zbar_pt, zbar_ci = boot_median(np.where(al, q["zbar"], np.nan), rng)
        settle_zbar = [float(np.median(_tail(z, "layer1_zbar", w)[_alive(z, w)])) for w in settles]
        settle_zmax = [float(np.median(_tail(z, "layer1_zmax", w)[_alive(z, w)])) for w in settles]
        rf = restore_field(z)
        co = c_out_from_ckpt(ckpts, a)
        Q[a] = q
        rows.append(dict(
            section="arm", arm=a, c=offset[a], family=family[a], label="—",
            n_seeds=len(z), dropped=dropped[a], n_alive=int(al.sum()),
            death_rate=float(1.0 - al.mean()),
            zmax_med=zmax_pt, zmax_ci=zmax_ci, zbar_med=zbar_pt, zbar_ci=zbar_ci,
            half_med=float(np.median(q["half"][al])), mob_med=float(np.median(q["mob"][al])),
            v_abs_med=float(np.median(np.abs(q["v"][al]))), w_med=float(np.median(q["w"][al])),
            lin_rate=float(np.mean(np.abs(q["v"]) < LIN_V)),
            settle_zbar=settle_zbar, settle_zmax=settle_zmax,
            restore_zstar=rf["zstar"], restore_stiffness=rf["stiffness"], restore_n=rf["n"],
            c_out=co["c_out"], sum_v=co["sum_v"], c_times_sum_v=offset[a] * co["sum_v"]))

    # ---- Δ(c) = 腕 c − 腕 c=0（判定 4 腕・ユニット対応・両腕 ALIVE）----
    deltas: dict[float, dict] = {}
    unsettled: dict[str, bool] = {}
    for a in judged:
        if a not in Q or ref_arm not in Q:
            rows.append(dict(section="delta", arm=a, c=offset[a], label="NOT_RUN"))
            continue
        common, ia, ib = _pair(S[a], S[ref_arm])
        n_drop = len(set(dropped[a]) | set(dropped[ref_arm]))
        qa, qb = Q[a], Q[ref_arm]
        both = qa["alive"][ia] & qb["alive"][ib]
        d_zmax = np.where(both, qa["zmax"][ia] - qb["zmax"][ib], np.nan)
        d_zbar = np.where(both, qa["zbar"][ia] - qb["zbar"][ib], np.nan)
        dz_pt, dz_ci = boot_median(d_zmax, rng)
        db_pt, db_ci = boot_median(d_zbar, rng)
        # ALL（除外なし）版 — G3 相当の併記
        dz_all = boot_median(qa["zmax"][ia] - qb["zmax"][ib], rng)
        db_all = boot_median(qa["zbar"][ia] - qb["zbar"][ib], rng)
        # 定着（G2 相当）: settle 3 窓の Δz̄ が単調で、その振れが末尾 CI 幅を超える
        settle = []
        for w in settles:
            za, zb = _tail(S[a], "layer1_zbar", w)[ia], _tail(S[ref_arm], "layer1_zbar", w)[ib]
            m = _alive(S[a], w)[ia] & _alive(S[ref_arm], w)[ib]
            settle.append(float(np.median((za - zb)[m])) if m.any() else np.nan)
        monotone = (settle[0] < settle[1] < settle[2]) or (settle[0] > settle[1] > settle[2])
        drift = float(np.nanmax(settle) - np.nanmin(settle))
        ci_w = float(db_ci[1] - db_ci[0])
        uns = bool(monotone and np.isfinite(drift) and drift > ci_w)
        unsettled[a] = uns
        nd = bool(n_drop > 3)
        deltas[offset[a]] = dict(dzmax=(dz_pt, dz_ci), dzbar=(db_pt, db_ci))
        rows.append(dict(
            section="delta", arm=a, c=offset[a], ref=ref_arm, label="—",
            n_pairs=int(both.sum()), n_seeds=len(common), n_dropped=n_drop,
            dzmax=dz_pt, dzmax_ci=dz_ci, dzbar=db_pt, dzbar_ci=db_ci,
            dzmax_all=dz_all[0], dzmax_all_ci=dz_all[1], dzbar_all=db_all[0], dzbar_all_ci=db_all[1],
            settle_dzbar=settle, unsettled=uns, divergence_over_3=nd,
            within_bands=bool(ci_within(dz_ci, -bands["dzmax_irrelevant"], bands["dzmax_irrelevant"])
                              and ci_within(db_ci, -bands["dzbar_irrelevant"], bands["dzbar_irrelevant"]))))

    not_determined = (any(r.get("divergence_over_3") for r in rows if r.get("section") == "delta")
                      or any(unsettled.values())
                      or len(deltas) < len(JUDGED_C))
    label = offset_label(deltas, bands, not_determined)
    # ALL 版のラベルも併記（G3 相当）
    all_deltas = {}
    for r in rows:
        if r.get("section") == "delta" and "dzmax_all" in r:
            all_deltas[r["c"]] = dict(dzmax=(r["dzmax_all"], r["dzmax_all_ci"]),
                                      dzbar=(r["dzbar_all"], r["dzbar_all_ci"]))
    label_all = offset_label(all_deltas, bands, not_determined)
    rows.append(dict(section="verdict", arm="leaky c in {-2,-0.5,+0.5,+2} vs c=0",
                     judgment="spec §4", role="confirmatory", label=label,
                     label_all_units=label_all, not_determined=not_determined,
                     unsettled=unsettled, reasons=dict(
                         divergence=[r["arm"] for r in rows
                                     if r.get("section") == "delta" and r.get("divergence_over_3")],
                         unsettled=[a for a, u in unsettled.items() if u],
                         missing=[a for a in judged + [ref_arm] if a not in Q])))

    # ---- ELU 2 腕（併記・判定に入れない）: c=+1 − c=−1 の対応差 ----
    if "Eoffp1_1216" in Q and "Eoffm1_1216" in Q:
        common, ia, ib = _pair(S["Eoffp1_1216"], S["Eoffm1_1216"])
        qa, qb = Q["Eoffp1_1216"], Q["Eoffm1_1216"]
        both = qa["alive"][ia] & qb["alive"][ib]
        dz = boot_median(np.where(both, qa["zmax"][ia] - qb["zmax"][ib], np.nan), rng)
        db = boot_median(np.where(both, qa["zbar"][ia] - qb["zbar"][ib], np.nan), rng)
        rows.append(dict(section="elu", arm="Eoffp1_1216 − Eoffm1_1216", role="report",
                         label="REPORT_ONLY", n_pairs=int(both.sum()),
                         dzmax=dz[0], dzmax_ci=dz[1], dzbar=db[0], dzbar_ci=db[1],
                         note="ELU は φ′ も |φ| も非対称で 2 候補が同時に動くので判定に入れない（spec §8-3）。"
                              "c=0 の ELU 腕はこの走に無い（元マシンの Enull とは水準比較しない）"))

    result = dict(label=label, label_all_units=label_all, missing=missing,
                  dropped=dropped, rows=rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    keys = sorted({k for r in rows for k in r})
    with (out / "verdict.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (r.get(k) if isinstance(r.get(k), (str, int, float, type(None)))
                             else json.dumps(r.get(k), default=str, ensure_ascii=False))
                         for k in keys})
    (out / "summary.md").write_text(_markdown(rows, missing, dropped, tail), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------
def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], tuple):
        return f"{x[0]:.{nd}f} [{x[1][0]:.{nd}f}, {x[1][1]:.{nd}f}]"
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _markdown(rows, missing, dropped, tail) -> str:
    L = ["# act_offset_0906 判定まとめ（spec §4）", "",
         f"窓: タスク {tail[0]}–{tail[1]}（タスク終端記録の平均）。ALIVE = `layer1_denom` > 0.25。"
         f"Δ(c) = 腕 c − 腕 c=0 のユニット対応差（両腕 ALIVE）。CI = seed bootstrap {NBOOT} 回・"
         f"`default_rng({RNG_SEED})`。主量 zmax・副量 z̄。", ""]
    if missing:
        L += [f"**欠落腕**: {missing}", ""]
    if any(dropped.values()):
        L += [f"**発散で落とした seed**: { {a: d for a, d in dropped.items() if d} }", ""]
    L += ["## 腕ごとの末尾（REPORT）", "",
          "| 腕 | c | seed | 生存 | zmax 中央値 | z̄ 中央値 | 半幅 | mob | \\|v\\| | ‖w‖ | 線形化率 | settle z̄ (3 窓) | c_out (5M) | Σv | c·Σv | 復元場 z* | 剛性 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("section") != "arm":
            continue
        if r.get("label") == "NOT_RUN":
            L.append(f"| `{r['arm']}` | {r['c']:+g} | 0 | — | NOT_RUN |||||||||||||")
            continue
        L.append(f"| `{r['arm']}` | {r['c']:+g} | {r['n_seeds']} (落 {len(r['dropped'])}) | "
                 f"{r['n_alive']} ({1 - r['death_rate']:.2f}) | {_f((r['zmax_med'], r['zmax_ci']))} | "
                 f"{_f((r['zbar_med'], r['zbar_ci']))} | {r['half_med']:.2f} | {r['mob_med']:.3f} | "
                 f"{r['v_abs_med']:.3f} | {r['w_med']:.2f} | {r['lin_rate']:.2f} | "
                 f"{['%.2f' % v for v in r['settle_zbar']]} | {_f(r['c_out'])} | {_f(r['sum_v'])} | "
                 f"{_f(r['c_times_sum_v'])} | {_f(r['restore_zstar'], 2)} | {_f(r['restore_stiffness'])} |")
    L += ["", "## Δ(c) = 腕 c − 腕 c=0（判定 4 腕・ユニット対応）", "",
          "| 腕 | c | n 対 | 落 seed | Δzmax [CI] | Δz̄ [CI] | Δzmax ALL | Δz̄ ALL | settle Δz̄ (3 窓) | 未定着 | 帯内 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("section") != "delta":
            continue
        if r.get("label") == "NOT_RUN":
            L.append(f"| `{r['arm']}` | {r['c']:+g} | NOT_RUN |||||||||")
            continue
        L.append(f"| `{r['arm']}` | {r['c']:+g} | {r['n_pairs']} | {r['n_dropped']} | "
                 f"{_f((r['dzmax'], r['dzmax_ci']))} | {_f((r['dzbar'], r['dzbar_ci']))} | "
                 f"{_f((r['dzmax_all'], r['dzmax_all_ci']))} | {_f((r['dzbar_all'], r['dzbar_all_ci']))} | "
                 f"{['%.2f' % v for v in r['settle_dzbar']]} | {r['unsettled']} | {r['within_bands']} |")
    for r in rows:
        if r.get("section") == "verdict":
            L += ["", f"## 登録判定: **`{r['label']}`**（ALL ユニット版 `{r['label_all_units']}`）", "",
                  f"- not_determined = {r['not_determined']}（理由 {r['reasons']}）",
                  "- OFFSET_IRRELEVANT: 4 つの c で Δzmax の CI ⊂ ±0.3 かつ Δz̄ の CI ⊂ ±0.5／"
                  "OFFSET_SIGNED: Δz̄(+2) の CI < 0 かつ Δz̄(−2) の CI > 0 で |Δz̄| が ±2 > ±0.5／"
                  "OFFSET_OTHER: いずれかの CI が 0 を外すが SIGNED の型でない／それ以外 INCONCLUSIVE"]
    for r in rows:
        if r.get("section") == "elu":
            L += ["", "## ELU 2 腕（併記・判定に入れない）", "",
                  f"- `{r['arm']}`: Δzmax {_f((r['dzmax'], r['dzmax_ci']))}・Δz̄ {_f((r['dzbar'], r['dzbar_ci']))}"
                  f"・n 対 {r['n_pairs']}。{r['note']}"]
    L += ["", "## 引用上の注意", "",
          "1. 参照は本走の `LRoff0_1216`。元マシンの登録済み腕（`LRnull_1216`・`Enull_1216`）とは水準比較しない（spec §8-1）。",
          "2. 定数 c は c_out と冗長。「c が効いた」は v の学習を経由した効果である可能性を常に併記する（§1 H-v）。",
          "3. 復元場・c_out・線形化率は REPORT（登録外）。"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(ROOT) / "results/act_offset_0906/logs"))
    ap.add_argument("--ckpts", default=str(Path(ROOT) / "results/act_offset_0906/ckpts"))
    ap.add_argument("--out", default=str(Path(ROOT) / "results/act_offset_0906"))
    a = ap.parse_args()
    ck = Path(a.ckpts) if Path(a.ckpts).exists() else None
    r = analyze(Path(a.logs), Path(a.out), ck)
    print(json.dumps({k: r[k] for k in ("label", "label_all_units", "missing", "dropped")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
