"""p3_analyze_0902: p3_extend_0902（延長 15M）と valley_clamp_0902（谷埋め）の集計。

事前登録: vault `可塑性喪失/論点/現象3_非ReLU戻り道の対応づけ_0902.md` §7（commit 00c5026）。
窓・U・発症の定義は宿主 gate_dial_0902 / gate_dose_0830 を逐語で使う:
  U^(10)_k = タスク k-9..k の**タスク終端記録**（step % 10000 == 0）の unfit 平均、床 1e-16、
  発症 = U >= 0.05。凍結 = |E_x phi'| < 1e-6（layer1_mob）。沈下 = p_hat == 0。

    .venv/bin/python -m src.p3_analyze_0902 --exp clamp --selftest   # 委託先ログで窓関数を検算
    .venv/bin/python -m src.p3_analyze_0902 --exp extend
    .venv/bin/python -m src.p3_analyze_0902 --exp clamp
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .common import ROOT, load_config
from .gate_dial_0902 import CONFIG, _revival_counts
from .mlp2_phase0b import _window_indices
from .p3_runs_0902 import EXPS, s_ext

PERIOD = 10_000
FROZEN_TOL = 1e-6
BOOT_SEED = 20260907
BOOT_N = 10_000
U_STAR = {"S_b1_1216": 1.2785, "S_b0p3_1216": 4.2616, "G_b0p3_1216": 2.5063,
          "G_b1_1216": 0.7519, "S_b3_1216": 0.4262,
          "Gc_b1_1216": 0.7519, "Sc_b3_1216": 0.4262}
WINDOWS_EXTEND = {"5M": [491, 500], "10M": [991, 1000], "15M": [1491, 1500]}
WINDOWS_CLAMP = {"5M": [491, 500]}
CLAMP_REF = {"Gc_b1_1216": "G_b1_1216", "Sc_b3_1216": "S_b3_1216"}
DIAL_LOGS = Path(ROOT) / "results/gate_dial_0902/logs"


def _cfg() -> dict:
    return load_config(str(CONFIG))


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def window_stats(d: dict, tasks: list[int], floor: float, threshold: float,
                 u_star: float = float("nan")) -> dict:
    idx = _window_indices(d["step"], PERIOD, tasks)
    raw = float(np.asarray(d["unfit"], dtype=np.float64)[idx].mean())
    u = max(raw, floor)
    p = d["layer1_p_hat"][idx]
    zbar = d["layer1_zbar"][idx].astype(np.float64)
    sub = p == 0
    out = dict(n_records=int(len(idx)), raw_u=raw, u=u, log10_u=float(np.log10(u)),
               onset=int(u >= threshold), sub_frac=float(sub.mean()))
    out["depth_q50"] = float(np.median(-zbar[sub])) if sub.any() else float("nan")
    out["depth_q90"] = float(np.percentile(-zbar[sub], 90)) if sub.any() else float("nan")
    if "layer1_mob" in d:
        mob = d["layer1_mob"][idx].astype(np.float64)
        out["frozen_frac"] = float((np.abs(mob) < FROZEN_TOL).mean())
        v = np.abs(d["layer1_v_unit"][idx].astype(np.float64))
        out["absv_sub_med"] = float(np.median(v[sub])) if sub.any() else float("nan")
    else:
        out["frozen_frac"] = float("nan")
        out["absv_sub_med"] = float("nan")
    out["valley_frac"] = (float(((-zbar) >= u_star).mean()) if np.isfinite(u_star)
                          else float("nan"))
    return out


def _boot_median_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    draws = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    meds = np.median(x[draws], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _sign_test(x: np.ndarray) -> dict:
    from math import comb
    neg, pos = int((x < 0).sum()), int((x > 0).sum())
    n = neg + pos
    k = min(neg, pos)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2 if n else float("nan")
    return dict(neg=neg, pos=pos, p_two_sided=min(1.0, p))


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
def analyze_extend(outdir: Path) -> dict:
    cfg = _cfg()
    floor, thr = float(cfg["phase1"]["unfit_floor"]), float(cfg["phase1"]["onset_threshold"])
    E = EXPS["extend"]
    seed_rows, arm_rows, sanity = [], [], {}
    for arm in E["arms"]:
        sanity[arm] = s_ext("extend", arm, outdir)
        per = {w: [] for w in WINDOWS_EXTEND}
        for seed in range(10):
            d = _load(outdir / "logs" / f"{arm}_seed{seed}.npz")
            for w, tasks in WINDOWS_EXTEND.items():
                st = window_stats(d, tasks, floor, thr, U_STAR.get(arm, float("nan")))
                st.update(arm=arm, seed=seed, window=w)
                per[w].append(st)
                seed_rows.append(st)
        row = dict(arm=arm, s_ext_pass=int(sanity[arm]["pass_"]))
        for w in WINDOWS_EXTEND:
            rows = per[w]
            row[f"n_onset_{w}"] = int(sum(r["onset"] for r in rows))
            for key in ("log10_u", "sub_frac", "frozen_frac", "depth_q50", "valley_frac", "absv_sub_med"):
                row[f"{key}_{w}"] = float(np.nanmedian([r[key] for r in rows]))
        d5 = np.array([r["depth_q50"] for r in per["5M"]])
        d15 = np.array([r["depth_q50"] for r in per["15M"]])
        row["depth_growth_5M_to_15M"] = float(np.nanmedian(d15 - d5))
        arm_rows.append(row)
    A = {r["arm"]: r for r in arm_rows}
    # 事前登録 §7.1 の予測行
    preds = [
        dict(arm="E_1216", item="n_onset_15M == 0", value=A["E_1216"]["n_onset_15M"],
             hit=A["E_1216"]["n_onset_15M"] == 0),
        dict(arm="E_1216", item="sub_frac_15M in [0.30, 0.50]", value=A["E_1216"]["sub_frac_15M"],
             hit=0.30 <= A["E_1216"]["sub_frac_15M"] <= 0.50),
        dict(arm="E_1216", item="depth_q50_15M in 8.8 +/- 1.5", value=A["E_1216"]["depth_q50_15M"],
             hit=abs(A["E_1216"]["depth_q50_15M"] - 8.8) <= 1.5),
        dict(arm="LR_1216", item="n_onset_15M == 0", value=A["LR_1216"]["n_onset_15M"],
             hit=A["LR_1216"]["n_onset_15M"] == 0),
        dict(arm="LR_1216", item="sub_frac_15M in [0.55, 0.80]", value=A["LR_1216"]["sub_frac_15M"],
             hit=0.55 <= A["LR_1216"]["sub_frac_15M"] <= 0.80),
        dict(arm="LR_1216", item="depth_q50_15M in 4.2 +/- 1.5", value=A["LR_1216"]["depth_q50_15M"],
             hit=abs(A["LR_1216"]["depth_q50_15M"] - 4.2) <= 1.5),
        dict(arm="S_b1_1216", item="frozen_frac_15M >= 0.8", value=A["S_b1_1216"]["frozen_frac_15M"],
             hit=A["S_b1_1216"]["frozen_frac_15M"] >= 0.8),
        dict(arm="S_b1_1216", item="log10_u_15M >= -0.3", value=A["S_b1_1216"]["log10_u_15M"],
             hit=A["S_b1_1216"]["log10_u_15M"] >= -0.3),
        dict(arm="G_b0p3_1216", item="frozen_frac_15M >= 0.8", value=A["G_b0p3_1216"]["frozen_frac_15M"],
             hit=A["G_b0p3_1216"]["frozen_frac_15M"] >= 0.8),
        dict(arm="G_b0p3_1216", item="log10_u_15M >= -0.3", value=A["G_b0p3_1216"]["log10_u_15M"],
             hit=A["G_b0p3_1216"]["log10_u_15M"] >= -0.3),
        dict(arm="S_b0p3_1216", item="frozen_frac_15M <= 0.1", value=A["S_b0p3_1216"]["frozen_frac_15M"],
             hit=A["S_b0p3_1216"]["frozen_frac_15M"] <= 0.1),
        dict(arm="S_b0p3_1216", item="depth_growth_5M_to_15M <= 5", value=A["S_b0p3_1216"]["depth_growth_5M_to_15M"],
             hit=A["S_b0p3_1216"]["depth_growth_5M_to_15M"] <= 5),
    ]
    ctrl_zero = A["E_1216"]["n_onset_15M"] == 0 and A["LR_1216"]["n_onset_15M"] == 0
    ctrl_present = A["E_1216"]["n_onset_15M"] >= 5 or A["LR_1216"]["n_onset_15M"] >= 5
    flight_ok = A["S_b1_1216"]["frozen_frac_15M"] >= 0.8 and A["G_b0p3_1216"]["frozen_frac_15M"] >= 0.8
    flight_stall = A["S_b1_1216"]["frozen_frac_15M"] < 0.5 or A["G_b0p3_1216"]["frozen_frac_15M"] < 0.5
    hits = []
    if ctrl_present:
        hits.append("SOFT_ARMS_DRIFT_TO_RELU")
    if flight_stall:
        hits.append("FLIGHT_STALLS")
    if ctrl_zero and flight_ok:
        hits.append("EQUILIBRIUM_VS_FLIGHT_CONFIRMED")
    verdict = hits[0] if hits else "MIXED"
    return dict(verdict=verdict, hits=hits, arms=arm_rows, seeds=seed_rows, preds=preds,
                sanity={a: dict(pass_=s["pass_"]) for a, s in sanity.items()})


def analyze_clamp(outdir: Path) -> dict:
    cfg = _cfg()
    floor, thr = float(cfg["phase1"]["unfit_floor"]), float(cfg["phase1"]["onset_threshold"])
    rng = np.random.default_rng(BOOT_SEED)
    seed_rows, arm_rows, contrasts = [], [], []
    for arm, ref in CLAMP_REF.items():
        us, ur, dc, dr, rc, rr, fc, fr = [], [], [], [], [], [], [], []
        for seed in range(10):
            pc, pr = outdir / "logs" / f"{arm}_seed{seed}.npz", DIAL_LOGS / f"{ref}_seed{seed}.npz"
            sc = window_stats(_load(pc), WINDOWS_CLAMP["5M"], floor, thr, U_STAR[arm])
            sr = window_stats(_load(pr), WINDOWS_CLAMP["5M"], floor, thr, U_STAR[ref])
            revc, revr = _revival_counts(pc, U_STAR[arm]), _revival_counts(pr, U_STAR[ref])
            sc.update(arm=arm, seed=seed, role="clamp", revive_across=revc["events_across_boundary"],
                      revive_within=revc["events_within_task"])
            sr.update(arm=ref, seed=seed, role="reference", revive_across=revr["events_across_boundary"],
                      revive_within=revr["events_within_task"])
            seed_rows += [sc, sr]
            us.append(sc["log10_u"]); ur.append(sr["log10_u"])
            dc.append(sc["depth_q50"]); dr.append(sr["depth_q50"])
            rc.append(revc["events_across_boundary"]); rr.append(revr["events_across_boundary"])
            fc.append(sc["frozen_frac"]); fr.append(sr["frozen_frac"])
        delta = np.array(us) - np.array(ur)
        med, lo, hi = _boot_median_ci(delta, rng)
        sign = _sign_test(delta)
        n_on_c, n_on_r = int(sum(u >= np.log10(thr) for u in us)), int(sum(u >= np.log10(thr) for u in ur))
        row = dict(arm=arm, reference=ref, n_onset_clamp=n_on_c, n_onset_ref=n_on_r,
                   log10_u_clamp=float(np.median(us)), log10_u_ref=float(np.median(ur)),
                   delta_median=med, delta_ci_lo=lo, delta_ci_hi=hi,
                   sign_neg=sign["neg"], sign_pos=sign["pos"], sign_p=sign["p_two_sided"],
                   depth_q50_clamp=float(np.nanmedian(dc)), depth_q50_ref=float(np.nanmedian(dr)),
                   frozen_clamp=float(np.nanmedian(fc)), frozen_ref=float(np.nanmedian(fr)),
                   revive_across_clamp=float(np.median(rc)), revive_across_ref=float(np.median(rr)))
        level_ok = med <= -0.3 and sign["neg"] >= 8
        depth_ok = row["depth_q50_clamp"] <= 8.0
        equiv = lo >= -0.15 and hi <= 0.15
        row["family_label"] = ("FLIGHT_CARRIES_OVERSHOOT" if (level_ok and depth_ok)
                               else "OVERSHOOT_NOT_FLIGHT" if equiv else "PARTIAL")
        row["pred_onset_10_of_10"] = int(n_on_c == 10)
        row["pred_level"] = int(level_ok)
        row["pred_depth_le_8"] = int(depth_ok)
        row["pred_churn_positive"] = int(np.median(rc) > 0)
        arm_rows.append(row)
        contrasts.append(dict(arm=arm, delta=delta.tolist()))
    labels = {r["family_label"] for r in arm_rows}
    verdict = labels.pop() if len(labels) == 1 else "PARTIAL"
    return dict(verdict=verdict, arms=arm_rows, seeds=seed_rows, contrasts=contrasts)


def selftest_clamp() -> None:
    """委託先（gate_dial_0902）の committed 値と窓関数の一致を確認する。"""
    cfg = _cfg()
    floor, thr = float(cfg["phase1"]["unfit_floor"]), float(cfg["phase1"]["onset_threshold"])
    table = {}
    with open(Path(ROOT) / "results/gate_dial_0902/dial_table.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[r["arm"]] = float(r["median_log10_U_5m"])
    for ref in ("G_b1_1216", "S_b3_1216", "S_b1_1216", "G_b0p3_1216", "S_b0p3_1216"):
        vals = [window_stats(_load(DIAL_LOGS / f"{ref}_seed{s}.npz"), [491, 500], floor, thr,
                             U_STAR[ref])["log10_u"] for s in range(10)]
        med = float(np.median(vals))
        print(f"selftest {ref}: recomputed median log10U_5m={med:.6f} committed={table[ref]:.6f} "
              f"{'OK' if abs(med - table[ref]) < 1e-9 else 'MISMATCH'}")


def write_outputs(exp: str, outdir: Path, res: dict) -> None:
    _write_csv(outdir / "verdict.csv", [dict(verdict=res["verdict"], **r) for r in res["arms"]])
    _write_csv(outdir / "seed_values.csv", res["seeds"])
    if "preds" in res:
        _write_csv(outdir / "predictions.csv", res["preds"])
    prov = dict(exp=exp, git_head=_git_head(), verdict=res["verdict"],
                config_sha256=_sha(CONFIG), window_definition="task-end records, U^(10)",
                logs={p.name: _sha(p) for p in sorted((outdir / "logs").glob("*.npz"))},
                arm_status={p.name: json.loads(p.read_text()) for p in sorted((outdir / "arm_status").glob("*_done.json"))},
                sanity=res.get("sanity"))
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {exp} summary", "", f"**verdict: {res['verdict']}**", ""]
    if res.get("hits"):
        lines.append(f"hit labels (in precedence order): {res['hits']}")
    keys = list(res["arms"][0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for r in res["arms"]:
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    if "preds" in res:
        lines += ["", "## 事前登録の予測（vault 00c5026 §7.1）", "", "| arm | item | value | hit |", "|---|---|---|---|"]
        for p in res["preds"]:
            lines.append(f"| {p['arm']} | {p['item']} | {p['value']:.4g} | {'✓' if p['hit'] else '✗'} |")
    lines += ["", "引用上の注意: 用量 12.16・1 層・幅 100・seed 0–9。0/10 は片側 95% 上限 0.2589 の強さ。"
              "延長走の対照 2 腕は gate_dose_0830 と同一の init・入力列（S-ext で bit 一致を検査）。"]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["extend", "clamp"])
    p.add_argument("--outdir", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest_clamp()
        return
    outdir = Path(a.outdir).resolve() if a.outdir else Path(ROOT) / EXPS[a.exp]["outdir"]
    res = analyze_extend(outdir) if a.exp == "extend" else analyze_clamp(outdir)
    write_outputs(a.exp, outdir, res)


if __name__ == "__main__":
    main()
