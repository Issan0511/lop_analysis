#!/usr/bin/env python3
"""Verdict for mucap_el_0916 (specs/spec_mucap_el_0916.md sections 4-5).

    python3 analysis/mucap_el_0916/verdict.py [--src results/mucap_el_0916/runs]

Reads the 40 shards, applies the registered endpoints and labels, and writes
results/mucap_el_0916/{paired.csv, verdict.csv, secondary.csv, summary.md, provenance.json}.
analyze() is pure (shards in, results out) so that checks.py S8 can run it on synthetic shards.

Endpoints (spec 4.1), per seed: window value = image mean -> arithmetic mean over all 100 units ->
arithmetic mean over the window's task ends; minus the same at the task-1 end.  E1 = mean phi' of
the first layer, E2 = d = zbar/sigma of the first layer (units with sigma = 0 are nan there and are
dropped from E2's unit mean and counted).  The arm effect is the within-seed difference to ref.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "mucap_el_0916"
ARMS = ("ref", "cap_par", "cap_perp", "cap_both")
CAP_ARMS = ARMS[1:]
CAPPED = {"cap_par": ("rows_par",), "cap_perp": ("rows_perp",), "cap_both": ("rows_par", "rows_perp")}
SEEDS = tuple(range(10))
N_TASKS = 150
SPT = 6000                                  # updates per task: the task-end diagnostic step
MAIN_WIN = (51, 150)                        # spec 4.1
FORM_WINS = ((2, 10), (11, 50))             # spec 4.1, reported only
MIN_SEEDS = 8                               # spec 5.2 (A)
CHANCE = 0.1                                # 10 uniform labels: the fit fraction's floor (spec 4.2-7)
EARLY = (2, 5)                              # spec 5.2 IMPAIRED
ABSORB_RUN = 5                              # spec 4.2-1: consecutive task ends with U < 0
MAIN_LEVEL = 0.975                          # two-sided, two main endpoints (Bonferroni)
COMP_LEVEL = 0.95                           # component comparisons, exploratory
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256",
               "task1_end_state_sha256")
ENDPOINTS = {"E1": "gate_mean_l1", "E2": "d_l1"}
LAYER2 = {"E1_l2": "gate_mean_l2", "E2_l2": "d_l2"}
LABELS = {("+", "+"): "BOTH_HELD", ("+", "0"): "RESPONSE_ONLY", ("0", "+"): "POSITION_ONLY",
          ("-", "-"): "CAP_DEEPENS", ("-", "0"): "CAP_DEEPENS_PARTIAL", ("0", "-"): "CAP_DEEPENS_PARTIAL",
          ("+", "-"): "MIXED", ("-", "+"): "MIXED", ("0", "0"): "UNRESOLVED"}


def shard_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


# --------------------------------------------------------------------------
# Student t without scipy: regularized incomplete beta by Lentz's continued fraction
# (checks.py S8 pins t_quantile against the published table)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: int) -> float:
    """Upper quantile for p in (0.5, 1) by bisection to 1e-12."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def paired(x: np.ndarray, level: float) -> dict:
    """Mean, SD and the two-sided `level` t interval of within-seed differences.  SD = 0 gives no
    interval: lo = hi = the exact common difference, flagged degenerate (spec 5.1)."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    out = {"n": n, "mean": float(x.mean()) if n else float("nan"),
           "sd": float(x.std(ddof=1)) if n > 1 else float("nan"), "level": level, "degenerate": False}
    if n < 2 or not np.isfinite(out["sd"]):
        out |= {"lo": float("nan"), "hi": float("nan"), "sign": "0"}
        return out
    if out["sd"] == 0.0:
        out |= {"lo": out["mean"], "hi": out["mean"], "degenerate": True}
    else:
        half = t_quantile(0.5 + level / 2.0, n - 1) * out["sd"] / math.sqrt(n)
        out |= {"lo": out["mean"] - half, "hi": out["mean"] + half}
    out["sign"] = "+" if out["lo"] > 0 else "-" if out["hi"] < 0 else "0"
    return out


# --------------------------------------------------------------------------
# per shard
# --------------------------------------------------------------------------

def task_ends(u: dict) -> dict[int, int]:
    """task -> the row of units.npz holding that task's end point (step == SPT)."""
    t, s = np.asarray(u["task"]), np.asarray(u["step"])
    return {int(t[i]): int(i) for i in np.where(s == SPT)[0]}


def unit_mean(v: np.ndarray) -> tuple[float, int]:
    """Arithmetic mean over all units; nan units (sigma = 0 for d) are dropped and counted."""
    v = np.asarray(v, dtype=np.float64)
    bad = ~np.isfinite(v)
    return (float(v[~bad].mean()) if (~bad).any() else float("nan")), int(bad.sum())


def window_delta(u: dict, key: str, win: tuple[int, int]) -> tuple[float, int]:
    ends = task_ends(u)
    base, nan0 = unit_mean(u[key][ends[1]])
    vals, nans = [], nan0
    for t in range(win[0], win[1] + 1):
        m, k = unit_mean(u[key][ends[t]])
        vals.append(m)
        nans += k
    return float(np.mean(vals)) - base, nans


def fit_early(pt: pd.DataFrame) -> float:
    a = pt.set_index("task")["online_acc"]
    return float(np.mean([(a.loc[t] - CHANCE) / (a.loc[1] - CHANCE) for t in range(EARLY[0], EARLY[1] + 1)]))


def absorption(u: dict) -> dict:
    """Spec 4.2-1 on the task ends.  first_run = the first task starting ABSORB_RUN consecutive ends
    with U < 0; units that never do are right-censored -- counted, never given a time."""
    ends = task_ends(u)
    U = np.stack([np.asarray(u["U_l1"][ends[t]], dtype=np.float64) for t in range(1, N_TASKS + 1)])
    neg = U < 0
    n_units = U.shape[1]
    first_run, first_any, reentry = [], [], []
    for i in range(n_units):
        col = neg[:, i]
        hits = np.where(col)[0]
        first_any.append(int(hits[0]) + 1 if hits.size else None)
        fr = None
        for s0 in range(0, N_TASKS - ABSORB_RUN + 1):
            if col[s0:s0 + ABSORB_RUN].all():
                fr = s0 + 1
                break
        first_run.append(fr)
        reentry.append(int(((col[:-1]) & (~col[1:])).sum()))
    times = [x for x in first_run if x is not None]
    return {"frac_absorbed": len(times) / n_units,
            "median_absorb_task": float(np.median(times)) if times else float("nan"),
            "n_censored": n_units - len(times),
            "frac_neg_at_end": float(neg[-1].mean()),
            "frac_ever_neg": float(np.mean([x is not None for x in first_any])),
            "mean_reentries": float(np.mean(reentry))}


# --------------------------------------------------------------------------
# loading and validity
# --------------------------------------------------------------------------

def load(src: Path) -> tuple[dict, list]:
    shards, missing = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            d = src / shard_name(arm, seed)
            if not (d / "provenance.json").exists():
                missing.append(shard_name(arm, seed))
                continue
            pre = f"s{seed}_"
            with np.load(d / "units.npz") as z:
                units = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            ledger = {}
            if (d / "ledger.npz").exists():
                with np.load(d / "ledger.npz") as z:
                    ledger = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            shards[(arm, seed)] = {"units": units, "per_task": pd.read_csv(d / "per_task.csv"),
                                   "prov": json.loads((d / "provenance.json").read_text()), "ledger": ledger}
    return shards, missing


def seed_validity(shards: dict, seed: int) -> str | None:
    """None if the seed is valid for spec 5.2 (A), else the reason."""
    infos = {}
    for arm in ARMS:
        sh = shards.get((arm, seed))
        if sh is None:
            return f"{arm} missing"
        info = sh["prov"]["per_seed"][str(seed)]
        if info.get("tasks_completed") != N_TASKS or info.get("divergence", {}).get("diverged"):
            return f"{arm} incomplete or diverged"
        ends = task_ends(sh["units"])
        if sorted(ends) != list(range(1, N_TASKS + 1)):
            return f"{arm} task ends missing"
        g = np.stack([sh["units"]["gate_mean_l1"][ends[t]] for t in ends])
        if not np.isfinite(g).all():
            return f"{arm} non-finite response"
        infos[arm] = tuple(info.get(k) for k in STREAM_KEYS)
    if len(set(infos.values())) != 1:
        return "streams or task-1 state differ across arms"
    return None


# --------------------------------------------------------------------------

def analyze(shards: dict) -> dict:
    invalid = {s: r for s in SEEDS if (r := seed_validity(shards, s)) is not None}
    valid = [s for s in SEEDS if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "rows": [], "secondary": [], "applicability": {}}

    # per-seed deltas for every arm, endpoint and window
    D = {}
    for arm in ARMS:
        for s in valid:
            u = shards[(arm, s)]["units"]
            for ep, key in {**ENDPOINTS, **LAYER2, "pplus": "pplus_l1"}.items():
                for win in (MAIN_WIN, *FORM_WINS):
                    D[(arm, s, ep, win)] = window_delta(u, key, win)

    def deltas(arm, ep, win=MAIN_WIN):
        return np.array([D[(arm, s, ep, win)][0] for s in valid])

    def diffs(arm, ep, win=MAIN_WIN):
        return deltas(arm, ep, win) - deltas("ref", ep, win)

    # (A)
    A_ok = len(valid) >= MIN_SEEDS
    # (B) ref alone, 95%
    B = {ep: paired(deltas("ref", ep), 0.95) for ep in ENDPOINTS}
    B_ok = A_ok and all(b["hi"] < 0 for b in B.values())
    # (C) the cap wrote rows in every main-window task
    C = {}
    for arm in CAP_ARMS:
        ok = True
        for s in valid:
            pt = shards[(arm, s)]["per_task"].set_index("task")
            w = pt.loc[MAIN_WIN[0]:MAIN_WIN[1]]
            for col in CAPPED[arm]:
                ok &= bool(len(w) == MAIN_WIN[1] - MAIN_WIN[0] + 1 and (w[col] > 0).all())
        C[arm] = ok
    res["applicability"] = {"A_valid_seeds": len(valid), "A_ok": A_ok,
                            "B": {ep: {k: b[k] for k in ("mean", "lo", "hi")} for ep, b in B.items()},
                            "B_ok": B_ok, "C": C}

    # IMPAIRED flag (reading only)
    impaired = {}
    for arm in CAP_ARMS:
        hits = 0
        for s in valid:
            f_ref = fit_early(shards[("ref", s)]["per_task"])
            f_arm = fit_early(shards[(arm, s)]["per_task"])
            hits += f_arm < f_ref - (1.0 - f_ref)
        impaired[arm] = {"seeds_below": hits, "flag": hits > len(valid) / 2}
    res["impaired"] = impaired

    # labels
    labels = {}
    for arm in CAP_ARMS:
        for level in (MAIN_LEVEL, COMP_LEVEL):
            st = {ep: paired(diffs(arm, ep), level) for ep in ENDPOINTS}
            lab = LABELS[(st["E1"]["sign"], st["E2"]["sign"])]
            if not (A_ok and B_ok and C[arm]):
                lab = "INAPPLICABLE"
            labels[(arm, level)] = lab
            for ep, x in st.items():
                res["rows"].append({"arm": arm, "endpoint": ep, "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                    "level": level, **{k: x[k] for k in ("n", "mean", "sd", "lo", "hi",
                                                                         "sign", "degenerate")},
                                    "label": lab,
                                    "role": "main" if (arm == "cap_both" and level == MAIN_LEVEL) else
                                    ("component" if level == COMP_LEVEL else "report")})
    res["labels"] = {"main": labels[("cap_both", MAIN_LEVEL)],
                     "cap_par": labels[("cap_par", COMP_LEVEL)],
                     "cap_perp": labels[("cap_perp", COMP_LEVEL)],
                     "cap_both_95": labels[("cap_both", COMP_LEVEL)]}
    inter = {ep: paired(diffs("cap_both", ep) - diffs("cap_par", ep) - diffs("cap_perp", ep), COMP_LEVEL)
             for ep in ENDPOINTS}
    res["interaction"] = inter
    res["per_seed"] = {arm: {ep: diffs(arm, ep).tolist() for ep in ENDPOINTS} for arm in CAP_ARMS}
    res["ref_delta"] = {ep: deltas("ref", ep).tolist() for ep in ENDPOINTS}

    # secondary (never a verdict)
    for arm in ARMS:
        for ep in ("pplus", *LAYER2, *ENDPOINTS):
            for win in (MAIN_WIN, *FORM_WINS):
                v = deltas(arm, ep, win)
                row = {"arm": arm, "quantity": f"delta_{ep}", "window": f"{win[0]}-{win[1]}",
                       "mean": float(v.mean()) if v.size else float("nan")}
                if arm != "ref":
                    p = paired(diffs(arm, ep, win), COMP_LEVEL)
                    row |= {"diff_mean": p["mean"], "diff_lo95": p["lo"], "diff_hi95": p["hi"]}
                row["nan_units"] = int(sum(D[(arm, s, ep, win)][1] for s in valid))
                res["secondary"].append(row)
        ab = [absorption(shards[(arm, s)]["units"]) for s in valid]
        for k in ("frac_absorbed", "median_absorb_task", "n_censored", "frac_neg_at_end",
                  "frac_ever_neg", "mean_reentries"):
            vals = np.array([a[k] for a in ab], dtype=np.float64)
            res["secondary"].append({"arm": arm, "quantity": f"absorb_{k}", "window": "1-150",
                                     "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                                     "n_seeds_defined": int(np.isfinite(vals).sum())})
        for t in (1, 10, 50, 100, 150):
            for key in ("U_l1", "K_l1", "sigma_l1", "zbar_l1", "b1", "q_l1", "v_norm_l1", "eps_frac_l1"):
                vals = []
                for s in valid:
                    u = shards[(arm, s)]["units"]
                    if key in u:
                        vals.append(unit_mean(u[key][task_ends(u)[t]])[0])
                if vals:
                    res["secondary"].append({"arm": arm, "quantity": f"level_{key}", "window": f"t{t}",
                                             "mean": float(np.mean(vals))})
        led = [shards[(arm, s)]["ledger"] for s in valid if shards[(arm, s)]["ledger"]]
        if led:
            for comp in ("q", "v2", "m", "wt2"):
                for part in ("adam", "proj"):
                    for term in ("align", "sq"):
                        k = f"{part}_{comp}_{term}"
                        tot = []
                        for L in led:
                            rows_ = (L["task"] >= MAIN_WIN[0]) & (L["task"] <= MAIN_WIN[1])
                            tot.append(float(L[k][rows_].mean(axis=1).sum()))
                        res["secondary"].append({"arm": arm, "quantity": f"ledger_{k}",
                                                 "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                                 "mean": float(np.mean(tot))})
    res["impaired_rows"] = impaired
    return res


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def _f(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def summary_md(res: dict, missing: list, env: dict) -> str:
    L = [f"# {RUN_ID} — 判定（第1層の平均画像方向成分と直交成分の上限・ELU→leaky・150 タスク）", "",
         "> 自動生成: `analysis/mucap_el_0916/verdict.py`。spec: `specs/spec_mucap_el_0916.md`。数値は `verdict.csv`・"
         "`paired.csv`・`secondary.csv` から。", "",
         "## 0. 実行したもの", "",
         f"- 集計時の HEAD: `{env['head']}`・run の commit: {env['run_commits']}",
         f"- shard: {env['n_shards']}/40・欠損 {len(missing)}・有効 seed {len(res['valid_seeds'])}"
         f"（無効: {res['invalid_seeds'] or 'なし'}）", "",
         "## 1. 適用条件（spec §5.2）", ""]
    ap = res["applicability"]
    L += [f"- (A) 有効 seed {ap['A_valid_seeds']} ≥ {MIN_SEEDS}: **{ap['A_ok']}**",
          f"- (B) ref の現象（95%・上端 < 0）: **{ap['B_ok']}** — "
          + "、".join(f"{ep} {_f(b['mean'])} [{_f(b['lo'])}, {_f(b['hi'])}]" for ep, b in ap["B"].items()),
          f"- (C) 上限が主窓の全タスクで書いた: " + "、".join(f"{a} **{v}**" for a, v in ap["C"].items()),
          f"- IMPAIRED（読みのフラグ）: " + "、".join(
              f"{a} {v['seeds_below']}/{len(res['valid_seeds'])} seed → **{v['flag']}**" for a, v in res["impaired"].items()),
          "", "## 2. 判定", "",
          f"| 比較 | 水準 | ラベル |", "|---|---|---|",
          f"| **主: cap_both − ref** | 97.5% | **{res['labels']['main']}** |",
          f"| cap_par − ref | 95% | {res['labels']['cap_par']} |",
          f"| cap_perp − ref | 95% | {res['labels']['cap_perp']} |",
          f"| cap_both − ref（参考） | 95% | {res['labels']['cap_both_95']} |", "",
          "## 3. 腕間差（主窓 task 51–150・seed 内対応差）", "",
          "| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['arm']} | {r['endpoint']} | {r['level']} | {_f(r['mean'])} | {_f(r['sd'])} | "
                 f"[{_f(r['lo'])}, {_f(r['hi'])}]{' (退化)' if r['degenerate'] else ''} | {r['sign']} |")
    L += ["", "交互作用 δ_both − δ_par − δ_perp（95%）: " + "、".join(
        f"{ep} {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}]" for ep, x in res["interaction"].items()), "",
          "### 3.1 seed 別の差", "", "| 腕 | endpoint | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |",
          "|---|---|" + "---|" * len(res["valid_seeds"])]
    for arm, eps in res["per_seed"].items():
        for ep, v in eps.items():
            L.append(f"| {arm} | {ep} | " + " | ".join(_f(x, 3) for x in v) + " |")
    L += ["", "## 4. 副 endpoint（判定に使わない）", "", "| 腕 | 量 | 窓 | 値 | 差 [95%] |", "|---|---|---|---|---|"]
    for r in res["secondary"]:
        diff = (f"{_f(r['diff_mean'])} [{_f(r['diff_lo95'])}, {_f(r['diff_hi95'])}]"
                if "diff_mean" in r else "")
        L.append(f"| {r['arm']} | {r['quantity']} | {r['window']} | {_f(r['mean'])} | {diff} |")
    L += ["", "## 5. 読み方（spec §5.4 のまま）", "",
          "- UNRESOLVED は効果なしではない。単独腕が両方 0 でも「両方必要」とは書かない。",
          "- $U$ の水準差は「上端が残った」の証拠にしない。吸収は §4 の absorb_* だけで読む。",
          "- cap_par は平均画像チャンネルを初期化の大きさに閉じ込める操作として読む。",
          "- IMPAIRED の腕では「可塑性を守った」と書かない。"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    shards, missing = load(src)
    res = analyze(shards)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_commits = sorted({sh["prov"].get("git_hash") for sh in shards.values()})
    env = {"head": head, "run_commits": run_commits, "n_shards": len(shards)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame([{"arm": a, "endpoint": ep, "seed": s, "delta_minus_ref": v}
                  for a, eps in res["per_seed"].items() for ep, vals in eps.items()
                  for s, v in zip(res["valid_seeds"], vals)]).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(res["secondary"]).to_csv(out / "secondary.csv", index=False)
    (out / "summary.md").write_text(summary_md(res, missing, env))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": run_commits, "src": str(src), "n_shards": len(shards), "missing": missing,
        "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "applicability": res["applicability"], "impaired": res["impaired"], "labels": res["labels"],
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "judgement": {"main_window": MAIN_WIN, "main_level": MAIN_LEVEL, "comp_level": COMP_LEVEL,
                      "min_seeds": MIN_SEEDS, "absorb_run": ABSORB_RUN, "early": EARLY, "chance": CHANCE},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid seeds {len(res['valid_seeds'])}  missing {len(missing)}")


if __name__ == "__main__":
    main()
