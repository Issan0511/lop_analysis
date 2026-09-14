#!/usr/bin/env python3
"""Reuse check, aggregation and registered labels for l2split_rlmnist_0914 (spec sections 2.3, 4, 5).

    python3 analysis/l2split_rlmnist_0914/verdict.py --reuse-check        # after the 4 verification runs
    python3 analysis/l2split_rlmnist_0914/verdict.py                      # after all runs
    python3 analysis/l2split_rlmnist_0914/verdict.py --src <smoke>/runs --reuse-src <smoke>/reuse_src \
        --reuse-json <smoke>/reuse.json --out <smoke>                    # smoke: always pass --src explicitly

Shards are enumerated by name (act x arm x seed), never globbed.  ref / l2 come from --reuse-src when
reuse.json says REUSE_OK and from --src when it says REUSE_MISMATCH (spec 2.3).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "l2split_rlmnist_0914"
ACTS = ("R", "LR")
NEW_ARMS = ("l2wt", "l2rest")
REUSED_ARMS = ("ref", "l2")
ARMS8 = [(a, m) for a in ACTS for m in ("ref", "l2", "l2wt", "l2rest")]
SEEDS = tuple(range(10))
VERIFY = ("R_ref_s0", "R_l2_s0", "LR_ref_s0", "LR_l2_s0")
N_TASKS = 50
WIN = (31, 50)                 # G = A(1) - A(WIN)
BAND_LO, BAND_HI = 0.1, 0.9    # spec 5.1
GUARD = 2.0                    # kappa >= 2
TAX_PT = 1.0                   # side-effect guard (i): A(1)_arm >= A(1)_l2 - 1.0 pt
DEAD_TOL = 0.1                 # side-effect guard (ii), R only: dead1(t=1) <= l2's + 0.1
MIN_SEEDS = 8
N_BOOT = 10_000
RNG_SEED = 20260914
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")
TRAJ_TASKS = (1, 2, 3, 5, 10, 20, 30, 40, 50)


def shard_name(act: str, arm: str, seed: int) -> str:
    return f"{act}_{arm}_s{seed}"


# --------------------------------------------------------------------------
# reuse (spec 2.3)
# --------------------------------------------------------------------------

def compare_reuse(new_runs: Path, old_runs: Path) -> dict:
    """The 4 full-length verification shards vs the committed wcap shards: final state sha256, task-1-end
    state sha256 and the online_acc column (as written, compared as text) must all match."""
    items = {}
    for name in VERIFY:
        n, o = new_runs / name, old_runs / name
        if not (n / "provenance.json").exists() or not (o / "provenance.json").exists():
            items[name] = {"present": False}
            continue
        pn = json.loads((n / "provenance.json").read_text())["runs"]["0"]
        po = json.loads((o / "provenance.json").read_text())["runs"]["0"]
        an = pd.read_csv(n / "per_task.csv", dtype=str).online_acc.tolist()
        ao = pd.read_csv(o / "per_task.csv", dtype=str).online_acc.tolist()
        items[name] = {"present": True,
                       "final_state_sha256": pn["final_state_sha256"] == po["final_state_sha256"],
                       "task1_end_state_sha256": pn.get("task1_end_state_sha256") == po.get("task1_end_state_sha256"),
                       "online_acc": an == ao}
    ok = all(v.get("present") and v["final_state_sha256"] and v["task1_end_state_sha256"] and v["online_acc"]
             for v in items.values())
    return {"status": "REUSE_OK" if ok else "REUSE_MISMATCH", "items": items}


# --------------------------------------------------------------------------
# loading and validity
# --------------------------------------------------------------------------

def load(src: Path, reuse_src: Path, reuse_status: str | None):
    pts, lms, provs, missing, origin = [], [], {}, [], {}
    for act, arm in ARMS8:
        base = reuse_src if (arm in REUSED_ARMS and reuse_status == "REUSE_OK") else src
        for seed in SEEDS:
            d = base / shard_name(act, arm, seed)
            if reuse_status is None and arm in REUSED_ARMS or not (d / "provenance.json").exists():
                missing.append(shard_name(act, arm, seed))
                continue
            provs[(act, arm, seed)] = json.loads((d / "provenance.json").read_text())
            origin[shard_name(act, arm, seed)] = str(d)
            pt = pd.read_csv(d / "per_task.csv")
            pt["arm"] = arm                                   # the wcap shards carry the same arm names
            pts.append(pt)
            if (d / "layer_metrics.csv").exists():
                lm = pd.read_csv(d / "layer_metrics.csv")
                lm["arm"] = arm
                lms.append(lm)
    pt = pd.concat(pts, ignore_index=True) if pts else pd.DataFrame()
    lm = pd.concat(lms, ignore_index=True) if lms else pd.DataFrame()
    return pt, lm, provs, missing, origin


def valid_seeds(pt: pd.DataFrame, provs: dict) -> tuple[list[int], dict]:
    ok, why = [], {}
    for seed in SEEDS:
        keys = [(a, m, seed) for a, m in ARMS8]
        if any(k not in provs for k in keys):
            why[seed] = "missing shard"
            continue
        runs = [provs[k]["runs"][str(seed)] for k in keys]
        if any(r["divergence"]["diverged"] or r["tasks_completed"] != N_TASKS for r in runs):
            why[seed] = "diverged or incomplete"
            continue
        n_rows = [len(pt[(pt.act == a) & (pt.arm == m) & (pt.seed == seed) & pt.online_acc.notna()]) for a, m, _ in keys]
        if any(n != N_TASKS for n in n_rows):
            why[seed] = f"per_task rows {n_rows}"
            continue
        if len({json.dumps({f: r[f] for f in STREAM_KEYS}, sort_keys=True) for r in runs}) != 1:
            why[seed] = "stream sha256 differ across arms"
            continue
        ok.append(seed)
    return ok, why


# --------------------------------------------------------------------------
# endpoints and labels
# --------------------------------------------------------------------------

def tensor_win(lm: pd.DataFrame, act: str, arm: str, seed: int, tensor: str) -> float:
    q = lm[(lm.act == act) & (lm.arm == arm) & (lm.seed == seed) & (lm.tensor == tensor)
           & (lm.task >= WIN[0]) & (lm.task <= WIN[1])]
    return float(q.norm.mean()) if len(q) else float("nan")


def per_seed(pt: pd.DataFrame, lm: pd.DataFrame, act: str, arm: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        g = pt[(pt.act == act) & (pt.arm == arm) & (pt.seed == s)].set_index("task").sort_index()
        A = g.online_acc
        win = A.loc[WIN[0]:WIN[1]].mean()
        h = A.loc[11:50]
        rows.append(dict(
            seed=s, A1=A.loc[1], Awin=win, G=A.loc[1] - win,
            P=A.loc[11:20].mean() - A.loc[41:50].mean(), D=A.loc[2:6].mean() - win,
            slope=float(np.polyfit(h.index.to_numpy(float), h.to_numpy(float), 1)[0]) * 10,
            memo_win=g.memo_acc.loc[WIN[0]:WIN[1]].mean(), dead1_t1=g.dead_frac_l1.loc[1],
            wt1_t1=g.wt_med_l1.loc[1], wt1_t50=g.wt_med_l1.loc[N_TASKS], wt1_win=g.wt_med_l1.loc[WIN[0]:WIN[1]].mean(),
            w3_win=tensor_win(lm, act, arm, s, "W3"), b1_win=tensor_win(lm, act, arm, s, "b1")))
    return pd.DataFrame(rows).set_index("seed")


def boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(RNG_SEED).integers(0, n, (N_BOOT, n))


def ci(boot: np.ndarray) -> dict:
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    lo975, hi975 = np.percentile(boot, [1.25, 98.75])
    return dict(lo95=float(lo95), hi95=float(hi95), lo975=float(lo975), hi975=float(hi975))


def removal(Gm: np.ndarray, Gref: np.ndarray, idx: np.ndarray) -> dict:
    """rho = 1 - mean(G_m) / mean(G_ref), seed-paired bootstrap (same index for numerator and denominator)."""
    est = 1.0 - Gm.mean() / Gref.mean()
    boot = 1.0 - Gm[idx].mean(axis=1) / Gref[idx].mean(axis=1)
    return dict(est=float(est), **ci(boot), per_seed=[float(v) for v in 1.0 - Gm / Gref])


def mean_ci(x: np.ndarray, idx: np.ndarray) -> dict:
    return dict(est=float(x.mean()), **ci(x[idx].mean(axis=1)))


def band_label(st: dict, lvl: str) -> str:
    lo, hi = st[f"lo{lvl}"], st[f"hi{lvl}"]
    if lo >= BAND_HI:
        return "SUFFICIENT"
    if hi <= BAND_LO:
        return "NOT_LEVER"
    if lo > BAND_LO and hi < BAND_HI:
        return "PARTIAL"
    return "UNRESOLVED"


def pattern(wt: str, rest: str, rho_l2: float) -> str:
    wt, rest = wt.split("+")[0], rest.split("+")[0]
    if wt == "SUFFICIENT" and rest == "NOT_LEVER":
        return "WT_SUFFICES"
    if wt == "NOT_LEVER" and rest == "SUFFICIENT":
        return "REST_SUFFICES"
    if wt == "SUFFICIENT" and rest == "SUFFICIENT":
        return "EITHER_SUFFICES"
    if wt == "NOT_LEVER" and rest == "NOT_LEVER" and rho_l2 >= BAND_HI:
        return "BOTH_NEEDED"
    return "MIXED"


def analyze(pt: pd.DataFrame, lm: pd.DataFrame, provs: dict) -> dict:
    seeds, why = valid_seeds(pt, provs) if len(pt) else ([], {s: "no data" for s in SEEDS})
    res = {"valid_seeds": seeds, "invalid_seeds": {str(k): v for k, v in why.items()}, "n_valid": len(seeds),
           "labels": {}, "patterns": {}, "stats": {}, "tables": {}, "guards": {}}
    if len(seeds) < MIN_SEEDS:
        for act in ACTS:
            for arm in NEW_ARMS:
                res["labels"][f"{act}_{arm}"] = {lv: ("INCOMPLETE", f"{len(seeds)} valid seeds < {MIN_SEEDS}")
                                                 for lv in ("95", "97.5")}
            res["patterns"][act] = {"95": "MIXED", "97.5": "MIXED"}
        return res
    idx = boot_idx(len(seeds))
    T = {f"{a}_{m}": per_seed(pt, lm, a, m, seeds) for a, m in ARMS8}
    res["tables"] = T
    st = res["stats"]
    for act in ACTS:
        ref, l2 = T[f"{act}_ref"], T[f"{act}_l2"]
        for arm in ("l2", "l2wt", "l2rest"):
            q = T[f"{act}_{arm}"]
            st[f"rho|{act}_{arm}"] = removal(q.G.to_numpy(), ref.G.to_numpy(), idx)
        for arm in ("ref", "l2", "l2wt", "l2rest"):
            q = T[f"{act}_{arm}"]
            for col in ("G", "P", "D", "slope", "Awin", "A1"):
                st[f"{col}|{act}_{arm}"] = mean_ci(q[col].to_numpy(), idx)
        kw = float((ref.wt1_win / T[f"{act}_l2wt"].wt1_win).median())
        kr = float(np.maximum(ref.w3_win / T[f"{act}_l2rest"].w3_win, ref.b1_win / T[f"{act}_l2rest"].b1_win).median())
        res["guards"][f"{act}_l2wt"] = {"kappa": kw}
        res["guards"][f"{act}_l2rest"] = {"kappa": kr}
        for arm in NEW_ARMS:
            q = T[f"{act}_{arm}"]
            tax = float(q.A1.mean() - l2.A1.mean())
            dead = float(q.dead1_t1.median() - l2.dead1_t1.median())
            harmful = tax < -TAX_PT / 100 or (act == "R" and dead > DEAD_TOL)
            res["guards"][f"{act}_{arm}"].update(a1_minus_l2_pt=100 * tax, dead1_t1_minus_l2=dead, harmful=bool(harmful))
    for lvl in ("95", "97.5"):
        k = lvl.replace(".", "")
        for act in ACTS:
            for arm in NEW_ARMS:
                g = res["guards"][f"{act}_{arm}"]
                r = st[f"rho|{act}_{arm}"]
                if g["kappa"] < GUARD:
                    lab = "WEAK_MANIPULATION"
                else:
                    lab = band_label(r, k)
                if g["harmful"]:
                    lab += "+HARMFUL"
                res["labels"].setdefault(f"{act}_{arm}", {})[lvl] = (
                    lab, f"rho = {r['est']:.4f} [{r[f'lo{k}']:.4f}, {r[f'hi{k}']:.4f}]; kappa = {g['kappa']:.2f}")
            res["patterns"].setdefault(act, {})[lvl] = pattern(res["labels"][f"{act}_l2wt"][lvl][0],
                                                                res["labels"][f"{act}_l2rest"][lvl][0],
                                                                st[f"rho|{act}_l2"]["est"])
    return res


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def verdict_rows(res: dict) -> list[dict]:
    rows = []
    for q, v in res["labels"].items():
        for lvl, (lab, reason) in v.items():
            rows.append({"kind": "label", "question": q, "level": lvl, "label": lab, "reason": reason,
                         "n_valid_seeds": res["n_valid"]})
    for act, v in res["patterns"].items():
        for lvl, p in v.items():
            rows.append({"kind": "pattern", "question": act, "level": lvl, "label": p, "n_valid_seeds": res["n_valid"]})
    for key, g in res["guards"].items():
        for gk, gv in g.items():
            rows.append({"kind": "guard", "question": f"{key}|{gk}", "est": gv, "n_valid_seeds": res["n_valid"]})
    for key, s in res["stats"].items():
        rows.append({"kind": "stat", "question": key, "est": s["est"],
                     **{k: s.get(k) for k in ("lo95", "hi95", "lo975", "hi975")}, "n_valid_seeds": res["n_valid"]})
    return rows


def pct(s: dict, sign: bool = True) -> str:
    f = "{:+.2f}" if sign else "{:.2f}"
    return f"{f.format(100 * s['est'])} [{f.format(100 * s['lo95'])}, {f.format(100 * s['hi95'])}]"


def summary_md(res: dict, pt: pd.DataFrame, lm: pd.DataFrame, provs: dict, missing: list, reuse: dict | None,
               checks: dict | None, head: str) -> str:
    L = [f"# {RUN_ID} — 判定（Random Label MNIST・L2 を中心化 W̃ の減衰と残りの減衰に割る 2×2）", "",
         "> 自動生成: `analysis/l2split_rlmnist_0914/verdict.py`。spec: `specs/spec_l2split_rlmnist_0914.md`。数値はこのファイルと `verdict.csv` から転記する。", "",
         "## 0. 実行したもの", ""]
    new = {k: v for k, v in provs.items() if v.get("run_id") == RUN_ID}
    walls = [v["wall_clock_s"] for v in new.values()]
    L += [f"- 新規 run の commit: {', '.join(sorted('`' + v['git_hash'] + '`' for v in new.values())[:1])}"
          f"（種類 {len({v['git_hash'] for v in new.values()})}・未 commit 変更: {sorted({str(v.get('git_dirty_code')) for v in new.values()})}）",
          f"- 集計時の HEAD: `{head}`",
          f"- 再利用: **{reuse['status'] if reuse else '（reuse.json なし）'}**"
          + (f"（照合 {sum(1 for v in reuse['items'].values() if v.get('present') and all(v[x] for x in ('final_state_sha256', 'task1_end_state_sha256', 'online_acc')))}/4 一致）" if reuse else ""),
          f"- shard: {len(provs)}/80 を読んだ（新規 {len(new)}・再利用 {len(provs) - len(new)}）・欠損 {len(missing)}・有効 seed {res['n_valid']}（無効: {res['invalid_seeds'] or 'なし'}）",
          (f"- 新規 run の壁時計 中央値 {np.median(walls) / 60:.1f} 分（最大 {max(walls) / 60:.1f} 分）" if walls else "- 新規 run: なし"),
          f"- 検査: `checks.json` all_pass = **{checks.get('all_pass') if checks else '（なし）'}**", ""]
    L += ["## 1. 結論（spec §5.2 を上から適用）", "", "| act | arm | ラベル（95%） | 理由 | Bonferroni 版（97.5%） |", "|---|---|---|---|---|"]
    for act in ACTS:
        for arm in NEW_ARMS:
            v = res["labels"][f"{act}_{arm}"]
            L.append(f"| {act} | {arm} | **{v['95'][0]}** | {v['95'][1]} | {v['97.5'][0]} |")
    L += ["", "| act | 2×2 の型（95%） | Bonferroni 版 |", "|---|---|---|"]
    for act in ACTS:
        L.append(f"| **{act}** | **{res['patterns'][act]['95']}** | {res['patterns'][act]['97.5']} |")
    L += ["", "主の問いは R の型。限定: Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。G は画像固定による正の転移を含む正味。", ""]
    if res["n_valid"] < MIN_SEEDS:
        return "\n".join(L) + "\n"
    st, T = res["stats"], res["tables"]
    L += ["## 2. 時間劣化と除去率（pt・[ ] は seed 対応 bootstrap 95% CI）", "",
          "| 腕 | A(1) | A(31–50) | G = A(1)−A(31–50) | ρ = 1 − G/G_ref | P = A(11–20)−A(41–50) | D = A(2–6)−A(31–50) | 傾き 11–50 /10 タスク |",
          "|---|---|---|---|---|---|---|---|"]
    for act, arm in ARMS8:
        k = f"{act}_{arm}"
        rho = st.get(f"rho|{k}")
        rs = f"{rho['est']:.4f} [{rho['lo95']:.4f}, {rho['hi95']:.4f}]" if rho else "—"
        L.append(f"| {k} | {pct(st[f'A1|{k}'], False)} | {pct(st[f'Awin|{k}'], False)} | {pct(st[f'G|{k}'])} | {rs} | "
                 f"{pct(st[f'P|{k}'])} | {pct(st[f'D|{k}'])} | {pct(st[f'slope|{k}'])} |")
    L += ["", "### 2.1 ガード", "", "| 腕 | κ（≥ 2） | A(1) − A_l2(1)（pt・≥ −1.0） | dead1(t1) − l2 の中央値（R・≤ 0.1） | HARMFUL |", "|---|---|---|---|---|"]
    for act in ACTS:
        for arm in NEW_ARMS:
            g = res["guards"][f"{act}_{arm}"]
            L.append(f"| {act}_{arm} | {g['kappa']:.2f} | {g['a1_minus_l2_pt']:+.2f} | {g['dead1_t1_minus_l2']:+.3f} | {g['harmful']} |")
    L += ["", "### 2.2 ρ の seed 別の値", "", "| 腕 | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |",
          "|---|" + "---|" * res["n_valid"]]
    for act in ACTS:
        for arm in ("l2", "l2wt", "l2rest"):
            L.append(f"| {act}_{arm} | " + " | ".join(f"{v:.3f}" for v in st[f"rho|{act}_{arm}"]["per_seed"]) + " |")
    ptv = pt[pt.seed.isin(res["valid_seeds"])]
    L += ["", "## 3. 軌道（online_acc の seed 平均・pt）", "", "| 腕 | " + " | ".join(f"t{t}" for t in TRAJ_TASKS) + " |",
          "|---|" + "---|" * len(TRAJ_TASKS)]
    for act, arm in ARMS8:
        q = ptv[(ptv.act == act) & (ptv.arm == arm)].groupby("task").online_acc.mean()
        L.append(f"| {act}_{arm} | " + " | ".join(f"{100 * q.loc[t]:.2f}" for t in TRAJ_TASKS) + " |")
    L += ["", "## 4. 状態（seed 中央値）", "",
          "| 腕 | median‖W̃1_i‖ t1 → t50 | 窓 | ‖W3‖ 窓 | ‖b1‖ 窓 | mob1 t1 → t50 | mob2 t1 → t50 | z̄1 t1 → t50 | z̄2 t1 → t50 | dead1 t1 → t50 | median\\|m_i\\| t50 | memo 窓 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for act, arm in ARMS8:
        q = T[f"{act}_{arm}"].median()
        g = ptv[(ptv.act == act) & (ptv.arm == arm)]
        g1, g50 = g[g.task == 1].median(numeric_only=True), g[g.task == N_TASKS].median(numeric_only=True)
        L.append(f"| {act}_{arm} | {q.wt1_t1:.3f} → {q.wt1_t50:.3f} | {q.wt1_win:.3f} | {q.w3_win:.2f} | {q.b1_win:.2f} | "
                 f"{g1.mob_l1:.3f} → {g50.mob_l1:.3f} | {g1.mob_l2:.3f} → {g50.mob_l2:.3f} | {g1.zbar_l1:.2f} → {g50.zbar_l1:.2f} | "
                 f"{g1.zbar_l2:.2f} → {g50.zbar_l2:.2f} | {g1.dead_frac_l1:.3f} → {g50.dead_frac_l1:.3f} | {g50.rowmean_abs_med_l1:.4f} | {100 * q.memo_win:.2f} |")
    if checks:
        L += ["", "## 5. 検査（`checks.json`）", ""]
        for k, v in checks.items():
            if isinstance(v, dict) and "pass" in v:
                muts = v.get("mutations", [])
                ms = f"・mutation {sum(m['detected'] for m in muts)}/{len(muts)} 検出" if muts else ""
                L.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}{ms}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--reuse-src", default=str(REPO / "results" / "wcap_rlmnist_0914" / "runs"))
    ap.add_argument("--reuse-json", default=str(REPO / "results" / RUN_ID / "reuse.json"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    ap.add_argument("--checks", default=str(REPO / "results" / RUN_ID / "checks.json"))
    ap.add_argument("--reuse-check", action="store_true", help="write reuse.json from the 4 verification shards and exit")
    args = ap.parse_args()
    src, reuse_src, out = Path(args.src), Path(args.reuse_src), Path(args.out)
    if args.reuse_check:
        r = compare_reuse(src, reuse_src)
        r["checked_at"] = dt.datetime.now().astimezone().isoformat()
        Path(args.reuse_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reuse_json).write_text(json.dumps(r, indent=2))
        print(r["status"], json.dumps(r["items"]))
        sys.exit(0 if r["status"] == "REUSE_OK" else 3)
    reuse = json.loads(Path(args.reuse_json).read_text()) if Path(args.reuse_json).exists() else None
    real = src.resolve() == (REPO / "results" / RUN_ID / "runs").resolve()
    pt, lm, provs, missing, origin = load(src, reuse_src, reuse["status"] if reuse else None)
    if real and missing:
        sys.exit(f"REFUSE: {len(missing)} shards missing for the real run; aggregate only after every run finished")
    res = analyze(pt, lm, provs)
    out.mkdir(parents=True, exist_ok=True)
    pt.to_csv(out / "per_task.csv", index=False)
    lm.to_csv(out / "layer_metrics.csv", index=False)
    pd.DataFrame(verdict_rows(res)).to_csv(out / "verdict.csv", index=False)
    checks = json.loads(Path(args.checks).read_text()) if Path(args.checks).exists() else None
    if checks and Path(args.checks).resolve() != (out / "checks.json").resolve():
        shutil.copy(args.checks, out / "checks.json")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "summary.md").write_text(summary_md(res, pt, lm, provs, missing, reuse, checks, head))
    agg = {"run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
           "src": str(src), "reuse_src": str(reuse_src), "reuse": reuse, "n_shards": len(provs), "missing": missing,
           "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"], "shard_origin": origin,
           "judgement": {"window": list(WIN), "bands": [BAND_LO, BAND_HI], "guard": GUARD, "tax_pt": TAX_PT,
                         "dead_tol": DEAD_TOL, "n_boot": N_BOOT, "rng_seed": RNG_SEED, "min_seeds": MIN_SEEDS},
           "labels": res["labels"], "patterns": res["patterns"], "guards": res["guards"],
           "shards": {shard_name(*k): {"run_id": v.get("run_id"), "git_hash": v["git_hash"], "hostname": v["hostname"],
                                       "wall_clock_s": v["wall_clock_s"], "code_sha256": v["code_sha256"],
                                       "git_dirty_code": v.get("git_dirty_code")} for k, v in provs.items()}}
    (out / "provenance.json").write_text(json.dumps(agg, indent=2, default=str))
    for key, v in res["labels"].items():
        print(f"{key}: {v['95'][0]}  ({v['95'][1]})  | 97.5%: {v['97.5'][0]}")
    for act, v in res["patterns"].items():
        print(f"pattern {act}: {v['95']}  | 97.5%: {v['97.5']}")


if __name__ == "__main__":
    main()
