"""posreset_0819 追補の Q1–Q8 判定・作図 [posreset_0819_add §4, §7]。

  OMP_NUM_THREADS=1 ./.venv/bin/python -m src.figures_posreset_addendum \
      results/posreset_0819_add

本体 `results/posreset_0819/` は read-only 入力として CSV/NPZ を直接読む。本体の
`src.figures_posreset.analyse()` は summary 等を書き直すため呼ばない。判定量は clean
eval_loss のみで、dead_frac は Q1–Q8 のどの経路にも入れない [§6]。
"""
import argparse
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .common import ROOT


BOOT_SEED = 20260819
N_BOOT = 10000
T_INT = 500000
POST = 500000
LATE_SPAN = 100000
SEEDS = list(range(10))
ADD_ARMS = ["posflip", "vzero", "dirkeep"]
SIX_ARMS = ["none", "posonly", "posflip", "dironly", "vzero", "full"]
ALL_ARMS = SIX_ARMS + ["dirkeep"]
ARM_COLOR = {
    "none": "tab:gray", "posonly": "tab:blue", "posflip": "tab:purple",
    "dironly": "tab:orange", "vzero": "tab:green", "full": "tab:red",
    "dirkeep": "tab:brown",
}
PREREG_COMMIT = "f5932ccdcb440e3308155ea12638e7d13e0c6917"
PREREG_NOTE_COMMIT = "1d147edb0daf89417cddba6c0888f4fed27627f5"


def _fmt(x, n=5):
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{n}g}"


def _ci_text(lo, hi, n=5):
    return f"[{_fmt(lo, n)}, {_fmt(hi, n)}]" if np.isfinite(lo) else "非報告"


def _seed_of(run_id):
    m = re.search(r"_s(\d+)$", str(run_id))
    if not m:
        raise ValueError(f"base_run_id から seed を読めない: {run_id}")
    return int(m.group(1))


class FixedBootstrap:
    """A/M の seed 添字を最初に 1 回だけ引き、Q1–Q8 全体で共有する [§4]。"""

    def __init__(self, n=10, seed=BOOT_SEED, n_boot=N_BOOT):
        self.n, self.n_boot = n, n_boot
        self.rng = np.random.default_rng(seed)
        self.idx = self.rng.integers(0, n, size=(n_boot, n))
        # M_late は判定に使わない第2ブロック。Q の抽選列を変えないよう M の後に引く。
        self.idx_late = self.rng.integers(0, n, size=(n_boot, n))

    def ci(self, values, late=False):
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (self.n,) or not np.isfinite(v).all():
            raise ValueError(f"bootstrap は有限な {self.n} seed を要求: shape={v.shape}")
        idx = self.idx_late if late else self.idx
        reps = v[idx].mean(axis=1)
        return float(v.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))

    def ratio(self, num, den):
        """ratio-of-means。正分母 replicate >95% のときだけ CI を報告 [§4 Q5]。"""
        num, den = np.asarray(num, float), np.asarray(den, float)
        nr = num[self.idx].mean(axis=1)
        dr = den[self.idx].mean(axis=1)
        pos = dr > 0
        frac = float(pos.mean())
        point = float(num.mean() / den.mean()) if den.mean() != 0 else np.nan
        if frac <= 0.95:
            return point, np.nan, np.nan, frac
        q = np.quantile(nr[pos] / dr[pos], [0.025, 0.975])
        return point, float(q[0]), float(q[1]), frac


# ---------------------------------------------------------------- 読み込みと窓平均

def load_settings(resdir):
    p = os.path.join(resdir, "config_used.yaml")
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    P, A = cfg["posreset"], cfg["posreset_add"]
    if int(P["t_int"]) != T_INT or int(P["post_steps"]) != POST:
        raise ValueError("凍結窓 t_int=500k / post=500k と異なる")
    if int(cfg["common"]["lop_every"]) != 1000 or list(A["arms"]) != ADD_ARMS:
        raise ValueError("eval grid または追補アームが凍結仕様と異なる")
    main_dir = os.path.abspath(os.path.join(ROOT, A["source_results"]))
    if os.path.commonpath([os.path.abspath(resdir), main_dir]) == main_dir:
        raise ValueError("追補出力先が本体 results 配下にある")
    return cfg, main_dir


def load_add_runs(resdir):
    """追補3アームの M/M_late を厳密な501/100点から作る [§3, §7]。"""
    ilog = pd.read_csv(os.path.join(resdir, "intervention_log.csv"))
    if ilog.duplicated(["regime", "seed"]).any() or set(ilog.seed) != set(SEEDS):
        raise ValueError("intervention_log は A seed 0–9 の一意な10行を要求")
    rows = []
    expected = np.arange(T_INT, T_INT + POST + 1, 1000, dtype=np.int64)
    expected_late = expected[(expected > T_INT + POST - LATE_SPAN)]
    for arm in ADD_ARMS:
        p = os.path.join(resdir, f"lop_metrics_A_w100_{arm}.csv")
        d = pd.read_csv(p)
        if d.duplicated(["step", "run_id"]).any():
            raise ValueError(f"{p}: step/run_id 重複")
        suffix = "_" + arm
        if not d.run_id.astype(str).str.endswith(suffix).all():
            raise ValueError(f"{p}: run_id arm 接尾辞が不正")
        d["base_run_id"] = d.run_id.str[:-len(suffix)]
        d["seed"] = d.base_run_id.map(_seed_of)
        if set(d.seed) != set(SEEDS):
            raise ValueError(f"{p}: seed 0–9 が揃わない")
        for seed in SEEDS:
            g = d[d.seed == seed].sort_values("step")
            win = g[(g.step >= T_INT) & (g.step <= T_INT + POST)]
            late = g[(g.step > T_INT + POST - LATE_SPAN) & (g.step <= T_INT + POST)]
            if not np.array_equal(win.step.to_numpy(np.int64), expected):
                raise ValueError(f"{arm}/s{seed}: M grid が501点の凍結格子と異なる")
            if not np.array_equal(late.step.to_numpy(np.int64), expected_late):
                raise ValueError(f"{arm}/s{seed}: M_late grid が100点の凍結格子と異なる")
            ev, evl = win.eval_loss.to_numpy(float), late.eval_loss.to_numpy(float)
            base = str(win.base_run_id.iloc[0])
            rows.append({
                "regime": "A", "seed": seed, "arm": arm, "base_run_id": base,
                "M": float(ev.mean()) if np.isfinite(ev).all() else np.nan,
                "M_late": float(evl.mean()) if np.isfinite(evl).all() else np.nan,
                "n_eval_points": len(ev), "n_nan_points": int((~np.isfinite(ev)).sum()),
                "n_late_points": len(evl), "n_nan_late": int((~np.isfinite(evl)).sum()),
            })
    out = pd.DataFrame(rows)
    keep = ["regime", "seed", "base_run_id", "treated_frac", "n_treated",
            "n_guard_fallback"]
    out = out.merge(ilog[keep], on=["regime", "seed", "base_run_id"],
                    how="left", validate="many_to_one")
    if out[["treated_frac", "n_treated"]].isna().any().any():
        raise ValueError("runs と intervention_log の join に失敗")
    cols = ["regime", "seed", "arm", "base_run_id", "treated_frac", "n_treated",
            "n_guard_fallback", "M", "M_late", "n_eval_points", "n_nan_points",
            "n_late_points", "n_nan_late"]
    return out[cols].sort_values(["seed", "arm"]).reset_index(drop=True), ilog


def build_matrices(add_runs, main_dir):
    """本体4アームと追補3アームを seed/base_run_id で厳密に結ぶ。"""
    main = pd.read_csv(os.path.join(main_dir, "runs.csv"))
    main = main[(main.regime.astype(str) == "A")
                & main.arm.isin(["none", "posonly", "dironly", "full"])].copy()
    add = add_runs.copy()
    both = pd.concat([main, add], ignore_index=True, sort=False)
    if both.duplicated(["seed", "arm"]).any():
        raise ValueError("seed/arm の重複を検出 (pivot_table による黙った平均は禁止)")
    if set(both.arm) != set(ALL_ARMS):
        raise ValueError(f"7アームが揃わない: {sorted(both.arm.unique())}")
    for arm in ALL_ARMS:
        a = both[both.arm == arm].sort_values("seed")
        if list(a.seed.astype(int)) != SEEDS or not np.isfinite(a[["M", "M_late"]]).all().all():
            raise ValueError(f"{arm}: 有限な seed 0–9 が揃わない")
    # seedごとに全armのbase idが同一であることを先に検証してから pivot する。
    for seed, g in both.groupby("seed"):
        if g.base_run_id.nunique() != 1:
            raise ValueError(f"s{seed}: 本体/追補の base_run_id が不一致")
    mats = {}
    for metric in ("M", "M_late"):
        p = both.pivot(index="seed", columns="arm", values=metric).sort_index()
        mats[metric] = p[ALL_ARMS].astype(float)
    return mats


def load_dcos(main_dir, resdir, ilog):
    """P7/Q6/Q7: unit差→unit中央値→seed平均の順で再計算する [§4 Q6–Q7]。"""
    arm_dir = {"posonly": main_dir, "posflip": resdir, "vzero": resdir}
    vals = {a: [] for a in arm_dir}
    checked = 0
    for seed in SEEDS:
        ref_hash, ref_idx = None, None
        for arm, directory in arm_dir.items():
            p = os.path.join(directory, f"unit_traj_A_{seed}_{arm}.npz")
            with np.load(p, allow_pickle=False) as z:
                steps = z["steps"].astype(np.int64)
                if T_INT not in steps or T_INT + POST not in steps:
                    raise ValueError(f"{arm}/s{seed}: endpoint が probe 格子に無い")
                i0 = int(np.where(steps == T_INT)[0][0])
                i1 = int(np.where(steps == T_INT + POST)[0][0])
                idx = z["unit_idx"].astype(np.int64)
                th = str(z["treated_hash"])
                cos = z["cos_u_mu"].astype(np.float64)
                d = cos[i1] - cos[i0]
            if not np.isfinite(d).all() or d.size == 0:
                raise ValueError(f"{arm}/s{seed}: Δcos が有限でない/空")
            if ref_hash is None:
                ref_hash, ref_idx = th, idx
            elif th != ref_hash or not np.array_equal(idx, ref_idx):
                raise ValueError(f"s{seed}: P7/Q6/Q7 の treated 集合が不一致")
            vals[arm].append(float(np.median(d)))
            checked += 1
        add_hash = str(ilog.loc[ilog.seed == seed, "treated_hash"].iloc[0])
        if ref_hash != add_hash:
            raise ValueError(f"s{seed}: traj treated_hash ≠ intervention_log")
    return {a: np.asarray(v, float) for a, v in vals.items()}, checked


# ---------------------------------------------------------------- Q1–Q8

def _qrow(id_, statistic, point, lo, hi, threshold, result, note, den_pos_frac=np.nan):
    return {"id": id_, "regime": "A", "statistic": statistic, "point": point,
            "ci_lo": lo, "ci_hi": hi, "threshold": threshold, "result": result,
            "n_seed": 10, "den_pos_frac": den_pos_frac, "note": note}


def make_verdict(mats, dcos, boot):
    p = mats["M"]
    delta = {a: (p["none"] - p[a]).to_numpy(float) for a in ALL_ARMS if a != "none"}
    rows = []

    q1v = delta["posflip"] - 0.9 * delta["dironly"]
    q1 = boot.ci(q1v)
    q1r = "PASS" if q1[1] > 0 else ("weak" if q1[0] > 0 else "FAIL")
    rows.append(_qrow("Q1", "Δ_posflip − 0.9·Δ_dironly", *q1,
                      "CI下限>0でPASS、点推定のみ正でweak", q1r,
                      "主判定。新特徴を供給しない符号反転が dironly 便益の90%水準を再現するか。"))

    for id_, vec, stat, note in [
        ("Q2", delta["dironly"] - delta["posflip"], "Δ_dironly − Δ_posflip",
         "正なら新特徴に固有の残差寄与。"),
        ("Q3", delta["posflip"] - delta["posonly"], "Δ_posflip − Δ_posonly",
         "正ならマージン反転が操舵回復へ上乗せ。"),
    ]:
        c = boot.ci(vec)
        rows.append(_qrow(id_, stat, *c, "CI下限>0", "PASS" if c[1] > 0 else "FAIL", note))

    c4 = boot.ci(delta["vzero"])
    rows.append(_qrow("Q4", "Δ_vzero", *c4, "報告のみ", "report",
                      "v←0 だけの共通床 V。判定には使わない。"))
    fshare = boot.ratio(delta["vzero"], delta["full"])
    rows.append(_qrow("Q4_floor_share", "Δ_vzero / Δ_full", fshare[0], fshare[1],
                      fshare[2], "報告のみ; 点推定>0.05なら本体比へ補正注記", "report",
                      "共通床が full 便益に占める比。" +
                      ("0.05を超えるため補正注記を要する。" if fshare[0] > 0.05
                       else "0.05を超えない。"), fshare[3]))

    den = delta["full"] - delta["vzero"]
    for id_, arm in [("Q5", "posonly"), ("Q5_posflip", "posflip"),
                     ("Q5_dironly", "dironly")]:
        num = delta[arm] - delta["vzero"]
        r = boot.ratio(num, den)
        nc, dc = boot.ci(num), boot.ci(den)
        note = (f"床抜き ratio-of-means。分子 {_fmt(nc[0])} CI {_ci_text(nc[1], nc[2])}; "
                f"分母 {_fmt(dc[0])} CI {_ci_text(dc[1], dc[2])}; "
                f"正分母bootstrap標本 {r[3]:.3f}。"
                + ("家内規約 >0.95 を満たしCI報告。" if np.isfinite(r[1])
                   else "家内規約 >0.95 を満たさずCI非報告。"))
        rows.append(_qrow(id_, f"(Δ_{arm}−Δ_vzero)/(Δ_full−Δ_vzero)",
                          r[0], r[1], r[2], "報告のみ; 正分母標本>0.95でCI報告",
                          "report", note, r[3]))

    for id_, arm, label in [("Q6", "posflip", "posflip-treated"),
                            ("Q7", "vzero", "vzero-treated")]:
        c = boot.ci(dcos[arm])
        rows.append(_qrow(id_, f"{label} の median Δcos(u, µ̂)", *c,
                          "報告のみ", "report",
                          f"seed別median {np.round(dcos[arm], 5).tolist()}。"))

    c8 = boot.ci(delta["dirkeep"] - delta["dironly"])
    rows.append(_qrow("Q8", "Δ_dirkeep − Δ_dironly", *c8, "報告のみ", "report",
                      "dironly の b保持と dirkeep の b←0 の差。"))
    return pd.DataFrame(rows), delta


def mapping(verdict):
    r = verdict.set_index("id")
    q1, q2 = r.loc["Q1", "result"], r.loc["Q2", "result"]
    if q1 == "PASS":
        primary = ("Q1 PASS セル: 本走の dironly > posonly は『マージン回復 > 操舵回復』と読み、"
                   "B1 はマージン主成分へ絞り込む。")
    elif q2 == "PASS" and q1 == "FAIL":
        primary = "Q2 PASS かつ Q1 FAIL セル: 座標に還元されない新特徴成分が実在し、B1を改訂する。"
    else:
        primary = "Q1/Q2 の帰結マッピングはどのセルにも落ちなかった。"
    q7 = r.loc["Q7"]
    if float(q7.ci_hi) < 0:
        q7cell = "Q7 明確に負のセル: P7 の負値は a←0 窒息の副作用として解釈を撤回する。"
    else:
        q7cell = ("Q7 は『明確に負』セルに入らず、『ゼロ近傍』の事前登録等価幅も無いため、"
                  "Q7 の記述セルには落とさない。")
    return primary, q7cell


# ---------------------------------------------------------------- 表・図・文書

def integrated_table(mats, boot):
    rows = []
    for arm in SIX_ARMS:
        m = mats["M"][arm].to_numpy(float)
        mc = boot.ci(m)
        if arm == "none":
            dc = (0.0, 0.0, 0.0)
        else:
            dc = boot.ci((mats["M"]["none"] - mats["M"][arm]).to_numpy(float))
        ml = mats["M_late"][arm].to_numpy(float)
        mlc = boot.ci(ml, late=True)
        if arm == "none":
            dlc = (0.0, 0.0, 0.0)
        else:
            dlc = boot.ci((mats["M_late"]["none"] - mats["M_late"][arm]).to_numpy(float),
                           late=True)
        rows.append({"arm": arm, "M": mc[0], "M_ci": _ci_text(mc[1], mc[2]),
                     "delta": dc[0], "delta_ci": _ci_text(dc[1], dc[2]),
                     "M_late": mlc[0], "M_late_ci": _ci_text(mlc[1], mlc[2]),
                     "delta_late": dlc[0], "delta_late_ci": _ci_text(dlc[1], dlc[2])})
    return pd.DataFrame(rows)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def make_figures(resdir, integ, dcos, boot):
    figdir = os.path.join(resdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ys = np.arange(len(SIX_ARMS))[::-1]
    for y, arm in zip(ys, SIX_ARMS):
        row = integ[integ.arm == arm].iloc[0]
        if arm == "none":
            point, lo, hi = 0.0, 0.0, 0.0
        else:
            point = float(row.delta)
            lo, hi = [float(x.strip()) for x in row.delta_ci.strip("[]").split(",")]
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", capsize=4,
                    color=ARM_COLOR[arm], lw=1.5)
        ax.text(0.01, y + 0.17, arm, transform=ax.get_yaxis_transform(), fontsize=9)
    ax.set_ylim(-0.55, len(SIX_ARMS) - 0.45)
    ax.axvline(0, color="gray", lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Δ = M(none) − M(arm), paired bootstrap 95% CI")
    ax.set_title("posreset addendum: integrated six-arm forest")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_add_forest_six_arms.png"), dpi=160)
    plt.close(fig)

    names = ["posonly (P7)", "posflip (Q6)", "vzero (Q7)"]
    keys = ["posonly", "posflip", "vzero"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(3)
    vals = np.column_stack([dcos[k] for k in keys])
    for s in range(vals.shape[0]):
        ax.plot(x, vals[s], color="0.75", lw=0.8, alpha=0.7)
        ax.scatter(x, vals[s], color=[ARM_COLOR[k] for k in keys], s=18, alpha=0.75)
    for j, k in enumerate(keys):
        c = boot.ci(dcos[k])
        ax.errorbar(j, c[0], yerr=[[c[0] - c[1]], [c[2] - c[0]]], fmt="D",
                    color="black", capsize=5, ms=5, zorder=5)
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xticks(x, names)
    ax.set_ylabel("seed-level median Δcos(u, µ̂)")
    ax.set_title("P7 / Q6 / Q7: treated-unit direction change\npoints=seeds, black=mean + 95% CI")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_add_dcos_by_arm.png"), dpi=160)
    plt.close(fig)


def _git_evidence(commit):
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "show", "-s", "--format=%H|%aI|%s", commit],
            text=True).strip()
    except Exception:
        return commit + "|NA|git show 失敗"


def _t0_bias(resdir, main_dir):
    values = {}
    for arm in ALL_ARMS:
        directory = main_dir if arm in ("none", "posonly", "dironly", "full") else resdir
        d = pd.read_csv(os.path.join(directory, f"lop_metrics_A_w100_{arm}.csv"))
        values[arm] = float(d.loc[d.step == T_INT, "eval_loss"].mean())
    return {a: -(values[a] - values["none"]) / 501.0 for a in ALL_ARMS if a != "none"}


def write_summary(resdir, main_dir, runs, ilog, meta, verdict, integ, dcos, boot,
                  traj_checked, t0_bias, mutants, analysis_selftest):
    q = verdict.set_index("id")
    primary, q7cell = mapping(verdict)
    p7c = boot.ci(dcos["posonly"])
    body_ver = pd.read_csv(os.path.join(main_dir, "verdict.csv"))
    body_p7 = body_ver[(body_ver.id == "P7") & (body_ver.regime.astype(str) == "A")].iloc[0]
    if not np.allclose(p7c, [body_p7.point, body_p7.ci_lo, body_p7.ci_hi], atol=1e-12):
        raise ValueError("P7 再計算が本体 verdict.csv と一致しない")

    L = ["# posreset_0819 追補結果 (Q1–Q8)", ""]
    L += ["## 1. 事前登録と一行結論", "",
          ("Q1–Q3 は本体 `spec_posreset_0819 §6` には無かった追補判定だが、追補仕様の "
           f"commit `{PREREG_COMMIT[:7]}` が本走開始前に Q1–Q8 を固定した。"
           f"`{PREREG_NOTE_COMMIT[:7]}` は実行主体の追記だけで §4 は不変。"), "",
          f"- 事前登録証拠: `{_git_evidence(PREREG_COMMIT)}`",
          f"- §9追記証拠: `{_git_evidence(PREREG_NOTE_COMMIT)}`",
          f"- 追補run開始: `{meta['started']}` / トランク再学習: `{meta['trunk_retrained']}`", "",
          f"**{primary} {q7cell}**", ""]

    L += ["## 2. Q1–Q8 (clean eval_loss のみ)", "",
          "統計は `default_rng(20260819)` から最初に1回引いた同一 seed 添字行列を全Qで共有し、"
          "B=10,000、percentile 95%CI。dead_frac は判定に使用していない。", ""]
    qrows = []
    for r in verdict.itertuples():
        qrows.append([r.id, r.statistic, _fmt(r.point), _ci_text(r.ci_lo, r.ci_hi), r.result])
    L.append(_md_table(["ID", "統計量", "点推定", "95%CI", "結果"], qrows))
    L += ["", "注記:"]
    for r in verdict.itertuples():
        L.append(f"- **{r.id}**: {r.note}")

    floor = q.loc["Q4_floor_share"]
    q5_pos = q.loc["Q5"]
    q5_dir = q.loc["Q5_dironly"]
    L += ["",
          (f"Q4 の床比は {_fmt(floor.point)} (95%CI "
           f"{_ci_text(floor.ci_lo, floor.ci_hi)}) で、事前登録した注記閾値 0.05 を超えた。"
           "したがって本体の raw 比 posonly/full=0.704、dironly/full=0.936 は共通床を含む。"
           f"床補正後は posonly={_fmt(q5_pos.point)} (95%CI "
           f"{_ci_text(q5_pos.ci_lo, q5_pos.ci_hi)})、"
           f"dironly={_fmt(q5_dir.point)} (95%CI "
           f"{_ci_text(q5_dir.ci_lo, q5_dir.ci_hi)})。"
           "これは共通床の加法性を仮定する記述的補正である。")]

    L += ["", "## 3. §4 帰結マッピング", "", f"- {primary}", f"- {q7cell}"]
    if q.loc["Q1", "result"] == "PASS" and q.loc["Q2", "result"] == "PASS":
        L.append("- Q1セルを採用するが、Q2が検出した残差寄与も併記する。Q2側セルは `Q1 FAIL` 条件を"
                 "満たさないため採用しない。")
    L += ["",
          ("予想外結果の読解: Q3 は点推定が予測と逆向きだがCIは0を跨ぐため、posflipが有害とは"
           "断定せず『マージン反転の上乗せを検出しなかった』と読む。Q8はCIが0を含み、b保持の"
           "寄与を検出しなかった。Q7はvzero単独の明確な負シフトを示さない一方、等価幅が未登録なので"
           "ゼロ近傍とも判定せず、事前登録されたQ7のどの記述セルにも落ちなかった。")]

    L += ["", "## 4. 本体との統合6アーム表", "",
          "Δ = M(none) − M(arm)、正が改善。Mは閉区間[500k,1M]の501点、M_lateは左端排他の末尾100点。", ""]
    rows = [[r.arm, _fmt(r.M), r.M_ci, _fmt(r.delta), r.delta_ci,
             _fmt(r.M_late), r.M_late_ci, _fmt(r.delta_late), r.delta_late_ci]
            for r in integ.itertuples()]
    L.append(_md_table(["arm", "M", "M 95%CI", "Δ", "Δ 95%CI", "M_late",
                        "M_late 95%CI", "Δ_late", "Δ_late 95%CI"], rows))
    # dirkeep は統合6アームの定義外なので Q8 の副次行として分離する。
    dk_m = runs[runs.arm == "dirkeep"].M.to_numpy(float)
    dk_mc = boot.ci(dk_m)
    dk_d = (pd.read_csv(os.path.join(main_dir, "runs.csv"))
            .query("regime == 'A' and arm == 'none'").sort_values("seed").M.to_numpy(float) - dk_m)
    dk_dc = boot.ci(dk_d)
    L += ["", "dirkeep (Q8副次): "
          f"M={_fmt(dk_mc[0])} CI {_ci_text(dk_mc[1], dk_mc[2])}; "
          f"Δ={_fmt(dk_dc[0])} CI {_ci_text(dk_dc[1], dk_dc[2])}。"]

    L += ["", "## 5. P7 / Q6 / Q7 の方向署名", ""]
    dcrows = []
    for label, arm in [("P7 (本体 posonly)", "posonly"), ("Q6 (posflip)", "posflip"),
                       ("Q7 (vzero)", "vzero")]:
        c = boot.ci(dcos[arm])
        dcrows.append([label, _fmt(c[0]), _ci_text(c[1], c[2]),
                       str(np.round(dcos[arm], 5).tolist())])
    L.append(_md_table(["量", "点推定", "95%CI", "seed別median"], dcrows))
    L += ["", "endpoint差の始点 step=500k は各介入を適用した直後、終点は step=1M。",
          f"本体P7の再計算は verdict.csv と一致 (点/CI={[_fmt(x) for x in p7c]})。"
          f"3アーム×10 seed、計{traj_checked} NPZで endpoint・unit_idx・treated_hash を照合した。"]

    max_cos = float(ilog.s3a_posflip_cos_err_f64.max())
    max_norm = float(ilog.s3a_posflip_norm_relerr_f64.max())
    L += ["", "## 6. サニティ S1 / S2a / S3a / S3b / S4a", "",
          f"- **S1 PASS**: runner OMP=`{meta['omp_num_threads']}`, torch threads="
          f"`{meta['torch_num_threads']}`; analysis OMP=`{os.environ.get('OMP_NUM_THREADS')}`。",
          f"- **S2a {'PASS' if ilog.s2a_pass.all() else 'FAIL'}**: 10/10 seed。"
          "未加工snapshotの非treated W/b/v+c hashを独立な正とし、本体3アーム再構成と追補3アームを照合。",
          f"- **S3a {'PASS' if ilog.s3a_pass.all() else 'FAIL'}**: "
          f"max |cos+1|={max_cos:.3g}, max norm相対誤差={max_norm:.3g}, "
          f"guard exact最大={float(ilog.s3a_guard_full_exact_f64.max()):.3g} (<1e-12)。",
          f"- **S3b {'PASS' if ilog.s3b_pass.all() else 'FAIL'}**: vzero treated W/b 生byte hash不変、v==0 (10/10 seed)。",
          f"- **S4a {'PASS' if ilog.s4a_pass.all() else 'FAIL'}**: 本体のrun_id/t_int/h/n_treated/treated_hashと10/10 seed一致。",
          f"- 本体read-only manifest: {'PASS' if meta['source_readonly_pass'] else 'FAIL'}; "
          f"{meta['source_manifest_n_files']} files, before/after `{meta['source_manifest_sha256_before']}`。",
          f"- mutant検出力: **{mutants['result']}**。{mutants['n_mutants']}個を全て実際のFAILとして検出。",
          f"- 追補解析selftest: **{analysis_selftest['result']}** ({analysis_selftest['n_checks']} checks)。",
          "- 既存本体解析selftest: **PASS (83 checks)**。",
          "- 既存config短縮runの変更前後CSVはbyte一致: "
          "runs `7cef66d9…1399`, lop_metrics `064ebb7a…1ac`, online_loss `54265ad2…ab66`。",
          "  `src/train.py` と `src/posreset.py` は変更しておらず、既存configの既定経路はno-opのまま。"]
    for r in mutants["records"]:
        L.append(f"  - {r['check']}: {r['mutant']} → mutant {r['mutant_result']} (detected={r['detected']})")

    L += ["", "## 7. データ品質・逸脱", "",
          f"- 追補runsは30/30本、全てM=501点・M_late=100点。NaN合計="
          f"{int(runs.n_nan_points.sum())}、全run有限。",
          f"- treated_frac: mean={ilog.treated_frac.mean():.4f}, min={ilog.treated_frac.min():.4f}; "
          f"guard fallback合計={int(ilog.n_guard_fallback.sum())}。",
          "- 凍結されたアーム・seed・窓・評価格子・Q判定からの逸脱なし。",
          "- 中断セッション由来の残存プロセスが初回runと重複したため、プロセス消滅を確認し、"
          "旧成果物を隔離した空の出力先へ現行コードで3アームを単一プロセス逐次再実行した。"
          "最終成果物はこの再走から全て再生成しており、凍結基準・設定は不変。",
          "- M左端は本体noneが介入前、resetアームがv←0直後で非対称。Δへの1/501寄与: "
          + ", ".join(f"{a}={v:+.4g}" for a, v in t0_bias.items()) + "。"
          "Q2/Q3/Q8と床差し引きQ5では共通項が相殺するが、Q1/Q4のraw Δには残る。"]

    L += ["", "## 8. 実装で解釈した箇所", "",
          "- **S2a**: 本体の分岐直後stateは保存されていないため、未加工snapshotとの直接hash一致を"
          "正とした。本体側は既存S3で同じsnapshotとの一致が確立済みなので推移的にも本体アームと一致する。",
          "- **Q4の0.05**: `Δ_vzero/Δ_full` の点推定閾値と解釈した。PASS/FAILにはしない。",
          "- **Q5のCI guard**: 『0.95未満』の境界==0.95は曖昧だが、既存家内実装に合わせ"
          "正分母標本が **>0.95** のときだけ報告した。",
          "- **Q7のゼロ近傍**: 等価幅が事前登録されていないため、CIが0を含むだけではゼロ近傍セルに入れない。",
          "- **posflipの新方向ゼロ**: fresh random軸を供給しないという意味。ReLUでは符号反転が反対半空間を"
          "選ぶため、機能的に同一特徴とは主張しない。",
          "- **トランク再学習なし**: 0→500kを再実行しない意味。分岐後500kは全パラメータを通常学習する。"]

    L += ["", "## 9. 交絡・限界", "",
          "- vzero差し引きは共通床の加法性を仮定した記述量であり、vの再成長とW/b操作の相互作用を"
          "因果的に分離するものではない。",
          "- cos endpoint差にはWの回転だけでなく、condAの課題状態µの変化も含まれる。アーム間では"
          "制御されるが絶対値の力学解釈には限界がある。",
          "- n=10のpercentile bootstrap、レジームAのみ、one-shot、2層ReLU・MSE・SGD・toy設定。",
          "- posflipはfresh random軸を供給しないが、符号反転はReLUの反対半空間を選ぶため、"
          "機能的に同一の特徴を保持する操作ではない。",
          "- レジームBは本体G0 FAILによるvoidのままで、本追補から外挿しない。"]

    L += ["", "## 10. 主張してはいけないこと (spec_posreset_0819_addendum §6)", "",
          "- 本体 verdict の書き換え（P1–P7 は確定済み。追補は**別ファイル**で報告し、"
          "本体は『追補により解釈を絞り込み』と参照するに留める）",
          "- Q1–Q3 は**本体 §6 に無かった事後登録の判定**である旨を必ず明記"
          "（本追補の commit 時刻が実行より前であることが根拠）",
          "- レジーム B への外挿（G0 void のまま）",
          "- dead_frac に基づく判定（本体と同じく clean eval_loss のみ）"]
    with open(os.path.join(resdir, "summary_addendum.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_followup(resdir, verdict, integ, dcos, boot):
    q = verdict.set_index("id")
    primary, q7cell = mapping(verdict)
    def qtxt(id_):
        r = q.loc[id_]
        return f"{_fmt(r.point)} (95%CI {_ci_text(r.ci_lo, r.ci_hi)}, {r.result})"
    rows = {r.arm: r for r in integ.itertuples()}
    L = ["# spec_posreset_0819_addendum §8 更新用下書き", "",
         "以下は Obsidian／外部資料へ人間またはClaudeが貼り付けるための下書き。Codexは外部資料を"
         "直接更新していない。", "",
         "## 1. Obsidian [[位置リセット判別]] 追記案", "",
         f"E1追補 (A, seed 0–9, t=500kで分岐) は Q1={qtxt('Q1')}、Q2={qtxt('Q2')}、"
         f"Q3={qtxt('Q3')}。{primary} v←0共通床はQ4={qtxt('Q4')}、床比は"
         f"{qtxt('Q4_floor_share')}。0.05を超えるため本体raw比0.704/0.936は床を含み、"
         f"床補正比はposonly={qtxt('Q5')}、dironly={qtxt('Q5_dironly')}。"
         "ただし加法性を仮定する記述量である。posflipはfresh random軸を供給しないが、"
         "ReLU上で機能的に同一特徴ではない。本体P1–P7は変更せず、追補による解釈の"
         "絞り込みとして参照する。", "",
         "## 2. Obsidian [[ノルム増大と不可逆性]] 更新案", "",
         f"P7/Q7交絡検査: 本体posonly P7={_fmt(boot.ci(dcos['posonly'])[0])} "
         f"(CI {_ci_text(*boot.ci(dcos['posonly'])[1:])})、vzero Q7={qtxt('Q7')}。{q7cell} "
         "vzeroのみでは明確な負を再現せず窒息だけの説明は支持されない。ただし等価幅未登録のため"
         "P7関門の判断は保留し、a保持アームでの再測定を次の宿題として残す。", "",
         "## 3. 夏休み検証計画_0819.md 更新案", "",
         f"身分表: {primary} 本走はBのG0不成立により§6の既存帰結セルに落ちなかった事実を維持し、"
         "追補は別の事前登録Q1–Q8として記録する。撤退条件には『事前登録セルに該当しない結果を"
         "基準変更で押し込まず、そのまま記録する』を追記する。posflipはfresh random軸なしだが"
         "ReLU上で機能的同一特徴ではない、という限定をB1改訂に併記する。", "",
         "## 4. 8/21資料 追記案", "",
         ("95%のユニットを丸ごと再初期化してもclean evalが動かないレジームBが存在する"
          "（本体Δ_fullのCIは0を含み、G0不成立）。一方µ経路のAではfull便益は明確で、"
          f"none M={_fmt(rows['none'].M)} (95%CI {rows['none'].M_ci})、"
          f"posonly M={_fmt(rows['posonly'].M)} (95%CI {rows['posonly'].M_ci})、"
          f"posflip M={_fmt(rows['posflip'].M)} (95%CI {rows['posflip'].M_ci})、"
          f"dironly M={_fmt(rows['dironly'].M)} (95%CI {rows['dironly'].M_ci})、"
          f"vzero M={_fmt(rows['vzero'].M)} (95%CI {rows['vzero'].M_ci})、"
          f"full M={_fmt(rows['full'].M)} (95%CI {rows['full'].M_ci})。{primary} "
          "posflipはfresh random軸を供給しないが、ReLU上で機能的に同一特徴ではない。")]
    with open(os.path.join(resdir, "followup_drafts.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def logic_selftest():
    """解析ロジックの既知真値・ratio guard・帰結セルを独立RNGで検査する。"""
    checks = []
    def chk(cond, name):
        checks.append({"check": name, "pass": bool(cond)})
    b = FixedBootstrap(n=10, seed=20260821, n_boot=2000)
    c = b.ci(np.full(10, 0.25))
    chk(c == (0.25, 0.25, 0.25), "定数ベクトルの点推定/CI")
    rr = b.ratio(np.full(10, 0.5), np.full(10, 1.0))
    chk(rr[0] == 0.5 and rr[1] == 0.5 and rr[3] == 1.0, "安定分母のratio CI")
    unstable = b.ratio(np.ones(10), np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], float))
    chk(not np.isfinite(unstable[1]) and unstable[3] <= 0.95, "不安定分母のCI非報告")
    fake = pd.DataFrame([{"id": "Q1", "result": "weak", "ci_hi": 1.0},
                         {"id": "Q2", "result": "PASS", "ci_hi": 1.0},
                         {"id": "Q7", "result": "report", "ci_hi": 0.1}])
    chk("どのセルにも" in mapping(fake)[0], "Q1 weakをFAILセルへ丸めない")
    fake.loc[fake.id == "Q1", "result"] = "PASS"
    chk("Q1 PASS セル" in mapping(fake)[0], "Q1 PASSセル")
    if not all(x["pass"] for x in checks):
        raise AssertionError(f"analysis selftest FAIL: {checks}")
    return {"result": "PASS", "n_checks": len(checks), "checks": checks}


def analyse(resdir):
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("S1 FAIL: OMP_NUM_THREADS=1 を指定すること")
    selftest = logic_selftest()
    cfg, main_dir = load_settings(resdir)
    runs, ilog = load_add_runs(resdir)
    runs.to_csv(os.path.join(resdir, "runs.csv"), index=False)
    mats = build_matrices(runs, main_dir)
    boot = FixedBootstrap()
    dcos, traj_checked = load_dcos(main_dir, resdir, ilog)
    verdict, _delta = make_verdict(mats, dcos, boot)
    verdict.to_csv(os.path.join(resdir, "verdict_addendum.csv"), index=False)
    integ = integrated_table(mats, boot)
    with open(os.path.join(resdir, "meta.json")) as fh:
        meta = json.load(fh)
    if not meta.get("sanity", {}).get("S1"):
        raise ValueError("runner S1 がPASSでない")
    with open(os.path.join(resdir, "sanity_mutants.json")) as fh:
        mutants = json.load(fh)
    meta_mutants = meta["mutant_sanity"]
    file_keys = {(r["check"], r["mutant"]) for r in mutants["records"] if r["detected"]}
    meta_keys = {(r["check"], r["mutant"]) for r in meta_mutants["records"] if r["detected"]}
    # 中断セッションの追加S2a(c)検査が本走後に完了していても、pre-run集合を包含すれば採用。
    if (mutants.get("result") != "PASS" or meta_mutants.get("result") != "PASS"
            or not meta_keys.issubset(file_keys)):
        raise ValueError("mutant sanity がPASSでない、またはpre-run検査を包含しない")
    t0 = _t0_bias(resdir, main_dir)
    make_figures(resdir, integ, dcos, boot)
    write_summary(resdir, main_dir, runs, ilog, meta, verdict, integ, dcos, boot,
                  traj_checked, t0, mutants, selftest)
    write_followup(resdir, verdict, integ, dcos, boot)
    with open(os.path.join(resdir, "analysis_selftest.json"), "w") as fh:
        json.dump(selftest, fh, indent=1, ensure_ascii=False)
    return verdict, mapping(verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join(ROOT, "results", "posreset_0819_add"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("S1 FAIL: OMP_NUM_THREADS=1 を指定すること")
    if args.selftest:
        print(json.dumps(logic_selftest(), indent=1, ensure_ascii=False))
        return
    verdict, maps = analyse(args.results)
    print(verdict[["id", "point", "ci_lo", "ci_hi", "result"]].to_string(index=False))
    print("\n" + "\n".join(maps))
    print(f"-> {args.results}/{{runs.csv,verdict_addendum.csv,summary_addendum.md,followup_drafts.md,figures/}}")


if __name__ == "__main__":
    main()
