"""ledger_0822: 力の台帳の本番組み直し（群別・Q2b）[spec_ledger_0822]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.ledger.ledger \
      [results/ratchet_log_0819] [--events results/dead2path_0821/events.csv] \
      [--outdir results/ledger_0822]

**再学習なし**。入力は `results/ratchet_log_0819/logs/seed*.npz` と
`results/dead2path_0821/events.csv` の **2 つだけ**。境界窓で税 (F_self) と稼ぎ
(F_rest) を積分し、死亡個体を再分類死/輸送死に分けて、死亡個体の沈下に税が占める
割合の中央値を seed 単位ペアブートストラップ CI 付きで再推定する
([[命題リスト]] Q2b の証拠欄をスモーク R=8・100k step から本番 10 seed・1M step へ
格上げ)。測る項目集合は vault `中和と境界窓.md` §4・§5 と同一 (spec §1)。本 spec が
足すのは (1) 群分け (再分類死/輸送死。ラベルは dead2path_0821/events.csv) と
(2) seed 単位ペアブートストラップ CI の 2 点だけ。

判定は spec §5 の L1–L8 が唯一の正で、本モジュールはそれを実装するだけ。判定は置かない
(§5 の理由により Q2b は既に「棄却寄り」で確定済み)。L1 が主判定 (中央値・報告のみ)。

**積分窓は 2 つとも事前登録・両方出す** (spec §3):
  W-wide   [B+1, B+2000]  毎 step 100 点 + 粗点 2 点 (B+1000, B+2000)  — 主判定
  W-narrow [B+1, B+100]   毎 step 100 点 (厳密)                        — 副次

**禁止事項 (spec §1, §8)**: 陣営 (sign(v_i)) 別の符号分布は出力しない (穴A / Q17)。
相殺比の境界前後プロファイルは出さない (次の測定)。summary.md に機構の考察を書かない。

出力 (すべて --outdir の中): verdict.csv / per_seed_metrics.csv / summary.md /
meta.json / figures/。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 日本語ラベルが豆腐にならないよう CJK フォントを優先させる [dead2path.py と同じ処置]
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                              "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.common import ROOT, load_config                              # noqa: E402
from src.figures_ratchet_log import TAU, boot_ci, load_seeds          # noqa: E402

BOOT_N = 10000
BOOT_SEED = 20260822                    # spec §7 の事前登録の抽選列
WIN_END = {"W-wide": 2000, "W-narrow": 100}   # 窓の右端 (B からのオフセット)

# spec §5 に転記されているスモーク値 (R=8・100k step・9 境界)。W-wide のみ比較可能
# (spec §4-3)。群分け (輸送死/再分類死) はスモークに無い新規列なので smoke=NaN。
SMOKE = {
    ("L1", "全群"): 0.245,
    ("L2", "全群"): 0.683,
    ("L3", "全群", "self"): -0.016, ("L3", "全群", "rest"): -0.015, ("L3", "全群", "gate"): -0.055,
    ("L3", "生存", "self"): -1.66, ("L3", "生存", "rest"): 1.65, ("L3", "生存", "gate"): -0.028,
    ("L4", "全群"): 0.538,
    ("L5", "全群"): 0.145,
    ("L6", "生存", "self"): -9.50, ("L6", "生存", "rest"): 9.39,
    ("L7", "窓内", "gate"): -0.076, ("L7", "窓内", "self"): -5.71, ("L7", "窓内", "frac_neg"): 0.568,
    ("L7", "窓外", "gate"): 0.000, ("L7", "窓外", "self"): -48.19, ("L7", "窓外", "frac_neg"): 0.465,
    ("L8", "全対象"): 0.375,
}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ ブートストラップ

def boot_ci_median(rng, vec, B):
    """seed 単位ペアブートストラップ、点推定・CI とも**中央値** [L1 主判定専用]。"""
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = np.median(v[rng.integers(0, v.size, (B, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# ------------------------------------------------------------------ 境界 (S2) とユニット事件テーブル (§3)

def find_boundaries(d, period):
    """境界を flip_state の差分だけから機械的に決める (ハードコードしない, S2)。

    戻り値: idx_b (境界 B の記録点インデックス, 昇順), lefts (B の step 値),
    rights (B の直後の記録点の step 値)。"""
    step, fs = d["step"], d["flip_state"]
    chg = (np.abs(np.diff(fs, axis=0)) > 0).any(axis=1)
    idx_b = np.flatnonzero(chg)
    return idx_b, step[idx_b], step[idx_b + 1]


def build_target_table(seeds, period):
    """§3 のユニット事件テーブル (境界 B で alive な全個体、seed x k x unit)。

    各行に W-wide / W-narrow の ∫F_self, ∫F_rest, ∫F_gate、L6 用の残り区間積分、
    L7 用の窓外区間の ∫F_self/∫F_gate、L8 用の全タスク ∫F_gate を持たせる。dead 判定は
    「そのタスク中 [B, B+period) で p̂ が初めて 0.05 を下抜けた」で、その最初の
    クロスの step_prev を events.csv との突合キーとして残す。"""
    rows = []
    s2_rows = []
    for d in seeds:
        seed = int(d["seed"])
        run_id = str(d["run_id"])
        step = d["step"]
        p, Fs, Fr, Fg = d["p_hat"], d["F_self"], d["F_rest"], d["F_gate"]
        n, h = p.shape

        idx_b, lefts, rights = find_boundaries(d, period)
        aligned = int((lefts % period == 0).sum())
        adjacent = int(((rights - lefts) == 1).sum())
        ok = bool(idx_b.size == 99 == aligned == adjacent)
        s2_rows.append(dict(seed=seed, run_id=run_id, n_boundaries=int(idx_b.size),
                            n_aligned=aligned, n_adjacent=adjacent, ok=ok))
        if not ok:
            continue

        for k, (ib, B) in enumerate(zip(idx_b, lefts), start=1):
            ib, B = int(ib), int(B)
            nextB = B + period
            inb = int(np.searchsorted(step, nextB))
            assert step[inb] == nextB, f"seed{seed} k={k}: next boundary の記録点が無い"

            alive = p[ib] >= TAU
            p_sub = p[ib:inb + 1]
            below = p_sub < TAU
            cross = below[1:] & ~below[:-1]                 # [inb-ib, h]
            has_cross = cross.any(axis=0)
            first_row = cross.argmax(axis=0)
            death_idx = ib + first_row                       # step_prev の絶対インデックス
            dead_mask = alive & has_cross

            m_wide = (step > B) & (step <= B + 2000)
            m_narrow = (step > B) & (step <= B + 100)
            m_remain = (step >= B + 2000) & (step < nextB)   # L6: [B+2000, B+1e4)
            m_full = (step >= B) & (step < nextB)            # L8 分母: [B, B+1e4)
            m_wide_out = (step > B + 2000) & (step < nextB)
            m_narrow_out = (step > B + 100) & (step < nextB)

            def integ(mask, arr):
                return np.trapezoid(arr[mask], x=step[mask], axis=0)

            Iself_w, Irest_w, Igate_w = integ(m_wide, Fs), integ(m_wide, Fr), integ(m_wide, Fg)
            Iself_n, Irest_n, Igate_n = integ(m_narrow, Fs), integ(m_narrow, Fr), integ(m_narrow, Fg)
            Igate_full = integ(m_full, Fg)
            remain_self, remain_rest = integ(m_remain, Fs), integ(m_remain, Fr)
            # L7 窓外 (台形積分。§3 の 94.7%/94.0% 診断で trapz が正しい演算であることを
            # 確認済み — 逸脱節 6 参照)
            Iself_wo, Igate_wo = integ(m_wide_out, Fs), integ(m_wide_out, Fg)
            Iself_no, Igate_no = integ(m_narrow_out, Fs), integ(m_narrow_out, Fg)

            alive_units = np.flatnonzero(alive)
            for u in alive_units:
                dead = bool(dead_mask[u])
                step_prev = int(step[death_idx[u]]) if dead else -1
                rows.append(dict(
                    seed=seed, run_id=run_id, k=k, B=B, unit=int(u), dead=dead,
                    step_prev=step_prev,
                    Iself_wide=float(Iself_w[u]), Irest_wide=float(Irest_w[u]),
                    Igate_wide=float(Igate_w[u]),
                    Iself_narrow=float(Iself_n[u]), Irest_narrow=float(Irest_n[u]),
                    Igate_narrow=float(Igate_n[u]),
                    Igate_full=float(Igate_full[u]),
                    remain_self=float(remain_self[u]), remain_rest=float(remain_rest[u]),
                    Iself_wide_out=float(Iself_wo[u]), Igate_wide_out=float(Igate_wo[u]),
                    Iself_narrow_out=float(Iself_no[u]), Igate_narrow_out=float(Igate_no[u]),
                ))
    return pd.DataFrame(rows), pd.DataFrame(s2_rows)


def check_s5(seeds, target_df):
    """S5: step 999,900–999,999 の毎 step 記録が境界窓として扱われていないことを確認する。

    この区間は switch_steps の最後の候補 (t=total_steps) の周りの記録だが、そこでは
    flip が起きない (ループが range(period, total) で終わるため) ので `find_boundaries`
    はそもそもここを境界として拾わない。ここでは (i) 全 seed で境界が 99 個ちょうど
    であること、(ii) 999,900–999,999 のどの step も検出済み境界 B の ±100 窓に
    入っていないこと、の両方を機械的に確認する。"""
    ok = True
    detail = []
    for d in seeds:
        seed = int(d["seed"])
        step = d["step"]
        idx_b, lefts, _ = find_boundaries(d, int(d["period"]))
        tail = step[(step >= 999900) & (step <= 999999)]
        if tail.size == 0:
            ok = False
            detail.append(dict(seed=seed, tail_points=0, min_dist_to_boundary=np.nan,
                               treated_as_boundary=False))
            continue
        dist = np.abs(tail[:, None] - lefts[None, :]).min(axis=1) if lefts.size else \
            np.full(tail.shape, np.inf)
        min_dist = float(dist.min())
        treated = bool((dist <= 100).any())
        ok &= (min_dist > 100) and not treated
        detail.append(dict(seed=seed, tail_points=int(tail.size),
                           min_dist_to_boundary=min_dist, treated_as_boundary=treated))
    return bool(ok), pd.DataFrame(detail)


# ------------------------------------------------------------------ 集計ヘルパ

def per_seed_stat(df, seed_order, statfn):
    """seed_order (run_id ソート順) に沿って、seed ごとに statfn(部分df) を評価する。"""
    return np.array([statfn(df[df.seed == s]) for s in seed_order], dtype=float)


def safe(fn, sub):
    return float(fn(sub)) if len(sub) else np.nan


# ------------------------------------------------------------------ L1–L8 (spec §5)

def compute_ledger(target_df, dead_j, seed_order, rng):
    """spec §5 の L1–L8 (+ 副次 L1-win) を W-wide / W-narrow の両方で計算する。

    target_df: 全対象 (境界で alive な全ユニット事件)。
    dead_j   : target_df のうち dead な行に events.csv の path を突合した部分集合。"""
    V = []
    per_seed_cols = {}

    def track(name, vals):
        per_seed_cols[name] = vals
        return vals

    def add(id_, statistic, group, window, vals_for_boot, agg, smoke_key=None,
            note=""):
        pt, lo, hi = agg(rng, vals_for_boot, BOOT_N)
        smoke = SMOKE.get(smoke_key, np.nan) if (smoke_key and window == "W-wide") else np.nan
        V.append(dict(id=id_, statistic=statistic, group=group, window=window,
                      point=pt, ci_lo=lo, ci_hi=hi, smoke=smoke,
                      n_seed=int(np.sum(np.isfinite(vals_for_boot))), note=note))

    surv_df = target_df[~target_df.dead]
    dead_all = dead_j                      # dead 全群 (輸送死+再分類死), path 付き

    for window in ("W-wide", "W-narrow"):
        gate = f"Igate_{'wide' if window == 'W-wide' else 'narrow'}"
        selff = f"Iself_{'wide' if window == 'W-wide' else 'narrow'}"
        restt = f"Irest_{'wide' if window == 'W-wide' else 'narrow'}"

        # ---- L1 主判定: 死亡個体 (窓内正味が下向き) の ∫F_self/∫F_gate の中央値
        groups3 = [("全群", lambda df: df), ("輸送死", lambda df: df[df.path == "transport"]),
                  ("再分類死", lambda df: df[df.path == "reclass"])]
        for gname, sel in groups3:
            def statfn(sub, sel=sel, gate=gate, selff=selff):
                g = sel(sub)
                g = g[g[gate] < 0]
                if len(g) == 0:
                    return np.nan
                return float(np.median(g[selff] / g[gate]))
            vals = track(f"L1_{gname}_{window}_ratio_median", per_seed_stat(dead_all, seed_order, statfn))
            add("L1", "死亡個体(窓内正味が下向き)の ∫F_self/∫F_gate の中央値", gname, window,
                vals, boot_ci_median, ("L1", gname),
                note="判定なし・点推定とCIを報告。比が1を超えてもクリップしない")

        # ---- L1-win (副次: 死亡が窓内で起きた個体に限定。判定に使わない)
        we = WIN_END[window]
        for gname, sel in groups3:
            def statfn(sub, sel=sel, gate=gate, selff=selff, we=we):
                g = sel(sub)
                g = g[(g.step_prev - g.B) <= we]           # 死亡 (最初のクロス) が窓内
                g = g[g[gate] < 0]
                if len(g) == 0:
                    return np.nan
                return float(np.median(g[selff] / g[gate]))
            vals = track(f"L1win_{gname}_{window}_ratio_median", per_seed_stat(dead_all, seed_order, statfn))
            add("L1-win", "死亡が窓内で起きた個体に限定した L1 (副次・判定に使わない)",
                gname, window, vals, boot_ci_median, None,
                note="§9 の選択: 主判定はタスク全体死亡、L1-win は窓内死亡限定")

        # ---- L2: 死亡個体のうち窓内正味が下向きの割合
        for gname, sel in groups3:
            def statfn(sub, sel=sel, gate=gate):
                g = sel(sub)
                return safe(lambda x: (x[gate] < 0).mean(), g)
            vals = track(f"L2_{gname}_{window}_frac_neg", per_seed_stat(dead_all, seed_order, statfn))
            add("L2", "死亡個体のうち窓内正味が下向きの割合", gname, window, vals, boot_ci,
                ("L2", gname))

        # ---- L3: ∫F_self/∫F_rest/∫F_gate の平均 (死亡3群 + 生存)
        groups4 = groups3 + [("生存", None)]
        for gname, sel in groups4:
            src = surv_df if gname == "生存" else sel(dead_all)
            for stat_name, col in (("self", selff), ("rest", restt), ("gate", gate)):
                def statfn(sub, col=col):
                    return safe(lambda x: x[col].mean(), sub)
                vals = track(f"L3_{gname}_{window}_{stat_name}_mean",
                            per_seed_stat(src, seed_order, statfn))
                add("L3", f"∫F_{stat_name} の平均", gname, window, vals, boot_ci,
                    ("L3", gname, stat_name))

        # ---- L4: 死亡個体のうち稼ぎ自体が負 (∫F_rest<0) の割合 (同上 4 群)
        for gname, sel in groups4:
            src = surv_df if gname == "生存" else sel(dead_all)
            def statfn(sub, restt=restt):
                return safe(lambda x: (x[restt] < 0).mean(), sub)
            vals = track(f"L4_{gname}_{window}_frac_rest_neg", per_seed_stat(src, seed_order, statfn))
            add("L4", "稼ぎ自体が負 (∫F_rest<0) の割合", gname, window, vals, boot_ci,
                ("L4", gname))

        # ---- L5: 稼ぎを失って税に沈んだ (∫F_rest>=0 かつ ∫F_gate<0) の割合 (同上 4 群)
        for gname, sel in groups4:
            src = surv_df if gname == "生存" else sel(dead_all)
            def statfn(sub, restt=restt, gate=gate):
                return safe(lambda x: ((x[restt] >= 0) & (x[gate] < 0)).mean(), sub)
            vals = track(f"L5_{gname}_{window}_frac_earn_ge0_gate_neg",
                        per_seed_stat(src, seed_order, statfn))
            add("L5", "稼ぎを失って税に沈んだ (∫F_rest>=0 かつ ∫F_gate<0) の割合", gname, window,
                vals, boot_ci, ("L5", gname))

        # ---- L6: 生存個体のタスク残り区間 [B+2000,B+1e4) の ∫F_self/∫F_rest
        #     (定義が窓に依存しないので W-wide/W-narrow で同一値。両方の行に出す)
        for stat_name, col, smk in (("self", "remain_self", "self"), ("rest", "remain_rest", "rest")):
            def statfn(sub, col=col):
                return safe(lambda x: x[col].mean(), sub)
            vals = track(f"L6_{window}_{stat_name}_mean", per_seed_stat(surv_df, seed_order, statfn))
            add("L6", "生存個体のタスク残り区間 [B+2000,B+1e4) の ∫F の平均 (粗いグリッド)",
                "生存", window, vals, boot_ci, ("L6", "生存", smk),
                note="窓の定義に依存しない量。W-wide/W-narrow の両行に同一値")

        # ---- L7: 窓内/窓外の ΣF_gate・ΣF_self (台形積分) と正味が負の割合 (全対象)
        suf = "wide" if window == "W-wide" else "narrow"
        for side, gcol, scol in (("窓内", gate, selff),
                                 ("窓外", f"Igate_{suf}_out", f"Iself_{suf}_out")):
            for stat_name, col, agg, smk in (
                    ("gate", gcol, boot_ci, "gate"), ("self", scol, boot_ci, "self"),
                    ("frac_neg", gcol, boot_ci, "frac_neg")):
                if stat_name == "frac_neg":
                    def statfn(sub, col=col):
                        return safe(lambda x: (x[col] < 0).mean(), sub)
                else:
                    def statfn(sub, col=col):
                        return safe(lambda x: x[col].mean(), sub)
                vals = track(f"L7_{side}_{window}_{stat_name}", per_seed_stat(target_df, seed_order, statfn))
                label = ("窓内/窓外の ΣF_gate (正味)" if stat_name == "gate" else
                        "窓内/窓外の ΣF_self (税)" if stat_name == "self" else
                        "窓内/窓外で正味 (ΣF_gate) が負な割合")
                add("L7", label, "全対象", window, vals, agg, ("L7", side, smk),
                    note=f"{side}。台形積分 (L1/L3/L6/L8 と同じ演算)")

        # ---- L8: 正味変位に占める窓内の割合 (全対象・pooled ratio)
        def statfn(sub, gate=gate):
            num = sub[gate].sum()
            den = sub["Igate_full"].sum()
            return float(num / den) if den != 0 else np.nan
        vals = track(f"L8_{window}_frac_window", per_seed_stat(target_df, seed_order, statfn))
        add("L8", "正味変位 (∫F_gate) に占める窓内の割合 (seed内 pooled 比)", "全対象", window,
            vals, boot_ci, ("L8",), note="pooled: Σ窓内積分 / Σ全タスク積分 (seedごと)")

    return pd.DataFrame(V), per_seed_cols


# ------------------------------------------------------------------ 図

def make_figures(outdir, target_df, dead_j):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) 群別 ∫F_self vs ∫F_rest 散布 (W-wide, 死亡個体)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for path, c, lab in (("transport", "tab:blue", "輸送死"), ("reclass", "tab:orange", "再分類死")):
        g = dead_j[dead_j.path == path]
        ax.scatter(g.Irest_wide, g.Iself_wide, s=6, alpha=.35, color=c,
                  label=f"{lab} (n={len(g)})")
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("∫F_rest (稼ぎ, W-wide)")
    ax.set_ylabel("∫F_self (税, W-wide)")
    ax.set_title("死亡個体: 群別の ∫税 / ∫稼ぎ 散布 (W-wide)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_scatter_self_rest_by_path.png"), dpi=140)
    plt.close(fig)

    # 2) L1 (∫F_self/∫F_gate) の分布 (W-wide, 窓内正味が下向きの死亡個体)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    g = dead_j[dead_j.Igate_wide < 0].copy()
    g["ratio"] = g.Iself_wide / g.Igate_wide
    bins = np.linspace(-1, 3, 81)
    for path, c, lab in (("transport", "tab:blue", "輸送死"), ("reclass", "tab:orange", "再分類死")):
        v = g[g.path == path].ratio.values
        if v.size:
            ax.hist(v, bins=bins, alpha=.55, color=c, density=True,
                   label=f"{lab} (n={v.size}, med={np.median(v):.3f})")
    ax.axvline(0.245, color="k", ls="--", lw=1.2, label="スモーク中央値 0.245")
    ax.set_xlabel("∫F_self / ∫F_gate (W-wide)")
    ax.set_ylabel("密度")
    ax.set_title("L1: 群別の分布 (窓内正味が下向きの死亡個体)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_L1_distribution.png"), dpi=140)
    plt.close(fig)
    return fig_dir


# ------------------------------------------------------------------ summary.md

def _md(df, cols, fmt=".4f"):
    f = lambda v: (format(v, fmt) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                  else ("" if isinstance(v, float) and not np.isfinite(v) else str(v)))
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(outdir, V, s2_df, s5_df, s5_ok, s3_missing, ev_key_dup,
                  n_target, n_dead, n_surv, meta):
    L = ["# ledger_0822: 力の台帳の本番組み直し (群別・Q2b)", "",
         "仕様: `specs/spec_ledger_0822.md` (実行前に commit 済み = 事前登録)。生成: "
         f"`{meta['date']}`、git `{meta['git_hash']}`。**再学習なし** — "
         "`results/ratchet_log_0819/logs/seed*.npz` と `results/dead2path_0821/events.csv` "
         "のみを読む事後解析。", "",
         "## 0. 一行", "",
         "境界窓で税 (F_self) と稼ぎ (F_rest) を積分し、死亡個体を再分類死/輸送死に "
         "分けて、死亡個体の沈下に税が占める割合の中央値 (L1) を seed 単位ペア "
         "ブートストラップ CI 付きで報告する。判定は置かない (spec §5)。", "",
         "## 1. サニティ", "",
         f"**S1**: OMP_NUM_THREADS={meta['omp_num_threads']}、"
         f"python {meta['python']} / numpy {meta['numpy']} / pandas {meta['pandas']}。", "",
         "**S2** (境界を flip_state の差分から機械的に決定。ハードコードしない): "
         f"{'PASS' if bool(s2_df.ok.all()) else '**FAIL**'} — 全 {len(s2_df)} seed で "
         "99/99 が step ≡ 0 (mod 10⁴) かつ隣接。", "",
         _md(s2_df, ["seed", "run_id", "n_boundaries", "n_aligned", "n_adjacent", "ok"]), "",
         f"**S3** (events.csv とのキー一意性・死亡個体の全数ラベル付け): "
         f"{'PASS' if (ev_key_dup == 0 and s3_missing == 0) else '**FAIL**'} — "
         f"events.csv 内の (seed, unit, step_prev) 重複 {ev_key_dup} 件、"
         f"ラベル未突合の死亡個体 {s3_missing} 件 (対象 n_dead={n_dead})。", "",
         "**S4** (群別件数の照合。結果ではなく整合性の確認): 本 spec が使う "
         "`results/dead2path_0821/events.csv` は 2026-08-22 に commit 済みコード "
         "(`db3ff77`) を再走して再生成したもので、`死の二経路.md` と一致する値と "
         "一致しない値がある (spec §9)。§5 逸脱節に §9 の一致/不一致表をそのまま "
         "転記する。本 spec が使うのは「一致」範囲 (降下窓 766/596/170、母集団 995、"
         "PE・E1 の全数値、キー一意性) だけで、「不一致」の 5 項目は一切参照しない。", "",
         f"**S5** (step 999,900–999,999 の毎 step 記録が境界窓として扱われていないこと): "
         f"{'PASS' if s5_ok else '**FAIL**'} — 全 seed で最近傍の検出済み境界までの "
         "距離が 100 step 超 (= どの境界窓にも属さない)。", "",
         _md(s5_df, ["seed", "tail_points", "min_dist_to_boundary", "treated_as_boundary"]), "",
         "**S6** (陣営 sign(v_i) 別の集計が出力に含まれていないこと): PASS — "
         "`verdict.csv` / `per_seed_metrics.csv` の全列名を機械的に走査し、"
         "`camp` / `陣営` / `sign_v` / `sign(v)` を含む列が無いことを確認した "
         "(実行後に本モジュール自身がチェックし、見つかれば例外で中止する)。", "",
         f"母集団: 対象 (境界で alive) {n_target} 件、うち死亡 {n_dead} 件・"
         f"生存 {n_surv} 件 (10 seed x 99 境界 x 100 unit = 99,000 が上限)。", "",
         "## 2. L1 (主判定・判定なし)", "",
         _md(V[(V.id == "L1")][["group", "window", "point", "ci_lo", "ci_hi", "smoke", "n_seed"]],
             ["group", "window", "point", "ci_lo", "ci_hi", "smoke", "n_seed"]), "",
         "点推定と 95%CI を報告するのみ (判定なし)。スモーク値 (R=8・100k・9 境界) は "
         "W-wide の「全群」行にのみ併記する — 輸送死/再分類死の群分けはスモークに無い "
         "本 spec の新規列であり、比較対象が無い。**比が 1 を超えてもクリップしていない**。", "",
         "## 3. L2–L8", ""]
    for lid, title in (("L2", "L2: 窓内正味が下向きの割合"),
                       ("L3", "L3: ∫F_self/∫F_rest/∫F_gate の平均"),
                       ("L4", "L4: 稼ぎ自体が負の割合"),
                       ("L5", "L5: 稼ぎを失って税に沈んだ割合"),
                       ("L6", "L6: 生存個体のタスク残り区間の平均"),
                       ("L7", "L7: 窓内/窓外のΣ (台形積分) と正味が負の割合"),
                       ("L8", "L8: 正味変位に占める窓内の割合")):
        sub = V[V.id == lid]
        L += [f"### {title}", "",
             _md(sub[["group", "window", "statistic", "point", "ci_lo", "ci_hi", "smoke"]],
                 ["group", "window", "statistic", "point", "ci_lo", "ci_hi", "smoke"]), ""]
    L += ["## 4. L1-win (副次・判定に使わない)", "",
         "spec §9 の選択: 主判定 L1 は「タスク全体 [B, B+10⁴) で死んだ個体」を母集団に "
         "取る。「窓内 (死亡の最初のクロスの step_prev が window_end 以下) で死んだ個体」"
         "に限定した版を L1-win として副次に併記する。判定には使わない。", "",
         _md(V[V.id == "L1-win"][["group", "window", "point", "ci_lo", "ci_hi", "n_seed"]],
             ["group", "window", "point", "ci_lo", "ci_hi", "n_seed"]), "",
         "## 5. 逸脱節", "",
         "1. **§9 前提の来歴 (events.csv)**。`results/dead2path_0821/` は起草時点で "
         "ディスク上にも git 履歴上にも存在しなかった。commit 済みの "
         "`analysis/dead2path/dead2path.py` (`db3ff77`) を既定引数で再走して再生成した "
         "(2026-08-22、`git ba02104`)。再走の S2 は 10 seed で 99/99 PASS、内蔵 S3 も PASS。", "",
         "   **一致 (本 spec が使う範囲)**", "",
         "   | 項目 | 値 |", "   |---|---|",
         "   | median PE（分離前） | `0.39592990555641694` 全桁一致 |",
         "   | E1 self / rest / 比 | `1.0` / `0.8059758` / `0.6176014` 全桁一致 |",
         "   | D2（輸送死）／帰無／n | `0.375432` / `0.2085` / `596` |",
         "   | 再分類死・輸送増分のみ／n | `0.3068` / `170` |",
         "   | 分離前後差 | `−0.020498` |",
         "   | 全窓 / 輸送死 / 再分類死 | `766` / `596` / `170` |",
         "   | 一度でも死んだユニット | `995` |",
         "   | キー一意性 | `(seed, unit, pair_index)` 一意、再分類死の "
         "`(seed, unit, 境界 index)` 一意・境界 1–99 |", "",
         "   **不一致 (本 spec は使わない)**", "",
         "   | 項目 | `死の二経路.md` | 再走 |", "   |---|---|---|",
         "   | 初回死が再分類のユニット数 | 281（0.282） | **252（0.2533）** |",
         "   | D0 恒久死 n / r | 32 / 0.313 [0.086, 0.593] | **942 / 0.1428 "
         "[0.1179, 0.1687]** |",
         "   | D1 対照 q95 以下の割合 | 96.6% | **94.28%** |",
         "   | D4 境界跨ぎ増分 | 16.1% | **15.9031%** |",
         "   | §3 再分類死の窓・全増分 | 0.4843 | **その指標を出力しない** |", "",
         "   2026-08-22 の決定: 再走の値を正として扱い、台帳側の訂正は人間側の作業として "
         "分離する。本 spec は上表の「一致」範囲のみを使い、「不一致」の 5 項目を一切 "
         "参照しない。", "",
         "2. **§9 の S3 曖昧性への選択**: vault `中和と境界窓.md` §4 の「そのタスク中に "
         "死んだ個体」は「窓 [B+1,B+2000] 内で死んだ個体」とも読める。本 spec は "
         "**タスク全体 [B, B+10⁴) で死んだ個体**を主判定の母集団に採り、窓内死亡に "
         "限った版を **L1-win として副次**に併記した (判定に使わない)。この選択は "
         "実行前に固定した。", "",
         "3. **S1 の窓の状況**: スモークの窓 [B+1, B+2000] は本番の記録グリッドでは "
         "毎 step 再現できない (記録間隔は §2 の表の通り、(B+100, B+1000] に 1 点、"
         "(B+1000, B+2000] に 1 点しかない)。W-wide は事前登録どおりスモークと同じ "
         "**窓定義** (積分区間) を維持するが、**窓内の記録点数はスモークの fidelity と "
         "異なる** — (B+100, B+2000] の区間は粗点 2 点だけで代表される。W-narrow "
         "[B+1, B+100] はこの粗さを含まない対照として副次的に報告する。", "",
         "4. **群の列構成の解釈**: spec §5 の表で L2・L4・L5 が「同上」と書く参照先を、"
         "直前の行の群構成として読んだ — L2 は L1 (全群/輸送死/再分類死の 3 群、"
         "生存は対象外)、L4・L5 は L3 (全群/輸送死/再分類死/生存の 4 群) を継承する。"
         "L4・L5 の統計量の文言は「死亡個体のうち」だが、生存群にも同じ判定条件 "
         "(∫F_rest<0 等) を単純に適用して報告した。L7・L8 は「全対象」(alive-at-B の "
         "全個体、死亡/生存を区別しない) の 1 列のみ。", "",
         "5. **L6 は窓に依存しない**: 定義域 [B+2000, B+10⁴) が W-wide/W-narrow どちらの "
         "窓とも独立なので、両方の行に同一の値を出している。", "",
         "6. **L7 の Σ は台形積分として実装した**: vault §5 の表記は L1/L3/L6 の ∫ と "
         "異なり Σ を使っているが、F_self 等は 1 SGD step あたりの期待増分であり、"
         "記録点をΔt 非加重で単純加算すると (B+100,B+2000] の粗点 2 点の寄与が過小に "
         "なる。spec §3 が言う「(B+100,B+2000] が積分の 94.7%/94.0% を占める」という "
         "診断は台形積分 (Δt 加重) でのみ再現できる (本番データで自己再現: "
         "94.6%/94.9%、seed0 のみ)。Δt 非加重の生和ではこの診断値を再現できないため、"
         "L7 も L1/L3/L6/L8 と同じ台形積分で実装した。", "",
         "7. **L8 は seed 内 pooled 比**: 「正味変位に占める窓内の割合」は "
         "dead2path D4 の `frac_abs_dcos_boundary` と同じ作法 (seed 内で Σ窓内積分 / "
         "Σ全タスク積分の pooled 比を取ってから seed 間ブートストラップ) で計算した。", "",
         "8. **Q2b / 穴A スコープ**: 陣営 (sign(v_i)) 別の符号分布、相殺比 "
         "|F_gate|/|F_self| の境界前後プロファイルは spec §1 の禁止により出力していない。"
         "S6 でこれを機械的に確認している。", "",
         "## 6. スコープ・禁止事項 (spec §8 の転記)", "",
         "- スコープ: **condA・w100・T=1e4・batch=1・`ratchet_log_0819` のログ限定**。"
         "condB へ外挿しない", "- 再学習しない", "- 穴A (Q17) に踏み込まない",
         "- `surv_hist_0822` と相互参照して解釈を調整しない (独立に判定する)",
         "- 台帳の代数 (縮退・増幅・二速修理) を混ぜない",
         "- スモーク値と並べるときは「スモーク（R=8, 100k）」「本番（10 seed, 1M）」を "
         "明記し、差の解釈は書かない (本 summary もこれに従う)",
         "- vault を書き換えない (反映は人間側の作業)", ""]
    p = os.path.join(outdir, "summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return p


# ------------------------------------------------------------------ main

def check_s6(verdict_df, per_seed_df):
    banned = ["camp", "陣営", "sign_v", "sign(v)"]
    cols = list(verdict_df.columns) + list(per_seed_df.columns) + \
        [str(x) for x in verdict_df.get("group", pd.Series(dtype=object)).unique()] + \
        [str(x) for x in verdict_df.get("statistic", pd.Series(dtype=object)).unique()]
    hit = [c for c in cols for b in banned if b.lower() in str(c).lower()]
    if hit:
        raise SystemExit(f"[ledger] S6 FAIL — 陣営/camp 関連の列/値が見つかった: {hit}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(ROOT, "results", "ratchet_log_0819"))
    ap.add_argument("--events", default=os.path.join(ROOT, "results", "dead2path_0821",
                                                      "events.csv"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "ledger_0822"))
    args = ap.parse_args()

    t0 = time.time()
    resdir = args.results
    cfg = load_config(os.path.join(resdir, "config_used.yaml"))
    period = int(cfg["condA"]["T_values"][0])

    seeds = load_seeds(resdir)
    seeds.sort(key=lambda d: str(d["run_id"]))          # run_id ソートで seed をペアリング [§7]
    seed_order = [int(d["seed"]) for d in seeds]
    print(f"loaded {len(seeds)} seeds, {len(seeds[0]['step'])} 記録点, period={period}",
         flush=True)

    target_df, s2_df = build_target_table(seeds, period)
    print("S2:", "PASS" if bool(s2_df.ok.all()) else "FAIL", flush=True)
    print(s2_df.to_string(index=False), flush=True)
    if not bool(s2_df.ok.all()):
        raise SystemExit("[ledger] S2 FAIL — 境界の同定が壊れている。中止 (spec §6 S2)")

    s5_ok, s5_df = check_s5(seeds, target_df)
    print("S5:", "PASS" if s5_ok else "FAIL", flush=True)
    if not s5_ok:
        raise SystemExit("[ledger] S5 FAIL — 999,900-999,999 が境界窓として混入している。中止")

    ev = pd.read_csv(args.events)
    ev_key_dup = int(ev.duplicated(subset=["seed", "unit", "step_prev"]).sum())
    if ev_key_dup:
        raise SystemExit(f"[ledger] S3 FAIL — events.csv の (seed,unit,step_prev) に "
                         f"{ev_key_dup} 件の重複。中止 (spec §6 S3)")

    dead_df = target_df[target_df.dead].copy()
    surv_df_n = int((~target_df.dead).sum())
    dead_j = dead_df.merge(ev[["seed", "unit", "step_prev", "path"]],
                           on=["seed", "unit", "step_prev"], how="left")
    s3_missing = int(dead_j.path.isna().sum())
    print(f"S3: events.csv key dup={ev_key_dup}, ラベル未突合={s3_missing} "
         f"(対象死亡 n={len(dead_j)})", flush=True)
    if s3_missing:
        raise SystemExit(f"[ledger] S3 FAIL — {s3_missing} 件の死亡個体に群ラベルが "
                         "付かない。中止 (spec §6 S3)")

    rng = np.random.default_rng(BOOT_SEED)
    V, per_seed_cols = compute_ledger(target_df, dead_j, seed_order, rng)

    os.makedirs(args.outdir, exist_ok=True)
    per_seed_df = pd.DataFrame({"seed": seed_order,
                                "run_id": [str(d["run_id"]) for d in seeds]})
    for k, v in per_seed_cols.items():
        per_seed_df[k] = v

    check_s6(V, per_seed_df)          # S6: 陣営/camp 列が無いことを機械的に確認

    V.to_csv(os.path.join(args.outdir, "verdict.csv"), index=False)
    per_seed_df.to_csv(os.path.join(args.outdir, "per_seed_metrics.csv"), index=False)
    fig_dir = make_figures(args.outdir, target_df, dead_j)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
               spec="specs/spec_ledger_0822.md", source=resdir, events=args.events,
               device="cpu", elapsed_sec=round(time.time() - t0, 1),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
               rng_seed=BOOT_SEED,
               python=platform.python_version(), numpy=np.__version__,
               pandas=pd.__version__, bootstrap_B=BOOT_N, n_seeds=len(seeds), period=period,
               n_target=int(len(target_df)), n_dead=int(len(dead_j)), n_survivor=surv_df_n,
               s2_pass=bool(s2_df.ok.all()), s5_pass=bool(s5_ok),
               s3_key_dup=ev_key_dup, s3_missing=s3_missing)
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)

    sp = write_summary(args.outdir, V, s2_df, s5_df, s5_ok, s3_missing, ev_key_dup,
                       len(target_df), len(dead_j), surv_df_n, meta)

    print(V[V.id == "L1"].to_string(index=False), flush=True)
    print(f"-> {args.outdir}/verdict.csv, per_seed_metrics.csv, {sp}, {fig_dir}/", flush=True)
    print("LEDGER DONE", flush=True)
    return V, per_seed_df


if __name__ == "__main__":
    main()
