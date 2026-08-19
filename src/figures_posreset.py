"""posreset_0819 の解析・判定・図 (spec_posreset_0819 §5–6, §10)。

  python -m src.figures_posreset [results/posreset_0819]
  python -m src.figures_posreset --selftest      # 合成フィクスチャで統計・判定を自己検証

ランナー (`src/posreset.py`) が書いた成果物を読むだけで **再学習は一切しない**。
出力は runs.csv / verdict.csv / summary.md / figures/ の 4 点 [posreset_0819 §10]。

判定は **clean eval_loss のみ**。dead_frac は PASS/FAIL のいかなる経路にも入れない
[posreset_0819 §5, §9]。統計は §6 で凍結された paired seed bootstrap
(rng = default_rng(20260819), B=10,000, percentile 95%CI) で、同一 replicate の
seed 添字ベクトルを全アームに適用することでペアリングを成立させる。

読むファイル (FROZEN INTERFACE CONTRACT):
  lop_metrics_{gbase}_{cont,none,posonly,dironly,full}.csv
  intervention_log.csv   (regime × seed: treated_frac, ガード件数, S3 の数値)
  unit_traj_{regime}_{seed}_{arm}.npz  (treated ユニットの p̂ / ‖w‖ / β / cos)
  meta.json              (S2 resume bit 一致ほか)
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- FROZEN INTERFACE CONTRACT ------------------------------------------------
ARMS = ["none", "posonly", "dironly", "full"]      # ブランチ 4 アーム [§3.4]
CONT = "cont"                                       # トランク兼参照 (0→1M 連続 run)
GBASE = {"A": "A_w100", "B": "B_w20"}               # group_name(gkey)
REGIMES = ["A", "B"]
ARM_COLOR = {"none": "tab:gray", "posonly": "tab:blue", "dironly": "tab:orange",
             "full": "tab:red", CONT: "black"}
ARM_LS = {"none": "-", "posonly": "-", "dironly": "--", "full": "-"}

# ---- 凍結された統計手続き [§6] -------------------------------------------------
BOOT_SEED = 20260819
N_BOOT = 10000
LATE_SPAN = 100000        # M_late の窓幅 [§5]
P6_OFFSET = 100000        # P6 のゲート再開率を測る step オフセット [§6 P6]

# config が読めなかった場合の既定値 (FROZEN INTERFACE CONTRACT の posreset ブロック)
DEFAULT_PR = dict(t_int=500000, post_steps=500000, p_hat_tau=0.05, probe_every=10000,
                  treated_frac_min=0.3, treated_frac_min_seeds=8)

SEC9_FALLBACK = """- dead_frac に基づくいかなる判定・主張（clean eval_loss のみ）
- P1–P4 が全て PASS しても「B1 確定」（E2 の燃料溶接・E3 の組合せ代数・E4 の基準対決が残る）
- 「新特徴は無用」への飛躍（B の等価性は弱い証拠。強い証拠は A の P4 のみ）
- CBP（継続適用）・K=10⁴・実スケール・Transformer への外挿
- 忘却側（OP10）への言及"""


# ================================================================ 読み込み

def load_pr_cfg(resdir):
    """posreset ブロック (t_int / post_steps / p̂ 閾値 …) の解決順:
    results/config_used.yaml → configs/posreset_0819.yaml → intervention_log の
    t_int → 既定値。窓の定義が実行時設定とずれると M が別物になるので、
    どこから採ったかを必ず summary に記録する。"""
    src = "default"
    cfg = {}
    for cand, tag in [(os.path.join(resdir, "config_used.yaml"), "results/config_used.yaml"),
                      (os.path.join(ROOT, "configs", "posreset_0819.yaml"),
                       "configs/posreset_0819.yaml")]:
        if os.path.exists(cand):
            with open(cand) as fh:
                y = yaml.safe_load(fh) or {}
            if isinstance(y.get("posreset"), dict):
                cfg = dict(y["posreset"])
                cfg["lop_every"] = (y.get("common") or {}).get("lop_every")
                cfg["seeds"] = (y.get("common") or {}).get("seeds")
                src = tag
                break
    out = dict(DEFAULT_PR)
    out.update({k: v for k, v in cfg.items() if v is not None})
    out["cfg_source"] = src
    return out


def _split_arm(rid):
    """run_id -> (base_run_id, arm)。arm 接尾辞は既知の 5 種のみ [contract]。"""
    for a in ARMS + [CONT]:
        if rid.endswith("_" + a):
            return rid[: -(len(a) + 1)], a
    return rid, "unknown"


def _seed_of(base_run_id):
    m = re.search(r"_s(\d+)$", str(base_run_id))
    return int(m.group(1)) if m else -1


def load_lop(resdir):
    """全レジーム × 全アームの lop_metrics を縦結合。eval_loss だけを使う。"""
    frames, missing = [], []
    for regime, gbase in GBASE.items():
        for arm in [CONT] + ARMS:
            p = os.path.join(resdir, f"lop_metrics_{gbase}_{arm}.csv")
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
                continue
            df = pd.read_csv(p)
            df["regime"] = regime
            df["gbase"] = gbase
            ba = df.run_id.map(_split_arm)
            df["base_run_id"] = [b for b, _ in ba]
            df["arm"] = [a for _, a in ba]
            # ファイル名の arm と run_id 接尾辞の arm は一致するはず (contract)
            df.loc[df.arm == "unknown", "arm"] = arm
            df["seed"] = df.base_run_id.map(_seed_of)
            frames.append(df[["regime", "gbase", "arm", "base_run_id", "seed",
                              "step", "eval_loss"]])
    if not frames:
        raise SystemExit(f"lop_metrics_*.csv が {resdir} に無い (ランナー未完了?)")
    return pd.concat(frames, ignore_index=True), missing


def load_ilog(resdir):
    p = os.path.join(resdir, "intervention_log.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    il = pd.read_csv(p)
    if "regime" in il.columns:
        il["regime"] = il.regime.astype(str)
    return il


def load_meta(resdir):
    p = os.path.join(resdir, "meta.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def load_unit_traj(resdir, regime, seed, arm):
    p = os.path.join(resdir, f"unit_traj_{regime}_{seed}_{arm}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ================================================================ 主指標 M / M_late

def _nanmean(v):
    v = np.asarray(v, dtype=np.float64)
    fin = np.isfinite(v)
    return float(v[fin].mean()) if fin.any() else np.nan


def arm_metrics(lop, t_int, post):
    """(regime, arm, base_run_id) ごとの M / M_late [§5]。

    M      = 窓 [t_int, t_int+post] (両端含む) の clean eval_loss 平均
    M_late = 窓末尾 100k: t_int+post−100000 < step ≤ t_int+post
    発散 run の NaN 行は **黙って落とさず** 件数を n_nan_points に残し、平均は
    有限行のみで取る (全行 NaN なら M=NaN → 以降のペアから除外して件数を報告)。"""
    end = t_int + post
    rows = []
    for (regime, arm, base), g in lop.groupby(["regime", "arm", "base_run_id"]):
        w = g[(g.step >= t_int) & (g.step <= end)]
        lt = g[(g.step > end - LATE_SPAN) & (g.step <= end)]
        ev, evl = w.eval_loss.values, lt.eval_loss.values
        rows.append(dict(regime=regime, arm=arm, base_run_id=base, seed=_seed_of(base),
                         M=_nanmean(ev), M_late=_nanmean(evl),
                         n_eval_points=int(len(ev)),
                         n_nan_points=int((~np.isfinite(ev)).sum()),
                         n_late_points=int(len(evl)),
                         n_nan_late=int((~np.isfinite(evl)).sum())))
    return pd.DataFrame(rows).sort_values(["regime", "arm", "base_run_id"]).reset_index(drop=True)


def t_int_row_effect(lop, t_int, n_pts):
    """M 窓の左端 (step == t_int) の非対称性を数値化する [§5]。

    ランナーは介入直後に t_int の lop 行を書くので、**none アームの t_int 行は介入前の値**
    (cont と一致し S2 で検証される) だが **reset 3 アームの t_int 行は介入直後の値**
    (v[treated]=0 直後のスパイク)。窓が閉区間 [t_int, t_int+post] である以上 §5 のとおりで
    あり除外はしないが、Δ_arm に 1/n_pts の系統差が入るので必ず数値で併記する。

    3 つの reset アームは t_int で同一値 (a←0 が共通) になるため差は共通の下方シフト b<0 と
    して入り、G0/P1/P2/P4 は **保守側** (通りにくくなる)、P5 は不偏、**P3 だけは
    0.25·Δ_full − Δ_dironly = 真値 + 0.75|b| で通りやすくなる (反保守)**。"""
    out = {}
    for regime in REGIMES:
        sub = lop[(lop.regime == regime) & (lop.step == t_int)]
        if not len(sub):
            continue
        lv = sub.groupby("arm").eval_loss.mean()
        if "none" not in lv.index:
            continue
        for arm in ARMS:
            if arm == "none" or arm not in lv.index:
                continue
            out[(regime, arm)] = dict(at_t_int=float(lv[arm]), none_at_t_int=float(lv["none"]),
                                      delta_bias=-(float(lv[arm]) - float(lv["none"]))
                                      / max(int(n_pts), 1) + 0.0)   # −0.0 表示を潰す
    return out


def build_runs_table(mtab, ilog):
    """runs.csv: アーム別の M/M_late に intervention_log (regime×seed) を join。"""
    keep = ["regime", "seed", "n_treated", "treated_frac", "n_guard_fallback",
            "base_run_id"]
    if len(ilog):
        cols = [c for c in keep if c in ilog.columns]
        il = ilog[cols].copy()
        il = il.rename(columns={"base_run_id": "ilog_base_run_id"})
        out = mtab.merge(il, on=["regime", "seed"], how="left")
    else:
        out = mtab.copy()
        for c in ["n_treated", "treated_frac", "n_guard_fallback"]:
            out[c] = np.nan
    cols = ["regime", "seed", "arm", "base_run_id", "treated_frac", "n_treated",
            "n_guard_fallback", "M", "M_late", "n_eval_points", "n_nan_points"]
    extra = [c for c in ["n_late_points", "n_nan_late"] if c in out.columns]
    return out[cols + extra].sort_values(["regime", "seed", "arm"]).reset_index(drop=True)


# ================================================================ paired seed bootstrap [§6]

class PairedBoot:
    """§6 で凍結された手続き: rng は **1 個だけ** 作り、B=10,000 replicate の
    seed 添字行列を (regime, metric) ブロックごとに 1 回だけ引いて、そのブロックの
    全アーム・全派生統計に同じ添字を使い回す (= ペアリングの実体)。

    引く順序は REGIMES × METRICS の固定ループなので、verdict 表全体が
    default_rng(20260819) から完全再現できる。"""

    def __init__(self, seed=BOOT_SEED, n_boot=N_BOOT):
        self.rng = np.random.default_rng(seed)
        self.n_boot = n_boot
        self._idx = {}

    def index(self, key, n):
        if key not in self._idx:
            self._idx[key] = self.rng.integers(0, n, size=(self.n_boot, n))
        idx = self._idx[key]
        if idx.shape[1] != n:
            raise ValueError(f"boot block {key}: n が {idx.shape[1]} から {n} に変わった")
        return idx

    def reps(self, key, vals):
        """replicate ごとの seed 平均 [B]。"""
        v = np.asarray(vals, dtype=np.float64)
        return v[self.index(key, len(v))].mean(axis=1)

    def ci(self, key, vals):
        """点推定 (実データの seed 平均) と percentile 95%CI。"""
        v = np.asarray(vals, dtype=np.float64)
        if len(v) < 2:
            return dict(point=float(v.mean()) if len(v) else np.nan,
                        lo=np.nan, hi=np.nan, n=len(v))
        bm = self.reps(key, v)
        return dict(point=float(v.mean()), lo=float(np.quantile(bm, 0.025)),
                    hi=float(np.quantile(bm, 0.975)), n=len(v))


def ratio_ci(boot, key, num, den):
    """比 mean(num)/mean(den) の bootstrap CI。分母が 0 を跨ぐと比の CI は発散して
    無意味になるので、**分母が正の replicate が 95% を超えるときだけ** CI を返す
    (results/coupling_fbw_0813 の振幅比で確立した家内規約)。"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    dm, nm = float(den.mean()), float(num.mean())
    out = dict(point=nm / dm if dm != 0 else np.nan, lo=np.nan, hi=np.nan,
               frac_pos_den=np.nan, reported=False)
    if len(den) < 2:
        return out
    idx = boot.index(key, len(den))
    dr, nr = den[idx].mean(axis=1), num[idx].mean(axis=1)
    pos = dr > 0
    out["frac_pos_den"] = float(pos.mean())
    if pos.mean() > 0.95:
        rb = nr[pos] / dr[pos]
        out.update(lo=float(np.quantile(rb, 0.025)), hi=float(np.quantile(rb, 0.975)),
                   reported=True)
    return out


def delta_arrays(mtab, regime, metric):
    """Δ_arm = M(none) − M(arm)。ペアは **base_run_id のソート順** で取る [§6]。

    どれか 1 アームでも M が非有限 (完全発散) の seed は、その (regime, metric)
    ブロックの全統計から一括除外する (ブロック内でペアの seed 集合を揃えないと
    replicate 添字を共有できないため)。除外件数は verdict の note に出す。"""
    sub = mtab[(mtab.regime == regime) & (mtab.arm.isin(ARMS))]
    piv = sub.pivot(index="base_run_id", columns="arm", values=metric).sort_index()
    for a in ARMS:
        if a not in piv.columns:
            piv[a] = np.nan
    piv = piv[ARMS]
    ok = np.isfinite(piv.values).all(axis=1)
    used, dropped = piv.index[ok], list(piv.index[~ok])
    none = piv.loc[used, "none"].values.astype(float)
    return dict(regime=regime, metric=metric, n=int(ok.sum()),
                seeds=[_seed_of(b) for b in used], dropped=dropped,
                none_level=none,
                delta={a: none - piv.loc[used, a].values.astype(float)
                       for a in ARMS if a != "none"})


# ================================================================ 副次指標 (P6 / P7)

def reopen_frac(resdir, regime, seeds, arm, tau, step_target):
    """P6: treated ユニットのうち step_target で p̂ > tau のもの割合 (seed 別)。"""
    vals, notes = [], []
    for s in seeds:
        z = load_unit_traj(resdir, regime, s, arm)
        if z is None or "p_hat" not in z:
            vals.append(np.nan)
            continue
        steps = np.asarray(z["steps"]).astype(np.int64)
        hit = np.where(steps == step_target)[0]
        if len(hit) == 0:                       # 格子上に無ければ最近傍 (逸脱として記録)
            j = int(np.argmin(np.abs(steps - step_target)))
            notes.append(f"{regime}/s{s}/{arm}: step {step_target} が probe 格子に無く "
                         f"{int(steps[j])} で代用")
        else:
            j = int(hit[0])
        p = np.asarray(z["p_hat"], dtype=np.float64)[j]
        # 非有限の p̂ は `p > tau` が黙って False になり再開率を過小評価する。
        # 分母から外して件数を逸脱として記録する [§5 副次]
        fin = np.isfinite(p)
        if p.size and not fin.all():
            notes.append(f"{regime}/s{s}/{arm}: p̂ が非有限のユニット "
                         f"{int((~fin).sum())}/{int(p.size)} を再開率の分母から除外")
        vals.append(float(np.mean(p[fin] > tau)) if fin.any() else np.nan)
    return np.asarray(vals, float), notes


def dcos_median(resdir, regime, seeds, arm, t_int, end):
    """P7: seed ごとに treated ユニット上の median Δcos(u, µ̂)
    = cos@(t_int+post) − cos@t_int。"""
    vals = []
    for s in seeds:
        z = load_unit_traj(resdir, regime, s, arm)
        if z is None or "cos_u_mu" not in z:
            vals.append(np.nan)
            continue
        steps = np.asarray(z["steps"]).astype(np.int64)
        c = np.asarray(z["cos_u_mu"], dtype=np.float64)
        i0 = int(np.argmin(np.abs(steps - t_int)))
        i1 = int(np.argmin(np.abs(steps - end)))
        d = c[i1] - c[i0]
        d = d[np.isfinite(d)]
        vals.append(float(np.median(d)) if d.size else np.nan)
    return np.asarray(vals, float)


# ================================================================ 判定 [§6]

def _vrow(id_, regime, statistic, point, lo, hi, threshold, result, note):
    return dict(id=id_, regime=regime, statistic=statistic, point=point,
                ci_lo=lo, ci_hi=hi, threshold=threshold, result=result, note=note)


def _fmt(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{n}g}"


def eligibility(ilog, pr):
    """§3.4 事前登録の適格性: treated_frac ≥ 0.3 が 10 seed 中 8 以上。
    満たさないレジームは主判定を **参考格** に降格する (void ではない)。"""
    out = {}
    for regime in REGIMES:
        if not len(ilog) or "treated_frac" not in ilog.columns:
            out[regime] = dict(n_ok=0, n=0, ok=None)
            continue
        g = ilog[ilog.regime == regime]
        n_ok = int((g.treated_frac >= pr["treated_frac_min"]).sum())
        out[regime] = dict(n_ok=n_ok, n=int(len(g)),
                           ok=bool(n_ok >= pr["treated_frac_min_seeds"]))
    return out


def verdicts(mtab, resdir, ilog, pr, boot):
    """G0 / P1–P7 と、その M_late 版 (id 接尾辞 "_late") を verdict.csv 用に組む。

    - 主判定は M。M_late 版は §5 の併記要求を満たすための追加行で、判定の格は同じだが
      「主判定は M」と note に明記する。
    - G0 が落ちたレジームの P 行は全て void [§6]。
    - 適格性 (§3.4) 未達のレジームは全行に「参考格」を付す。"""
    t_int, post = pr["t_int"], pr["post_steps"]
    metrics = ["M", "M_late"]
    da = {(r, m): delta_arrays(mtab, r, m) for r in REGIMES for m in metrics}
    # replicate 添字は (regime, metric) ごとに 1 回だけ、この固定順で引く
    for r in REGIMES:
        for m in metrics:
            if da[(r, m)]["n"] >= 2:
                boot.index((r, m), da[(r, m)]["n"])

    elig = eligibility(ilog, pr)
    rows, notes = [], []

    def suffix(regime, metric, d):
        s = f"n_seed={d['n']}"
        if d["dropped"]:
            s += f", M 非有限で除外 {len(d['dropped'])} seed ({', '.join(d['dropped'])})"
        if elig.get(regime, {}).get("ok") is False:
            s += (f"; **参考格** (§3.4: treated_frac≥{pr['treated_frac_min']} が "
                  f"{elig[regime]['n_ok']}/{elig[regime]['n']} seed で "
                  f"{pr['treated_frac_min_seeds']} 未満)")
        if metric == "M_late":
            s += "; 主判定は M (本行は §5 の併記)"
        return s

    # ---- G0 (前提ゲート): Δ_full > 0
    g0 = {}
    for regime in REGIMES:
        for metric in metrics:
            d = da[(regime, metric)]
            key = (regime, metric)
            c = boot.ci(key, d["delta"]["full"]) if d["n"] >= 2 else dict(
                point=np.nan, lo=np.nan, hi=np.nan, n=d["n"])
            ok = bool(np.isfinite(c["lo"]) and c["lo"] > 0)
            g0[key] = ok
            # ペア可能な seed が 2 本未満 / 点推定が非有限 = **データ不足**。
            # 「Δ_full > 0 が示せなかった」という科学的 FAIL と混同させない
            # (下流は結局 void になるが、verdict.csv の result 列を読む人に嘘をつかない)。
            insufficient = (d["n"] < 2) or not np.isfinite(c["point"])
            rows.append(_vrow("G0" + ("_late" if metric == "M_late" else ""), regime,
                              f"Δ_full = {metric}(none) − {metric}(full)",
                              c["point"], c["lo"], c["hi"], "CI 下限 > 0",
                              "NA (データ不足)" if insufficient else ("PASS" if ok else "FAIL"),
                              f"前提ゲート。{metric}(none) 平均 "
                              f"{_fmt(float(np.mean(d['none_level'])) if d['n'] else np.nan)}; "
                              + suffix(regime, metric, d)))

    def voided(regime, metric):
        return not (g0.get((regime, "M"), False) and g0.get((regime, metric), False))

    def emit(id_, regime, metric, statistic, c, threshold, result, note):
        if voided(regime, metric):
            result = "void"
            note = "G0 不成立のため void (点推定は参考値)。" + note
        rows.append(_vrow(id_ + ("_late" if metric == "M_late" else ""), regime,
                          statistic, c["point"], c["lo"], c["hi"], threshold,
                          result, note))

    for metric in metrics:
        dB, dA = da[("B", metric)], da[("A", metric)]
        keyB, keyA = ("B", metric), ("A", metric)
        nan_ci = dict(point=np.nan, lo=np.nan, hi=np.nan, n=0)

        # ---- P1 (B): Δ_posonly > 0
        c = boot.ci(keyB, dB["delta"]["posonly"]) if dB["n"] >= 2 else nan_ci
        p1 = bool(np.isfinite(c["lo"]) and c["lo"] > 0)
        emit("P1", "B", metric, f"Δ_posonly ({metric})", c, "CI 下限 > 0",
             "PASS" if p1 else "FAIL",
             "座標完全復元 (‖w‖ と b) が単独で効くか。FAIL なら B1 棄却の主成分。"
             + suffix("B", metric, dB))

        # ---- P2 (B, 主判定): Δ_posonly ≥ 0.5·Δ_full
        if dB["n"] >= 2:
            s05 = dB["delta"]["posonly"] - 0.5 * dB["delta"]["full"]
            c2 = boot.ci(keyB, s05)
        else:
            c2 = nan_ci
        if np.isfinite(c2["lo"]) and c2["lo"] > 0:
            r2 = "PASS"
        elif np.isfinite(c2["point"]) and c2["point"] > 0:
            r2 = "weak PASS"
        else:
            r2 = "FAIL"
        emit("P2", "B", metric, f"Δ_posonly − 0.5·Δ_full ({metric})", c2,
             "CI 下限 > 0 で PASS / 点推定のみ正で weak PASS", r2,
             "**主判定**。FAIL なら混合説 (座標＋特徴鮮度) へ改訂。" + suffix("B", metric, dB))

        # ---- P2 「強」併記: Δ_posonly ≥ 0.75·Δ_full
        if dB["n"] >= 2:
            c3 = boot.ci(keyB, dB["delta"]["posonly"] - 0.75 * dB["delta"]["full"])
        else:
            c3 = nan_ci
        strong = bool(np.isfinite(c3["lo"]) and c3["lo"] > 0)
        emit("P2_strong", "B", metric, f"Δ_posonly − 0.75·Δ_full ({metric})", c3,
             "CI 下限 > 0 なら「強」", ("PASS (強)" if strong else "not 強"),
             ("**強**: 座標復元だけで full の 75% 超を再現。" if strong
              else "75% 水準は CI 下限が 0 を超えず「強」は付かない。")
             + suffix("B", metric, dB))

        # ---- P2 の生比 Δ_posonly/Δ_full (分母が 0 を跨ぐと CI 不安定)
        if dB["n"] >= 2:
            rc = ratio_ci(boot, keyB, dB["delta"]["posonly"], dB["delta"]["full"])
        else:
            rc = dict(point=np.nan, lo=np.nan, hi=np.nan, frac_pos_den=np.nan,
                      reported=False)
        emit("P2_ratio", "B", metric, f"Δ_posonly / Δ_full ({metric})",
             dict(point=rc["point"], lo=rc["lo"], hi=rc["hi"]), "参考値 (閾値なし)",
             "report",
             (f"分母が正の bootstrap 標本 {_fmt(rc['frac_pos_den'], 3)} "
              + ("> 0.95 のため CI を報告 (coupling_fbw_0813 の家内規約)。"
                 if rc["reported"] else
                 "≤ 0.95 のため **CI は報告しない** (分母が 0 を跨ぐと比の CI は発散する。"
                 "coupling_fbw_0813 の家内規約)。"))
             + suffix("B", metric, dB))

        # ---- P3 (B): Δ_dironly ≤ 0.25·Δ_full
        if dB["n"] >= 2:
            c4 = boot.ci(keyB, 0.25 * dB["delta"]["full"] - dB["delta"]["dironly"])
        else:
            c4 = nan_ci
        p3 = bool(np.isfinite(c4["lo"]) and c4["lo"] > 0)
        emit("P3", "B", metric, f"0.25·Δ_full − Δ_dironly ({metric})", c4,
             "CI 下限 > 0", "PASS" if p3 else "FAIL",
             "b が深い負のままゲートが開かない所で新方向が効いてはいけない。"
             "FAIL は **β/ゲート機構への警報**。" + suffix("B", metric, dB))

        # ---- P4 (A): Δ_posonly > 0
        c5 = boot.ci(keyA, dA["delta"]["posonly"]) if dA["n"] >= 2 else nan_ci
        p4 = bool(np.isfinite(c5["lo"]) and c5["lo"] > 0)
        emit("P4", "A", metric, f"Δ_posonly ({metric})", c5, "CI 下限 > 0",
             "PASS" if p4 else "FAIL",
             "**H_feat の主戦場**: 新特徴ゼロで µ 経路の便益が出るか。"
             "FAIL は B1 の棄却ではなく改訂 (操舵単独では不十分)。" + suffix("A", metric, dA))

        # ---- P5 (A): Δ_full − Δ_posonly (report only)
        if dA["n"] >= 2:
            c6 = boot.ci(keyA, dA["delta"]["full"] - dA["delta"]["posonly"])
        else:
            c6 = nan_ci
        sign = "正 (full が優位 = マージン寄与あり)" if (np.isfinite(c6["point"])
                                                        and c6["point"] > 0) \
            else ("負 (posonly が優位)" if np.isfinite(c6["point"]) else "NA")
        emit("P5", "A", metric, f"Δ_full − Δ_posonly ({metric})", c6,
             "報告のみ (PASS/FAIL なし)", "report",
             f"マージン寄与の分解量。符号: {sign}。CI は記述用。" + suffix("A", metric, dA))

    # ---- P6 (B, 副次): ゲート再開率 @ t_int+100k、点推定順序のみ
    seedsB = da[("B", "M")]["seeds"]
    rf, nts = {}, []
    for arm in ARMS:
        rf[arm], n = reopen_frac(resdir, "B", seedsB, arm, pr["p_hat_tau"],
                                 t_int + P6_OFFSET)
        nts += n
    notes += nts
    for arm in ["posonly", "dironly"]:
        v = rf[arm]
        emit("P6_" + arm, "B", "M", f"reopen_frac({arm}) @ t_int+100k",
             dict(point=_nanmean(v), lo=np.nan, hi=np.nan), "報告のみ", "report",
             f"treated のうち p̂ > {pr['p_hat_tau']} の割合 (seed 平均)。"
             f"seed 別 {np.round(v, 3).tolist()}")
    dif = rf["posonly"] - rf["dironly"]
    fin = np.isfinite(dif)
    cd = boot.ci(("B", "M"), dif) if (fin.all() and len(dif) >= 2) else dict(
        point=_nanmean(dif), lo=np.nan, hi=np.nan)
    order = ("posonly > dironly (予測どおり)" if np.isfinite(cd["point"]) and cd["point"] > 0
             else ("posonly ≤ dironly (予測と逆)" if np.isfinite(cd["point"]) else "NA"))
    emit("P6", "B", "M", "reopen_frac(posonly) − reopen_frac(dironly) @ t_int+100k", cd,
         "点推定の順序のみ (PASS/FAIL なし)", "report",
         f"順序: {order}。posonly {_fmt(_nanmean(rf['posonly']))} / "
         f"dironly {_fmt(_nanmean(rf['dironly']))} / none {_fmt(_nanmean(rf['none']))} / "
         f"full {_fmt(_nanmean(rf['full']))}。CI は記述用 (事前登録は点推定のみ)。"
         "機構署名の不発は判定に波及しない。")

    # ---- P7 (A, 副次): posonly-treated の median Δcos(u, µ̂)
    seedsA = da[("A", "M")]["seeds"]
    dc = dcos_median(resdir, "A", seedsA, "posonly", t_int, t_int + post)
    fin = np.isfinite(dc)
    c7 = boot.ci(("A", "M"), dc) if (fin.all() and len(dc) >= 2) else dict(
        point=_nanmean(dc), lo=np.nan, hi=np.nan)
    emit("P7", "A", "M", "median_unit Δcos(u, µ̂) (posonly, t_int → t_int+post)", c7,
         "点推定のみ (PASS/FAIL なし)", "report",
         f"seed 別 median {np.round(dc, 4).tolist()}。正なら二段回復署名 "
         "(操舵回復 → u が +µ̂ 半空間へ)。CI は記述用。")

    ver = pd.DataFrame(rows, columns=["id", "regime", "statistic", "point", "ci_lo",
                                      "ci_hi", "threshold", "result", "note"])
    return ver, da, elig, notes, dict(reopen=rf, dcos=dc)


def conclusion_text(ver, elig):
    """§6 の帰結マッピングを機械的に適用した一行結論。"""
    def res(i, r):
        s = ver[(ver.id == i) & (ver.regime == r)].result
        return s.iloc[0] if len(s) else "NA"

    def pt(i, r):
        s = ver[(ver.id == i) & (ver.regime == r)].point
        return float(s.iloc[0]) if len(s) else np.nan

    g0a, g0b = res("G0", "A"), res("G0", "B")
    p1, p2, p3, p4 = res("P1", "B"), res("P2", "B"), res("P3", "B"), res("P4", "A")
    bits = [f"G0: A {g0a} / B {g0b}", f"P1(B) {p1}", f"P2(B) {p2}", f"P3(B) {p3}",
            f"P4(A) {p4}"]
    maps = []
    if g0b == "PASS" and p1 == "FAIL":
        maps.append("**B1 棄却**: 座標を完全復元しても効かない = 回復変数は座標ではない")
    if p1 == "PASS" and p2 == "FAIL":
        maps.append("**混合説へ改訂**: 座標は必要だが特徴鮮度も寄与する")
    if p4 == "FAIL" and p1 == "PASS":
        maps.append("**B1 改訂**: マージン必須・操舵は補助")
    if p3 == "FAIL":
        maps.append("**ゲート理論の見直しを最優先課題に昇格** (P3 FAIL)")
    # §6 の帰結マッピングは「P1–P4 PASS」としか書いておらず、P2 の 弱 PASS は
    # PASS とも FAIL とも定義されていない。弱 PASS を無条件に §11 (最強の主張) へ
    # 流すのは事前登録の過剰読みなので、留保を明示して分ける。
    if p1 == "PASS" and p2 == "PASS" and p3 == "PASS" and p4 == "PASS":
        maps.append("§11 の主張へ (ただし「確定」とは書かない。E2–E4 が残る)")
    elif p1 == "PASS" and p2 == "weak PASS" and p3 == "PASS" and p4 == "PASS":
        maps.append("P1・P3・P4 PASS だが **P2 は弱 PASS (点推定のみ正、CI は 0 を跨ぐ)**。"
                    "§6 の帰結マッピングは PASS/FAIL の 2 値しか定義していないため、"
                    "§11 の主張は **留保つき** (「full の 50% 以上」は点推定水準の支持に"
                    "とどまり、CI では支持されていない)")
    if not maps:
        maps.append("§6 の帰結マッピングに該当する組合せなし (個別行を参照)")
    ratio = pt("P2_ratio", "B")
    if np.isfinite(ratio):
        bits.append(f"Δ_posonly/Δ_full = {ratio:.2f}")
    for r in REGIMES:
        if elig.get(r, {}).get("ok") is False:
            maps.append(f"レジーム {r} は §3.4 適格性未達のため **参考格**")
    return " / ".join(bits) + " → " + "；".join(maps)


# ================================================================ サニティ S1–S4

# step == t_int の境界行だけは resume 側に「前の lop 行」が無いため NaN になる派生量。
# compute_b_metrics が prev_dead / prev_pzero を跨いで作る列で、cont 側は 1.0 を持つ。
# 再開の bit 一致とは無関係なので境界行に限り比較から外す (ランナーの s2_compare と同じ扱い)。
PERSIST_COLS = ("dead_persist_frac", "p_zero_persist_frac")


def compare_cont_vs_none(resdir, regime, t_int):
    """S2 の独立再検査: cont (連続 run) と none アーム (snapshot からの無介入 resume) の
    lop_metrics を step ≥ t_int で **文字列レベル** 比較 (rank_int の compare_logs 踏襲)。

    注意 2 点 (どちらも実データで誤 FAIL を出す):
      1. `keep_default_na=False` が必須。既定では "nan" トークンが float NaN に変換され、
         NaN != NaN で **両側が同じ NaN の列が全行不一致** と判定される。dead≈0.99 の
         本走では b_mean_alive / beta_p10 / stable_rank_W_alive などが両側 NaN になる。
      2. 境界行 (step == t_int) の PERSIST_COLS は原理的に一致しない (上のコメント)。"""
    gbase = GBASE[regime]
    fa = os.path.join(resdir, f"lop_metrics_{gbase}_{CONT}.csv")
    fb = os.path.join(resdir, f"lop_metrics_{gbase}_none.csv")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return "NA (ファイルなし)"
    a = pd.read_csv(fa, dtype=str, keep_default_na=False)
    b = pd.read_csv(fb, dtype=str, keep_default_na=False)
    for d, arm in [(a, CONT), (b, "none")]:
        d["run_id"] = d.run_id.str.replace(f"_{arm}$", "", regex=True)
    a = a[a.step.astype(int) >= t_int].reset_index(drop=True)
    b = b[b.step.astype(int) >= t_int].reset_index(drop=True)
    if len(a) != len(b):
        return f"FAIL (行数 {len(a)} vs {len(b)})"
    common = [c for c in a.columns if c in b.columns]
    ne = (a[common] != b[common])
    bnd = a.step.astype(int) == t_int
    for c in PERSIST_COLS:
        if c in ne.columns:
            ne.loc[bnd, c] = False
    neq = int(ne.any(axis=1).sum())
    if neq == 0:
        return "PASS"
    bad = [c for c in common if ne[c].any()]
    return f"FAIL ({neq}/{len(a)} 行が不一致; 列 {bad[:6]})"


def _as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes", "pass"])


def sanity(resdir, ilog, meta, pr, mtab):
    """S1–S4 を集約。S2/S4 はランナー申告に加えて **こちら側で独立に再計算** する。"""
    s = {}
    s["S1_omp_analysis"] = os.environ.get("OMP_NUM_THREADS", "(未設定)")
    s["S1_omp_runner"] = str(meta.get("omp_num_threads", meta.get("OMP_NUM_THREADS", "(meta に記録なし)")))
    s["S2_runner"] = json.dumps(meta.get("sanity", meta.get("per_regime", "(meta に記録なし)")),
                                ensure_ascii=False, default=str)
    s["S2_recheck"] = {r: compare_cont_vs_none(resdir, r, pr["t_int"]) for r in REGIMES}

    if len(ilog):
        # S3 列の振り分けは**列名ではなく中身**で行う。ランナーは cos/ノルム誤差だけでなく
        # *_exact_* も float (厳密 0 が期待値) で書くので、名前で「論理」に振ると 0.0 が
        # False と解釈されて S3 が誤って FAIL 表示になる (実際に一度そうなった)。
        # 真の論理列は s3_*_ok / s3_pass のみで、いずれも bool か "True"/"False" 文字列。
        s3 = [c for c in ilog.columns if c.startswith("s3_")]
        def _is_logical(col):
            if ilog[col].dtype == bool:
                return True
            return bool(ilog[col].astype(str).str.strip().str.lower()
                        .isin(["true", "false"]).all())
        boolc = [c for c in s3 if _is_logical(c)]
        errc = [c for c in s3 if c not in boolc]
        # *_exact_* は許容ではなく厳密 0 が要求値なので別枠で判定する
        exactc = [c for c in errc if "_exact_" in c]
        tolc = [c for c in errc if c not in exactc]
        s["S3_max_f64"] = {c: float(pd.to_numeric(ilog[c], errors="coerce").abs().max())
                           for c in tolc if c.endswith("_f64")}
        s["S3_max_f32"] = {c: float(pd.to_numeric(ilog[c], errors="coerce").abs().max())
                           for c in tolc if c.endswith("_f32")}
        s["S3_exact_max"] = {c: float(pd.to_numeric(ilog[c], errors="coerce").abs().max())
                             for c in exactc}
        s["S3_exact_ok"] = bool(all(v == 0.0 for v in s["S3_exact_max"].values())) \
            if s["S3_exact_max"] else None
        s["S3_bool"] = {c: bool(_as_bool(ilog[c]).all()) for c in boolc}
        s["S3_pass_all"] = bool(all(s["S3_bool"].values())
                                and (s["S3_exact_ok"] is not False)) if s["S3_bool"] else None
        s["S3_tol_f64"] = pr.get("s3_tol_f64")
        if s.get("S3_tol_f64") is not None and s["S3_max_f64"]:
            s["S3_within_tol_f64"] = bool(max(s["S3_max_f64"].values()) < pr["s3_tol_f64"])

        # S4: treated hash が seed 内で 4 アーム一致 (npz の treated_hash / unit_idx を照合)
        bad, n_chk = [], 0
        for regime in REGIMES:
            g = ilog[ilog.regime == regime]
            for r in g.itertuples():
                ref, ridx = None, None
                for arm in ARMS:
                    z = load_unit_traj(resdir, regime, int(r.seed), arm)
                    if z is None:
                        continue
                    n_chk += 1
                    h = str(z["treated_hash"]) if "treated_hash" in z else None
                    idx = np.asarray(z["unit_idx"]) if "unit_idx" in z else None
                    if ref is None:
                        ref, ridx = h, idx
                        if h is not None and hasattr(r, "treated_hash") and \
                                str(r.treated_hash) != h:
                            bad.append(f"{regime}/s{r.seed}: npz hash ≠ ilog hash")
                    else:
                        if h != ref:
                            bad.append(f"{regime}/s{r.seed}/{arm}: treated_hash 不一致")
                        if idx is not None and ridx is not None and \
                                not np.array_equal(idx, ridx):
                            bad.append(f"{regime}/s{r.seed}/{arm}: unit_idx 不一致")
        s["S4_checked"] = n_chk
        s["S4_pass"] = bool(n_chk > 0 and not bad)
        s["S4_bad"] = bad[:20]
    # 発散 (NaN eval_loss) の集計 — 判定には使わないが必ず報告する
    nn = mtab.groupby(["regime", "arm"]).agg(n_nan=("n_nan_points", "sum"),
                                             n_pts=("n_eval_points", "sum"),
                                             n_run_all_nan=("M", lambda v: int((~np.isfinite(v)).sum())))
    s["NaN_by_regime_arm"] = nn.reset_index().to_dict("records")
    return s


# ================================================================ 図 [§10]

def fig_evalloss(resdir, lop, regime, pr, figdir):
    """(i) eval_loss 時系列。x=0 から (pre-t_int は cont)、4 アーム重ね、seed 帯
    (median + IQR)、t_int に vline。"""
    t_int, post = pr["t_int"], pr["post_steps"]
    sub = lop[lop.regime == regime]
    if not len(sub):
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    pre = sub[(sub.arm == CONT) & (sub.step <= t_int)]
    if len(pre):
        q = pre.groupby("step").eval_loss.quantile([0.25, 0.5, 0.75]).unstack()
        ax.plot(q.index, q[0.5], lw=1.3, color=ARM_COLOR[CONT], label="trunk (cont)")
        ax.fill_between(q.index, q[0.25], q[0.75], color=ARM_COLOR[CONT], alpha=0.15, lw=0)
    for arm in ARMS:
        g = sub[(sub.arm == arm) & (sub.step >= t_int) & (sub.step <= t_int + post)]
        if not len(g):
            continue
        q = g.groupby("step").eval_loss.quantile([0.25, 0.5, 0.75]).unstack()
        ax.plot(q.index, q[0.5], lw=1.3, color=ARM_COLOR[arm], ls=ARM_LS[arm], label=arm)
        ax.fill_between(q.index, q[0.25], q[0.75], color=ARM_COLOR[arm], alpha=0.18, lw=0)
    ax.axvline(t_int, color="black", lw=1.0, ls="--")
    ax.axvspan(t_int + post - LATE_SPAN, t_int + post, color="black", alpha=0.05, lw=0)
    lo = np.nanmin(sub.eval_loss.values) if len(sub) else 1.0
    if np.isfinite(lo) and lo > 0:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("clean eval_loss (seed median, IQR band)")
    ax.set_title(f"regime {regime} ({GBASE[regime]}): eval_loss by arm\n"
                 f"dashed = intervention at t_int={t_int}, shaded = M_late window",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_pr_evalloss_{regime}.png"), dpi=150)
    plt.close(fig)


def fig_gate_reopen(resdir, regime, seeds, pr, figdir):
    """(ii) treated ユニットの p̂ 再開曲線。上: p̂>τ の割合 (= P6 の量)、下: 平均 p̂。"""
    t_int, tau = pr["t_int"], pr["p_hat_tau"]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    drew = False
    for arm in ARMS:
        curves_f, curves_p, steps = [], [], None
        for s in seeds:
            z = load_unit_traj(resdir, regime, s, arm)
            if z is None or "p_hat" not in z:
                continue
            steps = np.asarray(z["steps"]).astype(np.int64)
            p = np.asarray(z["p_hat"], dtype=np.float64)
            if p.size == 0:
                continue
            curves_f.append((p > tau).mean(axis=1))
            curves_p.append(np.nanmean(p, axis=1))
        if not curves_f:
            continue
        drew = True
        for ax, cur in [(axes[0], np.array(curves_f)), (axes[1], np.array(curves_p))]:
            med = np.nanmedian(cur, axis=0)
            q1 = np.nanquantile(cur, 0.25, axis=0)
            q3 = np.nanquantile(cur, 0.75, axis=0)
            ax.plot(steps, med, lw=1.3, color=ARM_COLOR[arm], ls=ARM_LS[arm], label=arm)
            ax.fill_between(steps, q1, q3, color=ARM_COLOR[arm], alpha=0.18, lw=0)
    if not drew:
        plt.close(fig)
        return
    for ax in axes:
        ax.axvline(t_int, color="black", lw=1.0, ls="--")
        ax.axvline(t_int + P6_OFFSET, color="black", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"frac of treated units with p_hat > {tau}")
    axes[1].set_ylabel("mean p_hat over treated units")
    axes[1].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"regime {regime}: gate reopening of treated units\n"
                      f"seed median + IQR band; dashed = t_int, dotted = P6 at t_int+100k",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_pr_gate_reopen_{regime}.png"), dpi=150)
    plt.close(fig)


def fig_cos_A(resdir, seeds, pr, figdir):
    """(iii) レジーム A の cos(u, µ̂) 軌跡 (符号つき)。treated ユニットの median。"""
    t_int = pr["t_int"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    drew = False
    for arm in ARMS:
        curves, steps = [], None
        for s in seeds:
            z = load_unit_traj(resdir, "A", s, arm)
            if z is None or "cos_u_mu" not in z:
                continue
            c = np.asarray(z["cos_u_mu"], dtype=np.float64)
            if c.size == 0 or not np.isfinite(c).any():
                continue
            steps = np.asarray(z["steps"]).astype(np.int64)
            curves.append(np.nanmedian(c, axis=1))
        if not curves:
            continue
        drew = True
        cur = np.array(curves)
        med = np.nanmedian(cur, axis=0)
        ax.plot(steps, med, lw=1.3, color=ARM_COLOR[arm], ls=ARM_LS[arm], label=arm)
        ax.fill_between(steps, np.nanquantile(cur, 0.25, axis=0),
                        np.nanquantile(cur, 0.75, axis=0),
                        color=ARM_COLOR[arm], alpha=0.18, lw=0)
    if not drew:
        plt.close(fig)
        return
    ax.axvline(t_int, color="black", lw=1.0, ls="--")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("signed cos(u_i, mu_hat), median over treated units")
    ax.set_title("regime A: steering of treated units toward the +mu half-space\n"
                 "seed median + IQR band; dashed = t_int", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_pr_cos_A.png"), dpi=150)
    plt.close(fig)


def fig_forest(da, boot, figdir):
    """(iv) 全 Δ の forest plot (M / M_late の 2 面)。"""
    metrics = ["M", "M_late"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), squeeze=False)
    for j, metric in enumerate(metrics):
        ax = axes[0][j]
        items = []
        for regime in REGIMES:
            d = da[(regime, metric)]
            for arm in ["posonly", "dironly", "full"]:
                if d["n"] < 2:
                    continue
                c = boot.ci((regime, metric), d["delta"][arm])
                items.append((f"{regime}  Delta_{arm}", arm, c))
        ys = np.arange(len(items))[::-1]
        for y, (lbl, arm, c) in zip(ys, items):
            ax.errorbar(c["point"], y,
                        xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]],
                        fmt="o", ms=5, capsize=4, lw=1.6, color=ARM_COLOR[arm])
            ax.text(0.02, y + 0.15, lbl, transform=ax.get_yaxis_transform(),
                    fontsize=8, va="bottom")
        ax.set_ylim(-0.7, len(items) - 0.25)
        ax.axvline(0, color="gray", lw=0.9)
        ax.set_yticks([])
        ax.set_xlabel(f"Delta = {metric}(none) - {metric}(arm), 95% CI")
        ax.grid(alpha=0.3, axis="x")
        ax.set_title(metric)
    fig.suptitle("benefit of each reset arm: positive = better than none "
                 "(paired seed bootstrap, B=10000)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_pr_forest.png"), dpi=150)
    plt.close(fig)


# ================================================================ summary.md

def spec_sec9():
    """§9「主張してはいけないこと」を仕様ファイルから逐語で取り出す (drift 防止)。"""
    p = os.path.join(ROOT, "specs", "spec_posreset_0819.md")
    try:
        with open(p) as fh:
            txt = fh.read()
        i = txt.index("## 9. 主張してはいけないこと")
        j = txt.index("\n## ", i + 1)
        body = txt[i:j].split("\n", 1)[1].strip()
        return body if body else SEC9_FALLBACK
    except Exception:
        return SEC9_FALLBACK


def delta_table(da, boot):
    """Δ 表 (4 アーム × 2 レジーム、M と M_late、点推定 + 95%CI)。"""
    rows = []
    for regime in REGIMES:
        for arm in ARMS:
            row = dict(regime=regime, arm=arm)
            for metric in ["M", "M_late"]:
                d = da[(regime, metric)]
                if arm == "none":
                    row[f"{metric}(none)"] = (float(np.mean(d["none_level"]))
                                              if d["n"] else np.nan)
                    row[f"Δ_{metric}"] = 0.0
                    row[f"CI_{metric}"] = "— (基準アーム)"
                elif d["n"] >= 2:
                    c = boot.ci((regime, metric), d["delta"][arm])
                    row[f"{metric}(none)"] = float(np.mean(d["none_level"]))
                    row[f"Δ_{metric}"] = c["point"] + 0.0      # −0.0 表示を潰す
                    row[f"CI_{metric}"] = f"[{_fmt(c['lo'])}, {_fmt(c['hi'])}]"
                else:
                    row[f"{metric}(none)"] = np.nan
                    row[f"Δ_{metric}"] = np.nan
                    row[f"CI_{metric}"] = "NA"
            rows.append(row)
    cols = ["regime", "arm", "M(none)", "Δ_M", "CI_M", "M_late(none)", "Δ_M_late",
            "CI_M_late"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]].round(5)


def write_summary(resdir, ver, da, boot, elig, san, runs, ilog, pr, notes, missing,
                  t0=None):
    L = []
    L.append("# posreset_0819 summary (spec_posreset_0819 §6 事前登録判定)\n")
    L.append("同方向・小ノルムリセット判別 (2×2 要因)。レジーム A = condA A_w100 (µ 経路 dead)、"
             "レジーム B = cbp_harm routeK K=100 の w20 (b 経路 dead)。"
             f"t_int={pr['t_int']}、窓 [t_int, t_int+{pr['post_steps']}]。\n")
    L.append("**判定は clean eval_loss のみ。dead_frac は PASS/FAIL のいかなる経路にも入れない**"
             " [§5, §9]。統計は §6 凍結の paired seed bootstrap "
             f"(rng=default_rng({BOOT_SEED}), B={N_BOOT}, percentile 95%CI)。"
             "**「CI が 0 を除外」は、正が予測されている量については "
             "『CI 下限 > 0』と読む** (片側の読み。本 summary・verdict.csv 全体で同じ規約)。\n")

    # (1) 一行結論
    L.append("## 1. 一行結論\n")
    L.append(conclusion_text(ver, elig))

    # (2) G0
    L.append("\n## 2. G0 (前提ゲート: Δ_full > 0)\n")
    g0 = ver[ver.id.isin(["G0", "G0_late"])]
    L.append(g0[["id", "regime", "point", "ci_lo", "ci_hi", "result"]]
             .round(5).to_string(index=False))
    voided = [r for r in REGIMES
              if not len(ver[(ver.id == "G0") & (ver.regime == r) & (ver.result == "PASS")])]
    L.append("\n" + ("- 全レジームで G0 成立 → P 判定は有効。"
                     if not voided else
                     f"- **G0 不成立: レジーム {', '.join(voided)} の P 判定は全て void "
                     f"(記録して保留) [§6]**。"))

    # (3) Δ 表
    L.append("\n## 3. Δ 表 (4 アーム × 2 レジーム、M と M_late、点推定 + 95%CI)\n")
    L.append(delta_table(da, boot).to_string(index=False))
    L.append("\nΔ_arm = M(none) − M(arm)。**正 = そのアームが none より良い**。"
             "ペアは base_run_id ソート順の seed 対応 [§6]。")

    # (4) P 表
    L.append("\n## 4. P 表 (P1–P7、M 基準が主判定。_late は §5 の併記)\n")
    L.append(ver[["id", "regime", "statistic", "point", "ci_lo", "ci_hi", "result"]]
             .round(5).to_string(index=False))
    L.append("\n根拠と注記 (verdict.csv の note 列):\n")
    for r in ver.itertuples():
        L.append(f"- **{r.id}** ({r.regime}) [{r.result}] {r.statistic}: "
                 f"{_fmt(r.point)} CI [{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] "
                 f"— 閾値 {r.threshold}。{r.note}")

    # (5) treated_frac
    L.append("\n## 5. treated_frac 表 (§3.4 適格性)\n")
    if len(ilog):
        cols = [c for c in ["regime", "seed", "n_treated", "treated_frac",
                            "n_guard_fallback", "pre_dead_frac", "pre_eval_loss"]
                if c in ilog.columns]
        L.append(ilog[cols].sort_values(["regime", "seed"]).round(5).to_string(index=False))
        L.append("")
        for r in REGIMES:
            e = elig.get(r, {})
            L.append(f"- レジーム {r}: treated_frac ≥ {pr['treated_frac_min']} は "
                     f"{e.get('n_ok')}/{e.get('n')} seed → "
                     + ("適格 (主判定として扱う)" if e.get("ok") else
                        f"**{pr['treated_frac_min_seeds']} seed 未満 → 主判定を参考格に降格 "
                        f"[§3.4]**"))
        if "n_guard_fallback" in ilog.columns:
            L.append(f"- ガード発動 (‖w_i‖ < norm_guard で posonly→full にフォールバック): "
                     f"合計 {int(ilog.n_guard_fallback.sum())} 件 "
                     f"({(ilog.n_guard_fallback > 0).sum()} seed で発生) [§3.4]")
    else:
        L.append("intervention_log.csv が無い (ランナー未完了)。")

    # (6) サニティ
    L.append("\n## 6. サニティ S1–S4 (§7)\n")
    L.append(f"- **S1** (OMP_NUM_THREADS=1): 解析プロセス `{san['S1_omp_analysis']}` / "
             f"ランナー申告 `{san['S1_omp_runner']}`")
    L.append(f"- **S2** (resume bit 一致) ランナー申告: {san['S2_runner']}")
    L.append("- **S2 独立再検査** (cont と none アームの lop_metrics を step ≥ t_int で"
             "文字列比較): " + ", ".join(f"{k} {v}" for k, v in san["S2_recheck"].items()))
    if "S3_max_f64" in san:
        L.append(f"- **S3** (介入の数値保証) float64 最大誤差: {san['S3_max_f64']} "
                 f"(許容 {san.get('S3_tol_f64')}) → "
                 f"{'PASS' if san.get('S3_within_tol_f64') else '要確認'}")
        L.append(f"  - float32 丸め後 (学習再開に使う値、eps≈1.2e-7 律速): {san.get('S3_max_f32')}")
        L.append(f"  - 厳密一致列 (w_post ≡ g / ガード時の full フォールバック) の最大差: "
                 f"{san.get('S3_exact_max')} → "
                 f"{'PASS (全て厳密 0)' if san.get('S3_exact_ok') else '要確認'}")
        L.append(f"  - 論理判定 (a←0 / b 規約 / treated 外 hash 不変 / ランナー総合): "
                 f"{san.get('S3_bool')} → 総合 {san.get('S3_pass_all')}")
    if "S4_pass" in san:
        L.append(f"- **S4** (treated 集合の hash が seed 内 4 アームで一致): "
                 f"{'PASS' if san['S4_pass'] else 'FAIL'} "
                 f"({san['S4_checked']} npz を照合" +
                 (f"; 不一致 {san['S4_bad']}" if san.get("S4_bad") else "") + ")")
    L.append("- 発散 (eval_loss = NaN) の内訳 — **判定には使わないが必ず報告する**:\n")
    L.append(pd.DataFrame(san["NaN_by_regime_arm"]).to_string(index=False))

    # (7) 逸脱記録
    L.append("\n## 7. 逸脱記録\n")
    dev = []
    dev.append(f"- 窓・閾値の設定は `{pr['cfg_source']}` から解決した "
               f"(t_int={pr['t_int']}, post={pr['post_steps']}, "
               f"p_hat_tau={pr['p_hat_tau']}, probe_every={pr['probe_every']})")
    nan_runs = runs[(~np.isfinite(runs.M)) & (runs.arm.isin(ARMS))]
    if len(nan_runs):
        dev.append("- **発散により M が非有限**の run: "
                   + ", ".join(f"{r.regime}/s{r.seed}/{r.arm}" for r in nan_runs.itertuples())
                   + "。当該 seed は同レジームの paired bootstrap から一括除外し、"
                     "除外件数を verdict.csv の note に記載した (行ごとの黙殺はしない)")
    part = runs[(runs.n_nan_points > 0) & np.isfinite(runs.M) & runs.arm.isin(ARMS)]
    if len(part):
        dev.append("- 窓内に NaN 行を含むが有限な M を持つ run (有限行のみで平均): "
                   + ", ".join(f"{r.regime}/s{r.seed}/{r.arm} ({int(r.n_nan_points)}/"
                               f"{int(r.n_eval_points)} 行)" for r in part.itertuples()))
    # 行そのものが足りない run (打ち切り・欠損) は NaN 計数に一切現れないので別に検出する。
    # M が「同じ窓の平均」でなくなり、傾きのある系列では seed 間で直接バイアスになる。
    arms_only = runs[runs.arm.isin(ARMS)]
    if len(arms_only) and arms_only.n_eval_points.nunique() > 1:
        modal = int(arms_only.n_eval_points.mode().iloc[0])
        odd = arms_only[arms_only.n_eval_points != modal]
        dev.append(f"- **M 窓の eval 点数が不揃い (最頻 {modal} 点)** — 当該 run は途中打ち切り"
                   "/行欠損の疑いがあり、M が同一窓の平均になっていない (NaN 計数には現れない): "
                   + ", ".join(f"{r.regime}/s{r.seed}/{r.arm} {int(r.n_eval_points)}点"
                               for r in odd.itertuples()))
    if t0:
        dev.append("- M 窓の左端 (step=t_int) は **none では介入前・reset 3 アームでは介入直後**"
                   "の値 (ランナーが介入直後に t_int 行を書くため)。窓が閉区間 [t_int, "
                   "t_int+post] である以上 §5 のとおりで除外はしないが、この 1 行だけで "
                   "Δ_arm が受ける系統差 (seed 平均, 窓 "
                   f"{int(runs[runs.arm.isin(ARMS)].n_eval_points.max()) if len(arms_only) else 0}"
                   " 点で割った値): "
                   + ", ".join(f"{r}/{a} {_fmt(v['delta_bias'])}"
                               for (r, a), v in sorted(t0.items()))
                   + "。3 アーム共通の下方シフトなので G0/P1/P2/P4 は保守側、P5 は不偏、"
                     "**P3 のみ通りやすくなる向き**に働く")
    if missing:
        dev.append(f"- 欠落ファイル: {missing}")
    for n in notes:
        dev.append(f"- {n}")
    ratio_rows = ver[ver.id.str.startswith("P2_ratio")]
    for r in ratio_rows.itertuples():
        if not np.isfinite(r.ci_lo):
            dev.append(f"- {r.id}: 分母 (Δ_full) が bootstrap で 0 を跨ぐため比の CI は"
                       "報告しない (coupling_fbw_0813 の家内規約)")
    for r in REGIMES:
        if elig.get(r, {}).get("ok") is False:
            dev.append(f"- レジーム {r} は §3.4 の適格性 (treated_frac≥"
                       f"{pr['treated_frac_min']} が {pr['treated_frac_min_seeds']} seed 以上) "
                       "を満たさないため、当該レジームの主判定は参考格として報告する "
                       "(seed の除外はしない)")
    if len(dev) == 1:
        dev.append("- 事前登録手順からの逸脱なし")
    L += dev

    # (8) §9 逐語再掲
    L.append("\n## 8. 主張してはいけないこと (spec_posreset_0819 §9 の逐語再掲・厳守)\n")
    L.append(spec_sec9())

    with open(os.path.join(resdir, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# ================================================================ main

def analyse(resdir):
    pr = load_pr_cfg(resdir)
    lop, missing = load_lop(resdir)
    ilog = load_ilog(resdir)
    meta = load_meta(resdir)
    if len(ilog) and "t_int" in ilog.columns and ilog.t_int.nunique() == 1:
        pr["t_int"] = int(ilog.t_int.iloc[0])       # ランナーの実測値を優先
    obs_end = int(lop[lop.arm.isin(ARMS)].step.max()) if len(lop) else 0
    if obs_end and obs_end - pr["t_int"] != pr["post_steps"]:
        pr["post_steps"] = obs_end - pr["t_int"]
        pr["cfg_source"] += " (post_steps は観測データの最終 step から補正)"

    mtab = arm_metrics(lop, pr["t_int"], pr["post_steps"])
    runs = build_runs_table(mtab, ilog)
    runs.to_csv(os.path.join(resdir, "runs.csv"), index=False)

    boot = PairedBoot()
    ver, da, elig, notes, aux = verdicts(mtab, resdir, ilog, pr, boot)
    ver.to_csv(os.path.join(resdir, "verdict.csv"), index=False)

    san = sanity(resdir, ilog, meta, pr, mtab)
    figdir = os.path.join(resdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    for regime in REGIMES:
        fig_evalloss(resdir, lop, regime, pr, figdir)
        fig_gate_reopen(resdir, regime, da[(regime, "M")]["seeds"], pr, figdir)
    fig_cos_A(resdir, da[("A", "M")]["seeds"], pr, figdir)
    fig_forest(da, boot, figdir)
    _ar = runs[runs.arm.isin(ARMS)] if len(runs) else runs
    n_pts = int(_ar.n_eval_points.max()) if len(_ar) else 1
    t0 = t_int_row_effect(lop, pr["t_int"], n_pts)
    write_summary(resdir, ver, da, boot, elig, san, runs, ilog, pr, notes, missing, t0)
    return dict(runs=runs, verdict=ver, da=da, boot=boot, elig=elig, sanity=san,
                pr=pr, aux=aux, mtab=mtab)


# ================================================================ 自己検証 (--selftest)
#
# ランナーの本物の出力はまだ存在しないので、FROZEN INTERFACE CONTRACT に厳密に一致する
# 合成フィクスチャを **既知の真値つき** で作り、統計と判定ロジックが真値を復元することを
# 証明する。真値の作り方:
#   eval_loss(step) = level[regime][arm] + amp[arm]*z[seed] + tr*trend(step)
#   z[seed] = (seed − 4.5)/4.5 は 10 seed で総和 0、trend は M 窓で総和 0 の対称直線。
#   ⇒ M(regime, arm, seed) は level + amp*z に **厳密**一致し、seed 平均 Δ は真値に一致。

_FIX_LEVEL = {
    "main": {"A": {"none": 1.00, "posonly": 0.75, "dironly": 0.95, "full": 0.60},
             "B": {"none": 2.00, "posonly": 1.40, "dironly": 2.00, "full": 1.00}},
    # null: レジーム A の Δ_full を厳密に 0 にして G0 を落とす (void 伝播の検証)
    "null": {"A": {"none": 1.00, "posonly": 1.00, "dironly": 1.00, "full": 1.00},
             "B": {"none": 2.00, "posonly": 1.40, "dironly": 2.00, "full": 1.00}},
}
# amp は「Δ_posonly − 0.5·Δ_full」等の合成統計が偶然ゼロ分散にならないよう選ぶ
# (分散が消えると CI が退化して CI 側のロジックを検証できない)
_FIX_AMP = {"none": 0.020, "posonly": 0.030, "dironly": 0.025, "full": 0.045}
_FIX_TREND = 0.10
_FIX_SEEDS = list(range(10))
_FIX_TREATED = {"A": 80, "B": 15}      # width A=100 / B=20
_FIX_H = {"A": 100, "B": 20}
_FIX_P6 = {"none": 0.0, "posonly": 0.7, "dironly": 0.1, "full": 0.8}   # 概ねの再開率
_FIX_BASE = {"A": "A_w100_T10000_std_lr0.01", "B": "B_w20_K100_c0.0_lr0.01"}


def _fix_z(seed):
    return (seed - 4.5) / 4.5


def _fix_series(level, arm, seed, steps, t_int, post):
    """M 窓で平均が厳密に level + amp*z になる系列を作る (対称トレンド)。"""
    k = (steps - t_int) / max(post, 1)                     # 窓内 0..1
    trend = _FIX_TREND * (2 * k - 1.0)                     # 窓平均 0 (対称)
    return level + _FIX_AMP[arm] * _fix_z(seed) + trend


def make_fixture(outdir, scenario="main"):
    """contract どおりの results ディレクトリを合成し、真値の dict を返す。"""
    import hashlib
    os.makedirs(outdir, exist_ok=True)
    t_int, post, probe, tau = 500000, 500000, 10000, 0.05
    lop_every = 1000
    levels = _FIX_LEVEL["null" if scenario == "null" else "main"]
    truth = dict(t_int=t_int, post=post, level=levels, seeds=_FIX_SEEDS,
                 M={}, M_late={}, reopen={}, dcos={}, nan_points={})

    grid_arm = np.arange(t_int, t_int + post + 1, lop_every)          # 501 点
    grid_cont = np.concatenate([np.array([0]),
                                np.arange(lop_every, t_int + post + 1, lop_every)])
    late = grid_arm[(grid_arm > t_int + post - LATE_SPAN) & (grid_arm <= t_int + post)]

    for regime, gbase in GBASE.items():
        for arm in [CONT] + ARMS:
            steps = grid_cont if arm == CONT else grid_arm
            rows = []
            for seed in _FIX_SEEDS:
                base = f"{_FIX_BASE[regime]}_s{seed}"
                lvl = levels[regime]["none" if arm == CONT else arm]
                v = _fix_series(lvl, "none" if arm == CONT else arm, seed, steps,
                                t_int, post)
                if arm == CONT:
                    # 窓外 (step < t_int) は減衰する適当なトランク軌道
                    pre = steps < t_int
                    v = np.where(pre, lvl + 3.0 * np.exp(-steps / 150000.0), v)
                if scenario == "nan":
                    if (regime, seed, arm) == ("A", 7, "dironly"):
                        v = v.copy(); v[-50:] = np.nan
                    if (regime, seed, arm) == ("B", 9, "full"):
                        v = v.copy(); v[:] = np.nan
                if arm != CONT:
                    win = np.isfinite(v)
                    truth["M"][(regime, arm, seed)] = (float(v[win].mean())
                                                       if win.any() else np.nan)
                    lv = v[np.isin(steps, late)]
                    truth["M_late"][(regime, arm, seed)] = (
                        float(lv[np.isfinite(lv)].mean())
                        if np.isfinite(lv).any() else np.nan)
                    truth["nan_points"][(regime, arm, seed)] = int((~np.isfinite(v)).sum())
                for st, val in zip(steps, v):
                    rows.append((int(st), f"{base}_{arm}", val,
                                 0.5, 0.5))                # dead_frac / neg_gate_frac ダミー
            df = pd.DataFrame(rows, columns=["step", "run_id", "eval_loss",
                                             "dead_frac", "neg_gate_frac"])
            # 本物のログは write_logs が %.6g で書くが、ここは解析側の算術を検証したい
            # ので丸め誤差を持ち込まない (%.17g = float64 の往復無損失)
            df.to_csv(os.path.join(outdir, f"lop_metrics_{gbase}_{arm}.csv"),
                      index=False, float_format="%.17g")

    # ---- intervention_log.csv
    ilog = []
    for regime in REGIMES:
        for seed in _FIX_SEEDS:
            n_tr = _FIX_TREATED[regime]
            frac = n_tr / _FIX_H[regime]
            if scenario == "null" and regime == "A" and seed in (0, 1, 2, 3):
                n_tr, frac = 10, 0.10          # 適格性 (§3.4) を落とす seed
            mask = np.zeros(_FIX_H[regime], dtype=bool)
            mask[:n_tr] = True
            h = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            ilog.append(dict(regime=regime, exp=regime, width=_FIX_H[regime], seed=seed,
                             base_run_id=f"{_FIX_BASE[regime]}_s{seed}", t_int=t_int,
                             h=_FIX_H[regime], n_treated=n_tr, treated_frac=frac,
                             n_guard_fallback=0, treated_hash=h,
                             pre_dead_frac=frac, pre_eval_loss=levels[regime]["none"],
                             s3_posonly_cos_err_f64=1e-16, s3_posonly_norm_relerr_f64=2e-16,
                             s3_dironly_norm_relerr_f64=0.0, s3_full_exact_f64=True,
                             s3_posonly_cos_err_f32=6e-8, s3_posonly_norm_relerr_f32=9e-8,
                             s3_dironly_norm_relerr_f32=0.0, s3_full_exact_f32=True,
                             s3_readout_zero_ok=True, s3_bias_ok=True,
                             s3_untreated_hash_ok=True, s3_pass=True))
    ilog = pd.DataFrame(ilog)
    ilog.to_csv(os.path.join(outdir, "intervention_log.csv"), index=False)

    # ---- unit_traj_{regime}_{seed}_{arm}.npz
    steps_u = np.arange(t_int, t_int + post + 1, probe, dtype=np.int64)
    j6 = int(np.where(steps_u == t_int + P6_OFFSET)[0][0])
    for regime in REGIMES:
        for seed in _FIX_SEEDS:
            r = ilog[(ilog.regime == regime) & (ilog.seed == seed)].iloc[0]
            U = int(r.n_treated)
            for arm in ARMS:
                p_hat = np.zeros((len(steps_u), U), dtype=np.float32)
                k = int(round(_FIX_P6[arm] * U))
                # 介入後、ゲートが開くユニットは +100k までに立ち上がる階段
                p_hat[j6:, :k] = 0.30
                p_hat[max(j6 - 2, 0):j6, :k] = 0.02
                truth["reopen"][(regime, arm, seed)] = k / U
                w_norm = np.full((len(steps_u), U), 0.5, dtype=np.float32)
                if regime == "B":
                    beta = np.full((len(steps_u), U), -1.0, dtype=np.float32)
                    cos = np.full((len(steps_u), U), np.nan, dtype=np.float32)
                else:
                    beta = np.full((len(steps_u), U), np.nan, dtype=np.float32)
                    tgt = 0.2 + 0.01 * _fix_z(seed)        # unit median を厳密に tgt に
                    dv = tgt + (np.arange(U) - (U - 1) / 2.0) * 0.001
                    cos = np.zeros((len(steps_u), U), dtype=np.float32)
                    cos[0] = 0.05 * np.arange(U) / max(U, 1)
                    for t in range(1, len(steps_u)):
                        frac = t / (len(steps_u) - 1)
                        cos[t] = cos[0] + (dv * frac if arm == "posonly" else 0.0)
                    if arm == "posonly":
                        truth["dcos"][(regime, seed)] = float(np.median(dv))
                np.savez(os.path.join(outdir, f"unit_traj_{regime}_{seed}_{arm}.npz"),
                         steps=steps_u, unit_idx=np.arange(U, dtype=np.int64),
                         p_hat=p_hat, w_norm=w_norm, beta=beta, cos_u_mu=cos,
                         regime=np.array(regime), seed=np.array(str(seed)),
                         arm=np.array(arm), t_int=np.array(str(t_int)),
                         treated_hash=np.array(str(r.treated_hash)))

    # ---- meta.json / config_used.yaml
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(dict(elapsed_sec=1.0, device="cpu", date="2026-08-19 00:00:00",
                       omp_num_threads="1",
                       sanity=[dict(regime=r, S2="PASS",
                                    treated_frac=dict(
                                        min=float(ilog[ilog.regime == r].treated_frac.min()),
                                        mean=float(ilog[ilog.regime == r].treated_frac.mean())))
                               for r in REGIMES]), fh, indent=1)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(dict(common=dict(lop_every=lop_every, seeds=_FIX_SEEDS),
                            posreset=dict(t_int=t_int, post_steps=post, p_hat_tau=tau,
                                          probe_every=probe, treated_frac_min=0.3,
                                          treated_frac_min_seeds=8,
                                          s3_tol_f64=1.0e-12)), fh)
    return truth


def _close(a, b, tol=1e-9):
    return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol


def check_contract(chk):
    """フィクスチャの前提 (gbase / base_run_id) が **実物の config** と一致するか。
    ここが合わないと本走の成果物を 1 行も読めないので、フィクスチャ内での整合だけでは
    足りない。torch を引く重い import なので selftest 専用・失敗しても致命にしない。"""
    try:
        from .common import load_config, build_runs, group_runs, group_name
    except Exception as e:                                     # torch 未導入など
        chk(True, f"(skip) config 突き合わせ: src.common を import できない ({e})")
        return
    p = os.path.join(ROOT, "configs", "posreset_0819.yaml")
    if not os.path.exists(p):
        chk(True, "(skip) configs/posreset_0819.yaml がまだ無い")
        return
    cfg = load_config(p)
    runs = build_runs(cfg)
    groups = group_runs(runs)
    chk(set(groups) == {("A", 100, 1, "none"), ("B", 20, 1, "none")},
        f"build_runs が 2 グループちょうど: {sorted(groups)}")
    for gkey, regime in [(("A", 100, 1, "none"), "A"), (("B", 20, 1, "none"), "B")]:
        if gkey not in groups:
            continue
        chk(group_name(gkey) == GBASE[regime],
            f"gbase({regime}) = {group_name(gkey)} == {GBASE[regime]}")
        chk(len(groups[gkey]) == 10, f"レジーム {regime} は 10 seed")
        rid = groups[gkey][0]["run_id"]
        chk(rid == f"{_FIX_BASE[regime]}_s0",
            f"base_run_id({regime}) = {rid} (フィクスチャと一致)")
        for arm in ARMS + [CONT]:
            b, a = _split_arm(f"{rid}_{arm}")
            if not (b == rid and a == arm and _seed_of(b) == 0):
                chk(False, f"run_id 分解に失敗: {rid}_{arm} -> ({b}, {a})")
                break
        else:
            chk(True, f"run_id から arm 接尾辞と seed を全 5 アームで正しく分解 ({regime})")


def selftest(base):
    """3 シナリオで統計・判定を検証する。合格なら "SELFTEST PASS" を印字する。"""
    fails, lines = [], []

    def chk(cond, msg):
        lines.append(("  ok " if cond else "  NG ") + msg)
        if not cond:
            fails.append(msg)

    lines.append("[contract] 実物の configs/posreset_0819.yaml との突き合わせ")
    check_contract(chk)

    # ---------- シナリオ main: 既知の Δ を厳密に復元できるか
    d1 = os.path.join(base, "fixture_main")
    t1 = make_fixture(d1, "main")
    r1 = analyse(d1)
    ver, runs = r1["verdict"].set_index(["id", "regime"]), r1["runs"]
    lines.append("[main] 既知の Δ の復元 (真値: A full 0.40 / posonly 0.25 / dironly 0.05, "
                 "B full 1.00 / posonly 0.60 / dironly 0.00)")
    mt = r1["mtab"].set_index(["regime", "arm", "seed"])
    worst = max(abs(mt.loc[(rg, ar, sd), "M"] - t1["M"][(rg, ar, sd)])
                for (rg, ar, sd) in t1["M"])
    chk(worst < 1e-12, f"M が窓定義どおり厳密に一致 (最大誤差 {worst:.2e})")
    worst_l = max(abs(mt.loc[(rg, ar, sd), "M_late"] - t1["M_late"][(rg, ar, sd)])
                  for (rg, ar, sd) in t1["M_late"])
    chk(worst_l < 1e-12, f"M_late が窓定義どおり厳密に一致 (最大誤差 {worst_l:.2e})")
    chk(int(runs[runs.arm.isin(ARMS)].n_eval_points.iloc[0]) == 501,
        "M 窓の eval 点数 = 501 (両端含む)")
    chk(int(runs[runs.arm.isin(ARMS)].n_late_points.iloc[0]) == 100,
        "M_late 窓の eval 点数 = 100")

    for (i, rg, truth) in [("G0", "A", 0.40), ("G0", "B", 1.00), ("P1", "B", 0.60),
                           ("P4", "A", 0.25), ("P2", "B", 0.60 - 0.5 * 1.00),
                           ("P2_strong", "B", 0.60 - 0.75 * 1.00),
                           ("P3", "B", 0.25 * 1.00 - 0.00), ("P5", "A", 0.40 - 0.25),
                           ("P2_ratio", "B", 0.60)]:
        p = float(ver.loc[(i, rg), "point"])
        chk(_close(p, truth), f"{i}({rg}) 点推定 {p:.6f} == 真値 {truth:.6f}")
    for i, rg, exp in [("G0", "A", "PASS"), ("G0", "B", "PASS"), ("P1", "B", "PASS"),
                       ("P2", "B", "PASS"), ("P3", "B", "PASS"), ("P4", "A", "PASS"),
                       ("P5", "A", "report"), ("P2_strong", "B", "not 強"),
                       ("P6", "B", "report"), ("P7", "A", "report")]:
        got = ver.loc[(i, rg), "result"]
        chk(got == exp, f"{i}({rg}) result = {got!r} (期待 {exp!r})")
    lo = float(ver.loc[("G0", "A"), "ci_lo"])
    chk(lo > 0, f"G0(A) の CI 下限 {lo:.5f} > 0 (片側の読み)")

    # 既知 null: Δ_dironly(B) は真値ちょうど 0 → CI は 0 を含むこと
    dt = delta_table(r1["da"], r1["boot"]).set_index(["regime", "arm"])
    dn = float(dt.loc[("B", "dironly"), "Δ_M"])
    ci = dt.loc[("B", "dironly"), "CI_M"]
    lo_, hi_ = [float(x) for x in ci.strip("[]").split(",")]
    chk(_close(dn, 0.0), f"[既知 null] Δ_dironly(B) 点推定 {dn:.2e} == 0")
    chk(lo_ < 0 < hi_, f"[既知 null] その 95%CI [{lo_:.4f}, {hi_:.4f}] は 0 を含む")
    dfull = float(dt.loc[("B", "full"), "Δ_M"])
    chk(_close(dfull, 1.0), f"Δ_full(B) = {dfull:.6f} == 1.0")

    # 再現性: 同じ入力 → 同じ verdict (rng を 1 個だけ使う要件)
    r1b = analyse(d1)
    chk(r1b["verdict"].round(12).equals(r1["verdict"].round(12)),
        "同一入力で verdict.csv が bit 再現する (default_rng(20260819) 固定)")

    # 副次指標
    p6p = float(ver.loc[("P6_posonly", "B"), "point"])
    p6d = float(ver.loc[("P6_dironly", "B"), "point"])
    tp = np.mean([t1["reopen"][("B", "posonly", s)] for s in _FIX_SEEDS])
    td = np.mean([t1["reopen"][("B", "dironly", s)] for s in _FIX_SEEDS])
    chk(_close(p6p, tp) and _close(p6d, td),
        f"P6 再開率 posonly {p6p:.4f}=={tp:.4f} / dironly {p6d:.4f}=={td:.4f}")
    p7 = float(ver.loc[("P7", "A"), "point"])
    t7 = np.mean([t1["dcos"][("A", s)] for s in _FIX_SEEDS])
    chk(_close(p7, t7, 1e-6), f"P7 median Δcos {p7:.6f} == 真値 {t7:.6f}")
    # 比の CI の家内規約 (coupling_fbw_0813): 分母が 0 を跨ぐ replicate が 5% 以上なら
    # CI を出さない。狙い撃ちで両側を検証する
    b2 = PairedBoot()
    stable = ratio_ci(b2, ("t", "stable"), np.full(10, 0.6), np.full(10, 1.0))
    unstable = ratio_ci(b2, ("t", "unstable"), np.full(10, 0.6),
                        np.array([-1.0, -0.6, -0.2, 0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0]))
    chk(stable["reported"] and _close(stable["point"], 0.6)
        and np.isfinite(stable["lo"]), "比の CI: 分母が常に正なら CI を報告する")
    chk((not unstable["reported"]) and not np.isfinite(unstable["lo"])
        and unstable["frac_pos_den"] <= 0.95,
        f"比の CI: 分母が 0 を跨ぐと CI を報告しない "
        f"(分母が正の replicate {unstable['frac_pos_den']:.3f} ≤ 0.95)")

    chk(r1["sanity"]["S2_recheck"]["A"] == "PASS"
        and r1["sanity"]["S2_recheck"]["B"] == "PASS", "S2 独立再検査 PASS")
    chk(r1["sanity"]["S4_pass"] is True, "S4 (treated_hash 4 アーム一致) PASS")
    chk(r1["sanity"].get("S3_within_tol_f64") is True, "S3 float64 誤差が許容内")
    for f in ["runs.csv", "verdict.csv", "summary.md",
              "figures/fig_pr_evalloss_A.png", "figures/fig_pr_evalloss_B.png",
              "figures/fig_pr_gate_reopen_A.png", "figures/fig_pr_gate_reopen_B.png",
              "figures/fig_pr_cos_A.png", "figures/fig_pr_forest.png"]:
        chk(os.path.exists(os.path.join(d1, f)), f"成果物 {f} を出力")
    smry = open(os.path.join(d1, "summary.md")).read()
    chk("主張してはいけないこと" in smry and "dead_frac に基づくいかなる判定" in smry,
        "summary.md に §9 が逐語で入っている")
    chk("CI 下限 > 0" in smry, "summary.md に「CI が 0 を除外」の読み方が明記されている")
    chk("dead_frac" not in str(r1["verdict"].statistic.tolist()),
        "verdict の統計量に dead_frac が一切入っていない (§9)")

    # ---------- シナリオ null: G0(A) FAIL → A の P 行が全て void
    d2 = os.path.join(base, "fixture_null")
    make_fixture(d2, "null")
    r2 = analyse(d2)
    v2 = r2["verdict"].set_index(["id", "regime"])
    lines.append("[null] Δ_full(A) を厳密に 0 にしたときの挙動")
    p = float(v2.loc[("G0", "A"), "point"])
    chk(_close(p, 0.0), f"G0(A) 点推定 {p:.2e} == 0")
    chk(float(v2.loc[("G0", "A"), "ci_lo"]) < 0 < float(v2.loc[("G0", "A"), "ci_hi"]),
        "G0(A) の CI が 0 を含む (既知 null)")
    chk(v2.loc[("G0", "A"), "result"] == "FAIL", "G0(A) = FAIL")
    a_rows = r2["verdict"][(r2["verdict"].regime == "A")
                           & (~r2["verdict"].id.str.startswith("G0"))]
    chk((a_rows.result == "void").all(),
        f"A の P 行 {len(a_rows)} 本が全て void (G0 不成立)")
    b_rows = r2["verdict"][(r2["verdict"].regime == "B")
                           & (~r2["verdict"].id.str.startswith("G0"))]
    chk(not (b_rows.result == "void").any(), "B の P 行は void でない (レジーム独立)")
    chk(r2["elig"]["A"]["ok"] is False and r2["elig"]["B"]["ok"] is True,
        "§3.4 適格性: A 未達 (4/10 seed が treated_frac<0.3) / B 適格")
    chk("参考格" in "".join(a_rows.note.tolist()), "A の行に「参考格」が付いている")

    # ---------- シナリオ nan: 発散行の計数と seed 除外
    d3 = os.path.join(base, "fixture_nan")
    t3 = make_fixture(d3, "nan")
    r3 = analyse(d3)
    lines.append("[nan] 発散 (NaN eval_loss) の計数と除外")
    rr = r3["runs"].set_index(["regime", "seed", "arm"])
    chk(int(rr.loc[("A", 7, "dironly"), "n_nan_points"]) == 50,
        "部分的 NaN (A/s7/dironly) を 50 行として計上")
    chk(np.isfinite(rr.loc[("A", 7, "dironly"), "M"]),
        "部分 NaN の run は有限行のみで M を算出 (黙殺しない)")
    chk(int(rr.loc[("B", 9, "full"), "n_nan_points"]) == 501
        and not np.isfinite(rr.loc[("B", 9, "full"), "M"]),
        "全 NaN (B/s9/full) は M=NaN")
    v3 = r3["verdict"].set_index(["id", "regime"])
    chk("n_seed=9" in str(v3.loc[("P1", "B"), "note"]),
        "B の統計は seed 9 を除外して n_seed=9 と明示")
    chk("n_seed=10" in str(v3.loc[("P4", "A"), "note"]),
        "A は除外なしで n_seed=10")
    chk(_close(float(v3.loc[("P4", "A"), "point"]), 0.25),
        "A の Δ_posonly は NaN 混入の影響を受けず 0.25 のまま")
    chk("発散" in open(os.path.join(d3, "summary.md")).read(),
        "summary.md の逸脱記録に発散 run が載っている")

    # ---------- 回帰: 一度直した欠陥が戻らないようにする (verdict-logic 監査 0819)
    lines.append("[regress] 監査で見つかった欠陥の回帰チェック")

    # (R1) S2 独立再検査: 両側 NaN の列と、境界行の persist 列で誤 FAIL しないこと。
    #      ランナー実出力では dead≈0.99 のため両側 NaN 列が常時出る。
    d4 = os.path.join(base, "fixture_s2")
    os.makedirs(d4, exist_ok=True)
    cols = ["step", "run_id", "eval_loss", "b_mean_alive", "dead_persist_frac"]
    stp = [500000, 501000, 502000]
    ca = pd.DataFrame([(s, "X_s0_cont", 1.0, np.nan, 1.0) for s in stp], columns=cols)
    nb = pd.DataFrame([(s, "X_s0_none", 1.0, np.nan, np.nan if s == 500000 else 1.0)
                       for s in stp], columns=cols)
    gb4 = GBASE["A"]
    ca.to_csv(os.path.join(d4, f"lop_metrics_{gb4}_{CONT}.csv"), index=False)
    nb.to_csv(os.path.join(d4, f"lop_metrics_{gb4}_none.csv"), index=False)
    chk(compare_cont_vs_none(d4, "A", 500000) == "PASS",
        "S2 再検査: 両側 NaN の列と境界行の persist 列で誤 FAIL しない")
    nb2 = nb.copy(); nb2.loc[nb2.step == 502000, "eval_loss"] = 9.0
    nb2.to_csv(os.path.join(d4, f"lop_metrics_{gb4}_none.csv"), index=False)
    chk(compare_cont_vs_none(d4, "A", 500000).startswith("FAIL"),
        "S2 再検査: 本物の不一致 (eval_loss 1 セル) は依然 FAIL")

    # (R2) P2 が weak PASS のとき §11 へ無条件には流さない
    fake = pd.DataFrame([_vrow(i, r, "", 1.0, 0.5, 1.5, "", res, "")
                         for i, r, res in [("G0", "A", "PASS"), ("G0", "B", "PASS"),
                                           ("P1", "B", "PASS"), ("P2", "B", "weak PASS"),
                                           ("P3", "B", "PASS"), ("P4", "A", "PASS")]])
    txt = conclusion_text(fake, {})
    chk("留保つき" in txt and "弱 PASS" in txt,
        "P2 が weak PASS のとき一行結論は §11 を留保つきにする")
    fake.loc[fake.id == "P2", "result"] = "PASS"
    chk("§11 の主張へ" in conclusion_text(fake, {}),
        "P2 が PASS なら従来どおり §11 の主張へ")

    # (R3) データ不足の G0 を科学的 FAIL と誤報しない / 打ち切り run を警告する
    d5 = os.path.join(base, "fixture_short")
    make_fixture(d5, "main")
    for gb in GBASE.values():                      # B の全アームを 700k で打ち切り
        for arm in ARMS:
            p5 = os.path.join(d5, f"lop_metrics_{gb}_{arm}.csv")
            if gb != GBASE["B"]:
                continue
            x = pd.read_csv(p5)
            x[x.step <= 700000].to_csv(p5, index=False, float_format="%.17g")
    r5 = analyse(d5)
    chk("eval 点数が不揃い" in open(os.path.join(d5, "summary.md")).read(),
        "打ち切りで窓点数が不揃いになった run を逸脱記録で警告する")
    os.remove(os.path.join(d5, f"lop_metrics_{GBASE['A']}_full.csv"))
    r6 = analyse(d5)
    v6 = r6["verdict"].set_index(["id", "regime"])
    chk(str(v6.loc[("G0", "A"), "result"]).startswith("NA"),
        f"full アーム欠落 → G0(A) は {v6.loc[('G0', 'A'), 'result']!r} (科学的 FAIL ではない)")
    chk((r6["verdict"][(r6["verdict"].regime == "A")
                       & (~r6["verdict"].id.str.startswith("G0"))].result == "void").all(),
        "データ不足の G0 でも A の P 行は void になる")

    # (R4) t_int 行の非対称性が逸脱記録に数値つきで載る
    chk("M 窓の左端" in open(os.path.join(d1, "summary.md")).read(),
        "M 窓左端 (t_int 行) の none/reset 非対称を逸脱記録に明記")

    print("\n".join(lines))
    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)} 件)")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"SELFTEST PASS ({len(lines)} チェック / フィクスチャ {base})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join(ROOT, "results",
                                                               "posreset_0819"))
    ap.add_argument("--selftest", action="store_true",
                    help="合成フィクスチャで統計・判定ロジックを自己検証する")
    ap.add_argument("--selftest-dir", default=None)
    args = ap.parse_args()

    if args.selftest:
        base = args.selftest_dir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "posreset_selftest")
        os.makedirs(base, exist_ok=True)
        sys.exit(selftest(base))

    out = analyse(args.results)
    print(out["verdict"][["id", "regime", "point", "ci_lo", "ci_hi", "result"]]
          .round(5).to_string(index=False))
    print()
    print(conclusion_text(out["verdict"], out["elig"]))
    print(f"-> {args.results}/{{runs.csv, verdict.csv, summary.md, figures/}}")


if __name__ == "__main__":
    main()
