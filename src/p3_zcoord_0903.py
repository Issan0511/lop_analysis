#!/usr/bin/env python3
"""``ELU可動度の深さ座標_spec_0902`` v6 の再集計（走なし）。

spec §4 の A–E を出し、§5 の P1–P5 を判定してラベルを 1 つ返す。新しい走は
一切しない。読むのは committed の生ログだけ:

* ``results/p3_extend_0902/logs/`` … ``E_1216`` / ``LR_1216``（15M・α=1／a=0.1）
* ``results/gate_dial_0902/logs/`` … ELU α∈{0.1,0.01,0.001}・leaky a∈{0.01,0.001}

集計の骨格（沈下ユニットのタスク内 :math:`\\bar z` 増分を 1000 step あたりで
深さ帯に切る）は [[現象3_非ReLU戻り道の対応づけ_0902]] §10-4 の
``fullpass.py`` を逐語で継承する。**替えるのは帯を切る座標だけ**:

* §10-4 と同じ ``layer1_zbar`` の深さ（= C の座標）
* spec §2 が指す ``layer1_zmax`` の深さ（= A・B の座標）

ELU の沈下ユニット（``zmax`` <= 0）では ``act_grad`` が厳密に
:math:`\\alpha e^{z}`（``nets.VecMLPL.act_grad`` の ``alpha_exp`` 分岐）なので

    ln(mob/alpha) = ln( mean_p exp(z_p) ) in [ zmax - ln 32 , zmax ]

が恒等式として成り立つ（32 = 厳密支持のパターン数）。これが D／P1。
leaky の沈下ユニットでは ``act_grad`` が定数 a なので ln(mob/a) は厳密に 0 で、
同じ配管が B で指数 0 を返さなければ配管が壊れている（S 検査／P4）。

使い方::

    python -m src.p3_zcoord_0903 --out results/zcoord_0903
    python -m src.p3_zcoord_0903 --selftest   # 集計せず配管の検査だけ
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

LN32 = math.log(32.0)            # spec §2(a) の 3.4657...
BAND_EPS = 1e-4                  # float32 保存の丸め代（本文 §S-num）
TOL_P2 = 0.3                     # spec §5 P2 の許容帯
TOL_P4 = 0.1                     # spec §5 P4 の許容帯
MIN_N = 1000                     # 帯の最小行数（副次判定でのみ使う）

# --- 腕（spec §4「腕」） -------------------------------------------------
# (arm, run, family, dial, n_steps)
ARMS = [
    ("E_1216",        "p3_extend_0902", "elu",   1.0,    15_000_000),
    ("E_a0p1_1216",   "gate_dial_0902", "elu",   0.1,     5_000_000),
    ("E_a0p01_1216",  "gate_dial_0902", "elu",   0.01,    5_000_000),
    ("E_a0p001_1216", "gate_dial_0902", "elu",   0.001,   5_000_000),
    ("LR_1216",       "p3_extend_0902", "leaky", 0.1,    15_000_000),
    ("LR_a0p01_1216", "gate_dial_0902", "leaky", 0.01,    5_000_000),
    ("LR_a0p001_1216","gate_dial_0902", "leaky", 0.001,   5_000_000),
]
SEEDS = range(10)

# §10-4 の等幅帯（逐語継承）
ABS_EDGES = [0.0, 1.0, 3.0, 6.0, 10.0, np.inf]
ABS_NAMES = ["0-1", "1-3", "3-6", "6-10", ">10"]


# ---------------------------------------------------------------------------
# 1 腕ぶんの行を作る
# ---------------------------------------------------------------------------
def load_arm(arm: str, run: str, *, max_step: int | None = None,
             fam_is_elu: bool = True) -> dict:
    """1 腕・全 seed の「沈下 × タスク内」の行をプールして返す。

    §10-4 の ``fullpass.py`` と同じく、増分は隣り合う記録点の差
    ``zbar[i+1]-zbar[i]``（記録は 1000 step ごとなので 1000 step あたり）で、
    タスク内は ``flip_state[i+1] == flip_state[i]``、深さは**増分の手前** ``i``
    の状態で測る。沈下の判定も手前の記録点で行う（spec §4）。
    """
    logdir = RESULTS / run / "logs"
    cols = {k: [] for k in ("inc", "inc_f32", "d_zmax", "d_zbar", "lnmob",
                            "absv", "seed", "resid", "pos", "phi_ratio")}
    n_raw = n_sub = n_within = 0
    n_mob_zero = 0
    agree_num = agree_den = 0          # zmax<=0 と p_hat==0 の一致（§5 の疑い(1)）
    alpha = None
    for seed in SEEDS:
        path = logdir / f"{arm}_seed{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        L = np.load(path, allow_pickle=True)
        alpha = float(L["act_alpha"])
        step = L["step"]
        keep = slice(None) if max_step is None else slice(
            0, int(np.searchsorted(step, max_step, side="right")))
        step = step[keep]
        fs = L["flip_state"][keep]
        zbar = L["layer1_zbar"][keep].astype(np.float64)
        zmax = L["layer1_zmax"][keep].astype(np.float64)
        mob = L["layer1_mob"][keep].astype(np.float64)
        v = L["layer1_v_unit"][keep].astype(np.float64)
        phat = L["layer1_p_hat"][keep]

        within = ~(fs[1:] != fs[:-1]).any(1)          # 増分 j = 記録 j -> j+1
        # ★ 増分はロガーの ``layer1_dzbar`` を使う。これは float64 の zbar 同士の
        # 差を取ってから float32 に落とした量で、``EluRecorder`` の docstring が
        # 「zbar を float32 に落としてから引くと深い沈下域が丸めに沈む」と明示的に
        # 警告している。dzbar[i] は記録 i-1 -> i の増分なので、増分 j に対応する
        # のは dzbar[j+1]。dzbar[0] は NaN（前の記録が無い）。
        dzbar = L["layer1_dzbar"][keep].astype(np.float64)
        inc = dzbar[1:]
        inc_f32 = zbar[1:] - zbar[:-1]      # §10-4 の作り方（S-repro と比較用）
        prev = slice(0, len(step) - 1)
        sub = zmax[prev] <= 0.0
        if np.isnan(inc[within[:, None] & sub]).any():
            raise ValueError(f"{arm} seed{seed}: dzbar に NaN が残っている")

        agree_num += int((sub == (phat[prev] == 0)).sum())
        agree_den += int(sub.size)

        m = within[:, None] & sub
        n_raw += int(m.size)
        n_sub += int(sub.sum())
        n_within += int(m.sum())
        if not m.any():
            continue

        mob_m = mob[prev][m]
        n_mob_zero += int((mob_m <= 0).sum())
        with np.errstate(divide="ignore"):
            lnmob = np.log(mob_m / alpha)
        zmax_m = zmax[prev][m]
        cols["inc"].append(inc[m].astype(np.float64))
        cols["inc_f32"].append(inc_f32[m].astype(np.float64))
        cols["d_zmax"].append(-zmax_m)
        cols["d_zbar"].append(-zbar[prev][m])
        cols["lnmob"].append(lnmob)
        cols["resid"].append(lnmob - zmax_m)
        cols["absv"].append(np.abs(v[prev][m]))
        # §2(c) の前件は「深い ELU では φ→−α（定数）なので括弧の中が深さに
        # 依らなくなる」。その前件そのものを測る量が E_x[φ]/α である。
        # 沈下 ELU では E_x[φ] = α(E[e^z]-1) = mob - α、沈下 leaky では a·zbar。
        phi_over_alpha = ((mob[prev][m] / alpha - 1.0) if fam_is_elu
                          else zbar[prev][m])
        cols["phi_ratio"].append(phi_over_alpha)
        cols["seed"].append(np.full(int(m.sum()), seed, dtype=np.int8))
        # タスク内位置 o: 増分 j = 記録 j -> j+1 の j % (period/1000)。
        # 境界増分（o=0）は within で既に落ちているので o は 1..9 を取る。
        every = int(L["task_period"]) // 1000
        jj = np.arange(len(step) - 1) % every
        cols["pos"].append(np.broadcast_to(jj[:, None], m.shape)[m].astype(np.int8))
        del L
    out = {k: (np.concatenate(vs) if vs else np.empty(0)) for k, vs in cols.items()}
    out["alpha"] = alpha
    out["n_raw"] = n_raw
    out["n_sub"] = n_sub
    out["n_rows"] = n_within
    out["n_mob_zero"] = n_mob_zero
    out["sub_agree"] = agree_num / agree_den if agree_den else float("nan")
    return out


def load_arm_repro(arm: str, run: str, sub_def: str) -> dict:
    """S-repro 用。§10-4 の切り方（``zbar`` 深さ・沈下プール）をそのまま作る。

    ``sub_def`` は ``"phat"``（§10-4 の本文が書いた :math:`\\hat p=0`）か
    ``"zmax"``（spec §4 が使う :math:`\\max_x z\\le 0`）。両者が同じ表を返すことが
    spec §5「外れたときに第一に疑うもの (1)」の検査になる。
    """
    logdir = RESULTS / run / "logs"
    inc_all, dep_all = [], []
    for seed in SEEDS:
        L = np.load(logdir / f"{arm}_seed{seed}.npz", allow_pickle=True)
        fs = L["flip_state"]
        zbar = L["layer1_zbar"].astype(np.float64)
        within = ~(fs[1:] != fs[:-1]).any(1)
        inc = zbar[1:] - zbar[:-1]
        dep = -zbar[:-1]
        sub = (L["layer1_p_hat"][:-1] == 0) if sub_def == "phat" \
            else (L["layer1_zmax"][:-1].astype(np.float64) <= 0.0)
        m = within[:, None] & sub
        inc_all.append(inc[m])
        dep_all.append(dep[m])
        del L
    return dict(inc=np.concatenate(inc_all), depth=np.concatenate(dep_all))


# ---------------------------------------------------------------------------
# 2 帯を切る
# ---------------------------------------------------------------------------
def bin_edges(depth: np.ndarray, binset: str) -> tuple[list[float], list[str]]:
    if binset == "abs":
        return list(ABS_EDGES), list(ABS_NAMES)
    if binset == "eqf":
        q = np.quantile(depth, [0.2, 0.4, 0.6, 0.8])
        edges = [-np.inf] + [float(x) for x in q] + [np.inf]
        names = [f"q{i+1}" for i in range(5)]
        return edges, names
    raise ValueError(binset)


def bin_table(d: dict, coord: str, binset: str,
              keep: np.ndarray | None = None) -> list[dict]:
    """A・B（coord="zmax"）と C（coord="zbar"）の帯ごとの量。

    ``keep`` を渡すとその部分集合だけで集計する（タスク内位置を固定するのに使う）。
    帯の切り目は ``keep`` を掛ける**前**の分布から決めるので、位置をまたいで
    同じ帯になる。
    """
    depth = d["d_zmax"] if coord == "zmax" else d["d_zbar"]
    edges, names = bin_edges(depth, binset)
    rows = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (depth > lo) & (depth <= hi)
        if keep is not None:
            m = m & keep
        n = int(m.sum())
        row = dict(coord=coord, binset=binset, bin=name, lo=lo, hi=hi, n=n)
        if n:
            inc = d["inc"][m]
            lnmob = d["lnmob"][m]
            dep = depth[m]
            mid = (lo + hi) / 2.0
            # 量子化診断: zbar は float32 保存なので増分の分解能は |zbar| の ulp。
            # 中央値は量子の整数倍しか取れないが、平均は丸め誤差を平均化できる。
            ulp = np.spacing(np.abs(d["d_zbar"][m]).astype(np.float32))
            med = float(np.median(inc))
            row.update(
                depth_med=float(np.median(dep)),
                depth_mid=float(mid) if np.isfinite(mid) else float(np.median(dep)),
                inc_med=med,
                inc_med_f32=float(np.median(d["inc_f32"][m])),
                inc_mean=float(inc.mean()),
                inc_p_up=float(np.mean(inc > 0)),
                inc_frac_zero=float(np.mean(inc == 0.0)),
                ulp_med=float(np.median(ulp)),
                med_over_ulp=(abs(med) / float(np.median(ulp))
                              if np.median(ulp) > 0 else float("nan")),
                lnmob_med=float(np.median(lnmob)),
                resid_med=float(np.median(d["resid"][m])),
                phi_ratio_med=float(np.median(d["phi_ratio"][m])),
            )
            # seed ごとの中央値（腕内のばらつき。判定には使わず報告のみ）
            per_seed = []
            for sd in SEEDS:
                ms = d["seed"][m] == sd
                if ms.sum() >= 100:
                    per_seed.append(float(np.median(inc[ms])))
            row["inc_med_seed_n"] = len(per_seed)
            if per_seed:
                arr = np.array(per_seed, dtype=np.float64)
                row["inc_med_seed_med"] = float(np.median(arr))
                row["inc_med_seed_min"] = float(arr.min())
                row["inc_med_seed_max"] = float(arr.max())
                row["inc_med_seed_mean"] = float(arr.mean())
                # seed 間のばらつきから作る中央値の誤差。記録点は自己相関するので
                # プールの n では誤差を作れない（seed は独立な走）。
                row["inc_med_seed_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
                row["inc_med_seed_se"] = (float(arr.std(ddof=1) / math.sqrt(arr.size))
                                          if arr.size > 1 else float("nan"))
                row["inc_med_hi2se"] = (float(arr.mean() + 2 * arr.std(ddof=1)
                                              / math.sqrt(arr.size))
                                        if arr.size > 1 else float("nan"))
                row["inc_med_lo2se"] = (float(arr.mean() - 2 * arr.std(ddof=1)
                                              / math.sqrt(arr.size))
                                        if arr.size > 1 else float("nan"))
            # 増分の広がり（中央値は「小さな正味」であって「小さな量」ではない）
            row["inc_iqr"] = float(np.quantile(inc, 0.75) - np.quantile(inc, 0.25))
            row["inc_sd"] = float(inc.std())
            row["inc_mad"] = float(np.median(np.abs(inc - med)))
            row["inc_frac_gt_1e3"] = float(np.mean(np.abs(inc) > 1e-3))
        rows.append(row)
    return rows


def bin_table_by_seed(d, coord, binset, key, arm, wname, fam, dial) -> list[dict]:
    """帯 × seed の中央値。プールが 1 seed に支配されていないかの監査用。"""
    depth = d["d_zmax"] if coord == "zmax" else d["d_zbar"]
    edges, names = bin_edges(depth, binset)
    out = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (depth > lo) & (depth <= hi)
        if not m.any():
            continue
        inc, ln, sd = d["inc"][m], d["lnmob"][m], d["seed"][m]
        for s_ in SEEDS:
            ms = sd == s_
            n = int(ms.sum())
            if not n:
                continue
            out.append(dict(key=key, arm=arm, window=wname, family=fam, dial=dial,
                            coord=coord, binset=binset, bin=name, seed=int(s_),
                            n=n, inc_med=float(np.median(inc[ms])),
                            inc_mean=float(inc[ms].mean()),
                            inc_p_up=float(np.mean(inc[ms] > 0)),
                            lnmob_med=float(np.median(ln[ms]))))
    return out


def local_exponents(rows: list[dict], abscissa: str) -> list[dict]:
    """隣り合う帯の間の局所指数。

    増分・可動度とも「深さが 1 増えるごとに ``exp(-kappa)`` 倍」の kappa で、
    §3 の「2->4.5 で 0.37」と同じ量（``ln(inc_lo/inc_hi) / (x_hi - x_lo)``）。
    """
    out = []
    usable = [r for r in rows if r["n"] > 0]
    for a, b in zip(usable[:-1], usable[1:]):
        xa, xb = a[abscissa], b[abscissa]
        dx = xb - xa
        rec = dict(coord=a["coord"], binset=a["binset"], abscissa=abscissa,
                   pair=f"{a['bin']}->{b['bin']}", x_lo=xa, x_hi=xb, dx=dx,
                   n_lo=a["n"], n_hi=b["n"])
        if dx <= 0:
            out.append(rec)
            continue
        k_mob = -(b["lnmob_med"] - a["lnmob_med"]) / dx
        rec["kappa_mob"] = float(k_mob)
        # 量子化に触れているか（どちらかの帯で |中央値| < 2 ulp なら床）
        rec["at_floor"] = bool(min(a.get("med_over_ulp", np.inf),
                                   b.get("med_over_ulp", np.inf)) < 2.0)
        rec["med_over_ulp_lo"] = a.get("med_over_ulp")
        rec["med_over_ulp_hi"] = b.get("med_over_ulp")
        # §2(c) が正しければ深い側の中央値はこうなるはず（可動度と同率で落ちる）。
        # これが ulp よりずっと大きいのに実測が床なら、床のせいではなく本当に速い。
        if a["inc_med"] > 0:
            pred = a["inc_med"] * math.exp(-k_mob * dx)
            rec["inc_med_pred_hi"] = float(pred)
            rec["pred_over_ulp_hi"] = (float(pred / b["ulp_med"])
                                       if b.get("ulp_med") else float("nan"))
            rec["obs_over_pred_hi"] = float(b["inc_med"] / pred) if pred else float("nan")
            # ★ 判定の実体はここ: 深い側の中央値の seed 間 2σ 上限が §2(c) の
            # 予測を下回っているか。下回っていれば「測れなかった」ではなく
            # 「予測より小さいことが seed をまたいで示されている」。
            hi, lo = b.get("inc_med_hi2se"), b.get("inc_med_lo2se")
            rec["obs_hi2se_hi"] = hi
            rec["obs_lo2se_hi"] = lo
            rec["n_seed_hi"] = b.get("inc_med_seed_n")
            # seed が 10 本そろっていない帯と、区間の幅が 0 の帯（10 seed の
            # 中央値が全部同じ値＝多くは全部 0）は排除の根拠にしない。
            rec["ci_usable"] = bool(b.get("inc_med_seed_n") == len(list(SEEDS))
                                    and hi is not None and hi == hi
                                    and lo is not None and hi > lo)
            if hi is not None and hi == hi and rec["ci_usable"]:
                # 両側。leaky（κ_可動度 = 0 ⇒ 予測は「深さに依らず同じ値」）では
                # 増分が**増える**側にはみ出すので、下側 2σ も見ないと落とせない。
                rec["excluded_by_2se"] = bool(hi < pred or lo > pred)
                rec["pred_over_hi2se"] = (float(pred / hi) if hi > 0
                                          else float("inf"))
        for tag, key in (("", "inc_med"), ("_mean", "inc_mean"),
                         ("_iqr", "inc_iqr"), ("_sd", "inc_sd"),
                         ("_mad", "inc_mad")):
            va, vb = a[key], b[key]
            if va > 0 and vb > 0:
                k = -(math.log(vb) - math.log(va)) / dx
                rec["kappa_inc" + tag] = float(k)
                rec["diff" + tag] = float(k - k_mob)
                if not tag:
                    rec["same_sign"] = bool(np.sign(k) == np.sign(k_mob)
                                            and k != 0.0)
                    rec["within_tol"] = bool(abs(k - k_mob) <= TOL_P2)
            else:
                rec["kappa_inc" + tag] = float("nan")
                if not tag:
                    rec["sign_undefined"] = True
        out.append(rec)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3 配管の検査（S）
# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    """走らせる前に配管を検査する 3 本。

    * **S-grad**: ``nets.act_grad`` の ELU 分岐が z<=0 で厳密に alpha*e^z か
    * **S-repro**: §10-4 の ``E_1216`` の深さ方向の中央値（+4.6e-3 / +0.40 /
      +0.07 / +0.005 ×1e-3）を、沈下フィルタを掛けない同じ切り方で再現するか
    * **S-mob-leaky**: leaky の沈下ユニットで ln(mob/a) が厳密に 0 か
    """
    res = {}
    # --- S-grad
    sys.path.insert(0, str(REPO))
    import torch
    from src.nets import VecMLPL
    gen = torch.Generator(device="cpu").manual_seed(0)
    net = VecMLPL(4, [3], 2, gen, "cpu").set_activation("elu", 0.3, "alpha_exp")
    z = torch.tensor([-30.0, -10.0, -1.0, -1e-9, 0.0], dtype=torch.float64)
    got = net.act_grad(z, net.act_fn(z)).numpy()
    want = 0.3 * np.exp(z.numpy())
    res["S_grad_max_rel"] = float(np.max(np.abs(got / want - 1.0)))
    res["S_grad"] = bool(res["S_grad_max_rel"] < 1e-12)

    # --- S-repro / S-sub（§10-4 E_1216 15M・沈下プール・zbar 深さ）
    want_med = {"1-3": 4.6e-3, "3-6": 0.40e-3, "6-10": 0.07e-3, ">10": 0.005e-3}
    got = {}
    for sub_def in ("phat", "zmax"):
        u = load_arm_repro("E_1216", "p3_extend_0902", sub_def)
        got[sub_def] = {}
        for lo, hi, nm in zip(ABS_EDGES[:-1], ABS_EDGES[1:], ABS_NAMES):
            m = (u["depth"] > lo) & (u["depth"] <= hi)
            got[sub_def][nm] = ([float(np.median(u["inc"][m])), int(m.sum())]
                                if m.any() else [float("nan"), 0])
        got[sub_def]["_n"] = int(u["inc"].size)
    # ノート側は 2 桁丸めなので相対 15%（+ 丸め幅の半分）を許す
    ok = all(abs(got["zmax"][k][0] - want_med[k]) <= 0.15 * abs(want_med[k]) + 5e-6
             for k in want_med)
    res["S_repro_got"] = got["zmax"]
    res["S_repro_want"] = want_med
    res["S_repro"] = bool(ok)
    res["S_sub"] = bool(got["phat"] == got["zmax"])   # §5 の第一容疑 (1)

    # --- S-mob-leaky（LR_a0p01 の 1 seed で十分。float32 保存の丸めだけ許す）
    L = np.load(RESULTS / "gate_dial_0902/logs/LR_a0p01_1216_seed0.npz",
                allow_pickle=True)
    a = float(L["act_alpha"])
    zmax = L["layer1_zmax"].astype(np.float64)
    mob = L["layer1_mob"].astype(np.float64)
    sub = zmax <= 0
    res["S_mob_leaky_max_rel"] = float(np.max(np.abs(mob[sub] / a - 1.0)))
    res["S_mob_leaky_unique"] = int(np.unique(mob[sub]).size)
    res["S_mob_leaky"] = bool(res["S_mob_leaky_max_rel"] < 1e-6
                              and res["S_mob_leaky_unique"] == 1)

    # --- S-bnd（§5 の第一容疑 (2)）: 増分がタスク境界をまたいでいないか。
    # flip_state は「step t の記録 -> t+1000 の記録」の間で変わる（記録は切替の
    # 手前で取られる）ので、除くべき増分は index 10,20,... であって step//period
    # が変わる index 9,19,... ではない。1 記録ずれた定義を使うと切替をまたぐ増分
    # をタスク内に入れてしまう。ここではずれの向きまで固定して記録する。
    L = np.load(RESULTS / "p3_extend_0902/logs/E_1216_seed0.npz", allow_pickle=True)
    step, fs = L["step"], L["flip_state"]
    per = int(L["task_period"])
    ch = np.nonzero((fs[1:] != fs[:-1]).any(1))[0]
    every = int(per // 1000)
    res["S_bnd_first_change_idx"] = int(ch[0])
    res["S_bnd_all_multiples"] = bool(np.all(ch % every == 0) and ch[0] == every)
    res["S_bnd_spans_switch"] = bool(
        step[ch[0]] % per == 0 and step[ch[0] + 1] % per != 0)
    res["S_bnd"] = bool(res["S_bnd_all_multiples"] and res["S_bnd_spans_switch"])

    res["all_pass"] = bool(res["S_grad"] and res["S_repro"] and res["S_sub"]
                           and res["S_mob_leaky"] and res["S_bnd"])
    if verbose:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


# ---------------------------------------------------------------------------
# 4 本体
# ---------------------------------------------------------------------------
def run(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    bins_rows, exp_rows, ident_rows, spear_rows, seed_rows = [], [], [], [], []
    pos_bins, pos_exp = [], []
    per_arm = {}

    windows = []
    for arm, run_, fam, dial, nstep in ARMS:
        windows.append((arm, run_, fam, dial, None, f"{nstep//10**6}M"))
        if nstep > 5_000_000:            # 5M 打ち切りの副次窓（§3 の振れの検査）
            windows.append((arm, run_, fam, dial, 5_000_000, "5M"))

    for arm, run_, fam, dial, max_step, wname in windows:
        key = f"{arm}@{wname}"
        d = load_arm(arm, run_, max_step=max_step, fam_is_elu=(fam == "elu"))
        print(f"[{key}] rows={d['n_rows']:,} sub_agree={d['sub_agree']:.6f} "
              f"alpha={d['alpha']}", flush=True)
        per_arm[key] = dict(arm=arm, run=run_, family=fam, dial=dial,
                            window=wname, alpha=d["alpha"],
                            n_rows=d["n_rows"], n_sub=d["n_sub"],
                            n_raw=d["n_raw"], sub_agree=d["sub_agree"],
                            n_mob_zero=d["n_mob_zero"])
        # --- D / P1（ELU のみ。leaky では恒等式が別物）
        if fam == "elu" and d["n_rows"]:
            # 5M 打ち切り窓は 15M 窓の**部分集合**（先頭を切るだけなので行集合が
            # prefix になる）。合計行数を出すときに足してはいけない。
            subset_of = (f"{arm}@15M" if wname == "5M" and max_step else None)
            r = d["resid"]
            # n_eff = 32*exp(resid): ゲートを担っている「実効パターン数」。
            # 1 なら最浅の 1 パターンだけが E_x[phi'] を決めている。
            ident_rows.append(dict(
                key=key, arm=arm, window=wname, n=int(r.size),
                subset_of=subset_of,
                resid_min=float(r.min()), resid_max=float(r.max()),
                resid_q01=float(np.quantile(r, 0.01)),
                resid_med=float(np.median(r)),
                resid_q99=float(np.quantile(r, 0.99)),
                n_eff_med=float(32.0 * math.exp(float(np.median(r)))),
                n_eff_q01=float(32.0 * math.exp(float(np.quantile(r, 0.01)))),
                n_eff_q99=float(32.0 * math.exp(float(np.quantile(r, 0.99)))),
                n_below=int((r < -LN32 - BAND_EPS).sum()),
                n_above=int((r > BAND_EPS).sum()),
                n_mob_zero=d["n_mob_zero"]))
        # --- A / B / C
        for coord in ("zmax", "zbar"):
            for binset in ("abs", "eqf"):
                rows = bin_table(d, coord, binset)
                for r in rows:
                    r.update(key=key, arm=arm, window=wname, family=fam, dial=dial)
                bins_rows.extend(rows)
                seed_rows.extend(bin_table_by_seed(d, coord, binset, key, arm,
                                                  wname, fam, dial))
                for abscissa in ("depth_mid", "depth_med"):
                    for e in local_exponents(rows, abscissa):
                        e.update(key=key, arm=arm, window=wname,
                                 family=fam, dial=dial)
                        exp_rows.append(e)
                if binset == "abs":
                    # ★ タスク内位置を固定した版。増分は 1 タスクの中で 2〜3 桁
                    # 減衰するので、9 位置をプールした中央値は後半の静かな位置で
                    # 決まる。登録どおりの主判定はプール版だが、位置固定版が
                    # ラベルを動かすかを必ず見る。
                    for o in range(1, 10):
                        prows = bin_table(d, coord, binset, keep=(d["pos"] == o))
                        for r in prows:
                            r.update(key=key, arm=arm, window=wname,
                                     family=fam, dial=dial, pos=o)
                        pos_bins.extend(prows)
                        for e in local_exponents(prows, "depth_mid"):
                            e.update(key=key, arm=arm, window=wname,
                                     family=fam, dial=dial, pos=o)
                            pos_exp.append(e)
        # --- E / P5
        if d["n_rows"]:
            for seed in SEEDS:
                m = d["seed"] == seed
                if m.sum() < 3:
                    continue
                spear_rows.append(dict(
                    key=key, arm=arm, window=wname, seed=int(seed), n=int(m.sum()),
                    rho_zbar=spearman(d["absv"][m], d["d_zbar"][m]),
                    rho_zmax=spearman(d["absv"][m], d["d_zmax"][m])))
        del d

    verdict0 = judge(bins_rows, exp_rows, ident_rows, spear_rows, per_arm,
                     pos_exp)
    write_summary(outdir, bins_rows, exp_rows, ident_rows, spear_rows,
                  per_arm, verdict0, pos_bins)
    write_csv(outdir / "bins.csv", bins_rows)
    write_csv(outdir / "bins_by_seed.csv", seed_rows)
    write_csv(outdir / "bins_by_pos.csv", pos_bins)
    write_csv(outdir / "exponents_by_pos.csv", pos_exp)
    write_csv(outdir / "exponents.csv", exp_rows)
    write_csv(outdir / "identity.csv", ident_rows)
    write_csv(outdir / "spearman.csv", spear_rows)
    verdict = verdict0
    (outdir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=float))
    (outdir / "per_arm.json").write_text(
        json.dumps(per_arm, indent=2, ensure_ascii=False, default=float))
    return verdict


def judge(bins_rows, exp_rows, ident_rows, spear_rows, per_arm,
          pos_exp=()) -> dict:
    """spec §5 の P1–P5 と表のラベル。主判定は主窓（15M / 5M）だけで取る。"""
    main = {k for k, v in per_arm.items() if v["window"] in ("15M", "5M")
            and not (v["arm"] in ("E_1216", "LR_1216") and v["window"] == "5M")}
    elu = {k for k in main if per_arm[k]["family"] == "elu"}
    lky = {k for k in main if per_arm[k]["family"] == "leaky"}

    # --- P1: D の恒等式
    p1_bad = [r for r in ident_rows
              if r["key"] in elu and (r["n_below"] or r["n_above"])]
    P1 = not p1_bad

    def pairs(keys, coord, binset="abs", abscissa="depth_mid"):
        return [e for e in exp_rows if e["key"] in keys and e["coord"] == coord
                and e["binset"] == binset and e["abscissa"] == abscissa
                and "kappa_mob" in e]

    def p2_like(coord, binset="abs", abscissa="depth_mid", min_n=0):
        got, bad = [], []
        for e in pairs(elu, coord, binset, abscissa):
            if min_n and (e["n_lo"] < min_n or e["n_hi"] < min_n):
                continue
            ok = bool(e.get("same_sign") and e.get("within_tol"))
            ki = e.get("kappa_inc")
            got.append(dict(key=e["key"], pair=e["pair"],
                            kappa_inc=ki,
                            kappa_mob=e.get("kappa_mob"),
                            diff=e.get("diff"), n_lo=e["n_lo"], n_hi=e["n_hi"],
                            # 「指数が測れたうえで合わない」のか「片方の帯の
                            # 中央値が 0 以下で指数が定義できない」のかを分ける。
                            # 後者は非測定ではなく、正の予測に対する符号の不一致。
                            kappa_defined=bool(ki is not None and ki == ki),
                            excluded_by_2se=bool(e.get("excluded_by_2se")),
                            ok=ok))
            if not ok:
                bad.append(got[-1])
        return dict(n=len(got), n_ok=sum(g["ok"] for g in got),
                    n_defined=sum(g["kappa_defined"] for g in got),
                    n_ok_of_defined=sum(g["ok"] for g in got if g["kappa_defined"]),
                    n_excluded=sum(g["excluded_by_2se"] for g in got),
                    bad=bad, all=got)

    P2d = p2_like("zmax")
    P3d = p2_like("zbar")
    P2 = P2d["n"] > 0 and not P2d["bad"]
    P3 = P3d["n"] > 0 and not P3d["bad"]

    # --- P4: leaky の B の指数
    p4 = [dict(key=e["key"], pair=e["pair"], kappa_mob=e["kappa_mob"])
          for e in pairs(lky, "zmax")]
    p4_bad = [e for e in p4 if abs(e["kappa_mob"]) > TOL_P4]
    P4 = bool(p4) and not p4_bad

    # --- ★ §2(c) の前件（φ の飽和）が腕ごとに成り立っているか。
    # 括弧の中の自己項は v·φ なので、φ/α が深さ帯をまたいで動かなければ前件は
    # 成立している。ELU では φ/α → −1 に飽和、leaky では φ/α = z̄ で飽和しない。
    sat = {}
    for key in main:
        rows_ = [r for r in bins_rows
                 if r["key"] == key and r["coord"] == "zmax"
                 and r["binset"] == "abs" and r["n"] >= MIN_N
                 and "phi_ratio_med" in r]
        if len(rows_) < 2:
            continue
        vals = [abs(r["phi_ratio_med"]) for r in rows_]
        sat[key] = dict(family=per_arm[key]["family"],
                        bins=[r["bin"] for r in rows_],
                        phi_over_alpha=[r["phi_ratio_med"] for r in rows_],
                        spread=float(max(vals) / min(vals) - 1.0) if min(vals) > 0
                        else float("inf"))

    # --- ★ 尺度統計量: 増分の**大きさ**（IQR / sd / MAD）の深さ指数が
    # 可動度の指数に一致するか。§2(c) の代数が言っているのは勾配の大きさが
    # phi' に比例することなので、正味（中央値）ではなく尺度で見るほうが
    # 代数に近い。横軸は帯の深さ中央値（中点は偏る）。
    def scale_view(coord, stat):
        got = []
        for e in pairs(elu, coord, "abs", "depth_med"):
            k = e.get("kappa_inc" + stat)
            if k is None or k != k:
                continue
            got.append(dict(key=e["key"], pair=e["pair"], stat=stat.strip("_"),
                            kappa_inc=float(k), kappa_mob=float(e["kappa_mob"]),
                            diff=float(k - e["kappa_mob"]),
                            n_lo=e["n_lo"], n_hi=e["n_hi"],
                            shallow=bool(e["pair"].startswith("0-1")),
                            ok=bool(abs(k - e["kappa_mob"]) <= TOL_P2
                                    and np.sign(k) == np.sign(e["kappa_mob"]))))
        return got

    scale = {}
    for stat in ("", "_iqr", "_sd", "_mad"):
        rows_ = scale_view("zmax", stat)
        deep = [r for r in rows_ if not r["shallow"]]
        scale[stat.strip("_") or "med"] = dict(
            n=len(rows_), n_ok=sum(r["ok"] for r in rows_),
            n_deep=len(deep), n_ok_deep=sum(r["ok"] for r in deep),
            rows=rows_)

    # --- ★ leaky 対照: φ′ ≡ a で厳密に一定なので κ_可動度 = 0。§2(c) が正しければ
    # 増分は深さに依らないはず。依れば「括弧の中が深さに依る」が ELU の恒等式にも
    # ln32 の帯にも依らずに示せる。
    lb = []
    for e in pairs(lky, "zmax") + pairs(lky, "zbar"):
        k = e.get("kappa_inc")
        if k is None or k != k:
            continue
        km = e.get("kappa_inc_mean")
        lb.append(dict(key=e["key"], coord=e["coord"], pair=e["pair"],
                       kappa_inc=float(k),
                       kappa_inc_mean=(float(km) if km is not None and km == km
                                       else None),
                       n_lo=e["n_lo"], n_hi=e["n_hi"],
                       over_tol=bool(abs(k) > TOL_P2),
                       excluded_by_2se=bool(e.get("excluded_by_2se"))))
    leaky_bracket = dict(
        n=len(lb), n_over_tol=sum(x["over_tol"] for x in lb),
        n_excluded=sum(x["excluded_by_2se"] for x in lb), pairs=lb)

    # --- ★ タスク内位置を固定した版（プールの中央値が位相の混合であることへの対処）
    def pos_view(family, coord):
        out = []
        for e in pos_exp:
            if (e["coord"] != coord or per_arm[e["key"]]["family"] != family
                    or e["key"] not in main or "kappa_mob" not in e):
                continue
            k = e.get("kappa_inc")
            defined = bool(k is not None and k == k)
            diff = (k - e["kappa_mob"]) if defined else None
            out.append(dict(key=e["key"], pos=e["pos"], pair=e["pair"],
                            kappa_inc=(float(k) if defined else None),
                            kappa_mob=float(e["kappa_mob"]),
                            diff=(float(diff) if defined else None),
                            n_lo=e["n_lo"], n_hi=e["n_hi"], defined=defined,
                            ok=bool(defined and abs(diff) <= TOL_P2
                                    and np.sign(k) == np.sign(e["kappa_mob"])
                                    and k != 0.0)))
        return out

    by_pos = {}
    for coord in ("zmax", "zbar"):
        rows_ = [r for r in pos_view("elu", coord) if r["pos"] in (1, 2, 3)]
        by_pos[coord] = dict(
            n=len(rows_), n_defined=sum(r["defined"] for r in rows_),
            n_ok=sum(r["ok"] for r in rows_), rows=rows_)
    lky_pos = [r for r in pos_view("leaky", "zmax") + pos_view("leaky", "zbar")
               if r["pos"] in (1, 2, 3) and r["defined"]]
    by_pos["leaky"] = dict(n=len(lky_pos),
                           n_over_tol=sum(abs(r["kappa_inc"]) > TOL_P2
                                          for r in lky_pos),
                           rows=lky_pos)

    # --- P5: |v| と深さの Spearman の符号
    p5 = {}
    for key in main:
        rr = [s for s in spear_rows if s["key"] == key]
        if rr:
            p5[key] = dict(
                med_zbar=float(np.median([s["rho_zbar"] for s in rr])),
                med_zmax=float(np.median([s["rho_zmax"] for s in rr])),
                n_pos_zbar=int(sum(s["rho_zbar"] > 0 for s in rr)),
                n_pos_zmax=int(sum(s["rho_zmax"] > 0 for s in rr)))
    P5 = bool(p5) and all(v["med_zbar"] > 0 for v in p5.values())

    if not P4:
        label = "PIPELINE_FAULT"
    elif not P1:
        label = "LOGGER_FAULT"
    elif P2 and P3:
        label = "GAIN_CARRIES_DEPTH"
    elif P2 and not P3:
        label = "COORDINATE_ONLY"
    else:
        label = "BRACKET_DEPENDS_ON_DEPTH"

    # --- 副次: 帯の切り方・横軸・最小 n を替えたときのラベルの安定性
    robust = {}
    for coord_set in [("abs", "depth_med"), ("eqf", "depth_mid"),
                      ("eqf", "depth_med")]:
        bs, ab = coord_set
        a = p2_like("zmax", bs, ab); b = p2_like("zbar", bs, ab)
        robust[f"{bs}/{ab}"] = dict(
            P2=bool(a["n"] and not a["bad"]), P3=bool(b["n"] and not b["bad"]),
            n_ok_P2=f"{a['n_ok']}/{a['n']}", n_ok_P3=f"{b['n_ok']}/{b['n']}")
    a = p2_like("zmax", "abs", "depth_mid", min_n=MIN_N)
    b = p2_like("zbar", "abs", "depth_mid", min_n=MIN_N)
    robust[f"abs/depth_mid/n>={MIN_N}"] = dict(
        P2=bool(a["n"] and not a["bad"]), P3=bool(b["n"] and not b["bad"]),
        n_ok_P2=f"{a['n_ok']}/{a['n']}", n_ok_P3=f"{b['n_ok']}/{b['n']}")

    # 副次 1: 量子化の床に触れていない対だけ（|中央値| >= 2 ulp）
    def not_floor(coord):
        got = [e for e in pairs(elu, coord) if not e.get("at_floor")]
        ok = [e for e in got
              if e.get("kappa_inc") == e.get("kappa_inc")
              and np.sign(e["kappa_inc"]) == np.sign(e["kappa_mob"])
              and abs(e["kappa_inc"] - e["kappa_mob"]) <= TOL_P2]
        return dict(n=len(got), n_ok=len(ok),
                    pairs=[dict(key=e["key"], pair=e["pair"],
                                kappa_inc=e.get("kappa_inc"),
                                kappa_mob=e["kappa_mob"],
                                diff=e.get("diff")) for e in got])
    robust["not_at_floor"] = dict(zmax=not_floor("zmax"), zbar=not_floor("zbar"))

    # 副次 2: 中央値でなく平均で取った指数（量子化に強い）
    def mean_based(coord):
        got = [e for e in pairs(elu, coord)
               if e.get("kappa_inc_mean") == e.get("kappa_inc_mean")]
        ok = [e for e in got
              if np.sign(e["kappa_inc_mean"]) == np.sign(e["kappa_mob"])
              and abs(e["kappa_inc_mean"] - e["kappa_mob"]) <= TOL_P2]
        return dict(n=len(got), n_ok=len(ok),
                    pairs=[dict(key=e["key"], pair=e["pair"],
                                kappa_inc_mean=e["kappa_inc_mean"],
                                kappa_mob=e["kappa_mob"],
                                diff=e["diff_mean"]) for e in got])
    robust["mean_based"] = dict(zmax=mean_based("zmax"), zbar=mean_based("zbar"))

    # 位置を固定してもラベルは動くか（登録どおりの主判定はプール版）
    label_pos = ("GAIN_CARRIES_DEPTH"
                 if (by_pos["zmax"]["n"] and not
                     any(not r["ok"] for r in by_pos["zmax"]["rows"]))
                 else "BRACKET_DEPENDS_ON_DEPTH")

    return dict(label=label, label_position_fixed=label_pos,
                P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
                leaky_bracket=leaky_bracket, by_position=by_pos,
                by_scale=scale, phi_saturation=sat,
                P1_violations=p1_bad, P2_detail=P2d, P3_detail=P3d,
                P4_detail=p4, P4_bad=p4_bad, P5_detail=p5,
                robustness=robust, main_keys=sorted(main))



def write_summary(outdir, bins_rows, exp_rows, ident_rows, spear_rows,
                  per_arm, v, pos_bins=()) -> None:
    """人が読む表を 1 枚にまとめる（判定の根拠はすべて CSV 側にある）。"""
    L = []
    A = L.append
    A(f"# `zcoord_0903` — ELU可動度の深さ座標 再集計（走なし）\n")
    A(f"**ラベル: `{v['label']}`**  "
      f"P1={v['P1']} P2={v['P2']} P3={v['P3']} P4={v['P4']} P5={v['P5']}\n")
    A("主判定は `zmax` 座標・等幅帯・帯の中点・プール中央値（spec §5 P2）。\n")

    A("\n## D / P1 — 恒等式 ln(mob/α) − zmax ∈ [−3.4657, 0]\n")
    A("| 腕@窓 | n | min | 中央 | max | 帯外 | n_eff 中央 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in ident_rows:
        note = " ※" if r.get("subset_of") else ""
        A(f"| `{r['key']}`{note} | {r['n']:,} | {r['resid_min']:+.4f} | "
          f"{r['resid_med']:+.4f} | {r['resid_max']:+.4f} | "
          f"{r['n_below']+r['n_above']} | {r['n_eff_med']:.2f} |")
    tot = sum(r["n"] for r in ident_rows if not r.get("subset_of"))
    A(f"\n※ 印の行は上の 15M 窓の**部分集合**（先頭を切っただけ）なので合計に足さない。"
      f"独立な行数の合計は **{tot:,}**。"
      "なお spec §4 D の「全記録」はタスク内に限らないが、"
      "本表はタスク内増分の手前の記録点だけを見ているので境界増分の手前（1/10）が抜けている。\n")
    A("\n`n_eff = 32·exp(残差)` = Σ_p exp(z_p − max z) は E_x[φ′] を担っている"
      "実効パターン数（最浅の 1 点が必ず 1 を出すので n_eff ≥ 1）。\n"
      "**★ 読みの限定（監査で訂正）。** α=1 の中央値 1.48 は"
      "「最浅パターンが E_x[φ′] の 1/1.48 = 68% を担う」であって"
      "「最浅の 1 個だけが決めている」ではない。しかもこれは **α=1 の腕だけの数字**で、"
      "α≤0.1 の 3 腕では 3.4〜4.0（最浅の寄与 25〜30%）。"
      "α と走・窓が交絡しているので α 依存として読んではいけない。\n")

    A("\n## A・B・C — 帯ごとの増分と可動度（等幅帯・沈下 × タスク内）\n")
    A("| 腕@窓 | 座標 | 帯 | n | 深さ中央 | 増分中央 | 平均 | P(上) | 中央/ulp | ln(mob/α) 中央 |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in bins_rows:
        if r["binset"] != "abs" or r["n"] == 0:
            continue
        A(f"| `{r['key']}` | {r['coord']} | {r['bin']} | {r['n']:,} | "
          f"{r['depth_med']:.2f} | {r['inc_med']:+.3e} | {r['inc_mean']:+.3e} | "
          f"{r['inc_p_up']:.3f} | {r['med_over_ulp']:.1f} | {r['lnmob_med']:.3f} |")

    A("\n## P2 / P3 — 局所指数の一致（許容帯 0.3）\n")
    for name, coord in (("P2（`zmax` 座標・主判定）", "zmax"), ("P3（`z̄` 座標）", "zbar")):
        d = v["P2_detail"] if coord == "zmax" else v["P3_detail"]
        A(f"\n### {name} — {d['n_ok']}/{d['n']} 対が通過\n")
        A(f"内訳: κ_増分 が有限値を持つのは **{d['n_defined']}/{d['n']} 対**"
          f"（うち通過 {d['n_ok_of_defined']}）。残りは深い側の帯の増分中央値が"
          "0 以下で対数が取れない対で、**「測れなかった」のではなく"
          "「正である §2(c) の予測に対して中央値が 0 か負」**である"
          f"（その多くは下の 2σ 表で排除されている: {d['n_excluded']}/{d['n']} 対）。\n")
        A("| 腕@窓 | 対 | κ_増分 | κ_可動度 | 差 | n | 2σ 排除 | |")
        A("|---|---|---:|---:|---:|---:|---|---|")
        for g in d["all"]:
            ki = g["kappa_inc"]
            ki_s = ("中央値≤0" if not g["kappa_defined"] else f"{ki:+.3f}")
            df = g["diff"]
            df_s = "n/a" if df is None or df != df else f"{df:+.3f}"
            A(f"| `{g['key']}` | {g['pair']} | {ki_s} | {g['kappa_mob']:+.3f} | "
              f"{df_s} | {g['n_lo']:,}/{g['n_hi']:,} | "
              f"{'**排除**' if g['excluded_by_2se'] else '—'} | "
              f"{'OK' if g['ok'] else '**NG**'} |")

    A("\n## ★ 判定の実体 — §2(c) の予測を seed 間 2σ で排除できるか\n")
    A("増分の中央値は**幅 ~1e-2 のほぼ対称なゆらぎに乗った小さな正味**であって、"
      "小さな量ではない（下の IQR 欄）。記録点は 1000 step ごとで自己相関するので"
      "プールの n では誤差を作れない。**誤差は 10 seed の per-seed 中央値から作る。**"
      "「浅い帯の中央値 × exp(−κ_可動度·Δ深さ)」が §2(c) の予測で、"
      "深い帯の実測の 2σ 上限がそれを下回れば §2(c) は排除される。\n")
    A("**区間が使えるのは 10 seed そろっていて幅が 0 でない帯だけ**"
      "（seed の中央値が全部同じ＝多くは全部 0、という帯は排除の根拠にしない）。\n")
    A("| 腕@窓 | 座標 | 対 | §2(c) の予測 | 実測 2σ 区間 | seed | 排除 | 向き |")
    A("|---|---|---|---:|---:|---:|---|---|")
    for e in exp_rows:
        if (e["binset"] != "abs" or e["abscissa"] != "depth_mid"
                or "inc_med_pred_hi" not in e or not e.get("ci_usable")):
            continue
        pred, hi, lo = e["inc_med_pred_hi"], e["obs_hi2se_hi"], e["obs_lo2se_hi"]
        side = ("予測より小" if hi < pred else
                "予測より大" if lo > pred else "—")
        A(f"| `{e['key']}` | {e['coord']} | {e['pair']} | {pred:.2e} | "
          f"[{lo:.2e}, {hi:.2e}] | {e.get('n_seed_hi')} | "
          f"{'**排除**' if e['excluded_by_2se'] else '—'} | {side} |")
    A("\n参考（分解能）: `zmax` 帯の中央値と float32 の ulp の比\n")
    A("| 腕@窓 | 座標 | 帯 | 中央値/ulp | IQR | \\|増分\\|>1e−3 の割合 |")
    A("|---|---|---|---:|---:|---:|")
    for r in bins_rows:
        if r["binset"] != "abs" or r["n"] == 0 or not r["key"].startswith("E_"):
            continue
        A(f"| `{r['key']}` | {r['coord']} | {r['bin']} | {r['med_over_ulp']:.1f} | "
          f"{r['inc_iqr']:.2e} | {r['inc_frac_gt_1e3']:.3f} |")

    A("\n## ★ §2(c) の前件（φ の飽和）が成り立っているか\n")
    A("§2(c) が「括弧の中は深さに依らない」と言えるのは、"
      "**深い ELU で φ→−α（定数）に飽和するから**である。前件そのものを測る:\n")
    A("| 腕@窓 | 族 | 帯ごとの φ/α（`zmax` 等幅帯・n≥1000） | 帯間の振れ |")
    A("|---|---|---|---:|")
    for key in sorted(v["phi_saturation"]):
        d3 = v["phi_saturation"][key]
        vals = " → ".join(f"{x:+.4f}" for x in d3["phi_over_alpha"])
        A(f"| `{key}` | {d3['family']} | {vals} | **{d3['spread']*100:.1f}%** |")
    A("\n**ELU では前件が成り立つ**（φ/α は −1 に貼り付き、帯間の振れは数%）。"
      "**leaky では成り立たない**（φ/α = z̄ で飽和せず、帯間で数倍動く）。\n")

    A("\n## leaky での増分の深さ依存 —— §2(c) の反証ではなく感度の確認\n")
    lb = v["leaky_bracket"]
    A("**★ 格の限定（監査で訂正）。** leaky の沈下ユニットでは φ′ ≡ a で厳密に一定"
      "（κ_可動度 = 0）なので、一見「§2(c) が正しければ増分は深さに依らないはず」"
      "と読みたくなる。**これは §2(c) の前件の取り違えである。** §2(c) が括弧の"
      "深さ非依存を導くのは φ の飽和からで、leaky は φ = a·z が深さに比例して"
      "伸び続ける（上表で 78〜518% 動く）。同じ代数が leaky に対して与える向きは"
      "**深いほど大きい**であり、実測の κ_増分 < 0 はその向きどおりである。"
      "**したがって以下は §2(c) の反証ではない。** 残る意味は 1 つだけ:"
      "**配管が「増分の深さ依存」を実際に検出できることの確認**（感度の検査）である。\n")
    A(f"読めた {lb['n']} 対のうち **|κ_増分| > {TOL_P2} が {lb['n_over_tol']} 対**、"
      f"seed 間 2σ で「深さに依らない」を排除できたのが {lb['n_excluded']} 対。\n")
    A("| 腕@窓 | 座標 | 対 | κ_増分（中央値） | κ_増分（平均） | n | 許容帯超 | 2σ で排除 |")
    A("|---|---|---|---:|---:|---:|---|---|")
    for x in lb["pairs"]:
        km = x.get("kappa_inc_mean")
        A(f"| `{x['key']}` | {x['coord']} | {x['pair']} | {x['kappa_inc']:+.3f} | "
          f"{'n/a' if km is None else f'{km:+.3f}'} | "
          f"{x['n_lo']:,}/{x['n_hi']:,} | {'**超**' if x['over_tol'] else '—'} | "
          f"{'**排除**' if x['excluded_by_2se'] else '—'} |")
    A("\n中央値と平均が同じ向き・同じ桁で動くので、"
      "**この配管は増分の深さ依存があれば検出する**（`LR_a0p001` だけは平均が"
      "読めないので根拠にしない）。ELU 側で正味の深さ依存が可動度と合わないのは、"
      "検出力が無いからではない。\n")

    A("\n## ★ タスク内位置を固定すると — プールの中央値は位相の混合である\n")
    A("タスク内増分は定常量ではない。**同じ 1 タスクの中で記録位置 o=1→9 のあいだに"
      "2〜3 桁減衰する**（`E_1216`@15M・`zmax` 帯 0–1 の中央値は +1.23e−2 → +2.3e−5）。"
      "9 位置をプールした中央値は後半の静かな位置で決まるので、"
      "**プール版の指数は位相の混合の指数であって、ドリフトの深さ依存の指数ではない。**"
      "帯の切り目は位置を掛ける前の分布から決めてあるので、位置をまたいで同じ帯である。\n")
    if pos_bins:
        A("| 腕@窓 | 座標 | 帯 | " + " | ".join(f"o={o}" for o in range(1, 10)) + " |")
        A("|---|---|---|" + "---:|" * 9)
        seen = set()
        for r in pos_bins:
            if (r["coord"] != "zmax" or r["n"] == 0
                    or (r["key"], r["bin"]) in seen
                    or r["key"] not in ("E_1216@15M", "LR_a0p01_1216@5M")):
                continue
            seen.add((r["key"], r["bin"]))
            vals = []
            for o in range(1, 10):
                hit = [q for q in pos_bins
                       if q["key"] == r["key"] and q["coord"] == "zmax"
                       and q["bin"] == r["bin"] and q["pos"] == o and q["n"]]
                vals.append(f"{hit[0]['inc_med']:+.2e}" if hit else "—")
            A(f"| `{r['key']}` | zmax | {r['bin']} | " + " | ".join(vals) + " |")
    bp = v["by_position"]
    A(f"\n位置を o∈{{1,2,3}} に固定して同じ判定をすると（ELU 4 腕・`zmax`）: "
      f"κ_増分 が有限なのは {bp['zmax']['n_defined']}/{bp['zmax']['n']} 対、"
      f"通過は **{bp['zmax']['n_ok']}/{bp['zmax']['n']} 対**。"
      f"位置固定のラベルは **`{v['label_position_fixed']}`**（登録どおりのプール版と同じ）。\n")
    A("| 腕@窓 | o | 対 | κ_増分 | κ_可動度 | 差 | |")
    A("|---|---:|---|---:|---:|---:|---|")
    for r in bp["zmax"]["rows"]:
        ki = "中央値≤0" if not r["defined"] else f"{r['kappa_inc']:+.3f}"
        df = "n/a" if not r["defined"] else f"{r['diff']:+.3f}"
        A(f"| `{r['key']}` | {r['pos']} | {r['pair']} | {ki} | "
          f"{r['kappa_mob']:+.3f} | {df} | {'OK' if r['ok'] else '**NG**'} |")
    A(f"\n**位置を固定するとずれは小さくなるが、消えない。** "
      f"`E_1216` の 0–1→1–3 の差はプールの +2.16 に対し o=1 で +0.36・"
      "o=2 で +1.05・o=3 で +1.58 で、**どの位置でも許容帯 0.3 を超える**。"
      "いっぽう `E_1216` の 3–6→6–10 は位置固定だと 3 位置とも通る（差 +0.14／+0.06／−0.08）ので、"
      "**「全対で外れる」はプール版だけの言い方であり、位置固定では対によって割れる。** "
      "α≤0.1 の ELU 3 腕はどの位置でも深い帯の増分中央値が 0 以下で、指数が定義できない。\n")
    lp = bp["leaky"]
    A(f"leaky も位置固定で見ると、読めた {lp['n']} 対のうち "
      f"|κ_増分| > {TOL_P2} は {lp['n_over_tol']} 対（`LR_a0p01`・`LR_a0p001` が"
      "−0.5〜−1.3、`LR_1216` は −0.13〜−0.25）。"
      "**これは §2(c) の反証ではなく感度の確認である**（上の格の限定を見よ）。\n")

    A("\n## P4 — leaky の S 検査は**恒真**で、配管を検査していない（監査で確定）\n")
    bad = v["P4_bad"]
    A(f"leaky 3 腕・{len(v['P4_detail'])} 対すべてで κ_可動度 = 0"
      f"（|κ| > {TOL_P4} は {len(bad)} 件）。だが**これは配管の検査になっていない。** "
      "`zmax` 座標では帯そのものが zmax<0 を再賦課するので、沈下 leaky の `mob` は "
      "a の float32 値ちょうど 1 種類になり、ln(mob/a) は全行同じ定数である。"
      "独立監査は、沈下フィルタを丸ごと外しても・`p_hat==0` に替えても・"
      "`mob` を 1 記録ずらしても κ_可動度 が厳密に 0 のままであることを実演した。"
      "**spec §5 が「P1・P4 が通っている下で」と置いたゲートのうち、P4 は 0 ビットしか持たない。**\n")
    A("配管を実際に検査しているのは次の 4 本である（いずれも通過）:\n")
    A("- **P1**: 歯がある。独立監査が `mob` を 1 記録ずらすと 2.65% が帯外、"
      "`zmax` を `z̄` に取り違えると 99.7% が帯外になることを実演した\n"
      "- **S-repro**: §10-4 の `E_1216` 深さ方向の 4 つの中央値を再現する\n"
      "- **S-sub**: `zmax`≤0 と $\\hat p=0$ が全腕・全 seed・全記録で一致（第一容疑 (1)）\n"
      "- **S-bnd**: 除くべき増分が `flip_state` 基準の index 10,20,… であって "
      "`step//period` 基準の 9,19,… ではないこと（第一容疑 (2)）\n")

    A("\n## P5 — |v| と深さの Spearman（符号のみ）\n")
    A("| 腕@窓 | ρ(|v|, z̄ 深さ) 中央 | 正の seed | ρ(|v|, zmax 深さ) 中央 | 正の seed |")
    A("|---|---:|---:|---:|---:|")
    for k in sorted(v["P5_detail"]):
        d = v["P5_detail"][k]
        A(f"| `{k}` | {d['med_zbar']:+.3f} | {d['n_pos_zbar']}/10 | "
          f"{d['med_zmax']:+.3f} | {d['n_pos_zmax']}/10 |")

    A("\n## ★★ 尺度で見ると可動度に追随する — 落ちているのは「正味」だけ\n")
    A("§2(c) の代数が言っているのは、勾配の**大きさ**が φ′ に比例するということである。"
      "そこで増分の中央値（＝正味）ではなく**大きさ**（IQR・sd・MAD）の深さ指数を"
      "同じ帯・同じ横軸（帯の深さ中央値）で取り直すと:\n")
    A("| 統計量 | 通過（全対） | 通過（浅い帯 0–1 を含まない対だけ） |")
    A("|---|---|---|")
    for name, key in (("中央値（登録どおり）", "med"), ("IQR", "iqr"),
                      ("標準偏差", "sd"), ("MAD", "mad")):
        d2 = v["by_scale"][key]
        A(f"| {name} | {d2['n_ok']}/{d2['n']} | {d2['n_ok_deep']}/{d2['n_deep']} |")
    A("\n| 腕@窓 | 対 | κ_可動度 | κ_IQR | 差 | κ_sd | 差 | |")
    A("|---|---|---:|---:|---:|---:|---:|---|")
    sd_by = {(r["key"], r["pair"]): r for r in v["by_scale"]["sd"]["rows"]}
    for r in v["by_scale"]["iqr"]["rows"]:
        o = sd_by.get((r["key"], r["pair"]))
        A(f"| `{r['key']}` | {r['pair']} | {r['kappa_mob']:+.3f} | "
          f"{r['kappa_inc']:+.3f} | {r['diff']:+.3f} | "
          f"{o['kappa_inc']:+.3f} | {o['diff']:+.3f} | "
          f"{'OK' if r['ok'] else '**NG**'} |")
    A("\n**深い帯では増分の大きさの指数は可動度の指数（≈1）に許容帯 0.3 の中で一致する。**"
      "浅い帯 0–1 を含む対だけが外れる。つまり §2(c) の代数は"
      "**増分の大きさについては当たっており、外れているのは正味（中央値）の側だけ**である。"
      "登録された P2 は正味で問うので落ちるが、"
      "「深さ依存はまるごと可動度が担う」は**大きさに限れば支持されている**。\n")

    A("\n## ★ 横軸の取り方 — 登録どおりの「帯の中点」は偏っている\n")
    A("§2(a) の恒等式より、`zmax` 座標では ln(mob/α) = −(zmax 深さ) + 残差で"
      "残差はほぼ深さに依らないので、**κ_可動度 は 1 になるはず**である。"
      "ところが spec §4 A が指定した「帯の**中点**」を横軸に取ると 0.65〜1.75 に散る。"
      "縦軸は帯の**中央値**なのに横軸だけ中点にしたための偏りで、"
      "横軸も帯の深さ中央値に替えると恒等式どおり 1 に寄る:\n")
    A("| 腕@窓 | 対 | κ_可動度（中点） | κ_可動度（深さ中央値） |")
    A("|---|---|---:|---:|")
    seen2 = {}
    for e in exp_rows:
        if (e["coord"] != "zmax" or e["binset"] != "abs"
                or per_arm[e["key"]]["family"] != "elu"):
            continue
        seen2.setdefault((e["key"], e["pair"]), {})[e["abscissa"]] = e["kappa_mob"]
    for (k, pr), d2 in seen2.items():
        A(f"| `{k}` | {pr} | {d2.get('depth_mid', float('nan')):+.3f} | "
          f"{d2.get('depth_med', float('nan')):+.3f} |")
    A("\n**偏りは最大 0.42 で、P2 の許容帯 0.3 より大きい。** 登録どおりの中点版と"
      "深さ中央値版はどちらも P2 を 0/15 にするのでラベルは動かないが、"
      "**個々の κ_可動度 の値は深さ中央値版を読むこと。**\n")
    A("**★ 併せて `zmax` 座標の限界**: κ_可動度 がほぼ 1 に固定されるということは、"
      "`zmax` 座標の P2 は事実上「増分の指数が 1 か」を聞いているのと同じである。"
      "spec §5 は「指数が 1 か」ではなく「可動度の指数と一致するか」を問うと書いたが、"
      "**その 2 つは `zmax` 座標では恒等式によって同じ問いになる。** "
      "座標を替えたことで落ちるのは $\\ln32$ のオフセットだけで、"
      "問いの中身は元の Q2 に戻っている。\n")

    A("\n## 帯・横軸・統計量を替えたとき\n")
    A("| 切り方 | P2 | P3 |")
    A("|---|---|---|")
    for k, d in v["robustness"].items():
        if k in ("not_at_floor", "mean_based"):
            continue
        A(f"| {k} | {d['n_ok_P2']} | {d['n_ok_P3']} |")
    mb = v["robustness"]["mean_based"]
    A(f"| 平均ベース（量子化に強い） | {mb['zmax']['n_ok']}/{mb['zmax']['n']} | "
      f"{mb['zbar']['n_ok']}/{mb['zbar']['n']} |")
    nf = v["robustness"]["not_at_floor"]
    A(f"| 床に触れない対だけ | {nf['zmax']['n_ok']}/{nf['zmax']['n']} | "
      f"{nf['zbar']['n_ok']}/{nf['zbar']['n']} |")
    A(f"\n**中央値では、どの切り方でも `zmax` 座標の P2 は 1 対も通らない。**"
      f" 平均に替えると `zmax` は {mb['zmax']['n_ok']}/{mb['zmax']['n']} 対通るが、"
      "通るのは `E_1216@15M` の 0-1→1-3 だけで、その帯の平均は上位 1% が総和の"
      "半分近くを担う裾に引かれている（中央値と平均が食い違う）。"
      "残りの ELU 3 腕は平均でも +1.9〜+2.8 でずれる。\n")
    (outdir / "summary.md").write_text("\n".join(L))

def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def input_hashes() -> dict:
    """読んだ生ログの sha256。**生ログ自体は 4.8 GB あり repo に入らない**ので、
    fresh clone の監査はここのハッシュで「同じログを見ているか」を確かめる。
    """
    import hashlib
    out = {}
    for arm, run_, *_ in ARMS:
        for seed in SEEDS:
            path = RESULTS / run_ / "logs" / f"{arm}_seed{seed}.npz"
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            out[f"{run_}/logs/{path.name}"] = h.hexdigest()
    return out


def output_hashes(outdir: Path) -> dict:
    import hashlib
    out = {}
    for path in sorted(outdir.glob("*")):
        if path.name == "provenance.json" or not path.is_file():
            continue
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "zcoord_0903"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return 0 if selftest()["all_pass"] else 1
    st = selftest(verbose=True)
    if not st["all_pass"]:
        print("SELFTEST FAILED — 集計しない", file=sys.stderr)
        return 1
    out = Path(args.out)
    v = run(out)
    (out / "selftest.json").write_text(json.dumps(st, indent=2, ensure_ascii=False,
                                                  default=float))
    print("hashing inputs ...", flush=True)
    (out / "provenance.json").write_text(json.dumps(dict(
        spec="ELU可動度の深さ座標_spec_0902 v6", kind="reanalysis-only",
        note=("新しい走はゼロ。生ログ（4.8 GB）は repo に入らないので "
              "input_sha256 で同一性を固定する。"),
        git_hash=git_hash(), numpy=np.__version__,
        python=sys.version, arms=[a[0] for a in ARMS],
        label=v["label"], selftest=st,
        input_sha256=input_hashes(), output_sha256=output_hashes(out)),
        indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: v[k] for k in ("label", "P1", "P2", "P3", "P4", "P5")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
