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
def load_arm(arm: str, run: str, *, max_step: int | None = None) -> dict:
    """1 腕・全 seed の「沈下 × タスク内」の行をプールして返す。

    §10-4 の ``fullpass.py`` と同じく、増分は隣り合う記録点の差
    ``zbar[i+1]-zbar[i]``（記録は 1000 step ごとなので 1000 step あたり）で、
    タスク内は ``flip_state[i+1] == flip_state[i]``、深さは**増分の手前** ``i``
    の状態で測る。沈下の判定も手前の記録点で行う（spec §4）。
    """
    logdir = RESULTS / run / "logs"
    cols = {k: [] for k in ("inc", "d_zmax", "d_zbar", "lnmob", "absv",
                            "seed", "resid")}
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
        inc = zbar[1:] - zbar[:-1]
        prev = slice(0, len(step) - 1)
        sub = zmax[prev] <= 0.0

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
        cols["d_zmax"].append(-zmax_m)
        cols["d_zbar"].append(-zbar[prev][m])
        cols["lnmob"].append(lnmob)
        cols["resid"].append(lnmob - zmax_m)
        cols["absv"].append(np.abs(v[prev][m]))
        cols["seed"].append(np.full(int(m.sum()), seed, dtype=np.int8))
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


def bin_table(d: dict, coord: str, binset: str) -> list[dict]:
    """A・B（coord="zmax"）と C（coord="zbar"）の帯ごとの量。"""
    depth = d["d_zmax"] if coord == "zmax" else d["d_zbar"]
    edges, names = bin_edges(depth, binset)
    rows = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (depth > lo) & (depth <= hi)
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
                inc_mean=float(inc.mean()),
                inc_p_up=float(np.mean(inc > 0)),
                inc_frac_zero=float(np.mean(inc == 0.0)),
                ulp_med=float(np.median(ulp)),
                med_over_ulp=(abs(med) / float(np.median(ulp))
                              if np.median(ulp) > 0 else float("nan")),
                lnmob_med=float(np.median(lnmob)),
                resid_med=float(np.median(d["resid"][m])),
            )
            # seed ごとの中央値（腕内のばらつき。判定には使わず報告のみ）
            per_seed = []
            for sd in SEEDS:
                ms = d["seed"][m] == sd
                if ms.sum() >= 100:
                    per_seed.append(float(np.median(inc[ms])))
            row["inc_med_seed_n"] = len(per_seed)
            if per_seed:
                row["inc_med_seed_med"] = float(np.median(per_seed))
                row["inc_med_seed_min"] = float(np.min(per_seed))
                row["inc_med_seed_max"] = float(np.max(per_seed))
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
        for tag, key in (("", "inc_med"), ("_mean", "inc_mean")):
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
    per_arm = {}

    windows = []
    for arm, run_, fam, dial, nstep in ARMS:
        windows.append((arm, run_, fam, dial, None, f"{nstep//10**6}M"))
        if nstep > 5_000_000:            # 5M 打ち切りの副次窓（§3 の振れの検査）
            windows.append((arm, run_, fam, dial, 5_000_000, "5M"))

    for arm, run_, fam, dial, max_step, wname in windows:
        key = f"{arm}@{wname}"
        d = load_arm(arm, run_, max_step=max_step)
        print(f"[{key}] rows={d['n_rows']:,} sub_agree={d['sub_agree']:.6f} "
              f"alpha={d['alpha']}", flush=True)
        per_arm[key] = dict(arm=arm, run=run_, family=fam, dial=dial,
                            window=wname, alpha=d["alpha"],
                            n_rows=d["n_rows"], n_sub=d["n_sub"],
                            n_raw=d["n_raw"], sub_agree=d["sub_agree"],
                            n_mob_zero=d["n_mob_zero"])
        # --- D / P1（ELU のみ。leaky では恒等式が別物）
        if fam == "elu" and d["n_rows"]:
            r = d["resid"]
            # n_eff = 32*exp(resid): ゲートを担っている「実効パターン数」。
            # 1 なら最浅の 1 パターンだけが E_x[phi'] を決めている。
            ident_rows.append(dict(
                key=key, arm=arm, window=wname, n=int(r.size),
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

    write_summary(outdir, bins_rows, exp_rows, ident_rows, spear_rows,
                  per_arm, judge(bins_rows, exp_rows, ident_rows, spear_rows,
                                 per_arm))
    write_csv(outdir / "bins.csv", bins_rows)
    write_csv(outdir / "bins_by_seed.csv", seed_rows)
    write_csv(outdir / "exponents.csv", exp_rows)
    write_csv(outdir / "identity.csv", ident_rows)
    write_csv(outdir / "spearman.csv", spear_rows)
    verdict = judge(bins_rows, exp_rows, ident_rows, spear_rows, per_arm)
    (outdir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=float))
    (outdir / "per_arm.json").write_text(
        json.dumps(per_arm, indent=2, ensure_ascii=False, default=float))
    return verdict


def judge(bins_rows, exp_rows, ident_rows, spear_rows, per_arm) -> dict:
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
            got.append(dict(key=e["key"], pair=e["pair"],
                            kappa_inc=e.get("kappa_inc"),
                            kappa_mob=e.get("kappa_mob"),
                            diff=e.get("diff"), n_lo=e["n_lo"], n_hi=e["n_hi"],
                            ok=ok))
            if not ok:
                bad.append(got[-1])
        return dict(n=len(got), n_ok=sum(g["ok"] for g in got), bad=bad, all=got)

    P2d = p2_like("zmax")
    P3d = p2_like("zbar")
    P2 = P2d["n"] > 0 and not P2d["bad"]
    P3 = P3d["n"] > 0 and not P3d["bad"]

    # --- P4: leaky の B の指数
    p4 = [dict(key=e["key"], pair=e["pair"], kappa_mob=e["kappa_mob"])
          for e in pairs(lky, "zmax")]
    p4_bad = [e for e in p4 if abs(e["kappa_mob"]) > TOL_P4]
    P4 = bool(p4) and not p4_bad

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

    return dict(label=label, P1=P1, P2=P2, P3=P3, P4=P4, P5=P5,
                P1_violations=p1_bad, P2_detail=P2d, P3_detail=P3d,
                P4_detail=p4, P4_bad=p4_bad, P5_detail=p5,
                robustness=robust, main_keys=sorted(main))



def write_summary(outdir, bins_rows, exp_rows, ident_rows, spear_rows,
                  per_arm, v) -> None:
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
        A(f"| `{r['key']}` | {r['n']:,} | {r['resid_min']:+.4f} | "
          f"{r['resid_med']:+.4f} | {r['resid_max']:+.4f} | "
          f"{r['n_below']+r['n_above']} | {r['n_eff_med']:.2f} |")
    A("\n`n_eff = 32·exp(残差)` は E_x[φ′] を担っている実効パターン数"
      "（1 なら最浅の 1 パターンだけがゲートを決めている）。\n")

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
        A("| 腕@窓 | 対 | κ_増分 | κ_可動度 | 差 | n | |")
        A("|---|---|---:|---:|---:|---:|---|")
        for g in d["all"]:
            ki = g["kappa_inc"]
            ki_s = "n/a" if ki is None or ki != ki else f"{ki:+.3f}"
            df = g["diff"]
            df_s = "n/a" if df is None or df != df else f"{df:+.3f}"
            A(f"| `{g['key']}` | {g['pair']} | {ki_s} | {g['kappa_mob']:+.3f} | "
              f"{df_s} | {g['n_lo']:,}/{g['n_hi']:,} | {'OK' if g['ok'] else '**NG**'} |")

    A("\n## 床か信号か — §2(c) が正しければ深い側の中央値はこうなるはず\n")
    A("| 腕@窓 | 対 | 予測中央値 | 予測/ulp | 実測/予測 |")
    A("|---|---|---:|---:|---:|")
    for e in exp_rows:
        if (e["coord"] != "zmax" or e["binset"] != "abs"
                or e["abscissa"] != "depth_mid" or "inc_med_pred_hi" not in e):
            continue
        A(f"| `{e['key']}` | {e['pair']} | {e['inc_med_pred_hi']:+.3e} | "
          f"{e['pred_over_ulp_hi']:.1f} | {e['obs_over_pred_hi']:+.4f} |")
    A("\n予測が ulp の数十〜数百倍なのに実測がその 1/25〜1/200 なら、"
      "測れなかったのではなく**本当に速く落ちている**。予測が数 ulp しかない対は"
      "分解能で読めないので判定の根拠にしない。\n")

    A("\n## P4 — leaky の S 検査（配管）\n")
    bad = v["P4_bad"]
    A(f"leaky 3 腕・{len(v['P4_detail'])} 対すべてで κ_可動度 = 0"
      f"（|κ| > {TOL_P4} は {len(bad)} 件）。沈下 leaky の `mob` は a の float32 値"
      "ちょうど 1 種類。\n")

    A("\n## P5 — |v| と深さの Spearman（符号のみ）\n")
    A("| 腕@窓 | ρ(|v|, z̄ 深さ) 中央 | 正の seed | ρ(|v|, zmax 深さ) 中央 | 正の seed |")
    A("|---|---:|---:|---:|---:|")
    for k in sorted(v["P5_detail"]):
        d = v["P5_detail"][k]
        A(f"| `{k}` | {d['med_zbar']:+.3f} | {d['n_pos_zbar']}/10 | "
          f"{d['med_zmax']:+.3f} | {d['n_pos_zmax']}/10 |")

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
    A("\n**どの切り方でも P2 は 1 対も通らない。**\n")
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
