"""q3_boundary_race: タスク境界での「µ のジャンプ vs バイアスの追随」レース。

**事後計算・未事前登録**（spec なし・事前登録された判定基準なし。引用には事前登録つきの
昇格が要る）。既存ログの再解析のみで、新しい学習走は行わない。

## 仮説（発案は事後。ここは検証だけを行う）

condA の hidden unit の pre-activation は a = w·µ + w·δ + b。µ は周期内で固定、
δ は自由 5 次元（±0.5）の 32 パターン。周期内で定数の部分を

    s := w·µ + b = ||w||·||µ||·cos(w,µ) + b        （npz から厳密復元できる）
    M := max_δ (w·δ) = ||w_free||_1 / 2,  κ := M/||w||
    p_hat = 0  <=>  s + M <= 0                      （厳密。q3_margin_pooled で検証済み）

- H-a: タスク境界で flip ビットが 1 本反転すると w·µ が階段状にジャンプする。b は
  勾配で少しずつしか動かないので、1 step ではジャンプに追随できない。
- H-b: 境界後に b（および w）が s を回復させようとするが、時定数がある。
- H-c: ジャンプが下向きで深すぎた unit はその場で消灯し（s+M<=0 で勾配ゼロ）、
  死亡は境界直後に集中する。
- H-d: 生き残った unit も境界を 100 回通るうちに削られていく（ラチェット）。
- centered との対比: centered は境界直後だけ ||µ|| がパルス状に立つ
  （bulk 0.073 -> 境界 +1 で 0.99）ので、死は境界直後にさらに強く集中するはず。

## 定義（すべてコードに合わせる。採用した定義をここに明記する）

- `TAU = 0.05` と死亡判定 `death_events` は **`src/figures_ratchet_log.py` から直接
  import** して同一実装を使う（`ratchet.death_recover_periods = 1` に対応:
  「p_hat が TAU を下方クロスし、その後 1 周期のあいだ回復しない最初の記録点」）。
  p_hat = k/32 なので p_hat < 0.05 <=> k <= 1。
- 「点灯 (on)」= `p_hat > 0`（<=> s + M > 0、厳密）。勾配が入りうる状態。
- 「生存 (alive)」= `p_hat >= TAU`（= 死亡閾値の裏返し）。ハザードの分母に使う。
- 境界 B の flip は記録点 B と B+1 の**間**で起きる（`src/ratchet_log.py` の probe は
  ループ本体先頭で呼ばれる）。したがって offset 0 はまだ flip 前、変更後の flip_state が
  最初に見えるのは +1。実現 flip は 99 回（t=total の境界はループを通らない）。
- 記録グリッドは非一様（境界 ±100 が毎 step、それ以外は 1000 step ごと）。死亡時刻の
  集中度は、記録点 i が代表する step 数 `coverage_i = step[i] - step[i-1]` を使った
  **グリッド整合な帰無値**で評価する（[+1,+100] の帰無割合は 9900/10^6 = 0.99%）。
- κ は checkpoint 由来の**近似**（`analysis/q3_margin_pooled.py` の `reg_logw_interp`:
  step 0 と 1,000,000 の κ(log||w||) 回帰を t/10^6 で線形内挿）。step 1M の checkpoint は
  本走ではなく同 config の別実現なので **per-unit の主張には使えない**。本解析の主結果
  （H-a/H-b/H-c/H-d の判定）は κ を使わない厳密量だけで出し、κ は「余裕の物理スケール」を
  与える補助としてのみ使う。κ̂ 規則の per-sample 精度も併記する。

## 入力

- std:      `results/ratchet_log_0819/logs/seed{0..9}.npz`
- centered: `results/ratchet_centered_0822/logs/seed{0..9}.npz`
- checkpoint（κ̂ 用）: `results/<run>/ckpts/` -> `~/q3_out/verify/pooled/rerun_<arm>/ckpts/`

## 実行

    OMP_NUM_THREADS=1 python3 analysis/q3_boundary_race.py

引数なし・決定論（乱数を使わない）。出力は `~/q3_out/race/`（repo 外）。
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, switch_steps                      # noqa: E402
from src.figures_ratchet_log import TAU, death_events          # noqa: E402
from analysis.q3_gate_curve_ci import (                        # noqa: E402
    seed_paths, check_source_run, md_table, git_hash)
from analysis.q3_margin_pooled import (                        # noqa: E402
    load_ckpt_W, kappa_exact, fit_kappa_regression, apply_kappa_regression)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTDIR = Path.home() / "q3_out" / "race"
FIGDIR = OUTDIR / "figures"

PERIOD = 10_000
TOTAL = 1_000_000
HALF_W = 100                      # ratchet.boundary_window
BULK = 1000                       # ratchet.bulk_every
M_DIM, F_DIM = 20, 15             # condA: m=20, f=15 -> 自由 5 次元
CKPT_STEPS = (0, 1_000_000)
N_SEED = 10

FINE = np.arange(-HALF_W, HALF_W + 1)
COARSE = np.concatenate([np.arange(-9 * BULK, 0, BULK), np.arange(BULK, 10 * BULK, BULK)])
OFFSETS = np.sort(np.concatenate([FINE, COARSE]))
I_FINE = np.searchsorted(OFFSETS, FINE)
I_AT1 = int(np.searchsorted(OFFSETS, 1))         # offset +1 = flip 後の最初の記録点
REPORT_OFF = [1, 2, 5, 10, 20, 50, 100, 1000, 2000, 5000, 9000]
PBINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32)]   # p_hat = k/32 の k 区間

ARMS = [
    dict(label="std", resdir=Path(ROOT) / "results" / "ratchet_log_0819",
         config=Path(ROOT) / "configs" / "ratchet_log_0819.yaml",
         spec="specs/spec_ratchet_log_0819.md"),
    dict(label="centered", resdir=Path(ROOT) / "results" / "ratchet_centered_0822",
         config=Path(ROOT) / "configs" / "ratchet_centered_0822.yaml",
         spec="specs/spec_ratchet_centered_0822.md"),
]


# ------------------------------------------------------------------ 下ごしらえ

def load_seed(path: Path) -> dict:
    """1 seed の npz から必要な列だけ取り、s / w·µ を厳密復元する。"""
    with np.load(path) as z:
        d = {k: z[k] for k in ("step", "p_hat", "w_norm", "cos_u_mu", "b",
                               "mu_norm", "flip_state")}
    d["step"] = d["step"].astype(np.int64)
    d["wmu"] = (d["w_norm"] * d["mu_norm"][:, None] * d["cos_u_mu"]).astype(np.float32)
    d["s"] = (d["wmu"] + d["b"]).astype(np.float32)
    d["index"] = {int(v): i for i, v in enumerate(d["step"])}
    return d


def realized(d: dict) -> list[tuple[int, int, int]]:
    """実現 flip の (B, i_before, i_after)。B と B+1 の flip_state が違うものだけ。"""
    out = []
    for B in switch_steps(PERIOD, TOTAL):
        i0, i1 = d["index"].get(int(B)), d["index"].get(int(B) + 1)
        if i0 is None or i1 is None:
            continue
        if np.abs(d["flip_state"][i1] - d["flip_state"][i0]).sum() > 0:
            out.append((int(B), i0, i1))
    return out


def freeze_events_all(step: np.ndarray, p: np.ndarray) -> list[list[int]]:
    """`death_events` と**同じ判定規則**を、各 unit の「最初の 1 回」に限らず全ての
    下方クロスに適用したもの（H-d の状態ベース分母用の拡張）。

    `src/figures_ratchet_log.py: death_events` は unit ごとに最初の資格クロスで break
    するので、いったん死んで復活した unit はもう二度と死亡としてカウントされない。
    H-d の「境界 1 回あたりの死亡確率」を、その時点で生存している unit を分母に取って
    見るにはこの拡張が要る（判定規則そのものは同一: TAU 下方クロス + 1 周期回復なし）。"""
    below = p < TAU
    out = []
    for i in range(p.shape[1]):
        bb = below[:, i]
        cross = np.flatnonzero(bb[1:] & ~bb[:-1]) + 1
        ok = [int(c) for c in cross
              if bb[(step >= step[c]) & (step <= step[c] + PERIOD)].all()]
        out.append(ok)
    return out


def build_kappa(arm: dict):
    """κ̂(||w||, t): q3_margin_pooled の `reg_logw_interp` と同一構成（**近似**）。"""
    knots, info = {}, {}
    for st in CKPT_STEPS:
        W, b, rm, fs, src = load_ckpt_W(arm, st)
        kap, wn, rho = kappa_exact(W, M_DIM, F_DIM)
        knots[st] = fit_kappa_regression(kap, wn)
        info[st] = dict(source=src, kappa_median=float(np.median(kap)),
                        kappa_q1=float(np.quantile(kap, .25)),
                        kappa_q3=float(np.quantile(kap, .75)))
    k0, k1 = knots[CKPT_STEPS[0]], knots[CKPT_STEPS[1]]

    def kappa_hat(w_norm, step):
        lam = np.clip(np.asarray(step, dtype=np.float64) / float(CKPT_STEPS[1]), 0, 1)
        return ((1.0 - lam) * apply_kappa_regression(k0, w_norm)
                + lam * apply_kappa_regression(k1, w_norm))
    return kappa_hat, info


# ------------------------------------------------------------------ イベント収集

def collect(arm: dict, kappa_hat) -> dict:
    """境界イベントを整列して集める。イベント = (seed, 境界, unit)。

    軌跡は「on at B（p_hat>0）」の unit だけ集める。完全消灯 unit は勾配がゼロで
    追随の議論の対象外だが、µ が動く分だけ s は動くので別途 frac_on で追う。"""
    paths = seed_paths(arm["resdir"])
    if len(paths) != N_SEED:
        raise SystemExit(f"{arm['label']}: seed 数が {len(paths)}")
    n_off = len(OFFSETS)
    tr = {k: [] for k in ("ds", "db", "dwmu", "p", "on_all")}
    ev = {k: [] for k in ("seed", "bidx", "unit", "s0", "w0", "p0", "mu0",
                          "kap0", "step0", "jump", "jump_wmu", "jump_b",
                          "p1", "died_win", "on_run")}
    mu_traj = np.zeros((n_off, 0), dtype=np.float32)
    mu_rows = []
    ctl = {k: [] for k in ("ds", "db", "dwmu")}     # 非境界 1-step 対照（on の unit）
    bnd = {k: [] for k in ("ds", "db", "dwmu")}     # 境界 1-step（同じ on 集合）
    deaths, haz_rows, cov_rows, seed_conc = [], [], [], []
    san = dict(bits=set(), mu2_dev=0.0, dmu2=set(), mu0=[], mu1=[])

    for sd, p in enumerate(paths):
        d = load_seed(p)
        step, s, b, wmu, ph, wn, mun = (d["step"], d["s"], d["b"], d["wmu"],
                                        d["p_hat"], d["w_norm"], d["mu_norm"])
        rz = realized(d)
        Ba = np.array([B for B, _, _ in rz], dtype=np.int64)
        idx = d["index"]

        # --- 構造サニティ（H-a の機構そのもの）: std では mu = [flip ‖ 0.5·1_5] なので
        # ‖mu‖^2 = Σflip + 5·0.25 が厳密に成り立つ。境界では 1 ビットだけ反転するので
        # Δ‖mu‖^2 = ±1。centered は running_mean が入るので前者は成り立たない。
        san["mu2_dev"] = max(san["mu2_dev"], float(
            np.abs(mun.astype(np.float64) ** 2
                   - d["flip_state"].sum(axis=1) - 1.25).max()))
        for B, i0, i1 in rz:
            san["bits"].add(int(np.abs(d["flip_state"][i1] - d["flip_state"][i0]).sum()))
            san["dmu2"].add(int(np.rint(float(mun[i1]) ** 2 - float(mun[i0]) ** 2)))
            san["mu0"].append(float(mun[i0])); san["mu1"].append(float(mun[i1]))

        # --- 死亡イベント（定義は src.figures_ratchet_log と同一実装）
        de = death_events({"step": step, "p_hat": ph}, PERIOD)
        death_idx = {u: c for u, c in de}
        fz = freeze_events_all(step, ph)          # 同一規則・全クロス（H-d 用）
        fz_steps = [np.array([int(step[c]) for c in v], dtype=np.int64) for v in fz]
        for u, c in de:
            t = int(step[c])
            k = int(np.argmin(np.abs(t - Ba)))
            deaths.append(dict(seed=sd, unit=u, i_death=c, step=t,
                               offset=int(t - Ba[k]), b_index=k + 1,
                               final_dead=bool(ph[-1, u] < TAU)))

        # --- 記録点ごとの coverage と最寄り境界 offset（グリッド整合な帰無値用）
        cov = np.zeros(len(step), dtype=np.int64)
        cov[1:] = np.diff(step)
        near = step[:, None] - Ba[None, :]
        off_rec = near[np.arange(len(step)), np.abs(near).argmin(axis=1)]
        cov_rows.append(pd.DataFrame(dict(offset=off_rec, coverage=cov)))

        # --- イベント整列
        for k, (B, i0, i1) in enumerate(rz):
            if B - 9 * BULK < 0 or B + 9 * BULK > TOTAL:
                rows = None                     # 全 offset が揃わない境界は軌跡から除く
            else:
                rows = np.array([idx[B + int(o)] for o in OFFSETS])
            on0 = ph[i0] > 0
            alive0 = ph[i0] >= TAU
            n_on = int(on0.sum())

            # 境界 1-step と非境界 1-step（対照）— on の unit に限る
            bnd["ds"].append((s[i1] - s[i0])[on0])
            bnd["db"].append((b[i1] - b[i0])[on0])
            bnd["dwmu"].append((wmu[i1] - wmu[i0])[on0])
            for o in range(-HALF_W, HALF_W):
                if o == 0:
                    continue
                j0, j1 = idx.get(B + o), idx.get(B + o + 1)
                if j0 is None or j1 is None:
                    continue
                m = ph[j0] > 0
                ctl["ds"].append((s[j1] - s[j0])[m])
                ctl["db"].append((b[j1] - b[j0])[m])
                ctl["dwmu"].append((wmu[j1] - wmu[j0])[m])

            if rows is not None and n_on:
                tr["ds"].append(s[rows][:, on0] - s[i0][on0])
                tr["db"].append(b[rows][:, on0] - b[i0][on0])
                tr["dwmu"].append(wmu[rows][:, on0] - wmu[i0][on0])
                tr["p"].append(ph[rows][:, on0])
                fine_rows = rows[I_FINE]
                on_run = (ph[fine_rows][FINE >= 0][:, on0] > 0).all(axis=0)
                tr["on_all"].append(on_run)
                mu_rows.append(mun[rows])

                dw = death_idx
                died = np.array([(u in dw) and (B + 1 <= step[dw[u]] <= B + HALF_W)
                                 for u in np.flatnonzero(on0)])
                ev["seed"].append(np.full(n_on, sd, np.int32))
                ev["bidx"].append(np.full(n_on, k + 1, np.int32))
                ev["unit"].append(np.flatnonzero(on0).astype(np.int32))
                ev["s0"].append(s[i0][on0]); ev["w0"].append(wn[i0][on0])
                ev["p0"].append(ph[i0][on0]); ev["mu0"].append(np.full(n_on, mun[i0]))
                ev["step0"].append(np.full(n_on, B, np.int64))
                ev["kap0"].append(kappa_hat(wn[i0][on0], np.full(n_on, float(B))))
                ev["jump"].append((s[i1] - s[i0])[on0])
                ev["jump_wmu"].append((wmu[i1] - wmu[i0])[on0])
                ev["jump_b"].append((b[i1] - b[i0])[on0])
                ev["p1"].append(ph[i1][on0])
                ev["died_win"].append(died)
                ev["on_run"].append(on_run)

            # --- H-d ハザード（分母 = B で alive かつ B 以前に死亡記録が無い unit）
            not_dead = np.array([not (u in death_idx and step[death_idx[u]] <= B)
                                 for u in range(ph.shape[1])])
            at_risk = alive0 & not_dead
            nd = sum(1 for u in np.flatnonzero(at_risk)
                     if u in death_idx and B + 1 <= step[death_idx[u]] <= B + HALF_W)
            # 状態ベース: B で alive な unit のうち、窓内に「同一規則の消灯イベント」が
            # 起きたもの（最初の 1 回に限らない）。分母が枯れないので後半も読める。
            nfz = sum(1 for u in np.flatnonzero(alive0)
                      if fz_steps[u].size
                      and ((fz_steps[u] >= B + 1) & (fz_steps[u] <= B + HALF_W)).any())
            haz_rows.append(dict(seed=sd, b_index=k + 1, step=B,
                                 n_at_risk=int(at_risk.sum()), n_death_win=int(nd),
                                 n_alive=int(alive0.sum()), n_freeze_win=int(nfz),
                                 n_on=n_on, frac_on=float(on0.mean())))

        # --- seed 単位の集中度（seed 間ばらつき用）
        offs = np.array([x["offset"] for x in deaths if x["seed"] == sd])
        if offs.size:
            seed_conc.append(dict(seed=sd, n_death=int(offs.size),
                                  frac_win=float(((offs >= 1) & (offs <= HALF_W)).mean()),
                                  frac_at1=float((offs == 1).mean())))
        del d

    out = dict(
        traj={k: np.concatenate(v, axis=1 if v[0].ndim == 2 else 0) for k, v in tr.items()},
        mu=np.stack(mu_rows, axis=1) if mu_rows else mu_traj,
        ev=pd.DataFrame({k: np.concatenate(v) for k, v in ev.items()}),
        bnd={k: np.concatenate(v) for k, v in bnd.items()},
        ctl={k: np.concatenate(v) for k, v in ctl.items()},
        deaths=pd.DataFrame(deaths), hazard=pd.DataFrame(haz_rows),
        cov=pd.concat(cov_rows, ignore_index=True),
        seed_conc=pd.DataFrame(seed_conc),
        sanity=dict(bits_changed_per_boundary=sorted(san["bits"]),
                    max_dev_mu2_minus_flipsum_1p25=san["mu2_dev"],
                    delta_mu2_at_boundary=sorted(san["dmu2"]),
                    mu_norm_med_at_B=float(np.median(san["mu0"])),
                    mu_norm_med_at_B1=float(np.median(san["mu1"])),
                    n_boundary_events=len(san["mu0"])))
    return out


# ------------------------------------------------------------------ H-a / H-b

def jump_table(R: dict, label: str) -> pd.DataFrame:
    rows = []
    for kind, D in (("boundary_0_to_1", R["bnd"]), ("control_1step", R["ctl"])):
        for q, name in (("ds", "Δs"), ("dwmu", "Δ(w·µ)"), ("db", "Δb")):
            a = np.abs(D[q].astype(np.float64))
            rows.append(dict(arm=label, kind=kind, quantity=name, n=a.size,
                             median_abs=float(np.median(a)), mean_abs=float(a.mean()),
                             q90_abs=float(np.quantile(a, .90)),
                             frac_nonzero=float((a > 0).mean())))
    df = pd.DataFrame(rows)
    return df


def traj_table(R: dict, label: str) -> pd.DataFrame:
    T = R["traj"]
    jump = T["ds"][I_AT1]
    grp = dict(down=jump < 0, up=jump > 0, all=np.ones(jump.shape, bool))
    mu_med = np.median(R["mu"], axis=1)
    rows = []
    for gname, m in grp.items():
        if not m.any():
            continue
        for i, o in enumerate(OFFSETS):
            rows.append(dict(arm=label, group=gname, offset=int(o), n=int(m.sum()),
                             med_ds=float(np.median(T["ds"][i][m])),
                             mean_ds=float(T["ds"][i][m].mean()),
                             med_db=float(np.median(T["db"][i][m])),
                             med_dwmu=float(np.median(T["dwmu"][i][m])),
                             med_p=float(np.median(T["p"][i][m])),
                             frac_p_zero=float((T["p"][i][m] == 0).mean()),
                             med_mu_norm=float(mu_med[i])))
    return pd.DataFrame(rows)


def gap_table(R: dict, label: str) -> tuple[pd.DataFrame, dict]:
    """κ フリーの回復指標: gap(o) = median_up(Δs) − median_down(Δs)。共通ドリフトが消える。

    b だけの gap も出す。b の gap は「連続点灯 (on_run)」部分集合でも出す
    （消灯した unit は勾配ゼロで b が動かないため、選択の効果を分けて見る）。"""
    T = R["traj"]
    jump = T["ds"][I_AT1]
    dn, up = jump < 0, jump > 0
    run = T["on_all"]
    rows = []
    for i, o in enumerate(OFFSETS):
        g = float(np.median(T["ds"][i][up]) - np.median(T["ds"][i][dn]))
        gb = float(np.median(T["db"][i][up]) - np.median(T["db"][i][dn]))
        gw = float(np.median(T["dwmu"][i][up]) - np.median(T["dwmu"][i][dn]))
        mdn, mup = dn & run, up & run
        gbr = (float(np.median(T["db"][i][mup]) - np.median(T["db"][i][mdn]))
               if mdn.any() and mup.any() else np.nan)
        gr = (float(np.median(T["ds"][i][mup]) - np.median(T["ds"][i][mdn]))
              if mdn.any() and mup.any() else np.nan)
        rows.append(dict(arm=label, offset=int(o), gap_s=g, gap_b=gb, gap_wmu=gw,
                         gap_s_onrun=gr, gap_b_onrun=gbr))
    df = pd.DataFrame(rows)
    g1 = float(df.loc[df.offset == 1, "gap_s"].iloc[0])
    df["gap_ratio"] = df["gap_s"] / g1
    # 時定数: 微細窓 (1..100) と 1 周期 (1000..9000) の対数線形フィット
    fits = {}
    for tag, sel in (("fine_1_100", (df.offset >= 1) & (df.offset <= 100)),
                     ("bulk_1k_9k", (df.offset >= 1000) & (df.offset <= 9000))):
        sub = df[sel & (df.gap_ratio > 0)]
        if len(sub) >= 3:
            x = sub.offset.to_numpy(float) - 1.0
            y = np.log(sub.gap_ratio.to_numpy(float))
            sl = np.polyfit(x, y, 1)[0]
            fits[tag] = float(-1.0 / sl) if sl < 0 else np.inf
        else:
            fits[tag] = np.nan
    half = df[(df.offset >= 1) & (df.gap_ratio <= 0.5)]
    fits["half_offset"] = int(half.offset.iloc[0]) if len(half) else None
    fits["gap_at_1"] = g1
    for o in (100, 1000, 9000):
        fits[f"gap_ratio_at_{o}"] = float(df.loc[df.offset == o, "gap_ratio"].iloc[0])
    return df, fits


# ------------------------------------------------------------------ H-c

WINDOWS = ((1, 1, "offset = +1"), (1, 10, "+1..+10"), (1, HALF_W, "+1..+100"),
           (-99, -1, "-99..-1"), (-HALF_W, HALF_W, "-100..+100"),
           (1, BULK, "+1..+1000"), (0, 0, "offset = 0"))


def _conc_rows(de: pd.DataFrame, H: pd.DataFrame, tot_cov: float,
               arm: str, subset: str) -> list[dict]:
    """窓ごとの観測割合・グリッド整合な帰無割合・率比。

    `-99..-1` は物理的にはただのバルク時間だが記録が 1 step 刻みなので、そこの率比は
    **密グリッドによる検出バイアスそのもの**の推定値になる（短い一過性の落ち込みは
    粗いグリッドでは見逃されうる）。それで割った `rate_ratio_corr` も併記する。"""
    n = len(de)
    rows = []
    for lo, hi, name in WINDOWS:
        m = (de.offset >= lo) & (de.offset <= hi)
        mc = (H.offset >= lo) & (H.offset <= hi)
        c_in = float(H.loc[mc, "coverage"].sum())
        k = int(m.sum())
        rate_in = k / c_in if c_in else np.nan
        rate_out = (n - k) / (tot_cov - c_in) if tot_cov > c_in else np.nan
        rows.append(dict(arm=arm, subset=subset, window=name, n_death=k,
                         frac=k / n if n else np.nan,
                         coverage_steps=c_in, null_frac=c_in / tot_cov,
                         rate_ratio=rate_in / rate_out if rate_out else np.nan))
    df = pd.DataFrame(rows)
    bias = float(df.loc[df.window == "-99..-1", "rate_ratio"].iloc[0])
    df["detection_bias_est"] = bias
    df["rate_ratio_corr"] = df["rate_ratio"] / bias if bias > 0 else np.nan
    return df.to_dict("records")


def death_concentration(R: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    de, cov = R["deaths"], R["cov"]
    n = len(de)
    cov_by = cov.groupby("offset")["coverage"].sum()
    tot_cov = float(cov_by.sum())
    hist = de.groupby("offset").size()
    H = pd.DataFrame(dict(offset=cov_by.index, coverage=cov_by.to_numpy()))
    H["count"] = H["offset"].map(hist).fillna(0).astype(int)
    H["expected_uniform"] = n * H["coverage"] / tot_cov
    H.insert(0, "arm", label)

    rows = _conc_rows(de, H, tot_cov, label, "all")
    rows += _conc_rows(de[de.final_dead], H, tot_cov, label, "still_dead_at_1M")
    C = pd.DataFrame(rows)
    info = dict(n_death=int(n), n_final_dead=int(de.final_dead.sum()),
                frac_final_dead=float(de.final_dead.mean()),
                total_coverage=tot_cov)
    return H, C, info


def silencing_table(R: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ジャンプ直前の余裕・ジャンプ幅と「境界直後の消灯 / 窓内死亡」の関係。

    余裕の**厳密**な序数版は p_hat(B) 自身（p_hat = 0 <=> s+M<=0 なので p_hat は
    s+M の単調増加関数）。物理スケール版 margin_hat = s + κ̂||w|| は近似。"""
    e = R["ev"].copy()
    e["silenced1"] = (e.p1 == 0).astype(int)
    e["k0"] = np.rint(e.p0 * 32).astype(int)
    e["margin_hat"] = e.s0 + e.kap0 * e.w0
    e["margin_hat_norm"] = e.margin_hat / e.w0
    e["jump_norm"] = e.jump / e.w0
    rows = []
    for lo, hi in PBINS:
        m = (e.k0 >= lo) & (e.k0 <= hi)
        if not m.any():
            continue
        rows.append(dict(arm=label, bin=f"k={lo}" if lo == hi else f"k={lo}-{hi}",
                         p_hat_lo=lo / 32, p_hat_hi=hi / 32, n=int(m.sum()),
                         med_margin_hat_norm=float(e.margin_hat_norm[m].median()),
                         P_silenced_at_1=float(e.silenced1[m].mean()),
                         P_death_in_win=float(e.died_win[m].mean()),
                         P_silenced_at_1_downjump=float(
                             e.silenced1[m & (e.jump < 0)].mean())
                         if (m & (e.jump < 0)).any() else np.nan))
    Bp = pd.DataFrame(rows)

    dn = e[e.jump < 0].copy()
    q = np.quantile(-dn.jump_norm, [0, .2, .4, .6, .8, 1.0])
    dn["qi"] = np.clip(np.searchsorted(q[1:-1], -dn.jump_norm, "right"), 0, 4)
    rows = []
    for qi in range(5):
        m = dn.qi == qi
        if not m.any():
            continue
        rows.append(dict(arm=label, jump_quintile=qi + 1, n=int(m.sum()),
                         med_neg_jump_over_w=float((-dn.jump_norm[m]).median()),
                         med_margin_hat_norm=float(dn.margin_hat_norm[m].median()),
                         P_silenced_at_1=float(dn.silenced1[m].mean()),
                         P_death_in_win=float(dn.died_win[m].mean())))
    Bj = pd.DataFrame(rows)

    pred = (e.margin_hat + e.jump) <= 0
    obs = e.silenced1 == 1
    info = dict(
        n_event=int(len(e)), frac_down=float((e.jump < 0).mean()),
        P_silenced_at_1=float(e.silenced1.mean()),
        P_silenced_at_1_down=float(e.silenced1[e.jump < 0].mean()),
        P_silenced_at_1_up=float(e.silenced1[e.jump > 0].mean()),
        med_margin_silenced=float(e.margin_hat_norm[obs].median()),
        med_margin_survived=float(e.margin_hat_norm[~obs].median()),
        med_p0_silenced=float(e.p0[obs].median()),
        med_p0_survived=float(e.p0[~obs].median()),
        kappa_rule_accuracy=float((pred == obs).mean()),
        kappa_rule_TP=int((pred & obs).sum()), kappa_rule_FP=int((pred & ~obs).sum()),
        kappa_rule_FN=int((~pred & obs).sum()), kappa_rule_TN=int((~pred & ~obs).sum()),
        med_abs_jump_over_w=float(np.median(np.abs(e.jump_norm))))
    return Bp, Bj, info


# ------------------------------------------------------------------ H-d

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項の Wilson 区間（n=0 は NaN）。"""
    if n <= 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def hazard_table(R: dict, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """境界 1 回あたりのハザードを **2 通りの分母**で出す。

    - `hazard_first`: 分母 = B で alive かつ**まだ一度も死亡記録が無い** unit。
      `death_events` が unit ごとに最初の 1 回しか数えないので、この分母が
      「最初の死亡」の正しい risk set。std では後半に分母が枯れる。
    - `hazard_state`: 分母 = B で alive な unit すべて、分子 = 同一規則の消灯イベント
      （復活後の再消灯も数える `freeze_events_all`）。分母が枯れないので後半も読める。"""
    h = R["hazard"].groupby("b_index", as_index=False).agg(
        n_at_risk=("n_at_risk", "sum"), n_death_win=("n_death_win", "sum"),
        n_alive=("n_alive", "sum"), n_freeze_win=("n_freeze_win", "sum"),
        n_on=("n_on", "sum"), step=("step", "first"))
    h["hazard_first"] = h.n_death_win / h.n_at_risk.replace(0, np.nan)
    h["hazard_state"] = h.n_freeze_win / h.n_alive.replace(0, np.nan)
    h["alive_frac"] = h.n_alive / (N_SEED * 100)
    h.insert(0, "arm", label)
    blocks = []
    for lo in range(1, 100, 11):
        hi = min(lo + 10, 99)
        m = (h.b_index >= lo) & (h.b_index <= hi)
        na, kd = int(h.n_at_risk[m].sum()), int(h.n_death_win[m].sum())
        nl, kf = int(h.n_alive[m].sum()), int(h.n_freeze_win[m].sum())
        lo1, hi1 = wilson(kd, na)
        lo2, hi2 = wilson(kf, nl)
        blocks.append(dict(arm=label, block=f"{lo}-{hi}",
                           alive_frac=float(h.alive_frac[m].mean()),
                           n_at_risk=na, n_death_win=kd,
                           hazard_first=kd / na if na else np.nan,
                           hazard_first_lo=lo1, hazard_first_hi=hi1,
                           n_alive=nl, n_freeze_win=kf,
                           hazard_state=kf / nl if nl else np.nan,
                           hazard_state_lo=lo2, hazard_state_hi=hi2))
    return h, pd.DataFrame(blocks)


# ------------------------------------------------------------------ 図

def fig_aligned(res: dict):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for r, arm in enumerate(ARMS):
        lab = arm["label"]
        T, mu = res[lab]["traj"], res[lab]["mu"]
        jump = T["ds"][I_AT1]
        dn, up = jump < 0, jump > 0
        ax = axes[r, 0]
        ax.plot(OFFSETS[I_FINE], np.median(T["ds"][I_FINE][:, dn], axis=1),
                color="tab:red", label=f"下向きジャンプ (n={dn.sum()})")
        ax.plot(OFFSETS[I_FINE], np.median(T["ds"][I_FINE][:, up], axis=1),
                color="tab:blue", label=f"上向きジャンプ (n={up.sum()})")
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=.6)
        ax.set_title(f"{lab}: 境界整列した Δs = s(B+o) − s(B)")
        ax.set_xlabel("境界からの offset [step]"); ax.set_ylabel("median Δs")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = axes[r, 1]
        ax.plot(OFFSETS[I_FINE], np.median(T["dwmu"][I_FINE][:, dn], axis=1),
                color="tab:red", label="Δ(w·µ) 下向き")
        ax.plot(OFFSETS[I_FINE], np.median(T["db"][I_FINE][:, dn], axis=1),
                color="tab:red", ls="--", label="Δb 下向き")
        ax.plot(OFFSETS[I_FINE], np.median(T["dwmu"][I_FINE][:, up], axis=1),
                color="tab:blue", label="Δ(w·µ) 上向き")
        ax.plot(OFFSETS[I_FINE], np.median(T["db"][I_FINE][:, up], axis=1),
                color="tab:blue", ls="--", label="Δb 上向き")
        ax.axvline(0.5, color="k", ls=":", lw=1); ax.axhline(0, color="gray", lw=.6)
        ax.set_title(f"{lab}: 内訳 (実線 w·µ / 破線 b)")
        ax.set_xlabel("offset [step]"); ax.legend(fontsize=7); ax.grid(alpha=.3)

        ax = axes[r, 2]
        gp = res[lab]["gap"]
        pos = gp[gp.offset >= 1]
        ax.plot(pos.offset, pos.gap_ratio, "o-", ms=3, color="tab:purple",
                label="gap_s(o)/gap_s(1)  (κ フリー)")
        ax2 = ax.twinx()
        ax2.plot(OFFSETS[OFFSETS >= 1], np.median(mu, axis=1)[OFFSETS >= 1],
                 color="tab:green", ls="--", lw=1.2, label="median ‖µ‖")
        ax2.set_ylabel("‖µ‖", color="tab:green")
        ax2.set_ylim(0, max(1e-6, float(np.median(mu, axis=1).max())) * 1.15)
        ax.set_xscale("log"); ax.set_xlabel("offset [step]  (log)")
        ax.set_ylabel("gap 比"); ax.set_ylim(-0.05, 1.15)
        ax.axhline(0.5, color="gray", lw=.6, ls=":")
        ax.set_title(f"{lab}: 上下ジャンプ群の gap 減衰と ‖µ‖")
        hs, ls_ = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(hs + h2, ls_ + l2, fontsize=7, loc="lower left",
                  framealpha=.9)
        ax.grid(alpha=.3)
    fig.suptitle("境界イベント整列（on at B の unit のみ・両 arm）事後計算", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_race_aligned.png", dpi=130)
    plt.close(fig)


def fig_death_hist(res: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    for r, arm in enumerate(ARMS):
        lab = arm["label"]
        H = res[lab]["dhist"]
        fine = H[(H.offset >= -HALF_W) & (H.offset <= HALF_W)]
        ax = axes[r, 0]
        ax.bar(fine.offset, fine["count"], width=1.0, color="tab:red")
        ax.plot(fine.offset, fine.expected_uniform, color="k", lw=1,
                label="一様帰無 (coverage 重み)")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"{lab}: 死亡が検出された記録点の境界 offset (±100)")
        ax.set_xlabel("offset [step]"); ax.set_ylabel("死亡数")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = axes[r, 1]
        de = res[lab]["deaths"]
        bins = [-10001, -1000, -100, 0, 1, 2, 11, 101, 1001, 10001]
        labels = ["≤−1000", "−999..−101", "−100..−1", "0", "+1", "+2..+10",
                  "+11..+100", "+101..+1000", "≥+1001"]
        cnt = [int(((de.offset >= bins[i]) & (de.offset < bins[i + 1])).sum())
               for i in range(len(bins) - 1)]
        cov = res[lab]["cov"]
        ncov = [float(cov.coverage[(cov.offset >= bins[i])
                                   & (cov.offset < bins[i + 1])].sum())
                for i in range(len(bins) - 1)]
        tot = sum(ncov); n = len(de)
        ax.bar(range(len(cnt)), np.array(cnt) / n, color="tab:red", label="観測割合")
        ax.plot(range(len(cnt)), np.array(ncov) / tot, "ko-", ms=4,
                label="一様帰無割合")
        ax.set_xticks(range(len(cnt))); ax.set_xticklabels(labels, rotation=35,
                                                           ha="right", fontsize=7)
        ax.set_ylabel("割合"); ax.set_title(f"{lab}: 粗ビン（帰無と比較）")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("死亡時刻の境界集中（death_events は src/figures_ratchet_log と同一実装）"
                 "／事後計算", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_death_offset_hist.png", dpi=130)
    plt.close(fig)


def fig_hazard(res: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for arm in ARMS:
        lab = arm["label"]
        h, hb = res[lab]["haz"], res[lab]["haz_blocks"]
        axes[0].plot(h.b_index, h.alive_frac, "o-", ms=3, label=lab)
        x = np.arange(len(hb)) + 1
        axes[1].errorbar(x, hb.hazard_state,
                         yerr=[hb.hazard_state - hb.hazard_state_lo,
                               hb.hazard_state_hi - hb.hazard_state], fmt="o-",
                         ms=4, capsize=3, label=lab)
        axes[2].plot(h.b_index, h.n_at_risk / (N_SEED * 100), "o-", ms=3,
                     label=f"{lab} (最初の死亡 risk set)")
    axes[0].set_xlabel("境界 index (1..99)"); axes[0].set_ylabel("生存割合 (p̂ ≥ τ)")
    axes[0].set_title("境界直前の生存割合")
    axes[1].set_xticks(np.arange(len(res["std"]["haz_blocks"])) + 1)
    axes[1].set_xticklabels(res["std"]["haz_blocks"]["block"], rotation=40,
                            ha="right", fontsize=7)
    axes[1].set_xlabel("境界 index ブロック"); axes[1].set_yscale("log")
    axes[1].set_ylabel("窓内消灯 / 生存 (log)")
    axes[1].set_title("境界 1 回あたりのハザード（状態ベース・95% CI）")
    axes[2].set_xlabel("境界 index (1..99)"); axes[2].set_ylabel("at-risk 割合")
    axes[2].set_title("「最初の死亡」risk set は後半に枯れる")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("H-d ラチェット（事後計算）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_hazard_boundary_index.png", dpi=130)
    plt.close(fig)


def fig_margin(res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for arm in ARMS:
        lab = arm["label"]
        Bp, Bj = res[lab]["silence_p"], res[lab]["silence_j"]
        axes[0].plot(range(len(Bp)), Bp.P_silenced_at_1, "o-", ms=4, label=lab)
        axes[1].plot(Bj.jump_quintile, Bj.P_silenced_at_1, "o-", ms=4, label=lab)
    axes[0].set_xticks(range(len(res["std"]["silence_p"])))
    axes[0].set_xticklabels(res["std"]["silence_p"]["bin"], rotation=25, fontsize=8)
    axes[0].set_xlabel("境界直前の p̂ = k/32（余裕の厳密な序数）")
    axes[0].set_ylabel("P(境界 +1 で消灯)")
    axes[0].set_title("余裕が小さい unit ほど消灯するか")
    axes[1].set_xlabel("下向きジャンプ幅 |Δs|/‖w‖ の五分位 (1=浅い)")
    axes[1].set_ylabel("P(境界 +1 で消灯)")
    axes[1].set_title("ジャンプが深いほど消灯するか（下向きジャンプのみ）")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("H-c: 余裕とジャンプ幅（事後計算）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_margin_vs_silencing.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    res, metas = {}, {}
    for arm in ARMS:
        lab = arm["label"]
        metas[lab] = check_source_run(arm["resdir"], arm["spec"])
        kappa_hat, kinfo = build_kappa(arm)
        print(f"[{lab}] collecting ...", flush=True)
        R = collect(arm, kappa_hat)
        R["jump"] = jump_table(R, lab)
        R["traj_tab"] = traj_table(R, lab)
        R["gap"], R["gap_fit"] = gap_table(R, lab)
        R["dhist"], R["dconc"], R["dinfo"] = death_concentration(R, lab)
        R["silence_p"], R["silence_j"], R["sinfo"] = silencing_table(R, lab)
        R["haz"], R["haz_blocks"] = hazard_table(R, lab)
        R["kinfo"] = kinfo
        res[lab] = R
        print(f"[{lab}] events={len(R['ev'])} deaths={len(R['deaths'])} "
              f"({time.time()-t0:.0f}s)", flush=True)

    cat = lambda k: pd.concat([res[a["label"]][k] for a in ARMS], ignore_index=True)
    cat("jump").to_csv(OUTDIR / "jump_stats.csv", index=False)
    cat("traj_tab").to_csv(OUTDIR / "aligned_trajectory.csv", index=False)
    cat("gap").to_csv(OUTDIR / "recovery_gap.csv", index=False)
    cat("dhist").to_csv(OUTDIR / "death_offset_hist.csv", index=False)
    cat("dconc").to_csv(OUTDIR / "death_concentration.csv", index=False)
    cat("silence_p").to_csv(OUTDIR / "silencing_by_margin.csv", index=False)
    cat("silence_j").to_csv(OUTDIR / "silencing_by_jump.csv", index=False)
    cat("haz").to_csv(OUTDIR / "hazard_by_boundary.csv", index=False)
    cat("haz_blocks").to_csv(OUTDIR / "hazard_blocks.csv", index=False)
    pd.concat([res[a["label"]]["deaths"].assign(arm=a["label"]) for a in ARMS],
              ignore_index=True).to_csv(OUTDIR / "death_events.csv", index=False)
    pd.concat([res[a["label"]]["seed_conc"].assign(arm=a["label"]) for a in ARMS],
              ignore_index=True).to_csv(OUTDIR / "death_concentration_by_seed.csv",
                                        index=False)

    fig_aligned(res); fig_death_hist(res); fig_hazard(res); fig_margin(res)
    write_results(res, metas)

    meta = dict(script="analysis/q3_boundary_race.py", git=git_hash(),
                generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, preregistered=False, new_training_runs=0,
                tau=TAU, period=PERIOD, half_window=HALF_W, bulk_every=BULK,
                offsets=[int(o) for o in OFFSETS],
                arms={a["label"]: dict(resdir=str(a["resdir"]),
                                       spec=a["spec"],
                                       kappa=res[a["label"]]["kinfo"],
                                       n_event=int(len(res[a["label"]]["ev"])),
                                       n_death=int(len(res[a["label"]]["deaths"])),
                                       sanity=res[a["label"]]["sanity"])
                      for a in ARMS},
                elapsed_sec=round(time.time() - t0, 1))
    (OUTDIR / "analysis_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUTDIR}/results.md ({time.time()-t0:.0f}s)", flush=True)


def _fmt(x, nd=4):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else (f"{x:.{nd}g}" if isinstance(x, (float, np.floating)) else str(x))


def _jump_med(R: dict, kind: str, quantity: str) -> float:
    J = R["jump"]
    return float(J[(J["kind"] == kind) & (J["quantity"] == quantity)].median_abs.iloc[0])


def _jump_mean(R: dict, kind: str, quantity: str) -> float:
    J = R["jump"]
    return float(J[(J["kind"] == kind) & (J["quantity"] == quantity)].mean_abs.iloc[0])


def _conc(R: dict, window: str, subset: str = "all"):
    C = R["dconc"]
    return C[(C["window"] == window) & (C["subset"] == subset)].iloc[0]


def _gap_at(R: dict, offset: int, col: str) -> float:
    g = R["gap"]
    return float(g[g["offset"] == offset][col].iloc[0])


def write_results(res: dict, metas: dict):
    L = ["# q3_boundary_race: 境界ジャンプ vs バイアス追随のレース", "",
         "**事後計算・未事前登録**（spec なし・事前登録された判定基準なし。"
         "本文の「支持/不支持」は事前に固定した閾値ではなく記述的判断であり、"
         "引用には事前登録つきの昇格が要る）。**新しい学習走はしていない**"
         "（既存ログの再解析のみ）。",
         f"生成 {time.strftime('%Y-%m-%d %H:%M:%S')} / `analysis/q3_boundary_race.py` "
         f"@ {git_hash()}。", "",
         "スコープ: condA・w100・T=10^4・batch=1・center_alpha=0.01・"
         "seed 0..9・実現 flip 99 回。", ""]

    L += ["## 0. 一行", ""]
    st, ce = res["std"], res["centered"]
    jw = _jump_med(st, "boundary_0_to_1", "Δ(w·µ)")
    jb = _jump_med(st, "boundary_0_to_1", "Δb")
    ws, wc = _conc(st, "+1..+100"), _conc(ce, "+1..+100")
    tau_std = st["gap_fit"]["bulk_1k_9k"]
    L += [f"- 境界の 1 step で s は w·µ 側だけが跳ぶ（std: median |Δ(w·µ)| = "
          f"{jw:.3f}、同 |Δb| = {jb:.5f}）→ **H-a 支持**。",
          f"- 死亡は境界直後に極端に集中する（std: offset +1..+100 に "
          f"{ws.frac*100:.1f}%、一様帰無 {ws.null_frac*100:.2f}%、"
          f"率比 {ws.rate_ratio:.0f}倍）→ **H-c 支持**。",
          f"- ただし std では追随がほぼ効かない: 上下ジャンプ群の gap は 1 周期後でも "
          f"{st['gap_fit']['gap_ratio_at_9000']*100:.0f}% 残り、"
          f"半減 offset は 1 周期内に存在しない（バルク区間の見かけの時定数 "
          f"{_fmt(tau_std, 3)} step ≈ {tau_std / PERIOD:.0f} 周期）"
          "→ **H-b は「時定数がある」どころか「周期内では実質回復しない」**。",
          f"- centered は gap が {ce['gap_fit']['half_offset']} step で半減し 1 周期後に "
          f"{ce['gap_fit']['gap_ratio_at_9000']*100:.0f}% まで戻るが、これは b の追随ではなく "
          "**running_mean（α=0.01, 1/α=100 step）が µ パルスを吸収する**ため。",
          f"- centered の死は境界直後に集中はするが std より**弱い**"
          f"（+1..+100 に {wc.frac*100:.1f}% vs std {ws.frac*100:.1f}%、"
          f"検出バイアス補正後の率比 {wc.rate_ratio_corr:.0f}倍 vs "
          f"{ws.rate_ratio_corr:.0f}倍）→ 「centered ではさらに強く集中するはず」"
          "という予想は**不支持**。",
          f"- 境界 1 回あたりの消灯ハザードは時間で下がらない"
          f"（std 0.10–0.19 / centered 0.004–0.012）。生存割合は std で "
          f"{res['std']['haz'].alive_frac.iloc[0]:.2f} → "
          f"{res['std']['haz'].alive_frac.iloc[-1]:.2f}、centered で "
          f"{res['centered']['haz'].alive_frac.iloc[0]:.2f} → "
          f"{res['centered']['haz'].alive_frac.iloc[-1]:.2f} → **H-d 支持**。", ""]

    L += ["## 1. 定義と前提", "",
          "- `s := w·µ + b = ‖w‖·‖µ‖·cos(w,µ) + b`。npz の "
          "`w_norm` / `mu_norm` / `cos_u_mu` / `b` から厳密に復元（恒等式は "
          "`src/ratchet_log.py: exact_record` に由来）。",
          "- `p_hat = 0 <=> s + M <= 0`（M = max_δ w·δ）。`p_hat` は 32 パターンの "
          "厳密値 k/32。",
          f"- 死亡判定は `src/figures_ratchet_log.py` の `death_events` を **import して"
          f"そのまま使用**（TAU = {TAU}、`death_recover_periods = 1` に対応する "
          "「下方クロス後 1 周期回復しない最初の記録点」）。p_hat = k/32 なので "
          "p_hat < 0.05 <=> k <= 1。",
          "- 「on」= `p_hat > 0`（勾配が入りうる）、「alive / at-risk」= `p_hat >= TAU`。",
          "- 境界 B の flip は記録点 B と B+1 の間で起きる。offset 0 は flip 前。",
          "- 記録グリッドが非一様なので、死亡時刻の帰無値は記録点 i の "
          "`coverage_i = step[i] − step[i−1]` で重み付ける（[+1,+100] の帰無割合 "
          "= 9900/10^6 = 0.99%）。",
          "- κ は checkpoint 由来の**近似**（`reg_logw_interp`）。step 1M checkpoint は"
          "本走ではなく別実現なので per-unit の主張には使えない。主結果は κ を使わない。", ""]

    L += ["## 2. H-a: 境界での 1 step ジャンプ", "",
          "境界対 (B, B+1) と、同じ窓内の非境界 1 step 対を、いずれも「その時点で on」"
          "の unit に限って比べる。`Δs = Δ(w·µ) + Δb` は厳密な分解。", ""]
    L += [md_table(pd.concat([res[a["label"]]["jump"] for a in ARMS],
                             ignore_index=True)), ""]
    for a in ARMS:
        lab = a["label"]; J = res[lab]["jump"]
        b_w = _jump_med(res[lab], "boundary_0_to_1", "Δ(w·µ)")
        b_b = _jump_med(res[lab], "boundary_0_to_1", "Δb")
        c_w = _jump_med(res[lab], "control_1step", "Δ(w·µ)")
        c_b = _jump_med(res[lab], "control_1step", "Δb")
        mj = res[lab]["sinfo"]["med_abs_jump_over_w"]
        mb_w = _jump_mean(res[lab], "boundary_0_to_1", "Δ(w·µ)")
        mc_w = _jump_mean(res[lab], "control_1step", "Δ(w·µ)")
        mb_b = _jump_mean(res[lab], "boundary_0_to_1", "Δb")
        mc_b = _jump_mean(res[lab], "control_1step", "Δb")
        L += [f"- **{lab}**: 境界の |Δ(w·µ)| は median {b_w:.4f} / mean {mb_w:.4f}、"
              f"非境界 1 step は median {c_w:.5f} / mean {mc_w:.5f}"
              f"（mean 比 {mb_w / mc_w:.0f}倍）。同じ境界 1 step の |Δb| は "
              f"median {b_b:.5f} / mean {mb_b:.5f} で、非境界の mean {mc_b:.5f} の "
              f"{mb_b / mc_b:.1f} 倍にしかならない。つまり **Δs のジャンプは w·µ 側で "
              f"mean 比 {mb_w / mb_b:.0f}:1 で w·µ が支配**する。"
              f" 規格化ジャンプ median |Δs|/‖w‖ = {mj:.4f}"
              f"（1 ビット反転なら Δ(w·µ) = ±w_j なので |w_j|/‖w‖ の桁）。"]
    L += ["", "**構造サニティ（H-a の機構が代数どおりか）**", ""]
    rows = []
    for a in ARMS:
        rows.append(dict(arm=a["label"], **res[a["label"]]["sanity"]))
    L += [md_table(pd.DataFrame(rows)), "",
          f"std では µ = [flip ‖ 0.5·1_5] なので ‖µ‖² = Σflip + 1.25 が成り立ち"
          f"（全 20,901 × 10 seed で最大偏差 "
          f"{res['std']['sanity']['max_dev_mu2_minus_flipsum_1p25']:.1e} = float32 の丸め"
          "）、境界では Δ‖µ‖² = ±1、すなわち **flip は毎回ちょうど 1 ビット**。"
          "したがって Δ(w·µ) は（1 step 分の SGD を除けば）その座標の重み ±w_j そのもの。"
          "centered では running_mean が入るので前者の恒等式は成り立たないが、境界で "
          "‖µ‖ が bulk 値からほぼ 1 へ跳ぶ（= 1 ビット分だけ中心化が外れる）ことが読める。", ""]
    L += ["**判定: H-a 支持。** ジャンプは w·µ 側にほぼ 100% 集中し、b は 1 step では"
          "文字通り動かない（多くの unit ではその step にゲートが開かず勾配自体が入らない）。", ""]

    L += ["## 3. H-b: 境界後の追随と時定数", "",
          "共通ドリフトを消すため、κ フリーの指標 "
          "`gap_s(o) = median_up(Δs(o)) − median_down(Δs(o))` を主に使う"
          "（Δs(o) = s(B+o) − s(B)、群は offset +1 のジャンプ符号で定義）。"
          "ジャンプの記憶が消えれば gap → 0 になる。", ""]
    rows = []
    for a in ARMS:
        lab = a["label"]; g = res[lab]["gap"]; f = res[lab]["gap_fit"]
        for o in REPORT_OFF:
            r = g[g.offset == o].iloc[0]
            rows.append(dict(arm=lab, offset=o, gap_s=r.gap_s, gap_ratio=r.gap_ratio,
                             gap_wmu=r.gap_wmu, gap_b=r.gap_b,
                             gap_b_onrun=r.gap_b_onrun))
    L += [md_table(pd.DataFrame(rows)), ""]
    rows = []
    for a in ARMS:
        lab = a["label"]; f = res[lab]["gap_fit"]
        rows.append(dict(arm=lab, gap_at_offset1=f["gap_at_1"],
                         tau_fine_1_100=f["fine_1_100"], tau_bulk_1k_9k=f["bulk_1k_9k"],
                         half_offset=f["half_offset"] if f["half_offset"] else np.nan,
                         gap_ratio_100=f["gap_ratio_at_100"],
                         gap_ratio_1000=f["gap_ratio_at_1000"],
                         gap_ratio_9000=f["gap_ratio_at_9000"]))
    L += ["時定数（gap 比の対数線形フィット。半減 offset は fine グリッド上の最初の点）:", "",
          md_table(pd.DataFrame(rows)), ""]
    f = res["std"]["gap_fit"]
    L += [f"- **std**: gap は offset 1 の {f['gap_at_1']:.3f} から 9000 step 後でも "
          f"{f['gap_ratio_at_9000']*100:.0f}% 残る。減衰は単一指数ではなく、"
          f"微細窓 (1..100) のフィットで {_fmt(f['fine_1_100'],3)} step、"
          f"バルク区間 (1000..9000) で {_fmt(f['bulk_1k_9k'],3)} step ≈ "
          f"{f['bulk_1k_9k']/PERIOD:.0f} 周期。どちらの読み方でも"
          "**次の境界（10^4 step 後）までに回復し切らない**（半減 offset は"
          "1 周期内に存在しない）。b が担う分 "
          f"(gap_b) は 9000 step 後で {_gap_at(res['std'], 9000, 'gap_b'):.4f}"
          f" で、gap 全体の "
          f"{abs(_gap_at(res['std'], 9000, 'gap_b') / _gap_at(res['std'], 9000, 'gap_s')) * 100:.1f}% "
          "しか埋めていない（`gap_s = gap_wmu + gap_b` なので gap_b < 0 は"
          "**回復向き**。向きは合っているが桁が足りない）。gap のほぼ全部は "
          "w·µ 側に残ったまま = **w も b も新しい µ に合わせ直さない**。"]
    f = res["centered"]["gap_fit"]
    L += [f"- **centered**: gap は {f['half_offset']} step で半減し、"
          f"offset 100 で {f['gap_ratio_at_100']*100:.0f}%、1000 step で "
          f"{f['gap_ratio_at_1000']*100:.0f}% まで落ちる。fine 窓のフィット時定数は "
          f"{_fmt(f['fine_1_100'],3)} step で、**center_alpha = 0.01 の "
          "running_mean 時定数 1/α = 100 step と一致**する。回復の担い手は b ではなく "
          "µ パルスの減衰（‖µ‖ が +1 の 0.99 から bulk の 0.075 へ戻る）。",
          f"- centered の残差 gap（9000 step 後 {f['gap_ratio_at_9000']*100:.0f}%）は"
          f"ほぼ全て b 側にあり（gap_b = {_gap_at(res['centered'], 9000, 'gap_b'):.4f} / "
          f"gap_wmu = {_gap_at(res['centered'], 9000, 'gap_wmu'):.4f}）、しかも符号は "
          "std と**逆で反回復**（gap_b > 0 = 下向きにはたかれた unit の b はむしろ"
          "下がり、gap を広げる）。µ パルスが引いた後に **b の側に境界の記憶が残る**。"
          "ただしこの量は生存者バイアスを含みうる（§7）。", ""]
    L += ["**判定: H-b は仮説の形では不支持。** 「b が追いつこうとするが時定数がある」ではなく、"
          "std では **b の追随が 1 周期で gap の数 % しか埋めない**（時定数 10^5 step 台 = "
          "10 周期以上）。centered で見える回復は b ではなく入力側の中心化フィルタによるもの。"
          "なお `on_run`（境界後 100 step 連続で点灯）に限った gap_b も併記した"
          "（消灯 unit は勾配ゼロで b が動かないため、選択効果を分けて見るため）。", ""]

    L += ["## 4. H-c: 死のタイミング", ""]
    L += [md_table(pd.concat([res[a["label"]]["dconc"] for a in ARMS],
                             ignore_index=True)), ""]
    rows = []
    for a in ARMS:
        lab = a["label"]; i = res[lab]["dinfo"]; sc = res[lab]["seed_conc"]
        rows.append(dict(arm=lab, n_death=i["n_death"],
                         frac_still_dead_at_1M=i["frac_final_dead"],
                         seed_min_frac_win=float(sc.frac_win.min()),
                         seed_max_frac_win=float(sc.frac_win.max()),
                         seed_min_frac_at1=float(sc.frac_at1.min()),
                         seed_max_frac_at1=float(sc.frac_at1.max())))
    L += ["seed 間ばらつき（10 seed それぞれで計算した割合の範囲）:", "",
          md_table(pd.DataFrame(rows)), ""]
    L += ["`-99..-1` は物理的にはただのバルク時間（前の境界から 9900 step 後）だが、"
          "そこだけ記録が 1 step 刻みなので、その率比は**密グリッドによる検出バイアス**"
          "の推定値になる（粗い 1000 step 粒度では短い一過性の落ち込みを見逃しうる）。"
          "`rate_ratio_corr` はそれで割った補正値。`subset = still_dead_at_1M` は "
          "step 10^6 でも消灯したままの死亡だけに絞った頑健性チェック。", ""]
    for a in ARMS:
        lab = a["label"]; i = res[lab]["dinfo"]
        w = _conc(res[lab], "+1..+100")
        o1 = _conc(res[lab], "offset = +1")
        pre = _conc(res[lab], "-99..-1")
        wp = _conc(res[lab], "+1..+100", "still_dead_at_1M")
        L += [f"- **{lab}**: 全 {i['n_death']} 死亡のうち {w.frac*100:.1f}% が "
              f"offset +1..+100（時間占有 {w.null_frac*100:.2f}%、率比 "
              f"{w.rate_ratio:.0f}倍、検出バイアス補正後 {w.rate_ratio_corr:.0f}倍）。"
              f"**{o1.frac*100:.1f}% は offset +1 ちょうど**（= flip の 1 step 後）。"
              f"境界**前**の窓 −99..−1 は {pre.frac*100:.1f}%"
              f"（帰無 {pre.null_frac*100:.2f}%、率比 {pre.rate_ratio:.1f}倍 "
              f"= 検出バイアスの推定値）。恒久的な死亡だけに絞っても "
              f"{wp.frac*100:.1f}% が +1..+100 で、結論は変わらない。"]
    L += ["", "**判定: H-c 支持（強い）。** 死は「境界のジャンプ直後」に集中し、"
          "しかも最頻値は flip 直後の 1 step 目である。検出バイアスを補正しても "
          f"std {_conc(res['std'], '+1..+100').rate_ratio_corr:.0f}倍 / "
          f"centered {_conc(res['centered'], '+1..+100').rate_ratio_corr:.0f}倍 の"
          "濃縮が残る。", ""]

    L += ["### 4.1 ジャンプ直前の余裕・ジャンプ幅と消灯", "",
          "余裕の**厳密**な序数は境界直前の p̂ = k/32 自身（p̂ は s+M の単調増加関数）。"
          "物理スケール `margin_hat/‖w‖ = s/‖w‖ + κ̂` は κ̂ 近似を含む。", ""]
    L += [md_table(pd.concat([res[a["label"]]["silence_p"] for a in ARMS],
                             ignore_index=True)), ""]
    L += ["下向きジャンプのみを幅の五分位で層別:", "",
          md_table(pd.concat([res[a["label"]]["silence_j"] for a in ARMS],
                             ignore_index=True)), ""]
    rows = []
    for a in ARMS:
        lab = a["label"]; i = res[lab]["sinfo"]
        rows.append(dict(arm=lab, n_event=i["n_event"], frac_down=i["frac_down"],
                         P_sil=i["P_silenced_at_1"], P_sil_down=i["P_silenced_at_1_down"],
                         P_sil_up=i["P_silenced_at_1_up"],
                         med_p0_silenced=i["med_p0_silenced"],
                         med_p0_survived=i["med_p0_survived"],
                         med_margin_silenced=i["med_margin_silenced"],
                         med_margin_survived=i["med_margin_survived"],
                         kappa_rule_acc=i["kappa_rule_accuracy"]))
    L += [md_table(pd.DataFrame(rows)), ""]
    for a in ARMS:
        lab = a["label"]; i = res[lab]["sinfo"]; Bp = res[lab]["silence_p"]
        L += [f"- **{lab}**: 境界直後に消灯する確率は k=1（余裕が最小）で "
              f"{Bp.P_silenced_at_1.iloc[0]*100:.1f}%、k=17-32 で "
              f"{Bp.P_silenced_at_1.iloc[-1]*100:.2f}% と**単調に減る**。消灯した"
              f"イベントの境界直前 p̂ 中央値は {i['med_p0_silenced']:.4f}、"
              f"生き残った側は {i['med_p0_survived']:.4f}。"
              f"κ̂ 規則 `margin_hat + Δs <= 0` の per-event 正解率は "
              f"{i['kappa_rule_accuracy']*100:.1f}%（κ̂ が近似であることの目安）。"]
    L += ["", "**判定: 「余裕が小さい unit ほど死ぬ」は支持。** ジャンプ幅の効果も同方向"
          "（深いほど消灯しやすい）。ただし両者は厳密には同じ不等式 s+M+Δs<=0 の"
          "二つの項なので、独立な証拠ではない。", ""]

    L += ["## 5. H-d: ラチェット（境界 index に対する推移）", ""]
    L += [md_table(pd.concat([res[a["label"]]["haz_blocks"] for a in ARMS],
                             ignore_index=True)), ""]
    for a in ARMS:
        lab = a["label"]; h = res[lab]["haz"]; hb = res[lab]["haz_blocks"]
        L += [f"- **{lab}**: 生存割合（p̂ >= τ）は境界 1 回目の "
              f"{h.alive_frac.iloc[0]:.3f} から 99 回目の {h.alive_frac.iloc[-1]:.3f} へ"
              f"減る。状態ベースのハザードは最初のブロック {hb.hazard_state.iloc[0]:.4f}"
              f"（95% CI {hb.hazard_state_lo.iloc[0]:.4f}–{hb.hazard_state_hi.iloc[0]:.4f}）"
              f"、最後のブロック {hb.hazard_state.iloc[-1]:.4f}"
              f"（{hb.hazard_state_lo.iloc[-1]:.4f}–{hb.hazard_state_hi.iloc[-1]:.4f}）、"
              f"全 9 ブロックの範囲は {hb.hazard_state.min():.4f}–"
              f"{hb.hazard_state.max():.4f}。一方「最初の死亡」の**実数**は "
              f"{int(hb.n_death_win.iloc[0])} 件 → {int(hb.n_death_win.iloc[-1])} 件と"
              f"落ち、その risk set も {int(hb.n_at_risk.iloc[0])} → "
              f"{int(hb.n_at_risk.iloc[-1])} と枯れる。"]
    L += ["", "**判定: H-d 支持（ただし読み方に注意）。** 境界 1 回あたりの消灯"
          "ハザードは**時間とともに下がらない**（std は全ブロックで 0.10–0.19、"
          "centered は 0.004–0.012 で、むしろ後半のほうがやや高い）。"
          "「削られた実数」が前半に偏るのは、ハザードが下がるからではなく"
          "**削られる母集団（生存 unit）が先に枯れる**ため。std の生存割合は "
          f"{res['std']['haz'].alive_frac.iloc[0]:.2f}（境界 1）→ "
          f"{res['std']['haz'].alive_frac.iloc[29]:.2f}（境界 30）→ "
          f"{res['std']['haz'].alive_frac.iloc[-1]:.2f}（境界 99）。",
          "- 留保: 後半 std に残る少数の生存 unit は「境界で消灯 → その後復活」を"
          "繰り返す辺縁的な集団で、`hazard_state` はこの再消灯も数える"
          "（`death_events` は unit ごとに最初の 1 回しか数えないので、"
          "その分母は後半に枯れて読めなくなる）。二つの分母は別の問いに答えている。", ""]

    L += ["## 6. centered との対比", ""]
    rows = []
    for a in ARMS:
        lab = a["label"]; C = res[lab]["dconc"]; T = res[lab]["traj_tab"]
        mm = (T["group"] == "all")
        mu1 = float(T[mm & (T["offset"] == 1)].med_mu_norm.iloc[0])
        mub = float(T[mm & (T["offset"] == 5000)].med_mu_norm.iloc[0])
        rows.append(dict(arm=lab, mu_norm_at_off1=mu1, mu_norm_bulk=mub,
                         frac_death_win=float(_conc(res[lab], "+1..+100").frac),
                         rate_ratio_win=float(_conc(res[lab], "+1..+100").rate_ratio),
                         rate_ratio_win_corr=float(
                             _conc(res[lab], "+1..+100").rate_ratio_corr),
                         detection_bias=float(
                             _conc(res[lab], "+1..+100").detection_bias_est),
                         rate_ratio_at1_corr=float(
                             _conc(res[lab], "offset = +1").rate_ratio_corr),
                         frac_death_at1=float(_conc(res[lab], "offset = +1").frac),
                         n_death=res[lab]["dinfo"]["n_death"],
                         frac_still_dead=res[lab]["dinfo"]["frac_final_dead"],
                         gap_ratio_9000=res[lab]["gap_fit"]["gap_ratio_at_9000"]))
    L += [md_table(pd.DataFrame(rows)), ""]
    L += ["- 予想は「centered は bulk で µ がほぼ無いので、死が境界直後に**さらに強く**"
          "集中するはず」だった。実測は逆で、集中は std のほうが強い"
          f"（+1..+100 の率比 {rows[0]['rate_ratio_win']:.0f}倍 vs "
          f"{rows[1]['rate_ratio_win']:.0f}倍、検出バイアス補正後だと "
          f"{rows[0]['rate_ratio_win_corr']:.0f}倍 vs "
          f"{rows[1]['rate_ratio_win_corr']:.0f}倍でさらに差が開く）。"
          "**この副仮説は不支持**。",
          f"- ただし centered でも **offset +1 ちょうど**への集中は鋭い"
          f"（補正後の率比 {rows[1]['rate_ratio_at1_corr']:.0f}倍）。centered の"
          "µ パルスは 1/α = 100 step で減衰するのに、死は最初の 1 step に偏る。"
          "パルスの**立ち上がり**（ステップ関数）が効いていて、パルスの尾は効いて"
          "いない、と読める。",
          "- 一方で「centered に残る死の相当部分が境界イベントである」ことは"
          f"支持される（centered の死の {rows[1]['frac_death_win']*100:.0f}% が "
          f"+1..+100、{rows[1]['frac_death_at1']*100:.0f}% が offset +1 ちょうど）。"
          "残り（bulk での死）は µ ではなく b の下方ドリフトが M を下回る経路とみられるが、"
          "本解析では分離していない。",
          f"- 重要な留保: centered の「死亡」イベントのうち step 10^6 時点でも消灯している"
          f"のは {rows[1]['frac_still_dead']*100:.0f}% しかない"
          f"（std は {rows[0]['frac_still_dead']*100:.0f}%）。centered の死は"
          "「1 周期は戻らないが、その後戻ることがある」ものを多く含む。", ""]

    L += ["## 7. 留保", "",
          "- すべて**事後計算・未事前登録**。群（上下ジャンプ）の定義も offset +1 の"
          "観測値に基づく事後的なものである。ジャンプは 1 step の SGD 変化より 2 桁大きい"
          "ので符号の誤分類はほぼ無いが、形式的には選択が入っている。",
          "- `on_run`（境界後 100 step 連続点灯）で層別した量は**生存者バイアス**を含む"
          "（消灯した unit は勾配が入らず b が動かないので、b の動きを見るには除く必要が"
          "あるが、除くこと自体がジャンプ方向と相関する）。",
          "- κ̂ は checkpoint 由来の近似で、step 1,000,000 の checkpoint は本走ではなく"
          "同 config の別実現である（`q3_margin_pooled` の checkpoint_fidelity 参照）。"
          "per-unit の margin_hat は指標としてのみ使い、判定には使っていない。",
          "- 死亡時刻は「記録点」の解像度でしか分からない。バルク区間（1000 step 粒度）に"
          "落ちた死は最大 900 step の不確かさを持つ。集中度の帰無値は coverage 重みで"
          "この非一様性を吸収してあるが、窓外の死の**細かい**時刻分布は分からない。",
          "- 死亡定義は `death_recover_periods = 1`（1 周期戻らない）。この定義では"
          "「その後に復活する死」も数える。arm ごとの復活率を §6 に併記した。",
          "- 検出バイアスの推定（`-99..-1` の率比）はそれ自体が推定であり、境界後の窓に"
          "同じ倍率が当てはまる保証はない。ただし **offset +1 と `-99..-1` は同じ密"
          "グリッド上にある**ので、+1 の突出（std 28%・centered 26% が 1 step に集中）は"
          "グリッドの粗密では説明できない。",
          "- H-d の `hazard_state` は `death_events` の判定規則を「最初の 1 回」制限"
          "なしに適用した**拡張**であり、事前登録された死亡定義そのものではない"
          "（`freeze_events_all`）。分母が枯れない代わりに、復活後の再消灯も数える。",
          "- 単一の設定（condA・w100・T=10^4・batch=1・lr=0.01・center_alpha=0.01）のみ。"
          "T や lr を振っていないので、時定数の比較（追随 10^5 step vs 境界間隔 10^4 step）"
          "が設定依存かどうかは本解析では分からない。", ""]

    L += ["## 8. 生成物", "",
          "- `jump_stats.csv` — H-a: 境界 1 step vs 非境界 1 step の Δ 統計",
          "- `aligned_trajectory.csv` — 境界整列した Δs/Δb/Δ(w·µ)/p̂/‖µ‖ の群別軌跡",
          "- `recovery_gap.csv` — 上下ジャンプ群の gap とその内訳（κ フリー）",
          "- `death_offset_hist.csv` / `death_concentration.csv` / "
          "`death_concentration_by_seed.csv` / `death_events.csv` — H-c",
          "- `silencing_by_margin.csv` / `silencing_by_jump.csv` — 余裕・ジャンプ幅と消灯",
          "- `hazard_by_boundary.csv` / `hazard_blocks.csv` — H-d",
          "- `figures/fig_race_aligned.png` / `fig_death_offset_hist.png` / "
          "`fig_hazard_boundary_index.png` / `fig_margin_vs_silencing.png`", ""]
    (OUTDIR / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
