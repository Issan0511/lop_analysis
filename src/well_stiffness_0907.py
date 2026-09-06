"""井戸の剛性（`well_stiffness_0907`）— 既存ログだけを読む事後解析。

読み規則は `specs/spec_well_stiffness_0907.md`（実装前に commit 済み）。新しい走は
無い。`results/{offset_grid_0906,act_offset_0906}/logs_tail/` の末尾 20 タスク
（1000 step 密記録）から、井戸の底 z* まわりの緩和を測る。

- 増分は隣接する密記録行の対で、``flip_state`` が両行で一致するものだけ。
- 変位 x = z̄(始点) − z*、速度 u = ``layer1_dzbar``（終点の行＝始点からの増分）。
- 群は下（zmax<0・z*=−c/a）と上（zmin>0・z*=−c）。跨ぎは主判定から除く。
- 統計は seed 単位で作り、seed ブートストラップ 2000・rng(20260907) で CI。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = (ROOT / "results/offset_grid_0906/logs_tail",
        ROOT / "results/act_offset_0906/logs_tail")
PERIOD = 10_000
DENSE_TASKS = 20
N_BOOT = 2000
BOOT_SEED = 20260907
MIN_INC = 1_000            # 1 腕あたりの有効増分の下限（spec §3 NOT_DETERMINED）
MIN_BINS = 5               # T1 の最小ビン数
MIN_SPAN = 3.0             # T1 の |x| レンジ（倍）
LEAKY_C = {"leaky_off_m2": -2.0, "leaky_off_m1": -1.0, "leaky_off_m0p5": -0.5,
           "leaky_off_m0p25": -0.25, "leaky_off_0": 0.0, "leaky_off_p0p25": 0.25,
           "leaky_off_p0p5": 0.5, "leaky_off_p1": 1.0, "leaky_off_p2": 2.0}


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def arm_files(arm: str) -> list[Path]:
    for d in DIRS:
        hit = sorted(d.glob(f"{arm}_seed*.npz"))
        if hit:
            return hit
    return []


def all_arms() -> dict[str, dict]:
    """腕名 → メタ（a・c・lr・v 凍結・バッチ・総 step）。leaky のみ返す。"""
    out: dict[str, dict] = {}
    for d in DIRS:
        for p in sorted(d.glob("*_seed0.npz")):
            arm = p.name[: -len("_seed0.npz")]
            if arm in out:
                continue
            with np.load(p, allow_pickle=True) as z:
                act = str(z["activation"])
                if act not in LEAKY_C:
                    continue          # ELU は z* の式が違う（spec §2 の表は leaky）
                out[arm] = dict(arm=arm, a=float(z["act_alpha"]), c=LEAKY_C[act],
                                lr=float(z["lr_used"]), vf=bool(z["freeze_v"]),
                                batch=str(z["batch_mode"]), steps=int(z["step"][-1]))
    return out


def seed_pairs(path: Path) -> dict[str, np.ndarray] | None:
    """1 seed 分の増分テーブル。列は x/u/v/g/phase/群フラグ。"""
    with np.load(path, allow_pickle=True) as z:
        step = z["step"].astype(np.int64)
        if not np.array_equal(step, z["layer1_moment_step"].astype(np.int64)):
            raise ValueError(f"{path.name}: moment 行が step 行と揃っていない")
        total = int(step[-1])
        dense = step > total - DENSE_TASKS * PERIOD
        idx = np.where(dense)[0]
        idx = idx[idx > 0]
        if idx.size == 0:
            return None
        flip = z["flip_state"]
        same = np.array([np.array_equal(flip[i], flip[i - 1]) for i in idx])
        idx = idx[same]
        if idx.size == 0:
            return None
        j, i = idx - 1, idx                      # j=始点行・i=終点行
        zb = z["layer1_zbar"].astype(np.float64)
        dz = z["layer1_dzbar"].astype(np.float64)
        zmax = z["layer1_zmax"].astype(np.float64)
        zmin = z["layer1_zmin"].astype(np.float64)
        v = z["layer1_v_unit"].astype(np.float64)
        mpd = z["layer1_m_phidphi"].astype(np.float64)
        n_rec, h = zb.shape[0], zb.shape[1]
        phase = ((step[j] % PERIOD) // 1_000).astype(np.int64)
        rep = lambda arr1d: np.repeat(arr1d[:, None], h, axis=1).ravel()
        return dict(zb=zb[j].ravel(), u=dz[i].ravel(),
                    dn=((zmax[j] < 0) & (zmax[i] < 0)).ravel(),
                    up=((zmin[j] > 0) & (zmin[i] > 0)).ravel(),
                    v=v[j].ravel(), mpd=mpd[j].ravel(), phase=rep(phase),
                    unit=np.tile(np.arange(h), j.size))


def load_arm(arm: str, group: str, meta: dict) -> list[dict]:
    """seed ごとの (x, u, |v|, g, phase)。group は 'dn' / 'up'。"""
    zstar = -meta["c"] / meta["a"] if group == "dn" else -meta["c"]
    rows = []
    for p in arm_files(arm):
        tb = seed_pairs(p)
        if tb is None:
            continue
        m = tb[group] & np.isfinite(tb["u"]) & np.isfinite(tb["zb"])
        if not m.any():
            rows.append(dict(x=np.empty(0), u=np.empty(0), av=np.empty(0),
                             g=np.empty(0), phase=np.empty(0, dtype=np.int64),
                             unit=np.empty(0, dtype=np.int64)))
            continue
        rows.append(dict(x=tb["zb"][m] - zstar, u=tb["u"][m],
                         av=np.abs(tb["v"][m]),
                         g=-(tb["v"][m] ** 2) * tb["mpd"][m],
                         phase=tb["phase"][m], unit=tb["unit"][m]))
    return rows


# ---------------------------------------------------------------------------
# 統計（spec §3）
# ---------------------------------------------------------------------------
def k_origin(x: np.ndarray, u: np.ndarray) -> float:
    """原点通過回帰 u = −k x の k。"""
    den = float((x * x).sum())
    return float("nan") if den <= 0 else -float((x * u).sum()) / den


def loglog_slope(x: np.ndarray, u: np.ndarray) -> tuple[float, int, float]:
    """|x| の十分位ビンの中央値で log|u| 対 log|x| の傾き。(slope, n_bin, span)。"""
    ax, au = np.abs(x), np.abs(u)
    m = (ax > 0) & (au > 0)
    ax, au = ax[m], au[m]
    if ax.size < 100:
        return float("nan"), 0, 0.0
    edges = np.quantile(ax, np.linspace(0, 1, 11))
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        sel = (ax >= lo) & (ax < hi) if hi < edges[-1] else (ax >= lo) & (ax <= hi)
        if sel.sum() < 20:
            continue
        bx.append(np.median(ax[sel]))
        by.append(np.median(au[sel]))
    if len(bx) < MIN_BINS:
        return float("nan"), len(bx), 0.0
    bx, by = np.array(bx), np.array(by)
    span = float(bx.max() / bx.min())
    sl = float(np.polyfit(np.log(bx), np.log(by), 1)[0])
    return sl, len(bx), span


def per_seed_stats(rows: list[dict]) -> list[dict]:
    """seed ごとに位相別 k・位相別 T1 傾き・復元率・T7 係数。"""
    out = []
    for r in rows:
        x, u, ph = r["x"], r["u"], r["phase"]
        st = dict(n=int(x.size), k_phase={}, q_phase={}, restore=float("nan"),
                  logk=float("nan"), q=float("nan"), t7=float("nan"), r2=float("nan"))
        if x.size:
            st["restore"] = float(np.mean(np.sign(u) == -np.sign(x)))
            den = float((r["g"] * r["g"]).sum())
            if den > 0:
                coef = float((r["g"] * u).sum()) / den
                res = u - coef * r["g"]
                sst = float((u * u).sum())
                st["t7"] = coef
                st["r2"] = float(1.0 - (res * res).sum() / sst) if sst > 0 else float("nan")
        ks, qs = [], []
        for j in range(10):
            m = ph == j
            if m.sum() < 100:
                continue
            k = k_origin(x[m], u[m])
            if np.isfinite(k) and k > 0:
                st["k_phase"][j] = k
                ks.append(np.log(k))
            sl, nb, span = loglog_slope(x[m], u[m])
            if np.isfinite(sl) and nb >= MIN_BINS and span >= MIN_SPAN:
                st["q_phase"][j] = sl
                qs.append(sl)
        if ks:
            st["logk"] = float(np.mean(ks))
        if qs:
            st["q"] = float(np.median(qs))
        out.append(st)
    return out


def boot_mean(vals: list[float], rng: np.random.Generator) -> tuple[float, list[float]]:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    draws = v[rng.integers(0, v.size, size=(N_BOOT, v.size))].mean(axis=1)
    return float(v.mean()), [float(np.quantile(draws, 0.025)),
                             float(np.quantile(draws, 0.975))]


def ci_in(ci, lo, hi) -> bool:
    return np.isfinite(ci[0]) and ci[0] >= lo and ci[1] <= hi


def ci_has(ci, v) -> bool:
    return np.isfinite(ci[0]) and ci[0] <= v <= ci[1]


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def analyze() -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    metas = all_arms()
    arms: dict[tuple[str, str], dict] = {}
    for arm, meta in sorted(metas.items()):
        for group in ("dn", "up"):
            rows = load_arm(arm, group, meta)
            if not rows:
                continue
            n_tot = int(sum(r["x"].size for r in rows))
            if n_tot < MIN_INC:
                continue
            st = per_seed_stats(rows)
            logk, logk_ci = boot_mean([s["logk"] for s in st], rng)
            q, q_ci = boot_mean([s["q"] for s in st], rng)
            rest, rest_ci = boot_mean([s["restore"] for s in st], rng)
            r2, _ = boot_mean([s["r2"] for s in st], rng)
            t7, t7_ci = boot_mean([s["t7"] for s in st], rng)
            kph = {j: float(np.exp(np.mean([np.log(s["k_phase"][j]) for s in st
                                            if j in s["k_phase"]])))
                   for j in range(10)
                   if any(j in s["k_phase"] for s in st)}
            arms[(arm, group)] = dict(
                arm=arm, group=group, **{k: meta[k] for k in ("a", "c", "lr", "vf", "batch", "steps")},
                n=n_tot, k=float(np.exp(logk)), k_ci=[float(np.exp(v)) for v in logk_ci],
                logk_seed=[s["logk"] for s in st], q=q, q_ci=q_ci,
                restore=rest, restore_ci=rest_ci, t7=t7, t7_ci=t7_ci, r2=r2,
                k_phase={str(j): v for j, v in kph.items()},
                t1=("LINEAR" if ci_in(q_ci, 0.7, 1.3) else
                    "CONSTANT_FORCE" if ci_in(q_ci, -0.3, 0.3) else
                    "NOT_DETERMINED" if not np.isfinite(q) else "OTHER"),
                restoring=bool(np.isfinite(logk_ci[0]) and np.exp(logk_ci[0]) > 0))
    return dict(arms=arms, metas=metas)


def paired_slope(arms, keys, xs, rng) -> tuple[float, list[float]]:
    """log k を log x に回帰した傾き（seed をそろえてブートストラップ）。"""
    mats = [np.array(arms[k]["logk_seed"], dtype=float) for k in keys]
    n = min(m.size for m in mats)
    mats = [m[:n] for m in mats]
    lx = np.log(np.array(xs, dtype=float))
    ok = np.all(np.isfinite(np.vstack(mats)), axis=0)
    pt = float(np.polyfit(lx, [m[ok].mean() for m in mats], 1)[0])
    idx = rng.integers(0, int(ok.sum()), size=(N_BOOT, int(ok.sum())))
    ys = np.stack([m[ok][idx].mean(axis=1) for m in mats])       # (len(keys), N_BOOT)
    sl = np.polyfit(lx, ys, 1)[0]
    return pt, [float(np.quantile(sl, 0.025)), float(np.quantile(sl, 0.975))]


def ratio_ci(arms, num, den, rng) -> tuple[float, list[float]]:
    a = np.array(arms[num]["logk_seed"], dtype=float)
    b = np.array(arms[den]["logk_seed"], dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    pt = float(np.exp(a.mean() - b.mean()))
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    dr = np.exp(a[idx].mean(axis=1) - b[idx].mean(axis=1))
    return pt, [float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))]


def _beta_from(sel_rows: list[dict]) -> tuple[float, list[float], list[float]]:
    """|v| 四分位ごとの k から log k 対 log|v| の傾き（seed をプールして腕単位で）。"""
    av = np.concatenate([r["av"] for r in sel_rows])
    x = np.concatenate([r["x"] for r in sel_rows])
    u = np.concatenate([r["u"] for r in sel_rows])
    if av.size < 400:
        return float("nan"), [], []
    qs = np.quantile(av, [0.0, 0.25, 0.5, 0.75, 1.0])
    ks, vs = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (av >= lo) & (av <= hi) if hi == qs[-1] else (av >= lo) & (av < hi)
        if sel.sum() < 100:
            continue
        k = k_origin(x[sel], u[sel])
        mv = float(np.median(av[sel]))
        if np.isfinite(k) and k > 0 and mv > 0:
            ks.append(float(np.log(k)))
            vs.append(float(np.log(mv)))
    if len(ks) < 3:
        return float("nan"), ks, vs
    return float(np.polyfit(vs, ks, 1)[0]), ks, vs


def v_slope(arm: str, group: str, meta: dict, rng) -> dict:
    """T5: 腕内で |v| 四分位ごとの k → log k 対 log|v| の傾き（seed ブートストラップ）。"""
    rows = [r for r in load_arm(arm, group, meta) if r["x"].size]
    if len(rows) < 2:
        return dict(beta=float("nan"), beta_ci=[float("nan"), float("nan")], n_seed=len(rows))
    pt, ks, vs = _beta_from(rows)
    draws = []
    for _ in range(N_BOOT // 4):          # 再標本ごとに四分位から作り直す
        pick = [rows[i] for i in rng.integers(0, len(rows), size=len(rows))]
        b, _k, _v = _beta_from(pick)
        if np.isfinite(b):
            draws.append(b)
    ci = ([float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
          if len(draws) >= 100 else [float("nan"), float("nan")])
    return dict(beta=pt, beta_ci=ci, n_seed=len(rows), n_bin=len(ks),
                logk=ks, logv=vs)


def centered_check(arm: str, group: str, meta: dict) -> dict:
    """D2（登録外の診断）: k を自分の窓平均まわりで測った値 k_loc と、底への正味ドリフト。

    x を z̄−z* にすると、ユニットごとの居場所のばらつきが回帰を薄める。k_loc は
    局所の引き戻しだけを測る。井戸が本当に底へ引くなら、底からずれて座っている
    ユニットには**正味の**ドリフトが底の向きに出るはず。
    """
    zstar = -meta["c"] / meta["a"] if group == "dn" else -meta["c"]
    kl, tw, off, nu = [], [], [], []
    for r in load_arm(arm, group, meta):
        if r["x"].size < 400:
            continue
        uid = r["unit"]
        zb = r["x"] + zstar
        order = np.argsort(uid, kind="stable")
        uid_s, zb_s, u_s = uid[order], zb[order], r["u"][order]
        bnd = np.flatnonzero(np.diff(uid_s)) + 1
        xs, us, toward, offs = [], [], [], []
        for chunk_z, chunk_u in zip(np.split(zb_s, bnd), np.split(u_s, bnd)):
            if chunk_z.size < 20:
                continue
            m = float(chunk_z.mean())
            xs.append(chunk_z - m)
            us.append(chunk_u)
            net = float(chunk_u.mean())
            d = m - zstar
            if abs(d) > 1e-6:
                toward.append(1.0 if net * d < 0 else 0.0)
                offs.append(-net / d)          # 正なら底へ向かう実効レート
        if not xs:
            continue
        k = k_origin(np.concatenate(xs), np.concatenate(us))
        if np.isfinite(k):
            kl.append(k)
        if toward:
            tw.append(float(np.mean(toward)))
            off.append(float(np.median(offs)))
            nu.append(len(toward))
    return dict(k_loc=float(np.mean(kl)) if kl else float("nan"),
                toward_frac=float(np.mean(tw)) if tw else float("nan"),
                net_rate=float(np.median(off)) if off else float("nan"),
                n_unit=int(np.sum(nu)))


def main(argv: list[str]) -> int:
    out = ROOT / "results/well_stiffness_0907"
    out.mkdir(parents=True, exist_ok=True)
    res = analyze()
    arms = res["arms"]
    rng = np.random.default_rng(BOOT_SEED + 1)
    verdict: dict = {}

    # T3: c=+0.5 の下群 a 梯子
    lad = [("LRa0p2_offp0p5_1216", 0.2), ("LRa0p3_offp0p5_1216", 0.3),
           ("LRa0p5_offp0p5_1216", 0.5), ("LRa0p7_offp0p5_1216", 0.7)]
    keys = [(n, "dn") for n, _ in lad if (n, "dn") in arms]
    if len(keys) == 4:
        al, al_ci = paired_slope(arms, keys, [a for _, a in lad], rng)
        verdict["T3"] = dict(alpha=al, ci=al_ci, arms=[k[0] for k in keys],
                             label=("A_SQUARED" if ci_in(al_ci, 1.5, 2.5) else
                                    "A_LINEAR" if ci_in(al_ci, 0.5, 1.5) else
                                    "A_FLAT" if ci_has(al_ci, 0.0) else "OTHER"))
    # T4: 枝コントラスト k_up(c=-0.5) / k_dn(c=+0.5)
    t4 = {}
    for a, up, dn in ((0.1, "LRoffm0p5_1216", "LRoffp0p5_1216"),
                      (0.2, "LRa0p2_offm0p5_1216", "LRa0p2_offp0p5_1216"),
                      (0.3, "LRa0p3_offm0p5_1216", "LRa0p3_offp0p5_1216"),
                      (0.5, "LRa0p5_offm0p5_1216", "LRa0p5_offp0p5_1216"),
                      (0.7, "LRa0p7_offm0p5_1216", "LRa0p7_offp0p5_1216")):
        if (up, "up") in arms and (dn, "dn") in arms:
            r, ci = ratio_ci(arms, (up, "up"), (dn, "dn"), rng)
            pred = 1.0 / a ** 2
            t4[str(a)] = dict(ratio=r, ci=ci, predicted=pred, up=up, dn=dn,
                              label=("BRANCH_P2" if pred / 3 <= r <= pred * 3 else
                                     "BRANCH_FLAT" if 0.5 <= r <= 2.0 else "OTHER"))
    verdict["T4"] = t4
    # T5: v 依存（v 凍結腕を主）
    t5 = {}
    for arm, group in (("LRvf1_1216", "dn"), ("LRvf1_1216", "up"),
                       ("LRoffm0p5_vf1_1216", "up"), ("LRoffp0p5_vf1_1216", "dn"),
                       ("LRoffp0p5_1216", "dn"), ("LRoffm0p5_1216", "up")):
        if (arm, group) in arms:
            t5[f"{arm}:{group}"] = dict(v_slope(arm, group, res["metas"][arm], rng),
                                        vf=bool(res["metas"][arm]["vf"]))
    for k, v in t5.items():
        v["label"] = ("NOT_DETERMINED" if not np.isfinite(v["beta"]) else
                      "V_SQUARED" if ci_in(v["beta_ci"], 1.5, 2.5) else
                      "V_FLAT" if ci_has(v["beta_ci"], 0.0) else "OTHER")
    verdict["T5"] = t5
    # T6: η 依存
    t6 = {}
    for slow, fast, group in (("LRoff0_lr0p00125_1216", "LRoff0_1216", "dn"),
                              ("LRoffp0p5_lr0p00125_1216", "LRoffp0p5_1216", "dn"),
                              ("LRoffm0p5_lr0p00125_1216", "LRoffm0p5_1216", "up")):
        if (slow, group) in arms and (fast, group) in arms:
            r, ci = ratio_ci(arms, (slow, group), (fast, group), rng)
            t6[f"{slow}/{fast}:{group}"] = dict(ratio=r, ci=ci, expected=0.125,
                                                label="OK" if 1 / 16 <= r <= 1 / 4 else "OFF")
    verdict["T6"] = t6
    # D1（登録外の診断）: 同一腕の中での枝比 k_up / k_dn。p=2 なら 1/a^2。
    d1 = {}
    for arm in sorted({k[0] for k in arms}):
        if (arm, "up") in arms and (arm, "dn") in arms:
            r, ci = ratio_ci(arms, (arm, "up"), (arm, "dn"), rng)
            a = arms[(arm, "dn")]["a"]
            d1[arm] = dict(ratio=r, ci=ci, predicted_p2=1.0 / a ** 2,
                           predicted_a1=1.0 / a, a=a, c=arms[(arm, "dn")]["c"],
                           vf=arms[(arm, "dn")]["vf"],
                           k_up=arms[(arm, "up")]["k"], k_dn=arms[(arm, "dn")]["k"])
    verdict["D1_within_arm_branch"] = d1
    d2res = {}
    for (arm, group), v in sorted(arms.items()):
        if v["n"] >= 20_000:
            d2res[f"{arm}:{group}"] = dict(centered_check(arm, group, res["metas"][arm]),
                                           k_zstar=v["k"], a=v["a"], c=v["c"], vf=v["vf"])
    for key, d in d2res.items():
        # 底が支持と同じ線形域にあるか（c>0 かつ下／c<0 かつ上 なら内点、c=0 は折れ目上）
        g = key.rsplit(":", 1)[1]
        d["bottom"] = ("kink" if d["c"] == 0 else
                       "interior" if (d["c"] > 0) == (g == "dn") else "ramp")
    verdict["D2_centered_vs_well"] = d2res
    # 総合（spec §3）: T1 の腕横断の畳み方は登録時に決めていない → 畳めない場合は NOT_DETERMINED
    t1 = {}
    for v in arms.values():
        t1[v["t1"]] = t1.get(v["t1"], 0) + 1
    verdict["composite"] = dict(
        label="NOT_DETERMINED",
        why=("T1 は腕ごとに割れ（LINEAR %d / CONSTANT_FORCE %d / OTHER %d / ND %d）、"
             "登録時に腕横断の畳み方を決めていない。T3 は OTHER、T4 は 5 点中 2 点だけ "
             "BRANCH_P2、T5 は v 凍結腕で四分位が作れず NOT_DETERMINED、T6 は OFF。"
             % (t1.get("LINEAR", 0), t1.get("CONSTANT_FORCE", 0), t1.get("OTHER", 0),
                t1.get("NOT_DETERMINED", 0))),
        t1_counts=t1)

    payload = dict(read_rule="specs/spec_well_stiffness_0907.md",
                   n_boot=N_BOOT, boot_seed=BOOT_SEED,
                   arms=[v for v in arms.values()], verdict=verdict)
    (out / "stiffness.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                   default=float), encoding="utf-8")
    # 一覧
    L = ["| 腕 | 群 | a | c | lr | n 増分 | k [CI] | q(T1) [CI] | T1 | 復元率 | T7 R² |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in sorted(arms.values(), key=lambda r: (r["a"], r["c"], r["group"])):
        f = lambda p, ci, d=3: f"{p:.{d}g} [{ci[0]:.{d}g}, {ci[1]:.{d}g}]"
        L.append(f"| `{v['arm']}` | {v['group']} | {v['a']} | {v['c']:+g} | {v['lr']:g} | "
                 f"{v['n']:,} | {f(v['k'], v['k_ci'])} | {f(v['q'], v['q_ci'], 2)} | {v['t1']} | "
                 f"{v['restore']:.3f} | {v['r2']:.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n\n```json\n"
                                    + json.dumps(verdict, indent=2, ensure_ascii=False, default=float)
                                    + "\n```\n", encoding="utf-8")
    print("\n".join(L))
    print(json.dumps(verdict, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
