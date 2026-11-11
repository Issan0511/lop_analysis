#!/usr/bin/env python3
"""stage1_ctrl_1111 の集計と判定（spec specs/spec_stage1_ctrl_1111.md §3〜§6。規則は登録どおり）。
使い方: python analysis/stage1_ctrl_1111/judge.py  → results/stage1_ctrl_1111/judge.md・judge.json
出来ている走だけで集計する（足りない所は「走が無い」と書く）。
"""
import json, sys
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[2]
OUT = W / "results/stage1_ctrl_1111"; RAW = OUT / "raw"
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]
NAME = {"LR": "leaky ReLU", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
SEEDS = list(range(10))
BANDS = [("2〜3", 2, 3), ("4〜8", 4, 8), ("9〜19", 9, 19), ("20〜30", 20, 30)]
DBANDS = [("3〜10", 3, 10), ("21〜30", 21, 30)]
NB = 10000
RNG = np.random.default_rng(20261111)


# ---------- 読み込みと量 ----------
def load_branch(kind, lr, act, seed):
    f = RAW / f"branch_{kind}_lr{lr:g}_{act}_s{seed}.npz"
    return np.load(f) if f.exists() else None


def upt_index(d, key):
    return list(d[key + "|upts"]).index


def measures(d, arm, t):
    """腕 arm の課題 t の量。数えるユニット = 更新 0 で p+ > 0。"""
    k = f"{arm}|t{t:02d}"
    if f"{k}|units" not in d.files: return None
    u = d[f"{k}|units"]; up = list(d[f"{k}|upts"])
    m0, s0, p0 = u[0][0], u[0][1], u[0][2]
    al = p0 > 0
    i200 = up.index(200) if 200 in up else len(up) - 1
    dm = u[i200][0] - m0
    out = dict(n=int(al.sum()), acc=float(d[f"{k}|eval"][-1, 0]),
               closed=float((u[-1][2] == 0).mean()))
    if al.sum() > 0:
        out["D200"] = float((dm[al] < 0).mean())
        out["sink200"] = float(np.median(-dm[al] / np.maximum(s0[al], 1e-12)))
        out["push"] = float(d[f"{k}|push"][0][al].sum())
        out["push_down"] = float((d[f"{k}|push"][0][al] > 0).mean())
    return out


def relu_last(d):
    """ReLU: S の数えるユニットが 3 個以上の最後の課題。"""
    last = 1
    for t in range(2, 31):
        m = measures(d, "S", t)
        if m is None or m["n"] < 3: break
        last = t
    return last


def band_mean(d, arm, lo, hi, key, act, ref="fa"):
    lim = relu_last(d) if act == "R" else 30
    vals = []
    for t in range(lo, min(hi, lim) + 1):
        m = measures(d, arm, t)
        if m is None: continue
        if key == "def":
            r = measures(d, ref, t)
            if r is None: continue
            vals.append(r["acc"] - m["acc"])
        elif key in m:
            vals.append(m[key])
    return float(np.mean(vals)) if vals else np.nan


# ---------- bootstrap ----------
def boot(fn, cols):
    """cols: dict name -> array(seeds)。fn(dict of means) -> 値。seed を再標本化した百分位の 95% 区間。"""
    cols = {k: np.asarray(v, float) for k, v in cols.items()}
    ok = np.all(np.stack([np.isfinite(v) for v in cols.values()]), axis=0)
    cols = {k: v[ok] for k, v in cols.items()}
    n = int(ok.sum())
    if n < 3: return np.nan, (np.nan, np.nan), n
    pt = fn({k: v.mean() for k, v in cols.items()})
    idx = RNG.integers(0, n, size=(NB, n))
    bs = np.array([fn({k: v[ix].mean() for k, v in cols.items()}) for ix in idx])
    bs = bs[np.isfinite(bs)]
    return float(pt), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), n


def ratio(a, b, c):         # (a - b)/(a - c)
    return lambda m: (m[a] - m[b]) / (m[a] - m[c]) if (m[a] - m[c]) != 0 else np.nan


def label3(lo, hi, low_lab, high_lab, mid="MIXED", thr=0.5):
    if not np.isfinite(lo): return "走が無い"
    if hi < thr: return low_lab
    if lo > thr: return high_lab
    return mid


# ---------- B1・B2・B3 の判定 ----------
def judge_branch(kind, lr_of, arms_present):
    res = {}
    for act in ACTS:
        lr = lr_of(act)
        ds = {s: load_branch(kind, lr, act, s) for s in SEEDS}
        ds = {s: d for s, d in ds.items() if d is not None}
        if not ds: continue
        R = res[act] = {"n_seeds": len(ds), "lr": lr, "bands": {}, "dbands": {}}
        for bn, lo, hi in BANDS:
            col = lambda arm, key: [band_mean(ds[s], arm, lo, hi, key, act) if s in ds else np.nan for s in SEEDS]
            B = R["bands"][bn] = {}
            for Y in ("D200", "sink200"):
                c = {arm: col(arm, Y) for arm in ["S"] + arms_present}
                B[Y] = {arm: float(np.nanmean(c[arm])) for arm in c}
                pt, ci, n = boot(lambda m: m["S"] - m["H0"], {"S": c["S"], "H0": c["H0"]}) if "H0" in c else (np.nan, (np.nan, np.nan), 0)
                B[Y + "_S_minus_H0"] = dict(pt=pt, ci=ci)
                nfin = int(np.sum(np.isfinite(np.asarray(c["S"], float)) & np.isfinite(np.asarray(c.get("H0", c["S"]), float))))
                j0 = ("データ不足（seed < 3 か数えるユニットが無い）" if nfin < 3 else
                      "READOUT_EFFECT" if np.isfinite(ci[0]) and ci[0] > 0 else "NO_READOUT_EFFECT")
                nulls = {}
                for nl in ("P0", "P0m"):
                    if nl in c:
                        pt_, ci_, _ = boot(ratio("S", nl, "H0"), {"S": c["S"], nl: c[nl], "H0": c["H0"]})
                        nulls[nl] = dict(pt=pt_, ci=ci_, contains0=bool(ci_[0] <= 0 <= ci_[1]) if np.isfinite(ci_[0]) else None)
                B[Y + "_null"] = nulls
                null_fail = any(v["contains0"] is False for v in nulls.values())
                for arm, name in (("P", "align"), ("Q", "norm"), ("R", "R"), ("F200", "F200"), ("A0", "A0"), ("H0A", "H0A")):
                    if arm not in c: continue
                    pt_, ci_, _ = boot(ratio("S", arm, "H0"), {"S": c["S"], arm: c[arm], "H0": c["H0"]})
                    if name == "align":
                        lab = (j0 if j0 != "READOUT_EFFECT" else
                               "INCONCLUSIVE(NULL_FAILS)" if null_fail else
                               label3(*ci_, "ALIGN_NOT_NEEDED", "ALIGN_NEEDED"))
                    elif name == "norm":
                        lab = j0 if j0 != "READOUT_EFFECT" else label3(*ci_, "NORM_NOT_NEEDED", "NORM_NEEDED")
                    else:
                        lab = None
                    B[f"{Y}_rho_{name}"] = dict(pt=pt_, ci=ci_, label=lab)
                B[Y + "_J0"] = j0
            if "A0" in arms_present:
                c = {"S": col("S", "D200"), "A0": col("A0", "D200")}
                pt_, ci_, _ = boot(lambda m: m["A0"] - m["S"], c)
                B["J3a"] = dict(pt=pt_, ci=ci_, label=("走が無い" if not np.isfinite(ci_[0]) else "SINK_SAME" if ci_[0] <= 0 <= ci_[1]
                                                     else "SINK_MORE" if ci_[0] > 0 else "SINK_LESS"))
            B["push"] = {arm: float(np.nanmean(col(arm, "push"))) for arm in ["S"] + arms_present}
            B["push_down"] = {arm: float(np.nanmean(col(arm, "push_down"))) for arm in ["S"] + arms_present}
            B["closed"] = {arm: float(np.nanmean(col(arm, "closed"))) for arm in ["S"] + arms_present}
        for bn, lo, hi in DBANDS:
            col = lambda arm: [band_mean(ds[s], arm, lo, hi, "def", act) if s in ds else np.nan for s in SEEDS]
            c = {arm: col(arm) for arm in ["S"] + arms_present}
            DB = R["dbands"][bn] = {"def": {arm: float(np.nanmean(c[arm])) for arm in c}}
            pt, ci, _ = boot(lambda m: m["S"], {"S": c["S"]})
            DB["def_S"] = dict(pt=pt, ci=ci)
            for arm in arms_present:
                pt_, ci_, _ = boot(lambda m, a=arm: (m["S"] - m[a]) / m["S"] if m["S"] != 0 else np.nan, {"S": c["S"], arm: c[arm]})
                DB[f"rem_{arm}"] = dict(pt=pt_, ci=ci_)
            if "A0" in arms_present and bn == "21〜30":
                r = DB["rem_A0"]
                DB["J3b"] = ("データ不足" if not np.isfinite(ci[0]) else "NO_LOSS_IN_S" if not ci[0] > 0 else
                             label3(*r["ci"], "LATE_LOSS_NOT_ADAM", "LATE_LOSS_MOSTLY_ADAM"))
            if kind != "adam" and bn == "21〜30":
                # 登録の規則は二つの条件が同時に成り立ちうる（下端 > 0 かつ上端 < 0.02）。そのときは両方を書く（集計の時点の追記・spec §8）
                pres = np.isfinite(ci[0]) and ci[0] > 0; absn = np.isfinite(ci[1]) and ci[1] < 0.02
                DB["J4_loss"] = ("LOSS_PRESENT かつ LOSS_ABSENT の条件（0 より大きいが 0.02 未満）" if pres and absn else
                                 "LOSS_PRESENT" if pres else "LOSS_ABSENT" if absn else "UNRESOLVED")
        if kind != "adam":
            for bn in ("2〜3", "4〜8"):
                x = R["bands"][bn]["D200_S_minus_H0"]
                R["bands"][bn]["J4_sink"] = ("SINK_PRESENT" if np.isfinite(x["ci"][0]) and x["ci"][0] > 0 else
                                             "SINK_ABSENT（読み出しを 0 にした腕も同じだけ沈む）" if np.isfinite(x["ci"][1]) and x["ci"][1] < 0.05 else "UNRESOLVED")
    return res


# ---------- T（毎回の介入）----------
def judge_persist():
    res = {}
    for act in ACTS:
        R = {}
        for arm in ("H0", "P", "A0", "R", "Q", "F200"):
            dX, dS, dB = [], [], []
            sX, sS = [], []
            for s in SEEDS:
                fp = RAW / f"persist_{arm}_adam_lr0.001_{act}_s{s}.npz"
                b1 = load_branch("adam", 1e-3, act, s)
                if not fp.exists() or b1 is None: dX.append(np.nan); dS.append(np.nan); sX.append(np.nan); sS.append(np.nan); continue
                p = np.load(fp)
                fa = [measures(b1, "fa", t)["acc"] for t in range(21, 31)]
                accX = [float(p[f"{arm}all|t{t:02d}|eval"][-1, 0]) for t in range(21, 31)]
                accS = [measures(b1, "S", t)["acc"] for t in range(21, 31)]
                dX.append(float(np.mean(np.array(fa) - np.array(accX)))); dS.append(float(np.mean(np.array(fa) - np.array(accS))))
                # 前活性の平均の積もり: 課題 2 の頭の数えるユニットの m の、課題 2 の頭 → 課題 30 の末の変化の中央値
                u2S = b1["S|t02|units"]; u30S = b1["S|t30|units"]
                u2X = p[f"{arm}all|t02|units"]; u30X = p[f"{arm}all|t30|units"]
                alS, alX = u2S[0][2] > 0, u2X[0][2] > 0
                sS.append(float(np.median((u30S[-1][0] - u2S[0][0])[alS]))); sX.append(float(np.median((u30X[-1][0] - u2X[0][0])[alX])))
            pt, ci, n = boot(lambda m: m["X"] / m["S"] if m["S"] != 0 else np.nan, {"X": dX, "S": dS})
            lab = label3(*ci, "LOSS_REMOVED", "LOSS_KEPT", "PARTIAL")
            pt2, ci2, _ = boot(lambda m: m["X"] / m["S"] if m["S"] != 0 else np.nan, {"X": sX, "S": sS})
            lab2 = label3(*ci2, "SINK_REMOVED", "SINK_KEPT", "PARTIAL")
            R[arm] = dict(n=n, def_X=float(np.nanmean(dX)), def_S=float(np.nanmean(dS)), ratio=dict(pt=pt, ci=ci, label=lab),
                          dm_X=float(np.nanmean(sX)), dm_S=float(np.nanmean(sS)), sink_ratio=dict(pt=pt2, ci=ci2, label=lab2))
        res[act] = R
    return res


# ---------- D（課題 1 の途中）----------
def judge_dial():
    res = {}
    for act in ACTS:
        R = {}
        for tag in ("init", "a20", "a50", "a80", "a95", "end"):
            pr, dd, ds_ = [], [], []
            for s in (0, 1):
                f = RAW / f"dial_{act}_s{s}.npz"
                if not f.exists(): continue
                d = np.load(f)
                for k in range(5):
                    def q(arm):
                        u = d[f"{arm}|{tag}|d{k}|units"]; up = list(d[f"{arm}|{tag}|d{k}|upts"])
                        al = u[0][2] > 0; dm = u[up.index(200)][0] - u[0][0]
                        return float(d[f"{arm}|{tag}|d{k}|push"][0][al].sum()), float((dm[al] < 0).mean())
                    pS, dS_ = q("S"); pP, dP = q("P")
                    pr.append(pP / pS if pS != 0 else np.nan); dd.append(dP - dS_); ds_.append(dS_)
            if pr:
                R[tag] = dict(push_ratio=float(np.nanmean(pr)), D200_P_minus_S=float(np.mean(dd)), D200_S=float(np.mean(ds_)), n=len(pr))
        res[act] = R
    return res


# ---------- 書き出し ----------
def fmt(x, ci=None, p=2):
    if x is None or not np.isfinite(x): return "—"
    s = f"{x:.{p}f}"
    if ci is not None and np.isfinite(ci[0]): s += f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"
    return s


def main():
    J = {}
    J["B1"] = judge_branch("adam", lambda a: 1e-3, ["P0", "P0m", "P", "Q", "R", "H0", "H0A", "A0", "F200"])
    sel = json.loads((OUT / "sgd_lr.json").read_text()) if (OUT / "sgd_lr.json").exists() else None
    if sel: J["B2"] = judge_branch("sgd", lambda a: sel[a]["lr"], ["P0", "P0m", "P", "Q", "R", "H0", "F200"])
    J["B3"] = judge_branch("adaml", lambda a: 1e-3, ["P", "H0", "A0"])
    J["T"] = judge_persist()
    J["D"] = judge_dial()
    (OUT / "judge.json").write_text(json.dumps(J, indent=1, ensure_ascii=False, default=float))
    L = ["# stage1_ctrl_1111 の判定（spec §5 の規則）", ""]
    for key, title in (("B1", "B1 Adam"), ("B2", "B2 SGD"), ("B3", "B3 Lyle の Adam")):
        if key not in J or not J[key]: continue
        L += [f"## {title}", "", "| 活性化 | 課題 | seed | D200: S・P0・P・Q・R・H0 | ρ_align（D200） | J1 | ρ_norm（D200） | J2 | ρ_null P0・P0m | J3(a) A0−S |", "|---" * 10 + "|"]
        for act, R in J[key].items():
            for bn, B in R["bands"].items():
                D = B["D200"]
                vals = "・".join(fmt(D.get(a)) for a in ("S", "P0", "P", "Q", "R", "H0"))
                ra = B.get("D200_rho_align", {}); rn = B.get("D200_rho_norm", {})
                nl = B.get("D200_null", {})
                L.append(f"| {NAME[act]} | {bn} | {R['n_seeds']} | {vals} | {fmt(ra.get('pt'), ra.get('ci'))} | {ra.get('label', '—')} | "
                         f"{fmt(rn.get('pt'), rn.get('ci'))} | {rn.get('label', '—')} | "
                         f"{fmt(nl.get('P0', {}).get('pt'), nl.get('P0', {}).get('ci'))}・{fmt(nl.get('P0m', {}).get('pt'), nl.get('P0m', {}).get('ci'))} | "
                         f"{fmt(B.get('J3a', {}).get('pt'), B.get('J3a', {}).get('ci'))} {B.get('J3a', {}).get('label', '')} |")
        L += ["", "| 活性化 | 課題 | 不足 def: S・P・Q・R・H0・H0A・A0・F200 | rem_P | rem_H0 | rem_A0 | 判定 |", "|---" * 7 + "|"]
        for act, R in J[key].items():
            for bn, DB in R["dbands"].items():
                dd = DB["def"]
                L.append(f"| {NAME[act]} | {bn} | " + "・".join(fmt(dd.get(a)) for a in ("S", "P", "Q", "R", "H0", "H0A", "A0", "F200")) +
                         f" | {fmt(DB.get('rem_P', {}).get('pt'), DB.get('rem_P', {}).get('ci'))} | {fmt(DB.get('rem_H0', {}).get('pt'), DB.get('rem_H0', {}).get('ci'))} | "
                         f"{fmt(DB.get('rem_A0', {}).get('pt'), DB.get('rem_A0', {}).get('ci'))} | {DB.get('J3b', '')} {DB.get('J4_loss', '')} |")
        if key != "B1":
            L += ["", "J4（沈み）: " + "; ".join(f"{NAME[a]} " + "・".join(f"{bn} {R['bands'][bn].get('J4_sink', '—')}" for bn in ('2〜3', '4〜8'))
                                              for a, R in J[key].items())]
        L.append("")
    if J["T"]:
        L += ["## T 毎回の介入（課題 21〜30 の不足の比・前活性の平均の積もりの比）", "", "| 活性化 | 腕 | 不足 X・S | 比 | J5 | Δm X・S | 比 | 判定 |", "|---" * 8 + "|"]
        for act, R in J["T"].items():
            for arm, x in R.items():
                L.append(f"| {NAME[act]} | {arm}all | {fmt(x['def_X'])}・{fmt(x['def_S'])} | {fmt(x['ratio']['pt'], x['ratio']['ci'])} | {x['ratio']['label']} | "
                         f"{fmt(x['dm_X'])}・{fmt(x['dm_S'])} | {fmt(x['sink_ratio']['pt'], x['sink_ratio']['ci'])} | {x['sink_ratio']['label']} |")
        L.append("")
    if J["D"]:
        L += ["## D 課題 1 の途中からの切替（5 draw × 2 seed）", "", "| 活性化 | checkpoint | Σpush_P/Σpush_S | D200_S | D200_P − D200_S |", "|---" * 5 + "|"]
        for act, R in J["D"].items():
            for tag, x in R.items():
                L.append(f"| {NAME[act]} | {tag} | {fmt(x['push_ratio'])} | {fmt(x['D200_S'])} | {fmt(x['D200_P_minus_S'])} |")
    (OUT / "judge.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
