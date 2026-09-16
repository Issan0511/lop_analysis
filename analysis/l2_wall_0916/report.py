#!/usr/bin/env python3
"""Aggregates, labels and prediction checks for l2_wall_0916 (spec §5, §6).
Usage (repo root):  python3 analysis/l2_wall_0916/report.py [--src results/l2_wall_0916] [--parts chimera,ext150,variant]
Reads only what the runner wrote under --src; writes events.csv, windows.csv, verdict.csv, predictions.csv, report.md there."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(src, tag):
    d = np.load(src / "logs" / f"diag_{tag}.npz")
    l = np.load(src / "logs" / f"ledger_{tag}.npz")
    w = np.load(src / "logs" / f"w2_{tag}.npz")
    prov = json.loads((src / f"provenance_{tag}.json").read_text())
    with open(src / f"taskend_{tag}.csv", newline="") as f:
        te = list(csv.DictReader(f))
    return dict(d=d, l=l, w=w, models=prov["models"], te=te, prov=prov)


def med(a, axis=-1):
    with np.errstate(all="ignore"):
        return np.nanmedian(a, axis=axis)


def label_of(m):
    base = f"{m['env']}_{m.get('act', m.get('act1', '') + '>' + m.get('act2', ''))}"
    if "arm" in m:
        base += f"_{m['arm']}"
    return base + f"_s{m['seed']}"


def floors(te, models, late):
    """parent's rule: late-window online mean <= F + 3 s / sqrt(10), F and s from the late window's floor_acc."""
    out = {}
    by = {}
    for r in te:
        by.setdefault(int(r["_j"]), []).append(r)
    for j, m in enumerate(models):
        rr = sorted(by[j], key=lambda r: int(r["task"]))
        lw = [r for r in rr if late[0] <= int(r["task"]) <= late[1]]
        fl = np.array([float(r["floor_acc"]) for r in lw])
        thr = fl.mean() + 3 * fl.std(ddof=1) / math.sqrt(len(fl))
        la = float(np.mean([float(r["online_acc"]) for r in lw]))
        first = next((int(r["task"]) for r in rr if float(r["online_acc"]) <= thr), None)
        out[j] = dict(late_acc=la, thr=thr, at_floor=la <= thr, first_floor_task=first)
    return out


def attach_index(te, models, keys):
    idx = {tuple(str(m[k]) for k in keys): j for j, m in enumerate(models)}
    for r in te:
        r["_j"] = idx[tuple(str(r[k]) for k in keys)]


def point_index(d):
    return list(zip(d["task"].tolist(), d["step"].tolist()))


def end_point(pts, task):
    steps = [s for t, s in pts if t == task]
    return pts.index((task, max(steps)))


def window_sums(L, pts, i0p, i1p):
    """sum of every per-unit ledger field over the intervals between point i0p and point i1p."""
    i0 = L["i0"]
    sel = (i0 >= i0p) & (L["i1"] <= i1p)
    return {k[2:]: L[k][sel].sum(0) for k in L.files if k.startswith("l_")}


def events_for(D, j, pts, thr_eps=0.5):
    d = D["d"]
    rho = d["u_rho_l2"][:, j]
    sr = med(np.sqrt(np.clip(rho, 0, None)))
    cone = d["m_in2_k"][:, j] / d["m_in2_c"][:, j]
    dmed = med(d["u_d_l2"][:, j])
    c2 = d["m_in2_c"][:, j]
    eps = med(d["u_adam_epsfrac_l2"][:, j])
    gate = np.nanmean(d["u_gate_mean_l2"][:, j], -1)
    absorbed = np.mean(d["u_hU_l2"][:, j] < 0, -1)
    first = lambda mask: next((i for i, v in enumerate(mask) if v), None)
    ev = dict(cone=first(sr >= cone), pin=first(dmed <= -0.9 * c2), eps=first(eps > thr_eps),
              gate=first(gate < 0.05), absorbed=first(absorbed > 0.5))
    fmt = lambda i: None if i is None else f"t{pts[i][0]}+{pts[i][1]}"
    return ev, {k: fmt(v) for k, v in ev.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results/l2_wall_0916"))
    ap.add_argument("--parts", default="chimera,ext150,variant")
    a = ap.parse_args()
    src = Path(a.src)
    parts = a.parts.split(",")
    P = {}
    for tag in parts:
        if (src / f"provenance_{tag}.json").exists():
            P[tag] = load(src, tag)
    keys = dict(chimera=("seed", "env", "act1", "act2"), ext150=("seed", "env", "act"),
                variant=("seed", "env", "act1", "act2", "arm", "eps"))
    for tag, D in P.items():
        attach_index(D["te"], D["models"], keys[tag])
        late = (141, 150) if tag == "ext150" else (41, 50)
        D["floor"] = floors(D["te"], D["models"], late)
        D["pts"] = point_index(D["d"])
    events, windows, verdict, preds = [], [], [], []

    # ------------------------------------------------ events and windows for every model
    for tag, D in P.items():
        pts, d, L = D["pts"], D["d"], D["l"]
        T = max(t for t, _ in pts)
        wins = ([(0, 5), (5, 10), (10, 20), (20, 50), (0, 10), (0, 20), (0, 50)] if T == 50
                else [(0, 20), (20, 50), (50, 100), (100, 150), (0, 150)])
        wins = [w for w in wins if w[1] <= T]
        for j, m in enumerate(D["models"]):
            ev, evs = events_for(D, j, pts)
            fl = D["floor"][j]
            events.append(dict(part=tag, model=label_of(m), **{f"first_{k}": v for k, v in evs.items()},
                               first_floor_task=fl["first_floor_task"], late_acc=fl["late_acc"], floor_thr=fl["thr"],
                               at_floor=fl["at_floor"]))
            ends = {0: 0, **{t: end_point(pts, t) for t in range(1, T + 1)}}
            extra = []
            if ev["eps"] is not None and ev["eps"] < ends[T]:
                extra.append(("eps", ev["eps"], ends[T]))
            for (wa, wb) in wins:
                extra.append((f"t{wa}-t{wb}", ends[wa], ends[wb]))
            for name, i0p, i1p in extra:
                S = window_sums(L, pts, i0p, i1p)
                row = dict(part=tag, model=label_of(m), window=name if name != "eps" else f"{evs['eps']}-t{T}")
                for k in ("meas", "self", "up", "cross", "bias", "stretch", "rot", "beta",
                          "meas1", "self1", "up1", "cross1", "bias1"):
                    row[f"mean_{k}"] = float(np.nanmean(S[k][j]))
                    row[f"med_{k}"] = float(med(S[k][j]))
                for fct in ("d", "rho"):
                    tot = S[f"d_{fct}"][j]
                    row[f"med_d_{fct}"] = float(med(tot))
                    absval = {g: np.abs(S[f"sh_{fct}_{g}"][j]) for g in ("W", "mu", "S")}
                    den = absval["W"] + absval["mu"] + absval["S"]
                    for g in ("W", "mu", "S"):
                        row[f"sh_{fct}_{g}_med"] = float(med(S[f"sh_{fct}_{g}"][j]))
                        row[f"sh_{fct}_{g}_absmed"] = float(med(absval[g]))
                        with np.errstate(all="ignore"):
                            row[f"sh_{fct}_{g}_share"] = float(med(absval[g] / den))
                row["sh_sig_W_med"] = float(med(S["sh_sig_W"][j]))
                row["sh_sig_S_med"] = float(med(S["sh_sig_S"][j]))
                row["med_absSb_minus_absSmu"] = float(med(np.abs(S["bias"][j]) - np.abs(S["self"][j] + S["up"][j] + S["cross"][j])))
                windows.append(row)
            D.setdefault("ev", {})[j] = ev

    W = {(r["part"], r["model"], r["window"]): r for r in windows}

    def pick(tag, **kw):
        return [j for j, m in enumerate(P[tag]["models"]) if all(m.get(k) == v for k, v in kw.items())]

    def endval(tag, j, field, task, fn=med):
        D = P[tag]
        i = end_point(D["pts"], task)
        return float(fn(D["d"][field][i, j]))

    def late_mean(tag, j, fn, t0=41, t1=50):
        D = P[tag]
        return float(np.mean([fn(D, end_point(D["pts"], t), j) for t in range(t0, t1 + 1)]))

    def add_pred(pid, owner, text, hit, detail):
        preds.append(dict(id=pid, owner=owner, prediction=text, outcome=("NA" if hit is None else ("HIT" if hit else "MISS")), detail=detail))

    dn = lambda D, i, j: float(med(D["d"]["u_d_l2"][i, j])) / float(D["d"]["m_in2_c"][i, j])
    dz = lambda D, i, j: float(med(D["d"]["u_d_l2"][i, j]))
    cc = lambda D, i, j: float(D["d"]["m_in2_c"][i, j])

    # ------------------------------------------------ Part A predictions
    if "chimera" in P:
        D = P["chimera"]
        ee = pick("chimera", env="RL", act1="ELU1", act2="ELU1")
        res = []
        for j in ee:
            ev = D["ev"][j]
            ok = ev["cone"] is not None and (ev["eps"] is None or ev["cone"] <= ev["eps"])
            res.append((ok, ev["cone"], ev["eps"]))
        add_pred("P-A1", "Claude", "RL EE: cone entry no later than W2 eps regime (3/3)", all(r[0] for r in res),
                 "; ".join(f"cone={D['pts'][c] if c is not None else None} eps={D['pts'][e] if e is not None else None}" for _, c, e in res))
        res = []
        for j in ee:
            r = W[("chimera", label_of(D["models"][j]), "t0-t20")]
            vals = {g: r[f"sh_d_{g}_absmed"] for g in ("W", "mu", "S")}
            res.append((max(vals, key=vals.get) == "W", vals))
        add_pred("P-A2", "Claude", "RL EE t0-t20: W2 has the largest Shapley share of Δd2 (3/3)", all(x[0] for x in res),
                 "; ".join(json.dumps({k: round(v, 3) for k, v in x[1].items()}) for x in res))
        res = []
        for j in ee:
            r = W[("chimera", label_of(D["models"][j]), "t0-t10")]
            ok = abs(r["mean_self"]) > abs(r["mean_up"]) and abs(r["mean_cross"]) < 0.1 * abs(r["mean_meas"])
            res.append((ok, r["mean_meas"], r["mean_self"], r["mean_up"], r["mean_cross"], r["mean_bias"]))
        add_pred("P-A3", "Claude", "RL EE t0-t10: |self|>|up| and |cross|<0.1|Δz̄2| (3/3)", all(x[0] for x in res),
                 "; ".join("meas %.2f self %.2f up %.2f cross %.2f bias %.2f" % x[1:] for x in res))
        res = []
        for j in ee:
            ev = D["ev"][j]
            if ev["eps"] is None:
                res.append((None, None))
                continue
            r = next((w for w in windows if w["part"] == "chimera" and w["model"] == label_of(D["models"][j]) and w["window"].endswith("-t50") and w["window"].startswith("t") and "+" in w["window"]), None)
            res.append((abs(r["mean_up"]) > abs(r["mean_self"]), r))
        hit = None if any(x[0] is None for x in res) else all(x[0] for x in res)
        add_pred("P-A4", "Claude", "RL EE after eps regime to t50: |up|>|self| (3/3)", hit,
                 "; ".join("NA" if x[1] is None else "%s self %.2f up %.2f cross %.2f" % (x[1]["window"], x[1]["mean_self"], x[1]["mean_up"], x[1]["mean_cross"]) for x in res))
        le = pick("chimera", env="RL", act1="LR", act2="ELU1")
        res = []
        for j in le:
            i = end_point(D["pts"], 50)
            pmin = float(D["d"]["m_in2_pmin"][i, j]); c2 = cc(D, i, j); d2 = dz(D, i, j)
            res.append((pmin > 0 and abs(d2 + c2) < 0.2, pmin, c2, float(D["d"]["m_in2_k"][i, j]), d2))
        add_pred("P-A5", "Claude", "RL LE t50: min e2'a1>0 and |d2+c2|<0.2 (3/3)", all(x[0] for x in res),
                 "; ".join("pmin %.2f c2 %.2f k2 %.2f d2 %.2f" % x[1:] for x in res))
    if "ext150" in P:
        D = P["ext150"]
        def pinned(j, t=150):
            i = end_point(D["pts"], t)
            rho = float(med(D["d"]["u_rho_l2"][i, j]))
            bs = float(med(np.abs(D["d"]["u_b_l2"][i, j]) / D["d"]["u_sig_l2"][i, j]))
            return rho >= 0.8 and bs <= 0.3, rho, bs, dz(D, i, j), cc(D, i, j), float(D["d"]["m_in2_k"][i, j])
        res = [pinned(j) for j in pick("ext150", env="PM", act="GELU")]
        add_pred("P-A6", "Claude", "ext150 PM GELU t150: median rho>=0.8 and median |b|/sigma<=0.3 (3/3)", all(x[0] for x in res),
                 "; ".join("rho %.2f |b|/s %.2f d2 %.2f c2 %.2f k2 %.2f" % x[1:] for x in res))
        res = [pinned(j) for j in pick("ext150", env="RL", act="SILU") if P["ext150"]["models"][j]["seed"] in (0, 1)]
        add_pred("P-A7", "Claude", "ext150 RL SILU s0,s1 t150: same pin criterion (2/2)", all(x[0] for x in res),
                 "; ".join("rho %.2f |b|/s %.2f d2 %.2f c2 %.2f k2 %.2f" % x[1:] for x in res))

    # ------------------------------------------------ Part B
    if "variant" in P:
        D = P["variant"]
        pts = D["pts"]
        rowsB = []
        for a1, a2, nm in (("ELU1", "ELU1", "EE"), ("LR", "ELU1", "LE")):
            ctrl = {D["models"][j]["seed"]: j for j in pick("variant", env="RL", act1=a1, act2=a2, arm="none")}
            sds = []
            for s, j in ctrl.items():
                v = [dn(D, end_point(pts, t), j) for t in range(41, 51)]
                sds.append(float(np.std(v, ddof=1)))
            delta = 3 * max(sds)
            dd = {}
            for arm in ("eps1e-6", "eps1e-30"):
                for j in pick("variant", env="RL", act1=a1, act2=a2, arm=arm):
                    s = D["models"][j]["seed"]
                    Dm = late_mean("variant", j, dn)
                    D0 = late_mean("variant", ctrl[s], dn)
                    z50 = abs(endval("variant", j, "u_zbar_l2", 50))
                    z50c = abs(endval("variant", ctrl[s], "u_zbar_l2", 50))
                    dd[arm, s] = Dm - D0
                    rowsB.append(dict(pair=nm, arm=arm, seed=s, D=Dm, D_none=D0, dD=Dm - D0, delta=delta,
                                      absz50=z50, absz50_none=z50c, at_floor=D["floor"][j]["at_floor"],
                                      late_acc=D["floor"][j]["late_acc"], first_floor=D["floor"][j]["first_floor_task"],
                                      eps_first=None if D["ev"][j]["eps"] is None else f"t{pts[D['ev'][j]['eps']][0]}+{pts[D['ev'][j]['eps']][1]}",
                                      eps_first_none=None if D["ev"][ctrl[s]]["eps"] is None else f"t{pts[D['ev'][ctrl[s]]['eps']][0]}+{pts[D['ev'][ctrl[s]]['eps']][1]}"))
            small = all(abs(v) <= delta for v in dd.values())
            big = all(abs(v) > delta for v in dd.values()) and all(np.sign(dd["eps1e-6", s]) != np.sign(dd["eps1e-30", s]) for s in range(3))
            lab = "PIN" if small else ("BRAKE" if big else "MIXED")
            verdict.append(dict(section=f"B_{nm}", verdict=lab, detail=f"delta={delta:.4f} " + " ".join(f"{k[0]}/s{k[1]}:{v:+.4f}" for k, v in dd.items())))
            hit = {"EE": lab == "PIN", "LE": lab == "PIN"}[nm]
            add_pred(f"P-B-{nm}", "Claude", f"Part B {nm} = PIN", hit, verdict[-1]["detail"])
            r1 = [r for r in rowsB if r["pair"] == nm and r["arm"] == "eps1e-30"]
            r2 = [r for r in rowsB if r["pair"] == nm and r["arm"] == "eps1e-6"]
            add_pred(f"P-B1-{nm}", "Claude", f"{nm}: eps1e-30 deeper |z̄2(t50)| than none (3/3)", all(r["absz50"] > r["absz50_none"] for r in r1),
                     "; ".join("s%d %.1f vs %.1f" % (r["seed"], r["absz50"], r["absz50_none"]) for r in r1))
            add_pred(f"P-B2-{nm}", "Claude", f"{nm}: eps1e-6 shallower |z̄2(t50)| than none (3/3)", all(r["absz50"] < r["absz50_none"] for r in r2),
                     "; ".join("s%d %.1f vs %.1f" % (r["seed"], r["absz50"], r["absz50_none"]) for r in r2))
        add_pred("P-B3", "Claude", "all eps arms at floor (12/12)", all(r["at_floor"] for r in rowsB),
                 f"{sum(r['at_floor'] for r in rowsB)}/{len(rowsB)}")
        # ------------------------------------------------ Part C
        def floor_label(js):
            fl = [D["floor"][j]["at_floor"] for j in js]
            return "COLLAPSED" if all(fl) else ("RESCUED" if not any(fl) else "SPLIT")
        rowsC = []
        for j in [j for j, m in enumerate(D["models"]) if m["eps"] == 1e-8]:
            m = D["models"][j]
            i50 = end_point(pts, 50)
            s = m["seed"]
            ctrl = next(k for k, mm in enumerate(D["models"]) if mm["arm"] == "none" and mm["eps"] == 1e-8 and all(mm[q] == m[q] for q in ("seed", "env", "act1", "act2")))
            wr = W[("variant", label_of(m), "t0-t50")]
            rowsC.append(dict(env=m["env"], pair=m["act1"] + ">" + m["act2"], arm=m["arm"], seed=s,
                              late_acc=D["floor"][j]["late_acc"], floor_thr=D["floor"][j]["thr"], at_floor=D["floor"][j]["at_floor"],
                              first_floor=D["floor"][j]["first_floor_task"],
                              d2_late=late_mean("variant", j, dz), c2_late=late_mean("variant", j, cc),
                              c2_none_late=late_mean("variant", ctrl, cc), d2_none_late=late_mean("variant", ctrl, dz),
                              rho_t50=float(med(D["d"]["u_rho_l2"][i50, j])), k2_t50=float(D["d"]["m_in2_k"][i50, j]),
                              c2_t50=float(D["d"]["m_in2_c"][i50, j]), pmin_t50=float(D["d"]["m_in2_pmin"][i50, j]),
                              cos1_t50=float(D["d"]["m_in2_cos1"][i50, j]), r2_t50=float(D["d"]["m_in2_r"][i50, j]),
                              hbar_t50=float(med(D["d"]["u_hbar_l2"][i50, j])), zbar_t50=float(med(D["d"]["u_zbar_l2"][i50, j])),
                              gate_t50=float(np.nanmean(D["d"]["u_gate_mean_l2"][i50, j])),
                              absorbed_t50=float(np.mean(D["d"]["u_hU_l2"][i50, j] < 0)),
                              h2pos_zero_t50=float(D["d"]["m_h2pos_zero"][i50, j]), h2pos_min_t50=float(D["d"]["m_h2pos_min"][i50, j]),
                              beta_med_t50=float(np.median(D["w"]["bp2"][-1][j])), gamma_med_t50=float(np.median(D["w"]["gp2"][-1][j])),
                              betain_med_t50=float(np.median(D["w"]["bi"][-1][j])),
                              Wbeta_med_t50=float(med(D["d"]["u_beta_l2"][i50, j])),
                              led_meas=wr["mean_meas"], led_self=wr["mean_self"], led_up=wr["mean_up"], led_cross=wr["mean_cross"],
                              led_bias=wr["mean_bias"], led_beta=wr["mean_beta"], absSb_minus_absSmu=wr["med_absSb_minus_absSmu"]))
        C = {(r["env"], r["pair"], r["arm"], r["seed"]): r for r in rowsC}
        # C1
        labs = []
        for s in range(3):
            r = C["RL", "ELU1>ELU1", "inLN", s]
            labs.append("RESCUED" if not r["at_floor"] else ("B_PATH" if r["absSb_minus_absSmu"] > 0 else "MU_PATH"))
        c1 = labs[0] if len(set(labs)) == 1 else "SPLIT"
        verdict.append(dict(section="C1_RL_EE_inLN", verdict=c1, detail=" ".join(labs)))
        c2 = floor_label(pick("variant", env="RL", act1="LR", act2="ELU1", arm="inLN"))
        verdict.append(dict(section="C2_RL_LE_inLN", verdict=c2, detail=""))
        c3 = floor_label(pick("variant", env="RL", act1="ELU1", act2="ELU1", arm="preLN"))
        verdict.append(dict(section="C3_RL_EE_preLN", verdict=c3, detail=""))
        labs = []
        for s in range(3):
            r = C["RL", "ELU1>ELU1", "preLN_a", s]
            labs.append("RESCUED" if not r["at_floor"] else ("COLLAPSED_VIA_BETA" if (r["beta_med_t50"] < 0 and r["hbar_t50"] < 0) else "COLLAPSED_OTHER"))
        c4 = labs[0] if len(set(labs)) == 1 else "SPLIT"
        verdict.append(dict(section="C4_RL_EE_preLN_a", verdict=c4, detail=" ".join(labs)))
        add_pred("P-C1", "Claude", "C1 = MU_PATH", c1 == "MU_PATH", c1)
        add_pred("P-C1-other", "pasted reading (via Issa)", "C1 = B_PATH", c1 == "B_PATH", c1)
        res = [(C["RL", "ELU1>ELU1", "inLN", s]["d2_late"] < -C["RL", "ELU1>ELU1", "inLN", s]["c2_none_late"],
                C["RL", "ELU1>ELU1", "inLN", s]["d2_late"], C["RL", "ELU1>ELU1", "inLN", s]["c2_none_late"]) for s in range(3)]
        add_pred("P-C1b", "Claude", "RL EE inLN: late d2 deeper than -c2(none) (3/3)", all(x[0] for x in res),
                 "; ".join("d2 %.2f vs -%.2f" % x[1:] for x in res))
        add_pred("P-C2", "Claude", "C2 = RESCUED", c2 == "RESCUED", c2)
        add_pred("P-C2-other", "pasted reading (via Issa)", "C2 = COLLAPSED", c2 == "COLLAPSED", c2)
        add_pred("P-C3", "Claude", "C3 = RESCUED", c3 == "RESCUED", c3)
        add_pred("P-C4", "Claude", "C4 = RESCUED", c4 == "RESCUED", c4)
        with open(src / "partB.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rowsB[0])); w_.writeheader(); w_.writerows(rowsB)
        with open(src / "partC.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rowsC[0])); w_.writeheader(); w_.writerows(rowsC)

    for name, rows in (("events", events), ("windows", windows), ("verdict", verdict), ("predictions", preds)):
        if rows:
            keys_ = list(dict.fromkeys(k for r in rows for k in r))
            with open(src / f"{name}.csv", "w", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=keys_); w_.writeheader(); w_.writerows(rows)
    lines = ["# l2_wall_0916 report", ""]
    for v in verdict:
        lines.append(f"- **{v['section']}**: `{v['verdict']}` {v['detail']}")
    lines += ["", "| id | owner | prediction | outcome | detail |", "|---|---|---|---|---|"]
    for p in preds:
        lines.append(f"| {p['id']} | {p['owner']} | {p['prediction']} | {p['outcome']} | {p['detail']} |")
    (src / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
