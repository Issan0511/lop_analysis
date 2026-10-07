"""drive_cifar5p1_1007 report (spec §5-§8): completeness, fixture audit, calibration-only window,
registered verdicts D1-D5, prediction scores.

    python -m analysis.drive_cifar5p1_1007.report audit     --src RAW/production
    python -m analysis.drive_cifar5p1_1007.report calibrate --src RAW/production --window results/drive_cifar5p1_1007/window_calibration.json
    git commit window_calibration.json            <- before anything of seeds 0-4 is opened
    python -m analysis.drive_cifar5p1_1007.report report    --src RAW/production --window ... --out results/drive_cifar5p1_1007

Only `report` opens primary/*.npz, and it refuses to unless the window file is committed at HEAD.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from src import drive_cifar5p1_1007 as D

TASKS = list(range(2, D.NT + 1))
HARD = [t for t in TASKS if t % 2 == 1]          # 3, 5, ..., 29
EASY = [t for t in TASKS if t % 2 == 0]          # 2, 4, ..., 30
K = {k: i for i, k in enumerate(D.KEYS)}
TR = {k: i for i, k in enumerate(D.TRANSPORT)}

PREDICTIONS = {
    "parent": {"window_hard": .65, "window_easy": .55,
               "D1": {"BOUNDARY_ENRICHED": .60, "UNRESOLVED": .30, "LATER_ENRICHED": .10},
               "D2": {"UPSTREAM_DOWN": .60}, "D3": {"EASY_SINKS": .70}, "D4": {"SELF_NET_DOWN": .55},
               "D5a": .40},
    "claude": {"window_hard": .75, "window_easy": .70,
               "D1": {"BOUNDARY_ENRICHED": .75, "UNRESOLVED": .20, "LATER_ENRICHED": .05},
               "D2": {"UPSTREAM_DOWN": .60, "UNRESOLVED": .30, "UPSTREAM_UP": .10},
               "D3": {"EASY_SINKS": .45, "UNRESOLVED": .30, "HARD_SINKS": .25},
               "D4": {"SELF_NET_DOWN": .70, "UNRESOLVED": .25, "SELF_NET_UP": .05},
               "D5a": .50, "D5b": .80},
}
MAIN = {"D2": "UPSTREAM_DOWN", "D3": "EASY_SINKS", "D4": "SELF_NET_DOWN"}


# --------------------------------------------------------------------------
# registered statistics
# --------------------------------------------------------------------------

def window(v) -> dict:
    """Spec §5: constant vs one-break two-level least squares on the 30 bin means, BIC, sign rule."""
    v = np.asarray(v, float)
    assert v.shape == (D.NBIN,) and np.isfinite(v).all()
    n = len(v)
    rss0 = float(((v - v.mean()) ** 2).sum())
    cands = []
    for b in range(1, n):
        e, l = float(v[:b].mean()), float(v[b:].mean())
        cands.append((float(((v[:b] - e) ** 2).sum() + ((v[b:] - l) ** 2).sum()), b, e, l))
    rss1, b, e, l = min(cands)                       # equal RSS -> smallest b
    base = dict(rss0=rss0, rss1=rss1, early=e, late=l, candidate_break_bin=b, v=v.tolist(),
                reason="operational segmentation; not a significance test")
    if rss0 == 0:
        return dict(base, label="WINDOW_NOT_IDENTIFIED", break_bin=None, boundary_updates=None,
                    bic0=None, bic1=None, why="constant series: the simple model")
    bic0 = n * math.log(rss0 / n) + 1 * math.log(n)
    bic1 = n * math.log(rss1 / n) + 3 * math.log(n) if rss1 > 0 else -math.inf
    ok = bic1 < bic0 and e < 0 and e < l
    return dict(base, label="WINDOW_IDENTIFIED" if ok else "WINDOW_NOT_IDENTIFIED", bic0=bic0,
                bic1=bic1 if np.isfinite(bic1) else None, break_bin=b if ok else None,
                boundary_updates=D.BIN * b if ok else None,
                why=None if ok else ("BIC1 >= BIC0" if not bic1 < bic0 else "early level not < 0 and < late"))


def directional(values, negative, positive) -> dict:
    """Spec §6: 5/5 negative -> `negative`, 5/5 positive -> `positive`, anything else UNRESOLVED."""
    a = np.asarray(values, float)
    if a.shape != (5,):
        return dict(label="INCOMPLETE", values=a.tolist())
    if not np.isfinite(a).all():
        return dict(label="DIVERGED", values=a.tolist())
    n, kneg, kpos = int((a != 0).sum()), int((a < 0).sum()), int((a > 0).sum())
    tail = lambda k: sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else 1.0  # noqa: E731
    label = negative if (a < 0).all() else positive if (a > 0).all() else "UNRESOLVED"
    return dict(label=label, values=a.tolist(), negative=kneg, positive=kpos, nonzero=n,
                one_sided_p_negative=tail(kneg), one_sided_p_positive=tail(kpos))


def brier_binary(p, outcome):
    return (p - float(outcome)) ** 2


def brier_multi(dist, label):
    return sum((p - float(k == label)) ** 2 for k, p in dist.items()) + (0.0 if label in dist else 1.0)


# --------------------------------------------------------------------------
# completeness and shards
# --------------------------------------------------------------------------

def validate(out, production=True) -> dict:
    """Raises on anything incomplete, foreign, forged or failed (spec §6 completeness, §9 S-manifest)."""
    out = Path(out)
    m = json.loads((out / "complete.json").read_text())
    ident = json.loads((out / "input_manifest.json").read_text())
    assert m["identity"] == ident, "identity mismatch"
    assert ident["run_id"] == D.RUN, "foreign run"
    assert ident["source_sha256"] == D.sources(), "source mismatch"
    if production:
        assert ident["mode"] == "production" and ident["seeds"] == D.PRODUCTION, "INCOMPLETE production"
        assert ident["groups"] == D.GROUPS_PRODUCTION and ident["n_tasks"] == D.NT and ident["fresh"]
    groups, nt = ident["groups"], ident["n_tasks"]
    paths = [f["path"] for f in m["files"]]
    assert len(paths) == len(set(paths)), "duplicate shard"
    expected = {f"{g}/t{t:02d}.npz" for g in groups for t in range(1, nt + 1)}
    expected |= {f"fixtures/t{t:02d}_j{j:03d}.pt" for t, j in D.FIXTURES if t <= nt}
    expected |= {"per_task.csv"} | ({"fresh_control.csv"} if ident["fresh"] else set())
    assert set(paths) == expected, f"missing or foreign shard: {sorted(set(paths) ^ expected)[:5]}"
    for f in m["files"]:
        p = out / f["path"]
        assert p.is_relative_to(out) and ".." not in Path(f["path"]).parts
        assert D.sha(p) == f["sha256"], f"checksum {f['path']}"
    validate_counts(m["status"], nt)
    validate_status(m["status"], nt)
    rows = list(csv.DictReader((out / "per_task.csv").open()))
    assert len(rows) == len(ident["seeds"]) * nt, "INCOMPLETE host rows"
    return m


def validate_counts(st, nt) -> None:
    """Every task observed in full: all images (2,500 hard / 500 easy), all 780 updates, 26 per bin."""
    assert [c["task"] for c in st["counts"]] == list(range(1, nt + 1)), "INCOMPLETE tasks"
    assert st["tc"] == D.STEPS * nt, "INCOMPLETE updates"
    for c in st["counts"]:
        assert c["N"] == (2500 if c["task"] % 2 else 500), "image count"
        assert c["updates"] == D.STEPS and all(x == D.BIN for x in c["bin_updates"]), "bin/update count"


def validate_status(st, nt) -> None:
    assert all(v == 0 for v in st["flags_total"].values()), f"CHECK_FAILED {st['flags_total']}"
    assert st["rng_unchanged"] and st["rng_probes"] == 2 * nt, "observer touched the RNG"
    assert not st["diverged"], "DIVERGED"
    for c in st["counts"]:
        assert c["telescoping_violations"] == 0 and c["J_violations"] == 0, "CHECK_FAILED telescoping"


def load_group(out, group, seeds, tasks=None) -> dict:
    """All task shards of one group, every invariant re-checked on read."""
    out, tasks = Path(out), list(range(1, D.NT + 1) if tasks is None else tasks)
    keys = ("acc", "rmax", "unit_bin", "unit_task", "J2", "J1", "m2_start", "m2_end", "m1_start", "m1_end",
            "gate2_end", "tel_defect", "tel_bound", "J2_defect", "J2_bound")
    res = {k: [] for k in keys}
    for t in tasks:
        with np.load(out / group / f"t{t:02d}.npz", allow_pickle=False) as d:
            assert d["seeds"].tolist() == list(seeds) and int(d["task"]) == t and int(d["hard"]) == t % 2
            assert d["keys"].tolist() == list(D.KEYS) and d["transport"].tolist() == list(D.TRANSPORT)
            assert d["unit_keys"].tolist() == list(D.UNIT_KEYS) and d["ratios"].tolist() == list(D.RATIOS)
            assert int(d["N"]) == (2500 if t % 2 else 500) and int(d["updates"]) == D.STEPS and int(d["every"]) == 1
            acc = d["acc"]
            assert acc.shape == (len(seeds), D.NBIN, len(D.KEYS)) and np.isfinite(acc).all()
            assert (acc[..., K["updates"]] == D.BIN).all(), "bin count"
            for k in D.FLAGS:
                assert (acc[..., K[k]] == 0).all(), f"CHECK_FAILED {k}"
            assert (d["tel_defect"] <= d["tel_bound"]).all()
            for k in keys:
                res[k].append(d[k])
    return {k: np.stack(v, 1) for k, v in res.items()}      # (seeds, tasks, ...)


# --------------------------------------------------------------------------
# independent fixture audit (spec §9 S-self-total (a)/(d)): CPU float64, math.fsum, rows from the plan
# --------------------------------------------------------------------------

def _fsum_cols(a):
    """Correctly rounded column sums of a 2-D float64 array (math.fsum per column)."""
    return np.array([math.fsum(col) for col in a.T])


def audit_fixture(fx, seeds, cifar, device, X_all=None) -> dict:
    import torch
    t, j, R = fx["t"], fx["j"], len(seeds)
    X_all = cifar.inputs("train", D.COND, device) if X_all is None else X_all
    rows = D.C.class_rows(cifar.train_y)
    plans = [D.C.task_plan(s) for s in seeds]
    task_rows = lambda tt: [torch.cat([rows[q] for q in plans[r][tt - 1][1]]) for r in range(R)]  # noqa: E731
    X = torch.stack([X_all[x] for x in task_rows(t)])
    N = X.shape[1]
    assert N == fx["N"]

    def native_h(Pl, Xs):
        with torch.no_grad():
            z1 = torch.baddbmm(Pl[1].to(device)[:, None, :], Xs, Pl[0].to(device).transpose(1, 2))
            return z1.clamp(min=0.0).double().cpu().numpy()

    h_old, h_new = native_h(fx["p_old"], X), native_h(fx["p_new"], X)
    Xc = X.double().cpu().numpy()
    worst = {}

    def note(name, gpu, ind, bound):
        r = float(np.max(np.abs(np.asarray(gpu) - np.asarray(ind)) / np.asarray(bound)))
        worst[name] = max(worst.get(name, 0.0), r)

    g = D.gamma
    for r in range(R):
        mu0 = _fsum_cols(h_old[r]) / N
        mu1 = _fsum_cols(h_new[r]) / N
        mux = _fsum_cols(Xc[r]) / N
        f0, f1 = fx["f0"], fx["f1"]
        note("mu_old", f0["mu"][r].numpy(), mu0, f0["mue"][r].numpy() + 2 * D.U64 * mu0 + D.TINY64)
        note("mu_new", f1["mu"][r].numpy(), mu1, f1["mue"][r].numpy() + 2 * D.U64 * mu1 + D.TINY64)
        absx = _fsum_cols(np.abs(Xc[r])) / N
        note("mu_x", fx["mux"][r].numpy(), mux, g(N + 2) * absx + 2 * D.U64 * np.abs(mux) + D.TINY64)
        W2o, b2o = fx["p_old"][2][r].double().numpy(), fx["p_old"][3][r].double().numpy()
        W2n, b2n = fx["p_new"][2][r].double().numpy(), fx["p_new"][3][r].double().numpy()
        W1o, b1o = fx["p_old"][0][r].double().numpy(), fx["p_old"][1][r].double().numpy()
        W1n, b1n = fx["p_new"][0][r].double().numpy(), fx["p_new"][1][r].double().numpy()
        mue0 = f0["mue"][r].numpy() + 2 * D.U64 * mu0
        mue1 = f1["mue"][r].numpy() + 2 * D.U64 * mu1
        S = np.array([math.fsum(list((W2n[i] - W2o[i]) * mu0) + [b2n[i] - b2o[i]]) for i in range(D.HID)])
        U = np.array([math.fsum(list(W2n[i] * (mu1 - mu0))) for i in range(D.HID)])
        S1 = np.array([math.fsum(list((W1n[i] - W1o[i]) * mux) + [b1n[i] - b1o[i]]) for i in range(D.HID)])
        dW2, dmu = np.abs(W2n - W2o), np.abs(mu1 - mu0)
        bS = g(2 * (D.HID + 1) + 4) * (dW2 @ mu0 + np.abs(b2n - b2o)) + dW2 @ mue0 + D.U64 * np.abs(S) + 4 * D.TINY64
        bU = g(2 * D.HID + 4) * (np.abs(W2n) @ dmu) + np.abs(W2n) @ (mue0 + mue1) + D.U64 * np.abs(U) + 4 * D.TINY64
        dW1 = np.abs(W1n - W1o)
        mxe = g(N + 2) * absx + 2 * D.U64 * np.abs(mux)
        bS1 = g(2 * (3072 + 1) + 4) * (dW1 @ np.abs(mux) + np.abs(b1n - b1o)) + dW1 @ mxe + D.U64 * np.abs(S1) + 4 * D.TINY64
        v = fx["values"]
        note("S2", v["S2"][r].numpy(), S, bS)
        note("U2", v["U2"][r].numpy(), U, bU)
        note("D2", v["D2"][r].numpy(), S + U, bS + bU + D.U64 * np.abs(S + U))
        note("S1", v["S1"][r].numpy(), S1, bS1)
        if j == 1 and fx["J2"] is not None:
            Xp = torch.stack([X_all[x] for x in task_rows(t - 1)])
            hp = native_h(fx["p_old"], Xp)
            Np = Xp.shape[1]
            mup = _fsum_cols(hp[r]) / Np
            mupx = _fsum_cols(Xp[r].double().cpu().numpy()) / Np
            J2 = np.array([math.fsum(list(W2o[i] * (mu0 - mup))) for i in range(D.HID)])
            J1 = np.array([math.fsum(list(W1o[i] * (mux - mupx))) for i in range(D.HID)])
            bJ = g(2 * D.HID + 4) * (np.abs(W2o) @ np.abs(mu0 - mup)) + np.abs(W2o) @ (mue0 + g(Np + 2) * mup + 2 * D.U64 * mup) + D.U64 * np.abs(J2) + 4 * D.TINY64
            absxp = _fsum_cols(np.abs(Xp[r].double().cpu().numpy())) / Np
            bJ1 = g(2 * 3072 + 4) * (np.abs(W1o) @ np.abs(mux - mupx)) + np.abs(W1o) @ (mxe + g(Np + 2) * absxp + 2 * D.U64 * np.abs(mupx)) + D.U64 * np.abs(J1) + 4 * D.TINY64
            note("J2", fx["J2"][r].numpy(), J2, bJ)
            note("J1", fx["J1"][r].numpy(), J1, bJ1)
    return dict(t=t, j=j, worst_ratio=worst, pass_=all(x <= 1 for x in worst.values()))


def audit(out, device=None, cifar=None) -> dict:
    import torch
    out = Path(out)
    ident = json.loads((out / "input_manifest.json").read_text())
    device = device or D.setup()
    cifar = cifar or D.load_cifar()
    X_all = cifar.inputs("train", D.COND, device)
    res = []
    for t, j in D.FIXTURES:
        if t > ident["n_tasks"]:
            continue
        fx = torch.load(out / "fixtures" / f"t{t:02d}_j{j:03d}.pt", weights_only=False)
        res.append(audit_fixture(fx, ident["seeds"], cifar, device, X_all))
    ok = all(r["pass_"] for r in res)
    return dict(status="PASS" if ok else "CHECK_FAILED", fixtures=res)


# --------------------------------------------------------------------------
# calibration (seeds 5-9 only) and the primary report
# --------------------------------------------------------------------------

def v_series(acc, tasks) -> np.ndarray:
    """Spec §5 v_b: per (seed, task) the bin's Dm unit-sum / (26 x 100), equal weights."""
    x = acc[:, [t - 1 for t in tasks], :, K["D2_sum"]] / (D.BIN * D.HID)
    return x.mean(axis=(0, 1))


def calibration_core(out) -> dict:
    """Opens calibration/*.npz and nothing else."""
    out = Path(out)
    seeds = D.PRODUCTION[5:]
    g = load_group(out, "calibration", seeds)
    files = {f"calibration/t{t:02d}.npz": D.sha(out / "calibration" / f"t{t:02d}.npz") for t in range(1, D.NT + 1)}
    return dict(hard=window(v_series(g["acc"], HARD)), easy=window(v_series(g["acc"], EASY)),
                calibration_seeds=seeds, calibration_files=files)


def calibrate(out, destination) -> dict:
    out = Path(out)
    validate(out)
    w = calibration_core(out)
    w.update(run_id=D.RUN, complete_sha256=D.sha(out / "complete.json"), rule_source_sha256=D.sources(),
             rule="spec §5: BIC0 = 30 log(RSS0/30) + log 30, BIC1 = 30 log(RSS1/30) + 3 log 30; "
                  "identified iff BIC1 < BIC0 and early < 0 and early < late; ties -> simple / smallest b")
    D.put(destination, w)
    return {k: w[k]["label"] for k in ("hard", "easy")}


def committed(path) -> bytes:
    path = Path(path).resolve()
    rel = path.relative_to(D.ROOT)
    blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=D.ROOT)
    assert blob == path.read_bytes(), "commit window_calibration.json before opening primary outcomes"
    return blob


def per_rate(x, n_updates):
    return x / (n_updates * D.HID)


def analyse(g, w) -> dict:
    """All registered quantities for the 5 primary seeds.  g: load_group('primary'); w: windows."""
    acc, ut = g["acc"], g["unit_task"]                     # (5, 30, 30, K), (5, 30, 100, 7)
    S = acc[..., K["S2_sum"]]                              # (5, 30 tasks, 30 bins) unit sums
    U = acc[..., K["U2_sum"]]
    Dm = acc[..., K["D2_sum"]]
    brk = {1: w["hard"]["break_bin"], 0: w["easy"]["break_bin"]}
    ti = lambda ts: [t - 1 for t in ts]  # noqa: E731

    def bl(t):
        b = brk[t % 2]
        if b is None:
            return None
        bs, ls = S[:, t - 1, :b].sum(-1), S[:, t - 1, b:].sum(-1)
        return bs, ls, per_rate(bs, D.BIN * b) - per_rate(ls, D.BIN * (D.NBIN - b))

    res = {}
    both = brk[0] is not None and brk[1] is not None
    diffs = {t: bl(t) for t in TASKS}
    for name, ts in (("D1", TASKS), ("D1_hard", HARD), ("D1_easy", EASY)):
        if all(diffs[t] is not None for t in ts):
            val = np.mean([diffs[t][2] for t in ts], axis=0)
            res[name] = directional(val, "BOUNDARY_ENRICHED", "LATER_ENRICHED")
        else:
            res[name] = dict(label="NOT_JUDGED", why="WINDOW_NOT_IDENTIFIED")
    res["D2"] = directional(U[:, ti(TASKS)].sum((1, 2)), "UPSTREAM_DOWN", "UPSTREAM_UP")
    rate = lambda X, ts: per_rate(X[:, ti(ts)].sum(-1), D.STEPS).mean(1)  # noqa: E731
    for name, X in (("D3", Dm), ("D3_S", S), ("D3_U", U)):
        res[name] = directional(rate(X, EASY) - rate(X, HARD), "EASY_SINKS", "HARD_SINKS")
    res["D4"] = directional(S[:, ti(TASKS)].sum((1, 2)), "SELF_NET_DOWN", "SELF_NET_UP")
    ts5 = [t for t in TASKS if diffs[t] is not None]
    bsum = np.sum([diffs[t][0] for t in ts5], axis=0) if ts5 else np.full(5, np.nan)
    tsum = S[:, ti(ts5)].sum((1, 2)) if ts5 else np.full(5, np.nan)
    share = np.where(tsum != 0, bsum / np.where(tsum == 0, 1, tsum), np.nan)
    res["D5a"] = dict(share=share.tolist(), count_over_half=int(np.nansum(share > .5)),
                      tasks="all 2-30" if both else ("hard only" if brk[1] else "easy only" if brk[0] else "none"),
                      label=("5/5" if np.nansum(share > .5) == 5 else f"{int(np.nansum(share > .5))}/5"))
    sc = acc[..., K["S2c_sum"]][:, ti(TASKS)].sum((1, 2))
    res["D5b"] = dict(S_conf_total=sc.tolist(), negative=int((sc < 0).sum()),
                      label="5/5 negative" if (sc < 0).all() else "5/5 positive" if (sc > 0).all() else "mixed")
    res["D5c"] = dict(task1_first78_S=S[:, 0, :3].sum(-1).tolist(),
                      negative=int((S[:, 0, :3].sum(-1) < 0).sum()))
    tables = dict(S_total=S[:, ti(TASKS)].sum((1, 2)), U_total=U[:, ti(TASKS)].sum((1, 2)),
                  D_total=Dm[:, ti(TASKS)].sum((1, 2)),
                  J_total=np.nansum(g["J2"][:, ti(TASKS)].sum(-1), 1),
                  J_into_easy=g["J2"][:, ti(EASY)].sum((1, 2)), J_into_hard=g["J2"][:, ti(HARD)].sum((1, 2)),
                  boundary_S=bsum, later_S=(tsum - bsum) if ts5 else np.full(5, np.nan),
                  D1_value=np.array(res["D1"].get("values", [np.nan] * 5)),
                  easy_D_rate=rate(Dm, EASY), hard_D_rate=rate(Dm, HARD),
                  easy_S_rate=rate(S, EASY), hard_S_rate=rate(S, HARD),
                  easy_U_rate=rate(U, EASY), hard_U_rate=rate(U, HARD),
                  S_conf_total=sc, S_label_total=acc[..., K["S2l_sum"]][:, ti(TASKS)].sum((1, 2)),
                  S_hist_total=acc[..., K["S2h_sum"]][:, ti(TASKS)].sum((1, 2)),
                  S1_total=acc[..., K["S1_sum"]][:, ti(TASKS)].sum((1, 2)),
                  J1_total=np.nansum(g["J1"][:, ti(TASKS)].sum(-1), 1),
                  S_neg_frequency=acc[..., K["S2_nneg"]][:, ti(TASKS)].sum((1, 2)) / (len(TASKS) * D.STEPS * D.HID),
                  task1_first78_S=S[:, 0, :3].sum(-1), share=share)
    return dict(verdicts=res, tables={k: np.asarray(v, float).tolist() for k, v in tables.items()})


def score(verdicts, w) -> dict:
    out = {}
    ident = {"window_hard": w["hard"]["label"] == "WINDOW_IDENTIFIED",
             "window_easy": w["easy"]["label"] == "WINDOW_IDENTIFIED"}
    for who, pr in PREDICTIONS.items():
        s = {}
        for k in ("window_hard", "window_easy"):
            s[k] = brier_binary(pr[k], ident[k])
        d1 = verdicts["D1"]["label"]
        s["D1_multiclass"] = None if d1 == "NOT_JUDGED" else brier_multi(pr["D1"], d1)
        s["D1_binary_BOUNDARY_ENRICHED"] = None if d1 == "NOT_JUDGED" else brier_binary(pr["D1"]["BOUNDARY_ENRICHED"], d1 == "BOUNDARY_ENRICHED")
        for k, lab in MAIN.items():
            s[f"{k}_binary_{lab}"] = brier_binary(pr[k][lab], verdicts[k]["label"] == lab)
            if len(pr[k]) == 3:
                s[f"{k}_multiclass"] = brier_multi(pr[k], verdicts[k]["label"])
        s["D5a_binary_5of5"] = brier_binary(pr["D5a"], verdicts["D5a"]["count_over_half"] == 5) if verdicts["D5a"]["tasks"] != "none" else None
        if "D5b" in pr:
            s["D5b_binary_5of5_negative"] = brier_binary(pr["D5b"], verdicts["D5b"]["negative"] == 5)
        out[who] = s
    return out


def report(out, window_path, destination, deviation=None) -> dict:
    out, destination = Path(out), Path(destination)
    m = validate(out)
    a = json.loads((destination / "audit.json").read_text())
    assert a["status"] == "PASS", "fixture audit failed"
    w = json.loads(committed(window_path))
    assert w["complete_sha256"] == D.sha(out / "complete.json") and w["calibration_seeds"] == D.PRODUCTION[5:]
    ident = m["identity"]
    rep = "analysis/drive_cifar5p1_1007/report.py"
    if ident["source_sha256"][rep] != D.sources()[rep]:
        assert deviation, "report.py changed after the run: record a --deviation"
    g = load_group(out, "primary", D.PRODUCTION[:5])         # the first read of seeds 0-4
    res = analyse(g, w)
    sc = score(res["verdicts"], w)
    verdict = dict(run_id=D.RUN, windows={k: {x: w[k][x] for x in ("label", "break_bin", "boundary_updates", "early", "late", "bic0", "bic1")} for k in ("hard", "easy")},
                   **res["verdicts"], applicability=dict(host_identity=json.loads((out / "host_identity.json").read_text()),
                                                          flags_total=m["status"]["flags_total"],
                                                          audit=a["status"]),
                   complete_sha256=D.sha(out / "complete.json"), window_sha256=D.sha(window_path),
                   deviation=deviation,
                   limitations="observation, not intervention; five independent seeds, 5/5 rule; "
                               "moves of the mean preactivation on the current task's images only")
    D.put(destination / "verdict.json", verdict)
    D.put(destination / "predictions.json", dict(predictions=PREDICTIONS, brier=sc))
    T = res["tables"]
    with (destination / "per_seed.csv").open("w") as fh:
        cols = list(T)
        fh.write("seed," + ",".join(cols) + "\n")
        for s in range(5):
            fh.write(f"{s}," + ",".join(f"{T[c][s]:.10g}" for c in cols) + "\n")
    S, U, Dm = (g["acc"][..., K[k]] for k in ("S2_sum", "U2_sum", "D2_sum"))
    with (destination / "per_task.csv").open("w") as fh:
        cols = ["S", "U", "D", "J2", "S_conf", "S_label", "S_hist", "S1", "J1", "m2_start_mean", "m2_end_mean", "G2_end"]
        fh.write("seed,task,hard," + ",".join(cols) + "\n")
        for s in range(5):
            for t in range(1, D.NT + 1):
                i = t - 1
                vals = [S[s, i].sum(), U[s, i].sum(), Dm[s, i].sum(), np.sum(g["J2"][s, i]),
                        g["acc"][s, i, :, K["S2c_sum"]].sum(), g["acc"][s, i, :, K["S2l_sum"]].sum(),
                        g["acc"][s, i, :, K["S2h_sum"]].sum(), g["acc"][s, i, :, K["S1_sum"]].sum(),
                        np.sum(g["J1"][s, i]), g["m2_start"][s, i].mean(), g["m2_end"][s, i].mean(),
                        g["gate2_end"][s, i].mean()]
                fh.write(f"{s},{t},{t % 2}," + ",".join(f"{float(v):.10g}" for v in vals) + "\n")
    D.save_npz(destination / "per_bin.npz", dict(S=S, U=U, D=Dm, S_conf=g["acc"][..., K["S2c_sum"]],
                                                 seeds=np.arange(5), tasks=np.arange(1, D.NT + 1)))
    D.save_npz(destination / "per_unit_task.npz", dict(unit_task=g["unit_task"], J2=g["J2"], J1=g["J1"],
                                                       m2_start=g["m2_start"], m2_end=g["m2_end"],
                                                       transport=np.asarray(D.TRANSPORT)))
    return dict(verdict=verdict, brier=sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "calibrate", "report"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--window")
    ap.add_argument("--out", default=str(D.OUT))
    ap.add_argument("--deviation")
    a = ap.parse_args()
    if a.stage == "audit":
        validate(a.src)
        with D.gpu_lock(2.0):
            res = audit(a.src)
        D.put(Path(a.out) / "audit.json", res)
        print("audit", res["status"], {f"t{r['t']}_j{r['j']}": round(max(r["worst_ratio"].values()), 4) for r in res["fixtures"]})
    elif a.stage == "calibrate":
        print(calibrate(a.src, a.window))
    else:
        r = report(a.src, a.window, a.out, a.deviation)
        print({k: v["label"] for k, v in r["verdict"].items() if isinstance(v, dict) and "label" in v})


if __name__ == "__main__":
    main()
