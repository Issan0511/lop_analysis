"""shell_l2_rlmnist_0913 -- spec section 4 judgements (Random Label MNIST; R and SNA judged independently).

    .venv/bin/python analysis/shell_l2_rlmnist_0913/verdict.py \
        --src results/shell_l2_rlmnist_0913/runs --out results/shell_l2_rlmnist_0913 \
        --checks results/shell_l2_rlmnist_0913/checks.json

Shards are enumerated explicitly as {act}_{reg}_s{seed}; a missing one is reported, never globbed
around.  Never point --src at a main run that is still in flight (spec section 6): pass a smoke or
check directory explicitly instead.

Window = tasks 31-50 of online_acc.  Primary contrasts per activation: d1 = shell - l2,
d2 = shell - l2init, d3 = l2init - l2 (seed-paired).  Labels A-D follow the ordered table of spec 4.4;
the Bonferroni variant (alpha 0.025, 97.5% CI) only licenses cross-activation sentences (spec 4.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import socket
import subprocess
import time
from importlib import metadata
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "shell_l2_rlmnist_0913"
ACTS = ("R", "SNA")
REGS = ("none", "l2", "l2init", "shell")
SEEDS = tuple(range(10))
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
WIN = (31, 50)
N_WIN = WIN[1] - WIN[0] + 1
DELTA = 0.005                      # equivalence band on d2 (spec 4.1)
N_BOOT = 10_000
RNG_SEED = 20260913
GUARD_LO, GUARD_HI = 0.9, 1.1      # norm-match guard (spec 4.3)
WEIGHTS = ("W1", "W2", "W3")       # ndim >= 2: the guard is judged on these only
BIASES = ("b1", "b2", "b3")        # REPORT only
EPS_INERT_SQ = 1e-12 * 2 ** 25     # spec 2.3: SHELL_EPS2 is bit-inert above this sum of squares
CONTRASTS = (("d1", "shell", "l2"), ("d2", "shell", "l2init"), ("d3", "l2init", "l2"))
REPORT_CONTRASTS = (("r_l2_none", "l2", "none"), ("r_l2init_none", "l2init", "none"),
                    ("r_shell_none", "shell", "none"))
LEVELS = {"a05": (0.05, "ci95"), "bonf": (0.025, "ci975")}
SENTENCE = {
    "A": "初期方向への回帰は主要因ではなく、初期半径の維持でL2-Initの利得の大部分を説明できる",
    "B": "初期方向へのアンカーに、半径制御を超える追加価値がある",
    "C": "初期方向へのアンカーは新タスク適応を妨げている",
    "D": "方向効果は未同定。性能比較のみ確定",
}
REF_0906 = {("R", "none"): "R", ("R", "l2"): "R_l2", ("R", "l2init"): "R_l2init",
            ("SNA", "none"): "SNA", ("SNA", "l2"): "SNA_l2", ("SNA", "l2init"): "SNA_l2init"}


def shard_name(act: str, reg: str, seed: int) -> str:
    return f"{act}_{reg}_s{seed}"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(src: Path):
    pts, lms, provs, missing = [], [], {}, []
    for act in ACTS:
        for reg in REGS:
            for seed in SEEDS:
                d = src / shard_name(act, reg, seed)
                files = (d / "per_task.csv", d / "layer_metrics.csv", d / "provenance.json")
                if not all(f.exists() for f in files):
                    missing.append(d.name)
                    continue
                pts.append(pd.read_csv(files[0]))
                lms.append(pd.read_csv(files[1]))
                provs[d.name] = json.loads(files[2].read_text())
    pt = pd.concat(pts, ignore_index=True) if pts else pd.DataFrame(
        columns=["act", "reg", "seed", "task", "online_acc", "memo_acc"])
    lm = pd.concat(lms, ignore_index=True) if lms else pd.DataFrame(
        columns=["act", "reg", "seed", "task", "tensor", "norm_ratio"])
    return pt, lm, provs, missing


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def sign_p(wins: int, n: int) -> float:
    """Two-sided exact sign test."""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def window_table(pt: pd.DataFrame, act: str) -> dict:
    """reg -> {seed: mean online_acc over tasks 31-50}, only for seeds with all 20 finite tasks."""
    g = pt[(pt.act == act) & (pt.task >= WIN[0]) & (pt.task <= WIN[1])]
    tab = {}
    for reg in REGS:
        tab[reg] = {}
        for seed in SEEDS:
            x = g[(g.reg == reg) & (g.seed == seed)].sort_values("task")
            vals = x.online_acc.to_numpy(dtype=float)
            if len(vals) == N_WIN and np.isfinite(vals).all() and \
                    list(x.task) == list(range(WIN[0], WIN[1] + 1)):
                tab[reg][seed] = float(vals.mean())
    return tab


def contrast(tab: dict, a: str, b: str, act_i: int, c_i: int) -> dict:
    seeds = sorted(set(tab[a]) & set(tab[b]))
    d = np.array([tab[a][s] - tab[b][s] for s in seeds], dtype=float)
    n = len(d)
    nan = float("nan")
    st = {"a": a, "b": b, "n": n, "seeds": seeds, "diffs": d.tolist()}
    if n == 0:
        st.update(mean=nan, se=nan, ci95_lo=nan, ci95_hi=nan, ci975_lo=nan, ci975_hi=nan,
                  wins=0, n_sign=0, p_sign=nan)
        return st
    nz = d[d != 0]
    wins = int((nz > 0).sum())
    rng = np.random.default_rng([RNG_SEED, act_i, c_i])
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = d[idx].mean(axis=1)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    lo975, hi975 = np.percentile(boot, [1.25, 98.75])
    st.update(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else nan,
              ci95_lo=float(lo95), ci95_hi=float(hi95), ci975_lo=float(lo975), ci975_hi=float(hi975),
              wins=wins, n_sign=int(len(nz)), p_sign=sign_p(wins, int(len(nz))), boot_idx=idx)
    return st


def greater(st: dict, level: str) -> bool:
    alpha, ci = LEVELS[level]
    return st["n"] > 0 and st[f"{ci}_lo"] > 0 and st["p_sign"] < alpha


def d2_class(st: dict, level: str) -> str:
    alpha, ci = LEVELS[level]
    if st["n"] == 0:
        return "AMBIG"
    lo, hi = st[f"{ci}_lo"], st[f"{ci}_hi"]
    if -DELTA <= lo and hi <= DELTA:
        return "EQUIV"
    if lo > DELTA and st["p_sign"] < alpha:
        return "SHELL_ABOVE"
    if hi < -DELTA and st["p_sign"] < alpha:
        return "SHELL_BELOW"
    return "AMBIG"


def contrast_class(name: str, st: dict, level: str) -> str:
    if name == "d2":
        return d2_class(st, level)
    alpha, ci = LEVELS[level]
    if greater(st, level):
        return "GREATER"
    if st["n"] > 0 and st[f"{ci}_hi"] < 0 and st["p_sign"] < alpha:
        return "LESS"
    return "UNRESOLVED"


def guard(lm: pd.DataFrame, act: str, tensors=WEIGHTS) -> tuple[bool, dict]:
    g = lm[(lm.act == act) & (lm.task >= WIN[0]) & (lm.task <= WIN[1])]
    out, ok = {}, True
    for w in tensors:
        xs = g[(g.reg == "shell") & (g.tensor == w)].norm_ratio.to_numpy(dtype=float)
        xi = g[(g.reg == "l2init") & (g.tensor == w)].norm_ratio.to_numpy(dtype=float)
        if len(xs) and len(xi) and np.isfinite(xs).all() and np.isfinite(xi).all():
            ms, mi = float(np.median(xs)), float(np.median(xi))
            ratio = ms / mi
            passed = bool(GUARD_LO <= ratio <= GUARD_HI)
        else:
            ms = mi = ratio = float("nan")
            passed = False
        ok &= passed
        out[w] = {"shell_median": ms, "l2init_median": mi, "ratio": ratio, "pass": passed,
                  "n_shell": int(len(xs)), "n_l2init": int(len(xi))}
    return bool(ok), out


def validity(pt, lm, provs, missing, act) -> list[str]:
    """Spec 4.4 row 0: every one of the activation's 40 runs present, finite, complete, stream-identical."""
    problems = []
    for reg in REGS:
        for seed in SEEDS:
            name = shard_name(act, reg, seed)
            if name in missing:
                problems.append(f"{name}: missing")
                continue
            run = provs.get(name, {}).get("runs", {}).get(str(seed))
            if run is None:
                problems.append(f"{name}: no run record for seed {seed}")
                continue
            if run.get("divergence", {}).get("diverged"):
                problems.append(f"{name}: diverged at task {run['divergence'].get('task')}")
            g = pt[(pt.act == act) & (pt.reg == reg) & (pt.seed == seed)]
            gw = g[(g.task >= WIN[0]) & (g.task <= WIN[1])]
            if len(gw) != N_WIN or not np.isfinite(gw.online_acc.to_numpy(dtype=float)).all():
                problems.append(f"{name}: window tasks {WIN[0]}-{WIN[1]} incomplete or non-finite")
            glw = lm[(lm.act == act) & (lm.reg == reg) & (lm.seed == seed)
                     & (lm.task >= WIN[0]) & (lm.task <= WIN[1])]
            if len(glw) != N_WIN * len(TENSORS):
                problems.append(f"{name}: layer_metrics window rows {len(glw)} != {N_WIN * len(TENSORS)}")
    # spec 2.4: the data streams a seed's runs consumed must be bit-identical across all 8 arms
    for seed in SEEDS:
        keys = {}
        for a in ACTS:
            for reg in REGS:
                run = provs.get(shard_name(a, reg, seed), {}).get("runs", {}).get(str(seed))
                if run is not None:
                    keys[shard_name(a, reg, seed)] = tuple(run.get(k) for k in (
                        "init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256"))
        if len(set(keys.values())) > 1:
            problems.append(f"seed {seed}: in-run stream hashes differ across arms "
                            f"({len(set(keys.values()))} distinct among {len(keys)} shards)")
    return problems


def label_for(problems: list[str], guard_ok: bool, st: dict, level: str) -> tuple[str, str]:
    """Spec 4.4, applied top to bottom."""
    alpha, ci = LEVELS[level]
    if problems:
        return "D", "INCOMPLETE"
    if not guard_ok:
        return "D", "NORM_MATCH_FAIL"
    c2 = d2_class(st["d2"], level)
    if c2 == "SHELL_BELOW":
        return "B", ""
    if c2 == "SHELL_ABOVE":
        return "C", ""
    if c2 == "AMBIG":
        return "D", "CI_AMBIGUOUS"
    if greater(st["d1"], level) and st["d3"][f"{ci}_lo"] >= 2 * DELTA and st["d3"]["p_sign"] < alpha:
        return "A", ""
    return "D", "EQUIV_GAIN_UNRESOLVED"


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(pt: pd.DataFrame, lm: pd.DataFrame, provs: dict, missing: list[str]) -> dict:
    res = {"acts": {}}
    for act_i, act in enumerate(ACTS):
        tab = window_table(pt, act)
        problems = validity(pt, lm, provs, missing, act)
        st = {name: contrast(tab, a, b, act_i, c_i + 1) for c_i, (name, a, b) in enumerate(CONTRASTS)}
        rep = {name: contrast(tab, a, b, act_i, c_i + 1 + len(CONTRASTS))
               for c_i, (name, a, b) in enumerate(REPORT_CONTRASTS)}
        g_ok, g_w = guard(lm, act, WEIGHTS)
        gb_ok, g_b = guard(lm, act, BIASES)
        lab, reason = label_for(problems, g_ok, st, "a05")
        lab_b, reason_b = label_for(problems, g_ok, st, "bonf")
        perf = {}
        for reg in REGS:
            v = np.array(list(tab[reg].values()), dtype=float)
            memo = pt[(pt.act == act) & (pt.reg == reg) & (pt.task >= WIN[0]) & (pt.task <= WIN[1])]
            memo_seed = memo.groupby("seed").memo_acc.mean() if len(memo) and "memo_acc" in memo else pd.Series(dtype=float)
            perf[reg] = {"n": int(len(v)),
                         "mean": float(v.mean()) if len(v) else float("nan"),
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                         "median": float(np.median(v)) if len(v) else float("nan"),
                         "min": float(v.min()) if len(v) else float("nan"),
                         "max": float(v.max()) if len(v) else float("nan"),
                         "memo_acc_mean": float(memo_seed.mean()) if len(memo_seed) else float("nan")}
        # REPORT: share of L2-Init's gain recovered by shell, f = d1 / d3, same seed resamples
        frac = {"point": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
        if st["d1"]["n"] and st["d1"]["seeds"] == st["d3"]["seeds"] and st["d3"]["mean"] != 0:
            d1, d3, idx = np.array(st["d1"]["diffs"]), np.array(st["d3"]["diffs"]), st["d1"]["boot_idx"]
            with np.errstate(divide="ignore", invalid="ignore"):
                bf = d1[idx].mean(1) / d3[idx].mean(1)
            bf = bf[np.isfinite(bf)]
            frac = {"point": float(d1.mean() / d3.mean()),
                    "ci95_lo": float(np.percentile(bf, 2.5)) if len(bf) else float("nan"),
                    "ci95_hi": float(np.percentile(bf, 97.5)) if len(bf) else float("nan")}
        # REPORT: per-tensor window medians for all four arms
        lw = lm[(lm.act == act) & (lm.task >= WIN[0]) & (lm.task <= WIN[1])]
        tens = {}
        for reg in REGS:
            for w in TENSORS:
                x = lw[(lw.reg == reg) & (lw.tensor == w)]
                if not len(x):
                    continue
                share = (x.pen_shell / x.pen_l2init).replace([np.inf, -np.inf], np.nan)
                tens[f"{reg}|{w}"] = {"norm_ratio": float(x.norm_ratio.median()),
                                      "cos_w0": float(x.cos_w0.median()),
                                      "pen_l2": float(x.pen_l2.median()),
                                      "pen_l2init": float(x.pen_l2init.median()),
                                      "pen_shell": float(x.pen_shell.median()),
                                      "radial_share_of_l2init_pen": float(share.median()),
                                      "reg_loss": float(x.reg_loss.median())}
        la = lm[lm.act == act]
        min_sq = {}
        for reg in REGS:
            x = la[la.reg == reg]
            if len(x):
                i = x.sq_norm.astype(float).idxmin()
                min_sq[reg] = {"min_sq_norm": float(x.loc[i, "sq_norm"]), "tensor": str(x.loc[i, "tensor"]),
                               "task": int(x.loc[i, "task"]), "seed": int(x.loc[i, "seed"]),
                               "above_eps_inert_threshold": bool(float(x.loc[i, "sq_norm"]) > EPS_INERT_SQ)}
        res["acts"][act] = {
            "window_means": {reg: {str(s): v for s, v in tab[reg].items()} for reg in REGS},
            "problems": problems, "perf": perf,
            "contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "boot_idx"} for k, v in st.items()},
            "classes": {k: contrast_class(k, v, "a05") for k, v in st.items()},
            "classes_bonf": {k: contrast_class(k, v, "bonf") for k, v in st.items()},
            "report_contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "boot_idx"} for k, v in rep.items()},
            "guard_pass": g_ok, "guard": g_w, "guard_bias_report_pass": gb_ok, "guard_bias_report": g_b,
            "label": lab, "reason": reason, "label_bonf": lab_b, "reason_bonf": reason_b,
            "sentence": SENTENCE[lab], "recovered_fraction_report": frac,
            "tensor_medians_report": tens, "min_sq_norm_report": min_sq}
    la, lb = res["acts"]["R"], res["acts"]["SNA"]
    res["cross_activation_statement_allowed"] = bool(
        la["label"] == lb["label"] and la["label_bonf"] == la["label"] and lb["label_bonf"] == lb["label"])
    return res


def ref_0906(root: Path) -> dict:
    """Window means of the 0906 (CUDA) runs for the six shared arms -- reference only, never judged."""
    out = {}
    if not root.exists():
        return out
    for (act, reg), sub in REF_0906.items():
        f = root / sub / "per_task.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.task >= WIN[0]) & (d.task <= WIN[1]) & d.online_acc.notna()]
        per_seed = d.groupby("seed").online_acc.mean()
        out[f"{act}|{reg}"] = {"n": int(len(per_seed)), "mean": float(per_seed.mean()),
                               "median": float(per_seed.median())}
    return out


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def sh(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:                                  # provenance must not kill the verdict
        return f"<unavailable: {e}>"


def environment() -> dict:
    md = "http://metadata.google.internal/computeMetadata/v1/instance/"
    gcp = {k: sh(["curl", "-s", "-m", "3", "-H", "Metadata-Flavor: Google", md + k])
           for k in ("name", "machine-type", "zone")}
    uv = Path.home() / ".local" / "bin" / "uv"
    vers = {}
    for pkg in ("torch", "numpy", "pandas"):
        try:
            vers[pkg] = metadata.version(pkg)
        except Exception:
            vers[pkg] = None
    return {"hostname": socket.gethostname(), "uname_a": sh(["uname", "-a"]), "lscpu": sh(["lscpu"]),
            "uv_pip_freeze": sh([str(uv), "pip", "freeze", "--python", str(REPO / ".venv" / "bin" / "python")]),
            "python": platform.python_version(), "packages": vers, "gcp_instance": gcp}


def git_head() -> dict:
    return {"head": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "branch": sh(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
            "code_status": sh(["git", "-C", str(REPO), "status", "--porcelain", "--", "src",
                               "analysis/shell_l2_rlmnist_0913"])}


def f4(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def verdict_rows(res: dict) -> list[dict]:
    rows = []
    for act, r in res["acts"].items():
        for kind, block, names in (("contrast", r["contrasts"], [c[0] for c in CONTRASTS]),
                                   ("contrast_report", r["report_contrasts"], [c[0] for c in REPORT_CONTRASTS])):
            for name in names:
                s = block[name]
                rows.append({"act": act, "kind": kind, "name": name, "a": s["a"], "b": s["b"], "n": s["n"],
                             "mean": s["mean"], "se": s["se"], "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                             "ci975_lo": s["ci975_lo"], "ci975_hi": s["ci975_hi"], "wins": s["wins"],
                             "n_sign": s["n_sign"], "p_sign": s["p_sign"],
                             "class": r["classes"].get(name, ""), "class_bonf": r["classes_bonf"].get(name, "")})
        for kind, g in (("guard", r["guard"]), ("guard_bias_report", r["guard_bias_report"])):
            for w, v in g.items():
                rows.append({"act": act, "kind": kind, "name": w, "shell_median": v["shell_median"],
                             "l2init_median": v["l2init_median"], "ratio": v["ratio"], "pass": v["pass"]})
        rows.append({"act": act, "kind": "guard_summary", "name": "weights_W1_W2_W3", "pass": r["guard_pass"]})
        rows.append({"act": act, "kind": "label", "name": "conclusion", "n": r["contrasts"]["d2"]["n"],
                     "label": r["label"], "reason": r["reason"], "label_bonf": r["label_bonf"],
                     "reason_bonf": r["reason_bonf"], "sentence": r["sentence"],
                     "problems": " | ".join(r["problems"])})
    rows.append({"act": "R+SNA", "kind": "cross_activation", "name": "statement_allowed",
                 "pass": res["cross_activation_statement_allowed"]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, float_format="%.10g")


def summary_md(res: dict, prov: dict, checks: dict | None, ref: dict) -> str:
    L = [f"# {EXPERIMENT} — 判定（Random Label MNIST・Shell 正則化で L2-Init の利得を分解）", "",
         "> 自動生成: `analysis/shell_l2_rlmnist_0913/verdict.py`。spec: `specs/spec_shell_l2_rlmnist_0913.md`。"
         "数値はこのファイルと `verdict.csv` から転記する。", ""]
    run_hashes = sorted({p.get("git_hash") for p in prov["shards"].values()})
    dirty = sorted({str(p.get("git_dirty_code")) for p in prov["shards"].values()})
    env = prov["environment"]
    cpu = next((ln.split(":", 1)[1].strip() for ln in env["lscpu"].splitlines() if ln.startswith("Model name")), "?")
    walls = [p["wall_clock_s"] for p in prov["shards"].values()]
    ndiv = sum(len(p.get("divergences", [])) for p in prov["shards"].values())
    L += ["## 0. 実行したもの", "",
          f"- **本走の commit**: {', '.join(f'`{h}`' for h in run_hashes) or '—'}（run 時の code の未 commit 変更: {', '.join(dirty) or '—'}）",
          f"- 集計時の HEAD: `{prov['git']['head']}`（branch `{prov['git']['branch']}`）",
          f"- 環境: host `{env['hostname']}`・GCP `{env['gcp_instance'].get('machine-type', '?').split('/')[-1]}`"
          f"（`{env['gcp_instance'].get('zone', '?').split('/')[-1]}`）・CPU {cpu}・Python {env['python']}・"
          f"torch {env['packages'].get('torch')}・numpy {env['packages'].get('numpy')}・pandas {env['packages'].get('pandas')}・"
          "各 run 1 スレッド（`torch.set_num_threads(1)`・`OMP_NUM_THREADS=1`）・CPU",
          f"- run: **{len(prov['shards'])}/80 揃い**・欠損 {len(prov['missing'])}・発散 {ndiv}・"
          f"1 run の壁時計 中央値 {np.median(walls) / 60 if walls else float('nan'):.1f} 分（最大 {max(walls) / 60 if walls else float('nan'):.1f} 分）",
          f"- 検査: `checks.json` all_pass = **{checks.get('all_pass') if checks else '（未添付）'}**", ""]
    L += ["## 1. 結論（活性化ごとに独立・spec §4.4 の表を上から適用）", "",
          "| act | ラベル | 理由コード | Bonferroni 版（α=0.025） | 結論 |", "|---|---|---|---|---|"]
    for act, r in res["acts"].items():
        L.append(f"| **{act}** | **{r['label']}** | {r['reason'] or '—'} | {r['label_bonf']} {r['reason_bonf'] or ''} | {r['sentence']} |")
    L += ["", "限定: **Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・50 タスク・seed 0–9・CPU の、この箱の中だけの結論**。"
          "Shell は tensor ごとの半径の拘束で、ユニットごとのノルムや hard projection については何も言わない。", ""]
    allowed = res["cross_activation_statement_allowed"]
    L += [f"- **活性化を跨ぐ文（「R でも SNA でも〜」）**: {'**書いてよい**（両ラベル一致かつ Bonferroni 版でも同じ）' if allowed else '**書かない**（ラベルが違う、または Bonferroni 版で変わる・spec §4.5）'}", ""]
    for act, r in res["acts"].items():
        if r["problems"]:
            L += [f"- {act} の INCOMPLETE の理由: " + "; ".join(r["problems"][:12]) + (" …" if len(r["problems"]) > 12 else "")]
    L += ["", "## 2. 4 腕の成績（窓 = タスク 31–50 の `online_acc`・seed ごとの窓平均を seed 間で要約）", ""]
    for act, r in res["acts"].items():
        L += [f"### {act}", "", "| reg | n | 平均 ± SD | 中央値 | 最小–最大 | memo_acc 平均 |", "|---|---|---|---|---|---|"]
        for reg in REGS:
            p = r["perf"][reg]
            L.append(f"| `{reg}` | {p['n']} | {f4(p['mean'])} ± {f4(p['sd'])} | {f4(p['median'])} | {f4(p['min'])}–{f4(p['max'])} | {f4(p['memo_acc_mean'])} |")
        L.append("")
    L += ["## 3. 対応差（seed 対応・bootstrap は seed の復元抽出 10,000 回）", ""]
    for act, r in res["acts"].items():
        L += [f"### {act}", "", "| 対比 | 平均 ± SE | 95% CI | 97.5% CI | 勝ち/n | 両側 sign p | 分類（α=0.05） | 分類（Bonf） |",
              "|---|---|---|---|---|---|---|---|"]
        for name, a, b in CONTRASTS:
            s = r["contrasts"][name]
            L.append(f"| {name} = `{a}` − `{b}` | {f4(s['mean'], True)} ± {f4(s['se'])} | [{f4(s['ci95_lo'], True)}, {f4(s['ci95_hi'], True)}] | "
                     f"[{f4(s['ci975_lo'], True)}, {f4(s['ci975_hi'], True)}] | {s['wins']}/{s['n_sign']} | {s['p_sign']:.4f} | "
                     f"{r['classes'][name]} | {r['classes_bonf'][name]} |")
        L += ["", f"同等性帯 δ = ±{DELTA}（d2 のみ）。A には d3 の 95% CI 下限 ≥ {2 * DELTA:.3f} も要る（spec §4.4 の解釈 2）。", ""]
    L += ["## 4. norm-match ガード（W1–W3・タスク 31–50 × seed 0–9 のタスク末 ‖p‖/‖p0‖ の中央値の比 ∈ [0.9, 1.1]）", ""]
    for act, r in res["acts"].items():
        L += [f"### {act}: **{'PASS' if r['guard_pass'] else 'FAIL'}**", "",
              "| tensor | M(shell) | M(l2init) | 比 | 判定 |", "|---|---|---|---|---|"]
        for w, v in r["guard"].items():
            L.append(f"| {w} | {f4(v['shell_median'])} | {f4(v['l2init_median'])} | {f4(v['ratio'])} | {'PASS' if v['pass'] else 'FAIL'} |")
        for w, v in r["guard_bias_report"].items():
            L.append(f"| {w}（REPORT） | {f4(v['shell_median'])} | {f4(v['l2init_median'])} | {f4(v['ratio'])} | ({'in' if v['pass'] else 'out'}) |")
        L.append("")
    L += ["## 5. REPORT_ONLY（判定に使わない）", "", "### 5.1 `none` との対比", "",
          "| act | 対比 | 平均 ± SE | 95% CI | 勝ち/n | p |", "|---|---|---|---|---|---|"]
    for act, r in res["acts"].items():
        for name, a, b in REPORT_CONTRASTS:
            s = r["report_contrasts"][name]
            L.append(f"| {act} | `{a}` − `{b}` | {f4(s['mean'], True)} ± {f4(s['se'])} | [{f4(s['ci95_lo'], True)}, {f4(s['ci95_hi'], True)}] | {s['wins']}/{s['n_sign']} | {s['p_sign']:.4f} |")
    L += ["", "### 5.2 shell が取り戻した L2-Init の利得の割合 f = d1 / d3", ""]
    for act, r in res["acts"].items():
        f = r["recovered_fraction_report"]
        L.append(f"- {act}: f = {f4(f['point'])}（95% CI [{f4(f['ci95_lo'])}, {f4(f['ci95_hi'])}]・d3 が小さいと不安定）")
    L += ["", "### 5.3 tensor ごとの窓内中央値（seed × タスク 31–50 をまとめた中央値）", "",
          "| act | reg | tensor | ‖p‖/‖p0‖ | cos(p,p0) | 自腕の罰則 | L2-Init 罰則に占める半径成分 |", "|---|---|---|---|---|---|---|"]
    for act, r in res["acts"].items():
        for reg in REGS:
            for w in TENSORS:
                v = r["tensor_medians_report"].get(f"{reg}|{w}")
                if v:
                    L.append(f"| {act} | `{reg}` | {w} | {f4(v['norm_ratio'])} | {f4(v['cos_w0'])} | {v['reg_loss']:.3e} | {f4(v['radial_share_of_l2init_pen'])} |")
    L += ["", f"### 5.4 ε の不活性（全記録点の sq_norm の最小 > {EPS_INERT_SQ:.3g} なら ε² は 1 bit も効いていない）", ""]
    for act, r in res["acts"].items():
        for reg, v in r["min_sq_norm_report"].items():
            L.append(f"- {act} `{reg}`: min S = {v['min_sq_norm']:.4g}（{v['tensor']}・task {v['task']}・seed {v['seed']}）→ {'不活性' if v['above_eps_inert_threshold'] else '**閾値以下に落ちた**'}")
    L += ["", "### 5.5 0906（CUDA・別の走）の窓平均 — 参考。bit 一致は求めず、主判定に使わない", "",
          "| act | reg | 0906 平均（n） | 本走 平均（n） |", "|---|---|---|---|"]
    for act, r in res["acts"].items():
        for reg in REGS:
            v = ref.get(f"{act}|{reg}")
            if v:
                L.append(f"| {act} | `{reg}` | {f4(v['mean'])}（{v['n']}） | {f4(r['perf'][reg]['mean'])}（{r['perf'][reg]['n']}） |")
    if checks:
        L += ["", "## 6. 検査（`checks.json`）", ""]
        for k, v in checks.items():
            if isinstance(v, dict) and "pass" in v:
                muts = v.get("mutations", [])
                det = sum(1 for m in muts if m.get("detected"))
                L.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}" + (f"・mutation {det}/{len(muts)} 検出" if muts else ""))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the {act}_{reg}_s{seed} shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", default=None, help="checks.json to copy into --out and cite")
    ap.add_argument("--ref0906", default=str(REPO / "results" / "pmnist_rlmnist_0906"))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pt, lm, provs, missing = load(src)
    res = analyze(pt, lm, provs, missing)

    order = {"act": list(ACTS), "reg": list(REGS)}
    for df in (pt, lm):
        for c, cats in order.items():
            if c in df:
                df[c] = pd.Categorical(df[c], categories=cats, ordered=True)
    pt = pt.sort_values(["act", "reg", "seed", "task"]).astype({"act": str, "reg": str})
    lm = lm.assign(_t=pd.Categorical(lm["tensor"], categories=list(TENSORS), ordered=True)) \
           .sort_values(["act", "reg", "seed", "task", "_t"]).drop(columns="_t").astype({"act": str, "reg": str})
    pt.to_csv(out / "per_task.csv", index=False, float_format="%.10g")
    lm.to_csv(out / "layer_metrics.csv", index=False, float_format="%.10g")
    write_csv(out / "verdict.csv", verdict_rows(res))

    checks = None
    if args.checks:
        cp = Path(args.checks)
        if cp.exists():
            checks = json.loads(cp.read_text())
            if cp.resolve() != (out / "checks.json").resolve():
                shutil.copyfile(cp, out / "checks.json")
    shards = {}
    for name, p in provs.items():
        runs = p.get("runs", {})
        shards[name] = {k: p.get(k) for k in ("git_hash", "git_dirty_code", "argv", "hostname", "torch",
                                              "torch_num_threads", "env_threads", "wall_clock_s",
                                              "divergences", "code_sha256", "data_sha256")}
        shards[name]["runs"] = runs
    prov = {"run_id": EXPERIMENT, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "src": str(src), "git": git_head(), "environment": environment(),
            "spec": "specs/spec_shell_l2_rlmnist_0913.md",
            "judgement": {"window": list(WIN), "delta": DELTA, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
                          "guard": [GUARD_LO, GUARD_HI], "guard_tensors": list(WEIGHTS)},
            "missing": missing, "shards": shards,
            "checks_sha256": hashlib.sha256((out / "checks.json").read_bytes()).hexdigest()
            if (out / "checks.json").exists() else None,
            "labels": {a: {"label": r["label"], "reason": r["reason"], "label_bonf": r["label_bonf"]}
                       for a, r in res["acts"].items()}}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    ref = ref_0906(Path(args.ref0906)) if args.ref0906 != "none" else {}
    (out / "summary.md").write_text(summary_md(res, prov, checks, ref))
    for a, r in res["acts"].items():
        print(f"{a}: label {r['label']} {r['reason']}  (bonf {r['label_bonf']} {r['reason_bonf']})  "
              f"guard {'PASS' if r['guard_pass'] else 'FAIL'}  problems {len(r['problems'])}")
    print(f"wrote {out}/{{per_task.csv, layer_metrics.csv, verdict.csv, summary.md, provenance.json}}")


if __name__ == "__main__":
    main()
