#!/usr/bin/env python3
"""chargpt_cl_1019 の本走前の検査（spec §8 と追補 0 §G）。

  python src/chargpt_cl_1019_checks.py --out results/chargpt_cl_1019/checks

各検査は変異対照（壊した版が落ちること）を持つ。結果は checks.json と checks.md に書いてから最後に判定する。
"""
import argparse
import copy
import filecmp
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019 as C  # noqa: E402

PY = sys.executable
SRC = Path(__file__).resolve().parent / "chargpt_cl_1019.py"
RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(name=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, ensure_ascii=False, default=float)[:400]}",
          flush=True)


# ---------------------------------------------------------------- S-entropy
def s_entropy(dev):
    rows, worst = [], 0.0
    ok_all = True
    for alpha in (0.01, 0.03, 0.1):
        for seed in (0, 1):
            for task in (1, 2):
                ch = C.ChainM(seed, task, alpha)
                g = torch.Generator().manual_seed(C.stream_seed(seed, "check", task, "entropy"))
                seq = ch.sample(8192, g)
                ce = ch.true_ce(seq).double()
                m, se = float(ce.mean()), float(ce.std() / math.sqrt(len(ce)))
                d = m - ch.H
                ok = abs(d) < 0.01 and se <= 0.0025
                ok_all &= ok
                worst = max(worst, abs(d))
                rows.append(dict(alpha=alpha, seed=seed, task=task, H=ch.H, ce=m, diff=d, se=se,
                                 pi_iters=ch.pi_iters, rowsum_err=ch.rowsum_err))
    # 変異対照: 文脈を 1 つずらした予測（壊れた評価）は帯を大きく外れる
    ch = C.ChainM(0, 1, 0.03)
    g = torch.Generator().manual_seed(C.stream_seed(0, "check", 1, "entropy_mut"))
    seq = ch.sample(8192, g)
    ctx = seq[:, :-3] * C.V + seq[:, 1:-2]          # 1 つ古い文脈 (s_{j-3}, s_{j-2}) で s_j を予測
    ce_mut = float(-ch.logP[ctx, seq[:, 3:]].mean())
    mut_fails = abs(ce_mut - ch.H) >= 0.01
    record("S-entropy", ok_all and mut_fails, worst_abs_diff=worst, max_se=max(r["se"] for r in rows),
           mutant_shifted_ctx_diff=ce_mut - ch.H, rows=rows)


# ---------------------------------------------------------------- S-rule
def s_rule(dev, boxes=("M", "W")):
    info = {}
    ok = True
    # 箱 M: 行の和・課題間の独立
    for alpha in (0.01, 0.03, 0.1):
        P = [C.ChainM(0, t, alpha).P for t in (1, 2, 3)]
        rs = max(float(np.abs(p.sum(2) - 1).max()) for p in P)
        r12 = float(np.corrcoef(P[0].ravel(), P[1].ravel())[0, 1])
        r23 = float(np.corrcoef(P[1].ravel(), P[2].ravel())[0, 1])
        band = 4 / math.sqrt(P[0].size)
        o = rs < 1e-13 and abs(r12) <= band and abs(r23) <= band
        ok &= o
        info[f"M_alpha{alpha}"] = dict(rowsum_err=rs, r12=r12, r23=r23, band=band, ok=o)
    # 変異対照: 同じ規則を使い回すと相関 1 で落ちる
    Pa, Pb = C.ChainM(0, 1, 0.03).P, C.ChainM(0, 1, 0.03).P
    mut_r = float(np.corrcoef(Pa.ravel(), Pb.ravel())[0, 1])
    info["M_mutant_reuse_r"] = mut_r
    ok &= mut_r > 4 / math.sqrt(Pa.size)
    if "W" not in boxes:
        record("S-rule(generators)", ok, **info)
        return
    # 箱 W: 窓・ラベルの一様性（Bonferroni）・課題間の独立
    W = C.WindowsW(0, dev)
    st = W.starts.sort().values
    n90 = int(0.9 * 1115394)
    o_win = bool((st[1:] - st[:-1] >= C.T).all()) and int(st.max()) + C.T <= n90 and len(set(st.tolist())) == C.W_N_WIN
    info["W_windows"] = dict(nonoverlap_in_first90=o_win, offset=W.offset)
    ok &= o_win
    pmin, eq_dev = 1.0, 0.0
    labs = [W.labels(t).cpu() for t in range(1, 51)]
    for lb in labs:
        cnt = torch.bincount(lb, minlength=10).double()
        chi = float(((cnt - 120) ** 2 / 120).sum())
        pmin = min(pmin, chi2_sf(chi, 9))
    band_eq = 4 * math.sqrt(0.09 / C.W_N_WIN)
    for a, b in zip(labs[:-1], labs[1:]):
        eq_dev = max(eq_dev, abs(float((a == b).double().mean()) - 0.1))
    o_lab = pmin > 0.01 / 50 and eq_dev <= band_eq
    info["W_labels"] = dict(min_p=pmin, bonferroni=0.01 / 50, max_eq_dev=eq_dev, band=band_eq, ok=o_lab)
    ok &= o_lab
    # 変異対照: 9 値しか出ないラベルは χ² で落ち、使い回しは一致率で落ちる
    g = torch.Generator().manual_seed(1)
    bad = torch.randint(0, 9, (C.W_N_WIN,), generator=g)
    cnt = torch.bincount(bad, minlength=10).double()
    p_bad = chi2_sf(float(((cnt - 120) ** 2 / 120).sum()), 9)
    eq_bad = abs(float((labs[0] == labs[0]).double().mean()) - 0.1)
    info["W_mutants"] = dict(p_9values=p_bad, eq_dev_reuse=eq_bad)
    ok &= p_bad <= 0.01 / 50 and eq_bad > band_eq
    record("S-rule(generators)", ok, **info)


def chi2_sf(x, k):
    """χ² の上側確率（k は奇数でも可）。正則化不完全ガンマ Q(k/2, x/2) を級数で。"""
    a, z = k / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    # P(a,z) の級数
    term = 1.0 / a
    s = term
    n = 0
    while True:
        n += 1
        term *= z / (a + n)
        s += term
        if term < s * 1e-15 or n > 10000:
            break
    P = s * math.exp(-z + a * math.log(z) - math.lgamma(a))
    return max(0.0, 1.0 - P)


# ---------------------------------------------------------------- S-init（腕の間で共有する初期値）
def s_init():
    sds = {arm: C.build_model(arm, "W", 0, "cpu").state_dict() for arm in C.ARMS}
    common = set.intersection(*[set(sd) for sd in sds.values()])
    diff = [k for k in common if not all(torch.equal(sds["C3"][k], sds[a][k]) for a in C.ARMS)]
    other = C.build_model("C3", "W", 1, "cpu").state_dict()
    mut_same = torch.equal(sds["C3"]["h.0.mlp.up.weight"], other["h.0.mlp.up.weight"])
    record("S-init(shared across arms)", not diff and not mut_same, n_common=len(common), differing=diff,
           mutant_other_seed_identical=mut_same)


# ---------------------------------------------------------------- S-engine
def flat_grads(model):
    return torch.cat([p.grad.detach().double().reshape(-1) for p in model.parameters()])


def s_engine(dev, boxes=("M", "W")):
    rows, ok = [], True
    for box in boxes:
        W = C.WindowsW(0, dev) if box == "W" else None
        ctx = C.TaskCtx(box, 0, 1, 20, dev, alpha=0.03, W=W)
        for arm in ("C3", "B3", "N4"):
            x, y = ctx.batch(5)
            x, y = x.long(), y.long()
            m64 = C.build_model(arm, box, 0, dev).double()
            out = m64(x)
            loss = F.cross_entropy(out[:, 1:].reshape(-1, C.V), y[:, 1:].reshape(-1)) if box == "M" else F.cross_entropy(out, y)
            loss.backward()
            g64 = flat_grads(m64)
            me = C.build_model(arm, box, 0, dev)
            ee = C.Engine(me, arm, box, graph=False)
            bl, ba = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
            ee.step(x, y, bl, ba, 0)
            ge = flat_grads(me)
            mg = C.build_model(arm, box, 0, dev)
            eg = C.Engine(mg, arm, box, graph=True)
            bl2, ba2 = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
            eg.step(x, y, bl2, ba2, 0)
            gg = flat_grads(mg)
            err_e = float((ge - g64).norm() / g64.norm())
            err_g = float((gg - g64).norm() / g64.norm())
            o = err_g <= 4 * err_e
            ok &= o
            rows.append(dict(box=box, arm=arm, loss64=float(loss), loss_eager=float(bl), loss_graph=float(bl2),
                             grad_relerr_eager=err_e, grad_relerr_graph=err_g, ok=o))
            # 変異対照: 1 つ先の batch で取った勾配は帯を大きく外れる
            if arm == "C3":
                mm = C.build_model(arm, box, 0, dev)
                em = C.Engine(mm, arm, box, graph=False)
                xm, ym = ctx.batch(6)
                em.step(xm.long(), ym.long(), bl, ba, 0)
                err_m = float((flat_grads(mm) - g64).norm() / g64.norm())
                rows[-1]["mutant_other_batch_relerr"] = err_m
                ok &= err_m > 4 * err_e
    record("S-engine(graph = eager)", ok, rows=rows)


# ---------------------------------------------------------------- S-online
def s_online(dev, boxes=("M", "W")):
    rows, ok = [], True
    for box in boxes:
        W = C.WindowsW(0, dev) if box == "W" else None
        ctx = C.TaskCtx(box, 0, 1, 60, dev, alpha=0.03, W=W)
        m = C.build_model("C3", box, 0, dev)
        eng = C.Engine(m, "C3", box, graph=True)
        bl, ba = torch.zeros(60, device=dev), torch.zeros(60, device=dev)
        for u in range(50):
            eng.set_lr(eng.base_lr * min(1.0, (u + 1) / C.WARMUP))
            x, y = ctx.batch(u)
            eng.step(x, y, bl, ba, u)

        def eager_loss(model, u):
            x, y = ctx.batch(u)
            x, y = x.long(), y.long()
            with torch.no_grad():
                out = model(x)
                if box == "M":
                    return float(F.cross_entropy(out[:, 1:].reshape(-1, C.V), y[:, 1:].reshape(-1))), float("nan")
                return float(F.cross_entropy(out, y)), float((out.argmax(1) == y).float().mean())

        before, acc_b = eager_loss(m, 50)
        eng.set_lr(eng.base_lr * min(1.0, 51 / C.WARMUP))
        x, y = ctx.batch(50)
        eng.step(x, y, bl, ba, 50)
        after, _ = eager_loss(m, 50)
        rec = float(bl[50])
        gap = abs(after - before)
        o = abs(rec - before) <= 0.01 * gap and gap > 0
        ok &= o
        rows.append(dict(box=box, recorded=rec, eager_before=before, eager_after=after,
                         recorded_acc=float(ba[50]), eager_acc_before=acc_b, ok=o))
    record("S-online(pre-update metric)", ok, rows=rows)


# ---------------------------------------------------------------- 走を使う検査（S-repro・S-inert・S-rule(arms)・S-probe・S-decomp）
def run_cli(out, box, arm, n, tasks=2, extra=()):
    cmd = [PY, str(SRC), "run", "--box", box, "--arm", arm, "--seed", "0", "--tasks", str(tasks),
           "--n_per_task", str(n), "--out", str(out), "--ckpt", "0"] + (["--alpha", "0.03"] if box == "M" else []) + list(extra)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise RuntimeError(f"run failed: {cmd}")
    return time.time() - t0


def compare_dirs(a, b):
    files = sorted(f.name for f in Path(a).iterdir() if f.is_file() and f.name != "provenance.json")
    diff = [f for f in files if not filecmp.cmp(Path(a) / f, Path(b) / f, shallow=False)]
    pa = json.loads((Path(a) / "provenance.json").read_text())
    pb = json.loads((Path(b) / "provenance.json").read_text())
    for k in ("wall_sec", "task_secs", "argv"):
        pa.pop(k, None)
        pb.pop(k, None)
    if pa != pb:
        diff.append("provenance.json(non-timing)")
    return files, diff


def s_runs(base, dev, boxes=("M", "W")):
    base = Path(base)
    out = {}
    n = 400
    for box in boxes:
        # S-repro: 同じ走を 2 回（fresh 込み）
        a, b = base / f"repro_{box}_a", base / f"repro_{box}_b"
        sec = run_cli(a, box, "C3", n, extra=["--fresh_tasks", "2"])
        run_cli(b, box, "C3", n, extra=["--fresh_tasks", "2"])
        files, diff = compare_dirs(a, b)
        record(f"S-repro({box})", not diff and len(files) >= 7, files=files, differing=diff, wall_sec=sec)
        # S-inert: 測定なしの走と最終重みがビット一致
        c = base / f"inert_{box}_nomeasure"
        run_cli(c, box, "C3", n, extra=["--fresh_tasks", "--no_measure"])
        pa = json.loads((a / "provenance.json").read_text())
        pc = json.loads((c / "provenance.json").read_text())
        # 変異対照: 1 更新だけ少ない走は一致しない
        d = base / f"inert_{box}_short"
        run_cli(d, box, "C3", n - 1, extra=["--fresh_tasks", "--no_measure"])
        pd = json.loads((d / "provenance.json").read_text())
        record(f"S-inert({box})", pa["final_sha"] == pc["final_sha"] and pd["final_sha"] != pa["final_sha"],
               with_measure=pa["final_sha"][:16], without=pc["final_sha"][:16], mutant_short=pd["final_sha"][:16])
        # S-rule(arms)・S-probe: 腕の間で規則・batch 列・probe がビット一致
        arms_info = {}
        for arm in ("C4", "B3", "N4"):
            e = base / f"arm_{box}_{arm}"
            run_cli(e, box, arm, n, extra=["--fresh_tasks", "2"])
            arms_info[arm] = json.loads((e / "provenance.json").read_text())
        keys = ("data_sha", "rule_sha", "probe_sha")
        same = all(ti[k] == tj[k] for arm in arms_info for ti, tj in zip(pa["tasks_info"], arms_info[arm]["tasks_info"]) for k in keys)
        if box == "W":
            same &= all(arms_info[arm]["W_window_starts"] == pa["W_window_starts"] and arms_info[arm]["W_probe_idx"] == pa["W_probe_idx"] for arm in arms_info)
        else:
            same &= all(arms_info[arm]["probe_uniform_sha"] == pa["probe_uniform_sha"] for arm in arms_info)
        diff_tasks = pa["tasks_info"][0]["data_sha"] != pa["tasks_info"][1]["data_sha"] and pa["tasks_info"][0]["probe_sha"] != pa["tasks_info"][1]["probe_sha"] if box == "M" else pa["tasks_info"][0]["data_sha"] != pa["tasks_info"][1]["data_sha"]
        # fresh は継続と同じ batch 列
        import csv
        fr = list(csv.DictReader(open(a / "fresh.csv")))
        pt = list(csv.DictReader(open(a / "per_task.csv")))
        fresh_same = fr[0]["data_sha"] == pt[1]["data_sha"]
        record(f"S-rule+S-probe(arms,{box})", same and diff_tasks and fresh_same, arms=list(arms_info),
               tasks_differ=diff_tasks, fresh_uses_same_batches=fresh_same)
        # S-decomp: 全走の全時点で分解の誤差
        errs = {}
        for dd in [a] + [base / f"arm_{box}_{arm}" for arm in arms_info]:
            ls = list(csv.DictReader(open(dd / "layer_stats.csv")))
            errs[dd.name] = max(float(r["decomp_err"]) for r in ls)
        out[box] = errs
        record(f"S-decomp({box})", max(errs.values()) < 1e-5, max_err=errs)
    return out


def s_decomp_mutant(dev):
    """B3 を少し学ばせ、β の項を落とした分解が帯を外れることを確かめる。"""
    W = None
    ctx = C.TaskCtx("M", 0, 1, 300, dev, alpha=0.03)
    m = C.build_model("B3", "M", 0, dev)
    eng = C.Engine(m, "B3", "M", graph=True)
    bl, ba = torch.zeros(300, device=dev), torch.zeros(300, device=dev)
    for u in range(300):
        eng.set_lr(eng.base_lr * min(1.0, (u + 1) / C.WARMUP))
        x, y = ctx.batch(u)
        eng.step(x, y, bl, ba, u)
    res = C.measure(m, ctx.probe)
    full = max(r["agg"]["decomp_err"] for r in res)
    mut = max(float(np.abs(r["units"]["mu"].astype(np.float64) - r["units"]["t_wgm"] - r["units"]["t_b"]).max()) for r in res)
    record("S-decomp(mutant: drop beta term)", full < 1e-5 and mut > 1e-5, full_err=full, mutant_err=mut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boxes", nargs="+", default=["M", "W"], choices=["M", "W"])
    a = ap.parse_args()
    bx = tuple(a.boxes)
    C.setup_torch()
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fns = ([lambda: s_entropy(dev)] if "M" in bx else []) + [lambda: s_rule(dev, bx), s_init, lambda: s_engine(dev, bx),
           lambda: s_online(dev, bx)] + ([lambda: s_decomp_mutant(dev)] if "M" in bx else []) + [lambda: s_runs(out / "runs", dev, bx)]
    for fn in fns:
        try:
            fn()
        except Exception as e:  # 検査そのものが壊れたら FAIL として残す
            import traceback
            record(f"ERROR in {getattr(fn, '__name__', 'check')}", False, error=repr(e), tb=traceback.format_exc()[-1500:])
    gh, dirty = C.git_info()
    summary = dict(run_id=C.RUN_ID, git_hash=gh, dirty_src=dirty, torch=torch.__version__, boxes=list(bx),
                   host=__import__("socket").gethostname(),
                   device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                   wall_sec=time.time() - t0, all_pass=all(r["ok"] for r in RESULTS), results=RESULTS)
    (out / "checks.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    lines = ["# chargpt_cl_1019 本走前の検査", "", f"git {gh}・{summary['device']}・{summary['wall_sec']:.0f} 秒", "",
             "| 検査 | 結果 |", "| -- | -- |"]
    lines += [f"| {r['name']} | {'PASS' if r['ok'] else 'FAIL'} |" for r in RESULTS]
    (out / "checks.md").write_text("\n".join(lines) + "\n")
    print("ALL PASS" if summary["all_pass"] else "SOME FAIL", flush=True)
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
