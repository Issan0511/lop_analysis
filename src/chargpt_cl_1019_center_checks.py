#!/usr/bin/env python3
"""chargpt_cl_1019 の追加腕（C3H・B3H・B3HB）の検査。各検査に変異対照を付ける。

  python src/chargpt_cl_1019_center_checks.py --out results/chargpt_cl_1019/center_checks
"""
import argparse
import csv
import filecmp
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019_center as CC  # noqa: E402  （本走のモジュールに腕と部品を足す）
C = CC.C

PY = sys.executable
SRC = Path(__file__).resolve().parent / "chargpt_cl_1019_center.py"
SRC_BASE = Path(__file__).resolve().parent / "chargpt_cl_1019.py"
ARMS = ("C3H", "B3H", "B3HB")
RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(name=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, ensure_ascii=False, default=float)[:300]}", flush=True)


def grads(model):
    return torch.cat([p.grad.detach().double().reshape(-1) for p in model.parameters()])


def bufs(model):
    return torch.cat([b.detach().double().reshape(-1) for n, b in model.named_buffers() if "ema" in n])


def s_engine(dev, ctx):
    rows, ok = [], True
    for arm in ARMS:
        x, y = ctx.batch(3)
        me = C.build_model(arm, "M", 0, dev)
        ee = C.Engine(me, arm, "M", graph=False)
        mg = C.build_model(arm, "M", 0, dev)
        eg = C.Engine(mg, arm, "M", graph=True)
        b1, b2 = torch.zeros(4, device=dev), torch.zeros(4, device=dev)
        for u in range(3):                          # 3 歩：EMA が動いた状態の勾配も比べる
            xb, yb = ctx.batch(u)
            ee.step(xb, yb, b1, b1.clone(), u)
            eg.step(xb, yb, b2, b2.clone(), u)
        dg = float((grads(me) - grads(mg)).abs().max())
        db = float((bufs(me) - bufs(mg)).abs().max())
        dl = float((b1[:3] - b2[:3]).abs().max())
        o = dg == 0.0 and db == 0.0 and dl == 0.0
        # 変異対照: 1 歩少ない eager とは buffer が一致しない
        mm = C.build_model(arm, "M", 0, dev)
        em = C.Engine(mm, arm, "M", graph=False)
        for u in range(2):
            xb, yb = ctx.batch(u)
            em.step(xb, yb, b1, b1.clone(), u)
        mut = float((bufs(mm) - bufs(mg)).abs().max())
        ok &= o and mut > 0
        rows.append(dict(arm=arm, grad_maxdiff=dg, buf_maxdiff=db, loss_maxdiff=dl, mutant_buf_diff=mut))
    record("S-engine-c(graph = eager, 3 steps incl. EMA)", ok, rows=rows)


def s_ema(dev, ctx):
    """EMA の m/c が、学習の forward の y のバッチ平均の偏り補正つき EMA と一致する。"""
    arm = "C3H"
    m = C.build_model(arm, "M", 0, dev)
    eng = C.Engine(m, arm, "M", graph=False)
    beta = CC.EMA_BETA
    ref = [torch.zeros(C.D, dtype=torch.float64, device=dev) for _ in range(C.L)]
    ok, errs, muts = True, [], []
    b = torch.zeros(6, device=dev)
    for u in range(5):
        x, y = ctx.batch(u)
        # 更新前の重みでの y を、学習の forward と同じ中心化のもとで取る（測定なので EMA は動かない）
        cap = []
        with torch.no_grad():
            m.body(x.long(), cap)
            ys = [m.h[li].n2(cap[2 * li]).double().mean((0, 1)) for li in range(C.L)]
        for li in range(C.L):
            ref[li] = (1 - beta) * ref[li] + beta * ys[li]
        eng.step(x, y, b, b.clone(), u)
        c = 1 - (1 - beta) ** (u + 1)
        for li in range(C.L):
            mv = m.h[li].mlp.mean_vec().double()
            errs.append(float(((mv - ref[li] / c).norm() / (ref[li] / c).norm())))
            muts.append(float(((mv - ref[li]).norm() / (ref[li] / c).norm())))   # 偏り補正を落とした変異
    ok = max(errs) < 1e-5 and min(muts[:C.L]) > 1e-2
    record("S-ema(m equals bias-corrected EMA of batch means)", ok, max_relerr=max(errs), mutant_no_correction_min=min(muts[:C.L]))


def s_causal(dev, ctx):
    """学習モードの forward で、位置 j の入力を変えても j より前の logits は変わらない。"""
    arm = "C3H"
    m = C.build_model(arm, "M", 0, dev)
    eng = C.Engine(m, arm, "M", graph=False)
    b = torch.zeros(4, device=dev)
    for u in range(3):
        x, y = ctx.batch(u)
        eng.step(x, y, b, b.clone(), u)
    x, _ = ctx.batch(4)
    x = x.long().clone()
    j = 64
    x2 = x.clone()
    x2[:, j] = (x2[:, j] + 1) % C.V
    snap = [bb.detach().clone() for n, bb in m.named_buffers() if "ema" in n]
    l1 = m(x)                                 # 勾配が有効：学習の forward と同じ経路（EMA も更新される）
    for bb, q in zip([bb for n, bb in m.named_buffers() if "ema" in n], snap):
        bb.copy_(q)
    l2 = m(x2)
    d_before = float((l1[:, :j] - l2[:, :j]).abs().max())
    d_after = float((l1[:, j:] - l2[:, j:]).abs().max())
    # 変異対照: そのバッチ自身の平均で中心化すると、前の位置にも未来が漏れる
    def leaky(xx):
        h = m.wte(xx)
        for blk in m.h:
            h = h + blk.sc * blk.attn(blk.n1(h))
            yv = blk.n2(h)
            z = blk.mlp.up(yv - yv.mean((0, 1), keepdim=True))
            h = h + blk.mlp.down(F.relu(z))
        return m.head(m.nf(h))
    with torch.no_grad():
        dl = float((leaky(x)[:, :j] - leaky(x2)[:, :j]).abs().max())
    record("S-causal(no future leak in training-mode forward)", d_before == 0.0 and d_after > 0 and dl > 0,
           before_j=d_before, at_or_after_j=d_after, mutant_batchmean_before_j=dl)


def run_cli(src, out, arm, n, extra=()):
    cmd = [PY, str(src), "run", "--box", "M", "--arm", arm, "--seed", "0", "--tasks", "2", "--n_per_task", str(n),
           "--alpha", "0.01", "--out", str(out), "--fresh_tasks", "2"] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise RuntimeError(cmd)


def s_runs(base):
    base = Path(base)
    for arm in ARMS:
        a, b = base / f"{arm}_a", base / f"{arm}_b"
        run_cli(SRC, a, arm, 300)
        run_cli(SRC, b, arm, 300)
        files = sorted(f.name for f in a.iterdir() if f.is_file() and f.name != "provenance.json")
        diff = [f for f in files if not filecmp.cmp(a / f, b / f, shallow=False)]
        record(f"S-repro-c({arm})", not diff and len(files) >= 7, differing=diff)
        c = base / f"{arm}_nomeasure"
        run_cli(SRC, c, arm, 300, ["--no_measure", "--within_tasks"])
        pa = json.loads((a / "provenance.json").read_text())
        pc = json.loads((c / "provenance.json").read_text())
        record(f"S-inert-c({arm})", pa["final_sha"] == pc["final_sha"], with_measure=pa["final_sha"][:16], without=pc["final_sha"][:16])
        ls = list(csv.DictReader(open(a / "layer_stats.csv")))
        err = max(float(r["decomp_err"]) for r in ls)
        lag = max(float(r["center_lag_rel"]) for r in ls if int(r["u"]) > 0)
        adam = sorted(p.name for p in (a / "ckpts").glob("*_adam.pt"))
        st = torch.load(a / "ckpts" / "task1_adam.pt", map_location="cpu")
        steps = {float(v["step"]) for v in st["state"].values()}
        record(f"S-decomp-c+S-adam({arm})", err < 1e-5 and adam == ["task1_adam.pt"] and steps == {300.0},
               decomp_maxerr=err, max_lag_rel=lag, adam_files=adam, adam_steps=sorted(steps))
    # 変異対照（分解）: 中心化の項を落とすと帯を外れる
    m = C.build_model("C3H", "M", 0, "cuda")
    m.load_state_dict(torch.load(base / "C3H_a" / "ckpts" / "task1.pt", map_location="cuda"))
    probe = C.task_cpu_M(0, 2, 10, 0.01)["probe"].cuda()
    res = C.measure(m, probe)
    mut = max(float(abs(r["units"]["mu"].astype(float) - r["units"]["t_wgm"] - r["units"]["t_wbeta"] - r["units"]["t_b"]).max()) for r in res)
    record("S-decomp-c(mutant: drop centering term)", mut > 1e-5, mutant_err=mut)
    # 本走の腕（C3）は、このモジュールを通しても学習が変わらない（最終の重み・成績が同じ）
    d1, d2 = base / "C3_via_center", base / "C3_via_base"
    run_cli(SRC, d1, "C3", 200)
    run_cli(SRC_BASE, d2, "C3", 200)
    p1 = json.loads((d1 / "provenance.json").read_text())
    p2 = json.loads((d2 / "provenance.json").read_text())
    same_pt = filecmp.cmp(d1 / "per_task.csv", d2 / "per_task.csv", shallow=False)
    record("S-base-unchanged(C3 through the centering module)", p1["final_sha"] == p2["final_sha"] and same_pt,
           final_equal=p1["final_sha"] == p2["final_sha"], per_task_equal=same_pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C.setup_torch()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = C.TaskCtx("M", 0, 1, 10, dev, alpha=0.01)
    for fn in (lambda: s_engine(dev, ctx), lambda: s_ema(dev, ctx), lambda: s_causal(dev, ctx), lambda: s_runs(out / "runs")):
        try:
            fn()
        except Exception as e:
            import traceback
            record("ERROR", False, error=repr(e), tb=traceback.format_exc()[-1500:])
    gh, dirty = C.git_info()
    summ = dict(git_hash=gh, dirty_src=dirty, wall_sec=time.time() - t0, all_pass=all(r["ok"] for r in RESULTS), results=RESULTS)
    (out / "checks.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))
    print("ALL PASS" if summ["all_pass"] else "SOME FAIL")
    sys.exit(0 if summ["all_pass"] else 1)


if __name__ == "__main__":
    main()
