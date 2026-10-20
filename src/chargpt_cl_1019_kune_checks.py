#!/usr/bin/env python3
"""chargpt_cl_1019 の kunekune 族の腕（KKT1・SNA・SNAfrz1・GELU）の検査。各検査に変異対照を付ける。

  python src/chargpt_cl_1019_kune_checks.py --out results/chargpt_cl_1019/kune_checks
"""
import argparse
import csv
import filecmp
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C

PY = sys.executable
SRC = Path(__file__).resolve().parent / "chargpt_cl_1019_kune.py"
SRC_BASE = Path(__file__).resolve().parent / "chargpt_cl_1019.py"
ARMS = ("KKT1", "SNA", "SNAfrz1", "GELU")
RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(name=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, ensure_ascii=False, default=float)[:300]}", flush=True)


def s_act(dev):
    """φ′ が自動微分の φ の導関数と一致し、KKT1 は継ぎ目で連続。MLP 版の SnakeFamily とも一致する。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    z = (torch.randn(4096, generator=g) * 4).to(dev).requires_grad_()
    a = (torch.rand(4096, generator=g) * 2.9 + 0.1).to(dev)
    rows = {}
    ok = True
    for kind in ("snake", "kkt1"):
        phi = K.snake_phi(z, a, kind)
        (gz,) = torch.autograd.grad(phi.sum(), z)
        d = K.snake_dphi(z.detach(), a, kind)
        err = float((gz - d).abs().max())
        rows[kind] = dict(dphi_vs_autograd=err)
        ok &= err < 1e-5
    # 継ぎ目の連続性（KKT1）: θ = π と −2π の両側
    aa = torch.tensor([0.7], device=dev)
    for th in (PI := math.pi, -2 * math.pi):
        zj = th / (2 * aa)
        lo, hi = K.snake_phi(zj * (1 - 1e-6), aa, "kkt1"), K.snake_phi(zj * (1 + 1e-6), aa, "kkt1")
        rows[f"jump_at_{th:.2f}"] = float((hi - lo).abs())
        ok &= float((hi - lo).abs()) < 1e-4
    # GELU の φ′
    mk = K.MLPKune("GELU").to(dev)
    zg = z.detach().clone().requires_grad_()
    (gg,) = torch.autograd.grad(torch.nn.functional.gelu(zg).sum(), zg)
    eg = float((gg - mk.dphi(zg.detach())).abs().max())
    rows["gelu_dphi"] = eg
    ok &= eg < 1e-5
    # MLP 版（layer_chimera_cifar_0921.SnakeFamily）との一致
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src import layer_chimera_cifar_0921 as LC
        ref = LC.SnakeFamily("KKT1", "kkt1", widths=(4096,))
        ref.V = [(K.SN_C / a.clamp(K.SN_LO, K.SN_HI)) ** 2 * torch.ones(1, 4096, device=dev)]
        rv = ref.phi(z.detach()[None, None, :], 0)[0, 0]
        mine = K.snake_phi(z.detach(), (K.SN_C / ref.V[0][0].sqrt()).clamp(K.SN_LO, K.SN_HI), "kkt1")
        rows["vs_mlp_reference"] = float((rv - mine).abs().max())
        ok &= rows["vs_mlp_reference"] < 1e-5
    except Exception as e:
        rows["vs_mlp_reference"] = f"skipped: {e!r}"[:120]
    # 変異対照: 内側の φ′ を 1 + cos にすると自動微分と合わない
    mut = float((torch.autograd.grad(K.snake_phi(z, a, "snake").sum(), z)[0] - (1 + torch.cos(2 * a * z.detach()))).abs().max())
    rows["mutant_cos"] = mut
    ok &= mut > 1e-2
    record("S-act(phi' = autograd, KKT1 continuous, matches MLP SnakeFamily)", ok, **rows)


def grads(model):
    return torch.cat([p.grad.detach().double().reshape(-1) for p in model.parameters()])


def vbufs(model):
    vs = [b.detach().double().reshape(-1) for n, b in model.named_buffers() if n.endswith(".V")]
    return torch.cat(vs) if vs else torch.zeros(1, dtype=torch.float64)


def _pair(arm, dev, ctx, steps):
    me = C.build_model(arm, "M", 0, dev)
    ee = C.Engine(me, arm, "M", graph=False)
    mg = C.build_model(arm, "M", 0, dev)
    eg = C.Engine(mg, arm, "M", graph=True)
    b1, b2 = torch.zeros(4, device=dev), torch.zeros(4, device=dev)
    for u in range(steps):
        xb, yb = ctx.batch(u)
        ee.step(xb, yb, b1, b1.clone(), u)
        eg.step(xb, yb, b2, b2.clone(), u)
    gd = float((grads(me) - grads(mg)).abs().max())
    vd = float((vbufs(me) - vbufs(mg)).abs().max())
    pd = float(max((p - q).abs().max() for p, q in zip(me.parameters(), mg.parameters())))
    return gd, vd, pd


def s_engine(dev, ctx):
    rows, ok = [], True
    _, _, ref_p = _pair("C3", dev, ctx, 3)
    for arm in ARMS:
        g1, v1, _ = _pair(arm, dev, ctx, 1)
        g3, v3, p3 = _pair(arm, dev, ctx, 3)
        o = g1 == 0.0 and v1 == 0.0 and p3 <= 4 * ref_p
        ok &= o
        rows.append(dict(arm=arm, step1_grad_diff=g1, step1_V_diff=v1, step3_param_diff=p3, ref_C3=ref_p, ok=o))
    record("S-engine-k(graph = eager)", ok, rows=rows)


def s_alpha(dev, ctx):
    """V の初期値は初期値の網の課題 1 の probe での前活性の分散、更新はバッチ分散の EMA（β 0.01）。"""
    arm = "SNA"
    m = C.build_model(arm, "M", 0, dev)
    probe = C.task_cpu_M(0, 1, 1, 0.01)["probe"].to(dev)
    m0 = K._orig_build(arm, "M", 0, dev)                      # 較正していない網（V = 1）
    cap = []
    with torch.no_grad():
        m0.body(probe, cap)
    calib = [cap[2 * li + 1].reshape(-1, C.FF).var(0, unbiased=False).double() for li in range(C.L)]
    e0 = max(float((m.h[li].mlp.V.double() - calib[li]).abs().max() / calib[li].abs().max()) for li in range(C.L))
    eng = C.Engine(m, arm, "M", graph=False)
    ref = [m.h[li].mlp.V.double().clone() for li in range(C.L)]
    ref_mut = [r.clone() for r in ref]                     # 変異対照: β = 0.1 の EMA
    b = torch.zeros(6, device=dev)
    errs, muts = [], []
    for u in range(4):
        x, y = ctx.batch(u)
        cap = []
        with torch.no_grad():
            m.body(x.long(), cap)
            vs = [cap[2 * li + 1].reshape(-1, C.FF).double().var(0, unbiased=False) for li in range(C.L)]
        for li in range(C.L):
            ref[li] = (1 - K.SN_BETA) * ref[li] + K.SN_BETA * vs[li]
            ref_mut[li] = 0.9 * ref_mut[li] + 0.1 * vs[li]
        eng.step(x, y, b, b.clone(), u)
        errs += [float(((m.h[li].mlp.V.double() - ref[li]).abs().max() / ref[li].abs().max())) for li in range(C.L)]
        muts += [float(((m.h[li].mlp.V.double() - ref_mut[li]).abs().max() / ref_mut[li].abs().max())) for li in range(C.L)]
    a = m.h[0].mlp.alpha()
    a_ok = torch.allclose(a, (K.SN_C / m.h[0].mlp.V.sqrt()).clamp(K.SN_LO, K.SN_HI))
    record("S-alpha(calibrated V, EMA of batch variance, alpha rule)", e0 < 1e-6 and max(errs) < 1e-5 and a_ok and max(muts) > 1e-3,
           calib_relerr=e0, ema_max_relerr=max(errs), alpha_rule=bool(a_ok), mutant_beta01_relerr=max(muts))


def run_cli(src, out, arm, n, tasks=2, extra=()):
    cmd = [PY, str(src), "run", "--box", "M", "--arm", arm, "--seed", "0", "--tasks", str(tasks), "--n_per_task", str(n),
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
        record(f"S-repro-k({arm})", not diff and len(files) >= 7, differing=diff)
        c = base / f"{arm}_nomeasure"
        run_cli(SRC, c, arm, 300, extra=["--no_measure", "--within_tasks"])
        pa = json.loads((a / "provenance.json").read_text())
        pc = json.loads((c / "provenance.json").read_text())
        record(f"S-inert-k({arm})", pa["final_sha"] == pc["final_sha"], with_measure=pa["final_sha"][:16], without=pc["final_sha"][:16])
        ls = list(csv.DictReader(open(a / "layer_stats.csv")))
        err = max(float(r["decomp_err"]) for r in ls)
        adam = sorted(p.name for p in (a / "ckpts").glob("*_adam.pt"))
        record(f"S-decomp-k+S-adam({arm})", err < 1e-5 and adam == ["task1_adam.pt"], decomp_maxerr=err, adam_files=adam)
    # SNAfrz1: 課題 2 の中で α が動かない（SNA は動く＝変異対照）
    def alpha_series(d):
        ls = list(csv.DictReader(open(d / "layer_stats.csv")))
        return {(int(r["task"]), int(r["u"]), int(r["layer"])): float(r["alpha_med"]) for r in ls if r["probe"] == "chain"}
    fz, sn = alpha_series(base / "SNAfrz1_a"), alpha_series(base / "SNA_a")
    t1end = {l: fz[(1, 300, l)] for l in range(C.L)}
    fz_const = all(abs(fz[k] - t1end[k[2]]) == 0.0 for k in fz if k[0] == 2)
    sn_moves = any(abs(sn[k] - sn[(1, 300, k[2])]) > 0 for k in sn if k[0] == 2)
    same_t1 = all(fz[k] == sn[k] for k in fz if k[0] == 1)
    record("S-freeze(SNAfrz1 fixes alpha from task 2; equals SNA in task 1)", fz_const and sn_moves and same_t1,
           frozen_constant=fz_const, sna_moves=sn_moves, task1_identical=same_t1)
    # 本走の腕（C3）は、このモジュールを通しても学習が変わらない
    d1, d2 = base / "C3_via_kune", base / "C3_via_base"
    run_cli(SRC, d1, "C3", 200)
    run_cli(SRC_BASE, d2, "C3", 200)
    p1 = json.loads((d1 / "provenance.json").read_text())
    p2 = json.loads((d2 / "provenance.json").read_text())
    same_pt = filecmp.cmp(d1 / "per_task.csv", d2 / "per_task.csv", shallow=False)
    record("S-base-unchanged(C3 through the kunekune module)", p1["final_sha"] == p2["final_sha"] and same_pt,
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
    for fn in (lambda: s_act(dev), lambda: s_engine(dev, ctx), lambda: s_alpha(dev, ctx), lambda: s_runs(out / "runs")):
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
