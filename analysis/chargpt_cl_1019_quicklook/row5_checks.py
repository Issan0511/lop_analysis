#!/usr/bin/env python3
"""spec 追補 9 §7 の実装検査（走る前）。row5_l0resp.py の部品を使う。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/row5_checks.py --out results/chargpt_cl_1019/row5/checks
検査:
  A 自己（C3self・capself）は元の走と丸め誤差の範囲で一致（seed 0、課題 91〜92）
  B 非自己の場（受け手を変位させた F-swap・β-FA）で、実効の活動 ã を独立な式と照合、∂ã/∂z を実効ゲートと照合、
    受け手の W_up,0 の勾配を有限差分（float64、折れ目の近くの成分は除く）と照合。自然網の勾配とは違うこと（自明でない）
  C 場の入れ替え: バッチの順序の入れ替えで場が同じだけ入れ替わる・別のバッチ・別の供与者で場が変わる
  D 参照網に勾配も状態更新も無い（数歩の更新と測定のあと）
  E graph と eager の数歩が一致
  F Adam の状態の復元が checkpoint と厳密に一致
"""
import argparse, json, subprocess, sys, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
import row5_l0resp as R  # noqa: E402
C = R.C


def build_op(kind, model, R0, D0, seed, t0, dev, dtype):
    cal, _ = R.calib_sets(seed, t0); xc = cal[:, :C.T].to(dev)
    r0c = R.pre0(R0, xc).reshape(-1, C.FF).double(); D = (r0c > 0).double().mean(0) < 0.005
    beta = torch.zeros(C.FF, dtype=torch.float64, device=dev); q = torch.quantile(r0c, 1 - R.P_STAR, dim=0); beta[D] = -q[D]
    return R.Op(kind, R0, D0, beta.to(dtype), None), D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); assert C.L == 2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = {}
    C.setup_torch(); torch.manual_seed(0)
    seed, t0 = 0, 90
    # ---------------- A: 自己は元の走と一致
    for arm, rarm in (("C3self", "C3"), ("capself", "ReLUcap")):
        d = out / f"A_{arm}"
        __import__("shutil").rmtree(d, ignore_errors=True)                 # 古い出力は使わない（いまのコードで回し直す）
        if True:
            subprocess.run([sys.executable, str(Path(__file__).parent / "row5_l0resp.py"), "--arm", arm, "--seed", "0", "--t0", "90", "--ntasks", "2", "--out", str(d)],
                           check=True, env=dict(os.environ, CHARGPT_NLAYER="2"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mine = np.load(d / "online.npz")["loss"]; ref = np.load(R.M2 / rarm / "seed0" / "online.npz")["loss"][90:92]
        res[f"A_{arm}"] = dict(max_abs=float(np.abs(mine - ref).max()), task91_first10_max_abs=float(np.abs(mine[0, :10] - ref[0, :10]).max()),
                               task91_first100_max_abs=float(np.abs(mine[0, :100] - ref[0, :100]).max()),
                               mean_task_diff=[float(mine[i].mean() - ref[i].mean()) for i in range(2)], bitwise=bool(np.array_equal(mine, ref)))
    # ---------------- B: 非自己の場（CPU float64）
    dev = torch.device("cpu"); torch.set_num_threads(8)
    rec = R.load_net("C3", seed, t0, dev).double(); don = R.load_net("ReLUcap", seed, t0, dev).double()
    R0 = R.freeze_copy(rec); D0 = R.freeze_copy(don)
    with torch.no_grad():                                          # 受け手を変位させる（出発点からずらす）
        for p in rec.parameters(): p.add_(0.02 * p.std() * torch.randn_like(p))
    cal, val = R.calib_sets(seed, t0); x = val[:8, :C.T]
    for kind in ("swap", "betaFA"):
        m = __import__("copy").deepcopy(rec); op, Dset = build_op(kind, m, R0, D0, seed, t0, dev, torch.float64); R.install(m, op)
        cap = []; got = {}
        hk = m.h[0].mlp.register_forward_hook(lambda mod, i, o: got.__setitem__("f0", o))
        m.body(x, cap); hk.remove()
        z = cap[1]
        # 独立な式: 各前活性をここで計算し直す
        def pre_indep(net, x):
            h = net.wte(x); b = net.h[0]; h = h + b.sc * b.attn(b.n1(h)); return b.mlp.up(b.n2(h))
        with torch.no_grad():
            z_i = pre_indep(m, x); r0_i = pre_indep(R0, x); d0_i = pre_indep(D0, x)
            base_i = d0_i if kind == "swap" else r0_i + op.beta
            a_i = F.relu(r0_i) + F.relu(base_i + z_i - r0_i) - F.relu(base_i)
            f0_i = m.h[0].mlp.down(a_i)
        err_f0 = float((got["f0"] - f0_i).abs().max() / f0_i.abs().max()); err_z = float((z - z_i).abs().max())
        # ∂ã/∂z = 1[z_eff > 0]
        zz = z_i.clone().requires_grad_(True); m.h[0].mlp._r0, m.h[0].mlp._base = r0_i, base_i
        ga = torch.autograd.grad(m.h[0].mlp.phi(zz).sum(), zz)[0]; gate = ((base_i + z_i - r0_i) > 0).double()
        err_gate = float((ga - gate).abs().max())
        frac_changed = float((gate != (z_i > 0).double()).double().mean())
        # 受け手の W_up,0 の勾配: autograd 対 有限差分
        W = m.h[0].mlp.up.weight; wvec = torch.randn(C.V, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        def L_of():
            return (m(x)[:, 1:] * wvec).sum()
        g_auto = torch.autograd.grad(L_of(), W)[0]
        zeff = (base_i + z_i - r0_i).reshape(-1, C.FF); margin = zeff.abs().min(0).values      # unit ごとの折れ目までの距離
        units = torch.nonzero(margin > 1e-3).flatten()[:40]; errs = []
        eps = 1e-7
        for u in units[:20].tolist():
            j = int(torch.randint(0, C.D, (1,)))
            with torch.no_grad():
                W[u, j] += eps; lp = float(L_of()); W[u, j] -= 2 * eps; lm = float(L_of()); W[u, j] += eps
            fd = (lp - lm) / (2 * eps); errs.append(abs(fd - float(g_auto[u, j])) / (abs(fd) + abs(float(g_auto[u, j])) + 1e-12))
        nat = __import__("copy").deepcopy(rec); g_nat = torch.autograd.grad((nat(x)[:, 1:] * wvec).sum(), nat.h[0].mlp.up.weight)[0]
        res[f"B_{kind}"] = dict(err_f0_vs_indep=err_f0, err_z=err_z, err_gate=err_gate, frac_gate_differs_from_natural=frac_changed,
                                fd_rel_err_max=float(max(errs)), fd_rel_err_med=float(np.median(errs)), n_fd=len(errs),
                                grad_rel_diff_vs_natural=float((g_auto - g_nat).norm() / g_nat.norm()), n_D=int(Dset.sum()))
    # ---------------- C: 場の入れ替え
    op = R.Op("swap", R0, D0, None, None)
    x1, x2 = val[:4, :C.T], val[4:8, :C.T]; perm = torch.tensor([2, 0, 3, 1])
    r_a, b_a = op.fields(x1); r_p, b_p = op.fields(x1[perm])
    _, b_2 = op.fields(x2)
    don2 = R.freeze_copy(R.load_net("ReLUcap", 1, t0, dev).double()); _, b_d2 = R.Op("swap", R0, don2, None, None).fields(x1)
    res["C"] = dict(perm_err=float((b_p - b_a[perm]).abs().max()), other_batch_diff=float((b_2 - b_a).abs().mean() / b_a.abs().mean()),
                    other_donor_diff=float((b_d2 - b_a).abs().mean() / b_a.abs().mean()))
    # ---------------- D・E・F（GPU）
    dev = torch.device("cuda"); C.setup_torch()
    cpu = C.task_cpu_M(seed, t0 + 1, 10000, 0.01)
    states = {}
    for graph in (True, False):
        m = R.load_net("C3", seed, t0, dev); R0g = R.freeze_copy(m); D0g = R.freeze_copy(R.load_net("ReLUcap", seed, t0, dev))
        snap = [(t, t.detach().clone()) for net in (R0g, D0g) for t in list(net.parameters()) + list(net.buffers())]
        op, _ = build_op("swap", m, R0g, D0g, seed, t0, dev, torch.float32); R.install(m, op)
        eng = C.Engine(m, "C3", "M", graph=graph)
        sd = torch.load(R.M2 / "C3" / "seed0" / "ckpts" / f"task{t0}_adam.pt", map_location=dev)
        params = list(m.parameters())
        with torch.no_grad():
            for i, p in enumerate(params):
                if not graph:                                      # eager の Adam は初回の step まで状態を持たない
                    sv = sd["state"][i]
                    eng.opt.state[p] = dict(step=sv["step"].detach().clone().float().cpu(), exp_avg=sv["exp_avg"].to(dev).clone(), exp_avg_sq=sv["exp_avg_sq"].to(dev).clone())
                    continue
                for k in ("exp_avg", "exp_avg_sq", "step"): eng.opt.state[p][k].copy_(sd["state"][i][k].to(eng.opt.state[p][k].dtype))
        if graph:
            res["F_adam_restore_exact"] = all(torch.equal(eng.opt.state[p][k].float(), sd["state"][i][k].to(dev).float()) for i, p in enumerate(params) for k in ("exp_avg", "exp_avg_sq", "step"))
        ctx = C.TaskCtx("M", seed, t0 + 1, 10000, dev, alpha=0.01, W=None, cpu=cpu)
        bl = torch.zeros(5, device=dev); ba = torch.zeros(5, device=dev)
        for u in range(5):
            eng.set_lr(eng.base_lr); xb, yb = ctx.batch(u); eng.step(xb, yb, bl, ba, u)
        C.measure(m, ctx.probe)
        states[graph] = ([p.detach().clone() for p in params], bl.cpu().numpy())
        if graph:
            res["D_ref_unchanged"] = all(torch.equal(t_, s_) for t_, s_ in snap); res["D_ref_no_grad"] = all(p.grad is None for net in (R0g, D0g) for p in net.parameters())
        del eng; C._ENGINES.clear(); torch.cuda.empty_cache()
    # F2: 復元後の独立な Adam の 1 歩（C3 の受け手・F-swap の操作つき、eager）。式は torch に頼らず手で書く
    m = R.load_net("C3", seed, t0, dev); R0g = R.freeze_copy(m); D0g = R.freeze_copy(R.load_net("ReLUcap", seed, t0, dev))
    op, _ = build_op("swap", m, R0g, D0g, seed, t0, dev, torch.float32); R.install(m, op)
    names = [n for n, _ in m.named_parameters()]; params = list(m.parameters())
    sd = torch.load(R.M2 / "C3" / "seed0" / "ckpts" / f"task{t0}_adam.pt", map_location=dev)
    shape_ok = all(sd["state"][i]["exp_avg"].shape == p.shape for i, p in enumerate(params))
    ref_names = [n for n, _ in C.build_model("C3", "M", seed, "cpu").named_parameters()]
    eng = C.Engine(m, "C3", "M", graph=False)
    with torch.no_grad():
        for i, p in enumerate(params):
            sv = sd["state"][i]; eng.opt.state[p] = dict(step=sv["step"].detach().clone().float().cpu(), exp_avg=sv["exp_avg"].to(dev).clone(), exp_avg_sq=sv["exp_avg_sq"].to(dev).clone())
    ctx = C.TaskCtx("M", seed, t0 + 1, 10000, dev, alpha=0.01, W=None, cpu=cpu); xb, yb = ctx.batch(0)
    p_before = [p.detach().clone() for p in params]
    m.zero_grad(set_to_none=True); loss, _ = eng._loss(xb.long(), yb.long()); loss.backward(); grads = [p.grad.detach().clone() for p in params]
    eng.opt.step()
    errs = []
    lr, b1, b2, ep = eng.base_lr, 0.9, 0.999, 1e-8
    for i, (p, p0, g) in enumerate(zip(params, p_before, grads)):
        sv = sd["state"][i]; st = float(sv["step"]) + 1
        mm = b1 * sv["exp_avg"].to(dev) + (1 - b1) * g; vv = b2 * sv["exp_avg_sq"].to(dev) + (1 - b2) * g * g
        p_ref = p0 - lr * (mm / (1 - b1 ** st)) / ((vv / (1 - b2 ** st)).sqrt() + ep)
        errs.append(float((p.detach() - p_ref).abs().max() / (p0 - p_ref).abs().max().clamp_min(1e-30)))
    res["F2_adam_one_step"] = dict(names_match_fresh_model=names == ref_names, shapes_match=shape_ok, rel_err_max=max(errs))
    del eng; C._ENGINES.clear(); torch.cuda.empty_cache()
    # G: ELR と文字どおりの形（短い走を回し直して確かめる）
    for mode in ("elr", "literal"):
        d = out / f"G_{mode}"; __import__("shutil").rmtree(d, ignore_errors=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "row5_elr.py"), "--mode", mode, "--seed", "0", "--to_task", "3", "--n_per_task", "200", "--out", str(d)],
                       check=True, env=dict(os.environ, CHARGPT_NLAYER="2"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import csv as _csv
    rows = list(_csv.DictReader(open(out / "G_elr" / "per_task.csv")))
    rn = np.load(R.M2 / "C3" / "seed0" / "unit_stats.npz")["rownorm"][:, 0, :]
    res["G_elr"] = dict(norm_ratio_med=[float(r["w0_norm_over_n1_med"]) for r in rows], k_end_med=[float(r["k_med"]) for r in rows],
                        k_expected_med=[float(np.median(rn[t - 1] / rn[0])) for t in (2, 3)])
    # 射影が行ごとに厳密に効いているか（保存は課題 50・100 だけなので、短い走の最後の重みを再現して確かめる代わりに式を直接検査）
    Wt = torch.randn(8, 4); n1 = torch.rand(8) + 0.5; Wt.mul_((n1 / Wt.norm(dim=1))[:, None]); res["G_proj_formula_err"] = float((Wt.norm(dim=1) - n1).abs().max())
    W0 = torch.randn(8, 4); W1 = W0 + 0.01 * torch.randn(8, 4); n1 = W0.norm(dim=1) * 0.5
    Wl = W0 + (W1 - W0) * (W0.norm(dim=1) / n1)[:, None]; res["G_literal_formula_err"] = float(((Wl - W0) - 2 * (W1 - W0)).abs().max())
    # H: 測定の φ′ が実効ゲート（β-FA、課題 91 の probe）
    m = R.load_net("C3", seed, t0, dev); R0g = R.freeze_copy(m); op, Dset = build_op("betaFA", m, R0g, None, seed, t0, dev, torch.float32); R.install(m, op)
    agg = C.measure(m, cpu["probe"].to(dev))[0]["agg"]; pe, _ = R.eff_gate(m, op, cpu["probe"].to(dev))
    res["H_phi1_is_effective"] = dict(phi1_mean=agg["phi1_mean"], eff_open_mean=float(pe.mean()), diff=abs(agg["phi1_mean"] - float(pe.mean())))
    pg, lg = states[True]; pe, le = states[False]
    res["E_graph_vs_eager"] = dict(param_max_abs=float(max((x - y).abs().max() for x, y in zip(pg, pe))), loss_max_abs=float(np.abs(lg - le).max()),
                                   bitwise=all(torch.equal(x, y) for x, y in zip(pg, pe)))
    tol = dict(
        A_first10=all(res[f"A_{a}"]["task91_first10_max_abs"] < 1e-4 for a in ("C3self", "capself")),
        B=all(res[f"B_{k}"]["err_f0_vs_indep"] < 1e-12 and res[f"B_{k}"]["err_gate"] == 0 and res[f"B_{k}"]["fd_rel_err_max"] < 1e-4
              and res[f"B_{k}"]["grad_rel_diff_vs_natural"] > 0.1 and res[f"B_{k}"]["frac_gate_differs_from_natural"] > 0.01 for k in ("swap", "betaFA")),
        C=res["C"]["perm_err"] == 0 and res["C"]["other_batch_diff"] > 0.1 and res["C"]["other_donor_diff"] > 0.1,
        D=res["D_ref_unchanged"] and res["D_ref_no_grad"],
        E=res["E_graph_vs_eager"]["param_max_abs"] < 1e-3,
        F=res["F_adam_restore_exact"] and res["F2_adam_one_step"]["names_match_fresh_model"] and res["F2_adam_one_step"]["shapes_match"] and res["F2_adam_one_step"]["rel_err_max"] < 1e-4,
        G=all(abs(x - 1) < 1e-4 for x in res["G_elr"]["norm_ratio_med"]) and all(abs(a - b) < 1e-3 * b for a, b in zip(res["G_elr"]["k_end_med"], res["G_elr"]["k_expected_med"]))
          and res["G_proj_formula_err"] < 1e-5 and res["G_literal_formula_err"] < 1e-5,
        H=res["H_phi1_is_effective"]["diff"] < 1e-6)
    res["PASS"] = tol; res["ALL_PASS"] = all(tol.values())
    (out / "checks.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    sys.exit(0 if res["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
