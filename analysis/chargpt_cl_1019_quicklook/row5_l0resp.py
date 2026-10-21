#!/usr/bin/env python3
"""spec 追補 9: 2 ブロックで L0 の応答だけを操作して学び直しが変わるかを問う（事後の介入・判定つき）。
受け手の L0 の ReLU に入る活動を、凍結した参照網の前活性を使って置き換える（固定項は no_grad）:
  ã = φ(r0) + {φ(base + (z − r0)) − φ(base)}、z = 受け手の今の L0 前活性、r0 = 受け手の出発点の凍結写しの前活性
  base: self = r0、betaFA = r0 + β、Fswap / Frev = d0（供与者の凍結写しの前活性）、Fswapscale = c ⊙ d0
  betaF（報告のみ）: ã = φ(z + β)
参照網は各自の埋め込み・LN1₀・注意 0（残差加算込み）・LN2₀・W_up,0 を通す。各 forward（訓練のバッチ・評価集合）で毎回計算する。
参照網は deep copy で optimizer に入れず（子 module として登録しない）、勾配を流さず、V も更新しない。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/row5_l0resp.py --arm betaFA --seed 0 --t0 90 --out <dir>
"""
import argparse, copy, csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
M2 = ROOT / "results/chargpt_cl_1019/M2"
P_STAR = 0.3321
RECEIVER = dict(C3self="C3", capself="ReLUcap", betaFA="C3", Fswap="C3", Frev="ReLUcap", Fswapscale="C3", betaF="C3")
DONOR = dict(Fswap="ReLUcap", Fswapscale="ReLUcap", Frev="C3")


def load_net(arm, seed, t0, dev):
    m = C.build_model(arm, "M", seed, dev)
    m.load_state_dict(torch.load(M2 / arm / f"seed{seed}" / "ckpts" / f"task{t0}.pt", map_location=dev))
    return m


def freeze_copy(m):
    r = copy.deepcopy(m)
    for p in r.parameters(): p.requires_grad_(False)
    r.eval()
    return r


def pre0(net, x):
    """参照網の L0 の前活性（no_grad、V は更新しない）。"""
    with torch.no_grad():
        h = net.wte(x); b = net.h[0]
        h = h + b.sc * b.attn(b.n1(h))
        return b.mlp.up(b.n2(h))


def calib_sets(seed, t0):
    ch = C.ChainM(seed, t0 + 1, 0.01)
    cal = ch.sample(C.M_PROBE, C.cpu_gen(seed, "row5", t0 + 1, "calib"))
    val = ch.sample(C.M_PROBE, C.cpu_gen(seed, "row5", t0 + 1, "valid"))
    return cal, val                                                # 長さ 129（入力は [:, :128]、目標は [:, 1:]）


class Op:
    """受け手の L0 の活動の置き換え。fields は x を受けて (r0, base) を返す。"""

    def __init__(s, kind, R0, D0, beta, c):
        s.kind, s.R0, s.D0, s.beta, s.c = kind, R0, D0, beta, c

    def fields(s, x):
        r0 = pre0(s.R0, x)
        if s.kind == "self": return r0, r0
        if s.kind == "betaFA": return r0, r0 + s.beta
        if s.kind == "swap": return r0, pre0(s.D0, x)
        if s.kind == "swapscale": return r0, s.c * pre0(s.D0, x)
        raise ValueError(s.kind)


def install(model, op):
    """受け手の body と L0 の φ を置き換える（グラフを取る前に呼ぶ）。"""
    mlp0 = model.h[0].mlp
    if op.kind == "betaF":
        beta = op.beta
        mlp0.phi = lambda z, beta=beta: F.relu(z + beta)
        mlp0.dphi = lambda z, beta=beta: ((z + beta) > 0).to(z.dtype)        # 測定の φ′ も実効ゲートに
        return
    orig_body = model.body

    def body(x, cap=None, orig=orig_body, op=op, mlp0=mlp0):
        mlp0._r0, mlp0._base = op.fields(x)
        return orig(x, cap)
    model.body = body

    def phi(z, mlp0=mlp0):
        r0, base = mlp0._r0, mlp0._base
        return F.relu(r0) + (F.relu(base + (z - r0)) - F.relu(base))
    mlp0.phi = phi
    mlp0.dphi = lambda z, mlp0=mlp0: ((mlp0._base.reshape(z.shape) + (z - mlp0._r0.reshape(z.shape))) > 0).to(z.dtype)    # 測定の φ′ も実効ゲート（body が場を置いた直後に呼ばれる）


@torch.no_grad()
def eff_gate(model, op, x):
    """実効ゲートの開く割合（unit ごと）。z_eff = base + (z − r0)。betaF は z + β。"""
    cap = []; model.body(x, cap); z = cap[1].reshape(-1, C.FF)
    if op.kind == "betaF": ze = z + op.beta
    else:
        r0, base = op.fields(x); ze = base.reshape(-1, C.FF) + (z - r0.reshape(-1, C.FF))
    return (ze > 0).double().mean(0), (z > 0).double().mean(0)


@torch.no_grad()
def update0_record(model, R0, seq):
    """更新 0（操作の直後）: 受け手（操作あり）と凍結写し（操作なし＝自然網）の差を同じ独立入力で。"""
    x, y = seq[:, :C.T], seq[:, 1:]
    out = {}
    def run(net):
        cap = []; hk = []; got = {}
        hk.append(net.h[0].mlp.register_forward_hook(lambda m, i, o: got.__setitem__("f0", o.detach())))
        hk.append(net.nf.register_forward_hook(lambda m, i, o: got.__setitem__("nf_in", i[0].detach())))
        lg = net(x, cap)
        for h in hk: h.remove()
        res = got["nf_in"].reshape(-1, C.D).double()
        return lg, got["f0"], res, (cap[3] > 0).double().mean(), (res.var(1, unbiased=False) + net.nf.eps).sqrt().mean()
    la, fa, ra, ga, da = run(model); lb, fb, rb, gb, db = run(R0)
    ce = lambda lg: float(F.cross_entropy(lg[:, 1:].reshape(-1, C.V), y[:, 1:].reshape(-1)))
    dh = (fa - fb).reshape(-1, C.D).double()
    out.update(dh_rms_over_res=float(dh.pow(2).mean().sqrt() / rb.pow(2).mean().sqrt()), dh_common_norm=float(dh.mean(0).norm()),
               logit_diff_rms=float((la - lb).double().pow(2).mean().sqrt()), loss_op=ce(la), loss_nat=ce(lb), loss_diff=ce(la) - ce(lb),
               l1_open_op=float(ga), l1_open_nat=float(gb), nf_denom_op=float(da), nf_denom_nat=float(db))
    return out


def three_phase(loss, H):
    ex = loss.astype(np.float64) - H; ma = np.convolve(ex, np.ones(50) / 50, "valid"); hit = np.nonzero(ma <= 0.15)[0]
    return dict(T015=(int(hit[0]) + 50 if len(hit) else 10000), reached=bool(len(hit)), E1000=float(ex[950:1050].mean()), ELast=float(ex[-1000:].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(RECEIVER)); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t0", type=int, required=True); ap.add_argument("--ntasks", type=int, default=10); ap.add_argument("--n_per_task", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); assert C.L == 2
    C.setup_torch(); dev = torch.device("cuda"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0w = time.time()
    rarm = RECEIVER[a.arm]; seed, t0 = a.seed, a.t0
    model = load_net(rarm, seed, t0, dev)
    R0 = freeze_copy(model); D0 = freeze_copy(load_net(DONOR[a.arm], seed, t0, dev)) if a.arm in DONOR else None
    ref_snap = [(t, t.detach().clone()) for net in ([R0] + ([D0] if D0 is not None else [])) for t in list(net.parameters()) + list(net.buffers())]
    cal, val = calib_sets(seed, t0); xc, xv = cal[:, :C.T].to(dev), val[:, :C.T].to(dev)
    r0c = pre0(R0, xc).reshape(-1, C.FF).double(); p_cal = (r0c > 0).double().mean(0); Dset = p_cal < 0.005
    sd_r = r0c.std(0); beta = torch.zeros(C.FF, dtype=torch.float64, device=dev)
    q = torch.quantile(r0c, 1 - P_STAR, dim=0)
    use = Dset & (sd_r > 0); beta[use] = -q[use]
    info = dict(arm=a.arm, receiver=rarm, donor=DONOR.get(a.arm), seed=seed, t0=t0, p_star=P_STAR, n_D=int(Dset.sum()), n_D_zero_var=int((Dset & (sd_r == 0)).sum()))
    c = None
    if a.arm == "Fswapscale":
        d0c = pre0(D0, xc).reshape(-1, C.FF).double(); sd_d = d0c.std(0); c = sd_r / sd_d.clamp_min(1e-30)
        bad = (sd_d < 1e-3) | (c > 1000); cmed = c.median()                 # 9.B: その seed の c の中央値（全 unit）
        c[bad] = cmed if torch.isfinite(cmed) and cmed <= 1000 else torch.ones((), dtype=c.dtype, device=c.device); info.update(c_bad=int(bad.sum()), c_min=float(c.min()), c_med=float(c.median()), c_max=float(c.max()),
                                                                           width_match_err_med=float(((c * sd_d) / sd_r - 1).abs().median()))
        c = c.float()
    kind = {"C3self": "self", "capself": "self", "betaFA": "betaFA", "Fswap": "swap", "Frev": "swap", "Fswapscale": "swapscale", "betaF": "betaF"}[a.arm]
    op = Op(kind, R0, D0, beta.float(), c)
    install(model, op)
    # 操作の成立（更新 0、検証集合）
    pe_v, pn_v = eff_gate(model, op, xv)
    info["update0"] = update0_record(model, R0, val.to(dev))
    info["valid_D_eff_open_mean"] = float(pe_v[Dset].mean()) if Dset.any() else None
    info["valid_D_eff_open_ge_half_pstar"] = float((pe_v[Dset] >= P_STAR / 2).double().mean()) if Dset.any() else None
    info["valid_eff_open_mean"] = float(pe_v.mean()); info["valid_nat_open_mean"] = float(pn_v.mean())
    if D0 is not None:
        info["valid_donor_open_mean"] = float((pre0(D0, xv) > 0).double().mean())
    eng = C.Engine(model, rarm, "M", graph=True)
    sd = torch.load(M2 / rarm / f"seed{seed}" / "ckpts" / f"task{t0}_adam.pt", map_location=dev)
    params = list(model.parameters()); assert len(sd["state"]) == len(params), (len(sd["state"]), len(params))
    with torch.no_grad():
        for i, p in enumerate(params):
            st, sv = eng.opt.state[p], sd["state"][i]
            assert st["exp_avg"].shape == sv["exp_avg"].shape
            for k in ("exp_avg", "exp_avg_sq", "step"): st[k].copy_(sv[k].to(st[k].dtype))
    probe_u = C.uniform_probe_M(seed, dev)
    tasks = list(range(t0 + 1, t0 + 1 + a.ntasks)); n = a.n_per_task
    online = np.zeros((len(tasks), n), np.float32); rows, layer_rows = [], []
    for ti, t in enumerate(tasks):
        cpu = C.task_cpu_M(seed, t, n, 0.01); ctx = C.TaskCtx("M", seed, t, n, dev, alpha=0.01, W=None, cpu=cpu); tt = time.time()
        loss, aux = C.run_task(model, eng, ctx, warmup_from=(t - 1) * n); online[ti] = loss
        for pn, xp in dict(chain=ctx.probe, uniform=probe_u).items():
            for li, r in enumerate(C.measure(model, xp)): layer_rows.append(dict(task=t, probe=pn, layer=li, **r["agg"]))
        pe_c, pn_c = eff_gate(model, op, ctx.probe); pe_v, _ = eff_gate(model, op, xv)
        ph = three_phase(loss, ctx.H) if n == 10000 else {}
        row = dict(task=t, H=ctx.H, **C.summarize_online("M", loss, aux, ctx.H), **ph,
                   eff_open_chain=float(pe_c.mean()), nat_open_chain=float(pn_c.mean()), nat_dead_chain=float((pn_c < 0.005).double().mean()),
                   eff_dead_chain=float((pe_c < 0.005).double().mean()),
                   D_eff_open_valid=float(pe_v[Dset].mean()) if Dset.any() else float("nan"), D_eff_open_ge_half_pstar_valid=float((pe_v[Dset] >= P_STAR / 2).double().mean()) if Dset.any() else float("nan"),
                   finite=bool(np.isfinite(loss).all()))
        rows.append(row)
        print(f"[{a.arm} s{seed} t0={t0}] task {t} online {row['online_all']:.4f} E1000 {ph.get('E1000', float('nan')):.3f} | L0 実効 open {row['eff_open_chain']:.3f} 自然 open {row['nat_open_chain']:.3f}"
              f" D の実効 open {row['D_eff_open_valid']:.3f} | {time.time() - tt:.0f}s", flush=True)
        del ctx
    ref_ok = all(torch.equal(t_, s_) for t_, s_ in ref_snap) and all(p.grad is None for net in ([R0] + ([D0] if D0 is not None else [])) for p in net.parameters())
    info["ref_unchanged_and_no_grad"] = ref_ok

    def wcsv(path, rws):
        keys = list(dict.fromkeys(k for r in rws for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rws: w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    wcsv(out / "per_task.csv", rows); wcsv(out / "layer_stats.csv", layer_rows); np.savez_compressed(out / "online.npz", loss=online)
    gh, dirty = C.git_info()
    (out / "provenance.json").write_text(json.dumps(dict(script="row5_l0resp.py", **info, tasks=tasks, n_per_task=n, git_hash=gh, dirty_src=dirty,
                                                       gpu=torch.cuda.get_device_name(0), host=__import__("socket").gethostname(), seconds=time.time() - t0w,
                                                       grade="spec 追補 9（判定つき）"), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
