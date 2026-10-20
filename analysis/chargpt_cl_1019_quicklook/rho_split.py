#!/usr/bin/env python3
"""位置の動きを「① 自分の重み ② 上流を通る自分の信号 ③ 上流を通る他人の信号」に割る（事後・判定外、Issa の依頼を Neokunekune の fork 経由、1020 夜）。

層 ℓ の unit j、y_n = LN2_ℓ(r_n)、z_jn = w_j·y_n（bias なし）、位置 m_j = w_j·ȳ、誤差 e_in = ∂L/∂z_in。
θ_ℓ = 層 ℓ の FFN の入力 y に効くパラメータすべて（埋め込み〔読み出しと共有〕、下のブロック、ブロック ℓ の LN1・注意・LN2 の γ）。
  ①  = −Σ_n e_jn K_n、K_n = y_n·ȳ                     （= −(∂L/∂w_j)·ȳ）
  ②  = −⟨v_j, g^{(j)}⟩、g^{(j)} = ∇_θ Σ_n sg(e_jn) z_jn  （自分の信号が上流の重みを通る分）
  ③a = −⟨v_j, g^{FFNℓ} − g^{(j)}⟩                       （同じ層の他の unit の信号）
  ③b = −⟨v_j, g_θ − g^{FFNℓ}⟩                            （残差・読み出しなど、層 ℓ の FFN を通らない勾配）
計量 A（SGD）は v_j = ∇_θ m_j、計量 D（Adam の分母、運動量なし）は v_j = D_θ ⊙ ∇_θ m_j、① は D_W で重みづけ。D = 1/(√v̂ + ε)。
計算: 層ごとに ȳ のヤコビアンの行 r_k = ∇_θ ȳ_k（k = 1..d）を逆伝播で取り、その方向への y_n の方向微分 dY_k を float64 の中心差分で取る。
  C_ij = ⟨v_j, g^{(i)}⟩ = Σ_k W_jk a_ik、a_ik = Σ_n e_in w_i·dY_k[n]（D では方向を D⊙r_k に）。② = −C_jj、③a = −Σ_{i≠j} C_ij。
  T_j = −⟨v_j, g_θ⟩ = −w_j·(J_ȳ v[g_θ]) を 1 本の方向微分で。③b = T − ② − ③a。
単位: 各項を unit の幅 σ_j で割る。D は lr 1e−3 を掛けて「Adam の分母つきの 1 歩」に、A は η = 1 のまま（比と符号だけを読む）。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
SEED, ALPHA, LR, EPS, B2 = 1000, 0.01, 1e-3, 1e-8, 0.999


DEV = torch.device("cpu")


def build(arm, sd):
    m = C.build_model(arm, "M", SEED, "cpu"); m.load_state_dict(sd); m = m.double().to(DEV); m.eval()
    for p in m.parameters(): p.requires_grad_(True)
    return m


def probe_of(kind, task, nseq):
    x = C.task_cpu_M(SEED, task, 10, ALPHA)["probe"] if kind == "chain" else C.uniform_probe_M(SEED, "cpu")
    return x[:nseq].to(DEV)


def loss_of(m, X, cap=None):
    lg = m(X, cap) if cap is not None else m(X)
    return F.cross_entropy(lg[:, 1:-1].reshape(-1, C.V), X[:, 2:].reshape(-1))


def theta_of(m, l):
    ps, seen = [], set()
    for p in [m.wte.weight] + [p for b in m.h[:l] for p in b.parameters()] + [m.h[l].n1.weight] + list(m.h[l].attn.parameters()) + [m.h[l].n2.weight]:
        if id(p) not in seen: seen.add(id(p)); ps.append(p)
    return ps


@torch.no_grad()
def y_at(m, X, l):
    h = m.wte(X)
    for b in m.h[:l]: h = b(h)
    b = m.h[l]; h = h + b.sc * b.attn(b.n1(h))
    return b.n2(h).reshape(-1, C.D)


@torch.no_grad()
def jvp_y(m, X, l, theta, dirs, rel=1e-7):
    """y_ℓ の方向微分（中心差分、float64）。dirs は theta と同じ形のテンソルの並び。"""
    tn = torch.sqrt(sum((p * p).sum() for p in theta)); dn = torch.sqrt(sum((d * d).sum() for d in dirs))
    if float(dn) == 0: return torch.zeros(X.numel(), C.D, dtype=torch.float64, device=DEV)
    h = float(rel * tn / dn); orig = [p.detach().clone() for p in theta]
    for p, d in zip(theta, dirs): p.add_(h * d)
    yp = y_at(m, X, l)
    for p, o, d in zip(theta, orig, dirs): p.copy_(o); p.sub_(h * d)
    ym = y_at(m, X, l)
    for p, o in zip(theta, orig): p.copy_(o)
    return (yp - ym) / (2 * h)


def jvp_exact(y, theta, dirs):
    """y の厳密な方向微分（二重逆伝播）。J v = ∂/∂u [ (∂(u·y)/∂θ)·v ]。ReLU の折れ目をまたぐ差分の誤差が無い。"""
    u = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad((u * y).sum(), theta, create_graph=True, allow_unused=True)
    tot = sum((gi * di).sum() for gi, di in zip(g, dirs) if gi is not None)
    return torch.autograd.grad(tot, u, retain_graph=True)[0].detach()


JVP_MODE = "fd"


def adam_D(m, adam_sd):
    D = {}
    for i, p in enumerate(m.parameters()):
        st = adam_sd["state"][i]; v = st["exp_avg_sq"].double(); step = float(st["step"])
        D[id(p)] = (1.0 / (torch.sqrt(v / (1 - B2 ** step)) + EPS)).to(DEV)
    return D


def analyze(arm, sd, adam_sd, task, kind="chain", nseq=64, layers=None, flip=False, top=5, log=print, light=False):
    m = build(arm, sd); X = probe_of(kind, task, nseq); nl = len(m.h); layers = list(range(nl)) if layers is None else layers
    mfd = build(arm, sd)                                      # 中心差分はこちらで動かす（グラフを持つ m の版数を変えないため）
    for p_ in mfd.parameters(): p_.requires_grad_(False)
    Dm = adam_D(m, adam_sd) if adam_sd is not None else None
    cap = []; L = loss_of(m, X, cap) * (-1.0 if flip else 1.0)
    zs = [cap[2 * l + 1] for l in range(nl)]; xin = [cap[2 * l] for l in range(nl)]
    params = list(m.parameters())
    gz = torch.autograd.grad(L, zs, retain_graph=True)
    gfull = dict(zip([id(p) for p in params], torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)))
    out = dict(arm=arm, task=task, kind=kind, nseq=nseq, loss=float(L), layers={})
    for l in layers:
        t0 = time.time(); blk = m.h[l]; theta = theta_of(m, l); thfd = theta_of(mfd, l)
        y = blk.n2(xin[l]).reshape(-1, C.D); ybar = y.mean(0)
        e = gz[l].reshape(-1, C.FF).detach(); W = blk.mlp.up.weight.detach()
        yd, yb = y.detach(), ybar.detach(); zd = yd @ W.T; sd = zd.std(0); alive = (zd > 0).any(0)
        Kn = yd @ yb; GW = e.T @ yd
        res = {}
        diag_units = [int(u) for u in np.random.default_rng(1).choice(C.FF, 4, replace=False)]
        chk = dict(GW_vs_autograd=float((GW - gfull[id(blk.mlp.up.weight)]).abs().max() / gfull[id(blk.mlp.up.weight)].abs().max()))
        metrics = ["A"] + (["D"] if Dm is not None else [])
        gth = [gfull[id(p)] if gfull[id(p)] is not None else torch.zeros_like(p) for p in theta]
        Ac = {mt: torch.zeros(C.FF, C.D, dtype=torch.float64, device=DEV) for mt in metrics}
        for k in (range(0) if light else range(C.D)):         # ȳ のヤコビアンの行ごと（light では ② と ③ を分けず合計だけ）
            rk = torch.autograd.grad(ybar[k], theta, retain_graph=True, allow_unused=True)
            rk = [r if r is not None else torch.zeros_like(p) for r, p in zip(rk, theta)]
            for mt in metrics:
                dirs = rk if mt == "A" else [Dm[id(p)] * r for p, r in zip(theta, rk)]
                dY = jvp_y(mfd, X, l, thfd, dirs) if JVP_MODE == "fd" else jvp_exact(y, theta, dirs)
                Ac[mt][:, k] = (e * (dY @ W.T)).sum(0)
        # 逆伝播で g^{FFNℓ} と ⟨∇m_j, g^{FFNℓ}⟩（中心差分の検査用）
        gffn = torch.autograd.grad((e * (y @ W.T)).sum(), theta, retain_graph=True, allow_unused=True)
        gffn = [g if g is not None else torch.zeros_like(p) for g, p in zip(gffn, theta)]
        xhat = F.layer_norm(xin[l].detach(), (C.D,), eps=blk.n2.eps).reshape(-1, C.D); xbar = xhat.mean(0); gam = id(blk.n2.weight)
        for mt in metrics:
            Cm = Ac[mt] @ W.T                                    # C_ij（行 i = 信号の持ち主、列 j = 位置を動かされる unit）
            wg = (lambda gs: gs) if mt == "A" else (lambda gs: [Dm[id(p)] * g for p, g in zip(theta, gs)])
            jv = (lambda dd: jvp_y(mfd, X, l, thfd, dd)) if JVP_MODE == "fd" else (lambda dd: jvp_exact(y, theta, dd))
            uT = jv(wg(gth)).mean(0); T = -(W @ uT)                       # −⟨v_j, g_θ⟩
            uF = jv(wg(gffn)).mean(0); SF = W @ uF                         # ⟨v_j, g^{FFN}⟩（逆伝播側）
            own = -(e * Kn[:, None]).sum(0) if mt == "A" else -((Dm[id(blk.mlp.up.weight)] * GW) @ yb)
            selfu = -torch.diagonal(Cm); oth_a = -(Cm.sum(0) - torch.diagonal(Cm)); oth_b = T - selfu - oth_a
            # ③ のうち LN2_ℓ の γ の分（解析的）: v_j[γ]_d = (D_γ)·W_jd x̄_d、g^{(j)}[γ]_d = W_jd Σ_n e_jn x̂_nd
            vg = W * xbar[None, :] * (1.0 if mt == "A" else Dm[gam][None, :])
            gj = W * (e.T @ xhat); gt = gfull[gam][None, :]
            oth_gamma = -(vg * (gt - gj)).sum(1)
            scale = (1.0 if mt == "A" else LR) / (sd + 1e-30)
            Coff = Cm.clone(); Coff.fill_diagonal_(0)
            topi = torch.topk(Coff.abs(), top, dim=0).indices        # 列 j ごとに寄与の大きい他人 i
            Wn = W / W.norm(dim=1, keepdim=True)
            res[mt] = dict(own=(own * scale).tolist(), self_up=(selfu * scale).tolist(), others_same=(oth_a * scale).tolist(), others_path=(oth_b * scale).tolist(),
                           others_gamma=(oth_gamma * scale).tolist(),
                           top_others=topi.T.tolist(), top_contrib=(-(torch.gather(Cm, 0, topi)) * scale[None, :]).T.tolist(),
                           top_cos=torch.gather(Wn @ Wn.T, 0, topi).T.tolist())
            chk[f"jvp_vs_backward_{mt}"] = float((Cm.sum(0) - SF).abs().max() / (SF.abs().max() + 1e-300))
            chk[f"MUT_transposed_C_{mt}"] = float((Cm.sum(1) - SF).abs().max() / (SF.abs().max() + 1e-300))      # 変異: C_ji を使う
            errs = []
            for j in ([] if light else diag_units):
                gm = torch.autograd.grad((W[j] * ybar).sum(), theta, retain_graph=True, allow_unused=True)
                gm = [g if g is not None else torch.zeros_like(p) for g, p in zip(gm, theta)]
                gjj = torch.autograd.grad((e[:, j] * (y @ W[j])).sum(), theta, retain_graph=True, allow_unused=True)
                gjj = [g if g is not None else torch.zeros_like(p) for g, p in zip(gjj, theta)]
                vj = gm if mt == "A" else [Dm[id(p)] * g for p, g in zip(theta, gm)]
                direct = sum((a_ * b_).sum() for a_, b_ in zip(vj, gjj))
                errs.append(float(abs(direct - Cm[j, j]) / (abs(direct) + 1e-300)))
            chk[f"diag_direct_{mt}"] = max(errs) if errs else float("nan")
        res["sigma"] = sd.tolist(); res["alive"] = alive.tolist(); res["K_mean"] = float(Kn.mean()); res["check"] = chk
        out["layers"][str(l)] = res
        log(f"  [{arm} t{task} {kind}] L{l}: {time.time() - t0:.0f}s 検査 {json.dumps({k: f'{v:.1e}' for k, v in chk.items()})}")
    out["_model"] = m; out["_X"] = X; out["_gfull"] = gfull; out["_Dm"] = Dm
    return out


def closure_check(arm, sd, adam_sd, task, res, kind="chain", nseq=64, layers=None):
    """全パラメータを η だけ動かした実際の Δm_j と η(① + ② + ③) の一致。A は SGD、D は Adam の分母つきの 1 歩（運動量なし）。"""
    out = {}
    for mt in (["A", "D"] if adam_sd is not None else ["A"]):
        for eta in (1e-4, 1e-5) if mt == "A" else (1e-5, 1e-6):
            m0 = build(arm, sd); X = probe_of(kind, task, nseq); g = res["_gfull"]; Dm = res["_Dm"]
            m1 = build(arm, sd)
            with torch.no_grad():
                for (p0, p1), pr in zip(zip(m0.parameters(), m1.parameters()), res["_model"].parameters()):
                    gg = g[id(pr)]
                    if gg is None: continue
                    step = gg if mt == "A" else Dm[id(pr)] * gg
                    p1.sub_(eta * step)
            for l in (layers or range(len(m0.h))):
                y0, y1 = y_at(m0, X, l).mean(0), y_at(m1, X, l).mean(0)
                dm = (m1.h[l].mlp.up.weight.detach() @ y1) - (m0.h[l].mlp.up.weight.detach() @ y0)
                r = res["layers"][str(l)][mt]; sd_ = torch.tensor(res["layers"][str(l)]["sigma"], dtype=torch.float64, device=DEV)
                scale = (1.0 if mt == "A" else LR) / (sd_ + 1e-30)
                pred = eta * torch.tensor([a + b + c + d for a, b, c, d in zip(r["own"], r["self_up"], r["others_same"], r["others_path"])], dtype=torch.float64, device=DEV) / scale
                out[f"{mt}_eta{eta:g}_L{l}"] = float((dm - pred).norm() / dm.norm())
                mut = eta * torch.tensor([a + b + c for a, b, c in zip(r["own"], r["self_up"], r["others_same"])], dtype=torch.float64, device=DEV) / scale   # 変異: ③b を落とす
                out[f"MUT_no_path_{mt}_eta{eta:g}_L{l}"] = float((dm - mut).norm() / dm.norm())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--state", required=True); ap.add_argument("--adam", default=None)
    ap.add_argument("--task", type=int, required=True, help="probe と誤差を取る課題（新しい課題 = k+1）")
    ap.add_argument("--kind", default="chain"); ap.add_argument("--nseq", type=int, default=64); ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    ap.add_argument("--light", action="store_true", help="① と ②+③ の合計だけ（一様 probe の比較用）")
    ap.add_argument("--device", default="cpu", help="cuda なら float64 のまま GPU で計算する")
    ap.add_argument("--jvp", default="fd", choices=["fd", "exact"], help="方向微分: fd = 中心差分、exact = 二重逆伝播（ReLU の折れ目でも厳密）")
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.set_default_dtype(torch.float32)
    global DEV, JVP_MODE; DEV = torch.device(a.device); JVP_MODE = a.jvp
    sd = torch.load(a.state, map_location="cpu"); adam = torch.load(a.adam, map_location="cpu") if a.adam else None
    res = analyze(a.arm, sd, adam, a.task, a.kind, a.nseq, a.layers, light=a.light)
    if a.check:
        res["closure"] = closure_check(a.arm, sd, adam, a.task, res, a.kind, a.nseq, a.layers)
        flip = analyze(a.arm, sd, adam, a.task, a.kind, a.nseq, a.layers, flip=True, log=lambda s: None)
        fc = {}
        for l in res["layers"]:
            for mt in ("A", "D"):
                if mt not in res["layers"][l]: continue
                worst = 0.0
                for key in ("own", "self_up", "others_same", "others_path"):
                    xs, ys = res["layers"][l][mt][key], flip["layers"][l][mt][key]
                    den = max(abs(v) for v in xs) + 1e-300
                    worst = max(worst, max(abs(x + y) for x, y in zip(xs, ys)) / den)
                fc[f"{mt}_L{l}"] = worst
        res["flip_check"] = fc
        print("closure:", json.dumps({k: f"{v:.1e}" for k, v in res["closure"].items()}), "\nflip:", json.dumps({k: f"{v:.1e}" for k, v in res["flip_check"].items()}))
    for k in ("_model", "_X", "_gfull", "_Dm"): res.pop(k, None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res))
    print("written", a.out)


if __name__ == "__main__":
    main()
