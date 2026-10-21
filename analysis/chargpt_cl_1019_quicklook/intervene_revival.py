#!/usr/bin/env python3
"""2 ブロックの C3-2 で L1 が生き返る仕組みを介入で分ける（事後・判定外・未登録、seed 1000）。
課題 T の末（重み＋Adam）から再生し、各課題末に L1 の死・最終 LN の分母・L1 の有効な読み出しを測る。
腕（--arm）:
  control   そのまま（元の走の online.npz とビット一致するはず）
  lnfix     最終 LN の分母（トークンごとの √(Var_d(h)+ε)）を、課題 T の末の probe での平均 c に固定する。中心化と γ はそのまま
  lnex1     最終 LN の分母を FFN1 の出力を除いた残差 h − f₁ から取る（k·√(Var_d(h − f₁)+ε)、k は課題 T の末で平均の分母が元と等しくなる定数）。
            生き返る L1 自身の書き込みでは読み出しが縮まない。他の成分の尺度の変化はふつうに正規化される
  nocommon  FFN1（読み出しに隣接する FFN）の出力から、バッチのトークン平均（共通の書き込み）を各 forward で引く
  l0freeze  FFN0 の up・down と LN2₀ の γ を、各更新のあとで課題 T の末の値に戻す（Adam の状態は動くが重みは動かない）
  gamma03   最終 LN の γ を課題 T の末に 0.3 倍してから続ける（γ は学習で戻れる）
  gamma03f  最終 LN の γ を 0.3 倍して、そのまま固定する
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/intervene_revival.py --arm lnfix --from_task 40 --to_task 55 --out <dir>
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
SRC = ROOT / "results/chargpt_cl_1019/quicklook/M2/C3/seed1000"

# 各更新のあとで重みを戻す（グラフの中に取り込む）
FREEZE = []
DEN_FN = []          # lnex1: 実際に使う分母の関数（extras が読む）
_prev_body = C.Engine._step_body


def _step_body_freeze(s):
    out = _prev_body(s)
    with torch.no_grad():
        for p, q in FREEZE:
            p.copy_(q)
    return out


C.Engine._step_body = _step_body_freeze


def ln_in_denoms(model, x):
    """最終 LN の入力 h（トークン × D）と、トークンごとの分母 √(Var_d(h)+ε)。"""
    got = {}
    hk = model.nf.register_forward_hook(lambda mod, inp, out: got.__setitem__("h", inp[0].detach()))
    with torch.no_grad(): model.body(x)
    hk.remove()
    h = got["h"].reshape(-1, C.D).double()
    return h, (h.var(1, unbiased=False) + model.nf.eps).sqrt()


@torch.no_grad()
def extras(model, x, denom_fixed=None):
    """各課題末の追加の量: 最終 LN の入力 rms と分母、FFN・注意の出力 rms、FFN1 の共通成分の大きさ、γ_f、有効な読み出し。"""
    got = {}
    hooks = [model.nf.register_forward_hook(lambda mod, inp, out: got.__setitem__("nf_in", inp[0].detach()))]
    for l, b in enumerate(model.h):
        hooks.append(b.mlp.register_forward_hook(lambda mod, inp, out, l=l: got.__setitem__(f"ffn{l}", out.detach())))
        hooks.append(b.attn.register_forward_hook(lambda mod, inp, out, l=l, sc=b.sc: got.__setitem__(f"attn{l}", (sc * out).detach())))
    model.body(x)
    for hk in hooks: hk.remove()
    rms = lambda t: float(t.double().pow(2).mean().sqrt())
    h = got["nf_in"].reshape(-1, C.D).double()
    den = (h.var(1, unbiased=False) + model.nf.eps).sqrt()
    den_used = float(denom_fixed) if denom_fixed is not None else (float(DEN_FN[0](got["nf_in"]).double().mean()) if DEN_FN else float(den.mean()))
    E = model.head.weight.double(); gf = model.nf.weight.double()
    r = dict(nf_in_rms=rms(h), denom_mean=float(den.mean()), denom_used=den_used, gamma_f_med=float(gf.median()), gamma_f_mean=float(gf.mean()))
    for l, b in enumerate(model.h):
        f = got[f"ffn{l}"].reshape(-1, C.D).double()
        r[f"ffn{l}_rms"] = rms(f); r[f"ffn{l}_common_norm"] = float(f.mean(0).norm()); r[f"attn{l}_rms"] = rms(got[f"attn{l}"])
        Wd = b.mlp.down.weight.double()
        veff = (E @ (gf[:, None] * Wd)).norm(dim=0) / den_used        # unit j の活性化 1 単位が直通でロジットを動かす大きさ
        r[f"veff{l}_med"] = float(veff.median()); r[f"v{l}_med"] = float(Wd.norm(dim=0).median()); r[f"w{l}_med"] = float(b.mlp.up.weight.double().norm(dim=1).median())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "lnfix", "lnex1", "nocommon", "l0freeze", "gamma03", "gamma03f"])
    ap.add_argument("--from_task", type=int, required=True); ap.add_argument("--to_task", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--n_per_task", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--src", default=str(SRC))
    a = ap.parse_args(); assert C.L == 2, "CHARGPT_NLAYER=2 で起動する"
    C.setup_torch(); dev = torch.device("cuda"); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gh, dirty = C.git_info(); t0 = time.time()
    model = C.build_model("C3", "M", a.seed, dev)
    model.load_state_dict(torch.load(src / "ckpts" / f"task{a.from_task}.pt", map_location=dev))
    # 介入（グラフを取る前に入れる）
    info = dict(arm=a.arm)
    probe_T = C.task_cpu_M(a.seed, a.from_task, 10, a.alpha)["probe"].to(dev)
    denom_fixed = None
    if a.arm == "lnfix":
        _, den = ln_in_denoms(model, probe_T); c = float(den.mean()); denom_fixed = c; info["denom_fixed"] = c
        nf = model.nf
        nf.forward = lambda x, nf=nf, c=c: (x - x.mean(-1, keepdim=True)) / c * nf.weight
    if a.arm == "lnex1":
        mlp1 = model.h[1].mlp; orig1 = mlp1.forward

        def fwd1(y, cap=None, orig=orig1, mod=mlp1):
            o = orig(y, cap); mod._last_out = o
            return o
        mlp1.forward = fwd1
        nf = model.nf; den_of = lambda x, nf=nf, mod=mlp1: ((x - mod._last_out).var(-1, unbiased=False, keepdim=True) + nf.eps).sqrt()
        h, den = ln_in_denoms(model, probe_T)
        with torch.no_grad(): model.body(probe_T)
        k = float(den.mean() / den_of(h.reshape(mlp1._last_out.shape).float()).mean()); info["lnex1_k"] = k
        nf.forward = lambda x, nf=nf, k=k: (x - x.mean(-1, keepdim=True)) / (k * den_of(x)) * nf.weight
        DEN_FN.append(lambda x, k=k: k * den_of(x))
    if a.arm == "nocommon":
        mlp = model.h[1].mlp; orig = mlp.forward

        def fwd(y, cap=None, orig=orig):
            o = orig(y, cap)
            return o - o.mean((0, 1), keepdim=True)
        mlp.forward = fwd
    if a.arm in ("gamma03", "gamma03f"):
        with torch.no_grad(): model.nf.weight.mul_(0.3)
        info["gamma_f_after_scale_med"] = float(model.nf.weight.median())
    if a.arm == "l0freeze":
        for p in (model.h[0].mlp.up.weight, model.h[0].mlp.down.weight, model.h[0].n2.weight): FREEZE.append((p, p.detach().clone()))
    if a.arm == "gamma03f":
        FREEZE.append((model.nf.weight, model.nf.weight.detach().clone()))
    eng = C.Engine(model, "C3", "M", graph=True)
    sd = torch.load(src / "ckpts" / f"task{a.from_task}_adam.pt", map_location=dev)
    params = list(model.parameters()); assert len(sd["state"]) == len(params)
    with torch.no_grad():
        for i, p in enumerate(params):
            st, sv = eng.opt.state[p], sd["state"][i]
            for k in ("exp_avg", "exp_avg_sq", "step"): st[k].copy_(sv[k].to(st[k].dtype))
    probe_u = C.uniform_probe_M(a.seed, dev)
    rows, layer_rows, ex_rows = [], [], []
    tasks = list(range(a.from_task + 1, a.to_task + 1)); n = a.n_per_task
    online = np.zeros((len(tasks), n), np.float32)
    ex_rows.append(dict(task=a.from_task, **extras(model, probe_T, denom_fixed)))          # 介入の直後（課題 T の末）
    for li, r in enumerate(C.measure(model, probe_T)): layer_rows.append(dict(task=a.from_task, u=n, probe="chain", layer=li, **r["agg"]))
    for ti, t in enumerate(tasks):
        cpu = C.task_cpu_M(a.seed, t, n, a.alpha); ctx = C.TaskCtx("M", a.seed, t, n, dev, alpha=a.alpha, W=None, cpu=cpu)
        tt = time.time()
        loss, aux = C.run_task(model, eng, ctx, warmup_from=(t - 1) * n); online[ti] = loss
        for p, xp in dict(chain=ctx.probe, uniform=probe_u).items():
            for li, r in enumerate(C.measure(model, xp)): layer_rows.append(dict(task=t, u=n, probe=p, layer=li, **r["agg"]))
        ex_rows.append(dict(task=t, **extras(model, ctx.probe, denom_fixed)))
        endm = C.eval_M(model, ctx.eval_seq, ctx.H)
        rows.append(dict(task=t, n=n, H=ctx.H, **C.summarize_online("M", loss, aux, ctx.H), end_metric=endm, data_sha=ctx.data_sha[:16]))
        L1 = [r for r in layer_rows if r["task"] == t and r["probe"] == "chain" and r["layer"] == 1][0]
        L0 = [r for r in layer_rows if r["task"] == t and r["probe"] == "chain" and r["layer"] == 0][0]
        print(f"[{a.arm}] task {t}/{a.to_task} online {rows[-1]['online_all']:.4f} | 死 L0 {L0['frac_p_lt005']:.2f} L1 {L1['frac_p_lt005']:.2f} | μ/σ L0 {L0['r_med']:+.2f} L1 {L1['r_med']:+.2f}"
              f" | LN 入力 rms {ex_rows[-1]['nf_in_rms']:.1f} FFN1 rms {ex_rows[-1]['ffn1_rms']:.1f} v_eff1 {ex_rows[-1]['veff1_med']:.3f} γ_f {ex_rows[-1]['gamma_f_med']:.3f} | {time.time() - tt:.0f}s", flush=True)
        del ctx
    torch.save(model.state_dict(), out / f"state_task{a.to_task}.pt")

    def wcsv(path, rws):
        keys = list(dict.fromkeys(k for r in rws for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rws: w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    wcsv(out / "per_task.csv", rows); wcsv(out / "layer_stats.csv", layer_rows); wcsv(out / "extras.csv", ex_rows)
    np.savez_compressed(out / "online.npz", loss=online)
    info["freeze_max_abs_diff"] = max([float((p - q).abs().max()) for p, q in FREEZE], default=None)     # 検査: 固定した重みは最後まで動かない
    match = None
    if a.arm == "control":
        ref = np.load(src / "online.npz")["loss"][a.from_task:a.to_task]
        match = bool(np.array_equal(ref, online)); print(f"[control] 元の走の online（課題 {a.from_task + 1}〜{a.to_task}）とビット一致: {match}", flush=True)
    (out / "provenance.json").write_text(json.dumps(dict(script="intervene_revival.py", src=str(src), from_task=a.from_task, to_task=a.to_task, seed=a.seed, n_per_task=n, n_layer=C.L,
                                                       git_hash=gh, dirty_src=dirty, gpu=torch.cuda.get_device_name(0), host=__import__("socket").gethostname(),
                                                       seconds=time.time() - t0, control_bitwise_match=match, grade="事後・判定外・未登録", **info), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
