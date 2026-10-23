#!/usr/bin/env python3
"""chargpt_seed10_1022 の C（L1 の生き返りの帰還）: 保存状態からの再生。
写し元: $CG/analysis/chargpt_cl_1019_quicklook/intervene_revival.py
        （lop_analysis branch claude/chargpt_cl_1019、commit a970f12c = HEAD 033679e0 で同じ blob f9b7ccde29079466d45c5b3fabde3bc462f7b63b）
腕の定義（control・lnex1・nocommon・gamma03・gamma03f・lnfix・l0freeze）は写し元のまま変えない。
変えた所:
  1. ROOT を環境変数 CHARGPT_SRC（= $CG）から取る。
  2. --model_arm（C3 か ReLUcap）を足した。網の構築と Engine の腕に使う（写し元は "C3" 固定）。C3 の部（ReLUcap-2）用。
  3. 開始の課題 T の末にも一様 probe で C.measure を取る（写し元は chain だけ）。各課題末に、開始の課題 T の chain probe でも測る（probe="chain_start"）。
     写し元の「chain」（その課題の chain probe）と「uniform」（固定の一様 probe）はそのまま残す。
  4. 元の走の記録との照合を全腕で書く（compare_orig.csv）: 課題ごとの online の差の最大・ビット一致、
     元の layer_stats.csv（u=10000）の L0・L1 の frac_p_lt005 との差（uniform と chain）。provenance に control の要約を書く。
  5. lnex1 の定数の検査: 課題 T の chain probe で、介入後の平均の分母（k·√(Var(h−f₁)+ε) の平均）と元の平均の分母の相対差を provenance に書く。
  6. 各課題の秒（sec）を per_task.csv に、全体の秒を provenance に書く。最後の重みは --raw_out に *.raw.pt で置く（写し元は out に置いた）。
  7. 走の途中の失敗で半端な結果を完走と読まないよう、provenance.json は最後に書き、complete=true を入れる。
  CHARGPT_SRC=$CG CHARGPT_NLAYER=2 python revival10.py --arm control --from_task 25 --to_task 31 --seed 0 --src $CG/results/chargpt_cl_1019/M2/C3/seed0 --out <dir> --raw_out <dir>
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(os.environ["CHARGPT_SRC"]).resolve(); sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C

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


def orig_layer(orig_dir):
    """元の走の layer_stats.csv の課題末（u=10000）: {(task, probe, layer): frac_p_lt005}。"""
    out = {}
    p = Path(orig_dir) / "layer_stats.csv"
    if not p.exists():
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["u"] == "10000":
                out[(int(r["task"]), r["probe"], int(r["layer"]))] = float(r["frac_p_lt005"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "lnfix", "lnex1", "nocommon", "l0freeze", "gamma03", "gamma03f"])
    ap.add_argument("--from_task", type=int, required=True); ap.add_argument("--to_task", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n_per_task", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--src", required=True)
    ap.add_argument("--model_arm", default="C3", choices=["C3", "ReLUcap"])
    ap.add_argument("--orig", default=None, help="元の走の記録（online.npz・layer_stats.csv）。既定は --src")
    ap.add_argument("--raw_out", default=None, help="最後の重み（*.raw.pt）の置き場。既定は保存しない")
    a = ap.parse_args(); assert C.L == 2, "CHARGPT_NLAYER=2 で起動する"
    C.setup_torch(); dev = torch.device("cuda"); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    orig = Path(a.orig) if a.orig else src
    gh, dirty = C.git_info(); t0 = time.time()
    model = C.build_model(a.model_arm, "M", a.seed, dev)
    model.load_state_dict(torch.load(src / "ckpts" / f"task{a.from_task}.pt", map_location=dev))
    # 介入（グラフを取る前に入れる）
    info = dict(arm=a.arm, model_arm=a.model_arm)
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
        # 追加（変えた所 5）: 定数の検査。介入後の forward で、最終 LN が実際に使う分母の平均を元の平均の分母と比べる
        got = {}
        hk = model.nf.register_forward_hook(lambda mod, inp, o: got.__setitem__("x", inp[0].detach()))
        with torch.no_grad(): model.body(probe_T)
        hk.remove()
        used = float((k * den_of(got["x"])).double().mean()); base = float(den.mean())
        info.update(lnex1_den_orig_mean=base, lnex1_den_used_mean=used, lnex1_den_rel_diff=abs(used - base) / base)
    if a.arm == "nocommon":
        mlp = model.h[1].mlp; orig_f = mlp.forward

        def fwd(y, cap=None, orig=orig_f):
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
    eng = C.Engine(model, a.model_arm, "M", graph=True)
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
    for li, r in enumerate(C.measure(model, probe_u)): layer_rows.append(dict(task=a.from_task, u=n, probe="uniform", layer=li, **r["agg"]))   # 追加（変えた所 3）
    for ti, t in enumerate(tasks):
        cpu = C.task_cpu_M(a.seed, t, n, a.alpha); ctx = C.TaskCtx("M", a.seed, t, n, dev, alpha=a.alpha, W=None, cpu=cpu)
        tt = time.time()
        loss, aux = C.run_task(model, eng, ctx, warmup_from=(t - 1) * n); online[ti] = loss
        for p, xp in dict(chain=ctx.probe, uniform=probe_u, chain_start=probe_T).items():
            for li, r in enumerate(C.measure(model, xp)): layer_rows.append(dict(task=t, u=n, probe=p, layer=li, **r["agg"]))
        ex_rows.append(dict(task=t, **extras(model, ctx.probe, denom_fixed)))
        endm = C.eval_M(model, ctx.eval_seq, ctx.H)
        torch.cuda.synchronize()
        rows.append(dict(task=t, n=n, H=ctx.H, **C.summarize_online("M", loss, aux, ctx.H), end_metric=endm, data_sha=ctx.data_sha[:16], sec=time.time() - tt))
        Lr = {(r["probe"], r["layer"]): r for r in layer_rows if r["task"] == t}
        print(f"[{a.arm} s{a.seed}] task {t}/{a.to_task} online {rows[-1]['online_all']:.4f} | 死 uniform L0 {Lr[('uniform', 0)]['frac_p_lt005']:.2f} L1 {Lr[('uniform', 1)]['frac_p_lt005']:.2f}"
              f" chain L0 {Lr[('chain', 0)]['frac_p_lt005']:.2f} L1 {Lr[('chain', 1)]['frac_p_lt005']:.2f}"
              f" | LN 入力 rms {ex_rows[-1]['nf_in_rms']:.1f} FFN1 rms {ex_rows[-1]['ffn1_rms']:.1f} v_eff1 {ex_rows[-1]['veff1_med']:.3f} γ_f {ex_rows[-1]['gamma_f_med']:.3f} | {rows[-1]['sec']:.0f}s", flush=True)
        del ctx
    if a.raw_out:
        ro = Path(a.raw_out); ro.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ro / f"state_task{a.to_task}.raw.pt")

    def wcsv(path, rws):
        keys = list(dict.fromkeys(k for r in rws for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rws: w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    wcsv(out / "per_task.csv", rows); wcsv(out / "layer_stats.csv", layer_rows); wcsv(out / "extras.csv", ex_rows)
    np.savez_compressed(out / "online.npz", loss=online)
    info["freeze_max_abs_diff"] = max([float((p - q).abs().max()) for p, q in FREEZE], default=None)     # 検査: 固定した重みは最後まで動かない
    # 追加（変えた所 4）: 元の走の記録との照合
    ref_all = np.load(orig / "online.npz")["loss"] if (orig / "online.npz").exists() else None
    ol = orig_layer(orig)
    cmp_rows = []
    for t in [a.from_task] + tasks:
        r = dict(task=t)
        if t > a.from_task and ref_all is not None:
            ti = t - a.from_task - 1; ref = ref_all[t - 1]
            r.update(online_bitwise=bool(np.array_equal(ref, online[ti])), online_maxabs=float(np.abs(ref.astype(np.float64) - online[ti]).max()),
                     online_mean_diff=float(online[ti].astype(np.float64).mean() - ref.astype(np.float64).mean()), online_nonfinite=int((~np.isfinite(online[ti])).sum()))
        for p in ("uniform", "chain"):
            for li in (0, 1):
                mine = [x["frac_p_lt005"] for x in layer_rows if x["task"] == t and x["probe"] == p and x["layer"] == li]
                o = ol.get((t, p, li))
                r[f"{p}_L{li}_replay"] = mine[0] if mine else None
                r[f"{p}_L{li}_orig"] = o
                r[f"{p}_L{li}_diff"] = (mine[0] - o) if (mine and o is not None) else None
        cmp_rows.append(r)
    wcsv(out / "compare_orig.csv", cmp_rows)
    rest = [r for r in cmp_rows if r["task"] > a.from_task]
    match = bool(all(r.get("online_bitwise") for r in rest)) if ref_all is not None and rest else None
    summ = dict(online_bitwise_all=match,
                online_maxabs_max=max((r["online_maxabs"] for r in rest if "online_maxabs" in r), default=None),
                uniform_L1_absdiff_max=max((abs(r["uniform_L1_diff"]) for r in rest if r["uniform_L1_diff"] is not None), default=None),
                uniform_L0_absdiff_max=max((abs(r["uniform_L0_diff"]) for r in rest if r["uniform_L0_diff"] is not None), default=None),
                start_uniform_L1_diff=cmp_rows[0]["uniform_L1_diff"], start_uniform_L0_diff=cmp_rows[0]["uniform_L0_diff"],
                start_chain_L1_diff=cmp_rows[0]["chain_L1_diff"])
    if a.arm == "control":
        print(f"[control s{a.seed}] 元の走の online（課題 {a.from_task + 1}〜{a.to_task}）とビット一致: {match}・差の最大 {summ['online_maxabs_max']}・L1 の死（uniform）の差の最大 {summ['uniform_L1_absdiff_max']}", flush=True)
    me = Path(__file__).resolve()
    (out / "provenance.json").write_text(json.dumps(dict(script="revival10.py", script_sha256=hashlib.sha256(me.read_bytes()).hexdigest(),
                                                       copied_from="analysis/chargpt_cl_1019_quicklook/intervene_revival.py@a970f12c (blob f9b7ccde)",
                                                       src=str(src), orig=str(orig), from_task=a.from_task, to_task=a.to_task, seed=a.seed, n_per_task=n, n_layer=C.L,
                                                       git_hash_CG=gh, dirty_src=dirty, gpu=torch.cuda.get_device_name(0), host=__import__("socket").gethostname(),
                                                       torch=torch.__version__, seconds=time.time() - t0, gpu_max_mem_alloc_MB=torch.cuda.max_memory_allocated() / 2**20, sec_per_task=[r["sec"] for r in rows],
                                                       control_bitwise_match=match if a.arm == "control" else None, compare_orig=summ,
                                                       grade="chargpt_seed10_1022 C（spec §5・追補 1）", complete=True, **info), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
