#!/usr/bin/env python3
"""ρ の分割（rho_split.py の出力）の集計と、伸び・回転の分割、実際の窓の分解との橋渡し（事後・判定外）。
  python analysis/chargpt_cl_1019_quicklook/rho_summary.py --out results/chargpt_cl_1019/quicklook/rho/summary
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
Q = ROOT / "results/chargpt_cl_1019/quicklook"
BLOCKS = {"C3": {"t2-10": range(2, 11), "t40-50": range(40, 51)},
          "SiLU": {"t2-10": range(2, 11), "t11-15": range(11, 16), "t40-50": range(40, 51), "t60-100": range(60, 101, 5)}}


def load(arm, t, st, kind="chain"):
    f = Q / "rho/out" / arm / f"t{t}_{st}_{kind}.json"
    return json.loads(f.read_text()) if f.exists() else None


def pool(arm, ts, st, l, mt, kind="chain"):
    cols = {k: [] for k in ("own", "self_up", "others_same", "others_path", "others_gamma", "alive", "top_contrib", "top_cos")}
    n = 0
    for t in ts:
        r = load(arm, t, st, kind)
        if r is None or str(l) not in r["layers"] or mt not in r["layers"][str(l)]: continue
        L = r["layers"][str(l)]; M = L[mt]; n += 1
        for k in ("own", "self_up", "others_same", "others_path", "others_gamma"):
            if k in M: cols[k] += M[k]
        cols["alive"] += L["alive"]
        if "top_contrib" in M: cols["top_contrib"] += M["top_contrib"]; cols["top_cos"] += M["top_cos"]
    return n, {k: np.array(v, dtype=float) for k, v in cols.items()}


def stats(c):
    own, sf, oa, ob = c["own"], c["self_up"], c["others_same"], c["others_path"]
    oth = oa + ob; rho = np.abs(own + sf) / (np.abs(oth) + 1e-300)
    nz = (np.abs(own) + np.abs(sf)) > 0                       # 厳密に 0 の unit（ReLU の死）を除く
    mag = np.abs(own) + np.abs(sf); big = nz & (mag >= np.median(mag[nz])) if nz.any() else nz
    agree = np.sign(own) == np.sign(sf)
    q = lambda x: np.percentile(x, [25, 50, 75]) if len(x) else [np.nan] * 3
    out = dict(n_units=int(len(own)), rho_q=[float(v) for v in q(rho[nz])], rho_gt1=float((rho[nz] > 1).mean()) if nz.any() else np.nan,
               med_own=float(np.median(own)), med_self=float(np.median(sf)), med_oth_a=float(np.median(oa)), med_oth_b=float(np.median(ob)),
               neg_own=float((own < 0).mean()), neg_self=float((sf < 0).mean()), neg_oth_a=float((oa < 0).mean()), neg_oth_b=float((ob < 0).mean()),
               mean_own=float(own.mean()), mean_self=float(sf.mean()), mean_oth_a=float(oa.mean()), mean_oth_b=float(ob.mean()),
               agree_nz=float(agree[nz].mean()) if nz.any() else np.nan, agree_big=float(agree[big].mean()) if big.any() else np.nan,
               agree_wt=float((mag[nz] * agree[nz]).sum() / mag[nz].sum()) if nz.any() else np.nan,
               agree_alive=float(agree[nz & (c["alive"] > 0)].mean()) if (nz & (c["alive"] > 0)).any() else np.nan,
               self_share_med=float(np.median((np.abs(sf) / (np.abs(own) + np.abs(sf) + 1e-300))[nz])) if nz.any() else np.nan,
               gamma_share=float(np.abs(c["others_gamma"]).sum() / (np.abs(oth).sum() + 1e-300)) if len(c["others_gamma"]) else np.nan,
               oth_same_vs_path=float(np.abs(oa).sum() / (np.abs(oa).sum() + np.abs(ob).sum() + 1e-300)),
               oth_sign_as_own=float((np.sign(oth) == np.sign(own))[nz].mean()) if nz.any() else np.nan)
    if len(c["top_contrib"]):
        tc = c["top_contrib"]; out["top5_share_of_oth_a"] = float(np.median(np.abs(tc).sum(1) / (np.abs(oa) + 1e-300)))
        out["top5_cos_med"] = float(np.median(c["top_cos"])); out["top5_cos_abs_med"] = float(np.median(np.abs(c["top_cos"])))
    return out


@torch.no_grad()
def stretch_rotation(arm, k, l_list):
    """窓（課題 k の末 → 課題 k+1 の更新 100）の実際の w_j·Δȳ を、伸び (w_j·ŷ)(ŷ·Δȳ) と回転 w_j·Δȳ_⊥ に割る（ŷ = ȳ_a/‖ȳ_a‖、σ_a で割る）。"""
    t = k + 1; pa = Q / "M" / arm / "seed1000/ckpts" / f"task{k}.pt"
    pb = Q / "rho/states" / arm / f"u100_task{t}.pt"
    if not pb.exists(): pb = Q / "rho/cont" / f"{arm}_states" / f"u100_task{t}.pt"
    if not pa.exists():
        for cand in (Q / "rho/cont").glob(f"{arm}_*/ckpts/task{k}.pt"): pa = cand
    if not (pa.exists() and pb.exists()): return None
    probe = C.task_cpu_M(1000, t, 10, 0.01)["probe"][:32]; out = {}
    ms = []
    for p in (pa, pb):
        m = C.build_model(arm, "M", 1000, "cpu"); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); cap = []; m.body(probe, cap); ms.append((m, cap))
    for l in l_list:
        (ma, ca), (mb, cb) = ms
        ya = ma.h[l].n2(ca[2 * l]).reshape(-1, C.D).double(); yb = mb.h[l].n2(cb[2 * l]).reshape(-1, C.D).double()
        Wa = ma.h[l].mlp.up.weight.double(); sd = (ya @ Wa.T).std(0)
        ybar, dy = ya.mean(0), yb.mean(0) - ya.mean(0); yh = ybar / ybar.norm()
        st = (Wa @ yh) * (yh @ dy) / sd; rot = (Wa @ (dy - (yh @ dy) * yh)) / sd
        out[str(l)] = (st.numpy(), rot.numpy())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(Q / "rho/summary")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows, lines = [], ["# ρ の分割の集計（事後・判定外、seed 1000、4 ブロック）", ""]
    for arm, blocks in BLOCKS.items():
        for bname, ts in blocks.items():
            for st in ("a", "b"):
                for mt in ("A", "D"):
                    for l in range(4):
                        n, c = pool(arm, list(ts), st, l, mt)
                        if n == 0: continue
                        s = stats(c); s.update(arm=arm, block=bname, state=st, metric=mt, layer=l, n_windows=n); rows.append(s)
    # 伸びと回転
    sr = []
    for arm, blocks in BLOCKS.items():
        for bname, ts in blocks.items():
            acc = {l: ([], []) for l in range(4)}
            for t in ts:
                r = stretch_rotation(arm, t - 1, range(4))
                if r is None: continue
                for l in range(4): acc[l][0].append(r[str(l)][0]); acc[l][1].append(r[str(l)][1])
            for l in range(4):
                if not acc[l][0]: continue
                s_, r_ = np.concatenate(acc[l][0]), np.concatenate(acc[l][1])
                sr.append(dict(arm=arm, block=bname, layer=l, med_stretch=float(np.median(s_)), med_rot=float(np.median(r_)),
                               stretch_share=float(np.abs(s_).sum() / (np.abs(s_).sum() + np.abs(r_).sum())),
                               stretch_share_neg=float(np.abs(s_[(s_ + r_) < 0]).sum() / (np.abs(s_[(s_ + r_) < 0]).sum() + np.abs(r_[(s_ + r_) < 0]).sum() + 1e-300)),
                               stretch_share_pos=float(np.abs(s_[(s_ + r_) > 0]).sum() / (np.abs(s_[(s_ + r_) > 0]).sum() + np.abs(r_[(s_ + r_) > 0]).sum() + 1e-300))))
    # 実際の窓の分解との橋渡し（窓 i、unit の平均）
    bridge = []
    for arm in ("C3", "SiLU"):
        f = Q / f"decomp_dmu_{arm}" / "decomp_dmu_summary.csv"
        if not f.exists(): continue
        act = {(r["block"], int(r["layer"])): r for r in csv.DictReader(open(f)) if r["window"] == "i"}
        for r in rows:
            if r["arm"] != arm or r["state"] != "a" or (r["block"], r["layer"]) not in act: continue
            ar = act[(r["block"], r["layer"])]; aown = float(ar["own"]); aup = float(ar["upstream"])
            bridge.append(dict(arm=arm, block=r["block"], layer=r["layer"], metric=r["metric"], step_own=r["mean_own"], step_up=r["mean_self"] + r["mean_oth_a"] + r["mean_oth_b"],
                               act_own=aown, act_up=aup))
    (out / "rho_summary.json").write_text(json.dumps(dict(rows=rows, stretch_rotation=sr, bridge=bridge), indent=1, ensure_ascii=False))
    for r in rows:
        lines.append(f"{r['arm']:4s} {r['block']:7s} 状態{r['state']} {r['metric']} L{r['layer']} (窓 {r['n_windows']}): ρ 四分位 {r['rho_q'][0]:.2f}/{r['rho_q'][1]:.2f}/{r['rho_q'][2]:.2f} ρ>1 {r['rho_gt1']:.2f}"
                     f" | 中央 ① {r['med_own']:+.2e} ② {r['med_self']:+.2e} ③a {r['med_oth_a']:+.2e} ③b {r['med_oth_b']:+.2e} | 負の割合 {r['neg_own']:.2f}/{r['neg_self']:.2f}/{r['neg_oth_a']:.2f}/{r['neg_oth_b']:.2f}"
                     f" | ①②符号一致 0除く {r['agree_nz']:.2f} 大きい半分 {r['agree_big']:.2f} 重み付き {r['agree_wt']:.2f} | ②/(①+②) {r['self_share_med']:.2f} | ③の符号が①と同じ {r['oth_sign_as_own']:.2f}"
                     f" | ③a/(③a+③b) {r['oth_same_vs_path']:.2f} γ分 {r['gamma_share']:.2f}" + (f" | ③a 上位5の割合 {r['top5_share_of_oth_a']:.2f} cos中央 {r['top5_cos_med']:+.2f}" if 'top5_cos_med' in r else ""))
    lines += ["", "## 伸びと回転（実際の窓の w·Δȳ、σ で割る）"] + [f"{s['arm']:4s} {s['block']:7s} L{s['layer']}: 中央 伸び {s['med_stretch']:+.3f} 回転 {s['med_rot']:+.3f} | 伸びの割合 全 {s['stretch_share']:.2f} 沈む unit {s['stretch_share_neg']:.2f} 浮く unit {s['stretch_share_pos']:.2f}" for s in sr]
    lines += ["", "## 1 歩の絵と実際の窓（unit の平均。① 対 ②+③、実際は 自分の重み 対 上流）"] + [f"{b['arm']:4s} {b['block']:7s} L{b['layer']} {b['metric']}: 1 歩 ① {b['step_own']:+.2e} ②+③ {b['step_up']:+.2e}（比 {b['step_own']/(b['step_up'] if b['step_up'] else np.nan):+.2f}）| 実際 自分 {b['act_own']:+.3f} 上流 {b['act_up']:+.3f}（比 {b['act_own']/(b['act_up'] if b['act_up'] else np.nan):+.2f}）" for b in bridge]
    (out / "rho_summary.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
