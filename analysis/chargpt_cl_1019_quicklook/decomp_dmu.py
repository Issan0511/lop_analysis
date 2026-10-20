#!/usr/bin/env python3
"""位置の変化 Δμ を 自己項・上流・その他 に割る（下見・未登録、Neokunekune 経由の Issa の依頼 1020 夕方）。

  python analysis/chargpt_cl_1019_quicklook/decomp_dmu.py --arms C3 SiLU GELU --out results/chargpt_cl_1019/quicklook/decomp_dmu

対象は CHARGPT_DECOMP_CKPT=1 で回した腕（課題 1〜10・40〜50 の更新 100 と、課題末 = 次の課題の更新 0 の checkpoint がある）。
bias の無い腕だけ（μ_i = w_i·ȳ が厳密）。CPU だけで動く。

窓
  (i)  切替直後の 100 更新: 課題 t−1 の末（t = 1 は初期値）→ 課題 t の更新 100
  (ii) 課題の残り:          課題 t の更新 100 → 課題 t の末
どちらの窓も、窓の両端の網を同じ probe（課題 t の chain probe）で評価する（データの違いを除く）。

割り方（unit i、層ごと。a = 窓の始まり、b = 終わり。ȳ = FFN 入力 y = γ⊙x̂ のトークン平均）
  Δμ_i = Δw_i·ȳ_a ＋ w_a,i·Δȳ ＋ Δw_i·Δȳ    （恒等式。残差 resid を記録する）
窓 (i) では Δw_i·ȳ_a をさらに、自己形の向き d_i = Σ_n K_n φ(z_in) φ′(z_in) y_n（K_n = ȳ_a·y_n、状態 a・課題 t の probe）
への射影 (Δw_i·d̂_i)(d̂_i·ȳ_a) とその残りに割る。
  自己項 = 射影、上流 = w_a·Δȳ ＋ 交差、その他 = Δw·ȳ_a − 射影
どの項も窓の始まりの幅 σ_a,i で割る。層ごとに unit の平均（項の和が Δμ/σ の平均に一致する）と、Δμ/σ の中央値を出す。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C

Q = ROOT / "results/chargpt_cl_1019/quicklook/M"
SEED, ALPHA = 1000, 0.01
TASKS = list(range(1, 11)) + list(range(40, 51))


def parse_tasks(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load(arm, d, name):
    m = C.build_model(arm, "M", SEED, "cpu")
    m.load_state_dict(torch.load(d / "ckpts" / f"{name}.pt", map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def ffn_io(m, probe):
    """層ごとに FFN の入力 y（γ⊙x̂、[N, D]）と前活性 z（[N, FF]）、up の重み W（[FF, D]）。"""
    cap = []
    m.body(probe, cap)
    out = []
    for li, blk in enumerate(m.h):
        y = blk.n2(cap[2 * li]).reshape(-1, C.D).double()
        z = cap[2 * li + 1].reshape(-1, C.FF).double()
        out.append(dict(y=y, z=z, W=blk.mlp.up.weight.double().clone(), mlp=blk.mlp))
    return out


def window(arm, d, a_name, b_name, probe, self_split):
    A = ffn_io(load(arm, d, a_name), probe)
    B = ffn_io(load(arm, d, b_name), probe)
    rows = []
    for li in range(C.L):
        a, b = A[li], B[li]
        ya, yb = a["y"].mean(0), b["y"].mean(0)
        dW, dy = b["W"] - a["W"], yb - ya
        mu_a, mu_b = a["W"] @ ya, b["W"] @ yb
        sd = a["z"].std(0)
        own, up, cross = dW @ ya, a["W"] @ dy, dW @ dy
        dmu = mu_b - mu_a
        resid = float((dmu - (own + up + cross)).abs().max())
        r = dict(layer=li, dmu=dmu / sd, own=own / sd, up=up / sd, cross=cross / sd)
        # 負の成分の量（unit 平均）。符号付きの和でなく、沈める向きの絶対量。相殺 = 項ごとの負の量の和 − 合計の負の量
        r["neg_dmu"] = (-dmu / sd).clamp(min=0)
        r["neg_up"] = (-(up + cross) / sd).clamp(min=0)
        r["neg_own"] = (-own / sd).clamp(min=0)
        if self_split:
            Kn = a["y"] @ ya                                            # [N]
            z = a["z"]
            pp = a["mlp"].phi(z.float()).double() * a["mlp"].dphi(z.float()).double()   # φ φ′, [N, FF]
            D_ = (pp * Kn[:, None]).T @ a["y"]                          # [FF, D] 自己形の向き
            dh = D_ / (D_.norm(dim=1, keepdim=True) + 1e-30)
            proj = (dW * dh).sum(1) * (dh @ ya)
            r["self"] = proj / sd
            r["other"] = (own - proj) / sd
            r["cos_dw_d"] = (dW * dh).sum(1) / (dW.norm(dim=1) + 1e-30)
        rows.append((r, resid))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C3", "SiLU", "GELU"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--root", default=str(Q), help="腕のディレクトリの親（2 ブロックは quicklook/M2）")
    ap.add_argument("--tasks", default="1-10,40-50", help="例: 1-10,40-50,90-100")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    global SEED; SEED = a.seed
    root = Path(a.root); tasks = parse_tasks(a.tasks)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for arm in a.arms:
        d = root / arm / f"seed{SEED}"
        for t in tasks:
            probe = C.task_cpu_M(SEED, t, 10, ALPHA)["probe"]
            start = "init" if t == 1 else f"task{t - 1}"
            for wname, (an, bn, ss) in dict(i=(start, f"u100_task{t}", True), ii=(f"u100_task{t}", f"task{t}", False)).items():
                for r, resid in window(arm, d, an, bn, probe, ss):
                    rec = dict(arm=arm, task=t, window=wname, layer=r["layer"], resid=resid,
                               dmu_mean=float(r["dmu"].mean()), dmu_med=float(r["dmu"].median()),
                               own_mean=float(r["own"].mean()), up_mean=float(r["up"].mean()), cross_mean=float(r["cross"].mean()),
                               neg_dmu=float(r["neg_dmu"].mean()), neg_up=float(r["neg_up"].mean()), neg_own=float(r["neg_own"].mean()))
                    if ss:
                        rec.update(self_mean=float(r["self"].mean()), other_mean=float(r["other"].mean()),
                                   cos_dw_d_med=float(r["cos_dw_d"].median()))
                    recs.append(rec)
            print(arm, "task", t, "done", flush=True)
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(out / "decomp_dmu.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    # 集計: 腕 × 窓 × 塊（課題 2〜10 / 40〜50。窓 (i) の課題 1 は切替でなく学び始めなので別）× 層
    blocks = {"t1": [1], "t2-10": list(range(2, 11)), "t40-50": list(range(40, 51)), "t91-100": list(range(91, 101))}
    summ = []
    for arm in a.arms:
        for wname in ("i", "ii"):
            for bname, bt in blocks.items():
                for li in range(C.L):
                    sel = [r for r in recs if r["arm"] == arm and r["window"] == wname and r["task"] in bt and r["layer"] == li]
                    if not sel:
                        continue
                    m = lambda k: float(np.mean([r[k] for r in sel])) if k in sel[0] else float("nan")
                    summ.append(dict(arm=arm, window=wname, block=bname, layer=li, n_tasks=len(sel), dmu_mean=m("dmu_mean"), dmu_med=m("dmu_med"),
                                     self=m("self_mean"), upstream=m("up_mean") + m("cross_mean"), other=m("other_mean"),
                                     own=m("own_mean"), up=m("up_mean"), cross=m("cross_mean"), cos_dw_d_med=m("cos_dw_d_med"),
                                     resid_max=max(r["resid"] for r in sel),
                                     neg_dmu=m("neg_dmu"), neg_up=m("neg_up"), neg_own=m("neg_own")))
    keys = list(summ[0])
    with open(out / "decomp_dmu_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summ:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in summ:
        s = (f"{r['arm']:5s} 窓{r['window']:2s} {r['block']:6s} L{r['layer']}: Δμ/σ 平均 {r['dmu_mean']:+.3f}（中央 {r['dmu_med']:+.3f}）"
             f" = 自分 {r['own']:+.3f} ＋ 上流 {r['upstream']:+.3f}（w·Δȳ {r['up']:+.3f}・交差 {r['cross']:+.3f}）")
        if r["window"] == "i":
            s += f" | 自分の内訳: 自己形 {r['self']:+.3f}・その他 {r['other']:+.3f}（cos(Δw,d) 中央 {r['cos_dw_d_med']:+.2f}）"
        print(s)
    gh, dirty = C.git_info()
    (out / "provenance.txt").write_text(f"script=decomp_dmu.py git_hash={gh} dirty_src={dirty} arms={a.arms} seed={SEED} grade=事後・未登録・下見\n")


if __name__ == "__main__":
    main()
