"""bias_margin_0814 Phase 0 (spec §3): rank_int_0814 のスナップショットから b 系指標を測る
(再学習なし)。

  python -m src.bias_margin_phase0

条件A (SCR) の入力は x = [flip_state (f 個, タスク内で定数), rnd (m−f 個, 毎ステップ U{0,1})]。
プリ活性は  w_flip·flip + w_rnd·rnd + b  なので、タスク内の最大値は
  pre_max = w_flip·flip + b + Σ_j max(w_rnd_j, 0)     (rnd の全 2^(m-f) 列挙と等価)
これが ≤ 0 のとき「タスク内 dead (厳密判定)」。dead の負性を w_flip·flip と b に分解し、
どちらが負性を担っているか (dead_b_dominant_frac) を出す。

仕様 §3 の再現目標と照合し、乖離したら止めて報告する。
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from .common import ROOT

SNAP = os.path.join(ROOT, "results", "rank_int_0814", "snapshots")
OUT = os.path.join(ROOT, "results", "bias_margin_0814", "phase0")

# 仕様 §3 の再現目標 (相対 5% 以内、割合は絶対 0.02 以内)
EXPECT = {
    10: dict(b_mean=-0.079, b_min=-1.106, b_neg_frac=0.500),
    20: dict(b_mean=-0.116, b_min=-0.618, b_neg_frac=0.690,
             dead_frac_strict=0.380, dead_wflip_mean=-1.463, dead_b_mean=-0.151,
             dead_b_dominant_frac=0.053),
}


def analyze(width):
    p = os.path.join(SNAP, f"A_w{width}_bfull_none_step150000.pt")
    s = torch.load(p, map_location="cpu", weights_only=False)
    W = s["net"]["W"].double()                      # [R,h,d]
    b = s["net"]["b"].double()                      # [R,h]
    flip = s["env"]["flip_state"].double()          # [R,f]
    R, h, d = W.shape
    f = flip.shape[1]

    w_flip, w_rnd = W[:, :, :f], W[:, :, f:]
    contrib_flip = torch.einsum("rhf,rf->rh", w_flip, flip)      # [R,h]
    pre_max = contrib_flip + b + w_rnd.clamp_min(0).sum(dim=2)   # rnd 全列挙の最大
    dead = pre_max <= 0

    rows = []
    for i, r in enumerate(s["runs"]):
        db = b[i][dead[i]]
        dfl = contrib_flip[i][dead[i]]
        # 負性を b が担う = |b| が |w_flip·flip| を上回る dead ユニット
        dom = (db.abs() > dfl.abs()) & (db < 0) if dead[i].any() else torch.zeros(0, dtype=torch.bool)
        rows.append(dict(
            width=width, seed=r["seed"], run_id=r["run_id"], h=h,
            b_mean=float(b[i].mean()), b_min=float(b[i].min()),
            b_std=float(b[i].std(unbiased=False)),
            b_neg_frac=float((b[i] < 0).double().mean()),
            dead_frac_strict=float(dead[i].double().mean()),
            dead_wflip_mean=float(dfl.mean()) if dead[i].any() else np.nan,
            dead_b_mean=float(db.mean()) if dead[i].any() else np.nan,
            dead_b_dominant_frac=float(dom.double().mean()) if dead[i].any() else np.nan,
            alive_b_mean=float(b[i][~dead[i]].mean()) if (~dead[i]).any() else np.nan,
            w_norm_mean=float(W[i].norm(dim=1).mean()),
            beta_mean=float((b[i] / W[i].norm(dim=1).clamp_min(1e-12)).mean()),
        ))
    return pd.DataFrame(rows), dead, b, contrib_flip


def pooled(df, width):
    """ユニット単位でプールした集計 (seed をまたぐ。b<0 割合などは全ユニット基準)。"""
    g = df[df.width == width]
    n = g.h.iloc[0]
    tot = n * len(g)
    return dict(
        b_mean=float((g.b_mean * n).sum() / tot),
        b_min=float(g.b_min.min()),
        b_neg_frac=float((g.b_neg_frac * n).sum() / tot),
        dead_frac_strict=float((g.dead_frac_strict * n).sum() / tot),
        dead_wflip_mean=float(np.average(g.dead_wflip_mean.dropna(),
                                         weights=(g.dead_frac_strict * n)[g.dead_wflip_mean.notna()])
                              if g.dead_wflip_mean.notna().any() else np.nan),
        dead_b_mean=float(np.average(g.dead_b_mean.dropna(),
                                     weights=(g.dead_frac_strict * n)[g.dead_b_mean.notna()])
                          if g.dead_b_mean.notna().any() else np.nan),
        dead_b_dominant_frac=float(np.average(
            g.dead_b_dominant_frac.dropna(),
            weights=(g.dead_frac_strict * n)[g.dead_b_dominant_frac.notna()])
            if g.dead_b_dominant_frac.notna().any() else np.nan),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    dfs = []
    for width in [10, 20]:
        df, *_ = analyze(width)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(OUT, "phase0_metrics.csv"), index=False)

    ver = []
    for width, exp in EXPECT.items():
        obs = pooled(df, width)
        for k, want in exp.items():
            got = obs[k]
            if k.endswith("_frac"):
                ok = abs(got - want) <= 0.02
                err = abs(got - want)
            else:
                err = abs(got - want) / abs(want)
                ok = err < 0.05
            ver.append(dict(width=width, quantity=k, expected=want, observed=got,
                            err=err, ok=bool(ok)))
    ver = pd.DataFrame(ver)
    ver.to_csv(os.path.join(OUT, "phase0_replication.csv"), index=False)
    allok = bool(ver.ok.all())

    lines = ["# bias_margin_0814 Phase 0 (spec §3): rank_int_0814 スナップショットの b 分解\n",
             "入力: results/rank_int_0814/snapshots/A_w{10,20}_bfull_none_step150000.pt "
             "(条件A, full-batch, µ≠0)。再学習なし。\n",
             "タスク内 dead は厳密判定: pre_max = w_flip·flip + b + Σ_j max(w_rnd_j,0) ≤ 0 "
             "(rnd の全 2^(m−f) 列挙と等価)。\n",
             "## 仕様 §3 の再現目標との照合\n", ver.to_string(index=False), ""]
    lines.append(f"\n- **再現判定: {'PASS' if allok else 'FAIL'}**"
                 + ("" if allok else " — 仕様 §3 の指示により停止して報告"))

    lines.append("\n## seed 別の実測\n")
    lines.append(df.round(4).to_string(index=False))

    lines.append("\n## 所見 (仕様 §3 が明記を求める非対称性)\n")
    p20 = pooled(df, 20)
    lines.append(f"- b は予測どおり負にドリフトしている (w10 平均 {pooled(df,10)['b_mean']:+.3f} / "
                 f"w20 平均 {p20['b_mean']:+.3f}、b<0 の割合 "
                 f"{pooled(df,10)['b_neg_frac']:.3f} / {p20['b_neg_frac']:.3f})")
    lines.append(f"- **しかし µ≠0 の条件A では µ 経路が圧倒的に速く b は脇役**: dead ユニットの"
                 f"負性内訳は w_flip·flip {p20['dead_wflip_mean']:+.3f} に対し b は "
                 f"{p20['dead_b_mean']:+.3f} で約 "
                 f"{abs(p20['dead_b_mean']/p20['dead_wflip_mean'])*100:.0f}%、"
                 f"b が負性を担う dead は {p20['dead_b_dominant_frac']*100:.1f}% のみ")
    lines.append("- これは仮説の反証ではない。b 経路の効果が見えるのは µ を抜いた後だという "
                 "§1 の主張と整合する (µ≠0 では µ 経路が b 経路を覆い隠す)")
    with open(os.path.join(OUT, "phase0_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(dict(replication_pass=allok), open(os.path.join(OUT, "phase0_meta.json"), "w"))
    print(ver.to_string(index=False))
    print(f"\n再現判定: {'PASS' if allok else 'FAIL'}")
    return allok


if __name__ == "__main__":
    main()
