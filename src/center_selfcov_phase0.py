"""center_selfcov_0814 Phase 0 (spec §3): 既存 aniso_perp_0812 の npz から
項目3の指標を予備的に測る (再学習なし)。

  python -m src.center_selfcov_phase0

各 followup_Eg_*.npz の W / dead から §2.2 と同一定義で cos_e1W_e1Sig 等を算出し、
freeze_neurons_*.csv の cos_spike (= |cos(E[g], u)|) と同じ表に並べる。
仕様の期待値 (w100 κ=16: |cos(e1W,u)| 0.179→0.058→0.100 / |cos(Eg,u)| 0.698→0.647→0.710)
が再現しなければ止めて報告する。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .common import ROOT, load_config
from .w_direction import spike_dir_vec, w_dir_metrics_np

SRC = os.path.join(ROOT, "results", "aniso_perp_0812")
OUT = os.path.join(ROOT, "results", "center_selfcov_0814", "phase0")

# 仕様 §3 の期待値 (w100, κ=16)。相対許容 5%。
EXPECT = {("cos_e1W", 0): 0.179, ("cos_e1W", 100000): 0.058, ("cos_e1W", 1000000): 0.100,
          ("cos_Eg", 0): 0.698, ("cos_Eg", 100000): 0.647, ("cos_Eg", 1000000): 0.710}


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config(os.path.join(SRC, "config_used.yaml"))
    B = cfg["condB"]
    d = B["d"]
    u = spike_dir_vec(B.get("spike_dir", "ones"), d)
    runs = pd.read_csv(os.path.join(SRC, "runs.csv")).set_index("run_id")

    rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "followup_Eg_*.npz"))):
        z = np.load(path, allow_pickle=True)
        step = int(z["step"])
        rids = [str(x) for x in z["run_ids"]]
        W, dead, Eg = z["W"], z["dead"], z["Eg_W"]          # [R,h,d], [R,h], [R,h,d]
        mu_true = z["mu_true"]                              # [R,d]
        for i, rid in enumerate(rids):
            r = runs.loc[rid]
            m = w_dir_metrics_np(W[i], dead[i], u, mu_true[i],
                                 kappa=int(r.kappa))
            m.pop("e1_vec", None)
            # |cos(E[g], u)|: alive ニューロンの勾配ベクトル別 |cos| の平均
            # (freeze_neurons.cos_spike と同じ per-neuron 定義)
            alive = ~dead[i]
            g = Eg[i][alive]
            cg = np.abs(g @ u) / np.maximum(np.linalg.norm(g, axis=1), 1e-30)
            rows.append(dict(run_id=rid, step=step, width=int(r.width),
                             kappa=int(r.kappa), lr=float(r.lr), c=float(r.c),
                             seed=int(r.seed),
                             cos_Eg_u_mean=float(np.mean(cg)) if alive.any() else np.nan,
                             **m))
    df = pd.DataFrame(rows).sort_values(["width", "kappa", "lr", "seed", "step"])
    df.to_csv(os.path.join(OUT, "phase0_metrics.csv"), index=False)

    # 仕様 §3 の再現確認 (w100, κ=16, seed 平均。lr は両方をプール)
    chk = df[(df.width == 100) & (df.kappa == 16)]
    got = chk.groupby("step")[["cos_e1W_e1Sig", "cos_Eg_u_mean"]].mean()
    ver = []
    for (which, step), exp in EXPECT.items():
        col = "cos_e1W_e1Sig" if which == "cos_e1W" else "cos_Eg_u_mean"
        obs = float(got.loc[step, col]) if step in got.index else np.nan
        rel = abs(obs - exp) / exp if np.isfinite(obs) else np.nan
        ver.append(dict(quantity=which, step=step, expected=exp, observed=obs,
                        rel_err=rel, ok=bool(np.isfinite(rel) and rel < 0.05)))
    ver = pd.DataFrame(ver).sort_values(["quantity", "step"])
    ver.to_csv(os.path.join(OUT, "phase0_replication.csv"), index=False)

    floor = np.sqrt(2 / (np.pi * d))
    lines = ["# center_selfcov_0814 Phase 0 (spec §3): aniso_perp_0812 の再解析\n",
             f"入力: {SRC} の followup_Eg_*.npz (再学習なし)。"
             f"u = spike_dir '{B.get('spike_dir')}'、d={d}、ランダム床 |cos| ≈ {floor:.3f}\n",
             "## 仕様 §3 の期待値との照合 (w100, κ=16, lr/seed プール平均)\n",
             ver.to_string(index=False), ""]
    ok = bool(ver.ok.all())
    lines.append(f"\n- **再現判定: {'PASS' if ok else 'FAIL'}** "
                 + ("(全項目が相対 5% 以内)" if ok
                    else "(乖離あり → 仕様 §3 の指示により停止して報告)"))

    lines.append("\n## W 方向指標の時系列 (幅×κ 別、lr/seed プール平均)\n")
    piv = df.groupby(["width", "kappa", "step"])[
        ["cos_e1W_e1Sig", "cos_e1W_e1Sig_pca", "cos_e1W_mu", "cos_Eg_u_mean",
         "srank_alive", "top1_frac_alive", "w_norm_mean"]].mean().round(4)
    lines.append(piv.to_string())

    lines.append("\n## 所見\n")
    g = df[(df.width == 100) & (df.kappa == 16)].groupby("step")
    lines.append(f"- 勾配場は Σ 軸を向く (|cos(E[g],u)| ≈ "
                 f"{g.cos_Eg_u_mean.mean().mean():.2f}) が、重みは床付近に留まる "
                 f"(|cos(e1W,u)| ≈ {g.cos_e1W_e1Sig.mean().mean():.2f} vs 床 {floor:.3f})"
                 " → 「drift は Σ 軸を向くが重みはそこに乗らない」乖離 (P3-7 の予備証拠)")
    lines.append(f"- ただしこの設定は LoP 非発現 (dead_frac=0, srank ほぼ不変) なので"
                 "判別になっていない。Phase 1 で教師幅を分離してレジームを選び直す")
    with open(os.path.join(OUT, "phase0_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(dict(replication_pass=ok, floor=floor),
              open(os.path.join(OUT, "phase0_meta.json"), "w"), indent=1)
    print(ver.to_string(index=False))
    print(f"\n再現判定: {'PASS' if ok else 'FAIL'}")
    print(f"-> {OUT}")
    return ok


if __name__ == "__main__":
    main()
