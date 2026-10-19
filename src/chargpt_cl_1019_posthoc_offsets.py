#!/usr/bin/env python3
"""chargpt_cl_1019 の事後解析 2（未登録）: 中心化の腕で学ぶ力が落ちる理由を、保存済みの記録と checkpoint で調べる。

  python src/chargpt_cl_1019_posthoc_offsets.py --out results/chargpt_cl_1019/posthoc_offsets_seed0

依頼は Neokunekune のセッション（1020 早朝）。読み：「この箱では unit ごとのオフセット（閾値）が課題を学ぶのに要り、中心化はそれを禁じる」。
  1. 開く割合 p = 1 − q（q は記録の「z < 0 のトークンの割合」）の分布。p < 0.05（ほぼ開かない）、0.05〜0.3（まばらな検出器）、0.3〜0.7、> 0.7。
  2. 課題ごとのオンライン超過損失（50 点）と、課題の中の区間（最初の 100 更新・100〜1,000・最後の 1,000）。細かい時点の評価損失（更新 0・100・1,000・末）。
  3. FFN 隠れ層の実効階数（checkpoint の課題）と、前活性の幅 σ（全課題の末）。
  4. 担い手の幾何の壁：c = mean(K)/sd(K)（K_n = ȳ·y_n）と、マハラノビスの上限 √(ȳᵀΣ⁻¹ȳ)。y は up の直前の入力（中心化の腕では y − m）。
     bias の無い unit では |μᵢ/σᵢ| ≤ √(ȳᵀΣ⁻¹ȳ) が恒等的に成り立つ（Cauchy–Schwarz）。c は ȳ の向きの unit の μ/σ。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019_center as CC  # noqa: E402  （中心化の腕も組めるように）
C = CC.C

R = Path("results/chargpt_cl_1019")
ARMS = {"C3": R / "crosscheck_white-san/M/C3/seed0", "B3": R / "crosscheck_white-san/M/B3/seed0",
        "C3H": R / "M/C3H/seed0", "B3H": R / "M/B3H/seed0", "B3HB": R / "M/B3HB/seed0"}
CK_TASKS = (1, 10, 25, 50)
SEED, ALPHA, N_M = 0, 0.01, 10000


def wcsv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def ffn_inputs(model, probe):
    """層ごとに、up が実際に受け取る入力（中心化の腕では y − m）と z を返す。"""
    cap = []
    out = []
    with torch.no_grad():
        model.body(probe, cap)
        for li, blk in enumerate(model.h):
            xin, z = cap[2 * li], cap[2 * li + 1]
            y = blk.n2(xin)
            if getattr(blk.mlp, "center", False):
                y = y - blk.mlp.mean_vec()
            out.append((y.reshape(-1, C.D).double(), z.reshape(-1, C.FF).double()))
    return out


def erank(phi):
    x = phi - phi.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qrows, lrows, wrows, erows, srows, grows = [], [], [], [], [], []
    probes = {t: C.task_cpu_M(SEED, t, 10, ALPHA)["probe"] for t in CK_TASKS}
    for arm, d in ARMS.items():
        U = np.load(d / "unit_stats.npz")
        on = np.load(d / "online.npz")["loss"].astype(np.float64)
        pt = list(csv.DictReader(open(d / "per_task.csv")))
        H = np.array([float(r["H"]) for r in pt])
        # 1. 開く割合の分布（全課題の末）
        for t in range(1, len(pt) + 1):
            for li in range(C.L):
                p = 1.0 - U["chain_q"][t - 1, li].astype(np.float64)
                qrows.append(dict(arm=arm, task=t, layer=li, p_lt005=float((p < 0.05).mean()),
                                  p_005_03=float(((p >= 0.05) & (p < 0.3)).mean()),
                                  p_03_07=float(((p >= 0.3) & (p <= 0.7)).mean()), p_gt07=float((p > 0.7).mean()),
                                  dead=float(U["chain_dead"][t - 1, li].mean())))
        # 2. 課題ごとの学習（オンライン超過損失の区間）
        for t in range(1, len(pt) + 1):
            ex = on[t - 1] - H[t - 1]
            lrows.append(dict(arm=arm, task=t, online_all=float(ex.mean()), first100=float(ex[:100].mean()),
                              u100_1000=float(ex[100:1000].mean()), last1000=float(ex[-1000:].mean()),
                              end_eval=float(pt[t - 1]["end_metric"])))
        for r in csv.DictReader(open(d / "eval_trace.csv")):
            if int(r["u"]) in (0, 100, 1000, 10000):
                wrows.append(dict(arm=arm, task=int(r["task"]), u=int(r["u"]), eval_excess=float(r["eval"])))
        # 3. 前活性の幅（全課題の末）
        for t in range(1, len(pt) + 1):
            srows.append(dict(arm=arm, task=t, **{f"sigma_med_l{li}": float(np.median(U["chain_sd"][t - 1, li])) for li in range(C.L)}))
        # 3・4. checkpoint の課題で：実効階数と担い手の幾何
        for t in CK_TASKS:
            m = C.build_model(arm, "M", SEED, "cpu")
            m.load_state_dict(torch.load(d / "ckpts" / f"task{t}.pt", map_location="cpu"))
            m.eval()
            ins = ffn_inputs(m, probes[t])
            for li, (y, z) in enumerate(ins):
                phi = z.clamp(min=0)
                erows.append(dict(arm=arm, task=t, layer=li, erank=erank(phi)))
                ybar = y.mean(0)
                K = y @ ybar
                c = float(K.mean() / K.std())
                Sig = torch.cov(y.T) + 1e-10 * torch.eye(C.D, dtype=torch.float64)
                mahal = float(torch.sqrt(ybar @ torch.linalg.solve(Sig, ybar)))
                mu, sd = z.mean(0), z.std(0)
                r = mu / (sd + 1e-12)
                grows.append(dict(arm=arm, task=t, layer=li, c=c, mahal=mahal, r_med=float(r.median()),
                                  r_q10=float(torch.quantile(r, 0.1)), frac_below_minus_c=float((r < -c).double().mean()),
                                  frac_beyond_mahal=float((r.abs() > mahal * (1 + 1e-6)).double().mean())))
            print(arm, "task", t, "done", flush=True)
        # 初期値での c（中心化の前の値でもある：m = 0）
        m = C.build_model(arm, "M", SEED, "cpu")
        m.load_state_dict(torch.load(d / "ckpts" / "init.pt", map_location="cpu"))
        for li, (y, z) in enumerate(ffn_inputs(m, probes[1])):
            ybar = y.mean(0)
            K = y @ ybar
            Sig = torch.cov(y.T) + 1e-10 * torch.eye(C.D, dtype=torch.float64)
            grows.append(dict(arm=arm, task=0, layer=li, c=float(K.mean() / K.std()),
                              mahal=float(torch.sqrt(ybar @ torch.linalg.solve(Sig, ybar)))))
    wcsv(out / "open_fraction.csv", qrows)
    wcsv(out / "learning_by_task.csv", lrows)
    wcsv(out / "within_eval.csv", wrows)
    wcsv(out / "sigma.csv", srows)
    wcsv(out / "erank.csv", erows)
    wcsv(out / "walls.csv", grows)
    gh, dirty = C.git_info()
    (out / "provenance.json").write_text(json.dumps(dict(script="chargpt_cl_1019_posthoc_offsets.py", git_hash=gh, dirty_src=dirty,
                                                         arms={k: str(v) for k, v in ARMS.items()}, grade="事後・未登録・seed 0"), indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
