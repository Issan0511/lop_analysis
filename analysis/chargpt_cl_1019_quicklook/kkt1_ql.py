import csv, sys, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src")
import chargpt_cl_1019_verdict as V
R = Path("results/chargpt_cl_1019")
RUNS = {"KKT1_s1000": R/"quicklook/M/KKT1/seed1000", "KKT1b3_s1000": R/"quicklook/M/KKT1b3/seed1000", "C3_s0": R/"crosscheck_white-san/M/C3/seed0",
        "B3_s0": R/"crosscheck_white-san/M/B3/seed0", "C4_s0": R/"crosscheck_white-san/M/C4/seed0",
        "N4_s0": R/"crosscheck_white-san/M/N4/seed0"}
for a in ("C3H","B3H","B3HB"):
    for s in (0,1,2):
        d = R/f"M/{a}/seed{s}"
        if (d/"fresh.csv").exists(): RUNS[f"{a}_s{s}"] = d
print("== 要約（verdict の seed_stats）")
for k, d in RUNS.items():
    s = V.seed_stats(d, "M")
    pt = V.rd(d/"per_task.csv"); fr = {int(r["task"]): float(r["online_all"]) for r in V.rd(d/"fresh.csv")}
    on = {int(r["task"]): float(r["online_all"]) for r in pt}
    gaps = {t: on[t]-fr[t] for t in (10,25,50) if t in fr}
    print(f"{k:12s} early {s['on_early']:.4f} late {s['on_late']:.4f} Δ {s['on_worse']:+.4f} | gap10/25/50 " +
          " ".join(f"{gaps.get(t,float('nan')):+.4f}" for t in (10,25,50)) +
          f" | fresh50 {fr.get(50,float('nan')):.4f} | neg10 {s['neg10']:.3f} rmed e/l {s['rmed_early']:+.3f}/{s['rmed_late']:+.3f} push {s['push_mean']:+.3f}")
def end_rows(d):
    pt = V.rd(d/"per_task.csv"); n = int(pt[0]["n"]); out = {}
    for r in V.rd(d/"layer_stats.csv"):
        if r["probe"] == "chain" and int(r["u"]) == n:
            out[(int(r["task"]), int(r["layer"]))] = r
    return out
print("\n== 層別（課題末・chain probe）")
for k in ("KKT1b3_s1000", "KKT1_s1000", "C3_s0"):
    d = RUNS[k]; E = end_rows(d); U = np.load(d/"unit_stats.npz")
    print(k)
    for t in (1, 10, 25, 50):
        line = []
        for li in range(4):
            r = E[(t, li)]
            q = U["chain_q"][t-1, li].astype(float)        # unit ごとの z<0 の割合
            sdm = float(np.median(U["chain_sd"][t-1, li]))
            if "phi1_mean" in r:
                ph = float(r["phi1_mean"]); lt01 = float(r["units_phi1_lt01"])
            else:
                ph = float(1 - q.mean()); lt01 = float((1 - q < 0.1).mean())
            line.append(f"L{li}: neg {float(r['neg_frac']):.2f} r {float(r['r_med']):+.2f} dead {float(r['dead']):.2f} φ' {ph:.2f} φ'<.1 {lt01:.2f} σ {sdm:.2f}")
        print(f"  t{t:2d} " + " | ".join(line))
    if "two_alpha_W_med" in E[(1,0)]:
        for t in (1, 10, 25, 50):
            print(f"  t{t:2d} α_med " + " ".join(f"{float(E[(t,li)]['alpha_med']):.2f}" for li in range(4)) +
                  " | 2αW " + " ".join(f"{float(E[(t,li)]['two_alpha_W_med']):.2f}" for li in range(4)) +
                  " | 2ασ " + " ".join(f"{float(E[(t,li)]['two_alpha_sd_med']):.2f}" for li in range(4)) +
                  " | in_curve " + " ".join(f"{float(E[(t,li)]['frac_in_curve']):.3f}" for li in range(4)) +
                  " | resp<.05/.05-.3/.3-.7/>.7 L0 " + "/".join(f"{float(E[(t,0)][c]):.2f}" for c in ("resp_lt005","resp_005_03","resp_03_07","resp_gt07")))
print("\n== up の重みのノルム（checkpoint）")
for k in ("KKT1b3_s1000", "KKT1_s1000", "C3_s0"):
    d = RUNS[k]
    for t in ("init", "task1", "task10", "task25", "task50"):
        sd = torch.load(d/"ckpts"/f"{t}.pt", map_location="cpu")
        ns = [float(sd[f"h.{li}.mlp.up.weight"].norm()) for li in range(4)]
        tot = math.sqrt(sum(x*x for x in ns))
        dn = [float(sd[f"h.{li}.mlp.down.weight"].norm()) for li in range(4)]
        print(f"  {k:11s} {t:6s} up " + " ".join(f"{x:7.1f}" for x in ns) + f" | 全体 {tot:7.1f} | down " + " ".join(f"{x:6.1f}" for x in dn))
print("\n== 課題ごとのオンライン超過損失（5 課題ごとの平均）")
for k in ("KKT1b3_s1000", "KKT1_s1000", "C3_s0", "B3_s0"):
    on = [float(r["online_all"]) for r in V.rd(RUNS[k]/"per_task.csv")]
    print(f"  {k:11s} " + " ".join(f"{np.mean(on[i:i+5]):.3f}" for i in range(0, 50, 5)))
