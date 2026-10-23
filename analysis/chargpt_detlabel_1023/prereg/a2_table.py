"""A2（C3-2）と ReLUcap-2 の参照表（spec §2.1。§7.1 の J1 の規則・追補 6 の条件 1 を既存の記録に当てる）。学習なし。
入力: <ref>/<腕>/seed<k>/{per_task.csv, online.npz, fresh_online_t100.npz, fresh.csv, unit_stats.npz}（extract_a2.sh で
ブランチ claude/chargpt_cl_1019 の results/chargpt_cl_1019/M2/ から取り出す）。
使い方: python -I a2_table.py <ref> <out_json>"""
import csv, json, math, sys
import numpy as np
REF = sys.argv[1]
DELTA = 3 * 0.0084
def phases(loss, H, n=10000):
    ex = np.asarray(loss, dtype=np.float64) - H
    ma = np.convolve(ex, np.ones(50) / 50, mode="valid")
    hit = np.nonzero(ma <= 0.15)[0]
    return (float(hit[0] + 50) if len(hit) else float(n)), float(ex[950:1050].mean()), float(ex[-1000:].mean())
def rd(p): return list(csv.DictReader(open(p)))
def sign_p(d):
    d = [x for x in d if x != 0]; n = len(d); k = sum(x > 0 for x in d)
    if n == 0: return float("nan")
    cdf = lambda j: sum(math.comb(n, i) for i in range(j + 1)) / 2 ** n
    return min(1.0, 2 * min(cdf(k), 1 - cdf(k - 1)))
def mr(v): v = np.asarray(v, float); return f"{np.median(v):.4g} [{v.min():.4g}, {v.max():.4g}]"
res = {}
for arm in ("C3", "ReLUcap"):
    S = []
    for s in range(10):
        d = f"{REF}/{arm}/seed{s}"
        pt = rd(f"{d}/per_task.csv"); H = np.array([float(r["H"]) for r in pt])
        on = np.load(f"{d}/online.npz")["loss"].astype(np.float64)
        P = np.array([phases(on[t], H[t]) for t in range(100)])
        F100 = phases(np.load(f"{d}/fresh_online_t100.npz")["loss"], H[99])
        early, late, cont100 = P[1:10].mean(0), P[90:100].mean(0), P[99]
        resolved = [not (i == 0 and ((early[0] == 50 and late[0] == 50) or (early[0] == 10000 and late[0] == 10000))) for i in range(3)]
        phase_loss = [bool(late[i] > early[i] and cont100[i] > F100[i]) for i in range(3)]
        end1 = float(pt[0]["end_metric"])
        j1 = bool(end1 <= 0.1 and sum(resolved) >= 2 and all(pl for pl, rs in zip(phase_loss, resolved) if rs))
        U = np.load(f"{d}/unit_stats.npz"); rn = U["rownorm"].astype(np.float64)
        dead_u = ((1 - U["uniform_q"].astype(np.float64)) < 0.005).mean(2)
        L1d = dead_u[:, 1]; Rk = next((t for t in range(11, 101) if L1d[t - 1] <= 0.5), None)
        onl = np.array([float(r["online_all"]) for r in pt]); fr = {int(r["task"]): float(r["online_all"]) for r in rd(f"{d}/fresh.csv")}
        head = (on[:, :100] - H[:, None]).mean(1)
        c1 = bool(end1 <= 0.1 and onl[90:100].mean() - onl[1:10].mean() >= DELTA and (onl[99] - fr[100]) - (onl[9] - fr[10]) >= DELTA and head[90:100].mean() > head[1:10].mean())
        L0on = next((t for t in range(1, 101) if dead_u[t - 1, 0] >= 0.5), None)
        S.append(dict(seed=s, end1=end1, early=early.tolist(), late=late.tolist(), cont100=cont100.tolist(), fresh100=list(F100),
                      phase_loss=phase_loss, j1=j1, c1=c1, dE=float(late[1] - early[1]), dE110=float(late[1] - P[0:10, 1].mean()),
                      g0=float(np.median(rn[99, 0]) / np.median(rn[0, 0])), g1=float(np.median(rn[99, 1]) / np.median(rn[0, 1])),
                      rn0_t1=float(np.median(rn[0, 0])), rn0_t10=float(np.median(rn[9, 0])), rn0_t50=float(np.median(rn[49, 0])), rn0_t100=float(np.median(rn[99, 0])),
                      L0d100=float(dead_u[99, 0]), L0d_91_100=float(dead_u[90:100, 0].mean()), L0d50=float(dead_u[49, 0]), L0_onset50=L0on,
                      L1d10=float(L1d[9]), L1d50=float(L1d[49]), L1d100=float(L1d[99]), Rk=Rk, H_med=float(np.median(H))))
    res[arm] = S
    print(f"===== {arm}")
    for k, lab in [("end1", "課題1末 eval 超過"), ("H_med", "H 中央値")]:
        print(lab, mr([x[k] for x in S]))
    for i, ph in enumerate(("T015", "E1000", "ELast")):
        print(ph, "早期2-10", mr([x["early"][i] for x in S]), "| 後期91-100", mr([x["late"][i] for x in S]), "| 課題100", mr([x["cont100"][i] for x in S]), "| 新品100", mr([x["fresh100"][i] for x in S]),
              "| 相ごとの喪失 seed", sum(x["phase_loss"][i] for x in S))
    print("J1 規則の LOSS seed", sum(x["j1"] for x in S), "| 追補6 条件1", sum(x["c1"] for x in S))
    for k in ("dE", "dE110", "g0", "g1", "rn0_t1", "rn0_t10", "rn0_t50", "rn0_t100", "L0d100", "L0d_91_100", "L0d50", "L1d10", "L1d50", "L1d100"):
        print(k, mr([x[k] for x in S]))
    print("L0 死 >= 0.5 の seed（課題100）", sum(x["L0d100"] >= 0.5 for x in S), "| L0 死が初めて 0.5 以上の課題", [x["L0_onset50"] for x in S])
    print("L1 課題10 死 >= 0.9", sum(x["L1d10"] >= 0.9 for x in S), "| R_k", [x["Rk"] for x in S], "| 生き返る seed", sum(x["Rk"] is not None for x in S))
json.dump(res, open(sys.argv[2], "w"), indent=1)
