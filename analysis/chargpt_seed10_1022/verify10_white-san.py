import csv, json, sys, numpy as np, statistics as st
R = sys.argv[1]
# ---------- A ----------
rows = list(csv.DictReader(open(f"{R}/load/states.csv")))
A = {}
for s in range(10):
    g = [r for r in rows if r["arm"] == "C3" and int(r["seed"]) == s and r["layer"] == "1"]
    init = [r for r in g if r["stage"] == "init"][0]
    ends = {int(r["task"]): r for r in g if r["stage"] == "end"}
    s_init = float(init["own_sink"]); s_early = st.median(float(ends[t]["own_sink"]) for t in range(1, 10))
    r_mag = float(init["own_abs_median"]) / float(ends[1]["own_abs_median"])
    A[s] = (s_init, s_early, r_mag)
up = sum(v[1] > v[0] for v in A.values()); rm = sum(v[2] >= 0.1 for v in A.values())
ends_all = [r for r in rows if r["stage"] == "end" and r["layer"] == "1"]
kpos = sum(float(r["head_kappa"]) > 0 for r in ends_all); viol = sum(int(float(r["head_lemma_violation"])) for r in ends_all)
print(f"A1: task-end states {len(ends_all)}, kappa>0 {kpos}, lemma violations {viol}")
print(f"A2: s_early > s_init {up}/10 | s_init {min(v[0] for v in A.values()):.3f}-{max(v[0] for v in A.values()):.3f} | s_early {min(v[1] for v in A.values()):.3f}-{max(v[1] for v in A.values()):.3f}")
print(f"A3: r_mag >= 0.1 {rm}/10 | r_mag {min(v[2] for v in A.values()):.2f}-{max(v[2] for v in A.values()):.2f}")
# L0 of C3 (record)
for lay in ("0",):
    v = []
    for s in range(10):
        g = [r for r in rows if r["arm"] == "C3" and int(r["seed"]) == s and r["layer"] == lay]
        init = [r for r in g if r["stage"] == "init"][0]; ends = {int(r["task"]): r for r in g if r["stage"] == "end"}
        v.append((float(init["own_sink"]), st.median(float(ends[t]["own_sink"]) for t in range(1, 10))))
    print(f"A rec C3 L0: s_init {min(x[0] for x in v):.3f}-{max(x[0] for x in v):.3f}  s_early {min(x[1] for x in v):.3f}-{max(x[1] for x in v):.3f}")
# ---------- B ----------
def shares(j, layer, metric):
    L = j["layers"][layer][metric]
    a = np.array(L["own"]); b = np.array(L["self_up"]); c = np.array(L["others_same"]); d = np.array(L["others_path"])
    tot = a + b + c + d; m = tot < 0
    S = tot[m].sum()
    return (c[m].sum() / S, (a[m].sum() + b[m].sum()) / S, d[m].sum() / S)
cnt = {"3a": 0, "self": 0, "3b": 0}; lo = []
for s in range(10):
    sh = [shares(json.load(open(f"{R}/rho/C3/seed{s}/t{k}_a_chain.json")), "1", "D") for k in (2, 3, 5, 10)]
    med = [st.median(x[i] for x in sh) for i in range(3)]
    lo.append(med[0]); key = ["3a", "self", "3b"][int(np.argmax(med))]; cnt[key] += 1
print(f"B: C3 L1 D early, largest term counts {cnt} | share(3a) {min(lo):.2f}-{max(lo):.2f}")
# ---------- C ----------
SE = {0: (50, 64), 1: (43, 59), 2: (50, 93), 4: (50, 65), 5: (25, 46), 6: (50, 77), 7: (41, 53), 8: (50, 74), 9: (50, 66)}
def death(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r["probe"] == "uniform" and r["layer"] == "1":
            d[int(r["task"])] = float(r["frac_p_lt005"])  # last row per task
    return d
elig = 0; held = {"lnex1": 0, "nocommon": 0}; det = []
for s, (S, E) in SE.items():
    c = death(f"{R}/revival/C1/seed{s}/control/layer_stats.csv")
    Rp = next((t for t in range(S + 1, E + 1) if c.get(t, 1) <= 0.5), None)
    if Rp is None:
        det.append(f"s{s}: control no revival in [{S+1},{E}]"); continue
    elig += 1; row = f"s{s}: R'={Rp} win={E-Rp+1}"
    for arm in held:
        d = death(f"{R}/revival/C1/seed{s}/{arm}/layer_stats.csv")
        ok = all(d[t] > 0.5 for t in range(Rp, E + 1)); held[arm] += ok
        row += f" {arm}:{'held' if ok else 'revived'}(min {min(d[t] for t in range(Rp, E+1)):.2f})"
    det.append(row)
print(f"C1: eligible n={elig}, held {held}"); print("  " + " | ".join(det))
c2 = []
for s in range(10):
    c = death(f"{R}/revival/C2/seed{s}/control/layer_stats.csv"); g = death(f"{R}/revival/C2/seed{s}/gamma03f/layer_stats.csv")
    c2.append((c[26], g[26], max(g[t] for t in range(27, 32))))
ok = sum(1 for c, g, _ in c2 if c >= 0.9 and g <= 0.5); redie = sum(1 for c, g, mx in c2 if g <= 0.5 and mx >= 0.9)
print(f"C2: control>=0.9 & gamma03f<=0.5 at 26: {ok}/10 | re-death by 31: {redie}/10 | gamma03f@26 {min(x[1] for x in c2):.3f}-{max(x[1] for x in c2):.3f}")
# ---------- D ----------
esc = []; summ = []
for s in range(10):
    rr = [r for r in csv.DictReader(open(f"{R}/silu4/SiLU/seed{s}/layer_stats.csv")) if r["probe"] == "uniform" and r["u"] == "10000"]
    by = {(int(r["task"]), int(r["layer"])): r for r in rr}
    ok = [t for t in range(1, 47) if all(float(by[(u, 3)]["r_med"]) > 1 and float(by[(u, 3)]["phi1_mean"]) >= 0.75 for u in range(t, t + 5))]
    esc.append(ok[0] if ok else None)
    f = lambda t, l, k: float(by[(t, l)][k])
    summ.append(f"s{s}: L3 r {f(10,3,'r_med'):+.1f}/{f(50,3,'r_med'):+.1f}/{f(100,3,'r_med'):+.1f} dead {f(50,3,'frac_p_lt005'):.2f}/{f(100,3,'frac_p_lt005'):.2f} phi {f(50,3,'phi1_mean'):.2f} | L2 r {f(50,2,'r_med'):+.1f}/{f(100,2,'r_med'):+.1f} dead {f(100,2,'frac_p_lt005'):.2f} | L1 dead {f(100,1,'frac_p_lt005'):.2f} L0 dead {f(100,0,'frac_p_lt005'):.2f}")
print(f"D1: escape onsets {esc} -> {sum(e is not None for e in esc)}/10")
print("\n".join("  " + x for x in summ))
