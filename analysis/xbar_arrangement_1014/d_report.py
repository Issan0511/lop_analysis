"""xbar_arrangement_1014 / agent D: tables D1-D6 from results/xbar_arrangement_1014/d_bright_<arm>.json, d_mnist_control.json and the P176 out json
(read-only).  Writes results/xbar_arrangement_1014/d_summary.json and the table part of analysis/xbar_arrangement_1014/REPORT_D_3layer_brightness.md
(between the markers <!-- TABLES --> ... <!-- /TABLES --> if the report exists, otherwise a fresh file)."""
import os, json, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
P176 = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003/stage16_1008_transplant/out"
REPORT = os.path.join(HERE, "REPORT_D_3layer_brightness.md")
D = {a: json.load(open(os.path.join(RES, f"d_bright_{a}.json"))) for a in ("LR", "ELU", "R")}
MN = json.load(open(os.path.join(RES, "d_mnist_control.json")))
ARMS = [("LR", "raw"), ("LR", "std"), ("ELU", "raw"), ("ELU", "std"), ("R", "raw"), ("R", "std")]
GROUPS = [("t0", lambda S: S["t"] == 0), ("判定 t1", lambda S: S["verdict"] and S["t"] == 1), ("判定 t2–9", lambda S: S["verdict"] and 2 <= S["t"] <= 9),
          ("判定 t≥10", lambda S: S["verdict"] and S["t"] >= 10), ("判定外 t≥1", lambda S: S["t"] >= 1 and not S["verdict"])]
L = []; J = {}
def P(s=""): L.append(s)
def states(arm, cond, sel=lambda S: True):
    for s in range(3):
        R = D[arm]["runs"].get(f"{cond}_s{s}", {})
        for t, S in sorted(R.items(), key=lambda kv: int(kv[0])):
            S["t"] = int(t); S["seed"] = s
            if sel(S): yield S
def med(a): a = np.asarray([x for x in a if x is not None], float); a = a[np.isfinite(a)]; return float(np.median(a)) if a.size else float("nan")
def fr8(c): c = np.asarray(c, float); return (c / c.sum()).tolist() if c.sum() > 0 else [float("nan")] * 8
def s8(v, d=2): return " ".join("—" if not np.isfinite(x) else f"{x:.{d}f}" for x in v)
def fl(x, d=2): return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
def flag(S): return "v" if S["verdict"] else ("L" if S["input_lost"] else ("D" if S["dead"] else "-"))

# ---------------- checks ----------------
chk = dict(z2_relmed_max=0.0, Kbar_id=0.0, m_id=0.0, u_mean=0.0, zbar_id=0.0, nb_states=0, nb_mismatch=0, unit_mismatch=0)
for arm, cond in ARMS:
    for S in states(arm, cond):
        c = S["chk"]; chk["z2_relmed_max"] = max(chk["z2_relmed_max"], c["z2_vs_stored_relmed"] or 0); chk["Kbar_id"] = max(chk["Kbar_id"], abs(c["Kbar_minus_meanK2"]))
        chk["m_id"] = max(chk["m_id"], c["m_minus_meanz_maxabs"] / max(c["z_scale"], 1e-300)); chk["u_mean"] = max(chk["u_mean"], c["u_mean_maxabs"] / max(c["z_scale"], 1e-300))
        for k in ("all", "alive"):
            if S["D6"][k]: chk["zbar_id"] = max(chk["zbar_id"], S["D6"][k]["identity_maxabs"] / max(c["z_scale"], 1e-300))
    for s in range(3):
        f = os.path.join(P176, f"p176_{arm}_{cond}_s{s}.json")
        if not os.path.exists(f): continue
        p = json.load(open(f))
        for t, Sp in p["snaps"].items():
            if "units" not in Sp: continue
            Sm = D[arm]["runs"][f"{cond}_s{s}"].get(t)
            if Sm is None or "units" not in Sm: chk["nb_mismatch"] += 1; continue
            chk["nb_states"] += 1; chk["nb_mismatch"] += int(Sp["units"]["n_bridge"] != Sm["units"]["n_bridge"]); chk["unit_mismatch"] += int(Sp["units"]["unit"] != Sm["units"]["unit"])
J["checks"] = chk

# ---------------- D1 ----------------
P("### D1. K2 の分布（状態をまとめた値。min・max は状態をまたいだ最小・最大、mean・sd・K̄2 は状態の中央値）[実測]"); P()
P("| 腕 | 群 | 状態 | K2 min | K2 max | K2 mean | K2 sd | K̄2 | K2 ≤ 0 のある状態 | K2 ≤ 0 の入力（Σ, 割合） | K2 < 0.2·K̄2（Σ, 割合） |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
J["D1"] = {}
for arm, cond in ARMS:
    for g, sel in GROUPS:
        ss = list(states(arm, cond, sel))
        if not ss: continue
        n = len(ss) * 1200; le = sum(S["D1"]["n_le0"] for S in ss); lt = sum(S["D1"]["n_lt02"] for S in ss)
        rec = dict(states=len(ss), min=min(S["D1"]["min"] for S in ss), max=max(S["D1"]["max"] for S in ss), mean_med=med([S["D1"]["mean"] for S in ss]), sd_med=med([S["D1"]["sd"] for S in ss]),
                   Kbar_med=med([S["D1"]["Kbar"] for S in ss]), states_le0=sum(S["D1"]["n_le0"] > 0 for S in ss), n_le0=le, frac_le0=le / n, n_lt02=lt, frac_lt02=lt / n)
        J["D1"][f"{arm}_{cond}_{g}"] = rec
        P(f"| {arm} {cond} | {g} | {len(ss)} | {rec['min']:.3g} | {rec['max']:.4g} | {rec['mean_med']:.4g} | {rec['sd_med']:.3g} | {rec['Kbar_med']:.4g} | {rec['states_le0']} | {le}（{rec['frac_le0']:.2e}） | {lt}（{rec['frac_lt02']:.2e}） |")
m1 = MN["D1"]; P(f"| 対照 MNIST 1 層 LR_ref_s0 | T = 50 | 1 | {m1['min']:.4g} | {m1['max']:.4g} | {m1['mean']:.4g} | {m1['sd']:.3g} | {m1['Kbar']:.4g} | 0 | {m1['n_le0']} | {m1['n_lt02']} |")
P()
P("K2 ≤ 0 の入力を持つ t ≥ 1 の状態（全部）。f = その入力の占有（生きた unit の割合）、f 順位 = 入力の中の f の百分位（1 = 最大）、開/u⊥>0/位置の項だけで開く = その入力で z > 0 の生きた unit 数・u⊥ > 0 の数・z > 0 かつ u⊥ ≤ 0 の数、z̄ = 全 unit の平均 unit の前活性、L1 開 = その入力で開く第 1 層 unit 数（状態の中央値）、‖h1−µ₂‖ 順位 = 中心化ノルムの百分位 [実測]:"); P()
P("| 腕 | seed | t | 旗 | 生きた | K2（K̄2） | f | f 順位 | f の argmax か | 開 / u⊥>0 / 位置の項だけで開く | 同じ u⊥ で K2 = K̄2 なら開く | f_u 順位 | z̄（m̄） | L1 開（中央値） | ‖h1−µ₂‖ 順位 | P176 の閾値で橋がその入力を覆う対 | K2 を外した閾値で覆う対 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
J["D1_le0_states"] = []
for arm, cond in ARMS:
    for S in states(arm, cond, lambda S: S["t"] >= 1 and S["D1"]["n_le0"] > 0):
        D2 = S.get("D2", {}); dec = D2.get("decomp_le0", []); am = D2.get("decomp_argmax", {})
        for j, n in enumerate(S["D1"]["le0_idx"]):
            dj = dec[j] if j < len(dec) else {}
            rec = dict(arm=arm, cond=cond, seed=S["seed"], t=S["t"], flag=flag(S), n_alive=S["n_alive"], input=n, K2=S["D1"]["le0_K2"][j], Kbar=S["D1"]["Kbar"],
                       f=D2.get("f_at_le0", [None] * 99)[j], fpct=D2.get("fpct_at_le0", [None] * 99)[j], is_argmax=(am.get("n") == n), dec=dj, fupct=D2.get("fupct_at_le0", [None] * 99)[j],
                       zbar=S["D6"]["all"]["zbar_at_le0"][j], mbar=S["D6"]["all"]["mbar"], n1=S["D5"]["n1_at_le0"][j], n1_med=S["D5"]["n1_med"], normpct=S["D5"]["norm_pct_at_le0"][j],
                       pairs_P176=S.get("D3", {}).get("P176", {}).get("pairs_with_le0"), pairs_c=S.get("D3", {}).get("K2free", {}).get("pairs_with_le0"))
            J["D1_le0_states"].append(rec)
            P(f"| {arm} {cond} | {S['seed']} | {S['t']} | {rec['flag']} | {S['n_alive']} | {rec['K2']:.1f}（{rec['Kbar']:.0f}） | {fl(rec['f'])} | {fl(rec['fpct'], 3)} | {'○' if rec['is_argmax'] else '×'} | "
              f"{dj.get('n_open', '—')} / {dj.get('n_u_pos', '—')} / {dj.get('n_open_u_le0', '—')} | {dj.get('n_open_typ', '—')} | {fl(rec['fupct'], 3)} | {rec['zbar']:+.2f}（{rec['mbar']:+.1f}） | {rec['n1']:.0f}（{rec['n1_med']:.0f}） | {rec['normpct']:.3f} | {rec['pairs_P176']} | {rec['pairs_c']} |")
P(); P("（t0 の K2 ≤ 0: ELU std s0・s1・s2 に 34・25・9 入力、K2 は −1.43〜0、K̄2 1.7〜1.9。初期化では m̄ ≈ 0 で、表 D2 の t0 行に含まれる。）"); P()

# ---------------- D2 ----------------
P("### D2. 占有 f_n と K2（生きた unit のある状態）[実測]"); P()
P("f の 8 分位 = K2 の 8 分位（暗い方から）ごとの f の平均を状態で平均。f ≥ 1/2 の入力は状態の和。argmax は np.argmax（同点は最初の入力）。f_w = P161 の読み出しの重み付き版（P176 の f）。f_u = u⊥ > 0 の割合（位置の項を外した占有）。"); P()
P("| 腕 | 群 | 状態 | 順位相関 (f, K2) 中央値 [min, max] | f の 8 分位の平均（暗→明） | f ≥ 1/2（Σ） | うち K2 ≤ 0 | うち最暗 8 分位 | argmax f が最暗 8 分位の状態 | argmax f の K2 百分位（中央値） | f_w ≥ 1/2（Σ, K2 ≤ 0） | 順位相関 (f_u, K2) 中央値 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
J["D2"] = {}
for arm, cond in ARMS:
    for g, sel in GROUPS:
        ss = [S for S in states(arm, cond, sel) if "D2" in S]
        if not ss: continue
        sp = [S["D2"]["sp_f_K2"] for S in ss]; spv = [x for x in sp if x is not None and np.isfinite(x)]
        octf = np.nanmean(np.array([[np.nan if v is None else v for v in S["D2"]["oct_f"]] for S in ss], float), 0)
        rec = dict(states=len(ss), sp_med=med(sp), sp_min=min(spv) if spv else float("nan"), sp_max=max(spv) if spv else float("nan"), oct_f=octf.tolist(),
                   n_ge_half=sum(S["D2"]["n_ge_half"] for S in ss), ge_half_le0=sum(S["D2"]["ge_half_le0"] for S in ss), ge_half_oct0=sum(S["D2"]["ge_half_oct0"] for S in ss),
                   argmax_oct0=sum(S["D2"]["argmax_oct"] == 0 for S in ss), argmax_pct_med=med([S["D2"]["argmax_pct"] for S in ss]),
                   fw_ge_half=sum(S["D2"]["fw_n_ge_half"] for S in ss), fw_ge_half_le0=sum(S["D2"]["fw_ge_half_le0"] for S in ss), fu_sp_med=med([S["D2"]["fu_sp_K2"] for S in ss]))
        J["D2"][f"{arm}_{cond}_{g}"] = rec
        P(f"| {arm} {cond} | {g} | {len(ss)} | {fl(rec['sp_med'])} [{fl(rec['sp_min'])}, {fl(rec['sp_max'])}] | {s8(octf, 3)} | {rec['n_ge_half']} | {rec['ge_half_le0']} | {rec['ge_half_oct0']} | {rec['argmax_oct0']}/{len(ss)} | {fl(rec['argmax_pct_med'], 3)} | {rec['fw_ge_half']}（{rec['fw_ge_half_le0']}） | {fl(rec['fu_sp_med'])} |")
m2 = MN["D2"]; P(f"| 対照 MNIST 1 層 | T = 50 | 1 | {m2['sp_f_K']:.2f} | {s8(m2['oct_f'], 3)} | {m2['n_ge_half']} | 0 | 0 | {int(m2['argmax_oct'] == 0)}/1 | {m2['argmax_pct']:.3f} | — | — |")
P()
# K2 <= 0 present (t >= 1): summary
sub = [r for r in J["D1_le0_states"]]
nst = len({(r['arm'], r['cond'], r['seed'], r['t']) for r in sub})
J["D2_le0"] = dict(inputs=len(sub), states=nst, f_ge_half=sum(1 for r in sub if r["f"] is not None and r["f"] >= 0.5), fpct_min=min(r["fpct"] for r in sub if r["fpct"] is not None),
                   fpct_med=med([r["fpct"] for r in sub]), argmax_states=len({(r['arm'], r['cond'], r['seed'], r['t']) for r in sub if r["is_argmax"]}),
                   open_by_pos_only=sum(r["dec"].get("n_open_u_le0", 0) for r in sub), open_total=sum(r["dec"].get("n_open", 0) for r in sub),
                   zbar_pos=sum(1 for r in sub if r["zbar"] > 0))
for lab, fsel in (("判定する状態", lambda r: r["flag"] == "v"), ("判定外（死）", lambda r: r["flag"] != "v")):
    rs = [r for r in sub if fsel(r)]
    if not rs: continue
    stt = {(r['arm'], r['cond'], r['seed'], r['t']) for r in rs}
    P(f"- K2 ≤ 0 の入力（t ≥ 1、{lab}）: {len(rs)} 入力 / {len(stt)} 状態。f ≥ 1/2 は {sum(1 for r in rs if r['f'] >= 0.5)}/{len(rs)}、f の百分位の最小 {min(r['fpct'] for r in rs):.3f}・中央値 {med([r['fpct'] for r in rs]):.3f}、"
      f"argmax f が K2 ≤ 0 の入力の状態 {len({(r['arm'], r['cond'], r['seed'], r['t']) for r in rs if r['is_argmax']})}/{len(stt)}、"
      f"開いた unit のうち位置の項だけで開く（z > 0 ≥ u⊥）{sum(r['dec'].get('n_open_u_le0', 0) for r in rs)}/{sum(r['dec'].get('n_open', 0) for r in rs)}、"
      f"同じ u⊥ で K2 = K̄2 なら開く unit {sum(r['dec'].get('n_open_typ', 0) for r in rs)}（実際 {sum(r['dec'].get('n_open', 0) for r in rs)}）、f_u の百分位の中央値 {med([r['fupct'] for r in rs]):.3f}（最小 {min(r['fupct'] for r in rs):.3f}）、"
      f"z̄ > 0 は {sum(1 for r in rs if r['zbar'] > 0)}/{len(rs)} [実測]")
P()

# ---------------- D3 ----------------
P("### D3. 橋が覆う標的の閾値の入力の K2 [実測・恒等式]"); P()
P("事象 = (標的 i, 橋 j, n ∈ T_i ∩ O_j)。一意 = 少なくとも 1 本の橋が覆う (i, n)。全閾値 = 生きた標的すべての T_i（橋の有無によらない）。全入力の 8 分位は各 1/8 = 0.125。"); P()
P("**恒等式**: P176 の閾値 T_i は e′ = K2·D′_i の上位 3。D′_i ≥ 0（なめらかなゲート > 0、ELU は閉じた側 e^z > 0・開いた側 0）なので K2_n ≤ 0 の入力は e′ ≤ 0 で、e′ > 0 の入力が 3 個以上ある unit では T_i に入らない。下の表の P176 版の「K2 ≤ 0」列は構成上 0 になる。K2 を外した閾値（D′_i の上位 3）を併記した。"); P()
P("| 腕 | 群 | 状態 | 橋の対 | 事象 | 事象の 8 分位の割合（暗→明） | 全閾値の 8 分位の割合 | 事象 K2 ≤ 0 | 一意 K2 ≤ 0 | K2 を外した閾値: 対 | 事象の 8 分位の割合 | 事象 K2 ≤ 0 | 全閾値の 8 分位の割合 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
J["D3"] = {}
for arm, cond in ARMS:
    for g, sel in GROUPS:
        ss = [S for S in states(arm, cond, sel) if "D3" in S]
        if not ss: continue
        A = {v: {k: np.sum([S["D3"][v][k] for S in ss], 0).tolist() for k in ("ev_oct", "thr_oct", "un_oct")} for v in ("P176", "K2free")}
        for v in ("P176", "K2free"):
            for k in ("n_pairs", "ev_n", "ev_le0", "un_le0", "thr_le0", "un_n", "thr_n", "pairs_with_le0"): A[v][k] = int(sum(S["D3"][v][k] for S in ss))
        A["states"] = len(ss); J["D3"][f"{arm}_{cond}_{g}"] = A; a, c = A["P176"], A["K2free"]
        P(f"| {arm} {cond} | {g} | {len(ss)} | {a['n_pairs']} | {a['ev_n']} | {s8(fr8(a['ev_oct']))} | {s8(fr8(a['thr_oct']))} | {a['ev_le0']} | {a['un_le0']} | {c['n_pairs']} | {s8(fr8(c['ev_oct']))} | {c['ev_le0']}（対 {c['pairs_with_le0']}） | {s8(fr8(c['thr_oct']))} |")
m3 = MN["D3"]["P176"]; m3c = MN["D3"]["K2free"]
P(f"| 対照 MNIST 1 層 | T = 50 | 1 | {m3['n_pairs']} | {m3['ev_n']} | {s8(fr8(m3['ev_oct']))} | {s8(fr8(m3['thr_oct']))} | 0 | 0 | {m3c['n_pairs']} | {s8(fr8(m3c['ev_oct']))} | 0 | {s8(fr8(m3c['thr_oct']))} |")
P()

# ---------------- D4 ----------------
P("### D4. ELU raw の条件の内側の覆し 28 個（P176: Cov(e′,d) > 0 かつ A^単 > 0 > A、多クラスの核）[実測]"); P()
over = []; comp = []; agneg = []
for s in range(3):
    p = json.load(open(os.path.join(P176, f"p176_ELU_raw_s{s}.json")))
    for t, Sp in p["snaps"].items():
        t = int(t)
        if t < 1 or Sp["input_lost"] or Sp["dead"] or Sp["n_alive"] == 0 or "units" not in Sp or "mc" not in Sp["units"]: continue
        U = Sp["units"]; Sm = D["ELU"]["runs"][f"raw_s{s}"][str(t)]; Um = Sm["units"]; pos = {u: j for j, u in enumerate(Um["unit"])}
        for j, mrow in enumerate(U["mc"]):
            u = mrow["unit"]; jm = pos[u]
            rec = dict(seed=s, t=t, unit=u, p=U["p"][j], n_bridge=U["n_bridge"][j], n_bridge_mine=Um["n_bridge"][jm], T=Um["T"][jm], T_K2=Um["T_K2"][jm], T_pct=Um["T_pct"][jm], T_oct=Um["T_oct"][jm],
                       Tc_K2=Um["Tc_K2"][jm], Tc_pct=Um["Tc_pct"][jm], Tc_oct=Um["Tc_oct"][jm], n_bridge_c=Um["n_bridge_c"][jm], A=mrow["A"], Ag=mrow["Ag"], Aal=mrow["Aal"], cov=mrow["cov_ed"],
                       Kbar=Sm["D1"]["Kbar"], K2_min=Sm["D1"]["min"], n_le0_state=Sm["D1"]["n_le0"], br_ev_K2=Um["br_ev_K2"][jm], e_pos=Um["e_pos"][jm],
                       open_le0=Um["open_le0"][jm])
            isov = mrow["cov_ed"] > 0 and mrow["Aal"] > 0 and mrow["A"] < 0
            if isov: over.append(rec)
            if mrow["cov_ed"] > 0 and mrow["Ag"] < 0: agneg.append(rec)
            if mrow["cov_ed"] > 0 and not isov and U["n_bridge"][j] >= 3: comp.append(rec)
snaps_over = sorted({(r["seed"], r["t"]) for r in over}); comp = [r for r in comp if (r["seed"], r["t"]) in snaps_over]
P(f"覆し {len(over)} 個（P176 の 28 と一致するか: {len(over) == 28}）、橋の数は P176 の json と本走で一致 {sum(r['n_bridge'] == r['n_bridge_mine'] for r in over)}/{len(over)}。比較群 = 同じ状態（{', '.join(f's{a} t{b}' for a, b in snaps_over)}）の条件の内側・覆しでない・橋 ≥ 3 の unit {len(comp)} 個。"); P()
P("| seed | t | unit | p | 橋 | T_i の K2（その状態の K̄2・K2 min） | T_i の K2 百分位 | T_i の 8 分位 | K2 ≤ 0 の T | 最暗 8 分位の T | A^g | 橋が覆う閾値の入力の K2（中央値） | K2 を外した T の 8 分位 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(over, key=lambda r: (r["seed"], r["t"], r["unit"])):
    P(f"| {r['seed']} | {r['t']} | {r['unit']} | {r['p']:.3f} | {r['n_bridge']} | {', '.join(f'{x:.1f}' for x in r['T_K2'])}（{r['Kbar']:.0f}・{r['K2_min']:.1f}） | {', '.join(f'{x:.3f}' for x in r['T_pct'])} | {','.join(str(x) for x in r['T_oct'])} | "
      f"{sum(x <= 0 for x in r['T_K2'])} | {sum(x == 0 for x in r['T_oct'])} | {r['Ag']:+.3g} | {fl(med(r['br_ev_K2']), 1)} | {','.join(str(x) for x in r['Tc_oct'])} |")
P()
def grp_sum(rs):
    if not rs: return None
    T = np.array([r["T_oct"] for r in rs]); Tp = np.array([r["T_pct"] for r in rs]); TK = np.array([r["T_K2"] for r in rs]); Tc = np.array([r["Tc_oct"] for r in rs]); Tcp = np.array([r["Tc_pct"] for r in rs])
    ol = [any(r["open_le0"]) for r in rs if r["open_le0"]]
    return dict(n=len(rs), open_le0=f"{sum(ol)}/{len(ol)}" if ol else "—", T_oct0_frac=float((T == 0).mean()), T_dec0_frac=float((Tp < 0.1).mean()), unit_any_dec0=float((Tp < 0.1).any(1).mean()), unit_any_oct0=float((T == 0).any(1).mean()), unit_all_oct0=float((T == 0).all(1).mean()), T_le0=int((TK <= 0).sum()),
                T_pct_med=float(np.median(Tp)), bridges_med=float(np.median([r["n_bridge"] for r in rs])), Tc_oct0_frac=float((Tc == 0).mean()), Tc_pct_med=float(np.median(Tcp)),
                T_oct_hist=np.bincount(T.ravel(), minlength=8).tolist(), p_med=float(np.median([r["p"] for r in rs])))
J["D4"] = dict(n_over=len(over), snaps=snaps_over, over=over, over_sum=grp_sum(over), comp_sum=grp_sum(comp),
               per_snap={f"s{a}_t{b}": dict(over=grp_sum([r for r in over if (r['seed'], r['t']) == (a, b)]), comp=grp_sum([r for r in comp if (r['seed'], r['t']) == (a, b)])) for a, b in snaps_over},
               agneg_in_cond=len(agneg), agneg_sum=grp_sum(agneg))
P("| 群 | unit | 橋（中央値） | p（中央値） | T の入力のうち最暗 8 分位（最暗 10 分位） | 最暗 8 分位の T を 1 つ以上持つ unit（最暗 10 分位） | 3 つとも最暗 8 分位 | K2 ≤ 0 の T | T の K2 百分位（中央値） | T の 8 分位の分布（暗→明） | K2 を外した T: 最暗 8 分位・百分位 | K2 ≤ 0 の入力で開く unit（その入力がある状態） |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = [("覆し（全 28）", J["D4"]["over_sum"]), ("比較群（同じ状態・覆しでない・橋 ≥ 3）", J["D4"]["comp_sum"]), (f"参考: 条件の内側で A^g < 0（{len(agneg)} 個、全状態）", J["D4"]["agneg_sum"])]
for k, v in J["D4"]["per_snap"].items():
    rows.append((f"{k} 覆し", v["over"])); rows.append((f"{k} 比較群", v["comp"]))
for lab, g in rows:
    if g is None: P(f"| {lab} | 0 | | | | | | | | | | |"); continue
    P(f"| {lab} | {g['n']} | {g['bridges_med']:.0f} | {g['p_med']:.3f} | {g['T_oct0_frac']:.3f}（{g['T_dec0_frac']:.3f}） | {g['unit_any_oct0']:.3f}（{g['unit_any_dec0']:.3f}） | {g['unit_all_oct0']:.3f} | {g['T_le0']} | {g['T_pct_med']:.3f} | {' '.join(str(x) for x in g['T_oct_hist'])} | {g['Tc_oct0_frac']:.3f}・{g['Tc_pct_med']:.3f} | {g['open_le0']} |")
P()

# ---------------- D5 ----------------
P("### D5. ハブの K2 [実測]"); P()
P("各セル = 最暗 8 分位の割合 / K2 ≤ 0 の入力数（Σ）/ K2 百分位の中央値。norm5・norm1 = ‖h1 − µ₂‖ の上位 5%・1%（P178 の「ハブ = 中心化ノルムの大きい入力」）、o top1% = 第 2 層の開いた unit 数 o_n の上位 1%（P178 diag）、L1 開 下位 5% = 第 1 層の開いた unit 数が 5% 分位以下（依頼文の再導出）、L1 開 上位 5% = 95% 分位以上。"); P()
P("| 腕 | 群 | 状態 | norm5 | norm1 | o top1% | L1 開 下位 5%（入力数の中央値） | L1 開 上位 5%（入力数の中央値） | 順位相関 (K2, ‖h1−µ₂‖) | 順位相関 (K2, L1 開) |"); P("|---|---|---|---|---|---|---|---|---|---|")
J["D5"] = {}
for arm, cond in ARMS:
    for g, sel in GROUPS[:4]:
        ss = list(states(arm, cond, sel))
        if not ss: continue
        rec = {}
        for h in ("norm5", "norm1", "open2_top1", "l1low5", "l1high5"):
            hs = [S["D5"][h] for S in ss if h in S["D5"]]
            if not hs: rec[h] = None; continue
            oc = np.sum([x["oct"] for x in hs], 0); rec[h] = dict(oct=oc.tolist(), oct0_frac=float(oc[0] / max(oc.sum(), 1)), le0=int(sum(x["le0"] for x in hs)), pct_med=med([x["pct_med"] for x in hs]), n_med=med([x["n"] for x in hs]))
        rec["sp_K2_norm"] = med([S["D5"]["sp_K2_norm"] for S in ss]); rec["sp_K2_n1"] = med([S["D5"]["sp_K2_n1"] for S in ss]); rec["states"] = len(ss)
        J["D5"][f"{arm}_{cond}_{g}"] = rec
        c = lambda h: "—" if rec[h] is None else f"{rec[h]['oct0_frac']:.2f} / {rec[h]['le0']} / {fl(rec[h]['pct_med'])}"
        P(f"| {arm} {cond} | {g} | {len(ss)} | {c('norm5')} | {c('norm1')} | {c('open2_top1')} | {c('l1low5')}（{fl(rec['l1low5']['n_med'], 0)}） | {c('l1high5')}（{fl(rec['l1high5']['n_med'], 0)}） | {fl(rec['sp_K2_norm'])} | {fl(rec['sp_K2_n1'])} |")
P()

# ---------------- D6 ----------------
P("### D6. 平均 unit（全 H = 100 unit）[恒等式・実測]"); P()
P("z̄_n = mean_k z_kn = m̄ K2_n/K̄2 + ū⊥_n [恒等式、残差は下の検算]。比 = ū⊥_n / (|m̄| K2_n/K̄2) を K2 > 0 の入力で最大。m̄ < 0 のとき K2 > 0 の入力で z̄_n > 0 ⟺ 比 > 1 [恒等式]。"); P()
P("| 腕 | 群 | 状態 | m̄ 中央値 | m̄ < 0 の状態 | z̄ > 0 の入力（Σ） | うち K2 ≤ 0 | K2 ≤ 0 の入力（Σ）のうち z̄ > 0 | 比の最大（状態の最大、中央値） | 比 > 1 の状態 |"); P("|---|---|---|---|---|---|---|---|---|---|")
J["D6"] = {}
for arm, cond in ARMS:
    for g, sel in GROUPS:
        ss = [S for S in states(arm, cond, sel) if S["D6"]["all"]]
        if not ss: continue
        A = [S["D6"]["all"] for S in ss]; mr = [a["max_ratio_K2gt0"] for a in A]; mrv = [x for x in mr if x is not None and np.isfinite(x)]
        rec = dict(states=len(ss), mbar_med=med([a["mbar"] for a in A]), mneg=sum(a["mbar"] < 0 for a in A), n_pos=sum(a["n_pos"] for a in A), n_pos_le0=sum(a["n_pos_K2le0"] for a in A),
                   n_le0=sum(a["n_K2le0"] for a in A), max_ratio=max(mrv) if mrv else float("nan"), med_ratio=med(mr), states_ratio_gt1=sum(1 for x in mrv if x > 1))
        J["D6"][f"{arm}_{cond}_{g}"] = rec
        P(f"| {arm} {cond} | {g} | {len(ss)} | {rec['mbar_med']:+.3g} | {rec['mneg']} | {rec['n_pos']} | {rec['n_pos_le0']} | {rec['n_pos_le0']}/{rec['n_le0']} | {fl(rec['max_ratio'])}（{fl(rec['med_ratio'])}） | {rec['states_ratio_gt1']} |")
m6 = MN["D6"]; P(f"| 対照 MNIST 1 層 | T = 50 | 1 | {m6['mbar']:+.3g} | 1 | {m6['n_pos']} | 0 | 0/0 | {m6['max_ratio']:.2f} | 0 |")
P()
P(f"検算: z2 の再計算と保存値の相対差の中央値 ≤ {chk['z2_relmed_max']:.1e}（保存は float16）、|K̄2 − mean K2| ≤ {chk['Kbar_id']:.1e}、|m_k − mean_n z_kn|/max|z| ≤ {chk['m_id']:.1e}、|mean_n u⊥|/max|z| ≤ {chk['u_mean']:.1e}、"
  f"z̄ の恒等式の残差/max|z| ≤ {chk['zbar_id']:.1e}。橋の数（標的ごと）と生きた unit の集合は P176 の json と {chk['nb_states'] - chk['nb_mismatch']}/{chk['nb_states']}・{chk['nb_states'] - chk['unit_mismatch']}/{chk['nb_states']} 状態で一致。"
  f"対照: K̄ − mean K {MN['chk']['Kbar_minus_meanK']:.1e}、m の恒等式 {MN['chk']['m_minus_meanz_maxabs']:.1e}、z̄ の恒等式 {MN['chk']['zbar_identity_maxabs']:.1e}。")

json.dump(J, open(os.path.join(RES, "d_summary.json"), "w"), indent=1, default=lambda o: o.item() if isinstance(o, (np.integer, np.floating)) else str(o))
body = "\n".join(L)
if os.path.exists(REPORT):
    txt = open(REPORT).read()
    if "<!-- TABLES -->" in txt:
        txt = re.sub(r"<!-- TABLES -->.*<!-- /TABLES -->", lambda _: "<!-- TABLES -->\n" + body + "\n<!-- /TABLES -->", txt, flags=re.S)
        open(REPORT, "w").write(txt)
    else:
        open(REPORT + ".tables", "w").write(body)
else:
    open(REPORT, "w").write("# REPORT D\n\n<!-- TABLES -->\n" + body + "\n<!-- /TABLES -->\n")
print(body)
