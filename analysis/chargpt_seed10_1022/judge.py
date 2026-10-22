#!/usr/bin/env python3
"""chargpt_seed10_1022 の判定（spec_chargpt_seed10_1022.md の §3・§4・§5 と追補 1・§6。手順書 §3.4）。

  CHARGPT_SRC=$CG $CG/.venv/bin/python $S10/analysis/chargpt_seed10_1022/judge.py            # 既定の置き場を読んで judge.md・judge.json を書く
  ... judge.py --out <dir> [--load_root ..] [--rho_root ..] [--revival_root ..] [--silu_root ..]   # 検査用に入力と出力の置き場を替える

- 判定の閾値・窓・規則は spec のまま（行番号は各関数の docstring）。予測（spec の表の Opus 5.5 の欄）は採点だけする。
- 揃っていない単位は「未完」と数え、その部分のラベルは出さない（未完の数と、揃った分の数だけを書く）。
- 走らせる学習・GPU は無い（出来た出力を読むだけ）。torch を import しない。
- 読みの選択（spec の文だけで決まらない所）は READINGS に書き、judge.md の「読みの選択」の節に出す。
"""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S10 = HERE.parents[1]
RES = S10 / "results/chargpt_seed10_1022"
SPEC = S10 / "specs/spec_chargpt_seed10_1022.md"
sys.path.insert(0, str(HERE))
import revival10_plan as PLAN  # noqa: E402  C1・C3 の表（spec §5 の値）

SEEDS = list(range(10))
PEND = "未完"
NAN = float("nan")

# ---------------------------------------------------------------- 予測（spec の表の Opus 5.5 の欄。書き換えない）
PRED = {
    "A1": dict(line=67, kind="bin", p=0.95, text="700/700 で κ > 0、破れ 0"),
    "A2": dict(line=68, kind="cat", p={"LOAD_ALIGNS": .60, "MIXED": .35, "LOAD_REVERSED": .05}),
    "A2rec": dict(line=69, kind="bin", p=0.40, text="s_init が参照の帯に入る seed が 8/10 以上"),
    "A3": dict(line=70, kind="bin", p=0.70, text="INIT_PUSH_COMPARABLE"),
    "Arec": dict(line=71, kind="bin", p=0.50, text="課題 89〜100 の s の中央値が 0.6 未満の seed が 7/10 以上"),
    "B": dict(line=93, kind="cat", p={"OTHER_UNITS_CARRY": .55, "MIXED": .35, "SELF_CARRIES": .10}),
    "C1": dict(line=157, kind="cat", p={"FEEDBACK_NEEDED": .60, "MIXED": .30, "FEEDBACK_NOT_NEEDED": .10}),
    "C1rec": dict(line=158, kind="bin", p=0.50, text="nocommon で 8/9 以上が生き返らない"),
    "C2": dict(line=159, kind="cat", p={"READOUT_SHRINK_REVIVES": .70, "MIXED": .20, "NO_REVIVAL": .10}),
    "C2rec": dict(line=160, kind="bin", p=0.50, text="gamma03f で課題 31 末までに再び死ぬ（≥ 0.9）seed が 7/10 以上"),
    "D1": dict(line=185, kind="cat", p={"ESCAPE_REPLICATES": .55, "MIXED": .35, "NO_ESCAPE": .10}),
    "D2": dict(line=186, kind="bin", p=0.40, text="SINK_MOVES（D1 が起きた seed の中で）"),
    "D3": dict(line=187, kind="bin", p=0.45, text="LOSS_REPLICATES"),
}

READINGS = [
    ("(1) B の「沈む unit」", "生きた unit に限らない（4 項の和 < 0 の全 unit）。spec §4（spec:79）の「沈む unit は 4 項の和が負の unit」の数を出した写し元の集計 "
     "`$CG/analysis/chargpt_cl_1019_quicklook/rho_kappa_answers.py`（a6517244、blob 015e3c5f）の carriers()（37〜46 行）が、alive で絞らずに "
     "tot < 0 の unit で share_dn = Σ項/Σ和 を取る（spec §2 の 0.02/0.05/0.84/0.09 はこの値）。rho_split の出力（rho10.py）は unit ごとの 4 項と alive を持ち、集計の定義は持たないので、この集計に合わせた。"
     "③b は others_path（others_gamma は ③b の内訳で 4 項の和に入れない。rho_split.py:184 の閉合と同じ）。記録として、生きた unit に限った割合も並べる。"),
    ("(2) D1 の「課題 50 までに 5 課題以上続く」", "5 課題の続きが課題 50 までに終わる（始まり t で t+4 ≤ 50）と読む。判定の量は固定の一様 probe（layer_stats.csv の probe = uniform、u = 10000、layer = 3）の "
     "r_med > +1 かつ phi1_mean ≥ 0.75（spec_chargpt_cl_1019.md:581 の追補 6 §5 の定義。前例 judge_m2.py:138 の「t から t+4 まで全部」の形を一様 probe に移した）。"
     "記録として、続きの始まりだけが課題 50 以内の読み（t ≤ 50）の数も出す。"),
    ("(3) D2 の基準", "基準 = 逃げ始めの課題 t_esc（D1 の続きの始まり）の前の 5 課題（t_esc−5〜t_esc−1。課題 1 より前は無いので、t_esc ≤ 5 なら有る課題だけ）の L2 の r_med（一様 probe）の平均。"
     "課題 t_esc〜100 の課題末に r_med ≤ 基準 − 1 があれば「移る」。t_esc = 1 なら基準が無く判定できない（数に入れず、その数を書く）。"
     "前例 judge_m2.py:140 は「直前 5 課題の移動平均」で別の形なので使わない（spec:173 は固定の基準）。"),
    ("(4) A の生きた unit の probe", "load10.py の n_alive・own_sink は写し元 measure.py（:121）の定義のまま、課題 t の鎖の probe（init は課題 1 の鎖）で生きた unit を決める。"
     "spec:42 は「一様 probe で z > 0 のトークンがある unit」と書くが、spec:39 の「定義は chargpt_load_fold_1021 §2 のまま」と照合（seed 1000 で一致）に合わせた load10.py の出力を、そのまま使う。"
     "A2 の記録の帯 0.5 ± 3√(0.25/n_live) の n_live も、この n_alive（init の L1）。一様 probe で数え直すかは主の判断（compute_A の報告の 1）。"),
    ("(5) A の対象の腕と層", "A2・A3 は C3-2 の L1（spec:53 で ReLUcap-2 と L0 は記録だけ）。A1 は 2 腕 × 10 seed × 課題末 35 = 700 状態（spec:48）。"
     "r_mag は写し元 report.py:36 と同じく own_abs_median（全 512 unit の |G_own| の中央値）。生きた unit だけの own_abs_median_alive の比は記録。"),
    ("(6) A2 の符号検定", "s_early > s_init を +、< を −、等しいものは除いて両側 exact 符号検定（二項 1/2）。spec:50 の「9/10 で 0.021」と一致する。"),
    ("(7) B の「割合の中央値」と最大", "seed ごとに、課題 k+1 ∈ {2,3,5,10} の各状態で share(③a)、share(①)+share(②)、share(③b) を出し、4 課題の中央値を取って最大の項を決める。"
     "rho10 の検査（check_eval.all_pass）が落ちた状態は使わず、その seed は「検査落ち」として未完と同じく数えない。"),
    ("(8) C1 の符号検定の窓", "spec:131 の「窓の L1 の死の平均」の窓は spec:101 の定義（S_k+1〜E_k）とした。記録として判定窓 [R'_k, E_k] の平均の差の検定も出す。"
     "対の差は lnex1 − control（どちらも byakko の再生、追補 1）。0 の差は除く。"),
    ("(9) C の「死」の量", "各課題末の layer_stats.csv の probe = uniform・layer = 1・frac_p_lt005（u = 課題の更新数）。境は spec のまま「≤ 0.5 で生き返り」「> 0.5 で生き返らない」「≥ 0.9 で死」。"),
    ("(10) C2 の記録（再び死ぬ）", "gamma03f が課題 26 末に生き返り（≤ 0.5）、課題 27〜31 末のどこかで ≥ 0.9 になった seed を数え、予測（spec:160）は 10 seed のうちの数で採点する。資格（control ≥ 0.9）の seed の中の数も並べる。"),
    ("(11) C1 の記録の予測（nocommon 8/9）", "対象の数 n が 9 のときは文どおり 8 以上。n ≠ 9 のときは追補 1 の規則（n − 1 以上）で読み、その旨を書く。"),
    ("(12) D3 の δ", "δ = 3 × 0.0084 = 0.0252（前例 judge_m2.py:16 と同じ。spec:174 の 0.025 はその丸め）。0.025 と 0.0252 の間に入る値があれば記録に書く。"
     "量は judge_m2.py:23-33 のまま（online_all の課題 2〜10 と 91〜100 の平均、fresh gap = online_all − fresh の online_all、頭 100 更新 = online.npz の loss[:, :100] − H の平均）。"
     "条件 1 の「課題 1 末 ≤ 0.1」は 3 つに入れず記録にする。"),
    ("(13) 予測の採点", "結果のラベルに付けた確率と Brier（多クラスは Σ(p − 1[結果])²）を書く。結果が表に無いラベル（INSUFFICIENT・未完など）は採点しない。"),
]


# ---------------------------------------------------------------- 小さな道具
def sign_test_p(npos, nneg):
    """両側 exact 符号検定（二項 1/2）。0 の差は呼ぶ前に除く。"""
    n = npos + nneg
    if n == 0: return NAN
    k = min(npos, nneg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sign_counts(diffs):
    d = [x for x in diffs if np.isfinite(x)]
    return sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)


def rd(p): return list(csv.DictReader(open(p)))


def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return NAN


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else NAN


# ---------------------------------------------------------------- ラベル（spec の規則そのまま）
def label_A2(k_up):
    """spec:50。s_early > s_init の seed が 9/10 以上 → LOAD_ALIGNS、1/10 以下 → LOAD_REVERSED、それ以外 MIXED。"""
    return "LOAD_ALIGNS" if k_up >= 9 else ("LOAD_REVERSED" if k_up <= 1 else "MIXED")


def label_A3(k):
    """spec:52。r_mag ≥ 0.1 の seed が 8/10 以上 → INIT_PUSH_COMPARABLE。"""
    return "INIT_PUSH_COMPARABLE" if k >= 8 else "not INIT_PUSH_COMPARABLE"


def label_B(k_other, k_self):
    """spec:84。③a が最大の seed が 8/10 以上 → OTHER_UNITS_CARRY、① + ② が 8/10 以上 → SELF_CARRIES、それ以外 MIXED。"""
    if k_other >= 8: return "OTHER_UNITS_CARRY"
    if k_self >= 8: return "SELF_CARRIES"
    return "MIXED"


def label_C1(n, k):
    """追補 1（spec:242）。n ≥ 6 で、塞いで生き返らない seed が n − 1 以上 → FEEDBACK_NEEDED、2 以下 → FEEDBACK_NOT_NEEDED、それ以外 MIXED。n < 6 → INSUFFICIENT。"""
    if n < 6: return "INSUFFICIENT"
    if k >= n - 1: return "FEEDBACK_NEEDED"
    if k <= 2: return "FEEDBACK_NOT_NEEDED"
    return "MIXED"


def label_C2(n, k):
    """追補 1（spec:243）。n ≥ 6 で、gamma03f の課題 26 末の L1 の死 ≤ 0.5 の seed が n − 2 以上 → READOUT_SHRINK_REVIVES、2 以下 → NO_REVIVAL、それ以外 MIXED。n < 6 → INSUFFICIENT。"""
    if n < 6: return "INSUFFICIENT"
    if k >= n - 2: return "READOUT_SHRINK_REVIVES"
    if k <= 2: return "NO_REVIVAL"
    return "MIXED"


def label_D1(k):
    """spec:172。8/10 以上 → ESCAPE_REPLICATES、2/10 以下 → NO_ESCAPE、それ以外 MIXED。"""
    return "ESCAPE_REPLICATES" if k >= 8 else ("NO_ESCAPE" if k <= 2 else "MIXED")


def label_D2(m, k):
    """spec:173。D1 が起きた m seed のうち移った k seed の割合が 8 割以上 → SINK_MOVES。m = 0 は対象なし。"""
    if m == 0: return "対象なし"
    return "SINK_MOVES" if k / m >= 0.8 else "not SINK_MOVES"


def label_D3(k):
    """spec:174。3 つをすべて満たす seed が 8/10 以上 → LOSS_REPLICATES。"""
    return "LOSS_REPLICATES" if k >= 8 else "not LOSS_REPLICATES"


# ---------------------------------------------------------------- A（spec §3、spec:35-73）
A_END = list(range(1, 11)) + [25] + list(range(39, 51)) + list(range(89, 101))   # 課題末 35


def a_seed(states, head):
    """1 腕 1 seed の量。states = parts の *_states.csv の行、head = *_head.csv の行。"""
    U = {(int(float(r["task"])), r["stage"], int(float(r["layer"]))): r for r in states}
    Hd = {(int(float(r["task"])), r["stage"]): r for r in head}
    out = {}
    end = [Hd.get((t, "end")) for t in A_END]
    out["n_end_head"] = sum(r is not None for r in end)
    out["kappa_pos"] = sum(fnum(r["kappa"]) > 0 for r in end if r is not None)
    out["lemma_viol"] = int(sum(fnum(r["lemma_violation"]) for r in end if r is not None))
    out["kappa_nonfinite"] = sum(not np.isfinite(fnum(r["kappa"])) for r in end if r is not None)
    for l in (0, 1):
        g = lambda t, st="end", k="own_sink": fnum(U[(t, st, l)][k]) if (t, st, l) in U else NAN
        L = dict(s_init=g(0, "init"), n_live_init=g(0, "init", "n_alive"),
                 s_early=med([g(t) for t in range(1, 10)]),
                 s_1_10=[g(t) for t in range(1, 11)],
                 s_late39=med([g(t) for t in range(39, 51)]), s_late89=med([g(t) for t in range(89, 101)]),
                 full_init=g(0, "init", "full_sink"), full_early=med([g(t, k="full_sink") for t in range(1, 10)]),
                 full_late89=med([g(t, k="full_sink") for t in range(89, 101)]),
                 fixed_next_early=med([g(t, k="fixed_next_own_sink") for t in range(1, 10)]),
                 abs_init=g(0, "init", "own_abs_median"), abs_t1=g(1, "end", "own_abs_median"),
                 abs_alive_init=g(0, "init", "own_abs_median_alive"), abs_alive_t1=g(1, "end", "own_abs_median_alive"),
                 n_alive_end={t: g(t, k="n_alive") for t in (1, 10, 25, 50, 100)})
        L["r_mag"] = L["abs_init"] / L["abs_t1"] if L["abs_t1"] else NAN
        L["r_mag_alive"] = L["abs_alive_init"] / L["abs_alive_t1"] if L["abs_alive_t1"] else NAN
        n = L["n_live_init"]
        L["band_half"] = 3 * math.sqrt(0.25 / n) if n and np.isfinite(n) and n > 0 else NAN
        L["in_band"] = bool(abs(L["s_init"] - 0.5) <= L["band_half"]) if np.isfinite(L["band_half"]) and np.isfinite(L["s_init"]) else None
        L["early_gt_init"] = (L["s_early"] > L["s_init"]) if np.isfinite(L["s_early"]) and np.isfinite(L["s_init"]) else None
        out[f"L{l}"] = L
    out["cos2"] = {t: fnum(Hd[(t, "end")]["cos2"]) for t in (1, 10, 25, 50, 100) if (t, "end") in Hd}
    out["Q"] = {t: fnum(Hd[(t, "end")]["Q"]) for t in (1, 10, 25, 50, 100) if (t, "end") in Hd}
    return out


def judge_A(root):
    parts = Path(root) / "parts"; per = {}; status = {}
    for arm in ("C3", "ReLUcap"):
        for k in SEEDS:
            done = parts / f"{arm}_seed{k}_done.json"
            if not done.exists(): status[(arm, k)] = PEND; continue
            st, hd = rd(parts / f"{arm}_seed{k}_states.csv"), rd(parts / f"{arm}_seed{k}_head.csv")
            per[(arm, k)] = a_seed(st, hd); status[(arm, k)] = "完"
    n_pend = sum(v == PEND for v in status.values())
    res = dict(n_units_done=20 - n_pend, n_units_pending=n_pend,
               pending=[f"{a}/seed{k}" for (a, k), v in status.items() if v == PEND])
    # A1（検算）
    done = [v for v in per.values()]
    n_end = sum(v["n_end_head"] for v in done); kp = sum(v["kappa_pos"] for v in done); lv = sum(v["lemma_viol"] for v in done)
    a1 = dict(n_states=n_end, kappa_pos=kp, lemma_violations=lv, kappa_nonfinite=sum(v["kappa_nonfinite"] for v in done))
    a1["label"] = PEND if (n_pend or n_end != 700) else ("PASS（700/700 で κ > 0・破れ 0）" if kp == 700 and lv == 0 else "FAIL")
    a1["pred_outcome"] = None if a1["label"] == PEND else (kp == 700 and lv == 0)
    res["A1"] = a1
    # A2・A3（C3-2 の L1）
    c3 = {k: per[("C3", k)] for k in SEEDS if ("C3", k) in per}; full = len(c3) == 10
    rows = {k: v["L1"] for k, v in c3.items()}
    up = sum(1 for v in rows.values() if v["early_gt_init"] is True)
    npos, nneg, nz = sign_counts([v["s_early"] - v["s_init"] for v in rows.values()])
    res["A2"] = dict(n=len(rows), k_up=up, sign=(npos, nneg, nz), p=sign_test_p(npos, nneg),
                     label=label_A2(up) if full else PEND,
                     band_in=sum(v["in_band"] is True for v in rows.values()))
    res["A2"]["band_pred_outcome"] = (res["A2"]["band_in"] >= 8) if full else None
    k3 = sum(1 for v in rows.values() if np.isfinite(v["r_mag"]) and v["r_mag"] >= 0.1)
    res["A3"] = dict(n=len(rows), k=k3, label=label_A3(k3) if full else PEND)
    k_late = sum(1 for v in rows.values() if np.isfinite(v["s_late89"]) and v["s_late89"] < 0.6)
    res["Arec"] = dict(k_late89_lt06=k_late, n=len(rows), pred_outcome=(k_late >= 7) if full else None)
    res["per_seed"] = {f"{a}/seed{k}": v for (a, k), v in sorted(per.items())}
    return res


# ---------------------------------------------------------------- B（spec §4、spec:75-95）
B_EARLY, B_MID, B_LATE = (2, 3, 5, 10), (41, 45, 50), (91, 95, 100)
COMP = ("own", "self_up", "others_same", "others_path")


def b_state(js, layer, metric, alive_only=False):
    """1 状態の沈む unit（4 項の和 < 0）の各項の割合（Σ項/Σ和。rho_kappa_answers.carriers の share_dn）と、ρ・符号の一致の材料。"""
    L = js["layers"][str(layer)]; M = L[metric]
    u = {k: np.asarray(M[k], dtype=float) for k in COMP}; alive = np.asarray(L["alive"], dtype=bool)
    tot = sum(u[k] for k in COMP); dn = tot < 0
    if alive_only: dn = dn & alive
    sh = {k: float(u[k][dn].sum() / (tot[dn].sum() + 1e-300)) for k in COMP} if dn.any() else {k: NAN for k in COMP}
    return dict(share=sh, n_sink=int(dn.sum()), n_units=int(len(tot)), n_alive=int(alive.sum()),
                own=u["own"], self_up=u["self_up"], oth=u["others_same"] + u["others_path"])


def b_block(states):
    """states = 状態ごとの b_state の list。4 課題（または 3 課題）の割合の中央値と、最大の項。"""
    a = med([s["share"]["others_same"] for s in states]); s12 = med([s["share"]["own"] + s["share"]["self_up"] for s in states])
    b = med([s["share"]["others_path"] for s in states])
    vals = {"③a": a, "①+②": s12, "③b": b}
    top = max(vals, key=lambda k: vals[k]) if all(np.isfinite(v) for v in vals.values()) else None
    own = np.concatenate([s["own"] for s in states]) if states else np.zeros(0)
    sf = np.concatenate([s["self_up"] for s in states]) if states else np.zeros(0)
    oth = np.concatenate([s["oth"] for s in states]) if states else np.zeros(0)
    nz = (np.abs(own) + np.abs(sf)) > 0
    rho = float(np.median(np.abs(own + sf)[nz] / (np.abs(oth[nz]) + 1e-300))) if nz.any() else NAN   # rho_summary.stats の rho_q[1]
    agree = float((np.sign(own) == np.sign(sf))[nz].mean()) if nz.any() else NAN                       # rho_summary.stats の agree_nz
    return dict(med_share=vals, top=top, rho_med=rho, agree_12=agree, n_states=len(states), n_sink=[s["n_sink"] for s in states])


def judge_B(root):
    root = Path(root); res = {}
    for arm in ("C3", "ReLUcap"):
        per = {}; miss = []; failed = []
        for k in SEEDS:
            js = {}
            for t in B_EARLY + B_MID + B_LATE:
                f = root / arm / f"seed{k}" / f"t{t}_a_chain.json"
                if not f.exists(): miss.append(f"seed{k}/t{t}"); continue
                d = json.loads(f.read_text())
                if not d.get("check_eval", {}).get("all_pass", False): failed.append(f"seed{k}/t{t}"); continue
                if not all(str(l) in d["layers"] for l in (0, 1)): miss.append(f"seed{k}/t{t}(層)"); continue
                js[t] = d
            S = {}
            for l in (0, 1):
                for mt in ("A", "D"):
                    for bn, ts in (("early", B_EARLY), ("mid", B_MID), ("late", B_LATE)):
                        if all(t in js for t in ts):
                            S[f"L{l}_{mt}_{bn}"] = b_block([b_state(js[t], l, mt) for t in ts])
                            S[f"L{l}_{mt}_{bn}_alive"] = b_block([b_state(js[t], l, mt, alive_only=True) for t in ts])
            for v in S.values():
                for key in ("own", "self_up", "oth"): v.pop(key, None)
            per[k] = S
        main = {k: v.get("L1_D_early") for k, v in per.items()}
        done = {k: v for k, v in main.items() if v is not None}
        k_other = sum(v["top"] == "③a" for v in done.values()); k_self = sum(v["top"] == "①+②" for v in done.values())
        k_b = sum(v["top"] == "③b" for v in done.values())
        r = dict(n_seed_done=len(done), missing=miss, check_failed=failed, k_other=k_other, k_self=k_self, k_3b=k_b,
                 per_seed={f"seed{k}": v for k, v in per.items()})
        if arm == "C3":
            r["label"] = label_B(k_other, k_self) if len(done) == 10 else PEND
        else:
            r["label_record"] = label_B(k_other, k_self) if len(done) == 10 else PEND
        res[arm] = r
    return res


# ---------------------------------------------------------------- C（spec §5 と追補 1、spec:97-162・240-243）
def c_unit(root, part, k, arm, frm, to):
    d = Path(root) / part / f"seed{k}" / arm
    p = d / "provenance.json"
    if not p.exists(): return None
    pv = json.loads(p.read_text())
    if not (pv.get("complete") and pv.get("from_task") == frm and pv.get("to_task") == to): return None
    n = pv.get("n_per_task", 10000); death = {}; death0 = {}
    for r in rd(d / "layer_stats.csv"):
        if r["probe"] != "uniform" or int(float(r["u"])) != n: continue
        t, l = int(float(r["task"])), int(float(r["layer"]))
        (death if l == 1 else death0)[t] = fnum(r["frac_p_lt005"])
    return dict(death=death, death0=death0, compare_orig=pv.get("compare_orig"), bitwise=pv.get("control_bitwise_match"),
                lnex1_den_rel_diff=pv.get("lnex1_den_rel_diff"))


def c1_seed(ctrl, arm_death, S, E):
    """追補 1 の C1。ctrl・arm_death = {課題: L1 の死}。R' = control で S+1〜E のうち死 ≤ 0.5 になる最初の課題。
    塞いで生き返らない = 腕の死が [R', E] のすべての課題末で > 0.5。"""
    win = list(range(S + 1, E + 1))
    if any(t not in ctrl for t in win) or any(t not in arm_death for t in win):
        return dict(eligible=None, reason="課題の欠け")
    Rp = next((t for t in win if ctrl[t] <= 0.5), None)
    if Rp is None: return dict(eligible=False, Rp=None, reason="control が窓で生き返らない")
    jw = list(range(Rp, E + 1))
    no_rev = all(arm_death[t] > 0.5 for t in jw)
    return dict(eligible=True, Rp=Rp, no_revival=no_rev,
                diff_window=float(np.mean([arm_death[t] for t in win]) - np.mean([ctrl[t] for t in win])),
                diff_judge_window=float(np.mean([arm_death[t] for t in jw]) - np.mean([ctrl[t] for t in jw])),
                arm_min_in_judge_window=float(min(arm_death[t] for t in jw)))


def c1_like(root, part, table, arms, src_label):
    per = {}; pend = []
    for k, (R, S, E) in sorted(table.items()):
        u = {a: c_unit(root, part, k, a, S, E) for a in ("control",) + tuple(arms)}
        for a, v in u.items():
            if v is None: pend.append(f"seed{k}/{a}")
        row = dict(R_spec=R, S=S, E=E)
        if u["control"] is not None:
            row["control_compare_orig"] = u["control"]["compare_orig"]; row["control_bitwise"] = u["control"]["bitwise"]
        for a in arms:
            if u["control"] is None or u[a] is None: row[a] = dict(eligible=None, reason=PEND); continue
            row[a] = c1_seed(u["control"]["death"], u[a]["death"], S, E)
            if a == "lnex1": row[a]["den_rel_diff"] = u[a]["lnex1_den_rel_diff"]
        per[k] = row
    out = dict(pending=pend, per_seed={f"seed{k}": v for k, v in per.items()})
    for a in arms:
        rows = [v[a] for v in per.values()]
        pend_a = sum(r["eligible"] is None for r in rows)
        el = [r for r in rows if r["eligible"] is True]; n = len(el); kk = sum(r["no_revival"] for r in el)
        p1 = sign_counts([r["diff_window"] for r in el]); p2 = sign_counts([r["diff_judge_window"] for r in el])
        out[a] = dict(n_pending=pend_a, n_excluded=sum(r["eligible"] is False for r in rows), n=n, k_no_revival=kk,
                      label=(label_C1(n, kk) if pend_a == 0 else PEND),
                      sign_window=p1, p_window=sign_test_p(p1[0], p1[1]), sign_judge_window=p2, p_judge_window=sign_test_p(p2[0], p2[1]))
    return out


def judge_C(root):
    res = {}
    res["C1"] = c1_like(root, "C1", PLAN.C1_TABLE, ("lnex1", "nocommon"), "C3")
    n, k = res["C1"]["nocommon"]["n"], res["C1"]["nocommon"]["k_no_revival"]
    res["C1"]["rec_pred_outcome"] = None if res["C1"]["nocommon"]["label"] == PEND else (k >= (8 if n == 9 else n - 1))
    res["C1"]["rec_pred_note"] = "n = 9 で文どおり" if n == 9 else f"n = {n} なので n − 1 以上で読んだ"
    # C2
    per = {}; pend = []
    for k in SEEDS:
        u = {a: c_unit(root, "C2", k, a, PLAN.C2_FROM, PLAN.C2_TO) for a in ("control", "gamma03f", "gamma03")}
        for a, v in u.items():
            if v is None: pend.append(f"seed{k}/{a}")
        row = {}
        if u["control"] is not None:
            c26 = u["control"]["death"].get(26, NAN); row.update(control_26=c26, eligible=bool(c26 >= 0.9),
                                                                 control_compare_orig=u["control"]["compare_orig"], control_bitwise=u["control"]["bitwise"],
                                                                 control_27_31=[u["control"]["death"].get(t, NAN) for t in range(27, 32)])
        for a in ("gamma03f", "gamma03"):
            if u[a] is None: row[a] = None; continue
            d = u[a]["death"]; d26 = d.get(26, NAN); later = [d.get(t, NAN) for t in range(27, 32)]
            row[a] = dict(d26=d26, revived=bool(d26 <= 0.5), d27_31=later,
                          redie=bool(d26 <= 0.5 and any(np.isfinite(x) and x >= 0.9 for x in later)))
        per[k] = row
    def c2_count(a):
        ok = [v for v in per.values() if "eligible" in v and v.get(a) is not None]
        pending = 10 - len(ok)
        el = [v for v in ok if v["eligible"]]; n = len(el); kk = sum(v[a]["revived"] for v in el)
        return dict(n_pending=pending, n=n, n_not_eligible=len(ok) - n, k_revived=kk,
                    label=(label_C2(n, kk) if pending == 0 else PEND),
                    redie_all10=sum(v[a]["redie"] for v in ok), redie_eligible=sum(v[a]["redie"] for v in el))
    res["C2"] = dict(pending=pend, gamma03f=c2_count("gamma03f"), gamma03=c2_count("gamma03"), per_seed={f"seed{k}": v for k, v in per.items()})
    g = res["C2"]["gamma03f"]
    res["C2"]["rec_pred_outcome"] = None if g["n_pending"] else (g["redie_all10"] >= 7)
    # C3（記録だけ）
    res["C3"] = c1_like(root, "C3", PLAN.C3_TABLE, ("lnex1",), "ReLUcap")
    return res


# ---------------------------------------------------------------- D（spec §6、spec:164-189）
DELTA = 3 * 0.0084


def d1_seed(rmed3, phi3, last=50):
    """rmed3・phi3 = {課題: L3 の r_med・phi1_mean}（一様 probe、課題末）。読み (2): 5 課題の続きが課題 last までに終わる。"""
    ok = {t: (np.isfinite(rmed3.get(t, NAN)) and rmed3[t] > 1 and np.isfinite(phi3.get(t, NAN)) and phi3[t] >= 0.75) for t in rmed3}
    run = lambda t: all(ok.get(s, False) for s in range(t, t + 5))
    esc = next((t for t in range(1, last - 3) if run(t)), None)                 # t + 4 ≤ last
    esc_b = next((t for t in range(1, last + 1) if run(t)), None)               # 記録: 始まりだけ ≤ last
    return dict(escape=esc is not None, t_esc=esc, t_esc_start_le50=esc_b)


def d2_seed(rmed2, t_esc, last=100):
    """読み (3)。基準 = t_esc の前の 5 課題（有る課題だけ）の L2 の r_med の平均。t_esc〜last に ≤ 基準 − 1 があれば移る。"""
    pre = [rmed2[t] for t in range(max(1, t_esc - 5), t_esc) if t in rmed2 and np.isfinite(rmed2[t])]
    if not pre: return dict(judgeable=False, base=NAN, moved=None, t_move=None, n_pre=0)
    base = float(np.mean(pre))
    t_move = next((t for t in range(t_esc, last + 1) if t in rmed2 and rmed2[t] <= base - 1.0), None)
    return dict(judgeable=True, base=base, moved=t_move is not None, t_move=t_move, n_pre=len(pre),
                min_after=float(min(rmed2[t] for t in range(t_esc, last + 1) if t in rmed2)) if any(t in rmed2 for t in range(t_esc, last + 1)) else NAN)


def d3_seed(on, head, fresh):
    """on・head = 課題 1〜100 の list（0 始まりで課題 t は [t−1]）、fresh = {課題: fresh の online_all}。judge_m2.py:23-33・66 の形。"""
    early, late = float(np.mean(on[1:10])), float(np.mean(on[90:100]))
    gap = {t: on[t - 1] - fresh[t] for t in fresh if t <= len(on)}
    gs = gap[100] - gap[10] if (10 in gap and 100 in gap) else NAN
    he, hl = float(np.mean(head[1:10])), float(np.mean(head[90:100]))
    c = dict(worse=late - early, gap_shift=gs, head_early=he, head_late=hl)
    c["ok_worse"] = bool(c["worse"] >= DELTA); c["ok_gap"] = bool(np.isfinite(gs) and gs >= DELTA); c["ok_head"] = bool(hl > he)
    c["loss"] = c["ok_worse"] and c["ok_gap"] and c["ok_head"]
    c["near_delta"] = [k for k in ("worse", "gap_shift") if np.isfinite(c[k]) and 0.025 <= c[k] < DELTA]
    c["gap"] = {str(t): v for t, v in sorted(gap.items())}
    return c


def d_unit(d):
    p = d / "provenance.json"
    if not p.exists(): return None
    pv = json.loads(p.read_text())
    if not (pv.get("tasks") == 100 and pv.get("n_layer") == 4): return None
    pt = rd(d / "per_task.csv")
    if len(pt) != 100: return None
    E = {}
    for r in rd(d / "layer_stats.csv"):
        if r["probe"] == "uniform" and int(float(r["u"])) == 10000: E[(int(float(r["task"])), int(float(r["layer"])))] = r
    on = [fnum(r["online_all"]) for r in pt]; H = np.array([fnum(r["H"]) for r in pt])
    loss = np.load(d / "online.npz")["loss"].astype(np.float64); head = list((loss[:, :100] - H[:, None]).mean(1))
    fresh = {int(float(r["task"])): fnum(r["online_all"]) for r in rd(d / "fresh.csv")}
    g = lambda l, key: {t: fnum(E[(t, l)][key]) for t in range(1, 101) if (t, l) in E}
    return dict(r3=g(3, "r_med"), p3=g(3, "phi1_mean"), r2=g(2, "r_med"), on=on, head=head, fresh=fresh, E=E, end1=fnum(pt[0].get("end_metric")))


def judge_D(root, arms=("SiLU", "GELU", "SiLUcap")):
    root = Path(root); res = {}
    for arm in arms:
        ad = root / arm
        if not ad.exists() or not any(ad.glob("seed*")):
            res[arm] = dict(label_D1="未実施", label_D2="未実施", label_D3="未実施", n_done=0); continue
        per = {}; pend = []
        for k in SEEDS:
            u = d_unit(ad / f"seed{k}")
            if u is None: pend.append(f"seed{k}"); continue
            r1 = d1_seed(u["r3"], u["p3"])
            r2 = d2_seed(u["r2"], r1["t_esc"]) if r1["escape"] else None
            r3 = d3_seed(u["on"], u["head"], u["fresh"])
            rec = {f"L{l}": {str(t): dict(dead=fnum(u["E"][(t, l)].get("frac_p_lt005", u["E"][(t, l)].get("dead"))), r_med=fnum(u["E"][(t, l)]["r_med"]),
                                          neg_frac=fnum(u["E"][(t, l)]["neg_frac"])) for t in (10, 25, 50, 75, 100) if (t, l) in u["E"]} for l in range(4)}
            per[k] = dict(D1=r1, D2=r2, D3=r3, end1=u["end1"], layers=rec)
        full = not pend
        k1 = sum(v["D1"]["escape"] for v in per.values())
        d2 = [v["D2"] for v in per.values() if v["D2"] is not None]
        m = sum(x["judgeable"] for x in d2); k2 = sum(bool(x["moved"]) for x in d2 if x["judgeable"])
        k3 = sum(v["D3"]["loss"] for v in per.values())
        res[arm] = dict(n_done=len(per), pending=pend, k_D1=k1, k_D1_start_le50=sum(v["D1"]["t_esc_start_le50"] is not None for v in per.values()),
                        D2_m=m, D2_k=k2, D2_not_judgeable=len(d2) - m, k_D3=k3,
                        label_D1=label_D1(k1) if full else PEND, label_D2=label_D2(m, k2) if full else PEND, label_D3=label_D3(k3) if full else PEND,
                        push_direction_L3="未実施（新しい checkpoint に measure.py を当てる測定は judge.py の外）",
                        per_seed={f"seed{k}": v for k, v in per.items()})
    return res


# ---------------------------------------------------------------- 予測の採点
def score(key, outcome):
    P = PRED[key]
    if outcome is None or outcome == PEND or outcome == "未実施":
        return dict(question=key, spec_line=P["line"], outcome=outcome if outcome is not None else f"{PEND}・未実施・対象なし", scored=False)
    if P["kind"] == "bin":
        o = bool(outcome); p = P["p"]
        return dict(question=key, spec_line=P["line"], text=P["text"], outcome=o, p_assigned=p, p_of_outcome=p if o else 1 - p, brier=(p - o) ** 2, scored=True)
    if outcome not in P["p"]:
        return dict(question=key, spec_line=P["line"], outcome=outcome, scored=False, note="表に無い結果")
    return dict(question=key, spec_line=P["line"], outcome=outcome, p=P["p"], p_of_outcome=P["p"][outcome],
                brier=sum((p - (lab == outcome)) ** 2 for lab, p in P["p"].items()), scored=True)


def predictions(A, B, C, D):
    s = [score("A1", A["A1"]["pred_outcome"]), score("A2", A["A2"]["label"]), score("A2rec", A["A2"]["band_pred_outcome"]),
         score("A3", None if A["A3"]["label"] == PEND else A["A3"]["label"] == "INIT_PUSH_COMPARABLE"), score("Arec", A["Arec"]["pred_outcome"]),
         score("B", B["C3"]["label"]), score("C1", C["C1"]["lnex1"]["label"]), score("C1rec", C["C1"]["rec_pred_outcome"]),
         score("C2", C["C2"]["gamma03f"]["label"]), score("C2rec", C["C2"]["rec_pred_outcome"])]
    sd = D.get("SiLU", {})
    l1, l2, l3 = sd.get("label_D1", "未実施"), sd.get("label_D2", "未実施"), sd.get("label_D3", "未実施")
    s.append(score("D1", l1))
    s.append(score("D2", None if l2 in (PEND, "未実施", "対象なし") else l2 == "SINK_MOVES"))
    s.append(score("D3", None if l3 in (PEND, "未実施") else l3 == "LOSS_REPLICATES"))
    return s


# ---------------------------------------------------------------- 書き出し
def f3(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def render(R):
    A, B, C, D, S = R["A"], R["B"], R["C"], R["D"], R["predictions"]
    L = ["# chargpt_seed10_1022 の判定（judge.py）", "",
         f"- 作成 {R['created']}・script sha256 {R['script_sha256'][:16]}…・spec sha256 {R['spec_sha256'][:16]}…",
         "- 規則・閾値・窓は spec のまま。予測は spec の表の Opus 5.5 の欄を採点しただけ。「未完」= 出力が揃っていない単位がある（その部分のラベルは出さない）。", "",
         "## ラベル", "", "| 部分 | ラベル | 数 | 符号検定の p（両側 exact） | spec |", "| --- | --- | --- | --- | --- |"]
    a2 = A["A2"]; c1 = C["C1"]["lnex1"]; c2 = C["C2"]["gamma03f"]
    L += [f"| A1 検算 | {A['A1']['label']} | κ > 0 {A['A1']['kappa_pos']}/{A['A1']['n_states']}・破れ {A['A1']['lemma_violations']} | — | :48 |",
          f"| A2（主） | {a2['label']} | s_early > s_init {a2['k_up']}/{a2['n']}（+{a2['sign'][0]} −{a2['sign'][1]} 0:{a2['sign'][2]}） | {f3(a2['p'], 4)} | :49-50 |",
          f"| A3 | {A['A3']['label']} | r_mag ≥ 0.1 {A['A3']['k']}/{A['A3']['n']} | — | :52 |",
          f"| B（C3-2 L1 計量 D 早期） | {B['C3']['label']} | ③a {B['C3']['k_other']}・①+② {B['C3']['k_self']}・③b {B['C3']['k_3b']} / {B['C3']['n_seed_done']} | — | :81-84 |",
          f"| C1（lnex1） | {c1['label']} | 生き返らない {c1['k_no_revival']}/{c1['n']}（対象外 {c1['n_excluded']}・未完 {c1['n_pending']}） | 窓 S+1〜E: {f3(c1['p_window'], 4)}（+{c1['sign_window'][0]} −{c1['sign_window'][1]}） | :127-131・:241-242 |",
          f"| C2（gamma03f） | {c2['label']} | 26 末 ≤ 0.5 {c2['k_revived']}/{c2['n']}（資格外 {c2['n_not_eligible']}・未完 {c2['n_pending']}） | — | :138・:243 |"]
    for arm, d in D.items():
        L.append(f"| D（{arm}） | D1 {d['label_D1']}・D2 {d['label_D2']}・D3 {d['label_D3']} | " +
                 (f"D1 {d['k_D1']}/{d['n_done']}・D2 {d['D2_k']}/{d['D2_m']}・D3 {d['k_D3']}/{d['n_done']}" if d.get("n_done") else "出力なし") + " | — | :170-174 |")
    L += ["", "## 記録", "", "### A（C3-2 の L1、seed ごと）", "",
          "| seed | n_live init | s_init | 帯 ±（内） | s_early | s 39〜50 | s 89〜100 | r_mag | r_mag（生きた unit） | full s early | fixed-next s early |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, v in A["per_seed"].items():
        if not key.startswith("C3/"): continue
        x = v["L1"]
        L.append(f"| {key} | {f3(x['n_live_init'], 0)} | {f3(x['s_init'])} | {f3(x['band_half'])}（{x['in_band']}） | {f3(x['s_early'])} | {f3(x['s_late39'])} | {f3(x['s_late89'])} | {f3(x['r_mag'])} | {f3(x['r_mag_alive'])} | {f3(x['full_early'])} | {f3(x['fixed_next_early'])} |")
    L += ["", f"- 帯に入る seed {a2['band_in']}/{a2['n']}・課題 89〜100 の s の中央値 < 0.6 の seed {A['Arec']['k_late89_lt06']}/{A['Arec']['n']}。未完の単位: {', '.join(A['pending']) or 'なし'}。",
          "- ReLUcap-2（L1）と L0: " + "; ".join(f"{k} s_init {f3(v['L1']['s_init'])} s_early {f3(v['L1']['s_early'])} / L0 s_init {f3(v['L0']['s_init'])} s_early {f3(v['L0']['s_early'])}" for k, v in A["per_seed"].items() if k.startswith("ReLUcap/")) or "- ReLUcap-2: 出力なし",
          "- cos²・Q（C3-2、課題 1・10・25・50・100）: " + "; ".join(f"{k} cos² " + "/".join(f3(c) for c in v["cos2"].values()) for k, v in A["per_seed"].items() if k.startswith("C3/")), "",
          "### B（seed ごとの沈む unit の割合の中央値 ③a / ①+② / ③b と最大、ρ の中央値、①②の符号の一致）", ""]
    for arm in ("C3", "ReLUcap"):
        b = B[arm]
        L.append(f"- {arm}: 揃った seed {b['n_seed_done']}・欠け {len(b['missing'])}・検査落ち {len(b['check_failed'])}" + (f"（記録のラベル {b.get('label_record')}）" if arm == "ReLUcap" else ""))
        for k, S_ in b["per_seed"].items():
            for blk in ("L1_D_early", "L1_A_early", "L1_D_mid", "L1_D_late", "L0_D_early", "L1_D_early_alive"):
                if blk in S_:
                    v = S_[blk]; ms = v["med_share"]
                    L.append(f"  - {k} {blk}: {f3(ms['③a'], 2)} / {f3(ms['①+②'], 2)} / {f3(ms['③b'], 2)} → {v['top']}・ρ {f3(v['rho_med'], 2)}・一致 {f3(v['agree_12'], 2)}")
    L += ["", "### C", ""]
    for part in ("C1", "C3"):
        c = C[part]; L.append(f"- {part}{'（記録だけ・ReLUcap-2）' if part == 'C3' else ''}: 未完の単位 {len(c['pending'])}")
        for a in [x for x in ("lnex1", "nocommon") if x in c]:
            q = c[a]
            L.append(f"  - {a}: 生き返らない {q['k_no_revival']}/{q['n']}・ラベル（同じ規則）{q['label']}・符号検定 窓 S+1〜E p {f3(q['p_window'], 4)}・判定窓 p {f3(q['p_judge_window'], 4)}")
        for k, v in c["per_seed"].items():
            seg = [f"{k} S {v['S']} E {v['E']}"]
            for a in [x for x in ("lnex1", "nocommon") if x in v]:
                q = v[a]
                seg.append(f"{a}: " + (f"R' {q.get('Rp')}・生き返らない {q.get('no_revival')}・差 {f3(q.get('diff_window'))}" if q.get("eligible") else str(q.get("reason"))))
            co = v.get("control_compare_orig")
            if co: seg.append(f"control と元の差: online 最大 {f3(co.get('online_maxabs_max'), 4)}・L1 死 最大 {f3(co.get('uniform_L1_absdiff_max'), 4)}・ビット一致 {v.get('control_bitwise')}")
            L.append("  - " + "；".join(seg))
    c2r = C["C2"]
    L.append(f"- C2: 未完の単位 {len(c2r['pending'])}・gamma03（副）{c2r['gamma03']['label']}（{c2r['gamma03']['k_revived']}/{c2r['gamma03']['n']}）・gamma03f で再び死ぬ seed {c2r['gamma03f']['redie_all10']}（10 seed のうち）/ 資格の中 {c2r['gamma03f']['redie_eligible']}")
    for k, v in c2r["per_seed"].items():
        if "control_26" in v:
            g = v.get("gamma03f") or {}
            L.append(f"  - {k}: control 26 末 {f3(v['control_26'])}（資格 {v['eligible']}）・gamma03f 26 末 {f3(g.get('d26'))}・27〜31 " + "/".join(f3(x, 2) for x in g.get("d27_31", [])))
    L += ["", "### D", ""]
    for arm, d in D.items():
        if not d.get("n_done"): L.append(f"- {arm}: 未実施（出力なし）"); continue
        L.append(f"- {arm}: 揃った seed {d['n_done']}・未完 {d['pending']}・D1（読み (2)）{d['k_D1']}・始まり ≤ 50 の読み {d['k_D1_start_le50']}・D2 判定できない {d['D2_not_judgeable']}・L3 の押しの向き: {d['push_direction_L3']}")
        for k, v in d["per_seed"].items():
            d3 = v["D3"]
            L.append(f"  - {k}: t_esc {v['D1']['t_esc']}・D2 {v['D2']}・D3 悪化 {f3(d3['worse'], 4)} gap 差 {f3(d3['gap_shift'], 4)} 頭 {f3(d3['head_early'], 3)}→{f3(d3['head_late'], 3)} → {d3['loss']}（δ 近傍 {d3['near_delta']}）・課題 1 末 {f3(v['end1'])}")
    L += ["", "## 予測の採点（Opus 5.5 の欄。数は書き換えていない）", "", "| 問い | spec | 結果 | 結果に付けた確率 | Brier |", "| --- | --- | --- | --- | --- |"]
    for s in S:
        L.append(f"| {s['question']} | :{s['spec_line']} | {s['outcome']} | {f3(s.get('p_of_outcome'), 2) if s['scored'] else '採点しない' + ('（' + s['note'] + '）' if s.get('note') else '')} | {f3(s.get('brier'), 3) if s['scored'] else '—'} |")
    L += ["", "## 読みの選択（spec の文だけで決まらない所）", ""] + [f"- **{a}**: {b}" for a, b in READINGS]
    L += ["", "## 入力", ""] + [f"- {k}: {v}" for k, v in R["inputs"].items()]
    return "\n".join(L) + "\n"


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES))
    ap.add_argument("--load_root", default=str(RES / "load"))
    ap.add_argument("--rho_root", default=str(RES / "rho"))
    ap.add_argument("--revival_root", default=str(RES / "revival"))
    ap.add_argument("--silu_root", default=str(RES / "silu4"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A, B, C, D = judge_A(a.load_root), judge_B(a.rho_root), judge_C(a.revival_root), judge_D(a.silu_root)
    R = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %z"), script_sha256=sha256(__file__), spec_sha256=sha256(SPEC),
             inputs=dict(load=a.load_root, rho=a.rho_root, revival=a.revival_root, silu4=a.silu_root,
                         revival_plan_sha256=sha256(HERE / "revival10_plan.py")),
             A=A, B=B, C=C, D=D, readings=[dict(item=x, reading=y) for x, y in READINGS])
    R["predictions"] = predictions(A, B, C, D)
    (out / "judge.json.tmp").write_text(json.dumps(jsonable(R), indent=1, ensure_ascii=False)); os.replace(out / "judge.json.tmp", out / "judge.json")
    (out / "judge.md.tmp").write_text(render(R)); os.replace(out / "judge.md.tmp", out / "judge.md")
    print(f"A2 {A['A2']['label']} | A3 {A['A3']['label']} | B {B['C3']['label']} | C1 {C['C1']['lnex1']['label']} | C2 {C['C2']['gamma03f']['label']} | D(SiLU) {D.get('SiLU', {}).get('label_D1')}")
    return R


if __name__ == "__main__":
    main()
