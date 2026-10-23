#!/usr/bin/env python3
"""judge.py の単体の検査（合成した小さな入力でラベルの境目、実の出力の形での読み込み、欠けを「未完」と数えること）。
  CHARGPT_SRC=$CG $CG/.venv/bin/python $S10/analysis/chargpt_seed10_1022/check_judge.py [--tmp <dir>] [--out <json>]
本走の出力は読むだけ（書かない）。合成の入力と出力は --tmp（既定は一時 dir）に置く。
"""
import argparse, csv, importlib.util, json, math, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as J  # noqa: E402

RES = J.RES
CG = Path(os.environ.get("CHARGPT_SRC", "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019"))
OK = []


def check(name, cond, detail=""):
    OK.append(dict(name=name, pass_=bool(cond), detail=str(detail)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def wcsv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]


# ---------------------------------------------------------------- 1. ラベルの境目と符号検定
def t_labels():
    check("sign_test 9/10 = 22/1024", abs(J.sign_test_p(9, 1) - 22 / 1024) < 1e-15, J.sign_test_p(9, 1))
    check("sign_test 8/9 = 20/512（spec:131 の 0.039）", abs(J.sign_test_p(8, 1) - 20 / 512) < 1e-15, J.sign_test_p(8, 1))
    check("sign_test 対称・0 件は NaN", J.sign_test_p(1, 9) == J.sign_test_p(9, 1) and math.isnan(J.sign_test_p(0, 0)))
    check("sign_test 5/10 = 1", J.sign_test_p(5, 5) == 1.0)
    check("A2 10,9→ALIGNS 8,2→MIXED 1,0→REVERSED", [J.label_A2(k) for k in (10, 9, 8, 2, 1, 0)] == ["LOAD_ALIGNS"] * 2 + ["MIXED"] * 2 + ["LOAD_REVERSED"] * 2)
    check("A3 8→COMPARABLE 7→not", J.label_A3(8) == "INIT_PUSH_COMPARABLE" and J.label_A3(7) != "INIT_PUSH_COMPARABLE")
    check("B 8 ③a→OTHER 7→MIXED 8 自分→SELF", J.label_B(8, 0) == "OTHER_UNITS_CARRY" and J.label_B(7, 3) == "MIXED" and J.label_B(2, 8) == "SELF_CARRIES" and J.label_B(1, 7) == "MIXED")
    exp = {(9, 9): "FEEDBACK_NEEDED", (9, 8): "FEEDBACK_NEEDED", (9, 7): "MIXED", (9, 3): "MIXED", (9, 2): "FEEDBACK_NOT_NEEDED",
           (6, 5): "FEEDBACK_NEEDED", (6, 4): "MIXED", (6, 2): "FEEDBACK_NOT_NEEDED", (5, 5): "INSUFFICIENT", (0, 0): "INSUFFICIENT"}
    check("C1 n−1・2 以下・n<6（追補 1）", all(J.label_C1(n, k) == v for (n, k), v in exp.items()), {k: J.label_C1(*k) for k in exp})
    exp = {(10, 8): "READOUT_SHRINK_REVIVES", (10, 7): "MIXED", (10, 3): "MIXED", (10, 2): "NO_REVIVAL", (6, 4): "READOUT_SHRINK_REVIVES",
           (6, 3): "MIXED", (6, 2): "NO_REVIVAL", (5, 5): "INSUFFICIENT", (7, 5): "READOUT_SHRINK_REVIVES", (7, 4): "MIXED"}
    check("C2 n−2・2 以下・n<6（追補 1）", all(J.label_C2(n, k) == v for (n, k), v in exp.items()), {k: J.label_C2(*k) for k in exp})
    check("D1 8→REPL 7,3→MIXED 2→NO", [J.label_D1(k) for k in (8, 7, 3, 2)] == ["ESCAPE_REPLICATES", "MIXED", "MIXED", "NO_ESCAPE"])
    check("D2 4/5=0.8→MOVES 3/4→not 0→対象なし", J.label_D2(5, 4) == "SINK_MOVES" and J.label_D2(4, 3) == "not SINK_MOVES" and J.label_D2(0, 0) == "対象なし")
    check("D3 8→LOSS 7→not", J.label_D3(8) == "LOSS_REPLICATES" and J.label_D3(7) != "LOSS_REPLICATES")


# ---------------------------------------------------------------- 2. 1 seed の関数の境目
def t_seed_funcs():
    # C1: R' と窓の端
    S, E = 10, 20
    ctrl = {t: 0.95 for t in range(11, 21)}; ctrl.update({15: 0.5, 16: 0.2})
    arm = {t: 0.95 for t in range(11, 21)}
    r = J.c1_seed(ctrl, arm, S, E); check("C1 R' = 死 ≤ 0.5 の最初（0.5 ちょうどを含む）", r["Rp"] == 15 and r["no_revival"] is True, r)
    arm2 = dict(arm); arm2[20] = 0.5
    check("C1 窓の最後の課題 E で 0.5 ちょうど → 生き返った", J.c1_seed(ctrl, arm2, S, E)["no_revival"] is False)
    arm3 = dict(arm); arm3[14] = 0.1
    check("C1 R' より前の生き返りは判定窓の外", J.c1_seed(ctrl, arm3, S, E)["no_revival"] is True)
    check("C1 control が生き返らない → 対象外", J.c1_seed({t: 0.6 for t in range(11, 21)}, arm, S, E)["eligible"] is False)
    check("C1 課題の欠け → 未完扱い", J.c1_seed({t: 0.6 for t in range(11, 20)}, arm, S, E)["eligible"] is None)
    # D1
    def d1(lo, hi, r=1.5, p=0.8):
        rr = {t: (r if lo <= t <= hi else 0.0) for t in range(1, 101)}; pp = {t: (p if lo <= t <= hi else 0.5) for t in range(1, 101)}
        return J.d1_seed(rr, pp)
    check("D1 46〜50 の 5 課題 → 逃げる（t+4 = 50）", d1(46, 50)["t_esc"] == 46)
    x = d1(47, 60); check("D1 47〜60 → 逃げない（続きが 51 で終わる）・始まり ≤ 50 の読みでは 47", x["escape"] is False and x["t_esc_start_le50"] == 47, x)
    check("D1 4 課題だけ → 逃げない", d1(20, 23)["escape"] is False)
    check("D1 φ′ = 0.75 ちょうどは満たす", d1(11, 15, p=0.75)["t_esc"] == 11)
    check("D1 r_med = 1 ちょうどは満たさない（> +1）", d1(11, 15, r=1.0)["escape"] is False)
    # D2
    r2 = {t: 0.0 for t in range(1, 101)}; r2.update({t: 0.5 for t in range(6, 11)}); r2[80] = -0.5
    x = J.d2_seed(r2, 11); check("D2 基準 = 前の 5 課題の平均、基準 − 1 ちょうど → 移る", x["base"] == 0.5 and x["moved"] and x["t_move"] == 80, x)
    r2[80] = -0.4999; check("D2 基準 − 1 より上 → 移らない", J.d2_seed(r2, 11)["moved"] is False)
    x = J.d2_seed({t: 0.0 for t in range(1, 101)}, 3); check("D2 t_esc = 3 → 前の 2 課題で基準", x["n_pre"] == 2)
    check("D2 t_esc = 1 → 判定できない", J.d2_seed({t: 0.0 for t in range(1, 101)}, 1)["judgeable"] is False)
    # D3
    on = [0.1] * 100; head = [1.0] * 100
    for t in range(91, 101): on[t - 1] = 0.1 + J.DELTA; head[t - 1] = 1.01
    fresh = {10: 0.1 - 0.0, 100: 0.1 + J.DELTA - J.DELTA}   # gap(10) = 0、gap(100) = δ
    x = J.d3_seed(on, head, fresh); check("D3 悪化・gap の差が δ ちょうど、頭が上 → 喪失", x["loss"] is True, x)
    head2 = list(head); head2[90:100] = [1.0] * 10
    check("D3 頭 100 更新が等しい → 喪失でない（> を要る）", J.d3_seed(on, head2, fresh)["loss"] is False)
    on2 = list(on); on2[90:100] = [0.1 + 0.0251] * 10
    x = J.d3_seed(on2, head, {10: 0.1, 100: 0.1}); check("D3 悪化 0.0251（0.025 ≤ x < 0.0252）→ 満たさない・δ 近傍に記録", x["ok_worse"] is False and "worse" in x["near_delta"], x["worse"])
    # A
    def arow(task, stage, layer, s, absm, n=512):
        return dict(arm="C3", seed=0, task=task, stage=stage, layer=layer, n_alive=n, own_sink=s, own_abs_median=absm, own_abs_median_alive=absm,
                    full_sink=0.5, fixed_next_own_sink=0.5)
    st = [arow(0, "init", l, 0.5, 0.01) for l in (0, 1)] + [arow(t, "end", l, 0.7, 0.1) for t in J.A_END for l in (0, 1)]
    hd = [dict(task=0, stage="init", kappa=1, lemma_violation=0, cos2=0.1, Q=1)] + [dict(task=t, stage="end", kappa=1, lemma_violation=0, cos2=0.1, Q=1) for t in J.A_END]
    x = J.a_seed(st, hd)["L1"]
    check("A r_mag = 0.1 ちょうど（≥ 0.1）・s_early 0.7 > s_init 0.5", abs(x["r_mag"] - 0.1) < 1e-12 and x["early_gt_init"] is True, x["r_mag"])
    check("A 帯 512 unit → ±0.0663", abs(x["band_half"] - 3 * math.sqrt(0.25 / 512)) < 1e-15 and x["in_band"] is True)
    st2 = [dict(r, own_sink=0.7) if r["stage"] == "init" else r for r in st]
    check("A s_early = s_init は + に数えない", J.a_seed(st2, hd)["L1"]["early_gt_init"] is False)


# ---------------------------------------------------------------- 3. 合成の木で e2e（欠け → 未完）
def mk_A(root, n_up, drop=None, r_small=0):
    for arm in ("C3", "ReLUcap"):
        for k in range(10):
            if drop == (arm, k): continue
            s_init, s_end = 0.5, (0.7 if k < n_up else 0.3)
            absi = 0.001 if k < r_small else 0.05
            st = [dict(arm=arm, seed=k, task=0, stage="init", layer=l, n_alive=512, own_sink=s_init, own_abs_median=absi, own_abs_median_alive=absi, full_sink=.5, fixed_next_own_sink=.5) for l in (0, 1)]
            st += [dict(arm=arm, seed=k, task=t, stage="end", layer=l, n_alive=400, own_sink=s_end, own_abs_median=0.1, own_abs_median_alive=0.1, full_sink=.5, fixed_next_own_sink=.5) for t in J.A_END for l in (0, 1)]
            hd = [dict(task=0, stage="init", kappa=1e-5, lemma_violation=0, cos2=.1, Q=1)] + [dict(task=t, stage="end", kappa=1e-3, lemma_violation=0, cos2=.2, Q=1) for t in J.A_END]
            wcsv(root / "parts" / f"{arm}_seed{k}_states.csv", st); wcsv(root / "parts" / f"{arm}_seed{k}_head.csv", hd)
            (root / "parts" / f"{arm}_seed{k}_done.json").write_text("{}")


def mk_B(root, n_other, fail=None, drop=None):
    rng = np.random.default_rng(0)
    for arm in ("C3", "ReLUcap"):
        for k in range(10):
            for t in J.B_EARLY + J.B_MID + J.B_LATE:
                if drop == (arm, k, t): continue
                n = 16
                if k < n_other: comp = dict(own=-.1, self_up=-.1, others_same=-.5, others_path=-.3)
                else: comp = dict(own=-.5, self_up=-.3, others_same=-.1, others_path=-.1)
                L = {}
                for l in (0, 1):
                    M = {kk: (np.full(n, v) * (1 + 0.1 * rng.random(n))).tolist() for kk, v in comp.items()}
                    M["others_gamma"] = [0.0] * n
                    # 浮く unit を 4 個（沈む unit の割合に入らないこと）
                    for kk in ("own", "self_up", "others_same", "others_path"): M[kk][:4] = [5.0] * 4
                    L[str(l)] = dict(A=M, D=M, alive=[True] * n)
                d = dict(layers=L, check_eval=dict(all_pass=not (fail == (arm, k, t))))
                p = root / arm / f"seed{k}" / f"t{t}_a_chain.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(d))


def mk_C(root, part, table, arms, pattern, drop=None):
    """pattern(seed, arm, task) → L1 の死。"""
    for k, (R, S, E) in table.items():
        for a in arms:
            if drop == (k, a): continue
            d = root / part / f"seed{k}" / a; d.mkdir(parents=True, exist_ok=True)
            rows = [dict(task=t, u=10000, probe=pr, layer=l, frac_p_lt005=(pattern(k, a, t) if l == 1 else 0.1)) for t in range(S, E + 1) for pr in ("chain", "uniform") for l in (0, 1)]
            wcsv(d / "layer_stats.csv", rows)
            (d / "provenance.json").write_text(json.dumps(dict(complete=True, from_task=S, to_task=E, n_per_task=10000, compare_orig={}, control_bitwise_match=False)))


def mk_D(root, n_esc, n_loss, drop=None):
    for k in range(10):
        if drop == k: continue
        d = root / "SiLU" / f"seed{k}"; d.mkdir(parents=True, exist_ok=True)
        esc = k < n_esc
        rows = []
        for t in range(1, 101):
            for l in range(4):
                r = 0.0
                if l == 3 and esc and t >= 11: r = 1.5
                if l == 2 and esc and t >= 60: r = -1.5
                rows.append(dict(task=t, u=10000, probe="uniform", layer=l, r_med=r, phi1_mean=(0.8 if (l == 3 and esc and t >= 11) else 0.5), neg_frac=0.5, frac_p_lt005=0.0))
                rows.append(dict(task=t, u=100, probe="uniform", layer=l, r_med=9.0, phi1_mean=0.9, neg_frac=0.5, frac_p_lt005=0.0))   # u ≠ 10000 は読まないこと
        wcsv(d / "layer_stats.csv", rows)
        lose = k < n_loss
        on = [0.1 + (0.05 if (lose and t > 90) else 0.0) for t in range(1, 101)]
        wcsv(d / "per_task.csv", [dict(task=t, online_all=on[t - 1], H=1.0, end_metric=0.05) for t in range(1, 101)])
        loss = np.ones((100, 200)) * 1.1; loss[90:, :100] += (0.1 if lose else 0.0)
        np.savez_compressed(d / "online.npz", loss=loss.astype(np.float32))
        wcsv(d / "fresh.csv", [dict(task=10, online_all=0.1), dict(task=25, online_all=0.1), dict(task=50, online_all=0.1), dict(task=100, online_all=(0.1 if lose else 0.15))])
        (d / "provenance.json").write_text(json.dumps(dict(tasks=100, n_layer=4)))


def run(tmp, name, **kw):
    out = tmp / name / "out"
    args = ["--out", str(out)] + sum([[f"--{k}", str(v)] for k, v in kw.items()], [])
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return J.main(args)


def t_e2e(tmp):
    import revival10_plan as PL
    empty = tmp / "empty"; empty.mkdir(parents=True, exist_ok=True)
    base = dict(load_root=empty, rho_root=empty, revival_root=empty, silu_root=empty)
    # A
    for n_up, exp in ((9, "LOAD_ALIGNS"), (8, "MIXED"), (1, "LOAD_REVERSED")):
        d = tmp / f"A{n_up}"; mk_A(d, n_up, r_small=3)
        R = run(tmp, f"A{n_up}", **dict(base, load_root=d))
        check(f"e2e A2 {n_up}/10 → {exp}", R["A"]["A2"]["label"] == exp and R["A"]["A1"]["label"].startswith("PASS"), (R["A"]["A2"]["label"], R["A"]["A2"]["p"]))
    check("e2e A2 9/10 の p = 0.0215", abs(run(tmp, "A9b", **dict(base, load_root=tmp / "A9"))["A"]["A2"]["p"] - 22 / 1024) < 1e-12)
    check("e2e A3 r_mag < 0.1 が 3 seed → 7/10 → not", run(tmp, "A9c", **dict(base, load_root=tmp / "A9"))["A"]["A3"]["k"] == 7)
    d = tmp / "Adrop"; mk_A(d, 9, drop=("ReLUcap", 4)); R = run(tmp, "Adrop", **dict(base, load_root=d))
    check("e2e A ReLUcap の 1 単位の欠け → A1 未完・A2 は C3 だけで出る", R["A"]["A1"]["label"] == J.PEND and R["A"]["n_units_pending"] == 1 and R["A"]["A2"]["label"] == "LOAD_ALIGNS")
    d = tmp / "Adrop2"; mk_A(d, 9, drop=("C3", 4)); R = run(tmp, "Adrop2", **dict(base, load_root=d))
    check("e2e A C3 の欠け → A2・A3 未完", R["A"]["A2"]["label"] == J.PEND and R["A"]["A3"]["label"] == J.PEND)
    # B
    for n_o, exp in ((8, "OTHER_UNITS_CARRY"), (7, "MIXED"), (2, "SELF_CARRIES")):
        d = tmp / f"B{n_o}"; mk_B(d, n_o); R = run(tmp, f"B{n_o}", **dict(base, rho_root=d))
        check(f"e2e B ③a 最大 {n_o}/10 → {exp}", R["B"]["C3"]["label"] == exp, (R["B"]["C3"]["k_other"], R["B"]["C3"]["k_self"]))
    sh = R["B"]["C3"]["per_seed"]["seed0"]["L1_D_early"]["med_share"]
    check("e2e B 浮く unit を割合に入れない（③a の割合 ≈ 0.5）", abs(sh["③a"] - 0.5) < 0.05, sh)
    d = tmp / "Bfail"; mk_B(d, 8, fail=("C3", 3, 5)); R = run(tmp, "Bfail", **dict(base, rho_root=d))
    check("e2e B 検査落ちの 1 状態 → その seed を数えず 未完", R["B"]["C3"]["label"] == J.PEND and R["B"]["C3"]["n_seed_done"] == 9 and len(R["B"]["C3"]["check_failed"]) == 1)
    d = tmp / "Bdrop"; mk_B(d, 8, drop=("C3", 0, 95)); R = run(tmp, "Bdrop", **dict(base, rho_root=d))
    check("e2e B 終盤の 1 状態の欠け → 判定（早期）は出る・終盤の塊は無い", R["B"]["C3"]["label"] == "OTHER_UNITS_CARRY" and "L1_D_late" not in R["B"]["C3"]["per_seed"]["seed0"] and len(R["B"]["C3"]["missing"]) == 1)
    # C1: control は R_spec で生き返る。lnex1 は seed の先頭 n_no 個で生き返らない
    def pat(n_no, excl=()):
        def f(k, a, t):
            R_, S_, E_ = PL.C1_TABLE[k]
            if a == "control": return 0.95 if (t < R_ or k in excl) else 0.2
            if a in ("lnex1", "nocommon"): return 0.95 if sorted(PL.C1_TABLE).index(k) < n_no else (0.95 if t < R_ else 0.3)
            return 0.95
        return f
    for n_no, excl, exp in ((8, (), "FEEDBACK_NEEDED"), (7, (), "MIXED"), (2, (), "FEEDBACK_NOT_NEEDED"), (6, (8, 9), "FEEDBACK_NEEDED"), (5, (7, 8, 9), "FEEDBACK_NEEDED"), (4, (7, 8, 9), "MIXED"), (5, (6, 7, 8, 9), "INSUFFICIENT")):
        d = tmp / f"C1_{n_no}_{len(excl)}"; mk_C(d, "C1", PL.C1_TABLE, ("control", "lnex1", "nocommon"), pat(n_no, excl))
        R = run(tmp, f"C1_{n_no}_{len(excl)}", **dict(base, revival_root=d)); c = R["C"]["C1"]["lnex1"]
        check(f"e2e C1 生き返らない {n_no}・対象外 {len(excl)} → {exp}", c["label"] == exp, (c["n"], c["k_no_revival"], c["label"], c["p_window"]))
    d = tmp / "C1_8_0"; R = run(tmp, "C1_8_0b", **dict(base, revival_root=d))
    check("e2e C1 8/9 の符号検定（全 seed で lnex1 − control > 0 → 9/9）p = 2/512", abs(R["C"]["C1"]["lnex1"]["p_window"] - 2 / 512) < 1e-12, R["C"]["C1"]["lnex1"]["sign_window"])
    check("e2e C1 nocommon 8/9 → 記録の予測の結果 True", R["C"]["C1"]["rec_pred_outcome"] is True)
    d = tmp / "C1drop"; mk_C(d, "C1", PL.C1_TABLE, ("control", "lnex1", "nocommon"), pat(8), drop=(5, "lnex1")); R = run(tmp, "C1drop", **dict(base, revival_root=d))
    check("e2e C1 lnex1 の 1 単位の欠け → 未完 1・ラベル 未完", R["C"]["C1"]["lnex1"]["label"] == J.PEND and R["C"]["C1"]["lnex1"]["n_pending"] == 1 and len(R["C"]["C1"]["pending"]) == 1)
    # C2
    C2T = {k: (None, PL.C2_FROM, PL.C2_TO) for k in range(10)}
    def p2(n_rev, n_inel=0, redie=0):
        def f(k, a, t):
            if a == "control": return 0.9 if k < 10 - n_inel else 0.89     # 0.9 ちょうど = 資格あり。最後の n_inel 個が資格外
            if t == 25: return 0.95
            if k < n_rev: return 0.5 if t == 26 else (0.95 if k < redie else 0.3)   # 0.5 ちょうど = 生き返り
            return 0.8
        return f
    for n_rev, n_inel, exp in ((8, 0, "READOUT_SHRINK_REVIVES"), (7, 0, "MIXED"), (2, 0, "NO_REVIVAL"), (6, 4, "READOUT_SHRINK_REVIVES"), (5, 5, "INSUFFICIENT")):
        d = tmp / f"C2_{n_rev}_{n_inel}"; mk_C(d, "C2", C2T, ("control", "gamma03f", "gamma03"), p2(n_rev, n_inel, redie=7))
        R = run(tmp, f"C2_{n_rev}_{n_inel}", **dict(base, revival_root=d)); c = R["C"]["C2"]["gamma03f"]
        check(f"e2e C2 生き返り seed 0..{n_rev - 1}・資格外 {n_inel} → {exp}", c["label"] == exp, (c["n"], c["k_revived"], c["label"]))
    R = run(tmp, "C2_8_0b", **dict(base, revival_root=tmp / "C2_8_0"))
    check("e2e C2 再び死ぬ seed 7 → 記録の予測 True", R["C"]["C2"]["gamma03f"]["redie_all10"] == 7 and R["C"]["C2"]["rec_pred_outcome"] is True)
    d = tmp / "C2drop"; mk_C(d, "C2", C2T, ("control", "gamma03f", "gamma03"), p2(8), drop=(3, "control")); R = run(tmp, "C2drop", **dict(base, revival_root=d))
    check("e2e C2 control の欠け → 未完", R["C"]["C2"]["gamma03f"]["label"] == J.PEND and R["C"]["C2"]["gamma03f"]["n_pending"] == 1)
    # D
    for n_esc, n_loss, exp in ((8, 8, ("ESCAPE_REPLICATES", "SINK_MOVES", "LOSS_REPLICATES")), (2, 7, ("NO_ESCAPE", "SINK_MOVES", "not LOSS_REPLICATES")), (0, 0, ("NO_ESCAPE", "対象なし", "not LOSS_REPLICATES"))):
        d = tmp / f"D{n_esc}_{n_loss}"; mk_D(d, n_esc, n_loss); R = run(tmp, f"D{n_esc}_{n_loss}", **dict(base, silu_root=d)); s = R["D"]["SiLU"]
        check(f"e2e D 逃げ {n_esc}・喪失 {n_loss} → {exp}", (s["label_D1"], s["label_D2"], s["label_D3"]) == exp and R["D"]["GELU"]["label_D1"] == "未実施", (s["label_D1"], s["label_D2"], s["label_D3"], s["k_D1"], s["k_D3"]))
    d = tmp / "Ddrop"; mk_D(d, 8, 8, drop=6); R = run(tmp, "Ddrop", **dict(base, silu_root=d))
    check("e2e D 1 seed の欠け → 未完", R["D"]["SiLU"]["label_D1"] == J.PEND and R["D"]["SiLU"]["pending"] == ["seed6"])
    R = run(tmp, "all_empty", **base)
    check("e2e 入力が全部無い → 壊れず 未完・未実施、予測は採点しない", all(not s["scored"] for s in R["predictions"]) and R["D"]["SiLU"]["label_D1"] == "未実施")


# ---------------------------------------------------------------- 4. 実の出力の形での読み込み（値は判定に使わない）
def t_real_formats():
    # B: rho10 の出力を、写し元の集計 rho_kappa_answers.carriers と比べる
    spec = importlib.util.spec_from_file_location("rka", CG / "analysis/chargpt_cl_1019_quicklook/rho_kappa_answers.py")
    rka = importlib.util.module_from_spec(spec); spec.loader.exec_module(rka)
    worst = 0.0; n = 0
    for f in sorted((RES / "checks/B").glob("*_a_chain*.json")):
        d = json.loads(f.read_text())
        for l, L in d["layers"].items():
            for mt in ("A", "D"):
                mine = J.b_state(d, int(l), mt)["share"]
                ref = rka.carriers({k: np.asarray(L[mt][k], float) for k in rka.COMP} | {"alive": np.asarray(L["alive"], float)})["share_dn"]
                worst = max(worst, max(abs(mine[k] - ref[k]) for k in J.COMP)); n += 1
    check("実形式 B: 沈む unit の割合 = 写し元 rho_kappa_answers.carriers の share_dn", n > 0 and worst < 1e-12, f"{n} 状態 × 層 × 計量、差の最大 {worst:.1e}")
    d = json.loads((RES / "checks/B/C3_seed0_t2_a_chain.json").read_text())
    b = J.b_block([J.b_state(d, 1, "D")] * 4)
    check("実形式 B: b_block が動き、最大の項が決まる", b["top"] in ("③a", "①+②", "③b") and np.isfinite(b["rho_med"]))
    # C: checks/C の 1 課題の再生（C2 の control、25 → 26）
    u = J.c_unit(RES / "checks/C/replay1b", "C2", 0, "control", 25, 26)
    check("実形式 C: revival10 の出力を読める（課題 25・26 の L1 の死）", u is not None and set(u["death"]) == {25, 26} and all(np.isfinite(v) for v in u["death"].values()))
    check("実形式 C: to_task が窓と違う単位は未完（読まない）", J.c_unit(RES / "checks/C/replay1b", "C2", 0, "control", 25, 31) is None)
    # D: 4 ブロック SiLU seed 1000（white-san、50 課題）の layer_stats.csv の一様 probe で D1 の関数を回す（spec §2: L3 は課題 11〜15 に正側へ）
    p = CG / "results/chargpt_cl_1019/quicklook/M/SiLU/seed1000/layer_stats.csv"
    E = {(int(r["task"]), int(r["layer"])): r for r in J.rd(p) if r["probe"] == "uniform" and int(r["u"]) == 10000}
    r3 = {t: float(E[(t, 3)]["r_med"]) for t in range(1, 51) if (t, 3) in E}; p3 = {t: float(E[(t, 3)]["phi1_mean"]) for t in range(1, 51) if (t, 3) in E}
    x = J.d1_seed(r3, p3)
    check("実形式 D: seed 1000（4 ブロック SiLU）の一様 probe で D1 が動く", len(r3) == 50, f"t_esc = {x['t_esc']}（参考: spec §2 の「課題 11〜15 に正側へ」は μ/σ の 0 越えで、D1 の基準 > +1・φ′ ≥ 0.75 とは別）")
    try:
        check("実形式 D: 4 ブロック 50 課題の走は d_unit で未完になる（100 課題でない）", J.d_unit(p.parent) is None)
    except Exception as e:  # noqa: BLE001
        check("実形式 D: d_unit が壊れない", False, repr(e))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tmp", default=None); ap.add_argument("--out", default=None); a = ap.parse_args()
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="check_judge_"))
    t_labels(); t_seed_funcs(); t_e2e(tmp); t_real_formats()
    nf = sum(not r["pass_"] for r in OK)
    print(f"\n{len(OK) - nf}/{len(OK)} PASS（合成の入力: {tmp}）")
    if a.out: Path(a.out).write_text(json.dumps(dict(n=len(OK), n_fail=nf, judge_sha256=J.sha256(J.__file__), checks=OK), indent=1, ensure_ascii=False))
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
