#!/usr/bin/env python3
"""judge.py の検査（compute_G）。

(a) A2 の記録だけで、judge.py の関数が spec §2.1 の表（prereg/a2_table.json）・§2.2 の 1 − cos・§2.3 の J6 の参照（loadsign の
    components_windows.csv・windows.csv）・§7.4 の net100 の参照（prereg/a2_j4_fallback.txt、a2_motion_alive.json の課題ごとの値から）を出し直すか。
(b) 合成の入力で、ラベルの境（8/10・2/10・7/10、ρ の 1/2・2・1/3、引き分け、欠けた seed、INSUFFICIENT、J5b の早期に死んだ seed < 6）。
    合成の seed の記録を一時の置き場に書き、judge.main() を通しで回して judge.json のラベルを確かめる。

使い方: $CG/.venv/bin/python -I check_judge.py <出力フォルダ>   （出力 = check_judge.json・check_judge.md）
"""
import csv, json, math, shutil, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge as J  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PRE = J.RES / "prereg"
RES = []


def ok(name, cond, detail=""):
    RES.append(dict(name=name, ok=bool(cond), detail=str(detail)))


# ================================================================ (a) A2 の参照値の再現
def part_a():
    T = json.load(open(PRE / "a2_table.json"))
    for arm in ("C3", "ReLUcap"):
        S, why = J.arm_seeds(J.A2_ROOT.parent / arm)
        ok(f"a/{arm}: 10 seed が読める", len(S) == 10, why)
        P = J.per_seed(S)
        diffs = {}
        for s in range(10):
            ref = T[arm][s]; j1 = P[s]["j1"]
            cmp = dict(end1=(j1["end1"], ref["end1"]), dE=(j1["dE"], ref["dE"]), dE110=(j1["dE110"], ref["dE110"]),
                       g0=(P[s]["j2"]["g0"], ref["g0"]), g1=(P[s]["j2"]["g1"], ref["g1"]),
                       L0d100=(P[s]["j3"]["d"], ref["L0d100"]), L0d_91_100=(P[s]["j3"]["d_91_100"], ref["L0d_91_100"]), L0d50=(P[s]["j3"]["d50"], ref["L0d50"]),
                       L1d10=(P[s]["j5"]["L1d10"], ref["L1d10"]), L1d50=(P[s]["j5"]["L1d50"], ref["L1d50"]), L1d100=(P[s]["j5"]["L1d100"], ref["L1d100"]))
            for i in range(3):
                cmp[f"early{i}"] = (j1["early"][i], ref["early"][i]); cmp[f"late{i}"] = (j1["late"][i], ref["late"][i])
                cmp[f"cont100_{i}"] = (j1["cont100"][i], ref["cont100"][i]); cmp[f"fresh100_{i}"] = (j1["fresh100"][i], ref["fresh100"][i])
            for k, (a, b) in cmp.items():
                diffs[k] = max(diffs.get(k, 0.0), abs(a - b))
            ok(f"a/{arm}/seed{s}: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致",
               j1["loss"] == ref["j1"] and j1["phase_loss"] == ref["phase_loss"] and P[s]["j1b"]["pass_"] == ref["c1"]
               and P[s]["j3"]["onset50"] == ref["L0_onset50"] and P[s]["j5"]["Rk"] == ref["Rk"],
               dict(j1=(j1["loss"], ref["j1"]), c1=(P[s]["j1b"]["pass_"], ref["c1"]), onset=(P[s]["j3"]["onset50"], ref["L0_onset50"]), Rk=(P[s]["j5"]["Rk"], ref["Rk"])))
        worst = max(diffs.values())
        ok(f"a/{arm}: 連続量の最大の差（a2_table.json との、全 seed・全量）≤ 1e-12", worst <= 1e-12, {k: f"{v:.3g}" for k, v in diffs.items()})
        g = [P[s]["j2"]["g0"] for s in range(10)]; nloss = sum(P[s]["j1"]["loss"] for s in range(10))
        ok(f"a/{arm}: J1 の LOSS の数・g の中央値（§2.1）", (nloss, round(float(np.median(g)), 2 if arm == "C3" else 2)) == ((10, 15.1) if arm == "C3" else (0, 0.2)) or (arm == "C3" and nloss == 10 and abs(np.median(g) - 15.1) < 0.05) or (arm == "ReLUcap" and nloss == 0 and abs(np.median(g) - 0.20) < 0.005),
           dict(n_loss=nloss, g=J.mrange(g)))
        if arm == "C3":
            d = [P[s]["j3"]["d"] for s in range(10)]; early = [s for s in range(10) if P[s]["j5"]["early_dead"]]
            rk = [P[s]["j5"]["Rk"] for s in range(10)]; rv = [r for r in rk if r is not None]
            ok("a/C3: L0 の死 0.87 [0.00, 0.96]・d ≥ 0.5 が 8/10・J5a 10/10・R_k 9/10 で 37〜84（中央値 56）・J1b 10/10",
               abs(np.median(d) - 0.87) < 0.005 and abs(min(d)) < 0.005 and abs(max(d) - 0.96) < 0.005 and sum(x >= 0.5 for x in d) == 8 and len(early) == 10
               and len(rv) == 9 and min(rv) == 37 and max(rv) == 84 and np.median(rv) == 56 and sum(P[s]["j1b"]["pass_"] for s in range(10)) == 10,
               dict(d=J.mrange(d), n_ge05=sum(x >= 0.5 for x in d), J5a=len(early), Rk=rk))
            # 死の 2 つの読み口（layer_stats.csv の frac_p_lt005 と unit_stats.npz の uniform_q）の一致（読みの選択 3）
            mx = 0.0
            for s in range(10):
                U = S[s]["U"]; du = ((1 - U["uniform_q"].astype(np.float64)) < 0.005).mean(2)
                mx = max(mx, float(np.nanmax(np.abs(du - S[s]["dead_u"]))))
            ok("a/C3: 死の量 layer_stats.csv frac_p_lt005 と unit_stats.npz (1 − uniform_q) < 0.005 の差（全課題・両層）", mx == 0.0, f"最大の差 {mx:.3g}")
            # §2.2 の 1 − cos（全 512 行・生きた unit）
            C = J.cos_records(J.A2_ROOT, S)
            got = {w: (J.med([C[s][w][0] for s in C]), J.med([C[s][w][1] for s in C])) for w, _ in J.J4_WIN_NAMED}
            ref = {"2-10": (0.116, 0.116), "41-50": (0.031, 0.031), "91-100": (0.0010, 0.011)}
            ok("a/C3: §2.2 の課題末どうしの 1 − cos（表の 2〜3 桁）",
               all(abs(got[w][i] - ref[w][i]) <= 0.5 * 10 ** (math.floor(math.log10(ref[w][i])) - 1) + 1e-12 for w in ref for i in (0, 1)), got)
            # §7.4 / §12.3 の net100 の参照（a2_motion_alive.json の課題ごとの生きた unit の中央値 → M_k）
            A = json.load(open(PRE / "a2_motion_alive.json")); B = json.load(open(PRE / "a2_motion.json"))
            for l, refv, refall in ((0, (1.663, 1.370, 1.779), (1.165, 1.085, 1.779)), (1, (1.359, 1.221, 1.484), (1.028, 0.462, 1.356))):
                Ma = [J.M_of({t: A[str(s)][f"t{t}_L{l}_disp_alive"] for t in J.T_J4}, J.T_J4)[0] for s in range(10)]
                Mb = [J.M_of({t: B[str(s)][f"t{t}_L{l}_disp100_med"] for t in J.T_J4}, J.T_J4)[0] for s in range(10)]
                m1, m2 = J.mrange(Ma), J.mrange(Mb)
                ok(f"a/C3: §7.4 の net100 の参照 L{l}（a2_j4_fallback.txt の 3 桁）",
                   all(abs(x - y) <= 5e-4 for x, y in zip((m1["median"], m1["min"], m1["max"], m2["median"], m2["min"], m2["max"]), refv + refall)), dict(alive=m1, all=m2))
    # §2.3 の J6 の参照（loadsign）
    for arm, lab, labln in (("C3", "OTHER_TERM_FLIPS", "LN_HALVES"), ("ReLUcap", "OTHER_TERM_KEPT", "LN_HALVES")):
        Jd = J.read_j6(J.LOADSIGN, arm)
        js = {s: J.j6_seed(Jd, s) for s in range(10)}
        k6 = sum(js[s]["late"]["s_original_other"] >= 0.75 for s in js); kln = sum(js[s]["late"]["s_original_exact_K"] <= 0.60 for s in js)
        l6 = "OTHER_TERM_KEPT" if k6 >= J.thr(7, 10) else "OTHER_TERM_FLIPS"; lln = "LN_HALVES" if kln >= J.thr(7, 10) else "LN_NOT_HALVES"
        ok(f"a/J6 {arm}: §2.3 のラベル（{lab} {0 if arm == 'C3' else 10}/10・{labln} 10/10）", (l6, lln, k6, kln) == (lab, labln, 0 if arm == "C3" else 10, 10), (l6, k6, lln, kln))
        mx = 0.0; nc = 0
        for f in ("windows.csv", "components_windows.csv"):
            for r in J.rd(J.LOADSIGN / f):
                if r["arm"] != arm: continue
                v = js[int(r["seed"])][r["window"]]
                for c in J.J6_COLS:
                    if c in r and c != "n_alive":
                        mx = max(mx, abs(v[c] - float(r[c]))); nc += 1
        ok(f"a/J6 {arm}: seed 内の窓の中央値が loadsign の windows.csv・components_windows.csv と一致（{nc} 値）", mx <= 1e-12, f"最大の差 {mx:.3g}")
    # A2 の再生の開始の checkpoint（task t−1 末）が T_J4 の全課題で在るか（§6.4）
    miss = [(s, t) for s in range(10) for t in J.T_J4 if not (J.A2_ROOT / f"seed{s}/ckpts/task{t-1}.pt").exists() or not (J.A2_ROOT / f"seed{s}/ckpts/task{t-1}_adam.pt").exists()]
    ok("a/A2: 再生の開始の checkpoint（重みと Adam）が 290 課題すべてに在る", not miss, miss[:10])
    nu100 = sum(1 for s in range(10) for _ in (J.A2_ROOT / f"seed{s}/ckpts").glob("u100_task*.pt"))
    RES.append(dict(name="a/A2: 更新 100 の checkpoint（u100_task*.pt）の数（記録）", ok=True, detail=f"{nu100} 個（byakko の展開には無い。net100 は prereg の値から照合）"))


# ================================================================ (b) 合成の入力
H0 = 0.5


def loss_row(tau, floor):
    u = np.arange(1, J.NU + 1, dtype=np.float64)
    return (H0 + np.exp(-u / tau) + floor).astype(np.float32)


def gen_seed(d, loss=True, g=15.0, d100=0.9, l1_early=0.95, Rk=50, motion=1.5, nalive=512, end1=0.05, dE_scale=1.0):
    d.mkdir(parents=True, exist_ok=True)
    on = np.zeros((J.NT, J.NU), np.float32)
    for t in range(1, J.NT + 1):
        late = loss and t > 50
        on[t - 1] = loss_row(3000 if late else 300, (0.05 * dE_scale if late else 0.01))
    fresh = loss_row(300, 0.01)
    np.savez(d / "online.npz", loss=on, aux=np.zeros_like(on))
    np.savez(d / "fresh_online_t100.npz", loss=fresh, aux=np.zeros_like(fresh))
    with open(d / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "n", "H", "online_first", "online_all", "end_metric", "end_loss", "data_sha", "rule_sha"])
        for t in range(1, J.NT + 1): w.writerow([t, J.NU, H0, on[t - 1, 0], float(on[t - 1].astype(np.float64).mean()), end1 if t == 1 else 0.05, "nan", "x", "y"])
    with open(d / "fresh.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "n", "H", "online_first", "online_all", "end_metric"])
        for t in (10, 25, 50, 100): w.writerow([t, J.NU, H0, fresh[0], float(fresh.astype(np.float64).mean()), 0.05])
    rn = np.ones((J.NT, 2, 512), np.float32)
    for t in range(1, J.NT + 1): rn[t - 1] *= np.float32(1 + (g - 1) * (t - 1) / 99)
    rng = np.random.default_rng(0)
    mu0 = rng.normal(size=(J.NT, 2, 512)).astype(np.float32)
    np.savez(d / "unit_stats.npz", rownorm=rn, chain_dead=np.zeros((J.NT, 2, 512), bool), uniform_q=np.full((J.NT, 2, 512), 0.5, np.float32),
             chain_mu_u0=mu0, chain_mu_u100=mu0 - 0.5, chain_sd_u0=np.ones_like(mu0))
    with open(d / "layer_stats.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "u", "probe", "layer", "frac_p_lt005", "frac_p_ge099", "r_med"])
        for t in range(1, J.NT + 1):
            d0 = d100 * t / 100
            d1 = l1_early if t <= 10 else (0.3 if (Rk is not None and t >= Rk) else 0.95)
            for probe in ("uniform", "chain"):
                w.writerow([t, J.NU, probe, 0, d0, 0, -t / 30]); w.writerow([t, J.NU, probe, 1, d1, 0, -1])
    mo = np.full((J.NT, 2, 512), motion, np.float32); al = np.zeros((J.NT, 2, 512), bool); al[:, :, :nalive] = True
    M = np.stack([mo, mo * 4, mo * 1.2, mo * 0.01, mo * 0 + 10], -1)                     # compute_W の形（save_records）
    np.savez(d / "within_motion.npz", motion=M, alive=np.stack([al, al], -1), have=np.ones(J.NT, bool), quantities=np.array(J.MOTION_Q),
             alive_points=np.array(["u_half", "u_end"]))
    with open(d / "chain_stats.csv", "w", newline="") as f:                              # 縮退 0・S-entropy の外れ 0
        w = csv.writer(f); w.writerow(["task", "distinct_ctx_500upd", *J.ENTROPY_COLS])
        for t in range(1, J.NT + 1): w.writerow([t, 4000, H0, H0, 0.0, 0.01, 0.01, 0.01, True, True, "file"])


def gen_replay(src, dst, tasks=J.T_J4):
    """合成の A2 の within_motion から、compute_W の replay の形（replay_task<t>.npz・replay_compare.csv・replay_provenance.json）を作る。"""
    dst.mkdir(parents=True, exist_ok=True); z = np.load(src / "within_motion.npz")
    for t in tasks:
        np.savez(dst / f"replay_task{t}.npz", motion=z["motion"][t - 1], alive=z["alive"][t - 1], online=np.zeros(J.NU, np.float32),
                 quantities=z["quantities"], alive_points=z["alive_points"])
    with open(dst / "replay_compare.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "task", "online_bitwise", "online_absdiff_max", "online_absdiff_mean", "state_bitwise", "state_reldiff_max"])
        for t in tasks: w.writerow([0, t, False, 0.01, 0.002, False, 0.5])
    json.dump(dict(complete=True), open(dst / "replay_provenance.json", "w"))


def gen_j6(path, arm, other_by_seed, exact_by_seed, seeds):
    path.mkdir(parents=True, exist_ok=True)
    fa, fb = path / "states.csv", path / "components_states.csv"
    newa, newb = not fa.exists(), not fb.exists()
    with open(fa, "a", newline="") as f1, open(fb, "a", newline="") as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        if newa: w1.writerow(["arm", "seed", "task", "layer", "n_alive", "s_original_direct_K", "s_radial", "s_original_exact_K"])
        if newb: w2.writerow(["arm", "seed", "task", "n_alive", "s_original_self", "s_original_other", "s_original_residual", "original_other_abs_share"])
        for s in seeds:
            for t in J.CKPT_TASKS:
                w1.writerow([arm, s, t, 1, 100, 0.9, 0.3, exact_by_seed[s]]); w2.writerow([arm, s, t, 100, 1.0, other_by_seed[s], 0.5, 0.3])


def run_case(tmp, name, X_specs, A2_specs, j6X=None, replay=True, extra=()):
    root = tmp / name; a2 = root / "A2"; res = root / "res"
    for s, kw in A2_specs.items(): gen_seed(a2 / f"seed{s}", **kw)
    for s, kw in X_specs.items(): gen_seed(res / "M2" / "A4" / f"seed{s}", **kw)
    if replay:
        for s, kw in A2_specs.items(): gen_replay(a2 / f"seed{s}", res / "replay_A2" / f"seed{s}")
    gen_j6(root / "a2j6", "C3", {s: 0.2 for s in range(10)}, {s: 0.5 for s in range(10)}, range(10))
    if j6X: gen_j6(res / "j6", "A4", j6X[0], j6X[1], j6X[2])
    O = J.main(["--out", str(root / "out"), "--res", str(res), "--arms", "A4", "--a2_root", str(a2), "--replay_root", str(res / "replay_A2"),
                "--j6_root", str(res / "j6"), "--a2_j6", str(root / "a2j6"), "--no_ckpt", *extra])
    return json.load(open(root / "out" / "judge.json"))


def part_b():
    # 小道具の境
    ok("b/sign: 9/10 → p = 0.021、8/10 → 0.109、引き分けを除く", abs(J.sign_test_p(9, 1) - 0.021484375) < 1e-15 and abs(J.sign_test_p(8, 2) - 0.109375) < 1e-15
       and J.sign_summary([1] * 8 + [0, 0])["n_pairs"] == 10 and abs(J.sign_summary([1] * 8 + [0, 0])["p"] - 2 / 256) < 1e-15 and np.isnan(J.sign_test_p(0, 0)))
    ok("b/thr: ceil(k·n/10)", [J.thr(8, n) for n in (10, 9, 8)] == [8, 8, 7] and [J.thr(2, n) for n in (10, 9, 8)] == [2, 2, 2] and [J.thr(7, n) for n in (10, 9, 8)] == [7, 7, 6])
    L = lambda k, n: J.label_count(k, n, 8, 2, ("P", "G"))
    ok("b/label_count: n=10 の 8/7/3/2、n=9 の 8/7、n=8 の 7/6、n=7 は INSUFFICIENT",
       [L(8, 10), L(7, 10), L(3, 10), L(2, 10), L(8, 9), L(7, 9), L(7, 8), L(6, 8), L(7, 7)] == ["P", "MIXED", "MIXED", "G", "P", "MIXED", "P", "MIXED", "INSUFFICIENT"])
    R = lambda r: J.label_ratio(r, 10, 10, "S", "L", "M")
    ok("b/label_ratio: ρ = 0.5 → SIMILAR、0.4999 → SMALLER、2 → SIMILAR、2.0001 → LARGER、NaN と n < 8 → INSUFFICIENT",
       [R(0.5), R(0.4999), R(2.0), R(2.0001), R(float("nan")), J.label_ratio(0.1, 7, 10, "S", "L", "M")] == ["M", "S", "M", "L", "INSUFFICIENT", "INSUFFICIENT"])
    ok("b/label_J5b: m = 5 → INSUFFICIENT、6 中 5 → REVIVES、10 中 8/3/2、使える 7 → INSUFFICIENT",
       [J.label_J5b(10, 5, 5), J.label_J5b(10, 6, 5), J.label_J5b(10, 10, 8), J.label_J5b(10, 10, 3), J.label_J5b(10, 10, 2), J.label_J5b(7, 7, 7)]
       == ["INSUFFICIENT", "L1_REVIVES", "L1_REVIVES", "MIXED", "NO_REVIVAL", "INSUFFICIENT"])
    tmp = Path(tempfile.mkdtemp(prefix="cj_", dir=str(OUT)))
    try:
        A2 = {s: dict(loss=True, g=15.0, d100=0.9, motion=1.5) for s in range(10)}
        # case 1: X が A2 と同じ → PERSISTS・SIMILAR・PERSISTS・SIMILAR・EARLY・REVIVES、ΔE は引き分け 10 → P1 偽
        O = run_case(tmp, "same", dict(A2), A2, j6X=({s: 0.2 for s in range(10)}, {s: 0.5 for s in range(10)}, range(10)))
        R1 = O["arms"]["A4"]
        lab = [R1[k]["label"] for k in ("J1", "J2", "J3", "J4", "J5a", "J5b", "J6", "J6LN")]
        ok("b/case same: ラベル", lab == ["LOSS_PERSISTS", "L0_GROWTH_SIMILAR", "L0_DEATH_PERSISTS", "MOTION_SIMILAR", "L1_EARLY_DEATH", "L1_REVIVES", "OTHER_TERM_FLIPS", "LN_HALVES"], lab)
        ok("b/case same: ΔE の引き分け 10 → 符号検定 n = 0・p = NaN、P1 は偽", R1["J1"]["dE_sign"]["n_tie"] == 10 and R1["J1"]["dE_sign"]["p"] is None
           and [x["outcome"] for x in O["predictions"] if x["pred"] == "P1"] == [False, False], R1["J1"]["dE_sign"])
        # case 2: 境の上: g 7.5（ρ2 = 0.5 → SIMILAR）、d 中央値 = A2/2 ちょうど（→ PERSISTS）、動き 0.5（ρ4 = 1/3 → SMALLER・P3 真）、
        #         J6 の他 unit 項 0.75 ちょうどが 7 seed（→ KEPT）、exact 0.60 ちょうどが 7 seed（→ LN_HALVES）、L1 早期の死 0.9 ちょうど（→ EARLY）
        X = {s: dict(loss=(s < 8), g=7.5, d100=0.45, motion=0.5, l1_early=0.9, Rk=(50 if s < 8 else None)) for s in range(10)}
        oth = {s: (0.75 if s < 7 else 0.7499) for s in range(10)}; ex = {s: (0.60 if s < 7 else 0.6001) for s in range(10)}
        O = run_case(tmp, "edge_hi", X, A2, j6X=(oth, ex, range(10)))
        R2 = O["arms"]["A4"]; pr = {(x["pred"], x["who"]): x["outcome"] for x in O["predictions"]}
        lab = [R2[k]["label"] for k in ("J1", "J2", "J3", "J4", "J5a", "J5b", "J6", "J6LN")]
        ok("b/case edge_hi: J1 8/10 → PERSISTS、ρ2 = 0.5 → SIMILAR、d = 半分ちょうど → PERSISTS、ρ4 = 1/3 → SMALLER、L1 0.9 ちょうど → EARLY、8/10 戻る → REVIVES、7/10 → KEPT・LN_HALVES",
           lab == ["LOSS_PERSISTS", "L0_GROWTH_SIMILAR", "L0_DEATH_PERSISTS", "MOTION_SMALLER", "L1_EARLY_DEATH", "L1_REVIVES", "OTHER_TERM_KEPT", "LN_HALVES"]
           and abs(R2["J2"]["rho2"] - 0.5) < 1e-12 and R2["J4"]["rho4"] == 1 / 3, (lab, R2["J2"]["rho2"], R2["J4"]["rho4"], R2["J3"]["d"], R2["J3"]["half_A2"]))
        ok("b/case edge_hi: P3（ρ4 = 1/3 ちょうど）は真、P6 真、P6-LN 真、P2 偽", pr[("P3", "Fable")] is True and pr[("P6", "Fable")] is True and pr[("P6-LN", "Fable")] is True and pr[("P2", "Fable")] is False, pr)
        # case 3: 境の下: J1 7/10 → MIXED、J6 6/10 → FLIPS、J5b 7/10 戻る（m = 10）→ MIXED、ρ2 < 1/2、d < 半分、ρ4 > 2
        X = {s: dict(loss=(s < 7), g=7.4, d100=0.44, motion=3.01, Rk=(50 if s < 7 else None)) for s in range(10)}
        oth = {s: (0.75 if s < 6 else 0.1) for s in range(10)}; ex = {s: (0.60 if s < 6 else 0.7) for s in range(10)}
        R3 = run_case(tmp, "edge_lo", X, A2, j6X=(oth, ex, range(10)))["arms"]["A4"]
        lab = [R3[k]["label"] for k in ("J1", "J2", "J3", "J4", "J5b", "J6", "J6LN")]
        ok("b/case edge_lo: 7/10 → MIXED、ρ2 < 1/2 → SMALLER、d < 半分 → REDUCED、ρ4 > 2 → LARGER、7/10 → MIXED、6/10 → FLIPS・NOT_HALVES",
           lab == ["MIXED", "L0_GROWTH_SMALLER", "L0_DEATH_REDUCED", "MOTION_LARGER", "MIXED", "OTHER_TERM_FLIPS", "LN_NOT_HALVES"], lab)
        # case 4: 喪失なし 2/10 → GONE、3/10 → MIXED。L1 早期の死なし（0.5）→ NO_EARLY_DEATH、J5b INSUFFICIENT（m = 0）
        X = {s: dict(loss=(s < 2), l1_early=0.5) for s in range(10)}
        R4 = run_case(tmp, "gone", X, A2)["arms"]["A4"]
        X = {s: dict(loss=(s < 3)) for s in range(10)}
        R4b = run_case(tmp, "gone3", X, A2)["arms"]["A4"]
        ok("b/case gone: 2/10 → LOSS_GONE、3/10 → MIXED、早期の死 0 → NO_EARLY_DEATH・J5b INSUFFICIENT、J6 の入力なし → INSUFFICIENT",
           [R4["J1"]["label"], R4b["J1"]["label"], R4["J5a"]["label"], R4["J5b"]["label"], R4["J6"]["label"]] == ["LOSS_GONE", "MIXED", "NO_EARLY_DEATH", "INSUFFICIENT", "INSUFFICIENT"],
           [R4["J1"]["label"], R4b["J1"]["label"], R4["J5a"]["label"], R4["J5b"]["label"], R4["J6"]["label"]])
        # case 5: J5b の早期に死んだ seed が 5（< 6）→ INSUFFICIENT、6 で 5 戻る → REVIVES
        X = {s: dict(l1_early=(0.95 if s < 5 else 0.5)) for s in range(10)}
        R5 = run_case(tmp, "j5b5", X, A2)["arms"]["A4"]
        X = {s: dict(l1_early=(0.95 if s < 6 else 0.5), Rk=(50 if s < 5 else None)) for s in range(10)}
        R5b = run_case(tmp, "j5b6", X, A2)["arms"]["A4"]
        ok("b/case J5b: 早期に死んだ seed 5 → INSUFFICIENT（J5a は MIXED）、6 中 5 戻る → REVIVES", [R5["J5a"]["label"], R5["J5b"]["label"], R5b["J5b"]["label"]] == ["MIXED", "INSUFFICIENT", "L1_REVIVES"],
           [R5["J5a"]["label"], R5["J5b"]["label"], R5b["J5b"]["label"], R5b["J5b"]["n_revive"], R5b["J5b"]["n_early"]])
        # case 6: 欠けた seed。n = 9 で 7 LOSS → MIXED（境 8）、n = 8 で 7 LOSS → PERSISTS（境 7）、n = 7 → INSUFFICIENT（全ラベル）
        X9 = {s: dict(loss=(s < 7)) for s in range(9)}; X8 = {s: dict(loss=(s < 7)) for s in range(8)}; X7 = {s: dict(loss=True) for s in range(7)}
        oth8 = {s: 0.8 for s in range(10)}; ex8 = {s: 0.5 for s in range(10)}
        R9 = run_case(tmp, "n9", X9, A2)["arms"]["A4"]
        R8 = run_case(tmp, "n8", X8, A2, j6X=({s: (0.8 if s < 6 else 0.1) for s in range(10)}, ex8, range(8)))["arms"]["A4"]
        O7 = run_case(tmp, "n7", X7, A2, j6X=(oth8, ex8, range(7)))
        R7 = O7["arms"]["A4"]
        ok("b/case 欠けた seed: n = 9・7 LOSS → MIXED、n = 8・7 LOSS → PERSISTS、n = 8 の J6 6 → KEPT（境 ceil(5.6) = 6）",
           [R9["J1"]["label"], R9["J1"]["thr_hi"], R8["J1"]["label"], R8["J1"]["thr_hi"], R8["J6"]["label"], R8["J6"]["thr"]] == ["MIXED", 8, "LOSS_PERSISTS", 7, "OTHER_TERM_KEPT", 6],
           [R9["J1"]["label"], R9["J1"]["thr_hi"], R8["J1"]["label"], R8["J1"]["thr_hi"], R8["J6"]["label"], R8["J6"]["thr"]])
        lab7 = [R7[k]["label"] for k in ("J1", "J2", "J3", "J4", "J5a", "J5b", "J6", "J6LN")]
        ok("b/case 欠けた seed: n = 7 → 全ラベル INSUFFICIENT、予測は採点しない", set(lab7) == {"INSUFFICIENT"} and not any(x["scored"] for x in O7["predictions"]), lab7)
        ok("b/case 欠けた seed: 欠けの理由を記録", set(R9["missing"]) == {"9"}, R9["missing"])
        # case 7: A2 の再生が無い → J4 INSUFFICIENT・P3 採点なし。再生の一部の課題に生きた unit が無い → 欠測で nanmedian
        R7b = run_case(tmp, "noreplay", {s: dict() for s in range(10)}, A2, replay=False)
        ok("b/case 再生なし: J4 INSUFFICIENT、P3 採点なし", R7b["arms"]["A4"]["J4"]["label"] == "INSUFFICIENT" and not any(x["scored"] for x in R7b["predictions"] if x["pred"] == "P3"))
        X = {s: dict(motion=1.5, nalive=0 if s == 0 else 512) for s in range(10)}
        R8b = run_case(tmp, "noalive", X, A2)["arms"]["A4"]
        ok("b/case 生きた unit なし: その seed は 29 課題が欠測で J4 に使えず（n = 9）、ラベルは SIMILAR", R8b["J4"]["n"] == 9 and R8b["J4"]["n_missing"]["0"] == 29 and R8b["J4"]["label"] == "MOTION_SIMILAR",
           (R8b["J4"]["n"], R8b["J4"]["n_missing"], R8b["J4"]["label"]))
        # 縮退した課題と S-entropy の記録（chain_stats.csv の列）
        d = tmp / "same/res/M2/A4/seed0"; rows = J.rd(d / "chain_stats.csv")
        for r in rows:
            if int(r["task"]) in (95, 96): r["distinct_ctx_500upd"] = "100"
            if int(r["task"]) == 7: r.update(true_ce_eval=str(H0 + 0.05), true_ce_eval_minus_H="0.05", ok_entropy="False")
        with open(d / "chain_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        cs = J.read_chain_stats(d); dg, col = J.degenerate_tasks(cs); E = J.entropy_records({0: cs})
        ok("b/縮退（≤ 100 で縮退、列 distinct_ctx_500upd）と S-entropy の外れの一覧（ok_entropy = False の連鎖、z = (CE − H)/SE）",
           dg == {95, 96} and col == "distinct_ctx_500upd" and E["n_chains"] == 100 and E["n_fail_old_rule"] == 1 and E["fails"][0]["task"] == 7 and abs(E["fails"][0]["z_token"] - 5.0) < 1e-12,
           (dg, col, E["n_fail_old_rule"], E["fails"][0]["z_token"] if E["fails"] else None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================ (c) compute_W・compute_J の実の出力の形
WCH = J.RES / "checks/W"


def part_c():
    for name in ("trial_A4/seed0", "trial_S2/seed0", "srec_with"):
        d = WCH / name
        mo = J.read_motion(d / "within_motion.npz"); U = np.load(d / "unit_stats.npz")
        same = all(np.array_equal(mo[t]["alive10k"], ~U["chain_dead"][t - 1].astype(bool)) for t in mo)
        fin = all(np.isfinite(mo[t][q]).all() for t in mo for q in J.MOTION_Q)
        n5 = {t: int((mo[t]["alive5k"] & mo[t]["alive10k"]).sum()) for t in mo}
        ok(f"c/W {name}: within_motion.npz を読める（課題 {sorted(mo)}、量 5 つ有限）、alive[..., u_end] が unit_stats.npz の chain_dead の否定とビットで同じ（向きの確かめ）",
           len(mo) >= 2 and fin and same, dict(tasks=sorted(mo), alive_both=n5, net5k_L0_med={t: J.j4_seed(mo, 0, "net5k", [t])[t] for t in mo}))
        cs = J.read_chain_stats(d); dg, col = J.degenerate_tasks(cs); E = J.entropy_records({0: cs})
        ok(f"c/W {name}: chain_stats.csv の縮退の列と S-entropy の列を読める", col == "distinct_ctx_500upd" and E["n_chains"] == len(cs),
           dict(degenerate=sorted(dg), ctx={t: cs[t]["distinct_ctx_500upd"] for t in cs}, entropy_fail=[(r["task"], round(r["z_token"], 2)) for r in E["fails"]]))
    on = np.load(WCH / "trial_S2/seed0/online.npz"); pt = J.rd(WCH / "trial_S2/seed0/per_task.csv")
    oa = [float(r["online_all"]) for r in pt]; H = [float(r["H"]) for r in pt]
    ok("c/W trial_S2: loss ≠ aux、per_task.csv の online_all = mean(loss) − H（J1 は loss を使う）",
       not np.array_equal(on["loss"], on["aux"]) and max(abs(oa[i] - (on["loss"][i].astype(np.float64).mean() - H[i])) for i in range(len(pt))) < 1e-6,
       [(oa[i], on["loss"][i].astype(np.float64).mean() - H[i], on["aux"][i].astype(np.float64).mean() - H[i]) for i in range(len(pt))])
    mo, info = J.read_replay(WCH / "replay_try/seed0")
    ok("c/W replay_try/seed0: replay_task<t>.npz・replay_compare.csv・replay_provenance.json を読める", mo is not None and info["complete"] and sorted(mo) == [2, 42],
       dict(info=info, net5k_L0={t: J.j4_seed(mo, 0, "net5k", [t])[t] for t in mo}, net100_L0={t: J.j4_seed(mo, 0, "net100", [t], "end")[t] for t in mo}))
    ref = J.read_a2_net100_ref()
    ok("c/net100 の登録前の参照を a2_j4_fallback.txt から読む（L0 1.663 [1.370, 1.779]・全行 1.165 [1.085, 1.779]）",
       ref and ref["L0"]["alive"] == dict(median=1.663, min=1.370, max=1.779) and ref["L0"]["allrows"] == dict(median=1.165, min=1.085, max=1.779), ref)
    Jd = J.read_j6(J.RES / "checks/J6/A2_full", "A2"); JL = J.read_j6(J.LOADSIGN, "C3")
    a = {s: J.j6_seed(Jd, s)["late"] for s in range(10)}; b = {s: J.j6_seed(JL, s)["late"] for s in range(10)}
    k = (sum(a[s]["s_original_other"] >= 0.75 for s in a), sum(a[s]["s_original_exact_K"] <= 0.60 for s in a))
    dmax = {c: max(abs(a[s][c] - b[s][c]) for s in a) for c in J.J6_COLS}
    ok("c/J compute_J の A2_full（arm A2）を読める。後期の J6・J6-LN の数（loadsign は 0/10・10/10）と、loadsign との窓の中央値の差（記録）", k == (0, 10), dict(counts=k, late_window_absdiff_max=dmax))
    # 通しの試し: 合成の 100 課題の記録に、W の試走の within_motion・chain_stats・logit_stats を重ね、再生は replay_try、J6 は A2_full
    tmp = Path(tempfile.mkdtemp(prefix="cjW_", dir=str(OUT)))
    try:
        res = tmp / "res"; a2 = tmp / "A2"
        for s in range(10): gen_seed(a2 / f"seed{s}")
        for arm, src in (("A4", "trial_A4/seed0"), ("S2", "trial_S2/seed0")):
            for s in range(10):
                d = res / "M2" / arm / f"seed{s}"; gen_seed(d)
                for f in ("within_motion.npz", "chain_stats.csv", "logit_stats.csv"): shutil.copy(WCH / src / f, d / f)
        for s in range(10): (res / "replay_A2").mkdir(parents=True, exist_ok=True); (res / "replay_A2" / f"seed{s}").symlink_to(WCH / "replay_try/seed0")
        (tmp / "j6").mkdir(); (tmp / "j6" / "A4").symlink_to(J.RES / "checks/J6/A2_full")
        J.main(["--out", str(tmp / "out"), "--res", str(res), "--arms", "A4", "S2", "--a2_root", str(a2), "--replay_root", str(res / "replay_A2"),
                "--j6_root", str(tmp / "j6"), "--j6_names", "A4=A2", "--no_ckpt"])
        O = json.load(open(tmp / "out/judge.json")); R = O["arms"]["A4"]
        md = (tmp / "out/judge.md").read_text()
        ok("c/通し: W の形の within_motion（課題 1〜3）と replay（課題 2・42）で J4 が出る、J6 は A2_full、S-entropy と net100 の 2 つの値の行が judge.md に出る",
           R["J4"]["n"] == 10 and R["J4"]["n_A2"] == 10 and R["J6"]["n"] == 10 and "S-entropy の記録" in md and "byakko での再生" in md and "登録前の参照" in md
           and O["A2_replay_info"]["0"]["n_tasks"] == 2 and R["degenerate"]["col"] == "distinct_ctx_500upd",
           dict(J4=(R["J4"]["label"], R["J4"]["rho4"], R["J4"]["n_missing"]["0"]), J6=(R["J6"]["label"], R["J6"]["n_kept"]), entropy=(R["entropy"]["n_chains"], R["entropy"]["n_fail_old_rule"]),
                S2_J1=O["arms"]["S2"]["J1"]["label"]))
        shutil.copy(tmp / "out/judge.md", OUT / "smoke_W_format_judge.md")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


part_a()
part_b()
part_c()
n_ok = sum(r["ok"] for r in RES)
json.dump(dict(n=len(RES), n_ok=n_ok, results=RES), open(OUT / "check_judge.json", "w"), ensure_ascii=False, indent=1)
lines = ["# check_judge（judge.py の検査）", "", f"- {n_ok}/{len(RES)} 通過。judge.py sha256 {J.sha256(J.__file__)}", "", "| 検査 | 合否 | 詳細 |", "| --- | --- | --- |"]
lines += [f"| {r['name']} | {'通過' if r['ok'] else '**不合格**'} | {r['detail'][:400].replace('|', '/')} |" for r in RES]
(OUT / "check_judge.md").write_text("\n".join(lines) + "\n")
print(f"{n_ok}/{len(RES)} 通過")
for r in RES:
    if not r["ok"]: print("不合格:", r["name"], r["detail"][:600])
