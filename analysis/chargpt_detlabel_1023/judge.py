#!/usr/bin/env python3
"""chargpt_detlabel_1023 の判定（spec_chargpt_detlabel_1023.md の §7（J1〜J6・J6-LN・J1b・記録）・§7.0 の集約・§3 の予測の採点・§12）。

  $CG/.venv/bin/python $DL/analysis/chargpt_detlabel_1023/judge.py                 # 既定の置き場を読み judge.md・judge.json を書く
  ... judge.py --out <dir> [--arms A4 A3 S2] [--res ..] [--a2_root ..] [--replay_root ..] [--j6_root ..] [--no_ckpt]

- 判定の閾値・窓・規則は spec のまま（行番号は各関数の docstring）。予測（§3 の Fable と Opus 5.5 の欄、§12.1 の 4）は A4 について採点だけする。
- 対は同じ seed の A2（§7.0、spec:287）。A4 は主、A3 は記録、S2 は §12.1 の 1 で判定（並べて報告。採点しない）。
- J4 の主は net5k（§12.1 の 2、spec:454）。A2 の net5k は再生の出力（§6.4）を read_motion() 1 つで読む。net100 は記録（§12.3、spec:470）。
- 欠けた seed（§7.0、spec:290）: 使える seed n が 10 未満なら境の k/10 を ceil(k·n/10) に読み替え、n < 8 なら INSUFFICIENT。
- 学習・GPU は無い。torch は記録（課題末どうしの 1 − cos、§7.2 の記録）のときだけ import する（--no_ckpt で外す）。
- 写したコード:
  * phases(): $CG/analysis/chargpt_cl_1019_quicklook/three_phase.py（branch claude/chargpt_cl_1019 033679e0、blob b2306485）の phases() をそのまま。
  * cond1_seed(): 同 judge_m2.py（blob aade7511）の learning() と条件 1（C3 の分岐、行 23〜33・63〜66）。δ = 3 × 0.0084。
  * cond2_seed(): 同 judge_m2.py の条件 2（行 71〜83、C3 の分岐）。§7.7 の記録だけ。
  * sign_test_p(): analysis/chargpt_seed10_1022/judge.py の sign_test_p()。
  * T015 の「判定できない相」と J1 の LOSS: analysis/chargpt_detlabel_1023/prereg/a2_table.py の resolved・j1 の式。
- 読みの選択（spec の文だけで決まらない所）は READINGS に書き、judge.md の「読みの選択」に出す。
"""
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DL = HERE.parents[1]
CG = Path("/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019")
RES = DL / "results/chargpt_detlabel_1023"
A2_ROOT = CG / "results/chargpt_cl_1019/M2/C3"
LOADSIGN = DL / "results/chargpt_loadsign_1023"
SPEC = DL / "specs/spec_chargpt_detlabel_1023.md"

SEEDS = list(range(10))
NAN = float("nan")
NT, NU = 100, 10000
DELTA = 3 * 0.0084                                            # 追補 6 の δ（judge_m2.py）
T_J4 = list(range(2, 11)) + list(range(41, 51)) + list(range(91, 101))   # spec:333
J4_WIN = {"early": list(range(2, 11)), "middle": list(range(41, 51)), "late": list(range(91, 101))}
J6_WIN = {"early": list(range(1, 10)), "middle": list(range(39, 51)), "late": list(range(89, 101))}   # spec:350・loadsign report.py
J4_WIN_NAMED = (("2-10", range(2, 11)), ("41-50", range(41, 51)), ("91-100", range(91, 101)))   # §2.2 の窓
CKPT_TASKS = list(range(1, 11)) + [25] + list(range(39, 51)) + list(range(89, 101))
DEGEN_CTX = 100                                               # spec:368（最初の 500 更新の異なる文脈が 100 以下）
J6_COLS = ["s_original_self", "s_original_other", "s_original_residual", "s_original_direct_K", "s_radial", "s_original_exact_K",
           "original_other_abs_share", "n_alive"]
MOTION_Q = ["net5k", "path5k", "net100", "rot5k", "norm5k"]

# ---------------------------------------------------------------- 予測（spec §3 の表 spec:100〜106・115〜119。書き換えない）
PRED = {
    "P1": dict(line=102, text="J1 が LOSS_PERSISTS、かつ ΔE が A4 < A2 の seed が 8/10 以上", Fable=0.55, Opus=0.35),
    "P2": dict(line=103, text="J2 が L0_GROWTH_SMALLER", Fable=0.60, Opus=0.35),
    "P3": dict(line=104, text="J4 の比 ρ4 ≤ 1/3（net5k、§12.1 の 2）", Fable=0.80, Opus=0.15),
    "P6": dict(line=105, text="J6 が OTHER_TERM_KEPT", Fable=0.40, Opus=None),
    "P6-LN": dict(line=106, text="J6-LN が LN_HALVES", Fable=0.70, Opus=None),
}

READINGS = [
    ("(1) 欠けた seed の境", "spec:290 の「k/10 の境を k × n/10 の切り上げに読み替える」を、以上の境（8/10・7/10）にも以下の境（2/10）にも同じく当てる（thr(k, n) = ceil(k·n/10)、整数演算）。n は判定に使える seed の数。ρ の比（J2・J4）と J3 の中央値の比較は、両方の腕で使える seed が 8 以上のときだけラベルを出す。"),
    ("(2) 使える seed", "腕の seed は、per_task.csv が 100 行・online.npz が [100, 10000]・fresh_online_t100.npz・fresh.csv（課題 10・100）・unit_stats.npz・layer_stats.csv（課題 100 の u = 10000）が揃うとき使える。J4 はさらに within_motion.npz（A2 は再生の出力）が読めて M_k が有限のとき、J6 は後期の 12 状態（課題末 89〜100）が全部あるとき使える。"),
    ("(3) 死の量", "J3・J5 の死は spec:324・343 のとおり layer_stats.csv（probe = uniform、u = 10000）の frac_p_lt005 を使う。登録前の a2_table.py は unit_stats.npz の uniform_q から (1 − q) < 0.005 を数えた。検査 (a) で両者の一致を確かめた。"),
    ("(4) J4 の生きた unit", "net5k の m_{k,t} は、(r2) の更新 5,000 と 10,000 の鎖 probe の印（within_motion.npz の alive[..., u_half]・alive[..., u_end]、True = z > 0 のトークンがある）が両方とも生きている unit（spec:332）。記録の net100 は §7.4 の代案の定義（課題末の鎖 probe、spec:339）に合わせ、更新 10,000 の印だけで決める（A4・A3・S2 は (r2) の alive10k、無ければ unit_stats.npz の chain_dead の否定。A2 の再生は再生の alive10k）。"),
    ("(5) J4 の欠けた課題", "T_J4 の課題が within_motion に無い・生きた unit が無い課題は欠測（NaN）にし、M_k は残りの課題の中央値（nanmedian）。欠測の数を記録に書く。29 課題すべてが欠測なら、その seed は J4 に使えない。"),
    ("(6) 縮退した課題（§7.7）", "chain_stats.csv の最初の 500 更新の異なる文脈の数（列名は CTX500_COLS のどれか）が 100 以下の課題。A2 は chain_stats.csv を持たないので縮退した課題 0 とする（登録前の計算 §4.3 で α = 0.01 は 0/30、最小 4,151）。縮退を除いた J1 は、早期・後期の窓の平均からその課題を除く（本体(100) と新品(100) は課題 100 のまま。課題 100 が縮退なら印を付ける）。窓の課題がすべて縮退なら、その seed はその J1 に使えない。縮退を除いた J4 は T_J4 からその課題を除く。"),
    ("(7) J5b の数", "J5b は、J5a で早期に死んだ seed の数 m を分母にした割合（spec:345）。m < 6 なら INSUFFICIENT。使える seed が 8 未満なら §7.0 により INSUFFICIENT（m の条件より先に見る）。"),
    ("(8) J6 の行", "J6 の入力（compute_J の出力、loadsign と同じ列名）で layer 列があれば layer = 1 の行だけを判定に使い、layer = 0 は L0 の記録にする。components_states.csv に layer 列が無ければ全行を L1 とみなす。腕の名は arm 列で引く（既定 A4・A3・S2。A2 の参照は loadsign の C3）。"),
    ("(9) Spearman（J5 の記録）", "chargpt_handover_1023 の P4 の記録の形（spec_chargpt_handover_1023.md:97）に従い、layer_stats.csv（chain probe、u = 10000）の Δ(1 − frac_p_lt005)_L1 と Δr_med_L0 の連続する課題末の 99 対で seed ごとの ρ を出す（同順位は平均順位、片方が定数なら未定義）。checkpoint のある 31 対に限った値も並べる。自前の forward を使う handover の主の数とは別物。"),
    ("(10) 予測の採点", "予測は真偽の 2 値。結果が INSUFFICIENT・未完のときは採点しない。p_of_outcome は結果が真なら p、偽なら 1 − p、Brier は (p − 1[真])²。"),
    ("(11) §7.7 の条件 3", "追補 6 の条件 3 は decomp_dmu.csv（別の解析の出力）を要るので、この judge.py では出さない（記録だけ。判定に入らない）。条件 2 と切替の押し s_{l,t} は unit_stats.npz から judge_m2.py の式で出す。"),
    ("(12) S2 の J1", "S2 の online.npz は loss がサンプルした目標の CE（A2 と同じ量）、aux が勾配に使った真の分布の CE（compute_W の報告 §2）。J1・J1b・ΔE は loss と per_task.csv の online_all（loss − H の平均。試走で確かめた）を使い、aux は使わない（§4.5 の記録の揃え方）。"),
    ("(13) net100 の A2 の値", "Issa の判断（2026-10-24）により 2 つを区別して並べる。(i) 登録前の参照 1.66 [1.37, 1.78]（white-san の保存状態 u100 から、出典 prereg/a2_j4_fallback.{py,txt}。judge.py は txt を読み sha256 を残す）。(ii) byakko での再生の net100（再生の出力の alive[..., u_end] で生きた unit）。(ii) は元の軌道の値と呼ばない（spec:395・469）。"),
    ("(14) S-entropy", "Issa の判断（2026-10-24）により止める検査から記録に替わった。chain_stats.csv の ok_entropy（旧規則）が False の連鎖を、3 通りの標準誤差（トークンごと true_ce_eval_se・系列ごと _se_seq・真の分散 _se_truevar）と CE − H、z とともに一覧にする。判定・連鎖・データは変えない。"),
]


# ---------------------------------------------------------------- 小道具
def sign_test_p(npos, nneg):
    """両側 exact 符号検定（二項 1/2）。0 の差は呼ぶ前に除く（spec:287）。"""
    n = npos + nneg
    if n == 0: return NAN
    k = min(npos, nneg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sign_summary(diffs):
    d = [float(x) for x in diffs if x is not None and np.isfinite(x)]
    npos, nneg, ntie = sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)
    return dict(n_pairs=len(d), n_pos=npos, n_neg=nneg, n_tie=ntie, p=sign_test_p(npos, nneg))


def thr(k, n):
    """§7.0（spec:290）: k/10 の境を ceil(k·n/10) に読み替える。"""
    return -(-k * n // 10)


def med(x):
    x = [float(v) for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else NAN


def fmean(x):
    x = [float(v) for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else NAN


def mrange(x):
    x = [float(v) for v in x if v is not None and np.isfinite(v)]
    return dict(median=float(np.median(x)), min=float(min(x)), max=float(max(x)), n=len(x)) if x else dict(median=NAN, min=NAN, max=NAN, n=0)


def rd(p): return list(csv.DictReader(open(p)))


def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return NAN


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def rankavg(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]): return NAN
    ra, rb = rankavg(a), rankavg(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- 3 相（three_phase.py blob b2306485 の phases() そのまま）
def phases(loss, H, n=10000):
    ex = np.asarray(loss, dtype=np.float64) - H
    ma = np.convolve(ex, np.ones(50) / 50, mode="valid")          # ma[k] = 更新 k+1〜k+50 の平均 → 更新 k+50 での後ろ向き移動平均
    hit = np.nonzero(ma <= 0.15)[0]
    t015 = float(hit[0] + 50) if len(hit) else float(n)
    return t015, float(ex[950:1050].mean()), float(ex[-1000:].mean())


# ---------------------------------------------------------------- 入力の読み口
def load_seed(d):
    """腕の 1 seed の学習コードの記録（§6.1）を読む。揃わなければ (None, 理由)。"""
    d = Path(d)
    need = ["per_task.csv", "online.npz", "fresh_online_t100.npz", "fresh.csv", "unit_stats.npz", "layer_stats.csv"]
    miss = [f for f in need if not (d / f).exists()]
    if miss: return None, f"無い: {','.join(miss)}"
    pt = rd(d / "per_task.csv")
    if len(pt) != NT: return None, f"per_task.csv が {len(pt)} 行"
    on = np.load(d / "online.npz")["loss"].astype(np.float64)
    if on.shape != (NT, NU): return None, f"online.npz の形 {on.shape}"
    fr_rows = rd(d / "fresh.csv"); fr = {int(r["task"]): float(r["online_all"]) for r in fr_rows}
    if 10 not in fr or 100 not in fr: return None, "fresh.csv に課題 10・100 が無い"
    U = np.load(d / "unit_stats.npz"); Ud = {k: U[k] for k in U.files}
    if Ud["rownorm"].shape[0] != NT: return None, "unit_stats.npz の課題数"
    LS = {}
    for r in rd(d / "layer_stats.csv"):
        if int(r["u"]) == NU: LS[(int(r["task"]), r["probe"], int(r["layer"]))] = r
    if (NT, "uniform", 0) not in LS: return None, "layer_stats.csv に課題 100 が無い"
    def arr(probe, col):
        return np.array([[fnum(LS[(t, probe, l)][col]) if (t, probe, l) in LS else NAN for l in (0, 1)] for t in range(1, NT + 1)])
    H = np.array([float(r["H"]) for r in pt])
    S = dict(dir=str(d), pt=pt, H=H, on=on, fr=fr, U=Ud,
             on_all=np.array([float(r["online_all"]) for r in pt]), end1=float(pt[0]["end_metric"]),
             fresh_loss100=np.load(d / "fresh_online_t100.npz")["loss"].astype(np.float64),
             dead_u=arr("uniform", "frac_p_lt005"), dead_c=arr("chain", "frac_p_lt005"),
             ge099_u=arr("uniform", "frac_p_ge099"), rmed_u=arr("uniform", "r_med"), rmed_c=arr("chain", "r_med"))
    S["P"] = np.array([phases(on[t], H[t]) for t in range(NT)])
    S["F100"] = np.array(phases(S["fresh_loss100"], H[NT - 1]))
    return S, None


CTX500_COLS = ["distinct_ctx_500upd"]          # compute_W の chain_stats.csv（src/chargpt_detlabel_1023.py の CHAIN_KEYS）
ENTROPY_COLS = ["H", "true_ce_eval", "true_ce_eval_minus_H", "true_ce_eval_se", "true_ce_eval_se_seq", "true_ce_eval_se_truevar",
                "ok_entropy", "ok_entropy_seq_or_truevar", "pi_src"]


def read_chain_stats(d):
    """(r3) chain_stats.csv（compute_W の CHAIN_KEYS の列）。返り値 {課題: 行の dict}。無ければ None。"""
    p = Path(d) / "chain_stats.csv"
    if not p.exists(): return None
    return {int(r["task"]): r for r in rd(p)}


def degenerate_tasks(cs):
    """§7.7（spec:368）: 最初の 500 更新の異なる文脈（distinct_ctx_500upd）が 100 以下の課題の集合。chain_stats が無い（A2）なら空集合（読みの選択 6）。"""
    if cs is None: return set(), None
    col = next((c for c in CTX500_COLS if c in next(iter(cs.values()))), None)
    if col is None: return None, None
    return {t for t, r in cs.items() if fnum(r[col]) <= DEGEN_CTX}, col


def entropy_records(cs_by_seed):
    """Issa の判断（2026-10-24）: S-entropy は止める検査から記録に替わる。chain_stats.csv の 3 通りの標準誤差・CE − H と、
    旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差 = ok_entropy）で外れた連鎖の一覧。外れた連鎖を除かない・引き直さない。"""
    rows, n_all = [], 0
    for s, cs in sorted(cs_by_seed.items()):
        if cs is None: continue
        for t, r in sorted(cs.items()):
            n_all += 1
            if str(r.get("ok_entropy")) == "True": continue
            e = {c: (r.get(c) if c in ("ok_entropy", "ok_entropy_seq_or_truevar", "pi_src") else fnum(r.get(c))) for c in ENTROPY_COLS}
            e.update(seed=s, task=t)
            for k, c in (("z_token", "true_ce_eval_se"), ("z_seq", "true_ce_eval_se_seq"), ("z_truevar", "true_ce_eval_se_truevar")):
                e[k] = e["true_ce_eval_minus_H"] / e[c] if e[c] and np.isfinite(e[c]) and e[c] > 0 else NAN
            rows.append(e)
    return dict(n_chains=n_all, n_fail_old_rule=len(rows), n_fail_alt=sum(r["ok_entropy_seq_or_truevar"] != "True" for r in rows), fails=rows)


def _motion_entry(motion, alive, qn, ap):
    """compute_W の形の 1 課題: motion [層, unit, 量]（量の名 qn）、alive [層, unit, 2]（点の名 ap = u_half・u_end、True = 生きている）。"""
    e = {q: np.asarray(motion[..., qn.index(q)], np.float64) for q in MOTION_Q if q in qn}
    e["alive5k"] = np.asarray(alive[..., ap.index("u_half")], bool); e["alive10k"] = np.asarray(alive[..., ap.index("u_end")], bool)
    return e


def read_motion(path):
    """(r1)(r2) within_motion.npz（compute_W の save_records の形）の読み口。
    鍵: motion float32 [課題, 層, unit, 量]・quantities（net5k・path5k・net100・rot5k・norm5k）・alive bool [課題, 層, unit, 2]・
    alive_points（u_half = 更新 n/2 = 5,000、u_end = 10,000）・have bool [課題]（記録のある課題。課題 = 添字 + 1）。
    alive は「その課題の鎖 probe で z > 0 のトークンが 1 つでもある = True」（unit_stats.npz の chain_dead と逆向き）。
    返り値 {課題: {量: [層, unit] float64, alive5k, alive10k}}。無ければ None。"""
    p = Path(path)
    if not p.exists(): return None
    z = np.load(p, allow_pickle=False)
    qn = [str(q) for q in z["quantities"]]; ap = [str(a) for a in z["alive_points"]]
    if "net5k" not in qn: raise ValueError(f"{p}: net5k が無い（quantities {qn}）")
    have = z["have"].astype(bool) if "have" in z.files else np.ones(z["motion"].shape[0], bool)
    return {i + 1: _motion_entry(z["motion"][i], z["alive"][i], qn, ap) for i in np.nonzero(have)[0]}


def read_replay(seed_dir):
    """A2 の再生の出力（§6.4、compute_W の replay サブコマンド。replay_A2/seed<k>/）の読み口。
    replay_task<t>.npz（motion [層, unit, 量]・alive [層, unit, 2]・online [10,000]・quantities・alive_points）、
    replay_compare.csv（課題ごとの A2 の online.npz・task<t>.pt との差）、replay_provenance.json（complete）。
    返り値 (motion の dict（read_motion と同じ形）, 付帯の dict) か (None, 理由)。complete でなくても読める課題は読む（付帯に印）。"""
    d = Path(seed_dir)
    if not d.exists(): return None, "置き場が無い"
    files = sorted(d.glob("replay_task*.npz"), key=lambda p: int(p.stem[len("replay_task"):]))
    if not files: return None, "replay_task*.npz が無い"
    mo = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        mo[int(f.stem[len("replay_task"):])] = _motion_entry(z["motion"], z["alive"], [str(q) for q in z["quantities"]], [str(a) for a in z["alive_points"]])
    prov = json.load(open(d / "replay_provenance.json")) if (d / "replay_provenance.json").exists() else {}
    cmp_rows = rd(d / "replay_compare.csv") if (d / "replay_compare.csv").exists() else []
    info = dict(complete=bool(prov.get("complete", False)), n_tasks=len(mo), tasks_missing=[t for t in T_J4 if t not in mo],
                online_bitwise=sum(r.get("online_bitwise") == "True" for r in cmp_rows), state_bitwise=sum(r.get("state_bitwise") == "True" for r in cmp_rows),
                online_absdiff_max=max([fnum(r["online_absdiff_max"]) for r in cmp_rows] or [NAN]),
                online_absdiff_mean_med=med([fnum(r["online_absdiff_mean"]) for r in cmp_rows]),
                state_reldiff_max=max([fnum(r["state_reldiff_max"]) for r in cmp_rows] or [NAN]), n_compare=len(cmp_rows))
    return mo, info


def replay_seed_dir(replay_root, seed):
    return Path(replay_root) / f"seed{seed}"


def read_a2_net100_ref(path=None):
    """§7.4・§12.3 の A2 の net100 の参照値（登録前、white-san の保存状態 u100 から。出典 prereg/a2_j4_fallback.{py,txt}）。
    この台（byakko）の再生の net100 とは別のものとして並べる（Issa の判断 2026-10-24）。"""
    import re
    p = Path(path) if path else RES / "prereg/a2_j4_fallback.txt"
    if not p.exists(): return None
    out = dict(source=str(p.relative_to(DL)) if str(p).startswith(str(DL)) else str(p), sha256=sha256(p), script="analysis/chargpt_detlabel_1023/prereg/a2_j4_fallback.py",
               origin="white-san の A2 の保存状態（u100_task<t>.pt と task<t>.pt）から登録前に出した値。生きた unit は課題末の鎖 probe")
    for line in open(p):
        m = re.match(r"L(\d) net100 生きた unit M_k: median ([\d.]+) \[([\d.]+), ([\d.]+)\] \| 全行 ([\d.]+) \[([\d.]+), ([\d.]+)\]", line.strip())
        if m:
            v = [float(x) for x in m.groups()[1:]]
            out[f"L{m.group(1)}"] = dict(alive=dict(median=v[0], min=v[1], max=v[2]), allrows=dict(median=v[3], min=v[4], max=v[5]))
    return out


def read_j6(root, arm, folder=None):
    """J6 の入力（compute_J の出力、または loadsign の結果。states.csv・components_states.csv、列名は loadsign と同じ）。
    返り値 {"L1": {seed: {task: {列: 値}}}, "L0": {...}}。無ければ None。"""
    root = Path(root)
    for sub in (folder, arm):                                           # compute_J の置き場 j6/<腕>/
        if sub and (root / sub / "states.csv").exists(): root = root / sub; break
    a, b = root / "states.csv", root / "components_states.csv"
    if not a.exists() or not b.exists(): return None
    out = {"L1": {}, "L0": {}}
    for f in (a, b):
        for r in rd(f):
            if r.get("arm") != arm: continue
            lay = "L0" if r.get("layer") not in (None, "", "1") and int(float(r["layer"])) == 0 else "L1"
            e = out[lay].setdefault(int(r["seed"]), {}).setdefault(int(r["task"]), {})
            for c in J6_COLS:
                if c in r and c not in e: e[c] = fnum(r[c])
    return out


# ---------------------------------------------------------------- seed ごとの量
def j1_seed(S, degen=frozenset()):
    """J1（spec:293〜312）。degen の課題を早期・後期の窓から除く（§7.7 の記録、読みの選択 6）。"""
    P = S["P"]
    ew = [t for t in range(2, 11) if t not in degen]; lw = [t for t in range(91, 101) if t not in degen]
    if not ew or not lw: return None
    early = P[[t - 1 for t in ew]].mean(0); late = P[[t - 1 for t in lw]].mean(0); cont100 = P[NT - 1]; F100 = S["F100"]
    resolved = [not (i == 0 and ((early[0] == 50 and late[0] == 50) or (early[0] == NU and late[0] == NU))) for i in range(3)]
    phase_loss = [bool(late[i] > early[i] and cont100[i] > F100[i]) for i in range(3)]
    learned = bool(S["end1"] <= 0.1)
    loss = bool(learned and sum(resolved) >= 2 and all(pl for pl, rs in zip(phase_loss, resolved) if rs))
    e110 = [t for t in range(1, 11) if t not in degen]
    return dict(end1=S["end1"], learned=learned, early=early.tolist(), late=late.tolist(), cont100=cont100.tolist(), fresh100=F100.tolist(),
                resolved=resolved, phase_loss=phase_loss, loss=loss, dE=float(late[1] - early[1]),
                dE110=float(late[1] - P[[t - 1 for t in e110], 1].mean()),
                E1000_dec=[float(P[10 * i:10 * i + 10, 1].mean()) for i in range(10)], task100_degenerate=bool(NT in degen),
                n_early=len(ew), n_late=len(lw))


def cond1_seed(S):
    """J1b（spec:310）: judge_m2.py の条件 1（C3 の分岐）そのもの。"""
    on = S["on_all"]; fr = S["fr"]; head = (S["on"][:, :100] - S["H"][:, None]).mean(1)
    worse = on[90:100].mean() - on[1:10].mean(); gap_shift = (on[99] - fr[100]) - (on[9] - fr[10])
    return dict(worse=float(worse), gap_shift=float(gap_shift), head_early=float(head[1:10].mean()), head_late=float(head[90:100].mean()),
                pass_=bool(S["end1"] <= 0.1 and worse >= DELTA and gap_shift >= DELTA and head[90:100].mean() > head[1:10].mean()))


def cond2_seed(S, top=1):
    """§7.7 の記録: judge_m2.py の条件 2（C3 の分岐）と切替の押し s_{l,t}。"""
    U = S["U"]; n = NT
    s = np.median((U["chain_mu_u100"].astype(float) - U["chain_mu_u0"].astype(float)) / (U["chain_sd_u0"].astype(float) + 1e-12), axis=2)
    pu = 1 - U["uniform_q"].astype(float); Rt = ((pu < 0.005) | (pu >= 0.99)).mean(2)
    early = S["on_all"][1:10].mean()
    onset_resp = next((t for t in range(2, n + 1) if Rt[t - 1, top] - Rt[0, top] >= 0.10), None)
    onset_loss = next((t for t in range(11, n + 1) if S["on_all"][t - 5:t].mean() - early >= DELTA), None)
    lead_ok = None if onset_loss is None else (onset_resp is not None and onset_resp <= onset_loss + 5)
    c2 = dict(push_mean_2_50=float(s[1:50, top].mean()), negfrac_2_50=float((s[1:50, top] < 0).mean()),
              push_mean_51_100=float(s[50:100, top].mean()), negfrac_51_100=float((s[50:100, top] < 0).mean()),
              push_by_layer_2_50=[float(s[1:50, l].mean()) for l in range(2)], onset_resp=onset_resp, onset_loss=onset_loss, lead_ok=lead_ok)
    c2["pass_"] = bool(c2["push_mean_2_50"] <= -0.206 and c2["negfrac_2_50"] >= 0.75 and (lead_ok is True))
    return c2


def j2_seed(S):
    """J2（spec:314〜320）。g = 課題 100 末 / 課題 1 末の up 行ノルムの中央値（unit_stats.npz の rownorm）。"""
    rn = S["U"]["rownorm"].astype(np.float64); alive = ~S["U"]["chain_dead"].astype(bool)
    def g(l, t): return float(np.median(rn[t - 1, l]) / np.median(rn[0, l]))
    def ga(l):
        a1, a100 = alive[0, l], alive[NT - 1, l]
        return float(np.median(rn[NT - 1, l][a100]) / np.median(rn[0, l][a1])) if a1.any() and a100.any() else NAN
    return dict(g0=g(0, 100), g1=g(1, 100), g0_t10=g(0, 10), g0_t50=g(0, 50), g1_t10=g(1, 10), g1_t50=g(1, 50),
                g0_alive=ga(0), g1_alive=ga(1), rn0=[float(np.median(rn[t - 1, 0])) for t in (1, 10, 50, 100)])


def j3_seed(S):
    """J3（spec:322〜327）。d = 課題 100 末の L0 の死（layer_stats.csv uniform u=10000 layer 0 の frac_p_lt005）。"""
    du = S["dead_u"][:, 0]
    return dict(d=float(du[NT - 1]), d_91_100=float(du[90:100].mean()), d50=float(du[49]),
                onset50=next((t for t in range(1, NT + 1) if du[t - 1] >= 0.5), None),
                d_chain100=float(S["dead_c"][NT - 1, 0]), ge099_100=float(S["ge099_u"][NT - 1, 0]))


def j5_seed(S):
    """J5（spec:341〜346）。L1 の死 = layer_stats.csv uniform u=10000 layer 1 の frac_p_lt005。"""
    d1 = S["dead_u"][:, 1]
    Rk = next((t for t in range(11, NT + 1) if d1[t - 1] <= 0.5), None)
    al1 = 1 - S["dead_c"][:, 1]; r0 = S["rmed_c"][:, 0]
    da, dr = np.diff(al1), np.diff(r0)                       # 対 t → t+1（t = 1〜99）
    pairs31 = [t for t in range(1, NT) if t in CKPT_TASKS and t + 1 in CKPT_TASKS]
    idx = [t - 1 for t in pairs31]
    return dict(L1d10=float(d1[9]), early_dead=bool(d1[9] >= 0.9), Rk=Rk, L1d50=float(d1[49]), L1d100=float(d1[99]),
                spearman99=spearman(da, dr), spearman31=spearman(da[idx], dr[idx]))


def records_misc(S):
    """§7.7 の記録: 層ごとの位置 r の中央値（layer_stats.csv の r_med、u = 10000）。"""
    out = {}
    for probe, A in (("uniform", S["rmed_u"]), ("chain", S["rmed_c"])):
        for l in (0, 1):
            out[f"rmed_{probe}_L{l}"] = {w: float(np.nanmean(A[[t - 1 for t in ts], l])) for w, ts in
                                         (("1-10", range(1, 11)), ("41-50", range(41, 51)), ("91-100", range(91, 101)))}
    return out


def j4_seed(mo, layer=0, qty="net5k", tasks=T_J4, alive_rule="both"):
    """J4（spec:329〜338）: m_{k,t} = 課題 t の生きた unit の量の中央値（生きた unit が無い・課題が無いと欠測）、M_k = T_J4 の nanmedian。
    alive_rule: both = 更新 5,000 と 10,000 の両方（主）、end = 10,000 だけ（net100 の記録）、all = 全行。"""
    m = {}
    for t in tasks:
        e = mo.get(t) if mo is not None else None
        need = {"ratio_net_path": ("net5k", "path5k"), "rel_net": ("net5k", "norm5k")}.get(qty, (qty,))
        if e is None or any(q not in e for q in need): m[t] = NAN; continue
        if qty == "ratio_net_path": v = e["net5k"][layer] / np.where(e["path5k"][layer] > 0, e["path5k"][layer], NAN)
        elif qty == "rel_net": v = e["net5k"][layer] / np.where(e["norm5k"][layer] > 0, e["norm5k"][layer], NAN)
        else: v = e[qty][layer]
        if alive_rule == "all": sel = np.ones(v.shape, bool)
        elif alive_rule == "end": sel = e["alive10k"][layer] if e["alive10k"] is not None else e.get("alive_end_fallback", np.zeros(v.shape, bool))[layer]
        else:
            if e["alive5k"] is None or e["alive10k"] is None: m[t] = NAN; continue
            sel = e["alive5k"][layer] & e["alive10k"][layer]
        sel = sel & np.isfinite(v)
        m[t] = float(np.median(v[sel])) if sel.any() else NAN
    return m


def M_of(m, tasks):
    v = [m.get(t, NAN) for t in tasks]
    return med(v), sum(not np.isfinite(x) for x in v)


def j6_seed(J, seed, layer="L1"):
    """J6（spec:348〜363）: 状態ごとの割合を seed 内で窓の中央値にする。後期 12 状態が揃わなければ late は None。"""
    if J is None or seed not in J[layer]: return None
    st = J[layer][seed]; out = {}
    for w, ts in J6_WIN.items():
        have = [t for t in ts if t in st]
        if w == "late" and len(have) < len(ts): out[w] = None; continue
        out[w] = {c: med([st[t].get(c, NAN) for t in have]) for c in J6_COLS} if have else None
    return out


# ---------------------------------------------------------------- ラベル（spec の規則そのまま）
def label_count(k, n, hi, lo, labs):
    """seed の数のラベル: k ≥ ceil(hi·n/10) → labs[0]、k ≤ ceil(lo·n/10) → labs[1]、それ以外 MIXED。n < 8 → INSUFFICIENT（spec:290）。"""
    if n < 8: return "INSUFFICIENT"
    if k >= thr(hi, n): return labs[0]
    if lo is not None and k <= thr(lo, n): return labs[1]
    return "MIXED" if lo is not None else labs[1]


def label_ratio(rho, nA, nB, small, large, similar):
    """spec:288: ρ < 1/2 → SMALLER、ρ > 2 → LARGER、その間 SIMILAR。"""
    if nA < 8 or nB < 8 or not np.isfinite(rho): return "INSUFFICIENT"
    return small if rho < 0.5 else (large if rho > 2 else similar)


def label_J5b(n_usable, m_early, k_rev):
    """spec:345: 早期に死んだ seed のうち戻る割合 ≥ 0.8 → L1_REVIVES、≤ 0.2 → NO_REVIVAL、それ以外 MIXED。早期に死んだ seed < 6 → INSUFFICIENT。"""
    if n_usable < 8 or m_early < 6: return "INSUFFICIENT"
    f = k_rev / m_early
    return "L1_REVIVES" if f >= 0.8 else ("NO_REVIVAL" if f <= 0.2 else "MIXED")


# ---------------------------------------------------------------- 腕の判定
def arm_seeds(root, seeds=SEEDS):
    S, why = {}, {}
    for s in seeds:
        x, r = load_seed(Path(root) / f"seed{s}")
        if x is None: why[s] = r
        else: S[s] = x
    return S, why


def per_seed(S):
    out = {}
    for s, x in S.items():
        out[s] = dict(j1=j1_seed(x), j1b=cond1_seed(x), j2=j2_seed(x), j3=j3_seed(x), j5=j5_seed(x), cond2=cond2_seed(x), misc=records_misc(x))
    return out


def motion_block(MO, degen_by_seed=None):
    """J4 の seed ごとの量（主と記録）。MO = {seed: read_motion の返り値}。"""
    out = {}
    for s, mo in MO.items():
        if mo is None: continue
        tasks = [t for t in T_J4 if not (degen_by_seed and t in degen_by_seed.get(s, set()))]
        r = {}
        for l in (0, 1):
            m = j4_seed(mo, l, "net5k", tasks, "both"); M, nmiss = M_of(m, tasks)
            r[f"L{l}"] = dict(M=M, n_missing=nmiss, m={str(t): v for t, v in m.items()},
                              win={w: M_of(m, [t for t in ts if t in tasks])[0] for w, ts in J4_WIN.items()},
                              M_allrows=M_of(j4_seed(mo, l, "net5k", tasks, "all"), tasks)[0],
                              M_net_over_path=M_of(j4_seed(mo, l, "ratio_net_path", tasks, "both"), tasks)[0],
                              M_net_over_norm=M_of(j4_seed(mo, l, "rel_net", tasks, "both"), tasks)[0],
                              M_rot5k=M_of(j4_seed(mo, l, "rot5k", tasks, "both"), tasks)[0] if "rot5k" in next(iter(mo.values())) else NAN,
                              M_net100_end=M_of(j4_seed(mo, l, "net100", tasks, "end"), tasks)[0] if "net100" in next(iter(mo.values())) else NAN,
                              M_net100_allrows=M_of(j4_seed(mo, l, "net100", tasks, "all"), tasks)[0] if "net100" in next(iter(mo.values())) else NAN)
        out[s] = r
    return out


def load_motion_arm(root, S, seeds=SEEDS):
    MO = {}
    for s in seeds:
        mo = read_motion(Path(root) / f"seed{s}" / "within_motion.npz")
        if mo is not None and s in S:
            for t, e in mo.items():                                    # net100 の記録の印の予備（読みの選択 4）
                e["alive_end_fallback"] = ~S[s]["U"]["chain_dead"][t - 1].astype(bool)
        MO[s] = mo
    return MO


def load_motion_replay(replay_root, seeds=SEEDS):
    MO, INFO = {}, {}
    for s in seeds:
        mo, info = read_replay(replay_seed_dir(replay_root, s))
        MO[s] = mo; INFO[s] = info
    return MO, INFO


def cos_records(root, S, seeds=SEEDS):
    """§7.2 の記録: 課題末どうしの 1 − cos（L0、全行と生きた unit）。checkpoint が要る（torch）。"""
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(8)
    out = {}
    for s in seeds:
        ck = Path(root) / f"seed{s}" / "ckpts"
        if s not in S or not ck.exists(): continue
        W = {}
        for t in CKPT_TASKS:
            p = ck / f"task{t}.pt"
            if p.exists(): W[t] = torch.load(p, map_location="cpu", weights_only=True)["h.0.mlp.up.weight"].double().numpy()
        dead = S[s]["U"]["chain_dead"].astype(bool); r = {}
        for t in CKPT_TASKS:
            if t in W and t - 1 in W:
                a, b = W[t - 1], W[t]; c = 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
                both = ~dead[t - 1, 0] & ~dead[t - 2, 0]
                r[t] = (float(np.median(c)), float(np.median(c[both])) if both.any() else NAN)
        out[s] = {w: (med([r[t][0] for t in ts if t in r]), med([r[t][1] for t in ts if t in r]))
                  for w, ts in J4_WIN_NAMED}
    return out


def judge_pair(name, X, A2, MO_X, MO_A2, JX, JA2, degen):
    """§7 の J1〜J6 を腕 X と A2 の対に当てる。X・A2 は per_seed の返り値。"""
    R = dict(arm=name, usable=sorted(X), usable_A2=sorted(A2))
    n = len(X); pairs = sorted(set(X) & set(A2))
    # J1
    kL = sum(X[s]["j1"]["loss"] for s in X)
    R["J1"] = dict(label=label_count(kL, n, 8, 2, ("LOSS_PERSISTS", "LOSS_GONE")), n_loss=kL, n=n, thr_hi=thr(8, n), thr_lo=thr(2, n),
                   per_seed={s: X[s]["j1"] for s in X},
                   phase_counts=[sum(X[s]["j1"]["phase_loss"][i] for s in X) for i in range(3)],
                   n_learned=sum(X[s]["j1"]["learned"] for s in X),
                   dE=mrange([X[s]["j1"]["dE"] for s in X]), dE110=mrange([X[s]["j1"]["dE110"] for s in X]),
                   dE_sign=sign_summary([X[s]["j1"]["dE"] - A2[s]["j1"]["dE"] for s in pairs]),
                   n_dE_smaller=sum(X[s]["j1"]["dE"] < A2[s]["j1"]["dE"] for s in pairs), n_pairs=len(pairs),
                   dE_ratio_median=med([X[s]["j1"]["dE"] / A2[s]["j1"]["dE"] for s in pairs if A2[s]["j1"]["dE"] != 0]),
                   E1000_dec=[mrange([X[s]["j1"]["E1000_dec"][i] for s in X]) for i in range(10)])
    # 縮退を除いた J1・J4（§7.7）は judge_arm で出す
    # J1b
    k1b = sum(X[s]["j1b"]["pass_"] for s in X)
    R["J1b"] = dict(n_pass=k1b, n=n, per_seed={s: X[s]["j1b"] for s in X})
    # J2
    gX = [X[s]["j2"]["g0"] for s in X]; gA = [A2[s]["j2"]["g0"] for s in A2]
    rho2 = med(gX) / med(gA) if gA else NAN
    R["J2"] = dict(label=label_ratio(rho2, len(gX), len(gA), "L0_GROWTH_SMALLER", "L0_GROWTH_LARGER", "L0_GROWTH_SIMILAR"),
                   rho2=rho2, g=mrange(gX), g_A2=mrange(gA), sign=sign_summary([X[s]["j2"]["g0"] - A2[s]["j2"]["g0"] for s in pairs]),
                   n_smaller=sum(X[s]["j2"]["g0"] < A2[s]["j2"]["g0"] for s in pairs),
                   rec={k: mrange([X[s]["j2"][k] for s in X]) for k in ("g0_t10", "g0_t50", "g1", "g0_alive", "g1_alive")},
                   rec_A2={k: mrange([A2[s]["j2"][k] for s in A2]) for k in ("g0_t10", "g0_t50", "g1", "g0_alive", "g1_alive")})
    # J3
    dX = [X[s]["j3"]["d"] for s in X]; dA = [A2[s]["j3"]["d"] for s in A2]
    lab3 = "INSUFFICIENT" if len(dX) < 8 or len(dA) < 8 else ("L0_DEATH_REDUCED" if med(dX) < med(dA) / 2 else "L0_DEATH_PERSISTS")
    R["J3"] = dict(label=lab3, d=mrange(dX), d_A2=mrange(dA), half_A2=med(dA) / 2,
                   sign=sign_summary([X[s]["j3"]["d"] - A2[s]["j3"]["d"] for s in pairs]),
                   n_smaller=sum(X[s]["j3"]["d"] < A2[s]["j3"]["d"] for s in pairs), n_ge05=sum(x >= 0.5 for x in dX),
                   rec={k: mrange([X[s]["j3"][k] for s in X]) for k in ("d_91_100", "d50", "d_chain100", "ge099_100")},
                   onset50={s: X[s]["j3"]["onset50"] for s in X})
    # J4
    MX = motion_block(MO_X); MA = motion_block(MO_A2)
    def Ms(B, l, key="M"): return {s: B[s][f"L{l}"][key] for s in B if np.isfinite(B[s][f"L{l}"][key])}
    mX, mA = Ms(MX, 0), Ms(MA, 0)
    rho4 = med(list(mX.values())) / med(list(mA.values())) if mA and mX else NAN
    p4 = sorted(set(mX) & set(mA))
    R["J4"] = dict(label=label_ratio(rho4, len(mX), len(mA), "MOTION_SMALLER", "MOTION_LARGER", "MOTION_SIMILAR"), rho4=rho4,
                   M=mrange(list(mX.values())), M_A2=mrange(list(mA.values())), n=len(mX), n_A2=len(mA),
                   sign=sign_summary([mX[s] - mA[s] for s in p4]),
                   rho4_win={w: (med([MX[s]["L0"]["win"][w] for s in mX]) / med([MA[s]["L0"]["win"][w] for s in mA])) if mA and mX else NAN for w in J4_WIN},
                   rec={f"L{l}_{k}": (mrange(list(Ms(MX, l, k).values())), mrange(list(Ms(MA, l, k).values())))
                        for l in (0, 1) for k in ("M", "M_allrows", "M_net_over_path", "M_net_over_norm", "M_rot5k", "M_net100_end", "M_net100_allrows")},
                   n_missing={s: MX[s]["L0"]["n_missing"] for s in MX}, n_missing_A2={s: MA[s]["L0"]["n_missing"] for s in MA},
                   per_seed={s: MX[s] for s in MX}, per_seed_A2={s: MA[s] for s in MA})
    # J5
    early = [s for s in X if X[s]["j5"]["early_dead"]]; krev = sum(X[s]["j5"]["Rk"] is not None for s in early)
    R["J5a"] = dict(label=label_count(len(early), n, 8, 2, ("L1_EARLY_DEATH", "NO_EARLY_DEATH")), n_early=len(early), n=n)
    R["J5b"] = dict(label=label_J5b(n, len(early), krev), n_revive=krev, n_early=len(early),
                    Rk={s: X[s]["j5"]["Rk"] for s in X},
                    Rk_sign=sign_summary([(X[s]["j5"]["Rk"] or 101) - (A2[s]["j5"]["Rk"] or 101) for s in pairs]),
                    rec={k: mrange([X[s]["j5"][k] for s in X]) for k in ("L1d50", "L1d100", "spearman99", "spearman31")},
                    n_neg_spearman99=sum(X[s]["j5"]["spearman99"] < 0 for s in X if np.isfinite(X[s]["j5"]["spearman99"])),
                    n_neg_spearman31=sum(X[s]["j5"]["spearman31"] < 0 for s in X if np.isfinite(X[s]["j5"]["spearman31"])))
    # J6
    jX = {s: j6_seed(JX, s) for s in SEEDS}; jX = {s: v for s, v in jX.items() if v and v.get("late")}
    jA = {s: j6_seed(JA2, s) for s in SEEDS}; jA = {s: v for s, v in jA.items() if v and v.get("late")}
    n6 = len(jX)
    k6 = sum(jX[s]["late"]["s_original_other"] >= 0.75 for s in jX); k6ln = sum(jX[s]["late"]["s_original_exact_K"] <= 0.60 for s in jX)
    lab6 = "INSUFFICIENT" if n6 < 8 else ("OTHER_TERM_KEPT" if k6 >= thr(7, n6) else "OTHER_TERM_FLIPS")
    lab6ln = "INSUFFICIENT" if n6 < 8 else ("LN_HALVES" if k6ln >= thr(7, n6) else "LN_NOT_HALVES")
    p6 = sorted(set(jX) & set(jA))
    R["J6"] = dict(label=lab6, n_kept=k6, n=n6, thr=thr(7, n6) if n6 else None,
                   sign=sign_summary([jX[s]["late"]["s_original_other"] - jA[s]["late"]["s_original_other"] for s in p6]),
                   win={w: {c: mrange([jX[s][w][c] for s in jX if jX[s].get(w)]) for c in J6_COLS} for w in J6_WIN},
                   per_seed=jX, L0={s: j6_seed(JX, s, "L0") for s in SEEDS if JX and s in JX["L0"]})
    R["J6LN"] = dict(label=lab6ln, n_halves=k6ln, n=n6)
    return R, MX, MA


def judge_arm(name, root, A2S, A2P, MO_A2, JA2, j6_root, ckpt, j6_name=None):
    S, why = arm_seeds(root)
    X = per_seed(S)
    cs = {s: read_chain_stats(Path(root) / f"seed{s}") for s in S}
    dg, dcol = {}, None
    for s, c in cs.items():
        d, col = degenerate_tasks(c)
        if d is not None: dg[s] = d; dcol = dcol or col
    MO = load_motion_arm(root, S)
    J = read_j6(j6_root, j6_name or name, folder=name) if j6_root else None
    R, MX, MA = judge_pair(name, X, A2P, MO, MO_A2, J, JA2, dg)
    R["missing"] = {s: why[s] for s in why}
    # §7.7 の記録: 縮退した課題と、それを除いた J1・J4
    R["degenerate"] = dict(col=dcol, n_by_seed={s: len(v) for s, v in dg.items()}, chain_stats_seeds=sorted(s for s in cs if cs[s] is not None))
    if dg:
        j1d = {s: j1_seed(S[s], dg.get(s, set())) for s in S}; j1d = {s: v for s, v in j1d.items() if v}
        kd = sum(v["loss"] for v in j1d.values())
        R["degenerate"]["J1"] = dict(label=label_count(kd, len(j1d), 8, 2, ("LOSS_PERSISTS", "LOSS_GONE")), n_loss=kd, n=len(j1d))
        MXd = motion_block(MO, dg); mXd = [MXd[s]["L0"]["M"] for s in MXd if np.isfinite(MXd[s]["L0"]["M"])]
        mA = [MA[s]["L0"]["M"] for s in MA if np.isfinite(MA[s]["L0"]["M"])]
        rho = med(mXd) / med(mA) if mXd and mA else NAN
        R["degenerate"]["J4"] = dict(rho4=rho, label=label_ratio(rho, len(mXd), len(mA), "MOTION_SMALLER", "MOTION_LARGER", "MOTION_SIMILAR"))
    # §7.7 の記録: 条件 2・位置 r・連鎖と logit の記録の推移
    R["cond2"] = dict(n_pass=sum(X[s]["cond2"]["pass_"] for s in X), per_seed={s: X[s]["cond2"] for s in X})
    R["rmed"] = {k: {w: mrange([X[s]["misc"][k][w] for s in X]) for w in ("1-10", "41-50", "91-100")} for k in (X[next(iter(X))]["misc"] if X else {})}
    R["chain_logit"] = trajectory_records(root, S)
    R["entropy"] = entropy_records(cs)                                   # Issa の判断 2026-10-24（記録）
    R["cos"] = cos_records(root, S) if ckpt else None
    return R, S, X


def trajectory_records(root, S):
    """§7.7 の記録: chain_stats.csv (r3)・logit_stats.csv (r5) の数値の列を 10 課題ごとの seed 間の中央値にする（列名は問わない）。"""
    out = {}
    for f in ("chain_stats.csv", "logit_stats.csv"):
        rows = {s: rd(Path(root) / f"seed{s}" / f) for s in S if (Path(root) / f"seed{s}" / f).exists()}
        if not rows: continue
        cols = [c for c in next(iter(rows.values()))[0] if c != "task" and np.isfinite(fnum(next(iter(rows.values()))[0][c]))]
        out[f] = {c: [med([fmean([fnum(r[c]) for r in rr if 10 * i < int(r["task"]) <= 10 * i + 10]) for rr in rows.values()])
                      for i in range(10)] for c in cols}
    return out


# ---------------------------------------------------------------- 予測の採点（§3、A4 だけ）
def score_predictions(R):
    o = {}
    j1 = R["J1"]
    o["P1"] = None if j1["label"] == "INSUFFICIENT" or j1["n_pairs"] < 8 else bool(j1["label"] == "LOSS_PERSISTS" and j1["n_dE_smaller"] >= thr(8, j1["n_pairs"]))
    o["P2"] = None if R["J2"]["label"] == "INSUFFICIENT" else R["J2"]["label"] == "L0_GROWTH_SMALLER"
    o["P3"] = None if R["J4"]["label"] == "INSUFFICIENT" else bool(R["J4"]["rho4"] <= 1 / 3)
    o["P6"] = None if R["J6"]["label"] == "INSUFFICIENT" else R["J6"]["label"] == "OTHER_TERM_KEPT"
    o["P6-LN"] = None if R["J6LN"]["label"] == "INSUFFICIENT" else R["J6LN"]["label"] == "LN_HALVES"
    out = []
    for k, P in PRED.items():
        for who in ("Fable", "Opus"):
            p = P[who]
            if p is None: continue
            if o[k] is None:
                out.append(dict(pred=k, who=who, spec_line=P["line"], text=P["text"], p=p, scored=False, outcome="INSUFFICIENT・未完")); continue
            out.append(dict(pred=k, who=who, spec_line=P["line"], text=P["text"], p=p, scored=True, outcome=o[k],
                            p_of_outcome=p if o[k] else 1 - p, brier=(p - o[k]) ** 2))
    return out


# ---------------------------------------------------------------- 書き出し
def f3(x, nd=3):
    if x is None: return "—"
    if isinstance(x, (bool, np.bool_)): return str(bool(x))
    if isinstance(x, (int, np.integer)): return str(int(x))
    try: x = float(x)
    except (TypeError, ValueError): return str(x)
    return "NaN" if not np.isfinite(x) else f"{x:.{nd}g}"


def fr_(m, nd=3): return f"{f3(m['median'], nd)} [{f3(m['min'], nd)}, {f3(m['max'], nd)}]（n {m['n']}）"


def fs(s): return f"+{s['n_pos']} / −{s['n_neg']} / 0 {s['n_tie']}（p = {f3(s['p'])}）"


def render(OUT):
    L = [f"# chargpt_detlabel_1023 の判定（judge.py）", "", f"- 作成: {OUT['time']}　judge.py sha256 {OUT['judge_sha256'][:16]}…　spec sha256 {OUT['spec_sha256'][:16]}…",
         f"- A2 の記録: {OUT['inputs']['a2_root']}（使える seed {len(OUT['A2_usable'])}/10）。A2 の J4 は再生の出力 {OUT['inputs']['replay_root']}（使える seed {OUT['A2_J4_usable']}）。A2 の J6 は loadsign（{OUT['inputs']['a2_j6']}）。",
         "- ラベルの境は §7.0（spec:284〜291）。使える seed が 10 未満なら k/10 → ceil(k·n/10)、8 未満なら INSUFFICIENT。", ""]
    for name, R in OUT["arms"].items():
        role = {"A4": "主", "A3": "記録", "S2": "判定（§12.1 の 1）"}.get(name, "")
        L += [f"## {name} − A2（{role}）", "", f"使える seed {len(R['usable'])}/10" + (f"。欠け: {R['missing']}" if R["missing"] else ""), "",
              "| 判定 | ラベル | 数 | 併記 |", "| --- | --- | --- | --- |"]
        j1 = R["J1"]
        L.append(f"| J1 喪失（spec:293） | **{j1['label']}** | LOSS {j1['n_loss']}/{j1['n']}（境 ≥ {j1['thr_hi']}・≤ {j1['thr_lo']}） | 相ごと T015/E1000/ELast {j1['phase_counts']}、学習成立 {j1['n_learned']}、ΔE {fr_(j1['dE'])}、ΔE の {name} < A2 の対 {j1['n_dE_smaller']}/{j1['n_pairs']}、符号 {fs(j1['dE_sign'])}、比の中央値 {f3(j1['dE_ratio_median'])} |")
        L.append(f"| J1b 条件 1（記録、spec:310） | — | {R['J1b']['n_pass']}/{R['J1b']['n']} | — |")
        j2 = R["J2"]
        L.append(f"| J2 L0 の伸び（spec:314） | **{j2['label']}** | ρ2 = {f3(j2['rho2'])} | g {fr_(j2['g'])} 対 A2 {fr_(j2['g_A2'])}、g 小の対 {j2['n_smaller']}、符号 {fs(j2['sign'])} |")
        j3 = R["J3"]
        L.append(f"| J3 L0 の死（spec:322） | **{j3['label']}** | median d {f3(j3['d']['median'])}（境 < {f3(j3['half_A2'])}） | d {fr_(j3['d'])}、d 小の対 {j3['n_smaller']}、符号 {fs(j3['sign'])}、d ≥ 0.5 の seed {j3['n_ge05']} |")
        j4 = R["J4"]
        L.append(f"| J4 後半の動き net5k（spec:329・454） | **{j4['label']}** | ρ4 = {f3(j4['rho4'])} | M {fr_(j4['M'])} 対 A2 再生 {fr_(j4['M_A2'])}、符号 {fs(j4['sign'])}、窓の ρ4 {', '.join(f'{w} {f3(v)}' for w, v in j4['rho4_win'].items())} |")
        L.append(f"| J5a L1 の早期の死（spec:344） | **{R['J5a']['label']}** | {R['J5a']['n_early']}/{R['J5a']['n']} | — |")
        j5 = R["J5b"]
        L.append(f"| J5b 後期の戻り（spec:345） | **{j5['label']}** | {j5['n_revive']}/{j5['n_early']} | R_k {j5['Rk']}、R_k の対の符号 {fs(j5['Rk_sign'])} |")
        L.append(f"| J6 他 unit 項（spec:359） | **{R['J6']['label']}** | {R['J6']['n_kept']}/{R['J6']['n']}（境 ≥ {R['J6']['thr']}） | 後期 s_original_other の A − A2 の符号 {fs(R['J6']['sign'])} |")
        L.append(f"| J6-LN（spec:360） | **{R['J6LN']['label']}** | {R['J6LN']['n_halves']}/{R['J6LN']['n']} | — |")
        L += ["", "記録（判定に入れない）:", ""]
        L.append(f"- J1: ΔE（早期 1〜10）{fr_(j1['dE110'])}。10 課題ごとの E1000 の平均の中央値: " + ", ".join(f3(m["median"]) for m in j1["E1000_dec"]))
        L.append("- J2: " + "; ".join(f"{k} {fr_(v)}（A2 {fr_(j2['rec_A2'][k])}）" for k, v in j2["rec"].items()))
        if R.get("cos"):
            L.append("- J2: 課題末どうしの 1 − cos（L0、seed 内の窓の中央値の seed 間の中央値、全行 / 生きた unit）: " + "; ".join(
                f"{w} {f3(med([R['cos'][s][w][0] for s in R['cos']]))} / {f3(med([R['cos'][s][w][1] for s in R['cos']]))}" for w, _ in J4_WIN_NAMED))
        L.append("- J3: " + "; ".join(f"{k} {fr_(v)}" for k, v in j3["rec"].items()) + f"; 死が初めて 0.5 以上の課題 {j3['onset50']}")
        L.append("- J4（A / A2 再生）: " + "; ".join(f"{k} {fr_(a)} / {fr_(b)}" for k, (a, b) in j4["rec"].items()))
        L.append(f"- J4 の欠けた課題の数（seed ごと）: {j4['n_missing']}、A2 再生 {j4['n_missing_A2']}")
        ref = OUT.get("A2_net100_ref_prereg")
        if ref:
            for l in (0, 1):
                if f"L{l}" in ref:
                    r_ = ref[f"L{l}"]
                    L.append(f"- J4 net100 の A2 の値 L{l}: (i) 登録前の参照（white-san の保存状態 u100 から、出典 {ref['source']}・sha256 {ref['sha256'][:16]}…）生きた unit {f3(r_['alive']['median'])} [{f3(r_['alive']['min'])}, {f3(r_['alive']['max'])}]・全行 {f3(r_['allrows']['median'])} [{f3(r_['allrows']['min'])}, {f3(r_['allrows']['max'])}]"
                             f"　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit {fr_(j4['rec'][f'L{l}_M_net100_end'][1])}・全行 {fr_(j4['rec'][f'L{l}_M_net100_allrows'][1])}　(iii) {name} 生きた unit {fr_(j4['rec'][f'L{l}_M_net100_end'][0])}")
        L.append("- J5: " + "; ".join(f"{k} {fr_(v)}" for k, v in j5["rec"].items()) + f"; Spearman 負の seed 99 対 {j5['n_neg_spearman99']}・31 対 {j5['n_neg_spearman31']}")
        for w in J6_WIN:
            L.append(f"- J6 {w}: " + "; ".join(f"{c} {fr_(R['J6']['win'][w][c])}" for c in J6_COLS))
        dg = R["degenerate"]
        L.append(f"- §7.7 縮退した課題（列 {dg['col'] or 'chain_stats.csv が無いか列が読めない'}）: seed ごとの数 {dg['n_by_seed']}" + (f"、除いた J1 {dg['J1']['label']}（{dg['J1']['n_loss']}/{dg['J1']['n']}）、除いた J4 {dg['J4']['label']}（ρ4 {f3(dg['J4']['rho4'])}）" if "J1" in dg else ""))
        L.append(f"- §7.7 追補 6 の条件 2（judge_m2.py）を満たす seed {R['cond2']['n_pass']}/{len(R['usable'])}。条件 3 は出さない（読みの選択 11）。")
        L.append("- §7.7 位置 r の中央値（窓の平均の seed 間の中央値）: " + "; ".join(f"{k} " + ", ".join(f"{w} {f3(v['median'])}" for w, v in d.items()) for k, d in R["rmed"].items()))
        for f, cols in R["chain_logit"].items():
            L.append(f"- §7.7 {f}（10 課題ごと、seed 間の中央値）: " + "; ".join(f"{c} [{', '.join(f3(x) for x in v)}]" for c, v in cols.items()))
        L.append("")
    L += ["## A2 の再生（§6.4・§12.3。この台での再生）", ""]
    for sd, inf in OUT.get("A2_replay_info", {}).items():
        if "missing" in inf: L.append(f"- seed {sd}: {inf['missing']}"); continue
        L.append(f"- seed {sd}: complete {inf['complete']}、課題 {inf['n_tasks']}（T_J4 の欠け {inf['tasks_missing']}）、online のビット一致 {inf['online_bitwise']}/{inf['n_compare']}・差の最大 {f3(inf['online_absdiff_max'])}・平均の中央値 {f3(inf['online_absdiff_mean_med'])}、終わりの状態のビット一致 {inf['state_bitwise']}/{inf['n_compare']}・相対差の最大 {f3(inf['state_reldiff_max'])}")
    L += ["", "## S-entropy の記録（Issa の判断 2026-10-24: 止める検査から記録に替えた。判定は変えない。外れた連鎖を除かない・引き直さない）", ""]
    for name, R in OUT["arms"].items():
        e = R.get("entropy") or {}
        L.append(f"- {name}: 連鎖 {e.get('n_chains', 0)}、旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差）で外れ {e.get('n_fail_old_rule', 0)}、そのうち代案（系列ごとと真の分散の大きい方の標準誤差）でも外れ {e.get('n_fail_alt', 0)}")
        if e.get("fails"):
            L += ["", f"| {name} seed | 課題 | H | CE(eval) | CE − H | SE トークン | SE 系列 | SE 真の分散 | z トークン | z 系列 | z 真の分散 | 代案で通る | π |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in e["fails"]:
                L.append(f"| {r['seed']} | {r['task']} | {f3(r['H'], 4)} | {f3(r['true_ce_eval'], 4)} | {f3(r['true_ce_eval_minus_H'], 3)} | {f3(r['true_ce_eval_se'], 3)} | {f3(r['true_ce_eval_se_seq'], 3)} | {f3(r['true_ce_eval_se_truevar'], 3)} | {f3(r['z_token'], 3)} | {f3(r['z_seq'], 3)} | {f3(r['z_truevar'], 3)} | {r['ok_entropy_seq_or_truevar']} | {r['pi_src']} |")
            L.append("")
    L += ["", "## 予測の採点（§3、A4 について。数は書き換えない）", "", "| 予測 | 誰 | p | 規則 | 結果 | p(結果) | Brier |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for x in OUT["predictions"]:
        L.append(f"| {x['pred']}（spec:{x['spec_line']}） | {x['who']} | {x['p']} | {x['text']} | {x['outcome']} | {f3(x.get('p_of_outcome'))} | {f3(x.get('brier'))} |")
    L += ["", "## 読みの選択", ""] + [f"- **{k}** {v}" for k, v in READINGS]
    return "\n".join(L) + "\n"


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, set): return sorted(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES)); ap.add_argument("--res", default=str(RES))
    ap.add_argument("--arms", nargs="*", default=["A4", "A3", "S2"])
    ap.add_argument("--a2_root", default=str(A2_ROOT)); ap.add_argument("--replay_root", default=str(RES / "replay_A2"))
    ap.add_argument("--j6_root", default=str(RES / "j6")); ap.add_argument("--a2_j6", default=str(LOADSIGN)); ap.add_argument("--a2_j6_arm", default="C3")
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--j6_names", nargs="*", default=[], help="J6 の入力の arm 列の名（例 A4=A4）。既定は腕の名そのもの")
    a = ap.parse_args(argv)
    A2S, why2 = arm_seeds(a.a2_root); A2P = per_seed(A2S)
    MO_A2, RINFO = load_motion_replay(a.replay_root)
    JA2 = read_j6(a.a2_j6, a.a2_j6_arm)
    OUT = dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), judge_sha256=sha256(__file__), spec_sha256=sha256(SPEC) if SPEC.exists() else None,
               inputs=dict(a2_root=a.a2_root, replay_root=a.replay_root, a2_j6=a.a2_j6, res=a.res, j6_root=a.j6_root),
               A2_usable=sorted(A2S), A2_missing=why2, A2_J4_usable=sum(v is not None for v in MO_A2.values()), arms={},
               A2_replay_info={s: (v if isinstance(v, dict) else dict(missing=v)) for s, v in RINFO.items()},
               A2_net100_ref_prereg=read_a2_net100_ref())
    A2rec = dict(J1_loss=sum(A2P[s]["j1"]["loss"] for s in A2P), J1b=sum(A2P[s]["j1b"]["pass_"] for s in A2P),
                 g=mrange([A2P[s]["j2"]["g0"] for s in A2P]), d=mrange([A2P[s]["j3"]["d"] for s in A2P]))
    OUT["A2_reference"] = A2rec
    for arm in a.arms:
        root = Path(a.res) / "M2" / arm
        if not root.exists():
            OUT["arms"][arm] = None; continue
        j6n = dict(x.split("=", 1) for x in a.j6_names)
        R, S, X = judge_arm(arm, root, A2S, A2P, MO_A2, JA2, a.j6_root, not a.no_ckpt, j6n.get(arm))
        if arm == "A4": OUT["predictions"] = score_predictions(R)
        OUT["arms"][arm] = R
    OUT["arms"] = {k: v for k, v in OUT["arms"].items() if v is not None}
    OUT.setdefault("predictions", [dict(pred=k, who=w, spec_line=P["line"], text=P["text"], p=P[w], scored=False, outcome="A4 未完")
                                   for k, P in PRED.items() for w in ("Fable", "Opus") if P[w] is not None])
    OUT["readings"] = [dict(key=k, text=v) for k, v in READINGS]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(jsonable(OUT), open(out / "judge.json", "w"), ensure_ascii=False, indent=1)
    (out / "judge.md").write_text(render(OUT))
    print(f"judge.md・judge.json → {out}（腕 {list(OUT['arms'])}）")
    return OUT


if __name__ == "__main__":
    main()
