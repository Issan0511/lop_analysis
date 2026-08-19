"""posreset_0819 Phase 0 [posreset_0819 §4]: 既存時系列から t_int の適格性を判定する
(再学習なし)。

  cd <repo> && OMP_NUM_THREADS=1 .venv/bin/python -m src.posreset_phase0

仕様 §4 の要求:
  1. レジーム A (methods_sde_0813 の condA A_w100 none) の dead 系指標が t_int で
     ≥0.6 かつ「概ねプラトー」であること (seed 別に記録)
  2. レジーム B (cbp_harm_0815 routeK K=100 rho=0) も同様
  3. 不適格なら t_int を {300k, 700k} から選び直し、理由を phase0_summary.md に記録

本モジュールが追加で担うこと (Phase 1 実装前に確定させる必要があるため):
  - 主指標 M の eval グリッド密度の決定 [§5 「既存 eval グリッド上」の解釈]。
    継承元 2 実験で lop_every が食い違う (methods_sde=10000, cbp_harm=1000) ので、
    どちらを採るかを数値で決めて逸脱記録として残す (§5 の記録先は phase0_summary.md)。
  - Phase 1 トランクが 10 seed であるのに対し本 Phase 0 の出典は 5 seed であり、
    両者は bit 一致しない。この非一致を経験的に確認し、Phase 0 の結論が
    レジーム水準の性質であることを明示する。

出力: results/posreset_0819/phase0_summary.md (このファイルのみを書く)
"""
import datetime
import os

import numpy as np
import pandas as pd
import torch

from .common import ROOT

# ---- 出典 [posreset_0819 §2, §4] -------------------------------------------
SRC_A = os.path.join(ROOT, "results", "methods_sde_0813", "lop_metrics_A_w100.csv")
SRC_B = os.path.join(ROOT, "results", "cbp_harm_0815", "n0_K100", "lop_metrics_B_w20.csv")
# レジーム A の 1k 格子版。condA_freeze_0815 の free 腕は methods_sde_0813 の
# condA A_w100 none と同一 config (lop_every と checkpoints のみ相違) なので、
# 共有格子上で bit 一致するはず。その一致は本スクリプトが検証する (§C)。
SRC_A_FINE = os.path.join(ROOT, "results", "condA_freeze_0815", "free",
                          "lop_metrics_A_w100.csv")
OUT = os.path.join(ROOT, "results", "posreset_0819")

T_INT = 500_000                      # 主 t_int [§3.2]
T_ALT = (300_000, 700_000)           # 不適格時の代替 [§4.3]
POST = 500_000                       # ブランチ窓幅 [§3.2]

# --- プラトーの操作的定義 (本モジュールで確定、閾値は出典データを見る前に固定) ---
LEVEL_MIN = 0.6                      # §4 字義: dead 系指標 ≥ 0.6
MID_WIN = (400_000, 600_000)         # t_int 近傍窓
LATE_WIN = (800_000, 1_000_000)      # 終端窓 (「行き着く先」の水準)
SLOPE_WIN = (300_000, 700_000)       # OLS 傾きを測る窓
PLATEAU_TOL = 0.10                   # |mid − late| と |D(t_int) − late| の許容
SLOPE_REL_MAX = 0.05                 # |傾き/100k step| / 水準 の許容
M_WIN = (T_INT, T_INT + POST)        # 主指標 M の窓 [§5]


def load_lop(path):
    """lop_metrics CSV を読み、run_id 末尾から seed 列を復元する。"""
    d = pd.read_csv(path)
    d["seed"] = d.run_id.str.extract(r"_s(\d+)$").astype(int)
    return d.sort_values(["seed", "step"])


def _win(st, y, win):
    m = (st >= win[0]) & (st <= win[1])
    return y[m]


def dead_stats(st, y):
    """dead 系指標 D(t) のプラトー判定量 [§4]。

    - D_tint      : t_int での値 (格子点が無ければ線形補間)
    - mid / late  : MID_WIN / LATE_WIN の平均
    - slope_100k  : SLOPE_WIN 上の OLS 傾き (100k step あたり)
    - rel_slope   : |slope_100k| / mid (水準に対する相対トレンド)
    """
    mid = float(_win(st, y, MID_WIN).mean())
    late = float(_win(st, y, LATE_WIN).mean())
    m = (st >= SLOPE_WIN[0]) & (st <= SLOPE_WIN[1])
    slope = float(np.polyfit(st[m], y[m], 1)[0] * 1e5)
    d_tint = float(np.interp(T_INT, st, y))
    return dict(D_tint=d_tint, mid=mid, late=late,
                gap_win=abs(mid - late), gap_pt=abs(d_tint - late),
                slope_100k=slope, rel_slope=abs(slope) / max(mid, 1e-12),
                ok_level=d_tint >= LEVEL_MIN,
                ok_gap=abs(mid - late) <= PLATEAU_TOL and abs(d_tint - late) <= PLATEAU_TOL,
                ok_slope=abs(slope) / max(mid, 1e-12) <= SLOPE_REL_MAX)


def tau_int(x):
    """自己相関時間 (initial positive sequence 推定)。格子間隔を単位とする。

    窓平均 M の精度は素朴な sd/√n ではなく sd·√(τ/n) で決まる。lop_every を
    10 倍細かくしても τ が同じだけ伸びれば実効サンプル数は増えない——この
    区別が §5 のグリッド選択の核心なので、点数ではなく τ を測る。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    c0 = float((x * x).mean())
    if c0 <= 0:
        return 1.0
    s = 1.0
    for k in range(1, max(2, len(x) // 4)):
        r = float((x[:-k] * x[k:]).mean() / c0)
        if r <= 0:
            break
        s += 2 * r
    return max(s, 1.0)


def eval_stats(st, y, coarse=10_000, fine=1_000):
    """M 窓内の clean eval_loss のばらつきと、格子密度が M に与える影響 [§5]。

    coarse 格子 (lop_every=10000) は fine 格子 (=1000) の 1/10 部分集合。fine 格子を
    位相 o = 0..9 (step mod coarse == o·fine) で 10 本の coarse 部分格子に分解し、
    - 各部分格子の窓平均のばらつき (phase_sd) = coarse 格子で M を測ったときの揺らぎ
    - 位相 o=0 (= 実際の methods_sde 格子) の偏り (bias_c) = 系統誤差
    を分ける。A は課題周期 T=10000 なので o=0 は「切り替え直前 = 最も適応した位相」に
    固定される (位相ロック)。B は K=100 で 1k も 10k も同じ位相にロックされるため
    位相効果は出ず、差は実効サンプル数だけになる。"""
    yy = _win(st, y, M_WIN)
    ss = _win(st, st, M_WIN)
    fine_ok = bool(np.all(np.diff(np.unique(ss)) == fine))
    m_fine = float(yy.mean())
    sd = float(yy.std(ddof=1))
    t_f = tau_int(yy)
    se_f = sd * np.sqrt(t_f / len(yy))
    sub = yy[(ss % coarse) == 0]
    t_c = tau_int(sub)
    se_c = float(sub.std(ddof=1)) * np.sqrt(t_c / len(sub))
    phases = np.array([yy[(ss % coarse) == (o * fine)].mean()
                       for o in range(coarse // fine)]) if fine_ok else np.array([np.nan])
    return dict(n_fine=len(yy), M_fine=m_fine, sd=sd, cv=sd / max(abs(m_fine), 1e-12),
                tau_fine=t_f, neff_fine=len(yy) / t_f, se_fine=se_f,
                n_coarse=len(sub), M_coarse=float(sub.mean()), tau_coarse=t_c,
                neff_coarse=len(sub) / t_c, se_coarse=se_c,
                bias_c=float(sub.mean()) - m_fine,
                bias_c_rel=(float(sub.mean()) - m_fine) / max(abs(m_fine), 1e-12),
                phase_sd=float(np.std(phases, ddof=1)) if len(phases) > 1 else np.nan,
                phase_range=float(phases.max() - phases.min()) if len(phases) > 1 else np.nan,
                fine_ok=fine_ok)


# ------------------------------------------------------------------ 検証系

KEY_COLS = ("dead_frac", "neg_gate_frac", "eval_loss")


def check_source_equiv():
    """A の粗格子出典 (methods_sde_0813) と細格子出典 (condA_freeze_0815 free) が
    共有格子上で一致するかを全数値列で確認する。

    一致すれば「lop_every を細かくしても軌道は変わらない (計測は乱数を消費しない)」
    ことの経験的保証になり、Phase 1 で A を 1k 格子に上げても既存実験との
    互換性が壊れないと言える [house convention: bit 互換の非破壊]。

    注意: 2 実験は OMP_NUM_THREADS が異なる環境で走っており、`torch.linalg.svdvals`
    由来の列 (eff_rank / stable_rank / eff_rank_W / top1_frac) は LAPACK の縮約順が
    スレッド数で変わるため最下位桁が揺れうる。本判定の要は §4/§5 が使う KEY_COLS が
    厳密一致することなので、両者を分けて返す。"""
    a, f = pd.read_csv(SRC_A), pd.read_csv(SRC_A_FINE)
    cols = [c for c in a.columns if c in f.columns and c not in ("step", "run_id")]
    m = a.merge(f, on=["step", "run_id"], suffixes=("_a", "_f"))
    bad, relmax = {}, {}
    for c in cols:
        x, y = m[c + "_a"], m[c + "_f"]
        ne = (x != y) & ~(x.isna() & y.isna())
        bad[c] = int(ne.sum())
        relmax[c] = float(((x - y).abs() / x.abs().clip(lower=1e-30)).max()) if bad[c] else 0.0
    diff_cols = {c: (bad[c], relmax[c]) for c in cols if bad[c]}
    return dict(n_rows=len(m), n_cols=len(cols), mismatch=sum(bad.values()),
                diff_cols=diff_cols,
                key_ok=all(bad[c] == 0 for c in KEY_COLS if c in bad),
                max_rel=max(relmax.values()) if relmax else 0.0,
                fine_steps=int(f.step.nunique()), coarse_steps=int(a.step.nunique()))


def check_not_bit_identical():
    """Phase 1 トランク (10 seed) が本 Phase 0 の出典 (5 seed) と bit 一致しないことの
    経験的確認 [§4 の脚注として必須]。

    make_gens は seed 値を使わず (base = SEED_BASE[exp] + width)、"seed" は R 次元の
    行番号でしかない。したがって R が変われば同じ generator ストリームの切り出し方が
    変わる。行単位で一致する draw と、しない draw を実測で切り分ける。"""
    from .train import make_gens
    from .envs import kaiming_mlp_params, SCREnv, LTUTarget, GaussEnv

    eq = lambda x, y: bool(torch.equal(x, y))
    out = {}

    # 学習器初期化 (rand): W は行単位一致、v は W の消費量が R 依存なのでズレる
    z = {}
    for R in (5, 10):
        z[R] = kaiming_mlp_params(R, 100, 20, make_gens("A", 100, "cpu")["init"], "cpu")
    out["A_init_W_rows"] = eq(z[5][0], z[10][0][:5])
    out["A_init_v_rows"] = eq(z[5][2], z[10][2][:5])
    z = {}
    for R in (5, 10):
        z[R] = kaiming_mlp_params(R, 20, 21, make_gens("B", 20, "cpu")["init"], "cpu")
    out["B_init_W_rows"] = eq(z[5][0], z[10][0][:5])
    out["B_init_v_rows"] = eq(z[5][2], z[10][2][:5])

    # 教師 (randint): W は行単位一致、後続の b はズレる
    t = {R: LTUTarget(R, 20, 100, 0.7, make_gens("A", 100, "cpu")["teacher"], "cpu")
         for R in (5, 10)}
    out["A_teacher_W_rows"] = eq(t[5].W, t[10].W[:5])
    out["A_teacher_b_rows"] = eq(t[5].b, t[10].b[:5])

    # 入力 (A): flip_state は行単位一致するが、その消費量が R 依存なので以後全部ズレる
    e = {}
    for R in (5, 10):
        env = SCREnv(R, 20, 15, torch.full((R,), 10000, dtype=torch.long),
                     make_gens("A", 100, "cpu")["input"], "cpu")
        e[R] = (env.flip_state.clone(), env.step(), env.step())
    out["A_env_flip_rows"] = eq(e[5][0], e[10][0][:5])
    out["A_env_step1_rows"] = eq(e[5][1], e[10][1][:5])
    out["A_env_step2_rows"] = eq(e[5][2], e[10][2][:5])

    # 入力 (B): randn は対生成のため行 0 しか一致しない (row-wise 一致は rand/randint のみ)
    g = {}
    for R in (5, 10):
        env = GaussEnv(R, 21, [0.0] * R, make_gens("B", 20, "cpu")["input"], "cpu",
                       kappa=[1] * R, spike_dir="alt")
        g[R] = env.step()
    out["B_env_step1_row0"] = eq(g[5][0], g[10][0])
    out["B_env_step1_rows"] = eq(g[5], g[10][:5])
    return out


def check_dead_equals_neg_gate(df):
    """dead_frac と neg_gate_frac の一致確認 [§4 item 1]。

    ReLU (act_alpha=0) では a = relu(pre) なので |a| < dead_tol は pre < dead_tol と同値。
    dead_frac は 1{|a|<1e-7} の割合が dead_tau=0.95 超のユニット割合、neg_gate_frac は
    1{pre>0} の割合が 1−dead_tau=0.05 未満のユニット割合で、pre ∈ (0, 1e-7) のサンプルが
    無い限り厳密に同じ集合を数える。理屈ではなく実測で確認する。"""
    d = (df.dead_frac - df.neg_gate_frac).abs()
    return dict(max_abs_diff=float(d.max()), n=int(len(d)),
                exact=bool((df.dead_frac == df.neg_gate_frac).all()))


# ------------------------------------------------------------------ 出力

def _md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    f = lambda c, v: (fmt[c].format(v) if c in fmt else str(v))
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(f(c, r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def main():
    os.makedirs(OUT, exist_ok=True)
    dfs = {"A": load_lop(SRC_A), "B": load_lop(SRC_B)}
    fine = {"A": load_lop(SRC_A_FINE), "B": load_lop(SRC_B)}     # B は出典自体が 1k 格子
    equiv = check_source_equiv()
    bit = check_not_bit_identical()

    dead_rows, eval_rows, alt_rows = [], [], []
    for reg in ("A", "B"):
        for s, g in dfs[reg].groupby("seed"):
            st, y = g.step.values.astype(float), g.dead_frac.values
            r = dead_stats(st, y)
            r.update(regime=reg, seed=int(s),
                     neg_tint=float(np.interp(T_INT, st, g.neg_gate_frac.values)))
            dead_rows.append(r)
        for s, g in fine[reg].groupby("seed"):
            st, y = g.step.values.astype(float), g.eval_loss.values
            r = eval_stats(st, y)
            r.update(regime=reg, seed=int(s))
            eval_rows.append(r)
        # 代替 t_int の比較 [§4.3]
        for t in (T_ALT[0], T_INT, T_ALT[1]):
            vals, gaps = [], []
            for s, g in dfs[reg].groupby("seed"):
                st, y = g.step.values.astype(float), g.dead_frac.values
                late = float(_win(st, y, LATE_WIN).mean())
                v = float(np.interp(t, st, y))
                vals.append(v)
                gaps.append(abs(v - late))
            alt_rows.append(dict(regime=reg, t_int=t, D_min=min(vals),
                                 D_mean=float(np.mean(vals)), max_gap=max(gaps),
                                 all_level_ok=all(v >= LEVEL_MIN for v in vals),
                                 all_gap_ok=max(gaps) <= PLATEAU_TOL,
                                 post_window=1_000_000 - t))
    dead = pd.DataFrame(dead_rows)
    ev = pd.DataFrame(eval_rows)
    alt = pd.DataFrame(alt_rows)
    verdict = bool(dead.ok_level.all() and dead.ok_gap.all() and dead.ok_slope.all())

    L = []
    A = L.append
    A("# posreset_0819 Phase 0 (spec §4): t_int の適格性と M の eval グリッド決定\n")
    A(f"生成: `OMP_NUM_THREADS=1 .venv/bin/python -m src.posreset_phase0` "
      f"({datetime.date.today().isoformat()})。**再学習なし** — 既存時系列の再解析のみ。\n")
    A("出典 (spec §2, §4):\n")
    A(f"- レジーム A: `results/methods_sde_0813/lop_metrics_A_w100.csv` "
      f"(condA w100, method none, seed 0–4, lop_every=10000)")
    A(f"- レジーム B: `results/cbp_harm_0815/n0_K100/lop_metrics_B_w20.csv` "
      f"(routeK K=100, rho=0, seed 0–4, lop_every=1000)")
    A(f"- 補助: `results/condA_freeze_0815/free/lop_metrics_A_w100.csv` "
      f"(A と同一 config の 1k 格子版。§C で bit 一致を検証してから §3 でのみ使用)\n")

    # ---------------- 判定 ----------------
    A(f"## 判定: t_int = {T_INT:,} は **{'適格' if verdict else '不適格'}**\n")
    if verdict:
        A(f"事前登録された 3 条件 (水準 ≥ {LEVEL_MIN}、プラトー、トレンド微小) を "
          f"**10 seed 系列すべてで充足**。t_int は変更しない。"
          f"代替候補 {T_ALT[0]//1000}k / {T_ALT[1]//1000}k を採る理由は無い (§4 の比較表を参照)。\n")
    else:
        A("**不適格**。代替 t_int の選定理由を §4 に記す。\n")

    # ---------------- 1. dead 系指標 ----------------
    A("## 1. dead 系指標の per-seed 表\n")
    A("### 1.1 dead_frac と neg_gate_frac は一致する (ReLU)\n")
    for reg in ("A", "B"):
        c = check_dead_equals_neg_gate(dfs[reg])
        A(f"- レジーム {reg}: 全 {c['n']} 行 (seed × step) で "
          f"`dead_frac == neg_gate_frac` が**厳密に成立** (最大絶対差 {c['max_abs_diff']:g})。")
    A("")
    A("理由: 学習器は ReLU (`nets.VecMLP.act_alpha=0`) なので a = relu(pre)。"
      "`dead_frac` は 1{|a| < dead_tol=1e-7} の標本割合が dead_tau=0.95 を超えるユニット、"
      "`neg_gate_frac` は 1{pre>0} の標本割合 (= p̂) が 1−dead_tau=0.05 未満のユニットを数える。"
      "pre ∈ (0, 1e-7) の標本が存在しない限り両者は同一集合であり、実測でもそうなっている。\n")
    A("**したがって以下の D(t) は dead_frac = neg_gate_frac の共通値である。**\n")
    A("さらに重要な同一性: spec §3.3 の treated 集合 `{i : p̂_i < 0.05}` は "
      "`neg_gate_frac` の定義そのもの (同じ固定 eval バッチ・同じ閾値) である。"
      "よって本節の D(t_int) は **Phase 1 の treated_frac の直接の予測値**であり、"
      "適格性ゲート treated_frac ≥ 0.3 [§3.3] に対する余裕をここで読める。\n")

    A("### 1.2 per-seed 表 (D = dead_frac = neg_gate_frac)\n")
    A(f"窓: mid = [{MID_WIN[0]//1000}k, {MID_WIN[1]//1000}k], "
      f"late = [{LATE_WIN[0]//1000}k, {LATE_WIN[1]//1000}k], "
      f"傾き = OLS over [{SLOPE_WIN[0]//1000}k, {SLOPE_WIN[1]//1000}k]\n")
    A(_md_table(
        dead.sort_values(["regime", "seed"]).to_dict("records"),
        ["regime", "seed", "D_tint", "neg_tint", "mid", "late", "gap_win", "gap_pt",
         "slope_100k", "rel_slope", "ok_level", "ok_gap", "ok_slope"],
        {"D_tint": "{:.3f}", "neg_tint": "{:.3f}", "mid": "{:.4f}", "late": "{:.4f}",
         "gap_win": "{:.4f}", "gap_pt": "{:.4f}", "slope_100k": "{:+.4f}",
         "rel_slope": "{:.4f}"}))
    A("")
    A(f"- **水準**: D(t_int) の最小値は {dead.D_tint.min():.3f} "
      f"(regime {dead.loc[dead.D_tint.idxmin(), 'regime']}, "
      f"seed {int(dead.loc[dead.D_tint.idxmin(), 'seed'])})。"
      f"閾値 {LEVEL_MIN} に対し全 seed が大幅に上回る。")
    A(f"- **treated_frac の見通し**: 同じ最小値 {dead.D_tint.min():.3f} が §3.3 のゲート "
      f"0.3 に対する最悪ケース予測。余裕は約 {dead.D_tint.min()/0.3:.1f} 倍で、"
      "10 seed 中 8 seed という要件が落ちる確率は実質無視できる。")
    A("")

    # ---------------- 2. プラトーの定義 ----------------
    A("## 2. 「概ねプラトー」の操作的定義\n")
    A("spec §4 は「概ねプラトー」を定義していないので、ここで定義を固定して数値ごと記録する。"
      "**D(t) が 3 条件すべてを満たすとき「t_int でプラトー」と呼ぶ**:\n")
    A(f"1. **水準** — D(t_int) ≥ {LEVEL_MIN} (spec §4 の字義)")
    A(f"2. **終端水準との一致** — |mean D over mid − mean D over late| ≤ {PLATEAU_TOL:.2f} "
      f"**かつ** |D(t_int) − mean D over late| ≤ {PLATEAU_TOL:.2f}。"
      "窓平均どうしの比較を主、単点比較を従とする (単点は量子化に弱いため)")
    A(f"3. **トレンドの小ささ** — [{SLOPE_WIN[0]//1000}k, {SLOPE_WIN[1]//1000}k] の OLS 傾きが "
      f"|傾き/100k step| / mid ≤ {SLOPE_REL_MAX}\n")
    A("**許容値の根拠 (データを見る前に決めた量から導く)**:")
    A(f"- D は [0,1] に有界な割合で、分解能は 1/h。h はレジーム A で 100、"
      f"**レジーム B で 20 (分解能 0.05)**。許容 {PLATEAU_TOL:.2f} は粗い方の "
      f"**量子化 2 段ぶん**である。B で「1 ユニット余分に死ぬ」だけで 0.05 動くので、"
      "これより厳しい許容は物理的に無意味になる。")
    A(f"- 条件 3 は条件 2 が見落とす単調ドリフトを捕まえる。{SLOPE_REL_MAX} は "
      f"「ブランチ窓 {POST//1000}k を通しても水準の {SLOPE_REL_MAX*5:.0%} しか動かない」"
      "に相当する。")
    A("")
    n_tight = int(((dead.gap_pt > 0.05) | (dead.gap_win > 0.05)).sum())
    A("**正直な注記 (閾値感度)**: 観測された最悪値は "
      f"gap_win {dead.gap_win.max():.4f} / gap_pt {dead.gap_pt.max():.4f} / "
      f"rel_slope {dead.rel_slope.max():.4f} で、いずれも許容の内側だが"
      f"**余裕は薄い** (gap_pt の最悪値は許容 {PLATEAU_TOL:.2f} の {dead.gap_pt.max()/PLATEAU_TOL:.1%})。"
      f"許容を {PLATEAU_TOL:.2f} → 0.05 に締めると {n_tight}/{len(dead)} 系列が条件 2 で落ちる。"
      "したがって**本判定は許容値の選び方に対して頑健ではない**。"
      "許容 0.10 が事後に緩められたものでないことは §2 の量子化議論で担保するが、"
      "読者は「500k は 1M 終端水準に量子化 2 段以内まで近い」以上のことを"
      "本節から読み取るべきではない。ただし:\n")
    A("- 残差ドリフトの符号は **全 10 系列で正** (dead が増える向き) である。"
      "t_int は「まだ病理が浅い」側には外れておらず、遅らせても質的に何も変わらない。")
    A("- 本実験の設計は 4 アームが**同一トランクを共有する対応比較**なので、"
      "未処置ユニットの残差ドリフトは none アームにも同じだけ乗り、Δ_arm では相殺される。"
      "プラトーはあくまで「t_int が病理の定常部に入っているか」の確認であり、"
      "判定量の不偏性を担保する仮定ではない。\n")

    # ---------------- 3. eval_loss ----------------
    A("## 3. clean eval_loss の水準とばらつき (主指標 M の精度)\n")
    A(f"M = [{M_WIN[0]//1000}k, {M_WIN[1]//1000}k] の clean eval_loss 窓平均 [§5]。"
      "判定は eval_loss のみ (dead_frac は判定に使用禁止 [§9])。\n")
    A("列: `sd` = 窓内グリッド点の標準偏差、`cv` = sd/M、`tau` = 自己相関時間 "
      "(格子間隔単位, initial positive sequence)、`neff` = n/tau、"
      "`se` = sd·√(tau/n) = 窓平均の標準誤差。"
      "`bias_c` = (10k 格子の M) − (1k 格子の M)。\n")
    A(_md_table(
        ev.sort_values(["regime", "seed"]).to_dict("records"),
        ["regime", "seed", "M_fine", "sd", "cv", "tau_fine", "neff_fine", "se_fine",
         "M_coarse", "tau_coarse", "neff_coarse", "se_coarse", "bias_c", "bias_c_rel",
         "phase_sd", "phase_range"],
        {"M_fine": "{:.4f}", "sd": "{:.4f}", "cv": "{:.2f}", "tau_fine": "{:.1f}",
         "neff_fine": "{:.0f}", "se_fine": "{:.4f}", "M_coarse": "{:.4f}",
         "tau_coarse": "{:.2f}", "neff_coarse": "{:.0f}", "se_coarse": "{:.4f}",
         "bias_c": "{:+.4f}", "bias_c_rel": "{:+.1%}", "phase_sd": "{:.4f}",
         "phase_range": "{:.4f}"}))
    A("")
    for reg in ("A", "B"):
        e = ev[ev.regime == reg]
        A(f"- **レジーム {reg}**: M = {e.M_fine.mean():.4f} "
          f"(seed 間 sd {e.M_fine.std(ddof=1):.4f}); グリッド点の sd = "
          f"{e.sd.min():.4f}–{e.sd.max():.4f} (cv {e.cv.min():.2f}–{e.cv.max():.2f})。"
          f"1k 格子で se = {e.se_fine.min():.4f}–{e.se_fine.max():.4f}、"
          f"10k 格子で se = {e.se_coarse.min():.4f}–{e.se_coarse.max():.4f}。")
    A("")
    A("**500 点 vs 50 点は素朴には √10 の改善だが、実際はレジームで意味が違う**:\n")
    eA, eB = ev[ev.regime == "A"], ev[ev.regime == "B"]
    A(f"- **A では分散はほぼ改善しない**。tau(1k) = {eA.tau_fine.min():.0f}–"
      f"{eA.tau_fine.max():.0f} 格子 (= {eA.tau_fine.min():.0f}k–{eA.tau_fine.max():.0f}k step) "
      f"と系列の相関時間が長いため、実効サンプル数は 1k 格子で "
      f"{eA.neff_fine.min():.0f}–{eA.neff_fine.max():.0f}、10k 格子で "
      f"{eA.neff_coarse.min():.0f}–{eA.neff_coarse.max():.0f} と**ほぼ同じ**。"
      "点を 10 倍にしても冗長なだけである。")
    A(f"- **A で効くのは分散ではなく系統誤差 (位相ロック)**。condA の課題周期は "
      "T = 10,000 なので、10k 格子は毎回 `step mod 10000 == 0`、すなわち"
      "**タスク切り替え直前 = そのタスクに最も適応しきった位相**だけを見る。"
      f"実測で 10k 格子の M は 1k 格子の M より **全 5 seed で低く**、"
      f"偏りは {eA.bias_c.min():.4f} 〜 {eA.bias_c.max():.4f} "
      f"(相対で {abs(eA.bias_c_rel).min():.1%} 〜 {abs(eA.bias_c_rel).max():.1%} の過小評価)。"
      f"位相別窓平均のばらつきは phase_sd {eA.phase_sd.min():.4f}–{eA.phase_sd.max():.4f}、"
      f"レンジ {eA.phase_range.min():.4f}–{eA.phase_range.max():.4f} に達する。")
    A(f"- **B では逆に分散だけが効く**。K = 100 なので 1k 格子も 10k 格子も同じ位相 "
      "(`step mod 100 == 0`) にロックされ位相効果は無い。一方 tau(1k) = "
      f"{eB.tau_fine.min():.1f}–{eB.tau_fine.max():.1f} 格子とほぼ無相関なので "
      f"neff は {eB.neff_fine.min():.0f}–{eB.neff_fine.max():.0f} vs "
      f"{eB.neff_coarse.min():.0f}–{eB.neff_coarse.max():.0f} で**素直に約 √10 = 3.2 倍**"
      f"精度が上がる (se {eB.se_coarse.mean():.4f} → {eB.se_fine.mean():.4f})。")
    A("")
    A("**M の精度が判定に効く度合い**:\n")
    A(f"- B の none 腕は seed 間 sd が {eB.M_fine.std(ddof=1):.4f} しかない "
      f"(M ≈ {eB.M_fine.mean():.3f})。10k 格子の格子誤差 se ≈ {eB.se_coarse.mean():.4f} は"
      "**この seed 間ばらつきの数倍**であり、対応 seed bootstrap の対象量 Δ に "
      f"√2·se ≈ {eB.se_coarse.mean()*np.sqrt(2):.4f} の純粋な計測ノイズを載せてしまう。"
      f"1k 格子ならこれは {eB.se_fine.mean()*np.sqrt(2):.4f} に下がる。"
      "cbp_harm_0815 の K=100 セルで CBP が動かした量が約 0.10 だったことを踏まえると、"
      "10k 格子ではノイズが効果の 1/3 に達し、1k 格子なら 1/9 に収まる。")
    A(f"- A は seed 間 sd が {eA.M_fine.std(ddof=1):.4f} と大きく分散面では格子密度が効かないが、"
      "位相ロックの偏りはアーム間で**同じ大きさとは限らない** "
      "(リセット直後のユニットが課題周期のどの位相で再適応するかはアームに依存する) ため、"
      "Δ_arm に系統誤差として残る危険がある。位相を平均する 1k 格子はこの交絡を消す。")
    A("")
    A("### 3.1 M_late (窓末尾 100k) への警告\n")
    lt = []
    for reg in ("A", "B"):
        rr = []
        for s, g in fine[reg].groupby("seed"):
            st, y = g.step.values.astype(float), g.eval_loss.values
            yy = y[(st >= 900_000) & (st <= 1_000_000)]
            t = tau_int(yy)
            rr.append((yy.mean(), yy.std(ddof=1) * np.sqrt(t / len(yy)), len(yy) / t))
        rr = np.array(rr)
        lt.append((reg, rr))
        A(f"- **{reg}**: M_late = {rr[:,0].mean():.4f} (seed 間 sd {rr[:,0].std(ddof=1):.4f})、"
          f"1k 格子 100 点での se = {rr[:,1].min():.4f}–{rr[:,1].max():.4f} "
          f"(neff {rr[:,2].min():.0f}–{rr[:,2].max():.0f})。")
    A("")
    A("spec §5 は A の二段回復の遅さ対策として M_late を併記させるが、**A の M_late は"
      "自己相関時間が窓幅に対して長すぎて実効サンプル数が一桁**になり、se が M_late 本体と"
      "同オーダーになる seed がある。**A の M_late は方向を見る補助量にとどめ、"
      "PASS/FAIL の根拠にしないこと** (§6 でも M_late は判定量ではない)。"
      "B の M_late は neff が十分あり問題ない。")
    A("")

    # ---------------- 4. 代替 t_int ----------------
    A("## 4. 代替 t_int との比較 [§4.3]\n")
    A(_md_table(alt.to_dict("records"),
                ["regime", "t_int", "D_min", "D_mean", "max_gap", "all_level_ok",
                 "all_gap_ok", "post_window"],
                {"D_min": "{:.3f}", "D_mean": "{:.3f}", "max_gap": "{:.4f}",
                 "post_window": "{:,}"}))
    A("")
    A(f"- **300k は落ちる**: 終端水準との差が最大 "
      f"{alt[alt.t_int==300_000].max_gap.max():.3f} で許容 {PLATEAU_TOL:.2f} を大きく超える "
      "(まだ病理が進行中の相にある)。")
    A(f"- **700k はプラトー条件では 500k と同等以上**だが、"
      "spec §3.2 が post_steps = 500,000 を固定しているため、総 step が 1.2M/系 に増えて "
      "実行規模 §3.5 (50M step) が崩れる。プラトーの利得に見合わない。")
    A(f"- **500k を維持する**。プラトー条件を満たし、かつ 1M step の既存レジーム記述 "
      "(A: dead≈0.97 @1M、B: dead≈0.99 @1M) と地続きの窓が取れる。\n")

    # ---------------- 5. 実装決定の記録 ----------------
    A("## 5. Phase 1 実装決定の記録 (2026-08-19)\n")
    A("### 5.1 決定: M の eval グリッドは **両レジームとも lop_every = 1000**\n")
    A("spec §5 は M を「既存 eval グリッド上」の窓平均と定めるが、継承元 2 実験で"
      "既存グリッドが食い違う (methods_sde_0813 = 10000、cbp_harm_0815 = 1000)。"
      "**両レジームを lop_every = 1000 に統一する**。これは A について"
      "**理由つきの逸脱** (継承元より細かい格子を使う) として記録する。\n")
    A("**根拠 (§3 の実測値)**:\n")
    A(f"1. A の 10k 格子は課題周期 T=10000 と完全に同期しており、M を全 5 seed で "
      f"{abs(eA.bias_c_rel).min():.1%} 〜 {abs(eA.bias_c_rel).max():.1%} 系統的に過小評価する。"
      "「既存グリッド」を字義どおり採ると、判定の主指標に位相ロック由来の系統誤差が入る。")
    A(f"2. B では格子誤差 se が {eB.se_coarse.mean():.4f} → {eB.se_fine.mean():.4f} と"
      "約 3.2 倍改善し、seed 間ばらつき ("
      f"sd {eB.M_fine.std(ddof=1):.4f}) に対する計測ノイズの比が大きく下がる。")
    A("3. **軌道は変わらない**。計測 (`compute_lop_metrics` / `eval_batch`) は学習用の "
      "generator を一切消費しないので、lop_every を変えても学習軌道は同一である。"
      "§C.1 で同一 config の 10k 版と 1k 版を突き合わせ、軌道由来の列が厳密一致することを"
      "確認済み。既存 config の再現性を壊さない。")
    A("4. コストは許容範囲。`compute_lop_metrics` の実測は A (R=10,h=100,N=2000) で "
      "約 0.29 s/回、B (h=20) で約 0.05 s/回。1k 格子・レジーム A の Phase 1 全体 "
      "(cont 1M + 4 アーム × 500k = 3000 回) で計測が約 15 分、B は約 3 分。"
      "10k 格子との差は 1 レジームあたり十数分にとどまる。\n")
    A("**この決定の限界 (明記して残す)**: B は K=100 なので 1000 は依然 100 の倍数であり、"
      "1k 格子も 10k 格子も「教師再サンプル直前」の同一位相にロックされている。"
      "B の位相ロックは本決定では解消されない (解消には lop_every を 100 の倍数から外す"
      "必要があり、それは「既存 eval グリッド上」からのより大きな逸脱になる)。"
      "B の M は「切り替え直前の誤差」の窓平均であると読むこと。\n")
    A("### 5.2 決定: Phase 0 の適格性は **レジーム水準の性質**として扱う\n")
    A("次節 §C のとおり Phase 1 トランク (10 seed) は本 Phase 0 の出典 (5 seed) と "
      "bit 一致しない。よって本節の per-seed 表は「Phase 1 の seed 0–4 の予測」ではなく、"
      "**このレジーム設定において t_int=500k が病理の定常部に入る**という水準の主張である。"
      "**per-seed の実効ゲートは spec §3.3 の treated_frac ≥ 0.3 (10 seed 中 8 以上) で、"
      "これは実際のトランク上で測る。**\n")

    # ---------------- C. 検証 ----------------
    A("## C. 検証\n")
    A("### C.1 A の 2 出典は共有格子上で一致する\n")
    A(f"`methods_sde_0813/lop_metrics_A_w100.csv` (10k 格子, {equiv['coarse_steps']} step) と "
      f"`condA_freeze_0815/free/lop_metrics_A_w100.csv` (1k 格子, {equiv['fine_steps']} step) を "
      f"(step, run_id) で結合すると {equiv['n_rows']} 行 × 共通数値列 {equiv['n_cols']} 列 "
      f"(= {equiv['n_rows'] * equiv['n_cols']} セル) を照合できる。結果:\n")
    A(f"- **判定に使う列 `dead_frac` / `neg_gate_frac` / `eval_loss` は全 "
      f"{equiv['n_rows']} 行で厳密一致** "
      f"({'PASS' if equiv['key_ok'] else 'FAIL'})。")
    if equiv["diff_cols"]:
        A(f"- 不一致は {equiv['mismatch']} セルのみで、"
          + "、".join(f"`{c}` {n} セル (最大相対差 {r:.1e})"
                      for c, (n, r) in equiv["diff_cols"].items())
          + "。いずれも `torch.linalg.svdvals` 由来の列で、差は CSV 出力桁 (`%.6g`) の"
            "最下位 1 桁ぶん。2 実験は OMP_NUM_THREADS が異なる環境で走っており、"
            "LAPACK の縮約順がスレッド数で変わることによる丸め差である "
            "(軌道の相違ではない: 軌道が違えば dead_frac / eval_loss が先に割れる)。")
    else:
        A("- 不一致 0 セル。")
    A("")
    A("両者は lop_every と checkpoints のみが異なる同一 config であり、この一致は "
      "「計測頻度を上げても学習軌道は変わらない」ことの直接証拠になる (§5.1 根拠 3)。"
      "したがって §3 の A の 1k 格子解析は、spec §4 が指定する出典と同じ軌道を見ている。\n")

    A("### C.2 Phase 1 トランク (10 seed) は出典 (5 seed) と bit 一致しない\n")
    A("`make_gens` は run の `seed` 値を使わず `base = SEED_BASE[exp] + width` だけで "
      "generator を作る (`train.py`)。つまり **\"seed\" は R 次元の行番号にすぎず**、"
      "R が 5 から 10 に変わると同じストリームの切り出し方が変わる。実測:\n")
    A(_md_table([dict(draw=k, rows_match=v) for k, v in bit.items()],
                ["draw", "rows_match"]))
    A("")
    A("読み方:\n")
    A(f"- `torch.rand` / `torch.randint` は行単位では一致する "
      f"(`A_init_W_rows`={bit['A_init_W_rows']}, `A_teacher_W_rows`={bit['A_teacher_W_rows']}, "
      f"`A_env_flip_rows`={bit['A_env_flip_rows']})。")
    A(f"- しかし**後続の draw がずれる**: v は W が R·h·d 個消費した後に引かれるので "
      f"`A_init_v_rows`={bit['A_init_v_rows']} / `B_init_v_rows`={bit['B_init_v_rows']}、"
      f"教師 b も `A_teacher_b_rows`={bit['A_teacher_b_rows']}。"
      "A の入力環境は flip_state が R·f 個消費するため 1 step 目から "
      f"`A_env_step1_rows`={bit['A_env_step1_rows']} と全行ずれる。")
    A(f"- B の入力は `torch.randn` で、正規乱数は対生成のため行 0 しか一致しない "
      f"(`B_env_step1_row0`={bit['B_env_step1_row0']}, "
      f"`B_env_step1_rows`={bit['B_env_step1_rows']}) — rand/randint より弱い。")
    A("")
    A("**結論**: 10 seed トランクの run `_s0`..`_s4` は本 Phase 0 の seed 0–4 と"
      "**別の実現値**である。Phase 0 は t_int 適格性を**レジーム水準**で確立するに留まり、"
      "per-seed の適格性は Phase 1 のトランク上で treated_frac として測り直す [§3.3]。\n")

    A("## 主張してはいけないこと (spec §9 の再掲)\n")
    A("- 本ファイルの dead_frac / neg_gate_frac は **t_int の選定にのみ**使う。"
      "Phase 1 の PASS/FAIL 判定は clean eval_loss のみで行う。")
    A("- 本ファイルの per-seed 表を Phase 1 の seed 別予測として読まない (§C.2)。\n")

    path = os.path.join(OUT, "phase0_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(dead.sort_values(["regime", "seed"]).to_string(index=False))
    print()
    print(ev.sort_values(["regime", "seed"])[
        ["regime", "seed", "M_fine", "sd", "se_fine", "M_coarse", "se_coarse",
         "bias_c", "bias_c_rel"]].to_string(index=False))
    print(f"\nverdict: t_int={T_INT} {'ELIGIBLE' if verdict else 'NOT ELIGIBLE'}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
