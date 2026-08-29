"""spec_retention_0829 (保持測定): ratchet_log_0819 の事後解析。再学習なし。

正本: 可塑性喪失/spec/保持測定spec_0829.md (vault commit 669a61e)
repo 側複製: specs/spec_retention_0829.md

段階:
  --stage sanity : S1-S5 (S3 は承認済み代替 S3') + Phase 0 校正 + R0 の計数のみ。
                   rho を一切計算しない。
  --stage full   : R1-R5 と全出力。spec §9 の前提 (repo commit+push、§5 の Issa
                   事前予測の記入) が満たされたときだけ実行する。
"""
import argparse, csv, glob, json, os, platform, sys
import numpy as np

T = 10_000
N_TASK = 100
TAUS = np.arange(1, 101)
MAIN_WIN = np.arange(96, 101)
SUB_WIN = np.arange(1, 6)
BOOT_B = 10_000
BOOT_SEED = 20260829
PERM_N = 1000
LOGDIR = "results/ratchet_log_0819/logs"
OUTDIR = "results/retention_0829"


# ----------------------------------------------------------------- ローダ

def load_seed(path):
    z = np.load(path)
    return dict(step=z["step"], flip=z["flip_state"], loss=z["eval_loss_exact"],
                p_hat=z["p_hat"], seed=int(z["seed"]))


def boundaries_from_flip(d):
    """S2: 境界を flip_state 差分から機械的に決める (ハードコードしない)。"""
    ch = np.where((d["flip"][1:] != d["flip"][:-1]).any(axis=1))[0]
    return int((d["step"][ch] % T == 0).sum()), len(ch)


def task_table(d):
    """タスク k=0..99 の同一性・新規/再訪・初出 k0・lag。

    タスク k の flip_state は step = k*T + 1 から読む (offset 0 は flip 前)。
    k=0 だけは到達境界が無く step=1 も記録されていないので step=0 から読む。
    その結果 L(0, tau) は存在せず、タスク 0 は再訪ペアの k0 としても (spec §3 の
    除外) トレンド対照の新規到達集合としても使えない。
    """
    idx = {int(s): i for i, s in enumerate(d["step"])}
    seen, rows = {}, []
    for k in range(N_TASK):
        key = tuple(int(v) for v in d["flip"][idx[k * T + 1] if k > 0 else idx[0]])
        if key in seen:
            rows.append(dict(k=k, is_new=False, k0=seen[key], lag=k - seen[key]))
        else:
            seen[key] = k
            rows.append(dict(k=k, is_new=True, k0=None, lag=None))
    return rows


def logL_matrix(d):
    """logL[k, tau-1] = log eval_loss_exact(k*T + tau)。欠測と非正値は nan。"""
    idx = {int(s): i for i, s in enumerate(d["step"])}
    M = np.full((N_TASK, len(TAUS)), np.nan)
    for k in range(N_TASK):
        for j, t in enumerate(TAUS):
            i = idx.get(k * T + int(t))
            if i is not None and d["loss"][i] > 0:
                M[k, j] = np.log(d["loss"][i])
    return M


# ----------------------------------------------------------- 推定量 (rho)

def trend_hat(lbar, new_ks, k, exclude):
    """index 距離が近い新規到達 4 点 (前後 2 点ずつ、端は片側 4 点) の中央値。

    exclude はペアの k0 (leave-one-out)。k 自身も除く (実データでは k は再訪なので
    もともと新規到達集合に居ないが、R5 の置換 null では k が新規到達なので明示的に
    除いて推定量の構造を実データと同一に保つ)。
    """
    cand = [x for x in new_ks if x != k and x != exclude and not np.isnan(lbar[x])]
    below = [x for x in cand if x < k]
    above = [x for x in cand if x > k]
    lo, hi = below[-2:], above[:2]
    need = 4 - len(lo) - len(hi)
    if need > 0:
        if len(lo) < 2:
            hi = above[:2 + need]
        else:
            lo = below[-(2 + need):]
    sel = lo + hi
    return float(np.median([lbar[x] for x in sel])) if sel else np.nan


def rho_pairs(lbar, new_ks, pairs):
    out = []
    for k0, k, lag in pairs:
        if k0 == 0 or np.isnan(lbar[k]) or np.isnan(lbar[k0]):
            continue
        Tk = trend_hat(lbar, new_ks, k, exclude=k0)
        Tk0 = trend_hat(lbar, new_ks, k0, exclude=k0)
        if np.isnan(Tk) or np.isnan(Tk0):
            continue
        out.append((k0, k, lag, lbar[k], lbar[k0], Tk, Tk0,
                    (lbar[k] - lbar[k0]) - (Tk - Tk0)))
    return out


def boot_ci(by_seed, rng, B=BOOT_B):
    """seed クラスタ bootstrap (seed 単位の復元抽出)。"""
    seeds = sorted(by_seed)
    if not seeds:
        return np.nan, np.nan, np.nan, 0
    allv = np.concatenate([by_seed[s] for s in seeds])
    draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        cat = np.concatenate([by_seed[seeds[i]] for i in pick])
        draws[b] = cat.mean() if len(cat) else np.nan
    return (float(allv.mean()), float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)), len(allv))


# ------------------------------------------------------------------ 実行

def run(stage):
    paths = sorted(glob.glob(os.path.join(LOGDIR, "seed*.npz")),
                   key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
    assert len(paths) == 10, f"expected 10 seeds, got {len(paths)}"
    data = [load_seed(p) for p in paths]
    tabs = [task_table(d) for d in data]
    sanity = {}
    meta = dict(python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(), omp=os.environ.get("OMP_NUM_THREADS"),
                stage=stage, boot_B=BOOT_B, boot_seed=BOOT_SEED, perm_n=PERM_N)

    sanity["S1_omp_is_1"] = os.environ.get("OMP_NUM_THREADS") == "1"
    s2 = [boundaries_from_flip(d) for d in data]
    sanity["S2_aligned"] = all(a == 99 and n == 99 for a, n in s2)
    sanity["S2_detail"] = [f"seed{d['seed']}:{a}/{n}" for d, (a, n) in zip(data, s2)]
    if not sanity["S2_aligned"]:
        raise SystemExit("S2 FAIL -> 中止 (spec §6)")

    pm = {int(r["seed"]): r for r in
          csv.DictReader(open("results/ratchet_log_0819/per_seed_metrics.csv"))}
    s3 = []
    for d, (a, n) in zip(data, s2):
        last = int(np.argmax(d["step"]))
        dead = float((d["p_hat"][last] < 0.05).mean())
        s3.append(dict(seed=d["seed"], dead=dead, ref_dead=float(pm[d["seed"]]["dead_frac_final"]),
                       nb=n, ref_nb=int(pm[d["seed"]]["n_boundary"])))
    sanity["S3p_dead_ok"] = all(abs(x["dead"] - x["ref_dead"]) < 1e-12 for x in s3)
    sanity["S3p_nb_ok"] = all(x["nb"] == x["ref_nb"] for x in s3)

    revisit = [sum(1 for r in t if not r["is_new"]) for t in tabs]
    l2 = [sum(1 for r in t if not r["is_new"] and r["lag"] == 2) for t in tabs]
    sanity.update(S4_revisit_per_seed=revisit, S4_revisit_total=int(sum(revisit)),
                  S4_l2_per_seed=l2, S4_l2_total=int(sum(l2)),
                  S4_match_spec=(revisit == [9, 15, 8, 6, 8, 10, 9, 11, 6, 10]
                                 and l2 == [8, 9, 5, 3, 6, 8, 8, 7, 4, 8]))

    mats = [logL_matrix(d) for d in data]
    sanity["S5_missing_main_window"] = int(sum(
        np.isnan(M[1:, MAIN_WIN - 1]).sum() for M in mats))

    real_pairs = [[(r["k0"], r["k"], r["lag"]) for r in t if not r["is_new"] and r["k0"] != 0]
                  for t in tabs]
    new_ks = [[r["k"] for r in t if r["is_new"] and r["k"] != 0] for t in tabs]
    sanity["R0_n"] = sum(len(p) for p in real_pairs)
    sanity["R0_n2"] = sum(1 for p in real_pairs for x in p if x[2] == 2)
    sanity["R0_pass"] = sanity["R0_n"] >= 40 and sanity["R0_n2"] >= 25

    os.makedirs(OUTDIR, exist_ok=True)
    print("=== S1 OMP=1 ===", sanity["S1_omp_is_1"])
    print("=== S2 境界整合 ===", sanity["S2_aligned"], sanity["S2_detail"])
    print("=== S3' 代替ローダ検算 (承認済み・正本 S3 は実行不能: spec §10) ===",
          "dead", sanity["S3p_dead_ok"], "/ n_boundary", sanity["S3p_nb_ok"])
    print("=== S4 Phase 0 ===", revisit, "計", sum(revisit), "| ℓ=2", l2, "計", sum(l2),
          "| spec 表と一致", sanity["S4_match_spec"])
    print("=== S5 主窓欠測 ===", sanity["S5_missing_main_window"])
    print(f"=== R0 === n={sanity['R0_n']} n2={sanity['R0_n2']} -> "
          f"{'PASS' if sanity['R0_pass'] else 'FAIL'}")

    if stage == "sanity":
        json.dump(dict(meta=meta, sanity=sanity), open(f"{OUTDIR}/meta.json", "w"),
                  ensure_ascii=False, indent=1, default=str)
        print("\n[stage=sanity] rho 未計算。")
        return
    if not sanity["R0_pass"]:
        print("R0 FAIL -> 統計判定を行わず記述のみ (spec §5)")
        return

    # ---- 窓平均 rho
    def rows_for(win):
        out = []
        for d, M, nk, pr in zip(data, mats, new_ks, real_pairs):
            lbar = np.nanmean(M[:, win - 1], axis=1)
            lbar = np.where(np.isnan(M[:, win - 1]).any(axis=1), np.nan, lbar)
            for k0, k, lag, a, b, tk, tk0, r in rho_pairs(lbar, nk, pr):
                out.append(dict(seed=d["seed"], k0=k0, k=k, lag=lag, lbar_k=a, lbar_k0=b,
                                That_k=tk, That_k0=tk0, rho=r))
        return out

    rows_main, rows_sub = rows_for(MAIN_WIN), rows_for(SUB_WIN)

    def bys(rows, pred):
        o = {}
        for r in rows:
            if pred(r):
                o.setdefault(r["seed"], []).append(r["rho"])
        return {k: np.array(v) for k, v in o.items()}

    V = []
    p, lo, hi, n = boot_ci(bys(rows_main, lambda r: r["lag"] == 2), np.random.default_rng(BOOT_SEED))
    r1 = "保持あり" if hi < 0 else ("逆向き所見 (正)" if lo > 0 else "保持なし (null)")
    V.append(dict(id="R1", stat="rho(96..100) lag=2", point=p, lo=lo, hi=hi, n=n, result=r1))
    g2 = bys(rows_main, lambda r: r["lag"] == 2)
    g3 = bys(rows_main, lambda r: r["lag"] >= 3)
    rng = np.random.default_rng(BOOT_SEED)
    seeds = sorted(set(g2) | set(g3))
    dr = np.empty(BOOT_B)
    for b in range(BOOT_B):
        pick = [seeds[i] for i in rng.choice(len(seeds), len(seeds), replace=True)]
        a = np.concatenate([g2[s] for s in pick if s in g2]) if any(s in g2 for s in pick) else np.array([np.nan])
        c = np.concatenate([g3[s] for s in pick if s in g3]) if any(s in g3 for s in pick) else np.array([np.nan])
        dr[b] = a.mean() - c.mean()
    p2 = boot_ci(g2, np.random.default_rng(BOOT_SEED))[0]
    p3 = boot_ci(g3, np.random.default_rng(BOOT_SEED))[0]
    V.append(dict(id="R2", stat="rho(96..100) lag2 - lag>=3", point=p2 - p3,
                  lo=float(np.nanpercentile(dr, 2.5)), hi=float(np.nanpercentile(dr, 97.5)),
                  n=sum(len(v) for v in g2.values()) + sum(len(v) for v in g3.values()),
                  result="判定なし (n 不足・報告のみ)"))
    p, lo, hi, n = boot_ci(bys(rows_sub, lambda r: r["lag"] == 2), np.random.default_rng(BOOT_SEED))
    V.append(dict(id="R3", stat="rho(1..5) lag=2", point=p, lo=lo, hi=hi, n=n, result="判定なし"))
    # R4 の窓平均 (lag>=3 側)。R4 は「ℓ=2 / ℓ>=3 別の曲線」を事前登録した記述項目なので
    # その窓平均も R4 の範囲内。判定には使わない。
    for rid, rows_, win in (("R4a", rows_main, "96..100"), ("R4b", rows_sub, "1..5")):
        p, lo, hi, n = boot_ci(bys(rows_, lambda r: r["lag"] >= 3), np.random.default_rng(BOOT_SEED))
        V.append(dict(id=rid, stat=f"rho({win}) lag>=3", point=p, lo=lo, hi=hi, n=n,
                      result="記述 (R4 の窓平均・判定に使わない)"))

    # ---- R4: rho(tau) 曲線
    curve = {}
    for lab, pred in (("lag2", lambda r: r["lag"] == 2), ("lag3plus", lambda r: r["lag"] >= 3)):
        pt, cl, ch = [], [], []
        for t in TAUS:
            rr = rows_for(np.array([t]))
            a, b, c, _ = boot_ci(bys(rr, pred), np.random.default_rng(BOOT_SEED), B=1000)
            pt.append(a); cl.append(b); ch.append(c)
        curve[lab] = (np.array(pt), np.array(cl), np.array(ch))

    # ---- R5: 置換 null (新規到達のみに「再訪」ラベルを振り直す)
    prng = np.random.default_rng(BOOT_SEED + 1)
    null = np.empty(PERM_N)
    lbars = []
    for M in mats:
        lb = np.nanmean(M[:, MAIN_WIN - 1], axis=1)
        lbars.append(np.where(np.isnan(M[:, MAIN_WIN - 1]).any(axis=1), np.nan, lb))
    for b in range(PERM_N):
        vals = []
        for lb, nk, pr in zip(lbars, new_ks, real_pairs):
            nks = set(nk)
            fake = []
            for _, _, lag in pr:
                cand = [k for k in nk if (k - lag) in nks and k - lag >= 1]
                if cand:
                    k = int(prng.choice(cand))
                    fake.append((k - lag, k, lag))
            vals += [x[7] for x in rho_pairs(lb, nk, fake)]
        null[b] = np.mean(vals) if vals else np.nan
    n_lo, n_hi = float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))
    V.append(dict(id="R5", stat="置換 null の平均 rho(96..100)", point=float(np.nanmean(null)),
                  lo=n_lo, hi=n_hi, n=PERM_N,
                  result="PASS (0 を中心)" if n_lo <= 0 <= n_hi else "FAIL (中心がずれる=推定量のバグ)"))

    for v in V:
        print(f"{v['id']:3s} {v['stat']:30s} {v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] "
              f"n={v['n']:5d}  {v['result']}")

    # ---- 出力
    with open(f"{OUTDIR}/verdict.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "stat", "point", "lo", "hi", "n", "result"])
        w.writeheader(); w.writerows(V)
    with open(f"{OUTDIR}/pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_main[0].keys()) + ["rho_sub"])
        w.writeheader()
        sub = {(r["seed"], r["k0"], r["k"]): r["rho"] for r in rows_sub}
        for r in rows_main:
            w.writerow({**r, "rho_sub": sub.get((r["seed"], r["k0"], r["k"]))})
    with open(f"{OUTDIR}/per_seed_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "n_pairs", "n_lag2", "n_lag3plus", "mean_rho_main_lag2",
                    "mean_rho_main_lag3plus", "mean_rho_sub_lag2"])
        for d in data:
            s = d["seed"]
            m2 = [r["rho"] for r in rows_main if r["seed"] == s and r["lag"] == 2]
            m3 = [r["rho"] for r in rows_main if r["seed"] == s and r["lag"] >= 3]
            b2 = [r["rho"] for r in rows_sub if r["seed"] == s and r["lag"] == 2]
            w.writerow([s, sum(1 for r in rows_main if r["seed"] == s), len(m2), len(m3),
                        np.mean(m2) if m2 else "", np.mean(m3) if m3 else "",
                        np.mean(b2) if b2 else ""])
    np.savez(f"{OUTDIR}/r4_curve.npz", tau=TAUS,
             **{f"{k}_{n}": v[i] for k, v in curve.items()
                for i, n in enumerate(("point", "lo", "hi"))}, null=null)
    json.dump(dict(meta=meta, sanity=sanity), open(f"{OUTDIR}/meta.json", "w"),
              ensure_ascii=False, indent=1, default=str)
    make_figures(curve, null, rows_main, V)
    write_summary(sanity, V, s3, curve, meta)
    print(f"\n出力: {OUTDIR}/")


def make_figures(curve, null, rows_main, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK KR", "DejaVu Sans"]
    os.makedirs(f"{OUTDIR}/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, col in (("lag2", "tab:blue"), ("lag3plus", "tab:orange")):
        p, lo, hi = curve[lab]
        ax.plot(TAUS, p, color=col, label=f"{lab} (n={sum(1 for r in rows_main if (r['lag']==2)==(lab=='lag2'))})")
        ax.fill_between(TAUS, lo, hi, color=col, alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(96, 100, color="gray", alpha=0.15)
    ax.axvspan(1, 5, color="green", alpha=0.10)
    ax.set_xlabel("tau (steps after arrival)"); ax.set_ylabel("rho (DiD, log loss)")
    ax.set_title("R4: rho(tau)  gray=main window 96-100, green=sub window 1-5")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUTDIR}/figures/fig_r4_rho_curve.png", dpi=140)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([r["rho"] for r in rows_main if r["lag"] == 2], bins=20, alpha=0.6, label="lag=2")
    ax.hist([r["rho"] for r in rows_main if r["lag"] >= 3], bins=20, alpha=0.6, label="lag>=3")
    ax.axvline(0, color="k"); ax.set_xlabel("rho (96..100)"); ax.legend()
    ax.set_title("rho distribution by lag"); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/figures/fig_rho_dist.png", dpi=140)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, color="gray")
    ax.axvline(0, color="k", ls="--", label="0")
    ax.axvline(V[0]["point"], color="r", label="observed R1")
    ax.set_xlabel("mean rho under permutation null"); ax.legend()
    ax.set_title("R5: permutation null"); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/figures/fig_perm_null.png", dpi=140)
    plt.close("all")


def write_summary(sanity, V, s3, curve, meta):
    g = {v["id"]: v for v in V}
    L = ["# retention_0829: 保持測定（タスク保持＝後ろ向きの破壊）", "",
         "仕様: `specs/spec_retention_0829.md`（vault 正本 `可塑性喪失/保持測定spec_0829.md` commit `669a61e`）。",
         "**再学習なし**。入力は `results/ratchet_log_0819/logs/seed*.npz` のみ。", "",
         "## 0. 一行", "",
         f"R0 {'PASS' if sanity['R0_pass'] else 'FAIL'} / **R1 {g['R1']['result']}** / R5 {g['R5']['result']}", "",
         "## 1. サニティ（S1–S5）", "",
         "| ID | 内容 | 結果 |", "|---|---|---|",
         f"| S1 | OMP_NUM_THREADS=1、版記録 | {'PASS' if sanity['S1_omp_is_1'] else 'FAIL'}（python {meta['python']} / numpy {meta['numpy']}） |",
         f"| S2 | 境界を flip_state 差分から決定、step≡0 (mod 1e4) | PASS（10 seed とも 99/99） |",
         f"| S3′ | ローダ検算（**承認済み代替**、§10） | dead_frac_final {'10/10 一致' if sanity['S3p_dead_ok'] else 'FAIL'} / n_boundary {'10/10 一致' if sanity['S3p_nb_ok'] else 'FAIL'} |",
         f"| S4 | Phase 0 表の独立再計算 | {'PASS（per-seed まで完全一致）' if sanity['S4_match_spec'] else 'FAIL'}　再訪 {sanity['S4_revisit_total']} / ℓ=2 {sanity['S4_l2_total']} |",
         f"| S5 | 主窓 τ=96..100 の欠測 | {sanity['S5_missing_main_window']} 件 |", "",
         "## 2. 判定", "",
         "| ID | 統計量 | 点推定 | 95% CI | n | 結果 |", "|---|---|---:|---:|---:|---|"]
    for v in V:
        L.append(f"| {v['id']} | {v['stat']} | {v['point']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['n']} | {v['result']} |")
    r1 = g["R1"]
    issa = "外れ" if not (r1["lo"] > 0) else "的中"
    cl = "外れ" if not (r1["lo"] <= 0 <= r1["hi"]) else "的中"
    L += ["", "## 3. 事前予測との照合", "",
          f"実測 R1 = {r1['point']:+.4f} [{r1['lo']:+.4f}, {r1['hi']:+.4f}] → **{r1['result']}**", "",
          f"- **Issa（実行前記入）**: R1 は**正**（以前の占有が到達を悪化させる） → **{issa}**",
          f"- **Claude（実行前記入）**: R1 は **null 寄り**（構造は上書きされる） → **{cl}**",
          "",
          "**両者とも外れた場合の扱い**: §5 の規則は結果側だけで決まる。CI が 0 を含まず負なので「保持あり」。",
          "予測が外れたこと自体は判定を変えないが、事前登録の意味として明記する。", "",
          "### ℓ 別の内訳（R4 の窓平均・記述）", "",
          "| 群 | 主窓 τ=96..100 | 副窓 τ=1..5 | n |", "|---|---:|---:|---:|",
          f"| ℓ=2 | {g['R1']['point']:+.4f} [{g['R1']['lo']:+.4f}, {g['R1']['hi']:+.4f}] | "
          f"{g['R3']['point']:+.4f} [{g['R3']['lo']:+.4f}, {g['R3']['hi']:+.4f}] | {g['R1']['n']} |",
          f"| ℓ≥3 | {g['R4a']['point']:+.4f} [{g['R4a']['lo']:+.4f}, {g['R4a']['hi']:+.4f}] | "
          f"{g['R4b']['point']:+.4f} [{g['R4b']['lo']:+.4f}, {g['R4b']['hi']:+.4f}] | {g['R4a']['n']} |", "",
          "**§4-3 の交絡について**: 「ℓ=2 の負の ρ は台帳復元機構でも説明でき、構造の保持の証拠にならない」という "
          "事前の注記は、**副窓のパターンと整合する**（副窓は ℓ=2 が −0.218 CI[−0.338,−0.125] と強く負なのに対し "
          "ℓ≥3 は −0.008 CI[−0.117,+0.142] とゼロ付近＝ビット j を反転して戻す ℓ=2 固有の機構と同じ形）。"
          "主窓の ℓ≥3 は点推定 −0.116 と ℓ=2（−0.160）に近いが **CI が 0 を跨ぐ（n=26）**ので、"
          "「主窓の効果は ℓ=2 固有ではない」とはまだ言えない。ℓ=2 と ℓ≥3 の差（R2）の CI も 0 を跨ぐ。"
          "**ℓ 依存性は本走の検出力では未決**（spec §4-5 の n 不足がそのまま効いている）。",
          "",
          "**R5 の事後の読み（事前登録の検定ではない）**: 観測値 −0.1602 は置換 null の 2.5 パーセンタイル "
          "−0.1238 より下にある。R5 は推定量が 0 を中心とするかの検査として事前登録されたもので、"
          "この比較は検定として登録されていないため、補助的な観察として置く。",
          "", "## 4. 逸脱", "",
          "1. **S3 → S3′ への差し替え（Issa 承認済み・2026-08-29）**。正本 S3 は `results/ratchet_log_0819/` のテキスト出力に `eval_loss_exact` 由来の既出値が存在せず対象が空集合のため実行不能。意図（ローダ検算）に正対する `dead_frac_final` / `n_boundary` の再現に差し替えた。**損失列の値そのものは、この走行では原理的に外部検算できない**（`W` 未保存のため再計算も不可、§4-1 と同じ理由）。",
          "2. **タスク 0 をトレンド対照の新規到達集合からも除外**。spec §3 は「k₀=0 のペア」だけを除外と書くが、step=1 が記録されておらず `L(0,τ)` が存在しないため対照としても使えない（データ側から強制された帰結。§3 の除外理由がそのまま適用される）。",
          "", "## 5. スコープ・禁止事項（spec §8 の転記）", "",
          "- **condA・w100・T=1e4・batch=1・`ratchet_log_0819`（std ＝ µ 維持アーム）限定**。condB へ外挿しない",
          "- **本 spec は µ 仮説を検定しない**（単一アームなので µ 水準間比較は原理的に不可）",
          "- 本 spec の実行は作業5（`mu_titration_0823`）の loss を見る許可を与えない",
          "- 台帳の代数は解釈の枠として引くだけで結果に混ぜない",
          "- ユニット別共変量による層別は本 spec では行わない",
          "- µ維持の利得 §4（休眠による保護）には答えない（centered アームが要る）",
          "", "## 6. 出力", "",
          "`verdict.csv` / `pairs.csv` / `per_seed_metrics.csv` / `r4_curve.npz` / `meta.json` / `figures/`"]
    open(f"{OUTDIR}/summary.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "full"], default="sanity")
    run(ap.parse_args().stage)
