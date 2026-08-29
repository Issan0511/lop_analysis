"""spec_retention_0829 (保持測定): ratchet_log_0819 の事後解析。再学習なし。

正本: 可塑性喪失/spec/保持測定spec_0829.md (vault commit 669a61e)
repo 側複製: specs/spec_retention_0829.md

段階:
  --stage sanity : S1-S5 (S3 は代替 S3') + Phase 0 校正 + R0 ゲートの計数のみ。
                   ρ を一切計算しないので事前予測の記入前に実行してよい。
  --stage full   : R1-R5 と全出力。**§5 の Issa 事前予測が記入済みのときだけ実行する。**
"""
import argparse, json, os, platform, sys
import numpy as np

T = 10_000
N_TASK = 100
MAIN_WIN = list(range(96, 101))     # 主読み出し窓 tau in {96..100}
SUB_WIN = list(range(1, 6))         # 副読み出し窓 tau in {1..5}
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
    F, S = d["flip"], d["step"]
    ch = np.where((F[1:] != F[:-1]).any(axis=1))[0]
    return S[ch], int((S[ch] % T == 0).sum()), len(ch)


def task_table(d):
    """タスク k=0..99 の同一性・新規/再訪・初出 k0・lag。

    タスク k の flip_state は step = k*T + 1 の記録点から読む (offset 0 は flip 前)。
    ただし k=0 だけは到達境界が無く step=1 も記録されていないので step=0 から読む。
    その結果 L(0, tau) は存在せず、タスク 0 は再訪ペアの k0 としても (spec §3 の除外)
    トレンド対照の新規到達集合としても使えない (実装注記・summary の逸脱節)。
    """
    S, F = d["step"], d["flip"]
    idx = {int(s): i for i, s in enumerate(S)}
    ident, seen = [], {}
    rows = []
    for k in range(N_TASK):
        i = idx[k * T + 1] if k > 0 else idx[0]
        key = tuple(int(v) for v in F[i])
        ident.append(key)
        if key in seen:
            rows.append(dict(k=k, key=key, is_new=False, k0=seen[key], lag=k - seen[key]))
        else:
            seen[key] = k
            rows.append(dict(k=k, key=key, is_new=True, k0=None, lag=None))
    return rows


def loss_at(d, k, taus):
    """log L(k, tau) を tau ごとに返す。記録が無い tau は nan。"""
    idx = {int(s): i for i, s in enumerate(d["step"])}
    out = []
    for t in taus:
        i = idx.get(k * T + t)
        out.append(np.log(d["loss"][i]) if i is not None and d["loss"][i] > 0 else np.nan)
    return np.array(out, dtype=float)


# ----------------------------------------------------------- 推定量 (rho)

def trend_hat(lbar, new_ks, k, exclude):
    """T_hat(k): index 距離が近い新規到達 4 点 (前後 2 点ずつ、端は片側 4 点) の中央値。

    exclude はペアの k0 (leave-one-out)。タスク 0 は到達境界を持たないので
    新規到達集合から除く (§3 の除外理由と同一。summary の実装注記に記載)。
    """
    cand = [x for x in new_ks if x != k and x != exclude]
    below = [x for x in cand if x < k][-2:]
    above = [x for x in cand if x > k][:2]
    need = 4 - len(below) - len(above)
    if need > 0:
        if len(below) < 2:
            above = [x for x in cand if x > k][:2 + need]
        else:
            below = [x for x in cand if x < k][-(2 + need):]
    sel = below + above
    if not sel:
        return np.nan
    return float(np.median([lbar[x] for x in sel]))


def rho_for_pairs(d, window, pairs=None, new_ks=None, rows=None):
    """窓平均 log L を先に作り、DiD の rho を返す。"""
    rows = rows if rows is not None else task_table(d)
    new_ks = new_ks if new_ks is not None else [r["k"] for r in rows if r["is_new"] and r["k"] != 0]
    lbar = {}
    for k in range(N_TASK):
        v = loss_at(d, k, window)
        lbar[k] = float(np.mean(v)) if not np.isnan(v).any() else np.nan
    out = []
    src = pairs if pairs is not None else [(r["k0"], r["k"], r["lag"])
                                           for r in rows if not r["is_new"]]
    for k0, k, lag in src:
        if k0 == 0:                                  # §3 の除外
            continue
        if np.isnan(lbar[k]) or np.isnan(lbar[k0]):  # S5 による除外
            continue
        Tk = trend_hat(lbar, new_ks, k, exclude=k0)
        Tk0 = trend_hat(lbar, new_ks, k0, exclude=k0)
        if np.isnan(Tk) or np.isnan(Tk0):
            continue
        out.append(dict(seed=d["seed"], k0=k0, k=k, lag=lag,
                        lbar_k=lbar[k], lbar_k0=lbar[k0], That_k=Tk, That_k0=Tk0,
                        rho=(lbar[k] - lbar[k0]) - (Tk - Tk0)))
    return out


def boot_ci(per_seed_values, rng, B=BOOT_B):
    """seed クラスタ bootstrap: seed を復元抽出し、抽出された seed の全イベントの平均。"""
    seeds = sorted(per_seed_values)
    if not seeds:
        return np.nan, np.nan, np.nan, 0
    allv = np.concatenate([per_seed_values[s] for s in seeds])
    point = float(np.mean(allv))
    draws = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        cat = np.concatenate([per_seed_values[seeds[i]] for i in pick])
        draws[b] = cat.mean() if len(cat) else np.nan
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5)), len(allv)


# ------------------------------------------------------------------ 実行

def run(stage):
    paths = sorted(__import__("glob").glob(os.path.join(LOGDIR, "seed*.npz")),
                   key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
    assert len(paths) == 10, f"expected 10 seeds, got {len(paths)}"
    data = [load_seed(p) for p in paths]
    lines, sanity = [], {}

    # ---- S1
    meta = dict(python=sys.version.split()[0], numpy=np.__version__,
                platform=platform.platform(),
                omp=os.environ.get("OMP_NUM_THREADS"), stage=stage)
    sanity["S1_omp_is_1"] = (os.environ.get("OMP_NUM_THREADS") == "1")

    # ---- S2
    s2 = [boundaries_from_flip(d) for d in data]
    sanity["S2_aligned"] = all(a == 99 and n == 99 for _, a, n in s2)
    sanity["S2_detail"] = [f"seed{d['seed']}:{a}/{n}" for d, (_, a, n) in zip(data, s2)]
    if not sanity["S2_aligned"]:
        raise SystemExit("S2 FAIL: 境界が step≡0 (mod 1e4) に揃っていない -> 中止 (spec §6)")

    # ---- S3' (代替。正本 S3 は対象が空集合で実行不能。spec §10)
    import csv
    pm = {}
    with open("results/ratchet_log_0819/per_seed_metrics.csv") as fh:
        for r in csv.DictReader(fh):
            pm[int(r["seed"])] = r
    s3 = []
    for d, (_, a, n) in zip(data, s2):
        last = int(np.argmax(d["step"]))
        dead = float((d["p_hat"][last] < 0.05).mean())
        ref_dead = float(pm[d["seed"]]["dead_frac_final"])
        ref_nb = int(pm[d["seed"]]["n_boundary"])
        s3.append(dict(seed=d["seed"], dead=dead, ref_dead=ref_dead,
                       ok_dead=abs(dead - ref_dead) < 1e-12, nb=n, ref_nb=ref_nb, ok_nb=(n == ref_nb)))
    sanity["S3p_dead_ok"] = all(x["ok_dead"] for x in s3)
    sanity["S3p_nb_ok"] = all(x["ok_nb"] for x in s3)
    sanity["S3p_detail"] = s3

    # ---- S4 / Phase 0
    tabs = [task_table(d) for d in data]
    revisit = [sum(1 for r in t if not r["is_new"]) for t in tabs]
    l2 = [sum(1 for r in t if (not r["is_new"]) and r["lag"] == 2) for t in tabs]
    sanity["S4_revisit_per_seed"] = revisit
    sanity["S4_revisit_total"] = int(sum(revisit))
    sanity["S4_l2_per_seed"] = l2
    sanity["S4_l2_total"] = int(sum(l2))
    sanity["S4_match_spec"] = (sum(revisit) == 92 and sum(l2) == 66
                               and revisit == [9, 15, 8, 6, 8, 10, 9, 11, 6, 10]
                               and l2 == [8, 9, 5, 3, 6, 8, 8, 7, 4, 8])

    # ---- S5
    miss = []
    for d in data:
        have = set(int(s) for s in d["step"])
        for k in range(1, N_TASK):
            for t in MAIN_WIN:
                if k * T + t not in have:
                    miss.append((d["seed"], k, t))
    sanity["S5_missing_main_window"] = len(miss)

    # ---- R0 (計数のみ。入力側の量で rho を含まない)
    pairs_all, pairs_l2 = 0, 0
    for t in tabs:
        for r in t:
            if r["is_new"] or r["k0"] == 0:
                continue
            pairs_all += 1
            if r["lag"] == 2:
                pairs_l2 += 1
    sanity["R0_n"] = pairs_all
    sanity["R0_n2"] = pairs_l2
    sanity["R0_pass"] = (pairs_all >= 40 and pairs_l2 >= 25)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "meta.json"), "w") as fh:
        json.dump(dict(meta=meta, sanity=sanity), fh, ensure_ascii=False, indent=1, default=str)

    print("=== S1 OMP_NUM_THREADS=1 ===", sanity["S1_omp_is_1"])
    print("=== S2 境界整合 (flip 差分から決定) ===", sanity["S2_aligned"], sanity["S2_detail"])
    print("=== S3' 代替ローダ検算 (正本 S3 は実行不能: spec §10) ===")
    print("   dead_frac_final 再現:", sanity["S3p_dead_ok"], " n_boundary 再現:", sanity["S3p_nb_ok"])
    for x in s3:
        print(f"     seed{x['seed']}: dead {x['dead']:.4f} vs 既出 {x['ref_dead']:.4f} "
              f"({'OK' if x['ok_dead'] else 'NG'}) / n_boundary {x['nb']} vs {x['ref_nb']}")
    print("=== S4 Phase 0 校正 ===")
    print("   再訪 per seed:", revisit, "計", sum(revisit), "(spec 92)")
    print("   ℓ=2 per seed:", l2, "計", sum(l2), "(spec 66)")
    print("   spec の表と完全一致:", sanity["S4_match_spec"])
    print("=== S5 主窓 τ=96..100 の欠測 ===", sanity["S5_missing_main_window"])
    print(f"=== R0 ゲート === n={pairs_all} (>=40), n2={pairs_l2} (>=25) -> "
          f"{'PASS' if sanity['R0_pass'] else 'FAIL'}")

    if stage == "sanity":
        print("\n[stage=sanity] ρ は計算していない。R1-R5 は --stage full で、"
              "§5 の Issa 事前予測が記入済みのときだけ実行すること。")
        return

    # ------------------------------------------------------- R1-R5 (full)
    if not sanity["R0_pass"]:
        print("R0 FAIL -> 統計判定を行わず記述のみ (spec §5)")
        return
    rng = np.random.default_rng(BOOT_SEED)
    rows_main = [r for d, t in zip(data, tabs) for r in rho_for_pairs(d, MAIN_WIN, rows=t)]
    rows_sub = [r for d, t in zip(data, tabs) for r in rho_for_pairs(d, SUB_WIN, rows=t)]

    def bys(rows, pred):
        out = {}
        for r in rows:
            if pred(r):
                out.setdefault(r["seed"], []).append(r["rho"])
        return {k: np.array(v) for k, v in out.items()}

    verdict = []
    p, lo, hi, n = boot_ci(bys(rows_main, lambda r: r["lag"] == 2), rng)
    res = "保持あり" if hi < 0 else ("逆向き所見" if lo > 0 else "保持なし (null)")
    verdict.append(dict(id="R1", stat="rho(96..100), lag=2", point=p, lo=lo, hi=hi, n=n, result=res))
    d2 = bys(rows_main, lambda r: r["lag"] == 2)
    d3 = bys(rows_main, lambda r: r["lag"] >= 3)
    p2, l2lo, l2hi, n2 = boot_ci(d2, np.random.default_rng(BOOT_SEED))
    p3, l3lo, l3hi, n3 = boot_ci(d3, np.random.default_rng(BOOT_SEED))
    verdict.append(dict(id="R2", stat="rho(96..100) lag2 - lag>=3", point=p2 - p3,
                        lo=np.nan, hi=np.nan, n=n2 + n3, result="判定なし (n 不足)"))
    p, lo, hi, n = boot_ci(bys(rows_sub, lambda r: r["lag"] == 2), np.random.default_rng(BOOT_SEED))
    verdict.append(dict(id="R3", stat="rho(1..5), lag=2", point=p, lo=lo, hi=hi, n=n, result="判定なし"))

    for v in verdict:
        print(f"{v['id']:3s} {v['stat']:28s} {v['point']:+.4f} "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']}  {v['result']}")
    with open(os.path.join(OUTDIR, "verdict.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "stat", "point", "lo", "hi", "n", "result"])
        w.writeheader(); w.writerows(verdict)
    with open(os.path.join(OUTDIR, "pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_main[0].keys()))
        w.writeheader(); w.writerows(rows_main)
    print(f"\n出力: {OUTDIR}/verdict.csv, pairs.csv, meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sanity", "full"], default="sanity")
    run(ap.parse_args().stage)
