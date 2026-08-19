"""posreset_0819 の**事後追加**解析: アーム間コントラスト。

  OMP_NUM_THREADS=1 python -m src.posreset_posthoc [results/posreset_0819]

spec §6 の事前登録判定 (G0/P1–P7) は `src.figures_posreset` が出す summary.md /
verdict.csv が唯一の正であり、本モジュールはそれを一切書き換えない。ここで計算するのは
spec §8 が「主戦場」と名指ししながら P 表には (レジーム B の P3 を除いて) 入れなかった
**アーム間の直接比較**である。

  「a←0 は none と非対称（意図: none は無治療対照）。ランダム特徴回帰の便益は 3 reset
   アーム間で整合済みだが、none との比較には reset 共通コストが乗る——アーム間
   コントラスト（posonly vs full vs dironly）が主戦場である理由」 [§8]

事後追加であることの明記は cbp_harm_0815 の教訓 (「事後追加である旨の明記が必須」)。
手続き自体は §6 と同一 (paired seed bootstrap, B=10000, percentile 95%CI) だが、
**rng は別立て** (20260819+1) にして事前登録側の抽選列を汚さない。

比の CI は分母が 0 を跨ぐと発散するので、分母が正の bootstrap 標本が 95% を超えるとき
だけ報告する (coupling_fbw_0813 の家内規約)。
"""
import os
import sys

import numpy as np
import pandas as pd

from .common import ROOT

B = 10000
RNG_SEED = 20260819 + 1          # 事前登録 (20260819) とは別立て
ARMS = ["posonly", "dironly", "full"]


def _ci(rng, vec):
    vec = np.asarray(vec, dtype=float)
    n = len(vec)
    bs = vec[rng.integers(0, n, (B, n))].mean(axis=1)
    return float(vec.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _ratio_ci(rng, num, den):
    """比 mean(num)/mean(den) の paired bootstrap CI。分母が 0 を跨ぐ標本が
    5% を超えたら CI は報告しない (発散するため) [coupling_fbw_0813]。"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    idx = rng.integers(0, n, (B, n))
    dn, nu = den[idx].mean(axis=1), num[idx].mean(axis=1)
    frac_pos = float((dn > 0).mean())
    point = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
    if frac_pos <= 0.95:
        return point, float("nan"), float("nan"), frac_pos
    q = np.quantile(nu / dn, [0.025, 0.975])
    return point, float(q[0]), float(q[1]), frac_pos


def analyse(resdir):
    runs = pd.read_csv(os.path.join(resdir, "runs.csv"))
    rows, L = [], []
    L.append("# posreset_0819 事後追加解析: アーム間コントラスト")
    L.append("")
    L.append("**これは事前登録 (spec_posreset_0819 §6) の判定ではない。** 事前登録の "
             "PASS/FAIL は `summary.md` / `verdict.csv` が唯一の正で、本ファイルはそれを "
             "書き換えない。ここで見るのは §8 が「主戦場」と呼びながら P 表に入れなかった "
             "アーム間の直接比較である。")
    L.append("")
    L.append(f"手続きは §6 と同一 (paired seed bootstrap, B={B}, percentile 95%CI) だが、"
             f"事前登録側の抽選列を汚さないよう rng は別立て (`default_rng({RNG_SEED})`)。")
    L.append("")
    L.append("## なぜアーム間比較が要るか")
    L.append("")
    L.append("3 つの reset アームは全て treated ユニットの読み出し `v` を 0 にする "
             "(CBP と同じ規約)。したがって `Δ_arm = M(none) − M(arm)` はどれも "
             "**「v←0 の効果」+「w 座標をどう触ったかの効果」** の和であり、共通床 "
             "V (= v←0 だけを行うアームの便益) が乗っている。本実験には V を測る "
             "対照アーム (w も b も保持して v だけ 0 にする腕) が無いので、"
             "`Δ_posonly/Δ_full` のような比は **1 に向かって膨らむ向きに偏る**。")
    L.append("")
    L.append("一方で **アーム間の差 (Δ_x − Δ_y = M(y) − M(x)) は共通床 V が相殺されて消える**。"
             "よって「どの座標操作がより効くか」の順序判断は差で行うのが正しく、"
             "比は上限として読む。")
    L.append("")

    rng = np.random.default_rng(RNG_SEED)
    for reg in sorted(runs.regime.unique()):
        p = runs[runs.regime == reg].pivot_table(index="seed", columns="arm", values="M")
        p = p.sort_index()
        d = {a: (p["none"] - p[a]).values for a in ARMS}
        L.append(f"## レジーム {reg}")
        L.append("")
        L.append("### 差 (共通床 V が相殺される = 順序判断はこちら)")
        L.append("")
        L.append("| 比較 | 点推定 | 95%CI | 0 を除外 |")
        L.append("|---|---|---|---|")
        for a, b in [("dironly", "posonly"), ("full", "dironly"), ("full", "posonly")]:
            m, lo, hi = _ci(rng, d[a] - d[b])
            exc = "はい" if (lo > 0 or hi < 0) else "いいえ"
            L.append(f"| Δ_{a} − Δ_{b} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {exc} |")
            rows.append(dict(regime=reg, kind="diff", statistic=f"D_{a}-D_{b}",
                             point=m, ci_lo=lo, ci_hi=hi))
        L.append("")
        L.append("### 比 (共通床 V ぶん 1 に膨らむ = 上限として読む)")
        L.append("")
        L.append("| 比 | 点推定 | 95%CI |")
        L.append("|---|---|---|")
        for a in ["posonly", "dironly"]:
            m, lo, hi, fp = _ratio_ci(rng, d[a], d["full"])
            ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else \
                 f"非報告 (分母>0 の bootstrap 標本 {fp:.1%} ≤ 95%)"
            L.append(f"| Δ_{a} / Δ_full | {m:.3f} | {ci} |")
            rows.append(dict(regime=reg, kind="ratio", statistic=f"D_{a}/D_full",
                             point=m, ci_lo=lo, ci_hi=hi))
        L.append("")

    L.append("## 限界 (次に埋めるべき対照)")
    L.append("")
    L.append("- **`vzero` アーム (w も b も保持し v[treated]←0 のみ) が無い**。これがあれば "
             "共通床 V を直接測れ、比を「w 座標操作の固有寄与」に補正できる。追加コストは "
             "1 レジームあたり 500k step × 10 seed ≒ 数分で、既存スナップショットから "
             "`--reuse-snapshot` で分岐できる。**本実験の解釈上いちばん効く追試**。")
    L.append("- 差の判断は V に依らないので、上表の順序結論はこの対照が無くても成立する。")
    return "\n".join(L) + "\n", pd.DataFrame(rows)


def main():
    resdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "results", "posreset_0819")
    md, df = analyse(resdir)
    with open(os.path.join(resdir, "posthoc_arm_contrasts.md"), "w") as fh:
        fh.write(md)
    df.to_csv(os.path.join(resdir, "posthoc_arm_contrasts.csv"), index=False)
    print(df.to_string(index=False))
    print(f"-> {resdir}/posthoc_arm_contrasts.{{md,csv}}")


if __name__ == "__main__":
    main()
