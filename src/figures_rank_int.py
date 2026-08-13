"""rank_int_0814 の判定・図・summary (spec_rank_int_0814 §6–8)。

  python -m src.figures_rank_int results/rank_int_0814

出力:
  fig_ri_loss.png      — アーム別 online loss vs step (t_int 起点、タスク境界線、seed 平均±SE)
  fig_ri_series.png    — stable_rank_W_alive / dead_frac の 3 アーム重ね描き (介入マーカー)
  fig_ri_forest.png    — M の paired 差の CI 森プロット
  arm_metrics.csv      — seed × arm × width の M / M_tail / dead 増分
  summary.md           — 事前登録判定 (P-int-1/2/3, G1, 反証条件) + サニティ + 確認事項
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = ["none", "svdrec", "shuffle", "svdrec_alive"]
ARM_COLOR = {"none": "tab:gray", "svdrec": "tab:blue", "shuffle": "tab:orange",
             "svdrec_alive": "tab:green"}
N_BOOT = 4000


def arms_present(df):
    return [a for a in ARMS if a in set(df.arm)]


def load(resdir):
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")

    def cat(prefix):
        fs = [f for f in sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
              if "s1resume" not in f]
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        return df.join(runs[["seed", "width", "arm"]], on="run_id", how="inner")

    import yaml
    with open(os.path.join(resdir, "config_used.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return dict(runs=runs, loss=cat("online_loss"), lop=cat("lop_metrics"),
                ilog=pd.read_csv(os.path.join(resdir, "intervention_log.csv")),
                phase0=pd.read_csv(os.path.join(resdir, "phase0_targets.csv")),
                cfg=cfg)


def arm_metrics(d):
    """seed×arm×width の主指標 M (介入後 20 タスクの online loss 平均) と
    副指標 M_tail (各タスク末尾 2k の平均)、dead/srank の介入後増分。"""
    cfg = d["cfg"]
    t_int = cfg["rank_int"]["t_int"]
    post = cfg["rank_int"]["post_steps"]
    period = cfg["condA"]["T_values"][0]
    loss = d["loss"]
    win = loss[(loss.step > t_int) & (loss.step <= t_int + post)].copy()
    win["task"] = (win.step - t_int - 1) // period
    win["pos"] = (win.step - t_int - 1) % period // cfg["common"]["loss_bin"]
    nbin = period // cfg["common"]["loss_bin"]

    rows = []
    for (width, arm, seed), g in win.groupby(["width", "arm", "seed"]):
        task_mean = g.groupby("task").loss.mean()
        tail = g[g.pos >= nbin - 2].groupby("task").loss.mean()
        lop = d["lop"]
        gl = lop[(lop.width == width) & (lop.arm == arm) & (lop.seed == seed)]
        gl = gl.sort_values("step")
        v_at = lambda col, s: float(np.interp(s, gl.step.values, gl[col].values))
        rows.append(dict(width=width, arm=arm, seed=seed,
                         M=task_mean.mean(), M_tail=tail.mean(),
                         dead_t_int=v_at("dead_frac", t_int),
                         dead_end=v_at("dead_frac", t_int + post),
                         d_dead=v_at("dead_frac", t_int + post) - v_at("dead_frac", t_int),
                         srank_end=v_at("stable_rank_W_alive", t_int + post),
                         eval_end=v_at("eval_loss", t_int + post)))
    return pd.DataFrame(rows)


def paired_ci(a, b, rng):
    """paired seed bootstrap: mean(a−b) と 95%CI。a, b は seed 揃いの配列。"""
    diff = np.asarray(a) - np.asarray(b)
    bs = rng.choice(len(diff), (N_BOOT, len(diff)), replace=True)
    bm = diff[bs].mean(axis=1)
    return dict(mean=float(diff.mean()), lo=float(np.quantile(bm, 0.025)),
                hi=float(np.quantile(bm, 0.975)))


def pair_table(am, metric, rng):
    """width × アームペアの paired 差集計。"""
    rows = []
    for width, g in am.groupby("width"):
        piv = g.pivot_table(index="seed", columns="arm", values=metric)
        pairs = [("svdrec", "none"), ("shuffle", "none"), ("svdrec", "shuffle")]
        if "svdrec_alive" in piv.columns:
            pairs += [("svdrec_alive", "none"), ("svdrec_alive", "shuffle")]
        for a, b in pairs:
            ci = paired_ci(piv[a].values, piv[b].values, rng)
            rows.append(dict(width=width, metric=metric, pair=f"{a}-{b}", **ci,
                             excl_zero=bool(ci["lo"] > 0 or ci["hi"] < 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 図

def fig_loss(d, figdir):
    cfg = d["cfg"]
    t_int, post = cfg["rank_int"]["t_int"], cfg["rank_int"]["post_steps"]
    period = cfg["condA"]["T_values"][0]
    loss = d["loss"]
    widths = sorted(loss.width.unique())
    fig, axes = plt.subplots(1, len(widths), figsize=(7 * len(widths), 3.8),
                             squeeze=False)
    for k, width in enumerate(widths):
        ax = axes[0][k]
        sub = loss[(loss.width == width) & (loss.step > t_int - 20000)
                   & (loss.step <= t_int + post)]
        for arm in arms_present(sub):
            g = sub[sub.arm == arm]
            m = g.groupby("step").loss.agg(["mean", "sem"])
            ax.plot(m.index - t_int, m["mean"], lw=1.2, color=ARM_COLOR[arm], label=arm)
            ax.fill_between(m.index - t_int, m["mean"] - m["sem"], m["mean"] + m["sem"],
                            color=ARM_COLOR[arm], alpha=0.2, lw=0)
        for s in range(0, post + 1, period):
            ax.axvline(s, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("step - t_int")
        ax.set_title(f"w={width}")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("online loss (1k bin, seed mean±SE)")
            ax.legend(fontsize=8)
    fig.suptitle("post-intervention online loss by arm (dashed = intervention at task boundary)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_ri_loss.png"), dpi=150)
    plt.close(fig)


def fig_series(d, figdir):
    cfg = d["cfg"]
    t_int, post = cfg["rank_int"]["t_int"], cfg["rank_int"]["post_steps"]
    lop = d["lop"]
    lop = lop[lop.step % cfg["common"]["lop_every"] == 0]
    widths = sorted(lop.width.unique())
    mets = [("stable_rank_W_alive", "stable_rank_W_alive"), ("dead_frac", "dead_frac")]
    fig, axes = plt.subplots(len(mets), len(widths),
                             figsize=(7 * len(widths), 3.2 * len(mets)), squeeze=False)
    for i, (met, lbl) in enumerate(mets):
        for k, width in enumerate(widths):
            ax = axes[i][k]
            for arm in arms_present(lop):
                g = lop[(lop.width == width) & (lop.arm == arm)]
                m = g.groupby("step")[met].agg(["mean", "sem"])
                ax.plot(m.index, m["mean"], lw=1.2, color=ARM_COLOR[arm], label=arm)
                ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                                color=ARM_COLOR[arm], alpha=0.2, lw=0)
            ax.axvline(t_int, color="black", lw=1, ls="--")
            ax.set_xlim(0, t_int + post)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"w={width}")
            if i == len(mets) - 1:
                ax.set_xlabel("step")
            if k == 0:
                ax.set_ylabel(lbl)
                if i == 0:
                    ax.legend(fontsize=8)
    fig.suptitle("rank / dead trajectories by arm (dashed = intervention)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_ri_series.png"), dpi=150)
    plt.close(fig)


def fig_forest(pt_all, figdir):
    sub = pt_all[pt_all.pair.isin(["svdrec-none", "shuffle-none"])]
    mets = sub.metric.unique()
    fig, axes = plt.subplots(1, len(mets), figsize=(4.6 * len(mets), 3.2), squeeze=False)
    for j, met in enumerate(mets):
        ax = axes[0][j]
        g = sub[sub.metric == met].reset_index(drop=True)
        ys = np.arange(len(g))[::-1]
        for y, r in zip(ys, g.itertuples()):
            color = ARM_COLOR[r.pair.split("-")[0]]
            ax.errorbar(r.mean, y, xerr=[[r.mean - r.lo], [r.hi - r.mean]],
                        fmt="o", ms=5, capsize=4, color=color, lw=1.6)
            ax.text(0.02, y + 0.18, f"w{r.width} {r.pair}", transform=ax.get_yaxis_transform(),
                    fontsize=8, va="bottom")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks([])
        ax.set_xlabel(f"paired diff of {met} (95% CI)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("intervention effects: negative = better than none")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_ri_forest.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 判定と summary

def verdicts(pt_M, am, ilog, rng):
    """§6 の P-int-1/2/3, G1, 反証条件を width 別に判定。"""
    out = {}
    for width in sorted(am.width.unique()):
        p = pt_M[pt_M.width == width].set_index("pair")
        sn = p.loc["svdrec-none"]
        hn = p.loc["shuffle-none"]
        sh = p.loc["svdrec-shuffle"]
        pint1 = bool(sn["mean"] < 0 and sn.hi < 0)
        pint2 = bool((hn.lo <= 0 <= hn.hi) or (sh["mean"] < 0 and sh.hi < 0))
        # G1: 介入直後 Δdead の svdrec − shuffle
        il = ilog[ilog.width == width]
        dsv = il.post_svdrec_dead_frac - il.pre_dead_frac
        dsh = il.post_shuffle_dead_frac - il.pre_dead_frac
        g1ci = paired_ci(dsv.values, dsh.values, rng)
        g1 = bool(-0.05 <= g1ci["lo"] and g1ci["hi"] <= 0.05)
        # P-int-3: t_int→+200k の dead 増分
        piv = am[am.width == width].pivot_table(index="seed", columns="arm", values="d_dead")
        d_sn = paired_ci(piv["svdrec"].values, piv["none"].values, rng)
        d_hn = paired_ci(piv["shuffle"].values, piv["none"].values, rng)
        pint3 = bool(d_sn["mean"] < 0 and d_sn["hi"] < 0
                     and d_hn["lo"] <= 0 <= d_hn["hi"])
        # 反証条件の前提: svdrec が srank_target の 80% 以上まで回復したか。
        # 字義は「M も dead 増分も none と区別できない」(= 両 CI がゼロ含有)
        rec80 = bool((il.post_svdrec_stable_rank_W_alive
                      >= 0.8 * il.srank_target).all())
        m_indist = bool(sn.lo <= 0 <= sn.hi)
        dead_indist = bool(d_sn["lo"] <= 0 <= d_sn["hi"])
        refuted = bool(rec80 and m_indist and dead_indist)
        sn_worse = bool(sn.lo > 0)          # svdrec が none より有意に悪い (想定外方向)
        if pint1 and pint2:
            cell = "ランク因果を支持 (P-int-1 ∧ P-int-2)"
        elif pint1 and not pint2:
            cell = "svdrec ≈ shuffle < none → 摂動効果。ランク因果は不支持"
        elif sn_worse:
            cell = ("**判定表想定外**: svdrec が none より有意に悪化 (ランク回復介入は有害)。"
                    "ランク因果 (ランク回復→可塑性回復) は不支持")
        else:
            cell = "全アーム同等 → この regime では回復不能"
        out[width] = dict(pint1=pint1, pint2=pint2, pint3=pint3, cell=cell,
                          g1_pass=g1, g1_ci=g1ci, sn=dict(sn), hn=dict(hn), sh=dict(sh),
                          d_dead_sn=d_sn, d_dead_hn=d_hn, rec80=rec80, refuted=refuted,
                          sn_worse=sn_worse)
    return out


def write_summary(resdir, d, am, pt_all, vd, meta):
    p0 = d["phase0"]
    lines = ["# rank_int_0814 summary (spec_rank_int_0814 §6 事前登録判定)\n"]

    lines.append("## サニティ\n")
    lines.append(f"- S1 (resume bit 一致) / S2 (介入の数値保証): {meta.get('sanity')}")
    il = d["ilog"]
    lines.append(f"- S2 最大誤差: sv_preserved {il.s2_sv_preserved.max():.2e}, "
                 f"dF_match {il.s2_dF_match.max():.2e}, "
                 f"normF {max(il.s2_normF_svdrec.max(), il.s2_normF_shuffle.max()):.2e} "
                 f"(許容 {d['cfg']['rank_int']['match_tol']:.0e})")
    lines.append(f"- svdrec ε clipped (target 到達不能): {int(il.svd_eps_clipped.sum())}/{len(il)} 件, "
                 f"shuffle abort: {int((il.shf_abort.fillna('') != '').sum())}/{len(il)} 件")
    if "shf_n_draw" in il.columns and (il.shf_n_draw > 1).any():
        redraw = il[il.shf_n_draw > 1]
        lines.append(f"- **逸脱**: shuffle の G が単発抽選で ΔF 目標に届かず、同一 generator 列"
                     f"からの棄却サンプリングで再抽選した seed: "
                     + ", ".join(f"w{r.width}/s{r.seed} (n_draw={r.shf_n_draw})"
                                 for r in redraw.itertuples()))

    lines.append("\n## Phase 0 ラベル分布 (回復/予防 — 結論の主張文言を規定)\n")
    lines.append(p0.groupby(["width", "label"]).size().to_string())
    lines.append("\n注: 仕様字義の離陸定義は退化 (phase0_summary.md 参照)。ラベルはロバスト定義。")
    lines.append(f"- t_int=150k 適格性: srank t50 通過 {int(p0.srank_t50_passed_150k.sum())}/10, "
                 f"dead≤0.15 は {int(p0.dead_at_tint_ok.sum())}/10 (不適格も除外せず全 seed 使用)")

    lines.append("\n## 主判定 (M = 介入後 20 タスクの online loss、paired seed bootstrap 95%CI)\n")
    lines.append(pt_all[pt_all.metric == "M"].to_string(index=False))
    lines.append("")
    for width, v in vd.items():
        lines.append(f"\n### width {width}")
        lines.append(f"- P-int-1 (svdrec < none): {'成立' if v['pint1'] else '不成立'} "
                     f"(diff {v['sn']['mean']:.4g}, CI [{v['sn']['lo']:.4g}, {v['sn']['hi']:.4g}])")
        lines.append(f"- P-int-2 (摂動一般で説明不能): {'成立' if v['pint2'] else '不成立'} "
                     f"(shuffle−none CI [{v['hn']['lo']:.4g}, {v['hn']['hi']:.4g}], "
                     f"svdrec−shuffle CI [{v['sh']['lo']:.4g}, {v['sh']['hi']:.4g}])")
        lines.append(f"- **判定表**: {v['cell']}")
        lines.append(f"- G1 (介入直後 Δdead svdrec−shuffle ±0.05 内): "
                     f"{'PASS' if v['g1_pass'] else 'FAIL → dead 復活交絡を疑い感度分析を追試'} "
                     f"(CI [{v['g1_ci']['lo']:.3f}, {v['g1_ci']['hi']:.3f}])")
        lines.append(f"- P-int-3 (dead 増分 svdrec<none ∧ shuffle≈none): "
                     f"{'成立' if v['pint3'] else '不成立'} "
                     f"(svdrec−none CI [{v['d_dead_sn']['lo']:.3f}, {v['d_dead_sn']['hi']:.3f}], "
                     f"shuffle−none CI [{v['d_dead_hn']['lo']:.3f}, {v['d_dead_hn']['hi']:.3f}])")
        lines.append(f"- 反証条件 (80%回復 {'済' if v['rec80'] else '未達'} かつ M/dead とも none と同等): "
                     f"{'**該当 → 低ランクは原因ではなく随伴症状**' if v['refuted'] else '非該当'}")
    w_signs = [np.sign(v["sn"]["mean"]) for v in vd.values()]
    lines.append(f"\n- 両 width の svdrec−none 符号一致: "
                 f"{'一致 (頑健)' if len(set(w_signs)) == 1 else '不一致'}")

    lines.append("\n## 結論と所見\n")
    end_tbl = am.pivot_table(index="width", columns="arm",
                             values=["srank_end", "dead_end"], aggfunc="mean").round(3)
    lines.append("+200k 時点のアーム別平均 (srank_alive / dead_frac):\n")
    lines.append(end_tbl.to_string())
    lines.append("""
1. **ランク回復は一過性**: svdrec / svdrec_alive は介入直後に srank_target 付近まで
   回復するが、訓練の継続で ~100k step かけて none と同水準まで再崩壊する
   (fig_ri_series 上段)。低ランク整列はこの regime の学習力学のアトラクターであり、
   スペクトルだけ戻しても維持されない。
2. **svdrec 系は dead をむしろ加速**: P-int-3 の予測 (svdrec が dead 蓄積を抑える) の
   逆で、介入直後および +200k の dead_frac は none より高い (fig_ri_series 下段)。
   ノルム保存の一様再スケールが支配方向 (学習済み解) を縮め、持ち上げられた
   小特異値方向はタスクと不整合なため、SGD がユニットごと殺す方向に働くと解釈できる。
3. **判定**: 両幅・両ラベル (回復/予防) を通じて svdrec (alive 限定版含む) は M を有意に
   悪化させ、等 ΔF の shuffle は none と区別できない。事前登録の判定表では
   「ランク因果不支持」側だが、想定された「全アーム同等」ではなく
   **ランク回復介入が積極的に有害**という、より強い形の不支持である。
   反証条件の字義 (「none と区別できない」) には該当しないが、
   「低ランクは LoP の原因ではなく随伴症状」という結論は M の悪化方向によって
   さらに強く支持される。""")

    alog_path = os.path.join(resdir, "intervention_log_svdrec_alive.csv")
    if os.path.exists(alog_path):
        al = pd.read_csv(alog_path)
        lines.append("\n## 感度分析 (G1 破れ時の追試, spec §6): alive 行のみ SVD 介入 (svdrec_alive)\n")
        dd = (al.post_svdrec_alive_dead_frac - al.pre_dead_frac)
        lines.append(f"- 介入直後 Δdead: 平均 {dd.mean():+.3f} "
                     f"(svdrec 本体は dead 行も再構成するため直後 Δdead が正になりがちだが、"
                     f"alive 限定版は dead 行を触らない)")
        lines.append(f"- 介入直後 srank_alive 回復: "
                     + ", ".join(f"w{r.width}/s{r.seed} {r.pre_stable_rank_W_alive:.2f}"
                                 f"→{r.post_svdrec_alive_stable_rank_W_alive:.2f}"
                                 f"(target {r.srank_target:.2f})"
                                 for r in al.itertuples()))
        sa = pt_all[(pt_all.metric == "M") & pt_all.pair.str.startswith("svdrec_alive")]
        lines.append("\n" + sa.to_string(index=False))

    lines.append("\n## 副指標 (M_tail = タスク末尾 2k 平均)\n")
    lines.append(pt_all[pt_all.metric == "M_tail"].to_string(index=False))

    lines.append("\n## seed 別 (回復/予防 層別用)\n")
    lab = p0.set_index(["width", "seed"]).label
    am2 = am.copy()
    am2["label"] = [lab.loc[(r.width, r.seed)] for r in am2.itertuples()]
    lines.append(am2.sort_values(["width", "seed", "arm"]).round(4).to_string(index=False))

    lines.append("\n## 先生への確認事項 (仕様 §8)\n")
    lines.append("1. **shuffle の連続化**: 先生の記述は「空間内要素のみシャッフル」だが、"
                 "ΔF を svdrec と厳密一致させるため top-k 部分空間内の連続ランダム回転 "
                 "W'(θ)=U_k Q(θ) S_k V_kᵀ + 残差 を採用した (特異値・両 span・ノルム・ランク不変)。"
                 "この連続化が意図と整合するかご確認ください。")
    lines.append("2. **ε の決定規則**: svdrec の ε は「介入直後の stable_rank_W_alive ≈ "
                 "step0 の値 (srank_target)」を pre-intervention の dead マスク下で bisect。"
                 "介入がゲートを開き直すため、実測の介入後 srank_alive は target を"
                 "やや上回ることがある (intervention_log の post_svdrec_stable_rank_W_alive 参照)。")
    lines.append("3. (Phase 0 逸脱) eval_loss 離陸時刻の字義定義は full-batch の高い初期損失で"
                 "退化するため、ロバスト定義 (argmin 以降で min+0.5*(v_1M−min) 上抜き) で"
                 "ラベル付けした。")
    lines.append("4. (Phase 1 逸脱) shuffle の G は「seed 固定の単発抽選」だと回転角"
                 "スペクトル次第で ΔF 目標に僅かに届かないケースがある (w20/s4 で "
                 "9.08 < 9.24)。同一 seed 列からの決定論的棄却サンプリング (到達可能な G "
                 "が出るまで再抽選、最大50回) に拡張した。")

    with open(os.path.join(resdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    rng = np.random.default_rng(0)

    d = load(args.results)
    with open(os.path.join(args.results, "meta.json")) as fh:
        meta = json.load(fh)
    am = arm_metrics(d)
    am.to_csv(os.path.join(args.results, "arm_metrics.csv"), index=False)
    pt_all = pd.concat([pair_table(am, "M", rng), pair_table(am, "M_tail", rng)],
                       ignore_index=True)
    vd = verdicts(pt_all[pt_all.metric == "M"], am, d["ilog"], rng)

    fig_loss(d, figdir)
    fig_series(d, figdir)
    fig_forest(pt_all, figdir)
    write_summary(args.results, d, am, pt_all, vd, meta)
    print(pt_all.to_string(index=False))
    for w, v in vd.items():
        print(f"w{w}: {v['cell']} | G1 {'PASS' if v['g1_pass'] else 'FAIL'} "
              f"| P-int-3 {'成立' if v['pint3'] else '不成立'} | 反証 {v['refuted']}")
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
