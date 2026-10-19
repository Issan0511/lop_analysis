#!/usr/bin/env python3
"""measure.py の rows.json・runs.json から、帯ごとの表（summary.md）、切替ごとの全行（table_switch.csv）、図を作る。"""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("results/kappa_reload_1020")
rows = json.loads((OUT / "rows.json").read_text()); runs = json.loads((OUT / "runs.json").read_text())
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]; NAME = {"LR": "leaky 0.1", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
COLOR = {"LR": "#2a78d6", "R": "#eb6834", "ELU": "#1baf7a", "SILU": "#eda100", "GELU": "#e87ba4"}
BANDS = [(1, 2), (3, 7), (8, 18), (19, 29)]           # 切替 k = 課題 k → k+1。表の見出しは入る課題の番号
SEEDS = [0, 1]
nan = float("nan")

def get(act, seed, k):
    for r in rows:
        if r["act"] == act and r["seed"] == seed and r["switch"] == k: return r
def val(r, key):
    v = r[key]
    if isinstance(v, list): v = v[1]                    # 四分位の列は中央値
    return nan if v is None else float(v)
def band_stat(act, key, lo, hi, how="median"):
    per_seed = []
    for s in SEEDS:
        xs = [val(get(act, s, k), key) for k in range(lo, hi + 1)]
        xs = [x for x in xs if not math.isnan(x)]
        if xs: per_seed.append(float(np.median(xs)) if how == "median" else float(np.sum(xs)))
    return float(np.mean(per_seed)) if per_seed else nan
def band_share(act, part, lo, hi):
    per_seed = []
    for s in SEEDS:
        xs = []
        for k in range(lo, hi + 1):
            r = get(act, s, k)
            if not math.isnan(r["kappa"]) and r["kappa"] != 0: xs.append(r[part] / r["kappa"])
        if xs: per_seed.append(float(np.median(xs)))
    return float(np.mean(per_seed)) if per_seed else nan
def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "—"
    if pct:
        v = round(100 * x); return f"{0 if v == 0 else v:.0f}%"
    if x == 0: return "0"
    if abs(x) >= 100: return f"{x:.0f}"
    if abs(x) >= 1: return f"{x:.{nd}f}"
    if abs(x) >= 0.001: return f"{x:.{max(nd + 1, 4) if nd >= 3 else nd + 1}f}"
    return f"{x:.1e}"
blabel = lambda lo, hi: f"課題 {lo + 1}〜{hi + 1}"

# ---------- 全行の CSV
keys = [k for k in rows[0].keys()]
with open(OUT / "table_switch.csv", "w", newline="") as f:
    w = csv.writer(f); flat = []
    for k in keys:
        if isinstance(rows[0][k], list): flat += [f"{k}_q25", f"{k}_q50", f"{k}_q75"]
        else: flat.append(k)
    w.writerow(flat)
    for r in rows:
        line = []
        for k in keys:
            v = r[k]
            if isinstance(v, list): line += v
            else: line.append(v)
        w.writerow(line)

# ---------- 表
L = []
L.append("# κ の再装填 kappa_reload_1020（事後解析）\n")
L.append("1 隠れ層の乱数ラベル MNIST（784-100-10、Adam 1e-3・batch 16・4,000 更新/課題・30 課題）の保存済みの状態（`obsidian-research-data/push_direction_proof_0930/states`）を float64 で再計算した。"
         "値は各 seed で帯の中の切替の中央値をとり、seed 0・1 で平均した。切替ごと・seed ごとの全行は `table_switch.csv`。ReLU は課題 14〜15 で全 unit が死に、その後は測れない（—）。\n")

L.append("## 1 κ_k（課題末の読み出しが旧ラベルから学んだ量）\n")
L.append("| 活性化 | 帯 | κ_k | 学んだ読み出しの大きさ ‖κF‖/√H | 現課題の構造で説明できる読み出しの割合 | κ の分解: 特徴の変化 | 読み出しの変化 | 偶然 | 課題末の正答率 | 正解クラスの予測確率 |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for act in ACTS:
    for lo, hi in [(1, 1), (2, 2), (3, 7), (8, 18), (19, 29)]:
        lab = f"課題 {lo + 1}" if lo == hi else blabel(lo, hi)
        L.append(f"| {NAME[act]} | {lab} | {fmt(band_stat(act, 'kappa', lo, hi), 3)} | {fmt(band_stat(act, 'learned_norm', lo, hi))} | "
                 f"{fmt(band_stat(act, 'expl_total', lo, hi), pct=True)} | {fmt(band_share(act, 'kap_feat', lo, hi), pct=True)} | "
                 f"{fmt(band_share(act, 'kap_read', lo, hi), pct=True)} | {fmt(band_share(act, 'kap_chance', lo, hi), pct=True)} | "
                 f"{fmt(band_stat(act, 'acc_end', lo, hi))} | {fmt(band_stat(act, 'conf_end', lo, hi))} |")
L.append("")
L.append("κ_k の最小（切替 1〜29 のうち測れた所）と課題 1 の値:\n")
L.append("| 活性化 | seed | κ_1 | 最小 κ_k（課題） | 最大 κ_k | 最後の κ_k |")
L.append("|---|---|---|---|---|---|")
for act in ACTS:
    for s in SEEDS:
        ks = [(r["switch"], r["kappa"]) for r in rows if r["act"] == act and r["seed"] == s and r["switch"] >= 1 and not math.isnan(r["kappa"])]
        kv = np.array([k for _, k in ks]); j = int(kv.argmin())
        L.append(f"| {NAME[act]} | {s} | {kv[0]:.4f} | {kv.min():.4f}（課題 {ks[j][0] + 1}） | {kv.max():.4f} | {kv[-1]:.4f}（課題 {ks[-1][0] + 1}） |")
L.append("")

L.append("## 2 切替の押し: 符号と大きさ\n")
L.append("生きた unit（切替の時点で開いた入力が 1 つ以上）で数えた。S_exact は実際の読み出しと予測で新ラベルについて平均した押し、S_self は H1 の自己形 κ_k(1 − 1/K)Σ K_n φ′φ、GD 一歩は実際の新ラベルでの全バッチ勾配の一歩、Adam 200 は実際の最初の 200 更新の位置の変位。Adam の最初の 1 歩は、Adam の状態が保存されていないので測れない。\n")
L.append("| 活性化 | 帯 | 押しが沈める側の unit（S_exact） | 自己形と S_exact の符号一致 | GD 一歩と S_exact の一致 | Adam 200 と S_exact の一致 | Adam 200 で沈んだ unit | S_self/S_exact（中央値） | 期待の押し S_exact/N（中央値） | Adam 200 の沈み −Δm（中央値） | 沈み/幅 |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for act in ACTS:
    for lo, hi in BANDS:
        L.append(f"| {NAME[act]} | {blabel(lo, hi)} | {fmt(band_stat(act, 'frac_sink_exact', lo, hi), pct=True)} | {fmt(band_stat(act, 'agree_self_exact', lo, hi), pct=True)} | "
                 f"{fmt(band_stat(act, 'agree_exact_gd', lo, hi), pct=True)} | {fmt(band_stat(act, 'agree_exact_adam', lo, hi), pct=True)} | "
                 f"{fmt(band_stat(act, 'frac_sink_adam', lo, hi), pct=True)} | {fmt(band_stat(act, 'ratio_self_exact_q', lo, hi))} | "
                 f"{fmt(band_stat(act, 'S_exact_medN', lo, hi), 3)} | {fmt(-band_stat(act, 'dm_sw_med', lo, hi))} | {fmt(-band_stat(act, 'dm_sw_rel_med', lo, hi))} |")
L.append("")
L.append("参照: 初期値から課題 1 に入るとき（読み出しは乱数で κ = 0）:\n")
L.append("| 活性化 | 押しが沈める側の unit（S_exact） | 期待の押し S_exact/N（中央値） | Adam 200 で沈んだ unit | 課題 1 の間に p ≥ 1/2 へ上がった unit | p < 1/2 へ下がった unit |")
L.append("|---|---|---|---|---|---|")
for act in ACTS:
    r0 = [get(act, s, 0) for s in SEEDS]
    L.append(f"| {NAME[act]} | {fmt(np.mean([r['frac_sink_exact'] for r in r0]), pct=True)} | {fmt(np.mean([r['S_exact_medN'] for r in r0]), 3)} | "
             f"{fmt(np.mean([r['frac_sink_adam'] for r in r0]), pct=True)} | {' / '.join(str(r['n_p_up_next']) for r in r0)} | {' / '.join(str(r['n_p_down_next']) for r in r0)} |")
L.append("")

L.append("## 3 課題の中の戻り\n")
L.append("最初の 200 更新で沈んだ生きた unit について、課題の残り 3,800 更新の変位を見た。戻りの比 = Σ 残りの変位 / Σ |沈み|（1 なら全部戻る）。\n")
L.append("| 活性化 | 帯 | 戻りの比（総和） | unit ごとの戻りの比（中央値） | 全部戻った unit | 課題全体で位置が下がった unit | 課題末に p < 1/2 から p ≥ 1/2 へ戻った unit（帯の合計、seed 平均） |")
L.append("|---|---|---|---|---|---|---|")
for act in ACTS:
    for lo, hi in BANDS:
        L.append(f"| {NAME[act]} | {blabel(lo, hi)} | {fmt(band_stat(act, 'R_agg', lo, hi))} | {fmt(band_stat(act, 'R_q', lo, hi))} | "
                 f"{fmt(band_stat(act, 'frac_R_ge1', lo, hi), pct=True)} | {fmt(band_stat(act, 'frac_net_neg', lo, hi), pct=True)} | "
                 f"{fmt(band_stat(act, 'n_p_up_next', lo, hi, how='sum'), 1)} |")
L.append("")
L.append("一度 p < 1/2（または m < 0）に入った unit が、後のどこかの課題末に戻った数（課題末 30 点の列で数えた）:\n")
L.append("| 活性化 | seed | p < 1/2 に入った unit | そのうち p ≥ 1/2 に戻った | m < 0 に入った | そのうち m ≥ 0 に戻った | 初めて入った課題（四分位） |")
L.append("|---|---|---|---|---|---|---|")
for r in runs:
    q = "/".join(f"{x:.0f}" for x in r["p_first_task_q"])
    L.append(f"| {NAME[r['act']]} | {r['seed']} | {r['p_entered']} | {r['p_returned']} | {r['m_entered']} | {r['m_returned']} | {q} |")
L.append("")
ret = [(r["act"], r["seed"], w) for r in runs for w in r["p_returners"]]
if ret:
    L.append("戻った unit の内訳:\n")
    L.append("| 活性化 | seed | unit | 初めて p < 1/2（課題） | p ≥ 1/2 に戻った課題 | 課題 1〜6 の末の p | 課題 30 の末の p | 最後の戻りの後ずっと下 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for act, s, w in ret:
        L.append(f"| {NAME[act]} | {s} | {w['unit']} | {w['first_below_task']} | {', '.join(map(str, w['back_tasks']))} | "
                 f"{' '.join(f'{v:.2f}' for v in w['level_first6'])} | {w['level_last']:.3f} | {'はい' if w['below_after_last_return'] else 'いいえ'} |")
    L.append("")
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "summary.md", OUT / "table_switch.csv")

# ---------- 図
fam = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK KR", "Noto Sans CJK SC"):
    if cand in fam:
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "font.size": 10})
panels = [("kappa", "(a) κ_k（課題末の読み出しが旧ラベルから学んだ量）", True, 1.0),
          ("S_exact_medN", "(b) 期待の押し S/N（切替の時点、生きた unit の中央値）", True, 1.0),
          ("dm_sw_med", "(c) 実際の沈み −Δm（切替後 200 更新、中央値）", False, -1.0),
          ("R_agg", "(d) 課題の中の戻りの比 Σ戻り/Σ|沈み|（1 = 全部戻る）", False, 1.0)]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
for ax, (key, title, logy, sgn) in zip(axes.flat, panels):
    ends = []
    for act in ACTS:
        xs = list(range(2, 31)); ys = []
        for k in range(1, 30):
            v = [sgn * val(get(act, s, k), key) for s in SEEDS]
            v = [x for x in v if not math.isnan(x)]
            ys.append((np.mean(v), min(v), max(v)) if v else (nan, nan, nan))
        y = np.array(ys)
        ok = ~np.isnan(y[:, 0])
        if logy: ok &= y[:, 1] > 0
        xa = np.array(xs)[ok]
        ax.fill_between(xa, y[ok, 1], y[ok, 2], color=COLOR[act], alpha=0.15, linewidth=0)
        ax.plot(xa, y[ok, 0], color=COLOR[act], linewidth=2, label=NAME[act])
        if ok.any(): ends.append((act, xa[-1], y[ok, 0][-1]))
    if logy: ax.set_yscale("log")
    if key == "R_agg":
        ax.axhline(1.0, color="#8a8984", linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(0.0, color="#8a8984", linewidth=0.8)
    ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
    ax.set_xlabel("入る課題（課題 k → k+1 の切替を k+1 に置く）"); ax.set_xlim(1.5, 33)
    # 線の端の名前: 軸の割合の座標で縦に 0.05 以上離す
    fig.canvas.draw()
    to_frac = lambda x, yv: ax.transAxes.inverted().transform(ax.transData.transform((x, yv)))
    pts = sorted([(to_frac(x, yv)[1], act, x, yv) for act, x, yv in ends])
    placed = []
    for fy, act, x, yv in pts:
        if placed and fy - placed[-1] < 0.05: fy = placed[-1] + 0.05
        placed.append(fy)
        fx = to_frac(x, yv)[0]
        ax.annotate(NAME[act], xy=(x, yv), xycoords="data", xytext=(fx + 0.012, fy), textcoords="axes fraction",
                    fontsize=8, color="#52514e", va="center")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("κ の再装填: 1 隠れ層・乱数ラベル MNIST・Adam。線は seed 0・1 の平均、帯は 2 seed の幅。ReLU は課題 15 前後で全滅", fontsize=10, color="#52514e")
fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "kappa_push_return.png", dpi=150)
print("wrote", OUT / "kappa_push_return.png")
