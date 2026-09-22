#!/usr/bin/env python3
"""V11 原稿の作図規約（図表一覧 §2.5・2026-09-21 決定）を 1 か所に集めたもの。

全図がここから色・ラベル・体裁を引く。ラベルは日本語で作り、9/28-29 の英語化では
LABEL / AXIS / TITLE の辞書だけを差し替える（§2.5-9）。

規約の要点（§2.5）:
  1. 集約は走の登録判定に合わせる  -> agg="median" / "mean" を図ごとに渡す
  2. 散らばりは seed の全範囲。信頼区間ではない -> band() が必ず全範囲を描く
  3. online accuracy の y 軸は (0, 1.02) 固定 -> acc_axis()
  4. 登録窓は薄い帯 -> window_span()
  5. 登録ラベルと格を全図に -> grade()
  6. 検定の数値は図に書かない（キャプションへ）
  7. 5+1 は hard/easy を分ける -> HARD_5P1 / EASY_5P1
  8. 順序のある腕は連続色・独立な腕は Okabe-Ito -> ladder() / OKABE
 10. 寸法と出力は既存慣行 -> save()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "v11_figures_0921"

# --- 体裁（§2.5-10）------------------------------------------------------
for _f in ("Noto Sans CJK JP", "Noto Sans JP", "IPAGothic", "DejaVu Sans"):
    if any(_f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 9,            # 個別指定はせず、ここだけで段組に合わせる
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# --- 色（§2.5-8）---------------------------------------------------------
# 独立な腕: Okabe-Ito（色覚安全）
OKABE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
GREY = "#7a7a7a"


def ladder(n: int, cmap: str = "viridis", lo: float = 0.12, hi: float = 0.88) -> list[str]:
    """順序のある腕のための連続色。梯子の段数 n を受けて濃淡を返す。"""
    m = matplotlib.colormaps[cmap]
    return [matplotlib.colors.to_hex(m(t)) for t in np.linspace(lo, hi, n)]


# 家族ごとに 1 つの連続色の梯子を割り当て、家族の間は色相で分ける（§2.5-8）。
#   扉（C→H→CH→CHB）= 青の梯子 / Snake・kunekune = 緑の梯子 / leaky = 黄土の梯子
#   基準（ref・R）= 灰 / 別種の正規化・平滑な活性化 = Okabe-Ito の独立色
_DOORS = ladder(4, "Blues", 0.34, 0.97)
_SNAKE = ladder(4, "Greens", 0.45, 0.92)
_LEAKY = ladder(4, "YlOrBr", 0.38, 0.86)
COLOR = {
    # 基準
    "ref": GREY, "R": GREY,
    # 図 2・3・4: 中心化の扉（梯子）と、別種の正規化（独立色・破線）
    "C": _DOORS[0], "H": _DOORS[1], "CH": _DOORS[2], "CHB": _DOORS[3],
    # CH と CHB は水準がほぼ重なるので、bias の扱いを変えた CHB は破線にして下の線を見せる
    "CHB0": OKABE["sky"], "CS": OKABE["purple"], "LN": OKABE["vermillion"],
    # 図 7b: 成長上限の梯子
    "cap1": None, "cap2": None, "cap12": None, "cap12_bfix": None,
    # 図 8・10: 活性化。Snake 族と leaky 族はそれぞれ梯子、平滑な活性化は独立色
    "SNA": _SNAKE[0], "KKA": _SNAKE[1], "KKA23": _SNAKE[2], "KKT1": _SNAKE[3],
    "LK001": _LEAKY[0], "LR": _LEAKY[1], "LK03": _LEAKY[2], "LK07": _LEAKY[3],
    "ELU": "#E15759", "SILU": "#B07AA1", "GELU": "#9467BD",
    "SL": "#7E9FD4", "RSL": "#3F6FB5",
    "DF": "#9C6B4F", "CR": "#5A3D2B",
    "l2init": "#C9A227",
}
_CAP_LADDER = ladder(4, "plasma", 0.15, 0.80)
for _k, _c in zip(("cap1", "cap2", "cap12", "cap12_bfix"), _CAP_LADDER):
    COLOR[_k] = _c

# --- ラベル（§2.5-9・英語化はこの辞書だけ差し替える）----------------------
LABEL = {
    "ref": "ref（素の ReLU）", "C": "C（入力の中心化）", "H": "H（中間層の中心化）",
    "CH": "CH（入力＋中間層）", "CHB": "CHB（＋bias）", "CHB0": "CHB0（bias 除去）",
    "CS": "CS（分散だけ）", "LN": "LN（素の LayerNorm）",
    "cap1": "cap1（第 1 層の上限）", "cap2": "cap2（第 2 層の上限）",
    "cap12": "cap12（両層の上限）", "cap12_bfix": "cap12＋bias 固定",
    "R": "R（ReLU）", "l2init": "L2 Init",
}
AXIS = {
    "task": "課題",
    "online": "オンライン精度",
    "zbar2": "第 2 層の $\\bar z_2$",
    "zsd2": "第 2 層の $\\mathrm{sd}_2$",
    "dead2": "第 2 層の死亡率",
    "bias2": "$|b_2| / \\mathrm{sd}_2$",
    "effrank": "実効階数",
    "gate0": "gate が厳密に 0 の割合",
    "mu2": "$\\|\\mu_2\\|$",
    "window": "後期窓のオンライン精度",
    "gap": "fresh gap",
}

# 格（§2.5-5）
GRADE = {
    "registered": "登録",
    "column": "登録列の読み",
    "posthoc": "事後",
}

SEED_BAND_NOTE = "帯は seed の全範囲（信頼区間ではない）"

# 線種: 梯子の腕は実線、別種の介入（対照）は破線。重なったときに読めるようにする。
LS = {"CHB": "--", "CS": "--", "LN": "--", "CHB0": ":", "cap12_bfix": "--"}


# --- 描画のヘルパ --------------------------------------------------------
def band(ax, x, Y, color, label=None, agg="median", lw=1.7, ls="-", alpha=0.16, zorder=2):
    """seed × 課題 の行列 Y を、集約線＋seed 全範囲の帯で描く（§2.5-1, 2）。

    agg は走の登録判定に合わせて "median" か "mean" を渡す。
    """
    Y = np.asarray(Y, float)
    assert Y.ndim == 2, Y.shape
    centre = np.median(Y, 0) if agg == "median" else Y.mean(0)
    ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(x, centre, color=color, lw=lw, ls=ls, label=label, zorder=zorder)
    return centre


def seeds_lines(ax, x, Y, color, label=None, lw=0.8, alpha=0.75, ls="-"):
    """seed ごとに転移の時刻が違い、集約が形を壊す図だけ（§2.5-2 の例外・図 3）。"""
    Y = np.asarray(Y, float)
    for i, row in enumerate(Y):
        ax.plot(x, row, color=color, lw=lw, ls=ls, alpha=alpha, label=label if i == 0 else None)


def acc_axis(ax, label=True):
    """online accuracy の軸は必ず 0-1.02 固定。autoscale しない（§2.5-3）。"""
    ax.set_ylim(0, 1.02)
    if label:
        ax.set_ylabel(AXIS["online"])


def window_span(ax, lo, hi, text=None):
    """登録窓の帯（§2.5-4）。"""
    ax.axvspan(lo, hi, color="#222222", alpha=0.035, zorder=-5, lw=0)
    if text:
        ax.annotate(text, xy=((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="#555555")


def grade(ax, kind, label=None, loc="lower right"):
    """登録ラベルと格をパネルの隅に（§2.5-5）。kind は GRADE のキー。"""
    txt = GRADE[kind] if label is None else f"{GRADE[kind]} · {label}"
    face = {"registered": "#e8efe6", "column": "#eef0f4", "posthoc": "#f6efe4"}[kind]
    edge = {"registered": "#7fa06f", "column": "#8b93a5", "posthoc": "#c39a52"}[kind]
    xy = {"lower right": (0.985, 0.03), "lower left": (0.015, 0.03),
          "upper right": (0.985, 0.965), "upper left": (0.015, 0.965),
          "center right": (0.985, 0.5), "center left": (0.015, 0.5)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "lower" in loc else ("top" if "upper" in loc else "center")
    ax.annotate(txt, xy=xy, xycoords="axes fraction", ha=ha, va=va, fontsize=7,
                color="#33383f", zorder=20,
                bbox=dict(boxstyle="round,pad=0.28", fc=face, ec=edge, lw=0.6))


# --- 5+1 の hard/easy（§2.5-7）-------------------------------------------
N_TASKS_5P1 = 30
HARD_5P1 = list(range(1, N_TASKS_5P1 + 1, 2))     # 5 クラス CIFAR
EASY_5P1 = list(range(2, N_TASKS_5P1 + 1, 2))
EARLY_5P1, LATE_5P1 = HARD_5P1[:5], HARD_5P1[10:]  # src/cifar5p1_mlp_0920_report.py:31


def value_labels(ax, y, values, fmt="{:.3f}", flip_at=0.80, dx=0.012, color="#555555", at=None):
    """水平の順位図で、点のそばに数値を置く（§2.5-3 で軸を固定したまま順位を読めるように）。

    値が詰まっている帯では点の位置から順位が読めないので、数値そのものを添える。
    右端に寄った点は左側に出す。`at` を渡すと、字を置く位置だけ別にできる
    （棒の版で、棒の端ではなく seed 範囲の右端の外に出すため）。
    """
    lo, hi = ax.get_xlim()
    span = hi - lo
    at = values if at is None else at
    for yi, v, a in zip(y, values, at):
        right = v < flip_at
        ax.annotate(fmt.format(v), xy=(a + (dx if right else -dx) * span, yi),
                    ha="left" if right else "right", va="center",
                    fontsize=7.5, color=color, family=plt.rcParams["font.family"],
                    zorder=6)


def breathe(ax, axis="y", frac=0.045):
    """端の値が軸線に張り付くと読めないので、両端に余白を入れる。"""
    get, set_ = (ax.get_ylim, ax.set_ylim) if axis == "y" else (ax.get_xlim, ax.set_xlim)
    lo, hi = get()
    pad = (hi - lo) * frac
    set_(lo - pad, hi + pad)


def save(fig, name: str, note: str | None = None, out: Path | None = None):
    """PNG と PDF の両方（§2.5-10）と、図ごとの figure_notes。"""
    out = OUT if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    if note:
        (out / f"{name}.txt").write_text(note.rstrip() + "\n", encoding="utf-8")
    plt.close(fig)
    return out / f"{name}.png"


def grid(nrow, ncol, w=13.0, h=4.5, **kw):
    fig, axes = plt.subplots(nrow, ncol, figsize=(w, h), layout="constrained", **kw)
    return fig, np.atleast_1d(axes).ravel() if nrow * ncol > 1 else [axes]
