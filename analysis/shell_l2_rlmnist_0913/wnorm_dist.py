"""shell_l2_rlmnist_0913 -- post-hoc REPORT (not a registered judgement): distribution of weight-tensor norms
for l2 / l2init / shell, from the task-end norms already in layer_metrics.csv.

    .venv/bin/python analysis/shell_l2_rlmnist_0913/wnorm_dist.py

Writes results/shell_l2_rlmnist_0913/posthoc_wnorm/{wnorm_dist.html, wnorm_quantiles.csv}.
Standard library + numpy/pandas only (nothing added to the experiment .venv); the figure is inline SVG.
Only tensor-level norms were logged: per-unit (row) or per-element distributions are not available.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "shell_l2_rlmnist_0913" / "layer_metrics.csv"
OUT = REPO / "results" / "shell_l2_rlmnist_0913" / "posthoc_wnorm"
REGS = ("l2", "l2init", "shell")
ACTS = (("R", "ReLU"), ("SNA", "適応 Snake"))
WEIGHTS = (("W1", "784→100"), ("W2", "100→100"), ("W3", "100→10"))
WIN = (31, 50)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    step = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(step))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= step), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def dist_panel(win: pd.DataFrame, tensor: str) -> str:
    """Box (quartiles, whiskers = min/max) plus the 200 seed x task values as a jittered strip."""
    W, Hh = 300, 220
    L, R, T, B = 44, 10, 12, 30
    vals = {r: win[win.reg == r].norm_ratio.to_numpy(float) for r in REGS}
    lo = min(v.min() for v in vals.values())
    hi = max(v.max() for v in vals.values())
    pad = (hi - lo) * 0.08
    ticks = nice_ticks(lo - pad, hi + pad, 4)
    y0, y1 = min(ticks[0], lo - pad), max(ticks[-1], hi + pad)

    def y(v):
        return T + (y1 - v) / (y1 - y0) * (Hh - T - B)

    band = (W - L - R) / len(REGS)
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{tensor} の ‖W‖/‖W0‖ 分布">']
    for t in ticks:
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{fmt(t)}</text>')
    rng = np.random.default_rng(13)
    for i, r in enumerate(REGS):
        v = np.sort(vals[r])
        cx = L + band * (i + 0.5)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        jit = rng.uniform(-band * 0.30, band * 0.30, len(v))
        for xv, vv in zip(jit, v):
            s.append(f'<circle class="pt {r}" cx="{cx + xv:.1f}" cy="{y(vv):.1f}" r="1.6"/>')
        bw = band * 0.26
        s.append(f'<line class="whisk" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(v.max()):.1f}" y2="{y(v.min()):.1f}"/>'
                 f'<rect class="box {r}" x="{cx - bw / 2:.1f}" y="{y(q3):.1f}" width="{bw:.1f}" height="{max(y(q1) - y(q3), 1):.1f}"/>'
                 f'<line class="med" x1="{cx - bw / 2:.1f}" x2="{cx + bw / 2:.1f}" y1="{y(med):.1f}" y2="{y(med):.1f}"/>'
                 f'<text class="xlab {r}" x="{cx:.1f}" y="{Hh - 10}" text-anchor="middle">{r}</text>')
    s.append("</svg>")
    return "".join(s)


def traj_panel(sub: pd.DataFrame, tensor: str, ymax: float) -> str:
    """Median over the 10 seeds at each task end, with the seed min-max band; window 31-50 shaded."""
    W, Hh = 300, 200
    L, R, T, B = 44, 10, 12, 30
    ticks = nice_ticks(0, ymax, 4)
    y1 = max(ticks[-1], ymax)

    def x(t):
        return L + t / 50 * (W - L - R)

    def y(v):
        return T + (y1 - v) / y1 * (Hh - T - B)

    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{tensor} の ‖W‖/‖W0‖ のタスク推移">',
         f'<rect class="win" x="{x(WIN[0]):.1f}" y="{T}" width="{x(WIN[1]) - x(WIN[0]):.1f}" height="{Hh - T - B}"/>']
    for t in ticks:
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{fmt(t)}</text>')
    s.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{y(1):.1f}" y2="{y(1):.1f}"/>')
    for t in (0, 10, 20, 30, 40, 50):
        s.append(f'<text class="tick" x="{x(t):.1f}" y="{Hh - 14}" text-anchor="middle">{t}</text>')
    s.append(f'<text class="tick" x="{W - R}" y="{Hh - 2}" text-anchor="end">タスク</text>')
    for r in REGS:
        g = sub[sub.reg == r].groupby("task").norm_ratio
        med, mn, mx = g.median(), g.min(), g.max()
        ts = med.index.to_numpy()
        up = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(ts, mx))
        dn = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(ts[::-1], mn.to_numpy()[::-1]))
        s.append(f'<polygon class="band {r}" points="{up} {dn}"/>')
        s.append(f'<polyline class="line {r}" points="{" ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(ts, med))}"/>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--ground:#f5f7f8;--panel:#ffffff;--ink:#1c2631;--muted:#5a6773;--rule:#d6dde3;--win:#e9eef2;
--l2:#b4532a;--l2init:#2f66a8;--shell:#2d8467;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#11161b;--panel:#171e25;--ink:#e2e8ed;
--muted:#95a2ad;--rule:#2b3640;--win:#1d262e;--l2:#e08a5e;--l2init:#72a4df;--shell:#5fc19c;}}
:root[data-theme="dark"]{--ground:#11161b;--panel:#171e25;--ink:#e2e8ed;--muted:#95a2ad;--rule:#2b3640;--win:#1d262e;
--l2:#e08a5e;--l2init:#72a4df;--shell:#5fc19c;}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans JP","Hiragino Sans","Noto Sans JP",system-ui,sans-serif;
font-size:15px;line-height:1.7;margin:0;padding-inline:20px;padding-block:32px 56px}
.wrap{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:36px}
header{display:flex;flex-direction:column;gap:10px;max-width:68ch}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
h1{font-size:28px;line-height:1.3;margin:0;font-weight:600;text-wrap:balance}
h2{font-size:20px;margin:0;font-weight:600;text-wrap:balance}
h3{font-size:14px;margin:0;font-weight:600}
p{margin:0}
.muted{color:var(--muted)}
.mono,.tick,.xlab,td.num{font-family:"IBM Plex Mono",ui-monospace,monospace}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:12px;height:12px;border-radius:2px;display:inline-block}
.sw.l2{background:var(--l2)}.sw.l2init{background:var(--l2init)}.sw.shell{background:var(--shell)}
section{display:flex;flex-direction:column;gap:14px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 12px 6px;display:flex;flex-direction:column;gap:4px}
.panel .cap{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.panel .cap .mono{font-size:12px;color:var(--muted)}
.rowlab{font-size:13px;color:var(--muted)}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--muted);font-size:10px}
.xlab{font-size:11px;font-weight:600}
.xlab.l2{fill:var(--l2)}.xlab.l2init{fill:var(--l2init)}.xlab.shell{fill:var(--shell)}
.pt{fill-opacity:.35}.pt.l2{fill:var(--l2)}.pt.l2init{fill:var(--l2init)}.pt.shell{fill:var(--shell)}
.box{fill-opacity:.18;stroke-width:1.5}.box.l2{fill:var(--l2);stroke:var(--l2)}.box.l2init{fill:var(--l2init);stroke:var(--l2init)}.box.shell{fill:var(--shell);stroke:var(--shell)}
.med{stroke:var(--ink);stroke-width:2}
.whisk{stroke:var(--muted);stroke-width:1}
.win{fill:var(--win)}
.ref{stroke:var(--ink);stroke-width:1;stroke-dasharray:3 3;opacity:.6}
.band{fill-opacity:.16;stroke:none}.band.l2{fill:var(--l2)}.band.l2init{fill:var(--l2init)}.band.shell{fill:var(--shell)}
.line{fill:none;stroke-width:1.8}.line.l2{stroke:var(--l2)}.line.l2init{stroke:var(--l2init)}.line.shell{stroke:var(--shell)}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
tr:last-child td{border-bottom:none}
td.reg.l2{color:var(--l2)}td.reg.l2init{color:var(--l2init)}td.reg.shell{color:var(--shell)}
.tabs{display:flex;gap:8px;border-bottom:1px solid var(--rule);flex-wrap:wrap}.tabin{position:absolute;opacity:0;pointer-events:none}.tab{cursor:pointer;padding:8px 16px 9px;border:1px solid transparent;border-bottom:none;border-radius:6px 6px 0 0;font-weight:600;font-size:16px;color:var(--muted);display:inline-flex;gap:8px;align-items:baseline;margin-bottom:-1px}.tab .tabsub{font-weight:400;font-size:12px}.tab:hover{color:var(--ink)}.tabin:checked+.tab{color:var(--ink);background:var(--panel);border-color:var(--rule)}.tabin:focus-visible+.tab{outline:2px solid var(--l2init);outline-offset:2px}.pane{display:none;flex-direction:column;gap:36px}.wrap:has(#tab-R:checked) #pane-R,.wrap:has(#tab-SNA:checked) #pane-SNA{display:flex}
.note{font-size:13px;color:var(--muted);max-width:75ch;display:flex;flex-direction:column;gap:6px}
@media (max-width:760px){.grid3{grid-template-columns:minmax(0,1fr)}h1{font-size:23px}}
"""


def main() -> None:
    lm = pd.read_csv(SRC)
    w = lm[lm.tensor.isin([t for t, _ in WEIGHTS]) & lm.reg.isin(REGS)]
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for act, _ in ACTS:
        for tensor, _ in WEIGHTS:
            for r in REGS:
                x = w[(w.act == act) & (w.tensor == tensor) & (w.reg == r) & w.task.between(*WIN)]
                q = np.percentile(x.norm_ratio, [0, 25, 50, 75, 100])
                qa = np.percentile(x.norm, [0, 25, 50, 75, 100])
                rows.append({"act": act, "tensor": tensor, "reg": r, "n": len(x),
                             "norm0_median": float(x.norm0.median()),
                             **{f"ratio_{k}": float(v) for k, v in zip(("min", "q25", "median", "q75", "max"), q)},
                             **{f"norm_{k}": float(v) for k, v in zip(("min", "q25", "median", "q75", "max"), qa)}})
    qdf = pd.DataFrame(rows)
    qdf.to_csv(OUT / "wnorm_quantiles.csv", index=False, float_format="%.6g")

    H = ['<title>Shell 実験の重みノルム分布</title>',
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600'
         '&family=IBM+Plex+Sans+JP:wght@400;600&display=swap">',
         f"<style>{CSS}</style>", '<div class="wrap"><header>',
         '<div class="eyebrow">shell_l2_rlmnist_0913 · 事後の REPORT（登録判定ではない）</div>',
         '<h1>l2・l2init・shell の重みテンソルの大きさ</h1>',
         '<p>Random Label MNIST（784–100–100–10・Adam lr 1e−3・λ 1e−3・50 タスク）。各タスク末の '
         '<span class="mono">‖W‖/‖W0‖</span>（初期値からの倍率）を W1–W3 ごとに示す。'
         '上段は窓（タスク 31–50）の 10 seed × 20 タスク = 200 値の分布、下段はタスク推移（seed の中央値と最小–最大）。</p>',
         '<div class="legend"><span><i class="sw l2"></i>l2（0 へ引く）</span>'
         '<span><i class="sw l2init"></i>l2init（p0 へ引く）</span><span><i class="sw shell"></i>shell（半径を ‖p0‖ へ引く）</span>'
         '<span class="muted">破線 = 初期値（×1）　灰帯 = 窓 31–50</span></div></header>']
    # one tab per activation (CSS-only radio tabs; R open on load)
    H.append('<div class="tabs" role="tablist">')
    for i, (act, actname) in enumerate(ACTS):
        H.append(f'<input type="radio" name="act" id="tab-{act}" class="tabin"{" checked" if i == 0 else ""}>'
                 f'<label for="tab-{act}" class="tab" role="tab">{act}<span class="tabsub">{actname}</span></label>')
    H.append('</div>')
    for act, actname in ACTS:
        sub = w[w.act == act]
        H.append(f'<div class="pane" id="pane-{act}" role="tabpanel"><section><h2>{act}（{actname}）</h2>')
        H.append('<div class="rowlab">窓内の分布（縦軸はパネルごとにデータに合わせている）</div><div class="grid3">')
        for tensor, shape in WEIGHTS:
            win = sub[(sub.tensor == tensor) & sub.task.between(*WIN)]
            n0 = float(win.norm0.median())
            H.append(f'<div class="panel"><div class="cap"><h3>{tensor}</h3><span class="mono">{shape} · ‖W0‖≈{n0:.2f}</span></div>'
                     f'{dist_panel(win, tensor)}</div>')
        H.append('</div><div class="rowlab">タスク推移（縦軸は 0 から）</div><div class="grid3">')
        for tensor, shape in WEIGHTS:
            t_sub = sub[sub.tensor == tensor]
            ymax = float(t_sub.norm_ratio.max()) * 1.05
            H.append(f'<div class="panel"><div class="cap"><h3>{tensor}</h3><span class="mono">‖W‖/‖W0‖</span></div>'
                     f'{traj_panel(t_sub, tensor, ymax)}</div>')
        H.append('</div></section>')
        H.append(f'<section><h2>{act} の窓内分位点（‖W‖/‖W0‖）</h2><div class="tablewrap"><table><thead><tr>'
                 '<th>tensor</th><th>reg</th><th>‖W0‖</th><th>最小</th><th>25%</th><th>中央値</th><th>75%</th><th>最大</th>'
                 '<th>‖W‖ 中央値</th></tr></thead><tbody>')
        for rw in (r for r in rows if r["act"] == act):
            H.append(f'<tr><td>{rw["tensor"]}</td><td class="reg {rw["reg"]}">{rw["reg"]}</td>'
                     f'<td class="num">{rw["norm0_median"]:.3f}</td>'
                     + "".join(f'<td class="num">×{rw[f"ratio_{k}"]:.3f}</td>' for k in ("min", "q25", "median", "q75", "max"))
                     + f'<td class="num">{rw["norm_median"]:.3f}</td></tr>')
        H.append('</tbody></table></div></section></div>')
    H.append('<div class="note"><p>数値の出所: <span class="mono">results/shell_l2_rlmnist_0913/layer_metrics.csv</span>'
             '（タスク末の float32 重みから float64 で再計算したテンソルのノルム）。分位点は同じフォルダの '
             '<span class="mono">wnorm_quantiles.csv</span>。</p>'
             '<p>記録しているのはテンソル単位のノルムだけで、ユニット（行）ごとや要素ごとの分布は残していない。'
             'それを見るには重みを保存して走り直す必要がある。</p>'
             '<p>この図は登録判定の外の REPORT。norm-match ガード（W の中央値の比 ∈ [0.9, 1.1]）の判定は '
             '<span class="mono">summary.md</span> §4 のとおり: R は W3 で 0.8975（FAIL）、SNA は W3 で 0.9010（PASS）。</p></div></div>')
    (OUT / "wnorm_dist.html").write_text("\n".join(H))
    print(f"wrote {OUT}/wnorm_dist.html and wnorm_quantiles.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
