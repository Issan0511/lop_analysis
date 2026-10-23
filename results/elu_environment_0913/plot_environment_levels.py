#!/usr/bin/env python3
"""Plot registered endpoint accuracy / layer2 low-gate occupancy; no new experiment.

Run with the experiment Python environment:
python results/elu_environment_0913/plot_environment_levels.py
Sources: rows.csv, units.npz, provenance.json in this directory.
Each plotted point is the mean across three seeds. Low-gate occupancy is first
averaged across the 100 layer2 units (each stored value already averages inputs).
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "rows.csv").open()))
units = np.load(HERE / "units.npz")
models = json.loads((HERE / "provenance.json").read_text())["models"]
tasks = np.arange(1, 51)
assert len(rows) == 1200 and len(models) == 24
styles = {"ELU1": ("#d66535", "ELU (alpha=1)"), "LR": ("#3273a8", "Leaky (0.1)")}
series = {}

for env in ("PM", "RL"):
    for act in ("ELU1", "LR"):
        for iv in ("ref", "wclamp"):
            chosen = [r for r in rows if r["env"] == env and r["act"] == act and r["iv"] == iv]
            chosen.sort(key=lambda r: (int(r["seed"]), int(r["task"])))
            assert len(chosen) == 150
            endpoint = np.array([float(r["train_acc"]) for r in chosen]).reshape(3, 50)
            ix = [i for i, m in enumerate(models) if m["env"] == env and m["act"] == act and m["iv"] == iv]
            assert len(ix) == 3
            lowgate = np.stack([units[f"lowgate_l2_t{t}"][ix].mean(axis=1) for t in tasks], axis=1)
            assert endpoint.shape == lowgate.shape == (3, 50)
            assert np.isfinite(endpoint).all() and np.isfinite(lowgate).all()
            series[env, act, iv] = (100 * endpoint.mean(axis=0), 100 * lowgate.mean(axis=0))

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 15})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey="row")
for col, env in enumerate(("PM", "RL")):
    for act in ("ELU1", "LR"):
        for iv in ("ref", "wclamp"):
            color, name = styles[act]
            linestyle = "-" if iv == "ref" else "--"
            for row in range(2):
                axes[row, col].plot(tasks, series[env, act, iv][row], color=color,
                                    linestyle=linestyle, linewidth=2.2,
                                    label=f"{name}, " + ("control" if iv == "ref" else "W1 clamp"))
    for row in range(2):
        ax = axes[row, col]
        ax.axvline(11, color="#555555", linestyle=":", linewidth=1.5, zorder=0)
        ax.set_xlim(1, 50)
        ax.set_ylim(0, 102)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelleft=True)
    axes[0, col].set_title("Permuted MNIST (PM)" if env == "PM" else "Random-label MNIST (RL)", pad=12)
    axes[1, col].set_xlabel("Task")

axes[0, 0].set_ylabel("Endpoint training accuracy (%)")
axes[1, 0].set_ylabel("Layer 2 low-gate occupancy (%)\n[activation derivative < 0.05]")
axes[0, 0].text(.49, .51, "Accuracy is at the ceiling", transform=axes[0, 0].transAxes,
                 color="#444444", ha="center", fontsize=12)
axes[0, 1].annotate("ELU declines before\nintervention starts",
                   xy=(8, series["RL", "ELU1", "ref"][0][7]),
                   xytext=(21, 36), color="#d66535", fontsize=12,
                   arrowprops=dict(arrowstyle="->", color="#d66535", lw=1.2))
axes[1, 1].text(.57, .60, "L2 gates are observed;\nonly W1 is constrained",
               transform=axes[1, 1].transAxes, ha="center", color="#444444", fontsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([], [], color="#555555", linestyle=":", linewidth=1.5))
labels.append("Intervention begins: task 11")
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(.5, .057), fontsize=10.5, columnspacing=1.6, handlelength=3)
fig.suptitle("Endpoint performance and layer 2 gates across environments", y=.98, fontsize=17)
fig.text(.5, .935, "Mean of 3 seeds | Same 1,200 images, MLP and 6,000 updates per task",
         ha="center", fontsize=11.5, color="#444444")
fig.text(.5, .018, "The clamp fixes centered W1 row norms from task 11. It is not an intervention on layer 2.",
         ha="center", fontsize=11, color="#444444")
fig.subplots_adjust(left=.095, right=.98, top=.865, bottom=.205, wspace=.20, hspace=.22)
fig.savefig(HERE / "environment_levels.png", dpi=180, facecolor="white")
plt.close(fig)
print(HERE / "environment_levels.png")
