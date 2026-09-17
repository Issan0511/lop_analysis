"""Plot independent observed-versus-predicted updates and numerical residuals."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/aq_identity_0917"
RAW = Path(json.loads((OUT/"provenance.json").read_text())["raw_directory"])
rng = np.random.default_rng(20260917)
parts = {"float64": [], "float32": []}
for dtype in parts:
    for path in sorted(RAW.glob(f"*_{dtype}_frozen.npz")):
        with np.load(path) as z:
            prefix = "lr0.005_row_centered_"
            actual = z[prefix+"observed"].ravel()
            predicted = z[prefix+"predicted"].ravel()
            bound = z[prefix+"tol"].ravel()
            ix = rng.choice(actual.size, size=min(1000, actual.size), replace=False)
            parts[dtype].append(np.column_stack([predicted[ix], actual[ix], bound[ix]]))
    parts[dtype] = np.concatenate(parts[dtype])

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
fig.suptitle("CondA: does the A/Q formula match actual SGD updates?", fontsize=16)
p, a, b = parts["float64"].T
lim = max(np.abs(p).max(), np.abs(a).max())*1.15
ax[0].plot([-lim, lim], [-lim, lim], color="#9ca3af", lw=2, label="Exact agreement")
ax[0].scatter(p, a, s=7, alpha=.4, color="#2563eb", rasterized=True, label="Float64 probes")
ax[0].set(xscale="symlog", yscale="symlog", xlabel=r"Predicted $-2\eta A+\eta^2Q$",
          ylabel=r"Observed $\|W'_c\|^2-\|W_c\|^2$", title="Independent norm differences")
ax[0].set_xscale("symlog", linthresh=1e-9)
ax[0].set_yscale("symlog", linthresh=1e-9)
ax[0].legend(loc="upper left", fontsize=9)
ax[0].grid(alpha=.18)
for dtype, color in [("float64", "#2563eb"), ("float32", "#dc6b26")]:
    p, a, b = parts[dtype].T
    ratio = np.sort(np.maximum(np.abs(a-p)/b, 1e-10))
    ax[1].plot(ratio, np.arange(1,len(ratio)+1)/len(ratio), label=dtype, color=color, lw=2)
ax[1].axvline(1, color="#991b1b", ls="--", label="Registered error bound")
ax[1].set(xscale="log", xlabel="Absolute error / rounding-aware bound",
          ylabel="Cumulative fraction", title="Finite-precision discrepancies")
ax[1].set_xlim(1e-10, 2)
ax[1].set_ylim(0,1.02)
ax[1].legend(fontsize=9)
ax[1].grid(alpha=.18)
fig.get_layout_engine().set(rect=(0,.06,1,1))
fig.text(.5,.025,"Row-centered weights · learning rate 0.005 · sampled checkpoint / input / unit events",
         ha="center",fontsize=10,color="#4b5563")
fig.savefig(OUT/"identity_check.png",dpi=180)
fig.savefig(OUT/"identity_check.pdf")
