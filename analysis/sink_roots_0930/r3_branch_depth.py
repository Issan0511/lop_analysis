#!/usr/bin/env python3
"""Round 3 R2p_branch_depth (spec_sink_roots_0930_round3.md): branches from the same switch state (branch_probe.py of the
parent's derivation), depth measured over the whole task (the request: '深さは課題全体の最小で測る').
Per switch and arm, over units alive at the switch (k > 0): depth = median of the trajectory minimum of m - m(switch)
(10-update resolution, the arm's whole task: 4,000 updates, 16,000 for 'long'); turn = median time of the minimum;
return / depth = median(end - min) / -depth (end = the arm's last record).  Ratios against 'base' of the same switch."""
import glob, json, re, sys
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round3/R2p/branches")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
ARMS = ("base", "b2_0.9", "b2_0.99", "b2_0.9999", "v10", "v01", "sgd_0.01", "sgd_0.1", "long")


def main():
    out = {}; L = ["act seed switch arm | depth (ratio to base) | turn (ratio) | return/depth"]
    for fn in sorted(glob.glob(str(RAW / "branch_*_s*_sw*.npz"))):
        act, seed = re.match(r"branch_(\w+?)_s(\d+)_", Path(fn).name).groups()
        d = np.load(fn)
        for t in sorted({int(k.split("_")[0][1:]) for k in d.files}):
            live = d[f"t{t}_k"] > 0; res = {}
            for arm in ARMS:
                if f"t{t}_{arm}" not in d.files:
                    continue
                tr = d[f"t{t}_{arm}"][:, live].astype(float)
                mn = tr.min(0); tmin = (tr.argmin(0) + 1) * 10
                res[arm] = dict(depth=float(np.median(mn)), turn=float(np.median(tmin)), ret=float(np.median(tr[-1] - mn)))
            for arm, r in res.items():
                b = res["base"]
                r.update(depth_ratio=r["depth"] / b["depth"], turn_ratio=r["turn"] / b["turn"], ret_over_depth=r["ret"] / -r["depth"])
                L.append(f"{act:4s} s{seed} sw{t:2d} {arm:9s} | {r['depth']:+.3f} ({r['depth_ratio']:.2f}) | {r['turn']:6.0f} ({r['turn_ratio']:.2f}) | {r['ret_over_depth']:.2f}")
            out[f"{act}_s{seed}_sw{t}"] = res
    # the registered reading of the predictions
    L.append("")
    for act in ("ELU", "GELU", "SILU", "LR"):
        keys = [k for k in out if k.startswith(act + "_")]
        if not keys:
            continue
        if act == "ELU":
            rr = [out[k][a]["depth_ratio"] for k in keys for a in out[k] if a != "base"]
            L.append(f"ELU: depth ratio to base over all arms/switches/seeds: min {min(rr):.2f} max {max(rr):.2f}; share in [0.8, 1.25] {np.mean([(0.8 <= x <= 1.25) for x in rr]):.2f} (n {len(rr)})")
        if act in ("GELU", "SILU"):
            tq = [out[k]["b2_0.9"]["turn_ratio"] for k in keys if "b2_0.9" in out[k]]
            rq = [(out[k]["b2_0.9"]["ret_over_depth"], out[k]["base"]["ret_over_depth"]) for k in keys if "b2_0.9" in out[k]]
            L.append(f"{act}: beta2 0.9 turn ratio {np.round(tq, 2).tolist()} (>= 2 in {sum(x >= 2 for x in tq)}/{len(tq)}); "
                     f"return/depth b2 0.9 vs base {[(round(a, 2), round(b, 2)) for a, b in rq]} (smaller in {sum(a < b for a, b in rq)}/{len(rq)})")
    txt = "\n".join(L); print(txt)
    (RES / "round3_R2p_branch_depth.txt").write_text(txt + "\n")
    (RES / "round3_R2p_branch_depth.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
