"""spec §2.2: A2（α = 0.01 の C3-2、seed 0〜9）の保存 checkpoint から、課題の中の up 行の変位（更新 100 → 10,000）と、課題末どうしの回転を出す（全 512 行）。
読むだけ（wt/chargpt_cl_1019 の ckpts）。学習なし。使い方: python -I a2_motion.py <out_json>"""
import json, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
R = Path("/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/M2/C3")
T_J4 = list(range(2, 11)) + list(range(41, 51)) + list(range(91, 101))
def W(p, l):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return sd[f"h.{l}.mlp.up.weight"].double()
out = {}
for s in range(10):
    ck = R / f"seed{s}" / "ckpts"
    rec = {}
    for t in T_J4:
        for l in (0, 1):
            a, b = W(ck / f"u100_task{t}.pt", l), W(ck / f"task{t}.pt", l)
            d = (b - a).norm(dim=1)
            rec[f"t{t}_L{l}_disp100_med"] = float(d.median())
            rec[f"t{t}_L{l}_rel100_med"] = float((d / a.norm(dim=1)).median())
            rec[f"t{t}_L{l}_norm_end_med"] = float(b.norm(dim=1).median())
            if (ck / f"task{t-1}.pt").exists():
                c = W(ck / f"task{t-1}.pt", l)
                cos = (b * c).sum(1) / (b.norm(dim=1) * c.norm(dim=1))
                rec[f"t{t}_L{l}_rot_med"] = float((1 - cos).median())
                rec[f"t{t}_L{l}_dispend_med"] = float((b - c).norm(dim=1).median())
    out[s] = rec
json.dump(out, open(sys.argv[1], "w"), indent=0)
def win(s, key, ts): return np.median([out[s][f"t{t}_{key}"] for t in ts if f"t{t}_{key}" in out[s]])
for l in (0, 1):
    print(f"== L{l}")
    for name, ts in (("2-10", range(2, 11)), ("41-50", range(41, 51)), ("91-100", range(91, 101))):
        for key in (f"L{l}_disp100_med", f"L{l}_rel100_med", f"L{l}_norm_end_med", f"L{l}_rot_med", f"L{l}_dispend_med"):
            v = np.array([win(s, key, ts) for s in range(10)])
            print(f"  {name} {key}: median {np.median(v):.4f} range [{v.min():.4f}, {v.max():.4f}]")
    v = np.array([np.median([out[s][f"t{t}_L{l}_disp100_med"] for t in T_J4]) for s in range(10)])
    print(f"  T_J4 全体 disp100: median {np.median(v):.4f} range [{v.min():.4f}, {v.max():.4f}]")
