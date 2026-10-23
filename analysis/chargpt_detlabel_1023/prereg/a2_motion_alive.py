"""spec §2.2: A2 の課題の中の変位・回転を、生きた unit（その課題末の鎖 probe で z > 0 のトークンがある unit。unit_stats.npz の chain_dead）に限って出す。
読むだけ。使い方: python -I a2_motion_alive.py <ref> <out_json>（<ref> は a2_table.py と同じ）"""
import json, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
R = Path("/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/M2/C3")
REF = Path(sys.argv[1])
def W(p, l): return torch.load(p, map_location="cpu", weights_only=True)[f"h.{l}.mlp.up.weight"].double()
res = {}
for s in range(10):
    U = np.load(REF / f"C3/seed{s}/unit_stats.npz")
    dead = U["chain_dead"]                     # [課題, 層, unit] 課題末の鎖 probe
    ck = R / f"seed{s}" / "ckpts"; r = {}
    for t in list(range(2, 11)) + list(range(41, 51)) + list(range(91, 101)):
        for l in (0, 1):
            alive = torch.from_numpy(~dead[t - 1, l])
            a, b = W(ck / f"u100_task{t}.pt", l), W(ck / f"task{t}.pt", l)
            d = (b - a).norm(dim=1)
            r[f"t{t}_L{l}_disp_alive"] = float(d[alive].median()) if alive.any() else float("nan")
            r[f"t{t}_L{l}_nalive"] = int(alive.sum())
            if (ck / f"task{t-1}.pt").exists():
                c = W(ck / f"task{t-1}.pt", l)
                both = alive & torch.from_numpy(~dead[t - 2, l])
                cos = (b * c).sum(1) / (b.norm(dim=1) * c.norm(dim=1))
                r[f"t{t}_L{l}_rot_alive"] = float((1 - cos)[both].median()) if both.any() else float("nan")
    res[s] = r
for l in (0, 1):
    print(f"== L{l}（生きた unit）")
    for name, ts in (("2-10", range(2, 11)), ("41-50", range(41, 51)), ("91-100", range(91, 101))):
        for key in ("disp_alive", "rot_alive", "nalive"):
            v = np.array([np.nanmedian([res[s].get(f"t{t}_L{l}_{key}", np.nan) for t in ts]) for s in range(10)])
            print(f"  {name} {key}: median {np.nanmedian(v):.4f} range [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
json.dump(res, open(sys.argv[2], "w"))
