"""spec §7.4 の「再生をしない場合」の J4 の参照値: A2 の M_k（T_J4 の課題ごとの値の中央値）を net100 で出す。
入力は a2_motion.py・a2_motion_alive.py の出力。使い方: python -I a2_j4_fallback.py <prereg の出力フォルダ>"""
import json, sys
import numpy as np
R = sys.argv[1]
a = json.load(open(f"{R}/a2_motion_alive.json")); b = json.load(open(f"{R}/a2_motion.json"))
T = list(range(2, 11)) + list(range(41, 51)) + list(range(91, 101))
for l in (0, 1):
    Ma = np.array([np.nanmedian([a[str(s)][f"t{t}_L{l}_disp_alive"] for t in T]) for s in range(10)])
    Mb = np.array([np.median([b[str(s)][f"t{t}_L{l}_disp100_med"] for t in T]) for s in range(10)])
    print(f"L{l} net100 生きた unit M_k: median {np.median(Ma):.3f} [{Ma.min():.3f}, {Ma.max():.3f}] | 全行 {np.median(Mb):.3f} [{Mb.min():.3f}, {Mb.max():.3f}]")
