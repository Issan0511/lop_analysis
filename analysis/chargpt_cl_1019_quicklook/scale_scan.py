"""切替の瞬間（課題 25 末の重み × 課題 26 の系列）で、層 l の FFN 出力 f を f̄（トークン平均 = 共通成分）と f−f̄ に分け、
s_c・s_v 倍して損失を見る。対照は収束した状態（課題 25 の系列）。両腕。"""
import sys, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(6)
sys.path.insert(0, "src")
import chargpt_cl_1019_kune as K
C = K.C
R = Path("results/chargpt_cl_1019")
RUNS = {"KKT1_s1000": (R/"quicklook/M/KKT1/seed1000", "KKT1", 1000), "C3_s0": (R/"crosscheck_white-san/M/C3/seed0", "C3", 0)}
SC = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
def loss_on(m, seq):
    with torch.no_grad():
        lg = m(seq[:, :C.T])
        return float(F.cross_entropy(lg[:, 1:].reshape(-1, C.V), seq[:, 2:C.T + 1].reshape(-1)))
def scaled(mlp, sc, sv):
    orig = mlp.forward
    def fwd(y, cap=None):
        f = orig(y, cap)
        fbar = f.reshape(-1, C.D).mean(0)
        return sc * fbar + sv * (f - fbar)
    return orig, fwd
for name, (d, arm, seed) in RUNS.items():
    print(f"\n==== {name}")
    m = C.build_model(arm, "M", seed, "cpu")
    m.load_state_dict(torch.load(d/"ckpts"/"task25.pt", map_location="cpu")); m.eval()
    seqs = {"切替(課題26)": C.task_cpu_M(seed, 26, 10, 0.01)["eval_seq"], "収束(課題25)": C.task_cpu_M(seed, 25, 10, 0.01)["eval_seq"]}
    for tag, seq in seqs.items():
        base = loss_on(m, seq)
        print(f"-- {tag}: 損失 {base:.3f}   （行: 層、列: s = {SC}。値は 損失(s) − 損失(1)）")
        for l, blk in enumerate(m.h):
            rows = {}
            for mode, mk in (("共通だけ", lambda s: (s, 1.0)), ("変動だけ", lambda s: (1.0, s)), ("両方", lambda s: (s, s))):
                vals = []
                for s in SC:
                    sc, sv = mk(s); orig, fwd = scaled(blk.mlp, sc, sv); blk.mlp.forward = fwd
                    vals.append(loss_on(m, seq) - base); blk.mlp.forward = orig
                rows[mode] = vals
            for mode, vals in rows.items():
                print(f"   L{l} {mode}: " + " ".join(f"{v:+.3f}" for v in vals))
