#!/usr/bin/env python3
"""課題 1 の最初の 500 更新を初期値から再生し（データ列は本走と同じ）、更新 100・200・300・400・500 の網を保存する。
本走のオンライン損失とビット一致することを確かめ、L0 の μ/σ の変化を部品ごとの入れ替えで割る（下見・未登録、1020 夜の Issa の問い「w が学習で沈んだのか」）。
L0 の前活性 z = W_up · (γ2 ⊙ LN(emb + attn0(LN1(emb))))。部品: W_up（自分の重み）、γ2（自分の LN）、埋め込み、注意 0（LN1 の γ・qkv・o）。
  python analysis/chargpt_cl_1019_quicklook/task1_replay_l0.py --arms SiLU C3 GELU
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
Q = ROOT / "results/chargpt_cl_1019/quicklook/M"
PTS = (100, 200, 300, 400, 500)
PARTS = {"自分の重み W_up": ["h.0.mlp.up.weight"], "自分の LN γ2": ["h.0.n2.weight"], "埋め込み": ["wte.weight", "head.weight"],
         "注意 0": ["h.0.n1.weight", "h.0.attn.qkv.weight", "h.0.attn.o.weight"]}


def replay(arm, dev):
    C.setup_torch()
    m = C.build_model(arm, "M", 1000, dev)
    eng = C.Engine(m, arm, "M", graph=True)
    ctx = C.TaskCtx("M", 1000, 1, 500, dev, alpha=0.01, W=None, cpu=C.task_cpu_M(1000, 1, 500, 0.01))
    snaps = {}

    def cb(u):
        snaps[u] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    loss, _ = C.run_task(m, eng, ctx, warmup_from=0, measure_cb=cb, points=PTS)
    ref = np.load(Q / arm / "seed1000" / "online.npz")["loss"][0][:500]
    return snaps, int((loss == ref).sum())


def rmed_l0(sd, probe, arm):
    m = C.build_model(arm, "M", 1000, "cpu"); m.load_state_dict(sd); m.eval(); cap = []
    with torch.no_grad():
        m.body(probe, cap)
    z = cap[1].reshape(-1, C.FF).double()
    return float((z.mean(0) / z.std(0)).median()), float(z.mean(0).median()), float(z.std(0).median())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default=["SiLU", "C3", "GELU"]); a = ap.parse_args()
    torch.set_num_threads(6)
    probe = C.task_cpu_M(1000, 1, 10, 0.01)["probe"]
    for arm in a.arms:
        snaps, eq = replay(arm, torch.device("cuda"))
        snaps["end"] = torch.load(Q / arm / "seed1000" / "ckpts" / "task1.pt", map_location="cpu")
        print(f"==== {arm}: 再生の一致 {eq}/500 更新")
        print("  L0 の μ/σ 中央・μ 中央・σ 中央: " + " | ".join(f"u{u} {rmed_l0(snaps[u], probe, arm)[0]:+.2f} ({rmed_l0(snaps[u], probe, arm)[1]:+.3f}, {rmed_l0(snaps[u], probe, arm)[2]:.3f})" for u in list(PTS) + ["end"]))
        for an, bn in ((100, 200), (200, 400), (400, "end")):
            A, B = snaps[an], snaps[bn]
            base, full = rmed_l0(A, probe, arm)[0], rmed_l0(B, probe, arm)[0]
            out = []
            for name, keys in PARTS.items():
                sd = dict(A)
                for k in keys:
                    if k in sd: sd[k] = B[k]
                out.append(f"{name} {rmed_l0(sd, probe, arm)[0] - base:+.2f}")
                sd2 = dict(B)                        # 逆向き: その部品だけ前のまま
                for k in keys:
                    if k in sd2: sd2[k] = A[k]
                out[-1] += f"（その部品だけ戻す {rmed_l0(sd2, probe, arm)[0] - full:+.2f}）"
            print(f"  u{an}→{bn}: {base:+.2f} → {full:+.2f}（{full - base:+.2f}） | " + " | ".join(out))


if __name__ == "__main__":
    main()
