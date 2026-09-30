"""sink_roots_0930 round 2, tau_eff_vs_fit_speed: run filter_real.analyze on the saved switch states of one group,
with the t_fit taus added as extra rows (spec_sink_roots_0930_round2.md §2).
t_fit of the state at switch t = the fitting time of task t-1 (the old task), from tfit_L1_<act>_K<K>_s<seed>.npz:
  fit    = eta * t_fit / N,  fitK = fit / K        (t_fit: first 25-update check with full-data acc >= 0.9)
  fit90  = eta * t_fit90 / N, fit90K = fit90 / K   (t_fit90: first check with acc >= 0.9 x the task's final acc)
Usage: python3 filter_tau.py --dir <T folder> --pattern 'L1_ELU_K10_s*_t*.npz' --out <file.npy>
"""
import argparse, glob, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_real as FR

ETA, N, EVERY = 1e-3, 1200, 25


def fit_times(d, act, K, seed, task_old):
    t = np.load(Path(d) / f"tfit_L1_{act}_K{K}_s{seed}.npz")
    tfit = int(t["tfit"][task_old - 1]); curve = t["acc_curve"][task_old - 1]
    j = int(np.argmax(curve >= 0.9 * curve[-1]))
    return tfit, (j + 1) * EVERY, float(curve[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--pattern", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.monotonic(); res = []
    for f in sorted(glob.glob(str(Path(a.dir) / a.pattern))):
        m = re.match(r"L1_(\w+?)_K(\d+)_s(\d+)_t(\d+)", Path(f).stem)
        act, K, seed, t = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        tfit, tfit90, acc_end = fit_times(a.dir, act, K, seed, t - 1)
        extra = {"fit90": ETA * tfit90 / N, "fit90K": ETA * tfit90 / N / K}
        if tfit > 0:
            extra.update({"fit": ETA * tfit / N, "fitK": ETA * tfit / N / K})
        r = FR.analyze(f, extra_taus=extra)
        r.update(tfit=tfit, tfit90=tfit90, acc_end_old=acc_end, T=int(np.load(f)["ks"][-1]), seed=seed, switch=t)
        res.append(r)
        pl = r["plain"]
        print(f"{r['name']:24s} T {r['T']:5d} t_fit {tfit:5d} t_fit90 {tfit90:5d} acc_end {acc_end:.2f} | obs Soh>0 {r['Soh_pos']:.2f} | "
              f"nat P {pl['rows'][-3]['meanP']:.2f} natK {pl['rows'][-2]['meanP']:.2f} | "
              + " ".join(f"{k} {v['meanP']:.2f}" for k, v in pl["extra"].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, np.array(res, dtype=object), allow_pickle=True)
    print(f"({time.monotonic() - t0:.0f}s, {len(res)} states)")
