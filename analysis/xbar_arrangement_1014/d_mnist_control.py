"""xbar_arrangement_1014 / agent D: control row, 1-hidden-layer MNIST box (LR_ref_s0, T = 50), same quantities as d_brightness.py.
X = pixels/255 (subset in the npz), x~ = (x, 1), K_n = x~_n . xbar;  W = end0[T-1], b = end1[T-1];  gate LR:0.1 (P176 smooth gate for thresholds).
usage: d_mnist_control.py  -> results/xbar_arrangement_1014/d_mnist_control.json"""
import os, sys, json, gzip
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
_s = importlib.util.spec_from_file_location("dB", os.path.join(HERE, "d_brightness.py")); dB = importlib.util.module_from_spec(_s); _s.loader.exec_module(dB)
NPZ = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/opus_bwdfloor/out/LR_ref_s0.npz"
MNIST = "/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz"
OUT = dB.OUT


def read_idx(p):
    b = gzip.open(p, "rb").read(); dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def main(T=50):
    d = np.load(NPZ); X = read_idx(MNIST).reshape(-1, 784).astype(np.float64)[d["subset"]] / 255; N = X.shape[0]
    W = d["end0"][T - 1].astype(np.float64); b = d["end1"][T - 1].astype(np.float64)
    Xt = np.c_[X, np.ones(N)]; xbar = Xt.mean(0); K = Xt @ xbar; Kbar = float(xbar @ xbar)
    Wt = np.c_[W, b]; Z = Xt @ Wt.T; m = Wt @ xbar; U = Z - np.outer(K / Kbar, m)
    Open = Z > 0; alive = np.where(Open.sum(0) > 0)[0]
    edges = np.quantile(K, np.arange(1, 8) / 8.0); octv = np.searchsorted(edges, K, side="right"); pct = (dB.rankdata(K) - 1) / (N - 1)
    fr = Open[:, alive].mean(1); ge = np.where(fr >= 0.5)[0]; am = int(np.argmax(fr))
    _, Dp = dB.gate_parts(Z, "LR:0.1"); res = dB.bridges_T(Z, K[:, None] * Dp, alive); resC = dB.bridges_T(Z, Dp, alive)
    zbar = Z.mean(1); mbar = float(m.mean()); ubar = U.mean(1); ratio = ubar / (abs(mbar) * K / Kbar)
    out = dict(npz=NPZ, T=T, act=str(d["act"]), N=N,
               chk=dict(Kbar_minus_meanK=float(Kbar - K.mean()), m_minus_meanz_maxabs=float(np.abs(m - Z.mean(0)).max()), u_mean_maxabs=float(np.abs(U.mean(0)).max()),
                        zbar_identity_maxabs=float(np.abs(zbar - (mbar * K / Kbar + ubar)).max())),
               D1=dict(min=float(K.min()), max=float(K.max()), mean=float(K.mean()), sd=float(K.std()), Kbar=Kbar, n_le0=int((K <= 0).sum()), n_lt02=int((K < 0.2 * Kbar).sum())),
               D2=dict(n_alive=int(alive.size), sp_f_K=dB.spearman(fr, K), oct_f=[float(fr[octv == o].mean()) for o in range(8)], f_max=float(fr.max()), n_ge_half=int(ge.size),
                       argmax_K=float(K[am]), argmax_pct=float(pct[am]), argmax_oct=int(octv[am])),
               D3=dict(P176=dB.bridge_block(res, K, octv, pct, Open), K2free=dB.bridge_block(resC, K, octv, pct, Open)),
               D6=dict(mbar=mbar, n_pos=int((zbar > 0).sum()), max_ratio=float(ratio.max()), n_m_pos=int((m[alive] > 0).sum())))
    nb = np.array(out["D3"]["P176"]["n_bridge"]); out["D3"]["P176"]["max_bridge"] = int(nb.max()); out["D3"]["P176"]["ge3"] = int((nb >= 3).sum())
    json.dump(out, open(os.path.join(OUT, "d_mnist_control.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "D3"}, indent=1)); print({k: v for k, v in out["D3"]["P176"].items() if k != "n_bridge"})


if __name__ == "__main__":
    main()
