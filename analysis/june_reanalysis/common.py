"""june_reanalysis (B1–B4) 共通ユーティリティ。

仕様書 june_reanalysis_spec.md §2 の共通規約:
  - cos は signed と |cos| の両方を常に計算・保存する
  - ランダム基準線 E|cos| ≈ sqrt(2/(pi*d))
  - dead ニューロンは除外前後の両方を保存
  - 乱数 seed 固定 numpy.random.default_rng(20260811)

重要な Phase 0 の発見 (data_inventory.md 参照):
  既報 A1 = 0.61–0.68 は **重み W ではなく凍結測定の期待勾配 Eg_W** に対する
  sign(v_i)sign(v_j) 補正つき pairwise cos (alive のみ, signed)。
  仕様書は「重み w_i」と書いているので、本パッケージは常に **両方の対象**
  (obj='W' と obj='Eg') について同じ統計を計算する。
"""
import json
import os

import numpy as np

SEED = 20260811
EPS = 1e-12

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_RESULTS = os.path.join(ROOT, "results", "drift_0809")
OUT = os.path.join(ROOT, "results", "june_reanalysis")

# 既報値 (怪文書§5 / RESULTS.md)。再現できない場合も止めず差分を記録する。
REPORTED = {
    "kaibunsho_inter_unit_absmean_cos": 0.27,
    "kaibunsho_mu_absmean_cos": 0.38,
    "kaibunsho_independent_residual_pred": 0.38 ** 2,
    "kaibunsho_second_direction_strength": float(np.sqrt(0.27 - 0.38 ** 2)),
    "A1_range": (0.61, 0.68),
    "two_over_pi": 2.0 / np.pi,
}


def rng():
    return np.random.default_rng(SEED)


def unit(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), EPS)


def chance_floor(d):
    """d 次元の独立ランダム単位ベクトル対の E|cos|。"""
    return float(np.sqrt(2.0 / (np.pi * d)))


# --------------------------------------------------------------- データ読み込み

def load_runs():
    """runs.csv を dict[run_id] -> 条件 dict で返す (pandas 非依存)。"""
    import csv
    path = os.path.join(SRC_RESULTS, "runs.csv")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["run_id"]] = dict(
                exp=r["exp"], width=int(r["width"]), period=int(r["period"]),
                enc=r["enc"], c=(float(r["c"]) if r["c"] else None),
                lr=float(r["lr"]), seed=int(r["seed"]), run_id=r["run_id"])
    return out


def load_npz(exp, width, step):
    p = os.path.join(SRC_RESULTS, f"followup_Eg_{exp}_w{width}_step{step}.npz")
    if not os.path.exists(p):
        return None
    z = dict(np.load(p, allow_pickle=True))
    for k, v in z.items():                       # 発散系列の NaN/Inf を 0 に潰す
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            z[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    z["run_ids"] = np.array([str(s) for s in z["run_ids"]])
    return z


GROUPS = [("A", 5), ("A", 100), ("B", 5), ("B", 100)]
STEPS = [0, 10000, 50000, 100000, 300000, 1000000]


def iter_units(steps=None, groups=None):
    """(exp, width, step, npz) を順に返す。"""
    for exp, width in (groups or GROUPS):
        for step in (steps or STEPS):
            z = load_npz(exp, width, step)
            if z is not None:
                yield exp, width, step, z


def cond_label(r):
    """条件ラベル (seed を除く)。"""
    if r["exp"] == "A":
        return f"A_w{r['width']}_T{r['period']}_{r['enc']}_lr{r['lr']}"
    return f"B_w{r['width']}_K{r['period']}_c{r['c']}_lr{r['lr']}"


# ------------------------------------------------------------ 解析対象の取り出し

def get_matrix(z, i, obj, alive_only=True):
    """run i のニューロン別ベクトル行列 [n, d] と sign(v) を返す。

    obj='W'  : 学習器 第1層の重み w_i               (仕様書の文面どおりの対象)
    obj='Eg' : 凍結測定の期待勾配 E[g_{W_i}]        (既報 A1 が実際に使った対象)
    """
    M = z["W"][i] if obj == "W" else z["Eg_W"][i]
    sv = np.sign(z["v"][i])
    sv[sv == 0] = 1.0
    if alive_only:
        keep = ~z["dead"][i]
        M, sv = M[keep], sv[keep]
    return M, sv


# ------------------------------------------------------------------ cos 統計

def pair_cos(M, sv=None):
    """単位化した行ベクトルの全ペア cos を 1 次元配列で返す。

    sv を渡すと sign(v_i)sign(v_j) 補正つき (既報 A1 の vpcos と同じ規約)。
    """
    if len(M) < 2:
        return np.array([])
    U = unit(M, axis=1)
    G = U @ U.T
    if sv is not None:
        G = (sv[:, None] * sv[None, :]) * G
    iu = np.triu_indices(len(U), k=1)
    return G[iu]


def cos_stats(c, prefix=""):
    """signed / abs の両方を返す (仕様書 §8: どちらかを必ず明記)。"""
    if len(c) == 0:
        return {f"{prefix}n_pairs": 0, f"{prefix}signed_mean": np.nan,
                f"{prefix}abs_mean": np.nan, f"{prefix}signed_median": np.nan}
    return {f"{prefix}n_pairs": int(len(c)),
            f"{prefix}signed_mean": float(np.mean(c)),
            f"{prefix}signed_median": float(np.median(c)),
            f"{prefix}abs_mean": float(np.mean(np.abs(c)))}


def boot_ci(x, n=2000, stat=np.mean, seed=SEED):
    """pair-level bootstrap 95% CI。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n, len(x)))
    b = stat(x[idx], axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def agg_seeds(per_seed):
    """seed 間 mean ± std。per_seed は数値リスト。"""
    a = np.asarray([v for v in per_seed if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, n_seeds=0)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1) if len(a) > 1 else 0.0),
                n_seeds=int(len(a)))


# ------------------------------------------------------------------ 入出力

def save_json(obj, *parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)

    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=enc)
    print(f"  wrote {os.path.relpath(p, ROOT)}")
    return p


def figpath(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def verdict(*parts_and_text):
    """判定 1 行を verdict.txt に書き、標準出力にも出す。"""
    *parts, text = parts_and_text
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text.rstrip() + "\n")
    print(f"  VERDICT: {text}")


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
