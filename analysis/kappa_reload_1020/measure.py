#!/usr/bin/env python3
"""κ の再装填（kappa_reload_1020）: 課題ごとの κ_k、切替の押し、課題の中の戻りを、保存済みの状態から測る。事後・未登録。

箱: push_direction_proof_0930/gen_states.py の 1 隠れ層 RL-MNIST（784-100-10、MNIST raw/255・subset 1,200、
Adam 1e-3・batch 16・4,000 更新/課題・30 課題、CPU・float32 で学習）。保存されているのは、初期値、各課題の末（end）、
各課題の +200 更新（mid）の重みと、各課題のラベル y。Adam の状態は保存されていない。ここでは全部 float64 で再計算する。

記号（V12 付録 C.3）:
  位置 m_i = w_i·x̄ + b_i、明るさ K_n = x̄·x_n + 1、傾き g = φ′(z)、v^c はクラスで中心化した読み出し、π は予測。
  全バッチ GD の一歩（損失は N 枚の平均）で位置は Δm_i = −(η/N) Σ_n δ_in K_n 動く。δ_in = g_in ⟨v_i, π_n − y_n⟩。
  新ラベルについて平均した押し S_i = Σ_n K_n g_in ⟨v_i^c, π_n⟩。期待の変位は −(η/N) S_i（S > 0 が沈める側）。
  旧ラベルの的を当てた形 S_oh_i = Σ_n K_n g_in v^c_{i, y_n}（π を旧ラベルの one-hot に置いた S）。
  H1 の読み出し模型 v_ic = v⁰_ic + κ Σ_m (y_mc − 1/K) φ_im。自己形 S_self_i = κ (1 − 1/K) Σ_n K_n g_in φ_in。

課題 k の末の状態から課題 k+1 への切替を「切替 k」と呼ぶ（k = 1..29）。切替 0 は初期値から課題 1 への入り（参照）。
"""
import argparse, gzip, json, math
from pathlib import Path
import numpy as np
import torch

STATES = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/states")
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
ETA = 1e-3
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]
SEEDS = [0, 1]
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


_ALL = None
def load_X(subset):
    global _ALL
    if _ALL is None:
        _ALL = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float64) / 255
    return _ALL[subset]


def erf(x):
    return torch.erf(torch.from_numpy(np.ascontiguousarray(x))).numpy()


def activ(z, act):
    if act == "LR":   return np.where(z > 0, z, 0.1 * z)
    if act == "ELU":  return np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
    if act == "R":    return np.maximum(z, 0)
    if act == "GELU": return z * 0.5 * (1.0 + erf(z / SQRT2))
    if act == "SILU": return z / (1.0 + np.exp(-z))
    raise ValueError(act)


def gate(z, act):
    if act == "LR":   return np.where(z > 0, 1.0, 0.1)
    if act == "ELU":  return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))
    if act == "R":    return (z > 0).astype(np.float64)
    if act == "GELU": return 0.5 * (1.0 + erf(z / SQRT2)) + z * np.exp(-0.5 * z * z) * INV_SQRT_2PI
    if act == "SILU":
        s = 1.0 / (1.0 + np.exp(-z)); return s * (1.0 + z * (1.0 - s))
    raise ValueError(act)


def softmax(l):
    l = l - l.max(1, keepdims=True); p = np.exp(l); return p / p.sum(1, keepdims=True)


def onehot(y, K):
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), y] = 1.0; return Y


def rankavg(x):
    """同順位は平均順位（memory: 順位相関は同順位を平均）。"""
    order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r); return (s / cnt)[inv]


def spearman(a, b):
    if len(a) < 3: return float("nan")
    ra, rb = rankavg(a), rankavg(b)
    if ra.std() == 0 or rb.std() == 0: return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quart(x):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    if len(x) == 0: return [float("nan")] * 3
    return [float(v) for v in np.percentile(x, [25, 50, 75])]


def agree(a, b, mask):
    """符号の一致率（mask の中で両方が 0 でない unit）。"""
    mk = mask & (a != 0) & (b != 0)
    if mk.sum() == 0: return float("nan"), 0
    return float((np.sign(a[mk]) == np.sign(b[mk])).mean()), int(mk.sum())


def evaluate(params, act, X, xbar):
    W1, b1, W2, b2 = params
    z = X @ W1.T + b1; a = activ(z, act); g = gate(z, act)
    pi = softmax(a @ W2.T + b2)
    W2c = W2 - W2.mean(0, keepdims=True)                     # [K, H]
    return dict(z=z, a=a, g=g, pi=pi, W2=W2, W2c=W2c, m=W1 @ xbar + b1, s=z.std(0),
                pplus=(z > 0).mean(0), kopen=(z > 0).sum(0))


def analyze(act, seed):
    d = np.load(STATES / f"{act}_K10_s{seed}.npz")
    K = int(d["K"]); X = load_X(d["subset"]); N = X.shape[0]
    xbar = X.mean(0); Kn = X @ xbar + 1.0
    T = d["end0"].shape[0]
    getp = lambda which, j: [d[f"{which}{q}"][j].astype(np.float64) for q in range(4)]
    init = [d[f"init{q}"].astype(np.float64) for q in range(4)]
    ylab = d["y"].astype(int)

    E = [evaluate(getp("end", t), act, X, xbar) for t in range(T)]      # E[t] = 課題 t+1 の末
    M = [evaluate(getp("mid", t), act, X, xbar) for t in range(T)]      # M[t] = 課題 t+1 の +200 更新
    I0 = evaluate(init, act, X, xbar)
    H = I0["m"].shape[0]

    rows = []
    unit = {k: [] for k in ("S_exact", "S_self", "S_oh", "dm_gd_real", "dm_sw", "dm_rest", "m_from", "s_from",
                            "p_from", "p_mid", "p_next", "m_next", "kappa_i", "alive")}

    for k in range(0, T):                     # 切替 k: 状態 = 課題 k の末（k = 0 は初期値）、新ラベル = 課題 k+1
        e = I0 if k == 0 else E[k - 1]
        mi, ne = M[k], E[k]
        y_new = ylab[k]; Yn = onehot(y_new, K)

        # --- 状態の量（課題 k の末）
        if k == 0:
            acc = conf = float("nan"); kappa = kappa_d = expl = expl_d = float("nan")
            kap_chance = kap_feat = kap_read = float("nan")
            kappa_i = np.full(H, np.nan); learned_norm = float("nan")
            S_oh = np.full(H, np.nan)
        else:
            y_old = ylab[k - 1]; Yo = onehot(y_old, K)
            acc = float((e["pi"].argmax(1) == y_old).mean()); conf = float(e["pi"][np.arange(N), y_old].mean())
            F = (Yo - 1.0 / K).T @ e["a"]                            # [K, H]  Σ_m (y_mc − 1/K) φ_im
            ff = float((F * F).sum())
            kappa = float((e["W2c"] * F).sum() / ff)
            expl = float((e["W2c"] * F).sum() ** 2 / ((e["W2c"] ** 2).sum() * ff))
            prev = I0 if k == 1 else E[k - 2]
            dW = e["W2c"] - prev["W2c"]
            kappa_d = float((dW * F).sum() / ff)
            expl_d = float((dW * F).sum() ** 2 / ((dW ** 2).sum() * ff))
            # κ_k の恒等式の分解: 課題 k の始めの読み出し・特徴に旧ラベルを当てた偶然の分 + 特徴の変化の分 + 読み出しの変化の分
            Fp = (Yo - 1.0 / K).T @ prev["a"]
            kap_chance = float((prev["W2c"] * Fp).sum() / ff)
            kap_feat = float((prev["W2c"] * (F - Fp)).sum() / ff)
            kap_read = kappa_d
            kappa_i = (e["W2c"] * F).sum(0) / (F * F).sum(0)
            learned_norm = float(abs(kappa) * math.sqrt(ff) / math.sqrt(H))   # 学んだ読み出し κF の unit あたりの大きさ
            S_oh = (Kn[:, None] * e["g"] * e["W2c"][y_old, :]).sum(0)

        SF = (Kn[:, None] * e["g"] * e["a"]).sum(0)                  # Σ_n K_n φ′ φ
        S_self = kappa * (1 - 1.0 / K) * SF if k > 0 else np.full(H, np.nan)
        S_exact = (Kn[:, None] * e["g"] * (e["pi"] @ e["W2c"])).sum(0)
        delta_real = e["g"] * ((e["pi"] - Yn) @ e["W2"])              # 実ラベルの δ_in
        dm_gd_real = -(ETA / N) * (Kn[:, None] * delta_real).sum(0)
        dm_gd_exp = -(ETA / N) * S_exact
        dm_sw = mi["m"] - e["m"]; dm_rest = ne["m"] - mi["m"]
        alive = e["kopen"] > 0

        # --- 符号と大きさ
        a_se, n_se = agree(S_self, S_exact, alive) if k > 0 else (float("nan"), 0)
        a_oe, _ = agree(S_oh, S_exact, alive) if k > 0 else (float("nan"), 0)
        a_eg, _ = agree(-S_exact, dm_gd_real, alive)
        a_ea, _ = agree(-S_exact, dm_sw, alive)
        a_sa, _ = agree(-S_self, dm_sw, alive) if k > 0 else (float("nan"), 0)
        al = alive
        ratio_se = (S_self[al] / S_exact[al]) if k > 0 else np.array([])
        ratio_ge = dm_gd_real[al] / dm_gd_exp[al]
        ratio_ag = dm_sw[al] / dm_gd_exp[al]

        # --- 戻り（切替の 200 更新で沈んだ生きた unit）
        sank = al & (dm_sw < 0)
        R = dm_rest[sank] / np.abs(dm_sw[sank])
        net = dm_sw + dm_rest

        # --- 1/2 と 0 を跨ぐ数（全 unit）
        p_from, p_mid, p_next = e["pplus"], mi["pplus"], ne["pplus"]
        row = dict(
            act=act, seed=seed, switch=k, acc_end=acc, conf_end=conf,
            kappa=kappa, kappa_delta=kappa_d, expl_total=expl, expl_delta=expl_d, learned_norm=learned_norm,
            kap_chance=kap_chance, kap_feat=kap_feat, kap_read=kap_read,
            kappa_i_q=quart(kappa_i[al]) if k > 0 else [float("nan")] * 3,
            kappa_i_negfrac=float((kappa_i[al] < 0).mean()) if k > 0 else float("nan"),
            n_alive=int(al.sum()), n_units=int(H),
            m_med=float(np.median(e["m"][al])) if al.any() else float("nan"),
            s_med=float(np.median(e["s"][al])) if al.any() else float("nan"),
            r_med=float(np.median((e["m"] / e["s"])[al])) if al.any() else float("nan"),
            S_exact_medN=float(np.median(S_exact[al]) / N) if al.any() else float("nan"),
            S_self_medN=float(np.median(S_self[al]) / N) if (k > 0 and al.any()) else float("nan"),
            S_oh_medN=float(np.median(S_oh[al]) / N) if (k > 0 and al.any()) else float("nan"),
            frac_sink_exact=float((S_exact[al] > 0).mean()) if al.any() else float("nan"),
            frac_sink_self=float((S_self[al] > 0).mean()) if (k > 0 and al.any()) else float("nan"),
            frac_sink_gd=float((dm_gd_real[al] < 0).mean()) if al.any() else float("nan"),
            frac_sink_adam=float((dm_sw[al] < 0).mean()) if al.any() else float("nan"),
            agree_self_exact=a_se, agree_oh_exact=a_oe, agree_exact_gd=a_eg, agree_exact_adam=a_ea, agree_self_adam=a_sa,
            ratio_self_exact_q=quart(ratio_se), ratio_gd_exp_q=quart(ratio_ge), ratio_adam_gd_q=quart(ratio_ag),
            sp_self_exact=spearman(S_self[al], S_exact[al]) if k > 0 else float("nan"),
            sp_exact_adam=spearman(S_exact[al], -dm_sw[al]),
            dm_sw_med=float(np.median(dm_sw[al])) if al.any() else float("nan"),
            dm_rest_med=float(np.median(dm_rest[al])) if al.any() else float("nan"),
            dm_sw_rel_med=float(np.median((dm_sw / e["s"])[al])) if al.any() else float("nan"),
            n_sank=int(sank.sum()), R_q=quart(R),
            R_agg=float(dm_rest[sank].sum() / np.abs(dm_sw[sank]).sum()) if sank.any() else float("nan"),
            frac_R_ge1=float((R >= 1).mean()) if len(R) else float("nan"),
            frac_rest_pos=float((dm_rest[sank] > 0).mean()) if sank.any() else float("nan"),
            frac_net_neg=float((net[al] < 0).mean()) if al.any() else float("nan"),
            n_p_below_from=int((p_from < 0.5).sum()), n_m_neg_from=int((e["m"] < 0).sum()),
            n_p_up_mid=int(((p_from < 0.5) & (p_mid >= 0.5)).sum()),
            n_p_up_next=int(((p_from < 0.5) & (p_next >= 0.5)).sum()),
            n_m_up_next=int(((e["m"] < 0) & (ne["m"] >= 0)).sum()),
            n_p_down_next=int(((p_from >= 0.5) & (p_next < 0.5)).sum()),
        )
        rows.append(row)
        for key, val in (("S_exact", S_exact), ("S_self", S_self), ("S_oh", S_oh), ("dm_gd_real", dm_gd_real),
                         ("dm_sw", dm_sw), ("dm_rest", dm_rest), ("m_from", e["m"]), ("s_from", e["s"]),
                         ("p_from", p_from), ("p_mid", p_mid), ("p_next", p_next), ("m_next", ne["m"]),
                         ("kappa_i", kappa_i), ("alive", alive.astype(np.float64))):
            unit[key].append(np.asarray(val, dtype=np.float64))

    # --- 一度下に入った unit が後の課題末に戻ったか（全 unit、課題末の列で）
    Pend = np.stack([E[t]["pplus"] for t in range(T)])        # [T, H]
    Mend = np.stack([E[t]["m"] for t in range(T)])
    def ever_return(below, level):
        n_entered = n_returned = 0; first_task = []; who = []
        for i in range(H):
            idx = np.flatnonzero(below[:, i])
            if len(idx) == 0: continue
            n_entered += 1; first_task.append(int(idx[0]) + 1)
            ups = [int(t) + 1 for t in range(idx[0], T) if not below[t, i]]
            if ups:
                n_returned += 1
                who.append(dict(unit=i, first_below_task=int(idx[0]) + 1, back_tasks=ups,
                                level_first6=[float(v) for v in level[:6, i]], level_last=float(level[-1, i]),
                                below_after_last_return=bool(below[ups[-1]:, i].all()) if ups[-1] < T else False))
        return n_entered, n_returned, first_task, who
    pe, pr, pft, pwho = ever_return(Pend < 0.5, Pend)
    me, mr, mft, mwho = ever_return(Mend < 0, Mend)
    run = dict(act=act, seed=seed, T=int(T), N=int(N), H=int(H),
               p_entered=pe, p_returned=pr, p_first_task_q=quart(pft), p_returners=pwho,
               m_entered=me, m_returned=mr, m_first_task_q=quart(mft), m_returners=mwho,
               acc=[float(v) for v in d["acc"]])
    units = {k: np.stack(v) for k, v in unit.items()}
    units["Pend"] = Pend; units["Mend"] = Mend
    return rows, run, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kappa_reload_1020")
    ap.add_argument("--raw", default="results/kappa_reload_1020/raw")
    a = ap.parse_args()
    out = Path(a.out); raw = Path(a.raw); out.mkdir(parents=True, exist_ok=True); raw.mkdir(parents=True, exist_ok=True)
    all_rows, runs = [], []
    for act in ACTS:
        for seed in SEEDS:
            rows, run, units = analyze(act, seed)
            all_rows += rows; runs.append(run)
            np.savez_compressed(raw / f"units_{act}_s{seed}.npz", **units)
            print(f"{act} s{seed}: switches {len(rows)}  p_entered {run['p_entered']} returned {run['p_returned']}  "
                  f"m_entered {run['m_entered']} returned {run['m_returned']}", flush=True)
    (out / "rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1))
    (out / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1))
    print("wrote", out / "rows.json", out / "runs.json")


if __name__ == "__main__":
    main()
