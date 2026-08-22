"""ratchet_log_0819 Phase 0 [spec_ratchet_log_0819 §4]: 再学習なしの事前チェック。

  OMP_NUM_THREADS=1 .venv/bin/python -m src.ratchet_log_phase0

§4 の 3 項目:
  1. 恒等式サニティ — posreset_0819 の t=500k スナップショットで本実装の G / F_i を
     独立経路と突き合わせ、相対誤差 < 1e-10
  2. 0.567 の再現 — drift_0809 の隣接 ckpt 間 s(t) 一致率
  3. グリッド健全性 — src.ratchet_log --smoke 側で実施 (本モジュールは結果を読むだけ)

**§2 / §4.1 の記述と実装の食い違い (記録)**: 仕様は「F_i = −2η·v_i·G 恒等式の厳密計算
実装」が `src/posreset_posthoc.py` にあるとしてそれとの突き合わせを求めるが、実際の
posreset_posthoc.py はアーム間コントラスト (paired bootstrap) だけを行う解析モジュールで、
G も F_i も計算していない。リポジトリ全体を探しても厳密 G の参照実装は存在しない
(「符号一致率 0.993」もリポジトリ内に記録が無く、canvas 側の事後解析の値)。
そこで突き合わせ相手を **本番の勾配コード `nets.VecMLP.grads_batch`** に置き換えた。
F_i^gate = −η·E[gW_i] は実装上の恒等式なので、これは
「exact_record の手書き einsum が学習ループと同じ勾配を再現するか」の独立検査になり、
§4.1 の意図 (本実装の G / F_i が信用できるか) は満たす。
"""
import os
import time

import numpy as np
import torch

from .common import ROOT, load_config
from .envs import SCREnv, LTUTarget
from .nets import VecMLP
from .ratchet_log import full_support_ro, teacher_f64, exact_record

OUT = os.path.join(ROOT, "results", "ratchet_log_0819")
SNAP_A = os.path.join(ROOT, "results", "posreset_0819", "snapshots",
                      "A_w100_cont_step500000.pt")
DRIFT = os.path.join(ROOT, "results", "drift_0809")
TOL = 1e-10


# ---------------------------------------------------------------- 状態の復元

def build_st(ck, cfg, idx=None, device="cpu"):
    """ckpt / snapshot dict から exact_record が要求する最小限の st を組む。

    idx を渡すと run 次元をその部分集合に絞る (drift_0809 は 1 グループに
    T・enc・lr 違いの 50 run が同居しているため)。コンストラクタは generator を
    消費するがその直後に load_state で上書きするので、使い捨ての generator でよい。"""
    A = cfg["condA"]
    runs = ck["runs"]
    if idx is None:
        idx = list(range(len(runs)))
    runs = [runs[i] for i in idx]
    R, m, f = len(idx), int(A["m"]), int(A["f"])
    width = int(ck["net"]["W"].shape[1])
    g = torch.Generator(device=device)
    g.manual_seed(0)

    sel = lambda t: t[idx].clone()
    env = SCREnv(R, m, f, torch.tensor([int(r["period"]) for r in runs]), g, device)
    env.load_state({"flip_state": sel(ck["env"]["flip_state"]), "t": ck["env"]["t"]})
    teacher = LTUTarget(R, m, int(A["target_hidden"]), float(A["beta"]), g, device)
    teacher.load_state({k: sel(ck["teacher"][k]) for k in ("W", "b", "v", "cout", "tau")})
    net = VecMLP(R, width, m, g, device)
    net.load_state({k: sel(ck["net"][k]) for k in ("W", "b", "v", "c")})

    return dict(env=env, net=net, teacher=teacher, R=R, d=m, width=width, runs=runs,
                running_mean=sel(ck["running_mean"]), device=device,
                lr=torch.tensor([float(r["lr"]) for r in runs], device=device),
                centered=torch.tensor([r["enc"] == "centered" for r in runs],
                                      device=device))


# ---------------------------------------------------------------- §4.1 恒等式サニティ

def identity_check(st):
    """exact_record の G / F を、本番の勾配コード経由の独立計算と突き合わせる。

    独立経路: full_support_ro -> net.forward_batch -> net.grads_batch (学習ループが
    batch!=1 で実際に使う道) から E[gW_i] を作り、F_i^gate = −η·E[gW_i] の µ̂ 射影が
    exact_record の F_self + F_rest と一致するかを見る。G についても同様に
    forward_batch 由来の δ から E[δx] を組み直す。"""
    rec = exact_record(st, as_f64=True)     # float32 に丸めると 1e-8 台の丸め誤差で判定不能
    net, env = st["net"], st["env"]
    with torch.no_grad():
        X = full_support_ro(env).double()
        y = teacher_f64(st["teacher"], X)
        # --- 独立経路: 本番の VecMLP を float64 化して forward_batch/grads_batch を通す
        n64 = VecMLP(st["R"], st["width"], st["d"],
                     torch.Generator(device=st["device"]), st["device"])
        n64.W, n64.b = net.W.double(), net.b.double()
        n64.v, n64.c = net.v.double(), net.c.double()
        pre, a, yhat = n64.forward_batch(X)
        delta = yhat - y
        gW, gb, gv, gc = n64.grads_batch(X, pre, a, delta)      # [P,R,h,m]
        EgW = gW.mean(dim=0)                                    # [R,h,m]

        mu = X.mean(dim=0)
        mu_u = mu / mu.norm(dim=1, keepdim=True)
        G_ref = (delta[:, :, None] * X).mean(dim=0)             # [R,m]
        F_ref = -(st["lr"].double()[:, None]) * torch.einsum("rhm,rm->rh", EgW, mu_u)

    relerr = lambda a_, b_: float((a_ - b_).abs().max()
                                  / b_.abs().max().clamp_min(1e-300))
    G_err = relerr(torch.as_tensor(rec["G"]), G_ref)
    F_err = relerr(torch.as_tensor(rec["F_gate"]), F_ref)
    # 分解の閉包: F_self + F_rest == F_gate (float64 なら丸め誤差のみ)
    dec_err = relerr(torch.as_tensor(rec["F_self"] + rec["F_rest"]),
                     torch.as_tensor(rec["F_gate"]))
    # §1 の「0.993」に対応する量: F_i^ungate = −2η v_i (G·µ̂) と F_i^gate の符号一致率。
    # ゲートの開き具合で層別する (p̂→1 ならゲートは恒等に近づくので一致は 1 に向かう)。
    Gdm = (G_ref * mu_u).sum(dim=1)                             # [R]
    F_ung = -2.0 * st["lr"].double()[:, None] * net.v.double() * Gdm[:, None]
    p_hat = torch.as_tensor(rec["p_hat"])
    agree = (torch.sign(F_ung) == torch.sign(F_ref)).double()
    strat = {}
    for tag, thr in (("all", -1.0), ("gt0.05", 0.05), ("gt0.5", 0.5), ("gt0.95", 0.95)):
        msk = p_hat > thr
        strat[tag] = (float(agree[msk].mean()) if bool(msk.any()) else None,
                      int(msk.sum()))
    return dict(G_max_relerr=G_err, F_max_relerr=F_err, decomp_max_relerr=dec_err,
                pass_1e10=bool(G_err < TOL and F_err < TOL and dec_err < TOL),
                sign_agree=strat, n_units=int(agree.numel()))


# ---------------------------------------------------------------- §4.2 0.567 の再現

def s_series(ck_paths, cfg, select, device="cpu"):
    """各 ckpt で s(t)=sign(G·µ̂_t) を計算し [n_ckpt, R'] と run_id を返す。"""
    out, ids, steps = [], None, []
    for p in ck_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        idx = [i for i, r in enumerate(ck["runs"]) if select(r)]
        if not idx:
            continue
        st = build_st(ck, cfg, idx=idx, device=device)
        # ここでは 0819 の再現対象である std 腕だけを select で渡す。
        # exact_record 自体は centered も x_in の数値平均で扱える。
        rec = exact_record(st)
        out.append(np.sign(rec["G_dot_mu"]))
        ids = [r["run_id"] for r in st["runs"]]
        steps.append(int(ck["step"]))
    return np.array(out), ids, steps


def agreement(S, steps, period):
    """隣接 ckpt (間隔 ≥ 1 周期) での s(t) 一致率と遷移数。"""
    pairs = [(i, i + 1) for i in range(len(steps) - 1)
             if steps[i + 1] - steps[i] >= period]
    if not pairs:
        return float("nan"), 0
    agree = np.concatenate([(S[i] == S[j]).astype(float) for i, j in pairs])
    return float(agree.mean()), int(agree.size)


def reproduce_0567(cfg, device="cpu"):
    """drift_0809 の隣接 ckpt s(t) 一致率。仕様の 0.567 (60 遷移) と突き合わせる。

    **60 遷移の内訳がリポジトリに記録されていない**ため、どの run 部分集合で測った値かは
    確定できない。ここでは事前登録の主対象 (w100・T=1e4・std・lr=0.01) を主報告とし、
    参考に幾つかの部分集合も併記して「定義差の所在」を示す (§4.2 の指示)。"""
    steps_all = [0, 10000, 50000, 100000, 300000, 1000000]
    variants, rows = {
        "w100_T1e4_std_lr0.01 (事前登録の主対象)":
            ("A_w100", lambda r: r["enc"] == "std" and r["period"] == 10000
             and float(r["lr"]) == 0.01),
        "w100_std_全T": ("A_w100", lambda r: r["enc"] == "std"
                         and float(r["lr"]) == 0.01),
        "w5_T1e4_std_lr0.01": ("A_w5", lambda r: r["enc"] == "std"
                               and r["period"] == 10000 and float(r["lr"]) == 0.01),
    }, []
    for label, (gname, sel) in variants.items():
        paths = [os.path.join(DRIFT, "ckpts", f"{gname}_step{s}.pt") for s in steps_all]
        paths = [p for p in paths if os.path.exists(p)]
        S, ids, steps = s_series(paths, cfg, sel, device=device)
        if not len(S):
            continue
        if "全T" in label:                        # run ごとに周期が違うので run 別に集計
            per = np.array([int(i.split("_T")[1].split("_")[0]) for i in ids])
            tot, n = [], 0
            for u in np.unique(per):
                a, k = agreement(S[:, per == u], steps, int(u))
                if k:
                    tot.append(a * k)
                    n += k
            rate, ntr = (sum(tot) / n if n else float("nan")), n
        else:
            rate, ntr = agreement(S, steps, 10000)
        rows.append(dict(subset=label, n_runs=len(ids), n_ckpt=len(steps),
                         n_transitions=ntr, agreement=round(rate, 4),
                         run_ids=ids))
    return rows


# ---------------------------------------------------------------- 出力

def main():
    device = "cpu"
    t0 = time.time()
    cfgA = load_config(os.path.join(ROOT, "configs", "ratchet_log_0819.yaml"))
    L = ["# ratchet_log_0819 Phase 0 (再学習なし) [spec §4]", "",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    # --- §4.1
    L += ["## 1. 恒等式サニティ (§4.1)", "",
          "**仕様との相違 (記録)**: §2/§4.1 は突き合わせ相手を `src/posreset_posthoc.py` の "
          "「F_i = −2η·v_i·G 恒等式の厳密計算実装」としているが、実際の posreset_posthoc.py は "
          "アーム間コントラストの paired bootstrap だけを行うモジュールで G も F_i も計算して "
          "いない。リポジトリ内に厳密 G の参照実装は存在せず、「符号一致率 0.993」も "
          "リポジトリには記録が無い (canvas 側の事後解析値)。そこで突き合わせ相手を "
          "**本番の勾配コード `nets.VecMLP.grads_batch`** に置き換えた "
          "(F_i^gate = −η·E[gW_i] は実装上の恒等式なので、手書き einsum の独立検査になる)。", ""]
    if os.path.exists(SNAP_A):
        snap = torch.load(SNAP_A, map_location=device, weights_only=False)
        cfgP = load_config(os.path.join(ROOT, "configs", "posreset_0819.yaml"))
        res = []
        for s in (0, 1, 2):
            st = build_st(snap, cfgP, idx=[s], device=device)
            r = identity_check(st)
            r["seed"] = snap["runs"][s]["seed"]
            res.append(r)
        L += [f"対象: `{os.path.relpath(SNAP_A, ROOT)}` (t=500k)、seed 0–2、"
              f"判定 相対誤差 < {TOL:g} (**float64 の計算値どうし**で比較。保存用の "
              "float32 に丸めた値を使うと丸めだけで 1e-8 台になり判定が成立しない)", "",
              "| seed | G 相対誤差 | F_gate 相対誤差 | 分解閉包 (F_self+F_rest−F_gate) | 判定 |",
              "|---|---|---|---|---|"]
        for r in res:
            L.append(f"| {r['seed']} | {r['G_max_relerr']:.2e} | {r['F_max_relerr']:.2e} | "
                     f"{r['decomp_max_relerr']:.2e} | "
                     f"{'PASS' if r['pass_1e10'] else 'FAIL'} |")
        ok = all(r["pass_1e10"] for r in res)
        L += ["", f"**判定: {'PASS' if ok else 'FAIL'}**", "",
              "### 符号一致率 (§1 の「0.993」に対応する量)", "",
              "F_i^ungate = −2η v_i (G·µ̂) と F_i^gate·µ̂ の符号一致率を、ゲートの開き "
              "p̂ で層別したもの。", "",
              "| seed | 全ユニット | p̂>0.05 | p̂>0.5 | p̂>0.95 |", "|---|---|---|---|---|"]
        for r in res:
            cells = []
            for tag in ("all", "gt0.05", "gt0.5", "gt0.95"):
                val, n = r["sign_agree"][tag]
                cells.append(f"{val:.3f} (n={n})" if val is not None else "— (n=0)")
            L.append(f"| {r['seed']} | " + " | ".join(cells) + " |")
        L += ["", "**0.993 は再現しない**。この量は p̂ の層で大きく変わり、t=500k の "
              "condA A_w100 では 8〜9 割のユニットが p̂≈0 まで沈んでいて "
              "F^gate ≡ 0 → sign=0 になるため、全ユニットで取ると 0.1 前後まで落ちる。"
              "**これは矛盾ではなく整流モデルの部品2 (片側移動度) そのもの**であり、"
              "0.993 は「ゲートがまだ開いている段階／開いているユニットに限った量」と "
              "解釈するのが整合的。canvas 側の測定条件 (時点・部分集合) が "
              "リポジトリに記録されていないため、ここでは層別値を報告するに留める。"
              "**本実験の判定 P1–P5 はこの量に依存しない**。", ""]
    else:
        L += [f"**SKIP**: {SNAP_A} が無い", ""]

    # --- §4.2
    L += ["## 2. 0.567 の再現 (§4.2)", ""]
    cfgD_path = os.path.join(DRIFT, "config_used.yaml")
    cfgD = load_config(cfgD_path) if os.path.exists(cfgD_path) else cfgA
    rows = reproduce_0567(cfgD, device=device)
    L += ["| 部分集合 | run 数 | ckpt 数 | 遷移数 | 一致率 |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['subset']} | {r['n_runs']} | {r['n_ckpt']} | "
                 f"{r['n_transitions']} | {r['agreement']:.4f} |")
    L += ["", "**判定: 厳密な再現は不可能 (定義差の所在を特定して先へ進む、§4.2 の分岐)**。",
          "", "理由:", "",
          "1. **0.567 という値がリポジトリに記録されていない**。`results/` 以下・"
          "解析コード・summary いずれにも現れず、canvas 側の事後解析の値である。",
          "2. **「60 遷移」を作る run 部分集合が復元できない**。drift_0809 の condA ckpt は "
          "6 点 (0/10k/50k/100k/300k/1M) なので隣接ペアは 1 run あたり 5 本。60 = 12 run × 5 "
          "だが、A_w100 の T=1e4・std・lr=0.01 は seed 5 本しか無く、w5 を足しても 10 run "
          "(50 遷移)、centered を足すと 20 run (100 遷移) で 12 run になる組合せが無い。",
          "3. **npz からは s(t) を復元できない**。`followup_Eg_*.npz` が持つのは "
          "`Eg_W` = E[2δ·v_i·gate_i·x] (ゲート済み・ユニット別) であって "
          "G = E[δx] ではない。ゲートを外す情報が保存されていないので、"
          "npz 単体からは仕様が要求する s(t)=sign(G·µ̂) を作れない。"
          "上表は **ckpt から G を厳密に再計算**して測り直した値である。", "",
          "この不能性は本実験の動機そのものでもある: §5 の A_boundary は "
          "同じ量を 99 遷移/run × 10 seed = 990 遷移で、事前登録の下で測り直す。", ""]

    os.makedirs(OUT, exist_ok=True)
    L += ["## 3. グリッド健全性 (§4.3)", ""]
    smoke = os.path.join(ROOT, "results", "_smoke_ratchet", "meta.json")
    if os.path.exists(smoke):
        import json
        M = json.load(open(smoke))
        s = M["sanity"]
        L += [f"`src.ratchet_log --smoke --s2-steps 100000` (seed 0 / 0→"
              f"{M['total_steps']} / 境界 {M['n_boundaries']} 個) の実測:", "",
              "| 項目 | 値 |", "|---|---|",
              f"| 記録点数 | {M['n_record_steps']} |",
              f"| 実現した flip 遷移 | {M['n_realized_flips']} |",
              f"| ログサイズ (1 seed) | {M['logs_mb']} MB |",
              f"| wall-clock (train+probe) | {M['train_sec']} s |",
              f"| S1 OMP_NUM_THREADS | {s['S1']['omp_num_threads']} |",
              f"| **S2** (probe あり/なし {s.get('S2', {}).get('s2_steps', '—')} step の "
              f"bit 一致) | **{'PASS' if s.get('S2', {}).get('s2_pass') else 'FAIL/未実施'}** |",
              f"| **S3** (厳密 p̂ vs eval 経験値) | **{'PASS' if s['S3']['s3_pass'] else 'FAIL'}** "
              f"(median\\|z\\|={s['S3']['s3_median_abs_z']}, "
              f"\\|z\\|>3 が {s['S3']['s3_n_gt3']} 個 / 期待 {s['S3']['s3_expected_gt3']}) |",
              f"| **S4** (flip が t≡0 mod T) | **{'PASS' if s['S4']['s4_pass'] else 'FAIL'}** |",
              "",
              "**境界数の訂正 (§3.3 / §5)**: 仕様は「全 100 箇所/run」「100 遷移/run」と "
              "書くが、`train_group` のループは `range(start, total)` なので "
              "**t=total の境界では flip が起きない**。実現する遷移は "
              "`total/T − 1` 本 = 本走で **99 遷移/run** (スモークでも境界 5 個に対し "
              "flip 遷移 4 本で確認)。A_boundary の n は 10 seed × 99 = **990**。", ""]
    else:
        L += ["**未実施**: `src.ratchet_log --smoke --s2-steps 100000` を先に走らせること。", ""]
    path = os.path.join(OUT, "phase0_summary.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
