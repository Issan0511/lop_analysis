"""teachw_0820 Phase 0 [spec_teachw_0820 §4]。本走の前に完了させるゲート。

  OMP_NUM_THREADS=1 .venv/bin/python -m src.teachw_phase0 --config configs/teachw_0820.yaml

検査項目:
  0-1 恒等性 : target_out_scale=1.0 の H_T=100 アームが ratchet_log_0819 と state hash 一致
  0-2 Var[y] : 各 H_T の t=0 全サポート上 Var[y_scaled] が H_T=100 比 [0.5, 2.0]
  0-3 学習可能性 : H_T=1 の 50k スモークで eval_loss_exact が初期値から低下
  0-4 probe 無擾乱 : probe あり / なしで 50k の最終 state hash が bit 一致 (追加項目)
  0-5 S2 先行確認 : 50k スモークの flip_state 軌跡 hash が seed ごとに全 6 アームで一致

**0-1 の逸脱 (記録)**: 仕様 §4-1 の字義は「seed 0 の 50k スモークが一致」だが、
学習は R 系列をベクトル化して回すので `torch.randint(..., (R, ...))` の抽選列が R に
依存する。seed 0 単独 (R=1) の軌道は seed 0–9 群 (R=10) の seed 0 成分と原理的に
一致しない。そこでアンカー側 (`results/ratchet_log_0819/meta.json` の S2 no-probe
ハッシュ = R=10・100k step・probe なし) と**同一条件**で突き合わせる。
比較相手がコミット済みの実測ハッシュそのものなので、字義版より強い検査になる。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device, build_runs, group_runs
from .ratchet_log import full_support_ro, teacher_f64, state_hash
from .teachw import Recorder, arm_cfg, arm_dir, record_steps, run_arm
from .train import setup_group, train_group

ANCHOR = "results/ratchet_log_0819/meta.json"
IDENT_STEPS = 100_000          # アンカー (ratchet_log S2) と同じ step 数
SMOKE_STEPS = 50_000           # §4-1 / §4-3 のスモーク長


def _group(cfg):
    runs = build_runs(cfg)
    groups = group_runs(runs)
    if len(groups) != 1:
        raise ValueError(f"単一グループ前提だが {len(groups)} 個")
    return next(iter(groups.items()))


# ---------------------------------------------------------------- 0-1 恒等性

def check_identity(cfg, device, outdir):
    """H_T=100 (out_scale=1.0) を probe なしで IDENT_STEPS 走らせ、
    ratchet_log_0819 が記録した同条件の state hash と突き合わせる。"""
    with open(os.path.join(ROOT, ANCHOR)) as fh:
        anchor = json.load(fh)["sanity"]["S2"]
    ref, ref_steps = anchor["s2_hash_no_probe"], int(anchor["s2_steps"])
    c = arm_cfg(cfg, 100)
    gkey, gruns = _group(c)
    t0 = time.time()
    st, _ = train_group(gkey, gruns, c, device, outdir, total_steps=IDENT_STEPS,
                        ckpts=[], gname="P0_identity")
    got = state_hash(st)
    diffs = sorted(k for k in ref if ref.get(k) != got.get(k))
    return dict(pass_=bool(not diffs and ref_steps == IDENT_STEPS), steps=IDENT_STEPS,
                anchor=ANCHOR, anchor_steps=ref_steps, diffs=diffs,
                out_scale=c["condA"]["target_out_scale"],
                hash_anchor=ref, hash_got=got, sec=round(time.time() - t0, 1),
                note="仕様 §4-1 の「seed 0 / 50k」からの逸脱: R が変わると抽選列が変わる"
                     "ため、アンカーと同条件 (R=10 / 100k / probe なし) で照合した")


# ---------------------------------------------------------------- 0-2 Var[y]

def measure_var_y(cfg, device):
    """各 H_T について t=0 の全サポート (2^(m-f)=32 パターン) 上の Var[y_scaled]。

    Var は 32 パターン一様分布上の母分散 (unbiased=False)。教師も学習器も
    まだ 1 step も動いていない初期状態で測る。

    **字義と意図を両方出す** [rank_int_0814 の教訓]。仕様 §4-2 は「中央値」が
    H_T=100 比 [0.5,2.0] と書くが、LTU は β=0.7 のしきい値が高く、教師ユニット j が
    周期内 (flip 固定) で変動するのは flip 側の一致数 k_j ∈ [10,14] のときだけ ――
    k_j ~ Binom(15, 1/2) なので確率 ≈ 0.151 ―― である。よって Var[y_raw] は
    「変動ユニット数 ~ Binom(H_T, 0.151)」に比例し、**期待値は O(H_T) だが分布は
    小さい H_T でゼロ過剰**になる (H_T=1 なら 85% の seed で Var=0 = 周期内で定数)。
    中央値はこのゼロ質量を拾うので、どんな乗法スケーリングでも低 H_T では 0 になる。
    交絡除去という §3 の意図に正対するのは**平均**なので、両方記録する。

    さらに t=0 の flip 状態は 1 標本にすぎない (本走は 99 回 flip する)。周期内 Var の
    **flip 状態平均** E_flip[Var[y]] は解析的に H_T·(定数) なので、スケーリング後は
    H_T に依らず一定になるはず。これを n_flip 個のランダム flip 状態で推定して併記する
    (教師は固定・入力生成器には触れない、純粋な後付け計測)。"""
    band = cfg["teachw"]["var_band"]
    rows = []
    for hd in cfg["teachw"]["hidden_values"]:
        c = arm_cfg(cfg, hd)
        gkey, gruns = _group(c)
        st = setup_group(gkey, gruns, c, device)
        with torch.no_grad():
            X = full_support_ro(st["env"]).double()            # [P,R,m]
            y = teacher_f64(st["teacher"], X)                  # [P,R]
            var = y.var(dim=0, unbiased=False).cpu().numpy()
            ymean = y.mean(dim=0).cpu().numpy()
        vflip = var_over_flip_states(st, n_flip=cfg["teachw"].get("var_n_flip", 512))
        rows.append(dict(H_T=int(hd), out_scale=c["condA"]["target_out_scale"],
                         var_flip_mean=float(vflip.mean()),
                         var_flip_per_seed=[float(v) for v in vflip],
                         var_median=float(np.median(var)), var_mean=float(var.mean()),
                         var_min=float(var.min()), var_max=float(var.max()),
                         n_zero_var=int((var == 0).sum()), n_seed=int(var.size),
                         mean_abs_median=float(np.median(np.abs(ymean))),
                         var_per_seed=[float(v) for v in var],
                         seeds=[int(r["seed"]) for r in gruns]))
    pick = lambda k: next(r[k] for r in rows if r["H_T"] == cfg["teachw"]["scale_ref"])
    ref, ref_mean, ref_flip = pick("var_median"), pick("var_mean"), pick("var_flip_mean")
    for r in rows:
        r["ratio_vs_ref"] = float(r["var_median"] / ref) if ref else float("nan")
        r["in_band"] = bool(band[0] <= r["ratio_vs_ref"] <= band[1])
        r["ratio_mean"] = float(r["var_mean"] / ref_mean) if ref_mean else float("nan")
        r["in_band_mean"] = bool(band[0] <= r["ratio_mean"] <= band[1])
        r["ratio_flip"] = float(r["var_flip_mean"] / ref_flip) if ref_flip else float("nan")
        r["in_band_flip"] = bool(band[0] <= r["ratio_flip"] <= band[1])
    return dict(pass_=all(r["in_band"] for r in rows),
                pass_intent=all(r["in_band_mean"] for r in rows),
                pass_flip=all(r["in_band_flip"] for r in rows), band=list(band),
                n_flip=int(cfg["teachw"].get("var_n_flip", 512)),
                ref_var_median=float(ref), ref_var_mean=float(ref_mean),
                ref_var_flip=float(ref_flip), rows=rows)


def var_over_flip_states(st, n_flip=512, chunk=64, seed=20260820):
    """周期内 Var[y_scaled] の flip 状態平均 E_flip[Var[y]] を n_flip 標本で推定。

    st は setup_group 済みの**使い捨て**状態 (Phase 0 の計測専用に構築したもの)。
    env の状態は読むだけ、乱数も専用 generator なので学習側のストリームには触れない。"""
    env, teacher = st["env"], st["teacher"]
    g = torch.Generator(device=env.device)
    g.manual_seed(int(seed))
    P, R, f, mf = env.patterns.shape[0], env.R, env.f, env.m - env.f
    pat = env.patterns.double()                                       # [P, m-f]
    W, b = teacher.W.double(), teacher.b.double()
    tau, v, cout = teacher.tau.double(), teacher.v.double(), teacher.cout.double()
    sc = float(getattr(teacher, "out_scale", 1.0))
    acc = torch.zeros(R, dtype=torch.float64, device=env.device)
    done = 0
    with torch.no_grad():
        while done < n_flip:
            n = min(chunk, n_flip - done)
            fs = torch.randint(0, 2, (n, R, f), generator=g, device=env.device).double()
            X = torch.cat([fs[:, None].expand(n, P, R, f),
                           pat[None, :, None, :].expand(n, P, R, mf)], dim=3)   # [n,P,R,m]
            pre = torch.einsum("rhm,nprm->nprh", W, X) + b
            y = (((pre >= tau).double() * v).sum(-1) + cout) * sc      # [n,P,R]
            acc += y.var(dim=1, unbiased=False).sum(dim=0)             # [n,R] -> [R]
            done += n
    return (acc / n_flip).cpu().numpy()


# ---------------------------------------------------------------- 0-3/0-5 スモーク

def smoke_arms(cfg, device, outdir, steps=SMOKE_STEPS):
    """全 6 アームを steps だけ probe つきで走らせ、
    (a) eval_loss_exact の低下 [§4-3] と (b) flip_state 軌跡 hash の全アーム一致 [§7 S2]
    を確認する。本走の縮小版なので実行経路そのものの健全性検査も兼ねる。"""
    rows, hashes = [], {}
    for hd in cfg["teachw"]["hidden_values"]:
        meta = run_arm(cfg, hd, device, outdir, total_steps=steps, snapshot=False)
        hashes[int(hd)] = {int(k): v for k, v in meta["flip_hash"].items()}
        d = [np.load(os.path.join(arm_dir(outdir, hd), "logs", f"seed{s}.npz"))
             for s in sorted(hashes[int(hd)])]
        l0 = np.array([float(x["eval_loss_exact"][0]) for x in d])
        l1 = np.array([float(x["eval_loss_exact"][-1]) for x in d])
        rows.append(dict(H_T=int(hd), steps=int(steps),
                         loss0_median=float(np.median(l0)), loss1_median=float(np.median(l1)),
                         ratio_median=float(np.median(l1 / np.where(l0 > 0, l0, np.nan))),
                         n_decreased=int((l1 < l0).sum()), n_seed=int(l0.size),
                         alive_final_median=float(np.median(
                             [float(x["alive"][-1]) for x in d]))))
    # §4-3 の主対象は H_T=1 だが、全アームで低下していることを確認する
    h1 = next(r for r in rows if r["H_T"] == 1)
    learn = dict(pass_=bool(h1["n_decreased"] == h1["n_seed"]),
                 all_arms_pass=all(r["n_decreased"] == r["n_seed"] for r in rows),
                 rows=rows)

    seeds = sorted(next(iter(hashes.values())))
    per_seed = {s: sorted({hashes[hd][s] for hd in hashes}) for s in seeds}
    s2 = dict(pass_=all(len(v) == 1 for v in per_seed.values()),
              n_seed=len(seeds), n_arm=len(hashes),
              n_distinct_per_seed={int(s): len(v) for s, v in per_seed.items()},
              hash_per_seed={int(s): v[0] for s, v in per_seed.items()},
              note="全アームで flip_state 軌跡が bit 一致 = 教師幅が入力ストリームに"
                   "触れていないことの実効証明 [§7 S2]")
    return learn, s2


# ---------------------------------------------------------------- 0-4 probe 無擾乱

def check_probe_noop(cfg, device, outdir, steps=SMOKE_STEPS):
    """probe あり / なしで最終 state hash が bit 一致すること (ratchet_log §7 S2 と同型)。"""
    c = arm_cfg(cfg, 100)
    gkey, gruns = _group(c)
    R, h, f = len(gruns), int(gkey[1]), int(c["condA"]["f"])
    grid = record_steps(steps, c["teachw"]["probe_every"])
    rec = Recorder(grid, R, h, f, c["teachw"]["p_hat_tau"])
    st_a, _ = train_group(gkey, gruns, c, device, outdir, total_steps=steps, ckpts=[],
                          gname="P0_probe_on", probe=rec, probe_steps=grid)
    st_b, _ = train_group(gkey, gruns, c, device, outdir, total_steps=steps, ckpts=[],
                          gname="P0_probe_off")
    ha, hb = state_hash(st_a), state_hash(st_b)
    diffs = sorted(k for k in ha if ha[k] != hb[k])
    return dict(pass_=not diffs, steps=int(steps), diffs=diffs,
                n_probe_calls=rec.n_calls, hash_on=ha, hash_off=hb)


# ---------------------------------------------------------------- 出力

def _tbl(rows, cols, fmts):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f.format(r[c]) if not isinstance(r[c], str) else r[c]
                                     for c, f in zip(cols, fmts)) + " |")
    return "\n".join(out)


def write_summary(outdir, res, cfg):
    ok = lambda b: "PASS" if b else "**FAIL**"
    L = ["# teachw_0820 Phase 0 (本走前ゲート)", "",
         f"仕様: `specs/spec_teachw_0820.md` §4  ",
         f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
         f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(未設定)')} / "
         f"torch_num_threads={torch.get_num_threads()}", "",
         "| 項目 | 内容 | 結果 |", "|---|---|---|",
         f"| 0-1 恒等性 | out_scale=1.0 の H_T=100 が ratchet_log_0819 と state hash 一致 "
         f"({res['identity']['steps']:,} step) | {ok(res['identity']['pass_'])} |",
         f"| 0-2 Var[y] (字義: 中央値) | 各 H_T の t=0 全サポート Var[y_scaled] が "
         f"H_T=100 比 {res['var_y']['band']} | {ok(res['var_y']['pass_'])} |",
         f"| 0-2' Var[y] (t=0 平均) | 同上を seed 平均で | "
         f"{ok(res['var_y']['pass_intent'])} |",
         f"| 0-2'' Var[y] (意図: flip 平均) | E_flip[Var[y_scaled]] "
         f"({res['var_y']['n_flip']} flip 標本) が H_T=100 比 {res['var_y']['band']} | "
         f"{ok(res['var_y']['pass_flip'])} |",
         f"| 0-3 学習可能性 | H_T=1 の {SMOKE_STEPS:,} step で eval_loss_exact 低下 "
         f"({res['learn']['rows'][0]['n_decreased']}/{res['learn']['rows'][0]['n_seed']} seed) "
         f"| {ok(res['learn']['pass_'])} |",
         f"| 0-4 probe 無擾乱 (追加) | probe あり/なしで最終 state hash bit 一致 | "
         f"{ok(res['probe_noop']['pass_'])} |",
         f"| 0-5 S2 先行確認 | flip_state 軌跡 hash が seed ごとに全 6 アーム一致 | "
         f"{ok(res['s2']['pass_'])} |", "",
         "## 0-1 恒等性", "",
         f"アンカー: `{res['identity']['anchor']}` の S2 no-probe ハッシュ "
         f"({res['identity']['anchor_steps']:,} step, R=10, probe なし)。",
         f"不一致キー: {res['identity']['diffs'] or 'なし'}", "",
         f"> 逸脱: {res['identity']['note']}", "",
         "## 0-2 Var[y] (t=0, 32 パターン全サポート上の母分散)", "",
         _tbl(res["var_y"]["rows"],
              ["H_T", "out_scale", "var_median", "ratio_vs_ref", "in_band",
               "var_mean", "ratio_mean", "in_band_mean",
               "var_flip_mean", "ratio_flip", "in_band_flip", "n_zero_var"],
              ["{:d}", "{:.4f}", "{:.4g}", "{:.4f}", "{}",
               "{:.4g}", "{:.4f}", "{}", "{:.4g}", "{:.4f}", "{}", "{:d}"]), "",
         f"基準 (H_T={cfg['teachw']['scale_ref']}): 中央値 "
         f"{res['var_y']['ref_var_median']:.4g} / 平均 {res['var_y']['ref_var_mean']:.4g}。"
         f"`n_zero_var` は 32 パターン上で y が定数になった seed の数 "
         f"(= その周期の課題が定数関数)。", "",
         "**字義 (中央値) と意図 (平均) を並記する理由**: LTU は β=0.7 のしきい値が高く、"
         "教師ユニット j が周期内 (flip 固定) で変動するのは flip 側の一致数 "
         "k_j ∈ [10,14] のときだけで、k_j ~ Binom(15,1/2) だからその確率は ≈ 0.151。"
         "したがって Var[y_raw] ∝ 「変動ユニット数 ~ Binom(H_T, 0.151)」で、"
         "**期待値は O(H_T) だが分布は小さい H_T でゼロ過剰**になる "
         "(H_T=1 なら 85% の seed が Var=0 = 周期内で定数関数)。中央値はこのゼロ質量を"
         "拾うので、**どんな乗法スケーリング則でも低 H_T では 0 のまま**であり、"
         "「スケーリング則の見直し」では直らない。§3 の意図 (複雑度と損失スケールの"
         "交絡除去) に正対するのは平均側。", "",
         f"さらに t=0 の flip 状態は 1 標本にすぎない (本走では 99 回 flip する)。"
         f"`var_flip_mean` は周期内 Var のランダム flip 状態 "
         f"{res['var_y']['n_flip']} 標本平均 = E_flip[Var[y_scaled]] の推定で、"
         f"**スケーリング則が交絡を落とせているかを直接測る量**。", "",
         f"字義判定 (t=0 中央値): {ok(res['var_y']['pass_'])} / "
         f"t=0 平均: {ok(res['var_y']['pass_intent'])} / "
         f"**flip 平均 (意図): {ok(res['var_y']['pass_flip'])}**", "",
         "## 0-3 学習可能性スモーク (全アーム)", "",
         _tbl(res["learn"]["rows"],
              ["H_T", "loss0_median", "loss1_median", "ratio_median", "n_decreased",
               "alive_final_median"],
              ["{:d}", "{:.4g}", "{:.4g}", "{:.4g}", "{:d}", "{:.1f}"]), "",
         f"全アームで低下: {res['learn']['all_arms_pass']}。"
         f"`alive_final_median` は {SMOKE_STEPS:,} step 時点の参考値 (本走は 1M step)。", "",
         "## 0-4 probe 無擾乱", "",
         f"probe 呼び出し {res['probe_noop']['n_probe_calls']} 回 / "
         f"{res['probe_noop']['steps']:,} step。不一致キー: "
         f"{res['probe_noop']['diffs'] or 'なし'}。", "",
         "## 0-5 S2 先行確認 (flip_state 軌跡)", "",
         f"seed {res['s2']['n_seed']} 本 × アーム {res['s2']['n_arm']} 本。"
         f"seed ごとの異なりハッシュ数: "
         f"{sorted(set(res['s2']['n_distinct_per_seed'].values()))} (1 なら一致)。", ""]
    p = os.path.join(outdir, "phase0_summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/teachw_0820.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    outdir = args.outdir or os.path.join(ROOT, "results", "teachw_0820", "phase0")
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    res = {}
    print("--- 0-2 Var[y]", flush=True)
    res["var_y"] = measure_var_y(cfg, device)
    print("--- 0-1 恒等性", flush=True)
    res["identity"] = check_identity(cfg, device, outdir)
    print("--- 0-4 probe 無擾乱", flush=True)
    res["probe_noop"] = check_probe_noop(cfg, device, outdir, steps=args.smoke_steps)
    print("--- 0-3 / 0-5 スモーク (全 6 アーム)", flush=True)
    scfg = load_config(args.config)
    scfg["common"]["total_steps"] = int(args.smoke_steps)
    res["learn"], res["s2"] = smoke_arms(scfg, device, outdir, steps=args.smoke_steps)

    res["env"] = dict(omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
                      torch_num_threads=torch.get_num_threads(),
                      torch=torch.__version__, device=device,
                      date=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(outdir, "phase0.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str, ensure_ascii=False)
    p = write_summary(outdir, res, cfg)

    keys = [("0-1 恒等性", res["identity"]["pass_"]), ("0-2 Var[y]", res["var_y"]["pass_"]),
            ("0-3 学習可能性", res["learn"]["pass_"]),
            ("0-4 probe 無擾乱", res["probe_noop"]["pass_"]), ("0-5 S2", res["s2"]["pass_"])]
    for k, v in keys:
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"  0-2' Var[y] t=0 平均: {'PASS' if res['var_y']['pass_intent'] else 'FAIL'} / "
          f"0-2'' flip 平均 (意図): {'PASS' if res['var_y']['pass_flip'] else 'FAIL'}",
          flush=True)
    print(f"-> {p}", flush=True)
    print("ALL DONE" if all(v for _, v in keys) else "PHASE0 に FAIL あり", flush=True)


if __name__ == "__main__":
    main()
