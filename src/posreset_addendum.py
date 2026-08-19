"""posreset_0819 追補アームの専用ランナー [posreset_0819_add §2–5]。

  OMP_NUM_THREADS=1 ./.venv/bin/python -m src.posreset_addendum \
      --config configs/posreset_0819_add.yaml

本体の完全再開 snapshot を読み取り専用の入力として使い、レジーム A の
``posflip / vzero / dirkeep`` を 1 本ずつ逐次実行する。トランクを作る経路は意図的に
持たず、snapshot が無い場合は即座に停止する。本体 `results/posreset_0819/` には
一切書き込まない [posreset_0819_add §3]。

介入は float64 で構成して S3a を 1e-12 で検査し、学習再開時だけ float32 に戻す。
S3a/S3b/S4a/S2a には mutant を注入する自己検査を毎回先に走らせ、壊れた入力が
実際に FAIL になることを確認してから本走へ進む [posreset_0819_add §5]。
"""
import argparse
import copy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from .common import (ROOT, build_runs, group_name, group_runs, load_config,
                     pick_device, resolve_outdir)
from .posreset import (_mask_hash, _max0, _sha, build_arm_params, fresh_draws,
                       make_probe, new_acc, treated_and_pre_metrics, write_traj)
from .rank_int import arm_runs
from .train import load_resume, setup_group, train_group


ADD_ARMS = ("posflip", "vzero", "dirkeep")
MAIN_ARMS = ("posonly", "dironly", "full")


# ---------------------------------------------------------------- 介入 [§2]

def build_add_arm_params(net, G32, treated, norm_guard):
    """追補 3 アームを float64 で構成する [posreset_0819_add §2]。

      posflip: w ← ‖g‖·(−w/‖w‖), b ← 0, v ← 0
      vzero  : w, b は保持             , v ← 0
      dirkeep: w ← ‖w‖·(g/‖g‖) , b ← 0, v ← 0

    posflip の ``‖w‖ < norm_guard`` は本体 posonly と同じく full (w=g) へ
    フォールバックする。treated 外と出力バイアス c は触らない。
    """
    W, b, v, G = net.W.double(), net.b.double(), net.v.double(), G32.double()
    wn, gn = W.norm(dim=2), G.norm(dim=2)
    assert bool((gn > 0).all()), "fresh draw g_i にゼロノルムのユニットが出た"
    guard = treated & (wn < float(norm_guard))
    tm, gm = treated[:, :, None], guard[:, :, None]
    z_b = torch.zeros_like(b)
    v_new = torch.where(treated, torch.zeros_like(v), v)

    W_flip = torch.where(tm, -(gn / wn.clamp_min(1e-300))[:, :, None] * W, W)
    W_flip = torch.where(gm, G, W_flip)
    W_keep = torch.where(tm, (wn / gn)[:, :, None] * G, W)

    return {
        "posflip": (W_flip, torch.where(treated, z_b, b), v_new),
        "vzero": (W.clone(), b.clone(), v_new),
        "dirkeep": (W_keep, torch.where(treated, z_b, b), v_new),
    }, guard


def _branch_hash(W, b, v, c, untouched):
    """seed 1 本の非 treated W/b/v と c をまとめた hash [§5 S2a]。"""
    parts = [_sha(W[untouched]), _sha(b[untouched]), _sha(v[untouched]), _sha(c)]
    return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()


def check_s3a(i, net, G32, arms64, treated, guard, tol):
    """posflip の厳密反転・fresh norm・ガードを検査する [§5 S3a]。"""
    t, g = treated[i], guard[i]
    regular = t & ~g
    pre = net.W[i].double()[regular]
    post = arms64["posflip"][0][i][regular]
    gd = G32[i].double()[regular]
    cos = (pre * post).sum(-1) / (pre.norm(dim=-1) * post.norm(dim=-1)).clamp_min(1e-300)
    cos_err = _max0((cos + 1.0).abs())
    gn = gd.norm(dim=-1)
    norm_err = _max0((post.norm(dim=-1) - gn).abs() / gn.clamp_min(1e-300))
    guard_err = _max0((arms64["posflip"][0][i][g] - G32[i].double()[g]).abs())
    _W, b, v = (x[i] for x in arms64["posflip"])
    b_ok, v_ok = bool((b[t] == 0).all()), bool((v[t] == 0).all())
    return {
        "s3a_n_regular": int(regular.sum()),
        "s3a_posflip_cos_err_f64": cos_err,
        "s3a_posflip_norm_relerr_f64": norm_err,
        "s3a_guard_full_exact_f64": guard_err,
        "s3a_posflip_bias_zero_ok": b_ok,
        "s3a_posflip_readout_zero_ok": v_ok,
        "s3a_pass": bool(cos_err < tol and norm_err < tol and guard_err == 0.0
                          and b_ok and v_ok),
    }


def check_s3b(i, net, arms32, treated):
    """vzero の treated W/b bit 不変・v==0 を検査する [§5 S3b]。"""
    t = treated[i]
    W, b, v = (x[i] for x in arms32["vzero"])
    # torch.equal は +0/-0 を同値扱いするため、生バイト hash で「bit 不変」を検査する。
    w_ok = _sha(W[t]) == _sha(net.W[i][t])
    b_ok = _sha(b[t]) == _sha(net.b[i][t])
    v_ok = bool((v[t] == 0).all())
    return {
        "s3b_vzero_w_bit_ok": bool(w_ok),
        "s3b_vzero_b_bit_ok": bool(b_ok),
        "s3b_vzero_readout_zero_ok": v_ok,
        "s3b_pass": bool(w_ok and b_ok and v_ok),
    }


def check_dirkeep(i, net, G32, arms64, arms32, treated, tol):
    """副次アーム dirkeep の表どおりの実装を数値保証する [§2]。"""
    t = treated[i]
    pre, post, gd = net.W[i].double()[t], arms64["dirkeep"][0][i][t], G32[i].double()[t]
    pn, qn, gn = pre.norm(dim=-1), post.norm(dim=-1), gd.norm(dim=-1)
    norm_err = _max0((qn - pn).abs() / pn.clamp_min(1e-300))
    nz = pn > 0
    cos = (post[nz] * gd[nz]).sum(-1) / (post[nz].norm(dim=-1) * gn[nz]).clamp_min(1e-300)
    cos_err = _max0((cos - 1.0).abs())
    _W32, b32, v32 = (x[i] for x in arms32["dirkeep"])
    b_ok, v_ok = bool((b32[t] == 0).all()), bool((v32[t] == 0).all())
    return {
        "dirkeep_norm_relerr_f64": norm_err,
        "dirkeep_cos_g_err_f64": cos_err,
        "dirkeep_bias_zero_ok": b_ok,
        "dirkeep_readout_zero_ok": v_ok,
        "dirkeep_definition_pass": bool(norm_err < tol and cos_err < tol and b_ok and v_ok),
    }


def check_s2a(i, net, main32, add32, treated, c_ref):
    """snapshot・追補・本体の全アームで非 treated hash を突き合わせる [§5 S2a]。

    本体は分岐直後 state を保存していないため、未加工 snapshot を独立な正とする。
    本体 3 アームは既存 S3 でも同じ snapshot と照合済みであり、本検査はその構築規約を
    同じ入力上で再現して追補 3 アームまで直接つなぐ。
    """
    untouched = ~treated[i]
    # c_ref は snapshot dict 由来の独立 clone。net.c を自分自身と比較すると、
    # c が誤って in-place 変更されても全アーム同じ値で空虚に PASS するため使わない。
    c = net.c[i]
    c_ok = _sha(c) == _sha(c_ref[i])
    snapshot_hash = _branch_hash(net.W[i], net.b[i], net.v[i], c_ref[i], untouched)
    main_hash = {
        a: _branch_hash(main32[a][0][i], main32[a][1][i], main32[a][2][i], c, untouched)
        for a in MAIN_ARMS
    }
    add_hash = {
        a: _branch_hash(add32[a][0][i], add32[a][1][i], add32[a][2][i], c, untouched)
        for a in ADD_ARMS
    }
    values = [snapshot_hash] + list(main_hash.values()) + list(add_hash.values())
    ok = len(set(values)) == 1 and c_ok
    return {
        "s2a_nontreated_hash": snapshot_hash,
        "s2a_snapshot_hash_match": bool(all(v == snapshot_hash for v in values[1:])),
        "s2a_main_arm_hash_match": bool(len(set(main_hash.values())) == 1),
        "s2a_add_arm_hash_match": bool(len(set(add_hash.values())) == 1),
        "s2a_c_snapshot_match": bool(c_ok),
        "s2a_pass": bool(ok),
    }


def check_s4a(current_hash, body_hash):
    """treated hash を本体の記録と直接比較する [§5 S4a]。"""
    return bool(str(current_hash) == str(body_hash))


def _clone_arms(arms):
    return {a: tuple(x.clone() for x in vals) for a, vals in arms.items()}


# ---------------------------------------------------------------- mutant 検出力 [§5]

def mutant_selftest():
    """S3a/S3b/S4a/S2a を意図的に壊し、各検査が FAIL することを確認する。"""
    net = SimpleNamespace(
        W=torch.tensor([[[1.0, 2.0], [-2.0, 1.0], [0.5, -1.5]]], dtype=torch.float32),
        b=torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32),
        v=torch.tensor([[0.7, -0.8, 0.9]], dtype=torch.float32),
        c=torch.tensor([0.1], dtype=torch.float32),
    )
    G = torch.tensor([[[0.3, -0.9], [1.2, 0.4], [-0.7, 0.6]]], dtype=torch.float32)
    treated = torch.tensor([[True, True, False]])
    tol = 1.0e-12
    add64, guard = build_add_arm_params(net, G, treated, 1.0e-8)
    add32 = {a: tuple(x.float() for x in vals) for a, vals in add64.items()}
    main64, _ = build_arm_params(net, G, treated, 1.0e-8)
    main32 = {a: tuple(x.float() for x in vals) for a, vals in main64.items()}

    c_ref = net.c.clone()
    baseline = {
        "S3a": check_s3a(0, net, G, add64, treated, guard, tol)["s3a_pass"],
        "S3b": check_s3b(0, net, add32, treated)["s3b_pass"],
        "S2a": check_s2a(0, net, main32, add32, treated, c_ref)["s2a_pass"],
        "S4a": check_s4a(_mask_hash(treated[0].numpy()),
                           _mask_hash(treated[0].numpy())),
    }
    records = []

    def record(check, mutant, failed):
        records.append({"check": check, "mutant": mutant,
                        "baseline": "PASS" if baseline[check] else "FAIL",
                        "mutant_result": "FAIL" if failed else "PASS",
                        "detected": bool(baseline[check] and failed)})

    x = _clone_arms(add64)
    x["posflip"][0][0, 0].mul_(-1.0)
    record("S3a", "posflip の treated 方向を反転前へ戻す",
           not check_s3a(0, net, G, x, treated, guard, tol)["s3a_pass"])
    x = _clone_arms(add64)
    x["posflip"][0][0, 0].mul_(1.01)
    record("S3a", "posflip の fresh norm を 1% 膨らませる",
           not check_s3a(0, net, G, x, treated, guard, tol)["s3a_pass"])

    for field, idx, label in [(0, (0, 0, 0), "treated W を変更"),
                              (1, (0, 0), "treated b を変更"),
                              (2, (0, 0), "treated v を非ゼロ化")]:
        x = _clone_arms(add32)
        x["vzero"][field][idx] += 0.125
        record("S3b", label, not check_s3b(0, net, x, treated)["s3b_pass"])

    x = _clone_arms(add32)
    x["dirkeep"][0][0, 2, 0] += 0.125
    record("S2a", "非 treated W を 1 要素変更",
           not check_s2a(0, net, main32, x, treated, c_ref)["s2a_pass"])
    net_bad = copy.deepcopy(net)
    net_bad.c[0] += 0.125
    record("S2a", "出力 bias c を変更",
           not check_s2a(0, net_bad, main32, add32, treated, c_ref)["s2a_pass"])
    mutated_mask = treated[0].numpy().copy()
    mutated_mask[0] = ~mutated_mask[0]
    record("S4a", "treated mask の 1 bit を反転",
           not check_s4a(_mask_hash(treated[0].numpy()), _mask_hash(mutated_mask)))

    if not all(r["detected"] for r in records):
        raise AssertionError(f"mutant 検出力テスト失敗: {records}")
    return {"result": "PASS", "n_mutants": len(records), "records": records}


# ---------------------------------------------------------------- 入力境界と本走

def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_manifest(path):
    """本体結果が実行前後で byte 不変だったことを確認する read-only manifest。"""
    out = {}
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, path)
            out[rel] = {"size": os.path.getsize(p), "sha256": _file_sha256(p)}
    return out


def _manifest_digest(manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalise_snapshot_run(run):
    r = dict(run)
    r.pop("arm", None)
    if str(r.get("run_id", "")).endswith("_cont"):
        r["run_id"] = r["run_id"][:-5]
    return r


def _validate_source_config(cfg, source_results):
    """snapshot が保存しない eval_fixed も再現できるよう本体 config と完全照合する。"""
    p = os.path.join(source_results, "config_used.yaml")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"本体 config_used.yaml が無い: {p}")
    with open(p) as fh:
        body = yaml.safe_load(fh)
    keys = ("common", "methods", "posreset", "condA", "condB")
    mismatch = [k for k in keys if cfg.get(k) != body.get(k)]
    if mismatch:
        raise ValueError(f"本体 config_used.yaml と追補 config が不一致: {mismatch}")


def run(config_path):
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("S1 FAIL: OMP_NUM_THREADS=1 を環境変数として指定すること")

    cfg = load_config(config_path)
    if "posreset_add" not in cfg:
        raise KeyError("posreset_add ブロックが無い (追補は明示 opt-in のみ)")
    device = pick_device(cfg)
    if device != "cpu":
        raise ValueError("本体 snapshot と同一の CPU 実行を要求する")
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    P, A = cfg["posreset"], cfg["posreset_add"]
    arms = list(A["arms"])
    if arms != list(ADD_ARMS):
        raise ValueError(f"追補アームは順序込みで {list(ADD_ARMS)} を要求: {arms}")
    t_int, post = int(P["t_int"]), int(P["post_steps"])
    total = t_int + post
    outdir = resolve_outdir(config_path)
    source_results = os.path.abspath(os.path.join(ROOT, A["source_results"]))
    snapshot_path = os.path.abspath(os.path.join(source_results, A["source_snapshot"]))
    if not os.path.isfile(snapshot_path):
        raise FileNotFoundError(
            f"本体 snapshot が無い: {snapshot_path}。トランク再学習にはフォールバックしない")
    if os.path.commonpath([os.path.abspath(outdir), source_results]) == source_results:
        raise ValueError("出力先が本体 results 配下に入っている")
    _validate_source_config(cfg, source_results)

    source_before = _tree_manifest(source_results)
    source_digest = _manifest_digest(source_before)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    mutants = mutant_selftest()
    with open(os.path.join(outdir, "sanity_mutants.json"), "w") as fh:
        json.dump(mutants, fh, indent=1, ensure_ascii=False)
    print(f"mutant sanity: PASS ({mutants['n_mutants']} mutants detected)", flush=True)

    groups = group_runs(build_runs(cfg))
    candidates = [(g, rs) for g, rs in groups.items() if g[0] == "A"]
    if len(candidates) != 1:
        raise ValueError(f"レジーム A の group は 1 本を要求: {[g for g, _ in candidates]}")
    gkey, base_runs = candidates[0]
    if len(base_runs) != 10 or [r["seed"] for r in base_runs] != list(range(10)):
        raise ValueError("seed 0–9 の順序を要求する")
    gbase = group_name(gkey)
    if gbase != "A_w100":
        raise ValueError(f"本体 group は A_w100 を要求: {gbase}")

    print(f"=== [{gbase}] source snapshot (read-only): {snapshot_path}", flush=True)
    snap = torch.load(snapshot_path, map_location=device, weights_only=False)
    if int(snap.get("step", -1)) != t_int:
        raise ValueError(f"snapshot step {snap.get('step')} != t_int {t_int}")
    snap_runs = [_normalise_snapshot_run(r) for r in snap.get("runs", [])]
    if snap_runs != base_runs:
        raise ValueError("snapshot の run/seed 順が追補 config と一致しない")

    with open(os.path.join(source_results, "meta.json")) as fh:
        body_meta = json.load(fh)
    body_A = next(x for x in body_meta["sanity"] if x["regime"] == "A")
    snap_hashes = {k: _sha(v) for k, v in snap["net"].items()}
    snap_hashes["running_mean"] = _sha(snap["running_mean"])
    if snap_hashes != body_A["snapshot_sha256"]:
        raise ValueError("snapshot tensor hash が本体 meta.json と一致しない")

    st = setup_group(gkey, base_runs, cfg, device)
    load_resume(st, snap)
    treated, _p_hat, pre_m = treated_and_pre_metrics(st, cfg, float(P["p_hat_tau"]))
    net, (R, h) = st["net"], treated.shape
    tfrac = treated.float().mean(dim=1).cpu().numpy()
    masks = treated.cpu().numpy()
    thash = [_mask_hash(masks[i]) for i in range(R)]

    body_ilog = pd.read_csv(os.path.join(source_results, "intervention_log.csv"),
                            dtype=str, keep_default_na=False)
    body_ilog = body_ilog[body_ilog.regime == "A"].copy()
    body_ilog["seed"] = body_ilog.seed.astype(int)
    if body_ilog.seed.duplicated().any():
        raise ValueError("本体 intervention_log の A seed が重複")
    body_ilog = body_ilog.set_index("seed")
    if list(body_ilog.index.sort_values()) != list(range(10)):
        raise ValueError("本体 intervention_log の A seed が 0–9 でない")

    G32 = fresh_draws("A", R, h, st["d"], P["reset_seed_base"], device)
    add64, guard = build_add_arm_params(net, G32, treated, P["norm_guard"])
    add32 = {a: tuple(x.float() for x in vals) for a, vals in add64.items()}
    main64, _main_guard = build_arm_params(net, G32, treated, P["norm_guard"])
    main32 = {a: tuple(x.float() for x in vals) for a, vals in main64.items()}

    ilog = []
    for i, r in enumerate(base_runs):
        s = int(r["seed"])
        body_hash = str(body_ilog.loc[s, "treated_hash"])
        row = {
            "regime": "A", "exp": r["exp"], "width": r["width"], "seed": s,
            "base_run_id": r["run_id"], "t_int": t_int, "h": h,
            "n_treated": int(masks[i].sum()), "treated_frac": float(tfrac[i]),
            "n_guard_fallback": int(guard[i].sum()), "treated_hash": thash[i],
            "body_treated_hash": body_hash,
            "pre_dead_frac": float(pre_m["dead_frac"][i]),
            "pre_eval_loss": float(pre_m["eval_loss"][i]),
        }
        row.update(check_s3a(i, net, G32, add64, treated, guard, float(P["s3_tol_f64"])))
        row.update(check_s3b(i, net, add32, treated))
        row.update(check_dirkeep(i, net, G32, add64, add32, treated,
                                 float(P["s3_tol_f64"])))
        row.update(check_s2a(i, net, main32, add32, treated, snap["net"]["c"]))
        row["s4a_base_run_id_match"] = str(body_ilog.loc[s, "base_run_id"]) == r["run_id"]
        row["s4a_t_int_match"] = int(body_ilog.loc[s, "t_int"]) == t_int
        row["s4a_h_match"] = int(body_ilog.loc[s, "h"]) == h
        row["s4a_n_treated_match"] = int(body_ilog.loc[s, "n_treated"]) == int(masks[i].sum())
        row["s4a_treated_hash_match"] = check_s4a(thash[i], body_hash)
        row["s4a_pass"] = bool(all(row[k] for k in
                                   ("s4a_base_run_id_match", "s4a_t_int_match",
                                    "s4a_h_match", "s4a_n_treated_match",
                                    "s4a_treated_hash_match")))
        row["sanity_all_pass"] = bool(row["s3a_pass"] and row["s3b_pass"]
                                      and row["dirkeep_definition_pass"] and row["s2a_pass"]
                                      and row["s4a_pass"])
        ilog.append(row)
    if not all(r["sanity_all_pass"] for r in ilog):
        raise SystemExit("追補サニティ FAIL。アーム実行を中止")
    print("S2a/S3a/S3b/S4a: PASS (10/10 seeds)", flush=True)

    probe_steps = list(range(t_int, total + 1, int(P["probe_every"])))
    arm_elapsed = {}
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    for arm in arms:  # 単一プロセスで 1 本ずつ逐次 [§3]
        snap_arm = copy.deepcopy(snap)
        W, b, v = add32[arm]
        snap_arm["net"]["W"] = W.clone()
        snap_arm["net"]["b"] = b.clone()
        snap_arm["net"]["v"] = v.clone()
        acc = new_acc(R)
        gname = f"{gbase}_{arm}"
        print(f"=== [{gbase}] arm {arm} {t_int}->{total}", flush=True)
        _, elapsed = train_group(
            gkey, arm_runs(base_runs, arm), cfg, device, outdir,
            total_steps=total, ckpts=[], gname=gname, start_step=t_int,
            resume_state=snap_arm, probe=make_probe(treated, acc), probe_steps=probe_steps)
        write_traj(outdir, "A", arm, base_runs, treated, acc, t_int, thash)
        arm_elapsed[arm] = round(elapsed, 1)
        print(f"    done {elapsed:.1f}s ({post / max(elapsed, 1e-9):.0f} steps/s)", flush=True)

    pd.DataFrame(ilog).to_csv(os.path.join(outdir, "intervention_log.csv"), index=False)
    source_after = _tree_manifest(source_results)
    if source_after != source_before:
        raise SystemExit("本体 results/posreset_0819 が実行中に変化した")
    meta = {
        "elapsed_sec": round(time.time() - t0, 1), "arm_elapsed_sec": arm_elapsed,
        "device": device, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started": run_started, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(), "arms": arms,
        "regimes": ["A"], "trunk_retrained": False,
        "source_results": os.path.relpath(source_results, ROOT),
        "source_snapshot": os.path.relpath(snapshot_path, ROOT),
        "source_snapshot_file_sha256": _file_sha256(snapshot_path),
        "source_manifest_n_files": len(source_before),
        "source_manifest_sha256_before": source_digest,
        "source_manifest_sha256_after": _manifest_digest(source_after),
        "source_readonly_pass": True, "snapshot_tensor_sha256": snap_hashes,
        "mutant_sanity": mutants,
        "sanity": {
            "S1": os.environ.get("OMP_NUM_THREADS") == "1" and torch.get_num_threads() == 1,
            "S2a": all(r["s2a_pass"] for r in ilog),
            "S3a": all(r["s3a_pass"] for r in ilog),
            "S3b": all(r["s3b_pass"] for r in ilog),
            "S4a": all(r["s4a_pass"] for r in ilog),
        },
    }
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print(f"source read-only manifest: PASS ({len(source_before)} files, {source_digest})", flush=True)
    print("ALL DONE", flush=True)
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/posreset_0819_add.yaml")
    ap.add_argument("--selftest", action="store_true",
                    help="mutant 検出力だけを実行し、学習はしない")
    args = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("S1 FAIL: OMP_NUM_THREADS=1 を指定すること")
    if args.selftest:
        out = mutant_selftest()
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return
    outdir = run(args.config)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
