"""mu_titration: centered running-mean rate alpha の固定グリッド走。

各 alpha は完全に新しい ``ratchet_log.run`` として起動する。そのため同一プロセスで
逐次実行しても ``setup_group`` が generator 群を同じ seed から作り直し、10 run は
alpha 間で同じ乱数実現を共有する。外部から alpha ごとに並列起動しても同じである。

実行例::

  OMP_NUM_THREADS=1 .venv/bin/python -m src.mu_titration --alpha 1e-6
  OMP_NUM_THREADS=1 .venv/bin/python -m src.mu_titration          # 全 arm を逐次
  .venv/bin/python -m src.mu_titration --selfcheck               # 学習なし Phase 0

既存 arm directory は、完走・途中失敗を問わず上書きしない。再利用や削除を暗黙に行う
``--force`` は意図的に提供しない。
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from . import ratchet_log
from .common import ROOT, build_runs, group_runs, load_config, resolve_outdir
from .train import setup_group


DEFAULT_CONFIG = "configs/mu_titration_0823.yaml"
CANONICAL_S2_STEPS = 100000
CANONICAL_ADDENDA = (
    "specs/spec_mu_titration_0823_addendum.md",
    "specs/spec_mu_titration_0823_addendum2.md",
)
PROVENANCE_SOURCE_FILES = (
    "src/mu_titration.py", "src/ratchet_log.py", "src/train.py",
    "src/common.py", "src/envs.py", "src/nets.py",
)
LEGACY_EXACT_KEYS = {
    "G", "flip_state", "E_delta", "mu_norm", "ratio_mu_cov", "cos_G_mu",
    "G_dot_mu", "eval_loss_exact", "cos_u_mu", "p_hat", "w_norm", "b", "v",
    "F_self", "F_rest", "F_gate",
}
NEW_EXACT_KEYS = {
    "M", "s", "b_plus_M", "cos_crit", "delta_b_field", "delta_wmu_field",
}


def canonical_s2_steps(cfg, cli_value=None):
    """full run の S2 を canonical 100,000 step に固定する。"""
    try:
        raw = cfg["mu_titration"]["s2_steps"]
    except (KeyError, TypeError) as exc:
        raise ValueError("full run には mu_titration.s2_steps が必須") from exc
    if type(raw) is not int or raw != CANONICAL_S2_STEPS:
        raise ValueError(f"mu_titration.s2_steps は {CANONICAL_S2_STEPS} 固定: {raw!r}")
    if cli_value is not None and cli_value != raw:
        raise ValueError(f"--s2-steps は canonical 値 {raw} と同じものだけ許可: "
                         f"{cli_value!r}")
    return raw


def canonical_device(cfg, cli_value=None):
    """事前登録どおり config / CLI とも CPU 以外を拒否する。"""
    configured = cfg.get("common", {}).get("device")
    if configured != "cpu":
        raise ValueError(f"mu titration の common.device は cpu 固定: {configured!r}")
    if cli_value is not None and cli_value != "cpu":
        raise ValueError(f"--device は cpu だけを許可: {cli_value!r}")
    return "cpu"


def canonical_addenda(cfg):
    """第1・第2追補を所定順序で指す canonical list だけを許可する。"""
    raw = cfg.get("addenda")
    expected = list(CANONICAL_ADDENDA)
    if type(raw) is not list or raw != expected:
        raise ValueError(f"config.addenda は順序つき固定値 {expected!r}: {raw!r}")
    return list(raw)


def _validate_provenance_addenda(value):
    """arm作成前にpath順序とSHA256形式を検査する。"""
    if not isinstance(value, list) or len(value) != len(CANONICAL_ADDENDA):
        raise RuntimeError("provenance.addenda が不完全")
    paths = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise RuntimeError(f"provenance.addenda[{index}] のschemaが不正")
        path, digest = item["path"], item["sha256"]
        if not isinstance(path, str) or not isinstance(digest, str):
            raise RuntimeError(f"provenance.addenda[{index}] の型が不正")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"provenance.addenda[{index}].sha256 が不正")
        paths.append(path)
    if paths != list(CANONICAL_ADDENDA):
        raise RuntimeError(f"provenance.addenda のpath/順序が不正: {paths!r}")
    return copy.deepcopy(value)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_output(args):
    """ROOT repository に対する read-only git query。"""
    proc = subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, check=False)
    if proc.returncode:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed ({proc.returncode}): {detail}")
    return proc.stdout


def _tracked_repo_path(path, label):
    """path が repository 内の tracked file であることを検査し相対pathを返す。"""
    absolute = os.path.realpath(path)
    root = os.path.realpath(ROOT)
    try:
        inside = os.path.commonpath([root, absolute]) == root
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(absolute):
        raise ValueError(f"{label} は repository 内の既存fileが必要: {path}")
    rel = os.path.relpath(absolute, root)
    _git_output(["ls-files", "--error-unmatch", "--", rel])
    return absolute, rel


def collect_run_provenance(config_path, cfg, allowed_output_root=None):
    """出力作成前に commit / clean source / preregistered material を固定する。

    並列armが既に生成した *untracked* fileだけは、同じ所定output root配下なら source
    dirty とみなさない。tracked fileの変更・他のuntracked file・staged変更は全て拒否する。
    これにより並列processは互いの成果物を許容しつつ、同一commit/fingerprintを記録できる。
    """
    config_abs, config_rel = _tracked_repo_path(os.path.abspath(config_path), "config")
    spec_value = cfg.get("spec")
    if not isinstance(spec_value, str) or not spec_value:
        raise ValueError("config.spec が必要")
    spec_path = spec_value if os.path.isabs(spec_value) else os.path.join(ROOT, spec_value)
    spec_abs, spec_rel = _tracked_repo_path(spec_path, "spec")
    addendum_paths = []
    for index, value in enumerate(canonical_addenda(cfg)):
        path = value if os.path.isabs(value) else os.path.join(ROOT, value)
        addendum_paths.append(_tracked_repo_path(path, f"addenda[{index}]"))

    source_paths = {}
    for rel in PROVENANCE_SOURCE_FILES:
        absolute, tracked_rel = _tracked_repo_path(os.path.join(ROOT, rel), "source")
        source_paths[tracked_rel] = absolute

    head = _git_output(["rev-parse", "HEAD"]).strip()
    if len(head) < 12 or any(c not in "0123456789abcdefABCDEF" for c in head):
        raise RuntimeError(f"不正な git HEAD: {head!r}")
    status_lines = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"]).splitlines()

    allowed_rel = None
    if allowed_output_root is not None:
        output_abs = os.path.realpath(allowed_output_root)
        root = os.path.realpath(ROOT)
        results_root = os.path.realpath(os.path.join(ROOT, "results"))
        try:
            # source treeをallowed outputに指定してdirty検査を迂回させない。
            if os.path.commonpath([results_root, output_abs]) == results_root:
                allowed_rel = os.path.relpath(output_abs, root).rstrip(os.sep)
        except ValueError:
            pass

    ignored_output, relevant = [], []
    for line in status_lines:
        path = line[3:].rstrip("/") if line.startswith("?? ") else None
        if (path is not None and allowed_rel is not None and
                (path == allowed_rel or path.startswith(allowed_rel + os.sep))):
            ignored_output.append(line)
        else:
            relevant.append(line)
    if relevant:
        preview = "\n".join(relevant[:20])
        raise RuntimeError("full run は clean committed worktree が必要。dirty status:\n" + preview)

    config_sha = _sha256_file(config_abs)
    spec_sha = _sha256_file(spec_abs)
    addenda = [dict(path=rel, sha256=_sha256_file(absolute))
               for absolute, rel in addendum_paths]
    source_sha = {rel: _sha256_file(path) for rel, path in source_paths.items()}
    material = dict(
        git_head=head, config_sha256=config_sha, spec_sha256=spec_sha,
        addenda=addenda,
        source_sha256=source_sha,
        center_alphas=[float(x) for x in cfg["mu_titration"]["center_alphas"]],
        s2_steps=canonical_s2_steps(cfg), device=canonical_device(cfg),
    )
    fingerprint = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(
        git_head=head, git_clean=True, git_source_clean=True,
        git_status_porcelain="",
        git_status_porcelain_raw="\n".join(status_lines),
        ignored_untracked_output_status=ignored_output,
        config_path=config_rel, config_sha256=config_sha,
        spec_path=spec_rel, spec_sha256=spec_sha,
        addenda=addenda,
        source_sha256=source_sha,
        sweep_commit=head, sweep_fingerprint=fingerprint,
        all_arm_alpha_tags=[f"alpha_{alpha_token(x)}" for x in material["center_alphas"]],
        same_commit_basis=("全armの arm_meta で sweep_commit と sweep_fingerprint の "
                           "双方が一致すること"),
        checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def alpha_token(alpha):
    """float を最短 round-trip 表現の directory token にする。

    Python ``repr`` の指数だけ正規化し、``1e-06`` を ``1e-6`` にする。float(token) で
    元値へ厳密に戻せるため安定・可逆で、grid 全体について衝突も別途拒否する。"""
    x = float(alpha)
    if not math.isfinite(x):
        raise ValueError(f"alpha は有限値であること: {alpha!r}")
    if x == 0.0:                         # -0.0 も同じ arm として一意化
        return "0"
    s = repr(x).lower()
    if "e" in s:
        mantissa, exponent = s.split("e")
        s = f"{mantissa}e{int(exponent)}"
    if float(s) != x:                    # repr の round-trip 契約を実行時にも確認
        raise AssertionError(f"alpha token が非可逆: {x!r} -> {s!r}")
    return s


def alpha_grid(cfg):
    """canonical ``mu_titration.center_alphas`` を検証して float list で返す。"""
    try:
        raw = cfg["mu_titration"]["center_alphas"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config に mu_titration.center_alphas が必要") from exc
    vals = [float(x) for x in raw]
    if not vals:
        raise ValueError("mu_titration.center_alphas は空にできない")
    if any(not math.isfinite(x) for x in vals):
        raise ValueError(f"center_alphas は有限値のみ: {vals}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"center_alphas に数値重複: {vals}")
    tags = [alpha_token(x) for x in vals]
    if len(set(tags)) != len(tags):
        raise ValueError(f"alpha tag が衝突: {dict(zip(vals, tags))}")
    return vals


def select_alpha(requested, grid):
    """CLI float が固定 grid の厳密な member なら canonical grid 値を返す。"""
    x = float(requested)
    for candidate in grid:
        if x == candidate:
            return candidate
    choices = ", ".join(alpha_token(a) for a in grid)
    raise ValueError(f"--alpha {requested!r} は固定 grid 外。選択肢: {choices}")


def arm_config(base_cfg, alpha):
    """1 arm 用 deep copy を作り、単一group・独立10 runという契約を検査する。"""
    cfg = copy.deepcopy(base_cfg)
    if cfg.get("condA", {}).get("encodings") != ["centered"]:
        raise ValueError("mu titration は condA.encodings: [centered] のみを許す")
    cfg["condA"]["center_alpha"] = float(alpha)
    cfg.setdefault("mu_titration", {})["active_alpha"] = float(alpha)
    runs = build_runs(cfg)
    groups = group_runs(runs)
    if len(groups) != 1 or len(runs) != 10:
        raise ValueError(f"各 alpha は単一 group・R=10 が必要: groups={len(groups)}, R={len(runs)}")
    seeds = [int(r["seed"]) for r in runs]
    if len(set(seeds)) != 10:
        raise ValueError(f"各 alpha は相異なる seed 10 本が必要: {seeds}")
    return cfg


def _s3_regression_selfcheck(base_records):
    """addendum2 の各必須conjunctと「zは診断のみ」を合成負例で固定する。"""
    def evaluate(records):
        return ratchet_log.check_s3(SimpleNamespace(s3=records))

    failures = {}
    records = copy.deepcopy(base_records)
    records[0]["p_exact"][0, 0] += 1e-6
    failures["exact_uniform_identity"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["p_reweighted"][0, 0] += 1e-6
    failures["empirical_reweighted_identity"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["n_bad_row_support_matches"] = 1
    records[0]["row_support_match_min"] = 0
    failures["unique_support_match"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["pattern_counts"][0] += 1
    failures["pattern_count_sum"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["N"] = ratchet_log.S3_EXPECTED_EVAL_N - 1
    failures["canonical_dimensions"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["p_reweighted"] = records[0]["p_reweighted"][:, :-1]
    failures["shape"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records.pop(2)
    failures["three_points"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    records[0]["p_empirical"][0, 0] = np.nan
    failures["finite"] = not evaluate(records)["s3_pass"]

    records = copy.deepcopy(base_records)
    degenerate = ((records[0]["p_uniform"] == 0.0) |
                  (records[0]["p_uniform"] == 1.0))
    indices = np.argwhere(degenerate)
    if indices.size:
        i, j = (int(x) for x in indices[0])
        records[0]["p_empirical"][i, j] += 1e-3
        records[0]["p_reweighted"][i, j] += 1e-3
        failures["degenerate_exact"] = not evaluate(records)["s3_pass"]
    else:
        failures["degenerate_exact"] = False

    # eval 2000行を同じsupport patternへ集中させる有効な合成頻度でzを極端化する。
    # direct経験率とsupport再重み付けは一致したままなので、旧family閾値によらずPASSする。
    records = copy.deepcopy(base_records)
    for item in records.values():
        match = np.zeros_like(item["support_match"])
        match[:, 0] = 1
        counts = np.zeros_like(item["pattern_counts"])
        counts[0] = item["N"]
        concentrated_rate = item["gate_support"][0].astype(np.float64)
        item["support_match"] = match
        item["pattern_counts"] = counts
        item["pattern_count_sum"] = item["N"]
        item["n_bad_row_support_matches"] = 0
        item["row_support_match_min"] = 1
        item["row_support_match_max"] = 1
        item["p_empirical"] = concentrated_rate.copy()
        item["p_reweighted"] = concentrated_rate.copy()
    z_only = evaluate(records)
    z_diagnostic_only = bool(
        z_only["s3_pass"] and z_only["s3_median_abs_z"] > 1.0 and
        z_only["s3_frac_gt3"] > 0.01
    )
    return dict(
        all_required_negative_cases_fail=bool(all(failures.values())),
        required_negative_cases=failures,
        z_diagnostic_only=z_diagnostic_only,
        z_diagnostic_example=dict(
            s3_pass=z_only["s3_pass"],
            median_abs_z=z_only["s3_median_abs_z"],
            frac_gt3=z_only["s3_frac_gt3"],
            binom_tail_p=z_only["s3_binom_tail_p"],
        ),
    )


def selfcheck(base_cfg, alpha, device):
    """学習をせず初期状態1点で新旧 exact_record 契約と恒等式を検査する。"""
    cfg = arm_config(base_cfg, alpha)
    runs = build_runs(cfg)
    gkey, gruns = next(iter(group_runs(runs).items()))
    st = setup_group(gkey, gruns, cfg, device)

    repro = ratchet_log.reproducibility_hash(st)
    repro_fingerprint = hashlib.sha256(
        json.dumps(repro, sort_keys=True).encode()).hexdigest()
    before = ratchet_log.state_hash(st)
    teacher_before = {k: v.clone() for k, v in st["teacher"].state_dict().items()}
    public = ratchet_log.exact_record(st)       # 従来の dict / float32 契約
    rec, field = ratchet_log.exact_record(st, as_f64=True, _with_sanity=True)
    # 同一初期状態を3つの仮想記録点として各回独立再列挙し、S3集約判定もPhase 0で通す。
    s3_records = {
        step: ratchet_log._s3_support_record(st, public["p_hat"])
        for step in (0, 1, 2)
    }
    s3 = ratchet_log.check_s3(SimpleNamespace(s3=s3_records))
    s3_regression = _s3_regression_selfcheck(s3_records)
    after = ratchet_log.state_hash(st)
    teacher_same = all(torch.equal(teacher_before[k], v)
                       for k, v in st["teacher"].state_dict().items())

    missing_legacy = sorted(LEGACY_EXACT_KEYS - set(public))
    missing_new = sorted(NEW_EXACT_KEYS - set(public))
    public_f32 = all(np.asarray(public[k]).dtype == np.float32 for k in public)
    closure_bM = float(np.max(np.abs(rec["b_plus_M"] - (rec["b"] + rec["M"]))))
    field_wmu = float(np.max(np.abs(
        rec["delta_wmu_field"] - rec["F_gate"] * rec["mu_norm"][:, None])))
    denom = np.maximum(rec["w_norm"] * rec["mu_norm"][:, None], 1e-300)
    crit_ref = -(rec["b"] + rec["M"]) / denom
    crit_err = float(np.max(np.abs(rec["cos_crit"] - crit_ref)))

    checks = {
        "public_return_is_dict": isinstance(public, dict),
        "legacy_keys_present": not missing_legacy,
        "new_keys_present": not missing_new,
        "public_values_float32": public_f32,
        "probe_read_only": before == after and teacher_same,
        "field_identity": (field["n_mismatch_beyond_tol"] == 0 and
                           field["n_mismatch_raw"] == 0 and
                           field["max_pre_identity_abs_err"] <= ratchet_log.FIELD_IDENTITY_ATOL and
                           field["max_M_support_abs_err"] <= ratchet_log.FIELD_IDENTITY_ATOL),
        "cos_crit_formula": (field["max_cos_crit_formula_abs_err"] <=
                             ratchet_log.FIELD_IDENTITY_ATOL),
        "delta_s_direct_closure": (field["max_delta_s_field_abs_err"] <=
                                   ratchet_log.FIELD_IDENTITY_ATOL),
        "p_hat_quantized": (field["max_p_hat_quantization_abs_err"] <=
                            ratchet_log.FIELD_IDENTITY_ATOL),
        "all_stats_finite": field["n_nonfinite_all_stats"] == 0,
        "b_plus_M_closure": closure_bM <= ratchet_log.FIELD_IDENTITY_ATOL,
        "delta_wmu_closure": field_wmu <= ratchet_log.FIELD_IDENTITY_ATOL,
        "cos_crit_closure": crit_err <= ratchet_log.FIELD_IDENTITY_ATOL,
        "s3_deterministic_support_reweighting": bool(s3["s3_pass"]),
        "s3_required_conjunct_regressions": bool(
            s3_regression["all_required_negative_cases_fail"]),
        "s3_z_statistics_are_diagnostic_only": bool(
            s3_regression["z_diagnostic_only"]),
    }
    result = dict(
        selfcheck_pass=bool(all(checks.values())), alpha=float(alpha),
        alpha_tag=alpha_token(alpha), device=str(device), R=len(gruns),
        step0_repro_fingerprint=repro_fingerprint,
        checks=checks, missing_legacy_keys=missing_legacy, missing_new_keys=missing_new,
        b_plus_M_max_abs_err=closure_bM,
        delta_wmu_vs_Fgate_mu_norm_max_abs_err=field_wmu,
        cos_crit_max_abs_err=crit_err, field_identity=field,
        s3_deterministic=s3,
        s3_regression=s3_regression,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["selfcheck_pass"]:
        raise RuntimeError("mu titration selfcheck FAIL")
    return result


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def run_arm(base_cfg, config_path, alpha, device, base_outdir, s2_steps=None,
            provenance=None):
    """directory を排他的に確保して1 armを実行する。既存 path は必ず abort。"""
    device = canonical_device(base_cfg, device)
    s2_steps = canonical_s2_steps(base_cfg, s2_steps)
    canonical_addenda(base_cfg)
    if provenance is None:
        provenance = collect_run_provenance(
            config_path, base_cfg, allowed_output_root=base_outdir)
    required_provenance = {"git_head", "git_clean", "git_status_porcelain",
                           "config_sha256", "spec_sha256", "addenda", "source_sha256",
                           "sweep_commit", "sweep_fingerprint"}
    missing = sorted(required_provenance - set(provenance))
    if missing or not provenance.get("git_clean"):
        raise RuntimeError(f"不完全/dirty provenance: missing={missing}")
    provenance_addenda = _validate_provenance_addenda(provenance["addenda"])

    cfg = arm_config(base_cfg, alpha)
    tag = f"alpha_{alpha_token(alpha)}"
    arm_dir = os.path.join(base_outdir, tag)
    os.mkdir(arm_dir)                    # 原子的な予約。並列同一 arm は一方だけ成功する。

    started = time.time()
    manifest_path = os.path.join(arm_dir, "arm_meta.json")
    manifest = dict(
        status="running", alpha=float(alpha), center_alpha=float(alpha), alpha_tag=tag,
        config_source=os.path.abspath(config_path), result_dir=os.path.abspath(arm_dir),
        R=10, seeds=[int(x) for x in cfg["common"]["seeds"]],
        total_steps=int(cfg["common"]["total_steps"]), s2_steps=int(s2_steps),
        device=str(device), pid=os.getpid(),
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_head=provenance["git_head"], git_clean=provenance["git_clean"],
        git_status_porcelain=provenance["git_status_porcelain"],
        git_status_porcelain_raw=provenance.get("git_status_porcelain_raw", ""),
        git_source_clean=provenance.get("git_source_clean", provenance["git_clean"]),
        config_sha256=provenance["config_sha256"],
        spec_sha256=provenance["spec_sha256"],
        addenda=provenance_addenda,
        source_sha256=provenance["source_sha256"],
        sweep_commit=provenance["sweep_commit"],
        sweep_fingerprint=provenance["sweep_fingerprint"],
        all_arm_alpha_tags=provenance.get("all_arm_alpha_tags", []),
        all_arms_same_commit_required=True,
        same_commit_basis=provenance.get("same_commit_basis"),
        provenance_file="provenance.json",
    )
    _write_json(manifest_path, manifest)
    _write_json(os.path.join(arm_dir, "provenance.json"), provenance)
    with open(os.path.join(arm_dir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

    print(f"\n=== {tag}: alpha={alpha!r} -> {arm_dir} ===", flush=True)
    try:
        meta = ratchet_log.run(cfg, device, arm_dir, s2_steps=s2_steps)
    except BaseException as exc:
        manifest.update(status="failed", elapsed_sec=round(time.time() - started, 3),
                        error_type=type(exc).__name__, error=str(exc))
        _write_json(manifest_path, manifest)
        raise

    sanity = meta.get("sanity", {})
    sanity_pass = bool(sanity.get("all_required_pass", False))
    if not sanity_pass:
        failed_checks = sorted(
            name for name, result in sanity.items()
            if isinstance(result, dict) and any(
                key.endswith("_pass") and value is False for key, value in result.items()))
        manifest.update(
            status="failed_sanity", elapsed_sec=round(time.time() - started, 3),
            ratchet_meta="meta.json", sanity_pass=False,
            failed_sanity_checks=failed_checks,
            error_type="RuntimeError",
            error="required sanity が FAIL（結果は除外・上書きせず全体停止）",
        )
        _write_json(manifest_path, manifest)
        raise RuntimeError(manifest["error"])

    manifest.update(status="complete", elapsed_sec=round(time.time() - started, 3),
                    ratchet_meta="meta.json",
                    sanity_pass=True)
    _write_json(manifest_path, manifest)
    return arm_dir, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--alpha", type=float, default=None,
                    help="固定 center_alphas のうち1 armだけ実行（省略時は全arm逐次）")
    ap.add_argument("--s2-steps", type=int, default=None,
                    help="canonical値100000の明示確認用（省略時もconfigの100000を使用）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=None,
                    help="base result dir（省略時 results/<config stem>）")
    ap.add_argument("--selfcheck", action="store_true",
                    help="学習・result書込みなしで exact_record Phase 0 を実施")
    args = ap.parse_args()

    cfg = load_config(args.config)
    grid = alpha_grid(cfg)
    try:
        requested = None if args.alpha is None else select_alpha(args.alpha, grid)
    except ValueError as exc:
        ap.error(str(exc))
    selected = grid if requested is None else [requested]
    try:
        device = canonical_device(cfg, args.device)
    except ValueError as exc:
        ap.error(str(exc))

    if args.selfcheck:
        # --alpha 指定時はその値、未指定時は全 grid の設定生成・reset契約を検査する。
        checks = selected if requested is not None else grid
        results = []
        for alpha in checks:
            results.append(selfcheck(cfg, alpha, device))
        if requested is None:
            fingerprints = {r["step0_repro_fingerprint"] for r in results}
            if len(fingerprints) != 1:
                raise RuntimeError("alpha 間で step0 / generator hash が不一致")
            print(f"CROSS-ALPHA RESET PASS ({len(results)} arms; "
                  f"fingerprint={next(iter(fingerprints))})", flush=True)
        print("SELF-CHECK PASS", flush=True)
        return

    try:
        s2_steps = canonical_s2_steps(cfg, args.s2_steps)
    except ValueError as exc:
        ap.error(str(exc))
    base_outdir = args.outdir or resolve_outdir(args.config)
    result_subdir = str(cfg["mu_titration"].get("result_subdir", "") or "")
    arm_root = os.path.join(base_outdir, result_subdir) if result_subdir else base_outdir
    # preregistration check は mkdir より先。dirty source のまま空のresultsを残さない。
    try:
        provenance = collect_run_provenance(
            args.config, cfg, allowed_output_root=arm_root)
    except (ValueError, RuntimeError) as exc:
        ap.error(str(exc))
    os.makedirs(arm_root, exist_ok=True)
    arm_dirs = [os.path.join(arm_root, f"alpha_{alpha_token(a)}") for a in selected]
    existing = [p for p in arm_dirs if os.path.lexists(p)]
    if existing:
        ap.error("既存 arm は上書きしません: " + ", ".join(existing))

    done = []
    for alpha in selected:
        try:
            arm_dir, meta = run_arm(cfg, args.config, alpha, device, arm_root,
                                    s2_steps=s2_steps, provenance=provenance)
        except FileExistsError:
            # preflight 後に別プロセスが同じ arm を確保した race も安全側に倒す。
            ap.error(f"arm が別プロセスにより既に確保されています: alpha_{alpha_token(alpha)}")
        done.append(dict(alpha=float(alpha), path=arm_dir,
                         sanity_pass=bool(meta["sanity"]["all_required_pass"])))
    print("\nALL DONE")
    print(json.dumps(done, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
