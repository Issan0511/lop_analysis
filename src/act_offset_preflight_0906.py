# -*- coding: utf-8 -*-
"""act_offset_0906 の短縮走行と S-limit（spec `specs/spec_act_offset_0906.md` §5）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.act_offset_preflight_0906 \\
        [--steps 30000] [--outdir results/_preflight_act_offset_0906]

1. **短縮走行**: 登録 7 腕を 30k step（10 seed・1 プロセス）回し、全列が有限・
   `lr_used` = 0.01・新列（zmin / w_free / モーメント 4 列 / payload）ありを確認する。
2. **S-limit（30k）**: 同じ config に `activation: leaky_relu` の腕 `LRleaky_1216` を
   足した一時 config で 30k 回し、`LRoff0_1216` と `state_hash_final` ＋ 全数値列が
   **バイト一致**すること。**変異対照**: `LRoffp0p5_1216` は同じ比較が**落ちる**こと
   （空虚な S 検査の再発防止・[[proj-004-edge-law-0905]]）。

結果は `<outdir>/preflight.json`。unittest（`src.test_act_offset_0906`）は本モジュールの
関数を呼ぶので、検査の中身はここ 1 か所にある。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import edge_law_0905 as E
from .common import ROOT, load_config

CONFIG = Path(ROOT) / "configs" / "act_offset_0906.yaml"
EXPERIMENT = "act_offset_0906"
S_LIMIT_REF_ARM = "LRleaky_1216"           # 一時 config にだけ足す参照腕（本走に無い）
S_LIMIT_ARM = "LRoff0_1216"
S_LIMIT_MUTANT = "LRoffp0p5_1216"
SHORT_STEPS = 30_000
LR_REGISTERED = 0.01
# 追補 1 §1 で**登録済み**の発散: lr 0.01 の c=±2 は λ ≈ 2hc² = 800 に対し lr·λ = 8 で
# step 1,000 に落ちる。ここは「落ちること」まで込みで登録なので、
#   * この 2 腕が落ちる → 前検査は PASS（ラダー A ではこの 2 腕を NOT_RUN として扱う）
#   * この 2 腕が**落ちなかった** → 追補 1 の前提が崩れているので FAIL
# の両方向で検査する（片側だけだと空虚な whitelist になる）。
EXPECTED_DIVERGENT = ("LRoffm2_1216", "LRoffp2_1216")
# 比較から外す列: 腕名を含む文字列と、1M 未満では空の state_hash_1m。
SKIP_COLUMNS = ("arm", "run_id", "activation", "state_hash_1m")
# 発散の検査から外す既知の NaN 列: フック無しの腕は init_hook_arg が NaN（payload の登録どおり）、
# layer1_eff_rank_per_alive は alive=0 の記録で 0/0（派生列・net の非有限とは無関係。ELU c=+1 の
# 30k 前検査で seed 0・9 に出た）。net 本体の発散は runner が NUMERIC_DIVERGENCE として別に拾う。
KNOWN_NAN_COLUMNS = ("init_hook_arg", "layer1_eff_rank_per_alive")
REQUIRED_COLUMNS = ("layer1_zmin", "layer1_w_free", "layer1_w_free_step",
                    "layer1_m_phi2", "layer1_m_dphi2", "layer1_m_phidphi",
                    "layer1_m_dphiddphi", "layer1_moment_step",
                    "init_hook", "init_hook_arg", "lr_used", "freeze_v", "batch_mode",
                    "layer1_zbar", "layer1_zmax", "layer1_mob", "layer1_v_unit",
                    "layer1_denom", "layer1_w_norm", "state_hash_final")


def augmented_config(tmpdir: Path, config: Path = CONFIG) -> Path:
    """登録 config に S-limit 参照腕（`leaky_relu`・同じ dial・フックなし）を 1 本足した写し。"""
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    base = next(a for a in raw["arms"] if a["name"] == S_LIMIT_ARM)
    ref = copy.deepcopy(base)
    ref["name"] = S_LIMIT_REF_ARM
    ref["activation"] = "leaky_relu"
    raw["arms"].append(ref)
    raw["activation"].setdefault("leaky_relu", {"name": "leaky_relu"})
    raw["output"] = dict(dir=f"results/_preflight_{EXPERIMENT}/slimit",
                         logs_tail_dir=f"results/_preflight_{EXPERIMENT}/slimit/logs_tail")
    out = Path(tmpdir) / "act_offset_0906_slimit.yaml"
    out.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


def run_short(arm: str, outdir: Path, steps: int = SHORT_STEPS,
              config: Path = CONFIG) -> dict:
    """腕を `steps` だけ回す（10 seed・登録 lr）。runner のモジュール状態は元に戻す。"""
    saved = (E.CONFIG, E._TABLE)
    try:
        E.CONFIG = Path(config).resolve()
        E._TABLE = E.arm_table(load_config(str(E.CONFIG)))
        cfg = E.build_cfg(E.CONFIG)
        t0 = time.time()
        res = E.run_single_arm(arm, steps=int(steps), outdir=Path(outdir), seeds=None,
                               cfg=cfg)
        return dict(arm=arm, steps=int(steps), wall_sec=time.time() - t0,
                    status=str(res.get("status", "")))
    finally:
        E.CONFIG, E._TABLE = saved


def _load(outdir: Path, arm: str, seed: int) -> dict:
    with np.load(Path(outdir) / "logs" / f"{arm}_seed{seed}.npz",
                 allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def compare_logs(dir_a: Path, arm_a: str, dir_b: Path, arm_b: str,
                 seeds=range(10)) -> dict:
    """2 腕の logs を seed ごとに突き合わせる: state_hash_final の文字列一致 ＋ 共通列のバイト一致。"""
    rows, hash_ok, n_mismatch = [], True, 0
    for seed in seeds:
        a, b = _load(dir_a, arm_a, seed), _load(dir_b, arm_b, seed)
        keys = sorted((set(a) & set(b)) - set(SKIP_COLUMNS))
        bad = []
        for k in keys:
            xa, xb = np.ascontiguousarray(a[k]), np.ascontiguousarray(b[k])
            if xa.dtype != xb.dtype or xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                bad.append(k)
        h = str(a["state_hash_final"]) == str(b["state_hash_final"])
        hash_ok &= h
        n_mismatch += len(bad)
        rows.append(dict(seed=int(seed), state_hash_equal=bool(h),
                         n_columns=len(keys), mismatched=bad[:12], n_mismatched=len(bad)))
    return dict(pass_=bool(hash_ok and n_mismatch == 0), arm_a=arm_a, arm_b=arm_b,
                state_hash_all_equal=bool(hash_ok), n_mismatched_total=int(n_mismatch),
                rows=rows)


def expected_lr(arm: str, config: Path = CONFIG, default: float = LR_REGISTERED) -> float:
    """腕の `lr` フックの値（無ければ登録 lr）。追補 1 の 3 腕は 0.00125（S-lr）。"""
    row = E.arm_table(load_config(str(config)))[str(arm)]
    hook = row["hook"]
    return float(hook["value"]) if hook and hook.get("type") == "lr" else float(default)


def divergence_detail(outdir: Path, arm: str) -> dict:
    """runner の `arm_status/<arm>_done.json` から発散の要点（検出 step・seed）を抜く。"""
    f = Path(outdir) / "arm_status" / f"{arm}_done.json"
    if not f.exists():
        return dict(status="unknown")
    d = json.loads(f.read_text(encoding="utf-8"))
    dv = d.get("divergence", {}) or {}
    return dict(status=d.get("status"), detected_step=dv.get("detected_step"),
                bad_seeds=dv.get("bad_seeds"),
                nonfinite=sorted({t for v in (dv.get("nonfinite_tensors") or {}).values()
                                  for t in v}))


def check_run(outdir: Path, arm: str, steps: int = SHORT_STEPS,
              lr: float = LR_REGISTERED, seeds=range(10)) -> dict:
    """短縮走行の logs: 全数値列が有限・lr_used が登録値・新列が揃っている・最終 step が合う。"""
    rows, ok = [], True
    for seed in seeds:
        z = _load(outdir, arm, seed)
        nonfinite = []
        for k, v in z.items():
            if not (isinstance(v, np.ndarray) and v.dtype.kind == "f"):
                continue
            if k in KNOWN_NAN_COLUMNS:
                continue
            if k == "layer1_dzbar":
                v = v[1:]                      # 先頭は前記録が無く NaN（登録どおり・S-null と同じ扱い）
            if not np.all(np.isfinite(v)):
                nonfinite.append(k)
        missing = [k for k in REQUIRED_COLUMNS if k not in z]
        lr_used = float(z["lr_used"]) if "lr_used" in z else float("nan")
        last = int(z["step"][-1]) if "step" in z else -1
        good = (not nonfinite and not missing and lr_used == float(lr)
                and last == int(steps))
        ok &= good
        rows.append(dict(seed=int(seed), nonfinite=nonfinite, missing=missing,
                         lr_used=lr_used, last_step=last, ok=bool(good)))
    return dict(pass_=bool(ok), arm=arm, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=SHORT_STEPS)
    ap.add_argument("--outdir", default=str(Path(ROOT) / f"results/_preflight_{EXPERIMENT}"))
    a = ap.parse_args()
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = dict(experiment=EXPERIMENT, steps=int(a.steps), config=str(CONFIG),
                        git_head=E._git_head(), runs=[], checks=[])

    # 1. 短縮走行（登録 7 腕）。発散した腕は runner が `_done.json` に
    #    status=NUMERIC_DIVERGENCE を書いて logs を残さないので、それをそのまま記録する
    #    （lr を下げる等の救済はしない — 登録外の介入になる・edge_law HANDOFF §4）。
    table = E.arm_table(load_config(str(CONFIG)))
    diverged: list[str] = []
    for arm in table:
        r = run_short(arm, out / "arms", a.steps, CONFIG)
        report["runs"].append(r)
        if r["status"] == "NUMERIC_DIVERGENCE":
            diverged.append(arm)
            report["checks"].append(dict(check="short_run", pass_=False, arm=arm, diverged=True,
                                         detail=divergence_detail(out / "arms", arm)))
            print(f"[short {arm}] DIVERGED {report['checks'][-1]['detail']}", flush=True)
            continue
        report["checks"].append(dict(check="short_run", diverged=False,
                                     **check_run(out / "arms", arm, a.steps,
                                                 expected_lr(arm, CONFIG))))
        print(f"[short {arm}] {'PASS' if report['checks'][-1]['pass_'] else 'FAIL'}", flush=True)
    report["diverged_arms"] = diverged

    # 2. S-limit（30k・同じマシン・同じ config に leaky_relu の腕を足して比べる）。
    #    変異対照は c≠0 の leaky 腕のうち発散しなかった最初の 1 本。
    slimit_cfg = augmented_config(out, CONFIG)
    report["runs"].append(run_short(S_LIMIT_REF_ARM, out / "slimit", a.steps, slimit_cfg))
    ref_dir, arm_dir = out / "slimit", out / "arms"
    if S_LIMIT_ARM in diverged:
        same = dict(pass_=False, note=f"{S_LIMIT_ARM} diverged")
    else:
        same = compare_logs(arm_dir, S_LIMIT_ARM, ref_dir, S_LIMIT_REF_ARM)
    mutants = [m for m in (S_LIMIT_MUTANT, "LRoffm0p5_1216", "LRoffp2_1216", "LRoffm2_1216")
               if m not in diverged]
    if mutants:
        mutant = compare_logs(arm_dir, mutants[0], ref_dir, S_LIMIT_REF_ARM)
        mutant_fails = not mutant["pass_"]
    else:
        mutant, mutant_fails = dict(pass_=None, note="every c≠0 leaky arm diverged"), False
    s_limit = dict(check="S-limit-30k", pass_=bool(same["pass_"] and mutant_fails),
                   same=same, mutant_control=mutant,
                   note="PASS = leaky_off_0 が leaky_relu と state_hash＋全列バイト一致、"
                        f"かつ {mutants[0] if mutants else '(none)'} は同じ比較が落ちる（変異対照）")
    report["checks"].append(s_limit)
    print(f"[S-limit-30k] {'PASS' if s_limit['pass_'] else 'FAIL'} "
          f"(same={same['pass_']}, mutant_fails={mutant_fails})", flush=True)

    # 合否: 「登録済みの発散 2 腕がちょうど落ちた」ことを要求し、それ以外の失敗は許さない。
    report["all_checks_pass"] = bool(all(c["pass_"] for c in report["checks"]))
    unexpected_fail = [c for c in report["checks"]
                       if not c["pass_"] and not (c.get("check") == "short_run"
                                                  and c.get("diverged")
                                                  and c.get("arm") in EXPECTED_DIVERGENT)]
    missing_divergence = [a for a in EXPECTED_DIVERGENT if a not in diverged]
    report["expected_divergent"] = list(EXPECTED_DIVERGENT)
    report["unexpected_failures"] = [c.get("arm", c.get("check")) for c in unexpected_fail]
    report["missing_expected_divergence"] = missing_divergence
    report["pass_"] = bool(not unexpected_fail and not missing_divergence)
    (out / "preflight.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[preflight] {'PASS' if report['pass_'] else 'FAIL'} "
          f"(登録済みの発散 {list(EXPECTED_DIVERGENT)} を除く失敗 {report['unexpected_failures']}"
          f" / 落ちるはずが落ちなかった腕 {missing_divergence}) -> {out / 'preflight.json'}")
    raise SystemExit(0 if report["pass_"] else 1)


if __name__ == "__main__":
    main()
