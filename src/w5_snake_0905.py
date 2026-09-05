# -*- coding: utf-8 -*-
"""w5_snake_0905 — 学習器幅 5・教師幅 100（容量不足）で Snake を回す（**未登録・事後**）。

    OMP_NUM_THREADS=1 python3 -m src.w5_snake_0905 --arm SN5_a1

Issa の指示（2026-09-05）。狙いは「**残差が残る設定で Snake の ‖w‖ が育つか**」。
深さ 1・幅 100 では課題を解き切って `EXACT_FIT` になり LoP を測れなかった。
幅 5・教師幅 100 なら容量不足で残差が必ず残るので、fit 依存でない検査になる。

宿主は `width5_gate_b_0901`（= `width5_gate_0901` の `_run_arm` をそのまま使う）。
**config も generator_offset も seeds も既存腕と同一**にするので、対照 R5 / LR5 / E5 /
LIN5（`results/width5_gate_b_0901/logs/`）と**同じ lr・同じ入力列**で並べられる。
`LIN5` は leaky(a=1.0) = 恒等なので線形対照になっている。

`width5_gate_0901.REGISTERED_ARMS` は 8 腕を逐語で持ち `setup_arm_width` がそこから
活性化を引くので、本モジュールは `LR5` の腕ブロックを複製して名前だけ変え、
setup のあとに `set_activation("snake", alpha)` で差し替える（`elu_swamp_0830` と同じフック。
活性化は RNG を消費しないので init と入力列は `LR5` と bit 一致する）。**verdict には入れない。**
"""
from __future__ import annotations

import argparse, copy, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np, torch, yaml

from . import width5_gate_0901 as base
from .common import ROOT, load_config
from .mlp2_phase0 import _sha_file, identity_sanity_pass, require_omp
from .mlp2_phase1 import NUMERIC_DIVERGENCE, NumericDivergenceError, _env_hashes

EXPERIMENT = "w5_snake_0905"
CONFIG = Path(ROOT) / "configs" / "width5_gate_b_0901.yaml"
CLONE_FROM = "LR5"                       # 幅 5・中心化なし・用量オラクルなしの腕ブロック
ARMS = {"SN5_a1": 1.0, "SN5_a3": 3.0, "SN5_a0p5": 0.5}
# 2026-09-05 追加: ‖J‖² 規則が処方する lr で活性化を比べるための一般化。
# 既存 3 腕の挙動は変えない（--lr 未指定なら config の 0.01 のまま）。
ACTS = {"SN5_a1": ("snake", 1.0), "SN5_a3": ("snake", 3.0), "SN5_a0p5": ("snake", 0.5),
        "LR5x": ("leaky_relu", 0.1), "R5x": ("relu", 1.0), "LIN5x": ("leaky_relu", 1.0),
        # snake_flip_0906（spec §2.2）: 周期を外した 1 葉と、零点を外した反転
        "SN1_a1": ("snake1", 1.0), "SNA05_a1": ("snake_amp0p5", 1.0),
        "SNA025_a1": ("snake_amp0p25", 1.0)}
UNREG = ("この走は事前登録されていない（Issa の指示・2026-09-05）。**verdict には入れない。**"
         " 変えた軸は活性化 1 本だけで、config・generator_offset・seeds・lr は "
         "`width5_gate_b_0901` と同一。対照 R5 / LR5 / E5 / LIN5 と同じハーネスに乗る。")

def _git(*a):
    try: return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    except Exception: return "unknown"

def _cfg_for(name: str, lr: float | None = None) -> dict:
    cfg = load_config(str(CONFIG))
    if lr is not None:
        cfg["common"]["lr_main"] = float(lr)
    src = next(a for a in cfg["arms"] if a["name"] == CLONE_FROM)
    arm = copy.deepcopy(src); arm["name"] = name
    cfg["arms"] = [arm]                       # 走らせるのは 1 腕だけ
    return cfg

def _setup(cfg: dict, name: str, alpha: float, device: str, act: str = "snake") -> dict:
    """base.setup_arm_width の写し。REGISTERED_ARMS を引かず snake を直に差す。"""
    from .gate_dose import setup_arm_gate
    arm_cfg = next(a for a in cfg["arms"] if a["name"] == name)
    probe = copy.deepcopy(arm_cfg); probe["name"] = CLONE_FROM   # 宿主は名前で引く
    st = setup_arm_gate(cfg, probe, device)
    st["net"].set_activation(act, float(alpha), "alpha_exp")
    st["activation"], st["act_alpha"] = act, float(alpha)
    st["activation_label"] = act
    st["arm"] = name
    st["generator_offset"] = int(cfg["common"]["generator_offset"])
    return st

def _run(cfg: dict, name: str, alpha: float, outdir: Path, seeds, total: int, act: str = "snake") -> dict:
    """base._run_arm の写し。setup だけが本モジュールのもの。"""
    c = copy.deepcopy(cfg); c["common"]["seeds"] = [int(v) for v in seeds]
    st = _setup(c, name, alpha, "cpu", act)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total: probes.append(total)
    _, s0 = base.exact_layer_record_width(
        st, float(c["sanity"]["sigma_degenerate_tol"]),
        float(c["width5_gate"]["mobility_floor_tolerance"]))
    if not identity_sanity_pass(s0, float(c["sanity"]["identity_tol"])):
        raise RuntimeError(f"{name} initial exact-support identity failed")
    rec = base.WidthGateRecorder(probes, st, c)
    ckpt = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{name}] width={st['hidden'][0]} act={act} alpha={alpha:g} "
          f"lr={c['common']['lr_main']} seeds={len(seeds)} steps={total:,}", flush=True)
    from .gate_dose import train_arm_gate
    t0 = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, ckpt)
    except NumericDivergenceError as exc:
        ev = dict(exc.event); ev.update(activation=act, act_alpha=alpha,
                                        width=st["hidden"][0], elapsed_sec=time.time()-t0)
        (outdir/"arm_status").mkdir(parents=True, exist_ok=True)
        (outdir/"arm_status"/f"{name}.json").write_text(
            json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{name}] {NUMERIC_DIVERGENCE} at step {ev.get('detected_step')}", flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=time.time()-t0, divergence=ev)
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise RuntimeError(f"{name} exact-support sanity failed: {sanity}")
    base.write_arm_logs(outdir, name, st, rec)
    print(f"[{name}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity, final_env=_env_hashes(st))

def jnorm_sq(name: str, alpha_override: float | None = None) -> dict:
    """‖J‖² 則の材料 [snake_flip_0906 §2.2]。J = ∂ŷ/∂θ を init・32 パターン・全 seed で測る。

    ∂ŷ/∂v_i = φ(z_i)、∂ŷ/∂w_i = v_i φ'(z_i) x、∂ŷ/∂b_i = v_i φ'(z_i)、∂ŷ/∂c = 1。
    パターンと seed で平均した ‖J‖² を返す（float64）。lr_arm = lr_ref · ‖J_ref‖² / ‖J_arm‖²。
    """
    from .ratchet_log import full_support_ro
    act, alpha = ACTS[name]
    if alpha_override is not None:
        alpha = float(alpha_override)
    cfg = _cfg_for(name)
    st = _setup(cfg, name, alpha, "cpu", act)
    net = st["net"]
    with torch.no_grad():
        x = full_support_ro(st["env"]).double()                 # (P, R, d)
        W, b, v = net.Ws[0].double(), net.bs[0].double(), net.v.double()
        z = torch.einsum("rhd,prd->prh", W, x) + b                # (P, R, h)
        phi, dphi = net.act_fn(z), net.act_grad(z, net.act_fn(z))
        x2 = (x ** 2).sum(-1)                                     # (P, R)
        jv = (phi ** 2).sum(-1)                                   # Σ_i φ²
        jwb = ((v[None] * dphi) ** 2 * (x2[..., None] + 1.0)).sum(-1)   # Σ_i v² φ'² (‖x‖²+1)
        total = jv + jwb + 1.0
    return dict(arm=name, activation=act, alpha=float(alpha),
                J2=float(total.mean()), J2_v=float(jv.mean()), J2_wb=float(jwb.mean()))


def lr_rule(name: str, ref: str = "LR5x", lr_ref: float = 0.01,
            alpha_override: float | None = None) -> dict:
    a, r = jnorm_sq(name, alpha_override), jnorm_sq(ref)
    return dict(arm=name, ref=ref, lr_ref=lr_ref, J2_arm=a["J2"], J2_ref=r["J2"],
                lr=lr_ref * r["J2"] / a["J2"])


def run(name: str, steps: int | None = None, seeds=None, lr: float | None = None,
        tag: str | None = None, alpha_override: float | None = None) -> dict:
    act, alpha = ACTS[name]
    if alpha_override is not None:
        alpha = float(alpha_override)          # 2026-09-05: alpha を広く振るため
    cfg = _cfg_for(name, lr); require_omp(cfg)
    total = int(steps or cfg["common"]["total_steps"])
    sd = [int(v) for v in (seeds if seeds is not None else cfg["common"]["seeds"])]
    outdir = Path(ROOT)/"results"/"_diag_w5_snake_0905"/(tag or name)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/"config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    t0 = time.time()
    got = _run(cfg, name, alpha, outdir, sd, total, act)
    prov = dict(experiment=EXPERIMENT, unregistered=True, unregistered_note=UNREG,
                arm_name=name, tag=tag or name, activation=act, act_alpha=alpha,
                width=5, teacher_width=int(cfg["condA"]["target_hidden"]),
                lr=float(cfg["common"]["lr_main"]),
                generator_offset=int(cfg["common"]["generator_offset"]),
                seeds=sd, steps=total, host="width5_gate_b_0901",
                controls="results/width5_gate_b_0901/logs (R5 LR5 E5 LIN5・同一 config/offset/seeds)",
                status=got.get("status"), divergence=got.get("divergence"),
                elapsed_sec=time.time()-t0, command=sys.argv, python=sys.version,
                platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                git_hash=_git("rev-parse","HEAD"), git_dirty=_git("status","--short"),
                output_sha256={f"logs/{p.name}": _sha_file(p) for p in sorted((outdir/"logs").glob("*.npz"))})
    (outdir/"provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[{EXPERIMENT}] {name} -> {got.get('status')} in {prov['elapsed_sec']:.0f}s", flush=True)
    return prov

def main():
    ap = argparse.ArgumentParser(description=EXPERIMENT)
    ap.add_argument("--arm", required=True, choices=sorted(ACTS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--lr-rule", action="store_true",
                    help="‖J‖² 則が処方する lr を印字して終わる（走らせない）")
    a = ap.parse_args()
    if a.lr_rule:
        print(json.dumps(lr_rule(a.arm, alpha_override=a.alpha), indent=1)); return
    run(a.arm, a.steps, a.seeds, a.lr, a.tag, a.alpha)

if __name__ == "__main__":
    main()
