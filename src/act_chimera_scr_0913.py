"""act_chimera_scr_0913 — SCR condA（箱 A）の runner（spec_act_chimera_0913 §2.4・§3.4・§6・§8）。

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.act_chimera_scr_0913 --arm chLR_1216
    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.act_chimera_scr_0913 --arm chLR_1216 --steps 30000 \\
        --outdir results/_smoke_act_chimera_0913/scr                      # スモーク（§6）
    ... --arm chLR_1216 --steps 30000 --generator-offset 1 --outdir results/_smoke_act_chimera_0913/scr_offset1
                                                                          # G1-SCR の変異対照（検査走だけ）
    ... --g1 --outdir <outdir> [--smoke]                                   # G1-SCR: chLR/chE を edge_law の null 腕と s_null
    ... --rss-probe --outdir results/_smoke_act_chimera_0913/scr           # §2.4 の自前の rss_probe
    ... --readout chSMAXH_1216 --outdir <outdir>                           # readout_{ARM}.npz（§3.4・§8）
    ... --provenance --outdir <outdir>                                     # 腕ごとの provenance を 1 つに束ねる
    ... --s14                                                              # 写しの検査（S14）
    ... --arm chLR_lr0p005_1216 --outdir results/act_chimera_0913/scr_lr0p005   # 退避枝（§2.4 の条件でのみ）

宿主は ``edge_law_0905``（1 層・幅 100・オラクル用量 12.16 固定・10 seed・lr 0.01）。**既存モジュールは
1 行も変えない**（§2.2「実装経路」・§2.4 経路）。

実装経路（§2.4）
1. ``ChimeraMLPL``（``src/act_chimera_0913.py``）が VecMLPL にキメラ 6 名と分離腕 4 名を足す。
2. ``setup_arm_chimera`` = ``gate_dial_0902.setup_arm_dial`` の写し ＋ ``set_activation`` の直前に
   ``st["net"].__class__ = ChimeraMLPL`` の 1 行（乱数も状態も触らない・S13）。
3. ``edge_law_0905.main --config`` は edge_law 自身の ``run_single_arm`` → ``_run_arm_edge`` を呼ぶので
   写したループが呼ばれない。本モジュールが自前の ``run_single_arm`` / ``main`` を持ち、起動時に
   ``edge_law_0905.CONFIG = configs/act_chimera_scr_0913.yaml``、``edge_law_0905._TABLE = None`` を設定する
   （``table()`` / ``_hook_of()`` / ``registered()`` がこの大域を読む・``bind_config``）。
4. ``_run_arm_chimera`` = ``edge_law_0905._run_arm_edge`` の写し ＋ 本体の先頭に
   ``setup_arm_dial = setup_arm_chimera`` の 1 行（edge_law の登録挿入方式・``_copy_opcodes``・S14）。
5. 親 2 行 ``chLR_1216`` / ``chE_1216`` は ``LRnull_1216`` / ``Enull_1216`` と family・activation・dial・
   checkpoints まで逐語（config）。G1-SCR は edge_law の ``s_null`` に ``ref_arm`` / ``ref_dir`` を明示で渡し、
   比べたキーの一覧を実装前に参照 npz から固定した ``S_NULL_EXPECTED_KEYS`` と照合し、
   ``n_records`` と ``state_hash_1m`` への到達を別に assert する。
6. 追加の読み出し（帯分率・8 関数の Γ/Φ/Ψ・出力層の λ・relax_rate・安定余裕）は、タスク終端の
   ``w_free`` と z̄ から 32 パターンの支持を再構成して ``build_readout`` が logs から作る（ランナーへの挿入は
   要らない）。全 Hessian の λ は ckpt（0・1M・5M）から冪乗法で取る（S19）。

使ってはならないもの: ``EG.make_act`` / ``EG.hdefect`` / ``EG.measure``（S8）、``GS.check_dphi``、
``edge_law_0905.rss_probe``（子を ``--config`` 無しで起こすのでキメラの腕名を拒否する・§2.4）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import act_chimera_0913 as AC
from . import edge_law_0905 as EL
from . import gate_dial_0902 as GD
from . import weird_act_0903 as WA
from .act_chimera_0913 import DPHI, PHI, ChimeraMLPL
from .common import ROOT, load_config
from .dose_const_5m import setup_arm_const
from .edge_law_0905 import (EdgeRecorder, N_FREE, PERIOD, RSS_HEADROOM, RSS_PARALLEL_CAP,
                            S_NULL_SKIP, _apply_hook, _free_gib, _git_head, _hook_of,
                            _train_fn, recorder_bytes, write_arm_logs_edge)
from .elu_swamp import exact_layer_record_elu
from .envs import LTUTarget
from .gate_dial_0902 import (SanityError, _activation, _arm, _arm_status_path,
                             _load_divergence_status)
from .gate_dose import IDENTITY_TOL, SIGMA_TOL
from .mlp2_phase0 import _sha_file, identity_sanity_pass, require_omp
from .mlp2_phase0b import _complete_arm_logs, _window_indices
from .mlp2_phase1 import NUMERIC_DIVERGENCE, NumericDivergenceError
from .ratchet_log import teacher_f64

EXPERIMENT = "act_chimera_0913"
ENV = "scr"
CONFIG = Path(ROOT) / "configs" / "act_chimera_scr_0913.yaml"
SPEC = Path(ROOT) / "specs" / "spec_act_chimera_0913.md"
LOP_EVERY = 1000                      # gate_dial_0902.yaml common.lop_every（記録間隔）
EPS64 = 2.0 ** -52
EPS32 = 2.0 ** -23
# 活性化名（VecMLPL の名前）→ act_chimera_0913 の表の名前
ACT_NAME = {"leaky_relu": "LR", "elu": "ELU1"}
# 32 パターンの自由 5 ビット。envs.SCREnv.patterns と同じ順（(arange(32)[:, None] >> arange(5)) & 1）。
X_FREE = (((np.arange(2 ** N_FREE)[:, None] >> np.arange(N_FREE)) & 1).astype(np.float64))

# G1-SCR で s_null が比べるキーの期待一覧（§2.4）。**実装前に**
# proj_004_drift/results/edge_law_0905/logs/LRnull_1216_seed0.npz（71 キー）から作った: 全キーから
# S_NULL_SKIP（arm, run_id, state_hash_final）と state_hash_1m（別に比べる）を除いた 67 キー。
# Enull_1216_seed0.npz のキー集合は同一（照合済み・2026-09-14）。
S_NULL_EXPECTED_KEYS = (
    'act_alpha', 'activation', 'batch_mode', 'dose_formula', 'dose_relative_error',
    'eval_loss_exact', 'family', 'flip_state', 'freeze_v', 'gamma', 'gamma_negative',
    'init_hook', 'init_hook_arg', 'layer1_B', 'layer1_M', 'layer1_absmob', 'layer1_alive',
    'layer1_denom', 'layer1_dose', 'layer1_dzbar', 'layer1_eff_rank', 'layer1_eff_rank_W',
    'layer1_eff_rank_per_alive', 'layer1_m_dphi2', 'layer1_m_dphiddphi', 'layer1_m_phi2',
    'layer1_m_phidphi', 'layer1_median_B', 'layer1_median_M', 'layer1_mob',
    'layer1_moment_step', 'layer1_mu_norm', 'layer1_n_na', 'layer1_p_hat',
    'layer1_preact_sd_median', 'layer1_q25_M', 'layer1_q75_M', 'layer1_sigma_rms',
    'layer1_sign_clone_frac', 'layer1_sign_match_mean', 'layer1_stable_rank_W',
    'layer1_strict_dead', 'layer1_submerged', 'layer1_top1_frac', 'layer1_v_unit',
    'layer1_w_free', 'layer1_w_free_step', 'layer1_w_norm', 'layer1_w_norm_median',
    'layer1_w_norm_q25', 'layer1_w_norm_q75', 'layer1_wcos_mean', 'layer1_zbar',
    'layer1_zmax', 'layer1_zmean', 'layer1_zmin', 'lr_used', 'mu_cos_off',
    'mu_norm_formula', 'residual_var', 'seed', 'signal_var', 'step', 'target_dose',
    'target_mu_norm', 'task_period', 'unfit')
assert len(S_NULL_EXPECTED_KEYS) == 67 and len(set(S_NULL_EXPECTED_KEYS)) == 67

# S14 の行数の期待値（実装前に宿主から `edge_law_0905._body` で数えて固定・2026-09-14）
S14_EXPECTED_LINES = {"_run_arm_edge": 45, "_run_arm_weird": 41, "setup_arm_dial": 9}
# S14 の AST の文の数（深さつき・act_chimera_0913.ast_body_stmts）。宿主の source から数えて固定（2026-09-14:
# edge_law_0905._run_arm_edge 35 文・gate_dial_0902.setup_arm_dial 7 文）。写しはそれぞれ登録挿入 1 文だけ多い。
S14_EXPECTED_STMTS = {"_run_arm_edge": 35, "setup_arm_dial": 7}
S14_AST_REGISTERED = {
    "run": (("R-SCR", "insert_before", '0:c = copy.deepcopy(cfg)', ('0:setup_arm_dial = setup_arm_chimera',)),),
    "setup": (("S-SCR", "insert_before", '0:st["net"].set_activation(act, alpha, "alpha_exp")',
               ('0:st["net"].__class__ = ChimeraMLPL',)),),
}

# §2.1: SCR は参照を作ったインタプリタに固定する（G0・main で assert し、全 provenance に書く）
PY_SCR = "/home/issan/Projects/claude/proj_004_drift/.venv/bin/python"
EXPECTED_ENV = dict(executable=PY_SCR, torch="2.13.0+cu130", numpy="2.5.1")

# provenance に sha256 を書くコード・spec・host（§8）
CODE_FILES = ("src/act_chimera_scr_0913.py", "src/act_chimera_0913.py", "src/nets.py",
              "src/edge_law_0905.py", "src/gate_dial_0902.py", "src/gate_dose.py",
              "src/weird_act_0903.py", "src/elu_swamp.py", "src/dose_const_5m.py",
              "src/mlp2_phase0.py", "src/mlp2_phase0b.py", "src/mlp2_phase1.py", "src/envs.py",
              "src/ratchet_log.py", "configs/act_chimera_scr_0913.yaml",
              "configs/edge_law_0905.yaml", "configs/gate_dial_0902.yaml",
              "specs/spec_act_chimera_0913.md")


# ---------------------------------------------------------------------------
# Config（§2.4 経路 3: edge_law の大域を差し替える）
# ---------------------------------------------------------------------------
def bind_config(config: str | Path | None = None) -> Path:
    """``edge_law_0905.CONFIG`` を本 config に、``_TABLE`` を None にする（``table()`` / ``_hook_of()`` /
    ``registered()`` / ``build_cfg()`` がこの大域を読む）。同じ config なら何もしない。"""
    path = Path(config or CONFIG).resolve()
    if Path(EL.CONFIG).resolve() != path:
        EL.CONFIG = path
        EL._TABLE = None
    return path


def registered(config: str | Path | None = None) -> dict:
    return load_config(str(bind_config(config)))


def table() -> dict:
    """腕名 → 腕行（config の逐語・edge_law の ``arm_table`` で読む）。"""
    bind_config()
    return EL.table()


def arm_order() -> list[str]:
    return list(table())


def build_cfg() -> dict:
    """宿主 config ＋ 本 config の activation マップと腕表（``edge_law_0905.build_cfg``）。"""
    bind_config()
    return EL.build_cfg()


def env_info() -> dict:
    return dict(executable=sys.executable, torch=torch.__version__, numpy=np.__version__)


def assert_env() -> dict:
    """§2.1: interpreter・torch・numpy が SCR の参照（proj_004_drift/.venv）と一致すること。CLI の入口で呼ぶ
    （pytest は /usr/bin/python3 から関数を直接呼ぶので、関数の中では assert しない）。"""
    got = env_info()
    bad = {k: (got[k], v) for k, v in EXPECTED_ENV.items() if got[k] != v}
    if bad:
        raise SanityError(f"SCR interpreter/torch/numpy differ from the reference (§2.1): {bad}")
    return got


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict:
    """§6: 走らせたコードが commit に入っているか（src/ configs/ specs/ の変更と未追跡を含む）。"""
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--", "src", "configs", "specs")
    return dict(git_hash=_git_head(), git_dirty_src=(None if dirty is None else
                                                    [l for l in dirty.splitlines() if "__pycache__" not in l]))


def registered_outdir_family(outdir: Path) -> str | None:
    """outdir が本走（output.dir）か退避枝（output.dir_lr0p005）か。それ以外（スモーク・検査の一時）は None。"""
    R = registered()["output"]
    p = Path(outdir).resolve()
    if p == (Path(ROOT) / R["dir"]).resolve():
        return "main"
    if p == (Path(ROOT) / R["dir_lr0p005"]).resolve():
        return "lr0p005"
    return None


def row_family(row: dict) -> str:
    """腕行が本走（lr 0.01・5M）か退避枝（hook lr・10M）か（§2.4）。"""
    hook = row.get("hook") or {}
    return "lr0p005" if (hook.get("type") == "lr" or "_lr0p005_" in str(row["name"])) else "main"


def act_name_of(activation: str) -> str:
    """ログ / config の activation 名 → ``act_chimera_0913`` の表の名前（未知名は KeyError）。"""
    name = ACT_NAME.get(str(activation), str(activation))
    if name not in AC.ACT_NAMES or name == "LR02":
        raise KeyError(f"{activation!r} is not an SCR activation of act_chimera_0913")
    return name


# ---------------------------------------------------------------------------
# 経路 2: setup_arm_dial の写し ＋ クラス差し替え 1 行（S13・S14）
# ---------------------------------------------------------------------------
SETUP_INSERT = ('st["net"].__class__ = ChimeraMLPL',)


def setup_arm_chimera(cfg: dict, arm_cfg: dict, device: str) -> dict:
    """``gate_dial_0902.setup_arm_dial``（:393-405）の写し ＋ 登録挿入 1 行。以下 2 行は宿主の docstring の逐語。
    ``set_activation`` は乱数を消費せず状態も書き換えないので、腕は
    ``gate_dose_0830`` と init・教師・入力列・flip が bit 一致する（S-pair）。
    """
    st = setup_arm_const(cfg, arm_cfg, device)
    act, alpha = _activation(cfg, arm_cfg)
    st["net"].__class__ = ChimeraMLPL           # ← 登録挿入（§2.4 経路 2・乱数も状態も触らない）
    st["net"].set_activation(act, alpha, "alpha_exp")
    st["activation"] = act
    st["act_alpha"] = float(alpha)
    st["family"] = str(arm_cfg.get("family", ""))
    return st


# ---------------------------------------------------------------------------
# 経路 4: _run_arm_edge の写し ＋ 登録挿入 1 行（S14 が逐語照合する）
# ---------------------------------------------------------------------------
RUN_INSERT = ('setup_arm_dial = setup_arm_chimera',)
# weird_act_0903._run_arm_weird に対しては、本挿入 ＋ edge_law の RUN_INSERTS（順序もこの通り）
RUN_INSERTS_VS_WEIRD = (RUN_INSERT,) + tuple(EL.RUN_INSERTS)


def _run_arm_chimera(cfg: dict, arm: str, device: str, outdir: Path,
                     seeds: list[int], total: int) -> dict:
    """``edge_law_0905._run_arm_edge``（:525-574）の写し（差は本体先頭の登録挿入 1 行だけ）。"""
    setup_arm_dial = setup_arm_chimera       # ← 登録挿入（§2.4 経路 4・`st = setup_arm_dial(...)` より前）
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = seeds
    st = setup_arm_dial(c, _arm(c, arm), device)
    # ↓ 登録挿入 1: init フック（in-place・乱数消費なし・恒等式検査より前）
    _apply_hook(st, _hook_of(arm), arm)
    every = int(c["common"]["lop_every"])
    probes = list(range(0, total + 1, every))
    if probes[-1] != total:
        probes.append(total)
    _, sanity0 = exact_layer_record_elu(st, SIGMA_TOL)
    if not identity_sanity_pass(sanity0, IDENTITY_TOL):
        raise SanityError(f"{arm} initial exact-support identity failed")
    # ↓ 登録挿入 2: 記録器・学習ループ・書き出しの差し替え（局所束縛）
    WeirdRecorder = EdgeRecorder
    train_arm_gate = _train_fn(st)
    write_arm_logs_dial = write_arm_logs_edge
    rec = WeirdRecorder(probes, st)          # ← 宿主との唯一の差（spec §10 追補 2）
    checkpoints = [int(v) for v in c["common"].get("checkpoints", []) if int(v) <= total]
    print(f"[{arm}] act={st['activation']} dial={st['act_alpha']:g} "
          f"dose={st.get('target_dose')} seeds={seeds} steps={total:,}", flush=True)
    started = time.time()
    try:
        elapsed = train_arm_gate(st, rec, probes, total, outdir, checkpoints)
    except NumericDivergenceError as exc:
        elapsed = time.time() - started
        event = dict(exc.event)
        event.update(probe_every=every, registered_total_steps=int(total),
                     registered_seeds=[int(v) for v in seeds],
                     activation=st["activation"], act_alpha=st["act_alpha"],
                     family=st.get("family"), elapsed_sec=float(elapsed),
                     detection="nonfinite_training_state_at_probe",
                     partial_logs_excluded=True, rescue="none")
        path = _arm_status_path(outdir, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[{arm}] {NUMERIC_DIVERGENCE} at step {event['detected_step']:,}",
              flush=True)
        return dict(status=NUMERIC_DIVERGENCE, elapsed_sec=elapsed,
                    sanity=dict(pass_=False, numeric_divergence=True, event=event),
                    divergence=event)
    sanity = rec.sanity()
    if not sanity["pass_"]:
        raise SanityError(f"{arm} exact-support sanity failed: {sanity}")
    write_arm_logs_dial(outdir, arm, st, rec)
    print(f"[{arm}] complete in {elapsed:.1f}s", flush=True)
    return dict(status="COMPLETE", elapsed_sec=elapsed, sanity=sanity)


# ---------------------------------------------------------------------------
# 腕プロセスの投入単位（edge_law_0905.run_single_arm:577-623 に倣う・provenance を足す）
# ---------------------------------------------------------------------------
def _machine() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "nvidia-smi unavailable"
    return dict(hostname=socket.gethostname(), cpu=cpu, gpu=gpu, device="cpu",
                platform=platform.platform())


def _other_processes() -> dict:
    """他セッションの学習プロセス（PID とコマンド行）と nvidia-smi の compute プロセス（§7・§8）。"""
    me = os.getpid()
    ps = []
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid,etimes,rss,args"], capture_output=True,
                             text=True, timeout=20).stdout.splitlines()
        for line in out[1:]:
            parts = line.split(None, 4)
            if len(parts) < 5 or int(parts[0]) == me:
                continue
            args = parts[4]
            if any(k in args for k in ("python", "torch", "src.")) and "ps -eo" not in args:
                ps.append(dict(pid=int(parts[0]), ppid=int(parts[1]), etimes=int(parts[2]),
                               rss_kib=int(parts[3]), args=args[:300]))
    except (OSError, subprocess.SubprocessError, ValueError):
        ps.append(dict(error="ps failed"))
    try:
        nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        nv = "nvidia-smi unavailable"
    return dict(ps=ps, nvidia_smi_compute=nv)


def _sha_or_none(path: Path) -> str | None:
    return _sha_file(path) if path.exists() else None


def code_sha256() -> dict:
    return {rel: _sha_or_none(Path(ROOT) / rel) for rel in CODE_FILES}


def reference_sha256(arms=None) -> dict:
    """G1 の参照 npz（edge_law の null 腕 × 10 系列・proj_004_drift では未追跡）の sha256（§2.4）。"""
    G = registered()["g1"]
    ref = Path(G["reference_logs"])
    out = {}
    for arm, ref_arm in G["pairs"].items():
        if arms is not None and arm not in arms:
            continue
        for seed in registered()["common_overrides"]["seeds"]:
            p = ref / f"{ref_arm}_seed{int(seed)}.npz"
            out[str(p)] = _sha_or_none(p)
    return out


def run_single_arm(arm: str, steps: int | None = None,
                   outdir: Path | None = None, seeds: list[int] | None = None,
                   cfg: dict | None = None, generator_offset: int | None = None) -> dict:
    """1 腕だけ走らせて logs と provenance を置いて終わる（launcher の投入単位・§7）。

    ``generator_offset`` は G1-SCR の変異対照（``generator_offset: 1`` の 30k で ``unfit[:31]`` が
    不一致になること・§6）のためだけにあり、``steps`` 無し（登録の地平線）では拒否する。
    """
    bind_config()
    row = table()[str(arm)]
    cfg = build_cfg() if cfg is None else cfg
    require_omp(cfg)
    total = int(steps) if steps else int(row["total_steps"])
    cfg = copy.deepcopy(cfg)
    cfg["common"]["total_steps"] = total
    cfg["common"]["checkpoints"] = list(row["checkpoints"])
    if generator_offset is not None:
        if steps is None:
            raise SanityError(f"{arm}: generator_offset={generator_offset} is a check-only "
                              f"mutation; the registered run uses the config's offset")
        cfg["common"]["generator_offset"] = int(generator_offset)
    fam = row_family(row)
    if outdir is None:
        if steps is not None or generator_offset is not None:
            # 短縮・検査の走は本走の出力先に書かない（arm_status/{arm}_done.json と logs で本走を取り違える・§6・§8）
            raise SanityError(f"{arm}: --outdir is required for shortened / check runs (steps={steps})")
        # §2.4: 退避枝の行は output.dir_lr0p005、主行は output.dir に固定する（混ざらない）
        out = Path(ROOT) / registered()["output"]["dir_lr0p005" if fam == "lr0p005" else "dir"]
    else:
        out = Path(outdir)
        reg = registered_outdir_family(out)
        if reg is not None and reg != fam:
            raise SanityError(f"{arm} ({fam} row) must not be written to the {reg} output dir {out} (§2.4)")
    out.mkdir(parents=True, exist_ok=True)
    use = [int(v) for v in (seeds if seeds is not None else cfg["common"]["seeds"])]
    if use != [int(v) for v in cfg["common"]["seeds"]]:
        # seed はベクトル化されている（1 本の系列から [R,...] を一度に引く）ので、部分集合は
        # 10 seed 走と**別の入力列**になる（edge_law_0905.py:593-606・§2.4）。
        if steps is None:
            raise SanityError(
                f"{arm}: a seed subset {use} is not a valid registered run — "
                f"the input stream differs from the 10-seed run from step 1 "
                f"(s_seed_split_note). Pass --steps for a shortened check run.")
        print(f"[{arm}] WARNING: running a seed subset {use}; the input stream "
              f"differs from the registered 10-seed run — checks only", flush=True)
    every = int(cfg["common"]["lop_every"])
    if _load_divergence_status(out, arm, use, total, every) is not None:
        print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; nothing to do", flush=True)
        return dict(status=NUMERIC_DIVERGENCE)
    if _complete_arm_logs(out, arm, use, total, every):
        # 完了したログの再利用は、同じ commit のコードで作られた場合だけ（コードを変えたら回し直す・§6）
        done_path = out / "arm_status" / f"{arm}_done.json"
        prev = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
        head = _git_head()
        if prev.get("git_head") != head:
            raise SanityError(f"{arm}: complete logs in {out} were made by git_head={prev.get('git_head')!r}, "
                              f"current HEAD={head!r}; move them away before re-running (§6)")
        print(f"[{arm}] complete logs found (same git_head); nothing to do", flush=True)
        return dict(status="COMPLETE", resumed=True)
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others_before = _other_processes()
    result = _run_arm_chimera(cfg, arm, "cpu", out, use, total)
    wall = time.time() - started
    status = dict(result)
    status.update(arm=arm, total_steps=total, seeds=use, wall_sec=wall, hook=row["hook"],
                  git_head=_git_head())
    status.pop("sanity", None)
    done = out / "arm_status" / f"{arm}_done.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    # --- provenance（§8: run id・arm・seed・env・lr・git_hash・sha256・interpreter・機械・wall・RSS・他プロセス）
    lr = float(row["hook"]["value"]) if (row["hook"] and row["hook"].get("type") == "lr") \
        else float(cfg["common"]["lr_main"])
    prov = dict(run_id=EXPERIMENT, env=ENV, arm=arm, family=row["family"],
                activation=row["activation"], act_name=act_name_of(row["activation"]),
                dial=row["dial"], u_fr=row["u_fr"], hook=row["hook"], lr=lr,
                seeds=use, total_steps=total, registered_total_steps=int(row["total_steps"]),
                checkpoints=list(row["checkpoints"]),
                generator_offset=int(cfg["common"]["generator_offset"]),
                shortened=bool(steps), outdir=str(out), status=status.get("status"),
                row_family=fam, **git_state(), config=str(CONFIG), spec=str(SPEC),
                sha256=code_sha256(),
                reference_npz_sha256=reference_sha256(arms=(arm,)),
                interpreter=sys.executable, torch=torch.__version__, numpy=np.__version__,
                env_matches_reference=(env_info() == EXPECTED_ENV), band_predicate_dtype=AC.BAND_PREDICATE_DTYPE,
                machine=_machine(), started_at=started_at, wall_sec=wall,
                peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                torch_num_threads=int(torch.get_num_threads()),
                other_processes_before=others_before, other_processes_after=_other_processes())
    (out / "arm_status" / f"{arm}_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# S14（SCR）: 写しの差分が登録挿入だけ・行数が固定した期待値（§6）
# ---------------------------------------------------------------------------
def s14_scr(mine_run_src: str | None = None, mine_setup_src: str | None = None,
            expected_stmts: dict | None = None) -> dict:
    """S14（SCR）。主の検査は AST（§6 S14「本体を AST で取り出し」）: 写しの本体の文（入れ子の深さつき・
    ``act_chimera_0913.ast_body_stmts``）= 宿主の本体 + 登録挿入 1 文、宿主の文の数 = 固定した期待値。
    字下げだけを変えた写し（制御の流れが変わる）は深さで捕まる（edge_law の ``_body`` は字下げを落とすので捕まえない）。
    副の検査として edge_law の ``_copy_opcodes``（行の正規化）と行数の期待値も残す。"""
    want_stmts = dict(S14_EXPECTED_STMTS if expected_stmts is None else expected_stmts)
    here = Path(__file__).read_text(encoding="utf-8")
    ast_run = AC.ast_copy_check(Path(EL.__file__).read_text(encoding="utf-8"), "_run_arm_edge",
                                here if mine_run_src is None else mine_run_src, "_run_arm_chimera",
                                S14_AST_REGISTERED["run"], want_stmts["_run_arm_edge"])
    ast_setup = AC.ast_copy_check(Path(GD.__file__).read_text(encoding="utf-8"), "setup_arm_dial",
                                  here if mine_setup_src is None else mine_setup_src, "setup_arm_chimera",
                                  S14_AST_REGISTERED["setup"], want_stmts["setup_arm_dial"])
    run_edge = EL._copy_opcodes(EL._run_arm_edge, _run_arm_chimera, (RUN_INSERT,))
    run_weird = EL._copy_opcodes(WA._run_arm_weird, _run_arm_chimera, RUN_INSERTS_VS_WEIRD)
    setup = EL._copy_opcodes(GD.setup_arm_dial, setup_arm_chimera, (SETUP_INSERT,))
    want = S14_EXPECTED_LINES
    lines_ok = (run_edge["host_lines"] == want["_run_arm_edge"]
                and run_edge["mine_lines"] == want["_run_arm_edge"] + len(RUN_INSERT)
                and run_weird["host_lines"] == want["_run_arm_weird"]
                and setup["host_lines"] == want["setup_arm_dial"]
                and setup["mine_lines"] == want["setup_arm_dial"] + len(SETUP_INSERT))
    return dict(pass_=bool(ast_run["pass_"] and ast_setup["pass_"] and run_edge["pass_"] and run_weird["pass_"]
                           and setup["pass_"] and lines_ok),
                ast_pass=bool(ast_run["pass_"] and ast_setup["pass_"]),
                expected_lines=dict(want), expected_stmts=want_stmts, lines_ok=lines_ok,
                ast_run_arm_vs_edge=ast_run, ast_setup_arm=ast_setup,
                run_arm_vs_edge=run_edge, run_arm_vs_weird=run_weird, setup_arm=setup)


# ---------------------------------------------------------------------------
# G1-SCR（§2.4・§6）: edge_law の null 腕と s_null の規則で bit 一致
# ---------------------------------------------------------------------------
def _compared_keys(mine: Path, ref: Path) -> list[str]:
    """``edge_law_0905.s_null`` が比べるキー（:1457-1459 と同じ規則）。"""
    with np.load(mine, allow_pickle=True) as a, np.load(ref, allow_pickle=True) as b:
        keys = sorted(set(a.files) & set(b.files))
    return [k for k in keys if k not in S_NULL_SKIP and k != "state_hash_1m"]


AUX_STEP_KEYS = {"layer1_moment_step": ("layer1_m_phi2", "layer1_m_dphi2", "layer1_m_phidphi",
                                        "layer1_m_dphiddphi"),
                 "layer1_w_free_step": ("layer1_w_free",)}


def s_null_prefix(arm: str, ref_dir: Path, outdir: Path, ref_arm: str) -> dict:
    """スモーク用の s_null（§6: 30k を参照の先頭 31 記録と比べる）。

    ``edge_law_0905.s_null`` と同じ規則（共通キー・S_NULL_SKIP と state_hash_1m を除く・
    ``np.array_equal(equal_nan=True)``・step 長の列は共通の記録数に切り詰め）に、s_null が扱えない
    独自の step 列を連れる 7 列（``layer1_moment_step`` / ``layer1_w_free_step`` とそれに従う列・
    edge_law_0905.py:353-358）の扱いを足す: 両ログの step 列の共通の値で行を揃えて比べる（短縮走行では
    末尾 20 タスクの 1000-step 記録が全区間に掛かるので、単純な先頭切り詰めでは揃わない）。
    本走（記録数が同じ）では s_null そのものを使う（``g1_scr``）。"""
    if str(ref_arm) == str(arm) and (
            Path(ref_dir).resolve() == (Path(outdir) / "logs").resolve()):
        raise SanityError(f"s_null_prefix would compare {arm!r} with itself; vacuous")
    rows = []
    for seed in range(10):
        pa = Path(outdir) / "logs" / f"{arm}_seed{seed}.npz"
        pb = Path(ref_dir) / f"{ref_arm}_seed{seed}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=seed, status="MISSING", mine=pa.exists(), reference=pb.exists()))
            continue
        a = np.load(pa, allow_pickle=True)
        b = np.load(pb, allow_pickle=True)
        n = min(len(a["step"]), len(b["step"]))
        aux = {}
        for skey, cols in AUX_STEP_KEYS.items():
            common = np.intersect1d(a[skey], b[skey])
            ia = np.flatnonzero(np.isin(a[skey], common))
            ib = np.flatnonzero(np.isin(b[skey], common))
            for key in (skey,) + cols:
                aux[key] = (ia, ib, int(common.size))
        bad, compared, aux_rows = {}, [], {}
        for key in sorted(set(a.files) & set(b.files)):
            if key in S_NULL_SKIP or key == "state_hash_1m":
                continue
            x, y = a[key], b[key]
            if key in aux:
                ia, ib, m = aux[key]
                x, y = x[ia], y[ib]
                aux_rows[key] = m
            else:
                if x.ndim >= 1 and x.shape[0] == len(a["step"]):
                    x = x[:n]
                if y.ndim >= 1 and y.shape[0] == len(b["step"]):
                    y = y[:n]
            compared.append(key)
            if x.shape != y.shape or x.dtype != y.dtype:
                bad[key] = f"shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}"
            elif x.dtype.kind in "fiub":
                if not np.array_equal(x, y, equal_nan=True):
                    bad[key] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
            elif not np.array_equal(x, y):
                bad[key] = "differs"
        reached = (int(a["step"][-1]) >= 1_000_000 and int(b["step"][-1]) >= 1_000_000)
        hash_equal = (str(a["state_hash_1m"]) == str(b["state_hash_1m"])) if reached else None
        rows.append(dict(seed=seed, status="OK", n_records=int(n), n_compared=len(compared),
                         n_bad=len(bad), bad=bad, state_hash_1m_equal=hash_equal,
                         aux_rows_compared=aux_rows))
    ok = bool(rows) and all(r.get("status") == "OK" and r["n_bad"] == 0
                            and r["state_hash_1m_equal"] in (None, True) for r in rows)
    return dict(pass_=ok, check="S-null-prefix", arm=arm, reference_arm=ref_arm,
                reference_dir=str(ref_dir), rows=rows)


def g1_scr(outdir: Path, ref_dir: Path | None = None, *, smoke: bool = False,
           arms: tuple[str, ...] | None = None) -> dict:
    """chLR_1216 / chE_1216（退避枝では *_lr0p005_1216）を参照と照合する。

    - ``ref_arm`` / ``ref_dir`` を明示で ``s_null`` に渡す（:1434-1440）。
    - 比べたキーの名前と件数が ``S_NULL_EXPECTED_KEYS``（67）に一致すること。
    - 本走: 10 系列すべてで ``n_records == 5001``（10M 行は 10001）、``state_hash_1m`` に到達し一致。
    - スモーク: 自分のログが 31 記録で、参照の先頭 31 記録と一致（``s_null`` の切り詰め）。
    """
    bind_config()
    R = registered()
    G = R["g1"]
    ref_dir = Path(ref_dir) if ref_dir is not None else Path(G["reference_logs"])
    outdir = Path(outdir)
    seeds = [int(v) for v in R["common_overrides"]["seeds"]]
    rows = {}
    fam = registered_outdir_family(outdir)
    for arm, ref_arm in G["pairs"].items():
        if arms is not None and arm not in arms:
            continue
        if fam is not None and row_family(table()[arm]) != fam:
            continue                                  # 本走の dir では主の親だけ、退避枝の dir では lr0p005 の親だけ
        if not (outdir / "logs" / f"{arm}_seed{seeds[0]}.npz").exists():
            continue
        res = (s_null_prefix(arm, ref_dir, outdir, str(ref_arm)) if smoke
               else EL.s_null(arm, ref_dir, outdir, ref_arm=str(ref_arm)))
        total = int(table()[arm]["total_steps"])
        want_n = int(R["smoke"]["n_records"]) if smoke else int(G["n_records"][str(total)])
        seed_rows = []
        for r in res["rows"]:
            seed = int(r["seed"])
            entry = dict(r)
            if r.get("status") != "OK":
                entry["pass_"] = False
                seed_rows.append(entry)
                continue
            keys = _compared_keys(outdir / "logs" / f"{arm}_seed{seed}.npz",
                                  ref_dir / f"{ref_arm}_seed{seed}.npz")
            with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=False) as z:
                n_mine = int(len(z["step"]))
            keys_ok = (tuple(keys) == S_NULL_EXPECTED_KEYS
                       and int(r["n_compared"]) == len(S_NULL_EXPECTED_KEYS))
            hash_ok = (r["state_hash_1m_equal"] is None) if smoke else \
                (r["state_hash_1m_equal"] is True)
            entry.update(keys_ok=keys_ok, keys_missing=sorted(set(S_NULL_EXPECTED_KEYS) - set(keys)),
                         keys_extra=sorted(set(keys) - set(S_NULL_EXPECTED_KEYS)),
                         n_records_mine=n_mine, n_records_want=want_n,
                         n_records_ok=(n_mine == want_n and int(r["n_records"]) == want_n),
                         state_hash_1m_ok=hash_ok,
                         pass_=bool(r["n_bad"] == 0 and keys_ok and n_mine == want_n
                                    and int(r["n_records"]) == want_n and hash_ok))
            seed_rows.append(entry)
        rows[arm] = dict(pass_=bool(res["pass_"] and seed_rows
                                    and all(e["pass_"] for e in seed_rows)
                                    and len(seed_rows) == len(seeds)),
                         reference_arm=str(ref_arm), reference_dir=str(ref_dir),
                         s_null_pass=res["pass_"], n_expected_keys=len(S_NULL_EXPECTED_KEYS),
                         rows=seed_rows)
    ok = bool(rows) and all(v["pass_"] for v in rows.values())
    return dict(pass_=ok, check="G1-SCR", smoke=bool(smoke), arms=rows,
                reference_npz_sha256=reference_sha256(arms=tuple(rows)))


def g1_smoke_control(outdir_ctrl: Path, ref_dir: Path | None = None,
                     arm: str = "chLR_1216", n_records_want: int | None = None) -> dict:
    """G1-SCR の変異対照: ``generator_offset: 1`` の走の ``unfit[:n]`` が参照と不一致であること（§6）。
    対照の走そのものが登録の記録数（スモーク 31・``n_records_want``）に届いていることも pass に要る
    （途中で止まった対照が「不一致」で通らないように）。"""
    bind_config()
    R = registered()
    G = R["g1"]
    ref_dir = Path(ref_dir) if ref_dir is not None else Path(G["reference_logs"])
    n = int(R["smoke"]["n_records"]) if n_records_want is None else int(n_records_want)
    rows = []
    for seed in R["common_overrides"]["seeds"]:
        pa = Path(outdir_ctrl) / "logs" / f"{arm}_seed{int(seed)}.npz"
        pb = ref_dir / f"{G['pairs'][arm]}_seed{int(seed)}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=int(seed), status="MISSING"))
            continue
        with np.load(pa, allow_pickle=False) as a, np.load(pb, allow_pickle=False) as b:
            ua, ub = a["unfit"][:n], b["unfit"][:n]
            off = int(a["step"].shape[0])
        differs = not np.array_equal(ua, ub)
        rows.append(dict(seed=int(seed), status="OK", n=int(min(len(ua), len(ub))),
                         n_records=off, unfit_differs=bool(differs),
                         maxabs=float(np.nanmax(np.abs(ua[:len(ub)] - ub[:len(ua)])))))
    ok = bool(rows) and all(r.get("status") == "OK" and r["unfit_differs"] and r["n_records"] == n for r in rows)
    return dict(pass_=ok, check="G1-SCR-control", arm=arm, generator_offset=1, n_records_want=n, rows=rows)


# ---------------------------------------------------------------------------
# RSS（§2.4・§7）: 自前の rss_probe（子を本モジュールで起こす・地平線は本表から）
# ---------------------------------------------------------------------------
def _rss_extrapolate(peak_kib: float, steps: int, horizons, free_gib: float | None = None,
                     headroom: float = RSS_HEADROOM) -> dict:
    """短縮走行のピーク RSS を、記録器の配列サイズの差分で本走の地平線へ外挿する（edge_law の
    ``rss_probe``:1224-1258 の算術・``recorder_bytes`` はそのまま使う）。J = ⌊free/(1.5·peak)⌋ − 1（§7）。"""
    gib = 1024.0 ** 3
    peak_gib = float(peak_kib) / (1024.0 ** 2)
    base_gib = peak_gib - recorder_bytes(int(steps)) / gib
    projected = {int(h): base_gib + recorder_bytes(int(h)) / gib for h in horizons}
    worst = max(projected.values())
    free = _free_gib() if free_gib is None else float(free_gib)
    edge_parallel = max(1, min(RSS_PARALLEL_CAP, int(free // (headroom * max(worst, 1e-9)))))
    j_spec = max(0, int(free // (headroom * max(worst, 1e-9))) - 1)
    return dict(steps=int(steps), peak_rss_gib=peak_gib, base_gib=base_gib,
                projected_peak_gib=projected, worst_peak_gib=worst, free_gib=free,
                edge_law_parallel=edge_parallel, J_spec=j_spec,
                formula="J = floor(MemAvailable_GiB / (1.5 * worst projected peak)) - 1 (spec §7); "
                        "edge_law_parallel = min(20, floor(free / (1.5 * worst)))")


def rss_probe(outdir: Path, arms: tuple[str, ...] | None = None,
              steps: int | None = None) -> dict:
    """§2.4: 子を ``python -m src.act_chimera_scr_0913 --arm <arm> --steps 50000 --outdir <smoke>/rss/run``
    で起こし、``/usr/bin/time -v`` のピーク RSS を本走の地平線（5M・退避枝 10M）に外挿する。
    測る腕は親・キメラ・分離腕の各経路から 1 つ（config ``rss_probe.arms``）で、外挿したピークの最大を使う。"""
    bind_config()
    P = registered()["rss_probe"]
    arms = tuple(arms) if arms is not None else tuple(P["arms"])
    steps = int(steps if steps is not None else P["steps"])
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    horizons = sorted({int(row["total_steps"]) for row in table().values()})
    per_arm = {}
    for arm in arms:
        run_dir = outdir / "rss" / f"run_{arm}"
        if run_dir.exists():                       # 前回のログがあると子が再開して学習せずに終わる（RSS を過小評価する）
            shutil.rmtree(run_dir)
        cmd = ["/usr/bin/time", "-v", sys.executable, "-m", "src.act_chimera_scr_0913",
               "--arm", arm, "--steps", str(steps), "--outdir", str(run_dir)]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        peak_kb = None
        for line in proc.stderr.splitlines():
            if "Maximum resident set size" in line:
                peak_kb = float(line.rsplit(":", 1)[1].strip())
        trained = ("complete in" in (proc.stdout or "")) and "nothing to do" not in (proc.stdout or "")
        ext = _rss_extrapolate(peak_kb if peak_kb is not None else 0., steps, horizons, headroom=float(P["headroom"]))
        ext.update(arm=arm, returncode=proc.returncode, trained=bool(trained), peak_measured=peak_kb is not None,
                   pass_=bool(proc.returncode == 0 and trained and peak_kb is not None),
                   command=cmd, stdout_tail=(proc.stdout or "")[-1000:], stderr_tail=proc.stderr[-2000:])
        per_arm[arm] = ext
    worst = max(v["worst_peak_gib"] for v in per_arm.values())
    free = _free_gib()
    hr = float(P["headroom"])
    proj = {int(h): max(v["projected_peak_gib"][int(h)] for v in per_arm.values()) for h in horizons}
    return dict(pass_=all(v["pass_"] for v in per_arm.values()), arms=per_arm, steps=steps,
                horizons=horizons, worst_peak_gib=worst, projected_peak_gib=proj, free_gib=free,
                J_spec=max(0, int(free // (hr * max(worst, 1e-9))) - 1),
                edge_law_parallel=max(1, min(RSS_PARALLEL_CAP, int(free // (hr * max(worst, 1e-9))))))


# ---------------------------------------------------------------------------
# 読み出し（§3.3・§3.4 SCR・§8）: logs から 32 パターンの支持を再構成する
# ---------------------------------------------------------------------------
def reconstruct_z(zbar, w_free) -> np.ndarray:
    """z_p = z̄ + w_free·(x_free,p − 0.5)（§3.3）。zbar (..., U)、w_free (..., U, 5) → (32, ..., U) float64。

    32 パターンは flip ビット（定数）と自由 5 ビット（{0,1}）で、z_p − z̄ = W_free·(x_free,p − ½)。
    logs の zbar は float32 なので、再構成の誤差は 2^-24·|z̄| が上限（S19 の許容値の導出）。"""
    zb = np.asarray(zbar, dtype=np.float64)
    wf = np.asarray(w_free, dtype=np.float64)
    return zb[None] + np.einsum("...ub,pb->p...u", wf, X_FREE - 0.5)


def lam_out_of_phi(phi: np.ndarray) -> np.ndarray:
    """λ_max(2·E_p[φφᵀ])（出力層 v の Hessian 塊・float64・eigvalsh）。phi (P, ..., U) → (...)."""
    P = phi.shape[0]
    M = 2.0 * np.einsum("p...i,p...j->...ij", phi, phi) / P
    return np.linalg.eigvalsh(M)[..., -1]


def lam_out_diag_approx(phi: np.ndarray) -> np.ndarray:
    """S19 の変異対照: 対角（m_phi2）だけから λ を取る近似 2·max_i E[φ_i²]。"""
    return 2.0 * (phi * phi).mean(0).max(-1)


def lam_out_tolerance(phi: np.ndarray, dz_bound: np.ndarray) -> np.ndarray:
    """再構成 z の**事前の**誤差上限 |δz| ≤ dz_bound（要素別）から λ の許容値を導く（観測した差は使わない）。
    |φ′| ≤ 1 なので |δφ| ≤ |δz|。Weyl: |Δλ| ≤ ‖E‖₂、E = 2·E_p[φ̃φ̃ᵀ − φφᵀ]、
    ‖φ̃φ̃ᵀ − φφᵀ‖₂ ≤ 2‖φ‖‖δ‖ + ‖δ‖²。加えて eigvalsh の 8·eps64·λ。"""
    nphi = np.linalg.norm(phi, axis=-1)                       # (P, ...)
    nd = np.linalg.norm(np.broadcast_to(dz_bound, phi.shape), axis=-1)
    weyl = 2.0 * (2.0 * nphi * nd + nd * nd).mean(0)
    return weyl + 8.0 * EPS64 * lam_out_of_phi(phi)


def relax_rate(step: np.ndarray, dzbar: np.ndarray, zmax: np.ndarray, denom: np.ndarray,
               window, *, period: int = PERIOD, every: int = LOP_EVERY,
               alive_denom: float = 0.25) -> dict:
    """腕ごとの緩和率（§3.4・新規の道具）。1 系列分（配列は (n_rec, U)）。

    タスク t ∈ window の終端記録 k（step = t·period）と直前の記録 k−1（step − every・assert）を使い、
    ユニット i とタスク t をプールして y = dzbar[k, i]（k−1 → k の増分）を
    x = zmax[k−1, i] − zmax*_i（zmax*_i = window 内の zmax[k(t), i] の中央値）に切片つき OLS で回帰する。
    relax_rate = −傾き。全ユニット版を主にし、生きているユニット（window 平均の denom > alive_denom）だけの
    版を併記する。x を増分の前の記録から取るのは y と同じノイズを x に入れないため（gate_dose の q2）。"""
    step = np.asarray(step, dtype=np.int64)
    k = _window_indices(step, int(period), list(window))
    if k.size < 2 or k.min() < 1:
        return dict(rate=float("nan"), rate_alive=float("nan"), n=0, n_alive=0,
                    n_tasks=int(k.size), intercept=float("nan"), r2=float("nan"))
    gap = step[k] - step[k - 1]
    assert np.all(gap == int(every)), f"record spacing before task ends is {set(gap.tolist())}, want {every}"
    zm = np.asarray(zmax, dtype=np.float64)
    dz = np.asarray(dzbar, dtype=np.float64)
    zstar = np.median(zm[k], axis=0)                          # (U,)
    x = zm[k - 1] - zstar[None]
    y = dz[k]
    alive = np.asarray(denom, dtype=np.float64)[k].mean(0) > float(alive_denom)

    def _ols(xx, yy):
        m = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[m], yy[m]
        if xx.size < 2 or np.var(xx) == 0.0:
            return float("nan"), float("nan"), float("nan"), int(xx.size)
        xm, ym = xx.mean(), yy.mean()
        sxx = ((xx - xm) ** 2).sum()
        slope = ((xx - xm) * (yy - ym)).sum() / sxx
        resid = yy - (ym + slope * (xx - xm))
        syy = ((yy - ym) ** 2).sum()
        r2 = float(1.0 - (resid ** 2).sum() / syy) if syy > 0 else float("nan")
        return float(slope), float(ym - slope * xm), r2, int(xx.size)

    slope, icpt, r2, n = _ols(x.ravel(), y.ravel())
    slope_a, _, _, n_a = _ols(x[:, alive].ravel(), y[:, alive].ravel())
    return dict(rate=-slope, rate_alive=-slope_a, n=n, n_alive=n_a, n_tasks=int(k.size),
                n_units_alive=int(alive.sum()), intercept=icpt, r2=r2)


# ---- 全 Hessian / 代替勾配場の Jacobian の λ（S19・冪乗法） -------------------------------------------
def _surrogate_field(W, b, v, c, x_in, y, net):
    """32 パターン平均の勾配場（``edge_law_0905.grads_centered_elu_batch`` の L = 1 版・逐語の式）。
    主 8 腕では損失 E_p[(ŷ−y)²] の真の勾配、分離腕では φ′_bwd による代替勾配（勾配流ではない）。"""
    z = torch.einsum("rhd,prd->prh", W, x_in) + b
    phi = net.act_fn(z)
    yhat = (phi * v).sum(dim=-1) + c
    d2 = 2.0 * (yhat - y)
    gv = (d2[..., None] * phi).mean(dim=0)
    gc = d2.mean(dim=0)
    dz = d2[..., None] * v * net.act_grad(z, phi)
    gb = dz.mean(dim=0)
    gW = (dz[..., None] * x_in[:, :, None, :]).mean(dim=0)
    return gW, gb, gv, gc


def _dot_per_series(u, w) -> torch.Tensor:
    return sum((ui * wi).reshape(ui.shape[0], -1).sum(dim=1) for ui, wi in zip(u, w))


def _scale_per_series(u, s):
    return [ui * s.reshape((-1,) + (1,) * (ui.dim() - 1)) for ui in u]


def power_iteration(net, params, x_in, y, *, max_iter: int = 2000, rtol: float = 1e-10,
                    seed: int = 0) -> dict:
    """勾配場の Jacobian J の支配固有値（系列ごと・同時）。Jᵀu を autograd で作る（J と Jᵀ の
    スペクトルは同じ。主腕では J = Hessian で対称）。Rayleigh 商 λ_k = ⟨u, Jᵀu⟩/⟨u, u⟩ の
    相対変化 ≤ rtol で止め、最後の変化幅と Aitken 型の誤差推定（Δ_k·ρ/(1−ρ)、ρ = Δ_k/Δ_{k−1}）を記録する。"""
    W, b, v, c = [p.detach().clone().double().requires_grad_(True) for p in params]
    x_in = x_in.double()
    y = y.double()

    def jt(u):
        g = _surrogate_field(W, b, v, c, x_in, y, net)
        s = sum((gi * ui).sum() for gi, ui in zip(g, u))
        return [t.detach() for t in torch.autograd.grad(s, (W, b, v, c))]

    gen = torch.Generator().manual_seed(int(seed))
    u = [torch.randn(p.shape, generator=gen, dtype=torch.float64) for p in (W, b, v, c)]
    lam = torch.zeros(W.shape[0], dtype=torch.float64)
    deltas, dl = [], torch.full_like(lam, float("inf"))
    it = 0
    for it in range(1, int(max_iter) + 1):
        nrm = _dot_per_series(u, u).sqrt().clamp_min(1e-300)
        u = _scale_per_series(u, 1.0 / nrm)
        w = jt(u)
        lam_new = _dot_per_series(u, w)
        dl = (lam_new - lam).abs()
        delta = dl.max().item()
        deltas.append(delta)
        lam = lam_new
        u = w
        if bool((dl <= rtol * lam.abs().clamp_min(1e-300)).all()):
            break
    conv = (dl <= rtol * lam.abs().clamp_min(1e-300)).numpy()
    rho = (deltas[-1] / deltas[-2]) if len(deltas) >= 2 and deltas[-2] > 0 else float("nan")
    err_est = (deltas[-1] * rho / (1.0 - rho)) if (np.isfinite(rho) and 0 < rho < 1) else deltas[-1]
    return dict(lam=lam.numpy(), iterations=int(it), last_delta=float(deltas[-1]),
                converged=bool(conv.all()), converged_series=conv, last_delta_series=dl.numpy(),
                rho=float(rho), error_estimate=float(err_est), rtol=float(rtol))


def dense_jacobian_series(net, params, x_in, y, series) -> np.ndarray:
    """J（勾配場の Jacobian）を指定した系列について陽に組む（(len(series), n, n)）。基底ベクトルを全系列に同時に置き、
    n 回の VJP で指定系列の列を取る（系列は独立なので互いに混ざらない）。冪乗法が収束しない・負の固有値が大きい
    ときの λ_max の取り直しに使う。"""
    W, b, v, c = [p.detach().clone().double().requires_grad_(True) for p in params]
    shapes = [p.shape[1:] for p in (W, b, v, c)]
    sizes = [int(np.prod(s)) if len(s) else 1 for s in shapes]
    n = sum(sizes)
    series = [int(s) for s in series]
    out = np.zeros((len(series), n, n), dtype=np.float64)
    g = _surrogate_field(W, b, v, c, x_in.double(), y.double(), net)
    for j in range(n):
        u, off = [], 0
        for p, s, size in zip((W, b, v, c), shapes, sizes):
            blk = torch.zeros_like(p)
            if off <= j < off + size:
                idx = j - off
                if len(s):
                    blk.reshape(p.shape[0], -1)[:, idx] = 1.0
                else:
                    blk[:] = 1.0
            u.append(blk)
            off += size
        s_ = sum((gi * ui).sum() for gi, ui in zip(g, u))
        col = torch.autograd.grad(s_, (W, b, v, c), retain_graph=True)
        flat = torch.cat([t.detach().reshape(t.shape[0], -1) for t in col], dim=1).numpy()
        out[:, :, j] = flat[series]
    return out


def lam_max_full(net, params, x_in, y, *, symmetric: bool, max_iter: int = 2000, rtol: float = 1e-10) -> dict:
    """§2.4 の λ_max（系列ごと）。冪乗法は絶対値最大の固有値を符号つきで返すので、収束して λ > 0 の系列だけその値を使い、
    それ以外（未収束・λ ≤ 0・分離腕の複素対）は J を陽に組んで取り直す: 主腕（Hessian・対称）は eigvalsh の最大、
    分離腕（勾配流でない J）は固有値の実部の最大（スペクトル半径も記録）。``ok`` は λ_max が有限に取れた系列。"""
    pi = power_iteration(net, params, x_in, y, max_iter=max_iter, rtol=rtol)
    lam = np.asarray(pi["lam"], dtype=np.float64).copy()
    S = lam.shape[0]
    method = np.zeros(S, dtype=np.int64)                     # 0 = 冪乗法、1 = 陽な J の固有値
    rho = np.abs(lam)
    redo = np.flatnonzero(~(pi["converged_series"] & (lam > 0)))
    if redo.size:
        J = dense_jacobian_series(net, params, x_in, y, redo)
        for i, s in enumerate(redo):
            if symmetric:
                ev = np.linalg.eigvalsh(0.5 * (J[i] + J[i].T))
                lam[s], rho[s] = float(ev[-1]), float(np.abs(ev).max())
            else:
                ev = np.linalg.eigvals(J[i])
                lam[s], rho[s] = float(ev.real.max()), float(np.abs(ev).max())
            method[s] = 1
    meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pi.items() if k != "lam"}
    return dict(lam=lam, rho=rho, method=method, ok=np.isfinite(lam), power=meta, redone_series=redo.tolist())


def dense_jacobian_t(net, params, x_in, y) -> np.ndarray:
    """小さい系列の対照用: Jᵀ を基底ベクトルで陽に組む（系列 0 だけ・(n, n)）。"""
    W, b, v, c = [p.detach().clone().double().requires_grad_(True) for p in params]
    shapes = [p.shape[1:] for p in (W, b, v, c)]
    sizes = [int(np.prod(s)) if len(s) else 1 for s in shapes]
    n = sum(sizes)
    out = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        flat = torch.zeros(n, dtype=torch.float64)
        flat[j] = 1.0
        u, off = [], 0
        for p, s, size in zip((W, b, v, c), shapes, sizes):
            blk = torch.zeros_like(p)
            blk[0] = flat[off:off + size].reshape(s) if len(s) else flat[off]
            u.append(blk)
            off += size
        g = _surrogate_field(W, b, v, c, x_in.double(), y.double(), net)
        s_ = sum((gi * ui).sum() for gi, ui in zip(g, u))
        col = torch.autograd.grad(s_, (W, b, v, c))
        out[:, j] = np.concatenate([t[0].detach().reshape(-1).numpy() for t in col])
    return out


def state_from_ckpt(path: Path, *, activation: str | None = None, act_alpha: float | None = None) -> dict:
    """ckpt（``gate_dose.save_checkpoint_gate``）から 32 パターンの支持・教師・net を組み直す。"""
    ck = torch.load(path, weights_only=False, map_location="cpu")
    W = ck["net"]["W"]
    R, hdim, d = W.shape
    gen = torch.Generator().manual_seed(0)      # 形を作るためだけ（state は load で上書き）
    net = ChimeraMLPL(R, [hdim], d, gen, "cpu")
    net.load_state(ck["net"])
    act = str(activation or ck["activation"])
    alpha = float(ck["act_alpha"] if act_alpha is None else act_alpha)
    net.set_activation(act, alpha, "alpha_exp")
    flip = ck["env"]["flip_state"].double()                          # (R, f)
    P = X_FREE.shape[0]
    X = torch.cat([flip[None].expand(P, -1, -1),
                   torch.as_tensor(X_FREE)[:, None, :].expand(-1, R, -1)], dim=2)   # (P, R, m)
    mean = ck["running_mean"].double()
    x_in = X - mean[None]
    teacher = LTUTarget(R, d, 100, 0.7, gen, "cpu")
    teacher.load_state(ck["teacher"])
    y = teacher_f64(teacher, X)
    z = torch.einsum("rhd,prd->prh", W.double(), x_in) + ck["net"]["b"].double()
    return dict(ck=ck, net=net, X=X, x_in=x_in, y=y, z=z, mean=mean, step=int(ck["step"]),
                params=(ck["net"]["W"], ck["net"]["b"], ck["net"]["v"], ck["net"]["c"]),
                act=act, alpha=alpha, act_name=act_name_of(act))


def lam_full_from_ckpt(path: Path, *, max_iter: int = 2000, rtol: float = 1e-10) -> dict:
    S = state_from_ckpt(path)
    r = lam_max_full(S["net"], S["params"], S["x_in"], S["y"], symmetric=S["act_name"] not in AC.SPLIT,
                     max_iter=max_iter, rtol=rtol)
    r.update(step=S["step"], act=S["act"])
    return r


def s19_stability(log_path: Path, ckpt_path: Path) -> dict:
    """S19: w_free から再構成した λ_out（logs の経路）が、ckpt の状態から直接組んだ 2·E[φφᵀ] の固有値と
    一致すること。許容値は再構成の事前誤差（float32 の z̄ の丸め 2^-24·|z̄| ＋ float64 の 8·eps64）から導く
    （``lam_out_tolerance``）。変異対照: 対角（m_phi2）だけの近似が許容値を超えて外れること。"""
    S = state_from_ckpt(ckpt_path)
    act = S["act_name"]
    z_direct = S["z"].numpy()                                        # (P, R, U) float64
    phi_direct = AC.phi_np(act, z_direct)
    lam_direct = lam_out_of_phi(phi_direct)
    with np.load(log_path, allow_pickle=False) as zf:
        step = zf["step"]
        k = int(np.flatnonzero(step == S["step"])[0])
        j = int(np.flatnonzero(zf["layer1_w_free_step"] == S["step"])[0])
        zbar32 = zf["layer1_zbar"][k].astype(np.float64)               # (U,)
        w_free = zf["layer1_w_free"][j].astype(np.float64)             # (U, 5)
        seed = int(zf["seed"])
        zmax32 = zf["layer1_zmax"][k].astype(np.float64)
        m_phi2 = zf["layer1_m_phi2"][int(np.flatnonzero(zf["layer1_moment_step"] == S["step"])[0])]
    r = seed                                                           # 系列 = seed（runs の順）
    # (i) 式の検査: ckpt の float64 z̄ から再構成 → 誤差は float64 の丸めだけ
    zbar64 = z_direct[:, r].mean(0)
    z_rec64 = reconstruct_z(zbar64, w_free)
    bound64 = 8.0 * EPS64 * (np.abs(zbar64) + 0.5 * np.abs(w_free).sum(-1))
    lam_rec64 = lam_out_of_phi(AC.phi_np(act, z_rec64))
    tol64 = float(lam_out_tolerance(phi_direct[:, r], bound64))
    # (ii) logs の経路: float32 の z̄（+ 同じ w_free）
    z_rec32 = reconstruct_z(zbar32, w_free)
    bound32 = EPS32 / 2.0 * np.abs(zbar32) + bound64
    lam_rec32 = lam_out_of_phi(AC.phi_np(act, z_rec32))
    tol32 = float(lam_out_tolerance(phi_direct[:, r], bound32))
    # zmax の再構成（支持の形の検査: log の float32 zmax と一致）
    zmax_rec = z_rec32.max(0)
    zmax_bound = EPS32 / 2.0 * (np.abs(zmax32) + np.abs(zbar32)) + bound64
    # 変異対照: 対角だけの近似
    lam_diag = float(lam_out_diag_approx(phi_direct[:, r]))
    lam_diag_log = float(2.0 * m_phi2.astype(np.float64).max())
    return dict(step=S["step"], seed=seed, act=act,
                lam_direct=float(lam_direct[r]), lam_rec64=float(lam_rec64),
                lam_rec32=float(lam_rec32),
                err64=float(abs(lam_rec64 - lam_direct[r])), tol64=tol64,
                err32=float(abs(lam_rec32 - lam_direct[r])), tol32=tol32,
                spec_tol_8eps=float(8.0 * EPS64 * lam_direct[r]),
                zmax_err=float(np.abs(zmax_rec - zmax32).max()),
                zmax_bound_min=float(zmax_bound.min()),
                zmax_ok=bool(np.all(np.abs(zmax_rec - zmax32) <= zmax_bound)),
                control_diag=lam_diag, control_diag_err=float(abs(lam_diag - lam_direct[r])),
                control_diag_from_log_m_phi2=lam_diag_log,
                pass_=bool(abs(lam_rec64 - lam_direct[r]) <= tol64
                           and abs(lam_rec32 - lam_direct[r]) <= tol32
                           and np.all(np.abs(zmax_rec - zmax32) <= zmax_bound)),
                control_fails=bool(abs(lam_diag - lam_direct[r]) > tol32))


def _windows_for(total: int, n_tasks: int) -> tuple[dict, bool]:
    """登録の窓（config ``analysis.windows[total]``）。ログが短い（スモーク）ときは手元のタスクに切り詰め、
    ``clipped=True`` を返す。"""
    A = registered()["analysis"]
    W = A["windows"].get(str(int(total)))
    clipped = False
    if W is None:
        W = dict(lop=[max(1, n_tasks - 9), n_tasks], mech=[max(1, n_tasks - 49), n_tasks])
        clipped = True
    out = {}
    for key, (lo, hi) in W.items():
        lo2, hi2 = int(lo), int(hi)
        if hi2 > n_tasks:                      # 窓の長さを保ったまま末尾へ寄せる（スモークだけ）
            clipped = True
            hi2 = n_tasks
            lo2 = max(1, hi2 - (int(hi) - int(lo)))
        out[key] = [lo2, hi2]
    return out, clipped


def build_readout(outdir: Path, arm: str, *, write: bool = True,
                  power_iter: bool = True) -> dict:
    """§3.4・§8: ``readout_{ARM}.npz``（T = タスク終端の記録数、S = 系列数、U = 100）。

    凍結キー（§8）: ``fn_mom`` (T,S,8,4,3) f64・``occ`` (T,S,U,4) f32・``mob_band`` (T,S,U,4) f32・
    ``lam_out`` (T,S) f64・``lam_full`` (n_ckpt,S) f64・``relax_rate`` (S,) f64。
    補助キー: ``task`` (T,)・``pabs1`` (T,S,U) f32・``lam_full_step`` (n_ckpt,)・``relax_rate_alive`` (S,)・
    ``eos_out_median`` (S,)・``eos_full`` (n_ckpt,S)・``near_eos`` (S,) bool・``lr_used``・``activation``。
    z は ``reconstruct_z``（z̄ + w_free·(x − ½)）で 32 パターンを再構成し、8 関数の帯別モーメントは
    ``act_chimera_0913.fn_mom``、帯分率は ``occupancy``、自腕の帯別ゲートは ``mob_band``（分離腕は φ′_bwd）。"""
    bind_config()
    R = registered()
    A = R["analysis"]
    outdir = Path(outdir)
    row = table()[str(arm)]
    seeds = [int(v) for v in R["common_overrides"]["seeds"]]
    logs = []
    for seed in seeds:
        with np.load(outdir / "logs" / f"{arm}_seed{seed}.npz", allow_pickle=True) as z:
            logs.append({k: z[k] for k in ("step", "layer1_zbar", "layer1_w_free",
                                          "layer1_w_free_step", "layer1_dzbar", "layer1_zmax",
                                          "layer1_denom", "layer1_mob", "lr_used", "activation",
                                          "act_alpha", "unfit")})
    act = act_name_of(str(logs[0]["activation"]))
    lr = float(logs[0]["lr_used"])
    step = logs[0]["step"].astype(np.int64)
    for lg in logs[1:]:
        assert np.array_equal(lg["step"], step)
    period = int(A["task_period"])
    k_all = np.flatnonzero((step > 0) & (step % period == 0))
    tasks = (step[k_all] // period).astype(np.int64)
    T, S = int(k_all.size), len(logs)
    U = int(logs[0]["layer1_zbar"].shape[1])
    fn_mom = np.empty((T, S, 8, 4, 3), dtype=np.float64)
    occ = np.empty((T, S, U, 4), dtype=np.float32)
    mob_band = np.empty((T, S, U, 4), dtype=np.float32)
    pabs1 = np.empty((T, S, U), dtype=np.float32)
    lam_out = np.empty((T, S), dtype=np.float64)
    mob_check = np.zeros((T, S), dtype=np.float64)
    mob_viol = dict(n_cells=0, n_seam_explained=0, worst_unexplained=0.)
    dphi_fn = DPHI[act]
    seams = np.array([0., AC.LNA, AC.ZV])
    for s, lg in enumerate(logs):
        wf_index = {int(v): i for i, v in enumerate(lg["layer1_w_free_step"])}
        for t, k in enumerate(k_all):
            j = wf_index[int(step[k])]
            zb64 = lg["layer1_zbar"][k].astype(np.float64)
            wf64 = lg["layer1_w_free"][j].astype(np.float64)
            z = reconstruct_z(zb64, wf64)                                          # (32, U)
            zt = torch.as_tensor(z)
            fn_mom[t, s] = AC.fn_mom(zt).numpy()
            occ[t, s] = AC.occupancy(zt).numpy().astype(np.float32)
            mb = AC.mob_band(dphi_fn(zt), zt).numpy()
            mob_band[t, s] = mb.astype(np.float32)
            pabs1[t, s] = AC.pabs1(zt).numpy().astype(np.float32)
            lam_out[t, s] = lam_out_of_phi(AC.phi_np(act, z))
            diff = np.abs(mb.sum(1) - lg["layer1_mob"][k].astype(np.float64))        # (U,)
            mob_check[t, s] = float(diff.max())
            # 事前の上限（観測した差は使わない）: logs の z̄・w_free は float32 なので |δz_u| ≤ 2^-24·(|z̄| + ½Σ|w|)
            # （＋ float64 の再構成 8·eps64 の同じ項）、|φ″| ≤ 1 なので |δφ′| ≤ |δz|、記録器の mob の float32 化で 2^-24。
            # 継ぎ目（0・z_c・z_v）から δz 以内のパターンを持つユニットは φ′ が跳びうるので除く（φ′ の跳びは δz に比例しない）。
            mag = np.abs(zb64) + 0.5 * np.abs(wf64).sum(-1)
            dz = (EPS32 / 2.0 + 8.0 * EPS64) * mag
            bound = dz + EPS32 / 2.0 + 8.0 * EPS64
            bad = diff > bound
            if bad.any():
                near_seam = (np.abs(z[:, :, None] - seams[None, None, :]) <= dz[None, :, None]).any(axis=(0, 2))
                mob_viol["n_cells"] += int(bad.sum())
                mob_viol["n_seam_explained"] += int((bad & near_seam).sum())
                if (bad & ~near_seam).any():
                    mob_viol["worst_unexplained"] = max(mob_viol["worst_unexplained"],
                                                        float((diff - bound)[bad & ~near_seam].max()))
    mob_recon_pass = mob_viol["n_cells"] == mob_viol["n_seam_explained"]
    windows, clipped = _windows_for(int(row["total_steps"]), int(tasks.max()) if T else 0)
    from . import edge_law_analyze_0905 as ELA                  # §3.4: 生きているユニットの規則は edge_law の値そのもの
    if float(A["alive_denom"]) != float(ELA.ALIVE_DENOM):
        raise SanityError(f"config analysis.alive_denom {A['alive_denom']} != edge_law_analyze_0905.ALIVE_DENOM {ELA.ALIVE_DENOM}")
    relax = [relax_rate(step, lg["layer1_dzbar"], lg["layer1_zmax"], lg["layer1_denom"],
                        windows["mech"], period=period, alive_denom=float(A["alive_denom"]))
             for lg in logs]
    # 安定余裕（§2.4）: 系列ごとに lr·λ/2。窓は lop（491–500・退避枝 991–1000）
    eos_out = lr * lam_out / 2.0
    lo, hi = windows["lop"]
    in_lop = (tasks >= lo) & (tasks <= hi)
    eos_out_median = (np.median(eos_out[in_lop], axis=0) if in_lop.any()
                      else np.full(S, np.nan))
    # 全 Hessian（分離腕は代替勾配場の Jacobian）の λ_max: ログが届いた地平線までの登録 ckpt ごと（§2.4・S19）。
    # 行は登録 ckpt の順に固定し、ファイルが無い・冪乗法を回さない ckpt は NaN（黙って詰めない）。
    ck_steps = [int(cs) for cs in row["checkpoints"] if int(cs) <= int(step[-1])]
    n_ck = len(ck_steps)
    lam_full = np.full((n_ck, S), np.nan)
    lam_rho = np.full((n_ck, S), np.nan)
    lam_method = np.full((n_ck, S), -1, dtype=np.int64)
    lam_ok = np.zeros((n_ck, S), dtype=bool)
    pi_meta, missing_ckpts = [], []
    for i, cs in enumerate(ck_steps):
        p = outdir / "ckpts" / f"{arm}_step{int(cs)}.pt"
        if not p.exists():
            missing_ckpts.append(int(cs))
            continue
        if not power_iter:
            continue
        pi = lam_full_from_ckpt(p, max_iter=int(A["stability"]["power_iter_max"]),
                                rtol=float(A["stability"]["power_iter_rtol"]))
        lam_full[i], lam_rho[i], lam_method[i], lam_ok[i] = pi["lam"], pi["rho"], pi["method"], pi["ok"]
        pi_meta.append(dict(step=int(cs), power=pi["power"], redone_series=pi["redone_series"]))
    eos_full = lr * lam_full / 2.0
    thr = float(A["stability"]["eos_threshold"])
    near = (np.nan_to_num(eos_out_median, nan=0.0) >= thr)
    reached = int(step[-1]) == int(row["total_steps"])
    full_valid = bool(n_ck and ck_steps[-1] == int(row["total_steps"]) and lam_ok[-1].all())
    if reached and full_valid:
        near = near | (eos_full[-1] >= thr)
    # §2.4 の NEAR_EOS の後半（最終 ckpt の lr·λ_full/2）が無い本走は OK と書かない
    lam_status = ("LAM_FULL_OK" if full_valid else
                  "LAM_FULL_MISSING" if reached else "NOT_REGISTERED_HORIZON")
    out = dict(fn_mom=fn_mom, occ=occ, mob_band=mob_band, lam_out=lam_out, lam_full=lam_full,
               relax_rate=np.array([r["rate"] for r in relax], dtype=np.float64),
               task=tasks, pabs1=pabs1, lam_full_step=np.array(ck_steps, dtype=np.int64),
               relax_rate_alive=np.array([r["rate_alive"] for r in relax], dtype=np.float64),
               relax_n=np.array([r["n"] for r in relax], dtype=np.int64),
               eos_out_median=np.asarray(eos_out_median, dtype=np.float64), eos_full=eos_full,
               near_eos=near.astype(bool), lr_used=np.float64(lr), activation=np.array(act),
               arm=np.array(arm), seeds=np.array(seeds, dtype=np.int64),
               window_lop=np.array(windows["lop"], dtype=np.int64),
               window_mech=np.array(windows["mech"], dtype=np.int64),
               window_clipped=np.bool_(clipped),
               mob_recon_maxabs=mob_check, lam_full_ok=lam_ok, lam_full_rho=lam_rho, lam_full_method=lam_method,
               lam_full_status=np.array(lam_status))
    meta = dict(arm=arm, act=act, lr=lr, T=T, S=S, U=U, windows=windows, window_clipped=clipped,
                near_eos=near.tolist(), eos_out_median=np.asarray(eos_out_median).tolist(),
                eos_full=eos_full.tolist(), lam_full_step=ck_steps, lam_full_status=lam_status,
                lam_full_ok=lam_ok.tolist(), missing_ckpts=missing_ckpts, power_iteration=pi_meta,
                relax=relax, mob_recon_maxabs=float(mob_check.max()) if T else float("nan"),
                mob_recon=dict(pass_=bool(mob_recon_pass), **mob_viol,
                               bound='(2^-24 + 8 eps64)(|zbar|+0.5 sum|w_free|) + 2^-24 + 8 eps64; seam-adjacent units exempt'),
                status=("NEAR_EOS" if bool(near.any()) else ("LAM_FULL_MISSING" if lam_status == "LAM_FULL_MISSING" else "OK")))
    if write:
        path = outdir / f"readout_{arm}.npz"
        np.savez(path, **out)
        meta["path"] = str(path)
        meta["bytes"] = int(path.stat().st_size)
        (outdir / "arm_status" / f"{arm}_readout.json").parent.mkdir(parents=True, exist_ok=True)
        (outdir / "arm_status" / f"{arm}_readout.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    meta["arrays"] = out
    if not mob_recon_pass:                                       # 書いてから落とす（監査できるように）
        raise SanityError(f"{arm}: sum_k mob_band != recorder layer1_mob beyond the a-priori bound "
                          f"(not explained by seam proximity): {mob_viol}")
    return meta


# ---------------------------------------------------------------------------
# provenance（§8）: 腕ごとの provenance を 1 つに束ね、G1・参照 sha256・写しの検査を足す
# ---------------------------------------------------------------------------
def merge_provenance(outdir: Path, *, smoke: bool = False) -> dict:
    outdir = Path(outdir)
    fam = registered_outdir_family(outdir)
    arms, foreign = {}, []
    for p in sorted((outdir / "arm_status").glob("*_provenance.json")):
        name = p.name[:-len("_provenance.json")]
        if fam is not None and name in table() and row_family(table()[name]) != fam:
            foreign.append(name)                     # 本走と退避枝を 1 つの provenance に混ぜない（§2.4）
            continue
        arms[name] = json.loads(p.read_text(encoding="utf-8"))
    if foreign:
        raise SanityError(f"{outdir} ({fam}) holds provenance of the other row family: {foreign}")
    g1 = g1_scr(outdir, smoke=smoke) if any(a in arms for a in registered()["g1"]["pairs"]) else None
    prov = dict(run_id=EXPERIMENT, env=ENV, outdir=str(outdir), smoke=bool(smoke), row_family=fam,
                **git_state(), config=str(CONFIG), sha256=code_sha256(), env_asserted=EXPECTED_ENV,
                reference_npz_sha256=reference_sha256(), interpreter=sys.executable,
                torch=torch.__version__, numpy=np.__version__, machine=_machine(),
                s14=s14_scr(), g1=g1, arms=arms,
                written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (outdir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if g1 is not None:
        (outdir / "s_null.json").write_text(
            json.dumps(g1, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return prov


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dump(result: dict, outdir: Path, name: str) -> Path:
    path = Path(outdir) / "sanity" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"[{name}] {'PASS' if result.get('pass_') else 'FAIL'} -> {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="act_chimera_0913 SCR runner (edge_law_0905 host)")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="検査・短縮走行用。本走では使わない（seed はベクトル化されている）")
    parser.add_argument("--generator-offset", type=int, default=None,
                        help="G1-SCR の変異対照（--steps とだけ使える）")
    parser.add_argument("--g1", action="store_true", help="G1-SCR（chLR/chE を edge_law の null 腕と照合）")
    parser.add_argument("--g1-control", default=None, metavar="CTRL_OUTDIR",
                        help="G1-SCR の変異対照: generator_offset 1 の走の unfit が参照と不一致")
    parser.add_argument("--smoke", action="store_true", help="--g1 をスモーク（31 記録）の規則で")
    parser.add_argument("--ref-dir", default=None)
    parser.add_argument("--rss-probe", action="store_true")
    parser.add_argument("--readout", default=None, metavar="ARM")
    parser.add_argument("--no-power-iter", action="store_true")
    parser.add_argument("--provenance", action="store_true")
    parser.add_argument("--s14", action="store_true")
    args = parser.parse_args()
    env = assert_env()                               # §2.1: .venv の interpreter・torch 2.13.0+cu130・numpy 2.5.1
    print(f"[env] {env}", flush=True)
    bind_config()
    out = Path(args.outdir).resolve() if args.outdir else None
    main_dir = Path(ROOT) / registered()["output"]["dir"]

    if args.s14:
        res = s14_scr()
        print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        if not res["pass_"]:
            raise SanityError("S14 (SCR copy) failed")
        return
    if args.g1:
        res = g1_scr(out or main_dir, Path(args.ref_dir) if args.ref_dir else None,
                     smoke=bool(args.smoke))
        _dump(res, out or main_dir, "g1_scr")
        (Path(out or main_dir) / "s_null.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        for arm, v in res["arms"].items():
            for r in v["rows"]:
                print("  ", arm, json.dumps(r, ensure_ascii=False, default=str)[:300], flush=True)
        if not res["pass_"]:                          # §7: G1-SCR の失敗は SCR を止める（非 0 で終わる）
            raise SanityError("G1-SCR failed (see sanity/g1_scr.json)")
        return
    if args.g1_control:
        res = g1_smoke_control(Path(args.g1_control), Path(args.ref_dir) if args.ref_dir else None)
        _dump(res, out or Path(args.g1_control), "g1_scr_control")
        if not res["pass_"]:
            raise SanityError("G1-SCR mutation control did not fail as required (see sanity/g1_scr_control.json)")
        return
    if args.rss_probe:
        res = rss_probe(out or (Path(ROOT) / registered()["output"]["smoke_dir"]))
        _dump(res, out or (Path(ROOT) / registered()["output"]["smoke_dir"]), "rss_probe")
        if not res["pass_"]:
            raise SanityError("rss_probe failed (a child did not train, or no peak RSS; see sanity/rss_probe.json)")
        return
    if args.readout:
        meta = build_readout(out or main_dir, args.readout, power_iter=not args.no_power_iter)
        meta.pop("arrays", None)
        print(json.dumps(meta, indent=1, ensure_ascii=False, default=str)[:4000])
        return
    if args.provenance:
        merge_provenance(out or main_dir, smoke=bool(args.smoke))
        print(f"provenance -> {(out or main_dir) / 'provenance.json'}")
        return
    if args.arm:
        if args.arm not in table():
            parser.error(f"--arm must be one of {arm_order()}")
        cfg = build_cfg()
        require_omp(cfg)
        run_single_arm(args.arm, args.steps, out, args.seeds, cfg=cfg,
                       generator_offset=args.generator_offset)
        return
    parser.error("nothing to do: pass --arm / --g1 / --g1-control / --rss-probe / --readout / "
                 "--provenance / --s14")


if __name__ == "__main__":
    main()
