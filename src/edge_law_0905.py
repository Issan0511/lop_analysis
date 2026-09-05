"""edge_law_0905 — 命題 1–5（前活性の釣り合いは支持の上端で決まるか）の runner。

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 --sanity
    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 --arm LRnull_1216
    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 --launch-plan

事前登録 spec: ``specs/spec_edge_law_0905.md``。腕表・フック・記録列・解析定数の正本は
``configs/edge_law_0905.yaml``（実装より先に commit 済み）。宿主は ``gate_dial_0902``
（1 層・幅 100・オラクル用量 12.16 固定・10 seed・lr 0.01）で、``weird_act_0903`` と同じ
流儀で写す（``_run_arm`` の写し＋recorder の列追加＋init フック＋v 凍結＋full-batch）。
**既存モジュールは 1 行も変えない**（spec §3.1・§3.3）。

宿主の ``validate_config`` / ``_s_dial`` / ``_geometry`` は呼ばない（ラベル辞書が
{relu, leaky_relu, elu} 固定で KeyError になる・spec §3.1）。

S-copy（spec §5）の都合で、``_run_arm_edge`` と ``train_arm_edge`` は宿主の関数本体の
**逐語の写し＋登録済みの挿入行だけ**でできている。フック・recorder・学習ループ・
書き出しの差し替えは、宿主の名前を関数内の局所名に束縛し直す 4 行の挿入で行う
（``WeirdRecorder = EdgeRecorder`` 等）。こうすると宿主の行が 1 行も書き換わらず、
difflib の opcodes が ``equal`` と登録済み ``insert`` だけになる。フックの無い腕では
``_train_fn`` が宿主の ``train_arm_gate`` **そのもの**（同一オブジェクト）を返すので、
オンライン経路は本モジュールを通してもバイト単位で不変である（S-null がこれを実測する）。
"""
from __future__ import annotations

import argparse
import copy
import difflib
import inspect
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import gate_dose as _host_gate
from . import weird_act_0903 as _host_weird
from .common import ROOT, load_config, pick_device
from .dose_const_5m import _refresh_fixed_offset
# S-mirror の列別パリティ規則は**解析側の 1 実装だけ**を使う（runner と解析で
# 別々に書くと零の符号の扱いが割れる — それが批評ラウンド 1 の SYM1/SYM2）。
from .edge_law_analyze_0905 import mirror_parity as _mirror_rule
from .elu_swamp import exact_layer_record_elu, grads_centered_elu
from .gate_dial_0902 import (CONFIG as HOST_CONFIG, SanityError, _activation,
                             _arm, _arm_status_path, _load_divergence_status,
                             setup_arm_dial, unit_extra_record,
                             write_arm_logs_dial)
from .gate_dose import (IDENTITY_TOL, SIGMA_TOL, forward_gate,
                        save_checkpoint_gate, train_arm_gate)
from .mlp2_phase0 import _sha_array, identity_sanity_pass, require_omp
from .mlp2_phase0b import _complete_arm_logs, _window_indices
from .mlp2_phase1 import (NUMERIC_DIVERGENCE, NumericDivergenceError,
                          _seed_state_hashes_p1)
from .nets import VecMLPL
from .ratchet_log import full_support_ro
from .weird_act_0903 import WeirdRecorder, unit_zmin_record

EXPERIMENT = "edge_law_0905"
CONFIG = Path(ROOT) / "configs" / "edge_law_0905.yaml"
PERIOD = 10_000
# 自由 5 ビットは envs.SCREnv が `cat([flip_state (f=15), rnd (m-f=5)], dim=1)` を返す
# ので入力の**末尾 5 列**（インデックス 15..19）。`ratchet_log.full_support_ro` も同じ
# 順で 32 パターンを並べる。S-support が `半幅 == 0.5*sum|w_free|` で実測する。
FREE_SLICE = slice(15, 20)
N_FREE = 5
MOMENT_KEYS = ("m_phi2", "m_dphi2", "m_phidphi", "m_dphiddphi")
_MOMENT_DENSE_TASKS = 20           # config: record.moments_dense_last_tasks
# spec §4.1-b（S-mirror の列別パリティ）
MIRROR_SIGN_FLIP = ("layer1_zbar", "layer1_dzbar", "layer1_zmean",
                    "layer1_v_unit", "layer1_M", "layer1_B")
MIRROR_INVARIANT = ("layer1_w_norm", "layer1_denom", "layer1_mob",
                    "layer1_absmob")
# S-null / S-mirror の参照腕（committed・再走しない）
S_NULL_REF = {"LRnull_1216": "LR_1216", "Enull_1216": "E_1216"}
S_MIRROR_REF = {"FLn_1216": "LR_1216"}
# S-null で比べない列（原理的に一致しない・spec §5）
S_NULL_SKIP = ("arm", "run_id", "state_hash_final")
C_CLOSED_FORM = 11.497681          # E|x_c|^2 + 1 = 3.041^2 + 5*0.25 + 1（spec §3.5）
RSS_PARALLEL_CAP = 20              # 並列数の上限（spec §5 RSS 行）
RSS_HEADROOM = 1.5                 # 並列数 = min(20, floor(free / (1.5*peak)))


# ---------------------------------------------------------------------------
# Config（spec §3.1）
# ---------------------------------------------------------------------------
def registered(config: str | Path | None = None) -> dict:
    """``configs/edge_law_0905.yaml`` を読むだけ（正本はこの config）。"""
    return load_config(str(config or CONFIG))


def arm_table(cfg_edge: dict | None = None) -> dict:
    """腕名 → 腕行（config の逐語）。hook / total_steps / checkpoints はここだけに置く。"""
    E = cfg_edge if cfg_edge is not None else registered()
    out = {}
    for row in E["arms"]:
        out[str(row["name"])] = dict(
            name=str(row["name"]), family=str(row["family"]),
            activation=str(row["activation"]), dial=float(row["dial"]),
            u_fr=(None if row.get("u_fr") is None else float(row["u_fr"])),
            hook=(None if row.get("hook") is None else dict(row["hook"])),
            total_steps=int(row["total_steps"]),
            checkpoints=[int(v) for v in (row.get("checkpoints") or [])])
    return out


_TABLE: dict | None = None


def table() -> dict:
    """腕表のモジュール内キャッシュ（config は実行中に変わらない）。"""
    global _TABLE
    if _TABLE is None:
        _TABLE = arm_table()
    return _TABLE


def arm_order() -> list[str]:
    return list(table())


def _hook_of(arm: str) -> dict | None:
    return table()[str(arm)]["hook"]


def build_cfg(config: str | Path | None = None) -> dict:
    """宿主 config を写し、``activation`` マップと腕表だけを足す（p3_runs_0902 の流儀）。

    宿主の ``validate_config`` は 14 腕を逐語照合するので**通さない**（spec §3.1）。
    ``common_overrides`` は config の逐語。腕ごとの ``total_steps`` / ``checkpoints`` /
    ``hook`` は cfg に入れず ``table()`` に置く（宿主の ``_run_arm`` が読む共通キーは
    ``run_single_arm`` が腕ごとに書き換える）。
    """
    E = registered(config)
    c = copy.deepcopy(load_config(str(HOST_CONFIG)))
    ov = E["common_overrides"]
    c["common"]["lr_main"] = float(ov["lr_main"])
    c["common"]["seeds"] = [int(v) for v in ov["seeds"]]
    c["common"]["generator_offset"] = int(ov["generator_offset"])
    c["sanity"]["omp_num_threads"] = int(ov["omp_num_threads"])
    for name, block in E["activation"].items():
        if not isinstance(block, dict):
            continue                      # `..._is_s_limit_only` などの真偽値フラグ
        c["activation"].setdefault(name, {"name": str(block["name"])})
    have = {a["name"] for a in c["arms"]}
    template = dict(E["arm_template"])
    for row in E["arms"]:
        name = str(row["name"])
        if name in have:
            raise ValueError(f"{name} already exists in the host config")
        c["arms"].append(dict(template, name=name, family=str(row["family"]),
                              activation=str(row["activation"]),
                              dial=float(row["dial"]),
                              u_fr=(None if row.get("u_fr") is None
                                    else float(row["u_fr"]))))
    return c


# ---------------------------------------------------------------------------
# init フック / 実行フラグ（spec §3.3・config の `hooks:` を逐語で実装）
# ---------------------------------------------------------------------------
def _check_aliases(net: VecMLPL, arm: str) -> None:
    """S-hook-inplace: 別名 ``net.W`` / ``net.b`` が生きたままであること。"""
    if not (net.W is net.Ws[0] and net.b is net.bs[0]):
        raise SanityError(f"{arm}: net.W/net.b aliases were rebound by the hook")


def _apply_hook(st: dict, hook: dict | None, arm: str = "") -> dict:
    """``setup_arm_dial`` 直後の状態へ init フックを **in-place** で当てる。

    すべて in-place（``neg_`` / ``add_`` / ``mul_`` / ``div_``）で書く。
    ``net.Ws[0] = -net.Ws[0]`` の形だと ``VecMLPL.__init__`` が張った別名
    ``net.W`` / ``net.b`` が古いテンソルを指したまま残る（spec §3.3）。乱数は
    一切消費しない（S-stream）。payload 列 5 つは既定値をここで必ず書く。
    """
    net = st["net"]
    st["init_hook"] = ""
    st["init_hook_arg"] = float("nan")
    st["lr_used"] = float(st["runs"][0]["lr"])
    st["freeze_v"] = False
    st["batch_mode"] = "online"
    if hook is None:
        _check_aliases(net, arm)
        return st
    kind = str(hook["type"])
    value = hook.get("value")
    st["init_hook"] = kind
    st["init_hook_arg"] = float("nan") if value is None else float(value)
    with torch.no_grad():
        if kind == "negate":
            net.Ws[0].neg_()
            net.bs[0].neg_()
            net.v.neg_()                      # c は不変（spec §3.3）
        elif kind == "b_offset":
            net.bs[0].add_(float(value))
        elif kind == "scale":
            s = float(value)
            net.Ws[0].mul_(s)
            net.bs[0].mul_(s)
            net.v.div_(s)
        elif kind == "lr":
            eta = float(value)
            st["lr"] = torch.full_like(st["lr"], eta)
            for run in st["runs"]:
                run["lr"] = eta
            st["lr_used"] = eta
        elif kind == "v_freeze":
            net.v.mul_(float(value))
            st["freeze_v"] = True
        elif kind == "full_batch":
            st["batch_mode"] = "full32"
        else:
            raise ValueError(f"unknown init hook {kind!r}")
    _check_aliases(net, arm)
    return st


def _setup_with_hook(cfg: dict, arm_cfg: dict, device: str,
                     hook: dict | None = None) -> dict:
    """``setup_arm_dial`` ＋ フック。``hook=None`` は宿主と bit 一致（S-hook-noop）。"""
    st = setup_arm_dial(cfg, arm_cfg, device)
    return _apply_hook(st, hook, str(arm_cfg.get("name", "")))


# ---------------------------------------------------------------------------
# 記録列（spec §3.4）
# ---------------------------------------------------------------------------
def unit_moment_record(st: dict) -> dict:
    """支持窓平均 ``E_W[phi^2] / E_W[phi'^2] / E_W[phi phi'] / E_W[phi' phi'']``。

    ``gate_dial_0902.unit_extra_record`` の ``z`` の作り方を逐語で真似る（32 点の
    厳密支持・float64・中心化フラグと ``layer_means`` の扱いまで同じ）。したがって
    モーメントは ``zmax`` / ``zmin`` と厳密に同じ ``z`` の上の量である。学習状態は
    読むだけで書き換えない（``full_support_ro`` は RNG を消費しない）。
    ``act_curv`` は未登録の活性化で ``NotImplementedError`` を上げる（黙って ELU の
    曲率に落ちない・spec §3.2）。
    """
    net = st["net"]
    flags = st.get("centered_layers") or [False] * len(net.Ws)
    means = st.get("layer_means") or [None] * len(net.Ws)
    with torch.no_grad():
        cur = full_support_ro(st["env"]).double()
        if flags[0]:
            cur = cur - means[0].double()[None]
        W, b = net.Ws[0].double(), net.bs[0].double()
        z = torch.einsum("rhd,prd->prh", W, cur) + b
        phi = net.act_fn(z)
        dphi = net.act_grad(z, phi)
        ddphi = net.act_curv(z)
        return dict(m_phi2=(phi * phi).mean(dim=0),
                    m_dphi2=(dphi * dphi).mean(dim=0),
                    m_phidphi=(phi * dphi).mean(dim=0),
                    m_dphiddphi=(dphi * ddphi).mean(dim=0))


def unit_w_free_record(st: dict) -> torch.Tensor:
    """自由 5 ビットの重み ``W[:, :, 15:20]``（支持の厳密形状・spec §3.4）。"""
    return st["net"].Ws[0][:, :, FREE_SLICE]


class EdgeRecorder(WeirdRecorder):
    """``WeirdRecorder``（= 宿主 5 列 ＋ zmin）に spec §3.4 の 3 種を足す。

    - ``layer1_w_free`` (rec, R, h, 5): タスク終端（10,000 step ごと・step 0 込み）
    - 4 モーメント列 (rec, R, h): タスク終端 ＋ **末尾 20 タスクは 1000 step ごと**
    - それぞれの記録点は ``layer1_w_free_step`` / ``layer1_moment_step`` に書く

    どちらの窓も ``probes[-1]``（= 総 step）から決まるので、署名は宿主の recorder と
    同じ ``(steps, st)`` のままにしてある（S-copy の挿入行が 1 行で済む）。
    """

    def __init__(self, steps: list[int], st: dict, *, record_units: bool = True):
        super().__init__(steps, st, record_units=record_units)
        self.dense_tasks = int(_MOMENT_DENSE_TASKS)
        total = int(self.steps[-1]) if len(self.steps) else 0
        task_end = (self.steps % PERIOD == 0)
        dense = self.steps > (total - self.dense_tasks * PERIOD)
        self.w_free_steps = self.steps[task_end].astype(np.int64)
        self.moment_steps = self.steps[task_end | dense].astype(np.int64)
        self.w_free_index = {int(v): i for i, v in enumerate(self.w_free_steps)}
        self.moment_index = {int(v): i for i, v in enumerate(self.moment_steps)}
        if self.record_units:
            runs, width = st["R"], st["hidden"][0]
            self.w_free = np.empty((len(self.w_free_steps), runs, width, N_FREE),
                                   dtype=np.float32)
            self.moments = {key: np.empty((len(self.moment_steps), runs, width),
                                          dtype=np.float32)
                            for key in MOMENT_KEYS}
        else:
            self.w_free = np.empty((0, 0, 0, N_FREE), dtype=np.float32)
            self.moments = {key: np.empty((0, 0, 0), dtype=np.float32)
                            for key in MOMENT_KEYS}

    def __call__(self, st: dict, step: int) -> None:
        super().__call__(st, step)
        if not self.record_units:
            return
        if self.index.get(int(step)) is None:
            return
        j = self.w_free_index.get(int(step))
        if j is not None:
            self.w_free[j] = (unit_w_free_record(st).detach().cpu().numpy()
                              .astype(np.float32))
        k = self.moment_index.get(int(step))
        if k is not None:
            moments = unit_moment_record(st)
            for key in MOMENT_KEYS:
                self.moments[key][k] = (moments[key].detach().cpu().numpy()
                                        .astype(np.float32))



def write_arm_logs_edge(outdir: Path, arm: str, st: dict,
                        rec: EdgeRecorder) -> list[Path]:
    """``gate_dial_0902.write_arm_logs_dial`` の写し ＋ spec §3.4 の新列と payload。

    既存列は 1 列も変えない・消さない。共通列がバイト同一であることは
    ``test_edge_law_runner_0905`` が宿主の書き出しと突き合わせて実測する。
    """
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ri, run in enumerate(st["runs"]):
        seed = int(run["seed"])
        payload = dict(
            step=rec.steps, run_id=np.array(run["run_id"]), arm=np.array(arm),
            seed=np.int64(seed), activation=np.array(st["activation"]),
            act_alpha=np.float64(st["act_alpha"]),
            family=np.array(st.get("family", "")),
            task_period=np.int64(run["period"]),
            target_mu_norm=np.float64(np.nan if st.get("target_mu_norm") is None
                                      else st["target_mu_norm"]),
            target_dose=np.float64(np.nan if st.get("target_dose") is None
                                   else st["target_dose"]),
            state_hash_final=np.array(json.dumps(
                _seed_state_hashes_p1(st, ri), sort_keys=True)),
            state_hash_1m=np.array(json.dumps(
                rec.state_hash_1m.get(seed, {}), sort_keys=True)))
        for key, value in rec.run.items():
            payload[key] = value[:, ri]
        payload["flip_state"] = rec.flip_state[:, ri]
        for key, value in rec.extra.items():
            payload[key] = value[:, ri]
        for li, layer in enumerate(rec.layers, start=1):
            for key, value in layer.items():
                payload[f"layer{li}_{key}"] = value[:, ri]
        for key, value in rec.unit.items():
            payload[f"layer1_{key}"] = value[:, ri]
        # --- spec §3.4 の新列（独自の step 列を連れる）------------------------
        payload["layer1_w_free"] = rec.w_free[:, ri]
        payload["layer1_w_free_step"] = rec.w_free_steps
        for key in MOMENT_KEYS:
            payload[f"layer1_{key}"] = rec.moments[key][:, ri]
        payload["layer1_moment_step"] = rec.moment_steps
        # --- payload スカラ（spec §3.3・解析の契約）----------------------------
        payload["init_hook"] = np.array(str(st.get("init_hook", "")))
        payload["init_hook_arg"] = np.float64(st.get("init_hook_arg", np.nan))
        payload["lr_used"] = np.float64(st.get("lr_used", np.nan))
        payload["freeze_v"] = np.bool_(bool(st.get("freeze_v", False)))
        payload["batch_mode"] = np.array(str(st.get("batch_mode", "online")))
        path = logdir / f"{arm}_seed{seed}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# full-batch（spec §3.3・32 パターン厳密勾配）。オンライン経路には一切触らない。
# ---------------------------------------------------------------------------
def forward_gate_batch(st: dict, X: torch.Tensor):
    """``gate_dose.forward_gate`` の先頭バッチ次元つき版（``x: [N,R,d]``）。

    用量固定腕（``target_mu_norm`` あり）だけを想定し、オンライン版と同じく
    ``_refresh_fixed_offset`` を呼んでから float32 境界でオフセットを引く。
    """
    fixed = st.get("target_mu_norm") is not None
    if not fixed:
        raise ValueError("full_batch is registered only for the fixed-dose arms")
    _refresh_fixed_offset(st)
    net, cur = st["net"], X
    inputs, pres, acts = [], [], []
    for li, (W, b) in enumerate(zip(net.Ws, net.bs)):
        mean = st["layer_means"][li]
        cur_in = cur - mean.to(cur.dtype)
        pre = torch.einsum("rhd,nrd->nrh", W, cur_in) + b
        cur = net.act_fn(pre)
        inputs.append(cur_in)
        pres.append(pre)
        acts.append(cur)
    yhat = (acts[-1] * net.v).sum(dim=-1) + net.c
    return inputs, pres, acts, yhat


def grads_centered_elu_batch(net, inputs, pres, acts, delta):
    """``elu_swamp.grads_centered_elu`` のバッチ平均版（32 パターン厳密勾配）。

    式は逐語の写しで、縮約が ``mean(dim=0)``（＝ 32 パターンの一様平均）になる点
    だけが違う。S-fb が「32 個の単標本勾配の平均と相対 1e-6 一致」を実測する。
    """
    d2 = 2.0 * delta                                        # [N,R]
    gv = (d2[..., None] * acts[-1]).mean(dim=0)
    gc = d2.mean(dim=0)
    dz = d2[..., None] * net.v * net.act_grad(pres[-1], acts[-1])
    gWs: list[torch.Tensor | None] = [None] * net.L
    gbs: list[torch.Tensor | None] = [None] * net.L
    for layer in range(net.L - 1, -1, -1):
        gbs[layer] = dz.mean(dim=0)
        gWs[layer] = (dz[..., None] * inputs[layer][:, :, None, :]).mean(dim=0)
        if layer:
            dz = (torch.einsum("rhi,nrh->nri", net.Ws[layer], dz)
                  * net.act_grad(pres[layer - 1], acts[layer - 1]))
    return gWs, gbs, gv, gc


def train_arm_full_batch(st: dict, recorder, probe_steps, total: int,
                         outdir: Path, checkpoints, stream_hook=None) -> float:
    """``gate_dose.train_arm_gate`` の full-batch 版（**別関数**・spec §3.3）。

    オンライン経路（宿主の ``train_arm_gate``）は 1 行も変えないので、写しではなく
    新規に書く。``env.step()`` は毎ステップ**必ず呼ぶ**（返り値は捨てる）。こうすると
    flip の乱数消費と時刻がオンライン腕と厳密に一致し、`FBLR_1216` は `LRnull_1216`
    と同じ flip 軌跡の上でノイズだけが消えた腕になる。
    """
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            save_checkpoint_gate(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()                  # 乱数消費と flip 時刻をオンラインと揃える
        X = full_support_ro(env)        # [32,R,m]（読み取り専用・env を進めない）
        Y = teacher(X)
        if stream_hook is not None:
            stream_hook(step, x, X)
        inputs, pres, acts, yhat = forward_gate_batch(st, X)
        grads = grads_centered_elu_batch(net, inputs, pres, acts, yhat - Y)
        if st.get("freeze_v"):
            grads = (grads[0], grads[1], torch.zeros_like(grads[2]), grads[3])
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_gate(st, st["arm"], total, outdir)
    return time.time() - started


# ---------------------------------------------------------------------------
# 学習ループ（v 凍結つき）— `gate_dose.train_arm_gate` の逐語の写し ＋ 登録挿入 2 行
# ---------------------------------------------------------------------------
# 登録挿入（S-copy が逐語照合する）。gv を 0 にしてから `sgd_step_layers` に渡す。
# `v -= lr*0.0` は有限な v に対して v と**バイト同一**（-0.0 も -0.0 のまま）で、
# W・b・c の更新には gv が一切入らないので、v 以外のテンソルは凍結なしと同じ演算・
# 同じ順序で進む。走った後に v を書き戻す実装（clone + copy_）だと、書き戻しの前に
# 一度 v が動くので `sgd_step_layers` の中の演算が同一である保証が弱くなる。
TRAIN_INSERTS = (
    'if st.get("freeze_v"):',
    'grads = (grads[0], grads[1], torch.zeros_like(grads[2]), grads[3])',
)


def train_arm_edge(st: dict, recorder, probe_steps, total: int, outdir: Path,
                   checkpoints, stream_hook=None) -> float:
    """``gate_dose.train_arm_gate`` の写し ＋ v 凍結の 2 行（S-copy が検算）。"""
    probe_set = {int(v) for v in probe_steps}
    checkpoint_set = {int(v) for v in checkpoints}
    net, env, teacher = st["net"], st["env"], st["teacher"]
    started = time.time()
    for step in range(total):
        if step in checkpoint_set:
            save_checkpoint_gate(st, st["arm"], step, outdir)
        if step in probe_set:
            recorder(st, step)
        x = env.step()
        y = teacher(x)
        if stream_hook is not None:
            stream_hook(step, x, y)
        inputs, pres, acts, yhat = forward_gate(st, x)
        grads = grads_centered_elu(net, inputs, pres, acts, yhat - y)
        if st.get("freeze_v"):
            grads = (grads[0], grads[1], torch.zeros_like(grads[2]), grads[3])
        net.sgd_step_layers(st["lr"], *grads)
    if total in probe_set:
        recorder(st, total)
    if total in checkpoint_set:
        save_checkpoint_gate(st, st["arm"], total, outdir)
    return time.time() - started


def _train_fn(st: dict):
    """腕に応じた学習ループ。フックの無い腕は**宿主の関数そのもの**を返す。

    ``_train_fn(st) is gate_dose.train_arm_gate`` が 30 腕中 26 腕で真であること
    （＝オンライン経路が本モジュールを通しても字義どおり不変であること）は
    ``test_edge_law_runner_0905`` が全腕について実測する。
    """
    if str(st.get("batch_mode", "online")) == "full32":
        return train_arm_full_batch
    if bool(st.get("freeze_v", False)):
        return train_arm_edge
    return train_arm_gate


# ---------------------------------------------------------------------------
# Runner — `weird_act_0903._run_arm_weird` の逐語の写し ＋ 登録挿入 4 行（S-copy）
# ---------------------------------------------------------------------------
# 挿入行は 2 か所。1 か所目は `setup_arm_dial` の直後（＝恒等式検査より前）で、
# spec §3.3 が要求する「フックは setup 直後・恒等式検査より前」を満たす。2 か所目は
# recorder を作る直前で、宿主の名前 3 つを局所に束縛し直す（宿主の行を 1 行も
# 書き換えないための手段。以降の宿主の行はそのまま新しい実装を呼ぶ）。
RUN_INSERTS = (
    ('_apply_hook(st, _hook_of(arm), arm)',),
    ('WeirdRecorder = EdgeRecorder',
     'train_arm_gate = _train_fn(st)',
     'write_arm_logs_dial = write_arm_logs_edge'),
)


def _run_arm_edge(cfg: dict, arm: str, device: str, outdir: Path,
                  seeds: list[int], total: int) -> dict:
    """``weird_act_0903._run_arm_weird`` の写し（差は登録挿入 4 行だけ）。"""
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


def run_single_arm(arm: str, steps: int | None = None,
                   outdir: Path | None = None, seeds: list[int] | None = None,
                   cfg: dict | None = None) -> dict:
    """腕プロセス並列の投入単位。1 腕だけ走らせて logs を置いて終わる。"""
    row = table()[str(arm)]
    cfg = build_cfg() if cfg is None else cfg
    require_omp(cfg)
    total = int(steps) if steps else int(row["total_steps"])
    cfg = copy.deepcopy(cfg)
    cfg["common"]["total_steps"] = total
    cfg["common"]["checkpoints"] = list(row["checkpoints"])
    out = Path(outdir) if outdir is not None else (
        Path(ROOT) / registered()["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    use = [int(v) for v in (seeds if seeds is not None else cfg["common"]["seeds"])]
    if use != [int(v) for v in cfg["common"]["seeds"]]:
        # seed はベクトル化されている（1 本の系列から [R,...] を一度に引く）ので、
        # 部分集合は 10 seed 走と**別の入力列**になる（s_seed_split_note）。
        if steps is None:
            # 登録の地平線で回そうとしている＝本走。ここで止める（警告だけだと
            # 30 腕のどれかが黙って別のストリームで走りうる）。
            raise SanityError(
                f"{arm}: a seed subset {use} is not a valid registered run — "
                f"the input stream differs from the 10-seed run from step 1 "
                f"(s_seed_split_note). Pass --steps for a shortened check run.")
        print(f"[{arm}] WARNING: running a seed subset {use}; the input stream "
              f"differs from the registered 10-seed run — checks only",
              flush=True)
    every = int(cfg["common"]["lop_every"])
    if _load_divergence_status(out, arm, use, total, every) is not None:
        print(f"[{arm}] saved {NUMERIC_DIVERGENCE}; nothing to do", flush=True)
        return dict(status=NUMERIC_DIVERGENCE)
    if _complete_arm_logs(out, arm, use, total, every):
        print(f"[{arm}] complete logs found; nothing to do", flush=True)
        return dict(status="COMPLETE", resumed=True)
    started = time.time()
    result = _run_arm_edge(cfg, arm, "cpu", out, use, total)
    status = dict(result)
    status.update(arm=arm, total_steps=total, seeds=use,
                  wall_sec=time.time() - started, hook=row["hook"],
                  git_head=_git_head())
    status.pop("sanity", None)
    done = out / "arm_status" / f"{arm}_done.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return result


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# 検査（spec §5）。走らせる前に PASS が要るものはすべて関数として置く。
# ---------------------------------------------------------------------------
def _body(fn, sig_end: str = "-> dict:") -> list[str]:
    """``weird_act_0903._s_copy`` の ``body`` と同じ正規化（署名・空行・注釈を落とす）。

    行頭の字下げは落ちるので、字下げだけの違いは捕まらない。その穴は
    ``_train_fn`` が宿主の関数**そのもの**を返すことを別に検査して塞ぐ。
    """
    lines = inspect.getsource(fn).splitlines()
    out, in_signature = [], True
    for line in lines:
        s = line.strip()
        if in_signature:
            if s.endswith(sig_end) or (s.endswith(":") and "->" in s):
                in_signature = False
            continue
        if not s or s.startswith("#") or s.startswith('"""'):
            continue
        out.append(s.split("#")[0].rstrip())
    return out


def _copy_opcodes(host, mine, registered_inserts) -> dict:
    theirs, ours = _body(host), _body(mine)
    ops = difflib.SequenceMatcher(None, theirs, ours, autojunk=False).get_opcodes()
    inserts, bad = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if tag == "insert":
            inserts.append(tuple(ours[j1:j2]))
            continue
        bad.append(dict(tag=tag, host=theirs[i1:i2], mine=ours[j1:j2]))
    want = [tuple(block) for block in registered_inserts]
    return dict(pass_=bool(not bad and inserts == want),
                host=host.__name__, mine=mine.__name__,
                host_lines=len(theirs), mine_lines=len(ours),
                inserted=[list(b) for b in inserts],
                registered=[list(b) for b in want],
                unregistered_opcodes=bad)


def s_copy() -> dict:
    """S-copy: 写した 2 つのループ本体の差が登録挿入行だけであること（spec §5）。"""
    run = _copy_opcodes(_host_weird._run_arm_weird, _run_arm_edge, RUN_INSERTS)
    train = _copy_opcodes(_host_gate.train_arm_gate, train_arm_edge,
                          (TRAIN_INSERTS,))
    return dict(pass_=bool(run["pass_"] and train["pass_"]),
                run_arm=run, train_arm=train)


def _probe_state(arm: str, device: str = "cpu", *, hook: bool = False,
                 cfg: dict | None = None) -> dict:
    cfg = build_cfg() if cfg is None else cfg
    blk = _arm(cfg, arm)
    if hook:
        return _setup_with_hook(cfg, blk, device, _hook_of(arm))
    return setup_arm_dial(cfg, blk, device)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().numpy().tobytes()


def _state_fingerprint(st: dict) -> dict:
    net = st["net"]
    return {
        "W": _tensor_bytes(net.Ws[0]), "b": _tensor_bytes(net.bs[0]),
        "v": _tensor_bytes(net.v), "c": _tensor_bytes(net.c),
        "running_mean": _tensor_bytes(st["running_mean"]),
        "layer_means0": _tensor_bytes(st["layer_means"][0]),
        "flip_state": _tensor_bytes(st["env"].flip_state),
    }


def s_hook_inplace(cfg: dict | None = None) -> dict:
    """S-hook-inplace: フック適用後も ``net.W is net.Ws[0]`` / ``net.b is net.bs[0]``。

    「別名が生きている」だけでなく「フックが実際に効いている」ことも要求する
    （効いていない恒等フックなら別名は自明に生きているので空虚になる）。
    """
    cfg = build_cfg() if cfg is None else cfg
    rows, ok = [], True
    for arm, row in table().items():
        if row["hook"] is None:
            continue
        st = setup_arm_dial(cfg, _arm(cfg, arm), "cpu")
        before = _state_fingerprint(st)
        lr_before = _tensor_bytes(st["lr"])
        net = st["net"]
        # 解析的な期待値を作るための控え（**フック適用前**のバイト）。
        w0 = net.Ws[0].detach().clone()
        b0 = net.bs[0].detach().clone()
        v0 = net.v.detach().clone()
        c0 = net.c.detach().clone()
        _apply_hook(st, row["hook"], arm)
        alias = bool(net.W is net.Ws[0] and net.b is net.bs[0])
        after = _state_fingerprint(st)
        changed = sorted(k for k in before if before[k] != after[k])
        kind = str(row["hook"]["type"])
        value = row["hook"].get("value")

        def _same(t: torch.Tensor, want: torch.Tensor) -> bool:
            """バイト比較（``torch.equal`` は ±0.0 に盲・本プロジェクトの落とし穴）。"""
            return bool(_tensor_bytes(t) == _tensor_bytes(want))

        # 「どのテンソルが動いたか」だけでなく「**いくつになったか**」を見る
        # （符号を逆に足す・0 を掛ける、といった変異はパターンだけでは通ってしまう）。
        if kind == "negate":
            exact = bool(_same(net.Ws[0], torch.negative(w0))
                         and _same(net.bs[0], torch.negative(b0))
                         and _same(net.v, torch.negative(v0))
                         and _same(net.c, c0))
            effective = bool(set(changed) == {"W", "b", "v"} and exact)
        elif kind == "b_offset":
            exact = bool(_same(net.bs[0], b0 + float(value))
                         and _same(net.Ws[0], w0) and _same(net.v, v0))
            effective = bool("b" in changed and "W" not in changed and exact)
        elif kind == "scale":
            s = float(value)
            # b は初期 0 なので `0*s == 0` でバイト不変。W と v が動くのが正。
            exact = bool(_same(net.Ws[0], w0 * s) and _same(net.bs[0], b0 * s)
                         and _same(net.v, v0 / s))
            effective = bool("W" in changed and "v" in changed
                             and "b" not in changed and exact)
        elif kind == "v_freeze":
            # ×1.0 は v をバイト単位で変えないのが正しい（浮動小数の恒等）。
            # 「効いている」の実体はフラグの方なので、両方を別々に要求する。
            exact = bool(_same(net.v, v0 * float(value))
                         and _same(net.Ws[0], w0) and _same(net.bs[0], b0))
            effective = bool(st["freeze_v"] is True
                             and (("v" in changed) == (float(value) != 1.0))
                             and exact)
        elif kind == "lr":
            eta = float(row["hook"]["value"])
            exact = bool(torch.equal(st["lr"], torch.full_like(st["lr"], eta))
                         and _same(net.Ws[0], w0) and _same(net.bs[0], b0)
                         and _same(net.v, v0))
            effective = bool(_tensor_bytes(st["lr"]) != lr_before
                             and all(r["lr"] == eta for r in st["runs"])
                             and exact)
        else:                                    # full_batch は状態を変えない
            exact = bool(_same(net.Ws[0], w0) and _same(net.bs[0], b0)
                         and _same(net.v, v0))
            effective = bool(not changed and st["batch_mode"] == "full32"
                             and exact)
        rows.append(dict(arm=arm, hook=kind, aliases_alive=alias,
                         changed=changed, effective=effective,
                         matches_expected_values=bool(exact)))
        ok = ok and alias and effective
    return dict(pass_=ok, rows=rows,
                note="どのテンソルが動いたかだけでなく、フック後の値が解析的な"
                     "期待値とバイト一致することも要求する（符号ミス・0 倍の"
                     "変異はパターン一致だけでは通ってしまう）")


def s_hook_noop(arm: str = "LRnull_1216", cfg: dict | None = None) -> dict:
    """S-hook-noop: ``hook=None`` が宿主の ``setup_arm_dial`` とバイト一致。

    **独立に 2 つの st を作る**（同じ物を比べる空虚な検査にしない）。同時に
    「``hook=negate`` にすると同じ比較が FAIL する」ことを記録し、検査自身が
    生きている証拠にする（spec §5）。
    """
    cfg = build_cfg() if cfg is None else cfg
    blk = _arm(cfg, arm)
    st_host = setup_arm_dial(cfg, blk, "cpu")
    st_mine = _setup_with_hook(cfg, blk, "cpu", None)
    host = _state_fingerprint(st_host)
    mine = _state_fingerprint(st_mine)
    same = sorted(k for k in host if host[k] == mine[k])
    diff = sorted(k for k in host if host[k] != mine[k])
    neg = _state_fingerprint(_setup_with_hook(cfg, blk, "cpu",
                                              {"type": "negate"}))
    neg_diff = sorted(k for k in host if host[k] != neg[k])
    # `mine` が本当に `_apply_hook` を通ったことの証拠。`_apply_hook` **だけ**が
    # 書く payload キーが `mine` に有り、宿主の st には無いことを要求する。
    # これが無いと `_setup_with_hook(..., None)` を `setup_arm_dial(...)` に
    # 差し替えても検査が緑のまま（＝自分自身との比較）になる。
    hook_keys = ("init_hook", "init_hook_arg", "lr_used", "freeze_v", "batch_mode")
    went_through_hook = bool(
        all(k in st_mine for k in hook_keys)
        and not any(k in st_host for k in hook_keys)
        and st_mine["init_hook"] == ""
        and st_mine["batch_mode"] == "online"
        and st_mine["freeze_v"] is False)
    return dict(pass_=bool(not diff and set(neg_diff) == {"W", "b", "v"}
                           and went_through_hook),
                arm=arm, equal=same, differing=diff,
                negate_differing=neg_diff,
                went_through_apply_hook=went_through_hook,
                hook_payload_keys={k: str(st_mine.get(k)) for k in hook_keys},
                host_has_hook_payload=sorted(k for k in hook_keys if k in st_host),
                note="negate が W/b/v だけを変えて FAIL するのが検査の生存証明。"
                     "went_through_apply_hook は「noop 側が本当に _apply_hook を"
                     "通った」証拠（自分自身との比較への退化を塞ぐ）")


def s_stream(cfg: dict | None = None, n_batches: int = 100) -> dict:
    """S-stream: 全 30 腕の step 0 の初期値と最初の 100 入力バッチがバイト一致。

    比較はフック適用**前**（フックは init を意図的に動かすため）。活性化・記録列が
    乱数を消費していないことの検査で、G5（腕を跨いだ (seed, unit) の対応）がこれに
    乗る。
    """
    cfg = build_cfg() if cfg is None else cfg
    ref_name, ref = None, None
    rows, ok = [], True
    for arm in arm_order():
        st = setup_arm_dial(cfg, _arm(cfg, arm), "cpu")
        finger = _state_fingerprint(st)
        env = st["env"]
        batches = b"".join(_tensor_bytes(env.step()) for _ in range(n_batches))
        finger["batches"] = batches
        finger["teacher"] = b"".join(
            _tensor_bytes(t) for t in st["teacher"].state_dict().values())
        if ref is None:
            ref_name, ref = arm, finger
            rows.append(dict(arm=arm, role="reference"))
            continue
        diff = sorted(k for k in ref if ref[k] != finger[k])
        rows.append(dict(arm=arm, differing=diff))
        ok = ok and not diff
    return dict(pass_=ok, reference=ref_name, n_batches=n_batches, rows=rows)


def _support_from_log(z: np.ndarray, w_free: np.ndarray) -> np.ndarray:
    """``z_p = zbar + sum_j s_j w_j/2`` で 32 点支持を復元（(rec, h) と (rec, h, 5)）。"""
    signs = np.array([[1.0 if (p >> j) & 1 else -1.0 for j in range(N_FREE)]
                      for p in range(2 ** N_FREE)], dtype=np.float64)
    return z[None, ...] + np.einsum("pj,rhj->prh", signs, w_free * 0.5)


def s_support(logdir: Path, arm: str, seeds=(0,), tol: float = 1e-5) -> dict:
    """S-support: ``zmin == 2*zbar - zmax``（相対 tol）かつ半幅 ``= 0.5*sum|w_free|``。"""
    rows, ok = [], True
    for seed in seeds:
        path = Path(logdir) / f"{arm}_seed{int(seed)}.npz"
        with np.load(path, allow_pickle=False) as z:
            step = z["step"].astype(np.int64)
            zbar = z["layer1_zbar"].astype(np.float64)
            zmax = z["layer1_zmax"].astype(np.float64)
            zmin = z["layer1_zmin"].astype(np.float64)
            wf = z["layer1_w_free"].astype(np.float64)
            wstep = z["layer1_w_free_step"].astype(np.int64)
        pred = 2.0 * zbar - zmax
        scale = np.maximum(np.abs(zmin).max(), np.abs(pred).max())
        mirror = float(np.abs(zmin - pred).max() / max(scale, 1e-300))
        sel = np.searchsorted(step, wstep)
        half = zmax[sel] - zbar[sel]
        half_w = 0.5 * np.abs(wf).sum(axis=-1)
        hscale = max(float(np.abs(half).max()), 1e-300)
        halfwidth = float(np.abs(half - half_w).max() / hscale)
        rows.append(dict(seed=int(seed), zmin_mirror_relerr=mirror,
                         halfwidth_relerr=halfwidth, n_records=int(len(step)),
                         n_task_rows=int(len(wstep))))
        ok = ok and mirror <= tol and halfwidth <= tol
    return dict(pass_=ok, arm=arm, tol=tol, rows=rows)


def s_moment(logdir: Path, arm: str, seeds=(0,), n_records: int = 10,
             tol: float = 1e-5) -> dict:
    """S-moment: 4 モーメント列を ``(zbar, w_free)`` から float64 で独立に再計算。

    recorder が使った経路（``full_support_ro`` → einsum）とは**別の道**（記録済みの
    ``zbar`` と自由ビット重みから支持を組み直す）で計算するので、恒真にならない。
    """
    row = table()[str(arm)]
    act, alpha = _act_of(arm)
    net = VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu")
    net.set_activation(act, alpha, "alpha_exp")
    rows, ok = [], True
    for seed in seeds:
        path = Path(logdir) / f"{arm}_seed{int(seed)}.npz"
        with np.load(path, allow_pickle=False) as z:
            step = z["step"].astype(np.int64)
            zbar = z["layer1_zbar"].astype(np.float64)
            wf = z["layer1_w_free"].astype(np.float64)
            wstep = z["layer1_w_free_step"].astype(np.int64)
            mstep = z["layer1_moment_step"].astype(np.int64)
            got = {k: z[f"layer1_{k}"].astype(np.float64) for k in MOMENT_KEYS}
        common = np.intersect1d(wstep, mstep)
        pick = common[np.linspace(0, len(common) - 1, min(n_records, len(common)),
                                  dtype=int)] if len(common) else common
        errs = {k: 0.0 for k in MOMENT_KEYS}
        for s in pick:
            zi = zbar[np.searchsorted(step, s)]
            wi = wf[np.searchsorted(wstep, s)]
            zz = torch.from_numpy(_support_from_log(zi[None, :], wi[None])[:, 0])
            phi = net.act_fn(zz)
            dphi = net.act_grad(zz, phi)
            ddphi = net.act_curv(zz)
            ref = {"m_phi2": (phi * phi).mean(dim=0),
                   "m_dphi2": (dphi * dphi).mean(dim=0),
                   "m_phidphi": (phi * dphi).mean(dim=0),
                   "m_dphiddphi": (dphi * ddphi).mean(dim=0)}
            mi = np.searchsorted(mstep, s)
            for k in MOMENT_KEYS:
                a = ref[k].numpy()
                b = got[k][mi]
                scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-12)
                errs[k] = max(errs[k], float(np.abs(a - b).max() / scale))
        rows.append(dict(seed=int(seed), n_sampled=int(len(pick)),
                         activation=act, relerr=errs))
        ok = ok and all(v <= tol for v in errs.values())
    return dict(pass_=ok, arm=arm, tol=tol, family=row["family"], rows=rows)


def _act_of(arm: str, cfg: dict | None = None) -> tuple[str, float]:
    cfg = build_cfg() if cfg is None else cfg
    return _activation(cfg, _arm(cfg, arm))


def s_C(logdir: Path, arm: str, seeds=(0,), tol: float = 1e-12,
        dose_tol: float = 1e-10) -> dict:
    """S-C: ``C = mu_norm^2 + 20*sigma_rms^2 + 1`` が閉形式 11.497681 と 1e-12 一致。

    spec §4.5-e は「``dose_relative_error`` が 0」と書くが、オラクル用量は float64 で
    解いた値なので実測は **1.5e-16 級**（committed 参照 `LR_1216` も同じ値を持つ）。
    ここは宿主 config の登録許容 ``sanity.s_dose_rel_tol = 1e-10`` を使い、実測値を
    そのまま記録する（字義の「0」は達成不能なので runner 側で読み替えた）。
    """
    rows, ok = [], True
    for seed in seeds:
        path = Path(logdir) / f"{arm}_seed{int(seed)}.npz"
        with np.load(path, allow_pickle=False) as z:
            mu = z["layer1_mu_norm"].astype(np.float64)
            sg = z["layer1_sigma_rms"].astype(np.float64)
            dose_err = z["dose_relative_error"].astype(np.float64)
        c = mu ** 2 + 20.0 * sg ** 2 + 1.0
        worst = float(np.abs(c - C_CLOSED_FORM).max())
        dose = float(np.abs(dose_err).max())
        rows.append(dict(seed=int(seed), C_min=float(c.min()),
                         C_max=float(c.max()), max_abs_dev=worst,
                         max_abs_dose_relative_error=dose))
        ok = ok and worst <= tol and dose <= dose_tol
    return dict(pass_=ok, arm=arm, closed_form=C_CLOSED_FORM, tol=tol,
                dose_tol=dose_tol, rows=rows,
                note="dose_relative_error の字義 0 は float64 の解の丸めで達成不能。"
                     "宿主 config の s_dose_rel_tol=1e-10 を使い実測値を残す")


def s_vfreeze(logdir: Path, arm: str = "Evf1_1216",
              ref_logdir: Path | None = None, ref_arm: str = "Enull_1216",
              seeds=(0,)) -> dict:
    """S-vfreeze: ``layer1_v_unit`` が全記録で厳密に定数、かつ初期 v が参照と bit 一致。"""
    rows, ok = [], True
    for seed in seeds:
        with np.load(Path(logdir) / f"{arm}_seed{int(seed)}.npz",
                     allow_pickle=False) as z:
            v = z["layer1_v_unit"]
            frozen = z["freeze_v"]
        ptp = float(np.ptp(v, axis=0).max())
        same_as_ref, ref_note = None, ""
        if ref_logdir is not None:
            ref_path = Path(ref_logdir) / f"{ref_arm}_seed{int(seed)}.npz"
            if ref_path.exists():
                with np.load(ref_path, allow_pickle=False) as zr:
                    same_as_ref = bool(v[0].tobytes() == zr["layer1_v_unit"][0]
                                       .tobytes())
            else:
                # 参照を渡されたのに読めないなら**落とす**（黙って None にすると
                # 「凍っている先が初期値である」という 5-c の因果の主張が消える）。
                same_as_ref = False
                ref_note = f"reference log missing: {ref_path}"
        rows.append(dict(seed=int(seed), v_ptp=ptp, freeze_v=bool(frozen),
                         initial_v_equals_reference=same_as_ref, note=ref_note))
        ok = ok and ptp == 0.0 and bool(frozen) and same_as_ref in (None, True)
    return dict(pass_=ok, arm=arm, reference=ref_arm, rows=rows)


def s_lr(logdir: Path, arm: str, seeds=(0,), cfg: dict | None = None) -> dict:
    """S-lr: ``lr_used`` 列が腕の η と一致し、``st['lr']`` と ``runs[i]['lr']`` の両方が
    書き換わっていること。"""
    cfg = build_cfg() if cfg is None else cfg
    hook = _hook_of(arm)
    want = (float(hook["value"]) if hook is not None and hook["type"] == "lr"
            else float(cfg["common"]["lr_main"]))
    st = _setup_with_hook(cfg, _arm(cfg, arm), "cpu", hook)
    # st["lr"] は float32 テンソルなので、python の 0.005 とは**そのままでは一致しない**。
    # 比べる相手は同じ dtype に落とした値（フックが書く `torch.full_like` と同じ丸め）。
    tensor_ok = bool(torch.equal(st["lr"], torch.full_like(st["lr"], want)))
    runs_ok = all(float(r["lr"]) == want for r in st["runs"])
    rows, ok = [], bool(tensor_ok and runs_ok)
    for seed in seeds:
        path = Path(logdir) / f"{arm}_seed{int(seed)}.npz"
        if not path.exists():
            rows.append(dict(seed=int(seed), status="MISSING"))
            ok = False
            continue
        with np.load(path, allow_pickle=False) as z:
            used = float(z["lr_used"])
        rows.append(dict(seed=int(seed), lr_used=used, expected=want))
        ok = ok and used == want
    return dict(pass_=ok, arm=arm, expected=want, st_lr_rewritten=tensor_ok,
                runs_lr_rewritten=runs_ok, rows=rows)


def _forward_f64(st: dict, X: torch.Tensor):
    """``forward_gate_batch`` の float64 版（S-fb を丸めから切り離すため）。"""
    net = st["net"]
    _refresh_fixed_offset(st)
    cur_in = X.double() - st["layer_means"][0].double()
    pre = torch.einsum("rhd,nrd->nrh", net.Ws[0].double(), cur_in) + net.bs[0].double()
    a = net.act_fn(pre)
    yhat = (a * net.v.double()).sum(dim=-1) + net.c.double()
    return [cur_in], [pre], [a], yhat


def s_fb(arm: str = "FBLR_1216", seeds=(0, 1, 2), steps: int = 5,
         tol: float = 1e-6, f32_tol: float = 1e-5,
         cfg: dict | None = None) -> dict:
    """S-fb: full-batch 勾配 = 32 個の単標本勾配の平均（相対 tol・3 seed × 5 step）。

    **主判定は float64** で行う。同じ float64 前向き量を (a) バッチ版と (b) 宿主の
    ``grads_centered_elu`` を 32 回呼んで平均、の両方に食わせるので、比べているのは
    「縮約の構造」だけになる（軸の取り違え・平均の掛け忘れは桁で落ちる）。
    実際に学習で使う float32 経路の差（32 項の総和の丸め・**相対 1e-7 級**）は
    ``relerr_f32`` として同じ行に記録し、別の（緩い）閾値で見る。
    """
    cfg = build_cfg() if cfg is None else cfg
    c = copy.deepcopy(cfg)
    c["common"]["seeds"] = [int(v) for v in seeds]
    st = _setup_with_hook(c, _arm(c, arm), "cpu", _hook_of(arm))
    net, env, teacher = st["net"], st["env"], st["teacher"]
    rows, ok = [], True
    for step in range(int(steps)):
        env.step()
        X = full_support_ro(env)
        Y = teacher(X)
        # --- 主判定: float64 で「バッチ平均」と「単標本 32 個の平均」を比べる ----
        i64, p64, a64, yhat64 = _forward_f64(st, X)
        d64 = yhat64 - Y.double()
        bWs, bbs, bgv, bgc = grads_centered_elu_batch(net, i64, p64, a64, d64)
        acc = None
        for p in range(X.shape[0]):
            one = grads_centered_elu(net, [i64[0][p]], [p64[0][p]], [a64[0][p]],
                                     d64[p])
            flat = [one[0][0], one[1][0], one[2], one[3]]
            acc = flat if acc is None else [u + v for u, v in zip(acc, flat)]
        ref = [u / float(X.shape[0]) for u in acc]
        got = [bWs[0], bbs[0], bgv, bgc]
        errs = {}
        for name, u, v in zip(("gW", "gb", "gv", "gc"), ref, got):
            scale = max(float(u.abs().max()), float(v.abs().max()), 1e-30)
            errs[name] = float((u - v).abs().max() / scale)
        # --- 記録: 実際に走る float32 経路の差（丸めの大きさを可視化しておく）----
        inputs, pres, acts, yhat = forward_gate_batch(st, X)
        gWs, gbs, gv, gc = grads_centered_elu_batch(net, inputs, pres, acts,
                                                    yhat - Y)
        acc32 = None
        for p in range(X.shape[0]):
            x = X[p]
            si, sp, sa, sy = forward_gate(st, x)
            one = grads_centered_elu(net, si, sp, sa, sy - teacher(x))
            flat = [one[0][0].double(), one[1][0].double(), one[2].double(),
                    one[3].double()]
            acc32 = flat if acc32 is None else [u + v for u, v in zip(acc32, flat)]
        ref32 = [u / float(X.shape[0]) for u in acc32]
        got32 = [gWs[0].double(), gbs[0].double(), gv.double(), gc.double()]
        errs32 = {}
        for name, u, v in zip(("gW", "gb", "gv", "gc"), ref32, got32):
            scale = max(float(u.abs().max()), float(v.abs().max()), 1e-30)
            errs32[name] = float((u - v).abs().max() / scale)
        rows.append(dict(step=step, relerr=errs, relerr_f32=errs32))
        ok = (ok and all(v <= tol for v in errs.values())
              and all(v <= f32_tol for v in errs32.values()))
        net.sgd_step_layers(st["lr"], gWs, gbs, gv, gc)
    return dict(pass_=ok, arm=arm, seeds=list(seeds), steps=int(steps),
                tol=tol, f32_tol=f32_tol, rows=rows,
                note="主判定は float64（縮約の構造）。float32 の差は 32 項の総和の丸め")


def s_par(outdir: Path, arm: str = "LRnull_1216", companion: str = "LIN_1216",
          steps: int = 100_000, cfg: dict | None = None) -> dict:
    """S-par: **腕プロセス並列**が logs をバイト単位で動かさないこと（spec §5）。

    (a) 対象の腕を単独で回す → (b) 同じ腕を別の腕と**同時に**回す、の 2 通りで
    ``logs/*.npz`` の全列を突き合わせる（宿主 ``gate_dial_0902.s_par`` と同じ流儀。
    同時に走る学習過程は 2 本まで・[[並列実行のメモリ予算]]）。

    **seed を分割した並列は S-par の対象ではない**（`s_seed_split_note` を参照）。
    この harness は seed をベクトル化して 1 本の乱数系列から ``[R, ...]`` を一度に
    引くので、``R`` を変えると 2 step 目以降の入力列がずれる。したがって登録した
    30 腕は必ず 10 seed をまとめて 1 プロセスで回す。
    """
    cfg = build_cfg() if cfg is None else cfg
    seeds = [int(v) for v in cfg["common"]["seeds"]]
    outdir = Path(outdir)
    serial = outdir / "serial"
    parallel = outdir / "parallel"
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))

    def cmd(name: str, out: Path) -> list[str]:
        return [sys.executable, "-m", "src.edge_law_0905", "--arm", name,
                "--steps", str(int(steps)), "--outdir", str(out)]

    solo = subprocess.Popen(cmd(arm, serial), cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    codes = [solo.wait()]
    procs = [solo]
    concurrent = [subprocess.Popen(cmd(name, parallel), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
                  for name in (arm, companion)]
    procs += concurrent
    codes += [p.wait() for p in concurrent]
    if any(codes):
        tails = [p.stdout.read().decode("utf-8", "replace")[-3000:] for p in procs]
        return dict(pass_=False, reason="a worker failed", returncodes=codes,
                    output=tails)
    rows, diffs = [], []
    for seed in seeds:
        a = serial / "logs" / f"{arm}_seed{seed}.npz"
        b = parallel / "logs" / f"{arm}_seed{seed}.npz"
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            if set(za.files) != set(zb.files):
                diffs.append(dict(seed=seed, where="columns"))
            for key in sorted(set(za.files) & set(zb.files)):
                if _sha_array(za[key]) != _sha_array(zb[key]):
                    diffs.append(dict(seed=seed, column=key))
            rows.append(dict(seed=seed, columns=len(za.files),
                             state_hash_equal=bool(str(za["state_hash_final"])
                                                   == str(zb["state_hash_final"]))))
    return dict(pass_=not diffs, arm=arm, companion=companion, steps=int(steps),
                rows=rows, differences=diffs,
                note="腕プロセス並列は runner の seed ループに触らないので決定性が保たれる")


def s_seed_split_note(arm: str = "LRnull_1216", steps: int = 3,
                      cfg: dict | None = None) -> dict:
    """記録用の**反例**: seed 部分集合は 10 seed 走と bit 一致しない（S-par の境界）。

    ``VecMLPL`` の init と教師は ``gens["init"]`` / ``gens["teacher"]`` から
    ``[R, ...]`` を 1 回で引くので、R を減らした走の値は 10 seed 走の**先頭 R 行**と
    一致する（実測）。一方 ``gens["input"]`` は ``SCREnv.__init__`` の
    ``flip_state (R,f)`` で既に R 個ぶん進むので、以降の ``env.step()`` の
    ``randint(0,2,(R,m-f))`` は **1 step 目から**ずれる。**ここが一致しないことを
    PASS 条件に登録する**ことで、`--seeds` を本走の分割に使う誤用を検査で止める。
    """
    cfg = build_cfg() if cfg is None else cfg
    full = copy.deepcopy(cfg)
    half = copy.deepcopy(cfg)
    half["common"]["seeds"] = [int(v) for v in cfg["common"]["seeds"]][:5]
    a = setup_arm_dial(full, _arm(full, arm), "cpu")
    b = setup_arm_dial(half, _arm(half, arm), "cpu")
    init_same = bool(_tensor_bytes(a["net"].Ws[0][:5])
                     == _tensor_bytes(b["net"].Ws[0]))
    flip_same = bool(_tensor_bytes(a["env"].flip_state[:5])
                     == _tensor_bytes(b["env"].flip_state))
    xs_a = [a["env"].step()[:5] for _ in range(int(steps))]
    xs_b = [b["env"].step() for _ in range(int(steps))]
    same = [bool(_tensor_bytes(u) == _tensor_bytes(v))
            for u, v in zip(xs_a, xs_b)]
    # 陽性対照: 同じ 10 seed 構成を 2 回作れば入力列は**一致する**。これが無いと
    # 「常に不一致を返すだけの検査」と区別できない（差が見えることの証明）。
    c = setup_arm_dial(copy.deepcopy(cfg), _arm(copy.deepcopy(cfg), arm), "cpu")
    d = setup_arm_dial(copy.deepcopy(cfg), _arm(copy.deepcopy(cfg), arm), "cpu")
    control = [bool(_tensor_bytes(c["env"].step()) == _tensor_bytes(d["env"].step()))
               for _ in range(int(steps))]
    n = int(steps)
    return dict(pass_=bool(init_same and flip_same and len(same) == n
                           and not any(same)
                           and len(control) == n and all(control)),
                arm=arm, init_rows_match=init_same,
                initial_flip_rows_match=flip_same, input_batches_match=same,
                identical_config_batches_match=control,
                note="R を変えると env.step() の入力列が 1 step 目からずれる。"
                     "登録 30 腕は 10 seed を必ず 1 プロセスで回すこと。"
                     "identical_config_batches_match は陽性対照（同じ構成なら"
                     "一致する＝この検査は等しさも見える）")


def recorder_bytes(total: int, runs: int = 10, width: int = 100,
                   every: int = 1000) -> int:
    """記録器が抱える配列の合計バイト数（step 数に比例する分の見積り）。

    短縮走行の RSS をそのまま本走に使うと**桁で過小評価**する（記録配列は
    ``(n_rec, R, h)`` なので 5M 走は 50k 走の 100 倍の行を持つ）。ここで解析的に
    出した差分を実測ピークに足して本走のピークを外挿する。
    """
    n_rec = total // every + 1
    n_task = total // PERIOD + 1
    n_mom = n_task + min(_MOMENT_DENSE_TASKS * PERIOD, total) // every
    unit = runs * width * 4                       # float32 の 1 記録ぶん
    # (n_rec, R, h) の列: M/B/denom/p_hat/w_norm・zbar/dzbar・mob/absmob/zmax/
    # zmean/v_unit・zmin = 13 本
    return int(13 * n_rec * unit + n_mom * 4 * unit + n_task * unit * N_FREE)


def rss_probe(outdir: Path, arm: str = "LRnull_1216", steps: int = 50_000) -> dict:
    """RSS: 短縮走行のピーク常駐量を実測し、並列数 min(20, floor(free/(1.5*peak)))。

    記録配列の分だけ本走の step 数へ外挿してから並列数を決める（spec §5 RSS 行と
    [[並列実行のメモリ予算]]）。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    cmd = ["/usr/bin/time", "-v", sys.executable, "-m", "src.edge_law_0905",
           "--arm", arm, "--steps", str(int(steps)), "--outdir",
           str(outdir / "run")]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    peak_kb = None
    for line in proc.stderr.splitlines():
        if "Maximum resident set size" in line:
            peak_kb = float(line.rsplit(":", 1)[1].strip())
    if peak_kb is None:                    # /usr/bin/time が無い環境の保険
        peak_kb = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    peak_gib = peak_kb / (1024.0 ** 2)
    free_gib = _free_gib()
    gib = 1024.0 ** 3
    base_gib = peak_gib - recorder_bytes(int(steps)) / gib
    horizons = sorted({row["total_steps"] for row in table().values()})
    projected = {int(h): base_gib + recorder_bytes(int(h)) / gib
                 for h in horizons}
    worst = max(projected.values())
    parallel = max(1, min(RSS_PARALLEL_CAP,
                          int(free_gib // (RSS_HEADROOM * max(worst, 1e-9)))))
    return dict(pass_=bool(proc.returncode == 0), arm=arm, steps=int(steps),
                peak_rss_gib=peak_gib, base_gib=base_gib,
                projected_peak_gib=projected, worst_peak_gib=worst,
                free_gib=free_gib, recommended_parallel=parallel,
                returncode=proc.returncode,
                formula="min(20, floor(free_GiB / (1.5 * worst projected peak)))")


def _free_gib() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {k.strip(): v for k, v in
                    (line.split(":", 1) for line in fh)}
        return float(info["MemAvailable"].strip().split()[0]) / (1024.0 ** 2)
    except (OSError, KeyError, ValueError):
        return 0.0


def _run_test_modules(modules: tuple[str, ...], covers=()) -> dict:
    """テストモジュールを走らせて結果だけ記録する（``--sanity`` から呼ぶ）。"""
    import io
    import unittest
    try:
        suite = unittest.TestLoader().loadTestsFromNames(list(modules))
    except Exception as exc:                       # モジュールが無い checkout
        return dict(pass_=False, error=repr(exc), modules=list(modules))
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    return dict(pass_=bool(res.wasSuccessful()), tests_run=res.testsRun,
                failures=[str(t) for t, _ in res.failures],
                errors=[str(t) for t, _ in res.errors],
                modules=list(modules), covers=list(covers))


def _nets_sanity() -> dict:
    """S-limit / S-flip / S-fd / S-curv / S-fallthrough / S-const / S-guard。"""
    return _run_test_modules(
        ("src.test_edge_law_nets_0905",),
        covers=["S-limit", "S-flip", "S-fd", "S-curv", "S-fallthrough",
                "S-const", "S-guard"])


def _suite_sanity() -> dict:
    """S-tests: runner・解析・宿主の写しのテストも ``--sanity`` の内側で走らせる。

    ``--sanity`` が PASS でも runner のテストが落ちている、という状態を作らない
    （列名の取り違えのように、`sanity_all` の項目では見えないが runner テストでは
    落ちる変異が実在する）。
    """
    return _run_test_modules(
        ("src.test_edge_law_runner_0905", "src.test_edge_law_analyze_0905",
         "src.test_weird_act_0903"),
        covers=["runner (hooks / recorder / writer / S-copy / parity)",
                "analysis (judgments / gates / selftest)",
                "host copy (weird_act_0903)"])


def _short_runs_for_sanity(outdir: Path, cfg: dict, steps: int = 30_000) -> Path:
    """S-support / S-moment / S-C / S-vfreeze / S-lr が読む短縮走行を用意する。"""
    base = Path(outdir) / "short"
    for arm in ("LRnull_1216", "Enull_1216", "Evf1_1216", "LRlr0p005_1216",
                "SP_1216"):
        run_single_arm(arm, steps=steps, outdir=base, cfg=cfg)
    return base / "logs"


def sanity_all(outdir: Path, *, quick: bool = False,
               logdir: Path | None = None, short_steps: int = 30_000) -> dict:
    """走らせる前に PASS が要る検査をまとめて回す（spec §5・``--sanity``）。

    config の ``sanity_required_before_run`` を全部満たす。ログを読む 5 つ
    （S-support / S-moment / S-C / S-vfreeze / S-lr）のために 30k step の短縮走行を
    5 腕ぶん自分で回す（``--logdir`` を渡せばそれを使う）。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg()
    logs = Path(logdir) if logdir is not None else _short_runs_for_sanity(
        outdir, cfg, short_steps)
    result = {
        "experiment": EXPERIMENT, "git_head": _git_head(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "short_run_logdir": str(logs), "short_run_steps": int(short_steps),
        "S-nets": _nets_sanity(),
        "S-copy": s_copy(),
        "S-hook-inplace": s_hook_inplace(cfg),
        "S-hook-noop": s_hook_noop(cfg=cfg),
        "S-stream": s_stream(cfg),
        "S-support": s_support(logs, "LRnull_1216", seeds=range(10)),
        "S-moment": {arm: s_moment(logs, arm, seeds=(0, 4, 9))
                     for arm in ("LRnull_1216", "Enull_1216", "SP_1216")},
        "S-C": s_C(logs, "Enull_1216", seeds=range(10)),
        "S-vfreeze": s_vfreeze(logs, "Evf1_1216", ref_logdir=logs,
                               ref_arm="Enull_1216", seeds=range(10)),
        "S-lr": {arm: s_lr(logs, arm, seeds=range(10), cfg=cfg)
                 for arm in ("LRnull_1216", "LRlr0p005_1216")},
        "S-fb": s_fb(cfg=cfg),
        "S-seed-split": s_seed_split_note(cfg=cfg),
        "columns": expected_column_agreement(logs),
    }
    if not quick:
        # 重いので --sanity-quick では回さない（それでも項目としては必ず出す）。
        result["S-tests"] = _suite_sanity()
    else:
        result["S-tests"] = dict(pass_=True, skipped="--sanity-quick")
    if not quick:
        result["S-par"] = s_par(outdir / "s_par", cfg=cfg)
        result["RSS"] = rss_probe(outdir / "rss")

    def _ok(value) -> bool:
        if isinstance(value, dict) and "pass_" in value:
            return bool(value["pass_"])
        if isinstance(value, dict):
            return all(_ok(v) for v in value.values())
        return True

    result["pass_"] = all(_ok(v) for v in result.values())
    path = outdir / "sanity.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"[sanity] {'PASS' if result['pass_'] else 'FAIL'} -> {path}", flush=True)
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}: {'PASS' if _ok(value) else 'FAIL'}", flush=True)
    return result


def expected_column_agreement(logdir: Path | None = None,
                              arm: str = "LRnull_1216", seed: int = 0) -> dict:
    """runner が**実際に書いた** npz の列名と、解析モジュールが読む列名の突き合わせ。

    以前はここが「解析の期待列 対 手書きのリテラル集合」で、``write_arm_logs_edge``
    を一切見ていなかった（列名を変えても PASS した）。正本は実ログにして、
    リテラルは「書いたつもり」との二重確認として残す。``logdir`` が無いときは
    リテラルだけで比べるが、その旨を ``from_real_logs=False`` で明記する。
    """
    from .edge_law_analyze_0905 import expected_columns    # 失敗したら例外のまま
    want = expected_columns()
    writes = set(want["run"]) | set(want["unit"]) | set(want["new_unit"])
    for pair in want["new_aux"]:
        writes |= set(pair)
    writes |= set(want["payload"])
    literal = {"step", "unfit", "layer1_zmin", "layer1_w_free",
               "layer1_w_free_step", "layer1_moment_step", "init_hook",
               "init_hook_arg", "lr_used", "freeze_v", "batch_mode"}
    literal |= {f"layer1_{k}" for k in MOMENT_KEYS}
    literal |= {f"layer1_{k}" for k in ("zbar", "zmean", "zmax", "dzbar", "denom",
                                        "v_unit", "w_norm", "mob", "absmob", "M",
                                        "B", "p_hat")}
    real, from_real = None, False
    if logdir is not None:
        path = Path(logdir) / f"{arm}_seed{int(seed)}.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                real = set(z.files)
            from_real = True
    mine = real if from_real else literal
    missing = sorted(writes - mine)
    # リテラルと実ログの食い違いも記録する（どちらかが古くなったら気づけるように）
    literal_only = sorted(literal - real) if from_real else []
    return dict(pass_=bool(not missing and not literal_only),
                from_real_logs=from_real, source=(str(logdir) if logdir else None),
                missing=missing, literal_not_written=literal_only,
                analysis_expects=sorted(writes))


# ---------------------------------------------------------------------------
# 走った後の比較（spec §4.1-b・§5）
# ---------------------------------------------------------------------------
def s_null(arm: str, ref_dir: Path, outdir: Path,
           ref_arm: str | None = None) -> dict:
    """S-null / S-null-E: 共通列が ``np.array_equal(..., equal_nan=True)``。

    「全列 bit 一致」は原理的に不可能（``arm`` / ``run_id`` は必ず違う・
    ``state_hash_final`` は参照が 15M 終端・新列は参照に無い）。部分走（1M）でも
    使えるように共通の記録数まで切り詰める（spec §5）。
    """
    # 参照腕は**明示**（既定で自分自身に落とさない）。以前は
    # `S_NULL_REF.get(arm, arm)` だったので、登録の 2 腕以外を渡すと同じファイルを
    # 2 回読んで「PASS・n_compared 54」と出てしまった（本プロジェクトが一度
    # 焼かれた「テンソルを自分自身と比べる空虚な S 検査」そのもの）。
    if ref_arm is None:
        if arm not in S_NULL_REF:
            raise SanityError(
                f"S-null has no registered reference for {arm!r} "
                f"(registered: {sorted(S_NULL_REF)}); pass ref_arm explicitly")
        ref_arm = S_NULL_REF[arm]
    if str(ref_arm) == str(arm) and (
            Path(ref_dir).resolve() == (Path(outdir) / "logs").resolve()):
        raise SanityError(
            f"S-null would compare {arm!r} with itself "
            f"({Path(ref_dir).resolve()}); that check is vacuous")
    rows = []
    for seed in range(10):
        pa = Path(outdir) / "logs" / f"{arm}_seed{seed}.npz"
        pb = Path(ref_dir) / f"{ref_arm}_seed{seed}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=seed, status="MISSING",
                             mine=pa.exists(), reference=pb.exists()))
            continue
        a = np.load(pa, allow_pickle=True)
        b = np.load(pb, allow_pickle=True)
        n = min(len(a["step"]), len(b["step"]))
        bad, compared = {}, []
        for key in sorted(set(a.files) & set(b.files)):
            if key in S_NULL_SKIP or key == "state_hash_1m":
                continue
            x, y = a[key], b[key]
            if x.ndim >= 1 and x.shape[0] == len(a["step"]):
                x = x[:n]
            if y.ndim >= 1 and y.shape[0] == len(b["step"]):
                y = y[:n]
            compared.append(key)
            if x.shape != y.shape or x.dtype != y.dtype:
                bad[key] = f"shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}"
            elif x.dtype.kind in "fiub":
                if not np.array_equal(x, y, equal_nan=True):
                    bad[key] = float(np.nanmax(np.abs(x.astype(float)
                                                      - y.astype(float))))
            elif not np.array_equal(x, y):
                bad[key] = "differs"
        # `state_hash_1m` は 1M step の checkpoint で書かれるので、**両方の走が
        # 1M step に届いている**ときだけ比べる（記録数 n > 1000 で判定していたが、
        # 記録間隔が変われば意味が変わるし、短縮走行では常に None になっていて
        # この節が生きているかを試験できなかった）。
        reached = (int(a["step"][-1]) >= 1_000_000
                   and int(b["step"][-1]) >= 1_000_000)
        hash_equal = (str(a["state_hash_1m"]) == str(b["state_hash_1m"])
                      if reached else None)
        rows.append(dict(seed=seed, status="OK", n_records=int(n),
                         n_compared=len(compared), n_bad=len(bad), bad=bad,
                         state_hash_1m_equal=hash_equal))
    ok = bool(rows) and all(
        r.get("status") == "OK" and r["n_bad"] == 0
        and r["state_hash_1m_equal"] in (None, True) for r in rows)
    return dict(pass_=ok, check="S-null", arm=arm, reference_arm=ref_arm,
                reference_dir=str(ref_dir), rows=rows)


def _bytes_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(x.shape == y.shape and x.dtype == y.dtype
                and x.tobytes() == y.tobytes())


def _mirror_column(x: np.ndarray, y: np.ndarray, *, flip: bool) -> dict:
    """列別パリティ（spec §4.1-b）。規則の実体は ``_mirror_rule``（解析側）。

    ここで規則を書き直さないこと。以前は ``0.0 - x`` で期待値を作っていたが、
    IEEE では ``0.0 - (±0.0) == +0.0`` なので**零ちょうどで比較が符号盲**になり、
    しかも不一致位置を ``!=`` で拾っていたので ``pass_=False, n_mismatch=0`` という
    矛盾した結果が出せた。期待値は ``np.negative``・比較はバイト・両腕とも厳密に
    ``0.0`` の要素だけを**数えて**通す（登録された例外）。
    """
    return _mirror_rule(x, y, flip=flip)


def s_mirror(arm: str, ref_dir: Path, outdir: Path,
             ref_arm: str | None = None) -> dict:
    """S-mirror（spec §4.1-b）: 列別パリティ・バイト比較・``p'+p == 1.0`` の厳密等号。"""
    ref_arm = ref_arm or S_MIRROR_REF.get(arm, "LR_1216")
    rows = []
    for seed in range(10):
        pa = Path(outdir) / "logs" / f"{arm}_seed{seed}.npz"
        pb = Path(ref_dir) / f"{ref_arm}_seed{seed}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=seed, status="MISSING"))
            continue
        a = np.load(pa, allow_pickle=True)
        b = np.load(pb, allow_pickle=True)
        n = min(len(a["step"]), len(b["step"]))
        cols = {}
        for key in MIRROR_SIGN_FLIP + MIRROR_INVARIANT:
            if key not in a.files or key not in b.files:
                cols[key] = dict(pass_=False, missing=True)
                continue
            cols[key] = _mirror_column(a[key][:n], b[key][:n],
                                       flip=key in MIRROR_SIGN_FLIP)
        pa_hat, pb_hat = a["layer1_p_hat"][:n], b["layer1_p_hat"][:n]
        total = pa_hat.astype(np.float64) + pb_hat.astype(np.float64)
        exceptions = int(np.count_nonzero(total != 1.0))
        rows.append(dict(seed=seed, status="OK", n_records=int(n),
                         columns={k: v for k, v in cols.items()},
                         p_hat_sum_exceptions=exceptions,
                         p_hat_exact=bool(exceptions == 0),
                         # 登録した例外: 両腕でちょうど 0.0 の要素（零の符号だけの差）
                         zero_sign_exceptions=int(sum(
                             int(v.get("n_zero_sign_exceptions", 0))
                             for v in cols.values())),
                         elements_compared=int(sum(
                             int(v.get("n_records_compared", 0))
                             for v in cols.values()))))
    ok = bool(rows) and all(
        r.get("status") == "OK" and r["p_hat_exact"]
        and all(c["pass_"] for c in r["columns"].values()) for r in rows)
    return dict(pass_=ok, check="S-mirror", arm=arm, reference_arm=ref_arm,
                reference_dir=str(ref_dir),
                sign_flipped=list(MIRROR_SIGN_FLIP),
                invariant=list(MIRROR_INVARIANT),
                excluded=["layer1_zmax (参照に zmin が無い)", "eff_rank/quantile 系"],
                rows=rows)


def s_mirror_zmax(arm: str, other: str, outdir: Path) -> dict:
    """§4.1-b の代替検査: 新列つき 2 腕の間で ``zmax'(FLn) == -zmin(LRnull)``。"""
    rows = []
    for seed in range(10):
        pa = Path(outdir) / "logs" / f"{arm}_seed{seed}.npz"
        pb = Path(outdir) / "logs" / f"{other}_seed{seed}.npz"
        if not (pa.exists() and pb.exists()):
            rows.append(dict(seed=seed, status="MISSING"))
            continue
        with np.load(pa, allow_pickle=False) as a, np.load(pb, allow_pickle=False) as b:
            n = min(len(a["step"]), len(b["step"]))
            got = _mirror_column(b["layer1_zmin"][:n], a["layer1_zmax"][:n],
                                 flip=True)
        rows.append(dict(seed=seed, status="OK", n_records=int(n), **got))
    ok = bool(rows) and all(r.get("status") == "OK" and r["pass_"] for r in rows)
    return dict(pass_=ok, check="S-mirror-zmax", arm=arm, other=other, rows=rows)


# ---------------------------------------------------------------------------
# 末尾抜き出し（`results/edge_law_0905/logs_tail/`・git add -f の対象）
# ---------------------------------------------------------------------------
def _tail_keep_steps(step: np.ndarray, total: int,
                     dense: bool = True) -> np.ndarray:
    """登録判定が読む記録だけを残す。

    窓（settle / lag / tail / タスク 100→300→500）は ``_window_indices`` が
    **タスク終端の記録しか拾わない**ので、タスク終端をすべて残せば窓統計は完全に
    再現できる。5-g (ii) だけは末尾 20 タスクの 1000-step 記録で ``dzbar`` を積む
    ので、そこは密に残す。step 0（初期値）も残す。
    """
    keep = (step == 0) | (step % PERIOD == 0)
    if not dense:
        return keep
    dense_from = total - _MOMENT_DENSE_TASKS * PERIOD
    return keep | (step > dense_from)


def tail_extract(logdir: Path, outdir: Path, arms=None,
                 dense: bool = True) -> dict:
    """判定が読む記録だけの縮約 npz を書く（spec §7）。

    ``dense=False`` は末尾 20 タスクの 1000-step 記録を落とす（容量は 4 割弱に
    なるが、§4.5-g (ii) の ``dz̄`` 対 ``-eta*G`` が logs_tail から計算できなくなる）。
    """
    logdir, outdir = Path(logdir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in (arms if arms is not None else arm_order()):
        for seed in range(10):
            src = logdir / f"{arm}_seed{seed}.npz"
            if not src.exists():
                continue
            with np.load(src, allow_pickle=True) as z:
                step = z["step"].astype(np.int64)
                keep = _tail_keep_steps(step, int(step[-1]), dense)
                payload = {}
                for key in z.files:
                    arr = z[key]
                    if arr.ndim >= 1 and arr.shape[0] == len(step):
                        payload[key] = arr[keep]
                    else:
                        payload[key] = arr
            dst = outdir / f"{arm}_seed{seed}.npz"
            np.savez_compressed(dst, **payload)
            rows.append(dict(arm=arm, seed=seed, kept=int(keep.sum()),
                             of=int(len(step)),
                             bytes=int(dst.stat().st_size)))
    total = sum(r["bytes"] for r in rows)
    print(f"[tail-extract] {len(rows)} files, {total / 1e6:.1f} MB -> {outdir}",
          flush=True)
    return dict(pass_=bool(rows), outdir=str(outdir), total_bytes=total,
                dense=bool(dense), rows=rows)


# ---------------------------------------------------------------------------
# 投入計画（spec §7）
# ---------------------------------------------------------------------------
def launch_plan(parallel: int | None = None, peak_gib: float | None = None,
                logdir: str = "logs_run", base_gib: float = 0.60) -> str:
    """30 腕の投入コマンド（bash）と推奨並列数を返す。

    ピークは「実測ベース（記録配列を除いた常駐分）＋ 腕の地平線ぶんの記録配列」で
    見積もる。50k 短縮走行の実測値をそのまま使うと 15M 腕で 2 倍以上の過小評価に
    なる（[[並列実行のメモリ予算]]）。
    """
    gib = 1024.0 ** 3
    peak = (float(peak_gib) if peak_gib
            else base_gib + recorder_bytes(15_000_000) / gib)
    peak_5m = base_gib + recorder_bytes(5_000_000) / gib
    free = _free_gib()
    rec = (int(parallel) if parallel
           else max(1, min(RSS_PARALLEL_CAP,
                           int(free // (RSS_HEADROOM * max(peak, 1e-9))))))
    rows = sorted(table().values(), key=lambda r: -r["total_steps"])
    lines = [
        "#!/usr/bin/env bash",
        "# edge_law_0905 本走（spec §7）。長い腕から投入する（臨界パス = 15M 腕）。",
        f"# 見積りピーク RSS: 5M 腕 {peak_5m:.2f} GiB / 15M 腕 {peak:.2f} GiB "
        f"（記録配列は step 数に比例する）",
        f"# 空き {free:.1f} GiB → 並列 min(20, floor({free:.1f}/(1.5*{peak:.2f}))) "
        f"= {rec}（空きが増えたら PAR を上げ直すこと）",
        "# 30 腕 = 5M×25 + 10M×2 + 15M×2 + 500k×1、CPU 合計 ≈ 620 分。",
        "set -u",
        f"cd {ROOT}",
        f"mkdir -p {logdir}",
        f"PAR={rec}",
        "run_arm() {",
        "  OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 \\",
        f"    --arm \"$1\" > {logdir}/edge_law_0905_$1.log 2>&1",
        "}",
        "export -f run_arm",
        "cat <<'ARMS' | xargs -P \"$PAR\" -I{} bash -c 'run_arm {}'",
    ]
    lines += [row["name"] for row in rows]
    lines += [
        "ARMS",
        "# 走り終わったら:",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 "
        "--s-null LRnull_1216 results/p3_extend_0902/logs",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 "
        "--s-null Enull_1216 results/p3_extend_0902/logs",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 "
        "--s-mirror FLn_1216 results/p3_extend_0902/logs",
        "# layer1_zmax は S-mirror を張れない（参照に zmin が無い）ので、§4.1-b が"
        " 登録した代替検査を本走の logs どうしで回す:",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 "
        "--s-mirror-zmax FLn_1216 LRnull_1216",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_0905 "
        "--tail-extract",
        "OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m "
        "src.edge_law_analyze_0905 --outdir results/edge_law_0905",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="edge_law_0905 runner")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="検査・短縮走行用。**本走では使わない**: seed は "
                             "ベクトル化されているので部分集合は別の入力列になる "
                             "(s_seed_split_note)")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--sanity-quick", action="store_true")
    parser.add_argument("--s-null", nargs=2, metavar=("ARM", "REF_DIR"),
                        default=None)
    parser.add_argument("--s-mirror", nargs=2, metavar=("ARM", "REF_DIR"),
                        default=None)
    parser.add_argument("--s-mirror-zmax", nargs=2, metavar=("ARM", "OTHER"),
                        default=None,
                        help="§4.1-b の代替検査（layer1_zmax は参照に zmin が無く "
                             "S-mirror を張れない）。**同じ走の outdir の中**の "
                             "新列つき 2 腕で zmax'(FLn) == -zmin(LRnull) を見る")
    parser.add_argument("--logdir", default=None,
                        help="--sanity がログ検査に使う既存の logs/（省略時は "
                             "30k step の短縮走行を自分で回す）")
    parser.add_argument("--tail-extract", action="store_true")
    parser.add_argument("--tail-no-dense", action="store_true",
                        help="末尾 20 タスクの 1000-step 記録を縮約から落とす"
                             "（§4.5-g (ii) が logs_tail から出せなくなる）")
    parser.add_argument("--launch-plan", action="store_true")
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--peak-gib", type=float, default=None)
    args = parser.parse_args()

    out = Path(args.outdir).resolve() if args.outdir else None
    main_dir = Path(ROOT) / registered()["output"]["dir"]

    if args.launch_plan:
        print(launch_plan(args.parallel, args.peak_gib), end="")
        return
    if args.sanity or args.sanity_quick:
        pre = out or (Path(ROOT) / f"results/_preflight_{EXPERIMENT}")
        result = sanity_all(pre, quick=bool(args.sanity_quick),
                            logdir=Path(args.logdir) if args.logdir else None)
        if not result["pass_"]:
            raise SanityError(f"sanity failed: see {pre / 'sanity.json'}")
        return
    if args.s_null:
        arm, ref = args.s_null
        result = s_null(arm, Path(ref).resolve(), out or main_dir)
        _report_check(result, out or main_dir)
        return
    if args.s_mirror:
        arm, ref = args.s_mirror
        result = s_mirror(arm, Path(ref).resolve(), out or main_dir)
        _report_check(result, out or main_dir)
        return
    if args.s_mirror_zmax:
        arm, other = args.s_mirror_zmax
        result = s_mirror_zmax(arm, other, out or main_dir)
        _report_check(result, out or main_dir)
        return
    if args.tail_extract:
        # --outdir を渡したときは縮約もその下に置く（検査走が results/ を汚さない）
        base = out or main_dir
        dest = (base / "logs_tail" if out is not None
                else Path(ROOT) / registered()["output"]["logs_tail_dir"])
        tail_extract(base / "logs", dest, dense=not args.tail_no_dense)
        return
    if args.arm:
        if args.arm not in table():
            parser.error(f"--arm must be one of {arm_order()}")
        cfg = build_cfg()
        require_omp(cfg)
        run_single_arm(args.arm, args.steps, out, args.seeds, cfg=cfg)
        return
    parser.error("nothing to do: pass --arm / --sanity / --s-null / --s-mirror "
                 "/ --s-mirror-zmax / --tail-extract / --launch-plan")


def _report_check(result: dict, outdir: Path) -> None:
    path = Path(outdir) / "sanity" / f"{result['check'].lower()}_{result['arm']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"[{result['check']} {result['arm']}] "
          f"{'PASS' if result['pass_'] else 'FAIL'} -> {path}", flush=True)
    for row in result["rows"]:
        print("  ", json.dumps(row, ensure_ascii=False, default=str)[:400],
              flush=True)


if __name__ == "__main__":
    main()
