#!/usr/bin/env python3
"""act_chimera_rlmnist_0913: Random-Label MNIST の runner（spec_act_chimera_0913 §2.3・§3・§6・§8）。

    /usr/bin/python3 -m src.act_chimera_rlmnist_0913 --arm SMAXH --seed 0 --device cpu \\
        --outdir results/act_chimera_0913/rlmnist                     # 1 ジョブ = 1 (腕, seed)
    /usr/bin/python3 -m src.act_chimera_rlmnist_0913 --merge --arm SMAXH --outdir ...   # seed の csv を結合
    /usr/bin/python3 -m src.act_chimera_rlmnist_0913 --g1-smoke --device cuda \\
        --outdir results/_smoke_act_chimera_0913/rlmnist               # G1-RL（GPU だけ・40 セル）と b1[0] +1e−3 の対照（§2.3・追補 2-1）
    /usr/bin/python3 -m src.act_chimera_rlmnist_0913 --determinism --device cpu|cuda   # 選んだ device での決定性と双子の対照
    /usr/bin/python3 -m src.act_chimera_rlmnist_0913 --g1-full --outdir results/act_chimera_0913/rlmnist  # GPU 本走の 3000 セル

プロトコルは ``pmnist_rlmnist_0906``（RL）と同一: 50 タスク × 400 epoch × 75 step、784-100-100-10、
手書き Adam lr 1e−3、seed ごとに 1200 枚を ``rl_subset`` で引き、ラベルはタスクごとに ``rl_labels`` で
引き直す。``RL.run_one``（RL:116-195）を **逐語で写し**、登録ブロック R1–R3（§2.3）だけを足す:

    R1 置換 :120  ``act = H.ARMS[arm]`` → ``act = act_obj``（活性化は ``act_chimera_0913.make_act``）
    R2 挿入 :123 の直後  双子の摂動（b1[k] に +1e−6・追補 2-2。旧 W1[k, 0] の 1 ulp）と読み出しの状態の初期化（W(0) の保存）
    R3 挿入 :192 の直後  §3.3–3.4 の per-unit 読み出し。``.cpu().double()`` に写してから計算する

S14（``test_act_chimera_0913``）が本体の AST 差分をこの 3 ブロックと照合する。``H.ARMS`` への
実行時登録はしない。host（``src/pmnist_0905.py``・``src/pmnist_rlmnist_0906.py``）は編集しない。

出力（§8）: ``rlmnist/{ARM}/s{seed}/per_task.csv``（``H.write_csv`` の ``.10g`` 整形・G1-RL は参照
``results/pmnist_rlmnist_0906/LR/per_task.csv`` とセルの文字列一致）、``provenance.json``、
``readout_s{seed}.npz``（凍結スキーマ ``READOUT_SCHEMA``）、INCOMPLETE の走だけ ``divergence.json``。
``--merge`` が ``rlmnist/{ARM}/per_task.csv`` を seed 順に結合する（行数を seed ごとに 50、INCOMPLETE
なら打ち切りまでと照合する）。

発散: 0906 は非有限の loss をタスク終端で検出して break する（RL:186-191）。本 runner はそれを
黙って落とさず ``INCOMPLETE`` と記録し、``divergence.json`` に最初の非有限タスクを書く（§2.3・§4.0）。
per_task.csv には完了したタスクの行だけを書く（host が足す ``acc = nan`` の行は書かない）。

null 双子（追補 2-2）: 双子 k は初期値の b1[k] に +1e−6（``AC.perturb_twin_b1``）。W1[k, 0] は RL では死んでいる
（MNIST の画素 0 は訓練 60,000 枚すべてで 0・RL は入力を置換しない）。``run_job`` は双子の init を S18
（``AC.twin_init_report``）で検査し、LR と双子はタスク終端ごとに b1[k] を除いた params の sha256 を provenance の
``checks.s18b.sha_excl_b1`` に書く（S18b。``twin_live_task`` は判定器が同じ seed の LR と比べて出す）。

RL の床（§4.0）: ``AT_FLOOR`` = O_arm(t31–50) ≤ F_seed + Φ⁻¹(1 − 0.05/24)·0.00026。F_seed は
``rl_labels`` 系列から厳密に計算する（``floor_seed``）。

読み出し（§3.3–3.4・R3）はすべて cpu float64 で計算する（``use_deterministic_algorithms(True)`` の CUDA
では histc・bincount・scatter_add 系が RuntimeError になる）。前活性は float32 の params を double に
写して 1200 枚の上で組み直す。``evaluate_rl`` の列（csv）は host のまま device の float32 で計算する。

data: worktree に ``data/`` は無いので ``H.DATA_DIR`` を main clone の ``data/mnist`` に固定する
（``boundary_groups_0908.py:10`` と同じ）。4 つの gz の sha256 は 0906 の provenance と一致することを
起動時に assert する（S11）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pmnist_rlmnist_0906 as RL                       # host loop（持ち込み・編集しない）
from src import act_chimera_0913 as AC
from src.pmnist_rlmnist_0906 import (Iv, subset_idx, task_labels, evaluate_rl,   # noqa: F401（写した本体が参照する）
                                     STEPS_PER_EPOCH, N_IMAGES, BATCH)

H = RL.H                                                          # src.pmnist_0905（持ち込み・編集しない）
MAIN_CLONE = Path('/home/issan/Projects/claude/proj_004_drift')
H.DATA_DIR = MAIN_CLONE / 'data' / 'mnist'                        # §2.3 data（boundary_groups_0908.py:10 と同じ固定）

RUN_ID = 'act_chimera_0913'
ENV = 'rlmnist'
SPEC = ROOT / 'specs' / 'spec_act_chimera_0913.md'
REF_DIR = ROOT / 'results' / 'pmnist_rlmnist_0906'               # 0906 の LR/R（33a0cab から持ち込み）
DEFAULT_OUT = ROOT / 'results' / RUN_ID / ENV
SMOKE_OUT = ROOT / 'results' / '_smoke_act_chimera_0913' / ENV

LR_RATE = 0.001                                                   # §2.1: 全腕で 0906 の lr
EPOCHS = 400                                                      # §2.3: 400 epoch を短縮しない
N_TASKS = 50
WINDOW = (31, 50)                                                 # §3.1: 判定窓（verdict_rlmnist.py:16）
EARLY_WINDOW = (1, 10)                                            # §4 副: 早期 online
U = H.DIMS[1]                                                     # 100 hidden units per layer

STAGE1_ARMS = ('LR', 'ELU1', 'SMAXH', 'SMINH', 'LRtw0')            # §2.1: H 家族 4 腕 + 双子 1
STAGE2_ARMS = ('VMIN', 'VMAX', 'SMAXS', 'SMINS')                  # §2.1 段 2: V・S 家族
STAGE2_OPTIONAL_ARMS = ('GN', 'FN')                               # §2.1 段 2: 分離腕（任意）
SEEDS = (0, 1, 2)
TWIN_KS = tuple(AC.twin_of(a)[1] for a in STAGE1_ARMS if AC.twin_of(a)[1] is not None)   # (0,)（追補 2-2・S18b）

# §2.1: 参照を作ったインタプリタに固定する（G0 で assert し全 provenance に書く）
INTERPRETER = '/usr/bin/python3'
TORCH_VERSION = '2.13.0+cu130'
NUMPY_VERSION = '2.5.2'

# §2.3: 33a0cab から同一 bytes で持ち込む 7 ファイル
CARRY7 = {
    'src/pmnist_0905.py': '53e2c10288ff91e39148e81f5756769e1a9a4ff88b375e31120f87745eab26ec',
    'src/pmnist_rlmnist_0906.py': '6f3624b59fd70ad0638da201eddcd5c1ca91d057df83c167352074380d7debdd',
    'results/pmnist_rlmnist_0906/LR/per_task.csv': '3d67114c2962c441efb458dd7891eaff71e07b86cab58ea6825000b9b3006817',
    'results/pmnist_rlmnist_0906/LR/provenance.json': '69f4ef6ff7770742352630731b435aae7cc83976ed5ed15fccd78525d3797149',
    'results/pmnist_rlmnist_0906/R/per_task.csv': '7a1903614e589d525f1c7637e1c3154aaad7be07d4e9f384f58366c1d9021158',
    'results/pmnist_rlmnist_0906/R/provenance.json': '45e4866c6c58490384f991d9332f256ad89a9152dd2cb095c8f21eaed3a219f4',
    'analysis/pmnist_0905/verdict_rlmnist.py': 'd555f23ad80e44b9bd276db4aee093d125f23fb0c690b60b7caf932d9aebc977',
}

# G1-RL（§2.3）: 数値 20 列（online_acc … w_norm_l3）を `.10g` の文字列で比べる
G1_COLS = ('online_acc', 'memo_acc', 'acc',
           'dead_frac_l1', 'zeroout_l1', 'zbar_l1', 'zsd_l1', 'zbar_min_l1', 'mob_l1', 'eff_rank_l1',
           'dead_frac_l2', 'zeroout_l2', 'zbar_l2', 'zsd_l2', 'zbar_min_l2', 'mob_l2', 'eff_rank_l2',
           'w_norm_l1', 'w_norm_l2', 'w_norm_l3')
G1_SEEDS = (0, 1, 2)
G1_SMOKE_CELLS = len(G1_COLS) * 2                                # LR s0 t1–2 = 40 セル
G1_FULL_CELLS = len(G1_COLS) * N_TASKS * len(G1_SEEDS)           # GPU 本走 = 3000 セル

# RL の床（§4.0）
FLOOR_SD = 0.00026          # 0906 の R（10 seed × 20 タスク）の (online_t − 最多比率_t) の sd 0.00116 / √20
FLOOR_FAMILY = 24           # 床規則を当てる 8 腕 × 3 seed（段 1 から 24 で固定・§4.0）
FLOOR_Z = NormalDist().inv_cdf(1. - 0.05 / FLOOR_FAMILY)        # Φ⁻¹(1 − 0.05/24) = 2.865（spec の「2.86」）
FLOOR_MARGIN = FLOOR_Z * FLOOR_SD

# §8 読み出しのスキーマ（RL: 箱 B と同じキー、E_ident を除く。T 軸を除いた形）
READOUT_SCHEMA = {
    'fn_mom': ((2, 8, 4, 3), np.float64),      # 8 関数 × 帯 B0–B3 × {φ′, φ, h} の (unit, 入力) 平均・層 1, 2
    'occ_start': ((2, U, 4), np.float32),      # 帯分率（タスク開始 = 前タスク終端、t = 1 は W(0)）
    'occ_end': ((2, U, 4), np.float32),
    'mob_band': ((2, U, 4), np.float32),       # 自腕の帯別ゲート（分離腕は φ′_bwd）
    'pabs1': ((2, U), np.float32),             # P[|z| < 1]
    'zbar_start': ((2, U), np.float64),
    'zbar_end': ((2, U), np.float64),
    'depth_vel': ((2, U), np.float64),         # 層 1: Δz̄_i、層 2: Δ(W2_i·μ_φ1 + b2_i)
    'step_num': ((2, U), np.float64),          # ‖ΔW̃_i‖
    'wt_norm': ((2, U), np.float64),           # ‖W̃_i‖
    'mu_comp': ((U,), np.float64),             # 層 2 の μ̂ 方向の成分 W2_i·μ̂
    'adam_ratio': ((2, U), np.float32),        # 行ごとの mean |m̂|/(√v̂ + ε)
    'adam_epsfrac': ((2, U), np.float32),      # 行ごとの √v̂ < 10ε の割合
    'gbar_l2': ((U,), np.float64),             # 層 2 の実現ゲート E_x[φ′(z2)]
    'mu_phi1': ((U,), np.float64),             # 1200 枚上の層 1 出力の平均
    'param_sha': ((), '<U64'),                 # タスク終端 params の sha256
}
READOUT_KEYS = tuple(READOUT_SCHEMA)


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def params_sha256(params) -> str:
    """タスク終端 params の sha256（H.check_init と同じ組み方: float32 の bytes を結合）。"""
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def assert_environment() -> dict:
    """§2.1: インタプリタ・torch・numpy を参照を作ったものに固定する。落ちたら黙って続けない。"""
    env = dict(executable=sys.executable, torch=torch.__version__, numpy=np.__version__)
    assert sys.executable == INTERPRETER, (sys.executable, INTERPRETER)
    assert torch.__version__ == TORCH_VERSION, (torch.__version__, TORCH_VERSION)
    assert np.__version__ == NUMPY_VERSION, (np.__version__, NUMPY_VERSION)
    return env


def check_carry7(root: Path = ROOT) -> dict:
    """§2.3: 持ち込み 7 ファイルの sha256 が表と一致すること（S11 の一部）。"""
    got = {k: sha256_file(root / k) for k in CARRY7}
    bad = {k: (got[k], v) for k, v in CARRY7.items() if got[k] != v}
    assert not bad, bad
    return got


def reference_data_sha256() -> dict:
    """0906 の provenance の data_sha256（LR と R で同一であることも見る）。"""
    lr = json.loads((REF_DIR / 'LR' / 'provenance.json').read_text())['data_sha256']
    r = json.loads((REF_DIR / 'R' / 'provenance.json').read_text())['data_sha256']
    assert lr == r, (lr, r)
    return lr


GATE_SHAPE_PROV = MAIN_CLONE / 'results' / 'gate_shape_0911' / 'LR_s0_provenance.json'


def check_data_sha256(mnist_sha: dict) -> dict:
    """S11: MNIST 4 ファイルの sha256 = 0906 の provenance **と** gate_shape_0911 の provenance の data_sha256（§6 S11）。"""
    ref = reference_data_sha256()
    assert set(mnist_sha) == set(ref) and all(mnist_sha[k] == ref[k] for k in ref), (mnist_sha, ref)
    gs = json.loads(GATE_SHAPE_PROV.read_text())['data_sha256']
    assert gs == ref, ('0906 and gate_shape_0911 data_sha256 differ', gs, ref)
    return dict(mnist_sha)


def git_state() -> dict:
    """§6: git_hash と、src/ configs/ specs/ の変更・未追跡（新しいモジュールが commit 前だと git_hash がコードを指さない）。"""
    def _git(*a):
        try:
            return subprocess.check_output(['git', *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    dirty = _git('status', '--porcelain', '--untracked-files=all', '--', 'src', 'configs', 'specs')
    return dict(git_hash=_git('rev-parse', 'HEAD'),
                git_dirty_src=(None if dirty is None else [l for l in dirty.splitlines() if '__pycache__' not in l]))


def machine_info(device: torch.device) -> dict:
    cpu = ''
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    return dict(hostname=socket.gethostname(), cpu=cpu, gpu=gpu, device=str(device),
                platform=platform.platform(), pid=os.getpid())


def other_processes() -> list:
    """他セッションの学習プロセス（§8 provenance）: 自分以外の python プロセスの ps 行。"""
    try:
        out = subprocess.run(['ps', '-eo', 'pid,rss,etimes,args'], capture_output=True, text=True,
                             timeout=10).stdout.splitlines()
    except Exception as e:  # noqa: BLE001
        return [f'ps failed: {e!r}']
    me = str(os.getpid())
    return [ln.strip()[:200] for ln in out[1:]
            if 'python' in ln and ln.split()[0] != me]


def nvidia_smi() -> str:
    try:
        return subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'nvidia-smi failed: {e!r}'


# --------------------------------------------------------------------------
# RL の床（§4.0）・endpoint（§3.1）
# --------------------------------------------------------------------------

def majority_ratio(seed: int, n_tasks: int = N_TASKS) -> np.ndarray:
    """タスク t = 1..n のラベルの最多クラス比率。``rl_labels`` 系列を run_one と同じ順に引く。"""
    g = H.stream('rl_labels', seed)
    out = np.empty(n_tasks)
    for t in range(n_tasks):
        y = task_labels(g)
        out[t] = float(torch.bincount(y, minlength=H.N_CLASSES).max()) / N_IMAGES
    return out


def floor_seed(seed: int, window: tuple = WINDOW) -> float:
    """F_seed = 窓内の最多クラス比率の平均（seed 0/1/2 で 0.11450 / 0.11400 / 0.11383・§4.0）。"""
    m = majority_ratio(seed, window[1])
    return float(m[window[0] - 1:window[1]].mean())


def floor_threshold(seed: int, window: tuple = WINDOW) -> float:
    return floor_seed(seed, window) + FLOOR_MARGIN


def at_floor(online_window_mean: float, seed: int, window: tuple = WINDOW) -> bool:
    """AT_FLOOR: O_arm(t31–50) ≤ F_seed + Φ⁻¹(1 − 0.05/24) × 0.00026（§4.0）。"""
    return bool(online_window_mean <= floor_threshold(seed, window))


def floor_sd_from_reference(csv_path: Path = REF_DIR / 'R' / 'per_task.csv', seeds=range(10),
                            window: tuple = WINDOW) -> dict:
    """0.00026 の導出の再現: R の (online_t − 最多比率_t) を 10 seed × 20 タスクでプールした sd（ddof = 1）/ √20。"""
    rows = read_csv_rows(csv_path)
    d, corr = [], {}
    for s in seeds:
        m = majority_ratio(s, window[1])[window[0] - 1:window[1]]
        on = np.array([float(r['online_acc']) for r in rows
                       if int(r['seed']) == s and window[0] <= int(r['task']) <= window[1]])
        assert on.shape == m.shape, (s, on.shape, m.shape)
        d += list(on - m)
        corr[s] = float(np.corrcoef(on, m)[0, 1])
    sd = float(np.std(d, ddof=1))
    n_win = window[1] - window[0] + 1
    return dict(sd_task=sd, sd_over_sqrt_n=sd / np.sqrt(n_win), n=len(d), corr_own_seed=corr)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def window_series(rows: list, col: str = 'online_acc', window: tuple = WINDOW) -> np.ndarray:
    """窓内のタスク系列（task 順）。窓が欠けていれば例外（INCOMPLETE の走は endpoint を持たない）。"""
    got = {int(r['task']): float(r[col]) for r in rows if window[0] <= int(r['task']) <= window[1]}
    missing = [t for t in range(window[0], window[1] + 1) if t not in got]
    assert not missing, f'window {window} missing tasks {missing}'
    return np.array([got[t] for t in range(window[0], window[1] + 1)])


def endpoints(rows: list, seed: int | None = None, window: tuple = WINDOW) -> dict:
    """§3.1 の co-primary: Y = −100·mean online_acc(t31–50) [pt]、Y_logit = −mean logit(online_acc)。
    ``seed`` を渡すと床（AT_FLOOR）も付ける。"""
    o = window_series(rows, 'online_acc', window)
    out = dict(Y_pt=float(-100. * o.mean()), Y_logit=float(-_logit(o).mean()), online_mean=float(o.mean()),
               n=int(o.size))
    if seed is not None:
        out.update(F_seed=floor_seed(seed, window), floor_threshold=floor_threshold(seed, window),
                   at_floor=at_floor(float(o.mean()), seed, window))
    return out


# --------------------------------------------------------------------------
# 読み出し（§3.3–3.4・R2/R3）
# --------------------------------------------------------------------------

class RLReadout:
    """§3.3–3.4 の per-unit 読み出し（RL）。R2 で W(0) を保存し、R3 でタスク終端に計算する。

    すべて cpu float64 の上で計算する（§2.3 R3）。前活性は params を double に写して 1200 枚 x の上で
    z1 = x W1ᵀ + b1、a1 = φ(z1)、z2 = a1 W2ᵀ + b2 と組み直す。タスク開始時の状態は前タスク終端の
    キャッシュ（t = 1 は R2 で保存した W(0)）。学習には触らない（S10）。

    - 帯・8 関数・帯別ゲートは ``act_chimera_0913`` の部品（帯は腕に依らない・φ′ は自腕の ``dphi``
      = 分離腕では φ′_bwd・``fn_mom`` は表の 8 関数を自腕の状態の上で評価する・§1.5）。
    - 層 2 の μ_φ1 は終端の値で固定して前後に使う（§3.4）。μ̂ = μ_φ1/‖μ_φ1‖。‖μ_φ1‖ = 0 なら
      μ̂ = 0（W̃2 = W2）とし、その回数を ``mu_zero_count`` に数える。
    - Adam の比は W_l の行ごと（bias は含めない）。m̂ = m/(1 − β₁^t)、v̂ = v/(1 − β₂^t)、ε = 1e−8。
    - ``g_batch`` は状態を **読む** だけ（sha256 を provenance の S9 に使う）。消費しないことは
      S10 (ii) が検査する。
    - ``sha_excl_elems`` = {名前: (j, idx)} を置くと、タスク終端ごとに params[j][idx] を 0 にした写しの sha256 を
      ``sha_excl[名前]`` に積む（S18b・追補 2-2。run_job が LR と双子に b1[k] を置く）。
    """
    BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

    def __init__(self, act):
        self.act = act
        self.p0 = None                        # R2: W(0)（float32・cpu）
        self.prev = None                      # 前タスク終端の (z1, z2, W1, b1, W2, b2)・cpu double
        self.rec = {k: [] for k in READOUT_KEYS}
        self.rng_sha = []                     # g_batch の状態の sha256（タスク終端）
        self.task_wall = []                   # タスク終端の時刻（処理量の実測・§7）
        self.last_params = None               # 最後のタスク終端 params（cpu clone・S10 用、保存しない）
        self.mu_zero_count = 0
        self.sha_excl_elems = {}              # S18b（追補 2-2）: 名前 -> (j, idx)
        self.sha_excl = {}
        self._xd = None
        self._t0 = time.time()

    # ---- R2
    def after_init(self, params):
        self.p0 = [p.detach().cpu().clone() for p in params]
        self._t0 = time.time()

    # ---- 部品
    @torch.no_grad()
    def _state(self, pc, xd):
        W1, b1, W2, b2 = pc[:4]
        z1 = xd @ W1.T + b1
        a1 = self.act.phi(z1)
        z2 = a1 @ W2.T + b2
        return z1, a1, z2

    # ---- R3
    @torch.no_grad()
    def end_task(self, t, params, adam, x, act, g_batch):
        assert act is self.act, 'readout was built for a different activation object'
        assert self.p0 is not None, 'after_init (R2) was not called'
        if self._xd is None:
            self._xd = x.detach().cpu().double()
        xd = self._xd
        pc = [p.detach().cpu().double() for p in params]
        if self.prev is None:                                  # t = 1: 開始状態は W(0)
            p0 = [p.double() for p in self.p0]
            z1s, _, z2s = self._state(p0, xd)
            self.prev = (z1s, z2s, *p0[:4])
        z1s, z2s, W1p, b1p, W2p, b2p = self.prev
        z1, a1, z2 = self._state(pc, xd)
        W1, b1, W2, b2 = pc[:4]
        r = {}
        fn_mom = torch.empty(2, 8, 4, 3, dtype=torch.float64)
        occ_s, occ_e, mobb = (torch.empty(2, U, 4, dtype=torch.float64) for _ in range(3))
        pabs, zb_s, zb_e = (torch.empty(2, U, dtype=torch.float64) for _ in range(3))
        d2 = None
        for l, (z, zs) in enumerate(((z1, z1s), (z2, z2s))):
            d = act.dphi(z)
            if l == 1:
                d2 = d
            fn_mom[l] = AC.fn_mom(z)
            occ_s[l] = AC.occupancy(zs)
            occ_e[l] = AC.occupancy(z)
            mobb[l] = AC.mob_band(d, z)
            pabs[l] = AC.pabs1(z)
            zb_s[l] = zs.mean(0)
            zb_e[l] = z.mean(0)
        r.update(fn_mom=fn_mom, occ_start=occ_s, occ_end=occ_e, mob_band=mobb, pabs1=pabs,
                 zbar_start=zb_s, zbar_end=zb_e)
        mu = a1.mean(0)                                        # μ_φ1（終端で固定・§3.4）
        r['mu_phi1'] = mu
        r['gbar_l2'] = d2.mean(0)
        dv = torch.empty(2, U, dtype=torch.float64)
        dv[0] = zb_e[0] - zb_s[0]                              # 層 1: v_i = Δz̄_i（画像が固定）
        dv[1] = (W2 @ mu + b2) - (W2p @ mu + b2p)              # 層 2: Δ(W2_i·μ_φ1 + b2_i)
        r['depth_vel'] = dv
        sn, wn = (torch.empty(2, U, dtype=torch.float64) for _ in range(2))
        Wt1, Wt1p = W1 - W1.mean(1, keepdim=True), W1p - W1p.mean(1, keepdim=True)   # 層 1: 中心化行
        sn[0], wn[0] = (Wt1 - Wt1p).norm(dim=1), Wt1.norm(dim=1)
        nrm = float(mu.norm())
        if nrm > 0.:
            muh = mu / nrm
        else:
            muh = torch.zeros_like(mu)
            self.mu_zero_count += 1
        Wt2, Wt2p = W2 - torch.outer(W2 @ muh, muh), W2p - torch.outer(W2p @ muh, muh)   # 層 2: μ̂ 方向を除く
        sn[1], wn[1] = (Wt2 - Wt2p).norm(dim=1), Wt2.norm(dim=1)
        r.update(step_num=sn, wt_norm=wn, mu_comp=W2 @ muh)
        ar, ef = (torch.full((2, U), float('nan'), dtype=torch.float64) for _ in range(2))
        if adam is not None:
            m, v, tc = adam
            c1, c2 = 1 - self.BETA1 ** tc[0], 1 - self.BETA2 ** tc[0]
            for l, i in ((0, 0), (1, 2)):                      # W1, W2 の行
                mh = m[i].detach().cpu().double() / c1
                vh = v[i].detach().cpu().double() / c2
                sv = vh.sqrt()
                ar[l] = (mh.abs() / (sv + self.EPS)).mean(1)
                ef[l] = (sv < 10 * self.EPS).double().mean(1)
        r.update(adam_ratio=ar, adam_epsfrac=ef)
        r['param_sha'] = params_sha256(params)
        for name, ex in self.sha_excl_elems.items():            # S18b（追補 2-2）: 摂動した要素を除いた sha256
            self.sha_excl.setdefault(name, []).append(AC.params_sha256(params, exclude=ex))
        for k in READOUT_KEYS:
            self.rec[k].append(r[k])
        self.rng_sha.append(hashlib.sha256(g_batch.get_state().numpy().tobytes()).hexdigest())
        self.task_wall.append(time.time() - self._t0)
        self.last_params = [p.detach().cpu().clone() for p in params]
        self.prev = (z1, z2, W1, b1, W2, b2)

    # ---- 保存
    def arrays(self) -> dict:
        out = {}
        for k, (shape, dt) in READOUT_SCHEMA.items():
            vals = self.rec[k]
            if k == 'param_sha':
                out[k] = np.array(vals, dtype='<U64')
            else:
                out[k] = (torch.stack(vals).numpy() if vals else np.empty((0,) + shape)).astype(dt)
        return out

    def save(self, path: Path) -> dict:
        arr = self.arrays()
        validate_readout(arr)
        np.savez(path, **arr)
        return {k: list(v.shape) for k, v in arr.items()}


def validate_readout(arr: dict, n_tasks: int | None = None) -> int:
    """§8 のスキーマ（キー・dtype・形）との照合。G0.5 と test が使う。返り値は T。"""
    missing = [k for k in READOUT_KEYS if k not in arr]
    extra = [k for k in arr if k not in READOUT_SCHEMA]
    assert not missing and not extra, (missing, extra)
    T = None
    for k, (shape, dt) in READOUT_SCHEMA.items():
        a = arr[k]
        assert a.dtype == np.dtype(dt), (k, a.dtype, dt)
        assert a.shape[1:] == shape, (k, a.shape, shape)
        T = a.shape[0] if T is None else T
        assert a.shape[0] == T, (k, a.shape[0], T)
    if n_tasks is not None:
        assert T == n_tasks, (T, n_tasks)
    return T


def load_readout(path: Path, n_tasks: int | None = None) -> dict:
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    validate_readout(arr, n_tasks)
    return arr


class PerturbInitReadout(RLReadout):
    """G1-RL の変異対照（§2.3・追補 2-1）: 初期値 b1[0] += delta（既定 1e−3）で 1 タスク回すと 1 セル以上が不一致になること。
    旧来の W1[0, 0] は、MNIST の画素 0 が訓練 60,000 枚すべてで 0 で RL は入力を置換しないので、出力にも勾配にも
    効かず、対照が空虚だった（スモークで 0/20）。層 1 のバイアスは全標本で生きている。"""

    def __init__(self, act, delta: float = 1e-3, param: int = 1, index=(0,)):
        super().__init__(act)
        self.delta, self.param, self.index = delta, param, index

    def after_init(self, params):
        with torch.no_grad():
            params[self.param][self.index] += self.delta
        super().after_init(params)


# --------------------------------------------------------------------------
# RL.run_one（RL:116-195）の逐語の写し + 登録ブロック R1–R3（§2.3・S14）
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist,
            device: torch.device, optimizer: str = "adam", epochs: int = 400,
            c: float = 0.6, beta: float = 0.01, iv: str = "none",
            debug: dict | None = None, *, act_obj, ro: RLReadout,
            twin: int | None = None) -> tuple[list[dict], dict]:
    act = act_obj                                     # R1（§2.3）: `act = H.ARMS[arm]` を置換
    if act.kind == "adaptive_snake":
        act = H.AdaptiveSnake(c, beta, device)        # fresh statistics per run
    params = H.init_params(seed, device)              # host init: bit-identical per seed
    if twin is not None:                              # R2（§2.3・追補 2-2）: 双子は b1[k] に +1e−6（S18）
        AC.perturb_twin_b1(params, twin)
    ro.after_init(params)                             # R2（§2.3）: 読み出しの状態の初期化（W(0) の保存）
    if debug is not None:                             # S-init / S-online hook, unused in real runs
        debug["init"] = [q.detach().cpu().clone() for q in params]
    ivo = Iv.parse(iv)
    p0 = [q.detach().clone() for q in params] if ivo.kind == "l2init" else None
    # Adam moments live for the whole run: resetting them between tasks would break
    # the continual definition the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0]) if optimizer == "adam" else None

    idx = subset_idx(seed).to(device)
    x = mnist.train_x[idx]                            # the task's inputs, fixed forever
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    spt = STEPS_PER_EPOCH * epochs
    rows = []
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        y = task_labels(g_lab).to(device)             # new labelling, same images
        if debug is not None:
            debug.setdefault("subset", []).append(idx.cpu().clone())
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            xs, ys = x[order], y[order]               # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = e * STEPS_PER_EPOCH + j
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = H.forward(params, xb, act)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss
                # came from, i.e. before this batch has been learned from.
                hit = (out[4].detach().argmax(1) == yb).float().mean()
                acc_sum += hit
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad,
                                           torch.tensor(s, device=device), bad_step)
                    if ivo.kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0))
                                 for i, (q, gr) in enumerate(zip(params, grads))]
                    if adam is None:
                        for p, gr in zip(params, grads):
                            p -= lr * gr
                    else:
                        m, v, tc = adam
                        tc[0] += 1
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        c1 = 1 - b1 ** tc[0]
                        c2 = 1 - b2 ** tc[0]
                        for p, gr, mi, vi in zip(params, grads, m, v):
                            mi.mul_(b1).add_(gr, alpha=1 - b1)
                            vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                            p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    if isinstance(act, H.AdaptiveSnake):
                        act.update(out[0], out[2])    # running var of this batch's preacts
                if debug is not None:
                    debug.setdefault("online", []).append(float(hit))

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                         "acc": float("nan")})
            break                                     # drop, never rescue
        m = evaluate_rl(params, x, y, act)
        ro.end_task(t, params, adam, x, act, g_batch)  # R3（§2.3）: §3.3–3.4 の per-unit 読み出し（.cpu().double() で計算）
        rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                     "online_acc": float(acc_sum) / spt, "memo_acc": m["acc"], **m})
    return rows, diverged


# --------------------------------------------------------------------------
# csv・G1-RL（§2.3）
# --------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list:
    with Path(path).open(newline='') as fh:
        return list(csv.DictReader(fh))


def csv_header() -> str:
    """0906 の per_task.csv の見出し行（H.write_csv と同じ列の順）。行が 0 の csv の見出しに使う。"""
    return (REF_DIR / 'LR' / 'per_task.csv').read_text().splitlines()[0]


def g1_rl_compare(new_csv: Path, ref_csv: Path = REF_DIR / 'LR' / 'per_task.csv', seeds=(0,),
                  tasks=None, max_report: int = 20, allow_missing: bool = False) -> dict:
    """G1-RL: 新ランナーの行と 0906 の LR の行を、数値 20 列のセルの **文字列** で比べる（`.10g` 整形
    どうし。`1` と `1.0` の違いも不一致）。``tasks`` = None は新 csv にある全タスク。``allow_missing`` なら、新 csv に
    無い (seed, task) の行は例外にせず、その 20 セルを不一致（``n_missing_rows``）に数える（打ち切りの走が
    比較を縮めて通らないように）。件数ガードは呼び出し側が ``n_cells`` に当てる（スモーク 40・GPU 本走 3000）。"""
    new = {(int(r['seed']), int(r['task'])): r for r in read_csv_rows(new_csv)}
    ref = {(int(r['seed']), int(r['task'])): r for r in read_csv_rows(ref_csv)}
    n_cells, mism, n_missing = 0, [], 0
    for s in seeds:
        ts = sorted(t for (ss, t) in new if ss == s) if tasks is None else list(tasks)
        for t in ts:
            assert (s, t) in ref, f'reference csv has no row for seed {s} task {t}'
            if (s, t) not in new:
                assert allow_missing, f'new csv has no row for seed {s} task {t}'
                n_missing += 1
                for c in G1_COLS:
                    n_cells += 1
                    mism.append(dict(seed=s, task=t, col=c, new=None, ref=ref[(s, t)][c]))
                continue
            for c in G1_COLS:
                n_cells += 1
                a, b = new[(s, t)][c], ref[(s, t)][c]
                if a != b:
                    mism.append(dict(seed=s, task=t, col=c, new=a, ref=b))
    return dict(n_cells=n_cells, n_mismatch=len(mism), n_missing_rows=n_missing, mismatches=mism[:max_report],
                pass_=(n_cells > 0 and not mism))


def g1_full(outdir: Path, seeds=G1_SEEDS, n_tasks: int = N_TASKS) -> dict:
    """G1-RL を GPU の本走に広げた場合（§2.3）: LR s0–2 の全 50 タスク = 3000 セルの文字列一致と件数ガード。"""
    per_seed, n_cells, n_mism = {}, 0, 0
    for s in seeds:
        p = Path(outdir) / 'LR' / f's{s}' / 'per_task.csv'
        prov = json.loads((p.parent / 'provenance.json').read_text()) if (p.parent / 'provenance.json').exists() else {}
        dev = (prov.get('machine') or {}).get('device')
        if not p.exists():
            per_seed[s] = dict(status='MISSING')
            n_cells += 0
            continue
        r = g1_rl_compare(p, seeds=(s,), tasks=range(1, n_tasks + 1), allow_missing=True)
        per_seed[s] = dict(n_cells=r['n_cells'], n_mismatch=r['n_mismatch'], n_missing_rows=r['n_missing_rows'],
                           device=dev, mismatches=r['mismatches'][:5])
        n_cells += r['n_cells']
        n_mism += r['n_mismatch']
    want = len(G1_COLS) * n_tasks * len(seeds)
    devices = {v.get('device') for v in per_seed.values()}
    return dict(check='G1-RL-full', n_cells=n_cells, n_cells_want=want, n_mismatch=n_mism, devices=sorted(map(str, devices)),
                per_seed=per_seed, pass_=bool(n_cells == want and n_mism == 0 and devices == {'cuda'}))


# --------------------------------------------------------------------------
# 1 ジョブ = 1 (腕, seed)（§8）
# --------------------------------------------------------------------------

def arm_activation(label: str):
    """腕ラベル -> (活性化名, 双子番号, 活性化オブジェクト)。未知名は make_act の KeyError。"""
    name, k = AC.twin_of(label)
    return name, k, AC.make_act(name)


def load_mnist(device: torch.device) -> H.Mnist:
    mnist = H.Mnist(device)
    assert mnist.train_x.shape[0] == RL.TRAIN_N, mnist.train_x.shape
    check_data_sha256(mnist.sha256)                            # S11（起動時）
    return mnist


G1_GATING = 'G1-RL'


def g1_gating(name: str, twin: int | None, seed: int, lr: float, epochs: int, ro_cls, device_type: str) -> str | None:
    """走ごとの G1-RL の扱い（2-4 #25・追補 2-1）。None = 当てない（``checks.g1_rl`` を書かない）。``G1_GATING`` = 落ちたら
    ``failed_checks`` に G1_RL（GPU・プロトコルの lr と 400 epoch の LR・seed 0–2・双子でない・**登録の読み出し RLReadout**）。
    それ以外の文字列 = 記録するが落とさない。変異対照の読み出し（``PerturbInitReadout`` など init を変える派生クラス）は
    設計上 1 セル以上食い違うので、ここで gating にすると ``g1_smoke`` の対照の走が AssertionError で落ち、
    g1_smoke.json が書かれない（追補 2-1 で対照が生きた後に顕在化）。"""
    if not (name == 'LR' and twin is None and seed in G1_SEEDS and lr == LR_RATE and epochs == EPOCHS):
        return None
    if ro_cls is not RLReadout:
        return f'informational (non-registered readout {getattr(ro_cls, "__name__", ro_cls)}: mutation control / test; not gating)'
    if device_type != 'cuda':
        return 'informational (CPU run; 0906 reference is GPU)'
    return G1_GATING


def run_job(label: str, seed: int, *, device: torch.device, outdir: Path = DEFAULT_OUT,
            n_tasks: int = N_TASKS, epochs: int = EPOCHS, lr: float = LR_RATE, mnist=None,
            ro_cls=RLReadout, threads: int | None = None, check_env: bool = True) -> dict:
    """1 (腕, seed) を回し、``outdir/{label}/s{seed}/`` に per_task.csv・readout_s{seed}.npz・
    provenance.json（INCOMPLETE なら divergence.json も）を書く。返り値は要約 dict。"""
    t_start = time.time()
    env = assert_environment() if check_env else dict(executable=sys.executable, torch=torch.__version__,
                                                      numpy=np.__version__)
    name, twin, act = arm_activation(label)
    carry = check_carry7()
    mnist = mnist if mnist is not None else load_mnist(device)
    out = Path(outdir) / label / f's{seed}'
    out.mkdir(parents=True, exist_ok=True)
    procs_before, smi_before = other_processes(), nvidia_smi()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    ro = ro_cls(act)
    # S18b（追補 2-2）: LR は b1[k]（k ∈ TWIN_KS）を、双子 k は自分の b1[k] を除いた sha256 をタスク終端ごとに記録する
    ks = (twin,) if twin is not None else (TWIN_KS if name == 'LR' else ())
    ro.sha_excl_elems = {str(k): (AC.TWIN_PARAM, (k,)) for k in ks}
    ro.sha_excl = {str(k): [] for k in ks}
    rows, div = run_one(label, seed, lr, n_tasks, mnist, device, epochs=epochs,
                        act_obj=act, ro=ro, twin=twin)
    complete = [r for r in rows if 'online_acc' in r]
    status = 'INCOMPLETE' if div['diverged'] else 'COMPLETE'
    assert len(complete) == len(ro.rec['param_sha']), (len(complete), len(ro.rec['param_sha']))

    if complete:
        H.write_csv(out / 'per_task.csv', complete)
    else:                                                      # task 1 で発散: 行 0 でも見出しは書く（merge が行数を照合できるように）
        (out / 'per_task.csv').write_text(csv_header() + '\n')
    div_path = out / 'divergence.json'
    removed_stale = False
    if status == 'INCOMPLETE':
        div_path.write_text(json.dumps(dict(run_id=RUN_ID, env=ENV, arm=label, seed=seed, status=status,
                                            first_nonfinite_task=div['task'], step=div['step'],
                                            n_complete_tasks=len(complete)), indent=2))
    elif div_path.exists():
        div_path.unlink()
        removed_stale = True
    shapes = ro.save(out / f'readout_s{seed}.npz')

    checks = dict(carry7_sha256=carry, data_sha256_matches_0906=True, interpreter=env,
                  rng_state_sha_t1_3=ro.rng_sha[:3], mu_zero_count=ro.mu_zero_count)
    failed = []
    if twin is not None:
        # S18（追補 2-2）: 走そのものの init（R2 の後に読み出しが保存した W(0)）が、同じ seed の host の init と b1[k] の
        # 1 要素だけ違い、値が float32 で b1[k] + 1e−6 に最も近い表現値であること。対照: 摂動 0 の双子は「差 0 要素」で落ちる。
        base = [q.detach().cpu() for q in H.init_params(seed, device)]
        s18 = AC.twin_init_report(base, ro.p0, twin)
        c0 = AC.twin_init_report(base, base, twin)
        s18['ctl_zero_perturbation'] = dict(n_diff=c0['n_diff'], pass_=c0['pass_'],
                                            fails_as_required=bool(c0['n_diff'] == 0 and not c0['pass_']))
        s18['init_sha256'] = params_sha256(ro.p0)
        checks['s18'] = s18
        if not (s18['pass_'] and s18['ctl_zero_perturbation']['fails_as_required']):
            failed.append('S18')
    if ks:
        checks['s18b'] = dict(excluded='params[1][k] = b1[k] set to 0 in a CPU copy before sha256', ks=list(ks),
                              sha_excl_b1={k: list(v) for k, v in ro.sha_excl.items()}, n_tasks=len(complete),
                              rule='twin_live_task = first task where the twin differs from same-seed LR; '
                                   'a twin not live by the window start (t31) is excluded from sigma_traj')
        assert all(len(v) == len(complete) for v in ro.sha_excl.values()), {k: len(v) for k, v in ro.sha_excl.items()}
    gating = g1_gating(name, twin, seed, lr, epochs, ro_cls, device.type)
    if gating is not None:
        # §2.3: 登録の地平線（n_tasks）の全タスクを数える（打ち切りで比較が縮まないように・欠けた行は不一致に数える）
        g1 = g1_rl_compare(out / 'per_task.csv', seeds=(seed,), tasks=range(1, n_tasks + 1), allow_missing=True)
        g1['count_ok'] = bool(g1['n_cells'] == len(G1_COLS) * n_tasks)
        g1['n_tasks_registered'] = n_tasks
        g1['gating'] = gating
        g1['pass_'] = bool(g1['pass_'] and g1['count_ok'])
        checks['g1_rl'] = g1
        if gating == G1_GATING and not g1['pass_']:
            failed.append('G1_RL')
    ep = None
    if status == 'COMPLETE' and n_tasks >= WINDOW[1]:
        ep = endpoints(complete, seed)
    wall = time.time() - t_start
    spt = STEPS_PER_EPOCH * epochs
    prov = dict(
        run_id=RUN_ID, env=ENV, arm=label, activation=name, act_kind=act.kind, twin=twin, seed=seed, lr=lr,
        n_tasks=n_tasks, epochs_per_task=epochs, steps_per_task=spt, batch=BATCH, n_images=N_IMAGES,
        dims=list(H.DIMS), optimizer='adam', status=status, n_rows=len(complete),
        divergence=div if div['diverged'] else None, removed_stale_divergence_json=removed_stale,
        **git_state(), spec_sha256=sha256_file(SPEC), checks_passed=not failed, failed_checks=failed,
        band_predicate_dtype=AC.BAND_PREDICATE_DTYPE,
        fn_mom_axes=dict(funcs=list(AC.FUNCS8), bands=list(AC.BANDS), moments=list(AC.MOM), shape='(T, layer, func, band, moment)'),
        code_sha256={'src/act_chimera_0913.py': sha256_file(ROOT / 'src' / 'act_chimera_0913.py'),
                     'src/act_chimera_rlmnist_0913.py': sha256_file(Path(__file__)),
                     **{k: v for k, v in carry.items()}},
        data_dir=str(H.DATA_DIR), data_sha256=mnist.sha256,
        subset_sha256=hashlib.sha256(np.sort(subset_idx(seed).numpy()).tobytes()).hexdigest(),
        rng_roles=['rl_subset', 'rl_labels', 'rl_batch', 'init'],
        interpreter=env, threads=threads if threads is not None else torch.get_num_threads(),
        machine=machine_info(device), wall_clock_s=wall, task_wall_s=ro.task_wall,
        steps_per_s=(len(complete) * spt / ro.task_wall[-1]) if ro.task_wall else None,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_max_memory_allocated=(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None),
        gpu_max_memory_reserved=(torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None),
        other_processes_before=procs_before, other_processes_after=other_processes(),
        nvidia_smi_before=smi_before, nvidia_smi_after=nvidia_smi(),
        readout_shapes=shapes, readout_file=f'readout_s{seed}.npz',
        endpoints=ep, checks=checks, window=list(WINDOW),
        floor=dict(sd=FLOOR_SD, family=FLOOR_FAMILY, z=FLOOR_Z, margin=FLOOR_MARGIN,
                   F_seed=floor_seed(seed) if n_tasks >= WINDOW[1] else None),
    )
    (out / 'provenance.json').write_text(json.dumps(prov, indent=2, default=str))
    summary = dict(arm=label, seed=seed, status=status, n_rows=len(complete), wall_clock_s=wall,
                   out=str(out), endpoints=ep, g1=checks.get('g1_rl'), checks_passed=not failed)
    if failed:                                                 # §7: 書いてから落とす（G1-RL の失敗は RL を止める）
        raise AssertionError(('per-run checks failed (outputs and provenance written)', label, seed, failed))
    return summary


def merge_arm(outdir: Path, label: str, n_tasks: int = N_TASKS, seeds=SEEDS) -> dict:
    """seed ごとの csv を ``outdir/{label}/per_task.csv`` に seed 順で結合する（§8）。行数は seed ごとに
    n_tasks、INCOMPLETE なら divergence.json の first_nonfinite_task − 1（task 1 で発散なら 0 行）と照合する。
    各行の seed 列 = そのディレクトリの seed、task 列 = 1..行数（順に）であることも照合する（取り違えた csv を通さない）。
    行は文字列のまま写す。"""
    base = Path(outdir) / label
    header, lines, counts = None, [], {}
    for s in seeds:
        d = base / f's{s}'
        p = d / 'per_task.csv'
        assert p.exists(), f'missing {p}'
        txt = p.read_text().splitlines()
        assert txt, f'empty {p} (not even a header)'
        if header is None:
            header = txt[0]
        assert txt[0] == header, (p, txt[0], header)
        body = txt[1:]
        div = d / 'divergence.json'
        if div.exists():
            j = json.loads(div.read_text())
            expect = j['first_nonfinite_task'] - 1
            counts[s] = dict(rows=len(body), expect=expect, status='INCOMPLETE')
        else:
            expect = n_tasks
            counts[s] = dict(rows=len(body), expect=expect, status='COMPLETE')
        assert len(body) == expect, (label, s, len(body), expect)
        cols = header.split(',')
        i_seed, i_task, i_arm = cols.index('seed'), cols.index('task'), cols.index('arm')
        got = [(int(b.split(',')[i_seed]), int(b.split(',')[i_task]), b.split(',')[i_arm]) for b in body]
        assert got == [(s, t, label) for t in range(1, len(body) + 1)], (label, s, got[:3])
        lines += body
    (base / 'per_task.csv').write_text(header + '\n' + ''.join(l + '\n' for l in lines))
    return dict(arm=label, n_rows=len(lines), per_seed=counts, path=str(base / 'per_task.csv'))


def twin_differs(rows_ref: list, rows_twin: list, col: str = 'online_acc') -> dict:
    """S18 の本走後の assert: 双子の acc 系列が LR と 1 タスク以上で異なること（揃っていたら σ_traj は
    NOT_DETERMINED_TWIN）。"""
    a = {int(r['task']): float(r[col]) for r in rows_ref}
    b = {int(r['task']): float(r[col]) for r in rows_twin}
    common = sorted(set(a) & set(b))
    n_diff = sum(a[t] != b[t] for t in common)
    return dict(n_common=len(common), n_diff=n_diff, first_diff_task=next((t for t in common if a[t] != b[t]), None),
                ok=n_diff >= 1)


# --------------------------------------------------------------------------
# G1-RL のスモーク（§2.3・§6）: GPU で LR s0 t1–2 の 40 セル、決定性、変異対照
# --------------------------------------------------------------------------

def g1_smoke(device: torch.device, outdir: Path = SMOKE_OUT, epochs: int = EPOCHS) -> dict:
    """G1-RL のスモーク（§2.3・§6）。**GPU だけ**（0906 の参照は GPU で作られた。CPU では文字列一致を求められないので
    CPU で呼ぶと例外にする。CPU の新しい側の決定性は ``determinism`` が別に検査する）。
    1. LR s0 t1–2 を回し、40 セルが参照と文字列一致（件数ガード 40）。
    2. 対照: 初期値 b1[0] += 1e−3 で 1 タスク → 1 セル以上が不一致（追補 2-1: 旧来の W1[0, 0] は画素 0 が常に 0 なので
       効かず、スモークで 0/20 だった。b1[0] は全標本で生きている）。"""
    if device.type != 'cuda':
        raise ValueError(f'G1-RL is a GPU check (0906 reference was made on GPU); got device={device} (§2.3)')
    out = Path(outdir) / 'g1'
    res = {}
    mnist = load_mnist(device)
    r1 = run_job('LR', 0, device=device, outdir=out / 'run1', n_tasks=2, epochs=epochs, mnist=mnist)
    g1 = g1_rl_compare(out / 'run1' / 'LR' / 's0' / 'per_task.csv', seeds=(0,), tasks=(1, 2), allow_missing=True)
    g1['gating'] = 'G1-RL'
    res['g1'] = g1
    res['g1_pass'] = bool(g1['n_cells'] == G1_SMOKE_CELLS and g1['n_mismatch'] == 0)
    ctl = run_job('LR', 0, device=device, outdir=out / 'ctl_b1_0_1e-3', n_tasks=1, epochs=epochs, mnist=mnist,
                  ro_cls=PerturbInitReadout)
    ga = g1_rl_compare(out / 'ctl_b1_0_1e-3' / 'LR' / 's0' / 'per_task.csv', seeds=(0,), tasks=(1,))
    # 対照の走は ``g1_gating`` で gating にならない（読み出しが PerturbInitReadout）ので、食い違っても run_job は落ちない
    res['control_b1_0_plus_1e-3'] = dict(perturbed='b1[0] += 1e-3 (params[1][0])', n_cells=ga['n_cells'],
                                         n_mismatch=ga['n_mismatch'], in_run_gating=(ctl.get('g1') or {}).get('gating'),
                                         fails_as_required=bool(ga['n_cells'] == len(G1_COLS) and ga['n_mismatch'] >= 1))
    res['runs'] = [r1, ctl]
    res['pass_'] = bool(res['g1_pass'] and res['control_b1_0_plus_1e-3']['fails_as_required'])
    (out / 'g1_smoke.json').write_text(json.dumps(res, indent=2, default=str))
    return res


def determinism(device: torch.device, outdir: Path = SMOKE_OUT, epochs: int = EPOCHS) -> dict:
    """新しい側の決定性（§2.3）: **選んだ device で** LR s0 の t1–2 を 2 回回し、t2 終端の params の sha256 が一致すること。
    変異対照: 双子 k = 0（初期値の b1[0] に +1e−6・追補 2-2）の走で、t2 終端の **b1[0] を除いた** params の sha256
    （S18b）が LR と変わること。全 params の sha256 は摂動した要素そのものを含むので、軌道が変わらなくても変わり、
    対照にならない（スモークで W1[0, 0] の双子がそれで空虚に通った）。G1 のセルは触らない。"""
    out = Path(outdir) / f'determinism_{device.type}'
    mnist = load_mnist(device)
    runs = []
    for tag, label in (('run1', 'LR'), ('run2', 'LR'), ('ctl_twin', 'LRtw0')):
        runs.append(run_job(label, 0, device=device, outdir=out / tag, n_tasks=2, epochs=epochs, mnist=mnist))
    sha = {tag: load_readout(out / tag / label / 's0' / 'readout_s0.npz')['param_sha']
           for tag, label in (('run1', 'LR'), ('run2', 'LR'), ('ctl_twin', 'LRtw0'))}
    t2 = {k: (str(v[-1]) if v.shape == (2,) else None) for k, v in sha.items()}
    excl = {}
    for tag, label in (('run1', 'LR'), ('run2', 'LR'), ('ctl_twin', 'LRtw0')):
        prov = json.loads((out / tag / label / 's0' / 'provenance.json').read_text())
        lst = ((prov.get('checks') or {}).get('s18b') or {}).get('sha_excl_b1', {}).get('0') or []
        excl[tag] = lst[-1] if len(lst) == 2 else None
    res = dict(device=device.type, sha_t2=t2, sha_excl_b1_0_t2=excl,
               determinism=dict(pass_=bool(t2['run1'] is not None and t2['run1'] == t2['run2']
                                           and excl['run1'] is not None and excl['run1'] == excl['run2'])),
               control_twin_b1_0=dict(perturbed='LRtw0: b1[0] + 1e-6', compared='sha256 of params with b1[0] set to 0 (S18b)',
                                      full_sha_differs=bool(t2['ctl_twin'] is not None and t2['ctl_twin'] != t2['run1']),
                                      fails_as_required=bool(excl['ctl_twin'] is not None and excl['run1'] is not None
                                                             and excl['ctl_twin'] != excl['run1'])),
               runs=runs)
    res['pass_'] = bool(res['determinism']['pass_'] and res['control_twin_b1_0']['fails_as_required'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'determinism.json').write_text(json.dumps(res, indent=2, default=str))
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', default=None, help='腕ラベル（LR, ELU1, SMAXH, SMINH, LRtw0, VMIN, ...）')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--tasks', type=int, default=N_TASKS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--lr', type=float, default=LR_RATE)
    ap.add_argument('--device', default=None, choices=['cpu', 'cuda'],
                    help='スモークの実測で決める（§2.3 未決 5）。auto は使わない')
    ap.add_argument('--threads', type=int, default=1, help='torch.set_num_threads（CPU は 1・§2.3）')
    ap.add_argument('--outdir', default=None, help='本走は results/act_chimera_0913/rlmnist（検査の既定は _smoke_）')
    ap.add_argument('--merge', action='store_true', help='seed ごとの csv を結合する（--arm）')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='--merge の seed')
    ap.add_argument('--g1-smoke', action='store_true', help='G1-RL（GPU・LR s0 t1–2・40 セル）と初期値の対照')
    ap.add_argument('--determinism', action='store_true', help='選んだ device での決定性（t2 の sha256）と双子の対照')
    ap.add_argument('--g1-full', action='store_true', help='GPU 本走の G1-RL（LR s0–2 × 50 タスク = 3000 セル）')
    args = ap.parse_args(argv)

    if args.merge:
        assert args.arm, '--merge needs --arm'
        print(json.dumps(merge_arm(Path(args.outdir or DEFAULT_OUT), args.arm, args.tasks,
                                   [int(s) for s in args.seeds.split(',')]), indent=2))
        return
    if args.g1_full:
        res = g1_full(Path(args.outdir or DEFAULT_OUT))
        out = Path(args.outdir or DEFAULT_OUT) / 'g1_full.json'
        out.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != 'per_seed'}, indent=2, default=str))
        if not res['pass_']:
            raise SystemExit('G1-RL (full, 3000 cells) failed: RL is invalid (§7)')
        return
    assert args.device is not None, '--device cpu|cuda is required'
    torch.set_num_threads(args.threads)
    device = H.setup(args.device)
    if args.g1_smoke or args.determinism:
        outdir = Path(args.outdir or SMOKE_OUT)
        res = g1_smoke(device, outdir, args.epochs) if args.g1_smoke else determinism(device, outdir, args.epochs)
        print(json.dumps({k: v for k, v in res.items() if k != 'runs'}, indent=2, default=str))
        if not res['pass_']:                              # §7: G1-RL・決定性の失敗は RL を止める（非 0 で終わる）
            raise SystemExit(('G1-RL smoke' if args.g1_smoke else 'determinism') + ' failed (§2.3・§7)')
        return
    assert args.arm is not None and args.seed is not None, '--arm and --seed are required'
    if args.outdir is None and (args.tasks != N_TASKS or args.epochs != EPOCHS or args.lr != LR_RATE):
        ap.error('--outdir is required for shortened / non-protocol runs (they must not land in the main results dir)')
    if args.outdir is None and args.device == 'cuda':
        # 追補 2-3: RL の本走は全腕 CPU。GPU の走（G1-RL の別検査）が本走の rlmnist/{ARM}/s{seed} に落ちると、LR と双子の
        # device が混ざる（S18b が空虚に生きる）。launcher の rl_job は常に --outdir を渡す
        ap.error('--outdir is required with --device cuda (the separate GPU G1-RL check must not land in the main '
                 'CPU results dir; addendum 2-3)')
    t0 = time.time()
    s = run_job(args.arm, args.seed, device=device, outdir=Path(args.outdir or DEFAULT_OUT), n_tasks=args.tasks,
                epochs=args.epochs, lr=args.lr, threads=args.threads)
    print(f"[{time.time() - t0:7.1f}s] {args.arm:<6} seed={args.seed} {s['status']} rows={s['n_rows']} "
          f"-> {s['out']}", flush=True)
    if s['g1'] is not None:
        print(f"  G1-RL ({s['g1']['gating']}): {s['g1']['n_mismatch']} / {s['g1']['n_cells']} cells mismatch")
    if s['endpoints'] is not None:
        e = s['endpoints']
        print(f"  Y_pt={e['Y_pt']:.3f} Y_logit={e['Y_logit']:.4f} AT_FLOOR={e['at_floor']}")


if __name__ == '__main__':
    main()
