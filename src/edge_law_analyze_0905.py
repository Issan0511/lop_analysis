"""edge_law_analyze_0905 — 命題 1–5「上端則」の登録解析（spec `specs/spec_edge_law_0905.md` §4）。

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_analyze_0905 --selftest
    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.edge_law_analyze_0905 --outdir results/edge_law_0905

正本は `configs/edge_law_0905.yaml`（窓・除外・$C$・bootstrap・ゲート）。本モジュールは
起動時に config の定数と自分の定数を突き合わせ、食い違えば例外にする（黙って別の窓で
判定しないため）。判定は spec §4 の 4.1-b/c/d/e・4.2-a..e・4.3-a/b/c・4.4-a/b/c・
4.5-a..i・4.6 のみ。§2 は事後・未登録なので判定に使わない（参照値の再現確認だけに使う）。

**依存は numpy だけ**（この checkout に scipy は無い）。Spearman・KS・歪度・最小二乗・
緩和フィットはすべて自前。npz は列単位で遅延読み（腕あたり 50–190 MB あるため）。

窓（spec §3.5）:
  - `_window_indices` はタスク終端記録（`step % 10000 == 0` かつ `step > 0`）しか拾わない。
    したがって主窓 = タスク 451–500 は **50 記録**、lag 窓 = 351–400 も 50 記録。
  - 15M 腕は主窓をタスク 1451–1500 にし、451–500 も併記する（`window` 列に書く）。
  - ユニット別の値は窓内記録の**平均**（§2 の参照値 0.112 / −3.598 / B 1.637 がこの
    約束で厳密に再現することを `--selftest` が毎回確かめる）。

支持の復元（spec §3.4・§5 S-support）: 入力は {0,1}^20 で末尾 5 ビットが自由。中心化は
`_refresh_fixed_offset` が全座標一律の offset 0.5γ（γ ≈ 2.2e−4）を引くだけなので、自由
ビットの寄与は中心化の値に依らず ±w_j/2 になる。すなわち

    z_p = z̄ + Σ_j s_j w_j / 2,   s ∈ {±1}^5,   半幅 = z_max − z̄ = ½ Σ_j |w_j|

（実機で `zmax − zbar − ½Σ|w_free|` の最大絶対誤差 8.9e−16、`zmin = 2z̄ − zmax` も同誤差で
一致することを確認済み。`layer1_w_free` は W[:, :, 15:20]、すなわち自由 5 ビットの重み）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, load_config
from .mlp2_phase0b import _window_indices

# ---------------------------------------------------------------------------
# 登録定数（spec §3.5 / §3.6・config が正本。_check_config が突き合わせる）
# ---------------------------------------------------------------------------
CONFIG = Path(ROOT) / "configs" / "edge_law_0905.yaml"
PERIOD = 10_000
C_CONST = 11.497681                      # E|x_c|^2 + 1 = 3.041^2 + 5*0.25 + 1（閉形式定数）
ALIVE_DENOM = 0.25                       # ALIVE = layer1_denom(窓平均) > 0.25
ALIVE_HALF_WIDTH = 0.25                  # 副次規則（REPORT）: (zmax - zbar) > 0.25
ALIVE_SECONDARY_N_GAP = 0.03             # 副次規則の n が 3% 以上違ったら記録する
TAIL_5M = (451, 500)
TAIL_15M = (1451, 1500)
LAG_5M = (351, 400)
LAG_15M = (1351, 1400)
SETTLE = ((301, 350), (376, 425), (451, 500))          # G2
TRAJ = ((51, 100), (251, 300), (451, 500))             # 「タスク 100→300→500」の 50 タスク窓
BOOT_N = 2000
BOOT_SEED = 20260905
BOOT_LEVEL = 0.95
DEATH_GAP_MAX = 0.10                     # G4
NAN_SEED_DROP_MAX = 2                    # G6
RUNAWAY_ABS_ZBAR = 50.0                  # G6
G1_RATIO_BAND = (0.5, 2.0)
G1_MIN_ZMAX_MOVE = 0.3

COMMITTED = {                            # 再走しない committed 対照（config の controls_committed）
    "LR_1216": "results/p3_extend_0902/logs",
    "E_1216": "results/p3_extend_0902/logs",
    "E_a0p1_1216": "results/gate_dial_0902/logs",
    "E_a0p01_1216": "results/gate_dial_0902/logs",
    "LR_a0p01_1216": "results/gate_dial_0902/logs",
}

VERDICT_FIELDS = ["arm", "judgment", "role", "statistic", "window", "exclusion",
                  "n", "death_rate", "n_seeds_dropped", "point", "ci_lo", "ci_hi",
                  "gate_G1", "gate_G2", "gate_G3", "gate_G4", "gate_G5",
                  "gate_G6", "label", "note"]

# --- 本モジュールが npz に期待する列（runner との契約・spec §3.4）--------------
# 宿主 `write_arm_logs_dial` が seed 別に切って書くので、per-seed npz での形は
# ユニット別列 (n_rec, h)・`layer1_w_free` (n_task, h, 5)・モーメント (n_mom, h)。
# recorder 側の (n_rec, R, h) 形も `ArmLog._drop_seed_axis` が受ける。
REQUIRED_RUN_COLUMNS = ("step", "unfit")
REQUIRED_UNIT_COLUMNS = ("layer1_zbar", "layer1_zmean", "layer1_zmax",
                         "layer1_dzbar", "layer1_denom", "layer1_v_unit",
                         "layer1_w_norm", "layer1_mob", "layer1_absmob",
                         "layer1_M", "layer1_B", "layer1_p_hat")
NEW_UNIT_COLUMNS = ("layer1_zmin",)                       # (n_rec, h) 1000 step ごと
NEW_AUX_COLUMNS = (("layer1_w_free", "layer1_w_free_step"),      # (n_task, h, 5)
                   ("layer1_m_phi2", "layer1_moment_step"),      # (n_mom, h)
                   ("layer1_m_dphi2", "layer1_moment_step"),
                   ("layer1_m_phidphi", "layer1_moment_step"),
                   ("layer1_m_dphiddphi", "layer1_moment_step"))
PAYLOAD_KEYS = ("init_hook", "init_hook_arg", "lr_used", "freeze_v", "batch_mode")


def expected_columns() -> dict:
    """runner 側の照合用: 本モジュールが読む列名を 1 か所にまとめて返す。"""
    return {"run": list(REQUIRED_RUN_COLUMNS),
            "unit": list(REQUIRED_UNIT_COLUMNS),
            "new_unit": list(NEW_UNIT_COLUMNS),
            "new_aux": [list(pair) for pair in NEW_AUX_COLUMNS],
            "payload": list(PAYLOAD_KEYS)}


def check_arm_columns(arm: "ArmLog") -> dict:
    """腕のログに何が有って何が無いか（判定の NOT_DETERMINED 理由になる）。"""
    files = set(arm.files()) if arm.available else set()
    aux = {key for key, step in NEW_AUX_COLUMNS} | {
        step for key, step in NEW_AUX_COLUMNS}
    return {"missing_required": [k for k in REQUIRED_RUN_COLUMNS
                                 + REQUIRED_UNIT_COLUMNS if k not in files],
            "missing_new": [k for k in tuple(NEW_UNIT_COLUMNS) + tuple(sorted(aux))
                            if k not in files],
            "missing_payload": [k for k in PAYLOAD_KEYS if k not in files]}


# 1-b 列別パリティ（spec §4.1-b）
MIRROR_NEG_COLUMNS = ("layer1_zbar", "layer1_dzbar", "layer1_zmean",
                      "layer1_v_unit", "layer1_M", "layer1_B")
MIRROR_SAME_COLUMNS = ("layer1_w_norm", "layer1_denom", "layer1_mob",
                       "layer1_absmob")
MIRROR_SEEDS_REQUIRED = 10               # spec §4.1-b「PASS = 10/10 seed で全列」

# 5-g の 5 族（spec §4.5-g）
FAMILY_ARMS = {
    "elu": ("Enull_1216", "E_a0p5_1216", "E_a2_1216", "E_a4_1216"),
    "shelf": ("SH_d0p5_1216", "SH_d1_1216", "SH_d2_1216", "SH_d3_1216",
              "SH_d30_1216"),
    "leaky": ("LRnull_1216",),
    "softplus": ("SP_1216",),
    "tanh": ("TH_1216",),
}


class AnalysisError(RuntimeError):
    """登録した約束からのずれ（config 不一致など）。黙って別解析に落ちない。"""


# ---------------------------------------------------------------------------
# 数値ユーティリティ（scipy 無し）
# ---------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均順位（scipy.stats.rankdata の 'average' と同じ）。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 順位相関（同順位は平均順位）。有効標本 < 3 なら NaN。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float("nan") if den == 0.0 else float((ra * rb).sum() / den)


def ks_d(a: np.ndarray, b: np.ndarray) -> float:
    """2 標本 Kolmogorov–Smirnov 統計量 D。"""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def skewness(x: np.ndarray) -> float:
    """標本歪度 g1。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    d = x - x.mean()
    m2 = float((d ** 2).mean())
    m3 = float((d ** 3).mean())
    return float("nan") if m2 <= 0 else m3 / m2 ** 1.5


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """単回帰 y = a + b x。返り値 (切片 a, 傾き b)。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(x)
    if int(m.sum()) < 2:
        return float("nan"), float("nan")
    A = np.stack([np.ones(int(m.sum())), x[m]], axis=1)
    coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return float(coef[0]), float(coef[1])


def boot_draws(n_seeds: int, seed: int = BOOT_SEED, n: int = BOOT_N) -> np.ndarray:
    """seed bootstrap の抽選表（腕を跨いでも同じ表を使う＝共通復元抽出）。"""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_seeds), size=(int(n), int(n_seeds)))


def boot_ci(stat_fn, n_seeds: int, *, seed: int = BOOT_SEED,
            draws: np.ndarray | None = None) -> tuple[float, float, float]:
    """点推定（全 seed）と percentile CI。stat_fn は seed 添字の配列を取る。"""
    if draws is None or int(np.asarray(draws).shape[1]) != int(n_seeds):
        # G6 で seed を落とすと腕ごとに再抽出単位の数が変わりうる。共通表が
        # 合わないときは黙って添字はみ出しを起こさず、その腕の数で引き直す。
        draws = boot_draws(n_seeds, seed)
    point = float(stat_fn(np.arange(n_seeds)))
    vals = np.array([stat_fn(d) for d in draws], dtype=np.float64)
    if not np.isfinite(vals).any():
        return point, float("nan"), float("nan")
    lo_q = 100.0 * (1.0 - BOOT_LEVEL) / 2.0
    lo, hi = np.nanpercentile(vals, [lo_q, 100.0 - lo_q])
    return point, float(lo), float(hi)


def nanmedian(x: np.ndarray) -> float:
    """全 NaN・空でも警告を出さない中央値（判定は NaN のまま下流に伝える）。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def ci_within(ci: tuple[float, float], lo: float, hi: float) -> bool:
    """CI が [lo, hi] に内包（同等性判定）。"""
    return bool(np.isfinite(ci[0]) and np.isfinite(ci[1])
                and ci[0] >= lo and ci[1] <= hi)


def ci_excludes_zero(ci: tuple[float, float]) -> bool:
    return bool(np.isfinite(ci[0]) and np.isfinite(ci[1])
                and (ci[0] > 0.0 or ci[1] < 0.0))


# ---------------------------------------------------------------------------
# 活性化（numpy 版・5-b の数値平衡専用。学習には使わない）
# ---------------------------------------------------------------------------
def _parse_depth(name: str, key: str) -> float:
    tail = name.split(key, 1)[1]
    return float(tail.replace("p", "."))


def act_numpy(activation: str, dial: float, cfg_act: dict | None = None):
    """(φ, φ′) を numpy で返す。spec §3.2 の式を逐語で写す。"""
    a = float(dial)
    meta = (cfg_act or {}).get(activation, {}) if cfg_act else {}
    if activation == "leaky_relu":
        return (lambda z: np.where(z > 0, z, a * z),
                lambda z: np.where(z > 0, 1.0, a))
    if activation == "flip_leaky":
        return (lambda z: np.where(z < 0, z, a * z),
                lambda z: np.where(z < 0, 1.0, a))
    if activation == "elu":
        return (lambda z: np.where(z > 0, z, a * (np.expm1(np.minimum(z, 0.0)))),
                lambda z: np.where(z > 0, 1.0, a * np.exp(np.minimum(z, 0.0))))
    if activation.startswith("shelf_leaky_d"):
        d = float(meta.get("depth", _parse_depth(activation, "shelf_leaky_d")))
        return (lambda z: np.where(z > -d, z, a * (z + d) - d),
                lambda z: np.where(z > -d, 1.0, a))
    if activation.startswith("steep_shelf_d"):
        d = float(meta.get("depth", _parse_depth(activation, "steep_shelf_d")))
        s = float(meta.get("lower_slope", 2.0))
        return (lambda z: np.where(z > -d, z, s * (z + d) - d),
                lambda z: np.where(z > -d, 1.0, s))
    if activation == "softplus_b":
        return (lambda z: np.logaddexp(0.0, z),
                lambda z: 1.0 / (1.0 + np.exp(-z)))
    if activation == "tanh_b":
        return (lambda z: np.tanh(z),
                lambda z: 1.0 - np.tanh(z) ** 2)
    raise AnalysisError(f"activation {activation!r} has no numpy form")


def act_depth(activation: str, cfg_act: dict | None = None) -> float:
    """棚族の折れ目の深さ d（棚以外は NaN）。"""
    meta = (cfg_act or {}).get(activation, {}) if cfg_act else {}
    if "depth" in meta:
        return float(meta["depth"])
    for key in ("shelf_leaky_d", "steep_shelf_d"):
        if activation.startswith(key):
            return _parse_depth(activation, key)
    return float("nan")


# ---------------------------------------------------------------------------
# ログアクセス
# ---------------------------------------------------------------------------
class ArmLog:
    """1 腕ぶんの per-seed npz（または合成 dict）への遅延アクセス。

    on-disk は `write_arm_logs_dial` が seed 別に切って書くので、ユニット別列は
    ``(n_rec, h)``、``layer1_w_free`` は ``(n_task, h, 5)``。recorder が持つ
    ``(n_rec, R, h)`` 形（合成データや将来の連結ログ）も受け取れるように、列ごとに
    次元数で判別して seed 軸を落とす。
    """

    def __init__(self, name: str, *, logdir: Path | None = None,
                 data: dict[int, dict] | None = None, seeds=range(10),
                 meta: dict | None = None):
        self.name = str(name)
        self.logdir = Path(logdir) if logdir is not None else None
        self.data = data
        self.meta = dict(meta or {})
        self._npz: dict[int, object] = {}
        self._cache: dict[tuple, np.ndarray] = {}
        if data is not None:
            self.seeds = sorted(int(s) for s in data)
        else:
            self.seeds = [int(s) for s in seeds
                          if (self.logdir / f"{self.name}_seed{int(s)}.npz").exists()]
        # G6（spec §3.6・§4.6）: NaN を出した seed は**落として**判定する。
        # `seeds` は生の seed 一覧（1-b のバイト比較はこちらを使う）、
        # `kept_seeds` は判定統計量が使う生き残り。`set_drops` が窓を見て決める。
        self.kept_seeds = list(self.seeds)
        self.dropped_seeds: list[int] = []

    # -- 基本 ---------------------------------------------------------------
    @property
    def available(self) -> bool:
        return len(self.seeds) > 0

    def _src(self, seed: int):
        if self.data is not None:
            return self.data[int(seed)]
        if int(seed) not in self._npz:
            path = self.logdir / f"{self.name}_seed{int(seed)}.npz"
            self._npz[int(seed)] = np.load(path, allow_pickle=True)
        return self._npz[int(seed)]

    def files(self, seed: int | None = None) -> list[str]:
        s = self.seeds[0] if seed is None else int(seed)
        src = self._src(s)
        return list(src.files) if hasattr(src, "files") else list(src)

    def has(self, key: str) -> bool:
        return key in self.files()

    def raw(self, seed: int, key: str) -> np.ndarray:
        """dtype をそのまま返す（1-b のバイト比較用）。"""
        return np.asarray(self._src(int(seed))[key])

    def col(self, seed: int, key: str) -> np.ndarray:
        """float64 に上げた列。seed 軸を持つ形なら落とす。"""
        ck = ("col", int(seed), key)
        if ck not in self._cache:
            arr = np.asarray(self._src(int(seed))[key])
            if arr.dtype.kind in "fiub":
                arr = arr.astype(np.float64)
            self._cache[ck] = self._drop_seed_axis(key, arr, int(seed))
        return self._cache[ck]

    def _drop_seed_axis(self, key: str, arr: np.ndarray, seed: int) -> np.ndarray:
        if arr.ndim == 3 and key != "layer1_w_free":
            return arr[:, self.seeds.index(seed), :]
        if arr.ndim == 4 and key == "layer1_w_free":
            return arr[:, self.seeds.index(seed), :, :]
        return arr

    def payload(self, key: str, default=None):
        """走のスカラ payload（`lr_used` / `freeze_v` / `init_hook` …）。"""
        if not self.available or key not in self.files():
            return default
        val = np.asarray(self._src(self.seeds[0])[key])
        if val.dtype.kind in "SU":
            return str(val)
        if val.dtype == bool:
            return bool(val)
        return float(val) if val.ndim == 0 else val

    def steps(self, seed: int) -> np.ndarray:
        return np.asarray(self._src(int(seed))["step"]).astype(np.int64)

    @property
    def max_step(self) -> int:
        return int(self.steps(self.seeds[0])[-1]) if self.available else 0

    def close(self) -> None:
        for handle in self._npz.values():
            if hasattr(handle, "close"):
                handle.close()
        self._npz.clear()
        self._cache.clear()

    # -- 窓 -----------------------------------------------------------------
    def window_index(self, seed: int, window) -> np.ndarray:
        return _window_indices(self.steps(seed), PERIOD, list(window))

    def window_count(self, window) -> int:
        """窓に入るタスク終端記録の数（腕の地平線が窓に届かなければ 0）。"""
        return min(len(self.window_index(s, window)) for s in self.kept_seeds)

    def unit_records(self, key: str, window) -> np.ndarray:
        """(S, n_rec_in_window, h)。窓内のタスク終端記録をそのまま返す。"""
        ck = ("recs", key, tuple(window), tuple(self.kept_seeds))
        if ck not in self._cache:
            out = [self.col(s, key)[self.window_index(s, window)]
                   for s in self.kept_seeds]
            n = min(a.shape[0] for a in out)
            self._cache[ck] = np.stack([a[:n] for a in out], axis=0)
        return self._cache[ck]

    def unit_window(self, key: str, window, reduce: str = "mean") -> np.ndarray:
        """(S, h)。ユニット別の値は窓内記録の平均（spec §3.5）。"""
        recs = self.unit_records(key, window)
        if recs.shape[1] == 0:                       # 地平線が窓に届いていない
            return np.full((recs.shape[0], recs.shape[2]), np.nan)
        if reduce == "mean":
            return recs.mean(axis=1)
        if reduce == "std":
            return recs.std(axis=1, ddof=0)
        if reduce == "median":
            return np.median(recs, axis=1)
        raise AnalysisError(f"unknown reduce {reduce!r}")

    def unit_at_step(self, key: str, step: int = 0) -> np.ndarray:
        """(S, h)。step 0 の記録＝初期値（spec §3.5）。"""
        out = []
        for s in self.kept_seeds:
            idx = int(np.flatnonzero(self.steps(s) == int(step))[0])
            out.append(self.col(s, key)[idx])
        return np.stack(out, axis=0)

    def run_window(self, key: str, window) -> np.ndarray:
        """(S,) 走レベル列（`unfit` など）の窓平均。"""
        out = []
        for s in self.kept_seeds:
            idx = self.window_index(s, window)
            out.append(float(np.asarray(self.col(s, key))[idx].mean()))
        return np.array(out, dtype=np.float64)

    def aux_steps(self, seed: int, step_key: str) -> np.ndarray:
        """補助列の step 列（`layer1_w_free_step` / `layer1_moment_step`）。"""
        return np.asarray(self._src(int(seed))[step_key]).astype(np.int64)

    def aux_window(self, key: str, step_key: str, window) -> np.ndarray:
        """独自 step 列を持つ補助列（`layer1_w_free` / モーメント）の窓抜き。"""
        out = []
        for s in self.kept_seeds:
            steps = self.aux_steps(s, step_key)
            idx = _window_indices(steps, PERIOD, list(window))
            out.append(self.col(s, key)[idx])
        n = min(a.shape[0] for a in out)
        return np.stack([a[:n] for a in out], axis=0)

    # -- 除外・発散 ----------------------------------------------------------
    def alive(self, window) -> np.ndarray:
        return self.unit_window("layer1_denom", window) > ALIVE_DENOM

    def alive_secondary(self, window) -> np.ndarray:
        """副次除外規則（spec §3.5）: 半幅 ``z_max − z̄ > 0.25``。REPORT 用。"""
        return (self.unit_window("layer1_zmax", window)
                - self.unit_window("layer1_zbar", window)) > ALIVE_HALF_WIDTH

    def death_rate(self, window) -> float:
        return float(1.0 - self.alive(window).mean())

    def nan_seeds(self, window) -> list[int]:
        """G6: 窓内に NaN を出した seed（**生の** seed 一覧の上で数える）。

        ``kept_seeds`` を経由する ``unit_window`` は使わない（落とした seed を
        自分自身の判定材料にすると、2 回目の呼び出しで 0 個になってしまう）。
        """
        bad = []
        for s in self.seeds:
            try:
                idx = self.window_index(s, window)
                zb = self.col(s, "layer1_zbar")[idx]
                zm = self.col(s, "layer1_zmax")[idx]
            except (KeyError, IndexError):
                continue
            if (not np.isfinite(zb).all()) or (not np.isfinite(zm).all()):
                bad.append(int(s))
        return bad

    def set_drops(self, window) -> list[int]:
        """G6 の seed 落とし（spec §3.6・§4.6）を確定させる。

        主窓で NaN を出した seed を ``kept_seeds`` から外す。落とした数が
        ``NAN_SEED_DROP_MAX`` を超える腕は ``g6_divergence`` が ``NOT_RUN`` に
        するので、ここでは落とさない（統計量を出す腕ではなくなる）。
        """
        try:
            bad = self.nan_seeds(window)
        except Exception:                                   # noqa: BLE001
            bad = []
        self.dropped_seeds = list(bad)
        if bad and len(bad) <= NAN_SEED_DROP_MAX:
            self.kept_seeds = [s for s in self.seeds if s not in set(bad)]
        else:
            self.kept_seeds = list(self.seeds)
        self._cache.clear()
        return self.dropped_seeds

    def runaway(self, window) -> bool:
        zb = self.unit_window("layer1_zbar", window)
        al = self.alive(window)
        if not al.any():
            return False
        return bool(abs(float(np.median(zb[al]))) > RUNAWAY_ABS_ZBAR)


class Ctx:
    """腕の集合＋窓の決め方。判定関数はすべてこれを受け取る（合成データも同じ口）。"""

    def __init__(self, arms: dict[str, ArmLog], cfg: dict | None = None):
        self.arms = dict(arms)
        self.cfg = cfg or {}
        self.cfg_act = (cfg or {}).get("activation", {})
        self.notes: list[str] = []
        # G6 の seed 落とし（spec §3.6・§4.6）は判定の**前**に 1 回だけ確定させる。
        # 以降のすべての統計量（pooled / median_stat_ci / boot_ci）は生き残った
        # seed だけを見る。落とした数は `n_seeds_dropped` 列に残る。
        for name, arm in self.arms.items():
            if arm.available:
                arm.set_drops(self.tail(name))

    def dropped(self, name: str) -> int:
        arm = self.arms.get(name)
        return len(arm.dropped_seeds) if arm is not None else 0

    def get(self, name: str) -> ArmLog | None:
        arm = self.arms.get(name)
        return arm if (arm is not None and arm.available) else None

    def tail(self, name: str):
        """主窓。15M **腕**だけ 1451–1500（spec §3.5）。

        committed 対照（`LR_1216` / `E_1216`）のログは 15M あるが、5M 腕
        （`LRnull` / `Enull`）の参照として使う登録上の地平線は 5M なので、
        `build_ctx` が `total_steps = 5,000,000` を meta に入れる。先頭 5M は
        `gate_dose_0830` と bit 一致（S-ext 済み）。15M 側の値は §2 の併記扱い。
        """
        arm = self.get(name)
        if arm is None:
            return TAIL_5M
        total = arm.meta.get("total_steps")
        total = int(total) if total is not None else arm.max_step
        if total >= 15_000_000:
            return TAIL_15M
        if total < 1_000_000:                        # FBLR_1216（500k = 50 タスク）
            last = total // PERIOD
            return (max(1, last - 9), last)
        return TAIL_5M

    def lag(self, name: str):
        return LAG_15M if self.tail(name) == TAIL_15M else LAG_5M

    def meta(self, name: str) -> dict:
        arm = self.arms.get(name)
        return dict(arm.meta) if arm is not None else {}

    def n_seeds(self, *names: str) -> int:
        """bootstrap の再抽出単位＝**生き残った** seed の数（G6 の落としを反映）。"""
        for name in names:
            arm = self.get(name)
            if arm is not None:
                return len(arm.kept_seeds)
        return 0

    def close(self) -> None:
        for arm in self.arms.values():
            arm.close()


# ---------------------------------------------------------------------------
# 行（verdict.csv）
# ---------------------------------------------------------------------------
def row(arm: str, judgment: str, role: str, statistic: str, window,
        exclusion: str, label: str, *, n: int = 0, death_rate: float = float("nan"),
        point: float = float("nan"), ci=(float("nan"), float("nan")),
        gates: dict | None = None, note: str = "",
        n_seeds_dropped: int = 0) -> dict:
    g = {f"G{i}": "NA" for i in range(1, 7)}
    g.update(gates or {})
    win = window if isinstance(window, str) else (
        "" if window is None else f"tasks {window[0]}-{window[1]}")
    return {"arm": arm, "judgment": judgment, "role": role, "statistic": statistic,
            "window": win, "exclusion": exclusion, "n": int(n),
            "death_rate": float(death_rate),
            "n_seeds_dropped": int(n_seeds_dropped), "point": float(point),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
            "gate_G1": g["G1"], "gate_G2": g["G2"], "gate_G3": g["G3"],
            "gate_G4": g["G4"], "gate_G5": g["G5"], "gate_G6": g["G6"],
            "label": label, "note": note}


def not_run(arm: str, judgment: str, role: str, statistic: str, label: str,
            reason: str) -> dict:
    return row(arm, judgment, role, statistic, None, "", label, note=reason)


# ---------------------------------------------------------------------------
# 共通の統計（プール中央値・対応差）
# ---------------------------------------------------------------------------
def common_seed_rows(a: "ArmLog", b: "ArmLog") -> tuple[list[int], np.ndarray, np.ndarray]:
    """2 腕の**同じ seed 番号**を突き合わせる行添字（G5 のペアと G6 の落としの両立）。

    G6 で落とす seed は腕ごとに違いうるので、位置での `[:n]` 切り詰めは
    (seed, unit) のペアを崩す。腕間比較はこの関数が返す添字で揃える。
    """
    sb = set(b.kept_seeds)
    common = [s for s in a.kept_seeds if s in sb]
    ia = np.array([a.kept_seeds.index(s) for s in common], dtype=int)
    ib = np.array([b.kept_seeds.index(s) for s in common], dtype=int)
    return common, ia, ib


def pooled(values: np.ndarray, mask: np.ndarray | None, seed_idx) -> np.ndarray:
    """seed 添字（復元抽出可）で (S,h) をプールした 1 次元標本。"""
    seed_idx = np.asarray(seed_idx, dtype=int)
    if mask is None:
        return values[seed_idx].reshape(-1)
    return np.concatenate([values[i][mask[i]] for i in seed_idx])


def pooled_median(values: np.ndarray, mask: np.ndarray | None):
    def _f(idx):
        x = pooled(values, mask, idx)
        return float(np.median(x)) if x.size else float("nan")
    return _f


def median_stat_ci(values: np.ndarray, mask: np.ndarray | None,
                   draws: np.ndarray | None = None):
    n = values.shape[0]
    point, lo, hi = boot_ci(pooled_median(values, mask), n, draws=draws)
    n_used = int(mask.sum()) if mask is not None else int(values.size)
    return point, (lo, hi), n_used


# ---------------------------------------------------------------------------
# ゲート（spec §3.6）
# ---------------------------------------------------------------------------
def g1_progress(ctx: Ctx, name: str) -> tuple[str, str]:
    """G1 進捗ゲート。返り値 (PASS|FROZEN|NA, 説明)。"""
    arm = ctx.get(name)
    if arm is None:
        return "NA", "arm missing"
    win = ctx.tail(name)
    al = arm.alive(win)
    wn = arm.unit_window("layer1_w_norm", win)
    if not al.any():
        return "FROZEN", "no ALIVE unit"
    med_w = float(np.median(wn[al]))
    family = str(arm.meta.get("family", ""))
    ref_name = {"leaky": "LRnull_1216", "elu": "Enull_1216"}.get(family)
    ref_val, ref_txt = float("nan"), ""
    if ref_name is not None:
        ref = ctx.get(ref_name) or ctx.get({"LRnull_1216": "LR_1216",
                                            "Enull_1216": "E_1216"}[ref_name])
        if ref is not None:
            rwin = ctx.tail(ref.name)
            ral = ref.alive(rwin)
            ref_val = float(np.median(ref.unit_window("layer1_w_norm", rwin)[ral]))
            ref_txt = f"family ref {ref.name} |w|={ref_val:.3f}"
    if not np.isfinite(ref_val):
        init_w = float(np.median(arm.unit_at_step("layer1_w_norm")))
        ref_val = 1.5 * init_w
        ref_txt = f"own init x1.5 = {ref_val:.3f}"
    ratio = med_w / ref_val if ref_val else float("nan")
    zt = float(np.median(arm.unit_window("layer1_zmax", win)[al]))
    z0 = float(np.median(arm.unit_at_step("layer1_zmax")))
    move = abs(zt - z0)
    ok = (G1_RATIO_BAND[0] <= ratio <= G1_RATIO_BAND[1]) and move >= G1_MIN_ZMAX_MOVE
    txt = (f"|w|={med_w:.3f} ratio={ratio:.3f} ({ref_txt}); "
           f"|dz_max|={move:.3f}")
    return ("PASS" if ok else "FROZEN"), txt


def g2_settled(stat_fn, ci_width: float) -> tuple[str, list[float]]:
    """G2 定着ゲート。stat_fn(window) -> 点推定。"""
    vals = []
    for w in SETTLE:
        try:
            vals.append(float(stat_fn(w)))
        except Exception:                                    # noqa: BLE001
            vals.append(float("nan"))
    if not np.isfinite(vals).all():
        return "NA", vals
    mono = (vals[0] <= vals[1] <= vals[2]) or (vals[0] >= vals[1] >= vals[2])
    drift = abs(vals[2] - vals[0])
    if mono and np.isfinite(ci_width) and drift > ci_width:
        return "NOT_SETTLED", vals
    return "PASS", vals


def g3_agreement(label_alive: str, label_all: str) -> str:
    return "PASS" if label_alive == label_all else "MISMATCH"


def g4_comparable(a: float, b: float) -> str:
    if not (np.isfinite(a) and np.isfinite(b)):
        return "NA"
    return "PASS" if abs(a - b) < DEATH_GAP_MAX else "NOT_COMPARABLE"


def g6_divergence(ctx: Ctx, name: str) -> tuple[str, str]:
    arm = ctx.get(name)
    if arm is None:
        return "NA", "arm missing"
    win = ctx.tail(name)
    bad = arm.nan_seeds(win)
    if len(bad) > NAN_SEED_DROP_MAX:
        return "NOT_RUN", f"{len(bad)} seeds with NaN"
    if arm.runaway(win):
        return "ARM_RUNAWAY", f"|median zbar| > {RUNAWAY_ABS_ZBAR}"
    return "PASS", (f"{len(bad)} NaN seeds" if bad else "no NaN")


def arm_gates(ctx: Ctx, name: str) -> dict:
    g1, g1txt = g1_progress(ctx, name)
    g6, g6txt = g6_divergence(ctx, name)
    return {"G1": g1, "G6": g6, "_g1": g1txt, "_g6": g6txt}


def blocked(gates: dict) -> str | None:
    """G1/G4/G6 で判定を落とすべきかどうか（spec §3.6）。

    G4（腕間比較で死亡率が 10 ポイント以上違う）は「その比較は `NOT_COMPARABLE`」
    なので、G4 を**計算した**判定は必ずここで止まる（計算しておいて読まない、が
    以前の穴だった）。
    """
    if gates.get("G6") == "NOT_RUN":
        return "G6: NOT_RUN"
    if gates.get("G1") == "FROZEN":
        return "G1: FROZEN"
    if gates.get("G1") == "NA":
        return "G1: arm missing"
    if gates.get("G4") == "NOT_COMPARABLE":
        return "G4: NOT_COMPARABLE"
    return None


# ---------------------------------------------------------------------------
# §4.1 命題 1
# ---------------------------------------------------------------------------
def _byte_neq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """要素ごとの**バイト**不一致マスク（``!=`` は ±0.0 に盲なので使わない）。"""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.size == 0:
        return np.zeros(0, dtype=bool)
    ua = np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(a.size, -1)
    ub = np.frombuffer(b.tobytes(), dtype=np.uint8).reshape(b.size, -1)
    return (ua != ub).any(axis=1)


def mirror_parity(x: np.ndarray, y: np.ndarray, *, flip: bool) -> dict:
    """S-mirror の列別パリティ規則（**runner と解析で共有する唯一の実装**）。

    規則（spec §4.1-b・§5）:

    - 期待値は ``np.negative(x)``（``0.0 - x`` は ``-0.0`` を ``+0.0`` に潰すので
      使わない — 零の符号ちょうどで比較が符号盲になる）。
    - 比較は**バイト**（``torch.equal`` / ``==`` は ±0.0 に盲）。
    - **登録された例外**: 両腕でちょうど ``0.0`` の要素は零の符号を問わず通す。
      ``dzbar`` は差 ``z̄_t − z̄_{t−1}`` なので、鏡像側でも ``a − a == +0.0`` に
      なる要素が実在し、``np.negative`` の期待値だけが ``−0.0`` になる。数を
      ``n_zero_sign_exceptions`` に残す（``p'+p == 1.0`` の例外数と同じ扱い）。
    - NaN は位置マスクだけを比べる（NaN を落とす経路が黙って記録を飛ばさない
      ようにするのが ``nan_pattern_equal`` の役目）。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return dict(pass_=False, flip=bool(flip), nan_pattern_equal=False,
                    n_records_compared=0, n_mismatch=int(max(x.size, y.size)),
                    n_zero_sign_exceptions=0,
                    note=f"shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}")
    nx, ny = np.isnan(x), np.isnan(y)
    nan_pattern = bool(np.array_equal(nx, ny))
    xf, yf = x[~nx], y[~ny]
    n_cmp = int(min(xf.size, yf.size))
    size_ok = bool(xf.size == yf.size == x.size - int(nx.sum()))
    if xf.size != yf.size:
        return dict(pass_=False, flip=bool(flip), nan_pattern_equal=nan_pattern,
                    n_records_compared=n_cmp, n_mismatch=int(abs(xf.size - yf.size)),
                    n_zero_sign_exceptions=0, note="NaN counts differ")
    target = np.negative(xf) if flip else xf
    neq = _byte_neq(target, yf)
    both_zero = (xf == 0.0) & (yf == 0.0) if flip else np.zeros(xf.shape, bool)
    zero_exc = int(np.count_nonzero(neq & both_zero.reshape(-1)))
    bad = neq & ~both_zero.reshape(-1)
    return dict(pass_=bool((not bad.any()) and nan_pattern and size_ok),
                flip=bool(flip), nan_pattern_equal=nan_pattern,
                n_records_compared=n_cmp, n_mismatch=int(bad.sum()),
                n_zero_sign_exceptions=zero_exc)


def judge_1b_mirror(ctx: Ctx) -> list[dict]:
    """1-b S-mirror（確証的・列別パリティのバイト比較）。"""
    flip = ctx.get("FLn_1216")
    ref = ctx.get("LR_1216")
    if flip is None or ref is None:
        miss = "FLn_1216" if flip is None else "LR_1216"
        return [not_run("FLn_1216", "4.1-b S-mirror", "confirmatory",
                        "column parity (bytes)", "MIRROR_NOT_RUN",
                        f"{miss} logs missing")]
    # --- 空虚な PASS を塞ぐ（spec §4.1-b: 「PASS = 10/10 seed で全列」）---------
    # 列が無い／seed が足りないときは「一致した」ではなく MIRROR_NOT_RUN。
    required = tuple(MIRROR_NEG_COLUMNS) + tuple(MIRROR_SAME_COLUMNS) + (
        "layer1_p_hat",)
    absent = sorted({f"{k}@{who}" for k in required
                     for who, arm in (("FLn_1216", flip), ("LR_1216", ref))
                     if not arm.has(k)})
    if absent:
        return [not_run("FLn_1216", "4.1-b S-mirror", "confirmatory",
                        "column parity (bytes)", "MIRROR_NOT_RUN",
                        "registered parity columns absent: " + ", ".join(absent))]
    if len(flip.seeds) != MIRROR_SEEDS_REQUIRED or not set(flip.seeds) <= set(ref.seeds):
        return [not_run("FLn_1216", "4.1-b S-mirror", "confirmatory",
                        "column parity (bytes)", "MIRROR_NOT_RUN",
                        f"needs {MIRROR_SEEDS_REQUIRED} seeds present in both logs; "
                        f"FLn seeds={list(flip.seeds)}, reference seeds="
                        f"{list(ref.seeds)}")]
    fails, checked = [], 0
    p_hat_exceptions, zero_sign_exceptions, compared = 0, 0, 0
    for s in flip.seeds:
        n = len(flip.steps(s))
        for key in MIRROR_NEG_COLUMNS + MIRROR_SAME_COLUMNS:
            got = mirror_parity(flip.raw(s, key), ref.raw(s, key)[:n],
                                flip=key in MIRROR_NEG_COLUMNS)
            checked += 1
            compared += int(got["n_records_compared"])
            zero_sign_exceptions += int(got["n_zero_sign_exceptions"])
            if not got["pass_"]:
                fails.append(f"seed{s}:{key}({got['n_mismatch']})")
        checked += 1
        ps = (flip.raw(s, "layer1_p_hat").astype(np.float64)
              + ref.raw(s, "layer1_p_hat")[:n].astype(np.float64))
        bad = int(np.count_nonzero(ps != 1.0))
        p_hat_exceptions += bad
        if bad:
            fails.append(f"seed{s}:p_hat_sum({bad})")
    label = "MIRROR_EXACT" if not fails else "MIRROR_BROKEN"
    note = (f"{checked} column-comparisons over {len(flip.seeds)} seeds "
            f"({compared} elements); p_hat'+p_hat != 1.0 exceptions = "
            f"{p_hat_exceptions}; registered zero-sign exceptions "
            f"(both arms exactly 0.0) = {zero_sign_exceptions}"
            + ("; excluded by registration: layer1_zmax (reference has no zmin), "
               "eff_rank/quantile columns" if not fails
               else "; failures: " + ", ".join(fails[:8])))
    return [row("FLn_1216", "4.1-b S-mirror", "confirmatory",
                "column parity (bytes)", "all records", "none", label,
                n=checked, note=note)]


def _seed_medians(arm: ArmLog, key: str, window, alive: bool = True) -> np.ndarray:
    vals = arm.unit_window(key, window)
    mask = arm.alive(window) if alive else np.ones_like(vals, dtype=bool)
    return np.array([float(np.median(vals[i][mask[i]])) if mask[i].any()
                     else float("nan") for i in range(vals.shape[0])])


def judge_1c_ensemble(ctx: Ctx, ks_q95: float) -> list[dict]:
    """1-c アンサンブル鏡像（REPORT_ONLY・seed 水準）。"""
    fl = ctx.get("FL_1216")
    ref = ctx.get("LR_1216")
    if fl is None or ref is None:
        return [not_run("FL_1216", "4.1-c ensemble mirror", "report",
                        "median_s[zbar(FL) + zbar(LR)]", "ENSEMBLE_NOT_RESOLVED",
                        "FL_1216 or LR_1216 logs missing")]
    win = ctx.tail("FL_1216")
    rwin = ctx.tail("LR_1216")
    _common, ia, ib = common_seed_rows(fl, ref)
    if len(ia) == 0:
        return [not_run("FL_1216", "4.1-c ensemble mirror", "report",
                        "median_s[zbar(FL) + zbar(LR)]", "ENSEMBLE_NOT_RESOLVED",
                        "no seed survives in both arms (G6)")]

    def _block(alive: bool):
        m_fl = _seed_medians(fl, "layer1_zbar", win, alive)[ia]
        m_lr = _seed_medians(ref, "layer1_zbar", rwin, alive)[ib]
        diff = m_fl - (-m_lr)
        n = len(diff)
        pt, lo, hi = boot_ci(lambda idx: nanmedian(diff[np.asarray(idx)]), n)
        va, vb = fl.unit_window("layer1_zbar", win), ref.unit_window("layer1_zbar", rwin)
        ma = fl.alive(win) if alive else np.ones_like(va, dtype=bool)
        mb = ref.alive(rwin) if alive else np.ones_like(vb, dtype=bool)
        ds = [ks_d(va[i][ma[i]], -vb[j][mb[j]]) for i, j in zip(ia, ib)]
        d_med = float(np.median(ds))
        if ci_within((lo, hi), -0.5, 0.5) and np.isfinite(ks_q95) and d_med < ks_q95:
            lab = "ENSEMBLE_SYMMETRIC"
        elif ci_excludes_zero((lo, hi)) and abs(pt) > 0.5:
            lab = "ENSEMBLE_ASYMMETRIC"
        else:
            lab = "ENSEMBLE_NOT_RESOLVED"
        return lab, pt, (lo, hi), n, d_med

    label, point, ci, n, d_med = _block(True)
    label_all, _p2, _c2, _n2, d_all = _block(False)
    gates = {"G5": "PASS", "G3": g3_agreement(label, label_all)}
    g2, vals = g2_settled(
        lambda w: nanmedian(_seed_medians(fl, "layer1_zbar", w)[ia]
                            + _seed_medians(ref, "layer1_zbar", w)[ib]),
        ci[1] - ci[0])
    gates["G2"] = g2
    return [row("FL_1216", "4.1-c ensemble mirror", "report",
                "median_s[med zbar(FL) - (-med zbar(LR))]", win, "ALIVE", label,
                n=n, death_rate=fl.death_rate(win), point=point, ci=ci,
                gates=gates, n_seeds_dropped=ctx.dropped("FL_1216"),
                note=f"median seed-wise KS D = {d_med:.4f} vs S-KSnull q95 "
                     f"= {ks_q95:.4f} (same unit: median of per-seed D); "
                     f"ALL label {label_all} (D={d_all:.4f}); "
                     f"settle {['%.3f' % v for v in vals]}")]


def _symmetry_block(arm: ArmLog, win, alive: bool) -> dict:
    vals = arm.unit_window("layer1_zbar", win)
    mask = arm.alive(win) if alive else np.ones_like(vals, dtype=bool)
    n = vals.shape[0]
    p, plo, phi = boot_ci(lambda idx: float((pooled(vals, mask, idx) > 0).mean()), n)
    m, mlo, mhi = boot_ci(pooled_median(vals, mask), n)
    a, alo, ahi = boot_ci(
        lambda idx: float(np.median(np.abs(pooled(vals, mask, idx)))), n)
    sk = skewness(pooled(vals, mask, np.arange(n)))
    return {"median": m, "median_ci": (mlo, mhi), "p_pos": p, "p_ci": (plo, phi),
            "abs_median": a, "abs_ci": (alo, ahi), "skew": sk,
            "n": int(mask.sum()), "death": float(1.0 - arm.alive(win).mean())}


def judge_1d_linear(ctx: Ctx) -> list[dict]:
    """1-d 線形網（LIN_1216）。"""
    name = "LIN_1216"
    arm = ctx.get(name)
    if arm is None:
        return [not_run(name, "4.1-d linear", "primary",
                        "median zbar / P(zbar>0) / skew", "NOT_DETERMINED",
                        "LIN_1216 logs missing")]
    gates = arm_gates(ctx, name)
    win = ctx.tail(name)
    labels = {}
    for tag, alive in (("ALIVE", True), ("ALL", False)):
        b = _symmetry_block(arm, win, alive)
        sym = (ci_within(b["median_ci"], -0.1, 0.1)
               and ci_within(b["p_ci"], 0.45, 0.55) and abs(b["skew"]) < 0.2)
        pinned = np.isfinite(b["abs_ci"][1]) and b["abs_ci"][1] < 0.3
        labels[tag] = ("LINEAR_PINNED" if (sym and pinned) else
                       "LINEAR_SYMMETRIC_NOT_PINNED" if sym else "LINEAR_ASYMMETRIC")
        if tag == "ALIVE":
            main = b
    gates["G3"] = g3_agreement(labels["ALIVE"], labels["ALL"])
    g2, vals = g2_settled(
        lambda w: float(np.median(np.abs(
            arm.unit_window("layer1_zbar", w)[arm.alive(w)]))),
        main["abs_ci"][1] - main["abs_ci"][0])
    gates["G2"] = g2
    reason = blocked(gates)
    label = labels["ALIVE"]
    if reason or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
        label = "NOT_DETERMINED"
    note = (f"P(zbar>0)={main['p_pos']:.3f} [{main['p_ci'][0]:.3f},"
            f"{main['p_ci'][1]:.3f}], skew={main['skew']:.3f}, "
            f"median zbar={main['median']:.3f}; ALL label {labels['ALL']}; "
            f"{gates['_g1']}; settle {['%.3f' % v for v in vals]}")
    if reason:
        note = reason + "; " + note
    return [row(name, "4.1-d linear", "primary", "median |zbar| (pinned) + symmetry",
                win, "ALIVE", label, n=main["n"], death_rate=main["death"],
                point=main["abs_median"], ci=main["abs_ci"], gates=gates, note=note)]


def judge_1e_odd(ctx: Ctx) -> list[dict]:
    """1-e 奇な非線形（TH_1216）。対称と釘付けを分ける。"""
    name = "TH_1216"
    arm = ctx.get(name)
    if arm is None:
        return [not_run(name, "4.1-e odd nonlinear", "primary",
                        "median |zbar| + symmetry", "NOT_DETERMINED",
                        "TH_1216 logs missing")]
    gates = arm_gates(ctx, name)
    win = ctx.tail(name)

    def _full_label(alive: bool):
        """spec §3.6 G3 は「**ラベル**が一致すること」なので、対称性の述語だけ
        でなく釘付け・逃走まで含めた最終ラベルを両方の除外規則で作る。"""
        b = _symmetry_block(arm, win, alive)
        sym = (ci_within(b["median_ci"], -0.1, 0.1)
               and ci_within(b["p_ci"], 0.45, 0.55) and abs(b["skew"]) < 0.2)
        pinned = np.isfinite(b["abs_ci"][1]) and b["abs_ci"][1] < 1.0
        traj = []
        for w in TRAJ:
            v = arm.unit_window("layer1_zbar", w)
            m = arm.alive(w) if alive else np.ones_like(v, dtype=bool)
            traj.append(boot_ci(lambda idx, v=v, m=m: float(
                np.median(np.abs(pooled(v, m, idx)))), v.shape[0]))
        runaway = (traj[0][0] < traj[1][0] < traj[2][0]
                   and traj[0][2] < traj[1][1] and traj[1][2] < traj[2][1])
        if not sym:
            lab = "ODD_ASYMMETRIC"
        elif pinned:
            lab = "ODD_SYMMETRIC_PINNED"
        elif runaway:
            lab = "ODD_SYMMETRIC_RUNAWAY"
        else:
            lab = "ODD_SYMMETRIC_INTERMEDIATE"
        return lab, b, traj, runaway

    label, b, traj, runaway = _full_label(True)
    label_all, _b_all, _t_all, _r_all = _full_label(False)
    gates["G3"] = g3_agreement(label, label_all)
    g2, svals = g2_settled(
        lambda w: float(np.median(np.abs(
            arm.unit_window("layer1_zbar", w)[arm.alive(w)]))),
        b["abs_ci"][1] - b["abs_ci"][0])
    gates["G2"] = g2
    reason = blocked(gates)
    # `ODD_SYMMETRIC_RUNAWAY` は「タスク 100→300→500 で単調増加」そのものが登録
    # 述語なので、5-c の `WELL_FROM_READOUT` と同じ理由で G2 では降格しない。
    g2_blocks = (g2 == "NOT_SETTLED" and label != "ODD_SYMMETRIC_RUNAWAY")
    if reason or gates["G3"] != "PASS" or g2_blocks:
        label = "NOT_DETERMINED"
    note = (f"P(zbar>0)={b['p_pos']:.3f}, skew={b['skew']:.3f}; "
            f"traj |zbar| {['%.3f' % t[0] for t in traj]}; "
            f"runaway={runaway}; ALL label {label_all}; {gates['_g1']}; "
            f"settle {['%.3f' % v for v in svals]}")
    if reason:
        note = reason + "; " + note
    return [row(name, "4.1-e odd nonlinear", "primary", "median |zbar|", win,
                "ALIVE", label, n=b["n"], death_rate=b["death"],
                point=b["abs_median"], ci=b["abs_ci"], gates=gates, note=note,
                n_seeds_dropped=ctx.dropped(name))]


# ---------------------------------------------------------------------------
# §4.2 命題 2
# ---------------------------------------------------------------------------
def judge_2a_sign(ctx: Ctx) -> list[dict]:
    out = []
    for name, want in (("FL_1216", +1), ("LR_1216", -1)):
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, "4.2-a sign", "report", "median zbar",
                               "SIGN_NOT_RUN", f"{name} logs missing"))
            continue
        win = ctx.tail(name)
        vals = arm.unit_window("layer1_zbar", win)
        mask = arm.alive(win)
        point, ci, n = median_stat_ci(vals, mask)
        _pa, ci_all, _na = median_stat_ci(vals, None)

        def _lab(c):
            ok_ = (c[0] > 2.0) if want > 0 else (c[1] < -2.0)
            return "SIGN_OK" if ok_ else "SIGN_UNEXPECTED"
        label = _lab(ci)
        gates = {"G3": g3_agreement(label, _lab(ci_all))}
        g2, svals = g2_settled(
            lambda w, arm=arm: float(np.median(
                arm.unit_window("layer1_zbar", w)[arm.alive(w)])),
            ci[1] - ci[0])
        gates["G2"] = g2
        out.append(row(name, "4.2-a sign", "report", "median zbar", win, "ALIVE",
                       label, n=n, gates=gates,
                       death_rate=arm.death_rate(win), point=point, ci=ci,
                       n_seeds_dropped=ctx.dropped(name),
                       note=f"registered expectation: {'CI>+2' if want>0 else 'CI<-2'}"
                            f"; ALL label {_lab(ci_all)}; "
                            f"settle {['%.3f' % v for v in svals]}"))
    return out


def paired_delta(ctx: Ctx, arm_a: str, arm_b: str, key: str = "layer1_zmax",
                 alive: bool = True, draws: np.ndarray | None = None):
    """G5 のペア（同じ (seed, unit)）の差。両腕 ALIVE の交わりで取る。"""
    a, b = ctx.get(arm_a), ctx.get(arm_b)
    if a is None or b is None:
        return None
    wa, wb = ctx.tail(arm_a), ctx.tail(arm_b)
    _common, ia, ib = common_seed_rows(a, b)
    if len(ia) == 0:
        return None
    va = a.unit_window(key, wa)[ia]
    vb = b.unit_window(key, wb)[ib]
    if alive:
        mask = a.alive(wa)[ia] & b.alive(wb)[ib]
    else:
        mask = np.ones_like(va, dtype=bool)
    diff = va - vb
    point, ci, used = median_stat_ci(diff, mask, draws)
    return {"point": point, "ci": ci, "n": used, "diff": diff, "mask": mask,
            "death_a": a.death_rate(wa), "death_b": b.death_rate(wb),
            "win_a": wa, "win_b": wb, "seed_rows": (ia, ib)}


def judge_2b_delta3(ctx: Ctx, reach: dict) -> list[dict]:
    """2-b 判別（確証的）: Δ_d = median_i[zmax(SH_d) − zmax(SH_d30)]。"""
    out = []
    for d, role in ((3, "confirmatory"), (2, "report")):
        arm = f"SH_d{d}_1216"
        res = paired_delta(ctx, arm, "SH_d30_1216")
        if res is None:
            out.append(not_run(arm, f"4.2-b Delta_{d}", role,
                               f"median_i[zmax({arm}) - zmax(SH_d30_1216)]",
                               "NOT_DETERMINED", "arm logs missing"))
            continue
        gates = arm_gates(ctx, arm)
        g30 = arm_gates(ctx, "SH_d30_1216")
        if g30["G1"] == "FROZEN":
            gates["G1"] = "FROZEN"
        gates["G4"] = g4_comparable(res["death_a"], res["death_b"])
        gates["G5"] = "PASS"
        res_all = paired_delta(ctx, arm, "SH_d30_1216", alive=False)

        def _lab(ci):
            if np.isfinite(ci[1]) and ci[1] < -0.3:
                return "CURVATURE_NONLOCAL"
            if ci_within(ci, -0.3, 0.3):
                return "CURVATURE_AT_INIT"
            return "MIXED"
        label = _lab(res["ci"])
        gates["G3"] = g3_agreement(label, _lab(res_all["ci"]))
        g2, vals = g2_settled(
            lambda w: _delta_in_window(ctx, arm, "SH_d30_1216", w),
            res["ci"][1] - res["ci"][0])
        gates["G2"] = g2
        reason = blocked(gates)
        note = f"ALL label {_lab(res_all['ci'])}; {gates['_g1']}"
        if reason:
            label, note = "NOT_DETERMINED", reason + "; " + note
        elif gates["G3"] != "PASS" or gates["G4"] == "NOT_COMPARABLE" or g2 == "NOT_SETTLED":
            label = "NOT_DETERMINED"
        rr = reach.get(arm)
        if d == 3 and rr is not None and np.isfinite(rr) and rr < 0.05:
            label = "NOT_DETERMINED"
            note += f"; 2-e reach rate {rr:.3f} < 0.05 (spec 4.2-e)"
        out.append(row(arm, f"4.2-b Delta_{d}", role,
                       f"median_i[zmax({arm}) - zmax(SH_d30_1216)]",
                       res["win_a"], "ALIVE (both arms)", label, n=res["n"],
                       death_rate=res["death_a"], point=res["point"],
                       ci=res["ci"], gates=gates, note=note))
    return out


def _delta_in_window(ctx: Ctx, arm_a: str, arm_b: str, window,
                     key: str = "layer1_zmax") -> float:
    a, b = ctx.get(arm_a), ctx.get(arm_b)
    if a is None or b is None:
        return float("nan")
    va, vb = a.unit_window(key, window), b.unit_window(key, window)
    n = min(va.shape[0], vb.shape[0])
    mask = a.alive(window)[:n] & b.alive(window)[:n]
    d = (va[:n] - vb[:n])[mask]
    return float(np.median(d)) if d.size else float("nan")


def judge_2c_smooth(ctx: Ctx) -> list[dict]:
    """2-c 滑らかな腕（SP_1216）: median_i[zbar(tail) − zbar(init)]。"""
    name = "SP_1216"
    arm = ctx.get(name)
    if arm is None:
        return [not_run(name, "4.2-c smooth", "primary",
                        "median_i[zbar(tail) - zbar(init)]", "NOT_DETERMINED",
                        "SP_1216 logs missing")]
    gates = arm_gates(ctx, name)
    win = ctx.tail(name)
    diff = arm.unit_window("layer1_zbar", win) - arm.unit_at_step("layer1_zbar")
    mask = arm.alive(win)
    point, ci, n = median_stat_ci(diff, mask)
    _, ci_all, _ = median_stat_ci(diff, None)

    def _lab(c):
        if np.isfinite(c[1]) and c[1] < -0.3:
            return "SMOOTH_DOWN"
        if np.isfinite(c[0]) and c[0] > 0.3:
            return "SMOOTH_UP"
        if ci_within(c, -0.3, 0.3):
            return "SMOOTH_STATIONARY"
        return "NOT_DETERMINED"
    label = _lab(ci)
    gates["G3"] = g3_agreement(label, _lab(ci_all))
    g2, vals = g2_settled(
        lambda w: float(np.median((arm.unit_window("layer1_zbar", w)
                                   - arm.unit_at_step("layer1_zbar"))[arm.alive(w)])),
        ci[1] - ci[0])
    gates["G2"] = g2
    reason = blocked(gates)
    note = f"ALL label {_lab(ci_all)}; {gates['_g1']}"
    if reason or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
        label = "NOT_DETERMINED"
        note = (reason or "gate") + "; " + note
    return [row(name, "4.2-c smooth", "primary",
                "median_i[zbar(tail) - zbar(init)]", win, "ALIVE", label,
                n=n, death_rate=arm.death_rate(win), point=point, ci=ci,
                gates=gates, note=note)]


def judge_2d_steep(ctx: Ctx) -> list[dict]:
    """2-d 曲率反転の対照（ST_d1/ST_d2）。"""
    out, labels = [], []
    for d in (1, 2):
        a, b = f"ST_d{d}_1216", f"SH_d{d}_1216"
        res = paired_delta(ctx, a, b)
        if res is None:
            out.append(not_run(a, f"4.2-d Delta_st(d={d})", "primary",
                               f"median_i[zmax({a}) - zmax({b})]",
                               "NOT_DETERMINED", "arm logs missing"))
            labels.append(None)
            continue
        gates = arm_gates(ctx, a)
        gates["G4"] = g4_comparable(res["death_a"], res["death_b"])
        gates["G5"] = "PASS"
        res_all = paired_delta(ctx, a, b, alive=False)

        def _lab(ci):
            if np.isfinite(ci[0]) and ci[0] > 0.3:
                return "UP"
            if np.isfinite(ci[1]) and ci[1] < -0.3:
                return "DOWN"
            if ci_within(ci, -0.3, 0.3):
                return "NULL"
            return "MIXED"
        lab = _lab(res["ci"])
        gates["G3"] = g3_agreement(lab, _lab(res_all["ci"]))
        g2, svals = g2_settled(lambda w, a=a, b=b: _delta_in_window(ctx, a, b, w),
                               res["ci"][1] - res["ci"][0])
        gates["G2"] = g2
        reason = blocked(gates)
        if reason or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
            lab = None
        labels.append(lab)
        out.append(row(a, f"4.2-d Delta_st(d={d})", "primary",
                       f"median_i[zmax({a}) - zmax({b})]", res["win_a"],
                       "ALIVE (both arms)", lab or "NOT_DETERMINED", n=res["n"],
                       death_rate=res["death_a"], point=res["point"], ci=res["ci"],
                       gates=gates, n_seeds_dropped=ctx.dropped(a),
                       note=(reason or "") + f"; ALL label {_lab(res_all['ci'])}"
                            f"; settle {['%.3f' % v for v in svals]}"))
    if all(v == "UP" for v in labels) and labels and None not in labels:
        overall = "ASYMMETRY_SIGN_OK"
    elif any(v == "DOWN" for v in labels):
        overall = "ASYMMETRY_SIGN_BROKEN"
    elif labels and all(v == "NULL" for v in labels):
        overall = "ASYMMETRY_SIGN_NULL"
    else:
        overall = "NOT_DETERMINED"
    out.append(row("ST_d1_1216+ST_d2_1216", "4.2-d overall", "primary",
                   "both d in {1,2}", None, "ALIVE (both arms)", overall,
                   note=f"per-d labels {labels}"))
    return out


def judge_2e_reach(ctx: Ctx) -> tuple[list[dict], dict]:
    """2-e 届いたか（REPORT）。zmin 列が要る。"""
    out, rates = [], {}
    for d in (0.5, 1, 2, 3):
        name = {0.5: "SH_d0p5_1216", 1: "SH_d1_1216", 2: "SH_d2_1216",
                3: "SH_d3_1216"}[d]
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, "4.2-e reach", "report",
                               "reach rate / sunk-given-reach", "NOT_RUN",
                               "arm logs missing"))
            rates[name] = float("nan")
            continue
        if not arm.has("layer1_zmin"):
            out.append(not_run(name, "4.2-e reach", "report",
                               "reach rate / sunk-given-reach", "NOT_DETERMINED",
                               "layer1_zmin column absent in the logs"))
            rates[name] = float("nan")
            continue
        win = ctx.tail(name)
        reach_task, reached = [], []
        for i, s in enumerate(arm.seeds):
            steps = arm.steps(s)
            zmin = arm.col(s, "layer1_zmin")
            hit = zmin < -d
            first = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
            t = np.where(first >= 0, steps[np.clip(first, 0, None)] / PERIOD,
                         np.nan)
            reach_task.append(t)
            reached.append(first >= 0)
        reached = np.array(reached)
        rate = float(reached.mean())
        rates[name] = rate
        zmax = arm.unit_window("layer1_zmax", win)
        alive = arm.alive(win)
        sunk_given = float((zmax[reached & alive] <= -d).mean()) if (reached & alive).any() else float("nan")
        nr = (~reached) & alive
        zbar_nr = float(np.median(np.abs(arm.unit_window("layer1_zbar", win)[nr]))) if nr.any() else float("nan")
        out.append(row(name, "4.2-e reach", "report",
                       f"reach rate (zmin < -{d})", win, "ALIVE", "REPORT_ONLY",
                       n=int(reached.size), death_rate=arm.death_rate(win),
                       point=rate, gates={"G5": "PASS"},
                       note=f"P(zmax<=-d | reached, ALIVE)={sunk_given:.4f}; "
                            f"median |zbar| of not-reached ALIVE = {zbar_nr:.4f}; "
                            f"median t_reach = "
                            f"{np.nanmedian(np.concatenate(reach_task)):.1f} tasks"))
    return out, rates


# ---------------------------------------------------------------------------
# §4.3 命題 3
# ---------------------------------------------------------------------------
def judge_3a_retention(ctx: Ctx) -> tuple[list[dict], dict]:
    """3-a 保持率 ρ（確証的）。"""
    pairs = (("LRbp5_1216", "LRnull_1216", "leaky/from above"),
             ("Ebp4_1216", "Enull_1216", "elu/from above"),
             ("LRbm5_1216", "LRnull_1216", "leaky/from below"),
             ("Ebm4_1216", "Enull_1216", "elu/from below"))
    out, labels = [], {}
    for arm_name, ref_name, tag in pairs:
        arm = ctx.get(arm_name)
        ref = ctx.get(ref_name) or ctx.get({"LRnull_1216": "LR_1216",
                                            "Enull_1216": "E_1216"}[ref_name])
        if arm is None or ref is None:
            out.append(not_run(arm_name, "4.3-a retention", "confirmatory",
                               "rho (zmax)", "NOT_DETERMINED",
                               "arm or reference logs missing"))
            labels[arm_name] = "NOT_DETERMINED"
            continue
        gates = arm_gates(ctx, arm_name)
        wa, wr = ctx.tail(arm_name), ctx.tail(ref.name)
        _cs, ja, jr = common_seed_rows(arm, ref)
        n = len(ja)
        if n == 0:
            out.append(not_run(arm_name, "4.3-a retention", "confirmatory",
                               "rho (zmax)", "NOT_DETERMINED",
                               "no seed survives in both arms (G6)"))
            labels[arm_name] = "NOT_DETERMINED"
            continue
        za = arm.unit_window("layer1_zmax", wa)[ja]
        zr = ref.unit_window("layer1_zmax", wr)[jr]
        ia = arm.unit_at_step("layer1_zmax")[ja]
        ir = ref.unit_at_step("layer1_zmax")[jr]
        ma, mr = arm.alive(wa)[ja], ref.alive(wr)[jr]
        gates["G4"] = g4_comparable(arm.death_rate(wa), ref.death_rate(wr))
        gates["G5"] = "PASS"
        d = paired_delta(ctx, arm_name, ref.name)

        def _rho_label(mask_a, mask_r):
            def rho(idx):
                idx = np.asarray(idx)
                num = (float(np.median(pooled(za, mask_a, idx)))
                       - float(np.median(pooled(zr, mask_r, idx))))
                den = (float(np.median(pooled(ia, mask_a, idx)))
                       - float(np.median(pooled(ir, mask_r, idx))))
                return num / den if den != 0 else float("nan")
            pt, lo_, hi_ = boot_ci(rho, n)
            both = mask_a & mask_r
            sp_ = spearman(za[both], zr[both])
            return pt, lo_, hi_, sp_

        point, lo, hi, sp = _rho_label(ma, mr)
        pair_ok = (ci_within(d["ci"], -0.3, 0.3) if d else False) and (
            np.isfinite(sp) and sp >= 0.5)

        def _lab(hi_, lo_, ok_):
            if np.isfinite(hi_) and hi_ < 0.15 and ok_:
                return "MEAN_INDEPENDENT"
            if np.isfinite(lo_) and lo_ > 0.40:
                return "MEAN_DEPENDENT"
            return "INTERMEDIATE"
        label = _lab(hi, lo, pair_ok)
        # G3: 除外なし（ALL）でも同じラベルが出るか（spec §3.6）
        all_a = np.ones_like(za, dtype=bool)
        all_r = np.ones_like(zr, dtype=bool)
        p_all, lo_all, hi_all, sp_all = _rho_label(all_a, all_r)
        d_all = paired_delta(ctx, arm_name, ref.name, alive=False)
        pair_ok_all = (ci_within(d_all["ci"], -0.3, 0.3) if d_all else False) and (
            np.isfinite(sp_all) and sp_all >= 0.5)
        label_all = _lab(hi_all, lo_all, pair_ok_all)
        gates["G3"] = g3_agreement(label, label_all)

        def _rho_point(w):
            zaw = arm.unit_window("layer1_zmax", w)[ja]
            zrw = ref.unit_window("layer1_zmax", w)[jr]
            maw, mrw = arm.alive(w)[ja], ref.alive(w)[jr]
            idx = np.arange(n)
            den = (float(np.median(pooled(ia, maw, idx)))
                   - float(np.median(pooled(ir, mrw, idx))))
            if den == 0:
                return float("nan")
            return (float(np.median(pooled(zaw, maw, idx)))
                    - float(np.median(pooled(zrw, mrw, idx)))) / den
        g2, svals = g2_settled(_rho_point, hi - lo)
        gates["G2"] = g2
        reason = blocked(gates)
        if reason:
            label = "NOT_DETERMINED"
        elif gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
            label = "NOT_DETERMINED"
        labels[arm_name] = label
        # 「それ以外 → 3-b へ」は**ラベルではなく経路**（spec §4.3-a）。
        row_label = "NOT_DETERMINED" if label == "INTERMEDIATE" else label
        note = (f"{tag}; paired Delta median="
                f"{d['point']:.3f} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}], "
                f"Spearman={sp:.3f}, pair_ok={pair_ok}; ref={ref.name}; "
                f"ALL label {label_all}; settle {['%.3f' % v for v in svals]}; "
                f"{gates['_g1']}")
        if label == "INTERMEDIATE":
            note = "ROUTED_TO_3B (spec §4.3-a の第 3 分岐は経路であってラベルでは"
            note += (f"ない); {tag}; paired Delta median={d['point']:.3f} "
                     f"CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}], Spearman={sp:.3f}; "
                     f"ALL label {label_all}; ref={ref.name}")
        if reason:
            note = reason + "; " + note
        out.append(row(arm_name, "4.3-a retention", "confirmatory", "rho (zmax)",
                       wa, "ALIVE", row_label, n=d["n"] if d else 0,
                       death_rate=arm.death_rate(wa), point=point, ci=(lo, hi),
                       gates=gates, note=note,
                       n_seeds_dropped=ctx.dropped(arm_name)))
        if wa == TAIL_15M:                       # spec §3.5: 15M 腕は 451–500 も併記
            za5 = arm.unit_window("layer1_zmax", TAIL_5M)[ja]
            ma5 = arm.alive(TAIL_5M)[ja]

            def rho5(idx, za5=za5, ma5=ma5):
                idx = np.asarray(idx)
                num = (float(np.median(pooled(za5, ma5, idx)))
                       - float(np.median(pooled(zr, mr, idx))))
                den = (float(np.median(pooled(ia, ma5, idx)))
                       - float(np.median(pooled(ir, mr, idx))))
                return num / den if den != 0 else float("nan")
            p5, l5, h5 = boot_ci(rho5, n)
            out.append(row(arm_name, "4.3-a retention (alt window)", "report",
                           "rho (zmax)", TAIL_5M, "ALIVE", "REPORT_ONLY",
                           n=int(ma5.sum()), death_rate=arm.death_rate(TAIL_5M),
                           point=p5, ci=(l5, h5),
                           note="spec §3.5: 15M 腕は主窓 1451–1500 のほかに "
                                "451–500 も併記する"))
    return out, labels


def relax_fit(t: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """z(t) = z_inf + (z0 − z_inf) e^{−t/τ} を τ の対数格子＋線形最小二乗で当てる。

    scipy 無し。返り値 (z_inf, z0, τ)。
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(z)
    t, z = t[m], z[m]
    if t.size < 4:
        return float("nan"), float("nan"), float("nan")
    t0 = t - t.min()
    best = (np.inf, float("nan"), float("nan"), float("nan"))
    for tau in np.geomspace(1.0, max(10.0, 5.0 * (t0.max() + 1.0)), 240):
        A = np.stack([np.ones_like(t0), np.exp(-t0 / tau)], axis=1)
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        rss = float(((A @ coef - z) ** 2).sum())
        if rss < best[0]:
            best = (rss, float(coef[0]), float(coef[0] + coef[1]), float(tau))
    return best[1], best[2], best[3]


def judge_3b_relax(ctx: Ctx, labels: dict) -> list[dict]:
    """3-b 緩和フィット（3-a が中間だったときだけ判定する）。"""
    out = []
    for arm_name, ref_name in (("LRbp5_1216", "LRnull_1216"),
                               ("Ebp4_1216", "Enull_1216"),
                               ("LRbm5_1216", "LRnull_1216"),
                               ("Ebm4_1216", "Enull_1216")):
        arm = ctx.get(arm_name)
        ref = ctx.get(ref_name) or ctx.get({"LRnull_1216": "LR_1216",
                                            "Enull_1216": "E_1216"}[ref_name])
        if arm is None or ref is None:
            out.append(not_run(arm_name, "4.3-b relaxation fit", "secondary",
                               "z_inf of exponential fit", "NOT_DETERMINED",
                               "arm or reference logs missing"))
            continue
        if labels.get(arm_name) in ("MEAN_INDEPENDENT", "MEAN_DEPENDENT"):
            out.append(row(arm_name, "4.3-b relaxation fit", "secondary",
                           "z_inf of exponential fit", None, "ALIVE",
                           "NOT_APPLIED", note=f"3-a resolved as {labels[arm_name]}"))
            continue
        last = 1500 if ctx.tail(arm_name) == TAIL_15M else 500
        wr = ctx.tail(ref.name)
        ref_med = float(np.median(ref.unit_window("layer1_zmax", wr)[ref.alive(wr)]))
        pert = 5.0 if arm_name.startswith("LR") else 4.0

        def _fit(alive: bool, upto: int = last):
            zinf_, taus_ = [], []
            for s in arm.kept_seeds:
                steps = arm.steps(s)
                idx = _window_indices(steps, PERIOD, [100, upto])
                tasks = steps[idx] / PERIOD
                zmax = arm.col(s, "layer1_zmax")[idx]
                den = arm.col(s, "layer1_denom")[idx]
                keep = [(den[k] > ALIVE_DENOM) if alive
                        else np.ones(zmax.shape[1], dtype=bool)
                        for k in range(len(idx))]
                med = np.array([np.median(zmax[k][keep[k]]) if keep[k].any()
                                else np.nan for k in range(len(idx))])
                zi, _z0, tau = relax_fit(tasks, med)
                zinf_.append(zi)
                taus_.append(tau)
            return np.array(zinf_, dtype=np.float64), taus_

        def _label(zinf_):
            n_ = len(zinf_)
            if n_ == 0:
                return "NOT_DETERMINED", float("nan"), (float("nan"), float("nan"))
            pt, lo_, hi_ = boot_ci(
                lambda idx: nanmedian(zinf_[np.asarray(idx)]), n_)
            if not np.isfinite(hi_ - lo_) or (hi_ - lo_) > pert:
                lab = "NOT_DETERMINED"
            elif lo_ <= ref_med <= hi_:
                lab = "MEAN_INDEPENDENT_SLOW"
            else:
                lab = "MEAN_DEPENDENT"
            return lab, pt, (lo_, hi_)

        zinf, taus = _fit(True)
        n = len(zinf)
        label, point, (lo, hi) = _label(zinf)
        label_all, _pa, _ca = _label(_fit(False)[0])
        gates = arm_gates(ctx, arm_name)
        gates["G3"] = g3_agreement(label, label_all)
        # G2: 定着ゲートは「短く切った窓で当てた z_inf が動かないか」で出す。
        g2, svals = g2_settled(
            lambda w: _label(_fit(True, upto=int(w[1]))[0])[1], hi - lo)
        gates["G2"] = g2
        if blocked(gates) or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
            label = "NOT_DETERMINED"
        out.append(row(arm_name, "4.3-b relaxation fit", "secondary",
                       "z_inf of exponential fit", f"tasks 100-{last}", "ALIVE",
                       label, n=n, death_rate=arm.death_rate(ctx.tail(arm_name)),
                       point=point, ci=(lo, hi), gates=gates,
                       n_seeds_dropped=ctx.dropped(arm_name),
                       note=f"reference median zmax={ref_med:.3f} ({ref.name}); "
                            f"tau median={np.nanmedian(taus):.1f} tasks (REPORT); "
                            f"ALL label {label_all}; "
                            f"settle z_inf {['%.3f' % v for v in svals]}"))
    return out


def judge_3_overall(rows_3a: list[dict], rows_3b: list[dict],
                    labels_3a: dict) -> list[dict]:
    """命題 3 の**登録ラベル**（spec §4.3・排他かつ網羅）を leaky / ELU 別に出す。

    3-a の第 3 分岐は「3-b へ」という経路なので、3-a と 3-b をここで 1 つに畳む。
    """
    fams = {"leaky": ("LRbp5_1216", "LRbm5_1216"),
            "elu": ("Ebp4_1216", "Ebm4_1216")}
    b_label = {r["arm"]: r["label"] for r in rows_3b}
    out = []
    for fam, arms in fams.items():
        per = {}
        for name in arms:
            lab = labels_3a.get(name, "NOT_DETERMINED")
            if lab == "INTERMEDIATE":
                lab = b_label.get(name, "NOT_DETERMINED")
                if lab == "NOT_APPLIED":
                    lab = "NOT_DETERMINED"
            per[name] = lab
        vals = list(per.values())
        if "NOT_DETERMINED" in vals or not vals:
            overall = "NOT_DETERMINED"
        elif "MEAN_DEPENDENT" in vals:
            overall = "MEAN_DEPENDENT"
        elif "MEAN_INDEPENDENT_SLOW" in vals:
            overall = "MEAN_INDEPENDENT_SLOW"
        else:
            overall = "MEAN_INDEPENDENT"
        out.append(row("+".join(arms), f"4.3 overall ({fam})", "confirmatory",
                       "3-a rho folded with 3-b z_inf", None, "ALIVE", overall,
                       note=f"per-arm resolved labels {per}; spec §4.3 の 4 ラベル"
                            f"（MEAN_INDEPENDENT / _SLOW / MEAN_DEPENDENT / "
                            f"NOT_DETERMINED）は排他かつ網羅"))
    return out


def judge_3c_return(ctx: Ctx) -> list[dict]:
    """3-c 下からの戻り道（REPORT・機構の判別）。"""
    out = []
    for name in ("LRbm5_1216", "Ebm4_1216"):
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, "4.3-c return path", "report",
                               "median zmax at tasks 100/300/500(/1500)",
                               "NOT_RUN", "arm logs missing"))
            continue
        wins = list(TRAJ)
        if ctx.tail(name) == TAIL_15M:
            wins = wins + [TAIL_15M]
        pts = []
        for w in wins:
            v = arm.unit_window("layer1_zmax", w)
            m = arm.alive(w)
            pts.append(median_stat_ci(v, m)[0])
        mono = all(pts[i] < pts[i + 1] for i in range(len(pts) - 1))
        win = ctx.tail(name)
        zmax = arm.unit_window("layer1_zmax", win)
        alive = arm.alive(win)
        sunk = alive & (zmax <= -1.0)
        deep = float((zmax[sunk] <= -2.0).mean()) if sunk.any() else float("nan")
        alt = ""
        if win == TAIL_15M:                      # spec §3.5: 451–500 も併記
            v5, m5 = arm.unit_window("layer1_zmax", TAIL_5M), arm.alive(TAIL_5M)
            b5 = arm.unit_window("layer1_zbar", TAIL_5M)
            alt = (f"; alt window 451-500: median zmax={nanmedian(v5[m5]):.3f}, "
                   f"median zbar={nanmedian(b5[m5]):.3f}, "
                   f"death rate={arm.death_rate(TAIL_5M):.3f}")
        out.append(row(name, "4.3-c return path", "report",
                       "median zmax trajectory", win, "ALIVE", "REPORT_ONLY",
                       n=int(alive.sum()), death_rate=arm.death_rate(win),
                       point=pts[-1],
                       note=f"trajectory {['%.3f' % p for p in pts]} "
                            f"(monotone rise={mono}); "
                            f"P(zmax<=-2 | sunk d=1)={deep:.3f}" + alt))
    return out


# ---------------------------------------------------------------------------
# §4.4 命題 4
# ---------------------------------------------------------------------------
def judge_4a_literal() -> list[dict]:
    return [row("(committed)", "4.4-a literal reading", "report",
                "Snake zero vs GELU valley / Gc_b1", None, "",
                "LITERAL_OUT_OF_SCOPE",
                note="spec §4.4-a・§8-2: Issa 自身が括弧で除外した場合に当たる。"
                     "REFUTED_BY_COMMITTED とは書かない（REPORT_ONLY）")]


def judge_4b_locality(ctx: Ctx) -> list[dict]:
    """4-b 局所性半径 d*（確証的）。"""
    lin = ctx.get("LIN_1216")
    ladder = [(0.5, "SH_d0p5_1216"), (1.0, "SH_d1_1216"), (2.0, "SH_d2_1216"),
              (3.0, "SH_d3_1216"), (30.0, "SH_d30_1216")]
    if lin is None:
        return [not_run("SH_d*", "4.4-b locality radius", "confirmatory",
                        "d* (|med zmax diff| and |med zbar diff| <= 0.5)",
                        "NOT_DETERMINED", "LIN_1216 logs missing")]
    out, diffs, diffs_all, undet = [], {}, {}, 0
    wl = ctx.tail("LIN_1216")
    # 基準線そのもののゲート（spec §3.6 G1）。LIN は ||w|| が**縮む**設計なので
    # ここが FROZEN になる可能性が高い（spec §2: 1.41→0.41）。基準線が凍っている
    # まま確証的な局所性ラベルを出さない。
    lin_gates = arm_gates(ctx, "LIN_1216")
    lin_blocked = blocked(lin_gates)
    for d, name in ladder:
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, "4.4-b locality radius", "secondary",
                               "median diff vs LIN_1216", "NOT_DETERMINED",
                               "arm logs missing"))
            undet += 1
            diffs[d] = None
            diffs_all[d] = None
            continue
        gates = arm_gates(ctx, name)
        wa = ctx.tail(name)
        gates["G4"] = g4_comparable(arm.death_rate(wa), lin.death_rate(wl))
        gates["G5"] = "PASS"
        if blocked(gates) or lin_blocked:
            undet += 1
        _cs, ja, jl = common_seed_rows(arm, lin)
        n = len(ja)
        res, res_all = {}, {}
        for key in ("layer1_zmax", "layer1_zbar"):
            va = arm.unit_window(key, wa)[ja]
            vb = lin.unit_window(key, wl)[jl]
            ma, mb = arm.alive(wa)[ja], lin.alive(wl)[jl]

            def stat(idx, va=va, vb=vb, ma=ma, mb=mb):
                idx = np.asarray(idx)
                return (float(np.median(pooled(va, ma, idx)))
                        - float(np.median(pooled(vb, mb, idx))))

            def stat_all(idx, va=va, vb=vb):
                idx = np.asarray(idx)
                return (float(np.median(pooled(va, None, idx)))
                        - float(np.median(pooled(vb, None, idx))))
            res[key] = boot_ci(stat, n)
            res_all[key] = boot_ci(stat_all, n)
        diffs[d] = res
        diffs_all[d] = res_all
        ok = abs(res["layer1_zmax"][0]) <= 0.5 and abs(res["layer1_zbar"][0]) <= 0.5
        ok_all = (abs(res_all["layer1_zmax"][0]) <= 0.5
                  and abs(res_all["layer1_zbar"][0]) <= 0.5)
        gates["G3"] = g3_agreement(str(ok), str(ok_all))

        def _diff_point(w, name=name, ja=ja, jl=jl):
            va = arm.unit_window("layer1_zmax", w)[ja]
            vb = lin.unit_window("layer1_zmax", w)[jl]
            ma, mb = arm.alive(w)[ja], lin.alive(w)[jl]
            idx = np.arange(len(ja))
            return (float(np.median(pooled(va, ma, idx)))
                    - float(np.median(pooled(vb, mb, idx))))
        g2, svals = g2_settled(_diff_point,
                               res["layer1_zmax"][2] - res["layer1_zmax"][1])
        gates["G2"] = g2
        out.append(row(name, "4.4-b locality radius", "secondary",
                       "median zmax(SH_d) - median zmax(LIN)", wa,
                       "ALIVE", "MATCHES_LINEAR" if ok else "DIFFERS_FROM_LINEAR",
                       n=int(arm.alive(wa)[ja].sum()),
                       death_rate=arm.death_rate(wa),
                       point=res["layer1_zmax"][0],
                       ci=(res["layer1_zmax"][1], res["layer1_zmax"][2]),
                       gates=gates, n_seeds_dropped=ctx.dropped(name),
                       note=f"zbar diff={res['layer1_zbar'][0]:.3f} "
                            f"[{res['layer1_zbar'][1]:.3f},{res['layer1_zbar'][2]:.3f}]"
                            f"; ALL matches={ok_all}; "
                            f"settle {['%.3f' % v for v in svals]}"
                            + (f"; baseline {lin_blocked}" if lin_blocked else "")))
    present = [(d, r) for d, r in sorted(diffs.items()) if r is not None]
    dstar = None
    for d, r in present:
        if abs(r["layer1_zmax"][0]) <= 0.5 and abs(r["layer1_zbar"][0]) <= 0.5:
            dstar = d
            break
    mags = [abs(r["layer1_zmax"][0]) for _d, r in present]
    monotone = all(mags[i] >= mags[i + 1] - 1e-12 for i in range(len(mags) - 1))
    present_all = [(d, r) for d, r in sorted(diffs_all.items()) if r is not None]
    dstar_all = next((d for d, r in present_all
                      if abs(r["layer1_zmax"][0]) <= 0.5
                      and abs(r["layer1_zbar"][0]) <= 0.5), None)
    mags_all = [abs(r["layer1_zmax"][0]) for _d, r in present_all]
    mono_all = all(mags_all[i] >= mags_all[i + 1] - 1e-12
                   for i in range(len(mags_all) - 1))

    def _overall(mono, ds):
        return "LOCALITY_MONOTONE" if (mono and ds is not None) else "NONLOCAL"
    label = _overall(monotone, dstar)
    label_all = _overall(mono_all, dstar_all)
    gates = {"G1": lin_gates["G1"], "G6": lin_gates["G6"], "G5": "PASS",
             "G3": g3_agreement(label, label_all)}
    if undet >= 2 or lin_blocked or gates["G3"] != "PASS":
        label = "NOT_DETERMINED"
    out.append(row("SH_d ladder", "4.4-b locality radius", "confirmatory",
                   "d* = min d matching LIN_1216 within 0.5", wl, "ALIVE", label,
                   n=len(present), point=float(dstar) if dstar else float("nan"),
                   gates=gates, n_seeds_dropped=ctx.dropped("LIN_1216"),
                   note=f"|zmax diff| by d: "
                        f"{ {d: round(abs(r['layer1_zmax'][0]), 3) for d, r in present} }; "
                        f"monotone={monotone}; d*={dstar}; ALL label {label_all}; "
                        f"{undet} shelf arms NOT_DETERMINED; "
                        f"baseline LIN_1216 gates: {lin_gates['_g1']}"
                        + (f" -> {lin_blocked}" if lin_blocked else "")))
    return out


def judge_4c_edge(ctx: Ctx) -> list[dict]:
    """4-c 上端は折れ目に追随するか（R とは独立の仮説）。"""
    out, labels = [], {}
    for d, role in ((2.0, "primary"), (3.0, "primary"), (0.5, "report"),
                    (1.0, "report")):
        name = {0.5: "SH_d0p5_1216", 1.0: "SH_d1_1216", 2.0: "SH_d2_1216",
                3.0: "SH_d3_1216"}[d]
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, f"4.4-c edge (d={d})", role,
                               "median zmax + d", "EDGE_NOT_DETERMINED",
                               "arm logs missing"))
            if role == "primary":
                labels[d] = None
            continue
        gates = arm_gates(ctx, name)
        win = ctx.tail(name)
        v, m = arm.unit_window("layer1_zmax", win), arm.alive(win)
        point, ci, n = median_stat_ci(v, m)
        _, ci_all, _ = median_stat_ci(v, None)

        def _lab(c):
            if ci_within((c[0] + d, c[1] + d), -0.3, 0.3):
                return "EDGE_AT_KINK"
            if np.isfinite(c[0]) and c[0] > -d + 0.3:
                return "EDGE_DETACHED_UP"
            if np.isfinite(c[1]) and c[1] < -d - 0.3:
                return "EDGE_DETACHED_DOWN"
            return "EDGE_MIXED"
        lab = _lab(ci)
        gates["G3"] = g3_agreement(lab, _lab(ci_all))
        g2, _vals = g2_settled(
            lambda w: float(np.median(arm.unit_window("layer1_zmax", w)[arm.alive(w)])),
            ci[1] - ci[0])
        gates["G2"] = g2
        reason = blocked(gates)
        if reason or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
            lab = "EDGE_NOT_DETERMINED"
        if role == "primary":
            labels[d] = None if lab == "EDGE_NOT_DETERMINED" else lab
        halfw = float(np.median((arm.unit_window("layer1_zmax", win)
                                 - arm.unit_window("layer1_zbar", win))[m]))
        state = ("A (linear)" if abs(point) < 0.5 and halfw < 1.0 else
                 "B (zmax~-d, wide)" if abs(point + d) < 0.5 and halfw >= 1.0 else
                 "C (zmax~0, wide)" if abs(point) < 0.5 and halfw >= 1.0 else "other")
        out.append(row(name, f"4.4-c edge (d={d})", role, "median zmax", win,
                       "ALIVE", lab, n=n, death_rate=arm.death_rate(win),
                       point=point, ci=ci, gates=gates,
                       note=f"median half-width={halfw:.3f}; 3-state REPORT: {state}"
                            + ("; " + reason if reason else "")))
    vals = [labels.get(2.0), labels.get(3.0)]
    if None in vals or len([v for v in vals if v]) < 2:
        overall = "EDGE_NOT_DETERMINED"
    elif vals[0] == vals[1]:
        overall = vals[0]
    else:
        overall = "EDGE_MIXED"
    out.append(row("SH_d2+SH_d3", "4.4-c edge overall", "primary",
                   "both d in {2,3}", None, "ALIVE", overall,
                   note=f"per-d labels {vals}; spec §4.4-c: R のラベルには入れない"))
    return out


# ---------------------------------------------------------------------------
# §4.5 命題 5
# ---------------------------------------------------------------------------
def b_slope(ctx: Ctx, name: str, *, lag: bool, submerged: bool,
            alive: bool = True, seed_idx=None, c_const: float = C_CONST,
            window=None, rows=None):
    """B: zmax を ln(1+C v^2) へ OLS した傾きの符号反転値（spec §4.5-a）。

    ``window`` は G2（定着ゲート）が主窓を差し替えるため、``rows`` は G6 で
    落とす seed が腕ごとに違うときに**共通 seed の行だけ**へ絞るため。
    """
    arm = ctx.get(name)
    if arm is None:
        return float("nan"), float("nan"), 0
    win = ctx.tail(name) if window is None else tuple(window)
    lwin = (ctx.lag(name) if window is None else win) if lag else win
    zmax = arm.unit_window("layer1_zmax", win)
    v = arm.unit_window("layer1_v_unit", lwin)
    mask = arm.alive(win) if alive else np.ones_like(zmax, dtype=bool)
    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        zmax, v, mask = zmax[rows], v[rows], mask[rows]
    if submerged:
        mask = mask & (zmax < 0.0)
    idx = np.arange(zmax.shape[0]) if seed_idx is None else np.asarray(seed_idx)
    y = pooled(zmax, mask, idx)
    x = np.log1p(c_const * pooled(v, mask, idx) ** 2)
    a, b = ols(y, x)
    return -b, a, int(y.size)


def judge_5a_alpha(ctx: Ctx) -> list[dict]:
    """5-a κ 依存（主判定は α 間のコントラスト）。"""
    arms = {0.5: "E_a0p5_1216", 1.0: "Enull_1216", 2.0: "E_a2_1216",
            4.0: "E_a4_1216"}
    if ctx.get("Enull_1216") is None and ctx.get("E_1216") is not None:
        arms[1.0] = "E_1216"
    out = []
    n_seeds = ctx.n_seeds(*arms.values())
    if n_seeds == 0:
        return [not_run("E_a*", "4.5-a alpha contrast", "primary",
                        "Delta B(alpha) = B(alpha) - B(1)", "NOT_DETERMINED",
                        "no ELU arm available")]
    draws = boot_draws(n_seeds)
    base = arms[1.0]
    # 副次 REPORT: B の 4 通り
    for alpha, name in sorted(arms.items()):
        if ctx.get(name) is None:
            continue
        for lag in (False, True):
            for sub in (False, True):
                point, icept, n = b_slope(ctx, name, lag=lag, submerged=sub)
                _, lo, hi = boot_ci(
                    lambda idx, name=name, lag=lag, sub=sub:
                        b_slope(ctx, name, lag=lag, submerged=sub, seed_idx=idx)[0],
                    n_seeds, draws=draws)
                tag = ("lag v" if lag else "simultaneous v") + (
                    " + fully submerged" if sub else "")
                out.append(row(name, "4.5-a B (secondary)", "secondary",
                               f"B ({tag})", ctx.tail(name), "ALIVE",
                               "REPORT_ONLY", n=n,
                               death_rate=ctx.get(name).death_rate(ctx.tail(name)),
                               point=point, ci=(lo, hi),
                               note=f"alpha={alpha}, intercept={icept:.3f}, "
                                    f"C={C_CONST} fixed"))
    # 主判定
    deltas, labels = {}, {}
    alpha_limited = []
    for alpha in (0.5, 2.0, 4.0):
        name = arms[alpha]
        if ctx.get(name) is None or ctx.get(base) is None:
            deltas[alpha] = None
            continue
        _cs, ja, jb = common_seed_rows(ctx.get(name), ctx.get(base))
        n_pair = len(ja)

        def dstat(idx, name=name, ja=ja, jb=jb):
            a = b_slope(ctx, name, lag=True, submerged=True, seed_idx=idx,
                        rows=ja)[0]
            b = b_slope(ctx, base, lag=True, submerged=True, seed_idx=idx,
                        rows=jb)[0]
            return a - b
        point, lo, hi = boot_ci(dstat, n_pair,
                                draws=(draws if n_pair == n_seeds else None))
        deltas[alpha] = (point, lo, hi)
        gates = arm_gates(ctx, name)
        gates["G4"] = g4_comparable(ctx.get(name).death_rate(ctx.tail(name)),
                                    ctx.get(base).death_rate(ctx.tail(base)))
        gates["G5"] = "PASS"

        def _lab(ci):
            return "CONSISTENT" if ci_within(ci, -0.3, 0.3) else (
                "HIGHER" if (np.isfinite(ci[0]) and ci[0] > 0.3) else "OTHER")

        def dstat_all(idx, name=name, ja=ja, jb=jb):
            a = b_slope(ctx, name, lag=True, submerged=True, alive=False,
                        seed_idx=idx, rows=ja)[0]
            b = b_slope(ctx, base, lag=True, submerged=True, alive=False,
                        seed_idx=idx, rows=jb)[0]
            return a - b
        _pa, lo_all, hi_all = boot_ci(dstat_all, n_pair,
                                      draws=(draws if n_pair == n_seeds else None))
        lab = _lab((lo, hi))
        gates["G3"] = g3_agreement(lab, _lab((lo_all, hi_all)))

        def _dpoint(w, name=name, ja=ja, jb=jb, n_pair=n_pair):
            a = b_slope(ctx, name, lag=False, submerged=True, window=w,
                        rows=ja)[0]
            b = b_slope(ctx, base, lag=False, submerged=True, window=w,
                        rows=jb)[0]
            return a - b
        g2, svals = g2_settled(_dpoint, hi - lo)
        gates["G2"] = g2
        reason = blocked(gates)
        # spec §4.6: NaN でない逸走は 5-a の **FAIL 側の証拠**。腕を外さない。
        limited = (gates["G6"] == "ARM_RUNAWAY")
        if limited:
            alpha_limited.append(alpha)
            out.append(row(name, f"4.5-a ALPHA_LIMITED(alpha={alpha})", "primary",
                           "runaway |median zbar| > 50 with no NaN", ctx.tail(name),
                           "ALIVE & zmax<0", "ALPHA_LIMITED",
                           death_rate=ctx.get(name).death_rate(ctx.tail(name)),
                           point=point, ci=(lo, hi), gates=gates,
                           n_seeds_dropped=ctx.dropped(name),
                           note="spec §4.6: 発散した alpha を外して残りで判定する"
                                "のは PASS 方向にしか働かない事後選択なので、腕は"
                                "掃引に残したまま FAIL 側の証拠として立てる"))
        labels[alpha] = None if (reason or gates["G3"] != "PASS"
                                 or g2 == "NOT_SETTLED") else lab
        out.append(row(name, f"4.5-a Delta B(alpha={alpha})", "primary",
                       "B(alpha) - B(1) [lag v, fully submerged]",
                       ctx.tail(name), "ALIVE & zmax<0", lab, n=n_pair,
                       death_rate=ctx.get(name).death_rate(ctx.tail(name)),
                       point=point, ci=(lo, hi), gates=gates,
                       n_seeds_dropped=ctx.dropped(name),
                       note=(reason or "") + f"; base arm {base}; ALL label "
                            f"{_lab((lo_all, hi_all))}; "
                            f"settle {['%.3f' % v for v in svals]}"
                            + ("; ARM_RUNAWAY -> ALPHA_LIMITED" if limited else "")))
    good = [a for a in (0.5, 2.0, 4.0) if labels.get(a) is not None]
    if alpha_limited:
        # 逸走した α が 1 本でもあれば掃引全体が FAIL 側（spec §4.6）。
        overall = "ALPHA_CONTRAST_INCONSISTENT"
    elif len(good) < 2:
        overall = "NOT_DETERMINED"
    elif all(labels[a] == "CONSISTENT" for a in good) and len(good) == 3:
        overall = "ALPHA_CONTRAST_CONSISTENT"
    elif (len(good) == 3
          and deltas[0.5][0] <= deltas[2.0][0] <= deltas[4.0][0]
          and np.isfinite(deltas[4.0][1]) and deltas[4.0][1] > 0.3):
        overall = "ALPHA_CONTRAST_MONOTONE"
    else:
        overall = "ALPHA_CONTRAST_INCONSISTENT"
    out.append(row("E_a*", "4.5-a alpha contrast", "primary",
                   "Delta B(alpha) for alpha in {0.5,2,4}", None,
                   "ALIVE & zmax<0", overall,
                   note=f"per-alpha {labels}"
                        + (f"; ALPHA_LIMITED at alpha {alpha_limited} "
                           f"(runaway kept in the sweep, spec §4.6)"
                           if alpha_limited else "")))
    # 副次: lag プロファイル・切片・信頼性比・自由 C
    out.extend(_5a_secondary(ctx, arms, n_seeds, draws))
    return out


def _5a_secondary(ctx: Ctx, arms: dict, n_seeds: int, draws) -> list[dict]:
    out = []
    for alpha, name in sorted(arms.items()):
        arm = ctx.get(name)
        if arm is None:
            continue
        win, lwin = ctx.tail(name), ctx.lag(name)
        # lag プロファイル（0 / 100 / 200 タスク）
        prof = []
        for back in (0, 100, 200):
            w = (win[0] - back, win[1] - back)
            zmax = arm.unit_window("layer1_zmax", win)
            v = arm.unit_window("layer1_v_unit", w)
            mask = arm.alive(win) & (zmax < 0)
            y = pooled(zmax, mask, np.arange(zmax.shape[0]))
            x = np.log1p(C_CONST * pooled(v, mask, np.arange(v.shape[0])) ** 2)
            prof.append(-ols(y, x)[1])
        out.append(row(name, "4.5-a lag profile", "report", "B at lag 0/100/200",
                       win, "ALIVE & zmax<0", "REPORT_ONLY",
                       point=prof[0], note=f"B = {['%.3f' % p for p in prof]}"))
        # 切片（理論: median(zmax + ln(1+kappa)) が [-1,+1]）
        zmax = arm.unit_window("layer1_zmax", win)
        v = arm.unit_window("layer1_v_unit", lwin)
        mask = arm.alive(win) & (zmax < 0)
        resid = zmax + np.log1p(C_CONST * v ** 2)
        point, ci, n = median_stat_ci(resid, mask, draws)
        out.append(row(name, "4.5-a intercept", "report",
                       "median(zmax + ln(1+kappa))", win, "ALIVE & zmax<0",
                       "IN_BAND" if ci_within(ci, -1.0, 1.0) else "OUT_OF_BAND",
                       n=n, point=point, ci=ci))
        # 信頼性比（EIV 補正）
        recs = arm.unit_records("layer1_v_unit", lwin)
        xi = np.log1p(C_CONST * recs ** 2)
        within = float(np.mean(xi.var(axis=1)[mask])) if mask.any() else float("nan")
        between = float(np.var(xi.mean(axis=1)[mask])) if mask.any() else float("nan")
        rel = 1.0 - within / between if between else float("nan")
        b_pt = b_slope(ctx, name, lag=True, submerged=True)[0]
        out.append(row(name, "4.5-a reliability-corrected B", "report",
                       "B / rho_reliability", win, "ALIVE & zmax<0", "REPORT_ONLY",
                       point=b_pt / rel if rel else float("nan"),
                       note=f"reliability rho={rel:.4f} "
                            f"(within={within:.4g}, between={between:.4g})"))
        # 自由 C フィット（REPORT_ONLY・profile RSS）
        best_c, best_rss = float("nan"), np.inf
        y = pooled(zmax, mask, np.arange(zmax.shape[0]))
        for cval in np.geomspace(0.5, 200.0, 200):
            x = np.log1p(cval * pooled(v, mask, np.arange(v.shape[0])) ** 2)
            a, b = ols(y, x)
            rss = float(((a + b * x - y) ** 2).sum())
            if rss < best_rss:
                best_rss, best_c = rss, float(cval)
        out.append(row(name, "4.5-a free-C fit", "report", "profile-RSS C-hat",
                       win, "ALIVE & zmax<0", "REPORT_ONLY", point=best_c,
                       note=f"registered C={C_CONST} (closed form); "
                            f"free-C fit is REPORT_ONLY by spec §4.5-a"))
    return out


def equilibrium_zmax(ctx: Ctx, name: str) -> dict | None:
    """5-b 数値平衡: 実測支持形状と κ から ∂R/∂z̄ = 0 を数値で解く。"""
    arm = ctx.get(name)
    if arm is None:
        return None
    if not (arm.has("layer1_w_free") and arm.has("layer1_w_free_step")):
        return {"missing": "layer1_w_free"}
    win, lwin = ctx.tail(name), ctx.lag(name)
    wf = arm.aux_window("layer1_w_free", "layer1_w_free_step", win).mean(axis=1)
    signs = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.float64) * 2 - 1
    offs = np.einsum("pf,shf->psh", signs, wf * 0.5)      # (32, S, h)
    half = np.abs(wf).sum(axis=-1) * 0.5                  # (S, h)
    v = arm.unit_window("layer1_v_unit", lwin)
    kappa = C_CONST * v ** 2
    meta = arm.meta
    phi, dphi = act_numpy(str(meta.get("activation", "")), float(meta.get("dial", 1.0)),
                          ctx.cfg_act)
    S, h = kappa.shape
    o = offs.reshape(32, -1)                              # (32, S*h)
    k = kappa.reshape(-1)
    zb = arm.unit_window("layer1_zbar", win).reshape(-1)
    grid = np.arange(-60.0, 20.0001, 0.05)
    best = np.full(o.shape[1], np.nan)
    best_r = np.full(o.shape[1], np.inf)
    for g in grid:
        z = g + o
        r = (phi(z) ** 2).mean(axis=0) + k * (dphi(z) ** 2).mean(axis=0)
        upd = r < best_r
        best_r[upd] = r[upd]
        best[upd] = g
    for _ in range(2):                                    # 粗→細の 2 段
        step = 0.05 if _ == 0 else 0.002
        fine = np.arange(-step, step + 1e-12, step / 25.0)
        cur = best.copy()
        for dg in fine:
            z = (cur + dg) + o
            r = (phi(z) ** 2).mean(axis=0) + k * (dphi(z) ** 2).mean(axis=0)
            upd = r < best_r
            best_r[upd] = r[upd]
            best[upd] = (cur + dg)[upd]
    zbar_star = best.reshape(S, h)
    return {"zbar_star": zbar_star, "zmax_star": zbar_star + half,
            "half": half, "kappa": kappa, "zbar_obs": zb.reshape(S, h)}


def judge_5b_equilibrium(ctx: Ctx) -> list[dict]:
    """5-b 数値平衡（主要判定）。"""
    families = {"elu": [(0.5, "E_a0p5_1216"), (1.0, "Enull_1216"),
                        (2.0, "E_a2_1216"), (4.0, "E_a4_1216")],
                "shelf": [(0.5, "SH_d0p5_1216"), (1.0, "SH_d1_1216"),
                          (2.0, "SH_d2_1216"), (3.0, "SH_d3_1216")]}
    out, ok_arms, pred, obs = [], 0, {}, {}
    missing = []
    for fam, items in families.items():
        for key, name in items:
            arm = ctx.get(name)
            if arm is None:
                out.append(not_run(name, "4.5-b numerical equilibrium", "primary",
                                   "median_i[zmax - zmax*]", "NOT_DETERMINED",
                                   "arm logs missing"))
                missing.append(name)
                continue
            eq = equilibrium_zmax(ctx, name)
            if eq is None or "missing" in eq:
                out.append(not_run(name, "4.5-b numerical equilibrium", "primary",
                                   "median_i[zmax - zmax*]", "NOT_DETERMINED",
                                   f"column {eq['missing']} absent in the logs"))
                missing.append(name)
                continue
            win = ctx.tail(name)
            diff = arm.unit_window("layer1_zmax", win) - eq["zmax_star"]
            mask = arm.alive(win)
            point, ci, n = median_stat_ci(diff, mask)
            _pa, ci_all, _na = median_stat_ci(diff, None)
            gates = arm_gates(ctx, name)
            inside = ci_within(ci, -1.0, 1.0)
            gates["G3"] = g3_agreement(str(inside), str(ci_within(ci_all, -1.0, 1.0)))
            gates["G5"] = "PASS"
            g2, svals = g2_settled(
                lambda w, arm=arm, eq=eq: float(np.median(
                    (arm.unit_window("layer1_zmax", w) - eq["zmax_star"])[arm.alive(w)])),
                ci[1] - ci[0])
            gates["G2"] = g2
            if gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
                inside = False
            if not blocked(gates) and gates["G3"] == "PASS" and g2 != "NOT_SETTLED":
                ok_arms += 1
                pred[(fam, key)] = float(np.median(eq["zmax_star"][mask]))
                obs[(fam, key)] = float(np.median(
                    arm.unit_window("layer1_zmax", win)[mask]))
            out.append(row(name, "4.5-b numerical equilibrium", "primary",
                           "median_i[zmax - zmax*]", win, "ALIVE",
                           "WITHIN_1.0" if inside else "OUTSIDE_1.0", n=n,
                           death_rate=arm.death_rate(win), point=point, ci=ci,
                           gates=gates, n_seeds_dropped=ctx.dropped(name),
                           note=f"median predicted zmax*="
                                f"{np.median(eq['zmax_star'][mask]):.3f}, "
                                f"median kappa={np.median(eq['kappa'][mask]):.3f}; "
                                f"ALL CI [{ci_all[0]:.3f},{ci_all[1]:.3f}]; "
                                f"settle {['%.3f' % v for v in svals]}"
                                + ("; " + blocked(gates) if blocked(gates) else "")))
    order_ok = True
    for fam in families:
        keys = sorted(k for (f, k) in pred if f == fam)
        if len(keys) < 2:
            continue
        p = [pred[(fam, k)] for k in keys]
        o = [obs[(fam, k)] for k in keys]
        if kendall_tau(p, o) != 1.0:
            order_ok = False
    inside_all = all(r["label"] == "WITHIN_1.0" for r in out
                     if r["judgment"] == "4.5-b numerical equilibrium"
                     and r["label"] in ("WITHIN_1.0", "OUTSIDE_1.0"))
    if ok_arms < 5:
        overall = "NOT_DETERMINED"
    elif order_ok and inside_all and ok_arms >= 8:
        overall = "EQUILIBRIUM_PREDICTED"
    elif order_ok:
        overall = "EQUILIBRIUM_ORDER_ONLY"
    else:
        overall = "EQUILIBRIUM_OFF"
    out.append(row("ELU x4 + shelf x4", "4.5-b overall", "primary",
                   "CI within +-1.0 for all 8 arms and Kendall tau = 1", None,
                   "ALIVE", overall,
                   n=ok_arms, note=f"determined arms={ok_arms}/8, order_ok={order_ok}, "
                                   f"all within +-1.0={inside_all}"
                                   + (f"; missing {missing}" if missing else "")))
    return out


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    if n < 2:
        return float("nan")
    num = den = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[j] - a[i])
            sb = np.sign(b[j] - b[i])
            if sa != 0 and sb != 0:
                num += 1 if sa == sb else -1
                den += 1
    return float(num / den) if den else float("nan")


def judge_5c_vfreeze(ctx: Ctx) -> list[dict]:
    """5-c v 凍結（確証的な因果検定）。"""
    ref_name = "Enull_1216" if ctx.get("Enull_1216") else "E_1216"
    ref = ctx.get(ref_name)
    arm = ctx.get("Evf1_1216")
    out = []
    if arm is None or ref is None:
        return [not_run("Evf1_1216", "4.5-c v-freeze", "confirmatory",
                        "median zmax(Evf1) - median zmax(Enull)",
                        "NOT_DETERMINED", "arm or reference logs missing")]
    gates = arm_gates(ctx, "Evf1_1216")
    win = ctx.tail("Evf1_1216")
    gates["G4"] = g4_comparable(arm.death_rate(win),
                                ref.death_rate(ctx.tail(ref_name)))
    gates["G5"] = "PASS"
    d = paired_delta(ctx, "Evf1_1216", ref_name)
    d_all = paired_delta(ctx, "Evf1_1216", ref_name, alive=False)
    unfit_arm = float(np.median(arm.run_window("unfit", win)))
    unfit_ref = float(np.median(ref.run_window("unfit", ctx.tail(ref_name))))
    unfit_bad = np.isfinite(unfit_ref) and unfit_arm > 3.0 * unfit_ref

    def _label(res, alive: bool):
        traj_ = []
        for w in TRAJ:
            v = arm.unit_window("layer1_zmax", w)
            m = arm.alive(w) if alive else np.ones_like(v, dtype=bool)
            traj_.append(median_stat_ci(v, m)[:2])
        dec = (traj_[0][0] > traj_[1][0] > traj_[2][0]
               and traj_[0][1][0] > traj_[1][1][1]
               and traj_[1][1][0] > traj_[2][1][1])
        if np.isfinite(res["ci"][1]) and res["ci"][1] < -1.0 and dec:
            return "WELL_FROM_READOUT", traj_, dec
        if ci_within(res["ci"], -0.5, 0.5):
            return "WELL_INDEPENDENT_OF_READOUT", traj_, dec
        return "WELL_PARTIAL", traj_, dec

    label, traj, decreasing = _label(d, True)
    label_all, _t2, _d2 = _label(d_all, False)
    gates["G3"] = g3_agreement(label, label_all)
    g2, svals = g2_settled(
        lambda w: _delta_in_window(ctx, "Evf1_1216", ref_name, w),
        d["ci"][1] - d["ci"][0])
    gates["G2"] = g2
    reason = blocked(gates)
    # G2 は「まだ落ち着いていないのに PASS/FAIL を出す」のを止めるゲートだが、
    # `WELL_FROM_READOUT` は**単調に沈み続けること（逃走）そのもの**が登録された
    # 述語なので、この枝だけは G2 の NOT_SETTLED を前提として扱い、降格しない
    # （降格すると spec §4.5-c の PASS 側ラベルが原理的に出せなくなる）。G2 の値は
    # 行にそのまま残す。
    g2_blocks = (g2 == "NOT_SETTLED" and label != "WELL_FROM_READOUT")
    if reason or unfit_bad or gates["G3"] != "PASS" or g2_blocks:
        label = "NOT_DETERMINED"
    out.append(row("Evf1_1216", "4.5-c v-freeze", "confirmatory",
                   "median_i[zmax(Evf1) - zmax(Enull)]", win, "ALIVE (both)",
                   label, n=d["n"], death_rate=arm.death_rate(win),
                   point=d["point"], ci=d["ci"], gates=gates,
                   n_seeds_dropped=ctx.dropped("Evf1_1216"),
                   note=f"trajectory median zmax {['%.3f' % t[0] for t in traj]} "
                        f"(monotone decrease with disjoint CI = {decreasing}); "
                        f"unfit {unfit_arm:.4g} vs ref {unfit_ref:.4g} "
                        f"(3x cutoff {'FAILED' if unfit_bad else 'ok'}); "
                        f"ALL label {label_all}; "
                        f"settle {['%.3f' % v for v in svals]}"
                        + ("; " + reason if reason else "")))
    vf4 = ctx.get("Evf4_1216")
    if vf4 is not None:
        d4 = paired_delta(ctx, "Evf4_1216", "Evf1_1216")
        kap = C_CONST * arm.unit_window("layer1_v_unit", ctx.lag("Evf1_1216")) ** 2
        m = arm.alive(win)
        shift = float(np.median(-np.log((1 + 16 * kap[m]) / (1 + kap[m]))))
        out.append(row("Evf4_1216", "4.5-c v-freeze x4", "secondary",
                       "median_i[zmax(Evf4) - zmax(Evf1)]", win, "ALIVE (both)",
                       "REPORT_ONLY", n=d4["n"], point=d4["point"], ci=d4["ci"],
                       note=f"R prediction if kappa is exogenous: "
                            f"median -ln((1+16k)/(1+k)) = {shift:.3f}"))
    else:
        out.append(not_run("Evf4_1216", "4.5-c v-freeze x4", "secondary",
                           "median_i[zmax(Evf4) - zmax(Evf1)]", "NOT_RUN",
                           "arm logs missing"))
    return out


def judge_5d_eta(ctx: Ctx) -> list[dict]:
    """5-d η（揺らぎのスケーリングが主判定）。"""
    out = []
    families = (("elu", ("Elr0p005_1216", "Enull_1216", "Elr0p02_1216"), "primary"),
                ("leaky", ("LRlr0p005_1216", "LRnull_1216", "LRlr0p02_1216"),
                 "secondary"))
    etas = np.array([0.005, 0.01, 0.02])
    main_label = "NOT_DETERMINED"
    for fam, names, role in families:
        names = tuple(n if ctx.get(n) else
                      {"Enull_1216": "E_1216", "LRnull_1216": "LR_1216"}.get(n, n)
                      for n in names)
        missing = [n for n in names if ctx.get(n) is None]
        if len(missing) >= 2:
            out.append(not_run(fam, "4.5-d fluctuation scaling", role,
                               "log-log slope of sd(zmax) vs eta",
                               "NOT_DETERMINED", f"missing arms {missing}"))
            continue
        n_seeds = ctx.n_seeds(*names)
        draws = boot_draws(n_seeds)

        def sd_med(name, idx, alive=True, window=None):
            arm = ctx.get(name)
            if arm is None:
                return float("nan")
            w = ctx.tail(name) if window is None else tuple(window)
            sd = arm.unit_window("layer1_zmax", w, reduce="std")
            m = arm.alive(w) if alive else None
            idx = np.clip(np.asarray(idx, dtype=int), 0, sd.shape[0] - 1)
            x = pooled(sd, m, idx)
            return float(np.median(x)) if x.size else float("nan")

        def slope(idx, alive=True, window=None):
            ys = np.array([sd_med(n, idx, alive, window) for n in names])
            ok = np.isfinite(ys) & (ys > 0)
            if ok.sum() < 2:
                return float("nan")
            return ols(np.log(ys[ok]), np.log(etas[ok]))[1]
        point, lo, hi = boot_ci(slope, n_seeds, draws=draws)
        _pa, lo_all, hi_all = boot_ci(lambda idx: slope(idx, alive=False),
                                      n_seeds, draws=draws)

        def _lab(ci):
            if ci_within(ci, 0.3, 0.7):
                return "FLUCT_SQRT_ETA"
            if ci_within(ci, 0.8, 1.2):
                return "FLUCT_LINEAR_ETA"
            return "FLUCT_OTHER"
        label = _lab((lo, hi))
        gates = {"G3": g3_agreement(label, _lab((lo_all, hi_all))), "G5": "PASS"}
        g2, svals = g2_settled(
            lambda w: slope(np.arange(n_seeds), window=w), hi - lo)
        gates["G2"] = g2
        n_bad = sum(1 for n in names if ctx.get(n) is None
                    or blocked(arm_gates(ctx, n)))
        if n_bad >= 2 or gates["G3"] != "PASS" or g2 == "NOT_SETTLED":
            label = "NOT_DETERMINED"
        if role == "primary":
            main_label = label
        out.append(row("+".join(names), "4.5-d fluctuation scaling", role,
                       "d ln sd(zmax) / d ln eta", "tail", "ALIVE", label,
                       point=point, ci=(lo, hi), gates=gates,
                       n_seeds_dropped=sum(ctx.dropped(n) for n in names),
                       note=f"sd medians "
                            f"{['%.4g' % sd_med(n, np.arange(n_seeds)) for n in names]} "
                            f"at eta {list(etas)}; {n_bad} arms failed G1/G6; "
                            f"ALL label {_lab((lo_all, hi_all))}; "
                            f"settle {['%.3f' % v for v in svals]}"))
    # 位置の η 不変（REPORT・eta*step 一致）
    ref = ctx.get("Enull_1216") or ctx.get("E_1216")
    for name, win in (("Elr0p005_1216", (951, 1000)), ("Elr0p02_1216", (201, 250)),
                      ("LRlr0p005_1216", (951, 1000)), ("LRlr0p02_1216", (201, 250))):
        arm = ctx.get(name)
        if arm is None or ref is None:
            out.append(not_run(name, "4.5-d position eta-invariance", "report",
                               "median zmax at matched eta*step", "NOT_RUN",
                               "arm logs missing"))
            continue
        if arm.window_count(win) == 0:
            out.append(not_run(name, "4.5-d position eta-invariance", "report",
                               "median zmax at matched eta*step", "NOT_DETERMINED",
                               f"tasks {win[0]}-{win[1]} are beyond this arm's horizon"))
            continue
        base = ctx.get("LRnull_1216") or ctx.get("LR_1216") if name.startswith("LR") else ref
        v, m = arm.unit_window("layer1_zmax", win), arm.alive(win)
        bw = ctx.tail(base.name)
        vb, mb = base.unit_window("layer1_zmax", bw), base.alive(bw)
        n = min(v.shape[0], vb.shape[0])

        def stat(idx):
            idx = np.asarray(idx)
            return (float(np.median(pooled(v, m, idx)))
                    - float(np.median(pooled(vb, mb, idx))))
        point, lo, hi = boot_ci(stat, n)
        out.append(row(name, "4.5-d position eta-invariance", "report",
                       "median zmax(arm) - median zmax(base) at matched eta*step",
                       win, "ALIVE", "IN_BAND" if ci_within((lo, hi), -0.3, 0.3)
                       else "OUT_OF_BAND", point=point, ci=(lo, hi),
                       note=f"base {base.name}; eta*step = 50000"))
    return out


def judge_5f_scale(ctx: Ctx) -> list[dict]:
    """5-f スケール（LRs0p5 / LRs2）。"""
    ref_name = "LRnull_1216" if ctx.get("LRnull_1216") else "LR_1216"
    out, zmax_ok, zbar_shift = [], [], []
    for name in ("LRs0p5_1216", "LRs2_1216"):
        arm = ctx.get(name)
        if arm is None or ctx.get(ref_name) is None:
            out.append(not_run(name, "4.5-f scale", "primary",
                               "median zmax(arm) - median zmax(LRnull)",
                               "NOT_DETERMINED", "arm or reference logs missing"))
            zmax_ok.append(None)
            zbar_shift.append(None)
            continue
        gates = arm_gates(ctx, name)
        dz = paired_delta(ctx, name, ref_name, key="layer1_zmax")
        db = paired_delta(ctx, name, ref_name, key="layer1_zbar")
        dz_all = paired_delta(ctx, name, ref_name, key="layer1_zmax", alive=False)
        gates["G4"] = g4_comparable(dz["death_a"], dz["death_b"])
        gates["G5"] = "PASS"
        gates["G3"] = g3_agreement(str(ci_within(dz["ci"], -0.5, 0.5)),
                                   str(ci_within(dz_all["ci"], -0.5, 0.5)))
        g2, svals = g2_settled(
            lambda w, name=name: _delta_in_window(ctx, name, ref_name, w),
            dz["ci"][1] - dz["ci"][0])
        gates["G2"] = g2
        reason = blocked(gates)
        bad = bool(reason) or gates["G3"] != "PASS" or g2 == "NOT_SETTLED"
        zmax_ok.append(None if bad else ci_within(dz["ci"], -0.5, 0.5))
        zbar_shift.append(None if bad else (not ci_within(db["ci"], -0.5, 0.5)))
        out.append(row(name, "4.5-f scale", "primary",
                       "median_i[zmax(arm) - zmax(LRnull)]", dz["win_a"],
                       "ALIVE (both)", "REPORT" if bad else
                       ("ZMAX_INVARIANT" if zmax_ok[-1] else "ZMAX_MOVED"),
                       n=dz["n"], death_rate=dz["death_a"], point=dz["point"],
                       ci=dz["ci"], gates=gates,
                       n_seeds_dropped=ctx.dropped(name),
                       note=f"zbar diff={db['point']:.3f} "
                            f"[{db['ci'][0]:.3f},{db['ci'][1]:.3f}]; "
                            f"settle {['%.3f' % v for v in svals]}"
                            + ("; " + reason if reason else "")))
    if None in zmax_ok or not zmax_ok:
        overall = "NOT_DETERMINED"
    elif all(zmax_ok) and any(zbar_shift):
        overall = "P5_SCALE_LAW_INVARIANT"
    elif all(zmax_ok):
        overall = "P5_SCALE_FULLY_INVARIANT"
    else:
        overall = "P5_SCALE_LAW_BROKEN"
    out.append(row("LRs0p5+LRs2", "4.5-f overall", "primary",
                   "zmax invariance + zbar shift", None, "ALIVE", overall,
                   note=f"zmax within +-0.5: {zmax_ok}; zbar moved: {zbar_shift}"))
    return out


def _moment_G(ctx: Ctx, name: str, window):
    """G_i = 2 m_phidphi + 2 kappa_i m_dphiddphi（窓平均・(S,h)）。"""
    arm = ctx.get(name)
    need = ("layer1_m_phidphi", "layer1_m_dphiddphi", "layer1_moment_step")
    if arm is None or not all(arm.has(k) for k in need):
        return None
    a = arm.aux_window("layer1_m_phidphi", "layer1_moment_step", window).mean(axis=1)
    b = arm.aux_window("layer1_m_dphiddphi", "layer1_moment_step", window).mean(axis=1)
    kappa = C_CONST * arm.unit_window("layer1_v_unit", ctx.lag(name)) ** 2
    return 2.0 * a + 2.0 * kappa * b


def judge_5g_stationarity(ctx: Ctx) -> list[dict]:
    """5-g 停留残差の直接検定（確証的）。"""
    out, fam_pass = [], 0
    fam_judged: set[str] = set()             # **腕**ではなく**族**を数える（spec §4.5-g）
    for fam, names in FAMILY_ARMS.items():
        fam_ok = None
        for name in names:
            arm = ctx.get(name)
            if arm is None:
                continue
            win, lag = ctx.tail(name), ctx.lag(name)
            g_tail = _moment_G(ctx, name, win)
            g_lag = _moment_G(ctx, name, lag)
            if g_tail is None or g_lag is None:
                out.append(not_run(name, "4.5-g stationarity residual",
                                   "confirmatory", "median |G_i|",
                                   "NOT_DETERMINED",
                                   "moment columns absent in the logs"))
                continue
            if not arm.has("layer1_dzbar"):
                out.append(not_run(name, "4.5-g stationarity residual",
                                   "confirmatory", "median |G_i|",
                                   "NOT_DETERMINED",
                                   "layer1_dzbar absent — (ii) cannot be formed"))
                continue
            gates = arm_gates(ctx, name)
            reason = blocked(gates)
            mask = arm.alive(win)
            pt, ci, n = median_stat_ci(np.abs(g_tail), mask)
            lag_pt = float(np.median(np.abs(g_lag)[mask]))
            cond_i = np.isfinite(ci[1]) and ci[1] < lag_pt / 3.0
            # (ii) dzbar と -eta G の対応（末尾 20 タスク・1000-step 記録）
            eta = float(arm.payload("lr_used", 0.01) or 0.01)
            rho, ratio = _dzbar_vs_G(ctx, name, eta)
            cond_ii = (np.isfinite(rho[1]) and rho[1] > 0.3
                       and ci_within(ratio, 0.3, 3.0))
            # G3: 除外なし（ALL）でも同じラベルか
            _pa, ci_all, _na = median_stat_ci(np.abs(g_tail), None)
            lag_all = float(np.median(np.abs(g_lag)))
            cond_i_all = np.isfinite(ci_all[1]) and ci_all[1] < lag_all / 3.0
            lab_alive = "STATIONARY" if (cond_i and cond_ii) else "NOT_STATIONARY"
            lab_all = "STATIONARY" if (cond_i_all and cond_ii) else "NOT_STATIONARY"
            gates["G3"] = g3_agreement(lab_alive, lab_all)
            gates["G5"] = "PASS"
            g2, svals = g2_settled(
                lambda w, name=name: nanmedian(
                    np.abs(_moment_G(ctx, name, w))[arm.alive(w)]),
                ci[1] - ci[0])
            gates["G2"] = g2
            judgeable = not (reason or gates["G3"] != "PASS" or g2 == "NOT_SETTLED")
            ok = bool(cond_i and cond_ii)
            if judgeable:
                # 判定できた腕だけが族の合否に入る（判定できない腕を FAIL 側に
                # 数えると FROZEN の腕が族を落としてしまう・spec §3.6）。
                fam_ok = ok if fam_ok is None else (fam_ok and ok)
            out.append(row(name, "4.5-g stationarity residual", "confirmatory",
                           "median |G_i| (tail) vs lag; Spearman(dzbar, -eta G)",
                           win, "ALIVE",
                           ("STATIONARY" if ok else "NOT_STATIONARY")
                           if judgeable else "NOT_DETERMINED",
                           n=n, death_rate=arm.death_rate(win), point=pt, ci=ci,
                           gates=gates, n_seeds_dropped=ctx.dropped(name),
                           note=f"lag median |G|={lag_pt:.4g} (1/3 = {lag_pt/3:.4g}); "
                                f"(i)={cond_i}; Spearman median={rho[0]:.3f} "
                                f"CI[{rho[1]:.3f},{rho[2]:.3f}]; ratio CI"
                                f"[{ratio[0]:.3f},{ratio[1]:.3f}]; (ii)={cond_ii}; "
                                f"ALL label {lab_all}; "
                                f"settle {['%.3f' % v for v in svals]}"
                                + ("; " + reason if reason else "")))
        if fam_ok is not None:
            fam_judged.add(fam)
            fam_pass += int(bool(fam_ok))
    if len(fam_judged) < 3:
        # spec §4.5-g: 「判定できる**族**が 3 未満」→ NOT_DETERMINED。
        overall = "NOT_DETERMINED"
    elif fam_pass >= 5:
        overall = "STATIONARITY_DIRECT_PASS"
    elif fam_pass >= 3:
        overall = "STATIONARITY_DIRECT_PARTIAL"
    else:
        overall = "STATIONARITY_DIRECT_FAIL"
    out.append(row("5 families", "4.5-g overall", "confirmatory",
                   "families passing (i) and (ii)", None, "ALIVE", overall,
                   n=fam_pass,
                   note=f"judgeable families ({len(fam_judged)}/5): "
                        f"{sorted(fam_judged)}; passing = {fam_pass}"))
    return out


def _dzbar_vs_G(ctx: Ctx, name: str, eta: float):
    """末尾 20 タスクの密なモーメント記録で per-unit Spearman と中央値比。"""
    arm = ctx.get(name)
    msteps = arm.aux_steps(arm.seeds[0], "layer1_moment_step")
    last_task = int(msteps.max()) // PERIOD
    keep = msteps > (last_task - 20) * PERIOD
    rhos, dz_meds, g_meds = [], [], []
    for s in arm.seeds:
        ms = arm.aux_steps(s, "layer1_moment_step")[keep]
        steps = arm.steps(s)
        dzb = arm.col(s, "layer1_dzbar")
        a = arm.col(s, "layer1_m_phidphi")[keep]
        b = arm.col(s, "layer1_m_dphiddphi")[keep]
        v = arm.unit_window("layer1_v_unit", ctx.lag(name))[arm.seeds.index(s)]
        g = 2.0 * a + 2.0 * (C_CONST * v ** 2)[None, :] * b
        dz = np.zeros_like(g)
        prev = ms[0] - (ms[1] - ms[0]) if len(ms) > 1 else ms[0] - 1000
        for k, step in enumerate(ms):
            sel = (steps > prev) & (steps <= step)
            seg = dzb[sel]
            dz[k] = np.nansum(seg, axis=0) if seg.size else np.nan
            prev = step
        rhos.append(np.array([spearman(dz[:, u], -eta * g[:, u])
                              for u in range(g.shape[1])]))
        dz_meds.append(np.nanmedian(dz, axis=0))
        g_meds.append(np.nanmedian(-eta * g, axis=0))
    rhos = np.stack(rhos)
    dz_meds, g_meds = np.stack(dz_meds), np.stack(g_meds)
    win = ctx.tail(name)
    mask = arm.alive(win)
    n = rhos.shape[0]
    rho = boot_ci(lambda idx: nanmedian(pooled(rhos, mask, idx)), n)

    def ratio(idx):
        a = nanmedian(pooled(dz_meds, mask, idx))
        b = nanmedian(pooled(g_meds, mask, idx))
        return a / b if b else float("nan")
    r = boot_ci(ratio, n)
    return rho, (r[1], r[2])


def judge_5h_order(ctx: Ctx) -> list[dict]:
    """5-h 沈下の順序（REPORT_ONLY・|v| で書く）。"""
    out = []
    for name in ("SH_d1_1216", "SH_d2_1216", "SH_d3_1216"):
        arm = ctx.get(name)
        if arm is None:
            out.append(not_run(name, "4.5-h sink order", "report",
                               "Spearman(sink task, |v| 200 tasks earlier)",
                               "NOT_RUN", "arm logs missing"))
            continue
        d = act_depth(str(arm.meta.get("activation", "")), ctx.cfg_act)
        ts, vs = [], []
        for s in arm.seeds:
            steps = arm.steps(s)
            zmax = arm.col(s, "layer1_zmax")
            v = arm.col(s, "layer1_v_unit")
            hit = zmax <= -d
            has = hit.any(axis=0)
            first = np.where(has, hit.argmax(axis=0), -1)
            for u in np.flatnonzero(has):
                t_task = steps[first[u]] / PERIOD
                back = steps[first[u]] - 200 * PERIOD
                if back < 0:
                    continue
                j = int(np.argmin(np.abs(steps - back)))
                ts.append(t_task)
                vs.append(abs(v[j, u]))
        rho = spearman(np.array(ts), np.array(vs)) if len(ts) >= 3 else float("nan")
        out.append(row(name, "4.5-h sink order", "report",
                       "Spearman(sink task, |v| 200 tasks earlier)", "whole run",
                       "sunk units", "REPORT_ONLY", n=len(ts), point=rho,
                       note=f"depth d={d}; spec §4.5-h: R は棚族の kappa 依存を"
                            f"予測しないので反証に数えない"))
    return out


def judge_5i_fullbatch(ctx: Ctx) -> list[dict]:
    """5-i full-batch（REPORT_ONLY・帰属）。"""
    arm = ctx.get("FBLR_1216")
    ref = ctx.get("LRnull_1216") or ctx.get("LR_1216")
    if arm is None:
        return [not_run("FBLR_1216", "4.5-i full batch", "report",
                        "median zmax (full-batch) vs online", "NOT_RUN",
                        "arm logs missing (S-fb failed or not run)")]
    win = ctx.tail("FBLR_1216")
    v, m = arm.unit_window("layer1_zmax", win), arm.alive(win)
    point, ci, n = median_stat_ci(v, m)
    note = f"batch_mode={arm.payload('batch_mode', '')}"
    if ref is not None:
        rw = ctx.tail(ref.name)
        note += (f"; online reference {ref.name} median zmax="
                 f"{np.median(ref.unit_window('layer1_zmax', rw)[ref.alive(rw)]):.3f}")
    return [row("FBLR_1216", "4.5-i full batch", "report", "median zmax", win,
                "ALIVE", "REPORT_ONLY", n=n, death_rate=arm.death_rate(win),
                point=point, ci=ci, note=note)]


def p5_overall(rows: list[dict]) -> list[dict]:
    """命題 5 の総合ラベル（spec §4.5 の決定木）。"""
    def _label(judgment):
        for r in rows:
            if r["judgment"] == judgment:
                return r["label"]
        return "NOT_DETERMINED"
    g = _label("4.5-g overall")
    b = _label("4.5-b overall")
    c = _label("4.5-c v-freeze")
    if g == "NOT_DETERMINED" or b == "NOT_DETERMINED":
        overall = "R_NOT_DETERMINED"
    elif g == "STATIONARITY_DIRECT_PASS" and b == "EQUILIBRIUM_PREDICTED":
        overall = "R_SUPPORTED"
    elif g == "STATIONARITY_DIRECT_FAIL" and b == "EQUILIBRIUM_OFF":
        overall = "R_REFUTED"
    else:
        overall = "R_PARTIAL"
    mod = {"WELL_FROM_READOUT": "CAUSAL",
           "WELL_INDEPENDENT_OF_READOUT": "NONCAUSAL"}.get(c, "UNDETERMINED")
    return [row("(proposition 5)", "4.5 overall", "confirmatory",
                "decision tree over 5-g and 5-b", None, "ALIVE",
                f"{overall}+{mod}",
                note=f"5-g={g}, 5-b={b}, 5-c={c}; "
                     f"5-a/5-d/5-f/5-h/5-i は総合ラベルに入れず併記（spec §4.5）")]


# ---------------------------------------------------------------------------
# §4.6 発散・§5 の走内検査
# ---------------------------------------------------------------------------
def judge_46_divergence(ctx: Ctx) -> list[dict]:
    out = []
    for name, arm in sorted(ctx.arms.items()):
        if not arm.available:
            continue
        win = ctx.tail(name)
        bad = arm.nan_seeds(win)
        g6, txt = g6_divergence(ctx, name)
        out.append(row(name, "4.6 divergence", "report", "NaN seeds / runaway",
                       win, "", g6, n=len(arm.seeds) - len(bad),
                       death_rate=arm.death_rate(win),
                       point=float(len(bad)), gates={"G6": g6},
                       note=f"dropped seeds {bad}; {txt}"))
    return out


def report_alive_secondary(ctx: Ctx) -> list[dict]:
    """副次除外規則（spec §3.5）の REPORT。

    「``ALIVE`` = ``layer1_denom`` > 0.25 … 副次規則として半幅 $(z_{\\max}-\\bar z)>0.25$
    を **REPORT** し、$n$ が **3% 以上違ったら記録する**」。判定には使わない。
    """
    out = []
    for name, arm in sorted(ctx.arms.items()):
        if not arm.available:
            continue
        win = ctx.tail(name)
        try:
            n_primary = int(arm.alive(win).sum())
            n_secondary = int(arm.alive_secondary(win).sum())
        except (KeyError, IndexError):
            continue
        base = max(n_primary, 1)
        gap = abs(n_secondary - n_primary) / base
        flagged = bool(gap >= ALIVE_SECONDARY_N_GAP)
        out.append(row(name, "3.5 alive rule (secondary)", "report",
                       "n(denom>0.25) vs n(zmax-zbar>0.25)", win,
                       "ALIVE vs half-width", "N_GAP_RECORDED" if flagged
                       else "N_AGREES", n=n_primary,
                       death_rate=arm.death_rate(win), point=gap,
                       n_seeds_dropped=ctx.dropped(name),
                       note=f"primary n={n_primary}, secondary n={n_secondary}, "
                            f"relative gap={gap:.4f} "
                            f"(record threshold {ALIVE_SECONDARY_N_GAP:.0%})"))
    return out


def s_taut(ctx: Ctx, arm_name: str = "LRnull_1216") -> dict:
    """S-taut: 新旧の沈下定義の率（spec §5）。参照腕で恒真でないことを示す。"""
    arm = ctx.get(arm_name) or ctx.get("LR_1216")
    if arm is None:
        return {"arm": None, "note": "no reference arm"}
    win = ctx.tail(arm.name)
    zmax = arm.unit_window("layer1_zmax", win)
    zbar = arm.unit_window("layer1_zbar", win)
    mask = arm.alive(win)
    new = {d: float((zmax[mask] <= -d).mean()) for d in (0.5, 1.0, 2.0, 3.0)}
    old = {d: float((zbar[mask] <= -d - 1.0).mean()) for d in (0.5, 1.0, 2.0, 3.0)}
    ok_new = all(new[d] < 0.05 for d in (2.0, 3.0))
    ok_old = all(old[d] > 0.7 for d in (2.0, 3.0))
    return {"arm": arm.name, "window": win, "new": new, "old": old,
            "new_below_0.05_at_d2_d3": ok_new, "old_above_0.7_at_d2_d3": ok_old,
            "n": int(mask.sum())}


def s_ksnull(ctx: Ctx, arm_name: str = "LR_1216", key: str = "layer1_zbar") -> dict:
    """S-KSnull: 1-c の閾値になる KS $D$ の帰無分布（spec §5）。

    **統計量と同じ土俵で作る**（これを外すと 1-c は原理的に PASS できない）:
    1-c の統計量は「seed ごとの $D_s$（そのユニット集合 ≈100 対 ≈100）の中央値」
    なので、帰無分布も seed ごとの $D_s$ の**中央値**の分布にする。10 seed を復元
    抽出して 2 つのアンサンブルにする（§5）ところは同じで、対にした seed どうしで
    $D_s$ を取り、その中央値を 1 反復ぶんの値にする。相手 seed は自分自身を引かない
    ように取る（$D=0$ を人工的に混ぜないため）。
    プールしたアンサンブル（≈1000 対 ≈1000）の $D$ は `q95_pooled` に**併記**する
    （こちらは per-seed 統計量の閾値としては 2 倍ほど厳しすぎる）。
    """
    arm = ctx.get(arm_name)
    if arm is None:
        return {"q95": float("nan"), "note": f"{arm_name} logs missing"}
    win = ctx.tail(arm_name)
    vals = arm.unit_window(key, win)
    mask = arm.alive(win)
    n = vals.shape[0]
    units = [vals[i][mask[i]] for i in range(n)]
    rng = np.random.default_rng(BOOT_SEED)
    ds = np.empty(BOOT_N, dtype=np.float64)
    ds_pooled = np.empty(BOOT_N, dtype=np.float64)
    for k in range(BOOT_N):
        ia = rng.integers(0, n, size=n)
        ib = ((ia + rng.integers(1, max(n, 2), size=n)) % n if n > 1 else ia)
        ds[k] = float(np.median([ks_d(units[a], units[b])
                                 for a, b in zip(ia, ib)]))
        ds_pooled[k] = ks_d(pooled(vals, mask, ia), pooled(vals, mask, ib))
    return {"arm": arm_name, "window": win, "q95": float(np.percentile(ds, 95)),
            "median": float(np.median(ds)), "max": float(ds.max()),
            "q95_pooled": float(np.percentile(ds_pooled, 95)),
            "median_pooled": float(np.median(ds_pooled)),
            "n_rep": BOOT_N, "key": key,
            "unit": "median over seeds of the per-seed KS D (matches 4.1-c)"}


# ---------------------------------------------------------------------------
# 走らせ口
# ---------------------------------------------------------------------------
def build_ctx(cfg: dict, logdir: Path | None = None, seeds=range(10)) -> Ctx:
    """config の腕表＋committed 対照から Ctx を作る。"""
    arms: dict[str, ArmLog] = {}
    base = Path(logdir) if logdir is not None else Path(ROOT) / cfg["output"]["dir"] / "logs"
    for blk in cfg.get("arms", []):
        meta = dict(blk)
        arms[str(blk["name"])] = ArmLog(str(blk["name"]), logdir=base, seeds=seeds,
                                        meta=meta)
    for name, rel in COMMITTED.items():
        meta = {"family": "leaky" if name.startswith("LR") else "elu",
                "activation": "leaky_relu" if name.startswith("LR") else "elu",
                "dial": 0.1 if name.startswith("LR") else _committed_alpha(name),
                # 登録上の参照地平線は 5M（先頭 5M は gate_dose_0830 と bit 一致）。
                "total_steps": 5_000_000, "committed": True}
        arms[name] = ArmLog(name, logdir=Path(ROOT) / rel, seeds=seeds, meta=meta)
    ctx = Ctx(arms, cfg)
    ctx.notes.append(
        "committed 対照（LR_1216 / E_1216）のログは 15M あるが、5M 腕の参照として "
        "タスク 451–500 で読む（spec §3.5・§2 の参照値と同じ窓）。")
    return ctx


def _committed_alpha(name: str) -> float:
    return {"E_1216": 1.0, "E_a0p1_1216": 0.1, "E_a0p01_1216": 0.01,
            "LR_a0p01_1216": 0.01}.get(name, 1.0)


def check_config(cfg: dict) -> None:
    """config の解析定数と本モジュールの定数を突き合わせる（黙って別窓で判定しない）。"""
    A = cfg["analysis"]
    checks = [
        (int(A["task_period"]), PERIOD, "task_period"),
        (tuple(A["tail_window_tasks"]), TAIL_5M, "tail_window_tasks"),
        (tuple(A["lag_window_tasks"]), LAG_5M, "lag_window_tasks"),
        (tuple(tuple(w) for w in A["settle_windows_tasks"]), SETTLE,
         "settle_windows_tasks"),
        (float(A["C"]), C_CONST, "C"),
        (int(A["bootstrap"]["n"]), BOOT_N, "bootstrap.n"),
        (int(A["bootstrap"]["seed"]), BOOT_SEED, "bootstrap.seed"),
        (float(A["bootstrap"]["level"]), BOOT_LEVEL, "bootstrap.level"),
        (float(A["gates"]["G4_comparability_death_rate_gap"]), DEATH_GAP_MAX,
         "G4 gap"),
        (int(A["gates"]["G6_divergence"]["nan_seed_drop_max"]), NAN_SEED_DROP_MAX,
         "G6 nan_seed_drop_max"),
        (float(A["gates"]["G6_divergence"]["runaway_abs_median_zbar"]),
         RUNAWAY_ABS_ZBAR, "G6 runaway"),
        (float(A["gates"]["G1_progress"]["min_zmax_median_move_from_init"]),
         G1_MIN_ZMAX_MOVE, "G1 move"),
        (tuple(A["gates"]["G1_progress"]["w_norm_ratio_to_family_ref"]),
         G1_RATIO_BAND, "G1 ratio band"),
    ]
    for got, want, what in checks:
        if got != want:
            raise AnalysisError(f"config {what} = {got!r} but the module uses "
                                f"{want!r} — 登録した窓/定数と食い違う")
    if "0.25" not in str(A["alive_rule"]):
        raise AnalysisError("alive_rule changed in the config")
    sec = str(A.get("alive_rule_secondary", ""))
    if "zmax" not in sec or "zbar" not in sec or str(ALIVE_HALF_WIDTH) not in sec:
        raise AnalysisError(
            f"alive_rule_secondary = {sec!r} but the module uses "
            f"(zmax - zbar) > {ALIVE_HALF_WIDTH} — 登録した副次規則と食い違う")


def run_all(ctx: Ctx) -> tuple[list[dict], dict]:
    """spec §4 の全判定を回す。"""
    rows: list[dict] = []
    ks = s_ksnull(ctx)
    rows += judge_1b_mirror(ctx)
    rows += judge_1c_ensemble(ctx, ks.get("q95", float("nan")))
    rows += judge_1d_linear(ctx)
    rows += judge_1e_odd(ctx)
    rows += judge_2a_sign(ctx)
    reach_rows, reach = judge_2e_reach(ctx)
    rows += judge_2b_delta3(ctx, reach)
    rows += judge_2c_smooth(ctx)
    rows += judge_2d_steep(ctx)
    rows += reach_rows
    ret_rows, ret_labels = judge_3a_retention(ctx)
    relax_rows = judge_3b_relax(ctx, ret_labels)
    rows += ret_rows
    rows += relax_rows
    rows += judge_3_overall(ret_rows, relax_rows, ret_labels)
    rows += judge_3c_return(ctx)
    rows += judge_4a_literal()
    rows += judge_4b_locality(ctx)
    rows += judge_4c_edge(ctx)
    p5: list[dict] = []
    p5 += judge_5a_alpha(ctx)
    p5 += judge_5b_equilibrium(ctx)
    p5 += judge_5c_vfreeze(ctx)
    p5 += judge_5d_eta(ctx)
    p5 += judge_5f_scale(ctx)
    p5 += judge_5g_stationarity(ctx)
    p5 += judge_5h_order(ctx)
    p5 += judge_5i_fullbatch(ctx)
    p5 += p5_overall(p5)
    rows += p5
    rows += judge_46_divergence(ctx)
    rows += report_alive_secondary(ctx)
    _fill_dropped(rows, ctx)
    extra = {"S-KSnull": ks, "S-taut": s_taut(ctx)}
    return rows, extra


def _fill_dropped(rows: list[dict], ctx: Ctx) -> None:
    """spec §4.6:「落とした seed 数はすべての判定について `verdict.csv` に列で残す」。

    行がまだ 0 のままなら、行の `arm` から解決できる腕の落とし数を入れる
    （`A+B` のように複数腕を指す行は合計）。
    """
    for r in rows:
        if int(r.get("n_seeds_dropped", 0)) != 0:
            continue
        names = [t.strip() for t in str(r.get("arm", "")).split("+")]
        total = sum(ctx.dropped(t) for t in names if t in ctx.arms)
        r["n_seeds_dropped"] = int(total)


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
def write_verdict(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=VERDICT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in VERDICT_FIELDS})


def _cell(text) -> str:
    """markdown の表セル（`|` と改行を潰す）。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _fmt(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if not np.isfinite(v) else f"{v:.4g}"


PROP_SECTIONS = (
    ("除外規則（§3.5 副次規則の REPORT）", ("3.5 ",)),
    ("命題 1（対称性）", ("4.1-",)),
    ("命題 2（曲率／傾きの非対称）", ("4.2-",)),
    ("命題 3（初期依存）", ("4.3-", "4.3 ")),
    ("命題 4（局所性）", ("4.4-",)),
    ("命題 5（R 仮説）", ("4.5",)),
    ("§4.6 発散・除外", ("4.6",)),
)


def write_summary(path: Path, rows: list[dict], extra: dict, ctx: Ctx) -> None:
    lines = ["# edge_law_0905 判定まとめ（spec `specs/spec_edge_law_0905.md` §4）", "",
             "窓: 主 = タスク 451–500（15M 腕は 1451–1500・451–500 も併記）／"
             "lag = 351–400。除外 = `layer1_denom`（窓平均）> 0.25。"
             f"$C$ = {C_CONST}（閉形式定数）。CI = seed bootstrap "
             f"{BOOT_N} 回・percentile {int(BOOT_LEVEL*100)}%・"
             f"`np.random.default_rng({BOOT_SEED})`。", "",
             "`REPORT_ONLY` / `report` 役の行は判定に使わない（併記のみ）。", "",
             "ゲート（spec §3.6）は**すべての登録判定に前置**する。`gate_G2` の 3 窓は "
             "15M 腕でも登録どおりタスク 301–350 / 376–425 / 451–500 のまま "
             "（主窓だけが 1451–1500 に動く）。`gate_G3` は「除外なし（ALL）でも"
             "**同じラベル**が出るか」で、対称性などの部分述語ではなくその判定の"
             "最終ラベルどうしを比べる。`gate_G4`（死亡率が 10 ポイント以上違う）は"
             "腕間比較を `NOT_COMPARABLE` として止める。G6 で落とした seed の数は "
             "`n_seeds_dropped` 列に全行ぶん残る（spec §4.6）。", ""]
    for title, prefixes in PROP_SECTIONS:
        sub = [r for r in rows if any(r["judgment"].startswith(p) for p in prefixes)]
        if not sub:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 腕 | 判定 | 役割 | 統計量 | 窓 | 除外 | n | 死亡率 | "
                     "落とした seed | 点推定 |"
                     " CI | G1 | G2 | G3 | G4 | G6 | ラベル | 備考 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
                     "---|---|---|---|")
        for r in sub:
            report = " (REPORT_ONLY)" if r["role"] == "report" else ""
            lines.append(
                f"| `{r['arm']}` | {r['judgment']} | {r['role']}{report} | "
                f"{_cell(r['statistic'])} | {r['window']} | {r['exclusion']} | "
                f"{r['n']} | {_fmt(r['death_rate'])} | "
                f"{r.get('n_seeds_dropped', 0)} | {_fmt(r['point'])} | "
                f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | {r['gate_G1']} | "
                f"{r['gate_G2']} | {r['gate_G3']} | {r['gate_G4']} | "
                f"{r['gate_G6']} | **{r['label']}** | {_cell(r['note'])} |")
        lines.append("")
    ks, taut = extra.get("S-KSnull", {}), extra.get("S-taut", {})
    lines += ["## 走内検査（spec §5）", "",
              f"- **S-KSnull**: `{ks.get('arm')}` の {ks.get('n_rep')} 反復で "
              f"KS $D$ 帰無分布 q95 = **{_fmt(ks.get('q95'))}**"
              f"（中央値 {_fmt(ks.get('median'))}・最大 {_fmt(ks.get('max'))}）。"
              "1-c の閾値はこの q95 を使う。",
              "  - 註（実装上の裁定）: §4.1-c の統計量は「**seed ごと**の $D_s$ の"
              "中央値」なので、帰無分布も**同じ単位**で作る（10 seed を復元抽出して"
              "2 つのアンサンブルにし、対にした seed どうしの $D_s$ の中央値を 1 "
              f"反復とする）。プールしたアンサンブル（≈1000 対 ≈1000）版の q95 は "
              f"{_fmt(ks.get('q95_pooled'))} で、per-seed 統計量の閾値としては 2 倍"
              "ほど厳しすぎる（実測で per-seed $D$ の中央値は帰無でも 0.11 前後）。"
              "この選択がないと 1-c は厳密な bit 鏡像以外で PASS できない。",
              f"- **S-taut**: `{taut.get('arm')}` の沈下率。"
              f"新定義 $z_{{\\max}}\\le-d$: "
              + ", ".join(f"d={d}: {_fmt(v)}" for d, v in (taut.get("new") or {}).items())
              + f"／旧定義 $\\bar z\\le-d-1$: "
              + ", ".join(f"d={d}: {_fmt(v)}" for d, v in (taut.get("old") or {}).items())
              + f"。d=2,3 で新定義 < 0.05: {taut.get('new_below_0.05_at_d2_d3')}／"
              f"旧定義 > 0.7: {taut.get('old_above_0.7_at_d2_d3')}。",
              "  - 註: §5 の S-taut は「新定義 < 0.05 かつ旧定義 > 0.7」を d=2,3 の"
              "両方で書いているが、旧定義の率は d=3 では §2 の事後値（0.349）でも"
              "0.7 を下回る。ここは**記録**であって PASS ゲートではないので、率を"
              "そのまま残す（新定義が d=2,3 で判別力を持つことが要点）。", ""]
    if ctx.notes:
        lines += ["## 注記", ""] + [f"- {n}" for n in ctx.notes] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 合成データ（--selftest 用・真のラベルが分かっている作り物）
# ---------------------------------------------------------------------------
def synth_arm(name: str, *, final: dict, init: dict | None = None, tau: float = 20.0,
              seeds: int = 10, h: int = 100, tasks: int = 500,
              denom: float = 0.5, v: float = 0.3, w_norm: float = 4.0,
              w_norm_init: float = 1.4, noise: float = 0.0, rng_seed: int = 0,
              meta: dict | None = None, extra: dict | None = None,
              payload: dict | None = None, drift: dict | None = None) -> ArmLog:
    """タスク終端記録だけの合成腕（`_window_indices` はタスク終端しか拾わない）。

    値は ``final + (init − final)·exp(−task/τ) + drift·task`` で作る（τ が小さいと
    G2 の 3 窓ではもう定着している）。
    """
    rng = np.random.default_rng(int(rng_seed))
    steps = np.arange(0, tasks + 1, dtype=np.int64) * PERIOD
    t = steps / PERIOD
    data: dict[int, dict] = {}
    keys = set(final) | set(init or {}) | {"layer1_denom", "layer1_v_unit",
                                           "layer1_w_norm", "layer1_zbar",
                                           "layer1_zmax"}
    for si in range(seeds):
        d = {"step": steps, "seed": np.int64(si),
             "run_id": np.array(f"{name}_seed{si}"), "arm": np.array(name),
             "unfit": np.full(len(steps), 0.01)}
        for key in sorted(keys):
            fin = final.get(key, {"layer1_denom": denom, "layer1_v_unit": v,
                                  "layer1_w_norm": w_norm}.get(key, 0.0))
            ini = (init or {}).get(key, {"layer1_denom": denom,
                                         "layer1_v_unit": v,
                                         "layer1_w_norm": w_norm_init}.get(key, fin))
            fin = np.broadcast_to(np.asarray(fin, dtype=np.float64), (seeds, h))[si]
            ini = np.broadcast_to(np.asarray(ini, dtype=np.float64), (seeds, h))[si]
            vals = fin[None, :] + (ini - fin)[None, :] * np.exp(-t[:, None] / tau)
            if drift and key in drift:
                vals = vals + np.asarray(drift[key], dtype=np.float64) * t[:, None]
            if noise:
                vals = vals + rng.normal(0.0, noise, size=vals.shape)
            d[key] = vals.astype(np.float32)
        for key, arr in (extra or {}).items():
            d[key] = arr[si] if isinstance(arr, dict) else arr
        for key, val in (payload or {}).items():
            d[key] = np.array(val)
        data[si] = d
    return ArmLog(name, data=data, meta=meta or {})


def _uniform_units(base: float, spread: float, seeds: int = 10, h: int = 100,
                   rng_seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return base + rng.uniform(-spread, spread, size=(seeds, h))


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest_synthetic() -> list[tuple[str, bool, str]]:
    """真のラベルが分かっている作り物で全判定を通す。"""
    res: list[tuple[str, bool, str]] = []

    def check(name, got, want):
        res.append((name, got == want, f"got {got!r}, want {want!r}"))

    # --- 1-b S-mirror: 完全な鏡像 → MIRROR_EXACT ---------------------------
    # seed 数は登録どおり 10（10 未満は MIRROR_NOT_RUN・spec §4.1-b）。
    rng = np.random.default_rng(7)
    n_rec, h, S = 40, 8, MIRROR_SEEDS_REQUIRED
    base = {}
    for si in range(S):
        z = rng.normal(size=(n_rec, h)).astype(np.float32)
        p = rng.integers(0, 33, size=(n_rec, h)).astype(np.float32) / 32.0
        base[si] = {"step": np.arange(n_rec, dtype=np.int64) * PERIOD,
                    "layer1_zbar": z, "layer1_dzbar": z * 0.5,
                    "layer1_zmean": z, "layer1_v_unit": z * 0.1,
                    "layer1_M": z * 2, "layer1_B": z * 3,
                    "layer1_w_norm": np.abs(z) + 1, "layer1_denom": np.abs(z) + 1,
                    "layer1_mob": np.abs(z), "layer1_absmob": np.abs(z),
                    "layer1_p_hat": p}
    mirrored = {si: {k: (np.negative(v) if k in MIRROR_NEG_COLUMNS else
                         (1.0 - v).astype(np.float32) if k == "layer1_p_hat" else v)
                     for k, v in d.items()} for si, d in base.items()}
    ctx = Ctx({"LR_1216": ArmLog("LR_1216", data=base, meta={"family": "leaky"}),
               "FLn_1216": ArmLog("FLn_1216", data=mirrored,
                                  meta={"family": "flip"})})
    check("1-b mirror exact", judge_1b_mirror(ctx)[0]["label"], "MIRROR_EXACT")
    broken = {si: {k: v.copy() for k, v in d.items()} for si, d in mirrored.items()}
    broken[1]["layer1_zbar"][3, 4] *= -1.0                 # 符号 1 個だけ反転
    ctx2 = Ctx({"LR_1216": ArmLog("LR_1216", data=base, meta={"family": "leaky"}),
                "FLn_1216": ArmLog("FLn_1216", data=broken, meta={"family": "flip"})})
    check("1-b mirror broken (mutant must fail)",
          judge_1b_mirror(ctx2)[0]["label"], "MIRROR_BROKEN")
    # 空虚な PASS を塞ぐ 2 経路（列が足りない・seed が足りない）
    thin = {si: {k: v for k, v in d.items()
                 if k in ("step", "layer1_zbar", "layer1_p_hat")}
            for si, d in mirrored.items()}
    check("1-b missing parity columns -> NOT_RUN",
          judge_1b_mirror(Ctx({
              "LR_1216": ArmLog("LR_1216", data={si: {k: v for k, v in d.items()
                                                      if k in ("step", "layer1_zbar",
                                                               "layer1_p_hat")}
                                                 for si, d in base.items()}),
              "FLn_1216": ArmLog("FLn_1216", data=thin)}))[0]["label"],
          "MIRROR_NOT_RUN")
    few = {si: d for si, d in mirrored.items() if si < 2}
    check("1-b fewer than ten seeds -> NOT_RUN",
          judge_1b_mirror(Ctx({"LR_1216": ArmLog("LR_1216", data=base),
                               "FLn_1216": ArmLog("FLn_1216",
                                                  data=few)}))[0]["label"],
          "MIRROR_NOT_RUN")
    # 零の符号: 鏡像を**独立の経路**（実際の引き算）で作り、両腕でちょうど +0.0 に
    # なる要素を含む列で `np.negative` の期待値が -0.0 になっても通ること。
    # 生成に `np.negative` を使わないのが要点（検査対象の演算で fixture を作らない）。
    zero_base, zero_flip = {}, {}
    for si in range(MIRROR_SEEDS_REQUIRED):
        raw = rng.normal(size=(n_rec + 1, h)).astype(np.float32)
        raw[3, 2] = raw[2, 2]                      # 差がちょうど +0.0 になる場所
        raw[7, 5] = raw[6, 5]
        dz = np.diff(raw, axis=0)                  # a - a == +0.0（両腕とも）
        dz_flip = np.diff(-raw, axis=0)            # 鏡像側も差で作る（+0.0 のまま）
        step = np.arange(n_rec, dtype=np.int64) * PERIOD
        p = rng.integers(0, 33, size=(n_rec, h)).astype(np.float32) / 32.0
        common = dict(step=step, layer1_p_hat=p)
        zero_base[si] = dict(common, layer1_dzbar=dz,
                             **{k: raw[1:] for k in
                                ("layer1_zbar", "layer1_zmean", "layer1_v_unit",
                                 "layer1_M", "layer1_B")},
                             **{k: np.abs(raw[1:]) + 1 for k in
                                ("layer1_w_norm", "layer1_denom", "layer1_mob",
                                 "layer1_absmob")})
        zero_flip[si] = dict(step=step, layer1_p_hat=(1.0 - p).astype(np.float32),
                             layer1_dzbar=dz_flip,
                             **{k: (-raw)[1:] for k in
                                ("layer1_zbar", "layer1_zmean", "layer1_v_unit",
                                 "layer1_M", "layer1_B")},
                             **{k: np.abs(raw[1:]) + 1 for k in
                                ("layer1_w_norm", "layer1_denom", "layer1_mob",
                                 "layer1_absmob")})
    n_zero = sum(int(np.count_nonzero((zero_base[si]["layer1_dzbar"] == 0.0)
                                      & (zero_flip[si]["layer1_dzbar"] == 0.0)))
                 for si in zero_base)
    res.append(("1-b fixture really contains both-zero entries", n_zero > 0,
                f"{n_zero} entries are exactly 0.0 in both arms"))
    ctx_zero = Ctx({"LR_1216": ArmLog("LR_1216", data=zero_base),
                    "FLn_1216": ArmLog("FLn_1216", data=zero_flip)})
    check("1-b mirror with +0.0 in both arms (zero-sign exception)",
          judge_1b_mirror(ctx_zero)[0]["label"], "MIRROR_EXACT")
    # 同じ fixture で 1 要素だけ本当に符号を壊すと落ちること
    zero_bad = {si: {k: (v.copy() if hasattr(v, "copy") else v)
                     for k, v in d.items()} for si, d in zero_flip.items()}
    zero_bad[0]["layer1_zbar"][5, 1] = -zero_bad[0]["layer1_zbar"][5, 1]
    check("1-b zero fixture mutant must fail",
          judge_1b_mirror(Ctx({"LR_1216": ArmLog("LR_1216", data=zero_base),
                               "FLn_1216": ArmLog("FLn_1216",
                                                  data=zero_bad)}))[0]["label"],
          "MIRROR_BROKEN")

    # --- 2-b Δ3: d30 と同じ → AT_INIT / −1.5 ずらし → NONLOCAL -------------
    z30 = _uniform_units(0.1, 0.6)
    shelf_meta = {"family": "shelf", "activation": "shelf_leaky_d3", "dial": 0.1}
    d30_meta = {"family": "shelf", "activation": "shelf_leaky_d30", "dial": 0.1}
    common = dict(init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                  w_norm=4.0, tau=8.0)
    for tag, shift, want in (("at-init", 0.0, "CURVATURE_AT_INIT"),
                             ("nonlocal", -1.5, "CURVATURE_NONLOCAL")):
        arms = {
            "SH_d3_1216": synth_arm("SH_d3_1216", meta=shelf_meta,
                                    final={"layer1_zmax": z30 + shift,
                                           "layer1_zbar": z30 + shift - 3.8},
                                    **common),
            "SH_d30_1216": synth_arm("SH_d30_1216", meta=d30_meta,
                                     final={"layer1_zmax": z30,
                                            "layer1_zbar": z30 - 3.8},
                                     **common),
        }
        arms["SH_d3_1216"].data[0]["layer1_zmin"] = np.full(
            (501, 100), -9.0, dtype=np.float32)
        got = [r for r in judge_2b_delta3(Ctx(arms), {"SH_d3_1216": 0.6})
               if r["judgment"] == "4.2-b Delta_3"][0]["label"]
        check(f"2-b Delta_3 {tag}", got, want)

    # --- 3-a 保持率: rho=0 → MEAN_INDEPENDENT / rho=0.6 → MEAN_DEPENDENT ----
    ref_tail = _uniform_units(0.11, 0.5, rng_seed=3)
    for tag, rho, want in (("rho=0", 0.0, "MEAN_INDEPENDENT"),
                           ("rho=0.6", 0.6, "MEAN_DEPENDENT")):
        pert = 5.0
        arms = {
            "LRnull_1216": synth_arm(
                "LRnull_1216", meta={"family": "leaky", "activation": "leaky_relu",
                                     "dial": 0.1},
                final={"layer1_zmax": ref_tail, "layer1_zbar": ref_tail - 3.7},
                init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0),
            "LRbp5_1216": synth_arm(
                "LRbp5_1216", meta={"family": "leaky", "activation": "leaky_relu",
                                    "dial": 0.1},
                final={"layer1_zmax": ref_tail + rho * pert,
                       "layer1_zbar": ref_tail + rho * pert - 3.7},
                init={"layer1_zmax": 0.753 + pert, "layer1_zbar": -0.6 + pert},
                tau=5.0),
        }
        rows, labels = judge_3a_retention(Ctx(arms))
        check(f"3-a retention {tag}", labels["LRbp5_1216"], want)

    # --- G1 進捗ゲート: 凍結した腕 → NOT_DETERMINED -------------------------
    frozen = synth_arm("LIN_1216", meta={"family": "linear",
                                         "activation": "leaky_relu", "dial": 1.0},
                       final={"layer1_zmax": 0.753, "layer1_zbar": 0.0},
                       init={"layer1_zmax": 0.753, "layer1_zbar": 0.0},
                       w_norm=1.4, w_norm_init=1.4, tau=1.0)
    ctx_f = Ctx({"LIN_1216": frozen})
    r = judge_1d_linear(ctx_f)[0]
    check("G1 frozen -> 1-d NOT_DETERMINED", r["label"], "NOT_DETERMINED")
    res.append(("G1 frozen gate says FROZEN", r["gate_G1"] == "FROZEN",
                f"gate_G1={r['gate_G1']}"))

    # --- 1-d 線形: 対称・釘付け → LINEAR_PINNED ----------------------------
    rngl = np.random.default_rng(11)
    sym = rngl.normal(0.0, 0.05, size=(10, 100))
    sym = sym - np.median(sym)
    lin = synth_arm("LIN_1216", meta={"family": "linear",
                                      "activation": "leaky_relu", "dial": 1.0},
                    final={"layer1_zbar": sym, "layer1_zmax": sym + 0.2},
                    init={"layer1_zbar": -0.6, "layer1_zmax": 0.753},
                    w_norm=2.5, w_norm_init=1.4, tau=5.0)
    check("1-d linear pinned", judge_1d_linear(Ctx({"LIN_1216": lin}))[0]["label"],
          "LINEAR_PINNED")

    # --- 2-c softplus: 下へ → SMOOTH_DOWN ----------------------------------
    sp = synth_arm("SP_1216", meta={"family": "softplus",
                                    "activation": "softplus_b", "dial": 1.0},
                   final={"layer1_zbar": _uniform_units(-2.6, 0.4, rng_seed=5),
                          "layer1_zmax": 0.1},
                   init={"layer1_zbar": -0.6, "layer1_zmax": 0.753}, tau=6.0)
    check("2-c smooth down", judge_2c_smooth(Ctx({"SP_1216": sp}))[0]["label"],
          "SMOOTH_DOWN")

    # --- 2-d 曲率反転: 上へ → ASYMMETRY_SIGN_OK ----------------------------
    arms = {}
    for d in (1, 2):
        arms[f"SH_d{d}_1216"] = synth_arm(
            f"SH_d{d}_1216", meta={"family": "shelf",
                                   "activation": f"shelf_leaky_d{d}", "dial": 0.1},
            final={"layer1_zmax": _uniform_units(-0.5, 0.4, rng_seed=10 + d),
                   "layer1_zbar": -3.5},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0)
        arms[f"ST_d{d}_1216"] = synth_arm(
            f"ST_d{d}_1216", meta={"family": "steep",
                                   "activation": f"steep_shelf_d{d}", "dial": 1.0},
            # G1 の「初期 zmax 0.753 から 0.3 以上動く」を満たすため 1.4 に置く
            final={"layer1_zmax": _uniform_units(1.4, 0.4, rng_seed=10 + d),
                   "layer1_zbar": -3.5},
            init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0)
    got = [r for r in judge_2d_steep(Ctx(arms))
           if r["judgment"] == "4.2-d overall"][0]["label"]
    check("2-d asymmetry sign", got, "ASYMMETRY_SIGN_OK")

    # --- 5-c v 凍結: 単調に沈む → WELL_FROM_READOUT -------------------------
    enull = synth_arm("Enull_1216", meta={"family": "elu", "activation": "elu",
                                          "dial": 1.0},
                      final={"layer1_zmax": _uniform_units(0.1, 0.4, rng_seed=21),
                             "layer1_zbar": -6.3},
                      init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                      w_norm=6.6, tau=6.0)
    evf = synth_arm("Evf1_1216", meta={"family": "elu", "activation": "elu",
                                       "dial": 1.0},
                    final={"layer1_zmax": _uniform_units(0.1, 0.4, rng_seed=21),
                           "layer1_zbar": -6.3},
                    init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                    w_norm=6.6, tau=6.0,
                    drift={"layer1_zmax": -0.012}, payload={"freeze_v": True})
    got = [r for r in judge_5c_vfreeze(Ctx({"Enull_1216": enull,
                                            "Evf1_1216": evf}))
           if r["judgment"] == "4.5-c v-freeze"][0]["label"]
    check("5-c well from readout", got, "WELL_FROM_READOUT")

    # --- 5-b 数値平衡: ELU の閉形式 z* を合成データで復元 -------------------
    got = _selftest_equilibrium()
    res.append(("5-b equilibrium solver reproduces the closed form (ELU)",
                got[0], got[1]))

    # --- 5-g 停留残差: 作り物のモーメント列 --------------------------------
    got = _selftest_stationarity()
    res.append(("5-g stationarity on synthetic moments", got[0], got[1]))

    # --- 1-c アンサンブル鏡像: FL = −LR → ENSEMBLE_SYMMETRIC ---------------
    zl = _uniform_units(-3.6, 1.5, rng_seed=31)
    lr_arm = synth_arm("LR_1216", meta={"family": "leaky",
                                        "activation": "leaky_relu", "dial": 0.1},
                       final={"layer1_zbar": zl, "layer1_zmax": zl + 3.7},
                       init={"layer1_zbar": -0.6, "layer1_zmax": 0.753}, tau=5.0)
    fl_arm = synth_arm("FL_1216", meta={"family": "flip",
                                        "activation": "flip_leaky", "dial": 0.1},
                       final={"layer1_zbar": -zl, "layer1_zmax": -zl + 3.7},
                       init={"layer1_zbar": 0.6, "layer1_zmax": 0.753}, tau=5.0)
    ctx_1c = Ctx({"LR_1216": lr_arm, "FL_1216": fl_arm})
    ks = s_ksnull(ctx_1c, "LR_1216")
    check("1-c ensemble symmetric",
          judge_1c_ensemble(ctx_1c, ks["q95"])[0]["label"], "ENSEMBLE_SYMMETRIC")
    check("2-a sign (FL above +2, LR below -2)",
          [r["label"] for r in judge_2a_sign(ctx_1c)], ["SIGN_OK", "SIGN_OK"])

    # --- 1-e 奇な非線形: 対称かつ釘付け / 逃走 -----------------------------
    rng_t = np.random.default_rng(41)
    sym_t = rng_t.normal(0.0, 0.03, size=(10, 100))
    sym_t = sym_t - np.median(sym_t)
    th = synth_arm("TH_1216", meta={"family": "tanh", "activation": "tanh_b",
                                    "dial": 1.0},
                   final={"layer1_zbar": sym_t, "layer1_zmax": sym_t + 1.2},
                   init={"layer1_zbar": -0.6, "layer1_zmax": 0.753},
                   w_norm=2.5, tau=5.0)
    check("1-e odd symmetric pinned",
          judge_1e_odd(Ctx({"TH_1216": th}))[0]["label"], "ODD_SYMMETRIC_PINNED")

    # --- 2-e 到達率（zmin 列） ---------------------------------------------
    n_rec = 501
    zmin_arr = np.full((n_rec, 100), -0.2, dtype=np.float32)
    zmin_arr[200:, :60] = -9.0                         # 60% が折れ目 −3 に到達
    sh3 = synth_arm("SH_d3_1216", meta={"family": "shelf",
                                        "activation": "shelf_leaky_d3", "dial": 0.1},
                    final={"layer1_zmax": -0.4, "layer1_zbar": -3.9},
                    init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=6.0,
                    extra={"layer1_zmin": zmin_arr})
    rr, rates = judge_2e_reach(Ctx({"SH_d3_1216": sh3}))
    res.append(("2-e reach rate = 0.60", abs(rates["SH_d3_1216"] - 0.60) < 1e-9,
                f"reach rate = {rates['SH_d3_1216']}"))

    # --- 4-b 局所性半径: d* = 3 → LOCALITY_MONOTONE ------------------------
    lin_meta = {"family": "linear", "activation": "leaky_relu", "dial": 1.0}
    arms = {"LIN_1216": synth_arm("LIN_1216", meta=lin_meta,
                                  final={"layer1_zmax": 0.11, "layer1_zbar": 0.03},
                                  init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                                  tau=5.0)}
    ladder = {0.5: (-2.0, -2.1), 1.0: (-1.5, -1.6), 2.0: (-0.8, -0.9),
              3.0: (0.2, 0.1), 30.0: (0.11, 0.03)}
    for d, (zm, zb) in ladder.items():
        nm = {0.5: "SH_d0p5_1216", 1.0: "SH_d1_1216", 2.0: "SH_d2_1216",
              3.0: "SH_d3_1216", 30.0: "SH_d30_1216"}[d]
        act = f"shelf_leaky_d{'0p5' if d == 0.5 else int(d)}"
        arms[nm] = synth_arm(nm, meta={"family": "shelf", "activation": act,
                                       "dial": 0.1},
                             final={"layer1_zmax": zm, "layer1_zbar": zb},
                             init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                             tau=6.0)
    loc = [r for r in judge_4b_locality(Ctx(arms))
           if r["judgment"] == "4.4-b locality radius"
           and r["role"] == "confirmatory"][0]
    check("4-b locality monotone", loc["label"], "LOCALITY_MONOTONE")
    res.append(("4-b d* = 3", loc["point"] == 3.0, f"d* = {loc['point']}"))

    # --- 4-c 上端は折れ目に釘付け / 上に離れる -----------------------------
    for tag, offset, z_init, want in (("at kink", 0.0, 0.753, "EDGE_AT_KINK"),
                                      ("detached up", 2.5, 2.0,
                                       "EDGE_DETACHED_UP")):
        arms = {}
        for d in (2.0, 3.0):
            nm = {2.0: "SH_d2_1216", 3.0: "SH_d3_1216"}[d]
            arms[nm] = synth_arm(
                nm, meta={"family": "shelf", "activation": f"shelf_leaky_d{int(d)}",
                          "dial": 0.1},
                final={"layer1_zmax": _uniform_units(-d + offset, 0.2,
                                                    rng_seed=int(50 + d)),
                       "layer1_zbar": -d + offset - 3.5},
                init={"layer1_zmax": z_init, "layer1_zbar": -0.6}, tau=6.0)
        got = [r for r in judge_4c_edge(Ctx(arms))
               if r["judgment"] == "4.4-c edge overall"][0]["label"]
        check(f"4-c edge {tag}", got, want)

    # --- 5-a α コントラスト: B が α に依らない → CONSISTENT ---------------
    rng_a = np.random.default_rng(61)
    x = rng_a.uniform(1.0, 3.0, size=(10, 100))          # ln(1+C v^2)
    v_a = np.sqrt(np.expm1(x) / C_CONST)
    arms = {}
    for alpha, nm in ((0.5, "E_a0p5_1216"), (1.0, "Enull_1216"),
                      (2.0, "E_a2_1216"), (4.0, "E_a4_1216")):
        arms[nm] = synth_arm(nm, meta={"family": "elu", "activation": "elu",
                                       "dial": alpha},
                             final={"layer1_zmax": 0.5 - 1.0 * x,
                                    "layer1_zbar": 0.5 - 1.0 * x - 6.0},
                             init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                             v=v_a, w_norm=6.6, tau=5.0)
    got = [r for r in judge_5a_alpha(Ctx(arms))
           if r["judgment"] == "4.5-a alpha contrast"][0]["label"]
    check("5-a alpha contrast consistent", got, "ALPHA_CONTRAST_CONSISTENT")

    # --- 5-d 揺らぎ: sd ∝ sqrt(eta) → FLUCT_SQRT_ETA ----------------------
    arms = {}
    for eta, nm in ((0.005, "Elr0p005_1216"), (0.01, "Enull_1216"),
                    (0.02, "Elr0p02_1216")):
        arms[nm] = synth_arm(nm, meta={"family": "elu", "activation": "elu",
                                       "dial": 1.0, "total_steps": 5_000_000},
                             final={"layer1_zmax": -1.0, "layer1_zbar": -6.5},
                             init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                             w_norm=6.6, tau=5.0,
                             noise=0.1 * math.sqrt(eta / 0.01),
                             payload={"lr_used": eta})
    got = [r for r in judge_5d_eta(Ctx(arms))
           if r["judgment"] == "4.5-d fluctuation scaling"][0]["label"]
    check("5-d fluctuation sqrt(eta)", got, "FLUCT_SQRT_ETA")

    # --- 5-f スケール: zmax 不変・z̄ だけずれる → LAW_INVARIANT ------------
    zbase = _uniform_units(0.11, 0.5, rng_seed=71)
    arms = {"LRnull_1216": synth_arm(
        "LRnull_1216", meta={"family": "leaky", "activation": "leaky_relu",
                             "dial": 0.1},
        final={"layer1_zmax": zbase, "layer1_zbar": zbase - 3.7},
        init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0)}
    for nm, shift in (("LRs0p5_1216", +1.85), ("LRs2_1216", -3.7)):
        arms[nm] = synth_arm(nm, meta={"family": "leaky",
                                       "activation": "leaky_relu", "dial": 0.1},
                             final={"layer1_zmax": zbase,
                                    "layer1_zbar": zbase - 3.7 + shift},
                             init={"layer1_zmax": 0.753, "layer1_zbar": -0.6},
                             tau=5.0)
    got = [r for r in judge_5f_scale(Ctx(arms))
           if r["judgment"] == "4.5-f overall"][0]["label"]
    check("5-f scale law invariant", got, "P5_SCALE_LAW_INVARIANT")

    # --- G6: 3 seed が NaN → NOT_RUN ---------------------------------------
    bad = synth_arm("E_a4_1216", meta={"family": "elu", "activation": "elu",
                                       "dial": 4.0},
                    final={"layer1_zmax": -1.0, "layer1_zbar": -6.5},
                    init={"layer1_zmax": 0.753, "layer1_zbar": -0.6}, tau=5.0)
    for si in (0, 1, 2):
        bad.data[si]["layer1_zbar"][-3:, :] = np.nan
    ctx_bad = Ctx({"E_a4_1216": bad})
    check("4.6 divergence NOT_RUN (3 NaN seeds)",
          judge_46_divergence(ctx_bad)[0]["label"], "NOT_RUN")

    # --- 3-b 緩和フィット: τ と z_inf を復元 -------------------------------
    t = np.arange(100, 501, dtype=np.float64)
    z = -1.0 + (4.0 - (-1.0)) * np.exp(-(t - 100) / 60.0)
    zi, z0, tau = relax_fit(t, z)
    res.append(("3-b relax fit recovers z_inf and tau",
                abs(zi + 1.0) < 0.05 and abs(tau - 60.0) < 6.0,
                f"z_inf={zi:.3f} (want -1.0), tau={tau:.1f} (want 60)"))
    return res


def _selftest_equilibrium() -> tuple[bool, str]:
    """完全沈水の ELU では平衡が z*_max = z0 − ln(1+κ) になる（spec §1）。

    支持を極端に狭くすると $\\mathbb E$ が 1 点評価に近づき、平衡は
    $\\bar z^\\ast = -\\ln(1+\\kappa)$（α に依らない）。数値解がこれを返すか。
    """
    h, S = 6, 2
    kappa = np.array([[0.5, 1.0, 2.0, 5.0, 10.0, 20.0]] * S)
    v = np.sqrt(kappa / C_CONST)
    w_free = np.zeros((S, h, 5), dtype=np.float32)
    w_free[..., 0] = 0.02                                   # ほぼ 1 点支持
    steps = np.arange(0, 501, dtype=np.int64) * PERIOD
    data = {}
    for si in range(S):
        n = len(steps)
        data[si] = {"step": steps,
                    "layer1_zmax": np.zeros((n, h), np.float32),
                    "layer1_zbar": np.zeros((n, h), np.float32),
                    "layer1_denom": np.ones((n, h), np.float32),
                    "layer1_v_unit": np.broadcast_to(
                        v[si].astype(np.float32), (n, h)).copy(),
                    "layer1_w_norm": np.full((n, h), 4.0, np.float32),
                    "layer1_w_free": np.broadcast_to(
                        w_free[si], (n, h, 5)).copy(),
                    "layer1_w_free_step": steps}
    arm = ArmLog("E_syn", data=data, meta={"family": "elu", "activation": "elu",
                                           "dial": 1.0})
    ctx = Ctx({"E_syn": arm})
    eq = equilibrium_zmax(ctx, "E_syn")
    want = -np.log1p(kappa[0])
    got = eq["zbar_star"][0]
    err = float(np.max(np.abs(got - want)))
    return err < 0.02, f"max |zbar* - (-ln(1+kappa))| = {err:.4f} (kappa {kappa[0]})"


def _selftest_stationarity() -> tuple[bool, str]:
    """モーメント列から G_i を作り、末尾で |G| が 1/3 未満に落ちる作り物。"""
    h, S, n = 5, 4, 501
    steps = np.arange(0, n, dtype=np.int64) * PERIOD
    rng = np.random.default_rng(3)
    data = {}
    for si in range(S):
        task = steps / PERIOD
        small = np.where(task[:, None] > 425, 0.001, 0.02) * np.ones((n, h))
        m_pd = small * (1.0 + 0.01 * rng.normal(size=(n, h)))
        m_dd = -small * (1.0 + 0.01 * rng.normal(size=(n, h)))
        dz = -0.01 * (2 * m_pd + 2 * 0.0 * m_dd)
        # G1 進捗ゲートを通すため zmax は初期 0.753 → 末尾 −1.0 に動かす
        zmax = -1.0 + 1.753 * np.exp(-task / 8.0)
        data[si] = {"step": steps,
                    "layer1_zmax": np.broadcast_to(
                        zmax[:, None], (n, h)).astype(np.float32).copy(),
                    "layer1_zbar": np.zeros((n, h), np.float32),
                    "layer1_dzbar": dz.astype(np.float32),
                    "layer1_denom": np.ones((n, h), np.float32),
                    "layer1_v_unit": np.zeros((n, h), np.float32),
                    "layer1_w_norm": np.full((n, h), 4.0, np.float32),
                    "layer1_m_phidphi": m_pd.astype(np.float32),
                    "layer1_m_dphiddphi": m_dd.astype(np.float32),
                    "layer1_moment_step": steps,
                    "lr_used": np.float64(0.01)}
    arm = ArmLog("LRnull_1216", data=data,
                 meta={"family": "leaky", "activation": "leaky_relu", "dial": 0.1})
    ctx = Ctx({"LRnull_1216": arm})
    rows = judge_5g_stationarity(ctx)
    per = [r for r in rows if r["arm"] == "LRnull_1216"]
    ok = bool(per) and per[0]["label"] == "STATIONARY"
    return ok, f"label={per[0]['label'] if per else 'missing'}"


def selftest_real(verbose: bool = True) -> dict:
    """committed の `LR_1216` / `E_1216` を loader に通し、spec §2 の参照値を再現する。"""
    cfg = load_config(str(CONFIG))
    ctx = build_ctx(cfg)
    out: dict = {}
    lr, e = ctx.get("LR_1216"), ctx.get("E_1216")
    if lr is None or e is None:
        return {"available": False}
    out["available"] = True
    win = TAIL_5M
    zmax = lr.unit_window("layer1_zmax", win)
    zbar = lr.unit_window("layer1_zbar", win)
    alive = lr.alive(win)
    out["LR_1216 tail zmax median"] = float(np.median(zmax[alive]))
    out["LR_1216 tail zbar median"] = float(np.median(zbar[alive]))
    out["LR_1216 death rate"] = lr.death_rate(win)
    out["LR_1216 tail half-width median"] = float(np.median((zmax - zbar)[alive]))
    out["LR_1216 tail |w| median"] = float(
        np.median(lr.unit_window("layer1_w_norm", win)[alive]))
    out["LR_1216 init zmax median"] = float(np.median(lr.unit_at_step("layer1_zmax")))
    out["E_1216 death rate"] = e.death_rate(win)
    out["E_1216 tail |w| median"] = float(
        np.median(e.unit_window("layer1_w_norm", win)[e.alive(win)]))
    out["E_1216 tail zmax median (ALL)"] = float(
        np.median(e.unit_window("layer1_zmax", win)))
    out["E_1216 tail zbar median (ALL)"] = float(
        np.median(e.unit_window("layer1_zbar", win)))
    n_seeds = len(e.seeds)
    draws = boot_draws(n_seeds)
    for tag, lag, sub in (("simultaneous v, ALIVE", False, False),
                          ("lag v, ALIVE", True, False),
                          ("simultaneous v, fully submerged", False, True),
                          ("lag v, fully submerged", True, True)):
        point, icept, n = b_slope(ctx, "E_1216", lag=lag, submerged=sub)
        _, lo, hi = boot_ci(
            lambda idx, lag=lag, sub=sub: b_slope(ctx, "E_1216", lag=lag,
                                                  submerged=sub, seed_idx=idx)[0],
            n_seeds, draws=draws)
        out[f"E_1216 B ({tag})"] = (point, lo, hi, n, icept)
    out["S-taut (LR_1216)"] = s_taut(ctx, "LR_1216")
    out["S-KSnull (LR_1216)"] = s_ksnull(ctx, "LR_1216")
    # C が閉形式どおりか（S-C の解析側の片割れ）
    with np.load(Path(ROOT) / "results/p3_extend_0902/logs/LR_1216_seed0.npz",
                 allow_pickle=True) as z:
        c_series = (z["layer1_mu_norm"] ** 2 + 20.0 * z["layer1_sigma_rms"] ** 2
                    + 1.0)
        out["C from logs (min, max)"] = (float(c_series.min()), float(c_series.max()))
        out["dose_relative_error max"] = float(np.abs(z["dose_relative_error"]).max())
    ctx.close()
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


REAL_REFERENCE = {
    "LR_1216 tail zmax median": (0.112, 0.002),
    "LR_1216 tail zbar median": (-3.598, 0.003),
    "LR_1216 death rate": (0.0, 1e-12),
    "E_1216 death rate": (0.174, 0.001),
}
REAL_B_REFERENCE = {
    "E_1216 B (simultaneous v, ALIVE)": (1.637, 1.506, 1.796),
    "E_1216 B (lag v, fully submerged)": (0.794, 0.652, 0.988),
}


def check_real_reference(out: dict, tol: float = 0.01) -> list[tuple[str, bool, str]]:
    res = []
    for key, (want, atol) in REAL_REFERENCE.items():
        got = out.get(key, float("nan"))
        res.append((key, abs(got - want) <= atol, f"got {got:.6g}, want {want} ± {atol}"))
    for key, (p, lo, hi) in REAL_B_REFERENCE.items():
        got = out.get(key, (float("nan"),) * 5)
        ok = (abs(got[0] - p) <= tol and abs(got[1] - lo) <= tol
              and abs(got[2] - hi) <= tol)
        res.append((key, ok, f"got {got[0]:.4f} [{got[1]:.4f}, {got[2]:.4f}], "
                             f"want {p} [{lo}, {hi}]"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None,
                    help="verdict.csv / summary.md の出力先")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--logdir", default=None, help="腕ログの場所（既定は config）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    check_config(cfg)

    if args.selftest:
        print("== synthetic selftests (true label known) ==")
        rows = selftest_synthetic()
        for name, ok, detail in rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        print("== real committed logs (spec §2 reference values) ==")
        out = selftest_real()
        checks = check_real_reference(out)
        for name, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        bad = [r for r in rows + checks if not r[1]]
        print(f"== {len(rows) + len(checks) - len(bad)} passed, {len(bad)} failed ==")
        if bad:
            sys.exit(1)
        if args.outdir is None:
            return

    outdir = Path(args.outdir) if args.outdir else (
        Path(ROOT) / cfg["output"]["dir"])
    if not outdir.is_absolute():
        outdir = Path(ROOT) / outdir
    ctx = build_ctx(cfg, Path(args.logdir) if args.logdir else None)
    rows, extra = run_all(ctx)
    write_verdict(outdir / "verdict.csv", rows)
    write_summary(outdir / "summary.md", rows, extra, ctx)
    ctx.close()
    print(json.dumps({"outdir": str(outdir), "rows": len(rows),
                      "labels": {r["judgment"]: r["label"] for r in rows
                                 if r["role"] == "confirmatory"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
