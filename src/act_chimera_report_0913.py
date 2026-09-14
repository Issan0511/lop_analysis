"""act_chimera_report_0913: 判定器（spec_act_chimera_0913 §3・§4・§8）。

3 環境（箱 B = Permuted MNIST・RL = Random-Label MNIST・SCR = condA）の保存配列から、
§4 のラベルを **純関数** で組む。1 回目（段 1: D-par・D0・D1–D3・仮説・D5a・RL の床・D8・副）と
2 回目（D4・D5b・D6・D7）は同じ入口 ``main`` から ``--pass 1|2|both`` で選ぶ。

設計（§4.0）
- 対比 c の状態は seed ごとの (c, m) から ``contrast_state`` で DIR+/DIR−/EQ/UNRES の 4 つに
  **排他** に決め、方向ラベルは DIR± から、等価ラベルは EQ から作る。
- m(c) = √(SE(c)² + Σw²·σ²_arm)。SE は §3.2 で固定した推定量 ``se_window``（ddof = 1・ρ₁ は隣接対の
  Pearson 相関を 0 で切る・n_eff = max(1, n(1−ρ₁)/(1+ρ₁))）。sd = 0 や r = NaN は ``DegenerateSeries``
  を投げる（m を 0 や NaN にしない・S16 (vi)）。判定器はそれを対比ごとに捕まえて
  ``NOT_DETERMINED_DEGENERATE_SERIES`` にする。
- co-primary は endpoint ごとにラベルを出してから ``merge_co_primary`` で状態ごとに合わせる
  （``_L_ONLY`` / ``_A_ONLY`` / ``_PT_ONLY`` / ``_LOGIT_ONLY`` / ``CO_PRIMARY_CONFLICT(...)``）。下流では
  それらをすべて UNRESOLVED として扱う（``decided``）。
- 用量ゲート（D0）は EQ を ``NOT_TESTABLE_*`` に置き換え、DIR には ``(WEAK_DOSE)`` を付ける（``gate_eq``）。
- SCR は 10 系列の符号 9/10（``sign_state``）。等価ラベルは無い。

読む物（§8）: ``results/act_chimera_0913/{pmnist,rlmnist,scr}``。梯子（D6）の LR001/LR03/LIN は
committed の ``results/gate_shape_0911/*_rows.csv`` を同じ seed で読む。RL の床の F_seed は
``rl_labels`` 系列から厳密に計算する（0906 の R/LR の行で S16 (iii) が検査する）。

書く物: ``verdict.csv``・``seed_contrasts.csv``・``summary.md``・``report_provenance.json``。
``--analyze-only`` に相当する経路で走の provenance は書き直さない（§8）。

スモーク（G0.5）: ``--smoke`` で窓を短縮走行の地平に置き換え、全行が値を持つこと（``NOT_DETERMINED_*`` 可）と
§8 のスキーマ（``READOUT_SCHEMA``）を ``check_schema`` で照合する。読み出しのキーを 1 つ消すと
``SchemaError`` で落ちる（launcher の G0.5 の変異対照）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import act_chimera_0913 as AC       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'act_chimera_0913'
OUT = ROOT / 'results' / RUN_ID
GATE_SHAPE = ROOT / 'results' / 'gate_shape_0911'
RL0906 = ROOT / 'results' / 'pmnist_rlmnist_0906'
SPEC = ROOT / 'specs' / 'spec_act_chimera_0913.md'

PARENTS = ('LR', 'ELU1')
FAMILY = AC.FAMILY                                       # {'H': (big, small), 'V': ..., 'S': ...}
FUNCS8 = AC.FUNCS8
IX = {f: i for i, f in enumerate(FUNCS8)}
LNA, ZV = AC.LNA, AC.ZV
BAND_OF = {('H', 'N'): (1,), ('H', 'D'): (2, 3), ('V', 'N'): (1, 2), ('V', 'D'): (3,),
           ('S', 'N'): (1, 2, 3), ('S', 'D'): (1, 2, 3)}    # D0: 因子が操作する帯（B0 は z>0）
LADDER = ('LR001', 'LR', 'LR02', 'LR03', 'LIN')          # D6 の 5 段（LR002 は本走・他は gate_shape）
LADDER_COMMITTED = ('LR001', 'LR03', 'LIN')
SPLIT_D5A = {'N': dict(A='LR', B='SMAXH', G='GN', F='FN'), 'D': dict(A='ELU1', B='SMAXH', G='GD', F='FD')}
N_TWINS = {'pmnist': 4, 'rlmnist': 1}
SCR_ARMS = {'LR': 'chLR_1216', 'ELU1': 'chE_1216', 'SMAXH': 'chSMAXH_1216', 'SMINH': 'chSMINH_1216',
            'VMIN': 'chVMIN_1216', 'VMAX': 'chVMAX_1216', 'SMAXS': 'chSMAXS_1216', 'SMINS': 'chSMINS_1216',
            'GN': 'chGN_1216', 'FN': 'chFN_1216', 'GD': 'chGD_1216', 'FD': 'chFD_1216'}
SCR_PERIOD = 10_000
SCR_EVERY = 1_000
SCR_U_FLOOR = 1e-16
SCR_ALIVE_DENOM = 0.25                                   # edge_law_analyze_0905.py:50
SCR_MASS_FLOOR = 1. / 3200.                              # §4 D0: 32 パターン × 100 ユニットの 1 セル
SCR_NEAR_EOS = 1.0                                       # lr·λ/2 ≥ 1（二次形式の算術・§2.4）
RL_FLOOR_SD = 0.00026                                    # §4.0: R の (online − 最多比率) のタスク sd 0.00116/√20
RL_FLOOR_FAMILY = 24                                     # 8 腕 × 3 seed（段 2 を足しても不変・§4.0）
N_IMAGES_RL = 1200


def _norm_ppf(p):
    """Φ⁻¹(p)（scipy 無し。math.erf の二分法・|誤差| < 1e-12）。"""
    lo, hi = -40., 40.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1. + math.erf(mid / math.sqrt(2.))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RL_FLOOR_Z = _norm_ppf(1. - 0.05 / RL_FLOOR_FAMILY)      # 2.865（§4.0）


# ------------------------------------------------------------------ 窓（固定・極値は選ばない・§3.1）
@dataclass(frozen=True)
class Windows:
    pm_base: tuple = (16, 20)
    pm_late: tuple = (101, 120)
    pm_issa_base: tuple = (2, 6)
    pm_issa_late: tuple = (116, 120)
    pm_tasks: int = 120
    rl_win: tuple = (31, 50)
    rl_early: tuple = (1, 10)
    rl_tasks: int = 50
    scr_win: tuple = (491, 500)
    scr_win_1m: tuple = (91, 100)
    scr_tail: tuple = (451, 500)
    scr_tasks: int = 500
    e_ident_tasks: tuple = (20, 40, 60, 80, 100, 120)

    @staticmethod
    def smoke():
        """スモーク（箱 B t1–3・RL 1 タスク・SCR 30k = 3 タスク）の地平に置いた窓。
        窓の長さが 3 未満の対比は NOT_DETERMINED_SHORT_WINDOW になる。"""
        return Windows(pm_base=(1, 3), pm_late=(1, 3), pm_issa_base=(1, 3), pm_issa_late=(1, 3), pm_tasks=3,
                       rl_win=(1, 1), rl_early=(1, 1), rl_tasks=1, scr_win=(2, 3), scr_win_1m=(1, 2),
                       scr_tail=(2, 3), scr_tasks=3, e_ident_tasks=(2,))

    @staticmethod
    def lr0p005():
        """SCR の退避枝（lr 0.005・10M・窓 991–1000・機構の窓 951–1000・§2.4）。"""
        return Windows(scr_win=(991, 1000), scr_win_1m=(91, 100), scr_tail=(951, 1000), scr_tasks=1000)


def _tasks(win):
    return list(range(win[0], win[1] + 1))


# ------------------------------------------------------------------ §3.2 SE・σ_traj・m
class DegenerateSeries(ValueError):
    """sd = 0 または隣接対の相関が NaN の系列（m を 0 や NaN にしない・S16 (vi)）。"""


def se_window(c, rho_estimator='pair'):
    """窓内ノイズ SE_W = sd_W(c)/√n_eff（§3.2 で固定した推定量）。

    sd は ddof = 1、ρ₁ = max(0, r)、r = Pearson(c[:-1], c[1:])（各部分列をそれぞれの平均で中心化・
    ``np.corrcoef``）、n_eff = max(1, n(1−ρ₁)/(1+ρ₁))。``rho_estimator='acf'`` は S16 (vii) の変異対照
    （全体平均で中心化した標本 ACF）で、判定には使わない。"""
    c = np.asarray(c, dtype=np.float64)
    n = c.size
    if n < 3:
        raise DegenerateSeries(f'window too short for the lag-1 estimator: n={n}')
    if not np.all(np.isfinite(c)):
        raise DegenerateSeries('non-finite value in the window')
    sd = float(np.std(c, ddof=1))
    if sd == 0.:
        raise DegenerateSeries('sd = 0 in the window')
    if rho_estimator == 'pair':
        with np.errstate(invalid='ignore', divide='ignore'):
            r = float(np.corrcoef(c[:-1], c[1:])[0, 1])
    elif rho_estimator == 'acf':
        cc = c - c.mean()
        r = float((cc[:-1] * cc[1:]).sum() / (cc ** 2).sum())
    else:
        raise ValueError(rho_estimator)
    if not np.isfinite(r):
        raise DegenerateSeries('lag-1 correlation is NaN (a sub-series has zero variance)')
    rho = max(0., r)
    n_eff = max(1., n * (1. - rho) / (1. + rho))
    return sd / math.sqrt(n_eff)


def window_stat(series, win, sign, rho_estimator='pair'):
    """1 窓の (sign·mean, SE)。series は t = 1 起点の 1 次元配列（index t−1）。"""
    lo, hi = win
    if hi > len(series):
        raise DegenerateSeries(f'window {win} not covered (len={len(series)})')
    c = np.asarray(series[lo - 1:hi], dtype=np.float64)
    return sign * float(c.mean()), se_window(c, rho_estimator)


def endpoint_value(series, spec, rho_estimator='pair'):
    """endpoint の (値, SE)。spec = [(窓, 符号), ...]（L: [(base, +1), (late, −1)]、−A_late: [(late, −1)]）。
    SE = √Σ SE_窓²（窓は互いに素で独立とみなす）。"""
    val, var = 0., 0.
    for win, sign in spec:
        v, se = window_stat(series, win, sign, rho_estimator)
        val += v
        var += se * se
    return val, math.sqrt(var)


def m_of(se, weights, sigma2_arm):
    """m(c) = √(SE² + Σ_a w_a²·σ²_arm)（§3.2）。"""
    if not (np.isfinite(se) and se > 0.):
        raise DegenerateSeries(f'SE must be a positive finite number: {se}')
    w2 = float(sum(float(w) ** 2 for w in weights.values()))
    m = math.sqrt(se * se + w2 * sigma2_arm)
    if not np.isfinite(m) or m <= 0.:
        raise DegenerateSeries(f'm is not a positive finite number: {m}')
    return m


def sigma2_arm_from_twins(twin_pairs):
    """σ²_arm = ½·max(0, mean_{seed,k}(c_tw² − SE_tw²))。twin_pairs = [(c_tw, SE_tw), ...]（§3.2）。
    空なら None（呼び手が 0 にして ``(M_SE_ONLY)`` を付ける）。"""
    if not twin_pairs:
        return None
    v = np.array([c * c - se * se for c, se in twin_pairs], dtype=np.float64)
    return 0.5 * max(0., float(v.mean()))


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


# ------------------------------------------------------------------ §4.0 対比の状態・合わせ方
DIRP, DIRM, EQ, UNRES = 'DIR+', 'DIR-', 'EQ', 'UNRES'


def contrast_state(c, m):
    """3 seed の (c, m) から排他の 4 状態: DIR+ (c > m 3/3)、DIR− (c < −m 3/3)、EQ (|c| ≤ 2m 3/3 かつ DIR± でない)、UNRES。"""
    c = np.asarray(c, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if c.size == 0 or c.shape != m.shape or not (np.all(np.isfinite(c)) and np.all(np.isfinite(m))):
        raise DegenerateSeries('contrast_state needs finite c and m of equal length')
    if np.any(m <= 0):
        raise DegenerateSeries('m must be positive for every seed')
    if bool(np.all(c > m)):
        return DIRP
    if bool(np.all(c < -m)):
        return DIRM
    if bool(np.all(np.abs(c) <= 2. * m)):
        return EQ
    return UNRES


def sign_state(c, k=9, n=10):
    """SCR: ≥ k/n の符号一致で DIR±、他は UNRES（等価は無い・§4.0）。"""
    c = np.asarray(c, dtype=np.float64)
    if c.size != n or not np.all(np.isfinite(c)):
        raise DegenerateSeries(f'sign_state needs {n} finite values, got {c.size}')
    if int((c > 0).sum()) >= k:
        return DIRP
    if int((c < 0).sum()) >= k:
        return DIRM
    return UNRES


def decided(label):
    """下流で「決まっている」と数える状態か（§4.0）。UNRES・*_UNRESOLVED・NOT_TESTABLE_*・NOT_DETERMINED_*・
    _*_ONLY・CO_PRIMARY_CONFLICT・REPORT_ONLY は決まっていない。``(WEAK_DOSE)`` 付きの方向は決まっている。"""
    s = str(label)
    if s in (UNRES, '') or s.endswith('UNRESOLVED') or 'UNRESOLVED(' in s or '_UNRESOLVED_' in s:
        return False
    if s.startswith(('NOT_TESTABLE', 'NOT_DETERMINED', 'CO_PRIMARY_CONFLICT', 'REPORT_ONLY', 'NOT_APPLICABLE',
                     'OTHER(')):                                   # OTHER(...) は D5a のチャネルの「他」（決まっていない）
        return False
    if s.endswith(('_L_ONLY', '_A_ONLY', '_PT_ONLY', '_LOGIT_ONLY')):
        return False
    return True


def merge_co_primary(a, b, tags, unresolved):
    """co-primary の合わせ方（§4.0）。両方同じ → その状態。一方だけ決まる → <状態>_<tag>_ONLY。
    両方決まり違う → CO_PRIMARY_CONFLICT(tagA=…, tagB=…)。両方決まらず違う → unresolved（NOT_TESTABLE が
    片側にあればそれを優先して残す）。

    ``(WEAK_DOSE)`` は状態ではなく注記（§4.0「DIR は残し (WEAK_DOSE) を付ける」）なので、比べる前に外し、
    どちらかの endpoint が持っていれば合わせた状態に付け直す（ρ_min は endpoint ごとに違うので片側だけに付きうる）。"""
    wd = '(WEAK_DOSE)'
    if strip_dose(a) == strip_dose(b):
        return strip_dose(a) + (wd if (wd in str(a) or wd in str(b)) else '')
    da, db = decided(a), decided(b)
    if da and db:
        return f'CO_PRIMARY_CONFLICT({tags[0]}={a}, {tags[1]}={b})'
    if da:
        return f'{a}_{tags[0]}_ONLY'
    if db:
        return f'{b}_{tags[1]}_ONLY'
    for s in (a, b):
        if str(s).startswith(('NOT_TESTABLE', 'NOT_DETERMINED')):
            return s
    return unresolved


def base_state(label):
    """合わせたラベルから元の状態を取る（`HELPS_L_ONLY` → 'HELPS' は決まっていないので None）。"""
    return label if decided(label) else None


def strip_dose(s):
    return str(s).replace('(WEAK_DOSE)', '')


def carry_dose(label, *states):
    """方向の状態から組んだラベルに、使った状態の ``(WEAK_DOSE)`` を残す（§4.0）。NOT_* には付けない。"""
    lab = str(label)
    if lab.startswith('NOT_') or '(WEAK_DOSE)' in lab:
        return lab
    return lab + ('(WEAK_DOSE)' if any('(WEAK_DOSE)' in str(s) for s in states) else '')


M_SE_ONLY = '(M_SE_ONLY)'


def is_pm(env):
    """箱 B（段 1 の 'pmnist' と段 2 の 'pmnist_lr{lr}'）。"""
    return str(env).startswith('pmnist')


# ------------------------------------------------------------------ §4 D0 の用量ゲート（状態の置換）
def gate_eq(state, factor, exist_ok, dose_ok):
    """用量が足りない対比の EQ を NOT_TESTABLE_* に置き換え、DIR には (WEAK_DOSE) を付ける（§4.0・D0）。
    factor ∈ {'N', 'D', 'I'}。存在が落ちた近傍は NOT_TESTABLE_NO_CONTRAST、深部はいずれも
    NOT_TESTABLE_JOINT_UNVISITED。I は両因子のゲートを要る（呼び手が and で渡す）。"""
    if state in (DIRP, DIRM):
        return state if (exist_ok and dose_ok) else state + '(WEAK_DOSE)'
    if state == EQ and not (exist_ok and dose_ok):
        if factor == 'D':
            return 'NOT_TESTABLE_JOINT_UNVISITED'
        if factor == 'N':
            return 'NOT_TESTABLE_NO_CONTRAST' if not exist_ok else 'NOT_TESTABLE_WEAK_MANIPULATION'
        return 'NOT_TESTABLE_WEAK_MANIPULATION'
    return state


def is_dir(s, sign=None):
    s = strip_dose(s)
    if sign is None:
        return s in (DIRP, DIRM)
    return s == (DIRP if sign > 0 else DIRM)


# ------------------------------------------------------------------ §8 読み出しのスキーマ（凍結・G0.5 が照合）
class SchemaError(KeyError):
    """読み出し npz のキー・dtype・形が §8 と違う（G0.5 の変異対照はこれで落ちる）。"""


def readout_schema(env, T, U=100):
    """§8 の表。T はタスク数（箱 B 120・RL 50・スモークでは短縮）。"""
    s = {'fn_mom': ((T, 2, 8, 4, 3), 'float64'),
         'occ_start': ((T, 2, U, 4), 'float32'), 'occ_end': ((T, 2, U, 4), 'float32'),
         'mob_band': ((T, 2, U, 4), 'float32'), 'pabs1': ((T, 2, U), 'float32'),
         'zbar_start': ((T, 2, U), 'float64'), 'zbar_end': ((T, 2, U), 'float64'),
         'depth_vel': ((T, 2, U), 'float64'), 'step_num': ((T, 2, U), 'float64'),
         'wt_norm': ((T, 2, U), 'float64'), 'mu_comp': ((T, U), 'float64'),
         'adam_ratio': ((T, 2, U), 'float32'), 'adam_epsfrac': ((T, 2, U), 'float32'),
         'gbar_l2': ((T, U), 'float64'), 'mu_phi1': ((T, U), 'float64'),
         'param_sha': ((T,), '<U64')}
    if env == 'pmnist':
        s['E_ident'] = (None, 'float64')                 # (n_ident, 3): 本走 6・スモークでは短い
    return s


def check_schema(env, arrays, T, U=100, where=''):
    """§8 のキー・dtype・形。足りないキーは SchemaError、形/dtype の違いも SchemaError。"""
    for k, (shape, dtype) in readout_schema(env, T, U).items():
        if k not in arrays:
            raise SchemaError(f'{where}: readout key {k!r} missing')
        a = arrays[k]
        if str(a.dtype) != dtype and not (dtype == '<U64' and a.dtype.kind == 'U' and a.dtype.itemsize >= 4 * 64):
            raise SchemaError(f'{where}: {k} dtype {a.dtype} != {dtype}')
        if shape is not None and tuple(a.shape) != tuple(shape):
            raise SchemaError(f'{where}: {k} shape {a.shape} != {shape}')
        if k == 'E_ident' and (a.ndim != 2 or a.shape[1] != 3):
            raise SchemaError(f'{where}: E_ident shape {a.shape} != (n, 3)')
        if a.dtype.kind == 'f' and not np.all(np.isfinite(a)):
            raise SchemaError(f'{where}: {k} has non-finite values')


SCR_READOUT_SCHEMA = {'fn_mom': (('T', 'S', 8, 4, 3), 'float64'), 'occ': (('T', 'S', 100, 4), 'float32'),
                      'mob_band': (('T', 'S', 100, 4), 'float32'), 'lam_out': (('T', 'S'), 'float64'),
                      'lam_full': ((3, 'S'), 'float64'), 'relax_rate': (('S',), 'float64')}


# ------------------------------------------------------------------ 環境のデータ（読み込み）
@dataclass
class Endpoint:
    """endpoint = 腕の系列の変換 + 窓の符号付き平均（§3.1）。tag は co-primary の合わせ方の添字。"""
    name: str
    tag: str
    spec: list                      # [(窓, 符号), ...]
    transform: object = None        # 系列 -> 系列（None は恒等）

    def series(self, s):
        return np.asarray(s, dtype=np.float64) if self.transform is None else self.transform(np.asarray(s, dtype=np.float64))


def endpoints_pmnist(W, issa=False):
    b, l = (W.pm_issa_base, W.pm_issa_late) if issa else (W.pm_base, W.pm_late)
    sfx = '_ISSA' if issa else ''
    return {'L' + sfx: Endpoint('L' + sfx, 'L', [(b, +1.), (l, -1.)]),
            'A' + sfx: Endpoint('A' + sfx, 'A', [(l, -1.)])}


def endpoints_rlmnist(W, early=False):
    w = W.rl_early if early else W.rl_win
    sfx = '_EARLY' if early else ''
    return {'PT' + sfx: Endpoint('PT' + sfx, 'PT', [(w, -1.)], lambda s: 100. * s),
            'LOGIT' + sfx: Endpoint('LOGIT' + sfx, 'LOGIT', [(w, -1.)], logit)}


@dataclass
class EnvData:
    """1 環境の保存配列。series[arm][seed] は t = 1 起点の系列（箱 B: acc pt・RL: online_acc 比率）。
    SCR では series[arm][seed] は記録系列でなく、判定器が logs から作った per-task の辞書を持つ。"""
    env: str
    seeds: list
    series: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)      # arm -> seed -> COMPLETE / DIVERGED / INCOMPLETE / MISSING
    readout: dict = field(default_factory=dict)     # arm -> seed -> dict of arrays
    rows: dict = field(default_factory=dict)        # arm -> seed -> list of row dicts（gbar 等）
    extra: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)      # 読んだファイル（provenance）

    def have(self, arm, seed):
        return self.status.get(arm, {}).get(seed) == 'COMPLETE'

    def arm_status(self, arms):
        """家族（腕の集合）の打ち切り: 全 seed COMPLETE なら None、そうでなければ NOT_DETERMINED_<状態>。
        腕の打ち切り（DIVERGED・INCOMPLETE 等）を先に返し、環境全体の無効（G1 の失敗・``extra['invalid']``）はその後。"""
        for a in arms:
            for s in self.seeds:
                st = self.status.get(a, {}).get(s, 'MISSING')
                if st != 'COMPLETE':
                    return f'NOT_DETERMINED_{st}'
        return self.extra.get('invalid')


def _read_rows(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    for r in rows:
        for k, v in list(r.items()):
            if k in ('arm', 'iv', 'clamp', 'kind'):
                continue
            try:
                r[k] = float(v) if v not in ('', None, 'None') else float('nan')
            except (ValueError, TypeError):
                pass
    return rows


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _load_npz_pickle(path):
    """SCR ランナーの readout（activation 等の 0-d 文字列配列を含む）。"""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


U_SCR = 100


RUN_IDENTITY_KEYS = ('device', 'hostname', 'cpu', 'threads', 'executable', 'torch', 'numpy', 'git_hash', 'code_sha256')


def run_identity(prov):
    """S18b の比較が意味を持つための走の同一性（追補 2-2）: device・ホスト・CPU・スレッド・インタプリタ・torch・numpy・
    git_hash・コードの sha256。RL の provenance は ``machine``・``interpreter``・``threads``、箱 B は ``env_*`` のキー。
    CPU と GPU（t1 で 20 セル中 14 が食い違う）や、別のコードで回った LR と双子を比べると、摂動に依らず「生きている」に
    なるので、``twin_liveness`` はこれが LR と双子で等しい対だけを比べる。"""
    m, it = prov.get('machine') or {}, prov.get('interpreter') or {}
    dev = m.get('device', prov.get('env_device'))
    return dict(device=None if dev is None else str(dev).split(':')[0],
                hostname=m.get('hostname', prov.get('env_hostname')), cpu=m.get('cpu', prov.get('env_cpu')),
                threads=prov.get('threads'), executable=it.get('executable', prov.get('env_executable')),
                torch=it.get('torch', prov.get('env_torch')), numpy=it.get('numpy', prov.get('env_numpy')),
                git_hash=prov.get('git_hash'), code_sha256=prov.get('code_sha256'))


def _keep_sha_excl(E, arm, seed, prov):
    """S18b（追補 2-2）: LR と双子の provenance の ``checks.s18b.sha_excl_b1``（{k: タスク終端ごとの b1[k] を除いた
    params の sha256}）を ``E.extra['sha_excl'][(arm, seed)]`` に、その走の同一性（``run_identity``）を
    ``E.extra['sha_excl_ident'][(arm, seed)]`` に置く。無ければ置かない（生存を示せない双子は死んだ扱い）。"""
    if AC.twin_of(arm)[0] != 'LR':
        return
    lists = ((prov.get('checks') or {}).get('s18b') or {}).get('sha_excl_b1')
    if isinstance(lists, dict):
        E.extra.setdefault('sha_excl', {})[(arm, seed)] = {str(k): list(v) for k, v in lists.items()}
        E.extra.setdefault('sha_excl_ident', {})[(arm, seed)] = run_identity(prov)


def load_pmnist(root, seeds, arms, W, U=100, strict=True, env='pmnist'):
    """箱 B: pmnist/{ARM}_s{seed}_rows.csv（acc・gbar）、_readout.npz、_divergence.json、_provenance.json。

    状態の優先順（§4.0・§7）: 行が無い MISSING → divergence.json がある DIVERGED → provenance の
    ``checks_passed`` が False の CHECK_FAILED（走ごとの検査が落ちた走は判定に使わない）→ 地平線が足りない
    INCOMPLETE → 読み出しが無い MISSING_READOUT → COMPLETE。strict（本走）では provenance の無い走は
    MISSING_PROVENANCE。G1-PM の錨（provenance の ``checks.g1_pm``）が落ちた走が 1 つでもあれば、箱 B 全体を
    ``NOT_DETERMINED_G1_PM_FAILED`` にする（§7「G1-PM の失敗 → 箱 B を無効」・腕の打ち切りの方を先に表示）。"""
    E = EnvData(env, list(seeds))
    root = Path(root)
    for arm in arms:
        E.series[arm], E.status[arm], E.readout[arm], E.rows[arm] = {}, {}, {}, {}
        for s in seeds:
            tag = f'{arm}_s{s}'
            rp, np_, dp = root / f'{tag}_rows.csv', root / f'{tag}_readout.npz', root / f'{tag}_divergence.json'
            pv = root / f'{tag}_provenance.json'
            if not rp.exists():
                E.status[arm][s] = 'MISSING'
                continue
            rows = _read_rows(rp)
            E.inputs.append(str(rp))
            tasks = [int(r['task']) for r in rows]
            acc = np.array([100. * float(r['acc']) for r in rows], dtype=np.float64)
            E.rows[arm][s] = rows
            prov = json.loads(pv.read_text()) if pv.exists() else None
            if prov is not None:
                E.inputs.append(str(pv))
                _keep_sha_excl(E, arm, s, prov)
                g1 = (prov.get('checks') or {}).get('g1_pm') or {}
                if g1.get('anchor'):
                    E.extra.setdefault('g1_pm', {})[tag] = dict(pass_=bool(g1.get('pass_')),
                                                              n_compared=g1.get('n_compared'), note=g1.get('note'))
            if dp.exists():
                E.status[arm][s] = 'DIVERGED'
                E.extra.setdefault('divergence', {})[tag] = json.loads(dp.read_text())
                E.inputs.append(str(dp))
                E.series[arm][s] = acc
                continue
            if prov is None and strict:
                E.status[arm][s] = 'MISSING_PROVENANCE'
                E.series[arm][s] = acc
                continue
            if prov is not None and prov.get('checks_passed') is False:
                E.status[arm][s] = 'CHECK_FAILED'
                E.extra.setdefault('check_failed', {})[tag] = prov.get('failed_checks')
                E.series[arm][s] = acc
                continue
            if tasks != list(range(1, W.pm_tasks + 1)):
                if strict:
                    E.status[arm][s] = 'INCOMPLETE'
                    E.series[arm][s] = acc
                    continue
            E.series[arm][s] = acc
            if np_.exists():
                arrs = _load_npz(np_)
                check_schema('pmnist', arrs, len(tasks), U, where=str(np_))
                E.readout[arm][s] = arrs
                E.inputs.append(str(np_))
                E.status[arm][s] = 'COMPLETE'
            else:
                E.status[arm][s] = 'MISSING_READOUT' if strict else 'COMPLETE'
    if any(not v['pass_'] for v in E.extra.get('g1_pm', {}).values()):
        E.extra['invalid'] = 'NOT_DETERMINED_G1_PM_FAILED'
    return E


def load_gate_shape_rows(arms, seeds, W):
    """D6 の梯子と S16 (i)/(vii): committed の results/gate_shape_0911/{ARM}_s{seed}_rows.csv。"""
    out, inputs = {}, []
    for arm in arms:
        out[arm] = {}
        for s in seeds:
            p = GATE_SHAPE / f'{arm}_s{s}_rows.csv'
            if not p.exists():
                continue
            rows = _read_rows(p)
            inputs.append(str(p))
            assert [int(r['task']) for r in rows] == list(range(1, 121)), (arm, s)
            out[arm][s] = dict(acc=np.array([100. * float(r['acc']) for r in rows]),
                               gbar=np.array([float(r['gbar']) for r in rows]), rows=rows)
    return out, inputs


def rl_floor_F(seed, W):
    """F_seed: 各タスクのラベルの最多クラス比率の窓平均（rl_labels 系列から厳密に・§4.0）。"""
    from src import pmnist_0905 as H05
    from src import pmnist_rlmnist_0906 as RL
    g = H05.stream('rl_labels', seed)
    fr = []
    for t in range(1, max(W.rl_tasks, W.rl_win[1]) + 1):
        y = RL.task_labels(g)
        fr.append(float(torch.bincount(y, minlength=10).max()) / N_IMAGES_RL)
    fr = np.array(fr)
    return float(fr[W.rl_win[0] - 1:W.rl_win[1]].mean()), fr


def at_floor(online, F, W):
    """AT_FLOOR: O_arm(窓) ≤ F_seed + z·0.00026（§4.0）。online は t = 1 起点の比率。"""
    lo, hi = W.rl_win
    if hi > len(online):
        return None
    O = float(np.mean(online[lo - 1:hi]))
    return bool(O <= F + RL_FLOOR_Z * RL_FLOOR_SD)


def load_rlmnist(root, seeds, arms, W, U=100, strict=True):
    """RL: rlmnist/{ARM}/s{seed}/per_task.csv・readout_s{seed}.npz・divergence.json・provenance.json。
    状態の優先順は箱 B と同じ（INCOMPLETE → CHECK_FAILED → 地平線 → 読み出し）。G1-RL を GPU で当てた走
    （provenance の ``checks.g1_rl.gating == 'G1-RL'``）が落ちていれば RL 全体を ``NOT_DETERMINED_G1_RL_FAILED``
    にする（§7「G1-RL の失敗 → その環境だけを無効」）。"""
    E = EnvData('rlmnist', list(seeds))
    root = Path(root)
    E.extra['F'] = {}
    for s in seeds:
        E.extra['F'][s] = rl_floor_F(s, W)[0]
    for arm in arms:
        E.series[arm], E.status[arm], E.readout[arm], E.rows[arm] = {}, {}, {}, {}
        for s in seeds:
            d = root / arm / f's{s}'
            pp, rp, dp, pv = d / 'per_task.csv', d / f'readout_s{s}.npz', d / 'divergence.json', d / 'provenance.json'
            if not pp.exists():
                E.status[arm][s] = 'MISSING'
                continue
            rows = _read_rows(pp)
            E.inputs.append(str(pp))
            rows = [r for r in rows if int(r['seed']) == s]
            ok = [r for r in rows if np.isfinite(float(r.get('online_acc', 'nan')))]
            E.rows[arm][s] = rows
            E.series[arm][s] = np.array([float(r['online_acc']) for r in ok], dtype=np.float64)
            prov = json.loads(pv.read_text()) if pv.exists() else None
            if prov is not None:
                E.inputs.append(str(pv))
                _keep_sha_excl(E, arm, s, prov)
                g1 = (prov.get('checks') or {}).get('g1_rl') or {}
                if g1.get('gating') == 'G1-RL':
                    E.extra.setdefault('g1_rl', {})[f'{arm}_s{s}'] = dict(pass_=bool(g1.get('pass_')),
                                                                          n_cells=g1.get('n_cells'))
            if dp.exists():
                E.status[arm][s] = 'INCOMPLETE'
                E.extra.setdefault('divergence', {})[f'{arm}_s{s}'] = json.loads(dp.read_text())
                E.inputs.append(str(dp))
                continue
            if prov is None and strict:
                E.status[arm][s] = 'MISSING_PROVENANCE'
                continue
            if prov is not None and prov.get('checks_passed') is False:
                E.status[arm][s] = 'CHECK_FAILED'
                continue
            if len(ok) != W.rl_tasks and strict:
                E.status[arm][s] = 'INCOMPLETE'
                continue
            if rp.exists():
                arrs = _load_npz(rp)
                check_schema('rlmnist', arrs, len(ok), U, where=str(rp))
                E.readout[arm][s] = arrs
                E.inputs.append(str(rp))
                E.status[arm][s] = 'COMPLETE'
            else:
                E.status[arm][s] = 'MISSING_READOUT' if strict else 'COMPLETE'
    E.extra['at_floor'] = {arm: {s: at_floor(E.series[arm][s], E.extra['F'][s], W)
                                 for s in seeds if s in E.series.get(arm, {})} for arm in arms}
    if any(not v['pass_'] for v in E.extra.get('g1_rl', {}).values()):
        E.extra['invalid'] = 'NOT_DETERMINED_G1_RL_FAILED'
    return E


def combine_rl_csv(root, arms, seeds, W):
    """rlmnist/{ARM}/per_task.csv = seed ごとの csv の結合（行数を seed ごとに照合・§8）。"""
    root = Path(root)
    out = {}
    for arm in arms:
        lines, header = [], None
        for s in seeds:
            p = root / arm / f's{s}' / 'per_task.csv'
            if not p.exists():
                continue
            txt = p.read_text(encoding='utf-8').splitlines()
            if not txt:
                continue
            if header is None:
                header = txt[0]
            assert txt[0] == header, (arm, s, 'header differs')
            body = txt[1:]
            expect = W.rl_tasks if not (root / arm / f's{s}' / 'divergence.json').exists() else None
            if expect is not None and len(body) != expect:
                raise AssertionError(f'{arm} s{s}: {len(body)} rows != {expect}')
            lines += body
        if header is not None:
            (root / arm / 'per_task.csv').write_text('\n'.join([header] + lines) + '\n', encoding='utf-8')
            out[arm] = len(lines)
    return out


# ------------------------------------------------------------------ 対比の評価（箱 B・RL・§3.2）
@dataclass
class Contrast:
    """線形対比 c = Σ_a w_a·Y_a の seed ごとの評価。error が None でなければラベルは NOT_DETERMINED_*。"""
    name: str
    weights: dict
    c: list = field(default_factory=list)
    se: list = field(default_factory=list)
    m: list = field(default_factory=list)
    state: str = UNRES
    error: str = None
    note: str = ''

    def med(self, key):
        v = getattr(self, key)
        return float(np.median(v)) if v else float('nan')


def eval_contrast(E, ep, weights, sigma2, name='', rho_estimator='pair'):
    """箱 B・RL の対比を endpoint ep で評価する。sigma2 = σ²_arm（None なら 0・(M_SE_ONLY)）。"""
    K = Contrast(name or '+'.join(f'{w:+g}{a}' for a, w in weights.items()), dict(weights))
    for a in weights:
        st = E.arm_status([a])
        if st is not None:
            K.error = st
            K.state = st
            return K
    s2 = 0. if sigma2 is None else float(sigma2)
    if sigma2 is None:
        K.note = '(M_SE_ONLY)'
    try:
        for s in E.seeds:
            n = min(len(E.series[a][s]) for a in weights)
            ct = sum(float(w) * ep.series(E.series[a][s][:n]) for a, w in weights.items())
            v, se = endpoint_value(ct, ep.spec, rho_estimator)
            K.c.append(v)
            K.se.append(se)
            K.m.append(m_of(se, weights, s2))
        K.state = contrast_state(K.c, K.m)
    except DegenerateSeries as e:
        K.error = 'NOT_DETERMINED_DEGENERATE_SERIES' if 'not covered' not in str(e) and 'too short' not in str(e) \
            else 'NOT_DETERMINED_SHORT_WINDOW'
        K.state = K.error
        K.note = str(e)
    return K


TWIN_MIN_LIVE_PAIRS = 2          # 追補 2-2: 生きた対が 2 未満なら σ_traj を出さない（NOT_DETERMINED_TWIN(M_SE_ONLY)）


def twin_liveness(E, n_twins, start):
    """S18b（追補 2-2）: 双子 k・seed ごとに ``twin_live_task``（b1[k] を除いた params の sha256 が同じ seed の LR と最初に
    食い違ったタスク・無ければ None）と、endpoint の窓の最初のタスク ``start``（箱 B t16・RL t31）までに生きているか
    （twin_live_task ≤ start）。LR か双子の provenance に S18b の列が無い対は、生存を示せないので死んだ扱い。
    LR と双子の走の同一性（``run_identity``: device・git_hash・コード・インタプリタ・ホスト・CPU・スレッド）が
    記録されていないか食い違う対も、sha の食い違いが摂動に由来すると言えないので死んだ扱い（why に食い違ったキー）。
    結果は ``E.extra['twin_live']`` にも置く。"""
    out = {}
    have = E.extra.get('sha_excl', {})
    idents = E.extra.get('sha_excl_ident', {})
    for k in range(n_twins):
        tw = f'LRtw{k}'
        for s in E.seeds:
            lr = (have.get(('LR', s)) or {}).get(str(k))
            tv = (have.get((tw, s)) or {}).get(str(k))
            if lr is None or tv is None:
                out[(tw, s)] = dict(twin_live_task=None, n_compared=0, window_start=int(start), live=False,
                                    why='no S18b record (checks.s18b.sha_excl_b1) in ' +
                                        ('LR' if lr is None else '') + (' and ' if lr is None and tv is None else '') +
                                        (tw if tv is None else '') + ' provenance')
                continue
            id_lr, id_tw = idents.get(('LR', s)), idents.get((tw, s))
            if id_lr is None or id_tw is None:
                out[(tw, s)] = dict(twin_live_task=None, n_compared=0, window_start=int(start), live=False,
                                    why='no run identity (device/git_hash/code/interpreter) recorded for '
                                        + ('LR' if id_lr is None else tw))
                continue
            diff = [key for key in RUN_IDENTITY_KEYS if id_lr.get(key) != id_tw.get(key)]
            if diff:
                out[(tw, s)] = dict(twin_live_task=None, n_compared=0, window_start=int(start), live=False,
                                    why='LR/twin provenance mismatch: ' + '/'.join(diff) + ' ('
                                        + ' | '.join(f'{key}: LR={str(id_lr.get(key))[:16]} {tw}={str(id_tw.get(key))[:16]}'
                                                    for key in diff) + ')')
                continue
            r = AC.twin_live_task(lr, tv)
            live = r['twin_live_task'] is not None and r['twin_live_task'] <= int(start)
            why = '' if live else ('never differs from LR' if r['twin_live_task'] is None
                                   else f"first differs at t{r['twin_live_task']} > window start t{int(start)}")
            out[(tw, s)] = dict(r, window_start=int(start), live=bool(live), why=why)
    E.extra['twin_live'] = out
    return out


def twin_sigma2(E, ep, n_twins, start, rho_estimator='pair'):
    """null 双子から σ²_arm（endpoint ごと）。返り値 (σ² または None, 使った生きた対の数, 注記)。

    §3.2「双子がラベルの前に揃わない（発散・未完）ときは σ_arm = 0 で計算し、全ラベルに (M_SE_ONLY) を付ける」:
    LR と LRtw0..k の (腕, seed) が 1 つでも COMPLETE でなければ、残りの対から部分的にプールせず None を返す
    （呼び手が σ² = 0 にし、環境の全ラベルに (M_SE_ONLY) を付ける）。
    追補 2-2（S18b）: 窓の最初のタスク ``start`` までに LR と食い違わない双子（``twin_liveness``）は
    ``NOT_DETERMINED_TWIN_DEAD`` として除く。生きていても acc 系列が LR と揃っていれば NOT_DETERMINED_TWIN として除く。
    使えた生きた対が ``TWIN_MIN_LIVE_PAIRS``（2）未満なら None。推定量の切り詰め（max(0, ·)）で σ² = 0 になったときは
    値を返し、注記に ``clipped`` を書く（ラベルは SIGMA_TRAJ_OK のまま）。"""
    pairs, notes = [], []
    incomplete = [f'{a}_s{s}:{E.status.get(a, {}).get(s, "MISSING")}'
                  for a in ['LR'] + [f'LRtw{k}' for k in range(n_twins)] for s in E.seeds
                  if E.status.get(a, {}).get(s, 'MISSING') != 'COMPLETE']
    if incomplete:
        return None, 0, 'twins not complete before the labels -> sigma_arm = 0, (M_SE_ONLY) (§3.2): ' + ';'.join(incomplete)
    live = twin_liveness(E, n_twins, start)
    for k in range(n_twins):
        tw = f'LRtw{k}'
        for s in E.seeds:
            lv = live[(tw, s)]
            if not lv['live']:
                notes.append(f"{tw}_s{s}:NOT_DETERMINED_TWIN_DEAD(twin_live_task={lv['twin_live_task']};{lv['why']})")
                continue
            a, b = E.series[tw][s], E.series['LR'][s]
            n = min(len(a), len(b))
            if np.array_equal(a[:n], b[:n]):
                notes.append(f'{tw}_s{s}:NOT_DETERMINED_TWIN')
                continue
            try:
                ct = ep.series(a[:n]) - ep.series(b[:n])
                v, se = endpoint_value(ct, ep.spec, rho_estimator)
                pairs.append((v, se))
            except DegenerateSeries as e:
                notes.append(f'{tw}_s{s}:{e}')
    if len(pairs) < TWIN_MIN_LIVE_PAIRS:
        notes.insert(0, f'live pairs {len(pairs)} < {TWIN_MIN_LIVE_PAIRS} -> sigma_arm = 0, (M_SE_ONLY) (追補 2-2)')
        return None, len(pairs), ';'.join(notes)
    s2 = sigma2_arm_from_twins(pairs)
    if s2 == 0.:
        raw = float(np.mean([c * c - se * se for c, se in pairs]))
        notes.insert(0, f'clipped (mean(c_tw^2 - SE_tw^2) = {raw:.4g} <= 0 -> sigma2 = 0)')
    return s2, len(pairs), ';'.join(notes)


# ------------------------------------------------------------------ §3.3 Γ・Φ・Ψ（読み出し fn_mom から）
def fam_gamma_series(E, fam, X, layer, win, q=0, bands=None):
    """家族 fam・因子 X の 4 走平均の Γ_X（q=0）/Φ_X（q=1）/Ψ_X（q=2）のタスク系列（層 layer・窓 win）。
    返り値 seed -> (n_win,) 配列。bands を渡すと帯別の成分の和（既定は全帯 = Γ_X）。"""
    big, small = FAMILY[fam]
    arms = ('LR', 'ELU1', big, small)
    out = {}
    for s in E.seeds:
        vals = []
        for t in _tasks(win):
            acc = 0.
            for a in arms:
                fm = E.readout[a][s]['fn_mom'][t - 1, layer]                    # (8, 4, 3)
                C = AC.family_contrasts(torch.as_tensor(fm))[fam][X].numpy()     # (4, 3)
                acc += float(C[list(bands) if bands is not None else slice(None), q].sum())
            vals.append(acc / len(arms))
        out[s] = np.array(vals)
    return out


def pair_gamma_series(E, P, Q, layer, win, q=0, states=None):
    """単純効果 P − Q の用量 Γ_PQ = E[φ′_P − φ′_Q]（q=0）を states（既定は P と Q の 2 走）の状態で平均した
    タスク系列。seed -> (n_win,)。"""
    states = states or (P, Q)
    out = {}
    for s in E.seeds:
        vals = []
        for t in _tasks(win):
            acc = 0.
            for a in states:
                fm = E.readout[a][s]['fn_mom'][t - 1, layer]
                acc += float(fm[IX[P], :, q].sum() - fm[IX[Q], :, q].sum())
            vals.append(acc / len(states))
        out[s] = np.array(vals)
    return out


def gamma_summary(series_by_seed):
    """Γ のタスク系列 -> seed ごとの (Γ, SE_Γ)（§3.3: 入力方向の標本 SE は使わない）。"""
    out = {}
    for s, v in series_by_seed.items():
        try:
            out[s] = (float(v.mean()), se_window(v))
        except DegenerateSeries:
            out[s] = (float(v.mean()), float('nan'))
    return out


def existence(gs, absval=False):
    """存在: Γ > 2·SE_Γ（absval なら |Γ| > 2·SE）が全 seed。SE が NaN なら False。"""
    ok = []
    for g, se in gs.values():
        if not np.isfinite(se):
            ok.append(False)
        else:
            ok.append((abs(g) if absval else g) > 2. * se)
    return bool(ok) and all(ok)


def med_gamma(gs):
    return float(np.median([g for g, _ in gs.values()]))


# ------------------------------------------------------------------ §4 D0 用量ゲート（箱 B・RL）
@dataclass
class Dose:
    exist: bool
    rho: float
    rho_min: float
    layers: dict          # layer -> dict(exist, gamma_med, se, rho)
    ok: bool
    note: str = ''


def rho_min_from(K):
    """ρ_min = 2·m̃/|c̃|（対比 K 自身が DIR± のとき）、それ以外は 1（§4 D0 の非錨の規則: K = 箱 B H の同じ対比）。"""
    if K is None or K.error is not None or not is_dir(K.state):
        return 1.
    return 2. * K.med('m') / abs(K.med('c'))


def rho_min_anchor(K_judged, K_anchor):
    """箱 B H の錨の規則（§4 D0）: ρ_min = 2·m̃(判定する対比)/|K̃_X|（K̃_X = 錨の対比の seed 中央値）。
    錨 K_X がその endpoint で DIR± でなければ 1。判定する対比に m が無い（打ち切り・退化）ときも 1。"""
    if K_anchor is None or K_anchor.error is not None or not is_dir(K_anchor.state):
        return 1.
    if K_judged is None or K_judged.error is not None or not K_judged.m:
        return 1.
    return 2. * K_judged.med('m') / abs(K_anchor.med('c'))


def dose_gate(gamma_by_layer, ref_by_layer, rho_min, factor):
    """gamma_by_layer[l] = seed -> (Γ, SE)（判定する対比の用量）、ref_by_layer[l] = 同じ層の基準用量
    （seed -> (Γ, SE)）。存在は全 seed で Γ > 2SE、ρ = max_l Γ̃_l/Γ̃_ref,l、通過は「どれかの層で存在と用量が通る」。"""
    layers, rhos, exist_any, ok_any = {}, [], False, False
    for l, gs in gamma_by_layer.items():
        ex = existence(gs)
        g = med_gamma(gs)
        ref = med_gamma(ref_by_layer[l]) if l in ref_by_layer else float('nan')
        rho = g / ref if (np.isfinite(ref) and ref > 0) else float('nan')
        layers[l] = dict(exist=ex, gamma_med=g, se_med=float(np.median([se for _, se in gs.values()])),
                         ref_med=ref, rho=rho)
        exist_any |= ex
        if np.isfinite(rho):
            rhos.append(rho)
        ok_any |= bool(ex and np.isfinite(rho) and rho >= rho_min)
    rho = max(rhos) if rhos else float('nan')
    return Dose(exist_any, rho, rho_min, layers, ok_any,
                note='' if ok_any else ('NO_CONTRAST' if not exist_any else 'WEAK'))


# ------------------------------------------------------------------ §4 D1・D2・D3・D5a のラベル（状態から・純関数）
def d1_label(a, b, floor_parent=False, big_above=None):
    """D1: a = big − LR、b = big − ELU1 の（用量ゲート後の）状態から 7 つの排他ラベル。
    RL で ELU1 が床なら floor_parent=True: BIG_BEATS_BOTH → BIG_BEATS_BOTH_FLOOR_PARENT（a DIR− かつ big が床より上 3/3）。"""
    return _d12_label('BIG', a, b, floor_parent, big_above)


def d2_label(a, b):
    """D2: a = small − LR、b = small − ELU1。"""
    return _d12_label('SMALL', a, b, False, None)


def _d12_label(P, a, b, floor_parent, above):
    if floor_parent:
        # §4 D1 RL: 床の腕（ELU1）を含む b = big − ELU1 には「床より上」の向きしか付けない（§4.0）ので、b は
        # UNRES として通常の 7 択に通す。置き換えるのは BIG_BEATS_BOTH だけ（a が DIR− 3/3 かつ big が 3/3 で
        # 床より上 → BIG_BEATS_BOTH_FLOOR_PARENT）。a だけで決まる BIG_EQUALS_LR（対 (big, LR) の用量ゲート）は残る。
        if strip_dose(a) == DIRM and above:
            return carry_dose(f'{P}_BEATS_BOTH_FLOOR_PARENT', a)
        b = UNRES
    sa, sb = strip_dose(a), strip_dose(b)
    nt = [s for s in (a, b) if str(s).startswith('NOT_TESTABLE')]
    if sa == DIRM and sb == DIRM:
        return carry_dose(f'{P}_BEATS_BOTH', a, b)
    if sa == DIRP and sb == DIRP:
        return carry_dose(f'{P}_BELOW_BOTH', a, b)
    if {sa, sb} == {DIRP, DIRM}:
        return carry_dose(f'{P}_BETWEEN', a, b)
    if sa == EQ and sb == EQ:
        return f'{P}_EQUALS_BOTH'
    if sa == EQ and sb != EQ:
        return f'{P}_EQUALS_LR'
    if sb == EQ and sa != EQ:
        return f'{P}_EQUALS_ELU1'
    if nt:                                   # EQ が用量で読めない側がある: NOT_TESTABLE を残す
        return nt[0]
    return f'{P}_UNRESOLVED'


def factor_state(s):
    """N・D・I の状態: HELPS (DIR+)・HARMS (DIR−)・INERT (EQ・用量ゲート済)・NOT_TESTABLE_*・UNRESOLVED。"""
    t = strip_dose(s)
    sfx = '(WEAK_DOSE)' if '(WEAK_DOSE)' in str(s) else ''
    if t == DIRP:
        return 'HELPS' + sfx
    if t == DIRM:
        return 'HARMS' + sfx
    if t == EQ:
        return 'INERT'
    if str(s).startswith(('NOT_TESTABLE', 'NOT_DETERMINED')):
        return str(s)
    return 'UNRESOLVED'


def interaction_state(s):
    t = strip_dose(s)
    sfx = '(WEAK_DOSE)' if '(WEAK_DOSE)' in str(s) else ''
    if t == DIRP:
        return 'SUBADDITIVE' + sfx
    if t == DIRM:
        return 'SUPERADDITIVE' + sfx
    if t == EQ:
        return 'ADDITIVE'
    if str(s).startswith(('NOT_TESTABLE', 'NOT_DETERMINED')):
        return str(s)
    return 'INTERACTION_UNRESOLVED'


def d3_heading(N, D):
    """見出し（合わせた N・D から）。"""
    n, d = strip_dose(N), strip_dose(D)
    if n == 'HELPS' and d == 'HELPS':
        return 'BOTH_FACTORS_HELP'
    if n == 'HELPS' and d == 'INERT':
        return 'NEAR_ONLY'
    if n == 'INERT' and d == 'HELPS':
        return 'DEEP_ONLY'
    if n == 'INERT' and d == 'INERT':
        return 'NEITHER'
    return f'FACTORIAL_OTHER(N={N}, D={D})'


def hypothesis_label(simple, parents, I_state):
    """仮説の判定（1 endpoint・§4 D3）。simple = {'sL': K(Y_small − Y_LR), 'sE': K(Y_small − Y_ELU1),
    'Lb': K(Y_LR − Y_big), 'Eb': K(Y_ELU1 − Y_big)}（用量ゲート後の state を持つ Contrast）、
    parents = K(Y_ELU1 − Y_LR)、I_state = I の（ゲート後の）状態。返り値 (ラベル, 注記)。"""
    states = {k: K.state for k, K in simple.items()}
    if any(K.error is not None for K in simple.values()) or parents.error is not None:
        errs = [K.error for K in list(simple.values()) + [parents] if K.error]
        return errs[0] if errs[0].startswith('NOT_DETERMINED') else 'CANCELLATION_UNRESOLVED', errs[0]
    all_dirp = all(is_dir(s, +1) for s in states.values())
    if all_dirp:
        # (ii) 相殺検定: 各 seed で |Δ| + m(Δ) < min(Y_small − Y_ELU1, Y_small − Y_LR)
        ii = all(abs(parents.c[i]) + parents.m[i] < min(simple['sE'].c[i], simple['sL'].c[i])
                 for i in range(len(parents.c)))
        iii = strip_dose(I_state) != DIRM
        if ii and iii:
            return carry_dose('CANCELLATION_SUPPORTED', *states.values()), ''
        note = ('cancellation test (ii) fails' if not ii else '') + ('; ' if not ii and not iii else '') \
            + ('SUPERADDITIVE (iii) fails' if not iii else '')
        return carry_dose('PARTIAL_CANCELLATION', *states.values()), note
    if any(strip_dose(s) in (EQ, DIRM) for s in states.values()):
        bad = [k for k, s in states.items() if strip_dose(s) in (EQ, DIRM)]
        return carry_dose('CANCELLATION_REFUTED', *[states[k] for k in bad if strip_dose(states[k]) == DIRM]), \
            'simple effect ' + ','.join(bad) + ' is EQ or reversed'
    return 'CANCELLATION_UNRESOLVED', ''


def channel_state(K_move, K_stay, gate_ok):
    """D5a のチャネルの状態。K_move = K(Y_A − Y_ch)、K_stay = K(Y_ch − Y_A)（同じ対比の符号違い）。
    MOVES: Y_A − Y_ch が DIR+。STAYS: EQ かつゲート通過。ゲート落ちの STAYS は NOT_TESTABLE_WEAK_MANIPULATION、
    ゲート落ちの MOVES は MOVES(WEAK_DOSE)。"""
    if K_move.error is not None:
        return K_move.error
    s = K_move.state
    if s == DIRP:
        return 'MOVES' if gate_ok else 'MOVES(WEAK_DOSE)'
    if s == EQ:
        return 'STAYS' if gate_ok else 'NOT_TESTABLE_WEAK_MANIPULATION'
    return 'OTHER(' + s + ')'


def d5a_label(X, G, F):
    """D5a のラベル（合わせたチャネル状態から・排他）。"""
    g, f = strip_dose(G), strip_dose(F)
    if g == 'MOVES' and f == 'STAYS':
        return f'{X}_VIA_GATE'
    if f == 'MOVES' and g == 'STAYS':
        return f'{X}_VIA_FORWARD'
    if g == 'MOVES' and f == 'MOVES':
        return f'{X}_BOTH_CHANNELS'
    if g == 'STAYS' and f == 'STAYS':
        return f'{X}_NEEDS_BOTH'
    nt = [s for s in (G, F) if 'NOT_TESTABLE' in str(s)]
    return f'{X}_CHANNEL_UNRESOLVED' + (f'({nt[0]})' if nt else '')


def dpar_label(state):
    t = strip_dose(state)
    if t == EQ:
        return 'PARENTS_TIE'
    if t == DIRM:
        return 'PARENTS_ELU_BETTER'
    if t == DIRP:
        return 'PARENTS_LR_BETTER'
    if str(state).startswith('NOT_DETERMINED'):
        return str(state)
    return 'PARENTS_UNRESOLVED'


# ------------------------------------------------------------------ 判定の器（verdict.csv / seed_contrasts.csv の行）
class Report:
    def __init__(self):
        self.rows = []            # verdict.csv
        self.seed_rows = []       # seed_contrasts.csv
        self.labels = {}          # (env, family, layer, item) -> label（下流の参照用）
        self.notes = []
        self.m_se_only = set()    # σ_arm = 0 で判定した環境（§3.2: 全ラベルに (M_SE_ONLY)・apply_m_se_only）

    def put(self, env, family, layer, item, label, note=''):
        key = (env, family, str(layer), item)
        self.labels[key] = str(label)
        self.rows.append(dict(env=env, family=family, layer=layer, item=item, label=str(label), note=str(note)))

    def get(self, env, family, layer, item, default='NOT_DETERMINED_MISSING'):
        return self.labels.get((env, family, str(layer), item), default)

    def seed(self, env, endpoint, family, contrast, K, **extra):
        for i, s in enumerate(K.c) if K.c else []:
            self.seed_rows.append(dict(env=env, endpoint=endpoint, family=family, contrast=contrast, seed=i,
                                       c=K.c[i], se=K.se[i], m=K.m[i], state=K.state, note=K.note, **extra))
        if not K.c:
            self.seed_rows.append(dict(env=env, endpoint=endpoint, family=family, contrast=contrast, seed=-1,
                                       c=float('nan'), se=float('nan'), m=float('nan'), state=K.state, note=K.note, **extra))

    def seed_scalar(self, env, endpoint, family, contrast, values, state='', **extra):
        for s, v in values.items():
            self.seed_rows.append(dict(env=env, endpoint=endpoint, family=family, contrast=contrast, seed=s,
                                       c=v, se=float('nan'), m=float('nan'), state=state, note='', **extra))


def merge_all(per_ep, tags, unresolved):
    """endpoint ごとのラベル辞書 {tagA: label, tagB: label} を合わせる。"""
    a, b = per_ep[tags[0]], per_ep[tags[1]]
    return merge_co_primary(a, b, tags, unresolved)


# ------------------------------------------------------------------ 箱 B・RL の段 1（家族ごと）
class DoseCtx:
    """箱 B H の基準用量とその対比（ρ_min の錨）。他の (環境, 家族) の用量ゲートがここを参照する（§4 D0）。

    - ``ref_gamma[(key, layer)]``: 箱 B H の同じ対比・同じ層の用量（非錨の ρ = max_l Γ^{F,e,l}/Γ^{H,箱B,l}）。
    - ``ref_fwd[(X, q, layer)]``: 箱 B H の Φ_X（q = 1）・Ψ_X（q = 2）（D5b の前提の用量）。
    - ``ref_state[(key, ep_tag)]``: 箱 B H の**その対比自身**の状態・m̃・c̃ と、非錨の ρ_min
      （DIR± なら 2·m̃/|c̃|、EQ/UNRES なら 1）。錨の ρ_min（2·m̃(判定する対比)/|K̃_X|）とは別物で、
      非錨の家族・環境にはこちらを渡す。"""

    def __init__(self):
        self.ref_gamma = {}
        self.ref_fwd = {}
        self.ref_state = {}

    def rho_min(self, key, tag, env):
        """非錨の ρ_min（§4 D0）。箱 B（他家族・段 2 以外の lr は使わない）は endpoint ごと、RL は max(ρ_min^L, ρ_min^A)。
        I は N と D の大きい方。箱 B H の対比が無ければ NaN（ゲートは落ちる）。"""
        if key == 'I':
            return max(self.rho_min('N', tag, env), self.rho_min('D', tag, env))
        tags = (tag,) if is_pm(env) else ('L', 'A')
        vals = []
        for t in tags:
            st = self.ref_state.get((key, t))
            if st is None:
                return float('nan')
            vals.append(st['rho_min_nonanchor'])
        return max(vals)


PAIR_ORIENT = {'sE': ('ELU1', 'small', 'N'), 'sL': ('LR', 'small', 'D'),      # 単純効果の用量: P（近傍 1 / 深部 0.1 側）, Q, 因子
               'Lb': ('big', 'LR', 'N'), 'Eb': ('big', 'ELU1', 'D')}
DOSE_KEYS = ('N', 'D', 'sE', 'sL', 'Lb', 'Eb')
# 対比 → そのEQ を読むときの用量ゲート（§4 D0: 主効果は 4 走平均、2 走の差は対の用量。D1 の a = big − LR は
# 対 (big, LR) = Lb、b = big − ELU1 は対 (big, ELU1) = Eb）。表に無い対比は KeyError（主効果に黙って落とさない）。
GATE_OF = {'N': 'N', 'D': 'D', 'I': 'I', 'sE': 'sE', 'sL': 'sL', 'Lb': 'Lb', 'Eb': 'Eb', 'a1': 'Lb', 'b1': 'Eb'}


def _resolve(fam, name):
    big, small = FAMILY[fam]
    return {'big': big, 'small': small}.get(name, name)


def _d0_label(ok, exist, factor):
    return ('MANIPULATION_OK' if ok else
            'NOT_TESTABLE_NO_CONTRAST' if (not exist and factor == 'N') else
            'NOT_TESTABLE_JOINT_UNVISITED' if factor == 'D' else 'NOT_TESTABLE_WEAK_MANIPULATION')


def family_doses(E, env, W, fam, eps, ctx, R, K_by_ep, layers=(0, 1)):
    """D0（家族 × 因子 × 層）: 存在と用量。返り値 gates[ep_tag][key] = Dose（key ∈ N, D, I, sE, sL, Lb, Eb）。

    錨（箱 B の H・§4 D0）: ρ = Γ̃_key/Γ̃_{K_X}、ρ_min = 2·m̃(key の対比)/|K̃_X|（K_X = Lb（近傍）/ Eb（深部）が
    その endpoint で DIR± でなければ 1）。I は ρ_min_I = max_X 2·m̃(I)/|K̃_X| を N と D の両方の ρ に当てる。
    非錨: ρ = max_l Γ^{F,e,l}/Γ^{H,箱B,l}（同じ対比・同じ層）、ρ_min = 箱 B H のその対比が DIR± なら 2·m̃/|c̃|、
    でなければ 1（RL は L と A の大きい方・I は N と D の大きい方）。"""
    win = W.pm_late if is_pm(env) else W.rl_win
    big, small = FAMILY[fam]
    gates = {}
    # 用量の系列（endpoint に依らない）
    gam = {}
    for X in ('N', 'D'):
        gam[X] = {l: gamma_summary(fam_gamma_series(E, fam, X, l, win)) for l in layers}
    for key, (P, Q, X) in PAIR_ORIENT.items():
        gam[key] = {l: gamma_summary(pair_gamma_series(E, _resolve(fam, P), _resolve(fam, Q), l, win)) for l in layers}
    is_anchor = (is_pm(env) and fam == 'H')
    if is_anchor:
        for key in DOSE_KEYS:
            for l in layers:
                ctx.ref_gamma[(key, l)] = gam[key][l]
        for X in ('N', 'D'):
            for q in (1, 2):
                for l in layers:
                    ctx.ref_fwd[(X, q, l)] = gamma_summary(fam_gamma_series(E, fam, X, l, win, q=q))
    for ep in eps.values():
        Ke = K_by_ep[ep.name]
        g = {}
        for key in DOSE_KEYS:
            factor = key if key in ('N', 'D') else PAIR_ORIENT[key][2]
            if is_anchor:
                anchor = 'Lb' if factor == 'N' else 'Eb'                 # K_N = Y_LR − Y_big、K_D = Y_ELU1 − Y_big
                ref = gam[anchor]
                rmin = rho_min_anchor(Ke[key], Ke[anchor])
                Kk = Ke[key]
                ctx.ref_state[(key, ep.tag)] = dict(
                    state=Kk.state, m_med=(Kk.med('m') if Kk.error is None else float('nan')),
                    c_med=(Kk.med('c') if Kk.error is None else float('nan')), rho_min_nonanchor=rho_min_from(Kk))
            else:
                ref = {l: ctx.ref_gamma.get((key, l), {}) for l in layers}
                rmin = ctx.rho_min(key, ep.tag, env)
            g[key] = dose_gate(gam[key], ref, rmin, factor)
            if is_anchor and key in ('Lb', 'Eb'):
                g[key].note = 'anchor (not scored)'
            for l in layers:
                d = g[key].layers[l]
                R.put(env, fam, l, f'D0_{key}_{ep.name}',
                      _d0_label(bool(d['exist'] and np.isfinite(d['rho']) and d['rho'] >= rmin), d['exist'], factor),
                      f"gamma={d['gamma_med']:.4g} se={d['se_med']:.3g} ref={d['ref_med']:.4g} rho={d['rho']:.3g} "
                      f"rho_min={rmin:.3g} ({'anchor: 2m(judged)/|K_X|' if is_anchor else 'box B H same contrast'})")
        # I（交互作用）: 両因子の用量ゲートを、N と D の大きい方の ρ_min で通す
        if is_anchor:
            rmin_I = max(rho_min_anchor(Ke['I'], Ke['Lb']), rho_min_anchor(Ke['I'], Ke['Eb']))
            KI = Ke['I']
            ctx.ref_state[('I', ep.tag)] = dict(state=KI.state, rho_min_anchor=rmin_I,
                                                rho_min_nonanchor=rho_min_from(KI))
        else:
            rmin_I = ctx.rho_min('I', ep.tag, env)
        okI = all(any(bool(g[X].layers[l]['exist'] and np.isfinite(g[X].layers[l]['rho'])
                           and g[X].layers[l]['rho'] >= rmin_I) for l in layers) for X in ('N', 'D'))
        g['I'] = Dose(g['N'].exist and g['D'].exist, min(g['N'].rho, g['D'].rho), rmin_I, {}, okI,
                      note='I: rho_N and rho_D both against max rho_min')
        R.put(env, fam, 'any', f'D0_I_{ep.name}', 'MANIPULATION_OK' if okI else 'NOT_TESTABLE_WEAK_MANIPULATION',
              f'rho_N={g["N"].rho:.3g} rho_D={g["D"].rho:.3g} rho_min_I={rmin_I:.3g}')
        gates[ep.tag] = g                                   # tag（L/A・PT/LOGIT）で引く: 副窓（_ISSA/_EARLY）も主窓のゲートを使う
        for X in ('N', 'D'):
            R.put(env, fam, 'any', f'D0_{X}_{ep.name}', _d0_label(g[X].ok, g[X].exist, X),
                  f'rho={g[X].rho:.3g} rho_min={g[X].rho_min:.3g}')
    # 記録用: 帯別 Γ/Φ/Ψ（seed ごと・層ごと・late 窓平均）
    for X in ('N', 'D', 'I'):
        for l in layers:
            for q, nm in enumerate(('Gamma', 'Phi', 'Psi')):
                for k in range(4):
                    ser = fam_gamma_series(E, fam, X, l, win, q=q, bands=(k,))
                    R.seed_scalar(env, 'late', fam, f'{nm}_{X}_B{k}_l{l}', {s: float(v.mean()) for s, v in ser.items()})
    if fam == 'V':
        for l in layers:
            pz = {s: float(np.mean([E.readout[a][s]['occ_end'][t - 1, l, :, 3].mean() for a in ('LR', 'ELU1', big, small)
                                    for t in _tasks(win)])) for s in E.seeds}
            R.seed_scalar(env, 'late', fam, f'P_z_below_zv_l{l}', pz)
    return gates


def judge_family(E, env, W, fam, eps, ctx, R, sigma2, tags, extra_tag='', floor=None, stage2_arms=False):
    """段 1 の家族判定（箱 B・RL）: 対比・D0・D1・D2・D3・仮説。extra_tag は副窓（'_ISSA'/'_EARLY'）。
    floor = RL の at_floor 辞書（arm -> seed -> bool）。返り値: 合わせた N・D の状態（D8 用）。"""
    big, small = FAMILY[fam]
    arms = ('LR', 'ELU1', big, small)
    cens = E.arm_status(arms)
    unres = 'UNRESOLVED'
    items = ['D_PAR', 'D1', 'D2', 'D3_N', 'D3_D', 'D3_I', 'D3_HEADING', 'HYPOTHESIS']
    if cens is not None:
        for it in items:
            R.put(env, fam, 'all', it + extra_tag, cens)
        return dict(N=cens, D=cens)
    contrasts = {'PAR': {'ELU1': 1, 'LR': -1}, 'a1': {big: 1, 'LR': -1}, 'b1': {big: 1, 'ELU1': -1},
                 'sL': {small: 1, 'LR': -1}, 'sE': {small: 1, 'ELU1': -1},
                 'Lb': {'LR': 1, big: -1}, 'Eb': {'ELU1': 1, big: -1},
                 'N': {'LR': .5, small: .5, 'ELU1': -.5, big: -.5},
                 'D': {'ELU1': .5, small: .5, 'LR': -.5, big: -.5},
                 'I': {big: .5, small: .5, 'LR': -.5, 'ELU1': -.5}}
    K = {ep.name: {k: eval_contrast(E, ep, w, sigma2.get(ep.tag), name=k) for k, w in contrasts.items()}
         for ep in eps.values()}
    for ep in eps.values():
        for k, Kc in K[ep.name].items():
            R.seed(env, ep.name, fam, k, Kc)
    # D0（主窓だけ。副窓は主窓の用量ゲートを使う。読み出しが無ければゲート無し = 生の状態を使い、その旨を記録）
    have_ro = all(s in E.readout.get(a, {}) for a in arms for s in E.seeds)
    if not extra_tag:
        if have_ro:
            gates = family_doses(E, env, W, fam, eps, ctx, R, K)
        else:
            gates = None
            for X in ('N', 'D'):
                R.put(env, fam, 'any', f'D0_{X}', 'NOT_DETERMINED_MISSING_READOUT')
        ctx.__dict__.setdefault('gates', {})[(env, fam)] = gates
    else:
        gates = ctx.__dict__.get('gates', {}).get((env, fam))
    per = {it: {} for it in items + ['N_DEEP']}
    for ep in eps.values():
        Ke = K[ep.name]
        g = gates[ep.tag] if gates else None

        def gs(key):
            """用量ゲート後の状態。ゲートは ``GATE_OF``（D1 の a/b は対の用量 Lb/Eb・§4 D0）。未知の対比は KeyError。"""
            gk = GATE_OF[key]
            Kc = Ke[key]
            if Kc.error is not None:
                return Kc.error
            if g is None:
                return Kc.state
            f = gk if gk in ('N', 'D', 'I') else PAIR_ORIENT[gk][2]
            d = g[gk]
            return gate_eq(Kc.state, f, d.exist, d.ok)

        def neg(s):
            return {DIRP: DIRM, DIRM: DIRP}.get(strip_dose(s), s) + ('(WEAK_DOSE)' if '(WEAK_DOSE)' in s else '')

        fl = {a: any(bool(floor[a].get(s)) for s in E.seeds) for a in arms} if floor else {a: False for a in arms}
        above = {a: all(floor[a].get(s) is False for s in E.seeds) for a in arms} if floor else {a: True for a in arms}
        # D-par
        if fl['ELU1'] or fl['LR']:
            per['D_PAR'][ep.tag] = ('PARENTS_LR_ABOVE_FLOOR_ARM' if fl['ELU1'] and above['LR'] else
                                    'PARENTS_ELU_ABOVE_FLOOR_ARM' if fl['LR'] and above['ELU1'] else 'PARENTS_UNRESOLVED')
        else:
            per['D_PAR'][ep.tag] = dpar_label(Ke['PAR'].state if Ke['PAR'].error is None else Ke['PAR'].error)
        # D1
        a1, b1 = gs('a1'), gs('b1')                       # 対 (big, LR)・対 (big, ELU1) の用量ゲート
        if fl[big]:
            per['D1'][ep.tag] = 'BIG_UNRESOLVED'
        elif fl['ELU1']:
            per['D1'][ep.tag] = d1_label(a1, b1, floor_parent=True, big_above=above[big]) if not fl['LR'] else 'BIG_UNRESOLVED'
        elif fl['LR']:
            per['D1'][ep.tag] = 'BIG_UNRESOLVED'
        else:
            per['D1'][ep.tag] = d1_label(a1, b1)
        # D2
        if fl[small] or fl['ELU1']:
            per['D2'][ep.tag] = 'NOT_TESTABLE_FLOOR'
        elif fl['LR']:
            per['D2'][ep.tag] = 'SMALL_UNRESOLVED'
        else:
            per['D2'][ep.tag] = d2_label(gs('sL'), gs('sE'))
        # D3
        any_floor = any(fl.values())
        if any_floor:
            for X in ('N', 'D', 'I'):
                per['D3_' + X][ep.tag] = 'NOT_TESTABLE_FLOOR'
            deep_dom = all(floor[a].get(s) for a in ('ELU1', small) for s in E.seeds) and above[big] and above['LR']
            if deep_dom:
                per['D3_HEADING'][ep.tag] = 'DEEP_DOMINATES_FLOOR'
                nd = gs('Lb')
                per['N_DEEP'][ep.tag] = {DIRP: 'NEAR_HELPS_GIVEN_DEEP', EQ: 'NEAR_INERT_GIVEN_DEEP',
                                         DIRM: 'NEAR_HURTS_GIVEN_DEEP'}.get(strip_dose(nd),
                                                                            nd if str(nd).startswith('NOT_TESTABLE') else 'NEAR_GIVEN_DEEP_UNRESOLVED')
                if '(WEAK_DOSE)' in nd and not per['N_DEEP'][ep.tag].startswith('NOT_'):
                    per['N_DEEP'][ep.tag] += '(WEAK_DOSE)'
            else:
                per['D3_HEADING'][ep.tag] = 'NOT_TESTABLE_FLOOR'
                per['N_DEEP'][ep.tag] = 'NOT_APPLICABLE'
            per['HYPOTHESIS'][ep.tag] = 'CANCELLATION_NOT_TESTABLE_RL_FLOOR'
            continue
        N, D, I = factor_state(gs('N')), factor_state(gs('D')), interaction_state(gs('I'))
        per['D3_N'][ep.tag], per['D3_D'][ep.tag], per['D3_I'][ep.tag] = N, D, I
        per['D3_HEADING'][ep.tag] = d3_heading(N, D)
        simple = {}
        for k in ('sL', 'sE', 'Lb', 'Eb'):
            Kc = Ke[k]
            Kg = Contrast(k, Kc.weights, Kc.c, Kc.se, Kc.m, gs(k), Kc.error, Kc.note)
            simple[k] = Kg
        lab, note = hypothesis_label(simple, Ke['PAR'], gs('I'))
        per['HYPOTHESIS'][ep.tag] = lab
        per.setdefault('HYP_NOTE', {})[ep.tag] = note
        per['N_DEEP'][ep.tag] = 'NOT_APPLICABLE'
    merged = {}
    for it in items + ['N_DEEP']:
        if not per[it]:
            continue
        u = {'D_PAR': 'PARENTS_UNRESOLVED', 'D1': 'BIG_UNRESOLVED', 'D2': 'SMALL_UNRESOLVED', 'D3_N': 'UNRESOLVED',
             'D3_D': 'UNRESOLVED', 'D3_I': 'INTERACTION_UNRESOLVED', 'HYPOTHESIS': 'CANCELLATION_UNRESOLVED',
             'N_DEEP': 'NEAR_GIVEN_DEEP_UNRESOLVED', 'D3_HEADING': 'FACTORIAL_UNRESOLVED'}[it]
        if it == 'D3_HEADING':
            merged[it] = d3_heading(merged['D3_N'], merged['D3_D']) if not per[it][tags[0]].endswith('FLOOR') \
                else merge_all(per[it], tags, u)
        else:
            merged[it] = merge_all(per[it], tags, u)
        note = ' | '.join(f'{t}={per[it][t]}' for t in tags if t in per[it])
        if it == 'HYPOTHESIS':
            hn = per.get('HYP_NOTE', {})
            note += ' | ' + ' | '.join(f'{t}:{hn[t]}' for t in tags if hn.get(t))
            if is_pm(env) and fam == 'H':
                merged[it] += '(SMINH_ONLY)'
        R.put(env, fam, 'all', it + extra_tag, merged[it], note)
    return dict(N=merged.get('D3_N', unres), D=merged.get('D3_D', unres))


def judge_d5a(E, env, W, eps, ctx, R, sigma2, tags):
    """D5a 分離腕（箱 B・段 1）。近傍: A=LR, B=SMAXH, G=GN, F=FN。深部: A=ELU1, B=SMAXH, G=GD, F=FD。"""
    win = W.pm_late if is_pm(env) else W.rl_win
    out = {}
    for X, d in SPLIT_D5A.items():
        A, B, G, F_ = d['A'], d['B'], d['G'], d['F']
        cens = E.arm_status([A, B, G, F_])
        if cens is not None:
            R.put(env, 'H', 'all', f'D5A_{X}', cens)
            out[X] = cens
            continue
        per_G, per_F, notes = {}, {}, []
        for ep in eps.values():
            s2 = sigma2.get(ep.tag)
            KAB = eval_contrast(E, ep, {A: 1, B: -1}, s2, name=f'{A}-{B}')
            KAG = eval_contrast(E, ep, {A: 1, G: -1}, s2, name=f'{A}-{G}')
            KAF = eval_contrast(E, ep, {A: 1, F_: -1}, s2, name=f'{A}-{F_}')
            for Kc in (KAB, KAG, KAF):
                R.seed(env, ep.name, 'H', f'D5A_{X}_' + Kc.name, Kc)
            if KAB.error is not None or not all(KAB.c[i] > 2 * KAB.m[i] for i in range(len(KAB.c))):
                per_G[ep.tag] = per_F[ep.tag] = 'NOT_TESTABLE_NO_SIMPLE_EFFECT'
                continue
            # 分離腕の操作ゲート（分離腕自身の late 状態・層ごと）
            gate = {}
            for ch, Kch, q, absv in (('G', KAG, 0, False), ('F', KAF, 1, True)):
                arm = G if ch == 'G' else F_
                rmin = 2. * Kch.med('m') / abs(KAB.med('c'))
                ok_any, lay = False, {}
                for l in (0, 1):
                    gs_ = gamma_summary(pair_gamma_series(E, B, A, l, win, q=q, states=(arm,)))
                    ref = gamma_summary(pair_gamma_series(E, B, A, l, win, q=q, states=(B,)))
                    ex = existence(gs_, absval=absv)
                    gm, rm = med_gamma(gs_), med_gamma(ref)
                    rho = gm / rm if rm != 0 else float('nan')
                    lay[l] = dict(exist=ex, gamma=gm, ref=rm, rho=rho)
                    ok_any |= bool(ex and np.isfinite(rho) and rho >= rmin)
                    R.put(env, 'H', l, f'D5A_{X}_GATE_{ch}_{ep.name}', 'MANIPULATION_OK' if (ex and np.isfinite(rho) and rho >= rmin)
                          else 'NOT_TESTABLE_WEAK_MANIPULATION', f'val={gm:.4g} ref={rm:.4g} rho={rho:.3g} rho_min={rmin:.3g} exist={ex}')
                gate[ch] = ok_any
            per_G[ep.tag] = channel_state(KAG, None, gate['G'])
            per_F[ep.tag] = channel_state(KAF, None, gate['F'])
            shares = {i: (KAG.c[i] / KAB.c[i], KAF.c[i] / KAB.c[i]) for i in range(len(KAB.c))}
            R.seed_scalar(env, ep.name, 'H', f'D5A_{X}_share_gate', {i: v[0] for i, v in shares.items()})
            R.seed_scalar(env, ep.name, 'H', f'D5A_{X}_share_forward', {i: v[1] for i, v in shares.items()})
        mG = merge_all(per_G, tags, 'CHANNEL_UNRESOLVED') if len(per_G) == 2 else 'CHANNEL_UNRESOLVED'
        mF = merge_all(per_F, tags, 'CHANNEL_UNRESOLVED') if len(per_F) == 2 else 'CHANNEL_UNRESOLVED'
        if 'NOT_TESTABLE_NO_SIMPLE_EFFECT' in (mG, mF) or all(v == 'NOT_TESTABLE_NO_SIMPLE_EFFECT' for v in per_G.values()):
            lab = 'NOT_TESTABLE_NO_SIMPLE_EFFECT' if all(v == 'NOT_TESTABLE_NO_SIMPLE_EFFECT' for v in per_G.values()) \
                else d5a_label(X, mG, mF)
        else:
            lab = d5a_label(X, mG, mF)
        R.put(env, 'H', 'all', f'D5A_{X}', lab, f'G={mG} F={mF} | ' + ' | '.join(f'{t}: G={per_G.get(t)} F={per_F.get(t)}' for t in tags))
        out[X] = lab
    return out


def secondary_series(E, env, W, fam, R):
    """副: t1 からの N_t・D_t・I_t と、キメラが親から離れる最初のタスク（param_sha）。"""
    big, small = FAMILY[fam]
    if E.arm_status(('LR', 'ELU1', big, small)) is not None:
        return
    for s in E.seeds:
        n = min(len(E.series[a][s]) for a in ('LR', 'ELU1', big, small))
        Y = {a: (100. if env == 'rlmnist' else 1.) * E.series[a][s][:n] for a in ('LR', 'ELU1', big, small)}
        Nt = .5 * (Y['LR'] + Y[small] - Y['ELU1'] - Y[big])
        Dt = .5 * (Y['ELU1'] + Y[small] - Y['LR'] - Y[big])
        It = .5 * (Y[big] + Y[small] - Y['LR'] - Y['ELU1'])
        # N/D/I: 損失の向き（Y = −acc の規約で >0 が「その因子が効く」）のタスク系列。N_acc/D_acc/I_acc: acc の向きの
        # 系列そのもの（endpoint の spec（L = base − late）を当てると §0.3 の N・D・I になる。D6 の反実仮想版はこちら）。
        E.extra.setdefault('series_NDI', {})[(fam, s)] = dict(N=(-Nt).tolist(), D=(-Dt).tolist(), I=(-It).tolist(),
                                                               N_acc=Nt.tolist(), D_acc=Dt.tolist(), I_acc=It.tolist())
        if fam == 'H':
            for ch, par in ((big, 'ELU1'), (small, 'LR')):
                if s in E.readout.get(ch, {}) and s in E.readout.get(par, {}):
                    a, b = E.readout[ch][s]['param_sha'], E.readout[par][s]['param_sha']
                    diff = [t + 1 for t in range(min(len(a), len(b))) if a[t] != b[t]]
                    R.seed_scalar(env, 'all', fam, f'first_departure_{ch}_from_{par}', {s: (diff[0] if diff else -1)})


# ------------------------------------------------------------------ SCR（§2.4・§3.3–3.4・§4.0）
def scr_patterns():
    """32 パターン x ∈ {0,1}^5 − 0.5 → (32, 5) float64。"""
    x = np.array([[(i >> j) & 1 for j in range(5)] for i in range(32)], dtype=np.float64)
    return x - 0.5


def scr_state(zbar, w_free):
    """z = z̄ + w_free·(x − 0.5)（厳密な 32 パターン支持・§3.3）。zbar (U,), w_free (U, 5) -> (32, U) float64。"""
    return np.asarray(zbar, dtype=np.float64)[None, :] + scr_patterns() @ np.asarray(w_free, dtype=np.float64).T


def scr_readout_from_log(log, act_name, W, lr=0.01):
    """1 系列の logs npz -> タスク t = 1..T の readout（fn_mom (T,8,4,3)・occ (T,U,4)・mob_band・lam_out (T,)）。
    act_name は 8 関数名 or 分離腕名（mob_band は自腕の dphi = φ′_bwd・§1.5）。"""
    step = log['step']
    T = W.scr_tasks
    zbar, dz = log['layer1_zbar'], log['layer1_dzbar']
    wf, wfs = log['layer1_w_free'], log['layer1_w_free_step']
    U = zbar.shape[1]
    fn = np.full((T, 8, 4, 3), np.nan)
    occ = np.full((T, U, 4), np.nan, dtype=np.float32)
    mob = np.full((T, U, 4), np.nan, dtype=np.float32)
    lam = np.full(T, np.nan)
    dphi_own = AC.DPHI[act_name]
    phi_own = AC.PHI[act_name]
    for t in range(1, T + 1):
        st = t * SCR_PERIOD
        k = int(np.searchsorted(step, st))
        if k >= len(step) or int(step[k]) != st:
            break                                                    # 早く止まった走: 残りは NaN
        kw = int(np.searchsorted(wfs, st))
        if kw >= len(wfs) or int(wfs[kw]) != st:
            break
        z = torch.as_tensor(scr_state(zbar[k], wf[kw]))
        fn[t - 1] = AC.fn_mom(z).numpy()
        occ[t - 1] = AC.occupancy(z).numpy().astype(np.float32)
        mob[t - 1] = AC.mob_band(dphi_own(z), z).numpy().astype(np.float32)
        ph = phi_own(z).numpy()                                      # (32, U)
        lam[t - 1] = float(np.linalg.eigvalsh(2. * ph.T @ ph / ph.shape[0])[-1])
    return dict(fn_mom=fn, occ=occ, mob_band=mob, lam_out=lam)


def scr_relax_rate(log, W, alive_only=False):
    """緩和率（§3.4 SCR）: tail 窓のタスク t の終端記録 k と直前 k−1（step 差 1000 を assert）で、
    y = dzbar[k, i] を x = zmax[k−1, i] − zmax*_i（zmax* = tail 窓の終端 zmax のユニット別中央値）に OLS。
    relax_rate = −傾き。alive_only は denom[k−1] > ALIVE_DENOM のユニットだけ。"""
    step, zmax, dz = log['step'], log['layer1_zmax'], log['layer1_dzbar']
    ks = []
    for t in _tasks(W.scr_tail):
        st = t * SCR_PERIOD
        k = int(np.searchsorted(step, st))
        if k >= len(step) or int(step[k]) != st or k == 0:
            continue
        assert int(step[k] - step[k - 1]) == SCR_EVERY, (t, step[k], step[k - 1])
        ks.append(k)
    if not ks:
        return float('nan'), 0
    zstar = np.median(np.stack([zmax[k] for k in ks]).astype(np.float64), axis=0)         # (U,)
    xs, ys = [], []
    for k in ks:
        x = zmax[k - 1].astype(np.float64) - zstar
        y = dz[k].astype(np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        if alive_only:
            keep &= log['layer1_denom'][k - 1].astype(np.float64) > SCR_ALIVE_DENOM
        xs.append(x[keep])
        ys.append(y[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.size < 3 or np.std(x) == 0:
        return float('nan'), int(x.size)
    A = np.stack([x, np.ones_like(x)], 1)
    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    return -slope, int(x.size)


def scr_level(log, win):
    """Y = log10 U(窓)。U = タスク終端の unfit の平均（床 1e−16）。窓の全記録が床なら INCONCLUSIVE_EXACT_FIT。"""
    step, unfit = log['step'], log['unfit']
    vals = []
    for t in _tasks(win):
        st = t * SCR_PERIOD
        k = int(np.searchsorted(step, st))
        if k >= len(step) or int(step[k]) != st:
            return float('nan'), 'INCOMPLETE'
        vals.append(float(unfit[k]))
    v = np.array(vals)
    if np.all(v <= SCR_U_FLOOR):
        return float('nan'), 'INCONCLUSIVE_EXACT_FIT'
    return float(np.log10(max(float(v.mean()), SCR_U_FLOOR))), 'OK'


def scr_mech(log, win):
    """窓の機構量: submerged 率（zmax ≤ 0 のユニット割合の窓平均）と z̄ の中央値（窓平均）。"""
    step = log['step']
    sub, zb, onset = [], [], 0
    for t in _tasks(win):
        st = t * SCR_PERIOD
        k = int(np.searchsorted(step, st))
        if k >= len(step) or int(step[k]) != st:
            return float('nan'), float('nan')
        sub.append(float((log['layer1_zmax'][k] <= 0).mean()))
        zb.append(float(np.median(log['layer1_zbar'][k])))
    return float(np.mean(sub)), float(np.mean(zb))


def load_scr(root, W, arms=SCR_ARMS, lr=0.01, lam_full_path=None, build_readout=True):
    """SCR: scr/logs/{ARM}_seed{s}.npz（10 系列）・arm_status/。判定器が readout_{ARM}.npz を作る（§8）。
    lam_full は S19（SCR ランナー側の冪乗法）が書く JSON {arm: [[ckpt0 per seed], [1M], [5M]]} を読む。
    無ければ NaN にし、NEAR_EOS は λ_out だけで判定して LAM_FULL_MISSING を記録する。"""
    root = Path(root)
    E = EnvData('scr', list(range(10)))
    E.extra['lr'] = lr
    lam_full = {}
    lf = Path(lam_full_path) if lam_full_path else root / 'lam_full.json'
    if lf.exists():
        lam_full = json.loads(lf.read_text())
        E.inputs.append(str(lf))
    E.extra['lam_full_source'] = str(lf) if lf.exists() else 'MISSING'
    for name, arm in arms.items():
        E.series[name], E.status[name], E.readout[name] = {}, {}, {}
        st_path = root / 'arm_status' / f'{arm}.json'
        done = root / 'arm_status' / f'{arm}_done.json'
        status = 'MISSING'
        if st_path.exists():
            ev = json.loads(st_path.read_text())
            if ev.get('status') == 'NUMERIC_DIVERGENCE':
                status = 'NUMERIC_DIVERGENCE'
                E.extra.setdefault('divergence', {})[arm] = ev
                E.inputs.append(str(st_path))
        if done.exists() and status == 'MISSING':
            status = 'COMPLETE'
            E.inputs.append(str(done))
        logs = {}
        for s in range(10):
            p = root / 'logs' / f'{arm}_seed{s}.npz'
            if p.exists():
                logs[s] = _load_npz(p)
                E.inputs.append(str(p))
        if len(logs) != 10:
            E.status[name] = {s: ('MISSING' if s not in logs else status) for s in range(10)}
            if status == 'COMPLETE':
                status = 'INCOMPLETE'
            E.status[name] = {s: status for s in range(10)}
            continue
        act = name
        ro = {'fn_mom': [], 'occ': [], 'mob_band': [], 'lam_out': [], 'relax_rate': [], 'relax_rate_alive': [],
              'n_records': [], 'Y': [], 'Y_status': [], 'Y_1m': [], 'sub': [], 'zbar': [], 'sub_1m': [], 'zbar_1m': [],
              'reached_1m': []}
        # SCR ランナーの build_readout（readout_{ARM}.npz・lam_full は冪乗法つき）があればそれを読む（§8）。
        # 無ければ logs から自前で作る（lam_full は NaN・LAM_FULL_MISSING を記録）。
        runner_ro = root / f'readout_{arm}.npz'
        use_runner = build_readout and runner_ro.exists()
        lam_valid = False
        if use_runner:
            rr_ = _load_npz_pickle(runner_ro)
            E.inputs.append(str(runner_ro))
            tasks = rr_['task'].astype(int)
            T = W.scr_tasks
            for k, shape in (('fn_mom', (T, 10, 8, 4, 3)), ('occ', (T, 10, U_SCR, 4)), ('mob_band', (T, 10, U_SCR, 4)),
                             ('lam_out', (T, 10))):
                if k not in rr_:
                    raise SchemaError(f'{runner_ro}: key {k!r} missing')
                full = np.full(shape, np.nan, dtype=rr_[k].dtype)
                for i, t in enumerate(tasks):
                    if 1 <= t <= T:
                        full[t - 1] = rr_[k][i]
                rr_[k] = full
            for k in ('relax_rate',):
                if k not in rr_ or rr_[k].shape != (10,):
                    raise SchemaError(f'{runner_ro}: {k} missing or shape {rr_.get(k, np.zeros(0)).shape} != (10,)')
            lam_full_arr = rr_['lam_full'] if rr_.get('lam_full') is not None and rr_['lam_full'].ndim == 2 else np.full((3, 10), np.nan)
            # ランナーの λ_full の妥当性（最終 ckpt = 登録の地平線まで計算され、全系列で収束・λ_max を取れた）
            total = W.scr_tasks * SCR_PERIOD
            lam_valid = bool(lam_full_arr.shape[0] > 0 and 'lam_full_step' in rr_ and len(rr_['lam_full_step'])
                             and int(rr_['lam_full_step'][-1]) == total
                             and bool(np.all(rr_['lam_full_ok'][-1])) if 'lam_full_ok' in rr_ else False)
            if arm not in lam_full and lam_full_arr.shape[0] > 0:
                lam_full[arm] = lam_full_arr.tolist()
                E.extra['lam_full_source'] = str(runner_ro) if E.extra['lam_full_source'] == 'MISSING' else E.extra['lam_full_source']
        for s in range(10):
            lg = logs[s]
            if use_runner:
                for k in ('fn_mom', 'occ', 'mob_band', 'lam_out'):
                    ro[k].append(rr_[k][:, s])
                ro['relax_rate'].append(float(rr_['relax_rate'][s]))
                ro['relax_rate_alive'].append(float(rr_['relax_rate_alive'][s]) if 'relax_rate_alive' in rr_ else float('nan'))
            elif build_readout:
                r = scr_readout_from_log(lg, act, W, lr)
                for k in ('fn_mom', 'occ', 'mob_band', 'lam_out'):
                    ro[k].append(r[k])
                rr, n = scr_relax_rate(lg, W)
                ra, _ = scr_relax_rate(lg, W, alive_only=True)
                ro['relax_rate'].append(rr)
                ro['relax_rate_alive'].append(ra)
            ro['n_records'].append(int(len(lg['step'])))
            y, ys = scr_level(lg, W.scr_win)
            ro['Y'].append(y)
            ro['Y_status'].append(ys)
            ro['Y_1m'].append(scr_level(lg, W.scr_win_1m)[0])
            sb, zb = scr_mech(lg, W.scr_win)
            ro['sub'].append(sb)
            ro['zbar'].append(zb)
            sb1, zb1 = scr_mech(lg, W.scr_win_1m)
            ro['sub_1m'].append(sb1)
            ro['zbar_1m'].append(zb1)
            ro['reached_1m'].append(bool(int(lg['step'][-1]) >= 1_000_000))
        for s in range(10):
            E.series[name][s] = dict(Y=ro['Y'][s], Y_1m=ro['Y_1m'][s], SUB=ro['sub'][s], ZBAR=-ro['zbar'][s],
                                     SUB_1m=ro['sub_1m'][s], ZBAR_1m=-ro['zbar_1m'][s])
        if build_readout:
            ro_arr = dict(fn_mom=np.stack(ro['fn_mom'], 1), occ=np.stack(ro['occ'], 1), mob_band=np.stack(ro['mob_band'], 1),
                          lam_out=np.stack(ro['lam_out'], 1),
                          lam_full=np.array(lam_full.get(arm, [[np.nan] * 10] * 3), dtype=np.float64),
                          relax_rate=np.array(ro['relax_rate']), relax_rate_alive=np.array(ro['relax_rate_alive']))
            ro_arr['lam_full_valid'] = bool(lam_valid) if use_runner else False
            E.readout[name] = ro_arr
        st = status
        if st == 'COMPLETE':
            if any(v == 'INCONCLUSIVE_EXACT_FIT' for v in ro['Y_status']):
                st = 'INCONCLUSIVE_EXACT_FIT'
            elif any(v == 'INCOMPLETE' for v in ro['Y_status']):
                st = 'INCOMPLETE'
            elif build_readout:
                lam = E.readout[name]['lam_out']
                lo, hi = W.scr_win
                med = np.nanmedian(lam[lo - 1:hi], axis=0)                         # (S,) 窓の中央値
                margin_out = lr * med / 2.
                lfull = E.readout[name]['lam_full']
                margin_full = lr * lfull[-1] / 2.
                # §2.4 NEAR_EOS = 窓の lr·λ_out/2 の中央値 ≥ 1 **または** 最終 ckpt の lr·λ_full/2 ≥ 1。登録の地平線の走で
                # λ_full が無い（ckpt 欠け・冪乗法未実行・未収束）なら、λ_out だけで NEAR_EOS にならない限り判定できない
                # （LAM_FULL_MISSING → 家族は NOT_DETERMINED_LAM_FULL_MISSING。黙って OK にしない）。
                registered_horizon = (W.scr_tasks * SCR_PERIOD) in (5_000_000, 10_000_000)
                full_ok = bool(np.all(np.isfinite(lfull[-1]))) and E.readout[name].get('lam_full_valid', True)
                E.extra.setdefault('margin', {})[name] = dict(out=margin_out.tolist(), full=margin_full.tolist(),
                                                              lam_full_valid=bool(full_ok))
                if np.any(margin_out >= SCR_NEAR_EOS):
                    st = 'NEAR_EOS'
                elif registered_horizon and not full_ok:
                    st = 'LAM_FULL_MISSING'
                elif np.any(np.nan_to_num(margin_full, nan=0.) >= SCR_NEAR_EOS):
                    st = 'NEAR_EOS'
        E.status[name] = {s: st for s in range(10)}
        E.extra.setdefault('n_records', {})[name] = ro['n_records']
    sn = root / 's_null.json'                                    # G1-SCR（§7: 落ちたら SCR だけを無効）
    if sn.exists():
        g1 = json.loads(sn.read_text())
        E.inputs.append(str(sn))
        E.extra['g1_scr'] = bool(g1.get('pass_'))
        if not g1.get('pass_'):
            E.extra['invalid'] = 'NOT_DETERMINED_G1_SCR_FAILED'
    return E


def scr_contrast(E, weights, key='Y'):
    """SCR の対比 c_s = Σ w_a Y_a,s（10 系列）。Contrast.c に 10 値、state は 9/10 の符号。"""
    K = Contrast('+'.join(f'{w:+g}{a}' for a, w in weights.items()), dict(weights))
    cens = E.arm_status(list(weights))
    if cens is not None:
        K.error = cens
        K.state = cens
        return K
    try:
        for s in E.seeds:
            K.c.append(float(sum(w * E.series[a][s][key] for a, w in weights.items())))
        K.se = [float('nan')] * len(K.c)
        K.m = [float('nan')] * len(K.c)
        K.state = sign_state(K.c)
    except DegenerateSeries as e:
        K.error, K.state, K.note = 'NOT_DETERMINED_DEGENERATE_SERIES', 'NOT_DETERMINED_DEGENERATE_SERIES', str(e)
    return K


def scr_existence(E, fam, X, prefix=''):
    """D0 SCR: 操作する帯の質量が ≥ 1/3200 の系列が ≥ 9/10、かつ Γ_X^F > 0（窓平均・4 走平均）。"""
    big, small = FAMILY[fam]
    arms = ('LR', 'ELU1', big, small)
    if E.arm_status(arms) is not None or any(not E.readout.get(a) for a in arms):
        return None, float('nan'), 0
    bands = BAND_OF[(fam, X)]
    win = E.extra['win']
    n_ok, gam = 0, []
    for s in E.seeds:
        mass = np.mean([E.readout[a]['occ'][t - 1, s, :, list(bands)].sum(-1).mean() for a in arms for t in _tasks(win)])
        n_ok += int(mass >= SCR_MASS_FLOOR)
        g = 0.
        for a in arms:
            for t in _tasks(win):
                C = AC.family_contrasts(torch.as_tensor(E.readout[a]['fn_mom'][t - 1, s]))[fam][X].numpy()
                g += float(C[:, 0].sum())
        gam.append(g / (len(arms) * len(_tasks(win))))
    G = float(np.median(gam))
    return bool(n_ok >= 9 and G > 0), G, n_ok


def judge_scr(E, W, R, key='Y', tag=''):
    """SCR の段 1（向きだけ・§4.0）。key = 'Y'（log10 U）/'SUB'/'ZBAR'/'Y_1m'。tag は見出しの前置（'SUB_' 等）。"""
    E.extra['win'] = W.scr_win
    env = 'scr'
    out = {}
    for fam, (big, small) in FAMILY.items():
        arms = ('LR', 'ELU1', big, small)
        cens = E.arm_status(arms)
        if cens is not None:
            for it in ('D_PAR', 'D1', 'D2', 'D3_N', 'D3_D', 'D3_I', 'D3_HEADING', 'HYPOTHESIS'):
                R.put(env, fam, 1, tag + it, cens)
            out[fam] = dict(N=cens, D=cens)
            continue
        Ks = {'PAR': scr_contrast(E, {'ELU1': 1, 'LR': -1}, key), 'a1': scr_contrast(E, {big: 1, 'LR': -1}, key),
              'b1': scr_contrast(E, {big: 1, 'ELU1': -1}, key), 'sL': scr_contrast(E, {small: 1, 'LR': -1}, key),
              'sE': scr_contrast(E, {small: 1, 'ELU1': -1}, key),
              'N': scr_contrast(E, {'LR': .5, small: .5, 'ELU1': -.5, big: -.5}, key),
              'D': scr_contrast(E, {'ELU1': .5, small: .5, 'LR': -.5, big: -.5}, key),
              'I': scr_contrast(E, {big: .5, small: .5, 'LR': -.5, 'ELU1': -.5}, key)}
        for k, Kc in Ks.items():
            R.seed(env, key, fam, tag + k, Kc)
        R.put(env, fam, 1, tag + 'D_PAR', dpar_label(Ks['PAR'].state), f"E<LR {int((np.array(Ks['PAR'].c) < 0).sum())}/10")
        R.put(env, fam, 1, tag + 'D1', d1_label(Ks['a1'].state, Ks['b1'].state))
        R.put(env, fam, 1, tag + 'D2', d2_label(Ks['sL'].state, Ks['sE'].state))
        N, D = factor_state(Ks['N'].state), factor_state(Ks['D'].state)
        N = 'UNRESOLVED' if N == 'INERT' else N
        D = 'UNRESOLVED' if D == 'INERT' else D
        I = interaction_state(Ks['I'].state)
        for it, v in (('D3_N', N), ('D3_D', D), ('D3_I', I)):
            R.put(env, fam, 1, tag + it, v)
        R.put(env, fam, 1, tag + 'D3_HEADING', d3_heading(N, D))
        simple = [Ks['sL'].state, Ks['sE'].state, {DIRP: DIRM, DIRM: DIRP}.get(Ks['a1'].state, Ks['a1'].state),
                  {DIRP: DIRM, DIRM: DIRP}.get(Ks['b1'].state, Ks['b1'].state)]
        if all(s == DIRP for s in simple):
            hyp = 'CANCELLATION_DIRECTIONAL_SUPPORT'
        elif any(s == DIRM for s in simple):
            hyp = 'CANCELLATION_DIRECTIONAL_AGAINST'
        else:
            hyp = 'CANCELLATION_UNRESOLVED'
        R.put(env, fam, 1, tag + 'HYPOTHESIS', hyp)
        if key == 'Y' and not tag:
            for X in ('N', 'D'):
                ok, G, n_ok = scr_existence(E, fam, X)
                lab = 'NOT_DETERMINED_MISSING_READOUT' if ok is None else ('MANIPULATION_OK' if ok else
                      ('NOT_TESTABLE_JOINT_UNVISITED' if X == 'D' else 'NOT_TESTABLE_NO_CONTRAST'))
                R.put(env, fam, 1, f'D0_{X}', lab, f'gamma={G:.4g} n_mass_ok={n_ok}/10')
            for it in ('D4', 'D5B', 'D6'):
                R.put(env, fam, 1, it, 'NOT_APPLICABLE_SCR')
        out[fam] = dict(N=N, D=D)
    # D5a（向きだけ）
    for X, d in SPLIT_D5A.items():
        A, B, G, F_ = d['A'], d['B'], d['G'], d['F']
        cens = E.arm_status([A, B, G, F_])
        if cens is not None:
            R.put(env, 'H', 1, tag + f'D5A_{X}', cens)
            continue
        KBA = scr_contrast(E, {B: 1, A: -1}, key)
        if KBA.state == UNRES:
            R.put(env, 'H', 1, tag + f'D5A_{X}', 'NOT_TESTABLE_NO_SIMPLE_EFFECT')
            continue
        sgn = 1 if KBA.state == DIRP else -1
        ch = {}
        for name, arm, q in (('G', G, 0), ('F', F_, 1)):
            K = scr_contrast(E, {arm: 1, A: -1}, key)
            moves = K.state == (DIRP if sgn > 0 else DIRM)
            if E.readout.get(arm) and E.readout.get(B):
                bands = BAND_OF[('H', X)]
                n_ok, vals = 0, []
                for s in E.seeds:
                    mass = np.mean([E.readout[arm]['occ'][t - 1, s, :, list(bands)].sum(-1).mean() for t in _tasks(W.scr_win)])
                    n_ok += int(mass >= SCR_MASS_FLOOR)
                    fm = np.mean([E.readout[arm]['fn_mom'][t - 1, s] for t in _tasks(W.scr_win)], 0)
                    vals.append(float(fm[IX[B], :, q].sum() - fm[IX[A], :, q].sum()))
                v = float(np.median(vals))
                fmB = np.mean([E.readout[B]['fn_mom'][t - 1, s] for s in E.seeds for t in _tasks(W.scr_win)], 0)
                vB = float(fmB[IX[B], :, q].sum() - fmB[IX[A], :, q].sum())
                gate_ok = n_ok >= 9 and ((v > 0) if q == 0 else (v != 0 and np.sign(v) == np.sign(vB)))
                if moves:
                    ch[name] = 'MOVES' if gate_ok else 'MOVES(WEAK_DOSE)'
                else:
                    ch[name] = 'STAYS' if gate_ok else 'NOT_TESTABLE_WEAK_MANIPULATION'
            else:                                              # 読み出しが無ければゲートを評価できない（黙って通さない）
                ch[name] = 'MOVES(GATE_NOT_EVALUATED)' if moves else 'NOT_DETERMINED_MISSING_READOUT'
            R.seed(env, key, 'H', tag + f'D5A_{X}_{arm}-{A}', K)
        # §4 D5a SCR: ゲートの落ちたチャネルを「動かない」側に数えない（そのラベルは NOT_TESTABLE_WEAK_MANIPULATION）。
        g_mv = ch['G'].startswith('MOVES')
        f_mv = ch['F'].startswith('MOVES')
        if g_mv and f_mv:
            lab = f'{X}_BOTH_DIRECTIONAL'
        elif g_mv or f_mv:
            other = ch['F'] if g_mv else ch['G']
            if other.startswith('NOT_'):
                lab = other
            else:
                lab = f'{X}_GATE_DIRECTIONAL' if g_mv else f'{X}_FORWARD_DIRECTIONAL'
        else:
            lab = f'{X}_CHANNEL_UNRESOLVED'
            nt = [c for c in ch.values() if c.startswith('NOT_')]
            if nt:
                lab += f'({nt[0]})'
        R.put(env, 'H', 1, tag + f'D5A_{X}', lab, f"G={ch['G']} F={ch['F']}")
    # D7 緩和率（H・9/10）
    if key == 'Y' and not tag:
        big, small = FAMILY['H']
        if E.arm_status(('LR', 'ELU1', big, small)) is None and all(E.readout.get(a) for a in ('LR', 'ELU1', big, small)):
            rr = {a: E.readout[a]['relax_rate'] for a in ('LR', 'ELU1', big, small)}
            c1, c2 = rr[big] - rr['ELU1'], rr['LR'] - rr[small]
            try:
                s1, s2 = sign_state(c1), sign_state(c2)
                lab = 'DEEP_RELAX_FASTER' if (s1 == DIRP and s2 == DIRP) else 'DEEP_RELAX_SLOWER' if (s1 == DIRM and s2 == DIRM) \
                    else 'DEEP_RELAX_UNRESOLVED'
            except DegenerateSeries as e:
                lab, s1, s2 = 'NOT_DETERMINED_DEGENERATE_SERIES', str(e), ''
            R.put(env, 'H', 1, 'D7_DEEP_RELAX', lab, f'c1={s1} c2={s2}')
            R.seed_scalar(env, 'tail', 'H', 'relax_rate_c1', dict(enumerate(c1.tolist())))
            R.seed_scalar(env, 'tail', 'H', 'relax_rate_c2', dict(enumerate(c2.tolist())))
        else:
            R.put(env, 'H', 1, 'D7_DEEP_RELAX', E.arm_status(('LR', 'ELU1', big, small)) or 'NOT_DETERMINED_MISSING_READOUT')
        for name in E.status:
            st = E.status[name][0]
            R.put(env, 'arm', name, 'SCR_STATUS', st, f"n_records={E.extra.get('n_records', {}).get(name)}"
                  + (f" margin_out_max={max(E.extra['margin'][name]['out']):.3g}" if name in E.extra.get('margin', {}) else ''))
        R.put(env, 'all', 1, 'LAM_FULL_SOURCE', 'PRESENT' if E.extra['lam_full_source'] != 'MISSING' else 'LAM_FULL_MISSING',
              E.extra['lam_full_source'])
        # 退避枝の起動条件（主 8 腕の NUMERIC_DIVERGENCE か NEAR_EOS・§2.4）
        trig = [n for n in FUNCS8 if E.status.get(n, {}).get(0) in ('NUMERIC_DIVERGENCE', 'NEAR_EOS')]
        undet = [n for n in FUNCS8 if E.status.get(n, {}).get(0) not in ('COMPLETE', 'NUMERIC_DIVERGENCE', 'NEAR_EOS')]
        R.put(env, 'all', 1, 'LR_FALLBACK_TRIGGER', 'TRIGGER_LR0P005' if trig else
              (f'NOT_DETERMINED_FALLBACK_TRIGGER' if undet else 'NO_TRIGGER'),
              'trigger=' + ','.join(trig) + ' undetermined=' + ','.join(f'{n}:{E.status.get(n, {}).get(0)}' for n in undet))
        R.put(env, 'all', 1, 'G1_SCR', {True: 'G1_SCR_OK', False: 'G1_SCR_FAILED'}.get(E.extra.get('g1_scr'), 'NOT_DETERMINED_S_NULL_MISSING'))
    return out


# ------------------------------------------------------------------ 2 回目の判定器: D4（箱の一貫性・H 対 S）
def d4_factor(sH, sS, dose_ok_H, dose_ok_S, s_dose_ge_h):
    """因子ごとの真理表（§4 D4）。s は合わせた D3 の状態（HELPS/HARMS/INERT/他）。"""
    if not (dose_ok_H and dose_ok_S):
        return 'BOX_CONSISTENCY_NOT_TESTABLE'
    h, s = strip_dose(sH), strip_dose(sS)
    dec = ('HELPS', 'HARMS', 'INERT')
    if h not in dec or s not in dec:
        return 'UNRESOLVED'
    if h == s:
        if h == 'INERT':
            return 'CONSISTENT_INERT' if s_dose_ge_h else 'UNRESOLVED'
        return 'CONSISTENT'
    return 'SHARPNESS_DEPENDENT'


def d4_heading(fN, fD):
    if 'SHARPNESS_DEPENDENT' in (fN, fD):
        return 'SHARPNESS_DEPENDENT'
    if fN in ('CONSISTENT', 'CONSISTENT_INERT') and fD in ('CONSISTENT', 'CONSISTENT_INERT'):
        return 'BOX_CONSISTENT_INERT' if fN == fD == 'CONSISTENT_INERT' else 'BOX_CONSISTENT'
    return 'BOX_CONSISTENCY_UNRESOLVED'


def judge_d4(env, R, ctx):
    gates = ctx.__dict__.get('gates', {})
    labs = {}
    for X in ('N', 'D'):
        sH, sS = R.get(env, 'H', 'all', 'D3_' + X), R.get(env, 'S', 'all', 'D3_' + X)
        gH, gS = gates.get((env, 'H')), gates.get((env, 'S'))
        okH = bool(gH) and any(g[X].ok for g in gH.values())
        okS = bool(gS) and any(g[X].ok for g in gS.values())
        ge = False
        if okH and okS:
            for e in gH:
                for l in gH[e][X].layers:
                    a, b = gH[e][X].layers[l]['gamma_med'], gS[e][X].layers[l]['gamma_med']
                    ge |= bool(np.isfinite(a) and np.isfinite(b) and b >= a)
        lab = d4_factor(sH, sS, okH, okS, ge)
        labs[X] = lab
        R.put(env, 'H+S', 'all', f'D4_{X}', lab + f'({X})', f'H={sH} S={sS} doseH={okH} doseS={okS}')
    R.put(env, 'H+S', 'all', 'D4', d4_heading(labs['N'], labs['D']), f"N={labs['N']} D={labs['D']}")


# ------------------------------------------------------------------ D5b（V 家族の帯別予言・RL のみ）
def judge_d5b(E, env, W, eps, ctx, R, sigma2, tags):
    if env != 'rlmnist':
        for X in ('N', 'D'):
            R.put(env, 'V', 'all', f'D5B_{X}', 'REPORT_ONLY_V',
                  'NOT_TESTABLE_COLLINEAR (box B near, §0.4-8)' if env == 'pmnist' else 'NOT_TESTABLE_JOINT_UNVISITED (SCR deep)')
        return
    bigV, smallV = FAMILY['V']
    cens = E.arm_status(('LR', 'ELU1', bigV, smallV, 'SMAXH', 'SMINH'))
    if cens is not None:
        for X in ('N', 'D'):
            R.put(env, 'V', 'all', f'D5B_{X}', cens)
        return
    win = W.rl_win
    gates = ctx.__dict__.get('gates', {}).get((env, 'H'))
    for X in ('N', 'D'):
        per = {}
        for ep in eps.values():
            s2 = sigma2.get(ep.tag)
            wH = {'N': {'LR': .5, 'SMINH': .5, 'ELU1': -.5, 'SMAXH': -.5}, 'D': {'ELU1': .5, 'SMINH': .5, 'LR': -.5, 'SMAXH': -.5}}[X]
            wV = {'N': {'LR': .5, smallV: .5, 'ELU1': -.5, bigV: -.5}, 'D': {'ELU1': .5, smallV: .5, 'LR': -.5, bigV: -.5}}[X]
            KH, KV = eval_contrast(E, ep, wH, s2, f'{X}_H'), eval_contrast(E, ep, wV, s2, f'{X}_V')
            if KH.error or KV.error:
                per[ep.tag] = KH.error or KV.error
                continue
            if R.get(env, 'H', 'all', 'D3_N').endswith('FLOOR') or R.get(env, 'H', 'all', 'D3_' + X) == 'NOT_TESTABLE_FLOOR':
                per[ep.tag] = 'NOT_TESTABLE_FLOOR'
                continue
            KN, KD = eval_contrast(E, ep, {'LR': .5, 'SMINH': .5, 'ELU1': -.5, 'SMAXH': -.5}, s2), \
                eval_contrast(E, ep, {'ELU1': .5, 'SMINH': .5, 'LR': -.5, 'SMAXH': -.5}, s2)
            # 前提（§4 D5b）: H の分母 Γ^H・Φ^H・Ψ^H が D0（存在と用量）を通る。予言は層 1（index 0）で組むので、
            # 前提も層 1 で見る。Γ は family_doses のゲートの層 0、Φ・Ψ は同じ規則（タスク系列の SE で |·| > 2SE が全 seed、
            # ρ = |Φ̃^{H,RL}|/|Φ̃^{H,箱B}| ≥ ρ_min）をここで当てる。
            pre = {}
            if gates:
                d0 = gates[ep.tag][X].layers.get(0, {})
                pre['Gamma'] = bool(d0.get('exist') and np.isfinite(d0.get('rho', np.nan))
                                    and d0['rho'] >= gates[ep.tag][X].rho_min)
                for q, nm in ((1, 'Phi'), (2, 'Psi')):
                    gsq = gamma_summary(fam_gamma_series(E, 'H', X, 0, win, q=q))
                    ref = ctx.ref_fwd.get((X, q, 0))
                    rho = (abs(med_gamma(gsq)) / abs(med_gamma(ref))) if ref and med_gamma(ref) != 0 else float('nan')
                    rmin = ctx.rho_min(X, ep.tag, env)
                    pre[nm] = bool(existence(gsq, absval=True) and np.isfinite(rho) and rho >= rmin)
            okH = bool(gates) and all(pre.values())
            sep = (KN.error is None and KD.error is None and bool(KN.c) and
                   all(abs(KN.c[i]) > 2 * KN.m[i] and abs(KD.c[i]) > 2 * KD.m[i] for i in range(len(KN.c))))
            if not okH or not sep:
                per[ep.tag] = 'NOT_TESTABLE_WEAK_MANIPULATION' if not okH else f'{X}_CHANNEL_UNRESOLVED_V'
                R.put(env, 'V', 0, f'D5B_{X}_PRECONDITION_{ep.name}', 'OK' if (okH and sep) else 'NOT_MET',
                      f'layer=1 (index 0) denominators={pre} |N^H|,|D^H|>2m={sep}')
                continue
            R.put(env, 'V', 0, f'D5B_{X}_PRECONDITION_{ep.name}', 'OK', f'layer=1 (index 0) denominators={pre}')
            # 層 1 の帯別 Γ（seed ごと・窓平均・4 走平均）
            res = []
            for i, s in enumerate(E.seeds):
                def gam(fam, XX, bands, q=0):
                    return float(np.mean(fam_gamma_series(E, fam, XX, 0, win, q=q, bands=bands)[s]))
                eB1 = KN.c[i] / gam('H', 'N', (1,))
                edeep = KD.c[i] / gam('H', 'D', (2, 3))
                pg = eB1 * gam('V', X, (1,)) + edeep * gam('V', X, (2, 3))
                XH = KH.c[i]
                phiH, phiV = gam('H', X, (0, 1, 2, 3), 1), gam('V', X, (0, 1, 2, 3), 1)
                psiH, psiV = gam('H', X, (0, 1, 2, 3), 2), gam('V', X, (0, 1, 2, 3), 2)
                pf = sorted([XH * phiV / phiH, XH * psiV / psiH])
                pg_global = XH * gam('V', X, (0, 1, 2, 3)) / gam('H', X, (0, 1, 2, 3))
                R.seed_rows.append(dict(env=env, endpoint=ep.name, family='V', contrast=f'D5B_{X}', seed=s, c=KV.c[i],
                                        se=KV.se[i], m=KV.m[i], state='', note=f'p_g={pg:.4g} P_f=[{pf[0]:.4g},{pf[1]:.4g}] p_g_global={pg_global:.4g}'))
                res.append((KV.c[i], KV.m[i], pg, pf))
            dist_pf = [max(0., pf[0] - x, x - pf[1]) for x, _, _, pf in res]
            if not all(max(0., pf[0] - pg, pg - pf[1]) >= 2 * m for _, m, pg, pf in res):
                per[ep.tag] = 'NOT_TESTABLE_COLLINEAR'
            elif all(abs(x - pg) <= m and d > m for (x, m, pg, pf), d in zip(res, dist_pf)):
                per[ep.tag] = f'{X}_VIA_GATE_V'
            elif all(d <= m and abs(x - pg) > m for (x, m, pg, pf), d in zip(res, dist_pf)):
                per[ep.tag] = f'{X}_VIA_FORWARD_V'
            else:
                per[ep.tag] = f'{X}_CHANNEL_UNRESOLVED_V'
        R.put(env, 'V', 'all', f'D5B_{X}', merge_all(per, tags, f'{X}_CHANNEL_UNRESOLVED_V') if len(per) == 2 else 'NOT_DETERMINED_MISSING',
              ' | '.join(f'{t}={v}' for t, v in per.items()))


# ------------------------------------------------------------------ D6（leaky 線からの残差・箱 B）
def ladder_bracket(G, Ga, rungs=LADDER):
    """D6: Ḡ_arm を挟む隣り合う 2 段 (lo, hi) と λ（§4 D6）。段は **Ḡ の値で** 選ぶ（名前の文字列順ではない:
    max('LR001', 'LR') == 'LR001' なので名前で選ぶと Ḡ_LR001 < Ḡ_LR ≤ Ḡ_arm のとき LR001 を拾う）。
    Ḡ_arm がどれかの段にちょうど等しければ hi は次の段（無ければ lo）で λ = 0。範囲外は None。"""
    order = sorted(rungs, key=lambda a: G[a])
    if not (G[order[0]] <= Ga <= G[order[-1]]):
        return None
    lo = max((a for a in order if G[a] <= Ga), key=lambda a: G[a])
    above = [a for a in order if G[a] >= Ga and a != lo]
    hi = min(above, key=lambda a: G[a]) if above else lo
    lam = 0. if (hi == lo or G[hi] == G[lo]) else (Ga - G[lo]) / (G[hi] - G[lo])
    return lo, hi, lam


def ladder_weights(arm, lo, hi, lam):
    """c = Y_arm − (1−λ)Y_lo − λY_hi の係数（同じ段が lo と hi を兼ねても重みを足し合わせ、0 の項は落とす。
    Σw² = 1 + (1−λ)² + λ²・hi = lo なら 2）。"""
    w = {}
    for a, v in ((arm, 1.), (lo, -(1. - lam)), (hi, -lam)):
        w[a] = w.get(a, 0.) + v
    return {a: v for a, v in w.items() if v != 0.}


def judge_d6(E, ladder, W, eps, R, sigma2, tags, rungs=LADDER):
    """梯子 = {LR001, LR, LR02, LR03, LIN}（同じ seed）。Ḡ = 層 1 gbar の late 窓平均。"""
    env = 'pmnist'
    win = W.pm_late

    def gbar_of(arm, s):
        if arm in ladder and s in ladder[arm]:
            g = ladder[arm][s]['gbar']
        elif s in E.rows.get(arm, {}):
            g = np.array([float(r['gbar']) for r in E.rows[arm][s]])
        else:
            return None
        if len(g) < win[1]:
            return None
        return float(g[win[0] - 1:win[1]].mean())

    def acc_of(arm, s):
        if arm in ladder and s in ladder[arm]:
            return ladder[arm][s]['acc']
        return E.series.get(arm, {}).get(s)

    arm_labels = {}
    for fam in ('H', 'S'):
        for arm in FAMILY[fam]:
            if E.arm_status([arm]) is not None:
                arm_labels[arm] = E.arm_status([arm])
                R.put(env, fam, 1, f'D6_{arm}', arm_labels[arm])
                continue
            per, notes, outside = {}, [], False
            for ep in eps.values():
                cs, ms, ses = [], [], []
                try:
                    for s in E.seeds:
                        G = {a: gbar_of(a, s) for a in rungs}
                        Ga = gbar_of(arm, s)
                        if Ga is None or any(v is None for v in G.values()):
                            raise DegenerateSeries('ladder rung missing')
                        br = ladder_bracket(G, Ga, rungs)
                        if br is None:
                            outside = True
                            raise DegenerateSeries(f'outside ladder: G={Ga:.3f} range=[{min(G.values()):.3f},{max(G.values()):.3f}]')
                        lo, hi, lam = br
                        w = ladder_weights(arm, lo, hi, lam)
                        n = min(len(acc_of(a, s)) for a in w)
                        ct = sum(wa * ep.series(acc_of(a, s)[:n]) for a, wa in w.items())
                        v, se = endpoint_value(ct, ep.spec)
                        cs.append(v)
                        ses.append(se)
                        ms.append(m_of(se, w, sigma2.get(ep.tag) or 0.))
                        notes.append(f's{s}:{lo}<{arm}<{hi} lam={lam:.2f} G={Ga:.3f}')
                        R.seed_rows.append(dict(env=env, endpoint=ep.name, family=fam, contrast=f'D6_resid_{arm}', seed=s,
                                                c=v, se=se, m=ms[-1], state='', note=notes[-1]))
                    st = contrast_state(cs, ms)
                    per[ep.tag] = {EQ: f'{arm}_ON_LEAKY_LINE', DIRM: f'{arm}_BELOW_LEAKY_LINE',
                                   DIRP: f'{arm}_ABOVE_LEAKY_LINE'}.get(st, f'{arm}_LINE_UNRESOLVED')
                except DegenerateSeries as e:
                    per[ep.tag] = 'NOT_TESTABLE_OUTSIDE_LADDER' if outside else 'NOT_DETERMINED_DEGENERATE_SERIES'
                    notes.append(str(e))
            lab = merge_all(per, tags, f'{arm}_LINE_UNRESOLVED') if len(per) == 2 else 'NOT_DETERMINED_MISSING'
            arm_labels[arm] = lab
            R.put(env, fam, 1, f'D6_{arm}', lab, ' | '.join(f'{t}={v}' for t, v in per.items()) + ' | ' + '; '.join(notes[:6]))
        big, small = FAMILY[fam]
        lb, ls = arm_labels.get(big, ''), arm_labels.get(small, '')
        if lb.endswith('ON_LEAKY_LINE') and ls.endswith('ON_LEAKY_LINE'):
            head = 'GATE_MASS_ONLY'
        elif lb.endswith('BELOW_LEAKY_LINE') and ls.endswith('ABOVE_LEAKY_LINE'):
            head = 'LOCATION_MATTERS'
        else:
            head = 'GATE_WORTH_UNRESOLVED'
        R.put(env, fam, 1, 'D6', head, f'{big}={lb} {small}={ls}')
        # 報告（ラベル無し）: 反実仮想版 c_t = N_t/Γ_N − D_t/Γ_D
        if E.arm_status(('LR', 'ELU1', big, small)) is None and all(s in E.readout.get(a, {}) for a in ('LR', 'ELU1', big, small) for s in E.seeds):
            for s in E.seeds:
                gN = float(np.mean(fam_gamma_series(E, fam, 'N', 0, win)[s]))
                gD = float(np.mean(fam_gamma_series(E, fam, 'D', 0, win)[s]))
                nd = E.extra.get('series_NDI', {}).get((fam, s))
                if nd and gN != 0 and gD != 0:
                    ct = np.array(nd['N_acc']) / gN - np.array(nd['D_acc']) / gD     # acc の向き: L の spec で N/Γ_N − D/Γ_D
                    try:
                        v, se = endpoint_value(ct, eps[list(eps)[0]].spec)
                    except DegenerateSeries:
                        v, se = float('nan'), float('nan')
                    R.seed_rows.append(dict(env=env, endpoint='L', family=fam, contrast='D6_counterfactual_N/GN-D/GD', seed=s,
                                            c=v, se=se, m=float('nan'), state='REPORT', note=f'GN={gN:.4g} GD={gD:.4g}'))


# ------------------------------------------------------------------ D7（機構・H 家族・層ごと・層別）
BINS = ((1. / 3., .5), (.5, .75), (.75, 1.0000001))


def stratum_units(occ_start_t, which):
    """タスク開始の帯分率 (U, 4) から層（near = argmax{B0, B1, B2∪B3} が B1、deep = B2∪B3）と支配分率。"""
    p = np.stack([occ_start_t[:, 0], occ_start_t[:, 1], occ_start_t[:, 2] + occ_start_t[:, 3]], 1)
    am = p.argmax(1)
    idx = 1 if which == 'near' else 2
    members = np.where(am == idx)[0]
    return members, p[members, idx]


def readout_values(ro, t, layer, name, U):
    """読み出し R のユニット別の値（タスク t・層 layer）。"""
    if name == 'NEAR_BAND_SINKING':
        return (ro['zbar_end'][t - 1, layer] < LNA).astype(np.float64)
    if name == 'DEEP_BAND_RETURN':
        return (ro['zbar_end'][t - 1, layer] >= LNA).astype(np.float64)
    if name == 'NEAR_BAND_STEP':
        return ro['step_num'][t - 1, layer].astype(np.float64)
    if name == 'NEAR_BAND_ADAM':
        return ro['adam_ratio'][t - 1, layer].astype(np.float64)
    if name == 'DEEP_BAND_VELOCITY':
        return ro['depth_vel'][t - 1, layer].astype(np.float64)
    raise KeyError(name)


D7_READOUTS = {'NEAR_BAND_SINKING': ('near', False), 'NEAR_BAND_STEP': ('near', True), 'NEAR_BAND_ADAM': ('near', True),
               'DEEP_BAND_VELOCITY': ('deep', True), 'DEEP_BAND_RETURN': ('deep', False)}


def bin_series(ro, t_list, layer, name, U=None):
    """1 走の読み出し R の、層（開始時の帯分率・§3.4）の bin ごとのタスク系列（その bin にユニットが無いタスクは NaN）、
    bin ごとの件数、層全体（bin に分けない）の系列。"""
    which, _ = D7_READOUTS[name]
    ser = {b: [] for b in range(len(BINS))}
    counts = {b: 0 for b in range(len(BINS))}
    whole = []
    for t in t_list:
        mem, dom = stratum_units(ro['occ_start'][t - 1, layer], which)
        vals = readout_values(ro, t, layer, name, U)
        for b, (lo, hi) in enumerate(BINS):
            sel = mem[(dom >= lo) & (dom < hi)]
            counts[b] += len(sel)
            ser[b].append(float(vals[sel].mean()) if len(sel) else np.nan)
        whole.append(float(vals[mem].mean()) if len(mem) else np.nan)
    return {b: np.array(v) for b, v in ser.items()}, counts, np.array(whole)


def stratified_series(readouts, arms, t_list, layer, name, U):
    """4 走で共通の bin（全走で 1 ユニット以上）を使い、4 走をプールした件数で重み付けした層別の読み出しの系列。
    返り値 (arm -> (n_t,) 配列, 共通 bin の数, 層全体の系列 arm -> (n_t,), 正規化した bin の重み {b: ŵ_b})。
    タスクごとには空の bin を除いて重みを付け直す。"""
    per = {a: bin_series(readouts[a], t_list, layer, name, U) for a in arms}
    common = [b for b in range(len(BINS)) if all(per[a][1][b] >= 1 for a in arms)]
    total = sum(sum(per[a][1][b] for a in arms) for b in common) or 1
    wb = {b: sum(per[a][1][b] for a in arms) / total for b in common}
    out, whole = {}, {}
    for a in arms:
        bs, _, wh = per[a]
        ser = []
        for i in range(len(t_list)):
            num = den = 0.
            for b in common:
                v = bs[b][i]
                if np.isfinite(v):
                    num += wb[b] * v
                    den += wb[b]
            ser.append(num / den if den > 0 else np.nan)
        out[a] = np.array(ser)
        whole[a] = wh
    return out, len(common), whole, wb


def d7_twin_sigma2(E, twins, t_list, layer, name, U, live=None):
    """D7 の σ²_arm（§3.4）: 読み出しごと・層ごと・**bin ごと**に、双子の同じ読み出しの対 (LRtw_k − LR) を
    **全 seed・全 k でプールして** §3.2 と同じ式で出す。双子か LR の bin にユニットが無いタスクを含む (seed, k) は
    その bin の平均から除く。層全体（bin に分けない）の σ² も同じ手順で出す（FLAT のガードの m_all 用）。
    ``live``（``twin_liveness`` の結果）を渡すと、窓の最初のタスクまでに生きていない (双子, seed) を除く（追補 2-2）。
    返り値 ({b: σ²_b or None}, σ²_all or None, {b: 対の数})。"""
    pairs_b = {b: [] for b in range(len(BINS))}
    pairs_all = []
    for s in E.seeds:
        bs_lr, _, wh_lr = bin_series(E.readout['LR'][s], t_list, layer, name, U)
        for tw in twins:
            if live is not None and not live.get((tw, s), {}).get('live'):
                continue
            bs_tw, _, wh_tw = bin_series(E.readout[tw][s], t_list, layer, name, U)
            for b in pairs_b:
                d = bs_tw[b] - bs_lr[b]
                if np.all(np.isfinite(d)):
                    try:
                        pairs_b[b].append((float(d.mean()), se_window(d)))
                    except DegenerateSeries:
                        pass
            d = wh_tw - wh_lr
            if np.all(np.isfinite(d)):
                try:
                    pairs_all.append((float(d.mean()), se_window(d)))
                except DegenerateSeries:
                    pass
    return ({b: sigma2_arm_from_twins(p) for b, p in pairs_b.items()}, sigma2_arm_from_twins(pairs_all),
            {b: len(p) for b, p in pairs_b.items()})


def judge_d7(E, env, W, R, n_twins):
    big, small = FAMILY['H']
    arms = ('LR', 'ELU1', big, small)
    win = W.pm_late if is_pm(env) else W.rl_win
    t_list = _tasks(win)
    if E.arm_status(arms) is not None or any(s not in E.readout.get(a, {}) for a in arms for s in E.seeds):
        for name in D7_READOUTS:
            for l in (0, 1):
                R.put(env, 'H', l, f'D7_{name}', E.arm_status(arms) or 'NOT_DETERMINED_MISSING_READOUT')
        return
    U = E.readout['LR'][E.seeds[0]]['occ_start'].shape[2]
    T_have = min(E.readout[a][s]['occ_start'].shape[0] for a in arms for s in E.seeds)
    if T_have < win[1]:
        for name in D7_READOUTS:
            for l in (0, 1):
                R.put(env, 'H', l, f'D7_{name}', 'NOT_DETERMINED_SHORT_WINDOW')
        return
    twins = [f'LRtw{k}' for k in range(n_twins) if all(E.status.get(f'LRtw{k}', {}).get(s) == 'COMPLETE' for s in E.seeds)
             and all(s in E.readout.get(f'LRtw{k}', {}) for s in E.seeds)]
    twins_complete = len(twins) == n_twins
    live = twin_liveness(E, n_twins, W.pm_base[0] if is_pm(env) else W.rl_win[0])     # 追補 2-2: 死んだ双子を除く
    n_live = sum(1 for tw in twins for s in E.seeds if live[(tw, s)]['live'])
    for name, (which, flat_ok) in D7_READOUTS.items():
        for l in (0, 1):
            cs, ms, ms_all, notes = {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}, []
            ok = True
            enough = twins_complete and n_live >= TWIN_MIN_LIVE_PAIRS
            s2_b, s2_all, n_pairs = d7_twin_sigma2(E, twins, t_list, l, name, U, live) if enough else \
                ({b: None for b in range(len(BINS))}, None, {b: 0 for b in range(len(BINS))})
            m_se_only = not enough or s2_all is None
            for s in E.seeds:
                ro = {a: E.readout[a][s] for a in arms}
                ser, ncommon, whole, wb = stratified_series(ro, arms, t_list, l, name, U)
                if ncommon == 0:
                    ok = False
                    notes.append(f's{s}: no common bin')
                    break
                # 層別の系列 Σ_b ŵ_b R_b の軌道分散は Σ_b ŵ_b² σ²_b（bin ごとのずれを独立とみなす）。σ²_b の無い bin は 0。
                if any(s2_b[b] is None for b in wb):
                    m_se_only = True
                s2_eff = sum(wb[b] ** 2 * (s2_b[b] or 0.) for b in wb)
                c1 = (ser[big] - ser['LR'], whole[big] - whole['LR']) if which == 'near' else (ser[big] - ser['ELU1'], whole[big] - whole['ELU1'])
                c2 = (ser['ELU1'] - ser[small], whole['ELU1'] - whole[small]) if which == 'near' else (ser['LR'] - ser[small], whole['LR'] - whole[small])
                try:
                    for j, (d, dall) in ((1, c1), (2, c2)):
                        if not np.all(np.isfinite(d)):
                            raise DegenerateSeries('NaN in stratified series')
                        cs[j].append(float(d.mean()))
                        ms[j].append(m_of(se_window(d), {'a': 1, 'b': -1}, s2_eff))
                        ms_all[j].append(m_of(se_window(dall), {'a': 1, 'b': -1}, s2_all or 0.)
                                         if np.all(np.isfinite(dall)) else float('nan'))
                except DegenerateSeries as e:
                    ok = False
                    notes.append(f's{s}: {e}')
                    break
            if not ok:
                lab = 'NOT_TESTABLE_NO_COMMON_BIN' if any('no common bin' in n for n in notes) else 'NOT_DETERMINED_DEGENERATE_SERIES'
                R.put(env, 'H', l, f'D7_{name}', lab, '; '.join(notes))
                continue
            s1, s2_ = contrast_state(cs[1], ms[1]), contrast_state(cs[2], ms[2])
            guard = all(np.isfinite(ma) and 2 * m <= 3 * ma for m, ma in zip(ms[1] + ms[2], ms_all[1] + ms_all[2]))
            if not flat_ok:
                s1 = UNRES if s1 == EQ else s1
                s2_ = UNRES if s2_ == EQ else s2_
            if s1 == DIRP and s2_ == DIRP:
                lab = f'{name}_UP'
            elif s1 == DIRM and s2_ == DIRM:
                lab = f'{name}_DOWN'
            elif s1 == EQ and s2_ == EQ:
                lab = f'{name}_FLAT' if guard else 'NOT_TESTABLE_LOW_OCCUPANCY'
            elif s1 != UNRES and s2_ != UNRES:
                lab = f'{name}_MIXED'
            else:
                lab = f'{name}_UNRESOLVED'
            if m_se_only and not lab.startswith('NOT_'):
                lab += M_SE_ONLY
            R.put(env, 'H', l, f'D7_{name}', lab, f'c1={s1} c2={s2_} guard={guard} twins={len(twins)} live_pairs={n_live} '
                  f'sigma2_bin={s2_b} sigma2_all={s2_all} pairs_bin={n_pairs}')
            for j in (1, 2):
                for i, s in enumerate(E.seeds):
                    R.seed_rows.append(dict(env=env, endpoint='late', family='H', contrast=f'D7_{name}_c{j}_l{l}', seed=s,
                                            c=cs[j][i], se=float('nan'), m=ms[j][i], state=(s1 if j == 1 else s2_),
                                            note=f'm_all={ms_all[j][i]:.4g}'))


# ------------------------------------------------------------------ D8・D9
def judge_d8(states, R):
    """states[env] = dict(N=..., D=...)（H 家族の合わせた D3 の状態）。"""
    labs = {}
    for X in ('N', 'D'):
        dec = {}
        for env, st in states.items():
            s = st.get(X, 'UNRESOLVED')
            t = strip_dose(s)
            if t in ('HELPS', 'HARMS') or (t == 'INERT' and env != 'scr'):
                dec[env] = t
        labs[X] = dec
    consistent = all(len(labs[X]) >= 2 and len(set(labs[X].values())) == 1 for X in ('N', 'D'))
    dep = [X for X in ('N', 'D') if 'HELPS' in labs[X].values() and any(v in ('INERT', 'HARMS') for v in labs[X].values())]
    if dep:
        lab = 'ENV_DEPENDENT(' + ','.join(dep) + ')'
    elif consistent:
        lab = 'CROSS_ENV_CONSISTENT'
    else:
        lab = 'CROSS_ENV_NOT_DETERMINED'
    R.put('all', 'H', 'all', 'D8', lab, ' | '.join(f'{X}: ' + ','.join(f'{e}={v}' for e, v in labs[X].items()) for X in ('N', 'D')))
    return lab


def judge_d9(states_by_lr, R):
    """states_by_lr[lr] = dict(N=..., D=...)（箱 B H の D3 の状態）。段 2 が無ければ NOT_DETERMINED。"""
    if len(states_by_lr) < 3:
        R.put('pmnist', 'H', 'all', 'D9', 'LR_ROBUSTNESS_NOT_DETERMINED', f'lrs={sorted(states_by_lr)}')
        return
    dep = []
    robust = True
    for X in ('N', 'D'):
        vals = {lr: strip_dose(st.get(X, 'UNRESOLVED')) for lr, st in states_by_lr.items()}
        if any(v not in ('HELPS', 'HARMS', 'INERT') for v in vals.values()) or len(set(vals.values())) != 1:
            robust = False
        if 'HELPS' in vals.values():
            for lr, v in vals.items():
                if v in ('INERT', 'HARMS'):
                    dep.append(f'{X},{lr}')
    lab = 'PATTERN_LR_ROBUST' if robust else ('PATTERN_LR_DEPENDENT(' + ';'.join(dep) + ')' if dep else 'LR_ROBUSTNESS_NOT_DETERMINED')
    R.put('pmnist', 'H', 'all', 'D9', lab, f'lrs={sorted(states_by_lr)} ' +
          ' | '.join(f'{lr}: N={st.get("N")} D={st.get("D")}' for lr, st in sorted(states_by_lr.items())))


# ------------------------------------------------------------------ 環境ごとの通し（段 1 + 段 2）
def pm_arms(stage2=False):
    arms = list(FUNCS8) + list(AC.SPLIT_NAMES) + ['LR02'] + [f'LRtw{k}' for k in range(N_TWINS['pmnist'])]
    return arms


def rl_arms(stage2=False):
    arms = ['LR', 'ELU1', 'SMAXH', 'SMINH', 'LRtw0']
    if stage2:
        arms += ['VMIN', 'VMAX', 'SMAXS', 'SMINS', 'GN', 'FN']
    return arms


def _twin_rows(E, env, eps, n_twins, R, start):
    """σ²_arm（endpoint ごと）と SIGMA_ARM の行。どれかの endpoint で σ² が無ければ環境を (M_SE_ONLY) に登録する。
    追補 2-2（S18b）: 双子 × seed ごとの生存を ``TWIN_LIVE`` の行（``TWIN_LIVE`` / ``NOT_DETERMINED_TWIN_DEAD``）と
    seed_contrasts.csv の ``twin_live_task_{双子}``（c = 最初に食い違ったタスク、無ければ NaN）に書く。"""
    sigma2 = {}
    for ep in eps.values():
        s2, n, note = twin_sigma2(E, ep, n_twins, start)
        sigma2[ep.tag] = s2
        # 2-4 A8: 切り詰めの注記 ``clipped (…)`` は verdict.csv の note でも先頭に置く（件数・σ² はその後、残りの注記は最後）
        head, _, rest = note.partition(';') if note.startswith('clipped') else ('', '', note)
        R.put(env, 'twins', 'all', f'SIGMA_ARM_{ep.name}', 'SIGMA_TRAJ_OK' if s2 is not None else 'NOT_DETERMINED_TWIN(M_SE_ONLY)',
              ' '.join(x for x in (head, f'n_live_pairs={n} sigma2={s2}', rest) if x))
        if s2 is None:
            R.m_se_only.add(env)
    for (tw, s), lv in sorted((E.extra.get('twin_live') or twin_liveness(E, n_twins, start)).items()):
        R.put(env, 'twins', f'{tw}_s{s}', 'TWIN_LIVE', 'TWIN_LIVE' if lv['live'] else 'NOT_DETERMINED_TWIN_DEAD',
              f"twin_live_task={lv['twin_live_task']} window_start=t{lv['window_start']} n_compared={lv['n_compared']} {lv['why']}")
        R.seed_scalar(env, 'twin_live', 'twins', f'twin_live_task_{tw}',
                      {s: float('nan') if lv['twin_live_task'] is None else float(lv['twin_live_task'])},
                      state='TWIN_LIVE' if lv['live'] else 'NOT_DETERMINED_TWIN_DEAD')
    return sigma2


def g1_pm_row(E, W, R, env='pmnist'):
    """G1-PM（§6・§7）: 錨 LR・ELU1・SMAXH × seed の provenance の g1_pm が全部 pass で、件数が 17 × 地平線。"""
    g = E.extra.get('g1_pm', {})
    want = 17 * W.pm_tasks
    n_anchor = 3 * len(E.seeds)
    ok = len(g) == n_anchor and all(v['pass_'] and v.get('n_compared') == want for v in g.values())
    lab = ('G1_PM_OK' if ok else 'G1_PM_FAILED' if any(not v['pass_'] for v in g.values())
           else 'NOT_DETERMINED_G1_PM_INCOMPLETE')
    R.put(env, 'all', 'all', 'G1_PM', lab, f'anchors={len(g)}/{n_anchor} want n_compared={want} ' +
          ';'.join(f"{k}:{v['pass_']}/{v.get('n_compared')}" for k, v in sorted(g.items())))
    return lab


def run_pmnist(root, W, R, ctx, seeds=(0, 1, 2), passes=('1', '2'), strict=True):
    E = load_pmnist(root, list(seeds), pm_arms(), W, strict=strict)
    env = 'pmnist'
    eps = endpoints_pmnist(W)
    tags = ('L', 'A')
    sigma2 = _twin_rows(E, env, eps, N_TWINS['pmnist'], R, start=W.pm_base[0])
    g1_pm_row(E, W, R)
    for arm in pm_arms():
        for s in seeds:
            R.put(env, 'arm', f'{arm}_s{s}', 'STATUS', E.status.get(arm, {}).get(s, 'MISSING'))
    states = {}
    if '1' in passes:
        for fam in ('H', 'V', 'S'):
            states[fam] = judge_family(E, env, W, fam, eps, ctx, R, sigma2, tags)
            secondary_series(E, env, W, fam, R)
            judge_family(E, env, W, fam, endpoints_pmnist(W, issa=True), ctx, R, sigma2, tags, extra_tag='_ISSA')
            for it in ('D1', 'D2', 'D3_HEADING'):
                a, b = R.get(env, fam, 'all', it), R.get(env, fam, 'all', it + '_ISSA')
                if decided(a) and decided(b) and strip_dose(a) != strip_dose(b):
                    R.put(env, fam, 'all', f'WINDOW_DEPENDENT_{it}', 'WINDOW_DEPENDENT', f'main={a} issa={b}')
        judge_d5a(E, env, W, eps, ctx, R, sigma2, tags)
        for X in ('N', 'D'):
            d5 = R.get(env, 'H', 'all', f'D5A_{X}')
            if d5.endswith('VIA_FORWARD'):
                key = (env, 'H', 'all', 'HYPOTHESIS')
                R.labels[key] += '_NEAR_VIA_FORWARD' if X == 'N' else '_DEEP_VIA_FORWARD'
                for r in R.rows:
                    if (r['env'], r['family'], str(r['layer']), r['item']) == key:
                        r['label'] = R.labels[key]
        judge_d5b(E, env, W, eps, ctx, R, sigma2, tags)
    if '2' in passes:
        judge_d4(env, R, ctx)
        ladder, inp = load_gate_shape_rows(LADDER_COMMITTED, list(seeds), W)
        E.inputs += inp
        judge_d6(E, ladder, W, eps, R, sigma2, tags)
        judge_d7(E, env, W, R, N_TWINS['pmnist'])
    # 報告（ラベル無し）: E 恒等式・§3.3–3.4 の報告だけの量
    for arm in pm_arms():
        for s in seeds:
            ro = E.readout.get(arm, {}).get(s)
            if ro is None:
                continue
            Ei = ro['E_ident']
            for j in range(Ei.shape[0]):
                R.seed_rows.append(dict(env=env, endpoint='E_ident', family='all', contrast=f'E_ident_{arm}_row{j}', seed=s,
                                        c=float(Ei[j, 0]), se=float(Ei[j, 1]), m=float(Ei[j, 2]), state='REPORT',
                                        note='E, E_pred, E_pred_wrong'))
    report_only_pmnist(E, W, R)
    E.extra['sigma2'] = sigma2
    return E, states


STAGE2_PM_LRS = ('5e-4', '2e-3')          # launcher の stage2/pmnist_lr{lr}（§2.1 段 2・D9）


def run_pmnist_stage2(root, lr_tag, W, R, sigma2, seeds=(0, 1, 2), strict=True):
    """段 2（D9）: stage2/pmnist_lr{lr} の H 家族 4 腕を、段 1 と同じ規則で判定する。錨（ρ_min・用量）はその lr の
    箱 B H 自身（新しい DoseCtx）。段 2 には双子が無いので σ²_arm は段 1（lr 1e−3）の双子の値を使う（§3.2 の
    「LR の双子のばらつきを全腕に使う」を lr にも広げる読み・追補に記す）。返り値 H の合わせた N・D。"""
    env = f'pmnist_lr{lr_tag}'
    arms = ('LR', 'ELU1', 'SMAXH', 'SMINH')
    E = load_pmnist(root, list(seeds), arms, W, strict=strict, env=env)
    for arm in arms:
        for s in seeds:
            R.put(env, 'arm', f'{arm}_s{s}', 'STATUS', E.status.get(arm, {}).get(s, 'MISSING'))
    R.put(env, 'twins', 'all', 'SIGMA_ARM', 'FROM_STAGE1_TWINS' if all(v is not None for v in sigma2.values())
          else 'NOT_DETERMINED_TWIN(M_SE_ONLY)', f'sigma2={sigma2} (stage 2 has no twins; lr 1e-3 twins)')
    if any(v is None for v in sigma2.values()):
        R.m_se_only.add(env)
    st = judge_family(E, env, W, 'H', endpoints_pmnist(W), DoseCtx(), R, sigma2, ('L', 'A'))
    return E, st


PM_FLOOR_H_INF = {'LR': (float('-inf'), 0.), 'LR02': (float('-inf'), 0.), 'ELU1': (-1., 1.),
                  'SMAXH': (float('-inf'), AC.C_SMAXH), 'SMINH': (-AC.C_SMINH, AC.C_SMINH), 'VMIN': (float('-inf'), 0.),
                  'VMAX': (-1., 1.), 'SMAXS': (float('-inf'), 0.9), 'SMINS': (AC.C_SMINS, -AC.C_SMINS),
                  'GN': (float('-inf'), 0.), 'FN': (float('-inf'), AC.C_SMAXH), 'GD': (-1., float('-inf')),
                  'FD': (float('-inf'), float('inf'))}   # §1.2・§1.5 の表: (負側の床/漸近の定数項, h(−∞))


def report_only_pmnist(E, W, R, env='pmnist'):
    """§3.3・§3.4・§8 の報告だけの量（ラベル無し）を seed_contrasts.csv に書く。
    - 幅の指数 p_late: ln cnorm を ln t に回帰した t60–120 の傾き（elu_growth_0909 V1）。地平線が足りなければ出さない。
    - 前向き値の交絡表: late 窓の自腕の E[φ]・E[h]（fn_mom の自腕の関数を帯で和・層ごと）、床と h(−∞) の表の値。
      分離腕の自腕の φ_fwd・h は fn_mom の 8 関数に無いので、φ_fwd は表の関数（GN/GD は LR/ELU1、FN/FD は SMAXH）で出し、
      h は出さない。LR02・双子は LR 系（LR02 は fn_mom に無いので出さない）。
    - 占有と P[|z|<1]: late 窓・層ごと・ユニット平均の帯分率（B0–B3）と pabs1。
    - 実現した Ḡ: 層 1 は rows の gbar、層 2 は readout の gbar_l2（ユニット平均）の late 窓平均と、家族ごとの
      big − LR・big − ELU1・small − LR・small − ELU1。"""
    lo, hi = W.pm_late
    gbar = {}
    for arm in E.status:
        for s in E.seeds:
            if E.status[arm].get(s) != 'COMPLETE':
                continue
            rows = E.rows.get(arm, {}).get(s)
            ro = E.readout.get(arm, {}).get(s)
            if rows and len(rows) >= 120 and all('cnorm' in r for r in rows[59:120]):
                t = np.arange(60, 121, dtype=np.float64)
                c = np.array([float(r['cnorm']) for r in rows[59:120]])
                if np.all(c > 0):
                    A_ = np.stack([np.log(t), np.ones_like(t)], 1)
                    R.seed_scalar(env, 'report', arm, 'p_late_cnorm_t60_120',
                                  {s: float(np.linalg.lstsq(A_, np.log(c), rcond=None)[0][0])})
            if ro is None or ro['fn_mom'].shape[0] < hi:
                continue
            base = AC.twin_of(arm)[0]
            fwd = AC.SPLIT[base][0] if base in AC.SPLIT else base
            for l in (0, 1):
                fm = ro['fn_mom'][lo - 1:hi, l]
                if fwd in IX:
                    R.seed_scalar(env, 'report', arm, f'confound_Ephi_l{l + 1}', {s: float(fm[:, IX[fwd], :, 1].sum(-1).mean())})
                if base in IX:
                    R.seed_scalar(env, 'report', arm, f'confound_Eh_l{l + 1}', {s: float(fm[:, IX[base], :, 2].sum(-1).mean())})
                occ = ro['occ_end'][lo - 1:hi, l].astype(np.float64).mean((0, 1))
                for k in range(4):
                    R.seed_scalar(env, 'report', arm, f'occ_B{k}_l{l + 1}', {s: float(occ[k])})
                R.seed_scalar(env, 'report', arm, f'pabs1_l{l + 1}', {s: float(ro['pabs1'][lo - 1:hi, l].astype(np.float64).mean())})
            if base in PM_FLOOR_H_INF:
                fl, hinf = PM_FLOOR_H_INF[base]
                R.seed_scalar(env, 'report', arm, 'confound_floor_const', {s: fl})
                R.seed_scalar(env, 'report', arm, 'confound_h_minus_inf', {s: hinf})
            if rows and len(rows) >= hi and 'gbar' in rows[0]:
                g1 = float(np.mean([float(r['gbar']) for r in rows[lo - 1:hi]]))
                g2 = float(ro['gbar_l2'][lo - 1:hi].mean())
                gbar[(arm, s)] = (g1, g2)
                R.seed_scalar(env, 'report', arm, 'realized_Gbar_l1', {s: g1})
                R.seed_scalar(env, 'report', arm, 'realized_Gbar_l2', {s: g2})
    for fam, (big, small) in FAMILY.items():
        for a, b in ((big, 'LR'), (big, 'ELU1'), (small, 'LR'), (small, 'ELU1')):
            for l in (0, 1):
                vals = {s: gbar[(a, s)][l] - gbar[(b, s)][l] for s in E.seeds if (a, s) in gbar and (b, s) in gbar}
                if vals:
                    R.seed_scalar(env, 'report', fam, f'realized_Gbar_diff_{a}-{b}_l{l + 1}', vals)


def apply_m_se_only(R):
    """§3.2: σ_arm = 0 で計算した環境（双子が揃わない）の、m から作った全ラベルに (M_SE_ONLY) を付ける。
    下流の合わせ方・見出し・D8 がラベルを読み終えた後（書き出しの直前）に当てる。D8 は該当環境があれば付ける。"""
    prefixes = ('D_PAR', 'D0_', 'D1', 'D2', 'D3_', 'HYPOTHESIS', 'N_DEEP', 'D5A_', 'D5B_', 'D6', 'D7_', 'D8', 'D9',
                'WINDOW_DEPENDENT')
    for r in R.rows:
        hit = (r['env'] in R.m_se_only) or (r['env'] == 'all' and r['item'] == 'D8' and R.m_se_only)
        if not hit or not str(r['item']).startswith(prefixes):
            continue
        lab = str(r['label'])
        if lab.startswith(('NOT_DETERMINED', 'NOT_APPLICABLE')) or M_SE_ONLY in lab:
            continue
        r['label'] = lab + M_SE_ONLY
        R.labels[(r['env'], r['family'], str(r['layer']), r['item'])] = r['label']


def run_rlmnist(root, W, R, ctx, seeds=(0, 1, 2), passes=('1', '2'), stage2=False, strict=True):
    arms = rl_arms(stage2)
    E = load_rlmnist(root, list(seeds), arms, W, strict=strict)
    env = 'rlmnist'
    eps = endpoints_rlmnist(W)
    tags = ('PT', 'LOGIT')
    sigma2 = _twin_rows(E, env, eps, N_TWINS['rlmnist'], R, start=W.rl_win[0])
    if E.extra.get('g1_rl'):
        g = E.extra['g1_rl']
        R.put(env, 'all', 'all', 'G1_RL', 'G1_RL_OK' if all(v['pass_'] for v in g.values()) else 'G1_RL_FAILED',
              ';'.join(f"{k}:{v['pass_']}/{v.get('n_cells')}" for k, v in sorted(g.items())))
    else:
        R.put(env, 'all', 'all', 'G1_RL', 'NOT_APPLICABLE_CPU_RUN', 'G1-RL is a GPU check (smoke g1_smoke.json)')
    for arm in arms:
        for s in seeds:
            fl = E.extra['at_floor'].get(arm, {}).get(s)
            R.put(env, 'arm', f'{arm}_s{s}', 'STATUS', E.status.get(arm, {}).get(s, 'MISSING'),
                  f"AT_FLOOR={fl} F={E.extra['F'][s]:.5f} thr={E.extra['F'][s] + RL_FLOOR_Z * RL_FLOOR_SD:.5f}")
            R.put(env, 'arm', f'{arm}_s{s}', 'AT_FLOOR', 'AT_FLOOR' if fl else ('ABOVE_FLOOR' if fl is False else 'NOT_DETERMINED_SHORT_WINDOW'))
    fams = ['H'] + (['V', 'S'] if stage2 else [])
    states = {}
    if '1' in passes:
        for fam in fams:
            states[fam] = judge_family(E, env, W, fam, eps, ctx, R, sigma2, tags, floor=E.extra['at_floor'])
            secondary_series(E, env, W, fam, R)
            judge_family(E, env, W, fam, endpoints_rlmnist(W, early=True), ctx, R, sigma2, tags, extra_tag='_EARLY',
                         floor=E.extra['at_floor'])
        for fam in ('V', 'S'):
            if fam not in fams:
                for it in ('D1', 'D2', 'D3_N', 'D3_D', 'D3_I', 'D3_HEADING', 'HYPOTHESIS'):
                    R.put(env, fam, 'all', it, 'NOT_DETERMINED_STAGE2_NOT_RUN')
        if stage2 and all(a in E.status for a in ('GN', 'FN')):
            judge_d5a(E, env, W, eps, ctx, R, sigma2, tags)
        else:
            for X in ('N', 'D'):
                R.put(env, 'H', 'all', f'D5A_{X}', 'NOT_DETERMINED_STAGE2_NOT_RUN')
    if '2' in passes:
        if stage2:
            judge_d4(env, R, ctx)
            judge_d5b(E, env, W, eps, ctx, R, sigma2, tags)
        else:
            R.put(env, 'H+S', 'all', 'D4', 'NOT_DETERMINED_STAGE2_NOT_RUN')
            for X in ('N', 'D'):
                R.put(env, 'V', 'all', f'D5B_{X}', 'NOT_DETERMINED_STAGE2_NOT_RUN')
        judge_d7(E, env, W, R, N_TWINS['rlmnist'])
    return E, states


def run_scr(root, W, R, lr=0.01, lam_full_path=None, build_readout=True, save_readout=None):
    E = load_scr(root, W, lr=lr, lam_full_path=lam_full_path, build_readout=build_readout)
    st = judge_scr(E, W, R, key='Y')
    judge_scr(E, W, R, key='Y_1m', tag='WIN1M_')
    judge_scr(E, W, R, key='SUB', tag='SUB_')
    judge_scr(E, W, R, key='ZBAR', tag='ZBAR_')
    if save_readout and build_readout:
        Path(save_readout).mkdir(parents=True, exist_ok=True)
        for name, ro in E.readout.items():
            np.savez(Path(save_readout) / f'readout_{SCR_ARMS[name]}.npz', **ro)
    return E, st


# ------------------------------------------------------------------ 出力（§8）
def write_outputs(out, R, inputs, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'verdict.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['env', 'family', 'layer', 'item', 'label', 'note'])
        w.writeheader()
        for r in R.rows:
            w.writerow(r)
    cols = ['env', 'endpoint', 'family', 'contrast', 'seed', 'c', 'se', 'm', 'state', 'note']
    with (out / 'seed_contrasts.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in R.seed_rows:
            w.writerow({k: (f'{v:.10g}' if isinstance(v, float) else v) for k, v in r.items()})
    (out / 'summary.md').write_text(summary_md(R, extra or {}), encoding='utf-8')
    prov = dict(run_id=RUN_ID, report_sha256=_sha(Path(__file__)), core_sha256=_sha(Path(AC.__file__)),
                spec_sha256=_sha(SPEC) if SPEC.exists() else None,
                inputs={p: _sha(p) for p in sorted(set(inputs)) if Path(p).exists()},
                written=[str(out / f) for f in ('verdict.csv', 'seed_contrasts.csv', 'summary.md')],
                time=time.strftime('%Y-%m-%dT%H:%M:%S'), extra=extra or {},
                note='--analyze-only 相当: 走の provenance は書き直さない（§8）')
    (out / 'report_provenance.json').write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return out / 'verdict.csv'


def summary_md(R, extra):
    lines = [f'# {RUN_ID} — verdict', '', f"windows: {extra.get('windows', '')}", '']
    main_items = ('D_PAR', 'D1', 'D2', 'D3_N', 'D3_D', 'D3_I', 'D3_HEADING', 'HYPOTHESIS', 'N_DEEP', 'D5A_N', 'D5A_D',
                  'D5B_N', 'D5B_D', 'D4', 'D6', 'D7_NEAR_BAND_SINKING', 'D7_NEAR_BAND_STEP', 'D7_NEAR_BAND_ADAM',
                  'D7_DEEP_BAND_VELOCITY', 'D7_DEEP_BAND_RETURN', 'D7_DEEP_RELAX', 'D8', 'D9', 'LR_FALLBACK_TRIGGER')
    lines += ['| env | family | layer | item | label | note |', '|---|---|---|---|---|---|']
    for r in R.rows:
        if r['item'] in main_items or r['item'].startswith(('D6_', 'SUB_', 'ZBAR_', 'WIN1M_', 'WINDOW_DEPENDENT')):
            lines.append(f"| {r['env']} | {r['family']} | {r['layer']} | {r['item']} | `{r['label']}` | {str(r['note'])[:160]} |")
    lines += ['', '## D0 / status / twins', '', '| env | family | layer | item | label | note |', '|---|---|---|---|---|---|']
    for r in R.rows:
        if r['item'].startswith(('D0_', 'STATUS', 'AT_FLOOR', 'SIGMA_ARM', 'TWIN_LIVE', 'SCR_STATUS', 'LAM_FULL', 'D5A_N_GATE', 'D5A_D_GATE')):
            lines.append(f"| {r['env']} | {r['family']} | {r['layer']} | {r['item']} | `{r['label']}` | {str(r['note'])[:120]} |")
    return '\n'.join(lines) + '\n'


def _twin_live_json(E):
    """``E.extra['twin_live']`` を JSON にする（{'LRtw0_s1': {twin_live_task, n_compared, window_start, live, why}}）。"""
    return {f'{tw}_s{s}': v for (tw, s), v in sorted((E.extra.get('twin_live') or {}).items())}


def all_rows_have_values(R):
    """G0.5: verdict.csv の全行が値を持つ（空・None・nan・KeyError 文字列を含まない）。"""
    for r in R.rows:
        lab = str(r['label'])
        if lab in ('', 'None', 'nan') or 'KeyError' in lab or 'nan' in lab.lower().split('(')[0]:
            raise AssertionError(f'row without a value: {r}')
    return len(R.rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description='act_chimera_0913 の判定器（§4・§8）')
    ap.add_argument('--root', default=str(OUT), help='results/act_chimera_0913（スモークは results/_smoke_act_chimera_0913）')
    ap.add_argument('--out', default=None, help='verdict.csv 等の出力先（既定は --root）')
    ap.add_argument('--envs', default='pmnist,scr,rlmnist')
    ap.add_argument('--pass', dest='passes', default='both', choices=['1', '2', 'both'])
    ap.add_argument('--smoke', action='store_true', help='窓をスモークの地平に置く（G0.5）')
    ap.add_argument('--scr-lr0p005', action='store_true', help='SCR を退避枝（scr_lr0p005・lr 0.005・10M）から読む')
    ap.add_argument('--stage2', action='store_true', help='RL の V・S 家族と分離腕（段 2）を読む')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--no-scr-readout', action='store_true', help='SCR の readout（fn_mom 等）を作らない（軽い検査用）')
    args = ap.parse_args(argv)
    W = Windows.smoke() if args.smoke else (Windows.lr0p005() if args.scr_lr0p005 else Windows())
    passes = ('1', '2') if args.passes == 'both' else (args.passes,)
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    seeds = [int(s) for s in args.seeds.split(',')]
    R, ctx, inputs, states = Report(), DoseCtx(), [], {}
    envs = args.envs.split(',')
    t0 = time.time()
    lr_states = {}
    twin_live = {}                                            # 追補 2-2（S18b）: report_provenance.json の twin_live
    if 'pmnist' in envs:
        E, st = run_pmnist(root / 'pmnist', W, R, ctx, seeds, passes, strict=not args.smoke)
        twin_live['pmnist'] = _twin_live_json(E)
        inputs += E.inputs
        states['pmnist'] = st.get('H', {})
        lr_states[0.001] = states['pmnist']
        if '1' in passes:                                     # D9（段 2）: stage2/pmnist_lr{lr} があれば同じ規則で判定する
            for lr_tag in STAGE2_PM_LRS:
                d2 = root / 'stage2' / f'pmnist_lr{lr_tag}'
                if d2.exists():
                    E2, st2 = run_pmnist_stage2(d2, lr_tag, W, R, E.extra['sigma2'], seeds, strict=not args.smoke)
                    inputs += E2.inputs
                    lr_states[float(lr_tag)] = st2
    if 'scr' in envs:
        sroot = root / ('scr_lr0p005' if args.scr_lr0p005 else 'scr')
        E, st = run_scr(sroot, W, R, lr=0.005 if args.scr_lr0p005 else 0.01, build_readout=not args.no_scr_readout,
                        save_readout=sroot / 'readout')
        inputs += E.inputs
        states['scr'] = st.get('H', {})
    if 'rlmnist' in envs:
        combine_rl_csv(root / 'rlmnist', rl_arms(args.stage2), seeds, W)
        E, st = run_rlmnist(root / 'rlmnist', W, R, ctx, seeds, passes, stage2=args.stage2, strict=not args.smoke)
        twin_live['rlmnist'] = _twin_live_json(E)
        inputs += E.inputs
        states['rlmnist'] = st.get('H', {})
    if '1' in passes:
        judge_d8(states, R)
    if '2' in passes:
        judge_d9(lr_states if lr_states else {0.001: {}}, R)
    apply_m_se_only(R)
    n = all_rows_have_values(R)
    path = write_outputs(out, R, inputs, dict(windows=str(W), envs=envs, passes=passes, n_rows=n, wall=time.time() - t0,
                                              twin_live=twin_live))
    print(f'wrote {path} ({n} rows, {time.time() - t0:.1f}s)')
    return R


if __name__ == '__main__':
    main()
