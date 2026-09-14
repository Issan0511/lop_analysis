"""test_act_chimera_0913: 活性化レベルの S 検査（spec_act_chimera_0913 §6）と変異対照。

対象: S0（§1.1 の定数）, S1, S2, S3, S3b, S4, S5, S6, S7, S8, S12（合成状態）, S13, S13b,
S15, S17。RL（``act_chimera_rlmnist_0913``）の S9/S10/S14/S18/S20・S11・G1-RL の比較器・床（§4.0・
S16 (iii)）・読み出しのスキーマ（§8）はファイル末尾の「RL の検査」にある（tiny の学習を CPU で回す）。
箱 B・SCR のループの写し（S14 の残り・S16 の残り・S19）はそれぞれのランナー・判定器側の検査。

すべての検査に変異対照を付け、対照が**実際に検査を落とす**ことを assert する（空虚な S 検査を
通算 6 回作った履歴・§6）。許容値は演算回数と eps から導く（eps32 = 2^-23、eps64 = 2^-52）。

実行: cd <worktree> && ~/.local/bin/pytest src/test_act_chimera_0913.py -q   （/usr/bin/python3）
      /usr/bin/python3 -m src.test_act_chimera_0913   → 値・許容値・対照値の JSON を stdout に出す
      （静的検査 static_checks.json を書く launcher は ``collect()`` を import して使う）

§1.3 の参照式は、モジュールの表を参照せずにこのファイルへ**別に逐語で写した**（REF_*）。
S7 / S13b の bit 一致検査が module 自身との同語反復にならないため。
"""
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import act_chimera_0913 as AC
from src import gate_shape_0911 as GS
from src import elu_growth_0909 as EG
from src import nets as N

H = GS.H
LNA, ZV, A = AC.LNA, AC.ZV, AC.A
C_SMAXH, C_SMINH, C_SMINS = AC.C_SMAXH, AC.C_SMINH, AC.C_SMINS
F32, F64 = torch.float32, torch.float64
DTYPES = (F32, F64)
EPS = {F32: 2. ** -23, F64: 2. ** -52}
SEAMS = (0., LNA, ZV)
MAX_DPHI = 1.0      # 全腕で |φ′| <= 1（S3 の max|φ′|）
MAX_DDPHI = 1.0     # 全腕で |φ″| <= 1（S3 の max|φ″|）
ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'src' / 'act_chimera_0913.py'
FORBIDDEN_EG = ('make_act', 'train', 'run', 'hdefect', 'measure')       # S8
BASE9 = AC.FUNCS8 + ('LR02',)
# S5 の極端な z（§6）
Z5 = (1e4, -1e4, 1e3, -1e3, -800., 746., -746., -745., 200., -200., -104., -103., -90., -50.,
      1e-30, -1e-30, 0., 50.)
NAN_FROM = {F32: 200., F64: 746.}        # clamp を外した式が NaN を出す下限（S5 の対照）


# ------------------------------------------------------------------ §1.3 の参照式（逐語・test 側の写し）
def REF_phi_LR(z):
    return torch.where(z > 0, z, 0.1 * z)


def REF_dphi_LR(z):
    return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))


def REF_phi_ELU1(z):
    return torch.where(z > 0, z, torch.expm1(z.clamp(max=0.)))


def REF_dphi_ELU1(z):
    return torch.where(z > 0, torch.ones_like(z), torch.exp(z.clamp(max=0.)))


def REF_phi_SMAXH(z):
    zc = z.clamp(max=0.)
    return torch.where(z > 0, z, torch.where(z >= LNA, torch.expm1(zc), 0.1 * (z - LNA) + (0.1 - 1.)))


def REF_dphi_SMAXH(z):
    return torch.where(z > 0, torch.ones_like(z), torch.clamp(torch.exp(z.clamp(max=0.)), min=0.1))


def REF_phi_SMINH(z):
    return torch.where(z > 0, z, torch.where(z >= LNA, 0.1 * z, torch.exp(z.clamp(max=LNA)) - C_SMINH))


def REF_dphi_SMINH(z):
    return torch.where(z > 0, torch.ones_like(z), torch.clamp(torch.exp(z.clamp(max=0.)), max=0.1))


def REF_phi_VMIN(z):
    return torch.where(z > 0, z, torch.where(z >= ZV, torch.expm1(z.clamp(ZV, 0.)), 0.1 * z))


def REF_dphi_VMIN(z):
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z >= ZV, torch.exp(z.clamp(ZV, 0.)), torch.full_like(z, 0.1)))


def REF_phi_VMAX(z):
    return torch.where(z > 0, z, torch.where(z >= ZV, 0.1 * z, torch.expm1(z.clamp(max=ZV))))


def REF_dphi_VMAX(z):
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z >= ZV, torch.full_like(z, 0.1), torch.exp(z.clamp(max=ZV))))


def REF_phi_SMAXS(z):
    zc = z.clamp(max=0.)
    return torch.where(z > 0, z, 0.1 * zc + 0.9 * torch.expm1(zc))


def REF_dphi_SMAXS(z):
    return torch.where(z > 0, torch.ones_like(z), 0.1 + 0.9 * torch.exp(z.clamp(max=0.)))


def REF_phi_SMINS(z):
    return torch.where(z > 0, z, 0.1 * F.softplus(z.clamp(max=0.) - LNA) + C_SMINS)


def REF_dphi_SMINS(z):
    return torch.where(z > 0, torch.ones_like(z), 0.1 * torch.sigmoid(z.clamp(max=0.) - LNA))


# §1.3 の φ″ の表（逐語）
def REF_ddphi_LR(z):
    return torch.zeros_like(z)


def REF_ddphi_ELU1(z):
    return torch.where(z > 0, torch.zeros_like(z), torch.exp(z.clamp(max=0.)))


def REF_ddphi_SMAXH(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= LNA, torch.exp(z.clamp(max=0.)), torch.zeros_like(z)))


def REF_ddphi_SMINH(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= LNA, torch.zeros_like(z), torch.exp(z.clamp(max=LNA))))


def REF_ddphi_VMIN(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= ZV, torch.exp(z.clamp(ZV, 0.)), torch.zeros_like(z)))


def REF_ddphi_VMAX(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= ZV, torch.zeros_like(z), torch.exp(z.clamp(max=ZV))))


def REF_ddphi_SMAXS(z):
    return torch.where(z > 0, torch.zeros_like(z), 0.9 * torch.exp(z.clamp(max=0.)))


def REF_ddphi_SMINS(z):
    s = torch.sigmoid(z.clamp(max=0.) - LNA)
    return torch.where(z > 0, torch.zeros_like(z), 0.1 * s * (1. - s))


REF_PHI = {'LR': REF_phi_LR, 'ELU1': REF_phi_ELU1, 'SMAXH': REF_phi_SMAXH, 'SMINH': REF_phi_SMINH,
           'VMIN': REF_phi_VMIN, 'VMAX': REF_phi_VMAX, 'SMAXS': REF_phi_SMAXS, 'SMINS': REF_phi_SMINS}
REF_DPHI = {'LR': REF_dphi_LR, 'ELU1': REF_dphi_ELU1, 'SMAXH': REF_dphi_SMAXH, 'SMINH': REF_dphi_SMINH,
            'VMIN': REF_dphi_VMIN, 'VMAX': REF_dphi_VMAX, 'SMAXS': REF_dphi_SMAXS, 'SMINS': REF_dphi_SMINS}
REF_DDPHI = {'LR': REF_ddphi_LR, 'ELU1': REF_ddphi_ELU1, 'SMAXH': REF_ddphi_SMAXH, 'SMINH': REF_ddphi_SMINH,
             'VMIN': REF_ddphi_VMIN, 'VMAX': REF_ddphi_VMAX, 'SMAXS': REF_ddphi_SMAXS, 'SMINS': REF_ddphi_SMINS}


# ------------------------------------------------------------------ 変異対照の式
def MUT_dphi_SMAXH_011(z):          # S1: 深部傾き 0.11
    return torch.where(z > 0, torch.ones_like(z), torch.clamp(torch.exp(z.clamp(max=0.)), min=0.11))


def MUT_dphi_SMINS_sigz(z):         # S1: φ′ = 0.1σ(z)
    return torch.where(z > 0, torch.ones_like(z), 0.1 * torch.sigmoid(z.clamp(max=0.)))


def MUT_dphi_VMIN_at_zc(z):         # S1: 継ぎ目を z_c に
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z >= LNA, torch.exp(z.clamp(LNA, 0.)), torch.full_like(z, 0.1)))


def MUT_phi_SMAXH_ln011(z):         # S2/S3: 切片を ln 0.11 で作る（述語は z_c のまま）
    zc = z.clamp(max=0.)
    deep = 0.1 * (z - math.log(0.11)) + (0.1 - 1.)
    return torch.where(z > 0, z, torch.where(z >= LNA, torch.expm1(zc), deep))


def MUT_phi_SMINH_expm1(z):         # S2: 深部を expm1 で書く（ELU の深部）
    return torch.where(z > 0, z, torch.where(z >= LNA, 0.1 * z, torch.expm1(z.clamp(max=LNA))))


def MUT_phi_VMIN_at_zc(z):          # S2: 継ぎ目を z_c に
    return torch.where(z > 0, z, torch.where(z >= LNA, torch.expm1(z.clamp(LNA, 0.)), 0.1 * z))


def MUT_dphi_VMIN_gt(z):            # S3b: φ′ の述語を `>=` から `>` に
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z > ZV, torch.exp(z.clamp(ZV, 0.)), torch.full_like(z, 0.1)))


def MUT_phi_SMINH_C011(z):          # S6: C_SMINH を ln 0.11 で作る
    c = 0.1 - 0.1 * math.log(0.11)
    return torch.where(z > 0, z, torch.where(z >= LNA, 0.1 * z, torch.exp(z.clamp(max=LNA)) - c))


def MUT_ddphi_SMAXS_noscale(z):     # S15: 係数 0.9 を落とす
    return torch.where(z > 0, torch.zeros_like(z), torch.exp(z.clamp(max=0.)))


# S5: clamp を外した式（捨てる側の枝が backward で 0·inf = NaN を作る）
MUT5_UNCLAMPED = {
    'ELU1': lambda z: torch.where(z > 0, z, torch.expm1(z)),
    'SMINH': lambda z: torch.where(z > 0, z, torch.where(z >= LNA, 0.1 * z, torch.exp(z) - C_SMINH)),
    'SMINS': lambda z: torch.where(z > 0, z, 0.1 * torch.log(0.1 + torch.exp(z))),
    'VMAX': lambda z: torch.where(z > 0, z, torch.where(z >= ZV, 0.1 * z, torch.expm1(z))),
}


# ------------------------------------------------------------------ 道具
_GRID = {}


def grid(dt, lo=-30., hi=10., n=400_001):
    key = (dt, lo, hi, n)
    if key not in _GRID:
        _GRID[key] = torch.linspace(lo, hi, n, dtype=dt)
    return _GRID[key]


def ulp(Z, dt):
    t = torch.tensor(Z, dtype=dt)
    return float(torch.nextafter(t, torch.tensor(float('inf'), dtype=dt)) - t)


def away_from_seams(z, k=4):
    """継ぎ目 ±k ulp（その dtype の ulp）を除くマスク。"""
    keep = torch.ones_like(z, dtype=torch.bool)
    for Z in SEAMS:
        keep &= (z - torch.tensor(Z, dtype=z.dtype)).abs() > k * ulp(Z, z.dtype)
    return keep


def autograd_of(fn, z):
    zg = z.detach().clone().requires_grad_(True)
    out = fn(zg)
    if not out.requires_grad:                       # 定数（leaky の φ′ など）: 勾配は 0
        return torch.zeros_like(z)
    g, = torch.autograd.grad(out.sum(), zg, allow_unused=True)
    return torch.zeros_like(z) if g is None else g


def maxabs(a, b, mask=None):
    d = (a - b).abs()
    if mask is not None:
        d = d[mask]
    return float(d.max())


def bit_equal(a, b):
    return bool(torch.equal(a, b) and torch.equal(torch.signbit(a), torch.signbit(b)))


def s1_err(phi, dphi, dt):
    z = grid(dt)
    return maxabs(dphi(z), autograd_of(phi, z), away_from_seams(z))


def kind_ok(act):
    k = getattr(act, 'kind', None)
    return isinstance(k, str) and k != 'adaptive_snake'


def mlpl(act, alpha, cls=AC.ChimeraMLPL, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return cls(2, 8, 20, gen, 'cpu', act=act, act_alpha=alpha)


def state_bytes(net):
    return b''.join(t.detach().cpu().numpy().tobytes() for _, t in sorted(net.state_dict().items()))


def synthetic_state(n=512, U=100, seed=20260914):
    """S12 用の合成状態: ユニットごとの平均を [-15, 3]、sd を [0.5, 4] に散らし、4 帯すべてに質量を置く。"""
    g = torch.Generator().manual_seed(seed)
    mu = torch.linspace(-15., 3., U, dtype=F64)
    sd = torch.linspace(0.5, 4., U, dtype=F64)
    return mu + sd * torch.randn(n, U, generator=g, dtype=F64)


# ------------------------------------------------------------------ S0: §1.1 の定数（50 桁 mpmath）
def check_S0(rec):
    import mpmath as mp
    mp.mp.dps = 50
    a = mp.mpf('0.1')
    zc = mp.log(a)
    zv = -10 - mp.lambertw(-10 * mp.exp(-10), 0)
    exact = {'LNA': zc, 'ZV': zv, 'EXP_ZV': mp.exp(zv), 'C_SMAXH': mp.mpf('0.9') + a * zc,
             'C_SMINH': a - a * zc, 'C_SMINS': a * mp.log(mp.mpf(1) / 11)}
    spec = {'LNA': -2.3025850929940455, 'ZV': -9.999545794446535, 'EXP_ZV': 4.542055534648271e-05,
            'C_SMAXH': 0.6697414907005954, 'C_SMINH': 0.3302585092994046, 'C_SMINS': -0.23978952727983707}
    spec32 = {'LNA': -2.3025851249694824, 'ZV': -9.99954605102539}
    # 直接丸めた定数は 50 桁の値から 2 ulp 以内。EXP_ZV = exp(float64 の ZV) は ZV の丸め
    # （≤ 0.5 ulp(10) = 8.9e−16、exp を通ると相対 8.9e−16 = 4 ulp）+ exp 自身の 0.5 ulp。
    ulp_tol = {k: 2. for k in spec}
    ulp_tol['EXP_ZV'] = 0.5 * ulp(abs(ZV), F64) / EPS[F64] + 0.5
    out = {}
    for k, v in spec.items():
        got = getattr(AC, k)
        assert got == v, (k, got, v)                                    # §1.1 の float64 の桁
        err_ulp = abs(float(mp.mpf(got) - exact[k])) / ulp(abs(got), F64)
        assert err_ulp <= ulp_tol[k], (k, err_ulp, ulp_tol[k])
        out[k] = dict(value=got, ulp_from_mp=err_ulp, ulp_tol=ulp_tol[k])
    for k, v in spec32.items():
        assert float(np.float32(getattr(AC, k))) == v, k
    resid = float(mp.exp(mp.mpf(ZV)) - 1 - a * mp.mpf(ZV))
    assert abs(resid) <= 4e-16, resid                                   # 根の残差（float64 の丸めだけ）
    assert C_SMAXH + C_SMINH == 1.0
    assert abs(C_SMINS + 0.1 * math.log(11)) <= 2 * EPS[F64]
    # 対照: ZV を 8 ulp ずらすと残差が丸め水準（≤ 4e−16）を超える
    zv_bad = float(np.nextafter(np.nextafter(ZV, 0), 0)) + 6 * ulp(abs(ZV), F64)
    resid_bad = float(mp.exp(mp.mpf(zv_bad)) - 1 - a * mp.mpf(zv_bad))
    assert abs(resid_bad) > 4e-16, resid_bad
    rec['S0'] = dict(constants=out, zv_residual=resid, ctl_zv_8ulp_residual=resid_bad, tol=4e-16)


# ------------------------------------------------------------------ S1: 解析 φ′ と autograd
def check_S1(rec):
    vals, ctl = {}, {}
    for dt in DTYPES:
        tol = 8 * EPS[dt]
        for name in BASE9:                                              # 分離腕は S17
            e = s1_err(AC.PHI[name], AC.DPHI[name], dt)
            assert e <= tol, (name, dt, e, tol)
            vals[f'{name}/{dt}'] = e
    tol64 = 8 * EPS[F64]
    ctl['SMAXH_deep_011'] = s1_err(AC.PHI['SMAXH'], MUT_dphi_SMAXH_011, F64)
    ctl['SMINS_sigma_z'] = s1_err(AC.PHI['SMINS'], MUT_dphi_SMINS_sigz, F64)
    ctl['VMIN_seam_at_zc'] = s1_err(AC.PHI['VMIN'], MUT_dphi_VMIN_at_zc, F64)
    for k, v in ctl.items():
        assert v > tol64, (k, v)
    assert abs(ctl['SMAXH_deep_011'] - 0.0100) < 1e-3 and abs(ctl['SMINS_sigma_z'] - 0.052) < 2e-3 \
        and abs(ctl['VMIN_seam_at_zc'] - 0.100) < 1e-3                # §6 の対照値
    rec['S1'] = dict(values=vals, tol={str(dt): 8 * EPS[dt] for dt in DTYPES}, controls=ctl)


# ------------------------------------------------------------------ S2: φ と ∫₀ᶻ φ′（mpmath quad）
def _mp_dphi(name):
    import mpmath as mp
    a = mp.mpf('0.1')
    zc = mp.log(a)
    zv = -10 - mp.lambertw(-10 * mp.exp(-10), 0)
    pos = lambda z: mp.mpf(1)
    neg = {
        'LR': lambda z: a,
        'LR02': lambda z: mp.mpf('0.2'),
        'ELU1': lambda z: mp.exp(z),
        'SMAXH': lambda z: max(mp.exp(z), a),
        'SMINH': lambda z: min(mp.exp(z), a),
        'VMIN': lambda z: mp.exp(z) if z >= zv else a,
        'VMAX': lambda z: a if z >= zv else mp.exp(z),
        'SMAXS': lambda z: a + mp.mpf('0.9') * mp.exp(z),
        'SMINS': lambda z: a / (1 + mp.exp(zc - z)),
    }[name]
    return (lambda z: pos(z) if z > 0 else neg(z)), zc, zv


def _s2_err(name, phi):
    import mpmath as mp
    mp.mp.dps = 30
    f, zc, zv = _mp_dphi(name)
    pts = np.linspace(-40., 5., 21)
    err = 0.
    for z in pts:
        if z > 0:
            integ = mp.mpf(z)
        else:
            knots = [mp.mpf(z)] + [k for k in (zv, zc) if z < k < 0] + [mp.mpf(0)]
            integ = -mp.quad(f, knots)                                # ∫₀ᶻ = −∫_z^0、{z_v, z_c} で分割
        got = float(phi(torch.tensor([z], dtype=F64))[0])
        err = max(err, abs(got - float(integ)))
    return err


def check_S2(rec):
    tol = 8 * EPS[F64] * (1 + 0.1 * 40)
    vals = {name: _s2_err(name, AC.PHI[name]) for name in BASE9}
    for k, v in vals.items():
        assert v <= tol, (k, v, tol)
    ctl = {'SMAXH_ln011': _s2_err('SMAXH', MUT_phi_SMAXH_ln011),
           'SMINH_expm1': _s2_err('SMINH', MUT_phi_SMINH_expm1),
           'VMIN_seam_at_zc': _s2_err('VMIN', MUT_phi_VMIN_at_zc)}
    for k, v in ctl.items():
        assert v > tol, (k, v)
    assert abs(ctl['SMAXH_ln011'] - 0.0095) < 2e-4 and abs(ctl['SMINH_expm1'] - 0.67) < 5e-3 \
        and abs(ctl['VMIN_seam_at_zc'] - 0.58) < 5e-3
    rec['S2'] = dict(values=vals, tol=tol, controls=ctl)


# ------------------------------------------------------------------ S3: 継ぎ目の連続性（Z⁻/Z⁺）
def _jump(name, Z):
    """C⁰ の継ぎ目での φ′ の跳び J = φ′(Z⁺) − φ′(Z⁻)（§6 S3）。それ以外は 0。"""
    if Z == 0.:
        return {'LR': 0.9, 'LR02': 0.8, 'SMINH': 0.9, 'VMAX': 0.9, 'SMINS': 10. / 11.}.get(name, 0.)
    if Z == ZV:
        return {'VMIN': AC.EXP_ZV - 0.1, 'VMAX': 0.1 - AC.EXP_ZV}.get(name, 0.)
    return 0.


def _s3_one(phi, dphi, name, Z, dt):
    eps = EPS[dt]
    Zt = torch.tensor([Z], dtype=dt)
    Zm = torch.nextafter(Zt, torch.tensor([-float('inf')], dtype=dt))
    Zp = torch.nextafter(Zt, torch.tensor([float('inf')], dtype=dt))
    w = float(Zp - Zm)
    dphi_ = maxabs(phi(Zp), phi(Zm))
    ddphi_ = float(dphi(Zp) - dphi(Zm) - _jump(name, Z))
    b_phi = MAX_DPHI * w + 4 * eps * max(1., float(phi(Zt).abs()))
    b_dphi = MAX_DDPHI * w + 4 * eps
    return dphi_, b_phi, abs(ddphi_), b_dphi


def check_S3(rec):
    vals = {}
    for dt in DTYPES:
        for name in BASE9:
            for Z in SEAMS:
                dphi_, b_phi, ddphi_, b_dphi = _s3_one(AC.PHI[name], AC.DPHI[name], name, Z, dt)
                assert dphi_ <= b_phi, (name, Z, dt, dphi_, b_phi)
                assert ddphi_ <= b_dphi, (name, Z, dt, ddphi_, b_dphi)
                vals[f'{name}/{Z:.4g}/{dt}'] = dict(dphi=dphi_, bound_phi=b_phi, ddphi=ddphi_, bound_dphi=b_dphi)
    ctl = {}
    for dt in DTYPES:
        dphi_, b_phi, _, _ = _s3_one(MUT_phi_SMAXH_ln011, AC.DPHI['SMAXH'], 'SMAXH', LNA, dt)
        assert dphi_ > b_phi, (dt, dphi_, b_phi)
        ctl[f'SMAXH_ln011/{dt}'] = dict(dphi=dphi_, bound_phi=b_phi)
    rec['S3'] = dict(values=vals, controls=ctl)


# ------------------------------------------------------------------ S3b: 継ぎ目ちょうど
def _s3b_one(phi, dphi, Z, dt):
    """継ぎ目ちょうどの (|φ′ − autograd(φ)|, 8·eps·|φ′|（spec の相対項）, 1 ulp, Z⁻, Z⁺)。

    **spec からの逸脱（要追補）**: §6 S3b の許容値 8·eps·|φ′| + 1 ulp·max|φ″| は z_v で成り立たない。
    torch の expm1 の backward は grad·(result + 1) で（bit で確認: autograd(expm1) == expm1 + 1）、result ≈ −1 のとき
    result + 1 の丸めは |φ′| に比例せず絶対で eps/2（|result| ≈ 1 の ulp の半分）まで出る。z_v では φ′ = e^{z_v} = 4.5e−5
    なので 8·eps·|φ′| = 4.3e−11（f32）に対し実測 1.8e−9（ELU1・VMIN）。そこで絶対項を 1 つだけ足す:
        tol = 8·eps·|φ′| + 1 ulp·max|φ″| + eps
    （expm1 の backward の丸め 1 回分 ≤ eps/2 を eps で覆う）。z_c と 0 では spec の値に eps だけ足した値になる
    （f32 z_c SMAXH: spec 1.19e−7 → 2.4e−7、実測 2.24e−8）。対照（VMIN の `>`・差 0.0999）はどちらでも落ちる。
    """
    eps = EPS[dt]
    Zt = torch.tensor([Z], dtype=dt)
    ana = dphi(Zt)
    aut = autograd_of(phi, Zt)
    Zm = torch.nextafter(Zt, torch.tensor([-float('inf')], dtype=dt))
    Zp = torch.nextafter(Zt, torch.tensor([float('inf')], dtype=dt))
    a = float(ana.abs())
    return float((ana - aut).abs()), 8 * eps * a, float(Zp - Zt), Zm, Zp


def check_S3b(rec):
    vals = {}
    spec_fail = []
    for dt in DTYPES:
        for name in BASE9:
            for Z in SEAMS:
                d, spec_base, one_ulp, Zm, Zp = _s3b_one(AC.PHI[name], AC.DPHI[name], Z, dt)
                curv = max(float(AC.DDPHI[name](Zm).abs()), float(AC.DDPHI[name](Zp).abs()))
                tol_spec = spec_base + one_ulp * curv
                tol = tol_spec + EPS[dt]
                assert d <= tol, (name, Z, dt, d, tol)
                if d > tol_spec:
                    spec_fail.append(f'{name}/{Z:.4g}/{dt}')
                vals[f'{name}/{Z:.4g}/{dt}'] = dict(diff=d, tol=tol, tol_spec_relative=tol_spec)
    ctl = {}
    for dt in DTYPES:
        d, spec_base, one_ulp, Zm, Zp = _s3b_one(AC.PHI['VMIN'], MUT_dphi_VMIN_gt, ZV, dt)
        tol = spec_base + one_ulp * max(float(AC.DDPHI['VMIN'](Zm).abs()), float(AC.DDPHI['VMIN'](Zp).abs())) + EPS[dt]
        assert d > tol and abs(d - 0.0999) < 1e-3, (dt, d, tol)
        ctl[f'VMIN_gt/{dt}'] = dict(diff=d, tol=tol)
    rec['S3b'] = dict(values=vals, controls=ctl, tol_rule='8*eps*|dphi| + 1ulp*max|ddphi| + eps (one absolute eps: expm1 backward)',
                      spec_relative_rule_fails_at=spec_fail)


# ------------------------------------------------------------------ S4: h = zφ′ − φ（分離腕は z·φ′_bwd − φ_fwd）
def _s4_err(name, hfn):
    z = grid(F64)
    phi, dphi = AC.PHI[name](z), AC.DPHI[name](z)         # 分離腕では φ_fwd / φ′_bwd（表の定義）
    ref = torch.where(z > 0, torch.zeros_like(z), z * dphi - phi)
    tol = 8 * EPS[F64] * (1 + z.abs() * dphi.abs() + phi.abs())
    d = (hfn(z) - ref).abs()
    return float(d.max()), float((d / tol).max())


def check_S4(rec):
    vals = {}
    for name in AC.ACT_NAMES:
        m, ratio = _s4_err(name, lambda z, n=name: AC.h(n, z))
        assert ratio <= 1., (name, m, ratio)
        vals[name] = dict(maxabs=m, ratio_to_tol=ratio)
    ctl = {}
    m, ratio = _s4_err('SMAXH', lambda z: AC.h('ELU1', z))            # SMAXH の h に ELU1 の h
    assert ratio > 1. and abs(m - 0.33) < 5e-3, (m, ratio)
    ctl['SMAXH_h_of_ELU1'] = m
    m, ratio = _s4_err('GN', lambda z: torch.zeros_like(z))           # GN の h に z·φ′_fwd − φ_fwd ≡ 0
    assert ratio > 1. and abs(m - 0.28) < 5e-3, (m, ratio)
    ctl['GN_h_zero'] = m
    z1 = torch.tensor([-1.], dtype=F64)
    assert abs(float(AC.h('GN', z1)) + 0.2679) < 1e-4                 # §1.5 の表: h_GN(−1) = −0.2679
    try:
        AC.h('nope', z1)
        raise AssertionError('h table returned for an unknown arm')
    except KeyError:
        pass
    assert float(EG.hdefect(H.ARMS['LR'], z1)) == 0.                  # EG.hdefect の 0 返し（使わない罠の実証）
    rec['S4'] = dict(values=vals, controls=ctl)


# ------------------------------------------------------------------ S5: 極端な z で有限
def _s5_finite(name, dt, device='cpu'):
    """φ・autograd 勾配・φ′・h・φ″（DDPHI）と、SCR の ChimeraMLPL の act_fn/act_grad/act_curv が極端な z で有限で、
    dtype と device を保つこと（LR02 は SCR に無いので MLPL を除く）。"""
    z = torch.tensor(Z5, dtype=dt, device=device)
    act = AC.make_act(name)
    out = dict(phi=act.phi(z), grad=autograd_of(act.phi, z), dphi=act.dphi(z), h=AC.h(name, z), ddphi=AC.DDPHI[name](z))
    if name != 'LR02':
        act_name = {'LR': 'leaky_relu', 'ELU1': 'elu'}.get(name, name)
        net = mlpl(act_name, 1.0 if name == 'ELU1' else 0.1)
        a = net.act_fn(z)
        out.update(mlpl_fn=a, mlpl_grad=net.act_grad(z, a), mlpl_curv=net.act_curv(z))
    fin = {k: bool(torch.isfinite(v).all()) for k, v in out.items()}
    same = {k: (v.dtype == dt and v.device.type == torch.device(device).type) for k, v in out.items()}
    return {k: fin[k] and same[k] for k in out}


def check_S5(rec):
    vals = {}
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for dt in DTYPES:
            for name in AC.ACT_NAMES:
                ok = _s5_finite(name, dt, device)
                assert all(ok.values()), (name, dt, device, ok)
                vals[f'{name}/{dt}/{device}'] = ok
    ctl = {}
    for dt in DTYPES:
        for name, fn in MUT5_UNCLAMPED.items():
            z = torch.tensor([v for v in Z5 if v >= NAN_FROM[dt]], dtype=dt)
            g = autograd_of(fn, z)
            assert bool(torch.isnan(g).all()), (name, dt, g)          # clamp を外すと backward が NaN
            ctl[f'{name}_unclamped/{dt}'] = dict(z=z.tolist(), grad_nan=True)
    rec['S5'] = dict(values=vals, controls=ctl, nan_from={str(k): v for k, v in NAN_FROM.items()},
                     devices=devices if len(devices) > 1 else devices + ['cuda unavailable'])


# ------------------------------------------------------------------ S6: 要因計画の恒等式
def _s6_ratio(big, small, table, dt, lo=-40., hi=10., n=200_001):
    z = grid(dt, lo, hi, n)
    tol = 8 * EPS[dt] * (1 + z.abs())
    r = (table[big](z) + table[small](z) - table['ELU1'](z) - table['LR'](z)).abs()
    return float(r.max()), float((r / tol).max())


def check_S6(rec):
    vals, ctl = {}, {}
    for dt in DTYPES:
        for fam in ('H', 'V'):
            big, small = AC.FAMILY[fam]
            for q, table in (('phi', AC.PHI), ('dphi', AC.DPHI)):
                m, ratio = _s6_ratio(big, small, table, dt)
                assert ratio <= 1., (fam, q, dt, m, ratio)
                vals[f'{fam}/{q}/{dt}'] = dict(maxabs=m, ratio_to_tol=ratio)
        m, ratio = _s6_ratio('SMAXS', 'SMINS', AC.PHI, dt)            # S 家族は加法的でない（対照）
        assert ratio > 1. and abs(m - 0.140) < 1e-3, (dt, m)
        ctl[f'S_phi/{dt}'] = m
        m, ratio = _s6_ratio('SMAXS', 'SMINS', AC.DPHI, dt)
        assert ratio > 1. and abs(m - 0.047) < 1e-3, (dt, m)
        ctl[f'S_dphi/{dt}'] = m
    mut = dict(AC.PHI)
    mut['SMINH'] = MUT_phi_SMINH_C011
    m, ratio = _s6_ratio('SMAXH', 'SMINH', mut, F64)
    assert ratio > 1. and abs(m - 0.0095) < 2e-4, m
    ctl['SMINH_C_ln011'] = m
    rec['S6'] = dict(values=vals, controls=ctl)


# ------------------------------------------------------------------ S7: 型と bit 一致・kind
def _bits_match(act, ref_phi, ref_dphi, n=2_000_001):
    ok = True
    for dt in DTYPES:
        z = grid(dt, -30., 10., n)
        ok &= bit_equal(act.phi(z), ref_phi(z))
        ok &= bit_equal(act.dphi(z), ref_dphi(z))
        ok &= bit_equal(autograd_of(act.phi, z), autograd_of(ref_phi, z))
    return ok


def check_S7(rec):
    smaxh = AC.make_act('SMAXH')
    assert type(smaxh) is GS.ELUFloor and smaxh.param == 0.1 and smaxh is AC.make_act('SMAXH')
    assert _bits_match(smaxh, REF_phi_SMAXH, REF_dphi_SMAXH)
    elu1 = AC.make_act('ELU1')
    assert type(elu1) is EG.ELU and elu1.param == 1.0 and _bits_match(elu1, REF_phi_ELU1, REF_dphi_ELU1)
    lr = AC.make_act('LR')
    assert lr is H.ARMS['LR'] and _bits_match(lr, REF_phi_LR, REF_dphi_LR)
    lr02 = AC.make_act('LR02')
    assert type(lr02) is H.Activation and lr02.kind == 'leaky' and lr02.param == 0.2
    kinds = {}
    for name in AC.ACT_NAMES:
        act = AC.make_act(name)
        assert kind_ok(act), name
        kinds[name] = act.kind
    for name in AC.CHIMERA_NAMES[1:] + AC.SPLIT_NAMES:
        assert kinds[name] == 'chimera_' + name
    # 対照
    assert not _bits_match(AC.make_act('SMAXS'), REF_phi_SMAXH, REF_dphi_SMAXH, n=200_001)

    class NoKind(AC.SMINH):
        kind = None
    assert not kind_ok(NoKind()) and not kind_ok(H.AdaptiveSnake(.6, .01, 'cpu'))
    rec['S7'] = dict(kinds=kinds, smaxh_type=type(smaxh).__name__, n_points=2_000_001,
                     controls=dict(SMAXS_vs_ELUFloor_bits=False, NoKind=False))


# ------------------------------------------------------------------ S8: make_act の KeyError と EG の禁止参照
NEW_MODULES = tuple(ROOT / 'src' / f for f in ('act_chimera_0913.py', 'act_chimera_pmnist_0913.py',
                                                'act_chimera_rlmnist_0913.py', 'act_chimera_scr_0913.py',
                                                'act_chimera_report_0913.py', 'act_chimera_launch_0913.py'))


def scan_forbidden_refs(path, module='src.elu_growth_0909', names=FORBIDDEN_EG):
    """``elu_growth_0909`` の names を参照する箇所を返す（AST）。捕まえる形:
    - ``from src import elu_growth_0909 as X`` / ``import src.elu_growth_0909 as X`` の別名 X 経由の ``X.measure``
    - ``from src.elu_growth_0909 import measure``（名前の直接 import）
    - 連鎖した属性 ``GS.EG.measure`` / ``C.EG.hdefect``（``.EG.`` を経由する参照・値が別名に解決される参照）
    - ``src.elu_growth_0909.measure`` の完全修飾。"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    pkg, mod = module.rsplit('.', 1)
    aliases, hits = {mod, 'EG'}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == pkg:
                aliases.update(a.asname or a.name for a in node.names if a.name == mod)
            if node.module == module:
                hits.extend((node.lineno, a.name) for a in node.names if a.name in names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module and a.asname:
                    aliases.add(a.asname)

    def dotted(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return list(reversed(parts))
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            chain = dotted(node.value)
            if chain is None:
                continue
            if chain[-1] in aliases or '.'.join(chain).endswith(module) or 'EG' in chain:
                hits.append((node.lineno, '.'.join(chain + [node.attr])))
    return sorted(set(hits))


def check_S8(rec):
    for bad in ('nope', 'ELUF', 'LRtw0', 'ELU03', 'CELU1', 'elu', ''):
        try:
            AC.make_act(bad)
            raise AssertionError(f'make_act accepted {bad!r}')
        except KeyError:
            pass
    assert AC.twin_of('LRtw3') == ('LR', 3) and AC.twin_of('LR') == ('LR', None)
    hits = {p.name: scan_forbidden_refs(p) for p in NEW_MODULES}                # 6 つの新モジュールすべて
    assert all(v == [] for v in hits.values()), hits
    # 対照 1: EG.make_act は未知名を黙って leaky にする（罠の実証）
    assert EG.make_act('SMINH', 0.) is H.ARMS['LR']
    # 対照 2: EG.measure を 1 行呼ぶ一時ファイル・連鎖した GS.EG.measure / C.EG.hdefect・名前の直接 import を走査器が検出する
    ctl = {}
    with tempfile.TemporaryDirectory() as d:
        for tag, text, want in (
                ('alias', 'from src import elu_growth_0909 as EG\nrow = EG.measure(None)\n', [(2, 'EG.measure')]),
                ('chained', 'from src import gate_shape_0911 as GS\nrow = GS.EG.measure(None)\n', [(2, 'GS.EG.measure')]),
                ('chained_C', 'from src import width_sink_clamp_0909 as C\nh = C.EG.hdefect(1, 2)\n', [(2, 'C.EG.hdefect')]),
                ('from_import', 'from src.elu_growth_0909 import measure\n', [(1, 'measure')]),
                ('dotted', 'import src.elu_growth_0909\nsrc.elu_growth_0909.train()\n', [(2, 'src.elu_growth_0909.train')])):
            p = Path(d) / f'bad_{tag}.py'
            p.write_text(text)
            got = scan_forbidden_refs(p)
            assert got == want, (tag, got, want)
            ctl[tag] = got
    rec['S8'] = dict(forbidden_hits=hits, modules=[p.name for p in NEW_MODULES],
                     controls=dict(EG_make_act_SMINH_is_leaky=True, scanner_hits=ctl))


# ------------------------------------------------------------------ S12: Γ の恒等式と帯分解（合成状態）
def check_S12(rec):
    z = synthetic_state()
    n, U = z.shape
    fm = AC.fn_mom(z)
    assert fm.shape == (8, 4, 3) and fm.dtype == F64
    C = AC.family_contrasts(fm)
    tol_g = 8 * EPS[F64] * 4
    tol_b = n * EPS[F64]
    ix = {f: i for i, f in enumerate(AC.FUNCS8)}
    vals = {}
    for fam in ('H', 'V'):
        big, _ = AC.FAMILY[fam]
        gN = float(C[fam]['N'][:, 0].sum())
        direct = float((AC.DPHI[big](z) - AC.DPHI['LR'](z)).mean())
        gI = float(C[fam]['I'][:, 0].sum())
        assert abs(gN - direct) <= tol_g, (fam, gN, direct)
        assert abs(gI) <= tol_g, (fam, gI)
        vals[fam] = dict(Gamma_N=gN, Gamma_N_direct=direct, Gamma_I=gI)
    # 帯分解 Σ_k = 帯なしの平均（8 関数 × 3 量）
    worst = 0.
    for f in AC.FUNCS8:
        for q, v in enumerate((AC.DPHI[f](z), AC.PHI[f](z), AC.h(f, z))):
            worst = max(worst, abs(float(fm[ix[f], :, q].sum()) - float(v.mean())))
    assert worst <= tol_b, worst
    # Σ_k mob_k = mob（ユニット別）
    mob_k = AC.mob_band(AC.DPHI['SMINH'](z), z)
    mob = AC.DPHI['SMINH'](z).mean(0)
    worst_mob = float((mob_k.sum(1) - mob).abs().max())
    assert worst_mob <= tol_b, worst_mob
    occ = AC.occupancy(z)
    assert occ.shape == (U, 4) and float((occ.sum(1) - 1).abs().max()) <= tol_b
    mass = occ.mean(0)
    assert bool((mass > 0).all()), mass                             # 4 帯すべてに質量（対照が実証できる前提）
    # 対照 1: S 家族の Γ_I は 0 でない
    gI_S = float(C['S']['I'][:, 0].sum())
    assert gI_S > tol_g and gI_S > 0., gI_S
    # 対照 2: B2 を落とした和は B2 の質量 > 0 のとき落ちる
    assert float(mass[2]) > 0.
    drop = abs(float(fm[ix['SMAXH'], [0, 1, 3], 0].sum()) - float(AC.DPHI['SMAXH'](z).mean()))
    assert drop > tol_b, drop
    rec['S12'] = dict(values=vals, band_sum_maxdiff=worst, mob_sum_maxdiff=worst_mob, tol_gamma=tol_g,
                      tol_band=tol_b, band_mass=mass.tolist(),
                      controls=dict(Gamma_I_S=gI_S, drop_B2_diff=drop))


# ------------------------------------------------------------------ S13: SCR のクラス差し替え
def check_S13(rec):
    vals, ctl = {}, {}
    for act, alpha in (('leaky_relu', 0.1), ('elu', 1.0)):
        base = mlpl(act, alpha, cls=N.VecMLPL)
        b0 = state_bytes(base)
        base.__class__ = AC.ChimeraMLPL                               # setup_arm_chimera と同じ 1 行
        assert state_bytes(base) == b0 and base.act == act and base.act_alpha == alpha
        ref = mlpl(act, alpha, cls=N.VecMLPL)
        for dt in DTYPES:
            z = grid(dt, -30., 30., 200_001)
            for m in ('act_fn', 'act_grad', 'act_curv'):
                a = getattr(base, m)(z, base.act_fn(z)) if m == 'act_grad' else getattr(base, m)(z)
                b = getattr(ref, m)(z, ref.act_fn(z)) if m == 'act_grad' else getattr(ref, m)(z)
                assert bit_equal(a, b), (act, m, dt)
        vals[act] = dict(state_bytes_equal=True, bits_equal=True)

    class Mut(AC.ChimeraMLPL):
        pass
    mut = mlpl('leaky_relu', 0.1, cls=Mut)
    mut.act_alpha = float(np.nextafter(0.1, 1.))
    ref = mlpl('leaky_relu', 0.1, cls=N.VecMLPL)
    z = grid(F64, -30., 30., 200_001)
    ctl['slope_nextafter'] = bit_equal(mut.act_fn(z), ref.act_fn(z))
    assert ctl['slope_nextafter'] is False
    rec['S13'] = dict(values=vals, controls=ctl)


# ------------------------------------------------------------------ S13b: ChimeraMLPL と §1.3 の参照の bit 一致
def _mlpl_bits(net, ref_phi, ref_dphi, ref_ddphi):
    for dt in DTYPES:
        for z in (grid(dt, -30., 10., 400_001), torch.tensor(Z5, dtype=dt)):
            if not (bit_equal(net.act_fn(z), ref_phi(z)) and bit_equal(net.act_grad(z, net.act_fn(z)), ref_dphi(z))
                    and bit_equal(net.act_curv(z), ref_ddphi(z))):
                return False
    return True


def check_S13b(rec):
    vals = {}
    for name in AC.CHIMERA_NAMES:
        net = mlpl(name, 0.1)
        assert _mlpl_bits(net, REF_PHI[name], REF_DPHI[name], REF_DDPHI[name]), name
        vals[name] = True
    assert _mlpl_bits(mlpl('leaky_relu', 0.1), REF_phi_LR, REF_dphi_LR, REF_ddphi_LR)
    assert _mlpl_bits(mlpl('elu', 1.0), REF_phi_ELU1, REF_dphi_ELU1, REF_ddphi_ELU1)
    vals['leaky_relu'] = vals['elu'] = True
    for name in AC.CHIMERA_NAMES + AC.SPLIT_NAMES:                    # dial の規約（§2.4）
        for bad in (1.0, 0.11, 0.0):
            try:
                mlpl(name, bad)
                raise AssertionError((name, bad))
            except ValueError:
                pass
    try:
        mlpl('SMINH', 0.1).set_activation('SMINH', 0.1, 'activation_plus_alpha')
        raise AssertionError('grad form')
    except ValueError:
        pass

    class Mut(AC.ChimeraMLPL):
        def act_grad(self, pre, a):
            if self.act == 'SMINH':
                return torch.where(pre > 0, torch.ones_like(pre),
                                   torch.clamp(torch.exp(pre.clamp(max=0.)), max=0.11))
            return super().act_grad(pre, a)
    ctl = _mlpl_bits(mlpl('SMINH', 0.1, cls=Mut), REF_phi_SMINH, REF_dphi_SMINH, REF_ddphi_SMINH)
    assert ctl is False
    rec['S13b'] = dict(values=vals, controls=dict(SMINH_max_011=ctl))


# ------------------------------------------------------------------ S15: φ″ と act_grad の autograd
def _s15_err(net):
    z = grid(F64)
    aut = autograd_of(lambda zz: net.act_grad(zz, net.act_fn(zz)), z)
    return maxabs(net.act_curv(z), aut, away_from_seams(z))


def check_S15(rec):
    tol = 8 * EPS[F64]
    vals = {}
    for name in AC.CHIMERA_NAMES + AC.SPLIT_NAMES:
        e = _s15_err(mlpl(name, 0.1))
        assert e <= tol, (name, e)
        vals[name] = e
    for act, alpha in (('leaky_relu', 0.1), ('elu', 1.0)):
        e = _s15_err(mlpl(act, alpha))
        assert e <= tol, (act, e)
        vals[act] = e

    class Mut(AC.ChimeraMLPL):
        def act_curv(self, pre):
            return MUT_ddphi_SMAXS_noscale(pre) if self.act == 'SMAXS' else super().act_curv(pre)
    e = _s15_err(mlpl('SMAXS', 0.1, cls=Mut))
    assert e > tol and abs(e - 0.1) < 1e-3, e
    rec['S15'] = dict(values=vals, tol=tol, controls=dict(SMAXS_no_0p9=e))


# ------------------------------------------------------------------ S17: 分離腕
def _s17_grad_err(act, bwd, dt):
    z = grid(dt)
    g = torch.linspace(-2., 2., z.numel(), dtype=dt)
    zg = z.clone().requires_grad_(True)
    grad, = torch.autograd.grad(act.phi(zg), zg, grad_outputs=g)
    return maxabs(grad, g * bwd(z), away_from_seams(z))


def check_S17(rec):
    vals = {}
    for name, (fwd, bwd) in AC.SPLIT.items():
        act = AC.make_act(name)
        assert isinstance(act, AC.SplitArm) and act.kind == 'chimera_' + name
        for dt in DTYPES:
            z = grid(dt)
            assert bit_equal(act.phi(z), AC.PHI[fwd](z))               # 前向き = φ_fwd（bit）
            assert bit_equal(act.dphi(z), AC.DPHI[bwd](z))             # dphi = φ′_bwd（bit）
            e = _s17_grad_err(act, AC.DPHI[bwd], dt)
            assert e <= 8 * EPS[dt], (name, dt, e)
            vals[f'{name}/{dt}'] = e
        assert bit_equal(act.phi(grid(F64)), REF_PHI[fwd](grid(F64)))
        # h 表 = z·φ′_bwd − φ_fwd（S4 と同じ許容値）
        m, ratio = _s4_err(name, lambda z, n=name: AC.h(n, z))
        assert ratio <= 1., (name, m)
        # SCR: act_fn = φ_fwd・act_grad = φ′_bwd・act_curv = φ″_bwd（bit）
        assert _mlpl_bits(mlpl(name, 0.1), REF_PHI[fwd], REF_DPHI[bwd], REF_DDPHI[bwd]), name
    # 対照 1: 逆向きに φ′_fwd を使う（GN なら (z_c, 0) で e^z − 0.1・最大 0.9）
    gn = AC.make_act('GN')
    gn_fwdgrad = AC.SplitArm('GN')
    gn_fwdgrad._bwd = AC.DPHI[AC.SPLIT['GN'][0]]
    e = _s17_grad_err(gn_fwdgrad, AC.DPHI['SMAXH'], F64)
    assert e > 8 * EPS[F64] and abs(e - 0.9) < 0.05, e                  # 上流 g は z=0 で 1.0（§6: 最大 0.9）
    # 対照 2: dphi に φ′_fwd を返させる
    gn_dphi_fwd = AC.SplitArm('GN')
    gn_dphi_fwd.dphi = lambda z: AC.DPHI['LR'](z)
    assert not bit_equal(gn_dphi_fwd.dphi(grid(F64)), gn.dphi(grid(F64)))
    e2 = maxabs(gn_dphi_fwd.dphi(grid(F64)), gn.dphi(grid(F64)))
    assert abs(e2 - 0.9) < 1e-3, e2
    rec['S17'] = dict(values=vals, controls=dict(bwd_uses_fwd_slope=e, dphi_returns_fwd=e2))


# ================================================================== RL（act_chimera_rlmnist_0913）の検査
# S9-RL, S10-RL, S11, S14-RL, S18-RL, S20-RL、G1-RL の比較器、RL の床（§4.0・S16 (iii)）、読み出しのスキーマ（§8）。
# 学習は tiny（epochs = 1 → 75 step/タスク・2〜3 タスク・CPU）。MNIST は 1 回だけ読む（約 190 MB）。
import difflib
import hashlib
import shutil

from src import act_chimera_rlmnist_0913 as RLC

RL = RLC.RL
CPU = torch.device('cpu')
_RL_MNIST = {}

# S14-RL の登録ブロック（§2.3 R1–R3）。表現は `ast.get_source_segment(src, stmt, padded=True)` を本体の
# 文ごとに取り出して改行で結んだもの（1 行の文は行末コメントと字下げを持たず、複合文は範囲内の
# コメントと字下げを保つ）。宿主の行数 72 は実装前に宿主から数えて固定した（RL:120-195 の 15 文）。
RL_HOST_BODY_LINES = 72
RL_R1 = ('act = H.ARMS[arm]', 'act = act_obj')
RL_R2 = ['    if twin is not None:                              # R2（§2.3・追補 2-2）: 双子は b1[k] に +1e−6（S18）',
         '        AC.perturb_twin_b1(params, twin)',
         'ro.after_init(params)']
RL_R3 = ['        ro.end_task(t, params, adam, x, act, g_batch)  # R3（§2.3）: §3.3–3.4 の per-unit 読み出し（.cpu().double() で計算）']
RL_R2_AFTER = 'params = H.init_params(seed, device)'          # 挿入位置: この文の直後
RL_R3_AFTER = '        m = evaluate_rl(params, x, y, act)'      # 挿入位置: この行の直後（for 文の中）


def rl_mnist():
    if 'm' not in _RL_MNIST:
        H.setup('cpu')
        torch.set_num_threads(1)
        _RL_MNIST['m'] = RLC.load_mnist(CPU)
    return _RL_MNIST['m']


def rl_tiny(label, seed=0, n_tasks=2, epochs=1, ro_cls=RLC.RLReadout, debug=None, twin='auto', sha_excl=None):
    """写したループを tiny で回す。返り値 (rows, diverged, readout)。``sha_excl`` = {名前: (j, idx)} は S18b の記録。"""
    name, k, act = RLC.arm_activation(label)
    ro = ro_cls(act)
    if sha_excl:
        ro.sha_excl_elems = dict(sha_excl)
    rows, div = RLC.run_one(label, seed, RLC.LR_RATE, n_tasks, rl_mnist(), CPU, epochs=epochs, debug=debug,
                            act_obj=act, ro=ro, twin=(k if twin == 'auto' else twin))
    return rows, div, ro


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _body_lines(src_text, fn_name='run_one'):
    tree = ast.parse(src_text)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name]
    assert len(fn) == 1, fn_name
    segs = [ast.get_source_segment(src_text, s, padded=True) for s in fn[0].body]
    return '\n'.join(segs).split('\n')


# ------------------------------------------------------------------ S14-RL: ループの写し（AST 差分 = R1–R3）
def _s14_rl(new_text, expect_lines=RL_HOST_BODY_LINES):
    host = _body_lines((ROOT / 'src' / 'pmnist_rlmnist_0906.py').read_text(encoding='utf-8'))
    new = _body_lines(new_text)
    assert len(host) == expect_lines, (len(host), expect_lines)          # 行数の期待値（実装前に固定）
    ops = [op for op in difflib.SequenceMatcher(None, host, new, autojunk=False).get_opcodes()
           if op[0] != 'equal']
    got = [(tag, host[i1:i2], new[j1:j2]) for tag, i1, i2, j1, j2 in ops]
    want = [('replace', [RL_R1[0]], [RL_R1[1]]), ('insert', [], RL_R2), ('insert', [], RL_R3)]
    assert got == want, '\n'.join(f'{g[0]}: -{g[1]} +{g[2]}' for g in got)
    # 挿入位置
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'insert' and new[j1:j2] == RL_R2:
            assert host[i1 - 1] == RL_R2_AFTER, host[i1 - 1]
        if tag == 'insert' and new[j1:j2] == RL_R3:
            assert host[i1 - 1] == RL_R3_AFTER, host[i1 - 1]
    assert len(new) == expect_lines + len(RL_R2) + len(RL_R3), (len(new), expect_lines)
    return dict(host_lines=len(host), new_lines=len(new), ops=[g[0] for g in got])


def check_S14_rl(rec):
    src = (ROOT / 'src' / 'act_chimera_rlmnist_0913.py').read_text(encoding='utf-8')
    vals = _s14_rl(src)
    # 対照 1: 2 つの登録ブロックの間の 1 文字を変える
    needle = 'x = mnist.train_x[idx]'
    assert src.count(needle) == 1
    mutated = src.replace(needle, 'x = mnist.train_x[idx ]')
    try:
        _s14_rl(mutated)
        raise AssertionError('S14-RL passed a 1-char mutation between R2 and R3')
    except AssertionError as e:
        if 'passed a 1-char' in str(e):
            raise
    # 対照 2: 行数の期待値を 0 にする
    try:
        _s14_rl(src, expect_lines=0)
        raise AssertionError('S14-RL passed with expect_lines = 0')
    except AssertionError as e:
        if 'expect_lines = 0' in str(e):
            raise
    rec['S14_rl'] = dict(values=vals, registered=dict(R1=list(RL_R1), R2=RL_R2, R3=RL_R3),
                         controls=dict(one_char_between_blocks='fails', expect_lines_zero='fails'))


# ------------------------------------------------------------------ S11: MNIST と持ち込み 7 ファイルの sha256
def _s11_data(data_dir):
    got = {f: hashlib.sha256((Path(data_dir) / f).read_bytes()).hexdigest() for f in H.Mnist.FILES.values()}
    RLC.check_data_sha256(got)
    return got


def check_S11(rec):
    got = _s11_data(RLC.H.DATA_DIR)
    gs = RLC.MAIN_CLONE / 'results' / 'gate_shape_0911' / 'LR_s0_provenance.json'
    assert gs.exists(), gs
    gs_sha = json.loads(gs.read_text())['data_sha256']
    assert gs_sha == got, (gs_sha, got)                                  # gate_shape_0911 の provenance とも一致
    carry = RLC.check_carry7()
    # 対照: 1 byte を変えた scratch のコピーで不一致
    with tempfile.TemporaryDirectory() as d:
        for f in H.Mnist.FILES.values():
            shutil.copy(RLC.H.DATA_DIR / f, Path(d) / f)
        p = Path(d) / 'train-labels-idx1-ubyte.gz'
        b = bytearray(p.read_bytes())
        b[-1] ^= 0x01
        p.write_bytes(bytes(b))
        try:
            _s11_data(d)
            raise AssertionError('S11 passed on a 1-byte mutated copy')
        except AssertionError as e:
            if 'passed on' in str(e):
                raise
    rec['S11'] = dict(data_sha256=got, carry7=carry, controls=dict(one_byte_mutation='fails'))


# ------------------------------------------------------------------ S9-RL: 乱数の共有（init・subset・t1–3 のラベル・t1 の epoch 置換）
def _rl_streams(label, seed, twin='auto'):
    d = {}
    _, _, ro = rl_tiny(label, seed, n_tasks=3, epochs=1, debug=d, twin=twin)
    return dict(init=[tsha(p) for p in d['init']], subset=tsha(d['subset'][0]),
                labels=[tsha(y) for y in d['labels'][:3]], batch_t1=ro.rng_sha[0], batch_t3=ro.rng_sha[2])


def check_S9_rl(rec):
    ref = _rl_streams('LR', 0)
    same = {}
    for label in AC.ACT_NAMES:                                           # §6 S9: 全腕（8 関数・LR02・分離腕 4）
        if label == 'LR':
            continue
        got = _rl_streams(label, 0)
        assert got == ref, (label, got, ref)
        same[label] = True
    tw = _rl_streams('LRtw0', 0)
    for k in ('subset', 'labels', 'batch_t1', 'batch_t3'):
        assert tw[k] == ref[k], k                                        # 双子: 乱数列は全部一致
    # init は b1 だけ違う（追補 2-2: b1[0] に +1e−6。W1・W2・b2・W3・b3 は一致）
    assert tw['init'][0] == ref['init'][0] and tw['init'][1] != ref['init'][1] and tw['init'][2:] == ref['init'][2:]
    ctl = _rl_streams('LR', 1)                                            # 対照: seed + 1
    diff = {k: ctl[k] != ref[k] for k in ref}
    assert all(diff.values()), diff
    rec['S9_rl'] = dict(same_as_LR=same, twin_init_differs_only_b1=True, sha=ref,
                        controls=dict(seed_plus_1_all_differ=diff))


# ------------------------------------------------------------------ S18-RL: 双子は init の b1[k] だけ・+1e−6 の最近接の float32（追補 2-2）
def _twin_check(init_ref, init_tw, k=0):
    """独立な再計算: 差は params[1][k] の 1 要素だけで、その値が (float64(b1[k]) + 1e−6) を float32 に丸めた値
    （torch の float64 → float32 の変換）で、両隣の float32 の表現値が float64 の目標により近くないこと。"""
    n_diff, where = 0, None
    for i, (a, b) in enumerate(zip(init_ref, init_tw)):
        ne = (a != b)
        n = int(ne.sum())
        if n:
            n_diff += n
            where = (i, [tuple(int(v) for v in ix) for ix in ne.nonzero()])
    assert n_diff == 1, f'twin differs in {n_diff} elements (need exactly 1)'
    i, ix = where
    assert i == 1 and ix == [(k,)], where
    a, b = init_ref[1][k], init_tw[1][k]
    target = a.double() + 1e-6
    assert bool(target.float() == b), (float(a), float(b), float(target))
    lo = torch.nextafter(b, torch.tensor(-float('inf')))
    hi = torch.nextafter(b, torch.tensor(float('inf')))
    err = (b.double() - target).abs()
    assert bool((lo.double() - target).abs() >= err) and bool((hi.double() - target).abs() >= err)
    return dict(n_diff=n_diff, index=where, realised_delta=float(b.double() - a.double()), target_err=float(err))


def check_S18_rl(rec):
    d_lr, d_tw = {}, {}
    rl_tiny('LR', 0, n_tasks=1, epochs=1, debug=d_lr)
    rl_tiny('LRtw0', 0, n_tasks=1, epochs=1, debug=d_tw)
    vals = _twin_check(d_lr['init'], d_tw['init'])
    assert tsha(d_lr['subset'][0]) == tsha(d_tw['subset'][0]) and tsha(d_lr['labels'][0]) == tsha(d_tw['labels'][0])
    rep_ = AC.twin_init_report(d_lr['init'], d_tw['init'], 0)                          # ランナーの在走 S18 と同じ関数
    assert rep_['pass_'] and rep_['realised_delta'] == vals['realised_delta'] and rep_['n_ulps'] >= 1, rep_
    # 対照 1: 摂動 0 の「双子」（twin=None で回した LRtw0）は「差が 0 要素」で落ちる
    d_zero = {}
    rl_tiny('LRtw0', 0, n_tasks=1, epochs=1, debug=d_zero, twin=None)
    try:
        _twin_check(d_lr['init'], d_zero['init'])
        raise AssertionError('S18 passed a zero-perturbation twin')
    except AssertionError as e:
        assert 'differs in 0 elements' in str(e), str(e)
    assert AC.twin_init_report(d_lr['init'], d_zero['init'], 0)['pass_'] is False
    # 対照 2: 旧来の双子（W1[0, 0] を nextafter）は位置で落ちる
    old = [q.clone() for q in d_lr['init']]
    old[0][0, 0] = torch.nextafter(old[0][0, 0], torch.tensor(float('inf')))
    try:
        _twin_check(d_lr['init'], old)
        raise AssertionError('S18 passed the old W1[0,0] 1-ulp twin')
    except AssertionError as e:
        assert 'passed the old' not in str(e), str(e)
    assert AC.twin_init_report(d_lr['init'], old, 0)['pass_'] is False
    # 対照 3: +2e−6 の双子は値で落ちる
    two = [q.clone() for q in d_lr['init']]
    AC.perturb_twin_b1(two, 0, 2e-6)
    assert AC.twin_init_report(d_lr['init'], two, 0)['pass_'] is False
    # 在走の S18（run_job が provenance に書く）: 双子の走そのものの init と、摂動 0 の対照
    with tempfile.TemporaryDirectory() as d:
        RLC.run_job('LRtw0', 0, device=CPU, outdir=Path(d), n_tasks=1, epochs=1, mnist=rl_mnist())
        prov = json.loads((Path(d) / 'LRtw0' / 's0' / 'provenance.json').read_text())
    s18 = prov['checks']['s18']
    assert prov['checks_passed'] and s18['pass_'] and s18['ctl_zero_perturbation']['fails_as_required'], s18
    assert s18['where'] == [[1, [0]]] and s18['realised_delta'] == vals['realised_delta'], s18
    rec['S18_rl'] = dict(values=dict(vals, n_ulps=rep_['n_ulps'], in_run=dict(realised_delta=s18['realised_delta'],
                                                                              n_ulps=s18['n_ulps'])),
                         controls=dict(zero_perturbation='fails: differs in 0 elements',
                                       old_W1_00_nextafter='fails: position', plus_2e_6='fails: value',
                                       in_run_zero_perturbation=s18['ctl_zero_perturbation']))


# ------------------------------------------------------------------ S18b-RL: 双子の生存（摂動した要素を除いた sha256・追補 2-2）
class _DeadPixelTwinReadout(RLC.RLReadout):
    """変異対照: 「双子」の摂動を死んだ画素の重み W1[0, 0] に +1e−6 で置く（MNIST の画素 0 は訓練 60,000 枚すべてで 0・
    RL は入力を置換しない）。摂動は R2 の後・W(0) の保存の前に入れる。"""

    def after_init(self, params):
        with torch.no_grad():
            params[0][0, 0] = torch.tensor(float(np.float32(float(params[0][0, 0]) + 1e-6)))
        super().after_init(params)


def check_S18b_rl(rec):
    ex_b1 = {'0': (AC.TWIN_PARAM, (0,))}
    ex_w1 = {'w1_00': (0, (0, 0))}
    _, _, ro_lr = rl_tiny('LR', 0, n_tasks=1, epochs=1, sha_excl={**ex_b1, **ex_w1})
    _, _, ro_tw = rl_tiny('LRtw0', 0, n_tasks=1, epochs=1, sha_excl=ex_b1)
    live = AC.twin_live_task(ro_lr.sha_excl['0'], ro_tw.sha_excl['0'])
    assert live == dict(twin_live_task=1, n_compared=1), live                    # b1[0] +1e−6 の双子は t1 で生きている
    assert ro_lr.rec['param_sha'][0] == AC.params_sha256(ro_lr.last_params)      # exclude=None は RL の params_sha256 と同じ
    # 変異対照: 死んだ画素の重み W1[0, 0] に +1e−6 を置いた「双子」は、1 タスク後に生きていないと判定される
    _, _, ro_dead = rl_tiny('LR', 0, n_tasks=1, epochs=1, ro_cls=_DeadPixelTwinReadout, sha_excl=ex_w1)
    dead = AC.twin_live_task(ro_lr.sha_excl['w1_00'], ro_dead.sha_excl['w1_00'])
    assert dead == dict(twin_live_task=None, n_compared=1), dead
    assert ro_dead.rec['param_sha'][0] != ro_lr.rec['param_sha'][0]              # 全 params の sha256 は変わる（旧対照の空虚さ）
    assert float(ro_dead.last_params[0][0, 0]) != float(ro_lr.last_params[0][0, 0])
    # 判定器の側: 死んだ双子が窓の最初のタスクまでに生きていないこと（twin_liveness）
    E = RP.EnvData('rlmnist', [0])
    same_run = RP.run_identity(dict(machine=dict(device='cpu'), git_hash='fixture'))                # 同じ device・コードの対
    E.extra['sha_excl_ident'] = {('LR', 0): same_run, ('LRtw0', 0): same_run}
    E.extra['sha_excl'] = {('LR', 0): {'0': ro_lr.sha_excl['w1_00']}, ('LRtw0', 0): {'0': ro_dead.sha_excl['w1_00']}}
    lv_dead = RP.twin_liveness(E, 1, start=1)[('LRtw0', 0)]
    E.extra['sha_excl'] = {('LR', 0): {'0': ro_lr.sha_excl['0']}, ('LRtw0', 0): {'0': ro_tw.sha_excl['0']}}
    lv_live = RP.twin_liveness(E, 1, start=1)[('LRtw0', 0)]
    assert lv_dead['live'] is False and lv_live['live'] is True, (lv_dead, lv_live)
    # 在走の記録（run_job の provenance の checks.s18b）: LR は k ∈ TWIN_KS、双子は自分の k
    with tempfile.TemporaryDirectory() as d:
        for label in ('LR', 'LRtw0'):
            RLC.run_job(label, 0, device=CPU, outdir=Path(d), n_tasks=1, epochs=1, mnist=rl_mnist())
        full = {label: json.loads((Path(d) / label / 's0' / 'provenance.json').read_text()) for label in ('LR', 'LRtw0')}
    pv = {label: full[label]['checks']['s18b'] for label in ('LR', 'LRtw0')}
    assert pv['LR']['ks'] == list(RLC.TWIN_KS) == [0] and pv['LRtw0']['ks'] == [0], pv
    # 同じ device・コードで回った LR と双子の走の同一性は等しい（pid・時刻など走ごとに変わるものを含まない）。対照: device を
    # cuda に書き換えた双子の provenance は、sha の列が同じでも判定器で生きていない
    id_lr, id_tw = RP.run_identity(full['LR']), RP.run_identity(full['LRtw0'])
    assert id_lr == id_tw and id_lr['device'] == 'cpu' and id_lr['git_hash'] and id_lr['code_sha256'], (id_lr, id_tw)
    Ep = RP.EnvData('rlmnist', [0])
    RP._keep_sha_excl(Ep, 'LR', 0, full['LR'])
    RP._keep_sha_excl(Ep, 'LRtw0', 0, full['LRtw0'])
    lv_prov = RP.twin_liveness(Ep, 1, start=1)[('LRtw0', 0)]
    gpu_tw = dict(full['LRtw0'], machine=dict(full['LRtw0']['machine'], device='cuda'))
    RP._keep_sha_excl(Ep, 'LRtw0', 0, gpu_tw)
    lv_gpu = RP.twin_liveness(Ep, 1, start=1)[('LRtw0', 0)]
    assert lv_prov['live'] is True and lv_gpu['live'] is False and 'mismatch: device' in lv_gpu['why'], (lv_prov, lv_gpu)
    assert pv['LR']['sha_excl_b1']['0'] == ro_lr.sha_excl['0'] and pv['LRtw0']['sha_excl_b1']['0'] == ro_tw.sha_excl['0']
    rec['S18b_rl'] = dict(values=dict(twin_b1_0=live, run_identity_equal=True),
                          controls=dict(dead_pixel_W1_00=dead, report_dead=lv_dead, twin_on_other_device=lv_gpu['why'][:80]))


# ------------------------------------------------------------------ S10-RL: 測定が学習を変えない（2 タスク後の params が maxabs 0.0）
class _NoReadout(RLC.RLReadout):
    """何も測らない読み出し（after_init と last_params だけ）。"""

    def end_task(self, t, params, adam, x, act, g_batch):
        self.last_params = [p.detach().cpu().clone() for p in params]


class _MutWriteReadout(RLC.RLReadout):
    """対照 (i): 測定の後に W1 += 1e−9。"""

    def end_task(self, t, params, adam, x, act, g_batch):
        super().end_task(t, params, adam, x, act, g_batch)
        with torch.no_grad():
            params[0] += 1e-9


class _MutRngReadout(RLC.RLReadout):
    """対照 (ii): 読み出しの中で g_batch を 1 回消費する。"""

    def end_task(self, t, params, adam, x, act, g_batch):
        super().end_task(t, params, adam, x, act, g_batch)
        torch.randperm(2, generator=g_batch)


def _params_maxabs(a, b):
    return max(float((p - q).abs().max()) for p, q in zip(a, b))


def check_S10_rl(rec):
    base = rl_tiny('SMAXH', 0, n_tasks=2, epochs=1)[2].last_params
    none = rl_tiny('SMAXH', 0, n_tasks=2, epochs=1, ro_cls=_NoReadout)[2].last_params
    d0 = _params_maxabs(base, none)
    assert d0 == 0.0, d0
    d1 = _params_maxabs(base, rl_tiny('SMAXH', 0, n_tasks=2, epochs=1, ro_cls=_MutWriteReadout)[2].last_params)
    d2 = _params_maxabs(base, rl_tiny('SMAXH', 0, n_tasks=2, epochs=1, ro_cls=_MutRngReadout)[2].last_params)
    assert d1 > 0. and d2 > 0., (d1, d2)
    rec['S10_rl'] = dict(maxabs_after_2_tasks=d0, controls=dict(W1_plus_1e9=d1, randperm_on_g_batch=d2))


# ------------------------------------------------------------------ 読み出しのスキーマ（§8）と恒等式
def check_readout_schema_rl(rec):
    rows, div, ro = rl_tiny('GD', 0, n_tasks=2, epochs=1)
    arr = ro.arrays()
    T = RLC.validate_readout(arr, 2)
    with tempfile.TemporaryDirectory() as d:
        shapes = ro.save(Path(d) / 'readout_s0.npz')
        back = RLC.load_readout(Path(d) / 'readout_s0.npz', 2)
    assert set(back) == set(RLC.READOUT_KEYS)
    tol = RL.N_IMAGES * EPS[F64]
    occ_sum = np.abs(arr['occ_end'].astype(np.float64).sum(-1) - 1).max()
    assert occ_sum <= 4 * EPS[F32] * 4, occ_sum                                    # 帯分率の和 = 1（float32 保存）
    mob_l2 = arr['mob_band'][:, 1].astype(np.float64).sum(-1)                       # Σ_k mob_k = mob（層 2 は gbar_l2）
    assert np.abs(mob_l2 - arr['gbar_l2']).max() <= 4 * EPS[F32] * 4, np.abs(mob_l2 - arr['gbar_l2']).max()
    assert np.array_equal(arr['zbar_start'][1], arr['zbar_end'][0])                # t = 2 の開始 = t = 1 の終端（bit）
    assert np.array_equal(arr['depth_vel'][:, 0], arr['zbar_end'][:, 0] - arr['zbar_start'][:, 0])
    assert arr['param_sha'][0] != arr['param_sha'][1] and all(len(s) == 64 for s in arr['param_sha'])
    assert (arr['step_num'] > 0).all() and (arr['wt_norm'] > 0).all()
    assert np.isfinite(arr['adam_ratio']).all() and (arr['adam_epsfrac'] >= 0).all()
    # fn_mom の帯和 = 帯なしの平均: LR の φ′ は 0.1 + 0.9·1(z>0) なので Σ_k E[φ′_LR·1(B_k)] = 0.1 + 0.9·P(B0)
    for t in range(T):
        for l in range(2):
            lhs = arr['fn_mom'][t, l, 0, :, 0].sum()
            rhs = 0.1 + 0.9 * arr['occ_end'][t, l, :, 0].astype(np.float64).mean()
            assert abs(lhs - rhs) <= 4 * EPS[F32], (t, l, lhs, rhs)
    # 分離腕 GD: dphi = φ′_bwd = SMAXH の φ′ ≥ 0.1 なので mob_band の和 ≥ 0.1（前向き ELU1 の φ′ なら 0 に近づける）
    assert (arr['mob_band'].sum(-1) >= 0.1 - 4 * EPS[F32]).all()
    # 対照: キーを 1 つ落とした配列でスキーマ検査が落ちる（G0.5 の対照と同型）
    bad = dict(arr)
    del bad['gbar_l2']
    try:
        RLC.validate_readout(bad)
        raise AssertionError('schema check passed with a missing key')
    except AssertionError as e:
        if 'passed with' in str(e):
            raise
    bad = dict(arr)
    bad['pabs1'] = bad['pabs1'].astype(np.float64)
    try:
        RLC.validate_readout(bad)
        raise AssertionError('schema check passed with a wrong dtype')
    except AssertionError as e:
        if 'passed with' in str(e):
            raise
    rec['readout_schema_rl'] = dict(T=T, shapes=shapes, occ_sum_err=float(occ_sum),
                                    controls=dict(missing_key='fails', wrong_dtype='fails'))


# ------------------------------------------------------------------ S20-RL: 打ち切りの経路（INCOMPLETE・divergence.json・部分行 1 行）
class _InjectInfReadout(RLC.RLReadout):
    """task 1 の学習の後に W1[0, 0] = inf を注入する（§6 S20）。"""

    def end_task(self, t, params, adam, x, act, g_batch):
        super().end_task(t, params, adam, x, act, g_batch)
        if t == 1:
            with torch.no_grad():
                params[0][0, 0] = float('inf')


class _InjectInfAtInit(RLC.RLReadout):
    """初期値の直後（task 1 の前）に W1[0, 0] = inf を注入する: task 1 で発散して 0 行の INCOMPLETE。"""

    def after_init(self, params):
        super().after_init(params)
        with torch.no_grad():
            params[0][0, 0] = float('inf')


def check_S20_rl(rec):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        s = RLC.run_job('LR', 0, device=CPU, outdir=out / 'inj', n_tasks=2, epochs=1, mnist=rl_mnist(),
                        ro_cls=_InjectInfReadout)
        run = out / 'inj' / 'LR' / 's0'
        assert s['status'] == 'INCOMPLETE' and s['n_rows'] == 1, s
        dj = json.loads((run / 'divergence.json').read_text())
        assert dj['first_nonfinite_task'] == 2 and dj['n_complete_tasks'] == 1 and dj['status'] == 'INCOMPLETE', dj
        rows = RLC.read_csv_rows(run / 'per_task.csv')
        assert len(rows) == 1 and int(rows[0]['task']) == 1 and rows[0]['arm'] == 'LR', rows
        assert RLC.load_readout(run / 'readout_s0.npz', 1)['param_sha'].shape == (1,)
        prov = json.loads((run / 'provenance.json').read_text())
        assert prov['status'] == 'INCOMPLETE' and prov['divergence']['task'] == 2 and prov['n_rows'] == 1
        m_inc = RLC.merge_arm(out / 'inj', 'LR', n_tasks=2, seeds=(0,))
        assert m_inc['n_rows'] == 1 and m_inc['per_seed'][0]['status'] == 'INCOMPLETE', m_inc
        # 対照: 注入しない同じ走が COMPLETE で divergence.json を書かない
        c = RLC.run_job('LR', 0, device=CPU, outdir=out / 'ctl', n_tasks=2, epochs=1, mnist=rl_mnist())
        crun = out / 'ctl' / 'LR' / 's0'
        assert c['status'] == 'COMPLETE' and c['n_rows'] == 2 and not (crun / 'divergence.json').exists(), c
        assert len(RLC.read_csv_rows(crun / 'per_task.csv')) == 2
        assert set(RLC.read_csv_rows(crun / 'per_task.csv')[0]) == set(RLC.read_csv_rows(RLC.REF_DIR / 'LR' / 'per_task.csv')[0])
        m_ok = RLC.merge_arm(out / 'ctl', 'LR', n_tasks=2, seeds=(0,))
        assert m_ok['n_rows'] == 2 and m_ok['per_seed'][0]['status'] == 'COMPLETE'
        try:                                                                       # 行数の照合が効くこと
            RLC.merge_arm(out / 'ctl', 'LR', n_tasks=3, seeds=(0,))
            raise AssertionError('merge passed with the wrong row count')
        except AssertionError as e:
            if 'merge passed' in str(e):
                raise
        assert 'g1_rl' not in prov['checks']                                    # epochs ≠ 400 の走には G1-RL を当てない
        assert RLC.read_csv_rows(crun / 'per_task.csv') and (crun / 'per_task.csv').read_text().splitlines()[0] == RLC.csv_header()
        # task 1 で発散（行 0）: 見出しだけの csv と divergence.json（first_nonfinite_task = 1）を書き、merge が 0 行で通る
        z = RLC.run_job('LR', 1, device=CPU, outdir=out / 'inj0', n_tasks=2, epochs=1, mnist=rl_mnist(),
                        ro_cls=_InjectInfAtInit)
        zrun = out / 'inj0' / 'LR' / 's1'
        assert z['status'] == 'INCOMPLETE' and z['n_rows'] == 0, z
        assert json.loads((zrun / 'divergence.json').read_text())['first_nonfinite_task'] == 1
        assert (zrun / 'per_task.csv').read_text().splitlines() == [RLC.csv_header()]
        m0 = RLC.merge_arm(out / 'inj0', 'LR', n_tasks=2, seeds=(1,))
        assert m0['n_rows'] == 0 and m0['per_seed'][1]['expect'] == 0, m0
        # merge の対照 2: 取り違えた csv（seed 列が違う）は行数が合っても落ちる
        shutil.copytree(out / 'ctl' / 'LR' / 's0', out / 'ctl' / 'LR' / 's1')
        try:
            RLC.merge_arm(out / 'ctl', 'LR', n_tasks=2, seeds=(0, 1))
            raise AssertionError('merge passed a csv copied into the wrong seed dir')
        except AssertionError as e:
            if 'merge passed' in str(e):
                raise
    rec['S20_rl'] = dict(status='INCOMPLETE', first_nonfinite_task=dj['first_nonfinite_task'], n_rows=1,
                         task1_divergence=dict(n_rows=0, merge_rows=m0['n_rows']),
                         controls=dict(no_injection='COMPLETE, 2 rows, no divergence.json',
                                       merge_wrong_count='fails', merge_wrong_seed='fails'))


# ------------------------------------------------------------------ G1-RL の比較器（`.10g` の文字列一致・件数ガード）
def check_G1_rl_comparator(rec):
    ref = RLC.REF_DIR / 'LR' / 'per_task.csv'
    full = RLC.g1_rl_compare(ref, ref, seeds=RLC.G1_SEEDS)
    assert full['n_cells'] == RLC.G1_FULL_CELLS == 3000 and full['n_mismatch'] == 0, full
    smoke = RLC.g1_rl_compare(ref, ref, seeds=(0,), tasks=(1, 2))
    assert smoke['n_cells'] == RLC.G1_SMOKE_CELLS == 40 and smoke['n_mismatch'] == 0, smoke
    with tempfile.TemporaryDirectory() as d:
        lines = ref.read_text().splitlines()
        hdr = lines[0].split(',')
        i_online, i_memo = hdr.index('online_acc'), hdr.index('memo_acc')
        # 対照 1: 1 セルの最後の桁を変える
        row = lines[1].split(',')
        row[i_online] = row[i_online][:-1] + ('0' if row[i_online][-1] != '0' else '1')
        p1 = Path(d) / 'a.csv'
        p1.write_text('\n'.join([lines[0], ','.join(row)] + lines[2:]) + '\n')
        c1 = RLC.g1_rl_compare(p1, ref, seeds=(0,), tasks=(1, 2))
        assert c1['n_cells'] == 40 and c1['n_mismatch'] == 1 and c1['mismatches'][0]['col'] == 'online_acc', c1
        # 対照 2: `1` を `1.0` に（同じ数値でも不一致）
        row = lines[1].split(',')
        assert row[i_memo] == '1', row[i_memo]
        row[i_memo] = '1.0'
        p2 = Path(d) / 'b.csv'
        p2.write_text('\n'.join([lines[0], ','.join(row)] + lines[2:]) + '\n')
        c2 = RLC.g1_rl_compare(p2, ref, seeds=(0,), tasks=(1,))
        assert c2['n_mismatch'] == 1 and c2['mismatches'][0]['col'] == 'memo_acc', c2
        # 対照 3: 打ち切り（task 2 の行が無い）は allow_missing で 20 セルの不一致に数える（比較が縮んで通らない）
        p3 = Path(d) / 'c.csv'
        p3.write_text('\n'.join([lines[0], lines[1]]) + '\n')
        c3 = RLC.g1_rl_compare(p3, ref, seeds=(0,), tasks=(1, 2), allow_missing=True)
        assert c3['n_cells'] == 40 and c3['n_missing_rows'] == 1 and c3['n_mismatch'] == 20 and not c3['pass_'], c3
    try:                                                                  # G1-RL は GPU の検査: CPU では呼べない
        RLC.g1_smoke(CPU)
        raise AssertionError('g1_smoke accepted a CPU device')
    except ValueError:
        pass
    rec['G1_rl_comparator'] = dict(full_cells=full['n_cells'], smoke_cells=smoke['n_cells'],
                                   controls=dict(last_digit=c1['n_mismatch'], one_vs_one_point_zero=c2['n_mismatch'],
                                                 truncated_run_missing_rows=c3['n_mismatch'], g1_smoke_on_cpu='ValueError'))


# ------------------------------------------------------------------ G1-RL の変異対照の生死（追補 2-1・CPU の tiny で）
def _g1_cells(rows):
    return [f'{r[c]:.10g}' if isinstance(r[c], float) else str(r[c]) for r in rows for c in RLC.G1_COLS]


def check_G1_rl_control_live(rec):
    """G1-RL の変異対照（``PerturbInitReadout``）は b1[0] += 1e−3 で、同じ device の LR と 1 タスクで 1 セル以上が
    `.10g` の文字列で食い違うこと。対照の対照: 旧来の W1[0, 0] += 1e−3 は 20 セルすべて一致する（空虚だった実証）。"""
    base = _g1_cells(rl_tiny('LR', 0, n_tasks=1, epochs=1)[0])
    ctl = _g1_cells(rl_tiny('LR', 0, n_tasks=1, epochs=1, ro_cls=RLC.PerturbInitReadout)[0])
    n_mis = sum(a != b for a, b in zip(base, ctl))
    assert len(base) == len(ctl) == len(RLC.G1_COLS) and n_mis >= 1, n_mis

    class _OldW1(RLC.PerturbInitReadout):
        def __init__(self, act):
            super().__init__(act, param=0, index=(0, 0))

    old = _g1_cells(rl_tiny('LR', 0, n_tasks=1, epochs=1, ro_cls=_OldW1)[0])
    n_old = sum(a != b for a, b in zip(base, old))
    assert n_old == 0, n_old
    # 生きた対照は走ごとの G1-RL の gating にしない（gating にすると GPU の g1_smoke で対照の run_job が AssertionError で
    # 落ち、g1_smoke.json が書かれない）。述語: 登録の RLReadout の LR s0 だけが GPU で gating、対照の読み出しは記録だけ
    proto = ('LR', None, 0, RLC.LR_RATE, RLC.EPOCHS)
    gate = dict(plain_cuda=RLC.g1_gating(*proto, RLC.RLReadout, 'cuda'),
                control_cuda=RLC.g1_gating(*proto, RLC.PerturbInitReadout, 'cuda'),
                plain_cpu=RLC.g1_gating(*proto, RLC.RLReadout, 'cpu'),
                twin_cuda=RLC.g1_gating('LR', 0, 0, RLC.LR_RATE, RLC.EPOCHS, RLC.RLReadout, 'cuda'),
                epochs1_cuda=RLC.g1_gating('LR', None, 0, RLC.LR_RATE, 1, RLC.RLReadout, 'cuda'))
    assert gate['plain_cuda'] == RLC.G1_GATING, gate
    assert gate['control_cuda'] not in (None, RLC.G1_GATING) and gate['plain_cpu'] not in (None, RLC.G1_GATING), gate
    assert gate['twin_cuda'] is None and gate['epochs1_cuda'] is None, gate
    # 端から端（CPU・1 epoch）: EPOCHS を 1 に、g1_gating の device を cuda に差し替え、g1_smoke の対照と同じ run_job を回す。
    # 対照の走は食い違っても COMPLETE で返る。変異対照: 同じ条件の登録の読み出し（= GPU の G1 走の扱い）は 1 epoch の行が
    # 0906 と食い違って G1_RL で落ちる（run_job は書いてから落とす）
    orig_gating, orig_epochs = RLC.g1_gating, RLC.EPOCHS
    RLC.EPOCHS = 1
    RLC.g1_gating = lambda *a: orig_gating(*a[:-1], 'cuda')
    try:
        with tempfile.TemporaryDirectory() as d:
            s_ctl = RLC.run_job('LR', 0, device=CPU, outdir=Path(d) / 'ctl', n_tasks=1, epochs=1, mnist=rl_mnist(),
                                ro_cls=RLC.PerturbInitReadout)
            assert s_ctl['checks_passed'] and s_ctl['g1']['n_mismatch'] >= 1 and s_ctl['g1']['gating'] != RLC.G1_GATING, s_ctl['g1']
            try:
                RLC.run_job('LR', 0, device=CPU, outdir=Path(d) / 'plain', n_tasks=1, epochs=1, mnist=rl_mnist())
                raise AssertionError('gating G1-RL run with mismatching cells did not fail')
            except AssertionError as e:
                if 'did not fail' in str(e) or 'G1_RL' not in str(e):
                    raise
            pv_plain = json.loads((Path(d) / 'plain' / 'LR' / 's0' / 'provenance.json').read_text())
            assert pv_plain['failed_checks'] == ['G1_RL'] and pv_plain['checks']['g1_rl']['gating'] == RLC.G1_GATING
    finally:
        RLC.g1_gating, RLC.EPOCHS = orig_gating, orig_epochs
    rec['G1_rl_control_live'] = dict(b1_0_plus_1e3_mismatch=n_mis, n_cells=len(base), gating=gate,
                                     in_run_control=dict(gating=s_ctl['g1']['gating'], n_mismatch=s_ctl['g1']['n_mismatch']),
                                     controls=dict(old_W1_00_mismatch=n_old, registered_readout_same_run='fails G1_RL'))


# ------------------------------------------------------------------ launcher のゲート（2-4 A7・追補 2-3）: 学習なし・JSON だけ
def check_launcher_gates_addendum2(rec):
    """(a) static_checks.json の G0・G0.5 は ``pass_`` だけでなく、いまのコードの sha256（共有モジュール・ランナー・判定器・
    launcher・2 テストファイル）・HEAD・判定器の sha256 に対して記録されていること。(b) RL の RSS_peak は ``--device`` の
    実測だけ（CPU の本走に GPU の host RSS を混ぜない）。(c) ``rl_state`` は provenance の device が違う走を DONE にしない。"""
    import copy as _copy
    from src import act_chimera_launch_0913 as LA
    saved = (LA.CHECKS, LA.SMOKE, LA.SMOKE_LAUNCH, LA.git_head)
    head = 'a' * 40
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            LA.CHECKS, LA.SMOKE, LA.SMOKE_LAUNCH = d / 'checks', d / 'smoke', d / 'smoke' / 'launch'
            LA.git_head = lambda: head
            now = LA.code_sha256_now()
            want = {'act_chimera_0913.py', 'act_chimera_pmnist_0913.py', 'act_chimera_rlmnist_0913.py', 'act_chimera_scr_0913.py',
                    'act_chimera_report_0913.py', 'act_chimera_launch_0913.py', 'test_act_chimera_0913.py',
                    'test_act_chimera_report_0913.py'}
            assert set(now) == want, sorted(now)
            good = {'_G0': dict(pass_=True, code_sha256=now, git_head=head), '_G0_scr': dict(pass_=True),
                    'G05': dict(pass_=True, report_sha256=now['act_chimera_report_0913.py'])}
            LA._dump(LA.CHECKS / 'static_checks.json', good)
            ok, _ = LA._static_checks_pass()
            assert all(ok.values()), ok
            muts = {}
            for name, mut in (('old_runner', lambda r: r['_G0']['code_sha256'].update({'act_chimera_rlmnist_0913.py': '0' * 64})),
                              ('no_test_file', lambda r: r['_G0']['code_sha256'].pop('test_act_chimera_0913.py')),
                              ('old_head', lambda r: r['_G0'].update(git_head='b' * 40)),
                              ('old_report', lambda r: r['G05'].update(report_sha256='0' * 64)),
                              ('pass_only_record', lambda r: (r['_G0'].pop('code_sha256'), r['_G0'].pop('git_head'),
                                                              r['G05'].pop('report_sha256')))):
                bad = _copy.deepcopy(good)
                mut(bad)
                LA._dump(LA.CHECKS / 'static_checks.json', bad)
                muts[name] = sorted(k for k, v in LA._static_checks_pass()[0].items() if not v)
            assert muts == dict(old_runner=['G0_code_current'], no_test_file=['G0_code_current'], old_head=['G0_git_head_current'],
                                old_report=['G05_report_current'],
                                pass_only_record=['G05_report_current', 'G0_code_current', 'G0_git_head_current']), muts
            try:                                                   # 最後の記録（pass_ だけ）で段 1 はゲートで拒否される
                LA.require_gates('stage1', 'pmnist')
                raise AssertionError('require_gates accepted a pass_-only static_checks.json')
            except LA.StageRefused as e:
                assert 'G0_code_current' in str(e), e
            out['static'] = muts
            # (b) RSS: CPU と GPU の実測を混ぜて置き、device ごとの最大だけを取る
            for sub, dev, kib in (('timing_cpu/LR/s0', 'cpu', 1068452), ('timing_cuda/LR/s0', 'cuda', 1692992),
                                  ('g1/run1/LR/s0', 'cuda', 1664412)):
                LA._dump(LA.SMOKE / 'rlmnist' / sub / 'provenance.json', dict(peak_rss_kib=kib, machine=dict(device=dev)))
            LA._dump(LA.SMOKE_LAUNCH / 'smoke-rlmnist-g1_status.json', dict(jobs=[dict(name='rl_g1_smoke', peak_rss_kib=1785004)]))
            LA._dump(LA.SMOKE_LAUNCH / 'smoke-rlmnist-determinism_status.json',
                     dict(jobs=[dict(name='rl_determinism_cpu', peak_rss_kib=1067840),
                                dict(name='rl_determinism_cuda', peak_rss_kib=1813572)]))
            LA._dump(LA.SMOKE_LAUNCH / 'smoke-rlmnist-timing-cpu_status.json', dict(jobs=[dict(name='rl_LR_s0_cpu', peak_rss_kib=1068000)]))
            cpu_v, cpu_src = LA.rss_peak_from_smoke('rlmnist', device='cpu')
            gpu_v, _ = LA.rss_peak_from_smoke('rlmnist', device='cuda')
            assert cpu_v == 1068452 / 1024. ** 2 and cpu_src.startswith('CPU smoke measurement (追補 2-3)'), (cpu_v, cpu_src)
            assert gpu_v == 1813572 / 1024. ** 2, gpu_v                       # 対照: GPU を選べば GPU の host RSS
            assert LA.resolve_rss('rlmnist', None, device='cpu')[0] == cpu_v
            try:
                LA.rss_peak_from_smoke('rlmnist')
                raise AssertionError('rss_peak_from_smoke(rlmnist) accepted no device')
            except ValueError:
                pass
            out['rss_gib'] = dict(cpu=cpu_v, cuda=gpu_v)
            # (c) rl_state: GPU で回った本走の配置の走を、CPU の段 1 は DONE にせず拒否する
            LA._dump(d / 'rl' / 'LR' / 's0' / 'provenance.json', dict(n_tasks=LA.RL_TASKS, status='COMPLETE', checks_passed=True,
                                                                     git_hash=head, machine=dict(device='cuda')))
            try:
                LA.rl_state(d / 'rl', 'LR', 0, device='cpu')
                raise AssertionError('rl_state accepted a cuda run for a cpu stage')
            except LA.StageRefused as e:
                assert 'machine.device=cuda != --device cpu' in str(e), e
            assert LA.rl_state(d / 'rl', 'LR', 0, device='cuda') == 'DONE'  # 対照: 同じ device なら DONE
            out['rl_state'] = dict(cuda_run_for_cpu_stage='StageRefused', same_device='DONE')
        finally:
            LA.CHECKS, LA.SMOKE, LA.SMOKE_LAUNCH, LA.git_head = saved
    rec['launcher_gates_addendum2'] = out


# ------------------------------------------------------------------ RL の床（§4.0）: F_seed・0.00026 の導出・S16 (iii)
def check_floor_rl(rec):
    F = {s: RLC.floor_seed(s) for s in (0, 1, 2)}
    spec = {0: 0.11450, 1: 0.11400, 2: 0.11383}
    for s in spec:
        assert abs(F[s] - spec[s]) < 5e-6, (s, F[s], spec[s])
    assert abs(RLC.FLOOR_Z - 2.865) < 5e-4, RLC.FLOOR_Z                          # Φ⁻¹(1 − 0.05/24)
    der = RLC.floor_sd_from_reference()
    assert der['n'] == 200 and abs(der['sd_task'] - 0.00116) < 5e-6 and abs(der['sd_over_sqrt_n'] - RLC.FLOOR_SD) < 5e-6, der
    assert min(der['corr_own_seed'].values()) >= 0.93, der['corr_own_seed']       # R は自分の seed の最多比率に追随
    # S16 (iii): 0906 の R seed 0–9 は全部 AT_FLOOR、LR seed 0–9 は全部そうでない
    rows_R = RLC.read_csv_rows(RLC.REF_DIR / 'R' / 'per_task.csv')
    rows_LR = RLC.read_csv_rows(RLC.REF_DIR / 'LR' / 'per_task.csv')
    lab = {}
    for name, rows in (('R', rows_R), ('LR', rows_LR)):
        for s in range(10):
            e = RLC.endpoints([r for r in rows if int(r['seed']) == s], s)
            lab[f'{name}/{s}'] = dict(online=e['online_mean'], F=e['F_seed'], at_floor=e['at_floor'],
                                      Y_pt=e['Y_pt'], Y_logit=e['Y_logit'])
    assert all(lab[f'R/{s}']['at_floor'] for s in range(10)), lab
    assert not any(lab[f'LR/{s}']['at_floor'] for s in range(10)), lab
    # 対照 1: 閾値そのもの。F + margin + δ は床でなく、F + margin − δ は床（δ = 1e−6 ≫ 丸め）
    ctl1 = [(RLC.at_floor(lab[f'R/{s}']['F'] + RLC.FLOOR_MARGIN + 1e-6, s),
             RLC.at_floor(lab[f'R/{s}']['F'] + RLC.FLOOR_MARGIN - 1e-6, s)) for s in range(10)]
    assert all(c == (False, True) for c in ctl1), ctl1
    # 対照 2: LR の online を自分の seed の F に置き換えると 10/10 が床になる
    ctl2 = [RLC.at_floor(lab[f'LR/{s}']['F'], s) for s in range(10)]
    assert all(ctl2), ctl2
    # 窓が欠けた行では endpoint を出さない
    try:
        RLC.endpoints([r for r in rows_LR if int(r['seed']) == 0 and int(r['task']) <= 49], 0)
        raise AssertionError('endpoints accepted an incomplete window')
    except AssertionError as e:
        if 'accepted' in str(e):
            raise
    rec['floor_rl'] = dict(F_seed=F, z=RLC.FLOOR_Z, sd_derivation=der, labels=lab,
                           controls=dict(R_plus_2margin_not_floor=True, LR_at_F_is_floor=True,
                                         incomplete_window='fails'))


# ================================================================== 箱 B（act_chimera_pmnist_0913）の検査
# S14-PM（ループの写し = GS.train + P1–P6・AST・深さつき）, S18-PM（双子の init）, S11-PM（MNIST の sha256 と
# 1 byte の対照）, S9-PM（乱数の共有: ループの中の指紋 = 外で引いた系列・双子は init だけ）, S10-PM（測定が学習を
# 変えない・W1 += 1e−9 と乱数消費の対照）, E 恒等式（EG:135-141 の比・h は AC.h の表）, S12 を実状態で,
# S20-PM（task 2 の後に inf を注入 → DIVERGED・divergence.json・部分行 2 行、対照は COMPLETE で t1–3 が錨と 51/51）。
# 学習は t1–2（S9/S10）と t1–3（S20）の短い走だけ（CPU 1 thread・各 1〜3 秒）。
from src import act_chimera_pmnist_0913 as PM

from src.test_act_chimera_report_0913 import scratch_root, make_fixture   # noqa: E402（一時ディレクトリ・G0.5 の合成出力）
from src import act_chimera_report_0913 as RP                              # noqa: E402

PM_SCRATCH = scratch_root() / 'test_act_chimera_pmnist'
_PM_MN = {}


def _pm_mnist():
    if 'm' not in _PM_MN:
        _PM_MN['m'] = PM.load_mnist()
        _PM_MN['probe'], _PM_MN['pidx'], _PM_MN['xg'], _PM_MN['yg'] = PM.make_probe(_PM_MN['m'], 0)
    return _PM_MN['m'], _PM_MN['probe'], _PM_MN['xg'], _PM_MN['yg']


def _pm_train(arm, seed=0, tasks=2, readout=None, mutate=None):
    m, probe, xg, yg = _pm_mnist()
    if seed != 0:
        probe, _, xg, yg = PM.make_probe(m, seed)
    act_name, twin = AC.twin_of(arm)
    rows, units = [], {}
    p, _ = PM.train(arm, seed, m, probe, tasks, mutate=mutate, rows=rows, units=units, ck={},
                    act_obj=AC.make_act(act_name), twin=twin, readout=readout)
    return p, rows, units


def _pm_readout(act_name, tasks=2, **kw):
    _, _, xg, yg = _pm_mnist()
    return PM.Readout(act_name, tasks, xg, yg, e_tasks=tuple(range(1, tasks + 1)), **kw)


# ------------------------------------------------------------------ S14-PM: ループの写し（AST・深さつき）
def check_S14_pmnist(rec):
    r = PM.s14_check()
    assert r['pass_'], r
    assert r['host_stmts'] == PM.HOST_BODY_STMTS == 45 and r['mine_stmts'] == 45 + 8, r
    assert [t for t, *_ in PM.REGISTERED] == ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    src = Path(PM.__file__).read_text(encoding='utf-8')
    # 対照 1: 2 つの登録ブロック（P4 と P5）の間の 1 文字を変えた source
    bad = src.replace('xs[(step - 1) * 16:step * 16], act)', 'xs[(step - 1) * 16:step * 17], act)')
    assert bad != src
    c1 = PM.s14_check(mine_src=bad)
    assert c1['pass_'] is False and len(c1['diff_vs_expected']) == 1, c1['diff_vs_expected']
    # 対照 2: 行数の期待値を 0 にする
    c2 = PM.s14_check(expected_host=0)
    assert c2['pass_'] is False and not c2['diff_vs_expected']
    # 対照 3: 登録した置換 P5 を当てていない写し（.001 のまま）
    bad5 = src.replace('q -= lr * (mi / c1)', 'q -= .001 * (mi / c1)')
    assert bad5 != src
    c3 = PM.s14_check(mine_src=bad5)
    assert c3['pass_'] is False and len(c3['diff_vs_expected']) == 1
    # 対照 4: 字下げだけを変えた写し（P1 を `with` の中に入れる）は深さで捕まる
    bad_ind = src.replace('\n    if twin is not None:\n        AC.perturb_twin_b1(p, twin)',
                          '\n            if twin is not None:\n                AC.perturb_twin_b1(p, twin)')
    assert bad_ind != src
    c4 = PM.s14_check(mine_src=bad_ind)
    assert c4['pass_'] is False
    # S8: 新ランナーの AST にも EG.make_act / train / run / hdefect / measure の参照が無いこと
    hits = scan_forbidden_refs(Path(PM.__file__))
    assert hits == [], hits
    rec['S14_pmnist'] = dict(host_stmts=r['host_stmts'], mine_stmts=r['mine_stmts'], n_inserted=r['n_inserted'],
                             opcodes_vs_host=r['opcodes_vs_host'], s8_forbidden_hits=hits,
                             controls=dict(one_char_between_P4_P5=c1['pass_'], expected_zero=c2['pass_'],
                                           P5_not_applied=c3['pass_'], indent_only=c4['pass_']))


# ------------------------------------------------------------------ S18-PM: 双子の init（b1[k] の 1 要素・+1e−6 の最近接の float32・追補 2-2）
def check_S18_pmnist(rec):
    vals = {}
    for seed in range(3):
        for k in range(4):
            t = PM.twin_report(seed, k)
            assert t['pass_'] and t['n_diff'] == 1 and t['where'] == [(1, (k,))], (seed, k, t)
            base = H.init_params(seed, torch.device('cpu'))
            want = float((base[1][k].detach().double() + 1e-6).float())              # 独立な再計算（torch の丸め）
            assert t['new'] == want and t['init_sha_excl_b1k_equal'] and t['n_ulps'] >= 100, (seed, k, t)
            assert abs(t['realised_delta'] - 1e-6) <= t['ulp_at_old'], t              # 最近接なら差は 1 ulp 以内
            vals[f's{seed}/k{k}'] = dict(realised_delta=t['realised_delta'], n_ulps=t['n_ulps'])
    ctl = PM.twin_report(0, 0, delta=0.)
    assert ctl['pass_'] is False and ctl['n_diff'] == 0, ctl                          # 対照 1: 摂動 0 は「差 0 要素」
    ctl2 = PM.twin_report(0, 0, delta=2e-6)
    assert ctl2['pass_'] is False and ctl2['n_diff'] == 1 and ctl2['exact_step'] is False, ctl2   # 対照 2: 値
    old = lambda p, k: p[0].data.__setitem__((k, 0), torch.nextafter(p[0].data[k, 0], torch.tensor(float('inf'))))
    ctl3 = PM.twin_report(0, 1, perturb=old)
    assert ctl3['pass_'] is False and ctl3['where'] == [(0, (1, 0))] and not ctl3['init_sha_excl_b1k_equal'], ctl3  # 対照 3: 位置
    assert AC.twin_of('LRtw3') == ('LR', 3) and PM.TWIN_KS == (0, 1, 2, 3)
    rec['S18_pmnist'] = dict(values=vals, controls=dict(zero_delta_ndiff=ctl['n_diff'], delta_2e6_exact=ctl2['exact_step'],
                                                        old_W1_k0_nextafter_where=ctl3['where']))


# ------------------------------------------------------------------ S11-PM: MNIST の sha256（ランナーの data_check）
def check_S11_pmnist(rec):
    import hashlib as _hl
    import shutil as _sh
    m, _, _, _ = _pm_mnist()
    d = PM.data_check(m)
    assert d['pass_'], d
    with tempfile.TemporaryDirectory() as td:
        f = 't10k-labels-idx1-ubyte.gz'
        dst = Path(td) / f
        _sh.copy2(Path(H.DATA_DIR) / f, dst)
        b = bytearray(dst.read_bytes())
        b[len(b) // 2] ^= 0x01
        dst.write_bytes(bytes(b))
        bad = _hl.sha256(dst.read_bytes()).hexdigest()
    assert bad != m.sha256[f] == PM.T.DATA_SHA[f]
    rec['S11_pmnist'] = dict(data_sha256=m.sha256, data_dir=d['data_dir'], controls=dict(one_byte_flipped_differs=True))


# ------------------------------------------------------------------ S9-PM / S10-PM / E 恒等式（t1–2 の短い走）
def check_S9_S10_pmnist(rec):
    m, probe, xg, yg = _pm_mnist()
    pa, _, _ = _pm_train('LR')                                        # 読み出し無し
    ro_b = _pm_readout('LR', sha_excl_ks=PM.TWIN_KS)
    pb, _, ub = _pm_train('LR', readout=ro_b)
    s10 = PM._maxabs_params(pa, pb)
    assert s10 == 0., s10                                             # S10: 新読み出しの有無で 2 タスク後の params が同じ
    ro_c = _pm_readout('LR', mut='add1e-9', sha_excl_ks=PM.TWIN_KS)
    pc, _, _ = _pm_train('LR', readout=ro_c)
    ctl_add = PM._maxabs_params(pa, pc)
    assert ctl_add > 0., ctl_add                                      # 対照 (i): 測定の後に W1 += 1e−9
    # 対照 (ii): ループの batch 系列を読み出しの中で 1 回消費する誤り（H.stream を差し替えて生成器を捕まえる）
    captured, orig = {}, H.stream

    def spy(role, seed):
        g = orig(role, seed)
        captured[role] = g
        return g

    class RngEater(PM.Readout):
        def task_start(self, *a, **k):
            super().task_start(*a, **k)
            torch.randperm(2, generator=captured['batch'])

    H.stream = spy
    try:
        ro_d = _pm_readout('LR')
        ro_d.__class__ = RngEater
        pd_, _, _ = _pm_train('LR', readout=ro_d)
    finally:
        H.stream = orig
    ctl_rng = PM._maxabs_params(pa, pd_)
    assert ctl_rng > 0., ctl_rng
    assert ro_d.a['param_sha'][0] == ro_b.a['param_sha'][0]           # task 1 の学習は変わらず、効くのは task 2 から
    assert ro_d.a['param_sha'][1] != ro_b.a['param_sha'][1]
    # S9: 乱数の共有。init・t1–2 の perm/idx/order が腕に依らない。双子は init の 1 要素（b1[k]）だけ。
    ros = {'LR': ro_b}
    for arm in ('SMINH', 'GN', 'LRtw0'):
        ro = _pm_readout(AC.twin_of(arm)[0], sha_excl_ks=((0,) if arm == 'LRtw0' else ()))
        _pm_train(arm, readout=ro)
        ros[arm] = ro
    # S18b（追補 2-2）: 双子 LRtw0（b1[0] +1e−6）は t1 で LR と食い違う。対照: 摂動 0 の「双子」（同じ LR をもう一度）は
    # 食い違わない（None）。t1 の測定の後に W1 += 1e−9 する走（ro_c）は t2 で初めて食い違う（最初のタスクの数え方）。
    live = AC.twin_live_task(ro_b.sha_excl[0], ros['LRtw0'].sha_excl[0])
    ro_same = _pm_readout('LR', sha_excl_ks=PM.TWIN_KS)
    _pm_train('LR', readout=ro_same)
    same = AC.twin_live_task(ro_b.sha_excl[0], ro_same.sha_excl[0])
    late = AC.twin_live_task(ro_b.sha_excl[0], ro_c.sha_excl[0])
    assert live == dict(twin_live_task=1, n_compared=2) and same == dict(twin_live_task=None, n_compared=2), (live, same)
    assert late == dict(twin_live_task=2, n_compared=2), late
    assert ro_b.a['param_sha'][1] == AC.params_sha256(pb)             # exclude=None は _sha_params と同じ
    fp = PM.rng_fingerprint(0, 2, m)
    for arm, ro in ros.items():
        for t in ('t1', 't2'):
            assert ro.rng_sha[t] == fp[t] == ro_b.rng_sha[t], (arm, t)
        assert (ro.rng_sha['init'] == fp['init']) == (not arm.startswith('LRtw')), arm
    # 対照: seed + 1 で不一致
    fp1 = PM.rng_fingerprint(1, 2, m)
    ro1 = _pm_readout('LR')
    _pm_train('LR', seed=1, readout=ro1)
    assert ro1.rng_sha['t1'] == fp1['t1'] and fp1['t1'] != fp['t1'] and fp1['init'] != fp['init']
    assert all(fp1['t1'][k] != fp['t1'][k] for k in ('perm', 'idx', 'order'))
    # E 恒等式（EG:135-141 の比・h は AC.h の表・誤った h は WRONG_H）
    ident = {}
    for arm, ro in ros.items():
        c = ro.identity_checks()
        assert c['evaluated'] and c['identity_rel'] < PM.IDENTITY_TOL, (arm, c)
        assert c['identity_mutctl'] > PM.IDENTITY_CTL, (arm, c)
        if AC.twin_of(arm)[0] in PM.LEAKY_ZERO:
            assert c['leaky_zero'] < PM.IDENTITY_TOL, (arm, c)
        ident[arm] = c
    # 走の中の S12（実状態）: Σ_k fn_mom = 帯なしの平均、Σ_k mob_k = mob（= gate_block の gbar_i）
    n_inputs = probe.px.shape[0]
    for arm, ro in ros.items():
        assert ro.ck['s12_band_sum_maxdiff'] <= n_inputs * EPS[F64], (arm, ro.ck)
        assert ro.ck['s12_mob_sum_maxdiff'] <= n_inputs * EPS[F64], (arm, ro.ck)
        assert ro.ck['mob_sum_vs_gbar_i_maxdiff'] <= n_inputs * EPS[F64], (arm, ro.ck)
        assert ro.ck['wtnorm_vs_cnorm_maxdiff'] == 0., (arm, ro.ck)
    rec['S9_S10_pmnist'] = dict(s10_noninvasive=s10, controls=dict(add1e_9=ctl_add, rng_consumed=ctl_rng,
                                                                    seed_plus_1_differs=True,
                                                                    s18b_zero_perturbation_twin=same, s18b_add1e_9_after_t1=late),
                                s18b_twin_LRtw0=live,
                                rng_sha=fp, identity=ident,
                                s12_in_run={a: dict(ro.ck) for a, ro in ros.items()})


# ------------------------------------------------------------------ S20-PM: 打ち切りの経路（t1–3・task 2 の後に inf を注入）
def _fresh(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def check_S20_pmnist(rec):
    import csv
    m, _, _, _ = _pm_mnist()
    out = _fresh(PM_SCRATCH / 's20')
    # 錨の腕（LR）の DIVERGED は G1-PM の失敗でもある（§4.x「切り詰めで早死にが通る」）: 出力を書いてから落ちる
    try:
        PM.run('LR', 0, tasks=3, out=out, controls=False, e_tasks=(1, 2, 3), inject_inf_after=2, mnist=m)
        raise AssertionError('a diverged anchor arm did not fail its per-run checks')
    except AssertionError as exc:
        if 'did not fail' in str(exc):
            raise
        assert 'G1_PM' in str(exc), exc
    dv = json.loads((out / 'LR_s0_divergence.json').read_text())
    assert dv['status'] == 'DIVERGED' and dv['first_nonfinite_task'] == 3 and dv['tasks_completed'] == 2, dv
    with (out / 'LR_s0_rows.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and [int(r['task']) for r in rows] == [1, 2]
    assert 'sat' in rows[0] and 'dead_units' in rows[0]                  # §3.4 の報告列（EG の sat・dead_units）
    u = np.load(out / 'LR_s0_units.npz')
    assert sorted({PM._task_of(k) for k in u.files}) == [1, 2] and len(u.files) == 32
    ro = np.load(out / 'LR_s0_readout.npz')
    assert ro['fn_mom'].shape == (2, 2, 8, 4, 3) and ro['E_ident'].shape == (3, 3) and np.isnan(ro['E_ident'][2]).all()
    assert ro['param_sha'].dtype == np.dtype('<U64') and ro['param_sha'].shape == (2,)
    prov = json.loads((out / 'LR_s0_provenance.json').read_text())
    assert prov['status'] == 'DIVERGED' and prov['tasks_completed'] == 2 and prov['checks_passed'] is False
    g1d = prov['checks']['g1_pm']
    assert g1d['n_equal'] == g1d['n_compared'] == 34 and g1d['n_expected'] == 51 and g1d['pass_'] is False, g1d
    assert 'G1_PM' in prov['failed_checks'], prov['failed_checks']
    # 対照: 注入しない同じ走が COMPLETE になり divergence.json を書かない（t1–3 が錨と 51/51 で一致）
    out2 = _fresh(PM_SCRATCH / 's20_ctl')
    st2 = PM.run('LR', 0, tasks=3, out=out2, controls=False, e_tasks=(1, 2, 3), mnist=m)
    assert st2 == 'COMPLETE' and not (out2 / 'LR_s0_divergence.json').exists()
    prov2 = json.loads((out2 / 'LR_s0_provenance.json').read_text())
    with (out2 / 'LR_s0_rows.csv').open() as fh:
        assert len(list(csv.DictReader(fh))) == 3
    g1 = prov2['checks']['g1_pm']
    assert g1['pass_'] and g1['n_equal'] == g1['n_expected'] == 51 and g1['maxabs_numeric'] == 0., g1
    assert prov2['checks']['g1_eg']['pass_'] and prov2['checks']['s9']['pass_'] and prov2['checks_passed'] is True
    assert prov2['checks']['identity']['evaluated'] and prov2['checks']['identity']['ctl_demonstrated']
    assert prov2['checks']['s12_in_run']['pass_'], prov2['checks']['s12_in_run']
    assert prov2['fn_mom_axes']['funcs'] == list(AC.FUNCS8) and prov2['band_predicate_dtype'] == AC.BAND_PREDICATE_DTYPE
    # §8 のスキーマ（T = 3）
    ro2 = np.load(out2 / 'LR_s0_readout.npz')
    U = H.DIMS[1]
    schema = dict(fn_mom=((3, 2, 8, 4, 3), 'float64'), occ_start=((3, 2, U, 4), 'float32'),
                  occ_end=((3, 2, U, 4), 'float32'), mob_band=((3, 2, U, 4), 'float32'), pabs1=((3, 2, U), 'float32'),
                  zbar_start=((3, 2, U), 'float64'), zbar_end=((3, 2, U), 'float64'), depth_vel=((3, 2, U), 'float64'),
                  step_num=((3, 2, U), 'float64'), wt_norm=((3, 2, U), 'float64'), mu_comp=((3, U), 'float64'),
                  adam_ratio=((3, 2, U), 'float32'), adam_epsfrac=((3, 2, U), 'float32'), gbar_l2=((3, U), 'float64'),
                  mu_phi1=((3, U), 'float64'), E_ident=((3, 3), 'float64'), param_sha=((3,), '<U64'))
    assert set(ro2.files) == set(schema), set(ro2.files) ^ set(schema)
    for k, (shape, dt) in schema.items():
        assert ro2[k].shape == shape and ro2[k].dtype == np.dtype(dt), (k, ro2[k].shape, ro2[k].dtype)
        if ro2[k].dtype.kind == 'f':
            assert np.isfinite(ro2[k]).all(), k
    for k in ('git_hash', 'spec_sha256', 'code_sha256', 'act_sha256', 'host_sha256', 'gs_sha256', 'data_sha256',
              'env_executable', 'env_torch', 'env_numpy', 'env_hostname', 'env_cpu', 'wall_seconds', 'peak_rss_kib',
              'other_sessions_before', 'other_sessions_after', 'git_dirty_src'):
        assert k in prov2, k
    # G0.5 の該当家族の行（§6 S20）: DIVERGED の LR_s0 と合成の他の走（スモークの地平）に判定器を掛けると、LR を親に持つ
    # H・V・S の全家族の D_PAR/D1/D2/D3_*/HYPOTHESIS が NOT_DETERMINED_DIVERGED。対照: LR_s0 を注入しない走に替える。
    Ws = RP.Windows.smoke()
    items = ('D_PAR', 'D1', 'D2', 'D3_N', 'D3_D', 'D3_I', 'D3_HEADING', 'HYPOTHESIS')
    labels = {}
    for tag, src_dir in (('diverged', out), ('control', out2)):
        root = _fresh(PM_SCRATCH / f's20_g05_{tag}')
        make_fixture(root, Ws, envs=('pmnist',))
        for f in list((root / 'pmnist').glob('LR_s0_*')):
            f.unlink()
        for f in src_dir.glob('LR_s0_*'):
            shutil.copy(f, root / 'pmnist' / f.name)
        E = RP.load_pmnist(root / 'pmnist', [0, 1, 2], RP.pm_arms(), Ws, strict=False)
        R, ctx = RP.Report(), RP.DoseCtx()
        for fam in ('H', 'V', 'S'):
            RP.judge_family(E, 'pmnist', Ws, fam, RP.endpoints_pmnist(Ws), ctx, R, {'L': 0., 'A': 0.}, ('L', 'A'))
        labels[tag] = {f'{fam}/{it}': R.get('pmnist', fam, 'all', it) for fam in ('H', 'V', 'S') for it in items}
    assert all(v == 'NOT_DETERMINED_DIVERGED' for v in labels['diverged'].values()), labels['diverged']
    assert not any(v == 'NOT_DETERMINED_DIVERGED' for v in labels['control'].values()), labels['control']
    rec['S20_pmnist'] = dict(status='DIVERGED', first_nonfinite_task=dv['first_nonfinite_task'], partial_rows=len(rows),
                             g1_diverged_anchor=g1d, g05_rows=labels['diverged'],
                             controls=dict(no_injection_status=st2, no_divergence_json=True, g1_t1_3=g1,
                                           g05_rows_without_divergence=labels['control']))


# ------------------------------------------------------------------ G1-PM の適用範囲（錨・lr・双子）
def check_G1_scope_pmnist(rec):
    """G1-PM と副次の錨は「錨の活性化・双子でない・lr = 1e−3」の走だけに当てる（§2.2 P5・§2.1 段 2）。
    段 2 の lr 2e−3 の LR と、双子 LRtw0 は、錨と比べず（anchor = None）検査を通って COMPLETE で終わる。対照: 同じ
    1 タスクの LR（lr 1e−3）は錨と 17/17 で一致する。双子の S18 は走そのものの init の sha256 と結ぶ。"""
    m, _, _, _ = _pm_mnist()
    got = {}
    for arm, lr in (('LR', 2e-3), ('LRtw0', 1e-3), ('LR', 1e-3)):
        out = _fresh(PM_SCRATCH / f'g1scope_{arm}_{lr}')
        st = PM.run(arm, 0, tasks=1, out=out, controls=False, lr=lr, e_tasks=(1,), mnist=m)
        prov = json.loads((out / f'{arm}_s0_provenance.json').read_text())
        assert st == 'COMPLETE' and prov['checks_passed'] is True, (arm, lr, prov['failed_checks'])
        got[f'{arm}@{lr}'] = dict(g1_pm=prov['checks']['g1_pm'], g1_eg=prov['checks']['g1_eg'],
                                  s18=prov['checks'].get('s18'), s18b=prov['checks'].get('s18b'),
                                  run_identity=RP.run_identity(prov))
    assert got['LR@0.002']['g1_pm']['anchor'] is None and 'lr differs' in got['LR@0.002']['g1_pm']['note']
    assert got['LRtw0@0.001']['g1_pm']['anchor'] is None and got['LRtw0@0.001']['s18']['tied_to_run'] is True
    assert got['LRtw0@0.001']['s18']['ctl_zero_perturbation']['fails_as_required'] is True
    assert got['LRtw0@0.001']['s18b']['ks'] == [0] and got['LR@0.001']['s18b']['ks'] == [0, 1, 2, 3]
    tl = AC.twin_live_task(got['LR@0.001']['s18b']['sha_excl_b1']['0'], got['LRtw0@0.001']['s18b']['sha_excl_b1']['0'])
    assert tl == dict(twin_live_task=1, n_compared=1), tl
    # S18b の前提（判定器の run_identity）: 同じ機械・コードの箱 B の LR と双子は同一性が等しく、device は cpu
    idl, idt = got['LR@0.001']['run_identity'], got['LRtw0@0.001']['run_identity']
    assert idl == idt and idl['device'] == 'cpu' and idl['git_hash'] and idl['code_sha256'] and idl['torch'], (idl, idt)
    assert got['LR@0.001']['g1_pm']['pass_'] and got['LR@0.001']['g1_pm']['n_compared'] == 17
    # 対照: 双子の S18 を別の init（seed 1 の摂動）と結ぶと落ちる
    other = PM.twin_report(1, 0)['perturbed_init_sha']
    assert PM.twin_report(0, 0, loop_init_sha=other)['pass_'] is False
    rec['G1_scope_pmnist'] = got


# ------------------------------------------------------------------ pytest の入口と collect()
CHECKS = [check_S0, check_S1, check_S2, check_S3, check_S3b, check_S4, check_S5, check_S6, check_S7,
          check_S8, check_S12, check_S13, check_S13b, check_S15, check_S17,
          check_S14_rl, check_S11, check_S9_rl, check_S18_rl, check_S18b_rl, check_S10_rl, check_readout_schema_rl,
          check_S20_rl, check_G1_rl_comparator, check_G1_rl_control_live, check_launcher_gates_addendum2, check_floor_rl,
          check_S14_pmnist, check_S18_pmnist, check_S11_pmnist, check_S9_S10_pmnist, check_S20_pmnist,
          check_G1_scope_pmnist]
RECORD = {}


def test_S0_constants():
    check_S0(RECORD)


def test_S1_analytic_vs_autograd():
    check_S1(RECORD)


def test_S2_phi_vs_integral():
    check_S2(RECORD)


def test_S3_seam_continuity():
    check_S3(RECORD)


def test_S3b_exact_seam():
    check_S3b(RECORD)


def test_S4_h_identity():
    check_S4(RECORD)


def test_S5_extreme_finite():
    check_S5(RECORD)


def test_S6_factorial_identity():
    check_S6(RECORD)


def test_S7_types_bits_kind():
    check_S7(RECORD)


def test_S8_make_act_and_forbidden_refs():
    check_S8(RECORD)


def test_S12_gamma_identities_synthetic():
    check_S12(RECORD)


def test_S13_class_swap():
    check_S13(RECORD)


def test_S13b_mlpl_bits():
    check_S13b(RECORD)


def test_S15_curvature():
    check_S15(RECORD)


def test_S17_split_arms():
    check_S17(RECORD)


# ---- RL（act_chimera_rlmnist_0913）
def test_S14_rl_loop_copy():
    check_S14_rl(RECORD)


def test_S11_data_and_carry7_sha256():
    check_S11(RECORD)


def test_S9_rl_rng_sharing():
    check_S9_rl(RECORD)


def test_S18_rl_twin_b1_step():
    check_S18_rl(RECORD)


def test_S18b_rl_twin_liveness():
    check_S18b_rl(RECORD)


def test_G1_rl_control_live():
    check_G1_rl_control_live(RECORD)


def test_launcher_gates_addendum2():
    check_launcher_gates_addendum2(RECORD)


def test_S10_rl_readout_noninvasive():
    check_S10_rl(RECORD)


def test_readout_schema_rl():
    check_readout_schema_rl(RECORD)


def test_S20_rl_incomplete_path():
    check_S20_rl(RECORD)


def test_G1_rl_comparator():
    check_G1_rl_comparator(RECORD)


def test_floor_rl():
    check_floor_rl(RECORD)


# ---- 箱 B（act_chimera_pmnist_0913）
def test_S14_pmnist_loop_copy():
    check_S14_pmnist(RECORD)


def test_S18_pmnist_twin_b1_step():
    check_S18_pmnist(RECORD)


def test_S11_pmnist_data_sha256():
    check_S11_pmnist(RECORD)


def test_S9_S10_pmnist_rng_and_noninvasive():
    check_S9_S10_pmnist(RECORD)


def test_S20_pmnist_diverged_path():
    check_S20_pmnist(RECORD)


def test_G1_scope_pmnist():
    check_G1_scope_pmnist(RECORD)


def collect(checks=None):
    """全検査（または checks）を走らせ、値・許容値・対照値の dict を返す（launcher が static_checks.json に写す）。"""
    rec = {}
    for fn in (CHECKS if checks is None else checks):
        fn(rec)
    rec['_env'] = dict(executable=sys.executable, torch=torch.__version__, numpy=np.__version__)
    return rec


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ================================================================== SCR（act_chimera_scr_0913）の検査
# S9-SCR（乱数の共有・generator_offset 1 の対照）, S13-SCR（setup_arm_chimera の差し替えが状態を触らない）,
# S14-SCR（_run_arm_chimera / setup_arm_chimera の写し・行数の期待値）, G1-SCR の短縮版（10 系列 × 2k/20k step を
# edge_law の null 腕の先頭記録と s_null の規則で比較・generator_offset 1 の対照）, S19（安定余裕: w_free の再構成と
# ckpt 直接の λ_out・冪乗法と eigvalsh・対角近似の対照）, 読み出しのスキーマ（§8・≤ 24 MB）, RSS の外挿。
# 学習は tiny（CPU 1 thread・数秒〜十数秒）。edge_law の参照 npz / ckpt は proj_004_drift から読むだけ（書かない）。
# 本走の interpreter は .venv（§2.1）。ここは import できる interpreter ならどちらでも走る。
import atexit
import copy
import inspect
import importlib.util
import os

from src import act_chimera_scr_0913 as SC
from src import edge_law_0905 as EL
from src import gate_dial_0902 as GD

SCR_REF = Path(SC.registered()['g1']['reference_logs']).parent     # proj_004_drift/results/edge_law_0905
SCR_AUX7 = ('layer1_m_dphi2', 'layer1_m_dphiddphi', 'layer1_m_phi2', 'layer1_m_phidphi',
            'layer1_moment_step', 'layer1_w_free', 'layer1_w_free_step')
_SCR = {}


def scr_env():
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    SC.bind_config()


def scr_tmp():
    if 'tmp' not in _SCR:
        _SCR['tmp'] = Path(tempfile.mkdtemp(prefix='act_chimera_scr_test_'))
        atexit.register(shutil.rmtree, _SCR['tmp'], True)
    return _SCR['tmp']


def scr_run(arm, steps, sub='main', generator_offset=None):
    """腕を steps だけ（10 系列・CPU）走らせた outdir。検査間で使い回す。"""
    key = (arm, steps, sub, generator_offset)
    if key not in _SCR:
        scr_env()
        out = scr_tmp() / sub
        r = SC.run_single_arm(arm, steps=steps, outdir=out, generator_offset=generator_offset)
        assert r['status'] == 'COMPLETE', r
        _SCR[key] = out
    return _SCR[key]


def _sha_t(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def _scr_fingerprint(st):
    h = {f'net.{k}': _sha_t(v) for k, v in st['net'].state_dict().items()}
    h.update({f'teacher.{k}': _sha_t(v) for k, v in st['teacher'].state_dict().items()})
    h['env.flip_state'] = _sha_t(st['env'].flip_state)
    h['running_mean'] = _sha_t(st['running_mean'])
    return h


SCR_S9_STEPS = 30_000                     # §6 S9: 先頭 30k の入力（flip は T_r = 10,000 step ごとなので 3 回の flip 抽選を含む）


def _scr_stream(st, n=SCR_S9_STEPS):
    """init・教師・flip 系列・先頭 n step の入力の sha256（S9）。env を進めるので使い捨ての state に掛ける。
    flip 系列は n step 後の flip_state（10,000 step ごとの flip 抽選が共有の生成器から引かれた結果）で見る。"""
    h = _scr_fingerprint(st)
    hx = hashlib.sha256()
    for _ in range(n):
        hx.update(np.ascontiguousarray(st['env'].step().detach().cpu().numpy()).tobytes())
    h['inputs'] = hx.hexdigest()
    h['env.flip_state_after'] = _sha_t(st['env'].flip_state)
    return h


# ------------------------------------------------------------------ S9-SCR: 腕は乱数に入らない
def check_S9_scr(rec):
    scr_env()
    cfg = SC.build_cfg()
    arms = tuple(SC.table())                                           # 主 12 行 + 退避枝 12 行（24 行すべて）
    assert len(arms) == 24, arms
    base = _scr_stream(SC.setup_arm_chimera(copy.deepcopy(cfg), GD._arm(cfg, arms[0]), 'cpu'))
    for arm in arms[1:]:
        h = _scr_stream(SC.setup_arm_chimera(copy.deepcopy(cfg), GD._arm(cfg, arm), 'cpu'))
        assert h == base, (arm, [k for k in base if h[k] != base[k]])
    cfg1 = copy.deepcopy(cfg)
    cfg1['common']['generator_offset'] = 1
    mut = _scr_stream(SC.setup_arm_chimera(cfg1, GD._arm(cfg1, 'chLR_1216'), 'cpu'))
    differing = [k for k in base if mut[k] != base[k]]
    for k in ('net.W', 'net.v', 'teacher.W', 'env.flip_state', 'inputs', 'env.flip_state_after'):
        assert k in differing, k
    # b は kaiming_mlp_params が 0 で初期化する（乱数を引かない）ので offset で変わらない: 0 であることを確かめる
    st0 = SC.setup_arm_chimera(copy.deepcopy(cfg), GD._arm(cfg, 'chLR_1216'), 'cpu')
    assert 'net.b' not in differing and not st0['net'].bs[0].any()
    rec['S9_scr'] = dict(values=dict(arms=list(arms), n_hashes=len(base), all_equal=True, inputs_steps=SCR_S9_STEPS),
                         controls=dict(generator_offset_1_differing=differing))


# ------------------------------------------------------------------ S13-SCR: setup_arm_chimera（差し替え 1 行）
def check_S13_scr(rec):
    scr_env()
    cfg = SC.build_cfg()
    row_lr = GD._arm(cfg, 'chLR_1216')
    host = GD.setup_arm_dial(copy.deepcopy(cfg), row_lr, 'cpu')
    mine = SC.setup_arm_chimera(copy.deepcopy(cfg), row_lr, 'cpu')
    assert type(host['net']) is N.VecMLPL and type(mine['net']) is AC.ChimeraMLPL
    assert _scr_fingerprint(host) == _scr_fingerprint(mine)
    assert (mine['activation'], mine['act_alpha'], mine['family']) == ('leaky_relu', 0.1, 'leaky')
    z = grid(F64, -30., 30., 200_001)
    assert bit_equal(mine['net'].act_fn(z), host['net'].act_fn(z))
    assert bit_equal(mine['net'].act_grad(z, mine['net'].act_fn(z)), host['net'].act_grad(z, host['net'].act_fn(z)))
    vals = {}
    for arm in ('chSMAXH_1216', 'chSMINS_1216', 'chGD_1216', 'chFD_lr0p005_1216'):
        st = SC.setup_arm_chimera(copy.deepcopy(cfg), GD._arm(cfg, arm), 'cpu')
        assert _scr_fingerprint(st) == _scr_fingerprint(host)                 # 腕は乱数に入らない
        act = SC.table()[arm]['activation']
        assert (st['net'].act, st['net'].act_alpha, st['activation'], st['family']) == (act, 0.1, act, 'chimera')
        assert bit_equal(st['net'].act_fn(z), REF_PHI[act if act in REF_PHI else AC.SPLIT[act][0]](z))
        vals[arm] = act
    ctl = {}
    for bad in (dict(row_lr, activation='SMAXH', dial=0.11), dict(row_lr, activation='LRtw0')):
        try:
            SC.setup_arm_chimera(copy.deepcopy(cfg), bad, 'cpu')
            raise AssertionError(bad)
        except (ValueError, KeyError) as exc:
            ctl[f"{bad['activation']}_{bad['dial']}"] = type(exc).__name__
    try:                                              # 差し替え無しの宿主はキメラ名を受け付けない（差し替えが効いている）
        GD.setup_arm_dial(copy.deepcopy(cfg), GD._arm(cfg, 'chSMAXH_1216'), 'cpu')
        raise AssertionError('host accepted SMAXH')
    except (ValueError, KeyError, NotImplementedError) as exc:
        ctl['host_without_swap'] = type(exc).__name__
    rec['S13_scr'] = dict(values=dict(state_equal=True, chimera_arms=vals), controls=ctl)


# ------------------------------------------------------------------ S14-SCR: 写しの差分と行数
def check_S14_scr(rec):
    r = SC.s14_scr()
    assert r['pass_'] and r['lines_ok'] and r['ast_pass'], r
    assert r['run_arm_vs_edge']['host_lines'] == 45 and r['run_arm_vs_edge']['mine_lines'] == 46
    assert r['run_arm_vs_weird']['host_lines'] == 41 and r['setup_arm']['host_lines'] == 9
    assert r['ast_run_arm_vs_edge']['host_stmts'] == 35 and r['ast_run_arm_vs_edge']['mine_stmts'] == 36
    assert r['ast_setup_arm']['host_stmts'] == 7 and r['ast_setup_arm']['mine_stmts'] == 8
    # 対照 (AST): 字下げだけの変異（書き出しを sanity 失敗の if の中へ入れる = 成功時にログを書かない）を AST が捕まえる。
    # edge_law の _copy_opcodes（字下げを落とす）はこれを通してしまう（レビューで再現済み）ことも記録する。
    here = Path(SC.__file__).read_text(encoding='utf-8')
    needle = ('        raise SanityError(f"{arm} exact-support sanity failed: {sanity}")\n'
              '    write_arm_logs_dial(outdir, arm, st, rec)\n')
    assert here.count(needle) == 1
    indent_mut = here.replace(needle, '        raise SanityError(f"{arm} exact-support sanity failed: {sanity}")\n'
                                      '        write_arm_logs_dial(outdir, arm, st, rec)\n')
    ind = SC.s14_scr(mine_run_src=indent_mut)
    assert ind['ast_pass'] is False and ind['ast_run_arm_vs_edge']['diff_vs_expected'], ind['ast_run_arm_vs_edge']
    zero_ast = SC.s14_scr(expected_stmts={'_run_arm_edge': 0, 'setup_arm_dial': 0})
    assert zero_ast['pass_'] is False and zero_ast['ast_pass'] is False
    # 対照 (a): 2 つの登録ブロックの**間**の 1 文字を変えた source（seeds → seed）で差分が出ること
    src = inspect.getsource(SC._run_arm_chimera)
    assert 'c["common"]["seeds"] = seeds' in src
    mut = src.replace('c["common"]["seeds"] = seeds', 'c["common"]["seeds"] = seed', 1)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'mut_run_arm.py'
        p.write_text('from pathlib import Path\n' + mut, encoding='utf-8')
        spec = importlib.util.spec_from_file_location('mut_run_arm', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bad = EL._copy_opcodes(EL._run_arm_edge, mod._run_arm_chimera, (SC.RUN_INSERT,))
    assert bad['pass_'] is False and bad['unregistered_opcodes'], bad
    # 対照 (b): 行数の期待値を 0 にすると落ちること
    old = SC.S14_EXPECTED_LINES
    try:
        SC.S14_EXPECTED_LINES = {k: 0 for k in old}
        zero = SC.s14_scr()
    finally:
        SC.S14_EXPECTED_LINES = old
    assert zero['pass_'] is False and zero['lines_ok'] is False
    rec['S14_scr'] = dict(values=dict(expected_lines=dict(old), inserted_vs_edge=r['run_arm_vs_edge']['inserted'],
                                      inserted_vs_weird=r['run_arm_vs_weird']['inserted'],
                                      inserted_setup=r['setup_arm']['inserted']),
                          controls=dict(one_char_mutation_opcodes=bad['unregistered_opcodes'],
                                        zero_expected_lines_pass=zero['pass_'],
                                        indent_only_mutation_ast_pass=ind['ast_pass'],
                                        zero_expected_stmts_pass=zero_ast['pass_']))


# ------------------------------------------------------------------ G1-SCR（短縮）: 参照の先頭記録と bit 一致・対照
def check_G1_scr_short(rec):
    scr_env()
    ref = SCR_REF / 'logs'
    assert ref.exists(), ref
    for ref_arm in ('LRnull_1216', 'Enull_1216'):                     # 期待一覧は参照 npz から（67 キー）
        with np.load(ref / f'{ref_arm}_seed0.npz', allow_pickle=True) as z:
            keys = tuple(k for k in sorted(z.files) if k not in EL.S_NULL_SKIP and k != 'state_hash_1m')
        assert keys == SC.S_NULL_EXPECTED_KEYS and len(keys) == 67, ref_arm
    out = scr_run('chLR_1216', 20_000)
    scr_run('chE_1216', 2_000)
    vals = {}
    for arm, ref_arm, n_rec in (('chLR_1216', 'LRnull_1216', 21), ('chE_1216', 'Enull_1216', 3)):
        pre = SC.s_null_prefix(arm, ref, out, ref_arm)
        assert pre['pass_'], pre
        assert all(r['n_records'] == n_rec and r['n_compared'] == 67 and r['n_bad'] == 0 for r in pre['rows'])
        keys = SC._compared_keys(out / 'logs' / f'{arm}_seed0.npz', ref / f'{ref_arm}_seed0.npz')
        assert tuple(keys) == SC.S_NULL_EXPECTED_KEYS
        raw = EL.s_null(arm, ref, out, ref_arm=ref_arm)                  # edge_law の s_null そのもの
        bad = sorted(raw['rows'][0]['bad'])
        assert tuple(bad) == SCR_AUX7, bad                                # 独自 step 列の 7 列だけが形で落ちる
        vals[arm] = dict(n_records=n_rec, n_compared=67, n_bad=0,
                         aux_rows_compared=pre['rows'][0]['aux_rows_compared'],
                         edge_law_s_null_shape_only_bad=bad)
    g = SC.g1_scr(out, ref, smoke=True)                                  # 31 記録の規則: 記録数だけで落ちる
    for arm in ('chLR_1216', 'chE_1216'):
        for r in g['arms'][arm]['rows']:
            assert r['n_bad'] == 0 and r['keys_ok'] and not r['n_records_ok'] and r['n_records_want'] == 31
    assert g['pass_'] is False
    assert all(v is not None for v in g['reference_npz_sha256'].values())
    ctrl = scr_run('chLR_1216', 2_000, sub='ctrl', generator_offset=1)
    c = SC.g1_smoke_control(ctrl, ref, n_records_want=3)
    assert c['pass_'] and all(r['unfit_differs'] for r in c['rows']), c
    c_short = SC.g1_smoke_control(ctrl, ref)                             # 対照: 登録の 31 記録に届かない対照は pass しない
    assert c_short['pass_'] is False, c_short
    pre_c = SC.s_null_prefix('chLR_1216', ref, ctrl, 'LRnull_1216')
    assert pre_c['pass_'] is False and all(r['n_bad'] > 0 for r in pre_c['rows'])
    try:                                                                  # 登録の地平線では offset を拒否する
        SC.run_single_arm('chLR_1216', steps=None, outdir=scr_tmp() / 'never', generator_offset=1)
        raise AssertionError('offset accepted at the registered horizon')
    except GD.SanityError:
        pass
    # キメラ・分離腕もランナーを通る（logs の payload・moments・ckpt・provenance）
    smax = scr_run('chSMAXH_1216', 20_000)
    scr_run('chGD_1216', 2_000)
    for arm, act in (('chSMAXH_1216', 'SMAXH'), ('chGD_1216', 'GD')):
        with np.load(smax / 'logs' / f'{arm}_seed0.npz', allow_pickle=True) as z:
            assert (str(z['activation']), float(z['act_alpha']), str(z['family'])) == (act, 0.1, 'chimera')
            assert np.isfinite(z['layer1_m_dphiddphi']).all() and np.isfinite(z['unfit']).all()
            assert float(z['lr_used']) == 0.01 and str(z['init_hook']) == ''
        assert (smax / 'ckpts' / f'{arm}_step0.pt').exists()
        prov = json.loads((smax / 'arm_status' / f'{arm}_provenance.json').read_text())
        assert prov['git_hash'] and prov['sha256']['src/act_chimera_scr_0913.py'] and prov['act_name'] == act
        assert prov['machine']['hostname'] and prov['peak_rss_kib'] > 0
    rec['G1_scr_short'] = dict(values=vals, controls=dict(
        generator_offset_1_unfit_differs=[r['unfit_differs'] for r in c['rows']],
        generator_offset_1_maxabs=[r['maxabs'] for r in c['rows']],
        smoke_rule_n_records_31_rejects_21=not g['pass_']))


# ------------------------------------------------------------------ S19: 安定余裕（λ_out の再構成・冪乗法）
def check_S19_scr(rec):
    scr_env()
    vals, ctl = {}, {}
    for log, ck in (('LRnull_1216_seed0.npz', 'LRnull_1216_step1000000.pt'),
                    ('Enull_1216_seed3.npz', 'Enull_1216_step1000000.pt')):
        r = SC.s19_stability(SCR_REF / 'logs' / log, SCR_REF / 'ckpts' / ck)
        assert r['pass_'] and r['control_fails'], r
        assert r['err64'] <= r['tol64'] and r['err32'] <= r['tol32'] and r['zmax_ok']
        vals[log] = {k: r[k] for k in ('lam_direct', 'err64', 'tol64', 'err32', 'tol32', 'spec_tol_8eps',
                                       'zmax_err', 'zmax_bound_min')}
        ctl[log] = dict(diag_approx_err=r['control_diag_err'], tol=r['tol32'])
    # 冪乗法と eigvalsh（小さい系列: R = 2・幅 6・パラメータ 133）。対称（主腕）と代替勾配場（分離腕・b を −5 して
    # SMAXH と ELU1 の φ′ が違う帯に z を置く）。許容値は反復の収束幅（Aitken 推定 × 10）と 64·eps64·λ の大きい方。
    S = SC.state_from_ckpt(SCR_REF / 'ckpts' / 'LRnull_1216_step1000000.pt')
    x_in, y = S['x_in'][:, :2].clone(), S['y'][:, :2].clone()
    for act, shift in (('SMAXH', 0.), ('leaky_relu', 0.), ('GD', -5.)):
        net = AC.ChimeraMLPL(2, [6], 20, torch.Generator().manual_seed(1), 'cpu').set_activation(act, 0.1, 'alpha_exp')
        with torch.no_grad():
            net.bs[0] += shift
        params = (net.Ws[0], net.bs[0], net.v, net.c)
        Jd = SC.dense_jacobian_t(net, params, x_in, y)
        asym = float(np.abs(Jd - Jd.T).max())
        ev = np.linalg.eigvals(Jd)
        lam_d = ev[np.argmax(np.abs(ev))]
        pi = SC.power_iteration(net, params, x_in, y)
        tol = max(10. * pi['error_estimate'], 64. * EPS[F64] * abs(lam_d))
        err = abs(pi['lam'][0] - lam_d)
        assert pi['converged'] and err <= tol, (act, err, tol, lam_d)
        one = SC.power_iteration(net, params, x_in, y, max_iter=1)
        assert abs(one['lam'][0] - lam_d) > tol, (act, one['lam'][0], lam_d)
        vals[f'power_vs_dense_{act}'] = dict(lam_dense=complex(lam_d).real, lam_power=float(pi['lam'][0]),
                                             err=float(err), tol=float(tol), iterations=pi['iterations'],
                                             last_delta=pi['last_delta'], error_estimate=pi['error_estimate'],
                                             jacobian_asymmetry=asym, imag=float(complex(lam_d).imag))
        ctl[f'power_1_iteration_{act}'] = float(abs(one['lam'][0] - lam_d))
    rec['S19'] = dict(values=vals, controls=ctl,
                      note='real 1M state (LRnull seed 0, 2201 params): dense eigvalsh 54.928382466493 vs power '
                           '54.928382466473 (|Δ| 2.0e-11 = error estimate; 17 s, not rerun here)')


# ------------------------------------------------------------------ 読み出しのスキーマ（§8・SCR）と relax_rate
def check_readout_scr(rec):
    scr_env()
    out = scr_run('chSMAXH_1216', 20_000)
    m = SC.build_readout(out, 'chSMAXH_1216')
    arr = m['arrays']
    T, S, U = m['T'], m['S'], m['U']
    assert (T, S, U) == (2, 10, 100) and m['window_clipped'] is True
    want = {'fn_mom': ((T, S, 8, 4, 3), np.float64), 'occ': ((T, S, U, 4), np.float32),
            'mob_band': ((T, S, U, 4), np.float32), 'lam_out': ((T, S), np.float64),
            'lam_full': ((1, S), np.float64), 'relax_rate': ((S,), np.float64)}
    for k, (shape, dt) in want.items():
        assert arr[k].shape == shape and arr[k].dtype == dt, (k, arr[k].shape, arr[k].dtype)
    per_T = sum(int(np.prod(v[1:])) * np.dtype(d).itemsize for v, d in
                ((want['fn_mom'][0], np.float64), (want['occ'][0], np.float32), (want['mob_band'][0], np.float32),
                 (want['lam_out'][0], np.float64)))
    projected_mb = (per_T * 500 + arr['pabs1'][0].nbytes * 500 + 3 * S * 8 + 2 * S * 8) / 1e6
    assert projected_mb <= 24.0, projected_mb
    assert np.allclose(arr['occ'].sum(-1), 1.0, atol=1e-6)
    # 自腕の帯別ゲートの和 = 記録器の mob（同じ状態を float32 の z̄ から再構成: 事前の上限 2^-24·(max|z̄| + 1) + 8 eps64）
    with np.load(out / 'logs' / 'chSMAXH_1216_seed0.npz', allow_pickle=False) as z:
        zb = np.abs(z['layer1_zbar']).max()
    bound = 2. ** -24 * (float(zb) + 1.) + 8 * EPS[F64]
    assert m['mob_recon_maxabs'] <= bound, (m['mob_recon_maxabs'], bound)
    fc = AC.family_contrasts(torch.as_tensor(arr['fn_mom'][-1, 0]))
    gI_H = float(fc['H']['I'][:, 0].sum())
    gI_S = float(fc['S']['I'][:, 0].sum())
    assert abs(gI_H) <= 8 * EPS[F64] * 4 and gI_S > 8 * EPS[F64] * 4, (gI_H, gI_S)
    assert np.isfinite(arr['relax_rate']).all() and (arr['relax_n'] > 0).all()
    assert not arr['near_eos'].any() and np.isfinite(arr['lam_out']).all() and np.isfinite(arr['lam_full']).all()
    assert (out / 'readout_chSMAXH_1216.npz').exists() and m['bytes'] < 1_000_000
    # 対照 (i): キーを 1 つ消したログでは KeyError
    mut = scr_tmp() / 'mut_readout'
    if mut.exists():
        shutil.rmtree(mut)
    shutil.copytree(out / 'logs', mut / 'logs')
    with np.load(mut / 'logs' / 'chSMAXH_1216_seed0.npz', allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files if k != 'layer1_w_free'}
    np.savez_compressed(mut / 'logs' / 'chSMAXH_1216_seed0.npz', **payload)
    try:
        SC.build_readout(mut, 'chSMAXH_1216', write=False, power_iter=False)
        raise AssertionError('missing key not detected')
    except KeyError:
        pass
    # 対照 (ii): タスク終端の直前の記録が 1000 step 前でないと relax_rate が assert で落ちる
    with np.load(out / 'logs' / 'chSMAXH_1216_seed0.npz', allow_pickle=False) as z:
        step, dz, zm, den = z['step'], z['layer1_dzbar'], z['layer1_zmax'], z['layer1_denom']
    r_ok = SC.relax_rate(step, dz, zm, den, [1, 2])
    assert np.isfinite(r_ok['rate']) and r_ok['n_tasks'] == 2
    try:
        SC.relax_rate(step + np.where(step % 10_000 == 0, 0, 500), dz, zm, den, [1, 2])
        raise AssertionError('spacing not asserted')
    except AssertionError as exc:
        assert 'record spacing' in str(exc)
    rec['readout_scr'] = dict(values=dict(shapes={k: list(arr[k].shape) for k in want}, projected_mb_at_T500=projected_mb,
                                          mob_recon_maxabs=m['mob_recon_maxabs'], mob_recon_bound=bound,
                                          Gamma_I_H=gI_H, Gamma_I_S=gI_S, relax=m['relax'][0],
                                          eos_out_median=m['eos_out_median'], lam_full_step=m['lam_full_step']),
                              controls=dict(missing_key='KeyError', spacing='AssertionError'))


# ------------------------------------------------------------------ RSS の外挿（§2.4・§7）
def check_rss_scr(rec):
    scr_env()
    r = SC._rss_extrapolate(800_000, 50_000, [5_000_000, 10_000_000], free_gib=18.0)
    gib = 1024. ** 3
    for h in (5_000_000, 10_000_000):
        assert abs((r['projected_peak_gib'][h] - r['base_gib']) * gib - EL.recorder_bytes(h)) < 1.0
    assert r['worst_peak_gib'] == r['projected_peak_gib'][10_000_000]
    assert r['J_spec'] == max(0, int(18.0 // (1.5 * r['worst_peak_gib'])) - 1)
    same = SC._rss_extrapolate(800_000, 5_000_000, [5_000_000], free_gib=18.0)
    assert abs(same['projected_peak_gib'][5_000_000] - same['peak_rss_gib']) < 1e-12   # 地平線で測れば外挿なし
    assert SC._rss_extrapolate(800_000, 50_000, [5_000_000], free_gib=0.0)['J_spec'] == 0
    # 子プロセスの起動行（--config 無しの edge_law ではなく本モジュール・腕名・--steps・--outdir）。subprocess は偽物。
    calls = []

    class _Fake:
        returncode, stderr, stdout = 0, '\tMaximum resident set size (kbytes): 800000\n', '[chGD_1216] complete in 1.0s\n'

    class _FakeResumed:
        returncode, stderr, stdout = 0, '\tMaximum resident set size (kbytes): 552000\n', '[chGD_1216] complete logs found; nothing to do\n'

    real = SC.subprocess.run
    try:
        SC.subprocess.run = lambda cmd, **kw: (calls.append(cmd), _Fake())[1]
        probe = SC.rss_probe(scr_tmp() / 'rss', arms=('chGD_1216',), steps=50_000)
    finally:
        SC.subprocess.run = real
    cmd = calls[0]
    assert cmd[:2] == ['/usr/bin/time', '-v'] and cmd[3:5] == ['-m', 'src.act_chimera_scr_0913']
    assert cmd[cmd.index('--arm') + 1] == 'chGD_1216' and cmd[cmd.index('--steps') + 1] == '50000'
    assert cmd[cmd.index('--outdir') + 1].endswith('run_chGD_1216')                  # 腕ごとの新しいディレクトリ
    assert probe['horizons'] == [5_000_000, 10_000_000] and probe['pass_']
    assert set(probe['projected_peak_gib']) == {5_000_000, 10_000_000}
    # 対照: 子が学習せずに再開して終わった（'nothing to do'）プローブは pass しない
    try:
        SC.subprocess.run = lambda cmd, **kw: _FakeResumed()
        resumed = SC.rss_probe(scr_tmp() / 'rss', arms=('chGD_1216',), steps=50_000)
    finally:
        SC.subprocess.run = real
    assert resumed['pass_'] is False and resumed['arms']['chGD_1216']['trained'] is False
    rec['rss_scr'] = dict(values=dict(example=r, child_command=cmd), controls=dict(free_0_J=0))


CHECKS_SCR = [check_S9_scr, check_S13_scr, check_S14_scr, check_G1_scr_short, check_S19_scr,
              check_readout_scr, check_rss_scr]
CHECKS.extend(CHECKS_SCR)


def test_S9_scr_rng_sharing():
    check_S9_scr(RECORD)


def test_S13_scr_setup_swap():
    check_S13_scr(RECORD)


def test_S14_scr_loop_copy():
    check_S14_scr(RECORD)


def test_G1_scr_short_bits():
    check_G1_scr_short(RECORD)


def test_S19_scr_stability():
    check_S19_scr(RECORD)


def test_readout_schema_scr():
    check_readout_scr(RECORD)


def test_rss_scr_extrapolation():
    check_rss_scr(RECORD)


# ================================================================== 判定器（act_chimera_report_0913）の検査
# S16 (i)–(vii) と G0.5 の通し（合成の走）は src/test_act_chimera_report_0913.py にある（同じ CHECKS / collect() に入る）。
from src.test_act_chimera_report_0913 import check_S16, check_G05_fixture, CHECKS_REPORT   # noqa: E402

CHECKS.extend(CHECKS_REPORT)


def test_S16_judge():
    check_S16(RECORD)


def test_G05_fixture_report():
    check_G05_fixture(RECORD)


def checks_for_scr_interpreter():
    """SCR の interpreter（proj_004_drift/.venv・§2.1）で G0 に回す部分集合: SCR の検査と、SCR が学習に使う
    act_grad / act_curv の bit 一致・曲率・分離腕（S13・S13b・S15・S17）。"""
    return [check_S13, check_S13b, check_S15, check_S17] + list(CHECKS_SCR)


if __name__ == '__main__':
    subset = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if subset == '--scr':
        SC.assert_env()                                                  # §2.1: この部分集合は .venv で回す
        print(json.dumps(_jsonable(collect(checks_for_scr_interpreter())), indent=1))
    else:
        print(json.dumps(_jsonable(collect()), indent=1))
