"""act_chimera_0913: 活性化キメラの 2×2（spec_act_chimera_0913 §1）。

ELU1 と leaky（a = 0.1）の「0 近傍の傾き」と「深部の傾き」を別々に持つ 6 本の
キメラ（H: SMAXH/SMINH、V: VMIN/VMAX、S: SMAXS/SMINS）、前向きと逆向きで別の
関数を使う分離腕 4 本（GN/FN/GD/FD・§1.5）、leaky の梯子腕 LR02、h = zφ′ − φ の表、
φ″ の表、SCR 用の ``ChimeraMLPL``、箱 B / RL 用の ``make_act`` を置く。

既存モジュールの sha256 は動かさない（§2.2「実装経路」）。
- ELU1 は箱 B では ``EG.ELU(1.0)``（elu_growth_0909:18-23）、SCR では ``VecMLPL`` の
  ``elu``。どちらも §1.3 の ``where(z>0, z, expm1(zc))`` と bit 一致する（S7・S13b）。
- LR は host の ``H.ARMS['LR']``、LR02 は ``H.Activation('LRx', 'leaky', .2)``
  （gate_shape_0911:78-81 の LR03/LR001 と同じ作り方）。
- SMAXH は ``GS.ELUFloor(0.1)`` のインスタンスをそのまま使う（再実装しない。演算順が
  変わると committed 軌道との bit 一致を失う・§1.3）。

§1.3 の実装上の規則
- φ の枝は定数の述語 ``z >= LNA`` / ``z >= ZV`` による ``torch.where`` で選ぶ。φ′ は
  同じ述語の where か clamp で書く。``torch.minimum/maximum`` は同値点で勾配を 50/50
  に割るので使わない。
- 捨てる側の枝も exp/log の引数を定義域に clamp する（しないと backward が
  0·inf = NaN を作る・S5）。
- 全クラスに ``kind`` 属性（``'chimera_<腕名>'``。host のクラスは自分の kind）を置く。
  RL.run_one:121 が読む（S7）。
- h は ``_hneg_elu(z) = z·exp(z) − expm1(z)`` を部品にし、腕名で引く表 ``HNEG`` として
  置く。未知名は KeyError（``EG.hdefect`` のように黙って 0 を返さない・S4）。
- 箱 B と RL の学習勾配は ``phi`` の autograd から来る。``dphi`` は読み出し専用。
  SCR は解析的な ``act_grad`` で学習する（nets.py:576）ので φ′ の正しさが軌道を決める
  （S13b）。EdgeRecorder は φ″（``m_dphiddphi``・edge_law_0905.py:71）を書くので
  ``act_curv`` も置く。
- 分離腕は ``torch.autograd.Function`` で forward = φ_fwd(z)、backward = grad·φ′_bwd(z)。
  ``dphi`` は φ′_bwd（学習の勾配が見るゲート）、h = z·φ′_bwd − φ_fwd（§1.5）。表
  ``PHI/DPHI/DDPHI`` の分離腕の項は、それぞれ φ_fwd / φ′_bwd / φ″_bwd を指す。

使ってはならないもの（S8）: ``EG.make_act``（未知名を黙って leaky にする）、
``EG.hdefect``（未知クラスで黙って 0）、``EG.measure``（中で hdefect を呼ぶ）、
``GS.check_dphi``（格子 [−6, 6] が z_v を含まない）。この module はどれも参照しない。

判定器と 3 つのランナーが共有する読み出しの部品（§3.3）も置く: 帯 B0–B3 の
``band_index`` / ``occupancy`` / ``pabs1``、自腕の帯別ゲート ``mob_band``、8 関数の
帯別モーメント ``fn_mom``、家族ごとの ½ 対比 ``family_contrasts``。

腕ラベルと活性化名は別物である。``make_act`` は活性化名（8 関数 + LR02 + 分離腕 4）だけを
受け、未知名（``'LRtw0'`` や ``'ELUF'`` も含む）は KeyError にする。null 双子のラベルは
``twin_of('LRtw3') == ('LR', 3)`` で分解する。双子の摂動（初期値の b1[k] に +1e−6・追補 2-2）と、その検査 S18
（``twin_init_report``）・生存の判定 S18b（``params_sha256(exclude=…)`` と ``twin_live_task``）も箱 B と RL で共有する。
"""
from __future__ import annotations

import ast
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from src import gate_shape_0911 as GS
from src import elu_growth_0909 as EG
from src import nets as N

H = GS.H

# ------------------------------------------------------------------ §1.1 定数
# float64。50 桁 mpmath で照合済み（test_act_chimera_0913 の S0 が再照合する）。
A = 0.1                                   # leaky の傾き a
LNA = math.log(A)                         # z_c = ln a = -2.3025850929940455
ZV = -9.999545794446535                   # z_v: e^z - 1 = a z の負根 = -10 - W0(-10 e^-10)
EXP_ZV = math.exp(ZV)                     # e^{z_v} = 4.542055534648271e-05
C_SMAXH = 0.9 + 0.1 * LNA                 # 0.6697414907005954
C_SMINH = 0.1 - 0.1 * LNA                 # 0.3302585092994046
C_SMINS = 0.1 * math.log(1. / 11.)        # -0.23978952727983707

FUNCS8 = ('LR', 'ELU1', 'SMAXH', 'SMINH', 'VMIN', 'VMAX', 'SMAXS', 'SMINS')   # fn_mom の軸順
CHIMERA_NAMES = ('SMAXH', 'SMINH', 'VMIN', 'VMAX', 'SMAXS', 'SMINS')
SPLIT = {'GN': ('LR', 'SMAXH'), 'FN': ('SMAXH', 'LR'),       # 腕 -> (前向き φ, 逆向き φ′)
         'GD': ('ELU1', 'SMAXH'), 'FD': ('SMAXH', 'ELU1')}
SPLIT_NAMES = tuple(SPLIT)
FAMILY = {'H': ('SMAXH', 'SMINH'), 'V': ('VMIN', 'VMAX'), 'S': ('SMAXS', 'SMINS')}   # (big, small)
BANDS = ('B0', 'B1', 'B2', 'B3')          # z>0 / z_c<=z<=0 / z_v<=z<z_c / z<z_v
MOM = ('dphi', 'phi', 'h')                # fn_mom の最後の軸
ACT_NAMES = FUNCS8 + ('LR02',) + SPLIT_NAMES

# ------------------------------------------------------------------ host のインスタンス
_LR = H.ARMS['LR']                        # kind 'leaky'（S7: LR ≡ H.ARMS['LR']）
_LR02 = H.Activation('LRx', 'leaky', .2)  # 梯子腕（GS:78-81 と同じ作り方）
_ELU1 = EG.ELU(1.0)                       # kind 'elu'（S7: ELU1 ≡ EG.ELU(1.0)）
_ELUF = GS.ELUFloor(0.1)                  # kind 'elufloor' = SMAXH（S7: 型そのもの）
assert _ELUF.lna == LNA and _ELUF.param == A


# ------------------------------------------------------------------ §1.3 φ・φ′（逐語）
def _phi_SMINH(z):
    return torch.where(z > 0, z, torch.where(z >= LNA, 0.1 * z,
                                             torch.exp(z.clamp(max=LNA)) - C_SMINH))


def _dphi_SMINH(z):
    return torch.where(z > 0, torch.ones_like(z),
                       torch.clamp(torch.exp(z.clamp(max=0.)), max=0.1))


def _phi_VMIN(z):
    return torch.where(z > 0, z, torch.where(z >= ZV, torch.expm1(z.clamp(ZV, 0.)), 0.1 * z))


def _dphi_VMIN(z):
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z >= ZV, torch.exp(z.clamp(ZV, 0.)), torch.full_like(z, 0.1)))


def _phi_VMAX(z):
    return torch.where(z > 0, z, torch.where(z >= ZV, 0.1 * z, torch.expm1(z.clamp(max=ZV))))


def _dphi_VMAX(z):
    return torch.where(z > 0, torch.ones_like(z),
                       torch.where(z >= ZV, torch.full_like(z, 0.1), torch.exp(z.clamp(max=ZV))))


def _phi_SMAXS(z):
    zc = z.clamp(max=0.)
    return torch.where(z > 0, z, 0.1 * zc + 0.9 * torch.expm1(zc))


def _dphi_SMAXS(z):
    return torch.where(z > 0, torch.ones_like(z), 0.1 + 0.9 * torch.exp(z.clamp(max=0.)))


def _phi_SMINS(z):
    # softplus の引数は <= -LNA = 2.3026 なので threshold 20 の段差は起きない（§1.3）
    return torch.where(z > 0, z, 0.1 * F.softplus(z.clamp(max=0.) - LNA) + C_SMINS)


def _dphi_SMINS(z):
    return torch.where(z > 0, torch.ones_like(z), 0.1 * torch.sigmoid(z.clamp(max=0.) - LNA))


class _Chimera:
    """箱 B / RL 用の duck-typed 活性化（host の Activation と同じ phi/dphi の口）。"""
    name: str
    kind: str
    param = A

    def phi(self, z):
        return PHI[self.name](z)

    def dphi(self, z):
        return DPHI[self.name](z)


class SMINH(_Chimera):
    """(0.1, 飽和): 0.1z (z_c<=z<=0), e^z − C_SMINH (z<z_c)。z_c で C¹、0 で C⁰。"""
    name, kind = 'SMINH', 'chimera_SMINH'


class VMIN(_Chimera):
    """(1, 0.1): min(ELU1, 0.1z)。z_v で C⁰（φ′ が 0.1 → e^{z_v} に跳ぶ）、0 で C¹。"""
    name, kind = 'VMIN', 'chimera_VMIN'


class VMAX(_Chimera):
    """(0.1, 飽和): max(ELU1, 0.1z)。z_v で C⁰、0 で C⁰。"""
    name, kind = 'VMAX', 'chimera_VMAX'


class SMAXS(_Chimera):
    """(1, 0.1) 滑らか: 0.1z + 0.9(e^z − 1)。0 で C¹。"""
    name, kind = 'SMAXS', 'chimera_SMAXS'


class SMINS(_Chimera):
    """(0.1, 飽和) 滑らか: 0.1·softplus(z − z_c) + C_SMINS。0 で C⁰（φ′(0⁻) = 1/11）。"""
    name, kind = 'SMINS', 'chimera_SMINS'


_CHIMERA_CLASSES = {c.name: c for c in (SMINH, VMIN, VMAX, SMAXS, SMINS)}

# 名前 -> callable。分離腕の項は φ_fwd / φ′_bwd（§1.5 の読み出し規則）。
PHI = {'LR': _LR.phi, 'LR02': _LR02.phi, 'ELU1': _ELU1.phi, 'SMAXH': _ELUF.phi,
       'SMINH': _phi_SMINH, 'VMIN': _phi_VMIN, 'VMAX': _phi_VMAX,
       'SMAXS': _phi_SMAXS, 'SMINS': _phi_SMINS}
DPHI = {'LR': _LR.dphi, 'LR02': _LR02.dphi, 'ELU1': _ELU1.dphi, 'SMAXH': _ELUF.dphi,
        'SMINH': _dphi_SMINH, 'VMIN': _dphi_VMIN, 'VMAX': _dphi_VMAX,
        'SMAXS': _dphi_SMAXS, 'SMINS': _dphi_SMINS}
for _arm, (_fwd, _bwd) in SPLIT.items():
    PHI[_arm] = PHI[_fwd]
    DPHI[_arm] = DPHI[_bwd]


# ------------------------------------------------------------------ §1.3 φ″（SCR の EdgeRecorder が書く）
def _ddphi_zero(z):
    return torch.zeros_like(z)


def _ddphi_ELU1(z):
    return torch.where(z > 0, torch.zeros_like(z), torch.exp(z.clamp(max=0.)))


def _ddphi_SMAXH(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= LNA, torch.exp(z.clamp(max=0.)), torch.zeros_like(z)))


def _ddphi_SMINH(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= LNA, torch.zeros_like(z), torch.exp(z.clamp(max=LNA))))


def _ddphi_VMIN(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= ZV, torch.exp(z.clamp(ZV, 0.)), torch.zeros_like(z)))


def _ddphi_VMAX(z):
    return torch.where(z > 0, torch.zeros_like(z),
                       torch.where(z >= ZV, torch.zeros_like(z), torch.exp(z.clamp(max=ZV))))


def _ddphi_SMAXS(z):
    return torch.where(z > 0, torch.zeros_like(z), 0.9 * torch.exp(z.clamp(max=0.)))


def _ddphi_SMINS(z):
    s = torch.sigmoid(z.clamp(max=0.) - LNA)
    return torch.where(z > 0, torch.zeros_like(z), 0.1 * s * (1. - s))


DDPHI = {'LR': _ddphi_zero, 'LR02': _ddphi_zero, 'ELU1': _ddphi_ELU1, 'SMAXH': _ddphi_SMAXH,
         'SMINH': _ddphi_SMINH, 'VMIN': _ddphi_VMIN, 'VMAX': _ddphi_VMAX,
         'SMAXS': _ddphi_SMAXS, 'SMINS': _ddphi_SMINS}
for _arm, (_fwd, _bwd) in SPLIT.items():
    DDPHI[_arm] = DDPHI[_bwd]


# ------------------------------------------------------------------ §1.2 / §1.5 h = zφ′ − φ
def _hneg_elu(z):
    """ELU1 の h の負側: z·e^z − (e^z − 1) = e^z(z − 1) + 1。部品（§1.3）。"""
    return z * torch.exp(z) - torch.expm1(z)


def _hneg_LR(z):
    return torch.zeros_like(z)


def _hneg_ELU1(z):
    return _hneg_elu(z.clamp(max=0.))


def _hneg_SMAXH(z):
    return torch.where(z >= LNA, _hneg_elu(z.clamp(max=0.)), torch.full_like(z, C_SMAXH))


def _hneg_SMINH(z):
    # z<z_c: z e^z − (e^z − C_SMINH) = _hneg_elu(z) − (1 − C_SMINH) = _hneg_elu(z) − C_SMAXH
    return torch.where(z >= LNA, torch.zeros_like(z), _hneg_elu(z.clamp(max=LNA)) - C_SMAXH)


def _hneg_VMIN(z):
    return torch.where(z >= ZV, _hneg_elu(z.clamp(ZV, 0.)), torch.zeros_like(z))


def _hneg_VMAX(z):
    return torch.where(z >= ZV, torch.zeros_like(z), _hneg_elu(z.clamp(max=ZV)))


def _hneg_SMAXS(z):
    return 0.9 * _hneg_elu(z.clamp(max=0.))


def _hneg_SMINS(z):
    zc = z.clamp(max=0.)
    return 0.1 * zc * torch.sigmoid(zc - LNA) - 0.1 * F.softplus(zc - LNA) - C_SMINS


def _hneg_GN(z):
    # z·φ′_SMAXH − φ_LR = z(max(e^z, 0.1) − 0.1)（z<z_c で 0）
    zc = z.clamp(max=0.)
    return zc * (torch.clamp(torch.exp(zc), min=0.1) - 0.1)


def _hneg_FN(z):
    # z·φ′_LR − φ_SMAXH: 0.1z − expm1(z) (z>=z_c); C_SMAXH (z<z_c)
    zc = z.clamp(max=0.)
    return torch.where(z >= LNA, 0.1 * zc - torch.expm1(zc), torch.full_like(z, C_SMAXH))


def _hneg_GD(z):
    # z·φ′_SMAXH − φ_ELU1: ELU1 の h (z>=z_c); 0.1z − expm1(z) (z<z_c・z_v で 0 を切り非有界)
    return torch.where(z >= LNA, _hneg_elu(z.clamp(max=0.)),
                       0.1 * z - torch.expm1(z.clamp(max=LNA)))


def _hneg_FD(z):
    # z·φ′_ELU1 − φ_SMAXH: ELU1 の h (z>=z_c); z e^z − 0.1z + C_SMAXH (z<z_c・+∞ に発散)
    zd = z.clamp(max=LNA)
    return torch.where(z >= LNA, _hneg_elu(z.clamp(max=0.)),
                       zd * torch.exp(zd) - 0.1 * zd + C_SMAXH)


HNEG = {'LR': _hneg_LR, 'LR02': _hneg_LR, 'ELU1': _hneg_ELU1, 'SMAXH': _hneg_SMAXH,
        'SMINH': _hneg_SMINH, 'VMIN': _hneg_VMIN, 'VMAX': _hneg_VMAX,
        'SMAXS': _hneg_SMAXS, 'SMINS': _hneg_SMINS,
        'GN': _hneg_GN, 'FN': _hneg_FN, 'GD': _hneg_GD, 'FD': _hneg_FD}


def h(name, z):
    """h(z) = zφ′ − φ（分離腕は z·φ′_bwd − φ_fwd）。z > 0 では全腕 0。未知名は KeyError。"""
    return torch.where(z > 0, torch.zeros_like(z), HNEG[name](z))


# ------------------------------------------------------------------ §1.5 分離腕（箱 B / RL: autograd）
class _SplitFn(torch.autograd.Function):
    """forward = φ_fwd(z)、backward = grad·φ′_bwd(z)。callable は非テンソル引数で渡す。"""

    @staticmethod
    def forward(ctx, z, fwd, bwd):
        ctx.save_for_backward(z)
        ctx.bwd = bwd
        return fwd(z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        return g * ctx.bwd(z), None, None


class SplitArm:
    """分離腕 GN/FN/GD/FD（§1.5）。``phi`` は autograd.Function、``dphi`` は φ′_bwd。"""
    param = A

    def __init__(self, name):
        if name not in SPLIT:
            raise KeyError(name)
        self.name = name
        self.kind = 'chimera_' + name
        self.fwd_name, self.bwd_name = SPLIT[name]
        self._fwd = PHI[self.fwd_name]
        self._bwd = DPHI[self.bwd_name]

    def phi(self, z):
        return _SplitFn.apply(z, self._fwd, self._bwd)

    def dphi(self, z):
        return self._bwd(z)


# ------------------------------------------------------------------ make_act（S8: 未知名は KeyError）
def make_act(name):
    """活性化名 -> 箱 B / RL の活性化オブジェクト。未知名は KeyError（黙って leaky にしない）。"""
    if name == 'LR':
        return _LR
    if name == 'LR02':
        return _LR02
    if name == 'ELU1':
        return _ELU1
    if name == 'SMAXH':
        return _ELUF
    if name in _CHIMERA_CLASSES:
        return _CHIMERA_CLASSES[name]()
    if name in SPLIT:
        return SplitArm(name)
    raise KeyError(name)


def twin_of(label):
    """腕ラベルを (活性化名, 双子番号) に分ける。'LRtw3' -> ('LR', 3)、それ以外 -> (label, None)。"""
    if label.startswith('LRtw') and label[4:].isdigit():
        return 'LR', int(label[4:])
    return label, None


# ------------------------------------------------------------------ null 双子の摂動と生存（追補 2-2・S18・S18b）
TWIN_DELTA = 1e-6          # 追補 2-2: 双子 k は初期値の b1[k] に +1e−6（絶対値）。W1 は触らない
TWIN_PARAM = 1             # params = [W1, b1, W2, b2, W3, b3] の b1（H.init_params の並び）


def _f32_ordinal(x):
    """float32 の値の順序番号（隣り合う表現値で 1 違う・±0 は 0）。2 値の差が、その間の ulp の数になる。"""
    bits = int(np.array(x, dtype=np.float32).view(np.uint32))
    mag = bits & 0x7FFFFFFF
    return -mag if bits & 0x80000000 else mag


def twin_b1_step(old, delta=TWIN_DELTA):
    """float32 の b1[k] = old に float64 で delta を足し、float32 に丸めた値（最も近い表現値・追補 2-2）。
    返り値: ``new``（float32 の値）、``target`` = old + delta（float64）、``realised_delta`` = new − old（float64 で厳密）、
    ``n_ulps``（old から new までの float32 の表現値の数）、``ulp_at_old``、``nearest``（new の両隣の表現値が target に
    new より近くないこと）。"""
    o = np.float32(old)
    target = float(o) + float(delta)
    n = np.float32(target)
    err = abs(float(n) - target)
    lo, hi = np.nextafter(n, np.float32(-np.inf)), np.nextafter(n, np.float32(np.inf))
    nearest = abs(float(lo) - target) >= err and abs(float(hi) - target) >= err
    return dict(old=float(o), new=float(n), target=target, realised_delta=float(n) - float(o),
                n_ulps=_f32_ordinal(n) - _f32_ordinal(o),
                ulp_at_old=float(np.nextafter(o, np.float32(np.inf))) - float(o), nearest=bool(nearest))


def perturb_twin_b1(params, k, delta=TWIN_DELTA):
    """null 双子（追補 2-2）: params[1][k]（b1[k]）を ``twin_b1_step`` の値に置き換える。乱数も他の要素も触らない。
    箱 B の P1 と RL の R2 が同じ関数を呼ぶ（値は CPU の float64 で決め、同じ dtype・device で書き戻す）。"""
    with torch.no_grad():
        b = params[TWIN_PARAM]
        assert b.dtype == torch.float32 and b.dim() == 1, (b.dtype, tuple(b.shape))
        rec = twin_b1_step(float(b[k]), delta)
        b[k] = torch.tensor(rec['new'], dtype=b.dtype, device=b.device)
    return rec


def params_sha256(params, exclude=None):
    """params（各 tensor の float32 の bytes を順に結合）の sha256。``exclude = (j, idx)`` なら params[j][idx] を 0 に
    した CPU の写しで計算する（S18b: 摂動した要素を除いた比較）。exclude = None の値は箱 B の ``_sha_params``・RL の
    ``params_sha256`` と同じ。"""
    h = hashlib.sha256()
    for j, q in enumerate(params):
        a = q.detach().cpu().contiguous().numpy()
        if exclude is not None and j == exclude[0]:
            a = a.copy()
            a[exclude[1]] = 0
        h.update(a.tobytes())
    return h.hexdigest()


def twin_live_task(lr_sha, twin_sha):
    """S18b（追補 2-2）: タスク終端ごとの「摂動した要素を除いた params の sha256」の列を、同じ seed の LR の列と比べ、
    最初に食い違ったタスク（1 起点）を ``twin_live_task`` に返す。食い違わなければ None。比べるのは両方にある
    タスク（短い方の長さ ``n_compared``）。学習は決定論なので、食い違わない双子は LR と同じ軌道にいる（死んだ双子）。"""
    n = min(len(lr_sha), len(twin_sha))
    first = next((t + 1 for t in range(n) if lr_sha[t] != twin_sha[t]), None)
    return dict(twin_live_task=first, n_compared=n)


def twin_init_report(base, twin, k):
    """S18（追補 2-2）: 双子の init ``twin`` が同じ seed の LR の init ``base`` と b1[k] の 1 要素だけ違い、その値が
    float32 で b1[k] + 1e−6（登録値 ``TWIN_DELTA``）に最も近い表現値であること。摂動した要素を 0 にした sha256 が
    両者で一致すること（S18b の除き方が摂動の位置と合っていること）も pass に要る。摂動 0 の双子は n_diff = 0 で
    落ちる（対照）。"""
    where, n_diff = [], 0
    for j, (a, b) in enumerate(zip(base, twin)):
        d = (a.detach().cpu() != b.detach().cpu()).nonzero()
        n_diff += int(d.shape[0])
        where += [(j, tuple(int(i) for i in row)) for row in d]
    old = float(base[TWIN_PARAM].detach().cpu()[k])
    got = float(twin[TWIN_PARAM].detach().cpu()[k])
    want = twin_b1_step(old, TWIN_DELTA)
    exact = bool(n_diff == 1 and where[0] == (TWIN_PARAM, (k,)) and got == want['new'] and want['nearest']
                 and want['n_ulps'] >= 1)
    excl = (TWIN_PARAM, (k,))
    sha_excl_equal = params_sha256(base, excl) == params_sha256(twin, excl)
    return dict(n_diff=n_diff, where=where, index=f'b1[{k}]', old=old, new=got, expected_new=want['new'],
                target=want['target'], registered_delta=TWIN_DELTA, realised_delta=got - old,
                expected_realised_delta=want['realised_delta'],
                n_ulps=_f32_ordinal(np.float32(got)) - _f32_ordinal(np.float32(old)), ulp_at_old=want['ulp_at_old'],
                nearest=want['nearest'], exact_step=exact, init_sha_excl_b1k_equal=bool(sha_excl_equal),
                pass_=bool(exact and sha_excl_equal))


# ------------------------------------------------------------------ SCR（§2.4 経路 1）
class ChimeraMLPL(N.VecMLPL):
    """``VecMLPL`` にキメラ 6 名と分離腕 4 名を足す。それ以外の名前は super() に委ねる。

    ``setup_arm_chimera`` が ``set_activation`` の直前に ``st["net"].__class__ = ChimeraMLPL``
    と 1 行入れて使う（乱数も状態も触らない・S13）。キメラ名・分離腕名の dial は leaky の
    傾き a = 0.1 を書く規約（§2.4）で、別の値は ValueError。
    分離腕: act_fn = φ_fwd、act_grad = φ′_bwd、act_curv = φ″_bwd（§1.5・S15・S17）。
    """
    CHIMERA_ACTIVATIONS = CHIMERA_NAMES + SPLIT_NAMES
    ACTIVATIONS = N.VecMLPL.ACTIVATIONS + CHIMERA_NAMES + SPLIT_NAMES

    def set_activation(self, act, act_alpha=1.0, act_grad_form=None):
        if act in self.CHIMERA_ACTIVATIONS:
            if float(act_alpha) != A:
                raise ValueError(f"{act} requires dial == {A} (leaky slope a), got {act_alpha!r}")
            if act_grad_form is not None:
                if act_grad_form != "alpha_exp":
                    raise ValueError(f"{act} supports act_grad_form 'alpha_exp' only, "
                                     f"got {act_grad_form!r}")
                self.act_grad_form = str(act_grad_form)
            self.act = str(act)
            self.act_alpha = float(act_alpha)
            return self
        return super().set_activation(act, act_alpha, act_grad_form)

    def act_fn(self, pre):
        if self.act in self.CHIMERA_ACTIVATIONS:
            return PHI[self.act](pre)
        return super().act_fn(pre)

    def act_grad(self, pre, a):
        if self.act in self.CHIMERA_ACTIVATIONS:
            return DPHI[self.act](pre)
        return super().act_grad(pre, a)

    def act_curv(self, pre):
        if self.act in self.CHIMERA_ACTIVATIONS:
            return DDPHI[self.act](pre)
        return super().act_curv(pre)


# ------------------------------------------------------------------ numpy 版（判定器用・§3.4 SCR）
def _np(fn, z):
    arr = np.asarray(z, dtype=np.float64)
    out = fn(torch.as_tensor(arr))
    return out.numpy() if arr.ndim else float(out)


def phi_np(name, z):
    """φ の numpy 版（float64・同じ torch 式で評価する）。分離腕は φ_fwd。"""
    return _np(PHI[name], z)


def dphi_np(name, z):
    """φ′ の numpy 版。分離腕は φ′_bwd。"""
    return _np(DPHI[name], z)


def h_np(name, z):
    return _np(lambda t: h(name, t), z)


# ------------------------------------------------------------------ §3.3 読み出しの部品（腕に依らない帯）
def band_index(z):
    """B0 = z>0, B1 = z_c<=z<=0, B2 = z_v<=z<z_c, B3 = z<z_v。述語は**渡された z の dtype** で評価する。

    規約（全ランナー共通・provenance の ``band_predicate_dtype``）: 箱 B と RL のランナーは forward の
    float32 の z を float64 に写してから帯を割り当てる（§2.2 P4 の例 ``zs1 = (...).double()``）。一方 φ/φ′ の
    枝は float32 の述語で選ばれる。両者は float32 の継ぎ目の値でだけ食い違う: f32(z_c) = −2.3025851249694824 は
    float32 では z >= LNA（B1）、float64 に写すと z < z_c（B2）。f32(z_v) も B2 対 B3。測度 0 の差で、
    帯分率・mob_band・fn_mom の値には効かない（S12 の恒等式は同じ述語の中で閉じる）。"""
    return torch.where(z > 0, 0, torch.where(z >= LNA, 1, torch.where(z >= ZV, 2, 3)))


def band_masks(z):
    """(4, *z.shape) の float64 指示関数。"""
    k = band_index(z)
    return torch.stack([(k == i).double() for i in range(4)])


def occupancy(z):
    """帯分率 p_{Bk,i}: z (n_inputs, U) -> (U, 4) float64（保存時に float32 へ落とす・§8）。"""
    return band_masks(z).mean(1).T


def pabs1(z):
    """P[|z| < 1]: (n_inputs, U) -> (U,) float64。"""
    return (z.abs() < 1).double().mean(0)


def mob_band(dphi_z, z):
    """帯別のゲート寄与 mob_k = E_x[φ′(z)·1(B_k)]（自腕の φ′。Σ_k mob_k = mob・S12）。
    dphi_z: 自腕の dphi(z)（分離腕は φ′_bwd）。(n_inputs, U) -> (U, 4) float64。"""
    g = dphi_z.double()
    return (band_masks(z) * g).mean(1).T


def fn_mom(z, funcs=FUNCS8):
    """8 関数の帯別モーメント: (len(funcs), 4, 3) float64。
    [f, k, q] = mean over (unit, 入力) of q_f(z)·1(z ∈ B_k)、q ∈ (φ′, φ, h)。
    指示関数の重み付き平均なので Σ_k は帯なしの平均に等しい（S12）。z は float64 の
    (n_inputs, U)。腕自身の φ は使わず、表の 8 関数を渡された状態の上で評価する（§1.5）。"""
    zd = z.double()
    m = band_masks(zd)                                          # (4, n, U)
    out = torch.empty(len(funcs), 4, 3, dtype=torch.float64)
    for i, f in enumerate(funcs):
        for q, v in enumerate((DPHI[f](zd), PHI[f](zd), h(f, zd))):
            out[i, :, q] = (m * v).mean((1, 2))
    return out


BAND_PREDICATE_DTYPE = 'float64 copy of the float32 forward z (box B / RL); float64 reconstructed z (SCR)'


# ------------------------------------------------------------------ §6 S14: ループの写しの AST 照合（3 ランナー共通）
_COMPOUND = (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)


def ast_body_stmts(src, name):
    """関数 name の本体を AST で文ごとに取り出す（§6 S14）。返り値は '<入れ子の深さ>:<文の source>' のリスト。
    複合文は見出し（':' で終わる行まで）だけ、docstring・コメント・空行は入らない。同じ行の複数文
    （GS:206 の c1/c2）は別々に数える。行頭の字下げは深さで残す（edge_law の ``_body`` が捕まえない穴）。"""
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fns) == 1, (name, len(fns))
    out = []

    def header(node):
        seg = ast.get_source_segment(src, node)
        lines = []
        for line in seg.splitlines():
            lines.append(line.strip())
            if line.rstrip().endswith(':'):
                break
        return ' '.join(lines)

    def walk(stmts, depth, skip_doc=False):
        for i, n in enumerate(stmts):
            if skip_doc and i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
                    and isinstance(n.value.value, str):
                continue
            text = header(n) if isinstance(n, _COMPOUND) else ' '.join(
                l.strip() for l in ast.get_source_segment(src, n).splitlines())
            out.append(f'{depth}:{text}')
            for fld in ('body', 'orelse', 'finalbody'):
                sub = getattr(n, fld, None)
                if not sub:
                    continue
                if fld == 'orelse' and not (isinstance(n, ast.If) and len(sub) == 1 and isinstance(sub[0], ast.If)
                                            and sub[0].col_offset == n.col_offset):
                    out.append(f'{depth}:else:')
                walk(sub, depth + 1)
            for hd in getattr(n, 'handlers', []):
                out.append(f'{depth}:{header(hd)}')
                walk(hd.body, depth + 1)

    walk(fns[0].body, 0, skip_doc=True)
    return out


def apply_registered(host_lines, registered):
    """宿主の本体に登録ブロック（(tag, kind, 錨の行, 新しい行の組)、kind ∈ replace / insert_after / insert_before）を
    当てた期待の写し。錨の行は宿主の中で一意でなければならない。"""
    lines = list(host_lines)
    for tag, kind, anchor, new in registered:
        hits = [i for i, l in enumerate(lines) if l == anchor]
        assert len(hits) == 1, (tag, anchor, hits)
        i = hits[0]
        if kind == 'replace':
            lines[i:i + 1] = list(new)
        elif kind == 'insert_after':
            lines[i + 1:i + 1] = list(new)
        elif kind == 'insert_before':
            lines[i:i] = list(new)
        else:
            raise ValueError(kind)
    return lines


def ast_copy_check(host_src, host_name, mine_src, mine_name, registered, expected_host):
    """S14 の共通形: 写し = 宿主 + 登録ブロック（差分 0）、宿主の文の数 = 実装前に固定した期待値。"""
    import difflib
    host = ast_body_stmts(host_src, host_name)
    mine = ast_body_stmts(mine_src, mine_name)
    expected = apply_registered(host, registered)
    n_ins = sum(len(new) - (1 if kind == 'replace' else 0) for _, kind, _, new in registered)
    ops = [dict(tag=tag, host=host[i1:i2], mine=mine[j1:j2])
           for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, host, mine, autojunk=False).get_opcodes()
           if tag != 'equal']
    diff = [dict(i=i, expected=e, mine=m) for i, (e, m) in enumerate(zip(expected, mine)) if e != m]
    if len(expected) != len(mine):
        diff.append(dict(i=None, expected=len(expected), mine=len(mine)))
    return dict(pass_=bool(not diff and len(host) == expected_host and len(mine) == expected_host + n_ins),
                host=host_name, mine=mine_name, host_stmts=len(host), host_expected=expected_host,
                mine_stmts=len(mine), mine_expected=expected_host + n_ins, n_inserted=n_ins,
                diff_vs_expected=diff, opcodes_vs_host=ops, registered=[t for t, *_ in registered])


def family_contrasts(fm, funcs=FUNCS8):
    """fn_mom の (8, 4, 3) から家族ごとの ½ 対比（§3.3）を帯別に出す。
    返り値 [family][X] は (4, 3)（帯 × {Γ, Φ, Ψ} = {φ′, φ, h} の対比）。
    Γ_N = ½E[big + ELU1 − LR − small]、Γ_D = ½E[big + LR − ELU1 − small]、
    Γ_I = ½E[big + small − LR − ELU1]（H・V で 0、S で > 0）。"""
    ix = {f: i for i, f in enumerate(funcs)}
    lr, elu = fm[ix['LR']], fm[ix['ELU1']]
    out = {}
    for fam, (big, small) in FAMILY.items():
        b, s = fm[ix[big]], fm[ix[small]]
        out[fam] = {'N': 0.5 * (b + elu - lr - s), 'D': 0.5 * (b + lr - elu - s),
                    'I': 0.5 * (b + s - lr - elu)}
    return out
