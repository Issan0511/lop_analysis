"""test_act_chimera_report_0913: 判定器の単体検査 S16（spec_act_chimera_0913 §6）と G0.5 の通し（合成の走）。

``test_act_chimera_0913`` から import され、同じ ``CHECKS`` / ``collect()`` に入る（ファイルを分けたのは、
3 つのランナーの検査が同じファイルに並行して足されているため）。

S16 (i) 陽性: committed の gate_shape_0911 の ELUF − LR が両 endpoint で BEATS 3/3（σ_traj = 0）。対照: 腕名の入れ替え。
    (ii) 合成系列: §3.2 の推定量そのものの到達率（4000 組・rng 20260914）が登録区間に入る。対照: m = SE/2 は外。
    (iii) 床: 0906 の R seed 0–9 は全部 AT_FLOOR、LR は全部そうでない。F_seed と 0.00026 の導出も照合。
    (iv) 用量と存在: z = −40 の擬キメラ・z = −8 の擬 big・擬 GD・SCR の空帯。算術（dose_gate）に加え、擬キメラの
         fn_mom を judge_family / judge_d5a / scr_existence に**端から端まで**通し、verdict の行で NOT_TESTABLE_* を
         確かめる。対照: 同じ入力で gate_eq を「常に通る」に差し替えると INERT / EQUALS / STAYS になる。
    (v) 仮説の写像: (0, 0, 0, −2) が REFUTED。small := LR + ノイズが SUPPORTED にならない。
    (vi) sd = 0 / NaN が例外を投げる。
    (vii) §0.4-1・§0.4-3 の SE 表を小数 2 桁で再現。対照: 全体平均の ACF は ELU1 − ELUF seed 2 の L で 0.19。
    回帰（改訂のレビューで見つかった配線の誤り）: D6 の段を Ḡ で選ぶ（§0.4-6 の残差を再現）、D1 の EQ は対の用量、
    ρ_min の出所（錨は判定する対比の m・非錨は箱 B H の同じ対比）、(WEAK_DOSE) の合わせ方、双子の未完で σ = 0、
    RL の床の D1、SCR の D5a の落ちたチャネル、(M_SE_ONLY) の付与、双子の生存（追補 2-2: 窓の最初のタスクまでに
    食い違わない双子を除く・生きた対が 2 未満なら M_SE_ONLY・σ² = 0 は clipped の注記で OK）。
G0.5（合成の走）: §8 のスキーマで作った合成出力に判定器を端から端まで掛け、全行が値を持つこと。対照: キーを 1 つ消すと落ちる。

一時ファイルは ``ACT_CHIMERA_SCRATCH``（無ければ tempfile.mkdtemp・終了時に消す）に書く（セッション固有のパスを持たない）。
"""
from __future__ import annotations

import atexit
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src import act_chimera_0913 as AC
from src import act_chimera_report_0913 as RP
from src import gate_shape_0911 as GS

S16_INTERVALS = {('L', 'EQ'): (0.828, 0.810, 0.846), ('L', 'DIR'): (0.892, 0.878, 0.907),
                 ('A', 'EQ'): (0.856, 0.839, 0.872), ('A', 'DIR'): (0.914, 0.900, 0.927)}   # 中心・下限・上限（§6 S16 (ii)）
LR_WINDOW_SD = {0: (0.226, 0.604), 1: (0.378, 0.561), 2: (0.516, 0.404)}                   # seed -> (base, late) [pt]
SE_TABLE = {('LR', 'ELUF'): ((0.17, 0.26, 0.16), (0.14, 0.09, 0.08)),                       # §0.4-1（L, −A_late）
            ('ELU1', 'ELUF'): ((0.22, 0.31, 0.20), (0.11, 0.12, 0.18)),
            ('ELU1', 'LR'): ((0.20, 0.12, 0.21), (0.13, 0.11, 0.18))}                       # §0.4-3
_SCRATCH = {}


def scratch_root():
    """検査の一時ディレクトリ: 環境変数 ACT_CHIMERA_SCRATCH があればその下、無ければ tempfile.mkdtemp（終了時に消す）。"""
    if 'root' not in _SCRATCH:
        base = os.environ.get('ACT_CHIMERA_SCRATCH')
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix='act_chimera_tests_', dir=base))
        else:
            root = Path(tempfile.mkdtemp(prefix='act_chimera_tests_'))
        atexit.register(shutil.rmtree, root, True)
        _SCRATCH['root'] = root
    return _SCRATCH['root']


IX8 = {f: i for i, f in enumerate(AC.FUNCS8)}


def _env_from_series(series, seeds=(0, 1, 2), env='pmnist'):
    E = RP.EnvData(env, list(seeds))
    for a, d in series.items():
        E.series[a] = {s: np.asarray(d[s], dtype=np.float64) for s in seeds}
        E.status[a] = {s: 'COMPLETE' for s in seeds}
        E.readout[a] = {}
    return E


# ------------------------------------------------------------------ (i)
def _s16_i(rec, lad, W):
    eps = RP.endpoints_pmnist(W)
    out = {}
    for swap in (False, True):
        a, b = ('LR', 'ELUF') if swap else ('ELUF', 'LR')
        E = _env_from_series({'X': {s: lad[a][s]['acc'] for s in range(3)}, 'LR': {s: lad[b][s]['acc'] for s in range(3)}})
        out['swapped' if swap else 'positive'] = {e.tag: RP.eval_contrast(E, e, {'X': 1, 'LR': -1}, 0.).state for e in eps.values()}
    assert out['positive'] == {'L': RP.DIRM, 'A': RP.DIRM}, out
    assert out['swapped'] == {'L': RP.DIRP, 'A': RP.DIRP}, out
    assert RP.d1_label(out['positive']['L'], out['positive']['L']) == 'BIG_BEATS_BOTH'
    assert RP.d1_label(out['swapped']['L'], out['swapped']['L']) == 'BIG_BELOW_BOTH'
    rec['i'] = out


# ------------------------------------------------------------------ (ii)
def _s16_ii(rec, W, n_sets=4000, seed=20260914):
    rng = np.random.default_rng(seed)
    eps = RP.endpoints_pmnist(W)
    nb, nl = W.pm_base[1] - W.pm_base[0] + 1, W.pm_late[1] - W.pm_late[0] + 1
    T = W.pm_late[1]
    rates = {}
    for tag in ('L', 'A'):
        ep = eps[tag]
        for cond in ('EQ', 'DIR'):
            hits = {1.: 0, .5: 0}
            for _ in range(n_sets):
                c, se = [], []
                for s in range(3):
                    sb, sl = LR_WINDOW_SD[s]
                    arms = np.zeros((4, T))                                  # 共通の基準系列 + 腕ごと独立の iid 正規ノイズ
                    arms[:, W.pm_base[0] - 1:W.pm_base[1]] += rng.normal(0., sb, (4, nb))
                    arms[:, W.pm_late[0] - 1:W.pm_late[1]] += rng.normal(0., sl, (4, nl))
                    if cond == 'DIR':
                        true_se = math.sqrt(2 * sb * sb / nb + 2 * sl * sl / nl) if tag == 'L' else math.sqrt(2 * sl * sl / nl)
                        arms[0, W.pm_late[0] - 1:W.pm_late[1]] -= 3. * true_se
                    v, e = RP.endpoint_value(arms[0] - arms[1], ep.spec)
                    c.append(v)
                    se.append(e)
                for scale in hits:
                    hits[scale] += int(RP.contrast_state(c, [scale * x for x in se]) == (RP.EQ if cond == 'EQ' else RP.DIRP))
            for scale in hits:
                rates[(tag, cond, scale)] = hits[scale] / n_sets
    out = {}
    for (tag, cond), (center, lo, hi) in S16_INTERVALS.items():
        r, rm = rates[(tag, cond, 1.)], rates[(tag, cond, .5)]
        assert lo <= r <= hi, (tag, cond, r, (lo, hi))
        assert not (lo <= rm <= hi), (tag, cond, 'm=SE/2 mutation inside interval', rm)
        out[f'{tag}/{cond}'] = dict(rate=r, interval=(lo, hi), center=center, ctl_half_m=rm)
    rec['ii'] = out


# ------------------------------------------------------------------ (iii)
def _s16_iii(rec, W):
    on = {}
    for arm in ('R', 'LR'):
        rows = list(csv.DictReader(open(RP.RL0906 / arm / 'per_task.csv')))
        d = {}
        for r in rows:
            d.setdefault(int(r['seed']), {})[int(r['task'])] = float(r['online_acc'])
        on[arm] = {s: np.array([d[s][t] for t in range(1, 51)]) for s in range(10)}
    FF = {s: RP.rl_floor_F(s, W) for s in range(10)}
    F = {s: v[0] for s, v in FF.items()}
    assert abs(F[0] - 0.11450) < 5e-6 and abs(F[1] - 0.11400) < 5e-6 and abs(F[2] - 0.11383) < 5e-6, F
    fl = {arm: [RP.at_floor(on[arm][s], F[s], W) for s in range(10)] for arm in ('R', 'LR')}
    assert all(fl['R']) and not any(fl['LR']), fl
    sds = [np.std(on['R'][s][30:50] - FF[s][1][30:50], ddof=1) for s in range(10)]
    pooled = float(np.sqrt(np.mean(np.square(sds))))
    assert abs(pooled - 0.00116) < 2e-5 and abs(pooled / math.sqrt(20) - RP.RL_FLOOR_SD) < 1e-5, pooled
    assert abs(RP.RL_FLOOR_Z - 2.865) < 1e-3
    # 対照: 閾値を 10 倍にすると LR も床になる側に動かないが、R の上限を 0 にすると R が床でなくなる
    assert not any(RP.at_floor(on['R'][s], F[s] - 10 * RP.RL_FLOOR_Z * RP.RL_FLOOR_SD, W) for s in range(10))
    rec['iii'] = dict(F=F, R_at_floor=fl['R'], LR_at_floor=fl['LR'], pooled_sd=pooled, z=RP.RL_FLOOR_Z)


# ------------------------------------------------------------------ (iv)
def _gauss_states(arm, seed, W, n=256, rng=None):
    d = np.load(GS.OUT / f'{arm}_s{seed}_units.npz')
    rng = rng or np.random.default_rng(seed)
    out = []
    for t in range(W.pm_late[0], W.pm_late[1] + 1):
        mu, sd = d[f'ref_zcur_i_t{t}'], d[f'ref_sdcur_i_t{t}']
        out.append(torch.as_tensor(mu + sd * rng.standard_normal((n, len(mu)))))
    return out


def _series(states_by_arm, fn):
    T = len(next(iter(states_by_arm.values())))
    return np.array([fn({a: st[t] for a, st in states_by_arm.items()}) for t in range(T)])


def _s16_iv(rec, lad, W):
    eps = RP.endpoints_pmnist(W)
    E = _env_from_series({a: {s: lad[a][s]['acc'] for s in range(3)} for a in ('LR', 'ELU1', 'ELUF')})
    K = {tag: {'D': RP.eval_contrast(E, ep, {'ELU1': 1, 'ELUF': -1}, 0.), 'N': RP.eval_contrast(E, ep, {'LR': 1, 'ELUF': -1}, 0.)}
         for tag, ep in eps.items()}
    rho_min = {tag: {X: RP.rho_min_from(K[tag][X]) for X in ('N', 'D')} for tag in eps}
    assert 0.45 < rho_min['L']['D'] < 0.6 and 0.35 < rho_min['A']['D'] < 0.5, rho_min     # §6: L 約 0.5・−A_late 約 0.4
    d = lambda name: AC.DPHI[name]
    one, pt = torch.ones_like, torch.full_like
    p_big40 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -40., torch.exp(z.clamp(max=0.)), pt(z, .1)))
    p_small40 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -40., pt(z, .1), torch.exp(z.clamp(max=0.))))
    p_big8 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -8., torch.exp(z.clamp(max=0.)), pt(z, .1)))
    out = {'seam40': {}, 'seam40D': {}, 'seam8': {}, 'gd40': {}}
    for s in range(3):
        st = {'LR': _gauss_states('LR', s, W), 'ELU1': _gauss_states('ELU1', s, W), 'big': _gauss_states('ELUF', s, W),
              'small': _gauss_states('LR', s, W, rng=np.random.default_rng(100 + s))}
        # 擬キメラ −40 の家族（big40 = ELU1 の上に −40 で 0.1 の床、small40 = LR の上に −40 で飽和）: 占有された範囲では親と同一。
        # 近傍因子 Γ_N は (−40, 0] で e^z − 0.1 を平均した符号不定の量（存在が落ちる）、深部因子 Γ_D は 0.1·P(z < −40)
        # （ELU1 の深い裾で存在は通り、用量 ρ ≈ 0.03–0.04 で落ちる・§6 S16 (iv)）。
        gamN = lambda fb, fs: (lambda Z: float(np.mean([.5 * float((fb(Z[a]) + d('ELU1')(Z[a]) - d('LR')(Z[a]) - fs(Z[a])).mean()) for a in Z])))
        gamD = lambda fb, fs: (lambda Z: float(np.mean([.5 * float((fb(Z[a]) + d('LR')(Z[a]) - d('ELU1')(Z[a]) - fs(Z[a])).mean()) for a in Z])))
        g40, gH = _series(st, gamN(p_big40, p_small40)), _series(st, gamN(d('SMAXH'), d('SMINH')))
        out['seam40'][s] = dict(gamma=float(g40.mean()), se=RP.se_window(g40), gamma_H=float(gH.mean()))
        g40D, gHD_ = _series(st, gamD(p_big40, p_small40)), _series(st, gamD(d('SMAXH'), d('SMINH')))
        out['seam40D'][s] = dict(gamma=float(g40D.mean()), se=RP.se_window(g40D), gamma_H=float(gHD_.mean()))
        pairD = lambda fb: (lambda Z: float(np.mean([float((fb(Z[a]) - d('ELU1')(Z[a])).mean()) for a in ('big', 'ELU1')])))
        g8, gHD = _series(st, pairD(p_big8)), _series(st, pairD(d('SMAXH')))
        out['seam8'][s] = dict(gamma=float(g8.mean()), se=RP.se_window(g8), gamma_H=float(gHD.mean()))
        gGD = _series({'G': st['ELU1']}, lambda Z: float((p_big40(Z['G']) - d('ELU1')(Z['G'])).mean()))
        gGDr = _series({'B': st['big']}, lambda Z: float((d('SMAXH')(Z['B']) - d('ELU1')(Z['B'])).mean()))
        out['gd40'][s] = dict(gamma=float(gGD.mean()), se=(RP.se_window(gGD) if gGD.std() > 0 else 0.), ref=float(gGDr.mean()))
    labels = {}
    for tag in eps:
        gs = {s: (v['gamma'], v['se']) for s, v in out['seam40'].items()}
        ref = {s: (v['gamma_H'], 0.) for s, v in out['seam40'].items()}
        dz = RP.dose_gate({0: gs}, {0: ref}, rho_min[tag]['N'], 'N')
        labN = RP.gate_eq(RP.EQ, 'N', dz.exist, dz.ok)
        assert labN.startswith('NOT_TESTABLE'), (tag, labN, dz)                        # 近傍: 符号不定の Γ で存在が落ちる
        gsD = {s: (v['gamma'], v['se']) for s, v in out['seam40D'].items()}
        refD = {s: (v['gamma_H'], 0.) for s, v in out['seam40D'].items()}
        dzD = RP.dose_gate({0: gsD}, {0: refD}, rho_min[tag]['D'], 'D')
        assert dzD.exist, ('seam40 deep: existence must pass (ELU1 deep tail), else the control is not demonstrable', gsD)
        labD = RP.gate_eq(RP.EQ, 'D', dzD.exist, dzD.ok)
        assert labD == 'NOT_TESTABLE_JOINT_UNVISITED' and dzD.rho < 0.1, (tag, labD, dzD.rho)   # §6: ρ ≈ 0.03–0.04
        labels[f'seam40/{tag}'] = dict(label_N=labN, exist_N=dz.exist, label_D=labD, rho_D=dzD.rho, rho_min_D=dzD.rho_min,
                                       gamma_D=float(np.median([g for g, _ in gsD.values()])))
        gs = {s: (v['gamma'], v['se']) for s, v in out['seam8'].items()}
        ref = {s: (v['gamma_H'], 0.) for s, v in out['seam8'].items()}
        dz = RP.dose_gate({0: gs}, {0: ref}, rho_min[tag]['D'], 'D')
        assert dz.exist, ('seam8: existence must pass', gs)
        lab = RP.gate_eq(RP.EQ, 'D', dz.exist, dz.ok)
        assert lab == 'NOT_TESTABLE_JOINT_UNVISITED' and 0.2 < dz.rho < 0.45, (tag, lab, dz.rho)
        labels[f'seam8/{tag}'] = dict(label=lab, rho=dz.rho, rho_min=dz.rho_min)
        gs = {s: (v['gamma'], v['se']) for s, v in out['gd40'].items()}
        labels[f'gd40/{tag}'] = dict(exist=RP.existence(gs), rho_vs_B=float(np.median([v['gamma'] for v in out['gd40'].values()])
                                                                       / np.median([v['ref'] for v in out['gd40'].values()])))
    rec['iv'] = dict(arithmetic=labels, rho_min=rho_min)
    _s16_iv_e2e(rec, lad, W)


# ---- (iv) 端から端まで: 擬キメラの fn_mom を判定器の関数に通し、verdict の行を見る
def _fm_series(states, W, slot_dphi=None, T=120):
    """late 窓のタスクの状態（_gauss_states）から fn_mom (T, 2, 8, 4, 3)。slot_dphi = {欄: φ′ の擬関数} は
    その欄の φ′ だけを擬関数で置き換える（fn_mom の 8 欄に無い擬キメラ・擬分離腕を表すため）。層 2 は層 1 の写し。"""
    fm = np.zeros((T, 2, 8, 4, 3))
    for i, z in enumerate(states):
        m = AC.fn_mom(z).numpy()
        if slot_dphi:
            masks = AC.band_masks(z)
            for slot, fn in slot_dphi.items():
                m[IX8[slot], :, 0] = (masks * fn(z).double()).mean((1, 2)).numpy()
        fm[W.pm_late[0] - 1 + i, 0] = m
        fm[W.pm_late[0] - 1 + i, 1] = m
    return fm


def _env_ro(series, readouts, seeds=(0, 1, 2), env='pmnist'):
    E = RP.EnvData(env, list(seeds))
    for a in series:
        E.series[a] = {s: np.asarray(series[a][s], dtype=np.float64) for s in seeds}
        E.status[a] = {s: 'COMPLETE' for s in seeds}
        E.readout[a] = ({s: {'fn_mom': readouts[a][s], 'occ_end': np.zeros(readouts[a][s].shape[:2] + (100, 4), np.float32)}
                         for s in seeds} if a in readouts else {})          # occ_end は V 家族の P(z < z_v) の報告だけが読む
    return E


class _GateRemoved:
    """変異対照: gate_eq を「存在も用量も通る」に差し替える（同じ入力で NOT_TESTABLE が INERT 等に戻ること）。"""

    def __enter__(self):
        self.orig = RP.gate_eq
        RP.gate_eq = lambda state, factor, exist_ok, dose_ok: self.orig(state, factor, True, True)

    def __exit__(self, *a):
        RP.gate_eq = self.orig


def _s16_iv_e2e(rec, lad, W):
    eps, tags = RP.endpoints_pmnist(W), ('L', 'A')
    s2 = {'L': 0., 'A': 0.}
    d = AC.DPHI
    one, full = torch.ones_like, torch.full_like
    p_big8 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -8., torch.exp(z.clamp(max=0.)), full(z, .1)))
    p_big40 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -40., torch.exp(z.clamp(max=0.)), full(z, .1)))
    p_small40 = lambda z: torch.where(z > 0, one(z), torch.where(z >= -40., full(z, .1), torch.exp(z.clamp(max=0.))))
    rng = np.random.default_rng(20260914)
    st = {s: {a: _gauss_states(a, s, W) for a in ('LR', 'ELU1', 'ELUF')} for s in range(3)}
    acc = {a: {s: lad[a][s]['acc'] for s in range(3)} for a in ('LR', 'ELU1', 'ELUF')}

    def noisy(a, sd=0.05):
        return {s: acc[a][s] + rng.normal(0., sd, 120) for s in range(3)}

    # (a) 箱 B H（錨）: LR・ELU1・SMAXH = ELUF は committed、small := LR + ノイズ（状態は LR）
    fmH = {a: {s: _fm_series(st[s][src], W) for s in range(3)} for a, src in
           (('LR', 'LR'), ('ELU1', 'ELU1'), ('SMAXH', 'ELUF'), ('SMINH', 'LR'))}
    EH = _env_ro({'LR': acc['LR'], 'ELU1': acc['ELU1'], 'SMAXH': acc['ELUF'], 'SMINH': noisy('LR', 0.3)}, fmH)
    ctx, RH = RP.DoseCtx(), RP.Report()
    RP.judge_family(EH, 'pmnist', W, 'H', eps, ctx, RH, s2, tags)
    wH = {'N': {'LR': .5, 'SMINH': .5, 'ELU1': -.5, 'SMAXH': -.5}, 'I': {'SMAXH': .5, 'SMINH': .5, 'LR': -.5, 'ELU1': -.5},
          'Lb': {'LR': 1, 'SMAXH': -1}, 'Eb': {'ELU1': 1, 'SMAXH': -1}}
    rec['iv']['_ctx'] = (ctx, {ep.tag: {k: RP.eval_contrast(EH, ep, w, 0.) for k, w in wH.items()} for ep in eps.values()})
    out = {}
    # (b) 深部の継ぎ目 −8 の擬 big を S 家族（非錨）に置く: big := ELU1 + ノイズ（状態 ELUF・SMAXS 欄 = 擬 φ′）、
    #     small の欄は SMINH の φ′（LR の状態）。LR は late で −2 pt（a = big − LR を両 endpoint で DIR にする）。
    #     b = big − ELU1 は EQ で、対 (big, ELU1) の用量（ρ ≈ 0.3 < ρ_min ≈ 0.5/0.4）で NOT_TESTABLE_JOINT_UNVISITED。
    #     4 走平均の Γ_D は SMINH の欄のおかげで ρ ≈ 0.6 を超える（主効果の用量で EQ を読む旧い配線なら EQUALS になる）。
    lr_low = {s: acc['LR'][s] - np.r_[np.zeros(100), np.full(20, 2.)] for s in range(3)}
    # small の欄の φ′ は深部（z < z_c）で −0.2 の合成の関数（物理ではない・4 走平均の Γ_D を対の用量より十分大きくするため）
    p_small_deep = lambda z: torch.where(z > 0, one(z), torch.where(z >= AC.LNA, full(z, .1), full(z, -.2)))
    slots = {'SMAXS': p_big8, 'SMINS': p_small_deep}
    fmS = {a: {s: _fm_series(st[s][src], W, slots) for s in range(3)}
           for a, src in (('LR', 'LR'), ('ELU1', 'ELU1'), ('SMAXS', 'ELUF'), ('SMINS', 'LR'))}   # 擬 big は ELUF の状態（算術の検査と同じ）
    ES = _env_ro({'LR': lr_low, 'ELU1': acc['ELU1'], 'SMAXS': noisy('ELU1'), 'SMINS': noisy('LR')}, fmS)
    K_b1 = {t: RP.eval_contrast(ES, ep, {'SMAXS': 1, 'ELU1': -1}, 0.).state for t, ep in eps.items()}
    K_a1 = {t: RP.eval_contrast(ES, ep, {'SMAXS': 1, 'LR': -1}, 0.).state for t, ep in eps.items()}
    assert K_b1 == {'L': RP.EQ, 'A': RP.EQ} and K_a1 == {'L': RP.DIRM, 'A': RP.DIRM}, \
        ('seam-8 control not demonstrable: b1 must be EQ and a1 DIR- on both endpoints', K_b1, K_a1)
    RS = RP.Report()
    RP.judge_family(ES, 'pmnist', W, 'S', eps, ctx, RS, s2, tags)
    gS = ctx.gates[('pmnist', 'S')]
    d0_eb = {t: RS.get('pmnist', 'S', 0, f'D0_Eb_{t}') for t in tags}
    assert all(v == 'NOT_TESTABLE_JOINT_UNVISITED' for v in d0_eb.values()), d0_eb
    assert all(gS[t]['D'].ok for t in tags), ('4-run D dose must pass here (else the pair-dose wiring is not tested)',
                                             {t: (gS[t]['D'].rho, gS[t]['D'].rho_min) for t in tags})
    d1 = RS.get('pmnist', 'S', 'all', 'D1')
    assert d1 == 'NOT_TESTABLE_JOINT_UNVISITED', d1
    with _GateRemoved():
        RS2 = RP.Report()
        RP.judge_family(ES, 'pmnist', W, 'S', eps, ctx, RS2, s2, tags)
    d1_ctl = RS2.get('pmnist', 'S', 'all', 'D1')
    assert d1_ctl == 'BIG_EQUALS_ELU1', d1_ctl
    out['seam8_pseudo_big'] = dict(D0_Eb=d0_eb, rho_Eb={t: gS[t]['Eb'].rho for t in tags},
                                   rho_min_Eb={t: gS[t]['Eb'].rho_min for t in tags}, D1=d1, control_gate_removed_D1=d1_ctl)
    # (c) 継ぎ目 −40 の擬キメラの家族（V・非錨）: big40 = ELU1 に −40 で 0.1 の床、small40 = LR に −40 で飽和。
    #     D = ½[(ELU1 + small) − (LR + big)] ≈ ノイズで EQ、用量は Γ_D ≈ 0.1·P(z < −40) で落ちる。
    fmV = {a: {s: _fm_series(st[s][src], W, {'VMIN': p_big40, 'VMAX': p_small40}) for s in range(3)}
           for a, src in (('LR', 'LR'), ('ELU1', 'ELU1'), ('VMIN', 'ELU1'), ('VMAX', 'LR'))}
    EV = _env_ro({'LR': acc['LR'], 'ELU1': acc['ELU1'], 'VMIN': noisy('ELU1'), 'VMAX': noisy('LR')}, fmV)
    KD = {t: RP.eval_contrast(EV, ep, {'ELU1': .5, 'VMAX': .5, 'LR': -.5, 'VMIN': -.5}, 0.).state for t, ep in eps.items()}
    assert KD == {'L': RP.EQ, 'A': RP.EQ}, ('seam-40 control not demonstrable: D must be EQ', KD)
    RV = RP.Report()
    RP.judge_family(EV, 'pmnist', W, 'V', eps, ctx, RV, s2, tags)
    d3d = RV.get('pmnist', 'V', 'all', 'D3_D')
    assert d3d == 'NOT_TESTABLE_JOINT_UNVISITED', d3d
    with _GateRemoved():
        RV2 = RP.Report()
        RP.judge_family(EV, 'pmnist', W, 'V', eps, ctx, RV2, s2, tags)
    assert RV2.get('pmnist', 'V', 'all', 'D3_D') == 'INERT', RV2.get('pmnist', 'V', 'all', 'D3_D')
    out['seam40_family'] = dict(D3_D=d3d, rho_D={t: ctx.gates[('pmnist', 'V')][t]['D'].rho for t in tags},
                                control_gate_removed='INERT')
    # (d) 擬 GD（逆向きの差を z < −40 にだけ置く）を judge_d5a に通す: G のチャネルは NOT_TESTABLE_WEAK_MANIPULATION。
    #     対照は本物の GD（φ′_bwd = SMAXH）を同じ状態に置いたもの（ゲートを通って STAYS）。
    # G = GD の系列は ELU1 + 交互の ±0.3 pt（窓平均がほぼ 0・ρ₁ < 0 は 0 に切る）: Y_A − Y_G は両 endpoint で決定的に EQ
    gd_acc = {s: acc['ELU1'][s] + 0.3 * (-1.) ** np.arange(120) for s in range(3)}
    fd_acc = noisy('ELUF', 0.05)

    def d5a_run(slot_fn):
        fm = {'ELU1': {s: _fm_series(st[s]['ELU1'], W) for s in range(3)},
              'SMAXH': {s: _fm_series(st[s]['ELUF'], W) for s in range(3)},
              'GD': {s: _fm_series(st[s]['ELU1'], W, {'SMAXH': slot_fn} if slot_fn else None) for s in range(3)},
              'FD': {s: _fm_series(st[s]['ELUF'], W) for s in range(3)}}
        E5 = _env_ro({'ELU1': acc['ELU1'], 'SMAXH': acc['ELUF'], 'GD': gd_acc, 'FD': fd_acc}, fm)   # GN/FN は無い（N は打ち切り）
        KAG = {t: RP.eval_contrast(E5, ep, {'ELU1': 1, 'GD': -1}, 0.).state for t, ep in eps.items()}
        assert KAG == {'L': RP.EQ, 'A': RP.EQ}, ('pseudo-GD control not demonstrable: Y_A - Y_G must be EQ', KAG)
        R5 = RP.Report()
        RP.judge_d5a(E5, 'pmnist', W, eps, RP.DoseCtx(), R5, s2, tags)
        return R5
    R5 = d5a_run(p_big40)
    gate_rows = {t: R5.get('pmnist', 'H', l, f'D5A_D_GATE_G_{t}') for t in tags for l in (0,)}
    note = next(r['note'] for r in R5.rows if r['item'] == 'D5A_D')
    assert all(v == 'NOT_TESTABLE_WEAK_MANIPULATION' for v in gate_rows.values()), gate_rows
    assert 'G=NOT_TESTABLE_WEAK_MANIPULATION' in note, note
    R5c = d5a_run(None)
    note_c = next(r['note'] for r in R5c.rows if r['item'] == 'D5A_D')
    assert 'G=STAYS' in note_c, note_c
    out['pseudo_GD'] = dict(gate_rows=gate_rows, label=R5.get('pmnist', 'H', 'all', 'D5A_D'), note=note[:120],
                            control_real_GD=R5c.get('pmnist', 'H', 'all', 'D5A_D'))
    # (e) SCR: z̄ をずらして操作する帯（H 近傍 = B1）を空にした再構成で、scr_existence が落ちる（対照: z̄ = −1 で通る）
    def scr_env(zbar):
        E = RP.EnvData('scr', list(range(10)))
        wf = np.random.default_rng(0).normal(0., .3, (100, 5))
        z = torch.as_tensor(RP.scr_state(np.full(100, zbar), wf))
        occ = np.broadcast_to(AC.occupancy(z).numpy().astype(np.float32), (2, 10, 100, 4)).copy()
        fm = np.broadcast_to(AC.fn_mom(z).numpy(), (2, 10, 8, 4, 3)).copy()
        for a in ('LR', 'ELU1', 'SMAXH', 'SMINH'):
            E.status[a] = {s: 'COMPLETE' for s in range(10)}
            E.readout[a] = dict(occ=occ, fn_mom=fm)
        E.extra['win'] = (1, 2)
        return E
    ok_e, G_e, n_e = RP.scr_existence(scr_env(-20.), 'H', 'N')
    ok_c, G_c, n_c = RP.scr_existence(scr_env(-1.), 'H', 'N')
    assert ok_e is False and n_e == 0 and ok_c is True and n_c == 10, (ok_e, n_e, ok_c, n_c)
    out['scr_empty_B1'] = dict(exist=ok_e, n_mass_ok=n_e, control_exist=ok_c, control_n_mass_ok=n_c)
    rec['iv']['end_to_end'] = out


# ------------------------------------------------------------------ (v)
def _s16_v(rec, lad, W):
    Kc = lambda c, st: RP.Contrast('x', {}, list(c), [0.] * 3, [1.] * 3, st)
    simple = {'sL': Kc([0, 0, 0], RP.EQ), 'sE': Kc([0, 0, 0], RP.EQ), 'Lb': Kc([2, 2, 2], RP.DIRP), 'Eb': Kc([2, 2, 2], RP.DIRP)}
    lab, note = RP.hypothesis_label(simple, Kc([0, 0, 0], RP.EQ), RP.DIRM)
    assert lab == 'CANCELLATION_REFUTED', (lab, note)
    # 改訂 2 の字義で SUPPORTED と REFUTED が同時に立った例（4 項 1.5m・両親 0）: いまは SUPPORTED だけ
    s15 = {k: Kc([1.5] * 3, RP.DIRP) for k in ('sL', 'sE', 'Lb', 'Eb')}
    assert RP.hypothesis_label(s15, Kc([0, 0, 0], RP.EQ), RP.EQ)[0] == 'CANCELLATION_SUPPORTED'
    assert RP.hypothesis_label(s15, Kc([0, 0, 0], RP.EQ), RP.DIRM)[0] == 'PARTIAL_CANCELLATION'      # (iii) SUPERADDITIVE
    assert RP.hypothesis_label(s15, Kc([1., 1., 1.], RP.DIRP), RP.EQ)[0] == 'PARTIAL_CANCELLATION'   # (ii) |Δ|+m = 2 > 1.5
    # small := LR + ノイズ（LR の実測の窓 sd）の通し: judge_family で SUPPORTED にならない（対照: −δ を足すと (i) が立つ）
    rng = np.random.default_rng(1)
    out = {}
    for delta in (0., 3.0):
        ser = {'LR': {}, 'ELU1': {}, 'SMAXH': {}, 'SMINH': {}}
        for s in range(3):
            ser['LR'][s], ser['ELU1'][s], ser['SMAXH'][s] = lad['LR'][s]['acc'], lad['ELU1'][s]['acc'], lad['ELUF'][s]['acc']
            noise = np.zeros(120)
            sb, sl = LR_WINDOW_SD[s]
            noise[W.pm_base[0] - 1:W.pm_base[1]] = rng.normal(0., sb, 5)
            noise[W.pm_late[0] - 1:W.pm_late[1]] = rng.normal(0., sl, 20) - delta
            ser['SMINH'][s] = lad['LR'][s]['acc'] + noise
        E = _env_from_series(ser)
        R, ctx = RP.Report(), RP.DoseCtx()
        RP.judge_family(E, 'pmnist', W, 'H', RP.endpoints_pmnist(W), ctx, R, {'L': 0., 'A': 0.}, ('L', 'A'))
        out[delta] = {it: R.get('pmnist', 'H', 'all', it) for it in ('D1', 'D2', 'D3_HEADING', 'HYPOTHESIS')}
    assert out[0.]['D1'] == 'BIG_BEATS_BOTH' and not out[0.]['HYPOTHESIS'].startswith('CANCELLATION_SUPPORTED'), out
    assert out[3.]['D2'] == 'SMALL_BELOW_BOTH' and out[3.]['HYPOTHESIS'].startswith(('CANCELLATION_SUPPORTED', 'PARTIAL_CANCELLATION')), out
    rec['v'] = dict(refuted=lab, noisy_small=out[0.], noisy_small_minus_3pt=out[3.])


# ------------------------------------------------------------------ (vi)
def _s16_vi(rec):
    got = {}
    for name, c in (('sd0', [1., 1., 1., 1., 1.]), ('nan_r', [1., 2., 2., 2., 2.]), ('short', [1., 2.]), ('nonfinite', [1., np.nan, 2.])):
        try:
            RP.se_window(np.array(c))
            raise AssertionError(f'{name}: no exception')
        except RP.DegenerateSeries as e:
            got[name] = str(e)
    for bad in (float('nan'), 0., -1.):
        try:
            RP.m_of(bad, {'a': 1}, 0.)
            raise AssertionError('m_of accepted a bad SE')
        except RP.DegenerateSeries:
            pass
    try:
        RP.contrast_state([1., 2., 3.], [1., float('nan'), 1.])
        raise AssertionError('contrast_state accepted NaN m')
    except RP.DegenerateSeries:
        pass
    assert RP.se_window(np.array([1., 2., 1., 2., 1.])) > 0                           # 対照: 正常な系列は通る
    rec['vi'] = got


# ------------------------------------------------------------------ (vii)
def _s16_vii(rec, lad, W):
    eps = RP.endpoints_pmnist(W)
    out = {}
    for (a, b), (seL, seA) in SE_TABLE.items():
        for tag, want in (('L', seL), ('A', seA)):
            got = [round(RP.endpoint_value(lad[a][s]['acc'] - lad[b][s]['acc'], eps[tag].spec)[1], 2) for s in range(3)]
            assert got == list(want), (a, b, tag, got, want)
            out[f'{a}-{b}/{tag}'] = got
    acf = round(RP.endpoint_value(lad['ELU1'][2]['acc'] - lad['ELUF'][2]['acc'], eps['L'].spec, 'acf')[1], 2)
    assert acf == 0.19 and acf != out['ELU1-ELUF/L'][2], acf                         # 対照: 全体平均の ACF は 0.19 で不一致
    rec['vii'] = dict(values=out, controls=dict(acf_ELU1_ELUF_s2_L=acf))


# ------------------------------------------------------------------ 回帰: レビューで見つかった配線の誤り
def _s16_regressions(rec, W):
    out = {}
    eps, tags = RP.endpoints_pmnist(W), ('L', 'A')
    lad, _ = RP.load_gate_shape_rows(('LR001', 'LR', 'LR03', 'LIN', 'ELUF', 'ELU1'), (0, 1, 2), W)
    # (1) D6: 段は Ḡ で選ぶ。§0.4-6 の 4 段の梯子で ELUF の残差 L −0.81/−0.58/−0.01・−A_late −0.74/−0.61/−0.65、
    #     λ = 0.57/0.46/0.58 を小数 2 桁で再現する（名前で選ぶと LR001 を拾って L −0.91/−0.64/−0.09 になる）。
    E = RP.EnvData('pmnist', [0, 1, 2])
    E.series['SMAXH'] = {s: lad['ELUF'][s]['acc'] for s in range(3)}
    E.rows['SMAXH'] = {s: lad['ELUF'][s]['rows'] for s in range(3)}
    E.status['SMAXH'] = {s: 'COMPLETE' for s in range(3)}
    for a in ('SMINH', 'SMAXS', 'SMINS'):
        E.status[a] = {s: 'MISSING' for s in range(3)}
    E.series['LR'] = {s: lad['LR'][s]['acc'] for s in range(3)}
    E.rows['LR'] = {s: lad['LR'][s]['rows'] for s in range(3)}
    R = RP.Report()
    rungs4 = ('LR001', 'LR', 'LR03', 'LIN')
    RP.judge_d6(E, {a: lad[a] for a in ('LR001', 'LR03', 'LIN')}, W, eps, R, {'L': 0., 'A': 0.}, tags, rungs=rungs4)
    got = {t: [round(r['c'], 2) for r in R.seed_rows if r['contrast'] == 'D6_resid_SMAXH' and r['endpoint'] == t] for t in tags}
    lam = [r['note'] for r in R.seed_rows if r['contrast'] == 'D6_resid_SMAXH' and r['endpoint'] == 'L']
    assert got == {'L': [-0.81, -0.58, -0.01], 'A': [-0.74, -0.61, -0.65]}, (got, lam)
    assert all('LR<SMAXH<LR03' in n for n in lam) and [n.split('lam=')[1][:4] for n in lam] == ['0.57', '0.46', '0.58'], lam
    assert R.get('pmnist', 'H', 1, 'D6_SMAXH') == 'SMAXH_BELOW_LEAKY_LINE_A_ONLY', R.get('pmnist', 'H', 1, 'D6_SMAXH')
    # 合成の Ḡ（LR001 .223・LR .246・LR03 .448・LIN 1.0、Ga = .362）で LR02 を上と下に置く
    G = {'LR001': .223, 'LR': .246, 'LR03': .448, 'LIN': 1.0}
    b_above = RP.ladder_bracket(dict(G, LR02=.40), .362)
    b_below = RP.ladder_bracket(dict(G, LR02=.30), .362)
    assert b_above[:2] == ('LR', 'LR02') and abs(b_above[2] - (.362 - .246) / (.40 - .246)) < 1e-12, b_above
    assert b_below[:2] == ('LR02', 'LR03') and abs(b_below[2] - (.362 - .30) / (.448 - .30)) < 1e-12, b_below
    assert RP.ladder_bracket(dict(G, LR02=.40), 1.2) is None and RP.ladder_bracket(dict(G, LR02=.40), .1) is None
    on_rung = RP.ladder_bracket(dict(G, LR02=.40), .40)
    w_on = RP.ladder_weights('X', *on_rung)
    assert on_rung[2] == 0. and sum(v * v for v in w_on.values()) == 2., (on_rung, w_on)       # hi = lo でも Σw² = 2
    name_based = max(a for a in G if G[a] <= .362)                                                # 対照: 名前の文字列順
    assert name_based == 'LR001' != b_above[0]
    out['D6'] = dict(residuals=got, lambda_notes=lam, lr02_above=b_above, lr02_below=b_below, name_based_control=name_based)
    # (2) (WEAK_DOSE) は状態でなく注記: 合わせる前に外して付け直す
    assert RP.merge_co_primary('HELPS(WEAK_DOSE)', 'HELPS', tags, 'UNRESOLVED') == 'HELPS(WEAK_DOSE)'
    m_ch = RP.merge_co_primary('MOVES(WEAK_DOSE)', 'MOVES', tags, 'CHANNEL_UNRESOLVED')
    assert m_ch == 'MOVES(WEAK_DOSE)' and RP.d5a_label('N', m_ch, 'STAYS') == 'N_VIA_GATE'
    assert RP.merge_co_primary('HELPS', 'HARMS(WEAK_DOSE)', tags, 'UNRESOLVED').startswith('CO_PRIMARY_CONFLICT')  # 対照
    assert RP.merge_co_primary('MOVES', 'OTHER(UNRES)', tags, 'CHANNEL_UNRESOLVED') == 'MOVES_L_ONLY'
    assert RP.d1_label(RP.DIRM + '(WEAK_DOSE)', RP.DIRM) == 'BIG_BEATS_BOTH(WEAK_DOSE)'
    # (3) RL の床の D1: 置き換えるのは BIG_BEATS_BOTH だけ。a だけで決まる BIG_EQUALS_LR は残る
    assert RP.d1_label(RP.EQ, RP.DIRP, floor_parent=True, big_above=True) == 'BIG_EQUALS_LR'
    assert RP.d1_label(RP.DIRM, RP.UNRES, floor_parent=True, big_above=True) == 'BIG_BEATS_BOTH_FLOOR_PARENT'
    assert RP.d1_label(RP.DIRM, RP.DIRM, floor_parent=True, big_above=False) == 'BIG_UNRESOLVED'
    assert RP.d1_label(RP.DIRP, RP.DIRP, floor_parent=True, big_above=True) == 'BIG_UNRESOLVED'
    # (4) 双子が 1 つでも未完なら σ² = None（部分的にプールしない・§3.2）。対照: 全部 COMPLETE なら値が出る
    start = W.pm_base[0]                                                             # 箱 B の窓の最初のタスク t16
    Et = RP.EnvData('pmnist', [0, 1, 2])
    rng = np.random.default_rng(5)
    for a in ('LR', 'LRtw0', 'LRtw1', 'LRtw2', 'LRtw3'):
        Et.series[a] = {s: lad['LR'][s]['acc'] + (0 if a == 'LR' else rng.normal(0., .3, 120)) for s in range(3)}
        Et.status[a] = {s: 'COMPLETE' for s in range(3)}
    _twin_sha_fixture(Et, 4, live_task={})                                             # 12 対すべて t1 で生きている
    s2_ok = RP.twin_sigma2(Et, eps['L'], 4, start)
    Et.status['LRtw0'][1] = 'DIVERGED'
    s2_bad = RP.twin_sigma2(Et, eps['L'], 4, start)
    assert s2_ok[0] is not None and s2_ok[1] == 12 and s2_bad[0] is None and s2_bad[1] == 0, (s2_ok[:2], s2_bad[:2])
    Et.status['LRtw0'][1] = 'COMPLETE'
    # (4b) 追補 2-2・S18b: 窓の最初のタスク（t16）までに LR と食い違わない双子は NOT_DETERMINED_TWIN_DEAD で除く。
    #      t16 で食い違う双子は生きている（境界は ≤）、t17 で初めて食い違う双子と、S18b の記録が無い双子は死んでいる。
    dead = {('LRtw0', 0): None, ('LRtw1', 1): 17, ('LRtw2', 2): 16}
    _twin_sha_fixture(Et, 4, live_task=dead)
    del Et.extra['sha_excl'][('LRtw3', 0)]
    s2_d = RP.twin_sigma2(Et, eps['L'], 4, start)
    lv = Et.extra['twin_live']
    assert s2_d[0] is not None and s2_d[1] == 9, s2_d[:2]
    assert [k for k, v in sorted(lv.items()) if not v['live']] == [('LRtw0', 0), ('LRtw1', 1), ('LRtw3', 0)], lv
    assert lv[('LRtw2', 2)]['twin_live_task'] == 16 and lv[('LRtw1', 1)]['twin_live_task'] == 17
    assert s2_d[2].count('NOT_DETERMINED_TWIN_DEAD') == 3 and 'no S18b record' in s2_d[2], s2_d[2]
    # 死んだ双子の系列を「巨大なずれ」に替えても σ² は変わらない（除かれている）。対照: 生きている扱いにすると変わる
    Et.series['LRtw1'][1] = Et.series['LRtw1'][1] + np.where(np.arange(120) < 20, 40., 0.)
    assert RP.twin_sigma2(Et, eps['L'], 4, start)[0] == s2_d[0]
    _twin_sha_fixture(Et, 4, live_task={('LRtw0', 0): None, ('LRtw3', 0): None})
    assert RP.twin_sigma2(Et, eps['L'], 4, start)[0] > 10. * s2_d[0]
    # (4c) 生きた対が 2 未満 → None（呼び手の行は NOT_DETERMINED_TWIN(M_SE_ONLY)・環境に (M_SE_ONLY)）
    only1 = {(f'LRtw{k}', s): None for k in range(4) for s in range(3)}
    only1.pop(('LRtw2', 1))
    _twin_sha_fixture(Et, 4, live_task=only1)
    s2_1 = RP.twin_sigma2(Et, eps['L'], 4, start)
    assert s2_1[0] is None and s2_1[1] == 1 and 'live pairs 1 < 2' in s2_1[2], s2_1
    Rt = RP.Report()
    RP._twin_rows(Et, 'pmnist', {'L': eps['L']}, 4, Rt, start)
    assert Rt.get('pmnist', 'twins', 'all', 'SIGMA_ARM_L') == 'NOT_DETERMINED_TWIN(M_SE_ONLY)' and 'pmnist' in Rt.m_se_only
    assert Rt.get('pmnist', 'twins', 'LRtw2_s1', 'TWIN_LIVE') == 'TWIN_LIVE'
    assert Rt.get('pmnist', 'twins', 'LRtw0_s0', 'TWIN_LIVE') == 'NOT_DETERMINED_TWIN_DEAD'
    tl_rows = [r for r in Rt.seed_rows if r['contrast'].startswith('twin_live_task_')]
    assert len(tl_rows) == 12 and sum(np.isfinite(r['c']) for r in tl_rows) == 1, tl_rows
    # (4d) 生きた対がちょうど 2 で、推定量の切り詰めで σ² = 0 → SIGMA_TRAJ_OK のまま clipped の注記。
    #      双子 − LR の系列を窓ごとに平均 0 の ±（base 5・late 20 の 5 周期）にすると c_tw = 0 < SE_tw なので max(0, ·) = 0。
    two = {(f'LRtw{k}', s): None for k in range(4) for s in range(3)}
    two.pop(('LRtw0', 0)); two.pop(('LRtw3', 2))
    _twin_sha_fixture(Et, 4, live_task=two)
    pat = np.tile([1., -1., 2., -2., 0.], 24) * 0.3
    for tw, s in (('LRtw0', 0), ('LRtw3', 2)):
        Et.series[tw][s] = Et.series['LR'][s] + pat
    s2_c = RP.twin_sigma2(Et, eps['L'], 4, start)
    assert s2_c[0] == 0. and s2_c[1] == 2 and s2_c[2].startswith('clipped'), s2_c
    Rc = RP.Report()
    RP._twin_rows(Et, 'pmnist', {'L': eps['L']}, 4, Rc, start)
    assert Rc.get('pmnist', 'twins', 'all', 'SIGMA_ARM_L') == 'SIGMA_TRAJ_OK' and 'pmnist' not in Rc.m_se_only
    note_c = next(r['note'] for r in Rc.rows if r['item'] == 'SIGMA_ARM_L')
    assert note_c.startswith('clipped') and 'n_live_pairs=2 sigma2=0.0' in note_c, note_c      # 2-4 A8: verdict.csv でも先頭
    # 対照: 同じ 2 対の base 窓（t16–20）にだけ +5 を足すと c_tw² > SE_tw² で σ² > 0、clipped の注記は無い
    for tw, s in (('LRtw0', 0), ('LRtw3', 2)):
        Et.series[tw][s] = Et.series[tw][s] + np.where((np.arange(120) >= 15) & (np.arange(120) < 20), 5., 0.)
    s2_nc = RP.twin_sigma2(Et, eps['L'], 4, start)
    assert s2_nc[0] > 0. and s2_nc[1] == 2 and 'clipped' not in s2_nc[2], s2_nc
    Rn = RP.Report()
    RP._twin_rows(Et, 'pmnist', {'L': eps['L']}, 4, Rn, start)
    assert next(r['note'] for r in Rn.rows if r['item'] == 'SIGMA_ARM_L').startswith('n_live_pairs=2 ')
    # (4e) 追補 2-2・S18b の前提: LR と双子が別の device・git_hash で回った対は、sha が t1 で食い違っていても生きていない
    #      （CPU と GPU は t1 の 20 セル中 14 で食い違う）。同一性の記録が無い対も同じ。対照: 同一性が等しい対は生きている。
    Et.series = {a: {s: lad['LR'][s]['acc'] + (0 if a == 'LR' else rng.normal(0., .3, 120)) for s in range(3)}
                 for a in ('LR', 'LRtw0', 'LRtw1', 'LRtw2', 'LRtw3')}
    _twin_sha_fixture(Et, 4, live_task={})
    Et.extra['sha_excl_ident'][('LRtw1', 0)]['device'] = 'cuda'
    Et.extra['sha_excl_ident'][('LR', 1)]['git_hash'] = 'e' * 40
    del Et.extra['sha_excl_ident'][('LRtw3', 2)]
    s2_id = RP.twin_sigma2(Et, eps['L'], 4, start)
    lv_id = Et.extra['twin_live']
    dead_id = [key for key, v in sorted(lv_id.items()) if not v['live']]
    assert dead_id == [('LRtw0', 1), ('LRtw1', 0), ('LRtw1', 1), ('LRtw2', 1), ('LRtw3', 1), ('LRtw3', 2)], dead_id
    assert lv_id[('LRtw1', 0)]['why'].startswith('LR/twin provenance mismatch: device') and lv_id[('LRtw1', 0)]['n_compared'] == 0
    assert 'provenance mismatch: git_hash' in lv_id[('LRtw2', 1)]['why'] and 'no run identity' in lv_id[('LRtw3', 2)]['why']
    assert s2_id[1] == 6 and s2_id[2].count('NOT_DETERMINED_TWIN_DEAD') == 6, s2_id[:2]
    # loader の側: RL の形（machine・interpreter）と箱 B の形（env_*）から同じ同一性を組み、device だけ違えば食い違う
    rl_prov = dict(machine=dict(device='cpu', hostname='h', cpu='c'), interpreter=dict(executable='/usr/bin/python3',
                   torch='t', numpy='n'), threads=1, git_hash='g', code_sha256={'a': 'b'})
    Ek = RP.EnvData('rlmnist', [0])
    for arm, pv in (('LR', rl_prov), ('LRtw0', dict(rl_prov, machine=dict(rl_prov['machine'], device='cuda:0')))):
        RP._keep_sha_excl(Ek, arm, 0, dict(pv, checks=dict(s18b=dict(sha_excl_b1={'0': ['x']}))))
    assert Ek.extra['sha_excl_ident'][('LRtw0', 0)]['device'] == 'cuda'
    lv_k = RP.twin_liveness(Ek, 1, start=31)[('LRtw0', 0)]
    assert lv_k['live'] is False and 'device' in lv_k['why'], lv_k
    pm_prov = dict(env_device='cpu', env_hostname='h', env_cpu='c', env_executable='/usr/bin/python3', env_torch='t',
                   env_numpy='n', git_hash='g', code_sha256='d')
    assert RP.run_identity(pm_prov) == dict(RP.run_identity(dict(rl_prov, code_sha256='d')), threads=None)
    out['twins_liveness'] = dict(dead_excluded=s2_d[1], lt2_live=s2_1[2][:60], clipped=s2_c[2][:80], not_clipped=s2_nc[0],
                                 identity_mismatch=dict(n_live_pairs=s2_id[1], why=lv_id[('LRtw1', 0)]['why'][:80]))
    # (5) (M_SE_ONLY) を環境の m 由来のラベルに付ける（NOT_DETERMINED・状態の行には付けない）
    Rm = RP.Report()
    Rm.put('pmnist', 'H', 'all', 'D1', 'BIG_BEATS_BOTH')
    Rm.put('pmnist', 'H', 'all', 'D2', 'NOT_DETERMINED_DIVERGED')
    Rm.put('pmnist', 'arm', 'LR_s0', 'STATUS', 'COMPLETE')
    Rm.put('scr', 'H', 1, 'D1', 'BIG_UNRESOLVED')
    Rm.put('all', 'H', 'all', 'D8', 'CROSS_ENV_NOT_DETERMINED')
    Rm.m_se_only.add('pmnist')
    RP.apply_m_se_only(Rm)
    labs = [r['label'] for r in Rm.rows]
    assert labs == ['BIG_BEATS_BOTH(M_SE_ONLY)', 'NOT_DETERMINED_DIVERGED', 'COMPLETE', 'BIG_UNRESOLVED',
                    'CROSS_ENV_NOT_DETERMINED(M_SE_ONLY)'], labs
    # (6) 退化: m が 1 seed でも 0 以下なら例外、OTHER(...) は決まっていない
    try:
        RP.contrast_state([0., 0., 0.], [0., 1., 1.])
        raise AssertionError('contrast_state accepted m = 0')
    except RP.DegenerateSeries:
        pass
    assert not RP.decided('OTHER(UNRES)') and RP.decided('MOVES(WEAK_DOSE)')
    # (7) SCR の D5a: ゲートの落ちたチャネル（F の操作帯 B1 が空）を「動かない」に数えない
    Ws = RP.Windows(scr_win=(1, 2), scr_win_1m=(1, 2), scr_tail=(1, 2), scr_tasks=2)
    rng = np.random.default_rng(3)
    S_, U = 10, 100

    def scr_d5a(f_has_mass):
        Es = RP.EnvData('scr', list(range(S_)))
        Es.extra['lam_full_source'] = 'MISSING'
        Y = {'LR': np.zeros(S_), 'SMAXH': -1. + 0.05 * rng.standard_normal(S_), 'GN': -1. + 0.05 * rng.standard_normal(S_),
             'FN': np.array([.3, -.2, .1, -.4, .2, -.1, .3, -.3, .05, -.05])}
        for a, y in Y.items():
            Es.series[a] = {s: dict(Y=float(y[s]), Y_1m=float(y[s]), SUB=float(y[s]), ZBAR=float(y[s])) for s in range(S_)}
            Es.status[a] = {s: 'COMPLETE' for s in range(S_)}
            occ = np.zeros((2, S_, U, 4), np.float32)
            fm = np.zeros((2, S_, 8, 4, 3))
            if a != 'FN' or f_has_mass:
                occ[..., 1] = 0.5
            occ[..., 0] = 1. - occ[..., 1]
            fm[:, :, IX8['SMAXH'], 1, 0] = 0.3
            fm[:, :, IX8['SMAXH'], 1, 1] = -0.2
            Es.readout[a] = dict(occ=occ, fn_mom=fm, relax_rate=np.zeros(S_))
        Rs = RP.Report()
        RP.judge_scr(Es, Ws, Rs, key='Y')
        return Rs.get('scr', 'H', 1, 'D5A_N')
    lab_drop, lab_ctl = scr_d5a(False), scr_d5a(True)
    assert lab_drop == 'NOT_TESTABLE_WEAK_MANIPULATION' and lab_ctl == 'N_GATE_DIRECTIONAL', (lab_drop, lab_ctl)
    out.update(merge_weak_dose='HELPS(WEAK_DOSE)', d1_floor='BIG_EQUALS_LR kept', twins_partial=s2_bad[2][:80],
               m_se_only=labs, scr_d5a=dict(dropped=lab_drop, control=lab_ctl))
    # (8) ρ_min の出所（§4 D0）: 錨は判定する対比の m（2·m̃(N)/|K̃_N|）、非錨は箱 B H の同じ対比（DIR± なら 2m̃/|c̃|）
    ctx_rec = rec.get('iv', {}).get('_ctx')
    if ctx_rec is not None:
        ctx, K = ctx_rec
        g = ctx.gates[('pmnist', 'H')]
        for t in tags:
            want_N = RP.rho_min_anchor(K[t]['N'], K[t]['Lb'])
            want_I = max(RP.rho_min_anchor(K[t]['I'], K[t]['Lb']), RP.rho_min_anchor(K[t]['I'], K[t]['Eb']))
            assert g[t]['N'].rho_min == want_N and g[t]['I'].rho_min == want_I, (t, g[t]['N'].rho_min, want_N)
            if RP.is_dir(K[t]['Lb'].state):
                assert want_N == 2. * K[t]['N'].med('m') / abs(K[t]['Lb'].med('c'))
                assert want_N != RP.rho_min_from(K[t]['Lb']), 'control: the old anchor-m formula must differ'
            assert ctx.ref_state[('N', t)]['rho_min_nonanchor'] == RP.rho_min_from(K[t]['N'])
        assert ctx.rho_min('N', 'PT', 'rlmnist') == max(ctx.rho_min('N', 'L', 'pmnist'), ctx.rho_min('N', 'A', 'pmnist'))
        assert ctx.rho_min('I', 'L', 'pmnist') == max(ctx.rho_min('N', 'L', 'pmnist'), ctx.rho_min('D', 'L', 'pmnist'))
        out['rho_min'] = {t: dict(N_anchor=g[t]['N'].rho_min, N_nonanchor=ctx.ref_state[('N', t)]['rho_min_nonanchor'])
                          for t in tags}
    rec['regressions'] = out


def _twin_sha_fixture(E, n_twins, live_task, T=120):
    """合成の S18b の列（追補 2-2）: LR と双子 k の「b1[k] を除いた sha256」。live_task[(双子, seed)] = t なら t で初めて
    食い違い、None なら食い違わない。辞書に無い対は t1 で食い違う。"""
    E.extra['sha_excl'], E.extra['sha_excl_ident'] = {}, {}
    for s in E.seeds:
        E.extra['sha_excl'][('LR', s)] = {str(k): [f'LR{s}k{k}t{t}' for t in range(1, T + 1)] for k in range(n_twins)}
        E.extra['sha_excl_ident'][('LR', s)] = dict(_FIXTURE_IDENT)
        for k in range(n_twins):
            tw = f'LRtw{k}'
            lt = live_task.get((tw, s), 1)
            E.extra['sha_excl'][(tw, s)] = {str(k): [f'LR{s}k{k}t{t}' if (lt is None or t < lt) else f'TW{s}k{k}t{t}'
                                                     for t in range(1, T + 1)]}
            E.extra['sha_excl_ident'][(tw, s)] = dict(_FIXTURE_IDENT)


# 合成の走の同一性（run_identity の形・LR と双子で同じ）
_FIXTURE_IDENT = dict(device='cpu', hostname='fixture-host', cpu='fixture-cpu', threads=1, executable='/usr/bin/python3',
                      torch='2.13.0+cu130', numpy='2.5.2', git_hash='f' * 40, code_sha256='c' * 64)


def check_S16(rec):
    W = RP.Windows()
    lad, _ = RP.load_gate_shape_rows(('LR', 'ELU1', 'ELUF'), (0, 1, 2), W)
    r = {}
    _s16_i(r, lad, W)
    _s16_ii(r, W)
    _s16_iii(r, W)
    _s16_iv(r, lad, W)
    _s16_v(r, lad, W)
    _s16_vi(r)
    _s16_vii(r, lad, W)
    _s16_regressions(r, W)
    r['iv'].pop('_ctx', None)
    rec['S16'] = r


# ------------------------------------------------------------------ G0.5（合成の走）: フィクスチャ
def _gauss_state(rng, mu, sd, n=32):
    return torch.as_tensor(mu[None, :] + sd[None, :] * rng.standard_normal((n, len(mu))))


def _s18b_prov(arm, seed, T, n_twins):
    """合成の S18b の記録（追補 2-2）: LR は k = 0..n−1、双子 k は自分の k。双子は t1 から食い違う（生きている）。"""
    base, twin = AC.twin_of(arm)
    if base != 'LR':
        return {}
    ks = [twin] if twin is not None else list(range(n_twins))
    tag = 'LR' if twin is None else 'TW'
    return dict(s18b=dict(ks=ks, sha_excl_b1={str(k): [f'{tag}{seed}k{k}t{t}' for t in range(1, T + 1)] for k in ks}))


def _pm_prov(arm, seed, T, lr=0.001):
    """合成の走の provenance（ランナーの形の最小限: checks_passed と錨の g1_pm と LR・双子の S18b）。"""
    base, twin = AC.twin_of(arm)
    anchored = base in ('LR', 'ELU1', 'SMAXH') and twin is None and lr == 0.001
    g1 = dict(anchor=f'fixture/{base}', n_compared=17 * T, n_expected=17 * T, pass_=True) if anchored else dict(anchor=None)
    return dict(run_id='fixture', arm=arm, seed=seed, tasks=T, status='COMPLETE', lr=lr, checks_passed=True,
                failed_checks=[], checks=dict(g1_pm=g1, **_s18b_prov(arm, seed, T, RP.N_TWINS['pmnist'])))


def make_fixture(root, W, seeds=(0, 1, 2), rng=None, U=100, floor_rl=True, rl_stage2=False,
                 envs=('pmnist', 'rlmnist', 'scr'), pm_stage2=False):
    """§8 のスキーマで 3 環境の合成出力を root に書く（G0.5 の通し・判定器の経路検査）。
    envs で書く環境を選び、pm_stage2 なら stage2/pmnist_lr{5e-4,2e-3} に H 家族 4 腕も書く（D9 の経路）。"""
    rng = rng or np.random.default_rng(7)
    root = Path(root)
    lad, _ = RP.load_gate_shape_rows(('LR', 'ELU1', 'ELUF', 'LR001', 'LR03', 'LIN'), list(seeds), RP.Windows())
    T = W.pm_tasks
    pm_dirs = []
    if 'pmnist' in envs:
        pm_dirs.append((root / 'pmnist', RP.pm_arms(), 0.001, 0.))
    if pm_stage2:
        pm_dirs += [(root / 'stage2' / f'pmnist_lr{t}', ('LR', 'ELU1', 'SMAXH', 'SMINH'), float(t), sh)
                    for t, sh in (('5e-4', -0.2), ('2e-3', 0.3))]
    for pm, arms, lr_run, shift in pm_dirs:
        pm.mkdir(parents=True, exist_ok=True)
        for arm in arms:
            base, _ = AC.twin_of(arm)
            for s in seeds:
                src = {'LR': 'LR', 'ELU1': 'ELU1', 'SMAXH': 'ELUF'}.get(base, 'LR')
                acc = lad[src][s]['acc'][:T] / 100. + shift / 100.
                gbar = lad[src][s]['gbar'][:T]
                off = {'SMINH': -0.6, 'VMIN': -0.3, 'VMAX': 0.1, 'SMAXS': -0.5, 'SMINS': 0.3, 'GN': -0.4, 'FN': -0.1,
                       'GD': -0.3, 'FD': 0.0, 'LR02': 0.2}.get(base, 0.)
                if arm.startswith('LRtw'):
                    acc = acc + rng.normal(0., 0.001, T)
                elif base not in ('LR', 'ELU1', 'SMAXH'):
                    acc = acc + off / 100. + rng.normal(0., 0.003, T)
                    gbar = gbar + {'SMINH': -0.05, 'LR02': 0.1, 'SMAXS': 0.15, 'SMINS': -0.03}.get(base, 0.) + rng.normal(0., 0.002, T)
                rows = [dict(task=t + 1, step=625, acc=float(acc[t]), gbar=float(gbar[t]), arm=arm, seed=s) for t in range(T)]
                with (pm / f'{arm}_s{s}_rows.csv').open('w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                ro = _readout_arrays(rng, base, T, U, with_E=True, seed=s, arm=arm)
                np.savez(pm / f'{arm}_s{s}_readout.npz', **ro)
                (pm / f'{arm}_s{s}_provenance.json').write_text(json.dumps(_pm_prov(arm, s, T, lr_run)))
    # ---- RL
    rl = root / 'rlmnist'
    Tr = W.rl_tasks
    for arm in (RP.rl_arms(rl_stage2) if 'rlmnist' in envs else ()):
        base, _ = AC.twin_of(arm)
        for s in seeds:
            d = rl / arm / f's{s}'
            d.mkdir(parents=True, exist_ok=True)
            F, fr = RP.rl_floor_F(s, W)
            if floor_rl and base in ('ELU1', 'SMINH'):
                on = fr[:Tr] + 0.0001 + rng.normal(0., 0.0002, Tr)
            else:
                lvl = {'LR': 0.80, 'SMAXH': 0.83}.get(base, 0.8)
                on = np.clip(lvl + rng.normal(0., 0.01, Tr) + (rng.normal(0., 0.0005, Tr) if arm.startswith('LRtw') else 0.), 0.01, 0.99)
            with (d / 'per_task.csv').open('w') as fh:
                fh.write('arm,seed,lr,task,iv,online_acc,memo_acc,acc\n')
                for t in range(Tr):
                    fh.write(f'{arm},{s},0.001,{t + 1},none,{on[t]:.10g},{on[t]:.10g},{on[t]:.10g}\n')
            ro = _readout_arrays(rng, base, Tr, U, with_E=False, seed=s, arm=arm, deep=(base in ('ELU1', 'SMINH')) and floor_rl)
            np.savez(d / f'readout_s{s}.npz', **ro)
            (d / 'provenance.json').write_text(json.dumps(dict(run_id='fixture', arm=arm, seed=s, status='COMPLETE',
                                                               checks_passed=True,
                                                               checks=_s18b_prov(arm, s, Tr, RP.N_TWINS['rlmnist']))))
    # ---- SCR（logs だけ・readout は判定器が作る）
    if 'scr' not in envs:
        return root
    sc = root / 'scr'
    (sc / 'logs').mkdir(parents=True, exist_ok=True)
    (sc / 'arm_status').mkdir(parents=True, exist_ok=True)
    Ts = W.scr_tasks
    n_rec = Ts * 10 + 1
    step = np.arange(n_rec) * RP.SCR_EVERY
    for name, arm in RP.SCR_ARMS.items():
        for s in range(10):
            lvl = {'LR': -2.6, 'ELU1': -2.7, 'SMAXH': -3.0, 'SMINH': -1.5, 'GN': -2.7, 'FN': -2.7, 'GD': -2.8, 'FD': -2.7}.get(name, -2.5)
            unfit = 10. ** (lvl + rng.normal(0., 0.3, n_rec) + 0.5 * np.exp(-step / 20000.))
            mu = rng.uniform(-4., 0.5, U)
            zbar = mu[None, :] + 0.3 * rng.standard_normal((n_rec, U)).cumsum(0) / np.sqrt(np.arange(1, n_rec + 1))[:, None]
            wf = rng.normal(0., 0.6, (Ts + 1, U, 5))
            zmax = zbar + 1.5
            dz = np.diff(zbar, axis=0, prepend=zbar[:1]) - 0.05 * (zmax - np.median(zmax, 0)[None, :])
            np.savez(sc / 'logs' / f'{arm}_seed{s}.npz', step=step, unfit=unfit.astype(np.float64),
                     layer1_zbar=zbar.astype(np.float32), layer1_dzbar=dz.astype(np.float32), layer1_zmax=zmax.astype(np.float32),
                     layer1_denom=np.full((n_rec, U), 0.5, dtype=np.float32), layer1_w_free=wf.astype(np.float32),
                     layer1_w_free_step=(np.arange(Ts + 1) * RP.SCR_PERIOD).astype(np.int64))
        (sc / 'arm_status' / f'{arm}_done.json').write_text(json.dumps(dict(status='COMPLETE', arm=arm)))
    return root


def _readout_arrays(rng, base, T, U, with_E, seed, arm, deep=False):
    mu = {0: rng.uniform(-6., 1., U), 1: rng.uniform(-5., 1., U)}
    if deep:
        mu = {0: rng.uniform(-60., -20., U), 1: rng.uniform(-150., -40., U)}
    sd = {0: np.full(U, 1.0), 1: np.full(U, 1.2)}
    dphi = AC.DPHI[base]
    fn = np.zeros((T, 2, 8, 4, 3))
    occ_s = np.zeros((T, 2, U, 4), np.float32)
    occ_e = np.zeros((T, 2, U, 4), np.float32)
    mob = np.zeros((T, 2, U, 4), np.float32)
    pab = np.zeros((T, 2, U), np.float32)
    zs = np.zeros((T, 2, U))
    ze = np.zeros((T, 2, U))
    for t in range(T):
        for l in (0, 1):
            m_s = mu[l] + 0.02 * t + 0.05 * rng.standard_normal(U)
            m_e = m_s + 0.02 + 0.05 * rng.standard_normal(U)
            z_s, z_e = _gauss_state(rng, m_s, sd[l]), _gauss_state(rng, m_e, sd[l])
            fn[t, l] = AC.fn_mom(z_e).numpy()
            occ_s[t, l] = AC.occupancy(z_s).numpy()
            occ_e[t, l] = AC.occupancy(z_e).numpy()
            mob[t, l] = AC.mob_band(dphi(z_e), z_e).numpy()
            pab[t, l] = AC.pabs1(z_e).numpy()
            zs[t, l], ze[t, l] = m_s, m_e
    sha = np.array([f'{arm}{seed}{t:03d}'.ljust(64, 'a') if not (arm.startswith('LRtw') and t == 0) else f'LR{seed}{t:03d}'.ljust(64, 'a')
                    for t in range(T)], dtype='<U64')
    ro = dict(fn_mom=fn, occ_start=occ_s, occ_end=occ_e, mob_band=mob, pabs1=pab, zbar_start=zs, zbar_end=ze,
              depth_vel=(ze - zs) + 0.01 * rng.standard_normal((T, 2, U)),
              step_num=np.abs(rng.normal(0.3, 0.05, (T, 2, U))), wt_norm=np.abs(rng.normal(8., 0.5, (T, 2, U))),
              mu_comp=rng.normal(0., 1., (T, U)), adam_ratio=np.abs(rng.normal(0.5, 0.1, (T, 2, U))).astype(np.float32),
              adam_epsfrac=rng.uniform(0., 0.1, (T, 2, U)).astype(np.float32),
              gbar_l2=rng.uniform(0.2, 0.5, (T, U)), mu_phi1=rng.normal(0., 0.5, (T, U)), param_sha=sha)
    if with_E:
        ro['E_ident'] = np.stack([np.ones(3) * 0.1, np.ones(3) * 0.1, np.zeros(3)], 1)[:min(3, T)] if T < 6 else \
            np.stack([np.ones(6) * 0.1, np.ones(6) * 0.1, np.zeros(6)], 1)
    return ro


def _mini_windows():
    return RP.Windows(pm_base=(3, 7), pm_late=(11, 20), pm_issa_base=(2, 6), pm_issa_late=(16, 20), pm_tasks=20,
                      rl_win=(6, 15), rl_early=(1, 5), rl_tasks=15, scr_win=(10, 12), scr_win_1m=(4, 6), scr_tail=(8, 12),
                      scr_tasks=12, e_ident_tasks=(10, 20))


def _run_report(root, W_flag, extra=()):
    out = Path(root) / 'report'
    return RP.main(['--root', str(root), '--out', str(out)] + list(W_flag) + list(extra)), out


def check_G05_fixture(rec):
    """合成の走に判定器を端から端まで掛ける: 全行が値を持ち、§8 のスキーマ照合が働く（対照: キーを 1 つ消す）。"""
    SCRATCH = scratch_root() / 'report_fixture'
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    # (a) スモークの地平（箱 B t1–3・RL 1 タスク・SCR 3 タスク）
    smoke = make_fixture(SCRATCH / 'smoke', RP.Windows.smoke(), floor_rl=False)
    R, out = _run_report(smoke, ['--smoke'])
    n = RP.all_rows_have_values(R)
    labs = {r['item']: r['label'] for r in R.rows}
    assert (out / 'verdict.csv').exists() and (out / 'seed_contrasts.csv').exists() and (out / 'report_provenance.json').exists()
    assert n > 100 and 'D8' in labs and 'D9' in labs, (n, sorted(labs)[:20])
    assert all(not v.startswith('NOT_DETERMINED_MISSING') for k, v in labs.items() if k in ('D1', 'D2', 'HYPOTHESIS')), labs
    # (b) 短縮した主窓（判定の算術・D0・D6・D7・床の経路を通す）
    Wm = _mini_windows()
    mini = make_fixture(SCRATCH / 'mini', Wm, pm_stage2=True)
    RP_W = RP.Windows
    RP.Windows = lambda *a, **k: Wm                     # main() が読む既定の窓を差し替える
    try:
        R2, out2 = _run_report(mini, [])
    finally:
        RP.Windows = RP_W
    n2 = RP.all_rows_have_values(R2)
    L = {(r['env'], r['family'], str(r['layer']), r['item']): r['label'] for r in R2.rows}
    # 経路が実際に通ったことを、期待できる行で確かめる（短縮窓 t11–20 では ELUF ≈ ELU1 なので D1 は EQUALS 側になりうる）
    assert L[('pmnist', 'H', 'all', 'D1')].startswith(('BIG_', 'NOT_TESTABLE')), L[('pmnist', 'H', 'all', 'D1')]
    assert L[('pmnist', 'H', 'all', 'D2')].startswith(('SMALL_', 'NOT_TESTABLE', 'CO_PRIMARY_CONFLICT')), L[('pmnist', 'H', 'all', 'D2')]
    assert L[('pmnist', 'H', 'all', 'HYPOTHESIS')].startswith(('CANCELLATION_', 'PARTIAL_CANCELLATION', 'CO_PRIMARY_CONFLICT')), L[('pmnist', 'H', 'all', 'HYPOTHESIS')]
    assert L[('pmnist', 'H', 'all', 'D5A_N')].startswith(('N_', 'NOT_TESTABLE')), L[('pmnist', 'H', 'all', 'D5A_N')]
    assert L[('pmnist', 'H', 'any', 'D0_N_L')] in ('MANIPULATION_OK', 'NOT_TESTABLE_WEAK_MANIPULATION', 'NOT_TESTABLE_NO_CONTRAST')
    assert L[('pmnist', 'H', '0', 'D0_D_A')].startswith(('MANIPULATION_OK', 'NOT_TESTABLE'))
    assert L[('rlmnist', 'H', 'all', 'D3_HEADING')] == 'DEEP_DOMINATES_FLOOR', L[('rlmnist', 'H', 'all', 'D3_HEADING')]
    assert L[('rlmnist', 'H', 'all', 'HYPOTHESIS')] == 'CANCELLATION_NOT_TESTABLE_RL_FLOOR'
    assert L[('rlmnist', 'H', 'all', 'D2')] == 'NOT_TESTABLE_FLOOR'
    assert any(k[3].startswith('D7_') and not v.startswith('NOT_DETERMINED') for k, v in L.items() if k[0] == 'pmnist'), 'D7 not evaluated'
    assert any(k[3].startswith('D6_') for k in L) and L[('scr', 'H', '1', 'D_PAR')] in ('PARENTS_ELU_BETTER', 'PARENTS_LR_BETTER', 'PARENTS_UNRESOLVED')
    assert L[('all', 'H', 'all', 'D8')].startswith(('CROSS_ENV', 'ENV_DEPENDENT'))
    assert L[('scr', 'H', '1', 'D7_DEEP_RELAX')].startswith('DEEP_RELAX')
    # D9: 段 2（stage2/pmnist_lr{5e-4,2e-3}）を同じ規則で判定し、3 つの lr の D3 の状態から組む
    d9_note = next(r['note'] for r in R2.rows if r['item'] == 'D9')
    assert 'lrs=[0.0005, 0.001, 0.002]' in d9_note, d9_note
    assert ('pmnist_lr5e-4', 'H', 'all', 'D3_N') in L and ('pmnist_lr2e-3', 'H', 'all', 'HYPOTHESIS') in L
    assert L[('pmnist', 'all', 'all', 'G1_PM')] == 'G1_PM_OK', L[('pmnist', 'all', 'all', 'G1_PM')]
    # 追補 2-2: 合成の双子は t1 から生きているので、箱 B 12 対・RL 3 対で σ_traj が出る（TWIN_LIVE の行は 15）
    assert L[('pmnist', 'twins', 'all', 'SIGMA_ARM_L')] == 'SIGMA_TRAJ_OK' and L[('rlmnist', 'twins', 'all', 'SIGMA_ARM_PT')] == 'SIGMA_TRAJ_OK'
    assert sum(1 for k, v in L.items() if k[3] == 'TWIN_LIVE' and v == 'TWIN_LIVE') == 15
    rp_prov = json.loads((out2 / 'report_provenance.json').read_text())
    assert rp_prov['extra']['twin_live']['rlmnist']['LRtw0_s2']['twin_live_task'] == 1, rp_prov['extra']['twin_live']
    assert any(k[3].startswith('realized_Gbar') for k in [(r['env'], r['family'], '', r['contrast']) for r in R2.seed_rows])
    # (c) RL の段 2（V・S 家族・分離腕）: D5b・D4（RL）・D9（3 つの lr）の経路
    st2 = make_fixture(SCRATCH / 'mini2', Wm, rl_stage2=True)
    RP.Windows = lambda *a, **k: Wm
    try:
        R3, _ = _run_report(st2, [], ['--stage2', '--envs', 'rlmnist'])
    finally:
        RP.Windows = RP_W
    L3 = {(r['env'], r['family'], str(r['layer']), r['item']): r['label'] for r in R3.rows}
    assert L3[('rlmnist', 'V', 'all', 'D5B_N')].endswith(('_V', 'NOT_TESTABLE_FLOOR', 'NOT_TESTABLE_COLLINEAR', 'NOT_TESTABLE_WEAK_MANIPULATION')) \
        or L3[('rlmnist', 'V', 'all', 'D5B_N')].startswith('NOT_'), L3[('rlmnist', 'V', 'all', 'D5B_N')]
    assert ('rlmnist', 'H+S', 'all', 'D4') in L3 and ('rlmnist', 'H', 'all', 'D5A_N') in L3
    Rd = RP.Report()
    RP.judge_d9({0.001: dict(N='HELPS', D='HELPS'), 0.0005: dict(N='HELPS', D='HELPS'), 0.002: dict(N='HELPS', D='HELPS')}, Rd)
    RP.judge_d9({0.001: dict(N='HELPS', D='HELPS'), 0.0005: dict(N='INERT', D='HELPS'), 0.002: dict(N='HELPS', D='HELPS')}, Rd)
    d9 = [r['label'] for r in Rd.rows]
    assert d9[0] == 'PATTERN_LR_ROBUST' and d9[1].startswith('PATTERN_LR_DEPENDENT(N,'), d9
    assert RP.Windows.lr0p005().scr_win == (991, 1000)
    # 対照: 読み出しのキーを 1 つ消すと SchemaError で落ちる（スモークと主窓の両方）
    ctl = {}
    for tag, src, flags in (('smoke', smoke, ['--smoke']), ('mini', mini, [])):
        bad = SCRATCH / f'{tag}_mut'
        shutil.copytree(src, bad)
        p = bad / 'pmnist' / 'SMINH_s1_readout.npz'
        arrs = {k: v for k, v in np.load(p).items() if k != 'depth_vel'}
        np.savez(p, **arrs)
        if tag == 'mini':
            RP.Windows = lambda *a, **k: Wm
        try:
            _run_report(bad, flags)
            raise AssertionError('deleted key did not fail G0.5')
        except RP.SchemaError as e:
            ctl[tag] = str(e)
        finally:
            RP.Windows = RP_W
    rec['G05_fixture'] = dict(smoke_rows=n, mini_rows=n2, controls=ctl,
                              mini_labels={k[3] + ('' if k[1] in ('H', 'all') else f'[{k[1]}]'): v for k, v in L.items()
                                           if k[3] in ('D1', 'D2', 'D3_HEADING', 'HYPOTHESIS', 'D5A_N', 'D5A_D', 'D4', 'D6', 'D8') and k[1] in ('H', 'H+S', 'all')})


CHECKS_REPORT = [check_S16, check_G05_fixture]
