#!/usr/bin/env python3
"""chargpt_seed10_1022 の A（装填）: 2 ブロック C3-2・ReLUcap-2 の seed 0〜9 の固定状態の計算（CPU・float64）。

写し元（定義は変えない）:
  analysis/chargpt_load_fold_1021/load/measure.py       git blob cd225d27b47a614e018ec3d8c6ecf301e68f91e2
  analysis/chargpt_load_fold_1021/load/reanalyze_p07.py git blob 2cb251ac1b11978717e949a80abf673893ed1617
  （どちらも $S10 の HEAD 90f3c805 での blob。main 21989864 の写し）
変えた所:
  - SOURCE を環境変数 CHARGPT_SRC に。SEED を引数に（data・一様 probe を seed ごとに作る）。
  - 置き場を results/chargpt_cl_1019/M2/<腕>/seed<k>（照合のときだけ quicklook/M2/C3/seed1000）に。腕は M2 の C3・ReLUcap。
  - 状態は init と課題末 1〜10・25・39〜50・89〜100（u100 は扱わない）。
  - 足した所（記録）: L0 の G_own・G_full（JVP の出力を [L0 の m, L1 の m] の連結にして 1 回で出す。各成分の定義は写し元と同じ）。
    n_alive と |G_own| の中央値は写し元の列（own_abs_median・own_abs_median_alive）をそのまま全層で出す。
    実際の次の課題の鎖 1 実現の s（reanalyze_p07.py の next_gradient の定義: 128 位置の文脈、最初の位置を損失から除く）を全層で出す。
    写し元の legacy の next_sink（128 位置 CE・次の課題の probe の集合）と、p07 の fixed-next の G_full は出さない。
  - 出力は腕 × seed の単位で parts/ に書き、完走した単位は飛ばす（再開できる）。完走ごとの時間を provenance.json に残す。
  - checks は spec §7 の A・追補 1 の許容・手順書 §3.1 の (i)〜(iii)。(ii) の有限差分は L0 にも当てる（新しく出す量のため）。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, csv, math, argparse, hashlib, subprocess, time, datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SOURCE = Path(os.environ['CHARGPT_SRC'])
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/chargpt_seed10_1022/load'
CHK = ROOT / 'results/chargpt_seed10_1022/checks/A'
REF = ROOT / 'results/chargpt_load_fold_1021/load'
sys.path.insert(0, str(SOURCE / 'src'))
import chargpt_cl_1019_kune as K
C = K.C
torch.set_num_threads(8)
DT = torch.float64
CHUNK = 8
ARMS = ['C3', 'ReLUcap']
SEEDS = list(range(10))
TASKS = list(range(1, 11)) + [25] + list(range(39, 51)) + list(range(89, 101))
STATES = [0] + TASKS          # 36 状態（init と課題末 35）
CACHE = {}
UCACHE = {}

def data(seed, t):
    if (seed, t) not in CACHE:
        chain = C.ChainM(seed, t, 0.01)
        seq = chain.sample(C.M_PROBE, C.cpu_gen(seed, 'probe', t, 'chain'))
        ev = chain.sample(C.M_EVAL, C.cpu_gen(seed, 'probe', t, 'eval'))
        CACHE[seed, t] = (seq[:, :C.T], seq[:, 1:], ev, chain.H)
    return CACHE[seed, t]

def uniform(seed):
    if seed not in UCACHE:
        UCACHE[seed] = C.uniform_probe_M(seed, 'cpu')
    return UCACHE[seed]

def directory(arm, seed):
    if seed == 1000:
        assert arm == 'C3'
        return SOURCE / 'results/chargpt_cl_1019/quicklook/M2/C3/seed1000'
    return SOURCE / 'results/chargpt_cl_1019/M2' / arm / f'seed{seed}'

def ckpt(arm, seed, t):
    return directory(arm, seed) / 'ckpts' / ('init.pt' if t == 0 else f'task{t}.pt')

def load(arm, seed, t):
    C.L = 2
    m = C.GPT(arm, 'M')
    m.load_state_dict(torch.load(ckpt(arm, seed, t), map_location='cpu', weights_only=True))
    m.eval().double()
    for b in m.h:
        b.mlp.track_v = False
    assert m.wte.weight is m.head.weight
    return m

def chunks(x):
    for i in range(0, len(x), CHUNK):
        yield i, x[i:i+CHUNK]

def ce(logits, y=None):
    if y is None:
        return (logits.logsumexp(-1)-logits.mean(-1)).mean()
    return F.cross_entropy(logits.reshape(-1, C.V), y.reshape(-1))

@torch.no_grad()
def moments(m, x):
    ys = [[] for _ in m.h]; zs = [[] for _ in m.h]
    for _, xx in chunks(x):
        cap = []; m.body(xx, cap)
        for l, b in enumerate(m.h):
            ys[l].append(b.n2(cap[2*l]).reshape(-1, C.D))
            zs[l].append(cap[2*l+1].reshape(-1, C.FF))
    return [(torch.cat(y), torch.cat(z)) for y, z in zip(ys, zs)]

@torch.no_grad()
def head(m, x, y, nperm=1000):
    hs = []; fs = []
    for _, xx in chunks(x):
        h = m.body(xx); hs.append(h.reshape(-1, C.D)); fs.append(m.head(h).reshape(-1, C.V))
    h = torch.cat(hs); f = torch.cat(fs); yy = y.reshape(-1)
    vc = m.head.weight-m.head.weight.mean(0)
    fm = (F.one_hot(yy, C.V).to(DT)-1/C.V).T@h
    ff = float(fm.square().sum()); A = float((vc*fm).sum())
    aid = float((f.gather(1, yy[:, None]).squeeze(1)-f.mean(1)).sum())
    loss = float(ce(f, yy)); D = len(yy)*(math.log(C.V)-loss)
    J = float((f.logsumexp(1)-f.mean(1)-math.log(C.V)).sum())
    gen = torch.Generator().manual_seed(1021000)
    mat = vc@fm.T
    null = torch.stack([mat[torch.randperm(C.V, generator=gen), torch.arange(C.V)].sum()/ff for _ in range(nperm)]).numpy()
    kap = A/ff
    r = dict(N=len(yy), feature_dim=C.D, kappa=kap, cos2=A*A/(ff*float(vc.square().sum())), Q=A/math.sqrt(ff), F_norm=math.sqrt(ff), v_norm=float(vc.norm()), A=A, A_id=aid, identity_rel=abs(A-aid)/max(abs(A), abs(aid), 1e-300), L=loss, D=D, J=J, kappa_lb=D/ff, kappa_over_lb=A/D if D > 0 else float('nan'), lemma_violation=int(A-D < -1e-10*max(1, abs(A), abs(D))), null_quantile=float((null <= kap).mean()), null_sd=float(null.std(ddof=1)))
    r.update({f'null_q{p:02d}': float(np.percentile(null, p)) for p in [1, 5, 50, 95, 99]})
    return r

def gradient(m, x, y=None):
    m.zero_grad(set_to_none=True)
    for i, xx in chunks(x):
        (ce(m(xx), None if y is None else y[i:i+len(xx)])*len(xx)/len(x)).backward()
    return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in m.named_parameters()}

def next_gradient(m, x, y):
    # reanalyze_p07.py の定義: 128 位置の文脈、最初の位置を損失から除く
    m.zero_grad(set_to_none=True)
    for i, xx in chunks(x):
        (ce(m(xx)[:, 1:], y[i:i+len(xx), 1:])*len(xx)/len(x)).backward()
    return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in m.named_parameters()}

class Position(nn.Module):
    # 写し元は最後の層の m だけ。ここでは全層の m を連結する（成分ごとの定義は同じ）
    def __init__(self, m):
        super().__init__(); self.model = m
    def forward(self, x):
        cap = []; self.model.body(x, cap)
        return torch.cat([cap[2*l+1].mean((0, 1)) for l in range(len(self.model.h))])

def full(m, x, g):
    """各層の G_full,j = <∇θ m_j, g>（JVP）。返り値は層ごとの list。"""
    wrapper = Position(m)
    p = {f'model.{n}': v.detach() for n, v in m.named_parameters()}
    tangent = {f'model.{n}': g[n] for n in g}
    nl = len(m.h)
    result = torch.zeros(nl*C.FF, dtype=DT)
    for _, xx in chunks(x):
        def fn(pp):
            return torch.func.functional_call(wrapper, pp, (xx,), tie_weights=True)
        _, j = torch.func.jvp(fn, (p,), (tangent,))
        result += j.detach()*len(xx)/len(x)
    return [result[l*C.FF:(l+1)*C.FF].clone() for l in range(nl)]

def agreement(a, b, mask):
    mk = mask & (a != 0) & (b != 0)
    return (float((a[mk].sign() == b[mk].sign()).double().mean()) if mk.any() else float('nan'), int(mk.sum()))

def sink(g, alive):
    return float((g[alive] > 0).double().mean()) if alive.any() else float('nan')

def units(m, x, g, gfs=None, gn=None):
    rows = []; raw = []
    for l, (y, z) in enumerate(moments(m, x)):
        b = m.h[l]; ybar = y.mean(0); mu = z.mean(0); alive = (z > 0).any(0)
        own = (g[f'h.{l}.mlp.up.weight']*ybar).sum(1)
        kn = y@ybar
        d = (kn[:, None]*b.mlp.phi(z)*b.mlp.dphi(z)).T@y
        scalar = d@ybar
        agree, n = agreement(own, scalar, alive)
        r = dict(layer=l, n_units=C.FF, n_alive=int(alive.sum()), m_median=float(mu.median()), closure_max=float((mu-b.mlp.up.weight.detach()@ybar).abs().max()), own_sink=sink(own, alive), own_abs_median=float(own.abs().median()), own_abs_median_alive=float(own[alive].abs().median()) if alive.any() else float('nan'), self_agreement=agree, self_n=n)
        q = dict(m=mu, alive=alive, own=own, self_scalar=scalar, ybar=ybar)
        if gfs is not None:
            gf = gfs[l]
            a, n = agreement(own, gf, alive)
            r.update(full_sink=sink(gf, alive), full_own_agreement=a, full_own_n=n, full_abs_median=float(gf.abs().median()))
            q['G_full'] = gf
        if gn is not None:
            ownn = (gn[f'h.{l}.mlp.up.weight']*ybar).sum(1)
            r.update(fixed_next_own_sink=sink(ownn, alive))
            r['fixed_next_minus_uniform'] = r['fixed_next_own_sink']-r['own_sink']
            q['fixed_next_own'] = ownn
        rows.append(r); raw.append(q)
    return rows, raw

@torch.no_grad()
def eval_metric(m, ev, H):
    val = 0.
    for _, s in chunks(ev):
        val += float(ce(m(s[:, :C.T])[:, 1:], s[:, 2:]))*len(s)/len(ev)
    return val-H

@torch.no_grad()
def loss_value(m, x, y=None):
    return sum(float(ce(m(xx), None if y is None else y[i:i+len(xx)]))*len(xx)/len(x) for i, xx in chunks(x))

def rounding_bound(text):
    v = abs(float(text))
    return 0.5*10**(math.floor(math.log10(v))-8) if v else 0.

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def measure_state(arm, seed, t, k1=None):
    """1 状態の計算。head（課題 t の鎖の probe）・units（全層、G_own・G_full・次の課題の s）。"""
    m = load(arm, seed, t); x, y, _, _ = data(seed, max(1, t)); xu = uniform(seed)
    h = head(m, x, y)
    if t == 1: k1 = h['kappa']          # 写し元と同じ: 課題 1 末で k1 を決めてから比を出す
    h['kappa_ratio'] = h['kappa']/k1 if k1 is not None else float('nan')
    g = gradient(m, xu); gfs = full(m, x, g)
    xn, yn, _, _ = data(seed, t+1); gn = next_gradient(m, xn, yn)
    rows, raw = units(m, x, g, gfs, gn)
    return h, rows, raw

# ---------------------------------------------------------------- 検査
def rel(a, b):
    a = float(a); b = float(b)
    if math.isnan(a) and math.isnan(b): return 0.
    if math.isnan(a) or math.isnan(b): return float('inf')
    d = max(abs(a), abs(b))
    return abs(a-b)/d if d > 0 else 0.

HEAD_CONT = ['kappa', 'cos2', 'Q', 'F_norm', 'v_norm', 'A', 'A_id', 'L', 'D', 'J', 'kappa_lb', 'kappa_over_lb', 'null_sd', 'null_q01', 'null_q05', 'null_q50', 'null_q95', 'null_q99', 'kappa_ratio']
HEAD_EXACT = ['N', 'feature_dim', 'lemma_violation']
UNIT_CONT = ['m_median', 'own_abs_median', 'own_abs_median_alive', 'full_abs_median']
UNIT_EXACT = ['n_units', 'n_alive', 'self_n', 'full_own_n']
# 割合: (列, 新しい列, 分母の列)
UNIT_FRAC = [('own_sink', 'own_sink', 'n_alive'), ('full_sink', 'full_sink', 'n_alive'), ('self_agreement', 'self_agreement', 'self_n'), ('full_own_agreement', 'full_own_agreement', 'full_own_n'), ('uniform_own_sink', 'own_sink', 'n_alive'), ('fixed_next_own_sink', 'fixed_next_own_sink', 'n_alive')]

def check_i(add, detail):
    """(i) seed 1000 の M2/C3 の init と課題末 35 を出し直して 21989864 の保存値と照合（追補 1 の許容）。"""
    th = {(int(r['task']), r['stage']): r for r in csv.DictReader(open(REF/'table_head.csv')) if r['box'] == 'M2' and r['arm'] == 'C3'}
    tu = {(int(r['task']), r['stage']): r for r in csv.DictReader(open(REF/'table_units.csv')) if r['box'] == 'M2' and r['arm'] == 'C3' and r['layer'] == '1'}
    k1 = None; worst = dict(head_cont=(0., None), unit_cont=(0., None), frac_in_units=(0., None)); fails = []
    nstate = 0; t0 = time.time()
    for t in STATES:
        stage = 'init' if t == 0 else 'end'
        h, rows, _ = measure_state('C3', 1000, t, k1)
        if t == 1: k1 = h['kappa']
        u = rows[1]; rh = th[t, stage]; ru = tu[t, stage]; nstate += 1
        for c in HEAD_CONT:
            e = rel(h[c], rh[c])
            if e > worst['head_cont'][0]: worst['head_cont'] = (e, f't{t}:{c}')
            if e > 1e-8: fails.append(dict(t=t, col=c, new=h[c], ref=rh[c], rel=e))
        for c in HEAD_EXACT:
            if int(h[c]) != int(rh[c]): fails.append(dict(t=t, col=c, new=h[c], ref=rh[c]))
        e = abs(h['null_quantile']-float(rh['null_quantile']))*1000
        if e > 1: fails.append(dict(t=t, col='null_quantile', new=h['null_quantile'], ref=rh['null_quantile'], in_permutations=e))
        if h['identity_rel'] > 1e-10: fails.append(dict(t=t, col='identity_rel', new=h['identity_rel']))
        for c in UNIT_CONT:
            e = rel(u[c], ru[c])
            if e > worst['unit_cont'][0]: worst['unit_cont'] = (e, f't{t}:{c}')
            if e > 1e-8: fails.append(dict(t=t, col=c, new=u[c], ref=ru[c], rel=e))
        for c in UNIT_EXACT:
            if int(u[c]) != int(ru[c]): fails.append(dict(t=t, col=c, new=u[c], ref=ru[c]))
        for c, cn, dn in UNIT_FRAC:
            n = int(ru[dn]); e = abs(float(u[cn])-float(ru[c]))*n   # 単位 = 1/n
            if e > worst['frac_in_units'][0]: worst['frac_in_units'] = (e, f't{t}:{c}')
            if e > 1+1e-9: fails.append(dict(t=t, col=c, new=u[cn], ref=ru[c], diff_in_units_of_1_over_n=e))
        if u['closure_max'] > 1e-10: fails.append(dict(t=t, col='closure_max', new=u['closure_max']))
        detail.append(dict(t=t, kappa=h['kappa'], own_sink=u['own_sink'], ref_own_sink=float(ru['own_sink']), fixed_next_own_sink=u['fixed_next_own_sink'], ref_fixed_next_own_sink=float(ru['fixed_next_own_sink'])))
        print(f'(i) t{t} done {time.time()-t0:.1f}s', flush=True)
    add('i_seed1000_reproduce', not fails, states=nstate, n_fail=len(fails), fails=fails[:50],
        max_rel_head_continuous=worst['head_cont'], max_rel_unit_continuous=worst['unit_cont'], max_fraction_diff_in_units_of_1_over_n=worst['frac_in_units'],
        tolerance='continuous rel 1e-8; fractions 1/n (n = n_alive, or self_n/full_own_n for agreements); null_quantile 1/1000; counts exact; identity_rel and closure_max <= 1e-10',
        elapsed_seconds=time.time()-t0)

def check_ii(add):
    """(ii) chargpt_load_fold_1021 §4 の検査（補題の恒等式・有限差分・変異対照・評価値）。L0 の有限差分を足す。"""
    for arm, seed in [('C3', 1000), ('C3', 0), ('ReLUcap', 0)]:
        tag = f'M2/{arm}/seed{seed}/task1'
        m = load(arm, seed, 1); x, y, ev, H = data(seed, 1); xu = uniform(seed)
        add('tied_weight', m.wte.weight is m.head.weight and m.wte.weight.data_ptr() == m.head.weight.data_ptr(), arm=tag)
        h = head(m, x, y)
        add('lemma_identity', h['identity_rel'] <= 1e-10, arm=tag, relative_error=h['identity_rel'])
        g = gradient(m, xu); gfs = full(m, x, g); ur, raw = units(m, x, g, gfs)
        add('position_closure', max(r['closure_max'] for r in ur) <= 1e-10, arm=tag, max_abs=max(r['closure_max'] for r in ur))
        old = m.head.weight
        m.head.weight = nn.Parameter(-old.detach().clone())
        hm = head(m, x, y)
        add('negate_head', h['kappa'] > 0 and hm['kappa'] < 0 and abs(hm['kappa']+h['kappa']) < 1e-12, arm=tag, kappa=h['kappa'], mutant_kappa=hm['kappa'], note='head-only mutant intentionally untied; embedding unchanged')
        m.head.weight = old
        gc = gradient(m, xu, torch.zeros_like(xu))
        uclass, rclass = units(m, x, gc)
        for l in range(len(m.h)):
            alive = raw[l]['alive']; go = raw[l]['own']; go1 = rclass[l]['own']
            flips = int(((go.sign() != go1.sign()) & alive).sum())
            add('one_class_mutant', flips > 0 and ur[l]['own_sink'] != uclass[l]['own_sink'], arm=tag, layer=l, sign_flips=flips, uniform_sink=ur[l]['own_sink'], one_class_sink=uclass[l]['own_sink'])
        base = {n: p.detach().clone() for n, p in m.named_parameters()}
        for l in range(len(m.h)):
            alive = raw[l]['alive']; go = raw[l]['own']; gf = gfs[l]
            ids = torch.where(alive & (go.abs() > go.abs().max()*1e-3) & (gf.abs() > gf.abs().max()*1e-3))[0]
            ids = ids[torch.linspace(0, len(ids)-1, min(3, len(ids))).long()].tolist()
            own_fd = []; full_fd = []; eps = 1e-6
            for j in ids:
                w = m.h[l].mlp.up.weight; yb = raw[l]['ybar']
                with torch.no_grad(): w[j].add_(eps*yb)
                lp = loss_value(m, xu)
                with torch.no_grad(): w[j].sub_(2*eps*yb)
                lm = loss_value(m, xu)
                with torch.no_grad(): w.copy_(base[f'h.{l}.mlp.up.weight'])
                fd = (lp-lm)/(2*eps); exact = float(go[j])
                own_fd.append(dict(unit=j, ad=exact, fd=fd, relative_error=abs(fd-exact)/max(abs(fd), abs(exact), 1e-300)))
            for eps_full in [1e-6, 1e-7, 1e-8]:
                vals = []
                for sgn in [1, -1]:
                    with torch.no_grad():
                        for n, p in m.named_parameters(): p.copy_(base[n]+sgn*eps_full*g[n])
                    vals.append(moments(m, x)[l][1].mean(0))
                fd = (vals[0]-vals[1])/(2*eps_full)
                trial = [dict(unit=j, ad=float(gf[j]), fd=float(fd[j]), relative_error=float(abs(fd[j]-gf[j])/torch.maximum(fd[j].abs(), gf[j].abs()).clamp_min(1e-300))) for j in ids]
                full_fd.append(dict(epsilon=eps_full, units=trial))
                if all(r['relative_error'] <= 1e-4 for r in trial): break
            with torch.no_grad():
                for n, p in m.named_parameters(): p.copy_(base[n])
            add('own_finite_difference', bool(ids) and all(r['relative_error'] <= 1e-4 for r in own_fd), arm=tag, layer=l, epsilon=eps, units=own_fd)
            add('full_finite_difference', bool(ids) and all(r['relative_error'] <= 1e-4 for r in full_fd[-1]['units']), arm=tag, layer=l, trials=full_fd)
        record = next(r for r in csv.DictReader(open(directory(arm, seed)/'per_task.csv')) if int(r['task']) == 1)
        got = eval_metric(m, ev, H); expected = float(record['end_metric'])
        csv_tol = rounding_bound(record['end_metric']); tol = 2e-6*abs(expected)+csv_tol
        add('recorded_eval_rounding', abs(got-expected) <= tol, arm=tag, computed_float64=got, recorded=expected, abs_error=abs(got-expected), relative_error=abs(got-expected)/abs(expected), tolerance=tol, rtol=2e-6, csv_rounding_bound=csv_tol, registered_rounding_pass=abs(got-expected) <= csv_tol, H_recomputed=H, H_recorded=float(record['H']), note='写し元 checks.json の all_pass_criterion（float32-adjusted: 2e-6*|recorded| + .9g の丸め）と同じ基準')

def check_iii(add):
    """(iii) 同じ seed の C3-2 と ReLUcap-2 の init.pt と task1.pt がビットで同じ。"""
    out = []
    for seed in SEEDS:
        for t in [0, 1]:
            a = torch.load(ckpt('C3', seed, t), map_location='cpu', weights_only=True)
            b = torch.load(ckpt('ReLUcap', seed, t), map_location='cpu', weights_only=True)
            ka, kb = set(a), set(b); common = sorted(ka & kb)
            neq = [k for k in common if not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]))]
            # 浮動小数のビット一致: torch.equal は -0/+0 と NaN を区別しないので、bytes でも比べる
            nbytes = [k for k in common if a[k].detach().contiguous().cpu().numpy().tobytes() != b[k].detach().contiguous().cpu().numpy().tobytes()]
            same_file = sha(ckpt('C3', seed, t)) == sha(ckpt('ReLUcap', seed, t))
            # ReLUcap だけにある buffer は cap の帳簿（V・cap・cap_on・v_rate）。課題 1 末までは cap が掛かっていないこと（cap_on = 0・cap = inf）を確かめる
            extra = sorted(kb-ka)
            extra_ok = all(k.split('.')[-1] in ('V', 'cap', 'cap_on', 'v_rate') for k in extra) and \
                all(float(b[k]) == 0. for k in extra if k.endswith('cap_on')) and all(bool(torch.isinf(b[k]).all()) for k in extra if k.endswith('.cap'))
            # 計算に使う形（float64・track_v off）での出力がビットで同じこと（一様 probe の 8 系列）
            xu = uniform(seed)[:8]
            with torch.no_grad():
                fa = load('C3', seed, t)(xu); fb = load('ReLUcap', seed, t)(xu)
            fwd_equal = fa.numpy().tobytes() == fb.numpy().tobytes()
            out.append(dict(seed=seed, task=t, n_common=len(common), only_C3=sorted(ka-kb), only_ReLUcap=extra, tensors_unequal=neq, bytes_unequal=nbytes,
                            cap_inactive=extra_ok, forward_bitwise_equal=fwd_equal, file_sha_equal=same_file))
    okf = lambda r: not r['tensors_unequal'] and not r['bytes_unequal'] and not r['only_C3'] and r['cap_inactive'] and r['forward_bitwise_equal']
    add('iii_C3_ReLUcap_init_task1_bitwise', all(okf(r) for r in out), n_pairs=len(out), n_pairs_equal=sum(okf(r) for r in out),
        n_pairs_identical_state_dict=sum(not r['tensors_unequal'] and not r['bytes_unequal'] and not r['only_C3'] and not r['only_ReLUcap'] for r in out),
        criterion='共通の全 tensor（重み・buffer）が bytes で同じ・C3 だけの key 無し・ReLUcap だけの key は cap の帳簿で cap_on=0・cap=inf・float64 の forward がビットで同じ。'
                  'ReLUcap の state_dict には cap の帳簿の buffer が足されているので、file の sha と state_dict 全体は一致しない（n_pairs_identical_state_dict に記録）',
        pairs=out)

def checks(which):
    CHK.mkdir(parents=True, exist_ok=True)
    rec = []; detail = []
    def add(name, ok, **kw):
        r = dict(check=name, pass_=bool(ok), **kw); rec.append(r)
        print(json.dumps(r, ensure_ascii=False)[:400], flush=True)
    t0 = time.time()
    if 'iii' in which: check_iii(add)
    if 'ii' in which: check_ii(add)
    if 'i' in which: check_i(add, detail)
    result = dict(all_pass=all(r['pass_'] for r in rec), parts_run=which, n_checks=len(rec), n_pass=sum(r['pass_'] for r in rec), elapsed_seconds=time.time()-t0, checks=rec, seed1000_detail=detail, provenance=provenance())
    (CHK/'checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    return result

def provenance():
    paths = [ROOT/'specs/spec_chargpt_seed10_1022.md', ROOT/'specs/chargpt_seed10_1022_runbook.md', SOURCE/'src/chargpt_cl_1019.py', SOURCE/'src/chargpt_cl_1019_kune.py']
    return dict(source_root=str(SOURCE), source_head=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(), required_source_head='033679e0',
                copied_from={'analysis/chargpt_load_fold_1021/load/measure.py': 'cd225d27b47a614e018ec3d8c6ecf301e68f91e2', 'analysis/chargpt_load_fold_1021/load/reanalyze_p07.py': '2cb251ac1b11978717e949a80abf673893ed1617'},
                device='cpu', dtype='float64', threads=torch.get_num_threads(), torch=torch.__version__, numpy=np.__version__, python=sys.version,
                script_sha256=sha(Path(__file__)), input_code_sha256={str(p): sha(p) for p in paths},
                probe=dict(sequences=C.M_PROBE, tokens_per_sequence=C.T, eval_sequences=C.M_EVAL, chain_alpha=0.01), null=dict(permutations=1000, seed=1021000),
                states=STATES, arms=ARMS, seeds=SEEDS)

# ---------------------------------------------------------------- 本走（腕 × seed 単位で再開できる）
def writecsv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(str(path)+'.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def run_unit(arm, seed):
    """1 腕 × 1 seed の 36 状態。parts/ に書く。"""
    parts = OUT/'parts'; rawd = OUT/'raw'; parts.mkdir(parents=True, exist_ok=True); rawd.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); heads = []; urows = []; payload = {}; k1 = None; per_state = []
    for t in STATES:
        ts = time.time()
        h, rows, raw = measure_state(arm, seed, t, k1)
        if t == 1: k1 = h['kappa']
        stage = 'init' if t == 0 else 'end'
        key = dict(arm=arm, seed=seed, task=t, stage=stage)
        heads.append({**key, **h})
        for r in rows:
            urows.append({**key, **r, **{f'head_{c}': h[c] for c in ['kappa', 'cos2', 'Q', 'F_norm', 'L', 'lemma_violation', 'kappa_over_lb', 'null_quantile', 'kappa_ratio']}})
        for l, q in enumerate(raw):
            for n, v in q.items():
                if n != 'ybar': payload[f't{t}_L{l}_{n}'] = v.numpy()
        per_state.append(dict(task=t, seconds=time.time()-ts))
        print(f'{arm} seed{seed} t{t} {time.time()-ts:.1f}s', flush=True)
    np.savez_compressed(rawd/f'units_{arm}_seed{seed}.raw.npz', **payload)
    writecsv(parts/f'{arm}_seed{seed}_head.csv', heads)
    writecsv(parts/f'{arm}_seed{seed}_states.csv', urows)
    done = dict(arm=arm, seed=seed, n_states=len(STATES), elapsed_seconds=time.time()-t0, finished=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                per_state_seconds=per_state, script_sha256=sha(Path(__file__)),
                inputs={str(ckpt(arm, seed, t).relative_to(SOURCE)): sha(ckpt(arm, seed, t)) for t in STATES},
                outputs={f'parts/{arm}_seed{seed}_head.csv': sha(parts/f'{arm}_seed{seed}_head.csv'), f'parts/{arm}_seed{seed}_states.csv': sha(parts/f'{arm}_seed{seed}_states.csv'), f'raw/units_{arm}_seed{seed}.raw.npz': sha(rawd/f'units_{arm}_seed{seed}.raw.npz')})
    tmp = parts/f'{arm}_seed{seed}_done.json.tmp'; tmp.write_text(json.dumps(done, indent=2)+'\n'); tmp.replace(parts/f'{arm}_seed{seed}_done.json')

def merge():
    parts = OUT/'parts'; heads = []; urows = []; done = []
    for arm in ARMS:
        for seed in SEEDS:
            dj = parts/f'{arm}_seed{seed}_done.json'
            if not dj.exists(): continue
            done.append(json.loads(dj.read_text()))
            heads += list(csv.DictReader(open(parts/f'{arm}_seed{seed}_head.csv')))
            urows += list(csv.DictReader(open(parts/f'{arm}_seed{seed}_states.csv')))
    if heads:
        writecsv(OUT/'head.csv', heads); writecsv(OUT/'states.csv', urows)
    pv = provenance(); pv['completed_units'] = done; pv['n_completed_units'] = len(done)
    pv['checks_A_sha256'] = sha(CHK/'checks.json') if (CHK/'checks.json').exists() else None
    tmp = OUT/'provenance.json.tmp'; tmp.write_text(json.dumps(pv, indent=2, ensure_ascii=False)+'\n'); tmp.replace(OUT/'provenance.json')

def measure(arms, seeds):
    ck = json.loads((CHK/'checks.json').read_text())
    if not ck['all_pass'] or set(ck['parts_run']) != {'i', 'ii', 'iii'}:
        raise RuntimeError('checks (i)〜(iii) が全部通っていない。本走は禁止。')
    if ck['provenance']['script_sha256'] != sha(Path(__file__)):
        raise RuntimeError('checks の後に script が変わった。checks を回し直す。')
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for seed in seeds:
            if (OUT/'parts'/f'{arm}_seed{seed}_done.json').exists():
                print(f'skip {arm} seed{seed} (done)', flush=True); continue
            run_unit(arm, seed); merge()
    merge()

def timing(arm, seed, tasks):
    """本走と同じ計算を数状態で回して時間を測る（load/ には書かない）。"""
    CHK.mkdir(parents=True, exist_ok=True); rec = []; k1 = None
    for t in tasks:
        ts = time.time(); h, rows, _ = measure_state(arm, seed, t, k1)
        if t == 1: k1 = h['kappa']
        rec.append(dict(task=t, seconds=time.time()-ts, kappa=h['kappa'], n_alive_L1=rows[1]['n_alive'], own_sink_L1=rows[1]['own_sink']))
        print(rec[-1], flush=True)
    sec = [r['seconds'] for r in rec]
    res = dict(arm=arm, seed=seed, states=rec, mean_seconds=float(np.mean(sec)), max_seconds=float(max(sec)), threads=torch.get_num_threads(),
               estimate_720_states_hours=float(np.mean(sec))*720/3600, note='1 process・8 threads。checks の 36 状態の時間も checks.json に残る')
    (CHK/'timing.json').write_text(json.dumps(res, indent=2)+'\n')
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['checks', 'measure', 'merge', 'timing'])
    ap.add_argument('--parts', nargs='+', default=['iii', 'ii', 'i'], choices=['i', 'ii', 'iii'])
    ap.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--arm', default='C3'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tasks', nargs='+', type=int, default=[0, 1, 50, 100])
    a = ap.parse_args()
    if a.mode == 'checks': checks(a.parts)
    elif a.mode == 'measure': measure(a.arms, a.seeds)
    elif a.mode == 'merge': merge()
    else: timing(a.arm, a.seed, a.tasks)
