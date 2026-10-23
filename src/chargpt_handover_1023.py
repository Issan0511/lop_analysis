#!/usr/bin/env python3
"""chargpt_handover_1023: 担い手の交代を保存状態で確かめる（spec: specs/spec_chargpt_handover_1023.md）。

新しい学習はしない。保存された課題末の状態で、次を CPU・float64 で出す。
  検査 1: FFN0・FFN1 の出力を 0（記録: 平均）に置いたときの現課題 probe の CE の悪化（担い）
  検査 2: 最終 LN の前の残差 t への 5 成分（埋め込み・注意 0・FFN0・注意 1・FFN1）の書き込みの比
  検査 4 の材料: 層ごとの生きた unit の割合・dead005・r_med
定義（probe・CE・生きた unit・checkpoint の読み込み）は analysis/chargpt_seed10_1022/load10.py を import して使う。

使い方（worktree の直下で、-I で起動する）:
  python -I src/chargpt_handover_1023.py extract   # 束から 700 個の checkpoint を展開し sha256 を照合（K0）
  python -I src/chargpt_handover_1023.py gate      # seed 0 で K1〜K5
  python -I src/chargpt_handover_1023.py run       # 全 700 状態（K1〜K3 を毎回）→ results/chargpt_handover_1023/rows.csv
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, csv, datetime, hashlib, importlib.util, json, math, platform, subprocess, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chargpt_handover_1023'
BUNDLE = Path('/home/issan/Projects/obsidian-research-data/chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar')
BUNDLE_MANIFEST = BUNDLE.parent / 'bundle_manifest.json'
BUNDLE_SHA = '5729ec0279f25d884cfec0aef8c4234d2f9788d57208fed99a5c7eff4ad23e61'
SRC_DIR = ROOT / 'analysis/chargpt_latepush_1023/source/src'      # 033679e0 の blob と同じ（spec §2）
SRC_BLOBS = {'chargpt_cl_1019.py': 'cc524f85d5f2d9813a272abb6b9476b065f1c17b', 'chargpt_cl_1019_kune.py': '272e75548fef16d364c464327c8e44d034b9cbb2'}
LOAD10 = ROOT / 'analysis/chargpt_seed10_1022/load10.py'
os.environ['CHARGPT_SRC'] = str(ROOT)       # load10 は $CHARGPT_SRC/results/chargpt_cl_1019/M2/... から checkpoint を読む
sys.path.insert(0, str(SRC_DIR))
_spec = importlib.util.spec_from_file_location('load10', LOAD10)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
np, torch, C, F = A.np, A.torch, A.C, A.F

ARMS = ['C3', 'ReLUcap']
SEEDS = list(range(10))
TASKS = A.TASKS                    # 課題末 35 点（init は使わない）
assert TASKS == list(range(1, 11)) + [25] + list(range(39, 51)) + list(range(89, 101))
COMPS = ['emb', 'attn0', 'ffn0', 'attn1', 'ffn1']
ABL = ['zero0', 'zero1', 'zero01', 'mean0', 'mean1']
TOL_K1 = 1e-8
TOL_K2 = 1e-12
TOL_K3 = 1e-12
TOL_K4 = 1e-12
GATE_TASKS = [1, 50, 100]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def rel(a, b):
    a, b = float(a), float(b)
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d > 0 else 0.


def member(arm, seed, t):
    return f'results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts/task{t}.pt'


# ---------------------------------------------------------------- 展開（K0）
def extract():
    man = json.loads(BUNDLE_MANIFEST.read_text())
    entries = {e['path']: e for e in man['files']}
    tar_sha = sha(BUNDLE)
    ok_tar = tar_sha == BUNDLE_SHA == man['tar_sha256']
    if not ok_tar:
        raise SystemExit(f'K0: 束の sha256 が違う {tar_sha}')
    want = [member(a, s, t) for a in ARMS for s in SEEDS for t in TASKS]
    bad = []
    with tarfile.open(BUNDLE, 'r:') as tf:
        for name in want:
            ti = tf.getmember(name)
            if not ti.isfile():
                raise SystemExit(f'K0: {name} が通常の file でない')
            dst = ROOT / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(ti) as src, open(dst, 'wb') as out:
                while b := src.read(1 << 22):
                    out.write(b)
            got = sha(dst)
            if got != entries[name]['sha256'] or dst.stat().st_size != entries[name]['bytes']:
                bad.append(dict(path=name, sha256=got, expected=entries[name]['sha256']))
    rec = dict(check='K0_bundle', pass_=ok_tar and not bad, tar_sha256=tar_sha, manifest_tar_sha256=man['tar_sha256'],
               n_extracted=len(want), n_bad=len(bad), bad=bad[:20], extracted_bytes=sum(entries[n]['bytes'] for n in want),
               extracted_root=str(ROOT), files=[dict(path=n, bytes=entries[n]['bytes'], sha256=entries[n]['sha256']) for n in want])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'k0_extract.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False) + '\n')
    print(f"K0 {'PASS' if rec['pass_'] else 'FAIL'}: {len(want)} files, bad {len(bad)}", flush=True)
    if bad:
        raise SystemExit('K0 FAIL')


# ---------------------------------------------------------------- forward（GPT.forward と同じ演算の順）
def forward_parts(m, x, repl=None):
    """repl: {('ffn'|'attn', l): f -> 置き換え}。返り値 logits・最終 LN の入力 t・成分・各層の前活性 z。"""
    h = m.wte(x)
    parts = {'emb': h}
    zs = []
    for l, b in enumerate(m.h):
        a = b.sc * b.attn(b.n1(h))
        if repl is not None and ('attn', l) in repl:
            a = repl[('attn', l)](a)
        h = h + a
        cap = []
        f = b.mlp(b.n2(h), cap)
        zs.append(cap[0])
        if repl is not None and ('ffn', l) in repl:
            f = repl[('ffn', l)](f)
        h = h + f
        parts[f'attn{l}'] = a
        parts[f'ffn{l}'] = f
    return m.head(m.nf(h)), h, parts, zs


def repl_of(kind, means):
    zero = lambda f: torch.zeros_like(f)
    if kind == 'zero0': return {('ffn', 0): zero}
    if kind == 'zero1': return {('ffn', 1): zero}
    if kind == 'zero01': return {('ffn', 0): zero, ('ffn', 1): zero}
    if kind == 'mean0': return {('ffn', 0): lambda f: means[0].expand_as(f).clone()}
    if kind == 'mean1': return {('ffn', 1): lambda f: means[1].expand_as(f).clone()}
    if kind == 'identity': return {('ffn', 0): lambda f: f, ('ffn', 1): lambda f: f}       # K5 (d)
    if kind == 'zero_attn0': return {('attn', 0): zero}                                     # K5 (c)
    raise ValueError(kind)


def ce_pair(logits, y):
    """load10 の head の L と同じ式（全 token）と、位置 0 を除いた CE。"""
    f = logits.reshape(-1, C.V); yy = y.reshape(-1)
    full = float(A.ce(f, yy))
    tok = F.cross_entropy(f, yy, reduction='none').reshape(y.shape)
    return full, float(tok[:, 1:].mean())


@torch.no_grad()
def layer_stats(z):
    alive = (z > 0).any(0)
    p = 1.0 - (z < 0).double().mean(0)
    mu = z.mean(0); sd = z.std(0)
    r = mu / (sd + 1e-12)
    return dict(n_alive=int(alive.sum()), alive=float(alive.double().mean()), dead005=float((p < 0.005).double().mean()),
                r_med=float(torch.quantile(r, 0.5)), m_median=float(mu.median()))


@torch.no_grad()
def measure_state(arm, seed, t, ckpt_task=None, probe_task=None, extra=()):
    """1 状態の計算。ckpt_task・probe_task は K5 の変異でだけ t と違う値にする。"""
    m = A.load(arm, seed, t if ckpt_task is None else ckpt_task)
    x, y, _, _ = A.data(seed, t if probe_task is None else probe_task)
    logits, ref_logits, zs, fsum, toks = [], [], [[], []], [None, None], 0
    norm = {c: [] for c in COMPS + ['t']}; ratio = {c: [] for c in COMPS}; proj = {c: [] for c in COMPS}; cent = {c: [] for c in COMPS}
    k3 = 0.
    for _, xx in A.chunks(x):
        lg, tt, parts, z = forward_parts(m, xx)
        got = {}
        hk = m.nf.register_forward_hook(lambda mod, inp, out: got.__setitem__('t', inp[0].detach()))
        ref = m(xx)
        hk.remove()
        logits.append(lg); ref_logits.append(ref)
        for l in range(2):
            zs[l].append(z[l].reshape(-1, C.FF))
            s = parts[f'ffn{l}'].reshape(-1, C.D).sum(0)
            fsum[l] = s if fsum[l] is None else fsum[l] + s
        toks += xx.numel()
        T = tt.reshape(-1, C.D)
        tot = sum(parts[c] for c in COMPS).reshape(-1, C.D)        # 埋め込み + 注意 0 + FFN0 + 注意 1 + FFN1（加算の順は残差と同じ）
        th = got['t'].reshape(-1, C.D)
        k3 = max(k3, float(((tot - th).norm(dim=1) / th.norm(dim=1)).max()), float(((T - th).norm(dim=1) / th.norm(dim=1)).max()))
        tn = T.norm(dim=1); Pt = T - T.mean(1, keepdim=True); ptn = Pt.norm(dim=1)
        norm['t'].append(tn)
        for c in COMPS:
            v = parts[c].reshape(-1, C.D)
            vn = v.norm(dim=1)
            norm[c].append(vn); ratio[c].append(vn / tn); proj[c].append((v * T).sum(1) / tn.square())
            cent[c].append((v - v.mean(1, keepdim=True)).norm(dim=1) / ptn)
    L = torch.cat(logits); R = torch.cat(ref_logits)
    k2 = float((L - R).abs().max()) / float(R.abs().max())
    ce_a, ce_a0 = ce_pair(L, y)
    means = [fsum[l] / toks for l in range(2)]
    row = dict(arm=arm, seed=seed, task=t, ce_a=ce_a, ce_a_x0=ce_a0)
    for kind in ABL + list(extra):
        out = torch.cat([forward_parts(m, xx, repl_of(kind, means))[0] for _, xx in A.chunks(x)])
        row[f'ce_{kind}'], row[f'ce_{kind}_x0'] = ce_pair(out, y)
    for kind, key in [('zero0', 'carry0'), ('zero1', 'carry1'), ('zero01', 'carry01'), ('mean0', 'mcarry0'), ('mean1', 'mcarry1')]:
        row[key] = row[f'ce_{kind}'] - ce_a
        row[f'{key}_x0'] = row[f'ce_{kind}_x0'] - ce_a0
    for l in range(2):
        st = layer_stats(torch.cat(zs[l]))
        row.update({f'L{l}_{k}': v for k, v in st.items()})
    for c in COMPS:
        row[f'w_{c}'] = float(torch.cat(ratio[c]).mean())
        row[f'pi_{c}'] = float(torch.cat(proj[c]).mean())
        row[f'wc_{c}'] = float(torch.cat(cent[c]).mean())
        row[f'norm_{c}'] = float(torch.cat(norm[c]).mean())
    row['norm_t'] = float(torch.cat(norm['t']).mean())
    row['pi_sum'] = sum(row[f'pi_{c}'] for c in COMPS)
    row['k2_rel'] = k2; row['k3_rel'] = k3
    return row, m, x, y, means


# ---------------------------------------------------------------- 照合（K1）
class Ref:
    def __init__(self):
        d = ROOT / 'results/chargpt_seed10_1022/load'
        self.head = {(r['arm'], int(r['seed']), int(r['task'])): r for r in csv.DictReader(open(d / 'head.csv'))}
        self.units = {(r['arm'], int(r['seed']), int(r['task']), int(r['layer'])): r for r in csv.DictReader(open(d / 'states.csv'))}
        self.paths = {str(d / 'head.csv'): sha(d / 'head.csv'), str(d / 'states.csv'): sha(d / 'states.csv')}

    def k1(self, row, arm, seed, t):
        h = self.head[(arm, seed, t)]
        e_ce = rel(row['ce_a'], h['L'])
        alive_ok = all(row[f'L{l}_n_alive'] == int(self.units[(arm, seed, t, l)]['n_alive']) for l in range(2))
        e_m = max(rel(row[f'L{l}_m_median'], self.units[(arm, seed, t, l)]['m_median']) for l in range(2))
        return dict(k1_ce_rel=e_ce, k1_mmed_rel=e_m, k1_alive_ok=alive_ok, k1_pass=bool(e_ce <= TOL_K1 and e_m <= TOL_K1 and alive_ok))


def k123(row):
    return bool(row['k1_pass'] and row['k2_rel'] <= TOL_K2 and row['k3_rel'] <= TOL_K3)


# ---------------------------------------------------------------- gate（K1〜K5、seed 0）
@torch.no_grad()
def patched_ce(m, x, y, kind, means):
    """K4: module の forward を差し替えて m(x) を通す別の経路。"""
    olds = []
    targets = {'zero0': [0], 'zero1': [1], 'zero01': [0, 1], 'mean0': [0], 'mean1': [1]}[kind]
    for l in targets:
        mod = m.h[l].mlp; orig = mod.forward
        if kind.startswith('zero'):
            mod.forward = lambda yy, cap=None, orig=orig: torch.zeros_like(orig(yy, cap))
        else:
            mod.forward = lambda yy, cap=None, orig=orig, mu=means[l]: mu.expand_as(orig(yy, cap)).clone()
        olds.append((mod, orig))
    try:
        out = torch.cat([m(xx) for _, xx in A.chunks(x)])
    finally:
        for mod, _ in olds:
            del mod.forward          # instance 属性を消して class の forward に戻す
    return ce_pair(out, y)[0]


def gate():
    ref = Ref(); rec = []; t0 = time.time()
    def add(name, ok, **kw):
        r = dict(check=name, pass_=bool(ok), **kw); rec.append(r)
        print(json.dumps(r, ensure_ascii=False)[:300], flush=True)
    # 1 seed・1 状態（両腕の seed 0 の課題末 1）で K1〜K3
    for arm in ARMS:
        row, *_ = measure_state(arm, 0, 1)
        row.update(ref.k1(row, arm, 0, 1))
        add('K1_gate_state', row['k1_pass'], arm=arm, seed=0, task=1, ce=row['ce_a'], ref_ce=float(ref.head[(arm, 0, 1)]['L']), ce_rel=row['k1_ce_rel'],
            n_alive=[row['L0_n_alive'], row['L1_n_alive']], ref_n_alive=[int(ref.units[(arm, 0, 1, l)]['n_alive']) for l in range(2)], mmed_rel=row['k1_mmed_rel'])
        add('K2_gate_state', row['k2_rel'] <= TOL_K2, arm=arm, seed=0, task=1, max_abs_rel=row['k2_rel'])
        add('K3_gate_state', row['k3_rel'] <= TOL_K3, arm=arm, seed=0, task=1, max_rel=row['k3_rel'])
    # K4: 別の経路の切除
    for arm in ARMS:
        for t in GATE_TASKS:
            row, m, x, y, means = measure_state(arm, 0, t, extra=('identity', 'zero_attn0'))
            diffs = {k: rel(row[f'ce_{k}'], patched_ce(m, x, y, k, means)) for k in ABL}
            add('K4_patched_path', max(diffs.values()) <= TOL_K4, arm=arm, seed=0, task=t, rel_diffs=diffs)
            # K5 (c) 注意 0 を消した CE は FFN0 を消した CE と違う、(d) 恒等の置き換えで差が 0
            add('K5c_attn_not_ffn', abs(row['ce_zero_attn0'] - row['ce_zero0']) > 1e-6, arm=arm, seed=0, task=t,
                ce_zero_attn0=row['ce_zero_attn0'], ce_zero0=row['ce_zero0'])
            add('K5d_identity_zero', row['ce_identity'] == row['ce_a'] and row['ce_identity_x0'] == row['ce_a_x0'], arm=arm, seed=0, task=t,
                diff=row['ce_identity'] - row['ce_a'])
    # K5 (a)(b): 取り違えた checkpoint・probe で K1 が落ちる
    for arm in ARMS:
        for t, (ck, pr) in [(1, (2, 1)), (1, (1, 2)), (50, (89, 50)), (50, (50, 89))]:
            row, *_ = measure_state(arm, 0, t, ckpt_task=ck, probe_task=pr)
            k = ref.k1(row, arm, 0, t)
            add('K5ab_mutant_fails_K1', not k['k1_pass'], arm=arm, seed=0, task=t, ckpt_task=ck, probe_task=pr, ce_rel=k['k1_ce_rel'],
                mmed_rel=k['k1_mmed_rel'], alive_ok=k['k1_alive_ok'])
    res = dict(all_pass=all(r['pass_'] for r in rec), n_checks=len(rec), n_pass=sum(r['pass_'] for r in rec),
               seconds=time.time() - t0, script_sha256=sha(Path(__file__)), checks=rec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gate.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(f"gate {'PASS' if res['all_pass'] else 'FAIL'} {res['n_pass']}/{res['n_checks']} {res['seconds']:.1f}s", flush=True)
    return res


# ---------------------------------------------------------------- 本走（全 700 状態）
def provenance(extra):
    srcs = {p.name: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), git_blob=git('hash-object', str(p))) for p in sorted(SRC_DIR.glob('*.py'))}
    assert all(srcs[k]['git_blob'] == v for k, v in SRC_BLOBS.items())
    return dict(experiment='chargpt_handover_1023', spec='specs/spec_chargpt_handover_1023.md', spec_sha256=sha(ROOT / 'specs/spec_chargpt_handover_1023.md'),
                git_hash=git('rev-parse', 'HEAD'), git_dirty=git('status', '--porcelain', '--untracked-files=no'),
                script=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(Path(__file__)),
                load10=str(LOAD10.relative_to(ROOT)), load10_sha256=sha(LOAD10), model_source=srcs, model_source_commit='033679e0f1a3d313e1e3110e22c8f1cac3c36628',
                bundle=str(BUNDLE), bundle_sha256=BUNDLE_SHA, device='cpu', dtype='float64', threads=torch.get_num_threads(), torch=torch.__version__,
                numpy=np.__version__, python=sys.version, host=platform.node(), probe=dict(kind='chain probe of the state task', sequences=C.M_PROBE, tokens=C.T),
                arms=ARMS, seeds=SEEDS, tasks=TASKS, **extra)


def run():
    g = json.loads((OUT / 'gate.json').read_text())
    if not g['all_pass']:
        raise SystemExit('gate が通っていない。本走は禁止。')
    if g['script_sha256'] != sha(Path(__file__)):
        raise SystemExit('gate の後に script が変わった。gate を回し直す。')
    ref = Ref(); rows = []; t0 = time.time(); started = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    for arm in ARMS:
        for seed in SEEDS:
            ts = time.time()
            for t in TASKS:
                row, *_ = measure_state(arm, seed, t)
                row.update(ref.k1(row, arm, seed, t)); row['k123_pass'] = k123(row)
                rows.append(row)
                if not row['k123_pass']:
                    print(f'K1〜K3 FAIL {arm} seed{seed} t{t}: {row}', flush=True)
            print(f'{arm} seed{seed}: {time.time()-ts:.1f}s（累計 {time.time()-t0:.0f}s）', flush=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = OUT / 'rows.csv.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    tmp.replace(OUT / 'rows.csv')
    k = dict(n_states=len(rows), k1_pass=sum(r['k1_pass'] for r in rows), k2_pass=sum(r['k2_rel'] <= TOL_K2 for r in rows),
             k3_pass=sum(r['k3_rel'] <= TOL_K3 for r in rows), max_k1_ce_rel=max(r['k1_ce_rel'] for r in rows), max_k1_mmed_rel=max(r['k1_mmed_rel'] for r in rows),
             all_alive_equal=all(r['k1_alive_ok'] for r in rows), max_k2_rel=max(r['k2_rel'] for r in rows), max_k3_rel=max(r['k3_rel'] for r in rows),
             max_abs_pi_sum_minus_1=max(abs(r['pi_sum'] - 1) for r in rows))
    k['all_pass'] = k['k1_pass'] == k['k2_pass'] == k['k3_pass'] == len(rows) == 700
    k0 = json.loads((OUT / 'k0_extract.json').read_text())
    checks = dict(all_pass_src=bool(k0['pass_'] and g['all_pass'] and k['all_pass']),
                  K0=dict(pass_=k0['pass_'], tar_sha256=k0['tar_sha256'], n_extracted=k0['n_extracted'], n_bad=k0['n_bad']),
                  gate=g, all_states=dict(tolerance=dict(K1=TOL_K1, K2=TOL_K2, K3=TOL_K3), **k))
    (OUT / 'checks.json').write_text(json.dumps(checks, indent=1, ensure_ascii=False) + '\n')
    pv = provenance(dict(started=started, finished=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), seconds=time.time() - t0,
                         reference_inputs=ref.paths, rows_sha256=sha(OUT / 'rows.csv'), checkpoints_sha256_from='results/chargpt_handover_1023/k0_extract.json'))
    (OUT / 'provenance.json').write_text(json.dumps(pv, indent=1, ensure_ascii=False) + '\n')
    print(f"run done {len(rows)} states {time.time()-t0:.0f}s, K1〜K3 {'PASS' if k['all_pass'] else 'FAIL'}", flush=True)


def timing():
    for arm, t in [('C3', 1), ('C3', 100)]:
        ts = time.time(); measure_state(arm, 0, t); print(f'{arm} t{t}: {time.time()-ts:.2f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['extract', 'gate', 'run', 'timing'])
    a = ap.parse_args()
    {'extract': extract, 'gate': gate, 'run': run, 'timing': timing}[a.mode]()
