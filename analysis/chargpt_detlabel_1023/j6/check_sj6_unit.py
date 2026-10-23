#!/usr/bin/env python3
"""S-J6 の続き（Issa の判断 (a)、2026-10-24）: loadsign とこちらで S > 0 の判定が食い違う unit が近零の unit に限られるかを確かめる。

近零の定義（1 つに固定）:
  v = 符号の割合の列が数える unit ごとの値（例: s_original_other なら第二段階の配列 original_other[i]
      = mean_n K_n φ′(z_ni) other_ni。s_*_self_plus_other は self[i] + other[i]、s_nonzero は original_exact_K[i]）。
  母集団 = その列の割合の分母の unit（生きた unit。s_nonzero だけは生きた unit のうち v ≠ 0）。
  max = 同じ状態・同じ列・同じ側（こちら、または loadsign）の母集団の上の max_i |v_i|。
  近零 ⇔ |v_i| ≤ 1e−12 × max（v_i = 0 を含む。0 の数は別に数える）。
  符号が機械で変わりうる unit（flippable）= 近零のうち、勾配の開かない unit の 0（gradient_active でなく v = 0）を除いたもの。
  unit 配列が無いときの必要条件: loadsign の正の数 − こちらの近零でない正の数 ∈ [0, flippable の数]。
判定用の S > 0 の数え方（load10.sink: 母集団の中で v > 0 の割合）は変えない。

入力:
  こちら = results/chargpt_detlabel_1023/checks/J6/A2_full/{raw,raw_components,raw_loading}/A2_seed<k>.npz（A2 の C3 seed 0〜9、35 状態）
  loadsign の csv = results/chargpt_loadsign_1023/{states,components_states,loading_sources_states}.csv（arm C3）
  loadsign の unit 配列（--ref-dir、任意）= <dir>/{raw,raw_components,raw_loading}/C3_seed<k>.npz。backup_manifest.json の sha256 で照合する。
  unit 配列が無いときは、csv の割合から数え上げで確かめられることだけを確かめる（unit の特定はできない）。
"""
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHK = ROOT/'results/chargpt_detlabel_1023/checks/J6'
OURS = CHK/'A2_full'
REF = ROOT/'results/chargpt_loadsign_1023'
TASKS = list(range(1, 11))+[25]+list(range(39, 51))+list(range(89, 101))
TOL = 1e-12
FILES = {'states.csv': 'raw', 'components_states.csv': 'raw_components', 'loading_sources_states.csv': 'raw_loading'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def columns(header):
    return [c for c in header if c.startswith('s_')]


def value(file, col, z, t):
    """列 col が数える unit ごとの値と母集団（file ごとに code の数え方と同じ）。"""
    alive = z['stage1'][f't{t}_alive'].astype(bool)
    name = col[2:]
    if file == 'states.csv':
        r = z['stage1']
        if name == 'nonzero':
            v = r[f't{t}_original_exact_K']; return v, alive & (v != 0)
        return r[f't{t}_{name}'], alive
    if file == 'components_states.csv':
        r = z['comp']
        for e in ['original', 'loaderror']:
            if name == f'{e}_self_plus_other': return r[f't{t}_{e}_self']+r[f't{t}_{e}_other'], alive
            if name == f'{e}_self_plus_residual': return r[f't{t}_{e}_self']+r[f't{t}_{e}_residual'], alive
        return r[f't{t}_{name}'], alive
    return z['load'][f't{t}_{name}'], alive


def load_side(base, prefix, seed):
    return dict(stage1=np.load(base/'raw'/f'{prefix}_seed{seed}.npz'), comp=np.load(base/'raw_components'/f'{prefix}_seed{seed}.npz'),
                load=np.load(base/'raw_loading'/f'{prefix}_seed{seed}.npz'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ref-dir', type=Path, default=None)
    ap.add_argument('--ref-prefix', default='C3'); ap.add_argument('--out', default='sj6_unit_level')
    args = ap.parse_args()
    ref_csv = {}
    for f in FILES:
        rows = list(csv.DictReader((REF/f).open()))
        ref_csv[f] = ({(int(r['seed']), int(r['task'])): r for r in rows if r['arm'] == 'C3'}, columns(rows[0]))
    ref_units = args.ref_dir is not None
    if ref_units and args.ref_prefix == 'C3':
        man = {m['source'].split('chargpt_loadsign_1023/', 1)[1]: m['sha256'] for m in json.load((REF/'backup_manifest.json').open())['files']}
        for s in range(10):
            for d in FILES.values():
                assert sha(args.ref_dir/d/f'C3_seed{s}.npz') == man[f'{d}/C3_seed{s}.npz'], (d, s)
    # 近零の幅の確かめ: 全 unit の |v|/max の分布（こちら側）
    edges = [0.0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1.0001]
    hist = np.zeros(len(edges)-1, dtype=np.int64); n_zero_all = 0; max_near = 0.0; min_far = math.inf
    n_state_cols = 0; n_mismatch_cols = 0; n_interval_fail = 0; n_nonint = 0; lower_bound_units = 0
    near_per_state = {}; near_dist = {}; tau_count = {}; mismatch_rows = []; unit_rows = []; n_units_disagree = 0; n_units_disagree_near = 0
    n_units_disagree_far = 0; per_col = {}
    for s in range(10):
        ours = load_side(OURS, 'A2', s)
        ref = load_side(args.ref_dir, args.ref_prefix, s) if ref_units else None
        for t in TASKS:
            near_state = 0
            for f, (rc, cols) in ref_csv.items():
                for col in cols:
                    n_state_cols += 1
                    v, mask = value(f, col, ours, t)
                    a = np.abs(v[mask]); mx = float(a.max()) if a.size else 0.0
                    ratio = a/mx if mx > 0 else np.zeros_like(a)
                    n_zero_all += int((a == 0).sum())
                    hist += np.histogram(ratio[a > 0], bins=edges)[0]
                    if (ratio[a > 0] <= TOL).any(): max_near = max(max_near, float(ratio[(a > 0) & (ratio <= TOL)].max()))
                    if (ratio > TOL).any(): min_far = min(min_far, float(ratio[ratio > TOL].min()))
                    near = mask & (np.abs(v) <= TOL*mx)
                    n_near = int(near.sum()); n0 = int((mask & (v == 0)).sum())
                    ga = ours['stage1'][f't{t}_gradient_active'].astype(bool)
                    cat = dict(zero_inactive=int((mask & (v == 0) & ~ga).sum()), zero_active=int((mask & (v == 0) & ga).sum()),
                               nonzero_near=int((near & (v != 0)).sum()))
                    for kc, vc in cat.items():
                        dd = near_dist.setdefault(col, {}).setdefault(kc, {}); dd[vc] = dd.get(vc, 0)+1
                    if col == 's_original_other': near_state = n_near
                    n = int(mask.sum()); pos_far = int((mask & ~near & (v > 0)).sum()); pos = int((mask & (v > 0)).sum())
                    rfrac = float(rc[s, t][col]); rpos_f = rfrac*n if n else 0.0; rpos = round(rpos_f)
                    if n and abs(rpos_f-rpos) > 1e-6: n_nonint += 1
                    # 必要条件: loadsign の正の数 − こちらの「近零でない正」の数 ∈ [0, 近零の数]
                    # 必要条件の検査では、勾配の開かない unit の 0（全項が門 0 で掛かる、どの機械でも 0）は符号が変わりえないので除く
                    flip = near & ~((v == 0) & ~ga); n_flip = int(flip.sum())
                    ok = 0 <= rpos-pos_far <= n_flip
                    n_interval_fail += (not ok)
                    pc = per_col.setdefault(f'{f}:{col}', dict(states=0, mismatch_states=0, interval_fail=0, near_units=0))
                    pc['states'] += 1; pc['near_units'] += n_near; pc['interval_fail'] += (not ok)
                    if rpos != pos:
                        # この食い違いを説明するのに要る最小の幅（近零 ⇔ |v| ≤ τ·max として、区間の条件が成り立つ最小の τ）
                        tau_need = None
                        for tau in [0.0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12]:
                            nr = mask & (np.abs(v) <= tau*mx) & ~((v == 0) & ~ga); pf = int((mask & ~nr & (v > 0)).sum())
                            if 0 <= rpos-pf <= int(nr.sum()): tau_need = tau; break
                        tau_count[str(tau_need)] = tau_count.get(str(tau_need), 0)+1
                        n_mismatch_cols += 1; lower_bound_units += abs(rpos-pos); pc['mismatch_states'] += 1
                        mismatch_rows.append(dict(seed=s, task=t, file=f, col=col, n=n, ours_pos=pos, ref_pos=rpos, n_near=n_near,
                                                  n_near_exact0=n0, n_flippable=n_flip, ours_pos_far=pos_far, interval_ok=ok, tau_needed=tau_need, **cat))
                    if ref_units:
                        rv, rmask = value(f, col, ref, t)
                        assert np.array_equal(rmask, mask) or col == 's_nonzero', (s, t, col)
                        rmx = float(np.abs(rv[rmask]).max()) if rmask.any() else 0.0
                        dis = (mask | rmask) & ((v > 0) != (rv > 0))
                        for i in np.flatnonzero(dis):
                            nb = bool(abs(v[i]) <= TOL*mx and abs(rv[i]) <= TOL*rmx)
                            n_units_disagree += 1; n_units_disagree_near += nb; n_units_disagree_far += (not nb)
                            unit_rows.append(dict(seed=s, task=t, file=f, col=col, unit=int(i), ours=float(v[i]), ref=float(rv[i]),
                                                  ours_max=mx, ref_max=rmx, ours_ratio=abs(float(v[i]))/mx if mx else None,
                                                  ref_ratio=abs(float(rv[i]))/rmx if rmx else None, near_both=nb))
            near_per_state[f'{s}_{t}'] = near_state
    dist = {}
    for k in near_per_state.values(): dist[k] = dist.get(k, 0)+1
    res = dict(definition='v = 列が数える unit ごとの値。母集団 = その列の割合の分母の unit（生きた unit、s_nonzero は v≠0 も）。'
                          'max = 同じ状態・同じ列・同じ側の母集団の上の max|v|。近零 ⇔ |v| ≤ 1e−12·max（0 を含む）。',
               tol=TOL, n_states=350, n_columns=sum(len(c) for _, c in ref_csv.values()), n_state_columns=n_state_cols,
               columns={f: c for f, (_, c) in ref_csv.items()},
               ratio_histogram=dict(edges=edges, counts_nonzero=hist.tolist(), n_exact_zero=n_zero_all),
               largest_nonzero_ratio_within_near=max_near, smallest_ratio_outside_near=min_far,
               n_mismatch_state_columns=n_mismatch_cols, units_disagreeing_lower_bound=lower_bound_units,
               n_ref_count_not_integer=n_nonint, n_interval_fail=n_interval_fail,
               near_units_per_state_original_other_distribution=dict(sorted(dist.items())),
               mismatch_min_tau_needed=tau_count,
               near_units_per_state_by_column=near_dist,
               per_column=per_col, mismatches=mismatch_rows, unit_level_available=ref_units)
    if ref_units:
        res.update(n_units_disagree=n_units_disagree, n_units_disagree_near_both=n_units_disagree_near,
                   n_units_disagree_not_near=n_units_disagree_far, units=unit_rows)
    (CHK/f'{args.out}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False)+'\n')
    print(json.dumps({k: res[k] for k in res if k not in ('per_column', 'mismatches', 'units', 'columns', 'near_units_per_state_by_column', 'definition')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
