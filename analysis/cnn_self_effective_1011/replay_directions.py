#!/usr/bin/env python3
"""Fixed raw-input PCA accounting of integrated, decomposed replay updates.

Variance/norm component attribution uses the symmetric endpoint convention:
u_k dot (w_after+w_before), not the per-step path convention in replay.py.
These additive attributions are conditional on the actual Adam denominator,
and are not counterfactual component-ablation effects.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BANDS = [(0, 1, 'PC1'), (1, 5, 'PC2-5'), (5, 20, 'PC6-20'), (20, 75, 'PC21-75')]


def f(x):
    return float(x)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w') as ff:
        writer = csv.DictWriter(ff, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_budget(early_dir, late_dir, prefix):
    """Actual endpoint changes only; no decomposed large cancelling updates."""
    e, l = np.load(early_dir/'raw.npz'), np.load(late_dir/'raw.npz')
    es, ls = np.load(early_dir/'input_stats.npz'), np.load(late_dir/'input_stats.npz')
    ep = json.loads((early_dir/'provenance.json').read_text())
    lp = json.loads((late_dir/'provenance.json').read_text())
    assert ep['slots'] == lp['slots'] and ep['task'] == lp['task']
    assert ep['start_step'] == 0 and ep['steps'] <= lp['start_step']
    assert np.array_equal(es['cov'],ls['cov']) and np.array_equal(es['mu'],ls['mu'])
    nr=len(ep['slots']); mu,cov=es['mu'],es['cov']
    lam,q=np.linalg.eigh(cov); lam,q=lam[:,::-1],q[:,:,::-1]
    ws=[x.astype(np.float64).reshape(nr,16,75) for x in [e['W_before'],e['W_after'],l['W_before'],l['W_after']]]
    bs=[x.astype(np.float64) for x in [e['b_before'],e['b_after'],l['b_before'],l['b_after']]]
    aa=[np.einsum('rci,rik->rck',w,q) for w in ws]
    mupc=np.einsum('ri,rik->rk',mu,q)
    phases=[('first75',0,1,ep['steps']),('middle',1,2,lp['start_step']-ep['steps']),
            ('last75',2,3,lp['steps']),('fulltask',0,3,lp['start_step']+lp['steps'])]
    allrows=[];seedrows=[];channels=[]
    for phase,i,j,steps in phases:
        dn=aa[j]**2-aa[i]**2; dv=dn*lam[:,None,:]
        dm=(aa[j]-aa[i])*mupc[:,None,:];db=bs[j]-bs[i]
        for lo,hi,band in BANDS+[(0,75,'all')]:
            n=dn[:,:,lo:hi].sum(-1);v=dv[:,:,lo:hi].sum(-1)
            m=dm[:,:,lo:hi].sum(-1);b=db if band=='all' else np.zeros_like(db)
            row=dict(phase=phase,steps=steps,band=band,mean_delta_norm2=f(n.mean()),
                     mean_delta_variance=f(v.mean()),mean_delta_mu=f(m.mean()),mean_delta_bias=f(b.mean()),
                     mean_delta_mean=f((m+b).mean()),norm_grows_channels=int((n>0).sum()),
                     variance_shrinks_channels=int((v<0).sum()),mean_rises_channels=int((m+b>0).sum()),channels=int(n.size))
            allrows.append(row)
            for r,(arm,seed) in enumerate(ep['slots']):
                seedrows.append(dict(phase=phase,steps=steps,seed=seed,band=band,
                                     mean_delta_norm2=f(n[r].mean()),mean_delta_variance=f(v[r].mean()),
                                     mean_delta_mean=f((m[r]+b[r]).mean()),norm_grows_channels=int((n[r]>0).sum())))
                for c in range(16):
                    channels.append(dict(phase=phase,seed=seed,channel=c+1,band=band,
                                         delta_norm2=f(n[r,c]),delta_variance=f(v[r,c]),
                                         delta_mu=f(m[r,c]),delta_bias=f(b[r,c])))
    closure={}
    for key in ['mean_delta_norm2','mean_delta_variance','mean_delta_mean']:
        closure[key]=max(abs(sum(r[key] for r in allrows if r['band']==band and r['phase']!='fulltask')-
                            next(r[key] for r in allrows if r['band']==band and r['phase']=='fulltask'))
                         for band in ['PC1','PC2-5','PC6-20','PC21-75','all'])
    assert max(closure.values())<1e-12,closure
    result=dict(method='Direct actual parameter endpoints; no self/cross/label component decomposition',
                task=ep['task'],slots=ep['slots'],early_dir=str(early_dir),late_dir=str(late_dir),
                closure=closure,summary=allrows,seeds=seedrows,channels=channels)
    Path(str(prefix)+'.json').write_text(json.dumps(result,indent=2))
    lines=['# task31 の実際の方向別増減と時期', '',
           '実際のfloat32パラメータをfloat64へ変換し、同じ固定入力共分散の主成分で端点を比較。自己項等への配分を使わないため、late窓の成分間の大きな打ち消しの影響を受けない。', '',
           '| 区間 | 更新数 | 方向 | Δ重み二乗ノルム | Δ分散 | μ経路Δ平均 | biasΔ平均 | 増加チャネル数 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in allrows:
        lines.append('| '+' | '.join([r['phase'],str(r['steps']),r['band']]+[f'{r[k]:+.7g}' for k in ['mean_delta_norm2','mean_delta_variance','mean_delta_mu','mean_delta_bias']]+[f'{r["norm_grows_channels"]}/{r["channels"]}'])+' |')
    lines+=['',f'区間加算の最大誤差: {max(closure.values()):.3g}。全値は160チャネル平均。', '',
            '課題のどの時期に増減したかを示す帳簿であり、その時期の更新を発生させた機構の同定ではない。']
    Path(str(prefix)+'.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-prefix', type=Path, required=True)
    ap.add_argument('--phase-early-dir', type=Path)
    ap.add_argument('--phase-budget-prefix', type=Path)
    args = ap.parse_args()
    src = args.input_dir
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    def dest(suffix):
        return Path(str(prefix)+suffix)
    raw = np.load(src/'raw.npz')
    stats = np.load(src/'input_stats.npz')
    provenance = json.loads((src/'provenance.json').read_text())
    comps = raw['components'].tolist()
    slots = provenance['slots']
    mu, cov = stats['mu'], stats['cov']
    nr = len(mu)
    before = raw['W_before'].astype(np.float64).reshape(nr, 16, 75)
    after = raw['W_after'].astype(np.float64).reshape(nr, 16, 75)
    bb = raw['b_before'].astype(np.float64)
    ba = raw['b_after'].astype(np.float64)
    u = raw['integrated_theta']
    assert u.shape == (len(comps), nr, 16, 76)
    closure = dict(parameter_max_abs=f(abs(u.sum(0)-np.concatenate([after-before,(ba-bb)[...,None]],-1)).max()))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]
    ab = np.einsum('rci,rik->rck', before, eigenvectors)
    aa = np.einsum('rci,rik->rck', after, eigenvectors)
    upc = np.einsum('nrci,rik->nrck', u[:,:,:,:75], eigenvectors)
    mupc = np.einsum('ri,rik->rk', mu, eigenvectors)
    norm_attr = upc*(aa+ab)[None]
    var_attr = norm_attr*eigenvalues[None,:,None,:]
    mu_attr = upc*mupc[None,:,None,:]
    bias_attr = u[:,:,:,75]
    actual_norm = aa**2-ab**2
    actual_var = actual_norm*eigenvalues[:,None,:]
    actual_mu = (aa-ab)*mupc[:,None,:]
    actual_mean = actual_mu.sum(-1)+ba-bb
    closure.update(
        norm_per_mode_max_abs=f(abs(norm_attr.sum(0)-actual_norm).max()),
        variance_per_mode_max_abs=f(abs(var_attr.sum(0)-actual_var).max()),
        mean_per_component_max_abs=f(abs(mu_attr.sum(-1)+bias_attr-raw['mean_contributions'].sum(0)).max()),
        norm_endpoint_max_abs=f(abs(norm_attr.sum((0,3))-(after**2-before**2).sum(-1)).max()),
        variance_endpoint_max_abs=f(abs(var_attr.sum((0,3))-(raw['variance_after']-raw['variance_before'])).max()),
        mean_endpoint_max_abs=f(abs(mu_attr.sum((0,3))+bias_attr.sum(0)-(raw['mean_after']-raw['mean_before'])).max()))
    # Explicitly retain absolute errors, and separately evaluate floating-point
    # closure relative to the magnitude of the cancelling operands. Passing
    # this arithmetic check does NOT make a highly cancelling attribution useful.
    scales=dict(parameter_max_abs=max(1.,f(abs(u).sum(0).max())),
                norm_per_mode_max_abs=max(1.,f(abs(norm_attr).sum(0).max())),
                variance_per_mode_max_abs=max(1.,f(abs(var_attr).sum(0).max())),
                mean_per_component_max_abs=max(1.,f(abs(mu_attr.sum(-1)+bias_attr).max())),
                norm_endpoint_max_abs=max(1.,f(abs(norm_attr).sum((0,3)).max())),
                variance_endpoint_max_abs=max(1.,f(abs(var_attr).sum((0,3)).max())),
                mean_endpoint_max_abs=max(1.,f((abs(mu_attr).sum((0,3))+abs(bias_attr).sum(0)).max())))
    scaled={k:closure[k]/scales[k] for k in closure}
    arithmetic_validation=dict(absolute_guard=1e-9,absolute_guard_pass=max(closure.values())<1e-9,
                               absolute_errors=closure,operand_scales=scales,scaled_errors=scaled,
                               scaled_guard=5e-12,scaled_guard_pass=max(scaled.values())<5e-12)
    assert arithmetic_validation['scaled_guard_pass'], arithmetic_validation
    compmean=mu_attr.sum(-1)+bias_attr
    compvar=var_attr.sum(-1)
    def cond(c,observed):
        return f(abs(c).sum()/max(f(abs(observed).sum()),1e-30))
    condition=dict(mean_l1_cancellation_index=cond(compmean,actual_mean),
                   variance_l1_cancellation_index=cond(compvar,actual_var.sum(-1)),
                   mean_aggregate_cancellation_index=f(abs(compmean.mean((1,2))).sum()/max(abs(f(actual_mean.mean())),1e-30)))
    ni=comps.index('numerical_residual')
    condition['numerical_residual_mean_l1_over_net_mean_l1']=f(abs(compmean[ni]).sum()/max(f(abs(actual_mean).sum()),1e-30))
    ill_conditioned=condition['mean_l1_cancellation_index']>1e6 or condition['numerical_residual_mean_l1_over_net_mean_l1']>.1
    phase='task_start' if provenance.get('start_step',0)==0 else 'within_task'
    status='ill_conditioned_do_not_use_component_percentages_or_causal_claims' if ill_conditioned else 'arithmetic_attribution_with_cancellation_reported'
    # Append the exact observed net update as a distinct row for comparison.
    names = comps+['net']
    norms = np.concatenate([norm_attr,actual_norm[None]],0)
    variances = np.concatenate([var_attr,actual_var[None]],0)
    means = np.concatenate([mu_attr,actual_mu[None]],0)
    biases = np.concatenate([bias_attr,(ba-bb)[None]],0)
    rows, seed_rows, summaries = [], [], []
    for ki,k in enumerate(names):
        for lo,hi,band in BANDS+[(0,75,'all')]:
            dn = norms[ki,:,:,lo:hi].sum(-1)
            dv = variances[ki,:,:,lo:hi].sum(-1)
            dm = means[ki,:,:,lo:hi].sum(-1)
            db = biases[ki] if band == 'all' else np.zeros_like(dm)
            for r,(arm,seed) in enumerate(slots):
                for c in range(16):
                    rows.append(dict(arm=arm,seed=seed,channel=c+1,component=k,band=band,
                                     delta_norm2=f(dn[r,c]),delta_variance=f(dv[r,c]),
                                     delta_mu=f(dm[r,c]),delta_bias=f(db[r,c]),delta_mean=f(dm[r,c]+db[r,c])))
                seed_rows.append(dict(arm=arm,seed=seed,component=k,band=band,
                                      mean_delta_norm2=f(dn[r].mean()),mean_delta_variance=f(dv[r].mean()),
                                      mean_delta_mu=f(dm[r].mean()),mean_delta_bias=f(db[r].mean()),
                                      mean_delta_mean=f((dm[r]+db[r]).mean()),
                                      norm2_negative_channels=int((dn[r]<0).sum()),
                                      variance_negative_channels=int((dv[r]<0).sum()),
                                      mean_positive_channels=int((dm[r]+db[r]>0).sum())))
            summaries.append(dict(component=k,band=band,mean_delta_norm2=f(dn.mean()),
                                  mean_delta_variance=f(dv.mean()),mean_delta_mu=f(dm.mean()),
                                  mean_delta_bias=f(db.mean()),mean_delta_mean=f((dm+db).mean()),
                                  norm2_negative_channels=int((dn<0).sum()),
                                  variance_negative_channels=int((dv<0).sum()),
                                  mean_positive_channels=int((dm+db>0).sum()),channels=int(dn.size)))
    write_csv(dest('_channels.csv'),rows)
    write_csv(dest('_seed_summary.csv'),seed_rows)
    out=dict(input_dir=str(src.resolve()),raw_sha256=sha(src/'raw.npz'),
             input_stats_sha256=sha(src/'input_stats.npz'),source_provenance=provenance,
             convention='Component variance and norm changes use integrated component update dotted with endpoint parameter sum. Exact additive symmetric endpoint convention, not per-step path convention, not causal ablation.',
             phase=phase,condition_index=condition,component_attribution_status=status,
             closure=closure,arithmetic_validation=arithmetic_validation,summaries=summaries)
    dest('_summary.json').write_text(json.dumps(out,indent=2))
    start=provenance.get('start_step',0)
    lines=[f'# 実更新の方向別帳簿: task{provenance["task"]}, step{start+1}–{start+provenance["steps"]}', '',
           f'元run: `{src}`。seeds={",".join(str(x[1]) for x in slots)}、{len(slots)*16}チャネル。', '',
           '実際のAdam更新の各成分を積算し、固定した生入力パッチ共分散の主成分に射影した。下記はチャネル平均で、ノルムは二乗ノルム。', '',
           '**分散・二乗ノルムの寄与は、各積算更新と始点＋終点の重みの内積で割り当てる対称的な端点規約。** 全成分で厳密に閉じるが、一意な因果寄与ではない。`replay.py`のステップごとの中点規約とは成分別の値が異なりうる。', '',
           '| 成分 | 方向 | Δ重み二乗ノルム | Δ前活性分散 | μ経路のΔ平均 | biasのΔ平均 | Δ平均 |',
           '|---|---|---:|---:|---:|---:|---:|']
    if ill_conditioned:
        lines[2:2]=['**成分別の割合や因果説明には使用不可。** 巨大な成分の打ち消しがあり、数値残差も実更新に対して大きい。下の成分値は計算上の帳簿としてのみ保存する。実パラメータ端点から計算したnetは別扱いで有効。', '']
    for row in summaries:
        if row['component'] not in ['self','cross_bias','label','history','net']:
            continue
        lines.append('| '+' | '.join([row['component'],row['band']]+[f'{row[k]:+.7g}' for k in ['mean_delta_norm2','mean_delta_variance','mean_delta_mu','mean_delta_bias','mean_delta_mean']])+' |')
    lines+=['', '数値残差と丸めもCSV/JSONには含む。`all`行だけbiasを含む。他の行のΔ平均はその主成分帯のμ経路であり、bias=0。', '',
            f'全成分の加算閉包の最大絶対誤差: `{max(closure.values()):.9g}`。元の絶対閾値1e−9: {arithmetic_validation["absolute_guard_pass"]}。',
            f'打ち消す項の大きさで割った最大誤差: `{max(scaled.values()):.9g}`。別途明記した相対閾値5e−12: {arithmetic_validation["scaled_guard_pass"]}。この相対検算の成功は成分解釈の安定性を保証しない。',
            f'平均のL1打ち消し指数: `{condition["mean_l1_cancellation_index"]:.9g}`。分散のL1打ち消し指数: `{condition["variance_l1_cancellation_index"]:.9g}`。数値残差の平均寄与L1 / 実平均変化L1: `{condition["numerical_residual_mean_l1_over_net_mean_l1"]:.9g}`。', '',
            '生入力PCAはgate/poolや下流誤差信号を含む実効Gramと異なる。このファイルは、生入力のどの方向へ実際の各更新成分が動いたかを示す。', '',
            '平均寄与は線形なので端点・経路の規約に依存しない。重みの二乗ノルムと分散については、成分同士の交差項の配分規約に依存する。Adam分母は実軌道の共通分母を固定して用いており、その成分を除いた再学習結果を示すものではない。']
    dest('_notes.md').write_text('\n'.join(lines)+'\n')
    if args.phase_early_dir:
        if not args.phase_budget_prefix:
            ap.error('--phase-early-dir requires --phase-budget-prefix')
        phase_budget(args.phase_early_dir,src,args.phase_budget_prefix)
    print(json.dumps(dict(closure=closure,summaries=[r for r in summaries if r['component'] in ['self','net']]),indent=2))


if __name__ == '__main__':
    main()
