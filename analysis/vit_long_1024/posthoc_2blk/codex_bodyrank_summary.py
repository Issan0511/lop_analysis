"""Summarize the newly measured decomposition and preregistered quantile model."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_codex_bodyrank'
STATES = [('SL', 0, 525), ('SL', 10, 200), ('SL', 40, 250), ('SL', 60, 425), ('SL', 100, 575),
          ('R0L', 10, 100), ('R0L', 40, 100), ('R0L', 60, 100), ('R0L', 100, 75)]


def tail(x):
    return dict(q99_med=float(np.quantile(x, .99) / np.median(x)),
                q90_q10=float(np.quantile(x, .9) / np.quantile(x, .1)),
                logvar=float(np.log(x).var()), median=float(np.median(x)),
                top5_trace=float(np.sort(x*x)[-int(len(x)*.05):].sum() / (x*x).sum()))


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    def rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        centers = np.cumsum(counts) - (counts + 1) / 2
        return centers[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def qmodel():
    rows = []
    for arm, t, T in STATES:
        obj = json.loads((ROOT / 'posthoc_tangent' / f'{arm}_t{t}.json').read_text())
        row = dict(arm=arm, t=t, observed=T)
        for key in ('rand', 'pc'):
            ns = np.array([d['norms'] for d in obj[key]])
            a2 = (ns*ns).mean(0)
            speed = a2 / np.sqrt(a2.mean())
            invspeed = 1/speed
            pr = np.array([d['pr'] for d in obj[key]])
            eff = (ns*ns).sum(1)**2 / (ns**4).sum(1)
            row[key] = dict(time90=float(np.quantile(invspeed, .9)), time_mean=float(1/speed.mean()),
                            pr=float(pr.mean()), pr_cos=float(np.mean([d['pr_cos'] for d in obj[key]])),
                            norm_only_pr=float(eff.mean()), diagonal_fraction=float(np.mean(pr/eff)),
                            inverse_pr_diagonal=float(np.mean(1/eff)), inverse_pr_offdiagonal=float(np.mean(1/pr-1/eff)))
        rows.append(row)
    for key in ('rand', 'pc'):
        anchor = next(r for r in rows if r['arm']=='SL' and r['t']==10)
        A = anchor['observed']/anchor[key]['time90']
        B = anchor['observed']/anchor[key]['time_mean']
        for row in rows:
            row[key]['predicted_quantile'] = A * row[key]['time90']
            row[key]['predicted_mean'] = B * row[key]['time_mean']
    (OUT/'quantile_model.json').write_text(json.dumps(rows,indent=2))
    print('Quantile model (anchor SL10=200):')
    for r in rows:
        print(r['arm'],r['t'],r['observed'],*[round(r[k]['predicted_quantile'],1) for k in ('rand','pc')])
    return rows


def decompositions():
    result = []
    for arm, t, T in STATES:
        p=OUT/f'decompose_{arm}_t{t}_n256.npz'
        if not p.exists():
            continue
        data=np.load(p); a=data['values'].astype(float); fields=data['fields'].tolist()
        val=lambda k,sl:a[:,sl,fields.index(k)]
        row=dict(arm=arm,t=t)
        for kind,sl in [('rand',slice(0,4)),('pc',slice(4,8))]:
            # RMS across independent readout directions, per image.
            total=val('total_sq',sl).mean(1)
            n=np.sqrt(total)
            sig=val('final.sigma',sl)[:,0]
            final_affine=val('param.final_norm',sl).mean(1)
            upn=np.sqrt(total-final_affine)
            delta=np.sqrt((val('final.delta_norm',sl)**2).mean(1))
            angular=delta*sig
            gain=upn/delta
            factors={'inverse_sigma':1/sig,'angular':angular,'upstream_gain':gain}
            logn=np.log(upn)
            attribution={k:float(np.mean((np.log(v)-np.log(v).mean())*(logn-logn.mean()))/logn.var()) for k,v in factors.items()}
            top=np.argsort(n)[-12:]
            groups={k[6:]:dict(trace_share=float(val(k,sl).mean(1).sum()/total.sum()),
                              top_trace_share=float(val(k,sl).mean(1)[top].sum()/total[top].sum()),
                              rho=corr(val(k,sl).mean(1),total),
                              removed_tail=tail(np.sqrt(np.maximum(total-val(k,sl).mean(1),1e-30)))) for k in fields if k.startswith('param.')}
            correlations={k:corr(val(k,sl).mean(1),n) for k in fields if not k.startswith('param.')}
            summaries={k:dict(median=float(np.median(val(k,sl))),q10=float(np.quantile(val(k,sl),.1)),q90=float(np.quantile(val(k,sl),.9))) for k in fields if not k.startswith('param.')}
            row[kind]=dict(tail=tail(n),sigma_corrected_tail=tail(n*sig),delta_corrected_tail=tail(upn/delta),
                           sigma=tail(sig), attribution=attribution,groups=groups,correlations=correlations,summaries=summaries)
        result.append(row)
    (OUT/'decomposition_summary.json').write_text(json.dumps(result,indent=2))
    for r in result:
        d=r['pc']; print(r['arm'],r['t'],'tail',round(d['tail']['q99_med'],2),'attribution',d['attribution'])
        print('groups',[(k,round(v['trace_share'],3),round(v['top_trace_share'],3)) for k,v in sorted(d['groups'].items(),key=lambda v:-v[1]['trace_share'])[:5]])
        print('correlations',sorted([(k,v) for k,v in d['correlations'].items() if v is not None and '.identity_error' not in k and k!='total_sq'],key=lambda v:-abs(v[1]))[:12])
    return result


if __name__ == '__main__':
    qmodel()
    decompositions()
