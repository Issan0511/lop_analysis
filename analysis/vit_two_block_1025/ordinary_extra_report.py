"""Read complete residual/token diagnostics and preserve seed-level outcomes."""
import argparse
import json
from pathlib import Path
import pandas as pd
import common as T


def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['ordinary_residual','ordinary_tokens']);args=ap.parse_args()
    parts=[];inputs=[]
    for marker in sorted((T.ARCHIVE/args.family).glob('*/complete.json')):
        info=json.loads(marker.read_text());assert info['status']=='complete'
        assert T.sha256(marker.parent/'arrays.npz')==info['arrays_sha256']
        parts.append(pd.read_csv(marker.parent/'rows.csv'))
        inputs.append(dict(path=str(marker),sha256=T.sha256(marker),checks=info['checks']))
    data=pd.concat(parts,ignore_index=True)
    keys=['task','start','layer','arm','mode','kind']+(['population'] if 'population' in data else [])
    groups=[]
    for values,g in data.groupby(keys):
        row=dict(zip(keys,values));assert len(set(g.seed))==len(g)
        row.update(seeds=len(g),r_mean=float(g.r_mean.mean()),r_down=int((g.r_mean<0).sum()),
                   mu_mean=float(g.mu_mean.mean()),mu_down=int((g.mu_mean<0).sum()))
        if g.ce_delta.notna().all():
            row.update(ce_delta_mean=float(g.ce_delta.mean()),ce_down=int((g.ce_delta<0).sum()))
        groups.append(row)
    out=T.ROOT/'results/vit_two_block_1025';summary=pd.DataFrame(groups)
    data.to_csv(out/f'{args.family}_cases.csv',index=False);summary.to_csv(out/f'{args.family}_groups.csv',index=False)
    T.atomic_json(out/f'{args.family}_inputs.json',dict(cases=len(parts),report_sha256=T.sha256(Path(__file__)),inputs=inputs))
    lines=[f'# {args.family}: 保存状態の追加診断','',f'完了 {len(parts)}/40 状態。数値はunit平均→seed平均。',
           '一歩の有限更新は勾配normを基準腕と揃え、入力とfc1以外を固定。固定計量はmomentumなし、Adamは保存履歴を維持。',
           '逆伝播を変える腕の一歩の結果を長期的な沈降除去や学習維持へ外挿しない。','',
           '| 課題 | 起点 | block | 母集団 | 腕 | optimizer | 窓 | seed数 | Δr | 沈降seed | Δμ | ΔCE |',
           '|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary[summary['kind']!='linear'].to_dict('records'):
        lines.append(f"| {row['task']} | {row['start']} | {row['layer']} | {row.get('population','cls')} | {row['arm']} | {row['mode']} | {row['kind']} | {row['seeds']} | {row['r_mean']:.8g} | {row['r_down']} | {row['mu_mean']:.8g} | {row.get('ce_delta_mean',float('nan')):.8g} |")
    (out/f'{args.family}_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(family=args.family,cases=len(parts),groups=len(summary))))


if __name__=='__main__':main()
