#!/usr/bin/env python3
"""Frozen J1-J4 aggregation for the L0 saved-state carrier ledger."""
import csv
import json
import subprocess

import measure as M
np,OUT=M.np,M.OUT
ENDPOINTS=[10,25,39,50,89,100]
DISPLAY_TERMS=['dr_total','dr_stretch','dr_rotate','dr_input','dr_cross','dr_width','dr_bias']


def read(name):
    rows=[]
    for r in csv.DictReader((OUT/f'{name}.csv').open()):
        row={}
        for k,v in r.items():
            if k in ['arm','probe','window','phase']:row[k]=v
            elif v in ['True','False']:row[k]=v=='True'
            else:row[k]=float(v) if v else None
        rows.append(row)
    return rows


def span(rows,key,n=4):
    values=[r[key] for r in rows]
    return f'{min(values):.{n}f}–{max(values):.{n}f}'


def state_map(rows):return {(r['arm'],int(r['seed']),int(r['task']),r['probe']):r for r in rows}


def judgment(states,windows):
    sm=state_map(states);seeds=[]
    for arm in M.A.ARMS:
        for seed in range(10):
            w=next(r for r in windows if r['arm']==arm and r['seed']==seed and r['probe']=='uniform' and r['window']=='cumulative39_100')
            candidates={'INPUT_MEAN_MOVES':w['cohort_dr_input_med'],'W_TERM':w['cohort_dr_w_med'],'BIAS_SINKS':w['cohort_dr_bias_med']}
            value=min(candidates.values());tied=[k for k,v in candidates.items() if abs(v-value)<=1e-10]
            winner=tied[0] if value<0 and len(tied)==1 else 'UNASSIGNED'
            subtype='rotation' if w['cohort_dr_rotate_med']<w['cohort_dr_stretch_med'] else 'stretch'
            a,b=sm[arm,seed,25,'uniform'],sm[arm,seed,100,'uniform']
            ratio=b['m_norm']/a['m_norm'];dc=b['cohort_c_med']-a['cohort_c_med'];sr=b['cohort_sigma_hat_med']/a['cohort_sigma_hat_med']
            seeds.append(dict(arm=arm,seed=seed,J1=winner,W_subtype=subtype,J2_INPUT_CONCENTRATES=ratio>=2,J2_ALIGNMENT_TURNS=dc<=-.2,
                J2_SPREAD_SHRINKS=sr<.7,m_norm_ratio_25_100=ratio,c_median_change_25_100=dc,sigma_hat_ratio_25_100=sr,
                w_norm_ratio_25_100=b['cohort_w_norm_med']/a['cohort_w_norm_med'],
                J3_flat=w['cohort_abs_cumulative_dr_med']<=.3,
                abs_cumulative_dr_med=w['cohort_abs_cumulative_dr_med'],abs_median_cumulative_dr=abs(w['cohort_dr_total_med']),
                total_variation_med=w['cohort_total_variation_med'],
                **{key:w[f'cohort_{key}_med'] for key in DISPLAY_TERMS+['dr_w']}))
    main=[r for r in seeds if r['arm']=='C3'];cap=[r for r in seeds if r['arm']=='ReLUcap']
    counts={name:sum(r['J1']==name for r in main) for name in ['INPUT_MEAN_MOVES','W_TERM','BIAS_SINKS','UNASSIGNED']}
    labels=[name for name,n in counts.items() if n>=7 and name!='UNASSIGNED'];label=labels[0] if labels else 'MIXED'
    sub={name:sum(r['W_subtype']==name for r in main if r['J1']=='W_TERM') for name in ['stretch','rotation']}
    j2={arm:{name:dict(n=sum(r[f'J2_{name}'] for r in seeds if r['arm']==arm),seeds=[r['seed'] for r in seeds if r['arm']==arm and r[f'J2_{name}']])
        for name in ['INPUT_CONCENTRATES','ALIGNMENT_TURNS','SPREAD_SHRINKS']} for arm in M.A.ARMS}
    ncap=sum(r['J3_flat'] for r in cap)
    result=dict(J1=dict(label=label,counts=counts,W_subtypes=sub,prior_probability={'INPUT_MEAN_MOVES':.40,'BIAS_SINKS':.30,'W_TERM':.25,'MIXED':.05}[label]),
        J2=j2,J3=dict(label='CAP_FLAT' if ncap>=8 else 'NOT_CAP_FLAT',n=ncap,denominator=10,seeds=[r['seed'] for r in cap if r['J3_flat']],prior_probability=.8),
        J4=dict(exception_seeds=[4,8],comparison_seeds=[0,1,2,3,5,6,7,9],classification='DESCRIPTIVE_ONLY'),
        prediction_note='Original Fable probabilities retained. BIAS_SINKS is structurally excluded because both arms have no up bias. J2 has no preregistered aggregate-vote threshold.')
    M.dump(OUT/'judge.json',result);M.M.write_csv(OUT/'seed_judgments.csv',seeds)
    return result,seeds


def aggregate_tables(states,windows):
    trajectory=[];ranges=[]
    skeys=['m_norm','m_norm_over_sqrt_d','cov_pr','cohort_c_med','cohort_sigma_hat_med','cohort_w_norm_med','cohort_B_med','cohort_r_med','cohort_s_med','cohort_dead','n_cohort']
    for arm in M.A.ARMS:
        for probe in ['uniform','chain']:
            for task in ENDPOINTS:
                rr=[r for r in states if r['arm']==arm and r['probe']==probe and r['task']==task];assert len(rr)==10
                row=dict(arm=arm,probe=probe,task=task,n_seed=10)
                for key in skeys:row.update({f'{key}_min':min(r[key] for r in rr),f'{key}_max':max(r[key] for r in rr),f'{key}_seed_median':float(np.median([r[key] for r in rr]))})
                trajectory.append(row)
            for name in M.WINDOWS:
                rr=[r for r in windows if r['arm']==arm and r['probe']==probe and r['window']==name];assert len(rr)==10
                row=dict(arm=arm,probe=probe,window=name,n_seed=10,n_pairs=rr[0]['n_pairs'])
                for key in M.TERMS:
                    vals=[r[f'cohort_{key}_med'] for r in rr]
                    row.update({f'{key}_min':min(vals),f'{key}_max':max(vals),f'{key}_seed_median':float(np.median(vals))})
                ranges.append(row)
    M.M.write_csv(OUT/'trajectory_summary.csv',trajectory);M.M.write_csv(OUT/'window_summary.csv',ranges)


def summary(states,windows,phases,result,seeds):
    checks=json.loads((OUT/'checks.json').read_text());gate=json.loads((OUT/'gate.json').read_text());pv=json.loads((OUT/'provenance.json').read_text())
    lines=['# L0の沈みの担い手：保存状態の結果','',
        '親: 文字GPTで継続学習_担い手の交代を保存状態で確かめる_結果_1023。事後の保存状態解析、量・集約・判定・予測は測定前にspec commit 2b94bc10で固定。新規学習なし。','',
        '| 判定 | 結果 |', '|---|---|',
        f"| J1 C3 39→100 | {result['J1']['label']}、票数 {result['J1']['counts']}、W内訳 {result['J1']['W_subtypes']} |",
        f"| J2 C3 25→100 | "+'、'.join(f"{k} {v['n']}/10" for k,v in result['J2']['C3'].items())+' |',
        f"| J3 ReLUcap 39→100 | {result['J3']['label']}、{result['J3']['n']}/10 |",
        '| J4 seed4・8 | 下の同じ帳簿・状態比で他8seedと比較（記述のみ） |','',
        '主は固定uniform probe。課題10末にuniformで一度でもz>0のunitを固定し、後で死んでも外さない。chain記録も同じコホート。',
        '状態中央値は偶数個で中央2個平均。元CSVのsd/mu中央値（中央下側）はK1専用列。両腕ともup biasとLN betaが無く、b/sとΔbは常に0。','',
        '## 窓の帳簿（uniform、固定コホート）','',
        '各unitで窓内の項を加算→コホート中央値→10seed範囲。負が沈める向き。W項=伸び+回転（unit別に加算）。幅と交差も含めた和がunitごとのΔrになる。中央値の列同士を足してもΔr中央値には一般に一致しない。',
        '10→25、25→39、50→89は保存端点間の長い橋であり、1課題分ではない。','',
        '| arm | 窓（対数） | Δr | 伸び/s_new | 回転/s_new | w_old·Δm/s_new | 交差/s_new | 幅 | bias/s_new |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in M.A.ARMS:
        for name in M.WINDOWS:
            rr=[r for r in windows if r['arm']==arm and r['probe']=='uniform' and r['window']==name]
            lines.append(f"| {arm} | {name} ({int(rr[0]['n_pairs'])}) | "+' | '.join(span(rr,f'cohort_{k}_med') for k in DISPLAY_TERMS)+' |')
    lines+=['','dmu（絶対位置）の各項、全512unit、各対の値、対ごとの中央値を窓内で中央値にした補助統計はledger.csv/windows.csv。chainも同CSVに記録。','',
        '## 状態推移（uniform、固定コホート）','',
        '| arm | task | ‖m‖ | c中央値 | sigma_hat中央値 | ‖w‖中央値 | b/s中央値 | r中央値 | sd中央値 | 共分散PR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=['m_norm','cohort_c_med','cohort_sigma_hat_med','cohort_w_norm_med','cohort_B_med','cohort_r_med','cohort_s_med','cov_pr']
    for arm in M.A.ARMS:
        for task in ENDPOINTS:
            rr=[r for r in states if r['arm']==arm and r['probe']=='uniform' and r['task']==task]
            lines.append(f'| {arm} | {task} | '+' | '.join(span(rr,k) for k in keys)+' |')
    lines+=['','mはFFN前LNのgamma込み入力の平均。||m||/sqrt(128)もstates.csv/trajectory_summary.csv。PRは中心化token共分散のtr(C)^2/tr(C²)。','',
        '## seed別の主累積・J2・J3','',
        '| arm | seed | J1 | W内訳 | Δr中央値 | median(abs累積Δr) | ‖m100‖/‖m25‖ | c100−c25 | sigma_hat100/25 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for r in seeds:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['J1']} | {r['W_subtype']} | {r['dr_total']:.4f} | {r['abs_cumulative_dr_med']:.4f} | {r['m_norm_ratio_25_100']:.4f} | {r['c_median_change_25_100']:.4f} | {r['sigma_hat_ratio_25_100']:.4f} |")
    lines+=['','J3は累積してから絶対値を取り、unit中央値を取った値を主判定に使用。|median(累積Δr)|とmedian(Σ|Δr|)もseed_judgments.csvに別記。後者は軌跡の総変動であり主判定ではない。',
        'J2は各seedで閾値を適用し、全体の合否閾値は後付けしない。','',
        '## J4：C3 seed4・8と他8seed','',
        '| seed群 | Δr | 伸び | 回転 | 入力平均 | 交差 | 幅 | m比25→100 | c差25→100 | sigma_hat比25→100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group,rr in [('seed4',[r for r in seeds if r['arm']=='C3' and r['seed']==4]),('seed8',[r for r in seeds if r['arm']=='C3' and r['seed']==8]),('他8seed',[r for r in seeds if r['arm']=='C3' and r['seed'] not in [4,8]])]:
        ks=DISPLAY_TERMS[:-1]+['m_norm_ratio_25_100','c_median_change_25_100','sigma_hat_ratio_25_100']
        lines.append(f'| {group} | '+' | '.join(span(rr,k) for k in ks)+' |')
    lines+=['','## 更新100の頭と胴（補助、uniform）','',
        '各seedの課題90–100の11課題について、頭は前課題末→更新100、胴は更新100→当課題末。固定課題10コホート。表は各unitで11区間を累積してからコホート中央値を取ったseed範囲。頭は100更新×11、胴は9900更新×11なので、率として比較しない。','',
        '| arm | 区間 | Δr | 伸び | 回転 | 入力平均 | 交差 | 幅 | bias |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in M.A.ARMS:
        for phase in ['head','body']:
            rr=[r for r in phases if r['arm']==arm and r['phase']==phase]
            lines.append(f'| {arm} | {phase} | '+' | '.join(span(rr,f'cohort_{k}_med') for k in DISPLAY_TERMS)+' |')
    lines+=['','## K1と恒等式','',
        '| arm | seed | task | r（再現値/指定値） | sd（再現値/指定値） | dead（再現値/指定値） |',
        '|---|---:|---:|---:|---:|---:|']
    known={}
    for r in gate['rounded_K1']:known.setdefault((r['arm'],r['seed'],r['task']),{})[r['metric']]=r
    for (arm,seed,task),r in known.items():
        lines.append(f'| {arm} | {seed} | {task} | '+' | '.join(f"{r[k]['actual']:.6f} / {r[k]['expected']:.4f}" for k in ['r_med','sd_med','dead'])+' |')
    lines+=['',f"- gate: 5状態×2probeを元CSVと照合して通過。全計測では参照照合{checks['n_reference_checks']}件通過。",
        f"- mean/sd/r/forwardの最大絶対誤差 {checks['max_identity_abs']:.3e}。帳簿恒等式 {checks['max_ledger_abs']:.3e}、累積と端点差 {checks['max_telescope_abs']:.3e}。",
        f"- 純粋な行スケール変更の伸び・幅相殺誤差 {gate['scale_check']['stretch_width_cancel_abs']:.3e}。頭+胴のΔrも課題末差に一致。",
        f"- 保存モデル700状態、2probeで{pv['state_probe_rows']}行、追加u100 {pv['u100_states']}状態。CPU float64/8threads、optimizer作成なし、学習更新0。",'',
        '## 予測の採点','',
        f"- J1: {result['J1']['label']}。元の割当確率 {result['J1']['prior_probability']:.2f}。最多予測INPUT_MEAN_MOVES(.40)の採否は上記ラベルによる。BIAS_SINKS(.30)はbias不在のため構造上不可、確率は再配分していない。",
        '- J2: '+', '.join(f"{name}（元確率{prob:.2f}）{result['J2']['C3'][name]['n']}/10" for name,prob in [('INPUT_CONCENTRATES',.5),('ALIGNMENT_TURNS',.4),('SPREAD_SHRINKS',.1)])+'。',
        f"- J3: CAP_FLAT(.80)は{'支持' if result['J3']['label']=='CAP_FLAT' else '不支持'}、{result['J3']['n']}/10。",'',
        '## 解釈の限界・再現','',
        'これは固定probe上の端点の厳密な帳簿。中間checkpointを挿入すると交差項などの配分が変わり得る。主判定は指定された保存端点の鎖に固定しており、頭/胴を入れた鎖へ変更していない。',
        'b=0ではr=||m||c/sigma_hatで行ノルムが消える。伸び項は幅項と合わせて読む。大きい負の伸び項だけで「行ノルム増加がrを沈めた」とは言えない。',
        '各成分中央値は加法的でない。unitごとに成り立つ恒等式と、各中央値の記述を区別する。chainには課題分布の変更も含まれる。',
        f"測定実行commit `{pv['git_hash']}`。spec登録`2b94bc10`。source/probeはload10.pyとloadsignのmodel loaderを再利用。",
        '結果ノート: 可塑性喪失/測定/文字GPTで継続学習_L0の沈みの担い手を保存状態で分ける_結果_1023.md。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    (OUT/'judge.md').write_text('\n'.join(lines[4:10])+'\n\n詳細はsummary.mdとjudge.json。\n')
    M.dump(OUT/'report_provenance.json',dict(code_sha256=M.sha(__file__),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip(),
        inputs={n:M.sha(OUT/f'{n}.csv') for n in ['states','windows','phase_windows']}))


def main():
    states,windows,phases=read('states'),read('windows'),read('phase_windows')
    assert len(states)==1400 and len(windows)==280 and len(phases)==40
    result,seeds=judgment(states,windows);aggregate_tables(states,windows);summary(states,windows,phases,result,seeds)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
