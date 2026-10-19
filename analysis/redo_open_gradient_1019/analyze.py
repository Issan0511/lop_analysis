#!/usr/bin/env python3
"""Per-example backpropagation and mean parameter gradients on saved current-task probes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

os.environ['MPLCONFIGDIR']='/tmp/redo_open_gradient_matplotlib_1019'
os.environ['OPENBLAS_NUM_THREADS']='2'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/redo_open_gradient_1019'
RAW=Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019/raw')


def fields(h,W,b,V,c,y):
    z=h@W.T+b;a=np.maximum(z,0);L=a@V.T+c
    L=L-L.max(1,keepdims=True);P=np.exp(L);P/=P.sum(1,keepdims=True)
    e=P.copy();e[np.arange(len(y)),y]-=1
    delta=(e@V)*(z>0)
    gw=delta.T@h/len(h);gb=delta.mean(0);gv=e.T@a/len(h)
    return z,a,e,delta,gw,gb,gv


def check():
    import torch
    torch.set_num_threads(2)
    g=np.random.default_rng(1019)
    h=g.normal(size=(7,4));W=g.normal(size=(3,4));b=g.normal(size=3);V=g.normal(size=(5,3));c=g.normal(size=5);y=g.integers(5,size=7)
    _,_,_,_,gw,gb,gv=fields(h,W,b,V,c,y)
    wh=torch.tensor(W,requires_grad=True);bh=torch.tensor(b,requires_grad=True);vh=torch.tensor(V,requires_grad=True)
    logits=torch.relu(torch.tensor(h)@wh.T+bh)@vh.T+torch.tensor(c)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor(y))
    grads=torch.autograd.grad(loss,(wh,bh,vh))
    error=max(float(np.abs(q.detach().numpy()-ref).max()) for q,ref in zip(grads,(gw,gb,gv)))
    if error>1e-10:raise RuntimeError(error)
    return error


def write(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
    verification=check();individual=[];sources=[];actual=[]
    for method in ('redo','none'):
        for group in (0,10):
            path=RAW/f'{method}_s{group}-{group+9}'
            meta=json.loads((path/'trace/events.json').read_text());ev=meta['events']
            with np.load(path/'trace/units.npz') as f:u={k:f[k] for k in f.files}
            with np.load(path/'trace/deltas.npz') as f:dd={k:f[k] for k in f.files}
            for k in range(len(dd['task'])):
                if int(dd['task'][k]) not in (3,29):continue
                for r,seed in enumerate(meta['seeds']):
                    ma=dd['pre_p'][k,r]>=.95
                    if not ma.any():continue
                    tiny=np.abs(dd['bias_gradient'][k,r])<1e-5
                    actual.append(dict(method=method,seed=seed,task=int(dd['task'][k]),step=int(dd['step'][k]),
                                       high_open=int(ma.sum()),tiny_bias_gradient=int((ma&tiny).sum()),
                                       tiny_bias_gradient_but_position_change_ge1e4=int((ma&tiny&(np.abs(dd['own_w'][k,r]+dd['own_b'][k,r])>=1e-4)).sum()),
                                       high_open_and_position_change_lt1e5=int((ma&(np.abs(dd['own_w'][k,r]+dd['own_b'][k,r])<1e-5)).sum())))
            for j,e in enumerate(ev):
                if not ((e['task']==3 and e['event']=='after_step' and e['step'] in (10,50,780)) or
                        (e['event']=='task_end' and e['task'] in (10,20,29))):continue
                stage=f't3_s{e["step"]}' if e['task']==3 else f't{e["task"]}_end'
                src=path/'trace'/f'deep_{j:04d}.npz'
                sources.append(dict(path=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
                with np.load(src) as f:d={k:f[k] for k in f.files}
                for r,seed in enumerate(meta['seeds']):
                    h,W,b,V,c=[d[k][r].astype(float) for k in ('current_h1','W2','b2','W3','b3')]
                    z,a,err,delta,gw,gb,gv=fields(h,W,b,V,c,d['current_y'][r])
                    rms=np.sqrt((delta*delta).mean(0));er=float(np.sqrt((err*err).sum(1).mean()))
                    vc=V-V.mean(0,keepdims=True);vn=np.linalg.norm(vc,axis=0)
                    win=np.sqrt((gw*gw).sum(1)+gb*gb);vout=np.linalg.norm(gv,axis=0)
                    pkey='t3_l2_p' if e['task']==3 else 'current_l2_p'
                    for i in range(100):
                        individual.append(dict(method=method,stage=stage,seed=seed,unit=i,
                                               p_full_current=float(u[pkey][j,r,i]),p_gradient_probe=float((z[:,i]>0).mean()),
                                               p_global=float(u['g_l2_p'][j,r,i]),delta_rms=float(rms[i]),
                                               delta_absmean=float(np.abs(delta[:,i]).mean()),delta_mean=float(gb[i]),
                                               incoming_gradient_norm=float(win[i]),outgoing_gradient_norm=float(vout[i]),
                                               v_norm=float(vn[i]),error_rms=er,
                                               signal_ratio=float(rms[i]/(vn[i]*er)) if vn[i]*er>0 else None,
                                               activation_rms=float(np.sqrt((a[:,i]**2).mean()))))
    summary=[]
    for method,stage in dict.fromkeys((r['method'],r['stage']) for r in individual):
        for threshold in (.95,.99):
            rs=[r for r in individual if r['method']==method and r['stage']==stage and r['p_full_current']>=threshold]
            if not rs:continue
            summary.append(dict(method=method,stage=stage,opening_threshold=threshold,n=len(rs),
                                delta_rms_median=float(np.median([r['delta_rms'] for r in rs])),
                                delta_lt1e3=sum(r['delta_rms']<1e-3 for r in rs),delta_lt1e4=sum(r['delta_rms']<1e-4 for r in rs),
                                incoming_median=float(np.median([r['incoming_gradient_norm'] for r in rs])),
                                outgoing_median=float(np.median([r['outgoing_gradient_norm'] for r in rs])),
                                both_parameter_norms_lt1e3=sum(r['incoming_gradient_norm']<1e-3 and r['outgoing_gradient_norm']<1e-3 for r in rs),
                                seeds_with_delta_lt1e3=len({r['seed'] for r in rs if r['delta_rms']<1e-3})))
    candidates=[r for r in individual if r['method']=='redo' and r['stage'] in ('t3_s780','t29_end') and
                r['p_full_current']>=.99 and r['v_norm']>1e-3 and r['error_rms']>.1]
    examples=sorted(candidates,key=lambda r:r['delta_rms'])[:5]
    result=dict(tier='posthoc_saved_state_analysis_no_training',gradient_formula_check_max_error=verification,
                stages=summary,examples=examples,source_files=sources,
                definition='delta is the per-example CE derivative before dividing by probe size; local evaluation uses float64 from saved float32 features and weights')
    (OUT/'verdict.json').write_text(json.dumps(result,indent=2))
    write(OUT/'summary.csv',summary);write(OUT/'examples.csv',examples);write(OUT/'actual_minibatch.csv',actual)
    np.savez_compressed(OUT/'raw/individuals.npz',**{k:np.asarray([r[k] if r[k] is not None else np.nan for r in individual]) for k in individual[0]})
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for stage,label,color in (('t3_s780','Task 3 end','#1469a8'),('t29_end','Task 29 end','#bd602b')):
        rs=[r for r in individual if r['method']=='redo' and r['stage']==stage]
        ax[0].scatter([r['p_full_current'] for r in rs],[max(r['delta_rms'],1e-10) for r in rs],s=5,alpha=.3,color=color,label=label)
        rr=[r for r in rs if r['p_full_current']>=.99]
        ax[1].scatter([max(r['incoming_gradient_norm'],1e-10) for r in rr],[max(r['outgoing_gradient_norm'],1e-10) for r in rr],s=7,alpha=.4,color=color,label=label)
    ax[0].set(xlabel='Opening fraction on full current-task images',ylabel='RMS per-example preactivation gradient',yscale='log',ylim=(1e-7,10))
    ax[0].axhline(1e-3,color='k',ls=':',lw=1)
    ax[1].set(xlabel='Incoming weight + bias gradient norm',ylabel='Outgoing weight gradient norm',xscale='log',yscale='log',title='Only units with opening >= 0.99')
    for aa in ax:aa.legend(frameon=False)
    fig.savefig(OUT/'opening_and_gradient.png',dpi=180);fig.savefig(OUT/'opening_and_gradient.pdf');plt.close(fig)
    ex=examples[0]
    table='\n'.join(f'| {r["stage"]} | {r["n"]} | {r["delta_rms_median"]:.3g} | {r["delta_lt1e3"]} | {r["delta_lt1e4"]} | {r["both_parameter_norms_lt1e3"]} |'
                    for r in summary if r['method']=='redo' and r['opening_threshold']==.99)
    txt=f'''# よく開いているのに逆伝播信号が小さい個体（1019）

状態: **保存状態の事後解析・新しい学習なし**。現在課題の全画像でp>=.99の個体を抽出し、固定256画像でCEの逆伝播を計算した。ReLUの局所微分は開いた入力で1でも、δ=1[z>0] v^T(prob−y)は小さくなりうる。小さな平均だけでなくper-example δのRMSも測った。

| 保存点 | p>=.99の個体数 | δRMS中央値 | δRMS<1e−3 | δRMS<1e−4 | incoming/outgoing両norm<1e−3 |
|---|---:|---:|---:|---:|---:|
{table}

20seedの個体件数の記述で、独立標本の検定ではない。閾値は可塑性喪失や凍結の定義ではない。incomingはmean(δh)とmean(δ)の結合norm、outgoingはmean(a(prob−y))のnorm。δは各画像のlossからの信号で1/256をかける前、パラメータ勾配は平均lossのもの。保存されたfloat32値から局所評価を倍精度で行った。式はTorchの倍精度autogradと最大誤差{verification:.3g}で一致した。

## 具体例（事後に小δの個体を選択）

ReDo {ex['stage']}、seed {ex['seed']}、第2層unit {ex['unit']}:

- 全現在課題画像の開口率 {ex['p_full_current']:.6f}、勾配用256画像の開口率 {ex['p_gradient_probe']:.6f}。
- δRMS {ex['delta_rms']:.6g}、平均絶対値 {ex['delta_absmean']:.6g}、平均 {ex['delta_mean']:.6g}。
- incoming勾配norm {ex['incoming_gradient_norm']:.6g}、outgoing勾配norm {ex['outgoing_gradient_norm']:.6g}。
- class-centered出力重みnorm {ex['v_norm']:.6g}、クラス誤差RMS {ex['error_rms']:.6g}、信号比 δRMS/(norm(v)*errorRMS) {ex['signal_ratio']:.6g}。

この例は出力重みが厳密0のreset直後を選んだものではなく、モデル全体の誤差RMS>.1も条件にした。全体に誤差があっても、特定個体のv方向へ返る誤差は小さい場合がある。

## 何を意味するか

よく開いていることはReLUの門が開いていること。loss gradientの大きさは出力重みと現在の誤差方向にも依る。per-example δが小さいことと、入力間で打ち消して平均だけが小さいことを区別する。またincomingが小さくてもoutgoingは学習している場合がある。

保存された実ミニバッチのbias勾配と本人W+biasによる位置の更新はactual_minibatch.csvへ別に残した。小勾配からAdamの小更新や恒久的な凍結は結論しない。今回測った局所勾配の小ささだけでは、次課題に対する可塑性喪失とも言えない。

数値の正本: verdict.json / summary.csv / examples.csv。具体例は高開口の候補からδRMSが小さい順に選んだもので、代表値ではない。
'''
    (OUT/'summary.md').write_text(txt)
    print(table);print(json.dumps(examples,indent=2))


if __name__=='__main__':main()
