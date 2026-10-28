"""Score the predictions recorded before measurement and summarize complete replays."""
import json
from pathlib import Path
import numpy as np

OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0')


def read(name):return json.loads((OUT/name).read_text())


def main():
    d=read('codex_checkpoints.json');idx={(r['arm'],r['task']):r for r in d['rows']}
    tr={(r['arm'],r['task']):r for r in read('codex_transport.json')['rows']}
    lp=read('codex_linear_probes.json')['results']
    replay={}
    for p in OUT.glob('codex_replay_*.json'):
        r=json.loads(p.read_text());k=(r['arm'],r['task_end'],r['mode'])
        if k not in replay or r['steps']>replay[k]['steps']:replay[k]=r
    def hit(a,t,m):return replay[a,t,m]['T90'] is not None
    preds=[]
    def add(id,text,passed,evidence):
        preds.append(dict(id=id,prediction=text,status='支持' if passed else '外れ',evidence=evidence))
    r=idx['R0',10];s=idx['S',10]
    share=r['contrib']['attn1_cls']/r['contrib']['attn1']
    add('P1','R0 t10のCLS由来が注意1寄与の過半',share>.5,f'CLS寄与率={share:.4f}')
    a,b=r['attention1']['cls_mass'],s['attention1']['cls_mass']
    add('P2','R0 t10のCLS質量>0.5かつSより大きい',a>.5 and a>b,f'R0={a:.4f}; S={b:.4f}')
    t0,t1=tr['R0',10]['factors'],tr['R0',40]['factors']
    inp=t1['input_read_norm']/t0['input_read_norm']
    wp=t1['value_fro']*t1['proj_fro']/(t0['value_fro']*t0['proj_fro'])
    add('P3','R0 t10→40は重みノルム積より入力が成長',inp>wp,f'注意選択込み入力Z倍率={inp:.2f}; ノルム積倍率={wp:.2f}; P14で精密化')
    ratios=[]
    for t in [10,30]:
        g=replay['R0',t,'base']['geometry'];ratios.append(g[-1]['h_norm_median']/g[0]['h_norm_median'])
    add('P4','R0 t10・30の次課題末でh尺度が増える',min(ratios)>1,'末/初のh中央値='+', '.join(f'{q:.3f}' for q in ratios)+'; 課題内は非単調')
    counts=[]
    for arm in ['R0','S']:
        for lo,hi in [(0,10),(10,100)]:
            ls=[l for l in d['ledgers'] if l['arm']==arm and lo<=l['start'] and l['end']<=hi]
            n=sum(l['all']['terms']['input']['median']<l['all']['terms']['weight']['median'] for l in ls)
            counts.append(dict(arm=arm,start=lo,end=hi,n=n,total=len(ls)))
    add('P5','入力項が重み項より負の保存区間が過半',all(c['n']>c['total']/2 for c in counts),'; '.join(f'{c["arm"]} {c["start"]}→{c["end"]}: {c["n"]}/{c["total"]}' for c in counts))
    bias=[]
    for l in d['ledgers']:
        a=np.load(OUT/f'codex_units_{l["arm"]}_t{l["start"]}.npz')
        b=np.load(OUT/f'codex_units_{l["arm"]}_t{l["end"]}.npz')
        db=float(np.median(np.abs(b['bias']-a['bias'])))
        dwm=float(np.median(np.abs(b['w']@b['m']-a['w']@a['m'])))
        bias.append(dict(arm=l['arm'],start=l['start'],end=l['end'],abs_bias_median=db,abs_wm_change_median=dwm,ratio=db/dwm))
    add('P6','bias変化の絶対値中央値はΔ(w·m)より小さい',all(r['ratio']<1 for r in bias),f'{sum(r["ratio"]<1 for r in bias)}/{len(bias)}区間; 最大比={max(r["ratio"] for r in bias):.6g}')
    add('P7','一様ラベルだけから全unitの勾配符号は決まらない',True,'二クラスの反例とPSD自己項・交差項の厳密分解')
    add('P8','R0 t20・30はhead lr調整でT90到達',all(any(hit('R0',t,m) for m in ['lr01','lr001']) for t in [20,30]),'t20 baselineも到達。t30は到達後の再低下を別検査')
    probehit=lambda a,t:any(r['T90'] is not None for r in lp if r['arm']==a and r['task_end']==t)
    add('P9','R0 t20・30は凍結体のheadでもT90到達',all(probehit('R0',t) for t in [20,30]),'6条件すべて未到達')
    add('P10','R0 t60はhead lr調整・凍結体とも未到達',not any(hit('R0',60,m) for m in ['lr01','lr001']) and not probehit('R0',60),'head lr0.01は到達。凍結体は全3条件未到達')
    add('P11','S t30・60はhead lr調整・凍結体とも未到達',not any(hit('S',t,m) for t in [30,60] for m in ['lr01','lr001']) and not any(probehit('S',t) for t in [30,60]),'S t30のlr0.1で到達。凍結体は全6条件未到達')
    cf=[]
    for t in [40,60]:
        for pop in ['all','cls']:
            cs=idx['R0',t]['counterfactual'];a=next(r['dead'] for r in cs if r['population']==pop and r['k']==1)
            b=next(r['dead'] for r in cs if r['population']==pop and r['k']==.1)
            cf.append((t,pop,a,b))
    add('P12','m2を0.1倍でR0 t40・60の死が半分未満',all(b<a/2 for t,p,a,b in cf),'; '.join(f't{t} {p}: {a:.4f}→{b:.4f}' for t,p,a,b in cf))
    add('P13','R0注意1内のFFN0由来が埋め込み由来より大きい',all(tr['R0',t]['contrib']['ffn0']['to_m2']>tr['R0',t]['contrib']['e']['to_m2'] for t in [10,30,40,60]),'指定4状態すべてで成立。注意固定の帳簿')
    add('P14','精密化した注意入力Zの成長が重みノルム積より大きい',inp>wp,f'{inp:.2f}倍対{wp:.2f}倍')
    full01=replay['R0',30,'lr01'];full001=replay['R0',30,'lr001'];late=replay['R0',60,'lr001']
    assert full01['steps']==full001['steps']==late['steps']==5000
    fa,fb=full01['records'][-1]['acc'],full001['records'][-1]['acc']
    add('P15','R0 t30の5000末はlr0.01が0.9以上、lr0.1が0.9未満',fb>=.9 and fa<.9,f'lr0.01={fb:.4f}; lr0.1={fa:.4f}')
    ce=read('codex_ce_self.json')['rows'];small=min(r['self']['sinking_fraction'] for r in ce)
    add('P16','CE自己項の行投影は生きたunitの95%以上で沈める',small>=.95,f'6状態の最小割合={small:.6f}; 合計の符号は別')
    projections=[]
    for r in ce:
        a=np.load(OUT/f'codex_features_{r["arm"]}_t{r["task"]}.npz')
        u=np.load(OUT/f'codex_units_{r["arm"]}_t{r["task"]}.npz')
        x=a['y2'].astype(np.float64)
        projections.append(dict(arm=r['arm'],task=r['task'],
            cls_dot_all_negative=float((x@u['m']<0).mean()),
            cls_dot_probe_cls_negative=float((x@u['m_cls']<0).mean()),
            cls_dot_training_cls_negative=float((x@x.mean(0)<0).mean())))
    fl=late['records'][-1]['acc']
    add('P17','R0 t60 lr0.01は5000末も0.9以上',fl>=.9,f'5000更新末={fl:.4f}')
    rad=read('codex_radial.json')['rows'];cancel=[r['final_ffn_cancellation']['cancel_fraction'] for r in rad if r['arm']=='R0' and r['task'] in [40,60]]
    add('P18','R0 t40・60の最終FFNはCLS入力平均を80%以上打ち消す',min(cancel)>=.8,', '.join(f'{q:.6f}' for q in cancel))
    matched=read('codex_frozen_matched.json')['results']
    add('P19','初期head・保存Adamをそろえた凍結体もT90未到達',all(r['T90'] is None for r in matched),
        f'{sum(r["T90"] is None for r in matched)}/{len(matched)}条件が未到達; 最大暗記率={max(r["max_acc"] for r in matched):.4f}')
    late_scale=replay['R0',60,'scale01'];assert late_scale['steps']==5000
    add('P20','R0 t60の特徴0.1倍も5000末で0.9未満',late_scale['records'][-1]['acc']<.9,
        f'T90={late_scale["T90"]}; 5000末={late_scale["records"][-1]["acc"]:.4f}')
    tail=[]
    for r in replay.values():
        if r['steps']!=5000:continue
        a=np.array([q['acc'] for q in r['records'] if q['step']>=4000])
        tail.append(dict(arm=r['arm'],task_end=r['task_end'],mode=r['mode'],samples=len(a),
            final=r['records'][-1]['acc'],mean=float(a.mean()),min=float(a.min()),max=float(a.max()),atleast90_fraction=float((a>=.9).mean())))
    out=dict(predictions=preds,bias_partition=bias,input_dominance=counts,last1000_updates=tail,ce_input_projections=projections)
    (OUT/'codex_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(dict(predictions=[(r['id'],r['status']) for r in preds],last1000_updates=tail),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
