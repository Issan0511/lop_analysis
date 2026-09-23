"""Check the corrected paper against committed data, without running experiments."""
from pathlib import Path
import re, json, hashlib, subprocess, sys
import pandas as pd
import numpy as np
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v11_corrections_0923'
text=(OUT/'V11_corrected_0923.md').read_text()
body=text.split('## 編集・照合記録')[0]
sources={}
def source(rel):
    p=ROOT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def csv(rel): return pd.read_csv(source(rel))
def verdict(run):return json.loads(source(f'results/{run}/verdict.json').read_text())
def close(a,b,tol):assert abs(a-b)<=tol,(a,b,tol)
def section(n):return body.split(f'### 表 {n}　',1)[1].split('\n## ',1)[0]
def rows(n):
    return [[c.strip().replace('**','') for c in re.split(r'(?<!\\)\|',l)[1:-1]] for l in section(n).splitlines() if l.startswith('|') and not l.startswith('|---')]
def window(d):return d[d.task.between(31,50)].groupby('seed').online_acc.mean()

assert re.findall(r'<!-- figure:(\w+) -->',body)==['1','2','3','4','5','6','7','8','9','10','A','B','C','D','E','F','J']
assert len(re.findall(r'^### 表 [1-5]　',body,re.M))==5
assert len(re.findall(r'^## 付録 [A-J][ 　]',body,re.M))==10
assert len(re.findall(r'^\[\d+\] ',body,re.M))==13
for bad in ('未作図','要確認（','結果 要確認','裾は使われていない','CH − CHB 中央値'):
    assert bad not in body,bad
width=None
for i,line in enumerate(body.splitlines(),1):
    if line.startswith('|'):
        n=len(re.split(r'(?<!\\)\|',line))-2
        if width is None:width=n
        assert width==n,(i,width,n)
    else:width=None
for image in re.findall(r'!\[[^\]]+\]\(([^)]+)\)',body):
    assert (ROOT/'results/v11_figures_0921'/Path(image).name).is_file()

alias={'ReLU':'R','SiLU':'SILU','leaky .3':'LK03'}
t3=rows(3)[1:14]
assert len(t3)==13
for row in t3:
    arm=alias.get(row[0].split('（')[0],row[0].split('（')[0])
    d=csv(f'results/rlcifar_mlp_battle_0918/{arm}/per_task.csv')
    for cond,col in [('raw',1),('std',2)]:
        w=window(d[d.cond==cond]); assert len(w)==10
        close(w.median(),float(row[col]),0.00005)
t4=rows(4)[1:17]; assert len(t4)==16
for row in t4:
    arm=alias.get(row[1].split('（')[0],row[1].split('（')[0])
    d=csv(f'results/cifar5p1_mlp_0920/{arm}_std_lr0.0001/per_task.csv')
    w=d[d.task.isin([21,23,25,27,29])].groupby('seed').online_acc.mean()
    assert len(w)==10
    close(w.median(),float(row[2]),0.00005)
doors=verdict('relu_doors_0919')['labels'];h=verdict('relu_doors_h_ref_0920')
assert (doors['D1']['n'],doors['D1']['pos'])==(6,1)
close(doors['D1']['p'],.21875,1e-12)
close(h['median_d_HC'],-.87369651041,1e-9)
assert 'C − ref 中央値' in body and 'H − CH の窓の対応差' in body

freeze={}
tail=csv('results/switch_push_cifar_0922/tail.csv')
for arm in ['LE_sw750','LE_md750']:
    sub=tail[(tail.task==5)&(tail.arm==arm)].set_index('seed').sort_index();assert len(sub)==10
    freeze[arm]=sub.mu2
assert int((freeze['LE_md750']>freeze['LE_sw750']).sum())==10
assert round(freeze['LE_sw750'].median())==141 and round(freeze['LE_md750'].median())==672

e=verdict('erosion_race_0919')['predictions']
miss=[k for k,v in e.items() if not k.startswith('_') and not v['hit']]
assert miss==['P1','P2','P5','P6']
hashes=['58c1819','ebe7231','d17530a','3755028','0c435d3','d6923d5']
for sha in hashes:subprocess.run(['git','merge-base','--is-ancestor',sha,'HEAD'],cwd=ROOT,check=True)
for run,files in [('swish_battle_0917',['verdict_mlp.json','verdict_cnn_A.json']),('edge_law_0905',['verdict.csv'])]:
    for f in files:source(f'results/{run}/{f}')
source('specs/spec_cifar5p1_mlp_0920.md')
prob=[.70,.30,.60,.50,.55,.45,.45,.55,.50,.75]
outcome=[True,True,False,False,False,False,False,True,False,True]
hits=[f'P{i}' for i,(p,y) in enumerate(zip(prob,outcome),1) if (p>=.5)==y]
assert hits==['P1','P6','P7','P8','P10']

# The audit identified these particular transcription and condition errors.
changes=json.loads((OUT/'corrections.json').read_text())
covered=sorted({a for c in changes for a in c['audit_ids']})
expected={f'{p}{i}' for p,n in [('T',6),('G',7),('A',8),('R',7)] for i in range(1,n+1)}-{'T3'}
assert set(covered)==expected
init=csv('results/initgeom_cifar_0920/report/display_rows.csv')
rmed=init[(init.layer==1)&(init.variant=='main')].groupby('condition').r.median()
assert rmed['C']<1e-14
close(rmed['std'],.125,.001); close(rmed['raw'],1.90,.005)
long=verdict('ch_chb_200_0919')
collapse=[v for v in long['seed_readouts']['CH'].values() if v['type']=='COLLAPSE']
assert len(collapse)==7
assert sum('BOTH' in x['reasons'] for x in collapse)==6
assert sum('DEPTH_ONLY' in x['reasons'] for x in collapse)==1
assert long['main']=='INCONCLUSIVE'
resp=csv('results/resp_cifar_ee_0920/paired.csv')
means=resp.groupby('metric').difference.mean()
close(means['P1'],.4674,.0001);close(means['P2'],.6786,.0001)
report=source('analysis/resp_cifar_ee_0920/report.py').read_text()
for m,a,b in [('P1','R1_10','N10'),('P2','N1r','S1_10r'),('restore_reset','R1_10r','N10r'),('sink_keep','N1','S1_10')]:
    assert f"('{m}','{a}','{b}'," in report
summary=source('results/rlcifar_mlp_battle_0918/summary.md').read_text()
assert '0.0139' in summary
pred=source('specs/spec_neffdir_ee_0918.md').read_text()
issa=pred.split('### 7.3 Issa')[1].split('## 8.')[0]
assert '| compensation | NO_COMPENSATION |' in issa and '| felu | FELU_HIGHER |' in issa
claude=pred.split('### 7.1 Claude')[1].split('### 7.2')[0]
assert 'COMPENSATES 85%' in claude and 'FELU_SAME 75%' in claude
for bad in ('出力 head を初期化','出力層を初期化すれば','どれも床のまま','全ユニットが正側に出て線形化','std と入力中心化 C で 0.121','床はどちらの層にも要る','ViT は喪失なし'):
    assert bad not in body,bad
# Captions embedded in the paper and delivered with the figures stay identical.
for num,file in [('2','fig02_doors'),('9','fig08_response'),('A','figA_mnist_chain'),('B','figB_5p1_budget'),('C','figC_initgeom_rest'),('E','fig02e_doors_controls'),('J','figJ_layer_chimera')]:
    block=body.split(f'<!-- figure:{num} -->')[1].split('<!-- /figure -->')[0]
    cap='\n'.join(l[2:] for l in block.splitlines() if l.startswith('> '))+'\n'
    assert cap==(ROOT/f'results/v11_figures_0921/{file}.txt').read_text(),num
before=(ROOT/'results/v11_claim_audit_0923/manuscript_snapshot.md').read_text()
assert re.findall(r'【決定 9[^】]*】',before)==re.findall(r'【決定 9[^】]*】',text)
assert '【B2・B8 保留・監査後の文案:' in text

pdf=PdfReader(ROOT/'output/pdf/V11_review_0923.pdf')
assert len(pdf.pages)>0
for i,p in enumerate(pdf.pages):assert len(p.extract_text())>200,(i+1,'empty or heading-only page')
pending=re.findall(r'【(?:B2・B8|決定 9)[^】]*】',body)
result={'date':'2026-09-23','new_experiments':0,'audit_items_reflected':covered,'edit_count':len(changes),'corrected_sha256':hashlib.sha256(text.encode()).hexdigest(),'initial_r_medians':rmed.to_dict(),'S3_collapse':{'both':6,'depth_only':1},'S4_mean_differences':means.to_dict(),'scope':'audit corrections, figure regeneration and checks against committed outputs; not independent peer review',
 'figures':17,'main_tables':5,'appendices':10,'references':13,'pdf_pages':len(pdf.pages),
 'table3_windows_checked':26,'table4_windows_checked':16,'table2_comparison_names_checked':['D1 = C-ref','S2 = H-CH'],
 'freeze_Q1':{'sw_median':float(freeze['LE_sw750'].median()),'md_median':float(freeze['LE_md750'].median()),'paired_direction':10},
 'GH_commits_verified':hashes,'erosion_misses':miss,
 '5p1_prediction_scoring':{'original_Claude': '3/10','posthoc_uniform_Claude':'5/10','posthoc_rule':'p >= .50 predicts true','hits':hits,'Issa':'2/2','P1_scope':'13 initial arms'},
 'pending_verbatim_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in pending],
 'source_sha256':sources,
 'visual_QA':'All 50 pages rendered and inspected in contact sheets; changed figures and selected pages checked at full size. No clipped table, missing image, or math rendering error found.'}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','pending_verbatim_sha256')},ensure_ascii=False,indent=2))
