"""Build the completed Japanese report and figures from immutable saved outputs."""
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation')
RESULTS=REPO/'results/vit_long_1024_2blk_rlfix'

def fmt(x,n=4):
    if x is None:return '—'
    if isinstance(x,str):return x
    return f'{x:.{n}f}'

def table(headers,rows):
    return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'

def label(d):
    return f"{d['arm']}{d['t']} {'共有' if d['head']=='shared' else 'head0'}"+(' score/4' if d['temperature']==4 else '')

def export_csv(path,rows):
    keys=list(rows[0])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def figures(s,cpu,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={'SL':'#b43c43','R0L':'#286b9d'}
    fig,ax=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    cp=s['checkpoints']
    for arm in ('SL','R0L'):
        ts=sorted(int(t) for t in cp['states'][arm] if int(t)>=10)
        ax[0,0].plot(ts,[cp['states'][arm][str(t)]['a_fixed'] for t in ts],'o-',color=colors[arm],label=arm)
    ax[0,0].set(xlabel='Task end',ylabel='F on fixed task-10 tokens',title='A. Fixed-reference scale');ax[0,0].legend()
    keys=['P','R','G','T','I','J'];vals=[cp['log_difference'][k] for k in keys]
    ax[0,1].bar(keys,vals,color=['#286b9d' if x>=0 else '#b43c43' for x in vals]);ax[0,1].axhline(0,color='black',lw=.6)
    ax[0,1].set(ylabel='SL minus R0L log growth',title='B. Exact factor accounting for S')
    for row,arm in enumerate(('SL','R0L')):
        z=ax[1,row];p=s['fits'][arm]['a_fixed']['early']
        z.plot(p['tasks'],p['actual'],'ko-',label='Measured')
        z.plot(p['tasks'],p['predicted'],'--',label='Power fit')
        m=s['fits'][arm]['factor_model'];z.plot(m['tasks'],m['predicted'],':',lw=2,label='Factor time model')
        for kind,ls in [('constant_energy','-.'),('affine_energy','--')]:
            d=a['forecasts'][arm][kind];z.plot([r['t'] for r in d['rows']],[r['predicted'] for r in d['rows']],ls,label=kind.replace('_',' '))
        z.axvspan(20,50,color='gray',alpha=.12);z.axvline(50,color='gray',lw=.6)
        z.set(xlabel='Task end',ylabel='F',title=f'{chr(67+row)}. {arm}: fit 20–50, forecast 60–100')
        z.legend(fontsize=7)
    fig.savefig(RESULTS/'codex_qkform_growth.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,7.7),layout='constrained')
    for arm,t,style in [('SL',10,'-'),('SL',60,'--'),('SL',100,':'),('R0L',10,'-'),('R0L',100,':')]:
        name=f'codex_replay_{arm}_t{t}_{"shared" if arm=="SL" else "zero"}_temp1'
        d=a['activity'][name]
        ax[0,0].plot([50,300,1000],[w['mean_c'] for w in d['windows']],style,marker='o',color=colors[arm],label=f'{arm}{t}')
    ax[0,0].set(xlabel='Window midpoint (update)',ylabel='Mean q/k c',title='A. Activity persists in late SL');ax[0,0].legend(fontsize=8)
    xx=[]
    for name,d in a['activity'].items():
        if name.endswith('temp4'):continue
        old=s['replays'][name];x=d['qk_path_length'];y=d['F_change_common'];xx.append(x)
        ax[0,1].scatter(x,y,color=colors[old['arm']])
        text=f"{old['arm']}{old['t']} {'S' if old['head']=='shared' else '0'}"
        offset=(-20,-17) if old['arm']=='R0L' and old['t']==100 else (5,7)
        align='right' if old['arm']=='R0L' and old['t']==100 else 'left'
        ax[0,1].annotate(text,(x,y),xytext=offset,textcoords='offset points',fontsize=7,ha=align)
    x=np.linspace(0,max(xx)*1.06,30);ax[0,1].plot(x,x*a['activity_calibration']['F_per_path'],'k--',lw=.9,label='Task-10 calibration')
    ax[0,1].axhline(0,color='gray',lw=.6);ax[0,1].set(xlabel='q/k path length = lr sqrt(n) sum(c)',ylabel='Delta F on task-10 tokens',title='B. Activity alone fails to predict growth');ax[0,1].legend(fontsize=7)
    keys=['common_task101_labels_q_energy','common_task101_labels_k_energy'];x=np.arange(2)
    for off,arm in [(-.17,'SL'),(.17,'R0L')]:
        ax[1,0].bar(x+off,[s['signals'][arm]['late'][k]['8'] for k in keys],width=.32,label=arm,color=colors[arm])
    ax[1,0].set(xticks=x,xticklabels=['q signal','k signal'],ylim=(0,1),ylabel='Mean cos² principal angle (k=8)',title='C. N8: SL persistence prediction fails');ax[1,0].legend()
    for i,temp in enumerate((1,4)):
        d=a['temperature'][str(temp)];bottom=0.
        for key,color,name in [('radial_common','#b43c43','Radial cross term'),('step_common','#286b9d','Squared step term')]:
            val=d[key];ax[1,1].bar(i,val,bottom=bottom if val>=0 else 0,color=color,label=name if i==0 else None)
            if val>=0:bottom+=val
        ax[1,1].plot(i,d['delta_F_common'],'kD',ms=5,label='Total delta F' if i==0 else None)
    ax[1,1].axhline(0,color='gray',lw=.6);ax[1,1].set(xticks=[0,1],xticklabels=['score','score / 4'],ylabel='Whole-task contribution to delta F',title='D. SL100 head0: fixed task-10 tokens');ax[1,1].legend(fontsize=7)
    fig.savefig(RESULTS/'codex_qkform_tests.png',dpi=180);plt.close(fig)

def main():
    s=json.loads((OUT/'codex_summary.json').read_text());cp=s['checkpoints']
    cpu=json.loads((OUT/'codex_finish_cpu.json').read_text())
    a=json.loads((OUT/'codex_finish_analysis.json').read_text())
    ver=json.loads((OUT/'codex_verification.json').read_text())
    now=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
    chunks=[f'''# block0 の q・k と注意の飽和の形成

{now}。完成版。
seed 10252010 の保存状態と、保存済みの一課題1500更新の回し直しを解析した。
今回の再開後は保存状態の差分をCPUで再集計した。

## 答えと限界

SLの尺度成長の腕差は、主にtokenへの保持と入力の変化に現れた。
課題内では、損失の残る間にAdamの歩が続いた。
その歩を尺度へ変換するには、時間的な整合と作用素の向きの変化も要った。
「当てはめの更新数×c」だけから固定token尺度の増分を決める模型は外れた。
共有headから上流信号の部分空間の持続へ至る予測N8も外れた。
後期のhead0が元のheadと同じ向きを読むという予測N9は逆だった。
共有headから100課題の飽和形成を閉じた自律式で導く段は残った。

| 問い | 今回の答え | 格 |
|---|---|---|
| 尺度の成長を何へ分けられるか | 重みノルム積・積の整合・γ・token保持・入力変化・biasの六因子 | 恒等式と測定 |
| 課題内のいつ歩くか | 当てはめの間。後期SLでは501〜1500更新にも大きな歩が残る | 測定 |
| 歩数×cだけで尺度が伸びるか | 同じ比例係数では符号も量も再現しない | 反証 |
| 軟化で歩幅が減るか | SL100 head0のdAとdaは減る | 局所介入 |
| SLのδq・δk空間がより持続するか | 主判定の上位8方向では逆 | N8の外れ |
| 後期の新headが元headに近づくか | 100更新後の主角は初期より離れた | N9の外れ |

## 方法と尺度の定義

画像は固定した32枚を用いた。
全65queryと6headを等しく重み付けたrmsを主尺度にした。
前回の逆伝播重み付き幾何平均とは集計が異なる。
部分空間の測定は固定256枚を用いた。
全回し直しで保存Adamを使った。
温度1の7本とscore/4の1本を比較した。
各回し直しは同じ次課題の最初の1500更新である。
本走の一課題5000更新の終点とは区別する。

ξ_tはblock0のLN1のaffine前の正規化tokenとする。
Q=W_q、K=W_k、Γ=diag(γ)と置く。
A_h=ΓK_hᵀQ_hΓ/√64と置く。
b_h=ΓK_hᵀ(Q_hβ+b_q,h)/√64と置く。
keyのbiasはqueryごとのscoreへ同じ定数を足すのでsoftmaxでは消える。

F_t=rms‖A_tξ_10‖を長期の固定token尺度とする。
S_t=rms‖A_tξ_t+b_t‖を現在の入力での実効尺度とする。
保存済み回し直しのa_fixedは、開始状態s自身のξ_sで固定した量だった。
これをF^(s)と記す。
再開後のCPU解析で、回し直しも共通のξ_10で測るFを加えた。
各表で両者を区別する。

## Q1：六因子の分解

B_h=K_hᵀQ_h/√64と置く。
P=‖Q‖_F‖K‖_F/√64と置く。
R=‖B‖_F/P、G=‖A‖_F/‖B‖_Fと置く。
T=rms‖Aξ_10‖/‖A‖_Fと置く。
I=rms‖Aξ_t‖/rms‖Aξ_10‖と置く。
J=rms‖Aξ_t+b‖/rms‖Aξ_t‖と置く。

**S=P·R·G·T·I·J、F=P·R·G·T。**

この積は厳密である。
各log変化は足し算になる。
Rは静的な積KᵀQの整合である。
更新ΔWの整合とは同一ではない。
因子の分け方は一意ではない。
この分解の比率を因果寄与率とは呼ばない。
''']
    rows=[]
    names={'P':'重みノルム積','R':'積KᵀQの整合','G':'γの実効作用','T':'固定tokenへの保持','I':'入力の変化','J':'bias等の加算'}
    for k in names:rows.append([k+'：'+names[k],fmt(cp['log_changes']['SL'][k]),fmt(cp['log_changes']['R0L'][k]),fmt(cp['log_difference'][k])])
    rows.append(['合計',fmt(sum(cp['log_changes']['SL'].values())),fmt(sum(cp['log_changes']['R0L'].values())),fmt(sum(cp['log_difference'].values()))])
    chunks.append(table(['課題10→100','SLのlog増加','R0Lのlog増加','SL−R0L'],rows))
    fraction=(cp['log_difference']['T']+cp['log_difference']['I'])/sum(cp['log_difference'].values())
    chunks.append(f'''保持Tと入力Iの合計は腕差の{fraction*100:.2f}%だった。
γの実効作用の差は腕差の{cp['log_difference']['G']/sum(cp['log_difference'].values())*100:.2f}%だった。
Rの成長率はR0Lの方が大きかった。
ΔAがSLで大きいことを、静的なRの成長率だけへ帰着させることはできない。

入力の静的入替では、patch埋め込みとpos＋CLSを二つの順序で入れ替えた。
両順序のlog変化を平均した。
これは順序に対称な代数分解である。
''')
    chunks.append(table(['腕','log I','patchのlog寄与','pos＋CLSのlog寄与'],[[arm,fmt(cp['log_changes'][arm]['I']),fmt(cp['input_mix'][arm]['log_patch']),fmt(cp['input_mix'][arm]['log_pos_cls'])] for arm in ('SL','R0L')]))
    chunks.append('入力側の変化はほぼpatch埋め込みに現れた。\n\n### 保存状態の軌跡\n\n初期の毎課題と、それ以降の保存された全時点を示す。\n')
    for arm in ('SL','R0L'):
        rows=[]
        for t in sorted(map(int,cp['states'][arm])):
            d=cp['states'][arm][str(t)]
            rows.append([t,fmt(d['Q_norm'],3),fmt(d['K_norm'],3),fmt(d['gamma_mean']),fmt(d['A_norm'],3),fmt(d['a_fixed'],3),fmt(d['S'],3),fmt(d['saturation'])])
        chunks.append(f'\n**{arm}。**\n'+table(['課題末','‖Q‖','‖K‖','γ平均','‖A‖','F：固定ξ10','S：現在ξ＋b','飽和率'],rows))
    chunks.append('''### 尺度対称性とγの符号

LNの出力をz=γ⊙ξ+βとする。
γ_jとβ_jをα_j倍し、qkvの第j列をα_jで割ればqkvの出力は変わらない。
従ってγ_j∂L/∂γ_j+β_j∂L/∂β_j=Σ_rW_rj∂L/∂W_rjが厳密に成り立つ。
和はvalueを含むqkv全体で取る。
これはγ単独の勾配の符号を固定しない。

Adamの正規化歩をu_γとすれば、Δ‖γ‖²=−2η〈γ,u_γ〉+η²‖u_γ‖²である。
従って縮小の条件は2〈γ,u_γ〉>η‖u_γ‖²である。
この条件の符号は実際の勾配とmomentを測って決める。
R0Lで縮む理由を尺度対称性だけから導いたとは主張しない。
''')
    chunks.append(table(['開始状態','Δγ平均：1500更新','γの半径項：二乗ノルム','γの歩の二乗項'],[[label(d),f"{sum(w['gamma_mean_change'] for w in d['windows']):.6g}",f"{sum(w['gamma_energy_radial'] for w in d['windows']):.6g}",f"{sum(w['gamma_energy_steps'] for w in d['windows']):.6g}"] for d in s['replays'].values() if d['temperature']==1 and (d['arm']=='R0L' or d['head']=='shared')]))
    chunks.append('''### q・kの積と更新の整合

生の逆伝播はδq_i=Σ_jδs_ij k_j/√64である。
生の重み勾配はΣ_iδq_i z_iᵀである。
座標別のAdamの割り算は、この生勾配の部分空間を一般には保たない。
勾配の集中だけからAdamの歩の階数は決まらない。
またδqの出力座標の空間とξの入力座標の空間は異なる。

γを固定した積の差には、KᵀΔQ、ΔKᵀQ、ΔKᵀΔQが全て入る。
γの変化は別の項になる。
今回の動的な整合の指標は‖KᵀΔQ‖/(‖K‖‖ΔQ‖)とした。
下見の‖KΔQᵀ‖/(‖K‖‖ΔQ‖)も生出力に残っている。
両者の行列の積の順序は異なる。

## Q2：成長は課題の中のいつ起きるか

c_qとc_kは、各更新のrms[mhat/(sqrt(vhat)+eps)]である。
下表のcは窓内での算術平均である。
損失は実際の学習batchの損失の平均である。
ΔFは共通ξ10での尺度増分である。
ΔF^(s)は開始状態のξ_sでの尺度増分である。
''')
    window_csv=[];rows=[]
    for name,d in s['replays'].items():
        common=cpu['replays'][name]['windows']
        for w,u in zip(d['windows'],common):
            rows.append([label(d),f"{w['lo']+1}–{w['hi']}",fmt(w['mean_loss'],5),fmt(w['q_c_mean']),fmt(w['k_c_mean']),fmt(u['delta_F'],5),fmt(w['delta_a_fixed'],5)])
            window_csv.append(dict(case=name,lo=w['lo'],hi=w['hi'],loss=w['mean_loss'],q_c=w['q_c_mean'],k_c=w['k_c_mean'],delta_F_common=u['delta_F'],delta_F_start=w['delta_a_fixed'],radial_common=u['radial_norm'],step_common=u['step_norm']))
    chunks.append(table(['回し直し','更新窓','損失','c_q','c_k','ΔF：ξ10','ΔF^(s)：開始ξ'],rows))
    chunks.append('''後期SLでは、501〜1500更新にも損失とcが残った。
R0Lでは同じ窓のcは小さかった。
SL10の共有headではSの増加の79.1%が101〜500更新に入った。
同じ状態のhead0よりSは大きく伸びた。
これは共有headが一課題内の形成を強める局所対照である。
全課題の形成を同じ因果経路で説明したことにはならない。

### 独立な更新の酔歩では再現しない

Vをq・kの重みを並べたベクトルとする。
n=2×384²と置く。
c_t=√[(c_q,t²+c_k,t²)/2]と置く。
u_tをAdamの正規化歩とすれば、‖ΔV‖=η√n(Σ_tc_t)ρ_timeが成り立つ。
ρ_time=‖Σ_tu_t‖/Σ_t‖u_t‖は時間方向の整合である。
これは0から1までの量である。
ρ_timeを既知としたときの恒等式であり、cだけの予測式ではない。

‖ΔV‖²=η²Σ_t‖u_t‖²+2η²Σ_{r<t}〈u_r,u_t〉である。
SL10のqueryでは、実変位2.095に対し、独立更新の見積もりは0.554だった。
二乗の比は14.27だった。
短い時間内の交差項を捨てる近似は外れた。
''')
    chunks.append(table(['温度1の7本','Σc','有効更新数','道のり η√nΣc','実変位 ‖ΔV‖','ρ_time','501以降のΣc割合','ΔF：ξ10'],[[label(s['replays'][name]),fmt(d['sum_c'],2),fmt(d['effective_steps'],1),fmt(d['qk_path_length'],3),fmt(d['qk_net_norm'],3),fmt(d['temporal_alignment']),fmt(d['tail_c_fraction'],3),fmt(d['F_change_common'],5)] for name,d in a['activity'].items() if s['replays'][name]['temperature']==1]))
    chunks.append('''有効更新数は(Σc)²/Σc²と定義した。
cが一定のN更新だけ残る場合にはNに一致する。
損失の閾値で当てはめ区間を選んだ量ではない。

SL100共有の道のりはR0L100の約3.2倍だった。
その時間整合はR0L100の約3分の1だった。
総変位の差は小さくなった。
さらに総変位からΔFへの変換には作用素とtokenに対する向きが入る。
SL60共有とSL100共有では、共通ξ10のΔFは負だった。
「長い尾があるからFが必ず増える」という比例則は成り立たなかった。

### 同じ係数の比例則の検査

事後的な検査として、SL10共有とR0L10の2本で原点を通る比例係数を較正した。
この較正は後述のt20〜50による時間外挿とは別である。
重み変位に対する係数ρ_timeと、ΔFに対する係数をそれぞれ一つに固定した。
''')
    chunks.append(table(['回し直し','重み変位の実測','同一ρによる予測','相対差','ΔF実測','道のり比例によるΔF予測'],[[label(s['replays'][name]),fmt(d['qk_net_norm'],3),fmt(d['predicted_net_norm'],3),fmt(d['net_relative_error']*100,1)+'%',fmt(d['F_change_common'],5),fmt(d['predicted_F_change_simple'],5)] for name,d in a['activity'].items() if s['replays'][name]['temperature']==1]))
    chunks.append('''同じ係数による比例則は後期のΔFの符号も再現しなかった。
多い歩が形成へどう加わるかは、次の離散式で半径項と歩の二乗項に分ける必要がある。

### 一課題の変位と10課題の保存差分

一課題のquery変位が同じ大きさで、課題間で直交し、1500更新後の寄与を無視できる場合を仮定する。
このとき10課題の変位の予測は√10‖ΔQ_1500‖になる。
この三条件は保存状態だけからは保証されない。
''')
    native=[d for d in s['replays'].values() if d['temperature']==1 and (d['arm']=='R0L' or d['head']=='shared')]
    chunks.append(table(['開始状態','‖ΔQ_1500‖','√10倍の見積もり','保存区間','保存‖ΔQ‖','相対差'],[[label(d),fmt(d['end']['walk']['q']['net_norm'],3),fmt(d['ten_task_q_projection'],3),f"{d['decade_comparison']['t0']}→{d['decade_comparison']['t1']}",fmt(d['decade_comparison']['dQ'],3),fmt((d['ten_task_q_projection']/d['decade_comparison']['dQ']-1)*100,1)+'%'] for d in native]))
    chunks.append('''この局所的な換算は重みの変位の桁を再現した。
課題100開始の2本は次の10課題の保存が無いため、90→100を比較対象にした。
この2本を未来予測の成功とは数えない。
ηc√(座標数×更新数)だけの式で同じ数が出たわけではない。

## Q3：飽和から形成への輪の検査

### 飽和率と次の区間の歩幅：N7

飽和率は32枚の全queryでmax softmax>0.99となる割合とした。
課題tの飽和率と、t→t+10の保存差分を対応させた。
''')
    chunks.append(table(['腕','飽和率 対 次の‖ΔA‖','飽和率 対 PR(ΔQ)','飽和率 対 積の整合','標本区間数'],[[arm,fmt(s['correlations'][arm]['dA']['spearman'],3),fmt(s['correlations'][arm]['Q_pr']['spearman'],3),fmt(s['correlations'][arm]['alignment_correct']['spearman'],3),9] for arm in ('SL','R0L')]))
    chunks.append(table(['SL：開始課題','飽和率','次の10課題の‖ΔA‖','PR(ΔQ)','‖KᵀΔQ‖/(‖K‖‖ΔQ‖)'],[[d['t0'],fmt(d['saturation']),fmt(d['dA'],3),fmt(d['Q_pr'],2),fmt(d['alignment'])] for d in a['lagged_rows']['SL']]))
    chunks.append('''N7の符号の予測は支持された。
ただし両腕で飽和率は課題番号と同じ順位に並んだ。
課題番号の順位を回帰すると説明変数の残差が消える。
従って課題番号と分離した順位相関は定義できなかった。
この相関だけでは飽和→次の大きな歩という因果の向きを決められない。

### 注意を軟らかくする反実仮想：N6

SL100から同じhead0・保存Adam・次課題・1500更新で比較した。
score/4は順伝播と勾配の両方を変える介入である。
表のAとdaは温度で割る前の重みから求めた。
''')
    rows=[]
    for temp in ('1','4'):
        d=a['temperature'][temp]
        rows.append(['基準' if temp=='1' else 'score/4',fmt(d['dA'],5),fmt(d['da_start'],5),fmt(d['da_common'],5),fmt(d['delta_F_start'],5),fmt(d['delta_F_common'],5)])
    chunks.append(table(['条件','‖ΔA‖','da：開始ξ100','da：共通ξ10','ΔF^(100)','ΔF：共通ξ10'],rows))
    chunks.append('''daはrms‖ΔAξ‖である。
尺度そのものの差ΔFとは異なる。
N6が予測した二つの変位は減った。
共通ξ10でのΔFも減った。
開始ξ100でのΔFはほぼ同じだった。
従って「変位の減少」と「全ての参照入力での尺度成長の減少」は区別する。

次は一課題全体の差分を一つの増分として分解した表である。
25更新ごとの分解を足した値とは配分が異なる。
''')
    rows=[]
    for ref in ('common','start'):
        for temp in ('1','4'):
            d=a['temperature'][temp]
            rows.append(['共通ξ10' if ref=='common' else '開始ξ100','基準' if temp=='1' else 'score/4',fmt(d[f'radial_{ref}'],6),fmt(d[f'step_{ref}'],6),fmt(d[f'parallel_{ref}'],6),fmt(d[f'perpendicular_{ref}'],6),fmt(d['delta_F_common' if ref=='common' else 'delta_F_start'],6)])
    chunks.append(table(['参照','条件','半径交差項','歩の二乗項','内訳：平行二乗','内訳：真の直交二乗','合計ΔF'],rows))
    chunks.append('''共通ξ10では、軟化した側の半径交差項は負になった。
開始ξ100では、軟化した側の正の半径交差項が二乗項の減少を補った。
この参照依存は今回の測定で見えた事実である。
温度介入は局所的な形成の一辺を支持した。
共有headから始まる長期の輪全体を証明したわけではない。

### 勾配の集中と歩の階数

token対の上位5%は画像ごとに選んだ。
その勾配二乗の和を全画像の勾配二乗の和で割った。
この指標はほぼ1に達した。
CLS queryなどへの集中も含むので、飽和による候補間の集中だけを取り出した量ではない。
''')
    chunks.append(table(['対照','101〜500更新の上位5%割合','501〜1500更新の上位5%割合','101〜500の生query勾配PR','1500更新のΔQのPR'],[[label(d),fmt(d['windows'][1]['score_top5_energy'],7),fmt(d['windows'][2]['score_top5_energy'],7),fmt(d['windows'][1]['gradient_Q_pr'],2),fmt(d['end']['delta_start']['Q_pr'],2)] for d in s['replays'].values() if d['arm']=='SL' and d['t'] in (10,100)]))
    chunks.append('''上位5%割合は天井に近く、共有とhead0の差を識別する指標として弱かった。
生勾配のPRと累積Adam変位のPRは異なる。
SL100の軟化では累積ΔQのPRが増えた。
これだけから「飽和が低階数の歩を作り、その歩が拡大する」という全媒介は同定できない。

## Q4：腕差の根をめぐる外れ

### 上流信号の部分空間：N8

δqとδkを全headで連結した384次元で主成分を取った。
画像とtokenをプールし、共分散を中心化した。
比較量は上位8方向の主角のcos²の平均である。
課題10→20から90→100までの9組を平均した。
元課題のラベルと、全保存状態で共通の課題101ラベルを分けた。
画像ごとの信号を単位ノルムにした診断も併記する。
''')
    rows=[]
    for mode,mname in [('trained_labels','元課題ラベル'),('common_task101_labels','共通ラベル')]:
        for kind in ('q','k'):
            for weight,wname in [('energy','勾配エネルギー'),('image_normalized','画像ごと正規化')]:
                key=f'{mode}_{kind}_{weight}'
                rows.append([mname,kind,wname,fmt(s['signals']['SL']['late'][key]['8']),fmt(s['signals']['R0L']['late'][key]['8'])])
    chunks.append(table(['残差の条件','信号','重み','SL','R0L'],rows))
    chunks.append('''主判定のエネルギー重みでは、q・kともSLがR0Lより小さかった。
N8は逆方向に外れた。
画像ごとの正規化でもSLが一貫して優位にはならなかった。
4方向と16方向の結果は生出力に保存した。
上位8方向という事前基準を別の次元へ差し替えない。
''')
    chunks.append(table(['q/k持続の順位相関','SL','R0L'],[[name,fmt(s['signal_correlations']['SL'][key],3),fmt(s['signal_correlations']['R0L'][key],3)] for key,name in [('persistence_to_retention_change','持続 対 保持Tのlog変化'),('persistence_to_alignment','持続 対 KᵀΔQの整合'),('persistence_to_A_step','持続 対 ‖ΔA‖')]]))
    chunks.append('''持続が保持Tと整合の差を運ぶという予測も数値から支持されなかった。
headの中心化行空間の隣接保存状態間のcos²は、SLが0.6968、R0Lが0.4856だった。
読み出し側の持続が、そのままδq・δkの持続へ伝わるわけではなかった。
逆伝播には状態と画像に依存するJacobianが挟まる。

δq=J_h,qᵀW_headᵀeという式は厳密である。
W_headの空間が持続しても、J_h,qとeが動けばδqの空間は変わり得る。
この間の段を物語で埋めない。

### head0から新しく育った読み出し：N9

lossに使う10行を平均中心化し、その9次元の行空間を比較した。
100更新後のhead0と元の共有headを比較した。
''')
    chunks.append(table(['保存状態','head0 100更新 対 元head','head0 100更新 対 共有100更新','共有100更新 対 元head'],[[f'SL{t}',fmt(d['zero100_to_original']['cos2']),fmt(d['zero100_to_shared100']['cos2']),fmt(d['shared100_to_original']['cos2'])] for t,d in s['head_comparisons'].items()]))
    chunks.append('''SL100の新headは、SL10の新headより元のheadから離れた。
N9は逆方向に外れた。
後期の形成対照の小さい差を「新headも同じ方向を読むラチェット」で説明する案は、この検査では採れない。
最終特徴が低参加比であるという前回の事実だけから、この説明は導けない。

## Q5：成長則、課題差分、外挿

### 離散の成長則

固定ξ10についてa=Aξ10と置く。
内積は画像・head・queryで平均する。
F²=〈a,a〉である。

**ΔF=[2〈a,Δa〉+‖Δa‖²]/(F_new+F_old)。**

この式は近似ではない。
第一項を半径交差項R_Fと呼ぶ。
第二項を歩の二乗項D_Fと呼ぶ。
依頼でいう直交の足し算に対応する項はD_Fである。
ただしD_Fには平行成分の二乗も含まれる。

r=〈a,Δa〉/F、Δa_perp=Δa−(r/F)aと置く。
‖Δa‖²=r²+‖Δa_perp‖²である。
真の直交成分は後者である。
表では半径交差項と歩の二乗項を主に示し、CSVには平行二乗と真の直交二乗も保存した。
25更新ごとの一次微分だけを足す近似は、特にSL10の伸びを過小評価した。
二乗項を捨てない離散式は測定と一致した。

### 保存された全区間の分解

課題0〜10は1課題ごとの測定である。
以後は10課題ごとの保存差分である。
例えば10→20の増分を10で割った値は区間平均であり、課題11の増分ではない。
保存の無い課題の個別値は復元できない。
区間を細分すると交差項と二乗項の配分は変わる。
10課題差分の二乗を10で割って、一課題の二乗変位と同一視しない。
''')
    increment_csv=[]
    for arm in ('SL','R0L'):
        rows=[]
        for d in cpu['checkpoints'][arm]:
            rows.append([f"{d['t0']}→{d['t1']}",d['tasks'],fmt(d['F0'],4),fmt(d['F1'],4),fmt(d['radial_norm'],5),fmt(d['step_norm'],5),fmt(d['delta_F'],5),fmt(d['delta_F_per_task'],5)])
            increment_csv.append(dict(arm=arm,**d))
        chunks.append(f'\n**{arm}：共通ξ10。**\n'+table(['保存区間','課題数','F_old','F_new','半径交差項R_F','歩の二乗項D_F','ΔF','ΔF/課題数'],rows))
    chunks.append('''課題10→100の10課題刻みの二乗尺度の集計も示す。
この集計の半径項は区間間で正負が相殺される。
''')
    chunks.append(table(['腕','Σ 半径交差エネルギー','Σ 歩の二乗エネルギー','合計ΔF²'],[[arm,fmt(s['energy'][arm]['a_energy_radial'],4),fmt(s['energy'][arm]['a_energy_step'],4),fmt(s['energy'][arm]['a_energy_radial']+s['energy'][arm]['a_energy_step'],4)] for arm in ('SL','R0L')]))
    chunks.append('''SLの伸びは正の半径項だけではない。
歩の二乗項も大きく寄与した。
R0Lでは半径項が負で、二乗項の一部を打ち消した。

### 回し直し7本との照合

全て共通ξ10で、一課題の最初の1500更新を一つの増分として分解した。
''')
    chunks.append(table(['温度1の7本','rms‖ΔAξ10‖','半径交差項R_F','歩の二乗項D_F','真の直交二乗の寄与','合計ΔF'],[[label(s['replays'][name]),fmt(d['total']['da'],4),fmt(d['total']['radial_norm'],5),fmt(d['total']['step_norm'],5),fmt(d['total']['perpendicular_norm'],5),fmt(d['total']['delta_F'],5)] for name,d in cpu['replays'].items() if s['replays'][name]['temperature']==1]))
    chunks.append('''後期SL共有の2本では半径項が二乗項を打ち消し、Fはわずかに減った。
これは保存状態間の長期増加と矛盾しない。
回し直しは一つの次課題の1500更新までである。
長期の各区間には他の課題と残り3500更新も含まれる。
この範囲外の符号を測らずに延長しない。

### t20〜50で較正してt60〜100へ外挿

較正点は課題20・30・40・50に固定した。
検証点は課題60・70・80・90・100に固定した。
全点でのあてはまりと、後半への外挿を区別する。
時間方向に較正と検証を分けたが、この軌跡は既に観測済みである。
盲検の未来予言とは数えない。

第一模型はF=C(t/20)^αである。
第二模型は‖Q‖²=a_q+b_qt、‖K‖²=a_k+b_ktと、R・G・Tそれぞれのべき則を使う。
第二模型のFは√[(a_q+b_qt)(a_k+b_kt)]R(t)G(t)T(t)/√64である。
R・G・Tの係数も較正点だけから求めた。

第三模型は、20→30・30→40・40→50のΔF²の半径項と二乗項の課題あたりの率を一定とする。
課題50の実測F²から、その率を加算して外挿した。
第四模型は、その二つの率を区間中点の一次式に当てる。
後二つは今回追加した事後的な模型である。
一次式の係数を外挿に都合よく修正しない。
''')
    rows=[]
    for arm in ('SL','R0L'):
        fit=s['fits'][arm]['a_fixed']
        rows.append([arm,'べき則',fmt(fit['early']['alpha']),fmt(fit['early']['heldout_mape']*100,2)+'%',fmt(fit['early']['predicted'][-1],4),fmt(fit['early']['actual'][-1],4)])
        m=s['fits'][arm]['factor_model'];rows.append([arm,'六因子のうちP・R・G・T','—',fmt(m['heldout_mape']*100,2)+'%',fmt(m['predicted'][-1],4),fmt(m['actual'][-1],4)])
        for key,name in [('constant_energy','一定の二乗増分'),('affine_energy','一次の二乗増分')]:
            d=a['forecasts'][arm][key];rows.append([arm,name,'—',fmt(d['heldout_mape']*100,2)+'%',fmt(d['rows'][-1]['predicted'],4),fmt(d['rows'][-1]['actual'],4)])
    chunks.append(table(['腕','模型','較正のα','後半MAPE','課題100の予測','課題100の実測'],rows))
    chunks.append('''MAPEは検証5点の平均絶対相対誤差である。
SLでは前半のべき則と一定の二乗増分が後半を過小予測した。
半径項の一次外挿はさらに大きく外れた。
R0Lでは一定の二乗増分が後半をよく近似した。
これはこの保存軌跡内での外挿であり、独立seedでの検証ではない。
''')
    forecast_csv=[]
    for arm in ('SL','R0L'):
        rows=[]
        power=s['fits'][arm]['a_fixed']['early'];factor=s['fits'][arm]['factor_model']
        for i,t in enumerate(range(60,101,10),4):
            c=a['forecasts'][arm]['constant_energy']['rows'][i-4];l=a['forecasts'][arm]['affine_energy']['rows'][i-4]
            row=dict(arm=arm,t=t,actual=power['actual'][i],power=power['predicted'][i],factor=factor['predicted'][i],constant_energy=c['predicted'],affine_energy=l['predicted'])
            forecast_csv.append(row);rows.append([t]+[fmt(row[k],4) for k in ('actual','power','factor','constant_energy','affine_energy')])
        chunks.append(f'\n**{arm}の検証点。**\n'+table(['課題末','実測F','べき則','P・R・G・T','一定の二乗増分','一次の二乗増分'],rows))
    chunks.append('''係数を以下に残す。
‖Q‖²と‖K‖²の係数は切片aと傾きbの順に示す。
R・G・TはC(t/20)^αである。
''')
    rows=[]
    for arm in ('SL','R0L'):
        d=s['fits'][arm]['factor_model'];q=d['q_norm2_slope_intercept'];k=d['k_norm2_slope_intercept']
        rows.extend([[arm,'‖Q‖²',fmt(q[1],6),fmt(q[0],6),'a+b t'],[arm,'‖K‖²',fmt(k[1],6),fmt(k[0],6),'a+b t']])
        for key,p in d['factors'].items():rows.append([arm,key,fmt(p['at20'],6),fmt(p['alpha'],6),'C(t/20)^α'])
    chunks.append(table(['腕','量','a または C','b または α','形'],rows))
    chunks.append(table(['腕','二乗尺度の率','一定模型の値/課題','一次模型の切片','一次模型の傾き/課題'],[[arm,'半径' if k=='radial_energy' else '歩の二乗',fmt(a['forecasts'][arm]['constant_energy']['coefficients'][k][0],6),fmt(a['forecasts'][arm]['affine_energy']['coefficients'][k][1],6),fmt(a['forecasts'][arm]['affine_energy']['coefficients'][k][0],6)] for arm in ('SL','R0L') for k in ('radial_energy','step_energy')]))
    chunks.append('''全t20〜100で記述的に当てたべき指数も示す。
これは較正後の外挿精度ではない。
''')
    chunks.append(table(['腕','固定ξ10のFの指数','現在ξ＋bのSの指数'],[[arm,fmt(s['fits'][arm]['a_fixed']['all']['alpha']),fmt(s['fits'][arm]['S']['all']['alpha'])] for arm in ('SL','R0L')]))
    chunks.append('''N10は固定token尺度について支持された。
現在のtokenを含むR0LのSの指数は0.5を超えた。
尺度の定義を変えて同じ指数の主張をしない。
固定FのSLの指数も1未満であり、単純なべき則から絶対増分の加速は導けない。

### 尺度から飽和への条件付き導出

固定したscore方向zが一意な最大値を持ち、その最小gapをg>0とすれば、softmax(λz)の最大候補以外の確率は合計で(N−1)e^(−λg)以下なので、λの増加で飽和する。
同点の最大候補がある場合は単一候補への飽和は導けない。
Aのノルムだけが増えても方向と入力が変わる場合には、この固定方向の結論をそのまま使えない。

### 腕の違いが入る図式

実線のうち「恒等式」と書いた段は導出である。
「測定」と書いた段はこの箱の観測または局所対照である。
破線の段は未支持または未導出である。

```mermaid
flowchart TD
  H[共有head・装填] -->|局所測定：当てはめの尾| C[更新が続く・Σcが増える]
  C -->|恒等式：時間整合ρも必要| W[重みの変位]
  W -->|恒等式：KᵀΔQ・ΔKᵀQ・積・γ| A[作用素の変位ΔA]
  A -->|恒等式：半径項＋二乗項| F[固定token尺度Fの変化]
  F -->|固定score方向と正のgapを仮定| S[注意の飽和]
  S -->|条件付き微分・測定| G[勾配の集中]
  G -. 低階数の歩を介した全媒介は未同定 .-> W
  S -->|score/4の局所対照で一部支持| A
  H -->|測定| R[読み出しの空間の持続]
  R -. N8：上流信号の持続の優位は外れ .-> U[δq・δkの部分空間]
  U -. 保持Tを運ぶ説明は未支持 .-> F
  S -->|前回の微分経路診断| B[勾配の集中と体の遅さ]
```

## 仮定と導出の範囲

| 式・説明 | 必要な条件 | 今回の格 |
|---|---|---|
| 六因子の積 | 同じrmsと同じhead集計を使う | 厳密 |
| Fの離散式 | 固定したξと同じ内積を使う | 厳密 |
| 真の直交二乗の分離 | a≠0 | 厳密 |
| Adamの累積歩 | 保存moment・bias correction・epsを含める | 数値照合 |
| γの縮小 | 前処理後の内積が縮小条件を満たす | 条件式は導出、符号は測定 |
| 10課題の√10換算 | 同じ一課題変位、課題間直交、1500後を無視 | 条件付き近似 |
| Fの時間模型 | 前半の係数を後半へ延長できる | 外挿で検査、一部外れ |
| 共有head→上流空間の持続 | 読み出し以外のJacobianや残差の制御が要る | N8は外れ |
| 特徴の潰れ→新headの同方向化 | Adamと課題による方向変化を制御する必要 | N9は逆 |
| 尺度→飽和 | 固定score方向、一意な最大候補、正のgap | 条件付き導出 |

## N1〜N10の採点

当初のPROGRESS.mdの基準を保った。
追加した事後的な外挿模型を事前予測の成功には数えない。
''')
    chunks.append(table(['ID','判定','証拠'],[[d['id'],d['verdict'],d['evidence']] for d in a['scores']]))
    chunks.append('''## Q6：既知の点とこの箱で測った点

以下は前の実行で原典を確認した記録に基づく。
原典の節と使った主張は `codex_literature.json` に保存した。
''')
    lit=json.loads((OUT/'codex_literature.json').read_text())
    chunks.append(table(['原典','使った主張','適用の限界'],[[f"[{d['authors']} ({d['year']})]({d['url']})",d['claim'],d['limit']] for d in lit]))
    chunks.append('''尺度の増加とentropy collapseの結び付き自体は既知である。
この箱で新しく測ったのは、head共有とhead0の局所対照、六因子の腕差、課題内の離散増分、軟化への変位応答、上流空間の持続の反証である。
tokenのrank collapseと、画像特徴の低参加比と、接核の階数低下を同一の現象とは呼ばない。

## 検証・資源・未解決

前の測定で入力checkpointとAdamの45ファイルのhashが不変だった。
再開後も既存生出力とscriptのhashを照合した。
今回のCPU差分ではCUDAを初期化しなかった。
''')
    rows=[['六因子の相対誤差',f"{ver['identity_max_relative_errors']['six_factor']:.3g}"],['作用素再構成の相対誤差',f"{ver['identity_max_relative_errors']['operator_direct']:.3g}"],['γの勾配恒等式の相対誤差',f"{ver['max_gauge_error']:.3g}"],['累積Adam変位の相対誤差',f"{ver['max_integrated_adam_error']:.3g}"],['CPU共通Fと既存Fの最大相対差',f"{cpu['max_reference_relative_error']:.3g}"],['CPU差分の最大恒等式絶対誤差',f"{max(d['identity_absolute_error'] for rows in cpu['checkpoints'].values() for d in rows):.3g}"],['以前の測定の最大RSS',f"{ver['max_rss_mb']:.0f} MiB"],['以前の測定の最大GPU割当',f"{ver['max_cuda_peak_mb']:.0f} MiB"],['再開後のCPU差分の最大RSS',f"{cpu['rss_mb']:.0f} MiB"],['再開後の既存ファイルhash照合',f"{cpu['input_files_checked']}件、変更{len(cpu['changed_inputs'])}件"]]
    chunks.append(table(['検証項目','値'],rows))
    chunks.append('''共有headから保持Tと入力Iの差が長期に形成される機構は未導出である。
γの長期の縮小をhead共有の有無だけから決める式も無い。
一課題1500更新の測定から、残り3500更新の符号は決まらない。
保存の無い中間課題の個別の増分は決まらない。
同じ温度介入でも参照tokenにより尺度変化の解釈は変わる。
一つのseedの記述相関と局所対照から、全課題の因果媒介率は決まらない。

## 作成ファイル

本文はこのファイルである。
生出力と進行記録は次のディレクトリに置いた。

`/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation/`

| ファイル | 内容 |
|---|---|
| codex_checkpoints.json | 六因子と全保存状態の軌跡 |
| codex_replay_*.json / npz | 温度1の7本とscore/4の1本の毎更新・25更新記録 |
| codex_signals.json / codex_signals_*.npz | 固定256枚の上流信号・特徴・部分空間 |
| codex_summary.json / codex_verification.json | 前の実行の集計と恒等式・資源監査 |
| codex_finish_cpu.json | 共通ξ10での全保存差分と回し直し差分 |
| codex_finish_analysis.json | 活動量の比例則、追加外挿、N1〜N10採点 |
| codex_finish_checkpoints.csv | 課題末差分の半径項・平行二乗・真の直交二乗 |
| codex_finish_windows.csv | 全回し直しの窓ごとの損失・c・尺度変化 |
| codex_finish_forecasts.csv | t60〜100の全外挿値 |
| codex_finish_input_hashes.json / codex_finish_manifest.json | 再開後の入力照合と成果物一覧 |
| codex_literature.json | 原典・使った主張・範囲 |
| PROGRESS.md | 事前予測、失敗、時刻付きの進行記録 |

| script：analysis/vit_long_1024/posthoc_2blk/ | 役割 |
|---|---|
| codex_qkform_common.py | 作用素・尺度・差分の共通関数 |
| codex_qkform_checkpoints.py | 六因子の全保存状態の測定 |
| codex_qkform_replay.py / codex_qkform_queue.py | 逐次の短い回し直し |
| codex_qkform_signals.py | 上流部分空間の測定 |
| codex_qkform_summarize.py / codex_qkform_verify.py | 前の実行の集計と検証 |
| codex_qkform_finish_cpu.py | 今回追加したCPU保存差分 |
| codex_qkform_finish_analysis.py | 今回追加した比例則と外挿の検査 |
| codex_qkform_finish_report.py | 今回追加した本文・表・図の生成 |

![固定尺度・六因子・後半外挿](codex_qkform_growth.png)

![課題内の活動と外れた持続予測・軟化対照](codex_qkform_tests.png)
''')
    export_csv(OUT/'codex_finish_checkpoints.csv',increment_csv)
    export_csv(OUT/'codex_finish_windows.csv',window_csv)
    export_csv(OUT/'codex_finish_forecasts.csv',forecast_csv)
    (RESULTS/'posthoc_qk_formation_codex.md').write_text('\n'.join(chunks))
    figures(s,cpu,a)
    with (OUT/'PROGRESS.md').open('a') as f:
        f.write(f'\n## {now} — 本文と全表の完成\n\nQ2の8本24窓、Q3の軟化対照とN7、Q4のN8・N9の逆方向の外れ、Q5の全38保存区間と7本の離散増分、較正20〜50・検証60〜100、N1〜N10を本文へ反映した。歩の数×cの同一係数によるΔF予測は符号も外れた。一定二乗増分の後半MAPEはSL 17.40%、R0L 0.889%。一次の二乗増分はSL 38.13%、R0L 15.44%で外れた。\n')
    print(f'Wrote {RESULTS/"posthoc_qk_formation_codex.md"}')

if __name__=='__main__':main()
