"""Build the staged Japanese report solely from saved analysis outputs."""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
ROOT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT=ROOT/'posthoc_r0'
REPORT=REPO/'results/vit_long_1024_2blk_rlfix/posthoc_r0_collapse_codex.md'


def read(name):return json.loads((OUT/name).read_text())
def num(x,n=3):
    if x is None:return '未到達'
    if isinstance(x,str):return x
    if isinstance(x,int):return str(x)
    return f'{x:.{n}f}'
def table(headers,rows):
    def cell(v):return str(v).replace('|',r'\|')
    return '\n| '+' | '.join(map(cell,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(cell,row))+' |\n' for row in rows)+'\n'


def main(args):
    d=read('codex_checkpoints.json');rows=d['rows'];index={(r['arm'],r['task']):r for r in rows}
    tr=read('codex_transport.json');transport={(r['arm'],r['task']):r for r in tr['rows']}
    out=[]
    def add(s):out.append(s.strip()+'\n')
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
    add(f'# LNなし・headを戻すR0の崩れの機構\n\n{stamp}。'+('最終版。' if args.final else '途中版。Q1・Q2・静的反実仮想まで確定し、回し直しを実行中。'))
    add('seed 10252010の保存状態からの事後解析。\n予測と途中の判断は `posthoc_r0/PROGRESS.md` に時刻付きで残した。\n一文一主張で、導出・測定・局所介入を分ける。')
    add('## 答え\n\nR0の注意1は、巨大なCLSを全tokenへ写していなかった。\n主にFFN0がpatch側へ書いた出力をvalue経路で運んでいた。\nCLSへの注意集中は、むしろSの中期に現れた。\n一様乱数ラベルだけから全unitの行が共通成分へ逆らうことは導けない。\n共通出力の抑制に必要な追加条件を式にできる。\n保存端点の沈みの帳簿では、入力移動が大きかった。\n後期R0もhead学習率の変更でT90に届いた。\n初期の死の割合は到達不能の判定量にならなかった。\n凍結体の新headは全条件で未到達だった。\n救済は体の後続学習を含むため、固定特徴に対するheadだけの尺度問題とは結論しない。')
    add(table(['問','結論','根拠の種類'],[
        ['Q1 書き手','FFN0のpatch出力を注意1が運ぶ。巨大CLSのbroadcastではない','保存状態の厳密な線形分解'],
        ['Q1 長期形成','入力の増幅を測った。head resetから成長率を閉じて導く式は未得','測定と未導出を区別'],
        ['Q2 負整列','一様ラベルだけでは導けない。自己項は抑制側だが交差項は不定','条件付き導出・反例・勾配照合'],
        ['Q3 直接原因','開始時の死の割合またはηΣ|h|の閾値だけでは結果が決まらない','初期状態をそろえたhead lr介入'],
    ]))
    if (OUT/'codex_summary.json').exists():
        add('中期R0ではhead学習率の変更で持続した暗記が回復した。\n後期R0では初回T90が回復しても、高い暗記率は持続しなかった。\n以下は同じ5,000更新予算での比較である。')
        main_results=[]
        for task,mode,suffix in [(30,'base',''),(30,'lr01','_full'),(30,'lr001','_full'),(60,'base',''),(60,'lr001','_full'),(60,'scale01','_full')]:
            r=read(f'codex_replay_R0_t{task}_{mode}{suffix}.json')
            tail=[q['acc'] for q in r['records'] if q['step']>=4000]
            main_results.append([f'{task}→{task+1}',mode,num(r['T90']),num(r['records'][-1]['acc'],4),num(sum(tail)/len(tail),4)])
        add(table(['R0の切替','介入','T90','5,000更新末の暗記率','最後1,000更新の平均暗記率'],main_results))
        add('baseは元の学習率である。\nlr01・lr001はheadだけを1/10・1/100へ変えた。\nscale01はhead入力を0.1倍した。\nこの表はeager回し直しの結果であり、本走のT90とは区別する。')
    add('## 方法・欠測・集計\n\n固定probeは既存の64枚と全65tokenを使う。\nCLS統計も同じ64枚で測る。\nlogit尺度と勾配診断は固定した暗記画像2,500枚を使う。\ncheckpointの順伝播はfloat32で、統計と端点帳簿はfloat64で計算した。\n勾配の式は別途float64のautogradと照合した。\nunitの状態中央値は中央2個の平均を使う。\n既存torch.medianの中央下側とはわずかに異なる。\n各unitの恒等式は加法的だが、各列の中央値は加法的ではない。\n全tokenの平均m2と画像平均のCLS残差は別の量である。\n本走の全層平均の死と、今回の最終FFNだけの死も区別する。')
    add(table(['対象','扱い'],[
        ['model_t25','未保存。介入は欠測。代替対照を課題20末に追加'],
        ['Adam t20・40','未保存。全介入をfresh Adamで比較'],
        ['Adam t10・30・60','保存momentと更新回数を引き継ぐ'],
        ['R0 head reset','重みとbiasだけをゼロへ戻す。保存momentは残す'],
        ['Sの全体学習','共有headを残す'],
        ['凍結体','新しい10行のゼロheadとfresh Adam'],
        ['追加の凍結体対照','R0/S t30・60は全体学習と同じ初期head・保存Adamを使い、体の更新だけ停止'],
        ['回し直し','同じ次課題、batch128、同じshuffle、25更新ごとに2,500枚評価'],
        ['更新予算','一課題5,000以内。成功後250更新追跡。指定baselineと追加4条件は5,000全体'],
    ]))
    add('実装のラベルはiid一様乱数である。\n各課題のクラス枚数が厳密に等しいわけではない。\n期待ラベルの一様性と、訓練後のgateとラベルの独立性は別の仮定である。')
    add('### 全token平均とCLS平均を混ぜない\n\n損失が直接見る最終FFNの位置はCLSである。\n全tokenのm2と、画像間のCLS平均m_CLSは別の母集団の平均である。\n特にSの中期ではノルムも向きも異なる。\n全tokenでの共通活性を、全画像に共通のlogitへ直ちに置換できない。')
    if (OUT/'codex_mean_populations.json').exists():
        pop=read('codex_mean_populations.json')
        add(table(['腕','課題末','‖m2‖ 全token','‖m_CLS‖','両平均のcos','CLS入力rms','最終FFN後CLS rms'],[
            [r['arm'],r['task'],num(r['all_mean_norm'],1),num(r['cls_mean_norm'],1),num(r['mean_cosine']),num(r['cls_u2_rms'],1),num(r['cls_after_ffn_rms'],1)]
            for r in pop if r['task'] in [10,30,40,60,100]]))
    add('### 本走のT90と課題末の暗記率は別\n\n依頼の「課題28からT90に届かない」は、一律には成り立たない。\n課題28・30は一度到達した後に暗記率が落ちた。\n後期にも到達する課題がある。\n各課題の初回到達と課題末の性能を別に示す。')
    original=[]
    for arm in ['R0','S']:
        path=ROOT/'runs'/arm/'seed10252010/per_task.csv'
        for r in csv.DictReader(path.open()):
            if int(r['task']) in [25,28,30,31,40,41,60,61,100]:
                original.append([arm,r['task'],r['T90'] if int(r['T90'])<=5000 else '未到達',num(float(r['mem_acc_end']),4)])
    add(table(['腕','課題','本走T90','課題末の暗記率'],original))
    add('## Q1：共通成分の書き手\n\n### 注意のCLS成分は厳密に分けられる\n\n注意headをr、画像をi、queryをq、keyをkとする。\nαᵣᵢqkはsoftmaxの重みとする。\nvᵣᵢk=Vᵣxᵢk+bᵣとする。\nPᵣは出力射影のheadに対応する列blockとする。\n\nAᵢq = ΣᵣPᵣαᵣᵢq0vᵣᵢ0 + ΣᵣPᵣΣₖ>₀αᵣᵢqkvᵣᵢk + b_P。\n\n第1項がCLS由来である。\n第2項がpatch由来である。\n第3項は出力射影のbiasである。\nvalueのbiasは各key側へ含めた。\n各項を画像・queryで平均してm2方向へ射影する。\n符号付きの射影なので、寄与が負または100%超になることもある。\nこの分解は保存状態の順伝播の帳簿であり、学習の因果媒介率ではない。')
    add('### 全保存時点：R0のCLS sink予測は外れた')
    for arm in ['R0','S']:
        add(f'**{arm}。**')
        trows=[]
        for r in rows:
            if r['arm']!=arm:continue
            a=r['attention1'];c=r['contrib'];n=r['norms']['h1']
            trows.append([r['task'],num(r['m2_norm'],1),num(a['cls_mass']),num(a['cls_mass_cls_query']),num(a['amax_mean']),num(n['cls_rms']/n['patch_rms']),num(c['attn1'],1),num(c['attn1_cls'],2),num(c['attn1_patch'],1)])
        add(table(['課題末','‖m2‖','CLS注意質量:全query','CLS query→CLS','最大注意重みの平均','注意前CLS/patch rms','注意1の寄与','うちCLS','うちpatch'],trows))
    add('R0 t10のCLS注意質量は、一様注意の基準より少し大きい程度だった。\nそのCLS由来の寄与は注意1全体のごく小さい部分だった。\nP1・P2は外れた。\nR0 t30でもCLS由来は過半に届かなかった。\n一部のheadでCLS集中が現れることと、出力全体の主な書き手がCLSであることを区別する。\nS t10〜60では注意1が主に小さいCLSを読んだ。\nS t70以降はこの経路の選択が変わった。\nこの切替の長期の原因は、この静的分解だけでは決まらない。')
    add('### FFN0の書き込みを注意1が運ぶ\n\nLNなしでは注意1の入力はx=e+a0+f0である。\n保存したαを固定すれば、valueの線形性により注意1の出力をe・a0・f0由来に分けられる。\n注意重み自体がf0に依存する非線形性は、この帳簿では固定している。')
    trows=[]
    for arm in ['R0','S']:
        for task in [1,5,10,20,30,40,60,80,100]:
            r=index[arm,task];t=transport[arm,task];c=t['contrib']
            trows.append([arm,task,num(r['contrib']['ffn0'],1),num(r['contrib']['attn1'],1),num(c['e']['to_m2'],1),num(c['attn0']['to_m2'],1),num(c['ffn0']['to_m2'],1),num(c['value_bias']['to_m2']+c['proj_bias']['to_m2'],2)])
    add(table(['腕','課題末','直接FFN0','注意1','注意内:埋め込み','注意内:注意0','注意内:FFN0','注意内:bias'],trows))
    add('R0では直接のFFN0だけを見ると、共通成分の主な書き手を取り逃がす。\n注意1を通って戻るFFN0由来の成分が大きかった。\nSの中期では、FFN0の大きなpatch残差が直接m2へ入る一方、注意1のCLS読み出しは小さかった。\n追加予測P13は指定した4状態で支持された。')
    add('### 平均注意出力の成長の恒等式\n\nZᵣ=EᵢqΣₖαᵣᵢqkxᵢkとする。\nZはheadごとの384次元入力を連結した量である。\nv=concatᵣ(VᵣZᵣ)、p=Pvとする。\nVはheadごとのvalue作用素をまとめたものとする。\n\n‖平均A‖ = ‖Z‖ ‖V‖_F ‖P‖_F G_V G_P G_b。\nG_V=‖v‖/(‖V‖_F‖Z‖)、G_P=‖p‖/(‖P‖_F‖v‖)、G_b=‖平均A‖/‖p‖。\n\nこの積は非零の分母で厳密である。\nG_VとG_Pは方向と打ち消しを含む。\n入力因子Zには入力の増大と注意による選択の変化が混ざる。\n各log差は加法的だが、因果寄与率ではない。')
    trows=[]
    for arm in ['R0','S']:
        for t0,t1 in [(1,10),(10,40),(40,60),(60,100)]:
            a,b=transport[arm,t0],transport[arm,t1]
            lg={k:math.log(b['factors'][k]/a['factors'][k]) for k in a['factors']}
            trows.append([arm,f'{t0}→{t1}',num(math.log(b['attention_mean_norm']/a['attention_mean_norm'])),num(lg['input_read_norm']),num(lg['value_fro']+lg['proj_fro']),num(lg['value_alignment']),num(lg['proj_alignment']),num(lg['bias_ratio'])])
    add(table(['腕','区間','全log増加','入力Z','重みノルム積','value方向','proj方向','bias比'],trows))
    add('R0 t10→40の入力Zの増加は、value・projのノルム積の増加を大きく上回った。\nvalueの方向因子は増大の一部を打ち消した。\n注意1の出力増大を、重みノルムだけの爆発とは呼べない。\nP3・P14は支持された。')
    trows=[]
    for arm in ['R0','S']:
        for task in [1,5,10,20,30,40,60,100]:
            r=index[arm,task];t=transport[arm,task]
            trows.append([arm,task,num(r['norms']['h1']['cls_rms'],1),num(r['norms']['h1']['patch_rms'],1),num(r['ffn0_open']),num(r['ffn0_mean_activation_norm'],1),num(r['weights']['fc2_0'],2),num(r['weights']['v1'],2),num(r['weights']['proj1'],2)])
    add(table(['腕','課題末','注意1前CLS rms','patch rms','FFN0開口割合','平均活性ノルム','FFN0 fc2ノルム','value1ノルム','proj1ノルム'],trows))
    add('FFN0の平均出力はU₀ E[ReLU(z₀)]+b₂,₀である。\nReLUの非負性だけでは、この出力ノルムの単調増加は導けない。\n開口割合だけでなく、開いた値の大きさとfc2による方向も必要である。\nこの段の長期成長は測った事実である。')
    add(table(['腕','課題末','埋め込みpatch rms','注意0後patch rms','FFN0出力patch rms','注意1出力patch rms'],[
        [arm,t,*[num(index[arm,t]['norms'][k]['patch_rms'],1) for k in ['e','u1','ffn0','attn1']]]
        for arm in ['R0','S'] for t in [10,20,30,40,60,100]]))
    add('R0 t10→40のpatch入力は、埋め込みから注意0、FFN0まで段階的に増大した。\nFFN0の出力増大が特に大きかった。\n平均だけでなくpatch残差のrmsでもこの増大を確認した。\nこの順伝播の増幅の所在と、長期にその重みが形成される原因を区別する。')
    add('### CLS側では注意1の後に大きくなる\n\nR0のCLS queryも、指定した状態では主にpatch側を読んだ。\n注意1より前のCLSが全token中で突出していたわけではない。\nCLSの大きな残差は注意1の出力を加えた後に現れる。\nさらに最終FFNが加わるため、head特徴と注意1直後の残差の尺度も同一ではない。')
    add(table(['腕','課題末','CLS query→CLS質量','注意1前CLS rms','注意1出力CLS rms','注意1後CLS rms','最終FFN後CLS rms'],[
        [arm,t,num(index[arm,t]['attention1']['cls_mass_cls_query']),*[num(index[arm,t]['norms'][k]['cls_rms'],1) for k in ['h1','attn1','u2','h2']]]
        for arm in ['R0','S'] for t in [10,30,40,60]]))
    add('### headと特徴の尺度\n\n使用する10行のheadをクラス中心化してWcとする。\nH=√E‖h‖²、L=√E‖Wc h‖²、κ=L/(‖Wc‖_F H)と置く。\nL=‖Wc‖_F Hκは恒等式である。\nbiasのクラス中心化成分は別に記録する。\nheadと特徴のノルムだけではlogit尺度は決まらない。')
    trows=[]
    for task in [1,10,20,30,40,60,100]:
        for arm in ['R0','S','R0L','SL']:
            t=index[arm,task]['train']
            trows.append([arm,task,num(t['h_norm_rms'],1),num(t['centered_head_norm']),num(t['alignment_factor'],4),num(t['centered_linear_logit_rms'],2),num(t['lr_l1_median']),num(t['acc'],4)])
    add(table(['腕','課題末','H:特徴rms','‖Wc‖','κ','L:logit rms','ηΣ|h|中央値','暗記率'],trows))
    add('R0では大きな特徴と小さなheadが同時に現れた。\n崩壊期にはκも低下した。\n「headより体のパラメータ数が多いから体が尺度を担うほうが速い」は導出にならない。\n一歩のlogit差は、厳密にはΔℓ=ΔW h + WΔh + ΔWΔh + Δbである。\n体側のΔhはJacobianとAdamの座標別の歩に依存する。\nパラメータ数だけではこれらの内積の符号も大きさも決まらない。\n\nLNもaffine後の‖h‖を厳密には固定しない。\naffine前の中心化rmsを規格化する。\nR0Lでは学習されたaffineを通したHが増えた。\nその増え方とR0の爆発は量的に異なる。')
    add('### CEが尺度の成長を促す限定した条件\n\nhを仮想的にαhへ変えると、∂L/∂α=E[(p−y)ᵀWh]が成り立つ。\nbiasを無視し、正解logitが全ての他クラスより大きい画像では、(p−y)ᵀWh=Σc≠y p_c(ℓ_c−ℓ_y)<0となる。\nその画像の損失は特徴尺度の増大で下がる。\n同じlogitの尺度はhead側を伸ばしても変えられる。\nこの局所的な符号は両側のどちらが速く成長するかを決めない。\n間違えている画像や大きなbiasがある場合の符号も固定しない。\nR0のheadリセット直後はW=0なので、この尺度微分もゼロである。\nhead再学習後のCEの尺度方向は、装填とは別の候補となる。\nただしθの更新がこの仮想的な一様尺度方向を実現するとは限らない。')
    if (OUT/'codex_radial.json').exists():
        rad=read('codex_radial.json')['rows']
        add(table(['腕','課題末','旧ラベルの平均尺度微分','旧ラベルで成長側の割合','新一様ラベル期待の平均尺度微分'],[
            [r['arm'],r['task'],num(r['old']['mean'],5),num(r['old']['growth_side_fraction']),num(r['uniform']['mean'],4)] for r in rad]))
        add('### 開いた最終FFNの出力が残差を打ち消す分\n\nf=h−xを最終FFNの出力とする。\n打ち消し割合は−平均f·平均x/‖平均x‖²と定義する。\nこれは訓練画像のCLSでの符号付き射影である。\n死の割合が高くても、残るunitの出力は大きな残差を打ち消し得る。\nこの帳簿は、死が打ち消しを生じさせたという因果の主張ではない。')
        add(table(['腕','課題末','CLS入力平均ノルム','FFN出力平均ノルム','head特徴平均ノルム','入力平均の打ち消し割合'],[
            [r['arm'],r['task'],*[num(r['final_ffn_cancellation'][k],3 if k=='cancel_fraction' else 1) for k in ['input_mean_norm','output_mean_norm','feature_mean_norm','cancel_fraction']]] for r in rad]))
    add('## Q2：行が共通成分へ逆らうことはどこまで導けるか\n\n### 一様ラベルから導ける式\n\nLNなしの最終FFNについて、xᵢ=m+εᵢ、zᵢⱼ=wⱼ·xᵢ+bⱼ、aᵢⱼ=ReLU(zᵢⱼ)と置く。\n残差を含むhead入力はhᵢ=xᵢ+ΣⱼUⱼaᵢⱼ+b₂である。\n使用headをW、vⱼ=WUⱼ、eᵢ=pᵢ−onehot(yᵢ)とする。\ngᵢⱼ=1[zᵢⱼ>0]vⱼᵀeᵢと置く。\n\n∇wⱼL=Eᵢ[gᵢⱼxᵢ]、∂bⱼL=Eᵢ[gᵢⱼ]。\n∇wⱼL·m=‖m‖² Eᵢgᵢⱼ+Eᵢ[gᵢⱼ(εᵢ·m)]。\n\nこの式は有限標本でも厳密である。\n新しいラベルが固定した網から独立で一様なら、期待誤差はpᵢ−1/Kへ置換できる。\nただしgateとvⱼと画像ごとのpᵢが残る。\nクラス数の釣合いだけでは行勾配の符号は消えない。\n訓練後のラベルに対して独立性を仮定することもできない。\n\n全画像に同じ活性をδだけ足す仮想操作では、∂L/∂δ=vⱼᵀ(平均p−平均onehot(y))となる。\nこの符号はvⱼと平均誤差の向きに依存する。\nReLU(wⱼ·m+bⱼ)を平均活性と同一視する等式は一般にはない。')
    add('### 共通成分を下げる十分条件と反例\n\nすべての画像のlogitがav+cᵢ1で、a≥0とする。\nラベルは独立一様とする。\n期待CEのa微分はvᵀ(softmax(av)−1/K)である。\nこれはa≥0で非負である。\n各vの値が等しくなければa>0で正になる。\n同じ結果はf(a)=logΣexp(av)−a mean(v)の凸性とf′(0)=0からも出る。\nこの限定した形では共通の活性を減らす勾配になる。\n\n他のlogit成分があると符号は反転できる。\nK=2、v=(1,−1)、他の成分=−3v、a=1を考える。\n期待CEの微分はtanh(a−3)=tanh(−2)<0となる。\n一様ラベルでもaを増やす方向へ更新する。\nしたがって全unitの負整列も、そのcosの数値も一様ラベルだけからは導けない。\n\nさらにReLUが閉じると、そのunit自身を通る現在勾配はゼロになる。\n共通成分の成長で負位置が深まっても、行が毎課題さらに回るとは限らない。\nbiasと下流biasによる補償も許される。')
    add('### 自己項の正は厳密に導ける\n\nB(ℓ)=∫₀¹[diag p(tℓ)−p(tℓ)p(tℓ)ᵀ]dtと置く。\n各softmax Jacobianは半正定値なのでBも半正定値である。\np(ℓ)−1/K=B(ℓ)ℓが積分の基本定理から成り立つ。\nℓᵢ=aᵢⱼvⱼ+rᵢ,−jと分ける。\n一様な新ラベルで期待を取った前活性勾配は次になる。\n\ngᵢⱼ=gateᵢⱼ aᵢⱼ vⱼᵀBᵢvⱼ + gateᵢⱼ vⱼᵀBᵢrᵢ,−j。\n\n第1項は非負である。\n第2項の符号は決まらない。\nrᵢ,−jには他unit・残差・biasのlogitが全て入る。\n行への共通方向の投影にはさらにxᵢ·mが掛かる。\nxᵢ·m≥0なら自己項の沈める符号は保たれる。\n交差項との大小関係と、Adamによる前処理は別に測る必要がある。\nこの導出もcosの定量的な着地点を与えない。')
    if (OUT/'codex_ce_self.json').exists():
        ce=read('codex_ce_self.json')['rows']
        add(table(['腕','課題末','訓練画像で生きたunit数','前活性の自己項が正の割合','行投影:自己項の沈める割合','行投影:交差項の沈める割合','行投影:合計の沈める割合','自己/交差の絶対値比中央値'],[
            [r['arm'],r['task'],r['alive'],num(r['self_bias']['sinking_fraction']),num(r['self']['sinking_fraction']),num(r['cross']['sinking_fraction']),
             num(r['total']['sinking_fraction']),num(r['median_abs_self_over_abs_cross'])] for r in ce]))
        add('Bの積分はGauss–Legendre求積で評価した。\n求積点数を増やし、Bℓ=p−1/Kの最大絶対誤差を検査した。\n自己項の正の導出と、その行投影の実測を区別する。')
        add('S t10では自己項の行投影が沈める割合も予測値を下回った。\nP16は外れた。\n前活性の自己項が非負であることと、全token平均への行投影が非負であることは別である。\nCLS入力と全token平均の内積の符号が関与する。')
        if (OUT/'codex_summary.json').exists():
            add(table(['腕','課題末','x_CLS·m_allが負の画像割合','x_CLS·probe CLS平均が負','x_CLS·訓練CLS平均が負'],[
                [r['arm'],r['task'],num(r['cls_dot_all_negative']),num(r['cls_dot_probe_cls_negative']),num(r['cls_dot_training_cls_negative'])] for r in read('codex_summary.json')['ce_input_projections']]))
    add(table(['主張','必要な条件','格'],[
        ['∇wの共通項＋残差項','同じ入力分布と固定m','恒等式'],
        ['新ラベル期待e=p−1/K','網から独立な新しい一様ラベル','条件付きで厳密'],
        ['共通活性を抑える符号','他のlogit差がなくℓ=av+c1、a≥0','十分条件'],
        ['全unitが負へ回る','他のunit・残差・gate・Adamの制御が必要','一般には導けない'],
        ['cos中央値−0.14〜−0.35','保存状態で測る','測定'],
        ['R0リセット直後のbody生勾配','W=0','厳密にゼロ。momentによる更新は別'],
    ]))
    add('### 保存状態での符号照合\n\n期待一様ラベルの誤差を保存headへ戻した。\nR0ではheadを戻さなかった反実仮想の診断である。\n実際のリセット直後のbody生勾配ではない。\n表の「減少側」は負勾配がw·mを減らす割合である。\n生勾配の符号をAdam更新の符号と同一視しない。')
    trows=[]
    for arm in ['R0','S']:
        for task in [1,5,10,20,30,40,60,100]:
            r=index[arm,task];g=r['train']['gradient_diagnostic']['uniform']
            trows.append([arm,task,num(r['cos']['median']),num(r['cos']['negative']),num(g['minus_grad_common_activation']['negative']),num(g['alive_minus_grad_w_dot_m']['negative']),num(r['bias']['median'],4),num(r['wm']['median'],2)])
    add(table(['腕','課題末','cos中央値','cos負の割合','共通活性の減少側','生きたunitのw·m減少側','bias中央値','w·m中央値'],trows))
    add('共通活性だけを動かす診断の符号はおよそ半々だった。\ngateと入力を含む行勾配では減少側が多かった。\nこの差を、すべての共通活性が一様に罰せられるという導出で置き換えられない。\nbiasの絶対値は大きな負のw·mに比べて小さかった。')
    add('### 保存端点の厳密な帳簿\n\nΔ(w·m)=Δw·m₀+w₀·Δm+Δw·Δm。\n前活性平均の変化にはΔbも加わる。\nw=n u、‖u‖=1と置く。\nΔw=(n₁−n₀)u₀+n₁(u₁−u₀)と伸びと回転へ分ける。\n幅sを含む位置r=μ/sには、各前活性項をs₁で割った分とμ₀(1/s₁−1/s₀)が加わる。\nこれらは端点の恒等式である。\n保存間隔を変えると交差項などの配分は変わる。\n課題10以降は10課題の端点差であり、毎課題の値ではない。')
    for arm in ['R0','S']:
        add(f'**{arm}：全token、全3,072unitの中央値。**')
        trows=[]
        for l in d['ledgers']:
            if l['arm']!=arm:continue
            z=l['all'];v=z['terms']
            trows.append([f'{l["start"]}→{l["end"]}',num(z['delta']['median'],2),*[num(v[k]['median'],3) for k in ['weight','input','cross','bias','rotation']]])
        add(table(['保存区間','Δμ','Δw·m','w·Δm','交差','Δbias','うち回転'],trows))
    add('R0の20→40では、負整列を保った行へ大きな入力が入る分が沈みを運んだ。\n後期R0では行が止まったunitが多く、回転の中央値もゼロになった。\n文字GPT L0の「W項が全seedで主」という結果を、そのまま移せない。\n共通しているのは、負整列と入力共通成分の成長が共存する形である。\n文字GPTの39→100とViTの20→40は異なる窓である。')
    add('### 幅も増えるが負の平均に追いつかない')
    add(table(['腕','課題末','平均前活性','幅中央値','r中央値','cos中央値','全token死'],[
        [arm,t,num(index[arm,t]['z']['all']['mean'],1),num(index[arm,t]['z']['all']['width'],1),
         num(index[arm,t]['z']['all']['r']),num(index[arm,t]['cos']['median']),num(index[arm,t]['z']['all']['dead'])]
        for arm in ['R0','S'] for t in [1,10,20,30,40,60,100]]))
    add('行のノルムだけを正の定数倍すると、biasを無視できる場合に平均と幅が同じ倍率になる。\nrはその純粋な倍率では変わらない。\n平均の帳簿とrの帳簿を別に示す。')
    add(table(['腕','保存区間','Δr','重み/s₁','入力/s₁','交差/s₁','bias/s₁','幅項'],[
        [l['arm'],f'{l["start"]}→{l["end"]}',num(l['all']['delta_r']['median']),
         *[num(l['all']['r_terms'][k]['median'],4) for k in ['weight','input','cross','bias','width']]]
        for l in d['ledgers'] if l['end']>=20]))
    add('P5は早期0→10と後期10→100で分けて採点する。')
    add(table(['腕','窓','入力項が重み項より負の区間数','区間数'],[
        [arm,f'{lo}→{hi}',sum(l['all']['terms']['input']['median']<l['all']['terms']['weight']['median']
                            for l in d['ledgers'] if l['arm']==arm and lo<=l['start'] and l['end']<=hi),
         sum(1 for l in d['ledgers'] if l['arm']==arm and lo<=l['start'] and l['end']<=hi)]
        for arm in ['R0','S'] for lo,hi in [(0,10),(10,100)]]))
    add('## Q3：死と尺度を分ける介入\n\n### ηΣ|h|から導けること\n\n固定hに対するhead一歩のlogit変化はΔℓ=ΔW h+Δbである。\n|Δℓ_c|≤Σⱼ|ΔW_cj||h_j|+|Δb_c|が厳密に成り立つ。\nfresh Adamの最初の一歩では、|ΔW_cj|≤ηである。\nその場合はηΣ|h|が重み部分の上界になる。\n保存momentとbias correctionを含む一般のAdamでは、|ΔW_cj|≤ηは保証されない。\nさらに座標ごとの相殺がある。\nηΣ|h|>1だけでは不安定性の必要条件にも十分条件にもならない。\n実際のhead一歩のlogit差とbodyのlogit差を別に保存した。\n\nhead lrだけを変える対照は、初期の特徴と死を同一に保つ。\n後の体の学習は変わり得る。\nhを0.1倍する介入は、順伝播とbodyへの勾配を変える。\nSでは開始logitも変わる。\nこの介入を純粋なoptimizer変更とは呼ばない。\n凍結体の失敗は、固定特徴で次の乱数ラベルが分離できないことも含む。\nそれだけで死の因果を確定できない。')
    add('### 最終FFNの死は体全体の微分の消失ではない\n\n最終FFNと残差の和はh=x+U ReLU(W₁x+b₁)+b₂である。\nその入力JacobianはJ_h,x=I+U diag(gate) W₁である。\n全gateがゼロならJ_h,x=Iとなる。\n最終FFN自身の重みへの現在勾配は消えるが、上流へ戻る残差経路は消えない。\nしたがって最終FFNの死の割合だけから、体全体の応答や次課題の到達不能は導けない。\n残る経路が画像ごとの必要な方向を学べるかは別の問題である。\nまたprobe上の死は、全2,500枚で一度も開かないことと同一ではない。')
    all_replays=[]
    for p in sorted(OUT.glob('codex_replay_*.json')):
        try:all_replays.append(json.loads(p.read_text()))
        except json.JSONDecodeError:pass
    longest={}
    for r in all_replays:
        key=(r['arm'],r['task_end'],r['mode'])
        if key not in longest or r['steps']>longest[key]['steps']:longest[key]=r
    replays=[longest[k] for k in sorted(longest)]
    if replays:
        add('### 一課題の回し直し')
        trows=[]
        for r in replays:
            rec=r['records'];tail=rec[-1]
            trows.append([r['arm'],r['task_end'],r['mode'],'保存' if r['loaded_adam'] else '空',num(r['T90']),r['steps'],num(tail['acc'],4),num(max(v['acc'] for v in rec),4),num(tail['loss'],3)])
        add(table(['腕','課題末','介入','Adam','T90','実行更新','最終暗記率','最大暗記率','最終損失'],trows))
        add('同じ条件を追加で5,000更新まで回した場合、この表は長い曲線を採用する。\n先に終えた短い曲線は別ファイルで保存している。\n実行更新が異なる行の最終暗記率を、同じ予算の優劣とは読まない。')
        add('R0 t30からのbaselineは本走の課題31と初回T90が異なった。\n本走は未到達だが、今回のeager回し直しは一度到達した。\n元のcompile実行との軌跡の完全一致は主張しない。\n介入の因果比較は、同じeager実装とbatch順を使った条件間に限定する。')
        if len(all_replays)>len(replays):
            add(table(['腕','課題末','介入','先行した短い追跡の更新数','その時点の暗記率'],[
                [r['arm'],r['task_end'],r['mode'],r['steps'],num(r['records'][-1]['acc'],4)]
                for r in all_replays if r['steps']<longest[r['arm'],r['task_end'],r['mode']]['steps']]))
        add('baseは元の学習率である。\nlr01・lr001はheadだけを1/10・1/100へ変える。\nscale01はhead入力を0.1倍する。\n途中版の実行中行はその時点までの結果である。' if not args.final else 'baseは元の学習率である。\nlr01・lr001はheadだけを1/10・1/100へ変える。\nscale01はhead入力を0.1倍する。')
        add('### R0の課題内の尺度軌跡')
        trows=[]
        for r in replays:
            if r['arm']=='R0' and r['task_end'] in [10,30] and r['mode']=='base':
                for g in r['geometry']:
                    trows.append([r['task_end'],g['step'],num(g['h_norm_median'],1),num(g['lr_l1_median']),num(g['m2_norm'],1),num(g['zmean'],1),num(g['dead']),num(g['cls_dead'])])
        add(table(['開始課題末','更新','probe ‖h‖','ηΣ|h|','‖m2‖','z平均','全token死','CLS死'],trows))
        add('### head lr介入はbodyの後続軌跡も変える')
        trows=[]
        for r in replays:
            if r['arm']=='R0' and r['task_end'] in [30,60]:
                a,b=r['geometry'][0],r['geometry'][-1]
                trows.append([r['task_end'],r['mode'],r['steps'],num(a['h_norm_median'],1),num(b['h_norm_median'],1),
                              num(a['cls_dead']),num(b['cls_dead']),num(r['records'][-1]['acc'],4)])
        add(table(['開始課題末','介入','更新','初期‖h‖','最終‖h‖','初期CLS死','最終CLS死','最終暗記率'],trows))
        add('この表の‖h‖は、特徴倍率を掛ける前の体出力である。\nscale01でheadへ実際に入る特徴は、この値をさらに0.1倍したものになる。')
        add('学習率の変更後には、体の特徴尺度も開口も変化した。\n到達の回復を、凍結した体の同じ特徴をheadだけで読めたという結果に置き換えない。\n特徴のノルムがさらに大きくても、head lrが小さい条件で到達することがある。\n特徴ノルムだけを到達不能の判定量にはできない。')
        add('### 診断した一歩のlogit差')
        trows=[]
        for r in replays:
            if r['arm']=='R0' and r['task_end'] in [30,60] and r['mode'] in ['base','lr001']:
                for s in r['step_diagnostics']:
                    if s['step'] in [1,100,1000,2000,5000]:
                        trows.append([r['task_end'],r['mode'],s['step'],num(s['eta_l1_median']),
                                      num(s['head_logit_rms'],4),num(s['body_logit_rms'],4),num(s['head_logit_max'],4)])
        add(table(['開始課題末','介入','更新','head ηΣ|h|','head一歩rms','body一歩rms','head一歩の最大絶対値'],trows))
        add('rmsは画像とクラスを平均し、各画像のクラス平均logitを引いてから取った。\nhead一歩は旧hを固定したΔW h+Δbである。\nbody一歩は更新後のheadを使ったW_new Δhである。\n両者の和は全logit差と一致する。\nrms同士は交差項があるため加法的ではない。\n少数の診断時点の記録なので、全5,000更新の最大値だとは呼ばない。')
        add('後半の診断時点では、head自身の更新よりbodyの変化が作るlogit差の方が大きかった。\nheadの学習率を下げる介入が効くことと、head自身の一歩がlogit変化の主成分であることは同じ主張ではない。\nbody一歩のrmsにも、学習不能を決める普遍的な閾値を置かない。')
    if (OUT/'codex_linear_probes.json').exists():
        lp=read('codex_linear_probes.json')
        add('### 体を凍結した新しいhead')
        add(table(['腕','課題末','head lr','特徴PR','T90','最大暗記率','最終暗記率'],[
            [r['arm'],r['task_end'],f'{r["head_lr"]:.0e}',num(r['pr'],2),num(r['T90']),num(r['max_acc'],4),num(r['final_acc'],4)] for r in lp['results']]))
        add('R0の中期でも凍結体は到達しなかった。\nP9は外れた。\n後期R0のhead lr変更は到達したので、P10も外れた。\nSもhead lr変更で到達したので、P11も外れた。\nSのbaseline自体も到達している。\nしたがってSのこの2状態を、死が到達不能を決めた対照とは扱えない。')
    if (OUT/'codex_summary.json').exists():
        ss=read('codex_summary.json')
        add('### 同じ5,000更新予算で見た後半の維持\n\n追加の長い回し直しは、同じ保存状態から同じ次課題だけを実行した。\n最後の1,000更新の集計は、25更新ごとの評価点に対する記述統計である。\n初回T90と持続した性能を区別するために示す。')
        add(table(['腕','開始課題末','介入','5,000末','4,000〜5,000平均','同最小','同最大','同区間で0.9以上の評価割合'],[
            [r['arm'],r['task_end'],r['mode'],*[num(r[k],4) for k in ['final','mean','min','max','atleast90_fraction']]] for r in ss['last1000_updates']]))
        add('R0 t30のhead lr0.01は後半の全評価点で高い暗記率を保った。\nR0 t30のhead lr0.1も、到達直後の再低下から回復した。\nP15のlr0.1は回復しないという部分は外れた。\nR0 t60のhead lr0.01は持続しなかった。\nP17は外れた。\nR0 t60の特徴0.1倍も、最終暗記率は到達水準を下回った。\nP20は支持された。\n尺度操作だけで後期の持続崩れまで解けたとは結論しない。')
    if (OUT/'codex_frozen_matched.json').exists():
        matched=read('codex_frozen_matched.json')['results']
        add('### 初期headとAdamをそろえて体だけを凍結\n\n先のlinear probeはheadとAdamも新しくしていた。\n追加対照では全体学習と初期head・保存Adamをそろえた。\nR0はheadをゼロへ戻し、Sは共有headを残した。\n同じ保存状態の固定特徴を使い、体への更新だけを止めた。')
        add(table(['腕','開始課題末','head lr','T90','最大暗記率','5,000末暗記率'],[
            [r['arm'],r['task_end'],f'{r["head_lr"]:.0e}',num(r['T90']),num(r['max_acc'],4),num(r['final_acc'],4)] for r in matched]))
        if all(r['T90'] is None for r in matched):
            add('初期状態をそろえた凍結体でも全条件が未到達だった。\nheadだけの最適化では今回の救済を再現できなかった。\n指定した予算と学習率の範囲では、体の後続の変化が必要だった。\nこれは別のoptimizerや無限の更新でも不可能という証明ではない。')
    add('### m2だけを縮める死の反実仮想\n\nxᵢq=m2+εᵢqを、x′ᵢq=k m2+εᵢqへ置換した。\n重み・biasと中心化残差εを固定した。\nz′ᵢqj=zᵢqj+(k−1)wⱼ·m2となる。\n従ってwⱼ·m2<0ならkの減少で前活性は上がる。\nこの一歩の関数変更を学習による回復と同一視しない。')
    trows=[]
    for arm in ['R0','S']:
        for task in [20,30,40,60,100]:
            r=index[arm,task]
            for k in [1.,.5,.1,0.]:
                a=next(c for c in r['counterfactual'] if c['population']=='all' and c['k']==k)
                b=next(c for c in r['counterfactual'] if c['population']=='cls' and c['k']==k)
                trows.append([arm,task,k,num(a['dead']),a['alive_any'],num(b['dead']),b['alive_any']])
    add(table(['腕','課題末','k','全token死','全tokenで一度でも開く','CLS死','CLSで一度でも開く'],trows))
    add('R0 t40・60では、k=0.1で全tokenの死はゼロになった。\nCLSでも大部分が開き直った。\nP12は支持された。\n保存された行の大部分が不可逆に応答を失ったという解釈は、この静的反実仮想と合わない。\n元の入力分布の下で応答が乏しいことと、重みを固定したまま別の入力で開けることは両立する。')
    add('## 機構の鎖と未導出の段\n\nR0の①装填は、次課題のheadリセットにより走らない。\nR0の②には、保存Adamの残りと、新headが育った後の通常の誤差伝播がある。\n②の長期の符号をパラメータ数だけからは決められない。\nR0の③は、FFN0のpatch出力を注意1が運ぶ経路として測った。\nR0の④では、絶対幅も増えるが負の平均の増大に追いつかない。\nLNありの取り分競合による幅の縮小とは区別する。\nR0の⑤は、固定probe上の開口減少として存在する。\n同じ初期の⑤でもhead lr変更で到達したため、⑤の割合は到達不能の十分条件ではない。\nηΣ|h|だけを使ったhead一歩の行き過ぎという判定も支持されなかった。\nhead lr介入は、最終FFNの再開口と特徴の成長を含む後続の体の学習を変えた。\n局所介入が支持するのは、このheadと体の結合した更新への依存である。\n\nSでは共有headの装填が残る。\nSの中期はFFN0の直接の共通書き込みで最終FFNが先に沈む。\nその時期の注意1は小さいCLSを主に読む。\n後期にCLS集中が崩れるとpatch側の大きな残差も読み出しへ流れる。\n最後の文は保存状態の対応であり、切替の動的因果を導いたものではない。')
    add('### 尺度と死の出現順序\n\n以下の閾値は時系列を要約するための記述上の目印である。\n不安定性や学習不能の必要十分条件ではない。\n最初に閾値を超えた保存時点を示す。\n未保存の課題内や課題間の正確な発生時刻は決まらない。')
    add(table(['腕','ηΣ|h|>1の最初の保存時点','全token死>0.5の最初の保存時点','CLS死>0.5の最初の保存時点'],[
        [arm,next(r['task'] for r in rows if r['arm']==arm and r['train']['lr_l1_median']>1),
         next(r['task'] for r in rows if r['arm']==arm and r['z']['all']['dead']>.5),
         next(r['task'] for r in rows if r['arm']==arm and r['z']['cls']['dead']>.5)] for arm in ['R0','S']]))
    add('この尺度ではR0は尺度の大きさが先で、Sは開口減少が先だった。\n死の時系列自体は単調ではない。\nS t70の全token死はゼロだったが、CLS死はほぼ全unitだった。\n全tokenとCLSでの応答を混ぜない。')
    add(table(['腕','課題末','全token死','CLS死'],[[a,t,num(index[a,t]['z']['all']['dead'],4),num(index[a,t]['z']['cls']['dead'],4)] for a,t in [('S',60),('S',70),('S',80)]]))
    add('```mermaid\nflowchart TD\n  R[head reset] -->|定義| Z[切替時の装填なし]\n  R -. 長期形成は未導出 .-> F[FFN0のpatch残差が増大]\n  F -->|保存状態の線形分解| A[注意1がpatch由来のvalueを運ぶ]\n  F -->|残差加算| M[最終FFNの共通入力m2]\n  A -->|測定| M\n  M -->|負整列の測定と恒等式| D[負の平均と開口減少]\n  A -->|測定| H[CLS残差の尺度増大]\n  H -->|最終FFNも加わる| O[headと体の結合した更新]\n  D -. 割合だけでは到達不能を決めない .-> T[T90と持続暗記の崩れ]\n  O -->|head lr介入で結果が変わる| T\n  S[共有head S] -->|既存の測定| L[装填の押し]\n  L -. 全過程は未導出 .-> F\n```')
    if (OUT/'codex_summary.json').exists():
        add('## 予測の採点\n\n測定後に予測を書き換えていない。\n追加予測も、対応する測定を始める前にPROGRESSへ記録した。\n「支持」は指定した状態・条件での判定である。')
        add(table(['ID','測定前の予測','判定','根拠'],[[r['id'],r['prediction'],r['status'],r['evidence']] for r in ss['predictions']]))
    figure_names=['writers','interventions','trajectories','mean_counterfactual']
    if all((REPORT.parent/f'codex_r0_{name}.png').exists() for name in figure_names):
        add('## 図\n\n図中の曲線は実際に実行した更新時点までを示す。\n介入図は同条件の短い追跡より、追加した長い追跡を優先する。')
        for name in figure_names:
            add(f'![{name}](codex_r0_{name}.png)')
    add('## 原典で確認した主張\n\n既知の現象と、この箱で測った経路を区別する。')
    literature=[
        ['Xiao et al. (2023/ICLR 2024)','https://arxiv.org/html/2309.17453v4','§3.1','意味的に重要でない初期tokenにも注意が集中し、そのKV保持がstreaming推論を支えると観測した。','本箱のR0でCLS sinkがあることの根拠にはならない。'],
        ['Sun et al. (COLM 2024)','https://arxiv.org/html/2402.17762v2','§4・§5','大きな活性と注意集中・暗黙biasを結び、ViTでは画像ごとに異なるpatch位置の大きな活性も観測した。','本箱の平均残差の増大を、同論文の稀な座標活性と同一とはしない。'],
        ['Xiong et al. (ICML 2020)','https://proceedings.mlr.press/v119/xiong20b.html','Abstract・理論','初期化時のpost-LNとpre-LNの勾配尺度の違いを解析した。','全LNを外した継続学習の長期爆発を証明した論文ではない。'],
        ['Dong et al. (ICML 2021)','https://proceedings.mlr.press/v139/dong21a.html','Abstract','残差とMLPのない純粋attentionは深さとともにrank 1へ急速に近づくと証明した。','本箱は残差とMLPを持つので、その定理をそのまま適用しない。'],
        ['Noci et al. (NeurIPS 2022)','https://papers.neurips.cc/paper_files/paper/2022/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html','Abstract','初期化時のtoken rank collapseがquery/keyの勾配を消す機構を解析した。','画像特徴の線形分離性や本箱の学習後T90を直接決めない。'],
        ['Joudaki et al. (ICLR 2026)','https://arxiv.org/abs/2510.00304v3','§4・Fig 4.1・Appendix B','正規化の有無を含むViT等の継続学習で、凍結・重複・有効階数と可塑性の関係を調べた。','本箱のhead reset・ReLU・2 blockの尺度経路は別の測定である。'],
    ]
    dump=OUT/'codex_literature.json'
    dump.write_text(json.dumps(literature,ensure_ascii=False,indent=2))
    add(table(['原典','確認した箇所','使った主張','範囲'],[[f'[{a}]({b})',c,e,f] for a,b,c,e,f in literature]))
    add('## 検証と残る問い')
    verify=read('codex_verification.json')
    err=max(v['analytic_grad_relative'] for k,v in verify.items() if isinstance(v,dict) and 'analytic_grad_relative' in v)
    add(table(['数値検証','値'],[
        ['順伝播の元実装との差','float64の指定3状態で0'],
        ['最終FFNの解析勾配とautogradの最大相対差',f'{err:.3g}'],
        ['同一lrのAdam群分割の一歩の最大差',num(verify['adam_equal_lr_group_split_max_abs'])],
        ['注意1の入力書き手再構成の最大相対差',f'{max(r["identity_relative"] for r in tr["rows"]):.3g}'],
        ['端点帳簿の最大絶対誤差',f'{max(l[p]["identity_max_abs"] for l in d["ledgers"] for p in ["all","cls"]):.3g}'],
        ['保存状態測定のRSS最大MiB',num(d['resources']['rss_mib'],1)],
        ['保存状態測定のGPU割当最大MiB',num(d['resources']['cuda_alloc_mib'],1)],
    ]))
    if (OUT/'codex_audit.json').exists():
        audit=read('codex_audit.json')
        add(table(['最終監査','値'],[
            ['SHA256を照合した既存入力数',audit['checked_existing_files']],
            ['変更された既存入力数',len(audit['changed_existing_files'])],
            ['一課題の全体学習条件数',audit['replay_count']],
            ['新headの凍結体条件数',audit['linear_probe_count']],
            ['初期head・Adamをそろえた凍結体条件数',audit['matched_frozen_count']],
            ['全実験RSS最大MiB',num(audit['max_rss_mib'],1)],
            ['GPU予約最大MiB',num(audit['max_cuda_reserved_mib'],1)],
            ['長い再実行と先行曲線の共通区間','4条件とも全評価点の暗記率が完全一致'],
        ]))
        add('GPU値はPyTorch allocatorの最大値であり、CUDA contextを含まない。\n全ての重い処理をnice 10で逐次実行した。\ngit操作・プロセス停止・vault書込は行わなかった。')
    add('共有headの有無からFFN0のpatch残差の成長率を閉じて導く式は得ていない。\n負整列の初期形成と長期の維持の全てを一様ラベルからは導いていない。\n保存のない課題25末や課題間の細かな増分は復元していない。\n同じ一seedの局所介入を、全seedの必要十分な機構とは呼ばない。\n元の本走はcompile、今回の回し直しはeagerである。\n浮動小数点演算順の差に敏感な領域では、軌跡の完全再現を保証しない。')
    add('R0は「装填なしでも③共通入力の成長から⑤開口減少まで走る」測定例として使える。\nただし開始時の死の割合だけでT90喪失を決める扱いは、今回の介入と合わない。\nhead lr変更後の再開口が救済を媒介するかは、まだ分離していない。\n後期の持続崩れが応答の制約によるか、体の更新の不安定さによるかも未同定である。\nV13では尺度とhead・体の結合した更新への依存を別経路として記す必要がある。')
    add('## 作成ファイル\n\n本文はこのファイルである。\n予測・途中経過・生出力は次に置いた。\n\n`/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0/`')
    add(table(['ファイル','内容'],[
        ['PROGRESS.md','時刻付き予測・外れ・進行記録'],
        ['codex_checkpoints.json','全4腕80保存状態、注意分解・尺度・勾配・端点帳簿・死の反実仮想'],
        ['codex_transport.json','LNなし40状態のvalue経路の入力書き手と成長因子'],
        ['codex_units_*.npz','unitごとの行・入力平均・bias・幅・開口'],
        ['codex_features_*.npz','2,500枚の固定特徴・最終FFNのCLS入力と前活性'],
        ['codex_replay_*.json / log','一課題の介入曲線・尺度軌跡・実際の一歩量'],
        ['codex_linear_probes.json','18条件の凍結体の新head'],
        ['codex_frozen_matched.json','初期head・保存Adamをそろえた12条件の凍結体'],
        ['codex_radial.json','特徴尺度のCE微分と最終FFNの打ち消し'],
        ['codex_ce_self.json','期待CEのPSD自己項・交差項と求積誤差'],
        ['codex_mean_populations.json','全token平均とCLS平均の区別'],
        ['codex_summary.json','事前予測の採点・bias分担・最後の1,000更新'],
        ['codex_state_table.csv / codex_replay_curves.csv','図の数表'],
        ['codex_audit.json / codex_manifest.json','既存入力・予算の監査と出力SHA256'],
        ['codex_verification.json','恒等式とAdam群分割の検証'],
        ['codex_input_hashes.json','開始時の入力SHA256'],
        ['codex_literature.json','原典・主張・適用範囲'],
    ]))
    add(table(['script：analysis/vit_long_1024/posthoc_2blk/','役割'],[
        ['codex_r0_common.py','保存状態読取と共通の順伝播・尺度'],
        ['codex_r0_checkpoints.py','Q1・Q2・死の静的反実仮想'],
        ['codex_r0_transport.py','注意1が運ぶ入力の書き手'],
        ['codex_r0_verify.py','式と実装の照合'],
        ['codex_r0_replay.py','一課題だけの回し直しと介入'],
        ['codex_r0_probe.py','体を凍結したhead学習'],
        ['codex_r0_queue.py','終了を確認する逐次実行'],
        ['codex_r0_finish_queue.py','追加のCPU分解と長い追跡の逐次実行'],
        ['codex_r0_radial.py','CE尺度微分と平均の打ち消し'],
        ['codex_r0_ce_self.py','CEのPSD自己項・交差項'],
        ['codex_r0_frozen_match.py','初期状態をそろえた凍結体'],
        ['codex_r0_summary.py','予測の採点と比較集計'],
        ['codex_r0_figures.py','図4枚とCSV'],
        ['codex_r0_audit.py','入力SHA256・予算・再実行の照合'],
        ['codex_r0_report.py','段階ごとの本文生成'],
    ]))
    REPORT.write_text('\n'.join(out))
    print(REPORT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args())
